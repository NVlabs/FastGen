# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Elastic router for the Elastification method.

Ports Megatron-LM's FlextronRouter
(megatron/elastification/router/hybrid_flex_router.py) to FastGen, adapted for
Wan diffusion transformers. Compared to the reference implementation:

- Uses plain nn.Linear instead of TransformerEngine parallel linears (FastGen
  is DDP/FSDP-only; no tensor parallelism).
- Only three axes: mlp, emb, skip (Wan has no Mamba or MoE).
- Layer-skip axis is retained to match the reference exactly: the router
  produces a router_skip output that feeds the analytical budget loss, but
  nothing wires it into an actual forward-pass skip. It is a phantom axis
  whose gradient flows through the budget-loss param count only.
- No per-block heterogeneity ("flex_hetero_*" branches dropped).
- No pipeline-parallel weight sync.
- Iteration is passed explicitly to forward() rather than pulled from a global
  args object.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from fastgen.methods.elastification.config import FlextronConfig
from fastgen.utils.distributed import get_rank, world_size


class FlextronRouter(nn.Module):
    """Router that maps a scalar budget to per-axis Gumbel-softmax choices.

    Structure:
      - self.gate_emb    : Sequential producing logits over emb_int_list
      - self.gate_mlp    : Sequential producing logits over mlp_int_list
      - self.gate_skip_layer (optional) : Sequential producing skip logits

    Sampling is DP-seeded: each (dp_rank, fwd_pass_count, iteration) triple
    draws a distinct Gumbel noise pattern via a deterministic seed. Different
    DP ranks and different grad-accum micro-steps therefore sample *different*
    sub-architectures around the same target budget — over one training step
    the router sees `dp_size * grad_accum_rounds` distinct sub-arch samples,
    which increases gradient-signal diversity per update.
    """

    def __init__(
        self,
        config: FlextronConfig,
        gumbel_base_seed: int = 42,
    ):
        super().__init__()
        self.config = config

        self.input_dim = len(self.config.budget_list)
        self.n_dim = self.config.router_inter_dim
        self.budget_map = {
            item: torch.tensor(idx)
            for idx, item in enumerate(self.config.budget_list)
        }

        # Initialize DP-aware Gumbel softmax
        self._init_dp_gumbel_softmax(base_seed=gumbel_base_seed)

        # Create init method for router layers
        # `init_method_normal` is inlined locally rather than importing from
        # `megatron.core.utils`.
        self.init_method = self._init_method_normal(self.config.router_std)

        self.add_router_for_mlp()
        self.add_router_for_emb()
        if self.config.add_skipping:
            self.add_router_for_skipping()

        self.hard_sample_th = self.config.hard_sample_th

        self.add_scaler_schedule()

        # FastGen has no PP/TP so `world_size()` is the DP world size.
        self.dp_size = world_size()
        self.fwd_pass_count = 0

    # ── Helpers ────────────────────────────────────────────────────────

    def _init_dp_gumbel_softmax(self, base_seed: int = 42) -> None:
        """Initialize DP-aware Gumbel-softmax state.

        FastGen has no PP/TP so `get_rank()` is the DP rank directly.
        """
        self.dp_rank = get_rank()
        self.gumbel_base_seed = base_seed

    def _init_method_normal(self, std: float):
        """Local inline replacement for `megatron.core.utils.init_method_normal`."""
        def init_fn(tensor: torch.Tensor) -> torch.Tensor:
            return nn.init.normal_(tensor, mean=0.0, std=std)
        return init_fn

    def _dp_gumbel_softmax(self, logits, tau=1.0, hard=False, curr_iteration=0):
        """DP-aware Gumbel softmax that uses different random seeds per DP rank and iteration"""
        # Create unique seed for this iteration and DP rank

        seed = (
            self.gumbel_base_seed
            + (self.dp_rank + self.fwd_pass_count * self.dp_size) % self.config.router_gbs
            + curr_iteration * 1000
        )
        # torch.manual_seed seeds both CPU and CUDA generators globally, so we
        # must save/restore both - otherwise the CUDA RNG leaks the deterministic
        # state we set here into other CUDA random ops elsewhere in the model.
        cpu_state = torch.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        torch.manual_seed(seed)

        try:
            return F.gumbel_softmax(logits, tau=tau, hard=hard)
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _create_linear_layer(
        self, input_size: int, output_size: int, bias: bool = False
    ) -> nn.Linear:
        """Standard router linear.

        Strips the TransformerEngine parallel-linear branch and the
        `is_first_layer` distinction from Megatron's version (both irrelevant
        without tensor parallelism).
        """
        layer = nn.Linear(input_size, output_size, bias=bias)
        self.init_method(layer.weight)
        return layer

    def add_router_for_mlp(self) -> None:
        mlp_list = self.config.mlp_int_list
        if self.config.flex_hetero_ffn:
            # Megatron reads block count from `hybrid_layer_pattern`; Wan uses
            # `num_mlp_blocks` (populated by the manager from the model).
            num_mlp = self.config.num_mlp_blocks
            gate_mlp_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False),
                nn.LeakyReLU(0.1),
                self._create_linear_layer(
                    self.n_dim, len(mlp_list) * num_mlp, bias=False
                ),
            ]
            # Defensive last-layer bias init (dead with plain nn.Linear + bias=False;
            # kept for parity with Megatron in case a caller flips bias on).
            if (
                hasattr(gate_mlp_layer_list[-1], 'bias')
                and gate_mlp_layer_list[-1].bias is not None
            ):
                last_layer_bias = [0.00 for _ in range(len(mlp_list))]
                last_layer_bias[-1] = 1.00
                gate_mlp_layer_list[-1].bias.data = torch.tensor(
                    last_layer_bias,
                    dtype=gate_mlp_layer_list[-1].weight.dtype,
                    device=gate_mlp_layer_list[-1].weight.device,
                ).repeat(num_mlp)
        else:
            gate_mlp_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False),
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(mlp_list), bias=False),
            ]
        self.gate_mlp = nn.Sequential(*gate_mlp_layer_list)

    def add_router_for_emb(self) -> None:
        emb_list = self.config.emb_int_list
        gate_emb_layer_list = [
            self._create_linear_layer(self.input_dim, self.n_dim, bias=False),
            nn.LeakyReLU(0.1),
            self._create_linear_layer(self.n_dim, len(emb_list), bias=False),
        ]
        self.gate_emb = nn.Sequential(*gate_emb_layer_list)

    def add_router_for_skipping(self) -> None:

        self.output_dim = int(len(self.config.layer_ranking_list) + 1)

        gate_skip_mlp_layer_list = [
            self._create_linear_layer(self.input_dim, self.n_dim, bias=False),
            nn.LeakyReLU(0.1),
            self._create_linear_layer(self.n_dim, self.output_dim, bias=False),
        ]

        self.gate_skip_layer = nn.Sequential(*gate_skip_mlp_layer_list)

    def mlp_forward(self, curr_iteration, budget_tensor, device, dtype, tau, hard_sample):

        # Megatron's `[0]` after each Linear call unpacks TEColumnParallelLinear's
        # (output, bias) tuple return. nn.Linear returns a tensor directly, so
        # the unpacks are dropped here.
        router_mlp_logits1 = self.gate_mlp[0](budget_tensor)
        router_mlp_logits2 = self.gate_mlp[1](router_mlp_logits1)
        router_mlp_logits = self.gate_mlp[2](router_mlp_logits2).flatten()
        # Megatron leaves `scale` unbound when `self.scaler is None`, then
        # references it unconditionally below → NameError for the common
        # scaler-off configuration. Default to 1.0 so the `scale * ...`
        # multiplications become no-ops in that case.
        scale = 1.0
        if self.scaler is not None:
            scale = self.scaler[curr_iteration].to(device=device, dtype=dtype)
        if self.config.flex_hetero_ffn:
            mlp_n = len(self.config.mlp_int_list)
            router_mlp_logits = router_mlp_logits.reshape(-1, mlp_n)
            if self.config.normalize_router_logits:
                router_mlp_logits = (
                    scale
                    * router_mlp_logits
                    / router_mlp_logits.std(dim=1, keepdim=True).clamp(min=1e-6)
                )
            else:
                router_mlp_logits = scale * router_mlp_logits
            router_mlp_logits = self._dp_gumbel_softmax(
                router_mlp_logits, tau=tau, hard=hard_sample, curr_iteration=curr_iteration
            )
            _, choices_mlp = torch.topk(router_mlp_logits, 1, dim=-1)
            return (
                router_mlp_logits,
                [self.config.mlp_int_list[i] for i in choices_mlp.flatten().tolist()],
            )
        else:
            if self.config.normalize_router_logits:
                # Std-normalize only when there's actually >1 choice; with a
                # single choice the std is 0 and the routing is trivial, so we
                # skip both the scale and the normalization (consistent with
                # the no-op semantics of a single-choice axis).
                if len(self.config.mlp_int_list) > 1:
                    router_mlp_logits = (
                        scale
                        * router_mlp_logits
                        / router_mlp_logits.std(dim=0, keepdim=True).clamp(min=1e-6)
                    )
            else:
                router_mlp_logits = scale * router_mlp_logits
            router_mlp_logits = self._dp_gumbel_softmax(
                router_mlp_logits, tau=tau, hard=hard_sample, curr_iteration=curr_iteration
            )
            _, choices_mlp = torch.topk(router_mlp_logits, 1, dim=-1)
            return (router_mlp_logits, self.config.mlp_int_list[choices_mlp.item()])

    def emb_forward(self, curr_iteration, budget_tensor, device, dtype, tau, hard_sample):

        # `[0]` unpacks removed vs Megatron — see comment in mlp_forward.
        router_emb_logits1 = self.gate_emb[0](budget_tensor)
        router_emb_logits2 = self.gate_emb[1](router_emb_logits1)
        router_emb_logits = self.gate_emb[2](router_emb_logits2).flatten()
        if self.scaler is not None:
            scale = self.scaler[curr_iteration].to(device=device, dtype=dtype)
            router_emb_logits = scale * router_emb_logits

        router_emb_logits = self._dp_gumbel_softmax(
            router_emb_logits, tau=tau, hard=hard_sample, curr_iteration=curr_iteration
        )
        _, choices_emb = torch.topk(router_emb_logits, 1, dim=-1)

        return (router_emb_logits, self.config.emb_int_list[choices_emb.item()])

    def skipping_forward(self, curr_iteration, budget_tensor, device, dtype, tau, hard_sample):

        # for layer skipping, skipping MLP layers
        # `[0]` unpacks removed vs Megatron — see comment in mlp_forward.
        router_skip_layer_logits1 = self.gate_skip_layer[0](budget_tensor)
        router_skip_layer_logits2 = self.gate_skip_layer[1](router_skip_layer_logits1)
        router_skip_layer_logits = self.gate_skip_layer[2](router_skip_layer_logits2).flatten()
        router_skip_layer_logits = torch.repeat_interleave(
            router_skip_layer_logits, repeats=1, dim=0
        )
        if self.scaler is not None:
            router_skip_layer_logits = router_skip_layer_logits * self.scaler[
                curr_iteration
            ].to(device=device, dtype=dtype)

        router_skip_layer_logits = self._dp_gumbel_softmax(
            router_skip_layer_logits, tau=tau, hard=hard_sample, curr_iteration=curr_iteration
        )
        _, choices_skip_layer = torch.topk(router_skip_layer_logits, 1, dim=-1)
        if choices_skip_layer.item() != 0:
            selected_to_drop = self.config.layer_ranking_list[: choices_skip_layer.item()]
            choices_skip_layer = torch.zeros(self.config.num_layers).to(device=device, dtype=dtype)
            choices_skip_layer[selected_to_drop] = 1
        else:
            choices_skip_layer = torch.zeros(self.config.num_layers).to(device=device, dtype=dtype)
        return (router_skip_layer_logits, choices_skip_layer)

    def get_curr_tau(self, curr_iteration: int) -> torch.Tensor:
        """Gumbel-softmax temperature: `tau_init * tau_decay ** curr_iteration`."""
        return self.config.tau_init * torch.pow(
            torch.tensor(self.config.tau_decay), curr_iteration
        )

    def add_scaler_schedule(self) -> None:
        """Precompute an optional linear logit-scaler over training iterations.

        Off unless both scaler endpoints and `train_iters` are set. When on,
        each forward multiplies its axis logits by `scaler[iteration]` before
        Gumbel-softmax — sharpens routing over training in a more controlled
        way than tau decay alone.
        """
        cfg = self.config
        if (
            cfg.linear_scaler_start is None
            or cfg.linear_scaler_end is None
            or cfg.train_iters is None
        ):
            self.scaler = None
            return
        self.scaler = torch.linspace(
            start=cfg.linear_scaler_start,
            end=cfg.linear_scaler_end,
            steps=cfg.train_iters,
        )

    def forward(self, budget, curr_iteration):

        hard_sample = random.random() > self.hard_sample_th

        tau = self.get_curr_tau(curr_iteration)

        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype

        if budget in self.budget_map.keys():
            budget_tensor = F.one_hot(
                self.budget_map[budget], len(self.config.budget_list)
            ).to(device=device, dtype=dtype)
        elif budget == 1.0:
            # Requested full model but 1.0 isn't a trained budget — fall back
            # to the largest configured budget. Using max() instead of [0]
            # makes this independent of budget_list ordering.
            budget_tensor = F.one_hot(
                self.budget_map[max(self.budget_map.keys())],
                len(self.config.budget_list),
            ).to(device=device, dtype=dtype)
        else:
            # budget_list is enforced descending by FlextronConfig.__attrs_post_init__.
            # We re-sort ascending locally for bucketize, then flip(0) below to
            # land back in the descending one-hot coordinate system the router
            # was trained against.
            budget_values = torch.tensor(
                sorted(self.config.budget_list), device=device, dtype=dtype
            )
            budget_t = torch.as_tensor(budget, device=device, dtype=dtype)

            # idx2 = first index where budget_values[idx] > budget (right=False gives >= behavior with floats)
            idx2 = torch.bucketize(budget_t, budget_values, right=False)
            # Clamp to valid interior so we always have a left neighbor
            idx2 = idx2.clamp(min=1, max=len(self.config.budget_list) - 1)
            idx1 = idx2 - 1

            b1 = budget_values.index_select(0, idx1.to(torch.long))
            b2 = budget_values.index_select(0, idx2.to(torch.long))
            denom = b2 - b1  # .clamp_min(1e-12)
            weight = (budget_t - b1) / denom  # in [0,1] when budget is between b1 and b2

            num_classes = len(self.config.budget_list)
            one_hot_1 = F.one_hot(
                idx1.to(torch.long), num_classes=num_classes
            ).to(device=device, dtype=dtype)
            one_hot_2 = F.one_hot(
                idx2.to(torch.long), num_classes=num_classes
            ).to(device=device, dtype=dtype)

            # If weight is scalar, broadcasting works; if vector, it blends per-sample
            budget_tensor = (1 - weight).unsqueeze(-1) * one_hot_1 + weight.unsqueeze(
                -1
            ) * one_hot_2
            budget_tensor = budget_tensor.squeeze(0).flip(0)

        budget_tensor = budget_tensor.unsqueeze(0)
        mlp_forward_outputs = self.mlp_forward(
            curr_iteration, budget_tensor, device, dtype, tau, hard_sample
        )

        if self.config.add_skipping:
            skipping_forward_outputs = self.skipping_forward(
                curr_iteration, budget_tensor, device, dtype, tau, hard_sample
            )
        else:
            skipping_forward_outputs = None

        emb_forward_outputs = self.emb_forward(
            curr_iteration, budget_tensor, device, dtype, tau, hard_sample
        )
        self.fwd_pass_count += 1
        # 3-tuple return (Megatron returns 5-tuple including mamba + moe_expert;
        # we drop those two positions rather than pad with None).
        return (
            mlp_forward_outputs,
            skipping_forward_outputs,
            emb_forward_outputs,
        )

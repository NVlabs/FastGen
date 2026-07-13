# Elastification for FastGen — Design Plan

Bringing the **Elastification / Flextron** one-shot NAS + trained router compression method
into FastGen, targeting Wan diffusion transformers.

- Reference paper: <https://arxiv.org/pdf/2511.16664>
- Reference implementation: `Megatron-LM/megatron/elastification/`

## 1. What the algorithm does

Trains a single "elastic" backbone that contains many pruned sub-architectures, plus a small
router that picks one sub-architecture at inference time based on a target **budget** (e.g.
fraction of full parameters). Sub-architectures are realized at inference by masking
activations rather than by changing the module structure — hooks apply prefix masks driven
by the router's Gumbel-softmax choices.

Five conceptual pieces (as in Megatron):

1. **Search space** — per-axis lists of candidate integer widths (`emb_int_list`,
   `mlp_int_list`, optional `layer_ranking_list`) plus a `budget_list` of target fractions.
2. **Router** — small MLP mapping a one-hot budget vector to Gumbel-softmax choice logits per
   axis. Tau anneals over training. Interpolates between one-hots for intermediate budgets.
3. **Elasticity hooks** — one hook manager per module class; each attaches
   `register_forward_pre_hook` / `register_forward_hook` that build prefix masks
   `[True]*k + [False]*(N-k)`, multiply activations by the mask, multiply by the router
   probability (STE — this is the gradient bridge back to the router), and rescale
   LayerNorm eps by the effective width ratio.
4. **Budget loss** — analytical parameter count applied to soft router outputs. Loss is
   `|current / (target * full) - 1|` with a small dead-zone and an MSE anchor at
   `budget = 1.0` that keeps the identity path clean.
5. **Wiring** — per training step: sample a budget → router forward → push soft choices to
   hook managers → normal forward through the net (hooks fire during it) → combine LM/DSM
   loss with budget loss (and optional distillation loss).

## 2. Locked-in decisions for this port

| Area | Decision | Rationale |
|---|---|---|
| Target network | Wan | Product priority. Uses diffusers `WanTransformer3DModel`. |
| Router axes | `emb_int_list`, `mlp_int_list`, optional `layer_ranking_list` | Matches Megatron's actually-implemented axes. |
| Attention heads axis | **Dropped** | Megatron doesn't implement it either (`flex_budget_utils.py:41` says "head elasticity itself is no longer supported"). Not needed for parity, avoids the `timm.Attention` / diffusers `attn_processor` forward-override work. |
| Head_dim axis | **Dropped** | Never varied in Megatron. |
| Mamba / MoE hooks | **Dropped** | Wan has neither. |
| Layout | Single directory `fastgen/methods/elastification/` | Method won't compose with other step-distillation methods (per user), so no shared-lib split needed. |
| Phases | All three (joint / freeze-model / freeze-router) via config flags | Matches Megatron. Simple `param.requires_grad` toggles. |
| Distillation | Included; **form open** | Choose A / B / C — see §7. |
| Checkpoint loading | Reuse FastGen's `pretrained_model_path` path | `load_student_weights_and_ema()` handles it. |
| Head_dim divisibility constraint | **No hard enforcement** | Not needed for correctness. Optional soft warning if a user picks an `emb_int` that would produce a non-standard sub-net shape at materialization time. |
| Materialization (weight cropping) | **v2, separate PR** | Not needed for algorithm correctness. Ships real inference perf savings. |

## 3. Wan integration points

FastGen's `Wan` (`fastgen/networks/Wan/network.py`) wraps diffusers'
`WanTransformer3DModel` and monkey-patches several methods via
`override_transformer_forward` (line 831-846). Relevant to us:

- `block.forward = block_forward` — FastGen's block replacement, handles Wan2.1 vs Wan2.2
  TI2V modulation + optional sCM `norm_temb`. This is what our hooks attach around.
- `transformer.forward = classify_forward` — FastGen's transformer-level forward.
  Supports `skip_layers` (guidance-time layer skipping) — **note: this is NOT our
  router-driven layer-skip**. Don't route through it; use our own block-level hook.
- `transformer.classify_forward_block_forward` — the per-block iteration loop.

**Hook attach order.** Our `apply_elasticity_to_wan(net, cfg)` runs during
`ElastificationModel.build_model()`, which happens after `Wan.__init__` (which runs
`override_transformer_forward`). So hooks bind to the FastGen-overridden `block_forward`.
This ordering is automatic; do not move elasticity setup earlier.

**Module targets inside each `WanTransformerBlock`:**

- `attn1` — self-attention (`WanAttention` with separate `to_q`, `to_k`, `to_v`,
  `to_out.0`, `norm_q`, `norm_k`).
- `attn2` — cross-attention over text embeddings. Same shape as `attn1` but K, V come from
  the text encoder side. Extra `add_k_proj`, `add_v_proj`, `norm_added_k` for I2V.
- `ffn` — diffusers `FeedForward` (need to check its internal structure at implementation
  time — likely has `.net.0.proj` and `.net.2`).
- `scale_shift_table` — AdaLN modulation tensor `[6, inner_dim]`; needs emb masking on the
  `inner_dim` axis so shift/scale/gate operate on the pruned channels only.

**Cross-attention specifics.** Router axes affect `attn2` as follows:

- `emb` — mask Q input (video-side); mask `to_k` and `to_v` **output** rows to the emb
  prefix (so K/V dims line up with Q's shrunk head layout); leave `encoder_hidden_states`
  (text side) untouched — that's the frozen text encoder's output dim, not ours to prune.
- `mlp_hidden` — no effect on cross-attn.
- `layer_skip` — dropping a block skips its cross-attn too.

## 4. File structure

### Implementation (new)

```
fastgen/methods/elastification/
├── __init__.py                       # exports ElastificationModel
├── PLAN.md                           # this file
├── README.md                         # end-user doc: search space, phases, inference
├── config.py                         # ElastConfig (attrs dataclass) + validation
├── router.py                         # ElasticRouter (Gumbel-softmax, DP-seeded, tau decay)
├── budget_utils.py                   # analytical Wan param count on soft router outputs
├── hooks/
│   ├── __init__.py                   # apply_elasticity_to_wan(net, cfg)
│   ├── self_attention.py             # WanAttention as attn1
│   ├── cross_attention.py            # WanAttention as attn2 (video-side)
│   ├── ffn.py                        # emb + mlp-hidden mask
│   ├── block.py                      # WanTransformerBlock: layer-skip + scale_shift_table
│   └── stack.py                      # WanTransformer3DModel: norm_out + proj_out emb mask
├── manager.py                        # ElastificationManager
├── loss.py                           # combine_losses, self_distill helper
├── model.py                          # ElastificationModel(FastGenModel)
└── materialize.py                    # v2 (later PR): crop weights to fixed budget
```

### Config (new)

```
fastgen/configs/methods/
└── config_elastification.py          # ModelConfig(BaseModelConfig) + Config(BaseConfig)
```

### Registration (edit existing)

```
fastgen/methods/__init__.py           # + from fastgen.methods.elastification.model import ElastificationModel
```

### Inference script (new)

```
scripts/inference/
└── run_elastification_inference.py   # standalone: load ckpt, set override_selected_budget, sample
```

### Tests (new)

```
tests/elastification/
├── __init__.py
├── test_router.py                    # ~200 L
├── test_budget_utils.py              # ~150 L
├── test_apply_elasticity_to_wan.py   # ~200 L
├── test_self_attention_hook.py       # ~150 L
├── test_cross_attention_hook.py      # ~150 L
├── test_ffn_hook.py                  # ~150 L
├── test_block_hook.py                # ~150 L
├── test_manager.py                   # ~150 L
├── test_config.py                    # ~100 L
├── test_model.py                     # ~200 L
├── test_loss.py                      # ~100 L
└── test_numerical_invariants.py      # ~300 L (checks Megatron doesn't have)
```

### Rough sizing

- Implementation: 13 files, ~2500 lines
- Config: 1 file, ~50 lines
- Inference script: 1 file, ~100 lines
- Tests: 13 files, ~2100 lines
- **Total: ~28 new files, ~4750 lines**

Comparable to Megatron's `elastification/` (~4800 impl + ~2300 tests), scoped down for
the smaller search space (no Mamba, no MoE, no per-block heterogeneity, no TP router).

## 5. Per-file design notes

### `config.py`

```python
@attrs.define(slots=False)
class ElastConfig:
    # ── Search space ──────────────────────────────────
    emb_int_list: List[int] = []                  # e.g. [5120, 3840, 2560, 1280]
    mlp_int_list: List[int] = []                  # e.g. [13824, 10368, 6912, 3456]
    layer_ranking_list: Optional[List[int]] = None  # ordered block indices eligible to skip

    # ── Budget ────────────────────────────────────────
    budget_list: List[float] = [1.0]
    budget_probs: Optional[List[float]] = None
    loss_alpha: float = 1.0
    router_beta: float = 1.0

    # ── Router ────────────────────────────────────────
    enable_router: bool = True
    router_inter_dim: int = 128
    tau_init: float = 1.0
    tau_decay: float = 0.9999
    hard_sample_th: float = 0.996
    router_std: float = 0.1
    lr_mult_router: float = 1.0
    normalize_router_logits: bool = False
    linear_scaler_start: Optional[float] = None
    linear_scaler_end: Optional[float] = None

    # ── Training schedule ─────────────────────────────
    original_model_sample_prob: float = 0.33      # identity anchor

    # ── Phase flags ───────────────────────────────────
    freeze_model: bool = False
    freeze_router: bool = False

    # ── Distillation ──────────────────────────────────
    distillation_mode: str = "none"               # "none" | "external_teacher" | "self_double_forward"
    distill_coeff: float = 1.0

    # ── Eval / inference override ─────────────────────
    override_selected_budget: Optional[float] = None
    is_eval: bool = False

    def __attrs_post_init__(self):
        # Sort budget_list descending (Megatron invariant)
        if self.budget_list:
            order = sorted(range(len(self.budget_list)), key=lambda i: -self.budget_list[i])
            self.budget_list = [self.budget_list[i] for i in order]
            if self.budget_probs:
                self.budget_probs = [self.budget_probs[i] for i in order]
```

### `router.py`

`ElasticRouter(nn.Module)`. Same shape as Megatron's `FlextronRouter` stripped of tensor-parallel
linear layers (FastGen uses DDP/FSDP, not TP). Just two `nn.Linear` layers per axis. Preserves:

- Gumbel-softmax with `tau = tau_init * tau_decay^iteration`
- DP-seeded Gumbel noise so all data-parallel ranks pick the same choice per step
- Budget one-hot lookup + linear interpolation for intermediate budgets
- `forward(budget) → dict[str, (soft_probs, chosen_int)]` per axis
- 3 axes: `mlp`, `emb`, `layer_skip`

### `budget_utils.py`

Wan-specific analytical param count. Modeled on Megatron's
`flex_budget_utils.py::get_num_parameters`. Accepts either concrete ints or router soft
probs (differentiable). Formula per block:

```
attn1_params = 4 * hidden_size * hidden_size                            # to_q + to_k + to_v + to_out.0
attn2_params = 2 * hidden_size * hidden_size                            # to_q + to_out.0
              + 2 * text_encoder_dim * hidden_size                       # to_k + to_v (text side untouched)
ffn_params   = 2 * hidden_size * mlp_hidden                              # up_proj + down_proj
misc_params  = 6 * hidden_size + 3 * hidden_size                         # scale_shift_table + norms
block_params = attn1_params + attn2_params + ffn_params + misc_params

total_blocks = Σ (1 - skip_prob_i) * block_params
```

Params scale linearly with `hidden_size` and `mlp_hidden`. `num_heads` and `head_dim` don't
appear because they never vary in the search space. Text-encoder dim is a constant.

### `hooks/` — five hook managers

Each follows the Megatron pattern: constructor stores config, `attach_hooks(module)`
registers pre/post hooks, `set_elasticity_params(...)` receives router choices per step,
`detach_hooks()` restores.

- **`self_attention.py`** (attn1) — emb-mask on input, eps rescale on `norm_q`/`norm_k`,
  emb mask + STE prob scaling on `to_out.0` output.
- **`cross_attention.py`** (attn2) — like self-attention but Q input only (video side);
  `to_k`/`to_v` output rows masked; text-side untouched.
- **`ffn.py`** — emb mask input, mlp-hidden mask on intermediate (post-hook on first linear),
  emb mask on final output.
- **`block.py`** — layer-skip: pre-hook stashes input, post-hook returns stashed input if
  the router selected this block for skipping. Also masks `scale_shift_table` on `inner_dim`.
- **`stack.py`** — top-level `norm_out` eps rescale + `sqrt(k/N)` restore. `proj_out`
  input is already masked upstream.

`hooks/__init__.py::apply_elasticity_to_wan(net, cfg)` walks
`net.transformer.blocks` and attaches every manager. Modeled directly on Megatron's
`apply_flextron_elasticity_to_model`. Returns list of managers for the orchestrator.

### `manager.py` — `ElastificationManager`

Orchestrator. Modeled on Megatron's `FlextronModelManager`.

- `__init__(net, cfg)`:
  - Store net + cfg
  - Build `ElasticRouter`
  - Call `apply_elasticity_to_wan(net, cfg)` → store hook managers
  - Pre-compute full-model param count for budget-loss normalization
  - Optional: warn if any `emb_int_list` entry is not a multiple of `head_dim` (soft warning; not enforced)
- `process_budget(budget) → soft_choices` — calls router forward
- `push_to_hooks(soft_choices)` — forwards to every hook manager's `set_elasticity_params`
- `compute_budget_loss(soft_choices, budget) → tensor` — delegates to `budget_utils`

### `loss.py`

- `budget_loss_func(current_soft_params, target_budget, full_params, budget_val)` —
  `|current / (target * full) - 1|`, clipped inside 5% dead-zone, MSE anchor at `budget = 1.0`.
- `distillation_loss_func(student_pred, teacher_pred)` — MSE.
- `combine_losses(dsm_loss, budget_loss, distill_loss, alpha, distill_coeff) → total_loss, loss_map`.
- Helper for KD mode B: `self_distill_forward(net, manager, inputs, sampled_budget)` — runs
  net twice with different router settings.

### `model.py` — `ElastificationModel(FastGenModel)`

A FastGen method (subclass of FastGenModel) with a Megatron-style hook orchestrator inside.

- `build_model()`:
  1. `super().build_model()` builds `self.net` (Wan)
  2. `self.load_student_weights_and_ema()` loads pretrained ckpt
  3. `self.manager = ElastificationManager(self.net, self.config.elast)` — sets up router + hooks
  4. If `distillation_mode == "external_teacher"`: `self.build_teacher()` (FastGen infra)
  5. Apply freeze flags: `self.net.requires_grad_(False)` if `freeze_model`;
     `self.manager.router.requires_grad_(False)` if `freeze_router`
- `single_train_step(data, iteration)`:
  1. Prepare data (SFT-style)
  2. Sample budget via DP-seeded choice (mirror Megatron's `get_grad_acc_based_random_choice`)
  3. `soft_choices = self.manager.process_budget(budget)`
  4. `budget_loss = self.manager.compute_budget_loss(soft_choices, budget)`
  5. `self.manager.push_to_hooks(soft_choices)`
  6. `net_pred = self.net(noisy, t, condition=cond_train)` — hooks fire inside
  7. `dsm_loss = denoising_score_matching_loss(...)`
  8. Distillation branch (if enabled)
  9. Combine: `total_loss = dsm_loss + loss_alpha * budget_loss + distill_coeff * distill_loss`
  10. Return `loss_map = {total_loss, dsm_loss, budget_loss, distill_loss, sampled_budget}`, `outputs`
- `init_optimizers()`: parent builds `net_optimizer`; add `router_optimizer` with
  `LR × lr_mult_router`.
- `get_optimizers(iteration) → [net_optimizer, router_optimizer]` (respects freeze flags).
- `optimizer_dict`, `scheduler_dict`, `model_dict`: include router entries for checkpointing.

### `materialize.py` (v2, later PR)

Post-training utility. Given trained elastic model + fixed budget, physically crop weights:

- Freeze router at hard choice for the budget
- Slice `attn1.to_q/to_k/to_v/to_out.0`, `attn2.to_q/to_out.0/norm_q` weights on emb axis;
  slice `attn2.to_k/to_v` on output rows; slice `ffn.net.0.proj/net.2` on mlp-hidden axis;
  slice `scale_shift_table` on `inner_dim`
- Delete blocks whose skip prob > 0.5
- Return a new `Wan` instance with cropped weights

Delivers real inference perf savings (activation masking alone doesn't). Separate PR.

## 6. Inference

### At runtime

Wan inference goes through `self.net.sample(noise, condition=..., ...)`, which internally
calls `self.net.forward(noisy, t, condition)` per denoising step. Since hooks are attached
to modules inside `self.net.transformer`, they fire on every forward automatically.

Recipe:

1. Set `config.model.elast.override_selected_budget = <b>`
2. Set `config.model.elast.is_eval = True`
3. Load elastic checkpoint (network + router)
4. Call standard FastGen inference script
5. Hooks apply the sub-net masking on every denoising step

### Standalone entry point

`scripts/inference/run_elastification_inference.py`:

1. Parse args + ElastConfig
2. Instantiate `ElastificationModel` from config
3. Load ckpt
4. Pin `override_selected_budget` and `is_eval`
5. Call `model.net.sample(noise, condition=..., ...)`

Megatron ships no equivalent script (the `--is-flex-eval` flag exists but no in-tree
caller uses it) — our port ships one from the start.

### Caveat: no wall-clock savings from masking alone

Activation masking does not reduce compute. Attention still runs at full head count,
FFN still projects to full `mlp_hidden`. For real perf savings, use v2 materialization
after training.

## 7. Open decisions

### KD form — needs a choice

The paper trains a self-distillation between the full model and the sub-model. Three ways
to realize this in FastGen:

| Option | Teacher | Cost per step | Notes |
|---|---|---|---|
| **A** External frozen teacher | Separate Wan checkpoint, `build_teacher()`, `requires_grad=False` | +1 teacher forward | Mirrors Megatron. Uses FastGen's existing `build_teacher()` infra. Doubles teacher-weights memory. |
| **B** Self-distillation via double forward | None — run live model twice: once at `budget=1.0` (`torch.no_grad`), once at sampled budget | +1 forward through same model | No extra weights. Teacher is a moving target — relies on the identity anchor keeping `budget=1.0` close to the original pretrained model. |
| **C** No KD | None | 0 | Rely solely on `original_model_sample_prob` identity anchor. Simplest. |

Recommendation: **A**, because it composes with FastGen's existing teacher infrastructure
and mirrors Megatron exactly. Confirm before starting `model.py`.

### Interaction with FastGen's `skip_layers` guidance flag

FastGen's `classify_forward_block_forward` already supports a `skip_layers: List[int]`
kwarg used for skip-layer guidance during teacher forward. Our router-driven layer-skip
uses a different mechanism (block-level pre/post hooks). They don't conflict, but:

- Our block hook must not be attached to a block that's already on the guidance
  `skip_layers` list, else it double-skips silently. Address by having the block hook
  no-op if the block is being called at all (which won't happen if it's on the skip list).
- Router-driven skip must never appear in the guidance skip_layers list — kept independent.

## 8. Test plan

Coverage mirrors Megatron's `tests/unit_tests/elastification/` (~2300 lines, plumbing-style
tests using stubs). We add numerical invariants Megatron doesn't check.

### Plumbing tests (Megatron parity)

- Router construction, DP-seeded Gumbel determinism, tau decay, budget interpolation.
- Budget-utils analytical param count, linearity, soft interpolation, gradient flow.
- Per-manager hook attach/detach lifecycle, right hooks on right sub-modules.
- Manager orchestration order (router → push → forward).
- Config validation.

### Numerical invariants (new)

1. **`test_budget_1_hard_choice_equals_original`** — the most valuable test. Take fresh
   pretrained tiny Wan, wrap with elastification, manually override router to hard one-hot
   on largest candidate in every axis, forward. Compare against unwrapped tiny Wan on same
   input. Must agree to ~1e-3 in bf16, ~1e-6 in fp32.
2. **`test_materialized_subnet_equals_elasticized`** (needs v2) — for a chosen budget:
   run elasticized model with router pinned to that budget's hard choices, run cropped-weight
   materialized model at those same choices. Should agree numerically.
3. **`test_layer_skip_is_identity_residual`** — force layer L to be skipped, verify the
   block's output equals its input.
4. **`test_hooks_detach_restores_original`** — attach hooks, detach, forward. Should equal
   a fresh forward with no hooks ever attached.
5. **`test_freeze_flags_actually_freeze`** — check `param.requires_grad` and post-step
   parameter deltas for all three phase combinations.
6. **`test_identity_anchor_holds`** — with `original_model_sample_prob=1.0`, train N steps;
   `max(router_prob)` on the largest candidate stays > 0.9.
7. **`test_checkpoint_roundtrip`** — save + reload elastic model; same output as before save.

### Debug playbook

If invariant 1 fails: (a) toggle each hook manager individually to isolate; (b) within a
manager, disable each hook individually; (c) at `budget=1.0` all `sqrt(k/N)` and `eps*(k/N)`
corrections should be identity — if they're not, that's the bug; (d) check
`router.gate_*.weight.grad` is non-None after backward.

## 9. Implementation order (~15 working days)

1. `router.py` — self-contained, unit-testable in isolation. (~2 days)
2. `budget_utils.py` — pure functions. Manual sanity check against
   `sum(p.numel() for p in wan.parameters())`. (~1 day)
3. `hooks/self_attention.py` — one hook manager fully working on a stub `WanAttention`.
   Test `budget_1_hard_choice_equals_original` for attention only. (~2 days)
4. Extend to `cross_attention.py`, `ffn.py`, `block.py`, `stack.py`. One at a time, each
   with its own numerical test. (~4 days)
5. `manager.py` + `hooks/__init__.py` — glue. Run
   `budget_1_hard_choice_equals_original` end-to-end on tiny Wan. (~1 day)
6. `loss.py`, `model.py`, `config.py`, `config_elastification.py` — full FastGen
   integration. Register in `fastgen/methods/__init__.py`. (~2 days)
7. `scripts/inference/run_elastification_inference.py` — verify sub-net inference actually
   runs on a tiny Wan checkpoint. (~1 day)
8. Full test suite pass + fix bugs uncovered. (~2 days)
9. **v2 PR (later):** `materialize.py` + `test_materialized_subnet_equals_elasticized`.
   (~2 days)

## 10. Non-goals for v1

- Mamba / MoE elasticity
- Attention-heads / head_dim elasticity
- Per-block heterogeneous choices (Megatron's `flex_hetero_*` axes)
- Memory-budget mode + memory-quantization profiles
- Wan / SDXL integration (SDXL later, after Wan pattern is proven)
- Tensor-parallel router weight sharding (FastGen uses DDP/FSDP, not TP)
- Weight materialization (v2, separate PR)

## Appendix A. Wan overrides we compose with

FastGen's `Wan.override_transformer_forward` (`fastgen/networks/Wan/network.py:831-846`)
does five monkey-patches at construction time:

1. Per-block `block.forward = block_forward` (handles Wan2.1 + Wan2.2 TI2V modulation + sCM `norm_temb`)
2. `transformer.classify_forward_prepare` (input prep helper)
3. `transformer.classify_forward_block_forward` (per-block iteration with `skip_layers`, `feature_indices`, etc.)
4. `transformer.forward = classify_forward` (adds `return_features_early`, `return_logvar`, etc.)
5. Optional `timesteps_proj.forward` (official WAN sinusoidal)

These exist because Wan uses diffusers as a third-party dep and FastGen composes rather
than forks. Our elasticity hooks bind to the FastGen-overridden `block.forward` because
`build_model()` runs after `Wan.__init__`. All compose cleanly at sub-module hook points
(`attn1`, `attn2`, `ffn`) — orthogonal to the FastGen overrides.

## Appendix B. What differs from the Megatron reference

| Concern | Megatron | Ours |
|---|---|---|
| Method registration | Bare imperative `pretrain_hybrid_flex.py` script | `ElastificationModel(FastGenModel)` subclass |
| How elasticity triggers per step | Monkey-patched `model.forward` runs router + hooks + original forward | `single_train_step` explicitly runs router + push-to-hooks + `self.net(...)` |
| Router LR override | `ParamGroupOverride` via patched `get_megatron_optimizer_config` | First-class second optimizer via `get_optimizers` |
| Distillation | ModelOpt `mtd.convert(model, ..., ("kd_loss", ...))` wraps in `DistillationModel` | FastGen's `build_teacher()` (option A) or double forward on live model (option B) |
| Tensor parallelism | TE column/row parallel linear in router | Plain `nn.Linear` — FastGen uses DDP/FSDP only |
| Search space axes | emb + mlp + Mamba heads + MoE experts + layer-skip | emb + mlp + layer-skip (no Mamba, no MoE) |
| Attention-heads axis | Not implemented (comment: "no longer supported") | Not implemented (match Megatron) |
| Memory-budget mode | Full memory profile support (`bpe_*`, YAML presets) | Param-only budget |
| Inference entry point | None shipped (flags exist, no caller) | `scripts/inference/run_elastification_inference.py` shipped from day one |
| Weight materialization | Not implemented | v2 (`materialize.py`) — separate PR |

# Elastification for FastGen — Design Plan

Bringing the **Elastification / Flextron** one-shot NAS + trained router compression method
into FastGen, targeting Wan diffusion transformers.

- Reference paper: <https://arxiv.org/pdf/2511.16664>
- Reference implementation: `Megatron-LM/megatron/elastification/`

## Design principle: mirror the Megatron reference

Match Megatron-LM's `elastification/` module 1:1 wherever practical — attribute
names, method names, `__init__` ordering, method-body structure, config field
names. The intent is frictionless side-by-side comparison for debugging and
code review.

Only deviate when Megatron requires machinery FastGen doesn't have (pipeline /
tensor parallelism, TransformerEngine parallel linears, `MegatronModule`,
`get_args()` globals, `hybrid_layer_pattern`, ModelOpt distillation), or when a
scope reduction from this plan applies (no Mamba/MoE/memory-mode/heterogeneous
Mamba). Every deviation is called out in an inline comment; parallels are not.

### Known Megatron quirks

Reference behaviors that look like bugs. Fixed here when they'd break
default configurations; noted for later verification against Megatron's
actual training scripts.

- **`FlextronRouter.mlp_forward` unbound `scale` — FIXED here, needs
  Megatron-side verification.** When `linear_scaler_start` /
  `linear_scaler_end` / `train_iters` are `None` (scaler disabled),
  Megatron's `mlp_forward` still references a local `scale` variable that
  was never assigned → `NameError` on the first forward. The only safe
  config that avoids all `scale` references is
  `normalize_router_logits=True` **and** `len(mlp_int_list) <= 1` (no MLP
  elasticity). Every other config either requires the scaler to be
  enabled or hits the NameError. Our port defaults `scale = 1.0` when
  `self.scaler is None` so the `scale * ...` multiplications become
  no-ops. **TODO:** double-check whether Megatron's actual training runs
  always enable the scaler (via `--linear-scaler-start` / `--linear-scaler-end`
  CLI args) — if so, the "bug" is dormant in their production configs and
  our default-1.0 fix is a strict superset of the reference behavior.

## 1. Locked-in decisions for this port

| Area | Decision | Rationale |
|---|---|---|
| Target network | Wan | Product priority. Uses diffusers `WanTransformer3DModel`. |
| Router axes | `emb_int_list`, `mlp_int_list`, optional `layer_ranking_list` | Matches Megatron's actually-implemented axes. |
| Attention heads axis | **Dropped** | Megatron doesn't implement it either (`flex_budget_utils.py:41` says "head elasticity itself is no longer supported"). Not needed for parity, avoids the `timm.Attention` / diffusers `attn_processor` forward-override work. |
| Head_dim axis | **Dropped** | Never varied in Megatron. |
| Mamba / MoE hooks | **Dropped** | Wan has neither. |
| Layout | Single directory `fastgen/methods/elastification/` | Method won't compose with other step-distillation methods (per user), so no shared-lib split needed. |
| Phases | All three (joint / freeze-model / freeze-router) via config flags | Matches Megatron. Simple `param.requires_grad` toggles. |
| Distillation | Included; **external frozen teacher** | Uses FastGen's `build_teacher()` infra; mirrors Megatron. Separate Wan checkpoint loaded frozen; +1 teacher forward per step. |
| Checkpoint loading | Reuse FastGen's `pretrained_model_path` path | `load_student_weights_and_ema()` handles it. |
| Head_dim divisibility constraint | **No hard enforcement** | Not needed for correctness. Optional soft warning if a user picks an `emb_int` that would produce a non-standard sub-net shape at materialization time. |
| Materialization (weight cropping) | **v2, separate PR** | Not needed for algorithm correctness. Ships real inference perf savings. |

## 2. Wan integration points

FastGen's `Wan` (`fastgen/networks/Wan/network.py`) wraps diffusers'
`WanTransformer3DModel` and monkey-patches several methods via
`override_transformer_forward` (line 831-846). Relevant to us:

- `block.forward = block_forward` — FastGen's block replacement, handles Wan2.1 T2V
  modulation + optional sCM `norm_temb`. This is what our hooks attach around.
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
  the text encoder side.
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

## 3. File structure

### Implementation (new)

```
fastgen/methods/elastification/
├── __init__.py                       # exports ElastificationModel
├── PLAN.md                           # this file
├── README.md                         # end-user doc: search space, phases, inference
├── config.py                         # FlextronConfig (attrs dataclass) + validation
├── router.py                         # FlextronRouter (Gumbel-softmax, DP-seeded, tau decay)
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

## 4. Per-file design notes

### `config.py`

```python
@attrs.define(slots=False)
class FlextronConfig:
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
    distillation: bool = False                    # enable external-teacher distillation
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

`FlextronRouter(nn.Module)`. Same shape as Megatron's `FlextronRouter` stripped of tensor-parallel
linear layers (FastGen uses DDP/FSDP, not TP). Just two `nn.Linear` layers per axis. Preserves:

- Gumbel-softmax with `tau = tau_init * tau_decay^iteration`
- DP-seeded Gumbel noise: seed is `base_seed + (dp_rank + fwd_count * dp_size) % router_gbs + iteration * 1000`, so each (rank, micro-step, iteration) triple draws distinct noise. Different DP ranks and different grad-accum micro-steps sample *different* sub-architectures at the same target budget — one training step covers `dp_size × grad_accum_rounds` distinct samples of the discrete choice space, increasing gradient-signal diversity per update. Deterministic across reruns.
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
  - Build `FlextronRouter`
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

### `model.py` — `ElastificationModel(FastGenModel)`

A FastGen method (subclass of FastGenModel) with a Megatron-style hook orchestrator inside.

- `build_model()`:
  1. `super().build_model()` builds `self.net` (Wan)
  2. `self.load_student_weights_and_ema()` loads pretrained ckpt
  3. `self.manager = ElastificationManager(self.net, self.config.elast)` — sets up router + hooks
  4. If `distillation`: `self.build_teacher()` (FastGen infra)
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

## 5. Inference

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

1. Parse args + FlextronConfig
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


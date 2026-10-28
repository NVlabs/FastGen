# Trajectory Matching

Methods that directly match teacher denoising trajectories.

## CM (Consistency Model)

**File:** [`CM.py`](CM.py) | **References:** [Song et al. (2023)](https://arxiv.org/abs/2303.01469), [Geng et al. (2024)](https://arxiv.org/abs/2406.14548)

Enforces consistency: `f(x_t, t) = f(x_r, r) = x_0` for points on the same trajectory.

**Key Parameters:**
- `loss_config.use_cd`: Use consistency distillation (requires teacher; CFG for the teacher is controlled via `guidance_cfg.cfg.scale`)
- `sample_t_cfg`: Config of the distribution for sampling `t`
- Requires [`CTScheduleCallback`](../../callbacks/ct_schedule.py), which controls the curriculum for the distance between `t` and `r`

**Configs:** [`EDM/config_cm_cifar10.py`](../../configs/experiments/EDM/config_cm_cifar10.py), [`EDM/config_cm_cifar10_fast.py`](../../configs/experiments/EDM/config_cm_cifar10_fast.py), [`EDM/config_cm_in64.py`](../../configs/experiments/EDM/config_cm_in64.py), [`EDM2/config_cm_s.py`](../../configs/experiments/EDM2/config_cm_s.py), [`EDM2/config_cm_xl.py`](../../configs/experiments/EDM2/config_cm_xl.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_cm_cifar10.py`](../../configs/experiments/EDM/config_cm_cifar10.py) | CIFAR-10 | No | 1 | 2.92 | 3.60 |
| [`EDM2/config_cm_s.py`](../../configs/experiments/EDM2/config_cm_s.py) | ImageNet-64 | Yes | 1 | 4.05 | 4.05 |


---

## sCM (Simple, Stable, and Scalable Consistency Model)

**File:** [`sCM.py`](sCM.py) | **Reference:** [Lu & Song (2024)](https://arxiv.org/abs/2410.11081)

Enforce continuous-time consistency `d/dt f(x_t,t) = 0` and use JVP-based training with TrigFlow parameterization for improved stability.

**Key Parameters:**
- `loss_config.use_cd`: Use consistency distillation (requires teacher; CFG for the teacher is controlled via `guidance_cfg.cfg.scale`)
- `loss_config.use_jvp_finite_diff`: Use finite difference for JVP (e.g., for compatibility with Flash Attention and FSDP)
- `precision_amp_jvp`: Autocast dtype for the JVP (`None` disables autocast there). The JVP state and its network input are kept in at least this dtype; a wider state keeps its own. The conditioning keeps its dtype: inputs that only reach the backbone stay in `precision_model`, and clean conditioning latents stay in `precision` and are cast up to the state. Unlike FastGen's own dropout, native dropout layers, such as LoRA's, replay the forward's masks only if this dtype equals `precision_amp`
- `sample_t_cfg`: Config of the distribution for sampling `t`

**Configs:** [`EDM/config_sct_cifar10.py`](../../configs/experiments/EDM/config_sct_cifar10.py), [`EDM/config_scd_cifar10.py`](../../configs/experiments/EDM/config_scd_cifar10.py), [`EDM/config_scd_in64.py`](../../configs/experiments/EDM/config_scd_in64.py), [`EDM2/config_scm_xl.py`](../../configs/experiments/EDM2/config_scm_xl.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_sct_cifar10.py`](../../configs/experiments/EDM/config_sct_cifar10.py) | CIFAR-10 | No | 1 | 3.23 | 2.85 |
| [`EDM/config_scd_cifar10.py`](../../configs/experiments/EDM/config_scd_cifar10.py) | CIFAR-10 | No | 1 | 3.22 | 3.66 |


---

## TCM (Truncated Consistency Model)

**File:** [`TCM.py`](TCM.py) | **Reference:** [Lee et al. (2024)](https://arxiv.org/abs/2410.14895)

Two-stage training: frozen Stage-1 CM for `t < transition_t`, trainable student for `t >= transition_t`.

**Key Parameters:**
- `transition_t`: Stage transition threshold
- `boundary_prob`, `w_boundary`: Probability of boundary timestep sampling and boundary loss weight
- Requires a pretrained CM checkpoint (Stage-1 model) via
  `trainer.checkpointer.pretrained_ckpt_path`, with
  `trainer.checkpointer.pretrained_ckpt_key_map` selecting which Stage-1
  checkpoint keys initialize the teacher, student, and EMA models

**Configs:** [`EDM/config_tcm_cifar10.py`](../../configs/experiments/EDM/config_tcm_cifar10.py), [`EDM2/config_tcm_s.py`](../../configs/experiments/EDM2/config_tcm_s.py), [`EDM2/config_tcm_xl.py`](../../configs/experiments/EDM2/config_tcm_xl.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_tcm_cifar10.py`](../../configs/experiments/EDM/config_tcm_cifar10.py) | CIFAR-10 | No | 1 | 2.70 | 2.46 |
| [`EDM2/config_tcm_xl.py`](../../configs/experiments/EDM2/config_tcm_xl.py) | ImageNet-64 (EDM2 version) | Yes | 1 | 2.23 | 2.20 |


---

## MeanFlow

**File:** [`mean_flow.py`](mean_flow.py) | **References:** [Geng et al. (2025)](https://arxiv.org/abs/2505.13447), [Sabour et al. (2025)](https://arxiv.org/abs/2506.14603)

Learns average velocity between trajectory points: `x_r = x_t - (t-r) · u(x_t, t, r)`.

**Key Parameters:**
- `loss_config.use_cd`: Use consistency distillation (requires teacher; CFG for the teacher is controlled via `guidance_cfg.cfg.scale`)
- `loss_config.use_jvp_finite_diff`: Use finite difference for JVP (e.g., for compatibility with Flash Attention and FSDP)
- `precision_amp_jvp`: Autocast dtype for the JVP (`None` disables autocast there). The JVP state and its network input are kept in at least this dtype; a wider state keeps its own. The conditioning keeps its dtype: inputs that only reach the backbone stay in `precision_model`, and clean conditioning latents stay in `precision` and are cast up to the state. Unlike FastGen's own dropout, native dropout layers, such as LoRA's, replay the forward's masks only if this dtype equals `precision_amp`
- `sample_t_cfg`, `sample_r_overrides`: Config of the distribution for sampling `t`, and changes to it for sampling `r` (e.g. `{"train_p_mean": -4.0}`; `None` samples `r` like `t`)
- `sample_t_cfg.flow_matching_ratio`: Fraction of the batch with `r = t` (flow matching loss)
- `guidance_cfg`, `guidance_fuse_scale`: If `guidance_fuse_scale` is `None`, use the target-side [`GuidanceConfig`](../../utils/guidance.py) (CFG (`cfg.scale`) and 3-way mixture (`mix.scale`) where t-windows are expressed via piecewise scales, e.g. `cfg.scale = [ω, 1.0]` + `cfg.boundary = [t_end]`), otherwise use prediction-side guidance (conditional output learns the guided flow directly)


**Configs:** [`EDM/config_mf_cifar10.py`](../../configs/experiments/EDM/config_mf_cifar10.py), [`DiT/config_mf_b.py`](../../configs/experiments/DiT/config_mf_b.py), [`DiT/config_mf_xl.py`](../../configs/experiments/DiT/config_mf_xl.py), [`WanT2V/config_mf.py`](../../configs/experiments/WanT2V/config_mf.py), [`WanT2V/config_anyflow.py`](../../configs/experiments/WanT2V/config_anyflow.py) (Stage 1 of [AnyFlow](../distribution_matching/README.md#anyflow))


**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_mf_cifar10.py`](../../configs/experiments/EDM/config_mf_cifar10.py) | CIFAR-10 | No | 1 | 2.82 | 2.92 |
| [`DiT/config_mf_xl.py`](../../configs/experiments/DiT/config_mf_xl.py) | ImageNet-256 | Yes | 1 | 3.19 | 3.43 |

---

## AYF (Align Your Flow)

**File:** [`ayf.py`](ayf.py) | **Reference:** [Sabour et al. (2025)](https://arxiv.org/abs/2506.14603)

Distillation with AYF's Eulerian Map Distillation (EMD) objective, which scales the JVP term of the MeanFlow target
`v - (t - r) du/dt` and clips each sample's residual to unit RMS. Unlike MeanFlow, AYF always distills from a teacher
and defaults to `sample_t_cfg.flow_matching_ratio=0`. The student can also learn guidance over an interval of scales.

**Key Parameters:**
- `teacher_guidance_type`: `"embedded"` (default) passes the sampled guidance scale to the teacher's guidance
  embedding; `"cfg"` runs classifier-free guidance on the teacher with it as the scale, composed by `guidance_cfg`,
  and needs `neg_condition` in the batch
- `teacher_guidance_range`: Interval `(min, max)` from which each sample draws its guidance scale; it applies only to
  networks with a `guidance_scale`
- `guidance_distill_warmup_steps`: Initial iterations that use `r = t` and skip the JVP, which reduces the loss to
  guided-velocity distillation (for teachers that are not guidance-distilled)
- See also the key parameters of MeanFlow above

**Configs:** [`Flux/config_ayf.py`](../../configs/experiments/Flux/config_ayf.py), [`QwenImage/config_ayf.py`](../../configs/experiments/QwenImage/config_ayf.py)

---

## KD (Knowledge Distillation)

**File:** [`KD.py`](KD.py) | **Reference:** [Luhman & Luhman (2021)](https://arxiv.org/abs/2101.02388)

Learns from pre-computed teacher denoising trajectories.

**Data Requirements:**
- Single-step: `{"real": clean, "noise": noise, "condition": cond}`
- Multi-step: `{"real": clean, "path": [B, steps, C, H, W], "condition": cond}`, one state per base-grid interval

**Key Parameters:**
- `student_grid_cfg.steps`: Number of student steps, taken from evenly spaced base-grid indices or `grid_kwargs.indices`
- `student_grid_cfg`: Path data needs an anchored grid at the path's timesteps: native with `grid_kwargs.base_steps`
  (optionally `shift`) or explicit `grid_kwargs.timesteps` ending at 0. Training checks only the interval count.

**Configs:** [`WanT2V/config_kd.py`](../../configs/experiments/WanT2V/config_kd.py) and [`SDXL/config_kd.py`](../../configs/experiments/SDXL/config_kd.py) (single-step), [`WanT2V/config_kd_path.py`](../../configs/experiments/WanT2V/config_kd_path.py) and [`WanT2V/config_kd_path_14b.py`](../../configs/experiments/WanT2V/config_kd_path_14b.py) (4-step paths on an explicit grid)

---

## CausalKD

**File:** [`KD.py`](KD.py) | **References:** [Yin et al. (2024)](https://arxiv.org/abs/2412.07772), [Zhu et al. (2026)](https://arxiv.org/abs/2602.02214)

KD for causal video models with inhomogeneous timesteps and autoregressive generation.

**Data Requirements:**
- `{"real": [B,C,T,H,W], "path": [B,steps,C,T,H,W], "condition": cond}`

**Key Parameters:**
- `context_noise`: Time to which chunk-by-chunk student sampling re-noises each finished chunk before writing it to
  the KV cache (`None`, the default, caches it clean); the training loss is unaffected
- See also the key parameters of KD above; CausalKD always trains on path data, so it needs an anchored grid

**Configs:** [`config_kd_causal.py`](../../configs/methods/config_kd_causal.py) (method config with the explicit 4-step grid of the Wan path-KD recipes); set `model.net` to a causal network such as `CausalWan_1_3B_Config` from [`net.py`](../../configs/net.py)

---

## PDD (Parallel Decoding Distillation)

**File:** [`pdd.py`](pdd.py) | **Reference:** [Shaul et al. (2026)](https://arxiv.org/abs/2607.26004)

PDD trains a widened student projection with one native sigma-velocity head at
each interval start of an anchored base grid. One network evaluation can fuse
those heads to advance several base-grid intervals.

**Key Parameters:**

- `student_grid_cfg.grid_kwargs.base_steps`: Number of intervals and velocity heads in an anchored native grid, with or without a shift.
- `loss_config.type_rk`: Teacher target integrator (`euler`, `midpoint`, `heun`, or `kutta4`).
- `loss_config.block_size_min`: Sets the largest NFE PDD trains for, `ceil(pdd_steps / block_size_min)`. Every training block starts on that NFE's student grid, so sampling with more steps runs but starts some blocks from states training never draws.
- `loss_config.block_size_max`: Hard maximum trained block size; every inference or carried interval must fit it.
- `loss_config.num_supervised_heads`: Number of heads sampled per example and averaged in each PDD loss (default `1`).
- `student_grid_cfg.steps`: Default balanced inference partition.
- `backward_simulation`: Take student inputs from the student's own trajectories instead of noised real data (`None`). `"staggered"` and `"synced"` carry one trajectory across iterations (rank-offset vs. common phases; require [`CarryCallback`](../../callbacks/carry.py)); `"default"` starts fresh at a random step each iteration.

Tensor and multimodal dict inputs are supported; modality losses are summed.

**Configs:** [`DiT/config_pdd_repa_xl.py`](../../configs/experiments/DiT/config_pdd_repa_xl.py), [`QwenImage/config_pdd.py`](../../configs/experiments/QwenImage/config_pdd.py), [`WanT2V/config_pdd.py`](../../configs/experiments/WanT2V/config_pdd.py), [`WanT2V/config_pdd_14b.py`](../../configs/experiments/WanT2V/config_pdd_14b.py), [`LTX2/config_pdd_ltx23.py`](../../configs/experiments/LTX2/config_pdd_ltx23.py), [`LTX2/config_pdd_ltx23_audio.py`](../../configs/experiments/LTX2/config_pdd_ltx23_audio.py), [`MiniMaxH3/config_pdd.py`](../../configs/experiments/MiniMaxH3/config_pdd.py)

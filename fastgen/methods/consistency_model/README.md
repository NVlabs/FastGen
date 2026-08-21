# Consistency Models

Methods that learn mappings between points on the deterministic probability flow ODE, also known as [flow maps](https://arxiv.org/abs/2505.18825).

## CM (Consistency Model)

**File:** [`CM.py`](CM.py) | **Reference:** [Song et al., 2023](https://arxiv.org/abs/2303.01469), [Geng et al., 2024](https://arxiv.org/abs/2406.14548)

Enforces consistency: `f(x_t, t) = f(x_r, r) = x_0` for points on the same trajectory.

**Key Parameters:**
- `loss_config.use_cd`: Use consistency distillation (requires teacher; `guidance_scale` controls CFG for the teacher)
- `sample_t_cfg`: Config of the distribution for sampling `t`
- Requires [`CTScheduleCallback`](../../callbacks/ct_schedule.py), which controls the curriculum for the distance between `t` and `r`

**Configs:** [`EDM/config_cm_cifar10.py`](../../configs/experiments/EDM/config_cm_cifar10.py), [`EDM/config_cm_in64.py`](../../configs/experiments/EDM/config_cm_in64.py), [`EDM2/config_cm_s.py`](../../configs/experiments/EDM2/config_cm_s.py), [`EDM2/config_cm_xl.py`](../../configs/experiments/EDM2/config_cm_xl.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_cm_cifar10.py`](../../configs/experiments/EDM/config_cm_cifar10.py) | CIFAR-10 | No | 1 | 2.92 | 3.60 |
| [`EDM2/config_cm_s.py`](../../configs/experiments/EDM2/config_cm_s.py) | ImageNet-64 | Yes | 1 | 4.05 | 4.05 |


---

## sCM (Simple, Stable, and Scalable Consistency Model)

**File:** [`sCM.py`](sCM.py) | **Reference:** [Lu & Song, 2024](https://arxiv.org/abs/2410.11081)

Enforce continous-time consistency `d/dt f(x_t,t) = 0` and use JVP-based training with TrigFlow parameterization for improved stability.

**Key Parameters:**
- `loss_config.use_cd`: Use consistency distillation (requires teacher; `guidance_scale` controls CFG for the teacher)
- `loss_config.use_jvp_finite_diff`: Use finite difference for JVP (e.g., for compatibility with Flash Attention and FSDP)
- `sample_t_cfg`: Config of the distribution for sampling `t`

**Configs:** [`EDM/config_sct_cifar10.py`](../../configs/experiments/EDM/config_sct_cifar10.py), [`EDM/config_scd_cifar10.py`](../../configs/experiments/EDM/config_scd_cifar10.py), [`EDM/config_scd_in64.py`](../../configs/experiments/EDM/config_scd_in64.py), [`EDM2/config_scm_xl.py`](../../configs/experiments/EDM2/config_scm_xl.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_sct_cifar10.py`](../../configs/experiments/EDM/config_sct_cifar10.py) | CIFAR-10 | No | 1 | 3.23 | 2.85 |
| [`EDM/config_scd_cifar10.py`](../../configs/experiments/EDM/config_scd_cifar10.py) | CIFAR-10 | No | 1 | 3.22 | 3.66 |


---

## TCM (Truncated Consistency Model)

**File:** [`TCM.py`](TCM.py) | **Reference:** [Lee et al., 2024](https://arxiv.org/abs/2410.14895)

Two-stage training: frozen Stage-1 CM for `t < transition_t`, trainable student for `t >= transition_t`.

**Key Parameters:**
- `transition_t`: Stage transition threshold
- `boundary_prob`, `w_boundary`: Probability of boundary timestep sampling and boundary loss weight
- Requires a pretrained CM checkpoint (Stage-1 model) via `trainer.checkpointer.pretrained_ckpt_path`

**Configs:** [`EDM/config_tcm_cifar10.py`](../../configs/experiments/EDM/config_tcm_cifar10.py), [`EDM2/config_tcm_s.py`](../../configs/experiments/EDM2/config_tcm_s.py), [`EDM2/config_tcm_xl.py`](../../configs/experiments/EDM2/config_tcm_xl.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_tcm_cifar10.py`](../../configs/experiments/EDM/config_tcm_cifar10.py) | CIFAR-10 | No | 1 | 2.70 | 2.46 |
| [`EDM2/config_tcm_xl.py`](../../configs/experiments/EDM2/config_tcm_xl.py) | ImageNet-64 (EDM2 version) | Yes | 1 | 2.23 | 2.20 |


---

## MeanFlow

**File:** [`mean_flow.py`](mean_flow.py) | **References:** [Geng et al., 2025](https://arxiv.org/abs/2505.13447), [Sabour et al., 2025](https://arxiv.org/abs/2506.14603)

Learns average velocity between trajectory points: `x_r = x_t - (t-r) · u(x_t, t, r)`.

**Key Parameters:**
- `loss_config.use_cd`: Use consistency distillation (requires teacher; `guidance_scale` controls CFG for the teacher)
- `loss_config.use_jvp_finite_diff`: Use finite difference for JVP (e.g., for compatibility with Flash Attention and FSDP)
- `sample_t_cfg`, `sample_r_cfg`: Configs of the distributions for sampling `t` and `r`
- `sample_t_cfg.flow_matching_ratio`: Fraction of the batch with `r = t` (flow matching loss)
- `guidance_scale`, `guidance_fuse_scale`: If `guidance_fuse_scale` is `None`, use target-side guidance (`guidance_scale`, 3-way mixture via `guidance_mixture_ratio`, t-window via `guidance_t_start` / `guidance_t_end`), otherwise use prediction-side guidance (conditional output learns the guided flow directly)


**Configs:** [`EDM/config_mf_cifar10.py`](../../configs/experiments/EDM/config_mf_cifar10.py), [`DiT/config_mf_b.py`](../../configs/experiments/DiT/config_mf_b.py), [`DiT/config_mf_xl.py`](../../configs/experiments/DiT/config_mf_xl.py), [`WanT2V/config_mf.py`](../../configs/experiments/WanT2V/config_mf.py), [`WanT2V/config_anyflow.py`](../../configs/experiments/WanT2V/config_anyflow.py) (Stage 1 of [AnyFlow](../distribution_matching/README.md#anyflow))


**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_mf_cifar10.py`](../../configs/experiments/EDM/config_mf_cifar10.py) | CIFAR-10 | No | 1 | 2.82 | 2.92 |
| [`DiT/config_mf_xl.py`](../../configs/experiments/DiT/config_mf_xl.py) | ImageNet-256 | Yes | 1 | 3.19 | 3.43 |


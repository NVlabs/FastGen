# Distribution Matching Methods

Methods that train a student to match the teacher's output distribution using adversarial and score-based objectives.

## DMD2 (Distribution Matching Distillation v2)

**File:** [`dmd2.py`](dmd2.py) | **Reference:** [Yin et al., 2024](https://arxiv.org/abs/2405.14867)

Combines VSD loss with GAN training. Alternates between student and fake_score/discriminator updates.

**Key Parameters:**
- `student_update_freq`: Student update frequency
- `gan_loss_weight_gen`: GAN loss weight
- `guidance_scale`: CFG scale for teacher
- `sample_t_cfg`: Config of the distribution for sampling the noising time `t` and to define custom timesteps for multi-step inference (requires setting `student_sample_steps` accordingly)

**Configs:** [`EDM/config_dmd2_cifar10.py`](../../configs/experiments/EDM/config_dmd2_cifar10.py), [`EDM/config_dmd2_in64.py`](../../configs/experiments/EDM/config_dmd2_in64.py), [`SD15/config_dmd2.py`](../../configs/experiments/SD15/config_dmd2.py), [`SDXL/config_dmd2.py`](../../configs/experiments/SDXL/config_dmd2.py), [`Flux/config_dmd2.py`](../../configs/experiments/Flux/config_dmd2.py), [`WanT2V/config_dmd2.py`](../../configs/experiments/WanT2V/config_dmd2.py), [`WanI2V/config_dmd2_14b.py`](../../configs/experiments/WanI2V/config_dmd2_14b.py), [`CosmosPredict2/config_dmd2.py`](../../configs/experiments/CosmosPredict2/config_dmd2.py), [`CogVideoX/config_dmd2.py`](../../configs/experiments/CogVideoX/config_dmd2.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_dmd2_cifar10.py`](../../configs/experiments/EDM/config_dmd2_cifar10.py) | CIFAR-10 | Yes | 1 | 1.99 | 2.13 |
| [`EDM/config_dmd2_in64.py`](../../configs/experiments/EDM/config_dmd2_in64.py) | ImageNet-64 (EDM version) | Yes | 1 | 1.12 | 1.28 |

| Config | Dataset | Task | Steps | VBench (Total) | VBench (Quality) | VBench (Semantic) |
|--------|---------|------|-------|----------------------|-------------------------|------------------------|
| [`WanT2V/config_dmd2.py`](../../configs/experiments/WanT2V/config_dmd2.py) | Generated from Wan2.1 14B with [VidProM prompts](https://vidprom.github.io/) | T2V | 2 | 84.53 | 85.69 | 79.92 |
| [`WanT2V/config_dmd2.py`](../../configs/experiments/WanT2V/config_dmd2.py) | Generated from Wan2.1 14B with [VidProM prompts](https://vidprom.github.io/) | T2V | 4 | 84.72 | 85.86 | 80.15 |


---

## f-Distill

**File:** [`f_distill.py`](f_distill.py) | **Reference:** [Xu et al., 2025](https://arxiv.org/abs/2502.15681)

DMD2 with f-divergence weighted VSD loss using discriminator density ratio estimates.

**Key Parameters:**
- `f_distill.f_div`: Divergence type (`js`, `kl`, `rkl`, `sh`, `neyman`, `jf`, `sf`)
- `f_distill.ratio_lower/upper`: Ratio clipping bounds
- See also key parameters of DMD2 above


**Configs:** [`EDM/config_f_distill_cifar10.py`](../../configs/experiments/EDM/config_f_distill_cifar10.py), [`EDM/config_f_distill_in64.py`](../../configs/experiments/EDM/config_f_distill_in64.py), [`SD15/config_f_distill.py`](../../configs/experiments/SD15/config_f_distill.py), [`SDXL/config_f_distill.py`](../../configs/experiments/SDXL/config_f_distill.py), [`WanT2V/config_fdistill.py`](../../configs/experiments/WanT2V/config_fdistill.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_dmd2_cifar10.py`](../../configs/experiments/EDM/config_dmd2_cifar10.py) | CIFAR-10 | Yes | 1 | 1.85 |1.92 |
| [`EDM/config_dmd2_in64.py`](../../configs/experiments/EDM/config_dmd2_in64.py) | ImageNet-64 (EDM version) | Yes | 1 | 1.11 | 1.16 |

---

## LADD (Latent Adversarial Distillation)

**File:** [`ladd.py`](ladd.py) | **Reference:** [Sauer et al., 2024](https://arxiv.org/abs/2403.12015)

Pure adversarial training without score matching. 

**Key Parameters:**
- `student_update_freq`: Update frequency
- `gan_r1_reg_weight`: R1 regularization weight
- `sample_t_cfg`: Config of the distribution for sampling the noising time `t` and to define custom timesteps for multi-step inference (requires setting `student_sample_steps` accordingly)

**Configs:** [`WanT2V/config_ladd.py`](../../configs/experiments/WanT2V/config_ladd.py)

---

## CausVid

**File:** [`causvid.py`](causvid.py) | **Reference:** [Yin et al., 2024](https://arxiv.org/abs/2412.07772)

DMD2 extended for causal video generation with autoregressive chunk-by-chunk processing and KV-caching.

**Key Parameters:**
- `student_sample_steps`: Denoising steps per chunk (default: 4)
- `context_noise`: Noise level added to cached context
- See also the key parameters of DMD2 above


**Configs:** [`WanT2V/config_causvid.py`](../../configs/experiments/WanT2V/config_causvid.py)

---

## Self-Forcing

**File:** [`self_forcing.py`](self_forcing.py) | **Reference:** [Huang et al., 2025](https://arxiv.org/abs/2506.08009)

DMD2 extended for causal video generation with gradients through autoregressive rollout and KV-caching.

**Key Parameters:**
- `enable_gradient_in_rollout`: Enable gradients at (stochastic) exit step
- `same_step_across_blocks`: Same exit step for all blocks
- `context_noise`: Noise level added to cached context
- See also the key parameters of DMD2 above


**Configs:** [`WanT2V/config_sf.py`](../../configs/experiments/WanT2V/config_sf.py), [`WanV2V/config_sf.py`](../../configs/experiments/WanV2V/config_sf.py)

**Expected results:**


| Config | Dataset | Task | Steps | VBench (Total) | VBench (Quality) | VBench (Semantic) |
|--------|---------|------|-------|----------------------|-------------------------|------------------------|
| [`WanT2V/config_sf.py`](../../configs/experiments/WanT2V/config_sf.py) | Generated from Wan2.1 14B with [VidProM prompts](https://vidprom.github.io/) | T2V | 4 | 84.27 | 85.31 | 80.14 |

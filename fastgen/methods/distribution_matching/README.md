# Distribution Matching Methods

Methods that train a student to match the teacher's output distribution using adversarial and score-based objectives.

## LADD (Latent Adversarial Distillation)

**File:** [`ladd.py`](ladd.py) | **Reference:** [Sauer et al. (2024)](https://arxiv.org/abs/2403.12015)

Pure adversarial training without score matching. 

**Key Parameters:**
- `student_update_freq`: Update frequency
- `gan_r1_reg_weight`: R1 regularization weight
- `sample_t_cfg`: Config of the distribution for sampling the noising time `t`
- `student_grid_cfg`: Scheduler-owned student grid policy and default inference NFE
- `student_generation`: `"x0"` (default) or `"step"`, as for DMD2; step generations train the GAN on the forward transition from the next grid time

**Configs:** [`WanT2V/config_ladd.py`](../../configs/experiments/WanT2V/config_ladd.py)

---

## DMD2 (Distribution Matching Distillation v2)

**File:** [`dmd2.py`](dmd2.py) | **Reference:** [Yin et al. (2024)](https://arxiv.org/abs/2405.14867); step generations (Phased DMD): [Fan et al. (2025)](https://arxiv.org/abs/2510.27684)

Combines VSD loss with GAN training. Every `student_update_freq`-th iteration updates the student and the others update the fake score and discriminator (upstream DMD2's `dfake_gen_update_ratio` also updates the fake score on generator steps). For multimodal data (e.g., joint video+audio), VSD and DSM losses are summed over modalities.

`student_generation` selects what a multistep student generates in training: its clean estimate (`"x0"`, default) or, as in Phased DMD, its `student_sample_type` step from grid time `t_i` to `t_{i+1}` (`"step"`). The losses renoise step generations with the marginal-preserving forward transition from `t_{i+1}`, so they match the generations' distribution to the data's at `t_{i+1}`. The [`dmd2.py`](dmd2.py) docstring details the losses.

**Key Parameters:**
- `student_update_freq`: Student update frequency
- `gan_loss_weight_gen`: GAN loss weight
- `guidance_cfg`: Composite [`GuidanceConfig`](../../utils/guidance.py) (CFG / mix / rescale) for the teacher
- `sample_t_cfg`: Config of the distribution for sampling the noising time `t`
- `student_grid_cfg`: Scheduler-owned student grid policy; `steps` selects the default inference NFE
- `student_generation`: `"x0"` (default) or `"step"`, see above. Step generations need a per-sample student step, so CausVid, Self-Forcing, and AnyFlow support only `"x0"`. Step generations with VSD need `fake_score_pred_type="eps"` and a `fake_score_weighting` other than `"native"`, which must be `"x0"` without `sample_t_in_student_bucket`

**Configs:** [`EDM/config_dmd2_fast.py`](../../configs/experiments/EDM/config_dmd2_fast.py), [`EDM/config_dmd2_cifar10.py`](../../configs/experiments/EDM/config_dmd2_cifar10.py), [`EDM/config_dmd2_in64.py`](../../configs/experiments/EDM/config_dmd2_in64.py), [`SD15/config_dmd2.py`](../../configs/experiments/SD15/config_dmd2.py), [`SDXL/config_dmd2.py`](../../configs/experiments/SDXL/config_dmd2.py), [`Flux/config_dmd2.py`](../../configs/experiments/Flux/config_dmd2.py), [`Flux2/config_dmd2.py`](../../configs/experiments/Flux2/config_dmd2.py), [`Flux2/config_dmd2_klein_base_4b.py`](../../configs/experiments/Flux2/config_dmd2_klein_base_4b.py), [`Flux2/config_dmd2_klein_base_9b.py`](../../configs/experiments/Flux2/config_dmd2_klein_base_9b.py), [`Flux2/config_dmd2_image_edit.py`](../../configs/experiments/Flux2/config_dmd2_image_edit.py), [`Flux2/config_dmd2_image_edit_klein_base_4b.py`](../../configs/experiments/Flux2/config_dmd2_image_edit_klein_base_4b.py), [`Flux2/config_dmd2_image_edit_klein_base_9b.py`](../../configs/experiments/Flux2/config_dmd2_image_edit_klein_base_9b.py), [`QwenImage/config_dmd2.py`](../../configs/experiments/QwenImage/config_dmd2.py), [`WanT2V/config_dmd2.py`](../../configs/experiments/WanT2V/config_dmd2.py), [`WanT2V/config_dmd2_14b.py`](../../configs/experiments/WanT2V/config_dmd2_14b.py), [`WanT2V/config_dmd2_wan22_5b.py`](../../configs/experiments/WanT2V/config_dmd2_wan22_5b.py), [`WanT2V/config_dmd2_wan22_14b.py`](../../configs/experiments/WanT2V/config_dmd2_wan22_14b.py), [`WanI2V/config_dmd2_14b.py`](../../configs/experiments/WanI2V/config_dmd2_14b.py), [`WanI2V/config_dmd2_wan22_5b.py`](../../configs/experiments/WanI2V/config_dmd2_wan22_5b.py), [`WanI2V/config_dmd2_wan22_14b.py`](../../configs/experiments/WanI2V/config_dmd2_wan22_14b.py), [`WanI2V/config_dmd2_lingbot_world.py`](../../configs/experiments/WanI2V/config_dmd2_lingbot_world.py), [`WanV2V/config_dmd2.py`](../../configs/experiments/WanV2V/config_dmd2.py), [`CosmosPredict2/config_dmd2.py`](../../configs/experiments/CosmosPredict2/config_dmd2.py), [`CosmosPredict2/config_dmd2_14b.py`](../../configs/experiments/CosmosPredict2/config_dmd2_14b.py), [`CosmosPredict2/config_dmd2_v2w.py`](../../configs/experiments/CosmosPredict2/config_dmd2_v2w.py), [`Cosmos3/config_dmd2.py`](../../configs/experiments/Cosmos3/config_dmd2.py), [`CogVideoX/config_dmd2.py`](../../configs/experiments/CogVideoX/config_dmd2.py), [`LTX2/config_dmd2.py`](../../configs/experiments/LTX2/config_dmd2.py), [`LTX2/config_dmd2_ltx23.py`](../../configs/experiments/LTX2/config_dmd2_ltx23.py), [`LTX2/config_dmd2_ltx25.py`](../../configs/experiments/LTX2/config_dmd2_ltx25.py), [`LTX2I2V/config_dmd2.py`](../../configs/experiments/LTX2I2V/config_dmd2.py), [`LTX2I2V/config_dmd2_ltx23.py`](../../configs/experiments/LTX2I2V/config_dmd2_ltx23.py), [`MiniMaxH3/config_dmd2.py`](../../configs/experiments/MiniMaxH3/config_dmd2.py)

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

**File:** [`f_distill.py`](f_distill.py) | **Reference:** [Xu et al. (2025)](https://arxiv.org/abs/2502.15681)

DMD2 with f-divergence weighted VSD loss using discriminator density ratio estimates.

**Key Parameters:**
- `f_distill.f_div`: Divergence type (`js`, `kl`, `rkl`, `sh`, `neyman`, `jf`, `sf`)
- `f_distill.ratio_lower/upper`: Ratio clipping bounds
- See also key parameters of DMD2 above


**Configs:** [`EDM/config_f_distill_cifar10.py`](../../configs/experiments/EDM/config_f_distill_cifar10.py), [`EDM/config_f_distill_in64.py`](../../configs/experiments/EDM/config_f_distill_in64.py), [`SD15/config_f_distill.py`](../../configs/experiments/SD15/config_f_distill.py), [`SDXL/config_f_distill.py`](../../configs/experiments/SDXL/config_f_distill.py), [`WanT2V/config_fdistill.py`](../../configs/experiments/WanT2V/config_fdistill.py)

**Expected results:**

| Config | Dataset | Conditional | Steps | FID (FastGen) | FID (paper) |
|--------|---------|------|-------|---------------|-------------|
| [`EDM/config_f_distill_cifar10.py`](../../configs/experiments/EDM/config_f_distill_cifar10.py) | CIFAR-10 | Yes | 1 | 1.85 |1.92 |
| [`EDM/config_f_distill_in64.py`](../../configs/experiments/EDM/config_f_distill_in64.py) | ImageNet-64 (EDM version) | Yes | 1 | 1.11 | 1.16 |


---

## AnyFlow

**File:** [`anyflow.py`](anyflow.py) | **Reference:** [Gu et al. (2026)](https://arxiv.org/abs/2605.13724)

DMD2 with a flow-map student `u(x_t, t, r)` (mean velocity from `t` back to `r`), supporting arbitrary inference NFE. The student generates by rolling out the flow map from pure noise with the NFE sampled per iteration and gradients through all segments, and co-trains the flow-map loss at every student update. Requires a Stage-1 flow-map pretrain, which runs directly on [`MeanFlowModel`](../trajectory_matching/mean_flow.py) using AnyFlow's hyperparameters.

**Key Parameters (Stage 2):**
- `student_steps_list`: Rollout NFEs sampled per iteration
- `max_steps_rollout`: Maximum number of student forwards in the differentiable rollout
- `cotrain_pretrain_weight`: Weight of the co-trained Stage-1 flow-map loss
- `cotrain_sample_t_cfg`, `cotrain_sample_r_overrides`: Separate `(t, r)` distributions for the co-trained flow-map loss; the overrides change `cotrain_sample_t_cfg` for `r` (`None` samples `r` like `t`)
- `student_grid_cfg.grid_kwargs.shift`: Shift for the scheduler-owned rollout grid
- Requires a Stage-1 pretrained checkpoint via
  `trainer.checkpointer.pretrained_ckpt_path`; the supplied Stage-2 config maps
  Stage-1 EMA weights with
  `trainer.checkpointer.pretrained_ckpt_key_map={"net": "ema", "ema": "ema"}`
- See also the key parameters of DMD2 above (and the key parameters of [MeanFlow](../trajectory_matching/README.md#meanflow) for Stage 1)

**Configs:** [`WanT2V/config_anyflow.py`](../../configs/experiments/WanT2V/config_anyflow.py) (Stage 1 pretrain), [`WanT2V/config_anyflow_onpolicy.py`](../../configs/experiments/WanT2V/config_anyflow_onpolicy.py) (Stage 2 on-policy distillation)

---

## CausVid

**File:** [`causvid.py`](causvid.py) | **Reference:** [Yin et al. (2024)](https://arxiv.org/abs/2412.07772)

DMD2 extended for causal video generation with autoregressive chunk-by-chunk processing and KV-caching.

**Key Parameters:**
- `student_grid_cfg.steps`: Denoising steps per chunk
- `context_noise`: Noise level added to cached context (`None`, the default, caches it clean)
- See also the key parameters of DMD2 above


**Configs:** [`WanT2V/config_causvid.py`](../../configs/experiments/WanT2V/config_causvid.py)

---

## Self-Forcing

**File:** [`self_forcing.py`](self_forcing.py) | **Reference:** [Huang et al. (2025)](https://arxiv.org/abs/2506.08009)

DMD2 extended for causal video generation with gradients through autoregressive rollout and KV-caching.

**Key Parameters:**
- `start_gradient_frame`: First frame whose stochastic exit step tracks gradients; must be a rollout block start
- `same_step_across_blocks`: Same exit step for all blocks
- `context_noise`: Noise level added to cached context (`None`, the default, caches it clean)
- `grad_through_kv_cache`: Keep gradients through the context chunks written to the KV cache (`False`, the default, stores them under `no_grad`)
- See also the key parameters of DMD2 above


**Configs:** [`WanT2V/config_sf.py`](../../configs/experiments/WanT2V/config_sf.py), [`WanT2V/config_sf_14b.py`](../../configs/experiments/WanT2V/config_sf_14b.py), [`WanV2V/config_sf.py`](../../configs/experiments/WanV2V/config_sf.py)

The Wan 1.3B T2V recipe downloads its configured ODE initialization through
`model.pretrained_student_net_path`. The
[`14B T2V`](../../configs/experiments/WanT2V/config_sf_14b.py) and V2V recipes
require a causal SFT or KD checkpoint: supply its `.pth` path through
`trainer.checkpointer.pretrained_ckpt_path` before training.

**Expected results:**


| Config | Dataset | Task | Steps | VBench (Total) | VBench (Quality) | VBench (Semantic) |
|--------|---------|------|-------|----------------------|-------------------------|------------------------|
| [`WanT2V/config_sf.py`](../../configs/experiments/WanT2V/config_sf.py) | Generated from Wan2.1 14B with [VidProM prompts](https://vidprom.github.io/) | T2V | 4 | 84.27 | 85.31 | 80.14 |

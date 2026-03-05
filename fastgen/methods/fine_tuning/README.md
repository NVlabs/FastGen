# Fine-Tuning Methods

Standard diffusion training with denoising score matching (DSM) / flow matching.

## SFT (Supervised Fine-Tuning)

**File:** [`sft.py`](sft.py) | **References:** [Ho et al., 2020](https://arxiv.org/abs/2006.11239), [Song et al., 2020](https://arxiv.org/abs/2011.13456), [Lipman et al., 2022](https://arxiv.org/abs/2210.02747), [Albergo et al., 2023](https://arxiv.org/abs/2303.08797)

Standard DSM / flow matching training: `L = ||f(x_t, t) - target||²`

**Key Parameters:**
- `cond_dropout_prob`: Condition dropout probability
- `guidance_scale`: CFG scale for inference
- `sample_t_cfg`: Config of the distribution for sampling `t`


**Configs:** [`Flux/config_sft.py`](../../configs/experiments/Flux/config_sft.py), [`QwenImage/config_sft.py`](../../configs/experiments/QwenImage/config_sft.py), [`WanT2V/config_sft.py`](../../configs/experiments/WanT2V/config_sft.py), [`CosmosPredict2/config_sft.py`](../../configs/experiments/CosmosPredict2/config_sft.py), [`SD15/config_sft.py`](../../configs/experiments/SD15/config_sft.py), [`SDXL/config_sft.py`](../../configs/experiments/SDXL/config_sft.py), [`EDM/config_sft_edm_cifar10.py`](../../configs/experiments/EDM/config_sft_edm_cifar10.py), [`EDM/config_sft_edm_in64.py`](../../configs/experiments/EDM/config_sft_edm_in64.py), [`EDM2/config_sft_s.py`](../../configs/experiments/EDM2/config_sft_s.py), [`EDM2/config_sft_xl.py`](../../configs/experiments/EDM2/config_sft_xl.py), [`DiT/config_sft_dit_xl.py`](../../configs/experiments/DiT/config_sft_dit_xl.py)

---

## CausalSFT

**File:** [`sft.py`](sft.py) | **Reference:** [Chen et al., 2024](https://arxiv.org/abs/2407.01392)

SFT for causal video models with inhomogeneous timesteps per frame chunk.

**Key Parameters:**
- `context_noise`: Noise level for context frames
- Inherits also the key parameters of SFT above

**Configs:** [`WanT2V/config_sft_causal.py`](../../configs/experiments/WanT2V/config_sft_causal.py), [`WanI2V/config_sft_causal_14b.py`](../../configs/experiments/WanI2V/config_sft_causal_14b.py), [`WanI2V/config_sft_causal_wan22_5b.py`](../../configs/experiments/WanI2V/config_sft_causal_wan22_5b.py), [`WanV2V/config_sft_causal.py`](../../configs/experiments/WanV2V/config_sft_causal.py)

# Fine-Tuning Methods

Standard diffusion training with denoising score matching (DSM) / flow matching.

## SFT (Supervised Fine-Tuning)

**File:** [`sft.py`](sft.py) | **References:** [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), [Song et al. (2020)](https://arxiv.org/abs/2011.13456), [Lipman et al. (2022)](https://arxiv.org/abs/2210.02747), [Albergo et al. (2023)](https://arxiv.org/abs/2303.08797)

Standard DSM / flow matching training: `L = ||f(x_t, t) - target||²`.
For multimodal data (e.g., joint video+audio), DSM losses are computed per modality and summed. 

**Key Parameters:**
- `cond_dropout_prob`: Condition dropout probability
- `guidance_cfg`: Composite [`GuidanceConfig`](../../utils/guidance.py) (CFG / mix / rescale) used by the network's `sample()` at inference time. Piecewise t-windows are encoded via `cfg.scale = [s_in, s_out]` + `cfg.boundary = [t_end]`
- `sample_t_cfg`: Config of the distribution for sampling `t`


**Configs:** [`Flux/config_sft.py`](../../configs/experiments/Flux/config_sft.py), [`Flux2/config_sft.py`](../../configs/experiments/Flux2/config_sft.py), [`Flux2/config_sft_klein_base_4b.py`](../../configs/experiments/Flux2/config_sft_klein_base_4b.py), [`Flux2/config_sft_klein_base_9b.py`](../../configs/experiments/Flux2/config_sft_klein_base_9b.py), [`Flux2/config_sft_image_edit.py`](../../configs/experiments/Flux2/config_sft_image_edit.py), [`Flux2/config_sft_image_edit_klein_base_4b.py`](../../configs/experiments/Flux2/config_sft_image_edit_klein_base_4b.py), [`Flux2/config_sft_image_edit_klein_base_9b.py`](../../configs/experiments/Flux2/config_sft_image_edit_klein_base_9b.py), [`QwenImage/config_sft.py`](../../configs/experiments/QwenImage/config_sft.py), [`WanT2V/config_sft.py`](../../configs/experiments/WanT2V/config_sft.py), [`WanT2V/config_sft_14b.py`](../../configs/experiments/WanT2V/config_sft_14b.py), [`WanT2V/config_sft_wan22_14b.py`](../../configs/experiments/WanT2V/config_sft_wan22_14b.py), [`WanI2V/config_sft_14b.py`](../../configs/experiments/WanI2V/config_sft_14b.py), [`WanI2V/config_sft_wan22_5b.py`](../../configs/experiments/WanI2V/config_sft_wan22_5b.py), [`WanI2V/config_sft_wan22_14b.py`](../../configs/experiments/WanI2V/config_sft_wan22_14b.py), [`WanI2V/config_sft_lingbot_world.py`](../../configs/experiments/WanI2V/config_sft_lingbot_world.py), [`WanV2V/config_sft.py`](../../configs/experiments/WanV2V/config_sft.py), [`WanV2V/config_sft_14b.py`](../../configs/experiments/WanV2V/config_sft_14b.py), [`CosmosPredict2/config_sft.py`](../../configs/experiments/CosmosPredict2/config_sft.py), [`CosmosPredict2/config_sft_14b.py`](../../configs/experiments/CosmosPredict2/config_sft_14b.py), [`CosmosPredict2/config_sft_v2w.py`](../../configs/experiments/CosmosPredict2/config_sft_v2w.py), [`Cosmos3/config_sft.py`](../../configs/experiments/Cosmos3/config_sft.py), [`CogVideoX/config_sft.py`](../../configs/experiments/CogVideoX/config_sft.py), [`SD15/config_sft.py`](../../configs/experiments/SD15/config_sft.py), [`SDXL/config_sft.py`](../../configs/experiments/SDXL/config_sft.py), [`EDM/config_sft_edm_cifar10.py`](../../configs/experiments/EDM/config_sft_edm_cifar10.py), [`EDM/config_sft_edm_in64.py`](../../configs/experiments/EDM/config_sft_edm_in64.py), [`EDM2/config_sft_s.py`](../../configs/experiments/EDM2/config_sft_s.py), [`EDM2/config_sft_xl.py`](../../configs/experiments/EDM2/config_sft_xl.py), [`DiT/config_sft_dit_xl.py`](../../configs/experiments/DiT/config_sft_dit_xl.py), [`DiT/config_sft_sit_xl.py`](../../configs/experiments/DiT/config_sft_sit_xl.py), [`LTX2/config_sft.py`](../../configs/experiments/LTX2/config_sft.py), [`LTX2/config_sft_ltx23.py`](../../configs/experiments/LTX2/config_sft_ltx23.py), [`LTX2/config_sft_ltx25.py`](../../configs/experiments/LTX2/config_sft_ltx25.py), [`LTX2I2V/config_sft.py`](../../configs/experiments/LTX2I2V/config_sft.py), [`LTX2I2V/config_sft_ltx23.py`](../../configs/experiments/LTX2I2V/config_sft_ltx23.py), [`MiniMaxH3/config_sft.py`](../../configs/experiments/MiniMaxH3/config_sft.py)

---

## CausalSFT

**Classes:** [`CausalSFTModel`](sft.py), [`CausalTeacherForcingSFTModel`](sft.py) | **References:** [Yin et al. (2024)](https://arxiv.org/abs/2412.07772), [Zhu et al. (2026)](https://arxiv.org/abs/2602.02214)

CausalSFT has two variants. `CausalSFTModel` uses diffusion forcing with
inhomogeneous timesteps per frame chunk. `CausalTeacherForcingSFTModel` uses
teacher forcing, feeding a clean-context stream and a noisy denoising stream
through the causal network together using an attention mask; its DSM loss is
computed on the noisy stream.

**Key Parameters:**
- `context_noise`: Time to which chunk-by-chunk student sampling re-noises each
  finished chunk before writing it to the KV cache (`None`, the default, caches
  it clean). It affects sampling only, not the training loss.
- `net.chunk_size`: Number of frames per causal chunk; controls timestep
  sharing and causal attention granularity. The first chunk also takes any
  remainder frames.
- `sample_t_context_cfg`: Teacher forcing only (method config
  [`config_sft_teacher_forcing.py`](../../configs/methods/config_sft_teacher_forcing.py)).
  The context stream is clean by default; `enabled=True` noise-augments it at
  separately drawn timesteps, and `teacher_forcing_context_match_train_t=True`
  instead feeds it the same noisy frames as the denoising stream.
- Inherits the SFT parameters above

**Configs:** [`WanT2V/config_sft_causal.py`](../../configs/experiments/WanT2V/config_sft_causal.py), [`WanT2V/config_sft_causal_14b.py`](../../configs/experiments/WanT2V/config_sft_causal_14b.py), [`WanI2V/config_sft_causal_14b.py`](../../configs/experiments/WanI2V/config_sft_causal_14b.py), [`WanI2V/config_sft_causal_wan22_5b.py`](../../configs/experiments/WanI2V/config_sft_causal_wan22_5b.py), [`WanV2V/config_sft_causal.py`](../../configs/experiments/WanV2V/config_sft_causal.py), [`WanT2V/config_sft_teacher_forcing.py`](../../configs/experiments/WanT2V/config_sft_teacher_forcing.py), [`WanT2V/config_sft_teacher_forcing_14b.py`](../../configs/experiments/WanT2V/config_sft_teacher_forcing_14b.py)

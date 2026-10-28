# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import copy
from omegaconf import DictConfig

from fastgen.networks.EDM.network import DhariwalUNet, EDMPrecond, SongUNet
from fastgen.networks.EDM2.network import EDM2Precond
from fastgen.networks.DiT.network import DiT
from fastgen.networks.SD15.network import StableDiffusion15
from fastgen.networks.SDXL.network import StableDiffusionXL
from fastgen.networks.Flux.network import Flux
from fastgen.networks.Flux2.network import Flux2
from fastgen.networks.QwenImage.network import QwenImage
from fastgen.networks.CogVideoX.network import CogVideoX
from fastgen.networks.LTX2.network import LTX2
from fastgen.networks.MiniMaxH3.network import MiniMaxH3
from fastgen.networks.Wan.network import Wan
from fastgen.networks.Wan.network_causal import CausalWan
from fastgen.networks.Wan.wan_moe import WanMoET2V
from fastgen.networks.WanI2V.wan_moe import WanMoEI2V
from fastgen.networks.WanI2V.network import WanI2V
from fastgen.networks.WanI2V.network_causal import CausalWanI2V
from fastgen.networks.VaceWan.modules.vace_depth_annotator import DEPTH_ANYTHING_V2_VITL_URL
from fastgen.networks.VaceWan.network import VACEWan
from fastgen.networks.VaceWan.network_causal import CausalVACEWan
from fastgen.networks.WanI2V.wan_moe import LingBotWorld
from fastgen.networks.cosmos_predict2.network import CosmosPredict2
from fastgen.networks.cosmos_predict2.modules import SACConfig, CheckpointMode
from fastgen.networks.cosmos3.network import Cosmos3

from fastgen.configs.instantiate import LazyCall as L
from fastgen.utils.env import output_root

CKPT_ROOT_DIR = os.getenv("CKPT_ROOT_DIR", f"{output_root()}/MODEL")

# Immutable/single-file model sources mirrored by ``fastgen.utils.cache_utils``.
DIT_XL_2_256_URL = "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
SIT_XL_2_256_URL = (
    "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/"
    "SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=1"
)
SIT_XL_2_REPA_256_URL = (
    "https://www.dl.dropboxusercontent.com/scl/fi/"
    "cxedbs4da5ugjq5wg3zrg/last.pt?"
    "rlkey=8otgrdkno0nd89po3dpwngwcc&dl=1"
)
SELF_FORCING_ODE_INIT_URL = (
    "https://huggingface.co/gdhe17/Self-Forcing/resolve/"
    "2f8b779212da279d212c22a509b66ad6552f350e/checkpoints/ode_init.pt"
)

EDM_CIFAR10_Config: DictConfig = L(EDMPrecond)(
    sigma_shift=0.0,
    sigma_data=0.5,
    backbone=L(SongUNet)(
        img_resolution=32,
        in_channels=3,
        out_channels=3,
        label_dim=10,
        augment_dim=9,
        model_channels=128,
        channel_mult=[2, 2, 2],
        channel_mult_noise=1,
        embedding_type="positional",
        encoder_type="standard",
        decoder_type="standard",
        resample_filter=[1, 1],
        dropout=0.0,
        label_dropout=0,
        r_timestep=False,
    ),
    drop_precond=None,
)

EDM_ImageNet64_Config: DictConfig = L(EDMPrecond)(
    sigma_shift=0.0,
    sigma_data=0.5,
    backbone=L(DhariwalUNet)(
        img_resolution=64,
        in_channels=3,
        out_channels=3,
        label_dim=1000,
        augment_dim=0,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=[32, 16, 8],
        dropout=0.0,
        label_dropout=0,
        r_timestep=False,
    ),
    drop_precond=None,
)

EDM2_IN64_S_Config: DictConfig = L(EDM2Precond)(
    img_resolution=64,
    img_channels=3,
    label_dim=1000,
    sigma_data=0.5,
    sigma_shift=0.0,
    logvar_channels=128,
    model_channels=192,
    channel_mult=[1, 2, 3, 4],
    channel_mult_noise=None,
    channel_mult_emb=None,
    num_blocks=3,
    attn_resolutions=[16, 8],
    label_balance=0.5,
    concat_balance=0.5,
    dropout=0.0,
    r_timestep=False,
    drop_precond=None,
)

EDM2_IN64_M_Config = copy.deepcopy(EDM2_IN64_S_Config)
EDM2_IN64_M_Config.model_channels = 256

EDM2_IN64_L_Config = copy.deepcopy(EDM2_IN64_S_Config)
EDM2_IN64_L_Config.model_channels = 320

EDM2_IN64_XL_Config = copy.deepcopy(EDM2_IN64_S_Config)
EDM2_IN64_XL_Config.model_channels = 384

# DiT on ImageNet-256 latents (patch size 2), trained from scratch: flow matching on FastGen's RF schedule, sampled
# from pure noise at t = 1 (the DDPM release below caps max_t at its table's 0.999).
DiT_IN256_Kwargs = dict(
    input_size=32,
    patch_size=2,
    in_channels=4,
    mlp_ratio=4.0,
    class_dropout_prob=0.1,
    enable_class_dropout=False,
    label_dim=1000,
    learn_sigma=False,
    r_timestep=False,
    scale_t=True,
    max_t=1.0,
)
DiT_IN256_S_Kwargs = dict(hidden_size=384, depth=12, num_heads=6, **DiT_IN256_Kwargs)
DiT_IN256_B_Kwargs = dict(hidden_size=768, depth=12, num_heads=12, **DiT_IN256_Kwargs)
DiT_IN256_L_Kwargs = dict(hidden_size=1024, depth=24, num_heads=16, **DiT_IN256_Kwargs)
DiT_IN256_XL_Kwargs = dict(hidden_size=1152, depth=28, num_heads=16, **DiT_IN256_Kwargs)

DiT_IN256_S_Config: DictConfig = L(DiT)(**DiT_IN256_S_Kwargs)
DiT_IN256_B_Config: DictConfig = L(DiT)(**DiT_IN256_B_Kwargs)
DiT_IN256_L_Config: DictConfig = L(DiT)(**DiT_IN256_L_Kwargs)
DiT_IN256_XL_Config: DictConfig = L(DiT)(**DiT_IN256_XL_Kwargs)

# Checkpoints trained with the official DiT, SiT and REPA code, at any size (released weights are XL/2 only); models
# trained from scratch in FastGen use the configs above. DiT and SiT's ``forward_with_cfg`` guides only the first 3
# latent channels, and DiT's 1000-step DDPM table caps max_t at 0.999.
DiT_Release_Kwargs = dict(learn_sigma=True, net_pred_type="eps", schedule_type="ddpm", max_t=0.999, cfg_channels=3)
# The SiT release: flow matching in SiT's time convention (t -> 1 - t, v -> -v) on continuous time in [0, 1],
# sampled from pure noise (max_t = 1 above), with sd-vae-ft-ema for evaluation.
SiT_Release_Kwargs = dict(
    learn_sigma=True,  # the checkpoints output 8 channels
    use_sit_convention=True,
    scale_t=False,
    vae_variant="ema",
    cfg_channels=3,
)
# The REPA release: the SiT architecture in FastGen's RF time convention, all-channel CFG.
REPA_Release_Kwargs = dict(scale_t=False, vae_variant="ema", cfg_channels=None)

DiT_S_2_256_Config: DictConfig = L(DiT)(**(DiT_IN256_S_Kwargs | DiT_Release_Kwargs))
DiT_B_2_256_Config: DictConfig = L(DiT)(**(DiT_IN256_B_Kwargs | DiT_Release_Kwargs))
DiT_L_2_256_Config: DictConfig = L(DiT)(**(DiT_IN256_L_Kwargs | DiT_Release_Kwargs))
DiT_XL_2_256_Config: DictConfig = L(DiT)(**(DiT_IN256_XL_Kwargs | DiT_Release_Kwargs))

SiT_S_2_256_Config: DictConfig = L(DiT)(**(DiT_IN256_S_Kwargs | SiT_Release_Kwargs))
SiT_B_2_256_Config: DictConfig = L(DiT)(**(DiT_IN256_B_Kwargs | SiT_Release_Kwargs))
SiT_L_2_256_Config: DictConfig = L(DiT)(**(DiT_IN256_L_Kwargs | SiT_Release_Kwargs))
SiT_XL_2_256_Config: DictConfig = L(DiT)(**(DiT_IN256_XL_Kwargs | SiT_Release_Kwargs))

# REPA reports B/2, L/2 and XL/2.
SiT_B_2_REPA_256_Config: DictConfig = L(DiT)(**(DiT_IN256_B_Kwargs | REPA_Release_Kwargs))
SiT_L_2_REPA_256_Config: DictConfig = L(DiT)(**(DiT_IN256_L_Kwargs | REPA_Release_Kwargs))
SiT_XL_2_REPA_256_Config: DictConfig = L(DiT)(**(DiT_IN256_XL_Kwargs | REPA_Release_Kwargs))

SD15Config: DictConfig = L(StableDiffusion15)()

SDXLConfig: DictConfig = L(StableDiffusionXL)()

# Flux.1-dev
FluxConfig: DictConfig = L(Flux)(
    guidance_scale=3.5,
)

# ------ Flux.2 models ------

# Flux.2-dev
Flux2Config: DictConfig = L(Flux2)(
    guidance_scale=4.0,
)

Flux2KleinBase4BConfig: DictConfig = L(Flux2)(
    model_id_or_local_path="black-forest-labs/FLUX.2-klein-base-4B",
    guidance_scale=None,  # Undistilled base model, no guidance embedding
)

# FLUX.2-klein-base-9B: undistilled 9B foundation model
# Architecture: 8 double + 24 single = 32 blocks, hidden=4096
# (32 heads x 128 head_dim). Text encoder is the Qwen3 "8B" variant;
# Like klein-base-4B, this variant was trained WITHOUT step or guidance
# distillation, so `guidance_scale=None` (no embedded guidance vector).
# Sample-time CFG happens at the pipeline level via `model.guidance_cfg.cfg.scale`.
Flux2KleinBase9BConfig: DictConfig = L(Flux2)(
    model_id_or_local_path="black-forest-labs/FLUX.2-klein-base-9B",
    guidance_scale=None,  # Undistilled base model, no guidance embedding
)

# ------ Flux.2 image-editing variants ------
Flux2ImageEditConfig: DictConfig = L(Flux2)(
    enable_image_edit=True,
)

Flux2KleinBase4BImageEditConfig: DictConfig = L(Flux2)(
    model_id_or_local_path="black-forest-labs/FLUX.2-klein-base-4B",
    guidance_scale=None,
    enable_image_edit=True,
)

Flux2KleinBase9BImageEditConfig: DictConfig = L(Flux2)(
    model_id_or_local_path="black-forest-labs/FLUX.2-klein-base-9B",
    guidance_scale=None,
    enable_image_edit=True,
)

QwenImageConfig: DictConfig = L(QwenImage)()

CogVideoXConfig: DictConfig = L(CogVideoX)(
    model_id_or_local_path="THUDM/CogVideoX-2b",
    disable_grad_ckpt=False,
    # cuDNN's reduced-precision SDPA backward returns non-finite gradients for CogVideoX-2B.
    sdpa_no_cudnn=True,
)
CogVideoX5BConfig: DictConfig = L(CogVideoX)(
    model_id_or_local_path="THUDM/CogVideoX-5b",
    disable_grad_ckpt=False,
)

LTX2Config: DictConfig = L(LTX2)(
    model_id_or_local_path="Lightricks/LTX-2",
    disable_grad_ckpt=False,
    fps=24.0,
    train_connectors=False,
)

LTX2_3_Config: DictConfig = L(LTX2)(
    model_id_or_local_path="diffusers/LTX-2.3-Diffusers",
    disable_grad_ckpt=False,
    fps=24.0,
    train_connectors=False,
)

# LTX-2.5 ships a distilled `transformer/` and a full/SFT `transformer_full/`.
LTX2_5_Config: DictConfig = L(LTX2)(
    model_id_or_local_path="Lightricks/LTX-2.5-Diffusers",
    transformer_subfolder="transformer_full",
    disable_grad_ckpt=False,
    fps=24.0,
    train_connectors=False,
)

# ------ MiniMax-H3 ------

# The ``transformer/`` checkpoint supports both text-only ``t2va`` and first/last-keyframe
# ``fl2va``; select the latter explicitly with ``is_fl2va=True``. The released recipe is
# guidance-distilled, so the default guidance config performs no CFG.
MiniMaxH3_Config: DictConfig = L(MiniMaxH3)(
    model_id_or_local_path="MiniMaxAI/MiniMax-H3",
    disable_grad_ckpt=False,
    fps=24.0,
    is_fl2va=False,
    # Honour the checkpoint's float32 carve-out when FastGen selects storage precision before
    # DDP/FSDP. Diffusers applies it at load; FastGen needs the same split after its explicit
    # precision boundary. ``False`` gives the stock uniform boundary cast.
    keep_in_fp32=True,
    # Hold the per-block AdaLN projections fixed.
    freeze_adaln=False,
    # Spatial tiling of the video VAE is on in the release, and the published frames are the
    # blended-tile ones, so turning it off changes the output.
    enable_vae_tiling=True,
)

LTX2I2VConfig: DictConfig = L(LTX2)(
    model_id_or_local_path="Lightricks/LTX-2",
    disable_grad_ckpt=False,
    fps=24.0,
    is_i2v=True,
    train_connectors=False,
)

LTX2I2V_3_Config: DictConfig = L(LTX2)(
    model_id_or_local_path="diffusers/LTX-2.3-Diffusers",
    disable_grad_ckpt=False,
    fps=24.0,
    is_i2v=True,
    train_connectors=False,
)

LTX2I2V_5_Config: DictConfig = L(LTX2)(
    model_id_or_local_path="Lightricks/LTX-2.5-Diffusers",
    transformer_subfolder="transformer_full",
    disable_grad_ckpt=False,
    fps=24.0,
    is_i2v=True,
    train_connectors=False,
)

# ------ Common Wan settings ------

Wan_Kwargs = dict(
    r_timestep=False,  # Enable r timestep for meanflow-like models
    # Precision options (Reference = original Wan code, Diffusers = WanTransformer3DModel loaded in a reduced dtype).
    # Compute the time path and the output projection under FP32 autocast like the reference, and keep the RoPE
    # tables FP32. Diffusers instead keeps selected modules' weights in FP32 when loading in a reduced dtype.
    # This matters for frozen teachers, which cannot adapt to precision drift.
    keep_original_prec=True,
    # Residual stream in fp32 across blocks. Reference=True (fp32 autocast on the residual adds); Diffusers=False
    # (rounds to bf16 after every residual add). False here rounds once per block and drops the fp32
    # checkpoint-boundary activations (about 2 bytes x tokens x hidden x blocks; 11 GiB/GPU for the 14B 480p recipe).
    fp32_residual=True,
)

# ------ Wan T2V models ------
Wan_1_3B_Config: DictConfig = L(Wan)(
    model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    fps=16,
    **Wan_Kwargs,
)

Wan22_T2V_5B_Config: DictConfig = L(Wan)(
    model_id_or_local_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    fps=24,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

Wan21_T2V_14B_Config: DictConfig = L(Wan)(
    model_id_or_local_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    fps=16,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

Wan22_T2V_14B_Config: DictConfig = L(WanMoET2V)(
    model_id_or_local_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    fps=16,
    guidance_boundary=0.875,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

# ------ Wan T2V causal models ------
CausalWan_1_3B_Config: DictConfig = L(CausalWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    fps=16,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

CausalWan_14B_Config: DictConfig = L(CausalWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    fps=16,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

# ------ Wan I2V models ------
Wan22_I2V_5B_Config: DictConfig = L(WanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    fps=24,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

Wan21_I2V_14B_480P_Config: DictConfig = L(WanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    fps=16,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

Wan21_I2V_14B_720P_Config: DictConfig = L(WanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    fps=16,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

Wan22_I2V_14B_Expert_Overrides = copy.deepcopy(WanMoEI2V.DEFAULT_EXPERT_OVERRIDES)
# The low expert's block-33 qk-norm outliers trigger a cuDNN bf16 attention-backward NaN.
Wan22_I2V_14B_Expert_Overrides["low"]["sdpa_no_cudnn_blocks"] = [33]

Wan22_I2V_14B_Config: DictConfig = L(WanMoEI2V)(
    model_id_or_local_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    fps=16,
    guidance_boundary=0.9,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    expert_overrides=Wan22_I2V_14B_Expert_Overrides,
    **Wan_Kwargs,
)

# ------ Wan I2V causal models ------
CausalWan22_I2V_5B_Config: DictConfig = L(CausalWanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    fps=24,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    chunk_size=3,
    # 30 of the native 31 latent frames, so every chunk holds chunk_size frames.
    total_num_frames=30,
    **Wan_Kwargs,
)

CausalWan21_I2V_14B_480P_Config: DictConfig = L(CausalWanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    fps=16,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

CausalWan21_I2V_14B_720P_Config: DictConfig = L(CausalWanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    fps=16,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

# ------ VACE-WAN models ------
VACE_Wan_1_3B_Config: DictConfig = L(VACEWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    fps=16,
    context_scale=1.0,
    depth_model_path=DEPTH_ANYTHING_V2_VITL_URL,
    **Wan_Kwargs,
)

VACE_Wan_14B_Config: DictConfig = L(VACEWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-VACE-14B-diffusers",
    fps=16,
    context_scale=1.0,
    depth_model_path=DEPTH_ANYTHING_V2_VITL_URL,
    **Wan_Kwargs,
)

# ------ VACE-WAN causal models ------
CausalVACE_Wan_1_3B_Config: DictConfig = L(CausalVACEWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    fps=16,
    context_scale=1.0,
    depth_model_path=DEPTH_ANYTHING_V2_VITL_URL,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

CausalVACE_Wan_14B_Config: DictConfig = L(CausalVACEWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-VACE-14B-diffusers",
    fps=16,
    context_scale=1.0,
    depth_model_path=DEPTH_ANYTHING_V2_VITL_URL,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

# Official LingBot-world checkpoint (Wan2.2-I2V-A14B MoE with camera conditioning)
LingBot_World_Expert_Overrides = copy.deepcopy(LingBotWorld.DEFAULT_EXPERT_OVERRIDES)
# The low expert's block-33 qk-norm outliers trigger a cuDNN bf16 attention-backward NaN.
LingBot_World_Expert_Overrides["low"]["sdpa_no_cudnn_blocks"] = [33]

LingBot_World_Config: DictConfig = L(LingBotWorld)(
    fps=16,
    guidance_boundary=0.947,
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    r_timestep=False,
    expert_overrides=LingBot_World_Expert_Overrides,
)

# ============ Cosmos Predict2.5 Configs ============
# Based on https://github.com/nvidia-cosmos/cosmos-predict2.5/tree/main/cosmos_predict2/_src/predict2/configs/text2world/defaults/net.py
# Common parameters are set as defaults in CosmosPredict2.__init__

# 2B config
CosmosPredict2_2B_Config: DictConfig = L(CosmosPredict2)(
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    sac_config=L(SACConfig)(mode=CheckpointMode.BLOCK_WISE),
    # Checkpoint-specific override: the released post-trained models use
    # 16 FPS with temporal RoPE FPS modulation disabled.
    fps=16,
    rope_enable_fps_modulation=False,
    # Checkpoint-specific override: the released 2B post-trained model samples on the Karras preset.
    sample_sigma_schedule="karras",
    is_video2world=False,
    num_conditioning_frames=1,
    # Skip the slow custom init when parameters will be overwritten by the pretrained checkpoint.
    skip_custom_init=True,
)

# 14B config: override only the architecture-specific parameters
CosmosPredict2_14B_Config = copy.deepcopy(CosmosPredict2_2B_Config)
CosmosPredict2_14B_Config.model_channels = 5120
CosmosPredict2_14B_Config.num_blocks = 36
CosmosPredict2_14B_Config.num_heads = 40
# Undo the 2B's Karras override: the 14B checkpoint samples on the native shifted-linear default.
CosmosPredict2_14B_Config.sample_sigma_schedule = "linear"

# AGGRESSIVE mode: only saves flash attention outputs (~50% memory savings, moderate compute overhead)
CosmosPredict2_2B_Aggressive_Config = copy.deepcopy(CosmosPredict2_2B_Config)
CosmosPredict2_2B_Aggressive_Config.sac_config = L(SACConfig)(mode=CheckpointMode.AGGRESSIVE)

CosmosPredict2_14B_Aggressive_Config = copy.deepcopy(CosmosPredict2_14B_Config)
CosmosPredict2_14B_Aggressive_Config.sac_config = L(SACConfig)(mode=CheckpointMode.AGGRESSIVE)


# ============ Cosmos3 Configs ============
# Omni Mixture-of-Transformers; FastGen integrates the video-generation tower only.
# Arch values are the released nvidia/Cosmos3-Nano transformer/config.json (8B-class generator).
# The pinned Hugging Face snapshot supplies its sharded transformer, tokenizer, and Wan2.2 VAE.
Cosmos3_Nano_Config: DictConfig = L(Cosmos3)(
    model_id_or_local_path="nvidia/Cosmos3-Nano",
    revision=Cosmos3.MODEL_REVISION,
    is_i2v=False,  # True: single-anchor-frame I2V conditioning
    # Match the reference's explicit FP32 timestep-embedding computation.
    keep_time_embedder_fp32=True,
    # Activation checkpointing on the 36 MoT decoder layers. The forward gates on autograd state,
    # so it is a no-op for no_grad inference but still covers DMD2's frozen eval-mode teacher when
    # the GAN-generator loss needs input gradients. Trades the per-layer activation stack for
    # recompute -> large training-memory saving; provides headroom for longer-clip (121f/189f)
    # training. Covers student + auto-derived fake_score + grad-enabled teacher feature passes.
    gradient_checkpointing=True,
)

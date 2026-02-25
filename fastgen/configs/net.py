# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import copy
from omegaconf import DictConfig

from fastgen.networks.EDM.network import EDMPrecond
from fastgen.networks.EDM2.network import EDM2Precond
from fastgen.networks.DiT.network import DiT
from fastgen.networks.SD15.network import StableDiffusion15
from fastgen.networks.SDXL.network import StableDiffusionXL
from fastgen.networks.Flux.network import Flux
from fastgen.networks.QwenImage.network import QwenImage
from fastgen.networks.CogVideoX.network import CogVideoX
from fastgen.networks.Wan.network import Wan
from fastgen.networks.Wan.network_causal import CausalWan
from fastgen.networks.WanI2V.network import WanI2V
from fastgen.networks.WanI2V.network_causal import CausalWanI2V
from fastgen.networks.VaceWan.network import VACEWan
from fastgen.networks.VaceWan.network_causal import CausalVACEWan
from fastgen.networks.cosmos_predict2.network import CosmosPredict2
from fastgen.networks.cosmos_predict2.modules import SACConfig, CheckpointMode

from fastgen.utils import LazyCall as L

OUTPUT_ROOT = os.environ.get("FASTGEN_OUTPUT_ROOT", "FASTGEN_OUTPUT")
CKPT_ROOT_DIR = os.getenv("CKPT_ROOT_DIR", f"{OUTPUT_ROOT}/MODEL")

EDM_CIFAR10_Config: DictConfig = L(EDMPrecond)(
    img_resolution=32,
    img_channels=3,
    label_dim=10,
    sigma_shift=0.0,
    sigma_data=0.5,
    model_type="SongUNet",
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
    drop_precond=None,
)

EDM_ImageNet64_Config: DictConfig = L(EDMPrecond)(
    img_resolution=64,
    img_channels=3,
    label_dim=1000,
    sigma_shift=0.0,
    sigma_data=0.5,
    model_type="DhariwalUNet",
    augment_dim=0,
    model_channels=192,
    channel_mult=[1, 2, 3, 4],
    channel_mult_emb=4,
    num_blocks=3,
    attn_resolutions=[32, 16, 8],
    dropout=0.0,
    label_dropout=0,
    r_timestep=False,
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

DiT_IN256_S_Config: DictConfig = L(DiT)(
    input_size=32,
    patch_size=2,
    in_channels=4,
    hidden_size=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    class_dropout_prob=0.1,
    enable_class_dropout=False,
    num_classes=1000,
    learn_sigma=False,
    r_timestep=False,
    scale_t=True,
)

DiT_IN256_B_Config = copy.deepcopy(DiT_IN256_S_Config)
DiT_IN256_B_Config.hidden_size = 768
DiT_IN256_B_Config.depth = 12
DiT_IN256_B_Config.num_heads = 12

DiT_IN256_L_Config = copy.deepcopy(DiT_IN256_S_Config)
DiT_IN256_L_Config.hidden_size = 1024
DiT_IN256_L_Config.depth = 24
DiT_IN256_L_Config.num_heads = 16

DiT_IN256_XL_Config = copy.deepcopy(DiT_IN256_S_Config)
DiT_IN256_XL_Config.hidden_size = 1152
DiT_IN256_XL_Config.depth = 28
DiT_IN256_XL_Config.num_heads = 16

SD15Config: DictConfig = L(StableDiffusion15)()

SDXLConfig: DictConfig = L(StableDiffusionXL)()

FluxConfig: DictConfig = L(Flux)()

QwenImageConfig: DictConfig = L(QwenImage)()

CogVideoXConfig: DictConfig = L(CogVideoX)(
    model_id_or_local_path="THUDM/CogVideoX-2b",
    disable_grad_ckpt=False,
)
CogVideoX5BConfig: DictConfig = L(CogVideoX)(
    model_id_or_local_path="THUDM/CogVideoX-5b",
    disable_grad_ckpt=False,
)

# ------ Common Wan settings ------

Wan_Kwargs = dict(
    enable_logvar_linear=False,  # Enable logvar_linear for sCM-like models
    r_timestep=False,  # Enable r timestep for meanflow-like models
    use_wan_official_sinusoidal=False,  # False = diffusers default, True = official WAN
)

# ------ Wan T2V models ------
Wan_1_3B_Config: DictConfig = L(Wan)(
    model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    **Wan_Kwargs,
)

Wan22_T2V_5B_Config: DictConfig = L(Wan)(
    model_id_or_local_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

Wan21_T2V_14B_Config: DictConfig = L(Wan)(
    model_id_or_local_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

# ------ Wan T2V causal models ------
CausalWan_1_3B_Config: DictConfig = L(CausalWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

CausalWan_14B_Config: DictConfig = L(CausalWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

# ------ Wan I2V models ------
Wan22_I2V_5B_Config: DictConfig = L(WanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

Wan21_I2V_14B_480P_Config: DictConfig = L(WanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

Wan21_I2V_14B_720P_Config: DictConfig = L(WanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    **Wan_Kwargs,
)

# ------ Wan I2V causal models ------
CausalWan22_I2V_5B_Config: DictConfig = L(CausalWanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

CausalWan21_I2V_14B_480P_Config: DictConfig = L(CausalWanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

CausalWan21_I2V_14B_720P_Config: DictConfig = L(CausalWanI2V)(
    model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    disable_efficient_attn=False,
    disable_grad_ckpt=False,
    chunk_size=3,
    total_num_frames=21,
    **Wan_Kwargs,
)

# ------ VACE-WAN models ------
VACE_Wan_1_3B_Config: DictConfig = L(VACEWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    context_scale=1.0,
    depth_model_path=None,
    **Wan_Kwargs,
)

VACE_Wan_14B_Config: DictConfig = L(VACEWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-VACE-14B-diffusers",
    context_scale=1.0,
    depth_model_path=None,
    **Wan_Kwargs,
)

# ------ VACE-WAN causal models ------
CausalVACE_Wan_1_3B_Config: DictConfig = L(CausalVACEWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    context_scale=1.0,
    depth_model_path=None,
    **Wan_Kwargs,
)

CausalVACE_Wan_14B_Config: DictConfig = L(CausalVACEWan)(
    model_id_or_local_path="Wan-AI/Wan2.1-VACE-14B-diffusers",
    context_scale=1.0,
    depth_model_path=None,
    **Wan_Kwargs,
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
    fps=24,
    is_video2world=False,
    num_conditioning_frames=1,
    enable_logvar_linear=False,  # Enable logvar_linear for sCM-like models
)

# 14B config: override only the architecture-specific parameters
CosmosPredict2_14B_Config = copy.deepcopy(CosmosPredict2_2B_Config)
CosmosPredict2_14B_Config.model_channels = 5120
CosmosPredict2_14B_Config.num_blocks = 36
CosmosPredict2_14B_Config.num_heads = 40

# AGGRESSIVE mode: only saves flash attention outputs (~50% memory savings, moderate compute overhead)
CosmosPredict2_2B_Aggressive_Config = copy.deepcopy(CosmosPredict2_2B_Config)
CosmosPredict2_2B_Aggressive_Config.sac_config = L(SACConfig)(mode=CheckpointMode.AGGRESSIVE)

CosmosPredict2_14B_Aggressive_Config = copy.deepcopy(CosmosPredict2_14B_Config)
CosmosPredict2_14B_Aggressive_Config.sac_config = L(SACConfig)(mode=CheckpointMode.AGGRESSIVE)

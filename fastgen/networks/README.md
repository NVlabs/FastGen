# Networks

Neural network architectures for FastGen. All networks inherit from `FastGenNetwork` (or `CausalFastGenNetwork` for autoregressive video models). Each declares `latent_compression_factors`, the pixel-to-latent downsampling of its VAE, from which `pixel_shape` computes the pixel size of a latent; see [`network.py`](network.py).

## Supported Networks

| Type | Networks |
|------|----------|
| Image | [`EDM`](EDM/), [`EDM2`](EDM2/), [`DiT`](DiT/), [`SD15`](SD15/), [`SDXL`](SDXL/), [`Flux`](Flux/), [`Flux2`](Flux2/), [`QwenImage`](QwenImage/) |
| Video | [`Wan`](Wan/), [`WanI2V`](WanI2V/), [`WanMoE`](Wan/wan_moe.py), [`LingBotWorld`](WanI2V/wan_moe.py), [`VaceWan`](VaceWan/), [`CogVideoX`](CogVideoX/), [`Cosmos-Predict2`](cosmos_predict2/), [`Cosmos3`](cosmos3/) |
| Video+Audio | [`LTX2`](LTX2/), [`MiniMaxH3`](MiniMaxH3/) |


## Pretrained Models

Pretrained teacher models can be loaded using `model.pretrained_model_path=/path/to/checkpoint.pth`. If the student initialization differs from the teacher, one can load a separate checkpoint using `model.pretrained_student_net_path=/path/to/student-checkpoint.pth`.

With `model.load_student_weights=False`, neither checkpoint is loaded into the student, so it keeps the weights from its constructor. This is a random initialization only for EDM, EDM2, DiT, and Cosmos-Predict2, which do not load weights themselves; all other networks, listed under [Hugging Face Hub models](#hugging-face-hub-models-auto-download) below, start from their pretrained weights. To randomly initialize those, add `+model.net.load_pretrained=False` (the `+` adds a key the network config does not set; see [Command-Line Arguments](../configs/README.md#command-line-arguments)). Since the teacher config defaults to `model.net`, this also disables self-loading for the teacher and the DMD2 fake score unless `model.teacher` is set.

### Hugging Face Hub Models (auto-download)

The following networks load pretrained weights automatically from HuggingFace Hub (with the default setting `load_pretrained=True`):

| Network | HuggingFace Model ID |
|---------|---------------------|
| `SD15` | [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |
| `SDXL` | [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| `Flux` | [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| `Flux2` | [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev), [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B), [`black-forest-labs/FLUX.2-klein-base-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| `Wan` | [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), [`Wan-AI/Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers), [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (T2V) |
| `WanI2V` | [`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers), [`Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers), [`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) |
| `WanMoE` | [`Wan-AI/Wan2.2-T2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers), [`Wan-AI/Wan2.2-I2V-A14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) |
| `LingBotWorld` | [`robbyant/lingbot-world-base-cam`](https://huggingface.co/robbyant/lingbot-world-base-cam) |
| `VaceWan` | [`Wan-AI/Wan2.1-VACE-1.3B-diffusers`](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers), [`Wan-AI/Wan2.1-VACE-14B-diffusers`](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers) |
| `QwenImage` | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |
| `CogVideoX` | [`THUDM/CogVideoX-2b`](https://huggingface.co/THUDM/CogVideoX-2b), [`THUDM/CogVideoX-5b`](https://huggingface.co/THUDM/CogVideoX-5b) |
| `Cosmos3` | [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) |
| `LTX2` | [`Lightricks/LTX-2`](https://huggingface.co/Lightricks/LTX-2), [`diffusers/LTX-2.3-Diffusers`](https://huggingface.co/diffusers/LTX-2.3-Diffusers), [`Lightricks/LTX-2.5-Diffusers`](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) |
| `MiniMaxH3` | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) |

Set `HF_HOME` for cache location, `LOCAL_FILES_ONLY=true` for offline use.

### HTTP Checkpoints (auto-download)

The following configs and model assets reference published single-file checkpoints directly:

| Config family | Source |
|---------------|--------|
| [Self-Forcing/CausVid WAN 1.3B](https://github.com/guandeh17/Self-Forcing) | [`ode_init.pt`](https://huggingface.co/gdhe17/Self-Forcing/blob/2f8b779212da279d212c22a509b66ad6552f350e/checkpoints/ode_init.pt) |
| [VACE-Wan depth annotator](https://github.com/DepthAnything/Depth-Anything-V2) | [`depth_anything_v2_vitl.pth`](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/blob/cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9/depth_anything_v2_vitl.pth) |
| [DiT-XL/2](https://github.com/facebookresearch/DiT) | [`DiT-XL-2-256x256.pt`](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) |
| [SiT-XL/2](https://github.com/willisma/SiT) | [`SiT-XL-2-256.pt`](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=1) |
| [SiT-XL/2 + REPA](https://github.com/sihyun-yu/REPA) | [`last.pt`](https://www.dl.dropboxusercontent.com/scl/fi/cxedbs4da5ugjq5wg3zrg/last.pt?rlkey=8otgrdkno0nd89po3dpwngwcc&dl=1) |

The cache downloads each URL once and publishes it atomically in the shared
model cache under `$FASTGEN_OUTPUT_ROOT/MODEL`; later runs reuse the mirror. Local
`model.pretrained_model_path`, `model.pretrained_student_net_path`, and
`model.net.depth_model_path` overrides remain supported.

### Cosmos Predict2 Checkpoints

Cosmos Predict2 models automatically download the text encoder ([`nvidia/Cosmos-Reason1-7B`](https://huggingface.co/nvidia/Cosmos-Reason1-7B)), but require downloading the diffusion transformer with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli). Accept the model's access terms, then authenticate with `hf auth login` or set `HF_TOKEN` before downloading:
```bash
# Match FastGen's default checkpoint directory when CKPT_ROOT_DIR is unset.
CKPT_ROOT_DIR="${CKPT_ROOT_DIR:-${FASTGEN_OUTPUT_ROOT:-$PWD/FASTGEN_OUTPUT}/MODEL}"

# Download 2B model
hf download nvidia/Cosmos-Predict2.5-2B --local-dir "$CKPT_ROOT_DIR/cosmos_predict2/Cosmos-Predict2.5-2B"

# Download 14B model
hf download nvidia/Cosmos-Predict2.5-14B --local-dir "$CKPT_ROOT_DIR/cosmos_predict2/Cosmos-Predict2.5-14B"
```

The experiment configs in `fastgen/configs/experiments/CosmosPredict2/` are already configured to use these paths.

### EDM/EDM2 Checkpoints

Download and convert EDM/EDM2 checkpoints using:
```bash
# CIFAR-10 models
python scripts/download_data.py --dataset cifar10 --only-models

# ImageNet-64 models (EDM and EDM2)
python scripts/download_data.py --dataset imagenet-64 --only-models
```

The script downloads the [EDM](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/) and [EDM2](https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/) pickles, extracts and saves the state dicts, and places them in `$CKPT_ROOT_DIR` according to the following mapping:

| FastGen Path | Source Checkpoint |
|--------------|-------------------|
| `cifar10/edm-cifar10-32x32-uncond-vp.pth` | [`edm-cifar10-32x32-uncond-vp.pkl`](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl) |
| `cifar10/edm-cifar10-32x32-cond-vp.pth` | [`edm-cifar10-32x32-cond-vp.pkl`](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl) |
| `imagenet-64/edm-imagenet-64x64-cond-adm.pth` | [`edm-imagenet-64x64-cond-adm.pkl`](https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl) |
| `imagenet-64/edm2-img64-s-fid.pth` | [`edm2-img64-s-1073741-0.075.pkl`](https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl) |
| `imagenet-64/edm2-img64-m-fid.pth` | [`edm2-img64-m-2147483-0.060.pkl`](https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-m-2147483-0.060.pkl) |
| `imagenet-64/edm2-img64-l-fid.pth` | [`edm2-img64-l-1073741-0.040.pkl`](https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-l-1073741-0.040.pkl) |
| `imagenet-64/edm2-img64-xl-fid.pth` | [`edm2-img64-xl-0671088-0.040.pkl`](https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-xl-0671088-0.040.pkl) |

Note that EDM2 networks require the [`ForcedWeightNormCallback`](../callbacks/forced_weight_norm.py) callback to enforce weight normalization during training.


## Output conditioning and guidance

Every `forward` ends with `transform_output`: the conversion from the network's native prediction type
to `fwd_pred_type` and an optional `_condition_output` constraint, such as clean conditioning frames, in
high precision and rounded once to `output_dtype`: `"same"`, the default, is the state dtype, and `None` keeps
the high precision. The conversion from the native type to itself only casts, so a forward with the default
`fwd_pred_type`, `apply_output_conditioning=False` and `output_dtype=None` returns the native prediction
unrounded. Guidance runs every branch that way, composes and rescales the native predictions, and transforms the
result once, so cached branch predictions stay native and unconstrained.
`predict_with_guidance(ctx)` is an optional network-owned guidance recipe; it may add its own terms and
options in a `GuidanceConfig` subclass. The subclass's `validate` makes other recipes reject its configured
terms, and they ignore its options. Conditioning through concatenated inputs or extra context tokens
imposes no output constraint. The contracts are documented at their declarations in
[`network.py`](network.py) and at the top of [`utils/guidance.py`](../utils/guidance.py).


## Registered State

Each [`FastGenModule`](module.py) is prepared exactly once, by the code that builds it, through
`prepare_module`: it casts storage, sets train mode, and places or shards the whole registered tree.
Later loads and broadcasts copy into the existing tensors, preserving ties and optimizer references;
with meta initialization, rank 0 must hold the complete real state. Operations validate the registered
structure before changing it, and a failure on one rank is raised on all. The checks cover structure,
not values or communication, so the remaining caller contracts and the PyTorch patterns FastGen does not
support are documented at the top of [`module.py`](module.py), [`utils/module.py`](../utils/module.py),
[`utils/distributed/module.py`](../utils/distributed/module.py), and
[`utils/distributed/fsdp.py`](../utils/distributed/fsdp.py).

## Noise Schedules

Noise schedules are defined in `noise_schedule.py`. Timesteps are kept in the schedule's `t_precision`,
independently of the network's compute and storage precision; it defaults to `HIGH_PRECISION` (float64)
when the schedule is built.

| Schedule | Description | Networks |
|----------|-------------|----------|
| `edm` | EDM sigma-based (t = σ) | EDM, EDM2 |
| `sd` / `sdxl` | Stable Diffusion alphas | SD15, SDXL |
| `ddpm` | Linear betas | DiT |
| `rf` | Rectified Flow (α=1-t, σ=t) | Flux, QwenImage, WAN, Cosmos, LTX-2, MiniMax-H3 (video), DiT (from scratch, SiT, REPA) |
| `shifted_rf` | Rectified Flow under an exponential time shift (α=1-σ, σ=`time_shift(t, shift)`) | MiniMax-H3 (audio) |
| `cogvideox` | CogVideoX schedule | CogVideoX |
| `trig` | Trigonometric (α=cos, σ=sin) | sCM |

**Time distributions:** `sample_t` supports every distribution in `time_distributions.py` on every schedule.
`uniform`, `logitnormal`, `shifted` and `shifted_logitnormal` are mapped onto `[min_t, max_t]`; `lognormal`, `log_t`
and `polynomial` are truncated to it. A `window` conditions draws on a sub-range, e.g. DMD2's student-grid buckets.

**Grids:** `grid()`, used by samplers and native student grids, runs uniformly from `max_t` to 0. EDM's `grid(N)`
instead spaces its nonzero times evenly in `t**(1/rho)` from `max_t` to one spacing above `min_t`; in the reference
`edm_sampler(N)` they end at `min_t`. Times are computed in `HIGH_PRECISION`, shifted by `shift_t` for native
student grids, and rounded once to `t_precision` before any `dtype` cast.

**Prediction types:** `x0` (data), `eps` (noise), `v` (velocity), `flow` (forward-process velocity `dx/dt`,
`eps - x0` on rectified flow).

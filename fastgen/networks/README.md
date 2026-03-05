# Networks

Neural network architectures for FastGen. All networks inherit from `FastGenNetwork` (or `CausalFastGenNetwork` for autoregressive video models).

## Supported Networks

| Type | Networks |
|------|----------|
| Image | [`EDM`](EDM/), [`EDM2`](EDM2/), [`DiT`](DiT/), [`SD15`](SD15/), [`SDXL`](SDXL/), [`Flux`](Flux/), [`QwenImage`](QwenImage/) |
| Video | [`Wan`](Wan/), [`WanI2V`](WanI2V/), [`VaceWan`](VaceWan/), [`CogVideoX`](CogVideoX/), [`Cosmos-Predict2`](cosmos_predict2/) |


## Pretrained Models

Pretrained teacher models can be loaded using `model.pretrained_model_path=/path/to/checkpoint`. If the student initialization differs from the teacher, one can load a separate checkpoint using `model.pretrained_student_net_path=/path/to/student-checkpoint` or use `model.load_student_weights=False` for random initialization.

### Diffusers-based Models (auto-download)

The following networks load pretrained weights automatically from HuggingFace Hub (with the default setting `load_pretrained=True`):

| Network | HuggingFace Model ID |
|---------|---------------------|
| `SD15` | [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| `SDXL` | [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| `Flux` | [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| `Wan` | [`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), [`Wan-AI/Wan2.1-T2V-14B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers) |
| `WanI2V` | [`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers), [`Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers) |
| `VaceWan` | [`Wan-AI/Wan2.1-VACE-1.3B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-Diffusers) |
| `QwenImage` | [`Qwen/Qwen-Image`](https://huggingface.co/Qwen/Qwen-Image) |
| `CogVideoX` | [`THUDM/CogVideoX-2b`](https://huggingface.co/THUDM/CogVideoX-2b), [`THUDM/CogVideoX-5b`](https://huggingface.co/THUDM/CogVideoX-5b) |

Set `HF_HOME` for cache location, `LOCAL_FILES_ONLY=true` for offline use.

### Cosmos Checkpoints

Cosmos Predict2 models automatically download the text encoder ([`nvidia/Cosmos-Reason1-7B`](https://huggingface.co/nvidia/Cosmos-Reason1-7B)), but require downloading the diffusion transformer using the HuggingFace CLI and your [API key](https://huggingface.co/settings/tokens):
```bash
# Download 2B model
HF_TOKEN=YOUR-API-TOKEN huggingface-cli download nvidia/Cosmos-Predict2.5-2B --local-dir $CKPT_ROOT_DIR/cosmos_predict2/Cosmos-Predict2.5-2B

# Download 14B model
HF_TOKEN=YOUR-API-TOKEN huggingface-cli download nvidia/Cosmos-Predict2.5-14B --local-dir $CKPT_ROOT_DIR/cosmos_predict2/Cosmos-Predict2.5-14B
```

The experiment configs in `fastgen/configs/experiments/Cosmos-Predict2/` are already configured to use these paths.

### Self-Forcing KD checkpoints (Causal WAN 1.3B)

For causal video generation with [Self-Forcing](https://arxiv.org/abs/2506.08009) on WAN 1.3B models, download the pretrained checkpoint:

```bash
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir $CKPT_ROOT_DIR/Self-Forcing
```

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

### DiT/SiT Checkpoints

The `DiT` network supports both [DiT](https://github.com/facebookresearch/DiT) and [SiT](https://github.com/willisma/SiT) checkpoints. Download pretrained ImageNet-256 checkpoints and place them in `$CKPT_ROOT_DIR/imagenet-256/`:

```bash
mkdir -p $CKPT_ROOT_DIR/imagenet-256
# DiT-XL/2 ImageNet-256
wget -O $CKPT_ROOT_DIR/imagenet-256/DiT-XL-2-256x256.pt https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
# SiT-XL/2 ImageNet-256 (flow matching)
wget -O $CKPT_ROOT_DIR/imagenet-256/SiT-XL-2-256x256.pt "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=1"
```


## Noise Schedules

Defined in `noise_schedule.py`:

| Schedule | Description | Networks |
|----------|-------------|----------|
| `edm` | EDM sigma-based (t = σ) | EDM, EDM2 |
| `sd` / `sdxl` | Stable Diffusion alphas | SD15, SDXL |
| `rf` | Rectified Flow (α=1-t, σ=t) | Flux, QwenImage, WAN, Cosmos |
| `cogvideox` | CogVideoX schedule | CogVideoX |
| `trig` | Trigonometric (α=cos, σ=sin) | Consistency models |

**Prediction types:** `x0` (data), `eps` (noise), `v` (velocity), `flow`
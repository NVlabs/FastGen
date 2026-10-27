<h1 align="center">NVIDIA FastGen: Fast Generation from Diffusion Models</h1>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

> **Note:** The current codebase offers early access to new, experimental features. Results reported in the README files were obtained using [FastGen v0.1.0](https://github.com/NVlabs/FastGen/releases/tag/v0.1.0).

<p align="center">
<a href="https://jberner.info/">Julius Berner</a> • <a href="https://research.nvidia.com/labs/genair/author/chao-liu/">Chao Liu</a> • <a href="http://latentspace.cc/">Arash Vahdat</a>
</p>

<p align="center">
  <a href="https://youtu.be/xEKcP-SwBBY" target="_blank">
    <img src="assets/teaser.png" alt="Watch the video" width=100%/>
    <br/>
    <b>Click here for a demo video</b>
  </a>
</p>


FastGen is a PyTorch-based framework for building fast generative models using various distillation and acceleration techniques. It supports:
- large-scale training with ≥20B parameters.
- different tasks and modalities, including T2I, I2I, I2V, V2V, multimodal video+audio generation, and causal generation.
- various distillation methods, including flow maps, distribution matching, parallel decoding distillation, self-forcing, and more.

## Repository Structure

```
fastgen/
├── fastgen/
│   ├── callbacks/           # Training callbacks (EMA, profiling, etc.)
│   ├── configs/             # Configuration system
│   │   ├── experiments/     # Experiment configs
│   │   └── methods/         # Method-specific configs
│   ├── datasets/            # Dataset loaders
│   ├── methods/             # Trajectory matching, distribution matching, and fine-tuning
│   ├── networks/            # Neural network architectures
│   ├── third_party/         # Third-party dependencies
│   ├── trainer.py           # Main training loop
│   └── utils/               # Utilities (distributed, checkpointing)
├── comfyui/                 # ComfyUI custom nodes for interactive inference
├── scripts/                 # Inference and evaluation scripts
├── tests/                   # Unit tests
├── Makefile                 # Development commands (lint, format, test)
└── train.py                 # Main training entry point
```

## Setup

**Recommended:** Use the provided Docker container for a consistent environment. See [CONTRIBUTING.md](CONTRIBUTING.md) for Docker setup instructions. Otherwise, create a new [conda](https://www.anaconda.com/docs/getting-started/miniconda/install) environment with `conda create -y -n fastgen python=3.12.3 pip; conda activate fastgen`.

### Installation

```bash
git clone https://github.com/NVlabs/FastGen.git
cd FastGen
pip install -e .
```

On ARM64, PyPI only provides CPU builds of PyTorch, so install the CUDA build first:

```bash
pip install "torch>=2.10,<2.11" "torchvision>=0.25,<0.26" --index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

### Credentials (Optional)

For W&B logging, [get your API key](https://wandb.ai/settings) and save it to `credentials/wandb_api.txt` or set the `WANDB_API_KEY` environment variable.
Without either of these, W&B will prompt for your API key interactively.
For more details, including S3 storage and other environment variables, see [fastgen/configs/README.md](fastgen/configs/README.md#environment-variables).

## Quick Start

Before running the following commands, download the CIFAR-10 dataset and pretrained EDM models:

```bash
python scripts/download_data.py --dataset cifar10
```

For other datasets and models, see [fastgen/networks/README.md](fastgen/networks/README.md) and [fastgen/datasets/README.md](fastgen/datasets/README.md).

### Basic Training

```bash
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_fast.py
```

If you run out-of-memory, try a smaller batch-size, e.g., `dataloader_train.batch_size=32`, which automatically uses gradient accumulation to match the global batch-size.

**Expected Output:** See the training log for a link to the run on [wandb.ai](https://wandb.ai). Training outputs go to `$FASTGEN_OUTPUT_ROOT/{project}/{group}/{name}/`, where `name` defaults to the config file's name without `.py` (see [Run Directory](fastgen/configs/README.md#run-directory)). With default settings, outputs are organized as follows:
```
FASTGEN_OUTPUT/fastgen/cifar10/config_dmd2_fast/
├── checkpoints/    # Model checkpoints in the format {iteration:07d}.pth
│   ├── 0001000.pth
│   └── ...
├── config.yaml     # Resolved configuration for reproducibility
├── wandb_id.txt    # W&B run ID for resuming
└── ...
```

### DDP/FSDP2 Training

For multi-GPU training, use DDP:

```bash
torchrun --nproc_per_node=8 train.py \
    --config=fastgen/configs/experiments/EDM/config_dmd2_fast.py \
    - trainer.ddp=True log_config.name=test_ddp
```

For large models, use FSDP2 for model sharding by replacing `trainer.ddp=True` with `trainer.fsdp=True`.


### Inference

```bash
python scripts/inference/image_model_inference.py --config fastgen/configs/experiments/EDM/config_dmd2_fast.py \
  --classes=10 --prompt_file=scripts/inference/prompts/classes.txt --ckpt_path=FASTGEN_OUTPUT/fastgen/cifar10/config_dmd2_fast/checkpoints/0002000.pth - log_config.name=test_inference
```

For other inference modes and FID evaluations, see [scripts/README.md](scripts/README.md).
FastGen-distilled models can also be served with:

- [FlashDreams](scripts/README.md#flashdreams-example-self-forcing-wan-21-t2v) for optimized streaming inference of Self-Forcing models.
- [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/models/dmd2/) for DMD2 model inference.


### Command-Line Overrides

Override any config parameter using Hydra-style syntax (note the `-` separator):

```bash
python train.py --config=path/to/config.py - key=value nested.key=value
```

Use `+key=value` for a key the config does not define yet; see [fastgen/configs/README.md](fastgen/configs/README.md#command-line-arguments).

## Documentation

Detailed documentation is available in each component's README:

| Component | Documentation | Description |
|-----------|---------------|-------------|
| **Methods** | [fastgen/methods/README.md](fastgen/methods/README.md) | Trajectory matching, distribution matching, and fine-tuning methods |
| **Networks** | [fastgen/networks/README.md](fastgen/networks/README.md) | Network architectures (Flux, Qwen-Image, Wan, LingBot-world, Cosmos, LTX-2, etc.) and pretrained models |
| **Configs** | [fastgen/configs/README.md](fastgen/configs/README.md) | Configuration system, environment variables, and creating custom configs |
| **Datasets** | [fastgen/datasets/README.md](fastgen/datasets/README.md) | Dataset preparation and WebDataset loaders (image, video, video+audio) |
| **Callbacks** | [fastgen/callbacks/README.md](fastgen/callbacks/README.md) | Training callbacks (EMA, logging, gradient clipping, etc.) |
| **Inference** | [scripts/README.md](scripts/README.md) | Inference modes (T2I, I2I, T2V, I2V, V2V, etc.) and FID evaluation |
| **ComfyUI** | [comfyui/README.md](comfyui/README.md) | ComfyUI custom nodes for interactive teacher/student inference |
| **Third Party** | [fastgen/third_party/README.md](fastgen/third_party/README.md) | Third-party dependencies (Depth Anything V2, etc.) |

## Supported Methods

| Category | Methods |
|----------|---------|
| **Trajectory Matching** | [CM](https://arxiv.org/abs/2303.01469), [ECT](https://arxiv.org/abs/2406.14548), [sCM](https://arxiv.org/abs/2410.11081), [TCM](https://arxiv.org/abs/2410.14895), [MeanFlow](https://arxiv.org/abs/2505.13447), [AYF](https://arxiv.org/abs/2506.14603), [KD](https://arxiv.org/abs/2101.02388), [CausalKD](https://arxiv.org/abs/2412.07772), [PDD](https://arxiv.org/abs/2607.26004) |
| **Distribution Matching** | [DMD2](https://arxiv.org/abs/2405.14867), [f-Distill](https://arxiv.org/abs/2502.15681), [Phased DMD](https://arxiv.org/abs/2510.27684), [LADD](https://arxiv.org/abs/2403.12015), [AnyFlow](https://arxiv.org/abs/2605.13724), [CausVid](https://arxiv.org/abs/2412.07772), [Self-Forcing](https://arxiv.org/abs/2506.08009), [Causal-Forcing](https://arxiv.org/abs/2602.02214) |
| **Fine-Tuning** | [SFT](https://arxiv.org/abs/2006.11239), [CausalSFT (diffusion/teacher forcing)](https://arxiv.org/abs/2412.07772) |

See [fastgen/methods/README.md](fastgen/methods/README.md) for details.

## Supported Networks and Data

FastGen is designed to be **agnostic to the network and data** and you can add your own architectures and datasets (see [fastgen/networks/README.md](fastgen/networks/README.md) and [fastgen/datasets/README.md](fastgen/datasets/README.md)). For reference, we provide the following implementations:

| Data | Networks |
|------|----------|
| **Image** | EDM, EDM2, DiT, SiT, SD 1.5, SDXL, FLUX.1, FLUX.2, Qwen-Image |
| **Video** | Wan2.1, Wan2.2, Wan Vace, LingBot-world, CogVideoX, Cosmos Predict2, Cosmos 3 |
| **Video+Audio** | LTX-2, LTX-2.3, LTX-2.5, MiniMax H3 |

See [fastgen/networks/README.md](fastgen/networks/README.md) for details.
Not all combinations of methods and networks are currently supported. We provide typical use-cases in our predefined configs in [fastgen/configs/experiments](fastgen/configs/experiments).

## Adoption, Collaboration, and Usage

FastGen is used across NVIDIA and the community to build fast generative models.

### Models Distilled Using FastGen

- **[FastGen-AYF](https://huggingface.co/nvidia/FastGen-AYF)** (distilled to 4-8 steps using AYF from FastGen)
- **[Qwen-Image-Flash](https://huggingface.co/nvidia/Qwen-Image-Flash)** (distilled to 4 steps using DMD2 from FastGen, [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer), and [NVIDIA AutoModel](https://github.com/NVIDIA-NeMo/Automodel))
- **[Baseten FLUX.2](https://www.baseten.co/blog/faster-image-generation-timestep-distillation-flux2/)** (distilled to 8 steps using DMD2)
- **[Reka Real-Time Video Generation](https://reka.ai/labs/research/real-time-video-generation)** (distilled to a real-time video generator using FastGen)
- **[CorrDiff](https://developer.nvidia.com/blog/accelerating-diffusion-models-with-an-open-plug-and-play-offering/)** (distilled to 1 step using ECT)
- **[cBottle Super-Resolution](https://github.com/NVlabs/cBottle/blob/main/src/cbottle/training/distill_super_resolution.py)** (distilled to 2 steps using ECT)

### Projects Using FastGen

- **[Simplifying and Improving On-Policy Distillation for Few-Step Diffusion](https://zjwfufu.github.io/SMD-Page/)** (built upon FastGen)
- **[Parametric Trajectory Distillation for Few-Step Video Generation](https://alan-lanfeng.github.io/PTD/)** (built upon FastGen)
- **[Data-Forcing Distillation: Restoring Diversity and Fidelity in Few-Step Video Generation](https://csy2077.github.io/dfd-project-page/)** (built upon FastGen)
- **[Parallel Decoding Distillation for Image and Video Generation](https://research.nvidia.com/labs/genair/pdd/)** (built upon FastGen)
- **[Transition Matching Distillation for Fast Video Generation](https://research.nvidia.com/labs/genair/tmd/)** (built upon FastGen)
- **[Mode Seeking meets Mean Seeking for Fast Long Video Generation](https://primecai.github.io/mmm/)** (built upon FastGen)
- **[StreamingEffect: Real-Time Human-Centric Video Effect Generation](https://github.com/showlab/StreamingEffect)** (built upon FastGen)
- **[CARV: Variance Reduction for Expectations with Diffusion Teachers](https://research.nvidia.com/labs/sil/projects/CARV/)** (used FastGen for the DMD2 experiments)

### FastGen Collaborations

- **[FastH3 Omni](https://haoailab.com/blogs/fastvideo-omniref-update/)** (Minimax H3 Ref2VA distillation with [FastVideo](https://github.com/hao-ai-lab/FastVideo) and Nuva Lab)
- **[FastH3 Preview](https://haoailab.com/blogs/fasth3-preview/)** (MiniMax H3 distillation with [FastVideo](https://github.com/hao-ai-lab/FastVideo) and Nuva Lab)
- **[FlashDreams](https://github.com/NVIDIA/flashdreams)** (optimized serving for FastGen-distilled autoregressive video models)
- **[NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/Model-Optimizer)** (integration of FastGen step-distillation methods)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

We thank everyone who has helped design, build, and test FastGen!

- **Core contributors:** Julius Berner, Chao Liu
- **Other contributors:** Weili Nie, Amirmojtaba Sabour, James Lucas, David Pankratz, Sushil Koundinyan, Sihyun Yu, Willis Ma, Yilun Xu, Shengqu Cai, Xinyin Ma, Yanke Song
- **Collaborators:** Sophia Zalewski, Wei Xiong, Christian Laforte, Sajad Norouzi, Kaiwen Zheng, Miloš Hašan, Saeed Hadadan, Gene Liu, David Dynerman, Alicia Sui, Grace Lam, Pooya Jannaty, Jan Kautz, and many more.
- **Project lead:** Arash Vahdat

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details. Third-party licenses are documented in [licenses/README.md](licenses/README.md).

## Reference

```
@misc{fastgen2026,
  title={NVIDIA FastGen: Fast Generation from Diffusion Models},
  author={Berner, Julius and Liu, Chao and Vahdat, Arash},
  url={https://github.com/NVlabs/FastGen},
  year={2026},
}
```

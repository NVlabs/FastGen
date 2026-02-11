<h1 align="center">NVIDIA FastGen: Fast Generation from Diffusion Models</h1>
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVlabs/FastGen)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)

<p align="center">
<b></b> <a href="https://weilinie.github.io/">Weili Nie</a> • <a href="https://jberner.info/">Julius Berner</a> • <a href="https://research.nvidia.com/labs/genair/author/chao-liu/">Chao Liu</a> • <a href="http://latentspace.cc/">Arash Vahdat</a>
</p>

<p align="center">
  <img src="assets/teaser.png" alt="FastGen header" width=100%/>
</p>

FastGen is a PyTorch-based framework for building fast generative models using various distillation and acceleration techniques. It supports:
- large-scale training with ≥10B parameters.
- different tasks and modalities, including T2I, I2V, and V2V.
- various distillation methods, including consistency models, distribution matching distillation, self-forcing, and more.

## Repository Structure

```
fastgen/
├── fastgen/
│   ├── callbacks/           # Training callbacks (EMA, profiling, etc.)
│   ├── configs/             # Configuration system
│   │   ├── experiments/     # Experiment configs
│   │   └── methods/         # Method-specific configs
│   ├── datasets/            # Dataset loaders
│   ├── methods/             # Training methods (CM, DMD2, SFT, KD etc.)
│   ├── networks/            # Neural network architectures
│   ├── third_party/         # Third-party dependencies
│   ├── trainer.py           # Main training loop
│   └── utils/               # Utilities (distributed, checkpointing)
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
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_test.py
```

If you run out-of-memory, try a smaller batch-size, e.g., `dataloader_train.batch_size=32`, which automatically uses gradient accumulation to match the global batch-size.

**Expected Output:** See the training log for a link to the run on [wandb.ai](https://wandb.ai). Training outputs go to `$FASTGEN_OUTPUT_ROOT/{project}/{group}/{name}/`. With default settings, outputs are organized as follows:
```
FASTGEN_OUTPUT/fastgen/cifar10/debug/
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
    --config=fastgen/configs/experiments/EDM/config_dmd2_test.py \
    - trainer.ddp=True log_config.name=test_ddp
```

For large models, use FSDP2 for model sharding by replacing `trainer.ddp=True` with `trainer.fsdp=True`.


### Inference

```bash
python scripts/inference/image_model_inference.py --config fastgen/configs/experiments/EDM/config_dmd2_test.py \
  --classes=10 --prompt_file=scripts/inference/prompts/classes.txt --ckpt=FASTGEN_OUTPUT/fastgen/cifar10/debug/checkpoints/0002000.pth - log_config.name=test_inference
```

For other inferences modes and FID evaluations, see [scripts/README.md](scripts/README.md).


### Command-Line Overrides

Override any config parameter using Hydra-style syntax (note the `-` separator):

```bash
python train.py --config=path/to/config.py - key=value nested.key=value
```

## Documentation

Detailed documentation is available in each component's README:

| Component | Documentation | Description |
|-----------|---------------|-------------|
| **Methods** | [fastgen/methods/README.md](fastgen/methods/README.md) | Training methods (sCM, MeanFlow, DMD2, Self-Forcing, etc.) |
| **Networks** | [fastgen/networks/README.md](fastgen/networks/README.md) | Network architectures (EDM, SD, SDXL, Flux, WAN, CogVideoX, Cosmos) and pretrained models |
| **Configs** | [fastgen/configs/README.md](fastgen/configs/README.md) | Configuration system, environment variables, and creating custom configs |
| **Datasets** | [fastgen/datasets/README.md](fastgen/datasets/README.md) | Dataset preparation and WebDataset loaders |
| **Callbacks** | [fastgen/callbacks/README.md](fastgen/callbacks/README.md) | Training callbacks (EMA, logging, gradient clipping, etc.) |
| **Inference** | [scripts/README.md](scripts/README.md) | Inference modes (T2I, T2V, I2V, V2V, etc.) and FID evaluation |
| **Third Party** | [fastgen/third_party/README.md](fastgen/third_party/README.md) | Third-party dependencies (Depth Anything V2, etc.) |

## Supported Methods

| Category | Methods |
|----------|---------|
| **Consistency Models** | [CM](https://arxiv.org/abs/2303.01469), [sCM](https://arxiv.org/abs/2410.11081), [TCM](https://arxiv.org/abs/2410.14895), [MeanFlow](https://arxiv.org/abs/2505.13447) |
| **Distribution Matching** | [DMD2](https://arxiv.org/abs/2405.14867), [f-Distill](https://arxiv.org/abs/2502.15681), [LADD](https://arxiv.org/abs/2403.12015), [CausVid](https://arxiv.org/abs/2412.07772), [Self-Forcing](https://arxiv.org/abs/2506.08009) |
| **Fine-Tuning** | [SFT](https://arxiv.org/abs/2006.11239), [CausalSFT](https://arxiv.org/abs/2407.01392) |
| **Knowledge Distillation** | [KD](https://arxiv.org/abs/2101.02388), [CausalKD](https://arxiv.org/abs/2412.07772) |

See [fastgen/methods/README.md](fastgen/methods/README.md) for details.

## Supported Networks and Data

FastGen is designed to be **agnostic to the network and data** and you can add your own architectures and datasets (see [fastgen/networks/README.md](fastgen/networks/README.md) and [fastgen/datasets/README.md](fastgen/datasets/README.md)). For reference, we provide the following implementations:

| Data | Networks |
|------|----------|
| **Image** | EDM, EDM2, DiT, SD 1.5, SDXL, Flux |
| **Video** | WAN (T2V, I2V, VACE), CogVideoX, Cosmos Predict2 |

See [fastgen/networks/README.md](fastgen/networks/README.md) for details. 
Not all combinations of methods and networks are currently supported. We provide typical use-cases in our predefined configs in [fastgen/configs/experiments](fastgen/configs/experiments). 

**We plan to provide distilled student checkpoints for CIFAR-10 and ImageNet soon.**

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

We thank everyone who has helped design, build, and test FastGen!

- **Core contributors:** Weili Nie, Julius Berner, Chao Liu
- **Other contributors:** James Lucas, David Pankratz, Sihyun Yu, Willis Ma, Yilun Xu, Shengqu Cai, Xinyin Ma, Yanke Song
- **Collaborators:** Sophia Zalewski, Wei Xiong, Christian Laforte, Sajad Norouzi, Kaiwen Zheng, Miloš Hašan, Saeed Hadadan, Gene Liu, David Dynerman, Grace Lam, Pooya Jannaty, Jan Kautz, and many more.
- **Project lead:** Arash Vahdat

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details. Third-party licenses are documented in [licenses/README.md](licenses/README.md).

## Reference

```
@article{fastgen2026,
  title={NVIDIA FastGen: Fast Generation from Diffusion Models},
  author={Nie, Weili and Berner, Julius and Liu, Chao and Vahdat, Arash},
  url={https://github.com/NVlabs/FastGen},
  year={2026},
}
```

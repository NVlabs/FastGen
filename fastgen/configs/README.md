# Configuration System

FastGen uses a hierarchical Python-based configuration system built on [Hydra](https://hydra.cc/), [OmegaConf](https://omegaconf.readthedocs.io/), and [attrs](https://www.attrs.org/).

## Directory Structure

```
fastgen/configs/
├── experiments/         # Experiment configs (EDM, Flux, QwenImage, Wan, Cosmos3, LTX-2, etc.)
├── methods/             # Method-specific configs (DMD2, CM, KD, PDD, SFT, etc.)
├── callbacks.py         # Callback configurations (EMA, WandB, GradClip, etc.)
├── config_utils.py      # Utilities (import, override, serialize configs)
├── config.py            # Base config classes (BaseConfig, BaseModelConfig, BaseTrainerConfig)
├── data.py              # Dataset/dataloader configurations
├── discriminator.py     # Discriminator configs (for GAN-based methods)
├── instantiate.py       # LazyCall / instantiate helpers
├── net.py               # Network architecture configurations
└── opt.py               # Optimizer and scheduler configurations
```

## Config Hierarchy

1. **Base Configs** (`config.py`): Core classes defining model, trainer, data, and logging settings
2. **Method Configs** (`methods/`): Extend base configs with method-specific parameters
3. **Experiment Configs** (`experiments/`): Concrete configs for specific dataset/model combinations

## LazyCall Pattern

Deferred instantiation using `LazyCall`:

```python
from fastgen.configs.experiments.EDM.config_dmd2_fast import create_config
from fastgen.configs.instantiate import instantiate

config = create_config()  # config.model_class is a deferred LazyCall
model = instantiate(config.model_class, config=config.model)
```

## Custom Configs

A config is a Python file whose `create_config()` returns the config, usually an edited method or experiment config:

```python
# fastgen/configs/experiments/EDM/config_dmd2_fast_small.py
from fastgen.configs.experiments.EDM.config_dmd2_fast import create_config as base_create_config


def create_config():
    config = base_create_config()
    config.dataloader_train.batch_size = 32
    return config
```

`train.py` imports the file as a module, so `--config` must be a relative path below the working directory; launch from
the repository root or add the working directory to `PYTHONPATH`.

## Command-Line Arguments

```bash
python train.py --config=path/to/config.py [--log_level LEVEL] [--dryrun] - key=value
```

| Argument | Description |
|----------|-------------|
| `--config` | Path to the config file |
| `--log_level` | Log level: DEBUG, INFO (default), WARNING, ERROR |
| `--dryrun` | Print resolved config and exit without training |
| `-` | Separator before config overrides (required) |

Examples:

```bash
# Override training settings
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_fast.py - \
    trainer.max_iter=10000 \
    model.gan_loss_weight_gen=0. \
    log_config.name=my_experiment

# Debug config without training
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_fast.py --dryrun

# Verbose logging
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_fast.py --log_level DEBUG
```

Overrides use [Hydra's grammar](https://hydra.cc/docs/advanced/override_grammar/basic/): `key=value` changes a key that
the config defines, `+key=value` adds one that it does not, `++key=value` does either, and `key=null` sets `None`.
To replace a dict, see [`config_utils.py`](config_utils.py).

## Key Config Classes

| Class | Purpose |
|-------|---------|
| `BaseConfig` | Top-level: model, trainer, dataloader, logging |
| `BaseModelConfig` | Network, optimizer, precision, EMA, guidance |
| `BaseTrainerConfig` | Checkpointing, callbacks, DDP/FSDP, iterations |
| `LogConfig` | Project, group, name, wandb settings |

### Run Directory

A run writes its resolved `config.yaml`, W&B files, and by default its checkpoints to `log_config.save_path`,
`$FASTGEN_OUTPUT_ROOT/{project}/{group}/{name}`, where `name` defaults to the config file's name. With
`trainer.resume=True` (the default), a run resumes from its latest checkpoint, so give independent runs of one config
distinct names; `trainer.max_iter` counts the restored iterations too.

### Distributed Checkpoint Storage

For multi-node runs, `trainer.checkpointer.save_dir` must be on shared storage (e.g. Lustre or NFS) or S3; a relative
one (default `checkpoints`) goes in `<project>/<group>/<name>/` under `$FASTGEN_OUTPUT_ROOT`, or under `s3_container`
if `use_s3=True`. FSDP runs save sharded checkpoints and can resume only from those; to start one from a monolithic
(non-FSDP) checkpoint, use `trainer.checkpointer.pretrained_ckpt_path` and a new run directory. The module docstring of
[`checkpointer.py`](../utils/checkpointer.py) covers the formats and how they load.

## Training Data

The CIFAR-10 and ImageNet recipes read the datasets that `scripts/download_data.py` prepares. The other recipes ship
placeholder data paths that a run must override, e.g. `- dataloader_train.datatags=[WDS:/data/my_video_shards]`; see
[`data.py`](data.py).

## Large Models, Precision, and LoRA

`torchrun` distributes training only when the config sets `trainer.ddp` or `trainer.fsdp`. [`config.py`](config.py)
documents the sharding (`trainer.fsdp*`, `model.meta_init`), precision (`model.precision*`), and compilation
(`model.torch_compile_mode`) fields where it defines them, and [`lora_utils.py`](../networks/lora_utils.py) documents
LoRA training with `model.net_adapter`. Recipes with measured GPU-memory limits state them in their docstrings.

## Environment Variables

Export these in the shell or put them in a `.env` file at the repository root: `fastgen/__init__.py` loads it with python-dotenv when the package is imported, and variables already present in the environment take precedence. To start, copy the template `.env.example` to `.env`.

### Core Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FASTGEN_OUTPUT_ROOT` | Root directory for checkpoints, logs, and outputs, resolved to an absolute path on read. | `$(pwd)/FASTGEN_OUTPUT` |
| `DATA_ROOT_DIR` | Root directory for datasets | `$FASTGEN_OUTPUT_ROOT/DATA` |
| `CKPT_ROOT_DIR` | Root for local pretrained checkpoints; `$FASTGEN_OUTPUT_ROOT/MODEL` caches mirrored S3 and HTTP model files | `$FASTGEN_OUTPUT_ROOT/MODEL` |
| `HF_HOME` | HuggingFace cache directory. FastGen passes it as `cache_dir`, so model folders (`models--*`) land directly below it rather than under `hub/`; pre-download with `hf download --cache-dir "$HF_HOME" ...` | `$FASTGEN_OUTPUT_ROOT/.cache` |
| `LOCAL_FILES_ONLY` | Use only local files, skip downloads from HuggingFace | `false` |
| `OMP_NUM_THREADS` | Intra-op CPU threads per process. Use `1`: the main process is GPU-bound and extra OpenMP workers only compete with the DataLoader workers; `torchrun` sets it for `nproc_per_node > 1`, and `.env.example` sets it for single-process launches. | (PyTorch: all visible cores) |
| `PYTORCH_ALLOC_CONF`, `PYTORCH_CUDA_ALLOC_CONF` | PyTorch's CUDA-allocator settings; `.env.example` sets both to `expandable_segments:True` against fragmentation | (PyTorch default) |
| `NCCL_TIMEOUT` | Timeout in seconds of distributed collectives; raise it if rank 0 loads large models for longer | `3600` |
| `WANDB_API_KEY` | W&B API key for persistent logging ([get yours here](https://wandb.ai/settings)) | (none, W&B will prompt) |
| `WANDB_UPLOAD_CODE` | Upload the code in the working directory to the W&B run | `false` |
| `HF_TOKEN` | Hugging Face access token for gated models (e.g. FLUX.1-dev, Cosmos) | (none) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL`, `S3_ENDPOINT_URL` | S3 credentials for data and checkpoint storage | (none) |
| `REMOTE_DATA_ROOT_DIR` | Optional S3 or HTTP(S) mirror of `DATA_ROOT_DIR` for missing CIFAR-10 and ImageNet ZIPs | (none) |
| `VLM_BASE_URL` | OpenAI-compatible endpoint used by [`VLMLogger`](../callbacks/README.md#log-modules) | (none; required when constructing `VLMLogger` unless `base_url` is passed directly) |
| `OPENAI_API_KEY` | Bearer token for the VLM log module (read natively by the OpenAI SDK) | (none; missing credentials disable scoring with a warning on the first VLM scoring step) |

### Loading Credentials from Files

As an alternative to setting environment variables directly, credentials can be loaded automatically from files in the folder given by the config's `credentials_dir` (default `./credentials/`; override it with, e.g., `- credentials_dir=/path/to/credentials`). A non-empty file overrides the matching environment variable, and a missing or empty file leaves it untouched. Token files must hold the token on a single line.

| File in `credentials_dir` | Environment Variables Set |
|------|---------------------------|
| `wandb_api.txt` | `WANDB_API_KEY` (plain text file containing your API key) |
| `hf_token.txt` | `HF_TOKEN` (plain text Hugging Face access token) |
| `vlm_api_token.txt` | `OPENAI_API_KEY` (plain text bearer token for the VLM log module) |
| `s3.json` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL`, and `S3_ENDPOINT_URL` (JSON format below) |

**Format for `s3.json`:**

```json
{
    "aws_access_key_id": "<your_access_key>",
    "aws_secret_access_key": "<your_secret_key>",
    "aws_session_token": "<optional_session_token>",
    "region_name": "<region>",
    "endpoint_url": "<s3_endpoint_url>"
}
```

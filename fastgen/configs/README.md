# Configuration System

FastGen uses a hierarchical Python-based configuration system built on [Hydra](https://hydra.cc/), [OmegaConf](https://omegaconf.readthedocs.io/), and [attrs](https://www.attrs.org/).

## Directory Structure

```
fastgen/configs/
├── experiments/         # Experiment configs (EDM, Flux, QwenImage, Wan, etc.)
├── methods/             # Method-specific configs (DMD2, CM, KD, SFT, etc.)
├── callbacks.py         # Callback configurations (EMA, WandB, GradClip, etc.)
├── config_utils.py      # Utilities (import, override, serialize configs)
├── config.py            # Base config classes (BaseConfig, BaseModelConfig, BaseTrainerConfig)
├── data.py              # Dataset/dataloader configurations
├── discriminator.py     # Discriminator configs (for GAN-based methods)
├── net.py               # Network architecture configurations
├── opt.py               # Optimizer and scheduler configurations
```

## Config Hierarchy

1. **Base Configs** (`config.py`): Core classes defining model, trainer, data, and logging settings
2. **Method Configs** (`methods/`): Extend base configs with method-specific parameters
3. **Experiment Configs** (`experiments/`): Concrete configs for specific dataset/model combinations

## LazyCall Pattern

Deferred instantiation using `LazyCall`:

```python
from fastgen.utils import LazyCall as L
from fastgen.methods import DMD2Model

model_class = L(DMD2Model)(config=None)  # Config dict with _target_
model = instantiate(model_class)         # Instantiate later
```

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
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_test.py - \
    trainer.max_iter=10000 \
    model.gan_loss_weight_gen=0. \
    log_config.name=my_experiment

# Debug config without training
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_test.py --dryrun

# Verbose logging
python train.py --config=fastgen/configs/experiments/EDM/config_dmd2_test.py --log_level DEBUG
```

## Key Config Classes

| Class | Purpose |
|-------|---------|
| `BaseConfig` | Top-level: model, trainer, dataloader, logging |
| `BaseModelConfig` | Network, optimizer, precision, EMA, guidance |
| `BaseTrainerConfig` | Checkpointing, callbacks, DDP/FSDP, iterations |
| `LogConfig` | Project, group, name, wandb settings |

## Environment Variables

### Core Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FASTGEN_OUTPUT_ROOT` | Root directory for checkpoints, logs, and outputs | `FASTGEN_OUTPUT` |
| `DATA_ROOT_DIR` | Root directory for datasets | `$FASTGEN_OUTPUT_ROOT/DATA` |
| `CKPT_ROOT_DIR` | Root directory for pretrained checkpoints | `$FASTGEN_OUTPUT_ROOT/MODEL` |
| `HF_HOME` | HuggingFace cache directory | `$FASTGEN_OUTPUT_ROOT/.cache` |
| `LOCAL_FILES_ONLY` | Use only local files, skip downloads from HuggingFace | `false` |
| `WANDB_API_KEY` | W&B API key for persistent logging ([get yours here](https://wandb.ai/settings)) | (none, W&B will prompt) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT_URL` | S3 credentials for data and checkpoint storage | (none) |

### Loading Credentials from Files

As an alternative to setting environment variables directly, credentials can be loaded automatically from files in the `./credentials/` directory:

| File | Environment Variables Set |
|------|---------------------------|
| `./credentials/wandb_api.txt` | `WANDB_API_KEY` (plain text file containing your API key) |
| `./credentials/s3.json` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_ENDPOINT_URL` (JSON format below) |

**Format for `s3.json`:**

```json
{
    "aws_access_key_id": "<your_access_key>",
    "aws_secret_access_key": "<your_secret_key>",
    "region_name": "<region>",
    "endpoint_url": "<s3_endpoint_url>"
}
```
# Callbacks

Training callbacks for FastGen. All callbacks inherit from `Callback` and hook into the training lifecycle.

## Available Callbacks

| Callback | Description |
|----------|-------------|
| [`EMACallback`](ema.py) | Exponential Moving Average updates (constant, power, or halflife schedules) |
| [`GradClipCallback`](grad_clip.py) | Gradient norm clipping with NaN/Inf handling |
| [`WandbCallback`](wandb.py) | Weights & Biases logging (metrics, images, videos, audio); accepts pluggable [log modules](#log-modules) for extra per-step processing |
| [`GPUStatsCallback`](gpu_stats.py) | GPU memory and utilization logging |
| [`MemTrackerCallback`](gpu_mem_profiler.py) | GPU memory profiling with HTML visualizations |
| [`TrainProfilerCallback`](train_profiler.py) | Training speed and timing breakdown |
| [`ParamCountCallback`](param_count.py) | Parameter count logging |
| [`ForcedWeightNormCallback`](forced_weight_norm.py) | Forced weight normalization (EDM2) |
| [`CTScheduleCallback`](ct_schedule.py) | Consistency training curriculum schedule |
| [`CarryCallback`](carry.py) | Atomically carries a method-owned state dictionary across training iterations; validation carry is opt-in |
| [`RankConsistencyCallback`](rank_consistency.py) | Debug aid: checks low-memory XOR fingerprints of buffers and optional replicated parameters across ranks every N iterations |

## Usage

Configure callbacks as a dictionary in the config:

```python
from fastgen.configs.callbacks import WANDB_CALLBACK, GradClip_CALLBACK, EMA_CALLBACK
from omegaconf import DictConfig

config.trainer.callbacks = DictConfig({
    **GradClip_CALLBACK,
    **EMA_CALLBACK,
    **WANDB_CALLBACK,
})
```

Callbacks run in dictionary order. With `model.use_ema` set, W&B logs samples from the first EMA network, so place EMA callbacks before `wandb` (part of `BASE_CALLBACKS`); otherwise training samples lag by one EMA update.

Predefined configs are in `fastgen/configs/callbacks.py`.

## Lifecycle Hooks

Callbacks can override these hooks:

| Phase | Hooks |
|-------|-------|
| Setup | `on_app_begin`, `on_model_init_start/end`, `on_optimizer_init_start/end`, `on_load_checkpoint_start/end`, `on_dataloader_init_start/end` |
| Training | `on_train_begin/end`, `on_training_step_begin/end`, `on_training_accum_step_begin/end`, `on_backward_begin`, `on_optimizer_step_begin` |
| Validation | `on_validation_begin/end`, `on_validation_step_begin/end` |
| Checkpointing | `on_save_checkpoint_start/success/end`, `state_dict`, `load_state_dict` |
| Cleanup | `on_app_end` |

## Custom Callbacks

```python
from fastgen.callbacks.callback import Callback

class MyCallback(Callback):
    def on_training_step_end(self, model, data_batch, output_batch, loss_dict, iteration=0):
        if iteration % self.config.trainer.logging_iter == 0:
            print(f"Step {iteration}: loss = {loss_dict['total_loss'].item():.4f}")
```

Callbacks have access to `self.config` and `self.trainer` after initialization.

## Log Modules

`WandbCallback` accepts a dict of pluggable [`LogModule`](wandb.py) instances via its `log_modules` arg. Rank 0 runs each module on the cross-rank-gathered `List[MediaEntry]` per step and forwards any returned `loss_dict` through scalar aggregation. Subclass `LogModule` and override `process` to add your own — see the class docstring for the hook signature.

[`VLMLogger`](vlm_logger.py) scores student media with a vision-language model judge served by any OpenAI-compatible endpoint; set `VLM_BASE_URL` and `OPENAI_API_KEY` as described in [Environment Variables](../configs/README.md#environment-variables). [`VipeLogger`](vipe_logger.py) logs camera trajectories and action overlays for models trained on [camera-pose batches](../datasets/README.md#camera-conditioned-video-loader). See the module docstrings for media modes, metrics, and options.

Enable them with the presets in `fastgen/configs/callbacks.py` (`VLM_IMG_CALLBACK`, `VLM_VID_CALLBACK`, `VIPE_CALLBACK`), e.g. `config.trainer.callbacks.update(VLM_VID_CALLBACK)`.

# FastGen ComfyUI Nodes

Custom nodes for running FastGen teacher and distilled-student image and video generation inside [ComfyUI](https://github.com/Comfy-Org/ComfyUI).

## Setup

1. **Install ComfyUI** following the [official instructions](https://github.com/Comfy-Org/ComfyUI#installing).

2. **Set `FASTGEN_REPO` and symlink**:

   ```bash
   export FASTGEN_REPO=/path/to/fastgen
   ln -s $FASTGEN_REPO/comfyui /path/to/ComfyUI/custom_nodes/comfyui_fastgen
   ```

   > **Note:** `FASTGEN_REPO` must be set every time you start ComfyUI (add it to your shell profile or launch script).

3. **Install FastGen's dependencies** with ComfyUI's Python (3.10–3.12; this installs PyTorch 2.10), unless ComfyUI
   runs in a FastGen environment such as the [Docker container](../CONTRIBUTING.md#development-setup):

   ```bash
   python -m pip install -e "$FASTGEN_REPO"
   ```

4. **Start ComfyUI**:

   ```bash
   python main.py --listen 127.0.0.1
   ```

   > **Security Note:** `--listen 127.0.0.1` binds to localhost only. `--listen 0.0.0.0` exposes ComfyUI on **all** network interfaces — only use it when you have proper firewall rules and authentication in place.

## Nodes

| Node | Description |
|---|---|
| **FastGen Config Loader** | Load an experiment config (auto-discovers configs under `fastgen/configs/experiments/`). Accepts optional config overrides and environment variables as wired string inputs. |
| **FastGen Model Loader** | Load teacher and/or student networks from a config. Optionally pick a `.pth` checkpoint from ComfyUI's `input/fastgen_checkpoints/` folder or type its absolute path. For DCP, copy or symlink the complete containing checkpoint directory under that folder so each `.pth` marker remains beside its `<name>.<key>_model` shard directories. Outputs a `FASTGEN_MODEL` bundle used by all other nodes. |
| **FastGen Prompt Expand** | Expand a prompt with a Qwen model (text-only `Qwen2.5_*` or vision-language `QwenVL2.5_*`). VL models require a reference `IMAGE`; `None` is a passthrough. Wire its `prompt` output into FastGen Preprocess. |
| **FastGen Camera Loader** | Load raw camera poses and intrinsics from `poses.npy` / `intrinsics.npy`, or extract them from a source video with VIPE. The data is encoded only after it reaches FastGen Preprocess. |
| **FastGen Depth Latent Loader** | Load a precomputed raw depth latent for optional V2V conditioning. |
| **FastGen Preprocess** | Turn the prompt, negative prompt, and any raw `IMAGE`, native ComfyUI `VIDEO`, camera, or depth input into the workflow's single `FASTGEN_CONDITIONING` object. It uses the same task preprocessing path as training. Autoregressive `num_segments` / `overlap_frames` are set here so source-video conditioning and student sampling share one generation length; values other than 1 / 0 extrapolate, which needs a CausVid or Self-Forcing config. |
| **FastGen Noise Generator** | Generate deterministic noise from a seed for reproducible sampling. |
| **FastGen Teacher Sample** | Run multi-step teacher sampling. `sample_steps=0` uses the model/scheduler default. FPS comes from the configured network. Accepts an optional `config_overrides` string for temporary config changes during sampling. |
| **FastGen Student Sample** | Run few-step distilled student sampling from finalized conditioning. Also accepts `config_overrides`; autoregressive settings come from FastGen Preprocess. |
| **FastGen Decode & Preview** | Decode video/image/audio latents **once** and: show an inline preview (animated MP4/GIF for video, PNG for images, an audio player for audio), output frames as a ComfyUI `IMAGE` and audio as a ComfyUI `AUDIO`, and save the result to ComfyUI's `output/` directory under `filename_prefix` (subdirs like `fastgen/run1` allowed). Returns the saved path. Audio-video models save an MP4 with sound and audio-only models a WAV; an output the model does not produce (`image` for audio-only, `audio` without audio) is `None`. |

Use ComfyUI's string, **Load Image**, and **Load Video** nodes as raw inputs.
**FastGen Preprocess** is the only node that produces `FASTGEN_CONDITIONING`,
and either sampler consumes that finalized object directly. For vision-language
prompt expansion, connect the same image to both **FastGen Prompt Expand** and
**FastGen Preprocess**; the expanded string feeds the preprocess prompt input.

## Config Overrides

The **Config Loader**, **Teacher Sample**, and **Student Sample** nodes accept a `config_overrides` string input of
dotted `key=value` lines. The Config Loader applies them **permanently** with Hydra, like the command line of `train.py`
(see [Command-Line Arguments](../fastgen/configs/README.md#command-line-arguments)). The sampling nodes apply them
**for one run only**, setting each value as a Python literal, so a quoted dict replaces the whole dict. For example,
on a sampling node:

```
model.guidance_cfg.cfg.scale=5.0
model.student_grid_cfg.grid_type=explicit
model.student_grid_cfg.grid_kwargs='{"timesteps": [0.999, 0.937, 0.833, 0.624, 0.0]}'
```

## Example Workflows

Saved workflows are in the `workflows/` subdirectory. Load them via ComfyUI's **Load** button or drag-and-drop the JSON file onto the canvas.

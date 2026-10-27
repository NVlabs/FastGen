## WAN 2.1 Prompt Expansion

Expands short prompts into detailed descriptions optimized for WAN video generation using Qwen language models.

### Usage in Inference

Enable prompt expansion via command-line arguments in `scripts/inference/video_model_inference.py`:

```bash
PYTHONPATH=$(pwd) FASTGEN_OUTPUT_ROOT=/absolute/path/to/fastgen-output \
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanT2V/config_sft.py \
    --prompt_expand_model Qwen2.5_14B \
    --prompt_expand_model_seed 42 \
    - log_config.name=wan_prompt_expand
```

**Arguments:**
- `--prompt_expand_model`: Qwen model to use (`Qwen2.5_3B`,
  `Qwen2.5_7B`, `Qwen2.5_14B`, `QwenVL2.5_3B`, or `QwenVL2.5_7B`)
- `--prompt_expand_model_seed`: Random seed for reproducible expansion (default: 0)

When enabled, prompts are expanded on rank 0 and broadcast to all ranks. The expanded prompt includes subject details, visual style, camera angles, and motion attributes. Text-only `Qwen2.5_*` models expand each prompt on its own. Vision-language `QwenVL2.5_*` models also read the prompt's image from `--input_image_file`, which they require. The FastGen ComfyUI Prompt Expand node supports both model types.

### Source

Code from [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) at commit [7c81b2f](https://github.com/Wan-Video/Wan2.1/commit/7c81b2f27defa56c7e627a4b6717c8f2292eee58). We made the following changes to the original code:

1. Extract prompt expansion from `wan/utils` and remove the dashscope dependency.
2. Keep the model on its device across prompts instead of moving it back to
   the CPU after every expansion. FastGen frees the expander once all prompts
   are expanded.
3. Add `__init__.py` and headers.

### License

See [`licenses/Wan/LICENSE`](../../../licenses/Wan/LICENSE) for details.

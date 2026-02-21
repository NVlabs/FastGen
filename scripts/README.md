# FastGen Inference

Generate images and videos using pretrained or distilled models.

| Script | Modality | Modes |
|--------|----------|-------|
| [`image_model_inference.py`](inference/image_model_inference.py) | Image | Unconditional, class-conditional, T2I |
| [`video_model_inference.py`](inference/video_model_inference.py) | Video | T2V, I2V, V2V, Video2World |

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--do_student_sampling` | Run distilled student (few-step) |
| `--do_teacher_sampling` | Run teacher (multi-step) |
| `--ckpt_path` | Path to distilled checkpoint (drop suffix `.pth` for FSDP checkpoints) |
| `--num_steps` | Sampling steps for teacher |
| `--classes N` | Class-conditional with N classes |
| `--unconditional` | Unconditional generation |
| `--input_image_file` | Input images for I2V |
| `--source_video_file` | Source videos for V2V |
| `--fps` | Output video frame rate |
| `model.guidance_scale` | CFG scale (config override) |
| `trainer.seed` | Random seed for reproducibility (config override) |


## Example commands

### Unconditional
```bash
python scripts/inference/image_model_inference.py \
    --config fastgen/configs/experiments/EDM/config_sft_edm_cifar10.py \
    --do_student_sampling False --unconditional --num_samples 16 --num_steps 18
```

### Class-Conditional
```bash
python scripts/inference/image_model_inference.py \
    --config fastgen/configs/experiments/DiT/config_sft_sit_xl.py \
    --do_student_sampling False --classes 1000 --num_steps 50 \
    --prompt_file scripts/inference/prompts/classes.txt \
    - model.guidance_scale=4.0
```

### Text-to-Image (T2I)
```bash
python scripts/inference/image_model_inference.py \
    --config fastgen/configs/experiments/Flux/config_sft.py \
    --do_student_sampling False --num_steps 50 \
    - model.guidance_scale=3.5
```

### Text-to-Video (T2V)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanT2V/config_dmd2.py \
    --do_student_sampling False --num_steps 50 --fps 16 \
    --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \
    - model.guidance_scale=5.0
```

### Image-to-Video (I2V)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanI2V/config_dmd2_wan22_5b.py \
    --do_student_sampling False --num_steps 50 --fps 16 \
    --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \
    --input_image_file scripts/inference/prompts/source_image_paths.txt \
    - model.guidance_scale=5.0
```

### Video-to-Video (V2V)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanV2V/config_sft.py \
    --do_student_sampling False --num_steps 50 --fps 16 \
    --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \
    --source_video_file scripts/inference/prompts/source_video_paths.txt \
    - model.guidance_scale=5.0
```

### Video2World (Cosmos)
```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/CosmosPredict2/config_sft.py \
    --do_student_sampling False --num_steps 35 --fps 24 \
    --neg_prompt_file scripts/inference/prompts/negative_prompt_cosmos.txt \
    --input_image_file scripts/inference/prompts/source_image_paths.txt \
    --num_conditioning_frames 1 \
    - model.guidance_scale=5.0 model.net.is_video2world=True model.input_shape="[16, 24, 88, 160]"
```

### Causal / Autoregressive
Use causal configs (e.g., `config_sft_causal_wan22_5b.py`) for autoregressive generation.

```bash
python scripts/inference/video_model_inference.py \
    --config fastgen/configs/experiments/WanI2V/config_sft_causal_wan22_5b.py \
    --do_student_sampling False --num_steps 50 --fps 16 \
    --neg_prompt_file scripts/inference/prompts/negative_prompt.txt \
    --input_image_file scripts/inference/prompts/source_image_paths.txt \
    - model.guidance_scale=5.0
```

For generating longer videos via extrapolation:
- `--num_segments N`: Generate N consecutive video segments autoregressively (default: 1)
- `--overlap_frames K`: Overlap K latent frames between segments for temporal consistency (default: 0)

---

## FID Evaluation

Compute Fréchet Inception Distance (FID) for image models using [`fid/compute_fid_from_ckpts.py`](fid/compute_fid_from_ckpts.py).

### Usage

```bash
torchrun --nproc_per_node=8 scripts/fid/compute_fid_from_ckpts.py \
    --config fastgen/configs/experiments/EDM/config_dmd2_cifar10.py \
    - trainer.ddp=True
```

This script:
1. Loads checkpoints from `trainer.checkpointer.save_dir`
2. Generates `eval.num_samples` images using student sampling
3. Computes FID against reference statistics
4. Saves results to `{save_path}/{eval.samples_dir}/fid.json`


### Config Options

| Parameter | Description |
|-----------|-------------|
| `eval.num_samples` | Number of samples to generate (default: 50000) |
| `eval.min_ckpt` | Minimum checkpoint iteration to evaluate |
| `eval.max_ckpt` | Maximum checkpoint iteration to evaluate |
| `eval.samples_dir` | Subdirectory name for generated samples |
| `eval.save_images` | Save visualization grid instead of computing FID |

### Reference Statistics

FID reference statistics can be computed with the following commands:
```bash
# Prepare FID refs for CIFAR-10 (uses EDM's fid.py, outputs .npz)
python scripts/download_data.py --dataset cifar10 --compute-fid-refs
# Prepare all FID refs (uses EDM's fid.py for EDM format, EDM2's calculate_metrics.py for EDM2 format)
python scripts/download_data.py --dataset all --imagenet-source /path/to/imagenet --compute-fid-refs
```
This computes the FID reference statistics according to the [EDM](https://github.com/NVlabs/edm) and [EDM2](https://github.com/NVlabs/edm2) repositories and stores them in `$DATA_ROOT_DIR/fid-refs/`:

| Dataset | Reference File | Format |
|---------|----------------|--------|
| CIFAR-10 | `fid-refs/cifar10-32x32.npz` | EDM (`.npz`) |
| ImageNet-64 (EDM) | `fid-refs/imagenet-64x64.npz` | EDM (`.npz`) |
| ImageNet-64 (EDM2) | `fid-refs/imagenet-64x64-edmv2.pkl` | EDM2 (`.pkl`) |
| ImageNet-256 | `fid-refs/imagenet_256.pkl` | EDM2 (`.pkl`) |

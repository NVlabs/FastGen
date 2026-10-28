# Datasets

Data loaders for FastGen training, supporting class-conditional image datasets and scalable WebDataset-based loaders for large-scale image, video, and video+audio training.

## Overview

| File | Description | Key Classes |
|------|-------------|-------------|
| [class_cond_dataloader.py](class_cond_dataloader.py) | [Class-conditional image loaders](#class-conditional-datasets) | `ImageLoader` |
| [wds_dataloaders.py](wds_dataloaders.py) | [WebDataset loaders for images/videos/audio/camera](#webdataset-loaders) | `WDSLoader`, `ImageWDSLoader`, `VideoWDSLoader`, `AudioVideoWDSLoader`, `CameraVideoWDSLoader` |
| [../configs/data.py](../configs/data.py) | [Generic loader configs](#generic-loaders) | `ImageLoaderConfig`, `ImageEditLoaderConfig`, `PromptLoaderConfig`, `VideoLoaderConfig`, `VideoAudioLoaderConfig`, `CameraVideoLoaderConfig`, `ImageLatentLoaderConfig`, `VideoLatentLoaderConfig`, `PairLoaderConfig`, `PathLoaderConfig` |

---

## Class-Conditional Datasets

In the following, we provide commands to prepare different versions of CIFAR-10 and ImageNet. For FID reference statistics computations, see [scripts/README.md](../../scripts/README.md#reference-statistics).

### CIFAR-10

Preprocess the data using:
```bash
python scripts/download_data.py --dataset cifar10 --only-data
```
This prepares the dataset as described in the [EDM repo](https://github.com/NVlabs/edm/?tab=readme-ov-file#preparing-datasets) and places it at `$DATA_ROOT_DIR/cifar10/cifar10-32x32.zip`, compatible with `CIFAR10_Loader_Config`.

### ImageNet-64

ImageNet datasets require downloading ImageNet from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge). For instance, after installing `pip install kaggle` and retrieving your [API token](https://www.kaggle.com/settings), you can download it using:
```bash
KAGGLE_API_TOKEN=YOUR-API-TOKEN kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip -d /path/to/imagenet
``` 

Then, preprocess the data using:
```bash
python scripts/download_data.py --dataset imagenet-64 --imagenet-source /path/to/imagenet --only-data
```
The `--imagenet-source` flag points to the unzipped directory containing `ILSVRC/Data/CLS-LOC/train`. This prepares the datasets as described in the [EDM](https://github.com/NVlabs/edm/?tab=readme-ov-file#preparing-datasets) and [EDM2](https://github.com/NVlabs/edm2/tree/main) (with `--resolution=64x64` and skipping the VAE encoder) repos and places them at `$DATA_ROOT_DIR/imagenet-64/imagenet-64x64.zip` and `$DATA_ROOT_DIR/imagenet-64/imagenet-64x64-edmv2.zip`, compatible with the `ImageNet64_Loader_Config` and `ImageNet64_EDMV2_Loader_Config` configs.


### ImageNet-256

Preprocess the data using:
```bash
python scripts/download_data.py --dataset imagenet-256 --imagenet-source /path/to/imagenet --only-data
```
This creates the latent dataset according to the DiT/SiT preprocessing (mean and std from SD VAE, input normalized to [-1, 1]) and places it at `$DATA_ROOT_DIR/imagenet-256/imagenet_256_sd.zip`, compatible with the `ImageNet256_Loader_Config` config.

---

## WebDataset Loaders

FastGen provides WebDataset loaders for scalable training on large image and video datasets, supporting both local storage and S3 paths.

### Preparing Your Data

WebDataset stores data as tar archives (shards) containing files grouped by a common key:

```
00000.tar
├── sample_000000.mp4    # Video/image file
├── sample_000000.txt    # Caption
├── sample_000000.json   # Optional metadata
├── sample_000001.mp4
├── sample_000001.txt
└── ...
```

Create shards using the [webdataset](https://github.com/webdataset/webdataset) library:

```python
import webdataset as wds

with wds.ShardWriter("/path/to/video_shards/%05d.tar", maxcount=1000) as sink:
    for idx, (video_path, caption) in enumerate(your_dataset):
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        sink.write({
            "__key__": f"sample_{idx:06d}",
            "mp4": video_bytes,
            "txt": caption.encode("utf-8"),
        })
```


### Generic Loaders

FastGen comes with predefined loaders for common WDS layouts. In the following, we show how to adapt them to your specific dataset.
In each loader config, `key_map` links an output key in the batch to a file extension in the shard. Decoded media and prompt text use the explicit raw keys: `key_map={"real_raw": "mp4", "condition_raw": "txt"}` loads `{sample_id}.mp4` and `{sample_id}.txt` into `real_raw` and `condition_raw`. Precomputed model-space latents and embeddings instead use the plain `real` and `condition` keys, as in the next example.

#### WDSLoader

Base loader for precomputed latents and embeddings. Supports `.npy`, `.npz`, `.pth`, `.json`, and text files.

```python
from fastgen.datasets.wds_dataloaders import WDSLoader
from fastgen.configs.instantiate import LazyCall as L

MyLoader = L(WDSLoader)(
    datatags=["WDS:/path/to/latent_shards"],  # Prefix with "WDS:" (supports S3 via "WDS:s3://bucket/path/to/shards")
    batch_size=32,
    key_map={"real": "latent.pth", "condition": "txt_emb.pth"},
    # Constant loaded once; relative to the datatag's parent: /path/to/neg_prompt_emb.npy
    files_map={"neg_condition": "neg_prompt_emb.npy"},
)
```

#### ImageWDSLoader

For raw images (jpg, png, etc.) with automatic resize, center crop, and normalization.

Use raw-key names for every decoded image role. For example, an image-edit loader maps the edited
target to `real_raw` and the source/reference image to `condition_image_raw`. A top-level
`condition_image` is reserved for a precomputed model-space reference latent; the network packages
that latent under the same name inside its completed rich `condition` mapping.

```python
from fastgen.configs.data import ImageLoaderConfig

MyImageLoader = ImageLoaderConfig.copy()
MyImageLoader.datatags = ["WDS:/path/to/image_shards"]
MyImageLoader.input_res = 512  # Target resolution
```

#### ImageLatentLoaderConfig

For precomputed image latents and text embeddings (faster than encoding on-the-fly):

```python
from fastgen.configs.data import ImageLatentLoaderConfig

MyImageLatentLoader = ImageLatentLoaderConfig.copy()
MyImageLatentLoader.datatags = ["WDS:/path/to/image_latent_shards"]
MyImageLatentLoader.files_map = {"neg_condition": "/path/to/neg_prompt_emb.npy"}
```

Expected shard contents:
- `latent.pth` - Precomputed image latent
- `txt_emb.pth` - Precomputed text embedding

#### PromptLoaderConfig

For recipes whose student inputs come from its own trajectories (`backward_simulation`) and therefore read only
prompts:

```python
from fastgen.configs.data import PromptLoaderConfig

MyPromptLoader = PromptLoaderConfig.copy()
MyPromptLoader.datatags = ["WDS:/path/to/prompt_shards"]
```

Expected shard contents:
- `txt` - Prompt (the negative prompt is the empty string by default)

#### VideoWDSLoader

For raw videos (mp4, avi, etc.) with frame extraction and transforms.

```python
from fastgen.configs.data import VideoLoaderConfig

MyVideoLoader = VideoLoaderConfig.copy()
MyVideoLoader.datatags = ["WDS:/path/to/video_shards"]
MyVideoLoader.batch_size = 2
MyVideoLoader.sequence_length = 81
MyVideoLoader.img_size = (832, 480)
MyVideoLoader.target_fps = 16  # The network's frame rate (config.model.net.fps) unless the shards already run at it
```

#### VideoAudioLoaderConfig

For raw videos with embedded audio tracks (e.g., for multimodal video+audio models like LTX-2). Extends `VideoWDSLoader` to additionally decode the audio waveform from the video container.

```python
from fastgen.configs.data import VideoAudioLoaderConfig

MyVideoAudioLoader = VideoAudioLoaderConfig.copy()
MyVideoAudioLoader.datatags = ["WDS:/path/to/video_shards_with_audio"]
MyVideoAudioLoader.batch_size = 1
MyVideoAudioLoader.sequence_length = 81
MyVideoAudioLoader.img_size = (832, 480)
MyVideoAudioLoader.target_fps = 24  # The network's frame rate (config.model.net.fps); the audio slice tracks it
```

For each video key `key` in `video_keys` (default: `"real_raw"`), the loader produces a modality dict at `{key}` with the following entries:
- `{key}["video"]`: Video tensor `(C, T, H, W)` normalized to `[-1, 1]`
- `{key}["audio"]`: Stereo audio waveform `Tensor[A_C, num_samples]`, resampled to `target_sample_rate` when provided (otherwise kept at the clip's native rate) and trimmed/padded to `num_video_frames / fps` seconds. Omitted for clips whose container has no audio stream.

#### Camera-Conditioned Video Loader

For raw videos paired with pre-computed camera poses and intrinsics (e.g., from [VIPE](https://github.com/nv-tlabs/vipe)).

```python
from fastgen.configs.data import CameraVideoLoaderConfig

MyCamLoader = CameraVideoLoaderConfig.copy()
MyCamLoader.datatags = ["WDS:/path/to/videos_with_vipe"]
MyCamLoader.batch_size = 1
MyCamLoader.sequence_length = 81
MyCamLoader.img_size = (832, 480)
```

Expected shard contents:
- `mp4` - Video
- `txt` - Caption
- `poses.npy` - Camera-to-world poses, `[T, 4, 4]` (OpenCV convention)
- `intrinsics.npy` - Per-frame intrinsics, `[T, 4]` as `[fx, fy, cx, cy]`

**Preparing camera poses from a video** (requires [VIPE](https://github.com/nv-tlabs/vipe)):

```python
from fastgen.utils.camera_utils import encode_video_with_vipe

poses, intrinsics = encode_video_with_vipe(
    "video.mp4",
    output_path="camera_poses/",   # writes poses.npy + intrinsics.npy beside VIPE's artifacts
)
```

---

#### VideoLatentLoaderConfig

For precomputed video latents and text embeddings (faster than encoding on-the-fly):

```python
from fastgen.configs.data import VideoLatentLoaderConfig

MyVideoLatentLoader = VideoLatentLoaderConfig.copy()
MyVideoLatentLoader.datatags = ["WDS:/path/to/video_latent_shards"]
MyVideoLatentLoader.files_map = {"neg_condition": "/path/to/neg_prompt_emb.npy"}

# For v2v tasks, add condition latent (e.g., depth) to key_map:
MyVideoLatentLoader.key_map["depth_latent"] = "depth_latent.pth"
```

Expected shard contents:
- `latent.pth` - Precomputed video latent
- `txt_emb.pth` - Precomputed text embedding

### Knowledge Distillation Loaders

Specialized loaders for knowledge distillation training; see the section
on KD in [Trajectory Matching](../methods/trajectory_matching/README.md#kd-knowledge-distillation) 
for more details.

#### PairLoaderConfig

For single-step KD with (real, noise, condition) pairs:

```python
from fastgen.configs.data import PairLoaderConfig

MyPairLoader = PairLoaderConfig.copy()
MyPairLoader.datatags = ["WDS:/path/to/pair_shards"]
```

Expected shard contents:
- `latent.pth` - Clean latent (target)
- `noise.pth` - Noise sample
- `txt_emb.pth` - Text embedding

#### PathLoaderConfig

For multi-step KD with denoising trajectories:

```python
from fastgen.configs.data import PathLoaderConfig

MyPathLoader = PathLoaderConfig.copy()
MyPathLoader.datatags = ["WDS:/path/to/path_shards"]
```

Expected shard contents:
- `latent.pth` - Clean latent (target)
- `path.pth` - Denoising trajectory with shape `[steps, C, ...]`, with one non-clean state per interval of the anchored student base grid
- `txt_emb.pth` - Text embedding


### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `datatags` | Dataset paths prefixed with `WDS:`. Supports S3 (`WDS:s3://bucket/path`); S3 shard discovery requires `s5cmd` on `PATH`. |
| `key_map` | Maps output keys to file extensions in shards. |
| `files_map` | Maps output keys to file paths for constants (loaded once). A relative path resolves against the parent of the first datatag's directory, or against that directory itself if the tag ends in `/`. |
| `presets_map` | Maps output keys to preset names: `neg_prompt_wan`, `neg_prompt_lingbot_world`, `neg_prompt_cosmos`, `neg_prompt_cosmos3`, `neg_prompt_ltx2`, `neg_prompt_ltx2_short`, `empty_string`. |
| `presets_filter` | Filters applied before decoding, e.g. `{"score": {"threshold": 5.5, "score_key": "aesthetic_score"}, "empty_keys": {"keys": ["txt"]}}` drops items with an empty `txt` or a JSON `aesthetic_score` below 5.5 (items without a score are kept; invalid ones are dropped with a warning). |
| `sequence_length` | Number of output frames for raw video loaders. With `target_fps` set on `decoder_type="segment"`, the sampled segment spans `sequence_length / target_fps` seconds. |
| `target_fps` | Optional target FPS for `VideoWDSLoader` / `AudioVideoWDSLoader`; segment and full decoders resample source videos by timestamp before transforms. For `AudioVideoWDSLoader`, the audio slice is sized against `target_fps` (not the source FPS), keeping A/V aligned at the resampled rate. Set it to the network's frame rate (`config.model.net.fps`) unless the shards are already at that rate. |
| `deterministic` | Read the shards in order as one unshuffled stream, repeated for training. With `sampler_start_idx=None`, as in the presets, the trainer resumes the stream at the checkpoint; an optional `shard_count_file` mapping shard names to sample counts (JSON, e.g. `{"00000.tar": 1000}`) lets the resume skip whole shards. |
| `ignore_index_paths` | List of JSON files specifying samples to skip; a missing, unreadable or empty file raises. |

Deterministic loading resumes by sample count, so a sample not listed in `ignore_index_paths` that fails or is
dropped makes a later resume repeat one sample (or raises with `strict=True`), and `presets_filter` is rejected.
See the [`wds_utils.py`](wds_utils.py) module docstring for sample order, shard counts, and multi-rank behavior.

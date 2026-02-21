# Datasets

Data loaders for FastGen training, supporting class-conditional image datasets and scalable WebDataset-based loaders for large-scale image and video training.

## Overview

| File | Description | Key Classes |
|------|-------------|-------------|
| [class_cond_dataloader.py](class_cond_dataloader.py) | [Class-conditional image loaders](#class-conditional-datasets) | `ImageLoader` |
| [wds_dataloaders.py](wds_dataloaders.py) | [WebDataset loaders for images/videos](#webdataset-loaders) | `WDSLoader`, `ImageWDSLoader`, `VideoWDSLoader` |
| [../configs/data.py](../configs/data.py) | [Generic loader configs](#generic-loaders) | `ImageLoaderConfig`, `VideoLoaderConfig`, `ImageLatentLoaderConfig`, `VideoLatentLoaderConfig`, `PairLoaderConfig`, `PathLoaderConfig` |

---

## Class-Conditional Datasets

In the following, we provide commands to prepare different versions of CIFAR-10 and ImageNet. For FID reference statistics computations, see [scripts/README.md](../../scripts/README.md#computing-fid-reference-statistics).

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
In each loader config, `key_map` links an output key in the batch (e.g. `"real"`, `"condition"`) to a file extension in the shard. For instance, if files in the tar are named `{sample_id}.{extension}`, then `key_map={"real": "mp4", "condition": "txt"}` loads the file `{sample_id}.mp4` and `{sample_id}.txt` as keys `"real"` and `"condition"` in the batch.

#### WDSLoader

Base loader for precomputed latents and embeddings. Supports `.npy`, `.npz`, `.pth`, `.json`, and text files.

```python
from fastgen.datasets.wds_dataloaders import WDSLoader
from fastgen.utils import LazyCall as L

MyLoader = L(WDSLoader)(
    datatags=["WDS:/path/to/latent_shards"],  # Prefix with "WDS:" (supports S3 via "WDS:s3://bucket/path/to/shards")
    batch_size=32,
    key_map={"real": "latent.pth", "condition": "txt_emb.pth"},
    files_map={"neg_condition": "neg_prompt_emb.npy"},  # Constants loaded once
)
```

#### ImageWDSLoader

For raw images (jpg, png, etc.) with automatic resize, center crop, and normalization.

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

#### VideoWDSLoader

For raw videos (mp4, avi, etc.) with frame extraction and transforms.

```python
from fastgen.configs.data import VideoLoaderConfig

MyVideoLoader = VideoLoaderConfig.copy()
MyVideoLoader.datatags = ["WDS:/path/to/video_shards"]
MyVideoLoader.batch_size = 2
MyVideoLoader.sequence_length = 81
MyVideoLoader.img_size = (832, 480)
```

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

Specialized loaders for knowledge distillation training. See [fastgen/methods/knowledge_distillation/README.md](../methods/knowledge_distillation/README.md) for more details.

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
- `path.pth` - Denoising trajectory with shape `[steps, C, ...]` (typically 4 steps)
- `txt_emb.pth` - Text embedding


### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `datatags` | Dataset paths prefixed with `WDS:`. Supports S3 (`WDS:s3://bucket/path`). |
| `key_map` | Maps output keys to file extensions in shards. |
| `files_map` | Maps output keys to file paths for constants (loaded once). |
| `presets_map` | Maps output keys to preset names: `neg_prompt_wan`, `neg_prompt_cosmos`, `empty_string`. |
| `presets_filter` | Filter config, e.g., `{"score": {"threshold": 5.5, "score_key": "aesthetic_score"}}`. |
| `deterministic` | Enable resumable iteration (requires `shard_count_file`). |
| `ignore_index_paths` | List of JSON files specifying samples to skip. |


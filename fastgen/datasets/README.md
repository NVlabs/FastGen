# Datasets

Data loaders for FastGen training, supporting class-conditional image datasets and scalable WebDataset-based loaders for large-scale image and video training.

## Overview

| File | Description | Key Classes |
|------|-------------|-------------|
| [class_cond_dataloader.py](class_cond_dataloader.py) | [Class-conditional image loaders](#class-conditional-datasets) | `ImageLoader` |
| [wds_dataloaders.py](wds_dataloaders.py) | [WebDataset loaders for images/videos](#webdataset-loaders) | `WDSLoader`, `ImageWDSLoader`, `VideoWDSLoader` |
| [../configs/data.py](../configs/data.py) | [Generic loader configs](#generic-loaders) | `ImageLatentLoaderConfig`, `VideoLatentLoaderConfig`, `PairLoaderConfig`, `PathLoaderConfig` |

---

## Class-Conditional Datasets

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
This prepares the dataset as described in the [EDM2 repo](https://github.com/NVlabs/edm2/tree/main) (with `--resolution=256x256`) and places it at `$DATA_ROOT_DIR/imagenet-256/imagenet_256_sd.zip`, compatible with `ImageNet256_Loader_Config`.


---

## WebDataset Loaders

FastGen provides WebDataset loaders for scalable training on large image and video datasets, supporting both local storage and S3 paths.

### Preparing Your Data

WebDataset stores data as tar archives (shards) containing files grouped by a common key:

```
00000.tar
├── sample_001.mp4       # Video/image file
├── sample_001.txt       # Caption
├── sample_001.json      # Optional metadata
├── sample_002.mp4
├── sample_002.txt
└── ...
```

Create shards using the [webdataset](https://github.com/webdataset/webdataset) library:

```python
import webdataset as wds

with wds.ShardWriter("shards/shard-%05d.tar", maxcount=1000) as sink:
    for idx, (video_path, caption) in enumerate(your_dataset):
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        sink.write({
            "__key__": f"sample_{idx:06d}",
            "mp4": video_bytes,
            "txt": caption.encode("utf-8"),
        })
```

Place shards in a directory with numeric naming (`00000.tar`, `00001.tar`, etc.).

### Generic Loaders

#### WDSLoader

Base loader for precomputed latents and embeddings. Supports `.npy`, `.npz`, `.pth`, `.json`, and text files.

```python
from fastgen.datasets.wds_dataloaders import WDSLoader
from fastgen.utils import LazyCall as L

MyLoader = L(WDSLoader)(
    datatags=["WDS:/path/to/dataset"],  # Prefix with "WDS:" (supports S3 via "WDS:s3://path/to/dataset")
    batch_size=32,
    key_map={"real": "latents.npy", "condition": "text_embedding.npy"},
    files_map={"neg_condition": "neg_prompt_emb.npy"},  # Constants loaded once
    presets_map={"neg_condition_raw": "neg_prompt_wan"},  # Preset strings
)
```

#### ImageWDSLoader

For raw images (jpg, png, etc.) with automatic resize, center crop, and normalization.

```python
from fastgen.datasets.wds_dataloaders import ImageWDSLoader
from fastgen.utils import LazyCall as L

MyImageLoader = L(ImageWDSLoader)(
    datatags=["WDS:/path/to/images"],
    batch_size=32,
    key_map={"real": "jpg", "condition": "txt"},
    input_res=512,  # Target resolution
)
```

#### ImageLatentLoaderConfig

For precomputed image latents and text embeddings (faster than encoding on-the-fly):

```python
from fastgen.configs.data import ImageLatentLoaderConfig

# Override datatags and files_map for your dataset
MyImageLatentLoader = ImageLatentLoaderConfig.clone()
MyImageLatentLoader.datatags = ["WDS:/path/to/my_image_latents"]
MyImageLatentLoader.files_map = {"neg_condition": "/path/to/neg_prompt_emb.npy"}
```

Expected shard contents:
- `latent.pth` - Precomputed image latent
- `txt_emb.pth` - Precomputed text embedding

#### VideoWDSLoader

For raw videos (mp4, avi, etc.) with frame extraction and transforms.

```python
from fastgen.datasets.wds_dataloaders import VideoWDSLoader
from fastgen.utils import LazyCall as L

MyVideoLoader = L(VideoWDSLoader)(
    datatags=["WDS:/path/to/videos"],
    batch_size=2,
    key_map={"real": "mp4", "condition": "txt"},
    presets_map={"neg_condition": "neg_prompt_wan"},
    sequence_length=81,       # Frames to extract
    img_size=(832, 480),      # Target (width, height)
)
```

#### VideoLatentLoaderConfig

For precomputed video latents and text embeddings (faster than encoding on-the-fly):

```python
from fastgen.configs.data import VideoLatentLoaderConfig

# Override datatags and files_map for your dataset
MyVideoLatentLoader = VideoLatentLoaderConfig.clone()
MyVideoLatentLoader.datatags = ["WDS:/path/to/my_video_latents"]
MyVideoLatentLoader.files_map = {"neg_condition": "/path/to/neg_prompt_emb.npy"}

# For v2v tasks, add condition latent (e.g., depth):
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

# Override datatags for your dataset
MyPairLoader = PairLoaderConfig.clone()
MyPairLoader.datatags = ["WDS:/path/to/my_pairs"]
```

Expected shard contents:
- `latent.pth` - Clean latent (target)
- `noise.pth` - Noise sample
- `txt_emb.pth` - Text embedding

#### PathLoaderConfig

For multi-step KD with denoising trajectories:

```python
from fastgen.configs.data import PathLoaderConfig

# Override datatags for your dataset
MyPathLoader = PathLoaderConfig.clone()
MyPathLoader.datatags = ["WDS:/path/to/my_paths"]
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


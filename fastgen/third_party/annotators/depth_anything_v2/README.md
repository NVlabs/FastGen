## Depth Anything V2

The VACE WAN model requires pretrained weights for Depth Anything V2 to extract depth information from video frames.

### Automatic download

On first use, FastGen automatically downloads the
[official Depth Anything V2 Large checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)
from a pinned Hugging Face revision. Rank 0 publishes it in the shared cache
under `$FASTGEN_OUTPUT_ROOT/MODEL`; other ranks and later runs reuse that local
mirror. Enabling VACE preprocessors loads the annotator during model setup,
including when requests supply precomputed depth latents. For a custom
checkpoint, override the default with a local path, S3 URI, or HTTP(S) URL:

```python
config.model.net.depth_model_path = "/path/to/depth_anything_v2_vitl.pth"
```

### Notes

- The depth model is used by VACE (All-in-One Video Creation and Editing) as the depth control signal
- The model is loaded when VACE preprocessors are initialized

The module is imported by
`fastgen/networks/VaceWan/modules/vace_depth_annotator.py` and is essential for
VACE WAN functionality.

### Source

Code from [ali-vilab/VACE](https://github.com/ali-vilab/VACE) at commit [7d316a4](https://github.com/ali-vilab/VACE/tree/7d316a4cb19107dac3a85cd4beab815660f18af6/vace/annotators/depth_anything_v2) (`vace/annotators/depth_anything_v2`), which adapts [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). We made the following changes to the original code:

1. Replace the OpenCV-based `infer_image`/`image2tensor` with `infer_tensor`,
   which preprocesses tensors on their device, and delete the unused `util/transform.py`.
2. Log the missing-xFormers message at debug instead of warning level.
3. Reformat the code and apply lint fixes (explicit re-exports, no `== True`
   comparisons, one unused import).
4. Add headers.

### License

See [`licenses/depth_anything_v2/LICENSE`](../../../../licenses/depth_anything_v2/LICENSE) for details.

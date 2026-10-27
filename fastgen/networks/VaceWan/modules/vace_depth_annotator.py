# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from fastgen.utils.cache_utils import cache
from fastgen.networks.module import FastGenModule
from fastgen.utils.distributed import is_rank0
from fastgen.utils.distributed.collectives import raise_across_ranks
from fastgen.utils.io_utils import load_checkpoint_state_dict
from fastgen.utils.module import load_module_state_dict

# Import VACE annotators
from fastgen.third_party.annotators.depth_anything_v2.dpt import DepthAnythingV2

DEPTH_ANYTHING_V2_VITL_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/"
    "cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9/depth_anything_v2_vitl.pth"
)


class VACEDepthExtractor(FastGenModule):
    """Depth extractor using VACE's Depth Anything V2 model.

    This integrates VACE's Depth Anything V2 model for
    on-the-fly depth extraction in FastGen.
    """

    def __init__(
        self,
        model_path: str = DEPTH_ANYTHING_V2_VITL_URL,
    ):
        """Load Depth Anything V2 on CPU before the model lifecycle prepares this extractor.

        Args:
            model_path: Local path, S3 URI, or HTTP(S) URL for model weights. Defaults to the
                pinned official Depth Anything V2 Large checkpoint.
        """
        super().__init__()

        self.model_path = model_path

        # Mirror remote sources once before every rank loads the checkpoint. Construction happens
        # with the other preprocessors, so preparation can place or shard the complete module.
        with raise_across_ranks(f"cache VACE depth model {self.model_path}"):
            if is_rank0():
                cache(self.model_path, strict=False)
        with raise_across_ranks(f"load VACE depth model {self.model_path}"):
            with torch.inference_mode(False), torch.no_grad(), torch.device("cpu"):
                self.model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]).to(
                    dtype=self.precision_model
                )
                load_module_state_dict(
                    self.model,
                    load_checkpoint_state_dict(self.model_path, device="cpu"),
                    description=f"Depth Anything V2 model from {self.model_path}",
                )

    @property
    def precision_model(self) -> torch.dtype:
        """Depth Anything V2 keeps its weights and neural inputs in FP32."""
        return torch.float32

    def fully_shard(self, **kwargs) -> None:
        """Shard the complete extractor behind its one callable root boundary.

        Depth Anything invokes DINOv2 through ``get_intermediate_layers`` instead of the child's
        ``forward``. A separately wrapped DINOv2 child would therefore bypass its FSDP pre-forward
        hook and expose sharded parameters to ordinary tensor operations. One root group ensures
        the extractor hook materializes every parameter before any custom child method runs.
        """
        from torch.distributed.fsdp import fully_shard

        # The shared preprocessing boundary reshards this root after all depth-conditioning work.
        fully_shard(self, **kwargs)

    def process_frame_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Process a single frame tensor to extract depth.

        Args:
            image_tensor: Input tensor [C, H, W] in range [0, 1], RGB format

        Returns:
            Depth tensor [C, H, W] in range [0, 1], 3-channel grayscale
        """
        with torch.inference_mode():
            # Use the model's tensor-based inference
            depth = self.model.infer_tensor(image_tensor)  # Returns [H, W]

            # Normalize depth values
            depth_min = depth.min()
            depth_max = depth.max()

            # Handle case where all values are the same
            if depth_max - depth_min < 1e-8:
                depth_normalized = torch.ones_like(depth) * 0.5
            else:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)

            # Convert to 3-channel format by repeating
            depth_3ch = depth_normalized.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]

            return depth_3ch

    def extract_depth_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract depth maps from video frames.

        Args:
            frames: Video frames tensor [B, C, T, H, W] in range [0, 1], RGB format

        Returns:
            Depth maps tensor [B, C, T, H, W] in range [0, 1]
        """
        B, C, T, H, W = frames.shape

        # Process each frame in the batch
        all_depths = []

        for b in range(B):
            frame_depths = []

            for t in range(T):
                # Get single frame [C, H, W]
                frame = frames[b, :, t, :, :]

                # Process frame directly as tensor - no numpy conversion needed
                depth_tensor = self.process_frame_tensor(frame)

                frame_depths.append(depth_tensor)

            # Stack time dimension [C, T, H, W]
            batch_depth = torch.stack(frame_depths, dim=1)
            all_depths.append(batch_depth)

        # Stack batch dimension [B, C, T, H, W]
        depths = torch.stack(all_depths, dim=0)

        return depths

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract depth maps from input frames.

        Args:
            frames: Input tensor, either:
                - Video: [B, C, T, H, W] in RGB format
                - Images: [B, C, H, W] in RGB format
                Values should be in range [0, 1]

        Returns:
            Depth maps with same shape as input, in range [0, 1]
        """
        if frames.ndim not in (4, 5):
            raise ValueError(f"Expected 4D or 5D input, got {frames.ndim}D")
        model_frames = frames.to(dtype=self.precision_model)
        if frames.ndim == 5:
            # Video input
            depths = self.extract_depth_from_frames(model_frames)
        elif frames.ndim == 4:
            # Image batch - add time dimension, process, remove
            frames_5d = model_frames.unsqueeze(2)  # [B, C, 1, H, W]
            depths_5d = self.extract_depth_from_frames(frames_5d)
            depths = depths_5d.squeeze(2)  # [B, C, H, W]
        return depths

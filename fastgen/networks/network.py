# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import torch

from fastgen.networks.module import FastGenModule
from fastgen.networks.noise_schedule import NET_PRED_TYPES, get_noise_schedule
from fastgen.utils.distributed.collectives import raise_across_ranks
from fastgen.utils.tensor import high_precision_dtype
from fastgen.utils.tree_utils import OutputDtype, TensorOrDict, assert_same, cast_like, per_modality, to, to_dict
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.utils.guidance import GuidanceConfig


CONDITION_FIELDS = ("condition", "neg_condition")


def validate_conditioning(
    data: Mapping[str, Any],
    required_fields: Collection[str] = (),
    absent_fields: Collection[str] = (),
    require_tensor: bool = False,
    require_mapping: bool = False,
    absent_mapping_keys: Collection[str] = (),
) -> None:
    """Validate required/absent fields and plain conditioning branch types."""
    if require_tensor and require_mapping:
        raise ValueError("A condition cannot be required to be both a tensor and a mapping")

    present = [field for field in absent_fields if field in data]
    if present:
        fields = ", ".join(f"data['{field}']" for field in present)
        raise ValueError(f"build_conditioning requires absent {fields}")

    missing = [field for field in required_fields if data.get(field) is None]
    if missing:
        fields = ", ".join(f"data['{field}']" for field in missing)
        raise ValueError(f"build_conditioning requires non-None {fields}")

    for key in CONDITION_FIELDS:
        value = data.get(key)
        if isinstance(value, Mapping) and not require_mapping:
            raise ValueError(
                f"build_conditioning received a rich `{key}` mapping; "
                "complete conditions are builder outputs and cannot be prepared again"
            )
        if isinstance(value, Mapping):
            present_keys = sorted(field for field in absent_mapping_keys if field in value)
            if present_keys:
                raise ValueError(
                    f"build_conditioning received `{key}` with builder-output fields {present_keys} that must be absent; "
                    "complete conditions are builder outputs and cannot be prepared again"
                )
        elif require_mapping and value is not None:
            raise TypeError(f"build_conditioning requires `{key}` to be a mapping, got {type(value).__name__}")
        if require_tensor and value is not None and not torch.is_tensor(value):
            raise TypeError(f"build_conditioning requires `{key}` to be a tensor, got {type(value).__name__}")


def pixel_shape_from_factors(latent_shape: Sequence[int], factors: Sequence[int]) -> Tuple[int, ...]:
    """``(H, W)`` or ``(T, H, W)`` of the pixels that encode to a ``[C, H, W]`` or ``[C, T, H, W]`` latent.

    ``factors`` are the matching ``(H, W)`` or ``(T, H, W)`` downsampling factors. The first latent frame covers one
    pixel frame and every later one ``T``.
    """
    if len(factors) == 2 and len(latent_shape) == 3:
        h_factor, w_factor = factors
        return (latent_shape[1] * h_factor, latent_shape[2] * w_factor)
    if len(factors) == 3 and len(latent_shape) == 4:
        t_factor, h_factor, w_factor = factors
        return ((latent_shape[1] - 1) * t_factor + 1, latent_shape[2] * h_factor, latent_shape[3] * w_factor)
    raise ValueError(
        "Expected a [C, H, W] latent with (H, W) factors or a [C, T, H, W] latent with (T, H, W) factors, "
        f"got {tuple(latent_shape)} with {tuple(factors)}"
    )


def validate_latent_compression(network: "FastGenNetwork", vae: Any) -> None:
    """Raise unless ``vae`` downsamples by the ``latent_compression_factors`` that ``network`` declares.

    ``pixel_shape`` sizes pixel inputs from the declaration, not from a loaded VAE. A network that
    declares no factors is not checked.
    """
    if not hasattr(network, "latent_compression_factors"):
        return
    declared = tuple(network.latent_compression_factors)
    loaded = tuple(vae.latent_compression_factors)
    if loaded != declared:
        raise ValueError(
            f"{type(network).__name__} declares latent compression {declared}, but its VAE compresses by "
            f"{loaded}; load the checkpoint's own VAE"
        )


class FastGenNetwork(FastGenModule, ABC):
    """
    Abstract base class for FastGen neural network architectures.

    This class provides a common interface for all neural network models used in the FastGen framework
    for fast generative modeling. It defines the core methods that must be implemented by specific
    network architectures like EDM, EDM2, SD15, CogVideoX, etc.

    The class handles noise scheduling, prediction types, and provides abstract methods for
    forward passes and sampling operations.

    Attributes:
        net_pred_type (str): The prediction type of the network ('x0', 'eps', 'v', 'flow').
        schedule_type (str): The noise schedule type used by the network.
        noise_scheduler: The noise scheduler instance for handling diffusion timesteps.

    Args:
        net_pred_type (str, optional): The prediction type of the network.
            Must be one of ['x0', 'eps', 'v', 'flow']. Defaults to "x0".
        schedule_type (str, optional): The noise schedule type to use.
            Common types include 'edm', 'rf', 'sd'. Defaults to "edm".
        **schedule_kwargs: Additional keyword arguments passed to the noise scheduler.
    """

    def _validate_sample_kwargs(
        self,
        sample_type: str,
        supported_schedule_types: Sequence[str],
        supported_sample_types: Sequence[str],
    ) -> None:
        """Validate sampler policy before a sampler mutates scheduler state."""
        supported_schedule_types = tuple(supported_schedule_types)
        supported_sample_types = tuple(supported_sample_types)
        if self.schedule_type not in supported_schedule_types:
            raise ValueError(
                f"{type(self).__name__}.sample does not support schedule_type={self.schedule_type!r}; "
                f"supported schedule types are {supported_schedule_types}"
            )
        if sample_type not in supported_sample_types:
            raise ValueError(
                f"{type(self).__name__}.sample does not support sample_type={sample_type!r}; "
                f"supported sample types are {supported_sample_types}"
            )

    # Optional predict_with_guidance(ctx: GuidanceContext) recipe. It obtains the
    # native branches through ctx.predict, in guidance's working dtype, and returns
    # the composed/rescaled prediction in that space and dtype. ctx.predict returns
    # cached tensors, so the recipe must not modify them in place. The dispatcher
    # applies transform_output to the result once and owns cache returns.
    predict_with_guidance = None

    # Optional output constraint, the last step of transform_output:
    # _condition_output(prediction, x_t, t, condition, fwd_pred_type, **context).
    # None means the network constrains nothing.
    #
    # It receives the prediction converted to fwd_pred_type in high precision, with the
    # caller's raw x_t and t, and must preserve its structure, dtype, and device.
    # Conditions are exactly representable in the state dtype; apply them without rounding
    # (select with boolean masks, blend in the operands' promoted dtype). A constraint that
    # fwd_pred_type cannot express (e.g. non-x0 types at sigma = 0) raises a ValueError.
    _condition_output = None

    def transform_output(
        self,
        prediction: TensorOrDict,
        x_t: TensorOrDict,
        t: torch.Tensor,
        condition: Any = None,
        fwd_pred_type: Optional[str] = None,
        output_dtype: OutputDtype = "same",
        apply_output_conditioning: bool = True,
        **context: Any,
    ) -> TensorOrDict:
        """Turn a native prediction into the output; every ``forward`` ends with this call.

        Converts it from ``net_pred_type`` to ``fwd_pred_type`` in at least ``HIGH_PRECISION``, then applies the
        output constraint in that precision unless ``apply_output_conditioning=False``. Only then is the result
        rounded, once, to ``output_dtype``: a dtype, or a mapping of one per modality; ``"same"`` is each modality's
        state dtype, and ``None`` keeps the transform's precision. A forward with the default ``fwd_pred_type``,
        ``apply_output_conditioning=False`` and ``output_dtype=None`` thus returns the native prediction unrounded.
        Guidance requests its branches the same way but in its working dtype, and applies this once to the result.
        ``context`` holds the forward kwargs that the conversion and the constraint read.
        """
        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        elif fwd_pred_type not in NET_PRED_TYPES:
            raise ValueError(
                f"{type(self).__name__}.forward does not support fwd_pred_type={fwd_pred_type!r}; "
                f"supported types are {NET_PRED_TYPES}"
            )
        converted = self._convert_output(prediction, x_t, t, condition, fwd_pred_type, **context)
        # The conversion must not round: its result is at least the high precision of its inputs.
        per_modality(
            lambda c, p, x: assert_same(dtype=high_precision_dtype(c, p, x), converted=c), converted, prediction, x_t
        )
        if apply_output_conditioning and self._condition_output is not None:
            conditioned = self._condition_output(
                converted, x_t, t, condition=condition, fwd_pred_type=fwd_pred_type, **context
            )
            assert_same(dtype=True, device=True, converted=converted, conditioned=conditioned)
            converted = conditioned
        return cast_like(converted, x_t, output_dtype)

    def _convert_output(
        self,
        prediction: TensorOrDict,
        x_t: TensorOrDict,
        t: torch.Tensor,
        condition: Any,
        fwd_pred_type: str,
        **context: Any,
    ) -> TensorOrDict:
        """The conversion step of ``transform_output``, from ``net_pred_type`` to ``fwd_pred_type``, returned uncast
        in at least ``HIGH_PRECISION``.

        Networks whose conversion uses inputs or timestep layouts other than the caller's state and ``t``
        override it. From ``net_pred_type`` to itself it must only cast: guidance converts its composed native
        branches again.
        """
        return self.noise_scheduler.convert_model_output(
            x_t,
            prediction,
            t,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
            output_dtype=None,
        )

    def _validate_forward_kwargs(
        self,
        forward_kwargs: Mapping[str, Any],
        r: Any = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        skip_layers: Any = None,
        supports_r_timestep: bool = False,
        supports_skip_layers: bool = False,
        num_features: Optional[int] = None,
    ) -> None:
        """Reject unknown kwargs and validate optional forward controls before neural work.

        Feature and skip-layer indices share the ``num_features`` range. Networks pass
        their extraction-point count regardless of whether features are requested.
        """
        forward_kwargs = dict(forward_kwargs)
        r = forward_kwargs.pop("r", r)
        skip_layers = forward_kwargs.pop("skip_layers", skip_layers)
        if forward_kwargs:
            names = ", ".join(repr(name) for name in sorted(forward_kwargs))
            raise TypeError(f"{type(self).__name__}.forward got unexpected keyword argument(s): {names}")

        if r is not None and not supports_r_timestep:
            raise NotImplementedError(f"{type(self).__name__}.forward does not support secondary r timesteps")

        if feature_indices is not None and not isinstance(feature_indices, set):
            raise TypeError(
                f"{type(self).__name__}.forward requires feature_indices to be a set or None; "
                f"got {type(feature_indices).__name__}"
            )
        if return_features_early and feature_indices is None:
            raise ValueError(f"{type(self).__name__}.forward requires feature_indices when return_features_early=True")
        if feature_indices is not None:
            invalid_types = [
                index for index in feature_indices if not isinstance(index, int) or isinstance(index, bool)
            ]
            if invalid_types:
                raise TypeError(
                    f"{type(self).__name__}.forward requires feature_indices to contain integers excluding bool; "
                    f"got {sorted(invalid_types, key=repr)!r}"
                )

        for name, indices, supported in (
            ("feature_indices", feature_indices, True),
            ("skip_layers", skip_layers, supports_skip_layers),
        ):
            if not indices:
                continue
            if not supported:
                raise NotImplementedError(f"{type(self).__name__}.forward does not support {name!r}")
            if num_features is None:
                reason = "requesting features" if name == "feature_indices" else f"supporting {name}"
                raise RuntimeError(f"{type(self).__name__}.forward must provide num_features when {reason}")
            self._validate_index_range(name, indices, num_features)

    def _validate_index_range(self, name: str, indices: Collection[int], num_features: int) -> None:
        """Reject indices of the forward control ``name`` outside ``[0, num_features)``."""
        invalid = sorted(index for index in indices if index < 0 or index >= num_features)
        if invalid:
            raise ValueError(
                f"{type(self).__name__}.forward {name} contains out-of-range indices {invalid}; "
                f"valid range is [0, {num_features})"
            )

    @staticmethod
    def _is_in_meta_context() -> bool:
        """Check if we're currently in a torch.device('meta') context.

        This allows networks to detect when they're being instantiated inside
        a meta device context manager (used for FSDP memory-efficient loading).
        When in meta context, networks should use from_config instead of from_pretrained
        to avoid loading weights that will be broadcast from rank 0.

        Returns:
            bool: True if currently in a meta device context, False otherwise.
        """
        test_param = torch.nn.Parameter(torch.empty(0))
        return test_param.device.type == "meta"

    def __init__(
        self,
        net_pred_type: str = "x0",
        schedule_type: str = "edm",
        **schedule_kwargs,
    ):
        super().__init__()

        # Deliberately not a ModuleDict: preprocessors have their own setup lifecycle and
        # must not enter this generator's state_dict, parameter traversal, DDP, or FSDP tree.
        self.preprocessors: Dict[str, FastGenModule] = {}

        # Default prediction type - can be overridden by subclasses
        self.net_pred_type = net_pred_type
        self.schedule_type = schedule_type
        self._schedule_kwargs = schedule_kwargs
        self._validate_net_pred_type(self.net_pred_type)
        self._set_noise_schedule(**self._schedule_kwargs)

    def _validate_net_pred_type(self, pred_type: str) -> None:
        """Validate that the network prediction type is supported.

        Args:
            pred_type (str): The prediction type to validate.

        Raises:
            ValueError: If the prediction type is not supported.
        """
        if pred_type not in NET_PRED_TYPES:
            raise ValueError(f"Unsupported net_pred_type '{pred_type}'. " f"Supported types are: {NET_PRED_TYPES}")

    # Pixel-to-latent downsampling of the network's VAE: ``(T, H, W)`` for video networks and ``(H, W)`` for image
    # networks, all ones in pixel space. A network whose VAE depends on its checkpoint sets it per instance.
    latent_compression_factors: Tuple[int, ...]

    def pixel_shape(self, latent_shape: Sequence[int]) -> Tuple[int, ...]:
        """``(H, W)`` or ``(T, H, W)`` of the pixels that encode to a ``[C, H, W]`` or ``[C, T, H, W]`` latent."""
        return pixel_shape_from_factors(latent_shape, self.latent_compression_factors)

    @classmethod
    def pixel_shape_for(cls, latent_shape: Sequence[int]) -> Tuple[int, ...]:
        """``pixel_shape`` without an instance, so configs can size pixel inputs before any network exists."""
        return pixel_shape_from_factors(latent_shape, cls.latent_compression_factors)

    def init_preprocessors(self) -> None:
        """Construct this network's preprocessors (no-op for networks without any).

        Construction only: ``FastGenModel.init_preprocessors`` sets their precision and device.
        """

    def get_preprocessor(self, name: str, purpose: Optional[str] = None) -> FastGenModule:
        """Return ``self.preprocessors[name]``, or raise a clear error when it was never initialized.

        Preprocessors are constructed and prepared only by ``FastGenModel.init_preprocessors``; nothing
        builds one lazily. ``purpose`` names the operation for the error message.
        """
        preprocessor = self.preprocessors.get(name)
        if preprocessor is None:
            needed_for = f" to {purpose}" if purpose else ""
            raise ValueError(
                f"{type(self).__name__} requires the {name!r} preprocessor{needed_for}; preprocessors are built "
                "when config.model.enable_preprocessors is set (or supply the precomputed input instead)."
            )
        return preprocessor

    @contextlib.contextmanager
    def vae_context(self) -> Iterator[Optional["FastGenModule"]]:
        """Yield this network's VAE, borrowing a prepared temporary one when absent.

        Persistent preprocessors are reused unchanged. A VAE created here is owned only by
        this lexical context: it is prepared as a frozen, unsharded module, then removed on
        exit. Callers own execution policy such as inference mode and autocast; construction and
        preparation of the temporary VAE leave an inference context so its state is ordinary tensors.

        Temporary preparation contains distributed collectives. A distributed caller that
        may create a VAE must therefore enter this context on every rank in the same order.
        Those ranks must agree on whether a persistent VAE is already available.
        """
        if not hasattr(self, "init_vae"):
            yield None
            return

        vae = self.preprocessors.get("vae")
        if vae is not None:
            yield vae
            return

        temporary_vae = None
        try:
            with torch.inference_mode(False), torch.no_grad():
                with raise_across_ranks("temporary VAE initialization"):
                    if not self.is_prepared:
                        raise RuntimeError(
                            f"{type(self).__name__}.vae_context requires the host network to be prepared first."
                        )
                    self.init_vae()
                    temporary_vae = self.preprocessors.get("vae")
                    assert isinstance(
                        temporary_vae, FastGenModule
                    ), f"{type(self).__name__}.init_vae() must install a FastGenModule in preprocessors['vae']."
                    validate_latent_compression(self, temporary_vae)

                # Forward the model-precision request, not this trainable network's optimizer
                # precision or an effective network override. The VAE resolves its own precision.
                temporary_vae.prepare_module(
                    device=self.device,
                    precision_model=self._precision_model,
                )
            yield temporary_vae
        finally:
            # Preserve a replacement installed by code inside the context. When initialization
            # failed after installing a partial VAE, remove that newly introduced entry too.
            current_vae = self.preprocessors.get("vae")
            if temporary_vae is None or current_vae is temporary_vae:
                self.preprocessors.pop("vae", None)

    def _set_noise_schedule(self, schedule_type: Optional[str] = None, **schedule_kwargs) -> None:
        """Initialize the network's noise scheduler during construction, before preparation.

        Args:
            schedule_type (Optional[str]): Type of noise schedule to use.
                If None, uses the current self.schedule_type.
            **schedule_kwargs: Additional arguments passed to the noise scheduler.
        """
        assert not self.is_prepared, "Initialize the noise schedule before preparation"
        if schedule_type is not None:
            self.schedule_type = schedule_type
        self.noise_scheduler = get_noise_schedule(self.schedule_type, **schedule_kwargs)

    def reset_parameters(self):
        """Rebuild deterministic state after meta materialization.

        Rank 0's values replace every parameter and persistent buffer afterwards, so this hook must
        not initialize them or draw random values. Override it only to rebuild unregistered tensors
        and non-persistent buffers, preserving every registered object and its metadata. Parameter
        ties are already restored. Call ``super().reset_parameters()`` for the noise scheduler.
        """
        # Non-persistent schedule tables must be reconstructed after meta materialization.
        # Preserve the module identity, registered placement, and any shared scheduler references.
        if hasattr(self, "noise_scheduler") and self.noise_scheduler is not None:
            self.noise_scheduler.reset_parameters()
            logger.debug("Reinitialized noise scheduler")

    def fully_shard(self, **kwargs):
        """Shard this network's module boundaries with FSDP2.

        ``FastGenNetwork`` derives from ``ABC``, which conflicts with the ``__class__`` reassignment
        FSDP2 performs on the module it wraps, so subclasses never shard ``self``. They shard the
        inner module whose ``forward`` is the only entry point (usually ``self.transformer``):
        convert flag-gated checkpointing first (``apply_fsdp_checkpointing``), then ``fully_shard``
        each block, then the inner root. Plain ``FastGenModule`` subclasses may shard ``self`` when
        their entry points bypass child ``forward`` hooks (``MiniMaxH3VAE``, ``VACEDepthExtractor``).
        """
        raise NotImplementedError(
            f"Network {self.__class__.__name__} does not implement the fully_shard method. "
            "Use external sharding methods or implement this method in the subclass."
        )

    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        neg_condition: Optional[Any] = None,
        sample_steps: int = 50,
        sample_type: str = "ode",
        guidance_cfg: Optional["GuidanceConfig"] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using the trained network.

        This method performs the sampling/inference process to generate new data from noise.
        It should implement the specific sampling algorithm for the network architecture.

        Note: This method is optional. Networks that don't support direct sampling
        can leave this unimplemented and use external sampling methods instead.

        Args:
            noise (torch.Tensor): Initial noise tensor to start sampling from.
                Shape should match the expected input format of the network.
            condition (Any): Conditioning information for guided generation.
                Can be text embeddings, class labels, or other structured conditioning data.
                Can be a single tensor, list of tensors, or dictionary mapping condition names to tensors.
                Defaults to None for unconditional generation.
            neg_condition (Any): Negative conditioning for classifier-free guidance. Defaults to None.
            sample_steps (int, optional): Number of denoising steps to perform during sampling.
                More steps generally lead to higher quality but slower generation. Defaults to 50.
            sample_type: Integration policy. All samplers accept deterministic
                ``"ode"``; only samplers with an explicit SDE implementation accept ``"sde"``.
            guidance_cfg (Optional[GuidanceConfig]): Composite guidance config (CFG / mix /
                rescale, plus a subclass's network terms).  ``None`` means the subclass should fall
                back to its ``default_guidance_cfg()`` recipe.
            **kwargs: Additional keyword arguments specific to the sampling implementation.

        Returns:
            torch.Tensor: Generated samples with the same spatial dimensions as latents
                but potentially different channel dimensions based on the network's output.

        Raises:
            NotImplementedError: If the network doesn't provide its own sampling implementation.
        """
        raise NotImplementedError(
            f"Network {self.__class__.__name__} does not implement the sample method. "
            "Use external sampling methods or implement this method in the subclass."
        )

    @abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        apply_output_conditioning: bool = True,
        output_dtype: OutputDtype = "same",
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of diffusion model.

        This method performs a single forward pass through the network, predicting the denoised
        output or noise based on the current noisy input and timestep.

        Args:
            x_t (torch.Tensor): The diffused/noisy data sample at timestep t.
                Shape should be [batch_size, channels, ...spatial_dims].
            t (torch.Tensor): The current timestep(s). Can be a scalar or tensor
                with shape [batch_size].
            condition (Any, optional): Optional conditioning information such as text embeddings, class labels,
                or attention masks. Can be a single tensor, list of tensors, or dictionary
                mapping condition names to tensors. Format depends on the specific network implementation.
                Defaults to None.
            r (Optional[torch.Tensor], optional): Additional timestep parameter,
                primarily used by mean flow methods. Defaults to None.
            return_features_early (bool, optional): If True, returns intermediate features
                as soon as they are computed, without completing the full forward pass.
                Requires feature_indices, which may be empty; an empty set returns []
                immediately. Defaults to False.
            feature_indices (Optional[Set[int]], optional): Set of layer indices to extract
                features from. Any provided set includes a feature list in the return value,
                including an empty list for an empty set. None returns only the model output.
                Members must be integers (excluding bool) within the architecture's feature range;
                UNets index encoder stages, and transformers index their blocks.
            return_logvar (bool, optional): If True, returns the log-variance estimate
                along with the main output. Defaults to False.
            fwd_pred_type (Optional[str], optional): Override the network's prediction type
                for this forward pass. Must be in ['x0', 'eps', 'v', 'flow'].
                None uses the network's default net_pred_type. Defaults to None.
            apply_output_conditioning: Apply the output constraint after the conversion. Code that composes or
                reuses predictions, such as guidance and DMD2's shared teacher pass, disables it and constrains
                the result once.
            output_dtype: Dtype of the prediction, or a mapping of one per modality: ``"same"`` rounds it to each
                modality's state dtype, and ``None`` keeps the transform's high precision. Guidance requests its
                branches in its working dtype.
            **fwd_kwargs: Guidance modifiers validated by the concrete network.

        Returns:
            Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
                - If return_features_early=True: List of intermediate features
                - If feature_indices is not None: List containing [output, features]
                - If return_logvar=True: Tuple containing (output, logvar)
                - Otherwise: Single output tensor with predictions

        Raises:
            TypeError: If feature_indices is neither a set nor None, or contains non-integer members.
            ValueError: If fwd_pred_type is unsupported or early feature return is requested
                without feature_indices, or a feature index is outside this architecture's range.
            NotImplementedError: Must be implemented by concrete subclasses.
        """

    def encode_real(self, real_raw: TensorOrDict) -> TensorOrDict:
        """Encode raw signal-space state into this network's model space.

        Latent networks encode with their VAE and return its native precision, which the caller
        casts to the state precision; pixel-space networks override this to return ``real_raw``
        unchanged.
        """
        vae = self.preprocessors.get("vae")
        if vae is None:
            raise ValueError(f"{type(self).__name__} requires a VAE to materialize `real` from `real_raw`")
        real = vae.encode(real_raw)
        assert_same(device=True, **(to_dict(real_raw, "vae_input") | to_dict(real, "vae_output")))
        return real

    def _resolve_conditions(
        self, data: Mapping[str, Any], dtype: Optional[torch.dtype], **encode_kwargs
    ) -> Dict[str, Any]:
        """Resolve the ordinary positive and negative branches in ``dtype``.

        A non-``None`` plain condition is authoritative, so its raw mirror is not encoded. Otherwise the
        ``*_raw`` prompt is encoded with this network's text encoder, which receives ``encode_kwargs``.
        Cached and encoded branches are cast together, once; ``dtype=None`` keeps their dtype, and integer and
        boolean leaves, such as class indices and masks, always keep theirs.
        """
        out = {key: data[key] for key in CONDITION_FIELDS if data.get(key) is not None}
        requested = {
            key: data[f"{key}_raw"] for key in CONDITION_FIELDS if key not in out and data.get(f"{key}_raw") is not None
        }
        if requested:
            text_encoder = self.preprocessors.get("text_encoder")
            if text_encoder is None:
                raise ValueError(
                    f"{type(self).__name__} received raw text for {tuple(requested)} "
                    "but does not provide a text-conditioning implementation"
                )
            out.update({key: text_encoder.encode(raw, **encode_kwargs) for key, raw in requested.items()})
        out = to(out, dtype=dtype, cast_ints=False)
        # This is only the ordinary resolver's contract. Callers may enrich or replace either
        # branch afterwards and own compatibility with their guidance path, including
        # concatenability when they use batched classifier-free guidance.
        assert_same(
            dtype=True,
            device=True,
            condition=out.get("condition"),
            neg_condition=out.get("neg_condition"),
        )
        return out

    def build_conditioning(
        self,
        data: dict,
        input_shape: Optional[Union[Sequence[int], Mapping[str, Optional[Sequence[int]]]]] = None,
        precision: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Resolve ordinary raw/encoded branches and return updates for ``data``.

        A non-``None`` plain condition is authoritative.  Otherwise the
        corresponding ``*_raw`` value is encoded with this network's text
        encoder.  Raw values are never classified or mirrored, and both
        branches are returned in this network's compute dtype
        (``precision_model``), so cached and live branches always agree.
        Rich networks override this complete method, call ``super()`` once,
        and enrich its resolved branches using task inputs from ``data``
        without mutating the input mapping. A network that reads the branches
        in another precision calls ``_resolve_conditions`` with that dtype
        instead of ``super()``, so they are cast only once. What rich networks
        derive from raw inputs is cast once here as well: values that join the
        state, such as first-frame latents, to ``precision``, and values only
        the backbone reads, such as image embeddings, to ``precision_model``.
        Forwards reject conditioning in another dtype instead of casting it.

        Args:
            data: Complete request. Raw prompts use ``condition_raw`` and
                ``neg_condition_raw``; encoded branches use the plain keys.
            input_shape: Target latent shape ``(C, T, H, W)`` (or
                ``dict[str, list[int]]`` for multimodal networks).
                Required by subclasses that need the output size for
                conditioning padding (I2V ``concat_mask``, V2W
                ``condition_mask``); ignored by text-only networks.
                Trainer and inference both pass ``model.input_shape``.
            precision: Dtype of state (``model.precision``); ``None`` leaves the
                derived state values in their preprocessors' output dtype.

        Returns:
            Positive and negative updates produced from the available inputs.
            Branches with neither a non-``None`` encoded value nor a raw
            request are omitted. Producer outputs and overrides own their
            result semantics, including whether a returned value is ``None``.
        """
        # ``precision_model`` is None until the network is prepared.
        return self._resolve_conditions(data, dtype=self.precision_model)

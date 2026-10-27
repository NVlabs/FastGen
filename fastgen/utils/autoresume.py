# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Auto-resume interface for cluster-specific implementations.

This module provides an abstract interface that users can implement based on their
cluster needs (e.g., SLURM, PBS, Kubernetes, etc.). The concrete implementation is
selected from config: set ``config.trainer.auto_resume`` to a LazyCall spec that
builds an ``AutoResumeInterface``; leaving it ``None`` (the default) uses
``NoOpAutoResume`` (auto-resume disabled).

Example usage:

    # Create your custom implementation
    class SlurmAutoResume(AutoResumeInterface):
        def init(self) -> None:
            # Initialize SLURM-specific signal handlers
            ...

        def get_resume_details(self) -> Optional[Dict[str, Any]]:
            # Check for previous checkpoint
            ...

        def termination_requested(self) -> bool:
            # Check SLURM preemption signals
            ...

        def request_resume(self, user_dict: Dict[str, Any]) -> None:
            # Requeue the job
            ...

    # Wire it through config; the Trainer builds it from ``config.trainer.auto_resume``
    from fastgen.configs.instantiate import LazyCall as L
    config.trainer.auto_resume = L(SlurmAutoResume)()
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import fastgen.utils.logging_utils as logger


class AutoResumeInterface(ABC):
    """
    Abstract interface for auto-resume functionality.

    Users should implement this interface based on their cluster scheduler
    (e.g., SLURM, PBS, Kubernetes, etc.).

    The auto-resume system allows training jobs to:
    1. Save checkpoints before preemption/termination
    2. Automatically resume from the latest checkpoint when restarted
    3. Handle cluster-specific job management (requeuing, etc.)
    """

    @abstractmethod
    def init(self) -> None:
        """
        Initialize the auto-resume system.

        This is called once at the start of training. Use this to:
        - Set up signal handlers for preemption
        - Start timers for time-limit checks
        - Initialize any cluster-specific state
        """
        pass

    @abstractmethod
    def get_resume_details(self) -> Optional[Dict[str, Any]]:
        """
        Get details about a previous checkpoint to resume from.

        This is called after init() to check if the current job is a
        continuation of a previous job.

        Returns:
            Optional[Dict[str, Any]]: Dictionary with resume details containing
                at minimum:
                - 'save_path': Path to the checkpoint to resume from
                - 'id': Job ID or other identifier
                Returns None if this is a fresh start (not resuming).
        """
        pass

    @abstractmethod
    def termination_requested(self) -> bool:
        """
        Check if termination has been requested.

        This should return True when:
        - The job is being preempted
        - Time limit is approaching
        - A graceful shutdown signal was received

        Returns:
            bool: True if termination is requested, False otherwise
        """
        pass

    @abstractmethod
    def request_resume(self, user_dict: Dict[str, Any]) -> None:
        """
        Request that the job be resumed with the given details.

        This is called when termination is requested and a checkpoint has
        been saved. Use this to:
        - Requeue the job with the cluster scheduler
        - Store the checkpoint path for the next job
        - Update any job management state
        - Add cluster-specific identifiers (e.g., job ID) if not already present

        Args:
            user_dict: Dictionary with information needed to resume, including:
                - 'save_path': Path to the saved checkpoint
                - Any other user-specified data
        """
        pass


class NoOpAutoResume(AutoResumeInterface):
    """
    Default no-op implementation that disables auto-resume.

    Use this when:
    - Auto-resume is not needed
    - Running on a system without preemption
    - For testing/debugging purposes
    """

    def init(self) -> None:
        """No-op initialization."""
        pass

    def get_resume_details(self) -> Optional[Dict[str, Any]]:
        """Always returns None (never resumes)."""
        return None

    def termination_requested(self) -> bool:
        """Always returns False (never requests termination)."""
        return False

    def request_resume(self, user_dict: Dict[str, Any]) -> None:
        """No-op resume request."""
        pass


def create_auto_resume(
    auto_resume_config: Optional[Any] = None,
) -> AutoResumeInterface:
    """
    Resolve the AutoResume instance for a training run.

    Resolution order:
    1. ``auto_resume_config`` (a LazyCall spec from ``config.trainer.auto_resume``)
       is instantiated when provided.
    2. Otherwise ``NoOpAutoResume`` (auto-resume disabled).

    For production training on preemptible clusters (SLURM, PBS, Kubernetes, etc.),
    provide an ``AutoResumeInterface`` implementation via ``config.trainer.auto_resume``
    that handles signal handling, checkpoint saving before termination, and job requeuing.

    Args:
        auto_resume_config: Optional LazyCall/DictConfig spec to instantiate.

    Returns:
        The resolved ``AutoResumeInterface`` instance.
    """
    if auto_resume_config is not None:
        # Local import avoids any import-order coupling with fastgen.utils.__init__.
        from fastgen.configs.instantiate import instantiate

        resolved = instantiate(auto_resume_config)
        if resolved is not None:
            logger.info(f"Using auto-resume from config: {type(resolved).__name__}")
            return resolved

    logger.info("Auto-resume not configured, using NoOpAutoResume (auto-resume disabled)")
    return NoOpAutoResume()

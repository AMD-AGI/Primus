###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
BaseTrainer: Universal base class for all backend trainers.

This class provides a unified trainer interface that works across all
backends (Megatron, TorchTitan, JAX/Maxtext, etc.).

Responsibilities:
    - Provide access to configuration and runtime context
    - Define training lifecycle (setup, init, train, cleanup)
    - Access distributed training parameters from environment
    - Store backend-specific arguments
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from primus.core.utils.env import get_torchrun_env


class BaseTrainer(ABC):
    """
    Universal base trainer for all backend frameworks.

    This class provides a consistent training interface across all backends
    while allowing flexibility for backend-specific implementations.

    Subclasses (backend-specific trainers) must implement:
        - setup(): Pre-initialization setup (optional logic)
        - init(): Initialize training components
        - train(): The actual training logic

    Example hierarchy:
        BaseTrainer (this class)
            ↓
        MegatronBaseTrainer, TorchtitanBaseTrainer, MaxtextBaseTrainer
            ↓
        MegatronPretrainTrainer, TorchtitanPretrainTrainer, etc.
    """

    def __init__(self, backend_args: Any):
        """
        Initialize base trainer.

        Args:
            backend_args: Backend-specific arguments (e.g., from MegatronArgBuilder)

        Note:
            Distributed environment and logging should be initialized globally
            before creating trainer instances.
        """
        self.backend_args = backend_args

        # Resolve distributed environment directly from torchrun-style env vars
        dist_env = get_torchrun_env()
        self.rank = dist_env["rank"]
        self.world_size = dist_env["world_size"]
        self.local_rank = dist_env["local_rank"]
        self.master_addr = dist_env["master_addr"]
        self.master_port = dist_env["master_port"]

    @abstractmethod
    def setup(self):
        """Setup phase before initialization (optional)."""
        raise NotImplementedError

    @abstractmethod
    def init(self):
        """Initialize training components (model, optimizer, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """
        Execute the actual training loop.

        This method must be implemented by subclasses to provide
        the task-specific training logic.

        Example (MegatronPretrainTrainer):
            def train(self):
                from megatron.training import pretrain
                pretrain(train_valid_test_datasets_provider, model_provider, ...)

        Example (TorchtitanPretrainTrainer):
            def train(self):
                from torchtitan.train import train
                train(config, ...)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement train()")

    def cleanup(self, on_error: bool = False):
        """
        Cleanup and finalize training resources.

        This method is called after training completes (successfully or with error)
        to clean up resources and perform finalization tasks.

        Args:
            on_error: Whether cleanup is being called due to an error

        Typical cleanup tasks:
            - Save final checkpoint (if not saved)
            - Close file handles and logging
            - Release GPU memory
            - Cleanup temporary files
            - Finalize distributed processes
            - Generate training summary/report
        """
        # Default implementation does nothing
        # Subclasses can override to add cleanup logic

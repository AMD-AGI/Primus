###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

    def __init__(self, backend_args: Any = None, *args, **kwargs):
        """
        Initialize base trainer.

        Args:
            backend_args: Backend-specific arguments (e.g., from MegatronArgBuilder)
            *args, **kwargs: Additional arguments (filtered to prevent reaching object.__init__())

        Note:
            Distributed environment and logging should be initialized globally
            before creating trainer instances.
        """
        # Filter backend_args from kwargs if not provided as positional/keyword argument
        if backend_args is None:
            backend_args = kwargs.pop("backend_args", None)

        try:
            from abc import ABC

            ABC.__init__(self)

            self.backend_args = backend_args

            dist_env = get_torchrun_env()
            self.rank = dist_env["rank"]
            self.world_size = dist_env["world_size"]
            self.local_rank = dist_env["local_rank"]
            self.master_addr = dist_env["master_addr"]
            self.master_port = dist_env["master_port"]

            # Cooperative multiple inheritance: pass kwargs to BaseModule if present in MRO,
            # otherwise call super().__init__() with no args to avoid object.__init__() error.
            from primus.modules.base_module import BaseModule

            if BaseModule in type(self).__mro__:
                super().__init__(**kwargs)
            else:
                super().__init__()
        except Exception:
            import traceback

            traceback.print_exc()
            raise

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

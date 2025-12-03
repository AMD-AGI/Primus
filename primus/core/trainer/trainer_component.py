###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from primus.core.utils.env import get_torchrun_env


class TrainerComponent(ABC):
    """
    Base class for all Primus trainers.

    This is a simplified base class that focuses on training logic.
    Environment initialization (distributed setup, logging) is handled
    globally in train_launcher.py before trainers are instantiated.

    Responsibilities:
        - Provide access to configuration and runtime context
        - Define training lifecycle (init, run)
        - Access distributed training parameters from global context
    """

    def __init__(
        self,
        primus_config: Any,
        module_config: Any,
    ):
        """
        Initialize TrainerComponent.

        Args:
            primus_config: Primus configuration object
            module_config: Module-specific configuration

        Note:
            Distributed environment and logging should be initialized globally
            before creating trainer instances. Use RuntimeContext to access
            distributed parameters.
        """
        self.primus_config = primus_config
        self.module_config = module_config

        # Resolve distributed environment directly from torchrun-style env vars
        dist_env = get_torchrun_env()
        self.rank = dist_env["rank"]
        self.world_size = dist_env["world_size"]
        self.local_rank = dist_env["local_rank"]
        self.master_addr = dist_env["master_addr"]
        self.master_port = dist_env["master_port"]

    @abstractmethod
    def init(self, *args, **kwargs):
        """Initialize training components (model, optimizer, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Setup phase before initialization (optional)."""
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute training loop."""
        raise NotImplementedError

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

    # Properties for accessing configuration
    @property
    def exp_root_path(self) -> str:
        """Get experiment root path."""
        return self.primus_config.exp_root_path

    @property
    def exp_meta_info(self) -> dict:
        """Get experiment metadata (work_group, user_name, exp_name, etc.)."""
        return self.primus_config.exp_meta_info

    @property
    def trainable(self) -> bool:
        """Check if this module is trainable."""
        return getattr(self.module_config, "trainable", True)

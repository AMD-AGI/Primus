###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod

from primus.core.config.primus_config import PrimusConfig
from primus.core.runtime import RuntimeContext


class BaseModule(ABC):
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
        primus_config: PrimusConfig,
        module_config,
    ):
        """
        Initialize BaseModule.

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
        self.module_name = module_config.module  # e.g., "pre_trainer"

        # Get runtime context (initialized globally in train_launcher)
        self.context = RuntimeContext.get_instance()

        # Verify runtime is initialized
        if not self.context.distributed_initialized:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Distributed environment not initialized. "
                "Call init_distributed_env() before creating trainers."
            )
        if not self.context.logger_initialized:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Logger not initialized. "
                "Call init_global_logger() before creating trainers."
            )

        # Convenient access to distributed parameters
        self.rank = self.context.rank
        self.world_size = self.context.world_size
        self.local_rank = self.context.local_rank
        self.master_addr = self.context.master_addr
        self.master_port = self.context.master_port
        self.platform = self.context.platform

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

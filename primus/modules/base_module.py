###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import builtins
import os
from abc import ABC, abstractmethod

from primus.core.launcher.config import PrimusConfig
from primus.core.utils import checker, logger
from primus.core.utils.global_vars import (
    get_cli_args,
    get_target_platform,
    set_global_variables,
)

from .module_utils import debug_rank_all, set_logging_rank


class BaseModule(ABC):
    """
    Base class for all Primus modules (trainers, tools, etc.).

    Responsibilities:
        - Initialize distributed training environment
        - Setup logging for distributed workers
        - Provide access to platform and configuration
        - Manage environment variables for distributed training
    """

    def __init__(
        self,
        module_name: str,
        primus_config: PrimusConfig,
        module_rank: int = None,
        module_world_size: int = None,
        module_master_addr: str = None,
        module_master_port: int = None,
    ):
        """
        Initialize BaseModule.

        Args:
            module_name: Name of the module (e.g., "pre_trainer")
            primus_config: Primus configuration object
            module_rank: Rank of this worker (auto-detected from RANK env var if None)
            module_world_size: Total number of workers (auto-detected from WORLD_SIZE if None)
            module_master_addr: Master node address (auto-detected from MASTER_ADDR if None)
            module_master_port: Master node port (auto-detected from MASTER_PORT if None)
        """
        self.module_name = module_name
        self.primus_config = primus_config
        self.module_config = primus_config.get_module_config(module_name)

        # Set config into the global vars of worker process
        set_global_variables(primus_config)
        self.platform = get_target_platform()
        self.cli_args = get_cli_args()

        # Initialize distributed parameters
        self._init_distributed_params(module_rank, module_world_size, module_master_addr, module_master_port)

        # Setup logger for worker
        self.setup_worker_logger(self.module_rank, self.module_world_size)

    def _init_distributed_params(
        self,
        module_rank: int = None,
        module_world_size: int = None,
        module_master_addr: str = None,
        module_master_port: int = None,
    ):
        """
        Initialize distributed training parameters.

        Auto-detects from environment variables if not explicitly provided.
        Sets up environment variables for downstream frameworks.
        """
        # Auto-detect from environment if not provided
        module_rank = module_rank if module_rank is not None else int(os.environ.get("RANK", 0))
        module_world_size = (
            module_world_size if module_world_size is not None else int(os.environ.get("WORLD_SIZE", 1))
        )
        module_master_addr = (
            module_master_addr
            if module_master_addr is not None
            else os.environ.get("MASTER_ADDR", "localhost")
        )
        module_master_port = (
            module_master_port
            if module_master_port is not None
            else int(os.environ.get("MASTER_PORT", 29500))
        )

        # Validate distributed parameters
        if module_rank > 0 and module_master_addr == "localhost":
            checker.check_true(
                False,
                msg=f"Worker with rank {module_rank} must have a valid master address (not 'localhost')",
            )

        # Store distributed info
        self.module_rank = module_rank
        self.module_world_size = module_world_size
        # For rank 0, use actual platform address; for others, use provided address
        self.module_master_addr = self.platform.get_addr() if module_rank == 0 else module_master_addr
        self.module_master_port = module_master_port

        # Calculate local rank
        self.num_gpus_per_node = self.platform.get_gpus_per_node()
        self.module_local_rank = self.module_rank % self.num_gpus_per_node

        # Set environment variables for downstream frameworks (Megatron, PyTorch, etc.)
        os.environ["MASTER_ADDR"] = str(self.module_master_addr)
        os.environ["MASTER_PORT"] = str(self.module_master_port)
        os.environ["WORLD_SIZE"] = str(self.module_world_size)
        os.environ["RANK"] = str(self.module_rank)
        os.environ["LOCAL_RANK"] = str(self.module_local_rank)

    @abstractmethod
    def init(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def setup(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def setup_worker_logger(self, rank: int, world_size: int):
        """
        Setup distributed logging for this worker.

        Args:
            rank: Worker rank
            world_size: Total number of workers
        """
        # Determine log levels (use sink_level if specified, otherwise use defaults)
        sink_level = getattr(self.module_config, "sink_level", None)
        file_sink_level = getattr(self.module_config, "file_sink_level", sink_level or "INFO")
        stderr_sink_level = getattr(self.module_config, "stderr_sink_level", sink_level or "INFO")

        # Create logger configuration
        logger_cfg = logger.LoggerConfig(
            exp_root_path=self.primus_config.exp_root_path,
            work_group=self.primus_config.exp_meta_info["work_group"],
            user_name=self.primus_config.exp_meta_info["user_name"],
            exp_name=self.primus_config.exp_meta_info["exp_name"],
            module_name=self.module_name,
            file_sink_level=file_sink_level,
            stderr_sink_level=stderr_sink_level,
            node_ip=self.platform.get_addr(),
            rank=rank,
            world_size=world_size,
        )
        logger.setup_logger(logger_cfg, is_head=False)

        # Set logging rank for rank-aware logging (log_rank_0, etc.)
        set_logging_rank(rank, world_size)

        # Monkey patch print function for distributed logging
        self.original_print = builtins.print
        builtins.print = debug_rank_all

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

    # Methods for accessing distributed info
    def get_module_master_address(self) -> str:
        """Get master node address."""
        return self.module_master_addr

    def get_module_master_port(self) -> int:
        """Get master node port."""
        return self.module_master_port

    def get_module_rank(self) -> int:
        """Get this worker's rank."""
        return self.module_rank

    def get_module_world_size(self) -> int:
        """Get total number of workers."""
        return self.module_world_size

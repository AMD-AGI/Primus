###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Global runtime context for Primus training.

This module provides a singleton RuntimeContext that holds global state
initialized once at the start of training.
"""

from typing import Optional


class RuntimeContext:
    """
    Singleton class to hold global runtime context.

    This context is initialized once at the start of training and provides
    access to distributed training parameters, platform information, and
    configuration without requiring re-initialization.

    Attributes:
        rank: Global rank of this process
        world_size: Total number of processes
        local_rank: Local rank on this node
        master_addr: Master node address
        master_port: Master node port
        platform: Platform object (GPU info, node info, etc.)
        primus_config: Global Primus configuration
        logger_initialized: Whether global logger has been initialized
    """

    _instance: Optional["RuntimeContext"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if RuntimeContext._initialized:
            return

        # Distributed training parameters
        self.rank: int = 0
        self.world_size: int = 1
        self.local_rank: int = 0
        self.master_addr: str = "localhost"
        self.master_port: int = 29500

        # Platform and configuration
        self.platform = None
        self.primus_config = None

        # Logging state
        self.logger_config = None  # Store current LoggerConfig for re-configuration
        self.file_sink_handlers: list = []  # Track file sink handler IDs for updates

        # State flags
        self.logger_initialized: bool = False
        self.distributed_initialized: bool = False

        RuntimeContext._initialized = True

    @classmethod
    def get_instance(cls) -> "RuntimeContext":
        """Get the singleton instance of RuntimeContext."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
        cls._initialized = False

    def set_distributed_params(
        self,
        rank: int,
        world_size: int,
        local_world_size: int,
        local_rank: int,
        master_addr: str,
        master_port: int,
    ):
        """Set distributed training parameters."""
        self.rank = rank
        self.world_size = world_size
        self.local_world_size = local_world_size
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.distributed_initialized = True

    def set_platform(self, platform):
        """Set platform object."""
        self.platform = platform

    def set_primus_config(self, primus_config):
        """Set global Primus configuration."""
        self.primus_config = primus_config

    def is_rank_zero(self) -> bool:
        """Check if this is rank 0."""
        return self.rank == 0

    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1

    def __repr__(self) -> str:
        return (
            f"RuntimeContext(rank={self.rank}, world_size={self.world_size}, "
            f"local_rank={self.local_rank}, master_addr={self.master_addr}, "
            f"master_port={self.master_port}, local_world_size={self.local_world_size})"
        )

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod


class BaseModuleProfiler(ABC):
    """Abstract base class for transformer-like module profiler.
    Provides both estimated and measured statistics.
    """

    # def __init__(self, name: str):
    #     self.name = name

    def __init__(self, config, sub_profilers=None):
        self.config = config
        self.sub_profilers = sub_profilers

    # -------- Parameter related --------
    @abstractmethod
    def estimated_num_params(self) -> int:
        """Return estimated parameter count (based on formula)."""
        raise NotImplementedError

    @abstractmethod
    def measured_num_params(self) -> int:
        """Return measured parameter count (from real tensors)."""
        raise NotImplementedError

    # -------- Memory related --------
    @abstractmethod
    def estimated_memory(self, batch_size: int, seq_len: int) -> int:
        """Return estimated memory usage in bytes (activations)."""
        raise NotImplementedError

    @abstractmethod
    def measured_memory(self, batch_size: int, seq_len: int) -> int:
        """Return measured memory usage in bytes (via profiler/runtime stats)."""
        raise NotImplementedError

    # -------- Performance related --------
    @abstractmethod
    def estimated_forward_time(self, batch_size: int, seq_len: int) -> int:
        """Return estimated forward latency for forward pass in milliseconds."""
        raise NotImplementedError

    @abstractmethod
    def estimated_backward_time(self, batch_size: int, seq_len: int) -> int:
        """Return estimated latency for backward pass in milliseconds."""
        raise NotImplementedError

    @abstractmethod
    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        """Return measured forward latency in milliseconds."""
        raise NotImplementedError

    @abstractmethod
    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        """Return measured backward latency in milliseconds."""
        raise NotImplementedError

    # -------- Debugging / summary --------
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

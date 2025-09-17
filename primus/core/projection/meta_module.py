###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Union


class BaseMetaModule(ABC):
    """Abstract base class for transformer-like modules.
    Provides both estimated and measured statistics.
    """

    def __init__(self, name: str):
        self.name = name

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
        """Return estimated memory usage in bytes (activations + params)."""
        raise NotImplementedError

    @abstractmethod
    def measured_memory(self, batch_size: int, seq_len: int) -> int:
        """Return measured memory usage in bytes (via profiler/runtime stats)."""
        raise NotImplementedError

    # -------- Performance related --------
    @abstractmethod
    def estimated_forward_time(self, batch_size: int, seq_len: int) -> int:
        """Return estimated FLOPs for forward pass."""
        raise NotImplementedError

    @abstractmethod
    def estimated_backward_time(self, batch_size: int, seq_len: int) -> int:
        """Return estimated FLOPs for backward pass."""
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


@dataclass
class ModuleSpec:
    module: Union[Tuple, type]
    params: dict = field(default_factory=lambda: {})
    submodules: type = None


@dataclass
class TransformerLayerSubmodules:
    pass
    # input_layernorm: Union[ModuleSpec, type] = IdentityOp
    # input_layernorm: Optional[Union[ModuleSpec, type]] = None


@dataclass
class SelfAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler

from .core_attention import CoreAttentionProfiler
from .layer_norm import LayerNormProfiler
from .linear import LinearProfiler
from .linear_qkv import LinearQKVProfiler
from .rotary_embedding import RotaryEmbeddingProfiler


def get_self_attention_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=TransformerLayerProfiler,
        config=config,
        sub_profiler_specs={
            "linear_qkv": LinearQKVProfiler,
            "q_layernorm": LayerNormProfiler,
            "k_layernorm": LayerNormProfiler,
            "rotary_emb": RotaryEmbeddingProfiler,
            "core_attention": CoreAttentionProfiler,
            "linear_proj": LinearProfiler,
        },
    )


class SelfAttentionProfiler(BaseModuleProfiler):
    def __init__(self, name: str):
        self.name = name

    # -------- Parameter related --------
    @abstractmethod
    def estimated_num_params(self) -> int:
        # embedding + layers + outputlayer
        return 0

    @abstractmethod
    def measured_num_params(self) -> int:
        return 0

    # -------- Memory related --------
    @abstractmethod
    def estimated_memory(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def measured_memory(self, batch_size: int, seq_len: int) -> int:
        return 0

    # -------- Performance related --------
    @abstractmethod
    def estimated_forward_time(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def estimated_backward_time(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        return 0

    @abstractmethod
    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        return 0

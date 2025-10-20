###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from .embedding import EmbeddingProfiler
from .layer_norm import LayerNormProfiler
from .loss import LossProfiler
from .output_layer import OutputLayerProfiler
from .transformer_layer import get_transformer_layer_profiler_spec

# @dataclass
# class ModelParallelConfig:
#     """Base configuration for Megatron Core

#     The initialization function has an argument for each parameter.
#     """


def get_language_model_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=LanguageModelProfiler,
        config=config,
        sub_profiler_specs={
            "embedding": EmbeddingProfiler,
            "transformer_layers": get_transformer_layer_profiler_spec(config),
            "final_layernorm": LayerNormProfiler,
            "output_layer": OutputLayerProfiler,
            "calc_loss": LossProfiler,
        },
    )


# language profiler spec -> build_profiler() -> language profiler -> run profiling methods
class LanguageModelProfiler(BaseModuleProfiler):
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

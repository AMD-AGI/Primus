###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig

from .attention import AttentionProfiler
from .dense_mlp import DenseMLPProfiler
from .layer_norm import LayerNormProfiler
from .moe_mlp import MoEMLPProfiler
from .router import RouterProfiler

# Transformer Layer Data Flow
#
#             +----------------+
#             |     Input      |
#             +----------------+
#                     | ----------------------
#        +-------------------------+         |
#        |     Input LayerNorm     |         |
#        +-------------------------+         |
#                     |                      |
#        +------------------------+          |
#        |     Self-Attention     |          |
#        +------------------------+          |
#                     |                      |
#            +-----------------+             |
#            |     Dropout     |             |
#            +-----------------+             |
#                     |                      |
#                     o ---------------------|
#         +----------------------+
#         |     Residual Add     |
#         +----------------------+
#                     | ----------------------
#        +-------------------------+         |
#        |    Pre-mlp LayerNorm    |         |
#        +-------------------------+         |
#                     |                      |
#              +-------------+               |
#              |     MLP     |               |
#              +-------------+               |
#                     |                      |
#            +-----------------+             |
#            |     Dropout     |             |
#            +-----------------+             |
#                     |                      |
#                     o ---------------------|
#         +----------------------+
#         |     Residual Add     |
#         +----------------------+
#                     |
#             +----------------+
#             |     Output      |
#             +----------------+


class DenseTransformerLayerProfiler(BaseModuleProfiler):
    def estimated_num_params(self, rank: int | None = None) -> int:
        return (self.sub_profilers["layer_norm"].estimated_num_params(rank) * 2 +
                self.sub_profilers["self_attention"].estimated_num_params(rank) +
                self.sub_profilers["mlp"].estimated_num_params(rank))

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return (self.sub_profilers["layer_norm"].estimated_activation_memory(batch_size, seq_len) * 2 +
                self.sub_profilers["self_attention"].estimated_activation_memory(batch_size, seq_len) +
                self.sub_profilers["mlp"].estimated_activation_memory(batch_size, seq_len))


class MoETransformerLayerProfiler(BaseModuleProfiler):
    def estimated_num_params(self, rank: int | None = None) -> int:
        return (self.sub_profilers["layer_norm"].estimated_num_params(rank) * 2 +
                self.sub_profilers["self_attention"].estimated_num_params(rank) +
                self.sub_profilers["mlp"].estimated_num_params(rank) +
                self.sub_profilers["router"].estimated_num_params(rank))

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return (self.sub_profilers["layer_norm"].estimated_activation_memory(batch_size, seq_len) * 2 +
                self.sub_profilers["self_attention"].estimated_activation_memory(batch_size, seq_len) +
                self.sub_profilers["mlp"].estimated_activation_memory(batch_size, seq_len) +
                self.sub_profilers["router"].estimated_activation_memory(batch_size, seq_len))


def get_dense_transformer_layer_profiler_spec(config: TrainingConfig) -> "ModuleProfilerSpec":
    return ModuleProfilerSpec(
        profiler=DenseTransformerLayerProfiler,
        config=config,
        sub_profiler_specs={
            "layer_norm": LayerNormProfiler,
            "self_attention": AttentionProfiler,
            "mlp": DenseMLPProfiler,
        },
    )


def get_moe_transformer_layer_profiler_spec(config: TrainingConfig) -> "ModuleProfilerSpec":
    return ModuleProfilerSpec(
        profiler=MoETransformerLayerProfiler,
        config=config,
        sub_profiler_specs={
            "layer_norm": LayerNormProfiler,
            "self_attention": AttentionProfiler,
            "router": RouterProfiler,
            "mlp": MoEMLPProfiler,
        },
    )

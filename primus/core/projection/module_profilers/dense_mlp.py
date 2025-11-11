###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig


class DenseMLPProfiler(BaseModuleProfiler):
    def estimated_num_params(self, rank: int | None = None) -> int:
        # For SwiGLU: 3 projections (gate, up, down)
        # For standard FFN: 2 projections (up, down)
        num_ffn_projections = 3 if self.config.model_config.swiglu else 2
        return (
            self.config.model_config.hidden_size
            * self.config.model_config.ffn_hidden_size
            * num_ffn_projections
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        num_tokens = (
            batch_size
            * seq_len
            // self.config.model_parallel_config.tensor_model_parallel_size
            // self.config.model_parallel_config.context_model_parallel_size
        )
        # Calculate memory at different stages and take maximum
        input_memory = num_tokens * self.config.model_config.hidden_size * 2  # bf16
        
        # Memory after first projection(s)
        if self.config.model_config.swiglu:
            # Need to store both gate and up projections for backward
            intermediate_memory = 2 * num_tokens * self.config.model_config.ffn_hidden_size * 2  # bf16
        else:
            intermediate_memory = num_tokens * self.config.model_config.ffn_hidden_size * 2  # bf16
        
        output_memory = num_tokens * self.config.model_config.hidden_size * 2  # bf16
        
        # Peak memory is input + intermediate (both needed for backward)
        return input_memory + intermediate_memory + output_memory


def get_dense_mlp_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=DenseMLPProfiler,
        config=config,
        sub_profiler_specs=None,
    )

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig


class MoEMLPProfiler(BaseModuleProfiler):
    def estimated_num_params(self) -> int:
        if self.config.model_config.moe_ffn_hidden_size is not None:
            moe_ffn = self.config.model_config.moe_ffn_hidden_size
        else:
            moe_ffn = self.config.model_config.ffn_hidden_size
        
        # For SwiGLU: 3 projections per expert (gate, up, down)
        # For standard FFN: 2 projections per expert (up, down)
        num_ffn_projections = 3 if self.config.model_config.swiglu else 2
        per_expert_params = num_ffn_projections * self.config.model_config.hidden_size * moe_ffn
        all_experts_params = (self.config.model_config.num_experts *
                              per_expert_params //
                              self.config.model_parallel_config.expert_model_parallel_size)

        # Shared experts (if any)
        shared_sz = 0
        if self.config.model_config.moe_shared_expert_intermediate_size is not None:
            shared_sz = self.config.model_config.moe_shared_expert_intermediate_size
        shared_params = num_ffn_projections * self.config.model_config.hidden_size * shared_sz
        
        return all_experts_params + shared_params

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        num_tokens = (batch_size * seq_len //
                self.config.model_parallel_config.tensor_model_parallel_size //
                self.config.model_parallel_config.context_model_parallel_size *
                self.config.model_config.moe_router_topk)
        total = 0
        # First Gemm
        total += num_tokens * self.config.model_config.hidden_size * 2  # bf16
        # Activation layer
        # TODO: swiglu scalue
        swiglu_scale = 1
        if self.config.model_config.swiglu:
            swiglu_scale = 2
        total += (num_tokens * self.config.model_config.moe_ffn_hidden_size *
                  swiglu_scale) * 2  # bf16
        # Second Gemm
        total += num_tokens * self.config.model_config.moe_ffn_hidden_size * 2  # bf16

        denom = self.config.model_config.hidden_size * batch_size * seq_len * 2
        return total


def get_moe_mlp_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=MoEMLPProfiler,
        config=config,
        sub_profiler_specs=None,
    )
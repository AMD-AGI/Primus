###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler


class OutputLayerProfiler(BaseModuleProfiler):
    def estimated_num_params(self, rank: int | None = None) -> int:
        return self.config.model_config.padded_vocab_size * self.config.model_config.hidden_size

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return  (batch_size * seq_len //
                 self.config.model_parallel_config.tensor_model_parallel_size //
                 self.config.model_parallel_config.context_model_parallel_size * 
                 self.config.model_config.padded_vocab_size * 2)  # bf16

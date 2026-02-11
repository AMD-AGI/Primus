from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional


class LayerNormProfiler(BaseModuleProfiler):
    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        return self.config.model_config.hidden_size * 2  # gamma and beta

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return (
            batch_size
            * seq_len
            // self.config.model_parallel_config.tensor_model_parallel_size
            // self.config.model_parallel_config.context_model_parallel_size
            * self.config.model_config.hidden_size
            * 2
        )  # bf16

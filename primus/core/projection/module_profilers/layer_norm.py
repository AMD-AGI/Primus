###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler


class LayerNormProfiler(BaseModuleProfiler):
    def estimated_params_memory(self) -> int:
        # embedding + layers + outputlayer
        return 0

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return 0

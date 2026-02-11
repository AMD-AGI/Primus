from typing import Optional

from primus.core.projection.base_module_profiler import BaseModuleProfiler

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


class LossProfiler(BaseModuleProfiler):
    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        return 0

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return 1

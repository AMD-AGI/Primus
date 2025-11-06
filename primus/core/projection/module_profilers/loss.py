###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig


class LossProfiler(BaseModuleProfiler):
    def estimated_num_params(self, rank: int | None = None) -> int:
        return 0

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return 1

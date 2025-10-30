###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.profiler_spec import ModuleProfilerSpec
from primus.core.projection.training_config import TrainingConfig


class DenseMLPProfiler(BaseModuleProfiler):
    def estimated_num_params(self) -> int:
        # embedding + layers + outputlayer
        return 0

    def estimated_memory(self, batch_size: int, seq_len: int) -> int:
        return 0


def get_dense_mlp_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=DenseMLPProfiler,
        config=config,
        sub_profiler_specs=None,
    )
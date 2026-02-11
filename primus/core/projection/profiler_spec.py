###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

from primus.core.projection.base_module_profiler import BaseModuleProfiler
from primus.core.projection.training_config import TrainingConfig


@dataclass
class ModuleProfilerSpec:
    profiler: Type[BaseModuleProfiler]
    config: Type[TrainingConfig]
    sub_profiler_specs: Optional[
        Dict[str, Union[Type[BaseModuleProfiler], "ModuleProfilerSpec", None]]
    ] = field(default_factory=lambda: {})

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from dataclasses import dataclass
from typing import Union

from primus.core.projection.base_module_profiler import BaseModuleProfiler


@dataclass
class MLPSpec:
    """
    Submodules used inside the MLP block.

    Includes:
      - dense MLP components (two linear layers)
      - sparse MoE MLP components (router, dispatch/combine, grouped GEMM, shared experts)
    """

    # dense MLP
    linear_fc1: Union[ModuleProfilerSpec, BaseModuleProfiler] = IdentityModuleProfiler
    linear_fc2: Union[ModuleProfilerSpec, BaseModuleProfiler] = IdentityModuleProfiler
    # sparse MoE MLP
    router: Union[ModuleProfilerSpec, BaseModuleProfiler] = IdentityModuleProfiler
    dispatcher_dispach: Union[ModuleProfilerSpec, BaseModuleProfiler] = IdentityModuleProfiler
    dispatcher_combine: Union[ModuleProfilerSpec, BaseModuleProfiler] = IdentityModuleProfiler
    grouped_gemm: Union[ModuleProfilerSpec, BaseModuleProfiler] = IdentityModuleProfiler
    shared_experts: Union[ModuleProfilerSpec, BaseModuleProfiler] = IdentityModuleProfiler


###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from dataclasses import dataclass
from typing import Union

from primus.core.projection.base_module_profiler import BaseModuleProfiler

from .grouped_mlp import GroupedMLPProfiler
from .router import RouterProfiler
from .token_dispatcher import TokenDispatcherProfiler


# TODO: shared experts
def get_self_attention_profiler_spec(config: TrainingConfig) -> ModuleProfilerSpec:
    return ModuleProfilerSpec(
        profiler=TransformerLayerProfiler,
        config=config,
        sub_profiler_specs={
            "router": RouterProfiler,
            "token_dispatcher": TokenDispatcherProfiler,
            "grouped_mlp": GroupedMLPProfiler,
        },
    )


class SelfAttentionProfiler(BaseModuleProfiler):
    def __init__(self, name: str):
        self.name = name

    # -------- Parameter related --------
    @abstractmethod
    def estimated_num_params(self) -> int:
        # embedding + layers + outputlayer
        return 0

    @abstractmethod
    def measured_num_params(self) -> int:
        return 0

    # -------- Memory related --------
    @abstractmethod
    def estimated_memory(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def measured_memory(self, batch_size: int, seq_len: int) -> int:
        return 0

    # -------- Performance related --------
    @abstractmethod
    def estimated_forward_time(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def estimated_backward_time(self, batch_size: int, seq_len: int) -> int:
        return 0

    @abstractmethod
    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        return 0

    @abstractmethod
    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        return 0

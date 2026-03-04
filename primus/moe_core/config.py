###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEParallelConfig:
    """Parallel topology required by MoE runtime."""

    tensor_parallel_size: int
    expert_parallel_size: int
    context_parallel_size: int


@dataclass(frozen=True)
class RouterRuntimeConfig:
    """Common router runtime knobs used by fused/non-fused implementations."""

    num_experts: int
    router_topk: int
    router_num_groups: int | None
    router_group_topk: int | None
    router_score_function: str
    router_topk_scaling_factor: float


@dataclass(frozen=True)
class DispatchRuntimeConfig:
    """Dispatcher runtime knobs exposed to backend adapters."""

    num_experts: int
    router_topk: int
    expert_capacity_factor: float | None
    permute_fusion: bool
    permute_max_token_num: int
    num_worst_tokens: int
    use_comm_stream: bool
    num_cu: int
    use_cuda_num_tokens_per_expert: bool
    async_finish: bool = True
    allocate_on_comm_stream: bool = True

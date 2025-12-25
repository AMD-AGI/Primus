###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Primus-Turbo Async Tensor Parallel Patch

This patch installs Primus-Turbo async tensor-parallel collectives when
``primus_turbo.use_turbo_async_tp`` is enabled in the TorchTitan job config.

The original logic lives inside ``TorchTitanPretrainTrainer.patch_torch_async_tp``.
It is now also expressed as a backend patch so it can be managed via the
Primus patch system.
"""

from typing import Any, Optional

from primus.backends.torchtitan.patches.turbo.utils import get_primus_turbo_config
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "torchtitan.primus_turbo.turbo_async_tp",
    backend="torchtitan",
    phase="before_train",
    description="Use Primus-Turbo async tensor-parallel collectives",
    condition=lambda ctx: (
        (cfg := get_primus_turbo_config(ctx)) is not None
        and getattr(cfg, "enable_primus_turbo", False)
        and getattr(cfg, "use_turbo_async_tp", False)
        and (args := get_args(ctx)) is not None
        and getattr(getattr(args, "parallelism", None), "enable_async_tensor_parallel", False)
    ),
)
def patch_turbo_async_tp(ctx: PatchContext) -> None:
    """
    Monkey patch fused async TP collectives to use Primus-Turbo implementations.
    """
    import primus_turbo.pytorch as pt
    import torch
    import torch.distributed._symmetric_memory as symm_module
    import torch.distributed.distributed_c10d as c10d

    from primus.backends.torchtitan.tools.utils import get_backend_stream

    def _fused_all_gather_matmul_impl(
        mm_out_op: torch._ops.OpOverload,
        A_shard: torch.Tensor,
        Bs: list[torch.Tensor],
        A_scale: Optional[torch.Tensor],
        kwargs_list: list[dict[str, Any]],
        out_dtypes: list[Optional[torch.dtype]],
        gather_dim: int,
        group_name: str,
        return_A: bool,
    ) -> tuple[Optional[torch.Tensor], list[torch.Tensor]]:
        assert A_scale is None, "fused_all_gather_matmul not support for fp8"

        layouts = ["NN" for _ in range(len(Bs))]
        group = c10d._resolve_process_group(group_name)
        gemm_streams = [torch.cuda.current_stream()]
        comm_streams = get_backend_stream(size=group.size() - 1, priority=0, prefix="comm")
        copy_streams = get_backend_stream(size=1, priority=0, prefix="copy")

        A, outputs = pt.ops.fused_all_gather_matmul(
            A_shard,
            Bs,
            layouts,
            gather_dim=gather_dim,
            group_name=group_name,
            gemm_streams=gemm_streams,
            comm_streams=comm_streams,
            copy_streams=copy_streams,
            comm_method="pipeline",
            num_splits=4,
            return_A=return_A,
            out_dtypes=out_dtypes,
        )

        return A, outputs

    def _fused_matmul_reduce_scatter_impl(
        mm_out_op: torch._ops.OpOverload,
        A: torch.Tensor,
        B: torch.Tensor,
        kwargs: dict[str, Any],
        out_dtype: Optional[torch.dtype],
        reduce_op: str,
        scatter_dim: int,
        group_name: str,
    ) -> torch.Tensor:
        comm_method = "pipeline"
        group = c10d._resolve_process_group(group_name)

        if comm_method == "pipeline":
            gemm_streams = [torch.cuda.current_stream()]
            comm_streams = get_backend_stream(size=group.size(), priority=0, prefix="comm")
        elif comm_method == "tile":  # pragma: no cover - future extension
            gemm_streams = []
            comm_streams = []
        else:  # pragma: no cover - defensive
            raise ValueError(f"Only pipeline and tile supported, but {comm_method} provided")

        rs_output = pt.ops.fused_matmul_reduce_scatter(
            A,
            B,
            layout="NN",
            reduce_op=reduce_op,
            scatter_dim=scatter_dim,
            group_name=group_name,
            gemm_streams=gemm_streams,
            comm_streams=comm_streams,
            comm_method=comm_method,
            num_splits=4,
            out_dtype=out_dtype,
        )
        return rs_output.contiguous()

    symm_module._fused_all_gather_matmul_impl = _fused_all_gather_matmul_impl
    symm_module._fused_matmul_reduce_scatter_impl = _fused_matmul_reduce_scatter_impl
    log_rank_0(
        "[Patch:torchtitan.primus_turbo.turbo_async_tp] "
        "Primus-Turbo async tensor-parallel collectives successfully installed.",
    )

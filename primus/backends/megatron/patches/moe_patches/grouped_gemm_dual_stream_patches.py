###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Grouped GEMM Dual-Stream Backward Patch

Monkey-patches ``grouped_gemm.ops.GroupedGemm`` so that the activation-
gradient (dX) and weight-gradient (dW) GEMMs in the backward pass run on
two independent CUDA streams, overlapping their execution.

The stock ``GroupedGemm.backward`` serializes dX and dW on the same stream,
leaving ~40-50 % of each grouped-GEMM burst's wall time on the table.

Controlled by ``PRIMUS_MOE_EXPERT_GEMM_BWD_DUAL_STREAM`` (default ``"1"``).
Set to ``"0"`` to disable.
"""

import os
from typing import List, Optional

import torch

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def _dual_stream_enabled() -> bool:
    return os.getenv("PRIMUS_MOE_EXPERT_GEMM_BWD_DUAL_STREAM", "0") == "1"


def _make_dual_stream_grouped_gemm(backend):
    """Build a new ``torch.autograd.Function`` that wraps *backend.gmm*
    with a dual-stream backward."""

    class DualStreamGroupedGemm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, a, b, batch_sizes, trans_b):
            assert torch.count_nonzero(batch_sizes) != 0, (
                "Input batch_sizes should not be all zeros!"
            )
            ctx.save_for_backward(a, b, batch_sizes)
            ctx.trans_b = trans_b
            return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

        @staticmethod
        def backward(ctx, grad):
            grad = grad.contiguous()
            a, b, batch_sizes = ctx.saved_tensors
            trans_b = ctx.trans_b

            use_dual = (
                _dual_stream_enabled()
                and grad.is_cuda
                and torch.cuda.is_available()
            )

            if not use_dual:
                agrad = None
                bgrad = None
                if ctx.needs_input_grad[0]:
                    agrad = backend.gmm(
                        grad, b, batch_sizes,
                        trans_a=False, trans_b=not trans_b,
                    )
                if ctx.needs_input_grad[1]:
                    lhs, rhs = (grad, a) if trans_b else (a, grad)
                    bgrad = backend.gmm(
                        lhs, rhs, batch_sizes,
                        trans_a=True, trans_b=False,
                    )
                return agrad, bgrad, None, None

            dev = grad.device
            current = torch.cuda.current_stream(device=dev)
            stream_da = torch.cuda.Stream(device=dev)
            stream_db = torch.cuda.Stream(device=dev)

            stream_da.wait_stream(current)
            stream_db.wait_stream(current)

            agrad: Optional[torch.Tensor] = None
            bgrad: Optional[torch.Tensor] = None

            if ctx.needs_input_grad[0]:
                with torch.cuda.stream(stream_da):
                    agrad = backend.gmm(
                        grad, b, batch_sizes,
                        trans_a=False, trans_b=not trans_b,
                    )

            if ctx.needs_input_grad[1]:
                with torch.cuda.stream(stream_db):
                    lhs, rhs = (grad, a) if trans_b else (a, grad)
                    bgrad = backend.gmm(
                        lhs, rhs, batch_sizes,
                        trans_a=True, trans_b=False,
                    )

            current.wait_stream(stream_da)
            current.wait_stream(stream_db)

            return agrad, bgrad, None, None

    return DualStreamGroupedGemm


@register_patch(
    "megatron.moe.grouped_gemm_dual_stream",
    backend="megatron",
    phase="before_train",
    description="Overlap dX/dW grouped-GEMM backward on two CUDA streams",
    condition=lambda ctx: _dual_stream_enabled(),
)
def patch_grouped_gemm_dual_stream(ctx: PatchContext):
    log_rank_0(
        "[Patch:megatron.moe.grouped_gemm_dual_stream] "
        "Patching grouped_gemm backward with dual-stream dX||dW overlap ..."
    )

    try:
        import grouped_gemm
    except ImportError:
        log_rank_0(
            "[Patch:megatron.moe.grouped_gemm_dual_stream] "
            "grouped_gemm not installed; skipping."
        )
        return

    gg_backend = grouped_gemm.backend
    DualStreamGG = _make_dual_stream_grouped_gemm(gg_backend)

    original_cls = grouped_gemm.ops.GroupedGemm
    grouped_gemm.ops.GroupedGemm = DualStreamGG

    original_gmm = grouped_gemm.ops.gmm

    def patched_gmm(a, b, batch_sizes, trans_b=False):
        return DualStreamGG.apply(a, b, batch_sizes, trans_b)

    grouped_gemm.ops.gmm = patched_gmm

    from megatron.core.transformer.moe import grouped_gemm_util as gg_util
    if gg_util.ops is not None:
        gg_util.ops.GroupedGemm = DualStreamGG
        gg_util.ops.gmm = patched_gmm

    log_rank_0(
        "[Patch:megatron.moe.grouped_gemm_dual_stream] "
        f"  Replaced grouped_gemm.ops.GroupedGemm ({original_cls.__name__}) "
        f"-> DualStreamGroupedGemm"
    )
    log_rank_0(
        "[Patch:megatron.moe.grouped_gemm_dual_stream] "
        "  Done. dX and dW will execute on separate CUDA streams."
    )

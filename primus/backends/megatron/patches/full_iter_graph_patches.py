###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Full-iteration CUDA graph (``--cuda_graph_impl full_iteration``) on ROCm: pre-warm BLAS handles.

Megatron's FullCudaGraphWrapper warms up on the default stream and then captures on a separate
stream. The first BLAS GEMM of the captured backward (autograd device thread) then creates a new
hipBLASLt handle inside the capture, and ``hipblasLtCreate`` touches the legacy stream:
"operation would make the legacy stream depend on a capturing blocking stream" (hipblaslt.cpp:415),
after which the run hangs. Right before the capture this patch runs a few tiny GEMMs forward AND
backward (so the autograd device thread is covered too) on the shared capture stream, in every dtype
and op family the model uses (mm / addmm / bmm, bf16 / fp32), so every handle and per-stream
workspace exists before capture starts. No model state is touched.
"""

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _prewarm(stream):
    dev = torch.cuda.current_device()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for dt in (torch.bfloat16, torch.float32):
            a = torch.randn(256, 256, device=dev, dtype=dt, requires_grad=True)
            b = torch.randn(256, 256, device=dev, dtype=dt, requires_grad=True)
            c = torch.randn(4, 256, 256, device=dev, dtype=dt, requires_grad=True)
            bias = torch.randn(256, device=dev, dtype=dt, requires_grad=True)
            y = (a @ b).sum() + torch.nn.functional.linear(a, b, bias).sum() + torch.bmm(c, c).sum()
            y = y + torch.matmul(a.unsqueeze(0).expand(3, -1, -1), b).sum()
            y.backward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()


@register_patch(
    "megatron.full_iteration_graph.prewarm",
    backend="megatron",
    phase="before_train",
    description="Pre-warm BLAS handles/workspaces on the full-iteration graph capture stream (ROCm)",
    priority=60,
    condition=lambda ctx: True,  # only acts when FullCudaGraphWrapper reaches its capture iteration
)
def patch_full_iteration_prewarm(ctx: PatchContext) -> None:
    del ctx
    import megatron.core.full_cuda_graph as fcg

    W = fcg.FullCudaGraphWrapper
    if getattr(W.__call__, "_primus_prewarm", False):
        return
    orig_call = W.__call__

    def __call__(self, *args, **kwargs):
        stage = "validation" if kwargs.get("forward_only") else "training"
        if self.curr_iter(stage) == self.cuda_graph_warmup_steps and W.cuda_graph[stage] is None:
            _prewarm(fcg.get_shared_capture_stream())
            log_rank_0(f"[Patch:full_iteration_graph] BLAS handles pre-warmed on the capture stream ({stage})")
        return orig_call(self, *args, **kwargs)

    __call__._primus_prewarm = True
    W.__call__ = __call__
    log_rank_0("[Patch:full_iteration_graph] FullCudaGraphWrapper pre-capture BLAS warm-up installed")

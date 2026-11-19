###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL launches default to torch's CURRENT stream instead of the null stream.

FlyDSL launchers declare ``stream: fx.Stream = fx.Stream(None)`` and ``Stream.__c_abi_spec__`` turns
``None`` into ``c_void_p(0)`` -- the legacy null stream -- whatever torch's current stream is. Most
callers (Turbo gfx1250 sparse-MLA, GEMM, quant kernels) never pass a stream. In eager single-stream
runs torch's current stream is also 0, so nothing changes; but under CUDA graph capture (side
stream) those kernels are executed once at capture time and NOT recorded, so every replay reuses a
stale output. Seen in the DSv4 full-iteration graph: V4 attention output wrong in every layer
after the first, grad norm NaN. Same root cause as kimi_k3/kda_kernels/_flydsl_v1/_stream.py,
fixed here once for all FlyDSL launchers: ``None`` resolves to ``torch.cuda.current_stream()``.
An explicit stream passed by a caller still wins.
"""

import ctypes

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _flydsl_available() -> bool:
    try:
        import flydsl.expr.typing  # noqa: F401

        return True
    except Exception:
        return False


@register_patch(
    "flydsl.stream_default_current",
    backend="megatron",
    phase="before_train",
    description="FlyDSL Stream(None) -> torch current stream (graph capture / side streams)",
    priority=10,
    condition=lambda ctx: _flydsl_available(),
)
def patch_flydsl_default_stream(ctx: PatchContext) -> None:
    del ctx
    import torch
    from flydsl.expr import typing as fxt

    S = fxt.Stream
    if getattr(S, "_primus_current_stream", False):
        return

    def __c_abi_spec__(self):
        def fill(a, s):
            raw = a.value if hasattr(a, "_is_stream_param") else a
            if raw is None:
                s.value = torch.cuda.current_stream().cuda_stream
            elif isinstance(raw, int):
                s.value = raw
            elif hasattr(raw, "cuda_stream"):
                s.value = raw.cuda_stream
            else:
                raise ValueError(f"invalid stream value: {raw}")

        return [(ctypes.c_void_p, fill)]

    S.__c_abi_spec__ = __c_abi_spec__
    S._primus_current_stream = True
    log_rank_0("[Patch:flydsl.stream_default_current] FlyDSL Stream(None) now launches on torch's current stream")

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Env-gated MoE-layer wall-time probe (ROCMOE_MOE_TIMING=1).

Wraps the bound ``MoELayer.forward`` (works for both the stock Megatron MoELayer
and the patched ROCMoELayer) with sync-bounded host timing, so the per-MoE-layer
forward wall time is directly comparable between the baseline and ROCMoE runs.
Also times the layer's backward via a full-backward hook. Off by default; when
on it ADDS cuda syncs (perturbs timing) and is for A/B profiling only.

Priority 90 so it runs AFTER the class-swap patches (e.g. megatron.moe.rocmoe,
priority 50) and therefore wraps the final bound layer class.
"""

import os
import time

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0

_ENABLED = os.environ.get("ROCMOE_MOE_TIMING", "0") == "1"
_ACC = {"fwd": 0.0, "bwd": 0.0, "n_fwd": 0, "n_bwd": 0}


def _log_maybe():
    n = _ACC["n_fwd"]
    if n and n % 200 == 0:
        fwd_ms = _ACC["fwd"] / max(_ACC["n_fwd"], 1) * 1e3
        bwd_ms = _ACC["bwd"] / max(_ACC["n_bwd"], 1) * 1e3
        log_rank_0(
            f"[MoE timing] calls fwd={_ACC['n_fwd']} bwd={_ACC['n_bwd']} | "
            f"avg fwd={fwd_ms:.2f} ms  bwd={bwd_ms:.2f} ms  fwd+bwd={fwd_ms + bwd_ms:.2f} ms"
        )


@register_patch(
    "megatron.moe.timing",
    backend="megatron",
    phase="before_train",
    priority=90,
    description="Env-gated per-MoE-layer forward/backward wall-time probe",
    condition=lambda ctx: _ENABLED,
)
def patch_moe_timing(ctx: PatchContext):
    import torch

    import megatron.core.transformer.moe.moe_layer as moe_layer_mod

    layer_cls = moe_layer_mod.MoELayer
    if getattr(layer_cls, "_moe_timing_wrapped", False):
        return
    orig_forward = layer_cls.forward

    def _post_bwd_hook(mod, gin, gout):
        if getattr(mod, "_bwd_t0", None) is not None:
            torch.cuda.synchronize()
            _ACC["bwd"] += time.perf_counter() - mod._bwd_t0
            _ACC["n_bwd"] += 1
            mod._bwd_t0 = None

    def timed_forward(self, *args, **kwargs):
        # Register the module backward hook exactly once per instance.
        if not getattr(self, "_moe_timing_bwd_hooked", False):
            self.register_full_backward_hook(_post_bwd_hook)
            self._moe_timing_bwd_hooked = True

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_forward(self, *args, **kwargs)
        torch.cuda.synchronize()
        _ACC["fwd"] += time.perf_counter() - t0
        _ACC["n_fwd"] += 1
        _log_maybe()

        # Mark backward start when grad first reaches this layer's output.
        y = out[0] if isinstance(out, tuple) else out
        if isinstance(y, torch.Tensor) and y.requires_grad:

            def _mark(grad, _self=self):
                torch.cuda.synchronize()
                _self._bwd_t0 = time.perf_counter()
                return grad

            y.register_hook(_mark)
        return out

    layer_cls.forward = timed_forward
    layer_cls._moe_timing_wrapped = True
    log_rank_0(
        f"[Patch:megatron.moe.timing] Wrapped {layer_cls.__name__}.forward with wall-time probe"
    )

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_LM_HEAD_WGRAD_TORCH=1: Megatron's gradient-accumulation-fused weight gradient (apex
``fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16``, used by ``LinearWithGradAccumulationAndAsyncCommunication``,
i.e. the DSv4 lm_head ``[129408, 4096]``) runs as ``torch`` ``addmm_`` / ``mm(out=main_grad)`` instead.

Why: apex calls hipBLASLt directly, so PyTorch TunableOp never sees the GEMM and the default heuristic kernel
(NT 256x256x128, ~1.9 ms, ~57% of the 4 PF BF16 peak on MI455X) is used. Through torch the same GEMM
(``GemmTunableOp_BFloat16_NT,nt_4096_129408_4096_ld_4096_129408_4096``) is tunable.

Numerics: same math (fp32 accumulate, ``main_grad = round_bf16(acc + main_grad)``), BF16 in/out. With the default
heuristic the result is bit-identical to apex; a tuned solution may use a different K order.

With ``PRIMUS_WGRAD_BETA0=1`` the first write of a step overwrites main_grad (``torch.mm(out=main_grad)``, beta 0) via
the Turbo first-write policy, so the lm_head slice is neither cleared nor read (``PRIMUS_LM_HEAD_WGRAD_BETA0=0``
disables only this part). ``PRIMUS_LM_HEAD_WGRAD_LAYOUT=tn`` transposes both operands first (Triton, ~0.14 ms
for the 1 GB grad) so the GEMM runs as TN (see ``_LAYOUT``). fp32 main_grad and anything not contiguous / not BF16-FP16 fall back to apex.
"""

from __future__ import annotations

import os
import threading

import torch

_ENABLED = os.environ.get("PRIMUS_LM_HEAD_WGRAD_TORCH", "0").strip().lower() in ("1", "true")
_BETA0 = os.environ.get("PRIMUS_LM_HEAD_WGRAD_BETA0", "1") != "0"
# nt: torch mm on the training layouts (NT, both operands K-strided);
# tn: Triton-transpose grad_output [T, V] -> [V, T] and input [T, H] -> [H, T] first, so the GEMM is TN (K contiguous
#     for both operands, the layout the lm_head forward runs at ~76% of peak). Changes the GEMM's K order (not bit-identical).
_LAYOUT = os.environ.get("PRIMUS_LM_HEAD_WGRAD_LAYOUT", "nt").strip().lower()
_cur = threading.local()
_stats = {"logged": False}


def _log(msg: str) -> None:
    try:
        from primus.core.utils.module_utils import log_rank_0

        log_rank_0(msg)
    except Exception:  # noqa: BLE001
        print(msg)


def _first_write(weight) -> bool:
    if not _BETA0 or weight is None:
        return False
    try:
        from primus_turbo.pytorch.ops.utils import main_grad_first_write
    except ImportError:
        return False
    return main_grad_first_write(weight)


class _TorchWgradShim:
    """Stand-in for the apex ``fused_weight_gradient_mlp_cuda`` module inside Megatron ``tensor_parallel.layers``."""

    def __init__(self, orig):
        self._orig = orig

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def wgrad_gemm_accum_fp16(self, total_input, grad_output, main_grad):
        x = total_input.reshape(-1, total_input.shape[-1])
        g = grad_output.reshape(-1, grad_output.shape[-1])
        if (
            main_grad.dtype not in (torch.bfloat16, torch.float16)
            or x.dtype != main_grad.dtype
            or g.dtype != main_grad.dtype
            or not main_grad.is_contiguous()
        ):
            return self._orig.wgrad_gemm_accum_fp16(total_input, grad_output, main_grad)
        weight = getattr(_cur, "weight", None)
        if weight is not None and getattr(weight, "main_grad", None) is not main_grad:
            weight = None
        beta0 = _first_write(weight)
        if _LAYOUT == "tn" and x.stride(-1) == 1 and g.stride(-1) == 1:
            from primus.backends.megatron.core.fusions.bf16_transpose import transpose_last2

            gt = None
            if os.environ.get("PRIMUS_V4_CE_BWD_GT", "0") == "1":  # densegemm: dlogits^T written by the CE backward
                from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.fused_ce import (
                    pop_grad_t,
                )

                gt = pop_grad_t(g)
                if gt is None and not _stats.get("gt_miss_logged"):
                    _stats["gt_miss_logged"] = True
                    _log("[Patch:lm_head_wgrad_torch] PRIMUS_V4_CE_BWD_GT: no CE-written grad^T for this grad -> transpose")
            a = gt if gt is not None else transpose_last2(g)
            b = transpose_last2(x).t()  # [V, T], [T, H] view of [H, T]
        else:
            a, b = g.t(), x
        if beta0:
            torch.mm(a, b, out=main_grad)
        else:
            main_grad.addmm_(a, b)
        if not _stats["logged"]:
            _stats["logged"] = True
            _log(
                f"[Patch:lm_head_wgrad_torch] wgrad {tuple(main_grad.shape)} via torch "
                f"{'mm(out=main_grad) beta 0' if beta0 else 'addmm_ beta 1'}, layout {_LAYOUT} (TunableOp-visible)"
            )


def install() -> bool:
    from megatron.core.tensor_parallel import layers as mlayers

    orig = getattr(mlayers, "fused_weight_gradient_mlp_cuda", None)
    if orig is None or isinstance(orig, _TorchWgradShim):
        return False
    mlayers.fused_weight_gradient_mlp_cuda = _TorchWgradShim(orig)

    cls = mlayers.LinearWithGradAccumulationAndAsyncCommunication
    orig_bwd = cls.backward

    def backward(ctx, grad_output):
        # expose the weight (same Python object as the parameter) to the shim for the first-write policy
        try:
            _cur.weight = ctx.saved_tensors[1]
        except Exception:  # noqa: BLE001
            _cur.weight = None
        try:
            return orig_bwd(ctx, grad_output)
        finally:
            _cur.weight = None

    cls.backward = staticmethod(backward)
    _log(
        "[Patch:lm_head_wgrad_torch] Megatron grad-accum-fusion wgrad (apex wgrad_gemm_accum_fp16) -> torch "
        f"addmm_/mm (TunableOp-tunable); beta0 first write={'on' if _BETA0 else 'off'}; layout={_LAYOUT}"
    )
    return True


try:
    from primus.core.patches import PatchContext, register_patch

    @register_patch(
        "megatron.core.tensor_parallel.lm_head_wgrad_torch",
        backend="megatron",
        phase="before_train",
        description="Grad-accum-fused wgrad through torch addmm_/mm so TunableOp can tune it (lm_head).",
        condition=lambda ctx: _ENABLED,
        priority=72,
    )
    def patch_lm_head_wgrad_torch(ctx: PatchContext) -> None:
        del ctx
        install()

except ImportError:
    pass

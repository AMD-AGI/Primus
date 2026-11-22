# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""fuse2 line: weight gradients written straight into ``main_grad`` (no param.grad + DDP hook copy).

``linear_main_grad(x, w)`` (PRIMUS_FUSE2_COMP_MG=1, used by the V4 compressor's ``wkv_gate``): the same three GEMMs
as ``F.linear`` + autograd's ``MmBackward0`` (forward ``x2 @ w^T``; dgrad ``g2 @ w``; wgrad ``g2^T @ x2``, the
layout ``mm_mat2_backward`` picks for a transposed ``mat2``), with the wgrad's output being ``w.main_grad`` on the
first write of the step (wgrad_beta0 policy) and ``main_grad.add_(dW)`` otherwise -- exactly what the DDP hook does
with ``param.grad``. Requires ``main_grad`` with ``w``'s dtype / shape (bf16 main_grads in the DSv4 recipe).
"""
from __future__ import annotations

import os

import torch

from primus.backends.megatron.core.fusions.bf16tune import bt_mm as _bt_mm
from primus.backends.megatron.core.fusions.bf16tune import enabled as _bt_enabled

_BT = _bt_enabled()
# bf16grp: PRIMUS_BF16GRP_PAIR=1 -> the compressor dgrad + wgrad (beta 0, not deferred) as ONE bf16rot2 launch
_BT_PAIR = _BT and os.environ.get("PRIMUS_BF16GRP_PAIR", "0") == "1"


def fuse2_linear_mg_ok(x: torch.Tensor, w: torch.Tensor, env: str = "PRIMUS_FUSE2_COMP_MG") -> bool:
    if os.environ.get(env, "0") != "1":
        return False
    mg = getattr(w, "main_grad", None)
    return (
        torch.is_grad_enabled()
        and isinstance(w, torch.nn.Parameter)
        and w.requires_grad
        and hasattr(w, "grad_added_to_main_grad")
        and isinstance(mg, torch.Tensor)
        and mg.shape == w.shape
        and mg.dtype == w.dtype
        and mg.is_contiguous()
        and w.is_contiguous()
        and w.dim() == 2
        and x.is_cuda
        and x.dtype == w.dtype
        and x.is_contiguous()
        and not getattr(w, "shared", False)
    )


class AttGradSlot:
    """PRIMUS_FUSE2_ATT_GADD: the attention input feeds the wq_a||wkv concat GEMM and the compressor's wkv_gate.
    The compressor backward (autograd runs it first) leaves its dX here; the concat backward's dX GEMM adds into that
    buffer in its epilogue (dsv4up accumulate=2: bf16(product) + buffer, one rounding -- autograd's bf16 add), so the
    separate add kernel goes away."""

    __slots__ = ("g", "fused", "ev", "branch")

    def __init__(self, branch=False):
        self.g = self.fused = self.ev = None
        self.branch = branch  # PRIMUS_OVERLAP_CSA_BWD: the compressor (dX producer) runs on a branch stream

    def take(self, M, K, dtype):
        g, self.g = self.g, None
        ev, self.ev = self.ev, None
        if g is None or g.dtype != dtype or not g.is_contiguous() or g.numel() != M * K:
            return None
        if ev is not None:  # PRIMUS_OVERLAP_CSA_BWD: the compressor dX was produced on the compressor branch stream
            cur = torch.cuda.current_stream(g.device)
            cur.wait_event(ev)
            g.record_stream(cur)
        return g.view(M, K)

    def untake(self, g):
        self.g = g


class AttFanOut(torch.autograd.Function):
    """``x -> (x_concat, x_rest)`` aliases so the two input gradients meet here, not in autograd's add."""

    @staticmethod
    def forward(ctx, x, slot):
        ctx.slot = slot
        return x.view_as(x), x.view_as(x)

    @staticmethod
    def backward(ctx, g_cat, g_rest):
        slot, ctx.slot = ctx.slot, None
        fused = slot.fused if slot is not None else None
        if slot is not None:
            slot.fused = slot.g = None
        if fused is not None:
            fp = fused.data_ptr()
            if g_cat is None or g_rest is None or g_cat.data_ptr() != fp or g_rest.data_ptr() != fp:
                # the concat's dX was added into a compressor dX that is not the whole x_rest gradient: the
                # in-place sum cannot be undone, so refuse instead of returning a wrong gradient
                raise RuntimeError("PRIMUS_FUSE2_ATT_GADD: unexpected extra consumer of the attention input")
            return g_cat, None
        if g_cat is None:
            return g_rest, None
        if g_rest is None:
            return g_cat, None
        return g_rest + g_cat, None  # autograd's order: compressor first, then the concat


def att_gadd_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSE2_ATT_GADD", "0") == "1"


def _ovl_wgrad():
    """overlap module when PRIMUS_OVERLAP_OA_WGRAD has bit 2 (overlap2) and PRIMUS_OVERLAP_WGRAD is active, else None."""
    try:
        lv = int(os.environ.get("PRIMUS_OVERLAP_OA_WGRAD", "0"))
    except ValueError:
        lv = 0
    if not lv & 2:
        return None
    try:
        from primus_turbo.pytorch.core import overlap
    except ImportError:  # pragma: no cover
        return None
    return overlap if overlap.enabled("WGRAD") else None


class _LinearMainGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, slot=None):
        x2 = x.reshape(-1, x.shape[-1])
        out = _bt_mm(x2, w.t()) if _BT else None  # PRIMUS_BF16TUNE: planned shapes on Gluon bf16g2 / split-K
        if out is None:
            out = torch.mm(x2, w.t())
        ctx.save_for_backward(x2, w)
        ctx.wparam = w
        ctx.slot = slot
        ctx.lead = x.shape[:-1]
        w.grad_added_to_main_grad = True  # set in forward, like Turbo's fused-wgrad linears
        return out.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

        x2, w = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        dx = None
        pre = None  # bf16grp pair path: (main_grad, first, wgrad done) decided before the GEMMs
        if _BT_PAIR and ctx.needs_input_grad[0] and _ovl_wgrad() is None:
            from primus.backends.megatron.core.fusions.bf16grp import compressor_bwd_pair

            pmg = ctx.wparam.main_grad
            pfirst = main_grad_first_write(ctx.wparam)
            dxp = compressor_bwd_pair(g2, w, x2, pmg) if pfirst else None  # dgrad + wgrad (into main_grad)
            pre = (pmg, pfirst, dxp is not None)
            if dxp is not None:
                dx = dxp.view(*ctx.lead, w.shape[1])
        if dx is None and ctx.needs_input_grad[0]:
            dx = _bt_mm(g2, w) if _BT else None
            dx = (g2.mm(w) if dx is None else dx).view(*ctx.lead, w.shape[1])
        if ctx.slot is not None and dx is not None:
            ctx.slot.g = dx  # PRIMUS_FUSE2_ATT_GADD: the concat dX GEMM adds into this buffer
            if ctx.slot.branch:  # PRIMUS_OVERLAP_CSA_BWD: running on the compressor branch stream
                ctx.slot.ev = torch.cuda.Event()
                ctx.slot.ev.record(torch.cuda.current_stream(dx.device))
            ctx.slot = None
        wp, ctx.wparam = ctx.wparam, None
        if pre is None:
            mg = wp.main_grad
            first = main_grad_first_write(wp)
        else:
            mg, first = pre[0], pre[1]

        def _wgrad(mg=mg, g2=g2, x2=x2, first=first):
            if first:
                if not _BT or _bt_mm(g2.t(), x2, out=mg) is None:  # PRIMUS_BF16TUNE: planned shapes on Gluon
                    torch.mm(g2.t(), x2, out=mg)
            else:
                mg.add_(torch.mm(g2.t(), x2))

        ovl = _ovl_wgrad()
        if pre is not None and pre[2]:
            pass  # bf16grp: the wgrad ran in the dgrad's launch (beta 0 into main_grad)
        elif ovl is not None:  # PRIMUS_OVERLAP_OA_WGRAD (overlap2): same GEMM (+ add) on overlap's deferred wgrad stream
            ovl.defer(_wgrad, mg, g, g2, x2)
        else:
            _wgrad()
        return dx, _get_dummy_wgrad(list(w.shape), w.dtype), None


def linear_main_grad(x: torch.Tensor, w: torch.Tensor, slot=None) -> torch.Tensor:
    w._primus_fwd_claim = True  # grad_added_to_main_grad is flagged already in forward
    return _LinearMainGradFn.apply(x, w, slot)

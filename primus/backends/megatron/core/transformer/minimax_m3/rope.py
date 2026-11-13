###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unfused RoPE whose cos/sin tables are shared by the calls of one layer.

An MSA layer applies the same freqs four times: to its query and key, and to the
indexer's query and key. Megatron's ``_apply_rotary_pos_emb_bshd`` recomputes
``cos(freqs)`` and ``sin(freqs)`` on every call, six kernel launches each time,
and the MSA forward is bound by kernel launches on the CPU. :class:`RopeTables`
computes them once per dtype and then runs the same elementwise ops in the same
order, so its output is bitwise that of the upstream function.
"""

from typing import Dict, Tuple

import torch
from megatron.core.models.common.embeddings.rope_utils import _rotate_half


class RopeTables:
    """``_apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved)`` for many ``t``.

    Args:
        freqs: ``[sq, 1, 1, rotary_dim]`` rope freqs. ``rotary_dim`` may be less
            than the head dim; the tail of every head then passes through.
    """

    def __init__(self, freqs: torch.Tensor):
        self.freqs = freqs
        self._tables: Dict[torch.dtype, Tuple[torch.Tensor, torch.Tensor]] = {}

    # Tables shared across layers and steps. Megatron's RotaryEmbedding.forward is lru_cached, so
    # every layer of every step passes the same freqs object; an entry holds that object (its memory
    # cannot be reused while cached) and is used only when the caller passes that very tensor.
    _shared: Dict[Tuple[int, torch.dtype], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def cos_sin(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        tables = self._tables.get(dtype)
        if tables is None:
            key = (id(self.freqs), dtype)
            entry = RopeTables._shared.get(key)
            if entry is not None and entry[0] is self.freqs and not self.freqs.requires_grad:
                tables = entry[1:]
            else:
                # upstream's (cos(freqs) * mscale).to(dtype) at mscale 1.0, which is exact
                tables = (torch.cos(self.freqs).to(dtype), torch.sin(self.freqs).to(dtype))
                if not self.freqs.requires_grad:
                    if len(RopeTables._shared) >= 8:
                        RopeTables._shared.clear()
                    RopeTables._shared[key] = (self.freqs,) + tables
            self._tables[dtype] = tables
        return tables

    def apply(self, t: torch.Tensor, rotary_interleaved: bool = False) -> torch.Tensor:
        cos_, sin_ = self.cos_sin(t.dtype)
        if _fused_ok(t, cos_, rotary_interleaved):
            return _FusedRope.apply(t, cos_, sin_)
        rot_dim = cos_.shape[-1]
        x, x_pass = t[..., :rot_dim], t[..., rot_dim:]
        x = (x * cos_) + (_rotate_half(x, rotary_interleaved) * sin_)
        return torch.cat((x, x_pass), dim=-1)

# ---------------------------------------------------------------------------
# Fused path (appended to rope.py)
# ---------------------------------------------------------------------------
# The eager ``apply`` is a slice, a rotate_half (two slices, a neg and a cat), two
# muls, an add and a cat, and autograd adds a slice_backward, a cat backward and
# more muls: about ten ffn-sized kernels per call, four calls per MSA layer. The
# Triton kernels below do the same arithmetic in one pass each way and round to
# BF16 exactly where the eager ops do (each op computes in FP32 and rounds once),
# so the output and the input gradient are bitwise those of the eager path.

import os as _os

_rope_kernels = None


def _get_rope_kernels():
    global _rope_kernels
    if _rope_kernels is not None:
        return _rope_kernels
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd(T, COS, SIN, OUT, B, H, st_s, st_b, st_h, ROT: tl.constexpr, D: tl.constexpr,
             BH: tl.constexpr):
        pid = tl.program_id(0)
        s = pid // B
        b = pid % B
        h = tl.arange(0, BH)
        d = tl.arange(0, D)
        HALF: tl.constexpr = ROT // 2
        hm = h < H
        base = T + s.to(tl.int64) * st_s + b.to(tl.int64) * st_b + h[:, None].to(tl.int64) * st_h
        x = tl.load(base + d[None, :], mask=hm[:, None], other=0.0)
        in_rot = d < ROT
        partner = tl.where(d < HALF, d + HALF, d - HALF)
        partner = tl.where(in_rot, partner, d)
        xp = tl.load(base + partner[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
        c = tl.load(COS + s.to(tl.int64) * ROT + d, mask=in_rot, other=0.0).to(tl.float32)
        sn = tl.load(SIN + s.to(tl.int64) * ROT + d, mask=in_rot, other=0.0).to(tl.float32)
        sign = tl.where(d < HALF, -1.0, 1.0)
        a = (x.to(tl.float32) * c[None, :]).to(tl.bfloat16).to(tl.float32)
        r = ((sign[None, :] * xp) * sn[None, :]).to(tl.bfloat16).to(tl.float32)
        y = (a + r).to(tl.bfloat16)
        y = tl.where(in_rot[None, :], y, x.to(tl.bfloat16))
        out = OUT + (pid.to(tl.int64) * H + h[:, None].to(tl.int64)) * D + d[None, :]
        tl.store(out, y.to(OUT.dtype.element_ty), mask=hm[:, None])

    @triton.jit
    def _bwd(G, COS, SIN, DT, B, H, ROT: tl.constexpr, D: tl.constexpr, BH: tl.constexpr):
        pid = tl.program_id(0)
        s = pid // B
        h = tl.arange(0, BH)
        d = tl.arange(0, D)
        HALF: tl.constexpr = ROT // 2
        hm = h < H
        base = G + (pid.to(tl.int64) * H + h[:, None].to(tl.int64)) * D
        g = tl.load(base + d[None, :], mask=hm[:, None], other=0.0)
        in_rot = d < ROT
        partner = tl.where(d < HALF, d + HALF, d - HALF)
        partner = tl.where(in_rot, partner, d)
        gp = tl.load(base + partner[None, :], mask=hm[:, None], other=0.0).to(tl.float32)
        c = tl.load(COS + s.to(tl.int64) * ROT + d, mask=in_rot, other=0.0).to(tl.float32)
        sp = tl.load(SIN + s.to(tl.int64) * ROT + partner, mask=in_rot, other=0.0).to(tl.float32)
        # d < HALF: x1 receives +(g*sin)[d+HALF]; HALF <= d < ROT: x2 receives -(g*sin)[d-HALF]
        sign = tl.where(d < HALF, 1.0, -1.0)
        a = (g.to(tl.float32) * c[None, :]).to(tl.bfloat16).to(tl.float32)
        r = (sign[None, :] * (gp * sp[None, :]).to(tl.bfloat16).to(tl.float32))
        dx = (a + r).to(tl.bfloat16)
        dx = tl.where(in_rot[None, :], dx, g.to(tl.bfloat16))
        tl.store(DT + (pid.to(tl.int64) * H + h[:, None].to(tl.int64)) * D + d[None, :],
                 dx.to(DT.dtype.element_ty), mask=hm[:, None])

    _rope_kernels = (_fwd, _bwd)
    return _rope_kernels


class _FusedRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, cos_, sin_):
        S, B, H, D = t.shape
        rot = cos_.shape[-1]
        out = torch.empty((S, B, H, D), dtype=t.dtype, device=t.device)
        fwd, _ = _get_rope_kernels()
        import triton

        fwd[(S * B,)](t, cos_, sin_, out, B, H, t.stride(0), t.stride(1), t.stride(2),
                      ROT=rot, D=D, BH=triton.next_power_of_2(H))
        ctx.save_for_backward(cos_, sin_)
        ctx.shape = (S, B, H, D)
        return out

    @staticmethod
    def backward(ctx, g):
        cos_, sin_ = ctx.saved_tensors
        S, B, H, D = ctx.shape
        g = g.contiguous()
        dt = torch.empty((S, B, H, D), dtype=g.dtype, device=g.device)
        _, bwd = _get_rope_kernels()
        import triton

        bwd[(S * B,)](g, cos_, sin_, dt, B, H, ROT=cos_.shape[-1], D=D,
                      BH=triton.next_power_of_2(H))
        return dt, None, None


def _fused_ok(t: torch.Tensor, cos_: torch.Tensor, rotary_interleaved: bool) -> bool:
    if _os.environ.get("PRIMUS_MSA_FUSED_ROPE", "1") == "0" or rotary_interleaved:
        return False
    if not t.is_cuda or t.dtype != torch.bfloat16 or t.dim() != 4 or t.stride(-1) != 1:
        return False
    S, D = t.shape[0], t.shape[-1]
    rot = cos_.shape[-1]
    return (
        cos_.dtype == torch.bfloat16
        and cos_.is_contiguous()
        and cos_.numel() == S * rot
        and rot % 2 == 0
        and rot <= D
        and (D & (D - 1)) == 0
    )

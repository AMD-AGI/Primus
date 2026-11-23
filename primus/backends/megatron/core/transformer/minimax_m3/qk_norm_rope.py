###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""q/k RMSNorm + RoPE straight from the fused ``linear_qkv`` output, bit for bit.

Upstream splits ``mixed_qkv`` ``[sq, b, ng, (np/ng + 2) * hn]`` into strided q/k/v
views, copies q to ``[sq, b, np, hn]`` (its heads are not uniformly strided),
runs the Turbo RMSNorm on q and k, then RoPE on each; the backward runs RoPE
backward and RMSNorm backward, and ``SplitWithSizesBackward`` concatenates
dq/dk/dv back into a fresh ``dmixed_qkv``.

Here one kernel each for q and k reads the rows straight from ``mixed_qkv`` (row ``r``
of q is head ``r % (np/ng)`` of token-group ``r // (np/ng)``), normalises them and applies
RoPE; the backward writes dx straight into its slice of ``dmixed_qkv``, so the q copy and
the cat go (v is copied out of ``mixed_qkv`` row by row). The arithmetic is unchanged:

* the RMSNorm is Turbo's ``rmsnorm_{fwd,bwd}_kernel_multi_row`` except for the row
  addressing, launched with Turbo's own configs (``_pick_config`` / ``_pick_bwd_config``)
  and dgamma reduced by Turbo's ``_finalize_dgamma`` -- same tiles, same reductions,
  same partials;
* RoPE rounds to BF16 exactly where the eager ops of ``RopeTables.apply`` do.

The MSA indexer's q and k (``norm_rope``) use the same kernels on contiguous rows.

Both are checked against the eager ops on their first use in a process (``bitwise_check``),
which also catches a Turbo whose kernels no longer match these transcriptions. Norms with
module hooks (e.g. DDP's parameter all-gather under ``overlap_param_gather``) keep the eager
path. ``use_turbo_fused_qk_rmsnorm_rope`` (``patches/turbo/qk_rmsnorm_rope_patches.py``) is
the other packed-QKV norm + RoPE: Turbo's FlyDSL operator for full RoPE, not bitwise; this
one is MiniMax's partial RoPE, bit-identical.
"""

from __future__ import annotations

import functools
import inspect

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check
from primus.core.utils.module_utils import warning_rank_0

_COPY_ROWS = 64
_WARNED = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warning_rank_0(msg)


@triton.jit
def _norm_rope_fwd_rows(
    X_ptr,
    G_ptr,
    Y_ptr,
    RSTD_ptr,
    COS,
    SIN,
    x_off,
    HPG,
    stride_grp,
    ROWS_PER_S,
    B,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    ZC: tl.constexpr,
):
    # rmsnorm_fwd_kernel_multi_row with x row r at x_off + (r // HPG) * stride_grp + (r % HPG) * D,
    # then RoPE on the normed BF16 rows
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_rows = x_off + (row_offs // HPG).to(tl.int64) * stride_grp + (row_offs % HPG) * D
    x_ptrs = X_ptr + x_rows[:, None] + h_offs[None, :]
    y_ptrs = Y_ptr + row_offs[:, None].to(tl.int64) * D + h_offs[None, :]
    g_ptrs = G_ptr + h_offs

    full_mask = row_mask[:, None] & h_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    if ZC:
        # zero-centered gamma: the BF16 (1 + w) of the eager ``weight + 1``, formed in-register
        g = (g + 1.0).to(tl.bfloat16).to(tl.float32)

    var = tl.sum(x * x, axis=1) / H
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None] * g[None, :]).to(tl.bfloat16)
    tl.store(RSTD_ptr + row_offs, rstd, mask=row_mask)

    HALF: tl.constexpr = ROT // 2
    d = h_offs
    in_rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, d - HALF)
    partner = tl.where(in_rot, partner, d)
    xp = tl.load(X_ptr + x_rows[:, None] + partner[None, :], mask=full_mask, other=0.0).to(tl.float32)
    gp = tl.load(G_ptr + partner, mask=h_mask, other=0.0).to(tl.float32)
    if ZC:
        # zero-centered gamma: the BF16 (1 + w) of the eager ``weight + 1``, formed in-register
        gp = (gp + 1.0).to(tl.bfloat16).to(tl.float32)
    yp = (xp * rstd[:, None] * gp[None, :]).to(tl.bfloat16).to(tl.float32)
    s = (row_offs // ROWS_PER_S).to(tl.int64)
    c = tl.load(COS + s[:, None] * ROT + d[None, :], mask=row_mask[:, None] & in_rot[None, :], other=0.0).to(
        tl.float32
    )
    sn = tl.load(SIN + s[:, None] * ROT + d[None, :], mask=row_mask[:, None] & in_rot[None, :], other=0.0).to(
        tl.float32
    )
    sign = tl.where(d < HALF, -1.0, 1.0)
    a = (y.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    r = ((sign[None, :] * yp) * sn).to(tl.bfloat16).to(tl.float32)
    o = (a + r).to(tl.bfloat16)
    o = tl.where(in_rot[None, :], o, y)
    tl.store(y_ptrs, o.to(Y_ptr.dtype.element_ty), mask=full_mask)


@triton.jit
def _rope_norm_bwd_rows(
    DQ_ptr,
    X_ptr,
    G_ptr,
    RSTD_ptr,
    DX_ptr,
    DG_PART_ptr,
    COS,
    SIN,
    x_off,
    HPG,
    stride_grp,
    ROWS_PER_S,
    stride_dgp,
    B,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    ZC: tl.constexpr,
):
    # RoPE backward on the incoming gradient rows, then rmsnorm_bwd_kernel_multi_row with x and
    # dx rows addressed as in the forward
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    row_offs = row_start + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = row_offs < B
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    x_rows = x_off + (row_offs // HPG).to(tl.int64) * stride_grp + (row_offs % HPG) * D
    x_ptrs = X_ptr + x_rows[:, None] + h_offs[None, :]
    dq_base = DQ_ptr + row_offs[:, None].to(tl.int64) * D
    dx_ptrs = DX_ptr + x_rows[:, None] + h_offs[None, :]
    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    g_ptrs = G_ptr + h_offs
    full_mask = row_mask[:, None] & h_mask[None, :]

    HALF: tl.constexpr = ROT // 2
    d = h_offs
    in_rot = d < ROT
    partner = tl.where(d < HALF, d + HALF, d - HALF)
    partner = tl.where(in_rot, partner, d)
    gq = tl.load(dq_base + d[None, :], mask=full_mask, other=0.0)
    gqp = tl.load(dq_base + partner[None, :], mask=full_mask, other=0.0).to(tl.float32)
    s = (row_offs // ROWS_PER_S).to(tl.int64)
    rm2 = row_mask[:, None] & in_rot[None, :]
    c = tl.load(COS + s[:, None] * ROT + d[None, :], mask=rm2, other=0.0).to(tl.float32)
    sp = tl.load(SIN + s[:, None] * ROT + partner[None, :], mask=rm2, other=0.0).to(tl.float32)
    sign = tl.where(d < HALF, 1.0, -1.0)
    a = (gq.to(tl.float32) * c).to(tl.bfloat16).to(tl.float32)
    r = sign[None, :] * (gqp * sp).to(tl.bfloat16).to(tl.float32)
    dyb = (a + r).to(tl.bfloat16)
    dyb = tl.where(in_rot[None, :], dyb, gq.to(tl.bfloat16))

    x = tl.load(x_ptrs, mask=full_mask, other=0.0).to(tl.float32)
    dy = dyb.to(tl.float32)
    g = tl.load(g_ptrs, mask=h_mask, other=0.0).to(tl.float32)
    if ZC:
        # zero-centered gamma: the BF16 (1 + w) of the eager ``weight + 1``, formed in-register
        g = (g + 1.0).to(tl.bfloat16).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row_offs, mask=row_mask, other=0.0).to(tl.float32)

    x_hat = x * rstd[:, None]
    dxhat = dy * g[None, :]
    m = tl.sum(dxhat * x_hat, axis=1) / H
    dx = (dxhat - x_hat * m[:, None]) * rstd[:, None]

    tl.store(dx_ptrs, dx.to(DX_ptr.dtype.element_ty), mask=full_mask)

    dgp_block = (dy * x_hat) * row_mask[:, None].to(tl.float32)
    dgp_row = tl.sum(dgp_block, axis=0)
    tl.store(dgp_ptrs, dgp_row, mask=h_mask)


@triton.jit
def _copy_rows(
    SRC_ptr, DST_ptr, src_off, stride_src, stride_dst, dst_off, R, D: tl.constexpr, BR: tl.constexpr
):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    d = tl.arange(0, D)
    m = (r < R)[:, None]
    v = tl.load(SRC_ptr + src_off + r[:, None].to(tl.int64) * stride_src + d[None, :], mask=m)
    tl.store(DST_ptr + dst_off + r[:, None].to(tl.int64) * stride_dst + d[None, :], v, mask=m)


@functools.lru_cache(maxsize=None)
def _turbo_cfg():
    """Turbo's ``rmsnorm_impl`` if it has the private helpers these kernels launch with, else None."""
    try:
        from primus_turbo.pytorch.kernels.normalization import rmsnorm_impl as R
    except ImportError:
        return None
    if not all(hasattr(R, f) for f in ("_pick_config", "_pick_bwd_config", "_finalize_dgamma")):
        return None
    return R


def _norm_rope_fwd(mixed, x_off, hpg, stride_grp, n_rows, rows_per_s, gamma, eps, cos_, sin_, zc=False):
    """Normed + roped rows ``[n_rows, D]`` and rstd, in one pass."""
    R = _turbo_cfg()
    D = gamma.shape[0]
    out = torch.empty((n_rows, D), dtype=mixed.dtype, device=mixed.device)
    rstd = torch.empty(n_rows, device=mixed.device, dtype=torch.float32)
    BLOCK_H, ROWS, num_warps, num_stages = R._pick_config(D, n_rows)
    assert ROWS > 1 and BLOCK_H == D
    _norm_rope_fwd_rows[((n_rows + ROWS - 1) // ROWS,)](
        mixed,
        gamma,
        out,
        rstd,
        cos_,
        sin_,
        x_off,
        hpg,
        stride_grp,
        rows_per_s,
        B=n_rows,
        H=D,
        eps=eps,
        BLOCK_H=BLOCK_H,
        ROWS_PER_BLOCK=ROWS,
        D=D,
        ROT=cos_.shape[-1],
        ZC=bool(zc),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, rstd


def _rope_norm_bwd(
    dq, mixed, dmixed, x_off, hpg, stride_grp, n_rows, rows_per_s, gamma, rstd, cos_, sin_, zc=False
):
    R = _turbo_cfg()
    D = gamma.shape[0]
    mode, BLOCK_H, ROWS, num_warps, num_stages = R._pick_bwd_config(D, n_rows)
    assert mode == "multi" and BLOCK_H == D
    num_programs = (n_rows + ROWS - 1) // ROWS
    dg_partial = torch.empty(num_programs, D, device=mixed.device, dtype=torch.float32)
    _rope_norm_bwd_rows[(num_programs,)](
        dq,
        mixed,
        gamma,
        rstd,
        dmixed,
        dg_partial,
        cos_,
        sin_,
        x_off,
        hpg,
        stride_grp,
        rows_per_s,
        dg_partial.stride(0),
        B=n_rows,
        H=D,
        BLOCK_H=BLOCK_H,
        ROWS_PER_BLOCK=ROWS,
        D=D,
        ROT=cos_.shape[-1],
        ZC=bool(zc),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return R._finalize_dgamma(dg_partial, gamma.dtype)


def _turbo_norm_ok(norm, head_dim: int) -> bool:
    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboRMSNorm,
        )
    except ImportError:
        return False

    if not (
        type(norm) is PrimusTurboRMSNorm
        and tuple(norm.weight.shape) == (head_dim,)
        and norm.weight.dtype == torch.bfloat16
    ):
        return False
    if _turbo_cfg() is None:
        _warn_once(
            "turbo",
            "One-pass q/k RMSNorm + RoPE off: Primus-Turbo's rmsnorm_impl lacks "
            "_pick_config / _pick_bwd_config / _finalize_dgamma.",
        )
        return False
    # The kernels read norm.weight without calling the module, so no hook of it would run.
    if norm._forward_pre_hooks or norm._forward_hooks:
        _warn_once(
            "hooks",
            "One-pass q/k RMSNorm + RoPE off: the q/k norm has forward hooks (e.g. DDP's "
            "parameter all-gather with overlap_param_gather), which these kernels would skip.",
        )
        return False
    return True


def _rope_tables(rope, head_dim: int):
    """``rope``'s BF16 (cos, sin) if the kernels can index them (one dense row per position)."""
    cos_, sin_ = rope.cos_sin(torch.bfloat16)
    rot = cos_.shape[-1]
    if (
        cos_.is_contiguous()
        and sin_.is_contiguous()
        and cos_.numel() == cos_.shape[0] * rot
        and rot % 2 == 0
        and rot <= head_dim
    ):
        return cos_, sin_
    return None


def _multi_row(head_dim: int, n_rows: int) -> bool:
    """Whether Turbo runs ``n_rows`` rows with its multi-row kernels both ways (the ones
    transcribed here)."""
    R = _turbo_cfg()
    try:
        block_h, rows, _, _ = R._pick_config(head_dim, n_rows)
        mode, bwd_block_h, _, _, _ = R._pick_bwd_config(head_dim, n_rows)
    except (TypeError, ValueError) as e:
        _warn_once(
            "turbo_cfg", f"One-pass q/k RMSNorm + RoPE off: Primus-Turbo's RMSNorm configs changed ({e})."
        )
        return False
    return block_h == head_dim and rows > 1 and mode == "multi" and bwd_block_h == head_dim


class _QKNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixed, gq, gk, cos_, sin_, hpg, eps, zc=False):
        S, Bt, NG, W = mixed.shape
        D = gq.shape[0]
        assert mixed.dtype == torch.bfloat16 and mixed.is_contiguous()
        assert W == (hpg + 2) * D and cos_.shape[0] == S
        nq, nk = S * Bt * NG * hpg, S * Bt * NG
        q, rq = _norm_rope_fwd(mixed, 0, hpg, W, nq, Bt * NG * hpg, gq, eps, cos_, sin_, zc)
        k, rk = _norm_rope_fwd(mixed, hpg * D, 1, W, nk, Bt * NG, gk, eps, cos_, sin_, zc)
        v = torch.empty((S, Bt, NG, D), dtype=mixed.dtype, device=mixed.device)
        _copy_rows[(triton.cdiv(nk, _COPY_ROWS),)](mixed, v, (hpg + 1) * D, W, D, 0, nk, D=D, BR=_COPY_ROWS)
        ctx.save_for_backward(mixed, gq, gk, rq, rk, cos_, sin_)
        ctx.hpg = hpg
        ctx.zc = zc
        return q.view(S, Bt, NG * hpg, D), k.view(S, Bt, NG, D), v

    @staticmethod
    def backward(ctx, dq, dk, dv):
        mixed, gq, gk, rq, rk, cos_, sin_ = ctx.saved_tensors
        hpg, zc = ctx.hpg, ctx.zc
        S, Bt, NG, W = mixed.shape
        D = gq.shape[0]
        nq, nk = S * Bt * NG * hpg, S * Bt * NG
        dmixed = torch.empty_like(mixed)
        if dq is None:
            dq = torch.zeros((S, Bt, NG * hpg, D), dtype=mixed.dtype, device=mixed.device)
        if dk is None:
            dk = torch.zeros((S, Bt, NG, D), dtype=mixed.dtype, device=mixed.device)
        dgq = _rope_norm_bwd(
            dq.contiguous(), mixed, dmixed, 0, hpg, W, nq, Bt * NG * hpg, gq, rq, cos_, sin_, zc
        )
        dgk = _rope_norm_bwd(
            dk.contiguous(), mixed, dmixed, hpg * D, 1, W, nk, Bt * NG, gk, rk, cos_, sin_, zc
        )
        if dv is None:
            dmixed.view(S, Bt, NG, W)[..., (hpg + 1) * D :].zero_()
        else:
            _copy_rows[(triton.cdiv(nk, _COPY_ROWS),)](
                dv.contiguous(), dmixed, 0, D, W, (hpg + 1) * D, nk, D=D, BR=_COPY_ROWS
            )
        return dmixed, dgq, dgk, None, None, None, None, None


@functools.lru_cache(maxsize=None)
def _takes_split_qkv(attn_cls) -> bool:
    return "split_qkv" in inspect.signature(attn_cls.get_query_key_value_tensors).parameters


def _gamma(norm, weight, in_kernel: bool = False):
    """``(gamma, zc)`` for the kernels: the norm's gamma as the eager path forms it (``weight + 1``,
    one BF16 launch, for a zero-centered norm), or, with ``in_kernel`` (config switch
    ``msa_fused_zero_centered_gamma``), the raw BF16 weight and ``zc=True``: the kernels then form
    the same BF16 ``1 + w`` in-register and the launch goes. The gradient is unchanged (the add's
    backward passes dgamma through)."""
    if not norm.zero_centered_gamma:
        return weight, False
    if in_kernel and weight.dtype == torch.bfloat16:
        return weight, True
    return weight + 1, False


def _in_kernel_gamma(config) -> bool:
    return bool(getattr(config, "msa_fused_zero_centered_gamma", False))


def _probe_weight(weight, seed: int):
    """``weight`` moved off its init value, where a zero-centered gamma is exactly 1."""
    noise = bitwise_check.randn(weight.shape, weight.device, seed=seed)
    return (weight.detach().float() + 0.1 * noise).to(weight.dtype)


def _qk_self_check(attn, S: int, Bt: int, rope, in_kernel: bool = False) -> bool:
    """The kernels against upstream's split, q/k norms and ``rope.apply``, on random rows of the
    training shape, with the gammas moved off their init values."""
    qn, kn = attn.q_layernorm, attn.k_layernorm
    D = attn.hidden_size_per_attention_head
    NG = attn.num_query_groups_per_partition
    hpg = attn.num_attention_heads_per_partition // NG

    zc_key = _gamma(qn, qn.weight.detach(), in_kernel)[1] if in_kernel else False

    def inputs():
        mixed = bitwise_check.randn((S, Bt, NG, (hpg + 2) * D), qn.weight.device).to(torch.bfloat16)
        return mixed, _probe_weight(qn.weight, 100), _probe_weight(kn.weight, 101)

    def fused(m, wq, wk):
        cos_, sin_ = rope.cos_sin(m.dtype)
        (gq, zc), (gk, _) = _gamma(qn, wq, in_kernel), _gamma(kn, wk, in_kernel)
        return _QKNormRope.apply(m, gq, gk, cos_, sin_, hpg, qn.eps, zc)

    def eager(m, wq, wk):
        q, k, v = torch.split(m, [hpg * D, D, D], dim=3)
        q = torch.func.functional_call(qn, {"weight": wq}, (q.reshape(S, Bt, NG * hpg, D),))
        k = torch.func.functional_call(kn, {"weight": wk}, (k,))
        return rope.apply(q, False), rope.apply(k, False), v

    return bitwise_check.passed(
        "msa_fused_qk_norm_rope",
        lambda: bitwise_check.outputs_and_grads(fused, *inputs()),
        lambda: bitwise_check.outputs_and_grads(eager, *inputs()),
        # Turbo picks its launch configs by row count; the in-kernel gamma is another kernel variant
        key=(S, Bt, NG, hpg) + (("in-kernel gamma",) if zc_key else ()),
    )


def eligible(attn, hidden_states, rope) -> bool:
    """Whether :func:`qk_norm_rope` can stand in for upstream's qkv split, q/k norms and
    ``rope.apply`` in ``attn`` (q and k share ``rope``)."""
    cfg = attn.config
    qn, kn = getattr(attn, "q_layernorm", None), getattr(attn, "k_layernorm", None)
    D = attn.hidden_size_per_attention_head
    if (
        cfg.rotary_interleaved
        or getattr(cfg, "test_mode", False)
        or cfg.num_query_groups < attn.world_size
        or D != 128
        or not (hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16)
        or not _takes_split_qkv(type(attn))
        or not (_turbo_norm_ok(qn, D) and _turbo_norm_ok(kn, D))
        or qn.eps != kn.eps
        or qn.zero_centered_gamma != kn.zero_centered_gamma
        or qn.weight.dtype != kn.weight.dtype
    ):
        return False
    tables = _rope_tables(rope, D)
    S, Bt = hidden_states.shape[0], hidden_states.shape[1]
    if tables is None or tables[0].shape[0] != S:
        return False
    hpg = attn.num_attention_heads_per_partition // attn.num_query_groups_per_partition
    nk = S * Bt * attn.num_query_groups_per_partition
    return (
        _multi_row(D, nk * hpg)
        and _multi_row(D, nk)
        and _qk_self_check(attn, S, Bt, rope, _in_kernel_gamma(cfg))
    )


def qk_norm_rope(attn, mixed_qkv, rope):
    """``(query, key, value)`` exactly as upstream's split, q/k norms and ``rope.apply``:
    q ``[sq, b, np, hn]`` and k/v ``[sq, b, ng, hn]``, contiguous."""
    qn, kn = attn.q_layernorm, attn.k_layernorm
    cos_, sin_ = rope.cos_sin(mixed_qkv.dtype)
    hpg = attn.num_attention_heads_per_partition // attn.num_query_groups_per_partition
    in_kernel = _in_kernel_gamma(getattr(attn, "config", None))
    (gq, zc), (gk, _) = _gamma(qn, qn.weight, in_kernel), _gamma(kn, kn.weight, in_kernel)
    return _QKNormRope.apply(mixed_qkv.contiguous(), gq, gk, cos_, sin_, hpg, qn.eps, zc)


class _NormRope(torch.autograd.Function):
    """Turbo RMSNorm then ``RopeTables.apply`` on a contiguous ``[sq, b, n, hn]`` tensor, one pass
    each way (the indexer's q/k)."""

    @staticmethod
    def forward(ctx, x, gamma, cos_, sin_, eps, zc=False):
        S, Bt, N, D = x.shape
        n = S * Bt * N
        out, rstd = _norm_rope_fwd(x, 0, 1, D, n, Bt * N, gamma, eps, cos_, sin_, zc)
        ctx.save_for_backward(x, gamma, rstd, cos_, sin_)
        ctx.zc = zc
        return out.view(S, Bt, N, D)

    @staticmethod
    def backward(ctx, dout):
        x, gamma, rstd, cos_, sin_ = ctx.saved_tensors
        S, Bt, N, D = x.shape
        dx = torch.empty_like(x)
        dg = _rope_norm_bwd(
            dout.contiguous(), x, dx, 0, 1, D, S * Bt * N, Bt * N, gamma, rstd, cos_, sin_, ctx.zc
        )
        return dx, dg, None, None, None, None


def _norm_rope_self_check(shape, norm, rope, in_kernel: bool = False) -> bool:
    """The kernels against ``rope.apply(norm(x))`` on random rows of the training shape, with
    gamma moved off its init value."""

    def inputs():
        x = bitwise_check.randn(shape, norm.weight.device).to(torch.bfloat16)
        return x, _probe_weight(norm.weight, 100)

    def fused(x, w):
        cos_, sin_ = rope.cos_sin(x.dtype)
        gamma, zc = _gamma(norm, w, in_kernel)
        return _NormRope.apply(x, gamma, cos_, sin_, norm.eps, zc)

    def eager(x, w):
        return rope.apply(torch.func.functional_call(norm, {"weight": w}, (x,)), False)

    return bitwise_check.passed(
        "msa_fused_norm_rope",
        lambda: bitwise_check.outputs_and_grads(fused, *inputs()),
        lambda: bitwise_check.outputs_and_grads(eager, *inputs()),
        # Turbo picks its launch configs by row count; the in-kernel gamma is another kernel variant
        key=tuple(shape)
        + (("in-kernel gamma",) if in_kernel and _gamma(norm, norm.weight.detach(), True)[1] else ()),
    )


def norm_rope_ok(x, norm, rope, rotary_interleaved, in_kernel_gamma: bool = False) -> bool:
    """Whether :func:`norm_rope` (called with the same ``in_kernel_gamma``) can stand in for
    ``rope.apply(norm(x))``."""
    if rotary_interleaved or rope is None:
        return False
    D = x.shape[-1]
    if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 4 and x.is_contiguous() and D == 128):
        return False
    if not _turbo_norm_ok(norm, D):
        return False
    tables = _rope_tables(rope, D)
    return (
        tables is not None
        and tables[0].shape[0] == x.shape[0]
        and _multi_row(D, x.numel() // D)
        and _norm_rope_self_check(x.shape, norm, rope, in_kernel_gamma)
    )


def norm_rope(x, norm, rope, in_kernel_gamma: bool = False):
    """``rope.apply(norm(x))`` bitwise, for :func:`norm_rope_ok` inputs."""
    cos_, sin_ = rope.cos_sin(x.dtype)
    gamma, zc = _gamma(norm, norm.weight, in_kernel_gamma)
    return _NormRope.apply(x, gamma, cos_, sin_, norm.eps, zc)

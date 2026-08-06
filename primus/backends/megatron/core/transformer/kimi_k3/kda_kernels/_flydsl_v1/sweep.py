###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The inter-chunk state sweep, as one fused kernel launch plus batched GEMMs.

WP9's forward spent 95 % of its time in this stage while it was a 64-step Python
loop over torch GEMMs (``wp9/RESULTS.md`` §3). The fix has two halves:

* :mod:`.kda_state_sweep_kernel` runs the part that is **genuinely sequential**
  — the ``[K, V]`` state recurrence — in a single launch, one workgroup per
  ``(B*H, V/BV)``, state resident in LDS across the whole chunk loop.
* everything else is per-chunk and therefore *not* sequential, so it comes back
  out as batched GEMMs over all ``NC`` chunks at once. On the forward that is
  just ``O_n = Aqk_n @ T_n + Rq_n``; on the backward it is all five input
  adjoints.

:func:`state_sweep_torch` is the pure-torch twin: same recurrence, same rounding
points, so it is both the unit-test oracle for the kernel and the fallback when
the geometry is unsupported or ``flydsl`` is absent. It is the same discipline
:func:`..ops.decay_scores_torch` applies to the score kernel.

The adjoint
-----------
:class:`_FusedSweep` carries a hand-written backward rather than letting autograd
walk the recurrence, because a custom kernel is opaque to autograd. It is the
exact adjoint of the four lines of the forward::

    Rq_n = QG_n @ S_n                        dQG_n  =  dO_n @ S_n^T
    T_n  = U_n - W_n @ S_n                   dU_n   =  dT_n
    O_n  = Aqk_n @ T_n + Rq_n                dAqk_n =  dO_n @ T_n^T
    S_n+1 = dec_n * S_n + KG_n^T @ T_n       dKG_n  =  T_n @ P_n^T
                                             dW_n   = -dT_n @ S_n^T
                                             ddec_n =  sum_v S_n * P_n

with ``P_n = dS_{n+1}`` and ``dT_n`` themselves produced by *the same kernel*
run backwards — ``dT_n = Aqk_n^T dO_n + KG_n P_n``,
``P_{n-1} = dec_n * P_n - W_n^T dT_n + QG_n^T dO_n``. So the sequential half of
the backward costs one more launch and the rest is six batched GEMMs. The
forward's ``S_n`` and ``T_n`` are the only things kept between the passes, and
only when a gradient is actually wanted.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import torch

__all__ = [
    "state_sweep_torch",
    "fused_chunk_sweep",
    "sweep_operand_mode",
    "supports_sweep",
]

# Narrowest first. Pass 2 reasoned that 64 was the sweet spot because narrower
# blocks re-read the state-independent operands once per block; pass 4 measured
# the grid instead and the reasoning was wrong. On the production sweep
# (`wp9_p4/results/probe_sweep_tune.json`, identical operands, output bit-equal
# at every setting):
#
#     block_v   128     64      32      16
#     us       1395    786     802    527
#
# 16 re-reads `Amat` and `Xt` eight times instead of twice — 1.8 GB of extra
# HBM traffic — and is still 1.5x faster, because the kernel is latency bound,
# not bandwidth bound: 64 puts one workgroup of four waves on each of 192 CUs,
# i.e. **one wave per SIMD**, and its eight scalar `ds_read_b32` per MFMA then
# have nothing to hide behind. 16 puts 768 workgroups on 256 CUs. `waves_per_eu`
# made no difference at any width (< 1 %), so it stays at the build default.
#
# The support set does not shrink: `supports_sweep_geometry` already requires
# `chunk_size` and `head_dim` to be multiples of 64 (one 16-row MFMA tile per
# wave, four waves), which is exactly what `block_v = 16` needs of them.
_BLOCK_V_CHOICES = (16, 64, 32)
_KERNEL_CACHE: Dict[Tuple[Any, ...], Any] = {}
_KERNEL_LOCK = threading.Lock()


def _pick_block_v(v_dim: int) -> int:
    """The fastest measured ``V`` block that divides ``v_dim``."""
    for bv in _BLOCK_V_CHOICES:
        if v_dim % bv == 0:
            return bv
    return v_dim

# bf16 is the only input dtype routed onto the MFMA path: v_mfma_f32_16x16x32_bf16
# rounds its operands to bf16, which is exactly what `fla` does and is lossless
# for bf16 inputs, but would cost fp16 four mantissa bits and fp32 sixteen.
_MFMA_DTYPES = (torch.bfloat16,)


def sweep_operand_mode(dtype: torch.dtype) -> Tuple[str, torch.dtype]:
    """``(kernel mode, operand dtype)`` for an input dtype."""
    if dtype in _MFMA_DTYPES:
        return "mfma", torch.bfloat16
    return "valu", torch.float32


def supports_sweep(chunk_size: int, k_dim: int, v_dim: int) -> Optional[str]:
    """``None`` when the fused kernel can run this geometry, else why it cannot."""
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.kda_state_sweep_kernel import (  # noqa: E501
        supports_sweep_geometry,
    )

    return supports_sweep_geometry(chunk_size, k_dim, v_dim, _pick_block_v(v_dim))


# ---------------------------------------------------------------------------
# the recurrence, in torch
# ---------------------------------------------------------------------------


def state_sweep_torch(
    amat: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    dec: torch.Tensor,
    s0: torch.Tensor,
    *,
    num_chunks: int,
    sgn_t: float,
    sgn_x: float,
    e_in: Optional[torch.Tensor] = None,
    reverse: bool = False,
    emit_rq: bool = True,
    op_dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch twin of :mod:`.kda_state_sweep_kernel`.

    ``amat: [NB, MO, K]``, ``yc: [NB, C, V]``, ``xt: [NB, K, C]``,
    ``dec: [NB, K]``, ``s0: [B*H, K, V]``, ``e_in: [NB, K, V]``.
    Returns ``(rq, t, states, s_final)`` with ``rq``/``t`` ``[NB, C, V]``,
    ``states [NB, K, V]`` and ``s_final [B*H, K, V]``.

    ``op_dtype`` is where the kernel's rounding happens: it rounds ``amat`` and
    ``xt`` (its MFMA **A** operands) and the state and ``t`` it reads back out of
    LDS (its **B** operands), and nothing else. Quantising in exactly those four
    places and nowhere else is what makes this a twin rather than merely close.
    """
    nb, mo, k_dim = amat.shape
    chunk = yc.shape[-2]
    v_dim = yc.shape[-1]
    nbh = nb // num_chunks

    def q(x):
        return x.to(op_dtype).to(torch.float32)

    amat, xt, yc = q(amat), q(xt), yc.float()
    dec = dec.float()
    state = s0.float().clone()
    dev = amat.device

    rq = torch.empty(nb, chunk, v_dim, dtype=torch.float32, device=dev) if emit_rq else None
    t_all = torch.empty(nb, chunk, v_dim, dtype=torch.float32, device=dev)
    states = torch.empty(nb, k_dim, v_dim, dtype=torch.float32, device=dev)

    # [NB, ...] is chunk-major within each (b, h), so a plain view exposes the
    # chunk axis and every step below is a slice rather than a gather.
    def per_chunk(x, *tail):
        return None if x is None else x.view(nbh, num_chunks, *tail)

    amat_c, yc_c, xt_c = (
        per_chunk(amat, mo, k_dim),
        per_chunk(yc, chunk, v_dim),
        per_chunk(xt, k_dim, chunk),
    )
    dec_c, e_c = per_chunk(dec, k_dim), per_chunk(e_in, k_dim, v_dim)
    rq_c, t_c, st_c = (
        per_chunk(rq, chunk, v_dim),
        per_chunk(t_all, chunk, v_dim),
        per_chunk(states, k_dim, v_dim),
    )

    a_row_t = chunk if emit_rq else 0
    for n in range(num_chunks - 1, -1, -1) if reverse else range(num_chunks):
        st_c[:, n] = state
        # the state is *carried* in fp32 but read back as an operand at op_dtype,
        # which is where the kernel truncates its LDS copy for the MFMA fragment
        s_op = q(state)
        if emit_rq:
            rq_c[:, n] = amat_c[:, n, :chunk] @ s_op
        # T stays fp32 on the way out and is only rounded where it becomes an
        # operand, which is what the kernel's fp32 LDS copy does.
        tv = yc_c[:, n] + sgn_t * (amat_c[:, n, a_row_t : a_row_t + chunk] @ s_op)
        t_c[:, n] = tv
        state = dec_c[:, n].unsqueeze(-1) * state + sgn_x * (xt_c[:, n] @ q(tv))
        if e_c is not None:
            state = state + e_c[:, n].float()
    return rq, t_all, states, state


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------


def _get_kernel(**kw):
    key = tuple(sorted(kw.items()))
    with _KERNEL_LOCK:
        launch = _KERNEL_CACHE.get(key)
        if launch is None:
            from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.kda_state_sweep_kernel import (  # noqa: E501
                build_kda_state_sweep,
            )

            launch = build_kda_state_sweep(**kw)
            _KERNEL_CACHE[key] = launch
        return launch


def _sweep_kernel(
    amat: torch.Tensor,
    yc: torch.Tensor,
    xt: torch.Tensor,
    dec: torch.Tensor,
    s0: torch.Tensor,
    *,
    num_chunks: int,
    sgn_t: float,
    sgn_x: float,
    e_in: Optional[torch.Tensor],
    reverse: bool,
    emit_rq: bool,
    emit_states: bool,
    mode: str,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Launch the fused sweep. Shapes and dtypes as in :func:`state_sweep_torch`."""
    nb, _, k_dim = amat.shape
    chunk, v_dim = yc.shape[-2], yc.shape[-1]
    nbh = nb // num_chunks
    dev, op_dtype = amat.device, amat.dtype

    launch = _get_kernel(
        chunk_size=chunk,
        k_dim=k_dim,
        v_dim=v_dim,
        block_v=_pick_block_v(v_dim),
        mode=mode,
        emit_rq=bool(emit_rq),
        emit_states=bool(emit_states),
        has_e=e_in is not None,
        sgn_t=float(sgn_t),
        sgn_x=float(sgn_x),
        reverse=bool(reverse),
    )

    def _dummy(dtype):
        return torch.empty(1, dtype=dtype, device=dev)

    rq = (
        torch.empty(nb, chunk, v_dim, dtype=torch.float32, device=dev)
        if emit_rq
        else _dummy(torch.float32)
    )
    t_all = torch.empty(nb, chunk, v_dim, dtype=torch.float32, device=dev)
    states = (
        torch.empty(nb, k_dim, v_dim, dtype=torch.float32, device=dev)
        if emit_states
        else _dummy(torch.float32)
    )
    s_final = torch.empty(nbh, k_dim, v_dim, dtype=torch.float32, device=dev)

    launch(
        amat.reshape(-1),
        yc.reshape(-1),
        xt.reshape(-1),
        dec.reshape(-1),
        (e_in if e_in is not None else _dummy(torch.float32)).reshape(-1),
        s0.reshape(-1),
        rq.reshape(-1),
        t_all.reshape(-1),
        states.reshape(-1),
        s_final.reshape(-1),
        int(nbh),
        int(num_chunks),
    )
    return (rq if emit_rq else None), t_all, (states if emit_states else None), s_final


def _as_operand(x: torch.Tensor, op_dtype: torch.dtype) -> torch.Tensor:
    """``x`` as a contiguous operand-dtype tensor, without a needless copy."""
    if x.dtype is op_dtype and x.is_contiguous():
        return x
    return x.to(op_dtype).contiguous()


def _transposed_operand(x: torch.Tensor, op_dtype: torch.dtype) -> torch.Tensor:
    """``x.transpose(-1, -2)`` contiguous at ``op_dtype``, in **one** pass.

    The MFMA A operand has to have the contraction index contiguous, so ``KG``
    and ``W`` are handed to the kernel transposed. Spelling that as
    ``.transpose().to().contiguous()`` costs two passes over ~100 MB because
    ``.to()`` preserves strides; a strided ``copy_`` into a fresh contiguous
    buffer converts and transposes at the same time. Safe here because
    ``autograd.Function`` bodies run with grad disabled and the adjoint is
    hand-written.
    """
    src = x.transpose(-1, -2)
    out = torch.empty(src.shape, dtype=op_dtype, device=x.device)
    out.copy_(src)
    return out


def _run_sweep(use_kernel: bool, mode: str, op_dtype: torch.dtype, *, emit_states: bool, **kw):
    """Kernel when available, twin otherwise, with one calling convention."""
    if use_kernel:
        return _sweep_kernel(mode=mode, emit_states=emit_states, **kw)
    rq, t_all, states, s_final = state_sweep_torch(op_dtype=op_dtype, **kw)
    return rq, t_all, (states if emit_states else None), s_final


# ---------------------------------------------------------------------------
# autograd
# ---------------------------------------------------------------------------


class _FusedSweep(torch.autograd.Function):
    """The chunk recurrence and its analytic adjoint."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        qw: torch.Tensor,  # [NB, 2C, K]  stacked [QG; W]
        u: torch.Tensor,  # [NB, C, V]
        aqk: torch.Tensor,  # [NB, C, C]
        kg: torch.Tensor,  # [NB, C, K]
        dec: torch.Tensor,  # [NB, K]
        s0: Optional[torch.Tensor],  # [B*H, K, V]
        num_chunks: int,
        use_kernel: bool,
        mode: str,
        op_dtype: torch.dtype,
        scale: float = 1.0,
    ):
        nb, two_c, k_dim = qw.shape
        chunk = two_c // 2
        v_dim = u.shape[-1]
        nbh = nb // num_chunks
        dev = qw.device
        want_grad = any(ctx.needs_input_grad[:6])

        zero_state = s0 is None
        s0_t = (
            torch.zeros(nbh, k_dim, v_dim, dtype=torch.float32, device=dev)
            if zero_state
            else s0.float().contiguous()
        )
        rq, t_all, states, s_final = _run_sweep(
            use_kernel,
            mode,
            op_dtype,
            emit_states=want_grad,
            amat=_as_operand(qw, op_dtype),
            yc=_as_operand(u, torch.float32),
            xt=_transposed_operand(kg, op_dtype),
            dec=dec.float().contiguous(),
            s0=s0_t,
            num_chunks=num_chunks,
            sgn_t=-1.0,
            sgn_x=1.0,
            e_in=None,
            reverse=False,
            emit_rq=True,
        )
        # O_n = scale * (Aqk_n @ T_n + Rq_n) -- per chunk, so one batched GEMM
        # over all NC. In fp32: `aqk` carries the intra-chunk term and is the one
        # operand the kernel never rounds, so rounding it here would throw away
        # accuracy the sweep already paid for.
        #
        # `softmax_scale` rides on this GEMM's alpha and beta rather than being
        # applied to `q` upstream. Both output terms are linear in `q` and the
        # carried state is not a function of `q` at all, so scaling here is
        # exactly equivalent -- and free, where scaling `q` cost a 400 MB pass
        # in the forward and another in the backward's recompute.
        o = torch.baddbmm(rq, aqk.float(), t_all, beta=scale, alpha=scale)

        ctx.save_for_backward(qw, aqk, kg, dec, t_all, states)
        ctx.cfg = (num_chunks, use_kernel, mode, op_dtype, chunk, zero_state, scale)
        return o, s_final

    @staticmethod
    def backward(ctx, do: torch.Tensor, dsf: Optional[torch.Tensor]):  # type: ignore[override]
        qw, aqk, kg, dec, t_all, states = ctx.saved_tensors
        num_chunks, use_kernel, mode, op_dtype, chunk, zero_state, scale = ctx.cfg
        nb, _, k_dim = qw.shape
        v_dim = t_all.shape[-1]
        nbh = nb // num_chunks
        # `o = scale * (Rq + Aqk @ T)`, and `do` reaches every one of this
        # function's five outputs only through those two terms, so folding the
        # scale in once here is the whole adjoint of the forward's alpha/beta.
        do = do.float().contiguous()
        if scale != 1.0:
            do = do * scale
        qg, w = qw[:, :chunk], qw[:, chunk:]

        # the two state-independent adjoint sources, batched over every chunk
        y = aqk.float().transpose(-1, -2) @ do  # [NB, C, V]
        e_in = qg.float().transpose(-1, -2) @ do  # [NB, K, V]
        p0 = (
            dsf.float().contiguous()
            if dsf is not None
            else torch.zeros(nbh, k_dim, v_dim, dtype=torch.float32, device=do.device)
        )

        _, dvt, p_all, ds0 = _run_sweep(
            use_kernel,
            mode,
            op_dtype,
            emit_states=True,
            amat=_as_operand(kg, op_dtype),
            yc=_as_operand(y, torch.float32),
            xt=_transposed_operand(w, op_dtype),
            dec=dec.float().contiguous(),
            s0=p0,
            num_chunks=num_chunks,
            sgn_t=1.0,
            sgn_x=-1.0,
            e_in=e_in,
            reverse=True,
            emit_rq=False,
        )
        st = states.transpose(-1, -2)  # [NB, V, K]

        needs = ctx.needs_input_grad
        d_aqk = do @ t_all.transpose(-1, -2) if needs[2] else None
        d_kg = t_all @ p_all.transpose(-1, -2) if needs[3] else None
        d_qw = None
        if needs[0]:
            d_qw = torch.cat((do @ st, -(dvt @ st)), dim=-2)
        d_u = dvt if needs[1] else None
        d_dec = (states * p_all).sum(-1) if needs[4] else None
        d_s0 = ds0 if (needs[5] and not zero_state) else None
        return d_qw, d_u, d_aqk, d_kg, d_dec, d_s0, None, None, None, None, None


def fused_chunk_sweep(
    qw: torch.Tensor,
    u: torch.Tensor,
    aqk: torch.Tensor,
    kg: torch.Tensor,
    dec: torch.Tensor,
    s0: Optional[torch.Tensor],
    *,
    num_chunks: int,
    op_dtype: torch.dtype = torch.float32,
    use_kernel: bool = True,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(o, final_state)`` for the whole chunk sweep. ``o: [NB, C, V]`` fp32.

    ``qw`` is the stacked ``[QG; W]``; see :mod:`.chunk` for how the operands are
    built. ``op_dtype`` selects the arithmetic the kernel does — pass the caller's
    input dtype through :func:`sweep_operand_mode`. ``scale`` is the attention's
    ``softmax_scale``, applied to the chunk output rather than to ``q``.
    """
    mode, _ = sweep_operand_mode(op_dtype)
    return _FusedSweep.apply(
        qw, u, aqk, kg, dec, s0, int(num_chunks), bool(use_kernel), mode, op_dtype,
        float(scale),
    )

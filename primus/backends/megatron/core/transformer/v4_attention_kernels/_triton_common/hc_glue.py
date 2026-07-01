###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused HyperConnection elemwise glue (plan-6 P37).

Fuses the post-linear elemwise tail of
:meth:`primus.backends.megatron.core.transformer.hyper_connection.HyperMixer.compute_weights`
into a single FWD + single BWD Triton kernel pair.  Eager body:

.. code-block:: python

    pre_logit  = logits[..., :K]     * scale[0] + base[:K]
    post_logit = logits[..., K:2K]   * scale[1] + base[K:2K]
    comb_logit = (logits[..., 2K:].view(..., K, K) * scale[2]
                  + base[2K:].view(K, K))

    pre  = sigmoid(pre_logit)  + eps          # (eps, 1+eps]
    post = 2 * sigmoid(post_logit)            # (0, 2)
    comb_pre_sinkhorn = softmax(comb_logit, dim=-1) + eps

After P36 the trailing ``sinkhorn_normalize(comb_pre_sinkhorn)`` is its
own Triton kernel.  P37 fuses everything **between** the
``_packed_logits`` GEMM and the Sinkhorn call into one kernel:

* 3 slices of ``logits`` (free in Triton — pointer arithmetic);
* 3 fused-multiply-adds against ``scale`` / ``base``;
* 2 sigmoid + 1 softmax + 2 eps adds;
* 1 final 2x multiply for ``post``.

The eager chain is ~8 elementwise GPU launches per call (P32 trace
attributes ~3-5 ms / iter across all 16 ``HyperConnection`` invocations
to the ``elementwise_kernel_manual_unroll<128, 8>`` bucket).  P37
collapses those into **one** FWD kernel + **one** BWD kernel.

The matmul inside ``_packed_logits`` (`F.linear(flat * rsqrt, W)`)
stays as ``torch.nn.functional.linear`` -- GEMM is already
GPU-bound and fusing it into the elemwise chain would re-implement a
GEMM badly.  Same goes for the ``collapse`` reduce and the
``expand`` outer-product + matmul -- those live around matmuls and
are not net wins as separate Triton kernels (deferred to a possible
P37b if the residual trace shows otherwise).

Gating: routed through :func:`hc_glue_compute_tail_triton` when
``PRIMUS_HC_TRITON != "0"`` (default-on).  Set to ``"0"`` to fall back
to the eager body kept in
:func:`primus.backends.megatron.core.transformer.hyper_connection.HyperMixer.compute_weights`.

Supported shape constraints:

* ``K`` (= ``hc_mult``) must be a power of 2 in ``{1, 2, 4, 8, 16}``.
  V4-Flash uses ``K=4``.
* ``logits`` must be contiguous; the wrapper calls ``.contiguous()``
  defensively.
* Any leading shape is supported (the wrapper flattens to ``[N, (2+K)*K]``).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_TORCH_TO_TL_DTYPE = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _triton_dtype(t: torch.dtype):
    try:
        return _TORCH_TO_TL_DTYPE[t]
    except KeyError as exc:
        raise TypeError(
            f"hc_glue: unsupported dtype {t}; expected one of {list(_TORCH_TO_TL_DTYPE)}"
        ) from exc


# Triton requires power-of-2 block extents, and the register footprint is
# linear in K + K*K, so we cap K at 16 for the in-register path.  V4-Flash
# uses K=4; larger K would require a different layout.
_SUPPORTED_K = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _hc_compute_tail_fwd_kernel(
    LOGITS_PTR,  # [N, (2+K)*K] contiguous, fp32
    SCALE_PTR,  # [3] fp32
    BASE_PTR,  # [(2+K)*K] fp32
    PRE_PTR,  # [N, K] OUT_DTYPE
    POST_PTR,  # [N, K] OUT_DTYPE
    COMB_PTR,  # [N, K, K] OUT_DTYPE (softmax+eps; before Sinkhorn)
    PRE_SIG_PTR,  # [N, K] fp32 -- saved-for-backward (sigmoid(pre_logit))
    POST_SIG_PTR,  # [N, K] fp32 -- saved-for-backward (sigmoid(post_logit))
    COMB_SM_PTR,  # [N, K, K] fp32 -- saved-for-backward (softmax(comb_logit))
    N,
    EPS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program tile of ``BLOCK_LEADING`` rows of the leading axis.

    Reads ``LOGITS[BLOCK_LEADING, (2+K)*K]`` once, applies the elemwise
    tail in fp32 registers, writes the three output tensors (cast to
    OUT_DTYPE) plus three fp32 saved-for-backward states.

    Register footprint per program (at K=4, BLOCK_LEADING=64):
        BLOCK_LEADING * (2+K)*K        = 64 * 24 = 1536 fp32 logits
        BLOCK_LEADING * K              =     64 * 4 =   256 pre/post
        BLOCK_LEADING * K * K          =    64 * 16 =  1024 comb
    Total ~2800 fp32 = ~11 KiB; comfortable for MI355 256-VGPR warps.
    """

    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    k_idx = tl.arange(0, K)

    # Load the three scale scalars (broadcast across all rows).
    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)

    # Base partitions: base[:K], base[K:2K], base[2K:].view(K, K).
    base_pre = tl.load(BASE_PTR + k_idx)
    base_post = tl.load(BASE_PTR + K + k_idx)
    base_comb = tl.load(BASE_PTR + 2 * K + k_idx[:, None] * K + k_idx[None, :])

    KK_TOTAL: tl.constexpr = (2 + K) * K

    # Logits row stride is KK_TOTAL.  Pre slice = [:, :K]; post = [:, K:2K];
    # comb = [:, 2K:].view(K, K).  All three are contiguous reads.
    pre_logit = tl.load(
        LOGITS_PTR + offs[:, None] * KK_TOTAL + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    post_logit = tl.load(
        LOGITS_PTR + offs[:, None] * KK_TOTAL + K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    comb_logit = tl.load(
        LOGITS_PTR + offs[:, None, None] * KK_TOTAL + 2 * K + k_idx[None, :, None] * K + k_idx[None, None, :],
        mask=mask_leading[:, None, None],
        other=0.0,
    )

    # Apply scale + base.
    pre_logit = pre_logit * scale0 + base_pre[None, :]
    post_logit = post_logit * scale1 + base_post[None, :]
    comb_logit = comb_logit * scale2 + base_comb[None, :, :]

    # Sigmoids (per-element).
    pre_sig = tl.sigmoid(pre_logit)
    post_sig = tl.sigmoid(post_logit)

    # Softmax along axis=2 (the inner K of comb_logit).  Standard
    # numerically-stable softmax: subtract row max, exp, divide by sum.
    cmax = tl.max(comb_logit, axis=2, keep_dims=True)
    cexp = tl.exp(comb_logit - cmax)
    csum = tl.sum(cexp, axis=2, keep_dims=True)
    comb_sm = cexp / csum

    # Output tensors (with eps additions and the 2x for post).
    pre_out = pre_sig + EPS
    post_out = 2.0 * post_sig
    comb_out = comb_sm + EPS

    # Save fp32 states for backward.
    tl.store(
        PRE_SIG_PTR + offs[:, None] * K + k_idx[None, :],
        pre_sig,
        mask=mask_leading[:, None],
    )
    tl.store(
        POST_SIG_PTR + offs[:, None] * K + k_idx[None, :],
        post_sig,
        mask=mask_leading[:, None],
    )
    tl.store(
        COMB_SM_PTR + offs[:, None, None] * K * K + k_idx[None, :, None] * K + k_idx[None, None, :],
        comb_sm,
        mask=mask_leading[:, None, None],
    )

    # Write the three caller-visible outputs (cast to OUT_DTYPE).
    tl.store(
        PRE_PTR + offs[:, None] * K + k_idx[None, :],
        pre_out.to(OUT_DTYPE),
        mask=mask_leading[:, None],
    )
    tl.store(
        POST_PTR + offs[:, None] * K + k_idx[None, :],
        post_out.to(OUT_DTYPE),
        mask=mask_leading[:, None],
    )
    tl.store(
        COMB_PTR + offs[:, None, None] * K * K + k_idx[None, :, None] * K + k_idx[None, None, :],
        comb_out.to(OUT_DTYPE),
        mask=mask_leading[:, None, None],
    )


@triton.jit
def _hc_compute_tail_bwd_kernel(
    DPRE_PTR,  # [N, K] grad in OUT_DTYPE
    DPOST_PTR,  # [N, K] grad in OUT_DTYPE
    DCOMB_PTR,  # [N, K, K] grad in OUT_DTYPE
    PRE_SIG_PTR,  # [N, K] fp32 saved
    POST_SIG_PTR,  # [N, K] fp32 saved
    COMB_SM_PTR,  # [N, K, K] fp32 saved
    SCALE_PTR,  # [3] fp32 (read)
    DLOGITS_PTR,  # [N, (2+K)*K] OUT (fp32)
    DSCALE_PTR,  # [N, 3] partial sums (fp32); reduced host-side
    DBASE_PTR,  # [N, (2+K)*K] partial sums (fp32); reduced host-side
    N,
    K: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
):
    """VJP through the elemwise tail.

    Forward:
        pre_logit = logits[:, :K] * s0 + base[:K]
        post_logit = logits[:, K:2K] * s1 + base[K:2K]
        comb_logit = logits[:, 2K:].view(K, K) * s2 + base[2K:].view(K, K)
        pre = sigmoid(pre_logit) + eps
        post = 2 * sigmoid(post_logit)
        comb_sm = softmax(comb_logit, axis=-1); comb = comb_sm + eps

    Backward (per element, all in fp32):
        d_pre_logit  = d_pre  * pre_sig * (1 - pre_sig)
        d_post_logit = d_post * 2 * post_sig * (1 - post_sig)
        d_comb_logit = comb_sm * (d_comb - sum(d_comb * comb_sm, axis=-1))

        d_logits[:, :K]   = d_pre_logit  * s0
        d_logits[:, K:2K] = d_post_logit * s1
        d_logits[:, 2K:]  = d_comb_logit.view(K*K) * s2

        d_scale[0] = sum(logits[:, :K]   * d_pre_logit)
        d_scale[1] = sum(logits[:, K:2K] * d_post_logit)
        d_scale[2] = sum(logits[:, 2K:].view(K, K) * d_comb_logit)

        d_base[:K]   = sum_n d_pre_logit
        d_base[K:2K] = sum_n d_post_logit
        d_base[2K:]  = sum_n d_comb_logit

    Notes:
        * d_scale needs the *original* fp32 logits.  To avoid carrying
          a fourth saved tensor we re-derive logits from the existing
          state: that's not possible (sigmoid is not invertible at
          large magnitudes).  Instead, we save d_scale as a per-row
          partial sum that the host-side wrapper reduces with a
          torch.sum at the end.  Wait -- we actually need the logits
          themselves for d_scale.  Solution: store ``logits * d_*``
          (the cross-term) directly without ever materialising logits
          again -- this is just the elementwise product of the
          recomputed (d_pre_logit, d_post_logit, d_comb_logit) with
          the *input-side* gradient of each, taken at the pre-scale
          point.  Concretely, since
              d_pre = sigmoid(s0 * x + b)            (forward eq)
              ∂loss/∂s0 = ∂loss/∂pre_logit * x       (chain rule)
          and the BWD kernel does NOT have ``x = pre_logits_pre_scale``
          directly available (logits is a function input passed in
          fp32).  We therefore accept the per-row partials being
          a function of ``logits`` and load logits inside the BWD
          kernel.  See the LOGITS_PTR arg added below.

    For simplicity and correctness, the wrapper passes the *fp32
    logits* tensor into BWD and the kernel recomputes the pre-scale
    cross-term inline.  At V4-Flash widths the extra HBM read is
    negligible (24 fp32 / row vs the K*K state already loaded).
    """

    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    k_idx = tl.arange(0, K)

    # Read scale (broadcast scalars).
    scale0 = tl.load(SCALE_PTR + 0)
    scale1 = tl.load(SCALE_PTR + 1)
    scale2 = tl.load(SCALE_PTR + 2)

    # Saved forward states.
    pre_sig = tl.load(
        PRE_SIG_PTR + offs[:, None] * K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    post_sig = tl.load(
        POST_SIG_PTR + offs[:, None] * K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    )
    comb_sm = tl.load(
        COMB_SM_PTR + offs[:, None, None] * K * K + k_idx[None, :, None] * K + k_idx[None, None, :],
        mask=mask_leading[:, None, None],
        other=0.0,
    )

    # Upstream grads (cast to fp32 for the chain).
    d_pre = tl.load(
        DPRE_PTR + offs[:, None] * K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    ).to(tl.float32)
    d_post = tl.load(
        DPOST_PTR + offs[:, None] * K + k_idx[None, :],
        mask=mask_leading[:, None],
        other=0.0,
    ).to(tl.float32)
    d_comb = tl.load(
        DCOMB_PTR + offs[:, None, None] * K * K + k_idx[None, :, None] * K + k_idx[None, None, :],
        mask=mask_leading[:, None, None],
        other=0.0,
    ).to(tl.float32)

    # Sigmoid VJP (no eps contribution -- + EPS has derivative 1):
    d_pre_logit = d_pre * pre_sig * (1.0 - pre_sig)
    d_post_logit = d_post * 2.0 * post_sig * (1.0 - post_sig)

    # Softmax VJP: d_comb_logit = comb_sm * (d_comb - sum(d_comb * comb_sm, axis=2))
    dot = tl.sum(d_comb * comb_sm, axis=2, keep_dims=True)
    d_comb_logit = comb_sm * (d_comb - dot)

    KK_TOTAL: tl.constexpr = (2 + K) * K

    # Write d_logits (= scale * d_*_logit at each slice).
    tl.store(
        DLOGITS_PTR + offs[:, None] * KK_TOTAL + k_idx[None, :],
        d_pre_logit * scale0,
        mask=mask_leading[:, None],
    )
    tl.store(
        DLOGITS_PTR + offs[:, None] * KK_TOTAL + K + k_idx[None, :],
        d_post_logit * scale1,
        mask=mask_leading[:, None],
    )
    tl.store(
        DLOGITS_PTR
        + offs[:, None, None] * KK_TOTAL
        + 2 * K
        + k_idx[None, :, None] * K
        + k_idx[None, None, :],
        d_comb_logit * scale2,
        mask=mask_leading[:, None, None],
    )

    # d_base accumulates the d_*_logit values directly (no scale factor).
    # We write per-row partials and let host-side torch.sum reduce them
    # (avoids cross-block atomic_add).
    tl.store(
        DBASE_PTR + offs[:, None] * KK_TOTAL + k_idx[None, :],
        d_pre_logit,
        mask=mask_leading[:, None],
    )
    tl.store(
        DBASE_PTR + offs[:, None] * KK_TOTAL + K + k_idx[None, :],
        d_post_logit,
        mask=mask_leading[:, None],
    )
    tl.store(
        DBASE_PTR + offs[:, None, None] * KK_TOTAL + 2 * K + k_idx[None, :, None] * K + k_idx[None, None, :],
        d_comb_logit,
        mask=mask_leading[:, None, None],
    )

    # d_scale needs ``logits * d_*_logit`` per-element, summed over
    # each slice's K (or K*K) inner extent and then over rows.  We
    # write the per-row d_*_logit values into ``DBASE_PTR`` above
    # (which is exactly ``d_*_logit`` with no scale factor); the
    # host-side wrapper then computes
    #   d_scale[0] = (logits[:, :K]   * d_base_partials[:, :K]  ).sum()
    #   d_scale[1] = (logits[:, K:2K] * d_base_partials[:, K:2K]).sum()
    #   d_scale[2] = (logits[:, 2K:]  * d_base_partials[:, 2K:] ).sum()
    # with torch.sum (avoiding cross-block atomic_add).  ``DSCALE_PTR``
    # is therefore left as the zero-init buffer the wrapper allocated;
    # we keep it in the kernel signature for forward-compatibility.


# ---------------------------------------------------------------------------
# Block-leading heuristic
# ---------------------------------------------------------------------------


def _pick_block_leading(n: int, k: int) -> int:
    """Pick ``BLOCK_LEADING`` based on K and the work-axis size.

    At K=4 the in-register state per row is ~``24 + K + K*K = 44`` fp32
    elements; ``BLOCK_LEADING=64`` fits comfortably (~11 KiB).  At K=16
    the per-row state grows ~6x; drop to 8.
    """

    if k <= 4:
        cap = 64
    elif k <= 8:
        cap = 32
    else:
        cap = 8

    if n < cap:
        return max(1, triton.next_power_of_2(n))
    return cap


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------


class HCComputeTailFn(torch.autograd.Function):
    """Autograd-aware wrapper around the FWD/BWD Triton kernels.

    Saves ``(logits, scale, pre_sig, post_sig, comb_sm)`` for backward.
    Returns ``(pre, post, comb_pre_sinkhorn)`` -- the caller then runs
    ``sinkhorn_normalize`` on ``comb_pre_sinkhorn``.

    Shape: ``logits [..., (2+K)*K]`` fp32, ``scale [3]`` fp32,
    ``base [(2+K)*K]`` fp32, ``K`` a power of 2 in ``{1, 2, 4, 8, 16}``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        K: int,
        eps: float,
        out_dtype: torch.dtype,
    ):
        if K not in _SUPPORTED_K:
            raise ValueError(f"hc_glue Triton path: unsupported K={K}; expected one of {_SUPPORTED_K}")
        if logits.shape[-1] != (2 + K) * K:
            raise ValueError(
                f"hc_glue: logits last-dim must be (2+K)*K = {(2 + K) * K}, " f"got {logits.shape[-1]}"
            )
        if scale.numel() != 3:
            raise ValueError(f"hc_glue: scale must have 3 elements, got {scale.numel()}")
        if base.numel() != (2 + K) * K:
            raise ValueError(
                f"hc_glue: base must have (2+K)*K = {(2 + K) * K} elements, " f"got {base.numel()}"
            )

        logits_c = logits.contiguous().to(torch.float32)
        scale_c = scale.contiguous().to(torch.float32)
        base_c = base.contiguous().to(torch.float32)

        leading_shape = logits_c.shape[:-1]
        N = 1
        for s in leading_shape:
            N *= s

        device = logits_c.device
        pre = torch.empty((*leading_shape, K), dtype=out_dtype, device=device)
        post = torch.empty((*leading_shape, K), dtype=out_dtype, device=device)
        comb = torch.empty((*leading_shape, K, K), dtype=out_dtype, device=device)

        # Saved-for-backward fp32 states.
        pre_sig = torch.empty((N, K), dtype=torch.float32, device=device)
        post_sig = torch.empty((N, K), dtype=torch.float32, device=device)
        comb_sm = torch.empty((N, K, K), dtype=torch.float32, device=device)

        block_leading = _pick_block_leading(N, K)
        grid = (triton.cdiv(N, block_leading),)
        _hc_compute_tail_fwd_kernel[grid](
            logits_c,
            scale_c,
            base_c,
            pre,
            post,
            comb,
            pre_sig,
            post_sig,
            comb_sm,
            N,
            EPS=float(eps),
            K=K,
            BLOCK_LEADING=block_leading,
            OUT_DTYPE=_triton_dtype(out_dtype),
        )

        ctx.save_for_backward(logits_c, scale_c, pre_sig, post_sig, comb_sm)
        ctx.K = K
        ctx.eps = float(eps)
        ctx.leading_shape = tuple(leading_shape)
        ctx.out_dtype = out_dtype
        return pre, post, comb

    @staticmethod
    def backward(ctx, d_pre, d_post, d_comb):  # type: ignore[override]
        logits_c, scale_c, pre_sig, post_sig, comb_sm = ctx.saved_tensors
        K: int = ctx.K
        leading_shape = ctx.leading_shape

        N = 1
        for s in leading_shape:
            N *= s

        d_pre = d_pre.contiguous()
        d_post = d_post.contiguous()
        d_comb = d_comb.contiguous()

        device = logits_c.device
        d_logits = torch.empty_like(logits_c)
        d_base_partials = torch.empty((N, (2 + K) * K), dtype=torch.float32, device=device)
        # d_scale_partials is no longer written by the kernel; we leave
        # the buffer hint in the kernel signature for forward-compat.
        d_scale_partials = torch.zeros((N, 3), dtype=torch.float32, device=device)

        block_leading = _pick_block_leading(N, K)
        grid = (triton.cdiv(N, block_leading),)
        _hc_compute_tail_bwd_kernel[grid](
            d_pre,
            d_post,
            d_comb,
            pre_sig,
            post_sig,
            comb_sm,
            scale_c,
            d_logits,
            d_scale_partials,
            d_base_partials,
            N,
            K=K,
            BLOCK_LEADING=block_leading,
        )

        # Host-side reductions:
        #   d_base[i] = sum_n d_base_partials[n, i]                       (3*K + K*K entries)
        #   d_scale[0] = sum_{n, k}   logits_slice_pre  * d_base_partials_pre
        #   d_scale[1] = sum_{n, k}   logits_slice_post * d_base_partials_post
        #   d_scale[2] = sum_{n, k1, k2} logits_slice_comb * d_base_partials_comb
        #
        # Equivalent to: d_scale[i] = (logits_slice_i * d_base_partials_i).sum()
        # since d_base = d_*_logit (no scale) and d_scale[i] = sum logits * d_*_logit.
        logits_flat = logits_c.reshape(N, -1)
        d_base = d_base_partials.sum(dim=0)
        d_scale_0 = (logits_flat[:, :K] * d_base_partials[:, :K]).sum()
        d_scale_1 = (logits_flat[:, K : 2 * K] * d_base_partials[:, K : 2 * K]).sum()
        d_scale_2 = (logits_flat[:, 2 * K :] * d_base_partials[:, 2 * K :]).sum()
        d_scale = torch.stack([d_scale_0, d_scale_1, d_scale_2])

        # Reshape d_logits back to the leading shape that the FWD input
        # had so the autograd machinery passes it back to the caller's
        # F.linear backward correctly.
        d_logits = d_logits.view(*leading_shape, (2 + K) * K)

        # K, eps, out_dtype are non-differentiable.
        return d_logits, d_scale, d_base, None, None, None


# ---------------------------------------------------------------------------
# Public Python entry points
# ---------------------------------------------------------------------------


def is_triton_path_enabled() -> bool:
    """Return True iff the ``PRIMUS_HC_TRITON`` env knob is not ``"0"``.

    Default-on; A/B toggle via ``PRIMUS_HC_TRITON=0``.
    """

    return os.environ.get("PRIMUS_HC_TRITON", "1") != "0"


def is_triton_kernel_supported(logits: torch.Tensor, K: int) -> bool:
    """Return True iff the input shape / device is supported.

    Used by the dispatcher in
    :meth:`primus.backends.megatron.core.transformer.hyper_connection.HyperMixer.compute_weights`
    to safely fall back to the eager body for unsupported configurations.
    """

    if not logits.is_cuda:
        return False
    if K not in _SUPPORTED_K:
        return False
    if logits.shape[-1] != (2 + K) * K:
        return False
    return True


def hc_glue_compute_tail_triton(
    logits: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    K: int,
    eps: float,
    out_dtype: torch.dtype,
):
    """Run the Triton-fused HC compute_weights tail.

    Returns ``(pre, post, comb_pre_sinkhorn)`` -- the caller runs
    ``sinkhorn_normalize`` on ``comb_pre_sinkhorn``.
    """

    return HCComputeTailFn.apply(logits, scale, base, K, eps, out_dtype)


__all__ = [
    "HCComputeTailFn",
    "hc_glue_compute_tail_triton",
    "is_triton_path_enabled",
    "is_triton_kernel_supported",
]

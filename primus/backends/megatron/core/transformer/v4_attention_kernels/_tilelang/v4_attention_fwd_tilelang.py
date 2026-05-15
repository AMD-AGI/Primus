###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-8 P50 — V4 dense FWD tilelang kernel (compress_ratio == 0).

Drop-in replacement for ``_launch_v4_attention_fwd`` (plan-4 P25
Triton FWD) at the dense / SWA / sink subset of the V4 attention
contract.  HCA (``hca_local_seqlen > 0``) and explicit
``additive_mask`` paths are deferred to P52; the wrapper falls
back to the Triton kernel when those features are requested.

Scope at P50:

* ``Q [B, H, Sq, D]``, ``K [B, K_H, Sk, D]``, ``V [B, K_H, Sk, D]``
  with ``K_H ∈ {1 (MQA), H (MHA)}``.
* Optional per-head sink ``[H]`` — joined as a virtual key column
  at the end of the softmax (matches the plan-4 P25 / plan-2 P14
  sink contract).
* Optional sliding-window-causal mask (``swa_window > 0``) applied
  in the kernel via `start = max(0, (m + 1 - window_size) //
  block_N)`.
* MQA broadcast: when ``K_H == 1``, every query head reads the
  shared K/V head; computed via per-program ``h // (HQ // HK)``.

Out of scope at P50 (Triton fallback handles these):

* ``additive_mask`` not None  — needs the kernel to accept a
  [Sq, Sk] additive bias.  Deferred to P52 (alongside the HCA
  split-mask path).
* ``hca_local_seqlen > 0`` — HCA split-mask path.  Deferred to
  P52.

dtype contract (must match :func:`reference.eager_v4_attention`):

* Q / K / V matmuls run in input dtype on tensor cores; the
  matmul accumulator inside is fp32.
* Online softmax accumulator (``m_i``, ``l_i``, ``acc_o``) is fp32.
* Output is in ``v.dtype``; saved ``LSE`` is fp32 (BWD walks back
  from it).

The kernel borrows from
``tilelang/examples/attention_sink/example_mha_sink_fwd_bhsd.py``
(sink fusion at end of softmax via ``exp2(sinks * log2e - max *
scale)``) and ``tilelang/examples/amd/example_amd_flash_attn_fwd.py``
(MI355X-tuned block sizes + WMMA ``k_pack`` + ``T.use_swizzle``).

Implementation note: this module deliberately does NOT use
``from __future__ import annotations`` because tilelang's
``@T.prim_func`` decorator evaluates annotations eagerly via
``typing.get_type_hints``, which fails when annotations are lazy
strings + the referenced names live in an enclosing function's
local scope (e.g. ``q_shape`` defined inside the JIT factory).
"""

from typing import Optional, Tuple

import torch

# Lazy import — tilelang is heavy.  The wrapper checks
# ``_tilelang.should_dispatch(...)`` upstream so we only land here
# when the env knob is set + tilelang is importable.
import tilelang
import tilelang.language as T
from primus.backends.megatron.core.transformer.v4_attention_kernels import _tilelang
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_fwd import (
    _launch_v4_attention_fwd,
)

# Compiled-kernel cache keyed by (B, HQ, HK, Sq, Sk, D, has_sink,
# swa_window, dtype-str).  Tilelang's own JIT also caches per-shape;
# this dict is a Python-side memoisation that avoids the
# ``@tilelang.jit`` recompile cost on every wrapper call.
_KERNEL_CACHE: dict[tuple, object] = {}


# ---------------------------------------------------------------------------
# Tilelang kernel definition
# ---------------------------------------------------------------------------


@tilelang.jit(
    out_idx=[3, 4],
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
)
def _make_v4_attention_fwd_kernel(
    batch: int,
    heads_q: int,
    heads_k: int,
    seq_q: int,
    seq_k: int,
    dim: int,
    has_sink: bool,
    swa_window: int,
    dtype: "T.dtype" = T.bfloat16,
    block_M: int = 64,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
):
    """Plan-8 P50 tilelang FWD kernel JIT factory.

    `has_sink`, `swa_window`, MQA-vs-MHA branching are baked into the
    kernel at JIT time so the inner loop has no host-side branching
    cost.  All shape args + flags are kwargs to this JIT'd function;
    tilelang's `@tilelang.jit` decorator caches compiled binaries per
    (kwargs) tuple via its own cache.
    """

    groups = heads_q // heads_k  # 1 for MHA, heads_q for MQA
    accum_dtype = T.float32
    sm_scale = 1.0 / (dim**0.5)
    scale = sm_scale * 1.44269504  # log2(e) — enables exp2/FFMA

    q_shape = [batch, heads_q, seq_q, dim]
    kv_shape = [batch, heads_k, seq_k, dim]
    sink_shape = [heads_q]
    out_shape = [batch, heads_q, seq_q, dim]
    lse_shape = [batch, heads_q, seq_q]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(out_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Sinks: T.Tensor(sink_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads_q, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            sinks_reg = T.alloc_fragment([block_M], dtype)

            # MQA broadcast: kv-head index for this query head.
            k_head = by // groups

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            if has_sink:
                for i in T.Parallel(block_M):
                    sinks_reg[i] = Sinks[by]

            # K-tile loop bounds.  Causal: end at the right edge of
            # the m-tile.  SWA: start at `max(0, (m+1 - W) // BN)`.
            end = T.min(T.ceildiv(seq_k, block_N), T.ceildiv((bx + 1) * block_M, block_N))
            if swa_window > 0:
                start = T.max(0, (bx * block_M - swa_window) // block_N)
            else:
                start = 0

            for k in T.Pipelined(start, end, num_stages=num_stages):
                T.copy(K[bz, k_head, k * block_N : (k + 1) * block_N, :], K_shared)
                for i, j in T.Parallel(block_M, block_N):
                    q_idx = bx * block_M + i
                    k_idx = k * block_N + j
                    if swa_window > 0:
                        acc_s[i, j] = T.if_then_else(
                            q_idx >= k_idx and q_idx < k_idx + swa_window,
                            0,
                            -T.infinity(acc_s.dtype),
                        )
                    else:
                        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                # Check-inf for SWA: a tile that is fully outside the
                # window has scores_max = -inf; reset to 0 to avoid
                # NaN from `exp2(-inf - -inf)`.
                for i in T.Parallel(block_M):
                    if swa_window > 0:
                        scores_max[i] = T.if_then_else(
                            scores_max[i] == -T.infinity(accum_dtype),
                            0,
                            scores_max[i],
                        )
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, k_head, k * block_N : (k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Sink as virtual key column at end-of-loop.
            if has_sink:
                for i in T.Parallel(block_M):
                    logsum[i] += T.exp2(sinks_reg[i] * 1.44269504 - scores_max[i] * scale)

            # Final normalise + cast + LSE emit.
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

            # LSE = ln(l_i) + m_i*sm_scale (note: scores were
            # accumulated with `*scale = sm_scale*log2e`, so the
            # base-e LSE is `ln(l_i) + m_i*sm_scale`).
            for i in T.Parallel(block_M):
                logsum[i] = T.log(logsum[i]) + scores_max[i] * sm_scale
            T.copy(logsum, Lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return main


# ---------------------------------------------------------------------------
# Wrapper API
# ---------------------------------------------------------------------------


def _kernel_supports(
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    additive_mask: Optional[torch.Tensor],
    hca_local_seqlen: int,
) -> bool:
    """Return True iff the P50 tilelang kernel covers this call site.

    P50 ships only the dense / SWA / sink subset.  Additive-mask +
    HCA split-mask are P52 territory; we fall back to Triton there.
    """
    if additive_mask is not None:
        return False
    if hca_local_seqlen > 0:
        return False
    return True


def _torch_dtype_to_tilelang(dtype: torch.dtype) -> "T.dtype":
    if dtype == torch.bfloat16:
        return T.bfloat16
    if dtype == torch.float16:
        return T.float16
    if dtype == torch.float32:
        return T.float32
    raise ValueError(f"unsupported dtype {dtype}; expected bf16 / fp16 / fp32")


def _get_or_compile_kernel(
    *,
    batch: int,
    heads_q: int,
    heads_k: int,
    seq_q: int,
    seq_k: int,
    dim: int,
    has_sink: bool,
    swa_window: int,
    dtype: torch.dtype,
):
    """Compile (or fetch from tilelang's per-args cache) the FWD kernel
    for the given shape envelope.

    Tilelang's `@tilelang.jit` decorator caches compiled binaries per
    (kwargs) tuple via its own cache, so this wrapper is itself fast
    on cache hits.
    """
    key = (
        batch,
        heads_q,
        heads_k,
        seq_q,
        seq_k,
        dim,
        has_sink,
        int(swa_window > 0) * swa_window,
        str(dtype),
    )
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    tl_dtype = _torch_dtype_to_tilelang(dtype)
    # SMEM budget at head_dim=512 on MI355X is tight (160 KiB).
    # Each `T.alloc_shared([B, dim], bf16)` allocates `B * dim * 2`
    # bytes; the kernel uses Q + K + V + O shared = 4 * B_tile * D
    # * 2.  At D=512 even 32x32 hits some tilelang-internal hidden
    # allocations that overshoot.  Conservative choice: shrink to
    # 16x32 with threads=64 at D >= 256.
    if dim >= 256:
        block_M, block_N, threads = 32, 32, 64
    else:
        block_M, block_N, threads = 64, 64, 128
    kernel = _make_v4_attention_fwd_kernel(
        batch,
        heads_q,
        heads_k,
        seq_q,
        seq_k,
        dim,
        has_sink,
        swa_window,
        dtype=tl_dtype,
        block_M=block_M,
        block_N=block_N,
        threads=threads,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def v4_attention_fwd_tilelang_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sink: Optional[torch.Tensor] = None,
    additive_mask: Optional[torch.Tensor] = None,
    swa_window: int = 0,
    attn_dropout: float = 0.0,
    training: bool = False,
    scale: Optional[float] = None,
    hca_local_seqlen: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Same dispatch logic as :func:`v4_attention_fwd_tilelang` but
    also returns the saved ``LSE`` tensor when the tilelang path
    runs.

    When the wrapper falls back to Triton, ``LSE`` is computed by
    the Triton launcher and returned alongside ``out``; the
    autograd Function in P51 saves it for backward.
    """
    if attn_dropout > 0.0 and training:
        return _launch_v4_attention_fwd(
            q,
            k,
            v,
            sink=sink,
            swa_window=swa_window,
            additive_mask=additive_mask,
            scale=scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5),
            hca_local_seqlen=hca_local_seqlen,
        )
    if not _kernel_supports(
        sink=sink,
        swa_window=swa_window,
        additive_mask=additive_mask,
        hca_local_seqlen=hca_local_seqlen,
    ):
        return _launch_v4_attention_fwd(
            q,
            k,
            v,
            sink=sink,
            swa_window=swa_window,
            additive_mask=additive_mask,
            scale=scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5),
            hca_local_seqlen=hca_local_seqlen,
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "v4_attention_fwd_tilelang expects q / k / v of rank 4 "
            f"(got q.dim={q.dim()}, k.dim={k.dim()}, v.dim={v.dim()})"
        )
    B, HQ, Sq, D = q.shape
    _, HK, Sk, _ = k.shape
    assert HK in (1, HQ)
    has_sink = sink is not None
    kernel = _get_or_compile_kernel(
        batch=B,
        heads_q=HQ,
        heads_k=HK,
        seq_q=Sq,
        seq_k=Sk,
        dim=D,
        has_sink=has_sink,
        swa_window=int(swa_window),
        dtype=q.dtype,
    )
    if not has_sink:
        sink_arg = torch.zeros(HQ, device=q.device, dtype=q.dtype)
    else:
        sink_arg = sink
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    sink_c = sink_arg.contiguous()
    out, lse = kernel(q_c, k_c, v_c, sink_c)
    return out, lse


def v4_attention_fwd_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sink: Optional[torch.Tensor] = None,
    additive_mask: Optional[torch.Tensor] = None,
    swa_window: int = 0,
    attn_dropout: float = 0.0,
    training: bool = False,
    scale: Optional[float] = None,
    hca_local_seqlen: int = 0,
) -> torch.Tensor:
    """Plan-8 P50 tilelang dense / SWA / sink FWD wrapper.

    Drop-in replacement for the relevant subset of ``v4_attention``
    (the Triton autograd path).  When the call site requires features
    P50 doesn't cover (`additive_mask`, `hca_local_seqlen > 0`), the
    wrapper falls back to the Triton kernel via
    ``_launch_v4_attention_fwd`` so production smokes never see an
    error in the regression window.

    Returns ``out [B, H, Sq, D]`` in ``v.dtype``.

    ``attn_dropout`` and ``training`` are accepted for parity with
    the Triton wrapper but not used — dropout on attention is off
    in production V4 (plan-4 P25 design note); P50 inherits that
    contract.

    Note: this entry point is the **plain** (no-autograd) wrapper.
    For the autograd-aware path used by V4AttentionFn, the call
    site uses :func:`v4_attention_fwd_tilelang_with_lse` directly
    so it can save ``LSE`` for backward.
    """
    out, _ = v4_attention_fwd_tilelang_with_lse(
        q,
        k,
        v,
        sink=sink,
        additive_mask=additive_mask,
        swa_window=swa_window,
        attn_dropout=attn_dropout,
        training=training,
        scale=scale,
        hca_local_seqlen=hca_local_seqlen,
    )
    return out


# ---------------------------------------------------------------------------
# Module-level registration — flips
# `_tilelang.is_tilelang_kernel_available("v4_attention_fwd")` to True
# AND overrides the parent module's stub with the real wrapper, so
# callers can do `_tilelang.v4_attention_fwd_tilelang(...)` and get
# the real implementation after lazy-load.
# ---------------------------------------------------------------------------


_tilelang.register_available_kernel("v4_attention_fwd")
# Note: the parent module's `v4_attention_fwd_tilelang` attribute
# is restored by `_tilelang._lazy_load(...)` after the import_module
# call (Python's import machinery sets it to the submodule otherwise).


__all__ = ["v4_attention_fwd_tilelang"]

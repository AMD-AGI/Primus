###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-8 P51 — V4 dense BWD tilelang kernels (compress_ratio == 0).

Three-kernel pipeline matching the tilelang FlashAttention v2 BWD
reference (`tilelang/examples/amd/example_amd_flash_attn_bwd.py`):

* :func:`_make_preprocess_kernel` — ``Delta[b, h, m] =
  sum_d O[b, h, m, d] * dO[b, h, m, d]``.  One tiny kernel.
* :func:`_make_bwd_kernel` — main BWD pass.  One program per
  ``(h, n_tile, b)``; loops over m_tile; re-materialises ``P``
  from saved ``LSE``; accumulates ``dQ`` / ``dK`` / ``dV``
  via ``tl.atomic_add``.  Sink BWD reuses the dq stride.
* The sink gradient is computed in the Python wrapper as a small
  reduction over the saved ``LSE`` + ``Delta`` (cheap; one ATen
  launch).

Scope at P51 (matches P50):

* Q/K/V in BHSD layout with optional MQA broadcast (`K_H ∈ {1, H}`).
* Optional per-head sink.
* Optional sliding-window-causal mask.
* No `additive_mask`, no `hca_local_seqlen > 0` — those route
  through the plan-4 P25 / plan-5 P32 final Triton BWD kernel.

dtype contract (matches :func:`reference.eager_v4_attention`
backward): matmuls run in input dtype on tensor cores; the
accumulator + softmax recompute live in fp32.  Output gradients
are in `q.dtype` / `k.dtype` / `v.dtype` (typically all bf16);
internal accumulators are fp32 buffers cast at the end.

Implementation note: this module does NOT use ``from __future__
import annotations`` because tilelang's ``@T.prim_func``
decorator evaluates annotations eagerly via
``typing.get_type_hints``.
"""

from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels import _tilelang
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_attention_bwd import (
    _launch_v4_attention_bwd,
)

_PREPROCESS_CACHE: dict[tuple, object] = {}
_BWD_CACHE: dict[tuple, object] = {}


# ---------------------------------------------------------------------------
# Preprocess: Delta[b, h, m] = (O * dO).sum(-1)
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[2])
def _make_preprocess_kernel(
    batch: int,
    heads: int,
    seq_q: int,
    dim: int,
    dtype: "T.dtype" = T.bfloat16,
    block: int = 32,
):
    """Preprocess kernel: emit ``Delta [B, H, Sq]`` fp32.

    Tiles `(b, h, m_tile)`; each program reduces over `dim` in
    chunks of `block`.
    """
    accum_dtype = T.float32
    bhsd_shape = [batch, heads, seq_q, dim]
    delta_shape = [batch, heads, seq_q]

    @T.prim_func
    def main(
        O: T.Tensor(bhsd_shape, dtype),
        dO: T.Tensor(bhsd_shape, dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
    ):
        with T.Kernel(batch, heads, T.ceildiv(seq_q, block)) as (bz, bx, by):
            o = T.alloc_fragment([block, block], dtype)
            do = T.alloc_fragment([block, block], dtype)
            acc = T.alloc_fragment([block, block], accum_dtype)
            delta = T.alloc_fragment([block], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, block)):
                T.copy(
                    O[bz, bx, by * block : (by + 1) * block, k * block : (k + 1) * block],
                    o,
                )
                T.copy(
                    dO[bz, bx, by * block : (by + 1) * block, k * block : (k + 1) * block],
                    do,
                )
                for i, j in T.Parallel(block, block):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * block : (by + 1) * block])

    return main


# ---------------------------------------------------------------------------
# Main BWD: dQ, dK, dV via re-materialised P from saved LSE
# ---------------------------------------------------------------------------


# Note: dK / dV are emitted as fp32 (not the input dtype) so the
# MQA path can use `T.atomic_add` — tilelang's HIP runtime only
# supports `AtomicAddx2(float*, ...)`.  The wrapper casts to
# k.dtype / v.dtype before returning.
@tilelang.jit(out_idx=[6, 7, 8])
def _make_bwd_kernel(
    batch: int,
    heads_q: int,
    heads_k: int,
    seq_q: int,
    seq_k: int,
    dim: int,
    has_sink: bool,
    swa_window: int,
    dtype: "T.dtype" = T.bfloat16,
    block_M: int = 32,
    block_N: int = 32,
    threads: int = 64,
):
    """Plan-8 P51 main BWD: dQ / dK / dV via re-materialised softmax.

    Grid: one program per `(b, h, n_tile)` (n_tile along K/V seq dim).
    Each program loops over `m_tile` along Q.  Pattern mirrors the
    tilelang AMD example BWD with three adaptations:

    * BHSD layout instead of BSHD.
    * MQA via `k_head = bx // groups`.
    * SWA-aware m_tile loop bounds (skip tiles outside the window).

    Output `dQ` is in fp32 (caller casts back to q.dtype after a
    `dsink` host-side reduction); `dK`, `dV` are emitted in
    `k.dtype` / `v.dtype` via the kernel epilogue.
    """
    groups = heads_q // heads_k
    accum_dtype = T.float32
    sm_scale = 1.0 / (dim**0.5)
    q_shape = [batch, heads_q, seq_q, dim]
    kv_shape = [batch, heads_k, seq_k, dim]
    dq_shape = [batch, heads_q, seq_q, dim]
    dkv_shape = [batch, heads_k, seq_k, dim]
    lse_shape = [batch, heads_q, seq_q]
    delta_shape = [batch, heads_q, seq_q]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(q_shape, dtype),
        LSE: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(dq_shape, accum_dtype),
        dK: T.Tensor(dkv_shape, accum_dtype),
        dV: T.Tensor(dkv_shape, accum_dtype),
    ):
        with T.Kernel(heads_q, T.ceildiv(seq_k, block_M), batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            K_shared = T.alloc_shared([block_M, dim], dtype)
            V_shared = T.alloc_shared([block_M, dim], dtype)
            q_shared = T.alloc_shared([block_N, dim], dtype)
            do_shared = T.alloc_shared([block_N, dim], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta_shared = T.alloc_shared([block_N], accum_dtype)
            ds_shared = T.alloc_shared([block_M, block_N], dtype)
            p_cast = T.alloc_fragment([block_M, block_N], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_acc = T.alloc_fragment([block_M, block_N], accum_dtype)
            dP = T.alloc_fragment([block_M, block_N], accum_dtype)
            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq_tile = T.alloc_fragment([block_N, dim], accum_dtype)

            k_head = bx // groups

            T.copy(K[bz, k_head, by * block_M : (by + 1) * block_M, :], K_shared)
            T.copy(V[bz, k_head, by * block_M : (by + 1) * block_M, :], V_shared)
            T.clear(dv)
            T.clear(dk)

            # Causal: only m_tiles that touch this n_tile contribute.
            loop_st = by * block_M // block_N
            loop_ed = T.ceildiv(seq_q, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                T.copy(Q[bz, bx, k * block_N : (k + 1) * block_N, :], q_shared)
                T.clear(qkT)
                T.gemm(
                    K_shared,
                    q_shared,
                    qkT,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(LSE[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)

                for i, j in T.Parallel(block_M, block_N):
                    P_acc[i, j] = T.exp(qkT[i, j] * sm_scale - lse_shared[j])

                # Causal mask + optional SWA: keep entries where the
                # query token (`k * block_N + j`) attends to the key
                # token (`by * block_M + i`).  SWA: also require
                # `q_idx - k_idx < swa_window`.
                for i, j in T.Parallel(block_M, block_N):
                    q_idx = k * block_N + j
                    k_idx = by * block_M + i
                    valid = q_idx >= k_idx
                    if swa_window > 0:
                        valid = valid and (q_idx - k_idx < swa_window)
                    P_acc[i, j] = T.if_then_else(valid, P_acc[i, j], 0.0)

                T.copy(dO[bz, bx, k * block_N : (k + 1) * block_N, :], do_shared)
                T.clear(dP)
                T.gemm(
                    V_shared,
                    do_shared,
                    dP,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(P_acc, p_cast)
                T.gemm(p_cast, do_shared, dv, policy=T.GemmWarpPolicy.FullRow)

                T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta_shared)
                for i, j in T.Parallel(block_M, block_N):
                    p_cast[i, j] = P_acc[i, j] * (dP[i, j] - delta_shared[j]) * sm_scale
                T.gemm(p_cast, q_shared, dk, policy=T.GemmWarpPolicy.FullRow)
                T.copy(p_cast, ds_shared)

                T.clear(dq_tile)
                T.gemm(ds_shared, K_shared, dq_tile, transpose_A=True)
                for i, j in T.Parallel(block_N, dim):
                    T.atomic_add(dQ[bz, bx, k * block_N + i, j], dq_tile[i, j])

            # MQA: every query head writes to the same K/V head;
            # use atomic_add to merge across query-head programs.
            # `dK` / `dV` are fp32 buffers (tilelang's HIP atomic-add
            # only supports float*); the wrapper casts to k.dtype /
            # v.dtype after the kernel returns.
            if groups > 1:
                for i, j in T.Parallel(block_M, dim):
                    T.atomic_add(dV[bz, k_head, by * block_M + i, j], dv[i, j])
                    T.atomic_add(dK[bz, k_head, by * block_M + i, j], dk[i, j])
            else:
                for i, j in T.Parallel(block_M, dim):
                    dV[bz, k_head, by * block_M + i, j] = dv[i, j]
                    dK[bz, k_head, by * block_M + i, j] = dk[i, j]

    return main


# ---------------------------------------------------------------------------
# Wrapper API
# ---------------------------------------------------------------------------


def _kernel_supports(
    *,
    additive_mask: Optional[torch.Tensor],
    hca_local_seqlen: int,
) -> bool:
    if additive_mask is not None:
        return False
    if hca_local_seqlen > 0:
        return False
    return True


def _torch_dtype_to_tilelang(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return T.bfloat16
    if dtype == torch.float16:
        return T.float16
    if dtype == torch.float32:
        return T.float32
    raise ValueError(f"unsupported dtype {dtype}")


def _get_preprocess(batch, heads, seq_q, dim, dtype):
    key = (batch, heads, seq_q, dim, str(dtype))
    if key in _PREPROCESS_CACHE:
        return _PREPROCESS_CACHE[key]
    k = _make_preprocess_kernel(batch, heads, seq_q, dim, dtype=_torch_dtype_to_tilelang(dtype))
    _PREPROCESS_CACHE[key] = k
    return k


def _get_bwd(batch, heads_q, heads_k, seq_q, seq_k, dim, has_sink, swa_window, dtype):
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
    if key in _BWD_CACHE:
        return _BWD_CACHE[key]
    # Same SMEM-budget heuristic as P50.
    if dim >= 256:
        block_M, block_N, threads = 32, 32, 64
    else:
        block_M, block_N, threads = 64, 64, 128
    k = _make_bwd_kernel(
        batch,
        heads_q,
        heads_k,
        seq_q,
        seq_k,
        dim,
        has_sink,
        swa_window,
        dtype=_torch_dtype_to_tilelang(dtype),
        block_M=block_M,
        block_N=block_N,
        threads=threads,
    )
    _BWD_CACHE[key] = k
    return k


def v4_attention_bwd_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    *,
    sink: Optional[torch.Tensor] = None,
    additive_mask: Optional[torch.Tensor] = None,
    swa_window: int = 0,
    scale: Optional[float] = None,
    hca_local_seqlen: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Plan-8 P51 tilelang dense / SWA / sink BWD wrapper.

    Drop-in replacement for ``_launch_v4_attention_bwd`` (plan-5
    P32 final Triton split BWD) on the dense subset of the V4
    attention contract.  Falls back to Triton for `additive_mask`
    / `hca_local_seqlen > 0` paths (P53 territory).

    Returns ``(dq, dk, dv, dsink)`` matching the Triton launcher
    signature.
    """
    if not _kernel_supports(additive_mask=additive_mask, hca_local_seqlen=hca_local_seqlen):
        return _launch_v4_attention_bwd(
            q,
            k,
            v,
            out,
            lse,
            do,
            sink=sink,
            additive_mask=additive_mask,
            swa_window=swa_window,
            scale=scale if scale is not None else 1.0 / (q.shape[-1] ** 0.5),
            hca_local_seqlen=hca_local_seqlen,
        )

    B, HQ, Sq, D = q.shape
    _, HK, Sk, _ = k.shape
    assert HK in (1, HQ)
    dtype = q.dtype

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    do_c = do.contiguous()
    out_c = out.contiguous()
    lse_c = lse.contiguous()

    # Preprocess: Delta = (O * dO).sum(-1)
    preprocess = _get_preprocess(B, HQ, Sq, D, dtype)
    delta = preprocess(out_c, do_c)

    has_sink = sink is not None
    bwd = _get_bwd(B, HQ, HK, Sq, Sk, D, has_sink, int(swa_window), dtype)
    dq_fp32, dk_fp32, dv_fp32 = bwd(q_c, k_c, v_c, do_c, lse_c, delta)
    # All three grads are emitted in fp32 (MQA atomic-add target dtype);
    # cast back to input dtypes.
    dq = dq_fp32.to(dtype)
    dk = dk_fp32.to(k.dtype)
    dv = dv_fp32.to(v.dtype)

    # Sink BWD: computed host-side (cheap).  When sink is present:
    #   dsink[h] = sum_b sum_m exp(sink[h] - lse[b, h, m]) * delta[b, h, m]
    if has_sink:
        # Re-cast delta to fp32 for the reduce (it's already fp32 from
        # preprocess).
        dsink = (torch.exp(sink.float().unsqueeze(0).unsqueeze(-1) - lse_c.float()) * delta).sum(dim=(0, 2))
        dsink = dsink.to(sink.dtype)
    else:
        dsink = None

    return dq, dk, dv, dsink


# ---------------------------------------------------------------------------
# Module-level registration
# ---------------------------------------------------------------------------

_tilelang.register_available_kernel("v4_attention_bwd")
# `_tilelang._lazy_load(...)` restores the parent-module attribute
# (`_tilelang.v4_attention_bwd_tilelang`) after Python's import
# machinery would otherwise shadow the stub with this submodule.


__all__ = ["v4_attention_bwd_tilelang"]

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plain-Triton DeepSeek-V4 sparse-MLA backward (the "triton v2" backend).

Companion to :func:`sparse_mla_fwd_v4_triton`; API mirrors
:func:`sparse_mla_bwd_v4_gluon` -> ``(dq, dkv, d_sink)``.

Uses the **non-atomic chunked-gather** scheme (the same one the gluon backward
shares for its dKV gather): per rank-chunk it runs a pure-Triton dQ kernel and a
dKV-intermediate kernel (both ``tl.dot`` / MFMA, plain stores — no atomics),
then reduces the intermediate into ``dkv`` via a CSR inverted-topk gather. This
is the fully-Triton analogue of the gluon dQ/dKV-interm kernels, so its dKV is
not bottlenecked by global atomics. ``d_sink`` is the closed-form torch
reduction ``-sum_t exp(sink - lse) * delta``.
"""

import contextlib
import os

import torch
import triton

# CSR-inverted-topk builder + Delta preprocess are backend-neutral torch/host
# helpers; reuse them. The compute kernels are owned locally (dsa_bwd_kernels)
# so this backend can be tuned independently of the gluon path.
from .._gluon_dsa._dsa_bwd_gather import _build_inverted_topk_slice
from ._amd_knobs import amd_pingpong_disabled, is_gfx1250
from ._turbo_gfx1250 import sparse_mla_bwd_turbo_gfx1250, turbo_gfx1250_enabled
from .dsa_bwd_kernels import (
    _bwd_chunk_dkv_store_ds,
    _bwd_chunk_dq_dkv_fused,
    _bwd_chunk_dq_store_ds,
    _bwd_compute_dkv_intermediate,
    _bwd_dkv_gather_acc,
    _bwd_dkv_gather_split,
    _bwd_dq_from_ds,
)

# ---- gfx1250 (MI455X) fused backward config ---------------------------------
# Measured on MI455X (V4-Flash H=64, S=4096; ~/dsv41/work/csa/REPORT.md). Triton on
# gfx1250 gives a num_warps=4 (wave32) program the full 1024-VGPR budget at one wave
# per SIMD; num_warps 8/16 halve/quarter that and spill heavily (2-4x slower), so the
# gfx950 head/tile blocking does not carry over. The fused dQ + dKV-intermediate kernel
# keeps all 64 heads of a token in one program (heads = MMA M dim, each gathered KV
# tile loaded once) and drops the dS / P HBM round trip and a whole kernel.
_GFX1250_FUSED_TILE_K = 16
_GFX1250_FUSED_NUM_WARPS = 4
_GFX1250_FUSED_BLOCK_H = int(os.environ.get("PRIMUS_DSA_BWD_FUSED_BLOCK_H", "64"))
# Split dQ out of the fused kernel (default): the [64, 512] fp32 dQ accumulator is what
# pushes the fully fused kernel to 1024 VGPRs + ~150 spills on gfx1250 (ablation: dropping
# the dQ dot 2.52 -> 1.44 ms). _bwd_chunk_dkv_store_ds keeps the fused S / dP / dS / dKV
# path and writes dS (bf16, once); _bwd_dq_from_ds then forms dQ = dS @ K_gathered with a
# small D_V-split accumulator. cr4 flash: fused 2.94 ms -> 1.40 + 0.30 ms.
# PRIMUS_DSA_BWD_SPLIT_DQ=0 selects the fully fused kernel.
_GFX1250_SPLIT_DQ_DEFAULT = "1"
_GFX1250_DQ_BLOCK_D = 64
_GFX1250_DQ_TILE_K = 64
_GFX1250_DQ_NUM_WARPS = 4
_GFX1250_GATHER_NSPLIT = 8
_GFX1250_GATHER_BLOCK_K = 64
_GFX1250_GATHER_NUM_WARPS = 8
# Single-chunk (R_CHUNK = topk) workspace cap for the fused path: the bf16 dKV
# intermediate is T * R_CHUNK * 512 * 2 B. Above the cap fall back to 256-wide chunks.
_GFX1250_FUSED_MAX_WS_BYTES = int(float(os.environ.get("PRIMUS_DSA_BWD_FUSED_MAX_WS_GB", "4")) * (1 << 30))


def _gfx1250_tuned() -> bool:
    """gfx1250-specific backward (fused kernel + retuned configs). PRIMUS_DSA_GFX1250_TUNED=0
    restores the original (gfx950-tuned) path on gfx1250 for A/B runs; other archs never
    take it."""
    return is_gfx1250() and os.environ.get("PRIMUS_DSA_GFX1250_TUNED", "1") == "1"


def sparse_mla_bwd_v4_triton(q, kv, o, do, topk_indices, lse, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA backward (plain Triton / MFMA, non-atomic).

    Returns ``(dq, dkv, d_sink)`` with ``dkv`` shaped like ``kv`` and ``d_sink``
    ``[num_heads]`` fp32 (or ``None`` when ``attn_sink`` is None).
    """
    assert q.is_contiguous() and o.is_contiguous() and do.is_contiguous()
    assert topk_indices.is_contiguous() and lse.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    num_kv = kv.shape[0]

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    if turbo_gfx1250_enabled(q, num_heads, topk):
        return sparse_mla_bwd_turbo_gfx1250(q, kv, o, do, topk_indices, lse, attn_sink, kv_lora_rank, scale)
    if (
        _gfx1250_tuned()
        and num_heads <= 64
        and kv_lora_rank % 16 == 0
        and os.environ.get("PRIMUS_DSA_BWD_GFX1250_FUSED", "1") == "1"
    ):
        return _sparse_mla_bwd_gfx1250_fused(q, kv, o, do, topk_indices, lse, attn_sink, kv_lora_rank, scale)

    # Delta = rowsum(O*dO) is fused into the first dQ chunk (no preprocess kernel).
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    # ---- config (mirror the gluon bwd chunking) ----
    # R_CHUNK (rank-chunk width): dQ is read-modify-written across chunks, so more
    # chunks = more redundant dq reload passes + repeated launches/CSR builds. For
    # high head counts (H>=128) the dq RMW volume is large, so a single chunk over
    # the whole topk (bounded for memory) is a big win (-22% on pro cr4). For low
    # head counts (H<=64) the smaller dq RMW is outweighed by the larger per-chunk
    # buffers/occupancy, and 256 stays best — so keep the cap there.
    if num_heads >= 128:
        R_CHUNK = min(topk, 1536)
    else:
        R_CHUNK = min(256, topk)
    # The defaults above are tuned for SHORT sequences, where the per-chunk buffers are
    # small and the dq read-modify-write traffic dominates. At long context that trade
    # inverts hard: the buffers below scale with total_tokens * R_CHUNK, so at
    # S_local = 131072 (1M context, CP=8) R_CHUNK=256 makes `interm` alone
    # 131072 * 256 * 576 * 2 B = 36 GiB, plus 8 GiB for chunk_dS/chunk_P -- a 44 GiB
    # workspace spent to avoid some dq reloads. PRIMUS_DSA_BWD_R_CHUNK trades that back.
    # Chunking is a partition of the same computation, so any value is numerically
    # equivalent; note that R_CHUNK % 128 != 0 also disables the fused dKV path below.
    _r_override = os.environ.get("PRIMUS_DSA_BWD_R_CHUNK", "")
    if _r_override:
        R_CHUNK = max(1, min(int(_r_override), topk))
    # BH_DQ x D_V is the dominant LDS term of _bwd_chunk_dq_store_ds. With the V4
    # latent head_dim of 512, BH_DQ=64 needs 64*512*2 = 65536 B for the Q tile
    # alone -- exactly the whole 64 KB LDS budget of gfx942/CDNA3 -- and the kernel
    # asks for 73728 B total, so it fails to compile there. gfx950/CDNA4 has 160 KB
    # and is unaffected. Expose the head-block so CDNA3 can halve it (32 -> 32 KB
    # Q tile). Default is unchanged.
    # Triton multi-buffers LDS operand tiles across pipeline stages; on gfx942
    # (64 KB LDS vs gfx950's 160 KB) the default staging pushes this kernel to
    # 73728 B. num_stages=1 disables the double-buffering. 0 = leave to Triton.
    _BWD_NUM_STAGES = int(os.environ.get("PRIMUS_DSA_BWD_NUM_STAGES", "0")) or None
    BH_DQ = int(os.environ.get("PRIMUS_DSA_BWD_BLOCK_H", "64"))
    TK_DQ = int(os.environ.get("PRIMUS_DSA_BWD_TILE_K", "16"))
    # dKV-intermediate tiling. The default (BH_DKV=32, TK_DKV=64) is best for high
    # head counts (H>=128) and for chunk widths that are not 128-aligned. For low
    # head counts (H<=64) with a 128-aligned chunk, a wider TILE_K=128 over a single
    # head-group (BH_DKV=64, NUM_HG=1) reduces redundant Q/dO re-loads and issues
    # fuller MMAs (measured ~6-7% faster on the full flash bwd: cr=0 1.21->1.14 ms,
    # cr=4 5.44->5.09 ms in bench_v4_attention); it regresses for H>=128 (register
    # pressure) and for non-128-aligned chunks (partial tiles), so it is guarded.
    #
    # The wide config's per-launch LDS overflows the 160 KB limit ONLY when the AMD
    # ping-pong / async-copy knobs are on (primus_turbo's set_triton_knobs_gfx950()
    # enables them globally in training), which double-buffer the LDS operand tiles
    # and ~double this kernel's shared memory (BH64/TK128 R_CHUNK256: fits
    # standalone, 347904 B in training). Since the whole bwd now compiles with those
    # knobs disabled (see the amd_pingpong_disabled scope below), the wide config
    # fits in both the benchmark and training, so it is the default where it applies
    # (~6-7% faster on the full flash bwd). Set PRIMUS_DSA_DKV_SAFE=1 to force the
    # narrow 32/64 (which fits regardless of the knobs).
    use_wide_dkv = (
        num_heads <= 64 and R_CHUNK % 128 == 0 and os.environ.get("PRIMUS_DSA_DKV_SAFE", "0") != "1"
    )
    BH_DKV, TK_DKV = (64, 128) if use_wide_dkv else (32, 64)
    DKV_NUM_WARPS = 4
    if _gfx1250_tuned():
        # gfx1250 (wave32): num_warps=4 spills the [D_V, TILE_K] fp32 accumulator of the
        # wide tile (1023 VGPRs, 181 spills); 8 waves with a 32-head group is spill-free
        # and 2.4x faster on this kernel (cr4 flash 2.48 -> 0.99 ms; sweep in REPORT.md).
        BH_DKV, TK_DKV = (32, 128) if use_wide_dkv else (32, 64)
        DKV_NUM_WARPS = 8
    num_hg_dq = triton.cdiv(num_heads, BH_DQ)
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    # In the V4 single-latent form the D_ROPE block of q/kv is a zero pad (RoPE is
    # baked in-place over the 512 latent) and its gradient is discarded by the
    # adapter (dq[..., :D_V], dkv[..., :D_V]). So all rope MMAs / K_rope loads /
    # interm-rope traffic compute a provably-zero result that is thrown away.
    # Skip them: bit-identical non-rope outputs, zero rope outputs (== gluon).
    HAS_ROPE = False

    dq = torch.empty_like(q)
    chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    dkv_acc = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

    # pad topk to an R_CHUNK multiple (-1 = invalid).
    topk_padded_len = ((topk + R_CHUNK - 1) // R_CHUNK) * R_CHUNK
    if topk_padded_len != topk:
        pad = torch.full((total_tokens, topk_padded_len - topk), -1, dtype=torch.int32, device=q.device)
        topk_padded = torch.cat([topk_indices, pad], dim=1).contiguous()
    else:
        topk_padded = topk_indices

    all_csr = [
        _build_inverted_topk_slice(topk_padded[:, rs : rs + R_CHUNK], rs, R_CHUNK, num_kv=num_kv)
        for rs in range(0, topk, R_CHUNK)
    ]

    # The AMD ping-pong / async-copy knobs primus_turbo enables globally are a
    # pessimization for the whole triton_v2 backward (measured ~5-7% slower on the
    # full bwd for both flash and pro in bench_v4_attention), and they overflow the
    # 160 KB LDS limit for the wide dKV tiling. Compile the entire bwd (dQ /
    # dKV-intermediate / gather) with them disabled; the knobs are read at compile
    # time and are not in Triton's cache key, so this pins the faster non-ping-pong
    # schedule for these kernels without touching any kernel compiled elsewhere.
    # PRIMUS_DSA_BWD_PINGPONG_OFF=0 keeps the ambient knobs (the wide dKV still
    # forces them off below, since it does not fit otherwise).
    _bwd_ctx = (
        amd_pingpong_disabled()
        if os.environ.get("PRIMUS_DSA_BWD_PINGPONG_OFF", "1") == "1"
        else contextlib.nullcontext()
    )
    with _bwd_ctx:
        for chunk_idx, r_start in enumerate(range(0, topk, R_CHUNK)):
            is_first = r_start == 0

            _bwd_chunk_dq_store_ds[(total_tokens, num_hg_dq)](
                q,
                kv,
                do,
                topk_padded,
                lse,
                delta,
                o,
                dq,
                chunk_dS,
                chunk_P,
                q.stride(0),
                q.stride(1),
                kv.stride(0),
                do.stride(0),
                do.stride(1),
                o.stride(0),
                o.stride(1),
                dq.stride(0),
                dq.stride(1),
                topk_padded.stride(0),
                chunk_dS.stride(0),
                chunk_dS.stride(1),
                scale,
                num_heads,
                r_start,
                R_CHUNK=R_CHUNK,
                BLOCK_H=BH_DQ,
                TILE_K=TK_DQ,
                D_V=kv_lora_rank,
                D_ROPE=rope_rank,
                HAS_ROPE=HAS_ROPE,
                IS_FIRST_CHUNK=is_first,
                num_warps=4,
                waves_per_eu=1,
                num_stages=_BWD_NUM_STAGES,
            )

            # The wide dKV kernel MUST compile with ping-pong off to fit LDS, even
            # if the outer bwd gate is disabled — force it off here regardless.
            _dkv_ctx = amd_pingpong_disabled() if use_wide_dkv else contextlib.nullcontext()
            with _dkv_ctx:
                _bwd_compute_dkv_intermediate[(total_tokens,)](
                    q,
                    do,
                    chunk_dS,
                    chunk_P,
                    interm,
                    q.stride(0),
                    q.stride(1),
                    do.stride(0),
                    do.stride(1),
                    chunk_dS.stride(0),
                    chunk_dS.stride(1),
                    interm.stride(0),
                    interm.stride(1),
                    num_heads,
                    R_CHUNK=R_CHUNK,
                    TILE_K=TK_DKV,
                    BLOCK_H=BH_DKV,
                    NUM_HG=num_hg_dkv,
                    D_V=kv_lora_rank,
                    D_ROPE=rope_rank,
                    HAS_ROPE=HAS_ROPE,
                    num_warps=DKV_NUM_WARPS,
                    num_stages=_BWD_NUM_STAGES,
                )

            inv_ptr, inv_data = all_csr[chunk_idx]
            _bwd_dkv_gather_acc[(num_kv,)](
                interm,
                inv_ptr,
                inv_data,
                dkv_acc,
                interm.stride(1),
                dkv_acc.stride(0),
                D_V=kv_lora_rank,
                D_ROPE=rope_rank,
                HAS_ROPE=HAS_ROPE,
                num_warps=4,
                num_stages=_BWD_NUM_STAGES,
            )

    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dkv = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv, d_sink


def _build_inverted_topk_i32(topk_slice, num_kv):
    """Same CSR inverse as ``_build_inverted_topk_slice`` (identical ordering: stable sort
    by KV row, invalid -1 entries first and skipped by ``inv_ptr[0]``), but sorts the int32
    keys directly instead of an int64 copy: the radix sort does half the passes
    (cr4 flash single chunk 0.56 -> 0.34 ms on gfx1250)."""
    flat = topk_slice.reshape(-1)
    inv_data = torch.argsort(flat, stable=True).to(torch.int32)
    counts = torch.bincount(flat + 1, minlength=num_kv + 1)
    inv_ptr = torch.cumsum(counts, dim=0).to(torch.int32)
    return inv_ptr, inv_data


def _sparse_mla_bwd_gfx1250_fused(q, kv, o, do, topk_indices, lse, attn_sink, kv_lora_rank, scale):
    """gfx1250 backward (H <= 64): fused per-token dKV intermediate + split dQ + balanced gather.

    Per rank chunk (by default the whole top-k is ONE chunk: no dQ read-modify-write and no
    padding of a last partial chunk):

    1. ``_bwd_chunk_dkv_store_ds``: one program per token holding all heads (heads are the
       MMA M dim, each gathered KV tile is loaded once) forms S, P, dP, dS in registers, writes
       the head-reduced bf16 dKV intermediate ``dS^T Q + P^T dO`` -> [T, R_CHUNK, D_V] and dS.
       (No P round trip and no second Q / dO pass, unlike the gfx950 three-kernel scheme.)
    2. ``_bwd_dq_from_ds``: dQ = dS @ K_gathered with D_V split across programs.
       (``PRIMUS_DSA_BWD_SPLIT_DQ=0`` runs 1 + 2 as the single ``_bwd_chunk_dq_dkv_fused``.)
    3. ``_bwd_dkv_gather_split``: reduce the intermediate per KV row through the CSR inverse
       of the top-k map into NSPLIT fp32 partials (deterministic, no atomics), summed into dKV.
    """
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    num_kv = kv.shape[0]
    TK = _GFX1250_FUSED_TILE_K
    BLOCK_H = max(16, triton.next_power_of_2(num_heads))

    _r_override = os.environ.get("PRIMUS_DSA_BWD_R_CHUNK", "")
    if _r_override:
        R_CHUNK = max(1, min(int(_r_override), topk))
    else:
        R_CHUNK = topk
        if total_tokens * R_CHUNK * kv_lora_rank * 2 > _GFX1250_FUSED_MAX_WS_BYTES:
            R_CHUNK = min(256, topk)

    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)
    dq = torch.empty_like(q)
    BLOCK_H = min(BLOCK_H, _GFX1250_FUSED_BLOCK_H)
    num_hg = triton.cdiv(num_heads, BLOCK_H)
    interm = torch.empty(num_hg, total_tokens, R_CHUNK, kv_lora_rank, dtype=torch.bfloat16, device=q.device)
    nsplit = _GFX1250_GATHER_NSPLIT
    part = torch.empty(nsplit, num_kv, kv_lora_rank, dtype=torch.float32, device=q.device)
    dkv_acc = None

    topk_padded_len = ((topk + R_CHUNK - 1) // R_CHUNK) * R_CHUNK
    if topk_padded_len != topk:
        pad = torch.full((total_tokens, topk_padded_len - topk), -1, dtype=torch.int32, device=q.device)
        topk_padded = torch.cat([topk_indices, pad], dim=1).contiguous()
    else:
        topk_padded = topk_indices

    split_dq = os.environ.get("PRIMUS_DSA_BWD_SPLIT_DQ", _GFX1250_SPLIT_DQ_DEFAULT) == "1" and num_hg == 1
    if split_dq:
        dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)

    with amd_pingpong_disabled():
        for r_start in range(0, topk, R_CHUNK):
            if split_dq:
                _bwd_chunk_dkv_store_ds[(total_tokens,)](
                    q,
                    kv,
                    do,
                    topk_padded,
                    lse,
                    delta,
                    o,
                    dS,
                    interm[0],
                    q.stride(0),
                    q.stride(1),
                    kv.stride(0),
                    do.stride(0),
                    do.stride(1),
                    o.stride(0),
                    o.stride(1),
                    topk_padded.stride(0),
                    dS.stride(0),
                    dS.stride(1),
                    interm.stride(1),
                    interm.stride(2),
                    scale,
                    num_heads,
                    r_start,
                    R_CHUNK=R_CHUNK,
                    BLOCK_H=BLOCK_H,
                    TILE_K=TK,
                    D_V=kv_lora_rank,
                    IS_FIRST_CHUNK=r_start == 0,
                    num_warps=_GFX1250_FUSED_NUM_WARPS,
                )
                _bwd_dq_from_ds[(total_tokens, kv_lora_rank // _GFX1250_DQ_BLOCK_D)](
                    dS,
                    kv,
                    topk_padded,
                    dq,
                    dS.stride(0),
                    dS.stride(1),
                    kv.stride(0),
                    topk_padded.stride(0),
                    dq.stride(0),
                    dq.stride(1),
                    num_heads,
                    r_start,
                    R_CHUNK=R_CHUNK,
                    BLOCK_H=BLOCK_H,
                    BLOCK_D=_GFX1250_DQ_BLOCK_D,
                    TILE_K=_GFX1250_DQ_TILE_K,
                    D_V=kv_lora_rank,
                    D_ROPE=rope_rank,
                    IS_FIRST_CHUNK=r_start == 0,
                    num_warps=_GFX1250_DQ_NUM_WARPS,
                )
            else:
                _bwd_chunk_dq_dkv_fused[(total_tokens, num_hg)](
                    q,
                    kv,
                    do,
                    topk_padded,
                    lse,
                    delta,
                    o,
                    dq,
                    interm,
                    q.stride(0),
                    q.stride(1),
                    kv.stride(0),
                    do.stride(0),
                    do.stride(1),
                    o.stride(0),
                    o.stride(1),
                    dq.stride(0),
                    dq.stride(1),
                    topk_padded.stride(0),
                    interm.stride(1),
                    interm.stride(2),
                    interm.stride(0),
                    scale,
                    num_heads,
                    r_start,
                    R_CHUNK=R_CHUNK,
                    BLOCK_H=BLOCK_H,
                    TILE_K=TK,
                    D_V=kv_lora_rank,
                    D_ROPE=rope_rank,
                    IS_FIRST_CHUNK=r_start == 0,
                    num_warps=_GFX1250_FUSED_NUM_WARPS,
                )
            inv_ptr, inv_data = _build_inverted_topk_i32(topk_padded[:, r_start : r_start + R_CHUNK], num_kv)
            _bwd_dkv_gather_split[(num_kv, nsplit)](
                interm,
                inv_ptr,
                inv_data,
                part,
                interm.stride(2),
                part.stride(0),
                part.stride(1),
                interm.stride(0),
                NSPLIT=nsplit,
                NUM_HG=num_hg,
                D_V=kv_lora_rank,
                BLOCK_K=_GFX1250_GATHER_BLOCK_K,
                num_warps=_GFX1250_GATHER_NUM_WARPS,
            )
            dkv_acc = part.sum(0) if dkv_acc is None else dkv_acc.add_(part.sum(0))

    d_sink = None
    if attn_sink is not None:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dkv = torch.zeros(num_kv, 1, d_qk, dtype=kv.dtype, device=q.device)
    dkv[:, 0, :kv_lora_rank] = dkv_acc
    return dq, dkv, d_sink

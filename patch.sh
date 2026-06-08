#!/bin/bash
# Patch FLA Triton kernels to fix autotuning hangs on MI300X/ROCm.
#
# Two kernels hang during Triton autotuning with num_stages >= 3:
#   1. chunk_bwd_kernel_dh      in fla/ops/common/chunk_h.py
#   2. chunk_kda_bwd_kernel_intra in fla/ops/kda/chunk_intra.py
#
# Root cause: Triton's multi-stage software pipelining (num_stages >= 3)
# generates broken HIP code for kernels with dynamic loops, debug barriers,
# and complex control flow on MI300X.
#
# Fix: restrict num_stages to [2] for affected kernels.

set -e

echo "=== FLA Triton MI300X/ROCm Patch ==="
echo ""

########################################################################
# Step 0: Purge Triton cache + set up per-rank cache isolation
########################################################################

# echo "[Step 0] Purging Triton caches ..."
# rm -rf ~/.triton/cache 2>/dev/null && echo "  Removed ~/.triton/cache" || true
# rm -rf /tmp/triton* 2>/dev/null && echo "  Removed /tmp/triton*" || true
# echo "  Done."
# echo ""

# echo "[Step 0b] Installing per-rank Triton cache isolation ..."

# PERRANK_SCRIPT="/opt/venv/lib/python3.10/site-packages/fla/_triton_per_rank_cache.py"
# cat > "$PERRANK_SCRIPT" << 'PERRANK_EOF'
# """
# Per-rank Triton cache isolation for multi-GPU training.

# Import this BEFORE any Triton kernel usage (e.g. at the top of your
# training script or in an __init__.py) to give each GPU rank its own
# Triton cache directory, preventing race conditions during autotune.

# Works with torchrun, deepspeed, and bare CUDA_VISIBLE_DEVICES.
# """
# import os

# def _setup_per_rank_triton_cache():
#     rank = os.environ.get("LOCAL_RANK") or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK") or "0"
#     base = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
#     per_rank = os.path.join(base, f"rank_{rank}")
#     os.makedirs(per_rank, exist_ok=True)
#     os.environ["TRITON_CACHE_DIR"] = per_rank

# _setup_per_rank_triton_cache()
# PERRANK_EOF

# FLA_INIT="/opt/venv/lib/python3.10/site-packages/fla/__init__.py"
# if [ -f "$FLA_INIT" ]; then
#     cp "$FLA_INIT" "${FLA_INIT}.bak" 2>/dev/null || true
#     IMPORT_LINE="import fla._triton_per_rank_cache  # per-rank cache isolation"
#     if ! grep -qF "$IMPORT_LINE" "$FLA_INIT"; then
#         sed -i "1i\\$IMPORT_LINE" "$FLA_INIT"
#         echo "  Injected per-rank cache import into $FLA_INIT"
#     else
#         echo "  Per-rank cache import already present in $FLA_INIT"
#     fi
# else
#     echo "  WARNING: $FLA_INIT not found, per-rank cache not auto-injected."
#     echo "  Add 'import fla._triton_per_rank_cache' to the top of your training script."
# fi
# echo "  Each rank gets: ~/.triton/cache/rank_\$LOCAL_RANK/"
# echo ""

########################################################################
# Locate fla package (supports editable installs and site-packages)
########################################################################

FLA_ROOT=$(python3 -c "import fla, os; print(os.path.dirname(fla.__file__))" 2>/dev/null)
if [ -z "$FLA_ROOT" ]; then
    echo "ERROR: Could not locate the fla package. Is flash-linear-attention installed?"
    exit 1
fi
echo "FLA package found at: $FLA_ROOT"
echo ""

########################################################################
# Patch 1: chunk_kda_bwd_kernel_intra (fla/ops/kda/chunk_intra.py)
#   The autotuning tries num_stages in [2, 3, 4]. num_stages >= 3 hangs.
########################################################################

INTRA_TARGET="$FLA_ROOT/ops/kda/chunk_intra.py"

if [ -f "$INTRA_TARGET" ]; then
    cp "$INTRA_TARGET" "${INTRA_TARGET}.bak"
    echo "[Patch 1] Patching $INTRA_TARGET"

    sed -i 's/for num_stages in \[2, 3, 4\]/for num_stages in [2]/g' "$INTRA_TARGET"

    echo "  - Restricted num_stages to [2] for all autotune configs"
    echo "  - Backed up original to ${INTRA_TARGET}.bak"
    echo ""
else
    echo "[Patch 1] SKIP: $INTRA_TARGET not found"
    echo ""
fi

########################################################################
# Patch 2: chunk_bwd_kernel_dh (fla/ops/common/chunk_h.py)
#   The backward kernel autotuning hangs on ROCm.
#   Fix: replace autotuning with fixed config (num_warps=4, num_stages=2).
########################################################################

# TARGET="/opt/venv/lib/python3.10/site-packages/fla/ops/common/chunk_h.py"

# if [ ! -f "$TARGET" ]; then
#     echo "[Patch 2] SKIP: $TARGET not found"
# else

# cp "$TARGET" "${TARGET}.bak"
# echo "[Patch 2] Patching $TARGET"

# cat > "$TARGET" << 'PATCH_EOF'
# # Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# # Patched for MI300X/ROCm: disabled autotuning on backward kernel to prevent hang.

# import torch
# import triton
# import triton.language as tl

# from fla.ops.utils import prepare_chunk_offsets
# from fla.ops.utils.op import exp
# from fla.utils import autotune_cache_kwargs, check_shared_mem

# BKV_LIST = [32, 64] if check_shared_mem() else [16, 32]


# @triton.heuristics({
#     'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
#     'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
# @triton.autotune(
#     configs=[
#         triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
#         for BK in BKV_LIST
#         for BV in BKV_LIST
#         for num_warps in [4, 8]
#         for num_stages in [2, 3]
#     ],
#     key=['BT', 'USE_G', 'USE_GK', 'USE_GV'],
#     **autotune_cache_kwargs,
# )
# @triton.jit(do_not_specialize=['T'])
# def chunk_fwd_kernel_h(
#     k,
#     v,
#     h,
#     g,
#     g_gamma,
#     gk,
#     gv,
#     h0,
#     ht,
#     cu_seqlens,
#     split_offsets,
#     T,
#     H: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BS: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     USE_G: tl.constexpr,
#     USE_G_GAMMA: tl.constexpr,
#     USE_GK: tl.constexpr,
#     USE_GV: tl.constexpr,
#     USE_INITIAL_STATE: tl.constexpr,
#     STORE_FINAL_STATE: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
# ):
#     i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_n, i_h = i_nh // H, i_nh % H
#     if IS_VARLEN:
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#         NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
#         boh = tl.load(split_offsets + i_n).to(tl.int32)
#     else:
#         bos, eos = i_n * T, i_n * T + T
#         NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
#         boh = i_n * NS
#     NTS = BS // BT

#     if USE_G_GAMMA:
#         b_gamma = tl.load(g_gamma + i_h)
#         b_g = b_gamma * (tl.arange(0, BT) + 1)

#     # [BK, BV]
#     b_h = tl.zeros([BK, BV], dtype=tl.float32)
#     if USE_INITIAL_STATE:
#         p_h0 = tl.make_block_ptr(h0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
#         b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

#     for i_t in range(NT):
#         i_s = i_t // NTS
#         p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
#         p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

#         o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
#         p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

#         if i_t % NTS == 0:
#             tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
#         # [BK, BT]
#         b_k = tl.load(p_k, boundary_check=(0, 1))
#         # [BT, BV]
#         b_v = tl.load(p_v, boundary_check=(0, 1))
#         last_idx = min((i_t + 1) * BT, T) - 1

#         # scalar decay
#         if USE_G:
#             b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
#             p_g = g + bos*H + (i_t * BT + tl.arange(0, BT)) * H + i_h
#             b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
#             b_h *= exp(b_g_last)
#             b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

#         if USE_G_GAMMA:
#             b_g_last = b_gamma * min(BT, T - i_t * BT)
#             b_h *= exp(b_g_last)
#             b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

#         # vector decay, h = Diag(gk) @ h
#         if USE_GK:
#             p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
#             p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

#             b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
#             b_h *= exp(b_gk_last)[:, None]

#             b_gk = tl.load(p_gk, boundary_check=(0, 1))
#             b_k = (b_k * exp(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

#         # vector decay, h = h @ Diag(gv)
#         if USE_GV:
#             p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#             p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

#             b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
#             b_h *= exp(b_gv_last)[None, :]

#             b_gv = tl.load(p_gv, boundary_check=(0, 1))
#             b_v = (b_v * exp(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

#         b_h += tl.dot(b_k, b_v)

#     if STORE_FINAL_STATE:
#         p_ht = tl.make_block_ptr(ht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
#         tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# # PATCHED: Fixed config for backward kernel to avoid autotuning hang on MI300X/ROCm.
# # Original used @triton.autotune which hangs during config search on AMD GPUs.
# @triton.heuristics({
#     'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
#     'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
# })
# @triton.jit(do_not_specialize=['T'])
# def chunk_bwd_kernel_dh(
#     q,
#     g,
#     g_gamma,
#     gk,
#     gv,
#     do,
#     dh,
#     dht,
#     dh0,
#     cu_seqlens,
#     split_offsets,
#     scale,
#     T,
#     HQ: tl.constexpr,
#     H: tl.constexpr,
#     K: tl.constexpr,
#     V: tl.constexpr,
#     BT: tl.constexpr,
#     BS: tl.constexpr,
#     BK: tl.constexpr,
#     BV: tl.constexpr,
#     NG: tl.constexpr,
#     USE_G: tl.constexpr,
#     USE_G_GAMMA: tl.constexpr,
#     USE_GK: tl.constexpr,
#     USE_GV: tl.constexpr,
#     STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
#     USE_FINAL_STATE_GRADIENT: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
# ):
#     i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     i_n, i_hq = i_nh // HQ, i_nh % HQ
#     i_h = i_hq // NG
#     if IS_VARLEN:
#         bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
#         T = eos - bos
#         NT = tl.cdiv(T, BT)
#         NS = tl.cdiv(T, BS)
#         boh = tl.load(split_offsets + i_n).to(tl.int32)
#     else:
#         bos, eos = i_n * T, i_n * T + T
#         NT = tl.cdiv(T, BT)
#         NS = tl.cdiv(T, BS)
#         boh = i_n * NS

#     if USE_G_GAMMA:
#         b_gamma = tl.load(g_gamma + i_h)
#         b_g = b_gamma * (tl.arange(0, BT) + 1)

#     # [BK, BV]
#     b_dh = tl.zeros([BK, BV], dtype=tl.float32)
#     if USE_FINAL_STATE_GRADIENT:
#         p_dht = tl.make_block_ptr(dht + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
#         b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

#     for i_t in range(NT - 1, -1, -1):
#         i_s = i_t // (BS // BT)
#         o_dh = ((boh + i_s) * H + i_h).to(tl.int64) * K*V
#         p_dh = tl.make_block_ptr(dh + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

#         if i_t % (BS // BT) == 0:
#             tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
#         last_idx = min(i_t * BT + BT, T) - 1
#         # [BK, BT]
#         p_q = tl.make_block_ptr(q + (bos*HQ + i_hq) * K, (K, T), (1, HQ*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
#         p_do = tl.make_block_ptr(do + (bos*HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#         b_q = tl.load(p_q, boundary_check=(0, 1))
#         b_q = (b_q * scale).to(b_q.dtype)
#         # [BT, BV]
#         b_do = tl.load(p_do, boundary_check=(0, 1))

#         if USE_G:
#             p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
#             b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
#             b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
#             b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)
#             b_dh *= exp(b_g_last)

#         if USE_G_GAMMA:
#             b_g_last = b_gamma * min(BT, T - i_t * BT)
#             b_q = (b_q * exp(b_g)[None, :]).to(b_q.dtype)
#             b_dh *= exp(b_g_last)

#         if USE_GK:
#             p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
#             p_gk_last = gk + (bos + last_idx) * H*K + i_h * K + i_k * BK + tl.arange(0, BK)

#             b_gk = tl.load(p_gk, boundary_check=(0, 1))
#             b_q = (b_q * exp(b_gk)).to(b_q.dtype)
#             b_gk_last = tl.load(p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.)
#             b_dh *= exp(b_gk_last)[:, None]

#         if USE_GV:
#             p_gv = tl.make_block_ptr(gv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
#             p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

#             b_gv = tl.load(p_gv, boundary_check=(0, 1))
#             b_do = (b_do * exp(b_gv))

#             b_gv_last = tl.load(p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.)
#             b_dh *= exp(b_gv_last)[None, :]

#         b_dh += tl.dot(b_q, b_do.to(b_q.dtype))

#     if STORE_INITIAL_STATE_GRADIENT:
#         p_dh0 = tl.make_block_ptr(dh0 + i_nh * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
#         tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


# def chunk_fwd_h(
#     k: torch.Tensor,
#     v: torch.Tensor,
#     g: torch.Tensor | None = None,
#     g_gamma: torch.Tensor | None = None,
#     gk: torch.Tensor | None = None,
#     gv: torch.Tensor | None = None,
#     h0: torch.Tensor | None = None,
#     output_final_state: bool = False,
#     cu_seqlens: torch.Tensor | None = None,
#     chunk_size: int = 64,
#     split_size: int | None = None,
#     states_in_fp32: bool = False,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     B, T, H, K, V = *k.shape, v.shape[-1]
#     BT = chunk_size
#     BS = BT if split_size is None else split_size
#     assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
#     if cu_seqlens is None:
#         N, NS, split_offsets = B, triton.cdiv(T, BS), None
#     else:
#         split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
#         N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()

#     h = k.new_empty(B, NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
#     ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
#     def grid(meta): return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)
#     chunk_fwd_kernel_h[grid](
#         k=k,
#         v=v,
#         h=h,
#         g=g,
#         g_gamma=g_gamma,
#         gk=gk,
#         gv=gv,
#         h0=h0,
#         ht=ht,
#         cu_seqlens=cu_seqlens,
#         split_offsets=split_offsets,
#         T=T,
#         H=H,
#         K=K,
#         V=V,
#         BT=BT,
#         BS=BS,
#         USE_G=g is not None,
#         USE_G_GAMMA=g_gamma is not None,
#         USE_GK=gk is not None,
#         USE_GV=gv is not None,
#     )
#     return h, ht


# def chunk_bwd_dh(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     do: torch.Tensor,
#     h0: torch.Tensor,
#     dht: torch.Tensor,
#     scale: float,
#     g: torch.Tensor | None = None,
#     g_gamma: torch.Tensor | None = None,
#     gk: torch.Tensor | None = None,
#     gv: torch.Tensor | None = None,
#     cu_seqlens: torch.Tensor | None = None,
#     chunk_size: int = 64,
#     split_size: int | None = None,
#     states_in_fp32: bool = False,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     B, T, H, K, V = *k.shape, v.shape[-1]
#     HQ = q.shape[2]
#     BT = chunk_size
#     BS = BT if split_size is None else split_size
#     assert BS % BT == 0, f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
#     if cu_seqlens is None:
#         N, NS, split_offsets = B, triton.cdiv(T, BS), None
#     else:
#         split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
#         N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()
#     NG = HQ // H

#     dh = k.new_empty(B, NS, HQ, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
#     dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

#     # PATCHED: Use fixed BK/BV instead of autotuning to avoid hang on MI300X/ROCm.
#     BK_fixed = 64 if K >= 64 else 32
#     BV_fixed = 64 if V >= 64 else 32
#     def grid(meta): return (triton.cdiv(K, BK_fixed), triton.cdiv(V, BV_fixed), N * H)
#     chunk_bwd_kernel_dh[grid](
#         q=q,
#         g=g,
#         g_gamma=g_gamma,
#         gk=gk,
#         gv=gv,
#         do=do,
#         dh=dh,
#         dht=dht,
#         dh0=dh0,
#         cu_seqlens=cu_seqlens,
#         split_offsets=split_offsets,
#         scale=scale,
#         T=T,
#         HQ=HQ,
#         H=H,
#         K=K,
#         V=V,
#         BT=BT,
#         BS=BS,
#         BK=BK_fixed,
#         BV=BV_fixed,
#         NG=NG,
#         USE_G=g is not None,
#         USE_G_GAMMA=g_gamma is not None,
#         USE_GK=gk is not None,
#         USE_GV=gv is not None,
#         num_warps=4,
#         num_stages=2,
#     )
#     return dh, dh0
# PATCH_EOF

# echo "  - Forward kernel: unchanged (autotuning kept)"
# echo "  - Backward kernel: removed @triton.autotune, using fixed BK/BV with num_warps=4, num_stages=2"
# echo "  - Backed up original to ${TARGET}.bak"

# fi

echo ""
echo "=== Patch complete ==="
echo ""
echo "Per-rank cache: each GPU rank now uses ~/.triton/cache/rank_\$LOCAL_RANK/"
echo ""
echo "To restore originals:"
echo "  cp ${INTRA_TARGET}.bak $INTRA_TARGET"

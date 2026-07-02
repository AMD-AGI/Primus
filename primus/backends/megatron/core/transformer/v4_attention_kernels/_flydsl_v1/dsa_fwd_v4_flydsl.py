###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL-v1 sparse-MLA forward (native FlyDSL MFMA) launcher.

Public API mirrors :func:`sparse_mla_fwd_v4_gluon` / ``sparse_mla_fwd_v4_triton``:

    sparse_mla_fwd_v4_flydsl(q, kv, topk_indices, attn_sink=None,
                             kv_lora_rank=512, scale=None) -> (o, lse)

The heavy lifting is the native FlyDSL MFMA kernel in
``dsa_fwd_v4_flydsl_kernel.build_dsa_fwd_module``; this module just marshals the
tensors, caches the built launcher per (H, D, TOPK, has_sink, block_n), and
launches it. See the kernel module docstring for the design.
"""

from __future__ import annotations

import os
import sys
import threading

import torch

# flydsl_v1 uses only the installed `flydsl` pip package + its own local kernel
# modules — it does NOT need the /workspace/FlyDSL-amd source tree.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dsa_fwd_v4_flydsl_kernel import build_dsa_fwd_module  # noqa: E402

_KERNEL_CACHE = {}
_KERNEL_CACHE_LOCK = threading.Lock()

_DEFAULT_BLOCK_N = int(os.environ.get("PRIMUS_DSA_FLYDSL_FWD_BLOCK_N", "0"))  # 0 = shape-conditional
_DEFAULT_BLOCK_H = int(os.environ.get("PRIMUS_DSA_FLYDSL_FWD_BLOCK_H", "256"))
_DEFAULT_WPE = int(os.environ.get("PRIMUS_DSA_FLYDSL_FWD_WPE", "2"))


def _get_kernel(
    num_heads, kv_lora_rank, d_qk, topk, has_sink, block_n, block_h, single_latent, waves_per_eu, scale
):
    block_h = min(block_h, num_heads)
    while num_heads % block_h != 0:
        block_h -= 32
    key = (
        num_heads,
        kv_lora_rank,
        d_qk,
        topk,
        has_sink,
        block_n,
        block_h,
        single_latent,
        waves_per_eu,
        round(float(scale), 8),
    )
    with _KERNEL_CACHE_LOCK:
        launch = _KERNEL_CACHE.get(key)
        if launch is None:
            launch = build_dsa_fwd_module(
                num_heads=num_heads,
                kv_lora_rank=kv_lora_rank,
                d_qk=d_qk,
                topk=topk,
                dtype_str="bf16",
                sm_scale=float(scale),
                has_sink=has_sink,
                block_n=block_n,
                block_h=block_h,
                single_latent=single_latent,
                waves_per_eu=waves_per_eu,
            )
            _KERNEL_CACHE[key] = launch
        return launch


def sparse_mla_fwd_v4_flydsl(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA forward (native FlyDSL MFMA, gfx950)."""
    assert q.is_contiguous() and topk_indices.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    kv_lora_rank = int(kv_lora_rank)
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    assert kv.shape[0] >= total_tokens and kv.shape[-1] == d_qk
    assert q.dtype == torch.bfloat16, "bf16 only"

    # Shape-conditional tile: flash (H<=64) is LDS/occupancy-bound -> small tile
    # (BLOCK_N=32) doubles occupancy (occ 1->2). pro (H>=128) is VGPR-bound
    # (occupancy stuck at 1 regardless of LDS) -> a smaller tile only adds
    # barrier/softmax overhead, so keep the larger BLOCK_N=64 (fewer tiles).
    # flash (H<=64) is LDS/occupancy-bound → single-latent (one shared tile) halves
    # LDS (occ 1→2) at BLOCK_N=64 (single-latent already halves LDS, so no need for
    # a smaller tile — smaller tiles only add barriers). pro (H>=128) is VGPR-bound
    # → dual swizzled tile (conflict-free QK), extra LDS is free (occ stuck at 1).
    single_latent = num_heads <= 64
    block_n = _DEFAULT_BLOCK_N if _DEFAULT_BLOCK_N else 64
    topk = topk_indices.shape[1]
    # Pad TOPK up to a multiple of block_n with -1 (masked) if needed.
    if topk % block_n != 0:
        pad = ((topk + block_n - 1) // block_n) * block_n - topk
        topk_indices = torch.cat(
            [topk_indices, torch.full((total_tokens, pad), -1, dtype=torch.int32, device=q.device)],
            dim=1,
        ).contiguous()
        topk = topk_indices.shape[1]

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.is_contiguous() and attn_sink.dtype == torch.float32
        assert attn_sink.shape == (num_heads,)
        sink = attn_sink
    else:
        # kernel always folds the sink; -inf makes it a no-op (af=1, sink_e=0)
        sink = torch.full((num_heads,), float("-inf"), dtype=torch.float32, device=q.device)

    o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=q.dtype, device=q.device)
    lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    launch = _get_kernel(
        int(num_heads),
        kv_lora_rank,
        int(d_qk),
        int(topk),
        has_sink,
        block_n,
        _DEFAULT_BLOCK_H,
        single_latent,
        _DEFAULT_WPE,
        scale,
    )
    launch(
        q.reshape(-1),
        kv.reshape(-1),
        topk_indices.reshape(-1),
        sink.reshape(-1),
        o.reshape(-1),
        lse.reshape(-1),
        int(total_tokens),
    )
    return o, lse


__all__ = ["sparse_mla_fwd_v4_flydsl"]

"""V4 CSA attention forward FlyDSL launcher (Round 3 Step 2b).

Stage A: correctness only. Per-row design forked from Triton monolithic CSA.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from typing import Optional, Tuple

import torch

_FLYDSL_SRC = "/workspace/FlyDSL-amd"
if _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

os.environ.setdefault("FLYDSL_WAVES_PER_EU", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from v4_csa_fwd_kernel import build_v4_csa_fwd_module  # noqa: E402

_KERNEL_CACHE = {}
_KERNEL_CACHE_LOCK = threading.Lock()


def _get_kernel(num_heads_q, head_dim, swa_window, dtype_str,
                block_n, block_k, waves_per_eu, has_sink, has_sparse, mqa_kv,
                head_group=1):
    key = (num_heads_q, head_dim, swa_window, dtype_str,
           block_n, block_k, waves_per_eu, has_sink, has_sparse, mqa_kv, head_group)
    with _KERNEL_CACHE_LOCK:
        if key in _KERNEL_CACHE:
            return _KERNEL_CACHE[key]
        launch = build_v4_csa_fwd_module(
            num_heads=num_heads_q,
            head_dim=head_dim,
            swa_window=int(swa_window),
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            block_n=block_n,
            block_k=block_k,
            has_sink=has_sink,
            has_sparse=has_sparse,
            mqa_kv=mqa_kv,
            head_group=head_group,
        )
        _KERNEL_CACHE[key] = launch
        return launch


def _broadcast_kv_mqa(k: torch.Tensor, v: torch.Tensor, head_q: int):
    if k.shape[1] == head_q:
        return k, v
    if k.shape[1] != 1:
        raise ValueError(f"MQA expects K_H=1; got {k.shape[1]}")
    k_full = k.expand(-1, head_q, -1, -1).clone(memory_format=torch.contiguous_format)
    v_full = v.expand(-1, head_q, -1, -1).clone(memory_format=torch.contiguous_format)
    return k_full, v_full


def _launch_v4_attention_fwd_csa(
    q: torch.Tensor,           # [B, H, Sq, D]
    k_local: torch.Tensor,     # [B, H_KV, Sq, D]  (H_KV=1 for MQA)
    v_local: torch.Tensor,     # [B, H_KV, Sq, D]
    gathered: torch.Tensor,    # [B, Sq, K_topk, D]
    *,
    sink: Optional[torch.Tensor],  # [H] fp32 or None
    swa_window: int,
    sparse_mask: torch.Tensor,     # [B, Sq, K_topk] additive
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """V4 CSA forward launcher.

    Returns (out, lse) where out is bf16 [B, H, Sq, D] and lse is fp32 [B, H, Sq]
    in raw-qk-scaled-domain (m + ln(l), where m absorbs sm_scale).
    """
    if q.dim() != 4 or k_local.dim() != 4 or v_local.dim() != 4:
        raise ValueError("rank-4 q/k_local/v_local required")
    if gathered.dim() != 4:
        raise ValueError("rank-4 gathered [B,Sq,K_topk,D] required")
    if sparse_mask.dim() != 3:
        raise ValueError("rank-3 sparse_mask [B,Sq,K_topk] required")
    B, HQ, Sq, D = q.shape
    Bk, HK, Sk, Dk = k_local.shape
    if (Bk, Sk, Dk) != (B, Sq, D) or v_local.shape != k_local.shape:
        raise ValueError("k_local/v_local shape mismatch w.r.t. q")
    if HK != 1 and HK != HQ:
        raise ValueError(f"K_H must be 1 or {HQ}; got {HK}")
    Bg, Sqg, K_topk, Dg = gathered.shape
    if Bg != B or Sqg != Sq or Dg != D:
        raise ValueError(f"gathered shape mismatch {tuple(gathered.shape)}")
    Bm, Sqm, Km = sparse_mask.shape
    if Bm != B or Sqm != Sq or Km != K_topk:
        raise ValueError(f"sparse_mask shape mismatch {tuple(sparse_mask.shape)}")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(f"bf16 only; got {q.dtype}")
    if D != 512:
        raise NotImplementedError(f"head_dim=512 only; got {D}")
    if int(swa_window) <= 0:
        raise NotImplementedError("swa_window > 0 required")
    expected_scale = 1.0 / math.sqrt(D)
    if not math.isclose(float(scale), expected_scale, rel_tol=1e-4):
        raise NotImplementedError(f"only scale=1/sqrt(D) supported; got {scale}")

    has_sink = sink is not None
    has_sparse = int(K_topk) > 0
    mqa = (HK == 1) and (HQ > 1)

    if has_sink:
        sink_fp32 = sink.float().contiguous()
        if sink_fp32.shape != (HQ,):
            raise ValueError(f"sink shape must be ({HQ},); got {tuple(sink_fp32.shape)}")
    else:
        sink_fp32 = torch.zeros((max(HQ, 1),), dtype=torch.float32, device=q.device)

    # MQA: avoid the .expand().clone() broadcast — pass [B,1,Sq,D] directly,
    # kernel uses mqa_kv=True to drop head_idx from K/V indexing.
    q_bhld = q.contiguous()
    if mqa:
        k_bhld = k_local.contiguous()
        v_bhld = v_local.contiguous()
    else:
        k_bhld = k_local.contiguous()
        v_bhld = v_local.contiguous()
    o_bhld = torch.empty_like(q_bhld)
    lse = torch.zeros((B, HQ, Sq), device=q.device, dtype=torch.float32)

    # Sparse mask & gathered must be contiguous fp32 / bf16.
    if K_topk > 0:
        gathered_c = gathered.contiguous()
        # sparse_mask is bf16 (typically) -> upcast to fp32 here so kernel can
        # just add it as f32. (Triton path takes the dtype of input.)
        if sparse_mask.dtype != torch.float32:
            sparse_mask_fp32 = sparse_mask.float().contiguous()
        else:
            sparse_mask_fp32 = sparse_mask.contiguous()
    else:
        gathered_c = torch.empty((B, Sq, 1, D), dtype=q.dtype, device=q.device)
        sparse_mask_fp32 = torch.zeros((B, Sq, 1), dtype=torch.float32, device=q.device)

    block_n = int(os.environ.get("PRIMUS_V4_CSA_BLOCK_N", "8"))
    block_k = int(os.environ.get("PRIMUS_V4_CSA_BLOCK_K", "16"))
    waves_per_eu = int(os.environ.get("FLYDSL_WAVES_PER_EU", "2"))
    # Round 5: HG=2 is the banked default for sparse path (1.25x speedup vs HG=1 on
    # both pro_CSA_4096 and flash_CSA_4096 at the same VGPR=250 / spill=0 footprint).
    # K_topk==0 (dense fallback) still forces HG=1 below.
    head_group = int(os.environ.get("PRIMUS_V4_CSA_HEAD_GROUP", "2"))
    # Round 5 fix: dense-fallback path (K_topk==0) has no payoff from head-group
    # fusion AND triggered a flyc closure-cache collision when has_sparse differed
    # between HG=1 dense and HG>1 sparse compiles. Force HG=1 when dense.
    if not has_sparse:
        head_group = 1
    if HQ % head_group != 0:
        # Silently fall back to HG=1 if shape doesn't divide.
        head_group = 1
    launch = _get_kernel(HQ, D, int(swa_window), "bf16",
                          block_n, block_k, waves_per_eu,
                          has_sink, has_sparse, mqa,
                          head_group=head_group)

    launch(
        q_bhld.view(-1),
        k_bhld.view(-1),
        v_bhld.view(-1),
        gathered_c.view(-1),
        sparse_mask_fp32.view(-1),
        sink_fp32.view(-1),
        o_bhld.view(-1),
        lse.view(-1),
        B,
        Sq,
        int(K_topk),
    )
    return o_bhld, lse


__all__ = ["_launch_v4_attention_fwd_csa"]

"""V4 SWA attention forward FlyDSL launcher with MQA stride-trick (Round 3 Stage C).

Eliminates the K/V .expand().clone() broadcast that allocates H_Q copies of
K and V. Passes the un-broadcast [B, 1, Sk, D] view directly; the kernel reads
K/V with stride_kh=0 via the mqa_kv compile-time flag.
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

os.environ.setdefault("FLYDSL_SLA_FWD_ENABLE_DMA", "1")
os.environ.setdefault("FLYDSL_WAVES_PER_EU", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from v4_sla_fwd_kernel import build_v4_swa_fwd_module  # noqa: E402

_KERNEL_CACHE = {}
_KERNEL_CACHE_LOCK = threading.Lock()


def _get_kernel(
    num_heads_q, head_dim, swa_window, dtype_str, block_m, block_n, waves_per_eu, mqa_kv, flat_work_group_size
):
    key = (
        num_heads_q,
        head_dim,
        swa_window,
        dtype_str,
        block_m,
        block_n,
        waves_per_eu,
        mqa_kv,
        flat_work_group_size,
    )
    with _KERNEL_CACHE_LOCK:
        if key in _KERNEL_CACHE:
            return _KERNEL_CACHE[key]
        launch = build_v4_swa_fwd_module(
            num_heads=num_heads_q,
            head_dim=head_dim,
            swa_window=int(swa_window),
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            block_m=block_m,
            block_n=block_n,
            flat_work_group_size=flat_work_group_size,
            layout_bhld=True,
            mqa_kv=mqa_kv,
        )
        _KERNEL_CACHE[key] = launch
        return launch


def _launch_v4_attention_fwd_flydsl_mqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    additive_mask: Optional[torch.Tensor],
    scale: float,
    hca_local_seqlen: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SWA-only launcher that AVOIDS the MQA broadcast.

    For K_H=1 (MQA), passes the [B, 1, Sk, D] tensor directly and uses
    the mqa_kv kernel variant. For K_H=HQ (non-MQA), falls back to the
    standard kernel.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("rank-4 q/k/v required")
    B, HQ, Sq, D = q.shape
    Bk, HK, Sk, Dk = k.shape
    if (Bk, Sk, Dk) != (B, Sk, D) or v.shape != k.shape:
        raise ValueError(f"shape mismatch")
    if HK != 1 and HK != HQ:
        raise ValueError(f"K_H must be 1 or {HQ}; got {HK}")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(f"bf16 only")
    if D != 512:
        raise NotImplementedError(f"head_dim=512 only")
    if sink is not None:
        raise NotImplementedError("sink not yet supported in MQA wrapper")
    if additive_mask is not None or int(hca_local_seqlen) != 0:
        raise NotImplementedError("HCA mode not in this wrapper")
    if int(swa_window) <= 0:
        raise NotImplementedError("swa_window > 0 required")
    if Sq != Sk:
        raise NotImplementedError("SWA: Sq==Sk required")
    expected_scale = 1.0 / math.sqrt(D)
    if not math.isclose(float(scale), expected_scale, rel_tol=1e-4):
        raise NotImplementedError(f"scale=1/sqrt(D) only")

    mqa = (HK == 1) and (HQ > 1)

    q_bhld = q.contiguous()
    if mqa:
        # K/V are [B, 1, Sk, D]. We pass them flat-view but use mqa_kv kernel
        # variant that drops head_idx from K/V indexing.
        k_bhld = k.contiguous()
        v_bhld = v.contiguous()
    else:
        k_bhld = k.contiguous()
        v_bhld = v.contiguous()
    o_bhld = torch.empty_like(q_bhld)
    lse = torch.zeros((B, HQ, Sq), device=q.device, dtype=torch.float32)

    block_m = int(os.environ.get("PRIMUS_V4_FLYDSL_BLOCK_M", "128"))
    block_n = int(os.environ.get("PRIMUS_V4_FLYDSL_BLOCK_N", "32"))
    waves_per_eu = int(os.environ.get("FLYDSL_WAVES_PER_EU", "2"))
    fwgs_env = os.environ.get("PRIMUS_V4_FLYDSL_FWGS", "")
    flat_work_group_size = int(fwgs_env) if fwgs_env else None
    launch = _get_kernel(
        HQ, D, int(swa_window), "bf16", block_m, block_n, waves_per_eu, mqa, flat_work_group_size
    )

    launch(
        q_bhld.view(-1),
        k_bhld.view(-1),
        v_bhld.view(-1),
        o_bhld.view(-1),
        lse.view(-1),
        B,
        Sq,
    )
    return o_bhld, lse


__all__ = ["_launch_v4_attention_fwd_flydsl_mqa"]

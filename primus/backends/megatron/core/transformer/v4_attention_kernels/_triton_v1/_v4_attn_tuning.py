# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Architecture-aware tuned defaults for the V4 Triton attention kernels.

The V4 attention fwd/bwd/CSA kernels expose ~60 ``PRIMUS_V4_ATTN_*`` / ``PRIMUS_V4_CSA_*``
env knobs whose hard-coded defaults are the **gfx950 / MI355X** R1/R2 sweep winners. On
gfx1250 (MI450, CDNA-next) the optimum differs systematically -- larger ``BLOCK_M`` and
fewer warps/stages, and far more dKV head-split -- so a gfx950-tuned default leaves a lot on
the table. This module centralises the per-arch defaults so each launcher can ask for the
right value for the GPU it is running on; the env knobs still override everything.

gfx1250 values come from node-safe microbench sweeps (``ab_sweep/opt7b_*`` / ``opt7c_*``);
untuned knobs/archs fall back to the historical gfx950 defaults so nothing regresses.
"""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def gpu_arch() -> str:
    """Lower-cased GPU gfx arch (e.g. ``gfx1250``); ``""`` if it can't be determined."""
    try:
        import torch

        name = torch.cuda.get_device_properties(0).gcnArchName  # e.g. "gfx1250:sramecc+:xnack-"
        return name.split(":")[0].strip().lower()
    except Exception:
        return ""


def is_gfx1250() -> bool:
    return gpu_arch() == "gfx1250"


def fwd_attn_defaults(is_hca: bool):
    """``(BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES)`` for the V4 attention FWD kernel.

    gfx1250: ``BM=128, BN=32, W=4, S=1`` wins **both** shapes -- SWA **+62-70%** vs the gfx950
    winner ``BM=64/BN=16/W=8/S=2`` (``ab_sweep/opt7c``), and HCA (cr=128) **+15-21%** vs the
    gfx950 HCA winner ``BM=128/BN=16/W=8/S=1`` (``ab_sweep/opt7d``).
    """
    if is_gfx1250():
        return 128, 32, 4, 1
    # gfx950 / MI355X -- historical R2-sweep winners (shape-dependent BLOCK_M / stages).
    if is_hca:
        return 128, 16, 8, 1
    return 64, 16, 8, 2


def bwd_dkv_head_groups_default(hq: int, hk: int) -> int:
    """dKV head-split groups for the V4 attention BWD -- ONLY the MQA (HQ>=64, HK==1) path.

    gfx1250 sweep (``ab_sweep/opt7b``, MQA): ``HG=32`` is +37-53% vs the gfx950 default ``2``
    (the kernel notes "HG=4/8 regress" -- true on gfx950, inverted on gfx1250).

    NOTE: the current DeepSeek-V4-Pro attention runs **MHA at the kernel level** -- the
    single-latent KV is expanded to all H heads (deepseek_v4_attention.py), so HK==HQ and the
    caller's ``if HQ > HK`` guard means this function is **not reached** there (HG stays 1).
    The MHA bwd is already gfx950-optimal on gfx1250 (``ab_sweep/opt7e`` -- every alt config
    regresses), so there is no bwd retune for V4-Pro; this default only matters for a true-MQA
    config. The real gfx1250 attention win is the FWD (see ``fwd_attn_defaults``).
    """
    if not (hq >= 64 and hk == 1):
        return 1
    return 32 if is_gfx1250() else 2

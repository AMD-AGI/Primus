###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One-launch builders for the gfx1250 FlyDSL grouped GEMM's per-call integer metadata.

Primus-Turbo's ``build_m_tile_map`` (``sub``, ``add``, ``floor_divide``, ``cumsum``, ``cat``,
``to(int32)``: six launches) and ``build_expert_table`` (``sub``, ``stack``, ``to(int32)``)
run for every grouped GEMM call (four ``build_m_tile_map`` and two ``build_expert_table``
per MoE layer and step). Each becomes one Triton program here with the same integer
results and the same provenance tag, so ``check_prebuilt`` and the kernels see exactly the
tensors Turbo would have built.

``PRIMUS_FUSED_GG_META=0`` keeps Turbo's builders.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PRIMUS_FUSED_GG_META", "1") != "0"


@triton.jit
def _m_tile_map_kernel(OFFS, OUT, G, BM: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    lo = tl.load(OFFS + idx, mask=idx < G, other=0).to(tl.int64)
    hi = tl.load(OFFS + idx + 1, mask=idx < G, other=0).to(tl.int64)
    lens = hi - lo
    blocks = (lens + (BM - 1)) // BM
    incl = tl.cumsum(blocks, axis=0)
    tl.store(OUT + idx, incl.to(tl.int32), mask=idx < G)
    offs = tl.load(OFFS + idx, mask=idx < G + 1, other=0).to(tl.int64)
    tl.store(OUT + G + idx, offs.to(tl.int32), mask=idx < G + 1)


@triton.jit
def _expert_table_kernel(OFFS, MK, OUT, G, HAS_MK: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    m = idx < G
    lo = tl.load(OFFS + idx, mask=m, other=0).to(tl.int64)
    if HAS_MK:
        ln = tl.load(MK + idx, mask=m, other=0).to(tl.int64)
    else:
        ln = tl.load(OFFS + idx + 1, mask=m, other=0).to(tl.int64) - lo
    tl.store(OUT + 2 * idx, lo.to(tl.int32), mask=m)
    tl.store(OUT + 2 * idx + 1, ln.to(tl.int32), mask=m)


def _ok(group_offs) -> bool:
    return (
        group_offs.is_cuda
        and group_offs.dim() == 1
        and group_offs.is_contiguous()
        and group_offs.dtype in (torch.int64, torch.int32)
        and 2 <= group_offs.numel() <= 4097
    )


def install() -> bool:
    if not ENABLED:
        return False
    try:
        from primus_turbo.flydsl.grouped_gemm import grouped_gemm_bf16_kernel_gfx1250 as gg
    except Exception:
        return False
    if getattr(gg.build_m_tile_map, "_primus_fused", False):
        return True
    if not all(hasattr(gg, n) for n in ("build_m_tile_map", "build_expert_table", "_tag_prov", "_prov_key")):
        return False
    orig_map = gg.build_m_tile_map
    orig_tab = gg.build_expert_table

    def build_m_tile_map(group_offs, block_m: int) -> torch.Tensor:
        if not _ok(group_offs) or block_m <= 0:
            return orig_map(group_offs, block_m)
        G = group_offs.numel() - 1
        out = torch.empty(2 * G + 1, dtype=torch.int32, device=group_offs.device)
        _m_tile_map_kernel[(1,)](group_offs, out, G, BM=int(block_m), BLOCK=triton.next_power_of_2(G + 1))
        return gg._tag_prov(out, gg._prov_key(group_offs, block_m, G))

    def build_expert_table(group_offs, masked_k=None) -> torch.Tensor:
        G = group_offs.numel() - 1
        if not _ok(group_offs) or (
            masked_k is not None
            and not (
                masked_k.is_cuda
                and masked_k.numel() == G
                and masked_k.is_contiguous()
                and masked_k.dtype in (torch.int64, torch.int32)
            )
        ):
            return orig_tab(group_offs, masked_k)
        out = torch.empty(G, 2, dtype=torch.int32, device=group_offs.device)
        _expert_table_kernel[(1,)](
            group_offs,
            masked_k if masked_k is not None else group_offs,
            out,
            G,
            HAS_MK=masked_k is not None,
            BLOCK=triton.next_power_of_2(G),
        )
        return gg._tag_prov(out, gg._prov_key(group_offs, 0, G, masked_k))

    build_m_tile_map._primus_fused = True
    build_m_tile_map._primus_orig = orig_map
    build_expert_table._primus_orig = orig_tab
    gg.build_m_tile_map = build_m_tile_map
    gg.build_expert_table = build_expert_table
    return True

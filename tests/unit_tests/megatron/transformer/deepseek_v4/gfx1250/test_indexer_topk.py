###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Indexer row top-k in Triton (``indexer_topk``, ``PRIMUS_V4_INDEXER_TOPK_TRITON=1``) against a reference on
bf16 scores: the sorted values are bit-equal to a descending sort, ties resolve to the lowest column first (as a
stable sort does; ``torch.topk`` leaves their order unspecified), entries scored ``-inf`` are reported as -1.
Covers the bitonic ``tl.topk`` selection and the full-sort fallback, the half-row path (causal rows whose columns
>= k are all ``-inf``), a k that is not a power of two, and rows with ``-inf`` / NaN / ``-0`` / ``+0`` entries.
"""

import pytest
import torch

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


def _scores(R, P, kind, seed):
    torch.manual_seed(seed)
    if kind == "randn":
        return torch.randn(R, P, device=DEV).to(torch.bfloat16)
    if kind == "ties":  # few distinct values: many ties at the k boundary
        return torch.randint(-8, 8, (R, P), device=DEV).float().to(torch.bfloat16) * 0.25
    if kind == "special":  # -inf, NaN, -0 and +0 entries mixed into random rows (every third row mostly -inf)
        s = torch.randn(R, P, device=DEV).to(torch.bfloat16)
        u = torch.rand(R, P, device=DEV)
        dense = (torch.arange(R, device=DEV) % 3 == 0)[:, None]
        s = s.masked_fill(u < torch.where(dense, 0.9, 0.2), float("-inf"))
        s = s.masked_fill((u > 0.97) & (u < 0.98), float("nan"))
        s = s.masked_fill((u > 0.98) & (u < 0.99), -0.0)
        return s.masked_fill(u > 0.99, 0.0)
    # causal -inf masked tail like the indexer (row t sees (t+1)//4 pool entries)
    s = torch.randn(R, P, device=DEV).to(torch.bfloat16)
    vis = (torch.arange(R, device=DEV)[:, None] + 1) // 4
    s = s.masked_fill(torch.arange(P, device=DEV)[None, :] >= vis, float("-inf"))
    s[:, 0] = 0.0
    return s


@pytest.mark.parametrize("kind", ["randn", "ties", "causal", "special"])
@pytest.mark.parametrize(
    "R,P,k",
    [
        (512, 1024, 512),
        (256, 1024, 16),
        (256, 2048, 512),
        (4096, 1024, 512),
        (128, 4096, 512),
        (256, 1024, 300),
    ],
)
@pytest.mark.parametrize("tl_topk", [True, False], ids=["tl_topk", "tl_sort"])
def test_indexer_topk_matches_stable_sort(monkeypatch, kind, R, P, k, tl_topk):
    monkeypatch.setenv("PRIMUS_V4_INDEXER_TOPK_TRITON", "1")
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common import (
        indexer_topk as IT,
    )
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_topk import (
        indexer_topk,
        indexer_topk_supported,
    )

    if tl_topk and not IT._HAS_TL_TOPK:
        pytest.skip("this Triton has no tl.topk")
    monkeypatch.setattr(IT, "_HAS_TL_TOPK", tl_topk)

    s = _scores(R, P, kind, seed=R + P + k)
    assert indexer_topk_supported(s, k)
    # stable descending sort on torch.topk's bf16 radix key (-0 below +0, NaN on top), lower column first on ties
    b16 = s.view(torch.int16).to(torch.int32) & 0xFFFF
    key = torch.where((b16 & 0x8000) != 0, (~b16) & 0xFFFF, b16 | 0x8000)
    key = torch.where(torch.isnan(s), torch.full_like(key, 0xFFFF), key)
    ir = torch.sort(key, dim=-1, descending=True, stable=True).indices[:, :k].contiguous()
    vr = torch.gather(s, -1, ir)
    ir = torch.where(vr == float("-inf"), torch.full_like(ir, -1), ir)
    it, vt = indexer_topk(s, k)
    torch.cuda.synchronize()
    nan = torch.isnan(vr)
    assert torch.equal(nan, torch.isnan(vt)), "NaN positions differ"
    assert bits_equal(vr.masked_fill(nan, 0), vt.masked_fill(nan, 0)), "top-k values differ"
    assert torch.equal(ir, it), f"indices differ in {int((ir != it).any(-1).sum())} rows"


def test_indexer_topk_disabled_by_default_switch(monkeypatch):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_topk import (
        indexer_topk_supported,
    )

    s = _scores(8, 1024, "randn", 0)
    monkeypatch.setenv("PRIMUS_V4_INDEXER_TOPK_TRITON", "0")
    assert not indexer_topk_supported(s, 16)

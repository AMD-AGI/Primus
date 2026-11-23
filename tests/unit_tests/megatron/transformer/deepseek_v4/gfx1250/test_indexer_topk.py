###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Indexer row top-k in Triton (``indexer_topk``, ``PRIMUS_V4_INDEXER_TOPK_TRITON=1``) against a reference on
bf16 scores: the sorted values are bit-equal to a descending sort, ties resolve to the lowest column first (as a
stable sort does; ``torch.topk`` leaves their order unspecified), entries scored ``-inf`` are reported as -1.
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
    # causal -inf masked tail like the indexer (row t sees (t+1)//4 pool entries)
    s = torch.randn(R, P, device=DEV).to(torch.bfloat16)
    vis = (torch.arange(R, device=DEV)[:, None] + 1) // 4
    s = s.masked_fill(torch.arange(P, device=DEV)[None, :] >= vis, float("-inf"))
    s[:, 0] = 0.0
    return s


@pytest.mark.parametrize("kind", ["randn", "ties", "causal"])
@pytest.mark.parametrize("R,P,k", [(512, 1024, 512), (256, 1024, 16), (256, 2048, 512)])
def test_indexer_topk_matches_stable_sort(monkeypatch, kind, R, P, k):
    monkeypatch.setenv("PRIMUS_V4_INDEXER_TOPK_TRITON", "1")
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_topk import (
        indexer_topk,
        indexer_topk_supported,
    )

    s = _scores(R, P, kind, seed=R + P + k)
    assert indexer_topk_supported(s, k)
    vr, ir = torch.sort(s, dim=-1, descending=True, stable=True)
    vr, ir = vr[:, :k].contiguous(), ir[:, :k].contiguous()
    ir = torch.where(vr == float("-inf"), torch.full_like(ir, -1), ir)
    it, vt = indexer_topk(s, k)
    torch.cuda.synchronize()
    assert bits_equal(vr, vt), "top-k values differ"
    assert torch.equal(ir, it), f"indices differ in {int((ir != it).any(-1).sum())} rows"


def test_indexer_topk_disabled_by_default_switch(monkeypatch):
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_topk import (
        indexer_topk_supported,
    )

    s = _scores(8, 1024, "randn", 0)
    monkeypatch.setenv("PRIMUS_V4_INDEXER_TOPK_TRITON", "0")
    assert not indexer_topk_supported(s, 16)

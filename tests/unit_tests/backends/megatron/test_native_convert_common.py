###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared native-converter helpers (native_convert_common): used by the Llama, Qwen3, DeepSeek and
OLMo-3 converters."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _unique(rows, hidden, base):
    """Every element distinct: element (r, c) is base + r * 1000 + c, so a reshape or transpose
    along either axis shows."""
    r = torch.arange(rows, dtype=torch.float64).unsqueeze(1)
    c = torch.arange(hidden, dtype=torch.float64).unsqueeze(0)
    return base + r * 1000 + c


@pytest.mark.parametrize("nh, ng, hdim, hidden", [(8, 2, 4, 16), (4, 4, 8, 32), (6, 1, 2, 12), (8, 2, 4, 24)])
def test_fuse_qkv_golden_row_order(nh, ng, hdim, hidden):
    """Golden layout, written row by row (no reshape in the oracle): per query group g, the rows of
    its nh / ng query heads, then key head g, then value head g; each row is copied unchanged."""
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    q = _unique(nh * hdim, hidden, 1e7)
    k = _unique(ng * hdim, hidden, 2e7)
    v = _unique(ng * hdim, hidden, 3e7)
    fused = fuse_qkv(q, k, v, heads=nh, groups=ng, head_dim=hdim)
    per = nh // ng
    golden = []
    for g in range(ng):
        golden += [q[r] for r in range(g * per * hdim, (g + 1) * per * hdim)]
        golden += [k[r] for r in range(g * hdim, (g + 1) * hdim)]
        golden += [v[r] for r in range(g * hdim, (g + 1) * hdim)]
    assert torch.equal(fused, torch.stack(golden))


def test_fuse_qkv_is_the_inkling_mapping_helper(monkeypatch):
    """One implementation (core/models/common/qkv_layout.py), used by the converters and by
    inkling_hf_mapping; it round-trips through inkling's split_qkv."""
    import importlib.util
    import sys

    from primus.backends.megatron.checkpoint import native_convert_common as common

    # loaded by file path: the inkling package __init__ imports megatron.training (needs a GPU here)
    path = Path(common.__file__).resolve().parents[1] / "core/models/inkling/inkling_hf_mapping.py"
    spec = importlib.util.spec_from_file_location("_inkling_hf_mapping_by_path", path)
    mapping = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mapping)  # its dataclasses look the module up
    spec.loader.exec_module(mapping)
    from primus.backends.megatron.core.models.common.qkv_layout import (
        fuse_qkv as layout_fuse_qkv,
    )

    assert mapping.fuse_qkv is common.fuse_qkv is layout_fuse_qkv  # one implementation
    q, k, v = _unique(12, 5, 0), _unique(6, 5, 1e6), _unique(6, 5, 2e6)
    fused = mapping.fuse_qkv(q, k, v, heads=4, groups=2, head_dim=3)
    q2, k2, v2 = mapping.split_qkv(fused, heads=4, groups=2, head_dim=3)  # inkling's inverse
    assert torch.equal(q2, q) and torch.equal(k2, k) and torch.equal(v2, v)


@pytest.mark.parametrize("nh, ng", [(6, 4), (4, 0), (0, 2), (4, -2), (8, 3)])
def test_fuse_qkv_refuses_head_counts(nh, ng):
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    q, k, v = _unique(8, 8, 0), _unique(4, 8, 0), _unique(4, 8, 0)
    with pytest.raises(ValueError, match="positive multiple"):
        fuse_qkv(q, k, v, heads=nh, groups=ng, head_dim=2)


def test_fuse_qkv_refuses_a_transposed_or_mis_sized_weight():
    """Non-square GQA weights (q [32, 24], k / v [8, 24]): a transposed or wrongly sized weight is
    refused with the expected shape, not reshaped into scrambled rows."""
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    nh, ng, hdim, hidden = 8, 2, 4, 24
    q, k, v = _unique(nh * hdim, hidden, 0), _unique(ng * hdim, hidden, 0), _unique(ng * hdim, hidden, 0)
    fuse_qkv(q, k, v, heads=nh, groups=ng, head_dim=hdim)
    for bad in ((q.t(), k, v), (q, k.t(), v), (q, k, v.t()), (q[:, :-1], k, v), (q, k[:-1], v)):
        with pytest.raises(ValueError, match="expected"):
            fuse_qkv(*bad, heads=nh, groups=ng, head_dim=hdim)
    with pytest.raises(TypeError):
        fuse_qkv(q, k, v, nh, ng, hdim)  # keyword-only

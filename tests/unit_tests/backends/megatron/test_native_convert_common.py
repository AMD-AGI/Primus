###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""native_convert_common.fuse_qkv: the mcore linear_qkv layout shared by the Llama, Qwen3 and
OLMo-3 converters."""

import pytest

torch = pytest.importorskip("torch")


def _labelled(nh, ng, hdim, hidden):
    """q / k / v weights whose every value names its row: q row r is r, k row r is 1000 + r,
    v row r is 2000 + r (constant along the hidden dimension)."""

    def rows(n, base):
        return (base + torch.arange(n, dtype=torch.float64)).unsqueeze(1).expand(n, hidden).clone()

    return rows(nh * hdim, 0), rows(ng * hdim, 1000), rows(ng * hdim, 2000)


@pytest.mark.parametrize("nh, ng, hdim, hidden", [(8, 2, 4, 16), (4, 4, 8, 32), (6, 1, 2, 12), (6, 3, 4, 10)])
def test_fuse_qkv_groups_query_heads_then_key_then_value(nh, ng, hdim, hidden):
    """The fused rows, read back by their labels: group g holds the query rows of its nh / ng heads,
    then its key rows, then its value rows (an independent oracle, no reshape)."""
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    q, k, v = _labelled(nh, ng, hdim, hidden)
    fused = fuse_qkv(q, k, v, nh, ng, hdim, hidden)
    assert fused.shape == ((nh + 2 * ng) * hdim, hidden)
    assert torch.equal(fused, fused[:, :1].expand_as(fused))  # rows are kept whole
    per = nh // ng
    want = []
    for grp in range(ng):
        want += list(range(grp * per * hdim, (grp + 1) * per * hdim))
        want += [1000 + r for r in range(grp * hdim, (grp + 1) * hdim)]
        want += [2000 + r for r in range(grp * hdim, (grp + 1) * hdim)]
    assert fused[:, 0].long().tolist() == want


@pytest.mark.parametrize("nh, ng", [(6, 4), (4, 0), (0, 2), (4, -2)])
def test_fuse_qkv_refuses_head_counts(nh, ng):
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    q, k, v = _labelled(4, 2, 2, 8)
    with pytest.raises(ValueError, match="positive multiple"):
        fuse_qkv(q, k, v, nh, ng, 2, 8)


def test_fuse_qkv_refuses_a_transposed_or_mis_sized_weight():
    """GQA with non-square weights (nh * hdim != hidden, ng * hdim != hidden): a weight in the
    nn.Linear [in, out] orientation (a real transpose) or with a wrong size is refused with the
    expected shape, not reshaped into scrambled rows."""
    from primus.backends.megatron.checkpoint.native_convert_common import fuse_qkv

    nh, ng, hdim, hidden = 8, 2, 4, 24  # q [32, 24], k / v [8, 24]
    q, k, v = _labelled(nh, ng, hdim, hidden)
    fuse_qkv(q, k, v, nh, ng, hdim, hidden)
    for bad in ((q.t(), k, v), (q, k.t(), v), (q, k, v.t()), (q[:, :-1], k, v), (q, k[:-1], v)):
        with pytest.raises(ValueError, match="expected"):
            fuse_qkv(*bad, nh, ng, hdim, hidden)

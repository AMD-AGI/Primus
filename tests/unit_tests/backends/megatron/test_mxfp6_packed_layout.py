"""Bucket layout for the MXFP6 packed parameter all-gather (packed_param_gather_patches): the planner keeps
Megatron's bucket membership, splits each bucket into R / W4 / W6 sub-buckets, and puts every shard boundary inside a
weight on a 32-row edge."""

import math
import random

import pytest

from primus.backends.megatron.patches.packed_param_gather_patches import (
    check_layout,
    pack_kind,
    plan_layout,
    plan_weight_bucket,
)


def _megatron_layout(numels, bucket_size, dp):
    """Megatron's own pass (distributed optimizer, no shared embedding): bucket (start, end) list."""
    lcm = math.lcm(dp, 128)
    pad = lambda x, d: -(-x // d) * d  # noqa: E731
    out, pstart, bstart = [], 0, 0
    for n in numels[::-1]:
        pstart = pad(pstart, 64)
        pend = pstart + n
        if (pend - bstart) >= bucket_size:
            end = pad(pend, lcm)
            out.append((bstart, end))
            bstart = pstart = end
        else:
            pstart = pend
    if pstart > bstart:
        out.append((bstart, pad(pstart, lcm)))
    return out


def _flux_like(n_layers=12, seed=0):
    rnd = random.Random(seed)
    shapes, names = [], []
    for layer in range(n_layers):
        for what, s in (("self_attention.linear_qkv", (9216, 3072)), ("self_attention.linear_proj", (3072, 3072)),
                        ("mlp.linear_fc1", (12288, 3072)), ("mlp.linear_fc2", (3072, 12288))):
            shapes.append(s)
            names.append(f"module.transformer.layers.{layer}.{what}.weight")
            shapes.append((s[0], 1))
            names.append(f"module.transformer.layers.{layer}.{what}.bias")
        shapes.append((rnd.choice([3072 * 9, 3072 * 6]), 3072))
        names.append(f"module.transformer.layers.{layer}.adaln.adaLN_modulation.1.weight")
    return shapes, names


@pytest.mark.parametrize("dp", [8, 32])
def test_no_weights_matches_megatron(dp):
    shapes, _ = _flux_like()
    numels = [r * k for r, k in shapes]
    _, buckets = plan_layout(list(range(len(shapes))), numels, [None] * len(shapes), shapes, 200_000_000, dp)
    assert [(s, e) for s, e, *_ in buckets] == _megatron_layout(numels, 200_000_000, dp)


@pytest.mark.parametrize("dp", [8, 32])
@pytest.mark.parametrize("fc1_fp4", [False, True])
def test_weight_cuts_on_row_edges(dp, fc1_fp4):
    shapes, names = _flux_like()
    numels = [r * k for r, k in shapes]
    kinds = [pack_kind(n, 6, fc1_fp4, False, False) for n in names]
    index_map, buckets = plan_layout(list(range(len(shapes))), numels, kinds, shapes, 200_000_000, dp)
    assert check_layout(index_map, buckets, shapes, kinds, dp)
    assert sorted(index_map) == list(range(len(shapes)))
    # membership: the sub-buckets of one group hold exactly Megatron's bucket's params
    ref = _megatron_layout(numels, 200_000_000, dp)
    assert len({g for *_, g, _S, _u in buckets}) == len(ref)
    total = sum(numels)
    pad = sum(e - s for s, e, *_ in buckets) - total
    assert pad / total < 0.03, pad / total


def test_pack_kind():
    assert pack_kind("module.transformer.layers.3.mlp.linear_fc1.weight", 19, True, False, False) == "W6"
    assert pack_kind("module.transformer.layers.30.mlp.linear_fc1.weight", 19, True, False, False) == "W4"
    assert pack_kind("module.transformer.layers.30.mlp.linear_fc1.weight", 19, False, False, False) == "W6"
    assert pack_kind("module.transformer.layers.30.mlp.linear_fc1.bias", 19, True, False, False) is None
    assert pack_kind("module.transformer.layers.2.self_attention.added_linear_qkv.weight", 19, 1, 0, 0) == "W6"


def test_plan_weight_bucket_simple():
    S, starts, pad = plan_weight_bucket([(12288, 3072), (3072, 12288)], 8)
    assert S % 128 == 0 and pad >= 0 and starts[0] == 0

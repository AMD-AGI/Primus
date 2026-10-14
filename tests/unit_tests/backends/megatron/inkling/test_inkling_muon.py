###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muon's q/k/v split on Inkling, whose sliding and global layers differ in KV groups.

Newton-Schulz is swapped for a recorder, so these tests check exactly which
blocks Megatron's ``TensorParallelMuon.orthogonalize`` hands to it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.models.inkling.inkling_builders import (
    build_inkling_model,
    inkling_config_from_model_yaml,
)
from primus.backends.megatron.patches.inkling_muon_qkv_patches import (
    QKV_SPLIT_SHAPES_ATTR,
    layer_qkv_split_shapes,
    make_orthogonalize,
    propagate_to_master_weights,
    tag_qkv_split_shapes,
)
from tests.unit_tests.backends.megatron.inkling.inkling_test_utils import (
    RELEASE_VOCAB,
    preset,
)

muon = pytest.importorskip("megatron.core.optimizer.muon")
pytestmark = pytest.mark.skipif(not muon.HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers not installed")

HIDDEN = 8  # the split only reads rows; a narrow hidden keeps the gradients small


def _stub_muon(model_split_shapes):
    """``self`` for ``orthogonalize``: Megatron's model-wide split shapes, Newton-Schulz recorded."""
    seen = []

    def record(grad, tp_group, partition_dim):
        seen.append(tuple(grad.shape))
        return grad

    stub = SimpleNamespace(
        pg_collection=None,
        mode="blockwise",
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, "is_qkv", False),
        qkv_split_shapes=list(model_split_shapes),
        scaled_orthogonalize_fn=record,
    )
    return stub, seen


def _qkv_param(groups, shapes):
    p = torch.nn.Parameter(torch.empty(groups * sum(shapes), HIDDEN))
    p.is_qkv = True
    setattr(p, QKV_SPLIT_SHAPES_ATTR, shapes)
    return p


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE modules need a GPU")
def test_large_proxy_layers_get_their_own_split_shapes(init_parallel_state):
    config = inkling_config_from_model_yaml(preset("inkling_large_6L"))
    model = build_inkling_model(config, vocab_size=RELEASE_VOCAB, max_sequence_length=4096, meta=True)
    # sliding: 64 heads / 16 KV groups; global (layer 5): 64 heads / 8 KV groups
    assert tag_qkv_split_shapes(model) == {(512, 128, 128): 5, (1024, 128, 128): 1}
    rows = [layer.self_attention.linear_qkv.weight.shape[0] for layer in model.decoder.layers]
    assert rows == [12288] * 5 + [10240]
    assert layer_qkv_split_shapes(config) == (1024, 128, 128)  # what stock Megatron uses everywhere


def test_sliding_qkv_fails_with_model_shapes_and_splits_with_its_own():
    model_shapes, sliding = (1024, 128, 128), (512, 128, 128)
    p = _qkv_param(16, sliding)
    grad = torch.randn(p.shape)

    stub, _ = _stub_muon(model_shapes)
    with pytest.raises(RuntimeError, match="invalid for input"):
        muon.TensorParallelMuon.orthogonalize(stub, p, grad)

    stub, seen = _stub_muon(model_shapes)
    out = make_orthogonalize(muon.TensorParallelMuon.orthogonalize)(stub, p, grad)
    assert seen == [(16 * 512, HIDDEN), (16 * 128, HIDDEN), (16 * 128, HIDDEN)]
    # identity "Newton-Schulz": split and concat round-trip
    torch.testing.assert_close(out, grad, rtol=0, atol=0)
    assert stub.qkv_split_shapes == list(model_shapes)


def test_split_blocks_are_q_k_v_of_every_group():
    shapes, groups = (512, 128, 128), 16
    p = _qkv_param(groups, shapes)
    kind = torch.tensor([0.0] * 512 + [1.0] * 128 + [2.0] * 128).repeat(groups)
    grad = kind[:, None].expand(-1, HIDDEN).clone()
    blocks = []

    stub, _ = _stub_muon((1024, 128, 128))
    stub.scaled_orthogonalize_fn = lambda g, tp, pd: blocks.append(g.clone()) or g
    make_orthogonalize(muon.TensorParallelMuon.orthogonalize)(stub, p, grad)
    assert [b.unique().tolist() for b in blocks] == [[0.0], [1.0], [2.0]]


def test_untagged_params_take_the_stock_path():
    stub, seen = _stub_muon((512, 128, 128))
    p = torch.nn.Parameter(torch.empty(8 * 768, HIDDEN))
    p.is_qkv = True
    make_orthogonalize(muon.TensorParallelMuon.orthogonalize)(stub, p, torch.randn(p.shape))
    assert seen == [(8 * 512, HIDDEN), (8 * 128, HIDDEN), (8 * 128, HIDDEN)]

    stub, seen = _stub_muon((512, 128, 128))
    dense = torch.nn.Parameter(torch.empty(64, HIDDEN))
    make_orthogonalize(muon.TensorParallelMuon.orthogonalize)(stub, dense, torch.randn(dense.shape))
    assert seen == [(64, HIDDEN)]


def test_tag_reaches_fp32_master_weight():
    model = torch.nn.Linear(HIDDEN, 16 * 768, bias=False)
    setattr(model.weight, QKV_SPLIT_SHAPES_ATTR, (512, 128, 128))
    model.weight.main_param = model.weight.detach().float()
    assert propagate_to_master_weights(model) == 1
    assert getattr(model.weight.main_param, QKV_SPLIT_SHAPES_ATTR) == (512, 128, 128)

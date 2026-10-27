###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark's FLOPs reporting against Megatron's own num_floating_point_operations.

The shape is the released 62-layer text tower: 46 sliding-window layers and 16
full-attention ones (1, 5, ..., 61).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")

from primus.backends.megatron.patches import (  # noqa: E402
    muse_spark_flops_patches as muse_flops,
)

GBS, SEQ = 64, 4096
Q_SIZE = 128 * 64


@pytest.fixture(autouse=True)
def _silence_breakdown_log(monkeypatch):
    monkeypatch.setattr(muse_flops, "_BREAKDOWN_LOGGED", True)


def _upstream():
    from megatron.training.training import num_floating_point_operations

    return num_floating_point_operations


def _args(**overrides):
    args = dict(
        model_type="muse_spark",
        num_layers=62,
        hidden_size=8192,
        ffn_hidden_size=4096,
        num_attention_heads=128,
        group_query_attention=True,
        num_query_groups=16,
        kv_channels=64,
        attention_output_gate=True,
        multi_latent_attention=False,
        experimental_attention_variant=None,
        hybrid_layer_pattern=None,
        num_experts=256,
        moe_layer_freq=1,
        moe_router_topk=8,
        moe_ffn_hidden_size=4096,
        moe_shared_expert_intermediate_size=None,
        moe_latent_size=4096,
        mtp_num_layers=None,
        swiglu=True,
        seq_length=SEQ,
        padded_vocab_size=202112,
        muse_global_attn_interval=4,
        muse_sliding_window=2048,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _brute_keys(S, W):
    return sum(min(t + 1, W) for t in range(S)) / S


def test_keys_per_query():
    assert muse_flops.mean_keys_per_query(SEQ) == (SEQ + 1) / 2
    assert muse_flops.mean_keys_per_query(SEQ, 2048) == _brute_keys(SEQ, 2048) == 1536.25
    assert muse_flops.mean_keys_per_query(1000, 2048) == muse_flops.mean_keys_per_query(1000)
    assert muse_flops.mean_keys_per_query(32768, 2048) == _brute_keys(32768, 2048)


def test_correction_survives_pretrain_rebinding_model_type():
    from megatron.core.enums import ModelType

    wrapped = muse_flops.make_muse_spark_num_floating_point_operations(_upstream())
    expected = muse_flops.compute_muse_spark_flops(_args(), GBS, _upstream()).total
    assert wrapped(_args(model_type=ModelType.encoder_or_decoder), GBS) == expected


def test_only_sliding_window_layers_are_corrected():
    b = muse_flops.compute_muse_spark_flops(_args(), GBS, _upstream())
    assert b.num_swa_layers == 46 and b.keys_per_query == 1536.25
    assert b.upstream == _upstream()(_args(), GBS)
    expected = 3 * 2 * GBS * SEQ * 46 * Q_SIZE * 2 * (1536.25 - (SEQ + 1) / 2)
    assert b.swa_attention == pytest.approx(expected, rel=1e-12)
    assert b.total / b.upstream == pytest.approx(0.9917, abs=1e-3)


def test_window_at_least_seq_is_dense_causal():
    wrapped = muse_flops.make_muse_spark_num_floating_point_operations(_upstream())
    for window in (None, SEQ, 2 * SEQ):
        assert wrapped(_args(muse_sliding_window=window), GBS) == _upstream()(_args(), GBS)


@pytest.mark.parametrize("seq, ratio", [(8192, 0.9642), (32768, 0.8219)])
def test_correction_grows_with_context(seq, ratio):
    b = muse_flops.compute_muse_spark_flops(_args(seq_length=seq), GBS, _upstream())
    assert b.total / b.upstream == pytest.approx(ratio, abs=1e-3)

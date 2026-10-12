###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3's FLOPs reporting against Megatron's own num_floating_point_operations.

The shape is the 8-layer MI355X proxy: 3 dense-attention layers, then 5 MSA
layers (sparse_attention_freq), which are also the 5 MoE layers.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")

from primus.backends.megatron.patches import (  # noqa: E402
    minimax_m3_flops_patches as m3_flops,
)

GBS, SEQ, H = 64, 4096, 6144
TOKENS = GBS * SEQ
Q_SIZE = 64 * 128


@pytest.fixture(autouse=True)
def _silence_breakdown_log(monkeypatch):
    monkeypatch.setattr(m3_flops, "_BREAKDOWN_LOGGED", True)


def _upstream():
    from megatron.training.training import num_floating_point_operations

    return num_floating_point_operations


def _args(**overrides):
    args = dict(
        minimax_sparse_attention=True,
        num_layers=8,
        hidden_size=H,
        ffn_hidden_size=12288,
        num_attention_heads=64,
        group_query_attention=True,
        num_query_groups=4,
        kv_channels=128,
        attention_output_gate=False,
        multi_latent_attention=False,
        experimental_attention_variant=None,
        hybrid_layer_pattern=None,
        num_experts=128,
        moe_layer_freq=[0] * 3 + [1] * 5,
        moe_router_topk=4,
        moe_ffn_hidden_size=3072,
        moe_shared_expert_intermediate_size=3072,
        moe_latent_size=None,
        mtp_num_layers=0,
        swiglu=False,
        quick_geglu=True,
        seq_length=SEQ,
        padded_vocab_size=200064,
        sparse_attention_freq="([0]*3+[1]*5)",
        sparse_block_size=128,
        sparse_topk_blocks=16,
        sparse_num_index_heads=4,
        sparse_index_dim=128,
        sparse_indexer_loss_coeff=1.0e-2,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _brute_keys(S, B, topk):
    return sum((min(t // B + 1, topk) - 1) * B + t % B + 1 for t in range(S)) / S


def test_keys_per_query_counts_each_querys_visible_selection():
    assert m3_flops.msa_keys_per_query(256, 128, 1) == 64.5  # its own block only: mean of 1..128
    assert m3_flops.msa_keys_per_query(300, 128, 16) == 150.5  # every block kept: dense causal
    assert m3_flops.msa_keys_per_query(SEQ, 128, 16) == _brute_keys(SEQ, 128, 16) == 1504.5
    assert m3_flops.msa_keys_per_query(2148, 128, 4) == _brute_keys(2148, 128, 4)  # partial last block


def test_non_m3_args_fall_through():
    sentinel = object()
    wrapped = m3_flops.make_m3_num_floating_point_operations(lambda args, batch_size: sentinel)
    assert wrapped(SimpleNamespace(minimax_sparse_attention=False), GBS) is sentinel


def test_glu_counts_three_gemms():
    """quick_geglu is gate, up and down; upstream only counts three GEMMs for swiglu."""
    b = m3_flops.compute_m3_flops(_args(), GBS, _upstream())
    # one extra h x ffn GEMM per dense MLP, per routed expert and per shared expert
    extra_gemm = H * (12288 * 3 + (3072 * 4 + 3072) * 5)
    assert b.glu == pytest.approx(3 * 2 * TOKENS * extra_gemm, rel=1e-12)

    no_msa = _args(sparse_attention_freq="([0]*8)")
    wrapped = m3_flops.make_m3_num_floating_point_operations(_upstream())
    assert wrapped(no_msa, GBS) == _upstream()(_args(sparse_attention_freq="([0]*8)", swiglu=True), GBS)


def test_only_layers_the_sparse_freq_marks_are_charged_as_msa():
    """Dense layers keep upstream's causal S/2; every MSA layer moves the total by the same amount,
    wherever it sits."""
    wrapped = m3_flops.make_m3_num_floating_point_operations(_upstream())
    total = {k: wrapped(_args(sparse_attention_freq=f"([0]*{8 - k}+[1]*{k})"), GBS) for k in (0, 2, 5)}
    per_layer = (total[2] - total[0]) / 2
    assert total[5] - total[0] == pytest.approx(5 * per_layer, rel=1e-12)
    assert wrapped(_args(sparse_attention_freq=[1, 0] * 4), GBS) == wrapped(
        _args(sparse_attention_freq="([0]*4+[1]*4)"), GBS
    )

    # one MSA layer at 4k: 1504.5 keys per query instead of 2048, plus its indexer
    attention = 3 * 2 * TOKENS * Q_SIZE * 2 * (1504.5 - SEQ / 2)
    indexer = 2 * TOKENS * (H * (4 * 128 + 128) * 2 + 4 * 128 * (SEQ + 1) / 2 + 2 * 4 * 128 * 16)
    assert per_layer == pytest.approx(attention + indexer, rel=1e-12)
    assert per_layer < 0


def test_frozen_indexer_counts_only_its_forward():
    """sparse_indexer_loss_coeff 0.0 leaves the indexer without a gradient; unset means the default 1e-2."""
    wrapped = m3_flops.make_m3_num_floating_point_operations(_upstream())
    trained = wrapped(_args(), GBS)
    frozen = wrapped(_args(sparse_indexer_loss_coeff=0.0), GBS)
    assert trained - frozen == pytest.approx(2 * TOKENS * 5 * (H * 640 + 2 * 4 * 128 * 16), rel=1e-12)
    assert wrapped(_args(sparse_indexer_loss_coeff=None), GBS) == trained


def test_proxy_breakdown():
    b = m3_flops.compute_m3_flops(_args(), GBS, _upstream())
    assert b.num_msa_layers == 5 and b.keys_per_query == 1504.5
    assert b.upstream == _upstream()(_args(), GBS)
    assert b.msa_attention < 0 < b.indexer < b.glu
    # at 4k the GLU undercount outweighs the MSA overcount
    assert b.total / b.upstream == pytest.approx(1.178, abs=2e-3)

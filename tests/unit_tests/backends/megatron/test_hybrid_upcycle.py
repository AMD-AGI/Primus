###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from argparse import Namespace

import torch

from primus.backends.megatron.checkpoint.hybrid_upcycle import (
    UpcycleLayout,
    attention_ratio_from_args,
    build_upcycle_checkpoint,
    count_decoder_layers,
    hybrid_pattern_from_ratio,
    hylo_reinit_q,
    load_checkpoint,
    main,
    resolve_checkpoint_file,
    split_fused_qkv,
    upcycle_state_dict,
    validate_upcycle_pattern,
)


def _dense_layer(index, hidden=4, ffn=8, attn_out=4):
    prefix = f"decoder.layers.{index}"
    return {
        f"{prefix}.input_layernorm.weight": torch.full((hidden,), float(index + 1)),
        f"{prefix}.self_attention.linear_qkv.weight": torch.full((attn_out, hidden), float(index + 1)),
        f"{prefix}.self_attention.linear_proj.weight": torch.full((hidden, hidden), 10.0 + index),
        f"{prefix}.pre_mlp_layernorm.weight": torch.full((hidden,), 20.0 + index),
        f"{prefix}.mlp.linear_fc1.weight": torch.full((ffn, hidden), 30.0 + index),
        f"{prefix}.mlp.linear_fc2.weight": torch.full((hidden, ffn), 40.0 + index),
    }


def _dense_state(num_layers=2, hidden=4, ffn=8, attn_out=4):
    state = {
        "embedding.word_embeddings.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "output_layer.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2) + 1,
        "decoder.final_layernorm.weight": torch.ones(hidden),
    }
    for index in range(num_layers):
        state.update(_dense_layer(index, hidden, ffn, attn_out))
    return state


def _hybrid_state(pattern, hidden=4, ffn=8, attn_out=4):
    state = {
        "embedding.word_embeddings.weight": torch.zeros(3, 2),
        "output_layer.weight": torch.zeros(3, 2),
        "decoder.final_norm.weight": torch.zeros(hidden),
    }
    for index, symbol in enumerate(pattern):
        prefix = f"decoder.layers.{index}"
        if symbol == "-":
            state[f"{prefix}.pre_mlp_layernorm.weight"] = torch.zeros(hidden)
            state[f"{prefix}.mlp.linear_fc1.weight"] = torch.zeros(ffn, hidden)
            state[f"{prefix}.mlp.linear_fc2.weight"] = torch.zeros(hidden, ffn)
        elif symbol == "*":
            state[f"{prefix}.input_layernorm.weight"] = torch.zeros(hidden)
            state[f"{prefix}.self_attention.linear_qkv.weight"] = torch.zeros(attn_out, hidden)
            state[f"{prefix}.self_attention.linear_proj.weight"] = torch.zeros(hidden, hidden)
        else:
            state[f"{prefix}.mixer.in_proj.weight"] = torch.full((3, hidden), -1.0)
            state[f"{prefix}.mixer.A_log"] = torch.full((2,), -2.0)
    return state


def test_pattern_from_ratio_matches_hybrid_stack_allocation():
    assert hybrid_pattern_from_ratio(8, 0.0) == "M-M-M-M-"
    assert hybrid_pattern_from_ratio(8, 0.25) == "*-M-M-M-"
    assert hybrid_pattern_from_ratio(8, 0.5) == "*-M-*-M-"
    assert hybrid_pattern_from_ratio(8, 1.0) == "*-*-*-*-"
    assert hybrid_pattern_from_ratio(32, 0.25) == "*-M-M-M-" * 4


def test_validate_pattern_rejects_moe_and_unpaired_layouts():
    try:
        validate_upcycle_pattern("E-", 1)
        raise AssertionError("expected MoE pattern to fail")
    except ValueError as exc:
        assert "E" in str(exc)
    try:
        validate_upcycle_pattern("M*", 1)
        raise AssertionError("expected unpaired pattern to fail")
    except ValueError as exc:
        assert "pattern[1]" in str(exc)


def test_upcycle_copies_mlp_and_leaves_mixer_init():
    pattern = "*-M-"
    dense = _dense_state()
    hybrid = _hybrid_state(pattern)
    upcycled, report = upcycle_state_dict(dense, hybrid, pattern)

    assert torch.equal(
        upcycled["embedding.word_embeddings.weight"], dense["embedding.word_embeddings.weight"]
    )
    assert torch.equal(upcycled["decoder.final_norm.weight"], dense["decoder.final_layernorm.weight"])
    assert torch.equal(
        upcycled["decoder.layers.1.mlp.linear_fc1.weight"],
        dense["decoder.layers.0.mlp.linear_fc1.weight"],
    )
    assert torch.equal(
        upcycled["decoder.layers.3.mlp.linear_fc2.weight"],
        dense["decoder.layers.1.mlp.linear_fc2.weight"],
    )
    assert torch.equal(
        upcycled["decoder.layers.0.self_attention.linear_qkv.weight"],
        dense["decoder.layers.0.self_attention.linear_qkv.weight"],
    )
    assert torch.equal(
        upcycled["decoder.layers.2.mixer.in_proj.weight"],
        hybrid["decoder.layers.2.mixer.in_proj.weight"],
    )
    assert "decoder.layers.0.self_attention.linear_qkv.weight" not in report.attention_shape_mismatch
    assert report.mixer_layers_left_initialized == [2]
    assert not any("mixer" in key for key in report.copied)


def test_attention_shape_mismatch_stays_at_hybrid_init():
    pattern = "*-"
    dense = _dense_state(num_layers=1, attn_out=4)
    hybrid = _hybrid_state(pattern, attn_out=7)
    hybrid["decoder.layers.0.self_attention.linear_q_down_proj.weight"] = torch.zeros(2, 4)
    upcycled, report = upcycle_state_dict(dense, hybrid, pattern)

    assert torch.equal(
        upcycled["decoder.layers.0.self_attention.linear_qkv.weight"],
        hybrid["decoder.layers.0.self_attention.linear_qkv.weight"],
    )
    assert "decoder.layers.0.self_attention.linear_qkv.weight" in report.attention_shape_mismatch
    assert "decoder.layers.0.self_attention.linear_q_down_proj.weight" in report.attention_left_initialized
    assert torch.equal(
        upcycled["decoder.layers.0.self_attention.linear_proj.weight"],
        dense["decoder.layers.0.self_attention.linear_proj.weight"],
    )


def test_fused_mlp_norm_maps_onto_separate_pre_mlp_norm():
    pattern = "M-"
    dense = _dense_state(num_layers=1)
    fused_norm = dense.pop("decoder.layers.0.pre_mlp_layernorm.weight")
    dense["decoder.layers.0.mlp.linear_fc1.layer_norm_weight"] = fused_norm
    hybrid = _hybrid_state(pattern)
    upcycled, report = upcycle_state_dict(dense, hybrid, pattern)
    assert torch.equal(upcycled["decoder.layers.1.pre_mlp_layernorm.weight"], fused_norm)
    assert "decoder.layers.1.pre_mlp_layernorm.weight" in report.copied


def test_checkpoint_round_trip_resets_iteration_and_drops_optimizer(tmp_path):
    pattern = "M-"
    dense = _dense_state(num_layers=1)
    hybrid = _hybrid_state(pattern)
    hybrid_ckpt = {
        "iteration": 7,
        "checkpoint_version": 3.0,
        "model": hybrid,
        "optimizer": {"step": 7},
        "args": Namespace(iteration=7, hybrid_override_pattern=pattern, hybrid_attention_ratio=0.0),
    }
    dense_dir = tmp_path / "dense"
    hybrid_dir = tmp_path / "hybrid"
    for directory, payload in ((dense_dir, {"iteration": 4, "model": dense}), (hybrid_dir, hybrid_ckpt)):
        rank = directory / "iter_0000004" / "mp_rank_00"
        rank.mkdir(parents=True)
        torch.save(payload, rank / "model_optim_rng.pt")
        (directory / "latest_checkpointed_iteration.txt").write_text("4\n")

    output = tmp_path / "upcycled"
    report = main(
        [
            "--dense-checkpoint",
            str(dense_dir),
            "--hybrid-init-checkpoint",
            str(hybrid_dir),
            "--output-dir",
            str(output),
        ]
    )
    assert report.pattern == pattern
    assert report.mixer_layers_left_initialized == [0]
    loaded_state, loaded = load_checkpoint(output)
    assert loaded["iteration"] == 0
    assert "optimizer" not in loaded
    assert (output / "latest_checkpointed_iteration.txt").read_text() == "0\n"
    assert torch.equal(
        loaded_state["decoder.layers.1.mlp.linear_fc1.weight"],
        dense["decoder.layers.0.mlp.linear_fc1.weight"],
    )
    assert resolve_checkpoint_file(output).name == "model_optim_rng.pt"
    assert count_decoder_layers(loaded_state) == 2
    assert attention_ratio_from_args(loaded["args"]) == 0.0

    envelope = build_upcycle_checkpoint(hybrid_ckpt, loaded_state)
    assert envelope["iteration"] == 0
    assert "rng_state" not in envelope


def _fuse_qkv(q, k, v, num_heads, num_kv_heads, head_dim):
    heads_per_group = num_heads // num_kv_heads
    hidden = q.shape[1]
    q = q.view(num_kv_heads, heads_per_group * head_dim, hidden)
    k = k.view(num_kv_heads, head_dim, hidden)
    v = v.view(num_kv_heads, head_dim, hidden)
    return torch.cat((q, k, v), dim=1).reshape(-1, hidden)


def test_gdn_mixer_copies_hylo_qkv_slices():
    hidden, heads, kv_heads, head_dim = 4, 2, 2, 2
    key_dim, value_dim, value_heads = 4, 4, 2
    q = torch.arange(heads * head_dim * hidden, dtype=torch.float32).reshape(heads * head_dim, hidden)
    k = torch.arange(kv_heads * head_dim * hidden, dtype=torch.float32).reshape(kv_heads * head_dim, hidden) + 3
    v = torch.arange(kv_heads * head_dim * hidden, dtype=torch.float32).reshape(kv_heads * head_dim, hidden) + 5
    dense = _dense_state(num_layers=1, hidden=hidden, ffn=8, attn_out=1)
    dense["decoder.layers.0.self_attention.linear_qkv.weight"] = _fuse_qkv(q, k, v, heads, kv_heads, head_dim)
    dense["decoder.layers.0.self_attention.linear_proj.weight"] = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    in_rows = key_dim * 2 + value_dim * 2 + value_heads * 2
    hybrid = _hybrid_state("M-", hidden=hidden, ffn=8)
    hybrid["decoder.layers.0.mixer.in_proj.weight"] = torch.full((in_rows, hidden), -1.0)
    hybrid["decoder.layers.0.mixer.out_proj.weight"] = torch.zeros(hidden, value_dim)
    hybrid["decoder.layers.0.mixer.A_log"] = torch.full((2,), -2.0)
    layout = UpcycleLayout(
        num_attention_heads=heads,
        num_query_groups=kv_heads,
        head_dim=head_dim,
        linear_type="gdn",
        gdn_num_key_heads=2,
        gdn_key_head_dim=2,
        gdn_num_value_heads=value_heads,
        gdn_value_head_dim=2,
    )
    upcycled, report = upcycle_state_dict(dense, hybrid, "M-", layout)
    copied = upcycled["decoder.layers.0.mixer.in_proj.weight"]
    split_q, split_k, split_v = split_fused_qkv(
        dense["decoder.layers.0.self_attention.linear_qkv.weight"], heads, kv_heads, head_dim
    )
    assert torch.equal(copied[:key_dim], split_q[:key_dim])
    assert torch.equal(copied[key_dim : key_dim * 2], split_k[:key_dim])
    assert torch.equal(copied[key_dim * 2 : key_dim * 2 + value_dim], split_v[:value_dim])
    assert torch.equal(copied[key_dim * 2 + value_dim :], hybrid["decoder.layers.0.mixer.in_proj.weight"][key_dim * 2 + value_dim :])
    assert torch.equal(upcycled["decoder.layers.0.mixer.A_log"], hybrid["decoder.layers.0.mixer.A_log"])
    assert torch.equal(
        upcycled["decoder.layers.0.mixer.out_proj.weight"],
        dense["decoder.layers.0.self_attention.linear_proj.weight"],
    )
    assert report.mixer_layers_left_initialized == []


def test_mla_slot_uses_hylo_svd_and_keeps_rope_rows():
    hidden, heads, kv_heads, head_dim = 4, 2, 2, 2
    nope, rope, v_head, q_rank, kv_rank = 1, 1, 1, 2, 2
    q = torch.arange(heads * head_dim * hidden, dtype=torch.float32).reshape(heads * head_dim, hidden) + 1
    k = torch.arange(kv_heads * head_dim * hidden, dtype=torch.float32).reshape(kv_heads * head_dim, hidden) + 2
    v = torch.arange(kv_heads * head_dim * hidden, dtype=torch.float32).reshape(kv_heads * head_dim, hidden) + 3
    dense = _dense_state(num_layers=1, hidden=hidden, ffn=8)
    dense["decoder.layers.0.self_attention.linear_qkv.weight"] = _fuse_qkv(q, k, v, heads, kv_heads, head_dim)
    dense["decoder.layers.0.self_attention.linear_proj.weight"] = torch.arange(16, dtype=torch.float32).reshape(hidden, hidden)

    hybrid = {
        "embedding.word_embeddings.weight": torch.zeros(3, 2),
        "decoder.final_norm.weight": torch.zeros(hidden),
        "decoder.layers.0.input_layernorm.weight": torch.zeros(hidden),
        "decoder.layers.0.self_attention.linear_q_down_proj.weight": torch.zeros(q_rank, hidden),
        "decoder.layers.0.self_attention.linear_q_up_proj.weight": torch.zeros(heads * (nope + rope), q_rank),
        "decoder.layers.0.self_attention.linear_kv_down_proj.weight": torch.zeros(kv_rank + rope, hidden),
        "decoder.layers.0.self_attention.linear_kv_up_proj.weight": torch.zeros(heads * (nope + v_head), kv_rank),
        "decoder.layers.0.self_attention.linear_proj.weight": torch.zeros(hidden, heads * v_head),
        "decoder.layers.1.pre_mlp_layernorm.weight": torch.zeros(hidden),
        "decoder.layers.1.mlp.linear_fc1.weight": torch.zeros(8, hidden),
        "decoder.layers.1.mlp.linear_fc2.weight": torch.zeros(hidden, 8),
    }
    layout = UpcycleLayout(
        num_attention_heads=heads,
        num_query_groups=kv_heads,
        head_dim=head_dim,
        q_lora_rank=q_rank,
        kv_lora_rank=kv_rank,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=v_head,
    )
    upcycled, report = upcycle_state_dict(dense, hybrid, "*-", layout)
    split_q, _, _ = split_fused_qkv(dense["decoder.layers.0.self_attention.linear_qkv.weight"], heads, kv_heads, head_dim)
    q_down, q_up_nope = hylo_reinit_q(split_q, q_rank, heads, head_dim, nope)
    assert torch.equal(upcycled["decoder.layers.0.self_attention.linear_q_down_proj.weight"], q_down)
    q_up = upcycled["decoder.layers.0.self_attention.linear_q_up_proj.weight"].view(heads, nope + rope, q_rank)
    assert torch.equal(q_up[:, :nope, :].reshape(-1, q_rank), q_up_nope)
    assert torch.equal(q_up[:, nope:, :], torch.zeros(heads, rope, q_rank))
    assert "decoder.layers.0.self_attention.linear_q_up_proj.weight" in report.hylo_partial
    assert "decoder.layers.0.self_attention.linear_kv_down_proj.weight" in report.hylo_partial
    assert torch.equal(
        upcycled["decoder.layers.0.self_attention.linear_proj.weight"],
        dense["decoder.layers.0.self_attention.linear_proj.weight"][:, : heads * v_head],
    )

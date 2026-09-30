###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from argparse import Namespace

import torch

from primus.backends.megatron.checkpoint.hybrid_upcycle import (
    attention_ratio_from_args,
    build_upcycle_checkpoint,
    count_decoder_layers,
    hybrid_pattern_from_ratio,
    load_checkpoint,
    main,
    resolve_checkpoint_file,
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

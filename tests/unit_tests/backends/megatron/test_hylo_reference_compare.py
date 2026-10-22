###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from argparse import Namespace

import pytest
import torch

pytest.importorskip("safetensors")

from safetensors.torch import save_file

from primus.backends.megatron.checkpoint import hylo_reference_compare as cmp

D_MODEL, D_INNER, D_XB, NHEADS, FFN, VOCAB, LAYERS = 8, 8, 4, 2, 16, 10, 2


def _m2_student(seed=0):
    gen = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=gen)

    state = {
        "embedding.word_embeddings.weight": r(VOCAB, D_MODEL),
        "decoder.final_norm.weight": r(D_MODEL),
        "output_layer.weight": r(VOCAB, D_MODEL),
    }
    for i in range(LAYERS):
        a, m = f"decoder.layers.{2 * i}.mixer.", f"decoder.layers.{2 * i + 1}.mlp."
        state[a + "in_proj.weight"] = r(2 * D_INNER + 2 * D_XB + NHEADS, D_MODEL)
        state[a + "in_proj.layer_norm_weight"] = r(D_MODEL)
        state[a + "conv1d.weight"] = r(D_INNER + 2 * D_XB, 1, 4)
        state[a + "conv1d.bias"] = r(D_INNER + 2 * D_XB)
        state[a + "A_log"] = r(NHEADS)
        state[a + "D"] = r(NHEADS)
        state[a + "dt_bias"] = r(NHEADS)
        state[a + "norm.weight"] = r(D_INNER)
        state[a + "out_proj.weight"] = r(D_MODEL, D_INNER)
        state[m + "linear_fc1.weight"] = r(2 * FFN, D_MODEL)
        state[m + "linear_fc1.layer_norm_weight"] = r(D_MODEL)
        state[m + "linear_fc2.weight"] = r(D_MODEL, FFN)
    return state


def _save(tmp_path, name, state, args=None):
    root = tmp_path / name
    (root / "iter_0000000" / "mp_rank_00").mkdir(parents=True)
    torch.save(
        {"model": state, "iteration": 0, "args": args},
        root / "iter_0000000" / "mp_rank_00" / "model_optim_rng.pt",
    )
    (root / "latest_checkpointed_iteration.txt").write_text("0")
    return root


def test_m2_layout_and_every_tensor_maps_to_a_reference_name():
    state = _m2_student()
    layout = cmp.student_layout(state, "m2")
    assert (layout.num_layers, layout.ffn) == (LAYERS, FFN)
    assert layout.m2_in == {"z": (0, 8), "x": (8, 12), "B": (12, 16), "C": (16, 24), "dt": (24, 26)}
    assert layout.m2_conv == {"x": (0, 4), "B": (4, 8), "C": (8, 16)}
    named, left = cmp.to_reference_names(state, layout)
    assert left == []
    assert torch.equal(
        named["model.layers.1.mlp.up_proj.weight"], state["decoder.layers.3.mlp.linear_fc1.weight"][FFN:]
    )
    assert torch.equal(named["model.layers.0.mamba.dt_bias"], state["decoder.layers.0.mixer.dt_bias"])
    sections = cmp.derive(named, layout)
    assert sections["model.layers.0.mamba.in_proj.weight[B]"].shape == (D_XB, D_MODEL)


def test_reference_written_as_megatron_round_trips(tmp_path):
    student = _m2_student(0)
    layout = cmp.student_layout(student, "m2")
    reference, _ = cmp.to_reference_names(_m2_student(1), layout)
    ref_path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in reference.items()}, str(ref_path))
    template = _save(tmp_path, "template", student, Namespace(iteration=0))

    count = cmp.write_reference_as_megatron(template, ref_path, "m2", tmp_path / "out")
    written, _, _, _ = cmp.load_student(tmp_path / "out")
    named, _ = cmp.to_reference_names(written, layout)
    assert count == len(student)
    for name, value in reference.items():
        assert torch.equal(named[name], value), name


def test_reference_conversion_rejects_unmapped_student_tensors(tmp_path):
    student = _m2_student(0)
    layout = cmp.student_layout(student, "m2")
    reference, _ = cmp.to_reference_names(student, layout)
    ref_path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in reference.items()}, str(ref_path))
    student["decoder.layers.0.mixer.extra.weight"] = torch.zeros(2)
    template = _save(tmp_path, "template", student)
    with pytest.raises(ValueError, match="without a reference value"):
        cmp.write_reference_as_megatron(template, ref_path, "m2", tmp_path / "out")


def test_compare_scores_identical_updates_and_keeps_random_draws_apart():
    layout = cmp.student_layout(_m2_student(), "m2")
    init, _ = cmp.to_reference_names(_m2_student(0), layout)
    step, _ = cmp.to_reference_names(_m2_student(2), layout)
    ours = {k: init[k] + 0.1 * step[k] for k in init}
    ref = {k: init[k] + 0.1 * step[k] for k in init}
    for t in (ours, init, ref):
        t.update(cmp.derive(t, layout))
    rows = cmp.compare(ours, init, ref, {}, cmp.random_init_names(shared=False))
    out_proj = rows["model.layers.0.mamba.out_proj.weight"]
    assert out_proj["cos"] == pytest.approx(1.0) and out_proj["rel"] == pytest.approx(0.0)
    assert out_proj["upd_cos"] == pytest.approx(1.0) and out_proj["upd_ratio"] == pytest.approx(1.0)
    assert rows["model.layers.0.mamba.A_log"]["random_init"] is True
    assert (
        "cos"
        in cmp.compare(ours, init, ref, {}, cmp.random_init_names(shared=True))["model.layers.0.mamba.A_log"]
    )
    summary = cmp.summarize(rows)
    assert summary["mamba.out_proj.weight"]["n"] == LAYERS


def test_mla_factors_are_aligned_to_the_reference_signs():
    gen = torch.Generator().manual_seed(3)
    rank, rope = 4, 2
    layout = cmp.Layout(kind="mla", num_layers=1, ffn=FFN, mla_rank=rank, mla_rope=rope, mla_nope=2)
    h = "model.layers.0.mla."
    ref = {
        h + "q_a_proj.weight": torch.randn(6, D_MODEL, generator=gen),
        h + "q_b_proj.weight": torch.randn(12, 6, generator=gen),
        h + "kv_a_proj_with_mqa.weight": torch.randn(rank + rope, D_MODEL, generator=gen),
        h + "kv_b_proj.weight": torch.randn(16, rank, generator=gen),
    }
    q_sign = torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0, 1.0])
    kv_sign = torch.tensor([-1.0, 1.0, 1.0, -1.0])
    ours = dict(ref)
    ours[h + "q_a_proj.weight"] = ref[h + "q_a_proj.weight"] * q_sign[:, None]
    ours[h + "q_b_proj.weight"] = ref[h + "q_b_proj.weight"] * q_sign[None, :]
    kv_a = ref[h + "kv_a_proj_with_mqa.weight"].clone()
    kv_a[:rank] *= kv_sign[:, None]
    ours[h + "kv_a_proj_with_mqa.weight"] = kv_a
    ours[h + "kv_b_proj.weight"] = ref[h + "kv_b_proj.weight"] * kv_sign[None, :]

    aligned, flips = cmp.align_mla(ours, ref, layout)
    for name, value in ref.items():
        torch.testing.assert_close(aligned[name], value)
    assert flips[h + "q_a_proj.weight"] == 3 and flips[h + "kv_a_proj_with_mqa.weight"] == 2
    products = cmp.derive(ours, layout)
    torch.testing.assert_close(
        products[h[:-4] + "mla.q_b@q_a"], cmp.derive(ref, layout)[h[:-4] + "mla.q_b@q_a"]
    )


def test_mla_layout_reads_the_rope_and_rank_from_the_checkpoint_args():
    state = {
        "decoder.layers.0.self_attention.linear_kv_down_proj.weight": torch.zeros(160, D_MODEL),
        "decoder.layers.1.mlp.linear_fc1.weight": torch.zeros(2 * FFN, D_MODEL),
    }
    layout = cmp.student_layout(state, "mla", Namespace(qk_pos_emb_head_dim=32, qk_head_dim=32))
    assert (layout.mla_rank, layout.mla_rope, layout.mla_nope) == (128, 32, 32)
    with pytest.raises(ValueError, match="qk_pos_emb_head_dim"):
        cmp.student_layout(state, "mla", Namespace())

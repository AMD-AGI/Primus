###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
from argparse import Namespace

import pytest
import torch

pytest.importorskip("safetensors")

from safetensors.torch import save_file

from primus.backends.megatron.checkpoint import mamba2_reference_draw as draw

# Small widths whose large draws need more threads than one compute unit holds at 256 threads
# per unit, so the element order depends on the count as it does for a full-size model.
WIDTHS = draw.Mamba2Widths(d_model=64, d_inner=64, d_xb=32, nheads=8, d_conv=4, num_layers=3)
THREADS = 256
SEED = 7
PLANTED = 6


def _reference(tmp_path, compute_units=PLANTED, noise=0.02):
    """A reference model.safetensors whose random tensors are the draw for ``compute_units``."""
    gen = torch.Generator().manual_seed(0)
    offsets = draw.layer_offsets(WIDTHS, compute_units, THREADS)
    tensors = {}
    for layer in range(WIDTHS.num_layers):
        drawn = draw.rebuild_layer(WIDTHS, offsets[layer], compute_units, SEED, "rocrand", THREADS)
        in_proj = torch.randn(WIDTHS.in_rows, WIDTHS.d_model, generator=gen) * 0.1
        in_proj[: WIDTHS.d_inner] = drawn["in_proj.weight[z]"].float()
        in_proj[WIDTHS.dt_row :] = drawn["in_proj.weight[dt]"].float()

        def trained(t):
            t = t.float()
            return (t + noise * t.std() * torch.randn(t.shape, generator=gen)).to(torch.bfloat16)

        prefix = f"model.layers.{layer}.mamba."
        tensors[prefix + "in_proj.weight"] = trained(in_proj)
        tensors[prefix + "conv1d.weight"] = trained(drawn["conv1d.weight"])
        tensors[prefix + "conv1d.bias"] = trained(drawn["conv1d.bias"])
        tensors[prefix + "A_log"] = trained(drawn["A_log"])
        tensors[prefix + "dt_bias"] = trained(drawn["dt_bias"])
        tensors[prefix + "D"] = torch.ones(WIDTHS.nheads, dtype=torch.bfloat16)
        tensors[prefix + "norm.weight"] = torch.ones(WIDTHS.d_inner, dtype=torch.bfloat16)
        tensors[prefix + "out_proj.weight"] = trained(
            torch.randn(WIDTHS.d_model, WIDTHS.d_inner, generator=gen)
        )
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return draw.ReferenceMamba2(path)


def test_widths_are_read_from_the_reference(tmp_path):
    reference = _reference(tmp_path)
    assert reference.widths == WIDTHS
    assert reference.prefix == "model."


def test_scan_recovers_the_planted_count_and_breaks_offset_ties(tmp_path):
    reference = _reference(tmp_path)
    results = draw.scan(reference, range(1, 11), seed=SEED, threads_per_cu=THREADS)
    best = results[0]
    assert best.compute_units == PLANTED
    assert best.conv_median > 0.95 and best.z_rows > 0.95
    # Five units start every draw from the same offsets as six, so only the z rows tell them apart.
    five = next(r for r in results if r.compute_units == 5)
    assert five.conv_median == pytest.approx(best.conv_median)
    assert five.z_rows < 0.5


def test_write_replaces_only_the_random_tensors(tmp_path):
    reference = _reference(tmp_path)
    gen = torch.Generator().manual_seed(1)
    state = {}
    for layer in range(WIDTHS.num_layers):
        mixer = f"decoder.layers.{2 * layer}.mixer."
        state[mixer + "in_proj.weight"] = torch.randn(
            WIDTHS.in_rows, WIDTHS.d_model, generator=gen
        ).bfloat16()
        state[mixer + "conv1d.weight"] = torch.randn(
            WIDTHS.conv_dim, 1, WIDTHS.d_conv, generator=gen
        ).bfloat16()
        state[mixer + "conv1d.bias"] = torch.randn(WIDTHS.conv_dim, generator=gen).bfloat16()
        state[mixer + "A_log"] = torch.randn(WIDTHS.nheads, generator=gen).bfloat16()
        state[mixer + "dt_bias"] = torch.randn(WIDTHS.nheads, generator=gen).bfloat16()
        state[mixer + "D"] = torch.ones(WIDTHS.nheads).bfloat16()
        state[f"decoder.layers.{2 * layer + 1}.mlp.linear_fc2.weight"] = torch.randn(4, 4, generator=gen)
    template = tmp_path / "template"
    (template / "iter_0000000" / "mp_rank_00").mkdir(parents=True)
    torch.save(
        {"model": state, "iteration": 0, "args": Namespace(iteration=0), "optimizer": {"x": 1}},
        template / "iter_0000000" / "mp_rank_00" / "model_optim_rng.pt",
    )
    (template / "latest_checkpointed_iteration.txt").write_text("0")

    out = tmp_path / "out"
    report = draw.write(template, reference, PLANTED, out, seed=SEED, threads_per_cu=THREADS)

    written = torch.load(out / "iter_0000000" / "mp_rank_00" / "model_optim_rng.pt", weights_only=False)
    assert written["iteration"] == 0 and "optimizer" not in written
    offsets = draw.layer_offsets(WIDTHS, PLANTED, THREADS)
    for layer in range(WIDTHS.num_layers):
        mixer = f"decoder.layers.{2 * layer}.mixer."
        drawn = draw.rebuild_layer(WIDTHS, offsets[layer], PLANTED, SEED, "rocrand", THREADS)
        in_proj = written["model"][mixer + "in_proj.weight"]
        assert torch.equal(in_proj[: WIDTHS.d_inner], drawn["in_proj.weight[z]"])
        assert torch.equal(in_proj[WIDTHS.dt_row :], drawn["in_proj.weight[dt]"])
        assert torch.equal(
            in_proj[WIDTHS.d_inner : WIDTHS.dt_row],
            state[mixer + "in_proj.weight"][WIDTHS.d_inner : WIDTHS.dt_row],
        )
        for name in ("conv1d.weight", "conv1d.bias", "A_log", "dt_bias"):
            assert torch.equal(written["model"][mixer + name], drawn[name])
        assert torch.equal(written["model"][mixer + "D"], state[mixer + "D"])
    assert min(row["conv1d.weight"] for row in report["layers"]) > 0.95
    saved = json.loads((out / "reference_draw_report.json").read_text())
    assert saved["compute_units"] == PLANTED and saved["widths"]["d_xb"] == WIDTHS.d_xb


def test_write_rejects_a_template_with_another_layer_count(tmp_path):
    reference = _reference(tmp_path)
    template = tmp_path / "template"
    (template / "iter_0000000" / "mp_rank_00").mkdir(parents=True)
    state = {"decoder.layers.0.mixer.in_proj.weight": torch.zeros(WIDTHS.in_rows, WIDTHS.d_model)}
    torch.save(
        {"model": state, "iteration": 0}, template / "iter_0000000" / "mp_rank_00" / "model_optim_rng.pt"
    )
    (template / "latest_checkpointed_iteration.txt").write_text("0")
    with pytest.raises(ValueError, match="mixers"):
        draw.write(template, reference, PLANTED, tmp_path / "out", seed=SEED, threads_per_cu=THREADS)

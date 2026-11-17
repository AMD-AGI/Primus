###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib.util
from pathlib import Path

import pytest

from primus.core.config.primus_config import load_primus_config
from primus.core.utils import file_utils

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools" / "hybrid"
CONFIGS = ROOT / "examples" / "megatron" / "configs" / "MI300X"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _iteration_line(i, total, ild, hidden, mixer):
    return (
        f"[2026-01-01 00:00:00] iteration {i:8d}/{total:8d} | consumed samples: {96 * i:12d} | "
        "elapsed time per iteration (ms): 350.0/351.2 | learning rate: 1.000000E-04 | "
        f"global batch size:    96 | hylo ild: {ild:.6E} | hylo hidden ild: {hidden:.6E} | "
        f"hylo mixer ild: {mixer:.6E} | loss scale: 1.0 | grad norm: 12.500 |\n"
    )


def test_eval_summary_averages_every_logged_step(tmp_path):
    summary = _load("summarize_hylo_ild_eval")
    log = tmp_path / "eval.log"
    log.write_text(
        "noise\n"
        "[x] : [HyLo] hylo ild=30.000000, hylo hidden ild=18.000000, hylo mixer ild=12.000000\n"
        "[x] : [HyLo] hylo ild=34.000000, hylo hidden ild=20.000000, hylo mixer ild=14.000000\n"
    )
    s = summary.summarize(summary.parse(str(log)))
    assert s["n"] == 2
    assert (s["hylo ild"], s["hylo hidden ild"], s["hylo mixer ild"]) == (32.0, 19.0, 13.0)
    empty = tmp_path / "empty.log"
    empty.write_text("no evaluation lines\n")
    rows = summary.main([f"ckpt={log}", f"empty={empty}"])
    assert rows[0][1]["median"] == 32.0 and rows[1][1] == {}


def test_loss_summary_reports_ten_step_windows_and_writes_a_csv(tmp_path, capsys):
    loss = _load("summarize_hylo_ild_loss")
    log = tmp_path / "rank.log"
    log.write_text("".join(_iteration_line(i, 20, 100.0 - i, 60.0 - i, 40.0) for i in range(1, 21)))
    result = loss.main([str(log), "--csv", str(tmp_path / "loss.csv"), "--reference-loss", "85.0"])
    assert (result["last"], result["total"]) == (20, 20)
    assert result["windows"]["1-10"]["ild"] == pytest.approx(94.5)
    assert result["windows"]["11-20"]["ild"] == pytest.approx(84.5)
    assert result["windows"]["11-20"]["mixer"] == pytest.approx(40.0)
    assert "vs reference 85.0000: -0.5000" in capsys.readouterr().out
    assert (tmp_path / "loss.csv").read_text().splitlines()[1].startswith("1,99.0,59.0,40.0")


@pytest.mark.parametrize("kind", ["m2", "mla"])
def test_eval_config_scores_fixed_weights_against_the_training_teacher(kind, monkeypatch):
    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
    monkeypatch.delenv("HYLO_EVAL_ITERS", raising=False)
    evaluation = load_primus_config(CONFIGS / f"hylo_{kind}_1B-ild-eval.yaml", None).modules[0].params
    training = (
        load_primus_config(CONFIGS / f"hylo_{kind}_1B-enhanced-ild-packed.yaml", None).modules[0].params
    )
    for name in ("spec", "hybrid_override_pattern", "hylo_teacher_use_rope_scaling", "seq_length"):
        assert getattr(evaluation, name) == getattr(training, name), name
    assert (evaluation.lr, evaluation.min_lr, evaluation.weight_decay) == (0.0, 0.0, 0.0)
    assert (evaluation.micro_batch_size, evaluation.global_batch_size, evaluation.train_iters) == (2, 2, 348)
    assert evaluation.save is None and evaluation.disable_last_saving is True

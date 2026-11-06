###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the recommendation backend (dlrm_v4): config translation,
adapter, hooks.

No GPU and no trainer imports: run_config is pure, and the adapter is only
asked for paths, model families and environment defaults.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from primus.backends.recommendation import recommendation_adapter as ra
from primus.backends.recommendation.dlrm_v4 import run_config

SRC_ROOT = Path(ra.__file__).resolve().parent / "dlrm_v4" / "src"

PRIMUS_ROOT = Path(__file__).resolve().parents[4]


def test_build_env_skips_unset_and_formats_values():
    env = run_config.build_env(
        {
            "seed": 731542219,
            "batch_size": None,
            "dense_lr": 1e-06,
            "num_train_ts": "",
            "mlperf_logging": True,
            "output_trace": False,
            "data_path": "/data",
            "not_a_knob": 3,
        }
    )
    assert env == {
        "SEED": "731542219",
        "DENSE_LR": "1e-06",
        "MLPERF_LOGGING": "1",
        "OUTPUT_TRACE": "0",
        "DLRM_DATA_PATH": "/data",
    }


def test_every_env_param_is_in_the_module_preset():
    import yaml

    preset = yaml.safe_load(
        (PRIMUS_ROOT / "primus/configs/modules/recommendation/pre_trainer.yaml").read_text()
    )
    model = yaml.safe_load(
        (PRIMUS_ROOT / "primus/configs/models/recommendation/dlrm_v4_hstu_yambda_5b.yaml").read_text()
    )
    missing = [k for k in run_config.ENV_PARAMS if k not in preset and k not in model]
    assert not missing, f"documented nowhere: {missing}"
    # Module and model presets merge without override: no key may be in both.
    assert not set(preset) & set(model)


def test_gin_file_includes_base_then_bindings():
    text = run_config.gin_file_text("/x/yambda_5b.gin", ["a.b = 1", "  ", "c.d = 'e'"])
    lines = text.splitlines()
    assert lines[1] == "include '/x/yambda_5b.gin'"
    assert lines[2:] == ["a.b = 1", "c.d = 'e'"]


def test_resolve_gin_file(tmp_path):
    assert run_config.resolve_gin_file(tmp_path, "yambda-5b", None) == (
        tmp_path / "generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin"
    )
    assert run_config.resolve_gin_file(tmp_path, "yambda-5b", "my.gin") == tmp_path / "my.gin"
    assert run_config.resolve_gin_file(tmp_path, "yambda-5b", "/abs/my.gin") == Path("/abs/my.gin")


def test_shipped_gin_files_resolve():
    for dataset in ("yambda-5b", "debug"):
        assert run_config.resolve_gin_file(SRC_ROOT, dataset, None).is_file()


def test_missing_data(tmp_path):
    assert run_config.missing_data("", None) == ["data_path is not set (DLRM_DATA_PATH)"]
    assert len(run_config.missing_data(str(tmp_path), None)) == 3
    (tmp_path / "processed_5b/hstu_cache_L4086").mkdir(parents=True)
    (tmp_path / "shared_metadata").mkdir()
    for name in ("album_item_mapping.parquet", "artist_item_mapping.parquet"):
        (tmp_path / "shared_metadata" / name).write_bytes(b"")
    assert run_config.missing_data(str(tmp_path), None) == []
    assert run_config.missing_data(str(tmp_path), 2048) == [str(tmp_path / "processed_5b/hstu_cache_L2048")]


def _mllog(key, time_ms, value=None, metadata=None):
    event = {"key": key, "time_ms": time_ms, "value": value, "metadata": metadata or {}}
    return ":::MLLOG " + json.dumps(event) + "\n"


def test_run_started_and_summary(tmp_path):
    log = tmp_path / "mlperf.log"
    assert not run_config.mlperf_run_started(str(log))
    assert run_config.run_summary(str(log)) == {}
    log.write_text(_mllog("init_start", 1000) + "noise\n" + _mllog("run_start", 201000))
    assert run_config.mlperf_run_started(str(log))
    log.write_text(
        log.read_text()
        + _mllog("eval_accuracy", 500000, 0.74)
        + _mllog("eval_accuracy", 900000, 0.7571)
        + _mllog("run_stop", 1123800, metadata={"status": "success"})
    )
    assert run_config.run_summary(str(log)) == {
        "init_s": 200.0,
        "run_s": 922.8,
        "status": "success",
        "evals": 2,
        "last_auc": 0.7571,
    }


def test_model_family_picks_the_trainer():
    from types import SimpleNamespace

    adapter = ra.RecommendationAdapter()
    args = adapter.convert_config(SimpleNamespace(model_family="dlrm_v4", seed=1))
    assert args.model_family == "dlrm_v4" and args.seed == 1
    assert adapter.load_trainer_class("pretrain").__name__ == "DlrmV4Trainer"
    with pytest.raises(ValueError):
        adapter.load_trainer_class("posttrain")
    with pytest.raises(ValueError, match="model_family"):
        adapter.convert_config({"model_family": "dcnv2"})
    with pytest.raises(ValueError, match="model_family"):
        adapter.convert_config({})


def test_backend_path_overrides_trainer_source(tmp_path, monkeypatch):
    monkeypatch.delenv("BACKEND_PATH", raising=False)
    adapter = ra.RecommendationAdapter()
    saved = list(sys.path)
    try:
        here = adapter.setup_backend_path()
        assert sys.path == saved  # the trainer adds its own source root
        assert here == str(Path(ra.__file__).resolve().parent)
        assert "trainer_src" not in vars(adapter.convert_config({"model_family": "dlrm_v4"}))
        adapter.setup_backend_path(str(tmp_path))
        assert adapter.convert_config({"model_family": "dlrm_v4"}).trainer_src == str(tmp_path)
        with pytest.raises(FileNotFoundError):
            adapter.setup_backend_path(str(tmp_path / "missing"))
    finally:
        sys.path[:] = saved


def test_env_defaults_come_from_the_family():
    adapter = ra.RecommendationAdapter()
    assert adapter.env_defaults() == []
    adapter.family = "dlrm_v4"
    names = {e.name: e.value for e in adapter.env_defaults()}
    assert names["NUMPY_MADVISE_HUGEPAGE"] == "0"
    assert names["HSTU_HAMMER_KERNEL"] == "TRITON"


def test_adapter_registered():
    from primus.core.backend.backend_registry import BackendRegistry

    assert isinstance(BackendRegistry.get_adapter("recommendation"), ra.RecommendationAdapter)


def test_trainer_source_holds_only_the_package():
    # The source root goes on sys.path: anything else there would become a
    # top-level module.
    names = sorted(p.name for p in SRC_ROOT.iterdir() if not p.name.startswith((".", "__")))
    assert names == ["LICENSE", "generative_recommenders"]


def test_prepare_hook_selects_single_run_mode(tmp_path):
    hook = PRIMUS_ROOT / "runner/helpers/hooks/train/pretrain/recommendation/prepare.py"
    out = subprocess.run(
        [sys.executable, str(hook), "--config", "x.yaml", "--data_path", str(tmp_path), "--primus_path", "."],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "NODE_RANK": "0"},
    )
    assert "env.RUN_MODE=single" in out.stdout.splitlines()


@pytest.mark.parametrize("name", ["launch"])
def test_trainer_exposes_launch(name):
    # Parsed, not imported: importing the trainer needs torch/torchrec.
    import ast

    src = (SRC_ROOT / "generative_recommenders/dlrm_v4/train/train_ranker.py").read_text()
    funcs = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    assert [a.arg for a in funcs[name].args.args] == ["dataset", "mode", "gin_file"]


def test_private_packages_pin_flydsl_and_the_preset_points_at_them():
    import yaml

    hooks = PRIMUS_ROOT / "runner/helpers/hooks/train/pretrain/recommendation"
    pins = [
        line.strip()
        for line in (hooks / "requirements-recommendation-private.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert pins == ["flydsl==0.2.4"]
    install = (hooks / "00_install_requirements.sh").read_text()
    assert "PRIMUS_RECOMMENDATION_SITE-/opt/primus-recommendation/site-packages" in install
    preset = yaml.safe_load(
        (PRIMUS_ROOT / "primus/configs/modules/recommendation/pre_trainer.yaml").read_text()
    )
    assert preset["private_site"] == "${PRIMUS_RECOMMENDATION_SITE:/opt/primus-recommendation/site-packages}"

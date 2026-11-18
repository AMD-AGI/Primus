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
import time
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


def test_run_summary(tmp_path):
    log = tmp_path / "mlperf.log"
    assert run_config.run_summary(str(log)) == {}
    log.write_text(_mllog("init_start", 1000) + "noise\n" + _mllog("run_start", 201000))
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


def _fake_trainer_tree(tmp_path):
    src = tmp_path / "src"
    gin = src / "generative_recommenders" / "dlrm_v4" / "train" / "gin"
    gin.mkdir(parents=True)
    (src / "generative_recommenders" / "__init__.py").write_text("")
    (gin / "yambda_5b.gin").write_text("batch_size = 1\n")
    data = tmp_path / "data"
    (data / "processed_5b" / "hstu_cache_L4086").mkdir(parents=True)
    (data / "shared_metadata").mkdir()
    for name in ("album_item_mapping.parquet", "artist_item_mapping.parquet"):
        (data / "shared_metadata" / name).write_bytes(b"")
    return src, data


def test_trainer_setup_and_init_keep_paths_of_the_launch_directory(tmp_path, monkeypatch):
    from primus.backends.recommendation.dlrm_v4 import trainer as T

    src, data = _fake_trainer_tree(tmp_path)
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    real_exists = os.path.exists
    monkeypatch.setattr(T.os.path, "exists", lambda p: p == "/opt/rocm/bin/hipcc" or real_exists(p))
    for name in (
        "DLRM_DATA_PATH",
        "SEED",
        "MLPERF_LOG_PATH",
        "PYTHONPATH",
        "MLPERF_CACHE_CLEARED",
        "MLPERF_LOGGING",
        "DLRM_RUN_START_MARKER",
    ):
        monkeypatch.delenv(name, raising=False)
    params = {
        "trainer_src": str(src),
        "data_path": str(data),
        "seed": 7,
        "output_dir": "rel_out",  # relative: to the launch directory, not the trainer root
        "gin_bindings": ["a.b = 2"],
        "private_site": str(tmp_path / "missing_site"),
        "mlperf_logging": True,
    }
    saved = list(sys.path)
    try:
        t = T.DlrmV4Trainer(backend_args=params)
        t.setup()
        assert os.getcwd() == str(src) and os.environ["SEED"] == "7"
        assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(src)  # no private dir: not added
        assert os.environ["MLPERF_LOG_PATH"] == str(launch / "rel_out" / "mlperf_dlrm_v4.log")
        assert not t.run_start_marker and "DLRM_RUN_START_MARKER" not in os.environ  # no warm: no marker
        t.init()
        gin = launch / "rel_out" / "dlrm_v4.gin"
        assert t.gin_path == str(gin) and "a.b = 2" in gin.read_text()
        assert os.environ["MLPERF_CACHE_CLEARED"] == "0"
        t.cleanup()
        assert os.getcwd() == str(launch)
    finally:
        sys.path[:] = saved
        os.chdir(launch)


@pytest.mark.parametrize("node_rank, writes_marker", [("0", True), ("3", True), ("3", False)])
def test_trainer_warm_waits_for_its_node_run_start_marker(tmp_path, monkeypatch, node_rank, writes_marker):
    """The in-run warm keys on a per-node RUN_START marker (written by the trainer's
    local rank 0 of each node), not on the MLPerf log: only global rank 0 writes that
    log, so nodes 1+ never saw run_start in it. Node 0 still starts a fresh log; a
    shared log must not be deleted by another node. No marker: no warm, a warning."""
    from primus.backends.recommendation.dlrm_v4 import trainer as T

    src, data = _fake_trainer_tree(tmp_path)
    launch = tmp_path / "launch"
    launch.mkdir()
    monkeypatch.chdir(launch)
    real_exists = os.path.exists
    monkeypatch.setattr(T.os.path, "exists", lambda p: p == "/opt/rocm/bin/hipcc" or real_exists(p))
    for name in ("DLRM_DATA_PATH", "SEED", "MLPERF_LOG_PATH", "PYTHONPATH", "DLRM_RUN_START_MARKER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NODE_RANK", node_rank)
    params = {
        "trainer_src": str(src),
        "data_path": str(data),
        "output_dir": "rel_out",
        "private_site": str(tmp_path / "missing_site"),
        "mlperf_logging": True,
        "warm_cache_after_run_start": True,
    }
    saved = list(sys.path)
    try:
        t = T.DlrmV4Trainer(backend_args=params)
        t.setup()
        marker = launch / "rel_out" / f"run_start.node{node_rank}"
        assert t.run_start_marker == str(marker)
        assert os.environ["DLRM_RUN_START_MARKER"] == str(marker)
        # A log with run_start in it (stale on node 0, node 0's live log elsewhere) must
        # not start the warm; node 0 removes it, other nodes keep it; this node's stale
        # marker is removed.
        log = Path(os.environ["MLPERF_LOG_PATH"])
        log.write_text(':::MLLOG {"key": "run_start"}\n')
        marker.write_text("stale\n")
        warmed, warned = [], []
        monkeypatch.setattr(t, "_pagecache", lambda mode: warmed.append(mode))
        monkeypatch.setattr(T, "warning_rank_0", warned.append)

        class _TR:
            @staticmethod
            def launch(*args):
                assert not marker.exists()  # the stale marker was removed before launch
                assert log.exists() == (node_rank != "0")
                time.sleep(0.6)  # over one 0.5 s poll: nothing but the marker starts the warm
                assert warmed == []
                if writes_marker:
                    marker.write_text("1.0\n")  # what each node's local rank 0 does at RUN_START
                    t._warm_thread.join(timeout=10)

        monkeypatch.setitem(sys.modules, "generative_recommenders.dlrm_v4.train.train_ranker", _TR)
        import importlib

        try:  # the package may already be the real one: patch its attribute too
            monkeypatch.setattr(
                importlib.import_module("generative_recommenders.dlrm_v4.train"),
                "train_ranker",
                _TR,
                raising=False,
            )
        except ImportError:
            pass
        monkeypatch.setattr(T.run_config, "run_summary", lambda path: {})
        t.train()
        assert warmed == (["warm"] if writes_marker else [])
        assert warned == ([] if writes_marker else [f"[Primus:dlrm_v4] page cache not warmed: no {marker}"])
        t.cleanup()
    finally:
        sys.path[:] = saved
        os.chdir(launch)


def test_trainer_writes_the_run_start_marker_on_local_rank_0_of_each_node():
    # The writer the warm test stands in for, parsed (importing the trainer needs
    # torch/torchrec): it reads the variable setup() exports, after RUN_START, on local
    # rank 0 (rank % device count misses nodes when GPUS_PER_NODE < visible GPUs).
    src = (SRC_ROOT / "generative_recommenders/dlrm_v4/train/train_ranker.py").read_text()
    read = src.index('_marker = os.environ.get("DLRM_RUN_START_MARKER")')
    gate = src.index("if _marker and local_rank == 0:", read)
    assert src.index("mlperf_logger.log_run_start(") < read < gate


def test_trainer_refuses_missing_data(tmp_path, monkeypatch):
    from primus.backends.recommendation.dlrm_v4 import trainer as T

    src, _ = _fake_trainer_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    t = T.DlrmV4Trainer(backend_args={"trainer_src": str(src), "data_path": str(tmp_path / "nodata")})
    saved = list(sys.path)
    try:
        with pytest.raises(FileNotFoundError, match="dataset not ready"):
            t.setup()
    finally:
        sys.path[:] = saved

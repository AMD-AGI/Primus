###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from pathlib import Path

import pytest

from primus.backends.worldmirror.runtime import install_gsplat
from primus.backends.worldmirror.worldmirror_trainer import (
    build_hydra_overrides,
    primus_hydra_overrides,
    require_stage2_checkpoint,
    resolve_worldmirror_repo,
)


def _stage1(**overrides):
    cfg = {
        "hydra_config": "stage1_hypersim.yaml",
        "hypersim_dir": "/data/hypersim",
        "max_steps": 20,
        "max_images_per_gpu": 4,
        "pretrained": "",
        "output_dir": "./output/worldmirror-stage1",
    }
    cfg.update(overrides)
    return cfg


def test_missing_gsplat_is_built_from_the_submodule(tmp_path, monkeypatch):
    import sys
    import types

    source = tmp_path / "submodules" / "gsplat"
    source.mkdir(parents=True)
    (source / "setup.py").write_text("print('build')\n")
    monkeypatch.setenv("RANK", "0")
    calls = []

    def _pip(command, cwd=None, env=None):
        calls.append((command, Path(cwd), env))
        sys.modules["gsplat"] = types.ModuleType("gsplat")

    monkeypatch.setattr("primus.backends.worldmirror.runtime.subprocess.check_call", _pip)
    monkeypatch.setattr("primus.backends.worldmirror.runtime.gsplat_imports", lambda: False)
    real_imports = {"n": 0}

    def _imports_after_pip():
        real_imports["n"] += 1
        return real_imports["n"] > 1

    monkeypatch.setattr("primus.backends.worldmirror.runtime.gsplat_imports", _imports_after_pip)
    removed = sys.modules.pop("gsplat", None)
    try:
        install_gsplat(tmp_path)
    finally:
        sys.modules.pop("gsplat", None)
        if removed is not None:
            sys.modules["gsplat"] = removed

    assert calls
    command, cwd, env = calls[0]
    assert "-e" not in command
    assert command[1:4] == ["-m", "pip", "install"]
    assert "--no-build-isolation" in command
    assert Path(command[-1]) != source
    assert cwd == Path(command[-1])
    assert (Path(command[-1]) / "setup.py").is_file()
    assert env["GIT_CONFIG_GLOBAL"]
    assert env["GIT_CONFIG_SYSTEM"]


def test_primus_hydra_searchpath_points_at_primus_configs():
    overrides = primus_hydra_overrides(["train=stage1_hypersim.yaml"])
    assert overrides[0].startswith("hydra.searchpath=[file://")
    assert overrides[0].endswith("/hydra_configs]")
    assert overrides[1] == "train=stage1_hypersim.yaml"


def test_stage1_overrides_keep_ddp_recipe_and_skip_pretrained():
    overrides = build_hydra_overrides(_stage1(), env={})
    assert "train=stage1_hypersim.yaml" in overrides
    assert "trainer.max_steps=20" in overrides
    assert "data.max_images_per_gpu=4" in overrides
    assert "paths.hypersim_dir=/data/hypersim" in overrides
    assert not any(item.startswith("wrapper.pretrained=") for item in overrides)
    assert not any(item.startswith("trainer.devices=") for item in overrides)


def test_torchrun_env_sets_lightning_device_count():
    overrides = build_hydra_overrides(
        _stage1(),
        env={"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "8", "NNODES": "1"},
    )
    assert "trainer.devices=8" in overrides
    assert "trainer.num_nodes=1" in overrides


def test_stage2_requires_a_weight_checkpoint():
    with pytest.raises(ValueError, match="stage-1 checkpoint"):
        require_stage2_checkpoint("stage2_hypersim.yaml", "")


def test_stage2_override_uses_pretrained_not_ckpt_path():
    overrides = build_hydra_overrides(
        _stage1(
            hydra_config="stage2_hypersim.yaml",
            pretrained="/ckpt/last.ckpt",
            output_dir="./output/worldmirror-stage2",
        ),
        env={},
    )
    assert "wrapper.pretrained=/ckpt/last.ckpt" in overrides
    assert not any(item.startswith("ckpt_path=") for item in overrides)


def test_resolve_repo_uses_explicit_path(tmp_path):
    (tmp_path / "training").mkdir()
    (tmp_path / "training" / "launch.py").write_text("print('ok')\n")
    assert resolve_worldmirror_repo(str(tmp_path)) == tmp_path.resolve()


def test_resolve_repo_rejects_a_directory_without_launch(tmp_path):
    with pytest.raises(FileNotFoundError, match="training/launch.py"):
        resolve_worldmirror_repo(str(tmp_path))


def test_resolve_repo_uses_third_party_submodule(tmp_path, monkeypatch):
    primus_root = tmp_path / "Primus"
    sibling = tmp_path / "HunyuanWorld-Mirror-rocm"
    submodule = primus_root / "third_party" / "HunyuanWorld-Mirror-rocm"
    for repo in (sibling, submodule):
        (repo / "training").mkdir(parents=True)
        (repo / "training" / "launch.py").write_text("print('ok')\n")
    monkeypatch.delenv("WORLD_MIRROR_PATH", raising=False)
    assert resolve_worldmirror_repo("", primus_root=primus_root) == submodule.resolve()

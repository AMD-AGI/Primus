###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest

from primus.pretrain import load_backend_trainer, setup_backend_path


@pytest.fixture
def temp_backend_dirs():
    """
    Create temporary directories to simulate backend sources.
    Returns a dict: {"megatron": path1, "torchtitan": path2}
    """
    tmp_dirs = {}
    for name in ["megatron", "torchtitan"]:
        d = tempfile.mkdtemp(prefix=f"{name}_")
        tmp_dirs[name] = d
    yield tmp_dirs
    for d in tmp_dirs.values():
        shutil.rmtree(d, ignore_errors=True)


def test_setup_backend_path_with_cli(temp_backend_dirs):
    """Test that setup_backend_path works with CLI-provided backend_path."""
    path = setup_backend_path(framework="megatron", backend_path=temp_backend_dirs["megatron"])
    assert os.path.exists(path)
    assert path in sys.path


def test_setup_backend_path_with_env(monkeypatch, temp_backend_dirs):
    """Test that setup_backend_path uses BACKEND_PATH environment variable."""
    monkeypatch.setenv("BACKEND_PATH", temp_backend_dirs["torchtitan"])
    path = setup_backend_path(framework="torchtitan")
    assert os.path.exists(path)
    assert path in sys.path


def test_setup_backend_path_with_fallback():
    """Test fallback to third_party directory when no CLI or env path is set."""
    primus_root = Path(__file__).resolve().parent.parent / "primus"
    fallback_dir = primus_root.parent / "third_party" / "Megatron-LM"
    fallback_dir.mkdir(parents=True, exist_ok=True)

    path = setup_backend_path(framework="megatron")
    assert os.path.exists(path)
    assert path in sys.path

    # Optional cleanup
    fallback_dir.rmdir()


def test_setup_backend_path_failure():
    """Test that FileNotFoundError is raised when no valid backend path exists."""
    with pytest.raises(FileNotFoundError):
        setup_backend_path(framework="nonexistent_backend")


def test_load_backend_trainer_supported(monkeypatch):
    """Test that load_backend_trainer returns correct class for supported frameworks."""
    # Mock: Define a dummy class to simulate MegatronPretrainTrainer
    dummy_class = type("DummyTrainer", (), {})
    dummy_module = types.ModuleType("primus.modules.trainer.megatron.pre_trainer")
    dummy_module.MegatronPretrainTrainer = dummy_class

    # Inject into sys.modules to bypass real import
    sys.modules["primus.modules.trainer.megatron.pre_trainer"] = dummy_module

    trainer_cls = load_backend_trainer("megatron")
    assert trainer_cls is dummy_class


def test_load_backend_trainer_unsupported():
    """Test that unsupported framework raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported framework"):
        load_backend_trainer("invalid_framework")

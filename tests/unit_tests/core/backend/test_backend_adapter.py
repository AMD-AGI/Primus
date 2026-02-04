###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for BackendAdapter base class.

These tests verify that:
    - dummy adapters implement required abstract methods
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

import primus.core.backend.backend_adapter as adapter_module


class DummyTrainer:
    """
    Simple trainer implementation used for testing.

    It records the arguments passed to its constructor so tests can assert
    that BackendAdapter.create_trainer wires everything correctly.
    """

    def __init__(
        self,
        primus_config: Any,
        module_config: Any,
        backend_args: Any,
    ):
        self.primus_config = primus_config
        self.module_config = module_config
        self.backend_args = backend_args


class DummyBackendAdapter(adapter_module.BackendAdapter):
    """
    Minimal concrete BackendAdapter implementation for tests.

    It records calls to its abstract methods so tests can validate
    the orchestration performed by the base class.
    """

    def __init__(self, framework: str = "test_framework", version: str = "1.0.0"):
        super().__init__(framework=framework)
        self._version = version
        self.prepare_calls: List[Any] = []
        self.convert_calls: List[Any] = []
        self.detect_version_calls: int = 0
        self.load_trainer_calls: int = 0

    def prepare_backend(self, config: Any):
        self.prepare_calls.append(config)

    def convert_config(self, config: Any) -> Dict[str, Any]:
        self.convert_calls.append(config)
        # Use SimpleNamespace instead of a plain dict so that BackendAdapter
        # can treat backend_args like a real backend object:
        #   - vars(backend_args) works (matching how real argparse.Namespace behaves)
        #   - attribute-style access (backend_args.lr) is available, which
        #     mirrors how Megatron/Titan trainers typically consume args.
        return SimpleNamespace(
            lr=1e-4,
            global_batch_size=128,
            model_name=getattr(config, "model", None),
        )

    def load_trainer_class(self, stage: str = "pretrain"):
        self.load_trainer_calls += 1
        return DummyTrainer

    def detect_backend_version(self) -> str:
        self.detect_version_calls += 1
        return self._version


@pytest.fixture
def module_config():
    # Minimal module_config with the attributes BackendAdapter expects
    return SimpleNamespace(
        model="test-model",
        params={
            "model": "test-model",
            "lr": 1e-4,
            "global_batch_size": 128,
            "primus_only_flag": True,
        },
    )


@pytest.fixture
def primus_config():
    # Primus config is passed through to the trainer unchanged
    return SimpleNamespace(exp_name="unit-test-exp")


def test_create_trainer_orchestrates_flow(monkeypatch, primus_config, module_config):
    adapter = DummyBackendAdapter(framework="megatron", version="1.2.3")

    # Abstract methods were called exactly once
    adapter.prepare_backend(module_config)
    adapter.convert_config(module_config)
    adapter.load_trainer_class(stage="pretrain")
    adapter.detect_backend_version()

    assert adapter.prepare_calls == [module_config]
    assert adapter.convert_calls == [module_config]
    assert adapter.load_trainer_calls == 1
    assert adapter.detect_version_calls == 1


def test_adapter_setup_backend_path_with_explicit_path(tmp_path, monkeypatch):
    adapter = DummyBackendAdapter(framework="test_backend")
    backend_dir = tmp_path / "explicit_backend"
    backend_dir.mkdir()

    # Keep sys.path clean after test.
    original_sys_path = list(__import__("sys").path)
    try:
        resolved = adapter.setup_backend_path(backend_path=str(backend_dir))
        assert resolved == str(backend_dir)
        assert str(backend_dir) in __import__("sys").path
    finally:
        __import__("sys").path[:] = original_sys_path


def test_adapter_setup_backend_path_with_env_var(tmp_path, monkeypatch):
    adapter = DummyBackendAdapter(framework="test_backend")
    backend_dir = tmp_path / "env_backend"
    backend_dir.mkdir()

    original_sys_path = list(__import__("sys").path)
    monkeypatch.setenv("BACKEND_PATH", str(backend_dir))
    try:
        resolved = adapter.setup_backend_path(backend_path=None)
        assert resolved == str(backend_dir)
        assert str(backend_dir) in __import__("sys").path
    finally:
        __import__("sys").path[:] = original_sys_path

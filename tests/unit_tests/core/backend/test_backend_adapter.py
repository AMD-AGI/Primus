###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for BackendAdapter base class.

These tests verify that:
    - setup and build_args patches are invoked with correct context
    - the high-level create_trainer orchestration calls all required steps
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

    def load_trainer_class(self, stage: str | None = None):
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


def test_apply_setup_patches_invokes_run_patches(monkeypatch, module_config, primus_config):
    from primus.core import patches as patches_module

    calls = []

    def fake_run_patches(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(patches_module, "run_patches", fake_run_patches)

    adapter = DummyBackendAdapter(framework="megatron")

    adapter._apply_setup_patches(primus_config, module_config)

    assert len(calls) == 1
    call = calls[0]
    assert call["backend"] == "megatron"
    assert call["phase"] == "setup"
    assert call["backend_version"] is None
    assert call["model_name"] == "test-model"
    assert call["extra"]["module_config"] is module_config
    assert call["extra"]["primus_config"] is primus_config


def test_apply_build_args_patches_uses_detected_version(monkeypatch, module_config, primus_config):
    from primus.core import patches as patches_module

    calls = []

    def fake_run_patches(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(patches_module, "run_patches", fake_run_patches)

    adapter = DummyBackendAdapter(framework="megatron", version="0.9.0")
    backend_args = SimpleNamespace(dummy_arg=True)

    adapter._apply_build_args_patches(primus_config, module_config, backend_args)

    assert len(calls) == 1
    call = calls[0]
    assert call["backend"] == "megatron"
    assert call["phase"] == "build_args"
    assert call["backend_version"] == "0.9.0"
    assert call["model_name"] == "test-model"
    assert call["extra"]["backend_args"] is backend_args
    assert call["extra"]["module_config"] is module_config
    assert call["extra"]["primus_config"] is primus_config


def test_create_trainer_orchestrates_flow(monkeypatch, primus_config, module_config):
    # Silence logging in tests

    monkeypatch.setattr(adapter_module, "log_rank_0", lambda *args, **kwargs: None)
    # BackendAdapter uses log_dict_aligned without importing it explicitly.
    # Use raising=False so the test remains robust even if the module does not
    # yet define log_dict_aligned (older versions, refactors, etc.): in that
    # case monkeypatch.setattr becomes a no-op instead of failing the test.
    monkeypatch.setattr(
        adapter_module,
        "log_dict_aligned",
        lambda *args, **kwargs: None,
        raising=False,
    )

    # Capture patch invocations
    from primus.core import patches as patches_module

    patch_calls = []

    def fake_run_patches(**kwargs):
        patch_calls.append(kwargs)

    monkeypatch.setattr(patches_module, "run_patches", fake_run_patches)

    adapter = DummyBackendAdapter(framework="megatron", version="1.2.3")

    trainer = adapter.create_trainer(primus_config, module_config)

    # Trainer instance is created with the expected type and arguments
    assert isinstance(trainer, DummyTrainer)
    assert trainer.primus_config is primus_config
    assert trainer.module_config is module_config
    assert getattr(trainer.backend_args, "lr") == 1e-4
    assert getattr(trainer.backend_args, "global_batch_size") == 128

    # Abstract methods were called exactly once
    assert adapter.prepare_calls == [module_config]
    assert adapter.convert_calls == [module_config]
    assert adapter.detect_version_calls == 1
    assert adapter.load_trainer_calls == 1

    # Both setup and build_args patches were invoked
    phases = {c["phase"] for c in patch_calls}
    assert {"setup", "build_args"} <= phases

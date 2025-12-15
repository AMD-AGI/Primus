###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for BaseTrainer and its integration with the patch system.
"""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from primus.core.trainer.base_trainer import BaseTrainer


class DummyTrainer(BaseTrainer):
    """Minimal concrete implementation of BaseTrainer for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_calls: int = 0
        # Track how many times detect_version is called on this instance
        self.detect_calls: int = 0

    def init(self, *args, **kwargs):
        """No-op init for testing."""
        return None

    def setup(self, *args, **kwargs):
        """No-op setup for testing."""
        return None

    def run_train(self):
        self.run_calls += 1

    @classmethod
    def detect_version(cls) -> str:
        # Increment per-call counter to validate BaseTrainer.run usage
        # NOTE: For classmethod, we use a simple class attribute counter.
        if not hasattr(cls, "detect_calls_counter"):
            cls.detect_calls_counter = 0  # type: ignore[attr-defined]
        cls.detect_calls_counter += 1  # type: ignore[attr-defined]
        return "test-version"


class TestBaseTrainerPatchIntegration:
    """Verify that BaseTrainer.run wires patch execution correctly."""

    def test_run_invokes_patches_and_training(self, monkeypatch):
        # Silence logging inside BaseTrainer
        monkeypatch.setattr("primus.modules.module_utils.log_rank_0", lambda *args, **kwargs: None)
        # Capture run_patches calls
        calls: List[Dict[str, Any]] = []

        def fake_run_patches(**kwargs):
            calls.append(kwargs)
            # Return a deterministic patch count so BaseTrainer logging works
            return 1

        monkeypatch.setattr("primus.core.trainer.base_trainer.run_patches", fake_run_patches)

        primus_config = SimpleNamespace(exp_root_path="/tmp/exp", exp_meta_info={})
        module_config = SimpleNamespace(framework="megatron", model="llama2_7B", trainable=True)
        backend_args = {"lr": 1e-4}

        trainer = DummyTrainer(primus_config, module_config, backend_args=backend_args)

        # Reset class-level detect counter before run
        DummyTrainer.detect_calls_counter = 0  # type: ignore[attr-defined]

        trainer.run()

        # Training loop executed exactly once
        assert trainer.run_calls == 1

        # detect_version is called once per phase (before_train and after_train)
        assert DummyTrainer.detect_calls_counter == 2  # type: ignore[attr-defined]

        # Patches were invoked twice: before_train and after_train
        assert len(calls) == 2

        before_call, after_call = calls

        # Common assertions for both phases
        for call, phase in zip((before_call, after_call), ("before_train", "after_train")):
            assert call["backend"] == "megatron"
            assert call["phase"] == phase
            assert call["backend_version"] == "test-version"
            assert call["model_name"] == "llama2_7B"

            extra = call["extra"]
            assert extra["args"] is backend_args
            assert extra["primus_config"] is primus_config
            assert extra["module_config"] is module_config

    def test_missing_framework_or_model_raises(self):
        primus_config = SimpleNamespace()

        # Missing framework
        module_config = SimpleNamespace(model="llama2_7B", trainable=True)
        with pytest.raises(ValueError, match="'framework' is required"):
            DummyTrainer(primus_config, module_config)

        # Missing model
        module_config = SimpleNamespace(framework="megatron", trainable=True)
        with pytest.raises(ValueError, match="'model' is required"):
            DummyTrainer(primus_config, module_config)

    def test_run_allows_missing_backend_args(self, monkeypatch):
        """If backend_args is None, patches still receive the key with None value."""
        monkeypatch.setattr("primus.modules.module_utils.log_rank_0", lambda *args, **kwargs: None)

        calls: List[Dict[str, Any]] = []

        def fake_run_patches(**kwargs):
            calls.append(kwargs)
            return 1

        monkeypatch.setattr("primus.core.trainer.base_trainer.run_patches", fake_run_patches)

        primus_config = SimpleNamespace(exp_root_path="/tmp/exp", exp_meta_info={})
        module_config = SimpleNamespace(framework="megatron", model="llama2_7B", trainable=True)

        trainer = DummyTrainer(primus_config, module_config)
        trainer.run()

        # Two patch invocations, both should carry args=None
        assert len(calls) == 2
        for call in calls:
            assert call["extra"]["args"] is None
            assert call["extra"]["primus_config"] is primus_config
            assert call["extra"]["module_config"] is module_config

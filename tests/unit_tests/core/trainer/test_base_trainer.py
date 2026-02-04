###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for BaseTrainer."""

from types import SimpleNamespace

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
    """Verify that BaseTrainer.run delegates to run_train()."""

    def test_run_invokes_patches_and_training(self, monkeypatch):
        # Silence logging inside BaseTrainer
        monkeypatch.setattr("primus.modules.module_utils.log_rank_0", lambda *args, **kwargs: None)

        primus_config = SimpleNamespace(exp_root_path="/tmp/exp", exp_meta_info={})
        module_params = SimpleNamespace()
        module_config = SimpleNamespace(
            framework="megatron",
            model="llama2_7B",
            trainable=True,
            params=module_params,
        )
        backend_args = {"lr": 1e-4}

        trainer = DummyTrainer(primus_config, module_config, backend_args=backend_args)

        trainer.run()

        # Training loop executed exactly once
        assert trainer.run_calls == 1

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
        """If backend_args is None, trainer.run() still works."""
        monkeypatch.setattr("primus.modules.module_utils.log_rank_0", lambda *args, **kwargs: None)

        primus_config = SimpleNamespace(exp_root_path="/tmp/exp", exp_meta_info={})
        module_params = SimpleNamespace()
        module_config = SimpleNamespace(
            framework="megatron",
            model="llama2_7B",
            trainable=True,
            params=module_params,
        )

        # Reset Megatron global vars between tests so set_primus_global_variables
        # can be called multiple times in this test module without assertion.
        from primus.backends.megatron.training.global_vars import destroy_global_vars

        destroy_global_vars()

        trainer = DummyTrainer(primus_config, module_config)
        trainer.run()

        assert trainer.run_calls == 1

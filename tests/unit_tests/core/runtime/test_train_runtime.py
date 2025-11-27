###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# Unit tests for PrimusRuntime training orchestrator.
###############################################################################

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

from primus.core.backend.backend_registry import BackendRegistry
from primus.core.config.primus_config import PrimusConfig
from primus.core.runtime.train_runtime import PrimusRuntime


class DummyTrainer:
    """Minimal trainer used to verify lifecycle sequencing."""

    def __init__(self):
        self.calls: List[str] = []
        self.cleanup_on_error: Optional[bool] = None

    def setup(self):
        self.calls.append("setup")

    def init(self):
        self.calls.append("init")

    def run(self):
        self.calls.append("run")

    def cleanup(self, on_error: bool = False):
        self.calls.append("cleanup")
        self.cleanup_on_error = on_error


class TestPrimusRuntime:
    """End-to-end behaviour of PrimusRuntime with minimal mocks."""

    def setup_method(self):
        # Save and clear BackendRegistry adapters for isolation
        self._original_adapters = BackendRegistry._adapters.copy()
        BackendRegistry._adapters.clear()

    def teardown_method(self):
        BackendRegistry._adapters = self._original_adapters

    def _make_fake_primus_config(self, tmp_path: Path):
        # Build a minimal in-memory PrimusConfig via YAML to exercise from_file().
        # We include required workspace metadata fields so that PrimusConfig
        # validation passes without needing the full runner stack.
        cfg_path = tmp_path / "train.yaml"
        cfg_path.write_text(
            """
version: 1
work_group: ut
user_name: tester
exp_name: unittest
workspace: /tmp/primus-ut
modules:
  - module: pretrain
    name: pretrain
    framework: megatron
    params:
      a: 1
      nested:
        x: 10
"""
        )
        primus_cfg = PrimusConfig.from_file(cfg_path, SimpleNamespace())
        return cfg_path, primus_cfg

    def test_successful_run_invokes_lifecycle_and_trainer(self, monkeypatch, tmp_path):
        """Happy path: hooks + registry + trainer lifecycle are all called in order."""
        cfg_path, primus_cfg = self._make_fake_primus_config(tmp_path)

        # Fake args object matching CLI
        args = SimpleNamespace(
            config=str(cfg_path),
            data_path=str(tmp_path / "data"),
            backend_path=None,
        )

        # Prepare filesystem expectation for setup_training_env (no-op friendly)
        (tmp_path / "data").mkdir(exist_ok=True)

        # Monkeypatch PrimusConfig.from_file and set_global_variables to avoid
        # touching real platform / environment detection logic.
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.PrimusConfig.from_file",
            lambda path, _args: primus_cfg,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.set_global_variables",
            lambda *_a, **_k: None,
        )

        # Monkeypatch setup_training_env, init_distributed_env, init_global_logger and
        # log_rank_0 to avoid side effects (logger is not initialized in this UT).
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.setup_training_env",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.init_distributed_env",
            lambda: None,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.init_global_logger",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.log_rank_0",
            lambda *_a, **_k: None,
        )

        # Register a simple backend adapter that returns DummyTrainer
        class DummyAdapter:
            def __init__(self, framework: str):
                self.framework = framework

            def prepare_backend(self, config):
                pass

            def convert_config(self, config):
                return {}

            def load_trainer_class(self):
                return DummyTrainer

            def create_trainer(self, primus_config, module_config):
                return DummyTrainer()

        BackendRegistry.register_adapter("megatron", DummyAdapter)

        runtime = PrimusRuntime(args=args)
        runtime.run_train_module(module_name="pretrain", overrides=["a=2"])

        # Trainer lifecycle should have been executed
        assert isinstance(runtime.ctx.trainer, DummyTrainer)
        assert runtime.ctx.trainer.calls == ["setup", "init", "run", "cleanup"]
        assert runtime.ctx.trainer.cleanup_on_error is False

        # Overrides should have been applied (PrimusRuntime uses deep_merge on params)
        assert runtime.ctx.module_config.params["a"] == 2

    def test_error_in_run_triggers_cleanup_and_hooks(self, monkeypatch, tmp_path):
        """If trainer.run raises, PrimusRuntime should cleanup with on_error=True and wrap error."""
        cfg_path, primus_cfg = self._make_fake_primus_config(tmp_path)

        args = SimpleNamespace(
            config=str(cfg_path),
            data_path=str(tmp_path / "data"),
            backend_path=None,
        )
        (tmp_path / "data").mkdir(exist_ok=True)

        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.PrimusConfig.from_file",
            lambda path, _args: primus_cfg,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.set_global_variables",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.setup_training_env",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.init_distributed_env",
            lambda: None,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.init_global_logger",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "primus.core.runtime.train_runtime.log_rank_0",
            lambda *_a, **_k: None,
        )

        class FailingTrainer(DummyTrainer):
            def run(self):
                self.calls.append("run")
                raise RuntimeError("boom")

        class DummyAdapter:
            def __init__(self, framework: str):
                self.framework = framework

            def prepare_backend(self, config):
                pass

            def convert_config(self, config):
                return {}

            def load_trainer_class(self):
                return FailingTrainer

            def create_trainer(self, primus_config, module_config):
                return FailingTrainer()

        BackendRegistry.register_adapter("megatron", DummyAdapter)

        runtime = PrimusRuntime(args=args)

        with pytest.raises(RuntimeError, match="Training execution failed:"):
            runtime.run_train_module(module_name="pretrain", overrides=None)

        # Even on error, trainer.cleanup should be called with on_error=True
        assert isinstance(runtime.ctx.trainer, FailingTrainer)
        assert "run" in runtime.ctx.trainer.calls
        assert "cleanup" in runtime.ctx.trainer.calls
        assert runtime.ctx.trainer.cleanup_on_error is True

        # Error should have been wrapped and cleanup(on_error=True) executed

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from primus.core.runtime.train_runtime import PrimusRuntime
from tests.utils import PrimusUT


class TestPrimusRuntime(PrimusUT):
    def _build_args(self, config: str = "examples/megatron/exp_pretrain.yaml") -> argparse.Namespace:
        return argparse.Namespace(config=config, data_path="./data", backend_path=None)

    def test_missing_config_file_raises_file_not_found(self):
        args = self._build_args(config="non_existent.yaml")
        runtime = PrimusRuntime(args=args)

        with self.assertRaises(RuntimeError) as ctx:
            runtime.run_train_module(module_name="pre_trainer", overrides=[])

        # The original FileNotFoundError is wrapped in RuntimeError
        msg = str(ctx.exception)
        self.assertIn("Config file not found", msg)
        self.assertIn("non_existent.yaml", msg)

    def test_missing_module_raises_runtime_error(self):
        # Use a real example config but request an invalid module name.
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        with self.assertRaises(RuntimeError) as ctx:
            runtime.run_train_module(module_name="unknown_trainer", overrides=[])

        msg = str(ctx.exception)
        self.assertIn("Missing required module 'unknown_trainer'", msg)
        self.assertIn("Available modules:", msg)

    def test_apply_overrides_merges_into_module_params(self):
        # Prepare a minimal fake runtime with injected context to isolate _apply_overrides.
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        # Fake module config with params as SimpleNamespace (matching actual runtime behavior).
        module_cfg = SimpleNamespace(
            name="pre_trainer",
            framework="megatron",
            params=SimpleNamespace(a=1, nested=SimpleNamespace(b=2)),
        )

        # Inject a minimal TrainContext.
        runtime.ctx = SimpleNamespace(
            config_path=Path("dummy.yaml"),
            data_path=Path("./data"),
            module_name="pre_trainer",
            primus_config=SimpleNamespace(),
            module_config=module_cfg,
            framework="megatron",
        )

        overrides = ["a=10", "nested.b=20", "new_key=30"]
        runtime._apply_overrides(module_cfg, overrides)

        # After _apply_overrides, params is converted back to SimpleNamespace tree
        self.assertEqual(module_cfg.params.a, 10)
        self.assertEqual(module_cfg.params.nested.b, 20)
        self.assertEqual(module_cfg.params.new_key, 30)

    def test_initialize_backend_wraps_adapter_errors(self):
        """BackendRegistry.get_adapter errors should be wrapped with context."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        # Load a real module config first.
        runtime._initialize_configuration(module_name="pre_trainer", overrides=[])

        with patch(
            "primus.core.backend.backend_registry.BackendRegistry.get_adapter",
            side_effect=ValueError("backend boom"),
        ):
            with self.assertRaises(ValueError) as ctx:
                runtime._initialize_adapter()

        msg = str(ctx.exception)
        self.assertIn("backend boom", msg)

    def test_initialize_trainer_wraps_creation_errors(self):
        """Adapter.convert_config errors should be wrapped into RuntimeError."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        # Prepare configuration and inject a failing adapter.
        runtime._initialize_configuration(module_name="pre_trainer", overrides=[])

        class FailingAdapter:
            def detect_backend_version(self):
                return "unknown"

            def prepare_backend(self, module_config):
                return None

            def convert_config(self, params):
                raise RuntimeError("trainer boom")

            def load_trainer_class(self, stage: str = "pretrain"):
                raise AssertionError("should not be called")

        runtime.ctx.adapter = FailingAdapter()  # type: ignore[attr-defined]

        with self.assertRaises(RuntimeError) as ctx:
            runtime._initialize_trainer()

        msg = str(ctx.exception)
        self.assertIn("trainer boom", msg)

    def test_run_trainer_lifecycle_calls_trainer_methods_in_order(self):
        """Trainer lifecycle should call setup → init → train → cleanup in order."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        # Use a dummy trainer that records the call order.
        class DummyTrainer:
            def __init__(self, backend_args=None):
                self.calls = []
                self.backend_args = backend_args

            def setup(self):
                self.calls.append("setup")

            def init(self):
                self.calls.append("init")

            def train(self):
                self.calls.append("train")

            def cleanup(self, on_error: bool = False):
                self.calls.append("cleanup_error" if on_error else "cleanup")

        # Patch backend adapter creation to return our dummy adapter
        with patch(
            "primus.core.backend.backend_registry.BackendRegistry.get_adapter",
        ) as mock_get_adapter:
            mock_adapter = Mock()
            mock_adapter.detect_backend_version.return_value = "test-version"
            mock_adapter.prepare_backend.return_value = None
            mock_adapter.convert_config.return_value = SimpleNamespace(lr=1e-4)
            mock_adapter.load_trainer_class.return_value = DummyTrainer
            mock_get_adapter.return_value = mock_adapter

            # This will go through the full happy path, including lifecycle.
            runtime.run_train_module(module_name="pre_trainer", overrides=[])

        # The trainer instance is created inside runtime; verify it executed in order.
        trainer = runtime.ctx.trainer
        self.assertEqual(trainer.calls, ["setup", "init", "train", "cleanup"])

    def test_runtime_applies_patch_phases_in_expected_order(self):
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        phases = []

        def _fake_run_patches(**kwargs):
            phases.append(kwargs["phase"])
            return 0

        # Patch run_patches inside the runtime module
        with patch("primus.core.runtime.train_runtime.run_patches", side_effect=_fake_run_patches):
            # Dummy trainer
            class DummyTrainer:
                def __init__(self, backend_args=None):
                    self.backend_args = backend_args

                def setup(self):
                    pass

                def init(self):
                    pass

                def train(self):
                    pass

                def cleanup(self, on_error: bool = False):
                    pass

            with patch(
                "primus.core.backend.backend_registry.BackendRegistry.get_adapter"
            ) as mock_get_adapter:
                mock_adapter = Mock()
                mock_adapter.detect_backend_version.return_value = "test-version"
                mock_adapter.prepare_backend.return_value = None
                mock_adapter.convert_config.return_value = SimpleNamespace(lr=1e-4, stage="pretrain")
                mock_adapter.load_trainer_class.return_value = DummyTrainer
                mock_get_adapter.return_value = mock_adapter

                runtime.run_train_module(module_name="pre_trainer", overrides=[])

        assert phases == ["build_args", "setup", "before_train", "after_train"]


if __name__ == "__main__":
    unittest.main()

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
        runtime._apply_overrides("pre_trainer", module_cfg, overrides)

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
                runtime._initialize_backend()

        msg = str(ctx.exception)
        self.assertIn("backend boom", msg)

    def test_initialize_trainer_wraps_creation_errors(self):
        """Adapter.create_trainer errors should be wrapped into RuntimeError."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        # Prepare configuration and inject a failing adapter.
        runtime._initialize_configuration(module_name="pre_trainer", overrides=[])

        class FailingAdapter:
            def create_trainer(self, primus_config, module_config):
                raise RuntimeError("trainer boom")

        runtime.ctx.adapter = FailingAdapter()  # type: ignore[attr-defined]

        with self.assertRaises(RuntimeError) as ctx:
            runtime._initialize_trainer()

        msg = str(ctx.exception)
        self.assertIn("trainer boom", msg)

    def test_run_trainer_lifecycle_calls_trainer_methods_in_order(self):
        """Trainer lifecycle should call setup → init → run → cleanup in order."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        # Use a dummy trainer that records the call order.
        class DummyTrainer:
            def __init__(self):
                self.calls = []

            def setup(self):
                self.calls.append("setup")

            def init(self):
                self.calls.append("init")

            def run(self):
                self.calls.append("run")

            def cleanup(self, on_error: bool = False):
                self.calls.append("cleanup_error" if on_error else "cleanup")

        trainer = DummyTrainer()

        # Patch backend adapter creation to return our dummy trainer
        with patch(
            "primus.core.backend.backend_registry.BackendRegistry.get_adapter",
        ) as mock_get_adapter:
            mock_adapter = Mock()
            mock_adapter.create_trainer.return_value = trainer
            mock_get_adapter.return_value = mock_adapter

            # This will go through the full happy path, including lifecycle.
            runtime.run_train_module(module_name="pre_trainer", overrides=[])

        self.assertEqual(trainer.calls, ["setup", "init", "run", "cleanup"])


if __name__ == "__main__":
    unittest.main()

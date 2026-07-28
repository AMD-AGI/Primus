###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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
            def __init__(self, backend_args=None, **kwargs):
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
                def __init__(self, backend_args=None, **kwargs):
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


class TestPrimusRuntimeTrainerClassSelection(PrimusUT):
    """Test trainer class selection from config."""

    def _build_args(self, config: str = "examples/megatron/exp_pretrain.yaml"):
        """Build args for testing."""
        import argparse

        return argparse.Namespace(config=config, data_path="./data", backend_path=None)

    def test_initialize_trainer_extracts_trainer_class_from_module_config_top_level(self):
        """Test that trainer_class is extracted from module_config (top-level)."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        # Create a mock adapter that records calls
        mock_adapter = Mock()
        mock_adapter.prepare_backend = Mock()
        mock_adapter.convert_config = Mock(return_value=SimpleNamespace())

        # Mock trainer class
        mock_trainer_class = Mock()
        mock_trainer_class.__name__ = "FluxPretrainTrainer"
        mock_trainer_instance = Mock()
        mock_trainer_class.return_value = mock_trainer_instance
        mock_adapter.load_trainer_class = Mock(return_value=mock_trainer_class)

        # Setup context with trainer_class in module_config (top-level)
        runtime.ctx = SimpleNamespace(
            config_path=Path("dummy.yaml"),
            data_path=Path("./data"),
            module_name="pre_trainer",
            primus_config=SimpleNamespace(),
            rank=0,
            world_size=1,
            master_addr="localhost",
            master_port="12345",
            module_config=SimpleNamespace(
                framework="megatron",
                trainer_class="FluxPretrainTrainer",  # Top-level attribute
                params=SimpleNamespace(stage="pretrain"),
            ),
            framework="megatron",
            adapter=mock_adapter,
        )

        # Mock patches and logging
        with patch("primus.core.runtime.train_runtime.log_dict_aligned"), patch(
            "primus.core.runtime.train_runtime.log_rank_0"
        ) as mock_log, patch("primus.core.runtime.train_runtime.merge_namespace"), patch.object(
            runtime, "_run_phase_patches"
        ):

            runtime._initialize_trainer()

        # Verify adapter.load_trainer_class was called with trainer_class
        mock_adapter.load_trainer_class.assert_called_once_with(
            stage="pretrain", trainer_class="FluxPretrainTrainer"
        )

        # Verify logging indicates trainer_class usage
        log_calls = [str(call) for call in mock_log.call_args_list]
        assert any("Using trainer_class: FluxPretrainTrainer" in str(call) for call in log_calls)

    def test_initialize_trainer_extracts_trainer_class_from_params_when_not_top_level(self):
        """Test that trainer_class is extracted from params when not in top-level."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        mock_adapter = Mock()
        mock_adapter.prepare_backend = Mock()
        # Critically, backend_args does NOT carry trainer_class. This isolates the
        # params-extraction branch: the only way trainer_class can be resolved is
        # from module_config.params (the post-merge backend_args fallback cannot
        # mask a regression in that branch).
        backend_args = SimpleNamespace(stage="pretrain")
        mock_adapter.convert_config = Mock(return_value=backend_args)
        mock_trainer_class = Mock()
        mock_trainer_class.__name__ = "FluxPretrainTrainer"
        mock_trainer_class.return_value = Mock()
        mock_adapter.load_trainer_class = Mock(return_value=mock_trainer_class)

        # trainer_class in params, NOT top-level (no top-level attribute)
        runtime.ctx = SimpleNamespace(
            config_path=Path("dummy.yaml"),
            data_path=Path("./data"),
            module_name="pre_trainer",
            primus_config=SimpleNamespace(),
            rank=0,
            world_size=1,
            master_addr="localhost",
            master_port="12345",
            module_config=SimpleNamespace(
                framework="megatron",
                # No trainer_class attribute here
                params=SimpleNamespace(
                    stage="pretrain",
                    trainer_class="FluxPretrainTrainer",  # In params
                ),
            ),
            framework="megatron",
            adapter=mock_adapter,
        )

        with patch("primus.core.runtime.train_runtime.log_dict_aligned"), patch(
            "primus.core.runtime.train_runtime.log_rank_0"
        ), patch("primus.core.runtime.train_runtime.merge_namespace"), patch.object(
            runtime, "_run_phase_patches"
        ):
            runtime._initialize_trainer()

        # trainer_class must have been resolved from module_config.params and forwarded.
        mock_adapter.load_trainer_class.assert_called_once_with(
            stage="pretrain", trainer_class="FluxPretrainTrainer"
        )

    def test_initialize_trainer_top_level_takes_precedence_over_params(self):
        """Test that top-level trainer_class takes precedence over params.trainer_class."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        mock_adapter = Mock()
        mock_adapter.prepare_backend = Mock()
        mock_adapter.convert_config = Mock(return_value=SimpleNamespace())
        mock_trainer_class = Mock()
        mock_trainer_class.__name__ = "TopLevelTrainer"
        mock_trainer_class.return_value = Mock()
        mock_adapter.load_trainer_class = Mock(return_value=mock_trainer_class)

        # Both top-level and params have trainer_class (top-level should win due to if/elif)
        runtime.ctx = SimpleNamespace(
            config_path=Path("dummy.yaml"),
            data_path=Path("./data"),
            module_name="pre_trainer",
            primus_config=SimpleNamespace(),
            rank=0,
            world_size=1,
            master_addr="localhost",
            master_port="12345",
            module_config=SimpleNamespace(
                framework="megatron",
                trainer_class="TopLevelTrainer",  # Top-level (checked first)
                params=SimpleNamespace(
                    stage="pretrain",
                    trainer_class="ParamsTrainer",  # In params (should be ignored)
                ),
            ),
            framework="megatron",
            adapter=mock_adapter,
        )

        with patch("primus.core.runtime.train_runtime.log_dict_aligned"), patch(
            "primus.core.runtime.train_runtime.log_rank_0"
        ), patch("primus.core.runtime.train_runtime.merge_namespace"), patch.object(
            runtime, "_run_phase_patches"
        ):

            runtime._initialize_trainer()

        # Verify top-level trainer_class was used (not params)
        mock_adapter.load_trainer_class.assert_called_once_with(
            stage="pretrain", trainer_class="TopLevelTrainer"  # Top-level, not ParamsTrainer
        )

    def test_initialize_trainer_falls_back_to_stage_when_no_trainer_class(self):
        """Test fallback to stage-based selection when trainer_class not specified."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        mock_adapter = Mock()
        mock_adapter.prepare_backend = Mock()
        mock_adapter.convert_config = Mock(return_value=SimpleNamespace())
        mock_trainer_class = Mock()
        mock_trainer_class.__name__ = "MockTrainer"
        mock_trainer_class.return_value = Mock()
        mock_adapter.load_trainer_class = Mock(return_value=mock_trainer_class)

        # No trainer_class specified anywhere
        runtime.ctx = SimpleNamespace(
            config_path=Path("dummy.yaml"),
            data_path=Path("./data"),
            module_name="pre_trainer",
            primus_config=SimpleNamespace(),
            rank=0,
            world_size=1,
            master_addr="localhost",
            master_port="12345",
            module_config=SimpleNamespace(
                framework="megatron",
                params=SimpleNamespace(stage="pretrain"),
            ),
            framework="megatron",
            adapter=mock_adapter,
        )

        with patch("primus.core.runtime.train_runtime.log_dict_aligned"), patch(
            "primus.core.runtime.train_runtime.log_rank_0"
        ) as mock_log, patch("primus.core.runtime.train_runtime.merge_namespace"), patch.object(
            runtime, "_run_phase_patches"
        ):

            runtime._initialize_trainer()

        # Verify fallback to stage-based selection. When trainer_class is falsy
        # the kwarg is omitted entirely (see train_runtime.py:390-392), so the
        # adapter is called with stage only.
        mock_adapter.load_trainer_class.assert_called_once_with(stage="pretrain")

        # Verify logging indicates fallback
        log_calls = [str(call) for call in mock_log.call_args_list]
        assert any("trainer_class not found" in str(call) for call in log_calls)

    def test_initialize_trainer_handles_empty_string_trainer_class(self):
        """Test that empty string trainer_class falls back to stage."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        mock_adapter = Mock()
        mock_adapter.prepare_backend = Mock()
        mock_adapter.convert_config = Mock(return_value=SimpleNamespace())
        mock_trainer_class = Mock()
        mock_trainer_class.__name__ = "MockTrainer"
        mock_trainer_class.return_value = Mock()
        mock_adapter.load_trainer_class = Mock(return_value=mock_trainer_class)

        # Empty string trainer_class (falsy, should fall back)
        runtime.ctx = SimpleNamespace(
            config_path=Path("dummy.yaml"),
            data_path=Path("./data"),
            module_name="pre_trainer",
            primus_config=SimpleNamespace(),
            rank=0,
            world_size=1,
            master_addr="localhost",
            master_port="12345",
            module_config=SimpleNamespace(
                framework="megatron",
                trainer_class="",  # Empty string (falsy)
                params=SimpleNamespace(stage="pretrain"),
            ),
            framework="megatron",
            adapter=mock_adapter,
        )

        with patch("primus.core.runtime.train_runtime.log_dict_aligned"), patch(
            "primus.core.runtime.train_runtime.log_rank_0"
        ), patch("primus.core.runtime.train_runtime.merge_namespace"), patch.object(
            runtime, "_run_phase_patches"
        ):

            runtime._initialize_trainer()

        # Should fall back to stage (empty string is falsy): the trainer_class
        # kwarg is omitted entirely (see train_runtime.py:390-392).
        mock_adapter.load_trainer_class.assert_called_once_with(stage="pretrain")


class TestPrimusRuntimeLifecycle(PrimusUT):
    """Tests for PrimusRuntime trainer lifecycle execution."""

    def _build_args(self, config: str = "examples/megatron/exp_pretrain.yaml") -> argparse.Namespace:
        return argparse.Namespace(config=config, data_path="./data", backend_path=None)

    def test_run_trainer_lifecycle_applies_patches_before_train(self):
        """Test that patches are applied in before_train phase before train() is called."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        train_called = []
        patch_applied = []

        class MockTrainer:
            def setup(self):
                pass

            def init(self):
                pass

            def train(self):
                train_called.append(1)
                # Verify patch was applied before train
                assert len(patch_applied) > 0

            def cleanup(self, on_error=False):
                pass

        mock_trainer = MockTrainer()
        runtime.ctx = SimpleNamespace(
            trainer=mock_trainer, backend_args=SimpleNamespace(), runtime_state=None
        )

        def mock_run_phase_patches(phase, backend_args=None, runtime_state=None):
            if phase == "before_train":
                patch_applied.append(1)

        runtime._run_phase_patches = mock_run_phase_patches

        with patch("primus.core.runtime.train_runtime.log_rank_0"):
            runtime._run_trainer_lifecycle()

        # Verify patch was applied before train
        assert len(patch_applied) == 1
        assert len(train_called) == 1

    def test_run_trainer_lifecycle_passes_backend_args_to_patches(self):
        """Test that backend_args are passed to patch phases."""
        args = self._build_args()
        runtime = PrimusRuntime(args=args)

        backend_args_received = []

        class MockTrainer:
            def setup(self):
                pass

            def init(self):
                pass

            def train(self):
                pass

            def cleanup(self, on_error=False):
                pass

        mock_trainer = MockTrainer()
        test_backend_args = SimpleNamespace(test_param="value")
        runtime.ctx = SimpleNamespace(
            trainer=mock_trainer,
            backend_args=test_backend_args,
            runtime_state=None,
        )

        def mock_run_phase_patches(phase, backend_args=None, runtime_state=None):
            backend_args_received.append(backend_args)

        runtime._run_phase_patches = mock_run_phase_patches

        with patch("primus.core.runtime.train_runtime.log_rank_0"):
            runtime._run_trainer_lifecycle()

        # Verify backend_args were passed to all patch phases
        assert len(backend_args_received) == 3
        assert all(args is test_backend_args for args in backend_args_received)


if __name__ == "__main__":
    unittest.main()

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MegatronAdapter.

Focus areas:
    1. Version detection must succeed (fail fast on error)
    2. Config conversion produces valid Megatron args
    3. Trainer class loading from registry
    4. Backend preparation workflow
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer


class TestMegatronAdapterVersionDetection:
    """Test that version detection uses AST parsing without executing __init__.py."""

    def test_detect_version_without_executing_init(self, tmp_path, monkeypatch):
        """
        Test that detect_backend_version uses AST parsing and does NOT execute
        any __init__.py files in the megatron package hierarchy.
        """
        # Create a fake megatron package structure with __init__.py that would fail if executed
        megatron_dir = tmp_path / "megatron"
        core_dir = megatron_dir / "core"
        core_dir.mkdir(parents=True)

        # Create __init__.py files that raise an error if executed
        (megatron_dir / "__init__.py").write_text(
            'raise RuntimeError("megatron/__init__.py should NOT be executed!")'
        )
        (core_dir / "__init__.py").write_text(
            'raise RuntimeError("megatron/core/__init__.py should NOT be executed!")'
        )

        # Create a valid package_info.py with version info
        (core_dir / "package_info.py").write_text(
            """
MAJOR = 0
MINOR = 15
PATCH = 0
PRE_RELEASE = "rc8"

__version__ = f"{MAJOR}.{MINOR}.{PATCH}{PRE_RELEASE}"
"""
        )

        # Prepend tmp_path to sys.path so it's found first
        import sys

        monkeypatch.syspath_prepend(str(tmp_path))

        # Remove any cached megatron modules to ensure clean state
        modules_to_remove = [k for k in sys.modules if k.startswith("megatron")]
        for mod in modules_to_remove:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        adapter = MegatronAdapter()
        version = adapter.detect_backend_version()

        # If we get here without RuntimeError, __init__.py files were NOT executed
        assert version == "0.15.0rc8"

        # Double-check: megatron modules should NOT be in sys.modules
        assert "megatron" not in sys.modules, "megatron should not be imported"
        assert "megatron.core" not in sys.modules, "megatron.core should not be imported"
        assert (
            "megatron.core.package_info" not in sys.modules
        ), "megatron.core.package_info should not be imported"

    def test_detect_version_parses_version_correctly(self, tmp_path, monkeypatch):
        """Test that version string is correctly assembled from MAJOR, MINOR, PATCH, PRE_RELEASE."""
        megatron_dir = tmp_path / "megatron" / "core"
        megatron_dir.mkdir(parents=True)

        # Test without PRE_RELEASE
        (megatron_dir / "package_info.py").write_text(
            """
MAJOR = 1
MINOR = 2
PATCH = 3
"""
        )

        import sys

        monkeypatch.syspath_prepend(str(tmp_path))
        modules_to_remove = [k for k in sys.modules if k.startswith("megatron")]
        for mod in modules_to_remove:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        adapter = MegatronAdapter()
        version = adapter.detect_backend_version()

        assert version == "1.2.3"

    def test_detect_version_not_found_raises_error(self, tmp_path, monkeypatch):
        """Test that RuntimeError is raised when package_info.py cannot be found."""
        import sys

        # Clear sys.path to simulate missing megatron
        monkeypatch.setattr(sys, "path", [str(tmp_path)])
        modules_to_remove = [k for k in sys.modules if k.startswith("megatron")]
        for mod in modules_to_remove:
            monkeypatch.delitem(sys.modules, mod, raising=False)

        adapter = MegatronAdapter()

        with pytest.raises(RuntimeError) as exc_info:
            adapter.detect_backend_version()

        assert "Cannot locate" in str(exc_info.value)


class TestMegatronAdapterTrainerLoading:
    """Test trainer class loading."""

    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_load_trainer_class_success(self, mock_log):
        """Test successful trainer class loading."""
        adapter = MegatronAdapter()
        result = adapter.load_trainer_class()

        # Should return the actual MegatronPretrainTrainer class
        from primus.backends.megatron.megatron_pretrain_trainer import (
            MegatronPretrainTrainer,
        )

        assert result == MegatronPretrainTrainer

    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_load_sft_trainer_class_success(self, mock_log):
        """Test successful SFT trainer class loading."""
        adapter = MegatronAdapter()
        result = adapter.load_trainer_class(stage="sft")

        assert result == MegatronSFTTrainer

    def test_load_trainer_class_invalid_stage(self):
        """Test error handling for an unregistered stage."""
        adapter = MegatronAdapter()

        with pytest.raises(RuntimeError) as exc_info:
            adapter.load_trainer_class(stage="invalid_stage")

        assert "backend trainer not registered" in str(exc_info.value)


class TestMegatronAdapterDynamicTrainerLoading:
    """Test dynamic trainer class loading by name."""

    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_load_trainer_class_by_name_takes_precedence_over_stage(self, mock_log):
        """Test that trainer_class parameter takes precedence over stage."""
        adapter = MegatronAdapter()

        # Even with invalid stage, trainer_class should work
        result = adapter.load_trainer_class(stage="invalid_stage", trainer_class="MegatronPretrainTrainer")

        from primus.backends.megatron.megatron_pretrain_trainer import (
            MegatronPretrainTrainer,
        )

        assert result == MegatronPretrainTrainer

    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_load_trainer_class_fallback_tries_multiple_paths(self, mock_log):
        """Test fallback tries multiple import paths in order for unregistered trainers."""
        import importlib

        # Use a trainer name NOT in MEGATRON_TRAINERS to exercise the fallback
        mock_trainer_class = type("CustomExperimentalTrainer", (), {})
        mock_module = Mock()
        mock_module.CustomExperimentalTrainer = mock_trainer_class

        import_calls = []

        def side_effect(module_name, *args, **kwargs):
            import_calls.append(module_name)
            if len(import_calls) <= 1:
                raise ImportError(f"Path {len(import_calls)} failed")
            return mock_module

        with patch.object(importlib, "import_module", side_effect=side_effect):
            adapter = MegatronAdapter()
            result = adapter.load_trainer_class(trainer_class="CustomExperimentalTrainer")

        assert result == mock_trainer_class
        assert len(import_calls) == 2
        assert "primus.backends.megatron.customexperimentaltrainer" in import_calls[0].lower()

    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_load_trainer_class_fallback_all_paths_fail(self, mock_log):
        """Test error when all fallback paths fail."""
        import importlib

        # All import attempts fail
        def side_effect(module_name, *args, **kwargs):
            raise ImportError("Module not found")

        with patch.object(importlib, "import_module", side_effect=side_effect):
            adapter = MegatronAdapter()

            with pytest.raises(ValueError) as exc_info:
                adapter.load_trainer_class(trainer_class="NonExistentTrainer")

        error_msg = str(exc_info.value)
        # Verify exact error message format from implementation
        assert "Trainer class 'NonExistentTrainer' not found" in error_msg
        assert "Available trainers:" in error_msg
        assert "Hint:" in error_msg
        assert "MEGATRON_TRAINERS" in error_msg or "standard locations" in error_msg

    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_load_trainer_class_falsy_values_fall_back_to_stage(self, mock_log):
        """Test that falsy trainer_class values (None, empty string) fall back to stage-based selection."""
        adapter = MegatronAdapter()

        from primus.backends.megatron.megatron_pretrain_trainer import (
            MegatronPretrainTrainer,
        )

        # Test None falls back to stage
        result = adapter.load_trainer_class(stage="pretrain", trainer_class=None)
        assert result == MegatronPretrainTrainer

        # Test empty string falls back to stage (falsy)
        result = adapter.load_trainer_class(stage="pretrain", trainer_class="")
        assert result == MegatronPretrainTrainer

    def test_load_trainer_class_registry_import_error_provides_context(self):
        """Test that import errors from registry provide helpful context."""
        import importlib

        def side_effect(module_name, *args, **kwargs):
            raise ImportError("Module not found")

        with patch.object(importlib, "import_module", side_effect=side_effect):
            adapter = MegatronAdapter()

            with pytest.raises(ImportError) as exc_info:
                adapter.load_trainer_class(trainer_class="MegatronPretrainTrainer")

        error_msg = str(exc_info.value)
        # Verify exact error message format from implementation
        assert "Failed to load trainer class 'MegatronPretrainTrainer'" in error_msg
        assert "Hint:" in error_msg
        assert "Check that the module exists" in error_msg

    def test_load_trainer_class_registry_attribute_error_provides_context(self):
        """Test that attribute errors from registry provide helpful context."""
        import importlib

        # Module imports but class doesn't exist
        mock_module = Mock()
        del mock_module.MegatronPretrainTrainer  # Ensure attribute doesn't exist

        def side_effect(module_name, *args, **kwargs):
            return mock_module

        with patch.object(importlib, "import_module", side_effect=side_effect):
            adapter = MegatronAdapter()

            with pytest.raises(ImportError) as exc_info:
                adapter.load_trainer_class(trainer_class="MegatronPretrainTrainer")

        error_msg = str(exc_info.value)
        assert "Failed to load trainer class" in error_msg
        assert "Hint:" in error_msg


class TestMegatronAdapterIntegration:
    """Integration tests for complete adapter workflow."""

    @patch("primus.core.utils.module_utils.log_rank_0")
    @patch("primus.backends.megatron.megatron_adapter.MegatronAdapter.detect_backend_version")
    @patch("primus.backends.megatron.megatron_adapter.MegatronArgBuilder")
    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_full_workflow_success(self, mock_log, mock_builder_class, mock_detect, mock_base_log):
        """Test complete adapter workflow from config to trainer."""
        # Setup mocks
        mock_detect.return_value = "0.15.0rc8"

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_args = SimpleNamespace(micro_batch_size=4)
        mock_builder.finalize.return_value = mock_args

        mock_params = {"micro_batch_size": 4}

        adapter = MegatronAdapter()

        # 1. Prepare backend
        adapter.prepare_backend(Mock())

        # 2. Convert config
        backend_args = adapter.convert_config(mock_params)
        assert backend_args.micro_batch_size == 4

        # 3. Load trainer class
        from primus.backends.megatron.megatron_pretrain_trainer import (
            MegatronPretrainTrainer,
        )

        trainer_class = adapter.load_trainer_class()
        assert trainer_class == MegatronPretrainTrainer

        # Verify version detection worked
        version = adapter.detect_backend_version()
        assert version == "0.15.0rc8"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

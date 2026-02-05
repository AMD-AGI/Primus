###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
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


class TestMegatronAdapterConfigConversion:
    """Test configuration conversion from Primus to Megatron."""

    @patch("primus.backends.megatron.megatron_adapter.MegatronArgBuilder")
    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_convert_config_basic(self, mock_log, mock_builder_class):
        """Test basic config conversion workflow."""
        # Setup mock builder
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder

        # Mock finalize to return SimpleNamespace with args
        mock_args = SimpleNamespace(
            micro_batch_size=4,
            global_batch_size=32,
            seq_length=2048,
        )
        mock_builder.finalize.return_value = mock_args

        # Create mock params
        mock_params = {
            "micro_batch_size": 4,
            "global_batch_size": 32,
            "seq_length": 2048,
        }

        # Test conversion
        adapter = MegatronAdapter()
        result = adapter.convert_config(mock_params)

        # Verify builder was called correctly
        mock_builder.update.assert_called_once_with(mock_params)
        mock_builder.finalize.assert_called_once()

        # Verify result
        assert result == mock_args
        assert result.micro_batch_size == 4
        assert result.global_batch_size == 32
        assert result.seq_length == 2048

    @patch("primus.backends.megatron.megatron_adapter.MegatronArgBuilder")
    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_convert_config_empty_params(self, mock_log, mock_builder_class):
        """Test config conversion with empty params dict."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.finalize.return_value = SimpleNamespace()

        mock_params = {}

        adapter = MegatronAdapter()
        result = adapter.convert_config(mock_params)

        mock_builder.update.assert_called_once_with(mock_params)
        assert isinstance(result, SimpleNamespace)


class TestMegatronAdapterTrainerLoading:
    """Test trainer class loading from BackendRegistry."""

    @patch("primus.backends.megatron.megatron_adapter.BackendRegistry")
    def test_load_trainer_class_success(self, mock_registry):
        """Test successful trainer class loading."""
        # Mock trainer class
        mock_trainer_class = type("MegatronPretrainTrainer", (), {})
        mock_registry.get_trainer_class.return_value = mock_trainer_class

        adapter = MegatronAdapter()
        result = adapter.load_trainer_class()

        mock_registry.get_trainer_class.assert_called_once_with("megatron", stage="pretrain")
        assert result == mock_trainer_class

    @patch("primus.backends.megatron.megatron_adapter.BackendRegistry")
    def test_load_trainer_class_not_registered(self, mock_registry):
        """Test error handling when trainer not registered."""
        mock_registry.get_trainer_class.side_effect = ValueError("Backend not registered")

        adapter = MegatronAdapter()

        with pytest.raises(RuntimeError) as exc_info:
            adapter.load_trainer_class()

        error_msg = str(exc_info.value)
        assert "megatron" in error_msg
        assert "not registered" in error_msg


class TestMegatronAdapterBackendPreparation:
    """Test backend preparation workflow."""

    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    @patch("primus.backends.megatron.megatron_adapter.BackendRegistry")
    def test_prepare_backend_success(self, mock_registry, mock_log):
        """Test successful backend preparation."""
        adapter = MegatronAdapter()
        mock_config = Mock()

        # Should not raise
        adapter.prepare_backend(mock_config)

        # Verify setup was called
        mock_registry.run_setup.assert_called_once_with("megatron")

        # Verify logging
        mock_log.assert_called()
        log_message = mock_log.call_args[0][0]
        assert "Backend prepared" in log_message


class TestMegatronAdapterInitialization:
    """Test MegatronAdapter initialization."""

    def test_initialization_default(self):
        """Test adapter initialization with default framework name."""
        adapter = MegatronAdapter()
        assert adapter.framework == "megatron"

    def test_initialization_custom_framework(self):
        """Test adapter initialization with custom framework name."""
        adapter = MegatronAdapter(framework="custom_megatron")
        assert adapter.framework == "custom_megatron"


class TestMegatronAdapterIntegration:
    """Integration tests for complete adapter workflow."""

    @patch("primus.backends.megatron.megatron_adapter.MegatronAdapter.detect_backend_version")
    @patch("primus.backends.megatron.megatron_adapter.BackendRegistry")
    @patch("primus.backends.megatron.megatron_adapter.MegatronArgBuilder")
    @patch("primus.backends.megatron.megatron_adapter.log_rank_0")
    def test_full_workflow_success(self, mock_log, mock_builder_class, mock_registry, mock_detect):
        """Test complete adapter workflow from config to trainer."""
        # Setup mocks
        mock_detect.return_value = "0.15.0rc8"

        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_args = SimpleNamespace(micro_batch_size=4)
        mock_builder.finalize.return_value = mock_args

        mock_trainer_class = type("MegatronTrainer", (), {})
        mock_registry.get_trainer_class.return_value = mock_trainer_class

        mock_module_config = Mock()
        mock_module_config.params = {"micro_batch_size": 4}

        adapter = MegatronAdapter()

        # 1. Prepare backend
        adapter.prepare_backend(mock_module_config)

        # 2. Convert config
        backend_args = adapter.convert_config(mock_module_config.params)
        assert backend_args.micro_batch_size == 4

        # 3. Load trainer class
        trainer_class = adapter.load_trainer_class()
        assert trainer_class == mock_trainer_class

        # Verify version detection worked
        version = adapter.detect_backend_version()
        assert version == "0.15.0rc8"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

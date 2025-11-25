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

from primus.backends.megatron.adapters.megatron_adapter import MegatronAdapter


class TestMegatronAdapterVersionDetection:
    """Test version detection with strict fail-fast behavior."""

    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronAdapter.detect_backend_version")
    def test_detect_version_success(self, mock_detect):
        """Test successful version detection from megatron.core.package_info."""
        mock_detect.return_value = "0.15.0rc8"

        adapter = MegatronAdapter()
        version = adapter.detect_backend_version()

        assert version == "0.15.0rc8"

    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronAdapter.detect_backend_version")
    def test_detect_version_failure_raises_error(self, mock_detect):
        """Test that version detection failure raises RuntimeError (fail fast)."""
        mock_detect.side_effect = RuntimeError(
            "Failed to detect Megatron-LM version. "
            "Please ensure Megatron-LM is properly installed and "
            "megatron.core.package_info is available."
        )

        adapter = MegatronAdapter()

        with pytest.raises(RuntimeError) as exc_info:
            adapter.detect_backend_version()

        error_msg = str(exc_info.value)
        assert "Failed to detect Megatron-LM version" in error_msg
        assert "megatron.core.package_info" in error_msg

    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronAdapter.detect_backend_version")
    def test_detect_version_no_fallback_to_unknown(self, mock_detect):
        """Test that version detection does NOT fallback to 'unknown'."""
        # This is a critical test: we must fail fast, not return "unknown"
        mock_detect.side_effect = RuntimeError("Failed to detect Megatron-LM version")

        adapter = MegatronAdapter()

        # Should raise RuntimeError, NOT return "unknown"
        with pytest.raises(RuntimeError):
            adapter.detect_backend_version()

    def test_detect_version_with_real_megatron_if_available(self):
        """Test version detection with real Megatron if available."""
        try:
            adapter = MegatronAdapter()
            version = adapter.detect_backend_version()

            # If we get here, Megatron is installed and version was detected
            assert isinstance(version, str)
            assert len(version) > 0
            assert version != "unknown"  # Critical: should never return "unknown"
        except RuntimeError as e:
            # If Megatron is not available, should raise RuntimeError
            assert "Failed to detect Megatron-LM version" in str(e)


class TestMegatronAdapterConfigConversion:
    """Test configuration conversion from Primus to Megatron."""

    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronArgBuilder")
    @patch("primus.backends.megatron.adapters.megatron_adapter.log_rank_0")
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

        # Create mock module_config
        mock_module_config = Mock()
        mock_module_config.params = {
            "micro_batch_size": 4,
            "global_batch_size": 32,
            "seq_length": 2048,
        }

        # Test conversion
        adapter = MegatronAdapter()
        result = adapter.convert_config(mock_module_config)

        # Verify builder was called correctly
        mock_builder.update.assert_called_once_with(mock_module_config.params)
        mock_builder.finalize.assert_called_once()

        # Verify result
        assert result == mock_args
        assert result.micro_batch_size == 4
        assert result.global_batch_size == 32
        assert result.seq_length == 2048

    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronArgBuilder")
    @patch("primus.backends.megatron.adapters.megatron_adapter.log_rank_0")
    def test_convert_config_empty_params(self, mock_log, mock_builder_class):
        """Test config conversion with empty params dict."""
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.finalize.return_value = SimpleNamespace()

        mock_module_config = Mock()
        mock_module_config.params = {}

        adapter = MegatronAdapter()
        result = adapter.convert_config(mock_module_config)

        mock_builder.update.assert_called_once_with({})
        assert isinstance(result, SimpleNamespace)


class TestMegatronAdapterTrainerLoading:
    """Test trainer class loading from BackendRegistry."""

    @patch("primus.backends.megatron.adapters.megatron_adapter.BackendRegistry")
    def test_load_trainer_class_success(self, mock_registry):
        """Test successful trainer class loading."""
        # Mock trainer class
        mock_trainer_class = type("MegatronPretrainTrainer", (), {})
        mock_registry.get_trainer_class.return_value = mock_trainer_class

        adapter = MegatronAdapter()
        result = adapter.load_trainer_class()

        mock_registry.get_trainer_class.assert_called_once_with("megatron")
        assert result == mock_trainer_class

    @patch("primus.backends.megatron.adapters.megatron_adapter.BackendRegistry")
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

    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronAdapter.detect_backend_version")
    @patch("primus.backends.megatron.adapters.megatron_adapter.log_rank_0")
    @patch("primus.backends.megatron.adapters.megatron_adapter.BackendRegistry")
    def test_prepare_backend_success(self, mock_registry, mock_log, mock_detect):
        """Test successful backend preparation."""
        mock_detect.return_value = "0.15.0rc8"

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
        assert "0.15.0rc8" in log_message

    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronAdapter.detect_backend_version")
    @patch("primus.backends.megatron.adapters.megatron_adapter.BackendRegistry")
    def test_prepare_backend_version_detection_fails(self, mock_registry, mock_detect):
        """Test that prepare_backend fails if version detection fails."""
        mock_detect.side_effect = RuntimeError("Failed to detect Megatron-LM version")

        adapter = MegatronAdapter()
        mock_config = Mock()

        # prepare_backend calls detect_backend_version, which should raise
        with pytest.raises(RuntimeError) as exc_info:
            adapter.prepare_backend(mock_config)

        assert "Failed to detect Megatron-LM version" in str(exc_info.value)


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

    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronAdapter.detect_backend_version")
    @patch("primus.backends.megatron.adapters.megatron_adapter.BackendRegistry")
    @patch("primus.backends.megatron.adapters.megatron_adapter.MegatronArgBuilder")
    @patch("primus.backends.megatron.adapters.megatron_adapter.log_rank_0")
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
        backend_args = adapter.convert_config(mock_module_config)
        assert backend_args.micro_batch_size == 4

        # 3. Load trainer class
        trainer_class = adapter.load_trainer_class()
        assert trainer_class == mock_trainer_class

        # Verify version detection worked
        version = adapter.detect_backend_version()
        assert version == "0.15.0rc8"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

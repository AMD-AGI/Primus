###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for global_vars module.

Focus areas:
    1. upload_mlflow_artifacts parameter passing
    2. Handling of None mlflow_writer
    3. Integration with upload_artifacts_to_mlflow
"""

from unittest.mock import Mock, patch

import pytest

from primus.backends.megatron.training.global_vars import upload_mlflow_artifacts


class TestUploadMLflowArtifacts:
    """Test upload_mlflow_artifacts function."""

    @patch("primus.backends.megatron.training.global_vars.get_mlflow_writer")
    @patch("primus.backends.megatron.training.global_vars.upload_artifacts_to_mlflow")
    def test_parameter_passing_all_defaults(self, mock_upload_artifacts, mock_get_writer):
        """Test that all parameters are passed correctly with defaults."""
        # Setup mock mlflow_writer
        mock_writer = Mock()
        mock_get_writer.return_value = mock_writer
        mock_upload_artifacts.return_value = {"traces": 2, "logs": 3, "tracelens_reports": 1}

        # Call with default parameters
        result = upload_mlflow_artifacts()

        # Verify get_mlflow_writer was called
        mock_get_writer.assert_called_once()

        # Verify upload_artifacts_to_mlflow was called with correct default parameters
        mock_upload_artifacts.assert_called_once_with(
            mlflow_writer=mock_writer,
            tensorboard_dir=None,
            exp_root_path=None,
            upload_traces=True,
            upload_logs=True,
            upload_tracelens_report=False,
            tracelens_ranks=None,
            tracelens_max_reports=None,
            tracelens_output_format="all",
        )

        # Verify return value is passed through
        assert result == {"traces": 2, "logs": 3, "tracelens_reports": 1}

    @patch("primus.backends.megatron.training.global_vars.get_mlflow_writer")
    @patch("primus.backends.megatron.training.global_vars.upload_artifacts_to_mlflow")
    def test_parameter_passing_custom_values(self, mock_upload_artifacts, mock_get_writer):
        """Test that custom parameter values are passed correctly."""
        # Setup mock mlflow_writer
        mock_writer = Mock()
        mock_get_writer.return_value = mock_writer
        mock_upload_artifacts.return_value = {"traces": 5, "logs": 2, "tracelens_reports": 3}

        # Call with custom parameters
        result = upload_mlflow_artifacts(
            tensorboard_dir="/path/to/tensorboard",
            exp_root_path="/path/to/exp",
            upload_traces=False,
            upload_logs=True,
            upload_tracelens_report=True,
            tracelens_ranks=[0, 1, 2],
            tracelens_max_reports=10,
            tracelens_output_format="csv",
        )

        # Verify upload_artifacts_to_mlflow was called with custom parameters
        mock_upload_artifacts.assert_called_once_with(
            mlflow_writer=mock_writer,
            tensorboard_dir="/path/to/tensorboard",
            exp_root_path="/path/to/exp",
            upload_traces=False,
            upload_logs=True,
            upload_tracelens_report=True,
            tracelens_ranks=[0, 1, 2],
            tracelens_max_reports=10,
            tracelens_output_format="csv",
        )

        # Verify return value is passed through
        assert result == {"traces": 5, "logs": 2, "tracelens_reports": 3}

    @patch("primus.backends.megatron.training.global_vars.get_mlflow_writer")
    @patch("primus.backends.megatron.training.global_vars.upload_artifacts_to_mlflow")
    def test_none_mlflow_writer_returns_none(self, mock_upload_artifacts, mock_get_writer):
        """Test that None is returned when mlflow_writer is None."""
        # Setup mock to return None (MLflow not enabled)
        mock_get_writer.return_value = None

        # Call function
        result = upload_mlflow_artifacts(
            tensorboard_dir="/path/to/tensorboard",
            exp_root_path="/path/to/exp",
        )

        # Verify get_mlflow_writer was called
        mock_get_writer.assert_called_once()

        # Verify upload_artifacts_to_mlflow was NOT called
        mock_upload_artifacts.assert_not_called()

        # Verify None is returned
        assert result is None

    @patch("primus.backends.megatron.training.global_vars.get_mlflow_writer")
    @patch("primus.backends.megatron.training.global_vars.upload_artifacts_to_mlflow")
    def test_none_mlflow_writer_early_return(self, mock_upload_artifacts, mock_get_writer):
        """Test early return when MLflow is not enabled."""
        # Setup mock to return None
        mock_get_writer.return_value = None

        # Call with all parameters set
        result = upload_mlflow_artifacts(
            tensorboard_dir="/path/to/tensorboard",
            exp_root_path="/path/to/exp",
            upload_traces=True,
            upload_logs=True,
            upload_tracelens_report=True,
            tracelens_ranks=[0],
            tracelens_max_reports=5,
            tracelens_output_format="xlsx",
        )

        # Verify no artifact upload was attempted
        mock_upload_artifacts.assert_not_called()

        # Verify None is returned immediately
        assert result is None

    @patch("primus.backends.megatron.training.global_vars.get_mlflow_writer")
    @patch("primus.backends.megatron.training.global_vars.upload_artifacts_to_mlflow")
    def test_integration_with_upload_artifacts_to_mlflow(self, mock_upload_artifacts, mock_get_writer):
        """Test integration with upload_artifacts_to_mlflow function."""
        # Setup mock mlflow_writer
        mock_writer = Mock()
        mock_writer.log_artifact = Mock()
        mock_get_writer.return_value = mock_writer

        # Setup mock return value from upload_artifacts_to_mlflow
        expected_result = {
            "traces": 10,
            "logs": 5,
            "tracelens_reports": 2,
        }
        mock_upload_artifacts.return_value = expected_result

        # Call with tracelens enabled
        result = upload_mlflow_artifacts(
            tensorboard_dir="/path/to/tensorboard",
            exp_root_path="/path/to/exp",
            upload_tracelens_report=True,
            tracelens_ranks=[0, 1],
            tracelens_max_reports=5,
            tracelens_output_format="all",
        )

        # Verify the mlflow_writer instance was passed to upload_artifacts_to_mlflow
        call_args = mock_upload_artifacts.call_args
        assert call_args[1]["mlflow_writer"] == mock_writer

        # Verify tracelens parameters were passed correctly
        assert call_args[1]["upload_tracelens_report"] is True
        assert call_args[1]["tracelens_ranks"] == [0, 1]
        assert call_args[1]["tracelens_max_reports"] == 5
        assert call_args[1]["tracelens_output_format"] == "all"

        # Verify return value matches expected result
        assert result == expected_result

    @patch("primus.backends.megatron.training.global_vars.get_mlflow_writer")
    @patch("primus.backends.megatron.training.global_vars.upload_artifacts_to_mlflow")
    def test_tracelens_output_format_options(self, mock_upload_artifacts, mock_get_writer):
        """Test different tracelens_output_format values are passed correctly."""
        mock_writer = Mock()
        mock_get_writer.return_value = mock_writer
        mock_upload_artifacts.return_value = {"traces": 0, "logs": 0, "tracelens_reports": 1}

        # Test "xlsx" format
        upload_mlflow_artifacts(tracelens_output_format="xlsx")
        assert mock_upload_artifacts.call_args_list[-1][1]["tracelens_output_format"] == "xlsx"

        # Test "csv" format
        upload_mlflow_artifacts(tracelens_output_format="csv")
        assert mock_upload_artifacts.call_args_list[-1][1]["tracelens_output_format"] == "csv"

        # Test "all" format (default)
        upload_mlflow_artifacts(tracelens_output_format="all")
        assert mock_upload_artifacts.call_args_list[-1][1]["tracelens_output_format"] == "all"

        # Verify all three calls were made
        assert mock_upload_artifacts.call_count == 3

    @patch("primus.backends.megatron.training.global_vars.get_mlflow_writer")
    @patch("primus.backends.megatron.training.global_vars.upload_artifacts_to_mlflow")
    def test_empty_ranks_list(self, mock_upload_artifacts, mock_get_writer):
        """Test handling of empty ranks list."""
        mock_writer = Mock()
        mock_get_writer.return_value = mock_writer
        mock_upload_artifacts.return_value = {"traces": 0, "logs": 0, "tracelens_reports": 0}

        # Call with empty ranks list
        result = upload_mlflow_artifacts(
            tracelens_ranks=[],
            upload_tracelens_report=True,
        )

        # Verify empty list is passed through
        call_args = mock_upload_artifacts.call_args
        assert call_args[1]["tracelens_ranks"] == []

    @patch("primus.backends.megatron.training.global_vars.get_mlflow_writer")
    @patch("primus.backends.megatron.training.global_vars.upload_artifacts_to_mlflow")
    def test_upload_flags_all_false(self, mock_upload_artifacts, mock_get_writer):
        """Test when all upload flags are False."""
        mock_writer = Mock()
        mock_get_writer.return_value = mock_writer
        mock_upload_artifacts.return_value = {"traces": 0, "logs": 0, "tracelens_reports": 0}

        # Call with all upload flags set to False
        result = upload_mlflow_artifacts(
            upload_traces=False,
            upload_logs=False,
            upload_tracelens_report=False,
        )

        # Verify function was still called with correct flags
        call_args = mock_upload_artifacts.call_args
        assert call_args[1]["upload_traces"] is False
        assert call_args[1]["upload_logs"] is False
        assert call_args[1]["upload_tracelens_report"] is False

        # Verify result shows no uploads
        assert result == {"traces": 0, "logs": 0, "tracelens_reports": 0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

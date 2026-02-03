###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MLflow artifact upload utilities.

Focus areas:
    1. File discovery logic (_get_all_trace_files, _get_all_log_files)
    2. Upload functions with various scenarios (no files, multiple files, errors)
    3. Glob escaping for special characters in paths
    4. Relative path handling for artifact organization
"""

from unittest.mock import MagicMock, patch

from primus.backends.megatron.training.mlflow_artifacts import (
    _get_all_log_files,
    _get_all_trace_files,
    upload_artifacts_to_mlflow,
    upload_log_files_to_mlflow,
    upload_trace_files_to_mlflow,
)


class TestGetAllTraceFiles:
    """Test trace file discovery logic."""

    def test_finds_json_trace_files(self, tmp_path):
        """Should find .pt.trace.json files."""
        trace_file = tmp_path / "rank_0_step_2.pt.trace.json"
        trace_file.touch()

        files = _get_all_trace_files(str(tmp_path))

        assert len(files) == 1
        assert str(trace_file) in files

    def test_finds_gzipped_trace_files(self, tmp_path):
        """Should find .pt.trace.json.gz files."""
        trace_file = tmp_path / "rank_0_step_2.pt.trace.json.gz"
        trace_file.touch()

        files = _get_all_trace_files(str(tmp_path))

        assert len(files) == 1
        assert str(trace_file) in files

    def test_finds_nested_trace_files(self, tmp_path):
        """Should find trace files in subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        trace_file = subdir / "rank_1.pt.trace.json"
        trace_file.touch()

        files = _get_all_trace_files(str(tmp_path))

        assert len(files) == 1
        assert str(trace_file) in files

    def test_returns_empty_for_nonexistent_dir(self):
        """Should return empty list for non-existent directory."""
        files = _get_all_trace_files("/nonexistent/path")

        assert files == []

    def test_returns_empty_for_none(self):
        """Should return empty list for None input."""
        files = _get_all_trace_files(None)

        assert files == []

    def test_handles_special_characters_in_path(self, tmp_path):
        """Should handle paths with special glob characters like []."""
        # Create directory with brackets in name (common in experiment names)
        special_dir = tmp_path / "exp[rank0]_test"
        special_dir.mkdir()
        trace_file = special_dir / "trace.pt.trace.json"
        trace_file.touch()

        files = _get_all_trace_files(str(special_dir))

        assert len(files) == 1
        assert str(trace_file) in files

    def test_deduplicates_files(self, tmp_path):
        """Should not return duplicate file paths."""
        trace_file = tmp_path / "rank_0.pt.trace.json"
        trace_file.touch()

        files = _get_all_trace_files(str(tmp_path))

        # Each file should appear only once
        assert len(files) == len(set(files))


class TestGetAllLogFiles:
    """Test log file discovery logic."""

    def test_finds_log_files(self, tmp_path):
        """Should find .log files in logs directory."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "training.log"
        log_file.touch()

        files = _get_all_log_files(str(tmp_path))

        assert len(files) == 1
        assert str(log_file) in files

    def test_finds_nested_log_files(self, tmp_path):
        """Should find log files in nested directories."""
        logs_dir = tmp_path / "logs" / "rank-0"
        logs_dir.mkdir(parents=True)
        log_file = logs_dir / "debug.log"
        log_file.touch()

        files = _get_all_log_files(str(tmp_path))

        assert len(files) == 1
        assert str(log_file) in files

    def test_returns_empty_when_no_logs_dir(self, tmp_path):
        """Should return empty list when logs directory doesn't exist."""
        files = _get_all_log_files(str(tmp_path))

        assert files == []

    def test_returns_empty_for_none(self):
        """Should return empty list for None input."""
        files = _get_all_log_files(None)

        assert files == []


class TestUploadTraceFilesToMlflow:
    """Test trace file upload functionality."""

    def test_returns_zero_when_no_writer(self, tmp_path):
        """Should return 0 when mlflow_writer is None."""
        count = upload_trace_files_to_mlflow(None, str(tmp_path))

        assert count == 0

    def test_returns_zero_when_no_files(self, tmp_path):
        """Should return 0 when no trace files found."""
        mlflow_mock = MagicMock()

        count = upload_trace_files_to_mlflow(mlflow_mock, str(tmp_path))

        assert count == 0
        mlflow_mock.log_artifact.assert_not_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_uploads_trace_files(self, mock_warning, mock_log, tmp_path):
        """Should upload trace files and return count."""
        trace_file = tmp_path / "rank_0.pt.trace.json"
        trace_file.touch()
        mlflow_mock = MagicMock()

        count = upload_trace_files_to_mlflow(mlflow_mock, str(tmp_path))

        assert count == 1
        mlflow_mock.log_artifact.assert_called_once()

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_handles_upload_error(self, mock_warning, mock_log, tmp_path):
        """Should continue on upload error and log warning."""
        trace_file = tmp_path / "rank_0.pt.trace.json"
        trace_file.touch()
        mlflow_mock = MagicMock()
        mlflow_mock.log_artifact.side_effect = Exception("Upload failed")

        count = upload_trace_files_to_mlflow(mlflow_mock, str(tmp_path))

        assert count == 0
        mock_warning.assert_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_preserves_relative_path(self, mock_warning, mock_log, tmp_path):
        """Should preserve subdirectory structure in artifact path."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        trace_file = subdir / "rank_0.pt.trace.json"
        trace_file.touch()
        mlflow_mock = MagicMock()

        upload_trace_files_to_mlflow(mlflow_mock, str(tmp_path))

        # Check that artifact_path includes subdirectory
        call_args = mlflow_mock.log_artifact.call_args
        assert "subdir" in call_args.kwargs.get("artifact_path", "")


class TestUploadLogFilesToMlflow:
    """Test log file upload functionality."""

    def test_returns_zero_when_no_writer(self, tmp_path):
        """Should return 0 when mlflow_writer is None."""
        count = upload_log_files_to_mlflow(None, str(tmp_path))

        assert count == 0

    def test_returns_zero_when_no_files(self, tmp_path):
        """Should return 0 when no log files found."""
        mlflow_mock = MagicMock()

        count = upload_log_files_to_mlflow(mlflow_mock, str(tmp_path))

        assert count == 0

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_uploads_log_files(self, mock_warning, mock_log, tmp_path):
        """Should upload log files and return count."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_file = logs_dir / "training.log"
        log_file.touch()
        mlflow_mock = MagicMock()

        count = upload_log_files_to_mlflow(mlflow_mock, str(tmp_path))

        assert count == 1
        mlflow_mock.log_artifact.assert_called_once()


class TestUploadArtifactsToMlflow:
    """Test main artifact upload entry point."""

    def test_returns_zeros_when_no_writer(self):
        """Should return zero counts when mlflow_writer is None."""
        result = upload_artifacts_to_mlflow(None)

        assert result == {"traces": 0, "logs": 0}

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_respects_upload_traces_flag(self, mock_warning, mock_log, tmp_path):
        """Should skip trace upload when upload_traces=False."""
        trace_file = tmp_path / "rank_0.pt.trace.json"
        trace_file.touch()
        mlflow_mock = MagicMock()

        result = upload_artifacts_to_mlflow(
            mlflow_mock,
            tensorboard_dir=str(tmp_path),
            upload_traces=False,
            upload_logs=False,
        )

        assert result["traces"] == 0
        mlflow_mock.log_artifact.assert_not_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_warns_for_multi_node(self, mock_warning, mock_log, tmp_path, monkeypatch):
        """Should warn when multi-node training is detected."""
        monkeypatch.setenv("NNODES", "2")
        mlflow_mock = MagicMock()

        upload_artifacts_to_mlflow(mlflow_mock, tensorboard_dir=str(tmp_path))

        # Check that warning was called with multi-node message
        warning_calls = [str(call) for call in mock_warning.call_args_list]
        assert any("Multi-node" in str(call) for call in warning_calls)

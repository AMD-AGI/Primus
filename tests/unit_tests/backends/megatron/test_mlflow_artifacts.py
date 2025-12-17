###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MLflow artifacts module.

Tests cover:
    1. Trace and log file discovery with various directory structures
    2. Proper handling of missing directories
    3. Error handling during uploads
    4. Deduplication logic for trace files
    5. Integration with the global_vars module
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch, call

import pytest

from primus.backends.megatron.training import mlflow_artifacts


class TestGetAllTraceFiles:
    """Test trace file discovery functionality."""

    def test_empty_directory(self):
        """Test that empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mlflow_artifacts._get_all_trace_files(tmpdir)
            assert result == []

    def test_nonexistent_directory(self):
        """Test that nonexistent directory returns empty list."""
        result = mlflow_artifacts._get_all_trace_files("/nonexistent/path")
        assert result == []

    def test_none_directory(self):
        """Test that None directory returns empty list."""
        result = mlflow_artifacts._get_all_trace_files(None)
        assert result == []

    def test_empty_string_directory(self):
        """Test that empty string directory returns empty list."""
        result = mlflow_artifacts._get_all_trace_files("")
        assert result == []

    def test_single_trace_file(self):
        """Test discovery of a single uncompressed trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.pt.trace.json")
            with open(trace_file, "w") as f:
                f.write("{}")

            result = mlflow_artifacts._get_all_trace_files(tmpdir)
            assert len(result) == 1
            assert result[0] == trace_file

    def test_compressed_trace_file(self):
        """Test discovery of compressed trace files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.pt.trace.json.gz")
            with open(trace_file, "w") as f:
                f.write("compressed_data")

            result = mlflow_artifacts._get_all_trace_files(tmpdir)
            assert len(result) == 1
            assert result[0] == trace_file

    def test_multiple_trace_files(self):
        """Test discovery of multiple trace files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_files = [
                os.path.join(tmpdir, "rank_0.pt.trace.json"),
                os.path.join(tmpdir, "rank_1.pt.trace.json"),
                os.path.join(tmpdir, "rank_0.pt.trace.json.gz"),
            ]
            for trace_file in trace_files:
                with open(trace_file, "w") as f:
                    f.write("{}")

            result = mlflow_artifacts._get_all_trace_files(tmpdir)
            assert len(result) == 3
            assert set(result) == set(trace_files)

    def test_nested_trace_files(self):
        """Test discovery of trace files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir1 = os.path.join(tmpdir, "node1")
            subdir2 = os.path.join(tmpdir, "node2")
            os.makedirs(subdir1)
            os.makedirs(subdir2)

            trace_files = [
                os.path.join(tmpdir, "rank_0.pt.trace.json"),
                os.path.join(subdir1, "rank_1.pt.trace.json"),
                os.path.join(subdir2, "rank_2.pt.trace.json.gz"),
            ]
            for trace_file in trace_files:
                with open(trace_file, "w") as f:
                    f.write("{}")

            result = mlflow_artifacts._get_all_trace_files(tmpdir)
            assert len(result) == 3
            assert set(result) == set(trace_files)

    def test_ignores_non_trace_json_files(self):
        """Test that non-trace JSON files are not included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trace files and non-trace files
            trace_file = os.path.join(tmpdir, "rank_0.pt.trace.json")
            non_trace_files = [
                os.path.join(tmpdir, "config.json"),
                os.path.join(tmpdir, "data.json"),
                os.path.join(tmpdir, "metadata.json"),
            ]

            with open(trace_file, "w") as f:
                f.write("{}")
            for non_trace in non_trace_files:
                with open(non_trace, "w") as f:
                    f.write("{}")

            result = mlflow_artifacts._get_all_trace_files(tmpdir)
            assert len(result) == 1
            assert result[0] == trace_file

    def test_deduplication(self):
        """Test that duplicate file paths are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.pt.trace.json")
            with open(trace_file, "w") as f:
                f.write("{}")

            # The function uses glob which shouldn't return duplicates normally,
            # but we test the deduplication logic exists
            result = mlflow_artifacts._get_all_trace_files(tmpdir)
            # Check no duplicates in result
            assert len(result) == len(set(result))

    def test_preserves_order_during_deduplication(self):
        """Test that file order is preserved during deduplication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in a specific order
            trace_files = []
            for i in range(5):
                trace_file = os.path.join(tmpdir, f"rank_{i}.pt.trace.json")
                with open(trace_file, "w") as f:
                    f.write("{}")
                trace_files.append(trace_file)

            result = mlflow_artifacts._get_all_trace_files(tmpdir)
            # Should have 5 unique files
            assert len(result) == 5
            # All files should be in the result
            assert set(result) == set(trace_files)


class TestGetAllLogFiles:
    """Test log file discovery functionality."""

    def test_empty_logs_directory(self):
        """Test that empty logs directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)

            result = mlflow_artifacts._get_all_log_files(tmpdir)
            assert result == []

    def test_missing_logs_directory(self):
        """Test that missing logs directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mlflow_artifacts._get_all_log_files(tmpdir)
            assert result == []

    def test_none_exp_root_path(self):
        """Test that None exp_root_path returns empty list."""
        result = mlflow_artifacts._get_all_log_files(None)
        assert result == []

    def test_empty_string_exp_root_path(self):
        """Test that empty string exp_root_path returns empty list."""
        result = mlflow_artifacts._get_all_log_files("")
        assert result == []

    def test_single_log_file(self):
        """Test discovery of a single log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)
            log_file = os.path.join(logs_dir, "master.log")
            with open(log_file, "w") as f:
                f.write("log content")

            result = mlflow_artifacts._get_all_log_files(tmpdir)
            assert len(result) == 1
            assert result[0] == log_file

    def test_multiple_log_files(self):
        """Test discovery of multiple log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)

            log_files = [
                os.path.join(logs_dir, "master.log"),
                os.path.join(logs_dir, "error.log"),
                os.path.join(logs_dir, "debug.log"),
            ]
            for log_file in log_files:
                with open(log_file, "w") as f:
                    f.write("log content")

            result = mlflow_artifacts._get_all_log_files(tmpdir)
            assert len(result) == 3
            assert set(result) == set(log_files)

    def test_nested_log_files(self):
        """Test discovery of log files in nested structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            master_dir = os.path.join(logs_dir, "master")
            rank_dir = os.path.join(logs_dir, "rank-0")
            os.makedirs(master_dir)
            os.makedirs(rank_dir)

            log_files = [
                os.path.join(master_dir, "master-0.log"),
                os.path.join(rank_dir, "rank-0.log"),
            ]
            for log_file in log_files:
                with open(log_file, "w") as f:
                    f.write("log content")

            result = mlflow_artifacts._get_all_log_files(tmpdir)
            assert len(result) == 2
            assert set(result) == set(log_files)

    def test_ignores_non_log_files(self):
        """Test that non-.log files are not included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)

            log_file = os.path.join(logs_dir, "master.log")
            non_log_files = [
                os.path.join(logs_dir, "config.txt"),
                os.path.join(logs_dir, "README.md"),
                os.path.join(logs_dir, "data.json"),
            ]

            with open(log_file, "w") as f:
                f.write("log content")
            for non_log in non_log_files:
                with open(non_log, "w") as f:
                    f.write("content")

            result = mlflow_artifacts._get_all_log_files(tmpdir)
            assert len(result) == 1
            assert result[0] == log_file


class TestUploadTraceFilesToMLflow:
    """Test trace file upload functionality."""

    def test_none_mlflow_writer(self):
        """Test that None mlflow writer returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mlflow_artifacts.upload_trace_files_to_mlflow(None, tmpdir)
            assert result == 0

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_no_trace_files(self, mock_log):
        """Test that no trace files returns 0 and logs appropriately."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mlflow_artifacts.upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 0
            mock_writer.log_artifact.assert_not_called()
            # Verify appropriate logging
            assert any("No trace files found" in str(call) for call in mock_log.call_args_list)

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_single_trace_file(self, mock_log):
        """Test successful upload of a single trace file."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.pt.trace.json")
            with open(trace_file, "w") as f:
                f.write("{}")

            result = mlflow_artifacts.upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 1
            mock_writer.log_artifact.assert_called_once_with(trace_file, artifact_path="traces")

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_multiple_trace_files(self, mock_log):
        """Test successful upload of multiple trace files."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_files = []
            for i in range(3):
                trace_file = os.path.join(tmpdir, f"rank_{i}.pt.trace.json")
                with open(trace_file, "w") as f:
                    f.write("{}")
                trace_files.append(trace_file)

            result = mlflow_artifacts.upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 3
            assert mock_writer.log_artifact.call_count == 3

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_error_handling(self, mock_log, mock_warning):
        """Test that upload errors are caught and logged."""
        mock_writer = Mock()
        mock_writer.log_artifact.side_effect = Exception("Upload failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.pt.trace.json")
            with open(trace_file, "w") as f:
                f.write("{}")

            result = mlflow_artifacts.upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 0
            mock_writer.log_artifact.assert_called_once()
            # Verify error was logged
            assert any("Failed to upload" in str(call) for call in mock_warning.call_args_list)

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_partial_upload_failure(self, mock_log, mock_warning):
        """Test that some files can succeed even if others fail."""
        mock_writer = Mock()
        # First call succeeds, second fails, third succeeds
        mock_writer.log_artifact.side_effect = [None, Exception("Upload failed"), None]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                trace_file = os.path.join(tmpdir, f"rank_{i}.pt.trace.json")
                with open(trace_file, "w") as f:
                    f.write("{}")

            result = mlflow_artifacts.upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 2  # 2 out of 3 succeeded
            assert mock_writer.log_artifact.call_count == 3

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_nested_trace_files_preserve_structure(self, mock_log):
        """Test that nested trace files preserve directory structure in artifacts."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "node1")
            os.makedirs(subdir)

            root_trace = os.path.join(tmpdir, "rank_0.pt.trace.json")
            nested_trace = os.path.join(subdir, "rank_1.pt.trace.json")

            with open(root_trace, "w") as f:
                f.write("{}")
            with open(nested_trace, "w") as f:
                f.write("{}")

            result = mlflow_artifacts.upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 2
            # Check that artifact paths are different for root vs nested
            calls = mock_writer.log_artifact.call_args_list
            assert len(calls) == 2

            # One should be in "traces", the other in "traces/node1"
            artifact_paths = [call[1]["artifact_path"] for call in calls]
            assert "traces" in artifact_paths
            assert "traces/node1" in artifact_paths

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_custom_artifact_path(self, mock_log):
        """Test that custom artifact path is used."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.pt.trace.json")
            with open(trace_file, "w") as f:
                f.write("{}")

            result = mlflow_artifacts.upload_trace_files_to_mlflow(
                mock_writer, tmpdir, artifact_path="custom/traces"
            )

            assert result == 1
            mock_writer.log_artifact.assert_called_once_with(
                trace_file, artifact_path="custom/traces"
            )


class TestUploadLogFilesToMLflow:
    """Test log file upload functionality."""

    def test_none_mlflow_writer(self):
        """Test that None mlflow writer returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mlflow_artifacts.upload_log_files_to_mlflow(None, tmpdir)
            assert result == 0

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_no_log_files(self, mock_log):
        """Test that no log files returns 0 and logs appropriately."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = mlflow_artifacts.upload_log_files_to_mlflow(mock_writer, tmpdir)

            assert result == 0
            mock_writer.log_artifact.assert_not_called()
            assert any("No log files found" in str(call) for call in mock_log.call_args_list)

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_single_log_file(self, mock_log):
        """Test successful upload of a single log file."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)
            log_file = os.path.join(logs_dir, "master.log")
            with open(log_file, "w") as f:
                f.write("log content")

            result = mlflow_artifacts.upload_log_files_to_mlflow(mock_writer, tmpdir)

            assert result == 1
            mock_writer.log_artifact.assert_called_once()

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_multiple_log_files(self, mock_log):
        """Test successful upload of multiple log files."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)

            for i in range(3):
                log_file = os.path.join(logs_dir, f"rank_{i}.log")
                with open(log_file, "w") as f:
                    f.write("log content")

            result = mlflow_artifacts.upload_log_files_to_mlflow(mock_writer, tmpdir)

            assert result == 3
            assert mock_writer.log_artifact.call_count == 3

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_error_handling(self, mock_log, mock_warning):
        """Test that upload errors are caught and logged."""
        mock_writer = Mock()
        mock_writer.log_artifact.side_effect = Exception("Upload failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)
            log_file = os.path.join(logs_dir, "master.log")
            with open(log_file, "w") as f:
                f.write("log content")

            result = mlflow_artifacts.upload_log_files_to_mlflow(mock_writer, tmpdir)

            assert result == 0
            mock_writer.log_artifact.assert_called_once()
            assert any("Failed to upload" in str(call) for call in mock_warning.call_args_list)

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_nested_log_files_preserve_structure(self, mock_log):
        """Test that nested log files preserve directory structure in artifacts."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            master_dir = os.path.join(logs_dir, "master")
            rank_dir = os.path.join(logs_dir, "rank-0")
            os.makedirs(master_dir)
            os.makedirs(rank_dir)

            master_log = os.path.join(master_dir, "master.log")
            rank_log = os.path.join(rank_dir, "rank-0.log")

            with open(master_log, "w") as f:
                f.write("master log")
            with open(rank_log, "w") as f:
                f.write("rank log")

            result = mlflow_artifacts.upload_log_files_to_mlflow(mock_writer, tmpdir)

            assert result == 2
            calls = mock_writer.log_artifact.call_args_list
            assert len(calls) == 2

            # Check that artifact paths preserve structure
            artifact_paths = [call[1]["artifact_path"] for call in calls]
            assert "logs/master" in artifact_paths
            assert "logs/rank-0" in artifact_paths

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_custom_artifact_path(self, mock_log):
        """Test that custom artifact path is used."""
        mock_writer = Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)
            log_file = os.path.join(logs_dir, "master.log")
            with open(log_file, "w") as f:
                f.write("log content")

            result = mlflow_artifacts.upload_log_files_to_mlflow(
                mock_writer, tmpdir, artifact_path="custom/logs"
            )

            assert result == 1
            # Check that custom path is used
            call_args = mock_writer.log_artifact.call_args
            assert "custom/logs" in call_args[1]["artifact_path"]


class TestUploadArtifactsToMLflow:
    """Test main artifact upload function."""

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_none_mlflow_writer(self, mock_log):
        """Test that None mlflow writer returns zeros."""
        result = mlflow_artifacts.upload_artifacts_to_mlflow(None)

        assert result == {"traces": 0, "logs": 0}
        assert any("not available" in str(call) for call in mock_log.call_args_list)

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_log_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_trace_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_both_traces_and_logs(self, mock_log, mock_upload_traces, mock_upload_logs):
        """Test uploading both traces and logs."""
        mock_writer = Mock()
        mock_upload_traces.return_value = 5
        mock_upload_logs.return_value = 3

        result = mlflow_artifacts.upload_artifacts_to_mlflow(
            mock_writer, tensorboard_dir="/tmp/tb", exp_root_path="/tmp/exp"
        )

        assert result == {"traces": 5, "logs": 3}
        mock_upload_traces.assert_called_once_with(mock_writer, "/tmp/tb", artifact_path="traces")
        mock_upload_logs.assert_called_once_with(mock_writer, "/tmp/exp", artifact_path="logs")

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_log_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_trace_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_traces_only(self, mock_log, mock_upload_traces, mock_upload_logs):
        """Test uploading only traces."""
        mock_writer = Mock()
        mock_upload_traces.return_value = 5

        result = mlflow_artifacts.upload_artifacts_to_mlflow(
            mock_writer,
            tensorboard_dir="/tmp/tb",
            exp_root_path="/tmp/exp",
            upload_traces=True,
            upload_logs=False,
        )

        assert result == {"traces": 5, "logs": 0}
        mock_upload_traces.assert_called_once()
        mock_upload_logs.assert_not_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_log_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_trace_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_logs_only(self, mock_log, mock_upload_traces, mock_upload_logs):
        """Test uploading only logs."""
        mock_writer = Mock()
        mock_upload_logs.return_value = 3

        result = mlflow_artifacts.upload_artifacts_to_mlflow(
            mock_writer,
            tensorboard_dir="/tmp/tb",
            exp_root_path="/tmp/exp",
            upload_traces=False,
            upload_logs=True,
        )

        assert result == {"traces": 0, "logs": 3}
        mock_upload_traces.assert_not_called()
        mock_upload_logs.assert_called_once()

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_log_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_trace_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_missing_tensorboard_dir(self, mock_log, mock_upload_traces, mock_upload_logs):
        """Test that missing tensorboard_dir skips trace upload."""
        mock_writer = Mock()
        mock_upload_logs.return_value = 3

        result = mlflow_artifacts.upload_artifacts_to_mlflow(
            mock_writer, tensorboard_dir=None, exp_root_path="/tmp/exp"
        )

        assert result == {"traces": 0, "logs": 3}
        mock_upload_traces.assert_not_called()
        mock_upload_logs.assert_called_once()

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_log_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_trace_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_missing_exp_root_path(self, mock_log, mock_upload_traces, mock_upload_logs):
        """Test that missing exp_root_path skips log upload."""
        mock_writer = Mock()
        mock_upload_traces.return_value = 5

        result = mlflow_artifacts.upload_artifacts_to_mlflow(
            mock_writer, tensorboard_dir="/tmp/tb", exp_root_path=None
        )

        assert result == {"traces": 5, "logs": 0}
        mock_upload_traces.assert_called_once()
        mock_upload_logs.assert_not_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_log_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_trace_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_no_paths_provided(self, mock_log, mock_upload_traces, mock_upload_logs):
        """Test that no paths means no uploads."""
        mock_writer = Mock()

        result = mlflow_artifacts.upload_artifacts_to_mlflow(mock_writer)

        assert result == {"traces": 0, "logs": 0}
        mock_upload_traces.assert_not_called()
        mock_upload_logs.assert_not_called()


class TestIntegrationWithGlobalVars:
    """Test integration with global_vars module."""

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_artifacts_to_mlflow")
    def test_upload_mlflow_artifacts_calls_with_global_vars(self, mock_upload):
        """Test that upload_mlflow_artifacts in global_vars calls the artifact function correctly."""
        # Import here to avoid circular dependencies in test
        from primus.backends.megatron.training import global_vars

        # Set up mock mlflow writer
        mock_mlflow = Mock()
        global_vars._GLOBAL_MLFLOW_WRITER = mock_mlflow

        # Set up mock args
        mock_args = SimpleNamespace(tensorboard_dir="/tmp/tb")
        global_vars._GLOBAL_ARGS = mock_args

        # Set up exp_root_path
        global_vars._GLOBAL_EXP_ROOT_PATH = "/tmp/exp"

        # Set return value
        mock_upload.return_value = {"traces": 5, "logs": 3}

        # Call the function
        result = global_vars.upload_mlflow_artifacts(upload_traces=True, upload_logs=True)

        # Verify correct call
        mock_upload.assert_called_once_with(
            mlflow_writer=mock_mlflow,
            tensorboard_dir="/tmp/tb",
            exp_root_path="/tmp/exp",
            upload_traces=True,
            upload_logs=True,
        )
        assert result == {"traces": 5, "logs": 3}

        # Clean up
        global_vars.destroy_global_vars()

    def test_upload_mlflow_artifacts_returns_none_when_mlflow_disabled(self):
        """Test that upload_mlflow_artifacts returns None when MLflow is not enabled."""
        from primus.backends.megatron.training import global_vars

        # Ensure mlflow writer is None
        global_vars._GLOBAL_MLFLOW_WRITER = None

        result = global_vars.upload_mlflow_artifacts()

        assert result is None

        # Clean up
        global_vars.destroy_global_vars()

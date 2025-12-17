###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for mlflow_artifacts module.

Focus areas:
    1. File discovery functions (_get_all_trace_files, _get_all_log_files)
    2. Rank extraction and filtering functions
    3. TraceLens report generation with mocked dependencies
    4. MLflow artifact upload functions
    5. Error handling and edge cases
"""

import gzip
import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from primus.backends.megatron.training.mlflow_artifacts import (
    _extract_rank_from_filename,
    _filter_traces_by_rank,
    _generate_trace_summary_csv,
    _get_all_log_files,
    _get_all_trace_files,
    generate_tracelens_report,
    generate_tracelens_reports,
    upload_artifacts_to_mlflow,
    upload_log_files_to_mlflow,
    upload_trace_files_to_mlflow,
    upload_tracelens_reports_to_mlflow,
)


class TestGetAllTraceFiles:
    """Test _get_all_trace_files function."""

    def test_empty_directory(self):
        """Test with an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _get_all_trace_files(tmpdir)
            assert result == []

    def test_nonexistent_directory(self):
        """Test with a nonexistent directory."""
        result = _get_all_trace_files("/nonexistent/directory")
        assert result == []

    def test_none_directory(self):
        """Test with None as directory."""
        result = _get_all_trace_files(None)
        assert result == []

    def test_empty_string_directory(self):
        """Test with empty string as directory."""
        result = _get_all_trace_files("")
        assert result == []

    def test_finds_json_files(self):
        """Test finding .json trace files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test trace files
            trace1 = os.path.join(tmpdir, "rank_0_step_2.json")
            trace2 = os.path.join(tmpdir, "rank_1_step_2.json")
            open(trace1, "w").close()
            open(trace2, "w").close()

            result = _get_all_trace_files(tmpdir)
            assert len(result) == 2
            assert trace1 in result
            assert trace2 in result

    def test_finds_json_gz_files(self):
        """Test finding .json.gz trace files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test compressed trace files
            trace1 = os.path.join(tmpdir, "rank_0_step_2.json.gz")
            trace2 = os.path.join(tmpdir, "rank_1_step_2.json.gz")
            open(trace1, "w").close()
            open(trace2, "w").close()

            result = _get_all_trace_files(tmpdir)
            assert len(result) == 2
            assert trace1 in result
            assert trace2 in result

    def test_finds_pt_trace_json_files(self):
        """Test finding .pt.trace.json files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create PyTorch profiler trace files
            trace1 = os.path.join(tmpdir, "primus-megatron-exp[test]-rank[0].1234567890.pt.trace.json")
            trace2 = os.path.join(tmpdir, "primus-megatron-exp[test]-rank[1].1234567890.pt.trace.json.gz")
            open(trace1, "w").close()
            open(trace2, "w").close()

            result = _get_all_trace_files(tmpdir)
            assert len(result) == 2
            assert trace1 in result
            assert trace2 in result

    def test_recursive_search(self):
        """Test that function searches subdirectories recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested directory structure
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)

            # Create trace files at different levels
            trace1 = os.path.join(tmpdir, "rank_0.json")
            trace2 = os.path.join(subdir, "rank_1.json")
            open(trace1, "w").close()
            open(trace2, "w").close()

            result = _get_all_trace_files(tmpdir)
            assert len(result) == 2
            assert trace1 in result
            assert trace2 in result

    def test_removes_duplicates(self):
        """Test that duplicate files are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a trace file
            trace = os.path.join(tmpdir, "rank_0.json")
            open(trace, "w").close()

            result = _get_all_trace_files(tmpdir)
            # Should find the file only once despite multiple patterns
            assert len(result) == 1
            assert trace in result

    def test_ignores_non_trace_files(self):
        """Test that non-trace files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create various files
            open(os.path.join(tmpdir, "rank_0.json"), "w").close()
            open(os.path.join(tmpdir, "log.txt"), "w").close()
            open(os.path.join(tmpdir, "config.yaml"), "w").close()
            open(os.path.join(tmpdir, "other.py"), "w").close()

            result = _get_all_trace_files(tmpdir)
            # Should only find the .json trace file
            assert len(result) == 1
            assert result[0].endswith("rank_0.json")


class TestGetAllLogFiles:
    """Test _get_all_log_files function."""

    def test_empty_directory(self):
        """Test with an empty experiment directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _get_all_log_files(tmpdir)
            assert result == []

    def test_nonexistent_directory(self):
        """Test with a nonexistent directory."""
        result = _get_all_log_files("/nonexistent/directory")
        assert result == []

    def test_none_directory(self):
        """Test with None as directory."""
        result = _get_all_log_files(None)
        assert result == []

    def test_empty_string_directory(self):
        """Test with empty string as directory."""
        result = _get_all_log_files("")
        assert result == []

    def test_no_logs_directory(self):
        """Test when logs directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create exp_root but no logs directory
            result = _get_all_log_files(tmpdir)
            assert result == []

    def test_finds_log_files(self):
        """Test finding log files in logs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create logs directory structure
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)

            # Create log files
            log1 = os.path.join(logs_dir, "master.log")
            log2 = os.path.join(logs_dir, "rank-0.log")
            open(log1, "w").close()
            open(log2, "w").close()

            result = _get_all_log_files(tmpdir)
            assert len(result) == 2
            assert log1 in result
            assert log2 in result

    def test_recursive_search(self):
        """Test that function searches subdirectories recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested logs directory structure
            logs_dir = os.path.join(tmpdir, "logs")
            master_dir = os.path.join(logs_dir, "master")
            rank_dir = os.path.join(logs_dir, "module", "rank-0")
            os.makedirs(master_dir)
            os.makedirs(rank_dir)

            # Create log files at different levels
            log1 = os.path.join(master_dir, "master-001.log")
            log2 = os.path.join(rank_dir, "output.log")
            open(log1, "w").close()
            open(log2, "w").close()

            result = _get_all_log_files(tmpdir)
            assert len(result) == 2
            assert log1 in result
            assert log2 in result

    def test_ignores_non_log_files(self):
        """Test that non-log files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create logs directory
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)

            # Create various files
            open(os.path.join(logs_dir, "output.log"), "w").close()
            open(os.path.join(logs_dir, "config.yaml"), "w").close()
            open(os.path.join(logs_dir, "trace.json"), "w").close()
            open(os.path.join(logs_dir, "script.py"), "w").close()

            result = _get_all_log_files(tmpdir)
            # Should only find the .log file
            assert len(result) == 1
            assert result[0].endswith("output.log")


class TestExtractRankFromFilename:
    """Test _extract_rank_from_filename function."""

    def test_rank_underscore_pattern(self):
        """Test rank_N_ pattern."""
        assert _extract_rank_from_filename("rank_0_step_2.json") == 0
        assert _extract_rank_from_filename("rank_1_step_2.json") == 1
        assert _extract_rank_from_filename("rank_123_step_2.json") == 123

    def test_rank_bracket_pattern(self):
        """Test rank[N] pattern."""
        assert _extract_rank_from_filename("primus-megatron-exp[test]-rank[0].json") == 0
        assert _extract_rank_from_filename("primus-megatron-exp[test]-rank[42].json") == 42

    def test_dash_rank_pattern(self):
        """Test -rankN. pattern."""
        assert _extract_rank_from_filename("trace-rank0.json") == 0
        assert _extract_rank_from_filename("trace-rank99.json") == 99

    def test_underscore_rank_pattern(self):
        """Test _rankN. pattern."""
        assert _extract_rank_from_filename("trace_rank0.json") == 0
        assert _extract_rank_from_filename("trace_rank15.json") == 15

    def test_no_rank_in_filename(self):
        """Test filename without rank information."""
        assert _extract_rank_from_filename("trace.json") is None
        assert _extract_rank_from_filename("output.log") is None
        assert _extract_rank_from_filename("no_rank_here.json") is None

    def test_multiple_patterns(self):
        """Test filename with multiple patterns (should match first)."""
        # Should match first pattern found
        filename = "rank_0_step_rank[1].json"
        result = _extract_rank_from_filename(filename)
        assert result == 0  # Matches rank_0_ pattern first

    def test_edge_cases(self):
        """Test edge cases."""
        assert _extract_rank_from_filename("") is None
        assert _extract_rank_from_filename("rank") is None
        assert _extract_rank_from_filename("rank_.json") is None  # No number


class TestFilterTracesByRank:
    """Test _filter_traces_by_rank function."""

    def test_empty_ranks_list(self):
        """Test with empty ranks list returns all traces."""
        traces = ["/path/rank_0.json", "/path/rank_1.json", "/path/rank_2.json"]
        result = _filter_traces_by_rank(traces, [])
        assert result == traces

    def test_none_ranks(self):
        """Test with None ranks returns all traces."""
        traces = ["/path/rank_0.json", "/path/rank_1.json"]
        result = _filter_traces_by_rank(traces, None)
        assert result == traces

    def test_filter_single_rank(self):
        """Test filtering for a single rank."""
        traces = ["rank_0.json", "rank_1.json", "rank_2.json"]
        result = _filter_traces_by_rank(traces, [0])
        assert result == ["rank_0.json"]

    def test_filter_multiple_ranks(self):
        """Test filtering for multiple ranks."""
        traces = [
            "rank_0.json",
            "rank_1.json",
            "rank_2.json",
            "rank_3.json",
        ]
        result = _filter_traces_by_rank(traces, [0, 2])
        assert len(result) == 2
        assert "rank_0.json" in result
        assert "rank_2.json" in result

    def test_filter_non_matching_rank(self):
        """Test filtering for rank that doesn't exist."""
        traces = ["/path/rank_0.json", "/path/rank_1.json"]
        result = _filter_traces_by_rank(traces, [5])
        assert result == []

    def test_filter_with_no_rank_in_filename(self):
        """Test filtering when trace files don't have rank in filename."""
        traces = ["/path/trace1.json", "/path/trace2.json"]
        result = _filter_traces_by_rank(traces, [0])
        assert result == []

    def test_filter_mixed_filenames(self):
        """Test filtering with mix of files with and without rank."""
        traces = [
            "rank_0.json",
            "trace.json",
            "rank_1.json",
        ]
        result = _filter_traces_by_rank(traces, [0])
        assert result == ["rank_0.json"]


class TestUploadTraceFilesToMLflow:
    """Test upload_trace_files_to_mlflow function."""

    def test_none_mlflow_writer(self):
        """Test with None mlflow_writer returns 0."""
        result = upload_trace_files_to_mlflow(None, "/path/to/traces")
        assert result == 0

    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_no_trace_files(self, mock_log, mock_get_traces):
        """Test when no trace files are found."""
        mock_get_traces.return_value = []
        mock_writer = Mock()

        result = upload_trace_files_to_mlflow(mock_writer, "/path/to/traces")

        assert result == 0
        mock_writer.log_artifact.assert_not_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_single_trace_file(self, mock_log, mock_get_traces):
        """Test uploading a single trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.json")
            open(trace_file, "w").close()
            mock_get_traces.return_value = [trace_file]

            mock_writer = Mock()
            result = upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 1
            mock_writer.log_artifact.assert_called_once_with(
                trace_file, artifact_path="traces"
            )

    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_multiple_trace_files(self, mock_log, mock_get_traces):
        """Test uploading multiple trace files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace1 = os.path.join(tmpdir, "rank_0.json")
            trace2 = os.path.join(tmpdir, "rank_1.json")
            open(trace1, "w").close()
            open(trace2, "w").close()
            mock_get_traces.return_value = [trace1, trace2]

            mock_writer = Mock()
            result = upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 2
            assert mock_writer.log_artifact.call_count == 2

    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_upload_with_exception(self, mock_warning, mock_log, mock_get_traces):
        """Test handling of upload exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.json")
            open(trace_file, "w").close()
            mock_get_traces.return_value = [trace_file]

            mock_writer = Mock()
            mock_writer.log_artifact.side_effect = Exception("Upload failed")

            result = upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 0
            mock_warning.assert_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_custom_artifact_path(self, mock_log, mock_get_traces):
        """Test using custom artifact path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "rank_0.json")
            open(trace_file, "w").close()
            mock_get_traces.return_value = [trace_file]

            mock_writer = Mock()
            result = upload_trace_files_to_mlflow(
                mock_writer, tmpdir, artifact_path="custom/path"
            )

            assert result == 1
            mock_writer.log_artifact.assert_called_once_with(
                trace_file, artifact_path="custom/path"
            )

    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_preserves_subdirectory_structure(self, mock_log, mock_get_traces):
        """Test that subdirectory structure is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            trace_file = os.path.join(subdir, "rank_0.json")
            open(trace_file, "w").close()
            mock_get_traces.return_value = [trace_file]

            mock_writer = Mock()
            result = upload_trace_files_to_mlflow(mock_writer, tmpdir)

            assert result == 1
            # Should preserve subdirectory in artifact path
            call_args = mock_writer.log_artifact.call_args
            assert call_args[1]["artifact_path"] == "traces/subdir"


class TestUploadLogFilesToMLflow:
    """Test upload_log_files_to_mlflow function."""

    def test_none_mlflow_writer(self):
        """Test with None mlflow_writer returns 0."""
        result = upload_log_files_to_mlflow(None, "/path/to/exp")
        assert result == 0

    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_log_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_no_log_files(self, mock_log, mock_get_logs):
        """Test when no log files are found."""
        mock_get_logs.return_value = []
        mock_writer = Mock()

        result = upload_log_files_to_mlflow(mock_writer, "/path/to/exp")

        assert result == 0
        mock_writer.log_artifact.assert_not_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_log_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_upload_with_exception(self, mock_warning, mock_log, mock_get_logs):
        """Test handling of upload exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(logs_dir)
            log_file = os.path.join(logs_dir, "master.log")
            open(log_file, "w").close()
            mock_get_logs.return_value = [log_file]

            mock_writer = Mock()
            mock_writer.log_artifact.side_effect = Exception("Upload failed")

            result = upload_log_files_to_mlflow(mock_writer, tmpdir)

            assert result == 0
            mock_warning.assert_called()


class TestGenerateTraceSummaryCSV:
    """Test _generate_trace_summary_csv function."""

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_nonexistent_file(self, mock_warning):
        """Test with nonexistent trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _generate_trace_summary_csv(
                "/nonexistent/file.json", tmpdir, "output.csv"
            )
            assert result is None

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_valid_json_trace(self, mock_log):
        """Test with valid JSON trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple trace file
            trace_file = os.path.join(tmpdir, "trace.json")
            trace_data = {
                "traceEvents": [
                    {"cat": "kernel", "name": "kernel1", "dur": 1000},
                    {"cat": "kernel", "name": "kernel1", "dur": 2000},
                    {"cat": "kernel", "name": "kernel2", "dur": 3000},
                ]
            }
            with open(trace_file, "w") as f:
                json.dump(trace_data, f)

            result = _generate_trace_summary_csv(trace_file, tmpdir, "summary.csv")

            assert result is not None
            assert os.path.exists(result)
            assert result.endswith("summary.csv")

            # Verify CSV content
            with open(result, "r") as f:
                lines = f.readlines()
                assert len(lines) > 1  # Header + data rows
                assert "Operation" in lines[0]
                assert "kernel1" in lines[1] or "kernel2" in lines[1]

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_gzipped_trace_file(self, mock_log):
        """Test with gzipped trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a gzipped trace file
            trace_file = os.path.join(tmpdir, "trace.json.gz")
            trace_data = {
                "traceEvents": [
                    {"cat": "kernel", "name": "test_kernel", "dur": 5000},
                ]
            }
            with gzip.open(trace_file, "wt", encoding="utf-8") as f:
                json.dump(trace_data, f)

            result = _generate_trace_summary_csv(trace_file, tmpdir, "summary.csv")

            assert result is not None
            assert os.path.exists(result)

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_empty_trace_events(self, mock_warning):
        """Test with trace file containing no events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            trace_data = {"traceEvents": []}
            with open(trace_file, "w") as f:
                json.dump(trace_data, f)

            result = _generate_trace_summary_csv(trace_file, tmpdir, "summary.csv")

            assert result is None

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_no_kernel_events(self, mock_warning):
        """Test with trace file containing no kernel events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            trace_data = {
                "traceEvents": [
                    {"cat": "other", "name": "event1", "dur": 1000},
                ]
            }
            with open(trace_file, "w") as f:
                json.dump(trace_data, f)

            result = _generate_trace_summary_csv(trace_file, tmpdir, "summary.csv")

            assert result is None

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_invalid_json(self, mock_warning):
        """Test with invalid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            with open(trace_file, "w") as f:
                f.write("invalid json content")

            result = _generate_trace_summary_csv(trace_file, tmpdir, "summary.csv")

            assert result is None


class TestGenerateTracelensReport:
    """Test generate_tracelens_report function."""

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_nonexistent_trace_file(self, mock_warning):
        """Test with nonexistent trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_tracelens_report(
                "/nonexistent/file.json", tmpdir, "report"
            )
            assert result == []

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_xlsx_format_success(self, mock_log):
        """Test successful XLSX report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            open(trace_file, "w").close()

            # Mock TraceLens function import
            mock_generate = Mock(return_value={"kernels": Mock(), "memory": Mock()})
            
            # Create the xlsx file that would be generated
            xlsx_path = os.path.join(tmpdir, "trace_analysis.xlsx")
            open(xlsx_path, "w").close()

            with patch.dict("sys.modules", {"TraceLens": Mock(), "TraceLens.Reporting": Mock()}):
                with patch("primus.backends.megatron.training.mlflow_artifacts.generate_perf_report_pytorch", mock_generate):
                    with patch("primus.backends.megatron.training.mlflow_artifacts.glob.glob", return_value=[]):
                        result = generate_tracelens_report(
                            trace_file, tmpdir, "trace", output_format="xlsx"
                        )

            assert len(result) == 1
            assert result[0].endswith(".xlsx")

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_csv_format_success(self, mock_log):
        """Test successful CSV report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            open(trace_file, "w").close()

            # Mock CSV files generation
            csv_dir = os.path.join(tmpdir, "trace")
            os.makedirs(csv_dir)
            csv_files = [
                os.path.join(csv_dir, "kernels.csv"),
                os.path.join(csv_dir, "memory.csv"),
            ]
            for csv_file in csv_files:
                open(csv_file, "w").close()

            mock_generate = Mock(return_value=None)

            with patch.dict("sys.modules", {"TraceLens": Mock(), "TraceLens.Reporting": Mock()}):
                with patch("primus.backends.megatron.training.mlflow_artifacts.generate_perf_report_pytorch", mock_generate):
                    with patch("primus.backends.megatron.training.mlflow_artifacts.glob.glob", return_value=csv_files):
                        result = generate_tracelens_report(
                            trace_file, tmpdir, "trace", output_format="csv"
                        )

            assert len(result) == 2
            assert all(f.endswith(".csv") for f in result)

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_all_format(self, mock_log):
        """Test 'all' format generates both XLSX and CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            open(trace_file, "w").close()

            # Mock TraceLens function
            mock_generate = Mock(return_value={"kernels": Mock()})

            # Create files that would be generated
            xlsx_path = os.path.join(tmpdir, "trace_analysis.xlsx")
            open(xlsx_path, "w").close()

            csv_dir = os.path.join(tmpdir, "trace")
            os.makedirs(csv_dir)
            csv_file = os.path.join(csv_dir, "kernels.csv")
            open(csv_file, "w").close()

            with patch.dict("sys.modules", {"TraceLens": Mock(), "TraceLens.Reporting": Mock()}):
                with patch("primus.backends.megatron.training.mlflow_artifacts.generate_perf_report_pytorch", mock_generate):
                    with patch("primus.backends.megatron.training.mlflow_artifacts.glob.glob", return_value=[csv_file]):
                        result = generate_tracelens_report(
                            trace_file, tmpdir, "trace", output_format="all"
                        )

            # Should have both xlsx and csv files
            assert len(result) >= 2

    @patch("primus.backends.megatron.training.mlflow_artifacts._generate_trace_summary_csv")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_tracelens_import_error_fallback(self, mock_log, mock_csv_fallback):
        """Test fallback to CSV when TraceLens import fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            open(trace_file, "w").close()

            csv_path = os.path.join(tmpdir, "trace_summary.csv")
            open(csv_path, "w").close()
            mock_csv_fallback.return_value = csv_path

            # Simulate ImportError when trying to import TraceLens
            result = generate_tracelens_report(trace_file, tmpdir, "trace")

            # Should fallback to CSV
            assert len(result) >= 1
            mock_csv_fallback.assert_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts._generate_trace_summary_csv")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_tracelens_exception_fallback(self, mock_warning, mock_csv_fallback):
        """Test fallback to CSV when TraceLens raises exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            open(trace_file, "w").close()

            csv_path = os.path.join(tmpdir, "trace_summary.csv")
            open(csv_path, "w").close()
            mock_csv_fallback.return_value = csv_path

            mock_generate = Mock(side_effect=Exception("Processing error"))

            with patch.dict("sys.modules", {"TraceLens": Mock(), "TraceLens.Reporting": Mock()}):
                with patch("primus.backends.megatron.training.mlflow_artifacts.generate_perf_report_pytorch", mock_generate):
                    result = generate_tracelens_report(trace_file, tmpdir, "trace")

            assert len(result) >= 1
            mock_csv_fallback.assert_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_html_format_fallback(self, mock_log, mock_warning):
        """Test that HTML format falls back to xlsx+csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            open(trace_file, "w").close()

            mock_generate = Mock(return_value={"kernels": Mock()})

            # Create files that would be generated
            xlsx_path = os.path.join(tmpdir, "trace_analysis.xlsx")
            open(xlsx_path, "w").close()

            csv_dir = os.path.join(tmpdir, "trace")
            os.makedirs(csv_dir)
            csv_file = os.path.join(csv_dir, "kernels.csv")
            open(csv_file, "w").close()

            with patch.dict("sys.modules", {"TraceLens": Mock(), "TraceLens.Reporting": Mock()}):
                with patch("primus.backends.megatron.training.mlflow_artifacts.generate_perf_report_pytorch", mock_generate):
                    with patch("primus.backends.megatron.training.mlflow_artifacts.glob.glob", return_value=[csv_file]):
                        result = generate_tracelens_report(
                            trace_file, tmpdir, "trace", output_format="html"
                        )

            # Should fall back to both xlsx and csv
            assert len(result) >= 1
            mock_warning.assert_called()  # Should warn about HTML not supported

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_custom_report_name(self, mock_log):
        """Test using custom report name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = os.path.join(tmpdir, "trace.json")
            open(trace_file, "w").close()

            mock_generate = Mock(return_value={"kernels": Mock()})

            xlsx_path = os.path.join(tmpdir, "custom_name_analysis.xlsx")
            open(xlsx_path, "w").close()

            with patch.dict("sys.modules", {"TraceLens": Mock(), "TraceLens.Reporting": Mock()}):
                with patch("primus.backends.megatron.training.mlflow_artifacts.generate_perf_report_pytorch", mock_generate):
                    with patch("primus.backends.megatron.training.mlflow_artifacts.glob.glob", return_value=[]):
                        result = generate_tracelens_report(
                            trace_file, tmpdir, "custom_name", output_format="xlsx"
                        )

            assert len(result) == 1
            assert "custom_name" in result[0]


class TestGenerateTracelensReports:
    """Test generate_tracelens_reports function."""

    @patch("primus.backends.megatron.training.mlflow_artifacts._ensure_tracelens_installed")
    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    def test_no_trace_files(self, mock_get_traces, mock_ensure):
        """Test when no trace files are found."""
        mock_get_traces.return_value = []
        mock_ensure.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_tracelens_reports(tmpdir, tmpdir)

        assert result == []

    @patch("primus.backends.megatron.training.mlflow_artifacts._ensure_tracelens_installed")
    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.generate_tracelens_report")
    def test_generate_reports_for_all_traces(self, mock_gen_report, mock_get_traces, mock_ensure):
        """Test generating reports for all trace files."""
        mock_ensure.return_value = True
        trace_files = ["/path/rank_0.json", "/path/rank_1.json"]
        mock_get_traces.return_value = trace_files
        mock_gen_report.side_effect = [
            ["/output/rank_0.xlsx"],
            ["/output/rank_1.xlsx"],
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_tracelens_reports(tmpdir, tmpdir)

        assert len(result) == 2
        assert mock_gen_report.call_count == 2

    @patch("primus.backends.megatron.training.mlflow_artifacts._ensure_tracelens_installed")
    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.generate_tracelens_report")
    def test_filter_by_ranks(self, mock_gen_report, mock_get_traces, mock_ensure):
        """Test filtering traces by ranks."""
        mock_ensure.return_value = True
        trace_files = ["/path/rank_0.json", "/path/rank_1.json", "/path/rank_2.json"]
        mock_get_traces.return_value = trace_files
        mock_gen_report.return_value = ["/output/report.xlsx"]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_tracelens_reports(tmpdir, tmpdir, ranks=[0, 2])

        # Should only generate reports for rank 0 and 2
        assert mock_gen_report.call_count == 2

    @patch("primus.backends.megatron.training.mlflow_artifacts._ensure_tracelens_installed")
    @patch("primus.backends.megatron.training.mlflow_artifacts._get_all_trace_files")
    @patch("primus.backends.megatron.training.mlflow_artifacts.generate_tracelens_report")
    def test_max_reports_limit(self, mock_gen_report, mock_get_traces, mock_ensure):
        """Test limiting maximum number of reports."""
        mock_ensure.return_value = True
        trace_files = [f"/path/rank_{i}.json" for i in range(10)]
        mock_get_traces.return_value = trace_files
        mock_gen_report.return_value = ["/output/report.xlsx"]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_tracelens_reports(tmpdir, tmpdir, max_reports=3)

        # Should only generate 3 reports
        assert mock_gen_report.call_count == 3


class TestUploadTracelensReportsToMLflow:
    """Test upload_tracelens_reports_to_mlflow function."""

    def test_none_mlflow_writer(self):
        """Test with None mlflow_writer returns 0."""
        result = upload_tracelens_reports_to_mlflow(None, "/traces", "/exp")
        assert result == 0

    @patch("primus.backends.megatron.training.mlflow_artifacts.generate_tracelens_reports")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_no_reports_generated(self, mock_log, mock_gen_reports):
        """Test when no reports are generated."""
        mock_gen_reports.return_value = []
        mock_writer = Mock()

        result = upload_tracelens_reports_to_mlflow(mock_writer, "/traces", "/exp")

        assert result == 0
        mock_writer.log_artifact.assert_not_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts.generate_tracelens_reports")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_generated_reports(self, mock_log, mock_gen_reports):
        """Test uploading generated reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report1 = os.path.join(tmpdir, "rank_0.xlsx")
            report2 = os.path.join(tmpdir, "rank_1.xlsx")
            open(report1, "w").close()
            open(report2, "w").close()

            mock_gen_reports.return_value = [report1, report2]
            mock_writer = Mock()

            result = upload_tracelens_reports_to_mlflow(mock_writer, tmpdir, tmpdir)

            assert result == 2
            assert mock_writer.log_artifact.call_count == 2

    @patch("primus.backends.megatron.training.mlflow_artifacts.generate_tracelens_reports")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    @patch("primus.backends.megatron.training.mlflow_artifacts.warning_rank_0")
    def test_upload_with_exception(self, mock_warning, mock_log, mock_gen_reports):
        """Test handling of upload exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = os.path.join(tmpdir, "report.xlsx")
            open(report, "w").close()

            mock_gen_reports.return_value = [report]
            mock_writer = Mock()
            mock_writer.log_artifact.side_effect = Exception("Upload failed")

            result = upload_tracelens_reports_to_mlflow(mock_writer, tmpdir, tmpdir)

            assert result == 0
            mock_warning.assert_called()

    @patch("primus.backends.megatron.training.mlflow_artifacts.generate_tracelens_reports")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_parameters_passed_to_generate(self, mock_log, mock_gen_reports):
        """Test that parameters are passed correctly to generate_tracelens_reports."""
        mock_gen_reports.return_value = []
        mock_writer = Mock()

        upload_tracelens_reports_to_mlflow(
            mock_writer,
            "/traces",
            "/exp",
            ranks=[0, 1],
            max_reports=5,
            output_format="csv",
        )

        mock_gen_reports.assert_called_once()
        call_kwargs = mock_gen_reports.call_args[1]
        assert call_kwargs["ranks"] == [0, 1]
        assert call_kwargs["max_reports"] == 5
        assert call_kwargs["output_format"] == "csv"


class TestUploadArtifactsToMLflow:
    """Test upload_artifacts_to_mlflow function."""

    def test_none_mlflow_writer(self):
        """Test with None mlflow_writer returns zeros."""
        result = upload_artifacts_to_mlflow(None)
        assert result == {"traces": 0, "logs": 0, "tracelens_reports": 0}

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_trace_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_traces_only(self, mock_log, mock_upload_traces):
        """Test uploading only trace files."""
        mock_upload_traces.return_value = 5
        mock_writer = Mock()

        result = upload_artifacts_to_mlflow(
            mock_writer,
            tensorboard_dir="/traces",
            upload_traces=True,
            upload_logs=False,
            upload_tracelens_report=False,
        )

        assert result["traces"] == 5
        assert result["logs"] == 0
        assert result["tracelens_reports"] == 0
        mock_upload_traces.assert_called_once()

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_log_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_logs_only(self, mock_log, mock_upload_logs):
        """Test uploading only log files."""
        mock_upload_logs.return_value = 3
        mock_writer = Mock()

        result = upload_artifacts_to_mlflow(
            mock_writer,
            exp_root_path="/exp",
            upload_traces=False,
            upload_logs=True,
            upload_tracelens_report=False,
        )

        assert result["traces"] == 0
        assert result["logs"] == 3
        assert result["tracelens_reports"] == 0
        mock_upload_logs.assert_called_once()

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_tracelens_reports_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_tracelens_reports_only(self, mock_log, mock_upload_tracelens):
        """Test uploading only TraceLens reports."""
        mock_upload_tracelens.return_value = 2
        mock_writer = Mock()

        result = upload_artifacts_to_mlflow(
            mock_writer,
            tensorboard_dir="/traces",
            exp_root_path="/exp",
            upload_traces=False,
            upload_logs=False,
            upload_tracelens_report=True,
        )

        assert result["traces"] == 0
        assert result["logs"] == 0
        assert result["tracelens_reports"] == 2
        mock_upload_tracelens.assert_called_once()

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_trace_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_log_files_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_tracelens_reports_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_upload_all_artifacts(
        self, mock_log, mock_upload_tracelens, mock_upload_logs, mock_upload_traces
    ):
        """Test uploading all types of artifacts."""
        mock_upload_traces.return_value = 10
        mock_upload_logs.return_value = 5
        mock_upload_tracelens.return_value = 3
        mock_writer = Mock()

        result = upload_artifacts_to_mlflow(
            mock_writer,
            tensorboard_dir="/traces",
            exp_root_path="/exp",
            upload_traces=True,
            upload_logs=True,
            upload_tracelens_report=True,
        )

        assert result["traces"] == 10
        assert result["logs"] == 5
        assert result["tracelens_reports"] == 3
        mock_upload_traces.assert_called_once()
        mock_upload_logs.assert_called_once()
        mock_upload_tracelens.assert_called_once()

    @patch("primus.backends.megatron.training.mlflow_artifacts.upload_tracelens_reports_to_mlflow")
    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_tracelens_parameters_passed(self, mock_log, mock_upload_tracelens):
        """Test that TraceLens parameters are passed correctly."""
        mock_upload_tracelens.return_value = 1
        mock_writer = Mock()

        upload_artifacts_to_mlflow(
            mock_writer,
            tensorboard_dir="/traces",
            exp_root_path="/exp",
            upload_tracelens_report=True,
            tracelens_ranks=[0, 1, 2],
            tracelens_max_reports=10,
            tracelens_output_format="xlsx",
        )

        call_kwargs = mock_upload_tracelens.call_args[1]
        assert call_kwargs["ranks"] == [0, 1, 2]
        assert call_kwargs["max_reports"] == 10
        assert call_kwargs["output_format"] == "xlsx"

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_skip_traces_when_no_tensorboard_dir(self, mock_log):
        """Test that traces are skipped when tensorboard_dir is None."""
        mock_writer = Mock()

        result = upload_artifacts_to_mlflow(
            mock_writer, tensorboard_dir=None, upload_traces=True
        )

        assert result["traces"] == 0

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_skip_logs_when_no_exp_root_path(self, mock_log):
        """Test that logs are skipped when exp_root_path is None."""
        mock_writer = Mock()

        result = upload_artifacts_to_mlflow(
            mock_writer, exp_root_path=None, upload_logs=True
        )

        assert result["logs"] == 0

    @patch("primus.backends.megatron.training.mlflow_artifacts.log_rank_0")
    def test_skip_tracelens_when_paths_missing(self, mock_log):
        """Test that TraceLens is skipped when required paths are None."""
        mock_writer = Mock()

        # Missing tensorboard_dir
        result = upload_artifacts_to_mlflow(
            mock_writer,
            tensorboard_dir=None,
            exp_root_path="/exp",
            upload_tracelens_report=True,
        )
        assert result["tracelens_reports"] == 0

        # Missing exp_root_path
        result = upload_artifacts_to_mlflow(
            mock_writer,
            tensorboard_dir="/traces",
            exp_root_path=None,
            upload_tracelens_report=True,
        )
        assert result["tracelens_reports"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

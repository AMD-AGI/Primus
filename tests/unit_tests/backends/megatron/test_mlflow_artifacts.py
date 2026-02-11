###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for primus.backends.megatron.training.mlflow_artifacts.

Covers:
- Trace file discovery and filtering
- Rank extraction from filenames
- Report generation with mocked TraceLens API
- Upload logic with mocked mlflow_writer (file vs directory)
- Error handling and fallback behavior
- Cleanup logic validation
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from primus.backends.megatron.training import mlflow_artifacts as mlflow_artifacts_mod

# Use the module reference for patching in tests
_MODULE = "primus.backends.megatron.training.mlflow_artifacts"


@pytest.fixture(autouse=True)
def suppress_logging():
    """Suppress log_rank_0 and warning_rank_0 in all tests."""
    with patch(f"{_MODULE}.log_rank_0"), patch(f"{_MODULE}.warning_rank_0"):
        yield


# -----------------------------------------------------------------------------
# Trace file discovery
# -----------------------------------------------------------------------------


class TestGetAllTraceFiles:
    """Test _get_all_trace_files discovery and filtering."""

    def test_returns_empty_for_none_path(self):
        out = mlflow_artifacts_mod._get_all_trace_files(None)
        assert out == []

    def test_returns_empty_for_empty_string(self):
        out = mlflow_artifacts_mod._get_all_trace_files("")
        assert out == []

    def test_returns_empty_for_missing_directory(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert not missing.exists()
        out = mlflow_artifacts_mod._get_all_trace_files(str(missing))
        assert out == []

    def test_finds_pt_trace_json_in_root(self, tmp_path):
        (tmp_path / "rank_0_step_1.pt.trace.json").write_text("{}")
        (tmp_path / "rank_1_step_1.pt.trace.json").write_text("{}")
        (tmp_path / "other.json").write_text("{}")
        out = mlflow_artifacts_mod._get_all_trace_files(str(tmp_path))
        assert len(out) == 2
        basenames = {os.path.basename(p) for p in out}
        assert basenames == {"rank_0_step_1.pt.trace.json", "rank_1_step_1.pt.trace.json"}

    def test_finds_pt_trace_json_gz(self, tmp_path):
        (tmp_path / "rank_0.pt.trace.json.gz").write_text("")
        out = mlflow_artifacts_mod._get_all_trace_files(str(tmp_path))
        assert len(out) == 1
        assert out[0].endswith("rank_0.pt.trace.json.gz")

    def test_finds_traces_recursively_in_subdirs(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "rank_2.pt.trace.json").write_text("{}")
        out = mlflow_artifacts_mod._get_all_trace_files(str(tmp_path))
        assert len(out) == 1
        assert "rank_2" in out[0]

    def test_deduplicates_results(self, tmp_path):
        (tmp_path / "a.pt.trace.json").write_text("{}")
        out = mlflow_artifacts_mod._get_all_trace_files(str(tmp_path))
        assert len(out) == 1


# -----------------------------------------------------------------------------
# Rank extraction from filenames
# -----------------------------------------------------------------------------


class TestExtractRankFromFilename:
    """Test _extract_rank_from_filename patterns."""

    def test_rank_underscore_number_underscore(self):
        assert mlflow_artifacts_mod._extract_rank_from_filename("rank_0_step_2.json.gz") == 0
        assert mlflow_artifacts_mod._extract_rank_from_filename("rank_15_step_1.pt.trace.json") == 15

    def test_rank_underscore_number_dot(self):
        """Match rank_N. (dot after rank), e.g. rank_0.pt.trace.json.gz used by PyTorch profiler."""
        assert mlflow_artifacts_mod._extract_rank_from_filename("rank_0.pt.trace.json") == 0
        assert mlflow_artifacts_mod._extract_rank_from_filename("rank_0.pt.trace.json.gz") == 0
        assert mlflow_artifacts_mod._extract_rank_from_filename("rank_8.pt.trace.json") == 8

    def test_rank_bracket_number_bracket(self):
        assert (
            mlflow_artifacts_mod._extract_rank_from_filename("primus-megatron-exp-rank[0].pt.trace.json") == 0
        )
        assert mlflow_artifacts_mod._extract_rank_from_filename("prefix-rank[7].json") == 7

    def test_dash_rank_number_dot(self):
        assert mlflow_artifacts_mod._extract_rank_from_filename("trace-rank1.json") == 1

    def test_underscore_rank_number_dot(self):
        assert mlflow_artifacts_mod._extract_rank_from_filename("trace_rank2.json") == 2

    def test_returns_none_for_unknown_pattern(self):
        assert mlflow_artifacts_mod._extract_rank_from_filename("random_file.json") is None
        assert mlflow_artifacts_mod._extract_rank_from_filename("trace.json.gz") is None


# -----------------------------------------------------------------------------
# Filter traces by rank
# -----------------------------------------------------------------------------


class TestFilterTracesByRank:
    """Test _filter_traces_by_rank."""

    def test_returns_all_when_ranks_none(self, tmp_path):
        paths = [
            str(tmp_path / "rank_0.pt.trace.json"),
            str(tmp_path / "rank_1.pt.trace.json"),
        ]
        out = mlflow_artifacts_mod._filter_traces_by_rank(paths, None)
        assert out == paths

    def test_returns_empty_when_ranks_empty_list(self, tmp_path):
        paths = [str(tmp_path / "rank_0.pt.trace.json")]
        out = mlflow_artifacts_mod._filter_traces_by_rank(paths, [])
        assert out == []

    def test_filters_to_specified_ranks(self, tmp_path):
        paths = [
            str(tmp_path / "rank_0_step_1.pt.trace.json"),
            str(tmp_path / "rank_1_step_1.pt.trace.json"),
            str(tmp_path / "rank_2_step_1.pt.trace.json"),
        ]
        out = mlflow_artifacts_mod._filter_traces_by_rank(paths, [0, 2])
        assert len(out) == 2
        assert "rank_0" in out[0]
        assert "rank_2" in out[1]


# -----------------------------------------------------------------------------
# Report generation with mocked TraceLens
# -----------------------------------------------------------------------------


class TestGenerateTracelensReport:
    """Test generate_tracelens_report with mocked TraceLens."""

    def _install_fake_tracelens(self, mock_generate, xlsx_path=None, csv_dir=None):
        """Put fake TraceLens.Reporting into sys.modules so generate_tracelens_report can import it."""
        reporting = types.ModuleType("TraceLens.Reporting")
        reporting.generate_perf_report_pytorch = mock_generate
        tracelens = types.ModuleType("TraceLens")
        tracelens.Reporting = reporting
        sys.modules["TraceLens"] = tracelens
        sys.modules["TraceLens.Reporting"] = reporting

        def _side_effect(trace_file, output_xlsx_path=None, output_csvs_dir=None):
            if output_xlsx_path and xlsx_path is not False:
                Path(output_xlsx_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_xlsx_path).write_text("xlsx")
                return [{"tab1"}, {"tab2"}]
            if output_csvs_dir and csv_dir is not False:
                Path(output_csvs_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_csvs_dir) / "kernels.csv").write_text("kernels")
                (Path(output_csvs_dir) / "memory.csv").write_text("memory")
            return []

        mock_generate.side_effect = _side_effect
        return reporting

    def teardown_method(self):
        for key in list(sys.modules.keys()):
            if key == "TraceLens" or key.startswith("TraceLens."):
                del sys.modules[key]

    def test_report_generation_xlsx_with_mocked_tracelens(self, tmp_path):
        trace_file = tmp_path / "rank_0.pt.trace.json"
        trace_file.write_text('{"traceEvents": []}')
        output_dir = tmp_path / "reports"
        mock_gen = MagicMock()

        self._install_fake_tracelens(mock_gen, xlsx_path=True, csv_dir=None)
        try:
            with patch(f"{_MODULE}._ensure_openpyxl_installed"):
                result = mlflow_artifacts_mod.generate_tracelens_report(
                    str(trace_file), str(output_dir), output_format="xlsx"
                )
            assert len(result) == 1
            assert result[0].endswith("_analysis.xlsx")
            assert mock_gen.called
        finally:
            self.teardown_method()

    def test_report_generation_missing_trace_file(self, tmp_path):
        missing = tmp_path / "missing.pt.trace.json"
        assert not missing.exists()
        with patch(f"{_MODULE}.warning_rank_0") as warn:
            result = mlflow_artifacts_mod.generate_tracelens_report(
                str(missing), str(tmp_path), output_format="xlsx"
            )
        assert result == []
        assert warn.called


# -----------------------------------------------------------------------------
# Fallback CSV when TraceLens fails / not available
# -----------------------------------------------------------------------------


class TestGenerateTraceSummaryCsvFallback:
    """Test _generate_trace_summary_csv fallback."""

    def test_fallback_csv_from_valid_trace_json(self, tmp_path):
        trace_file = tmp_path / "rank_0.pt.trace.json"
        trace_file.write_text(
            '{"traceEvents": ['
            '{"name": "kernel1", "cat": "kernel", "dur": 100},'
            '{"name": "kernel1", "cat": "kernel", "dur": 200}'
            "]}"
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        with patch(f"{_MODULE}.log_rank_0"), patch(f"{_MODULE}.warning_rank_0"):
            path = mlflow_artifacts_mod._generate_trace_summary_csv(
                str(trace_file), str(out_dir), "summary.csv"
            )
        assert path is not None
        assert path.endswith("summary.csv")
        assert os.path.exists(path)
        content = Path(path).read_text()
        assert "kernel1" in content
        assert "Count" in content or "Total" in content

    def test_fallback_returns_none_for_missing_file(self, tmp_path):
        with patch(f"{_MODULE}.warning_rank_0"):
            path = mlflow_artifacts_mod._generate_trace_summary_csv(
                str(tmp_path / "missing.json"), str(tmp_path), "out.csv"
            )
        assert path is None

    def test_fallback_returns_none_for_empty_events(self, tmp_path):
        trace_file = tmp_path / "empty.pt.trace.json"
        trace_file.write_text('{"traceEvents": []}')
        with patch(f"{_MODULE}.warning_rank_0"):
            path = mlflow_artifacts_mod._generate_trace_summary_csv(str(trace_file), str(tmp_path), "out.csv")
        assert path is None


# -----------------------------------------------------------------------------
# Upload logic with mocked mlflow_writer
# -----------------------------------------------------------------------------


class TestUploadTraceFilesToMlflow:
    """Test upload_trace_files_to_mlflow with mocked writer."""

    def test_returns_zero_when_mlflow_writer_none(self, tmp_path):
        count = mlflow_artifacts_mod.upload_trace_files_to_mlflow(None, str(tmp_path), artifact_path="traces")
        assert count == 0

    def test_uploads_found_traces_and_returns_count(self, tmp_path):
        (tmp_path / "rank_0.pt.trace.json").write_text("{}")
        (tmp_path / "rank_1.pt.trace.json").write_text("{}")
        mock_writer = MagicMock()
        with patch(f"{_MODULE}.log_rank_0"):
            count = mlflow_artifacts_mod.upload_trace_files_to_mlflow(
                mock_writer, str(tmp_path), artifact_path="traces"
            )
        assert count == 2
        assert mock_writer.log_artifact.call_count == 2


class TestUploadTracelensReportsToMlflow:
    """Test upload_tracelens_reports_to_mlflow: file vs dir, cleanup, errors."""

    def test_returns_zero_when_mlflow_writer_none(self, tmp_path):
        with patch.object(
            mlflow_artifacts_mod,
            "generate_tracelens_reports",
            return_value=[],
        ):
            count = mlflow_artifacts_mod.upload_tracelens_reports_to_mlflow(
                None,
                str(tmp_path),
                str(tmp_path),
                ranks=[0],
                output_format="xlsx",
            )
        assert count == 0

    def test_uses_log_artifact_for_files_and_log_artifacts_for_dirs(self, tmp_path):
        file_report = tmp_path / "rank_0_analysis.xlsx"
        file_report.write_text("xlsx")
        dir_report = tmp_path / "rank_0"
        dir_report.mkdir()
        (dir_report / "kernels.csv").write_text("csv")
        reports = [str(file_report), str(dir_report)]
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "generate_tracelens_reports",
            return_value=reports,
        ):
            count = mlflow_artifacts_mod.upload_tracelens_reports_to_mlflow(
                mock_writer,
                str(tmp_path),
                str(tmp_path),
                ranks=[0],
                output_format="xlsx",
                artifact_path="trace_analysis",
            )
        assert count == 2
        mock_writer.log_artifact.assert_called_once()
        mock_writer.log_artifacts.assert_called_once()
        # Directory should be logged with subpath preserving name
        call_kw = mock_writer.log_artifacts.call_args[1]
        assert "artifact_path" in call_kw
        assert "rank_0" in call_kw["artifact_path"] or call_kw["artifact_path"] == "rank_0"

    def test_upload_failure_on_one_report_still_uploads_others(self, tmp_path):
        file1 = tmp_path / "r0.xlsx"
        file1.write_text("a")
        file2 = tmp_path / "r1.xlsx"
        file2.write_text("b")
        reports = [str(file1), str(file2)]
        mock_writer = MagicMock()
        mock_writer.log_artifact.side_effect = [None, Exception("upload failed")]
        with patch.object(
            mlflow_artifacts_mod,
            "generate_tracelens_reports",
            return_value=reports,
        ), patch(f"{_MODULE}.warning_rank_0"):
            count = mlflow_artifacts_mod.upload_tracelens_reports_to_mlflow(
                mock_writer,
                str(tmp_path),
                str(tmp_path),
                artifact_path="trace_analysis",
            )
        assert count == 1
        assert mock_writer.log_artifact.call_count == 2

    def test_cleanup_after_upload_calls_rmtree(self, tmp_path):
        reports_dir = tmp_path / "tracelens_reports"
        reports_dir.mkdir()
        file_report = reports_dir / "rank_0_analysis.xlsx"
        file_report.write_text("xlsx")
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "generate_tracelens_reports",
            return_value=[str(file_report)],
        ), patch("shutil.rmtree") as mock_rmtree:
            mlflow_artifacts_mod.upload_tracelens_reports_to_mlflow(
                mock_writer,
                str(tmp_path),
                str(tmp_path),
                artifact_path="trace_analysis",
                cleanup_after_upload=True,
            )
        mock_rmtree.assert_called_once()
        assert "tracelens_reports" in str(mock_rmtree.call_args[0][0])

    def test_no_cleanup_when_cleanup_after_upload_false(self, tmp_path):
        reports_dir = tmp_path / "tracelens_reports"
        reports_dir.mkdir()
        (reports_dir / "r0.xlsx").write_text("x")
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "generate_tracelens_reports",
            return_value=[str(reports_dir / "r0.xlsx")],
        ), patch("shutil.rmtree") as mock_rmtree:
            mlflow_artifacts_mod.upload_tracelens_reports_to_mlflow(
                mock_writer,
                str(tmp_path),
                str(tmp_path),
                cleanup_after_upload=False,
            )
        mock_rmtree.assert_not_called()

    def test_cleanup_skipped_when_some_uploads_failed(self, tmp_path):
        reports_dir = tmp_path / "tracelens_reports"
        reports_dir.mkdir()
        f1 = reports_dir / "r0.xlsx"
        f2 = reports_dir / "r1.xlsx"
        f1.write_text("a")
        f2.write_text("b")
        mock_writer = MagicMock()
        mock_writer.log_artifact.side_effect = [None, Exception("upload failed")]
        with patch.object(
            mlflow_artifacts_mod,
            "generate_tracelens_reports",
            return_value=[str(f1), str(f2)],
        ), patch("shutil.rmtree") as mock_rmtree, patch(f"{_MODULE}.warning_rank_0"):
            mlflow_artifacts_mod.upload_tracelens_reports_to_mlflow(
                mock_writer,
                str(tmp_path),
                str(tmp_path),
                artifact_path="trace_analysis",
                cleanup_after_upload=True,
            )
        mock_rmtree.assert_not_called()


# -----------------------------------------------------------------------------
# upload_artifacts_to_mlflow (main entry point)
# -----------------------------------------------------------------------------


class TestUploadArtifactsToMlflow:
    """Test upload_artifacts_to_mlflow: trace/log discovery, artifact paths, TraceLens logic, cleanup."""

    def test_returns_zero_dict_when_mlflow_writer_none(self, tmp_path):
        result = mlflow_artifacts_mod.upload_artifacts_to_mlflow(
            None,
            tensorboard_dir=str(tmp_path),
            exp_root_path=str(tmp_path),
        )
        assert result == {"traces": 0, "logs": 0, "tracelens_reports": 0}

    def test_upload_traces_called_with_correct_artifact_path(self, tmp_path):
        (tmp_path / "rank_0.pt.trace.json").write_text("{}")
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "upload_trace_files_to_mlflow",
            return_value=2,
        ) as mock_traces:
            result = mlflow_artifacts_mod.upload_artifacts_to_mlflow(
                mock_writer,
                tensorboard_dir=str(tmp_path),
                exp_root_path=str(tmp_path),
                upload_traces=True,
                upload_logs=False,
                generate_tracelens_report=False,
                upload_tracelens_report=False,
            )
        assert result["traces"] == 2
        mock_traces.assert_called_once()
        call_kw = mock_traces.call_args[1]
        assert call_kw["artifact_path"] == "traces"

    def test_upload_logs_called_with_correct_artifact_path(self, tmp_path):
        (tmp_path / "logs" / "master" / "master-0.log").parent.mkdir(parents=True)
        (tmp_path / "logs" / "master" / "master-0.log").write_text("log")
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "upload_log_files_to_mlflow",
            return_value=1,
        ) as mock_logs:
            result = mlflow_artifacts_mod.upload_artifacts_to_mlflow(
                mock_writer,
                tensorboard_dir=None,
                exp_root_path=str(tmp_path),
                upload_traces=False,
                upload_logs=True,
                generate_tracelens_report=False,
                upload_tracelens_report=False,
            )
        assert result["logs"] == 1
        mock_logs.assert_called_once()
        call_kw = mock_logs.call_args[1]
        assert call_kw["artifact_path"] == "logs"

    def test_tracelens_upload_called_with_artifact_path_and_cleanup(self, tmp_path):
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "upload_tracelens_reports_to_mlflow",
            return_value=3,
        ) as mock_upload_tracelens:
            result = mlflow_artifacts_mod.upload_artifacts_to_mlflow(
                mock_writer,
                tensorboard_dir=str(tmp_path),
                exp_root_path=str(tmp_path),
                upload_traces=False,
                upload_logs=False,
                generate_tracelens_report=False,
                upload_tracelens_report=True,
                tracelens_ranks=[0, 8],
                tracelens_output_format="xlsx",
                tracelens_cleanup_after_upload=True,
            )
        assert result["tracelens_reports"] == 3
        mock_upload_tracelens.assert_called_once()
        call_kw = mock_upload_tracelens.call_args[1]
        assert call_kw["artifact_path"] == "trace_analysis"
        assert call_kw["cleanup_after_upload"] is True
        assert call_kw["ranks"] == [0, 8]
        assert call_kw["output_format"] == "xlsx"

    def test_tracelens_generate_locally_only_when_generate_true_upload_false(self, tmp_path):
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "generate_tracelens_reports_locally",
            return_value=5,
        ) as mock_local:
            with patch.object(
                mlflow_artifacts_mod,
                "upload_tracelens_reports_to_mlflow",
            ) as mock_upload:
                result = mlflow_artifacts_mod.upload_artifacts_to_mlflow(
                    mock_writer,
                    tensorboard_dir=str(tmp_path),
                    exp_root_path=str(tmp_path),
                    upload_traces=False,
                    upload_logs=False,
                    generate_tracelens_report=True,
                    upload_tracelens_report=False,
                )
        assert result["tracelens_reports"] == 0
        mock_local.assert_called_once()
        mock_upload.assert_not_called()

    def test_no_tracelens_calls_when_both_generate_and_upload_false(self, tmp_path):
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "generate_tracelens_reports_locally",
        ) as mock_local:
            with patch.object(
                mlflow_artifacts_mod,
                "upload_tracelens_reports_to_mlflow",
            ) as mock_upload:
                result = mlflow_artifacts_mod.upload_artifacts_to_mlflow(
                    mock_writer,
                    tensorboard_dir=str(tmp_path),
                    exp_root_path=str(tmp_path),
                    upload_traces=False,
                    upload_logs=False,
                    generate_tracelens_report=False,
                    upload_tracelens_report=False,
                )
        assert result["tracelens_reports"] == 0
        mock_local.assert_not_called()
        mock_upload.assert_not_called()

    def test_trace_and_log_discovery_integration(self, tmp_path):
        """Trace and log files are discovered and upload helpers called with correct paths."""
        (tmp_path / "rank_0.pt.trace.json").write_text("{}")
        (tmp_path / "logs" / "master" / "m.log").parent.mkdir(parents=True)
        (tmp_path / "logs" / "master" / "m.log").write_text("log")
        mock_writer = MagicMock()
        with patch.object(
            mlflow_artifacts_mod,
            "upload_trace_files_to_mlflow",
            return_value=1,
        ) as mock_traces:
            with patch.object(
                mlflow_artifacts_mod,
                "upload_log_files_to_mlflow",
                return_value=1,
            ) as mock_logs:
                result = mlflow_artifacts_mod.upload_artifacts_to_mlflow(
                    mock_writer,
                    tensorboard_dir=str(tmp_path),
                    exp_root_path=str(tmp_path),
                    upload_traces=True,
                    upload_logs=True,
                    generate_tracelens_report=False,
                    upload_tracelens_report=False,
                )
        assert result["traces"] == 1
        assert result["logs"] == 1
        assert result["tracelens_reports"] == 0
        mock_traces.assert_called_once_with(mock_writer, str(tmp_path), artifact_path="traces")
        mock_logs.assert_called_once_with(mock_writer, str(tmp_path), artifact_path="logs")


# -----------------------------------------------------------------------------
# Log file discovery
# -----------------------------------------------------------------------------


class TestGetAllLogFiles:
    """Test _get_all_log_files."""

    def test_returns_empty_for_empty_exp_root(self):
        out = mlflow_artifacts_mod._get_all_log_files("")
        assert out == []

    def test_returns_empty_when_logs_dir_missing(self, tmp_path):
        out = mlflow_artifacts_mod._get_all_log_files(str(tmp_path))
        assert out == []

    def test_finds_log_files_recursively(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "master" / "master-0.log").parent.mkdir(parents=True)
        (logs_dir / "master" / "master-0.log").write_text("log")
        (logs_dir / "train" / "rank-0" / "rank-0.log").parent.mkdir(parents=True)
        (logs_dir / "train" / "rank-0" / "rank-0.log").write_text("log")
        out = mlflow_artifacts_mod._get_all_log_files(str(tmp_path))
        assert len(out) == 2


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


def test_tracelens_install_ref_constant():
    """TRACELENS_INSTALL_REF is set for reproducibility."""
    assert hasattr(mlflow_artifacts_mod, "TRACELENS_INSTALL_REF")
    assert isinstance(mlflow_artifacts_mod.TRACELENS_INSTALL_REF, str)
    assert len(mlflow_artifacts_mod.TRACELENS_INSTALL_REF) > 0

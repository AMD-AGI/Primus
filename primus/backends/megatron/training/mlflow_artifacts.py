###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MLflow Artifact Logging Utilities with TraceLens Integration

This module provides functions to upload trace files, log files, and
TraceLens analysis reports to MLflow when MLflow tracking is enabled.

Features:
- Upload profiler trace files from all profiled ranks (including multi-node)
- Upload log files from all levels and all ranks
- Generate and upload TraceLens trace analysis reports
- Supports both local and distributed training scenarios

MLflow Artifact Structure:
    artifacts/
    ├── traces/              # PyTorch profiler trace files
    │   ├── rank_0_step_2.json.gz
    │   └── ...
    ├── logs/                # Training log files
    │   └── log_mp_pretrain.txt
    └── trace_analysis/      # TraceLens analysis reports
        ├── rank_0_analysis.html
        └── ...
"""

import glob
import os
import subprocess
import sys
from typing import List, Optional

from primus.modules.module_utils import log_rank_0, warning_rank_0


def _get_all_trace_files(tensorboard_dir: str) -> list:
    """
    Find all profiler trace files in the tensorboard directory.

    Trace files are typically named like:
    - primus-megatron-exp[...]-rank[0].*.json
    - primus-megatron-exp[...]-rank[0].*.json.gz

    Args:
        tensorboard_dir: Path to the tensorboard directory containing trace files

    Returns:
        List of paths to trace files
    """
    if not tensorboard_dir or not os.path.exists(tensorboard_dir):
        return []

    trace_files = []
    # Look for JSON trace files (both compressed and uncompressed)
    patterns = ["*.json", "*.json.gz", "*.pt.trace.json", "*.pt.trace.json.gz"]
    for pattern in patterns:
        trace_files.extend(glob.glob(os.path.join(tensorboard_dir, pattern)))
        trace_files.extend(glob.glob(os.path.join(tensorboard_dir, "**", pattern), recursive=True))

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in trace_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def _get_all_log_files(exp_root_path: str) -> list:
    """
    Find all log files in the experiment logs directory.

    Log files are organized as:
    - {exp_root_path}/logs/master/master-*.log
    - {exp_root_path}/logs/{module_name}/rank-{rank}/*.log

    Args:
        exp_root_path: Root path of the experiment

    Returns:
        List of paths to log files
    """
    if not exp_root_path:
        return []

    logs_dir = os.path.join(exp_root_path, "logs")
    if not os.path.exists(logs_dir):
        return []

    log_files = []
    # Find all .log files recursively
    log_files.extend(glob.glob(os.path.join(logs_dir, "**", "*.log"), recursive=True))

    return log_files


def upload_trace_files_to_mlflow(
    mlflow_writer,
    tensorboard_dir: str,
    artifact_path: str = "traces",
) -> int:
    """
    Upload all profiler trace files to MLflow as artifacts.

    This function collects trace files from the tensorboard directory and
    uploads them to MLflow. In distributed settings, only rank 0 (or the
    last rank where MLflow writer is initialized) should call this.

    Args:
        mlflow_writer: The MLflow module instance (from get_mlflow_writer())
        tensorboard_dir: Path to the tensorboard directory containing trace files
        artifact_path: MLflow artifact subdirectory for trace files

    Returns:
        Number of trace files uploaded
    """
    if mlflow_writer is None:
        return 0

    log_rank_0(f"[MLflow] Searching for trace files in: {tensorboard_dir}")
    trace_files = _get_all_trace_files(tensorboard_dir)
    if len(trace_files) > 5:
        log_rank_0(f"[MLflow] Found {len(trace_files)} trace files: {trace_files[:5]}...")
    else:
        log_rank_0(f"[MLflow] Found {len(trace_files)} trace files: {trace_files}")

    if not trace_files:
        log_rank_0("[MLflow] No trace files found to upload")
        return 0

    uploaded_count = 0
    for trace_file in trace_files:
        try:
            # Get relative path from tensorboard_dir for artifact organization
            rel_path = os.path.relpath(trace_file, tensorboard_dir)
            # Determine artifact subdirectory based on file location
            artifact_subpath = (
                os.path.join(artifact_path, os.path.dirname(rel_path))
                if os.path.dirname(rel_path)
                else artifact_path
            )

            mlflow_writer.log_artifact(trace_file, artifact_path=artifact_subpath)
            uploaded_count += 1
            log_rank_0(f"[MLflow] Uploaded trace file: {os.path.basename(trace_file)}")
        except Exception as e:
            warning_rank_0(f"[MLflow] Failed to upload trace file {trace_file}: {e}")

    log_rank_0(f"[MLflow] Uploaded {uploaded_count} trace files to '{artifact_path}'")
    return uploaded_count


def upload_log_files_to_mlflow(
    mlflow_writer,
    exp_root_path: str,
    artifact_path: str = "logs",
) -> int:
    """
    Upload all log files to MLflow as artifacts.

    This function collects log files from all ranks and all log levels
    and uploads them to MLflow. The directory structure is preserved
    in the artifact path.

    Args:
        mlflow_writer: The MLflow module instance (from get_mlflow_writer())
        exp_root_path: Root path of the experiment
        artifact_path: MLflow artifact subdirectory for log files

    Returns:
        Number of log files uploaded
    """
    if mlflow_writer is None:
        return 0

    log_files = _get_all_log_files(exp_root_path)

    if not log_files:
        log_rank_0("[MLflow] No log files found to upload")
        return 0

    logs_base_dir = os.path.join(exp_root_path, "logs")
    uploaded_count = 0

    for log_file in log_files:
        try:
            # Preserve directory structure relative to logs base directory
            rel_path = os.path.relpath(log_file, logs_base_dir)
            artifact_subpath = (
                os.path.join(artifact_path, os.path.dirname(rel_path))
                if os.path.dirname(rel_path)
                else artifact_path
            )

            mlflow_writer.log_artifact(log_file, artifact_path=artifact_subpath)
            uploaded_count += 1
        except Exception as e:
            warning_rank_0(f"[MLflow] Failed to upload log file {log_file}: {e}")

    log_rank_0(f"[MLflow] Uploaded {uploaded_count} log files to '{artifact_path}'")
    return uploaded_count


# =============================================================================
# TraceLens Integration
# =============================================================================


def _ensure_tracelens_installed() -> bool:
    """
    Ensure TraceLens is installed. Install it if not present.

    Returns:
        True if TraceLens is available, False otherwise
    """
    try:
        import tracelens  # noqa: F401

        log_rank_0("[TraceLens] TraceLens is already installed")
        return True
    except ImportError:
        log_rank_0("[TraceLens] TraceLens not found, attempting to install...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "tracelens", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log_rank_0("[TraceLens] Successfully installed TraceLens")
            return True
        except subprocess.CalledProcessError as e:
            warning_rank_0(f"[TraceLens] Failed to install TraceLens: {e}")
            return False


def _extract_rank_from_filename(filename: str) -> Optional[int]:
    """
    Extract rank number from trace filename.

    Expected patterns:
    - rank_0_step_2.json.gz
    - primus-megatron-exp-rank[0].*.json

    Args:
        filename: The trace filename

    Returns:
        Rank number or None if not found
    """
    import re

    # Try pattern: rank_N_ or rank[N]
    patterns = [
        r"rank_(\d+)_",
        r"rank\[(\d+)\]",
        r"-rank(\d+)\.",
        r"_rank(\d+)\.",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))

    return None


def _filter_traces_by_rank(trace_files: List[str], ranks: List[int]) -> List[str]:
    """
    Filter trace files to only include specified ranks.

    Args:
        trace_files: List of trace file paths
        ranks: List of rank numbers to include

    Returns:
        Filtered list of trace files
    """
    if not ranks:
        return trace_files

    filtered = []
    for trace_file in trace_files:
        rank = _extract_rank_from_filename(os.path.basename(trace_file))
        if rank is not None and rank in ranks:
            filtered.append(trace_file)

    return filtered


def generate_tracelens_report(
    trace_file: str,
    output_dir: str,
    report_name: Optional[str] = None,
) -> Optional[str]:
    """
    Generate a TraceLens analysis report for a single trace file.

    Args:
        trace_file: Path to the PyTorch profiler trace file (JSON/JSON.GZ)
        output_dir: Directory to save the report
        report_name: Optional custom name for the report

    Returns:
        Path to the generated report, or None if generation failed
    """
    if not os.path.exists(trace_file):
        warning_rank_0(f"[TraceLens] Trace file not found: {trace_file}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Generate report name from trace filename if not provided
    if report_name is None:
        base_name = os.path.basename(trace_file)
        # Remove extensions like .json.gz
        for ext in [".json.gz", ".json", ".pt.trace.json.gz", ".pt.trace.json"]:
            if base_name.endswith(ext):
                base_name = base_name[: -len(ext)]
                break
        report_name = f"{base_name}_analysis.html"

    output_path = os.path.join(output_dir, report_name)

    try:
        # Try using tracelens CLI
        result = subprocess.run(
            ["tracelens", "analyze", trace_file, "-o", output_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per trace
        )

        if result.returncode == 0 and os.path.exists(output_path):
            log_rank_0(f"[TraceLens] Generated report: {report_name}")
            return output_path
        else:
            # Try alternative: tracelens as Python module
            result = subprocess.run(
                [sys.executable, "-m", "tracelens", "analyze", trace_file, "-o", output_path],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0 and os.path.exists(output_path):
                log_rank_0(f"[TraceLens] Generated report: {report_name}")
                return output_path

            warning_rank_0(f"[TraceLens] Failed to generate report: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        warning_rank_0(f"[TraceLens] Report generation timed out for: {trace_file}")
        return None
    except FileNotFoundError:
        warning_rank_0("[TraceLens] tracelens command not found")
        return None
    except Exception as e:
        warning_rank_0(f"[TraceLens] Error generating report: {e}")
        return None


def generate_tracelens_reports(
    tensorboard_dir: str,
    output_dir: str,
    ranks: Optional[List[int]] = None,
    max_reports: Optional[int] = None,
) -> List[str]:
    """
    Generate TraceLens analysis reports for trace files.

    Args:
        tensorboard_dir: Directory containing PyTorch profiler trace files
        output_dir: Directory to save the generated reports
        ranks: List of ranks to generate reports for (None = all ranks)
        max_reports: Maximum number of reports to generate (None = unlimited)

    Returns:
        List of paths to generated reports
    """
    if not _ensure_tracelens_installed():
        warning_rank_0("[TraceLens] Cannot generate reports: TraceLens not available")
        return []

    trace_files = _get_all_trace_files(tensorboard_dir)
    if not trace_files:
        log_rank_0("[TraceLens] No trace files found for analysis")
        return []

    # Filter by ranks if specified
    if ranks is not None:
        trace_files = _filter_traces_by_rank(trace_files, ranks)
        log_rank_0(f"[TraceLens] Filtered to {len(trace_files)} trace files for ranks: {ranks}")

    # Limit number of reports if specified
    if max_reports is not None and len(trace_files) > max_reports:
        trace_files = trace_files[:max_reports]
        log_rank_0(f"[TraceLens] Limited to {max_reports} reports")

    log_rank_0(f"[TraceLens] Generating reports for {len(trace_files)} trace files...")

    generated_reports = []
    for trace_file in trace_files:
        report_path = generate_tracelens_report(trace_file, output_dir)
        if report_path:
            generated_reports.append(report_path)

    log_rank_0(f"[TraceLens] Generated {len(generated_reports)} reports")
    return generated_reports


def upload_tracelens_reports_to_mlflow(
    mlflow_writer,
    tensorboard_dir: str,
    exp_root_path: str,
    ranks: Optional[List[int]] = None,
    max_reports: Optional[int] = None,
    artifact_path: str = "trace_analysis",
) -> int:
    """
    Generate TraceLens reports and upload them to MLflow.

    This function:
    1. Finds PyTorch profiler trace files
    2. Generates TraceLens analysis reports for specified ranks
    3. Uploads the reports to MLflow under the trace_analysis artifact path

    Args:
        mlflow_writer: The MLflow module instance (from get_mlflow_writer())
        tensorboard_dir: Directory containing PyTorch profiler trace files
        exp_root_path: Root path of the experiment (for saving reports)
        ranks: List of ranks to analyze (None = all ranks, [0] = rank 0 only)
        max_reports: Maximum number of reports to generate
        artifact_path: MLflow artifact subdirectory for reports

    Returns:
        Number of reports uploaded to MLflow
    """
    if mlflow_writer is None:
        log_rank_0("[TraceLens] MLflow writer not available, skipping report upload")
        return 0

    # Create output directory for reports
    reports_dir = os.path.join(exp_root_path, "tracelens_reports")
    os.makedirs(reports_dir, exist_ok=True)

    log_rank_0(f"[TraceLens] Generating reports from traces in: {tensorboard_dir}")
    log_rank_0(f"[TraceLens] Reports will be saved to: {reports_dir}")
    if ranks:
        log_rank_0(f"[TraceLens] Analyzing ranks: {ranks}")
    if max_reports:
        log_rank_0(f"[TraceLens] Max reports: {max_reports}")

    # Generate reports
    reports = generate_tracelens_reports(
        tensorboard_dir=tensorboard_dir,
        output_dir=reports_dir,
        ranks=ranks,
        max_reports=max_reports,
    )

    if not reports:
        log_rank_0("[TraceLens] No reports generated, nothing to upload")
        return 0

    # Upload reports to MLflow
    uploaded_count = 0
    for report_path in reports:
        try:
            mlflow_writer.log_artifact(report_path, artifact_path=artifact_path)
            uploaded_count += 1
            log_rank_0(f"[MLflow] Uploaded TraceLens report: {os.path.basename(report_path)}")
        except Exception as e:
            warning_rank_0(f"[MLflow] Failed to upload report {report_path}: {e}")

    log_rank_0(f"[TraceLens] Uploaded {uploaded_count} reports to '{artifact_path}'")
    return uploaded_count


# =============================================================================
# Main Entry Point
# =============================================================================


def upload_artifacts_to_mlflow(
    mlflow_writer,
    tensorboard_dir: Optional[str] = None,
    exp_root_path: Optional[str] = None,
    upload_traces: bool = True,
    upload_logs: bool = True,
    upload_tracelens_report: bool = False,
    tracelens_ranks: Optional[List[int]] = None,
    tracelens_max_reports: Optional[int] = None,
) -> dict:
    """
    Upload all artifacts (trace files, log files, TraceLens reports) to MLflow.

    This is the main entry point for uploading artifacts to MLflow.
    It handles:
    - Trace files from PyTorch profiler
    - Log files from training
    - TraceLens analysis reports (optional)

    MLflow Artifact Structure:
        artifacts/
        ├── traces/              # PyTorch profiler trace files
        ├── logs/                # Training log files
        └── trace_analysis/      # TraceLens analysis reports

    Args:
        mlflow_writer: The MLflow module instance (from get_mlflow_writer())
        tensorboard_dir: Path to the tensorboard directory containing trace files
        exp_root_path: Root path of the experiment for log files
        upload_traces: Whether to upload trace files
        upload_logs: Whether to upload log files
        upload_tracelens_report: Whether to generate and upload TraceLens reports
        tracelens_ranks: List of ranks to generate TraceLens reports for
                        (None = all ranks, [0] = rank 0 only)
        tracelens_max_reports: Maximum number of TraceLens reports to generate

    Returns:
        Dictionary with counts of uploaded files:
        {
            "traces": <number of trace files uploaded>,
            "logs": <number of log files uploaded>,
            "tracelens_reports": <number of TraceLens reports uploaded>
        }
    """
    if mlflow_writer is None:
        log_rank_0("[MLflow] MLflow writer not available, skipping artifact upload")
        return {"traces": 0, "logs": 0, "tracelens_reports": 0}

    log_rank_0("[MLflow] Starting artifact upload to MLflow...")
    log_rank_0(f"[MLflow] tensorboard_dir: {tensorboard_dir}")
    log_rank_0(f"[MLflow] exp_root_path: {exp_root_path}")
    log_rank_0(f"[MLflow] upload_traces: {upload_traces}, upload_logs: {upload_logs}")
    log_rank_0(f"[MLflow] upload_tracelens_report: {upload_tracelens_report}")

    result = {"traces": 0, "logs": 0, "tracelens_reports": 0}

    # Upload trace files
    if upload_traces and tensorboard_dir:
        result["traces"] = upload_trace_files_to_mlflow(
            mlflow_writer, tensorboard_dir, artifact_path="traces"
        )

    # Upload log files
    if upload_logs and exp_root_path:
        result["logs"] = upload_log_files_to_mlflow(mlflow_writer, exp_root_path, artifact_path="logs")

    # Generate and upload TraceLens reports
    if upload_tracelens_report and tensorboard_dir and exp_root_path:
        result["tracelens_reports"] = upload_tracelens_reports_to_mlflow(
            mlflow_writer=mlflow_writer,
            tensorboard_dir=tensorboard_dir,
            exp_root_path=exp_root_path,
            ranks=tracelens_ranks,
            max_reports=tracelens_max_reports,
            artifact_path="trace_analysis",
        )

    log_rank_0(
        f"[MLflow] Artifact upload complete: "
        f"{result['traces']} traces, {result['logs']} logs, "
        f"{result['tracelens_reports']} TraceLens reports"
    )

    return result

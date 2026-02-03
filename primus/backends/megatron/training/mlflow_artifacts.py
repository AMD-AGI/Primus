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
        ├── rank_0_analysis.xlsx   # Multi-tab Excel (default)
        └── ...

TraceLens Report Formats:
    - xlsx: Multi-tab Excel with sections for kernels, memory, communication, etc.
    - csv:  Multiple CSV files per rank (kernels, memory, communication, etc.)
    - all:  Both xlsx and csv files (default)
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
    - *.pt.trace.json
    - *.pt.trace.json.gz

    Args:
        tensorboard_dir: Path to the tensorboard directory containing trace files

    Returns:
        List of paths to trace files
    """
    if not tensorboard_dir or not os.path.exists(tensorboard_dir):
        return []

    trace_files = []
    # Look for PyTorch profiler trace files (both compressed and uncompressed)
    # Using specific patterns to avoid matching unrelated JSON files
    patterns = ["*.pt.trace.json", "*.pt.trace.json.gz"]
    # Escape directory path to handle special characters like [] in experiment names
    escaped_dir = glob.escape(tensorboard_dir)
    for pattern in patterns:
        trace_files.extend(glob.glob(os.path.join(escaped_dir, pattern)))
        trace_files.extend(glob.glob(os.path.join(escaped_dir, "**", pattern), recursive=True))

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
    # Find all .log files recursively (escape path to handle special characters)
    log_files.extend(glob.glob(os.path.join(glob.escape(logs_dir), "**", "*.log"), recursive=True))

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


def _ensure_openpyxl_installed() -> bool:
    """
    Ensure openpyxl is installed for XLSX generation.

    Returns:
        True if openpyxl is available, False otherwise
    """
    try:
        import openpyxl  # noqa: F401

        return True
    except ImportError:
        log_rank_0("[TraceLens] openpyxl not found, installing for XLSX support...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "openpyxl", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log_rank_0("[TraceLens] Successfully installed openpyxl")
            return True
        except subprocess.CalledProcessError as e:
            warning_rank_0(f"[TraceLens] Failed to install openpyxl: {e}")
            return False


def _ensure_tracelens_installed() -> bool:
    """
    Ensure TraceLens and its dependencies are installed.

    TraceLens is available from GitHub: https://github.com/AMD-AGI/TraceLens
    XLSX generation requires openpyxl which is installed separately.

    Returns:
        True if TraceLens is available, False otherwise
    """
    try:
        import TraceLens  # noqa: F401

        log_rank_0("[TraceLens] TraceLens is already installed")
    except ImportError:
        log_rank_0("[TraceLens] TraceLens not found, attempting to install from GitHub...")
        try:
            # TraceLens is on GitHub, not PyPI
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "git+https://github.com/AMD-AGI/TraceLens.git",
                    "-q",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log_rank_0("[TraceLens] Successfully installed TraceLens from GitHub")
        except subprocess.CalledProcessError as e:
            warning_rank_0(f"[TraceLens] Failed to install TraceLens: {e}")
            return False

    # Ensure openpyxl is installed for XLSX generation
    _ensure_openpyxl_installed()

    return True


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
    output_format: str = "all",
) -> List[str]:
    """
    Generate a TraceLens analysis report for a single trace file.

    Args:
        trace_file: Path to the PyTorch profiler trace file (JSON/JSON.GZ)
        output_dir: Directory to save the report
        report_name: Optional custom name for the report (base name for CSVs)
        output_format: Output format:
                      - "all" (default): Both XLSX and CSV files
                      - "xlsx": Single multi-tab Excel file with detailed analysis
                      - "csv": Multiple CSV files (kernels, memory, communication, etc.)

    Returns:
        List of paths to generated report files
    """
    if not os.path.exists(trace_file):
        warning_rank_0(f"[TraceLens] Trace file not found: {trace_file}")
        return []

    os.makedirs(output_dir, exist_ok=True)

    # Generate base name from trace filename if not provided
    if report_name is None:
        base_name = os.path.basename(trace_file)
        # Remove extensions like .json.gz
        for trace_ext in [".json.gz", ".json", ".pt.trace.json.gz", ".pt.trace.json"]:
            if base_name.endswith(trace_ext):
                base_name = base_name[: -len(trace_ext)]
                break
        report_name = base_name

    try:
        # Try using TraceLens Python API directly
        from TraceLens.Reporting import generate_perf_report_pytorch

        generated_files = []

        # For "all" format: TraceLens uses either/or logic - if output_csvs_dir is set,
        # it ONLY generates CSVs. So we need to call it twice for both formats.
        # Note: This means the trace file is parsed twice, roughly doubling processing time
        # compared to a single format. This is a TraceLens limitation, not a bug.
        if output_format == "all":
            xlsx_path = os.path.join(output_dir, f"{report_name}_analysis.xlsx")
            csv_subdir = os.path.join(output_dir, report_name)
            os.makedirs(csv_subdir, exist_ok=True)

            # First call: Generate XLSX only
            dfs = generate_perf_report_pytorch(trace_file, output_xlsx_path=xlsx_path)

            # Check XLSX output
            if os.path.exists(xlsx_path):
                num_tabs = len(dfs) if dfs else 0
                log_rank_0(
                    f"[TraceLens] Generated XLSX report with {num_tabs} tabs: {os.path.basename(xlsx_path)}"
                )
                generated_files.append(xlsx_path)

            # Second call: Generate CSVs only
            generate_perf_report_pytorch(trace_file, output_csvs_dir=csv_subdir)

            # Check CSV outputs (escape path to handle [] characters in filenames)
            csv_files = glob.glob(os.path.join(glob.escape(csv_subdir), "*.csv"))
            if csv_files:
                log_rank_0(f"[TraceLens] Generated {len(csv_files)} CSV files for {report_name}")
                generated_files.append(csv_subdir)  # Upload directory to preserve structure

        elif output_format == "xlsx":
            # XLSX only: Single file with multiple tabs
            xlsx_path = os.path.join(output_dir, f"{report_name}_analysis.xlsx")
            dfs = generate_perf_report_pytorch(trace_file, output_xlsx_path=xlsx_path)
            if os.path.exists(xlsx_path):
                num_tabs = len(dfs) if dfs else 0
                log_rank_0(
                    f"[TraceLens] Generated XLSX report with {num_tabs} tabs: {os.path.basename(xlsx_path)}"
                )
                generated_files.append(xlsx_path)

        elif output_format == "csv":
            # CSV only: Multiple files in a subdirectory per rank
            csv_subdir = os.path.join(output_dir, report_name)
            os.makedirs(csv_subdir, exist_ok=True)
            dfs = generate_perf_report_pytorch(trace_file, output_csvs_dir=csv_subdir)

            # Collect all generated CSV files (escape path to handle [] characters in filenames)
            csv_files = glob.glob(os.path.join(glob.escape(csv_subdir), "*.csv"))
            if csv_files:
                log_rank_0(f"[TraceLens] Generated {len(csv_files)} CSV files for {report_name}")
                generated_files.append(csv_subdir)  # Upload directory to preserve structure

        if generated_files:
            return generated_files

        warning_rank_0(f"[TraceLens] No output files generated for: {trace_file}")
        return []

    except ImportError:
        warning_rank_0(
            "[TraceLens] TraceLens not available. Using simplified fallback CSV summary. "
            "Install TraceLens for comprehensive kernel, memory, and communication analysis."
        )
        # Fallback to simple CSV summary (basic stats only, may not handle all trace formats)
        csv_path = _generate_trace_summary_csv(trace_file, output_dir, f"{report_name}_summary.csv")
        return [csv_path] if csv_path else []

    except Exception as e:
        warning_rank_0(
            f"[TraceLens] Error generating report: {e}. "
            "Using simplified fallback CSV summary with basic statistics only."
        )
        # Fallback to simple CSV summary (basic stats only, may not handle all trace formats)
        csv_path = _generate_trace_summary_csv(trace_file, output_dir, f"{report_name}_summary.csv")
        return [csv_path] if csv_path else []


def _generate_trace_summary_csv(
    trace_file: str,
    output_dir: str,
    report_name: str,
) -> Optional[str]:
    """
    Generate a CSV summary from a PyTorch profiler trace file.

    This is a fallback when TraceLens is not available.
    Extracts key metrics from the trace JSON and writes to CSV.

    Args:
        trace_file: Path to the trace file
        output_dir: Output directory
        report_name: Name for the CSV file

    Returns:
        Path to generated CSV or None if failed
    """
    import csv
    import gzip
    import json

    try:
        # Load trace file
        if trace_file.endswith(".gz"):
            with gzip.open(trace_file, "rt", encoding="utf-8") as f:
                trace_data = json.load(f)
        else:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace_data = json.load(f)

        # Extract events from trace
        events = trace_data.get("traceEvents", [])
        if not events:
            warning_rank_0(f"[TraceLens] No events found in trace: {trace_file}")
            return None

        # Aggregate kernel/operation statistics
        op_stats = {}
        for event in events:
            if event.get("cat") in ["kernel", "gpu_memcpy", "cuda_runtime", "cpu_op"]:
                name = event.get("name", "unknown")
                dur = event.get("dur", 0)  # duration in microseconds

                if name not in op_stats:
                    op_stats[name] = {"count": 0, "total_us": 0, "min_us": float("inf"), "max_us": 0}

                op_stats[name]["count"] += 1
                op_stats[name]["total_us"] += dur
                op_stats[name]["min_us"] = min(op_stats[name]["min_us"], dur)
                op_stats[name]["max_us"] = max(op_stats[name]["max_us"], dur)

        if not op_stats:
            warning_rank_0(f"[TraceLens] No kernel/op events found in trace: {trace_file}")
            return None

        # Sort by total time descending
        sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["total_us"], reverse=True)

        # Write CSV
        output_path = os.path.join(output_dir, report_name)
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Operation",
                    "Count",
                    "Total Time (ms)",
                    "Avg Time (ms)",
                    "Min Time (ms)",
                    "Max Time (ms)",
                    "% of Total",
                ]
            )

            total_time = sum(stats["total_us"] for _, stats in sorted_ops)
            for name, stats in sorted_ops:
                avg_us = stats["total_us"] / stats["count"] if stats["count"] > 0 else 0
                pct = (stats["total_us"] / total_time * 100) if total_time > 0 else 0
                writer.writerow(
                    [
                        name,
                        stats["count"],
                        f"{stats['total_us'] / 1000:.3f}",
                        f"{avg_us / 1000:.3f}",
                        f"{stats['min_us'] / 1000:.3f}",
                        f"{stats['max_us'] / 1000:.3f}",
                        f"{pct:.2f}",
                    ]
                )

        log_rank_0(f"[TraceLens] Generated CSV summary: {report_name} ({len(sorted_ops)} operations)")
        return output_path

    except json.JSONDecodeError as e:
        warning_rank_0(f"[TraceLens] Failed to parse trace JSON: {e}")
        return None
    except Exception as e:
        warning_rank_0(f"[TraceLens] Error generating CSV summary: {e}")
        return None


def generate_tracelens_reports(
    tensorboard_dir: str,
    output_dir: str,
    ranks: Optional[List[int]] = None,
    output_format: str = "all",
) -> List[str]:
    """
    Generate TraceLens analysis reports for trace files.

    Args:
        tensorboard_dir: Directory containing PyTorch profiler trace files
        output_dir: Directory to save the generated reports
        ranks: List of ranks to generate reports for (None = all ranks)
               To limit number of reports, specify fewer ranks in the list
        output_format: Output format:
                      - "all" (default): Both XLSX and CSV files
                      - "xlsx": Multi-tab Excel with detailed analysis
                      - "csv": Multiple CSV files per rank (kernels, memory, comm, etc.)

    Returns:
        List of paths to all generated report files
    """
    # Try to install tracelens, but continue with fallback if not available
    _ensure_tracelens_installed()

    # Normalize ranks parameter: handle string input from config parser
    if ranks is not None and isinstance(ranks, str):
        import ast

        try:
            ranks = ast.literal_eval(ranks)
            if not isinstance(ranks, list):
                log_rank_0(
                    f"[TraceLens] Warning: ranks evaluated to {type(ranks).__name__}, expected list. Using None."
                )
                ranks = None
        except (ValueError, SyntaxError) as e:
            log_rank_0(f"[TraceLens] Warning: Failed to parse ranks '{ranks}': {e}. Using None.")
            ranks = None

    trace_files = _get_all_trace_files(tensorboard_dir)
    if not trace_files:
        log_rank_0("[TraceLens] No trace files found for analysis")
        return []

    # Filter by ranks if specified
    if ranks is not None:
        original_count = len(trace_files)
        trace_files = _filter_traces_by_rank(trace_files, ranks)
        log_rank_0(f"[TraceLens] Filtered to {len(trace_files)} trace files for ranks: {ranks}")
        if not trace_files and original_count > 0:
            warning_rank_0(
                f"[TraceLens] Warning: No trace files match the specified ranks {ranks}. "
                f"Found {original_count} trace files but none matched. "
                "Check that the rank numbers are correct."
            )

    log_rank_0(
        f"[TraceLens] Generating {output_format.upper()} reports for {len(trace_files)} trace files..."
    )

    generated_reports = []
    for trace_file in trace_files:
        # generate_tracelens_report now returns a list of files
        report_paths = generate_tracelens_report(trace_file, output_dir, output_format=output_format)
        generated_reports.extend(report_paths)

    log_rank_0(f"[TraceLens] Generated {len(generated_reports)} report files from {len(trace_files)} traces")
    return generated_reports


def generate_tracelens_reports_locally(
    tensorboard_dir: str,
    exp_root_path: str,
    ranks: Optional[List[int]] = None,
    output_format: str = "all",
) -> int:
    """
    Generate TraceLens analysis reports locally (without MLflow upload).

    This function generates TraceLens reports and saves them to
    exp_root_path/tracelens_reports/ for local inspection.

    Args:
        tensorboard_dir: Directory containing PyTorch profiler trace files
        exp_root_path: Root path of the experiment (for saving reports)
        ranks: List of ranks to analyze (None = all ranks, [0] = rank 0 only)
               Specify fewer ranks to limit number of reports
        output_format: Report format - "all" (default, xlsx+csv), "xlsx", or "csv"

    Returns:
        Number of reports generated

    Example:
        >>> generate_tracelens_reports_locally(
        ...     tensorboard_dir="/path/to/tensorboard",
        ...     exp_root_path="/path/to/experiment",
        ...     ranks=[0, 8],  # Only 2 ranks = 2 reports
        ...     output_format="all"
        ... )
        26  # Generated 26 report files (XLSX + CSVs for 2 ranks)
    """
    # Create output directory for reports
    reports_dir = os.path.join(exp_root_path, "tracelens_reports")
    os.makedirs(reports_dir, exist_ok=True)

    log_rank_0(f"[TraceLens] Generating reports from traces in: {tensorboard_dir}")
    log_rank_0(f"[TraceLens] Reports will be saved to: {reports_dir}")
    if ranks:
        log_rank_0(f"[TraceLens] Analyzing ranks: {ranks}")

    # Generate reports
    reports = generate_tracelens_reports(
        tensorboard_dir=tensorboard_dir,
        output_dir=reports_dir,
        ranks=ranks,
        output_format=output_format,
    )

    if not reports:
        log_rank_0("[TraceLens] No reports generated")
        return 0

    log_rank_0(f"[TraceLens] Generated {len(reports)} report files locally")
    return len(reports)


def upload_tracelens_reports_to_mlflow(
    mlflow_writer,
    tensorboard_dir: str,
    exp_root_path: str,
    ranks: Optional[List[int]] = None,
    output_format: str = "all",
    artifact_path: str = "trace_analysis",
    cleanup_after_upload: bool = False,
) -> int:
    """
    Generate TraceLens reports and upload them to MLflow.

    This function:
    1. Finds PyTorch profiler trace files
    2. Generates TraceLens analysis reports for specified ranks
    3. Uploads the reports to MLflow under the trace_analysis artifact path
    4. Optionally cleans up local report files after successful upload

    Args:
        mlflow_writer: The MLflow module instance (from get_mlflow_writer())
        tensorboard_dir: Directory containing PyTorch profiler trace files
        exp_root_path: Root path of the experiment (for saving reports)
        ranks: List of ranks to analyze (None = all ranks, [0] = rank 0 only)
               Specify fewer ranks to limit number of reports
        output_format: Report format - "all" (default, xlsx+csv), "xlsx", or "csv"
        artifact_path: MLflow artifact subdirectory for reports
        cleanup_after_upload: If True, removes local reports after upload to save disk space.
                             If False, keeps reports locally for inspection. Default: False.

    Returns:
        Number of reports uploaded to MLflow

    Note:
        Reports are saved to exp_root_path/tracelens_reports/ and kept locally by default.
        Set cleanup_after_upload=True to remove them after upload and save disk space.
    """
    if mlflow_writer is None:
        log_rank_0("[TraceLens] MLflow writer not available, skipping report upload")
        return 0

    # Normalize ranks parameter: handle string input from config parser
    if ranks is not None and isinstance(ranks, str):
        import ast

        try:
            ranks = ast.literal_eval(ranks)
            if not isinstance(ranks, list):
                log_rank_0(
                    f"[TraceLens] Warning: ranks evaluated to {type(ranks).__name__}, expected list. Using None."
                )
                ranks = None
        except (ValueError, SyntaxError) as e:
            log_rank_0(f"[TraceLens] Warning: Failed to parse ranks '{ranks}': {e}. Using None.")
            ranks = None

    # Create output directory for reports
    reports_dir = os.path.join(exp_root_path, "tracelens_reports")
    os.makedirs(reports_dir, exist_ok=True)

    log_rank_0(f"[TraceLens] Generating reports from traces in: {tensorboard_dir}")
    log_rank_0(f"[TraceLens] Reports will be saved to: {reports_dir}")
    if ranks:
        log_rank_0(f"[TraceLens] Analyzing ranks: {ranks}")

    # Generate reports
    reports = generate_tracelens_reports(
        tensorboard_dir=tensorboard_dir,
        output_dir=reports_dir,
        ranks=ranks,
        output_format=output_format,
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

    # Optionally clean up local reports after successful upload to save disk space
    if cleanup_after_upload:
        try:
            import shutil

            shutil.rmtree(reports_dir)
            log_rank_0(f"[TraceLens] Cleaned up local reports directory: {reports_dir}")
        except Exception as e:
            warning_rank_0(f"[TraceLens] Failed to cleanup reports directory: {e}")
    else:
        log_rank_0(f"[TraceLens] Keeping local reports at: {reports_dir}")

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
    generate_tracelens_report: bool = False,
    upload_tracelens_report: bool = False,
    tracelens_ranks: Optional[List[int]] = None,
    tracelens_output_format: str = "all",
    tracelens_cleanup_after_upload: bool = False,
) -> dict:
    """
    Upload all artifacts (trace files, log files, TraceLens reports) to MLflow.

    This is the main entry point for uploading artifacts to MLflow.
    It handles:
    - Trace files from PyTorch profiler
    - Log files from training
    - TraceLens analysis reports (optional - generate locally and/or upload to MLflow)

    MLflow Artifact Structure:
        artifacts/
        ├── traces/              # PyTorch profiler trace files
        ├── logs/                # Training log files
        └── trace_analysis/      # TraceLens analysis reports (if uploaded)

    TraceLens Report Generation Logic:
        - If upload_tracelens_report=True:  Generate AND upload (auto-enables generation)
        - If generate_tracelens_report=True and upload_tracelens_report=False: Generate locally only
        - If both False: No report generation

        Examples:
            generate=False, upload=False  →  No reports
            generate=True,  upload=False  →  Generate locally only
            generate=False, upload=True   →  Generate AND upload (auto-enabled)
            generate=True,  upload=True   →  Generate AND upload (explicit)

    Args:
        mlflow_writer: The MLflow module instance (from get_mlflow_writer())
        tensorboard_dir: Path to the tensorboard directory containing trace files
        exp_root_path: Root path of the experiment for log files
        upload_traces: Whether to upload trace files
        upload_logs: Whether to upload log files
        generate_tracelens_report: Whether to generate TraceLens reports locally
        upload_tracelens_report: Whether to upload TraceLens reports to MLflow (implies generation)
        tracelens_ranks: List of ranks to generate TraceLens reports for
                        (None = all ranks, [0, 8] = ranks 0 and 8 only)
                        Specify fewer ranks to limit number of reports
        tracelens_output_format: Report format - "all" (default, xlsx+csv), "xlsx", or "csv"
        tracelens_cleanup_after_upload: If True, removes local reports after upload to save disk space.
                                       If False, keeps reports locally for inspection (default).

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
    log_rank_0(
        f"[MLflow] generate_tracelens_report: {generate_tracelens_report}, "
        f"upload_tracelens_report: {upload_tracelens_report}"
    )

    result = {"traces": 0, "logs": 0, "tracelens_reports": 0}

    # Upload trace files
    if upload_traces and tensorboard_dir:
        result["traces"] = upload_trace_files_to_mlflow(
            mlflow_writer, tensorboard_dir, artifact_path="traces"
        )

    # Upload log files
    if upload_logs and exp_root_path:
        result["logs"] = upload_log_files_to_mlflow(mlflow_writer, exp_root_path, artifact_path="logs")

    # TraceLens report generation and upload logic
    # If upload=True, auto-enable generation (even if generate=False)
    should_generate = generate_tracelens_report or upload_tracelens_report

    if should_generate and tensorboard_dir and exp_root_path:
        if upload_tracelens_report:
            # Generate AND upload to MLflow
            log_rank_0("[TraceLens] Mode: Generate and upload to MLflow")
            result["tracelens_reports"] = upload_tracelens_reports_to_mlflow(
                mlflow_writer=mlflow_writer,
                tensorboard_dir=tensorboard_dir,
                exp_root_path=exp_root_path,
                ranks=tracelens_ranks,
                output_format=tracelens_output_format,
                artifact_path="trace_analysis",
                cleanup_after_upload=tracelens_cleanup_after_upload,
            )
        else:
            # Generate locally only (no MLflow upload)
            log_rank_0("[TraceLens] Mode: Generate locally only (no MLflow upload)")
            num_generated = generate_tracelens_reports_locally(
                tensorboard_dir=tensorboard_dir,
                exp_root_path=exp_root_path,
                ranks=tracelens_ranks,
                output_format=tracelens_output_format,
            )
            # Don't count as "uploaded" since they're local-only
            log_rank_0(f"[TraceLens] Generated {num_generated} report files (not uploaded to MLflow)")

    log_rank_0(
        f"[MLflow] Artifact upload complete: "
        f"{result['traces']} traces, {result['logs']} logs, "
        f"{result['tracelens_reports']} TraceLens reports"
    )

    return result

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MLflow Artifact Logging Utilities

This module provides functions to upload trace files and log files to MLflow
when MLflow tracking is enabled.

Features:
- Upload profiler trace files from all profiled ranks (including multi-node)
- Upload log files from all levels and all ranks
- Supports both local and distributed training scenarios
"""

import glob
import os
from typing import Optional

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


def upload_artifacts_to_mlflow(
    mlflow_writer,
    tensorboard_dir: Optional[str] = None,
    exp_root_path: Optional[str] = None,
    upload_traces: bool = True,
    upload_logs: bool = True,
) -> dict:
    """
    Upload all artifacts (trace files and log files) to MLflow.

    This is the main entry point for uploading artifacts to MLflow.
    It handles both trace files from profiling and log files from training.

    Args:
        mlflow_writer: The MLflow module instance (from get_mlflow_writer())
        tensorboard_dir: Path to the tensorboard directory containing trace files
        exp_root_path: Root path of the experiment for log files
        upload_traces: Whether to upload trace files
        upload_logs: Whether to upload log files

    Returns:
        Dictionary with counts of uploaded files:
        {
            "traces": <number of trace files uploaded>,
            "logs": <number of log files uploaded>
        }
    """
    if mlflow_writer is None:
        log_rank_0("[MLflow] MLflow writer not available, skipping artifact upload")
        return {"traces": 0, "logs": 0}

    log_rank_0("[MLflow] Starting artifact upload to MLflow...")
    log_rank_0(f"[MLflow] tensorboard_dir: {tensorboard_dir}")
    log_rank_0(f"[MLflow] exp_root_path: {exp_root_path}")
    log_rank_0(f"[MLflow] upload_traces: {upload_traces}, upload_logs: {upload_logs}")

    result = {"traces": 0, "logs": 0}

    if upload_traces and tensorboard_dir:
        result["traces"] = upload_trace_files_to_mlflow(
            mlflow_writer, tensorboard_dir, artifact_path="traces"
        )

    if upload_logs and exp_root_path:
        result["logs"] = upload_log_files_to_mlflow(mlflow_writer, exp_root_path, artifact_path="logs")

    log_rank_0(
        f"[MLflow] Artifact upload complete: " f"{result['traces']} trace files, {result['logs']} log files"
    )

    return result

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

Note:
    Multi-node training requires shared storage (e.g., NFS) for artifact uploads.
    Only the last rank (world_size - 1) performs the upload, so it must have
    access to trace and log files from all nodes. If using node-local storage,
    only files from the uploading node will be uploaded.
"""

import glob
import os
import traceback
from typing import List, Optional

from primus.modules.module_utils import log_rank_last

# Note: This module is called on the last rank (where MLflow is initialized).
# Using log_rank_last ensures messages are visible. For warnings, we prefix
# with [WARNING] since warning_rank_last doesn't exist.
try:
    from mlflow.exceptions import MlflowException
except ModuleNotFoundError:

    class MlflowException(Exception):
        """Fallback exception when mlflow isn't available."""


def _log_warning(msg: str) -> None:
    """Log a warning message on the last rank."""
    log_rank_last(f"[WARNING] {msg}")


def _get_all_trace_files(tensorboard_dir: Optional[str]) -> List[str]:
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


def _get_all_log_files(exp_root_path: Optional[str]) -> List[str]:
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
    uploads them to MLflow. In distributed settings, only the last rank
    (world_size - 1) where MLflow writer is initialized should call this.

    Args:
        mlflow_writer: The MLflow module instance (from get_mlflow_writer())
        tensorboard_dir: Path to the tensorboard directory containing trace files
        artifact_path: MLflow artifact subdirectory for trace files

    Returns:
        Number of trace files uploaded
    """
    if mlflow_writer is None:
        return 0

    log_rank_last(f"[MLflow] Searching for trace files in: {tensorboard_dir}")
    trace_files = _get_all_trace_files(tensorboard_dir)
    if len(trace_files) > 5:
        log_rank_last(f"[MLflow] Found {len(trace_files)} trace files: {trace_files[:5]}...")
    else:
        log_rank_last(f"[MLflow] Found {len(trace_files)} trace files: {trace_files}")

    if not trace_files:
        log_rank_last("[MLflow] No trace files found to upload")
        return 0

    total_files = len(trace_files)

    # Warn about potentially long upload times for large uploads
    if total_files > 10:
        # Safely calculate total size (files may be deleted between discovery and size check)
        total_size_bytes = 0
        for f in trace_files:
            try:
                total_size_bytes += os.path.getsize(f)
            except OSError:
                pass  # File may have been deleted
        total_size_mb = total_size_bytes / (1024 * 1024)
        _log_warning(
            f"[MLflow] Uploading {total_files} trace files ({total_size_mb:.1f} MB total). "
            "This may take a while..."
        )

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
            # Progress logging with counter
            log_rank_last(
                f"[MLflow] Uploaded trace file ({uploaded_count}/{total_files}): "
                f"{os.path.basename(trace_file)}"
            )
        except (OSError, RuntimeError, ValueError, MlflowException) as e:
            _log_warning(f"[MLflow] Failed to upload trace file {trace_file}: {type(e).__name__}: {e}")
            _log_warning(traceback.format_exc().strip())

    log_rank_last(f"[MLflow] Uploaded {uploaded_count}/{total_files} trace files to '{artifact_path}'")
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
        log_rank_last("[MLflow] No log files found to upload")
        return 0

    total_files = len(log_files)

    # Warn about potentially long upload times for large uploads
    if total_files > 20:
        # Safely calculate total size (files may be deleted between discovery and size check)
        total_size_bytes = 0
        for f in log_files:
            try:
                total_size_bytes += os.path.getsize(f)
            except OSError:
                pass  # File may have been deleted
        total_size_mb = total_size_bytes / (1024 * 1024)
        _log_warning(
            f"[MLflow] Uploading {total_files} log files ({total_size_mb:.1f} MB total). "
            "This may take a while..."
        )

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
        except (OSError, RuntimeError, ValueError, MlflowException) as e:
            _log_warning(f"[MLflow] Failed to upload log file {log_file}: {type(e).__name__}: {e}")
            _log_warning(traceback.format_exc().strip())

    log_rank_last(f"[MLflow] Uploaded {uploaded_count}/{total_files} log files to '{artifact_path}'")
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

    Note:
        Multi-node training requires shared storage (e.g., NFS) for complete
        artifact uploads. Only the last rank performs the upload, so it must
        have filesystem access to trace/log files from all nodes.

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
        log_rank_last("[MLflow] MLflow writer not available, skipping artifact upload")
        return {"traces": 0, "logs": 0}

    # Warn about multi-node shared storage requirement
    try:
        nnodes = int(os.environ.get("NNODES", os.environ.get("SLURM_NNODES", "1")))
    except ValueError:
        nnodes = 1
        _log_warning("[MLflow] NNODES/SLURM_NNODES could not be parsed as integer; assuming 1 node.")
    if nnodes > 1:
        _log_warning(
            f"[MLflow] Multi-node training detected ({nnodes} nodes). "
            "Ensure shared storage (e.g., NFS) is used for complete artifact uploads. "
            "Only files accessible from this node will be uploaded."
        )

    log_rank_last("[MLflow] Starting artifact upload to MLflow...")
    log_rank_last(f"[MLflow] tensorboard_dir: {tensorboard_dir}")
    log_rank_last(f"[MLflow] exp_root_path: {exp_root_path}")
    log_rank_last(f"[MLflow] upload_traces: {upload_traces}, upload_logs: {upload_logs}")

    result = {"traces": 0, "logs": 0}

    if upload_traces and tensorboard_dir:
        result["traces"] = upload_trace_files_to_mlflow(
            mlflow_writer, tensorboard_dir, artifact_path="traces"
        )

    if upload_logs and exp_root_path:
        result["logs"] = upload_log_files_to_mlflow(mlflow_writer, exp_root_path, artifact_path="logs")

    log_rank_last(
        f"[MLflow] Artifact upload complete: {result['traces']} trace files, {result['logs']} log files"
    )

    return result

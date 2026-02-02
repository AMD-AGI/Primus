###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
MLflow artifact upload utilities.

This module provides functions for uploading artifacts (traces, logs) to MLflow.
Separated from global_vars.py to reduce merge conflicts.
"""

from .global_vars import get_args, get_mlflow_writer
from .mlflow_artifacts import upload_artifacts_to_mlflow

_GLOBAL_EXP_ROOT_PATH = None


def set_exp_root_path(exp_root_path):
    """Set the experiment root path for artifact logging."""
    global _GLOBAL_EXP_ROOT_PATH
    _GLOBAL_EXP_ROOT_PATH = exp_root_path


def get_exp_root_path():
    """Return experiment root path. Can be None."""
    return _GLOBAL_EXP_ROOT_PATH


def reset_exp_root_path():
    """Reset the experiment root path to None."""
    global _GLOBAL_EXP_ROOT_PATH
    _GLOBAL_EXP_ROOT_PATH = None


def upload_mlflow_artifacts(
    upload_traces: bool = True,
    upload_logs: bool = True,
):
    """
    Upload trace files and log files to MLflow as artifacts.

    This should be called before ending the MLflow run to ensure all
    artifacts are uploaded. Only the rank that initialized MLflow
    (typically rank world_size - 1) should call this.

    Args:
        upload_traces: Whether to upload profiler trace files
        upload_logs: Whether to upload training log files

    Returns:
        Dictionary with counts of uploaded files, or None if MLflow is not enabled
    """
    mlflow_writer = get_mlflow_writer()
    if mlflow_writer is None:
        return None

    args = get_args()
    exp_root_path = get_exp_root_path()
    tensorboard_dir = getattr(args, "tensorboard_dir", None)

    return upload_artifacts_to_mlflow(
        mlflow_writer=mlflow_writer,
        tensorboard_dir=tensorboard_dir,
        exp_root_path=exp_root_path,
        upload_traces=upload_traces,
        upload_logs=upload_logs,
    )

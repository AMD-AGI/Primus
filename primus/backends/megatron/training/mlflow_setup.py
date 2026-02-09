###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
MLflow artifact upload utilities.

This module provides functions for uploading artifacts (traces, logs, TraceLens
reports) to MLflow. Separated from global_vars.py to reduce merge conflicts.
"""

from typing import List, Optional

from .global_vars import get_mlflow_writer
from .mlflow_artifacts import (
    generate_tracelens_reports_locally,
    upload_artifacts_to_mlflow,
)


def upload_mlflow_artifacts(
    tensorboard_dir: Optional[str] = None,
    exp_root_path: Optional[str] = None,
    upload_traces: bool = True,
    upload_logs: bool = True,
    generate_tracelens_report: bool = False,
    upload_tracelens_report: bool = False,
    tracelens_ranks: Optional[List[int]] = None,
    tracelens_output_format: str = "all",
    tracelens_cleanup_after_upload: bool = False,
) -> Optional[dict]:
    """
    Upload trace files, log files, and TraceLens reports to MLflow as artifacts.

    This function should be called at the end of training to upload all
    artifacts to MLflow. Only the rank that initialized MLflow (last rank)
    should call this to avoid duplicate uploads.

    MLflow Artifact Structure:
        artifacts/
        ├── traces/              # PyTorch profiler trace files
        ├── logs/                # Training log files
        └── trace_analysis/      # TraceLens analysis reports (if uploaded)

    TraceLens Report Logic:
        - upload_tracelens_report=True: Generate AND upload (auto-enables generation)
        - generate_tracelens_report=True only: Generate locally without upload
        - Both False: No report generation

    Args:
        tensorboard_dir: Path to tensorboard directory with trace files
        exp_root_path: Root experiment path for log files
        upload_traces: Whether to upload trace files (default: True)
        upload_logs: Whether to upload log files (default: True)
        generate_tracelens_report: Whether to generate TraceLens reports locally
        upload_tracelens_report: Whether to upload TraceLens reports to MLflow (implies generation)
        tracelens_ranks: List of ranks to analyze with TraceLens
                        (None = all, [0, 8] = ranks 0 and 8 only)
                        Specify fewer ranks to limit number of reports
        tracelens_output_format: Report format - "all" (default, xlsx+csv), "xlsx", or "csv"
        tracelens_cleanup_after_upload: Remove local reports after upload (default: False)

    Returns:
        Dictionary with counts of uploaded files, or None if MLflow is not enabled
    """
    mlflow_writer = get_mlflow_writer()
    if mlflow_writer is None:
        # Local-only TraceLens generation: run even when MLflow is disabled
        if generate_tracelens_report and tensorboard_dir and exp_root_path:
            generate_tracelens_reports_locally(
                tensorboard_dir=tensorboard_dir,
                exp_root_path=exp_root_path,
                ranks=tracelens_ranks,
                output_format=tracelens_output_format,
            )
        return None

    return upload_artifacts_to_mlflow(
        mlflow_writer=mlflow_writer,
        tensorboard_dir=tensorboard_dir,
        exp_root_path=exp_root_path,
        upload_traces=upload_traces,
        upload_logs=upload_logs,
        generate_tracelens_report=generate_tracelens_report,
        upload_tracelens_report=upload_tracelens_report,
        tracelens_ranks=tracelens_ranks,
        tracelens_output_format=tracelens_output_format,
        tracelens_cleanup_after_upload=tracelens_cleanup_after_upload,
    )

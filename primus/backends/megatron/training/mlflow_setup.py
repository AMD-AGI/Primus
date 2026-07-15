###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
MLflow artifact upload utilities.

This module provides functions for uploading artifacts (traces, logs, TraceLens
reports) to MLflow. Separated from global_vars.py to reduce merge conflicts.
"""

from typing import List, Optional

import torch.distributed as dist

from .global_vars import get_mlflow_writer, get_primus_args
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
    tracelens_output_format: str = "xlsx",
    tracelens_cleanup_after_upload: bool = False,
    tracelens_auto_install: bool = True,
) -> Optional[dict]:
    """
    Upload trace files, log files, and TraceLens reports to MLflow as artifacts.

    This function should be called at the end of training to upload all
    artifacts to MLflow. It is safe to call on all ranks: non-writer ranks
    will no-op when MLflow is disabled, while the writer rank performs uploads
    (and local-only TraceLens generation may still occur when configured).

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
        tracelens_output_format: Report format - "xlsx" (default), "csv", or "all"
        tracelens_cleanup_after_upload: Remove local reports after upload (default: False)
        tracelens_auto_install: Whether to attempt auto-installing TraceLens if missing

    Returns:
        Dictionary with counts of uploaded files, or None if MLflow is not enabled
    """
    mlflow_writer = get_mlflow_writer()
    if mlflow_writer is None:
        # Local-only TraceLens generation: run on a single rank only to avoid duplicate
        # work and races writing exp_root_path/tracelens_reports (rank 0 when multi-rank).
        # If MLflow is enabled in a distributed run, the writer rank will handle generation,
        # so skip local generation on non-writer ranks.
        try:
            args = get_primus_args()
            is_rank_zero = args.rank == 0
            mlflow_expected = getattr(args, "mlflow_run_name", None) is not None
            is_distributed = args.world_size > 1
        except Exception:
            is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
            mlflow_expected = False
            is_distributed = dist.is_initialized()

        should_generate_locally = is_rank_zero and (not mlflow_expected or not is_distributed)
        if should_generate_locally and generate_tracelens_report and tensorboard_dir and exp_root_path:
            generate_tracelens_reports_locally(
                tensorboard_dir=tensorboard_dir,
                exp_root_path=exp_root_path,
                ranks=tracelens_ranks,
                output_format=tracelens_output_format,
                auto_install=tracelens_auto_install,
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
        tracelens_auto_install=tracelens_auto_install,
    )

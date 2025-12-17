###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import List, Optional

from primus.modules.module_utils import debug_rank_0

from .mlflow_artifacts import upload_artifacts_to_mlflow

_GLOBAL_ARGS = None
_GLOBAL_MLFLOW_WRITER = None


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def get_mlflow_writer():
    """Return mlflow writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_MLFLOW_WRITER


def set_primus_global_variables(args):
    """Set args, mlflow."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
    set_args(args)

    _set_mlflow_writer(args)


def _set_mlflow_writer(args):
    global _GLOBAL_MLFLOW_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_MLFLOW_WRITER, "mlflow writer")
    if getattr(args, "mlflow_run_name", None) is not None and args.rank == (args.world_size - 1):
        try:
            import mlflow
        except ModuleNotFoundError:
            debug_rank_0(
                "WARNING: MLflow logging requested but is not "
                "available, ensure mlflow is installed. "
                "No MLflow logs will be written."
            )
            return

        if args.mlflow_experiment_name:
            mlflow.set_experiment(experiment_name=args.mlflow_experiment_name)

        mlflow.start_run(run_name=args.mlflow_run_name)
        mlflow.log_params(vars(args))
        _GLOBAL_MLFLOW_WRITER = mlflow


def unset_global_variables():
    """Unset global vars."""

    global _GLOBAL_ARGS
    global _GLOBAL_MLFLOW_WRITER

    _GLOBAL_ARGS = None
    _GLOBAL_MLFLOW_WRITER = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None


def upload_mlflow_artifacts(
    tensorboard_dir: Optional[str] = None,
    exp_root_path: Optional[str] = None,
    upload_traces: bool = True,
    upload_logs: bool = True,
    upload_tracelens_report: bool = False,
    tracelens_ranks: Optional[List[int]] = None,
    tracelens_max_reports: Optional[int] = None,
    tracelens_output_format: str = "all",
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
        └── trace_analysis/      # TraceLens analysis reports

    Args:
        tensorboard_dir: Path to tensorboard directory with trace files
        exp_root_path: Root experiment path for log files
        upload_traces: Whether to upload trace files (default: True)
        upload_logs: Whether to upload log files (default: True)
        upload_tracelens_report: Whether to generate and upload TraceLens reports
        tracelens_ranks: List of ranks to analyze with TraceLens
                        (None = all, [0] = rank 0 only)
        tracelens_max_reports: Maximum number of TraceLens reports to generate
        tracelens_output_format: Report format - "all" (default, xlsx+csv), "xlsx", "csv", or "html"
                                 (not yet supported, falls back to xlsx+csv)

    Returns:
        Dictionary with counts of uploaded files, or None if MLflow is not enabled
    """
    mlflow_writer = get_mlflow_writer()
    if mlflow_writer is None:
        return None

    return upload_artifacts_to_mlflow(
        mlflow_writer=mlflow_writer,
        tensorboard_dir=tensorboard_dir,
        exp_root_path=exp_root_path,
        upload_traces=upload_traces,
        upload_logs=upload_logs,
        upload_tracelens_report=upload_tracelens_report,
        tracelens_ranks=tracelens_ranks,
        tracelens_max_reports=tracelens_max_reports,
        tracelens_output_format=tracelens_output_format,
    )

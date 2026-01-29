###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import time

from primus.modules.module_utils import debug_rank_0

_GLOBAL_ARGS = None
_GLOBAL_MLFLOW_WRITER = None
_TRAIN_START_TIME = None


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def set_train_start_time(start_time=None):
    """Set training start time. If not provided, use current time."""
    global _TRAIN_START_TIME
    if start_time is None:
        start_time = time.time()
    _TRAIN_START_TIME = start_time


def get_train_start_time():
    """Return training start time."""
    _ensure_var_is_initialized(_TRAIN_START_TIME, "train start time")
    return _TRAIN_START_TIME


def get_mlflow_writer():
    """Return mlflow writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_MLFLOW_WRITER


def get_primus_args():
    """Return primus arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


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


def end_mlflow_run(status="FINISHED", termination_reason=None):
    """End MLflow run with specified status.

    Args:
        status: MLflow run status - "FINISHED", "FAILED", or "KILLED".
        termination_reason: Optional free-form string tag recorded as
            "termination_reason" on the run.
    """
    global _GLOBAL_MLFLOW_WRITER

    if _GLOBAL_MLFLOW_WRITER is None:
        return

    try:
        # Optionally attach a coarse termination reason tag for debugging.
        if termination_reason is not None:
            try:
                _GLOBAL_MLFLOW_WRITER.set_tag("termination_reason", termination_reason)
            except Exception:
                # Ignore tagging failures; status update is more important.
                pass

        _GLOBAL_MLFLOW_WRITER.end_run(status=status)
    except Exception:
        # Swallow MLflow/network errors to avoid masking the original failure.
        pass
    finally:
        _GLOBAL_MLFLOW_WRITER = None


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

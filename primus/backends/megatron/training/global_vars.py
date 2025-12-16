###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.modules.module_utils import debug_rank_0

from .mlflow_artifacts import upload_artifacts_to_mlflow

_GLOBAL_ARGS = None
_GLOBAL_MLFLOW_WRITER = None
_GLOBAL_EXP_ROOT_PATH = None


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def set_exp_root_path(exp_root_path):
    """Set the experiment root path for artifact logging."""
    global _GLOBAL_EXP_ROOT_PATH
    _GLOBAL_EXP_ROOT_PATH = exp_root_path


def get_exp_root_path():
    """Return experiment root path. Can be None."""
    return _GLOBAL_EXP_ROOT_PATH


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
    tensorboard_dir = getattr(args, "tensorboard_dir", None) if args else None

    return upload_artifacts_to_mlflow(
        mlflow_writer=mlflow_writer,
        tensorboard_dir=tensorboard_dir,
        exp_root_path=exp_root_path,
        upload_traces=upload_traces,
        upload_logs=upload_logs,
    )


def unset_global_variables():
    """Unset global vars."""

    global _GLOBAL_ARGS
    global _GLOBAL_MLFLOW_WRITER
    global _GLOBAL_EXP_ROOT_PATH

    _GLOBAL_ARGS = None
    _GLOBAL_MLFLOW_WRITER = None
    _GLOBAL_EXP_ROOT_PATH = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


def destroy_global_vars():
    global _GLOBAL_ARGS
    global _GLOBAL_EXP_ROOT_PATH
    _GLOBAL_ARGS = None
    _GLOBAL_EXP_ROOT_PATH = None

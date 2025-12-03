###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import re
from primus.modules.module_utils import debug_rank_0

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

def get_env_variables():
    """
    Filter environment variables for secrets and return as dict.
    """
    SECRET_PATTERN = re.compile(r"(TOKEN|SECRET|KEY)", re.IGNORECASE)
    
    env_vars = {}
    for k, v in os.environ.items():
        # Skip anything that looks secret-ish
        if SECRET_PATTERN.search(k) is None:
            env_vars[k] = v
    return env_vars

def format_env_variables() -> str:
    """
    Format env vars as 'KEY=VALUE' lines.
    """
    return "\n".join(
        f"{k}={v}" 
        for k, v in sorted(get_env_variables().items())
    )

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
        # 1) Original args
        mlflow.log_params(vars(args))
        # 2) Env as params (with prefix to avoid collisions)
        try:
            env_params = {f"env__{k}": v for k, v in get_env_variables().items()}
            mlflow.log_params(env_params)
        except Exception as e:
            debug_rank_0(f"WARNING: Failed to log environment variables to MLflow: {e}")

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

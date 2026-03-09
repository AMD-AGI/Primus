###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import json
import time

from primus.backends.megatron.training.git_metadata import (
    collect_git_metadata,
    get_env_variables,
)
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

        # 1) Original args
        mlflow.log_params(vars(args))

        # 2) Env as params (with prefix to avoid collisions)
        try:
            env_params = {f"env__{k}": v for k, v in get_env_variables().items()}
            mlflow.log_params(env_params)
        except Exception as e:
            debug_rank_0(f"WARNING: Failed to log environment variables to MLflow: {e}")

        # 3) Git metadata
        try:
            # This will:
            #  - log Primus as git/primus_*
            #  - scan the parent directory of Primus (workspace_root, defaults to primus_root.parent) for other git repos
            git_meta = collect_git_metadata()

            if git_meta:
                # 3a) Auto-select top-level repo commits/dirty (excludes submodules)
                summary_tags = {
                    k: v
                    for k, v in git_meta.items()
                    if "/submodule/" not in k and k.endswith(("_commit", "_dirty"))
                }

                # MLflow "source" tags
                primus_commit = git_meta.get("git/primus_commit", None)
                primus_remote = git_meta.get("git/primus_remote", None)

                if primus_commit:
                    mlflow.set_tag("mlflow.source.git.commit", primus_commit)
                if primus_remote:
                    mlflow.set_tag("mlflow.source.git.repoURL", primus_remote)

                if summary_tags:
                    mlflow.set_tags(summary_tags)

                # 3b) Full metadata as a single artifact for exact reproducibility
                mlflow.log_text(
                    json.dumps(git_meta, indent=2, sort_keys=True),
                    "system/git_metadata.json",
                )

        except Exception as e:
            debug_rank_0(f"WARNING: Failed to log git metadata to MLflow: {e}")

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

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Prepare hook for the Primus MaxDiffusion (JAX) backend.

Invoked by ``examples/scripts/prepare_experiment.py`` (resolved as
``examples/<framework>/prepare.py`` where framework == ``maxdiffusion``).

Unlike MaxText (a git submodule under ``third_party/maxtext``), MaxDiffusion is
``pip install -e .`` from a clone at ``/workspace/maxdiffusion`` (see
docker/jax_maxdiffusion.*). We resolve that path (MAXDIFFUSION_PATH override,
else the default clone dir) and, when present, forward it as ``--backend_path``
so the adapter can put it on ``sys.path``. Datasets are synthetic for the
benchmark configs, so no dataset prep is needed.
"""

import argparse
from pathlib import Path
from typing import Optional

from examples.scripts.utils import (
    get_env_case_insensitive,
    log_error_and_exit,
    log_info,
    write_patch_args,
)
from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import load_primus_config

_DEFAULT_MAXDIFFUSION_PATH = "/workspace/maxdiffusion"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Primus MaxDiffusion environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (used during training phase)",
    )
    parser.add_argument(
        "--backend_path",
        type=str,
        default=None,
        help="Optional override for the MaxDiffusion checkout path.",
    )
    return parser.parse_known_args()


def resolve_maxdiffusion_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        path = Path(cli_path)
        log_info(f"Using MaxDiffusion path from CLI: {path}")
        return path
    env_value = get_env_case_insensitive("MAXDIFFUSION_PATH")
    if env_value:
        path = Path(env_value)
        log_info(f"MAXDIFFUSION_PATH found in environment: {path}")
        return path
    path = Path(_DEFAULT_MAXDIFFUSION_PATH)
    log_info(f"MAXDIFFUSION_PATH not set, falling back to: {path}")
    return path


def prepare_dataset_if_needed(
    primus_config: PrimusConfig, primus_path: Path, data_path: Path, patch_args: Path, env=None
):
    # No-op: benchmark configs use synthetic data; real datasets are prepared
    # outside Primus (see examples/diffusion/README.md for HF dataset notes).
    return


def main():
    args, unknown = parse_args()

    primus_path = Path(args.primus_path).resolve()
    data_path = Path(args.data_path).resolve()
    exp_path = Path(args.config).resolve()
    patch_args_file = Path(args.patch_args).resolve()

    log_info(f"PRIMUS_PATH: {primus_path}")
    log_info(f"DATA_PATH: {data_path}")
    log_info(f"EXP: {exp_path}")
    log_info(f"BACKEND_PATH: {args.backend_path}")
    log_info(f"PATCH-ARGS: {patch_args_file}")

    if not exp_path.is_file():
        log_error_and_exit(f"EXP file not found: {exp_path}")

    primus_config, _ = load_primus_config(args, unknown)

    maxdiffusion_path = resolve_maxdiffusion_path(args.backend_path)

    try:
        pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    except Exception:
        log_error_and_exit("Missing required module config: pre_trainer")

    if not hasattr(pre_trainer_cfg, "dataset_type") or pre_trainer_cfg.dataset_type is None:
        log_error_and_exit("Missing required field: pre_trainer.dataset_type")

    dataset_type = pre_trainer_cfg.dataset_type
    if dataset_type == "synthetic":
        log_info("'dataset_type: synthetic', Skipping dataset preparation.")
    else:
        prepare_dataset_if_needed(
            primus_config=primus_config,
            primus_path=primus_path,
            data_path=data_path,
            patch_args=patch_args_file,
            env=None,
        )

    # Forward the checkout as --backend_path only if it exists; MaxDiffusion is
    # usually importable as an installed package, and the adapter tolerates a
    # missing path.
    if maxdiffusion_path.exists():
        write_patch_args(patch_args_file, "train_args", {"backend_path": str(maxdiffusion_path.resolve())})
    else:
        log_info(f"MaxDiffusion checkout not found at {maxdiffusion_path}; relying on installed package.")


if __name__ == "__main__":
    log_info("========== Prepare MaxDiffusion Env ==========")
    main()

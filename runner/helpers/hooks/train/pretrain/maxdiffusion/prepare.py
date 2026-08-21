###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion pre-train preparation hook for primus-cli direct.

This script is invoked by:

    runner/helpers/hooks/train/pretrain/prepare_experiment.sh

via:

    python maxdiffusion/prepare.py \
        --config <exp.yaml> \
        --data_path <data_root> \
        --primus_path <primus_root> \
        --patch_args <patch_args.txt> \
        [--backend_path <override>] \
        [<extra CLI args>...]

It is the primus-cli counterpart of the MaxDiffusion branch in
``examples/run_pretrain.sh``: both launch paths must agree on the launcher mode,
the backend checkout location, and the JAX multi-node coordinator.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import load_primus_config
from runner.helpers.hooks.train.pretrain.utils import (
    default_backend_path,
    get_env_case_insensitive,
    log_error_and_exit,
    log_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Primus MaxDiffusion environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (kept for compatibility; not used by this hook)",
    )
    parser.add_argument(
        "--backend_path",
        type=str,
        default=None,
        help="Optional override for MaxDiffusion path; takes precedence over env and default.",
    )
    return parser.parse_known_args()


def resolve_backend_path(cli_path: Optional[str], primus_path: Path) -> Path:
    if cli_path:
        path = Path(cli_path).resolve()
        log_info(f"Using MaxDiffusion path from CLI: {path}")
        return path

    env_value = get_env_case_insensitive("MAXDIFFUSION_PATH")
    if env_value:
        path = Path(env_value).resolve()
        log_info(f"MAXDIFFUSION_PATH found in environment: {path}")
        return path

    path = default_backend_path(primus_path, "maxdiffusion")
    log_info(f"MAXDIFFUSION_PATH not found, falling back to: {path}")
    return path


def prepare_dataset_if_needed(
    primus_config: PrimusConfig, primus_path: Path, data_path: Path, patch_args: Path, env=None
):
    """
    Placeholder for future MaxDiffusion dataset preparation logic.

    The shipped WAN 2.1 / FLUX.1-dev benchmark configs all use
    ``dataset_type: synthetic``, and real datasets are prepared outside of
    Primus, so there is nothing to do here yet. Implement conversion or
    preprocessing in this function when a non-synthetic dataset that needs
    on-the-fly preparation is introduced.
    """

    return


def emit_env_if_unset(name: str, value: str) -> None:
    """Emit an ``env.<name>=<value>`` line unless the variable is already set.

    ``execute_hooks.sh`` exports every emitted line unconditionally, so guarding
    here is what preserves the outer-env-wins precedence that the equivalent
    ``${VAR:-default}`` assignments in ``examples/run_pretrain.sh`` provide.
    """
    if os.environ.get(name):
        log_info(f"{name} already set to '{os.environ[name]}'; leaving it untouched")
        return
    log_info(f"Exposing {name} via env.{name}={value}")
    print(f"env.{name}={value}")


def emit_jax_coordinator() -> None:
    """Wire the JAX coordinator so multi-node runs rendezvous.

    MaxDiffusion's GPU init (``max_utils.initialize_jax_for_gpu``) only calls
    ``jax.distributed.initialize()`` when ``JAX_COORDINATOR_IP`` is set, using
    num_processes=NNODES / process_id=NODE_RANK (one process per node). Without
    it each node silently initializes as a standalone single-node job.
    """
    try:
        nnodes = int(os.environ.get("NNODES", "1"))
    except ValueError:
        nnodes = 1

    if nnodes <= 1:
        log_info(f"NNODES={nnodes}, single-node run: no JAX coordinator needed.")
        return

    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")
    if not master_addr or not master_port:
        log_info(
            f"NNODES={nnodes} but MASTER_ADDR/MASTER_PORT are not both set; "
            "skipping JAX coordinator wiring. Multi-node rendezvous will not happen."
        )
        return

    emit_env_if_unset("JAX_COORDINATOR_IP", master_addr)
    emit_env_if_unset("JAX_COORDINATOR_PORT", master_port)


# ---------- Main ----------
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

    maxdiffusion_path = resolve_backend_path(args.backend_path, primus_path)

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

    # Forward the checkout as --backend_path only when it exists. MaxDiffusion is
    # often an editable/wheel install rather than a source tree, and the adapter
    # falls back to the installed package when no path is given.
    if maxdiffusion_path.exists():
        log_info(f"Exposing resolved backend path via extra.backend_path={maxdiffusion_path}")
        print(f"extra.backend_path={maxdiffusion_path}")
    else:
        log_info(
            f"MaxDiffusion checkout not found at {maxdiffusion_path}; relying on the "
            "installed package. Run `git submodule update --init third_party/maxdiffusion` "
            "if you meant to use the vendored source."
        )

    # MaxDiffusion's JAX/XLA/NVTE/RCCL tuning lives in the per-config top-level
    # `env:` block (see primus/backends/maxdiffusion/env_spec.py for why the
    # adapter deliberately contributes none). Only launcher-level decisions --
    # things that must be known before Python starts -- belong here.

    # RUN_MODE picks the launcher: JAX manages all devices from one process, so
    # torchrun is never correct for this backend. execute_hooks.sh exports it and
    # primus-cli-direct.sh reads $RUN_MODE when building the command.
    log_info("Exposing run mode via env.RUN_MODE=single")
    print("env.RUN_MODE=single")

    # TransformerEngine must be told it is running under JAX before it is imported.
    emit_env_if_unset("NVTE_FRAMEWORK", "jax")

    emit_jax_coordinator()


if __name__ == "__main__":
    log_info("========== Prepare MaxDiffusion Env (pre-train hook) ==========")
    main()

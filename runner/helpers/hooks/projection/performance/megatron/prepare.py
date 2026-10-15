###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import subprocess
from pathlib import Path

from primus.core.launcher.parser import load_primus_config
from runner.helpers.hooks.train.pretrain.utils import (
    get_env_case_insensitive,
    log_error_and_exit,
    log_info,
    write_patch_args,
)


# ---------- Helpers ----------
def check_dir_nonempty(path: Path, name: str):
    if not path.is_dir() or not any(path.iterdir()):
        log_error_and_exit(
            f"{name} ({path}) does not exist or is empty.\n"
            "Please ensure Primus is properly initialized.\n"
            "If not yet cloned, run:\n"
            "    git clone --recurse-submodules git@github.com:AMD-AGI/Primus.git\n"
            "Or if already cloned, initialize submodules with:\n"
            "    git submodule update --init --recursive"
        )


def build_megatron_helper(primus_path: Path, patch_args: Path, backend_path: str = None):
    """Build Megatron's helper C++ dataset library."""
    if backend_path:
        megatron_path = Path(backend_path).resolve()
        log_info(f"Using backend_path from argument: {megatron_path}")
    else:
        env_backend = get_env_case_insensitive("MEGATRON_PATH")
        if env_backend:
            megatron_path = Path(env_backend).resolve()
            log_info(f"Using backend_path from environment: {megatron_path}")
        else:
            megatron_path = primus_path / "third_party/Megatron-LM"
            log_info(f"No backend_path provided, falling back to: {megatron_path}")
    write_patch_args(Path(patch_args), "train_args", {"backend_path": str(megatron_path)})

    check_dir_nonempty(megatron_path, "megatron")

    # build C++ helper
    dataset_cpp_dir = megatron_path / "megatron/core/datasets"
    log_info(f"Building Megatron dataset helper in {dataset_cpp_dir}")

    # `-s` silences make's "Nothing to be done"/recipe echo on no-op rebuilds;
    # real compiler errors still surface and are handled below.
    ret = subprocess.run(["make", "-s"], cwd=dataset_cpp_dir)
    if ret.returncode != 0:
        log_error_and_exit("Building Megatron C++ helper failed.")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Prepare Primus environment")
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
        help="Optional path to backend (e.g., Megatron), will be added to PYTHONPATH",
    )
    args, unknown = parser.parse_known_args()

    log_info(f"BACKEND_PATH {args.backend_path}")
    load_primus_config(args, unknown)

    primus_path = Path(args.primus_path).resolve()
    log_info(f"PRIMUS_PATH is set to: {primus_path}")

    exp_path = Path(args.config).resolve()
    if not exp_path.is_file():
        log_error_and_exit(f"The specified EXP file does not exist: {exp_path}")
    log_info(f"EXP is set to: {exp_path}")

    patch_args_file = Path(args.patch_args).resolve()
    log_info(f"PATCH-ARGS is set to: {patch_args_file}")

    # Projection builds the model through setup_model_only(), which stops before
    # datasets are constructed, so no training data is prepared here.
    build_megatron_helper(primus_path=primus_path, backend_path=args.backend_path, patch_args=patch_args_file)


if __name__ == "__main__":
    log_info("========== Prepare Megatron projection ==========")
    main()

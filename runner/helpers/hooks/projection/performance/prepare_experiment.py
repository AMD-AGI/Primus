###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

from primus.core.launcher.parser import PrimusParser
from runner.helpers.hooks.train.pretrain.utils import log_error_and_exit, log_info


def log(msg, level="INFO"):
    if int(os.environ.get("NODE_RANK", "0")) == 0:
        print(f"[NODE-0({socket.gethostname()})] [{level}] {msg}", file=sys.stderr)
        if level == "ERROR":
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Primus Backend Preparation Entry")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config file")
    parser.add_argument(
        "--data_path", type=str, default="./data/", help="Root directory for datasets and tokenizer"
    )
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

    primus_path = Path.cwd()

    # Ensure config path is absolute
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    args.config = str(config_path.resolve())

    # Parse the config from CLI args
    config = PrimusParser().parse(args)

    # Use exp_root_path as a string if it's not callable
    if callable(config.exp_root_path):
        patch_args_dir = config.exp_root_path()
    else:
        patch_args_dir = Path(config.exp_root_path)

    # Convert patch_args_dir to an absolute path
    patch_args_dir = patch_args_dir.resolve()
    patch_args_dir.mkdir(parents=True, exist_ok=True)

    patch_args_path = patch_args_dir / "patch_args.txt"

    # Ensure patch_args file does not exist before creating it
    if patch_args_path.exists():
        patch_args_path.unlink()
    log(f"Preparing patch args file at: {patch_args_path}")

    # Parse the config from CLI args
    config = PrimusParser().parse(args)

    # Get framework name from pre_trainer module
    framework = config.get_module_config("pre_trainer").framework

    # Normalize alias: map "light-megatron" to actual folder name
    framework_map = {
        "megatron": "megatron",
        "light-megatron": "megatron",
        "torchtitan": "torchtitan",
        # Add more aliases here if needed
    }
    framework_dir = framework_map.get(framework, framework)

    # Construct the script path using the current file's directory
    current_dir = Path(__file__).parent
    script = current_dir / framework_dir / "prepare.py"

    if not script.exists():
        log_info(f"Backend prepare script not found: {script}")

    log_info(f"Running backend prepare: {script}")
    cmd = [
        "python",
        str(script),
        "--config",
        args.config,
        "--data_path",
        args.data_path,
        "--primus_path",
        str(primus_path),
        "--patch_args",
        str(patch_args_path),
    ]

    if args.backend_path:
        cmd += ["--backend_path", args.backend_path]

    cmd += unknown
    try:
        subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        log_error_and_exit(f"Backend script({script}) failed with exit code {e.returncode}")


if __name__ == "__main__":
    main()

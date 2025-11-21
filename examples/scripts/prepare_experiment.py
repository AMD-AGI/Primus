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

from examples.scripts.utils import log_error_and_exit, log_info
from primus.core.launcher.parser import PrimusParser


def log(msg, level="INFO"):
    if int(os.environ.get("NODE_RANK", "0")) == 0:
        print(f"[NODE-0({socket.gethostname()})] [{level}] {msg}", file=sys.stderr)
        if level == "ERROR":
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Primus Backend Preparation Entry")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config file")
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
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Optional backend (e.g., Megatron, TorchTitan and MaxText), check consistency between config and backend",
    )
    args, unknown = parser.parse_known_args()

    primus_path = Path.cwd()
    patch_args_path = Path(args.patch_args).resolve()
    patch_args_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse the config from CLI args
    config = PrimusParser().parse(args)

    # Get framework name from pre_trainer module
    framework = config.get_module_config("pre_trainer").framework
    if args.backend is not None and (args.backend.strip().lower() != framework):
        log_error_and_exit(
            f"The backend {args.backend} set in env is different with {framework} set in the config"
        )

    # Normalize alias: map "light-megatron" to actual folder name
    framework_map = {
        "megatron": "megatron",
        "light-megatron": "megatron",
        "torchtitan": "torchtitan",
        "maxtext": "maxtext",
        # Add more aliases here if needed
    }
    framework_dir = framework_map.get(framework, framework)

    # Construct the script path
    script = Path(primus_path) / "examples" / framework_dir / "prepare.py"

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

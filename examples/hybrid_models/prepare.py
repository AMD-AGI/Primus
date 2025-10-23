###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from requests.exceptions import HTTPError

from examples.scripts.utils import (
    get_env_case_insensitive,
    get_node_rank,
    log_error_and_exit,
    log_info,
    write_patch_args,
)
from primus.core.launcher.parser import PrimusParser

def parse_args():
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
        help="Optional override for Hybrid Models path; takes precedence over env and default.",
    )
    return parser.parse_args()


def pip_install_editable(path: Path, name: str):
    log_info(f"Installing {name} in editable mode via pip (path: {path})")
    ret = subprocess.run(["pip", "install", "-e", ".", "-q"], cwd=path)
    if ret.returncode != 0:
        log_error_and_exit(f"Failed to install {name} via pip.")


def resolve_backend_path(
    cli_path: Optional[str], env_var: str, default_subdir: str, primus_path: Path, name: str
) -> Path:
    if cli_path:
        path = Path(cli_path).resolve()
        log_info(f"Using {name} path from CLI: {path}")
    else:
        env_value = get_env_case_insensitive(env_var)
        if env_value:
            path = Path(env_value).resolve()
            log_info(f"{env_var.upper()} found in environment: {path}")
        else:
            path = primus_path / default_subdir
            log_info(f"{env_var.upper()} not found, falling back to: {path}")
    return path


def main():
    args = parse_args()

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

    primus_cfg = PrimusParser().parse(args)

    hybrid_models_path = resolve_backend_path(
        args.backend_path, "HYBRID_MODELS_PATH", "third_party/AMD-Hybrid-Models/Zebra-Llama", primus_path, "Hybrid_Models"
    )

    # Skip pip installation for hybrid models since it's not a Python package
    log_info(f"Hybrid Models backend path: {hybrid_models_path}")
    if not hybrid_models_path.exists():
        log_error_and_exit(f"Hybrid Models path does not exist: {hybrid_models_path}")

def detect_rocm_version() -> Optional[str]:
    """
    Detect ROCm version from /opt/rocm/.info/version (most reliable source).

    Example file content:
        7.0.0
    → returns '7.0'
    """
    info_file = "/opt/rocm/.info/version"
    if os.path.exists(info_file):
        try:
            with open(info_file, "r") as f:
                content = f.readline().strip()
                # Match like '7.0.0' or '6.3.1'
                match = re.match(r"^(\d+)\.(\d+)", content)
                if match:
                    major, minor = match.groups()
                    return f"{major}.{minor}"
        except Exception:
            pass

    return None


def install_torch_for_rocm(nightly=True):
    version = detect_rocm_version()
    if not version:
        log_error_and_exit("ROCm not detected.")

    tag = f"rocm{version}"
    base = "https://download.pytorch.org/whl/nightly" if nightly else "https://download.pytorch.org/whl"
    url = f"{base}/{tag}"

    log_info(f"Installing PyTorch for {tag} from {url}")
    subprocess.run(["pip", "install", "--pre", "torch", "--index-url", url, "--force-reinstall"], check=True)


if __name__ == "__main__":

    log_info("========== Prepare Hybrid Models dataset ==========")
    main()
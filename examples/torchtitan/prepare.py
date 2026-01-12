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
        help="Optional override for TorchTitan path; takes precedence over env and default.",
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


def run_titan_hf_download(
    torchtitan_path: Path, repo_id: str, local_dir: Path, hf_token: Optional[str] = None
):
    """Use Titan's own download_hf_assets.py to fetch tokenizer/model assets."""
    script_path = torchtitan_path / "scripts" / "download_hf_assets.py"
    if not script_path.is_file():
        log_error_and_exit(f"TorchTitan script not found: {script_path}")

    cmd = [
        "python",
        str(script_path),
        "--repo_id",
        repo_id,
        "--assets",
        "tokenizer",
        "--local_dir",
        str(local_dir),
    ]
    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token

    log_info(f"[rank0] Running Titan HF downloader:\n  {' '.join(cmd)}")
    ret = subprocess.run(cmd, env=env, cwd=torchtitan_path)
    if ret.returncode != 0:
        log_error_and_exit(f"TorchTitan HF download failed with code {ret.returncode}")


def resolve_hf_assets_path(data_path: Path, hf_assets_value: str) -> tuple[str, Path, bool]:
    """
    Resolve HuggingFace asset source — supports both repo IDs and local paths.

    Args:
        data_path (Path):
            Base data directory (e.g., /data/primus_data).
        hf_assets_value (str):
            Can be either:
              - A HuggingFace repo ID (e.g., "meta-llama/Llama-3.1-70B")
              - A local directory path (e.g., "/data/primus_data/torchtitan/Llama-3.1-70B")

    Returns:
        (repo_or_path, local_dir, need_download)
            repo_or_path: str — repo_id if remote; same path if local
            local_dir: Path — where assets are or will be located
            need_download: bool — True if download is required

    Behavior:
        1. If hf_assets_value is an existing directory path:
               → Treat it as an already downloaded local path.
        2. If it is not an existing directory (likely a repo_id):
               → Derive the local target dir as
                   data_path / "torchtitan" / <last_component_of_repo_id>
               → Mark need_download=True.
    """
    path_candidate = Path(hf_assets_value).expanduser()

    # Case 1: already-downloaded local directory
    if path_candidate.exists() and path_candidate.is_dir():
        log_info(f"Detected local HF assets path: {path_candidate}")
        return hf_assets_value, path_candidate.resolve(), False

    # Case 2: repo_id (e.g., meta-llama/Llama-3.1-70B) → need to download
    repo_id = hf_assets_value
    repo_name = Path(repo_id).name  # last segment, e.g., Llama-3.1-70B
    local_dir = data_path / "torchtitan" / repo_name
    log_info(f"Resolved HF repo_id={repo_id}, local_dir={local_dir}")
    return repo_id, local_dir, True


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

    torchtitan_path = resolve_backend_path(
        args.backend_path, "TORCHTITAN_PATH", "third_party/torchtitan", primus_path, "TorchTitan"
    )
    pip_install_editable(torchtitan_path, "TorchTitan")

    try:
        pre_trainer_cfg = primus_cfg.get_module_config("pre_trainer")
    except Exception:
        log_error_and_exit("Missing required module config: pre_trainer")

    if not hasattr(pre_trainer_cfg, "model") or pre_trainer_cfg.model is None:
        log_error_and_exit("Missing required field: pre_trainer.model")

    if not hasattr(pre_trainer_cfg.model, "hf_assets_path") or not pre_trainer_cfg.model.hf_assets_path:
        log_error_and_exit("Missing required field: pre_trainer.model.tokenizer_path")

    hf_assets_value = pre_trainer_cfg.model.hf_assets_path
    repo_id, local_dir, need_download = resolve_hf_assets_path(data_path, hf_assets_value)
    tokenizer_file = local_dir / "tokenizer.json"

    if need_download:
        # Remote repo_id case — download via Titan script
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            log_error_and_exit("HF_TOKEN not set. Please export HF_TOKEN before running prepare.")

        if get_node_rank() == 0:
            if not tokenizer_file.exists():
                log_info(f"Downloading HF assets from repo={repo_id} into {local_dir} ...")
                parent_dir = local_dir.parent
                parent_dir.mkdir(parents=True, exist_ok=True)
                run_titan_hf_download(torchtitan_path, repo_id, parent_dir, hf_token)
            else:
                log_info(f"Tokenizer assets already exist: {tokenizer_file}")
        else:
            # Other ranks wait until the file is available
            log_info(f"[rank{get_node_rank()}] waiting for tokenizer download ...")
            while not tokenizer_file.exists():
                time.sleep(5)
    else:
        # Local path case — skip download
        log_info(f"HF assets already available locally at {local_dir}")

    # Pass resolved path to training phase
    write_patch_args(patch_args_file, "train_args", {"model.hf_assets_path": str(local_dir)})
    write_patch_args(patch_args_file, "train_args", {"backend_path": str(torchtitan_path)})
    write_patch_args(patch_args_file, "torchrun_args", {"local-ranks-filter": "0"})


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
    # log_info("========== Prepare torch for Torchtitan ==========")
    # install_torch_for_rocm(nightly=True)

    log_info("========== Prepare Torchtitan dataset ==========")
    main()

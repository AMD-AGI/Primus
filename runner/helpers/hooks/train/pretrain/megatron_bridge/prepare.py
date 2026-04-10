###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import subprocess
from pathlib import Path
from typing import Any, Optional

from primus.backends.megatron_bridge.config_utils import (
    DATASET_CONFIG_FILE_KEYS,
    DATASET_PATH_KEYS,
    normalize_megatron_bridge_dataset_args,
)
from primus.core.launcher.parser import load_primus_config
from runner.helpers.hooks.train.pretrain.utils import (
    get_env_case_insensitive,
    log_error_and_exit,
    log_info,
)


# ---------- Helpers ----------
def check_dir_nonempty(path: Path, name: str):
    if not path.is_dir() or not any(path.iterdir()):
        log_error_and_exit(
            f"{name} ({path}) does not exist or is empty.\n"
            "Please ensure Primus is properly initialized.\n"
            "If not yet cloned, run:\n"
            "    git clone --recurse-submodules git@github.com:AMD-AIG-AIMA/Primus.git\n"
            "Or if already cloned, initialize submodules with:\n"
            "    git submodule update --init --recursive"
        )


def resolve_backend_path(cli_path: Optional[str], primus_path: Path) -> Path:
    """Resolve Megatron-Bridge backend path: CLI > env > default."""
    if cli_path:
        path = Path(cli_path).resolve()
        log_info(f"Using Megatron-Bridge path from CLI: {path}")
        return path

    env_value = get_env_case_insensitive("MEGATRON_BRIDGE_PATH")
    if env_value:
        path = Path(env_value).resolve()
        log_info(f"MEGATRON_BRIDGE_PATH found in environment: {path}")
        return path

    path = primus_path / "third_party" / "Megatron-Bridge"
    log_info(f"MEGATRON_BRIDGE_PATH not found, falling back to: {path}")
    return path


def build_megatron_helper(bridge_path: Path):
    """Build Megatron C++ dataset helper from Megatron-Bridge's bundled Megatron-LM."""
    megatron_lm_path = bridge_path / "3rdparty" / "Megatron-LM"
    dataset_cpp_dir = megatron_lm_path / "megatron" / "core" / "datasets"

    if not dataset_cpp_dir.is_dir():
        log_info(f"Megatron C++ dataset dir not found at {dataset_cpp_dir}, skipping build")
        return

    log_info(f"Building Megatron dataset C++ helper in {dataset_cpp_dir}")
    ret = subprocess.run(["make"], cwd=dataset_cpp_dir)
    if ret.returncode != 0:
        log_error_and_exit("Building Megatron C++ dataset helper failed.")
    log_info("Megatron C++ dataset helper built successfully")


def emit_extra(name: str, value: Any):
    if isinstance(value, (list, tuple)):
        print(f"extra.{name}={repr(list(value))}")
    else:
        print(f"extra.{name}={value}")


def resolve_user_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def is_weight_token(token: str) -> bool:
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def resolve_dataset_path_list(values: list[str] | None, base_dir: Path) -> list[str] | None:
    if not values:
        return None

    weighted = len(values) % 2 == 0 and all(is_weight_token(values[i]) for i in range(0, len(values), 2))
    resolved: list[str] = []
    for idx, value in enumerate(values):
        if weighted and idx % 2 == 0:
            resolved.append(str(value))
        else:
            resolved.append(str(resolve_user_path(str(value), base_dir)))
    return resolved


def extract_dataset_prefixes(values: list[str]) -> list[str]:
    weighted = len(values) % 2 == 0 and all(is_weight_token(values[i]) for i in range(0, len(values), 2))
    return values[1::2] if weighted else values


def validate_indexed_dataset_prefixes(name: str, values: list[str]) -> None:
    missing: list[str] = []
    for prefix in extract_dataset_prefixes(values):
        bin_path = Path(f"{prefix}.bin")
        idx_path = Path(f"{prefix}.idx")
        if not bin_path.is_file() or not idx_path.is_file():
            missing.append(f"{prefix} (.bin/.idx)")

    if missing:
        log_error_and_exit(
            f"Invalid {name}: missing indexed dataset files for: {missing}. "
            "Megatron-Bridge pretrain expects dataset prefixes backed by '<prefix>.bin' and '<prefix>.idx'."
        )


def collect_explicit_dataset_extras(pre_trainer_cfg: Any, base_dir: Path) -> dict[str, Any]:
    extras: dict[str, Any] = {}

    for key in DATASET_PATH_KEYS:
        values = getattr(pre_trainer_cfg, key, None)
        if not values:
            continue
        resolved = resolve_dataset_path_list(values, base_dir)
        assert resolved is not None
        validate_indexed_dataset_prefixes(key, resolved)
        extras[key] = resolved

    for key in DATASET_CONFIG_FILE_KEYS:
        value = getattr(pre_trainer_cfg, key, None)
        if not value:
            continue
        resolved = resolve_user_path(str(value), base_dir)
        if not resolved.is_file():
            log_error_and_exit(f"{key} does not exist: {resolved}")
        extras[key] = str(resolved)

    return extras


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Prepare Primus Megatron-Bridge pretrain environment")
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
        help="Optional path to backend (e.g., Megatron-Bridge), will be added to PYTHONPATH",
    )
    args, unknown = parser.parse_known_args()

    primus_config, _ = load_primus_config(args, unknown)

    primus_path = Path(args.primus_path).resolve()
    log_info(f"PRIMUS_PATH is set to: {primus_path}")

    data_path = Path(args.data_path).resolve()
    log_info(f"DATA_PATH is set to: {data_path}")

    exp_path = Path(args.config).resolve()
    if not exp_path.is_file():
        log_error_and_exit(f"The specified EXP file does not exist: {exp_path}")
    log_info(f"EXP is set to: {exp_path}")

    patch_args_file = Path(args.patch_args).resolve()
    log_info(f"PATCH-ARGS is set to: {patch_args_file}")

    bridge_path = resolve_backend_path(args.backend_path, primus_path)
    check_dir_nonempty(bridge_path, "Megatron-Bridge")

    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    normalize_megatron_bridge_dataset_args(pre_trainer_cfg)
    config_base_dir = exp_path.parent

    explicit_dataset_extras = collect_explicit_dataset_extras(pre_trainer_cfg, config_base_dir)
    mock = bool(getattr(pre_trainer_cfg, "mock", False))

    if explicit_dataset_extras:
        log_info(
            "Detected explicit indexed dataset configuration, validating and exporting normalized paths."
        )
        for key, value in explicit_dataset_extras.items():
            emit_extra(key, value)
        emit_extra("mock", False)
    elif mock:
        log_info("'mock: true', skipping real dataset preparation.")
    else:
        log_error_and_exit(
            "Megatron-Bridge pretrain requires explicit real dataset configuration when mock=false. "
            "Provide one of: data_paths, train_data_path/valid_data_path/test_data_path, "
            "data_args_path, or per_split_data_args_path. Dataset prefixes must point to "
            "'<prefix>.bin' and '<prefix>.idx'."
        )

    build_megatron_helper(bridge_path)

    log_info(f"Exposing resolved backend path via extra.backend_path={bridge_path}")
    print(f"extra.backend_path={bridge_path}")


if __name__ == "__main__":
    log_info("========== Prepare Megatron-Bridge pretrain env ==========")
    main()

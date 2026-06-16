###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import socket
import sys
from pathlib import Path

import yaml


# ---------- Logging ----------
def get_node_rank() -> int:
    return int(os.environ.get("NODE_RANK", "0"))


def get_hostname():
    return socket.gethostname()


def log_info(msg):
    if get_node_rank() == 0:
        print(f"[NODE-{get_node_rank()}({get_hostname()})] [INFO] {msg}", file=sys.stderr)


def log_error_and_exit(msg):
    if get_node_rank() == 0:
        print(f"[NODE-{get_node_rank()}({get_hostname()})] [ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def format_cli_args(args: dict) -> str:
    """Format a dictionary into CLI-style arguments: --key val --key2 val2 ..."""
    parts = []
    for k, v in args.items():
        parts.extend([f"--{k}", str(v)])
    return " ".join(parts)


def parse_cli_args_string(arg_string: str) -> dict:
    """Parse a CLI-style string into a dict: --key val --key2 val2 → {"key": "val", ...}"""
    parts = arg_string.strip().split()
    result = {}
    i = 0
    while i < len(parts):
        if parts[i].startswith("--") and i + 1 < len(parts):
            key = parts[i][2:]
            val = parts[i + 1]
            result[key] = val
            i += 2
        else:
            i += 1
    return result


def write_patch_args(path: Path, section: str, args_dict: dict):
    """Write or merge args_dict into the given section in YAML patch file"""
    if path.exists():
        with open(path, "r") as f:
            patch = yaml.safe_load(f) or {}
    else:
        patch = {}

    existing_section = patch.get(section, {})

    if isinstance(existing_section, str):
        existing_args = parse_cli_args_string(existing_section)
    elif isinstance(existing_section, dict):
        existing_args = existing_section
    else:
        existing_args = {}

    # Merge the new args into existing
    existing_args.update(args_dict)

    # Save the merged args
    patch[section] = format_cli_args(existing_args)

    with open(path, "w") as f:
        yaml.safe_dump(patch, f)

    log_info(f"   Wrote patch args to {path} under section '{section}' args {args_dict}.")


def get_env_case_insensitive(var_name: str) -> str | None:
    """Get environment variable by name, ignoring case."""
    for key, value in os.environ.items():
        if key.lower() == var_name.lower():
            return value
    return None


def default_backend_path(primus_path, dir_name: str) -> Path:
    """Default backend source dir for prepare hooks.

    Prefer the source-tree location ``<primus_path>/third_party/<dir_name>`` when
    it exists; otherwise fall back to the deps-sync dir used by installed wheels
    (``$PRIMUS_THIRDPARTY_DIR`` or ``~/.cache/Primus/third_party``), populated by
    ``primus-cli deps sync``.
    """
    src = Path(primus_path) / "third_party" / dir_name
    if src.exists():
        return src
    tp_root = os.getenv("PRIMUS_THIRDPARTY_DIR") or str(Path.home() / ".cache" / "Primus" / "third_party")
    return Path(tp_root) / dir_name

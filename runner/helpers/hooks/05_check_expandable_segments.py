#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: verify that PyTorch supports expandable segments when requested.
#
# Trigger:
#   expandable_segments:True in PYTORCH_ALLOC_CONF / PYTORCH_CUDA_ALLOC_CONF /
#   PYTORCH_HIP_ALLOC_CONF, either exported in the environment or set in the
#   top-level `env:` block of the --config YAML.
#
# When the platform does not support expandable segments, PyTorch only prints a
# one-time warning and silently falls back to regular segments. This hook reads
# the effective allocator setting and fails (exit 1) unless expandable segments
# are actually enabled.
#

import os
import sys

ALLOC_CONF_ENVS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")


def _config_path(argv):
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def _config_env(argv):
    """Allocator env vars from the config's top-level `env:` block (applied by the runtime later)."""
    path = _config_path(argv)
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 - the runtime treats a bad `env:` block as best-effort too
        print(f"[WARN] [expandable_segments] Could not read `env:` from {path}: {exc}")
        return {}
    env_block = raw.get("env") if isinstance(raw, dict) else None
    if not isinstance(env_block, dict):
        return {}
    return {str(k): os.path.expandvars(str(v)) for k, v in env_block.items() if str(k) in ALLOC_CONF_ENVS}


def _requests_expandable_segments(conf):
    for item in conf.split(","):
        key, _, value = item.partition(":")
        if key.strip() == "expandable_segments" and value.strip().lower() == "true":
            return True
    return False


def main(argv):
    os.environ.update(_config_env(argv))
    settings = {env: os.environ[env] for env in ALLOC_CONF_ENVS if os.environ.get(env)}
    if not any(_requests_expandable_segments(conf) for conf in settings.values()):
        return 0
    settings_str = ", ".join(f"{k}={v}" for k, v in settings.items())

    try:
        import torch
    except ImportError:
        print("[WARN] [expandable_segments] torch is not importable; skipping expandable segments check.")
        return 0

    if not torch.cuda.is_available():
        print("[WARN] [expandable_segments] No GPU visible; skipping expandable segments check.")
        return 0

    torch_ver = (
        f"torch {torch.__version__} (HIP {torch.version.hip or 'n/a'}, CUDA {torch.version.cuda or 'n/a'})"
    )
    try:
        allocator_settings = torch.cuda.memory._snapshot().get("allocator_settings", {})
    except Exception as exc:  # noqa: BLE001 - e.g. malformed allocator config
        print(
            f"[ERROR] [expandable_segments] Failed to apply allocator config ({settings_str}) with {torch_ver}: {exc}"
        )
        return 1

    if "expandable_segments" not in allocator_settings:
        print(
            f"[WARN] [expandable_segments] {torch_ver} does not expose allocator settings; "
            "skipping expandable segments check."
        )
        return 0

    if not allocator_settings["expandable_segments"]:
        print(
            f"[ERROR] [expandable_segments] expandable_segments:True is requested ({settings_str}) "
            f"but is not in effect with {torch_ver}: this PyTorch build does not support expandable "
            "segments on the current platform, or another allocator env var overrides it with "
            "expandable_segments:False. Use a PyTorch build that supports expandable segments, or "
            "remove expandable_segments:True."
        )
        return 1

    print(f"[INFO] [expandable_segments] Expandable segments verified ({settings_str}, {torch_ver}).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

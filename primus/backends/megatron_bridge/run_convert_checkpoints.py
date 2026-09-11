#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run Megatron-Bridge checkpoint conversion with Primus convert patches applied."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_bridge_convert_checkpoints() -> int:
    """Apply convert patches, then run bridge ``convert_checkpoints.main()``."""
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from primus.backends.megatron_bridge.convert_common import (
        apply_bridge_convert_patches,
    )

    apply_bridge_convert_patches()

    bridge_script = repo_root / "third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py"
    spec = importlib.util.spec_from_file_location("mb_convert_checkpoints", bridge_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Megatron-Bridge converter: {bridge_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["mb_convert_checkpoints"] = module
    spec.loader.exec_module(module)

    result = module.main()
    return int(result or 0)


def main() -> int:
    """Entry point for the Megatron-Bridge conversion wrapper script."""
    return run_bridge_convert_checkpoints()


if __name__ == "__main__":
    raise SystemExit(main())

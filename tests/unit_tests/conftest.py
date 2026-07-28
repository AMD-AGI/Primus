###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import sys
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help=(
            "run tests marked with @pytest.mark.slow (e.g. plan-4 release-tier "
            "shape gates at head_dim=512). Default: skip slow tests."
        ),
    )


def pytest_configure(config):
    # Add project root first to ensure main primus package takes precedence
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Add Megatron-LM after project root (append instead of insert at 0)
    megatron_path = os.environ.get("MEGATRON_PATH")
    if megatron_path is None or not os.path.exists(megatron_path):
        megatron_path = project_root / "third_party" / "Megatron-LM"
    if str(megatron_path) not in sys.path:
        sys.path.append(str(megatron_path))

    # TorchTitan v0.2.2 has PEP-420 namespace subpackages (e.g. torchtitan/tools,
    # no __init__.py) that only import with the source root on sys.path. Insert the
    # submodule at the FRONT so it wins over any stale torchtitan in site-packages.
    torchtitan_path = os.environ.get("TORCHTITAN_PATH")
    if torchtitan_path is None or not os.path.exists(torchtitan_path):
        torchtitan_path = project_root / "third_party" / "torchtitan"
    if str(torchtitan_path) not in sys.path:
        sys.path.insert(0, str(torchtitan_path))

    # Register custom markers used by the primus test suite.
    config.addinivalue_line(
        "markers",
        (
            "slow: marks tests as slow (release-tier shape gates that exercise "
            "production V4 dims at head_dim=512). Skipped by default; opt in "
            "with --run-slow or '-m slow'."
        ),
    )


def pytest_collection_modifyitems(config, items):
    # If the user explicitly opts in via --run-slow or '-m slow', run slow tests.
    if config.getoption("--run-slow", default=False):
        return
    marker_expr = config.getoption("-m") or ""
    if "slow" in marker_expr:
        return
    skip_slow = pytest.mark.skip(reason="slow test (use --run-slow or '-m slow' to enable)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

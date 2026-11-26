###############################################################################
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import builtins
import functools
import os
import re
import sys
from pathlib import Path

import pytest

# Captured before anything imports primus, so the guard below can always restore
# the real print even if something rebinds it during collection.
_PRISTINE_PRINT = builtins.print

# Each xdist worker gets its own GPU; this runs before tests/conftest.py's
# pytest_configure first touches HIP.
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
if _XDIST_WORKER:
    os.environ["HIP_VISIBLE_DEVICES"] = str(
        int(_XDIST_WORKER[2:]) % int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", 8))
    )

# A test module that starts processes, opens a process group or reaches past one
# GPU runs in the --dist-only pass, not one per xdist worker.
_MULTI_PROCESS = re.compile(
    r"torchrun|torch\.distributed\.run|mp\.spawn|multiprocessing|subprocess|MultiProcessTestCase"
    r"|init_process_group|device_count\(\)|cuda:[1-9]|MASTER_PORT"
)


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
    parser.addoption(
        "--dist-only",
        action="store_true",
        default=False,
        help="run only the multi-process tests, which xdist workers skip",
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


@pytest.fixture(autouse=True)
def _restore_builtin_print():
    """Undo Primus's process-wide ``builtins.print`` rebinding between tests.

    ``init_worker_logger`` and ``BaseModule`` repoint ``builtins.print`` at a
    DEBUG-level logger call and never put it back. Once any test starts a worker
    logger, every later test's ``print`` goes to the log sink instead of stdout,
    so ``capsys`` sees nothing.
    """
    yield
    if builtins.print is not _PRISTINE_PRINT:
        builtins.print = _PRISTINE_PRINT


@functools.lru_cache(maxsize=None)
def _is_multi_process(path):
    return bool(_MULTI_PROCESS.search(Path(path).read_text(errors="replace")))


def pytest_collection_modifyitems(config, items):
    # xdist workers take the single-process tests and --dist-only the rest, so
    # the two passes together report every test once.
    if _XDIST_WORKER or config.getoption("--dist-only", default=False):
        kept, dropped = [], []
        for item in items:
            (kept if _is_multi_process(str(item.path)) != bool(_XDIST_WORKER) else dropped).append(item)
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept

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

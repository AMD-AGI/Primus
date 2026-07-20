# Adapted from ODC (https://github.com/sail-sg/odc), which is distributed under
# the MIT License per its package metadata (pyproject.toml / setup.py
# classifiers). The upstream repository ships no LICENSE file or per-file
# copyright headers; upstream copyright is held by the ODC authors (Sea AI Lab).
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# See LICENSE for license information.

import logging

# Eagerly expose the runtime-config API only. This module is a stdlib-only leaf,
# so importing `odc` stays cheap and does NOT pull in odc.primitives -- which is
# essential: the ODC integration patch calls odc.set_runtime_config(...) BEFORE
# the primitives are first imported, so their import-time backend selection
# (odc_p2p_backend) reads the populated config rather than a stale default.
from odc.runtime_config import OdcRuntimeConfig
from odc.runtime_config import get_config as get_runtime_config  # noqa: F401
from odc.runtime_config import set_config as set_runtime_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


# The heavy primitives (which import torch/triton/mori and read the runtime
# config at import time) are exposed lazily via PEP 562 module __getattr__, so
# they are imported only on first access -- after set_runtime_config has run.
_LAZY_EXPORTS = {
    "init_shmem": ("odc.primitives.utils", "init_shmem"),
    "finalize_distributed": ("odc.primitives.utils", "finalize_distributed"),
    "SymmBufferRegistry": ("odc.primitives.utils", "SymmBufferRegistry"),
    "ReductionService": ("odc.primitives.scatter_accumulate", "ReductionService"),
    "GatherService": ("odc.primitives.gather", "GatherService"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'odc' has no attribute '{name}'")
    import importlib

    mod = importlib.import_module(target[0])
    return getattr(mod, target[1])


__all__ = [
    "init_shmem",
    "SymmBufferRegistry",
    "ReductionService",
    "GatherService",
    "finalize_distributed",
    "OdcRuntimeConfig",
    "get_runtime_config",
    "set_runtime_config",
]

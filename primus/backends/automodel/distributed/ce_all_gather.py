###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FSDP2's all-gathers on the GPUs' copy engines (RCCL's zero-CTA symmetric-memory path).

WHY:
  FSDP2's forward all-gathers run as RCCL kernels beside the GEMMs, and the two contend for HBM
  bandwidth, so the GEMMs around each all-gather slow down. RCCL's copy-engine collectives
  move the data with the SDMA engines instead, leaving the compute units and most of the memory
  bandwidth to the GEMMs.

WHAT:
  At the root FSDP2 state's lazy init (the first forward), gives every non-root FSDP2 unit, or
  those under ``fsdp_ce_all_gather_modules``, torch's ``SymmMemAllGather``: its all-gather output
  comes from the NCCL symmetric-memory pool, so RCCL takes the copy-engine path. Reduce-scatters
  and every other collective keep RCCL's kernels. The symmetric-memory pools are kept alive at
  exit, because destroying them after the communicators aborts the process.

REQUIRES:
  Linux >= 6.8 (RCCL's cuMem path hangs on older kernels), an RCCL with copy-engine collectives,
  and ``NCCL_CTA_POLICY=2`` and ``NCCL_CUMEM_ENABLE=1`` in every rank's environment before NCCL
  initializes (Primus's runner exports them with ``FSDP_ALL_GATHER_BACKEND=rccl_sdma``); without
  them the patch is not applied. RCCL's copy-engine path segfaults when it is called on the
  default stream; FSDP2 issues its all-gathers on its own stream.

NUMERICS:
  An all-gather copies bytes, so results are unchanged bit for bit.

Settings (module config, ``primus_perf:`` section):
    fsdp_ce_all_gather: false             apply this patch
    fsdp_ce_all_gather_modules: ""        comma-separated module-path prefixes, relative to the
                                          root FSDP2 module; empty: every unit but the root
"""
from __future__ import annotations

import atexit
import ctypes
import logging
import os
import platform
import re

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_ATTACHED_ATTR = "_primus_ce_all_gather"
# Read once at install; set by the runner before NCCL initializes, not settings.
REQUIRED_ENV = {"NCCL_CTA_POLICY": "2", "NCCL_CUMEM_ENABLE": "1"}
_installed = False


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.fsdp_ce_all_gather")


def kernel_supported() -> bool:
    """Whether the Linux kernel is new enough for RCCL's copy-engine path."""
    m = re.match(r"(\d+)\.(\d+)", platform.release())
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (6, 8)


def missing_env() -> list[str]:
    """``REQUIRED_ENV`` entries this process does not have, as ``NAME=value-it-has``."""
    return [f"{k}={os.environ.get(k)}" for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v]


def pin_symm_mem_pools() -> None:
    """Keep the symmetric-memory pools alive at exit: destroying them after the communicators aborts."""
    from torch.distributed import _symmetric_memory as symm_mem

    for pool in list(getattr(symm_mem, "_symm_mem_pools", {}).values()):
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(pool))


def module_prefixes(value: str) -> list[str]:
    """The module-path prefixes of a comma-separated ``*_modules`` setting."""
    return [p.strip() for p in value.split(",") if p.strip()]


def _attach(root_state, prefixes: list[str]) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import SymmMemAllGather

    root_module = root_state._modules[0]
    names = {module: name for name, module in root_module.named_modules()}
    comms, attached, skipped = {}, [], []
    for state in root_state._state_ctx.all_states:
        if state is root_state:
            continue
        module = state._modules[0]
        name = names.get(module, type(module).__name__)
        if prefixes and not any(name == p or name.startswith(p + ".") for p in prefixes):
            continue
        groups = state._fsdp_param_groups
        if len(groups) != 1:
            skipped.append(name)
            continue
        group = groups[0]._all_gather_process_group
        if group not in comms:
            comms[group] = SymmMemAllGather(group, "NCCL")
        module.set_custom_all_gather(comms[group])
        attached.append(name)
    logger.info(
        "[PrimusCEAllGather] copy-engine all-gather for %d FSDP2 units (%s ... %s)%s.",
        len(attached),
        attached[0] if attached else "-",
        attached[-1] if attached else "-",
        f"; skipped {len(skipped)} with several parameter groups" if skipped else "",
    )


def install() -> bool:
    global _installed
    if _installed:
        return True
    if not kernel_supported():
        logger.warning(
            "[PrimusCEAllGather] Linux %s: RCCL's copy-engine path needs kernel 6.8 or newer; not applied.",
            platform.release(),
        )
        return False
    wrong = missing_env()
    if wrong:
        logger.warning(
            "[PrimusCEAllGather] needs %s at process start (have %s); not applied.",
            " ".join(f"{k}={v}" for k, v in REQUIRED_ENV.items()),
            " ".join(wrong),
        )
        return False

    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    prefixes = module_prefixes(options.string(f"{_SECTION}.fsdp_ce_all_gather_modules", ""))
    orig = FSDPState._lazy_init

    def _lazy_init(self) -> None:
        orig(self)
        if self._is_root and not getattr(self, _ATTACHED_ATTR, False):
            setattr(self, _ATTACHED_ATTR, True)
            _attach(self, prefixes)

    FSDPState._lazy_init = _lazy_init
    atexit.register(pin_symm_mem_pools)
    _installed = True
    logger.info(
        "[PrimusCEAllGather] FSDP2 all-gathers will use RCCL's copy engines%s.",
        f" for units under {', '.join(prefixes)}" if prefixes else "",
    )
    return True

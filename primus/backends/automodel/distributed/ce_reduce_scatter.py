###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FSDP2's reduce-scatters on the GPUs' copy engines: an all-to-all on RCCL's zero-CTA
symmetric-memory path, then the mean of the ranks' copies.

WHY:
  FSDP2's backward reduce-scatters run as RCCL kernels beside the weight-gradient GEMMs, which
  slow down while they overlap. Capping RCCL's channels does not help, because the GEMMs lose to
  the collective's memory traffic, not to the compute units it holds. RCCL has no copy-engine
  reduce-scatter, since it has to add, but its all-to-all on symmetric memory runs on the SDMA
  engines under NCCL_CTA_POLICY=2, and a reduce-scatter is an all-to-all of the chunks followed by
  a local sum.

WHAT:
  At the root FSDP2 state's lazy init (the first forward), gives every non-root FSDP2 unit, or
  those under ``fsdp_ce_reduce_scatter_modules``, a reduce-scatter that runs ``all_to_all_single``
  from FSDP2's input into a staging buffer, then ``torch.mean`` (``torch.sum`` for SUM) over the
  ranks' copies into FSDP2's output. Only the staging buffer comes from the NCCL symmetric-memory
  pool: RCCL takes the copy-engine path when the receive side is symmetric, and FSDP2's buffers
  stay ordinary memory. Its output becomes the sharded gradients, which would otherwise pile up in
  the pool until fresh symmetric segments were needed for every unit, each registration holding
  file descriptors until the process limit ran out. FSDP2 waits for a reduce-scatter before
  copying in the next one, so one staging buffer, grown to the largest unit's gradients, serves
  every call on its stream. Other ops (FSDP2's premultiplied sum for a custom divide factor),
  async calls and calls on the default stream go to RCCL's reduce-scatter instead. The
  symmetric-memory pools are kept alive at exit, as in ``distributed/ce_all_gather.py``.

  ``fsdp_ce_reduce_scatter_debug_check`` runs RCCL's reduce-scatter as well for the first N calls
  and logs the largest relative difference, to validate the path on a new system or RCCL version.

REQUIRES:
  The same as ``distributed/ce_all_gather.py``: Linux >= 6.8, an RCCL with copy-engine collectives and
  ``NCCL_CTA_POLICY=2`` and ``NCCL_CUMEM_ENABLE=1`` in every rank's environment before NCCL
  initializes; without them the patch is not applied. RCCL's copy-engine path segfaults when it
  is called on the default stream; FSDP2 issues its reduce-scatters on its own stream. Memory:
  the staging buffer, the size of the largest unit's gradients.

NUMERICS:
  The mean adds the copies in FP32 and rounds to BF16 once, where RCCL's ring rounds after each
  of its steps, so gradients differ from the default run in the last bits.

Settings (module config, ``primus_perf:`` section):
    fsdp_ce_reduce_scatter: false            apply this patch
    fsdp_ce_reduce_scatter_modules: ""       comma-separated module-path prefixes, relative to the
                                             root FSDP2 module; empty: every unit but the root
    fsdp_ce_reduce_scatter_debug_check: 0    compare the first N calls with RCCL's reduce-scatter
"""
from __future__ import annotations

import atexit
import logging
import platform

from primus.backends.automodel import options
from primus.backends.automodel.distributed import ce_all_gather

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_ATTACHED_ATTR = "_primus_ce_reduce_scatter"
_installed = False


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.fsdp_ce_reduce_scatter")


def _comm_class(check_calls: int):
    import torch
    import torch.distributed as dist
    from torch.distributed import _symmetric_memory as symm_mem
    from torch.distributed.fsdp._fully_shard._fsdp_api import ReduceScatter
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllocMixin

    checked = [0]

    class CopyEngineReduceScatter(DefaultAllocMixin, ReduceScatter):
        def __init__(self, group) -> None:
            symm_mem.set_backend("NCCL")
            # The communicator must exist before the first rendezvous.
            dist.barrier(group=group)
            self._staging = None
            self._stream = None

        def _staging_for(self, input_tensor, group):
            nbytes = input_tensor.numel() * input_tensor.element_size()
            if self._staging is None or self._staging.numel() < nbytes:
                with torch.cuda.use_mem_pool(symm_mem.get_mem_pool(input_tensor.device)):
                    self._staging = torch.empty(nbytes, dtype=torch.uint8, device=input_tensor.device)
                symm_mem.rendezvous(self._staging, group=group.group_name)
            return self._staging[:nbytes].view(input_tensor.dtype)

        def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
            stream = torch.cuda.current_stream()
            if (
                async_op
                or op not in (dist.ReduceOp.AVG, dist.ReduceOp.SUM)
                or stream == torch.cuda.default_stream()
            ):
                return dist.reduce_scatter_tensor(
                    output_tensor, input_tensor, group=group, op=op, async_op=async_op
                )
            if self._stream is not None and stream != self._stream:
                stream.wait_stream(self._stream)  # the staging buffer is shared by all calls
            self._stream = stream
            staging = self._staging_for(input_tensor, group)
            dist.all_to_all_single(staging, input_tensor, group=group)
            copies = staging.view(group.size(), -1)
            if op == dist.ReduceOp.AVG:
                torch.mean(copies, dim=0, out=output_tensor)
            else:
                torch.sum(copies, dim=0, out=output_tensor)
            if checked[0] < check_calls:
                checked[0] += 1
                ref = torch.empty_like(output_tensor)
                dist.reduce_scatter_tensor(ref, input_tensor, group=group, op=op)
                scale = ref.float().abs().max().clamp_min(1e-30)
                err = ((output_tensor.float() - ref.float()).abs().max() / scale).item()
                logger.info(
                    "[PrimusCEReduceScatter] check %d (%d elements): max |diff| / max |RCCL| = %.2e",
                    checked[0],
                    input_tensor.numel(),
                    err,
                )
            return None

    return CopyEngineReduceScatter


def _attach(root_state, prefixes: list[str]) -> None:
    comm_cls = _comm_class(options.integer(f"{_SECTION}.fsdp_ce_reduce_scatter_debug_check", 0))
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
        group = groups[0]._reduce_scatter_process_group
        if group not in comms:
            comms[group] = comm_cls(group)
        module.set_custom_reduce_scatter(comms[group])
        attached.append(name)
    logger.info(
        "[PrimusCEReduceScatter] copy-engine reduce-scatter for %d FSDP2 units (%s ... %s)%s.",
        len(attached),
        attached[0] if attached else "-",
        attached[-1] if attached else "-",
        f"; skipped {len(skipped)} with several parameter groups" if skipped else "",
    )


def install() -> bool:
    global _installed
    if _installed:
        return True
    if not ce_all_gather.kernel_supported():
        logger.warning(
            "[PrimusCEReduceScatter] Linux %s: RCCL's copy-engine path needs kernel 6.8 or newer; not applied.",
            platform.release(),
        )
        return False
    wrong = ce_all_gather.missing_env()
    if wrong:
        logger.warning(
            "[PrimusCEReduceScatter] needs %s at process start (have %s); not applied.",
            " ".join(f"{k}={v}" for k, v in ce_all_gather.REQUIRED_ENV.items()),
            " ".join(wrong),
        )
        return False

    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    prefixes = ce_all_gather.module_prefixes(options.string(f"{_SECTION}.fsdp_ce_reduce_scatter_modules", ""))
    orig = FSDPState._lazy_init

    def _lazy_init(self) -> None:
        orig(self)
        if self._is_root and not getattr(self, _ATTACHED_ATTR, False):
            setattr(self, _ATTACHED_ATTR, True)
            _attach(self, prefixes)

    FSDPState._lazy_init = _lazy_init
    atexit.register(ce_all_gather.pin_symm_mem_pools)
    _installed = True
    logger.info(
        "[PrimusCEReduceScatter] FSDP2 reduce-scatters will use RCCL's copy engines%s.",
        f" for units under {', '.join(prefixes)}" if prefixes else "",
    )
    return True

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TorchTitan / FSDP2 MORI all-gather patch.

Activation:
    Export ``FSDP_ALL_GATHER_BACKEND=mori`` before launching a TorchTitan
    pretrain.
    The patch wraps ``torch.distributed.fsdp.fully_shard`` and attaches
    MORI's FSDP2 ``AllGather`` backend to each fully-sharded module via
    ``set_custom_all_gather``.

MORI routes intra-node traffic over SDMA and cross-node traffic over RDMA.
Reduce-scatter stays on the framework default path.
"""

import functools
import os
import sys

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _mori_all_gather_enabled(ctx: PatchContext) -> bool:
    """Gate on the MORI all-gather backend."""
    return os.environ.get("FSDP_ALL_GATHER_BACKEND", "") == "mori"


@register_patch(
    "torchtitan.fsdp.mori_allgather",
    backend="torchtitan",
    phase="setup",
    description=(
        "Auto-attach MORI HierAllGather to every fully_shard'd module so "
        "FSDP all-gather uses intra-node SDMA and cross-node RDMA. "
        "Gated on FSDP_ALL_GATHER_BACKEND=mori."
    ),
    condition=_mori_all_gather_enabled,
    priority=40,
)
def patch_torchtitan_fsdp_mori_allgather(ctx: PatchContext) -> None:
    """Install MORI's FSDP2 all-gather backend for TorchTitan modules."""
    import torch.distributed.fsdp as _fsdp_pkg
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _ffsc
    from torch.distributed.fsdp._fully_shard import _fully_shard as _ffs_mod

    from primus.backends.common.mori_allgather import MoriAllGather

    orig_fully_shard = _fsdp_pkg.fully_shard
    mori_all_gather = MoriAllGather()
    log_all = os.environ.get("MORI_LOG_ATTACH", "0") == "1"

    def _attach_mori_all_gather(fsdp_module) -> None:
        try:
            state = fsdp_module._get_fsdp_state()
        except Exception as e:
            if log_all:
                log_rank_0(
                    f"[Patch:torchtitan.fsdp.mori_allgather] "
                    f"skip (no _fsdp_state): {type(fsdp_module).__name__}: {e}"
                )
            return

        groups = getattr(state, "_fsdp_param_groups", None) or []
        if len(groups) != 1:
            if log_all:
                log_rank_0(
                    f"[Patch:torchtitan.fsdp.mori_allgather] "
                    f"skip multi-group module ({len(groups)} groups): "
                    f"{type(fsdp_module).__name__}"
                )
            return

        try:
            fsdp_module.set_custom_all_gather(mori_all_gather)
        except (AttributeError, ValueError, AssertionError) as e:
            log_rank_0(
                f"[Patch:torchtitan.fsdp.mori_allgather] WARN: failed to "
                f"attach MoriAllGather to {type(fsdp_module).__name__}: {e}"
            )
            return
        if log_all:
            pg = groups[0]._all_gather_process_group
            log_rank_0(
                f"[Patch:torchtitan.fsdp.mori_allgather] attached "
                f"MoriAllGather to {type(fsdp_module).__name__} "
                f"(group={pg.group_name})"
            )

    @functools.wraps(orig_fully_shard)
    def wrapped_fully_shard(module, *args, **kwargs):
        result = orig_fully_shard(module, *args, **kwargs)
        _attach_mori_all_gather(result if result is not None else module)
        return result

    for _attr in dir(orig_fully_shard):
        if _attr.startswith("__"):
            continue
        try:
            setattr(wrapped_fully_shard, _attr, getattr(orig_fully_shard, _attr))
        except (AttributeError, TypeError):
            pass

    _fsdp_pkg.fully_shard = wrapped_fully_shard
    if hasattr(_ffs_mod, "fully_shard"):
        _ffs_mod.fully_shard = wrapped_fully_shard

    # TorchTitan model modules may already hold a local
    # ``from torch.distributed.fsdp import fully_shard`` alias by the time
    # Primus runs setup patches. Replace every loaded alias that still points
    # at the exact original function so those call sites also attach MORI.
    patched_aliases = 0
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, "fully_shard", None) is orig_fully_shard:
                setattr(module, "fully_shard", wrapped_fully_shard)
                patched_aliases += 1
        except (AttributeError, TypeError):
            continue

    log_rank_0(
        "[Patch:torchtitan.fsdp.mori_allgather] installed: every "
        "fully_shard()-d module will use MoriAllGather for FSDP all-gather; "
        f"patched {patched_aliases} loaded aliases."
    )

    _ffsc._primus_mori_orig_fully_shard = orig_fully_shard
    _ffsc._primus_mori_attach = _attach_mori_all_gather

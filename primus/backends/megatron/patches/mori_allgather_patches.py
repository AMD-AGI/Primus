###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron FSDP2 MORI all-gather patch.

Activation:
    Export ``FSDP_ALL_GATHER_BACKEND=mori`` and run Megatron with
    ``use_torch_fsdp2: true``. This patch wraps the ``fully_shard`` symbol
    used by Megatron's Torch FSDP2 wrapper and attaches MORI's FSDP2
    ``AllGather`` backend to transformer-layer units.

Scope:
    Transformer layers only for the first integration pass. Embedding,
    lm_head, rotary, and reduce-scatter stay on framework defaults.
"""

import functools
import os
import sys

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _mori_all_gather_enabled(ctx: PatchContext) -> bool:
    """Gate on the MORI backend and the Megatron FSDP2 path."""
    if os.environ.get("FSDP_ALL_GATHER_BACKEND", "") != "mori":
        return False
    return getattr(get_args(ctx), "use_torch_fsdp2", False)


@register_patch(
    "megatron.fsdp.mori_allgather",
    backend="megatron",
    phase="before_train",
    description=(
        "Attach MORI HierAllGather to Megatron FSDP2 transformer-layer "
        "modules so all-gather uses intra-node SDMA and cross-node RDMA. "
        "Gated on FSDP_ALL_GATHER_BACKEND=mori and use_torch_fsdp2."
    ),
    condition=_mori_all_gather_enabled,
    priority=40,
)
def patch_megatron_fsdp_mori_allgather(ctx: PatchContext) -> None:
    """Install MORI's FSDP2 all-gather backend for Megatron transformer layers."""
    import megatron.core.distributed.torch_fully_sharded_data_parallel as _mfsdp_mod
    import torch.distributed.fsdp as _fsdp_pkg
    from torch.distributed.fsdp._fully_shard import _fully_shard as _ffs_mod

    from primus.backends.common.mori_allgather import MoriAllGather

    if not getattr(_mfsdp_mod, "HAVE_FSDP", False) or not hasattr(_mfsdp_mod, "fully_shard"):
        warning_rank_0(
            "[Patch:megatron.fsdp.mori_allgather] "
            "torch_fully_sharded_data_parallel.fully_shard not found; skipping."
        )
        return

    try:
        from megatron.core.transformer.transformer_layer import TransformerLayer
    except Exception as e:
        warning_rank_0(
            "[Patch:megatron.fsdp.mori_allgather] could not import " f"TransformerLayer; skipping: {e}"
        )
        return

    orig_fully_shard = _fsdp_pkg.fully_shard
    mori_all_gather = MoriAllGather()
    log_all = os.environ.get("MORI_LOG_ATTACH", "0") == "1"

    def _attach_mori_all_gather(fsdp_module) -> None:
        """Attach MoriAllGather to transformer-layer units only."""
        if not isinstance(fsdp_module, TransformerLayer):
            if log_all:
                log_rank_0(
                    "[Patch:megatron.fsdp.mori_allgather] skip non-transformer "
                    f"module: {type(fsdp_module).__name__}"
                )
            return
        try:
            state = fsdp_module._get_fsdp_state()
        except Exception as e:
            if log_all:
                log_rank_0(
                    "[Patch:megatron.fsdp.mori_allgather] skip "
                    f"{type(fsdp_module).__name__} without FSDP state: {e}"
                )
            return

        groups = getattr(state, "_fsdp_param_groups", None) or []
        if len(groups) != 1:
            if log_all:
                log_rank_0(
                    "[Patch:megatron.fsdp.mori_allgather] skip "
                    f"{type(fsdp_module).__name__} with {len(groups)} param groups"
                )
            return

        try:
            fsdp_module.set_custom_all_gather(mori_all_gather)
        except (AttributeError, ValueError, AssertionError) as e:
            warning_rank_0(
                f"[Patch:megatron.fsdp.mori_allgather] WARN: failed to "
                f"attach MoriAllGather to {type(fsdp_module).__name__}: {e}"
            )
            return
        if log_all:
            pg = groups[0]._all_gather_process_group
            log_rank_0(
                "[Patch:megatron.fsdp.mori_allgather] attached MoriAllGather "
                f"to {type(fsdp_module).__name__} (group={pg.group_name})"
            )

    @functools.wraps(orig_fully_shard)
    def wrapped_fully_shard(module, *args, **kwargs):
        result = orig_fully_shard(module, *args, **kwargs)
        _attach_mori_all_gather(result if result is not None else module)
        return result

    for attr in dir(orig_fully_shard):
        if attr.startswith("__"):
            continue
        try:
            setattr(wrapped_fully_shard, attr, getattr(orig_fully_shard, attr))
        except (AttributeError, TypeError):
            pass

    _fsdp_pkg.fully_shard = wrapped_fully_shard
    if hasattr(_ffs_mod, "fully_shard"):
        _ffs_mod.fully_shard = wrapped_fully_shard
    _mfsdp_mod.fully_shard = wrapped_fully_shard

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
        "[Patch:megatron.fsdp.mori_allgather] installed: FSDP2 "
        "transformer-layer modules in Megatron's TorchFullyShardedDataParallel "
        f"will use MoriAllGather; patched {patched_aliases} loaded aliases."
    )

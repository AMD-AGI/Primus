###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan / FSDP2 SDMA-eligible All-Gather Patch

What this patches:
    Wraps ``torch.distributed.fsdp.fully_shard`` so that every
    fully-sharded module also gets
    ``set_custom_all_gather(SymmMemAllGather(...))`` attached
    immediately after construction.

Activation:
    Export ``SDMA_ALL_GATHER=1`` in the shell before launching any
    torchtitan pretrain (e.g. ``primus-cli direct -- train pretrain
    --config <existing YAML>``). The patch is a no-op otherwise.

    The companion hook
    ``runner/helpers/hooks/06_enable_sdma_all_gather.sh`` runs at
    ``primus-cli`` startup and, when ``SDMA_ALL_GATHER=1``, exports
    the standard zero-CTA env (``NCCL_CTA_POLICY=2``,
    ``NCCL_CUMEM_ENABLE=1``, ...) and the LD_PRELOAD interposer so no
    YAML or script changes are required to opt in.

    ``SDMA_ALL_GATHER`` is the only knob; there are no sub-options.

Requirements:
    - PyTorch >= 2.12 (introduces ``SymmMemAllGather``).
    - On ROCm, the underlying RCCL transport must be able to take the
      copy-engine path; FSDP already requests zero-CTA via
      ``pg_options`` so this is normally automatic. Verify with
      ``NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1`` if in doubt; both
      paths produce correct results either way.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def _sdma_all_gather_enabled(ctx: PatchContext) -> bool:
    """Single env-driven gate. Triggered only by ``SDMA_ALL_GATHER=1``."""
    return os.environ.get("SDMA_ALL_GATHER", "0") == "1"


@register_patch(
    "torchtitan.fsdp.sdma_symm_mem_collectives",
    backend="torchtitan",
    phase="setup",
    description=(
        "Auto-attach SymmMemAllGather to every fully_shard'd module "
        "so FSDP's all-gather uses the SDMA (copy-engine) dispatch "
        "path. Gated on SDMA_ALL_GATHER=1."
    ),
    condition=_sdma_all_gather_enabled,
)
def patch_torchtitan_fsdp_sdma_symm_mem(ctx: PatchContext) -> None:
    """
    Wrap ``torch.distributed.fsdp.fully_shard`` so post-construction we
    attach ``SymmMemAllGather`` on the FSDP module, routing every
    all-gather through a cuMem-backed symm_mem buffer that the
    NCCL/RCCL transport recognizes as eligible for the SDMA dispatch
    path. Reduce-scatter is left on its FSDP default.
    """
    import functools

    import torch.distributed.fsdp as _fsdp_pkg
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _ffsc
    from torch.distributed.fsdp._fully_shard import _fully_shard as _ffs_mod
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import SymmMemAllGather

    # Hardcoded sensible defaults. SDMA_ALL_GATHER is the only knob.
    backend = "NCCL"
    log_all = False

    orig_fully_shard = _fsdp_pkg.fully_shard

    def _attach_symm_mem_all_gather(fsdp_module) -> None:
        """Switch a fully_shard'd module's AG comm to the SymmMem flavor."""
        try:
            state = fsdp_module._get_fsdp_state()
        except Exception as e:
            if log_all:
                log_rank_0(
                    f"[Patch:sdma_symm_mem] skip (no _fsdp_state): " f"{type(fsdp_module).__name__}: {e}"
                )
            return

        groups = getattr(state, "_fsdp_param_groups", None) or []
        if len(groups) != 1:
            # set_custom_all_gather rejects modules with multiple param
            # groups (e.g. per-param mesh via shard_placement_fn). Leave
            # those alone; they will use whatever comm FSDP chose.

            # Lorri: This was tested on Pytorch 2.12,
            # other versions may be different.
            if log_all:
                log_rank_0(
                    f"[Patch:sdma_symm_mem] skip multi-group module "
                    f"({len(groups)} groups): {type(fsdp_module).__name__}"
                )
            return

        pg = groups[0]._all_gather_process_group
        try:
            fsdp_module.set_custom_all_gather(SymmMemAllGather(pg, backend))
        except (AttributeError, ValueError, AssertionError) as e:
            log_rank_0(
                f"[Patch:sdma_symm_mem] WARN: failed to attach SymmMemAllGather "
                f"to {type(fsdp_module).__name__}: {e}"
            )
            return
        if log_all:
            log_rank_0(
                f"[Patch:sdma_symm_mem] attached SymmMemAllGather "
                f"to {type(fsdp_module).__name__} (group={pg.group_name})"
            )

    @functools.wraps(orig_fully_shard)
    def wrapped_fully_shard(module, *args, **kwargs):
        result = orig_fully_shard(module, *args, **kwargs)
        # fully_shard mutates `module` in place and returns it; this is
        # the FSDP-augmented module.
        _attach_symm_mem_all_gather(result)
        return result

    # Copy across any non-dunder attributes from the original function
    # to the wrapper. In particular, FSDP attaches `fully_shard.state`
    # (a method) that nested fully_shard()ing reads as
    # `fully_shard.state(modules[0])` in `_fully_shard.py`. If we don't
    # propagate it, the first inner call crashes with
    # `AttributeError: 'function' object has no attribute 'state'`.
    for _attr in dir(orig_fully_shard):
        if _attr.startswith("__"):
            continue
        try:
            setattr(wrapped_fully_shard, _attr, getattr(orig_fully_shard, _attr))
        except (AttributeError, TypeError):
            pass

    # Patch every alias of fully_shard that torchtitan (or anyone else)
    # may have already pulled in via `from ... import fully_shard`. We
    # patch the package re-export AND the inner module's binding; both
    # are the same function object at this point, and torchtitan
    # imports happen later (in TorchTitanPretrainTrainer.__init__,
    # after run_patches('setup') returns).
    _fsdp_pkg.fully_shard = wrapped_fully_shard
    if hasattr(_ffs_mod, "fully_shard"):
        _ffs_mod.fully_shard = wrapped_fully_shard

    log_rank_0(
        "[Patch:torchtitan.fsdp.sdma_symm_mem_collectives] installed: "
        f"every fully_shard()-d module will use "
        f"SymmMemAllGather (backend={backend})."
    )

    # Convenience: stash references for tests / debugging.
    _ffsc._primus_sdma_orig_fully_shard = orig_fully_shard
    _ffsc._primus_sdma_attach = _attach_symm_mem_all_gather

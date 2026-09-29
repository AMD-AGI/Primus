###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Mamba ROCm Patches

Patches for Mamba model compatibility on AMD ROCm GPUs.
Disables Triton buffer_ops in the chunk_state backward pass to avoid
ROCm-specific correctness issues.
"""

import torch

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0

# Megatron-Bridge configs carry no ``model_type``, they name models by recipe.
# Zebra hybrid models (Mamba+MLA) use MCoreMambaModel.
_BRIDGE_MAMBA_RECIPES = ("mamba", "zebra")


def _is_rocm_mamba_run(ctx: PatchContext) -> bool:
    """Return True on ROCm for a run that actually builds Mamba layers.

    Applying the patch imports ``mamba_ssm``, which reaches quack-kernels and
    CUTLASS DSL, whose MLIR bindings then abort the process on FlyDSL's already
    in nanobind's registry -- killing runs with no Mamba layer at all (#955).
    Genuine Mamba runs import ``mamba_ssm`` anyway, so those are protected by
    ``purge_cutlass_dsl`` in tools/installation/setup.sh, not by this gate.
    """
    if getattr(torch.version, "hip", None) is None:
        return False
    # Megatron: plain Mamba and the hybrids both set ``model_type: mamba``.
    if get_param(ctx, "model_type") == "mamba":
        return True
    recipe = str(get_param(ctx, "recipe", "") or "")
    return any(family in recipe for family in _BRIDGE_MAMBA_RECIPES)


def _make_triton_wrapper(original_fn):
    from triton import knobs as _triton_knobs

    def _chunk_state_bwd_db_no_buffer_ops(x, dt, dA_cumsum, dstates, seq_idx=None, B=None, ngroups=1):
        with _triton_knobs.amd.scope():
            _triton_knobs.amd.use_buffer_ops = False
            return original_fn(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups)

    return _chunk_state_bwd_db_no_buffer_ops


@register_patch(
    "megatron.mamba.rocm_chunk_state_bwd_db",
    backend="megatron",
    phase="before_train",
    description=(
        "Disable Triton buffer_ops in Mamba _chunk_state_bwd_db backward pass "
        "to work around ROCm-specific correctness issues."
    ),
    condition=_is_rocm_mamba_run,
    tags=["rocm", "mamba"],
)
def patch_mamba_rocm_chunk_state_bwd_db(ctx: PatchContext):
    """
    Patch mamba_ssm _chunk_state_bwd_db to disable Triton buffer_ops on ROCm.

    The Triton buffer_ops feature can cause correctness issues on AMD GPUs
    during the backward pass of the Mamba chunk state computation. This patch
    wraps the original function to set use_buffer_ops = False within an AMD
    Triton knobs scope.

    Both ``ssd_chunk_state`` (definition) and ``ssd_combined`` (import-time
    binding) module namespaces are patched so every call-site picks up the
    wrapper regardless of which module it was imported from.
    """
    import mamba_ssm.ops.triton.ssd_chunk_state as ssd_chunk_state
    import mamba_ssm.ops.triton.ssd_combined as ssd_combined

    original_fn = ssd_chunk_state._chunk_state_bwd_db
    wrapped_fn = _make_triton_wrapper(original_fn)

    # Patch the canonical definition in ssd_chunk_state
    ssd_chunk_state._chunk_state_bwd_db = wrapped_fn
    log_rank_0(
        "[Patch:megatron.mamba.rocm_chunk_state_bwd_db] "
        "Patched mamba_ssm.ops.triton.ssd_chunk_state._chunk_state_bwd_db"
    )

    # Patch the import-time binding in ssd_combined
    ssd_combined._chunk_state_bwd_db = wrapped_fn
    log_rank_0(
        "[Patch:megatron.mamba.rocm_chunk_state_bwd_db] "
        "Patched mamba_ssm.ops.triton.ssd_combined._chunk_state_bwd_db"
    )

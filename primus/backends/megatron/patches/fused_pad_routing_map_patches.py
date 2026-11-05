###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus fused_pad_routing_map patch.

Globally replaces Megatron's ``fused_pad_routing_map`` with the Primus Triton
implementation (``primus.backends.megatron.core.fusions.fused_pad_routing_map``)
without modifying upstream Megatron source.

The Primus version rewrites the kernel to operate directly on the native
``[num_tokens, num_experts]`` layout (no transpose/copy) and drops the
``@jit_fuser`` (``torch.compile``) wrapper, avoiding the Triton kernel
functionalization failure seen on some torch/triton combos.
"""

from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


@register_patch(
    "megatron.fused_pad_routing_map",
    backend="megatron",
    phase="before_train",
    description="Replace Megatron fused_pad_routing_map with the Primus Triton implementation",
)
def patch_fused_pad_routing_map(ctx: PatchContext):
    """Swap the ``fused_pad_routing_map`` symbol everywhere it is referenced."""
    import megatron.core.fusions.fused_pad_routing_map as meg_mod

    from primus.backends.megatron.core.fusions.fused_pad_routing_map import (
        fused_pad_routing_map as primus_fused_pad_routing_map,
    )

    log_rank_0("[Patch:megatron.fused_pad_routing_map] Patching fused_pad_routing_map...")

    current = getattr(meg_mod, "fused_pad_routing_map", None)
    if current is primus_fused_pad_routing_map:
        log_rank_0("[Patch:megatron.fused_pad_routing_map]   Already patched; skipping.")
        return
    if current is None:
        warning_rank_0(
            "[Patch:megatron.fused_pad_routing_map]   Megatron no longer defines "
            "megatron.core.fusions.fused_pad_routing_map.fused_pad_routing_map; skipping."
        )
        return

    # Replace on the source module so all *future* (incl. lazy) imports resolve to the
    # Primus version, and in every module that imported the symbol at top level before
    # this patch ran (precise: only objects that `is` the original).
    rebound = rebind_everywhere(meg_mod, "fused_pad_routing_map", primus_fused_pad_routing_map)

    log_rank_0(
        "[Patch:megatron.fused_pad_routing_map]   Patched "
        "megatron.core.fusions.fused_pad_routing_map.fused_pad_routing_map "
        f"-> primus (also updated already-imported refs: {rebound[1:] or 'none'})"
    )

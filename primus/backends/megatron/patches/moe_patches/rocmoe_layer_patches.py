###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron MoE ROCMoE Layer Patch.

Replace the Megatron ``MoELayer`` with ``ROCMoELayer`` (ROCMoE EP engine for
dispatch + experts + combine; Megatron router / shared expert kept) when the
``use_rocmoe`` flag is set.  Mirrors the class-swap approach of
``deprecated_layer_patches.py``: the MoE module spec is built (after this
``before_train`` patch) from ``moe_module_specs.MoELayer``, so rebinding that
name -- plus the ``moe_layer`` module attribute -- routes model construction to
our layer.
"""

import sys

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.moe.rocmoe",
    backend="megatron",
    phase="before_train",
    description="Replace MoELayer with the ROCMoE EP-engine-backed layer",
    condition=lambda ctx: getattr(get_args(ctx), "use_rocmoe", False),
)
def patch_rocmoe_layer(ctx: PatchContext):
    """Rebind MoELayer -> ROCMoELayer in the spec/module namespaces."""
    from primus.backends.megatron.core.transformer.moe.rocmoe.moe_layer import (
        ROCMoELayer,
    )

    sys.modules["megatron.core.transformer.moe.moe_layer"].MoELayer = ROCMoELayer
    log_rank_0(
        "[Patch:megatron.moe.rocmoe]   Patched "
        "megatron.core.transformer.moe.moe_layer.MoELayer "
        f"-> {ROCMoELayer.__name__}"
    )

    from megatron.core.models.gpt import moe_module_specs

    moe_module_specs.MoELayer = ROCMoELayer
    log_rank_0(
        "[Patch:megatron.moe.rocmoe]   Patched "
        "megatron.core.models.gpt.moe_module_specs.MoELayer "
        f"-> {ROCMoELayer.__name__}"
    )

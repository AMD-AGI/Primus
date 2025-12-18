###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE Deprecated Layer Patches

Patches for using deprecated MoE layer implementations from 2024-12-09.
"""

import sys

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.moe.deprecated_20241209",
    backend="megatron",
    phase="before_train",
    description="Replace MoELayer/experts with deprecated 20241209 versions",
    condition=lambda ctx: getattr(get_args(ctx), "use_deprecated_20241209_moe_layer", False),
)
def patch_deprecated_moe_layer(ctx: PatchContext):
    """
    Patch Megatron to use deprecated MoE layer implementations.

    Behavior:
        - Replace core MoELayer / MoESubmodules / experts with deprecated versions
        - Sync replacements into gpt.moe_module_specs
    """
    # patch module class
    from primus.backends.megatron.core.transformer.moe.deprecated_20251209.experts import (
        DeprecatedGroupedMLP,
        DeprecatedSequentialMLP,
        DeprecatedTEGroupedMLP,
    )
    from primus.backends.megatron.core.transformer.moe.deprecated_20251209.moe_layer import (
        DeprecatedMoELayer,
        DeprecatedMoESubmodules,
    )

    sys.modules["megatron.core.transformer.moe.moe_layer"].MoELayer = DeprecatedMoELayer
    sys.modules["megatron.core.transformer.moe.moe_layer"].MoESubmodules = DeprecatedMoESubmodules
    sys.modules["megatron.core.transformer.moe.experts"].GroupedMLP = DeprecatedGroupedMLP
    sys.modules["megatron.core.transformer.moe.experts"].SequentialMLP = DeprecatedSequentialMLP
    sys.modules["megatron.core.transformer.moe.experts"].TEGroupedMLP = DeprecatedTEGroupedMLP

    # patch imported module
    from megatron.core.models.gpt import moe_module_specs

    moe_module_specs.MoELayer = DeprecatedMoELayer
    moe_module_specs.MoESubmodules = DeprecatedMoESubmodules
    moe_module_specs.GroupedMLP = DeprecatedGroupedMLP
    moe_module_specs.SequentialMLP = DeprecatedSequentialMLP
    moe_module_specs.TEGroupedMLP = DeprecatedTEGroupedMLP

    log_rank_0(
        f"[Patch:megatron.moe.deprecated_20241209]   Patched megatron.core.models.gpt.moe_module_specs.MoELayer "
        f"-> {DeprecatedMoELayer.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.deprecated_20241209]   Patched megatron.core.models.gpt.moe_module_specs.MoESubmodules "
        f"-> {DeprecatedMoESubmodules.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.deprecated_20241209]   Patched megatron.core.models.gpt.moe_module_specs.GroupedMLP "
        f"-> {DeprecatedGroupedMLP.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.deprecated_20241209]   Patched megatron.core.models.gpt.moe_module_specs.SequentialMLP "
        f"-> {DeprecatedSequentialMLP.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.deprecated_20241209]   Patched megatron.core.models.gpt.moe_module_specs.TEGroupedMLP "
        f"-> {DeprecatedTEGroupedMLP.__name__}"
    )

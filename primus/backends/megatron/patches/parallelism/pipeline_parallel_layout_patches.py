###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron PipelineParallelLayerLayout replacement patches.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.core.transformer.pipeline_parallel_layer_layout",
    backend="megatron",
    phase="before_train",
    description="Replace PipelineParallelLayerLayout with Primus implementation",
)
def patch_pipeline_parallel_layer_layout(ctx: PatchContext):  # noqa: ARG001
    """
    Patch Megatron to use Primus's PipelineParallelLayerLayout.

    Behavior:
        - Replace megatron.core.transformer.pipeline_parallel_layer_layout.PipelineParallelLayerLayout
          with PrimusPipelineParallelLayerLayout
    """
    import megatron.core.transformer.pipeline_parallel_layer_layout as orig_pipeline_parallel_layer_layout

    from primus.backends.megatron.core.transformer.pipeline_parallel_layer_layout import (
        PrimusPipelineParallelLayerLayout,
    )

    orig_pipeline_parallel_layer_layout.PipelineParallelLayerLayout = PrimusPipelineParallelLayerLayout
    log_rank_0(
        "[Patch:megatron.core.transformer.pipeline_parallel_layer_layout]   "
        "Patched megatron.core.transformer.pipeline_parallel_layer_layout.PipelineParallelLayerLayout "
        f"-> {PrimusPipelineParallelLayerLayout.__name__}"
    )

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in Primus-Turbo tensorwise-FP8 GPT output projection.

``GPTModel`` constructs its output layer directly as Megatron's native
``ColumnParallelLinear``, bypassing the backend spec provider.  Replace that
constructor only for the duration of ``GPTModel.__init__`` so unrelated native
column-parallel layers remain untouched.
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _can_route_fp8_output_layer(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        bool(getattr(args, "use_turbo_fp8_output_layer", False))
        and bool(getattr(args, "use_turbo_gemm", False))
        and int(getattr(args, "tensor_model_parallel_size", 1)) == 1
        and is_primus_turbo_can_patch(ctx)
    )


@register_patch(
    "megatron.turbo.fp8_output_layer",
    backend="megatron",
    phase="before_train",
    description="Route the GPT output projection through Turbo tensorwise E4M3 FP8",
    condition=_can_route_fp8_output_layer,
)
def patch_fp8_output_layer(ctx: PatchContext) -> None:
    del ctx
    from megatron.core.models.gpt import gpt_model

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboFP8OutputColumnParallelLinear,
    )

    patch_key = "megatron.turbo.fp8_output_layer"
    if is_patched(gpt_model, patch_key):
        return

    original_gpt_init = gpt_model.GPTModel.__init__

    def _gpt_init_with_turbo_fp8_output(self, *args, **kwargs):
        original_column_parallel = gpt_model.tensor_parallel.ColumnParallelLinear
        gpt_model.tensor_parallel.ColumnParallelLinear = (
            PrimusTurboFP8OutputColumnParallelLinear
        )
        try:
            return original_gpt_init(self, *args, **kwargs)
        finally:
            gpt_model.tensor_parallel.ColumnParallelLinear = original_column_parallel

    gpt_model.GPTModel.__init__ = _gpt_init_with_turbo_fp8_output
    mark_patched(gpt_model, patch_key)

    log_rank_0(
        "[Patch:megatron.turbo.fp8_output_layer] Routed GPT output projection "
        "through Primus-Turbo tensorwise E4M3 FP8"
    )

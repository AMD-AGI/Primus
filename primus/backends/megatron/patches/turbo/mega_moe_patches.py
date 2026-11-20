###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo Mega MoE Patches

Replace Megatron's ``MoELayer`` with the PrimusTurbo ``MegaMoE`` adapter.
EP-only (TP==1) + bf16 params.

``turbo_mega_moe_precision`` (``bf16`` | ``mxfp8``) picks the expert flavour once the layer is
patched in. It is deliberately NOT wired to Megatron's ``--fp8``: that selects a TE fp8 recipe for
the dense layers and has no path to this fused op, so the MoE stays switchable on its own for A/B
runs.
"""

import functools

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _is_turbo_mega_moe_can_patch(ctx: PatchContext) -> bool:
    """
    Check if the PrimusTurbo fused Mega MoE layer is enabled.

    Requires:
      - primus_turbo package is installed
      - enable_primus_turbo == True
      - use_turbo_mega_moe == True
      - 1 <= expert_model_parallel_size <= 8 (EP=1: one rank holds every expert; the dispatch/combine
        stay the same kernels with the peer set reduced to itself)
    """
    args = get_args(ctx)
    use_turbo_mega_moe = bool(getattr(args, "use_turbo_mega_moe", False))
    bf16 = bool(getattr(args, "bf16", False))
    # The fused all-to-all rides intra-node peer transfers, so the EP group has to fit in one node.
    ep_size = int(getattr(args, "expert_model_parallel_size", 1))

    return use_turbo_mega_moe and bf16 and 1 <= ep_size <= 8 and is_primus_turbo_can_patch(ctx)


def _is_turbo_mega_moe_mxfp8(ctx: PatchContext) -> bool:
    """MegaMoE with mxfp8 experts -- the only flavour that caches quantized weights."""
    precision = getattr(get_args(ctx), "turbo_mega_moe_precision", "bf16")
    return precision == "mxfp8" and _is_turbo_mega_moe_can_patch(ctx)


@register_patch(
    "megatron.turbo.mega_moe",
    backend="megatron",
    phase="before_train",
    description="Replace MoELayer with the PrimusTurbo MegaMoE layer",
    condition=_is_turbo_mega_moe_can_patch,
)
def patch_mega_moe(ctx: PatchContext):
    """
    Patch Megatron to use the PrimusTurbo MegaMoE layer.

    Replaces ``MoELayer`` in both ``moe_layer`` and ``gpt.moe_module_specs`` so the
    ``== MoELayer`` identity check in ``transformer_layer.py`` stays consistent.
    """
    from megatron.core.models.gpt import moe_module_specs
    from megatron.core.transformer.moe import moe_layer

    from primus.backends.megatron.core.extensions.mega_moe import (
        PrimusTurboMegaMoELayer,
    )

    # read off ctx, not the layer's helper: megatron's get_args is not up yet in this phase
    precision = getattr(get_args(ctx), "turbo_mega_moe_precision", "bf16")
    log_rank_0(
        f"[Patch:megatron.turbo.mega_moe] Patching MoELayer with fused MegaMoE ({precision} experts)..."
    )

    moe_layer.MoELayer = PrimusTurboMegaMoELayer
    log_rank_0(
        "[Patch:megatron.turbo.mega_moe]   Patched "
        f"megatron.core.transformer.moe.moe_layer.MoELayer -> {PrimusTurboMegaMoELayer.__name__}"
    )

    moe_module_specs.MoELayer = PrimusTurboMegaMoELayer
    log_rank_0(
        "[Patch:megatron.turbo.mega_moe]   Patched "
        f"megatron.core.models.gpt.moe_module_specs.MoELayer -> {PrimusTurboMegaMoELayer.__name__}"
    )


@register_patch(
    "megatron.turbo.mega_moe_weight_generation",
    backend="megatron",
    phase="before_train",
    description="Drop the MegaMoE mxfp8 weight-quant cache after every optimizer step",
    condition=_is_turbo_mega_moe_mxfp8,
)
def patch_mega_moe_weight_generation(ctx: PatchContext):
    """Tie the mxfp8 expert weight-quant cache to the optimizer step.

    The op keys that cache on ``w._version``, which the precision-aware optimizer never bumps, so
    left alone the experts train the whole run against the weights they started with. Taking the
    signal from the step itself keeps it honest: counting the layer's own forwards is thrown off by
    anything that forwards without stepping (recompute, warm-up, eval). Megatron's
    ``set_is_first_microbatch()`` cannot carry it -- it no-ops unless Megatron's own fp8 recipe is on.
    """
    import megatron.training.training as megatron_training
    from primus_turbo.pytorch.kernels.fused_mega_moe import advance_weight_generation

    _original_train_step = megatron_training.train_step

    @functools.wraps(_original_train_step)
    def _patched_train_step(*args, **kwargs):
        result = _original_train_step(*args, **kwargs)
        advance_weight_generation()
        return result

    megatron_training.train_step = _patched_train_step
    log_rank_0(
        "[Patch:megatron.turbo.mega_moe_weight_generation] "
        "Patched train_step to advance the MegaMoE mxfp8 weight generation after optimizer.step()"
    )


def _is_turbo_mega_moe_fp8_block(ctx: PatchContext) -> bool:
    """MegaMoE with 128-block FP8 (UE8M0) experts: quantized weights in fixed buffers."""
    precision = getattr(get_args(ctx), "turbo_mega_moe_precision", "bf16")
    return precision == "fp8_block" and _is_turbo_mega_moe_can_patch(ctx)


@register_patch(
    "megatron.turbo.mega_moe_fp8blk_weights",
    backend="megatron",
    phase="before_train",
    description="Re-quantize the MegaMoE fp8_block expert weights after every optimizer step",
    condition=_is_turbo_mega_moe_fp8_block,
)
def patch_mega_moe_fp8blk_weights(ctx: PatchContext):
    """Refresh the 128x128-block FP8 copies (+ transposed) of every MegaMoE expert weight after each step.

    The forward reads fixed buffers and never checks weight versions on the host (CUDA-graph capturable), so
    the refresh has to follow the optimizer step itself.
    """
    import megatron.training.training as megatron_training
    from primus_turbo.pytorch.ops.moe.fused_mega_moe_fp8blk import refresh_fp8blk_weights

    _original_train_step = megatron_training.train_step

    @functools.wraps(_original_train_step)
    def _patched_train_step(*args, **kwargs):
        result = _original_train_step(*args, **kwargs)
        refresh_fp8blk_weights()
        return result

    megatron_training.train_step = _patched_train_step
    log_rank_0(
        "[Patch:megatron.turbo.mega_moe_fp8blk_weights] "
        "Patched train_step to re-quantize the MegaMoE fp8_block expert weights after optimizer.step()"
    )

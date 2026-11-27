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
import weakref

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _is_turbo_mega_moe_can_patch(ctx: PatchContext) -> bool:
    """
    Check if the PrimusTurbo fused Mega MoE layer is enabled.

    Requires:
      - primus_turbo package is installed
      - enable_primus_turbo == True
      - use_turbo_mega_moe == True
      - 1 < expert_model_parallel_size <= 8
    """
    args = get_args(ctx)
    use_turbo_mega_moe = bool(getattr(args, "use_turbo_mega_moe", False))
    bf16 = bool(getattr(args, "bf16", False))
    # The fused all-to-all rides intra-node peer transfers, so the EP group has to fit in one node.
    ep_size = int(getattr(args, "expert_model_parallel_size", 1))

    return use_turbo_mega_moe and bf16 and 1 < ep_size <= 8 and is_primus_turbo_can_patch(ctx)


def _is_turbo_mega_moe_mxfp8(ctx: PatchContext) -> bool:
    """MegaMoE with mxfp8 experts -- the only flavour that caches quantized weights."""
    precision = getattr(get_args(ctx), "turbo_mega_moe_precision", "bf16")
    return precision == "mxfp8" and _is_turbo_mega_moe_can_patch(ctx)


def _is_turbo_mega_moe_grad_reduce_ordered(ctx: PatchContext) -> bool:
    """MegaMoE under Megatron DDP's ``overlap_grad_reduce``, unless the option turns it off."""
    args = get_args(ctx)
    if not getattr(args, "overlap_grad_reduce", False):
        return False
    if getattr(args, "use_megatron_fsdp", False) or getattr(args, "use_torch_fsdp2", False):
        return False
    ordered = getattr(args, "turbo_mega_moe_ordered_grad_reduce", None)
    if ordered is not None and not ordered:
        return False
    return _is_turbo_mega_moe_can_patch(ctx)


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


def _bucket_groups(ddp_models):
    for model in ddp_models:
        yield from model.bucket_groups
        yield from model.expert_parallel_bucket_groups


def finish_issued_grad_reduce(ddp_models) -> int:
    """Finish every reduce-scatter ``ddp_models`` have issued and not yet waited for.

    ``finish_grad_sync`` puts the wait on the current stream (it does not block the host), and a
    second call in the same iteration, the one at the end of the step, is a no-op. Megatron
    drains a bucket group's predecessor the same way. Returns the number finished.
    """
    finished = 0
    for bucket_group in _bucket_groups(ddp_models):
        if bucket_group.grad_reduce_handle is not None:
            bucket_group.finish_grad_sync()
            finished += 1
    return finished


@register_patch(
    "megatron.turbo.mega_moe_grad_reduce_order",
    backend="megatron",
    phase="before_train",
    description="Finish the issued overlapped grad reduce-scatters before each MegaMoE backward stage",
    condition=_is_turbo_mega_moe_grad_reduce_ordered,
)
def patch_mega_moe_grad_reduce_order(ctx: PatchContext):
    """Keep ``overlap_grad_reduce`` from deadlocking the MegaMoE backward on gfx1250.

    A MegaMoE backward stage launches kernels that wait for the same kernels on the other ranks.
    While they wait, each of their workgroups keeps 278528 B of its compute unit's 327680 B of
    LDS, on every compute unit, so no RCCL workgroup (85472 B) can start on that GPU. A
    reduce-scatter Megatron issued just before needs all ranks as well. The host issues the two
    in the same order on every rank, but which one the GPU starts first is up to how the runtime
    shares its hardware queues between the streams, and that differs from rank to rank and from
    step to step. When one rank starts the MegaMoE kernel first and another the reduce-scatter,
    each waits for the other rank's kernel that cannot start: the job hangs with the GPUs
    spinning (MiniMax-M3.1-Flash, EP4, 4 x MI455X, ``GPU_MAX_HW_QUEUES=2``).

    With this patch the stream waits for the reduce-scatters already issued before each stage,
    so every rank runs them first. Reduce-scatters issued later still overlap the backward. The
    two could not share the GPU anyway, so the wait costs no time.
    """
    import torch
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )

    from primus.backends.megatron.core.extensions import mega_moe
    from primus.core.utils.gpu_arch import is_gfx1250

    patch = "[Patch:megatron.turbo.mega_moe_grad_reduce_order]"
    if mega_moe.before_backward_stage is not None:
        return

    ddp_models = weakref.WeakSet()
    original_init = DistributedDataParallel.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        ddp_models.add(self)

    option = getattr(get_args(ctx), "turbo_mega_moe_ordered_grad_reduce", None)

    def is_ordered() -> bool:
        # finish_grad_sync twice in an iteration is only safe with Megatron's own early drain,
        # which keeps this flag on each bucket group
        if not all(hasattr(group, "grad_reduce_finished") for group in _bucket_groups(ddp_models)):
            warning_rank_0(
                f"{patch} off: this Megatron cannot finish a reduce-scatter early. MegaMoE with "
                "overlap_grad_reduce can hang on gfx1250: turn overlap_grad_reduce off"
            )
            return False
        if option is None and not is_gfx1250():
            log_rank_0(f"{patch} off: not a gfx1250 GPU (turbo_mega_moe_ordered_grad_reduce: null)")
            return False
        return True

    # Decided at the first backward, not here: the models are not built yet, and asking for the
    # GPU's architecture now would open the default device in every rank before it has picked
    # its own.
    state = {"ordered": None}

    def before_backward_stage():
        if state["ordered"] is None:
            state["ordered"] = is_ordered()
            if not state["ordered"]:
                mega_moe.before_backward_stage = None
        # a captured backward replays kernels only; Megatron orders its wgrads there itself
        if state["ordered"] and not torch.cuda.is_current_stream_capturing():
            finish_issued_grad_reduce(ddp_models)

    DistributedDataParallel.__init__ = __init__
    mega_moe.before_backward_stage = before_backward_stage
    log_rank_0(
        f"{patch} MegaMoE backward stages wait for the grad reduce-scatters issued before them "
        f"(turbo_mega_moe_ordered_grad_reduce: {option}; null is on for gfx1250 only)"
    )

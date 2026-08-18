###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fire the in-place MXFP4 -> BF16 switch from Megatron's train_step.

The switch itself lives in
``primus.backends.megatron.core.extensions.mxfp4_inline_switch``; this module is
only its trigger surface, wrapping ``megatron.training.training.train_step`` the
way the delayed-scaling and grad-zero patches already do.

Why the wrapper and not the Flux trainer: ``FluxPretrainTrainer`` builds the model
and the forward step but does not own the iteration loop, which is Megatron's
``train()``. ``train_step`` is the only per-iteration surface that is handed both
the model chunks and the current iteration count, which are exactly the two things
the switch needs.

The check costs one integer comparison per step before the switch, and after it a
module-level bool short-circuits immediately, so leaving this patch installed on
runs that never switch is free.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _needs_inline_switch(ctx: PatchContext) -> bool:
    """Only for runs that asked for a switch.

    Registration in this package is global -- every Megatron job, LLM or
    diffusion, imports it -- so the condition has to be precise enough that an
    unrelated job never gets its train_step wrapped.
    """
    params = get_args(ctx)
    if params is None:
        return False
    try:
        return int(getattr(params, "mxfp4_switch_iter", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _model_config(model):
    """The live TransformerConfig carrying the mxfp4_switch_* knobs.

    Read off the model rather than from Megatron ``args`` because the switch is
    configured on the diffusion config built by the Flux trainer from recipe YAML,
    which is not mirrored onto ``args``.
    """
    chunks = list(model) if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        config = getattr(chunk, "config", None)
        if config is not None:
            return config
        inner = getattr(chunk, "module", None)
        while inner is not None:
            config = getattr(inner, "config", None)
            if config is not None:
                return config
            inner = getattr(inner, "module", None)
    return None


@register_patch(
    "megatron.mxfp4.inline_switch",
    backend="megatron",
    phase="before_train",
    description="Wrap train_step to switch MXFP4 linears to BF16 at mxfp4_switch_iter.",
    priority=42,
    condition=_needs_inline_switch,
)
def patch_mxfp4_inline_switch(ctx: PatchContext):
    import megatron.training.training as megatron_training

    from primus.backends.megatron.core.extensions.mxfp4_inline_switch import (
        apply_switch_if_due,
    )
    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    _PATCH_KEY = "megatron.mxfp4.inline_switch"
    if is_patched(megatron_training, _PATCH_KEY):
        log_rank_0("[Patch:mxfp4_inline_switch] Already applied; skipping re-wrap.")
        return

    _original_train_step = megatron_training.train_step

    def _patched_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        # The switch has to land before the step runs, so that the iteration that
        # first sees BF16 is the one after the configured count of MXFP4 steps.
        if iteration is not None:
            model_config = _model_config(model)
            if model_config is not None:
                apply_switch_if_due(model, model_config, iteration)

        return _original_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
            iteration=iteration,
        )

    megatron_training.train_step = _patched_train_step
    mark_patched(megatron_training, _PATCH_KEY)
    log_rank_0(
        "[Patch:mxfp4_inline_switch] Wrapped train_step; the switch fires when "
        "iteration reaches mxfp4_switch_iter (0 = never)."
    )

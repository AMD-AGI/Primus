###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling: check the deferred padded-label counts after each forward-backward pass.

``InklingModel`` refuses unmasked labels past the HF vocab. Inside the schedule this patch wraps,
it does so without a host sync per micro-batch: it queues the count (``_PaddedLabelCheck``, CPU or
GPU; outside the wrapped schedule a forward checks at once, on its own rank). This patch wraps the
schedule that Megatron's ``get_forward_backward_func`` returns (``train_step`` and ``evaluate``
both run it), so that after the forward-backward pass of every step, on every rank,
``flush_padded_label_checks`` sums the queued counts over all ranks and raises if any is bad.
That is before ``train_step`` runs ``optimizer.step()``: the bad step's update is never applied,
and on a padded-label failure every rank raises together (the count is all-reduced, so no
rank is left waiting in the collective). If the pass itself raises on one rank only (an OOM, a
data error), that rank drops its queued counts and re-raises; the other ranks finish the pass
and wait in the flush's all-reduce until the collective times out, as they would wait in the
pass's own collectives (pipeline send / receive, the gradient all-reduce): a one-rank failure
in a step is a job failure either way. No collective is added for it.
"""

import functools

from primus.core.patches import PatchContext, get_args, register_patch

_PATCH_KEY = "megatron.inkling.padded_label_check"


def _flushing(forward_backward_func, unpadded_vocab_size):
    from primus.backends.megatron.core.models.inkling import inkling_model

    @functools.wraps(forward_backward_func)
    def wrapped(*args, **kwargs):
        try:
            with inkling_model.deferring_padded_label_checks():
                out = forward_backward_func(*args, **kwargs)
        except BaseException:
            inkling_model.discard_padded_label_checks()  # host-side only: cannot hide the error
            raise
        inkling_model.flush_padded_label_checks(unpadded_vocab_size)
        return out

    return wrapped


def _wrap_getter(get_forward_backward_func, unpadded_vocab_size):
    @functools.wraps(get_forward_backward_func)
    def wrapped(*args, **kwargs):
        return _flushing(get_forward_backward_func(*args, **kwargs), unpadded_vocab_size)

    return wrapped


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    priority=101,  # after the schedule / zero-bubble / pp-dump patches: wrap the getter they install
    description="Inkling: check the deferred padded-label counts after each forward-backward pass.",
    condition=lambda ctx: getattr(get_args(ctx), "model_type", None) == "inkling",
)
def patch_inkling_padded_label_check(ctx: PatchContext):
    import megatron.core.pipeline_parallel as pp_module
    import megatron.training.training as training_module

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    # From the run's config, the same on every rank: it alone decides whether the flush runs its
    # collective (never which models happen to be alive on a rank).
    unpadded_vocab_size = getattr(get_args(ctx), "inkling_unpadded_vocab_size", None)
    # Both: training.py calls its own name, and the trainer re-wraps the pipeline_parallel one
    # into training.py when train() starts.
    for module in (pp_module, training_module):
        if not is_patched(module, _PATCH_KEY):
            module.get_forward_backward_func = _wrap_getter(
                module.get_forward_backward_func, unpadded_vocab_size
            )
            mark_patched(module, _PATCH_KEY)

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``model_type: mistral_large4``): when the first iteration of a run is also a
log step (``log_interval`` 1, or a resume at ``k * log_interval - 1``), reset the loss accumulators
and the ``interval-time`` timer after ``training_log``, as a normal log step does; otherwise the
next log line averages two iterations. Limited to this model so that other models' loss baselines
are unchanged.
"""

import functools
import inspect

import torch

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    is_mistral_large4_args,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# set on the training_log wrapper this patch installs (tells an already wrapped training_log)
_WRAPPED_MARK = "_primus_mistral_large4_first_log"
_COUNTERS = ("advanced iterations", "skipped iterations", "nan iterations")
# the parameters of Megatron's training_log the wrapper reads (bound by name)
REQUIRED_PARAMS = ("total_loss_dict", "iteration", "is_first_iteration")


def missing_params(fn) -> list:
    """The ``REQUIRED_PARAMS`` that ``fn``'s signature does not have (empty when it can be wrapped)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return list(REQUIRED_PARAMS)
    return [p for p in REQUIRED_PARAMS if p not in params]


def reset_loss_accumulators(total_loss_dict) -> None:
    """What Megatron's log step does to ``total_loss_dict`` when it resets: counters to 0, every
    other accumulator to 0 (a tensor keeps its shape, dtype and device)."""
    for key in list(total_loss_dict):
        value = total_loss_dict[key]
        if key in _COUNTERS:
            total_loss_dict[key] = 0
        elif torch.is_tensor(value):
            total_loss_dict[key] = torch.zeros_like(value)
        else:
            total_loss_dict[key] = 0.0


def make_training_log(original, log_interval_fn, reset_interval_timer=None):
    """``training_log`` that, on a first iteration that is a log step, also does what a normal log
    step does after logging: clears the loss accumulators and iteration counters, and resets
    Megatron's ``interval-time`` timer (``reset_interval_timer``). Both go together, so the next
    log line divides the time of its own iterations by their count (throughput, TFLOP/s and
    tokens/s stay right). ``is_first_iteration`` is forwarded to ``original`` unchanged.

    ``log_interval_fn()`` returns Megatron's ``args.log_interval`` (read at call time). The
    arguments are bound by name to ``original``'s signature (no positional index)."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def training_log(*args, **kwargs):
        out = original(*args, **kwargs)
        try:
            bound = signature.bind_partial(*args, **kwargs).arguments
        except TypeError:
            return out  # an unexpected call: Megatron's own behaviour is kept
        iteration = bound.get("iteration")
        total_loss_dict = bound.get("total_loss_dict")
        if (
            bound.get("is_first_iteration", False)
            and isinstance(iteration, int)
            and isinstance(total_loss_dict, dict)
            and iteration % int(log_interval_fn() or 1) == 0
        ):
            reset_loss_accumulators(total_loss_dict)
            if reset_interval_timer is not None:
                reset_interval_timer()
        return out

    setattr(training_log, _WRAPPED_MARK, True)
    return training_log


def _is_this_model(ctx: PatchContext) -> bool:
    """This model's run (``args.model_type``, ``is_mistral_large4_args``)."""
    return is_mistral_large4_args(get_args(ctx))


@register_patch(
    "megatron.mistral_large4.first_iteration_log",
    backend="megatron",
    phase="before_train",
    # before every other training_log wrapper (mlperf_logging 15, training_log.unified_patch 50):
    # those take (*args, **kwargs) without functools.wraps, so this wrapper binds its arguments by
    # name only on Megatron's own training_log; installed first, it is the innermost wrapper
    priority=10,
    description="Mistral Large 4: reset the loss accumulators on a first iteration that is a log step "
    "(otherwise, e.g. with log_interval 1 or after a resume, the next log line prints a two-iteration "
    "average)",
    condition=_is_this_model,
)
def patch_mistral_large4_first_iteration_log(ctx: PatchContext):
    del ctx
    import megatron.training.training as megatron_training
    from megatron.training.global_vars import get_args as get_megatron_args

    if getattr(megatron_training.training_log, _WRAPPED_MARK, False):
        return
    missing = missing_params(megatron_training.training_log)
    if missing:
        # another wrapper or a Megatron with a different signature: the wrapper could not see the
        # arguments it needs, so it is not installed (instead of installing a silent no-op)
        log_rank_0(
            "[Patch:megatron.mistral_large4.first_iteration_log] WARNING: training_log has no "
            f"parameter(s) {missing}; not wrapped (after a resume, the second log line may average two iterations)"
        )
        return

    def reset_interval_timer():
        # elapsed(barrier=True, reset=True) clears the time accumulated so far and keeps the timer
        # running, with the barrier Megatron's log step uses (training_log:
        # timers('interval-time').elapsed(barrier=True, reset=should_reset)), so every rank resets at
        # the same point. It runs right after training_log returns (the first iteration's log
        # step), not inside it: the next interval also counts the few host operations between the
        # two, as each later interval does after Megatron's own reset.
        from megatron.training.global_vars import get_timers

        get_timers()("interval-time").elapsed(barrier=True, reset=True)

    megatron_training.training_log = make_training_log(
        megatron_training.training_log, lambda: get_megatron_args().log_interval, reset_interval_timer
    )
    log_rank_0("[Patch:megatron.mistral_large4.first_iteration_log] wrapped training_log")

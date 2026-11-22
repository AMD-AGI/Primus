###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 (``model_type: mistral_large4``): when the first iteration of a run is also a
log step (``log_interval`` 1, or a resume at ``k * log_interval - 1``), reset the loss accumulators
and the timers Megatron's log step resets (``interval-time`` and the logged per-phase timers) after
``training_log``, as a normal log step does; otherwise the next log line averages two iterations. Limited to this model (ADR 0030: behaviour only this port
changes stays a patch of this model) so that other models' loss baselines are unchanged. As a
result, when the first iteration is a log step, this model's second log line (loss average and
throughput) covers only its own iterations, while other Primus models on the same version include
the first iteration in it: those first-interval numbers are not comparable across models. Later
log lines are the same for every model.
"""

import functools
import inspect

import torch

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    is_mistral_large4_args,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

# attribute set on the training_log wrapper this patch installs, so a second application detects
# it and does not wrap again
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


# the timers Megatron's training_log logs (and resets on a normal log step): its ``timers_to_log``
# per ``timing_log_level`` (training.py, training_log)
LOGGED_TIMERS_LEVEL_1 = (
    "forward-backward",
    "layernorm-grads-all-reduce",
    "embedding-grads-all-reduce",
    "all-grads-sync",
    "params-all-gather",
    "optimizer-copy-to-main-grad",
    "optimizer-unscale-and-check-inf",
    "optimizer-clip-main-grad",
    "optimizer-count-zeros",
    "optimizer-inner-step",
    "optimizer-copy-main-to-model-params",
    "optimizer",
)
LOGGED_TIMERS_LEVEL_2 = (
    "batch-generator",
    "forward-compute",
    "backward-compute",
    "forward-recv",
    "forward-send",
    "backward-recv",
    "backward-send",
    "forward-send-forward-recv",
    "forward-send-backward-recv",
    "backward-send-forward-recv",
    "backward-send-backward-recv",
    "forward-backward-send-forward-backward-recv",
)


def logged_timer_names(timing_log_level, extra=()) -> list:
    """The names Megatron's training_log passes to ``timers.log(..., reset=should_reset)``:
    level 1 and level 2 lists as ``timing_log_level`` selects, plus ``extra`` (the RL timers when
    ``perform_rl_step``)."""
    level = int(timing_log_level or 0)
    names = list(LOGGED_TIMERS_LEVEL_1) if level >= 1 else []
    if level >= 2:
        names += LOGGED_TIMERS_LEVEL_2
    return names + list(extra)


def reset_megatron_timers(timers, names) -> list:
    """Do what Megatron's log step does to its timers when it resets, and nothing more:
    ``interval-time`` with the barrier (``timers('interval-time').elapsed(barrier=True,
    reset=True)``), then the logged timers ``names`` (``timers.log(names, reset=True)`` calls
    ``elapsed(reset=True)`` on each one that exists). A running timer keeps running. Other timers
    (startup, checkpoint, evaluate) are not touched. Returns the names reset."""
    timers("interval-time").elapsed(barrier=True, reset=True)
    done = ["interval-time"]
    existing = getattr(timers, "_timers", None)
    for name in names:
        # Megatron's own timers.log skips a name it has no timer for (``if name in self._timers``);
        # calling timers(name) would create one
        if existing is None or name in existing:
            timers(name).elapsed(reset=True)
            done.append(name)
    return done


def make_training_log(original, log_interval_fn, reset_timers=None):
    """``training_log`` that, on a first iteration that is a log step, also does what a normal log
    step does after logging and Megatron skips on the first iteration (``should_reset = not
    is_first_iteration``): clears the loss accumulators and iteration counters, and resets
    the timers Megatron's log step resets (``reset_timers``: ``interval-time`` and the logged
    per-phase timers such as forward-backward and optimizer, ``reset_megatron_timers``). Both go together, so the next
    log line divides the time of its own iterations by their count (throughput, TFLOP/s,
    tokens/s and the per-phase timings cover the same iterations as the loss). ``is_first_iteration`` is forwarded to ``original`` unchanged.

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
            if reset_timers is not None:
                reset_timers()
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
            f"parameter(s) {missing}; not wrapped (with log_interval 1 or after a resume, the next log line may average two iterations)"
        )
        return

    def reset_timers():
        # runs right after training_log returns (the first iteration's log step), not inside it:
        # the next interval also counts the few host operations between the two, as each later
        # interval does after Megatron's own reset
        from megatron.training.global_vars import get_timers

        args = get_megatron_args()
        extra = ()
        if getattr(args, "perform_rl_step", False):
            extra = tuple(getattr(megatron_training, "RL_LOGGABLE_TIMER_NAMES", ()) or ())
        reset_megatron_timers(get_timers(), logged_timer_names(getattr(args, "timing_log_level", 0), extra))

    megatron_training.training_log = make_training_log(
        megatron_training.training_log, lambda: get_megatron_args().log_interval, reset_timers
    )
    log_rank_0("[Patch:megatron.mistral_large4.first_iteration_log] wrapped training_log")

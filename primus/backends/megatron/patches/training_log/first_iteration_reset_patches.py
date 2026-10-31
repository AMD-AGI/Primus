###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in (``reset_log_on_first_iteration``): log the first iteration of a run, or of a
resumed run, like any other ``log_interval`` boundary when it falls on one.

Upstream ``training_log`` does not reset the loss sums and timers after the first
iteration (``should_reset = not is_first_iteration``), so with ``log_interval: 1`` the
second printed loss is the mean of two iterations, and a resumed run prints different
losses than an uninterrupted one. On a boundary this wrapper calls the original with
``is_first_iteration=False``: Megatron's own reset code then runs, so its real timer
list and barriers are used. The one other effect: that iteration's iteration-time and
throughput (which include start-up time) are also written to TensorBoard/W&B.
"""

import functools
import inspect

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.flag_utils import as_bool
from primus.core.utils.module_utils import log_rank_0

_LOG = "[Patch:megatron.training_log.first_iteration_reset]"
_KEY = "megatron.training_log.first_iteration_reset"
_FLAG = "reset_log_on_first_iteration"


def reset_log_on_first_iteration_enabled(ctx: PatchContext) -> bool:
    """The recipe flag (default false); a value that is not a boolean raises ValueError."""
    return as_bool(get_param(ctx, _FLAG, False), _FLAG)


def reset_log_on_first_iteration_condition(ctx: PatchContext) -> bool:
    """Patch condition, which must not raise (``run_patches`` evaluates every condition
    unguarded): a value that is not a boolean returns True, so the patch runs and stops
    the run with a clear error."""
    try:
        return as_bool(get_param(ctx, _FLAG, False), _FLAG)
    except ValueError:
        return True


def _megatron_log_interval():
    from megatron.training import get_args

    return get_args().log_interval


def wrap(original, log_interval=None):
    """``original`` (Megatron's ``training_log``) with a first iteration on a
    ``log_interval()`` boundary (default: Megatron's final args, read at call time) logged
    as any boundary. Arguments are bound by name; the signature is checked at install."""
    sig = inspect.signature(original)
    if "iteration" not in sig.parameters or "is_first_iteration" not in sig.parameters:
        raise RuntimeError(f"{_LOG} training_log{sig} has no iteration / is_first_iteration parameter")

    @functools.wraps(original)
    def training_log(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        if bound.arguments.get("is_first_iteration"):
            iteration, interval = bound.arguments["iteration"], (log_interval or _megatron_log_interval)()
            if interval and iteration % interval == 0:
                log_rank_0(
                    f"{_LOG} first iteration {iteration} is on a log_interval boundary: reset after it"
                )
                bound.arguments["is_first_iteration"] = False
                return original(*bound.args, **bound.kwargs)
        return original(*args, **kwargs)

    return training_log


def install():
    """Wrap ``megatron.training.training.training_log`` once. Returns True if wrapped now."""
    import megatron.training.training as megatron_training

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    if is_patched(megatron_training, _KEY):
        return False
    megatron_training.training_log = wrap(megatron_training.training_log)
    mark_patched(megatron_training, _KEY)
    return True


@register_patch(
    _KEY,
    backend="megatron",
    phase="before_train",
    description="Opt-in: a first iteration on a log_interval boundary resets the log like any boundary",
    condition=reset_log_on_first_iteration_condition,
    # Before megatron.training.mlperf_logging (15), which wraps training_log again.
    priority=12,
)
def patch_first_iteration_reset(ctx: PatchContext):
    """Any error stops the run (``SystemExit``; ``run_patches`` logs and skips an Exception)."""
    try:
        reset_log_on_first_iteration_enabled(ctx)  # a value that is not a boolean stops here
        wrapped = install()
    except Exception as exc:
        raise SystemExit(f"{_LOG} {_FLAG} could not be applied: {exc}") from exc
    if wrapped:
        log_rank_0(f"{_LOG} wrapped megatron.training.training.training_log")

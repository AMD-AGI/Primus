###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in (``log_first_iteration_ends_interval: true``): the first logged iteration of a run ends
its log interval.

Upstream ``training_log`` logs the first iteration of a run (or of a resumed run) but does not
reset its interval state (``should_reset = not is_first_iteration``): the loss sums, the iteration
counters, the ``interval-time`` timer and the per-component timers (``timers.log(...,
reset=should_reset)``) all carry that iteration into the next interval. With ``log_interval: 1``
the second logged ``lm loss`` is then the mean of two iterations, and after a resume it does not
match the same iteration of an uninterrupted run.

When the first iteration is on the log schedule (``iteration % log_interval == 0``), this patch
calls upstream ``training_log`` with ``is_first_iteration=False``, so upstream itself logs it as a
scheduled iteration and resets everything it resets after every logged interval. Upstream also
uses the flag to keep the first iteration's ``iteration-time`` (it includes start-up overhead) out
of TensorBoard / W&B; the patch keeps that: the module's writer getters are wrapped once, at
install, and drop the ``iteration-time`` scalar only on the thread inside this one call (other
threads and other times get the original writers). A first iteration off the schedule is passed unchanged.
Training is not changed. The option is off by default, and ignored with ``mlperf_mode`` (the MLPerf
path replaces ``training_log``).
"""

import contextlib
import functools
import inspect
import threading

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

__all__ = ["patch_first_iteration_log_interval", "wrap_training_log"]

_MARK = "_primus_first_iteration_ends_interval"
# the scalar upstream keeps out of TensorBoard / W&B on the first iteration (start-up overhead)
_ITERATION_TIME_TAG = "iteration-time"


class _TensorboardWithout:
    """A TensorBoard writer that drops one scalar tag."""

    def __init__(self, writer, tag):
        self._writer, self._tag = writer, tag

    def add_scalar(self, tag, *args, **kwargs):
        if tag != self._tag:
            return self._writer.add_scalar(tag, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._writer, name)


class _WandbWithout:
    """A W&B writer that drops one key from ``log`` dicts."""

    def __init__(self, writer, key):
        self._writer, self._key = writer, key

    def log(self, data, *args, **kwargs):
        if isinstance(data, dict) and self._key in data:
            data = {k: v for k, v in data.items() if k != self._key}
            if not data:
                return None
        return self._writer.log(data, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._writer, name)


class _FirstIterationWriters:
    """Writer getters for ``training_log``'s module that drop the ``iteration-time`` scalar only on
    the thread that is inside a first-iteration call of the wrapper, and only during it. Installed
    once on the module (not swapped per call) for the rest of the process: every other caller (other
    threads such as async save or the energy monitor, and every call outside that one) gets exactly
    the writer the original getter returns. Nested calls are counted, so it is reentrant."""

    def __init__(self, get_tensorboard_writer, get_wandb_writer):
        self._get_tb, self._get_wb = get_tensorboard_writer, get_wandb_writer
        self._local = threading.local()

    @contextlib.contextmanager
    def filtering(self):
        self._local.depth = getattr(self._local, "depth", 0) + 1
        try:
            yield
        finally:
            self._local.depth -= 1

    def _active(self):
        return getattr(self._local, "depth", 0) > 0

    def get_tensorboard_writer(self):
        tb = self._get_tb()
        return _TensorboardWithout(tb, _ITERATION_TIME_TAG) if tb is not None and self._active() else tb

    def get_wandb_writer(self):
        wb = self._get_wb()
        return _WandbWithout(wb, _ITERATION_TIME_TAG) if wb is not None and self._active() else wb


def _training_log_signature(original, reference=None):
    """The signature to bind ``training_log`` calls with: ``original``'s own, else (a pass-through
    wrapper such as ``primus_training_log(*args, **kwargs)``) that of the function it wraps
    (``__wrapped__`` chain), else that of ``reference`` (upstream's ``training_log``)."""
    for fn in (original, inspect.unwrap(original), reference):
        if fn is None:
            continue
        sig = inspect.signature(fn, follow_wrapped=False)
        if "is_first_iteration" in sig.parameters and "iteration" in sig.parameters:
            return sig
    raise TypeError(
        "log_first_iteration_ends_interval: training_log has no iteration / is_first_iteration "
        f"parameter (signature {inspect.signature(original, follow_wrapped=False)}); this Megatron "
        "version is not supported, or a *args/**kwargs wrapper without __wrapped__ hides it"
    )


def wrap_training_log(original, get_log_interval, writers=None, reference=None):
    """``training_log`` that passes ``is_first_iteration=False`` to ``original`` when the first
        iteration's number is a multiple of ``get_log_interval()``; during that call ``writers``
        (a :class:`_FirstIterationWriters` installed as the module's writer getters) drops the
        ``iteration-time`` scalar on this thread. Arguments are bound with upstream's signature
        (``original``'s, or of the function a ``*args, **kwargs`` wrapper of it forwards to, or
        ``reference``). It must be installed before a ``*args, **kwargs`` wrapper without ``__wrapped__``
    (such as unified_patch's): it then finds no signature and raises ``TypeError``. An already wrapped
        ``original`` is returned unchanged."""
    if getattr(original, _MARK, False):
        return original
    signature = _training_log_signature(original, reference)  # raises TypeError if none fits

    @functools.wraps(original)
    def training_log(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        first = bound.arguments.get("is_first_iteration", False)
        iteration = bound.arguments.get("iteration")
        if not (first and iteration is not None and iteration % get_log_interval() == 0):
            return original(*args, **kwargs)
        bound.arguments["is_first_iteration"] = False
        if writers is None:
            return original(*bound.args, **bound.kwargs)
        with writers.filtering():
            return original(*bound.args, **bound.kwargs)

    setattr(training_log, _MARK, True)
    return training_log


def _switch(value, name: str) -> bool:
    """A boolean option as YAML or a CLI string gives it: ``true`` / ``false`` (any case), 1 / 0;
    None or an empty string is off. Anything else raises, so a string such as ``"false"`` is never
    read as on."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "false", "0", ""):
            return text in ("true", "1")
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _enabled(ctx: PatchContext) -> bool:
    # MLPerf logging replaces training_log with its own interval log
    on = _switch(
        get_param(ctx, "log_first_iteration_ends_interval", False), "log_first_iteration_ends_interval"
    )
    # off whenever the MLPerf logging patch installs (mlperf_logging_patches._mlperf_logging_enabled
    # tests plain truthiness, so the same test is used here: the two never wrap training_log together)
    return on and not get_param(ctx, "mlperf_mode", False)


def _reads_writer_getters(fn, module) -> bool:
    """True if ``fn`` (upstream ``training_log``) looks the writer getters up as globals of
    ``module`` at call time, the only way the installed getters reach it. Checked at install, so a
    Megatron that changes this gets no writer wrapping (and a warning) instead of a silent no-op."""
    code, glb = getattr(fn, "__code__", None), getattr(fn, "__globals__", None)
    return (
        code is not None
        and glb is vars(module)
        and {"get_tensorboard_writer", "get_wandb_writer"} <= set(code.co_names)
    )


def _upstream_training_log():
    """Megatron's own ``training_log`` (the function defined in megatron.training.training), found
    through the module attribute and its ``__wrapped__`` chain; None when a wrapper without
    ``__wrapped__`` (e.g. unified_patch's ``primus_training_log``) hides it."""
    import megatron.training.training as training_module

    fn = training_module.training_log
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        if getattr(fn, "__module__", None) == training_module.__name__ and fn.__name__ == "training_log":
            return fn
        fn = getattr(fn, "__wrapped__", None)
    return None


@register_patch(
    "megatron.training_log.first_iteration_ends_interval",
    backend="megatron",
    phase="before_train",
    description="Opt-in: a first iteration on the log schedule ends its log interval.",
    condition=_enabled,
    # runs before megatron.training_log.unified_patch (priority 50), whose *args/**kwargs wrapper has
    # no __wrapped__; if it ever ran after it, upstream's signature is not found and the option is
    # skipped with a warning (training is not stopped)
    priority=40,
)
def patch_first_iteration_log_interval(ctx: PatchContext):
    import megatron.training.training as training_module
    from megatron.training import get_args as get_megatron_args

    if getattr(training_module.training_log, _MARK, False):
        return  # already installed (a second apply in the same process)
    reference = _upstream_training_log()
    try:
        _training_log_signature(training_module.training_log, reference)
    except TypeError:
        # e.g. installed after a *args/**kwargs wrapper without __wrapped__ (unified_patch's): the
        # option cannot be applied, but it is a logging option, so training goes on unchanged
        warning_rank_0(
            "[Patch:megatron.training_log.first_iteration_ends_interval] upstream training_log not "
            "found (hidden by a wrapper without __wrapped__, or another Megatron signature); "
            "log_first_iteration_ends_interval is skipped and logging stays upstream's"
        )
        return
    writers = None
    if _reads_writer_getters(reference, training_module):
        writers = _FirstIterationWriters(
            training_module.get_tensorboard_writer, training_module.get_wandb_writer
        )
    elif reference is not None:
        warning_rank_0(
            "[Patch:megatron.training_log.first_iteration_ends_interval] upstream training_log does "
            "not read get_tensorboard_writer / get_wandb_writer from its module; the first "
            "iteration's iteration-time may be written to TensorBoard / W&B"
        )
    # everything is built first; the module attributes are then set together (and restored if
    # setting one fails), so a failed install leaves Megatron unchanged
    new = {
        "training_log": wrap_training_log(
            training_module.training_log,
            lambda: get_megatron_args().log_interval,
            writers=writers,
            reference=reference,
        )
    }
    if writers is not None:
        new["get_tensorboard_writer"] = writers.get_tensorboard_writer
        new["get_wandb_writer"] = writers.get_wandb_writer
    old = {name: getattr(training_module, name) for name in new}
    try:
        for name, value in new.items():
            setattr(training_module, name, value)
    except Exception:
        for name, value in old.items():
            setattr(training_module, name, value)
        raise
    log_rank_0(
        "[Patch:megatron.training_log.first_iteration_ends_interval] a first iteration on the log "
        "schedule ends its log interval"
    )

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-step logging without host syncs (``--light-log-sync-events``, on top of ``--light-log-sync``).

``--light-log-sync`` keeps the logging off the NCCL queue, but a logged step still stops the host three ways:

* ``timers('interval-time').elapsed(barrier=True)`` stops and restarts the timer, each with a (gloo) barrier and a
  compute-stream sync;
* ``training_log`` reads the accumulated loss with ``.item()``, and Primus' log line reads the step's average
  timestep the same way; on the last rank the TensorBoard writer converts the loss every step;
* every value it prints was produced by the step that has just been launched, so any read waits for that step's
  optimizer to finish.

The host then cannot launch the next step's data preparation or forward until the GPU has drained, and the GPU idles
at the step boundary for as long as the host takes to log.

This patch defers each ``training_log`` call by one step. At the call it only enqueues non-blocking device-to-host
copies of the tensors the call will read (losses, the gradient norm, the step metrics) and records a CUDA event. The
real ``training_log`` runs at the next call -- or before the interval timer is stopped (evaluation, checkpointing)
and when ``train()`` returns -- with host tensors in place of device ones. By then the event has long completed, so
nothing waits:

* the loss accumulators are created on the host (``light_log_sync``'s ``torch.tensor`` proxy), so the sums and the
  ``.item()`` reads are host operations; fp32 adds in the same order give the same values;
* ``consumed_train_samples`` / ``skipped_train_samples`` and the runtime metrics are restored to their values at the
  original call for its duration;
* ``interval-time`` is read without a barrier, as the GPU time between the events of successive logged steps (a
  rank-local step time; the timer is still started and stopped around evaluation and checkpoints as before).

Every value in the log line, TensorBoard and the MLPerf log is the one the synchronous path produces; the line is
printed one step later. Off by default; requires ``light_log_sync: true``.
"""

import time

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_INTERVAL_TIMER = "interval-time"


def _enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    if not getattr(args, "light_log_sync_events", False):
        return False
    if not getattr(args, "light_log_sync", False):
        raise ValueError("light_log_sync_events requires light_log_sync: true")
    return True


def _to_host(value):
    """Start a non-blocking copy of a CUDA tensor to (pinned) host memory; other values pass through."""
    if isinstance(value, torch.Tensor) and value.is_cuda:
        return value.detach().to("cpu", non_blocking=True)
    return value


class _Entry:
    __slots__ = ("bound", "event", "host_time", "consumed", "skipped", "metrics", "grad_norm_sq")


class _DeferredTrainingLog:
    def __init__(self, inner, args, runtime_state, get_timers):
        self.inner = inner
        self.args = args
        self.runtime_state = runtime_state
        self.timers = get_timers
        self.pending = None
        self.report_memory_flag = None
        # Start of the current measured interval: an event and a host time, set when the interval timer (re)starts
        # and advanced each time the deferred training_log reads it.
        self.prev_event = None
        self.prev_host_time = time.time()
        # Elapsed time carried over a non-resetting read (Megatron's first-iteration log keeps accumulating).
        self.carry = 0.0

    # -- capture ---------------------------------------------------------------------------------------------------

    def capture(self, bound):
        from primus.backends.megatron.core.optimizer.device_grad_clip import (
            DeviceGradNorm,
        )

        e = _Entry()
        a = bound.arguments
        a["loss_dict"] = {k: _to_host(v) for k, v in a["loss_dict"].items()}
        e.grad_norm_sq = None
        gn = a.get("grad_norm")
        if isinstance(gn, DeviceGradNorm):
            e.grad_norm_sq = (gn, _to_host(gn.sq))
        for key in ("params_norm", "num_zeros_in_grad", "max_attention_logit"):
            a[key] = _to_host(a.get(key))
        e.metrics = None
        rs = self.runtime_state
        if rs is not None and getattr(rs, "last_metrics", None) is not None:
            e.metrics = {k: _to_host(v) for k, v in rs.last_metrics.items()}
        e.consumed = self.args.consumed_train_samples
        e.skipped = self.args.skipped_train_samples
        e.event = torch.cuda.Event(enable_timing=True)
        e.event.record()
        e.host_time = time.time()
        e.bound = bound
        if self.report_memory_flag is None:
            self.report_memory_flag = a["report_memory_flag"]
        return e

    # -- deferred run ----------------------------------------------------------------------------------------------

    def _interval_elapsed(self, e):
        def elapsed(reset=True, barrier=False):
            if self.prev_event is not None:
                dt = self.prev_event.elapsed_time(e.event) / 1000.0
            else:
                dt = e.host_time - self.prev_host_time
            secs = self.carry + dt
            self.carry = 0.0 if reset else secs
            self.prev_event, self.prev_host_time = e.event, e.host_time
            return secs

        return elapsed

    def run(self, e):
        from primus.backends.megatron.patches import light_log_sync_patches as L

        e.event.synchronize()  # the copies above; long complete by now
        a = e.bound.arguments
        if e.grad_norm_sq is not None:
            gn, sq_host = e.grad_norm_sq
            gn.set_host_squared(sq_host.item())
            a["grad_norm"] = float(gn)
        a["report_memory_flag"] = self.report_memory_flag

        args, rs = self.args, self.runtime_state
        timer = self.timers()(_INTERVAL_TIMER)
        saved = (args.consumed_train_samples, args.skipped_train_samples)
        saved_metrics = rs.last_metrics if (rs is not None and e.metrics is not None) else None
        args.consumed_train_samples, args.skipped_train_samples = e.consumed, e.skipped
        if saved_metrics is not None:
            rs.last_metrics = e.metrics
        timer.elapsed = self._interval_elapsed(e)
        L.HOST_ZEROS[0] = True
        try:
            self.report_memory_flag = self.inner(*e.bound.args, **e.bound.kwargs)
        finally:
            L.HOST_ZEROS[0] = False
            del timer.elapsed
            args.consumed_train_samples, args.skipped_train_samples = saved
            if saved_metrics is not None:
                rs.last_metrics = saved_metrics

    def flush(self):
        e, self.pending = self.pending, None
        if e is not None:
            self.run(e)

    def __call__(self, bound):
        e = self.capture(bound)
        self.flush()
        self.pending = e
        return self.report_memory_flag

    # -- interval timer pauses (evaluation, checkpointing) ---------------------------------------------------------

    def on_timer_start(self):
        self.prev_event = torch.cuda.Event(enable_timing=True)
        self.prev_event.record()
        self.prev_host_time = time.time()

    def on_timer_stop(self):
        self.flush()
        self.prev_event = None


@register_patch(
    "megatron.training.light_log_sync_events",
    backend="megatron",
    phase="before_train",
    description="Defer training_log by one step and read every logged value back without a host sync.",
    # Outermost training_log wrapper: after MLPerf logging (15), the Primus log line (50), device_grad_clip (60).
    priority=96,
    condition=_enabled,
)
def patch_light_log_sync_events(ctx: PatchContext) -> None:
    import megatron.training.training as MT
    from megatron.core import timers as T
    from megatron.training.global_vars import get_timers

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
    from primus.backends.megatron.patches.training_log.training_log_args import (
        bind_training_log_args,
    )

    key = "megatron.training.light_log_sync_events"
    if is_patched(MT, key):
        return

    args = get_args(ctx)
    deferred = _DeferredTrainingLog(MT.training_log, args, ctx.extra.get("runtime_state"), get_timers)

    def training_log(*a, **k):
        bound = bind_training_log_args(a, k)
        if bound is None:
            deferred.flush()
            return deferred.inner(*a, **k)
        return deferred(bound)

    MT.training_log = training_log

    timer_start, timer_stop = T.Timer.start, T.Timer.stop

    def start(self, barrier=False):
        timer_start(self, barrier=barrier)
        if self.name == _INTERVAL_TIMER:
            deferred.on_timer_start()

    def stop(self, barrier=False):
        if self.name == _INTERVAL_TIMER:
            deferred.on_timer_stop()
        timer_stop(self, barrier=barrier)

    T.Timer.start, T.Timer.stop = start, stop

    orig_train = MT.train

    def train(*a, **k):
        result = orig_train(*a, **k)
        deferred.flush()
        return result

    MT.train = train
    mark_patched(MT, key)
    log_rank_0(
        "[Patch:megatron.training.light_log_sync_events] training_log deferred by one step; values read back "
        "asynchronously, interval time from CUDA events"
    )

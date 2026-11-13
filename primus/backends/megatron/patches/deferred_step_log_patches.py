###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deferred step-end log: the host enqueues step N+1's forward before it waits for step N.

``PRIMUS_DEFERRED_STEP_LOG=1`` (off by default; the MiniMax-M3 launcher turns it on).

Megatron's step is sync-free up to the optimizer when the loss scale is on the CPU and the
global norm and clip stay on the device (``PRIMUS_FUSED_ADAM_CLIP`` / ``PRIMUS_STATIC_BF16_LOSS_SCALE_CPU``).
The only host waits left between step N's optimizer launch and step N+1's forward are in
``training_log(N)``: the interval timer's ``torch.cuda.synchronize()`` and the ``.item()`` reads
of the loss, the grad norm and the MoE/indexer metric tracker. While the host waits there, the GPU runs
step N's Adam (about 17 ms on M3.1-Flash), and then the next forward starts host-bound: the host
needs about 14 ms to launch it, against about 13.5 ms of GPU work.

With this patch, ``training_log(N)`` records an event on the compute stream and returns. It runs
when step N+1's optimizer step starts, i.e. after that step's forward and backward are enqueued
(the backward is GPU-bound, so the host is several ms ahead of the GPU there):

* the interval timer reads the time at which step N's GPU work completed: the last exact host
  anchor plus the CUDA-event time between that anchor and step N's end event;
* it runs on a side stream that waits only on that event, so its ``.item()`` reads and small
  kernels do not queue behind step N+1's forward;
* the MoE metric tracker is swapped at step end, so step N+1's forward records into a fresh
  tracker and step N's values are reported unchanged.

The compute stream's kernels, their order and their arguments are unchanged, so the numerics are
identical. The last iteration and the first logged iteration run
synchronously as before; a memory report
reads the allocator counters as they were at step N's end. The pending log is flushed when ``train`` returns or raises.
"""

from __future__ import annotations

import functools
import inspect
import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


_MEM_COUNTERS = ("memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved")


def _enabled(ctx: PatchContext) -> bool:
    return os.environ.get("PRIMUS_DEFERRED_STEP_LOG", "0") == "1"


@register_patch(
    "megatron.training.deferred_step_log",
    backend="megatron",
    phase="before_train",
    description="Run training_log(N) after step N+1's forward is enqueued (PRIMUS_DEFERRED_STEP_LOG=1).",
    condition=_enabled,
)
def patch_deferred_step_log(ctx: PatchContext) -> None:
    import time

    import torch
    import megatron.core.timers as timers_mod  # type: ignore
    import megatron.training.training as training  # type: ignore
    from megatron.training.global_vars import get_args  # type: ignore

    original_training_log = training.training_log
    if getattr(original_training_log, "_primus_deferred", False):
        return
    original_train = training.train

    state = {"pending": None, "stream": None, "armed": False, "anchor": None, "direct": False}

    def _tracker():
        try:
            from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker  # type: ignore

            return get_moe_metrics_tracker()
        except Exception:
            return None

    class _FixedClock:
        """Stands in for the ``time`` module inside megatron.core.timers during a deferred log."""

        def __init__(self, t):
            self._t = t

        def time(self):
            return self._t

    def _anchor_now():
        # Host time at which the device is known to be idle up to this event.
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        ev.synchronize()
        state["anchor"] = (time.time(), ev)

    def _run_pending():
        pend = state["pending"]
        if pend is None:
            return
        state["pending"] = None
        event, args, kwargs, metrics, mem = pend
        # The interval timer must read the time at which step N's GPU work completed (what a
        # synchronous log reads). If we have to wait for the event, the host time right after the
        # wait is that moment. If the event had already completed, take it from the GPU clock:
        # last exact anchor + device time between the anchor's event and this one.
        done_before = event.query()
        event.synchronize()
        now = time.time()
        anchor = state.get("anchor")
        if done_before and anchor is not None:
            t_end = anchor[0] + anchor[1].elapsed_time(event) / 1000.0
        else:
            t_end = now
            state["anchor"] = (now, event)
        tracker = _tracker()
        live = None
        if tracker is not None and metrics is not None:
            live, tracker._metrics = tracker._metrics, metrics
        side = state["stream"]
        main = torch.cuda.current_stream()
        side.wait_event(event)
        sync = torch.cuda.synchronize
        # Everything training_log waits for is complete (the event); a device sync would wait for
        # step N+1's forward instead.
        torch.cuda.synchronize = lambda *a, **k: None
        clock = timers_mod.time
        timers_mod.time = _FixedClock(t_end)
        mem_fns = {name: getattr(torch.cuda, name) for name in mem}
        for name, value in mem.items():
            setattr(torch.cuda, name, lambda *a, _v=value, **k: _v)
        state["direct"] = True
        try:
            with torch.cuda.stream(side):
                # Through the outermost training_log wrapper (Primus' print/average extensions may wrap ours).
                training.training_log(*args, **kwargs)
        finally:
            state["direct"] = False
            torch.cuda.synchronize = sync
            timers_mod.time = clock
            for name, fn in mem_fns.items():
                setattr(torch.cuda, name, fn)
            if tracker is not None and metrics is not None:
                tracker._metrics = live
            # Later compute-stream work (and frees of tensors training_log touched) is ordered after it.
            main.wait_stream(side)

    def _defer_ok(args, report_memory_flag, iteration, is_first_iteration):
        if is_first_iteration:
            return False
        if iteration >= args.train_iters:
            return False
        if args.log_interval and iteration % args.log_interval != 0:
            return False  # nothing to wait for: training_log only accumulates
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            return False
        if getattr(args, "log_energy", False) or getattr(args, "record_memory_history", False):
            return False
        if getattr(args, "log_memory_interval", None):
            return False
        return True

    @functools.wraps(original_training_log)
    def training_log(loss_dict, total_loss_dict, learning_rate, iteration, loss_scale,
                     report_memory_flag, *rest, **kwargs):
        call = (loss_dict, total_loss_dict, learning_rate, iteration, loss_scale, report_memory_flag, *rest)
        if state["direct"]:  # the deferred call itself
            return original_training_log(*call, **kwargs)
        _run_pending()  # never two pending logs
        args = get_args()
        first = kwargs.get("is_first_iteration", False)
        if not _defer_ok(args, report_memory_flag, iteration, first):
            flag = original_training_log(*call, **kwargs)
            if first or (args.log_interval and iteration % args.log_interval == 0):
                _anchor_now()  # this log synchronized the device anyway
            return flag
        if not state["armed"]:
            return original_training_log(*call, **kwargs)
        if state["stream"] is None:
            state["stream"] = torch.cuda.Stream()
        tracker = _tracker()
        metrics = None
        if tracker is not None:
            metrics, tracker._metrics = tracker._metrics, {}
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        adam_stream = getattr(training, "_primus_adam_overlap_stream", lambda: None)()
        if adam_stream is not None:
            # Step N's Adam runs on its own stream (adam_forward_overlap_patches): the step ends when both have.
            adam_stream.wait_event(event)
            event = torch.cuda.Event(enable_timing=True)
            event.record(adam_stream)
        # A memory report reads the allocator's host-side counters: report them as of step N's end.
        mem = {name: getattr(torch.cuda, name)() for name in _MEM_COUNTERS} if report_memory_flag else {}
        state["pending"] = (event, call, kwargs, metrics, mem)
        # What training_log returns on a log iteration: the memory report has been made.
        return False

    train_sig = inspect.signature(original_train)

    def _arm(optimizer):
        """Run the pending log when step N+1's optimizer step starts: by then the forward and the
        backward are enqueued and the host is several ms ahead of the GPU (the backward is GPU-bound)."""
        if optimizer is None or getattr(optimizer, "_primus_deferred_step_log", False):
            return optimizer is not None
        original_step = optimizer.step

        @functools.wraps(original_step)
        def step(*a, **k):
            _run_pending()
            return original_step(*a, **k)

        optimizer.step = step
        optimizer._primus_deferred_step_log = True
        return True

    @functools.wraps(original_train)
    def train(*args, **kwargs):
        try:
            optimizer = train_sig.bind_partial(*args, **kwargs).arguments.get("optimizer")
        except TypeError:
            optimizer = None
        # Megatron's train(forward_step_func, model, optimizer, ...)
        if optimizer is None and len(args) > 2:
            optimizer = args[2]
        state["armed"] = _arm(optimizer)
        try:
            return original_train(*args, **kwargs)
        finally:
            _run_pending()

    for name in ("evaluate_and_print_results", "save_checkpoint_and_time"):
        fn = getattr(training, name, None)
        if fn is None:
            continue

        def _make(fn):
            @functools.wraps(fn)
            def wrapped(*a, **k):
                _run_pending()  # keep the log lines in order
                return fn(*a, **k)

            return wrapped

        setattr(training, name, _make(fn))

    training_log._primus_deferred = True
    training.training_log = training_log
    training.train = train
    log_rank_0("[Patch:megatron.training.deferred_step_log] training_log(N) runs when step N+1's optimizer step starts")

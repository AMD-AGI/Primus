###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Defer the diffusion train step's loss check and SIGTERM poll to the end of the step.

WHY:
  Besides the synchronizations that ``primus_train.local_log_metrics`` and
  ``primus_train.defer_host_sync`` remove, every steady-state step of the diffusion
  loop synchronizes the host with the GPU for its loss-explosion check and for its
  SIGTERM poll (``torch.cuda.set_sync_debug_mode("warn")`` lists them). Each one
  drains the GPU queue, so the GPU then idles while the CPU issues the next launches.

WHAT:
  Removes both:

    loss check         ``check_loss`` (NaN or > 100) reads a pinned copy of the
                       loss at the end of the same step, before any checkpoint is
                       saved
    preemption poll    the SIGTERM flag is all-reduced asynchronously; each poll
                       reads the previous poll's result

  No AutoModel source is edited: functions and properties are swapped at runtime.
  Off by default; idempotent.

WHAT CHANGES (read before enabling):
  * A loss explosion raises the same ``ValueError`` at the end of its step
    instead of before its backward pass, so that step's optimizer update is
    applied first. The run still stops before any checkpoint is written.
  * A SIGTERM is acted on one step later.
  * Numerics are unchanged: the same kernels run on the same data.

Setting (``primus_train:`` section, off by default):
    defer_step_syncs: false
"""
from __future__ import annotations

import inspect
import logging
import math
from types import SimpleNamespace

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_INSTALLED_FLAG = "_primus_defer_step_syncs"
_SECTION = "primus_train"
# The bound FlowMatchingPipeline.step's check_loss applies.
_LOSS_LIMIT = 100.0


def is_enabled() -> bool:
    """Whether the step synchronizations should be deferred."""
    return options.flag(f"{_SECTION}.defer_step_syncs")


def _patch_loss_check() -> None:
    import torch
    from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline
    from nemo_automodel.components.training.step_scheduler import StepScheduler

    orig_step = FlowMatchingPipeline.step
    signature = inspect.signature(orig_step)
    pending: list = []

    def check_pending() -> None:
        while pending:
            event, host = pending.pop(0)
            event.synchronize()
            value = float(host)
            if math.isnan(value) or value > _LOSS_LIMIT:
                logger.error(f"[ERROR] Loss explosion! Loss={value:.3f}")
                raise ValueError(f"Loss exploded: {value}")

    def step(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments.get("check_loss"):
            return orig_step(self, *args, **kwargs)
        check_pending()
        bound.arguments["check_loss"] = False
        out = orig_step(*bound.args, **bound.kwargs)
        host = torch.empty((), dtype=torch.float32, pin_memory=True)
        host.copy_(out[1].detach().float(), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        pending.append((event, host))
        return out

    orig_is_ckpt_step = StepScheduler.is_ckpt_step

    def is_ckpt_step(self):
        check_pending()
        return orig_is_ckpt_step.fget(self)

    FlowMatchingPipeline.step = step
    StepScheduler.is_ckpt_step = property(is_ckpt_step, doc=orig_is_ckpt_step.__doc__)


def _patch_sigterm_poll() -> None:
    import torch
    import torch.distributed as dist
    from nemo_automodel.components.training import signal_handler
    from nemo_automodel.components.training.step_scheduler import StepScheduler

    orig_sigterm_received = StepScheduler.sigterm_received

    def poll(scheduler):
        """Result of the previous poll's all-reduce, or None if not on a GPU group."""
        handler = scheduler.sig_handler
        state = getattr(scheduler, "_primus_sigterm_poll", None)
        if state is None:
            device = signal_handler.get_device()
            if device.type != "cuda" or not dist.is_initialized():
                return None
            state = scheduler._primus_sigterm_poll = SimpleNamespace(
                flag=torch.zeros(1, dtype=torch.int32, device=device),
                host=torch.zeros(1, dtype=torch.int32, pin_memory=True),
                event=None,
            )
        seen = False
        if state.event is not None:
            state.event.synchronize()
            seen = bool(state.host[0])
        state.flag.fill_(int(bool(handler._signal_received)))
        dist.all_reduce(state.flag, op=dist.ReduceOp.MAX, group=handler.group)
        state.host.copy_(state.flag, non_blocking=True)
        state.event = torch.cuda.Event()
        state.event.record()
        return seen

    def sigterm_received(self):
        if self.sigterm_flag:
            return True
        if self.sig_handler is None or self._sig_polled_step == self.step:
            return False
        seen = poll(self)
        if seen is None:
            return orig_sigterm_received.fget(self)
        self._sig_polled_step = self.step
        self.sigterm_flag = seen
        return seen

    StepScheduler.sigterm_received = property(sigterm_received, doc=orig_sigterm_received.__doc__)


def install() -> bool:
    """Apply the two replacements. Returns True if they are in place."""
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

    if getattr(TrainDiffusionRecipe, _INSTALLED_FLAG, False):
        return True
    _patch_loss_check()
    _patch_sigterm_poll()
    setattr(TrainDiffusionRecipe, _INSTALLED_FLAG, True)
    logger.info("[PrimusDeferStepSyncs] loss check and SIGTERM poll deferred to the end of the step.")
    return True

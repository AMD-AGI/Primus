###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Keep per-step logging off the GPU's communication queue (``--light-log-sync``).

With ``log_interval`` 1, every step's ``training_log`` reads Megatron's ``interval-time`` timer with ``barrier=True``,
and each ``Timer.start`` / ``stop`` does ``torch.distributed.barrier()`` (an NCCL collective) and
``torch.cuda.synchronize()`` (every stream on the device). Primus's ``print_rank_last`` patch then forwards the last
rank's log line to rank 0 with ``broadcast_object_list`` over NCCL. Two NCCL barriers and two NCCL broadcasts per step
occupy the NCCL queue, and the device-wide sync plus the NCCL barriers queue behind any param all-gather already in
flight, so the host cannot launch the next forward until every outstanding gather has finished (notably with
--ddp-param-gather-after-optimizer).

This patch, when enabled:
  * makes a world-group timer barrier a gloo barrier (a host-side rendezvous: same rank alignment, no GPU queue);
    a timer given an explicit ``barrier_group`` keeps it;
  * replaces the timer's ``torch.cuda.synchronize()`` with a sync of the current (compute) stream: the step's compute
    finishing is what the step time measures, and the optimizer step already waits on its gradient reduce-scatters;
  * forwards the single-node training log line over the same gloo group;
  * gives Megatron's ``training_log`` its per-step zero accumulators without a host-to-device copy: it builds
    ``torch.tensor([0.0], device='cuda')`` every step (a ``dict.get`` default, evaluated each call, and the reset after
    logging), and on HIP a synchronous copy from pageable host memory waits for the whole device. ``training.py`` gets
    a ``torch`` proxy that delegates everything and turns an all-zeros CUDA ``tensor(list)`` into ``torch.zeros`` (an
    async fill, same value and dtype).
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_GLOO = [None]

# Set by --light-log-sync-events (light_log_sync_events_patches) while it runs a deferred training_log: the all-zeros
# accumulators training_log creates then live on the host, so the loss sums it reads with .item() are host tensors.
HOST_ZEROS = [False]


def gloo_world_group():
    """A gloo group over all ranks, created on first use. Every caller reaches it on every rank in the same order (the
    timers and the log forwarder run on all ranks each logged step), so the collective creation is consistent.
    """
    if _GLOO[0] is None:
        _GLOO[0] = torch.distributed.new_group(backend="gloo")
    return _GLOO[0]


@register_patch(
    "megatron.args.light_log_sync",
    backend="megatron",
    phase="setup",
    description="Expose --light-log-sync on Megatron's logging argparse group.",
)
def patch_light_log_sync_arg(ctx: PatchContext) -> None:
    try:
        import megatron.training.arguments as margs
    except ImportError:
        return
    orig = getattr(margs, "_add_logging_args", None)
    if orig is None or getattr(orig, "_primus_light_log_sync", False):
        return

    def _add_logging_args(parser):
        parser = orig(parser)
        group = parser.add_argument_group(title="logging")
        group.add_argument(
            "--light-log-sync",
            action="store_true",
            default=False,
            help="Per-step logging without NCCL barriers/broadcasts or device-wide syncs (gloo barrier, compute-stream "
            "sync, gloo log forwarding).",
        )
        group.add_argument(
            "--light-log-sync-events",
            action="store_true",
            default=False,
            help="With --light-log-sync: no host syncs for logging at all. Values are read back asynchronously and "
            "each training_log call runs one step late; interval time is measured with CUDA events, rank-local.",
        )
        return parser

    _add_logging_args._primus_light_log_sync = True
    margs._add_logging_args = _add_logging_args
    log_rank_0("[Patch:megatron.args.light_log_sync] added --light-log-sync")


@register_patch(
    "megatron.timers.light_log_sync",
    backend="megatron",
    phase="before_train",
    description="Timer barriers on gloo, timer syncs on the compute stream, log forwarding on gloo.",
    condition=lambda ctx: bool(getattr(get_args(ctx), "light_log_sync", False)),
)
def patch_light_log_sync(ctx: PatchContext) -> None:
    import time

    from megatron.core import timers as T

    from primus.backends.megatron.patches.training_log import (
        print_rank_last_patches as P,
    )

    if getattr(T.Timer.start, "_primus_light_log_sync", False):
        return

    def _barrier(self):
        if self._barrier_group is None and torch.distributed.is_initialized():
            torch.distributed.barrier(group=gloo_world_group())
        else:
            torch.distributed.barrier(group=self._barrier_group)

    def start(self, barrier=False):
        assert not self._started, "timer has already been started"
        if barrier:
            _barrier(self)
        torch.cuda.current_stream().synchronize()
        self._start_time = time.time()
        self._started = True

    def stop(self, barrier=False):
        assert self._started, "timer is not started"
        if barrier:
            _barrier(self)
        torch.cuda.current_stream().synchronize()
        elapsed = time.time() - self._start_time
        self._elapsed += elapsed
        self._active_time += elapsed
        self._started = False

    start._primus_light_log_sync = True
    T.Timer.start, T.Timer.stop = start, stop

    import types

    import megatron.training.training as MT

    class _TorchProxy(types.ModuleType):
        def __getattr__(self, name):
            return getattr(torch, name)

    def _tensor(data, *a, **k):
        dev = k.get("device")
        if (
            not a
            and isinstance(data, list)
            and data
            and all(v == 0 for v in data)
            and dev is not None
            and str(dev).startswith("cuda")
        ):
            return torch.zeros(
                len(data),
                dtype=k.get("dtype"),
                device="cpu" if HOST_ZEROS[0] else dev,
                requires_grad=k.get("requires_grad", False),
            )
        return torch.tensor(data, *a, **k)

    proxy = _TorchProxy("torch")
    proxy.tensor = _tensor
    MT.torch = proxy
    P.FORWARD_LOG_GROUP = gloo_world_group
    log_rank_0(
        "[Patch:megatron.timers.light_log_sync] timer barriers on gloo, compute-stream syncs, gloo log forwarding"
    )

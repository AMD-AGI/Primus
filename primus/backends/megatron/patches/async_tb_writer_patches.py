###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""PRIMUS_ASYNC_TB_WRITER=1 (opt-in, default 0): Megatron's per-step TensorBoard scalars are written by a background thread.

``training_log`` runs on the last rank only and calls ``writer.add_scalar`` ~14 times per step (log_interval 1);
each call converts its value (``make_np``: a device read for tensors) and serializes an event on the training
thread, ~1 ms of host time per step on the DeepSeek-V4 EP4 proxy. The other ranks wait for it in the next
collective (the MoE-metrics all-reduce / timer barriers of ``training_log``), so it is step time on every rank.

The writer is wrapped: ``add_scalar`` snapshots its value (a device tensor is cloned on the current stream and an
event recorded; numbers are kept as is) and queues it; one daemon thread replays the calls in order into the real
SummaryWriter. Any other writer method first drains the queue, then runs inline. Same scalars, same steps, same
order; nothing on the training math changes.
"""

import os
import queue
import threading

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx=None) -> bool:
    return os.environ.get("PRIMUS_ASYNC_TB_WRITER", "0") == "1"


class _AsyncScalarWriter:
    def __init__(self, writer):
        self._w = writer
        self._q = queue.Queue()
        self._t = threading.Thread(target=self._run, name="primus-async-tb", daemon=True)
        self._t.start()
        import atexit

        atexit.register(self._drain)

    def _run(self):
        while True:
            item = self._q.get()
            try:
                tag, value, ev, args, kwargs = item
                if ev is not None:
                    ev.synchronize()
                self._w.add_scalar(tag, value, *args, **kwargs)
            except Exception:  # noqa: BLE001 - logging must never break training
                pass
            finally:
                self._q.task_done()

    def _drain(self):
        self._q.join()

    def add_scalar(self, tag, value, *args, **kwargs):
        import torch

        ev = None
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.is_cuda:
                value = value.clone()  # the source may be a static (graph) buffer the next step overwrites
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream())
            else:
                value = value.clone()
        self._q.put((tag, value, ev, args, kwargs))

    def __getattr__(self, name):
        attr = getattr(self._w, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            self._drain()
            return attr(*args, **kwargs)

        return call


@register_patch(
    "megatron.async_tb_writer",
    backend="megatron",
    phase="before_train",
    description="Write Megatron's per-step TensorBoard scalars from a background thread (PRIMUS_ASYNC_TB_WRITER)",
    condition=_enabled,
)
def patch_async_tb_writer(ctx: PatchContext) -> None:
    del ctx
    import megatron.training.global_vars as gv
    import megatron.training.training as mt

    orig = gv.get_tensorboard_writer
    if getattr(orig, "_primus_async", False):
        return
    cache = {}

    def get_tensorboard_writer():
        w = orig()
        if w is None:
            return None
        p = cache.get(id(w))
        if p is None or p._w is not w:
            p = cache[id(w)] = _AsyncScalarWriter(w)
        return p

    get_tensorboard_writer._primus_async = True
    gv.get_tensorboard_writer = get_tensorboard_writer
    if getattr(mt, "get_tensorboard_writer", None) is orig:
        mt.get_tensorboard_writer = get_tensorboard_writer
    log_rank_0("[Patch:megatron.async_tb_writer] TensorBoard scalars written by a background thread")

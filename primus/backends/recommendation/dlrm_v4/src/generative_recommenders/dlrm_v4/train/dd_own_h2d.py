# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""Make the next batch's input dist wait only for that batch's own H2D copy.

Problem (MNC iter 5 trace study, ship10 595b51e and ship8 2dfb933, 1n/2n/4n,
steps 53-55). progress() issues the H2D for batch N+2 first
(enqueue_batch_after_forward=True). Only after that does it run the deferred
start_sparse_data_dist(N+1) (TP_DEFER_INPUT_DIST, inside
wait_sparse_data_dist). torchrec's start_sparse_data_dist begins with
``_wait_for_batch(batch, memcpy_stream)``, which is
``data_dist.wait_stream(memcpy_stream)``. That waits for everything queued on
the memcpy stream so far, including the batch N+2 H2D issued a moment
earlier. Batch N+1 does not read that data. The effects:

- Batch N+1's dedup, bucketize and splits kernels on the high-priority (-1)
  data-dist queue sit behind a barrier on a ~4 ms SDMA copy. The traces show
  this in every rank-step at 1n, 2n and 4n, in both the candidate and the
  control: the first data-dist kernel is launched 0.3 ms after the H2D is
  issued but starts only 0.15-0.3 ms after the H2D ends.
- While that queue holds the blocked barrier, each main-stream dispatch costs
  about 12 us more at 2n/4n. Tiny elementwise kernels take 18.7 us instead of
  6.4 us; the extra cost is 2.6 us at 1n. The ``dedup_ec_indices`` ``.item()``
  hipStreamSynchronize also blocks the host for the same ~4.4 ms.

Fix: copy_batch_to_gpu records an event on the memcpy stream right after the
batch's own copies. The next-batch start_sparse_data_dist waits on that event
instead of the whole memcpy stream, and calls record_stream as
_wait_for_batch does. The wait still covers the batch's own H2D and
everything queued before it on the memcpy stream, but never a later batch's
copy. Every stream runs the same kernels on the same inputs in the same
order; the only change is the removed false dependency, so results are
bitwise identical.

Switch: DLRM_HSTU_FP8_DD_OWN_H2D (absent = off in code; gin item (66) turns it on;
=0 opts out). The prefix is only
what launch_slurm.sh forwards into the container; the knob has nothing to do
with FP8. When the knob is on, the arming line ``ARMED`` is logged.
"""

import logging
import os
from typing import Any, Dict, Optional

logger: logging.Logger = logging.getLogger(__name__)

ENV = "DLRM_HSTU_FP8_DD_OWN_H2D"
ARMED = (
    "armed OK: dd_own_h2d -- next-batch input dist waits on its own batch's "
    "H2D event, not the whole memcpy stream"
)
# Contexts in flight are at most 3 (batches N, N+1, N+2); keep a few spare for
# fill/eval/window-boundary paths that enqueue a batch whose input dist is
# started by the stock (non-deferred) path and never consumes its event.
_MAX_PENDING = 8


def enabled() -> bool:
    """On when DLRM_HSTU_FP8_DD_OWN_H2D is 1/true/on/yes (read per call).

    Absent = off in code; yambda_5b.gin item (66) turns it on through
    apply_env_bootstrap (a pre-set env var wins, so =0 opts out).
    """
    v = os.environ.get(ENV, "0").strip().lower()
    return v in ("1", "true", "on", "yes")


def _rank() -> str:
    """This process's rank for the log tag.

    torch.distributed's rank once the process group is up (the pipelines are
    built after setup(); the spawned ranks have no RANK in their env), else
    RANK, else LOCAL_RANK, else "?". torch is imported here, not at module
    level, so importing this module stays as cheap as before.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        dist = None
    if dist is not None and dist.is_available() and dist.is_initialized():
        return str(dist.get_rank())
    for key in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return "?"


class OwnH2DGate:
    """Per-pipeline bookkeeping: one H2D-complete event per enqueued batch.

    ``cuda`` is injectable for CPU unit tests; it defaults to ``torch.cuda``
    (imported lazily so this module stays importable without torch).
    """

    _built = 0

    def __init__(self, cuda: Optional[Any] = None) -> None:
        OwnH2DGate._built += 1
        self.on: bool = enabled()
        self._cuda = cuda
        self._events: Dict[int, Any] = {}
        self.used = 0
        self.fallback = 0
        self._tag = "rank=%s pipeline#%d" % (_rank(), OwnH2DGate._built)
        if self.on:
            logger.info("%s (%s=%s, %s)", ARMED, ENV, os.environ.get(ENV, "1"), self._tag)
        else:
            logger.info(
                "dd_own_h2d OFF (%s=%s, %s): next-batch input dist uses "
                "torchrec's wait_stream(memcpy_stream)",
                ENV, os.environ.get(ENV), self._tag,
            )

    def _c(self) -> Any:
        if self._cuda is None:
            import torch

            self._cuda = torch.cuda
        return self._cuda

    def after_copy(self, memcpy_stream: Any, batch: Any, context: Any) -> None:
        """Record "batch's H2D done" right after copy_batch_to_gpu queued it."""
        if not self.on or batch is None or context is None or memcpy_stream is None:
            return
        ev = self._c().Event()
        ev.record(memcpy_stream)
        self._events[int(context.index)] = ev
        if len(self._events) > _MAX_PENDING:
            for k in sorted(self._events)[: len(self._events) - _MAX_PENDING]:
                del self._events[k]

    def wait(self, batch: Any, context: Any) -> bool:
        """Call with the data-dist stream current, in place of
        ``_wait_for_batch(batch, memcpy_stream)``. Returns False when there
        is no event for this context; the caller then uses the stock wait."""
        if not self.on or batch is None or context is None:
            return False
        ev = self._events.pop(int(context.index), None)
        if ev is None:
            self.fallback += 1
            return False
        cur = self._c().current_stream()
        cur.wait_event(ev)
        # Same allocator bookkeeping as torchrec _wait_for_batch: the batch was
        # allocated on the memcpy stream and is now used on this stream.
        batch.record_stream(cur)
        self.used += 1
        if self.used == 1:
            logger.info(
                "dd_own_h2d: first next-batch input dist gated on its own H2D "
                "event (batch index %d, %s)", int(context.index), self._tag,
            )
        return True

    def discard(self, context: Any) -> None:
        if context is not None:
            self._events.pop(int(context.index), None)

    def clear(self) -> None:
        self._events.clear()

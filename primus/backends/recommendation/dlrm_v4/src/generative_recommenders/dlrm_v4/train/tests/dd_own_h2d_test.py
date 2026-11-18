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
"""CPU unit tests for dd_own_h2d.OwnH2DGate (no torch / GPU needed).

Fake streams log their operations. An event remembers the stream position at
which it was recorded. The steady-state progress() order is replayed:
copy(N+2), then the deferred start_sparse_data_dist(N+1). The test checks
that the data-dist stream waits for batch N+1's copies and not for N+2's.
"""
import contextlib
import os
import unittest
from unittest import mock

from generative_recommenders.dlrm_v4.train import dd_own_h2d


class _Stream:
    def __init__(self, name):
        self.name = name
        self.ops = []

    def copy(self, what):
        self.ops.append(("copy", what))

    def wait_event(self, ev):
        self.ops.append(("wait", ev.stream.name, ev.pos))


class _Event:
    def __init__(self):
        self.stream = None
        self.pos = None

    def record(self, stream):
        self.stream = stream
        self.pos = len(stream.ops)
        stream.ops.append(("record", self.pos))


class _Cuda:
    def __init__(self):
        self.current = None

    def Event(self):
        return _Event()

    def current_stream(self):
        return self.current


class _Batch:
    def __init__(self, idx):
        self.idx = idx
        self.recorded = []

    def record_stream(self, s):
        self.recorded.append(s.name)


class _Ctx:
    def __init__(self, index):
        self.index = index


class OwnH2DGateTest(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.pop(dd_own_h2d.ENV, None)
        # What yambda_5b.gin item (66) does through apply_env_bootstrap.
        os.environ[dd_own_h2d.ENV] = "1"

    def tearDown(self):
        os.environ.pop(dd_own_h2d.ENV, None)
        if self._old is not None:
            os.environ[dd_own_h2d.ENV] = self._old

    def _copy(self, gate, memcpy, n):
        b, c = _Batch(n), _Ctx(n)
        memcpy.copy("batch%d" % n)
        gate.after_copy(memcpy, b, c)
        return b, c

    def test_waits_on_own_batch_not_later_copy(self):
        cuda = _Cuda()
        memcpy, dd = _Stream("memcpy"), _Stream("data_dist")
        gate = dd_own_h2d.OwnH2DGate(cuda=cuda)
        self.assertTrue(gate.on)
        b1, c1 = self._copy(gate, memcpy, 1)      # enqueued in progress(N-1)
        pos_after_b1 = len(memcpy.ops)
        self._copy(gate, memcpy, 2)               # progress(N): enqueue_batch(N+2) ...
        cuda.current = dd                         # ... then deferred start_sparse_data_dist(N+1)
        self.assertTrue(gate.wait(b1, c1))
        kind, src, pos = dd.ops[-1]
        self.assertEqual((kind, src), ("wait", "memcpy"))
        # the event sits right behind batch 1's copy, before batch 2's copy
        self.assertEqual(pos, pos_after_b1 - 1)
        self.assertEqual(memcpy.ops[pos - 1], ("copy", "batch1"))
        self.assertIn(("copy", "batch2"), memcpy.ops[pos + 1:])
        self.assertEqual(b1.recorded, ["data_dist"])
        # consumed once; a second wait falls back to the stock path
        self.assertFalse(gate.wait(b1, c1))
        self.assertEqual((gate.used, gate.fallback), (1, 1))

    def test_absent_is_off(self):
        os.environ.pop(dd_own_h2d.ENV, None)
        gate = dd_own_h2d.OwnH2DGate(cuda=_Cuda())
        self.assertFalse(gate.on)

    def test_kill_switch(self):
        os.environ[dd_own_h2d.ENV] = "0"
        cuda = _Cuda()
        memcpy = _Stream("memcpy")
        gate = dd_own_h2d.OwnH2DGate(cuda=cuda)
        self.assertFalse(gate.on)
        b, c = self._copy(gate, memcpy, 1)
        self.assertEqual(memcpy.ops, [("copy", "batch1")])   # no event recorded
        cuda.current = _Stream("data_dist")
        self.assertFalse(gate.wait(b, c))

    def test_missing_event_and_bound(self):
        cuda = _Cuda()
        memcpy = _Stream("memcpy")
        gate = dd_own_h2d.OwnH2DGate(cuda=cuda)
        cuda.current = _Stream("data_dist")
        self.assertFalse(gate.wait(_Batch(7), _Ctx(7)))     # never copied -> stock path
        for n in range(20):
            self._copy(gate, memcpy, n)
        self.assertLessEqual(len(gate._events), dd_own_h2d._MAX_PENDING)
        self.assertIn(19, gate._events)
        gate.discard(_Ctx(19))
        self.assertNotIn(19, gate._events)
        gate.clear()
        self.assertEqual(gate._events, {})

    def _armed_line(self, env, dist_rank=None):
        """The arming line of a gate built with RANK / LOCAL_RANK as in `env`
        and, if dist_rank is set, an initialized process group of that rank
        (torch.distributed patched; skipped where torch is not importable)."""
        clean = {
            k: v for k, v in os.environ.items() if k not in ("RANK", "LOCAL_RANK")
        }
        clean.update(env)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, clean, clear=True))
            try:
                import torch.distributed as dist
            except ImportError:
                dist = None
            if dist is None and dist_rank is not None:
                self.skipTest("torch is not importable")
            if dist is not None:
                up = dist_rank is not None
                stack.enter_context(
                    mock.patch.object(dist, "is_available", return_value=True)
                )
                stack.enter_context(
                    mock.patch.object(
                        dist, "is_initialized", create=True, return_value=up
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        dist, "get_rank", create=True, return_value=dist_rank
                    )
                )
            with self.assertLogs(dd_own_h2d.logger, level="INFO") as cm:
                dd_own_h2d.OwnH2DGate(cuda=_Cuda())
        self.assertEqual(len(cm.output), 1, cm.output)
        self.assertIn(dd_own_h2d.ARMED, cm.output[0])
        return cm.output[0]

    def test_rank_tag_env_fallback(self):
        # The spawned ranks have no RANK in their env; without a process
        # group the tag falls back to RANK, then LOCAL_RANK, then "?".
        line = self._armed_line({"RANK": "5", "LOCAL_RANK": "3"})
        self.assertIn("rank=5 pipeline#", line)
        self.assertIn("rank=3 pipeline#", self._armed_line({"LOCAL_RANK": "3"}))
        self.assertIn("rank=? pipeline#", self._armed_line({}))

    def test_rank_tag_process_group_wins(self):
        line = self._armed_line({"RANK": "5"}, dist_rank=7)
        self.assertIn("rank=7 pipeline#", line)
        self.assertIn("rank=12 pipeline#", self._armed_line({}, dist_rank=12))


if __name__ == "__main__":
    unittest.main()

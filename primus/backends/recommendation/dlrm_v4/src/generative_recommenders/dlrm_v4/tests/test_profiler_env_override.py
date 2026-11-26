"""The $DLRM_TRACE_STEPS / $DLRM_TRACE_SKIP_HEAD diagnostic overrides in
dlrm_v4.utils.Profiler must be inert when unset: same schedule and skip_head as
the gin arguments (the MI350X / production path), and take effect only when set.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

from generative_recommenders.dlrm_v4 import utils as U

GIN_KW = dict(trace_steps=[41, 51], warmup=1, active=8, skip_head=1, keep_active=5)


def _actions(sched, n=80):
    return [sched(i) for i in range(n)]


class ProfilerEnvOverrideTest(unittest.TestCase):
    def _make(self, env):
        with mock.patch.dict(os.environ, env, clear=False):
            for k in ("DLRM_TRACE_STEPS", "DLRM_TRACE_SKIP_HEAD"):
                if k not in env:
                    os.environ.pop(k, None)
            with tempfile.TemporaryDirectory() as d:
                return U.Profiler(rank=0, trace_dir=d, **GIN_KW)

    def test_unset_is_identical_to_gin(self):
        p = self._make({})
        ref = U._multi_window_schedule(GIN_KW["trace_steps"], GIN_KW["warmup"], GIN_KW["active"])
        self.assertEqual(_actions(p._sched), _actions(ref))
        self.assertEqual(p._skip_head, GIN_KW["skip_head"])
        self.assertEqual(p._active, GIN_KW["active"])

    def test_empty_string_is_unset(self):
        p = self._make({"DLRM_TRACE_STEPS": "", "DLRM_TRACE_SKIP_HEAD": ""})
        ref = U._multi_window_schedule(GIN_KW["trace_steps"], GIN_KW["warmup"], GIN_KW["active"])
        self.assertEqual(_actions(p._sched), _actions(ref))
        self.assertEqual(p._skip_head, GIN_KW["skip_head"])

    def test_set_overrides(self):
        p = self._make({"DLRM_TRACE_STEPS": "800,2000", "DLRM_TRACE_SKIP_HEAD": "0"})
        ref = U._multi_window_schedule([800, 2000], GIN_KW["warmup"], GIN_KW["active"])
        self.assertEqual(_actions(p._sched, 2100), _actions(ref, 2100))
        self.assertEqual(p._skip_head, 0)
        self.assertNotEqual(_actions(p._sched), _actions(
            U._multi_window_schedule(GIN_KW["trace_steps"], GIN_KW["warmup"], GIN_KW["active"])))


if __name__ == "__main__":
    unittest.main()

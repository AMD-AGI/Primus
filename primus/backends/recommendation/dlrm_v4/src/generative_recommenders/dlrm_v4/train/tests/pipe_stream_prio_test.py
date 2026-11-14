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
"""CPU unit tests for item (67): the multi-node train-pipeline stream priority.

host_sync_free.train_pipe_stream_priority() decides the priority of the train
pipeline's memcpy / data-dist streams, and alloc_pregrow._pair_priorities()
creates the pre-grown (train, eval) pairs at the priority each pipeline will
ask for. No GPU and no process group are needed: torch.distributed is not
initialized here, so the node count comes from NNODES alone.
"""
import os
import unittest
from unittest import mock

from generative_recommenders.dlrm_v4 import alloc_pregrow, host_sync_free as hsf

_KEYS = (
    "DLRM_HSTU_FP8_PIPE_STREAM_PRIO",
    "DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN",
    "NNODES",
    "SLURM_NNODES",
    "LOCAL_WORLD_SIZE",
)


class TrainPipeStreamPriorityTest(unittest.TestCase):
    def _prios(self, **env):
        clean = {k: v for k, v in os.environ.items() if k not in _KEYS}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True):
            return (
                hsf.train_pipe_stream_priority(),
                hsf.pipe_stream_priority(),
                alloc_pregrow._pair_priorities(),
            )

    def test_unset_keeps_torchrec_priority(self):
        self.assertEqual(self._prios(), (None, None, (-1, -1)))
        self.assertEqual(self._prios(NNODES="4"), (None, None, (-1, -1)))

    def test_multi_node_default_applies_to_train_only(self):
        self.assertEqual(
            self._prios(NNODES="4", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0"),
            (0, None, (0, -1)),
        )
        self.assertEqual(
            self._prios(NNODES="2", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0"),
            (0, None, (0, -1)),
        )

    def test_single_node_is_unchanged(self):
        self.assertEqual(
            self._prios(NNODES="1", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0"),
            (None, None, (-1, -1)),
        )
        self.assertEqual(
            self._prios(DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0"),
            (None, None, (-1, -1)),
        )

    def test_ab_override_wins_for_every_pipeline(self):
        self.assertEqual(
            self._prios(
                NNODES="4",
                DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0",
                DLRM_HSTU_FP8_PIPE_STREAM_PRIO="-1",
            ),
            (-1, -1, (-1, -1)),
        )
        self.assertEqual(
            self._prios(NNODES="1", DLRM_HSTU_FP8_PIPE_STREAM_PRIO="0"),
            (0, 0, (0, 0)),
        )

    def test_empty_values_are_unset(self):
        self.assertEqual(
            self._prios(
                NNODES="4",
                DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="",
                DLRM_HSTU_FP8_PIPE_STREAM_PRIO="",
            ),
            (None, None, (-1, -1)),
        )

    def test_slurm_nnodes_and_strict_nnodes(self):
        self.assertEqual(
            self._prios(SLURM_NNODES="2", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0"),
            (0, None, (0, -1)),
        )
        # NNODES decides when set: 1 means single node, whatever SLURM says.
        self.assertEqual(
            self._prios(
                NNODES="1", SLURM_NNODES="4", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0"
            ),
            (None, None, (-1, -1)),
        )

    def test_opt_out_value(self):
        self.assertEqual(
            self._prios(NNODES="4", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="-1"),
            (-1, None, (-1, -1)),
        )

    def test_invalid_values_are_refused_at_every_node_count(self):
        for nn in ("1", "4"):
            for bad in ("x", "1", "-2", "0.5"):
                with self.assertRaises(ValueError):
                    self._prios(NNODES=nn, DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN=bad)
                with self.assertRaises(ValueError):
                    self._prios(NNODES=nn, DLRM_HSTU_FP8_PIPE_STREAM_PRIO=bad)

    def test_fallback_turns_item_off(self):
        clean = {k: v for k, v in os.environ.items() if k not in _KEYS}
        clean.update(NNODES="4", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0")
        with mock.patch.dict(os.environ, clean, clear=True):
            self.assertEqual(hsf.train_pipe_stream_priority(), 0)
            hsf.disable_train_pipe_priority("unit test")
            self.assertEqual(hsf.train_pipe_stream_priority(), -1)
            self.assertEqual(alloc_pregrow._pair_priorities(), (-1, -1))


class GuardTest(unittest.TestCase):
    """alloc_pregrow.guard_train_pair: three-valued probe -> item 67 on/off.

    shares_hw_queue returns True (shared queue), False (separate queues) or
    None (the probe could not run). Only False/False keeps item 67 on; an
    unverified placement must fail closed (MNC iter 11/12 fix)."""

    def _guard(self, results, **env):
        clean = {k: v for k, v in os.environ.items() if k not in _KEYS}
        clean.update(NNODES="4", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0")
        clean.update(env)
        it = iter(results)
        with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(
            hsf, "shares_hw_queue", side_effect=lambda a, b: next(it)
        ):
            fell_back = alloc_pregrow.guard_train_pair(object(), object(), object())
            return fell_back, hsf.train_pipe_stream_priority()

    def test_separate_queues_keep_item_on(self):
        with self.assertLogs(alloc_pregrow.logger, level="INFO") as cm:
            self.assertEqual(self._guard([False, False]), (False, 0))
        self.assertTrue(
            any(
                "DLRMV4 multi-node side-stream priority placement check: train "
                "memcpy/data_dist run beside main" in m
                for m in cm.output
            ),
            cm.output,
        )

    def test_shared_queue_falls_back(self):
        self.assertEqual(self._guard([False, True]), (True, -1))
        self.assertEqual(self._guard([True, False]), (True, -1))

    def test_unverified_probe_fails_closed(self):
        self.assertEqual(self._guard([None, False]), (True, -1))
        self.assertEqual(self._guard([False, None]), (True, -1))
        self.assertEqual(self._guard([None, None]), (True, -1))

    def test_shared_reason_wins_over_unverified(self):
        # A measured shared queue is reported as such even if the other
        # side's probe could not run (the fallback reason names the cause).
        with self.assertLogs(hsf.logger, level="WARNING") as cm:
            self.assertEqual(self._guard([True, None]), (True, -1))
        self.assertTrue(
            any("shares the main stream's hardware queue" in m for m in cm.output),
            cm.output,
        )
        self.assertTrue(
            any(
                "DLRMV4 multi-node side-stream priority fallback: train pipeline "
                "side streams keep torchrec's priority -1" in m
                for m in cm.output
            ),
            cm.output,
        )
        self.assertFalse(any("placement unverified" in m for m in cm.output), cm.output)
        with self.assertLogs(hsf.logger, level="WARNING") as cm:
            self.assertEqual(self._guard([None, False]), (True, -1))
        self.assertTrue(any("placement unverified" in m for m in cm.output), cm.output)

    def test_explicit_override_is_not_guarded(self):
        # The A/B override is left as asked: no probe runs at all.
        self.assertEqual(
            self._guard([], DLRM_HSTU_FP8_PIPE_STREAM_PRIO="0"), (False, 0)
        )


class SingleNodeInactiveLineTest(unittest.TestCase):
    """host_sync_free.log_train_pipe_priority_inactive: the 1-node line.

    One info line per process, train pipeline only, only when the multi-node
    priority is bound (0), the A/B override is unset and the job is 1-node.
    Log only: the env and every priority decision are the same afterwards."""

    LINE = (
        "DLRMV4 multi-node side-stream priority inactive: single node, train "
        "pipeline keeps torchrec priority -1"
    )

    def setUp(self):
        self._saved = hsf._mn_inactive_logged
        hsf._mn_inactive_logged = False

    def tearDown(self):
        hsf._mn_inactive_logged = self._saved

    def _call(self, role="train", **env):
        clean = {k: v for k, v in os.environ.items() if k not in _KEYS}
        clean.update(env)
        with mock.patch.dict(os.environ, clean, clear=True):
            decisions = (
                hsf.train_pipe_stream_priority(),
                hsf.pipe_stream_priority(),
                alloc_pregrow._pair_priorities(),
            )
            before = dict(os.environ)
            logged = hsf.log_train_pipe_priority_inactive(role)
            self.assertEqual(dict(os.environ), before)
            self.assertEqual(
                (
                    hsf.train_pipe_stream_priority(),
                    hsf.pipe_stream_priority(),
                    alloc_pregrow._pair_priorities(),
                ),
                decisions,
            )
            return logged, decisions

    def test_single_node_logs_once_per_process(self):
        with self.assertLogs(hsf.logger, level="INFO") as cm:
            logged, decisions = self._call(
                NNODES="1", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0"
            )
        self.assertTrue(logged)
        self.assertEqual(decisions, (None, None, (-1, -1)))
        self.assertEqual(cm.output, ["INFO:%s:%s" % (hsf.logger.name, self.LINE)])
        # Later train pipelines in the same process stay silent.
        with mock.patch.object(hsf.logger, "info") as info:
            self.assertFalse(
                self._call(NNODES="1", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0")[0]
            )
            self.assertFalse(self._call(DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0")[0])
        info.assert_not_called()

    def test_silent_for_eval_multi_node_unset_opt_out_and_override(self):
        cases = (
            ("eval", dict(NNODES="1", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0")),
            ("train", dict(NNODES="4", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0")),
            ("train", dict(SLURM_NNODES="2", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0")),
            ("train", dict(NNODES="1")),
            ("train", dict(NNODES="1", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="-1")),
            (
                "train",
                dict(
                    NNODES="1",
                    DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0",
                    DLRM_HSTU_FP8_PIPE_STREAM_PRIO="0",
                ),
            ),
        )
        with mock.patch.object(hsf.logger, "info") as info:
            for role, env in cases:
                self.assertFalse(self._call(role, **env)[0], (role, env))
        info.assert_not_called()
        # None of them used up the once-per-process line.
        with self.assertLogs(hsf.logger, level="INFO") as cm:
            self.assertTrue(
                self._call(NNODES="1", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0")[0]
            )
        self.assertEqual(cm.output, ["INFO:%s:%s" % (hsf.logger.name, self.LINE)])


class OneMultiNodeRuleTest(unittest.TestCase):
    """Items (64) and (67) share ONE multi-node rule:
    host_sync_free._spans_nodes calls train/_env_bootstrap._spans_nodes. A
    process-group answer patched there must reach item (67)'s gate (no
    process group exists here, so a private copy would answer False)."""

    def test_item64_gate_uses_the_bootstrap_rule(self):
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        base = {k: v for k, v in os.environ.items() if k not in _KEYS}
        envs = (
            {},
            {"NNODES": "1"},
            {"NNODES": "2"},
            {"SLURM_NNODES": "2"},
            {"NNODES": "1", "SLURM_NNODES": "4"},
        )
        for group in (None, False, True):
            for env in envs:
                with mock.patch.dict(
                    os.environ, dict(base, **env), clear=True
                ), mock.patch.object(eb, "_group_spans_nodes", return_value=group):
                    self.assertEqual(
                        hsf._spans_nodes(), eb._spans_nodes(), (group, env)
                    )
        with mock.patch.dict(
            os.environ, dict(base, DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN="0"), clear=True
        ), mock.patch.object(eb, "_group_spans_nodes", return_value=True):
            self.assertTrue(hsf._spans_nodes())
            self.assertEqual(hsf.train_pipe_stream_priority(), 0)


class _Stream:
    def __init__(self, prio):
        self.priority = prio


class _Pipe:
    def __init__(self):
        self._memcpy_stream = _Stream(-1)
        self._data_dist_stream = _Stream(-1)


class AdoptByRoleTest(unittest.TestCase):
    def setUp(self):
        self._saved = (list(alloc_pregrow._side_streams), list(alloc_pregrow._pair_roles))
        self.train = {"memcpy": _Stream(0), "data_dist": _Stream(0)}
        self.eval = {"memcpy": _Stream(-1), "data_dist": _Stream(-1)}
        alloc_pregrow._side_streams[:] = [self.train, self.eval]
        alloc_pregrow._pair_roles[:] = ["train", "eval"]

    def tearDown(self):
        alloc_pregrow._side_streams[:] = self._saved[0]
        alloc_pregrow._pair_roles[:] = self._saved[1]

    def test_eval_first_gets_the_eval_pair(self):
        ev, tr = _Pipe(), _Pipe()
        self.assertTrue(alloc_pregrow.adopt_streams(ev, "eval"))
        self.assertIs(ev._data_dist_stream, self.eval["data_dist"])
        self.assertTrue(alloc_pregrow.adopt_streams(tr, "train"))
        self.assertIs(tr._data_dist_stream, self.train["data_dist"])
        self.assertFalse(alloc_pregrow.adopt_streams(_Pipe(), "train"))

    def test_build_order_without_role(self):
        tr = _Pipe()
        self.assertTrue(alloc_pregrow.adopt_streams(tr))
        self.assertIs(tr._memcpy_stream, self.train["memcpy"])
        self.assertEqual(alloc_pregrow._pair_roles, ["eval"])

    def test_unknown_or_missing_role_is_not_adopted(self):
        alloc_pregrow._side_streams[:] = [self.train]
        alloc_pregrow._pair_roles[:] = ["train"]
        p = _Pipe()
        self.assertFalse(alloc_pregrow.adopt_streams(p, "eval"))
        self.assertEqual(p._data_dist_stream.priority, -1)


if __name__ == "__main__":
    unittest.main()

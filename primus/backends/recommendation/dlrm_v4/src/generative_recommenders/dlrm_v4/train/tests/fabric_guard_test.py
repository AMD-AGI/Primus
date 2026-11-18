#!/usr/bin/env python3
# pyre-unsafe
"""fabric_guard (runner plans/dlrmv4 09): on ionic NICs with RCCL's built-in
IB transport, NCCL_GDR_FLUSH_DISABLE=1 becomes 0 unless DLRM_ALLOW_GDR_FLUSH_DISABLE=1
or the ANP plugin is in use; misnamed knobs are reported, never rewritten."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from generative_recommenders.dlrm_v4 import fabric_guard  # noqa: E402


def _ib(devs):
    d = tempfile.mkdtemp()
    for n in devs:
        os.mkdir(os.path.join(d, n))
    return d


class FabricGuardTest(unittest.TestCase):
    def test_ionic_builtin_flush_off_is_turned_on(self):
        env = {"NCCL_GDR_FLUSH_DISABLE": "1"}
        notes = fabric_guard.check_rccl_env(env, _ib(["ionic_0", "ionic_1"]))
        self.assertEqual(env["NCCL_GDR_FLUSH_DISABLE"], "0")
        self.assertEqual(len(notes), 1)
        self.assertIn("NCCL_GDR_FLUSH_DISABLE=1 -> 0: 2 ionic NICs", notes[0])

    def test_escape_hatch_keeps_value(self):
        env = {"NCCL_GDR_FLUSH_DISABLE": "1", "DLRM_ALLOW_GDR_FLUSH_DISABLE": "1"}
        notes = fabric_guard.check_rccl_env(env, _ib(["ionic_0"]))
        self.assertEqual(env["NCCL_GDR_FLUSH_DISABLE"], "1")
        self.assertIn("kept", notes[0])

    def test_anp_plugin_or_other_nics_untouched(self):
        env = {"NCCL_GDR_FLUSH_DISABLE": "1", "NCCL_NET_PLUGIN": "/opt/anp/librccl-anp.so"}
        self.assertEqual(fabric_guard.check_rccl_env(env, _ib(["ionic_0"])), [])
        self.assertEqual(env["NCCL_GDR_FLUSH_DISABLE"], "1")
        env = {"NCCL_GDR_FLUSH_DISABLE": "1"}
        self.assertEqual(fabric_guard.check_rccl_env(env, _ib(["bnxt_re0", "mlx5_0"])), [])
        self.assertEqual(fabric_guard.check_rccl_env(env, "/nonexistent"), [])
        self.assertEqual(env["NCCL_GDR_FLUSH_DISABLE"], "1")

    def test_flush_on_or_unset_is_quiet(self):
        for env in ({"NCCL_GDR_FLUSH_DISABLE": "0"}, {}):
            self.assertEqual(fabric_guard.check_rccl_env(env, _ib(["ionic_0"])), [])

    def test_misnamed_reported_not_rewritten(self):
        env = {"NET_OPTIONAL_RECV_COMPLETION": "0", "NCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING": "0"}
        notes = fabric_guard.check_rccl_env(env, _ib([]))
        self.assertEqual(len(notes), 2)
        self.assertIn("NET_OPTIONAL_RECV_COMPLETION=0 has no effect: RCCL reads NCCL_NET_OPTIONAL_RECV_COMPLETION", notes[0])
        self.assertNotIn("NCCL_NET_OPTIONAL_RECV_COMPLETION", env)
        env = {"NET_OPTIONAL_RECV_COMPLETION": "1", "NCCL_NET_OPTIONAL_RECV_COMPLETION": "1"}
        self.assertEqual(fabric_guard.check_rccl_env(env, _ib([])), [])


if __name__ == "__main__":
    unittest.main()

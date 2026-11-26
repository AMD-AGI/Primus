#!/usr/bin/env python3
# pyre-strict
"""ITEM (64) (B1''): the node-count
defaults of its four switches.

The gin binds them True. Part (a), DLRM_HSTU_FP8_SEQ_A2A_FUSED, binds through
train/_env_bootstrap.py _bind_bool: "1" at every node count, one node
included. Parts (b)-(d) bind through _bind_bool_multinode, which turns True
into "1" only when the job spans more than one node (NNODES, else
SLURM_NNODES; the rule of MNC's host_sync_free._spans_nodes) and into "0" at
one node. A pre-set env var wins at any node count. CPU only: no process
group, no GPU, no gin parse (the env-wide replay is a separate check)."""

import ast
import os
import re
import unittest
from typing import Dict, Optional
from unittest import mock

from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

SWITCHES = (
    "DLRM_HSTU_FP8_SEQ_A2A_FUSED",
    "DLRM_HSTU_FP8_KJT_RECAT_FUSED",
    "DLRM_HSTU_FP8_TBE_REBAG",
    "DLRM_HSTU_FP8_TBE_BOUNDS_TRITON",
)
EVERY_NODE_COUNT = SWITCHES[:1]  # (a): _bind_bool
MULTINODE = SWITCHES[1:]  # (b)-(d): _bind_bool_multinode
_KEYS = SWITCHES + ("NNODES", "SLURM_NNODES", "LOCAL_WORLD_SIZE")
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MultiNodeDefaultTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in _KEYS}
        for k in _KEYS:
            os.environ.pop(k, None)
        # no process group in a unit test: the env alone decides
        self._group = mock.patch.object(eb, "_group_spans_nodes", return_value=None)
        self._group.start()

    def tearDown(self) -> None:
        self._group.stop()
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _bind(self, value: Optional[bool], **env: str) -> Optional[str]:
        for k in ("NNODES", "SLURM_NNODES", MULTINODE[0]):
            os.environ.pop(k, None)
        os.environ.update(env)
        eb._bind_bool_multinode(MULTINODE[0], value)
        return os.environ.get(MULTINODE[0])

    def test_spans_nodes_rule(self) -> None:
        cases = [
            ({"NNODES": "2"}, True),
            ({"NNODES": "4"}, True),
            ({"NNODES": "1"}, False),
            ({"SLURM_NNODES": "2"}, True),
            ({"SLURM_NNODES": "1"}, False),
            ({"NNODES": "1", "SLURM_NNODES": "4"}, False),  # NNODES decides first
            ({"NNODES": "x", "SLURM_NNODES": "2"}, True),  # unparsable: next key
            ({"NNODES": " ", "SLURM_NNODES": "2"}, True),  # blank: next key
            ({}, False),  # neither and no group: single node
        ]
        for env, want in cases:
            for k in ("NNODES", "SLURM_NNODES"):
                os.environ.pop(k, None)
            os.environ.update(env)
            self.assertEqual(eb._spans_nodes(), want, env)

    def test_group_fallback_only_without_env(self) -> None:
        with mock.patch.object(eb, "_group_spans_nodes", return_value=True):
            self.assertTrue(eb._spans_nodes())
            os.environ["NNODES"] = "1"
            self.assertFalse(eb._spans_nodes())

    def test_gin_true_is_on_only_across_nodes(self) -> None:
        self.assertEqual(self._bind(True, NNODES="2"), "1")
        self.assertEqual(self._bind(True, NNODES="4"), "1")
        self.assertEqual(self._bind(True, SLURM_NNODES="2"), "1")
        self.assertEqual(self._bind(True, NNODES="1"), "0")
        self.assertEqual(self._bind(True), "0")

    def test_gin_false_and_unbound(self) -> None:
        self.assertEqual(self._bind(False, NNODES="2"), "0")
        self.assertEqual(self._bind(False, NNODES="1"), "0")
        self.assertIsNone(self._bind(None, NNODES="2"))

    def test_preset_env_wins(self) -> None:
        for preset in ("1", "0", ""):
            for nn in ("1", "2"):
                got = self._bind(True, NNODES=nn, **{MULTINODE[0]: preset})
                self.assertEqual(got, preset, (preset, nn))

    def test_log_lines(self) -> None:
        with self.assertLogs(eb.logger, "INFO") as cm:
            self._bind(True, NNODES="1")
            self._bind(True, NNODES="2")
            self._bind(True, NNODES="1", **{MULTINODE[0]: "1"})
        text = "\n".join(cm.output)
        self.assertIn(
            "env bootstrap: DLRM_HSTU_FP8_KJT_RECAT_FUSED=0 (multi-node default: "
            "single node, off)",
            text,
        )
        self.assertIn(
            "env bootstrap: DLRM_HSTU_FP8_KJT_RECAT_FUSED=1 (multi-node default)",
            text,
        )
        self.assertIn(
            "honoring pre-set DLRM_HSTU_FP8_KJT_RECAT_FUSED=1 (overrides gin "
            "binding)",
            text,
        )

    def test_seq_a2a_fused_is_on_at_one_node(self) -> None:
        # Part (a) binds through _bind_bool: gin True -> "1" at every node
        # count (the 1n marker "env bootstrap: DLRM_HSTU_FP8_SEQ_A2A_FUSED=1");
        # a pre-set "0" is still its kill switch.
        name = EVERY_NODE_COUNT[0]
        for env in ({"NNODES": "1"}, {"SLURM_NNODES": "1"}, {}, {"NNODES": "4"}):
            for k in ("NNODES", "SLURM_NNODES", name):
                os.environ.pop(k, None)
            os.environ.update(env)
            with self.assertLogs(eb.logger, "INFO") as cm:
                eb._bind_bool(name, True)
            self.assertEqual(os.environ.get(name), "1", env)
            self.assertIn("env bootstrap: %s=1" % name, "\n".join(cm.output))
        os.environ["NNODES"] = "1"
        os.environ[name] = "0"
        eb._bind_bool(name, True)
        self.assertEqual(os.environ.get(name), "0")

    def test_bootstrap_binds_seq_a2a_everywhere_and_three_multinode(self) -> None:
        src = open(os.path.join(_TRAIN, "_env_bootstrap.py")).read()
        fn = [
            n
            for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap"
        ][0]
        params = {a.arg for a in fn.args.args}
        calls: Dict[str, str] = {}
        for n in ast.walk(fn):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.args
                and isinstance(n.args[0], ast.Constant)
                and n.args[0].value in SWITCHES
            ):
                calls[n.args[0].value] = n.func.id
                self.assertIsInstance(n.args[1], ast.Name)
                self.assertEqual(n.args[1].id, n.args[0].value)
        want = {s: "_bind_bool" for s in EVERY_NODE_COUNT}
        want.update({s: "_bind_bool_multinode" for s in MULTINODE})
        self.assertEqual(calls, want)
        self.assertTrue(set(SWITCHES) <= params)

    def test_gin_binds_true_once_with_item_text(self) -> None:
        # Anchored on the four bindings, not on the item's title or number, so
        # a later renumbering or retitling of the block does not break it.
        lines = open(os.path.join(_TRAIN, "gin", "yambda_5b.gin")).read().splitlines()
        at: Dict[str, int] = {}
        for s in SWITCHES:
            pat = re.compile(r"^apply_env_bootstrap\.%s\s*=\s*(\S+)\s*$" % s)
            hits = [(i, m.group(1)) for i, m in enumerate(map(pat.match, lines)) if m]
            self.assertEqual([v for _i, v in hits], ["True"], s)
            at[s] = hits[0][0]
        first, last = min(at.values()), max(at.values())
        # the item's text: from the separator line above its first binding
        start = max(i for i in range(first) if lines[i].startswith("# ----"))
        block = lines[start : last + 1]
        self.assertTrue(all(len(line) <= 80 for line in block))
        text = "\n".join(block)
        for marker in ("B1''", "=0 is its kill switch", "MUST BE ABSENT", "VERIFY", "ARMED"):
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()

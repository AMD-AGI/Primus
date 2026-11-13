#!/usr/bin/env python3
# pyre-unsafe
"""yambda_5b.gin item (61): the train pools on item (32)'s batched builder
($DL_FAST_BATCH, gin default 1) and its precondition, item (37)'s single
chained train pool, as fast_batch_pools resolves them.

* the shipped bindings of items (32), (61) and (37) (read from yambda_5b.gin
  itself), env unset: the train wrappers build with build_batch and each
  rank's train-pool line names the gin default; DL_FAST_BATCH=0 (env) is the
  kill switch; DL_FAST_BATCH=1 (env) is applied as given;
* the precondition: when item (37)'s single pool resolves off
  (DL_SINGLE_POOL <= 0 in the env, a gin override of its default, or no item
  (37) binding at all: whatever window_single_pool(), the loader's own
  switch, returns) and DL_FAST_BATCH is unset or empty, the train pools keep
  the per-sample path and each rank's train-pool line says why, naming the
  env value of DL_SINGLE_POOL or, when it is unset, the gin config; a
  DL_FAST_BATCH set in the env wins (1 = fast, 0 = per-sample, no hold);
  eval loaders that follow DL_FAST_BATCH (DL_FAST_BATCH_EVAL=-1) follow the
  held value, the others keep item (32)'s own switch;
* every fast_batch_pools call recomputes the hold (no stale reason);
* without the gin item (no fast_batch_pools binding) the env-only code
  default is unchanged and window_single_pool is not consulted.
The production pre-RUN_START path with the two pools (DL_SINGLE_POOL=0) is
covered by window_single_pool_dlstart_test and onestore_item49_test.
CPU only (synthetic yambda store from fast_batch_test)."""

import os
import unittest
from unittest import mock

import gin
import numpy as np

from generative_recommenders.dlrm_v4.datasets import dataset as dsmod
from generative_recommenders.dlrm_v4.datasets.dataset import collate_fn, Samples
from generative_recommenders.dlrm_v4.train import utils as tu
from generative_recommenders.dlrm_v4.train.tests import fast_batch_eval_test as _fbe
from generative_recommenders.dlrm_v4.train.tests.fast_batch_test import (
    _cmp_samples,
    _make_ds,
    _make_store,
)

# Every switch the cases set: unset unless a case sets it.
_KNOBS = ("DL_FAST_BATCH", "DL_FAST_BATCH_EVAL", "DL_SINGLE_POOL")
_FAST = (
    "armed OK: [fast-batch] train pools: fast builder (build_batch; "
    "DL_FAST_BATCH=1, %s) rank=%d"
)
_SLOW = "[fast-batch] train pools: per-sample path (DL_FAST_BATCH=0, %s) rank=%d"
_EVAL_FAST = (
    "armed OK: [fast-batch] eval pools: fast builder (build_batch; "
    "DL_FAST_BATCH_EVAL=1, gin default) rank=%d"
)
# item (61)'s hold: the why when DL_SINGLE_POOL is not set in the env
_WHY_GIN = "DL_SINGLE_POOL unset: its gin setting or no item (37) binding"


def _held(single_env=None) -> str:
    """Item (61)'s hold reason; ``single_env`` = DL_SINGLE_POOL in the env
    (None or empty = unset: the reason names the gin config instead)."""
    why = "env DL_SINGLE_POOL=%s" % single_env if single_env else _WHY_GIN
    return (
        "gin default 1 held because item (37)'s single train pool is off: "
        "window_single_pool() False, %s; env DL_FAST_BATCH=1 overrides" % why
    )


def _env(**kv):
    """os.environ without the switches above, plus ``kv`` (None = unset)."""
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update({k: v for k, v in kv.items() if v is not None})
    return mock.patch.dict(os.environ, env, clear=True)


def _gin_lines(prefixes, n: int) -> str:
    lines = [ln for ln in _fbe._GIN.read_text().splitlines() if ln.startswith(prefixes)]
    assert len(lines) == n, lines
    return "\n".join(lines) + "\n"


def _items32_58_gin() -> str:
    """Items (32) and (61): the fast_batch_pools bindings as yambda_5b.gin
    ships them (eval_pools: DL_FAST_BATCH_EVAL, train_pools: DL_FAST_BATCH)."""
    return _gin_lines(("fast_batch_pools.", "dlfb/", "dlfbe/"), 6)


def _item37_gin() -> str:
    """Item (37)'s single chained train pool binding, as shipped."""
    return _gin_lines(("window_single_pool.", "dlsp/"), 3)


class _Gin:
    """Context: ``text`` parsed into an empty gin config (cleared on exit)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __enter__(self) -> None:
        gin.clear_config()
        if self._text:
            with gin.unlock_config():
                gin.parse_config(self._text)

    def __exit__(self, *exc) -> None:
        gin.clear_config()


def _msgs(cm):
    return [r.split(":", 2)[2] for r in cm.output]


class FastBatchTrainDefaultTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.store, cls.positions, cls.i2a, cls.i2al = _make_store()
        rng = np.random.default_rng(13)
        cls.batch = sorted(rng.choice(len(cls.positions), 40, replace=False).tolist())

    def setUp(self) -> None:
        # Module state fast_batch_pools and the lines write: fresh per test,
        # restored afterwards.
        self._patches = [_fbe._counting(n) for n in _fbe._NAMES] + [
            mock.patch.object(dsmod, "_FAST_BATCH", None),
            mock.patch.object(dsmod, "_FAST_BATCH_EVAL", None),
            mock.patch.object(dsmod, "_FAST_BATCH_FROM_GIN", {"train": False, "eval": False}),
            mock.patch.object(tu, "_FAST_BATCH_TRAIN_HELD", None),
        ]
        for p in self._patches:
            p.start()
        for v in _fbe._CNT.values():
            v.value = 0
        tu._FAST_BATCH_POOLS_LOGGED.clear()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        tu._FAST_BATCH_POOLS_LOGGED.clear()

    def _ds(self):
        return _make_ds(self.store, self.positions, self.i2a, self.i2al)

    def _resolve(self, rank: int = 0, eval_follow: bool = False):
        """fast_batch_pools(), then each rank's lines for a train wrapper and
        an eval wrapper built as the loop builds them. Returns (train switch,
        hold reason, eval wrapper setting, lines)."""
        tu._FAST_BATCH_POOLS_LOGGED.clear()
        with mock.patch.object(dsmod, "_FAST_BATCH", None), mock.patch.object(
            dsmod, "_FAST_BATCH_EVAL", None
        ):
            tu.fast_batch_pools()
            ds = self._ds()
            train = tu.HammerToTorchDataset(ds)
            ev = tu._eval_dataset(train)
            with self.assertLogs(tu.logger, level="WARNING") as cm:
                tu._log_fast_batch_pools(rank, {"train": train, "eval": ev})
            return (
                dsmod.fast_batch_enabled(),
                tu._FAST_BATCH_TRAIN_HELD,
                getattr(ev, "fast_batch", None) if ev is not train else "follows",
                _msgs(cm),
            )

    def test_shipped_resolution_matrix(self) -> None:
        """Items (32) + (61) + (37) as shipped: the train switch, the hold and
        each rank's lines for every combination of DL_SINGLE_POOL and
        DL_FAST_BATCH in the env; the hold agrees with window_single_pool(),
        the switch the loader itself reads."""
        cases = (
            # (DL_SINGLE_POOL, DL_FAST_BATCH) -> (train on, held, train line)
            ((None, None), (True, False, _FAST % ("gin default", 4))),
            ((None, "0"), (False, False, _SLOW % ("env", 4))),
            ((None, "1"), (True, False, _FAST % ("env", 4))),
            (("1", None), (True, False, _FAST % ("gin default", 4))),
            (("2", None), (True, False, _FAST % ("gin default", 4))),
            (("0", None), (False, True, _SLOW % (_held("0"), 4))),
            (("0", ""), (False, True, _SLOW % (_held("0"), 4))),  # empty = unset
            (("0", "1"), (True, False, _FAST % ("env", 4))),
            (("0", "0"), (False, False, _SLOW % ("env", 4))),
            # any value < 1 turns item (37) off; the line names the value
            (("-1", None), (False, True, _SLOW % (_held("-1"), 4))),
            (("-1", "1"), (True, False, _FAST % ("env", 4))),
        )
        with _Gin(_items32_58_gin() + _item37_gin()):
            for (sp, fb), (on, held, line) in cases:
                with self.subTest(DL_SINGLE_POOL=sp, DL_FAST_BATCH=fb), _env(
                    DL_SINGLE_POOL=sp, DL_FAST_BATCH=fb
                ):
                    single = tu.window_single_pool()
                    self.assertEqual(single, sp is None or int(sp) > 0)
                    got_on, reason, ev_setting, lines = self._resolve(rank=4)
                    self.assertIs(got_on, on)
                    self.assertEqual(reason is not None, held)
                    self.assertEqual(held, not single and not fb)
                    if held:
                        self.assertEqual(reason, _held(sp))
                    # the eval pools keep item (32)'s own switch
                    self.assertIs(ev_setting, True)
                    self.assertEqual(lines, [line, _EVAL_FAST % 4])

    def test_hold_follows_the_loader_switch(self) -> None:
        """The hold reads item (37)'s switch exactly as the loader does: a gin
        override of its default and a gin without item (37)'s binding (the
        loader then builds the two pools as well) hold the train pools like
        DL_SINGLE_POOL=0, and the line names the gin config (DL_SINGLE_POOL
        unset) or the env value as given (a gin without item (37)'s binding
        does not read DL_SINGLE_POOL=1); a set DL_FAST_BATCH still wins."""
        override = "dlsp/env_int.default = 0\n"
        for text, sp, single_want in (
            (_items32_58_gin() + _item37_gin() + override, None, False),
            (_items32_58_gin(), None, False),
            (_items32_58_gin() + _item37_gin() + override, "1", True),
            (_items32_58_gin(), "1", False),
        ):
            with self.subTest(gin_text=text[-40:], DL_SINGLE_POOL=sp), _Gin(text):
                with _env(DL_SINGLE_POOL=sp):
                    single = tu.window_single_pool()
                    on, reason, _, lines = self._resolve()
                self.assertIs(single, single_want)
                self.assertIs(on, single)
                self.assertEqual(reason, None if single else _held(sp))
                self.assertEqual(
                    lines[0],
                    _FAST % ("gin default", 0) if single else _SLOW % (_held(sp), 0),
                )
                with _env(DL_SINGLE_POOL=sp, DL_FAST_BATCH="1"):
                    on, reason, _, lines = self._resolve()
                self.assertTrue(on)
                self.assertIsNone(reason)
                self.assertEqual(lines[0], _FAST % ("env", 0))

    def test_eval_follow_under_hold(self) -> None:
        """DL_FAST_BATCH_EVAL=-1 (eval loaders follow DL_FAST_BATCH) with the
        hold: the eval wrapper follows the held value and its line carries
        the reason; DL_FAST_BATCH_EVAL=1 keeps the eval builder."""
        with _Gin(_items32_58_gin() + _item37_gin()):
            with _env(DL_SINGLE_POOL="0", DL_FAST_BATCH_EVAL="-1"):
                on, reason, ev_setting, lines = self._resolve()
            self.assertFalse(on)
            self.assertEqual(reason, _held("0"))
            self.assertEqual(ev_setting, "follows")
            self.assertEqual(
                lines,
                [
                    _SLOW % (_held("0"), 0),
                    "[fast-batch] eval pools: per-sample path (DL_FAST_BATCH_EVAL unset "
                    "or < 0: follows DL_FAST_BATCH=0, %s) rank=0" % _held("0"),
                ],
            )
            with _env(DL_SINGLE_POOL="0", DL_FAST_BATCH_EVAL="1"):
                on, reason, ev_setting, lines = self._resolve()
            self.assertFalse(on)
            self.assertIs(ev_setting, True)
            self.assertEqual(
                lines,
                [
                    _SLOW % (_held("0"), 0),
                    "armed OK: [fast-batch] eval pools: fast builder (build_batch; "
                    "DL_FAST_BATCH_EVAL=1, env) rank=0",
                ],
            )

    def test_hold_recomputed_per_call(self) -> None:
        """fast_batch_pools recomputes the hold on every call (idempotent for
        the same gin + env; no stale reason once the single pool is back)."""
        with _Gin(_items32_58_gin() + _item37_gin()):
            with _env(DL_SINGLE_POOL="0"):
                first = self._resolve()
                second = self._resolve()
            self.assertEqual(first, second)
            self.assertEqual(first[1], _held("0"))
            with _env():
                on, reason, _, lines = self._resolve()
            self.assertTrue(on)
            self.assertIsNone(reason)
            self.assertEqual(lines[0], _FAST % ("gin default", 0))

    def test_wrappers_build_as_resolved(self) -> None:
        """The train wrapper's fetch follows the resolved switch: build_batch
        with the single pool (gin default) and with DL_FAST_BATCH=1 over the
        two pools, the per-sample list path under the hold; bit-identical to
        the per-sample reference in every case."""
        ds = self._ds()
        n = len(self.batch)

        def per_sample_ref():
            # fresh per comparison: comparing fills the KJTs' lazy caches
            with mock.patch.object(dsmod, "_FAST_BATCH", False), _env():
                return collate_fn(tu.HammerToTorchDataset(ds).__getitems__(self.batch))

        cases = (
            # (DL_SINGLE_POOL, DL_FAST_BATCH) -> train wrapper uses build_batch
            ((None, None), True),
            (("0", None), False),
            (("0", "1"), True),
        )
        with _Gin(_items32_58_gin() + _item37_gin()):
            for (sp, fb), fast in cases:
                with self.subTest(DL_SINGLE_POOL=sp, DL_FAST_BATCH=fb), _env(
                    DL_SINGLE_POOL=sp, DL_FAST_BATCH=fb
                ), mock.patch.object(dsmod, "_FAST_BATCH", None):
                    ref = per_sample_ref()
                    tu.fast_batch_pools()
                    before = _fbe._counts()
                    got = tu.HammerToTorchDataset(ds).__getitems__(self.batch)
                    after = _fbe._counts()
                    c = {k: after[k] - before[k] for k in _fbe._NAMES}
                    if fast:
                        self.assertIsInstance(got, Samples)
                        self.assertEqual(c, {"build_batch": 1, "_build_sample": 0, "_build_sample_np": 0})
                    else:
                        self.assertIsInstance(got, list)
                        self.assertEqual(c, {"build_batch": 0, "_build_sample": n, "_build_sample_np": 0})
                    self.assertEqual(_cmp_samples(ref, [collate_fn(got)], f"{sp}/{fb}"), [])

    def test_no_gin_binding(self) -> None:
        """No fast_batch_pools binding (tests, tools that do not parse
        yambda_5b.gin): the env-only code default (off) and DL_FAST_BATCH=1
        are unchanged, and the precondition is not consulted."""
        calls = []

        def probe():
            calls.append(1)
            return False

        with _Gin(""), mock.patch.object(tu, "window_single_pool", probe):
            for fb, on in ((None, False), ("1", True), ("0", False)):
                with self.subTest(DL_FAST_BATCH=fb), _env(
                    DL_SINGLE_POOL="0", DL_FAST_BATCH=fb
                ), mock.patch.object(dsmod, "_FAST_BATCH", None):
                    tu.fast_batch_pools()
                    self.assertIs(dsmod.fast_batch_enabled(), on)
                    self.assertIsNone(tu._FAST_BATCH_TRAIN_HELD)
                    self.assertEqual(
                        dsmod.fast_batch_knob_source("train"),
                        "env" if fb else "code default",
                    )
        self.assertEqual(calls, [])

    def test_gin_block_and_binding(self) -> None:
        """The shipped text: item (61)'s binding with default 1 on the same
        key as before, item (37)'s default 1, and the block naming the
        precondition, the kill switch, the hold line this code prints (both
        forms of its why) and the loader shapes the precondition does not
        hold."""
        txt = _fbe._GIN.read_text()
        self.assertIn(
            '\nfast_batch_pools.train_pools = @dlfb/env_int()\n'
            'dlfb/env_int.key = "DL_FAST_BATCH"\n'
            "dlfb/env_int.default = 1\n",
            txt,
        )
        self.assertIn('\ndlsp/env_int.key     = "DL_SINGLE_POOL"\ndlsp/env_int.default = 1\n', txt)
        start = txt.index("# (61) TRAIN POOLS:")
        block = txt[start: txt.index("\nfast_batch_pools.train_pools", start)]
        for s in ("WHAT:", "WHY:", "IMPACT:", "NUMERICS:", "MLPERF:", "ARMED",
                  "PRECONDITION:", "Kill switch: DL_FAST_BATCH=0"):
            self.assertIn(s, block)
        # the hold line, as the block quotes it (comment lines re-joined)
        flat = " ".join(ln.lstrip("# ").strip() for ln in block.splitlines())
        self.assertIn((_SLOW % (_held("0"), 0)).replace("rank=0", "rank=R"), flat)
        self.assertIn(_WHY_GIN, flat)
        for s in ("PERSISTENT_LOADER=0", "DOUBLE_BUFFER=0"):
            self.assertIn(s, flat)


if __name__ == "__main__":
    unittest.main()

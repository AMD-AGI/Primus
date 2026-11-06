#!/usr/bin/env python3
# pyre-unsafe
"""$DL_FAST_BATCH_EVAL: the eval loaders' own fast-batch setting (yambda_5b.gin
item (32): gin default 1, kill switch DL_FAST_BATCH_EVAL=0), next to the train
pools' DL_FAST_BATCH (item (61): gin default 1 with item (37)'s single chained
pool, kill switch DL_FAST_BATCH=0; its precondition is covered by
fast_batch_train_default_test).

DL_FAST_BATCH=0 DL_FAST_BATCH_EVAL=1: the eval pool builds its batches with
build_batch while the train pools stay on the per-sample Python-list path.
DL_FAST_BATCH=1 DL_FAST_BATCH_EVAL=0: the reverse, and the eval per-sample path
is the old list builder, not the numpy one. Without the gin item and with
DL_FAST_BATCH_EVAL unset (or < 0): the eval pool follows DL_FAST_BATCH exactly as
before. Batches are bit-identical in every case.

Also covers the knob parsing, the gin wiring (the item (32) and (61) bindings
taken from yambda_5b.gin itself, parsed with item (37)'s single-pool binding:
defaults, env override, kill switches, follow), the scope restore, the
per-kind arming lines, the launcher forwarding, and real persistent fork pools
built the way prestart_streaming_loaders builds them (eval pool re-armed by
iter() per pass, every batch rebuilt each pass).
CPU only (synthetic yambda store from fast_batch_test)."""

import contextlib
import multiprocessing as mp
import os
import pathlib
import unittest
from unittest import mock

import gin
import numpy as np

from generative_recommenders.dlrm_v4.datasets import dataset as dsmod
from generative_recommenders.dlrm_v4.datasets.dataset import collate_fn, Samples
from generative_recommenders.dlrm_v4.datasets.yambda import DLRMv4YambdaDataset
from generative_recommenders.dlrm_v4.train import cold_prefetch as cp
from generative_recommenders.dlrm_v4.train import dl_prestart as dlp
from generative_recommenders.dlrm_v4.train import utils as tu
from generative_recommenders.dlrm_v4.train.tests.fast_batch_test import (
    _cmp_samples,
    _make_ds,
    _make_store,
)

# Call counters shared with forked DataLoader workers.
_NAMES = ("build_batch", "_build_sample", "_build_sample_np")
_CNT = {n: mp.get_context("fork").Value("i", 0) for n in _NAMES}
_KNOBS = ("DL_FAST_BATCH", "DL_FAST_BATCH_EVAL")
# Cleared from the environment by _env: the knobs above and item (37)'s
# switch, which item (61)'s train default depends on.
_CLEARED = _KNOBS + ("DL_SINGLE_POOL",)
# recommendation/ (this file: recommendation/generative_recommenders/dlrm_v4/train/tests/)
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GIN = _REPO / "generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin"
_LAUNCHER = _REPO / "scripts/launch_slurm.sh"


def _counting(name):
    orig = getattr(DLRMv4YambdaDataset, name)

    def wrapper(self, *args, **kwargs):
        with _CNT[name].get_lock():
            _CNT[name].value += 1
        return orig(self, *args, **kwargs)

    return mock.patch.object(DLRMv4YambdaDataset, name, wrapper)


def _counts():
    return {n: _CNT[n].value for n in _NAMES}


def _env(eval_knob, train_knob=None):
    """os.environ with DL_FAST_BATCH_EVAL / DL_FAST_BATCH set as given (None =
    unset) and DL_SINGLE_POOL unset."""
    env = {k: v for k, v in os.environ.items() if k not in _CLEARED}
    if eval_knob is not None:
        env["DL_FAST_BATCH_EVAL"] = eval_knob
    if train_knob is not None:
        env["DL_FAST_BATCH"] = train_knob
    return mock.patch.dict(os.environ, env, clear=True)


def _item32_gin() -> str:
    """The item (32) and (61) bindings (fast_batch_pools: eval_pools, then
    train_pools) exactly as yambda_5b.gin ships them."""
    lines = [
        ln
        for ln in _GIN.read_text().splitlines()
        if ln.startswith(("fast_batch_pools.", "dlfb/", "dlfbe/"))
    ]
    assert len(lines) == 6, lines
    return "\n".join(lines) + "\n"


def _item37_gin() -> str:
    """Item (37)'s single chained train pool binding exactly as yambda_5b.gin
    ships it (item (61)'s precondition: with it the train pools take their
    gin default)."""
    lines = [
        ln
        for ln in _GIN.read_text().splitlines()
        if ln.startswith(("window_single_pool.", "dlsp/"))
    ]
    assert len(lines) == 3, lines
    return "\n".join(lines) + "\n"


class FastBatchEvalTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.store, cls.positions, cls.i2a, cls.i2al = _make_store()
        rng = np.random.default_rng(9)
        cls.batch = sorted(rng.choice(len(cls.positions), 48, replace=False).tolist())

    def setUp(self) -> None:
        self._patches = [_counting(n) for n in _NAMES] + [
            # Module state configure_fast_batch writes: fresh per test.
            mock.patch.object(dsmod, "_FAST_BATCH_EVAL", None),
            mock.patch.object(dsmod, "_FAST_BATCH_FROM_GIN", {"train": False, "eval": False}),
            mock.patch.object(tu, "_FAST_BATCH_TRAIN_HELD", None),
        ]
        for p in self._patches:
            p.start()
        for v in _CNT.values():
            v.value = 0
        tu._FAST_BATCH_POOLS_LOGGED.clear()
        dlp._reset_for_tests()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        tu._FAST_BATCH_POOLS_LOGGED.clear()

    def _ds(self) -> DLRMv4YambdaDataset:
        return _make_ds(self.store, self.positions, self.i2a, self.i2al)

    def _ref(self, ds) -> Samples:
        with mock.patch.object(dsmod, "_FAST_BATCH", False), _env(None):
            return collate_fn(tu.HammerToTorchDataset(ds).__getitems__(self.batch))

    def _pools(self, ds):
        """(base, train, eval) wrappers as prestart_streaming_loaders builds them."""
        base = tu.HammerToTorchDataset(ds)
        train = tu._ReadProbedDataset(base, dlp.new_read_slot("train"))
        ev = tu._eval_dataset(base, dlp.new_read_slot("eval"))
        return base, train, ev

    def _fetch(self, wrapper):
        before = _counts()
        out = wrapper.__getitems__(self.batch)
        after = _counts()
        return out, {n: after[n] - before[n] for n in _NAMES}

    def _gin(self):
        """Context: the shipped item (32)/(61) bindings and item (37)'s
        single-pool binding parsed (cleared on exit)."""

        class _Ctx:
            def __enter__(self):
                gin.clear_config()
                with gin.unlock_config():
                    gin.parse_config(_item32_gin() + _item37_gin())

            def __exit__(self, *exc):
                gin.clear_config()

        return _Ctx()

    def test_knob_parsing(self) -> None:
        for raw, want in ((None, None), ("", None), (" ", None), ("0", False),
                          ("1", True), ("2", True), ("-1", None), ("x", False)):
            with _env(raw):
                self.assertEqual(dsmod.eval_fast_batch(), want, raw)

    def test_gin_resolution(self) -> None:
        """The shipped gin text: eval and train on by default (items (32) and
        (61), single pool), env wins, 0 is each kill switch, -1 follows
        DL_FAST_BATCH; values reach the wrappers."""
        cases = (
            # (DL_FAST_BATCH_EVAL env, DL_FAST_BATCH env) -> (train, eval, eval src)
            ((None, None), (True, True, "gin default")),
            (("0", None), (True, False, "env")),
            (("1", None), (True, True, "env")),
            ((None, "1"), (True, True, "gin default")),
            ((None, "0"), (False, True, "gin default")),
            (("0", "1"), (True, False, "env")),
            (("0", "0"), (False, False, "env")),
            (("-1", "1"), (True, None, "env")),
            (("-1", "0"), (False, None, "env")),
            (("-1", None), (True, None, "env")),
        )
        ds = self._ds()
        with self._gin():
            for (ev_env, tr_env), (tr, ev, src) in cases:
                with self.subTest(DL_FAST_BATCH_EVAL=ev_env, DL_FAST_BATCH=tr_env), \
                        mock.patch.object(dsmod, "_FAST_BATCH", None), \
                        mock.patch.object(dsmod, "_FAST_BATCH_EVAL", None), \
                        _env(ev_env, tr_env):
                    tu.fast_batch_pools()
                    self.assertIs(dsmod.fast_batch_enabled(), tr)
                    self.assertIs(dsmod.eval_fast_batch(), ev)
                    self.assertEqual(dsmod.fast_batch_knob_source("eval"), src)
                    self.assertEqual(
                        dsmod.fast_batch_knob_source("train"),
                        "gin default" if tr_env is None else "env",
                    )
                    self.assertIsNone(tu._FAST_BATCH_TRAIN_HELD)
                    _, train, evw = self._pools(ds)
                    self.assertIsNone(train.fast_batch)
                    self.assertIs(evw.fast_batch, ev)
                    view = tu._eval_dataset(tu.HammerToTorchDataset(ds))
                    self.assertIs(view.fast_batch, ev)
        # No gin binding: the env-only code default (eval follows DL_FAST_BATCH).
        with mock.patch.object(dsmod, "_FAST_BATCH", None), \
                mock.patch.object(dsmod, "_FAST_BATCH_EVAL", None), \
                mock.patch.object(dsmod, "_FAST_BATCH_FROM_GIN", {"train": False, "eval": False}), \
                _env(None):
            gin.clear_config()
            tu.fast_batch_pools()
            self.assertFalse(dsmod.fast_batch_enabled())
            self.assertIsNone(dsmod.eval_fast_batch())
            self.assertEqual(dsmod.fast_batch_knob_source("eval"), "code default")

    def test_gin_defaults_fetch_and_arming(self) -> None:
        """Gin defaults, env unset: eval and train wrappers both call
        build_batch (items (32) and (61)), all bit-identical to the per-sample
        reference; the eval scope is restored; the per-rank lines."""
        ds = self._ds()
        ref = self._ref(ds)
        with self._gin(), mock.patch.object(dsmod, "_FAST_BATCH", None), _env(None):
            tu.fast_batch_pools()
            base, train, ev = self._pools(ds)
            got_eval, c_eval = self._fetch(ev)
            got_train, c_train = self._fetch(train)
            view = tu._eval_dataset(base)
            got_view, c_view = self._fetch(view)
            self.assertTrue(dsmod.fast_batch_enabled())
            self.assertIsNone(getattr(dsmod._FAST_BATCH_SCOPE, "on", None), "scope leaked")
            with self.assertLogs(tu.logger, level="WARNING") as cm:
                tu._log_fast_batch_pools(0, {"train": train, "eval": ev})
        self.assertEqual(c_eval, {"build_batch": 1, "_build_sample": 0, "_build_sample_np": 0})
        self.assertEqual(c_view, c_eval)
        self.assertEqual(c_train, c_eval)
        self.assertIsInstance(got_eval, Samples)
        self.assertIsInstance(got_train, Samples)
        self.assertEqual(
            _cmp_samples(ref, [got_eval, got_train, got_view], "gin_default"), []
        )
        self.assertEqual(
            cm.output,
            [
                "WARNING:generative_recommenders.dlrm_v4.train.utils:armed OK: [fast-batch] "
                "train pools: fast builder (build_batch; DL_FAST_BATCH=1, gin default) rank=0",
                "WARNING:generative_recommenders.dlrm_v4.train.utils:armed OK: [fast-batch] "
                "eval pools: fast builder (build_batch; DL_FAST_BATCH_EVAL=1, gin default) "
                "rank=0",
            ],
        )

    def test_gin_kill_switch(self) -> None:
        """DL_FAST_BATCH_EVAL=0 over the gin default: the eval wrappers run the
        old per-sample list path (their scope overrides the train pools' fast
        builder); the per-rank line says so."""
        ds = self._ds()
        ref = self._ref(ds)
        n = len(self.batch)
        with self._gin(), mock.patch.object(dsmod, "_FAST_BATCH", None), _env("0"):
            tu.fast_batch_pools()
            base, train, ev = self._pools(ds)
            self.assertIs(ev.fast_batch, False)
            got_eval, c_eval = self._fetch(ev)
            got_view, c_view = self._fetch(tu._eval_dataset(base))
            self.assertTrue(dsmod.fast_batch_enabled(), "scope leaked")
            with self.assertLogs(tu.logger, level="WARNING") as cm:
                tu._log_fast_batch_pools(0, {"train": train, "eval": ev})
        for c in (c_eval, c_view):
            self.assertEqual(c, {"build_batch": 0, "_build_sample": n, "_build_sample_np": 0})
        self.assertEqual(_cmp_samples(ref, [collate_fn(got_eval), collate_fn(got_view)], "kill"), [])
        self.assertEqual(len(cm.output), 2, cm.output)
        self.assertIn(
            "armed OK: [fast-batch] train pools: fast builder (build_batch; DL_FAST_BATCH=1, "
            "gin default) rank=0",
            cm.output[0],
        )
        self.assertNotIn("armed OK", cm.output[1])
        self.assertIn(
            "[fast-batch] eval pools: per-sample path (DL_FAST_BATCH_EVAL=0, env)", cm.output[1]
        )

    def test_gin_train_kill_switch(self) -> None:
        """DL_FAST_BATCH=0 over item (61)'s gin default: the train wrappers run
        the old per-sample list path (bit-identical); the eval wrappers keep
        item (32)'s builder; the per-rank lines say so."""
        ds = self._ds()
        ref = self._ref(ds)
        n = len(self.batch)
        with self._gin(), mock.patch.object(dsmod, "_FAST_BATCH", None), _env(None, "0"):
            tu.fast_batch_pools()
            base, train, ev = self._pools(ds)
            self.assertFalse(dsmod.fast_batch_enabled())
            got_train, c_train = self._fetch(train)
            got_eval, c_eval = self._fetch(ev)
            self.assertFalse(dsmod.fast_batch_enabled(), "scope leaked")
            with self.assertLogs(tu.logger, level="WARNING") as cm:
                tu._log_fast_batch_pools(0, {"train": train, "eval": ev})
        self.assertEqual(c_train, {"build_batch": 0, "_build_sample": n, "_build_sample_np": 0})
        self.assertEqual(c_eval, {"build_batch": 1, "_build_sample": 0, "_build_sample_np": 0})
        self.assertIsInstance(got_train, list)
        self.assertEqual(_cmp_samples(ref, [collate_fn(got_train), got_eval], "train_kill"), [])
        self.assertEqual(
            [o.split(":", 2)[2] for o in cm.output],
            [
                "[fast-batch] train pools: per-sample path (DL_FAST_BATCH=0, env) rank=0",
                "armed OK: [fast-batch] eval pools: fast builder (build_batch; "
                "DL_FAST_BATCH_EVAL=1, gin default) rank=0",
            ],
        )

    def test_eval_only_fast(self) -> None:
        """DL_FAST_BATCH=0 DL_FAST_BATCH_EVAL=1 (env, no gin)."""
        ds = self._ds()
        ref = self._ref(ds)
        n = len(self.batch)
        with mock.patch.object(dsmod, "_FAST_BATCH", False), _env("1"):
            base, train, ev = self._pools(ds)
            self.assertIsNone(train.fast_batch)
            self.assertIs(ev.fast_batch, True)
            got_eval, c_eval = self._fetch(ev)
            got_train, c_train = self._fetch(train)
            # the per-call eval loaders (non-double-buffer path, final eval)
            view = tu._eval_dataset(base)
            self.assertIsNot(view, base)
            got_view, c_view = self._fetch(view)
            self.assertFalse(dsmod.fast_batch_enabled(), "scope leaked")
        self.assertIsInstance(got_eval, Samples)
        self.assertEqual(c_eval, {"build_batch": 1, "_build_sample": 0, "_build_sample_np": 0})
        self.assertIsInstance(got_view, Samples)
        self.assertEqual(c_view, c_eval)
        self.assertIsInstance(got_train, list)
        self.assertEqual(c_train, {"build_batch": 0, "_build_sample": n, "_build_sample_np": 0})
        self.assertEqual(
            _cmp_samples(ref, [got_eval, collate_fn(got_train), got_view], "eval_only"), []
        )

    def test_train_fast_eval_old(self) -> None:
        """DL_FAST_BATCH=1 DL_FAST_BATCH_EVAL=0: eval runs the OLD list path."""
        ds = self._ds()
        ref = self._ref(ds)
        n = len(self.batch)
        with mock.patch.object(dsmod, "_FAST_BATCH", True), _env("0"):
            _, train, ev = self._pools(ds)
            self.assertIs(ev.fast_batch, False)
            got_eval, c_eval = self._fetch(ev)
            got_train, c_train = self._fetch(train)
            self.assertTrue(dsmod.fast_batch_enabled(), "scope leaked")
        self.assertIsInstance(got_eval, list)
        self.assertEqual(c_eval, {"build_batch": 0, "_build_sample": n, "_build_sample_np": 0})
        self.assertIsInstance(got_train, Samples)
        self.assertEqual(c_train, {"build_batch": 1, "_build_sample": 0, "_build_sample_np": 0})
        self.assertEqual(_cmp_samples(ref, [collate_fn(got_eval), got_train], "train_only"), [])

    def test_default_unchanged(self) -> None:
        """No gin, DL_FAST_BATCH_EVAL unset: every pool follows DL_FAST_BATCH."""
        ds = self._ds()
        n = len(self.batch)
        for fast in (False, True):
            ref = self._ref(ds)  # fresh: comparing fills the KJTs' lazy caches
            with self.subTest(DL_FAST_BATCH=int(fast)), \
                    mock.patch.object(dsmod, "_FAST_BATCH", fast), _env(None):
                base, train, ev = self._pools(ds)
                self.assertIsNone(ev.fast_batch)
                self.assertIs(tu._eval_dataset(base), base)
                outs = []
                for w in (ev, train, base):
                    got, c = self._fetch(w)
                    if fast:
                        self.assertIsInstance(got, Samples)
                        self.assertEqual(c, {"build_batch": 1, "_build_sample": 0, "_build_sample_np": 0})
                    else:
                        self.assertIsInstance(got, list)
                        self.assertEqual(c, {"build_batch": 0, "_build_sample": n, "_build_sample_np": 0})
                    outs.append(collate_fn(got))
                self.assertEqual(_cmp_samples(ref, outs, f"default{int(fast)}"), [])

    def test_scope_restores(self) -> None:
        with mock.patch.object(dsmod, "_FAST_BATCH", False):
            with dsmod.fast_batch_scope(True):
                self.assertTrue(dsmod.fast_batch_enabled())
                self.assertEqual(dsmod.fast_batch_knob(), "DL_FAST_BATCH_EVAL=1")
                with dsmod.fast_batch_scope(None):
                    self.assertTrue(dsmod.fast_batch_enabled())
                with dsmod.fast_batch_scope(False):
                    self.assertFalse(dsmod.fast_batch_enabled())
                self.assertTrue(dsmod.fast_batch_enabled())
            self.assertFalse(dsmod.fast_batch_enabled())
            self.assertEqual(dsmod.fast_batch_knob(), "DL_FAST_BATCH=0")
            with self.assertRaises(KeyError):
                with dsmod.fast_batch_scope(True):
                    raise KeyError("x")
            self.assertFalse(dsmod.fast_batch_enabled())

    def test_arming_lines(self) -> None:
        ds = self._ds()
        with mock.patch.object(dsmod, "_FAST_BATCH", False), _env(None):
            _, train, ev = self._pools(ds)
            with self.assertNoLogs(tu.logger, level="WARNING"):
                tu._log_fast_batch_pools(0, {"train": train, "eval": ev})
        with mock.patch.object(dsmod, "_FAST_BATCH", False), _env("1", "0"):
            _, train, ev = self._pools(ds)
            # Ship 12: every rank logs (tagged rank=R).
            with self.assertLogs(tu.logger, level="WARNING") as cm3:
                tu._log_fast_batch_pools(3, {"train": train, "eval": ev})
            self.assertTrue(all(o.endswith("rank=3") for o in cm3.output), cm3.output)
            tu._FAST_BATCH_POOLS_LOGGED.clear()
            with self.assertLogs(tu.logger, level="WARNING") as cm:
                tu._log_fast_batch_pools(0, {"train": train, "eval": ev})
                tu._log_fast_batch_pools(0, {"train": train, "eval": ev})  # once per kind
        self.assertEqual(len(cm.output), 2, cm.output)
        self.assertIn("[fast-batch] train pools: per-sample path (DL_FAST_BATCH=0, env)", cm.output[0])
        self.assertNotIn("armed OK", cm.output[0])
        self.assertIn(
            "armed OK: [fast-batch] eval pools: fast builder (build_batch; DL_FAST_BATCH_EVAL=1, env)",
            cm.output[1],
        )
        # follow: eval pool reports DL_FAST_BATCH's builder.
        tu._FAST_BATCH_POOLS_LOGGED.clear()
        with mock.patch.object(dsmod, "_FAST_BATCH", True), _env("-1", "1"):
            _, train, ev = self._pools(ds)
            with self.assertLogs(tu.logger, level="WARNING") as cm:
                tu._log_fast_batch_pools(0, {"eval": ev})
        self.assertIn(
            "armed OK: [fast-batch] eval pools: fast builder (build_batch; DL_FAST_BATCH_EVAL "
            "unset or < 0: follows DL_FAST_BATCH=1, env)",
            cm.output[0],
        )

    def test_launcher_forwards_knobs(self) -> None:
        text = _LAUNCHER.read_text()
        for k in _KNOBS:
            self.assertIn("${%s:+-e %s=$%s}" % (k, k, k), text)

    def _fork_pools(self, gin_defaults: bool) -> None:
        """Persistent fork pools: the eval pool's workers call build_batch; the
        train pool's workers call it too under the gin defaults (item (61))
        and run the list path with DL_FAST_BATCH off; batches identical; the
        eval pool keeps doing so when re-armed by iter() for the next pass,
        rebuilding every batch (nothing cached across passes)."""
        ds = self._ds()
        idx = np.arange(len(self.positions))[::5][:192].astype(np.int64)
        bs, nw = 32, 2
        if gin_defaults:
            ctx_gin, fb, env = self._gin(), None, _env(None)
        else:
            ctx_gin, fb, env = contextlib.nullcontext(), False, _env("1")
        with ctx_gin, mock.patch.object(dsmod, "_FAST_BATCH", fb), env:
            tu.fast_batch_pools()
            _, train, ev = self._pools(ds)
            self.assertIs(ev.fast_batch, True)
            mk = lambda w, s: tu.make_persistent_streaming_dataloader(  # noqa: E731
                dataset=w, sampler=s, batch_size=bs, num_workers=nw,
                prefetch_factor=2, pin_memory=False,
            )
            s_tr, s_ev = tu.StreamingWindowSampler(0, 1), tu.StreamingWindowSampler(0, 1)
            dl_tr, dl_ev = mk(train, s_tr), mk(ev, s_ev)
            s_tr.set_window(idx)
            s_ev.set_window(idx)
            try:
                c0 = _counts()
                got_tr = list(iter(dl_tr))
                c1 = _counts()
                passes = [list(iter(dl_ev)) for _ in range(2)]
                c2 = _counts()
            finally:
                for dl in (dl_tr, dl_ev):
                    it = getattr(dl, "_iterator", None)
                    if it is not None:
                        it._shutdown_workers()
        nb = len(idx) // bs
        self.assertEqual(len(got_tr), nb)
        self.assertEqual(
            {k: c1[k] - c0[k] for k in _NAMES},
            {"build_batch": nb, "_build_sample": 0, "_build_sample_np": 0}
            if gin_defaults
            else {"build_batch": 0, "_build_sample": nb * bs, "_build_sample_np": 0},
        )
        self.assertEqual(
            {k: c2[k] - c1[k] for k in _NAMES},
            {"build_batch": 2 * nb, "_build_sample": 0, "_build_sample_np": 0},
        )
        self.assertEqual([len(p) for p in passes], [nb, nb])
        for j, (a, b0, b1) in enumerate(zip(got_tr, *passes)):
            self.assertEqual(_cmp_samples(a, [b0, b1], f"b{j}"), [])

    def test_fork_pools_eval_only(self) -> None:
        self._fork_pools(gin_defaults=False)

    def test_fork_pools_gin_defaults(self) -> None:
        self._fork_pools(gin_defaults=True)

    def test_prestart_pools_gin_defaults(self) -> None:
        """prestart_streaming_loaders (the production pre-RUN_START path) under
        the shipped gin items (32), (61) and (37): the switches are resolved
        before the forks, the pre-forked eval pool carries
        DL_FAST_BATCH_EVAL=1 and the single train pool DL_FAST_BATCH=1 (gin
        default), nothing is built while the windows are empty, rank 0 logs
        both lines; a pass over the pre-forked eval pool then runs build_batch
        in its workers and matches the per-sample reference."""
        ds = self._ds()
        idx = np.arange(len(self.positions))[::5][:128].astype(np.int64)
        bs = 32
        with mock.patch.object(dsmod, "_FAST_BATCH", False), _env(None):
            refs = [
                collate_fn(tu.HammerToTorchDataset(ds).__getitems__(idx[j:j + bs].tolist()))
                for j in range(0, len(idx), bs)
            ]

        def factory(hstu_config, is_inference, embedding_config):
            return ds

        level = cp._PRESTART_LEVEL[0]
        bundle = None
        with self._gin(), mock.patch.object(dsmod, "_FAST_BATCH", None), _env(None), \
                mock.patch.object(tu, "get_dataset", lambda: (factory, {})):
            for p, v in (("batch_size", bs), ("num_workers", 2), ("prefetch_factor", 2)):
                gin.bind_parameter(f"make_persistent_streaming_dataloader.{p}", v)
            try:
                c0 = _counts()
                with self.assertLogs(tu.logger, level="WARNING") as cm:
                    ok = tu.prestart_streaming_loaders(
                        hstu_config=None, embedding_table_configs={}, rank=0,
                        world_size=1, enabled=1, persistent_loader=1, double_buffer=1,
                        eval_num_workers=2, eval_every_n_windows=0,
                        eval_every_data_pct=0.001,
                    )
                bundle = tu._take_prestart_bundle()
                c1 = _counts()
                self.assertTrue(ok)
                self.assertIsNotNone(bundle)
                ev_dl = bundle["eval_dl"]
                self.assertIsInstance(ev_dl.dataset, tu._ReadProbedDataset)
                self.assertIs(ev_dl.dataset.fast_batch, True)
                self.assertTrue(bundle["prefetcher"]._single)
                for dl in bundle["prefetcher"]._dls:
                    self.assertIsNone(dl.dataset.fast_batch)
                self.assertTrue(dsmod.fast_batch_enabled())
                bundle["eval_sampler"].set_window(idx)
                got = list(iter(ev_dl))
                c2 = _counts()
            finally:
                cp.set_prestart_level(level)
                if bundle is not None:
                    for dl in [bundle["eval_dl"]] + list(bundle["prefetcher"]._dls):
                        it = getattr(dl, "_iterator", None)
                        if it is not None:
                            it._shutdown_workers()
        self.assertEqual(c1, c0, "a sample was built while the windows were empty")
        self.assertEqual(
            {k: c2[k] - c1[k] for k in _NAMES},
            {"build_batch": len(refs), "_build_sample": 0, "_build_sample_np": 0},
        )
        self.assertEqual(len(got), len(refs))
        for j, (r, g) in enumerate(zip(refs, got)):
            self.assertEqual(_cmp_samples(r, [g], f"prestart/b{j}"), [])
        lines = [ln.split(":", 2)[2] for ln in cm.output]
        self.assertIn(
            "armed OK: [fast-batch] train pools: fast builder (build_batch; DL_FAST_BATCH=1, "
            "gin default) rank=0",
            lines,
        )
        self.assertIn(
            "armed OK: [fast-batch] eval pools: fast builder (build_batch; "
            "DL_FAST_BATCH_EVAL=1, gin default) rank=0",
            lines,
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# pyre-unsafe
"""Ship-12 merge: the single chained train pool (yambda_5b.gin item (37),
window_single_pool, DL_SINGLE_POOL gin default 1) keeps the behaviour of the
dlstart ship-12 loader items (18) DL_PRESTART, (32) DL_FAST_BATCH /
DL_FAST_BATCH_EVAL and (36) DL_COLD_PREFETCH_STREAM.

dlstart's loader tests build their pools without yambda_5b.gin, i.e. with the
two alternating pools. Here they run again with the single pool forced on:

  * dl_prestart_test: the pre-forked pool reads nothing until the stream
    starts, the global RNG stream and the batches equal the build without the
    pre-start (and the two-pool build), the first worker read is stamped after
    the release; with RUN_START expected but not logged the stream refuses to
    start before any window index is installed.
  * cold_prefetch_test: no file access before the release, identical batches,
    the streaming read-ahead starts after the release, reads past the first
    batch and ends with the stream; on the chained warm stream it walks the
    pool's batches in pull order and sees every window done.
  * fast_batch_eval_test: the pre-started eval pool keeps its own
    DL_FAST_BATCH_EVAL builder, the train pool DL_FAST_BATCH; nothing is built
    while the windows are empty.

Plus the production wiring with the shipped gin bindings of items (32), (61)
and (37): prestart_streaming_loaders builds ONE train pool (gin default) next
to the eval pool; every train batch is built by build_batch in that pool's
forked workers (item (61), gin default; also with DL_FAST_BATCH=1 set) and is
bit-identical to the per-sample batch of the same indices, DL_FAST_BATCH=0
restores the per-sample builder, and the eval pool keeps its own sampler and
builder. With DL_SINGLE_POOL=0 the two alternating pools keep the per-sample
builder (item (61)'s precondition, with its line) unless DL_FAST_BATCH=1 is
set in the environment.

CPU only (toy datasets and the synthetic yambda store of fast_batch_test)."""

import contextlib
import os
import time
import unittest
from unittest import mock

import gin
import numpy as np
import torch

from generative_recommenders.dlrm_v4.datasets import dataset as dsmod
from generative_recommenders.dlrm_v4.datasets.dataset import collate_fn, Samples
from generative_recommenders.dlrm_v4.train import cold_prefetch as cp
from generative_recommenders.dlrm_v4.train import dl_prestart as dlp
from generative_recommenders.dlrm_v4.train import utils as tu
from generative_recommenders.dlrm_v4.train.tests import cold_prefetch_test as _cpt
from generative_recommenders.dlrm_v4.train.tests import dl_prestart_test as _dpt
from generative_recommenders.dlrm_v4.train.tests import fast_batch_eval_test as _fbe
from generative_recommenders.dlrm_v4.train.tests.fast_batch_test import (
    _cmp_samples,
    _make_ds,
    _make_store,
)

# Loader switches cleared from the environment so the gin defaults apply.
_LOADER_KNOBS = (
    "DL_SINGLE_POOL", "DL_FAST_BATCH", "DL_FAST_BATCH_EVAL",
    "DL_PRESTART", "DL_COLD_PREFETCH", "DL_COLD_PREFETCH_STREAM",
    "DL_COLD_PREFETCH_LEAD", "DL_COLD_PREFETCH_GROUP", "DL_WBND_DIAG",
)
# yambda_5b.gin bindings of items (32)/(61) (fast-batch switches) and item
# (37) (single pool).
_GIN_PREFIXES = (
    "fast_batch_pools.", "dlfb/", "dlfbe/",
    "window_single_pool.", "dlsp/",
)


def _loader_env(**set_):
    env = {k: v for k, v in os.environ.items() if k not in _LOADER_KNOBS}
    env.update(set_)
    return mock.patch.dict(os.environ, env, clear=True)


def _shipped_gin() -> str:
    lines = [
        ln for ln in _fbe._GIN.read_text().splitlines() if ln.startswith(_GIN_PREFIXES)
    ]
    assert len(lines) == 9, lines
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def _gin_items():
    gin.clear_config()
    try:
        with gin.unlock_config():
            gin.parse_config(_shipped_gin())
        yield
    finally:
        gin.clear_config()


class _SinglePool:
    """Mixin: window_single_pool forced on for the inherited dlstart tests."""

    def setUp(self) -> None:
        p = mock.patch.object(tu, "window_single_pool", lambda: True)
        p.start()
        self.addCleanup(p.stop)
        super().setUp()


class SinglePoolDlPrestartTest(_SinglePool, _dpt.DlPrestartTest):
    def _prefetcher(self) -> "tu._PrefetchingWindowLoader":
        pf = super()._prefetcher()
        single = bool(tu.window_single_pool())
        self.assertEqual(pf._single, single)
        if single:
            self.assertEqual(len(pf._dls), 1)
            self.assertIs(pf._dls[0].batch_sampler, pf._chain)
        return pf

    def test_single_prestart_matches_two_pool_legacy(self) -> None:
        with mock.patch.object(tu, "collate_fn", _dpt._ints):
            with mock.patch.object(tu, "window_single_pool", lambda: False):
                ref, ref_rng, _, _ = self._run(prestart=False)
            got, got_rng, release, stamps = self._run(prestart=True)
        self.assertEqual(got, ref)
        self.assertTrue(torch.equal(got_rng, ref_rng), "RNG stream changed")
        self.assertGreaterEqual(stamps["train"], release)

    def test_refuses_before_run_start(self) -> None:
        """DL_PRESTART with RUN_START expected: the single pool's stream raises
        at the sampler release, before any window index is installed; after
        RUN_START it streams the windows."""
        with mock.patch.object(tu, "collate_fn", _dpt._ints):
            _dpt._READS.value = 0
            pf = self._prefetcher()
            pf.prestart()
            dlp._EXPECT_RUN_START[0] = True
            try:
                with self.assertRaisesRegex(RuntimeError, "before RUN_START"):
                    next(iter(pf.stream([0, 1, 2])))
                self.assertEqual(pf._chain.queued_windows(), 0)
                self.assertFalse(pf._chain.live)
                time.sleep(0.5)
                self.assertEqual(_dpt._READS.value, 0, "sample read before RUN_START")
                self.assertEqual(dlp.read_stamps(), {})
                dlp.set_run_start()
                run_start = dlp.run_start_wall()
                out = [(ts, nb, list(it)) for ts, it, nb in pf.stream([0, 1, 2])]
            finally:
                dlp._EXPECT_RUN_START[0] = False
                _cpt._shutdown(pf)
        want = []
        for ts in (0, 1, 2):
            per_rank = list(range(ts * 100, ts * 100 + 64))[0::2]
            want.append((ts, 4, [per_rank[b * 8:(b + 1) * 8] for b in range(4)]))
        self.assertEqual(out, want)
        self.assertGreaterEqual(dlp.read_stamps()["train"], run_start)


class SinglePoolColdPrefetchTest(_SinglePool, _cpt.ColdPrefetchTest):
    def _loader(self, ds, rank: int = 0, ws: int = 2):
        pf = super()._loader(ds, rank=rank, ws=ws)
        self.assertTrue(pf._single)
        return pf

    def test_stream_walk_follows_the_warm_stream(self) -> None:
        """Chained warm stream (the production pipeline path) with
        DL_COLD_PREFETCH_STREAM=1: the walk issues the pool's batches in pull
        order, counts every pull and sees every window done; stream end stops
        it; batches equal the two-pool stream."""
        issued = []
        real_ranges = cp._ranges

        def rec(pool, files, samples, *a, **k):
            if k.get("serial"):
                issued.append(np.asarray(samples).tolist())
            return real_ranges(pool, files, samples, *a, **k)

        env = {"DL_COLD_PREFETCH": "1", "DL_COLD_PREFETCH_STREAM": "1",
               "DL_COLD_PREFETCH_LEAD": "16", "DL_COLD_PREFETCH_GROUP": "2"}
        ts_list = [0, 1, 2, 3, 4]

        def run(single: bool):
            cp._reset_for_tests()
            ds = _cpt._ToyYambda(self.dir, self.dir)
            with mock.patch.object(tu, "window_single_pool", lambda: single):
                pf = self._loader(ds) if single else super(
                    SinglePoolColdPrefetchTest, self)._loader(ds)
            with mock.patch.dict(os.environ, env if single else {}, clear=False):
                if not single:
                    for k in env:
                        os.environ.pop(k, None)
                cp.set_prestart_level(0)
                pf.prestart()
                warm = tu._WarmWindowStream(
                    pf.stream(ts_list, window_cap=None, warm=True, per_window_cap=None),
                    per_window_cap=None,
                )
                got = []
                while True:
                    try:
                        _ts, nb = warm.advance()
                    except StopIteration:
                        break
                    for _ in range(nb):
                        got.append([s[0] for s in next(warm)])
                        time.sleep(0.01)
                threads = list(cp._THREADS)
                for th in threads:
                    th.join(timeout=30)
            _cpt._shutdown(pf)
            return got, threads

        ref, _ = run(single=False)
        with mock.patch.object(cp, "_ranges", rec):
            got, threads = run(single=True)
        self.assertEqual(got, ref)
        self.assertEqual(len(threads), 1)
        self.assertFalse(threads[0].is_alive())
        self.assertTrue(cp._STOP.is_set())
        self.assertEqual(cp._DONE[0], len(ts_list))
        self.assertEqual(cp._PULLED[0], len(got))
        walked = [g[k:k + 8] for g in issued for k in range(0, len(g), 8)]
        self.assertGreater(len(walked), 1)
        # an ordered subsequence of the pulled batches after the first one
        pos = 1
        for b in walked:
            while pos < len(got) and got[pos] != b:
                pos += 1
            self.assertLess(pos, len(got), f"walked batch {b[:2]}.. not in pull order")
            pos += 1


class SinglePoolFastBatchEvalTest(_SinglePool, _fbe.FastBatchEvalTest):
    pass


class SinglePoolFastBatchTrainTest(unittest.TestCase):
    """Production wiring under the shipped gin items (32), (61) and (37)."""

    BS, WS, RANK = 16, 2, 1
    # Anchors per window: 2, 0, 6, 1, 8, 0 and 4 batches per rank against a
    # prefetch depth of 2 workers x 2 = 4 batches.
    SIZES = (70, 0, 200, 33, 260, 5, 150)

    @classmethod
    def setUpClass(cls) -> None:
        cls.store, cls.positions, cls.i2a, cls.i2al = _make_store()

    def setUp(self) -> None:
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
        dlp._reset_for_tests()
        cp._reset_for_tests()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        tu._FAST_BATCH_POOLS_LOGGED.clear()
        cp._reset_for_tests()

    def _ds(self):
        ds = _make_ds(self.store, self.positions, self.i2a, self.i2al)
        rng = np.random.default_rng(5)
        g = np.sort(rng.choice(len(self.positions), sum(self.SIZES), replace=False))
        off = np.concatenate([[0], np.cumsum(self.SIZES)])
        wins = [g[off[i]:off[i + 1]].astype(np.int64) for i in range(len(self.SIZES))]
        ds.train_window_indices = lambda ts: wins[ts]
        return ds, wins

    def _per_sample(self, ds, idx) -> Samples:
        with mock.patch.object(dsmod, "_FAST_BATCH", False), _fbe._env(None):
            return collate_fn(
                tu.HammerToTorchDataset(ds).__getitems__([int(i) for i in idx])
            )

    def _rank_batches(self, g):
        n = (len(g) // self.WS) * self.WS
        per_rank = g[:n][self.RANK::self.WS]
        return [per_rank[b * self.BS:(b + 1) * self.BS] for b in range(len(per_rank) // self.BS)]

    def test_launcher_forwards_both_patches(self) -> None:
        """scripts/launch_slurm.sh forwards the single-pool knobs (item (37)
        and its diagnostics) and dlstart's loader knobs."""
        text = _fbe._LAUNCHER.read_text()
        for k in ("DL_SINGLE_POOL", "DL_WBND_DIAG", "DLQ_LOG_EVERY",
                  "DL_PRESTART", "DL_FAST_BATCH", "DL_FAST_BATCH_EVAL", "DL_COLD_PREFETCH",
                  "DL_COLD_PREFETCH_STREAM", "DL_COLD_PREFETCH_LEAD", "DL_COLD_PREFETCH_HORIZON",
                  "DL_COLD_PREFETCH_GROUP", "DL_COLD_PREFETCH_BATCHES", "DL_COLD_PREFETCH_THREADS",
                  "FP8_WARMUP", "FP8_WARMUP_STORES", "FP8_WARMUP_VERIFY", "FP8_WARMUP_SKIP",
                  "FP8_WARMUP_QUIET", "FP8_WARMUP_N11"):
            self.assertEqual(text.count("${%s:+-e %s=$%s}" % (k, k, k)), 1, k)

    def test_gin_defaults(self) -> None:
        with _gin_items():
            with _loader_env():
                self.assertTrue(tu.window_single_pool())
                tu.fast_batch_pools()  # item (61): train pools on build_batch
                self.assertTrue(dsmod.fast_batch_enabled())
                self.assertIsNone(tu._FAST_BATCH_TRAIN_HELD)
            with _loader_env(DL_SINGLE_POOL="0"):
                self.assertFalse(tu.window_single_pool())
                tu.fast_batch_pools()  # item (61)'s precondition holds them off
                self.assertFalse(dsmod.fast_batch_enabled())
                self.assertIsNotNone(tu._FAST_BATCH_TRAIN_HELD)

    def _prestart_run(self, train_env=None, single: bool = True) -> None:
        """prestart + a streamed run under the shipped items; ``train_env`` =
        DL_FAST_BATCH in the env (None = unset: item (61)'s gin default),
        ``single`` False = DL_SINGLE_POOL=0 (the two alternating pools)."""
        train_fast = train_env == "1" or (train_env is None and single)
        ds, wins = self._ds()
        ts_list = list(range(len(wins)))
        ref = [(ts, self._per_sample(ds, b)) for ts in ts_list for b in self._rank_batches(wins[ts])]
        ev_idx = np.arange(len(self.positions))[::7][:96].astype(np.int64)
        ev_ref = [self._per_sample(ds, b) for b in self._rank_batches(ev_idx)]
        for v in _fbe._CNT.values():
            v.value = 0

        def factory(hstu_config, is_inference, embedding_config):
            return ds

        env = {} if train_env is None else {"DL_FAST_BATCH": train_env}
        if not single:
            env["DL_SINGLE_POOL"] = "0"
        level = cp._PRESTART_LEVEL[0]
        bundle = None
        with _gin_items(), _loader_env(**env), \
                mock.patch.object(tu, "get_dataset", lambda: (factory, {})):
            for p, v in (("batch_size", self.BS), ("num_workers", 2), ("prefetch_factor", 2)):
                gin.bind_parameter(f"make_persistent_streaming_dataloader.{p}", v)
            try:
                with self.assertLogs(tu.logger, level="WARNING") as cm:
                    ok = tu.prestart_streaming_loaders(
                        hstu_config=None, embedding_table_configs={}, rank=self.RANK,
                        world_size=self.WS, enabled=1, persistent_loader=1, double_buffer=1,
                        eval_num_workers=2, eval_every_n_windows=0, eval_every_data_pct=0.001,
                    )
                bundle = tu._take_prestart_bundle()
                self.assertTrue(ok)
                pf, ev_dl = bundle["prefetcher"], bundle["eval_dl"]
                # one train pool (gin default) on its own chained sampler, or
                # two alternating pools (DL_SINGLE_POOL=0); the train wrapper
                # (DL_FAST_BATCH) and collate_fn; the eval pool is a separate
                # pool on its own sampler and builder
                self.assertIs(pf._single, single)
                self.assertEqual(len(pf._dls), 1 if single else 2)
                for dl in pf._dls:
                    if single:
                        self.assertIs(dl.batch_sampler, pf._chain)
                    self.assertIs(dl.collate_fn, tu.collate_fn)
                    self.assertIsInstance(dl.dataset, tu._ReadProbedDataset)
                    self.assertIsNone(dl.dataset.fast_batch)
                    self.assertIsNot(ev_dl, dl)
                self.assertIs(dsmod.fast_batch_enabled(), train_fast)
                # each rank's train-pool line (item (61); its precondition)
                lines = [ln.split(":", 2)[2] for ln in cm.output]
                if train_fast:
                    want = (
                        "armed OK: [fast-batch] train pools: fast builder (build_batch; "
                        "DL_FAST_BATCH=1, %s) rank=%d"
                        % ("env" if train_env else "gin default", self.RANK)
                    )
                elif train_env is None:
                    want = (
                        "[fast-batch] train pools: per-sample path (DL_FAST_BATCH=0, gin "
                        "default 1 held because item (37)'s single train pool is off: "
                        "window_single_pool() False, env DL_SINGLE_POOL=0; env "
                        "DL_FAST_BATCH=1 overrides) rank=%d" % self.RANK
                    )
                else:
                    want = (
                        "[fast-batch] train pools: per-sample path (DL_FAST_BATCH=0, env) "
                        "rank=%d" % self.RANK
                    )
                self.assertEqual([ln for ln in lines if "train pools" in ln], [want])
                self.assertIsInstance(bundle["eval_sampler"], tu.StreamingWindowSampler)
                self.assertIs(ev_dl.dataset.fast_batch, True)
                time.sleep(0.5)
                self.assertEqual(_fbe._counts(), {n: 0 for n in _fbe._NAMES},
                                 "a sample was built before the stream started")
                warm = tu._WarmWindowStream(
                    pf.stream(ts_list, window_cap=None, warm=True, per_window_cap=None),
                    per_window_cap=None,
                )
                got, ev_got, k = [], None, 0
                while True:
                    try:
                        ts, nb = warm.advance()
                    except StopIteration:
                        break
                    for _ in range(nb):
                        got.append((ts, next(warm)))
                        k += 1
                        if k == 3:  # one eval pass while the train pool streams
                            bundle["eval_sampler"].set_window(ev_idx)
                            ev_got = list(iter(ev_dl))
            finally:
                cp.set_prestart_level(level)
                if bundle is not None:
                    for d in [bundle["eval_dl"]] + list(bundle["prefetcher"]._dls):
                        it = getattr(d, "_iterator", None)
                        if it is not None:
                            it._shutdown_workers()
        self.assertEqual([ts for ts, _ in got], [ts for ts, _ in ref])
        for j, ((_, r), (_, g)) in enumerate(zip(ref, got)):
            self.assertIsInstance(g, Samples)
            self.assertEqual(_cmp_samples(r, [g], f"train/b{j}"), [])
        self.assertIsNotNone(ev_got)
        self.assertEqual(len(ev_got), len(ev_ref))
        for j, (r, g) in enumerate(zip(ev_ref, ev_got)):
            self.assertEqual(_cmp_samples(r, [g], f"eval/b{j}"), [])
        n_tr, n_ev = len(ref), len(ev_ref)
        self.assertEqual(
            _fbe._counts(),
            {"build_batch": n_ev + (n_tr if train_fast else 0),
             "_build_sample": 0 if train_fast else n_tr * self.BS,
             "_build_sample_np": 0},
        )

    def test_prestart_single_pool_fast_train(self) -> None:
        """DL_FAST_BATCH=1 (env): build_batch for every train and eval batch."""
        self._prestart_run(train_env="1")

    def test_prestart_single_pool_default_train(self) -> None:
        """Gin defaults (item (61)): build_batch for every train and eval
        batch."""
        self._prestart_run(train_env=None)

    def test_prestart_single_pool_train_kill_switch(self) -> None:
        """DL_FAST_BATCH=0: train pool per-sample, eval pool build_batch."""
        self._prestart_run(train_env="0")

    def test_prestart_two_pools_hold_train(self) -> None:
        """DL_SINGLE_POOL=0, DL_FAST_BATCH unset: item (61)'s precondition
        keeps the two pools on the per-sample builder and says why; the eval
        pool keeps build_batch."""
        self._prestart_run(train_env=None, single=False)

    def test_prestart_two_pools_env_fast_train(self) -> None:
        """DL_SINGLE_POOL=0 with DL_FAST_BATCH=1 set: the env wins, the two
        pools build with build_batch (bit-identical)."""
        self._prestart_run(train_env="1", single=False)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# pyre-strict
"""DL_COLD_PREFETCH: no file access before the sampler release, byte-identical
batches with and without it, exact mirroring of the rank's first batches, and
graceful handling of missing / short / mismatched column files.

Uses real persistent fork workers over a toy flat event store laid out exactly
like the yambda hstu_cache (.npy columns read through mmaps; CPU only)."""

import logging
import multiprocessing as mp
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch

from generative_recommenders.dlrm_v4.train import cold_prefetch as cp
from generative_recommenders.dlrm_v4.train import dl_prestart as dlp
from generative_recommenders.dlrm_v4.train import utils as tu

_READS = mp.get_context("fork").Value("i", 0)
_PER_WINDOW = 200
_SCAN = 3000


def _write_store(d: str) -> None:
    rng = np.random.default_rng(7)
    lens = rng.integers(2000, 9000, size=12)
    uid = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
    n = len(uid)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    cols = {
        "flat_uid": uid,
        "flat_item_ids": rng.integers(0, 1 << 40, n, dtype=np.int64),
        "flat_timestamps": np.arange(n, dtype=np.int64) * 7 + 1_600_000_000,
        "flat_event_types": rng.integers(0, 3, n, dtype=np.int64),
        "flat_played_ratio": rng.random(n, dtype=np.float32) * 100,
        "flat_is_organic": rng.integers(0, 2, n).astype(np.int8),
        "user_start": starts,
        "user_end": starts + lens,
    }
    r = rng.integers(0, 3, n)
    cols["flat_is_listen_plus"] = r == 0
    cols["flat_is_like"] = r == 1
    cols["flat_is_skip"] = r == 2
    for k, v in cols.items():
        np.save(os.path.join(d, f"{k}.npy"), v)
    idx = np.arange(n, dtype=np.int64)
    pos = idx[(idx - starts[uid] >= 10) & (idx % 5 == 0)]
    np.save(os.path.join(d, "positions_toy.npy"), pos)


class _Store:
    pass


class _ToyYambda:
    """The attributes cold_prefetch reads + a get_sample that touches the same
    scan-window / scalar bytes as yambda._build_sample."""

    is_eval = False

    def __init__(self, data_dir: str, prefetch_dir: str) -> None:
        self.store = _Store()
        for c in ("flat_uid", "flat_item_ids", "flat_timestamps", "flat_event_types",
                  "flat_played_ratio", "flat_is_organic", "flat_is_listen_plus",
                  "flat_is_like", "flat_is_skip", "user_start", "user_end"):
            setattr(self.store, c, np.load(os.path.join(data_dir, f"{c}.npy"), mmap_mode="r"))
        self._positions = np.load(os.path.join(data_dir, "positions_toy.npy"), mmap_mode="r")
        self._positions_name = "positions_toy.npy"
        self._cache_dir = prefetch_dir  # where cold_prefetch looks for the files
        self._scan_window = _SCAN

    def train_window_indices(self, ts: int) -> np.ndarray:
        lo = ts * _PER_WINDOW + (37 if ts % 2 else 0)  # windows of unequal size
        hi = min(len(self._positions), (ts + 1) * _PER_WINDOW)
        return np.arange(lo, hi, dtype=np.int64)

    def load_query_samples(self, idx) -> None:  # pyre-ignore[2]
        with _READS.get_lock():
            _READS.value += len(idx)

    def get_sample(self, i: int):  # pyre-ignore[3]
        s = self.store
        fp = int(self._positions[i])
        uid = int(s.flat_uid[fp])
        ss = max(int(s.user_start[uid]), fp - _SCAN)
        return (
            int(i), fp, uid,
            int(s.flat_item_ids[ss:fp + 1].sum() % (1 << 61)),
            int(s.flat_timestamps[ss:fp + 1].sum()),
            int(s.flat_is_listen_plus[ss:fp].sum()), int(s.flat_is_like[ss:fp].sum()),
            int(s.flat_is_skip[ss:fp].sum()), float(s.flat_played_ratio[fp]),
            int(s.flat_event_types[fp]), int(s.flat_is_organic[fp]),
        )

    def unload_query_samples(self, idx) -> None:  # pyre-ignore[2]
        pass


def _shutdown(pf) -> None:  # pyre-ignore[2]
    """Stop the persistent pools now (leaked pools torn down later by the GC
    race multiprocessing's queue feeder threads and print EBADF noise)."""
    for it in pf._iters:  # pyre-ignore[16]
        if it is not None:
            it._shutdown_workers()  # pyre-ignore[16]


def _ident(xs):  # pyre-ignore[2,3]
    return list(xs)


class ColdPrefetchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = tempfile.mkdtemp(prefix="cold_prefetch_test_")
        _write_store(cls.dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.dir, ignore_errors=True)

    def setUp(self) -> None:
        cp._reset_for_tests()
        dlp._reset_for_tests()
        self.calls = []  # (wall, name) of every prefetch file access
        lock = threading.Lock()

        def wrap(name, fn):  # pyre-ignore[2,3]
            def w(*a, **k):  # pyre-ignore[2,3]
                with lock:
                    self.calls.append((time.time(), name, a[3] if name == "fadvise" else None))
                return fn(*a, **k)
            return w

        self._patches = [
            mock.patch.object(cp, "_open", wrap("open", os.open)),
            mock.patch.object(cp, "_pread", wrap("pread", os.pread)),
            mock.patch.object(cp, "_fadvise", wrap("fadvise", os.posix_fadvise)),
            mock.patch.object(tu, "collate_fn", _ident),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        cp._reset_for_tests()
        cp.set_prestart_level(0)
        for p in self._patches:
            p.stop()

    def _loader(self, ds, rank: int = 0, ws: int = 2):  # pyre-ignore[2,3]
        base = tu.HammerToTorchDataset(ds)  # pyre-ignore[6]
        probed = tu._ReadProbedDataset(base, dlp.new_read_slot("train"))
        return tu._PrefetchingWindowLoader(
            dataset=base,
            sampler_factory=lambda: tu.StreamingWindowSampler(rank=rank, world_size=ws),
            dl_factory=lambda s: tu.make_persistent_streaming_dataloader(
                dataset=probed, sampler=s, batch_size=8, num_workers=2,
                prefetch_factor=2, pin_memory=False,
            ),
        )

    def _run(self, env: dict, prestart: bool = True, prefetch_dir=None):  # pyre-ignore
        _READS.value = 0
        torch.manual_seed(1234)
        ds = _ToyYambda(self.dir, prefetch_dir or self.dir)
        pf = self._loader(ds)
        with mock.patch.dict(os.environ, env, clear=False):
            for k in ("DL_COLD_PREFETCH", "DL_PRESTART", "DL_COLD_PREFETCH_STREAM",
                      "DL_COLD_PREFETCH_LEAD", "DL_COLD_PREFETCH_GROUP"):
                if k not in env:
                    os.environ.pop(k, None)
            # Since ship7 (gin item (18)) the pre-start level reaches the
            # prefetcher through utils.prestart_streaming_loaders ->
            # cp.set_prestart_level(level), not through the environment
            # (cold_prefetch.mode()); mirror that call. Reset in tearDown.
            cp.set_prestart_level(int(env.get("DL_PRESTART", "0")))
            if prestart:
                pf.prestart()
                time.sleep(0.5)
                self.assertEqual(_READS.value, 0)
                self.assertEqual(self.calls, [], "prefetch file access before release")
            release = time.time()
            out = [(ts, nb, list(it)) for ts, it, nb in pf.stream([0, 1, 2, 3, 4])]
            for th in list(cp._THREADS):
                th.join(timeout=30)
        _shutdown(pf)
        return out, release, ds

    def test_nothing_before_release_and_identical_batches(self) -> None:
        ref, _, _ = self._run({})
        self.assertEqual(self.calls, [])  # off by default
        for env in ({"DL_COLD_PREFETCH": "1"}, {"DL_COLD_PREFETCH": "2"}, {"DL_PRESTART": "2"}):
            cp._reset_for_tests()
            self.calls.clear()
            with self.assertLogs(cp.logger, level="WARNING") as lg:
                got, release, _ = self._run(env)
            self.assertEqual(got, ref, f"batches changed under {env}")
            self.assertTrue(self.calls, f"no prefetch I/O under {env}")
            self.assertGreaterEqual(min(t for t, _, _ in self.calls), release)
            armed = [m for m in lg.output if "[cold-prefetch] armed" in m]
            self.assertEqual(len(armed), 1, lg.output)
            self.assertIn("first_ranges=", armed[0])
            self.assertNotIn("first_ranges=0 ", armed[0])
            if env.get("DL_COLD_PREFETCH") != "2":
                self.assertTrue(any(n == "fadvise" and a == os.POSIX_FADV_WILLNEED
                                    for _, n, a in self.calls))

    def test_streaming_read_ahead_through_the_loader(self) -> None:
        """DL_COLD_PREFETCH_STREAM=1 through the real loader: identical batches,
        nothing before release, read-ahead past the first batch, and the walk
        ends with the stream (stream() stops it)."""
        ref, _, _ = self._run({})
        cp._reset_for_tests()
        self.calls.clear()
        env = {"DL_COLD_PREFETCH": "1", "DL_COLD_PREFETCH_STREAM": "1",
               "DL_COLD_PREFETCH_LEAD": "6", "DL_COLD_PREFETCH_GROUP": "2"}
        with self.assertLogs(cp.logger, level="WARNING") as lg:
            got, release, _ = self._run(env)
        self.assertEqual(got, ref)
        self.assertTrue(self.calls)
        self.assertGreaterEqual(min(t for t, _, _ in self.calls), release)
        armed = [m for m in lg.output if "[cold-prefetch] armed" in m]
        self.assertEqual(len(armed), 1, lg.output)
        self.assertIn("stream=on lead=6", armed[0])
        self.assertIn("batches=1 ", armed[0])  # only the first batch in wave 1
        done = [m for m in lg.output if "[cold-prefetch] stream done" in m]
        self.assertEqual(len(done), 1, lg.output)
        self.assertFalse(any("failed" in m for m in lg.output), lg.output)
        self.assertTrue(all(not th.is_alive() for th in cp._THREADS))

    def test_auto_stream_through_the_loader(self) -> None:
        """DL_COLD_PREFETCH_STREAM=auto through the real loader (the sampler's
        world size reaches start()): the walk on two nodes, with the reason on
        the armed line; one-shot only on one node, with one "resolved off"
        line. Identical batches and nothing before release either way."""
        ref, _, _ = self._run({})
        for nnodes, on in (("2", True), ("1", False)):
            with self.subTest(nnodes=nnodes):
                cp._reset_for_tests()
                self.calls.clear()
                env = {"DL_COLD_PREFETCH": "1", "DL_COLD_PREFETCH_STREAM": "auto",
                       "NNODES": nnodes, "DL_COLD_PREFETCH_LEAD": "6",
                       "DL_COLD_PREFETCH_GROUP": "2"}
                with self.assertLogs(cp.logger, level="WARNING") as lg:
                    got, release, _ = self._run(env)
                self.assertEqual(got, ref)
                self.assertTrue(self.calls)
                self.assertGreaterEqual(min(t for t, _, _ in self.calls), release)
                self.assertFalse(any("failed" in m for m in lg.output), lg.output)
                armed = [m for m in lg.output if "[cold-prefetch] armed" in m]
                off = [m for m in lg.output if "stream=auto resolved off" in m]
                done = [m for m in lg.output if "[cold-prefetch] stream done" in m]
                self.assertEqual(len(armed), 1, lg.output)
                if on:
                    self.assertIn("stream=on lead=6 ", armed[0])
                    self.assertIn(" auto=multi-node(nnodes=2)", armed[0])
                    self.assertIn("batches=1 ", armed[0])
                    self.assertEqual((len(off), len(done)), (0, 1), lg.output)
                else:
                    self.assertNotIn("stream=on lead=", armed[0])
                    self.assertIn("batches=8 ", armed[0])
                    self.assertEqual(len(off), 1, lg.output)
                    self.assertIn("[cold-prefetch] stream=auto resolved off: single node "
                                  "(nnodes=1)", off[0])
                    self.assertEqual(done, [])
                self.assertTrue(all(not th.is_alive() for th in cp._THREADS))

    def test_legacy_fork_path_starts_after_priming(self) -> None:
        ref, _, _ = self._run({}, prestart=False)
        cp._reset_for_tests()
        self.calls.clear()
        got, release, _ = self._run({"DL_COLD_PREFETCH": "1"}, prestart=False)
        self.assertEqual(got, ref)
        self.assertTrue(self.calls)
        self.assertGreaterEqual(min(t for t, _, _ in self.calls), release)

    def test_start_refuses_before_release(self) -> None:
        ds = _ToyYambda(self.dir, self.dir)
        with mock.patch.dict(os.environ, {"DL_COLD_PREFETCH": "1"}):
            self.assertIsNone(cp.start(ds, [0, 1], 0, 2, 8))
        self.assertEqual(self.calls, [])

    def test_mirrors_the_rank_first_batches(self) -> None:
        ds = _ToyYambda(self.dir, self.dir)
        for rank in (0, 1):
            pf = self._loader(ds, rank=rank, ws=2)
            got = [s for _, it, _ in pf.stream([0, 1, 2, 3]) for b in it for s in b]
            got = [g[0] for g in got]
            batches, used = cp.first_rank_samples(ds, [0, 1, 2, 3], rank, 2, 8, 1000)
            self.assertEqual(np.concatenate(batches).tolist(), got)
            _shutdown(pf)

    def test_gather_matches_mmap(self) -> None:
        f = cp._NpyFile(os.path.join(self.dir, "flat_uid.npy"), None)
        try:
            ref = np.load(os.path.join(self.dir, "flat_uid.npy"))
            idx = np.array([0, 5, 5, len(ref) - 1, 1234, -3, len(ref) + 9], dtype=np.int64)
            st = {"index_runs": 0, "index_bytes": 0}
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(4) as pool:
                got = cp._gather(pool, f, idx, st)
            want = [ref[i] if 0 <= i < len(ref) else -1 for i in idx]
            self.assertEqual(got.tolist(), [int(x) for x in want])
        finally:
            f.close()

    def test_missing_short_and_mismatched_files(self) -> None:
        ref, _, _ = self._run({})
        bad = tempfile.mkdtemp(prefix="cold_prefetch_bad_")
        try:
            for name in os.listdir(self.dir):
                shutil.copy(os.path.join(self.dir, name), bad)
            os.remove(os.path.join(bad, "flat_is_like.npy"))  # missing
            p = os.path.join(bad, "flat_timestamps.npy")  # short
            os.truncate(p, os.path.getsize(p) // 3)
            np.save(os.path.join(bad, "flat_is_skip.npy"), np.zeros(10, dtype=bool))  # mismatch
            for d, positions_ok in ((bad, True), ("/nonexistent_cold_prefetch_dir", False)):
                cp._reset_for_tests()
                self.calls.clear()
                with self.assertLogs(cp.logger, level="WARNING") as lg:
                    got, release, _ = self._run({"DL_COLD_PREFETCH": "1"}, prefetch_dir=d)
                self.assertEqual(got, ref)
                self.assertFalse(any("failed" in m for m in lg.output), lg.output)
                armed = [m for m in lg.output if "[cold-prefetch] armed" in m]
                self.assertEqual(len(armed), 1, lg.output)
                if positions_ok:
                    self.assertNotIn("first_ranges=0 ", armed[0])
                else:
                    self.assertIn("files=0 ", armed[0])
            # A short user_start makes some samples scalar-only; still no failure.
            os.truncate(os.path.join(bad, "user_start.npy"), 128 + 8 * 3)
            cp._reset_for_tests()
            self.calls.clear()
            with self.assertLogs(cp.logger, level="WARNING") as lg:
                got, _, _ = self._run({"DL_COLD_PREFETCH": "1"}, prefetch_dir=bad)
            self.assertEqual(got, ref)
            self.assertFalse(any("failed" in m for m in lg.output), lg.output)
        finally:
            shutil.rmtree(bad, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()

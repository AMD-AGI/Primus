#!/usr/bin/env python3
# pyre-unsafe
"""DL_FAST_BATCH_ONESTORE (yambda_5b.gin item (56)): build_batch carves a
batch's six tensors out of ONE storage, the UIH and candidate tensors being
views of the two halves of the merged tensors. Checked on the synthetic yambda
stores of fast_batch_test (CPU; the pin tests need a GPU):

* every KJT attribute (before and after the lazy accessors) is bit-identical to
  the one-storage-per-tensor fast batch, across the fast_batch_test configs;
  the uih/candidate views sit exactly on the halves of the merged views;
* pickling a batch for a worker hand-off creates ONE DupFd (6 without the knob);
* real fork DataLoader workers: same batches, each received batch is one
  shared storage, with pin_memory on and off;
* no two batches share storage bytes (per-sample, per-tensor fast and
  one-storage batches, in process and from fork workers): item (55)'s
  MADV_REMOVE release would zero a storage a worker reuses for later batches;
* Samples.pin_memory pins a one-storage batch with one pinned storage, equal to
  the per-tensor pin (and the device copies are equal, and do not alias each
  other); other batches fall back to the per-tensor pin;
* the env-only knob (no gin) parses like DL_FAST_BATCH (> 0 = the layout,
  resolved to level 2; the arming text shows a raw value other than the
  resolved level) and defaults to off.
* $DL_PIN_FREE_SHM (item (55), pin_free_shm_test.py) on top of the pin-once
  path: the one shared storage of a batch is released once after its pinned
  copy is complete, the size guard does not skip it (the union of the batch's
  views covers it up to the alignment padding), the alias guard keeps a
  storage the replacement shares, and stats / arming lines count 1 storage per
  one-storage batch and 6 per per-tensor batch.
The gin wiring of item (56) and the pin-once size guard are in
onestore_item49_test.py.

Every storage / layout check runs BEFORE _cmp_samples: its accessors add lazily
computed _offsets tensors (fresh heap storages) to the KJTs they touch.
"""

import multiprocessing.reduction as mpred
import os
import unittest
from multiprocessing.reduction import ForkingPickler
from unittest import mock

import numpy as np
import torch

from generative_recommenders.dlrm_v4.datasets import dataset as dsmod
from generative_recommenders.dlrm_v4.datasets.dataset import collate_fn, Samples
from generative_recommenders.dlrm_v4.train import utils as tu
from generative_recommenders.dlrm_v4.train.tests.fast_batch_test import (
    _cmp_samples,
    _fast,
    _ListSampler,
    _make_ds,
    _make_store,
)

_KJTS = ("uih_features_kjt", "candidates_features_kjt", "merged_sparse_features")
# The shipped layout (level 2; any value > 0 resolves to it).
_LEVELS = (2,)


def _one(level: int):
    return mock.patch.object(dsmod, "_ONESTORE", int(level))


def _tensors(s: Samples):
    out = []
    for f in _KJTS:
        k = getattr(s, f)
        out += [t for t in (k._values, k._weights, k._lengths, k._offsets) if t is not None]
    return out


def _storages(s: Samples):
    return {t.untyped_storage().data_ptr() for t in _tensors(s)}


def _spans(s: Samples):
    """[start, end) byte ranges of the non-empty storages behind ``s``."""
    out = set()
    for t in _tensors(s):
        st = t.untyped_storage()
        if st.nbytes():
            out.add((st.data_ptr(), st.data_ptr() + st.nbytes()))
    return out


def _overlaps(a, b):
    return sorted((x, y) for x in a for y in b if x[0] < y[1] and y[0] < x[1])


def _count_dupfd(obj) -> int:
    calls = []
    orig = mpred.DupFd

    def counting(fd):
        calls.append(fd)
        return orig(fd)

    with mock.patch.object(mpred, "DupFd", counting):
        ForkingPickler.dumps(obj)
    return len(calls)


class OneStoreTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.store, cls.positions, cls.i2a, cls.i2al = _make_store()
        st, pos = cls.store, cls.positions
        off = pos - st.user_start[st.flat_uid[pos]]
        g = np.arange(len(pos))
        rng = np.random.default_rng(3)
        special = np.concatenate([g[off == 0], g[(off > 0) & (off < 4)], g[off > 3000][:40]])
        cls.batches = [
            sorted(rng.choice(g, 64, replace=False).tolist()),
            special.tolist()[:64],
            rng.choice(g, 37, replace=False).tolist(),
            [int(g[-1])],
        ]

    def setUp(self) -> None:
        # The gin flags items (55) and (56) set, and their env variables: the
        # arming lines name "code default" here (fresh per test, restored
        # afterwards).
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DL_FAST_BATCH_ONESTORE", "DL_PIN_FREE_SHM")
        }
        self._patches = [
            mock.patch.object(dsmod, "_ONESTORE_FROM_GIN", False),
            mock.patch.object(dsmod, "_FREE_SHM_FROM_GIN", False),
            mock.patch.dict(os.environ, env, clear=True),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()

    def _ds(self, **kw):
        return _make_ds(self.store, self.positions, self.i2a, self.i2al, **kw)

    def _build(self, ds, batch, level: int) -> Samples:
        with _fast(True), _one(level):
            got = tu.HammerToTorchDataset(ds).__getitems__(batch)
        self.assertIsInstance(got, Samples)
        return collate_fn(got)

    def _check_layout(self, s: Samples, level: int, tag: str) -> None:
        self.assertEqual(level, 2, tag)
        ts = _tensors(s)
        self.assertEqual(len(ts), 6, tag)
        self.assertEqual(len(_storages(s)), 1, tag)
        nbytes = ts[0].untyped_storage().nbytes()
        for t in ts:
            self.assertTrue(t.is_contiguous(), tag)
            self.assertLessEqual((t.storage_offset() + t.numel()) * 8, nbytes, tag)
        u, c, m = (getattr(s, f) for f in _KJTS)
        mv, ml = m.values(), m.lengths()
        self.assertEqual(mv.storage_offset() * 8 % 4096, 0, tag)
        self.assertEqual(ml.storage_offset() * 8 % 4096, 0, tag)
        nu, nl = u.values().numel(), u.lengths().numel()
        self.assertEqual(u.values().storage_offset(), mv.storage_offset(), tag)
        self.assertEqual(c.values().storage_offset(), mv.storage_offset() + nu, tag)
        self.assertEqual(u.lengths().storage_offset(), ml.storage_offset(), tag)
        self.assertEqual(c.lengths().storage_offset(), ml.storage_offset() + nl, tag)
        self.assertEqual(nu + c.values().numel(), mv.numel(), tag)
        self.assertEqual(nl + c.lengths().numel(), ml.numel(), tag)

    def test_identical_across_configs(self) -> None:
        n_ctx = 8
        cases = [
            ("interleaved", dict()),
            ("trunc", dict(max_seq_len=300)),
            ("budget0", dict(max_seq_len=1 + n_ctx)),
            ("budget_neg", dict(max_seq_len=n_ctx - 3)),
            ("last_n", dict(strategy="last_n")),
            ("last_n_L0", dict(strategy="last_n", history_length=0)),
            ("mnc2", dict(mnc=2)),
            ("mnc0", dict(mnc=0)),
        ]
        for tag, kw in cases:
            ds = self._ds(**kw)
            for j, batch in enumerate(self.batches):
                with self.subTest(f"{tag}/b{j}"):
                    ref = self._build(ds, batch, 0)
                    got = [self._build(ds, batch, lv) for lv in _LEVELS]
                    for lv, g in zip(_LEVELS, got):
                        self._check_layout(g, lv, f"{tag}/b{j}/L{lv}")
                    self.assertEqual(len(_storages(ref)), 6)  # mnc0: one is the empty storage
                    self.assertEqual(_cmp_samples(ref, got, f"{tag}/b{j}"), [])

    def test_inference_and_active(self) -> None:
        ds = self._ds()
        ds._is_inference = True
        ds._max_num_candidates_inference = 2
        for batch in self.batches:
            got = [self._build(ds, batch, lv) for lv in _LEVELS]
            self.assertEqual(_cmp_samples(self._build(ds, batch, 0), got, "inf"), [])
        ds = self._ds()
        ds.set_active_indices(np.random.default_rng(5).permutation(len(self.positions))[:3000])
        b = list(range(0, 64))
        got = [self._build(ds, b, lv) for lv in _LEVELS]
        self.assertEqual(_cmp_samples(self._build(ds, b, 0), got, "active"), [])

    def test_one_dupfd_per_batch(self) -> None:
        ds = self._ds()
        # in the main process the storage is on the heap; pickling moves it to
        # shared memory once, like a worker's shared storage
        self.assertEqual(_count_dupfd(self._build(ds, self.batches[0], 0)), 6)
        for lv in _LEVELS:
            self.assertEqual(_count_dupfd(self._build(ds, self.batches[0], lv)), 1, lv)

    def _loader(self, ds, idx, pin: bool):
        return torch.utils.data.DataLoader(
            tu.HammerToTorchDataset(ds), batch_size=32, sampler=_ListSampler(idx),
            collate_fn=collate_fn, drop_last=True, num_workers=2, prefetch_factor=2,
            multiprocessing_context="fork", pin_memory=pin,
        )

    def test_dataloader_workers_identical(self) -> None:
        ds = self._ds()
        idx = np.arange(len(self.positions))[::7][:256].tolist()
        pins = (False, True) if torch.cuda.is_available() else (False,)
        for pin in pins:
            got = {}
            for lv in (0,) + _LEVELS:
                with _fast(True), _one(lv):
                    got[lv] = list(self._loader(ds, idx, pin))
            self.assertEqual(len(got[0]), 8)
            for j in range(len(got[0])):
                a = got[0][j]
                cands = [got[lv][j] for lv in _LEVELS]
                self.assertEqual(len(_storages(a)), 6, f"pin={pin} b{j}")
                for lv, b in zip(_LEVELS, cands):
                    self.assertEqual(len(_storages(b)), 1, f"pin={pin} L{lv} b{j}")
                    if pin:
                        self.assertTrue(all(t.is_pinned() for t in _tensors(b)))
                    else:
                        self.assertTrue(b.uih_features_kjt.values().is_shared())
                        self._check_layout(b, lv, f"dl/L{lv}/b{j}")
                if pin:
                    self.assertTrue(all(t.is_pinned() for t in _tensors(a)))
                self.assertEqual(_cmp_samples(a, cands, f"dl/pin{pin}/b{j}"), [])

    def test_consecutive_batches_share_no_storage(self) -> None:
        """Item (55) releases a received batch's shared storages with
        MADV_REMOVE, which also zeroes those pages in the worker that made
        them. That is safe only while every batch is built on fresh storages: a
        worker-side cache of a tensor that later batches reuse would hand them
        zeros. With every batch kept alive, no two batches may share storage
        bytes: per-sample path, per-tensor fast builder and the one-storage
        layout, built in this process and received from forked DataLoader
        workers (pin_memory off: the received storages themselves)."""
        ds = self._ds()
        idx = np.arange(len(self.positions))[::7][:256].tolist()
        arms = [("per-sample", False, 0), ("fast", True, 0)]
        arms += [(f"onestore{lv}", True, lv) for lv in _LEVELS]
        for tag, fast, lv in arms:
            with _fast(fast), _one(lv):
                local = [
                    collate_fn(tu.HammerToTorchDataset(ds).__getitems__(b))
                    for b in self.batches
                ]
                received = list(self._loader(ds, idx, False))
            for where, got in (("in-process", local), ("workers", received)):
                with self.subTest(f"{tag}/{where}"):
                    self.assertGreater(len(got), 1)
                    spans = [_spans(s) for s in got]
                    self.assertTrue(all(spans))
                    for i in range(len(spans)):
                        for j in range(i + 1, len(spans)):
                            self.assertEqual(
                                _overlaps(spans[i], spans[j]), [],
                                f"{tag}/{where}: batches {i} and {j} share storage",
                            )

    @unittest.skipUnless(torch.cuda.is_available(), "pin_memory needs a GPU")
    def test_pin_once(self) -> None:
        ds = self._ds()
        dev = torch.device("cuda", torch.cuda.current_device())
        for lv in _LEVELS:
            for j, batch in enumerate(self.batches):
                ref = self._build(ds, batch, 0)
                one = self._build(ds, batch, lv)
                with _one(lv):
                    p_one = self._build(ds, batch, lv).pin_memory()
                    p_ref = self._build(ds, batch, 0).pin_memory()  # per-tensor fallback
                with _one(0):
                    p_one_pt = self._build(ds, batch, lv).pin_memory()  # knob off in the pinning process
                self.assertEqual(len(_storages(p_one)), 1)
                self._check_layout(p_one, lv, f"pinned/L{lv}/b{j}")
                self.assertEqual(len(_storages(p_ref)), 6)
                self.assertEqual(len(_storages(p_one_pt)), 6)
                for s in (p_one, p_ref, p_one_pt):
                    self.assertTrue(all(t.is_pinned() for t in _tensors(s)))
                self.assertEqual(_cmp_samples(ref, [one, p_one, p_ref, p_one_pt], f"pin/L{lv}/b{j}"), [])
                # device copies of FRESH batches (the comparisons above ran the
                # KJT accessors, whose lazy caches .to() would carry over); the
                # H2D copy copies each tensor on its own: no aliasing on device
                with _one(lv):
                    d_one = self._build(ds, batch, lv).pin_memory().to(dev, non_blocking=True)
                d_ref = self._build(ds, batch, 0).to(dev)
                torch.cuda.synchronize()
                dts = _tensors(d_one)
                self.assertEqual(len(dts), 6)
                self.assertTrue(all(t.device.type == "cuda" for t in dts))
                self.assertEqual(len(_storages(d_one)), 6, f"dev/L{lv}/b{j}: device tensors alias")
                self.assertEqual(_cmp_samples(d_ref, [d_one], f"dev/L{lv}/b{j}"), [])

    @unittest.skipUnless(torch.cuda.is_available(), "pin_memory needs a GPU")
    def test_pin_once_fallbacks(self) -> None:
        ds = self._ds()
        b = self.batches[0]
        with _one(2):
            mixed = self._build(ds, b, 2)
            mixed.candidates_features_kjt._lengths = mixed.candidates_features_kjt._lengths.clone()
            self.assertFalse(dsmod._pin_one_storage(mixed))
            pinned = self._build(ds, b, 2).pin_memory()
            self.assertFalse(dsmod._pin_one_storage(pinned))  # already pinned
            again = pinned.pin_memory()
        self.assertEqual(_cmp_samples(self._build(ds, b, 0), [again], "again"), [])

    @unittest.skipUnless(torch.cuda.is_available(), "pin_memory needs a GPU")
    def test_free_shm_after_pin(self) -> None:
        """DL_PIN_FREE_SHM (hardened implementation): after pin_memory the
        source batch's shared pages are removed (reads give zeros), the pinned
        batch equals the reference; a one-storage batch is released ONCE (one
        storage, not skipped by the size guard), a per-tensor batch six times;
        heap storages and the knob-off path are left alone. One "armed OK:
        [onestore-free-shm]" line per process, only for one-storage batches."""
        ds = self._ds()
        b = self.batches[0]
        with mock.patch.object(dsmod, "_FREE_SHM_ONESTORE_PID", None), mock.patch.object(
            dsmod, "_FREE_SHM_ARMED_PID", None
        ), self.assertLogs(dsmod.logger, "WARNING") as cm:
            for lv in (0,) + _LEVELS:
                for free in (False, True):
                    src = self._build(ds, b, lv)
                    keep = _tensors(src)
                    for t in keep:
                        t.share_memory_()  # like a batch received from a worker
                    self.assertEqual(len(_storages(src)), 6 if lv == 0 else 1)
                    before = [t.clone() for t in keep]
                    s0 = dsmod.pin_free_shm_stats()
                    with _one(lv), mock.patch.object(dsmod, "_FREE_SHM", free):
                        pinned = src.pin_memory()
                    d = {k: v - s0[k] for k, v in dsmod.pin_free_shm_stats().items()}
                    self.assertTrue(all(t.is_pinned() for t in _tensors(pinned)))
                    self.assertEqual(len(_storages(pinned)), 6 if lv == 0 else 1)
                    self.assertEqual(_cmp_samples(self._build(ds, b, 0), [pinned], f"free/L{lv}/{free}"), [])
                    nz = [t for t, c in zip(keep, before) if c.numel() and int(c.abs().sum())]
                    self.assertTrue(nz)
                    if free:
                        self.assertTrue(all(int(t.abs().sum()) == 0 for t in nz), f"L{lv}: pages not removed")
                        want = 6 if lv == 0 else 1
                        self.assertEqual((d["batches"], d["freed"]), (1, want), f"L{lv}")
                        self.assertEqual(
                            (d["onestore_batches"], d["onestore_freed"]), (0, 0) if lv == 0 else (1, 1), f"L{lv}"
                        )
                        for k in ("skip_size", "skip_alias", "skip_file", "madvise_err", "errors"):
                            self.assertEqual(d[k], 0, f"L{lv} {k}")
                    else:
                        self.assertTrue(all(torch.equal(t, c) for t, c in zip(keep, before)), f"L{lv}")
                        self.assertEqual(set(d.values()), {0}, f"L{lv}")
        one_lines = [r for r in cm.output if "armed OK: [onestore-free-shm]" in r]
        self.assertEqual(len(one_lines), 1, cm.output)
        self.assertIn("DL_FAST_BATCH_ONESTORE=2 (code default) DL_PIN_FREE_SHM=1 (code default)", one_lines[0])
        self.assertIn("freed 1 shared storage(s)", one_lines[0])
        self.assertIn("kept 0", one_lines[0])
        self.assertEqual(len([r for r in cm.output if "armed OK: [pin-free-shm]" in r]), 1, cm.output)
        self.assertEqual([r for r in cm.output if "[pin-free-shm] WARN" in r], [])
        heap = self._build(ds, b, 2)
        hk = _tensors(heap)
        hb = [t.clone() for t in hk]
        s0 = dsmod.pin_free_shm_stats()
        with _one(2), mock.patch.object(dsmod, "_FREE_SHM", True):
            heap.pin_memory()
        d = {k: v - s0[k] for k, v in dsmod.pin_free_shm_stats().items()}
        self.assertTrue(all(torch.equal(t, c) for t, c in zip(hk, hb)))  # not shared: untouched
        self.assertEqual((d["batches"], d["freed"], d["onestore_batches"]), (1, 0, 1))

    def test_free_shm_size_guard_covers_onestore(self) -> None:
        """The size guard (storage nbytes <= 2 x the UNION of the batch's byte
        ranges + 64 KiB) never skips a one-storage batch: the merged
        values/lengths (which the UIH/candidate halves alias) cover the storage
        up to the 4 KiB alignment padding. CPU only: _free_shared_pages on the
        shared source, a heap copy as the pinned replacement."""
        ds = self._ds()
        for lv in _LEVELS:
            for j, batch in enumerate(self.batches):
                src = self._build(ds, batch, lv)
                ts = _tensors(src)
                ts[0].share_memory_()  # one storage: every view follows
                self.assertEqual(len(_storages(src)), 1)
                st = ts[0].untyped_storage()
                self.assertTrue(st.is_shared())
                spans = [
                    (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                    for t in ts
                    if t.numel()
                ]
                own = dsmod._covered_bytes(spans)
                pad = st.nbytes() - own
                self.assertGreaterEqual(pad, 0, f"L{lv}/b{j}")
                # padding: < 4 KiB per aligned slice (the merged values and lengths)
                self.assertLess(pad, 2 * 4096, f"L{lv}/b{j}")
                self.assertLessEqual(st.nbytes(), 2 * own + dsmod._FREE_SHM_SLACK)
                ref = [t.clone() for t in ts]
                r = dsmod._free_shared_pages(ts, ref)
                self.assertEqual(
                    (r["freed"], r["skip_size"], r["skip_alias"], r["skip_file"], r["madvise_err"]),
                    (1, 0, 0, 0, 0),
                    f"L{lv}/b{j}",
                )
                self.assertEqual(r["freed_bytes"], st.nbytes())
                nz = [t for t in ref if t.numel() and int(t.abs().sum())]
                self.assertTrue(nz)
                self.assertTrue(all(int(t.abs().sum()) == 0 for t in ts), f"L{lv}/b{j}: pages not removed")
                # the pinned replacement aliasing the source: never released
                src2 = self._build(ds, batch, lv)
                ts2 = _tensors(src2)
                ts2[0].share_memory_()
                r = dsmod._free_shared_pages(ts2, ts2)
                self.assertEqual((r["freed"], r["skip_alias"]), (0, 1))

    def test_free_shm_once_lines_under_threads(self) -> None:
        """A process runs several pin threads (train pools, eval pool): the
        once-per-process lines are decided under the stats lock, so threads
        releasing their first batches at the same instant print each line
        exactly once. CPU only (_release_source_pages on shared copies, a heap
        copy as the pinned replacement)."""
        import threading

        ds = self._ds()
        n = 8
        jobs = []
        for i in range(n):
            lv = 2 if i % 2 else 0
            src = self._build(ds, self.batches[0], lv)
            ts = _tensors(src)
            for t in ts:
                t.share_memory_()
            jobs.append((ts, [t.clone() for t in ts], bool(lv)))
        gate = threading.Barrier(n)

        def run(job):
            gate.wait()
            dsmod._release_source_pages(job[0], job[1], one_store=job[2])

        s0 = dsmod.pin_free_shm_stats()
        with mock.patch.object(dsmod, "_FREE_SHM_ARMED_PID", None), mock.patch.object(
            dsmod, "_FREE_SHM_ONESTORE_PID", None
        ), self.assertLogs(dsmod.logger, "WARNING") as cm:
            th = [threading.Thread(target=run, args=(j,)) for j in jobs]
            for t in th:
                t.start()
            for t in th:
                t.join()
        d = {k: v - s0[k] for k, v in dsmod.pin_free_shm_stats().items()}
        self.assertEqual(len([r for r in cm.output if "armed OK: [pin-free-shm]" in r]), 1, cm.output)
        self.assertEqual(len([r for r in cm.output if "armed OK: [onestore-free-shm]" in r]), 1, cm.output)
        self.assertEqual((d["batches"], d["onestore_batches"], d["onestore_freed"]), (n, n // 2, n // 2))
        self.assertEqual(d["freed"], (n // 2) * 6 + (n // 2) * 1)

    @unittest.skipUnless(torch.cuda.is_available(), "pin_memory needs a GPU")
    def test_pin_once_line_under_threads(self) -> None:
        """8 threads pinning one-storage batches at the same instant (as the
        train pools' and the eval pool's pin threads can): one "armed OK:
        [onestore]" line per process, every batch pinned as one storage."""
        import threading

        ds = self._ds()
        n = 8
        srcs = [self._build(ds, self.batches[i % 3], 2) for i in range(n)]
        gate = threading.Barrier(n)
        res = [None] * n

        def run(i):
            gate.wait()
            res[i] = dsmod._pin_one_storage(srcs[i])

        with _one(2), mock.patch.object(dsmod, "_PIN_ONCE_LOGGED_PID", None), self.assertLogs(
            dsmod.logger, "WARNING"
        ) as cm:
            th = [threading.Thread(target=run, args=(i,)) for i in range(n)]
            for t in th:
                t.start()
            for t in th:
                t.join()
        self.assertEqual(res, [True] * n)
        lines = [r for r in cm.output if "armed OK: [onestore]" in r]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertIn(
            "armed OK: [onestore] DL_FAST_BATCH_ONESTORE=2 (code default): batch pinned as one storage",
            lines[0],
        )
        for i, s in enumerate(srcs):
            self.assertEqual(len(_storages(s)), 1)
            self.assertTrue(all(t.is_pinned() for t in _tensors(s)))
            self._check_layout(s, 2, f"thread{i}")
            self.assertEqual(_cmp_samples(self._build(ds, self.batches[i % 3], 0), [s], f"t{i}"), [])

    def test_knob_parsing_default_off(self) -> None:
        """The env-only code default (no gin binding): unset, empty, 0, < 0 or
        not an integer = off; any value > 0 = the layout, resolved to level 2;
        read once per process."""
        for env, want in ((None, False), ("0", False), ("x", False), ("1", True)):
            with mock.patch.object(dsmod, "_FREE_SHM", None), mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DL_PIN_FREE_SHM", None)
                if env is not None:
                    os.environ["DL_PIN_FREE_SHM"] = env
                self.assertEqual(dsmod.pin_free_shm_enabled(), want, env)
        for env, want in ((None, 0), ("0", 0), ("", 0), ("x", 0), ("-1", 0), ("1", 2), ("2", 2), ("7", 2)):
            with mock.patch.object(dsmod, "_ONESTORE", None), mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DL_FAST_BATCH_ONESTORE", None)
                if env is not None:
                    os.environ["DL_FAST_BATCH_ONESTORE"] = env
                self.assertEqual(dsmod.fast_batch_onestore_level(), want, env)
                self.assertEqual(dsmod.fast_batch_onestore_enabled(), want > 0, env)
                self.assertEqual(
                    dsmod.fast_batch_onestore_knob_source(), "env" if (env or "").strip() else "code default"
                )
                # the arming text names a raw value other than the resolved level
                raw = (env or "").strip()
                shown = "code default" if not raw else ("env" if raw == str(want) else f"env: {raw}")
                self.assertEqual(
                    dsmod.fast_batch_onestore_knob(), f"DL_FAST_BATCH_ONESTORE={want} ({shown})", env
                )
                os.environ["DL_FAST_BATCH_ONESTORE"] = "2" if not want else "0"
                self.assertEqual(dsmod.fast_batch_onestore_level(), want, "cached once per process")
        # the knob alone does not turn the fast builder on
        ds = self._ds()
        with _fast(False), _one(2):
            out = tu.HammerToTorchDataset(ds).__getitems__(self.batches[0])
        self.assertIsInstance(out, list)


if __name__ == "__main__":
    unittest.main()

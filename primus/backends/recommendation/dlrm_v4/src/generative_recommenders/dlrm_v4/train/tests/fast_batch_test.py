#!/usr/bin/env python3
# pyre-unsafe
"""DL_FAST_BATCH: the numpy per-sample builder and the batched key-major builder
return batches bit-for-bit identical to the default per-sample Python-list path
+ collate_fn, on synthetic yambda stores (CPU only, no dataset files).

Covers empty / short / scan-window-clipped / truncated histories (including the
zero and negative UIH budgets the slicing code admits), both history
strategies, 0/1/2 candidates, is_inference, the _active index path, item ids
past the artist/album tables, the vectorized XXH64 cross hash, the
collate_fn pass-through, the fallback for layouts build_batch does not cover,
the knob parsing, and real fork DataLoader workers (shared-memory hand-off)."""

import os
import unittest
from unittest import mock

import numpy as np
import torch

from generative_recommenders.dlrm_v4.configs import get_hstu_configs, YAMBDA_5B_CROSS_SPECS
from generative_recommenders.dlrm_v4.datasets import dataset as dsmod
from generative_recommenders.dlrm_v4.datasets.dataset import (
    collate_fn,
    DLRMv4RandomDataset,
    Samples,
)
from generative_recommenders.dlrm_v4.datasets.utils import (
    xxhash_cross,
    xxhash_cross_np,
    xxhash_cross_np_supported,
)
from generative_recommenders.dlrm_v4.datasets.yambda import DLRMv4YambdaDataset
from generative_recommenders.dlrm_v4.train import utils as tu

_ACCESSORS = (
    "keys", "values", "lengths", "offsets", "weights_or_none", "stride",
    "stride_per_key", "variable_stride_per_key", "length_per_key", "offset_per_key",
)


class _Store:
    pass


def _make_store(seed: int = 0, n_users: int = 24):
    rng = np.random.default_rng(seed)
    # A few one-to-five-event users (empty / near-empty histories), the rest
    # long enough that the scan window (not the user start) bounds the scan.
    lens = np.concatenate([[1, 2, 3, 5], rng.integers(20, 9000, n_users - 4)]).astype(np.int64)
    uid = np.repeat(np.arange(n_users, dtype=np.int64) * 3 + 1, lens)  # sparse uid space
    n = int(lens.sum())
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    st = _Store()
    st.flat_uid = uid
    st.flat_item_ids = rng.integers(0, 1300, n).astype(np.int64)  # some past both tables
    st.flat_timestamps = (
        1_600_000_000 + np.cumsum(rng.integers(1, 4000, n))
    ).astype(np.int64)
    st.flat_event_types = rng.choice([0, 1, 2, 3], p=[0.75, 0.08, 0.12, 0.05], size=n).astype(np.int64)
    pr = (rng.random(n) * 100).astype(np.float32)
    pr[rng.random(n) < 0.05] = 50.0  # the listen+ threshold itself
    st.flat_played_ratio = pr
    st.flat_is_organic = rng.integers(0, 2, n).astype(np.int8)
    is_listen = st.flat_event_types == 0
    st.flat_is_listen_plus = is_listen & (pr >= 50)
    st.flat_is_like = st.flat_event_types == 1
    st.flat_is_skip = is_listen & (pr < 50)
    st.user_start = np.full(int(uid.max()) + 1, -1, dtype=np.int64)
    st.user_end = np.full(int(uid.max()) + 1, -1, dtype=np.int64)
    st.user_start[uid[starts]] = starts
    st.user_end[uid[starts]] = starts + lens
    positions = np.flatnonzero(is_listen).astype(np.int64)  # min_history = 0
    item_to_artist = rng.integers(0, 60, 1000).astype(np.int64)
    item_to_album = rng.integers(0, 90, 1100).astype(np.int64)
    return st, positions, item_to_artist, item_to_album


def _make_ds(
    store, positions, i2a, i2al, strategy="interleaved", history_length=600,
    scan_window=3000, max_seq_len=4096, mnc=1,
) -> DLRMv4YambdaDataset:
    hstu = get_hstu_configs("yambda-5b")
    hstu.max_seq_len = max_seq_len
    hstu.max_num_candidates = mnc
    hstu.max_num_candidates_inference = mnc
    ds = DLRMv4YambdaDataset.__new__(DLRMv4YambdaDataset)
    DLRMv4RandomDataset.__init__(ds, hstu_config=hstu, is_inference=False)
    ds.store = store
    ds._positions = positions
    ds._active = None
    ds.item_to_artist = i2a
    ds.item_to_album = i2al
    ds._history_length = history_length
    ds._scan_window = scan_window
    ds._history_strategy = strategy
    ds._cross_specs = [(nm, tuple(k), n, s) for (nm, k, n, s) in YAMBDA_5B_CROSS_SPECS]
    return ds


def _cmp(a, b, path, errs) -> None:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            errs.append(f"{path}: {type(a).__name__} vs {type(b).__name__}")
        elif (a.dtype, tuple(a.shape), a.stride()) != (b.dtype, tuple(b.shape), b.stride()):
            errs.append(
                f"{path}: {a.dtype}{tuple(a.shape)}/{a.stride()} vs {b.dtype}{tuple(b.shape)}/{b.stride()}"
            )
        elif not torch.equal(a, b):
            errs.append(f"{path}: values differ")
        return
    if type(a) is not type(b):
        errs.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, dict):
        if list(a) != list(b):
            errs.append(f"{path}: keys")
        for k in a:
            if k in b:
                _cmp(a[k], b[k], f"{path}[{k}]", errs)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            errs.append(f"{path}: len {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp(x, y, f"{path}[{i}]", errs)
    elif a != b:
        errs.append(f"{path}: {a!r} vs {b!r}")


def _cmp_kjts(ref, cands, path, errs) -> None:
    """Whole-object state of ref vs each candidate: the attribute dicts are
    snapshotted for ALL objects before any accessor runs (accessors fill the
    lazy caches), then the accessor results, then the post-accessor state."""
    objs = [ref] + list(cands)
    pre = [dict(vars(o)) for o in objs]
    outs = [[getattr(o, m)() for m in _ACCESSORS] for o in objs]
    post = [dict(vars(o)) for o in objs]
    for j in range(1, len(objs)):
        p = f"{path}/c{j}"
        if set(pre[0]) != set(pre[j]):
            errs.append(f"{p}: attrs {sorted(set(pre[0]) ^ set(pre[j]))}")
        for k in set(pre[0]) & set(pre[j]):
            _cmp(pre[0][k], pre[j][k], f"{p}.{k}", errs)
        for m, x, y in zip(_ACCESSORS, outs[0], outs[j]):
            _cmp(x, y, f"{p}.{m}()", errs)
        for k in set(post[0]) & set(post[j]):
            _cmp(post[0][k], post[j][k], f"{p}.{k}<post>", errs)


def _cmp_kjt(a, b, path, errs) -> None:
    _cmp_kjts(a, [b], path, errs)


def _cmp_samples(ref, cands, path) -> list:
    errs = []
    if not isinstance(cands, (list, tuple)):
        cands = [cands]
    for c in cands:
        if type(ref) is not type(c) or sorted(vars(ref)) != sorted(vars(c)):
            return [f"{path}: {type(ref).__name__} vs {type(c).__name__}"]
    for f in ("uih_features_kjt", "candidates_features_kjt", "merged_sparse_features"):
        _cmp_kjts(getattr(ref, f), [getattr(c, f) for c in cands], f"{path}.{f}", errs)
    return errs


def _fast(on: bool):
    return mock.patch.object(dsmod, "_FAST_BATCH", on)


def _ref(ds, batch) -> Samples:
    with _fast(False):
        return collate_fn(tu.HammerToTorchDataset(ds).__getitems__(batch))


class _ListSampler(torch.utils.data.Sampler):
    def __init__(self, idx) -> None:
        self._idx = list(idx)

    def __iter__(self):
        return iter(self._idx)

    def __len__(self) -> int:
        return len(self._idx)


class FastBatchTest(unittest.TestCase):
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
            rng.choice(g, 37, replace=False).tolist(),  # unsorted, odd size
            [int(g[-1])],
        ]
        cls.n_empty = int((off == 0).sum())

    def _ds(self, **kw) -> DLRMv4YambdaDataset:
        return _make_ds(self.store, self.positions, self.i2a, self.i2al, **kw)

    def _check_all(self, ds, tag: str) -> None:
        htd = tu.HammerToTorchDataset(ds)
        for j, batch in enumerate(self.batches):
            ref = _ref(ds, batch)
            with _fast(True):
                got = htd.__getitems__(batch)
                self.assertIsInstance(got, Samples, f"{tag}: build_batch fell back")
                self.assertIs(collate_fn(got), got)
                ds.load_query_samples(batch)
                per_sample = collate_fn([ds.get_sample(i) for i in batch])
                ds.unload_query_samples(batch)
            errs = _cmp_samples(ref, [got, per_sample], f"{tag}/b{j}")
            mnc = ds._max_num_candidates_inference if ds._is_inference else ds._max_num_candidates
            for i in batch[:8]:
                fp = ds.iloc(i)
                with _fast(False):
                    a = ds._build_sample(fp, mnc)
                with _fast(True):
                    b = ds._build_sample(fp, mnc)
                _cmp_kjt(a[0], b[0], f"{tag}/s{i}/uih", errs)
                _cmp_kjt(a[1], b[1], f"{tag}/s{i}/cand", errs)
            self.assertEqual(errs, [], f"{tag} batch {j}: {errs[:5]}")

    def test_store_has_edge_cases(self) -> None:
        self.assertGreater(self.n_empty, 0)
        self.assertTrue((self.store.flat_item_ids >= len(self.i2a)).any())
        self.assertTrue((self.store.flat_item_ids >= len(self.i2al)).any())

    def test_identical_across_configs(self) -> None:
        n_ctx = 8
        cases = [
            ("interleaved", dict()),
            ("trunc", dict(max_seq_len=300)),
            ("budget0", dict(max_seq_len=1 + n_ctx)),
            ("budget_neg", dict(max_seq_len=n_ctx - 3)),
            ("last_n", dict(strategy="last_n")),
            ("last_n_trunc", dict(strategy="last_n", history_length=900, max_seq_len=500)),
            ("last_n_L0", dict(strategy="last_n", history_length=0)),
            ("per_pool_1", dict(history_length=2)),
            ("wide_scan", dict(scan_window=20000)),
            ("mnc2", dict(mnc=2)),
            ("mnc0", dict(mnc=0)),
        ]
        for tag, kw in cases:
            with self.subTest(tag):
                self._check_all(self._ds(**kw), tag)

    def test_is_inference_and_active(self) -> None:
        ds = self._ds()
        ds._is_inference = True
        ds._max_num_candidates_inference = 2
        self._check_all(ds, "inference")
        ds = self._ds()
        rng = np.random.default_rng(5)
        ds.set_active_indices(rng.permutation(len(self.positions))[:3000])
        saved = self.batches
        try:
            self.batches = [list(range(0, 64)), list(range(2999, 2900, -3))]
            self._check_all(ds, "active")
        finally:
            self.batches = saved

    def test_xxhash_np_matches_xxhash(self) -> None:
        rng = np.random.default_rng(11)
        cols = [
            np.concatenate([rng.integers(-(1 << 62), 1 << 62, 500), [0, -1, (1 << 63) - 1, -(1 << 63)]])
            .astype(np.int64)
            for _ in range(3)
        ]
        names = ["a", "b", "c"]
        for n_keys in (2, 3):
            for table in (1, 7, 24_000_000, 100_000_000, 1 << 63):
                for salt in (0, 1, 987654321, (1 << 64) - 1):
                    self.assertTrue(xxhash_cross_np_supported(n_keys, table, salt))
                    got = xxhash_cross_np(cols[:n_keys], table, salt)
                    want = [
                        xxhash_cross({k: int(c[i]) for k, c in zip(names, cols)}, names[:n_keys], table, salt)
                        for i in range(len(cols[0]))
                    ]
                    self.assertEqual(got.dtype, np.int64)
                    self.assertEqual(got.tolist(), want, (n_keys, table, salt))
        self.assertFalse(xxhash_cross_np_supported(4, 10, 0))
        self.assertFalse(xxhash_cross_np_supported(1, 10, 0))
        self.assertFalse(xxhash_cross_np_supported(2, 0, 0))
        self.assertFalse(xxhash_cross_np_supported(2, (1 << 63) + 1, 0))
        self.assertFalse(xxhash_cross_np_supported(2, 10, -1))

    def test_unsupported_layout_falls_back(self) -> None:
        ds = self._ds()
        ds._candidates_keys = ds._candidates_keys[:5]
        with _fast(True):
            self.assertIsNone(ds.build_batch(self.batches[0]))
            out = tu.HammerToTorchDataset(ds).__getitems__(self.batches[0])
        self.assertIsInstance(out, list)
        bad = self._ds()
        bad.item_to_artist = self.i2a.astype(np.int32)
        with _fast(True):
            self.assertIsNone(bad.build_batch(self.batches[0]))
            self.assertIsNone(self._ds().build_batch([]))

    def test_knob_parsing_default_off(self) -> None:
        for env, want in ((None, False), ("0", False), ("", False), ("x", False), ("1", True), ("2", True)):
            with mock.patch.object(dsmod, "_FAST_BATCH", None), mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DL_FAST_BATCH", None)
                if env is not None:
                    os.environ["DL_FAST_BATCH"] = env
                self.assertEqual(dsmod.fast_batch_enabled(), want, env)
                os.environ["DL_FAST_BATCH"] = "1" if not want else "0"
                self.assertEqual(dsmod.fast_batch_enabled(), want, "cached once per process")
        ds = self._ds()
        with _fast(False):
            out = tu.HammerToTorchDataset(ds).__getitems__(self.batches[0])
        self.assertIsInstance(out, list)

    def test_dataloader_workers_identical(self) -> None:
        ds = self._ds()
        idx = np.arange(len(self.positions))[::7][:256].tolist()
        got = {}
        for on in (False, True):
            with _fast(on):
                dl = torch.utils.data.DataLoader(
                    tu.HammerToTorchDataset(ds), batch_size=32, sampler=_ListSampler(idx),
                    collate_fn=collate_fn, drop_last=True, num_workers=2, prefetch_factor=2,
                    multiprocessing_context="fork",
                )
                got[on] = list(dl)
        self.assertEqual(len(got[False]), 8)
        for j, (a, b) in enumerate(zip(got[False], got[True])):
            self.assertEqual(_cmp_samples(a, b, f"dl/b{j}"), [])


if __name__ == "__main__":
    unittest.main()

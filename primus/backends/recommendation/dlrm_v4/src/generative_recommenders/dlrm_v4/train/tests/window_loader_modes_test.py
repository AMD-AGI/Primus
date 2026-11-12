# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""The streaming train loader's boundary modes deliver identical batches.

``_PrefetchingWindowLoader`` + ``_WarmWindowStream`` are run on CPU with a fake
windowed dataset (windows longer than the prefetch depth, shorter ones, empty
ones; with and without a per-window cap) in two modes:

  * two alternating pools, the freed pool re-primed one-shot for the window
    after next (``window_single_pool`` off, DL_SINGLE_POOL=0),
  * one chained pool (``window_single_pool``).

Every mode must yield the same (window, batch) sequence -- the same per-rank
round-robin slices, drop_last per window and per-window counts -- as the
reference computed directly from the window index lists. The two-pool mode
must also prime every pool reset with its whole prefetch depth at once.
"""

import threading
import unittest
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from generative_recommenders.dlrm_v4.train import utils as tu


class _FakeWindows:
    """Stands in for the yambda dataset: window ts -> global anchor indices."""

    def __init__(self, sizes: List[int]) -> None:
        self._starts = np.cumsum([0] + sizes)
        self._sizes = sizes

    def train_window_indices(self, ts: int) -> np.ndarray:
        s = int(self._starts[ts])
        return np.arange(s, s + self._sizes[ts], dtype=np.int64)


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self, windows: _FakeWindows) -> None:
        self.dataset = windows

    def __len__(self) -> int:
        return 1 << 30

    def __getitem__(self, idx: int) -> int:
        return int(idx)


def _collate(samples):
    return torch.tensor(samples, dtype=torch.int64)


def _dl_factory(dataset, bs: int, nw: int, pf: int):
    def make(sampler):
        if isinstance(sampler, tu._ChainedWindowBatchSampler):
            sampler.set_batch_size(bs)
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=_collate,
                num_workers=nw,
                prefetch_factor=pf,
                persistent_workers=True,
                multiprocessing_context="fork",
            )
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            collate_fn=_collate,
            drop_last=True,
            num_workers=nw,
            prefetch_factor=pf,
            sampler=sampler,
            persistent_workers=True,
            multiprocessing_context="fork",
        )

    return make


def _reference(
    sizes,
    ts_list,
    rank,
    ws,
    bs,
    cap: Optional[int],
    skip: int = 0,
    fwt: int = 0,
    stop: Optional[Tuple[int, int]] = None,
):
    """Per-window reference. ``skip`` samples of the first window were already
    trained before a resume (``fwt`` = skip // bs batches count against its cap);
    ``stop`` = (ts, k) abandons window ts after k batches."""
    w = _FakeWindows(sizes)
    out = []
    for i, ts in enumerate(ts_list):
        g = w.train_window_indices(ts)
        n = (len(g) // ws) * ws
        per_rank = g[:n][rank::ws]
        if i == 0:
            per_rank = per_rank[skip:]
        nb = len(per_rank) // bs
        if cap is not None:
            nb = min(nb, max(0, cap - (fwt if i == 0 else 0)))
        if stop is not None and stop[0] == ts:
            nb = min(nb, stop[1])
        for b in range(nb):
            out.append((ts, per_rank[b * bs : (b + 1) * bs].tolist()))
    return out


def _make_loader(mode: str, sizes, rank, ws, bs):
    tu.window_single_pool = (lambda: True) if mode == "single" else (lambda: False)
    ds = _FakeDataset(_FakeWindows(sizes))
    return tu._PrefetchingWindowLoader(
        dataset=ds,
        sampler_factory=lambda: tu.StreamingWindowSampler(rank, ws),
        dl_factory=_dl_factory(ds, bs, nw=2, pf=2),
    )


def _shutdown(loader) -> None:
    for dl in loader._dls:
        it = getattr(dl, "_iterator", None)
        if it is not None:
            it._shutdown_workers()


def _run_per_window(
    mode: str,
    sizes,
    ts_list,
    rank,
    ws,
    bs,
    cap: Optional[int],
    stop: Optional[Tuple[int, int]] = None,
):
    """Per-window consumer (TP_WARM_WINDOWS=0 / TRAIN_PIPELINE=0 path): every
    window's iterator is consumed until StopIteration, the per-window cap or
    an early abandon."""
    orig_single = tu.window_single_pool
    try:
        loader = _make_loader(mode, sizes, rank, ws, bs)
        stream = loader.stream(
            ts_list, first_skip_samples=0, warm=False, per_window_cap=cap,
            first_window_trained=0,
        )
        out = []
        for ts, it, _n in stream:
            k = 0
            for batch in it:
                out.append((ts, batch.tolist()))
                k += 1
                if cap is not None and k >= cap:
                    break
                if stop is not None and stop[0] == ts and k >= stop[1]:
                    break
        _shutdown(loader)
        return out
    finally:
        tu.window_single_pool = orig_single


def _run(mode: str, sizes, ts_list, rank, ws, bs, cap: Optional[int], skip: int = 0):
    """Chained warm stream (the production pipeline path). ``skip`` = samples of
    the first window already trained before a mid-window resume."""
    orig_single = tu.window_single_pool
    try:
        loader = _make_loader(mode, sizes, rank, ws, bs)
        fwt = skip // bs
        stream = loader.stream(
            ts_list, first_skip_samples=skip, warm=True, per_window_cap=cap,
            first_window_trained=fwt,
        )
        warm = tu._WarmWindowStream(
            stream,
            per_window_cap=cap,
            first_window_trained=fwt,
        )
        out = []
        while True:
            try:
                ts, nb = warm.advance()
            except StopIteration:
                break
            for _ in range(nb):
                out.append((ts, next(warm).tolist()))
        _shutdown(loader)
        return out
    finally:
        tu.window_single_pool = orig_single


class WindowLoaderModesTest(unittest.TestCase):
    # bs 4, ws 2: per-rank windows of 0, 1, 3, 12, 40 and 9 batches against a
    # prefetch depth of 2 workers x 2 = 4 batches.
    SIZES = [0, 9, 26, 100, 323, 75, 2, 160]
    BS, WS = 4, 2

    def _check(self, cap: Optional[int], rank: int, skip: int = 0) -> None:
        ts_list = list(range(len(self.SIZES)))
        if skip:
            ts_list = ts_list[3:]  # resume inside a window with 12 batches/rank
        ref = _reference(
            self.SIZES, ts_list, rank, self.WS, self.BS, cap, skip=skip,
            fwt=skip // self.BS,
        )
        self.assertGreater(len(ref), 0)
        for mode in ("oneshot", "single"):
            got = _run(mode, self.SIZES, ts_list, rank, self.WS, self.BS, cap, skip=skip)
            self.assertEqual(got, ref, f"mode={mode} cap={cap} rank={rank} skip={skip}")

    def test_uncapped(self) -> None:
        self._check(cap=None, rank=0)
        self._check(cap=None, rank=1)

    def test_capped(self) -> None:
        self._check(cap=5, rank=1)

    def test_resume_mid_window(self) -> None:
        # first window resumed after 2 batches (8 samples) of its per-rank slice
        self._check(cap=None, rank=0, skip=2 * self.BS)
        self._check(cap=5, rank=1, skip=2 * self.BS)

    def test_per_window_consumer(self) -> None:
        # TP_WARM_WINDOWS=0 / TRAIN_PIPELINE=0: each window is consumed until
        # StopIteration (or the cap), one window is abandoned early.
        ts_list = list(range(len(self.SIZES)))
        for cap, stop in ((None, None), (5, None), (None, (4, 3))):
            ref = _reference(
                self.SIZES, ts_list, 1, self.WS, self.BS, cap, stop=stop
            )
            for mode in ("oneshot", "single"):
                got = _run_per_window(
                    mode, self.SIZES, ts_list, 1, self.WS, self.BS, cap, stop=stop
                )
                self.assertEqual(got, ref, f"mode={mode} cap={cap} stop={stop}")

    def test_two_pool_one_shot_prime(self) -> None:
        """DL_SINGLE_POOL=0: every pool reset sends its whole prefetch depth at
        once (the first two windows on the main thread, every later one in the
        background thread), the active pool then gets one index per pulled
        batch, and the batches equal the single pool's."""
        ts_list = list(range(len(self.SIZES)))
        depth = 2 * 2  # num_workers x prefetch_factor of _make_loader
        orig_single = tu.window_single_pool
        try:
            loader = _make_loader("oneshot", self.SIZES, 0, self.WS, self.BS)
            self.assertFalse(loader._single)
            self.assertEqual(len(loader._dls), 2)
            resets = []
            prepare = loader._prepare

            def recording_prepare(buf, ts, skip_samples=0):
                prepare(buf, ts, skip_samples=skip_samples)
                resets.append((
                    ts, buf, loader._iters[buf]._send_idx, len(loader._dls[buf]),
                    threading.current_thread() is threading.main_thread(),
                ))

            loader._prepare = recording_prepare
            warm = tu._WarmWindowStream(
                loader.stream(ts_list, warm=True), per_window_cap=None
            )
            got, sends = [], []
            while True:
                try:
                    ts, nb = warm.advance()
                except StopIteration:
                    break
                for k in range(1, nb + 1):
                    got.append((ts, next(warm).tolist()))
                    sends.append((ts, warm._active._send_idx, min(nb, depth + k)))
            _shutdown(loader)
        finally:
            tu.window_single_pool = orig_single
        n = len(ts_list)
        # The background resets of windows k+2 and k+3 run on two threads and can
        # finish in either order: compare per window, not in completion order.
        self.assertEqual(sorted(r[:2] for r in resets), [(ts, ts % 2) for ts in ts_list])
        self.assertEqual([r[4] for r in resets], [True, True] + [False] * (n - 2))
        for ts, _buf, sent, nb, _main in resets:
            self.assertEqual(sent, min(depth, nb), f"window {ts}: prime {sent}/{nb}")
        self.assertEqual([s[1] for s in sends], [s[2] for s in sends], sends)
        self.assertEqual(
            got, _run("single", self.SIZES, ts_list, 0, self.WS, self.BS, None)
        )


if __name__ == "__main__":
    unittest.main()

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
"""M4 speed-aware within-batch dealing (dlrm_v4/train/m4_partition.py), CPU only.

M4-OTF form: a window's N reaches the dealer as a sealed memfd (m4_otf); here the tests build
those memfds from a synthetic N array laid out like the withdrawn table (window order of the
offline ts index) and record every readiness boundary as all-ready unless a test says otherwise.
The OTF runtime itself (election, builder, readiness AND, gate) is m4_otf_test.py.

Invariants checked:
  * partition_owner: every block is an exact cover (exactly bs anchors per rank),
    deterministic, balanced toward the targets w_r / sum(w);
  * lag_k: the published boundary always precedes the earliest push of window j;
  * M4Dealer.deal over W emulated ranks: per block the ranks' sets are disjoint and
    cover exactly the round-robin block; every fallback (no factors, resume skip,
    index mismatch, N not ready) is byte-identical to the stock slice;
  * the costs handed to m4_speed sum to the block totals; owner hashes agree;
  * maybe_create: off -> None and m4_speed untouched; declines are identical; ON with a dataset
    without the offline ts index raises; a dataset configured otherwise than the on-the-fly N rule
    assumes declines before the helper starts (review C5); a second create closes the earlier unused dealer;
    activate() registers the boundary check (lazily), declines at prefetch depth 0 and when the
    run-start gate is closed;
  * factors: unusable vectors fall back to round-robin; only relative speed matters;
  * HelperBackend: the window N arrives as an fd and matches LocalBackend; a dead or hung helper
    poisons the backend (no stale replies); close() reaps;
  * boundary_check over a real 2-rank gloo group (needs torch): agree / mismatch / a window one
    rank does not hold is round-robin on both;
  * M4 v3 host-local dealing: hosts=None is byte-identical to global dealing (owner, load,
    owner hash) and a one-host map gives the same owner/load (its hash covers the map); with
    hosts every host keeps exactly its round-robin
    anchors per block (bs per rank, rank max N and every anchor's normalizer unchanged);
    host_map, the helper with a host map, and the full DLRM_HSTU_FP8_M4_HOST_LOCAL x
    DL_COLD_PREFETCH_STREAM x NNODES decision table of maybe_create (the walk decision is the
    walk's own two gates: cold_prefetch.mode() > 0 and cold_prefetch.resolve_stream()).
"""

import os
import sys
import types
import unittest
from typing import Dict, List, Optional

import numpy as np

from generative_recommenders.dlrm_v4.train import m4_partition as m4p


class _FakeSpeed:
    """Stands in for x4c19's m4_speed: versioned factors, cost sink, callbacks."""

    def __init__(self, factors: Dict[int, Optional[np.ndarray]]) -> None:
        self._f = factors
        self.costs: Dict[int, np.ndarray] = {}
        self.asked: List[int] = []
        self.callbacks = []
        self._enabled = True
        self.reason = ""

    def factors_for(self, k: int, timeout_s: float = 600.0):
        self.asked.append(k)
        if k not in self._f:
            raise TimeoutError(f"factors_for({k}) not published")
        return self._f[k]

    def note_window_costs(self, j: int, costs: np.ndarray) -> None:
        self.costs[j] = costs

    def register_boundary_callback(self, fn) -> None:
        self.callbacks.append(fn)

    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool, reason: str = "") -> None:
        self._enabled = bool(on)
        self.reason = reason


def _windows(sizes: List[int], seed: int = 0):
    rng = np.random.default_rng(seed)
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    tsi = np.arange(int(off[-1]), dtype=np.int64) * 3 + 7  # ascending global indices
    # user-major lengths: runs of equal N with occasional jumps
    n = np.empty(int(off[-1]), dtype=np.int16)
    i = 0
    while i < n.shape[0]:
        run = int(rng.integers(1, 200))
        n[i:i + run] = int(rng.integers(10, 4095))
        i += run
    return off, tsi, n


def _stock_slice(gi, rank, W, skip, bs, cap):
    n = (len(gi) // W) * W
    per = np.array(gi[:n][rank::W][skip:], copy=True)
    nb = len(per) // bs
    if cap is not None:
        nb = min(nb, max(0, int(cap)))
    return per, nb


def _memfd(arr) -> int:
    """A sealed int16 memfd holding ``arr`` (what the m4_otf builder publishes)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int16))
    fd = os.memfd_create("m4test", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    b = a.tobytes()
    while b:
        b = b[os.write(fd, b):]
    m4p.m4_otf.fcntl.fcntl(fd, m4p.m4_otf.F_ADD_SEALS, m4p.m4_otf.REQUIRED_SEALS | m4p.m4_otf.F_SEAL_SEAL)
    return fd


# Every runtime _otf() builds; tearDownModule() closes them (and with them the memfds they hold).
_RUNTIMES: List["m4p.m4_otf.OTFRuntime"] = []


def tearDownModule() -> None:
    """Close the OTF runtimes _otf() built, with the sealed window memfds they hold.

    The tests never close them. Left open, they keep 969 memfds in the test process after this
    module. Under a 1024 soft RLIMIT_NOFILE (the docker default in the Arbor correctness-gate
    container), a later module that forks DataLoader workers then hits EMFILE: the worker's queue
    feeder thread cannot share the batch storage ("unable to open shared memory object ...: Too
    many open files"), drops the batch, and the parent waits forever. Seen in
    onestore_item49_test.py::OneStoreItem49Test::test_fork_pools_train_and_eval on 2026-10-14.
    close() is idempotent: a later close() of the same runtime is a no-op, so no fd number can be
    closed twice."""
    _close_runtimes()


def _close_runtimes() -> None:
    """Close (idempotently) every runtime _otf() built so far. The classes that build dealers call it
    after every test, so at most one test's memfds (169) are open at once, not the module's 969."""
    while _RUNTIMES:
        _RUNTIMES.pop().close()


def _otf(rank, off, n, ts_list=None, held=None, ready=True):
    """An OTF runtime without a builder: holds the N of the windows ``held`` (default: all of ts_list,
    default ts_list = every window) and, with ``ready``, every readiness boundary recorded all-ready."""
    ts_list = list(range(len(off) - 1)) if ts_list is None else list(ts_list)
    sizes = [int(off[t + 1] - off[t]) for t in ts_list]
    rt = m4p.m4_otf.OTFRuntime(rank, ts_list, sizes)
    _RUNTIMES.append(rt)
    for j, t in enumerate(ts_list):
        if held is None or j in held:
            assert rt.hold(j, _memfd(n[int(off[t]):int(off[t + 1])]))
    if ready:
        for b in range(len(ts_list) + 1):
            rt.book.record(b, -1, -1)
    return rt


def _ref_compute_m4_table(n_table, off, lut, j, ts, n_anchors, nb, W, bs, wq, rank, oq=None, hosts=None):
    """The withdrawn 34d35d6 table form of compute_m4 (reference for the window-buffer form)."""
    G = W * bs
    lo = int(off[ts])
    blocks = np.asarray(n_table[lo:lo + nb * G]).reshape(nb, G)
    owner, load = m4p.partition_owner_load_np(blocks, lut, wq, bs, oq, hosts)
    pos = np.flatnonzero((owner == rank).reshape(-1)).astype(np.int64)
    h = m4p._h63(b"m4", np.int64([j, ts, n_anchors, nb]).tobytes(), wq.tobytes(),
                 (b"" if oq is None else np.asarray(oq, np.int64).tobytes()),
                 (b"" if hosts is None else b"hl" + np.asarray(hosts, np.int64).tobytes()), owner.tobytes())
    return pos, load[:, rank] * m4p.COST_QUANTUM_MS, h


class PartitionTest(unittest.TestCase):
    def test_exact_cover_and_balance(self) -> None:
        rng = np.random.default_rng(1)
        lut = m4p.cost_lut()
        for W, bs in ((8, 64), (16, 32), (32, 16)):
            G = W * bs
            n_blocks = rng.integers(10, 4095, size=(5, G)).astype(np.int16)
            g = rng.uniform(0.92, 1.08, size=W)
            wq = m4p.weights_from_factors(g, W)
            own = m4p.partition_owner(n_blocks, lut, wq, bs)
            self.assertEqual(own.shape, (5, G))
            for b in range(5):
                cnt = np.bincount(own[b].astype(np.int64), minlength=W)
                self.assertTrue(np.all(cnt == bs), cnt)
                cost = lut[n_blocks[b].astype(np.int64)]
                load = np.bincount(own[b].astype(np.int64), weights=cost, minlength=W)
                share = load / load.sum()
                want = wq / wq.sum()
                # within a few max-anchor costs of the target
                self.assertLess(np.max(np.abs(share - want) * load.sum()), 4 * cost.max())
            own2 = m4p.partition_owner(n_blocks.copy(), lut, wq.copy(), bs)
            self.assertTrue(np.array_equal(own, own2))

    def test_equal_weights_balances_cost(self) -> None:
        rng = np.random.default_rng(2)
        lut = m4p.cost_lut()
        W, bs = 8, 128
        n_blocks = rng.integers(10, 4095, size=(3, W * bs)).astype(np.int16)
        wq = m4p.weights_from_factors(np.ones(W), W)
        own = m4p.partition_owner(n_blocks, lut, wq, bs)
        cost = lut[n_blocks.astype(np.int64)]
        for b in range(3):
            load = np.bincount(own[b].astype(np.int64), weights=cost[b], minlength=W)
            self.assertLess(load.max() - load.min(), 2 * cost[b].max())

    def test_rejects_bad_inputs(self) -> None:
        lut = m4p.cost_lut()
        wq = m4p.weights_from_factors(np.ones(4), 4)
        with self.assertRaises(ValueError):
            m4p.partition_owner(np.zeros((2, 10), np.int16), lut, wq, 2)
        with self.assertRaises(ValueError):
            m4p.weights_from_factors(np.array([1.0, np.nan, 1.0, 1.0]), 4)
        with self.assertRaises(ValueError):
            m4p.weights_from_factors(np.ones(3), 4)


class NPreservingTest(unittest.TestCase):
    """partition_owner_load_np keeps every rank's batch max N (the attention normalizer) and
    every anchor's normalizer equal to the round-robin values; exact cover; bs per rank."""

    def _check(self, n_blocks, W, bs, g):
        lut = m4p.cost_lut()
        wq = m4p.weights_from_factors(g, W)
        own, load = m4p.partition_owner_load_np(n_blocks, lut, wq, bs)
        nb, G = n_blocks.shape
        for b in range(nb):
            o = own[b].astype(np.int64)
            self.assertTrue(np.all(np.bincount(o, minlength=W) == bs))
            rr_max = np.array([n_blocks[b, r::W].max() for r in range(W)])
            for r in range(W):
                self.assertEqual(n_blocks[b, o == r].max(), rr_max[r])
            # every anchor's normalizer unchanged: rr_max of its round-robin rank == rr_max of its new rank
            rr_rank = np.arange(G) % W
            self.assertTrue(np.array_equal(rr_max[rr_rank], rr_max[o]))
            cost = lut[n_blocks[b].astype(np.int64)]
            self.assertTrue(np.array_equal(np.bincount(o, weights=cost, minlength=W).astype(np.int64), load[b]))
        own2, _ = m4p.partition_owner_load_np(n_blocks.copy(), lut, wq.copy(), bs)
        self.assertTrue(np.array_equal(own, own2))
        return own, load

    def test_random_groups(self) -> None:
        rng = np.random.default_rng(11)
        for W, bs in ((8, 32), (16, 16), (32, 8)):
            n = rng.integers(10, 4095, size=(6, W * bs)).astype(np.int16)
            # force group structure: some ranks share the cap, some do not
            for b in range(6):
                caps = rng.choice([4095, 4000, 3900], size=W)
                for r in range(W):
                    col = n[b, r::W]
                    col[col > caps[r]] = caps[r]
                    col[rng.integers(0, bs)] = caps[r]
                    n[b, r::W] = col
            self._check(n, W, bs, rng.uniform(0.9, 1.1, size=W))

    def test_single_group_matches_balance(self) -> None:
        rng = np.random.default_rng(12)
        W, bs = 8, 64
        n = rng.integers(10, 4095, size=(3, W * bs)).astype(np.int16)
        n[:, :W] = 4095  # every rank's first anchor is the max: one group
        g = np.linspace(0.92, 1.08, W)
        own, load = self._check(n, W, bs, g)
        lut = m4p.cost_lut()
        for b in range(3):
            cost = lut[n[b].astype(np.int64)]
            share = load[b] / load[b].sum()
            want = (1.0 / g) / (1.0 / g).sum()
            self.assertLess(np.max(np.abs(share - want)) * load[b].sum(), 3 * cost.max())

    def test_offsets_shift_load_from_offset_rank(self) -> None:
        rng = np.random.default_rng(13)
        W, bs = 8, 64
        n = rng.integers(10, 4095, size=(4, W * bs)).astype(np.int16)
        n[:, :W] = 4095  # one group
        lut = m4p.cost_lut()
        wq = m4p.weights_from_factors(np.ones(W), W)
        F = np.zeros(W)
        F[7] = 2.0  # 2 ms of fixed work outside the span on rank 7
        own, load = m4p.partition_owner_load_np(n, lut, wq, bs, m4p.offsets_from_ms(F, W))
        own0, load0 = m4p.partition_owner_load_np(n, lut, wq, bs)
        for b in range(4):
            self.assertTrue(np.all(np.bincount(own[b].astype(np.int64), minlength=W) == bs))
            t = load[b] * m4p.COST_QUANTUM_MS + F
            t0 = load0[b] * m4p.COST_QUANTUM_MS + F
            self.assertLess(t.max() - t.min(), t0.max() - t0.min())
            self.assertLess(load[b][7], load0[b][7])

    def test_bs_one_is_round_robin(self) -> None:
        W = 8
        n = np.arange(2 * W, dtype=np.int16).reshape(2, W) + 10
        own, _ = m4p.partition_owner_load_np(n, m4p.cost_lut(), m4p.weights_from_factors(np.ones(W), W), 1)
        self.assertTrue(np.array_equal(own, np.tile(np.arange(W, dtype=np.int16), (2, 1))))


class LagTest(unittest.TestCase):
    def test_small_j_and_startup(self) -> None:
        self.assertEqual(m4p.lag_k(0, [], 48), -1)
        self.assertEqual(m4p.lag_k(1, [500], 48), -1)
        # startup pushes stop once queued >= depth: P[j] < depth -> no factors
        self.assertEqual(m4p.lag_k(5, [4, 5, 6, 7, 8], 48), -1)

    def test_bound_holds(self) -> None:
        rng = np.random.default_rng(3)
        for depth in (32, 48, 64):
            nbs = list(rng.integers(0, 300, size=200))
            for j in range(2, 200):
                k = m4p.lag_k(j, nbs, depth)
                P = np.concatenate([[0], np.cumsum(nbs[:j])])
                s = j - 2
                while s - 1 >= 0 and P[j] - P[s] < 2 * depth:
                    s -= 1
                if k >= 0:
                    self.assertGreaterEqual(P[s] - P[k + 1], depth + m4p.LAG_MARGIN_BATCHES)
                    self.assertLess(k, j - 1)
                    # newest such k
                    if k + 1 < s:
                        self.assertLess(P[s] - P[k + 2], depth + m4p.LAG_MARGIN_BATCHES)


def _dealer(r, W, off, tsi, n, sp, hosts=None, ts_list=None, held=None, ready=True, backend=None):
    be = m4p.LocalBackend(m4p.cost_lut(), W, r) if backend is None else backend
    d = m4p.M4Dealer(r, W, be, speed=sp, hosts=hosts)
    d.bind_index(off, tsi)
    d._otf = _otf(r, off, n, ts_list, held, ready)
    d.set_depth(8)
    d._margin = 0  # small synthetic windows; the margin rule itself is LagTest's
    return d


def _dealers(W, off, tsi, n, factors, rank_speed=None, hosts=None, held=None):
    out = []
    for r in range(W):
        sp = _FakeSpeed(factors) if rank_speed is None else rank_speed[r]
        out.append(_dealer(r, W, off, tsi, n, sp, hosts, held=held))
    return out


class DealTest(unittest.TestCase):
    def setUp(self) -> None:
        self.W, self.bs = 8, 16
        sizes = [5, 40, 300, 3, 2000, 1500, 0, 3000, 2500, 900]
        self.off, self.tsi, self.n = _windows(sizes, seed=4)
        self.sizes = sizes

    def tearDown(self) -> None:
        _close_runtimes()

    def _gi(self, ts):
        return self.tsi[self.off[ts]:self.off[ts + 1]]

    def test_rr_then_m4_exact_cover(self) -> None:
        W, bs = self.W, self.bs
        rng = np.random.default_rng(5)
        factors = {k: rng.uniform(0.9, 1.1, size=W) for k in range(1, 20)}
        factors[1] = None
        dl = _dealers(W, self.off, self.tsi, self.n, factors)
        modes = []
        for j, ts in enumerate(range(len(self.sizes))):
            gi = self._gi(ts)
            res = [d.deal(j, ts, gi, 0, None, bs) for d in dl]
            nbs = {nb for _, nb in res}
            self.assertEqual(len(nbs), 1)
            nb = res[0][1]
            hashes = {d.window_hash(j) for d in dl}
            self.assertEqual(len(hashes), 1)
            G = W * bs
            for b in range(nb):
                block = set(gi[b * G:(b + 1) * G].tolist())
                got = [set(res[r][0][b * bs:(b + 1) * bs].tolist()) for r in range(W)]
                self.assertEqual(sum(len(x) for x in got), G)
                self.assertEqual(set().union(*got), block)
                for r in range(W):
                    part = res[r][0][b * bs:(b + 1) * bs]
                    self.assertTrue(np.all(np.diff(part) > 0))  # dataset order inside a rank
            for r in range(W):
                self.assertIn(j, dl[r]._speed.costs)  # once per window on every rank (empty if nb == 0)
            if nb:
                tot = sum(dl[r]._speed.costs[j] for r in range(W))
                G = W * bs
                want = m4p.cost_lut()[self.n[self.off[ts]:self.off[ts] + nb * G].astype(np.int64)].reshape(nb, G).sum(1)
                self.assertTrue(np.allclose(tot, want * m4p.COST_QUANTUM_MS))
            modes.append(dl[0].counts["m4"])
        self.assertGreater(dl[0].counts["m4"], 0)
        self.assertGreater(dl[0].counts["rr"], 0)

    def test_rr_paths_are_stock(self) -> None:
        W, bs = self.W, self.bs
        # every version published as None (no valid boundary yet) -> round-robin
        dl = _dealers(W, self.off, self.tsi, self.n, {k: None for k in range(100)})
        for j, ts in enumerate(range(len(self.sizes))):
            gi = self._gi(ts)
            for r, d in enumerate(dl):
                per, nb = d.deal(j, ts, gi, 0, 7 if ts == 4 else None, bs)
                sp, snb = _stock_slice(gi, r, W, 0, bs, 7 if ts == 4 else None)
                self.assertEqual(nb, snb)
                self.assertTrue(np.array_equal(per, sp))
                self.assertEqual(per.dtype, sp.dtype)

    def test_skip_and_mismatch_and_coverage_fall_back(self) -> None:
        W, bs = self.W, self.bs
        factors = {k: np.linspace(0.9, 1.1, W) for k in range(1, 50)}
        # resumed first window with skip: stock slice
        d = _dealers(W, self.off, self.tsi, self.n, factors)[3]
        gi = self._gi(4)
        per, nb = d.deal(0, 4, gi, 2 * bs, None, bs)
        sp, snb = _stock_slice(gi, 3, W, 2 * bs, bs, None)
        self.assertTrue(np.array_equal(per, sp))
        self.assertEqual(nb, snb)
        # index mismatch (holdout-like subset): stock slice
        d = _dealers(W, self.off, self.tsi, self.n, factors)[2]
        sub = self._gi(5)[::2]
        per, nb = d.deal(0, 5, sub, 0, None, bs)
        sp, snb = _stock_slice(sub, 2, W, 0, bs, None)
        self.assertTrue(np.array_equal(per, sp))
        # N of the window not ready on every rank at boundary k + 1 (M4-OTF): stock slice
        nw = len(self.sizes)
        d = _dealer(1, W, self.off, self.tsi, self.n, _FakeSpeed(factors), held=set(range(nw)) - {8})
        d._otf.book = m4p.m4_otf.Readiness()
        for b in range(nw + 1):  # the AND of every boundary lacks window 8 (k=4: M4 otherwise)
            bits = sum(1 << (jj - b) for jj in range(b, nw) if jj != 8 and jj - b < 64)
            d._otf.book.record(b, m4p.m4_otf._i64(bits), 0)
        modes = {}
        for j in range(nw):
            gi = self._gi(j)
            per, nb = d.deal(j, j, gi, 0, None, bs)
            modes[j] = d._hash[j]
            if j == 8:
                sp, snb = _stock_slice(gi, 1, W, 0, bs, None)
                self.assertTrue(np.array_equal(per, sp))
                self.assertEqual(nb, snb)
        self.assertGreater(d._reasons.get("otf-not-ready", 0), 0)
        self.assertGreater(d.counts["m4"], 0)

    def test_factor_timeout_raises(self) -> None:
        W, bs = self.W, self.bs
        d = _dealers(W, self.off, self.tsi, self.n, {})[0]
        d.set_depth(1)
        d._margin = 0
        for j in range(len(self.sizes) - 1):
            try:
                d.deal(j, j, self._gi(j), 0, None, bs)
            except TimeoutError:
                return
        self.fail("an unpublished factor version must raise, never fall back silently")

    def test_out_of_order_raises(self) -> None:
        d = _dealers(self.W, self.off, self.tsi, self.n, {})[0]
        with self.assertRaises(RuntimeError):
            d.deal(3, 3, self._gi(3), 0, None, self.bs)


class HelperTest(unittest.TestCase):
    """The helper subprocess gets each window's N as a file descriptor (no path) and agrees with the
    in-process LocalBackend and with the withdrawn table form of compute_m4."""

    def test_helper_matches_local(self) -> None:
        W, bs = 8, 16
        off, tsi, n = _windows([500, 3000, 2600, 4100], seed=8)
        lut = m4p.cost_lut()
        wq = m4p.weights_from_factors(np.linspace(0.95, 1.05, W), W)
        for r in (0, 5):
            hb = m4p.HelperBackend(W, r, 60.0)
            lb = m4p.LocalBackend(lut, W, r)
            try:
                hb.ping()
                for ts in (1, 2, 3):
                    na = int(off[ts + 1] - off[ts])
                    nb = na // W // bs
                    fd = _memfd(n[off[ts]:off[ts + 1]])
                    try:
                        a = hb.m4(ts, ts, na, nb, bs, wq, fd=fd)
                        b = lb.m4(ts, ts, na, nb, bs, wq, fd=fd)
                    finally:
                        os.close(fd)
                    t = _ref_compute_m4_table(n, off, lut, ts, ts, na, nb, W, bs, wq, r)
                    for x in (b, t):
                        self.assertTrue(np.array_equal(a[0], x[0]))
                        self.assertTrue(np.allclose(a[1], x[1]))
                        self.assertEqual(a[2], x[2])
                    fd = _memfd(n[off[ts]:off[ts + 1]])
                    win = m4p.window_array(fd)
                    os.close(fd)  # the mapping keeps the window
                    self.assertFalse(win.flags.writeable)
                    rr = m4p.compute_rr_costs(win, lut, 0, nb, W, bs, r)
                    G = W * bs
                    want = lut[n[off[ts]:off[ts] + nb * G].reshape(nb, G)[:, r::W].astype(np.int64)].sum(1)
                    self.assertTrue(np.allclose(rr, want * m4p.COST_QUANTUM_MS))
                fd = _memfd(n[off[0]:off[1]])
                try:
                    with self.assertRaises(RuntimeError):
                        hb.m4(0, 0, 500, 99, bs, wq, fd=fd)  # beyond the window: helper error surfaces
                finally:
                    os.close(fd)
                self.assertEqual(hb.ping(), hb.pid)  # still in sequence after an error reply
            finally:
                hb.close()

    def test_helper_matches_local_with_host_map(self) -> None:
        """M4 v3: the helper subprocess and LocalBackend agree with a host map (and with offsets); the hash covers it."""
        W, bs = 8, 16
        off, tsi, n = _windows([500, 3000, 2600, 4100], seed=18)
        lut = m4p.cost_lut()
        wq = m4p.weights_from_factors(np.linspace(0.95, 1.05, W), W)
        oq = m4p.offsets_from_ms(np.linspace(0.0, 0.5, W), W)
        hosts = np.arange(W, dtype=np.int64) // 4
        for r in (0, 6):
            hb = m4p.HelperBackend(W, r, 60.0)
            lb = m4p.LocalBackend(lut, W, r)
            try:
                for ts in (1, 2, 3):
                    na = int(off[ts + 1] - off[ts])
                    nb = na // W // bs
                    fd = _memfd(n[off[ts]:off[ts + 1]])
                    try:
                        for o in (None, oq):
                            a = hb.m4(ts, ts, na, nb, bs, wq, o, hosts, fd)
                            b = lb.m4(ts, ts, na, nb, bs, wq, o, hosts, fd)
                            g = lb.m4(ts, ts, na, nb, bs, wq, o, None, fd)
                            t = _ref_compute_m4_table(n, off, lut, ts, ts, na, nb, W, bs, wq, r, o, hosts)
                            for x in (b, t):
                                self.assertTrue(np.array_equal(a[0], x[0]))
                                self.assertTrue(np.allclose(a[1], x[1]))
                                self.assertEqual(a[2], x[2])
                            self.assertNotEqual(a[2], g[2])
                    finally:
                        os.close(fd)
            finally:
                hb.close()

    def test_helper_never_gets_a_path(self) -> None:
        """The helper's command line and init message carry no path; its open files are only the
        interpreter's own (no file under a data directory, nothing but its socket and pipes)."""
        hb = m4p.HelperBackend(8, 0, 60.0)
        try:
            hb.ping()
            with open(f"/proc/{hb.pid}/cmdline", "rb") as f:
                argv = [a for a in f.read().split(b"\0") if a]
            self.assertEqual(len(argv), 7)  # python -B -c <code: imports this module> fd ppid pdeath
            self.assertTrue(all(a.isdigit() for a in argv[4:]), argv)
            self.assertFalse(any(b".npy" in a for a in argv), argv)
            for fd in os.listdir(f"/proc/{hb.pid}/fd"):
                try:
                    tgt = os.readlink(f"/proc/{hb.pid}/fd/{fd}")
                except OSError:
                    continue
                self.assertFalse(tgt.endswith(".npy"), tgt)
        finally:
            hb.close()


class _RecordingSpeed(_FakeSpeed):
    def __init__(self, factors) -> None:
        super().__init__(factors)
        self.calls: List[str] = []

    def factors_for(self, k: int, timeout_s: float = 600.0):
        self.calls.append("factors_for")
        return super().factors_for(k, timeout_s)

    def register_boundary_callback(self, fn) -> None:
        self.calls.append("register_boundary_callback")
        super().register_boundary_callback(fn)

    def enabled(self) -> bool:
        self.calls.append("enabled")
        return super().enabled()

    def set_enabled(self, on: bool, reason: str = "") -> None:
        self.calls.append("set_enabled")
        super().set_enabled(on, reason)


# The production configuration of the yambda dataset object (yambda_5b.gin defaults, get_dataset, the
# yambda-5b HSTU config): the one the on-the-fly N rule assumes (m4_otf.nrule_config_mismatch; review C5).
_YAMBDA_CONFIG = {
    "_history_strategy": "interleaved",
    "_history_length": 4086,
    "_scan_window": 20000,
    "_max_seq_len": 4096,
    "_is_inference": False,
    "_max_num_candidates": 1,
    "_max_num_candidates_inference": 1,
    "_contextual_feature_to_max_length": {k: 1 for k in (
        "uid", "user_x_artist", "user_x_album", "user_x_hour", "item_x_hour", "artist_x_hour", "user_x_is_organic",
        "user_x_artist_x_hour")},
}


class _FakeDataset:
    """The dataset attributes M4 reads: configuration before RUN_START, the window index after it."""

    def __init__(self, off, tsi, cache_dir="") -> None:
        self._off, self._tsi_all = off, tsi
        self._cache_dir = cache_dir
        self._positions_name = "positions_L4086.npy"
        self._streaming_window_seconds = 86400
        self._use_offline_ts_indices = True
        self._ts_offsets = None
        self._ts_indices = None
        self.index_reads = 0
        for k, v in _YAMBDA_CONFIG.items():
            setattr(self, k, dict(v) if isinstance(v, dict) else v)

    def _ensure_streaming_index(self) -> None:
        self.index_reads += 1
        self._ts_offsets, self._ts_indices = self._off, self._tsi_all


def _stub_otf_start(rank, ts_list, sizes, paths, lws, rs, name=None):
    """OTFRuntime.start without election/builder (CreateTest is about the dealer)."""
    return m4p.m4_otf.OTFRuntime(rank, ts_list, sizes)


class CreateTest(unittest.TestCase):
    def setUp(self) -> None:
        from generative_recommenders.dlrm_v4.train import dl_prestart

        self._dlp = dl_prestart
        dl_prestart._reset_for_tests()
        self._env = dict(os.environ)
        self._start = m4p.m4_otf.OTFRuntime.start
        m4p.m4_otf.OTFRuntime.start = staticmethod(_stub_otf_start)
        m4p._ACTIVE.clear()
        m4p._CREATED.clear()
        self.fake = _RecordingSpeed({})
        mod = types.ModuleType("generative_recommenders.dlrm_v4.train.m4_speed")
        for name in ("factors_for", "note_window_costs", "register_boundary_callback", "enabled", "set_enabled"):
            setattr(mod, name, getattr(self.fake, name))
        sys.modules["generative_recommenders.dlrm_v4.train.m4_speed"] = mod
        import generative_recommenders.dlrm_v4.train as pkg

        self._had = hasattr(pkg, "m4_speed")
        pkg.m4_speed = mod

    def tearDown(self) -> None:
        for d in list(m4p._CREATED) + list(m4p._ACTIVE):
            d.close()
        m4p._ACTIVE.clear()
        m4p._CREATED.clear()
        m4p.m4_otf.OTFRuntime.start = self._start
        self._dlp._reset_for_tests()
        os.environ.clear()
        os.environ.update(self._env)
        sys.modules.pop("generative_recommenders.dlrm_v4.train.m4_speed", None)
        import generative_recommenders.dlrm_v4.train as pkg

        if not self._had and hasattr(pkg, "m4_speed"):
            delattr(pkg, "m4_speed")

    def _ds(self, sizes):
        off, tsi, n = _windows(sizes, seed=6)
        return _FakeDataset(off, tsi), n

    def test_off_returns_none(self) -> None:
        for v in (None, "0", "", "true", "on"):  # only "1" (stripped) turns it on
            if v is None:
                os.environ.pop(m4p.ENV_ON, None)
            else:
                os.environ[m4p.ENV_ON] = v
            ds, _ = self._ds([10, 20])
            self.assertIsNone(m4p.maybe_create(ds, 0, 8, single_pool=True))
            self.assertEqual(ds.index_reads, 0)
        self.assertEqual(self.fake.calls, [])  # OFF: m4_speed is never touched
        self.assertEqual(m4p._CREATED, [])

    def test_declines_identically(self) -> None:
        os.environ[m4p.ENV_ON] = "1"
        ds, _ = self._ds([10, 20])
        self.assertIsNone(m4p.maybe_create(ds, 0, 8, single_pool=False))
        self.assertFalse(self.fake.enabled())
        self.assertIn("DL_SINGLE_POOL", self.fake.reason)

    def test_dataset_without_offline_index_raises(self) -> None:
        os.environ[m4p.ENV_ON] = "1"
        ds, _ = self._ds([10, 20])
        ds._use_offline_ts_indices = False  # DL_WINDOW_INDEX=0
        with self.assertRaises(RuntimeError):
            m4p.maybe_create(ds, 0, 8, single_pool=True)
        with self.assertRaises(RuntimeError):
            m4p.maybe_create(types.SimpleNamespace(_ts_offsets=None), 0, 8, single_pool=True)
        self.assertEqual(ds.index_reads, 0)  # configuration only: the index is read after RUN_START

    def test_creates_then_activates(self) -> None:
        os.environ[m4p.ENV_ON] = "1"
        ds, n = self._ds([100, 200, 300])
        d = m4p.maybe_create(ds, 1, 8, single_pool=True)
        self.assertIsNotNone(d)
        self.assertEqual(ds.index_reads, 0)  # nothing read at loader construction
        self.assertEqual(self.fake.callbacks, [])  # registered only when the stream starts
        self._dlp.set_run_start()
        self.assertTrue(d.activate(48, [0, 1, 2]))
        self.assertEqual(ds.index_reads, 1)
        self.assertEqual(len(self.fake.callbacks), 1)
        self.assertIn(d, m4p._ACTIVE)
        self.assertEqual(d._otf.sizes, [100, 200, 300])
        with self.assertRaises(RuntimeError):  # one active dealer per process
            d2 = m4p.maybe_create(ds, 1, 8, single_pool=True)
            d2.activate(48, [0])

    def test_gate_closed_declines_without_reading(self) -> None:
        """activate() before RUN_START in this process (a resume, no MLPerf logger): no index read, M4
        declined through m4_speed (its arming gather turns every rank off), helper reaped."""
        os.environ[m4p.ENV_ON] = "1"
        ds, n = self._ds([100, 200, 300])
        d = m4p.maybe_create(ds, 0, 8, single_pool=True)
        logger = "generative_recommenders.dlrm_v4.train.m4_partition"
        with self.assertLogs(logger, level="WARNING") as cm:
            self.assertFalse(d.activate(48, [0, 1, 2]))
        self.assertTrue(any("m4otf gate closed" in x for x in cm.output), cm.output)
        self.assertEqual(ds.index_reads, 0)
        self.assertFalse(self.fake.enabled())
        self.assertIn("m4otf gate closed", self.fake.reason)
        self.assertEqual(self.fake.callbacks, [])
        self.assertIsNone(d._otf)
        self.assertNotIn(d, m4p._ACTIVE)
        self.assertIsNotNone(d._be._p.returncode)  # helper stopped and reaped

    def test_depth_zero_declines_and_closes(self) -> None:
        os.environ[m4p.ENV_ON] = "1"
        ds, n = self._ds([100, 200, 300])
        d = m4p.maybe_create(ds, 0, 8, single_pool=True)
        self._dlp.set_run_start()
        self.assertFalse(d.activate(0, [0, 1]))
        self.assertFalse(self.fake.enabled())
        self.assertEqual(self.fake.callbacks, [])
        self.assertIsNotNone(d._be._p.returncode)  # helper stopped and reaped

    def test_second_create_closes_unused_dealer(self) -> None:
        os.environ[m4p.ENV_ON] = "1"
        ds, n = self._ds([100, 200, 300])
        d1 = m4p.maybe_create(ds, 0, 8, single_pool=True)  # prestart loader, then discarded
        d2 = m4p.maybe_create(ds, 0, 8, single_pool=True)
        self.assertTrue(d1._closed)
        self.assertIsNotNone(d1._be._p.returncode)
        self.assertEqual(m4p._CREATED, [d2])
        self._dlp.set_run_start()
        self.assertTrue(d2.activate(48, [0, 1]))
        self.assertEqual(len(self.fake.callbacks), 1)

    def test_nrule_config_declines_each_setting_before_the_helper(self) -> None:
        """Review C5 (iter59): a dataset configured otherwise than the on-the-fly N rule assumes DECLINES
        (configuration only, identical on every rank): one DECLINED line naming the setting, m4_speed off,
        no helper spawned, nothing read, no dealer kept."""
        os.environ[m4p.ENV_ON] = "1"
        spawned = []

        class _Spy(_DummyBackend):
            def __init__(self, *a, **k) -> None:
                spawned.append(a)
                super().__init__(*a, **k)

        real = m4p.HelperBackend
        m4p.HelperBackend = _Spy
        logger = "generative_recommenders.dlrm_v4.train.m4_partition"
        try:
            cases = [("_history_strategy", "last_n"), ("_history_length", 2039), ("_history_length", 4087),
                     ("_scan_window", 10000), ("_max_seq_len", 2048), ("_max_seq_len", 8192), ("_is_inference", True),
                     ("_max_num_candidates", 2), ("_contextual_feature_to_max_length", {"uid": 1}),
                     ("_scan_window", None)]  # None: the attribute is deleted (missing)
            for name, value in cases:
                ds, _ = self._ds([10, 20])
                if value is None:
                    delattr(ds, name)
                else:
                    setattr(ds, name, value)
                self.fake.set_enabled(True)
                m4p._DISABLED_REASON.clear()
                with self.assertLogs(logger, level="WARNING") as cm:
                    self.assertIsNone(m4p.maybe_create(ds, 3, 8, single_pool=True), name)
                lines = [x for x in cm.output if "DECLINED" in x]
                self.assertEqual(len(lines), 1, cm.output)
                self.assertIn("DLRMV4 m4 speed-aware dealing DECLINED on rank 3: the dataset configuration differs "
                              "from the m4otf N rule's (%s" % name, lines[0])
                self.assertIn("N would not be the model's sequence length", lines[0])
                self.assertFalse(self.fake.enabled(), name)  # every rank goes off at the arming gather
                self.assertIn(name, self.fake.reason)
                self.assertEqual(m4p._DISABLED_REASON, [self.fake.reason])
                self.assertEqual(spawned, [], name)  # before any helper spawn
                self.assertEqual(ds.index_reads, 0)
                self.assertEqual(m4p._CREATED, [])
        finally:
            m4p.HelperBackend = real

    def test_nrule_default_config_arms(self) -> None:
        """Review C5: the production configuration (the fake's default) passes the guard: dealer created with
        its helper, no DECLINED line, m4_speed on, activation after RUN_START."""
        os.environ[m4p.ENV_ON] = "1"
        ds, _ = self._ds([100, 200, 300])
        self.assertEqual(m4p.m4_otf.nrule_config_mismatch(ds), "")
        logger = "generative_recommenders.dlrm_v4.train.m4_partition"
        with self.assertLogs(logger, level="WARNING") as cm:
            d = m4p.maybe_create(ds, 0, 8, single_pool=True)
        self.assertIsNotNone(d)
        self.assertFalse(any("DECLINED" in x for x in cm.output), cm.output)
        self.assertTrue(any("DLRMV4 m4 dealer created rank=0/8" in x for x in cm.output), cm.output)
        self.assertTrue(self.fake.enabled())
        self.assertIsNone(d._be._p.poll())  # the helper is up
        self._dlp.set_run_start()
        self.assertTrue(d.activate(48, [0, 1, 2]))

    def test_decline_if_requested(self) -> None:
        os.environ.pop(m4p.ENV_ON, None)
        m4p.decline_if_requested(0, "x")
        self.assertEqual(self.fake.calls, [])
        os.environ[m4p.ENV_ON] = "1"
        m4p.decline_if_requested(0, "no single-pool train loader")
        self.assertFalse(self.fake.enabled())
        self.assertIn("single-pool", self.fake.reason)

    def test_stream_closed_calls_shutdown(self) -> None:
        got = []
        sp = _FakeSpeed({})
        sp.shutdown = got.append
        d = m4p.M4Dealer(0, 8, m4p.LocalBackend(m4p.cost_lut(), 8, 0), speed=sp)
        d._otf = m4p.m4_otf.OTFRuntime(0, [0], [8])
        d.stream_closed("done")
        self.assertEqual(got, ["done"])
        self.assertTrue(d._otf.closed())  # readiness waiters woken, windows closed
        m4p.M4Dealer(0, 8, None, speed=_FakeSpeed({})).stream_closed()  # no shutdown, no runtime: no-op


class FactorTest(unittest.TestCase):
    def tearDown(self) -> None:
        _close_runtimes()

    def test_scale_invariant(self) -> None:
        W = 32
        g = np.random.default_rng(21).uniform(0.9, 1.1, size=W)
        w = m4p.weights_from_factors(g, W)
        for c in (2.0 ** -10, 2.0 ** -3, 2.0 ** 4, 2.0 ** 10):  # exact binary scalings
            self.assertTrue(np.array_equal(m4p.weights_from_factors(g * c, W), w))
        for c in (1e-3, 0.1, 1e3):
            self.assertLessEqual(int(np.abs(m4p.weights_from_factors(g * c, W) - w).max()), 1)

    def test_unusable_vectors(self) -> None:
        W = 8
        self.assertEqual(m4p.factors_valid(np.ones(W), W), "")
        for bad in (np.zeros(W), np.ones(W - 1), np.r_[np.ones(W - 1), np.nan], np.r_[np.ones(W - 1), -1.0],
                    np.r_[np.ones(W - 1), 5.0], "x"):
            self.assertNotEqual(m4p.factors_valid(bad, W), "", bad)

    def test_bad_vector_deals_round_robin(self) -> None:
        W, bs = 8, 16
        sizes = [5000, 6000, 7000, 8000, 5000, 6000, 7000, 8000]
        off, tsi, n = _windows(sizes, seed=22)
        factors = {k: np.zeros(W) for k in range(1, 10)}
        dl = _dealers(W, off, tsi, n, factors)
        for d in dl:
            d.set_depth(1)
        for j in range(len(sizes)):
            gi = tsi[off[j]:off[j + 1]]
            for r, d in enumerate(dl):
                per, nb = d.deal(j, j, gi, 0, None, bs)
                sp, snb = _stock_slice(gi, r, W, 0, bs, None)
                self.assertTrue(np.array_equal(per, sp))
                self.assertEqual(nb, snb)
        self.assertEqual(dl[0].counts["m4"], 0)
        self.assertGreater(dl[0]._reasons.get("bad-factors", 0), 0)

    def test_permuted_window_falls_back(self) -> None:
        W, bs = 8, 16
        off, tsi, n = _windows([5000, 6000], seed=23)
        d = _dealers(W, off, tsi, n, {k: np.linspace(0.9, 1.1, W) for k in range(1, 9)})[4]
        gi = np.random.default_rng(0).permutation(tsi[off[1]:off[2]])  # same length, shuffled
        per, nb = d.deal(0, 1, gi, 0, None, bs)
        sp, _ = _stock_slice(gi, 4, W, 0, bs, None)
        self.assertTrue(np.array_equal(per, sp))
        self.assertEqual(d._reasons, {"indices-not-window-slice": 1})


class HelperFailureTest(unittest.TestCase):
    def _hb(self, timeout_s=60.0):
        return m4p.HelperBackend(8, 0, timeout_s)

    def _m4(self, hb):
        off, tsi, n = _windows([500, 3000], seed=24)
        fd = _memfd(n[off[1]:off[2]])
        try:
            return hb.m4(1, 1, int(off[2] - off[1]), 2, 16, m4p.weights_from_factors(np.ones(8), 8), fd=fd)
        finally:
            os.close(fd)

    def test_killed_helper_poisons(self) -> None:
        hb = self._hb()
        try:
            hb.ping()
            self._m4(hb)
            hb._p.kill()
            hb._p.wait(10)
            with self.assertRaisesRegex(RuntimeError, "m4 helper pid"):
                self._m4(hb)
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                self._m4(hb)
        finally:
            hb.close(1.0)

    def test_hung_helper_never_returns_stale_reply(self) -> None:
        import signal

        hb = self._hb(timeout_s=0.5)
        try:
            hb.ping()
            os.kill(hb.pid, signal.SIGSTOP)
            with self.assertRaisesRegex(RuntimeError, "no answer"):
                self._m4(hb)
            try:
                os.kill(hb.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            with self.assertRaisesRegex(RuntimeError, "unusable"):
                self._m4(hb)
            self.assertIsNotNone(hb._p.returncode)  # killed and reaped
        finally:
            hb.close(1.0)

    def test_close_reaps(self) -> None:
        import time as _t

        hb = self._hb()
        hb.ping()
        t0 = _t.monotonic()
        hb.close(5.0)
        self.assertLess(_t.monotonic() - t0, 5.0)
        self.assertIsNotNone(hb._p.returncode)
        with self.assertRaisesRegex(RuntimeError, "unusable"):
            hb.ping()
        hb.close(1.0)  # idempotent


def _gloo_worker(rank, world, store_path, scen, q):  # pragma: no cover - runs in a spawned process
    import logging

    import torch.distributed as dist

    logging.disable(logging.WARNING)
    # A per-run FileStore (a fresh temp file per test) instead of a fixed TCPStore port: two pytest processes on one
    # host (e.g. parallel test shards) can no longer rendezvous with each other ("Connection closed by peer").
    store = dist.FileStore(store_path, world)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
    try:
        W, bs = 8, 16
        off, tsi, n = _windows([4096, 8192, 8192, 8192, 8192, 8192], seed=25)
        # rank 1 sees a different vector (a uniform scaling would be a no-op: g is mean-normalised)
        tilt = np.r_[1.05, np.ones(W - 1)] if (scen == "mismatch" and rank == 1) else np.ones(W)
        fac = {k: np.linspace(0.9, 1.1, W) * tilt for k in range(1, 9)}
        if scen in ("ok", "mismatch"):
            d = _dealers(W, off, tsi, n, fac)[rank]  # every boundary pre-recorded all-ready
        else:
            # M4-OTF: readiness only from the gathered AND; "notready": rank 1 never got window 5's N
            held = set(range(6)) - ({5} if (scen == "notready" and rank == 1) else set())
            d = _dealer(rank, W, off, tsi, n, _FakeSpeed(fac), held=held, ready=False)
        d.set_depth(1)
        try:
            if scen in ("ok", "mismatch"):
                for j in range(6):
                    d.deal(j, j, tsi[off[j]:off[j + 1]], 0, None, bs)
                for i in range(6):
                    d.boundary_check(i)
            else:  # stream order: window i is pushed before boundary i
                for i in range(6):
                    d.deal(i, i, tsi[off[i]:off[i + 1]], 0, None, bs)
                    d.boundary_check(i)
            q.put((rank, "ok", d._checked, d.counts["m4"], dict(d._reasons)))
        except RuntimeError as e:
            q.put((rank, "raise", str(e)[:120], d.counts["m4"], dict(d._reasons)))
        finally:
            d.close()
    finally:
        dist.destroy_process_group()


try:
    import torch.distributed as _tdist  # noqa: F401

    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


@unittest.skipIf(not _HAVE_TORCH, "boundary_check needs torch.distributed (gloo)")
class BoundaryCheckTest(unittest.TestCase):
    def _run(self, scen):
        import multiprocessing as _mp
        import tempfile

        ctx = _mp.get_context("spawn")
        q = ctx.Queue()
        with tempfile.TemporaryDirectory(prefix="m4_gloo_") as d:
            store_path = os.path.join(d, "store")
            ps = [ctx.Process(target=_gloo_worker, args=(r, 2, store_path, scen, q)) for r in range(2)]
            for p in ps:
                p.start()
            got = sorted(q.get(timeout=300) for _ in ps)
            for p in ps:
                p.join(60)
        return got

    def test_agree(self) -> None:
        got = self._run("ok")
        self.assertEqual([g[1] for g in got], ["ok", "ok"], got)
        self.assertEqual([g[2] for g in got], [5, 5])  # barrier 0 carries no collective
        self.assertGreater(got[0][3], 0)

    def test_mismatch_raises_on_every_rank(self) -> None:
        got = self._run("mismatch")
        self.assertEqual([g[1] for g in got], ["raise", "raise"], got)
        self.assertTrue(all("MISMATCH" in g[2] for g in got))

    def test_readiness_and_engages(self) -> None:
        """Readiness from the one all_gather (int64 [hash, w0, w1]): window 5 (k=1) is ready at boundary 2."""
        got = self._run("otf")
        self.assertEqual([g[1] for g in got], ["ok", "ok"], got)
        self.assertEqual([g[2] for g in got], [5, 5])
        self.assertEqual([g[3] for g in got], [1, 1])

    def test_window_missing_on_one_rank_is_round_robin_everywhere(self) -> None:
        got = self._run("notready")
        self.assertEqual([g[1] for g in got], ["ok", "ok"], got)  # owner hashes agree: both dealt it round-robin
        self.assertEqual([g[3] for g in got], [0, 0])
        self.assertTrue(all(g[4].get("otf-not-ready") == 1 for g in got), got)


try:  # the chained sampler lives in train/utils.py (torchrec etc.: container only)
    from generative_recommenders.dlrm_v4.train import utils as _tu
except Exception:  # pragma: no cover - CPU venv without torchrec
    _tu = None


@unittest.skipIf(_tu is None, "train/utils.py needs the training container")
class ChainTest(unittest.TestCase):
    """_ChainedWindowBatchSampler with M4 over W emulated ranks: per global batch the ranks'
    batches are disjoint and cover exactly the round-robin global batch; without M4 (or
    without win_index) push() is the stock slice."""

    def tearDown(self) -> None:
        _close_runtimes()

    def _run(self, with_m4: bool):
        W, bs = 8, 16
        sizes = [5, 40, 300, 3, 2000, 1500, 0, 3000, 2500, 900]
        off, tsi, n = _windows(sizes, seed=9)
        rng = np.random.default_rng(10)
        factors = {k: rng.uniform(0.9, 1.1, size=W) for k in range(0, 50)}
        chains = []
        for r in range(W):
            c = _tu._ChainedWindowBatchSampler(r, W)
            c.set_batch_size(bs)
            if with_m4:
                c.m4 = _dealer(r, W, off, tsi, n, _FakeSpeed(factors))
            c.live = True
            chains.append(c)
        for j, ts in enumerate(range(len(sizes))):
            gi = tsi[off[ts]:off[ts + 1]]
            for c in chains:
                c.push(ts, gi, 0, None, win_index=j)
        for c in chains:
            c.finish()
        its = [iter(c) for c in chains]
        out = [list(it) for it in its]
        return W, bs, sizes, off, tsi, out, chains

    def test_m4_exact_cover(self) -> None:
        W, bs, sizes, off, tsi, out, chains = self._run(True)
        self.assertGreater(chains[0].m4.counts["m4"], 0)
        G = W * bs
        nbatch = len(out[0])
        self.assertTrue(all(len(o) == nbatch for o in out))
        # rebuild the global batches in window order
        glob = []
        for ts in range(len(sizes)):
            gi = tsi[off[ts]:off[ts + 1]]
            nb = (len(gi) // W) // bs
            glob += [set(gi[b * G:(b + 1) * G].tolist()) for b in range(nb)]
        self.assertEqual(len(glob), nbatch)
        for b in range(nbatch):
            got = [set(out[r][b]) for r in range(W)]
            self.assertEqual(sum(len(x) for x in got), G)
            self.assertEqual(set().union(*got), glob[b])

    def test_off_is_stock(self) -> None:
        W, bs, sizes, off, tsi, out, chains = self._run(False)
        for r in range(W):
            want = []
            for ts in range(len(sizes)):
                gi = tsi[off[ts]:off[ts + 1]]
                per = gi[: (len(gi) // W) * W][r::W]
                nb = len(per) // bs
                want += [per[b * bs:(b + 1) * bs].tolist() for b in range(nb)]
            self.assertEqual(out[r], want)



def _nblocks_with_groups(rng, nb, W, bs):
    """Random N blocks with round-robin max-N group structure (several ranks share a cap)."""
    n = rng.integers(10, 4095, size=(nb, W * bs)).astype(np.int16)
    for b in range(nb):
        caps = rng.choice([4095, 4094, 4000, 3999, 2500], size=W)  # adjacent maxima: catches key precedence slips
        for r in range(W):
            col = n[b, r::W]
            col[col > caps[r]] = caps[r]
            col[rng.integers(0, bs)] = caps[r]
            n[b, r::W] = col
    return n


class HostLocalTest(unittest.TestCase):
    """M4 v3: host-local dealing (groups keyed by (block, host, rr_max))."""

    def tearDown(self) -> None:
        _close_runtimes()

    def test_none_and_one_host_are_global(self) -> None:
        rng = np.random.default_rng(31)
        lut = m4p.cost_lut()
        for W, bs in ((8, 32), (16, 16), (32, 8)):
            n = _nblocks_with_groups(rng, 4, W, bs)
            wq = m4p.weights_from_factors(rng.uniform(0.95, 1.05, size=W), W)
            o0, l0 = m4p.partition_owner_load_np(n, lut, wq, bs)
            o1, l1 = m4p.partition_owner_load_np(n, lut, wq, bs, None, None)
            o2, l2 = m4p.partition_owner_load_np(n, lut, wq, bs, None, np.zeros(W, np.int64))
            for o, l in ((o1, l1), (o2, l2)):
                self.assertTrue(np.array_equal(o0, o) and np.array_equal(l0, l))
            oq = m4p.offsets_from_ms(np.linspace(0.0, 1.0, W), W)
            o3, l3 = m4p.partition_owner_load_np(n, lut, wq, bs, oq)
            o4, l4 = m4p.partition_owner_load_np(n, lut, wq, bs, oq, np.zeros(W, np.int64))
            self.assertTrue(np.array_equal(o3, o4) and np.array_equal(l3, l4))

    def test_hash_global_unchanged_and_hosts_hashed(self) -> None:
        """hosts=None: the owner hash is the v2 formula exactly; any explicit map (even an all-zero one, whose
        owner array equals the global one) is part of the hashed input."""
        sizes = [16 * 32 * 3 + 5, 16 * 32 * 2]
        off, _, n = _windows(sizes, seed=32)
        lut = m4p.cost_lut()
        W, bs, nb, j, ts, rank = 16, 32, 3, 3, 0, 5
        wq = m4p.weights_from_factors(np.linspace(0.95, 1.05, W), W)
        blocks = np.asarray(n[int(off[ts]):int(off[ts]) + nb * W * bs]).reshape(nb, W * bs)
        owner, _ = m4p.partition_owner_load_np(blocks, lut, wq, bs)
        v2 = m4p._h63(b"m4", np.int64([j, ts, sizes[0], nb]).tobytes(), wq.tobytes(), b"", owner.tobytes())
        win = n[int(off[ts]):int(off[ts + 1])]  # the window's N buffer (offset 0 = its first anchor)
        p0, c0, h0 = m4p.compute_m4(win, lut, j, ts, sizes[0], nb, W, bs, wq, rank)
        self.assertEqual(h0, v2)  # v2's hash formula, byte for byte
        pz, cz, hz = m4p.compute_m4(win, lut, j, ts, sizes[0], nb, W, bs, wq, rank, None, np.zeros(W, np.int64))
        self.assertTrue(np.array_equal(p0, pz) and np.allclose(c0, cz))  # same deal ...
        self.assertNotEqual(h0, hz)  # ... but the map is hashed
        hosts = np.arange(W, dtype=np.int64) // 8
        p2, _, h2 = m4p.compute_m4(win, lut, j, ts, sizes[0], nb, W, bs, wq, rank, None, hosts)
        own2, _ = m4p.partition_owner_load_np(blocks, lut, wq, bs, None, hosts)
        self.assertEqual(h2, m4p._h63(b"m4", np.int64([j, ts, sizes[0], nb]).tobytes(), wq.tobytes(), b"",
                                      b"hl" + hosts.tobytes(), own2.tobytes()))
        self.assertNotEqual(h0, h2)
        self.assertEqual(p2.shape[0], nb * bs)

    def test_host_local_invariants(self) -> None:
        rng = np.random.default_rng(33)
        lut = m4p.cost_lut()
        for W, bs, nh in ((16, 16, 2), (32, 8, 4), (32, 16, 2)):
            n = _nblocks_with_groups(rng, 5, W, bs)
            g = rng.uniform(0.95, 1.06, size=W)
            wq = m4p.weights_from_factors(g, W)
            hosts = np.arange(W, dtype=np.int64) // (W // nh)
            own, load = m4p.partition_owner_load_np(n, lut, wq, bs, None, hosts)
            own2, load2 = m4p.partition_owner_load_np(n.copy(), lut, wq.copy(), bs, None, hosts.copy())
            self.assertTrue(np.array_equal(own, own2) and np.array_equal(load, load2))
            G = W * bs
            rr_rank = np.arange(G) % W
            for b in range(n.shape[0]):
                o = own[b].astype(np.int64)
                self.assertTrue(np.all(np.bincount(o, minlength=W) == bs))
                # every anchor stays on its round-robin host -> each host's anchor set is its RR set
                self.assertTrue(np.array_equal(hosts[o], hosts[rr_rank]))
                rr_max = np.array([n[b, r::W].max() for r in range(W)])
                for r in range(W):
                    self.assertEqual(n[b, o == r].max(), rr_max[r])
                self.assertTrue(np.array_equal(rr_max[rr_rank], rr_max[o]))
                cost = lut[n[b].astype(np.int64)]
                self.assertTrue(np.array_equal(np.bincount(o, weights=cost, minlength=W).astype(np.int64), load[b]))

    def test_host_local_balances_inside_each_host(self) -> None:
        """One rr_max group per host (every rank's first anchor attains the cap): inside every host the
        predicted-time spread is far below round-robin's, and the hosts are not mixed."""
        rng = np.random.default_rng(36)
        lut = m4p.cost_lut()
        W, bs, nh = 32, 64, 4
        n = rng.integers(10, 4095, size=(4, W * bs)).astype(np.int16)
        n[:, :W] = 4095
        g = rng.uniform(0.94, 1.06, size=W)
        hosts = np.arange(W, dtype=np.int64) // (W // nh)
        own, load = m4p.partition_owner_load_np(n, lut, m4p.weights_from_factors(g, W), bs, None, hosts)
        rr_rank = np.arange(W * bs) % W
        for b in range(4):
            cost = lut[n[b].astype(np.int64)]
            t_hl = g * load[b]
            t_rr = g * np.bincount(rr_rank, weights=cost, minlength=W)
            for h in range(nh):
                sel = hosts == h
                self.assertLess(np.ptp(t_hl[sel]), 0.25 * np.ptp(t_rr[sel]))
                self.assertEqual(int(load[b][sel].sum()), int(cost[hosts[rr_rank] == h].sum()))

    def test_dealers_host_local_exact_cover(self) -> None:
        W, bs = 16, 8
        sizes = [5, 80, 600, 6, 4000, 3000, 0, 6000, 5000, 1800]
        off, tsi, n = _windows(sizes, seed=34)
        rng = np.random.default_rng(35)
        factors = {k: rng.uniform(0.9, 1.1, size=W) for k in range(1, 20)}
        hosts = np.arange(W, dtype=np.int64) // 8
        dl = _dealers(W, off, tsi, n, factors, hosts=hosts)
        pos_of = {int(v): i for i, v in enumerate(tsi.tolist())}
        engaged = 0
        for j, ts in enumerate(range(len(sizes))):
            gi = tsi[off[ts]:off[ts + 1]]
            res = [d.deal(j, ts, gi, 0, None, bs) for d in dl]
            self.assertEqual(len({nb for _, nb in res}), 1)
            self.assertEqual(len({d.window_hash(j) for d in dl}), 1)
            nb = res[0][1]
            G = W * bs
            for b in range(nb):
                block = gi[b * G:(b + 1) * G]
                for r in range(W):
                    part = res[r][0][b * bs:(b + 1) * bs]
                    self.assertTrue(np.all(np.diff(part) > 0))
                    # its block position's round-robin rank is on the same host
                    pos = np.array([pos_of[int(v)] - int(off[ts]) - b * G for v in part.tolist()])
                    self.assertTrue(np.all(hosts[pos % W] == hosts[r]))
                got = np.concatenate([res[r][0][b * bs:(b + 1) * bs] for r in range(W)])
                self.assertTrue(np.array_equal(np.sort(got), np.sort(block)))
            engaged = dl[0].counts["m4"]
        self.assertGreater(engaged, 0)

    def test_dealer_host_self_check_raises(self) -> None:
        """A host map lost on the way to the partition (backend deals globally) makes the dealer raise."""
        W, bs = 16, 8
        sizes = [5, 80, 600, 6, 4000, 3000, 0, 6000, 5000, 1800]
        off, tsi, n = _windows(sizes, seed=37)
        rng = np.random.default_rng(38)
        factors = {k: rng.uniform(0.9, 1.1, size=W) for k in range(1, 20)}
        hosts = np.arange(W, dtype=np.int64) // 8
        dl = _dealers(W, off, tsi, n, factors, hosts=hosts)
        for d in dl:  # drop the map inside the backend only: the dealer still believes it deals host-locally
            m = d._be.m4
            d._be.m4 = lambda *a, _m=m: _m(*a[:7], None, a[8])
        with self.assertRaises(RuntimeError):
            for j, ts in enumerate(range(len(sizes))):
                gi = tsi[off[ts]:off[ts + 1]]
                for d in dl:
                    d.deal(j, ts, gi, 0, None, bs)

    def test_host_map(self) -> None:
        env = dict(os.environ)
        try:
            os.environ.pop("GPUS_PER_NODE", None)
            h, why = m4p.host_map(16, 2)
            self.assertEqual(why, "")
            self.assertEqual(h.tolist(), [0] * 8 + [1] * 8)
            h, why = m4p.host_map(32, 4)
            self.assertEqual(h.tolist(), [i // 8 for i in range(32)])
            self.assertEqual(m4p.host_map(8, 1), (None, ""))
            h, why = m4p.host_map(12, 5)
            self.assertIsNone(h)
            self.assertIn("does not divide", why)
            os.environ["GPUS_PER_NODE"] = "8"
            self.assertEqual(m4p.host_map(16, 2)[1], "")
            os.environ["GPUS_PER_NODE"] = "4"
            h, why = m4p.host_map(16, 2)
            self.assertIsNone(h)
            self.assertIn("GPUS_PER_NODE", why)
            os.environ["GPUS_PER_NODE"] = "16"  # larger than world/NNODES: also a mismatch
            self.assertIsNone(m4p.host_map(16, 2)[0])
            os.environ["GPUS_PER_NODE"] = "0"  # train_ranker: 0 -> visible devices, not a check
            self.assertEqual(m4p.host_map(16, 2)[1], "")
            for bad in ([0] * 15, [0] * 15 + [16], [0] * 15 + [-1]):
                with self.assertRaises(ValueError):
                    m4p.host_keys(np.asarray(bad), 16)
        finally:
            os.environ.clear()
            os.environ.update(env)


class _DummyBackend:
    """HelperBackend stand-in for the mode table (no helper process)."""

    def __init__(self, W, rank, timeout_s, b_ms=0.0, c_ms=0.0) -> None:
        self.pid = 0

    def ping(self):
        return 0

    def close(self, timeout_s: float = 5.0) -> None:
        pass


class HostLocalModeTest(unittest.TestCase):
    """maybe_create: dealing mode and declines over the env grid (identical on every rank)."""

    def setUp(self) -> None:
        self._env = dict(os.environ)
        os.environ.pop("NODE_RANK", None)  # tests that need it set it
        self._hb = m4p.HelperBackend
        m4p.HelperBackend = _DummyBackend
        m4p._ACTIVE.clear()
        m4p._CREATED.clear()
        self.fake = _FakeSpeed({})
        mod = types.ModuleType("generative_recommenders.dlrm_v4.train.m4_speed")
        for name in ("factors_for", "note_window_costs", "register_boundary_callback", "enabled", "set_enabled"):
            setattr(mod, name, getattr(self.fake, name))
        sys.modules["generative_recommenders.dlrm_v4.train.m4_speed"] = mod
        import generative_recommenders.dlrm_v4.train as pkg

        self._had = hasattr(pkg, "m4_speed")
        pkg.m4_speed = mod
        from generative_recommenders.dlrm_v4.train import dl_prestart

        self._dlp = dl_prestart
        dl_prestart._reset_for_tests()
        dl_prestart.set_run_start()  # activate() below runs with the gate open
        self._start = m4p.m4_otf.OTFRuntime.start
        m4p.m4_otf.OTFRuntime.start = staticmethod(_stub_otf_start)
        self.ds = _FakeDataset(np.array([0, 100, 250], np.int64), np.arange(250))

    def tearDown(self) -> None:
        m4p.HelperBackend = self._hb
        m4p._ACTIVE.clear()
        m4p._CREATED.clear()
        os.environ.clear()
        os.environ.update(self._env)
        sys.modules.pop("generative_recommenders.dlrm_v4.train.m4_speed", None)
        import generative_recommenders.dlrm_v4.train as pkg

        if not self._had and hasattr(pkg, "m4_speed"):
            delattr(pkg, "m4_speed")
        m4p.m4_otf.OTFRuntime.start = self._start
        self._dlp._reset_for_tests()

    def _run(self, nnodes, stream, mode, gpn=None, prefetch="1"):
        for k in ("NNODES", "DL_COLD_PREFETCH_STREAM", m4p.ENV_HOST_LOCAL, "GPUS_PER_NODE", "DL_COLD_PREFETCH"):
            os.environ.pop(k, None)  # NODE_RANK: left as the test set it (setUp/tearDown restore the env)
        if prefetch is not None:  # the walk's first gate: cold_prefetch.mode() > 0 (production: DL_PRESTART=2)
            os.environ["DL_COLD_PREFETCH"] = prefetch
        os.environ[m4p.ENV_ON] = "1"
        if nnodes is not None:
            os.environ["NNODES"] = str(nnodes)
        if stream is not None:
            os.environ["DL_COLD_PREFETCH_STREAM"] = stream
        if mode is not None:
            os.environ[m4p.ENV_HOST_LOCAL] = mode
        if gpn is not None:
            os.environ["GPUS_PER_NODE"] = str(gpn)
        self.fake._enabled = True
        self.fake.reason = ""
        m4p._DISABLED_REASON.clear()
        W = 8 * (nnodes if isinstance(nnodes, int) and nnodes > 0 else 1)
        logger = "generative_recommenders.dlrm_v4.train.m4_partition"
        with self.assertLogs(logger, level="WARNING") as cm:
            d = m4p.maybe_create(self.ds, 3, W, single_pool=True)
        text = "\n".join(cm.output)
        for x in list(m4p._CREATED):
            x.close()
        m4p._CREATED.clear()
        return d, text, W

    def test_mode_table(self) -> None:
        n_cases = 0
        for nnodes in (None, 1, 2, 4):
            nn = nnodes or 1
            for stream in (None, "0", "1", "auto", "on"):
                walk = stream in ("1", "on") or (stream == "auto" and nn > 1)
                for mode in (None, "auto", " AUTO ", "1", "0", "bogus"):
                    d, text, W = self._run(nnodes, stream, mode)
                    m = (mode or "auto").strip().lower()
                    n_cases += 1
                    tag = f"nnodes={nnodes} stream={stream} mode={mode}"
                    if m not in ("auto", "1", "0"):
                        self.assertIsNone(d, tag)
                        self.assertIn("DECLINED", text, tag)
                        self.assertIn(m4p.ENV_HOST_LOCAL, self.fake.reason, tag)
                        self.assertFalse(self.fake._enabled, tag)
                        continue
                    self.assertIsNotNone(d, tag)
                    self.assertNotIn("DECLINED", text, tag)
                    hl = nn > 1 and (m == "1" or (m == "auto" and walk))
                    if hl:
                        self.assertEqual(d._hosts.tolist(), [r // 8 for r in range(W)], tag)
                        self.assertIn("dealing=host-local(hosts=%d)" % nn, text, tag)
                    else:
                        self.assertIsNone(d._hosts, tag)
                        self.assertIn("dealing=global", text, tag)
                    approx = nn > 1 and walk and m == "0"
                    self.assertEqual("walk coverage approximate" in text, approx, tag)
                    self.assertIn("walk=%s" % ("on" if walk else "off"), text, tag)
                    self.assertIn("(%s=%s, walk=" % (m4p.ENV_HOST_LOCAL, m), text, tag)
                    self.assertIn(", nnodes=%d)" % nn, text, tag)
        self.assertEqual(n_cases, 4 * 5 * 6)

    def test_gpus_per_node_mismatch_declines_only_host_local(self) -> None:
        d, text, _ = self._run(2, "1", "auto", gpn=4)
        self.assertIsNone(d)
        self.assertIn("GPUS_PER_NODE", self.fake.reason)
        d, text, _ = self._run(2, "0", "auto", gpn=4)  # global dealing does not need the host map
        self.assertIsNotNone(d)
        self.assertIsNone(d._hosts)
        d, text, _ = self._run(2, "1", "auto", gpn=8)
        self.assertEqual(d._hosts.tolist(), [r // 8 for r in range(16)])

    def test_walk_decision_is_cold_prefetch_resolve_stream(self) -> None:
        from generative_recommenders.dlrm_v4.train import cold_prefetch

        orig = cold_prefetch.resolve_stream
        try:
            cold_prefetch.resolve_stream = lambda: (True, "stub")
            d, text, _ = self._run(2, None, None)  # env says no walk; the walk's own decision says on
            self.assertEqual(d._hosts.tolist(), [r // 8 for r in range(16)])
            self.assertIn("walk=on (stub)", text)
            cold_prefetch.resolve_stream = lambda: (False, "stub-off")
            d, text, _ = self._run(2, "1", None)
            self.assertIsNone(d._hosts)
        finally:
            cold_prefetch.resolve_stream = orig

    def test_prefetcher_gate(self) -> None:
        """The walk needs the prefetcher armed (cold_prefetch.mode() > 0): DL_COLD_PREFETCH=0, or unset at
        pre-start level < 2, means no walk -> auto deals globally and the line says walk=off (prefetcher off)."""
        from generative_recommenders.dlrm_v4.train import cold_prefetch

        for pf in ("0", "x", None):
            d, text, _ = self._run(2, "1", "auto", prefetch=pf)
            self.assertIsNone(d._hosts, pf)
            self.assertIn("walk=off (prefetcher off", text, pf)
        saved = cold_prefetch._PRESTART_LEVEL[0]
        try:
            cold_prefetch.set_prestart_level(2)  # DL_PRESTART=2 arms the prefetcher without DL_COLD_PREFETCH
            d, text, _ = self._run(2, "1", "auto", prefetch=None)
            self.assertEqual(d._hosts.tolist(), [r // 8 for r in range(16)])
            self.assertIn("walk=on", text)
        finally:
            cold_prefetch.set_prestart_level(saved)

    def test_walk_on_matches_cold_prefetch_gates(self) -> None:
        """_walk_on == (mode() > 0 and resolve_stream()[0]) over an env grid; no cold_prefetch -> no walk."""
        from generative_recommenders.dlrm_v4.train import cold_prefetch

        for pf in (None, "", "0", "1", "2", "x"):
            for nnodes in (None, "1", "2", "4", "x"):
                for stream in (None, "0", "1", "true", "TRUE", "on", " On ", "auto", "AUTO", "yes", "bogus"):
                    for k, v in (("DL_COLD_PREFETCH", pf), ("NNODES", nnodes), ("DL_COLD_PREFETCH_STREAM", stream)):
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                    want = bool(cold_prefetch.mode()) and bool(cold_prefetch.resolve_stream()[0])
                    self.assertEqual(m4p._walk_on()[0], want, (pf, nnodes, stream))
        name = "generative_recommenders.dlrm_v4.train.cold_prefetch"
        saved = sys.modules.get(name)
        sys.modules[name] = None  # the import in _walk_on raises ImportError
        try:
            os.environ["DL_COLD_PREFETCH"] = "1"
            os.environ["DL_COLD_PREFETCH_STREAM"] = "1"
            self.assertEqual(m4p._walk_on()[0], False)
        finally:
            sys.modules[name] = saved

    def test_node_rank_layout_check(self) -> None:
        """host-local on a rank whose NODE_RANK is not its node-major host declines (locally); matching or
        unset NODE_RANK passes; global dealing never needs the check."""
        os.environ["NODE_RANK"] = "1"  # rank 3 of 16 is on host 0 in the node-major map
        d, text, _ = self._run(2, "1", "auto")
        self.assertIsNone(d)
        self.assertIn("not node-major", self.fake.reason)
        os.environ["NODE_RANK"] = "0"
        d, text, _ = self._run(2, "1", "auto")
        self.assertEqual(d._hosts.tolist(), [r // 8 for r in range(16)])
        os.environ["NODE_RANK"] = "1"
        d, text, _ = self._run(2, "0", "auto")  # walk off -> global: no host map, no check
        self.assertIsNotNone(d)
        self.assertIsNone(d._hosts)
        self.assertEqual(m4p._layout_check(9, np.arange(16) // 8), "")
        self.assertIn("not node-major", m4p._layout_check(3, np.arange(16) // 8))
        os.environ.pop("NODE_RANK", None)
        self.assertEqual(m4p._layout_check(3, np.arange(16) // 8), "")

    def test_armed_line_names_dealing(self) -> None:
        d, _, _ = self._run(2, "1", "auto")
        logger = "generative_recommenders.dlrm_v4.train.m4_partition"
        with self.assertLogs(logger, level="WARNING") as cm:
            self.assertTrue(d.activate(48, [0, 1]))
        self.assertTrue(any("armed OK: m4 speed-aware dealing" in x and "dealing=host-local(hosts=2)" in x
                            for x in cm.output), cm.output)
        d.close()
        m4p._ACTIVE.clear()  # one active dealer per process: this test activates two in turn
        d, _, _ = self._run(1, "1", "auto")
        with self.assertLogs(logger, level="WARNING") as cm:
            self.assertTrue(d.activate(48, [0, 1]))
        self.assertTrue(any("dealing=global" in x for x in cm.output), cm.output)
        d.close()


if __name__ == "__main__":
    unittest.main()

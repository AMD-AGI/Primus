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
"""M4-OTF (dlrm_v4/train/m4_otf.py + the M4Dealer hooks in m4_partition.py), CPU only.

Synthetic data only: every test writes a small fake dataset (the seven .npy files the builder
maps) into its own temporary directory. The N function the builder imports is the exact small
reference of the N rule below (``compute_window_n_incremental``; the module path is injected
through m4_otf.NFUNC_MODULE), so no x4c19 module is needed. Top-level imports stay light (stdlib +
numpy): the builder process imports this module.

Checked:
  * the reference N rule on edge cases (empty history -> 9, the 1362 per-pool cap, the
    p - 20000 scan start, user boundaries);
  * run-start gate: activate() before RUN_START declines through m4_speed and reads nothing
    (every open / os.open / np.load / mmap path of the trainer recorded, and the helper's and the
    trainer's /proc fd + maps, contain no path of the fake dataset); the release path re-checks
    the gate; the builder holds no dataset path before 'release' and maps the files after it;
  * election: several processes of one parent (multiprocessing spawn, like mp.start_processes)
    elect exactly one builder; every rank connects and receives every window (same bytes as the
    reference); the builder writes its per-window stderr line; a failing N function gives the
    one "m4otf builder failed" line;
  * publish: sealed memfd (SHRINK|GROW|WRITE|SEAL) of exactly 2 * anchors bytes via SCM_RIGHTS to
    every client, a late client gets the backlog; the received fd is immutable (write,
    ftruncate, writable shared mmap, write through /proc/self/fd, new seals all fail);
  * readiness: the AND over simulated ranks with per-rank drain delays (fake all_gather) gives
    identical decisions on every rank, equal to the AND of the delivery schedule; owner hashes
    agree and every block is an exact cover;
  * fallbacks: no connection, a dead builder, a missing N function module -> windows not ready
    -> round-robin (stock slice), never a stall;
  * owner arrays of the OTF path equal the withdrawn table form for the same N and factors;
  * deferred round-robin costs;
  * schedule (iter58 review): lag_k_all == lag_k; the elected rank's need list (pushed + predicted
    batch counts) is sent once and again only on a mismatch; pick_window (increasing order, horizon
    extended to decisions due within LEAD, jump only when in-order building would miss a deadline,
    back-fill once ahead, never an obsolete window); a slow real builder jumps;
  * election is sticky: after the elected builder died (killed, or failed at GO) a late rank never
    elects a second builder; peer credentials: the builder refuses a non-sibling client and a
    client beyond local_world_size, a rank refuses a listener that is not a sibling rank;
  * received window fds are close-on-exec; builder audit lines (every line t=<epoch ms>, in the
    order run_start <= started <= release <= go <= data mapped <= first memfd_create), its comm
    "dlrmv4-m4otf" and argv marker; the N module is imported at GO, not at release; DUTY rests;
  * M4 off for good (m4_speed): the elected rank stops the builder without waiting, pending
    round-robin costs are dropped and no later window computes them;
  * teardown: builder and helper reaped (no zombie), held fds and the client socket closed;
  * nothing written to disk (temp cwd, TMPDIR, fake data dir, no named shared memory; the
    builder never holds a writable file);
  * no MLLOG / seed logging in m4_otf.py and m4_partition.py; the gin binding (code row OFF,
    a gin bool; no N table anywhere; no absolute path in the item);
  * review O1 (iter59): with every socket on fd numbers >= 1100 (> FD_SETSIZE) the builder starts, takes
    release and GO and publishes every window (select() raised there); its poll() wait agrees with
    select() on timeout, data, a pending connection, EOF and a closed fd;
  * review C5 (iter59): the dataset configuration the N rule assumes (each setting differs, missing
    attributes differ; the constants are the rule's, m4_nfunc's and this tree's production defaults);
  * r3: the DLRM_HSTU_FP8_M4_OTF_GO_BATCHES default is 0, so the first progress sends 'go' (even with no
    trained batch) and the builder builds; the env still overrides it (PublishTest sets 32).
"""

import ast
import errno
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
from typing import Dict, List, Optional

import numpy as np

from generative_recommenders.dlrm_v4.train import m4_otf
from generative_recommenders.dlrm_v4.train import m4_partition as m4p

_THIS = "generative_recommenders.dlrm_v4.train.tests.m4_otf_test"
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_TRAIN = os.path.join(_ROOT, "generative_recommenders", "dlrm_v4", "train")
_LOG = "generative_recommenders.dlrm_v4.train.m4_otf"
_LOG_P = "generative_recommenders.dlrm_v4.train.m4_partition"


# -- the exact small reference of the N rule (test-only; x4c19 owns the production function) --------
def ref_n(p, user_start, flags) -> np.ndarray:
    """uid = searchsorted(user_start, p, 'right') - 1; s = max(user_start[uid], p - 20000);
    count_k = sum(flag_k[s:p]); N = min(4087, sum_k min(count_k, 1362)) + 9 (int16)."""
    p = np.asarray(p, dtype=np.int64)
    uid = np.searchsorted(user_start, p, side="right") - 1
    s = np.maximum(user_start[uid], p - 20000)
    tot = np.zeros(p.shape[0], dtype=np.int64)
    for f in flags:
        cs = np.concatenate([[0], np.cumsum(np.asarray(f, dtype=np.int64))])
        tot += np.minimum(cs[p] - cs[s], 1362)
    return (np.minimum(tot, 4087) + 9).astype(np.int16)


def compute_window_n_incremental(positions, user_start, flags, state, *, user_lo, user_hi):
    """Contract of m4_nfunc.compute_window_n_incremental, implemented by the reference rule. Test hooks
    (env of the builder): M4OTF_TEST_FAIL_AT=n raises at the (n+1)-th call; M4OTF_TEST_SLEEP_MS sleeps."""
    calls = 0 if state is None else int(state)
    fail_at = os.environ.get("M4OTF_TEST_FAIL_AT", "")
    if fail_at and calls >= int(fail_at):
        raise RuntimeError(f"injected failure at call {calls}")
    if os.environ.get("M4OTF_TEST_SLEEP_MS"):
        time.sleep(float(os.environ["M4OTF_TEST_SLEEP_MS"]) / 1e3)
    assert user_lo == 0 and user_hi == len(user_start) - 1
    return ref_n(positions, user_start, flags), calls + 1


# -- fake dataset ---------------------------------------------------------------------------------------
_FILES = ("positions_L4086.npy", "user_start.npy", "flat_is_listen_plus.npy", "flat_is_like.npy",
          "flat_is_skip.npy", "ts_indices_L4086_W86400.npy", "ts_offsets_L4086_W86400.npy")


def make_data(d: str, seed: int = 0, n_windows: int = 10, n_anchors: int = 12000) -> Dict[str, np.ndarray]:
    """The seven files the builder maps (users with uid holes; window-major ts index)."""
    rng = np.random.default_rng(seed)
    n_users = 30
    lens = rng.integers(40, 30000, n_users)
    lens[3] = 1  # a one-event user
    n_ev = int(lens.sum())
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    us = np.full(n_users + 7, -1, dtype=np.int64)
    us[np.sort(rng.choice(n_users + 7, n_users, replace=False))] = starts
    flags = [rng.random(n_ev) < q for q in (0.45, 0.04, 0.3)]
    flags[1][: n_ev // 3] |= rng.random(n_ev // 3) < 0.5  # saturate the like pool somewhere
    pos = np.sort(rng.choice(n_ev, n_anchors, replace=False)).astype(np.int64)
    win = rng.integers(0, n_windows, n_anchors)
    win[win == 2] = 3  # an empty window
    tsi = np.argsort(win, kind="stable").astype(np.int64)
    off = np.concatenate([[0], np.cumsum(np.bincount(win, minlength=n_windows))]).astype(np.int64)
    for name, a in zip(_FILES, (pos, us, flags[0], flags[1], flags[2], tsi, off)):
        np.save(os.path.join(d, name), a)
    csr = np.append(starts, np.int64(n_ev))
    table = ref_n(pos[tsi], csr, [f.view(np.uint8) for f in flags])  # window-major, like the withdrawn table
    return {"pos": pos, "tsi": tsi, "off": off, "csr": csr, "flags": flags, "table": table}


# The production dataset configuration (yambda_5b.gin defaults, get_dataset, the yambda-5b HSTU config):
# the one the on-the-fly N rule assumes (m4_otf.nrule_config_mismatch; review C5, iter59).
YAMBDA_CONFIG = {
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


class FakeDS:
    """The yambda dataset attributes M4 uses; the index is read only in _ensure_streaming_index."""

    def __init__(self, d: str) -> None:
        self._cache_dir = d
        self._positions_name = "positions_L4086.npy"
        self._streaming_window_seconds = 86400
        self._use_offline_ts_indices = True
        self._ts_offsets = None
        self._ts_indices = None
        for k, v in YAMBDA_CONFIG.items():
            setattr(self, k, dict(v) if isinstance(v, dict) else v)

    def _ensure_streaming_index(self) -> None:
        if self._ts_indices is None:
            self._ts_indices = np.load(os.path.join(self._cache_dir, _FILES[5]), mmap_mode="r")
            self._ts_offsets = np.load(os.path.join(self._cache_dir, _FILES[6]), mmap_mode="r")


class FakeSpeed:
    def __init__(self, factors=None) -> None:
        self.factors = {} if factors is None else factors
        self.costs: Dict[int, np.ndarray] = {}
        self.notes: List[tuple] = []
        self.callbacks = []
        self.enabled_ = True
        self.reason = ""

    def factors_for(self, k, timeout_s=600.0):
        return self.factors.get(k)

    def note_window_costs(self, j, costs) -> None:
        self.costs[j] = np.asarray(costs)
        self.notes.append((j, int(np.asarray(costs).shape[0])))

    def register_boundary_callback(self, fn) -> None:
        self.callbacks.append(fn)

    def enabled(self) -> bool:
        return self.enabled_

    def set_enabled(self, on, reason="") -> None:
        self.enabled_ = bool(on)
        self.reason = reason

    def shutdown(self, why="") -> None:
        pass


def _name() -> str:
    return "\0m4otf-test-%d-%s" % (os.getpid(), uuid.uuid4().hex[:12])


def _windows(sizes, seed=0):
    rng = np.random.default_rng(seed)
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    tsi = np.arange(int(off[-1]), dtype=np.int64) * 3 + 7
    n = rng.integers(9, 4096, size=int(off[-1])).astype(np.int16)
    return off, tsi, n


def _memfd(arr) -> int:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int16)).tobytes()
    fd = os.memfd_create("m4test", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    while a:
        a = a[os.write(fd, a):]
    m4_otf.fcntl.fcntl(fd, m4_otf.F_ADD_SEALS, m4_otf.REQUIRED_SEALS | m4_otf.F_SEAL_SEAL)
    return fd


def _proc_paths(pid: int) -> List[str]:
    """Every path in /proc/<pid>/fd and /proc/<pid>/maps (like the G3 poller)."""
    out = []
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                out.append(os.readlink(f"/proc/{pid}/fd/{fd}"))
            except OSError:
                pass
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) == 6:
                    out.append(parts[5].strip())
    except OSError:
        pass
    return out


def _wait(pred, timeout_s=60.0, step=0.02):
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(step)
    return pred()


class _Env:
    """Temporarily set env vars, NFUNC_MODULE and the dl_prestart run-start reference."""

    def __init__(self, run_start=True, nfunc=_THIS, **env) -> None:
        self.run_start, self.nfunc, self.env = run_start, nfunc, env

    def __enter__(self):
        from generative_recommenders.dlrm_v4.train import dl_prestart

        self._dlp = dl_prestart
        self._saved_env = dict(os.environ)
        self._nfunc = m4_otf.NFUNC_MODULE
        os.environ.update(self.env)
        m4_otf.NFUNC_MODULE = self.nfunc
        dl_prestart._reset_for_tests()
        if self.run_start:
            dl_prestart.set_run_start()
        return self

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self._saved_env)
        m4_otf.NFUNC_MODULE = self._nfunc
        self._dlp._reset_for_tests()
        return False


def _start(rank, data, d, name, lws=2, ts_list=None):
    ds = FakeDS(d)
    ts_list = list(range(len(data["off"]) - 1)) if ts_list is None else ts_list
    off = data["off"]
    sizes = [int(off[t + 1] - off[t]) for t in ts_list]
    return m4_otf.OTFRuntime.start(rank, ts_list, sizes, m4_otf.data_paths(ds), lws, time.time(), name=name)


# -- tests ----------------------------------------------------------------------------------------------
class RefRuleTest(unittest.TestCase):
    def test_edge_cases(self) -> None:
        us = np.array([0, 5, 30005, 30010], dtype=np.int64)  # users [0,5) [5,30005) [30005,30010)
        n_ev = 30010
        lp = np.zeros(n_ev, np.uint8)
        like = np.zeros(n_ev, np.uint8)
        skip = np.zeros(n_ev, np.uint8)
        lp[:] = 1
        like[5:5000] = 1
        p = np.array([0, 5, 6, 30004, 30005, 30009, 25005], dtype=np.int64)
        got = ref_n(p, us, (lp, like, skip))
        # p=0, p=5, p=30005: first event of a user -> empty history -> 9
        self.assertEqual(got[0], 9)
        self.assertEqual(got[1], 9)
        self.assertEqual(got[4], 9)
        self.assertEqual(got[2], 9 + 1 + 1)  # one listen+, one like before p=6 inside the user
        # p=30004: s = max(5, 30004 - 20000) = 10004: 20000 listen+ capped at 1362, likes in
        # [10004, 5000) none -> 1362 + 9
        self.assertEqual(got[3], 1362 + 9)
        self.assertEqual(got[5], 4 + 9)  # [30005, 30009): 4 listen+
        # p=25005: s = 5005: 20000 listen+ (cap 1362) + likes [5005, 5000) none
        self.assertEqual(got[6], 1362 + 9)
        like[:] = 1
        skip[:] = 1
        self.assertEqual(ref_n(np.array([30004]), us, (lp, like, skip))[0], 3 * 1362 + 9)  # max 4095
        self.assertEqual(got.dtype, np.int16)


class ReadinessWordsTest(unittest.TestCase):
    def test_bits_and_record(self) -> None:
        rt = m4_otf.OTFRuntime(0, list(range(300)), [4] * 300)
        for j in (10, 11, 73, 137, 200):
            self.assertTrue(rt.hold(j, _memfd(np.full(4, 9))))
        w0, w1 = rt.bits(10)  # windows 10 (bit 0), 11 (1), 73 (63: the int64 sign bit), 137 (127)
        self.assertLess(w0, 0)
        self.assertLess(w1, 0)
        rt.book.record(10, w0, w1)
        for j in range(5, 300):
            want = j in (10, 11, 73, 137)  # 200 is beyond the 128-window span
            self.assertEqual(rt.book.decided_ready(j, 10, 1.0), want, j)
        self.assertEqual(rt.dup_fd(73, ts=74), None)  # run-local window 73 is ts 73
        fd = rt.dup_fd(73, ts=73)
        self.assertIsNotNone(fd)
        os.close(fd)
        rt.close()
        self.assertIsNone(rt.dup_fd(73))

    def test_decided_ready_waits_times_out_and_closes(self) -> None:
        book = m4_otf.Readiness()
        out = {}

        def waiter():
            out["v"] = book.decided_ready(5, 3, 30.0)

        th = threading.Thread(target=waiter)
        th.start()
        time.sleep(0.1)
        self.assertTrue(th.is_alive())  # waits for the boundary's record
        book.record(3, 1 << 2, 0)
        th.join(10)
        self.assertTrue(out["v"])
        res = {}

        def timeout():
            try:
                book.decided_ready(9, 7, 0.2)
            except RuntimeError as e:
                res["e"] = str(e)

        th = threading.Thread(target=timeout)
        th.start()
        th.join(10)
        self.assertIn("not recorded", res["e"])  # never a silent local fallback
        with self.assertRaises(RuntimeError):  # the main thread records: it must not wait
            book.decided_ready(9, 7, 0.2)

        def closed():
            out["c"] = book.decided_ready(9, 8, 30.0)

        th = threading.Thread(target=closed)
        th.start()
        time.sleep(0.1)
        book.close()
        th.join(10)
        self.assertFalse(out["c"])


class PublishTest(unittest.TestCase):
    """The builder end to end with in-process clients (unique socket name per test)."""

    def test_publish_seal_backlog_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env(DLRM_HSTU_FP8_M4_OTF_GO_BATCHES="32"):
            data = make_data(d, seed=1)
            name = _name()
            nw = len(data["off"]) - 1
            rts = []
            try:
                logs = []
                with self.assertLogs(_LOG, level="WARNING") as cm:
                    rts.append(_start(0, data, d, name, lws=3))
                    rts.append(_start(1, data, d, name, lws=3))
                logs += cm.output
                self.assertIsNotNone(rts[0].builder)
                self.assertIsNone(rts[1].builder)
                self.assertTrue(all(rt.connected() for rt in rts))
                self.assertTrue(any("m4otf builder elected rank=0 (local clients expected 3)" in x for x in logs), logs)
                self.assertTrue(any("armed OK: m4otf release after run_start" in x for x in logs), logs)
                time.sleep(0.3)
                for rt in rts:
                    rt.drain(1)
                    self.assertEqual(rt.received, 0)  # nothing built before 'go'
                self.assertEqual(rts[0].go_batches, 32)  # the env override above (the default is 0)
                rts[0].progress(1, 31)  # below GO_BATCHES (32): progress only
                time.sleep(0.3)
                rts[1].drain(1)
                self.assertEqual(rts[1].received, 0)
                rts[0].progress(1, 32)  # go
                want = [j for j in range(2, nw) if int(data["off"][j + 1] - data["off"][j]) >= 0]

                def all_in(rt):
                    rt.drain(1)
                    return all(rt.holds(j) for j in want)

                self.assertTrue(_wait(lambda: all(all_in(rt) for rt in rts)))
                late = _start(2, data, d, name, lws=3)  # connects after the windows were published
                rts.append(late)
                self.assertTrue(_wait(lambda: all_in(late)))
                self.assertFalse(late.holds(0) or late.holds(1))  # obsolete at progress 1: never built
                for rt in rts:  # recvmsg(MSG_CMSG_CLOEXEC): no received window fd survives an exec
                    for fd in list(rt._fds.values()):
                        self.assertTrue(m4_otf.fcntl.fcntl(fd, m4_otf.fcntl.F_GETFD) & m4_otf.fcntl.FD_CLOEXEC, fd)
                lut = m4p.cost_lut()
                W, bs = 4, 16
                wq = m4p.weights_from_factors(np.linspace(0.95, 1.05, W), W)
                for rt in rts:
                    for j in want:
                        fd = rt.dup_fd(j, j)
                        try:
                            st = os.fstat(fd)
                            seals = m4_otf.fcntl.fcntl(fd, m4_otf.F_GET_SEALS)
                            self.assertEqual(seals & (m4_otf.REQUIRED_SEALS | m4_otf.F_SEAL_SEAL),
                                             m4_otf.REQUIRED_SEALS | m4_otf.F_SEAL_SEAL)
                            lo, hi = int(data["off"][j]), int(data["off"][j + 1])
                            self.assertEqual(st.st_size, 2 * (hi - lo))
                            self.assertTrue(os.readlink(f"/proc/self/fd/{fd}").startswith("/memfd:dlrmv4-m4otf-w%d" % j))
                            win = m4p.window_array(fd)
                            self.assertTrue(np.array_equal(win, data["table"][lo:hi]), j)
                            nb = (hi - lo) // W // bs
                            if nb and rt.rank == 2:
                                # owner arrays: the OTF window form == the withdrawn table form, same N and factors
                                for r in range(W):
                                    a = m4p.compute_m4(win, lut, j, j, hi - lo, nb, W, bs, wq, r)
                                    b = _ref_table_m4(data["table"], data["off"], lut, j, j, hi - lo, nb, W, bs, wq, r)
                                    self.assertTrue(np.array_equal(a[0], b[0]))
                                    self.assertTrue(np.array_equal(a[1], b[1]))
                                    self.assertEqual(a[2], b[2])
                            del win
                        finally:
                            os.close(fd)
                w0, w1 = late.bits(1)
                self.assertEqual(w0, sum(1 << (j - 1) for j in want))
                self.assertEqual(w1, 0)
            finally:
                for rt in rts:
                    rt.close()
            self.assertEqual(rts[0].builder._p.returncode, 0)  # exited on the control channel's EOF/None

    def test_received_fd_is_immutable(self) -> None:
        import mmap

        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env():
            data = make_data(d, seed=2)
            name = _name()
            rt = _start(0, data, d, name, lws=1)
            try:
                rt.progress(1, 10 ** 6)
                self.assertTrue(_wait(lambda: (rt.drain(1), rt.holds(4))[1]))
                fd = rt.dup_fd(4)
                try:
                    before = bytes(m4p.window_array(fd))
                    for bad in (
                        lambda: os.write(fd, b"\0\0"),
                        lambda: os.pwrite(fd, b"\1\1", 0),
                        lambda: os.ftruncate(fd, 0),
                        lambda: os.ftruncate(fd, 1 << 20),
                        lambda: mmap.mmap(fd, 2, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE),
                        lambda: m4_otf.fcntl.fcntl(fd, m4_otf.F_ADD_SEALS, 0),
                    ):
                        with self.assertRaises(OSError) as cm:
                            bad()
                        self.assertIn(cm.exception.errno, (errno.EPERM, errno.EACCES, errno.EBUSY))
                    try:
                        wfd = os.open(f"/proc/self/fd/{fd}", os.O_WRONLY)
                    except OSError:
                        wfd = -1  # reopening for write refused outright: also immutable
                    if wfd >= 0:
                        try:
                            with self.assertRaises(OSError):
                                os.write(wfd, b"\0\0")
                        finally:
                            os.close(wfd)
                    self.assertEqual(bytes(m4p.window_array(fd)), before)
                finally:
                    os.close(fd)
            finally:
                rt.close()


def _ref_table_m4(n_table, off, lut, j, ts, n_anchors, nb, W, bs, wq, rank, oq=None, hosts=None):
    """The withdrawn 34d35d6 table form of compute_m4 (reference)."""
    G = W * bs
    lo = int(off[ts])
    blocks = np.asarray(n_table[lo:lo + nb * G]).reshape(nb, G)
    owner, load = m4p.partition_owner_load_np(blocks, lut, wq, bs, oq, hosts)
    pos = np.flatnonzero((owner == rank).reshape(-1)).astype(np.int64)
    h = m4p._h63(b"m4", np.int64([j, ts, n_anchors, nb]).tobytes(), wq.tobytes(),
                 (b"" if oq is None else np.asarray(oq, np.int64).tobytes()),
                 (b"" if hosts is None else b"hl" + np.asarray(hosts, np.int64).tobytes()), owner.tobytes())
    return pos, load[:, rank] * m4p.COST_QUANTUM_MS, h


class _Recorder:
    """Records every path the trainer process passes to open / io.open / os.open / np.load / mmap."""

    def __init__(self) -> None:
        self.paths: List[str] = []

    def __enter__(self):
        import builtins
        import io
        import mmap as _mmap

        self._saved = (builtins.open, io.open, os.open, np.load, _mmap.mmap)
        rec = self.paths
        o_open, o_ioopen, o_osopen, o_npload, o_mmap = self._saved

        def _p(x):
            try:
                return os.fspath(x)
            except TypeError:
                return repr(x)

        def b_open(f, *a, **k):
            rec.append(_p(f))
            return o_open(f, *a, **k)

        def i_open(f, *a, **k):
            rec.append(_p(f))
            return o_ioopen(f, *a, **k)

        def os_open(f, *a, **k):
            rec.append(_p(f))
            return o_osopen(f, *a, **k)

        def np_load(f, *a, **k):
            rec.append(_p(f))
            return o_npload(f, *a, **k)

        class MM(o_mmap):
            def __new__(cls, fileno, *a, **k):
                try:
                    rec.append(os.readlink(f"/proc/self/fd/{int(fileno)}"))
                except OSError:
                    pass
                return o_mmap.__new__(cls, fileno, *a, **k)

        builtins.open, io.open, os.open, np.load, _mmap.mmap = b_open, i_open, os_open, np_load, MM
        return self

    def __exit__(self, *exc):
        import builtins
        import io
        import mmap as _mmap

        builtins.open, io.open, os.open, np.load, _mmap.mmap = self._saved
        return False


class _SpeedModule:
    """Installs FakeSpeed as generative_recommenders.dlrm_v4.train.m4_speed."""

    def __init__(self, speed) -> None:
        self.speed = speed

    def __enter__(self):
        mod = types.ModuleType("generative_recommenders.dlrm_v4.train.m4_speed")
        for nm in ("factors_for", "note_window_costs", "register_boundary_callback", "enabled", "set_enabled",
                   "shutdown"):
            setattr(mod, nm, getattr(self.speed, nm))
        self._saved = sys.modules.get(mod.__name__)
        sys.modules[mod.__name__] = mod
        import generative_recommenders.dlrm_v4.train as pkg

        self._had = getattr(pkg, "m4_speed", None)
        pkg.m4_speed = mod
        m4p._ACTIVE.clear()
        m4p._CREATED.clear()
        return self

    def __exit__(self, *exc):
        for dd in list(m4p._CREATED) + list(m4p._ACTIVE):
            dd.close()
        m4p._ACTIVE.clear()
        m4p._CREATED.clear()
        import generative_recommenders.dlrm_v4.train as pkg

        if self._saved is None:
            sys.modules.pop("generative_recommenders.dlrm_v4.train.m4_speed", None)
        else:
            sys.modules["generative_recommenders.dlrm_v4.train.m4_speed"] = self._saved
        if self._had is None:
            if hasattr(pkg, "m4_speed"):
                delattr(pkg, "m4_speed")
        else:
            pkg.m4_speed = self._had
        return False


class GateTest(unittest.TestCase):
    def test_activate_before_run_start_reads_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env(run_start=False, DLRM_HSTU_FP8_M4="1"):
            make_data(d, seed=3)
            sp = FakeSpeed()
            ds = FakeDS(d)
            with _SpeedModule(sp):
                with _Recorder() as rec, self.assertLogs(_LOG_P, level="WARNING") as cm:
                    dealer = m4p.maybe_create(ds, 0, 8, single_pool=True)
                    self.assertIsNotNone(dealer)
                    helper_paths = _proc_paths(dealer._be.pid)
                    dealer._otf_name = _name()
                    ok = dealer.activate(48, list(range(10)))
                self.assertFalse(ok)
                touched = [p for p in rec.paths + helper_paths + _proc_paths(os.getpid()) if d in p]
                self.assertEqual(touched, [])  # no path of the dataset, in any form, before RUN_START
                self.assertTrue(any("m4otf gate closed" in x for x in cm.output), cm.output)
                self.assertIn("m4otf gate closed: no run_start in this process", sp.reason)
                self.assertFalse(sp.enabled_)
                self.assertIsNone(ds._ts_indices)
                self.assertIsNone(dealer._otf)
                self.assertEqual(sp.callbacks, [])
                self.assertIsNotNone(dealer._be._p.returncode)
                # positive control: with RUN_START the same calls read the index (the recorder sees it)
                from generative_recommenders.dlrm_v4.train import dl_prestart

                dl_prestart.set_run_start()
                sp.enabled_ = True  # a fresh process: the estimator is on again
                d2 = m4p.maybe_create(ds, 0, 8, single_pool=True)
                d2._otf_name = _name()
                with _Recorder() as rec2:
                    self.assertTrue(d2.activate(48, list(range(10))))
                self.assertTrue(any(d in p for p in rec2.paths))
                self.assertIsNotNone(d2._otf.builder)
                d2.close()

    def test_release_rechecks_the_gate(self) -> None:
        with _Env(run_start=False):
            with self.assertRaises(RuntimeError):
                m4_otf.require_run_start("x")
            lsock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            lsock.bind(_name())
            lsock.listen(4)
            bh = m4_otf._BuilderHandle(lsock, 0)
            try:
                with self.assertRaisesRegex(RuntimeError, "run-start gate closed"):
                    bh.release({"positions": "x"}, [0], {}, 1)
            finally:
                bh.close()
            self.assertIsNotNone(bh._p.returncode)

    def test_builder_holds_no_data_path_before_release(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env():
            data = make_data(d, seed=4)
            lsock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            name = _name()
            lsock.bind(name)
            lsock.listen(4)
            bh = m4_otf._BuilderHandle(lsock, 0)
            try:
                # the builder is up (numpy loaded) and idle: poll it like the G3 audit does
                self.assertTrue(_wait(lambda: any("numpy" in p for p in _proc_paths(bh.pid))))
                seen = set()
                t_end = time.monotonic() + 1.0
                while time.monotonic() < t_end:
                    seen.update(_proc_paths(bh.pid))
                    time.sleep(0.02)
                self.assertEqual([p for p in seen if d in p], [])
                self.assertEqual([p for p in seen if p.endswith(".npy")], [])
                ts_list = list(range(len(data["off"]) - 1))
                self.assertTrue(bh.release(m4_otf.data_paths(FakeDS(d)), ts_list, {"nfunc": _THIS}, 1))
                # after the release but before GO: still no data path (the files are mapped at the first build)
                t_end = time.monotonic() + 0.6
                while time.monotonic() < t_end:
                    seen.update(_proc_paths(bh.pid))
                    time.sleep(0.02)
                self.assertEqual([p for p in seen if d in p], [])
                bh.progress(1, 10 ** 6, 32)
                # positive control: after GO the data files are mapped
                self.assertTrue(_wait(lambda: any(d in p for p in _proc_paths(bh.pid))))
            finally:
                bh.close()


class ElectionTest(unittest.TestCase):
    """Ranks = multiprocessing spawn children of one parent (like mp.start_processes)."""

    def _run(self, n, mode="ok", timeout_s=180):
        with tempfile.TemporaryDirectory(prefix="m4otf_el_") as d:
            data = make_data(d, seed=5)
            out = os.path.join(d, "result.json")
            env = dict(os.environ)
            env.pop("M4OTF_TEST_FAIL_AT", None)
            if mode == "fail":
                env["M4OTF_TEST_FAIL_AT"] = "2"
            code = ("import sys; sys.path.insert(0, %r); from %s import _election_driver; "
                    "_election_driver(%r, %d, %r)" % (_ROOT, _THIS, d, n, out))
            p = subprocess.run([sys.executable, "-B", "-c", code], cwd=d, env=env, capture_output=True,
                               text=True, timeout=timeout_s)
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            with open(out) as f:
                res = json.load(f)
            return data, res, p.stderr

    def test_one_builder_per_parent(self) -> None:
        data, res, err = self._run(4)
        started = res["start"]
        self.assertEqual(len(started), 4)
        self.assertEqual(sum(1 for s in started if s["elected"]), 1, started)
        self.assertTrue(all(s["connected"] for s in started), started)
        self.assertEqual(len({s["name"] for s in started}), 1)  # one name per parent
        nw = len(data["off"]) - 1
        want = list(range(2, nw))  # progress 1: windows 0, 1 are obsolete
        sums = {str(j): int(data["table"][data["off"][j]:data["off"][j + 1]].astype(np.int64).sum()) for j in want}
        for g in res["got"]:
            self.assertEqual(g["held"], want, g)
            self.assertEqual(g["sums"], sums, g)
        built = re.findall(r"\[m4otf\] node=\S+ j=(\d+) ts=(\d+) anchors=(\d+) build_ms=\S+ progress=1 need=\d+ "
                           r"pick=(\w+) t=(\d+)", err)
        self.assertEqual(sorted(int(x[0]) for x in built), want, err[-2000:])
        self.assertEqual([int(x[0]) for x in built], want)  # no schedule: increasing order
        self.assertNotIn("m4otf builder failed", err)
        self.assertNotIn("refused a client", err)
        self.assertEqual(res["builder_rc"], [0])  # one builder, exited on EOF, reaped
        # audit: every builder line carries t=<epoch ms>; run_start <= started <= release <= go <= data mapped
        # <= first memfd_create <= every window's memfd_create
        lines = [ln for ln in err.splitlines() if ln.startswith("[m4otf]")]
        self.assertTrue(lines and all(re.search(r" t=\d{13}\b", ln) for ln in lines), lines)

        def t_of(tag):
            m = [int(re.search(r" t=(\d+)", ln).group(1)) for ln in lines if tag in ln]
            self.assertEqual(len(m), 1, (tag, lines))
            return m[0]

        rs = max(s["rs_ms"] for s in started if s["elected"])
        seq = [rs] + [t_of(x) for x in ("builder started", "release received", " go progress=", "data mapped",
                                         "first memfd_create")]
        self.assertEqual(seq, sorted(seq), seq)
        self.assertGreaterEqual(min(int(x[4]) for x in built), seq[-1])
        self.assertIn("builder exit built=%d refused=0" % len(want), err)
        el = [s for s in started if s["elected"]][0]
        self.assertEqual(el["comm"], "dlrmv4-m4otf")  # ps / /proc/<pid>/comm name of the builder
        self.assertIn("dlrmv4-m4otf-builder", el["cmdline"])  # and its argv marker

    def test_failing_n_function_is_one_line_and_round_robin(self) -> None:
        data, res, err = self._run(2, mode="fail")
        self.assertEqual(len(re.findall(r"m4otf builder failed: RuntimeError: injected failure", err)), 1, err[-2000:])
        for g in res["got"]:
            self.assertEqual(g["held"], [2, 3])  # the two windows built before the failure stay valid
            self.assertTrue(g["eof"])


def _election_child(rank, n, d, q, go, done):  # pragma: no cover - runs in a spawned process
    import logging

    logging.basicConfig(level=logging.ERROR)
    from generative_recommenders.dlrm_v4.train import dl_prestart

    m4_otf.NFUNC_MODULE = _THIS
    dl_prestart.set_run_start()
    off = np.load(os.path.join(d, _FILES[6]))
    ts_list = list(range(len(off) - 1))
    sizes = [int(off[t + 1] - off[t]) for t in ts_list]
    rt = m4_otf.OTFRuntime.start(rank, ts_list, sizes, m4_otf.data_paths(FakeDS(d)), n, time.time())
    comm = cmdline = ""
    if rt.builder is not None:
        t_end = time.time() + 20
        while time.time() < t_end and comm != "dlrmv4-m4otf":  # PR_SET_NAME runs at the builder's start
            try:
                with open(f"/proc/{rt.builder.pid}/comm") as f:
                    comm = f.read().strip()
                with open(f"/proc/{rt.builder.pid}/cmdline", "rb") as f:
                    cmdline = f.read().replace(b"\0", b" ").decode()
            except OSError:
                pass
            time.sleep(0.01)
    q.put(("start", {"rank": rank, "elected": rt.builder is not None, "connected": rt.connected(),
                     "name": rt.name[1:], "rs_ms": int(dl_prestart.run_start_wall() * 1000), "comm": comm,
                     "cmdline": cmdline}))
    go.wait(60)
    if rt.builder is not None:
        rt.progress(1, 10 ** 6)
    want = set(range(2, len(ts_list)))
    t_end = time.time() + 60
    while time.time() < t_end:
        rt.drain(1)
        if want <= set(rt._fds) or rt._eof:
            break
        time.sleep(0.02)
    time.sleep(0.2)
    rt.drain(1)
    held = sorted(rt._fds)
    sums = {}
    for j in held:
        fd = rt.dup_fd(j)
        sums[str(j)] = int(m4p.window_array(fd).astype(np.int64).sum())
        os.close(fd)
    q.put(("got", {"rank": rank, "held": held, "sums": sums, "eof": rt._eof}))
    done.wait(60)
    rt.close()
    q.put(("closed", {"rank": rank, "rc": None if rt.builder is None else rt.builder._p.returncode}))


def _election_driver(d: str, n: int, out: str) -> None:  # pragma: no cover - runs in a subprocess
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    go, done = ctx.Event(), ctx.Event()
    ps = [ctx.Process(target=_election_child, args=(r, n, d, q, go, done)) for r in range(n)]
    for p in ps:
        p.start()
    res = {"start": [], "got": [], "closed": []}
    for _ in range(n):
        k, v = q.get(timeout=120)
        res[k].append(v)
    go.set()
    for _ in range(n):
        k, v = q.get(timeout=120)
        res[k].append(v)
    done.set()
    for _ in range(n):
        k, v = q.get(timeout=120)
        res[k].append(v)
    for p in ps:
        p.join(60)
    res["builder_rc"] = [c["rc"] for c in res["closed"] if c["rc"] is not None]
    res["got"].sort(key=lambda g: g["rank"])
    with open(out, "w") as f:
        json.dump(res, f)


class _BarrierGather:
    """all_gather across simulated ranks (threads)."""

    def __init__(self, W: int) -> None:
        self.rows: List[Optional[List[int]]] = [None] * W
        self.b = threading.Barrier(W)

    def fn(self, r: int):
        def gather(vals):
            self.rows[r] = list(vals)
            self.b.wait(60)
            out = [list(x) for x in self.rows]
            self.b.wait(60)
            return out

        return gather


class ReadinessAndTest(unittest.TestCase):
    def test_and_gives_identical_decisions_with_drain_delays(self) -> None:
        W, bs = 4, 8
        sizes = [300, 64, 900, 700, 0, 1200, 500, 800, 640, 960, 400, 1000, 720, 880]
        off, tsi, n = _windows(sizes, seed=41)
        nw = len(sizes)
        rng = np.random.default_rng(42)
        # t[r][j]: the boundary whose drain first gives rank r window j's N (None: never)
        t = [[None if rng.random() < 0.08 else int(rng.integers(1, j + 3)) for j in range(nw)] for _ in range(W)]
        factors = {k: rng.uniform(0.9, 1.1, size=W) for k in range(1, nw + 2)}
        gather = _BarrierGather(W)
        lut = m4p.cost_lut()
        dealers, speeds = [], []
        for r in range(W):
            sp = FakeSpeed(factors)
            dd = m4p.M4Dealer(r, W, m4p.LocalBackend(lut, W, r), speed=sp)
            dd.bind_index(off, tsi)
            dd.set_depth(8)
            dd._margin = 0
            rt = m4_otf.OTFRuntime(r, list(range(nw)), sizes)
            rr = np.random.default_rng(100 + r)

            def drain(i, rt=rt, r=r, rr=rr):
                rt.release_below(i)
                time.sleep(float(rr.uniform(0.0, 0.004)))  # per-rank drain delay
                for j in range(i, nw):
                    if t[r][j] is not None and t[r][j] <= i and not rt.holds(j):
                        rt.hold(j, _memfd(n[off[j]:off[j + 1]]))

            rt.drain = drain
            dd._otf = rt
            dd._gather = gather.fn(r)
            dealers.append(dd)
            speeds.append(sp)
        res = [[None] * nw for _ in range(W)]
        pushed = [[threading.Event() for _ in range(nw)] for _ in range(W)]
        errs = []

        def pusher(r):
            try:
                for j in range(nw):
                    time.sleep(float(np.random.default_rng(j * 7 + r).uniform(0, 0.002)))
                    res[r][j] = dealers[r].deal(j, j, tsi[off[j]:off[j + 1]], 0, None, bs)
                    pushed[r][j].set()
            except BaseException as e:  # pragma: no cover - reported below
                errs.append(e)

        def main(r):
            try:
                for i in range(nw):
                    if not pushed[r][i].wait(60):
                        raise RuntimeError("push timeout")
                    dealers[r].boundary_check(i)
            except BaseException as e:  # pragma: no cover - reported below
                errs.append(e)

        ths = [threading.Thread(target=f, args=(r,)) for r in range(W) for f in (pusher, main)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(120)
        self.assertEqual(errs, [])
        nbs = [res[0][j][1] for j in range(nw)]
        G = W * bs
        n_m4 = 0
        for j in range(nw):
            hashes = {dealers[r].window_hash(j) for r in range(W)}
            self.assertEqual(len(hashes), 1, j)
            k = m4p.lag_k(j, nbs[:j], 8, 0)
            want_m4 = (nbs[j] > 0 and k >= 1
                       and all(t[r][j] is not None and t[r][j] <= k + 1 for r in range(W)))
            modes = {res[r][j][0].tobytes() == _stock(tsi[off[j]:off[j + 1]], r, W, bs).tobytes()
                     for r in range(W)}
            got_m4 = dealers[0]._hash[j] != m4p._h63(b"rr", np.int64([j, j, (sizes[j] // W) * W, 0, nbs[j]]).tobytes())
            self.assertEqual(got_m4, want_m4, (j, k, [t[r][j] for r in range(W)]))
            n_m4 += int(got_m4)
            if not got_m4:
                self.assertEqual(modes, {True})  # round-robin: the stock slice on every rank
            for b in range(nbs[j]):
                block = set(tsi[off[j]:off[j + 1]][b * G:(b + 1) * G].tolist())
                got = [set(res[r][j][0][b * bs:(b + 1) * bs].tolist()) for r in range(W)]
                self.assertEqual(sum(len(x) for x in got), G)
                self.assertEqual(set().union(*got), block)
        self.assertGreater(n_m4, 0)
        self.assertGreater(sum(dd._reasons.get("otf-not-ready", 0) for dd in dealers), 0)
        self.assertEqual({dd._otf.book.ready for dd in dealers}, {n_m4})
        for dd in dealers:
            dd.close()


def _stock(gi, r, W, bs):
    n = (len(gi) // W) * W
    return np.array(gi[:n][r::W], copy=True)


class FallbackTest(unittest.TestCase):
    def _rr_dealer(self, rt, sizes, off, tsi):
        W = 4
        dd = m4p.M4Dealer(1, W, m4p.LocalBackend(m4p.cost_lut(), W, 1),
                          speed=FakeSpeed({k: np.linspace(0.9, 1.1, W) for k in range(1, 40)}))
        dd.bind_index(off, tsi)
        dd.set_depth(4)
        dd._margin = 0
        dd._otf = rt
        return dd

    def _deal_all(self, dd, off, tsi, nw, bs=8):
        """Single process (no group): boundary i records this rank's own readiness; stream order."""
        out = []
        for i in range(nw):
            out.append(dd.deal(i, i, tsi[off[i]:off[i + 1]], 0, None, bs))
            dd.boundary_check(i)
        return out

    def test_no_connection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env():
            data = make_data(d, seed=6)
            name = _name()
            squat = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            squat.bind(name)  # bound, never listening: bind fails (not elected), connect is refused
            try:
                t0 = time.monotonic()
                with self.assertLogs(_LOG, level="WARNING") as cm:
                    rt = _start(1, data, d, name)
                self.assertLess(time.monotonic() - t0, 2.0 + 1.5)  # <= 2 s of retries
                self.assertFalse(rt.connected())
                self.assertIsNone(rt.builder)
                self.assertTrue(any("no connection to the node builder" in x for x in cm.output), cm.output)
                rt.drain(1)
                self.assertEqual(rt.bits(1), (0, 0))
                off, tsi = data["off"], np.arange(int(data["off"][-1]))
                nw = len(off) - 1
                sizes = [int(off[t + 1] - off[t]) for t in range(nw)]
                dd = self._rr_dealer(rt, sizes, off, tsi)
                out = self._deal_all(dd, off, tsi, nw)
                self.assertEqual(dd.counts["m4"], 0)
                self.assertGreater(dd._reasons.get("otf-not-ready", 0), 0)
                for j, (per, nb) in enumerate(out):
                    self.assertTrue(np.array_equal(per, _stock(tsi[off[j]:off[j + 1]], 1, 4, 8)))
                dd.close()
            finally:
                squat.close()

    def test_dead_builder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env(M4OTF_TEST_SLEEP_MS="150"):
            data = make_data(d, seed=7)
            name = _name()
            rt = _start(0, data, d, name, lws=1)
            try:
                rt.progress(1, 10 ** 6)
                self.assertTrue(_wait(lambda: (rt.drain(1), rt.holds(2))[1]))
                os.kill(rt.builder.pid, 9)  # SIGKILL mid-run
                rt.builder._p.wait(10)
                rt.drain(1)
                held = sorted(rt._fds)
                self.assertTrue(_wait(lambda: (rt.drain(1), rt._eof)[1]))
                self.assertLess(len(held), len(data["off"]) - 3)  # killed before the last windows
                with self.assertLogs(_LOG, level="WARNING") as cm:
                    rt.progress(2, 10 ** 6)
                self.assertTrue(any("m4otf builder failed: builder pid" in x for x in cm.output), cm.output)
                fd = rt.dup_fd(held[0])
                self.assertEqual(os.fstat(fd).st_size, 2 * int(data["off"][held[0] + 1] - data["off"][held[0]]))
                os.close(fd)  # held windows stay valid after the builder died
                off = data["off"]
                tsi = np.arange(int(off[-1]))
                nw = len(off) - 1
                dd = self._rr_dealer(rt, [int(off[t + 1] - off[t]) for t in range(nw)], off, tsi)
                t0 = time.monotonic()
                self._deal_all(dd, off, tsi, nw)
                self.assertLess(time.monotonic() - t0, 30.0)  # never a stall
                missing = [j for j in range(nw) if j not in held]
                self.assertGreater(len(missing), 0)
                dd.close()
            finally:
                rt.close()

    def test_missing_n_function_module(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env(nfunc=_THIS + "_no_such_module"):
            data = make_data(d, seed=8)
            rt = _start(0, data, d, _name(), lws=1)
            try:
                time.sleep(0.5)
                self.assertIsNone(rt.builder._p.poll())  # released, no GO yet: the N module is not imported
                rt.progress(1, 10 ** 6)
                self.assertEqual(rt.builder._p.wait(30), 0)  # logged once and exited
                self.assertTrue(_wait(lambda: (rt.drain(1), rt._eof)[1]))
                self.assertEqual(rt.received, 0)
                self.assertEqual(rt.bits(1), (0, 0))
            finally:
                rt.close()


class DeferredCostsTest(unittest.TestCase):
    def test_now_deferred_and_dropped(self) -> None:
        W, bs = 4, 8
        sizes = [320, 640, 320, 960, 640]
        off, tsi, n = _windows(sizes, seed=51)
        lut = m4p.cost_lut()
        sp = FakeSpeed()  # no factors: every window round-robin
        dd = m4p.M4Dealer(2, W, m4p.LocalBackend(lut, W, 2), speed=sp, lut=lut)
        dd.bind_index(off, tsi)
        dd.set_depth(4)
        rt = m4_otf.OTFRuntime(2, list(range(5)), sizes)
        dd._otf = rt

        def want(j):
            G = W * bs
            nb = sizes[j] // G
            return lut[n[off[j]:off[j] + nb * G].reshape(nb, G)[:, 2::W].astype(np.int64)].sum(1) * m4p.COST_QUANTUM_MS

        rt.hold(0, _memfd(n[off[0]:off[1]]))
        for j in range(4):
            dd.deal(j, j, tsi[off[j]:off[j + 1]], 0, None, bs)
        self.assertTrue(np.allclose(sp.costs[0], want(0)))  # N held at push: noted at once
        for j in (1, 2, 3):
            self.assertEqual(sp.costs[j].shape[0], 0)  # empty note, pending
        rt.hold(2, _memfd(n[off[2]:off[3]]))
        dd.boundary_check(1)  # window 1 never arrived: dropped; window 2 arrived: noted
        self.assertTrue(np.allclose(sp.costs[2], want(2)))
        self.assertEqual(sp.costs[1].shape[0], 0)
        self.assertEqual((dd.deferred_costs, dd.dropped_costs), (1, 1))
        self.assertEqual(sorted(dd._pending), [3])
        rt.hold(3, _memfd(n[off[3]:off[4]]))
        dd.boundary_check(2)
        self.assertTrue(np.allclose(sp.costs[3], want(3)))
        self.assertEqual((dd.deferred_costs, dd.dropped_costs, dd._pending), (2, 1, {}))
        dd.close()


class TeardownTest(unittest.TestCase):
    def test_close_reaps_and_closes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env(DLRM_HSTU_FP8_M4="1"):
            data = make_data(d, seed=9)
            sp = FakeSpeed()
            with _SpeedModule(sp):
                dd = m4p.maybe_create(FakeDS(d), 0, 2, single_pool=True)
                dd._otf_name = _name()
                self.assertTrue(dd.activate(32, list(range(len(data["off"]) - 1))))
                rt = dd._otf
                rt.progress(1, 10 ** 6)
                self.assertTrue(_wait(lambda: (rt.drain(1), rt.holds(5))[1]))
                fds = list(rt._fds.values())
                sock = rt._sock
                pids = [rt.builder.pid, dd._be.pid]
                with self.assertLogs(_LOG_P, level="WARNING") as cm:
                    dd.stream_closed()
                    dd.summary()
                    dd.close()
                    dd.close()  # idempotent
                self.assertTrue(any("DLRMV4 m4otf summary rank=0: windows built=" in x and "deferred_costs=" in x
                                    for x in cm.output), cm.output)
                self.assertEqual(sock.fileno(), -1)
                for fd in fds:
                    try:
                        tgt = os.readlink(f"/proc/self/fd/{fd}")
                    except OSError:
                        tgt = ""  # closed
                    self.assertFalse(tgt.startswith("/memfd:dlrmv4-m4otf"), (fd, tgt))
                self.assertFalse(any(os.readlink(f"/proc/self/fd/{x}").startswith("/memfd:dlrmv4-m4otf")
                                     for x in os.listdir("/proc/self/fd") if os.path.exists(f"/proc/self/fd/{x}")))
                for pid in pids:
                    self.assertFalse(os.path.exists(f"/proc/{pid}"), pid)  # reaped: no zombie
                    with self.assertRaises(ChildProcessError):
                        os.waitpid(pid, os.WNOHANG)
                self.assertTrue(rt.book._closed)


class NoFilesTest(unittest.TestCase):
    def test_nothing_written(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_data_") as d, \
                tempfile.TemporaryDirectory(prefix="m4otf_cwd_") as cwd, \
                tempfile.TemporaryDirectory(prefix="m4otf_tmp_") as tmp:
            make_data(d, seed=10)

            def snap(root):
                out = {}
                for dirpath, _, files in os.walk(root):
                    for f in files:
                        p = os.path.join(dirpath, f)
                        st = os.stat(p)
                        out[p] = (st.st_size, st.st_mtime_ns)
                return out

            before = (snap(d), snap(_TRAIN), sorted(os.listdir("/dev/shm")))
            env = dict(os.environ, TMPDIR=tmp, HOME=tmp)
            env.pop("M4OTF_TEST_FAIL_AT", None)
            code = ("import sys; sys.path.insert(0, %r); from %s import _nofiles_driver; _nofiles_driver(%r)"
                    % (_ROOT, _THIS, d))
            p = subprocess.run([sys.executable, "-B", "-c", code], cwd=cwd, env=env, capture_output=True, text=True,
                               timeout=180)
            self.assertEqual(p.returncode, 0, p.stderr[-3000:])
            res = json.loads(p.stdout.strip().splitlines()[-1])
            self.assertGreater(res["windows"], 0)
            self.assertEqual(res["writable_files"], [])  # the builder never held a writable file
            self.assertTrue(res["data_mapped"])  # positive control: it did map the data (read-only)
            self.assertEqual(os.listdir(cwd), [])
            self.assertEqual(os.listdir(tmp), [])
            after = (snap(d), snap(_TRAIN), sorted(os.listdir("/dev/shm")))
            self.assertEqual(after[0], before[0])
            self.assertEqual(after[1], before[1])
            self.assertEqual([x for x in after[2] if x not in before[2] and ("m4otf" in x or "dlrmv4" in x)], [])


def _nofiles_driver(d: str) -> None:  # pragma: no cover - runs in a subprocess
    import logging

    logging.basicConfig(level=logging.ERROR)
    from generative_recommenders.dlrm_v4.train import dl_prestart

    m4_otf.NFUNC_MODULE = _THIS
    dl_prestart.set_run_start()
    data_off = np.load(os.path.join(d, _FILES[6]))
    ts_list = list(range(len(data_off) - 1))
    sizes = [int(data_off[t + 1] - data_off[t]) for t in ts_list]
    name = _name()
    paths = m4_otf.data_paths(FakeDS(d))
    rts = [m4_otf.OTFRuntime.start(r, ts_list, sizes, paths, 2, time.time(), name=name) for r in range(2)]
    pid = rts[0].builder.pid
    rts[0].progress(1, 10 ** 6)
    writable, mapped = set(), False
    t_end = time.time() + 60
    while time.time() < t_end:
        for rt in rts:
            rt.drain(1)
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
                    with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                        flags = int(next(x for x in f if x.startswith("flags:")).split()[1], 8)
                except (OSError, StopIteration):
                    continue
                if tgt.startswith("/") and not tgt.startswith(("/memfd:", "/dev/")) and flags & 3:
                    writable.add(tgt)
            mapped = mapped or any(d in p for p in _proc_paths(pid))
        except OSError:
            pass
        if all(rt.holds(len(ts_list) - 1) for rt in rts):
            break
        time.sleep(0.01)
    n = rts[1].received
    for rt in rts:
        rt.close()
    print(json.dumps({"windows": n, "writable_files": sorted(writable), "data_mapped": mapped}))


def _run_driver(test, fn_name: str, *args, env_extra=None, timeout_s=180):
    """Run ``fn_name(*args)`` of this module in a fresh interpreter (cwd = a temp dir); (json result, stderr)."""
    with tempfile.TemporaryDirectory(prefix="m4otf_drv_") as cwd:
        env = dict(os.environ)
        env.pop("M4OTF_TEST_FAIL_AT", None)
        env.update(env_extra or {})
        code = "import sys; sys.path.insert(0, %r); from %s import %s; %s(*%r)" % (_ROOT, _THIS, fn_name, fn_name,
                                                                                   tuple(args))
        p = subprocess.run([sys.executable, "-B", "-c", code], cwd=cwd, env=env, capture_output=True, text=True,
                           timeout=timeout_s)
    test.assertEqual(p.returncode, 0, p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1]), p.stderr


class ScheduleTest(unittest.TestCase):
    """The builder's schedule: lag_k_all, the need list and pick_window (pure functions)."""

    def test_lag_k_all_equals_lag_k(self) -> None:
        rng = np.random.default_rng(71)
        for _ in range(150):
            n = int(rng.integers(0, 80))
            nbs = [int(x) for x in rng.integers(0, int(rng.choice([4, 40, 400])), size=n)]
            for i in rng.integers(0, max(n, 1), size=n // 4):
                if n:
                    nbs[int(i)] = 0
            depth, margin = int(rng.choice([1, 8, 48])), int(rng.choice([0, 3, 32]))
            self.assertEqual(m4p.lag_k_all(nbs, depth, margin),
                             [m4p.lag_k(j, nbs, depth, margin) for j in range(n + 1)])

    def test_need_from_pushed_and_predicted_counts(self) -> None:
        W, bs = 4, 8
        sizes = [3200, 640, -1, 2560, 99, 3205, 640, 4000]
        self.assertEqual(m4p.otf_nbs([], sizes, W, bs), [100, 20, 0, 80, 3, 100, 20, 125])
        self.assertEqual(m4p.otf_nbs([7, 7], sizes, W, bs, nb_cap=50), [7, 7, 0, 50, 3, 50, 20, 50])
        full = m4p.otf_nbs([7, 7], sizes, W, bs)
        need = m4p.otf_need(full, 8, 3)
        for j, x in enumerate(need):
            k = m4p.lag_k(j, full, 8, 3)
            self.assertEqual(x, k + 1 if k >= 1 else j)
        self.assertTrue(any(x < j for j, x in enumerate(need)))  # some windows can use M4

    def test_pick_window(self) -> None:
        pw = m4_otf.pick_window
        ident = list(range(40))
        # no schedule: increasing order from progress + 1 within progress + lead; J0; idle when built
        self.assertEqual(pw(1, 0, set(), ident, 16, 0), (2, 2, "asc"))
        self.assertEqual(pw(1, 0, set(), ident, 16, 5), (5, 5, "asc"))
        self.assertEqual(pw(5, 3, set(), ident, 16, 0), (6, 6, "asc"))  # never an obsolete window
        self.assertEqual(pw(1, 18, set(range(2, 18)), ident, 16, 0), (None, 18, "idle"))
        # the first M4 window at 2n: j=35 decided at boundary 14 (22 windows ahead): within the horizon at once
        need = list(range(35)) + [35 - 21] + [j - 3 for j in range(36, 60)]
        built, cur, order = set(), 0, []
        for _ in range(60):
            j, cur, how = pw(1, cur, built, need, 16, 0)
            if j is None:
                break
            self.assertEqual(how, "asc")
            order.append(j)
            built.add(j)
            cur = j + 1
        self.assertEqual(order, list(range(2, 36)))  # 2..35 in order (no window above 35 is due yet)
        # steady state, window j >= 5 decided at boundary j - 3; builder behind (1 s per window, 1 s per
        # boundary): jump to the window decided at the next boundary (14) if it can still make it, else to the
        # first one it can make (15 with half the interval gone); without an estimate no jump
        need = list(range(5)) + [j - 3 for j in range(5, 60)]
        est = lambda j: 1.0  # noqa: E731
        self.assertEqual(pw(10, 11, set(), need, 16, 0, est, 1.0, 0.0), (14, 14, "jump"))
        self.assertEqual(pw(10, 11, set(), need, 16, 0, est, 1.0, 0.5), (15, 15, "jump"))
        self.assertEqual(pw(10, 11, set(), need, 16, 0, lambda j: 3.5, 1.0, 0.0), (17, 17, "jump"))  # 3.5 x slower
        self.assertEqual(pw(10, 11, set(), need, 16, 0, None, 1.0, 0.5), (11, 11, "asc"))
        self.assertEqual(pw(10, 11, set(), need, 16, 0, lambda j: 99.0, 1.0, 0.0), (11, 11, "asc"))  # none reachable
        # fast enough to build 11..14 in time: stay in order
        self.assertEqual(pw(10, 11, set(), need, 16, 0, lambda j: 0.01, 1.0, 0.5), (11, 11, "asc"))
        # after the jump the cursor is ahead: in order again; the windows jumped over are back-filled
        # (increasing order) only once everything up to the horizon is built
        self.assertEqual(pw(10, 15, {14}, need, 16, 0, est, 1.0, 0.0), (15, 15, "asc"))
        hi_built = {14} | set(range(15, 30))
        self.assertEqual(pw(10, 30, hi_built, need, 16, 0, est, 1.0, 0.0), (11, 30, "backfill"))
        # progress 12: 11, 12 obsolete; the horizon grew to 31 (decisions due by boundary 28): forward first
        self.assertEqual(pw(12, 30, hi_built | {11}, need, 16, 0, est, 1.0, 0.0), (30, 30, "asc"))
        self.assertEqual(pw(12, 32, hi_built | {11, 30, 31}, need, 16, 0, est, 1.0, 0.0), (13, 32, "backfill"))

    def test_dealer_sends_need_once_and_on_change(self) -> None:
        W, bs = 4, 8
        sizes = [320, 640, 320, 960, 640, 1280, 64, 2560, 960, 3200, 640, 1920]
        off, tsi, _ = _windows(sizes, seed=61)
        dd = m4p.M4Dealer(0, W, m4p.LocalBackend(m4p.cost_lut(), W, 0), speed=FakeSpeed())
        dd.bind_index(off, tsi)
        dd.set_depth(4)
        dd._margin = 3
        sent = []

        class _Builder:
            dead = ""

        class _Elected:
            elected, builder, stopped = True, _Builder(), ""

            def __init__(self) -> None:
                self.sizes = list(sizes)

            def schedule(self, need) -> bool:
                sent.append(list(need))
                return True

        otf = _Elected()
        dd._send_schedule(otf)
        self.assertEqual(sent, [])  # batch size not known before the first push
        for j in range(3):
            dd.deal(j, j, tsi[off[j]:off[j + 1]], 0, None, bs)
        dd._send_schedule(otf)
        self.assertEqual(len(sent), 1)
        full = m4p.otf_nbs(dd._nbs, sizes, W, bs)
        self.assertEqual(sent[0], m4p.otf_need(full, 4, 3))
        dd.deal(3, 3, tsi[off[3]:off[4]], 0, None, bs)  # as predicted
        dd._send_schedule(otf)
        self.assertEqual(len(sent), 1)
        dd.deal(4, 4, tsi[off[4]:off[5] - 40], 0, None, bs)  # not the window slice: another batch count
        dd._send_schedule(otf)
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[1], m4p.otf_need(m4p.otf_nbs(dd._nbs, sizes, W, bs), 4, 3))
        dd.close()


class StickyElectionTest(unittest.TestCase):
    def test_late_rank_never_elects_a_second_builder(self) -> None:
        for mode in ("killed", "failed_at_go"):
            nfunc = _THIS if mode == "killed" else _THIS + "_no_such_module"
            with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env(nfunc=nfunc):
                data = make_data(d, seed=11)
                name = _name()
                r0 = _start(0, data, d, name, lws=2)
                try:
                    self.assertTrue(r0.elected)
                    if mode == "killed":
                        os.kill(r0.builder.pid, 9)
                    else:
                        r0.progress(1, 10 ** 6)  # GO: the N module import fails, one line, exit
                    r0.builder._p.wait(30)
                    r1 = _start(1, data, d, name, lws=2)  # a late rank, after the builder is gone
                    try:
                        self.assertFalse(r1.elected, mode)
                        self.assertIsNone(r1.builder, mode)  # no second builder
                        r0.progress(2, 10 ** 6)
                        time.sleep(0.3)
                        r1.drain(2)
                        self.assertEqual((r1.received, r1.bits(2)), (0, (0, 0)), mode)  # nothing: round-robin
                    finally:
                        r1.close()
                finally:
                    r0.close()
                s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                s.bind(name)  # the elected rank's close() released the name
                s.close()


_FOREIGN_CLIENT = r"""
import json, socket, sys, time
s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
s.connect(bytes.fromhex(sys.argv[1]).decode("latin1"))
s.settimeout(0.2)
n, eof, t_end = 0, False, time.time() + float(sys.argv[2])
while time.time() < t_end:
    try:
        msg, anc, _, _ = s.recvmsg(256, socket.CMSG_SPACE(4))
    except socket.timeout:
        continue
    except OSError:
        eof = True
        break
    if not msg and not anc:
        eof = True
        break
    n += len(anc)
print(json.dumps({"fds": n, "eof": eof}))
"""

_FOREIGN_LISTENER = r"""
import socket, sys, time
s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
s.bind(bytes.fromhex(sys.argv[1]).decode("latin1"))
s.listen(8)
print("listening", flush=True)
time.sleep(float(sys.argv[2]))
"""


def _peer_driver(d: str) -> None:  # pragma: no cover - runs in a subprocess (this process plays the local rank)
    import logging

    logging.basicConfig(level=logging.ERROR)
    from generative_recommenders.dlrm_v4.train import dl_prestart

    m4_otf.NFUNC_MODULE = _THIS
    dl_prestart.set_run_start()
    off = np.load(os.path.join(d, _FILES[6]))
    ts_list = list(range(len(off) - 1))
    sizes = [int(off[t + 1] - off[t]) for t in ts_list]
    name = _name()
    rt = m4_otf.OTFRuntime.start(0, ts_list, sizes, m4_otf.data_paths(FakeDS(d)), 2, time.time(), name=name)
    # a foreign process: a child of this rank, not of the local ranks' parent
    child = subprocess.Popen([sys.executable, "-B", "-c", _FOREIGN_CLIENT, name.encode("latin1").hex(), "4"],
                             stdout=subprocess.PIPE, text=True)
    time.sleep(0.5)
    s2, why2 = m4_otf._connect(name)  # the second local client (same parent): served
    s3, why3 = m4_otf._connect(name)  # a third local client: over local_world_size, refused by the builder
    rt.progress(1, 10 ** 6)
    want = len(ts_list) - 2
    got2, eof3, t_end = 0, False, time.time() + 30
    while time.time() < t_end and (got2 < want or not eof3):
        rt.drain(1)
        for s, is3 in ((s2, False), (s3, True)):
            try:
                msg, fds, _ = m4_otf._recv_fd(s)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                eof3 = eof3 or is3
                continue
            for fd in fds:
                os.close(fd)
            if is3 and not msg and not fds:
                eof3 = True
            elif not is3:
                got2 += len(fds)
        time.sleep(0.02)
    foreign = json.loads(child.communicate(timeout=60)[0].strip().splitlines()[-1])
    out = {"connected": [rt.connected(), s2 is not None, s3 is not None], "why": [why2, why3], "held": rt.received,
           "got2": got2, "eof3": eof3, "foreign": foreign, "want": want}
    for s in (s2, s3):
        s.close()
    rt.close()
    print(json.dumps(out))


class PeerCredTest(unittest.TestCase):
    def test_builder_refuses_foreign_and_surplus_clients(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d:
            make_data(d, seed=13)
            res, err = _run_driver(self, "_peer_driver", d)
        self.assertEqual(res["connected"], [True, True, True], res)  # connect() succeeds into the backlog
        self.assertEqual(res["held"], res["want"])  # the two local clients get every window
        self.assertEqual(res["got2"], res["want"])
        self.assertTrue(res["eof3"], res)  # the surplus client: closed by the builder, nothing sent
        self.assertEqual(res["foreign"], {"fds": 0, "eof": True})  # the foreign client: closed, nothing sent
        refused = [ln for ln in err.splitlines() if "refused a client" in ln]
        self.assertEqual(len(refused), 2, err[-2000:])
        self.assertTrue(any("is not a local rank" in ln for ln in refused), refused)
        self.assertTrue(any("already serving 2 local clients" in ln for ln in refused), refused)

    def test_rank_refuses_a_foreign_listener(self) -> None:
        name = _name()
        lp = subprocess.Popen([sys.executable, "-B", "-c", _FOREIGN_LISTENER, name.encode("latin1").hex(), "20"],
                              stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(lp.stdout.readline().strip(), "listening")
            s, why = m4_otf._connect(name)
            self.assertIsNone(s)
            self.assertIn("is not a local rank", why)
            with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env():
                data = make_data(d, seed=14)
                with self.assertLogs(_LOG, level="WARNING") as cm:
                    rt = _start(0, data, d, name, lws=2)
                try:
                    self.assertFalse(rt.elected)  # the name is taken
                    self.assertFalse(rt.connected())  # and its listener is not a local rank
                    self.assertTrue(any("no connection to the node builder" in x and "not a local rank" in x
                                        for x in cm.output), cm.output)
                finally:
                    rt.close()
        finally:
            lp.kill()
            lp.wait(10)


class DutyTest(unittest.TestCase):
    def test_duty_rests_between_builds(self) -> None:
        gaps = {}
        for duty in ("1", "0.25"):
            with tempfile.TemporaryDirectory(prefix="m4otf_") as d, \
                    _Env(M4OTF_TEST_SLEEP_MS="80", DLRM_HSTU_FP8_M4_OTF_DUTY=duty):
                data = make_data(d, seed=15)
                rt = _start(0, data, d, _name(), lws=1)
                try:
                    rt.progress(1, 10 ** 6)
                    seen, t_seen = set(), []
                    t_end = time.monotonic() + 30
                    while time.monotonic() < t_end and len(seen) < 6:
                        rt.drain(1)
                        for j in list(rt._fds):
                            if j not in seen:
                                seen.add(j)
                                t_seen.append(time.monotonic())
                        time.sleep(0.005)
                    self.assertGreaterEqual(len(seen), 6)
                    gaps[duty] = float(np.median(np.diff(t_seen)))
                finally:
                    rt.close()
        # build >= 80 ms (sleep); DUTY 0.25 adds a rest of 3 x the build time (>= 240 ms) between builds
        self.assertGreater(gaps["0.25"] - gaps["1"], 0.15, gaps)


def _jump_driver(d: str) -> None:  # pragma: no cover - runs in a subprocess
    """A slow builder (M4OTF_TEST_SLEEP_MS) and window boundaries every 0.1 s: it must jump."""
    import logging

    logging.basicConfig(level=logging.ERROR)
    from generative_recommenders.dlrm_v4.train import dl_prestart

    m4_otf.NFUNC_MODULE = _THIS
    dl_prestart.set_run_start()
    off = np.load(os.path.join(d, _FILES[6]))
    ts_list = list(range(len(off) - 1))
    sizes = [int(off[t + 1] - off[t]) for t in ts_list]
    rt = m4_otf.OTFRuntime.start(0, ts_list, sizes, m4_otf.data_paths(FakeDS(d)), 1, time.time(), name=_name())
    need = [j if j < 6 else j - 3 for j in range(len(ts_list))]  # window j >= 6 decided at boundary j - 3
    rt.schedule(need)
    held_at = {}
    for i in range(1, len(ts_list) - 4):
        rt.drain(i)
        w0, w1 = rt.bits(i)
        bits = (w0 & ((1 << 64) - 1)) | ((w1 & ((1 << 64) - 1)) << 64)
        held_at[i] = [i + b for b in range(128) if (bits >> b) & 1]
        rt.progress(i, 10 ** 6)
        time.sleep(0.1)
    rt.close()
    ready = [j for j in range(6, len(ts_list)) if need[j] in held_at and j in held_at[need[j]]]
    print(json.dumps({"ready": ready}))


class JumpTest(unittest.TestCase):
    def test_slow_builder_jumps_to_decision_deadlines(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d:
            make_data(d, seed=16, n_windows=40)
            res, err = _run_driver(self, "_jump_driver", d, env_extra={"M4OTF_TEST_SLEEP_MS": "250"})
        picks = re.findall(r" j=(\d+) ts=\d+ anchors=\d+ build_ms=\S+ progress=(\d+) need=(\d+) pick=(\w+)", err)
        self.assertTrue(any(p[3] == "jump" for p in picks), err[-3000:])
        for j, q, need, how in picks:  # never an obsolete window, never one whose decision has passed when jumped to
            self.assertGreater(int(j), int(q))
            if how == "jump":
                self.assertGreater(int(need), int(q))
        self.assertGreater(len(res["ready"]), 0, (res, err[-2000:]))  # some decisions are made with N on time


class GlobalOffTest(unittest.TestCase):
    def test_off_stops_builder_without_waiting_and_no_more_costs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env(DLRM_HSTU_FP8_M4="1", M4OTF_TEST_SLEEP_MS="1500"):
            data = make_data(d, seed=12)
            nw = len(data["off"]) - 1
            off, tsi = data["off"], data["tsi"]
            sp = FakeSpeed()
            with _SpeedModule(sp):
                dd = m4p.maybe_create(FakeDS(d), 0, 2, single_pool=True)
                dd._otf_name = _name()
                self.assertTrue(dd.activate(8, list(range(nw))))
                rt = dd._otf
                for j in range(3):
                    dd.deal(j, j, tsi[off[j]:off[j + 1]], 0, None, 8)
                dd.boundary_check(1)  # elected: schedule + progress + GO
                self.assertTrue(_wait(lambda: (rt.drain(1), rt.holds(3))[1]))
                sp.enabled_ = False  # m4_speed off for good (a global OFF)
                t0 = time.monotonic()
                with self.assertLogs(_LOG, level="WARNING") as cm:
                    dd.boundary_check(2)
                self.assertLess(time.monotonic() - t0, 0.75)  # the builder is mid-build (1.5 s): not awaited
                self.assertTrue(any("m4otf builder stopped rank=0" in x for x in cm.output), cm.output)
                self.assertEqual(dd._pending, {})
                self.assertTrue(_wait(lambda: rt.builder._p.poll() is not None, 10))  # exits after that build
                self.assertTrue(rt.holds(3))  # held windows stay valid
                dd.deal(3, 3, tsi[off[3]:off[4]], 0, None, 8)
                self.assertEqual(sp.notes[-1], (3, 0))  # N held, but no round-robin costs any more
                with self.assertLogs(_LOG_P, level="WARNING") as cm:
                    dd.summary()
                self.assertTrue(any("stopped=(" in x for x in cm.output), cm.output)
                dd.close()


class NoMllogTest(unittest.TestCase):
    def test_no_mllog_no_seed(self) -> None:
        bad_name = re.compile(r"mllog|mlperf|log_event|seed", re.I)  # identifiers and imports
        bad_str = re.compile(r"mllog|log_event|seed|\"key\"", re.I)  # string constants (log text may name "MLPerf")
        for fn in ("m4_otf.py", "m4_partition.py"):
            with open(os.path.join(_TRAIN, fn)) as f:
                tree = ast.parse(f.read())
            docs = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    ds = ast.get_docstring(node, clean=False)
                    if ds is not None:
                        docs.add(ds)
            names, strs = [], []
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.append(node.id)
                elif isinstance(node, ast.Attribute):
                    names.append(node.attr)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names += [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value not in docs:
                    strs.append(node.value)
            self.assertEqual([x for x in names if bad_name.search(x)], [], fn)
            self.assertEqual([x for x in strs if bad_str.search(x)], [], fn)


class GinTest(unittest.TestCase):
    _GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")

    def test_binding_is_a_gin_bool_and_no_table(self) -> None:
        with open(self._GIN) as f:
            text = f.read()
        live = [ln.strip() for ln in text.splitlines() if not ln.lstrip().startswith("#") and "DLRM_HSTU_FP8_M4" in ln]
        self.assertIn(live, (
            ["apply_env_bootstrap.DLRM_HSTU_FP8_M4 = False", 'apply_env_bootstrap.DLRM_HSTU_FP8_M4_HOST_LOCAL = "0"'],
            ["apply_env_bootstrap.DLRM_HSTU_FP8_M4 = True", 'apply_env_bootstrap.DLRM_HSTU_FP8_M4_HOST_LOCAL = "0"'],
        ))
        for fn in ("gin/yambda_5b.gin", "_env_bootstrap.py", "m4_partition.py", "m4_otf.py", "m4_speed.py"):
            with open(os.path.join(_TRAIN, fn)) as f:
                self.assertNotIn("NTABLE", f.read(), fn)
        a = text.index("# (82) M4-OTF")
        b = text.index('apply_env_bootstrap.DLRM_HSTU_FP8_M4_HOST_LOCAL = "0"')
        item = text[a:b]
        self.assertEqual(re.findall(r"(?<![\w.<>])/(?:apps|var|home|tmp|data|mnt|scratch|dev|proc|usr|opt|root)\b",
                                    item), [])
        self.assertNotIn(".npy", item)

    def test_bool_binding_reaches_env(self) -> None:
        from unittest import mock

        from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap

        for v, want in ((False, "0"), (True, "1")):
            with mock.patch.dict("os.environ", {}, clear=True):
                apply_env_bootstrap(DLRM_HSTU_FP8_M4=v)
                self.assertEqual(os.environ.get("DLRM_HSTU_FP8_M4"), want)
                self.assertEqual(m4p.switch_on(), v)
        with mock.patch.dict("os.environ", {"DLRM_HSTU_FP8_M4": "0"}, clear=True):
            apply_env_bootstrap(DLRM_HSTU_FP8_M4=True)  # the kill switch wins over the gin-on row
            self.assertFalse(m4p.switch_on())


# -- review fixes, iter59: O1 (fd numbers >= FD_SETSIZE) and C5 (the N rule's dataset configuration) ---------
FD_FLOOR = 1100  # > FD_SETSIZE (1024): select() refuses these fd numbers, poll() takes them


class _FdFloor:
    """Occupy every free fd number below ``floor`` (as a large trainer does: RCCL, HIP, DataLoader pipes ...),
    so the sockets created inside get fd numbers >= floor. Raises RLIMIT_NOFILE's soft limit when it is too
    low (bounded by the hard limit; restored on exit); ``skip`` says why when that is impossible."""

    def __init__(self, floor: int = FD_FLOOR) -> None:
        self.floor, self.pad, self.saved, self.skip = int(floor), [], None, ""

    def __enter__(self):
        import resource

        if self.floor <= 0:
            return self
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        need = self.floor + 256
        if soft != resource.RLIM_INFINITY and soft < need:
            if hard != resource.RLIM_INFINITY and hard < need:
                self.skip = f"RLIMIT_NOFILE hard limit {hard} < {need}: this process cannot hold fd >= {self.floor}"
                return self
            resource.setrlimit(resource.RLIMIT_NOFILE, (need, hard))
            self.saved = (soft, hard)
        while True:  # os.open returns the lowest free fd: stop once every fd below the floor is taken
            fd = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
            self.pad.append(fd)
            if fd >= self.floor - 1:
                return self

    def __exit__(self, *exc):
        for fd in self.pad:
            os.close(fd)
        self.pad = []
        if self.saved is not None:
            import resource

            resource.setrlimit(resource.RLIMIT_NOFILE, self.saved)
        return False


class HighFdTest(unittest.TestCase):
    """Review O1 (iter59): the builder inherits its control and listening sockets with the trainer's fd
    numbers (pass_fds). It now waits on them with poll(); select() raised ValueError for fd >= FD_SETSIZE,
    so the builder failed at its first wait and every window dealt round-robin."""

    def test_wait_matches_select_and_takes_fds_above_fd_setsize(self) -> None:
        for floor in (0, FD_FLOOR):
            with _FdFloor(floor) as fl:
                if fl.skip:
                    self.skipTest(fl.skip)
                self._check_wait(floor)

    def _check_wait(self, floor: int) -> None:
        import select

        high = floor > 0
        name = _name()
        ctrl, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        lsock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        socks = [ctrl, peer, lsock]
        try:
            lsock.bind(name)
            lsock.listen(4)
            b = m4_otf._Builder(ctrl, lsock)
            self.assertGreaterEqual(min(ctrl.fileno(), lsock.fileno()), floor)
            if high:  # the former wait on the same fds: what failed the builder (O1)
                with self.assertRaisesRegex(ValueError, "filedescriptor out of range in select"):
                    select.select([ctrl, lsock], [], [], 0)

            def same(want: bool, timeout: float = 5.0) -> None:
                self.assertEqual(b._wait(timeout), want, (floor, want))
                if not high:  # select() on the same set agrees (low fds only: it cannot take high ones)
                    self.assertEqual(bool(select.select([ctrl, lsock], [], [], 0)[0]), want, (floor, want))

            t0 = time.monotonic()
            same(False, 0.05)  # nothing pending: the timeout
            self.assertGreaterEqual(time.monotonic() - t0, 0.045)
            t0 = time.monotonic()
            self.assertFalse(b._wait(-1.0))  # a negative timeout waits 0 s (poll() alone would block forever)
            self.assertLess(time.monotonic() - t0, 0.5)
            peer.send(b"x")  # a control message
            same(True)
            self.assertEqual(ctrl.recv(16), b"x")
            same(False, 0.0)
            c = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)  # a pending connection
            socks.append(c)
            c.connect(name)
            same(True)
            a, _ = lsock.accept()
            a.close()
            same(False, 0.0)
            peer.close()  # EOF / hang-up: the electing rank closed the channel (or died)
            t0 = time.monotonic()
            same(True)
            self.assertLess(time.monotonic() - t0, 2.0)
            self.assertEqual(ctrl.recv(16), b"")
            fd = ctrl.detach()  # a registered fd that is no longer open raises EBADF, as select() did
            os.close(fd)
            with self.assertRaises(OSError) as cm:
                b._wait(0.0)
            self.assertEqual(cm.exception.errno, errno.EBADF)
            if not high:
                with self.assertRaises(OSError) as cm:
                    select.select([fd], [], [], 0)
                self.assertEqual(cm.exception.errno, errno.EBADF)
        finally:
            for s in socks:
                s.close()

    def test_builder_serves_with_inherited_fds_above_fd_setsize(self) -> None:
        """Two local ranks (elected + client) whose sockets are all >= FD_FLOOR: the builder starts, receives
        the release and GO, publishes every window to both ranks with the reference bytes, and logs no
        'builder failed' line; it exits on the control channel at close. The per-rank partition helper
        (multiprocessing Connection.poll: a PollSelector) answers on fds >= FD_FLOOR too. The code row r1
        fails here: 'm4otf builder failed: ValueError: filedescriptor out of range in select()'."""
        with tempfile.TemporaryDirectory(prefix="m4otf_fd_") as d:
            data = make_data(d, seed=13)
            res, err = _run_driver(self, "_highfd_driver", d, FD_FLOOR, timeout_s=240)
        if res.get("skip"):
            self.skipTest(res["skip"])
        self.assertGreaterEqual(min(res["argv_fds"]), FD_FLOOR, res)  # fd numbers the builder inherited
        self.assertGreaterEqual(min(res["fds"].values()), FD_FLOOR, res)
        self.assertNotIn("m4otf builder failed", err)
        nw = len(data["off"]) - 1
        want = list(range(2, nw))  # progress 1: windows 0, 1 are obsolete
        self.assertEqual(len(re.findall(r"\[m4otf\] node=\S+ builder started pid=", err)), 1, err[-3000:])
        self.assertIn("release received windows=%d lws=2 " % nw, err)
        self.assertIn(" go progress=", err)
        built = re.findall(r"\[m4otf\] node=\S+ j=(\d+) ts=\d+ anchors=\d+ build_ms=", err)
        self.assertEqual([int(x) for x in built], want, err[-3000:])
        sums = {str(j): int(data["table"][data["off"][j]:data["off"][j + 1]].astype(np.int64).sum()) for j in want}
        self.assertEqual(len(res["got"]), 2)
        for g in res["got"]:
            self.assertEqual(g["held"], want, g)
            self.assertEqual(g["sums"], sums, g)
        self.assertTrue(res["builder_alive_after_build"], res)
        self.assertEqual(res["builder_rc"], 0, res)  # exited on the control channel's None/EOF, reaped
        self.assertIn("builder exit built=%d refused=0" % len(want), err)
        self.assertGreaterEqual(min(res["helper_fds"]), FD_FLOOR, res)
        self.assertTrue(res["helper_ok"], res)


def _highfd_driver(d: str, floor: int) -> None:  # pragma: no cover - runs in a subprocess
    import logging

    logging.basicConfig(level=logging.ERROR)
    from generative_recommenders.dlrm_v4.train import dl_prestart

    m4_otf.NFUNC_MODULE = _THIS
    dl_prestart.set_run_start()
    off = np.load(os.path.join(d, _FILES[6]))
    ts_list = list(range(len(off) - 1))
    sizes = [int(off[t + 1] - off[t]) for t in ts_list]
    paths = m4_otf.data_paths(FakeDS(d))
    with _FdFloor(floor) as fl:
        if fl.skip:
            print(json.dumps({"skip": fl.skip}))
            return
        name = _name()
        rts = [m4_otf.OTFRuntime.start(r, ts_list, sizes, paths, 2, time.time(), name=name) for r in range(2)]
        el = rts[0]
        res = {"argv_fds": [int(x) for x in el.builder._p.args[4:6]],  # the builder's ctrl and listening fds
               "fds": {"listen": el._lsock.fileno(), "ctrl": el.builder._ctrl.fileno(),
                       "client0": el._sock.fileno(), "client1": rts[1]._sock.fileno()}}
        time.sleep(0.5)  # the builder's first waits (r1: select() raised at the first one)
        el.progress(1, 10 ** 6)  # go
        want = set(range(2, len(ts_list)))
        t_end = time.time() + 60
        while time.time() < t_end:
            for rt in rts:
                rt.drain(1)
            if all(want <= set(rt._fds) for rt in rts) or el.builder._p.poll() is not None:
                break
            time.sleep(0.02)
        time.sleep(0.2)
        res["got"] = []
        for rt in rts:
            rt.drain(1)
            sums = {}
            for j in sorted(rt._fds):
                fd = rt.dup_fd(j)
                sums[str(j)] = int(m4p.window_array(fd).astype(np.int64).sum())
                os.close(fd)
            res["got"].append({"rank": rt.rank, "held": sorted(rt._fds), "sums": sums})
        res["builder_alive_after_build"] = el.builder._p.poll() is None
        for rt in rts:
            rt.close()
        res["builder_rc"] = el.builder._p.returncode
        hb = m4p.HelperBackend(4, 0, 60.0)  # the per-rank partition helper, also on fds >= floor
        try:
            res["helper_fds"] = [hb._conn.fileno(), int(hb._p.args[4])]
            res["helper_ok"] = hb.ping() == hb.pid
        finally:
            hb.close()
    print(json.dumps(res))


class NRuleConfigTest(unittest.TestCase):
    """Review C5 (iter59): m4_otf.nrule_config_mismatch, the dataset configuration the N rule assumes
    (maybe_create declines otherwise: m4_partition_test CreateTest)."""

    def test_default_matches_and_each_setting_differs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d:
            self.assertEqual(m4_otf.nrule_config_mismatch(FakeDS(d)), "")
            ds = FakeDS(d)
            ds._history_length, ds._max_seq_len, ds._scan_window = np.int64(4086), np.int32(4096), np.int64(20000)
            self.assertEqual(m4_otf.nrule_config_mismatch(ds), "")  # numpy integers are integers
            ctx = dict(YAMBDA_CONFIG["_contextual_feature_to_max_length"])
            cases = [
                ("_history_strategy", "last_n", "$HISTORY_STRATEGY"),
                ("_history_length", 2039, "$HISTORY_LENGTH"),  # the 2k shape
                ("_history_length", 4087, "$HISTORY_LENGTH"),
                ("_history_length", "4086", "$HISTORY_LENGTH"),  # not an integer
                ("_scan_window", 10000, "scan_window"),
                ("_max_seq_len", 2048, "$MAX_SEQ_LEN"),
                ("_max_seq_len", 8192, "$MAX_SEQ_LEN"),
                ("_is_inference", True, "(want False)"),
                ("_max_num_candidates", 2, "HSTU config"),
                ("_max_num_candidates", True, "HSTU config"),  # a bool is not a count
                ("_contextual_feature_to_max_length", dict(list(ctx.items())[:7]), "7 features"),
                ("_contextual_feature_to_max_length", dict(ctx, extra=1), "9 features"),
                ("_contextual_feature_to_max_length", dict(ctx, uid=2), "lengths [1, 2]"),
                ("_contextual_feature_to_max_length", None, "0 features"),
                ("_contextual_feature_to_max_length", ["uid"], "not a feature -> length map"),
            ]
            for name, value, tag in cases:
                ds = FakeDS(d)
                setattr(ds, name, value)
                why = m4_otf.nrule_config_mismatch(ds)
                self.assertTrue(why.startswith(name), (name, value, why))
                self.assertIn(tag, why, (name, value))
                self.assertNotIn(";", why)  # exactly the one setting
            for name in YAMBDA_CONFIG:
                ds = FakeDS(d)
                delattr(ds, name)
                if name == "_max_num_candidates_inference":  # not read: a training dataset (_is_inference False)
                    self.assertEqual(m4_otf.nrule_config_mismatch(ds), "")
                else:
                    self.assertEqual(m4_otf.nrule_config_mismatch(ds), f"{name} missing")
            ds = FakeDS(d)
            ds._history_length, ds._max_seq_len = 2039, 2048  # the 2k production shape: both named
            why = m4_otf.nrule_config_mismatch(ds)
            self.assertEqual(why.count(";"), 1, why)
            self.assertIn("_history_length=2039", why)
            self.assertIn("_max_seq_len=2048", why)
            self.assertEqual(m4_otf.nrule_config_mismatch(object()).count("missing"), 7)

    def test_constants_are_the_rule_and_the_production_defaults(self) -> None:
        o = m4_otf
        self.assertEqual(o.NRULE_HISTORY_LENGTH // 3, 1362)  # PER_POOL
        self.assertEqual(o.NRULE_MAX_SEQ_LEN - o.NRULE_CANDIDATES - o.NRULE_CONTEXTUAL, 4087)  # UIH_BUDGET
        self.assertEqual(o.NRULE_CONTEXTUAL + o.NRULE_CANDIDATES, o.N_MIN)  # N_EXTRA = 9 = N of an empty history
        self.assertEqual(3 * (o.NRULE_HISTORY_LENGTH // 3) + o.N_MIN, o.N_MAX)  # 4095 < MAX_SEQ_LEN: no truncation
        self.assertEqual(o.NRULE_SCAN_WINDOW, 20000)
        self.assertEqual(o.NRULE_STRATEGY, "interleaved")
        try:  # the nfunc row's built-in parameters (absent when only the code row is applied)
            from generative_recommenders.dlrm_v4.train import m4_nfunc
        except ImportError:
            m4_nfunc = None
        if m4_nfunc is not None:
            self.assertEqual((m4_nfunc.SCAN_WINDOW, m4_nfunc.PER_POOL, m4_nfunc.UIH_BUDGET, m4_nfunc.N_EXTRA),
                             (o.NRULE_SCAN_WINDOW, o.NRULE_HISTORY_LENGTH // 3,
                              o.NRULE_MAX_SEQ_LEN - o.NRULE_CANDIDATES - o.NRULE_CONTEXTUAL,
                              o.NRULE_CONTEXTUAL + o.NRULE_CANDIDATES))
        # The production defaults in this tree, from the source text (the dataset modules need torchrec).
        gr = os.path.join(_ROOT, "generative_recommenders")

        def src(*parts: str) -> str:
            with open(os.path.join(gr, *parts)) as f:
                return f.read()

        gin = src("dlrm_v4", "train", "gin", "yambda_5b.gin")

        def gin_env(binding: str, key: str) -> str:
            m = re.search(r"^%s\s*=\s*@(\w+/env_\w+)\(\)\s*$" % re.escape(binding), gin, re.M)
            self.assertIsNotNone(m, binding)
            self.assertRegex(gin, r'(?m)^%s\.key\s*=\s*"%s"\s*$' % (re.escape(m.group(1)), key))
            d = re.search(r"(?m)^%s\.default\s*=\s*(.+?)\s*$" % re.escape(m.group(1)), gin)
            self.assertIsNotNone(d, binding)
            return d.group(1)

        self.assertEqual(gin_env("get_dataset.history_strategy", "HISTORY_STRATEGY"), '"%s"' % o.NRULE_STRATEGY)
        self.assertEqual(int(gin_env("get_dataset.history_length", "HISTORY_LENGTH")), o.NRULE_HISTORY_LENGTH)
        self.assertEqual(int(gin_env("get_hstu_configs.max_seq_len", "MAX_SEQ_LEN")), o.NRULE_MAX_SEQ_LEN)
        self.assertIsNone(re.search(r"(?m)^get_hstu_configs\.max_num_candidates\b", gin))  # the HSTU default stays
        self.assertRegex(src("dlrm_v4", "utils.py"), r'"scan_window":\s*%d,' % o.NRULE_SCAN_WINDOW)
        cfg = src("dlrm_v4", "configs.py")
        tree = ast.parse(cfg)
        specs = [n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "YAMBDA_5B_CROSS_SPECS" for t in n.targets)]
        self.assertEqual(len(specs), 1)
        self.assertEqual(len(specs[0].elts) + 1, o.NRULE_CONTEXTUAL)  # uid + the cross features
        yb = cfg[cfg.index('elif "yambda" in dataset:'):]  # the yambda branch of get_hstu_configs
        yb = yb[:min(i for i in (yb.find("\n    elif "), yb.find("\n    else:"), yb.find("_gin_overrides")) if i >= 0)]
        self.assertRegex(yb, r"hstu_config\.max_num_candidates\s*=\s*%d\b" % o.NRULE_CANDIDATES)
        self.assertRegex(yb, r'contextual_feature_to_max_length\s*=\s*\{\s*"uid":\s*1,\s*\*\*\{name:\s*1 for name in '
                             r"cross_names\}")
        # the attributes the guard reads are the ones the dataset sets and the sample code uses
        ds_src, y = src("dlrm_v4", "datasets", "dataset.py"), src("dlrm_v4", "datasets", "yambda.py")
        for pat in (r"self\._max_seq_len(?:\s*:\s*int)?\s*=\s*hstu_config\.max_seq_len\b",
                    r"self\._max_num_candidates(?:\s*:\s*int)?\s*=\s*hstu_config\.max_num_candidates\b",
                    r"self\._contextual_feature_to_max_length(?:\s*:\s*[^=]+)?=\s*\(?\s*hstu_config\."
                    r"contextual_feature_to_max_length\b",
                    r"self\._is_inference\s*=\s*is_inference\b"):
            self.assertRegex(ds_src, pat)
        for pat in (r"self\._history_length(?:\s*:\s*int)?\s*=\s*history_length\b",
                    r"self\._scan_window(?:\s*:\s*int)?\s*=\s*scan_window\b",
                    r"self\._history_strategy(?:\s*:\s*str)?\s*=\s*history_strategy\b",
                    r"per_pool\s*=\s*max\(1,\s*self\._history_length\s*//\s*3\)",
                    r"max\(int\(user_start\),\s*int\(flat_pos\)\s*-\s*self\._scan_window\)",
                    r"self\._max_seq_len\s*-\s*max_num_candidates\s*-\s*len\(contextual\)",
                    r"self\._max_seq_len\s*-\s*mnc\s*-\s*n_ctx\b"):
            self.assertRegex(y, pat)


# -- r3 (lab 71897 decision): the shipped GO_BATCHES default -------------------------------------------------------
class GoDefaultTest(unittest.TestCase):
    """DLRM_HSTU_FP8_M4_OTF_GO_BATCHES defaults to 0 (r2: 32): the elected rank's first progress sends 'go' even
    before any trained batch, so the builder starts at window boundary 1. The env stays a diagnostic override."""

    def test_default_is_zero_and_the_first_progress_sends_go(self) -> None:
        self.assertEqual(m4_otf.DEFAULT_GO_BATCHES, 0)
        with tempfile.TemporaryDirectory(prefix="m4otf_") as d, _Env():
            os.environ.pop(m4_otf.ENV_GO_BATCHES, None)  # the shipped form: no override
            data = make_data(d, seed=17)
            want = list(range(2, len(data["off"]) - 1))  # progress 1: windows 0, 1 are obsolete
            rt = _start(0, data, d, _name(), lws=1)
            try:
                self.assertEqual(rt.go_batches, 0)
                self.assertFalse(rt.builder.go_sent)  # released: no 'go' before the first progress
                rt.progress(1, 0)  # window boundary 1, no batch trained yet
                self.assertTrue(rt.builder.go_sent)
                self.assertTrue(_wait(lambda: (rt.drain(1), all(rt.holds(j) for j in want))[1]))
            finally:
                rt.close()
        for v, want_gb in (("32", 32), ("0", 0), ("", 0)):  # the env wins; empty = the default
            with _Env(DLRM_HSTU_FP8_M4_OTF_GO_BATCHES=v):
                self.assertEqual(m4_otf.OTFRuntime(0, [0], [8]).go_batches, want_gb, v)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# pyre-strict
"""DL_COLD_PREFETCH_STREAM: the paced read-ahead walks exactly the sampler's
batch sequence (per-rank slice, drop_last, per-window cap, first-window resume
skip) in pull order, stays within the lead of the consumer, skips batches the
consumer already pulled, respects the horizon and stops with the stream.
``auto`` (AutoStreamTest) runs the walk only on multi-node runs ($NNODES > 1;
otherwise off) and names the outcome on rank 0.

numpy only (no torch, no files): ``_ranges`` and ``_files`` are replaced by
recorders, and a consumer thread drives ``note_pull`` / ``note_window_done``
exactly as ``_WarmWindowStream`` / ``_PrefetchingWindowLoader.stream`` do."""

import logging
import os
import re
import threading
import time
import unittest
from typing import Dict, List, Optional, Tuple
from unittest import mock

import numpy as np

from generative_recommenders.dlrm_v4.train import cold_prefetch as cp

_BS = 4
_WS = 2
# Per-window anchor counts: tiny windows (0..2 batches per rank) first, then
# larger ones, like the START_TS=0 production schedule.
_SIZES = [3, 9, 16, 0, 8, 25, 17, 40, 64, 70, 130, 200, 260, 300]


class _Ds:
    _scan_window = 100

    def __init__(self, quiet_kw: bool = True) -> None:
        self.calls: List[Tuple[int, bool]] = []
        self._quiet_kw = quiet_kw
        off = np.concatenate([[0], np.cumsum(_SIZES)]).astype(np.int64)
        self._win = [np.arange(off[i], off[i + 1], dtype=np.int64) * 3 + 7
                     for i in range(len(_SIZES))]
        if not quiet_kw:
            self.train_window_indices = self._legacy  # type: ignore[assignment]

    def train_window_indices(self, ts: int, quiet: bool = False) -> np.ndarray:
        self.calls.append((ts, quiet))
        return self._win[ts]

    def _legacy(self, ts: int) -> np.ndarray:
        self.calls.append((ts, False))
        return self._win[ts]


def _sequence(ds: _Ds, rank: int, cap: Optional[int] = None, skip: int = 0,
              first: int = 0) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """(window ordinal in the stream, batch, sample tuple) in pull order, as the
    sampler yields them for the stream ts_list = [first, ..., last]."""
    out = []
    for i, w in enumerate(ds._win[first:]):
        n = (len(w) // _WS) * _WS
        per_rank = w[:n][rank::_WS]
        if i == 0 and skip:
            per_rank = per_rank[skip:]
        nb = len(per_rank) // _BS
        if cap:  # _WarmWindowStream: the resumed window pulls cap - K batches
            nb = min(nb, max(0, cap - (skip // _BS if i == 0 else 0)))
        for b in range(nb):
            out.append((i, b, tuple(per_rank[b * _BS:(b + 1) * _BS].tolist())))
    return out


class _StreamHarness(unittest.TestCase):
    """The recorder + consumer harness (no tests of its own)."""

    def setUp(self) -> None:
        cp._reset_for_tests()
        self.issued: List[Tuple[float, Tuple[int, ...], int]] = []
        self.lock = threading.Lock()
        self.pos = [0]  # consumer: batches pulled

        def rec(pool, files, samples, scan_window, how, stats, gap_pages=cp._GAP_PAGES,
                serial=False):  # noqa
            with self.lock:
                self.issued.append((time.time(), tuple(np.asarray(samples).tolist()), self.pos[0]))

        class _F:
            def close(self) -> None:
                pass

        for p in (
            mock.patch.object(cp, "_ranges", rec),
            mock.patch.object(cp, "_files", lambda ds: {"positions": _F()}),
            mock.patch.object(cp, "_WAIT_S", 0.002),
        ):
            p.start()
            self.addCleanup(p.stop)  # stopped even if a test fails midway
        cp._RELEASED.set()

    def tearDown(self) -> None:
        cp._reset_for_tests()

    def _fresh(self) -> None:
        """Per-case reset inside one test (the patches stay as set up)."""
        cp._reset_for_tests()
        cp._RELEASED.set()
        self.issued.clear()
        self.pos[0] = 0

    def _batches_issued(self) -> List[Tuple[int, ...]]:
        """Split each recorded call back into batches of _BS samples."""
        out = []
        for _, s, _ in self.issued:
            for k in range(0, len(s), _BS):
                out.append(s[k:k + _BS])
        return out

    def _consume(self, ds: _Ds, rank: int, seq, delay: float, warm: bool = True,
                 cap: Optional[int] = None, first: int = 0) -> None:
        """Pull the sequence like _WarmWindowStream (+ stream() resume hooks)."""
        cur = 0
        for i, w in enumerate(ds._win[first:]):
            nb = sum(1 for (wi, _, _) in seq if wi == i)
            for _ in range(nb):
                time.sleep(delay)
                if warm:
                    cp.note_pull()
                with self.lock:
                    self.pos[0] += 1
                cur += 1
            cp.note_window_done(i + 1)

    def _go(self, ds, rank=0, cap=None, skip=0, lead=6, horizon=10_000, group=2,
            delay=0.004, warm=True, consume=True, k_batches=8, first=0):
        seq = _sequence(ds, rank, cap=cap, skip=skip, first=first)
        th = threading.Thread(
            target=cp._run,
            args=(ds, list(range(first, len(_SIZES))), rank, _WS, _BS, skip, 1, k_batches, 2),
            kwargs=dict(stream=True, window_cap=cap, lead=lead, horizon=horizon, group=group),
            daemon=True,
        )
        th.start()
        if consume:
            self._consume(ds, rank, seq, delay, warm=warm, cap=cap, first=first)
            cp.stop()
        th.join(timeout=30)
        self.assertFalse(th.is_alive())
        return seq



class StreamTest(_StreamHarness):
    def test_walks_the_sampler_sequence_in_order(self) -> None:
        for rank in (0, 1):
            # (a resume re-enters a capped window only at K < cap batches)
            for cap, skip, first in ((None, 0, 0), (1, 0, 0), (2, 0, 0), (None, 3, 0),
                                     (None, 8, 8), (3, 4, 8), (3, 8, 8), (5, 8, 9)):
                with self.subTest(rank=rank, cap=cap, skip=skip, first=first):
                    self._fresh()
                    ds = _Ds()
                    seq = self._go(ds, rank=rank, cap=cap, skip=skip, first=first)
                    want = [s for _, _, s in seq]
                    got = self._batches_issued()
                    self.assertEqual(got, want[:len(got)], (rank, cap, skip))
                    # paced and slow consumer: nothing is skipped as late
                    self.assertEqual(len(got), len(want), (rank, cap, skip))
                    self.assertTrue(all(q for _, q in ds.calls), "train_window_indices not quiet")

    def test_stays_within_lead(self) -> None:
        ds = _Ds()
        lead, group = 6, 2
        seq = self._go(ds, lead=lead, group=group, delay=0.003)
        idx = {s: n for n, (_, _, s) in enumerate(seq)}
        for _, s, pos in self.issued[1:]:  # [0] is the first-batch wave
            first = idx[s[:_BS]]
            self.assertLessEqual(first - pos, lead, (first, pos))

    def test_horizon(self) -> None:
        ds = _Ds()
        self._go(ds, horizon=7)
        # batch 1 from the first wave + the walk up to the horizon (7 batches)
        self.assertLessEqual(len(self._batches_issued()), 7)
        self.assertGreaterEqual(len(self._batches_issued()), 6)

    def test_skips_batches_already_pulled(self) -> None:
        ds = _Ds()
        seq = _sequence(ds, 0)
        # The consumer pulled everything before the walk starts.
        for i in range(len(_SIZES)):
            nb = sum(1 for (wi, _, _) in seq if wi == i)
            for _ in range(nb):
                cp.note_pull()
        cp.note_window_done(len(_SIZES))
        with mock.patch.object(cp, "_DONE", [len(_SIZES), cp._PULLED[0]]):
            self._go(ds, consume=False)
        self.assertLessEqual(len(self._batches_issued()), 1)  # only the first wave

    def test_window_granularity_without_pull_counts(self) -> None:
        """Without warm-keep pull counts the consumer position moves only at
        window ends: windows of <= lead batches are covered completely, larger
        ones up to the lead; everything issued is an ordered prefix per window."""
        ds = _Ds()
        lead, group = 6, 2
        seq = self._go(ds, lead=lead, group=group, warm=False)
        got = self._batches_issued()
        pos = {s: n for n, (_, _, s) in enumerate(seq)}
        order = [pos[s] for s in got]
        self.assertEqual(order, sorted(set(order)), "not an ordered subsequence")
        by_win: Dict[int, List[int]] = {}
        for n in order:
            by_win.setdefault(seq[n][0], []).append(seq[n][1])
        for w, bs in by_win.items():  # (the first wave's batch is in `got` too)
            self.assertEqual(bs, list(range(len(bs))), (w, bs))
        for w in {wi for wi, _, _ in seq}:
            nb = sum(1 for wi, _, _ in seq if wi == w)
            if nb <= lead:
                self.assertEqual(len(by_win.get(w, [])), nb, (w, nb))
            else:
                self.assertGreaterEqual(len(by_win.get(w, [])), lead - group, (w, nb))

    def test_stop_ends_the_walk(self) -> None:
        ds = _Ds()
        th = threading.Thread(
            target=cp._run, args=(ds, list(range(len(_SIZES))), 0, _WS, _BS, 0, 1, 8, 2),
            kwargs=dict(stream=True, lead=2, horizon=10_000, group=1), daemon=True)
        th.start()
        time.sleep(0.2)  # walk blocked on pacing (nobody pulls)
        self.assertTrue(th.is_alive())
        n = len(self._batches_issued())
        self.assertLessEqual(n, 1 + 2 + 1)
        cp.stop()
        th.join(timeout=5)
        self.assertFalse(th.is_alive())

    def test_no_offline_index_disables_the_walk(self) -> None:
        """DL_WINDOW_INDEX=0 (gin env_int -> int 0, also False/None): every
        window would be a full anchor scan, so only the first wave runs."""
        for flag in (0, False, None):
            with self.subTest(flag=flag):
                self._fresh()
                ds = _Ds()
                ds._use_offline_ts_indices = flag  # type: ignore[attr-defined]
                with self.assertLogs(cp.logger, level="WARNING") as lg:
                    self._go(ds)
                self.assertEqual(len(self._batches_issued()), 1)  # first wave only
                self.assertTrue(any("stream off: no offline window index" in m for m in lg.output))
        for flag in (1, True):
            with self.subTest(flag=flag):
                self._fresh()
                ds = _Ds()
                ds._use_offline_ts_indices = flag  # type: ignore[attr-defined]
                seq = self._go(ds)
                self.assertEqual(len(self._batches_issued()), len(seq))

    def test_legacy_dataset_without_quiet(self) -> None:
        ds = _Ds(quiet_kw=False)
        seq = self._go(ds)
        self.assertEqual(self._batches_issued(), [s for _, _, s in seq])

    def test_one_shot_mode_unchanged(self) -> None:
        ds = _Ds()
        th = threading.Thread(
            target=cp._run, args=(ds, list(range(len(_SIZES))), 0, _WS, _BS, 0, 1, 8, 2),
            daemon=True)
        th.start()
        th.join(timeout=10)
        seq = _sequence(ds, 0)
        # two waves: batch 1 alone, then batches 2..8 in one call
        self.assertEqual(len(self.issued), 2)
        self.assertEqual(self._batches_issued(), [s for _, _, s in seq][:8])

    def test_start_defaults(self) -> None:
        """start(): horizon = the whole stream (HORIZON_ALL) at every world size
        unless $DL_COLD_PREFETCH_HORIZON caps it; stream only with the env
        switch; the pacing counters restart at the stream's start."""
        seen = []

        class _T:
            def __init__(self, target, args, kwargs, name, daemon):  # noqa
                seen.append((args, kwargs))

            def start(self) -> None:
                pass

            def join(self, timeout=None) -> None:  # noqa
                pass

            def is_alive(self) -> bool:
                return False

        self.addCleanup(cp._THREADS.clear)
        cp._PULLED[0] = 17
        with mock.patch.object(cp.threading, "Thread", _T):
            for ws, bs, env, want_h, want_s in (
                (8, 1024, {}, cp.HORIZON_ALL, False),
                (16, 1024, {"DL_COLD_PREFETCH_STREAM": "1"}, cp.HORIZON_ALL, True),
                (32, 1024, {"DL_COLD_PREFETCH_STREAM": "1"}, cp.HORIZON_ALL, True),
                (4, 1024, {"DL_COLD_PREFETCH_STREAM": "1"}, cp.HORIZON_ALL, True),
                (16, 1024, {"DL_COLD_PREFETCH_STREAM": "1", "DL_COLD_PREFETCH_HORIZON": "12"}, 12, True),
            ):
                env = dict(env, DL_COLD_PREFETCH="1")
                with mock.patch.dict("os.environ", env, clear=True):
                    cp.start(_Ds(), [0, 1], 0, ws, bs, window_cap=3)
                args, kw = seen[-1]
                self.assertEqual(kw["horizon"], want_h, (ws, env))
                self.assertEqual(kw["stream"], want_s, (ws, env))
                self.assertEqual(kw["window_cap"], 3)
                self.assertEqual((kw["lead"], kw["group"]), (64, 4))
                self.assertEqual(cp._DONE, [0, 17])

    def test_env_switch(self) -> None:
        with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_STREAM": "1"}):
            self.assertTrue(cp.stream_on())
        with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_STREAM": "0"}):
            self.assertFalse(cp.stream_on())
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(cp.stream_on())
        # auto is not a force-on; start() resolves it (AutoStreamTest)
        with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_STREAM": "auto"}):
            self.assertFalse(cp.stream_on())


class _Lines(logging.Handler):
    """cold_prefetch's WARNING lines (an empty list is a valid outcome)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


_OFF = "[cold-prefetch] stream=auto resolved off: "
# Every variable the decision or start() reads: cleared unless a case sets it.
_KNOBS = (
    "DL_COLD_PREFETCH", "DL_PRESTART", "DL_COLD_PREFETCH_STREAM", "NNODES",
    "DL_COLD_PREFETCH_EVAL", "DL_COLD_PREFETCH_EVAL_LEAD", "DL_COLD_PREFETCH_EVAL_GAP_S",
    "LOCAL_WORLD_SIZE", "DL_COLD_PREFETCH_HORIZON", "DL_COLD_PREFETCH_LEAD",
    "DL_COLD_PREFETCH_GROUP", "DL_COLD_PREFETCH_BATCHES", "DL_COLD_PREFETCH_THREADS",
)


class AutoStreamTest(unittest.TestCase):
    """DL_COLD_PREFETCH_STREAM=auto: start() runs the walk iff the run spans
    more than one node ($NNODES > 1; NNODES=1, unset or not a positive integer
    keeps it off, whatever $LOCAL_WORLD_SIZE says). 1/true/on/yes force it
    on and unset/0/off/false/no keep it off at any node count. Rank 0 logs the
    outcome of auto once: the armed line ends with auto=multi-node(...), or one
    "stream=auto resolved off" line; the one-shot prefetch is unchanged."""

    def setUp(self) -> None:
        cp._reset_for_tests()
        cp._RELEASED.set()
        # the walk thread in _start_and_run runs outside self._env(): keep the
        # eval read-ahead knobs of the calling shell (e.g. a W0 arm's
        # DL_COLD_PREFETCH_EVAL=0) out of these tests
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("DL_COLD_PREFETCH_EVAL")}
        pe = mock.patch.dict("os.environ", env, clear=True)
        pe.start()
        self.addCleanup(pe.stop)
        self.log = _Lines()
        cp.logger.addHandler(self.log)
        self.addCleanup(cp.logger.removeHandler, self.log)
        level = cp.logger.level
        cp.logger.setLevel(logging.WARNING)
        self.addCleanup(cp.logger.setLevel, level)
        self.addCleanup(cp._reset_for_tests)

    @staticmethod
    def _env(**env: str):
        base = {k: v for k, v in os.environ.items() if k not in _KNOBS}
        base.update(env)
        return mock.patch.dict("os.environ", base, clear=True)

    def _start(self, rank: int = 0, ws: int = _WS, ts_list=(0, 1), **env: str):
        """start() with the prefetcher armed (DL_COLD_PREFETCH=1), a per-window
        cap of 3 and a recording Thread: (thread args, thread kwargs, lines it
        logged)."""
        seen = []

        class _T:
            def __init__(self, target, args, kwargs, name, daemon):  # noqa
                seen.append((args, kwargs))

            def start(self) -> None:
                pass

            def join(self, timeout=None) -> None:  # noqa
                pass

            def is_alive(self) -> bool:
                return False

        n0 = len(self.log.lines)
        with self._env(DL_COLD_PREFETCH="1", **env), \
                mock.patch.object(cp.threading, "Thread", _T):
            th = cp.start(_Ds(), list(ts_list), rank, ws, _BS, window_cap=3)
        cp._THREADS.clear()
        self.assertIsNotNone(th)
        self.assertEqual(len(seen), 1)
        args, kw = seen[0]
        return args, kw, self.log.lines[n0:]

    def test_resolve_stream(self) -> None:
        a = {"DL_COLD_PREFETCH_STREAM": "auto"}
        unknown = "node count unknown (NNODES unset)"
        for env, want in (
            # $NNODES decides (an emulated larger world on one node stays off)
            (dict(a, NNODES="1"), (False, "single node (nnodes=1)")),
            (dict(a, NNODES="2"), (True, "multi-node(nnodes=2)")),
            (dict(a, NNODES="4"), (True, "multi-node(nnodes=4)")),
            (dict(a, NNODES=" 2 "), (True, "multi-node(nnodes=2)")),
            # $LOCAL_WORLD_SIZE never decides, with or without NNODES
            (dict(a, NNODES="1", LOCAL_WORLD_SIZE="1"), (False, "single node (nnodes=1)")),
            (dict(a, NNODES="2", LOCAL_WORLD_SIZE="8"), (True, "multi-node(nnodes=2)")),
            (dict(a, LOCAL_WORLD_SIZE="1"), (False, unknown)),
            (dict(a, LOCAL_WORLD_SIZE="8"), (False, unknown)),
            # unset / not a positive integer: off (train_ranker: one node)
            (a, (False, unknown)),
            (dict(a, NNODES=""), (False, unknown)),
            (dict(a, NNODES="abc"),
             (False, "node count unknown (NNODES='abc' not a positive integer)")),
            (dict(a, NNODES="0"),
             (False, "node count unknown (NNODES='0' not a positive integer)")),
            (dict(a, NNODES="-2"),
             (False, "node count unknown (NNODES='-2' not a positive integer)")),
            (dict(a, NNODES="2.5", LOCAL_WORLD_SIZE="1"),
             (False, "node count unknown (NNODES='2.5' not a positive integer)")),
            ({"DL_COLD_PREFETCH_STREAM": " Auto ", "NNODES": "2"}, (True, "multi-node(nnodes=2)")),
            # explicit values: the node count is not consulted, no reason
            ({"DL_COLD_PREFETCH_STREAM": "1", "NNODES": "1"}, (True, "")),
            ({"DL_COLD_PREFETCH_STREAM": "1", "NNODES": "4"}, (True, "")),
            ({"DL_COLD_PREFETCH_STREAM": "1", "NNODES": "abc"}, (True, "")),
            ({"DL_COLD_PREFETCH_STREAM": "on"}, (True, "")),
            ({"DL_COLD_PREFETCH_STREAM": "True", "NNODES": "1"}, (True, "")),
            ({"DL_COLD_PREFETCH_STREAM": "yes"}, (True, "")),
            ({"NNODES": "4"}, (False, "")),  # unset: off (the default)
            ({}, (False, "")),
            ({"DL_COLD_PREFETCH_STREAM": "", "NNODES": "4"}, (False, "")),
            ({"DL_COLD_PREFETCH_STREAM": "0", "NNODES": "4"}, (False, "")),
            ({"DL_COLD_PREFETCH_STREAM": "off", "NNODES": "4"}, (False, "")),
            ({"DL_COLD_PREFETCH_STREAM": "false", "NNODES": "4"}, (False, "")),
            ({"DL_COLD_PREFETCH_STREAM": "no", "NNODES": "4"}, (False, "")),
            ({"DL_COLD_PREFETCH_STREAM": "2", "NNODES": "4"}, (False, "")),
        ):
            with self.subTest(env=env):
                with self._env(**env):
                    self.assertEqual(cp.resolve_stream(), want)

    def test_start_decides_and_logs_once_on_rank0(self) -> None:
        for env, ws, stream, line in (
            ({"NNODES": "1"}, 8, False, _OFF + "single node (nnodes=1)"),
            ({"NNODES": "2"}, 16, True, None),
            ({"NNODES": "4"}, 32, True, None),
            # a stray $LOCAL_WORLD_SIZE without NNODES never turns it on
            ({"LOCAL_WORLD_SIZE": "1"}, 8, False, _OFF + "node count unknown (NNODES unset)"),
            ({"LOCAL_WORLD_SIZE": "8"}, 16, False, _OFF + "node count unknown (NNODES unset)"),
            ({}, 16, False, _OFF + "node count unknown (NNODES unset)"),
            ({"NNODES": "abc"}, 16, False, _OFF + "node count unknown (NNODES='abc' not a "
                                                  "positive integer)"),
        ):
            for rank in (0, 1):
                with self.subTest(env=env, ws=ws, rank=rank):
                    _, kw, lines = self._start(rank=rank, ws=ws,
                                               DL_COLD_PREFETCH_STREAM="auto", **env)
                    self.assertEqual(kw["stream"], stream)
                    if stream:
                        self.assertRegex(kw["stream_auto"], r"^multi-node\(.+\)$")
                    else:
                        self.assertEqual(kw["stream_auto"], "")
                    # exactly one line, rank 0 only, and only when auto is off
                    self.assertEqual(lines, [line] if line and rank == 0 else [])
                    # nothing else changes with the mode
                    self.assertEqual(kw["horizon"], cp.HORIZON_ALL)
                    self.assertEqual((kw["lead"], kw["group"], kw["window_cap"]), (64, 4, 3))
        # explicit values and the default log nothing new at any node count
        for env, stream in (
            ({"DL_COLD_PREFETCH_STREAM": "1", "NNODES": "1"}, True),
            ({"DL_COLD_PREFETCH_STREAM": "1", "NNODES": "4"}, True),
            ({"DL_COLD_PREFETCH_STREAM": "1"}, True),
            ({"NNODES": "4"}, False),
            ({}, False),
            ({"DL_COLD_PREFETCH_STREAM": "0", "NNODES": "4"}, False),
        ):
            with self.subTest(env=env):
                _, kw, lines = self._start(ws=8, **env)
                self.assertEqual((kw["stream"], kw["stream_auto"], lines), (stream, "", []))
        # prefetcher off: start() does nothing and logs nothing, auto or not
        n0 = len(self.log.lines)
        with self._env(DL_COLD_PREFETCH_STREAM="auto", NNODES="1"), \
                mock.patch.object(cp, "_PRESTART_LEVEL", [0]):
            self.assertIsNone(cp.start(_Ds(), [0, 1], 0, 8, _BS))
        self.assertEqual(self.log.lines[n0:], [])
        self.assertEqual(cp._THREADS, [])

    def _start_and_run(self, **env: str):
        """start()'s decision, then its thread body for real: no files
        (``_ranges`` records the batches), pulled by a consumer like
        _WarmWindowStream (per-window cap 3) once the armed line is out.
        Returns (lines, the batches issued in order, the sampler sequence)."""
        cp._reset_for_tests()
        cp._RELEASED.set()
        ts_list = list(range(len(_SIZES)))
        args, kw, lines = self._start(ts_list=ts_list, **env)
        ds = args[0]
        issued: List[Tuple[int, ...]] = []

        def rec(pool, files, samples, *a, **k):  # noqa
            issued.append(tuple(np.asarray(samples).tolist()))

        class _F:
            def close(self) -> None:
                pass

        seq = _sequence(ds, 0, cap=3)
        n0 = len(self.log.lines)
        with mock.patch.object(cp, "_ranges", rec), \
                mock.patch.object(cp, "_files", lambda ds: {"positions": _F()}), \
                mock.patch.object(cp, "_WAIT_S", 0.002):
            th = threading.Thread(target=cp._run, args=args, kwargs=kw, daemon=True)
            th.start()
            t_end = time.time() + 10
            while not any("[cold-prefetch] armed" in m for m in self.log.lines[n0:]):
                self.assertLess(time.time(), t_end, "no armed line")
                time.sleep(0.002)
            for i in range(len(ts_list)):
                for _ in range(sum(1 for wi, _, _ in seq if wi == i)):
                    time.sleep(0.005)
                    cp.note_pull()
                cp.note_window_done(i + 1)
            cp.stop()
            th.join(timeout=30)
        self.assertFalse(th.is_alive())
        batches = [s[k:k + _BS] for s in issued for k in range(0, len(s), _BS)]
        return lines + self.log.lines[n0:], batches, [s for _, _, s in seq]

    @staticmethod
    def _armed(lines: List[str]) -> List[str]:
        return [m for m in lines if "[cold-prefetch] armed" in m]

    def test_armed_line_walk_and_one_shot(self) -> None:
        # auto on two nodes: the whole sequence is walked; the armed line has
        # the stream fields and the reason; no "resolved off" line
        lines, got, seq = self._start_and_run(DL_COLD_PREFETCH_STREAM="auto", NNODES="2")
        armed = self._armed(lines)
        self.assertEqual(len(armed), 1, lines)
        self.assertIn("stream=on lead=64 ", armed[0])
        self.assertIn("batches=1 ", armed[0])
        # (no eval plan in these tests: the eval read-ahead fields say so)
        self.assertTrue(armed[0].endswith(" cap=3 auto=multi-node(nnodes=2) eval=off(no plan)"),
                        armed[0])
        self.assertFalse(any("resolved off" in m for m in lines), lines)
        self.assertEqual(sum("[cold-prefetch] stream done" in m for m in lines), 1, lines)
        self.assertEqual(got, seq)
        # explicit 1 on one node: the same walk, no auto reason
        lines1, got1, _ = self._start_and_run(DL_COLD_PREFETCH_STREAM="1", NNODES="1")
        armed1 = self._armed(lines1)
        self.assertEqual(len(armed1), 1, lines1)
        self.assertTrue(armed1[0].endswith(
            " stream=on lead=64 horizon=all group=4 cap=3 eval=off(no plan)"), armed1[0])
        self.assertEqual(got1, seq)
        # auto on one node: one-shot only, exactly as with the switch unset
        lines0, got0, _ = self._start_and_run(NNODES="1")
        lines_off, got_off, _ = self._start_and_run(DL_COLD_PREFETCH_STREAM="auto", NNODES="1")
        self.assertEqual(got_off, seq[:8])  # first wave + second wave, K = 8
        self.assertEqual(got_off, got0)
        self.assertEqual([m for m in lines_off if "resolved off" in m],
                         [_OFF + "single node (nnodes=1)"])
        armed_off, armed0 = self._armed(lines_off), self._armed(lines0)
        self.assertEqual((len(armed_off), len(armed0)), (1, 1), lines_off + lines0)
        self.assertNotIn("stream=on lead=", armed_off[0])
        self.assertNotIn("auto=", armed_off[0])
        self.assertIn("batches=8 ", armed_off[0])

        def untimed(m: str) -> str:
            return re.sub(r" (t_after_release|issue_s|t_rel_run_start)=\S+", "", m)

        self.assertEqual(untimed(armed_off[0]), untimed(armed0[0]))
        self.assertFalse(any("stream done" in m for m in lines_off + lines0))
        # the default (unset) logs only the armed line
        self.assertEqual(lines0, armed0)


# Eval holdout anchors: distinct from every train anchor (train ids are < 3 *
# sum(_SIZES) + 7), so the recorder tells the two walks apart.
_EVAL_BASE = 1_000_000


def _holdout(n: int) -> np.ndarray:
    return (np.arange(n, dtype=np.int64) * 5 + _EVAL_BASE)


def _eval_sequence(idx: np.ndarray, rank: int, bs: int = _BS, cap: int = 0
                   ) -> List[Tuple[int, ...]]:
    """One eval pass of this rank as the eval loader sees it:
    StreamingWindowSampler.set_window (multiple-of-world_size prefix, rank::ws)
    + DataLoader(drop_last=True) + the num_eval_batches cap."""
    n = (len(idx) // _WS) * _WS
    per_rank = idx[:n][rank::_WS].tolist()
    nb = len(per_rank) // bs
    if cap:
        nb = min(nb, cap)
    return [tuple(per_rank[b * bs:(b + 1) * bs]) for b in range(nb)]


class EvalHoldoutTest(_StreamHarness):
    """Item (36)'s eval holdout read-ahead (cold_prefetch._EvalWalk): with an
    eval plan the walk issues exactly this rank's eval batches (sampler slice,
    drop_last, cap) in order, not before the consumer is within LEAD pulled
    batches of the first eval, one group at most every GAP_S; the rest at once
    when the train walk reaches the end of the stream; nothing when the switch
    is off, without a plan, after stop() or when the horizon cut the walk."""

    def setUp(self) -> None:
        super().setUp()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("DL_COLD_PREFETCH_EVAL")}
        p = mock.patch.dict("os.environ", env, clear=True)
        p.start()
        self.addCleanup(p.stop)
        # The fixture's stream is ~140 batches; production evals are thousands
        # of batches in. Shrink the "imminent eval" margin unless a test sets it.
        u = mock.patch.object(cp._EvalWalk, "URGENT", 2)
        u.start()
        self.addCleanup(u.stop)

    def _split(self):
        """(train calls, eval calls) as (pos at the call, batches) lists."""
        tr, ev = [], []
        for _, s, pos in self.issued:
            bl = [s[k:k + _BS] for k in range(0, len(s), _BS)]
            (ev if s and s[0] >= _EVAL_BASE else tr).append((pos, bl))
        return tr, ev

    @staticmethod
    def _flat(calls) -> List[Tuple[int, ...]]:
        return [b for _, bl in calls for b in bl]

    def test_set_eval_plan_validation(self) -> None:
        idx = _holdout(10)
        for args in ((None, 4, 3), (idx, 0, 3), (idx, 4, 0), (idx, 4, -1), (idx, "x", 3)):
            cp.set_eval_plan(*args)
            self.assertIsNone(cp._EVAL_PLAN[0], args)
        cp.set_eval_plan(idx, 4, 3, None)
        self.assertEqual({k: v for k, v in cp._EVAL_PLAN[0].items() if k != "indices"},
                         {"bs": 4, "at": 3, "cap": 0})
        self.assertIs(cp._EVAL_PLAN[0]["indices"], idx)  # stored, not copied
        cp.set_eval_plan(idx, 4, 3, -2)
        self.assertEqual(cp._EVAL_PLAN[0]["cap"], 0)

    def test_rank_eval_batches_is_the_eval_loader_sequence(self) -> None:
        for n in (0, 3, 37, 64, 65):
            idx = _holdout(n)
            for rank in (0, 1):
                for cap in (0, 1, 3, 100):
                    per_rank, nb = cp.rank_eval_batches(idx, rank, _WS, _BS, cap)
                    got = [tuple(per_rank[b * _BS:(b + 1) * _BS].tolist()) for b in range(nb)]
                    self.assertEqual(got, _eval_sequence(idx, rank, cap=cap), (n, rank, cap))

    def test_walks_the_eval_pass_after_the_lead(self) -> None:
        # 37 anchors: 36 = 18 per rank -> 4 batches (drop_last); trigger at
        # pulled >= at - lead = 30 - 10 = 20; no gap: one group per pump.
        idx = _holdout(37)
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self._fresh()
                cp.set_eval_plan(idx, _BS, 30)
                with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_EVAL_LEAD": "10",
                                                    "DL_COLD_PREFETCH_EVAL_GAP_S": "0"}):
                    seq = self._go(_Ds(), rank=rank, lead=6, group=2)
                tr, ev = self._split()
                self.assertEqual(self._flat(ev), _eval_sequence(idx, rank))
                self.assertTrue(all(pos >= 20 for pos, _ in ev), ev)
                self.assertTrue(all(len(bl) <= 2 for _, bl in ev), ev)  # one group per call
                # the train walk is unchanged by the eval read-ahead
                self.assertEqual(self._flat(tr), [s for _, _, s in seq])

    def test_cap_limits_the_eval_pass(self) -> None:
        idx = _holdout(64)  # 32 per rank -> 8 batches; cap 3
        cp.set_eval_plan(idx, _BS, 1, 3)
        with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_EVAL_GAP_S": "0"}):
            self._go(_Ds(), rank=1, lead=6, group=2)
        _, ev = self._split()
        self.assertEqual(self._flat(ev), _eval_sequence(idx, 1, cap=3))

    def test_kill_switch_and_no_plan(self) -> None:
        idx = _holdout(37)
        for env, plan in (({"DL_COLD_PREFETCH_EVAL": "0"}, True),
                          ({"DL_COLD_PREFETCH_EVAL": "off"}, True),
                          ({"DL_COLD_PREFETCH_EVAL": "disable"}, True),  # fail closed
                          ({}, False)):
            with self.subTest(env=env, plan=plan):
                self._fresh()
                if plan:
                    cp.set_eval_plan(idx, _BS, 1)
                with mock.patch.dict("os.environ", dict(env, DL_COLD_PREFETCH_EVAL_GAP_S="0")):
                    seq = self._go(_Ds(), rank=0, lead=6, group=2)
                tr, ev = self._split()
                self.assertEqual(ev, [])
                self.assertEqual(self._flat(tr), [s for _, _, s in seq])

    def test_rest_at_once_when_the_train_walk_ends_first(self) -> None:
        # trigger never reached (at huge); the walk covers the whole stream at
        # once (lead >= every batch) and then issues the whole eval pass
        idx = _holdout(37)
        cp.set_eval_plan(idx, _BS, 10 ** 9)
        with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_EVAL_GAP_S": "100"}):
            seq = self._go(_Ds(), rank=0, lead=100_000, group=2, consume=False)
        tr, ev = self._split()
        self.assertEqual(self._flat(tr), [s for _, _, s in seq])
        self.assertEqual(self._flat(ev), _eval_sequence(idx, 0))
        # after every train call (forced at the end, gap ignored)
        t_last_train = max(t for t, s, _ in self.issued if s and s[0] < _EVAL_BASE)
        t_first_eval = min(t for t, s, _ in self.issued if s and s[0] >= _EVAL_BASE)
        self.assertGreaterEqual(t_first_eval, t_last_train)

    def test_nothing_after_stop_or_horizon_cut(self) -> None:
        idx = _holdout(37)
        # stop(): the consumer never pulls, the walk waits at the lead, stop()
        cp.set_eval_plan(idx, _BS, 10 ** 9)
        th = threading.Thread(
            target=cp._run,
            args=(_Ds(), list(range(len(_SIZES))), 0, _WS, _BS, 0, 1, 8, 2),
            kwargs=dict(stream=True, window_cap=None, lead=6, horizon=10_000, group=2),
            daemon=True)
        th.start()
        time.sleep(0.2)
        cp.stop()
        th.join(timeout=30)
        self.assertFalse(th.is_alive())
        self.assertEqual(self._split()[1], [])
        # horizon: the walk ends cut (the read-around takes over); no eval pass
        self._fresh()
        cp.set_eval_plan(idx, _BS, 10 ** 9)
        self._go(_Ds(), rank=0, lead=100_000, horizon=5, group=2, consume=False)
        tr, ev = self._split()
        self.assertEqual(ev, [])
        self.assertLessEqual(len(self._flat(tr)), 1 + 5)  # first batch + horizon

    def test_gap_paces_the_eval_groups(self) -> None:
        idx = _holdout(64)  # 8 batches per rank -> 4 groups of 2
        # due from the start (lead >= at) but never imminent within the stream
        cp.set_eval_plan(idx, _BS, 1000)
        with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_EVAL_GAP_S": "0.05",
                                            "DL_COLD_PREFETCH_EVAL_LEAD": "100000"}):
            self._go(_Ds(), rank=0, lead=6, group=2, delay=0.01)
        _, ev = self._split()
        times = [t for t, s, _ in self.issued if s and s[0] >= _EVAL_BASE]
        self.assertEqual(self._flat(ev), _eval_sequence(idx, 0))
        self.assertEqual(len(times), 4)
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertTrue(all(g >= 0.045 for g in gaps), gaps)

    def test_imminent_eval_issues_the_rest_at_once(self) -> None:
        # no lead window (EVAL_LEAD=0): nothing until the eval is URGENT=10
        # pulled batches away, then every eval batch at once from the top of a
        # train group, in consecutive calls
        idx = _holdout(64)
        cp.set_eval_plan(idx, _BS, 40)
        with mock.patch.object(cp._EvalWalk, "URGENT", 10), \
                mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_EVAL_LEAD": "0",
                                               "DL_COLD_PREFETCH_EVAL_GAP_S": "100"}):
            seq = self._go(_Ds(), rank=0, lead=6, group=2)
        tr, ev = self._split()
        self.assertEqual(self._flat(ev), _eval_sequence(idx, 0))
        self.assertTrue(all(pos >= 30 for pos, _ in ev), ev)
        self.assertEqual(self._flat(tr), [s for _, _, s in seq])
        # consecutive: no train call between the first and the last eval call
        kinds = [s[0] >= _EVAL_BASE for _, s, _ in self.issued if s]
        first = kinds.index(True)
        last = len(kinds) - 1 - kinds[::-1].index(True)
        self.assertTrue(all(kinds[first:last + 1]), kinds)

    def test_eval_failure_is_isolated(self) -> None:
        # an exception in the eval read-ahead turns it off (one warning); the
        # train walk still covers the whole stream
        log = _Lines()
        cp.logger.addHandler(log)
        self.addCleanup(cp.logger.removeHandler, log)
        level = cp.logger.level
        cp.logger.setLevel(logging.WARNING)
        self.addCleanup(cp.logger.setLevel, level)
        rec = cp._ranges

        def boom(pool, files, samples, *a, **kw):  # noqa
            if len(samples) and int(np.asarray(samples)[0]) >= _EVAL_BASE:
                raise OSError("injected")
            return rec(pool, files, samples, *a, **kw)

        idx = _holdout(37)
        cp.set_eval_plan(idx, _BS, 1)
        with mock.patch.object(cp, "_ranges", boom), \
                mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_EVAL_GAP_S": "0"}):
            seq = self._go(_Ds(), rank=0, lead=6, group=2)
        tr, ev = self._split()
        self.assertEqual(ev, [])
        self.assertEqual(self._flat(tr), [s for _, _, s in seq])
        fails = [m for m in log.lines if "eval holdout walk failed (ignored)" in m]
        self.assertEqual(len(fails), 1, log.lines)
        self.assertFalse(any("[cold-prefetch] failed" in m for m in log.lines), log.lines)
        done = [m for m in log.lines if "[cold-prefetch] stream done" in m]
        self.assertEqual(len(done), 1, log.lines)

    def test_armed_and_done_lines(self) -> None:
        log = _Lines()
        cp.logger.addHandler(log)
        self.addCleanup(cp.logger.removeHandler, log)
        level = cp.logger.level
        cp.logger.setLevel(logging.WARNING)
        self.addCleanup(cp.logger.setLevel, level)
        idx = _holdout(37)
        cp.set_eval_plan(idx, _BS, 30, 2)
        with mock.patch.dict("os.environ", {"DL_COLD_PREFETCH_EVAL_LEAD": "10",
                                            "DL_COLD_PREFETCH_EVAL_GAP_S": "0"}):
            self._go(_Ds(), rank=0, lead=6, group=2)
        armed = [m for m in log.lines if "[cold-prefetch] armed" in m]
        self.assertEqual(len(armed), 1, log.lines)
        self.assertTrue(armed[0].endswith(" eval=on at=30 lead=10 gap_s=0 cap=2"), armed[0])
        done = [m for m in log.lines if "[cold-prefetch] eval holdout walk" in m]
        self.assertEqual(len(done), 1, log.lines)
        self.assertIn(" batches=2 bs=4 ", done[0])
        self.assertIn(" trigger=lead ", done[0])
        self.assertIn(" first_eval_at=30 lead=10 ", done[0])
        stream_done = [m for m in log.lines if "[cold-prefetch] stream done" in m]
        self.assertEqual(len(stream_done), 1, log.lines)
        self.assertIn(" eval_batches=2 eval_walked=2", stream_done[0])


if __name__ == "__main__":
    unittest.main()

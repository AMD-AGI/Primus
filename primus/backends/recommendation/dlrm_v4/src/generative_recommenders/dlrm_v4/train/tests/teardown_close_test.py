#!/usr/bin/env python3
# pyre-unsafe
"""DL_TEARDOWN_CLOSE_POOLS: after RUN_STOP the persistent input pools are
closed explicitly (pool_close.close_pools, _PrefetchingWindowLoader.
teardown_close) before the teardown barrier, instead of being left to
interpreter exit.

Real persistent fork workers on toy datasets (CPU only):

  * the chained single train pool stopped MID-WINDOW, with its pusher thread
    waiting on a full sampler queue (>= 2 windows and >= 2 x depth batches)
    and 48 batches outstanding (6 workers x prefetch 8, the production depth):
    the pusher exits, the workers exit, the references are dropped; also after
    the stream generator was closed first;
  * the two double-buffered train pools while a background _prepare thread is
    still resetting one of them: it is joined first, then both pools close; a
    reset that outlives the bound leaves that pool open (warned);
  * the persistent eval pool (make_persistent_streaming_dataloader) re-armed
    with a prime nobody consumed;
  * a pin-memory thread busy pinning when the close starts. A CPU build has no
    accelerator, so pin_memory=True starts no pin thread; the test emulates one
    by patching torch.accelerator: torch's own _pin_memory_loop runs, and the
    batch's pin_memory() only sleeps. The pin thread must be joined before the
    workers stop, and no pin may start after the close returns;
  * every wait is bounded, failures are logged and never raised, a second
    close is a no-op, rank 0 logs exactly one summary line;
  * the real streaming_train_eval_loop, run on CPU up to its first train
    window, records its persistent train loader, prefetcher and eval pool;
  * the kill switch, and the train_ranker hook position: right after the loop
    returns, inside the try, before the teardown barrier;
  * scripts/launch_slurm.sh forwards DL_TEARDOWN_CLOSE_POOLS and
    PYTHONFAULTHANDLER into the trainer container, each only when set.
"""

import ast
import contextlib
import pathlib
import re
import threading
import time
import unittest
from typing import List, Optional
from unittest import mock

import gin
import numpy as np
import torch
from torch.utils.data import DataLoader

from generative_recommenders.dlrm_v4.train import pool_close as pc
from generative_recommenders.dlrm_v4.train import utils as tu

_LINE = re.compile(r"^\[teardown\] input pools closed: (\d+) in (\d+) ms$")
# Generous bound on one close in these tests (production bound: 5 s per wait).
_BOUND_S = 10.0


class _Windows:
    """Stands in for the yambda dataset: window ts -> global anchor indices.
    ``slow`` = {ts: seconds} makes that window's index scan slow."""

    def __init__(self, sizes: List[int], slow: Optional[dict] = None) -> None:
        self._starts = np.cumsum([0] + list(sizes))
        self._sizes = list(sizes)
        self._slow = dict(slow or {})

    def train_window_indices(self, ts: int) -> np.ndarray:
        if self._slow.get(ts):
            time.sleep(self._slow[ts])
        s = int(self._starts[ts])
        return np.arange(s, s + self._sizes[ts], dtype=np.int64)


class _HoldoutWindows(_Windows):
    """The dataset API streaming_train_eval_loop's setup reads (fixed holdout,
    split contract, per-sample reads), for the real loop's setup on CPU."""

    _train_split_percentage = 0.9
    _split_salt = 0
    is_eval = False

    def num_windows(self) -> int:
        return len(self._sizes) + 2

    def eval_holdout_indices(self, ts: int, n: int) -> np.ndarray:
        return np.arange(1000, 1064, dtype=np.int64)

    def load_query_samples(self, idx) -> None:
        pass

    def unload_query_samples(self, idx) -> None:
        pass

    def get_sample(self, i: int) -> int:
        return int(i)


class _StopAtWindowLoop(Exception):
    """Raised by the train stream: ends the real loop at its first window."""


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, windows: _Windows, sample_s: float = 0.0) -> None:
        self.dataset = windows
        self._sample_s = sample_s

    def __len__(self) -> int:
        return 1 << 30

    def __getitem__(self, idx: int) -> int:
        if self._sample_s:
            time.sleep(self._sample_s)  # a worker busy building a batch
        return int(idx)


def _collate(samples):
    return torch.tensor(samples, dtype=torch.int64)


# Start times of the emulated pins (the pin thread runs in this process).
_PINS: List[float] = []


class _PinBatch:
    """A batch whose pin_memory() stands in for pinning: it only sleeps.
    Module level: the workers pickle it back to the main process."""

    def __init__(self, xs) -> None:
        self.xs = xs

    def pin_memory(self) -> "_PinBatch":
        _PINS.append(time.perf_counter())
        time.sleep(0.05)
        return self


def _pin_collate(samples) -> _PinBatch:
    return _PinBatch(list(samples))


def _dl_factory(dataset, bs: int, nw: int, pf: int):
    def make(sampler):
        if isinstance(sampler, tu._ChainedWindowBatchSampler):
            sampler.set_batch_size(bs)
            return DataLoader(
                dataset, batch_sampler=sampler, collate_fn=_collate,
                num_workers=nw, prefetch_factor=pf, persistent_workers=True,
                multiprocessing_context="fork",
            )
        return DataLoader(
            dataset, batch_size=bs, shuffle=False, collate_fn=_collate,
            drop_last=True, num_workers=nw, prefetch_factor=pf, sampler=sampler,
            persistent_workers=True, multiprocessing_context="fork",
        )

    return make


def _loader(single: bool, sizes, bs, nw, pf, slow=None, sample_s=0.0):
    ds = _Dataset(_Windows(sizes, slow), sample_s)
    with mock.patch.object(tu, "window_single_pool", lambda: single):
        return tu._PrefetchingWindowLoader(
            dataset=ds,
            sampler_factory=lambda: tu.StreamingWindowSampler(0, 1),
            dl_factory=_dl_factory(ds, bs, nw, pf),
        )


def _warm(loader, ts_list):
    """The production chained stream (TRAIN_PIPELINE=1, TP_WARM_WINDOWS=1)."""
    stream = loader.stream(
        ts_list, first_skip_samples=0, warm=True, per_window_cap=None,
        first_window_trained=0,
    )
    return tu._WarmWindowStream(stream)


def _force_close(loader) -> None:
    """Test cleanup only: whatever a failed assertion left running."""
    for th in loader._stream_threads:
        if th is not None:
            th.join(10)
    if loader._stream_stop is not None:
        loader._stream_stop.set()
    for it in [*loader._iters, *(getattr(d, "_iterator", None) for d in loader._dls)]:
        if it is not None and not getattr(it, "_shutdown", True):
            it._shutdown_workers()


class _Log:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, msg: str) -> None:
        self.lines.append(msg)

    def summary(self) -> List[re.Match]:
        return [m for m in map(_LINE.match, self.lines) if m]

    def warnings(self) -> List[str]:
        return [ln for ln in self.lines if not _LINE.match(ln)]


class TeardownCloseTest(unittest.TestCase):
    def assertPoolClosed(self, it) -> None:
        self.assertTrue(it._shutdown, "iterator not shut down")
        alive = [w.pid for w in it._workers if w.is_alive()]
        self.assertEqual(alive, [], f"worker processes still alive: {alive}")

    def assertOneSummary(self, log: _Log, n: int) -> None:
        got = log.summary()
        self.assertEqual(len(got), 1, log.lines)
        self.assertEqual(int(got[0].group(1)), n, log.lines)

    def test_single_pool_mid_window_stop(self) -> None:
        nw, pf, bs = 6, 8, 2
        depth = nw * pf  # 48, the production train pool depth
        loader = _loader(True, [2 * bs * depth] * 8, bs, nw, pf, sample_s=0.005)
        try:
            warm = _warm(loader, list(range(8)))
            ts, nb = warm.advance()
            self.assertEqual((ts, nb), (0, 2 * depth))
            for _ in range(5):  # stop 5 batches into a 96-batch window
                next(warm)
            it, chain = loader._iters[0], loader._chain
            pusher = loader._stream_threads[0]
            # Mid-window stop state: the pusher waits on a full queue and the
            # pool holds the full prefetch depth.
            t0 = time.time()
            while chain.queued_windows() < 2 and time.time() - t0 < 10:
                time.sleep(0.01)
            self.assertTrue(pusher.is_alive())
            self.assertGreaterEqual(chain.queued_windows(), 2)
            self.assertGreaterEqual(chain.queued_batches(), 2 * depth)
            self.assertEqual(it._send_idx - it._rcvd_idx, depth)  # in flight
            self.assertEqual(len(it._workers), nw)
            self.assertTrue(all(w.is_alive() for w in it._workers))

            log = _Log()
            n, dt = pc.close_pools([loader], rank=0, log=log)
            self.assertEqual(n, 1)
            self.assertLess(dt, _BOUND_S)
            self.assertFalse(pusher.is_alive(), "pusher thread still running")
            self.assertPoolClosed(it)
            self.assertIsNone(loader._iters[0])
            self.assertIsNone(loader._dls[0]._iterator)
            self.assertOneSummary(log, 1)
            self.assertEqual(log.warnings(), [])

            # Idempotent; the generator's own finally may still run later.
            log2 = _Log()
            self.assertEqual(pc.close_pools([loader], rank=0, log=log2)[0], 0)
            self.assertOneSummary(log2, 0)
            self.assertEqual(loader.teardown_close(), 0)
            warm._gen.close()
        finally:
            _force_close(loader)

    def test_single_pool_after_generator_closed(self) -> None:
        loader = _loader(True, [64] * 6, 2, 2, 4)
        try:
            warm = _warm(loader, list(range(6)))
            warm.advance()
            next(warm)
            pusher, it = loader._stream_threads[0], loader._iters[0]
            warm._gen.close()  # the stream's finally ran (stop set)
            log = _Log()
            self.assertEqual(pc.close_pools([loader], rank=0, log=log)[0], 1)
            self.assertFalse(pusher.is_alive())
            self.assertPoolClosed(it)
            self.assertOneSummary(log, 1)
        finally:
            _force_close(loader)

    def test_double_buffer_prepare_thread_joined(self) -> None:
        # Window 2's index scan (the background _prepare started when window 1
        # opens) takes 1 s; the close starts while it runs.
        loader = _loader(False, [40] * 4, 2, 2, 2, slow={2: 1.0})
        try:
            warm = _warm(loader, [0, 1, 2, 3])
            _, nb = warm.advance()
            for _ in range(nb):
                next(warm)
            warm.advance()  # opens window 1: starts _prepare(buffer 0, window 2)
            next(warm)
            prep = loader._stream_threads[0]
            self.assertTrue(prep is not None and prep.is_alive())
            its = list(loader._iters)
            log = _Log()
            n, dt = pc.close_pools([loader], rank=0, log=log)
            self.assertEqual(n, 2)
            self.assertLess(dt, _BOUND_S)
            self.assertFalse(prep.is_alive())
            for it in its:
                self.assertPoolClosed(it)
            self.assertEqual(loader._iters, [None, None])
            self.assertEqual([d._iterator for d in loader._dls], [None, None])
            self.assertOneSummary(log, 2)
            self.assertEqual(log.warnings(), [])
        finally:
            _force_close(loader)

    def test_double_buffer_busy_reset_left_open(self) -> None:
        loader = _loader(False, [40] * 4, 2, 2, 2, slow={2: 1.5})
        try:
            warm = _warm(loader, [0, 1, 2, 3])
            _, nb = warm.advance()
            for _ in range(nb):
                next(warm)
            warm.advance()
            next(warm)
            prep = loader._stream_threads[0]
            it1 = loader._iters[1]
            log = _Log()
            t0 = time.perf_counter()
            n, _ = pc.close_pools([loader], rank=0, log=log, timeout_s=0.2)
            self.assertLess(time.perf_counter() - t0, 1.2)
            self.assertEqual(n, 1)  # buffer 1 closed; buffer 0 is mid-reset
            self.assertPoolClosed(it1)
            self.assertTrue(prep.is_alive())
            self.assertIsNotNone(loader._dls[0]._iterator)
            self.assertEqual(len(log.warnings()), 2, log.lines)
            self.assertTrue(all(w.endswith("rank=0") for w in log.warnings()))
            self.assertOneSummary(log, 1)
            # A second call is a no-op, even once that reset has finished.
            prep.join(10)
            self.assertEqual(loader.teardown_close(timeout_s=0.2), 0)
            self.assertFalse(loader._dls[0]._iterator._shutdown)
        finally:
            _force_close(loader)

    def test_eval_pool_with_unconsumed_prime(self) -> None:
        ds = _Dataset(_Windows([0]), sample_s=0.005)
        sampler = tu.StreamingWindowSampler(0, 1)
        with mock.patch.object(tu, "collate_fn", _collate):
            dl = tu.make_persistent_streaming_dataloader(
                dataset=ds, sampler=sampler, batch_size=2, num_workers=2,
                prefetch_factor=4, pin_memory=False,
            )
            sampler.set_window(np.arange(200, dtype=np.int64))
            first = iter(dl)
            self.assertEqual(next(first).tolist(), [0, 1])
            it = iter(dl)  # re-armed for the next eval, never consumed
            self.assertIs(it, first)
            self.assertEqual(it._tasks_outstanding, 8)
            log = _Log()
            pools = [dl, None, dl]
            n, _ = pc.close_pools(pools, rank=0, log=log)
        self.assertEqual(n, 1)
        self.assertEqual(pools, [])  # references dropped
        self.assertIsNone(dl._iterator)
        self.assertPoolClosed(it)
        self.assertOneSummary(log, 1)

    def test_busy_pin_thread_joined_before_workers(self) -> None:
        acc = getattr(torch, "accelerator", None)
        needed = ("is_available", "current_accelerator", "current_device_index",
                  "set_device_index")
        if acc is None or not all(hasattr(acc, a) for a in needed):
            self.skipTest("torch.accelerator API not available for the emulation")
        _PINS.clear()
        nthreads = torch.get_num_threads()  # the pin loop sets 1 (process-wide)
        ds = _Dataset(_Windows([0]))
        order: List[List[bool]] = []
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(acc, "is_available", lambda: True))
            st.enter_context(mock.patch.object(
                acc, "current_accelerator", lambda *a, **k: torch.device("cuda")))
            st.enter_context(mock.patch.object(acc, "current_device_index", lambda: 0))
            st.enter_context(
                mock.patch.object(acc, "set_device_index", lambda *a, **k: None))
            dl = DataLoader(
                ds, batch_size=2, sampler=range(400), collate_fn=_pin_collate,
                num_workers=3, prefetch_factor=4, persistent_workers=True,
                multiprocessing_context="fork", pin_memory=True,
            )
            it = iter(dl)
            try:
                self.assertTrue(it._pin_memory, "pin-thread emulation not active")
                pin_th = it._pin_memory_thread
                self.assertEqual(next(it).xs, [0, 1])
                join = pin_th.join

                def recording_join(*a, **k):
                    order.append([w.is_alive() for w in it._workers])
                    return join(*a, **k)

                pin_th.join = recording_join
                self.assertTrue(pin_th.is_alive())
                log = _Log()
                n, dt = pc.close_pools([dl], rank=0, log=log)
                t_done = time.perf_counter()
                # State right after the close, before the cleanup below.
                pin_alive = pin_th.is_alive()
                shut = it._shutdown
                workers_alive = [w.is_alive() for w in it._workers]
                joins = [list(o) for o in order]
                cached = dl._iterator
            finally:
                if not it._shutdown:
                    it._shutdown_workers()
                torch.set_num_threads(nthreads)
        self.assertEqual(n, 1)
        self.assertLess(dt, _BOUND_S)
        self.assertFalse(pin_alive, "pin-memory thread still running")
        self.assertTrue(shut, "iterator not shut down")
        self.assertEqual(workers_alive, [False] * 3)
        self.assertIsNone(cached)
        # torch's order: the pin thread is joined while every worker still runs.
        self.assertEqual(joins, [[True, True, True]])
        self.assertFalse([t for t in _PINS if t > t_done], "pinned after the close")
        self.assertLess(len(_PINS), 12, "close did not cut the outstanding pins short")
        self.assertOneSummary(log, 1)

    def test_already_shut_pool_not_counted(self) -> None:
        # e.g. the DL_PRESTART eval pool a run does not use: the loop shuts it
        # down itself (utils.py, "pre-started eval pool unused").
        dl = DataLoader(_Dataset(_Windows([0])), batch_size=2, sampler=range(16),
                        collate_fn=_collate, num_workers=2, prefetch_factor=2,
                        persistent_workers=True, multiprocessing_context="fork")
        it = iter(dl)
        it._shutdown_workers()
        log = _Log()
        n, _ = pc.close_pools([dl, it], rank=0, log=log)
        self.assertEqual(n, 0)
        self.assertIsNone(dl._iterator)
        self.assertEqual(log.warnings(), [])
        self.assertOneSummary(log, 0)

    def test_bounded_never_raises(self) -> None:
        release = threading.Event()

        class _Stuck:  # a shutdown that blocks (e.g. a pin thread that never returns)
            _shutdown = False

            def _shutdown_workers(self) -> None:
                release.wait(10)

        class _Raises:
            _shutdown = False

            def _shutdown_workers(self) -> None:
                raise RuntimeError("boom")

        class _BadLoader:
            def teardown_close(self, timeout_s, warn):
                raise ValueError("bad loader")

        ds = _Dataset(_Windows([0]))
        dl = DataLoader(ds, batch_size=2, sampler=range(64), collate_fn=_collate,
                        num_workers=2, prefetch_factor=2, persistent_workers=True,
                        multiprocessing_context="fork")
        it = iter(dl)
        log = _Log()
        try:
            t0 = time.perf_counter()
            n, _ = pc.close_pools(
                [_Stuck(), _Raises(), _BadLoader(), dl], rank=3, log=log, timeout_s=0.3
            )
            elapsed = time.perf_counter() - t0
        finally:
            release.set()
        self.assertLess(elapsed, 3.0)
        self.assertEqual(n, 1)
        self.assertPoolClosed(it)
        self.assertEqual(log.summary(), [])  # not rank 0: no summary line
        warnings = log.warnings()
        self.assertEqual(len(warnings), 3, log.lines)
        self.assertTrue(all(w.endswith("rank=3") for w in warnings))
        self.assertEqual(pc.close_pools(None, rank=0, log=log)[0], 0)

    def test_loop_records_its_pools(self) -> None:
        # The real streaming_train_eval_loop (double-buffered persistent
        # loader, per-window eval), stopped where the train stream starts: it
        # has recorded the persistent train loader, the train prefetcher and
        # the eval pool, whose workers it forked and primed.
        gin.clear_config()
        for k, v in (("batch_size", 2), ("num_workers", 2), ("prefetch_factor", 2),
                     ("pin_memory", False)):
            gin.bind_parameter(f"make_persistent_streaming_dataloader.{k}", v)
        pools: List = []
        try:
            with mock.patch.object(tu, "get_dataset", return_value=(
                    lambda **kw: _HoldoutWindows([16] * 3), {})), \
                    mock.patch.object(tu.torch.distributed, "is_initialized",
                                      return_value=False), \
                    mock.patch.object(tu, "collate_fn", _collate), \
                    mock.patch.object(tu._PrefetchingWindowLoader, "stream",
                                      side_effect=_StopAtWindowLoop):
                with self.assertRaises(_StopAtWindowLoop):
                    tu.streaming_train_eval_loop(
                        rank=0, model=None,
                        optimizer=mock.Mock(param_groups=[{"lr": 0.1}]),
                        metric_logger=mock.Mock(), device=torch.device("cpu"),
                        num_train_ts=3, hstu_config=None, embedding_table_configs={},
                        use_pipeline=False, persistent_loader=True, double_buffer=True,
                        eval_every_n_windows=1, resume_cold_start=True,
                        teardown_pools=pools,
                    )
            self.assertEqual(len(pools), 3, pools)
            persistent_dl, prefetcher, eval_dl = pools
            self.assertIsNone(persistent_dl._iterator)  # never iterated here
            self.assertIsInstance(prefetcher, tu._PrefetchingWindowLoader)
            it = eval_dl._iterator
            self.assertIsNotNone(it)
            log = _Log()
            n, _ = pc.close_pools(pools, rank=0, log=log)
            self.assertEqual(n, 1)
            self.assertPoolClosed(it)
            self.assertOneSummary(log, 1)
        finally:
            for p in pools:
                it = getattr(p, "_iterator", None)
                if it is not None and not it._shutdown:
                    it._shutdown_workers()
            gin.clear_config()

    def test_kill_switch(self) -> None:
        self.assertTrue(pc.enabled({}))
        for v in ("1", "yes", "", "2"):
            self.assertTrue(pc.enabled({pc.ENV: v}), v)
        for v in ("0", "false", "OFF", " no "):
            self.assertFalse(pc.enabled({pc.ENV: v}), v)

    def test_launcher_forwards_switch_and_faulthandler(self) -> None:
        # The launcher passes only listed names into the trainer container:
        # without these lines DL_TEARDOWN_CLOSE_POOLS=0 (the OFF arm) and
        # PYTHONFAULTHANDLER=1 (every thread's Python stack on SIGABRT) would
        # be dropped silently. ${X:+...} forwards a name only when the caller
        # set it, so an unset name changes nothing.
        src = pathlib.Path(__file__).resolve().parents[4] / "scripts" / "launch_slurm.sh"
        text = src.read_text()
        lines = [s.strip() for s in text.splitlines()]
        # The worker `docker exec` that starts the trainer.
        start = lines.index("-e LAUNCH_SLURM_PHASE=worker \\")
        end = next(i for i, s in enumerate(lines)
                   if "LAUNCH_SLURM_PHASE=worker bash scripts/launch_slurm.sh" in s)
        for k in (pc.ENV, "PYTHONFAULTHANDLER"):
            fwd = "${%s:+-e %s=$%s}" % (k, k, k)
            self.assertEqual(text.count(fwd), 1, k)
            at = [i for i, s in enumerate(lines) if s == fwd + " \\"]
            self.assertEqual(len(at), 1, k)
            self.assertTrue(start < at[0] < end, k)

    def test_hook_after_loop_before_teardown_barrier(self) -> None:
        sig = tu.inspect.signature(tu.streaming_train_eval_loop)
        self.assertIsNone(sig.parameters["teardown_pools"].default)
        src = pathlib.Path(__file__).resolve().parents[1] / "train_ranker.py"
        tree = ast.parse(src.read_text())
        main = next(f for f in tree.body if isinstance(f, ast.FunctionDef)
                    and f.name == "_main_func")

        def calls(node, name):
            return [c for c in ast.walk(node) if isinstance(c, ast.Call)
                    and getattr(c.func, "id", getattr(c.func, "attr", None)) == name]

        tries = [t for t in ast.walk(main) if isinstance(t, ast.Try) and t.finalbody]
        hooked = []
        for t in tries:
            body_src = ast.Module(body=t.body, type_ignores=[])
            loop = calls(body_src, "streaming_train_eval_loop")
            if not loop:
                continue
            self.assertIn("teardown_pools", [k.arg for k in loop[0].keywords])
            fin = ast.Module(body=t.finalbody, type_ignores=[])
            self.assertTrue(calls(fin, "barrier") and calls(fin, "cleanup"))
            self.assertEqual(calls(fin, "close_pools"), [])
            for node in ast.walk(body_src):
                stmts = getattr(node, "body", None)
                if not isinstance(stmts, list):
                    continue
                for i, s in enumerate(stmts[:-1]):
                    if isinstance(s, ast.Expr) and s.value is loop[0]:
                        hooked.append(calls(stmts[i + 1], "close_pools"))
        self.assertEqual(len(hooked), 1)
        self.assertEqual(len(hooked[0]), 1, "close_pools must follow the loop call")
        # The list handed to the loop is the one closed.
        passed = next(k.value for k in loop[0].keywords if k.arg == "teardown_pools")
        closed = hooked[0][0].args[0]
        self.assertTrue(isinstance(passed, ast.Name) and isinstance(closed, ast.Name))
        self.assertEqual(passed.id, closed.id)


if __name__ == "__main__":
    unittest.main()

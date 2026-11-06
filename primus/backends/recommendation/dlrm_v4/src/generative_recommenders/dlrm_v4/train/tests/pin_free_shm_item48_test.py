#!/usr/bin/env python3
# pyre-unsafe
"""yambda_5b.gin item (55), $DL_PIN_FREE_SHM default ON: the gin wiring (gin
default 1, env override, kill switch DL_PIN_FREE_SHM=0, installed in the
trainer process before any DataLoader pin thread starts) and the
once-per-process lines with the several pin threads a trainer process runs
(the release itself is covered by pin_free_shm_test).

* the shipped bindings (read from yambda_5b.gin itself): env unset -> on (gin
  default), DL_PIN_FREE_SHM=0 -> off, DL_PIN_FREE_SHM=1 -> on (env); the value
  is read once per process; the setting line (every rank, tagged rank=R,
  once per process; "inert" when no loader pins); Samples.pin_memory follows
  the installed value (a shared batch is released, or left intact) and the
  arming line names the source; without the gin item the env-only code
  default (off) is unchanged; launch_slurm.sh forwards the variable;
* prestart_streaming_loaders (the production pre-RUN_START path) under the
  shipped items (32) + (61) + (55), with item (37)'s single chained train
  pool (the ship-12 default) and with its kill switch (two pools), env unset:
  the switch is on before each pool's DataLoader is built, so before any pin
  thread starts; a pass over the pre-forked eval pool matches the per-sample
  reference; with a GPU (the DataLoader pins only with one) the pool's pin
  thread releases every received batch, 0 skips, 0 errors, one arming line
  naming the gin default;
* 8 threads releasing their first batches at the same instant (as the train
  pools' and the eval pool's pin threads can): the arming line and the
  size-guard WARN print exactly once (decided under the stats lock);
* the release contract (nothing reads a batch's unpinned input after
  Samples.pin_memory): every .pin_memory() call in the non-test code is on a
  freshly built tensor or is Samples.pin_memory's own (a new call site fails
  here until it is checked against the contract).
CPU-capable (synthetic yambda store from fast_batch_test)."""

import ast
import os
import pathlib
import threading
import unittest
from unittest import mock

import gin
import numpy as np
import torch

from generative_recommenders.dlrm_v4.datasets import dataset as dsmod
from generative_recommenders.dlrm_v4.datasets.dataset import collate_fn
from generative_recommenders.dlrm_v4.train import cold_prefetch as cp
from generative_recommenders.dlrm_v4.train import dl_prestart as dlp
from generative_recommenders.dlrm_v4.train import utils as tu
from generative_recommenders.dlrm_v4.train.tests.fast_batch_test import (
    _make_ds,
    _make_store,
)
from generative_recommenders.dlrm_v4.train.tests.pin_free_shm_test import (
    _batch,
    _blocks,
    _delta,
    _pinning,
    _share,
    _SKIPS,
    _snap,
)

_KNOB = "DL_PIN_FREE_SHM"
_GPU = torch.cuda.is_available()
# recommendation/ (this file: recommendation/generative_recommenders/dlrm_v4/train/tests/)
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GIN = _REPO / "generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin"
_LAUNCHER = _REPO / "scripts/launch_slurm.sh"
# the setting line: % (source, inert suffix, rank)
_ON_LINE = (
    "[pin-free-shm] pin threads: release each received batch's shared memory "
    "after pinning (DL_PIN_FREE_SHM=1, %s%s) rank=%d"
)
_INERT = "; inert: no loader pins, DL_PIN_MEMORY=0"
# % (source, rank)
_OFF_LINE = (
    "[pin-free-shm] pin threads: shared memory left to the storage destructor "
    "(DL_PIN_FREE_SHM=0, %s) rank=%d"
)


def _gin_lines(prefixes, n: int) -> str:
    lines = [ln for ln in _GIN.read_text().splitlines() if ln.startswith(prefixes)]
    assert len(lines) == n, lines
    return "\n".join(lines) + "\n"


def _item48_gin() -> str:
    """The item (55) bindings exactly as yambda_5b.gin ships them."""
    return _gin_lines(("pin_free_shm_release.", "dlpfs/"), 3)


def _item37_single_pool_gin() -> str:
    """Item (37)'s single chained train pool, as yambda_5b.gin ships it (the
    ship-12 default; DL_SINGLE_POOL=0 restores the two pools)."""
    return _gin_lines(("window_single_pool.", "dlsp/"), 3)


def _item32_gin() -> str:
    """The item (32) and (61) bindings (fast_batch_pools: eval_pools,
    train_pools) exactly as yambda_5b.gin ships them."""
    return _gin_lines(("fast_batch_pools.", "dlfb/", "dlfbe/"), 6)


def _env(value):
    """os.environ with DL_PIN_FREE_SHM set to ``value`` (None = unset)."""
    env = {k: v for k, v in os.environ.items() if k != _KNOB}
    if value is not None:
        env[_KNOB] = value
    return mock.patch.dict(os.environ, env, clear=True)


class _Gin:
    """Context: ``text`` parsed into an empty gin config (cleared on exit)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __enter__(self) -> None:
        gin.clear_config()
        if self._text:
            with gin.unlock_config():
                gin.parse_config(self._text)

    def __exit__(self, *exc) -> None:
        gin.clear_config()


def _msgs(cm):
    return [r.split(":", 2)[2] for r in cm.output]


class PinFreeShmItem48Test(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        # Module state the gin item and the release path write: fresh per
        # test, restored afterwards (later tests see the code default again).
        self._patches = [
            mock.patch.object(dsmod, "_FREE_SHM", None),
            mock.patch.object(dsmod, "_FREE_SHM_FROM_GIN", False),
            mock.patch.object(dsmod, "_FREE_SHM_ARMED_PID", None),
            mock.patch.object(dsmod, "_FREE_SHM_WARN_SIZE_PID", None),
            mock.patch.object(dsmod, "_FREE_SHM_WARN_OTHER_PID", None),
            mock.patch.object(tu, "_PIN_FREE_SHM_LOGGED", False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()

    def test_gin_resolution(self) -> None:
        """Default ON (gin default), env 0 = off (kill switch), env 1 = on; the
        pin path follows; no gin binding = the env-only code default (off)."""
        cases = (
            # DL_PIN_FREE_SHM env -> (on, source)
            (None, True, "gin default"),
            ("0", False, "env"),
            ("1", True, "env"),
        )
        with _Gin(_item48_gin()):
            for env, on, src in cases:
                with self.subTest(DL_PIN_FREE_SHM=env), _env(env), mock.patch.object(
                    dsmod, "_FREE_SHM", None
                ), mock.patch.object(
                    dsmod, "_FREE_SHM_ARMED_PID", None
                ), mock.patch.object(tu, "_PIN_FREE_SHM_LOGGED", False):
                    tu.pin_free_shm_release()
                    self.assertIs(dsmod.pin_free_shm_enabled(), on)
                    self.assertEqual(dsmod.pin_free_shm_knob_source(), src)
                    # every rank (tagged), once per process
                    with self.assertLogs(tu.logger, level="WARNING") as cm:
                        tu._log_pin_free_shm(3)
                        tu._log_pin_free_shm(0)
                        tu._log_pin_free_shm(3, pinned=False)
                    self.assertEqual(
                        _msgs(cm),
                        [_ON_LINE % (src, "", 3) if on else _OFF_LINE % (src, 3)],
                    )
                    # no loader pins (DL_PIN_MEMORY=0): the setting is inert
                    with mock.patch.object(tu, "_PIN_FREE_SHM_LOGGED", False):
                        with self.assertLogs(tu.logger, level="WARNING") as cm:
                            tu._log_pin_free_shm(5, pinned=False)
                        self.assertEqual(
                            _msgs(cm),
                            [_ON_LINE % (src, _INERT, 5) if on else _OFF_LINE % (src, 5)],
                        )
                    # Samples.pin_memory follows the installed value
                    b = _batch(list(range(0, 48, 3)))
                    ref = _snap(_batch(list(range(0, 48, 3))))
                    src_t = _share(b)
                    n = len(src_t)
                    with self.assertLogs(dsmod.logger, level="WARNING") as cm2:
                        s0 = dsmod.pin_free_shm_stats()
                        with _pinning():
                            p = b.pin_memory()
                        d = _delta(s0, dsmod.pin_free_shm_stats())
                        blocks = [_blocks(t) for t in src_t]
                        dsmod.logger.warning("end")  # assertLogs needs >= 1 record
                    armed = [m for m in _msgs(cm2) if "armed OK: [pin-free-shm]" in m]
                    if on:
                        self.assertEqual((d["batches"], d["freed"]), (1, n))
                        self.assertEqual(blocks, [0] * n)
                        self.assertEqual(len(armed), 1, cm2.output)
                        self.assertIn(
                            f"DL_PIN_FREE_SHM=1 ({src}): freed {n} shared storage(s)",
                            armed[0],
                        )
                    else:
                        self.assertEqual(set(d.values()), {0})
                        self.assertTrue(all(x > 0 for x in blocks), blocks)
                        self.assertEqual(armed, [])
                    for k in _SKIPS:
                        self.assertEqual(d[k], 0, k)
                    self.assertEqual(_snap(p), ref)
                    self.assertEqual(
                        [m for m in _msgs(cm2) if "[pin-free-shm] WARN" in m], []
                    )
                    # read once per process: a later env change does not flip it
                    os.environ[_KNOB] = "0" if on else "1"
                    self.assertIs(dsmod.pin_free_shm_enabled(), on)
        # No gin binding: the env-only code default (off), silent setting line.
        with _Gin(""), _env(None), mock.patch.object(
            dsmod, "_FREE_SHM", None
        ), mock.patch.object(dsmod, "_FREE_SHM_FROM_GIN", False), mock.patch.object(
            tu, "_PIN_FREE_SHM_LOGGED", False
        ):
            tu.pin_free_shm_release()
            self.assertIsNone(dsmod._FREE_SHM)
            self.assertFalse(dsmod.pin_free_shm_enabled())
            self.assertEqual(dsmod.pin_free_shm_knob_source(), "code default")
            with self.assertNoLogs(tu.logger, level="WARNING"):
                tu._log_pin_free_shm(0)
        self.assertIn(
            "${DL_PIN_FREE_SHM:+-e DL_PIN_FREE_SHM=$DL_PIN_FREE_SHM}",
            _LAUNCHER.read_text(),
        )

    def test_prestart_installs_before_pin_threads(self) -> None:
        """prestart_streaming_loaders under the shipped items (32) + (61) +
        (55), env unset, with item (37)'s single chained train pool (the
        ship-12 default) and with DL_SINGLE_POOL=0 (two pools): on before
        every pool's DataLoader is built; eval pass identical to the
        per-sample reference; with a GPU, every received batch of the eval
        pool is released by its pin thread."""
        for single in (True, False):
            with self.subTest(single_pool=single), mock.patch.object(
                dsmod, "_FREE_SHM", None
            ), mock.patch.object(dsmod, "_FREE_SHM_FROM_GIN", False), mock.patch.object(
                dsmod, "_FREE_SHM_ARMED_PID", None
            ), mock.patch.object(tu, "_PIN_FREE_SHM_LOGGED", False):
                self._prestart_case(single)

    def _prestart_case(self, single: bool) -> None:
        store, positions, i2a, i2al = _make_store()
        ds = _make_ds(store, positions, i2a, i2al)
        idx = np.arange(len(positions))[::5][:128].astype(np.int64)
        bs = 32
        with _env(None), mock.patch.object(dsmod, "_FAST_BATCH", False):
            refs = [
                _snap(
                    collate_fn(
                        tu.HammerToTorchDataset(ds).__getitems__(idx[j : j + bs].tolist())
                    )
                )
                for j in range(0, len(idx), bs)
            ]

        def factory(hstu_config, is_inference, embedding_config):
            return ds

        seen = []
        orig = tu.make_persistent_streaming_dataloader

        def recording(*args, **kwargs):
            seen.append(dsmod._FREE_SHM)  # what a pin thread of this pool reads
            return orig(*args, **kwargs)

        level = cp._PRESTART_LEVEL[0]
        bundle = None
        dlp._reset_for_tests()
        tu._FAST_BATCH_POOLS_LOGGED.clear()
        gin_text = _item32_gin() + _item48_gin() + _item37_single_pool_gin()
        env = {k: v for k, v in os.environ.items() if k not in (_KNOB, "DL_SINGLE_POOL")}
        if not single:
            env["DL_SINGLE_POOL"] = "0"
        with _Gin(gin_text), mock.patch.dict(
            os.environ, env, clear=True
        ), mock.patch.object(
            dsmod, "_FAST_BATCH", None
        ), mock.patch.object(dsmod, "_FAST_BATCH_EVAL", None), mock.patch.object(
            dsmod, "_FAST_BATCH_FROM_GIN", {"train": False, "eval": False}
        ), mock.patch.object(
            # item (61)'s hold reason (set with DL_SINGLE_POOL=0), restored on
            # exit like the switches above
            tu, "_FAST_BATCH_TRAIN_HELD", None
        ), mock.patch.object(
            tu, "get_dataset", lambda: (factory, {})
        ), mock.patch.object(
            tu, "make_persistent_streaming_dataloader", recording
        ):
            for p, v in (
                ("batch_size", bs),
                ("num_workers", 2),
                ("prefetch_factor", 2),
                ("pin_memory", _GPU),
            ):
                gin.bind_parameter(f"make_persistent_streaming_dataloader.{p}", v)
            try:
                with self.assertLogs(tu.logger, level="WARNING") as cm:
                    ok = tu.prestart_streaming_loaders(
                        hstu_config=None,
                        embedding_table_configs={},
                        rank=0,
                        world_size=1,
                        enabled=1,
                        persistent_loader=1,
                        double_buffer=1,
                        eval_num_workers=2,
                        eval_every_n_windows=0,
                        eval_every_data_pct=0.001,
                    )
                bundle = tu._take_prestart_bundle()
                self.assertTrue(ok)
                self.assertIs(bundle["prefetcher"]._single, single)
                self.assertTrue(dsmod.pin_free_shm_enabled())
                self.assertEqual(dsmod.pin_free_shm_knob_source(), "gin default")
                ev_dl = bundle["eval_dl"]
                self.assertIs(ev_dl.dataset.fast_batch, True)
                bundle["eval_sampler"].set_window(idx)
                with self.assertLogs(dsmod.logger, level="WARNING") as cm2:
                    s0 = dsmod.pin_free_shm_stats()
                    got = list(iter(ev_dl))
                    d = _delta(s0, dsmod.pin_free_shm_stats())
                    dsmod.logger.warning("end")  # assertLogs needs >= 1 record
                # before any accessor: offsets() adds a lazily computed heap tensor
                pinned = [
                    all(t.is_pinned() for t in dsmod._samples_tensors(g)) for g in got
                ]
            finally:
                cp.set_prestart_level(level)
                if bundle is not None:
                    for dl in [bundle["eval_dl"]] + list(bundle["prefetcher"]._dls):
                        it = getattr(dl, "_iterator", None)
                        if it is not None:
                            it._shutdown_workers()
        # the eval pool + the train pool(s) (one chained pool, or two), each
        # DataLoader built with the switch on
        self.assertEqual(seen, [True] * (2 if single else 3))
        # the setting line: every rank; inert when the DataLoaders do not pin
        self.assertIn(_ON_LINE % ("gin default", "" if _GPU else _INERT, 0), _msgs(cm))
        self.assertEqual(len(got), len(refs))
        self.assertEqual([_snap(g) for g in got], refs)
        armed = [m for m in _msgs(cm2) if "armed OK: [pin-free-shm]" in m]
        self.assertEqual([m for m in _msgs(cm2) if "[pin-free-shm] WARN" in m], [])
        if _GPU:
            self.assertEqual(pinned, [True] * len(refs))
            self.assertEqual(d["batches"], len(refs))
            self.assertGreaterEqual(d["freed"], 6 * len(refs))
            for k in _SKIPS:
                self.assertEqual(d[k], 0, k)
            self.assertEqual(len(armed), 1, cm2.output)
            self.assertIn("armed OK: [pin-free-shm] DL_PIN_FREE_SHM=1 (gin default): freed", armed[0])
        else:  # no GPU: the DataLoader does not pin, no pin thread runs
            self.assertEqual(set(d.values()), {0})
            self.assertEqual(armed, [])

    def test_once_per_process_lines_under_threads(self) -> None:
        """A trainer process runs several pin threads (train pools, eval
        pool): threads releasing their first batches at the same instant print
        the arming line and the size-guard WARN exactly once each. CPU only
        (_release_source_pages on shared copies; heap clones stand in for the
        pinned copy)."""
        n = 8
        jobs = []
        for i in range(n):
            b = _batch(list(range(i, 48 + i, 3)))
            src = _share(b)
            if i % 2:  # every other batch: uih values as a view into a big shared storage
                v = b.uih_features_kjt._values
                big = torch.zeros(1 << 20, dtype=torch.int64)
                big[: v.numel()] = v
                big.share_memory_()
                src = [big[: v.numel()] if t is v else t for t in src]
            jobs.append((src, [t.clone() for t in src]))
        gate = threading.Barrier(n)

        def run(job):
            gate.wait()
            dsmod._release_source_pages(job[0], job[1])

        s0 = dsmod.pin_free_shm_stats()
        with self.assertLogs(dsmod.logger, level="WARNING") as cm, _env("1"):
            # the switch as the pin path sees it when it releases (env 1)
            configured = mock.patch.object(dsmod, "_FREE_SHM", True)
            with configured:
                th = [threading.Thread(target=run, args=(j,)) for j in jobs]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
        d = _delta(s0, dsmod.pin_free_shm_stats())
        msgs = _msgs(cm)
        armed = [m for m in msgs if "armed OK: [pin-free-shm]" in m]
        self.assertEqual(len(armed), 1, msgs)
        self.assertIn("armed OK: [pin-free-shm] DL_PIN_FREE_SHM=1 (env): freed", armed[0])
        self.assertEqual(
            len([m for m in msgs if "[pin-free-shm] WARN: size guard" in m]), 1, msgs
        )
        self.assertEqual(len(msgs), 2, msgs)
        self.assertEqual((d["batches"], d["skip_size"], d["errors"]), (n, n // 2, 0))
        self.assertEqual(d["freed"], sum(len(j[0]) for j in jobs) - n // 2)

    def test_pin_memory_call_sites(self) -> None:
        """The release contract: nothing reads a batch's unpinned input after
        Samples.pin_memory (a released storage reads as zeros). Its only caller
        is the DataLoader pin thread (torch). Every .pin_memory() call in the
        non-test code must be on a freshly built tensor (torch.tensor(...) and
        the like: never a received batch) or be one of Samples.pin_memory's
        own KJT pins in datasets/dataset.py. A new call site fails here until
        it is checked against the contract and listed. A listed local name
        (ship15 M4's `host`) is held to the fresh-tensor rule below."""
        fresh = {"tensor", "as_tensor", "empty", "zeros", "ones", "full", "from_numpy"}
        ds_py = "generative_recommenders/dlrm_v4/datasets/dataset.py"
        # Samples.pin_memory's per-tensor KJT pins (always present) and the
        # one-storage pin's rebuilt KJTs (item (56), _pin_one_storage)
        pins = {
            (ds_py, "self.uih_features_kjt"),
            (ds_py, "self.candidates_features_kjt"),
            (ds_py, "self.merged_sparse_features"),
        }
        own = pins | {(ds_py, "k2")}
        # Checked against the contract (ship15 M4 speed estimator):
        # m4_speed._all_gather_rows pins `host`, a fresh
        # torch.tensor(vals, dtype=torch.float64) of the estimator's own
        # floats (5 at the arming gather, then 11 at each later train-window
        # boundary: at most 88 bytes), once per train window while M4 is on
        # (never per step, never with M4 off): never a received batch. The
        # binding check after the scan keeps `host` a fresh tensor.
        m4_py = "generative_recommenders/dlrm_v4/train/m4_speed.py"
        own = own | {(m4_py, "host")}
        found, unexpected = set(), []
        for f in sorted((_REPO / "generative_recommenders").rglob("*.py")):
            rel = f.relative_to(_REPO).as_posix()
            if "/tests/" in rel:
                continue
            src = f.read_text()
            for node in ast.walk(ast.parse(src)):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "pin_memory"
                ):
                    continue
                recv = node.func.value
                if (
                    isinstance(recv, ast.Call)
                    and isinstance(recv.func, ast.Attribute)
                    and recv.func.attr in fresh
                    and isinstance(recv.func.value, ast.Name)
                    and recv.func.value.id == "torch"
                ):
                    continue
                key = (rel, " ".join(ast.get_source_segment(src, recv).split()))
                found.add(key)
                if key not in own:
                    unexpected.append(f"{rel}:{node.lineno}: {key[1]}.pin_memory()")
        self.assertEqual(unexpected, [])
        self.assertLessEqual(pins, found)
        if (m4_py, "host") not in found:
            return  # no M4 estimator pin in this tree
        # Each function of m4_speed.py that pins `host` takes no `host`
        # parameter and binds it only as `host = torch.<fresh>(...)`.
        tree = ast.parse((_REPO / m4_py).read_text())

        def _host_pins(n):
            return [
                c
                for c in ast.walk(n)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "pin_memory"
                and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "host"
            ]

        def _fresh_call(v):
            return (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Attribute)
                and v.func.attr in fresh
                and isinstance(v.func.value, ast.Name)
                and v.func.value.id == "torch"
            )

        checked = set()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = _host_pins(fn)
            if not calls:
                continue
            a = fn.args
            params = a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]
            self.assertNotIn("host", [p.arg for p in params if p], fn.name)
            stores = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Name)
                and n.id == "host"
                and isinstance(n.ctx, ast.Store)
            ]
            good = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "host"
                and _fresh_call(n.value)
            ]
            self.assertTrue(good, fn.name)
            self.assertEqual(len(stores), len(good), fn.name)
            checked |= {(c.lineno, c.col_offset) for c in calls}
        every = {(c.lineno, c.col_offset) for c in _host_pins(tree)}
        self.assertEqual(checked, every)


if __name__ == "__main__":
    unittest.main()

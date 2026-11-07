#!/usr/bin/env python3
# pyre-unsafe
"""yambda_5b.gin item (56), $DL_FAST_BATCH_ONESTORE default 2: the one-storage
layout of the fast builder's batches, its gin wiring (gin default 2, env
override, kill switch DL_FAST_BATCH_ONESTORE=0, installed in the trainer
process before any DataLoader pool forks or starts its pin thread) and its
reach (the batches the fast builder makes; the per-sample builder's batches are
untouched). The layout, the pin-once path and its item (55) release are
covered by onestore_test.py.

* the shipped bindings (read from yambda_5b.gin itself): env unset -> level 2
  (gin default), DL_FAST_BATCH_ONESTORE=0 -> off, 1, 2 or 7 -> level 2 (env;
  the arming lines show a raw value other than 2); the installed value is
  what build_batch uses (one storage, the UIH/candidate tensors on the halves
  of the merged tensors; or six storages) and does not move when the env
  changes later; the setting line (every rank, tagged rank=R, once per
  process; the fast-builder pools, or "inert"); the "[onestore]" suffix of
  the builder's arming line; without the gin item the env-only code default
  (off) is unchanged; launch_slurm.sh forwards the variable;
* prestart_streaming_loaders (the production pre-RUN_START path) under the
  shipped items (32) + (61) + (55) + (56), with item (37)'s single chained
  train pool (the ship-12 default) and with DL_SINGLE_POOL=0 (two pools), env
  unset and with the kill switch: the level is installed before each pool
  DataLoader is built (so before any worker forks or pin thread starts); the
  setting line lists the train pool among the fast-builder pools with the
  single pool (item (61)) and the eval pool alone with two pools (item
  (61)'s precondition); a pass over the
  pre-forked eval pool receives ONE storage per batch (the forked workers
  built the layout), or six with the kill switch, bit-identical to the
  per-sample reference; with a GPU the pool's pin thread pins every batch as
  one storage and item (55) releases exactly that storage, 0 skips, and the
  arming lines name the gin default;
* fork pools under the shipped items (32) + (61) + (56) and item (37)'s
  binding: the train pool builds one storage per batch by default (item
  (61)) and keeps the per-sample builder (six storages per batch, untouched)
  with DL_FAST_BATCH=0 or DL_SINGLE_POOL=0 (unless DL_FAST_BATCH=1); the eval
  pool rebuilds every batch on each pass (iter() re-arm). Bit-identical
  throughout;
* the host-side aliasing contract (until the H2D copy the UIH/candidate KJTs
  share memory with the merged KJT): no non-test module outside datasets/
  calls an in-place tensor method on, or assigns into, an expression rooted
  at a batch KJT (a pattern scan, the tripwire for future host transforms).
CPU-capable (synthetic yambda store from fast_batch_test)."""

import ast
import os
import pathlib
import unittest
from unittest import mock

import gin
import numpy as np
import torch

from generative_recommenders.dlrm_v4.datasets import dataset as dsmod
from generative_recommenders.dlrm_v4.datasets import yambda as ymod
from generative_recommenders.dlrm_v4.datasets.dataset import collate_fn, Samples
from generative_recommenders.dlrm_v4.train import cold_prefetch as cp
from generative_recommenders.dlrm_v4.train import dl_prestart as dlp
from generative_recommenders.dlrm_v4.train import utils as tu
from generative_recommenders.dlrm_v4.train.tests.fast_batch_test import (
    _cmp_samples,
    _fast,
    _make_ds,
    _make_store,
)
from generative_recommenders.dlrm_v4.train.tests.pin_free_shm_test import (
    _delta,
    _SKIPS,
    _snap,
)

_KNOB = "DL_FAST_BATCH_ONESTORE"
# Every switch of items (32), (37), (55), (56), (61): unset unless a test sets it.
_KNOBS = (_KNOB, "DL_PIN_FREE_SHM", "DL_FAST_BATCH", "DL_FAST_BATCH_EVAL", "DL_SINGLE_POOL")
_GPU = torch.cuda.is_available()
_KJTS = ("uih_features_kjt", "candidates_features_kjt", "merged_sparse_features")
# recommendation/ (this file: recommendation/generative_recommenders/dlrm_v4/train/tests/)
_REPO = pathlib.Path(__file__).resolve().parents[4]
_GIN = _REPO / "generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin"
_LAUNCHER = _REPO / "scripts/launch_slurm.sh"
# the setting line: % (source, pools suffix, rank)
_ON_LINE = (
    "[onestore] fast-builder batches: one storage per batch, UIH/candidate "
    "tensors are views of the merged tensors, pinned with one copy "
    "(DL_FAST_BATCH_ONESTORE=2, %s%s) rank=%d"
)
_EVAL_POOLS = "; fast-builder pools: eval"
# item (61): the single train pool on the fast builder too
_TRAIN_EVAL_POOLS = "; fast-builder pools: train, eval"
_INERT = "; inert: no pool uses the fast builder"
# item (61)'s train-pool lines: % rank
_TRAIN_FAST = (
    "armed OK: [fast-batch] train pools: fast builder (build_batch; "
    "DL_FAST_BATCH=1, gin default) rank=%d"
)
_TRAIN_HELD = (
    "[fast-batch] train pools: per-sample path (DL_FAST_BATCH=0, gin default 1 "
    "held because item (37)'s single train pool is off: window_single_pool() "
    "False, env DL_SINGLE_POOL=0; env DL_FAST_BATCH=1 overrides) rank=%d"
)
# % (source, rank)
_OFF_LINE = (
    "[onestore] fast-builder batches: one storage per tensor "
    "(DL_FAST_BATCH_ONESTORE=0, %s) rank=%d"
)


def _gin_lines(prefixes, n: int) -> str:
    lines = [ln for ln in _GIN.read_text().splitlines() if ln.startswith(prefixes)]
    assert len(lines) == n, lines
    return "\n".join(lines) + "\n"


def _item49_gin() -> str:
    """The item (56) bindings exactly as yambda_5b.gin ships them."""
    return _gin_lines(("fast_batch_onestore.", "dlfbos/"), 3)


def _item48_gin() -> str:
    """The item (55) bindings exactly as yambda_5b.gin ships them."""
    return _gin_lines(("pin_free_shm_release.", "dlpfs/"), 3)


def _item37_single_pool_gin() -> str:
    """Item (37)'s single chained train pool, as yambda_5b.gin ships it (the
    ship-12 default; DL_SINGLE_POOL=0 restores the two pools)."""
    return _gin_lines(("window_single_pool.", "dlsp/"), 3)


def _item32_gin() -> str:
    """The item (32) and (61) bindings exactly as yambda_5b.gin ships them."""
    return _gin_lines(("fast_batch_pools.", "dlfb/", "dlfbe/"), 6)


def _env(**kv):
    """os.environ without the item (32)/(37)/(55)/(56)/(61) switches, plus ``kv``
    (None = unset)."""
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update({k: v for k, v in kv.items() if v is not None})
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


def _tensors(s: Samples):
    return dsmod._samples_tensors(s)


def _storages(s: Samples):
    return {t.untyped_storage().data_ptr() for t in _tensors(s)}


def _aliased(s: Samples) -> bool:
    """The item (56) layout: the UIH/candidate values and lengths sit on the
    two halves of the merged values and lengths, all in one storage."""
    u, c, m = (getattr(s, f) for f in _KJTS)
    mv, ml = m._values, m._lengths
    nu, nl = u._values.numel(), u._lengths.numel()
    return len(_storages(s)) == 1 and (
        u._values.storage_offset(),
        c._values.storage_offset(),
        u._lengths.storage_offset(),
        c._lengths.storage_offset(),
        nu + c._values.numel(),
        nl + c._lengths.numel(),
    ) == (
        mv.storage_offset(),
        mv.storage_offset() + nu,
        ml.storage_offset(),
        ml.storage_offset() + nl,
        mv.numel(),
        ml.numel(),
    )


class OneStoreItem49Test(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.store, cls.positions, cls.i2a, cls.i2al = _make_store()

    def setUp(self) -> None:
        # Module state the gin items and the pin path write: fresh per test,
        # restored afterwards (later tests see the code defaults again).
        self._patches = [
            mock.patch.object(dsmod, "_ONESTORE", None),
            mock.patch.object(dsmod, "_ONESTORE_FROM_GIN", False),
            mock.patch.object(dsmod, "_PIN_ONCE_LOGGED_PID", None),
            mock.patch.object(dsmod, "_FREE_SHM", None),
            mock.patch.object(dsmod, "_FREE_SHM_FROM_GIN", False),
            mock.patch.object(dsmod, "_FREE_SHM_ARMED_PID", None),
            mock.patch.object(dsmod, "_FREE_SHM_ONESTORE_PID", None),
            mock.patch.object(dsmod, "_FREE_SHM_WARN_SIZE_PID", None),
            mock.patch.object(dsmod, "_FREE_SHM_WARN_OTHER_PID", None),
            mock.patch.object(dsmod, "_FAST_BATCH", None),
            mock.patch.object(dsmod, "_FAST_BATCH_EVAL", None),
            mock.patch.object(dsmod, "_FAST_BATCH_FROM_GIN", {"train": False, "eval": False}),
            mock.patch.object(ymod, "_FAST_BATCH_LOGGED_PID", None),
            mock.patch.object(tu, "_ONESTORE_LOGGED", False),
            mock.patch.object(tu, "_PIN_FREE_SHM_LOGGED", False),
            mock.patch.object(tu, "_FAST_BATCH_TRAIN_HELD", None),
        ]
        for p in self._patches:
            p.start()
        tu._FAST_BATCH_POOLS_LOGGED.clear()
        dlp._reset_for_tests()

    def tearDown(self) -> None:
        for p in reversed(self._patches):
            p.stop()
        tu._FAST_BATCH_POOLS_LOGGED.clear()

    def _ds(self):
        return _make_ds(self.store, self.positions, self.i2a, self.i2al)

    def _ref(self, ds, batch) -> Samples:
        """The per-sample builder + collate_fn (fresh: comparing fills lazy caches)."""
        with _env(), mock.patch.object(dsmod, "_FAST_BATCH", False):
            return collate_fn(tu.HammerToTorchDataset(ds).__getitems__(batch))

    def test_gin_resolution(self) -> None:
        """Default level 2 (gin default), env 0 = off (kill switch), env 1, 2 or
        7 = level 2 (a raw value other than 2 shows in the arming lines);
        build_batch follows the installed value; the setting line prints on
        every rank once per process; no gin binding = the env-only code
        default (off)."""
        ds = self._ds()
        batch = sorted(np.random.default_rng(7).choice(len(self.positions), 48, replace=False).tolist())
        cases = (
            # DL_FAST_BATCH_ONESTORE env -> (level, source, source as the arming lines print it)
            (None, 2, "gin default", "gin default"),
            ("0", 0, "env", "env"),
            ("2", 2, "env", "env"),
            ("1", 2, "env", "env: 1"),
            ("7", 2, "env", "env: 7"),
        )
        pools_eval = {
            "train": tu.HammerToTorchDataset(ds),
            "eval": tu.HammerToTorchDataset(ds, fast_batch=True),
        }
        pools_none = {
            "train": tu.HammerToTorchDataset(ds),
            "eval": tu.HammerToTorchDataset(ds, fast_batch=False),
        }
        with _Gin(_item49_gin()):
            for env, level, src, shown in cases:
                with self.subTest(DL_FAST_BATCH_ONESTORE=env), _env(
                    DL_FAST_BATCH_ONESTORE=env
                ), mock.patch.object(dsmod, "_ONESTORE", None), mock.patch.object(
                    tu, "_ONESTORE_LOGGED", False
                ), mock.patch.object(ymod, "_FAST_BATCH_LOGGED_PID", None):
                    tu.fast_batch_onestore()
                    self.assertEqual(dsmod.fast_batch_onestore_level(), level)
                    self.assertIs(dsmod.fast_batch_onestore_enabled(), level > 0)
                    self.assertEqual(dsmod.fast_batch_onestore_knob_source(), src)
                    self.assertEqual(
                        dsmod.fast_batch_onestore_knob(), f"DL_FAST_BATCH_ONESTORE={level} ({shown})"
                    )
                    # every rank (tagged), once per process
                    with self.assertLogs(tu.logger, level="WARNING") as cm:
                        tu._log_fast_batch_onestore(3, pools_eval)
                        tu._log_fast_batch_onestore(0, pools_eval)
                    self.assertEqual(
                        _msgs(cm),
                        [_ON_LINE % (shown, _EVAL_POOLS, 3) if level else _OFF_LINE % (shown, 3)],
                    )
                    # no pool on the fast builder: inert; pools unknown: no suffix
                    for pools, suffix in ((pools_none, _INERT), (None, "")):
                        with mock.patch.object(tu, "_ONESTORE_LOGGED", False), self.assertLogs(
                            tu.logger, level="WARNING"
                        ) as cm:
                            tu._log_fast_batch_onestore(6, pools)
                        self.assertEqual(
                            _msgs(cm),
                            [_ON_LINE % (shown, suffix, 6) if level else _OFF_LINE % (shown, 6)],
                        )
                    # build_batch (the fast builder) follows the installed value;
                    # its arming line carries the "[onestore]" suffix when on
                    ref = self._ref(ds, batch)
                    with _fast(True), self.assertLogs(ymod.logger, level="WARNING") as cm2:
                        got = tu.HammerToTorchDataset(ds).__getitems__(batch)
                    self.assertIsInstance(got, Samples)
                    armed = [m for m in _msgs(cm2) if "armed OK: [fast-batch]" in m]
                    self.assertEqual(len(armed), 1, cm2.output)
                    if level:
                        self.assertTrue(_aliased(got))
                        self.assertIn(
                            f"[onestore] DL_FAST_BATCH_ONESTORE=2 ({shown}): 1 storage (", armed[0]
                        )
                        self.assertIn(
                            "shared=False, uih/cand aliased into merged) per batch", armed[0]
                        )
                    else:
                        self.assertEqual(len(_storages(got)), 6)
                        self.assertNotIn("[onestore]", armed[0])
                    self.assertEqual(_cmp_samples(ref, [collate_fn(got)], f"gin/{env}"), [])
                    # the per-sample builder is untouched by the switch
                    with _fast(False):
                        old = collate_fn(tu.HammerToTorchDataset(ds).__getitems__(batch))
                    self.assertEqual(len(_storages(old)), 6)
                    self.assertEqual(_cmp_samples(self._ref(ds, batch), [old], f"old/{env}"), [])
                    # installed once: a later env change does not move it
                    os.environ[_KNOB] = "0" if level else "2"
                    self.assertEqual(dsmod.fast_batch_onestore_level(), level)
        # No gin binding: the env-only code default (off), silent setting line.
        with _Gin(""), _env(), mock.patch.object(dsmod, "_ONESTORE", None), mock.patch.object(
            dsmod, "_ONESTORE_FROM_GIN", False
        ), mock.patch.object(tu, "_ONESTORE_LOGGED", False):
            tu.fast_batch_onestore()
            self.assertIsNone(dsmod._ONESTORE)
            self.assertEqual(dsmod.fast_batch_onestore_level(), 0)
            self.assertEqual(dsmod.fast_batch_onestore_knob_source(), "code default")
            with self.assertNoLogs(tu.logger, level="WARNING"):
                tu._log_fast_batch_onestore(0)
        if _LAUNCHER.exists():  # not in Primus (primus-cli launches the trainer)
            self.assertIn(
                "${DL_FAST_BATCH_ONESTORE:+-e DL_FAST_BATCH_ONESTORE=$DL_FAST_BATCH_ONESTORE}",
                _LAUNCHER.read_text(),
            )

    def _prestart_eval_pass(self, env_level, single=True):
        """prestart_streaming_loaders under the shipped items (32) + (61) +
        (55) + (56) and item (37)'s single-pool binding, DL_FAST_BATCH_ONESTORE =
        ``env_level`` (None = unset), DL_SINGLE_POOL=0 unless ``single``, then
        one pass over the pre-forked eval pool. Returns the per-sample
        reference snapshots, (level, free-shm) seen when each pool DataLoader
        was built, the setting lines, per received batch (storages, pinned,
        shared) taken before any accessor, the received snapshots, the
        dataset-logger lines of the pass and the pin_free_shm_stats() delta of
        the pass."""
        ds = self._ds()
        idx = np.arange(len(self.positions))[::5][:128].astype(np.int64)
        bs = 32
        refs = [_snap(self._ref(ds, idx[j : j + bs].tolist())) for j in range(0, len(idx), bs)]

        def factory(hstu_config, is_inference, embedding_config):
            return ds

        seen = []
        orig = tu.make_persistent_streaming_dataloader

        def recording(*args, **kwargs):
            # what this pool's forked workers and pin thread read
            seen.append((dsmod._ONESTORE, dsmod._FREE_SHM))
            return orig(*args, **kwargs)

        level = cp._PRESTART_LEVEL[0]
        bundle = None
        with _Gin(
            _item32_gin() + _item48_gin() + _item49_gin() + _item37_single_pool_gin()
        ), _env(
            DL_FAST_BATCH_ONESTORE=env_level, DL_SINGLE_POOL=None if single else "0"
        ), mock.patch.object(tu, "get_dataset", lambda: (factory, {})), mock.patch.object(
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
                ev_dl = bundle["eval_dl"]
                self.assertIs(ev_dl.dataset.fast_batch, True)
                bundle["eval_sampler"].set_window(idx)
                with self.assertLogs(dsmod.logger, level="WARNING") as cm2:
                    s0 = dsmod.pin_free_shm_stats()
                    got = list(iter(ev_dl))
                    d = _delta(s0, dsmod.pin_free_shm_stats())
                    dsmod.logger.warning("end")  # assertLogs needs >= 1 record
                # before any accessor: offsets() adds a lazily computed heap tensor
                info = [
                    (
                        len(_storages(g)),
                        all(t.is_pinned() for t in _tensors(g)),
                        all(t.is_shared() for t in _tensors(g)),
                        _aliased(g),
                    )
                    for g in got
                ]
                snaps = [_snap(g) for g in got]
            finally:
                cp.set_prestart_level(level)
                if bundle is not None:
                    for dl in [bundle["eval_dl"]] + list(bundle["prefetcher"]._dls):
                        it = getattr(dl, "_iterator", None)
                        if it is not None:
                            it._shutdown_workers()
        return refs, seen, _msgs(cm), info, snaps, _msgs(cm2), d

    def test_prestart_gin_default(self) -> None:
        """Env unset, single chained train pool and two pools: level 2
        installed before the pools are built; the forked eval workers build
        one storage per batch; with a GPU the pin thread pins it once and item
        (55) releases that one storage."""
        for single in (True, False):
            with self.subTest(single_pool=single), mock.patch.object(
                dsmod, "_ONESTORE", None
            ), mock.patch.object(dsmod, "_PIN_ONCE_LOGGED_PID", None), mock.patch.object(
                dsmod, "_FREE_SHM", None
            ), mock.patch.object(dsmod, "_FREE_SHM_ARMED_PID", None), mock.patch.object(
                dsmod, "_FREE_SHM_ONESTORE_PID", None
            ), mock.patch.object(tu, "_ONESTORE_LOGGED", False), mock.patch.object(
                tu, "_PIN_FREE_SHM_LOGGED", False
            ), mock.patch.object(ymod, "_FAST_BATCH_LOGGED_PID", None):
                tu._FAST_BATCH_POOLS_LOGGED.clear()
                dlp._reset_for_tests()
                self._check_prestart_gin_default(single)

    def _check_prestart_gin_default(self, single: bool) -> None:
        refs, seen, lines, info, snaps, plines, d = self._prestart_eval_pass(None, single)
        n = len(refs)
        # the eval pool + the train pool(s): one chained pool, or two
        self.assertEqual(seen, [(2, True)] * (2 if single else 3))
        # item (61): the single pool's train batches take the layout too; the
        # two pools keep the per-sample builder (its precondition)
        self.assertIn(
            _ON_LINE % ("gin default", _TRAIN_EVAL_POOLS if single else _EVAL_POOLS, 0), lines
        )
        self.assertIn((_TRAIN_FAST if single else _TRAIN_HELD) % 0, lines)
        self.assertEqual(snaps, refs)
        self.assertEqual([x[3] for x in info], [True] * n)  # the UIH/candidate halves alias merged
        armed_one = [m for m in plines if "armed OK: [onestore]" in m]
        armed_of = [m for m in plines if "armed OK: [onestore-free-shm]" in m]
        armed_fs = [m for m in plines if "armed OK: [pin-free-shm]" in m]
        self.assertEqual([m for m in plines if "[pin-free-shm] WARN" in m], [])
        if _GPU:
            self.assertEqual([x[:3] for x in info], [(1, True, False)] * n)
            self.assertEqual((d["batches"], d["freed"]), (n, n))
            self.assertEqual((d["onestore_batches"], d["onestore_freed"]), (n, n))
            for k in _SKIPS:
                self.assertEqual(d[k], 0, k)
            self.assertEqual(len(armed_one), 1, plines)
            self.assertIn(
                "armed OK: [onestore] DL_FAST_BATCH_ONESTORE=2 (gin default): batch pinned "
                "as one storage (pid ",
                armed_one[0],
            )
            self.assertEqual(len(armed_of), 1, plines)
            self.assertIn(
                "armed OK: [onestore-free-shm] DL_FAST_BATCH_ONESTORE=2 (gin default) "
                "DL_PIN_FREE_SHM=1 (gin default): batch pinned as one storage, then freed "
                "1 shared storage(s), ",
                armed_of[0],
            )
            self.assertIn(", kept 0 (pid ", armed_of[0])
            self.assertEqual(len(armed_fs), 1, plines)
            self.assertIn("DL_PIN_FREE_SHM=1 (gin default): freed 1 shared storage(s)", armed_fs[0])
        else:  # no GPU: the DataLoader does not pin; the received batch is the shared one
            self.assertEqual([x[:3] for x in info], [(1, False, True)] * n)
            self.assertEqual(set(d.values()), {0})
            self.assertEqual(armed_one + armed_of + armed_fs, [])

    def test_prestart_kill_switch(self) -> None:
        """DL_FAST_BATCH_ONESTORE=0: one storage per tensor and the per-tensor
        pin, exactly as before the item; no "[onestore]" arming line."""
        refs, seen, lines, info, snaps, plines, d = self._prestart_eval_pass("0")
        n = len(refs)
        self.assertEqual(seen, [(0, True)] * 2)  # the eval pool + the single train pool
        self.assertIn(_OFF_LINE % ("env", 0), lines)
        self.assertEqual(snaps, refs)
        self.assertEqual([m for m in plines if "[onestore" in m], [])
        self.assertEqual([m for m in plines if "[pin-free-shm] WARN" in m], [])
        if _GPU:
            self.assertEqual([x[:3] for x in info], [(6, True, False)] * n)
            self.assertEqual((d["batches"], d["freed"]), (n, 6 * n))
            self.assertEqual((d["onestore_batches"], d["onestore_freed"]), (0, 0))
            for k in _SKIPS:
                self.assertEqual(d[k], 0, k)
        else:
            self.assertEqual([x[:3] for x in info], [(6, False, True)] * n)
            self.assertEqual(set(d.values()), {0})

    def test_fork_pools_train_and_eval(self) -> None:
        """Fork pools under the shipped items (32) + (61) + (56) and item
        (37)'s binding: the eval pool's batches are one storage on every pass;
        the train pool's are one storage by default (item (61)) and six (the
        per-sample builder) with DL_FAST_BATCH=0, or with DL_SINGLE_POOL=0
        (item (61)'s precondition) unless DL_FAST_BATCH=1 is set.
        Bit-identical."""
        ds = self._ds()
        idx = np.arange(len(self.positions))[::5][:192].astype(np.int64)
        bs, nw = 32, 2
        refs = [_snap(self._ref(ds, idx[j : j + bs].tolist())) for j in range(0, len(idx), bs)]
        cases = (
            # (DL_FAST_BATCH, DL_SINGLE_POOL) -> storages per train batch
            ((None, None), 1),
            (("1", None), 1),
            (("0", None), 6),
            ((None, "0"), 6),
            (("1", "0"), 1),
        )
        for (train_env, single_env), want_train in cases:
            with self.subTest(DL_FAST_BATCH=train_env, DL_SINGLE_POOL=single_env), _Gin(
                _item32_gin() + _item49_gin() + _item37_single_pool_gin()
            ), _env(DL_FAST_BATCH=train_env, DL_SINGLE_POOL=single_env), mock.patch.object(
                dsmod, "_FAST_BATCH", None
            ), mock.patch.object(dsmod, "_FAST_BATCH_EVAL", None), mock.patch.object(
                dsmod, "_ONESTORE", None
            ):
                dlp._reset_for_tests()
                tu.fast_batch_pools()
                tu.fast_batch_onestore()
                self.assertEqual(dsmod.fast_batch_onestore_level(), 2)
                base = tu.HammerToTorchDataset(ds)
                train = tu._ReadProbedDataset(base, dlp.new_read_slot("train"))
                ev = tu._eval_dataset(base, dlp.new_read_slot("eval"))
                self.assertIs(ev.fast_batch, True)

                def mk(w, s):
                    return tu.make_persistent_streaming_dataloader(
                        dataset=w, sampler=s, batch_size=bs, num_workers=nw,
                        prefetch_factor=2, pin_memory=False,
                    )

                s_tr, s_ev = tu.StreamingWindowSampler(0, 1), tu.StreamingWindowSampler(0, 1)
                dl_tr, dl_ev = mk(train, s_tr), mk(ev, s_ev)
                s_tr.set_window(idx)
                s_ev.set_window(idx)
                try:
                    got_tr = list(iter(dl_tr))
                    passes = [list(iter(dl_ev)) for _ in range(2)]
                finally:
                    for dl in (dl_tr, dl_ev):
                        it = getattr(dl, "_iterator", None)
                        if it is not None:
                            it._shutdown_workers()
                # layout first (the snapshots run the lazy accessors)
                n_tr = [len(_storages(b)) for b in got_tr]
                n_ev = [len(_storages(b)) for p in passes for b in p]
                al_ev = [_aliased(b) for p in passes for b in p]
                al_tr = [_aliased(b) for b in got_tr]
                self.assertEqual(n_tr, [want_train] * len(refs))
                self.assertEqual(al_tr, [want_train == 1] * len(refs))
                self.assertEqual(n_ev, [1] * (2 * len(refs)))
                self.assertEqual(al_ev, [True] * (2 * len(refs)))
                self.assertTrue(all(t.is_shared() for b in got_tr for t in _tensors(b)))
                self.assertEqual([_snap(b) for b in got_tr], refs)
                for p in passes:
                    self.assertEqual([_snap(b) for b in p], refs)

    def test_no_host_writes_to_batch_kjts(self) -> None:
        """Host-side aliasing contract of the one-storage layout: until the
        H2D copy, the UIH/candidate KJTs' values and lengths are views of the
        merged KJT's, so a host-side in-place write to one would change the
        other. No non-test module outside datasets/ (the builder fills its own
        fresh storage there) calls an in-place tensor method (name ending in
        "_") on, or assigns into, an expression that names a batch KJT. A
        pattern scan: it does not see writes through a local alias, but it
        trips on the direct forms a future host-side transform would add.
        Device copies are independent (KeyedJaggedTensor.to copies each
        tensor), so a hit on device-only code is to be checked and listed."""
        names = ("uih_features_kjt", "candidates_features_kjt", "merged_sparse_features")
        allowed = set()  # (path, line text) of checked device-only hits; none today
        hits = []
        for f in sorted((_REPO / "generative_recommenders").rglob("*.py")):
            rel = f.relative_to(_REPO).as_posix()
            if "/tests/" in rel or "/datasets/" in rel:
                continue
            src = f.read_text()
            for node in ast.walk(ast.parse(src)):
                expr = None
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr.endswith("_")
                    and not node.func.attr.startswith("_")
                ):
                    expr = node.func.value
                elif isinstance(node, (ast.Assign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    expr = next((t for t in targets if isinstance(t, ast.Subscript)), None)
                if expr is None:
                    continue
                txt = " ".join((ast.get_source_segment(src, expr) or "").split())
                if any(n in txt for n in names) and (rel, txt) not in allowed:
                    hits.append(f"{rel}:{node.lineno}: {txt}")
        self.assertEqual(hits, [])

    def test_pin_once_size_guard(self) -> None:
        """_pin_one_storage copies a storage whole only when the batch's
        tensors cover it up to the alignment padding (_ONESTORE_SLACK). A
        one-storage batch whose views sit in a larger storage falls back
        (returns False, batch untouched) without a pinned allocation, so this
        part runs on CPU; with a GPU, Samples.pin_memory then pins it per
        tensor (six pinned storages, same values)."""
        ds = self._ds()
        batch = sorted(np.random.default_rng(11).choice(len(self.positions), 40, replace=False).tolist())
        with _env(), mock.patch.object(dsmod, "_ONESTORE", 2), _fast(True):
            src = tu.HammerToTorchDataset(ds).__getitems__(batch)
        self.assertTrue(_aliased(src))
        ts = _tensors(src)
        st = ts[0].untyped_storage()
        spans = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in ts]
        self.assertLessEqual(st.nbytes(), dsmod._covered_bytes(spans) + dsmod._ONESTORE_SLACK)
        # the same bytes at the same offsets of a larger storage
        big = torch.UntypedStorage(st.nbytes() + dsmod._ONESTORE_SLACK + 4096)
        u8 = torch.empty(0, dtype=torch.uint8)
        u8.set_(big)[: st.nbytes()].copy_(torch.empty(0, dtype=torch.uint8).set_(st))
        for t in ts:
            t.set_(big, t.storage_offset(), t.size(), t.stride())
        before = [id(getattr(src, f)) for f in _KJTS]
        self.assertEqual(len(_storages(src)), 1)
        self.assertFalse(dsmod._pin_one_storage(src))
        self.assertEqual([id(getattr(src, f)) for f in _KJTS], before)
        self.assertTrue(_aliased(src))
        if _GPU:
            ref = self._ref(ds, batch)
            with mock.patch.object(dsmod, "_ONESTORE", 2), mock.patch.object(
                dsmod, "_FREE_SHM", False
            ):
                p = src.pin_memory()
            self.assertEqual(len(_storages(p)), 6)
            self.assertTrue(all(t.is_pinned() for t in _tensors(p)))
            self.assertEqual(_cmp_samples(ref, [p], "size-guard"), [])


if __name__ == "__main__":
    unittest.main()

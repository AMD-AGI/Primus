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
"""$DL_PIN_FREE_SHM: after Samples.pin_memory has made the pinned copy, the
shared-memory pages of the batch it was given are released (madvise
MADV_REMOVE) -- and nothing else is.

Runs on CPU. With a GPU the real pin path runs; without one,
``Tensor.pin_memory`` is replaced by a copy into fresh heap memory (what the
release logic sees from a real pin: a new, unshared storage), and the
DataLoader test applies the pin thread's ``pin_memory()`` in the consumer.

* the knob defaults to off, parses like an integer, is read once per process;
* a batch whose KJT tensors live in shared memory (as received from a worker):
  pinned values are identical to an independently collated reference, every
  source storage is released (its shm file has no blocks left, reads give
  zeros), one "armed OK" line per process; knob off: nothing is touched;
* size guard: a batch tensor that is a view into a larger shared storage
  (> 2 x its bytes + 64 KiB) leaves that storage intact (content and blocks),
  one WARN line per process with counts; exact threshold; overlapping views
  count once;
* a pin that returns the shared input itself (alias) never releases it; an
  already-pinned input (GPU) is never released;
* a file mapping (torch.from_file(shared=True)) is never released (and
  madvise would punch a hole in the file: the guard is needed);
* unshared (heap) input and the file_system strategy: nothing is released;
  an error in the release path leaves a complete pinned batch;
* a torch DataLoader with 2 fork workers + pin_memory and the production
  collate_fn yields identical batches with the knob 0 and 1; with 1 every
  received storage is released.
"""

import contextlib
import os
import shutil
import tempfile
import unittest
from typing import List, Tuple
from unittest import mock

import numpy as np
import torch
from generative_recommenders.dlrm_v4.datasets import dataset as dsmod
from generative_recommenders.dlrm_v4.datasets.dataset import collate_fn, Samples
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory as _dl_pin_memory
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_KJTS = ("uih_features_kjt", "candidates_features_kjt", "merged_sparse_features")
_UIH_KEYS = ["uih_ctx", "uih_item", "uih_ts"]
_CAND_KEYS = ["cand_item", "cand_ts"]
_GPU = torch.cuda.is_available()
_SHM_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else None
_SKIPS = ("skip_size", "skip_alias", "skip_file", "madvise_err", "errors")


def _sample(i: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    """One deterministic (uih, candidates) sample, like a dataset's get_sample."""
    g = torch.Generator().manual_seed(1009 + i)
    n = int(torch.randint(1, 700, (1,), generator=g))
    c = int(torch.randint(1, 4, (1,), generator=g))
    ul = torch.tensor([1, n, n], dtype=torch.int64)
    cl = torch.tensor([c, c], dtype=torch.int64)
    uv = torch.randint(1, 1 << 40, (int(ul.sum()),), generator=g, dtype=torch.int64)
    cv = torch.randint(1, 1 << 40, (int(cl.sum()),), generator=g, dtype=torch.int64)
    return (
        KeyedJaggedTensor(keys=_UIH_KEYS, values=uv, lengths=ul),
        KeyedJaggedTensor(keys=_CAND_KEYS, values=cv, lengths=cl),
    )


def _batch(idx: List[int]) -> Samples:
    return collate_fn([_sample(i) for i in idx])


class _SynthDS(torch.utils.data.Dataset):
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        return _sample(int(i))


def _fake_pin(t, *args, **kwargs):  # CPU stand-in for Tensor.pin_memory
    return t.clone()


def _pinning():
    """Real pin with a GPU, heap-copy stand-in without one."""
    if _GPU:
        return contextlib.nullcontext()
    return mock.patch.object(torch.Tensor, "pin_memory", _fake_pin)


def _knob(on: bool):
    return mock.patch.object(dsmod, "_FREE_SHM", bool(on))


@contextlib.contextmanager
def _fresh_log_flags():
    """The once-per-process log lines print again inside this context."""
    with mock.patch.object(dsmod, "_FREE_SHM_ARMED_PID", None), mock.patch.object(
        dsmod, "_FREE_SHM_WARN_SIZE_PID", None
    ), mock.patch.object(dsmod, "_FREE_SHM_WARN_OTHER_PID", None):
        yield


def _share(s: Samples) -> List[torch.Tensor]:
    """Move every KJT tensor of ``s`` to shared memory (file_descriptor: one
    unlinked shm file per storage), like a batch received from a worker."""
    ts = dsmod._samples_tensors(s)
    for t in ts:
        t.share_memory_()
        assert t.is_shared()
    return ts


def _blocks(t: torch.Tensor) -> int:
    """Allocated 512-byte blocks of the shm file behind a shared storage."""
    fd, _ = t.untyped_storage()._share_fd_cpu_()  # existing fd, no side effect
    return os.fstat(fd).st_blocks


def _snap(s: Samples):
    out = []
    for f in _KJTS:
        k = getattr(s, f)
        row = [f, list(k.keys()), k.stride()]
        for t in (k.values(), k.lengths(), k.offsets()):
            t = t.contiguous().cpu()
            row += [str(t.dtype), tuple(t.shape), t.numpy().tobytes()]
        w = k.weights_or_none()
        row.append(None if w is None else w.contiguous().cpu().numpy().tobytes())
        row.append([int(x) for x in k.length_per_key()])
        row.append([int(x) for x in k.offset_per_key()])
        out.append(row)
    return out


def _delta(before, after):
    return {k: after[k] - before[k] for k in after}


def _lines(cm, text: str) -> List[str]:
    return [r for r in cm.output if text in r]


class PinFreeShmTest(unittest.TestCase):
    maxDiff = None
    IDX = list(range(0, 48, 3))

    def test_knob_parsing_default_off(self) -> None:
        cases = (
            (None, False),
            ("", False),
            ("0", False),
            ("x", False),
            ("-1", False),
            ("1", True),
            ("2", True),
            (" 1 ", True),
        )
        for env, want in cases:
            with mock.patch.object(dsmod, "_FREE_SHM", None), mock.patch.dict(
                os.environ, {}, clear=False
            ):
                os.environ.pop("DL_PIN_FREE_SHM", None)
                if env is not None:
                    os.environ["DL_PIN_FREE_SHM"] = env
                self.assertEqual(dsmod.pin_free_shm_enabled(), want, repr(env))
                os.environ["DL_PIN_FREE_SHM"] = "0" if want else "1"
                self.assertEqual(
                    dsmod.pin_free_shm_enabled(), want, "read once per process"
                )

    def test_covered_bytes(self) -> None:
        cb = dsmod._covered_bytes
        self.assertEqual(cb([]), 0)
        self.assertEqual(cb([(0, 10)]), 10)
        self.assertEqual(cb([(0, 10), (5, 12), (20, 25)]), 17)
        self.assertEqual(cb([(20, 25), (0, 10), (10, 12)]), 17)
        self.assertEqual(cb([(0, 100), (10, 20), (30, 40)]), 100)

    def _release_case(self, on: bool) -> None:
        ref = _snap(_batch(self.IDX))
        b = _batch(self.IDX)
        src = _share(b)
        ptrs = {t.untyped_storage().data_ptr() for t in src}
        self.assertEqual(len(ptrs), len(src))  # one storage per tensor
        self.assertGreaterEqual(len(src), 6)  # values + lengths of 3 KJTs
        blocks = [_blocks(t) for t in src]
        self.assertTrue(all(x > 0 for x in blocks))
        before = [t.clone() for t in src]
        self.assertTrue(all(int(torch.count_nonzero(t)) > 0 for t in before))
        s0 = dsmod.pin_free_shm_stats()
        with _knob(on), _pinning():
            p = b.pin_memory()
        d = _delta(s0, dsmod.pin_free_shm_stats())
        self.assertIs(p, b)
        for t in dsmod._samples_tensors(p):
            self.assertFalse(t.is_shared())
            if _GPU:
                self.assertTrue(t.is_pinned())
        if on:
            # released: no blocks left (checked before any read faults pages in)
            self.assertEqual([_blocks(t) for t in src], [0] * len(src))
            self.assertTrue(all(int(torch.count_nonzero(t)) == 0 for t in src))
            self.assertEqual(d["batches"], 1)
            self.assertEqual(d["freed"], len(src))
            self.assertEqual(
                d["freed_bytes"], sum(t.untyped_storage().nbytes() for t in src)
            )
        else:
            self.assertEqual([_blocks(t) for t in src], blocks)
            self.assertTrue(all(torch.equal(t, c) for t, c in zip(src, before)))
            self.assertEqual(set(d.values()), {0})
        for k in _SKIPS:
            self.assertEqual(d[k], 0, k)
        self.assertEqual(_snap(p), ref)

    def test_release_after_pin(self) -> None:
        with _fresh_log_flags(), self.assertLogs(dsmod.logger, "WARNING") as cm:
            self._release_case(True)
            self._release_case(True)
            self._release_case(False)
            dsmod.logger.warning("end")  # assertLogs needs >= 1 record
        self.assertEqual(len(_lines(cm, "armed OK: [pin-free-shm]")), 1, cm.output)
        self.assertEqual(_lines(cm, "[pin-free-shm] WARN"), [])

    def test_size_guard_keeps_view_into_big_storage(self) -> None:
        ref = _snap(_batch(self.IDX))
        with _fresh_log_flags(), self.assertLogs(dsmod.logger, "WARNING") as cm:
            for _ in range(2):  # the WARN line is printed once per process
                b = _batch(self.IDX)
                _share(b)
                # uih values -> a view at element 4096 of a shared 8 MiB storage
                # holding the same values (the rest: a pattern)
                k = b.uih_features_kjt
                v = k._values
                big = torch.arange(1 << 20, dtype=torch.int64).mul_(7).add_(3)
                big[4096 : 4096 + v.numel()] = v
                big.share_memory_()
                k._values = big[4096 : 4096 + v.numel()]
                want = big.clone()
                bblocks = _blocks(big)
                bptr = big.untyped_storage().data_ptr()
                others = [
                    t
                    for t in dsmod._samples_tensors(b)
                    if t.untyped_storage().data_ptr() != bptr
                ]
                s0 = dsmod.pin_free_shm_stats()
                with _knob(True), _pinning():
                    p = b.pin_memory()
                d = _delta(s0, dsmod.pin_free_shm_stats())
                self.assertEqual(_blocks(big), bblocks)
                self.assertTrue(torch.equal(big, want))  # intact
                self.assertEqual([_blocks(t) for t in others], [0] * len(others))
                self.assertEqual(d["skip_size"], 1)
                self.assertEqual(d["skip_size_bytes"], 8 << 20)
                self.assertEqual(d["freed"], len(others))
                self.assertEqual(_snap(p), ref)
        warns = _lines(cm, "[pin-free-shm] WARN: size guard")
        self.assertEqual(len(warns), 1, cm.output)
        self.assertIn("kept 1 shared storage(s) of 8388608 B", warns[0])
        self.assertEqual(len(_lines(cm, "armed OK: [pin-free-shm]")), 1)

    def test_size_guard_threshold(self) -> None:
        own = 8192  # int64 elements = 64 KiB
        slack = dsmod._FREE_SHM_SLACK
        lim = (2 * own * 8 + slack) // 8  # largest storage (elements) released
        for elems, freed in ((lim, 1), (lim + 1, 0), (own, 1)):
            big = torch.ones(elems, dtype=torch.int64).share_memory_()
            view = big[:own]
            r = dsmod._free_shared_pages([view], [view.clone()])
            self.assertEqual((r["freed"], r["skip_size"]), (freed, 1 - freed), elems)
            self.assertEqual(int(big.sum()), 0 if freed else elems)
        # overlapping views count once: 2 x the same 64 KiB is still 64 KiB
        big = torch.ones(lim + 1, dtype=torch.int64).share_memory_()
        r = dsmod._free_shared_pages([big[:own], big[:own]], [big[:own].clone()])
        self.assertEqual((r["freed"], r["skip_size"]), (0, 1))
        self.assertEqual(int(big.sum()), lim + 1)
        # two disjoint views covering 2 x 64 KiB: released
        r = dsmod._free_shared_pages(
            [big[:own], big[own : 2 * own]], [big[:own].clone()]
        )
        self.assertEqual((r["freed"], r["skip_size"]), (1, 0))

    def test_alias_never_released(self) -> None:
        b = _batch(self.IDX)
        src = _share(b)
        before = [t.clone() for t in src]
        blocks = [_blocks(t) for t in src]
        identity = mock.patch.object(
            torch.Tensor, "pin_memory", lambda t, *a, **k: t
        )
        with _fresh_log_flags(), self.assertLogs(dsmod.logger, "WARNING") as cm:
            s0 = dsmod.pin_free_shm_stats()
            with _knob(True), identity:
                p = b.pin_memory()
            d = _delta(s0, dsmod.pin_free_shm_stats())
        self.assertEqual(d["freed"], 0)
        self.assertEqual(d["skip_alias"], len(src))
        self.assertEqual([_blocks(t) for t in src], blocks)
        self.assertTrue(all(torch.equal(t, c) for t, c in zip(src, before)))
        self.assertTrue(all(t.is_shared() for t in dsmod._samples_tensors(p)))
        self.assertEqual(len(_lines(cm, "[pin-free-shm] WARN: kept")), 1, cm.output)
        self.assertEqual(_lines(cm, "armed OK: [pin-free-shm]"), [])

    @unittest.skipUnless(_GPU, "an already-pinned input needs a GPU")
    def test_already_pinned_never_released(self) -> None:
        ref = _snap(_batch(self.IDX))
        b = _batch(self.IDX)
        _share(b)
        with _knob(False):
            p = b.pin_memory()
        pinned = dsmod._samples_tensors(p)
        self.assertTrue(all(t.is_pinned() for t in pinned))
        keep = [t.clone() for t in pinned]
        s0 = dsmod.pin_free_shm_stats()
        with _knob(True):
            p2 = p.pin_memory()
        d = _delta(s0, dsmod.pin_free_shm_stats())
        self.assertEqual(d["freed"], 0)
        self.assertTrue(all(torch.equal(t, c) for t, c in zip(pinned, keep)))
        self.assertTrue(
            all(x is y for x, y in zip(dsmod._samples_tensors(p2), pinned))
        )
        self.assertEqual(_snap(p2), ref)

    @unittest.skipUnless(_SHM_DIR, "needs /dev/shm (tmpfs supports hole punching)")
    def test_file_mapping_never_released(self) -> None:
        tmp = tempfile.mkdtemp(dir=_SHM_DIR, prefix="pin_free_shm_test_")
        try:
            b = _batch(self.IDX)
            _share(b)
            k = b.uih_features_kjt
            vals = k._values.clone().numpy()
            path = os.path.join(tmp, "uih_values.bin")
            vals.tofile(path)
            ft = torch.from_file(path, shared=True, size=vals.size, dtype=torch.int64)
            st = ft.untyped_storage()
            # it would pass every other guard
            self.assertTrue(st.is_shared())
            self.assertEqual(st.data_ptr() % dsmod._PAGE, 0)
            self.assertEqual(st.nbytes(), vals.nbytes)
            self.assertEqual(st.filename, path)
            k._values = ft
            n_src = len(dsmod._samples_tensors(b))
            blocks = os.stat(path).st_blocks
            s0 = dsmod.pin_free_shm_stats()
            with _knob(True), _pinning():
                b.pin_memory()
            d = _delta(s0, dsmod.pin_free_shm_stats())
            self.assertEqual(d["skip_file"], 1)
            self.assertEqual(d["freed"], n_src - 1)
            self.assertEqual(os.stat(path).st_blocks, blocks)
            self.assertTrue(np.array_equal(np.fromfile(path, dtype=np.int64), vals))
            # the hazard the guard prevents: MADV_REMOVE on a file mapping
            # punches a hole in the FILE
            path2 = os.path.join(tmp, "victim.bin")
            vals.tofile(path2)
            f2 = torch.from_file(path2, shared=True, size=vals.size, dtype=torch.int64)
            n = -(-f2.untyped_storage().nbytes() // dsmod._PAGE) * dsmod._PAGE
            self.assertEqual(dsmod._madvise_remove(f2.data_ptr(), n), 0)
            on_disk = np.fromfile(path2, dtype=np.int64)
            self.assertEqual(int(np.count_nonzero(on_disk)), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_heap_input_and_file_system_strategy(self) -> None:
        b = _batch(self.IDX)  # heap tensors: not shared
        src = dsmod._samples_tensors(b)
        before = [t.clone() for t in src]
        s0 = dsmod.pin_free_shm_stats()
        with _knob(True), _pinning():
            b.pin_memory()
        d = _delta(s0, dsmod.pin_free_shm_stats())
        self.assertEqual((d["batches"], d["freed"]), (1, 0))
        self.assertTrue(all(torch.equal(t, c) for t, c in zip(src, before)))
        old = torch.multiprocessing.get_sharing_strategy()
        self.assertEqual(old, "file_descriptor")
        try:
            b = _batch(self.IDX)
            src = _share(b)  # fd-backed shm, but the process strategy differs
            blocks = [_blocks(t) for t in src]
            torch.multiprocessing.set_sharing_strategy("file_system")
            s0 = dsmod.pin_free_shm_stats()
            with _knob(True), _pinning():
                b.pin_memory()
            d = _delta(s0, dsmod.pin_free_shm_stats())
        finally:
            torch.multiprocessing.set_sharing_strategy(old)
        self.assertEqual(d["freed"], 0)
        self.assertEqual([_blocks(t) for t in src], blocks)

    def test_release_error_is_contained(self) -> None:
        ref = _snap(_batch(self.IDX))
        b = _batch(self.IDX)
        _share(b)

        def boom(src, pinned):
            raise RuntimeError("synthetic")

        with _fresh_log_flags(), self.assertLogs(dsmod.logger, "WARNING") as cm:
            s0 = dsmod.pin_free_shm_stats()
            with _knob(True), _pinning(), mock.patch.object(
                dsmod, "_free_shared_pages", boom
            ):
                p = b.pin_memory()
            d = _delta(s0, dsmod.pin_free_shm_stats())
        self.assertEqual((d["batches"], d["errors"], d["freed"]), (1, 1, 0))
        self.assertEqual(len(_lines(cm, "release failed")), 1, cm.output)
        self.assertEqual(_snap(p), ref)

    def _loader_batches(self, on: bool, n: int, bs: int):
        dl = DataLoader(
            _SynthDS(n),
            batch_size=bs,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=2,
            prefetch_factor=2,
            multiprocessing_context="fork",
            pin_memory=_GPU,
        )
        out, nstor = [], []
        with _knob(on), _pinning():
            for b in dl:
                if not _GPU:  # what the pin thread does
                    src = dsmod._samples_tensors(b)
                    self.assertTrue(all(t.is_shared() for t in src))
                    nstor.append(len({t.untyped_storage().data_ptr() for t in src}))
                    b = _dl_pin_memory(b)
                ts = dsmod._samples_tensors(b)
                if _GPU:
                    self.assertTrue(all(t.is_pinned() for t in ts))
                    nstor.append(len(ts))
                out.append(_snap(b))
        return out, nstor

    def test_dataloader_fork_workers_identical(self) -> None:
        n, bs = 160, 16
        ref = [_snap(_batch(list(range(i, i + bs)))) for i in range(0, n, bs)]
        with _fresh_log_flags(), self.assertLogs(dsmod.logger, "WARNING") as cm:
            s0 = dsmod.pin_free_shm_stats()
            off, _ = self._loader_batches(False, n, bs)
            d_off = _delta(s0, dsmod.pin_free_shm_stats())
            s1 = dsmod.pin_free_shm_stats()
            on, nstor = self._loader_batches(True, n, bs)
            d_on = _delta(s1, dsmod.pin_free_shm_stats())
        self.assertEqual(len(ref), n // bs)
        self.assertEqual(off, ref)
        self.assertEqual(on, ref)
        self.assertEqual(set(d_off.values()), {0})
        self.assertEqual(d_on["batches"], n // bs)
        self.assertGreaterEqual(min(nstor), 6)  # values + lengths of 3 KJTs
        self.assertEqual(d_on["freed"], sum(nstor))
        for k in _SKIPS:
            self.assertEqual(d_on[k], 0, k)
        self.assertEqual(len(_lines(cm, "armed OK: [pin-free-shm]")), 1, cm.output)


if __name__ == "__main__":
    unittest.main()

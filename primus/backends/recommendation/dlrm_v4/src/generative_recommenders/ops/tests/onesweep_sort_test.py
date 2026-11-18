"""Raw-bit gates for yambda_5b.gin item (49) (a) and (b): the arithmetic posenc pos permutation
(ops/triton/triton_pos_perm.py) and the key-bound onesweep sort (ops/onesweep_sort.py) against
torch.sort, plus the extension's declines, its fail-closed return and its build lock (a stale torch
lock is recovered, a held lock times out instead of hanging). Item (49)(c), the EC joint segments on
the same extension, is gated in dlrm_v4/train/tests/ec_reduce_fused_test.py."""

import fcntl
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

import torch

from generative_recommenders.ops import onesweep_sort as osort
from generative_recommenders.ops.triton import triton_pos_perm as tpp


def _raw(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int64)


def _pos_keys(offsets, lengths, num_targets, interleave, C, M):
    """pos_inds as the item (20)(c) position kernels write them (triton_pos_perm's docstring):
    key(j) = j if j < C else min(h - min(j, h) + C, M - 1), h = Lengths[b] - targets."""
    B = lengths.numel()
    n = offsets[1:] - offsets[:-1]
    N = int(offsets[-1])
    if num_targets is None:
        h = lengths.clone()
    else:
        h = lengths - (2 * num_targets if interleave else num_targets)
    seq = torch.repeat_interleave(torch.arange(B, device=offsets.device), n)
    j = torch.arange(N, device=offsets.device) - offsets[:-1][seq]
    hh = h[seq]
    p = torch.clamp(hh - torch.minimum(j, hh) + C, max=M - 1)
    return torch.where(j < C, j, p)


def _run_child(code, timeout):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm GPU")
class PosPermTest(unittest.TestCase):
    def _check(self, so, lengths, nt, inter, C, M):
        dev = torch.device("cuda")
        so, lengths = so.to(dev), lengths.to(dev)
        nt = None if nt is None else nt.to(dev)
        B, N = lengths.numel(), int(so[-1])
        self.assertTrue(tpp.pos_perm_supported(N, B, C, M), (N, B, C, M))
        keys = _pos_keys(so, lengths, nt, inter, C, M).to(torch.int16)
        got = tpp.pos_sorted_pairs(so, lengths, nt, inter, C, M, N, torch.int16)
        ref = torch.sort(keys)
        for x, y in zip(got, ref):
            self.assertEqual((x.dtype, tuple(x.shape)), (y.dtype, tuple(y.shape)))
            self.assertTrue(x.is_contiguous())
            self.assertTrue(torch.equal(_raw(x), _raw(y)))
        return int((keys == M - 1).sum())

    def test_matches_torch_sort(self) -> None:
        g = torch.Generator().manual_seed(3401)
        clamp_cases = 0
        for i in range(40):
            B = [1, 2, 7, 300, 1024, 2048][i % 6]
            C = [0, 1, 8, 20][i % 4]
            M = [C + 2, C + 3, 64, 1024, 32768][i % 5]
            lo = max(C, 4097 // B + 1)
            n = torch.randint(lo, lo + 1500 // B + 40, (B,), generator=g)
            if i % 8 == 3:
                n[: B // 3] = 0  # empty sequences
            lengths = n.clone() if i % 3 else (n + torch.randint(-5, 6, (B,), generator=g)).clamp_min(0)
            nt = None if i % 5 == 0 else torch.randint(0, 4, (B,), generator=g)
            if nt is not None and i % 7 == 2:
                nt[: max(1, B // 4)] += lengths[: max(1, B // 4)]  # targets > length: h < 0
            inter = nt is not None and i % 6 == 0
            so = torch.zeros(B + 1, dtype=torch.int64)
            so[1:] = torch.cumsum(n, 0)
            if int(so[-1]) <= 4096:
                continue
            with self.subTest(i=i, B=B, C=C, M=M, nt=nt is not None, inter=inter):
                clamp_cases += self._check(so, lengths, nt, inter, C, M) > 0
        self.assertGreater(clamp_cases, 5)  # runs of the clamped key M - 1 were exercised

    def test_just_above_small_sort_cutoff(self) -> None:
        so = torch.tensor([0, 4097], dtype=torch.int64)
        lengths = torch.tensor([4097], dtype=torch.int64)
        self._check(so, lengths, torch.tensor([2]), False, 8, 32768)
        self._check(so, lengths, None, False, 0, 2)  # every key clamped to M - 1
        self.assertFalse(tpp.pos_perm_supported(4096, 1, 8, 32768))
        self.assertFalse(tpp.pos_perm_supported(5000, 2049, 8, 32768))
        self.assertFalse(tpp.pos_perm_supported(5000, 1, 8, 9))  # needs M >= C + 2


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm GPU")
class OnesweepSort16Test(unittest.TestCase):
    def test_matches_torch_sort(self) -> None:
        g = torch.Generator(device="cuda").manual_seed(3402)
        for n in (4097, 6144, 100003, 2_621_441):
            for mk in (0, 1, 63, 64, 4095, 4096, 32767):
                for dist in ("uniform", "max", "few", "desc"):
                    if dist == "uniform":
                        k = torch.randint(0, mk + 1, (n,), device="cuda", generator=g)
                    elif dist == "max":
                        k = torch.full((n,), mk, device="cuda")
                    elif dist == "few":
                        k = torch.randint(0, 3, (n,), device="cuda", generator=g) * (mk // 2)
                    else:
                        k = torch.sort(
                            torch.randint(0, mk + 1, (n,), device="cuda", generator=g),
                            descending=True,
                        ).values
                    k = k.to(torch.int16).contiguous()
                    with self.subTest(n=n, max_key=mk, dist=dist):
                        got = osort.sort16_pairs(k, mk)
                        self.assertIsNotNone(got)
                        ref = torch.sort(k)
                        self.assertEqual(got[1].dtype, torch.int64)
                        self.assertTrue(got[0].is_contiguous() and got[1].is_contiguous())
                        self.assertTrue(torch.equal(_raw(got[0]), _raw(ref[0])))
                        self.assertTrue(torch.equal(_raw(got[1]), _raw(ref[1])))

    def test_declines(self) -> None:
        k = torch.zeros(20000, dtype=torch.int16, device="cuda")
        self.assertIsNone(osort.sort16_pairs(k[:4096], 10))  # torch's small-sort range
        self.assertIsNone(osort.sort16_pairs(k.int(), 10))
        self.assertIsNone(osort.sort16_pairs(k[::2], 10))  # not contiguous
        self.assertIsNone(osort.sort16_pairs(k, -1))
        self.assertIsNone(osort.sort16_pairs(k, 32768))  # > 15 key bits
        # one onesweep batch only (30-bit lookback counts): the cap is 2^30 items, exclusive
        self.assertEqual(osort._MAX_ITEMS, 1 << 30)
        saved = osort._MAX_ITEMS
        try:
            osort._MAX_ITEMS = 6000
            self.assertIsNone(osort.sort16_pairs(k[:6000], 10))
            self.assertIsNotNone(osort.sort16_pairs(k[:5999], 10))
        finally:
            osort._MAX_ITEMS = saved

    def test_extension_unavailable_returns_none(self) -> None:
        self.assertIsNotNone(osort.extension())
        saved = (osort._EXT, osort._FAILED)
        try:
            osort._EXT, osort._FAILED = None, "forced by the test"
            k = torch.zeros(5000, dtype=torch.int16, device="cuda")
            self.assertIsNone(osort.sort16_pairs(k, 10))
            ids = torch.zeros(5000, dtype=torch.int64, device="cuda")
            self.assertIsNone(osort.ec_joint_segments(ids, ids, 10))
        finally:
            osort._EXT, osort._FAILED = saved


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA/ROCm GPU")
class OnesweepBuildLockTest(unittest.TestCase):
    """The build runs in a child process on a fresh build dir (a first build takes ~15 s)."""

    def test_stale_torch_lock_is_removed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="onesweep_lock_test_") as d:
            open(os.path.join(d, "lock"), "w").close()  # left by a rank killed mid-build
            r = _run_child(
                f"""
                from pathlib import Path
                from generative_recommenders.ops import onesweep_sort as osort
                osort._BUILD_DIR = Path({d!r})
                print("EXT_OK" if osort.extension() is not None else "EXT_NONE " + str(osort._FAILED))
                """,
                timeout=900,  # torch's FileBaton alone would wait on that lock forever
            )
            self.assertIn("EXT_OK", r.stdout, r.stdout + r.stderr)
            self.assertIn("stale torch build lock", r.stdout + r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "lock")))

    def test_held_build_lock_times_out_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="onesweep_lock_test_") as d:
            open(os.path.join(d, "lock"), "w").close()
            fd = os.open(os.path.join(d, "build.flock"), os.O_RDWR | os.O_CREAT, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)  # a live builder
            try:
                r = _run_child(
                    f"""
                    from pathlib import Path
                    from generative_recommenders.ops import onesweep_sort as osort
                    osort._BUILD_DIR = Path({d!r})
                    osort._BUILD_LOCK_TIMEOUT_S = 2.0
                    print("EXT_OK" if osort.extension() is not None else "EXT_NONE " + str(osort._FAILED))
                    """,
                    timeout=300,
                )
            finally:
                os.close(fd)
            self.assertIn("EXT_NONE TimeoutError", r.stdout, r.stdout + r.stderr)
            # only a lock found while holding build.flock is removed
            self.assertTrue(os.path.exists(os.path.join(d, "lock")))


if __name__ == "__main__":
    unittest.main()

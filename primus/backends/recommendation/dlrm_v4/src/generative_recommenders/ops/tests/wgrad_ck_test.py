#!/usr/bin/env python3

# pyre-unsafe

"""The custom FP8 K=tokens wgrad (wgrad_ck, DLRM_HSTU_FP8_BWD_WGRAD_CK) against
the hipBLASLt fp8t wgrad it replaces and an fp64 oracle.

Covers: W1 (N=2048) and W2 (N=1024) at token counts around the 128-token step
(1, 127, 128, 129), small and large M; run-to-run determinism (bitwise); a
NaN-prefilled partial workspace and output (no stale partial may leak);
contract declines (dtype / layout / N / alignment -> None, hipBLASLt runs);
the first-call guard (a corrupted reference is rejected and latches the custom
kernel off, the hipBLASLt result is returned); any exception and a bad hipcc
latch it off too; a stale HIP error left on the thread by another call does
not (it is drained); the build cache key covers the toolchain; with the knob
off `_run_tm_wgrad` is the hipBLASLt path. Step-tiled xT
(DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED): tiled == row-major bitwise (NaN steps
after KT never read), its own first-call guard, every fallback (knob off, a
failing launch, a rejected guard) runs hipBLASLt on the untiled operand
bitwise, untile / tiled_shape / _tm_operand_ok, tiled_active, and the LN
producers (store_t v2, recompute_fp8t) bitwise vs their row-major output.
The knob is overridden in-process with the test-only setter, so this runs
with DLRM_HSTU_FP8_BWD_WGRAD_CK on or off. Skipped without a gfx950 GPU or
hipcc.
"""

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import torch

_SIZES = (1, 127, 128, 129, 1000, 4097, 262144)


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    from generative_recommenders.ops.triton import wgrad_ck

    arch = wgrad_ck.device_arch(torch.device("cuda", 0))
    if arch != "gfx950":
        return f"wgrad_ck is gfx950-only (device {arch})"
    if not os.path.exists(wgrad_ck._hipcc()):
        return f"hipcc not found ({wgrad_ck._hipcc()})"
    return None


def _operands(m, n, seed=11, pad=128):
    """dz8 e5m2 [Mp, n] (zero tail rows) and xT e4m3 [512, Mp] (zero tail
    tokens), Mp = roundup(m, pad): the production producers' layout."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed + 7 * m + n)
    mp = (m + pad - 1) // pad * pad
    dz = torch.randn((mp, n), device=dev, generator=g)
    dz = dz * torch.exp2(torch.randn((mp, n), device=dev, generator=g) + 3.0)
    dz8 = dz.clamp(-8192.0, 8192.0).to(torch.float8_e5m2)
    dz8[m:].zero_()
    xt = (torch.randn((512, mp), device=dev, generator=g) * 24.0).clamp(-448.0, 448.0)
    x_t = xt.to(torch.float8_e4m3fn)
    x_t[:, m:].zero_()
    return dz8, x_t


def _scales():
    dev = torch.device("cuda")
    return (
        torch.full((1,), 2.0**-20, dtype=torch.float32, device=dev),
        torch.full((1,), 2.0**-7, dtype=torch.float32, device=dev),
    )


def _bits(a, b):
    return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def _rel(x, ref):
    r = ref.double()
    return float((x.double() - r).norm()) / max(float(r.norm()), 1e-300)


def _oracle(dz8, x_t, gs, as_, m, rows=None, cols=None):
    """fp64 dW (every product and sum in fp64), optionally a row/col slice."""
    x = x_t[:, :m]
    a = dz8[:m]
    if rows is not None:
        x = x.view(torch.uint8).index_select(0, rows).view(torch.float8_e4m3fn)
        a = a.view(torch.uint8).index_select(1, cols).view(torch.float8_e5m2)
    ref = torch.zeros((x.shape[0], a.shape[1]), dtype=torch.float64, device=x.device)
    step = 65536
    for s0 in range(0, m, step):
        e0 = min(m, s0 + step)
        ref += x[:, s0:e0].double() @ a[s0:e0].double()
    return ref * (float(gs.item()) * float(as_.item()))


@unittest.skipIf(_skip_reason() is not None, str(_skip_reason()))
class WgradCkTest(unittest.TestCase):
    def setUp(self) -> None:
        from generative_recommenders.ops.triton import bwd_fp8, wgrad_ck

        self.bw = bwd_fp8
        self.ck = wgrad_ck
        wgrad_ck.set_requested_for_testing(True)
        wgrad_ck.reset_for_testing()
        bwd_fp8._uvqk_tm_broken = None
        # hipBLASLt arms one solution per (device, K, N) at the first call's M
        # and runs it strictly: start each test from a fresh arming.
        bwd_fp8._uvqk_tm_armed.clear()

    def tearDown(self) -> None:
        self.ck.set_requested_for_testing(None)
        self.ck.set_tiled_requested_for_testing(None)
        self.ck.reset_for_testing()

    def _site(self, n):
        return "uvqk_projection" if n == 2048 else "vk_projection"

    def _hip(self, dz8, x_t, gs, as_, m, n):
        got = self.bw._run_tm_wgrad_hipblaslt(dz8, x_t, gs, as_, m, self._site(n))
        self.assertIsNotNone(got, self.bw._uvqk_tm_broken)
        return got

    def test_matches_hipblaslt_and_fp64(self) -> None:
        gs, as_ = _scales()
        for n in (2048, 1024):
            for m in _SIZES:
                with self.subTest(N=n, M=m):
                    # Fresh first-call arming (hipBLASLt solution and wgrad_ck guard)
                    # at this M: the hipBLASLt path runs its armed solution strictly.
                    self.bw._uvqk_tm_armed.clear()
                    self.ck.reset_for_testing()
                    dz8, x_t = _operands(m, n)
                    got = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, self._site(n))
                    self.assertIsNotNone(got)
                    d_w, label = got
                    self.assertEqual(label, "wgrad_ck_w1" if n == 2048 else "wgrad_ck_w2")
                    self.assertIsNone(self.ck.broken())
                    self.assertEqual(tuple(d_w.shape), (512, n))
                    self.assertEqual(d_w.dtype, torch.float32)
                    again = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, self._site(n))[0]
                    self.assertTrue(_bits(d_w, again), "not deterministic")
                    self.assertTrue(bool(torch.isfinite(d_w).all()))
                    hip, _ = self._hip(dz8, x_t, gs, as_, m, n)
                    self.assertLess(_rel(d_w, hip), 1e-5)
                    if m <= 4097:
                        ref = _oracle(dz8, x_t, gs, as_, m)
                        ck_err, hip_err = _rel(d_w, ref), _rel(hip, ref)
                    else:
                        g = torch.Generator().manual_seed(m + n)
                        rows = torch.randperm(512, generator=g)[:64].sort().values.cuda()
                        cols = torch.randperm(n, generator=g)[:256].sort().values.cuda()
                        ref = _oracle(dz8, x_t, gs, as_, m, rows, cols)
                        sl = d_w.index_select(0, rows).index_select(1, cols)
                        hsl = hip.index_select(0, rows).index_select(1, cols)
                        ck_err, hip_err = _rel(sl, ref), _rel(hsl, ref)
                    # Both feed the same 128-token chunks to the same MFMA (its
                    # internal 128-wide dot product dominates the error vs fp64);
                    # only the fp32 order of the chunk sums differs, so the custom
                    # kernel is as close to fp64.
                    self.assertLess(ck_err, 5e-4)
                    self.assertLess(ck_err, 2.0 * hip_err + 1e-6, (ck_err, hip_err))

    def test_nan_prefilled_workspace_and_output(self) -> None:
        gs, as_ = _scales()
        stream = torch.cuda.current_stream().cuda_stream
        for n in (2048, 1024):
            for m in (129, 1000, 262144):
                with self.subTest(N=n, M=m):
                    self.bw._uvqk_tm_armed.clear()
                    dz8, x_t = _operands(m, n, seed=5)
                    d_w, _ = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, self._site(n))
                    index = torch.cuda.current_device()
                    ws = self.ck._workspaces[
                        (index, n, int(torch.cuda.current_stream().cuda_stream))
                    ]
                    # 1) persistent workspace full of NaN -> same bits
                    ws.fill_(float("nan"))
                    d_w2, _ = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, self._site(n))
                    self.assertTrue(_bits(d_w, d_w2))
                    # 2) NaN workspace and NaN output through the raw entry point
                    ws.fill_(float("nan"))
                    o = torch.full((512, n), float("nan"), dtype=torch.float32, device="cuda")
                    rc = self.ck._funcs[n](
                        dz8.data_ptr(), x_t.data_ptr(), x_t.stride(0), ws.data_ptr(),
                        o.data_ptr(), gs.data_ptr(), as_.data_ptr(), m, stream,
                    )
                    self.assertEqual(rc, 0)
                    self.assertTrue(_bits(d_w, o))

    def test_decline_paths(self) -> None:
        gs, as_ = _scales()
        m, n = 1000, 2048
        dz8, x_t = _operands(m, n, seed=3)
        calls = []

        def ref():
            calls.append(1)
            raise AssertionError("reference must not run for a declined call")

        wide = torch.empty((dz8.shape[0], 2 * n), dtype=dz8.dtype, device=dz8.device)
        wide.view(torch.uint8).zero_()
        wide[:, :n].copy_(dz8)
        tight = x_t[:, :m].contiguous()
        unaligned = torch.empty((512, x_t.shape[1] + 16), dtype=x_t.dtype, device=x_t.device)
        unaligned.view(torch.uint8).zero_()
        unaligned[:, 8 : 8 + x_t.shape[1]].copy_(x_t)
        cases = {
            "dz8 e4m3": (dz8.view(torch.uint8).view(torch.float8_e4m3fn), x_t, gs, as_, m),
            "xT e5m2": (dz8, x_t.view(torch.uint8).view(torch.float8_e5m2), gs, as_, m),
            "N=1536": (dz8[:, :1536].contiguous(), x_t, gs, as_, m),
            "dz8 column slice": (wide[:, :n], x_t, gs, as_, m),
            "dz8 rows < M": (dz8[: m - 1], x_t, gs, as_, m),
            "xT transposed": (dz8, x_t.t().contiguous().t(), gs, as_, m),
            "xT ldm < roundup(M,128)": (dz8, tight, gs, as_, m),
            "xT 256 rows": (dz8, x_t[:256], gs, as_, m),
            "xT unaligned": (dz8, unaligned[:, 8 : 8 + x_t.shape[1]], gs, as_, m),
            "M=0": (dz8, x_t, gs, as_, 0),
            "gs fp64": (dz8, x_t, gs.double(), as_, m),
            "as 2 elements": (dz8, x_t, gs, torch.ones(2, device="cuda"), m),
        }
        for name, (a, b, g_, s_, mm) in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(self.ck.run(a, b, g_, s_, mm, "uvqk_projection", ref))
        self.assertEqual(calls, [])
        self.assertIsNone(self.ck.broken(), "a decline must not latch")
        # Through _run_tm_wgrad: hipBLASLt runs (and its result is returned).
        for name, (a, b) in {
            "N=1536": (dz8[:, :1536].contiguous(), x_t),
            "dz8 column slice": (wide[:, :n], x_t),
        }.items():
            with self.subTest(through="_run_tm_wgrad", case=name):
                got = self.bw._run_tm_wgrad(a, b, gs, as_, m, "uvqk_projection")
                self.assertIsNotNone(got)
                self.assertFalse(got[1].startswith("wgrad_ck"), got[1])
                hip = self._hip(a, b, gs, as_, m, a.shape[1])
                self.assertEqual(got[1], hip[1])
                self.assertTrue(_bits(got[0], hip[0]))
        self.assertIsNone(self.ck.broken())

    def test_guard_rejection_latches(self) -> None:
        gs, as_ = _scales()
        m, n = 4097, 2048
        dz8, x_t = _operands(m, n, seed=9)
        real = self.bw._run_tm_wgrad_hipblaslt
        corrupted = []

        def corrupt(*args, **kwargs):
            got = real(*args, **kwargs)
            bad = got[0] * 1.01
            corrupted.append(bad)
            return bad, got[1]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(
            self.bw, "_run_tm_wgrad_hipblaslt", side_effect=corrupt
        ):
            got = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, "uvqk_projection")
        text = buf.getvalue()
        self.assertIn("wgrad_ck guard uvqk_projection on cuda:", text)
        self.assertIn("-> rejected", text)
        self.assertIn("wgrad_ck DISABLED: guard rejected", text)
        self.assertNotIn("ARMED wgrad_ck", text)
        self.assertIsNotNone(got)
        self.assertFalse(got[1].startswith("wgrad_ck"))
        self.assertEqual(len(corrupted), 1, "reference computed once and reused")
        self.assertTrue(got[0] is corrupted[0], "the hipBLASLt result is returned")
        self.assertTrue(str(self.ck.broken()).startswith("guard rejected"))
        self.assertFalse(self.ck.enabled())
        # Latched: later calls never launch the custom kernel.
        with mock.patch.object(
            self.ck, "_launch", side_effect=AssertionError("custom kernel launched after the latch")
        ):
            got2 = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, "uvqk_projection")
        hip = self._hip(dz8, x_t, gs, as_, m, n)
        self.assertEqual(got2[1], hip[1])
        self.assertTrue(_bits(got2[0], hip[0]))

    def test_exception_latches(self) -> None:
        gs, as_ = _scales()
        m, n = 1000, 1024
        dz8, x_t = _operands(m, n, seed=13)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(
            self.ck, "_launch", side_effect=RuntimeError("injected launch failure")
        ):
            got = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, "vk_projection")
        self.assertIn("wgrad_ck DISABLED: RuntimeError: injected launch failure", buf.getvalue())
        hip = self._hip(dz8, x_t, gs, as_, m, n)
        self.assertEqual(got[1], hip[1])
        self.assertTrue(_bits(got[0], hip[0]))
        self.assertFalse(self.ck.enabled())

    def test_bad_hipcc_latches(self) -> None:
        gs, as_ = _scales()
        m, n = 1000, 2048
        dz8, x_t = _operands(m, n, seed=17)
        with tempfile.TemporaryDirectory() as cache:
            env = {
                "DLRM_HSTU_FP8_BWD_WGRAD_CK_HIPCC": "/nonexistent/rocm/bin/hipcc",
                "DLRM_HSTU_FP8_BWD_WGRAD_CK_CACHE": cache,
            }
            buf = io.StringIO()
            try:
                with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(buf):
                    self.ck.reset_for_testing(unload=True)
                    self.assertFalse(self.ck.prebuild())
                    got = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, "uvqk_projection")
            finally:
                self.ck.reset_for_testing(unload=True)
        self.assertIn("wgrad_ck DISABLED: FileNotFoundError", buf.getvalue())
        self.assertIn("/nonexistent/rocm/bin/hipcc", buf.getvalue())
        hip = self._hip(dz8, x_t, gs, as_, m, n)
        self.assertEqual(got[1], hip[1])
        self.assertTrue(_bits(got[0], hip[0]))

    def test_knob_off_is_hipblaslt(self) -> None:
        self.ck.set_requested_for_testing(False)
        gs, as_ = _scales()
        m, n = 4097, 2048
        dz8, x_t = _operands(m, n, seed=19)
        with mock.patch.object(
            self.ck, "run", side_effect=AssertionError("wgrad_ck.run called with the knob off")
        ):
            got = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, "uvqk_projection")
        hip = self._hip(dz8, x_t, gs, as_, m, n)
        self.assertEqual(got[1], hip[1])
        self.assertTrue(_bits(got[0], hip[0]))
        self.assertFalse(self.ck.prebuild())

    def test_stale_hip_error_is_not_a_launch_failure(self) -> None:
        # r14 iter38 review S1 (lab rv1 T2): the HIP last error is sticky per
        # thread, so an error that an unrelated call left behind must neither fail
        # the launch nor latch the custom kernel off; it is drained and logged.
        import ctypes

        hip_path = None
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "libamdhip64" in line:
                    hip_path = line.split()[-1]
                    break
        if hip_path is None:
            self.skipTest("libamdhip64 is not mapped in this process")
        hip = ctypes.CDLL(hip_path)
        hip.hipGetLastError.restype = ctypes.c_int
        hip.hipSetDevice.restype = ctypes.c_int
        hip.hipSetDevice.argtypes = [ctypes.c_int]
        gs, as_ = _scales()
        m, n = 262144, 2048
        dz8, x_t = _operands(m, n, seed=23)
        first = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, "uvqk_projection")
        self.assertEqual(first[1], "wgrad_ck_w1")
        hip.hipGetLastError()
        bad = hip.hipSetDevice(9999)  # fails, leaves a sticky error on this thread
        self.assertNotEqual(bad, 0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, "uvqk_projection")
        text = buf.getvalue()
        self.assertEqual(got[1], "wgrad_ck_w1", text)
        self.assertIsNone(self.ck.broken())
        self.assertNotIn("DISABLED", text)
        self.assertIn(f"stale HIP error {bad} drained", text)
        self.assertTrue(_bits(first[0], got[0]))
        self.assertEqual(hip.hipGetLastError(), 0, "the stale error was drained")
        # The raw entry returns the drained error negated; both kernels ran. (No
        # torch launch between the injection and the entry: torch's own launch
        # check would consume the stale error first and raise.)
        ws = self.ck._workspaces[
            (torch.cuda.current_device(), n, int(torch.cuda.current_stream().cuda_stream))
        ]
        o = torch.full((512, n), float("nan"), dtype=torch.float32, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        hip.hipSetDevice(9999)
        rc = self.ck._funcs[n](
            dz8.data_ptr(), x_t.data_ptr(), x_t.stride(0), ws.data_ptr(),
            o.data_ptr(), gs.data_ptr(), as_.data_ptr(), m, stream,
        )
        self.assertEqual(rc, -bad)
        self.assertEqual(hip.hipGetLastError(), 0)
        self.assertTrue(_bits(first[0], o))
        # A genuine argument error is still positive (and raises in _launch).
        rc = self.ck._funcs[n](
            dz8.data_ptr(), x_t.data_ptr(), x_t.stride(0) + 8, ws.data_ptr(),
            o.data_ptr(), gs.data_ptr(), as_.data_ptr(), m,
            torch.cuda.current_stream().cuda_stream,
        )
        self.assertGreater(rc, 0)

    def test_cache_key_includes_toolchain(self) -> None:
        # r14 iter38 review N4: a .so built by another hipcc is not reused.
        real = self.ck._cache_dir()
        self.assertEqual(self.ck._cache_dir(), real)
        with mock.patch.dict(self.ck._toolchains, {self.ck._hipcc(): "another hipcc 0.0"}):
            other = self.ck._cache_dir()
        self.assertNotEqual(real, other)
        self.assertEqual(real.parent, other.parent)

    def test_armed_key_has_engine(self) -> None:
        # r14 iter38 review N3: a fallback after the custom kernel's ARMED line gets
        # its own ARMED line; the hipBLASLt / BF16 key is unchanged.
        kind = "e5m2xe4m3 token-major"
        self.assertEqual(self.bw._tm_arm_kind(kind, True, "435627"), kind)
        self.assertEqual(self.bw._tm_arm_kind("bf16", False, ""), "bf16")
        self.assertNotEqual(self.bw._tm_arm_kind(kind, True, "wgrad_ck_w1"), kind)
        self.assertNotEqual(
            self.bw._tm_arm_kind(kind, True, "wgrad_ck_w2"),
            self.bw._tm_arm_kind(kind, True, "435627"),
        )

    def test_prebuild_line(self) -> None:
        if not self.bw.uvqk_fp8t_requested():
            self.skipTest("needs DLRM_HSTU_FP8_BWD_UVQK=1 DLRM_HSTU_FP8_BWD_UVQK_WGRAD=fp8t at import")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(self.bw.prebuild_uvqk_extension())
        self.assertIn("prebuild: wgrad_ck built (", buf.getvalue())
        self.assertIsNotNone(self.ck.library_path())

    # ---- DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED: step-tiled xT [KT, 512, 128] ----

    def _tiled(self, x_t, extra_nan_steps=0):
        """x_t [512, Mp] -> step-tiled [Mp/128, 512, 128]; extra steps after it hold NaN bytes."""
        kt = x_t.shape[1] // 128
        raw = torch.full(((kt + extra_nan_steps) * 65536,), 0x7F, dtype=torch.uint8, device=x_t.device)
        t = raw[: kt * 65536].view(torch.float8_e4m3fn).view(kt, 512, 128)
        t.copy_(x_t.view(512, kt, 128).permute(1, 0, 2))
        return t

    def test_tiled_matches_row_major(self) -> None:
        gs, as_ = _scales()
        for n in (2048, 1024):
            for m in _SIZES:
                with self.subTest(N=n, M=m):
                    self.bw._uvqk_tm_armed.clear()
                    self.ck.reset_for_testing()
                    dz8, x_t = _operands(m, n, seed=29)
                    tiled = self._tiled(x_t, extra_nan_steps=2)
                    rm = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, self._site(n))
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        got = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, self._site(n))
                    self.assertEqual(got[1], ("wgrad_ck_w1_t" if n == 2048 else "wgrad_ck_w2_t"))
                    self.assertIn("xT=tiled", buf.getvalue())
                    self.assertIn("-> armed", buf.getvalue())
                    self.assertEqual(rm[1], "wgrad_ck_w1" if n == 2048 else "wgrad_ck_w2")
                    self.assertTrue(_bits(got[0], rm[0]), "tiled != row-major")
                    again = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, self._site(n))[0]
                    self.assertTrue(_bits(got[0], again), "not deterministic")
                    self.assertIsNone(self.ck.broken())

    def test_tiled_fallbacks_read_the_untiled_operand(self) -> None:
        gs, as_ = _scales()
        m, n = 4097, 2048
        dz8, x_t = _operands(m, n, seed=31)
        tiled = self._tiled(x_t)
        hip = self._hip(dz8, x_t, gs, as_, m, n)
        # knob off: hipBLASLt on untile_to_row_major(tiled) == hipBLASLt on the row-major xT
        self.ck.set_requested_for_testing(False)
        got = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, "uvqk_projection")
        self.assertEqual(got[1], hip[1])
        self.assertTrue(_bits(got[0], hip[0]))
        # a failing tiled launch latches; the call returns hipBLASLt on the untiled operand
        self.ck.set_requested_for_testing(True)
        self.ck._load()  # before the patch: patch.dict restores the loaded entries
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.dict(
            self.ck._funcs_t, {n: lambda *args: 1}
        ):
            got = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, "uvqk_projection")
        self.assertIn("wgrad_ck DISABLED: RuntimeError: wgrad_ck_w1_t launch returned hip error 1", buf.getvalue())
        self.assertEqual(got[1], hip[1])
        self.assertTrue(_bits(got[0], hip[0]))
        self.assertFalse(self.ck.tiled_active(dz8.device))
        # a rejected guard on a tiled operand latches and returns the (corrupted) reference
        self.ck.reset_for_testing()
        real = self.bw._run_tm_wgrad_hipblaslt
        seen = []

        def corrupt(dz, act, *args, **kwargs):
            seen.append(act.dim())
            got = real(dz, act, *args, **kwargs)
            return got[0] * 1.01, got[1]

        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            self.bw, "_run_tm_wgrad_hipblaslt", side_effect=corrupt
        ):
            got = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, "uvqk_projection")
        self.assertEqual(seen, [2], "the guard's hipBLASLt reference reads the untiled 2-D operand")
        self.assertTrue(str(self.ck.broken()).startswith("guard rejected"))
        self.assertIn("xT=tiled", str(self.ck.broken()))
        self.assertFalse(got[1].startswith("wgrad_ck"))

    def test_untile_and_tiled_shape(self) -> None:
        for m in (1, 128, 129, 4097):
            dz8, x_t = _operands(m, 1024, seed=37)
            tiled = self._tiled(x_t)
            back = self.ck.untile_to_row_major(tiled)
            self.assertEqual(tuple(back.shape), tuple(x_t.shape))
            self.assertTrue(back.is_contiguous())
            self.assertTrue(_bits(back.view(torch.uint8), x_t.view(torch.uint8)))
            self.assertEqual(self.ck.tiled_shape(512, x_t.shape[1]), tuple(tiled.shape))
        self.assertIsNone(self.ck.tiled_shape(256, 1024))
        self.assertIsNone(self.ck.tiled_shape(512, 1000))
        with self.assertRaises(ValueError):
            self.ck.untile_to_row_major(x_t)
        # _tm_operand_ok: the tiled operand is consumable, malformed 3-D is not
        w = torch.empty((512, 1024), dtype=torch.bfloat16, device="cuda")
        self.assertTrue(self.bw._tm_operand_ok(tiled, w, 4097))
        self.assertFalse(self.bw._tm_operand_ok(tiled[:-1], w, 4097))
        self.assertFalse(self.bw._tm_operand_ok(tiled.permute(1, 0, 2), w, 4097))

    def test_tiled_active(self) -> None:
        dev = torch.device("cuda", torch.cuda.current_device())
        self.ck.set_tiled_requested_for_testing(False)
        try:
            self.assertTrue(self.ck.prebuild())
            self.assertFalse(self.ck.tiled_active(dev))
            self.ck.set_tiled_requested_for_testing(True)
            self.assertTrue(self.ck.tiled_active(dev))
            self.ck.set_requested_for_testing(False)
            self.assertFalse(self.ck.tiled_active(dev))
            self.ck.set_requested_for_testing(True)
            with contextlib.redirect_stdout(io.StringIO()):
                self.ck._disable("test latch")
            self.assertFalse(self.ck.tiled_active(dev))
        finally:
            self.ck.set_tiled_requested_for_testing(None)

    # ---- r14 iteration 40: S2 (untile copy out of memory) / S3 (tiled decline) ----

    def _oom(self, *args, **kwargs):
        raise torch.OutOfMemoryError("HIP out of memory. Tried to allocate 1.23 GiB (injected)")

    def test_untile_oom_falls_back_per_call(self) -> None:
        # S2: an OOM on the untiled copy (B11 pool / torch.empty) never latches: the
        # guard is deferred (no verdict, not armed), the call falls back (None -> the
        # caller's BF16 wgrad), and the next call runs the guard and arms as usual.
        from generative_recommenders.ops.triton import fp8_tm_alloc

        gs, as_ = _scales()
        index = torch.cuda.current_device()
        for n in (2048, 1024):
            with self.subTest(N=n):
                self.ck.reset_for_testing()
                self.bw.reset_for_testing()
                m = 4097
                dz8, x_t = _operands(m, n, seed=43)
                tiled = self._tiled(x_t)
                site = self._site(n)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), mock.patch.object(
                    fp8_tm_alloc, "empty_copy", side_effect=self._oom
                ) as copy:
                    got = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, site)
                    again = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, site)
                text = buf.getvalue()
                self.assertIsNone(got, "this call falls back (the caller's BF16 wgrad)")
                self.assertIsNone(again)
                self.assertEqual(copy.call_count, 2, "one copy attempt per call")
                self.assertIsNone(self.ck.broken(), text)
                self.assertTrue(self.ck.enabled())
                self.assertIsNone(self.bw._uvqk_tm_broken, text)
                self.assertIsNone(self.bw._uvqk_broken)
                self.assertNotIn((index, n, "t"), self.ck._armed)
                self.assertEqual(text.count(f"wgrad_ck guard {site} on cuda:{index}: deferred"), 1, text)
                self.assertEqual(text.count(f"{site}: untiled copy of the step-tiled xT"), 1, text)
                self.assertNotIn("DISABLED", text)
                # no OOM: the guard runs on this call and arms; bitwise the row-major entry
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    got = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, site)
                self.assertEqual(got[1], "wgrad_ck_w1_t" if n == 2048 else "wgrad_ck_w2_t")
                self.assertIn("-> armed", buf.getvalue())
                rm = self.bw._run_tm_wgrad(dz8, x_t, gs, as_, m, site)
                self.assertTrue(_bits(got[0], rm[0]))
                # latched (a tiled xT in flight): OOM -> None for this call, no fp8t latch;
                # without OOM hipBLASLt on the untiled copy == hipBLASLt on the row-major xT
                with contextlib.redirect_stdout(io.StringIO()):
                    self.ck._disable("test latch")
                with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    fp8_tm_alloc, "empty_copy", side_effect=self._oom
                ):
                    self.assertIsNone(self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, site))
                self.assertIsNone(self.bw._uvqk_tm_broken)
                got = self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, site)
                hip = self._hip(dz8, x_t, gs, as_, m, n)
                self.assertEqual(got[1], hip[1])
                self.assertTrue(_bits(got[0], hip[0]))
                # a non-OOM failure of the copy is not swallowed (unchanged)
                with mock.patch.object(
                    fp8_tm_alloc, "empty_copy", side_effect=RuntimeError("injected: invalid argument")
                ), self.assertRaises(RuntimeError):
                    self.bw._run_tm_wgrad(dz8, tiled, gs, as_, m, site)

    def test_untile_copy_from_the_pool(self) -> None:
        # S2: the untiled copy comes from the B11 pool when it is active (bitwise the same
        # bytes either way).
        from generative_recommenders.ops.triton import fp8_tm_alloc

        _, x_t = _operands(4097, 1024, seed=47)
        tiled = self._tiled(x_t)
        if fp8_tm_alloc.active():  # a mapped pool (in production: the prewarm)
            fp8_tm_alloc.empty_tm((512, 128), torch.float8_e4m3fn, x_t.device)
        back = self.ck.untile_to_row_major(tiled, empty=fp8_tm_alloc.empty_copy)
        self.assertTrue(_bits(back.view(torch.uint8), x_t.view(torch.uint8)))
        self.assertTrue(back.is_contiguous())
        if fp8_tm_alloc.active():
            seg = [
                s for s in torch.cuda.memory_snapshot()
                if s["address"] <= back.data_ptr() < s["address"] + s["total_size"]
            ]
            self.assertEqual(len(seg), 1)
            self.assertNotEqual(str(seg[0].get("segment_pool_id")), "(0, 0)")

    def test_untile_oom_through_uvqk_projection_backward(self) -> None:
        # S2 on the production entry: a tiled xT whose untiled copy is out of memory at
        # the first-call guard -> this call's wgrad is the BF16 torch.mm, the FP8
        # backward stays enabled (no _uvqk_broken, no fp8t latch, wgrad_ck not latched).
        from generative_recommenders.ops.triton import fp8_tm_alloc

        if not (self.bw.uvqk_fp8_backward_enabled() and self.bw.uvqk_wgrad_fp8t()):
            self.skipTest("needs DLRM_HSTU_FP8_BWD_UVQK=1 WGRAD=fp8t (e5m2 grads) at import")
        self.bw.reset_for_testing()
        dev = torch.device("cuda", torch.cuda.current_device())
        m = 4097
        g = torch.Generator(device=dev).manual_seed(53)
        w = torch.nn.Parameter((torch.randn((512, 2048), device=dev, generator=g) * 0.02).to(torch.bfloat16))
        normed = (torch.randn((m, 512), device=dev, generator=g)).to(torch.bfloat16)
        dz = (torch.randn((m, 2048), device=dev, generator=g) * 1e-3).to(torch.bfloat16)
        _, x_t = _operands(m, 2048, seed=53)
        tiled = self._tiled(x_t)
        st = self.bw._site_state(self.bw._site_name("uvqk_projection", w.data_ptr()), dev)
        st.accumulate(torch.tensor(0.05, device=dev))
        st.publish()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(
            fp8_tm_alloc, "empty_copy", side_effect=self._oom
        ):
            got = self.bw.uvqk_projection_backward(
                dz=dz, normed_x=normed, input_scale_value=0.05, weight=w,
                site="uvqk_projection", per_layer=True, normed_x_fp8_t=tiled,
            )
        text = buf.getvalue()
        self.assertIsNotNone(got, text)
        d_x, d_w, _ = got
        self.assertTrue(torch.equal(d_w, torch.mm(normed.t(), dz)), "BF16 wgrad for this call")
        self.assertIsNone(self.bw._uvqk_broken, text)
        self.assertIsNone(self.bw._uvqk_tm_broken, text)
        self.assertIsNone(self.ck.broken(), text)
        self.assertTrue(self.bw.uvqk_fp8_backward_enabled())
        self.assertIn("deferred", text)
        self.assertIn("wgrad bf16 (torch.mm)", text)
        # next call: the custom tiled wgrad runs (guard armed), FP8 path intact
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got2 = self.bw.uvqk_projection_backward(
                dz=dz, normed_x=normed, input_scale_value=0.05, weight=w,
                site="uvqk_projection", per_layer=True, normed_x_fp8_t=tiled,
            )
        self.assertIn("wgrad_ck_w1_t", buf.getvalue())
        self.assertFalse(torch.equal(got2[1], torch.mm(normed.t(), dz)))
        self.assertIsNone(self.bw._uvqk_broken)

    def test_tiled_decline_turns_tiling_off(self) -> None:
        # S3: a (steady) decline of a step-tiled xT turns the tiled producers off for the
        # rest of the process, without latching wgrad_ck; a 2-D decline does not.
        from generative_recommenders.ops.triton import triton_layer_norm as ln

        gs, as_ = _scales()
        dev = torch.device("cuda", torch.cuda.current_device())
        m, n = 1000, 2048
        dz8, x_t = _operands(m, n, seed=59)
        tiled = self._tiled(x_t)
        self.ck.set_tiled_requested_for_testing(True)
        try:
            self.assertTrue(self.ck.prebuild())
            self.assertTrue(self.ck.tiled_active(dev))

            def ref():
                raise AssertionError("reference must not run for a declined call")

            # a 2-D decline (gs fp64) leaves tiling on
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(self.ck.run(dz8, x_t, gs.double(), as_, m, "uvqk_projection", ref))
            self.assertTrue(self.ck.tiled_active(dev))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                for _ in range(3):
                    self.assertIsNone(self.ck.run(dz8, tiled, gs.double(), as_, m, "uvqk_projection", ref))
            text = buf.getvalue()
            self.assertEqual(text.count("wgrad_ck xt_tiled OFF: a step-tiled xT was declined"), 1, text)
            self.assertFalse(self.ck.tiled_active(dev))
            self.assertIsNone(self.ck.broken(), "a decline must not latch")
            self.assertTrue(self.ck.enabled())
            # the producers now write the row-major xT (what the call sites pass)
            g = torch.Generator(device=dev).manual_seed(61)
            x = (torch.randn((m, 512), device=dev, generator=g)).to(torch.bfloat16)
            w = torch.ones((512,), dtype=torch.bfloat16, device=dev)
            b = torch.zeros((512,), dtype=torch.bfloat16, device=dev)
            with mock.patch.object(ln, "_STORE_T_V2", True), contextlib.redirect_stdout(io.StringIO()):
                yt = ln.triton_weighted_layer_norm_fwd_fp8_store_t(
                    x=x, weight=w, bias=b, eps=1e-6, inv_scale=40.0, pad_to=128,
                    tiled_xt=self.ck.tiled_active(dev),
                )[3]
            self.assertEqual(yt.dim(), 2)
            # through _run_tm_wgrad: the declined tiled xT runs hipBLASLt on its untiled copy
            got = self.bw._run_tm_wgrad(dz8[:, :1536].contiguous(), tiled, gs, as_, m, "uvqk_projection")
            hip = self._hip(dz8[:, :1536].contiguous(), x_t, gs, as_, m, 1536)
            self.assertEqual(got[1], hip[1])
            self.assertTrue(_bits(got[0], hip[0]))
            # reset_for_testing clears it
            self.ck.reset_for_testing()
            self.assertTrue(self.ck.tiled_active(dev))
        finally:
            self.ck.set_tiled_requested_for_testing(None)

    def test_tiled_producers(self) -> None:
        from generative_recommenders.ops.triton import triton_layer_norm as ln

        dev = torch.device("cuda", torch.cuda.current_device())
        for m in (1, 129, 4097, 70001):
            g = torch.Generator(device=dev).manual_seed(41 + m)
            x = (torch.randn((m, 512), device=dev, generator=g) * 1.7 + 0.3).to(torch.bfloat16)
            w = (torch.randn((512,), device=dev, generator=g) * 0.2 + 1.0).to(torch.bfloat16)
            b = (torch.randn((512,), device=dev, generator=g) * 0.1).to(torch.bfloat16)
            kw = dict(x=x, weight=w, bias=b, eps=1e-6, inv_scale=40.0, pad_to=128)
            with self.subTest(M=m), mock.patch.object(ln, "_STORE_T_V2", True), contextlib.redirect_stdout(
                io.StringIO()
            ):
                rm = ln.triton_weighted_layer_norm_fwd_fp8_store_t(**kw)
                ti = ln.triton_weighted_layer_norm_fwd_fp8_store_t(**kw, tiled_xt=True)
                kt = (m + 127) // 128
                self.assertEqual(tuple(ti[3].shape), (kt, 512, 128))
                for a, c in zip(rm[:3], ti[:3]):
                    self.assertTrue(torch.equal(a.view(torch.uint8), c.view(torch.uint8)))
                self.assertTrue(torch.equal(self.ck.untile_to_row_major(ti[3]).view(torch.uint8),
                                            rm[3].view(torch.uint8)))
                rkw = dict(x=x, weight=w, bias=b, eps=1e-6, mean=rm[1], rstd=rm[2], inv_scale=40.0, pad_to=128)
                for also in (False, True):
                    r0 = ln.triton_weighted_layer_norm_recompute_fp8t(**rkw, also_bf16=also)
                    r1 = ln.triton_weighted_layer_norm_recompute_fp8t(**rkw, also_bf16=also, tiled_xt=True)
                    self.assertEqual(tuple(r1[1].shape), (kt, 512, 128))
                    self.assertTrue(torch.equal(self.ck.untile_to_row_major(r1[1]).view(torch.uint8),
                                                r0[1].view(torch.uint8)))
                    self.assertTrue(torch.equal(r0[1].view(torch.uint8), rm[3].view(torch.uint8)))
                    if also:
                        self.assertTrue(torch.equal(r0[0], r1[0]))
                # the v1 kernel declines tiling (row-major)
                with mock.patch.object(ln, "_STORE_T_V2", False):
                    v1 = ln.triton_weighted_layer_norm_fwd_fp8_store_t(**kw, tiled_xt=True)
                self.assertEqual(v1[3].dim(), 2)
                self.assertTrue(torch.equal(v1[3].view(torch.uint8), rm[3].view(torch.uint8)))


if __name__ == "__main__":
    unittest.main()

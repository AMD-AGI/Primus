#!/usr/bin/env python3

# pyre-unsafe

"""The custom FP8 K=tokens out-projection wgrad (wgrad_ck_w3,
DLRM_HSTU_FP8_BWD_WGRAD_CK_W3) against the hipBLASLt NN wgrad it replaces
(ops/hipblaslt_fp8_nn.fp8_wgrad_nn) and an fp64 oracle.

Covers: token counts around the 16-token chunk and the 128-token step (1, 15,
16, 17, 127, 128, 129), small and large M; run-to-run determinism (bitwise); a
NaN-prefilled persistent workspace and output (no stale partial may leak);
NaN in the dout_t padding [M, ldm) (also with a wider row pitch) and in the
memory after both operands (y8 rows >= M, past the dout_t allocation): no
leakage, bitwise equal to the clean result, and the padding fill touches only
[M, min(ldm, roundup(M, 128))); contract declines (dtype / layout / width /
alignment / scales -> None, no reference, no latch); the first-call guard (a
corrupted reference is rejected and latches the kernel off; the call-site
helper returns the reference computed once); an injected launch failure and a
bad hipcc latch it off too, without touching the W1/W2 latch (wgrad_ck); a
stale HIP error left on the thread by another call does not (it is drained),
also when the call zero-fills the dout_t padding first (M % 16 != 0); the
workspace is per (device, stream); with the knob off nothing runs. The knob
is overridden in-process with the test-only setter, so this runs with
DLRM_HSTU_FP8_BWD_WGRAD_CK_W3 on or off. Skipped without a gfx950 GPU or hipcc.

Step-tiled dout_t (B13, DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED, also overridden
in-process): the dual cast's tiled copy equals an independent re-layout of its
2-D copy (row-major copy and amax/partials unchanged) and untile_to_row_major
inverts it bit for bit; the tiled entry is bitwise equal to the row-major one;
its own first-call guard; contract declines; a NaN tail is zero-filled;
tiled_active() follows the knobs, the latch and the operand contract; the
call-site helper re-lays the operand out for its reference and after a latch.

x21 (r14 iteration 42, DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S, overridden in-process; pinned
OFF in setUp so every test above exercises wgrad_ck_w3_t whatever the env): the step-tiled
dout_t read by wgrad_ck_w3_t256s (256 x 256 tiles, 21 splits): its own first-call guard
(line names t256s); deterministic; rel_l2 vs hipBLASLt <= 1e-4 and in the fp32-order class
of wgrad_ck_w3_t (fp64-oracle error within 10% of it) at M in {1, 127, 128, 129, 4097,
300001, 1990209, 2519461}; NaN in the tail tokens, the workspace, the output and y8 rows >= M
cannot reach dW; knob off = wgrad_ck_w3_t; inert without the tiled knob or W3; a 2-D
dout_t still runs wgrad_ck_w3; workspace sized 21 / 16 (grown, never shrunk); guard
rejection latches; a decline turns tiling off (S3); an OOM on the guard's copy defers (S2);
prebuild binds the entry; the import-time requested / INERT lines (subprocess).
"""

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import torch

_SIZES = (1, 15, 16, 17, 127, 128, 129, 1000, 4097, 262144)
_C = 1536
_E4 = torch.float8_e4m3fn


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    from generative_recommenders.ops.triton import wgrad_ck

    arch = wgrad_ck.device_arch(torch.device("cuda", 0))
    if arch != "gfx950":
        return f"wgrad_ck_w3 is gfx950-only (device {arch})"
    if not os.path.exists(wgrad_ck._hipcc()):
        return f"hipcc not found ({wgrad_ck._hipcc()})"
    return None


def _roundup(m, k):
    return (m + k - 1) // k * k


def _operands(m, seed=11, ldm=None, poison=False):
    """y8 e4m3 [m, 1536] (unpadded) and dout_t e4m3 [512, ldm] (default ldm =
    roundup(m, 16): the dual cast's pad_to=16 layout) with a zero padding
    [m, ldm). poison=True: both are views at the start of larger buffers whose
    remainder is 0xFF (e4m3fn NaN), and the padding [m, ldm) is NaN too."""
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=dev).manual_seed(seed + 7 * m)
    ldm = _roundup(m, 16) if ldm is None else ldm
    # y8: LN*U*dropout at the activation scale (~10% dropout zeros); dout_t: a
    # heavy-tailed gradient over many binades, saturating at the e4m3 max.
    # Values depend on (seed, m) only, not on ldm.
    y = torch.randn((m, _C), device=dev, generator=g) * torch.randn(
        (m, _C), device=dev, generator=g
    )
    y = (y * 40.0 * (torch.rand((m, _C), device=dev, generator=g) >= 0.1)).clamp(
        -448.0, 448.0
    )
    d = torch.randn((512, m), device=dev, generator=g)
    d = (d * torch.exp2(torch.randn((512, m), device=dev, generator=g) * 1.5 + 3.0)).clamp(
        -448.0, 448.0
    )
    if poison:
        ybuf = torch.full(((m + 160) * _C,), 0xFF, dtype=torch.uint8, device=dev)
        y8 = ybuf[: m * _C].view(_E4).view(m, _C)
        xbuf = torch.full((512 * ldm + 65536,), 0xFF, dtype=torch.uint8, device=dev)
        x = xbuf[: 512 * ldm].view(_E4).view(512, ldm)
    else:
        y8 = torch.empty((m, _C), dtype=_E4, device=dev)
        x = torch.empty((512, ldm), dtype=_E4, device=dev)
    y8.copy_(y.to(_E4))
    x[:, :m].copy_(d.to(_E4))
    if poison:
        x[:, m:].view(torch.uint8).fill_(0xFF)
    else:
        x[:, m:].view(torch.uint8).zero_()
    return y8, x


def _scales():
    dev = torch.device("cuda", torch.cuda.current_device())
    # forward activation scale value (not a power of two), delayed dout scale * 2^5
    return (
        torch.full((1,), 0.0123457, dtype=torch.float32, device=dev),
        torch.full((1,), 3.1e-5 * 32, dtype=torch.float32, device=dev),
    )


def _bits(a, b):
    return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def _rel(x, ref):
    r = ref.double()
    return float((x.double() - r).norm()) / max(float(r.norm()), 1e-300)


def _oracle(y8, x, sy, sd, m, rows=None, cols=None):
    """fp64 dW [512, 1536] = sy*sd * x[:, :m] @ y8[:m] (every product and sum in
    fp64), optionally a row/col slice."""
    xs = x[:, :m]
    ys = y8[:m]
    if rows is not None:
        xs = xs.view(torch.uint8).index_select(0, rows).view(_E4)
        ys = ys.view(torch.uint8).index_select(1, cols).view(_E4)
    ref = torch.zeros((xs.shape[0], ys.shape[1]), dtype=torch.float64, device=x.device)
    step = 65536
    for s0 in range(0, m, step):
        e0 = min(m, s0 + step)
        ref += xs[:, s0:e0].double() @ ys[s0:e0].double()
    return ref * (float(sy.item()) * float(sd.item()))


def _hip(y8, x, sy, sd):
    from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn

    return fp8_wgrad_nn(y8, x, sy, sd, out_dtype=torch.float32)


def _tile(x, m):
    """Independent step-tiled re-layout of a row-major dout_t [512, >= roundup(m, 16)]:
    [ceil(m/128), 512, 128] e4m3 with tokens [m, ceil(m/128)*128) zero."""
    kt = _roundup(m, 128) // 128
    full = torch.zeros((512, kt * 128), dtype=torch.uint8, device=x.device)
    full[:, :m] = x.view(torch.uint8)[:, :m]
    return full.view(512, kt, 128).permute(1, 0, 2).contiguous().view(_E4)


@unittest.skipIf(_skip_reason() is not None, str(_skip_reason()))
class WgradCkW3Test(unittest.TestCase):
    def setUp(self) -> None:
        from generative_recommenders.ops.triton import wgrad_ck, wgrad_ck_w3

        self.w3 = wgrad_ck_w3
        self.ck = wgrad_ck
        wgrad_ck_w3.set_requested_for_testing(True)
        # x21: the tests below that want wgrad_ck_w3_t256s turn it on explicitly
        wgrad_ck_w3.set_t256s_requested_for_testing(False)
        wgrad_ck_w3.reset_for_testing()

    def tearDown(self) -> None:
        self.w3.set_requested_for_testing(None)
        self.w3.set_tiled_requested_for_testing(None)
        self.w3.set_t256s_requested_for_testing(None)
        self.w3.reset_for_testing()

    def _arm(self, m=4097):
        """First call on this device: the guard against hipBLASLt arms it."""
        sy, sd = _scales()
        y8, x = _operands(m, seed=1)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = self.w3.run_w3(y8, x, sy, sd, m, lambda: _hip(y8, x, sy, sd))
        self.assertIsNotNone(got, self.w3.broken())
        self.assertIn("-> armed", buf.getvalue())
        self.assertIn("ARMED wgrad_ck w3 on cuda:", buf.getvalue())

    def test_matches_hipblaslt_and_fp64(self) -> None:
        self._arm()
        sy, sd = _scales()
        for m in _SIZES:
            with self.subTest(M=m):
                y8, x = _operands(m)
                calls = []

                def ref():
                    calls.append(1)
                    raise AssertionError("guard must not run once armed")

                d_w = self.w3.run_w3(y8, x, sy, sd, m, ref)
                self.assertIsNotNone(d_w, self.w3.broken())
                self.assertEqual(calls, [])
                self.assertEqual(tuple(d_w.shape), (512, _C))
                self.assertEqual(d_w.dtype, torch.float32)
                again = self.w3.run_w3(y8, x, sy, sd, m, ref)
                self.assertTrue(_bits(d_w, again), "not deterministic")
                self.assertTrue(bool(torch.isfinite(d_w).all()))
                hip = _hip(y8, x, sy, sd)
                if hip is not None:  # the NN heuristic may offer nothing at tiny K
                    self.assertLess(_rel(d_w, hip), 1e-5)
                if m <= 4097:
                    ref64 = _oracle(y8, x, sy, sd, m)
                    ck_err = _rel(d_w, ref64)
                    hip_err = _rel(hip, ref64) if hip is not None else None
                else:
                    g = torch.Generator().manual_seed(m)
                    rows = torch.randperm(512, generator=g)[:64].sort().values.cuda()
                    cols = torch.randperm(_C, generator=g)[:256].sort().values.cuda()
                    ref64 = _oracle(y8, x, sy, sd, m, rows, cols)
                    ck_err = _rel(d_w.index_select(0, rows).index_select(1, cols), ref64)
                    hip_err = (
                        _rel(hip.index_select(0, rows).index_select(1, cols), ref64)
                        if hip is not None
                        else None
                    )
                # e4m3 x e4m3 products are exact in fp32; only the fp32 order
                # differs, so the custom kernel is as close to fp64 as hipBLASLt.
                self.assertLess(ck_err, 5e-4)
                if hip_err is not None:
                    self.assertLess(ck_err, 2.0 * hip_err + 1e-6, (ck_err, hip_err))
        self.assertIsNone(self.w3.broken())

    def test_nan_poison_no_leakage(self) -> None:
        self._arm()
        sy, sd = _scales()
        stream = torch.cuda.current_stream().cuda_stream
        index = torch.cuda.current_device()
        for m in (1, 17, 129, 1000, 4097, 262144):
            for ldm in (_roundup(m, 16), _roundup(m, 128) + 48):
                with self.subTest(M=m, ldm=ldm):
                    y_c, x_c = _operands(m, seed=5)
                    clean = self.w3.run_w3(y_c, x_c, sy, sd, m, None)
                    y8, x = _operands(m, seed=5, ldm=ldm, poison=True)
                    self.assertTrue(bool((x[:, m:].view(torch.uint8) == 0xFF).all()))
                    before = x.clone()
                    ws = self.w3._workspaces[(index, int(stream))]
                    ws.fill_(float("nan"))
                    d_w = self.w3.run_w3(y8, x, sy, sd, m, None)
                    self.assertIsNotNone(d_w, self.w3.broken())
                    self.assertTrue(bool(torch.isfinite(d_w).all()))
                    self.assertTrue(_bits(d_w, clean))
                    # the fill: exactly [m, min(ldm, roundup(m, 128))) set to 0
                    hi = min(ldm, _roundup(m, 128))
                    u8, b8 = x.view(torch.uint8), before.view(torch.uint8)
                    self.assertTrue(torch.equal(u8[:, :m], b8[:, :m]))
                    self.assertTrue(bool((u8[:, m:hi] == 0).all()))
                    self.assertTrue(torch.equal(u8[:, hi:], b8[:, hi:]))
                    # NaN workspace and NaN output through the raw entry point
                    ws.fill_(float("nan"))
                    o = torch.full((512, _C), float("nan"), dtype=torch.float32, device="cuda")
                    rc = self.w3._fn(
                        y8.data_ptr(), x.data_ptr(), x.stride(0), ws.data_ptr(),
                        o.data_ptr(), sy.data_ptr(), sd.data_ptr(), m, stream,
                    )
                    self.assertEqual(rc, 0)
                    self.assertTrue(_bits(o, clean))

    def test_decline_paths(self) -> None:
        sy, sd = _scales()
        m = 1000
        y8, x = _operands(m, seed=3)
        calls = []

        def ref():
            calls.append(1)
            raise AssertionError("reference must not run for a declined call")

        wide = torch.zeros((m, 2 * _C), dtype=torch.uint8, device="cuda").view(_E4)
        wide[:, :_C].copy_(y8)
        ldm = x.shape[1]
        xbuf = torch.zeros((512 * ldm + 16,), dtype=torch.uint8, device="cuda")
        x_unaligned = xbuf[8 : 8 + 512 * ldm].view(_E4).view(512, ldm)
        x_unaligned.copy_(x)
        ybuf = torch.zeros((m * _C + 16,), dtype=torch.uint8, device="cuda")
        y_unaligned = ybuf[8 : 8 + m * _C].view(_E4).view(m, _C)
        y_unaligned.copy_(y8)
        pitch =torch.zeros((512, x.shape[1] + 16), dtype=torch.uint8, device="cuda").view(_E4)
        odd = torch.zeros((512, 1003), dtype=torch.uint8, device="cuda").view(_E4)
        big = torch.empty((512, 8388608), dtype=_E4, device="cuda")  # 256 * ldm > 2^31 - 1
        cases = {
            "y8 e5m2": (y8.view(torch.uint8).view(torch.float8_e5m2), x, sy, sd, m),
            "dout_t e5m2": (y8, x.view(torch.uint8).view(torch.float8_e5m2), sy, sd, m),
            "y8 1024 wide": (y8[:, :1024].contiguous(), x, sy, sd, m),
            "y8 column slice": (wide[:, :_C], x, sy, sd, m),
            "y8 rows < M": (y8[: m - 1], x, sy, sd, m),
            "y8 3-D": (y8.view(1, m, _C), x, sy, sd, m),
            "dout_t transposed": (y8, x.t().contiguous().t(), sy, sd, m),
            "dout_t 256 rows": (y8, x[:256], sy, sd, m),
            "dout_t view narrower than its pitch": (y8, pitch[:, : x.shape[1]], sy, sd, m),
            "dout_t ldm % 16 != 0": (y8, odd, sy, sd, m),
            "dout_t ldm < roundup(M,16)": (y8, x[:, : _roundup(m, 16) - 16].contiguous(), sy, sd, m),
            "dout_t ldm >= 2^31 / 256": (y8, big, sy, sd, m),
            "dout_t unaligned": (y8, x_unaligned, sy, sd, m),
            "y8 unaligned": (y_unaligned, x, sy, sd, m),
            "M=0": (y8, x, sy, sd, 0),
            "scale_y fp64": (y8, x, sy.double(), sd, m),
            "scale_d 2 elements": (y8, x, sy, torch.ones(2, device="cuda"), m),
            "scale_d on cpu": (y8, x, sy, sd.cpu(), m),
        }
        for name, (a, b, s1, s2, mm) in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(self.w3.run_w3(a, b, s1, s2, mm, ref))
        del big
        self.assertEqual(calls, [])
        self.assertIsNone(self.w3.broken(), "a decline must not latch")
        self.assertEqual(len(self.w3._armed), 0)
        # The call-site helper: None -> the caller runs fp8_wgrad_nn unchanged.
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        with mock.patch.object(
            self.w3, "_launch", side_effect=AssertionError("launched for a declined call")
        ):
            got = thl._out_wgrad_nn_ck_w3(
                y8[:, :1024].contiguous(), x, sy, sd,
                mock.Mock(side_effect=AssertionError("reference ran for a declined call")),
            )
        self.assertIsNone(got)
        self.assertIsNone(self.w3.broken())

    def test_call_site_helper(self) -> None:
        from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        sy, sd = _scales()
        m = 262144
        y8, x = _operands(m, seed=21)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = thl._out_wgrad_nn_ck_w3(y8, x, sy, sd, fp8_wgrad_nn)
        self.assertIsNotNone(got)
        d_w, path = got
        self.assertEqual(path, "nn_ck")
        self.assertIn("wgrad_ck_w3 guard out_projection on cuda:", buf.getvalue())
        self.assertIn("ARMED wgrad_ck w3 on cuda:", buf.getvalue())
        hip = fp8_wgrad_nn(y8, x, sy, sd, out_dtype=torch.float32)
        self.assertLess(_rel(d_w, hip), 1e-5)
        again, path2 = thl._out_wgrad_nn_ck_w3(y8, x, sy, sd, fp8_wgrad_nn)
        self.assertEqual(path2, "nn_ck")
        self.assertTrue(_bits(d_w, again))

    def test_guard_rejection_latches(self) -> None:
        from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        sy, sd = _scales()
        m = 4097
        y8, x = _operands(m, seed=9)
        corrupted = []

        def corrupt(*args, **kwargs):
            bad = fp8_wgrad_nn(*args, **kwargs) * 1.01
            corrupted.append(bad)
            return bad

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = thl._out_wgrad_nn_ck_w3(y8, x, sy, sd, corrupt)
        text = buf.getvalue()
        self.assertIn("wgrad_ck_w3 guard out_projection on cuda:", text)
        self.assertIn("-> rejected", text)
        self.assertIn("wgrad_ck_w3 DISABLED: guard rejected", text)
        self.assertNotIn("ARMED wgrad_ck w3", text)
        self.assertIsNotNone(got)
        self.assertEqual(got[1], "nn")
        self.assertEqual(len(corrupted), 1, "reference computed once and reused")
        self.assertTrue(got[0] is corrupted[0], "the hipBLASLt result is returned")
        self.assertTrue(str(self.w3.broken()).startswith("guard rejected"))
        self.assertFalse(self.w3.enabled())
        self.assertIsNone(self.ck.broken(), "W3 latch must not latch W1/W2")
        # Latched: later calls never launch the custom kernel.
        with mock.patch.object(
            self.w3, "_launch", side_effect=AssertionError("custom kernel launched after the latch")
        ):
            self.assertIsNone(self.w3.run_w3(y8, x, sy, sd, m, corrupt))
            self.assertIsNone(thl._out_wgrad_nn_ck_w3(y8, x, sy, sd, corrupt))
        self.assertEqual(len(corrupted), 1)

    def test_exception_latches(self) -> None:
        from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        sy, sd = _scales()
        m = 1000
        y8, x = _operands(m, seed=13)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(
            self.w3, "_launch", side_effect=RuntimeError("injected launch failure")
        ):
            got = thl._out_wgrad_nn_ck_w3(y8, x, sy, sd, fp8_wgrad_nn)
        self.assertIn("wgrad_ck_w3 DISABLED: RuntimeError: injected launch failure", buf.getvalue())
        self.assertIsNone(got, "failed before the reference: the caller runs fp8_wgrad_nn")
        self.assertFalse(self.w3.enabled())
        self.assertIsNone(self.ck.broken())
        # A reference that raises (no NN algorithm) after a good launch: latched,
        # and the helper hands back that None so the caller takes its TN fallback.
        self.w3.reset_for_testing()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = thl._out_wgrad_nn_ck_w3(y8, x, sy, sd, lambda *a, **k: None)
        self.assertEqual(got, (None, "nn"))
        self.assertIn("hipBLASLt NN reference unavailable", buf.getvalue())
        self.assertFalse(self.w3.enabled())

    def test_bad_hipcc_latches(self) -> None:
        sy, sd = _scales()
        m = 1000
        y8, x = _operands(m, seed=17)
        with tempfile.TemporaryDirectory() as cache:
            env = {
                "DLRM_HSTU_FP8_BWD_WGRAD_CK_HIPCC": "/nonexistent/rocm/bin/hipcc",
                "DLRM_HSTU_FP8_BWD_WGRAD_CK_CACHE": cache,
            }
            buf = io.StringIO()
            try:
                with mock.patch.dict(os.environ, env), contextlib.redirect_stdout(buf):
                    self.w3.reset_for_testing(unload=True)
                    self.assertFalse(self.w3.prebuild())
                    got = self.w3.run_w3(y8, x, sy, sd, m, lambda: _hip(y8, x, sy, sd))
            finally:
                self.w3.reset_for_testing(unload=True)
        self.assertIsNone(got)
        self.assertIn("wgrad_ck_w3 DISABLED: FileNotFoundError", buf.getvalue())
        self.assertIn("/nonexistent/rocm/bin/hipcc", buf.getvalue())
        self.assertIsNone(self.ck.broken(), "W3 build failure must not latch W1/W2")

    def test_independent_of_w1w2_knob(self) -> None:
        sy, sd = _scales()
        m = 4097
        y8, x = _operands(m, seed=23)
        self.ck.set_requested_for_testing(False)
        try:
            self.assertFalse(self.ck.enabled())
            self.assertTrue(self.w3.enabled())
            got = self.w3.run_w3(y8, x, sy, sd, m, lambda: _hip(y8, x, sy, sd))
            self.assertIsNotNone(got, self.w3.broken())
            self.ck.set_requested_for_testing(True)
            self.w3.set_requested_for_testing(False)
            self.assertTrue(self.ck.enabled())
            self.assertFalse(self.w3.enabled())
        finally:
            self.ck.set_requested_for_testing(None)

    def test_workspace_per_stream(self) -> None:
        self._arm()
        sy, sd = _scales()
        m = 4097
        y8, x = _operands(m, seed=29)
        main = self.w3.run_w3(y8, x, sy, sd, m, None)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            other = self.w3.run_w3(y8, x, sy, sd, m, None)
        torch.cuda.current_stream().wait_stream(side)
        index = torch.cuda.current_device()
        self.assertIn((index, int(side.cuda_stream)), self.w3._workspaces)
        self.assertIn((index, int(torch.cuda.current_stream().cuda_stream)), self.w3._workspaces)
        self.assertTrue(_bits(main, other))

    def _hip_runtime(self):
        """ctypes handle on the HIP runtime of this process (the last error is per thread)."""
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
        # Never leak an injected error into a later test (torch would raise there).
        self.addCleanup(hip.hipGetLastError)
        return hip

    def _check_stale_hip_error(self, m) -> None:
        # r14 iter38 review S1, as wgrad_ck_test.test_stale_hip_error_is_not_a_launch_failure
        # (lab rv1 T2): the HIP last error is sticky per thread, so an error that an
        # unrelated call left behind must neither fail the launch nor latch the
        # custom kernel off; it is drained and logged. Every torch op of the test is
        # done before the injection; then only the production call-site helper runs.
        from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        hip = self._hip_runtime()
        sy, sd = _scales()
        y8, x = _operands(m, seed=31)
        with contextlib.redirect_stdout(io.StringIO()):
            first = thl._out_wgrad_nn_ck_w3(y8, x, sy, sd, fp8_wgrad_nn)  # arms (guard)
        self.assertIsNotNone(first, self.w3.broken())
        self.assertEqual(first[1], "nn_ck")
        index = torch.cuda.current_device()
        stream = torch.cuda.current_stream().cuda_stream
        ws = self.w3._workspaces[(index, int(stream))]
        o = torch.full((512, _C), float("nan"), dtype=torch.float32, device="cuda")
        no_ref = mock.Mock(side_effect=AssertionError("reference ran on an armed device"))
        torch.cuda.synchronize()
        hip.hipGetLastError()
        bad = hip.hipSetDevice(9999)  # fails, leaves a sticky error on this thread
        self.assertNotEqual(bad, 0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = thl._out_wgrad_nn_ck_w3(y8, x, sy, sd, no_ref)
        left = hip.hipGetLastError()
        text = buf.getvalue()
        self.assertIsNone(self.w3.broken(), text)
        self.assertNotIn("DISABLED", text)
        self.assertIsNotNone(got, text)
        self.assertEqual(got[1], "nn_ck", text)
        self.assertIn(f"stale HIP error {bad} drained", text)
        self.assertEqual(left, 0, "the stale error was drained")
        self.assertTrue(_bits(first[0], got[0]))
        # The raw entry returns the drained error negated; both kernels ran. (No
        # torch launch between the injection and the entry: torch's own launch
        # check would consume the stale error first and raise.)
        hip.hipSetDevice(9999)
        rc = self.w3._fn(
            y8.data_ptr(), x.data_ptr(), x.stride(0), ws.data_ptr(),
            o.data_ptr(), sy.data_ptr(), sd.data_ptr(), m, stream,
        )
        left = hip.hipGetLastError()
        self.assertEqual(rc, -bad)
        self.assertEqual(left, 0)
        self.assertTrue(_bits(first[0], o))
        # A genuine argument error is still positive (and raises in _launch).
        rc = self.w3._fn(
            y8.data_ptr(), x.data_ptr(), x.stride(0) + 8, ws.data_ptr(),
            o.data_ptr(), sy.data_ptr(), sd.data_ptr(), m, stream,
        )
        self.assertGreater(rc, 0)

    def test_stale_hip_error_is_not_a_launch_failure(self) -> None:
        # M % 16 == 0 at ldm = M: no padding fill, the C entry is the first launch.
        self._check_stale_hip_error(262144)

    def test_stale_hip_error_before_padding_fill_is_not_a_launch_failure(self) -> None:
        # M % 16 != 0 (as at most production M): ldm = roundup(M, 16) and run_w3
        # zero-fills dout_t[:, M:ldm) with a torch op before the C entry.
        self._check_stale_hip_error(262153)

    def test_knob_off_runs_nothing(self) -> None:
        self.w3.set_requested_for_testing(False)
        sy, sd = _scales()
        m = 1000
        y8, x = _operands(m, seed=19)
        self.assertFalse(self.w3.enabled())
        with mock.patch.object(
            self.w3, "_launch", side_effect=AssertionError("launched with the knob off")
        ):
            self.assertIsNone(
                self.w3.run_w3(y8, x, sy, sd, m, mock.Mock(side_effect=AssertionError("ref")))
            )
        self.assertFalse(self.w3.prebuild())
        self.assertIsNone(self.w3.broken())

    # ---- B13: step-tiled dout_t (wgrad_ck_w3_t) ----

    def test_dual_cast_tiled_layout(self) -> None:
        from generative_recommenders.ops.triton import bwd_fp8

        dev = torch.device("cuda", torch.cuda.current_device())
        inv = torch.full((1,), 57344.0 / 0.24, dtype=torch.float32, device=dev)
        for m in (1, 17, 128, 129, 1000, 4097):
            for kw in (
                dict(t_dtype=_E4, t_inv_mul=2.0**-5, t_limit=448.0),
                dict(t_dtype=_E4, t_inv_mul=2.0**-7, t_limit=448.0, return_partial=True),
                dict(),
            ):
                with self.subTest(M=m, kw=sorted(kw)):
                    g = torch.Generator(device=dev).manual_seed(m)
                    src = (
                        torch.randn((m, 512), device=dev, generator=g)
                        * torch.exp2(torch.randn((m, 512), device=dev, generator=g) * 1.5 - 6.0)
                    ).to(torch.bfloat16)
                    q0, t0, a0 = bwd_fp8.quantize_row_major_dual(src, inv, **kw)
                    q1, t1, a1 = bwd_fp8.quantize_row_major_dual(src, inv, t_tiled=True, **kw)
                    self.assertEqual(tuple(t0.shape), (512, _roundup(m, 16)))
                    self.assertEqual(tuple(t1.shape), (_roundup(m, 128) // 128, 512, 128))
                    self.assertTrue(t1.is_contiguous())
                    self.assertTrue(torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)))
                    self.assertTrue(_bits(a0, a1))
                    self.assertTrue(
                        torch.equal(t1.view(torch.uint8), _tile(t0, m).view(torch.uint8))
                    )
                    back = self.w3.untile_to_row_major(t1, m)
                    self.assertTrue(torch.equal(back.view(torch.uint8), t0.view(torch.uint8)))
                    # a 2-D operand passes through untouched
                    self.assertIs(self.w3.untile_to_row_major(t0, m), t0)

    def _arm_tiled(self, m=4097):
        """First tiled call on this device: its own guard (reference on the
        re-laid-out operand) arms the tiled entry."""
        sy, sd = _scales()
        y8, x = _operands(m, seed=2)
        xt = _tile(x, m)
        seen = []

        def ref():
            seen.append(1)
            return _hip(y8, self.w3.untile_to_row_major(xt, m), sy, sd)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = self.w3.run_w3(y8, xt, sy, sd, m, ref)
        self.assertIsNotNone(got, self.w3.broken())
        self.assertEqual(seen, [1])
        text = buf.getvalue()
        self.assertIn(f"tokens={m} dout_t=step-tiled rel=", text)
        self.assertIn("-> armed", text)
        self.assertIn("(dout_t step-tiled)", text)
        self.assertIn(torch.cuda.current_device(), self.w3._armed_t)

    def test_tiled_matches_rowmajor_bitwise(self) -> None:
        self._arm()
        self._arm_tiled()
        sy, sd = _scales()

        def ref():
            raise AssertionError("guard must not run once armed")

        for m in _SIZES:
            with self.subTest(M=m):
                y8, x = _operands(m)
                xt = _tile(x, m)
                rm = self.w3.run_w3(y8, x, sy, sd, m, ref)
                t = self.w3.run_w3(y8, xt, sy, sd, m, ref)
                self.assertIsNotNone(t, self.w3.broken())
                self.assertTrue(_bits(t, rm), "tiled != row-major")
                again = self.w3.run_w3(y8, xt, sy, sd, m, ref)
                self.assertTrue(_bits(t, again), "not deterministic")
                hip = _hip(y8, x, sy, sd)
                if hip is not None:
                    self.assertLess(_rel(t, hip), 1e-5)
        self.assertIsNone(self.w3.broken())

    def test_tiled_nan_tail_zero_filled(self) -> None:
        self._arm_tiled()
        sy, sd = _scales()
        stream = torch.cuda.current_stream().cuda_stream
        index = torch.cuda.current_device()
        ref = mock.Mock(side_effect=AssertionError("reference ran on an armed device"))
        for m in (1, 17, 129, 1000, 4097, 262144, 262153):
            with self.subTest(M=m):
                y8, x = _operands(m, seed=5)
                xt = _tile(x, m)
                clean = self.w3.run_w3(y8, xt, sy, sd, m, ref)
                self.assertIsNotNone(clean, self.w3.broken())
                rem = m - (xt.shape[0] - 1) * 128
                u8 = xt.view(torch.uint8)
                if rem < 128:
                    u8[-1, :, rem:].fill_(0xFF)
                before = u8.clone()
                ws = self.w3._workspaces[(index, int(stream))]
                ws.fill_(float("nan"))
                d_w = self.w3.run_w3(y8, xt, sy, sd, m, ref)
                self.assertIsNotNone(d_w, self.w3.broken())
                self.assertTrue(_bits(d_w, clean))
                self.assertTrue(bool((u8[-1, :, rem:] == 0).all()))
                self.assertTrue(torch.equal(u8[:-1], before[:-1]))
                self.assertTrue(torch.equal(u8[-1, :, :rem], before[-1, :, :rem]))
                # raw tiled entry: NaN workspace / output, ldm must be the step stride
                ws.fill_(float("nan"))
                o = torch.full((512, _C), float("nan"), dtype=torch.float32, device="cuda")
                rc = self.w3._fn_t(
                    y8.data_ptr(), xt.data_ptr(), 512 * 128, ws.data_ptr(),
                    o.data_ptr(), sy.data_ptr(), sd.data_ptr(), m, stream,
                )
                self.assertEqual(rc, 0)
                self.assertTrue(_bits(o, clean))
                rc = self.w3._fn_t(
                    y8.data_ptr(), xt.data_ptr(), _roundup(m, 16), ws.data_ptr(),
                    o.data_ptr(), sy.data_ptr(), sd.data_ptr(), m, stream,
                )
                self.assertGreater(rc, 0)

    def test_tiled_decline_paths(self) -> None:
        self._arm()
        sy, sd = _scales()
        m = 1000
        y8, x = _operands(m, seed=3)
        xt = _tile(x, m)
        calls = []

        def ref():
            calls.append(1)
            raise AssertionError("reference must not run for a declined call")

        wide = torch.zeros((9, 512, 128), dtype=torch.uint8, device="cuda").view(_E4)
        pitch = torch.zeros((8, 512, 256), dtype=torch.uint8, device="cuda").view(_E4)
        cases = {
            "KT too small": (y8, xt[:-1].contiguous(), m),
            "KT too large": (y8, wide, m),
            "rows 256": (y8, xt[:, :256].contiguous(), m),
            "non-contiguous": (y8, pitch[:, :, :128], m),
            "e5m2": (y8, xt.view(torch.uint8).view(torch.float8_e5m2), m),
            "4-D": (y8, xt.view(1, *xt.shape), m),
        }
        for name, (a, b, mm) in cases.items():
            with self.subTest(case=name):
                self.assertIsNone(self.w3.run_w3(a, b, sy, sd, mm, ref))
        self.assertEqual(calls, [])
        self.assertIsNone(self.w3.broken(), "a decline must not latch")
        self.assertEqual(len(self.w3._armed_t), 0)
        with self.assertRaises(ValueError):
            self.w3.untile_to_row_major(xt, m + 128)

    def test_tiled_active_decision(self) -> None:
        dev = torch.device("cuda", torch.cuda.current_device())
        m = 1000
        y8, _ = _operands(m, seed=7)
        self.w3.set_tiled_requested_for_testing(False)
        self.assertFalse(self.w3.tiled_requested())
        self.assertFalse(self.w3.tiled_active(dev, y8, 512))
        self.w3.set_tiled_requested_for_testing(True)
        self.assertTrue(self.w3.tiled_requested())
        self.assertTrue(self.w3.tiled_active(dev, y8, 512))
        self.assertTrue(self.w3.tiled_active(dev))
        self.assertFalse(self.w3.tiled_active(dev, y8, 256), "dout width != 512")
        self.assertFalse(self.w3.tiled_active(dev, y8[:, :1024].contiguous(), 512))
        self.assertFalse(
            self.w3.tiled_active(dev, y8.view(torch.uint8).view(torch.float8_e5m2), 512)
        )
        # the W3 knob gates it
        self.w3.set_requested_for_testing(False)
        self.assertFalse(self.w3.tiled_requested())
        self.assertFalse(self.w3.tiled_active(dev, y8, 512))
        self.w3.set_requested_for_testing(True)
        # latched -> 2-D again
        with contextlib.redirect_stdout(io.StringIO()):
            self.w3._disable("test latch")
        self.assertFalse(self.w3.tiled_active(dev, y8, 512))

    def test_call_site_tiled_reference_and_latch(self) -> None:
        from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        sy, sd = _scales()
        m = 262153
        y8, x = _operands(m, seed=41)
        xt = _tile(x, m)
        seen = []

        def nn(y, d, *a, **k):
            seen.append(d.dim())
            return fp8_wgrad_nn(y, d, *a, **k)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = thl._out_wgrad_nn_ck_w3(y8, xt, sy, sd, nn)
        self.assertIsNotNone(got, self.w3.broken())
        self.assertEqual(got[1], "nn_ck")
        self.assertEqual(seen, [2], "the guard reference reads the re-laid-out operand")
        self.assertIn("dout_t=step-tiled", buf.getvalue())
        hip = fp8_wgrad_nn(y8, x, sy, sd, out_dtype=torch.float32)
        self.assertLess(_rel(got[0], hip), 1e-5)
        rm = self.w3.run_w3(y8, x, sy, sd, m, lambda: hip)
        self.assertTrue(_bits(got[0], rm))
        # a launch failure on the tiled entry latches W3 off; the helper returns
        # None and the caller re-lays out + runs hipBLASLt (bitwise == 2-D path)
        self.w3.reset_for_testing()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(
            self.w3, "_launch", side_effect=RuntimeError("injected tiled failure")
        ):
            got = thl._out_wgrad_nn_ck_w3(y8, xt, sy, sd, nn)
        self.assertIsNone(got)
        self.assertIn("wgrad_ck_w3 DISABLED: RuntimeError: injected tiled failure", buf.getvalue())
        self.assertFalse(self.w3.enabled())
        self.assertFalse(self.w3.tiled_active(y8.device, y8, 512))
        fallback = fp8_wgrad_nn(
            y8, self.w3.untile_to_row_major(xt, m), sy, sd, out_dtype=torch.float32
        )
        self.assertTrue(_bits(fallback, hip))

    # ---- r14 iteration 40: S2 (re-laid-out copy out of memory) / S3 (tiled decline) ----

    @staticmethod
    def _oom(*args, **kwargs):
        raise torch.OutOfMemoryError("HIP out of memory. Tried to allocate 1.23 GiB (injected)")

    def test_untile_oom_guard_deferred(self) -> None:
        # S2 (W3): an OOM on the guard's re-laid-out copy defers the guard (nothing armed or
        # latched) and this call uses the all-finite custom result; the next call runs the
        # guard and arms; the result is bitwise the row-major entry's.
        from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
        from generative_recommenders.ops.triton import fp8_tm_alloc
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        sy, sd = _scales()
        m = 262153
        y8, x = _operands(m, seed=45)
        xt = _tile(x, m)
        index = torch.cuda.current_device()
        self.w3.set_tiled_requested_for_testing(True)
        nn_calls = []

        def nn(y, d, *a, **k):
            nn_calls.append(d.dim())
            return fp8_wgrad_nn(y, d, *a, **k)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(
            fp8_tm_alloc, "empty_copy", side_effect=self._oom
        ) as copy:
            got = thl._out_wgrad_nn_ck_w3(y8, xt, sy, sd, nn)
        text = buf.getvalue()
        self.assertIsNotNone(got, text)
        self.assertEqual(got[1], "nn_ck")
        self.assertEqual(copy.call_count, 1)
        self.assertEqual(nn_calls, [], "no hipBLASLt reference without the copy")
        self.assertIsNone(self.w3.broken(), text)
        self.assertNotIn(index, self.w3._armed_t)
        self.assertIn("deferred", text)
        self.assertIn("out_projection: re-laid-out copy of the step-tiled dout_t", text)
        self.assertNotIn("DISABLED", text)
        self.assertTrue(self.w3.tiled_active(y8.device, y8, 512), "tiling stays on")
        # next call: the guard runs (reference on the copy) and arms; same bits
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got2 = thl._out_wgrad_nn_ck_w3(y8, xt, sy, sd, nn)
        self.assertEqual(got2[1], "nn_ck")
        self.assertIn("dout_t=step-tiled rel=", buf.getvalue())
        self.assertIn(index, self.w3._armed_t)
        self.assertEqual(nn_calls, [2])
        self.assertTrue(_bits(got[0], got2[0]))
        rm = self.w3.run_w3(y8, x, sy, sd, m, lambda: fp8_wgrad_nn(y8, x, sy, sd, out_dtype=torch.float32))
        self.assertTrue(_bits(got[0], rm))
        # a non-finite custom result with a deferred guard latches (not used)
        self.w3.reset_for_testing()
        nan = torch.full((512, _C), float("nan"), dtype=torch.float32, device=y8.device)

        def deferred():
            raise self.ck.GuardDeferred("injected")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(self.w3, "_launch", return_value=nan):
            self.assertIsNone(self.w3.run_w3(y8, xt, sy, sd, m, deferred))
        self.assertIn("not all-finite", str(self.w3.broken()))

    def test_untile_oom_fallback_paths(self) -> None:
        # S2 (W3): the call-site helper; the copy comes from the B11 pool when active;
        # for a hipBLASLt fallback (required) an OOM is raised (no allocation-free
        # fallback exists there), for the guard reference it returns None; 2-D as is.
        from generative_recommenders.ops.triton import fp8_tm_alloc
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        m = 4097
        _, x = _operands(m, seed=47)
        xt = _tile(x, m)
        if fp8_tm_alloc.active():  # a mapped pool (in production: the prewarm)
            fp8_tm_alloc.empty_tm((512, 128), _E4, x.device)
        back = thl._dout_row_major(xt, m, required=True)
        self.assertTrue(torch.equal(back.view(torch.uint8), x.view(torch.uint8)))
        if fp8_tm_alloc.active():
            seg = [
                s for s in torch.cuda.memory_snapshot()
                if s["address"] <= back.data_ptr() < s["address"] + s["total_size"]
            ]
            self.assertNotEqual(str(seg[0].get("segment_pool_id")), "(0, 0)")
        self.assertIs(thl._dout_row_major(x, m, required=True), x)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            fp8_tm_alloc, "empty_copy", side_effect=self._oom
        ):
            self.assertIsNone(thl._dout_row_major(xt, m, required=False))
            with self.assertRaises(torch.OutOfMemoryError):
                thl._dout_row_major(xt, m, required=True)
        self.assertIsNone(self.w3.broken())

    def test_tiled_decline_turns_tiling_off(self) -> None:
        # S3 (W3): a decline of a step-tiled dout_t turns the tiled dual cast off for the
        # rest of the process without latching W3; a 2-D decline does not.
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        class Ctx:
            fp8_dout_token_major = True
            fp8_y_token_major = False

        self._arm()
        dev = torch.device("cuda", torch.cuda.current_device())
        sy, sd = _scales()
        m = 1000
        y8, x = _operands(m, seed=49)
        xt = _tile(x, m)
        dout = torch.zeros((m, 512), dtype=torch.bfloat16, device=dev)
        self.w3.set_tiled_requested_for_testing(True)

        def ref():
            raise AssertionError("reference must not run for a declined call")

        self.assertTrue(self.w3.tiled_active(dev, y8, 512))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(self.w3.run_w3(y8, x, sy.double(), sd, m, ref))  # 2-D decline
        self.assertTrue(self.w3.tiled_active(dev, y8, 512))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(3):
                self.assertIsNone(self.w3.run_w3(y8, xt, sy.double(), sd, m, ref))
        self.assertEqual(buf.getvalue().count("wgrad_ck_w3 dout_t tiling OFF"), 1, buf.getvalue())
        self.assertFalse(self.w3.tiled_active(dev, y8, 512))
        self.assertFalse(thl._out_wgrad_dout_tiled(Ctx(), dout, y8), "the cast writes 2-D")
        self.assertIsNone(self.w3.broken(), "a decline must not latch")
        self.assertTrue(self.w3.enabled())
        self.w3.reset_for_testing()
        self.assertTrue(self.w3.tiled_active(dev, y8, 512))

    def test_tiled_knob_off_casts_2d(self) -> None:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        class Ctx:
            fp8_dout_token_major = True
            fp8_y_token_major = False

        m = 1000
        y8, _ = _operands(m, seed=43)
        dout = torch.zeros((m, 512), dtype=torch.bfloat16, device="cuda")
        self.w3.set_tiled_requested_for_testing(False)
        self.assertFalse(thl._out_wgrad_dout_tiled(Ctx(), dout, y8))
        self.w3.set_tiled_requested_for_testing(True)
        self.assertTrue(thl._out_wgrad_dout_tiled(Ctx(), dout, y8))
        ctx = Ctx()
        ctx.fp8_y_token_major = True  # producer mode (TN) never tiles
        self.assertFalse(thl._out_wgrad_dout_tiled(ctx, dout, y8))
        ctx = Ctx()
        ctx.fp8_dout_token_major = False
        self.assertFalse(thl._out_wgrad_dout_tiled(ctx, dout, y8))

    def test_prebuild_line(self) -> None:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        if not (
            self.w3._REQUESTED and thl._FP8_OUT_WGRAD and thl._FP8_OUT_WGRAD_IMPL == "nn"
        ):
            self.skipTest(
                "needs DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1 DLRM_HSTU_FP8_BWD_OUT_WGRAD=1 (IMPL nn) at import"
            )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(thl.prebuild_wgrad_ck_w3())
        self.assertIn("prebuild: wgrad_ck_w3 built (", buf.getvalue())
        self.assertIsNotNone(self.w3.library_path())

    # ---- r14 iteration 42: x21, the step-tiled dout_t read by wgrad_ck_w3_t256s ----

    _T256S_SIZES = (1, 127, 128, 129, 4097, 300001, 1990209, 2519461)

    def _t256s_on(self) -> None:
        self.w3.set_tiled_requested_for_testing(True)
        self.w3.set_t256s_requested_for_testing(True)

    def _arm_t256s(self, m=4097):
        """First t256s call on this device: its own guard (hipBLASLt on the
        re-laid-out operand), its own armed set."""
        self._t256s_on()
        sy, sd = _scales()
        y8, x = _operands(m, seed=2)
        xt = _tile(x, m)
        seen = []

        def ref():
            seen.append(1)
            return _hip(y8, self.w3.untile_to_row_major(xt, m), sy, sd)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = self.w3.run_w3(y8, xt, sy, sd, m, ref)
        self.assertIsNotNone(got, self.w3.broken())
        self.assertEqual(seen, [1])
        text = buf.getvalue()
        self.assertIn(f"tokens={m} dout_t=step-tiled entry=wgrad_ck_w3_t256s rel=", text)
        self.assertIn("-> armed", text)
        self.assertIn("(dout_t step-tiled) entry=wgrad_ck_w3_t256s", text)
        index = torch.cuda.current_device()
        self.assertIn(index, self.w3._armed_s)
        self.assertNotIn(index, self.w3._armed_t)
        return got

    def test_t256s_matches_hipblaslt_fp64_and_t(self) -> None:
        self._arm_t256s()
        self.w3.set_t256s_requested_for_testing(False)
        with contextlib.redirect_stdout(io.StringIO()):
            self._arm_tiled()  # wgrad_ck_w3_t armed too: no guard inside the loop
        sy, sd = _scales()

        def ref():
            raise AssertionError("guard must not run once armed")

        for m in self._T256S_SIZES:
            with self.subTest(M=m):
                y8, x = _operands(m)
                xt = _tile(x, m)
                self.w3.set_t256s_requested_for_testing(True)
                s = self.w3.run_w3(y8, xt, sy, sd, m, ref)
                self.assertIsNotNone(s, self.w3.broken())
                self.assertEqual(tuple(s.shape), (512, _C))
                again = self.w3.run_w3(y8, xt, sy, sd, m, ref)
                self.assertTrue(_bits(s, again), "not deterministic")
                self.assertTrue(bool(torch.isfinite(s).all()))
                hip = _hip(y8, x, sy, sd)
                # the current tiled entry (knob off) on the same operand: fp32-order class
                self.w3.set_t256s_requested_for_testing(False)
                t = self.w3.run_w3(y8, xt, sy, sd, m, ref)
                self.assertIsNotNone(t, self.w3.broken())
                self.assertLess(_rel(s, t), 1e-5)
                if hip is not None:  # the NN heuristic may offer nothing at tiny K
                    self.assertLessEqual(_rel(s, hip), 1e-4)
                if m <= 4097:
                    ref64 = _oracle(y8, x, sy, sd, m)
                    s_err, t_err = _rel(s, ref64), _rel(t, ref64)
                else:
                    g = torch.Generator().manual_seed(m)
                    rows = torch.randperm(512, generator=g)[:64].sort().values.cuda()
                    cols = torch.randperm(_C, generator=g)[:256].sort().values.cuda()
                    ref64 = _oracle(y8, x, sy, sd, m, rows, cols)
                    s_err = _rel(s.index_select(0, rows).index_select(1, cols), ref64)
                    t_err = _rel(t.index_select(0, rows).index_select(1, cols), ref64)
                # same 128-token MFMA chunks, only the fixed split order differs
                self.assertLess(s_err, 5e-4)
                self.assertLessEqual(s_err, 1.1 * t_err + 1e-7, (s_err, t_err))
                self.assertLessEqual(t_err, 1.1 * s_err + 1e-7, (s_err, t_err))
                del y8, x, xt, s, again, t, hip
                torch.cuda.empty_cache()
        self.assertIsNone(self.w3.broken())

    def test_t256s_nan_padding_no_leakage(self) -> None:
        self._arm_t256s()
        sy, sd = _scales()
        stream = torch.cuda.current_stream().cuda_stream
        index = torch.cuda.current_device()
        ref = mock.Mock(side_effect=AssertionError("reference ran on an armed device"))
        for m in (1, 17, 127, 129, 1000, 4097, 262153):
            with self.subTest(M=m):
                y_c, x_c = _operands(m, seed=5)
                clean = self.w3.run_w3(y_c, _tile(x_c, m), sy, sd, m, ref)
                self.assertIsNotNone(clean, self.w3.broken())
                # y8: a view at the start of a NaN (0xFF) buffer (rows >= M are NaN)
                y8, x = _operands(m, seed=5, poison=True)
                xt = _tile(x, m)
                rem = m - (xt.shape[0] - 1) * 128
                u8 = xt.view(torch.uint8)
                if rem < 128:
                    u8[-1, :, rem:].fill_(0xFF)  # NaN tail tokens
                before = u8.clone()
                ws = self.w3._workspaces[(index, int(stream))]
                self.assertEqual(tuple(ws.shape), (21, 512, _C))
                ws.fill_(float("nan"))
                d_w = self.w3.run_w3(y8, xt, sy, sd, m, ref)
                self.assertIsNotNone(d_w, self.w3.broken())
                self.assertTrue(bool(torch.isfinite(d_w).all()))
                self.assertTrue(_bits(d_w, clean))
                self.assertTrue(bool((u8[-1, :, rem:] == 0).all()))
                self.assertTrue(torch.equal(u8[:-1], before[:-1]))
                self.assertTrue(torch.equal(u8[-1, :, :rem], before[-1, :, :rem]))
                # raw entry: NaN workspace / output; ldm must be the step stride
                ws.fill_(float("nan"))
                o = torch.full((512, _C), float("nan"), dtype=torch.float32, device="cuda")
                rc = self.w3._fn_s(
                    y8.data_ptr(), xt.data_ptr(), 512 * 128, ws.data_ptr(),
                    o.data_ptr(), sy.data_ptr(), sd.data_ptr(), m, stream,
                )
                self.assertEqual(rc, 0)
                self.assertTrue(_bits(o, clean))
                rc = self.w3._fn_s(
                    y8.data_ptr(), xt.data_ptr(), _roundup(m, 16), ws.data_ptr(),
                    o.data_ptr(), sy.data_ptr(), sd.data_ptr(), m, stream,
                )
                self.assertGreater(rc, 0)

    def test_t256s_knob_off_inert_and_2d_paths(self) -> None:
        sy, sd = _scales()
        m = 262153
        y8, x = _operands(m, seed=51)
        xt = _tile(x, m)
        index = torch.cuda.current_device()

        def hip():
            return _hip(y8, x, sy, sd)

        seen = []
        real = self.w3._launch

        def spy(*a, **k):
            seen.append((a[1].dim(), bool(k.get("t256s", False))))
            return real(*a, **k)

        # knob off (tiled on): wgrad_ck_w3_t, bitwise == the row-major entry; no t256s state
        self.w3.set_tiled_requested_for_testing(True)
        self.w3.set_t256s_requested_for_testing(False)
        self.assertFalse(self.w3.t256s_requested())
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            self.w3, "_launch", side_effect=spy
        ):
            t = self.w3.run_w3(y8, xt, sy, sd, m, hip)
            rm = self.w3.run_w3(y8, x, sy, sd, m, hip)
        self.assertTrue(_bits(t, rm))
        self.assertEqual(seen, [(3, False), (2, False)])
        self.assertEqual(len(self.w3._armed_s), 0)
        # inert without the tiled knob or without W3
        self.w3.set_t256s_requested_for_testing(True)
        self.w3.set_tiled_requested_for_testing(False)
        self.assertFalse(self.w3.t256s_requested())
        seen.clear()
        with mock.patch.object(self.w3, "_launch", side_effect=spy):
            t2 = self.w3.run_w3(y8, xt, sy, sd, m, hip)
        self.assertEqual(seen, [(3, False)])
        self.assertTrue(_bits(t2, t))
        self.w3.set_tiled_requested_for_testing(True)
        self.w3.set_requested_for_testing(False)
        self.assertFalse(self.w3.t256s_requested())
        self.w3.set_requested_for_testing(True)
        self.assertTrue(self.w3.t256s_requested())
        # knob on: a 2-D dout_t still runs the row-major entry (its own armed set)
        seen.clear()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(
            self.w3, "_launch", side_effect=spy
        ):
            rm2 = self.w3.run_w3(y8, x, sy, sd, m, hip)
            s = self.w3.run_w3(y8, xt, sy, sd, m, hip)
        self.assertEqual(seen, [(2, False), (3, True)])
        self.assertTrue(_bits(rm2, rm))
        self.assertLess(_rel(s, t), 1e-5)
        self.assertIn(index, self.w3._armed_s)
        self.assertIn("entry=wgrad_ck_w3_t256s", buf.getvalue())
        self.assertIsNone(self.w3.broken())

    def test_t256s_workspace_sizing(self) -> None:
        self._arm()
        self._arm_t256s()
        sy, sd = _scales()
        m = 4097
        y8, x = _operands(m, seed=53)
        xt = _tile(x, m)
        index = torch.cuda.current_device()
        main = torch.cuda.current_stream()
        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        s1.wait_stream(main)
        s2.wait_stream(main)
        with torch.cuda.stream(s1):  # t256s first: 21 splits; a 16-split call reuses it
            a = self.w3.run_w3(y8, xt, sy, sd, m, None)
            ws = self.w3._workspaces[(index, int(s1.cuda_stream))]
            self.assertEqual(tuple(ws.shape), (21, 512, _C))
            b = self.w3.run_w3(y8, x, sy, sd, m, None)
            self.assertIs(self.w3._workspaces[(index, int(s1.cuda_stream))], ws)
        with torch.cuda.stream(s2):  # 16-split first, then t256s grows it to 21
            c = self.w3.run_w3(y8, x, sy, sd, m, None)
            key = (index, int(s2.cuda_stream))
            self.assertEqual(tuple(self.w3._workspaces[key].shape), (16, 512, _C))
            d = self.w3.run_w3(y8, xt, sy, sd, m, None)
            self.assertEqual(tuple(self.w3._workspaces[key].shape), (21, 512, _C))
        main.wait_stream(s1)
        main.wait_stream(s2)
        self.assertTrue(_bits(a, d))
        self.assertTrue(_bits(b, c))
        # knob off: the tiled entry allocates exactly 16 splits, as before
        self.w3.set_t256s_requested_for_testing(False)
        with contextlib.redirect_stdout(io.StringIO()):
            self._arm_tiled()
        s3 = torch.cuda.Stream()
        s3.wait_stream(main)
        with torch.cuda.stream(s3):
            e = self.w3.run_w3(y8, xt, sy, sd, m, None)
            self.assertEqual(
                tuple(self.w3._workspaces[(index, int(s3.cuda_stream))].shape), (16, 512, _C)
            )
        main.wait_stream(s3)
        self.assertTrue(_bits(e, b))

    def test_t256s_guard_rejection_latches(self) -> None:
        from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        self._t256s_on()
        sy, sd = _scales()
        m = 4097
        y8, x = _operands(m, seed=55)
        xt = _tile(x, m)
        corrupted = []

        def corrupt(*args, **kwargs):
            bad = fp8_wgrad_nn(*args, **kwargs) * 1.01
            corrupted.append(bad)
            return bad

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = thl._out_wgrad_nn_ck_w3(y8, xt, sy, sd, corrupt)
        text = buf.getvalue()
        self.assertIn("dout_t=step-tiled entry=wgrad_ck_w3_t256s rel=", text)
        self.assertIn("-> rejected", text)
        self.assertIn("wgrad_ck_w3 DISABLED: guard rejected out_projection", text)
        self.assertNotIn("ARMED wgrad_ck w3", text)
        self.assertEqual(got[1], "nn")
        self.assertEqual(len(corrupted), 1)
        self.assertIn("entry=wgrad_ck_w3_t256s", str(self.w3.broken()))
        self.assertFalse(self.w3.enabled())
        self.assertFalse(self.w3.tiled_active(y8.device, y8, 512), "latched -> the cast writes 2-D")
        self.assertEqual(len(self.w3._armed_s), 0)
        with mock.patch.object(
            self.w3, "_launch", side_effect=AssertionError("custom kernel launched after the latch")
        ):
            self.assertIsNone(self.w3.run_w3(y8, xt, sy, sd, m, corrupt))
        self.assertIsNone(self.ck.broken(), "W3 latch must not latch W1/W2")

    def test_t256s_tiled_decline_turns_tiling_off(self) -> None:
        self._t256s_on()
        dev = torch.device("cuda", torch.cuda.current_device())
        sy, sd = _scales()
        m = 1000
        y8, x = _operands(m, seed=57)
        xt = _tile(x, m)

        def ref():
            raise AssertionError("reference must not run for a declined call")

        self.assertTrue(self.w3.tiled_active(dev, y8, 512))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(3):
                self.assertIsNone(self.w3.run_w3(y8, xt, sy.double(), sd, m, ref))
            self.assertIsNone(self.w3.run_w3(y8, xt[:-1].contiguous(), sy, sd, m, ref))
        self.assertEqual(buf.getvalue().count("wgrad_ck_w3 dout_t tiling OFF"), 1, buf.getvalue())
        self.assertFalse(self.w3.tiled_active(dev, y8, 512))
        self.assertIsNone(self.w3.broken(), "a decline must not latch")
        self.assertEqual(len(self.w3._armed_s), 0)

    def test_t256s_untile_oom_guard_deferred(self) -> None:
        from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
        from generative_recommenders.ops.triton import fp8_tm_alloc
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        self._t256s_on()
        sy, sd = _scales()
        m = 262153
        y8, x = _operands(m, seed=59)
        xt = _tile(x, m)
        index = torch.cuda.current_device()
        nn_calls = []

        def nn(y, d, *a, **k):
            nn_calls.append(d.dim())
            return fp8_wgrad_nn(y, d, *a, **k)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(
            fp8_tm_alloc, "empty_copy", side_effect=self._oom
        ):
            got = thl._out_wgrad_nn_ck_w3(y8, xt, sy, sd, nn)
        text = buf.getvalue()
        self.assertEqual(got[1], "nn_ck", text)
        self.assertEqual(nn_calls, [])
        self.assertIsNone(self.w3.broken(), text)
        self.assertNotIn(index, self.w3._armed_s)
        self.assertIn("entry=wgrad_ck_w3_t256s deferred (", text)
        self.assertTrue(self.w3.tiled_active(y8.device, y8, 512), "tiling stays on")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got2 = thl._out_wgrad_nn_ck_w3(y8, xt, sy, sd, nn)
        self.assertEqual(got2[1], "nn_ck")
        self.assertIn("dout_t=step-tiled entry=wgrad_ck_w3_t256s rel=", buf.getvalue())
        self.assertIn(index, self.w3._armed_s)
        self.assertEqual(nn_calls, [2])
        self.assertTrue(_bits(got[0], got2[0]))

    def test_t256s_prebuild_binds_entry(self) -> None:
        self._t256s_on()
        self.w3.reset_for_testing(unload=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(self.w3.prebuild())
        self.assertIsNotNone(self.w3._fn_s)
        self.assertIn(
            "prebuild: wgrad_ck_w3_t256s bound (step-tiled dout_t, nsplit=21)", buf.getvalue()
        )
        self.w3.set_t256s_requested_for_testing(False)
        self.w3.reset_for_testing(unload=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertTrue(self.w3.prebuild())
        self.assertNotIn("t256s", buf.getvalue())
        self.assertIsNotNone(self.w3._fn_s, "bound at load whatever the knob (never launched off)")

    def test_t256s_import_lines(self) -> None:
        import subprocess
        import sys

        code = (
            "from generative_recommenders.ops.triton import wgrad_ck_w3 as w; "
            "print('KNOBS', w.t256s_requested(), w.tiled_requested(), w.requested())"
        )
        base = {
            k: v for k, v in os.environ.items() if not k.startswith("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3")
        }
        w3, tl, s = (
            "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3",
            "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED",
            "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S",
        )
        inert = (
            "requested: wgrad_ck_w3_t256s=1 (INERT without DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1 "
            "and DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED=1)"
        )
        cases = (
            ({s: "1"}, "KNOBS False False False", "inert"),
            ({w3: "1", s: "1"}, "KNOBS False False True", "inert"),
            ({w3: "1", tl: "1", s: "1"}, "KNOBS True True True", "on"),
            ({w3: "1", tl: "1"}, "KNOBS False True True", "none"),
            ({w3: "1", tl: "1", s: "0"}, "KNOBS False True True", "none"),
        )
        for add, want, line in cases:
            with self.subTest(env=sorted(add.items())):
                r = subprocess.run(
                    [sys.executable, "-B", "-c", code], env={**base, **add},
                    capture_output=True, text=True, timeout=600,
                )
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertIn(want, r.stdout)
                if line == "inert":
                    self.assertIn(inert, r.stdout)
                elif line == "on":
                    self.assertIn("[fp8-out-wgrad] requested: wgrad_ck_w3_t256s=1\n", r.stdout)
                else:
                    self.assertNotIn("wgrad_ck_w3_t256s", r.stdout)



if __name__ == "__main__":
    unittest.main()

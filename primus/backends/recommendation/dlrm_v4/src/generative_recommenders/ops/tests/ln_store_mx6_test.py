#!/usr/bin/env python3

# pyre-unsafe

"""ln_store_mx6 (DLRM_HSTU_FP8_FWD_MXFP6 LayerNorm producer) against its two references, every output byte.

* mean / rstd / xT8 / x8: bit-identical to the shipped store_t v2 Triton LayerNorm
  (triton_weighted_layer_norm_fwd_fp8_store_t with _STORE_T_V2 and tiled_xt=True), same inputs.
* x6 (MX6-T) / xs (MX6-A-v1): byte-identical to pr7's torch reference (iter24 fp6fwd proto.py quant_A, tile_A,
  a_scale_native; copied below with only the device argument changed) applied to the kernel's own fp32 y.
* y: the kernel's pre-cast y, cast with torch to e4m3, equals its x8 bytes.
* all four compiled instantiations <X8, YD> (x8 written, y written): <1,1> carries the reference checks above; <1,0>,
  <0,1> and <0,0> (the steady-state kernel) run into their own poisoned buffers and must equal <1,1> byte for byte in
  x6 / xs / xT8 / mean / rstd, and in x8 (<1,0>) and y (<0,1>) (r14 iter49 review M1).

Outputs are prefilled with 0xA5, so an unwritten byte cannot pass; pad rows (M..Mp-1) must be zero xT8 bytes, zero
codes and scale byte 127. LN parameters are bf16 as in the model. The GPU tests need a gfx950 GPU and
/opt/rocm/bin/hipcc and skip otherwise; LnStoreMx6CpuTest (name parsing, API, the prebuild self-check design) runs on
any host with torch. Run (kernel lab): lab_py.sh <worktree root> -m pytest -q -s
generative_recommenders/ops/tests/ln_store_mx6_test.py
"""

from __future__ import annotations

import contextlib
import io
import os
import unittest
from typing import Optional
from unittest import mock

import torch

from generative_recommenders.ops.hip import ln_store_mx6 as L6

E4M3 = getattr(torch, "float8_e4m3fn", None)
# every compiled instantiation <X8, YD>; (1, 1) carries the reference checks
VARIANTS = ((1, 1), (1, 0), (0, 1), (0, 0))
M_VALUES = (1, 31, 127, 128, 129, 4097, 70001)
EPS = 1e-6
# the captured uvqk LN forward scale (r14 iter48 synthetic-input capture, L0_fwd_ln: N(0, 0.1^2) EC outputs,
# synthetic lengths and labels, no yambda data; test-only value)
INV_SCALE = 45.51110725308675


def _skip_reason() -> Optional[str]:
    if E4M3 is None or torch.version.hip is None or not torch.cuda.is_available():
        return "needs ROCm torch with float8_e4m3fn and a GPU"
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"needs gfx950 (found {arch or 'unknown'})"
    if not os.path.exists(L6.HIPCC):
        return f"needs {L6.HIPCC}"
    return None


_SKIP = _skip_reason()


# ---------------------------------------------------------------- pr7 iter24 proto.py reference (sha256 9edc2647...)
def q_e2m3(v):
    a = v.abs()
    step = torch.where(a < 2, torch.full_like(a, 0.125), torch.where(a < 4, torch.full_like(a, 0.25),
                                                                     torch.full_like(a, 0.5)))
    q = torch.clamp(torch.round(a / step) * step, max=7.5)
    return torch.copysign(q, v)


def enc_e2m3(q):
    a = q.abs()
    s = (q < 0).to(torch.int64)
    e = torch.where(a < 1, torch.zeros_like(a), torch.floor(torch.log2(torch.clamp(a, min=1.0))) + 1)
    m = torch.where(a < 1, a / 0.125, (a / torch.exp2(e - 1) - 1.0) * 8.0)
    return (s << 5) | (e.to(torch.int64) << 3) | torch.round(m).to(torch.int64)


def e8m0_exp(amax, fmax=7.5):
    e = torch.ceil(torch.log2(torch.clamp(amax, min=1e-38) / fmax))
    for _ in range(2):
        e = torch.where(amax / torch.exp2(e) > fmax, e + 1, e)
        e = torch.where(amax / torch.exp2(e - 1) <= fmax, e - 1, e)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    return torch.clamp(e, -127, 127)


def pack_rows(codes):
    r = codes.shape[0]
    c = codes.view(r, 16, 4, 8)
    sh = torch.arange(8, device=codes.device, dtype=torch.int64) * 6
    v = (c << sh).sum(-1)
    by = ((v.unsqueeze(-1) >> (torch.arange(6, device=codes.device, dtype=torch.int64) * 8)) & 255).to(torch.uint8)
    blk = by.view(r, 4, 4, 24)
    head = blk[..., :16].reshape(r, 4, 64)
    tail = blk[..., 16:].reshape(r, 4, 32)
    return torch.cat([head, tail], -1).reshape(r, 384).contiguous()


def quant_a(x, chunk=262144):
    r = x.shape[0]
    packed = torch.empty((r, 384), dtype=torch.uint8, device=x.device)
    ee = torch.empty((r, 16), dtype=torch.int32, device=x.device)
    for r0 in range(0, r, chunk):
        xb = x[r0:r0 + chunk].view(-1, 16, 32)
        e = e8m0_exp(xb.abs().amax(-1))
        q = q_e2m3(torch.clamp(xb / torch.exp2(e).unsqueeze(-1), -7.5, 7.5))
        packed[r0:r0 + chunk] = pack_rows(enc_e2m3(q))
        ee[r0:r0 + chunk] = e.to(torch.int32)
    return packed, ee


def a_scale_native(ee):
    r = ee.shape[0]
    t = (r + 127) // 128
    full = torch.zeros((t * 128, 16), dtype=torch.int32, device=ee.device)
    full[:r] = ee
    b = (full + 127).to(torch.uint8).view(t, 8, 16, 4, 4)
    return b.permute(0, 3, 4, 2, 1).contiguous().view(-1)


def lds_map(device):
    d = torch.arange(12288, device=device, dtype=torch.int64)
    b = d % 16
    lane = (d // 16) % 64
    w = (d // 1024) % 4
    i = d // 4096
    j = 4 * i + w
    head = j < 8
    rh = 16 * j + lane // 4
    qh = (lane & 3) ^ ((rh >> 2) & 3)
    rt = 32 * (j - 8) + lane // 2
    pt = (lane & 1) ^ ((rt >> 3) & 1)
    rr = torch.where(head, rh, rt)
    cb = torch.where(head, 16 * qh + b, 64 + 16 * pt + b)
    return rr, cb


def tile_a(a6):
    rr, cb = lds_map(a6.device)
    m = a6.shape[0]
    t = (m + 127) // 128
    ap = torch.zeros((t * 128, 384), dtype=torch.uint8, device=a6.device)
    ap[:m] = a6
    flat = ap.view(t, 128 * 384)
    return torch.stack([flat[:, rr * 384 + s * 96 + cb] for s in range(4)], 1).contiguous()


def _model_like_x(m: int, g: torch.Generator) -> torch.Tensor:
    mu = torch.randn(512, device="cuda", generator=g) * 0.3
    sd = torch.exp(torch.randn(512, device="cuda", generator=g) * 0.6)
    rs = torch.exp(torch.randn((m, 1), device="cuda", generator=g) * 0.5)
    x = ((torch.randn((m, 512), device="cuda", generator=g) * sd + mu) * rs).to(torch.bfloat16)
    x[0] = 0  # exact-zero row: y = bias * inv_scale
    if m > 4:
        x[m // 2] = 1.5  # constant row
        x[m // 3, :8] = 300.0  # outlier columns (saturating e4m3 bytes)
    return x


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class LnStoreMx6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            assert L6.prebuild()

    def _run(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, inv: float, x8: int = 1, yd: int = 1):
        """One instantiation <x8, yd> into freshly poisoned buffers -> (x6, xs, xT8, mean, rstd, x8 | None, y | None)."""
        m = x.shape[0]
        t = (m + 127) // 128
        poison = dict(
            xt_out=torch.full((t, 512, 128), 0xA5, dtype=torch.uint8, device="cuda").view(E4M3),
            x6_out=torch.full((t, 4, 12288), 0xA5, dtype=torch.uint8, device="cuda"),
            xs_out=torch.full((t, 4, 512), 0xA5, dtype=torch.uint8, device="cuda"),
            mean_out=torch.full((m,), float("nan"), device="cuda"),
            rstd_out=torch.full((m,), float("nan"), device="cuda"),
        )
        if x8:
            poison["x8_out"] = torch.full((m, 512), 0xA5, dtype=torch.uint8, device="cuda").view(E4M3)
        if yd:
            poison["y_out"] = torch.full((m, 512), float("nan"), device="cuda")
        res = L6.ln_store_mx6(x, w, b, EPS, inv, mp=t * 128, **poison)
        return res if yd else res + (None,)

    def _check(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, inv: float) -> None:
        from generative_recommenders.ops.triton import triton_layer_norm as ln

        m = x.shape[0]
        t = (m + 127) // 128
        with mock.patch.object(ln, "_STORE_T_V2", True), contextlib.redirect_stdout(io.StringIO()):
            y8, mean_t, rstd_t, yt = ln.triton_weighted_layer_norm_fwd_fp8_store_t(
                x=x, weight=w, bias=b, eps=EPS, inv_scale=inv, pad_to=128, tiled_xt=True)
        runs = {v: self._run(x, w, b, inv, *v) for v in VARIANTS}
        torch.cuda.synchronize()
        x6, xs, xt, mean, rstd, x8, y = runs[(1, 1)]
        u8, i32 = torch.uint8, torch.int32
        self.assertEqual(tuple(yt.shape), (t, 512, 128))
        self.assertTrue(torch.equal(mean.view(i32), mean_t.view(i32)), "mean")
        self.assertTrue(torch.equal(rstd.view(i32), rstd_t.view(i32)), "rstd")
        self.assertTrue(torch.equal(xt.view(u8), yt.view(u8)), "xT8")
        self.assertTrue(torch.equal(x8.view(u8), y8.view(u8)), "x8")
        self.assertTrue(torch.equal(y.clamp(-448.0, 448.0).to(E4M3).view(u8), x8.view(u8)), "y vs x8")
        packed, ee = quant_a(y)
        self.assertTrue(torch.equal(x6, tile_a(packed)), "x6")
        self.assertTrue(torch.equal(xs.view(-1), a_scale_native(ee)), "xs")
        self.assertFalse(bool((xs == 255).any()), "E8M0 byte 255 (NaN)")
        pad = t * 128 - m
        if pad:
            self.assertEqual(int(torch.count_nonzero(xt.view(u8)[-1, :, 128 - pad:])), 0)
        # the other three instantiations, byte for byte against <1,1>
        for (vx8, vyd), got in runs.items():
            for name, i in (("x6", 0), ("xs", 1), ("xT8", 2), ("mean", 3), ("rstd", 4)):
                self.assertTrue(torch.equal(got[i].view(u8), runs[(1, 1)][i].view(u8)), f"<{vx8},{vyd}> {name}")
            if vx8:
                self.assertTrue(torch.equal(got[5].view(u8), x8.view(u8)), f"<{vx8},{vyd}> x8")
            else:
                self.assertIsNone(got[5], f"<{vx8},{vyd}> returned x8")
            if vyd:
                self.assertTrue(torch.equal(got[6].view(i32), y.view(i32)), f"<{vx8},{vyd}> y")
            else:
                self.assertIsNone(got[6], f"<{vx8},{vyd}> returned y")

    def test_model_like(self) -> None:
        g = torch.Generator(device="cuda").manual_seed(2026)
        w = (1.0 + 0.2 * torch.randn(512, device="cuda", generator=g)).to(torch.bfloat16)
        b = (0.1 * torch.randn(512, device="cuda", generator=g)).to(torch.bfloat16)
        for m in M_VALUES:
            with self.subTest(M=m):
                self._check(_model_like_x(m, g), w, b, INV_SCALE)

    def test_edge_parameters(self) -> None:
        # zero / tiny / large weight columns with zero bias: all-zero, tiny and large-exponent MX blocks
        g = torch.Generator(device="cuda").manual_seed(7)
        w = (1.0 + 0.2 * torch.randn(512, device="cuda", generator=g)).to(torch.bfloat16)
        b = (0.1 * torch.randn(512, device="cuda", generator=g)).to(torch.bfloat16)
        w[0:32] = 0
        b[0:32] = 0
        w[32:64] = 1e-30
        b[32:64] = 0
        w[96:128] = 30.0
        b[96:128] = 0
        self._check(_model_like_x(4097, g), w, b, 1.0)

    def test_eligibility(self) -> None:
        x = torch.zeros((4, 512), dtype=torch.bfloat16, device="cuda")
        w = torch.ones((512,), dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(L6.ineligible_reason(x, w, w))
        self.assertIsNotNone(L6.ineligible_reason(x.float(), w, w))
        self.assertIsNotNone(L6.ineligible_reason(x, w.float(), w))
        self.assertIsNotNone(L6.ineligible_reason(x[:, :256], w, w))
        with self.assertRaises(ValueError):
            L6.ln_store_mx6(x, w, w, EPS, 1.0, mp=256)

    def test_ready_and_selfcheck(self) -> None:
        # setUpClass ran prebuild(); the self-check passes again on its own (both legs, pinned digests)
        self.assertIsNone(L6.ready(torch.device("cuda", torch.cuda.current_device())))
        self.assertIsNone(L6.failure())
        bad, note = L6._selfcheck(torch.device("cuda", torch.cuda.current_device()))
        self.assertEqual(bad, [], note)
        self.assertIsNotNone(L6.ready(torch.device("cpu")))
        if not os.environ.get(L6._SRC_ENV):
            self.assertEqual(L6.kernel().impl, L6.IMPL)


class LnStoreMx6CpuTest(unittest.TestCase):
    """Host-only checks (no GPU needed)."""

    def test_instantiation_tag(self) -> None:
        # the mangled names of the four kernels as hipcc emits them (r14 iter49 ISA, .name entries)
        tail = "EEEvPKtS2_S2_PfS3_PhS4_S4_S4_S3_ffii"
        for x8, yd in VARIANTS:
            name = f"_ZN5lnmx619ln_store_mx6_kernelILb{x8}ELb{yd}{tail}"
            self.assertEqual(L6.instantiation_tag(name), f"<X8,YD>=<{x8},{yd}>")
        log = "\n".join(
            f"remark: f.cu:441:0: Function Name: _ZN5lnmx619ln_store_mx6_kernelILb{x8}ELb{yd}{tail} [-Rpass-analysis="
            f"kernel-resource-usage]\nremark: f.cu:441:0:     VGPRs: {130 + 2 * x8 + yd} [-Rpass-analysis="
            f"kernel-resource-usage]" for x8, yd in VARIANTS)
        summary = L6.resource_summary(log)
        for x8, yd in VARIANTS:
            self.assertIn(f"<X8,YD>=<{x8},{yd}>: VGPRs={130 + 2 * x8 + yd}", summary)

    def test_impl_and_ready_without_gpu_device(self) -> None:
        self.assertTrue(isinstance(L6.IMPL, str) and L6.IMPL.startswith("ln_store_mx6 v3"))
        src = L6.source_path().read_text()
        self.assertIn('"ln_store_mx6 v3 (r14 iter49 + review fixes)', src)
        self.assertIsNotNone(L6.ready(torch.device("cpu")))

    def test_selfcheck_case_is_exact(self) -> None:
        # the prebuild self-check input: exact statistics, y on the e4m3 / E2M3 grids (see ln_store_mx6.py)
        x, w, b, eps, inv, mean, rstd = L6.selfcheck_case("cpu")
        xd = x.double()
        self.assertTrue(torch.equal(xd.sum(1) / 512, mean.double()))
        xm = xd - mean.double()[:, None]
        self.assertTrue(torch.equal((xm * xm).sum(1) / 512, (1.0 / rstd.double()) ** 2))
        self.assertTrue(torch.equal((xm * rstd.double()[:, None]).abs(), torch.ones_like(xd)))
        self.assertGreaterEqual(eps, 1.17549435e-38)
        y = (xm * rstd.double()[:, None] * w.double() + b.double()) * inv
        self.assertTrue(torch.equal(y.float().double(), y))
        a = y.abs()
        y8 = y.float().clamp(-448.0, 448.0).to(E4M3).double()
        self.assertTrue(torch.equal(y8[a <= 448], y[a <= 448]), "y not on the e4m3 grid")
        blk = a.view(-1, 16, 32).amax(-1)
        k = torch.tensor(L6._SC_K, dtype=torch.float64)
        self.assertTrue(torch.equal(blk, (7.5 * torch.exp2(k)).expand_as(blk)), "block amax != 7.5 * 2^k")
        q = (y.view(-1, 16, 32) / torch.exp2(k)[None, :, None]).abs()
        off = (q - q_e2m3(q)).abs()  # E2M3 rounding error of y / 2^k: zero except column 1 (2^-5, rounds to -0)
        self.assertTrue(torch.equal(off[:, :, 1], torch.full_like(off[:, :, 1], 2.0**-5)))
        off[:, :, 1] = 0
        self.assertEqual(int(torch.count_nonzero(off)), 0)
        self.assertTrue(bool((y.view(-1, 16, 32)[:, :, 1] < 0).all()))


if __name__ == "__main__":
    unittest.main()

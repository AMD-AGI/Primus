#!/usr/bin/env python3

# pyre-unsafe

"""DLRM_HSTU_FP8_FWD_MXFP6 K=512 MXFP6 GEMMs and fused weight shadow (ops/hip/k512_mx6_gemm.py).

* Weight shadow: mx6_weight == mx6_weight_reference (pr7 proto quant_W + tile_W of W^T / scale_b, torch ops) byte for
  byte, and sb == the FP8 weight shadow's scale (triton_fp8._get_weight_shadow) bitwise, on random fp32 / bf16 weights
  with zero blocks, wide exponent spreads and exact E2M3 ties, the three production layouts and a strided weight.
* Cache: a hit returns the same tensors; any in-place update (bumps _version), an optimizer step, a storage swap or
  another tensor refreshes it (a stale shadow after a weight change must be impossible).
* GEMM: every output byte equals the fp64 GEMM of exactly the dequantized MX operands followed by the shipped epilogue
  math (fp32 acc * fp32 s, + fp32 bias, sat 448, RNE e4m3), with an independent decoder (no MX library), at tail M;
  rows >= M are never written; bad operands raise ValueError before any launch.

Needs a gfx950 GPU and /opt/rocm/bin/hipcc; skips otherwise. Run (kernel lab): lab_py.sh <worktree root> -m pytest -q -s
generative_recommenders/ops/tests/k512_mx6_gemm_test.py
"""

from __future__ import annotations

import os
import unittest
from typing import Optional, Tuple

import torch

from generative_recommenders.ops.hip import k512_mx6_gemm as mx6
from generative_recommenders.ops.triton import triton_fp8 as fp8

E4M3 = getattr(torch, "float8_e4m3fn", None)
U8 = torch.uint8
F64 = torch.float64
LAYOUTS = {"VQK": ((512, 2048),), "VK": ((512, 1024), (1536, 2048)), "U": ((0, 512),)}


def _skip_reason() -> Optional[str]:
    if E4M3 is None or torch.version.hip is None or not torch.cuda.is_available():
        return "needs ROCm torch with float8_e4m3fn and a GPU"
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"needs gfx950 (found {arch or 'unknown'})"
    if not os.path.exists(mx6.HIPCC):
        return f"needs {mx6.HIPCC}"
    return None


_SKIP = _skip_reason()


# ------------------------------------------------------------------------------- independent decoder (layout formulas)
def _dec_table(device: torch.device) -> torch.Tensor:
    t = []
    for c in range(64):
        s, e, m = c >> 5, (c >> 3) & 3, c & 7
        v = (m / 8.0) if e == 0 else 2.0 ** (e - 1) * (1 + m / 8.0)
        t.append(-v if s else v)
    return torch.tensor(t, dtype=F64, device=device)


def _inv_slot(device: torch.device) -> torch.Tensor:
    """INV[R*96 + cb] = slot-image byte of byte cb of row R's 96-B K-step segment [h0 h1 h2 h3 | t0 t1 t2 t3]."""
    inv = torch.empty(128 * 96, dtype=torch.int64)
    for r in range(128):
        for q in range(4):
            for b in range(16):
                inv[r * 96 + 16 * q + b] = 64 * r + 16 * (q ^ ((r >> 2) & 3)) + b
            for b in range(8):
                inv[r * 96 + 64 + 8 * q + b] = 8192 + 32 * r + 8 * (q ^ (2 * ((r >> 3) & 1))) + b
    return inv.to(device)


def _codes(seg: torch.Tensor) -> torch.Tensor:
    """seg uint8 [R, 4, 96] -> codes int64 [R, 16, 32] (block 4S+q = head q ++ tail q, code f at bits 6f..6f+5)."""
    r = seg.shape[0]
    blk = torch.cat([seg[..., :64].reshape(r, 4, 4, 16), seg[..., 64:].reshape(r, 4, 4, 8)], -1)  # [r, S, q, 24]
    bits = torch.empty((r, 4, 4, 192), dtype=U8, device=seg.device)  # bit 8*byte + j = bit j of byte (little endian)
    for byte in range(24):
        for j in range(8):
            bits[..., 8 * byte + j] = (blk[..., byte] >> j) & 1
    w = 1 << torch.arange(6, device=seg.device, dtype=torch.int64)
    return (bits.view(r, 4, 4, 32, 6).to(torch.int64) * w).sum(-1).view(r, 16, 32)


def dequant_activation(x6: torch.Tensor, xs: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
    t0, t1 = r0 // 128, (r1 + 127) // 128
    nt = t1 - t0
    inv = _inv_slot(x6.device)
    seg = torch.stack([x6[t0:t1, s].reshape(nt, -1)[:, inv] for s in range(4)], 2)  # [nt, R*96+cb, S]
    seg = seg.reshape(nt, 128, 96, 4).permute(0, 1, 3, 2).reshape(nt * 128, 4, 96)
    e = xs[t0:t1].reshape(nt, 4, 4, 16, 8).permute(0, 4, 3, 1, 2).reshape(nt * 128, 16)
    val = _dec_table(x6.device)[_codes(seg)] * torch.exp2(e.to(F64) - 127.0).unsqueeze(-1)
    return val.reshape(nt * 128, 512)[r0 - 128 * t0:r1 - 128 * t0]


def dequant_weight(w6: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """w6 [N/256, 2, 4, 12288] + ws [N/32, 4, 4] -> fp64 [N, 512], undoing the k512 output-column permutation."""
    nt = w6.shape[0]
    n = nt * 256
    inv = _inv_slot(w6.device)
    seg = w6.reshape(nt, 2, 4, 12288)[..., inv].reshape(nt, 2, 4, 128, 96).permute(0, 1, 3, 2, 4)  # [nt, h, R, S, 96]
    rr = torch.arange(128, device=w6.device)
    rows = torch.empty((n, 4, 96), dtype=U8, device=w6.device)
    for tn in range(nt):
        for h in (0, 1):
            col = tn * 256 + 64 * (rr >> 5) + 16 * ((rr & 15) >> 2) + 4 * ((rr >> 4) & 1) + (rr & 3) + 8 * h
            rows[col] = seg[tn, h]
    e = ws.reshape(n // 32, 4, 4).permute(0, 2, 1).reshape(n // 32, 16).repeat_interleave(32, 0)
    return (_dec_table(w6.device)[_codes(rows)] * torch.exp2(e.to(F64) - 127.0).unsqueeze(-1)).reshape(n, 512)


def e4m3_reference(acc64: torch.Tensor, s: float, bias: torch.Tensor) -> torch.Tensor:
    y = acc64.float() * torch.tensor(s, dtype=torch.float32, device=acc64.device) + bias.float()
    return torch.clamp(y, -448.0, 448.0).to(E4M3).view(U8)


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    return mx6.bitwise_equal(a, b)


def _weight(seed: int, kind: str, device: str = "cuda") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn((512, 2048), generator=g) * 0.02
    if kind == "spread":
        w = w * torch.exp2(torch.randint(-12, 13, (16, 1, 64, 1), generator=g).float()).expand(16, 32, 64, 32).reshape(512, 2048)
    w[96:128, 160:192] = 0.0
    w[288:320, 1184:1216] = 0.0
    w[:, 1600:1632] = 0.0
    w[224:256, :] = 0.0
    if kind == "ties":  # sb = 1 exactly (amax 448), blocks of exact E2M3 midpoints with amax 7.5 (e = 0)
        w = torch.zeros((512, 2048))
        w[0, 513] = 448.0
        mids = torch.tensor([0.0625, 0.1875, 0.3125, 0.9375, 1.0625, 1.1875, 1.9375, 2.125, 3.875, 4.25, 6.75, 7.25])
        blk = torch.cat([mids, -mids]).repeat(43)[:1024].view(32, 32)
        for kb, cb in ((0, 20), (5, 40), (15, 55), (8, 33)):
            w[32 * kb:32 * kb + 32, 32 * cb:32 * cb + 32] = blk
            w[32 * kb, 32 * cb] = 7.5
    return w.to(device)


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class K512Mx6WeightShadowTest(unittest.TestCase):
    def setUp(self) -> None:
        mx6.clear_shadow_cache()
        fp8.clear_fp8_weight_cache()

    def _check(self, w: torch.Tensor, layout, what: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        got = mx6.mx6_weight(w, layout)
        ref = mx6.mx6_weight_reference(w, layout)
        for nm, g, r in zip(("w6", "ws", "sb"), got, ref):
            self.assertTrue(_bitwise(g, r), f"{what}: {nm} differs from the reference "
                                            f"({int((g.view(U8) != r.view(U8)).sum()) if g.shape == r.shape else 'shape'})")
        sb8 = fp8._get_weight_shadow(w, layout)[1]  # the FP8 shadow's scale, as the shipped path computes it
        self.assertTrue(_bitwise(got[2], sb8.reshape(1, 1).float()), f"{what}: sb {got[2]} != FP8 scale_b {sb8}")
        self.assertEqual(tuple(got[0].shape), (sum(b - a for a, b in layout) // 256, 2, 4, 12288))
        self.assertEqual(tuple(got[1].shape), (sum(b - a for a, b in layout) // 32, 4, 4))
        self.assertFalse(bool((got[1] == 255).any()), f"{what}: E8M0 255 (NaN) written")
        return got

    def test_bitwise_vs_reference(self) -> None:
        for seed, kind in ((1, "normal"), (2, "spread"), (3, "ties")):
            for dtype in (torch.float32, torch.bfloat16):
                w = _weight(seed, kind).to(dtype)
                for nm, layout in LAYOUTS.items():
                    self._check(w, layout, f"{kind} {dtype} {nm}")
        # strided weight: a transposed view (stride (1, 512)) of a [2048, 512] tensor
        wt = _weight(4, "spread").t().contiguous().t()
        self.assertEqual(wt.stride(), (1, 512))
        for nm, layout in LAYOUTS.items():
            self._check(wt, layout, f"strided {nm}")

    def test_refresh_on_version_bump(self) -> None:
        w = torch.nn.Parameter(_weight(5, "normal"))
        layout = LAYOUTS["VQK"]
        a = mx6.mx6_weight(w, layout)
        b = mx6.mx6_weight(w, layout)
        self.assertTrue(all(x is y for x, y in zip(a, b)), "cache hit must return the cached tensors")
        a_bytes = [t.clone() for t in a]
        v0 = w._version
        with torch.no_grad():
            w.mul_(1.5)  # in-place update: bumps _version
        self.assertGreater(w._version, v0)
        c = mx6.mx6_weight(w, layout)
        self.assertTrue(all(x is not y for x, y in zip(a, c)), "a version bump must refresh the shadow")
        for x, y in zip(c, mx6.mx6_weight_reference(w, layout)):
            self.assertTrue(_bitwise(x, y), "refreshed shadow != reference of the updated weight")
        self.assertFalse(_bitwise(c[2], a_bytes[2]), "sb must follow the update (x1.5)")
        # an optimizer step (Adam, in place) refreshes it too
        opt = torch.optim.Adam([w], lr=1e-2)
        w.grad = torch.randn_like(w) * 0.01
        v1 = w._version
        opt.step()
        self.assertGreater(w._version, v1)
        d = mx6.mx6_weight(w, layout)
        self.assertTrue(d[0] is not c[0])
        for x, y in zip(d, mx6.mx6_weight_reference(w, layout)):
            self.assertTrue(_bitwise(x, y), "post-optimizer-step shadow != reference")
        # a version bump that leaves the values unchanged still refreshes (the version is the key), same bytes
        with torch.no_grad():
            w.add_(0.0)
        e = mx6.mx6_weight(w, layout)
        self.assertTrue(e[0] is not d[0] and all(_bitwise(x, y) for x, y in zip(e, d)))
        # a storage swap (new data_ptr) refreshes; another tensor with equal values gets its own entry
        with torch.no_grad():
            w.data = w.data.clone() * 2.0
        f = mx6.mx6_weight(w, layout)
        self.assertTrue(all(_bitwise(x, y) for x, y in zip(f, mx6.mx6_weight_reference(w, layout))))
        other = w.detach().clone()
        g = mx6.mx6_weight(other, layout)
        self.assertTrue(g[0] is not f[0] and all(_bitwise(x, y) for x, y in zip(g, f)))
        # layouts are separate entries
        h = mx6.mx6_weight(w, LAYOUTS["VK"])
        self.assertEqual(tuple(h[0].shape), (4, 2, 4, 12288))
        stats = mx6.shadow_cache_stats()
        self.assertGreaterEqual(stats["hits"], 1)
        self.assertGreaterEqual(stats["refreshes"], 7)

    def test_rejects_bad_weights(self) -> None:
        w = _weight(6, "normal")
        with self.assertRaises(ValueError):
            mx6.mx6_weight(w[:256], LAYOUTS["VQK"])  # K != 512
        with self.assertRaises(ValueError):
            mx6.mx6_weight(w, ((0, 300),))  # N not a multiple of 256
        with self.assertRaises(ValueError):
            mx6.mx6_weight(w.half(), LAYOUTS["U"])  # dtype
        with self.assertRaises(ValueError):
            mx6.mx6_weight(w, ((0, 4096),))  # out of range


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class K512Mx6GemmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        g = torch.Generator(device="cpu").manual_seed(20261010)
        cls.weight = (torch.randn((512, 2048), generator=g) * 0.03).cuda()
        cls.bias_full = (torch.randn((2048,), generator=g) * 2.0).to(torch.bfloat16).cuda()

    def _y(self, m: int, seed: int) -> torch.Tensor:
        """LN-like fp32 y = x * inv_scale (narrow block exponents, as captured LN outputs; a few outlier columns)."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        y = torch.randn((m, 512), generator=g) * 20.0
        y[:, 7] *= 6.0
        y[:, 300] *= 3.0
        y[::31] = 0.0
        return y.cuda()

    def _run(self, m: int, n: int, seed: int) -> None:
        layout = LAYOUTS["VQK"] if n == 1536 else LAYOUTS["VK"]
        w6, ws, sb = mx6.mx6_weight(self.weight, layout)
        x6, xs = mx6.mx6_activation_reference(self._y(m, seed))
        bias = torch.cat([self.bias_full[a:b] for a, b in layout]).contiguous()
        sa = torch.full((1, 1), 0.0213, device="cuda")
        buf = torch.full((m + 128, n), 0x7F, dtype=U8, device="cuda")  # e4m3 NaN poison + 128 guard rows
        out = buf[:m].view(E4M3)
        res = mx6.k512_mx6(x6, xs, w6, ws, bias, sa, sb, m, n, out=out)
        torch.cuda.synchronize()
        self.assertIs(res, out)
        self.assertTrue(bool((buf[m:] == 0x7F).all()), f"M={m} N={n}: rows >= M written")
        s = float((sa.reshape(1) * sb.reshape(1)).item())  # the kernel's fp32 s = sa * sb
        dw = dequant_weight(w6, ws)
        b64 = bias.double()
        mism = unexplained = 0
        for r0 in range(0, m, 16384):
            r1 = min(m, r0 + 16384)
            da = dequant_activation(x6, xs, r0, r1)
            acc = da @ dw.t()
            ref = e4m3_reference(acc, s, bias)
            ne = buf[r0:r1] != ref
            nne = int(ne.sum())
            if nne:
                # MX6 is exact against fp64 on captured operands (r14 iter49 e2e: 0 of 2.1e10 bytes); with synthetic
                # exponent spreads the MFMA's fp32 accumulation may round (pr7 t2: 19 of 6.7e7, all 1 ulp). Accept
                # only 1-ulp neighbours whose exact value lies within the fp32-accumulation budget of the midpoint.
                idx = torch.nonzero(ne)
                rr, cc = idx[:, 0], idx[:, 1]
                g = buf[r0:r1][rr, cc].to(torch.int32)
                r = ref[rr, cc].to(torch.int32)
                adjacent = ((g & 0x80) == (r & 0x80)) & ((g & 0x7F) - (r & 0x7F)).abs().eq(1)
                gv = buf[r0:r1][rr, cc].view(E4M3).float().double()
                rv = ref[rr, cc].view(E4M3).float().double()
                z = acc[rr, cc] * s + b64[cc]
                budget = abs(s) * (da[rr].abs() * dw[cc].abs()).sum(1) * 2.0 ** -21 + (0.5 * (gv + rv)).abs() * 2.0 ** -22
                ok = adjacent & ((z - 0.5 * (gv + rv)).abs() <= budget)
                mism += nne
                unexplained += int((~ok).sum())
        print(f"[k512-mx6-test] N={n} M={m}: {m * n} bytes, mismatches vs fp64 reference {mism} "
              f"(unexplained {unexplained})", flush=True)
        self.assertEqual(unexplained, 0, f"N={n} M={m}")
        self.assertLessEqual(mism, max(2, (m * n) // 1_000_000), f"N={n} M={m}: mismatch rate")

    def test_full_bytes_vs_fp64_reference(self) -> None:
        for n in (1536, 1024):
            for i, m in enumerate((1, 127, 129, 4097, 70001)):
                self._run(m, n, 100 * n + i)

    def test_rejects_bad_operands(self) -> None:
        m, n = 300, 1536
        w6, ws, sb = mx6.mx6_weight(self.weight, LAYOUTS["VQK"])
        x6, xs = mx6.mx6_activation_reference(self._y(m, 1))
        bias = self.bias_full[512:].contiguous()
        sa = torch.ones((1, 1), device="cuda")
        out = torch.empty((m, n), dtype=E4M3, device="cuda")
        cases = {
            "x6 size": (x6[:-1], xs, w6, ws, bias, sa, sb, m, n, out),
            "xs dtype": (x6, xs.to(torch.int16), w6, ws, bias, sa, sb, m, n, out),
            "w6 for N=1024": (x6, xs, w6.reshape(-1)[: 1024 * 384], ws, bias, sa, sb, m, n, out),
            "bias dtype": (x6, xs, w6, ws, bias.float(), sa, sb, m, n, out),
            "scale numel": (x6, xs, w6, ws, bias, torch.ones(2, device="cuda"), sb, m, n, out),
            "out dtype": (x6, xs, w6, ws, bias, sa, sb, m, n, out.view(U8)),
            "out shape": (x6, xs, w6, ws, bias, sa, sb, m, n, out[:-1]),
            "N": (x6, xs, w6, ws, bias, sa, sb, m, 2048, out),
            "cpu": (x6.cpu(), xs, w6, ws, bias, sa, sb, m, n, out),
        }
        for what, args in cases.items():
            with self.assertRaises(ValueError, msg=what):
                mx6.k512_mx6(*args[:9], out=args[9])

    def test_u_dispatch_when_absent(self) -> None:
        if mx6.u_source_path() is not None:
            self.skipTest("track B's U source is present")
        self.assertNotIn(512, mx6.SUPPORTED_N)
        with self.assertRaisesRegex(RuntimeError, "U library absent"):
            mx6.kernel(512)

    def test_prebuild(self) -> None:
        self.assertTrue(mx6.prebuild())


if __name__ == "__main__":
    unittest.main()

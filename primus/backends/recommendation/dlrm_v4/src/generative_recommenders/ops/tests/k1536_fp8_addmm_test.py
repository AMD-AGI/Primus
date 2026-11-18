#!/usr/bin/env python3

# pyre-unsafe

"""Bit parity of the DLRM_HSTU_FP8_FWD_K1536 out-projection forward GEMM against hipBLASLt.

Both sides run through triton_fp8.scaled_addmm_fp8_preq, the production entry point, on its
residual-fused branch (DLRM_HSTU_FP8_FUSE_RESIDUAL=1), with the real _get_weight_shadow
(e4m3 [512, 1536] row-major shadow seen transposed). Only the module-level knob
_HSTU_FP8_FWD_K1536 is toggled, so the reference is the residual-fused hipBLASLt extension
(ops/triton/hipblaslt_scaled_addmm.cpp, algorithm index 1, 64 MiB workspace). Every output
word is compared; the custom path's output buffer is prefilled with 0xFFFF (bf16 NaN), so an
unwritten row cannot pass.

Production layout: out-projection weight [1536, 512] (K x N), activation = the e4m3
LN*silu(u)*dropout output [M, 1536], residual = the layer input x bf16 [M, 512]. Model-like
data: e4m3 activation of N(0, 32), weight N(0, 0.02) quantized by the production shadow,
residual N(0, 1), non power-of-two activation scale, so acc * (sa * sb) really rounds.

Needs a gfx950 GPU and /opt/rocm/bin/hipcc; skips otherwise. Run (kernel lab): lab_py.sh <worktree root> -m pytest -q -s
generative_recommenders/ops/tests/k1536_fp8_addmm_test.py
"""

from __future__ import annotations

import os
import unittest
from typing import List, Optional, Tuple
from unittest.mock import patch

import torch

from generative_recommenders.ops.hip import k1536_fp8_addmm as k1536
from generative_recommenders.ops.triton import triton_fp8 as fp8

E4M3 = getattr(torch, "float8_e4m3fn", None)
K = 1536
N = 512
M_VALUES: Tuple[int, ...] = (1, 255, 256, 257, 300, 1000, 300001, 2284545, 2376002, 2650962)
SIG_A = 32.0
SCALE_A = 0.0316  # ~1/SIG_A, not a power of two


def _skip_reason() -> Optional[str]:
    if E4M3 is None or torch.version.hip is None or not torch.cuda.is_available():
        return "needs ROCm torch with float8_e4m3fn and a GPU"
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"needs gfx950 (found {arch or 'unknown'})"
    if not os.path.exists(k1536.HIPCC):
        return f"needs {k1536.HIPCC}"
    return None


_SKIP = _skip_reason()


def _nan_output(residual: torch.Tensor) -> torch.Tensor:
    return torch.full(tuple(residual.shape), -1, dtype=torch.int16, device=residual.device).view(
        torch.bfloat16
    )


def _random_e4m3(rows: int, sigma: float, generator: torch.Generator) -> torch.Tensor:
    out = torch.empty((rows, K), dtype=E4M3, device="cuda")
    step = 1 << 17
    for i in range(0, rows, step):
        j = min(rows, i + step)
        out[i:j] = (
            torch.randn((j - i, K), device="cuda", generator=generator) * sigma
        ).clamp_(-448.0, 448.0).to(E4M3)
    return out


def _random_bf16(rows: int, sigma: float, generator: torch.Generator) -> torch.Tensor:
    return (torch.randn((rows, N), device="cuda", generator=generator) * sigma).to(
        torch.bfloat16
    )


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class K1536Fp8AddmmParityTest(unittest.TestCase):
    def setUp(self) -> None:
        fp8.clear_fp8_weight_cache()
        k1536._armed_logged = False
        k1536._decline_logged.clear()
        g = torch.Generator(device="cuda").manual_seed(20261009)
        self.generator = g
        self.weight = (torch.randn((K, N), device="cuda", generator=g) * 0.02).to(
            torch.bfloat16
        )
        self.scale_a = torch.full((1, 1), SCALE_A, device="cuda")

    def tearDown(self) -> None:
        fp8.clear_fp8_weight_cache()

    def _call(
        self,
        a: torch.Tensor,
        residual: torch.Tensor,
        *,
        custom: bool,
        weight: Optional[torch.Tensor] = None,
        scale_a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, List[str]]:
        lines: List[str] = []
        with (
            patch.object(fp8, "_HSTU_FP8_GEMM", True),
            patch.object(fp8, "_HSTU_FP8_FUSE_RESIDUAL", True),
            patch.object(fp8, "_HSTU_FP8_FWD_K1536", custom),
            patch.object(k1536, "_empty_output", side_effect=_nan_output),
            patch.object(k1536, "launch", wraps=k1536.launch) as launch,
            patch.object(k1536, "_print", side_effect=lines.append),
        ):
            out = fp8.scaled_addmm_fp8_preq(
                activation_fp8=a,
                scale_a=self.scale_a if scale_a is None else scale_a,
                weight=self.weight if weight is None else weight,
                bias_or_residual=residual,
                out_dtype=torch.bfloat16,
            )
            torch.cuda.synchronize()
            return out, launch.call_count, lines

    def _compare(self, got: torch.Tensor, ref: torch.Tensor, what: str) -> None:
        self.assertEqual(got.dtype, torch.bfloat16, what)
        self.assertEqual(tuple(got.shape), tuple(ref.shape), what)
        g = got.view(torch.int16)
        r = ref.view(torch.int16)
        total = g.numel()
        nan_got = int(((g & 0x7FFF) > 0x7F80).sum())
        nan_ref = int(((r & 0x7FFF) > 0x7F80).sum())
        diff = g != r
        n_diff = int(diff.sum())
        print(
            f"[k1536-test] {what}: bf16 words={total} mismatches={n_diff} "
            f"nan got/ref={nan_got}/{nan_ref}",
            flush=True,
        )
        if n_diff:
            row = int(torch.nonzero(diff.any(dim=1)).flatten()[0])
            cols = torch.nonzero(diff[row]).flatten()[:8].tolist()
            self.fail(
                f"{what}: {n_diff} of {total} words differ (first row {row}, cols "
                f"{cols}: got {g[row, cols].tolist()} ref {r[row, cols].tolist()})"
            )
        self.assertEqual(nan_ref, 0, f"{what}: reference produced NaN")

    def test_bitwise_vs_hipblaslt(self) -> None:
        armed = 0
        for m in M_VALUES:
            torch.cuda.empty_cache()
            a = _random_e4m3(m, SIG_A, self.generator)
            c = _random_bf16(m, 1.0, self.generator)
            ref, ref_launches, _ = self._call(a, c, custom=False)
            got, launches, lines = self._call(a, c, custom=True)
            self.assertEqual(ref_launches, 0)
            self.assertEqual(launches, 1, f"M={m}: custom path not taken: {lines}")
            self.assertFalse(any("declined" in s for s in lines), lines)
            armed += sum("armed: K=1536 N=512" in s for s in lines)
            self._compare(got, ref, f"M={m}")
            del a, c, ref, got
        self.assertEqual(armed, 1, "expected exactly one arming line")

    def test_edge_values_bitwise(self) -> None:
        # large scale and residual values in the bf16 rounding / overflow ranges,
        # saturated activations in the main tiles and in the tail rows
        g = torch.Generator(device="cuda").manual_seed(11)
        big = torch.full((1, 1), 64.0 * 1.0123, device="cuda")
        for m in (1000, 512):
            a = _random_e4m3(m, 1.0, g)
            a[8:16] = _random_e4m3(8, SIG_A, g)
            for base in (16, m - 16):
                a[base : base + 4] = torch.full((4, K), 448.0, device="cuda").to(E4M3)
                a[base + 4 : base + 8] = torch.full((4, K), -448.0, device="cuda").to(E4M3)
            c = _random_bf16(m, 3.0, g)
            c[:, 0:8] = torch.tensor(
                [0.0, -0.0, 3.0e38, -3.0e38, 1.0e-38, -1.0e-38, 65504.0, -65504.0],
                device="cuda",
            ).to(torch.bfloat16)
            for scale in (big, self.scale_a):
                ref, _, _ = self._call(a, c, custom=False, scale_a=scale)
                got, launches, lines = self._call(a, c, custom=True, scale_a=scale)
                self.assertEqual(launches, 1, lines)
                g16, r16 = got.view(torch.int16), ref.view(torch.int16)
                # inf/NaN only where both sides produce them identically
                self.assertTrue(torch.equal(g16, r16), f"edge M={m} scale={float(scale)}")

    def test_unsupported_shape_declines_to_hipblaslt(self) -> None:
        a = _random_e4m3(1000, SIG_A, self.generator)
        c = (torch.randn((1000, 256), device="cuda", generator=self.generator)).to(
            torch.bfloat16
        )
        w = (torch.randn((K, 256), device="cuda", generator=self.generator) * 0.02).to(
            torch.bfloat16
        )
        ref, _, _ = self._call(a, c, custom=False, weight=w)
        got, launches, lines = self._call(a, c, custom=True, weight=w)
        self.assertEqual(launches, 0)
        self.assertTrue(torch.equal(got.view(torch.int16), ref.view(torch.int16)))
        self.assertEqual(sum("declined" in s for s in lines), 1, lines)

    def test_prebuild(self) -> None:
        lines: List[str] = []
        with patch.object(k1536, "_print", side_effect=lines.append):
            self.assertTrue(k1536.prebuild())
        self.assertTrue(
            any("prebuilt: K=1536 N=512" in s and "self-check M=257 OK" in s for s in lines),
            lines,
        )


if __name__ == "__main__":
    unittest.main()

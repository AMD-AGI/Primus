#!/usr/bin/env python3

# pyre-unsafe

"""Bit parity of the DLRM_HSTU_FP8_FWD_K512 forward GEMM against hipBLASLt.

Both sides run through triton_fp8.scaled_mm_fp8_out_preq, the production entry
point, with the real _get_weight_shadow (e4m3 [N, 512] row-major shadow seen
transposed) and the input_scale fold of the output scale. Only the module-level
knob _HSTU_FP8_FWD_K512 is toggled. Every output byte is compared. The custom
path's output buffer is prefilled with 0xFF (e4m3 NaN), so an unwritten row
cannot pass.

Production layouts over a [512, 2048] uvqk weight (U | V | Q | K):
  N=1536: V/Q/K, weight_layout ((512, 2048),)
          (triton_hstu_preprocess_and_attention.py, fused QKV forward)
  N=1024: targets V/K, weight_layout ((512, 1024), (1536, 2048)), bias cat(b_v, b_k)
Model-like data, as in the iter8b eprobe "model" arm: e4m3 activation of
N(0, 32), weight N(0, 0.02) quantized by the production shadow, and non
power-of-two scales, so acc * (sa * sb) really rounds.

Needs a gfx950 GPU and /opt/rocm/bin/hipcc; skips otherwise. Run (kernel lab): lab_py.sh <worktree root> -m pytest -q -s
generative_recommenders/ops/tests/k512_fp8_gemm_test.py
"""

from __future__ import annotations

import os
import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import torch

from generative_recommenders.ops.hip import k512_fp8_gemm as k512
from generative_recommenders.ops.triton import triton_fp8 as fp8

E4M3 = getattr(torch, "float8_e4m3fn", None)
D = 512
UVQK_COLUMNS = 2048
LAYOUTS: Dict[int, Tuple[Tuple[int, int], ...]] = {
    1536: ((512, 2048),),
    1024: ((512, 1024), (1536, 2048)),
}
# Not multiples of 256 except 256 itself; 1/255/300 exercise only the tail path.
M_VALUES: Tuple[int, ...] = (1, 255, 256, 257, 300, 1000, 300001, 2284545, 2376002)
M_OVER_4GIB = 2900001  # N=1536: M*N = 4.45e9 output bytes > 2^32
SIG_A = 32.0
SCALE_A = 0.0316  # ~1/SIG_A, not a power of two
OUTPUT_SCALE = 0.0083  # ~amax(qkv) * 1.5 / 448 for this data, not a power of two


def _skip_reason() -> Optional[str]:
    if E4M3 is None or torch.version.hip is None or not torch.cuda.is_available():
        return "needs ROCm torch with float8_e4m3fn and a GPU"
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"needs gfx950 (found {arch or 'unknown'})"
    if not os.path.exists(k512.HIPCC):
        return f"needs {k512.HIPCC}"
    return None


_SKIP = _skip_reason()


def _nan_output(m: int, n: int, device: torch.device) -> torch.Tensor:
    return torch.full((m, n), 0xFF, dtype=torch.uint8, device=device).view(E4M3)


def _random_e4m3(rows: int, sigma: float, generator: torch.Generator) -> torch.Tensor:
    out = torch.empty((rows, D), dtype=E4M3, device="cuda")
    step = 1 << 18
    for i in range(0, rows, step):
        j = min(rows, i + step)
        out[i:j] = (
            torch.randn((j - i, D), device="cuda", generator=generator) * sigma
        ).clamp_(-448.0, 448.0).to(E4M3)
    return out


def _bias_for(
    layout: Tuple[Tuple[int, int], ...], bias_full: torch.Tensor
) -> torch.Tensor:
    # VQK passes uvqk_bias[u:], targets V/K torch.cat((b_v, b_k)): the same values
    if len(layout) == 1:
        return bias_full[layout[0][0] : layout[0][1]]
    return torch.cat([bias_full[s:e] for s, e in layout], dim=0)


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class K512Fp8GemmParityTest(unittest.TestCase):
    def setUp(self) -> None:
        fp8.clear_fp8_weight_cache()
        k512._armed_logged.clear()
        k512._decline_logged.clear()
        g = torch.Generator(device="cuda").manual_seed(20261008)
        self.generator = g
        self.weight = (
            torch.randn((D, UVQK_COLUMNS), device="cuda", generator=g) * 0.02
        ).to(torch.bfloat16)
        bias = torch.randn((UVQK_COLUMNS,), device="cuda", generator=g) * 0.05
        # a few columns well past the e4m3 range (saturate for most rows)
        bias[512:520] = torch.tensor([9.0, -9.0, 4.0, -4.0, 3.7, -3.7, 30.0, -30.0])
        bias[1536:1540] = torch.tensor([5.0, -5.0, 3.72, -3.72])
        self.bias_full = bias
        self.scale_a = torch.full((1, 1), SCALE_A, device="cuda")
        self.output_scale = torch.full((1, 1), OUTPUT_SCALE, device="cuda")

    def tearDown(self) -> None:
        fp8.clear_fp8_weight_cache()

    # ------------------------------------------------------------------ helpers
    def _call(
        self,
        a: torch.Tensor,
        n: int,
        *,
        custom: bool,
        bias_full: Optional[torch.Tensor] = None,
        scale_a: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        layout: Optional[Tuple[Tuple[int, int], ...]] = None,
    ) -> Tuple[torch.Tensor, int, List[str]]:
        bias_full = self.bias_full if bias_full is None else bias_full
        scale_a = self.scale_a if scale_a is None else scale_a
        output_scale = self.output_scale if output_scale is None else output_scale
        layout = LAYOUTS[n] if layout is None else layout
        lines: List[str] = []
        with (
            patch.object(fp8, "_HSTU_FP8_GEMM", True),
            patch.object(fp8, "_HSTU_FP8_FWD_K512", custom),
            patch.object(k512, "_empty_output", side_effect=_nan_output),
            patch.object(k512, "launch", wraps=k512.launch) as launch,
            patch.object(k512, "_print", side_effect=lines.append),
        ):
            out = fp8.scaled_mm_fp8_out_preq(
                activation_fp8=a,
                scale_a=scale_a,
                weight=self.weight,
                bias=_bias_for(layout, bias_full),
                weight_layout=layout,
                output_scale=output_scale,
            )
            torch.cuda.synchronize()
            return out, launch.call_count, lines

    def _compare(self, got: torch.Tensor, ref: torch.Tensor, what: str) -> None:
        self.assertEqual(got.dtype, E4M3, what)
        self.assertEqual(tuple(got.shape), tuple(ref.shape), what)
        g = got.view(torch.uint8)
        r = ref.view(torch.uint8)
        total = g.numel()
        nan_got = int(((g & 0x7F) == 0x7F).sum())
        nan_ref = int(((r & 0x7F) == 0x7F).sum())
        sat_ref = int(((r & 0x7F) == 0x7E).sum())
        zero_ref = int(((r & 0x7F) == 0).sum())
        diff = g != r
        n_diff = int(diff.sum())
        print(
            f"[k512-test] {what}: bytes={total} mismatches={n_diff} "
            f"exact_frac={(total - n_diff) / total:.9f} nan_bytes got/ref="
            f"{nan_got}/{nan_ref} ref_sat_frac={sat_ref / total:.4g} "
            f"ref_zero_frac={zero_ref / total:.4g}",
            flush=True,
        )
        if n_diff:
            bad_rows = torch.nonzero(diff.any(dim=1)).flatten()
            row = int(bad_rows[0])
            cols = torch.nonzero(diff[row]).flatten()[:8].tolist()
            self.fail(
                f"{what}: {n_diff} of {total} output bytes differ in "
                f"{bad_rows.numel()} rows (first row {row}, cols {cols}: got "
                f"{g[row, cols].tolist()} ref {r[row, cols].tolist()}); "
                f"NaN bytes got {nan_got} vs ref {nan_ref}"
            )
        self.assertEqual(nan_ref, 0, f"{what}: reference produced NaN bytes")

    # ------------------------------------------------------------------ tests
    def test_bitwise_vs_hipblaslt(self) -> None:
        cases = [(m, n) for m in M_VALUES for n in (1536, 1024)]
        cases.append((M_OVER_4GIB, 1536))
        armed: Dict[int, int] = {}
        a = None
        a_rows = -1
        for m, n in cases:
            if m != a_rows:
                a = None
                torch.cuda.empty_cache()
                a = _random_e4m3(m, SIG_A, self.generator)
                a_rows = m
            assert a is not None
            ref, ref_launches, _ = self._call(a, n, custom=False)
            got, launches, lines = self._call(a, n, custom=True)
            self.assertEqual(ref_launches, 0)
            self.assertEqual(launches, 1, f"N={n} M={m}: custom path not taken: {lines}")
            armed[n] = armed.get(n, 0) + sum(f"armed: N={n} " in s for s in lines)
            self.assertFalse(any("declined" in s for s in lines), lines)
            self._compare(got, ref, f"N={n} M={m}")
            del ref, got
        for n in (1536, 1024):
            self.assertEqual(armed.get(n), 1, f"expected exactly one arming line for N={n}")

    def test_edge_values_bitwise(self) -> None:
        # output_scale 1 keeps the effective bias exact; zero activation rows then
        # give e4m3_rne(sat448(bias)). The bias carries the edge values: +-0, e4m3
        # subnormal ties, normal RNE ties, around +-448/+-464, huge finite.
        edges = [
            0.0, -0.0, 2.0**-10, -(2.0**-10), 3 * 2.0**-10, -3 * 2.0**-10, 5 * 2.0**-11,
            2.0**-9, -(2.0**-9), 2.0**-6, 1.0625, -1.0625, 1.1875, -1.1875, 240.0,
            248.0, -248.0, 272.0, 440.0, -444.0, 448.0, -448.0, 456.0, -456.0, 464.0,
            -464.0, 466.0, -466.0, 480.0, -480.0, 10240.0, -10240.0, 2.0**127 * 1.5,
            -(2.0**127) * 1.5, 3.3895313892515355e38, -3.3895313892515355e38,
        ]
        edge_t = torch.tensor(edges, dtype=torch.float32)
        self.assertTrue(torch.equal(edge_t.to(torch.bfloat16).float(), edge_t))
        g = torch.Generator(device="cuda").manual_seed(7)
        for n in (1536, 1024):
            bias_full = torch.randn((UVQK_COLUMNS,), device="cuda", generator=g) * 3
            for start in (512, 1536):  # V and K columns (in both layouts)
                bias_full[start : start + len(edges)] = edge_t.cuda()
            for m in (1000, 512):
                rows = []
                kinds = [("zero", 0.0), ("pos448", 448.0), ("neg448", -448.0)]
                for kind, value in kinds:  # in the main tiles and in the tail rows
                    rows.append((kind, value, 4))
                a = _random_e4m3(m, 1.0, g)
                a[8:16] = _random_e4m3(8, SIG_A, g)
                for idx, (kind, value, count) in enumerate(rows):
                    for base in (16 + 4 * idx, m - 16 + 4 * idx):
                        a[base : base + count] = torch.full(
                            (count, D), value, device="cuda"
                        ).to(E4M3)
                big_scale_a = torch.full((1, 1), 64.0 * 1.0123, device="cuda")
                one = torch.ones((1, 1), device="cuda")
                ref, _, _ = self._call(
                    a, n, custom=False, bias_full=bias_full, scale_a=big_scale_a,
                    output_scale=one,
                )
                got, launches, lines = self._call(
                    a, n, custom=True, bias_full=bias_full, scale_a=big_scale_a,
                    output_scale=one,
                )
                self.assertEqual(launches, 1, lines)
                self._compare(got, ref, f"edge N={n} M={m}")

    def test_unsupported_shape_declines_to_hipblaslt(self) -> None:
        a = _random_e4m3(1000, SIG_A, self.generator)
        layout = ((0, 512),)  # the U slice: N=512 is not built
        ref, _, _ = self._call(a, 512, custom=False, layout=layout)
        got, launches, lines = self._call(a, 512, custom=True, layout=layout)
        self.assertEqual(launches, 0)
        self.assertTrue(torch.equal(got.view(torch.uint8), ref.view(torch.uint8)))
        self.assertEqual(sum("declined" in s for s in lines), 1, lines)

    def test_public_entry_rejects_bad_operands(self) -> None:
        a = _random_e4m3(4097, SIG_A, self.generator)
        w = _random_e4m3(1536, 90.0, self.generator)
        bias = (torch.randn((1536,), device="cuda", generator=self.generator) * 4).to(
            torch.bfloat16
        )
        sa = torch.full((1, 1), 0.0312, device="cuda")
        sb = torch.full((1, 1), 0.00049, device="cuda")
        with self.assertRaises(ValueError):
            k512.k512_gemm_e4m3(a[:, :256], w, bias, sa, sb)  # K != 512
        with self.assertRaises(ValueError):
            k512.k512_gemm_e4m3(a, w.t().contiguous().t(), bias, sa, sb)  # B layout
        with self.assertRaises(ValueError):
            k512.k512_gemm_e4m3(a, w, bias.float(), sa, sb)  # bias dtype
        with patch.object(k512, "_empty_output", side_effect=_nan_output):
            out = k512.k512_gemm_e4m3(a, w, bias, sa, sb)
        ref = fp8._unwrap_scaled_mm(
            torch._scaled_mm(
                a, w.t(), scale_a=sa, scale_b=sb, bias=bias, out_dtype=E4M3,
                use_fast_accum=False,
            )
        )
        self._compare(out, ref, "public entry N=1536 M=4097")


if __name__ == "__main__":
    unittest.main()

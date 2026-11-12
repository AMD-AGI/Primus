# pyre-strict
"""GPU checks of the Indexed-X L1 kernels (ops/triton/triton_ec_indexed_l1.py; DLRM_HSTU_FP8_EC_INDEXED_L1).

Skipped unless a ROCm gfx950 GPU and Triton are present (the tiles are validated there only):

    python -B -m pytest -q generative_recommenders/ops/tests/ec_indexed_l1_test.py

  * prebuild(): its exact integer-valued self-check passes;
  * exact integer-valued inputs ({-1, 0, 1}: every product and every fp32 partial sum is an exact integer, so the
    summation order cannot matter) at edge M (one row, masked forward blocks, one / many splits, a masked tail):
    h and dW1 raw-bit equal to the exact fp64 reference rounded once;
  * random inputs: h against the shipped torch.addmm on the materialized X and dW1 against the shipped
    (dh.t() @ X).to(fp32): the declared fp32 reduction-order class (few differing elements, the same SNR vs an
    fp64 oracle), deterministic run to run;
  * malformed inputs raise IndexedL1Unsupported (the caller's fail-closed decline) and launch nothing;
  * item (83) (DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT): the after-dot kernel _ecix_wgrad (ON) and ship14's IDXPF
    kernel _ecix_wgrad_idxpf (OFF) give raw-bit equal dW1 (and fp32 split-K partials P) on random and edge inputs.
"""

import math
import unittest
from typing import Tuple

import torch

try:
    from generative_recommenders.ops.triton import triton_ec_indexed_l1 as K
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"triton_ec_indexed_l1 not importable: {exc}")


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None or K._KERNELS_ERROR is not None:
        return False
    return str(torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName).startswith("gfx950")


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous()
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def _snr_db(ref: torch.Tensor, out: torch.Tensor) -> float:
    err = out.double() - ref
    se = float((err * err).sum())
    return math.inf if se == 0 else 10.0 * math.log10(float((ref * ref).sum()) / se)


@unittest.skipUnless(_gfx950(), "requires a ROCm gfx950 GPU with Triton")
class ECIndexedL1KernelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dev = torch.device("cuda", torch.cuda.current_device())
        # item (83): pin the gin default (ON, _ecix_wgrad) so the accuracy tests cover the shipped kernel even
        # without the gin bootstrap (absent env = OFF in code); the ON-vs-OFF test sets / restores it per call.
        self._old_after_dot = K._WGRAD_IDX_AFTER_DOT
        K._WGRAD_IDX_AFTER_DOT = True

    def tearDown(self) -> None:
        K._WGRAD_IDX_AFTER_DOT = self._old_after_dot

    def _exact(self, nh: int, m: int, n_rows: int) -> None:
        table, idx, w1, b1, dh = K._prebuild_inputs(nh, m, n_rows)
        h_ref, dw_ref = K._prebuild_reference(table, idx, nh, w1, b1, dh)

        def bf(x: torch.Tensor) -> torch.Tensor:
            return x.to(torch.bfloat16).to(self.dev)

        h = K.ecix_fwd(bf(table), idx.to(self.dev), nh, bf(w1), bf(b1))
        dw = K.ecix_wgrad(bf(dh), bf(table), idx.to(self.dev), nh, out_dtype=torch.float32)
        self.assertEqual(int((_bits(h.cpu()) != _bits(h_ref)).sum()), 0, f"h nh={nh} M={m}")
        self.assertEqual(int((_bits(dw.cpu()) != _bits(dw_ref)).sum()), 0, f"dW1 nh={nh} M={m}")

    def test_prebuild_exact(self) -> None:
        self.assertIn("prebuild OK", K.prebuild(self.dev))

    def test_exact_edges(self) -> None:
        # M = 1, a partly masked forward block, exactly one / more than one BM block, one split, many splits with a
        # masked tail, and a size that is no multiple of 16 or 128.
        for nh in (1, 2):
            for m in (1, 37, 64, 129, 4_099, 20_100, 70_001):
                self._exact(nh, m, 96)

    def test_random_vs_shipped_class_and_determinism(self) -> None:
        g = torch.Generator(device=self.dev).manual_seed(5)
        U, M = 5_003, 50_021
        table = (torch.randn(U, K.WIDTH, generator=g, device=self.dev)
                 * torch.exp(0.7 * torch.randn(U, 1, generator=g, device=self.dev)) * 0.05).to(torch.bfloat16)
        for nh in (1, 2):
            kt = nh * K.WIDTH
            idx = torch.randint(0, U, (M * nh,), generator=g, device=self.dev)
            w1 = (torch.randn(K.HID, kt, generator=g, device=self.dev) * (2.0 / (kt + K.HID)) ** 0.5).to(
                torch.bfloat16)
            b1 = (torch.randn(K.HID, generator=g, device=self.dev) * 0.02).to(torch.bfloat16)
            dh = (torch.randn(M, K.HID, generator=g, device=self.dev) * 1e-3).to(torch.bfloat16)
            x = K.materialize_x(table, idx, nh)
            h, h_ship = K.ecix_fwd(table, idx, nh, w1, b1), K.reference_fwd(table, idx, nh, w1, b1)
            dw, dw_ship = K.ecix_wgrad(dh, table, idx, nh), K.reference_wgrad(dh, table, idx, nh)
            h64 = x.double() @ w1.double().t() + b1.double()
            dw64 = dh.double().t() @ x.double()
            for name, a, b, ref, frac in (("h", h, h_ship, h64, 1e-3), ("dW1", dw, dw_ship, dw64, 2e-2)):
                ndiff = int((_bits(a) != _bits(b)).sum())
                self.assertLessEqual(ndiff, frac * a.numel(), f"{name} nh={nh}: {ndiff} elements differ")
                self.assertGreaterEqual(_snr_db(ref, a), _snr_db(ref, b) - 0.5, f"{name} nh={nh} SNR")
                self.assertTrue(bool(torch.isfinite(a).all()), name)
            # deterministic: no atomics, the split rule depends on (M, nh) only, fixed-order reduce
            self.assertTrue(torch.equal(_bits(K.ecix_fwd(table, idx, nh, w1, b1)), _bits(h)))
            self.assertTrue(torch.equal(_bits(K.ecix_wgrad(dh, table, idx, nh)), _bits(dw)))
            self.assertIsNone(K.check_index_range(idx, U))
            self.assertIsNotNone(K.check_index_range(idx, int(idx.max())))

    def _wgrad_order(self, after_dot: bool, dh: torch.Tensor, table: torch.Tensor, idx: torch.Tensor,
                     nh: int, ws: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
        old = K._WGRAD_IDX_AFTER_DOT
        K._WGRAD_IDX_AFTER_DOT = after_dot  # the flag wgrad_idx_after_dot() caches (env read once per process)
        try:
            ws.fill_(float("nan"))
            dw = K.ecix_wgrad(dh, table, idx, nh, workspace=ws)
            torch.cuda.synchronize(self.dev)
            name = str(K.LAST_KERNELS["wgrad%d" % nh].name)
        finally:
            K._WGRAD_IDX_AFTER_DOT = old
        s = K.nsplit(idx.numel() // nh, nh)[1]
        return dw, ws[: s * K.HID * nh * K.WIDTH].clone(), name

    def test_wgrad_idx_after_dot_on_off_bitwise(self) -> None:
        # item (83): ON (_ecix_wgrad) vs OFF (_ecix_wgrad_idxpf, ship14 c68395d order): same values, same tl.dot
        # sequence -> raw-bit equal P partials and dW1 (random inputs; one split, many splits, masked tails).
        g = torch.Generator(device=self.dev).manual_seed(83)
        U = 5_003
        table = (torch.randn(U, K.WIDTH, generator=g, device=self.dev) * 0.05).to(torch.bfloat16)
        for nh in (1, 2):
            for m in (1, 63, 64, 129, 20_100, 50_021):
                idx = torch.randint(0, U, (m * nh,), generator=g, device=self.dev)
                dh = (torch.randn(m, K.HID, generator=g, device=self.dev) * 1e-3).to(torch.bfloat16)
                ws = torch.empty((K.workspace_numel(m, nh),), dtype=torch.float32, device=self.dev)
                dw_on, p_on, n_on = self._wgrad_order(True, dh, table, idx, nh, ws)
                dw_off, p_off, n_off = self._wgrad_order(False, dh, table, idx, nh, ws)
                self.assertTrue("_ecix_wgrad" in n_on and "idxpf" not in n_on, n_on)
                self.assertIn("_ecix_wgrad_idxpf", n_off)
                self.assertTrue(torch.equal(_bits(p_on), _bits(p_off)), f"P nh={nh} M={m}")
                self.assertTrue(torch.equal(_bits(dw_on), _bits(dw_off)), f"dW1 nh={nh} M={m}")
        # the exact integer-valued edge check passes with either order
        for after_dot in (True, False):
            old = K._WGRAD_IDX_AFTER_DOT
            K._WGRAD_IDX_AFTER_DOT = after_dot
            try:
                for nh in (1, 2):
                    self._exact(nh, 20_100, 96)
            finally:
                K._WGRAD_IDX_AFTER_DOT = old

    def test_malformed_inputs_decline(self) -> None:
        dev = self.dev
        table = torch.zeros(96, K.WIDTH, dtype=torch.bfloat16, device=dev)
        idx = torch.zeros(256, dtype=torch.int64, device=dev)
        w1 = torch.zeros(K.HID, K.WIDTH, dtype=torch.bfloat16, device=dev)
        b1 = torch.zeros(K.HID, dtype=torch.bfloat16, device=dev)
        dh = torch.zeros(256, K.HID, dtype=torch.bfloat16, device=dev)
        bad_fwd = [
            (table.float(), idx, 1, w1, b1),  # fp32 table
            (table[:, :256], idx, 1, w1, b1),  # width
            (table, idx.to(torch.int32), 1, w1, b1),  # int32 idx
            (table, idx[::2], 1, w1, b1),  # strided idx
            (table, idx.cpu(), 1, w1, b1),  # idx on another device
            (table, idx, 3, w1, b1),  # nh
            (table, idx, 2, w1, b1),  # W1 [256, 512] for the cat member
            (table, idx, 1, w1.float(), b1),  # W1 not the autocast cast
            (table, idx, 1, w1, b1.float()),  # b1 not the autocast cast
            (table, idx, 1, w1.t().contiguous().t(), b1),  # W1 column-major
        ]
        for i, a in enumerate(bad_fwd):
            with self.assertRaises(K.IndexedL1Unsupported, msg=f"fwd case {i}"):
                K.ecix_fwd(*a)
        for i, (a, kw) in enumerate((((dh[:128], table, idx, 1), {}),  # dh rows != M
                                     ((dh.float(), table, idx, 1), {}),  # dh dtype
                                     ((dh, table, idx, 1), dict(out_dtype=torch.float16)))):  # out dtype
            with self.assertRaises(K.IndexedL1Unsupported, msg=f"wgrad case {i}"):
                K.ecix_wgrad(*a, **kw)


if __name__ == "__main__":
    unittest.main()

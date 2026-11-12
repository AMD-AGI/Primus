#!/usr/bin/env python3

# pyre-unsafe

"""Bit parity of the K=512 FP8 forward GEMM's tiled-A entry against its row-major entry.

Item B15T(L2) (Arbor P7 iteration 20). The tiled-A entry reads A from the step-tiled
token-major yT: e4m3 [ceil(M/128), 512, 128], with token t / feature c at
((t // 128) * 512 + c) * 128 + t % 128 and zero tail tokens. That is the layout of
DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED (the LayerNorm store_t v2 with YT_TILED). The
row-major entry reads y [M, 512]. Every output byte must be equal.

- Outputs are prefilled with 0xFF (e4m3 NaN), so an unwritten byte cannot pass.
- 256 guard rows after row M (0x5A) must survive.
- Two non-power-of-two scale pairs.
- M in {1, 127, 128, 129, 1000, 2376002, 2650962, 2900001}; N=1024 (the targets V/K
  call) and N=1536.
- Further cases: a NaN-filled tail, a 10-repeat stress, and the production producer's
  tiled yT (store_t v2).
- The startup self-check: it passes, and it latches off on a mismatch.
- Operand checks.
- The store-data WAR guard of the compiled tiled kernel (the P7 iteration-9 G1-A
  hazard).

Needs a gfx950 GPU and /opt/rocm/bin/hipcc; skips otherwise. Run (kernel lab):
lab_py.sh <worktree root> -m pytest -q -s generative_recommenders/ops/tests/k512_tiled_a_test.py
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import torch

from generative_recommenders.ops.hip import k512_fp8_gemm as k512

E4M3 = getattr(torch, "float8_e4m3fn", None)
U8 = torch.uint8
D = 512
STEP = 128
M_VALUES: Tuple[int, ...] = (1, 127, 128, 129, 1000, 2376002, 2650962, 2900001)
NS: Tuple[int, ...] = (1024, 1536)
GUARD_ROWS = 256
SIG_A, SIG_B, OUT_MULT = 32.0, 90.0, 32.0
# (scale_a, scale_b), neither a power of two: the iteration-9 harness pair (outputs
# spread over the e4m3 range with some saturation) and a smaller pair (no saturation)
SCALE_PAIRS: Tuple[Tuple[float, float], ...] = (
    ((1.0 / (SIG_A * D**0.25)) * OUT_MULT, 1.0 / (SIG_B * D**0.25)),
    (0.0123, 0.00317),
)
STRESS_M = 2376002
STRESS_REPS = 10
WAR_WAIT_STATES = 16


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


def _rand_e4m3(rows: int, cols: int, sigma: float, g: torch.Generator) -> torch.Tensor:
    out = torch.empty((rows, cols), dtype=E4M3, device="cuda")
    step = max(1, (1 << 28) // cols)
    for i in range(0, rows, step):
        j = min(rows, i + step)
        out[i:j] = (
            (torch.randn((j - i, cols), device="cuda", generator=g) * sigma)
            .clamp_(-448.0, 448.0)
            .to(E4M3)
        )
    return out


def _tile(y: torch.Tensor, m: int, tail: int = 0) -> torch.Tensor:
    """Step-tiled copy [ceil(m/128), 512, 128] of y[:m]; tail tokens = `tail` bytes."""
    blocks = (m + STEP - 1) // STEP
    padded = torch.full((blocks * STEP, D), tail, dtype=U8, device=y.device)
    padded[:m] = y[:m].view(U8)
    out = torch.empty((blocks, D, STEP), dtype=U8, device=y.device)
    chunk = 8192
    for b0 in range(0, blocks, chunk):
        b1 = min(blocks, b0 + chunk)
        out[b0:b1] = padded[b0 * STEP : b1 * STEP].view(b1 - b0, STEP, D).transpose(1, 2)
    del padded
    return out.view(E4M3)


def _scales(pair: Tuple[float, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    sa = torch.full((1, 1), pair[0], dtype=torch.float32, device="cuda")
    sb = torch.full((1, 1), pair[1], dtype=torch.float32, device="cuda")
    return sa, sb


def _guarded(m: int, n: int) -> torch.Tensor:
    buf = torch.full((m + GUARD_ROWS, n), 0xFF, dtype=U8, device="cuda")
    buf[m:].fill_(0x5A)
    return buf


def _bytes_differ(a: torch.Tensor, b: torch.Tensor) -> int:
    n = 0
    rows = max(1, (1 << 28) // max(1, a.shape[1]))
    for i in range(0, a.shape[0], rows):
        n += int((a[i : i + rows] != b[i : i + rows]).sum().item())
    return n


@unittest.skipIf(_SKIP is not None, _SKIP or "")
class K512TiledAParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        k512.reset_tiled_a_for_testing()
        g = torch.Generator(device="cuda").manual_seed(20261009)
        cls.g = g
        cls.y = _rand_e4m3(max(M_VALUES), D, SIG_A, g)
        cls.w: Dict[int, torch.Tensor] = {}
        cls.bias: Dict[int, torch.Tensor] = {}
        for n in NS:
            cls.w[n] = _rand_e4m3(n, D, SIG_B, g)
            b = (torch.randn((n,), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
            cls.bias[n] = (b.float() * OUT_MULT).to(torch.bfloat16)
        torch.cuda.synchronize()

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.y, cls.w, cls.bias
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers
    def _reference(self, m: int, n: int, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
        ref = torch.full((m, n), 0xFF, dtype=U8, device="cuda")
        k512.launch(k512.kernel(n), self.y[:m], self.w[n], self.bias[n], sa, sb, out=ref.view(E4M3))
        torch.cuda.synchronize()
        bad = int(((ref & 0x7F) == 0x7F).sum().item())
        self.assertEqual(bad, 0, f"row-major reference N={n} M={m}: {bad} NaN/unwritten bytes")
        return ref

    def _check(self, got_buf: torch.Tensor, ref: torch.Tensor, m: int, what: str) -> None:
        n = ref.shape[1]
        differ = _bytes_differ(got_buf[:m], ref)
        guard_bad = int((got_buf[m:] != 0x5A).sum().item())
        print(
            f"[k512-tiled-a-test] {what}: bytes={m * n} mismatches={differ} "
            f"guard_bytes_changed={guard_bad}",
            flush=True,
        )
        if differ:
            diff = got_buf[:m] != ref
            rows = torch.nonzero(diff.any(dim=1)).flatten()
            row = int(rows[0])
            cols = torch.nonzero(diff[row]).flatten()[:8].tolist()
            self.fail(
                f"{what}: {differ} bytes differ in {rows.numel()} rows (first row {row}, "
                f"cols {cols}: got {got_buf[row, cols].tolist()} ref {ref[row, cols].tolist()})"
            )
        self.assertEqual(guard_bad, 0, f"{what}: guard rows after M were written")

    # ------------------------------------------------------------------ tests
    def test_bitwise_vs_row_major_entry(self) -> None:
        for m in sorted(M_VALUES, reverse=True):
            yt = _tile(self.y, m)
            # The tail case: tokens [M, Mp) are NaN instead of zero (must not leak).
            yt_nan = _tile(self.y, m, tail=0x7F) if m % STEP else None
            for n in NS:
                for si, pair in enumerate(SCALE_PAIRS):
                    sa, sb = _scales(pair)
                    ref = self._reference(m, n, sa, sb)
                    for tail_name, a in (("zero", yt), ("nan", yt_nan)):
                        if a is None or (tail_name == "nan" and si):
                            continue
                        buf = _guarded(m, n)
                        out = k512.k512_scaled_mm_tiled_a(
                            a, m, self.w[n], self.bias[n], sa, sb, out=buf[:m].view(E4M3)
                        )
                        self.assertEqual(out.data_ptr(), buf.data_ptr())
                        torch.cuda.synchronize()
                        self._check(buf, ref, m, f"N={n} M={m} scales#{si} tail={tail_name}")
                        del buf
                    del ref
            del yt, yt_nan
            torch.cuda.empty_cache()
        # the public entry allocates its own output too
        m = 1000
        sa, sb = _scales(SCALE_PAIRS[0])
        got = k512.k512_scaled_mm_tiled_a(_tile(self.y, m), m, self.w[1024], self.bias[1024], sa, sb)
        self.assertEqual((got.dtype, tuple(got.shape)), (E4M3, (m, 1024)))
        self.assertEqual(_bytes_differ(got.view(U8), self._reference(m, 1024, sa, sb)), 0)

    def test_stress_repeat(self) -> None:
        m = STRESS_M
        yt = _tile(self.y, m)
        sa, sb = _scales(SCALE_PAIRS[1])
        for n in NS:
            ref = self._reference(m, n, sa, sb)
            kern = k512.kernel(n)
            self.assertTrue(k512.tiled_a_ready(n, build=True))
            out = torch.empty((m, n), dtype=U8, device="cuda")
            bad_reps = bad_bytes = 0
            for _ in range(STRESS_REPS):
                out.fill_(0xFF)
                k512.launch_tiled_a(kern, yt, m, self.w[n], self.bias[n], sa, sb, out=out.view(E4M3))
                torch.cuda.synchronize()
                differ = _bytes_differ(out, ref)
                bad_reps += differ > 0
                bad_bytes += differ
            print(
                f"[k512-tiled-a-test] stress N={n} M={m}: reps={STRESS_REPS} "
                f"bad_reps={bad_reps} bad_bytes={bad_bytes}",
                flush=True,
            )
            self.assertEqual(bad_reps, 0)
            del ref, out
        del yt
        torch.cuda.empty_cache()

    def test_production_store_t_v2_tiled_producer(self) -> None:
        # The LayerNorm store_t v2 kernel with YT_TILED (DLRM_HSTU_FP8_BWD_STORE_T_V2
        # + wgrad_ck tiled_xt) writes y [M, 512] and the step-tiled yT: the GEMM on
        # its yT must equal the GEMM on its y.
        from generative_recommenders.ops.triton import triton_layer_norm as tln

        g = torch.Generator(device="cuda").manual_seed(7)
        ln_w = (1.0 + 0.1 * torch.randn((D,), device="cuda", generator=g)).to(torch.bfloat16)
        ln_b = (0.1 * torch.randn((D,), device="cuda", generator=g)).to(torch.bfloat16)
        sa, sb = _scales(SCALE_PAIRS[1])
        for m in (1000, STRESS_M):
            x = torch.randn((m, D), device="cuda", generator=g).to(torch.bfloat16)
            with patch.object(tln, "_STORE_T_V2", True):
                y, _, _, yt = tln.triton_weighted_layer_norm_fwd_fp8_store_t(
                    x, ln_w, ln_b, 1e-6, 40.0, tiled_xt=True
                )
            self.assertEqual(tuple(yt.shape), k512.tiled_a_shape(m), "producer did not tile")
            self.assertTrue(torch.equal(k512.tiled_a_to_row_major(yt, m).view(U8), y.view(U8)))
            if m % STEP:
                tail = yt.view(U8)[-1, :, m % STEP :]
                self.assertEqual(int(tail.count_nonzero().item()), 0, "producer tail not zero")
            for n in (1024,) if m > 1000 else NS:
                ref = torch.full((m, n), 0xFF, dtype=U8, device="cuda")
                k512.launch(k512.kernel(n), y, self.w[n], self.bias[n], sa, sb, out=ref.view(E4M3))
                buf = _guarded(m, n)
                k512.k512_scaled_mm_tiled_a(
                    yt, m, self.w[n], self.bias[n], sa, sb, out=buf[:m].view(E4M3)
                )
                torch.cuda.synchronize()
                self.assertEqual(int(((ref & 0x7F) == 0x7F).sum().item()), 0)
                self._check(buf, ref, m, f"store_t_v2 producer N={n} M={m}")
                del ref, buf
            del x, y, yt
            torch.cuda.empty_cache()

    def test_selfcheck_and_latch(self) -> None:
        lines: List[str] = []
        try:
            k512.reset_tiled_a_for_testing()
            with patch.object(k512, "_print", side_effect=lines.append):
                self.assertTrue(k512.prebuild_tiled_a(NS))
            for n in NS:
                self.assertTrue(k512.tiled_a_ready(n))
                self.assertEqual(sum(f"tiled-A self-check OK: N={n} " in s for s in lines), 1, lines)
            # A wrong entry (the row-major kernel reading the tiled bytes) must fail the
            # self-check, latch the entry off, and log exactly once.
            k512.reset_tiled_a_for_testing()
            lines.clear()
            k512._tiled_a_fns[1024] = k512.kernel(1024).fn
            with patch.object(k512, "_print", side_effect=lines.append):
                self.assertFalse(k512.tiled_a_selfcheck(1024))
                self.assertFalse(k512.tiled_a_ready(1024, build=True))
                with self.assertRaises(k512.PrebuildError):
                    k512.prebuild_tiled_a((1024,))
                yt = _tile(self.y, 300)
                sa, sb = _scales(SCALE_PAIRS[0])
                with self.assertRaises(RuntimeError):
                    k512.k512_scaled_mm_tiled_a(yt, 300, self.w[1024], self.bias[1024], sa, sb)
                with self.assertRaises(RuntimeError):
                    k512.launch_tiled_a(k512.kernel(1024), yt, 300, self.w[1024], self.bias[1024], sa, sb)
            self.assertIn("self-check FAILED", k512.tiled_a_failure(1024) or "")
            self.assertEqual(sum("tiled-A OFF for N=1024" in s for s in lines), 1, lines)
            self.assertTrue(k512.tiled_a_ready(1536, build=True))  # other N unaffected
        finally:
            k512.reset_tiled_a_for_testing()
        self.assertTrue(k512.tiled_a_ready(1024, build=True))

    def test_rejects_bad_operands(self) -> None:
        m, n = 1000, 1024
        yt = _tile(self.y, m)
        w, bias = self.w[n], self.bias[n]
        sa, sb = _scales(SCALE_PAIRS[0])
        f = k512.k512_scaled_mm_tiled_a
        bad_cases = {
            "2-D activation": lambda: f(self.y[:m], m, w, bias, sa, sb),
            "[KT,128,512]": lambda: f(yt.view(-1, STEP, D), m, w, bias, sa, sb),
            "KT too small": lambda: f(yt[:-1], m, w, bias, sa, sb),
            "non-contiguous": lambda: f(yt.transpose(1, 2).contiguous().transpose(1, 2), m, w, bias, sa, sb),
            "M negative": lambda: f(yt, -1, w, bias, sa, sb),
            "M bool": lambda: f(yt, True, w, bias, sa, sb),
            "N=512": lambda: f(yt, m, w[:512], bias[:512], sa, sb),
            "no bias": lambda: f(yt, m, w, None, sa, sb),
            "bias fp32": lambda: f(yt, m, w, bias.float(), sa, sb),
            "out shape": lambda: f(yt, m, w, bias, sa, sb, out=torch.empty((m + 1, n), dtype=E4M3, device="cuda")),
        }
        raw = torch.empty(yt.numel() + 64, dtype=U8, device="cuda")
        bad_cases["misaligned"] = lambda: f(raw[1 : 1 + yt.numel()].view(yt.shape).view(E4M3), m, w, bias, sa, sb)
        for what, call in bad_cases.items():
            with self.assertRaises(ValueError, msg=what):
                call()
        # the C entry itself refuses a misaligned A (hipErrorInvalidValue)
        mis = raw[1 : 1 + yt.numel()].view(yt.shape).view(E4M3)
        with self.assertRaises(RuntimeError):
            k512.launch_tiled_a(k512.kernel(n), mis, m, w, bias, sa, sb)
        torch.cuda.synchronize()
        # M = 0: an empty output, no launch
        out = f(yt[:0], 0, w, bias, sa, sb)
        self.assertEqual(tuple(out.shape), (0, n))

    def test_resources_and_store_war_guard(self) -> None:
        # No spills in either kernel; every buffer_store_dwordx4 of the tiled kernel has
        # no write to its data VGPRs within WAR_WAIT_STATES wait states.
        for n in NS:
            res = k512.kernel(n).resources
            print(f"[k512-tiled-a-test] N={n} resources: {res}", flush=True)
            for name in ("k512_2wg_kernel:", "k512_2wg_kernel_tiled_a:"):
                self.assertIn(name, res)
                part = res.split(name, 1)[1].split(";")[0]
                self.assertIn("vspill=0", part, f"N={n} {name} {part}")
                self.assertIn("scratch=0", part, f"N={n} {name} {part}")
            with tempfile.TemporaryDirectory() as tmp:
                cmd = k512.hipcc_command(n, k512.source_path(), Path(tmp) / "lib.so", extra_flags=["--save-temps"])
                r = subprocess.run(cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
                self.assertEqual(r.returncode, 0, r.stdout[-2000:])
                asm = next(Path(tmp).glob("*gfx950*.s")).read_text()
            scan = _war_scan(asm, WAR_WAIT_STATES)
            tiled = [k for k in scan if k.startswith("_Z23k512_2wg_kernel_tiled_a")]
            row_major = [k for k in scan if k.startswith("_Z15k512_2wg_kernel")]
            self.assertEqual((len(tiled), len(row_major)), (1, 1), list(scan))
            stores, min_ws, tr_reads = scan[tiled[0]]
            print(
                f"[k512-tiled-a-test] N={n} WAR scan: tiled stores={stores} "
                f"min_ws_to_data_overwrite={min_ws} ds_read_b64_tr_b8={tr_reads}; row-major "
                f"stores={scan[row_major[0]][0]} min_ws={scan[row_major[0]][1]}",
                flush=True,
            )
            self.assertEqual(stores, 8)
            self.assertEqual(tr_reads, 128)
            self.assertIsNone(min_ws, f"N={n}: a store-data VGPR is rewritten within {WAR_WAIT_STATES} wait states")


def _war_scan(asm: str, window: int) -> Dict[str, Tuple[int, Optional[int], int]]:
    """Per kernel: (dwordx4 stores, min wait states from a store to a VALU / LDS /
    VMEM-load write of its data VGPRs within `window` (None: none), number of
    ds_read_b64_tr_b8). Same rule as probes/iter11/ytin/store_war_scan.py: an
    instruction counts one wait state, s_nop k counts k + 1."""

    def regs(tok: str) -> set:
        m = re.match(r"v\[(\d+):(\d+)\]", tok)
        if m:
            return set(range(int(m.group(1)), int(m.group(2)) + 1))
        m = re.match(r"v(\d+)$", tok)
        return {int(m.group(1))} if m else set()

    out: Dict[str, Tuple[int, Optional[int], int]] = {}
    kernel, insts = None, []

    def flush() -> None:
        if kernel is None:
            return
        stores, min_ws = 0, None
        for i, (op, args) in enumerate(insts):
            if not op.startswith("buffer_store_dwordx4") or not args:
                continue
            stores += 1
            data, ws = regs(args[0]), 0
            for op2, a2 in insts[i + 1 :]:
                ws += int(a2[0], 0) + 1 if op2 == "s_nop" and a2 else 1
                if ws > window:
                    break
                if op2.startswith(("v_", "ds_read", "buffer_load", "global_load")) and a2 and regs(a2[0]) & data:
                    min_ws = ws if min_ws is None else min(min_ws, ws)
                    break
        tr = sum(op == "ds_read_b64_tr_b8" for op, _ in insts)
        out[kernel] = (stores, min_ws, tr)

    for line in asm.splitlines():
        m = re.match(r"^(_Z\w+):", line)
        if m:
            flush()
            kernel, insts = m.group(1), []
            continue
        if re.match(r"^\.Lfunc_end\d+:", line):
            flush()
            kernel, insts = None, []
            continue
        s = line.strip()
        if kernel is None or not s or s.startswith((".", ";", "//")) or s.endswith(":"):
            continue
        parts = s.split(None, 1)
        args = [a.strip() for a in parts[1].split(";")[0].split("//")[0].split(",")] if len(parts) > 1 else []
        insts.append((parts[0], args))
    flush()
    return out


if __name__ == "__main__":
    unittest.main()

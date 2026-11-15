# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe
"""Tests of L11 (gin item (39), DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED): the out-proj FP8 dgrad fused with the
LN*U*dropout backward (ops/hip/l11_outproj_lnmd.{cu,py}).

CPU-level tests (no GPU kernel launches; some need a GPU device for tensor metadata): latch on a bad hipcc, toolchain
in the cache key, ABI mismatch, the resource check of both instantiations, every decline reason D0-D9 and D11 (not
prebuilt), B1a's env no longer declining, census tokens, the family label and "[rank N]" on every line, the e5m2-du
mode's ACTIVE line (once per rank, same census label as the bf16-du mode) and its E5 declines, the seam hook's gates
and argument order, prebuild build/load failures latching (never re-raised) and self-check device faults re-raising, a
stale HIP error is not a failure, knob off = nothing built and nothing printed.
GPU tests (gfx950): the fused wrapper vs the production chain (hipBLASLt dgrad + v2 + finalize) at small M with
NaN-prefilled outputs and guard bands, forced build/launch failures fall back to the chain bit for bit, the e5m2-du
mode vs B1a's producer on the same dy (r16 iter32 l11e5), the prebuild self-check. The large-M numerics matrix and
timing run from the lab drivers (probes/iter31/l11int/tasks, probes/iter32/l11e5/tasks) on top of the helpers below.
"""

import contextlib
import io
import os
import re
import unittest
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
from unittest import mock

import torch

from generative_recommenders.ops.hip import l11_outproj_lnmd as l11

E4, E5, BF, F32, F64, I16 = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int16,
)
D, NOUT, KD, DU_ROW = 512, 1536, 512, 2048
EPS, P_DROP = 1e-6, 0.1
SA, SB = 2.0**-19, 0.0613 / 448.0
GUARD_ROWS = 256
SENT16 = 0x5A5A


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return l11.device_arch(torch.cuda.current_device()) == "gfx950"


# ------------------------------------------------------------------------------------------------ data generators
# G8/W8: the L11-1a reviewer's generators (probes/iter30/l11res/review/rv_bitwise2.py gen_gauss / gen_heavy /
# gen_cancel, verbatim logic); x/u: the L11-1b builder's (probes/iter31/l11b/l11b_test.py rand_bf16).
def gen_gauss(M: int, seed: int = 4321) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    G = torch.empty((M, KD), dtype=E5, device="cuda")
    ch = max(1, (1 << 27) // KD)
    for i in range(0, M, ch):
        j = min(M, i + ch)
        G[i:j] = (torch.randn((j - i, KD), device="cuda", generator=g) * 1024.0).clamp_(-57344.0, 57344.0).to(E5)
    W = (torch.randn((NOUT, KD), device="cuda", generator=g) * 90.0).clamp_(-448.0, 448.0).to(E4)
    return G, W


def gen_heavy(M: int, seed: int = 778) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    G = torch.empty((M, KD), dtype=E5, device="cuda")
    ch = max(1, (1 << 26) // KD)
    for i in range(0, M, ch):
        j = min(M, i + ch)
        sc = torch.exp2(torch.rand((j - i, 1), device="cuda", generator=g) * 26.0 - 12.0)
        v = torch.randn((j - i, KD), device="cuda", generator=g) * sc
        u = torch.rand((j - i, KD), device="cuda", generator=g)
        v = torch.where(u < 0.15, torch.zeros_like(v), v)
        sgn = torch.where(torch.rand((j - i, KD), device="cuda", generator=g) < 0.5, -1.0, 1.0)
        v = torch.where(u > 0.995, sgn * 57344.0, v)
        G[i:j] = v.clamp_(-57344.0, 57344.0).to(E5)
    v = torch.randn((NOUT, KD), device="cuda", generator=g) * 90.0
    u = torch.rand((NOUT, KD), device="cuda", generator=g)
    sgn = torch.where(torch.rand((NOUT, KD), device="cuda", generator=g) < 0.5, -1.0, 1.0)
    v = torch.where(u > 0.99, sgn * 448.0, v)
    v[::8] *= 2.0**-8
    return G, v.clamp_(-448.0, 448.0).to(E4)


def gen_cancel(M: int, seed: int = 779) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    def codes(dt):
        c = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
        return c[torch.isfinite(c.view(dt).float())]

    c5, c4 = codes(E5), codes(E4)
    Gu = torch.empty((M, KD), dtype=torch.uint8, device="cuda")
    ch = max(1, (1 << 26) // KD)
    for i in range(0, M, ch):
        j = min(M, i + ch)
        Gu[i:j] = c5[torch.randint(0, c5.numel(), (j - i, KD), device="cuda", generator=g)]
    Wu = c4[torch.randint(0, c4.numel(), (NOUT, KD), device="cuda", generator=g)]
    Gu[:, 256:] = Gu[:, :256] ^ 0x80
    Wu[:, 256:] = Wu[:, :256]
    return Gu.view(E5), Wu.view(E4)


def rand_bf16(rows: int, cols: int, sigma: float, g: torch.Generator, heavy: bool = False) -> torch.Tensor:
    t = torch.empty((rows, cols), dtype=BF, device="cuda")
    ch = max(1, (1 << 27) // cols)
    for i in range(0, rows, ch):
        j = min(rows, i + ch)
        v = torch.randn((j - i, cols), device="cuda", generator=g) * sigma
        if heavy:  # per-row scale 2^U(-6,6), 1% x16 outliers (u spans sigmoid saturation incl. the IEEE fallback)
            v = v * torch.exp2(torch.empty((j - i, 1), device="cuda").uniform_(-6.0, 6.0, generator=g))
            v = torch.where(torch.rand((j - i, cols), device="cuda", generator=g) < 0.01, v * 16.0, v)
        t[i:j] = v.to(BF)
    return t


class Case:
    """One layer call's backward inputs in the production layout (x/u bf16 [M,512], int8 mask from the production
    mask generator, fp32 mean/rstd, bf16 norm weight/bias) + FP8 operands (G8 e5m2 [M,512], W8 e4m3 [1536,512])."""

    def __init__(self, M: int, mode: str = "gauss", seed: int = 4321) -> None:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        self.M, self.mode = M, mode
        np2 = mode.endswith("_np2")  # both decode scales non-power-of-two (the epilogue's fp32 sa*sb rounds)
        mode = mode[:-4] if np2 else mode
        gen = {"gauss": gen_gauss, "heavy": gen_heavy, "cancel": gen_cancel, "special": gen_gauss}[mode]
        self.G8, self.W8 = gen(M)
        g = torch.Generator(device="cuda")
        g.manual_seed(seed + 17)
        hv = mode not in ("gauss", "special")
        self.X = rand_bf16(M, D, 0.5, g, heavy=hv)
        self.U = rand_bf16(M, D, 1.0, g, heavy=hv)
        if mode == "special":
            # sigmoid IEEE-fallback rows (u <= -90), NaN/Inf u elements, NaN/Inf e5m2 dout codes
            self.U[0::97] = -100.0
            self.U[1::101, 3] = float("nan")
            self.U[2::103, 10] = float("inf")
            self.U[3::107, 20] = float("-inf")
            g8u = self.G8.view(torch.uint8)
            g8u[5::109, 7] = 0x7C  # +inf
            g8u[6::113, 9] = 0x7E  # NaN
            g8u[7::127, 11] = 0xFC  # -inf
        self.Wn = (1.0 + 0.05 * torch.randn(D, generator=g, device="cuda")).to(BF)
        self.Bn = (0.05 * torch.randn(D, generator=g, device="cuda")).to(BF)
        self.sa = torch.tensor([SA * (1.3712 if np2 else 1.0)], dtype=F32, device="cuda")
        self.sb = torch.tensor([SB * (1.1371 if np2 else 1.0)], dtype=F32, device="cuda")
        self.MEAN = torch.empty(M, dtype=F32, device="cuda")
        self.RSTD = torch.empty(M, dtype=F32, device="cuda")
        ch = 1 << 18
        for i in range(0, M, ch):
            j = min(M, i + ch)
            xf = self.X[i:j].float()
            self.MEAN[i:j] = xf.mean(dim=1)
            self.RSTD[i:j] = torch.rsqrt(xf.var(dim=1, unbiased=False) + EPS)
        self.MASK = thl._create_dropout_mask(
            N=M, D=D, BLOCK_D=D, concat_u=True, concat_x=True, dropout_ratio=P_DROP, seed=987654321,
            device=torch.device("cuda"),
        )
        torch.cuda.synchronize()

    def dgrad_hipblaslt(self) -> torch.Tensor:
        from generative_recommenders.ops.triton import bwd_fp8

        return bwd_fp8.output_projection_dgrad_gemm(self.G8, self.W8, self.sa, self.sb, BF)

    def chain(self, dy: torch.Tensor, alloc_du=None):
        """HSTUComputeOutputFunction.backward's consumer call (production wrapper: v2 + finalize + allocations)."""
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        ctx = mock.patch.object(thl, "_alloc_du", alloc_du) if alloc_du is not None else contextlib.nullcontext()
        with ctx:
            dx, du, dw, db, _ = thl.triton_layer_norm_mul_dropout_bwd(
                dy=dy, x=self.X, u=self.U, weight=self.Wn, bias=self.Bn, mean=self.MEAN, rstd=self.RSTD,
                BLOCK_D=D, num_warps=2, eps=EPS, training=True, dropout_ratio=P_DROP, seed=0, silu_u=True,
                concat_u=True, concat_x=True, mul_u_activation_type="silu", compute_y=False, random_mask=self.MASK,
                du_row_width=DU_ROW,
            )
        return dx, du, dw, db

    def fused(self, alloc_du=None, log_active: bool = False):
        """The wrapper the backward calls + the production finalize. Returns (dx, du, dw, db, DWP, DBP, P) or None."""
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        out = l11.run(self.G8, self.W8, self.sa, self.sb, self.X, self.U, self.MASK, self.MEAN, self.RSTD, self.Wn,
                      self.Bn, P_DROP, alloc_du or thl._alloc_du, log_active=log_active)
        if out is None:
            return None
        dx, du, dwp, dbp, P = out
        dw = torch.empty((D,), dtype=BF, device="cuda")
        db = torch.empty((D,), dtype=BF, device="cuda")
        thl._lnmd_bwd_finalize(dwp, dbp, dw, db, P, D, torch.cuda.get_device_properties(0).multi_processor_count)
        return dx, du, dw, db, dwp, dbp, P


def alloc_du_guarded(store: Dict[str, torch.Tensor]):
    """_alloc_du replacement: an [M + GUARD_ROWS, 2048] buffer, cols 0..511 NaN-prefilled (0xFFFF), cols 512..2047 and
    the guard rows prefilled with the sentinel 0x5A5A; the full buffer is kept in `store`."""

    def alloc(u: torch.Tensor, width: int) -> torch.Tensor:
        m = u.shape[0]
        buf = torch.full((m + GUARD_ROWS, width), SENT16, dtype=I16, device=u.device)
        buf[:m, : u.shape[1]] = -1
        store["buf"] = buf
        return buf.view(BF)[:m, : u.shape[1]]

    return alloc


def bits_ne(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.contiguous().view(I16) != b.contiguous().view(I16)


def bits_ne_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bit mismatches that are not NaN-vs-NaN (a NaN's sign/payload is not compared)."""
    both_nan = torch.isnan(a.float()) & torch.isnan(b.float())
    return bits_ne(a, b) & ~both_nan


def ulp_max(a: torch.Tensor, b: torch.Tensor) -> int:
    ai = a.contiguous().view(I16).to(torch.int32)
    bi = b.contiguous().view(I16).to(torch.int32)
    ao = torch.where(ai < 0, -(ai & 0x7FFF), ai)
    bo = torch.where(bi < 0, -(bi & 0x7FFF), bi)
    return int((ao - bo).abs().max()) if ai.numel() else 0


def fp64_dwdb(case: Case, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp64 oracle of dweight/dbias from a bf16 dy: sum_rows(dy_m * silu(u) * xhat), sum_rows(dy_m * silu(u))."""
    s = l11.dropout_scale(P_DROP)
    dw = torch.zeros(D, dtype=F64, device="cuda")
    db = torch.zeros(D, dtype=F64, device="cuda")
    ch = 1 << 17
    for i in range(0, case.M, ch):
        j = min(case.M, i + ch)
        keep = (case.MASK[i:j].to(torch.int32) & 1) != 0
        dym = torch.where(keep, dy[i:j, 2 * D:].double() * s, torch.zeros((), dtype=F64, device="cuda"))
        u = case.U[i:j].double()
        silu = u * torch.sigmoid(u)
        xh = (case.X[i:j].double() - case.MEAN[i:j, None].double()) * case.RSTD[i:j, None].double()
        gg = dym * silu
        dw += (gg * xh).sum(0)
        db += gg.sum(0)
    return dw, db


def rel_l2(a: torch.Tensor, ref: torch.Tensor) -> float:
    a64, r64 = a.double(), ref.double()
    return float((a64 - r64).norm() / r64.norm().clamp_min(1e-300))


def compare_case(case: Case, with_dy_debug: bool = True) -> Dict[str, object]:
    """Fused vs the production chain (hipBLASLt dy + v2) and vs v2 on L11's own dy; DP rows vs SK-tail rows; NaN
    prefill / guard coverage; dW/dB vs the chain and vs fp64. Returns a flat dict of counts."""
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    M = case.M
    tail = thl.l11_sk_tail_start(M)
    res: Dict[str, object] = {"M": M, "data": case.mode, "sk_tail_row": tail, "tail_rows": M - tail}
    store: Dict[str, torch.Tensor] = {}
    out = case.fused(alloc_du=alloc_du_guarded(store))
    assert out is not None, f"fused run declined/failed: {l11.broken()}"
    dx_l, du_l, dw_l, db_l, dwp, dbp, P = out
    buf = store["buf"]
    torch.cuda.synchronize()
    res["P"] = P
    res["du_guard_changed"] = int((buf[:M, D:] != SENT16).sum()) + int((buf[M:] != SENT16).sum())
    res["du_unwritten"] = int((buf[:M, :D] == -1).sum())
    res["nonfinite_out"] = int((~torch.isfinite(dx_l.float())).sum()) + int((~torch.isfinite(du_l.float())).sum())
    res["nonfinite_partials"] = int((~torch.isfinite(dwp)).sum()) + int((~torch.isfinite(dbp)).sum())
    # chain on hipBLASLt's dy
    dy_h = case.dgrad_hipblaslt()
    dx_c, du_c, dw_c, db_c = case.chain(dy_h)
    m_du, m_dx = bits_ne(du_l, du_c), bits_ne(dx_l, dx_c)
    res.update(
        du_dp=int(m_du[:tail].sum()), du_tail=int(m_du[tail:].sum()),
        dx_dp=int(m_dx[:tail].sum()), dx_tail=int(m_dx[tail:].sum()),
        du_tail_maxulp=ulp_max(du_l[tail:], du_c[tail:]) if M > tail else 0,
        dx_tail_maxulp=ulp_max(dx_l[tail:], dx_c[tail:]) if M > tail else 0,
    )
    res["nan_pos_diff"] = int((torch.isnan(dx_l.float()) != torch.isnan(dx_c.float())).sum()) + int(
        (torch.isnan(du_l.float()) != torch.isnan(du_c.float())).sum())
    r_du, r_dx = bits_ne_real(du_l, du_c), bits_ne_real(dx_l, dx_c)
    res.update(du_dp_real=int(r_du[:tail].sum()), dx_dp_real=int(r_dx[:tail].sum()),
               du_tail_real=int(r_du[tail:].sum()), dx_tail_real=int(r_dx[tail:].sum()))
    del m_du, m_dx, r_du, r_dx
    # dW/dB vs chain and vs fp64 (own dy)
    res.update(
        dw_rel_vs_chain=rel_l2(dw_l, dw_c), db_rel_vs_chain=rel_l2(db_l, db_c),
        dw_bf16_ne_chain=int(bits_ne(dw_l, dw_c).sum()), db_bf16_ne_chain=int(bits_ne(db_l, db_c).sum()),
        dw_maxulp_chain=ulp_max(dw_l, dw_c), db_maxulp_chain=ulp_max(db_l, db_c),
    )
    dwp_sum, dbp_sum = dwp.double().sum(0), dbp.double().sum(0)
    if with_dy_debug:
        dy_l = l11.dy_debug(case.G8, case.W8, case.sa, case.sb)
        m_dy = bits_ne(dy_l, dy_h)
        res.update(dy_dp=int(m_dy[:tail].sum()), dy_tail=int(m_dy[tail:].sum()))
        rows_bad = torch.nonzero(m_dy.any(dim=1)).flatten()
        res["dy_first_mismatch_row"] = int(rows_bad[0]) if rows_bad.numel() else None
        del m_dy, rows_bad
        dx_s, du_s, dw_s, db_s = case.chain(dy_l)
        res.update(du_same_dy=int(bits_ne(du_l, du_s).sum()), dx_same_dy=int(bits_ne(dx_l, dx_s).sum()),
                   du_same_dy_real=int(bits_ne_real(du_l, du_s).sum()), dx_same_dy_real=int(bits_ne_real(dx_l, dx_s).sum()))
        o_w, o_b = fp64_dwdb(case, dy_l)
        res.update(
            dw_rel_fp64_l11=rel_l2(dwp_sum, o_w), db_rel_fp64_l11=rel_l2(dbp_sum, o_b),
            dw_bf16_ne_rn64_l11=int(bits_ne(dw_l, o_w.to(BF)).sum()), db_bf16_ne_rn64_l11=int(bits_ne(db_l, o_b.to(BF)).sum()),
        )
        del dx_s, du_s, dy_l
    o_wc, o_bc = fp64_dwdb(case, dy_h)
    # chain's fp32 partial sums are not exposed by the wrapper: compare its bf16 outputs to RN(fp64)
    res.update(
        dw_bf16_ne_rn64_chain=int(bits_ne(dw_c, o_wc.to(BF)).sum()), db_bf16_ne_rn64_chain=int(bits_ne(db_c, o_bc.to(BF)).sum()),
        dw_rel_fp64_l11_vs_chaindy=rel_l2(dwp_sum, o_wc), db_rel_fp64_l11_vs_chaindy=rel_l2(dbp_sum, o_bc),
    )
    return res


def chain_partials_rel_fp64(case: Case, dy: torch.Tensor) -> Tuple[float, float]:
    """rel-L2 of the chain's own fp32 partial sums (v2 launched exactly as the wrapper does) vs fp64."""
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    tile = max(1, min(sms * 64, case.M // 4))
    dx = torch.empty_like(case.X)
    du = thl._alloc_du(case.U, DU_ROW)
    dW = torch.empty((tile, D), dtype=F32, device="cuda")
    dB = torch.empty((tile, D), dtype=F32, device="cuda")
    thl._ln_mul_dropout_bwd_dx_du_rng_v2[(tile,)](
        dx, du, dy, dW, dB, case.X, case.U, None, case.Wn, case.Bn, case.MEAN, case.RSTD, case.MASK,
        dx.stride(0), du.stride(0), dy.stride(0), case.X.stride(0), case.U.stride(0), 0, case.MASK.stride(0),
        D, EPS, P_DROP, case.M, SILU_U=True, BLOCK_D=D, TRAINING=True, CONCAT_U=True, CONCAT_X=True,
        MUL_U_ACTIVATION_TYPE="silu", COMPUTE_Y=False, NUM_STAGES=1, CONTIG=False, EVEN_D=True, SCM=".cs", LEV="",
        num_warps=1,
    )
    o_w, o_b = fp64_dwdb(case, dy)
    return rel_l2(dW.double().sum(0), o_w), rel_l2(dB.double().sum(0), o_b)


@contextlib.contextmanager
def captured():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def fake_ctx(**over) -> SimpleNamespace:
    base = dict(
        fp8_y_saved=True, group_norm=False, training=True, dropout_ratio=0.1, has_random_mask=True, silu_u=True,
        concat_u=True, concat_x=True, mul_u_activation_type="silu", num_warps=2, BLOCK_D=512, du_row_width=2048,
    )
    base.update(over)
    return SimpleNamespace(**base)


def fake_tensors(M: int = 1000, device: str = "cuda") -> Dict[str, torch.Tensor]:
    return dict(
        dout=torch.zeros((M, D), dtype=BF, device=device),
        output_weight=torch.zeros((NOUT, D), dtype=F32, device=device),
        attn=torch.zeros((M, D), dtype=BF, device=device),
        u=torch.zeros((M, D), dtype=BF, device=device),
        mean=torch.zeros((M,), dtype=F32, device=device),
        rstd=torch.zeros((M,), dtype=F32, device=device),
        norm_weight=torch.zeros((D,), dtype=BF, device=device),
        norm_bias=torch.zeros((D,), dtype=BF, device=device),
        random_mask=torch.zeros((M, D), dtype=torch.int8, device=device),
    )


def call_decline(ctx, t, snr=False, v2=True) -> Optional[str]:
    return l11.decline_reason(ctx, t["dout"], t["output_weight"], t["attn"], t["u"], t["mean"], t["rstd"],
                              t["norm_weight"], t["norm_bias"], t["random_mask"], snr_enabled=snr, lnmd_bwd_v2=v2)


# ------------------------------------------------------------------------------------------------ CPU-level tests
class L11CpuTest(unittest.TestCase):
    def setUp(self) -> None:
        l11.reset_for_testing(unload=False)
        l11.set_requested_for_testing(None)

    def tearDown(self) -> None:
        l11.reset_for_testing(unload=False)
        l11.set_requested_for_testing(None)

    def test_knob_values(self) -> None:
        for val, want in (("1", True), ("0", False), ("true", False), ("", False), (None, False)):
            env = dict(os.environ)
            env.pop(l11.KNOB, None)
            if val is not None:
                env[l11.KNOB] = val
            with mock.patch.dict(os.environ, env, clear=True):
                l11.set_requested_for_testing(None)
                self.assertEqual(l11.requested(), want, val)

    def test_vd_knob_selects_source(self) -> None:
        # L11-VD (r16 iter37; default off): l11_outproj_lnmd_vd.cu only for DLRM_HSTU_FP8_L11_VD == "1"; absent / "0" /
        # anything else keeps l11_outproj_lnmd.cu (the shipped source); the _SRC override wins over both.
        self.assertTrue(l11._SOURCE.is_file())
        self.assertTrue(l11._SOURCE_VD.is_file())
        self.assertEqual(l11._SOURCE.name, "l11_outproj_lnmd.cu")
        self.assertNotEqual(l11._SOURCE.read_bytes(), l11._SOURCE_VD.read_bytes())
        self.assertIn(b"+vd(", l11._SOURCE_VD.read_bytes())
        self.assertNotIn(b"+vd(", l11._SOURCE.read_bytes())
        for val, want in (("1", l11._SOURCE_VD), (" 1 ", l11._SOURCE_VD), ("0", l11._SOURCE), ("", l11._SOURCE),
                          ("true", l11._SOURCE), ("2", l11._SOURCE), (None, l11._SOURCE)):
            env = dict(os.environ)
            env.pop(l11.VD_KNOB, None)
            env.pop(l11.KNOB + "_SRC", None)
            if val is not None:
                env[l11.VD_KNOB] = val
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(l11._source(), want, val)
                self.assertEqual(l11.vd_requested(), want == l11._SOURCE_VD, val)
        with mock.patch.dict(os.environ, {l11.VD_KNOB: "1", l11.KNOB + "_SRC": "/x/y.cu"}):
            self.assertEqual(str(l11._source()), "/x/y.cu")
        # distinct sources -> distinct cache keys (each builds its own .so); needs hipcc on PATH, else skip that half
        try:
            k_vd, k_base = l11.cache_key(l11._SOURCE_VD), l11.cache_key(l11._SOURCE)
        except (RuntimeError, OSError):
            return
        self.assertNotEqual(k_vd, k_base)

    def test_knob_off_builds_nothing_prints_nothing(self) -> None:
        l11.set_requested_for_testing(False)
        with mock.patch.object(l11, "build", side_effect=AssertionError("built")), captured() as out:
            self.assertFalse(l11.prebuild(lambda: (True, ""), static_ok=True))
            self.assertIsNone(
                l11.run(*([torch.zeros(1)] * 11), 0.1, lambda u, w: u)  # pyre-ignore
            )
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(call_decline(fake_ctx(), fake_tensors(8, "cpu")), "D0 knob off")

    def test_bad_hipcc_latches(self) -> None:
        l11.set_requested_for_testing(True)
        with mock.patch.dict(os.environ, {l11.KNOB + "_HIPCC": "/nonexistent/hipcc"}):
            l11._toolchains.pop("/nonexistent/hipcc", None)
            with self.assertRaises(Exception):
                l11.build()
            l11.reset_for_testing(unload=True)
            with captured() as out:
                ok = l11.prebuild(lambda: (True, ""), static_ok=True)
        self.assertFalse(ok)
        if torch.cuda.is_available():
            self.assertIsNotNone(l11.broken())
            self.assertIn(f"{l11.FAMILY} DISABLED: prebuild build/load: ", out.getvalue())
            self.assertIn("[rank ", out.getvalue())
            self.assertFalse(l11.enabled())

    def test_cache_key_includes_toolchain(self) -> None:
        with mock.patch.object(l11, "_toolchain", return_value="hipcc A"):
            k1 = l11.cache_key()
        with mock.patch.object(l11, "_toolchain", return_value="hipcc B"):
            k2 = l11.cache_key()
        self.assertNotEqual(k1, k2)

    def test_abi_mismatch_raises(self) -> None:
        fake = SimpleNamespace(l11_abi_version=lambda: l11._ABI_VERSION - 1)
        with self.assertRaisesRegex(RuntimeError, "l11_abi_version"):
            l11._bind(fake)  # pyre-ignore

    def test_resource_check(self) -> None:
        ok = [{"Function Name": "_Z20l11_outproj_lnmd_bwdILi1ELi2ELi0ELi0ELi1ELi2ELi1ELi0E", "VGPRs": "251",
               "VGPRs Spill": "0", "ScratchSize [bytes/lane]": "0", "TotalSGPRs": "106", "SGPRs Spill": "20",
               "LDS Size [bytes/block]": "98304"}]
        self.assertEqual(l11.check_resources(ok)["vgpr"], 251)
        bad = [dict(ok[0], **{"VGPRs Spill": "1", "ScratchSize [bytes/lane]": "8"})]
        with self.assertRaisesRegex(RuntimeError, "spill"):
            l11.check_resources(bad)
        with self.assertRaises(RuntimeError):
            l11.check_resources([])
        # l11e5: one .so carries both production instantiations <1,2,0,0,1,2,1,0,E5>; both must be 0 spill / 0 scratch
        name = "_Z20l11_outproj_lnmd_bwdILi1ELi2ELi0ELi0ELi1ELi2ELi1ELi0ELi{}EEv8L11CArgs"
        both = [dict(ok[0], **{"Function Name": name.format(1), "VGPRs": "255"}),
                dict(ok[0], **{"Function Name": name.format(0)})]
        got = l11.check_resources(both)
        self.assertEqual((got["vgpr"], got["e5_vgpr"], got["e5_vgpr_spill"], got["e5_scratch"]), (251, 255, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "spill"):
            l11.check_resources([dict(both[0], **{"VGPRs Spill": "1"}), both[1]])

    def test_lines_carry_rank_and_census_tokens(self) -> None:
        l11.set_requested_for_testing(True)
        with captured() as out:
            l11.note_decline("D8 M=5000000 outside [1, 4194304]")
            l11.note_decline("D8 M=6000000 outside [1, 4194304]")  # same reason key: printed once
            l11.prebuild(lambda: (True, ""), static_ok=False)
            l11._disable("test reason")
            l11._disable("second reason")  # latched once
        lines = [ln for ln in out.getvalue().splitlines() if ln]
        self.assertEqual(len(lines), 3, lines)
        self.assertTrue(all(ln.endswith("]") and "[rank " in ln for ln in lines), lines)
        self.assertTrue(lines[0].startswith(f"{l11.FAMILY} NOT armed: D8"), lines[0])
        self.assertTrue(lines[1].startswith(f"{l11.FAMILY} INERT"), lines[1])
        self.assertTrue(lines[2].startswith(f"{l11.FAMILY} DISABLED: test reason"), lines[2])

    def test_e5_mode_lines_and_record(self) -> None:
        # run_e5 with the launch mocked (no GPU): the e5m2-du mode's ACTIVE line once per process on every rank, the
        # same census family label as the bf16-du ACTIVE line ("armed OK: <FAMILY> ACTIVE ..."), "mode=e5", no
        # decline token; the record has B1a's shape/fields; an E5 contract miss prints one family-prefixed NOT armed
        # line per reason and returns None (the seam then runs the GEMM + B1a's path on the same operands).
        from generative_recommenders.ops.triton import bwd_fp8

        l11.set_requested_for_testing(True)
        M, P = 64, 16
        t = fake_tensors(M, "cpu")
        g8 = torch.zeros((M, D), dtype=E5)
        w8 = torch.zeros((NOUT, D), dtype=E4)
        one = torch.ones((1,), dtype=F32)

        def fake_launch(*a, **k):
            return (torch.zeros((M, D), dtype=BF), torch.zeros((P, D)), torch.zeros((P, D)), torch.ones((P, D)),
                    torch.full((8,), 2.0), P)

        args = [g8, w8, one, one, t["attn"], t["u"], t["norm_weight"], t["norm_bias"], t["mean"], t["rstd"], D, 2,
                P_DROP, True, t["random_mask"], DU_ROW, 7, one, one, None]
        bad = list(args)
        bad[10] = 1024  # BLOCK_D: an E5 contract miss
        with mock.patch.object(l11, "_lib", object()), \
                mock.patch.object(l11, "launch_e5", side_effect=fake_launch), \
                mock.patch.object(l11, "dwdb_finalize", return_value=None), \
                mock.patch.object(bwd_fp8, "grad_dtype_and_limit", return_value=(E5, 57344.0)), \
                mock.patch.object(bwd_fp8, "uvqk_tm_pad", return_value=128), captured() as out:
            r1 = l11.run_e5(*args)
            r2 = l11.run_e5(*args)
            r3 = l11.run_e5(*bad)
            r4 = l11.run_e5(*bad)
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        self.assertIsNone(r3)
        self.assertIsNone(r4)
        lines = [ln for ln in out.getvalue().splitlines() if ln]
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[0].startswith(f"armed OK: {l11.FAMILY} ACTIVE mode=e5 (B1a contract)"), lines[0])
        self.assertTrue(lines[0].endswith("]") and "[rank " in lines[0], lines[0])
        for token in ("NOT armed", "NOT used", "DISABLED", "disabled", "INERT", "declin", "fall back", "fallback"):
            self.assertNotIn(token, lines[0])
        self.assertTrue(lines[1].startswith(f"{l11.FAMILY} NOT armed: E5 v3 launch shape BLOCK_D=1024"), lines[1])
        dx, du, dw, db, rec = r1
        self.assertEqual(tuple(du.shape), (M, D))
        self.assertEqual(du.stride(), (0, 0))
        self.assertEqual((rec.rows, rec.width, rec.rows_out, rec.u_cols, rec.site_key, rec.buffer_ptr),
                         (M, DU_ROW, 128, (0, D), 7, 0))
        self.assertEqual((rec.dz8.dtype, tuple(rec.dz8.shape)), (E5, (128, DU_ROW)))
        self.assertEqual(float(rec.u_amax), 2.0)
        self.assertTrue(torch.equal(rec.u_colsum, torch.full((D,), float(P))))
        self.assertEqual((dw.dtype, tuple(dw.shape), db.dtype, tuple(db.shape)), (BF, (D,), BF, (D,)))
        self.assertIsNone(l11.broken())

    def test_e5_hook_gates_and_arguments(self) -> None:
        # The seam's hook (triton_hstu_linear._l11_du_e5m2_or_none): None unless L11 is requested, prebuilt and not
        # latched; otherwise run_e5 with B1a's keywords in run_e5's positional order, its result returned as is.
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        kw = dict(x="X", u="U", weight="W", bias="B", mean="MEAN", rstd="RSTD", BLOCK_D=512, num_warps=2,
                  dropout_ratio=0.1, silu_u=True, random_mask="MASK", du_row_width=2048, site_key=7,
                  inv_scale="INV", cast_scale="CS", generation=("st", 3))
        result = ("dx", "du", "dw", "db", "emit")
        with mock.patch.object(l11, "run_e5", return_value=result) as run_e5, captured() as out:
            l11.set_requested_for_testing(False)
            self.assertIsNone(thl._l11_du_e5m2_or_none("G8", "W8", "SA", "SB", **kw))
            l11.set_requested_for_testing(True)
            l11.set_prebuilt_for_testing(False)
            self.assertIsNone(thl._l11_du_e5m2_or_none("G8", "W8", "SA", "SB", **kw))
            l11.set_prebuilt_for_testing(True)
            l11._broken = "x"
            self.assertIsNone(thl._l11_du_e5m2_or_none("G8", "W8", "SA", "SB", **kw))
            l11._broken = None
            self.assertIs(thl._l11_du_e5m2_or_none("G8", "W8", "SA", "SB", **kw), result)
        run_e5.assert_called_once_with("G8", "W8", "SA", "SB", "X", "U", "W", "B", "MEAN", "RSTD", 512, 2, 0.1, True,
                                       "MASK", 2048, 7, "INV", "CS", ("st", 3))
        self.assertEqual(out.getvalue(), "")

    def test_prebuild_build_failure_latches_never_raises(self) -> None:
        # S1: a hipcc / clang crash text that looks like a device fault ('Illegal instruction (core dumped)') must
        # latch L11 off with a DISABLED line, never abort the run.
        l11.set_requested_for_testing(True)
        crash = RuntimeError("hipcc rc=254: clang: error: unable to execute command: Illegal instruction (core dumped)")
        self.assertIsNotNone(l11._DEVICE_FAULT.search(str(crash)))
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                mock.patch.object(l11, "_load", side_effect=crash), captured() as out:
            ok = l11.prebuild(lambda: (True, ""), static_ok=True)
        self.assertFalse(ok)
        self.assertFalse(l11.prebuilt())
        self.assertIn("Illegal instruction", l11.broken() or "")
        self.assertIn(f"{l11.FAMILY} DISABLED: prebuild build/load: RuntimeError: hipcc rc=254", out.getvalue())
        self.assertNotIn("DEVICE FAULT", out.getvalue())

    def test_prebuild_self_check_device_fault_reraises(self) -> None:
        # S1: only the self-check's GPU work is matched against _DEVICE_FAULT; a fault there aborts before RUN_START.
        l11.set_requested_for_testing(True)

        def faulty_check() -> Tuple[bool, str]:
            raise RuntimeError("HIP error: hipErrorIllegalAddress")

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                mock.patch.object(l11, "_load", return_value=None), captured() as out:
            with self.assertRaisesRegex(RuntimeError, "hipErrorIllegalAddress"):
                l11.prebuild(faulty_check, static_ok=True)
        self.assertTrue(out.getvalue().startswith(f"{l11.FAMILY} prebuild DEVICE FAULT: "), out.getvalue())
        self.assertFalse(l11.prebuilt())

        def mismatch_check() -> Tuple[bool, str]:
            raise ValueError("not a device fault")

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                mock.patch.object(l11, "_load", return_value=None), captured() as out:
            self.assertFalse(l11.prebuild(mismatch_check, static_ok=True))
        self.assertIn(f"{l11.FAMILY} DISABLED: prebuild self-check: ValueError", out.getvalue())

    def test_device_fault_names_c3_2(self) -> None:
        # r16 C3-2 (ship17 MX6; r14 v1.2 open item 2, option (b); rc 709 added at r14's request): the five device-fatal
        # hipError NAMES that r14's outproj_dual_mx6._check_rc prints for rc 53 / 214 / 702 / 709 / 710 now match
        # (abort before RUN_START), in any case and inside the loader's own text. The pre-C3-2 pattern is kept verbatim
        # as the prefix, so every text it matched still matches and nothing else moves; a bare rc number (r16's
        # launchers) never matches.
        pre_c3_2 = (r"memory access fault|illegal (memory|address|instruction)|unspecified launch failure|"
                    r"device-side assert|hipErrorLaunchFailure|hipErrorIllegalAddress|\bECC\b|GPU core dump|"
                    r"HSA_STATUS_ERROR")
        codes = ((53, "hipErrorPriorLaunchFailure"), (214, "hipErrorECCNotCorrectable"),
                 (702, "hipErrorLaunchTimeOut"), (709, "hipErrorContextIsDestroyed"), (710, "hipErrorAssert"))
        self.assertEqual(l11._DEVICE_FAULT.pattern, pre_c3_2 + "|" + "|".join(name for _rc, name in codes))
        self.assertEqual(l11._DEVICE_FAULT.flags & re.I, re.I)
        old = re.compile(pre_c3_2, re.I)
        for rc, name in codes:
            loader = (f"OutprojMx6Error: [outproj-mx6] outproj_dual_mx6_launch rc={rc} ({name}) M=769: not launched, "
                      "outputs not written")
            for text in (name, name.upper(), name.lower(), loader):
                self.assertIsNotNone(l11._DEVICE_FAULT.search(text), text)
            self.assertIsNone(old.search(loader), loader)  # the change: these latched before C3-2
            for bare in (f"rc={rc}", f"l11_dy_debug rc={rc}", f"l11do8_e5_only_launch rc={rc} (hipError) M=64",
                         f"hipError {rc}"):
                self.assertIsNone(l11._DEVICE_FAULT.search(bare), bare)
        # unchanged: what the pre-C3-2 pattern matched still matches, what it did not match still does not (incl. the
        # neighbouring hipError names and the rc 702 / 709 descriptions)
        for text in ("Memory access fault by GPU node-6", "an illegal memory access was encountered",
                     "illegal instruction", "unspecified launch failure", "device-side assert triggered",
                     "hipErrorLaunchFailure", "hipErrorIllegalAddress", "uncorrectable ECC error encountered",
                     "GPU core dump failed", "HSA_STATUS_ERROR_EXCEPTION"):
            self.assertIsNotNone(old.search(text), text)
            self.assertIsNotNone(l11._DEVICE_FAULT.search(text), text)
        for text in ("hipErrorInvalidValue", "hipErrorOutOfMemory", "hipErrorInvalidContext", "hipErrorNotReady",
                     "hipErrorLaunchOutOfResources", "hipErrorSetOnActiveProcess", "hipErrorUnknown",
                     "context is destroyed", "the launch timed out and was terminated", "HIP out of memory",
                     "ValueError: not a device fault"):
            self.assertIsNone(old.search(text), text)
            self.assertIsNone(l11._DEVICE_FAULT.search(text), text)
        # description-only texts (Triton's form, a bare hipGetErrorString) and the other known texts: as before
        corpus = ("HIP error: an illegal memory access was encountered", "HIP error: unspecified launch failure",
                  "HIP error: unspecified launch failure in prior launch",
                  "HIP error: uncorrectable ECC error encountered",
                  "HIP error: the launch timed out and was terminated", "HIP error: device-side assert triggered",
                  "HIP error: invalid argument", "Triton Error [HIP]:  Code: 702, Messsage: the launch timed out and "
                  "was terminated", "Memory access fault by GPU node-6", "GPU core dump failed",
                  "HSA_STATUS_ERROR_EXCEPTION", "HIP error: hipErrorIllegalAddress",
                  "hipcc rc=254: clang: error: unable to execute command: Illegal instruction (core dumped)",
                  "RuntimeError: l11_dy_debug rc=700", "HIP out of memory", "ValueError: not a device fault",
                  "HIP error: context is destroyed", "Triton Error [HIP]:  Code: 709, Messsage: context is destroyed")
        for text in corpus:
            self.assertEqual(l11._DEVICE_FAULT.search(text) is None, old.search(text) is None, text)
        # torch 2.12.0+rocm7.2's REAL C10_HIP_CHECK text (kit U open item U-O9; the HIP 7.2 texts measured on the
        # training overlay, re-measured on the node by kit U's S3 premise row): "CUDA error: " + hipGetErrorString +
        # "\nSearch for `<hipGetErrorName>' in <url> for more information." + the check suffix + "\n" + a device-side
        # assertion sentence. It NAMES the hipError, and every such text already matched the pre-C3-2 pattern through
        # its "Device-side assert..." sentence: C3-2 changes none of them. Triton's form and the bare description
        # carry no name: both patterns classify them alike.
        url = "https://rocm.docs.amd.com/projects/HIP/en/latest/index.html"
        suffix = ("\nCUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace "
                  "below might be incorrect.\nFor debugging consider passing AMD_SERIALIZE_KERNEL=3")
        dsa = ("Device-side assertions were explicitly omitted for this error check; the error probably arose while "
               "initializing the DSA handlers.", "Device-side assertion tracking was not enabled by user.")
        hip72 = ((1, "hipErrorInvalidValue", "invalid argument"), (2, "hipErrorOutOfMemory", "out of memory"),
                 (53, "hipErrorPriorLaunchFailure", "unspecified launch failure in prior launch"),
                 (214, "hipErrorECCNotCorrectable", "uncorrectable ECC error encountered"),
                 (700, "hipErrorIllegalAddress", "an illegal memory access was encountered"),
                 (701, "hipErrorLaunchOutOfResources", "too many resources requested for launch"),
                 (702, "hipErrorLaunchTimeOut", "the launch timed out and was terminated"),
                 (709, "hipErrorContextIsDestroyed", "context is destroyed"),
                 (710, "hipErrorAssert", "device-side assert triggered"),
                 (719, "hipErrorLaunchFailure", "unspecified launch failure"),
                 (999, "hipErrorUnknown", "unknown error"))
        for rc, name, desc in hip72:
            core = f"CUDA error: {desc}\nSearch for `{name}' in {url} for more information.{suffix}\n"
            for text in [core + tail for tail in dsa] + [f"RuntimeError: {core}{tail}" for tail in dsa]:
                self.assertIsNotNone(old.search(text), (rc, text[-60:]))
                self.assertIsNotNone(l11._DEVICE_FAULT.search(text), (rc, text[-60:]))
            for text in (f"Triton Error [HIP]:  Code: {rc}, Messsage: {desc}", desc, f"HIP error: {desc}"):
                self.assertEqual(l11._DEVICE_FAULT.search(text) is None, old.search(text) is None, text)

    def test_fail_closed_raises(self) -> None:
        l11.set_requested_for_testing(True)
        with mock.patch.dict(os.environ, {l11.KNOB + "_FAIL_CLOSED": "1"}):
            with self.assertRaises(RuntimeError):
                l11._disable("boom")

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU device for tensor metadata")
    def test_decline_reasons(self) -> None:
        l11.set_requested_for_testing(True)
        t = fake_tensors(1000)
        arch_ok = mock.patch.object(l11, "device_arch", return_value="gfx950")
        l11.set_prebuilt_for_testing(True)
        with arch_ok:
            self.assertIsNone(call_decline(fake_ctx(), t))
            self.assertTrue(call_decline(fake_ctx(fp8_y_saved=False), t).startswith("D4"))
            self.assertTrue(call_decline(fake_ctx(), t, snr=True).startswith("D5"))
            for k, v in (("group_norm", True), ("training", False), ("dropout_ratio", 0.0), ("has_random_mask", False),
                         ("silu_u", False), ("concat_u", False), ("concat_x", False), ("mul_u_activation_type", "none")):
                self.assertTrue(call_decline(fake_ctx(**{k: v}), t).startswith("D6"), k)
            self.assertTrue(call_decline(fake_ctx(), t, v2=False).startswith("D7"))
            self.assertTrue(call_decline(fake_ctx(num_warps=4), t).startswith("D7"))
            self.assertTrue(call_decline(fake_ctx(BLOCK_D=1024), t).startswith("D7"))
            t2 = dict(t, dout=t["dout"].float())
            self.assertTrue(call_decline(fake_ctx(), t2).startswith("D8"))
            t2 = dict(t, attn=torch.zeros((1000, 1024), dtype=BF, device="cuda")[:, :512])
            self.assertTrue(call_decline(fake_ctx(), t2).startswith("D8"))
            t2 = dict(t, u=torch.zeros((1000 * 512 + 8,), dtype=BF, device="cuda")[4:4 + 1000 * 512].view(1000, 512))
            self.assertTrue(call_decline(fake_ctx(), t2).startswith("D8"), "misaligned u")
            t2 = dict(t, random_mask=None)
            self.assertTrue(call_decline(fake_ctx(), t2).startswith("D8"))
            t2 = dict(t, output_weight=torch.zeros((1024, 512), device="cuda"))
            self.assertTrue(call_decline(fake_ctx(), t2).startswith("D8"))
            with mock.patch.object(l11, "M_MAX", 999):
                self.assertTrue(call_decline(fake_ctx(), t).startswith("D8 M="))
            self.assertTrue(call_decline(fake_ctx(du_row_width=0), t).startswith("D9"))
            # B1a's env no longer declines (the backward routes B1a-emitting calls per call, see the seam).
            with mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_BWD_DU_E5M2": "1"}):
                # (l11e5 merge: no e5_capable argument and no D10 any more; the seam always has the e5 route.)
                self.assertIsNone(call_decline(fake_ctx(), t))
            l11.set_prebuilt_for_testing(False)
            self.assertTrue(call_decline(fake_ctx(), t).startswith("D11 not prebuilt"))
            self.assertTrue(call_decline(fake_ctx(du_row_width=0), t).startswith("D9"))  # contract reasons first
            l11.set_prebuilt_for_testing(True)
            with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
                self.assertTrue(call_decline(fake_ctx(), t).startswith("D3"))
        with mock.patch.object(l11, "device_arch", return_value="gfx942"):
            self.assertTrue(call_decline(fake_ctx(), t).startswith("D2 arch gfx942"))
        l11._broken = "x"
        self.assertTrue(call_decline(fake_ctx(), t).startswith("D1"))
        l11._broken = None
        l11.set_requested_for_testing(False)
        self.assertEqual(call_decline(fake_ctx(), t), "D0 knob off")
        self.assertEqual(l11.decline_key("D8 M=5000000 outside"), "D8 M")
        self.assertEqual(l11.decline_key("D11 not prebuilt before RUN_START"), "D11 not")
        self.assertEqual(l11.decline_key("D6 flags (need ...)"), "D6 flags")


# ------------------------------------------------------------------------------------------------ GPU tests
@unittest.skipUnless(_gfx950(), "needs gfx950")
class L11GpuTest(unittest.TestCase):
    def setUp(self) -> None:
        # The reference chain must be the LN*U*dropout backward v2 that L11 replicates op for op; the flag is read per
        # call, so pin it here (a plain `pytest` env would otherwise fall back to the v1 kernel and report a false
        # dx mismatch). Production is guarded by decline D7 and the prebuild self-check.
        self._env = mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_LNMD_BWD_V2": "1"})
        self._env.start()
        l11.set_requested_for_testing(True)
        l11.reset_for_testing(unload=False)

    def tearDown(self) -> None:
        l11.reset_for_testing(unload=False)
        l11.set_requested_for_testing(None)
        self._env.stop()

    def test_small_m_vs_chain(self) -> None:
        for M in (1, 2, 63, 64, 65, 1000, 32769):
            for mode in ("gauss", "heavy", "special"):
                r = compare_case(Case(M, mode))
                with self.subTest(M=M, mode=mode):
                    self.assertEqual(r["du_same_dy_real"], 0, r)
                    self.assertEqual(r["dx_same_dy_real"], 0, r)
                    self.assertEqual(r["nan_pos_diff"], 0, r)
                    if mode != "special":
                        self.assertEqual(r["du_same_dy"], 0, r)
                        self.assertEqual(r["dx_same_dy"], 0, r)
                    self.assertEqual(r["du_guard_changed"], 0, r)
                    self.assertEqual(r["du_unwritten"], 0, r)
                    self.assertEqual(r["du_dp_real"], 0, r)
                    self.assertEqual(r["dx_dp_real"], 0, r)
                    if mode != "special":  # NaN/Inf data: NaN positions must match; payload bits are not compared
                        self.assertEqual(r["dy_dp"], 0, r)
                        self.assertEqual(r["du_dp"], 0, r)
                        self.assertEqual(r["dx_dp"], 0, r)
                        self.assertEqual(r["nonfinite_partials"], 0, r)
                        self.assertLessEqual(r["dw_bf16_ne_chain"], 2, r)

    def test_partials_every_row_written(self) -> None:
        case = Case(32769, "gauss")
        out = case.fused()
        self.assertIsNotNone(out)
        G, fl, nf, P = l11.plan(case.M)
        self.assertEqual(P, G * 8 * nf)
        # NaN-prefilled partials through the C entry directly: no prefill word may survive
        dwp = torch.full((P, D), float("nan"), dtype=F32, device="cuda")
        dbp = torch.full((P, D), float("nan"), dtype=F32, device="cuda")
        dx = torch.full_like(case.X, float("nan"))
        du = torch.full((case.M, DU_ROW), float("nan"), dtype=BF, device="cuda")[:, :D]
        rc = l11._fn_launch(case.G8.data_ptr(), case.W8.data_ptr(), case.sa.data_ptr(), case.sb.data_ptr(), case.M,
                            case.X.data_ptr(), case.U.data_ptr(), case.MASK.data_ptr(), case.MEAN.data_ptr(),
                            case.RSTD.data_ptr(), case.Wn.data_ptr(), case.Bn.data_ptr(), dx.data_ptr(), du.data_ptr(),
                            dwp.data_ptr(), dbp.data_ptr(), P, l11.dropout_scale(P_DROP), 1.0 / D,
                            torch.cuda.current_stream().cuda_stream)
        self.assertEqual(rc, 0)
        torch.cuda.synchronize()
        self.assertEqual(int(torch.isnan(dwp).sum()) + int(torch.isnan(dbp).sum()), 0)
        self.assertEqual(int(torch.isnan(dx.float()).sum()), 0)
        # wrong P is rejected by the C entry (positive rc, nothing launched)
        rc = l11._fn_launch(case.G8.data_ptr(), case.W8.data_ptr(), case.sa.data_ptr(), case.sb.data_ptr(), case.M,
                            case.X.data_ptr(), case.U.data_ptr(), case.MASK.data_ptr(), case.MEAN.data_ptr(),
                            case.RSTD.data_ptr(), case.Wn.data_ptr(), case.Bn.data_ptr(), dx.data_ptr(), du.data_ptr(),
                            dwp.data_ptr(), dbp.data_ptr(), P + 8, l11.dropout_scale(P_DROP), 1.0 / D,
                            torch.cuda.current_stream().cuda_stream)
        self.assertGreater(rc, 0)

    def test_determinism(self) -> None:
        case = Case(65573, "heavy")
        outs = [case.fused() for _ in range(3)]
        for o in outs[1:]:
            for a, b in zip(outs[0][:4], o[:4]):
                self.assertEqual(int(bits_ne(a, b).sum()), 0)

    def test_side_stream(self) -> None:
        case = Case(65573, "gauss")
        ref = case.fused()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            x2 = case.X.clone()  # produced on the side stream, consumed there without a sync
            case2 = SimpleNamespace(**vars(case))
            case2.X = x2
            out = Case.fused(case2)  # pyre-ignore
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for a, b in zip(ref[:4], out[:4]):
            self.assertEqual(int(bits_ne(a, b).sum()), 0)

    def test_stale_error_is_not_a_failure(self) -> None:
        case = Case(1000, "gauss")
        self.assertIsNotNone(case.fused())
        real = l11._fn_launch

        def stale(*a):
            rc = real(*a)
            return -5 if rc == 0 else rc

        with mock.patch.object(l11, "_fn_launch", stale), captured() as out:
            r = case.fused()
        self.assertIsNotNone(r)
        self.assertIsNone(l11.broken())
        self.assertIn("stale HIP error 5", out.getvalue())

    def test_launch_failure_latches_and_falls_back(self) -> None:
        case = Case(1000, "gauss")
        self.assertIsNotNone(case.fused())  # library loaded (the patch below must not be undone by a lazy load)
        with mock.patch.object(l11, "_fn_launch", lambda *a: 1), captured() as out:
            self.assertIsNone(case.fused())
        self.assertIn("DISABLED: launch rc=1", out.getvalue())
        self.assertFalse(l11.enabled())
        self.assertIsNone(case.fused())  # latched: no further launch

    @unittest.skipUnless(hasattr(__import__("generative_recommenders.ops.triton.triton_hstu_linear", fromlist=["x"]),
                                 "_triton_layer_norm_mul_dropout_bwd_du_e5m2"), "needs B1a (ship-12 tree)")
    def test_e5_vs_b1a_same_dy(self) -> None:
        """l11e5: run_e5 / launch_e5 vs B1a's producer (v3 EMIT + finalize) on L11's own dy: e5m2 u block incl. the
        pad rows and the amax bitwise, columns 512.. of dz8 untouched, dx bitwise, record shape as B1a's; the u column
        sums are a declared fp32 order (close to v3's)."""
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        for M in (1, 65, 1000, 32769):
            for mode in ("gauss", "heavy", "special"):
                case = Case(M, mode)
                inv = torch.tensor([2.0 ** 20], dtype=F32, device="cuda")
                cs = torch.tensor([2.0 ** -20], dtype=F32, device="cuda")
                rows_out = (M + 127) // 128 * 128
                dz8 = torch.full((rows_out, DU_ROW), 0x5A, dtype=torch.uint8, device="cuda").view(E5)
                dx, dwp, dbp, ducs, amaxp, P = l11.launch_e5(case.G8, case.W8, case.sa, case.sb, case.X, case.U,
                                                             case.MASK, case.MEAN, case.RSTD, case.Wn, case.Bn, P_DROP,
                                                             inv, dz8)
                out = l11.run_e5(case.G8, case.W8, case.sa, case.sb, case.X, case.U, case.Wn, case.Bn, case.MEAN,
                                 case.RSTD, D, 2, P_DROP, True, case.MASK, DU_ROW, 1, inv, cs, None, log_active=False)
                dy = l11.dy_debug(case.G8, case.W8, case.sa, case.sb)
                ref = thl._triton_layer_norm_mul_dropout_bwd_du_e5m2(
                    dy=dy, x=case.X, u=case.U, weight=case.Wn, bias=case.Bn, mean=case.MEAN, rstd=case.RSTD,
                    BLOCK_D=D, num_warps=2, dropout_ratio=P_DROP, silu_u=True, random_mask=case.MASK,
                    du_row_width=DU_ROW, site_key=1, inv_scale=inv, cast_scale=cs, generation=None)
                torch.cuda.synchronize()
                with self.subTest(M=M, mode=mode):
                    self.assertIsNotNone(out, l11.broken())
                    rec, rrec = out[4], ref[4]
                    u8 = torch.uint8
                    self.assertTrue(torch.equal(dz8.view(u8)[:, :D], rrec.dz8.view(u8)[:, :D]))
                    self.assertTrue(torch.equal(rec.dz8.view(u8)[:, :D], rrec.dz8.view(u8)[:, :D]))
                    self.assertEqual(int((dz8.view(u8)[:, D:] != 0x5A).sum()), 0)
                    self.assertEqual(rec.u_amax.view(torch.int32).item(), rrec.u_amax.view(torch.int32).item())
                    self.assertEqual(int(bits_ne_real(dx, ref[0]).sum()), 0)
                    self.assertEqual((rec.rows, rec.width, rec.rows_out, rec.u_cols, rec.buffer_ptr),
                                     (rrec.rows, rrec.width, rrec.rows_out, rrec.u_cols, rrec.buffer_ptr))
                    self.assertEqual(out[1].stride(), ref[1].stride())
                    if mode != "special":
                        self.assertEqual(int(bits_ne(dx, ref[0]).sum()), 0)
                        self.assertLess(rel_l2(rec.u_colsum, rrec.u_colsum), 1e-5)

    def test_prebuild_self_check(self) -> None:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        with captured() as out:
            ok, detail = thl._l11_self_check()
        self.assertTrue(ok, detail)
        self.assertEqual(out.getvalue(), "")  # the self-check prints no ACTIVE line

    def test_prebuild_self_check_e5(self) -> None:
        # l11e5: the self-check's E5 part (forced on here; in the prebuild it runs when B1a is enabled): the E5
        # instantiation vs B1a's producer on the same dy and on hipBLASLt's dy, raw launch == record, guards.
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        with captured() as out:
            ok, detail = thl._l11_self_check(e5=True)
        self.assertTrue(ok, detail)
        self.assertIn("; e5 inv=", detail)
        self.assertEqual(out.getvalue(), "")  # no ACTIVE line of either mode


if __name__ == "__main__":
    unittest.main()

# pyre-unsafe
"""X2L-B (gin item (93); ops/hip/preproc_l2_x2lb.py + x2lb_pass.cu) on the GPU: DESIGN_stageC 8.2, the rung's UNIT
gate. The CPU half is preproc_l2_x2lb_cpu_test.py.

Needs gfx950 with hipcc, PR #7's FlyDSL and Triton; the tests skip only when the device is not gfx950. A missing
toolchain on gfx950 is an ERROR (the UNIT gate aborts). The class runs the real two-phase prebuild once (phase 1:
build, census, ISA gate, ABI; phase 2: the self-check), with E-M3a, the v2 LN forward and PR #7 switched on, then:

  G-a  da1 / dh1 bitwise against an independent data-parallel reference:
       - dp32 = hipBLASLt's top-1 run with a 32 MiB workspace, from the test-only x2lb_dpref copy of the lab's E0 kit;
       - the shipped chain = triton_fp8._scaled_mm + triton_weighted_layer_norm_bwd_dxsum(is_swish) (PR #7).
       Data-parallel points (2,356,560 / 2,487,418 / 2,507,492 / 2,546,646): ALL rows, no exemption, gauss and cancel;
       a point whose torch da1 differs from dp32 is re-classed as a stream-K point and reported, never passed by
       exemption. Stream-K points (2,424,832 / 2,492,120 / 2,520,152 / 2,787,229): S1+DA1 da1 == dp32 on all rows;
       S1 dh1 == the chain's dh1 outside the observed split rows (torch != dp32). S1 dh1 == PR #7(dy = S1+DA1's da1)
       on all rows everywhere. Without dp32, the stream-K points are NOT EVALUABLE, which is accepted only when
       phase 1's code object equals the lab's.
  G-b  finals (the R14 path: PR #7's finalize over X2L-B's PART) under the sealed G2c rule, per output:
       - relative: err <= 4 x PR #7's error x s(G) + 2^-23 max|ref|;
       - absolute: err <= gamma_h x sum|term| per column.
       At M_REF on gauss seeds 11-13, at M 70 and 16,383, and at G 7 / 64 / 256 (s(G) = max(1, h(G) / h(G_plan))).
       The absolute rule gates at M >= M_MIN (65,536, the integration's own minimum: below it tensor_reason
       declines) and is INFO below: gamma_h counts summation roundings only, and at a short summation path the
       per-term fp32 formation error (approximate exp / rcp, the s(1-s) cancellation) is not covered (UNIT job 70897:
       use 1.06 at M 70 G 2, the relative rule passing). The incumbent's own use against the same gamma is reported.
  G-c  end to end on a small model (the real E-M3a Functions under autocast(bf16)) at M 131,072 / 262,144 / 300,001
       (split-free by the replica), knob on vs off:
       - bitwise on every leaf except the three fp32-order outputs of DESIGN_stageC section 6 (norm0 weight / bias
         and the L1 bias of the two FP8 MLPs): W2, b2, norm1 w / b, the action MLP, and x / W1 (dh1-derived);
       - norm0 w / b and b1 (X2L-B's PART + PR #7's finalize vs PR #7's own partition: a different fp32 order) under
         the sealed G2c rule of G-b, against an fp64 oracle of the knob-off call's own pair-0 inputs;
       - every call fused (2 per backward, no DECLINED line), 3 fused reruns bitwise among themselves;
       - retain_graph x2: knob on == 2 x knob on x1 on every leaf, == knob off x2 on the bitwise set;
       - the eval forward unchanged, allocated bytes back after the step.
       (Review round 1, MAJOR-2: DESIGN 8.2's G-c sentence listing b1 and norm0 w / b as bitwise contradicts
       section 6; section 6 is followed here.)
  G-d  canaries: rows >= M of dh1 / da1 untouched, every PART word written.
  G-e  the prebuild: lines, census, the peak transient, memory_reserved / memory_allocated back after empty_cache, hipBLASLt
       solution A / B at the three check points, and a g3_kperm library refused by the ISA gate and failing phase-2
       check (a). The shipped chain (torch's hipBLASLt workspace, PR #7's ln_bwd, the TLN dispatch) is primed before
       the baseline, as production runs blas_warmup / E-M3a / PR #7's prebuild before phase 2 (UNIT job 70897 without
       the prime: reserved +130 MiB = those libraries' persistent allocations, not phase 2's transient).
  G-f  non-finite: the lab's nonfinite set gives the chain's NaN / inf positions in dh1 and in the finals.
  G-g  the in-model shadow knob: NOT IMPLEMENTED in this patch (skipped with that reason).

Each point prints one "X2LB_GPU <json>" line for the UNIT gate's log. X2LB_GPU_QUICK=1 keeps only the first point of
each list (a smoke run, never the gate).
"""

import importlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch

ENV = "DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD"
_HERE = os.path.dirname(os.path.abspath(__file__))
BF, F32, E4, E5 = torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e5m2
SWISH_EPS = 1e-5
M_REF = 2_492_120
DP_POINTS = (2_356_560, 2_487_418, 2_507_492, 2_546_646)  # production grid data-parallel (replica sweep)
SK_POINTS = (2_424_832, 2_492_120, 2_520_152, 2_787_229)  # grid 256 / split tiles
GC_POINTS = (131_072, 262_144, 300_001)
QUICK = os.environ.get("X2LB_GPU_QUICK") == "1"
CH = 1 << 18  # data generation chunk (labkit x2lb_ab.CH)


def _gfx950() -> bool:
    try:
        return (torch.cuda.is_available() and torch.version.hip is not None
                and str(torch.cuda.get_device_properties(0).gcnArchName).split(":")[0] == "gfx950")
    except Exception:
        return False


def _pts(p):
    return p[:1] if QUICK else p


def _say(tag, **kw):
    print(f"X2LB_GPU {json.dumps(dict(test=tag, **kw), default=str)}", flush=True)


def _load_dpref():
    spec = importlib.util.spec_from_file_location("x2lb_dpref_dpref", os.path.join(_HERE, "x2lb_dpref", "dpref.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ data (lab kit ports)
def _gen(dev, seed):
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed) % (1 << 62))
    return g


def _dx(s, e, seed, dev):
    """labkit x2lb_ab._dx(kind gauss): dx rows [s, e) fp32 [n, 512], regenerated bit-identically per chunk."""
    g = _gen(dev, seed * 1_000_003 + s // CH)
    n = e - s
    sc = 10.0 ** (torch.rand((n, 1), generator=g, device=dev) * 3.0 - 4.0)
    x = torch.randn((n, 512), generator=g, device=dev) * sc
    hv = torch.rand((n, 512), generator=g, device=dev) < 0.002
    return torch.where(hv, x * 30.0, x)


def gauss(M, seed, dev):
    """labkit x2lb_ab.make_data(kind="gauss"): production quantization of real-like operands (CODEMAP 4.1 / 5)."""
    amax = 0.0
    for s in range(0, M, CH):
        x = _dx(s, min(M, s + CH), seed, dev)
        amax = max(amax, float(x.abs().max().item()))
        del x
    g = _gen(dev, seed * 7 + 11)
    u = float(torch.rand((1,), generator=g, device=dev).item())
    sa = torch.tensor([max(amax * (1.0 + 0.5 * u), 1e-30) * 8.0 / 57344.0], dtype=F32, device=dev)
    inv = 1.0 / sa
    dh8 = torch.empty((M, 512), dtype=E5, device=dev)
    for s in range(0, M, CH):
        e = min(M, s + CH)
        dh8[s:e] = (_dx(s, e, seed, dev) * inv).clamp_(-57344.0, 57344.0).to(E5)
    gw = _gen(dev, seed * 7 + 3)
    W = torch.randn((256, 512), generator=gw, device=dev) * 0.05
    sb = W.abs().max().float().reshape(1) * torch.tensor([1.0 / 448.0], dtype=F32, device=dev)
    w8t = (W * (1.0 / sb)).clamp_(-448.0, 448.0).to(E4).contiguous()
    h1 = torch.empty((M, 256), dtype=BF, device=dev)
    mean1 = torch.empty((M,), dtype=F32, device=dev)
    rstd1 = torch.empty((M,), dtype=F32, device=dev)
    for s in range(0, M, CH):
        e = min(M, s + CH)
        gh = _gen(dev, seed * 1_000_003 + 500_000 + s // CH)
        n = e - s
        sc = 0.5 + 1.5 * torch.rand((n, 1), generator=gh, device=dev)
        off = torch.rand((n, 1), generator=gh, device=dev) - 0.5
        h = (torch.randn((n, 256), generator=gh, device=dev) * sc + off).to(BF)
        h1[s:e] = h
        hd = h.double()
        mean1[s:e] = hd.mean(1).float()
        rstd1[s:e] = (1.0 / torch.sqrt(hd.var(1, unbiased=False) + 1e-5)).float()
        del hd, h
    gn = _gen(dev, seed * 7 + 5)
    nw = (1.0 + 0.2 * torch.randn((256,), generator=gn, device=dev)).float().contiguous()
    nb = (0.3 * torch.randn((256,), generator=gn, device=dev)).float().contiguous()
    return dict(dh8=dh8, w8t=w8t, sa=sa.contiguous(), sb=sb.contiguous(), h1=h1, mean1=mean1, rstd1=rstd1, nw=nw,
                nb=nb)


def nonfinite(d, seed, dev):
    """labkit x2lb_ab._nonfinite: NaN / +-inf e5m2 codes at 3 positions of every 97th row (from row 5)."""
    u = d["dh8"].view(torch.uint8)
    M = u.shape[0]
    g = _gen(dev, seed * 41 + 3)
    rows = torch.arange(5, M, 97, device=dev)
    n = int(rows.numel())
    cols = torch.randint(0, 512, (n, 3), generator=g, device=dev, dtype=torch.int32).long()
    codes = torch.tensor([0x7C, 0xFC, 0x7E, 0x7F, 0xFD, 0xFE], dtype=torch.uint8, device=dev)
    pick = codes[torch.randint(0, 6, (n, 3), generator=g, device=dev, dtype=torch.int32).long()]
    u[rows.view(n, 1).expand(n, 3), cols] = pick
    return n


def ref_partials64(d, da1, M, chunk=1 << 17):
    """labkit x2lb_ab.ref_partials64: fp64 (dW, dB, dXS) from the same inputs in PR #7's op list, and sum |term| per
    column."""
    dev = da1.device
    acc = torch.zeros((3, 256), dtype=torch.float64, device=dev)
    aab = torch.zeros((3, 256), dtype=torch.float64, device=dev)
    w, b = d["nw"].double(), d["nb"].double()
    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        x = d["h1"][s:e].double()
        dy = da1[s:e].double()
        rs = d["rstd1"][s:e].double().view(-1, 1)
        mu = d["mean1"][s:e].double().view(-1, 1)
        xhat = (x - mu) * rs
        sig = torch.sigmoid(xhat * w + b)
        t = dy * x * (sig - sig * sig)
        g = w * t
        c1 = (xhat * g).sum(1, keepdim=True) / 256.0
        c2 = g.sum(1, keepdim=True) / 256.0
        dx = dy * sig + (g * rs - (xhat * c1 + c2) * rs)
        for o, v in enumerate((t * xhat, t, dx)):
            acc[o] += v.sum(0)
            aab[o] += v.abs().sum(0)
    return acc, aab


def h_path(G, ntiles):
    """labkit x2lb_ab.h_path: roundings on one leaf's path in X2L-B's partial order."""
    return 4 * (-(-int(ntiles) // int(G))) + 4 + (int(G) - 1)


def gamma(h):
    u = 2.0 ** -24
    return h * u / (1.0 - h * u)


def view(d, M):
    return {k: (v[:M] if k in ("dh8", "h1", "mean1", "rstd1") else v) for k, v in d.items()}


def nbad_rows(a, b):
    """Rows of two 2-D 16-bit tensors that differ as integers (-0 != +0, NaN payloads exact)."""
    return (a.view(torch.int16) != b.view(torch.int16)).any(dim=1)


@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU")
class X2LBGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        importlib.import_module("triton")  # a missing toolchain on gfx950 is an error, not a skip
        from generative_recommenders.ops import preproc_fp8_l2 as L2
        from generative_recommenders.ops.flydsl import ln_bwd as LB
        from generative_recommenders.ops.hip import preproc_l2_x2lb as X
        from generative_recommenders.ops.triton import triton_fp8 as TF8
        from generative_recommenders.ops.triton import triton_layer_norm as TLN

        cls.L2, cls.LB, cls.X, cls.TF8, cls.TLN = L2, LB, X, TF8, TLN
        cls.env = mock.patch.dict(os.environ, {ENV: "1", "DLRM_HSTU_FP8_PREPROC_L2": "1",
                                               "DLRM_HSTU_FP8_WLN_FWD_V2": "1", "DLRM_HSTU_FP8_WLN_BWD_FLYDSL": "1"})
        cls.env.start()
        cls.addClassCleanup(cls.env.stop)
        for k in (X.MLPS_ENV_VAR, X.FIN_ENV_VAR):
            os.environ.pop(k, None)
        L2.reset_for_testing()
        TLN._WLN_BWD_FLYDSL_ENABLED = None
        TLN._WLN_FWD_V2_ENABLED = None
        X.reset_for_testing()
        cls.dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.synchronize()
        cls.p1 = X.prebuild_build(cls.dev)
        if not cls.p1:
            raise RuntimeError(f"phase 1 did not pass: {X.broken()} (P1 {X._P1_STATE})")
        torch.cuda.empty_cache()
        mem = dict(before_prime=(torch.cuda.memory_allocated(cls.dev), torch.cuda.memory_reserved(cls.dev)))
        # Prime the shipped chain phase 2 calls (torch's hipBLASLt workspace on this stream, PR #7's ln_bwd, TLN's
        # dispatch) at a small M: production runs blas_warmup, E-M3a's warm-up and PR #7's prebuild before phase 2,
        # so their persistent allocations exist before it (module docstring, G-e).
        dp = gauss(1 << 16, 3, cls.dev)
        da1p = TF8._scaled_mm(dp["dh8"], dp["w8t"].t(), dp["sa"], dp["sb"], out_dtype=BF)
        LB.ln_bwd(da1p, dp["h1"], dp["nw"], dp["nb"], dp["mean1"], dp["rstd1"], is_swish=True, sum_dx=True)
        TLN.triton_weighted_layer_norm_bwd_dxsum(
            dy=da1p.contiguous(), x=dp["h1"], weight=dp["nw"], bias=dp["nb"], mean=dp["mean1"], rstd=dp["rstd1"],
            eps=SWISH_EPS, BLOCK_D=TLN.compute_BLOCK_D(dp["h1"]), is_swish=True)
        torch.cuda.synchronize()
        del dp, da1p
        mem["after_prime"] = (torch.cuda.memory_allocated(cls.dev), torch.cuda.memory_reserved(cls.dev))
        torch.cuda.empty_cache()
        cls.reserved0 = torch.cuda.memory_reserved(cls.dev)
        cls.alloc0 = torch.cuda.memory_allocated(cls.dev)
        torch.cuda.reset_peak_memory_stats(cls.dev)
        cls.token = X.prebuild_check(cls.dev)
        torch.cuda.synchronize()
        cls.peak_alloc = torch.cuda.max_memory_allocated(cls.dev)
        torch.cuda.empty_cache()  # blas_warmup releases the transient the same way
        cls.reserved1 = torch.cuda.memory_reserved(cls.dev)
        cls.alloc1 = torch.cuda.memory_allocated(cls.dev)
        mem.update(base=(cls.alloc0, cls.reserved0), peak_alloc=cls.peak_alloc, after_phase2=(cls.alloc1, cls.reserved1))
        cls.mem = mem
        if cls.token != "x2lb=1":
            raise RuntimeError(f"phase 2: {cls.token}: {X.broken()}")
        cls.lib = X._lib
        cls.ncu = X._RT.num_cu(cls.dev)
        cls.co_lab = (X.PHASE1_INFO.get("code_object") or {}).get("lab") == "identical"
        try:
            cls.dp = _load_dpref().DpRef(torch)
            cls.dp_err = None
        except Exception as exc:  # the UNIT gate decides what that means (module docstring, G-a)
            cls.dp, cls.dp_err = None, f"{type(exc).__name__}: {exc}"[:300]
        _say("setup", token=cls.token, phase1=X.PHASE1_INFO, prebuild=X.PREBUILD_INFO, ncu=cls.ncu,
             dp=cls.dp_err or "ok", code_object_lab=cls.co_lab, mem=cls.mem)

    @classmethod
    def tearDownClass(cls):
        cls.X.reset_for_testing()
        cls.L2.reset_for_testing()
        cls.TLN._WLN_BWD_FLYDSL_ENABLED = None
        cls.TLN._WLN_FWD_V2_ENABLED = None

    def setUp(self):
        self.assertEqual(self.token, "x2lb=1")
        self.assertIsNone(self.X.broken())

    def tearDown(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- helpers
    def canary(self, rows, dtype=BF):
        t = torch.empty((rows + self.X.CANARY_ROWS, 256), dtype=dtype, device=self.dev)
        t.view(torch.int16).fill_(self.X._BF16_CANARY)
        return t

    def part(self, G):
        p = torch.empty((3, G, 256), dtype=F32, device=self.dev)
        p.view(torch.int32).fill_(self.X._F32_CANARY)
        return p

    def emit(self, d, M, G):
        """S1+DA1 into canary-filled da1 / dh1 / PART (the lab's emit instance)."""
        X = self.X
        da1, dh1, part = self.canary(M), self.canary(M), self.part(G)
        st = torch.cuda.current_stream(self.dev)
        X._RT.launch(self.lib, "emit", d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"], d["nw"],
                     d["nb"], dh1, da1, part, M, G, st)
        return da1, dh1, part

    def s1(self, d, M, G):
        X = self.X
        dh1, part = self.canary(M), self.part(G)
        st = torch.cuda.current_stream(self.dev)
        X._RT.launch(self.lib, "s1", d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"], d["nw"],
                     d["nb"], dh1, None, part, M, G, st)
        return dh1, part

    def assertCanaries(self, t, M, what):
        bad = int((t[M:].view(torch.int16) != self.X._BF16_CANARY).sum().item())
        self.assertEqual(bad, 0, f"{what}: {bad} canary words past row M {M} written")

    def assertWritten(self, part, what):
        left = int((part.view(torch.int32) == self.X._F32_CANARY).sum().item())
        self.assertEqual(left, 0, f"{what}: {left} PART words never written")

    def torch_da1(self, d):
        return self.TF8._scaled_mm(d["dh8"], d["w8t"].t(), d["sa"], d["sb"], out_dtype=BF)

    def chain(self, d, dy):
        """The shipped pair-0 backward: TLN's dispatch to PR #7 (asserting PR #7 owns the call)."""
        TLN = self.TLN
        why = TLN.flydsl_ln_bwd_decline_reason(d["h1"], dy, d["nw"], d["nb"], d["mean1"], d["rstd1"], None,
                                               is_swish=True, sum_dx=True)
        self.assertIsNone(why, "PR #7 does not own the shipped call: the chain would be Triton")
        return TLN.triton_weighted_layer_norm_bwd_dxsum(
            dy=dy.contiguous(), x=d["h1"], weight=d["nw"], bias=d["nb"], mean=d["mean1"], rstd=d["rstd1"],
            eps=SWISH_EPS, BLOCK_D=TLN.compute_BLOCK_D(d["h1"]), is_swish=True)

    def pr7(self, d, da1):
        return self.LB.ln_bwd(da1, d["h1"], d["nw"], d["nb"], d["mean1"], d["rstd1"], is_swish=True, sum_dx=True)

    # ---------------------------------------------------------------- G-e
    def test_ge_prebuild(self):
        X = self.X
        info = X.PHASE1_INFO
        res = info["resources"]
        self.assertEqual((res["lds"], res["occupancy"], res["vgpr_spill"], res["scratch"]), (99328, 2, 0, 0))
        self.assertLessEqual(res["unified_vgpr"], X.LIM_VGPR)
        self.assertTrue(info["isa"]["ok"])
        p2 = X.PREBUILD_INFO[self.dev.index]
        pts = {p["M"]: p for p in p2["points"]}
        self.assertEqual(sorted(pts), sorted(M for M, _ in X.CHECK_POINTS))
        for M, rule in X.CHECK_POINTS:
            self.assertEqual(pts[M]["rule"], rule)
            self.assertEqual(pts[M]["grid"], self.lib.plan(M, self.ncu))
        U = X.M_B * X.KD
        G = pts[X.M_B]["grid"]
        self.assertEqual(p2["peak_bytes"], 5 * U + 8 * X.M_B + 2 * X.CANARY_ROWS * 512 + 3 * G * 256 * 4)
        # + the hash generator's chunk temporaries and allocator rounding
        self.assertLessEqual(self.peak_alloc - self.alloc0, p2["peak_bytes"] + (1 << 30))
        self.assertLessEqual(self.reserved1, self.reserved0 + (2 << 20), f"memory {self.mem}")  # released by empty_cache
        # the per-device zero sentinel scalar (2 B) is the only allocation phase 2 keeps
        self.assertLessEqual(self.alloc1, self.alloc0 + (1 << 20), f"memory {self.mem}")
        _say("G-e", phase1=info, phase2=p2, peak_alloc=self.peak_alloc, reserved=(self.reserved0, self.reserved1))
        # hipBLASLt solution A at M_DP / M_A, B at M_B (open question 3), on the phase-2 data
        dpref = _load_dpref()
        for M, want in ((X.M_DP, "256x256x128"), (X.M_A, "256x256x128"), (X.M_B, "256x240x128")):
            live = X._Live()
            d = X.cancel_data(M, self.dev, live)
            kn = dpref.kernel_name(torch, lambda d=d: self.torch_da1(d))
            _say("G-e kernel", M=M, kernel=kn)
            self.assertTrue(dpref.contract_kernel(kn) and dpref.tile_of(kn) == want, f"M {M}: {kn}")
            del d, live

    def test_ge_g3_kperm_refused(self):
        """A g3_kperm library (variant 1, the lab's K-order negative control) forced through test hooks: the ISA gate
        refuses it (C3), and phase-2 check (a) fails on the hash-built cancel data (open item O3)."""
        X = self.X
        tmp = tempfile.mkdtemp(prefix="x2lb_kperm_")
        self.addCleanup(shutil.rmtree, tmp, True)
        os.chmod(tmp, 0o700)
        with mock.patch.dict(os.environ, {X.CACHE_ENV_VAR: tmp}), \
                mock.patch.object(X, "_FLAGS", X._FLAGS + ("-DX2LB_G3_KPERM=1",)):
            with self.assertRaisesRegex(RuntimeError, "ISA gate refuses this build") as cm:
                X.build()
            self.assertIn("C3", str(cm.exception))
            ok = dict(ok=True, mode="forced (test)", asm_norm_sha256="", kernels={}, rc=0,
                      checker=dict(X.ISA_FILES_SHA256), seconds=0.0, tail=None)
            with mock.patch.object(X, "isa_gate", return_value=ok):
                so, meta, s, cached = X.build(fresh=True)
            self.addCleanup(shutil.rmtree, str(so.parent), True)
        with mock.patch.object(X, "VARIANT", 1):
            lib = X._Lib(so, meta, s, cached)
        self.assertEqual(int(lib.x2lb_variant()), 1)
        with mock.patch.object(X, "_LB", self.LB):
            with self.assertRaisesRegex(RuntimeError, r"^\(a\) S1\+DA1 da1 != hipBLASLt"):
                X.check_point(lib, X.M_A, "dp_region", self.dev, X._Live())

    # ---------------------------------------------------------------- G-a
    def ga_point(self, kind, M, cls):
        X = self.X
        d = view(gauss(M, 1234, self.dev), M) if kind == "gauss" else X.cancel_data(M, self.dev, X._Live())
        G = self.lib.plan(M, self.ncu)
        da1_t = self.torch_da1(d)
        da1_x, dh1_x, part_x = self.emit(d, M, G)
        self.assertCanaries(da1_x, M, "S1+DA1 da1")
        self.assertCanaries(dh1_x, M, "S1+DA1 dh1")
        self.assertWritten(part_x, "S1+DA1 PART")
        res, grid, _ = X._RT.fused(self.lib, d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"],
                                   d["nw"], d["nb"])  # the production call: S1 + PR #7's finalize
        self.assertEqual(grid, G)
        dh1 = res[0]
        rec = dict(kind=kind, M=M, grid=G, cls=cls)
        # S1 == S1+DA1 and S1+DA1 dh1 == PR #7(dy = S1+DA1's da1), all rows
        self.assertEqual(int(nbad_rows(dh1, dh1_x[:M]).sum()), 0, f"{rec}: S1 dh1 != S1+DA1 dh1")
        dx_p = self.pr7(d, da1_x[:M])[0]
        self.assertEqual(int(nbad_rows(dh1_x[:M], dx_p).sum()), 0, f"{rec}: dh1 != PR #7(own da1)")
        del dx_p
        ref = None
        if self.dp is not None:
            ref, dinfo = self.dp.dp32(d["dh8"], d["w8t"], d["sa"], d["sb"])
            rec["dp32"] = dinfo
            if ref is not None and not dinfo["contract"]:
                ref = None  # dp32's top-1 is not a contract solution here: not a reference
        split = nbad_rows(da1_t, ref) if ref is not None else None
        rec["split_rows"] = None if split is None else int(split.sum())
        if cls == "dp" and rec["split_rows"]:
            rec["cls"] = cls = "sk (re-classed: torch != dp32 at a data-parallel point)"
        chain_dh = self.chain(d, da1_t)[0]
        if cls == "dp":
            self.assertEqual(int(nbad_rows(da1_x[:M], da1_t).sum()), 0, f"{rec}: da1 != torch (all rows)")
            self.assertEqual(int(nbad_rows(dh1, chain_dh).sum()), 0, f"{rec}: dh1 != chain (all rows)")
            rec["result"] = "all rows bitwise"
        elif ref is None:
            rec["result"] = f"NOT EVALUABLE (no dp32: {self.dp_err or rec.get('dp32')})"
            if not self.co_lab:
                self.fail(f"{rec}: a stream-K point is not evaluable and the code object differs from the lab's")
        else:
            self.assertEqual(int(nbad_rows(da1_x[:M], ref).sum()), 0, f"{rec}: da1 != dp32 (all rows)")
            keep = ~split
            self.assertEqual(int(nbad_rows(dh1[keep], chain_dh[keep]).sum()), 0,
                             f"{rec}: dh1 != chain outside the observed split rows")
            rec["result"] = "da1 == dp32 all rows; dh1 == chain outside split rows"
        _say("G-a", **rec)

    def test_ga_data_parallel_points(self):
        for M in _pts(DP_POINTS):
            for kind in ("gauss", "cancel"):
                with self.subTest(M=M, kind=kind):
                    self.ga_point(kind, M, "dp")
                torch.cuda.empty_cache()

    def test_ga_stream_k_points(self):
        for M in _pts(SK_POINTS):
            for kind in ("gauss", "cancel"):
                with self.subTest(M=M, kind=kind):
                    self.ga_point(kind, M, "sk")
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- G-b
    def finals_rule(self, got, fly, ref, aab, G, G_plan, M, what):
        ntiles = -(-M // 64)
        s = max(1.0, h_path(G, ntiles) / h_path(G_plan, ntiles))
        gam = gamma(h_path(G, ntiles))
        uses = []
        for o, name in enumerate(("dW", "dB", "dXS")):
            r, f, x = ref[o], fly[o].double(), got[o].double()
            if not (bool(torch.isfinite(r).all()) and bool(torch.isfinite(f).all())):
                self.fail(f"{what} {name}: NOT EVALUABLE (reference or chain not finite)")
            self.assertTrue(bool(torch.isfinite(x).all()), f"{what} {name}: X2L-B finals not finite")
            err = float((x - r).abs().max())
            rel = 4.0 * float((f - r).abs().max()) * s + 2.0 ** -23 * float(r.abs().max())
            self.assertLessEqual(err, rel, f"{what} {name}: err {err:.3g} > relative bound {rel:.3g}")
            ab = float(((x - r).abs() / (gam * aab[o] + 1e-300)).max())
            ab_pr7 = float(((f - r).abs() / (gam * aab[o] + 1e-300)).max())  # INFO: the incumbent, same gamma
            gating = M >= self.X.M_MIN  # module docstring, G-b: below M_MIN the absolute rule is INFO
            if gating:
                self.assertLessEqual(ab, 1.0, f"{what} {name}: absolute bound use {ab:.3g} (PR #7's own {ab_pr7:.3g})")
            uses.append(dict(o=name, rel_use=err / rel if rel else 0.0, abs_use=ab, abs_use_pr7=ab_pr7,
                             abs_gating=gating))
        return uses

    def gb(self, d, M, G=None):
        X = self.X
        G_plan = self.lib.plan(M, self.ncu)
        G = G_plan if G is None else G
        da1_x, dh1_x, _ = self.emit(d, M, G_plan)
        dh1, part = self.s1(d, M, G)
        st = torch.cuda.current_stream(self.dev)
        got = X._RT.fin(self.lib, "pr7", part[:, :G], G, st)  # PR #7's finalize over X2L-B's PART (R14)
        fly = self.pr7(d, da1_x[:M])[1:]  # the FlyDSL chain on the same da1
        ref, aab = ref_partials64(d, da1_x, M)
        if G != G_plan:
            self.assertEqual(int(nbad_rows(dh1[:M], dh1_x[:M]).sum()), 0, f"grid {G} dh1 != grid {G_plan} dh1")
        return self.finals_rule(got, fly, ref, aab, G, G_plan, M, f"M {M} G {G}")

    def test_gb_finals(self):
        for seed in _pts((11, 12, 13)):
            d = gauss(M_REF, seed, self.dev)
            _say("G-b", M=M_REF, seed=seed, uses=self.gb(d, M_REF))
            if seed == 11:
                for G in (7, 64, 256):
                    _say("G-b", M=M_REF, seed=seed, G=G, uses=self.gb(d, M_REF, G))
            del d
            torch.cuda.empty_cache()
        for M in (70, 16_383):
            d = view(gauss(M, 11, self.dev), M)
            _say("G-b", M=M, uses=self.gb(d, M))

    # ---------------------------------------------------------------- G-d
    def test_gd_canaries(self):
        for M in (16_383, 300_001, M_REF):
            d = view(gauss(M, 21, self.dev), M)
            G = self.lib.plan(M, self.ncu)
            da1, dh1, part = self.emit(d, M, G)
            dh1_s, part_s = self.s1(d, M, G)
            for t, w in ((da1, "S1+DA1 da1"), (dh1, "S1+DA1 dh1"), (dh1_s, "S1 dh1")):
                self.assertCanaries(t, M, f"M {M} {w}")
            self.assertWritten(part, f"M {M} S1+DA1 PART")
            self.assertWritten(part_s, f"M {M} S1 PART")
            self.assertTrue(torch.equal(part.view(torch.int32), part_s.view(torch.int32)))
            del d, da1, dh1, part, dh1_s, part_s

    # ---------------------------------------------------------------- G-f
    def test_gf_nonfinite(self):
        M = 262_144
        d = view(gauss(M, 5678, self.dev), M)
        n = nonfinite(d, 5678, self.dev)
        self.assertGreater(n, 0)
        G = self.lib.plan(M, self.ncu)
        da1_x, dh1_x, _ = self.emit(d, M, G)
        res, _, _ = self.X._RT.fused(self.lib, d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"],
                                     d["nw"], d["nb"])
        dx_p, dw_p, db_p, dxs_p = self.pr7(d, da1_x[:M])
        for got, want, what in ((res[0], dx_p, "dh1"), (res[1], dw_p, "dW"), (res[2], db_p, "dB"),
                                (res[3], dxs_p, "dXS")):
            for f in (torch.isnan, torch.isinf):
                self.assertTrue(torch.equal(f(got.float()), f(want.float())), f"{what}: {f.__name__} positions")
        fin = torch.isfinite(dx_p.float()).all(dim=1)
        self.assertEqual(int(nbad_rows(res[0][fin], dx_p[fin]).sum()), 0, "finite dh1 rows bitwise")
        _say("G-f", M=M, nonfinite_rows=n, nonfinite_dh1_rows=int((~fin).sum()))

    # ---------------------------------------------------------------- G-c
    def model(self, M, seed, train=True):
        """Content / additional through E-M3a's FP8 pair 0 (sites pre-published with private scales), the action
        MLP as a BF16 entry, one FP8 grouped tail, under autocast(bf16); fp32 master weights."""
        L2 = self.L2
        L2._sites.clear()
        g = torch.Generator(device=self.dev).manual_seed(seed)
        dev = self.dev

        def rn(*s):
            return torch.randn(*s, generator=g, device=dev)

        leaves, plans, flat = {}, [], []
        with torch.autocast("cuda", dtype=BF):
            for k, din in ((0, 512), (1, 1024), (2, 24)):
                x = rn(M, din).to(BF).requires_grad_()
                w1 = (rn(256, din) * 0.05).requires_grad_()
                b1 = (rn(256) * 0.1).requires_grad_()
                nw0 = (1 + 0.1 * rn(256)).requires_grad_()
                nb0 = (0.1 * rn(256)).requires_grad_()
                w2 = (rn(512, 256) * 0.05).requires_grad_()
                b2 = (rn(512) * 0.1).requires_grad_()
                nw1 = (1 + 0.1 * rn(512)).requires_grad_()
                nb1 = (0.1 * rn(512)).requires_grad_()
                leaves.update({f"{k}.{n}": t for n, t in (("x", x), ("w1", w1), ("b1", b1), ("nw0", nw0),
                                                           ("nb0", nb0), ("w2", w2), ("b2", b2), ("nw1", nw1),
                                                           ("nb1", nb1))})
                slot = L2.Slot(k)
                slot.train = train
                if k in L2.FP8_MLPS:
                    slot.mode = "fp8"
                    for role, sc in (("a1", 4.0 / 448.0), ("dh2", 1e-4)):
                        s = L2._site(role, k, dev)
                        s.scale.fill_(sc)
                        s.inv_scale.fill_(1.0 / sc)
                        s.published = True
                    y = L2._LinearSwishLNEmitFunction.apply(x, w1, b1, nw0, nb0, SWISH_EPS, k,
                                                            L2._emu_for(types_ns(nw0, nb0))[0], train, slot, None,
                                                            None)
                else:
                    h = torch.addmm(b1.to(BF), x, w1.to(BF).t())
                    y = h * torch.sigmoid(h)
                plans.append(slot)
                flat += [y, w2, b2, nw1, nb1]
                del y
            out = L2.grouped_tail_apply(1e-5, plans, flat)
        del flat
        return leaves, out

    def step(self, M, seed, knob, retain=False, capture=False):
        """One forward + backward. capture (knob off only): also the inputs of each pair 0's shipped SwishLN backward
        {k: d} (dy = da1, h1, mean1, rstd1, norm0 w / b), for the fp64 oracle of the fp32-order outputs."""
        X, L2 = self.X, self.L2
        L2._X2LB_REQUESTED = knob
        before = dict(X.FUSED_CALLS)
        caps = []
        real = L2.triton_weighted_layer_norm_bwd_dxsum

        def rec(*, dy, x, weight, bias, mean, rstd, is_swish, **kw):
            if is_swish:
                caps.append(dict(da1=dy, h1=x, mean1=mean, rstd1=rstd, nw=weight, nb=bias))
            return real(dy=dy, x=x, weight=weight, bias=bias, mean=mean, rstd=rstd, is_swish=is_swish, **kw)

        with mock.patch.object(L2, "triton_weighted_layer_norm_bwd_dxsum", rec if capture else real):
            leaves, out = self.model(M, seed)
            loss = (out.float() * torch.linspace(-1, 1, 512, device=self.dev)).sum()
            if retain:
                loss.backward(retain_graph=True)
            loss.backward()
        torch.cuda.synchronize()
        grads = {n: None if t.grad is None else t.grad.clone() for n, t in leaves.items()}
        fused = {k: X.FUSED_CALLS.get(k, 0) - before.get(k, 0) for k in (0, 1)}
        if not capture:
            return grads, fused
        by_k = {}
        for d in caps:  # identify k by the saved norm0 weight (the leaf itself)
            ks = [k for k in (0, 1) if d["nw"].data_ptr() == leaves[f"{k}.nw0"].data_ptr()]
            self.assertEqual(len(ks), 1, "a pair-0 SwishLN backward that is not MLP 0's or 1's")
            by_k[ks[0]] = d
        self.assertEqual(sorted(by_k), [0, 1])
        return grads, fused, by_k

    def gc_check(self, ref, got, caps, M, what):
        """Knob off (ref) vs on (got): bitwise outside GC_ORDER; GC_ORDER under the sealed G2c rule (finals_rule) with
        the fp64 oracle of the knob-off call's own inputs (dy = its da1; bitwise the fused call's da1 on split-free
        M, which the bitwise x / W1 gradients confirm)."""
        bad = [n for n in ref if n not in GC_ORDER and not _bits_equal(ref[n], got[n])]
        self.assertEqual(bad, [], f"{what}: knob on != off on the bitwise set")
        G = self.lib.plan(M, self.ncu)
        uses = {}
        for k in (0, 1):
            d = caps[k]
            oracle, aab = ref_partials64(d, d["da1"], M)
            fly = tuple(ref[f"{k}.{n}"] for n in ("nw0", "nb0", "b1"))
            x2 = tuple(got[f"{k}.{n}"] for n in ("nw0", "nb0", "b1"))
            for t in fly + x2:
                self.assertEqual((t.dtype, tuple(t.shape)), (F32, (256,)))
            uses[k] = self.finals_rule(x2, fly, oracle, aab, G, G, M, f"{what} mlp {k}")
        return uses

    def test_gc_end_to_end(self):
        X = self.X
        said = []
        real_say = X._say
        with mock.patch.object(X, "_say", lambda m: (said.append(m), real_say(m))):
            for M in _pts(GC_POINTS):
                with self.subTest(M=M):
                    ref, f0, caps = self.step(M, 31, False, capture=True)
                    self.assertEqual(f0, {0: 0, 1: 0})
                    first = None
                    for rerun in range(3):
                        torch.cuda.synchronize()
                        a0 = torch.cuda.memory_allocated(self.dev)
                        got, f1 = self.step(M, 31, True)
                        torch.cuda.synchronize()
                        held = sum(g.untyped_storage().nbytes() for g in got.values() if g is not None)
                        # the handle's results are gone after the step: only the gradients stay allocated
                        self.assertLessEqual(torch.cuda.memory_allocated(self.dev) - a0 - held, 8 << 20)
                        self.assertEqual(f1, {0: 1, 1: 1}, "every call fused")
                        if first is None:
                            first = got
                            uses = self.gc_check(ref, got, caps, M, f"M {M}")
                        else:  # fused vs fused: bitwise on EVERY leaf (deterministic at fixed M, #CU)
                            bad = [n for n in first if not _bits_equal(first[n], got[n])]
                            self.assertEqual(bad, [], f"M {M} rerun {rerun}: fused runs differ")
                    del caps
                    r2, _ = self.step(M, 31, False, retain=True)
                    g2, f2 = self.step(M, 31, True, retain=True)
                    self.assertEqual(f2, {0: 2, 1: 2})
                    bad = [n for n in g2 if not _bits_equal(g2[n], None if first[n] is None else first[n] * 2)]
                    self.assertEqual(bad, [], "retain_graph x2: fused x2 != 2 x fused x1")
                    bad = [n for n in r2 if n not in GC_ORDER and not _bits_equal(r2[n], g2[n])]
                    self.assertEqual(bad, [], "retain_graph x2: knob on != off on the bitwise set")
                    del ref, got, first, r2, g2
                    with torch.no_grad():
                        self.L2._X2LB_REQUESTED = False
                        e0 = self.model(M, 32, train=False)[1]
                        self.L2._X2LB_REQUESTED = True
                        e1 = self.model(M, 32, train=False)[1]
                    self.assertTrue(torch.equal(e0.view(torch.int16), e1.view(torch.int16)), "eval forward")
                    _say("G-c", M=M, result="bitwise outside the fp32-order set, every call fused", finals=uses)
        self.assertEqual([m for m in said if "DECLINED" in m or "FAILED CLOSED" in m or "handshake" in m], [])
        self.L2._X2LB_REQUESTED = None

    # ---------------------------------------------------------------- G-g
    def test_gg_shadow(self):
        self.skipTest("G-g: the DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD_SHADOW knob is not implemented in this patch "
                      "(open item); the in-model split-row scope stays [D]")


# DESIGN_stageC section 6: the fp32-order outputs (X2L-B's partition + PR #7's finalize); every other leaf is bitwise.
GC_ORDER = frozenset(f"{k}.{n}" for k in (0, 1) for n in ("nw0", "nb0", "b1"))


def _bits_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    v = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return bool(torch.equal(a.contiguous().view(v), b.contiguous().view(v)))


def types_ns(nw, nb):
    """The norm0 stand-in _emu_for reads (weight / bias dtypes)."""
    import types

    return types.SimpleNamespace(weight=nw, bias=nb)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

# pyre-unsafe

"""DLRM_HSTU_FP8_DELTA_BWD_V2_DIET (yambda_5b.gin item (48)(b')): the VALU-diet
targets-only delta backward (ops/triton/hstu_delta_bwd_e5m2_v2_diet.py) vs the
v2 kernel (hstu_delta_bwd_e5m2._hstu_attn_delta_bwd_e5m2_v2), bit for bit, on
the image's own Triton build, and the dispatcher's runtime self-check.

GPU (gfx950) only. Every comparison runs the production dispatcher
hstu_attention_delta_bwd_e5m2 (BWD_VKP_V2 on) twice on the same inputs, with
DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=1 and =0, into output buffers pre-filled with a
sentinel, and compares every byte: the E5M2 dV|dK payload incl. the zeroed pad
rows, dQ, the returned amax and the fp32 bias column sum; the kernel-level
tests also compare PARTIAL_COL / PARTIAL_AMAX. Inputs are laid out as in
training (K / V views of one [rows, 1024] e4m3 pack, q a view of a [Z, 1024]
bf16 pack, payload rows padded to 128).

- test_random_small / test_random_big: random production-like values, many
  length mixes (0, 1, tile edges 31/32/33, 63/64/65, long sequences), Z from
  1 to 5000 (both LPT paths: the Triton sort for Z <= 4096, argsort above);
  _big: K, V and the payload in > 2 GiB storages (the in-model build).
- test_edge_cases: NaN codes in K / V (incl. NaN codes of both signs meeting in
  one column), NaN payloads / sNaN / +-inf / -0 in q and dout, products that
  overflow (inf - inf in the partial sums), subnormal q / dout, tiny and huge
  K/V scales, clamp-heavy and tiny inv scales, negative and zero alpha, all
  -0 rows; small and big builds; every byte incl. NaN payload and sign bits.
- test_v2_builds_agree: the reference's own <= 2 GiB and > 2 GiB builds agree
  on every non-NaN bit (the NaN payload / sign count is printed).
- test_kernel_level_partials: the diet kernel with the self-check's profile vs
  v2 on PARTIAL_COL / PARTIAL_AMAX directly.
- test_self_check_arms_once_and_caches / test_self_check_declines_wrong_profile
  / test_kill_switch: the dispatcher arms after one self-check per
  specialization (the verdict is cached), declines to v2 with a "declined"
  line when no profile matches (forced: only the other build's profile), and
  never runs the diet with the switch at 0.
Arming is REQUIRED on the image this kernel is ported to (Triton 3.9.0, the
ROCm 10.2 NATIVE image): any "declined" fails there. On another Triton the
self-check may decline a build and the tests then only require the dispatcher's
outputs to equal v2's: on Triton 3.6.0 + ROCm 10.2 (compat image) the
<= 2 GiB build arms (profile chain) and the > 2 GiB build declines on NaN
payload / sign alone (cross-thread NaN meetings; v2's own <= 2 GiB and > 2 GiB
builds disagree there as well, test_v2_builds_agree prints it).
Run: python -m pytest -q generative_recommenders/ops/tests/hstu_delta_bwd_e5m2_v2_diet_test.py
"""

import contextlib
import io
import os
import unittest

import torch

try:
    import triton as _triton

    _ARM_REQUIRED = _triton.__version__ == "3.9.0"
except Exception:  # noqa: BLE001
    _ARM_REQUIRED = False
_GPU = torch.cuda.is_available() and "gfx950" in str(
    getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")) if torch.cuda.is_available() else False

H, D = 4, 128
NCOLS = 2 * H * D
PAD = 128
LIMIT = 57344.0


def _mod():
    from generative_recommenders.ops.triton import hstu_delta_bwd_e5m2 as M

    return M


@contextlib.contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: v for k, v in kw.items() if v is not None})
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Arena:
    """> 2 GiB uint8 storage; carve() returns views of it (the > 2 GiB storage class)."""

    def __init__(self, dev):
        self.buf = torch.empty(2**31 + 2**28, dtype=torch.uint8, device=dev)
        self.pos = 0

    def carve(self, nbytes):
        start = (self.pos + 255) // 256 * 256
        self.pos = start + nbytes
        assert self.pos <= self.buf.numel()
        return self.buf[start:start + nbytes]


def _case(lens, seed, dev, big=None, kind="rand", **over):
    """Inputs of one call. big: an _Arena for K / V / payload (> 2 GiB storages)."""
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    Z = len(lens)
    so = torch.zeros(Z + 1, dtype=torch.int64)
    so[1:] = torch.tensor(lens, dtype=torch.int64).cumsum(0)
    rows = int(so[-1])
    so = so.to(dev)
    code = torch.randint(0, 0x7F, (rows, NCOLS), generator=g, device=dev, dtype=torch.int32)
    sign = torch.randint(0, 2, (rows, NCOLS), generator=g, device=dev, dtype=torch.int32) * 0x80
    vk_u8 = (code | sign).to(torch.uint8)
    q = torch.randn((Z, H, D), generator=g, device=dev) * over.get("q_std", 0.5)
    dout = torch.randn((Z, H, D), generator=g, device=dev) * over.get("d_std", 0.25)
    q16 = q.to(torch.bfloat16).view(torch.int16)
    d16 = dout.to(torch.bfloat16).view(torch.int16)

    def bits(x):
        return x - 0x10000 if x >= 0x8000 else x

    if kind == "nan" and rows:
        r = torch.randint(0, rows, (max(rows // 50, 4),), generator=g, device=dev)
        c = torch.randint(0, NCOLS, r.shape, generator=g, device=dev)
        vk_u8[r, c] = torch.where(torch.rand(r.shape, generator=g, device=dev) < 0.5, 0x7F, 0xFF).to(torch.uint8)
        # NaN codes of both signs meeting in one column of one sequence (V and K halves)
        for z in range(Z):
            a, n = int(so[z]), lens[z]
            if n >= 2:
                vk_u8[a, 17] = 0x7F
                vk_u8[a + n - 1, 17] = 0xFF
                vk_u8[a + n // 2, 600] = 0x80
        m = torch.rand((Z, H, D), generator=g, device=dev)
        pats = torch.tensor([bits(x) for x in (0x7FC1, 0xFFC3, 0x7F81, 0xFF85, 0x7F80, 0xFF80, 0x8000)],
                            device=dev, dtype=torch.int16)
        pick = torch.randint(0, len(pats), (Z, H, D), generator=g, device=dev)
        q16 = torch.where(m < 0.01, pats[pick], q16)
        m2 = torch.rand((Z, H, D), generator=g, device=dev)
        pick2 = torch.randint(0, len(pats), (Z, H, D), generator=g, device=dev)
        d16 = torch.where(m2 < 0.01, pats[pick2], d16)
    elif kind == "big":
        neg = torch.rand((Z, H, D), generator=g, device=dev) < 0.5
        q16 = torch.where(neg, bits(0xFF7F), 0x7F7F).to(torch.int16)
        neg2 = torch.rand((Z, H, D), generator=g, device=dev) < 0.5
        d16 = torch.where(neg2, bits(0xFF00), 0x7F00).to(torch.int16)
    elif kind == "tiny":
        for t in (q16, d16):
            mant = torch.randint(0, 0x80, (Z, H, D), generator=g, device=dev, dtype=torch.int32)
            sg = torch.randint(0, 2, (Z, H, D), generator=g, device=dev, dtype=torch.int32)
            t.copy_(torch.where(sg > 0, mant - 0x8000, mant).to(torch.int16))
    elif kind == "negzero":
        vk_u8.fill_(0x80)
        q16.fill_(bits(0x8000))
    mp = (rows + PAD - 1) // PAD * PAD if rows else PAD
    if big is not None:
        vk = big.carve(max(rows, 1) * NCOLS).view(torch.float8_e4m3fn)[: rows * NCOLS].view(rows, NCOLS)
        vk.view(torch.uint8).copy_(vk_u8)
        dvk_raw = big.carve(mp * NCOLS)
    else:
        vk = vk_u8.view(torch.float8_e4m3fn)
        dvk_raw = torch.empty(mp * NCOLS, dtype=torch.uint8, device=dev)
    v = vk[:, : H * D].view(rows, H, D)
    k = vk[:, H * D:].view(rows, H, D)
    uq = torch.zeros((Z, NCOLS), dtype=torch.bfloat16, device=dev)
    uq[:, H * D:].copy_(q16.view(torch.bfloat16).view(Z, H * D))
    qv = uq[:, H * D:].view(Z, H, D)
    dv = d16.view(torch.bfloat16).contiguous()
    return dict(q=qv, k=k, v=v, dout=dv, seq_offsets=so, rows=rows, dvk_raw=dvk_raw, mp=mp,
                N=over.get("N", 4096), alpha=over.get("alpha", 0.088388), k_scale=over.get("k_scale", 2.0**-6),
                v_scale=over.get("v_scale", 2.0**-6), inv=over.get("inv", 2.0**12))


def _run(c, diet):
    """Dispatcher call into sentinel-filled outputs -> bytes of every output."""
    M = _mod()
    dev = c["q"].device
    dq = torch.empty_like(c["q"])
    dq.view(torch.int16).fill_(-1)
    c["dvk_raw"].fill_(0xA5)
    dvk = c["dvk_raw"].view(torch.float8_e5m2).view(c["mp"], NCOLS)
    inv = torch.full((1,), c["inv"], dtype=torch.float32, device=dev)
    with _env(DLRM_HSTU_FP8_DELTA_BWD_V2_DIET="1" if diet else "0"):
        amax, colsum = M.hstu_attention_delta_bwd_e5m2(
            dout=c["dout"], q=c["q"], k=c["k"], v=c["v"], dq=dq, dvk_fp8=dvk, rows=c["rows"],
            seq_offsets=c["seq_offsets"], N=c["N"], alpha=c["alpha"], inv_scale=inv, limit=LIMIT,
            k_scale=c["k_scale"], v_scale=c["v_scale"])
    torch.cuda.synchronize()
    return dict(dvk=dvk.view(torch.uint8).clone(), dq=dq.clone(), amax=amax.reshape(1).clone(),
                colsum=colsum.clone())


def _diff(a, b):
    """-> ({key: differing elements}, {key: NaN-payload/sign-only elements}). Exact bits, except that an element of a
    cross-row fp sum (dq, colsum, pc) that is NaN in both arms counts as NaN-payload-only (see
    hstu_delta_bwd_e5m2._v2_diet_compare); dvk bytes, amax and pa are compared exactly."""
    d, nan_only = {}, {}
    for k in a:
        x, y = a[k], b[k]
        if k in ("dq", "colsum", "pc"):
            it = torch.int16 if x.element_size() == 2 else torch.int32
            ne = x.view(it) != y.view(it)
            both = torch.isnan(x) & torch.isnan(y)
            d[k] = int((ne & ~both).sum().item())
            nan_only[k] = int((ne & both).sum().item())
        else:
            it = {1: torch.uint8, 4: torch.int32}[x.element_size()]
            d[k] = int((x.view(it) != y.view(it)).sum().item())
    return d, nan_only


@unittest.skipUnless(_GPU, "needs a gfx950 GPU")
class DeltaBwdDietBitwiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        M = _mod()
        cls._vkp = M._VKP_V2
        M._VKP_V2 = True  # the (25)(b) v2 branch (gin binds it on)
        cls.dev = torch.device("cuda")

    @classmethod
    def tearDownClass(cls):
        _mod()._VKP_V2 = cls._vkp

    def setUp(self):
        M = _mod()
        M._V2_DIET_VERDICT.clear()
        M._V2_DIET_LOGGED[0] = False
        M._V2_DIET_DECLINE_LOGGED.clear()

    def _check(self, c, label):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            a = _run(c, diet=True)
            b = _run(c, diet=False)
        log = out.getvalue()
        if _ARM_REQUIRED:
            self.assertNotIn("declined", log, f"{label}: {log}")
        d, nan_only = _diff(a, b)
        self.assertEqual(sum(d.values()) + sum(nan_only.values()), 0,
                         f"{label}: differing {d}, NaN payload/sign {nan_only}")
        return log

    def _lens_sets(self):
        g = torch.Generator()
        g.manual_seed(7)
        sets = [
            [0], [1], [31], [32], [33], [4097],
            [0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000],
            torch.randint(1, 600, (300,), generator=g).tolist(),
            torch.randint(0, 3000, (64,), generator=g).tolist(),
            (torch.randint(0, 40, (5000,), generator=g)).tolist(),  # Z > 4096: argsort LPT path
        ]
        return sets

    def test_random_small(self):
        for i, lens in enumerate(self._lens_sets()):
            self._check(_case(lens, 100 + i, self.dev), f"small set {i} Z={len(lens)}")

    def test_random_big(self):
        arena = _Arena(self.dev)
        for i, lens in enumerate(self._lens_sets()[6:9]):
            arena.pos = 0
            self._check(_case(lens, 200 + i, self.dev, big=arena), f"big set {i} Z={len(lens)}")
        del arena

    def test_edge_cases(self):
        g = torch.Generator()
        g.manual_seed(11)
        lens = [0, 1, 2, 31, 32, 33, 64, 65, 97, 200] + torch.randint(1, 300, (40,), generator=g).tolist()
        arena = _Arena(self.dev)
        variants = [
            ("nan", {}), ("big", {}), ("tiny", {}), ("negzero", {}),
            ("rand", {"inv": 2.0**30}), ("rand", {"inv": 2.0**-20}), ("rand", {"k_scale": 2.0**-120}),
            ("rand", {"v_scale": 2.0**100}), ("rand", {"alpha": -0.5}), ("rand", {"alpha": 0.0}),
            ("nan", {"inv": 2.0**30, "alpha": -0.3}), ("rand", {"q_std": 3e3, "d_std": 1e4, "N": 7}),
        ]
        for big in (None, arena):
            for i, (kind, over) in enumerate(variants):
                arena.pos = 0
                self._check(_case(lens, 300 + i, self.dev, big=big, kind=kind, **over),
                            f"edge {kind} {over} big={big is not None}")
        del arena

    def test_kernel_level_partials(self):
        M = _mod()
        from generative_recommenders.ops.triton import hstu_delta_bwd_e5m2_v2_diet as DD

        c = _case([0, 1, 31, 32, 33, 333, 1000, 77], 400, self.dev, kind="nan")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _run(c, diet=True)
        # the call's key and its > 2 GiB siblings (_v2_diet_presibling): one profile
        small = [v for kk, v in M._V2_DIET_VERDICT.items() if all(p[0] for p in kk[2])]
        self.assertEqual(len(small), 1, M._V2_DIET_VERDICT)
        profile = small[0][0]
        if profile is None and not _ARM_REQUIRED:
            self.skipTest(f"diet declined on this Triton: {small[0][2][:200]}")
        self.assertIsNotNone(profile, M._V2_DIET_VERDICT)
        if _ARM_REQUIRED:
            self.assertEqual({v[0] for v in M._V2_DIET_VERDICT.values()}, {profile}, M._V2_DIET_VERDICT)
        Z = c["seq_offsets"].numel() - 1
        perm = torch.argsort(c["seq_offsets"][1:] - c["seq_offsets"][:-1], descending=True, stable=True).to(
            torch.int32)
        inv = torch.full((1,), c["inv"], dtype=torch.float32, device=self.dev)
        res = {}
        for which in ("v2", "diet"):
            dq = torch.empty_like(c["q"])
            dvk = c["dvk_raw"].view(torch.float8_e5m2).view(c["mp"], NCOLS)
            c["dvk_raw"].fill_(0x5A)
            pc = torch.full((Z, NCOLS), float("nan"), device=self.dev)
            pa = torch.full((Z * H,), float("nan"), device=self.dev)
            if which == "diet":
                DD.launch_kernel(c["dout"], c["q"], c["k"], c["v"], dq, dvk, c["seq_offsets"], c["N"], c["alpha"],
                                 inv, LIMIT, c["k_scale"], c["v_scale"], pc, pa, perm, True, profile=profile)
            else:
                M._hstu_attn_delta_bwd_e5m2_v2[(Z * H,)](
                    Q=c["q"], K=c["k"], V=c["v"], seq_offsets=c["seq_offsets"], DOut=c["dout"], DQ=dq, DVK8=dvk,
                    PARTIAL_COL=pc, PARTIAL_AMAX=pa, INV_SCALE=inv, stride_qm=c["q"].stride(0),
                    stride_qh=c["q"].stride(1), stride_kn=c["k"].stride(0), stride_kh=c["k"].stride(1),
                    stride_vn=c["v"].stride(0), stride_vh=c["v"].stride(1), stride_dom=c["dout"].stride(0),
                    stride_doh=c["dout"].stride(1), stride_dqm=dq.stride(0), stride_dqh=dq.stride(1),
                    stride_dvk=dvk.stride(0), alpha=c["alpha"], K_SCALE=c["k_scale"], V_SCALE=c["v_scale"], H=H,
                    MAX_SEQ_LEN=c["N"], COL_K0=H * D, BLOCK_D_Q=D, BLOCK_D_V=D, BLOCK_N=32, LIMIT=LIMIT,
                    REDUCE_EACH=False, PERM=perm, USE_PERM=True, NUM_STAGES=M._VKP_V2_NUM_STAGES,
                    AMAX_ROWMAX=M._VKP_V2_AMAX_ROWMAX, DQ_ELEM=M._VKP_V2_DQ_ELEM, num_warps=4)
            torch.cuda.synchronize()
            res[which] = dict(dvk=dvk.view(torch.uint8).clone(), dq=dq.clone(), pc=pc.clone(), pa=pa.clone())
        d, nan_only = _diff(res["v2"], res["diet"])
        self.assertEqual(sum(d.values()) + sum(nan_only.values()), 0, (d, nan_only))

    def test_self_check_arms_once_and_caches(self):
        M = _mod()
        c = _case([5, 64, 300, 0, 33], 500, self.dev)
        calls = []
        orig = M._v2_diet_probe_run

        def counting(*a, **k):
            calls.append(a[0])
            return orig(*a, **k)

        M._v2_diet_probe_run = counting
        arena = _Arena(self.dev)
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                _run(c, diet=True)  # <= 2 GiB key: its check, then its two > 2 GiB siblings' (_v2_diet_presibling)
                _run(c, diet=True)
                c2 = _case([7, 99, 1], 501, self.dev)  # same specialization: no new check
                _run(c2, diet=True)
                c3 = _case([7, 99, 1, 40], 502, self.dev, big=arena)  # sibling (a): verdict and binaries ready
                b3 = _run(c3, diet=True)
                a3 = _run(c3, diet=False)
                n_calls_a = len(calls)
                c4 = _case([7, 99, 1, 40], 503, self.dev)  # sibling (b): K/V <= 2 GiB, payload > 2 GiB
                c4["dvk_raw"] = arena.carve(c4["mp"] * NCOLS)
                b4 = _run(c4, diet=True)
                a4 = _run(c4, diet=False)
        finally:
            M._v2_diet_probe_run = orig
            del arena
        log = out.getvalue()
        self.assertEqual(len(M._V2_DIET_VERDICT), 3, list(M._V2_DIET_VERDICT))
        for a_, b_ in ((a3, b3), (a4, b4)):
            d, nan_only = _diff(a_, b_)
            self.assertEqual(sum(d.values()) + sum(nan_only.values()), 0, (d, nan_only))
        if not _ARM_REQUIRED and any(v[0] is None for v in M._V2_DIET_VERDICT.values()):
            self.skipTest("diet declined a build on this Triton: " + "; ".join(
                str(v[2])[:160] for v in M._V2_DIET_VERDICT.values() if v[0] is None))
        self.assertEqual(len(calls), 3, calls)
        self.assertEqual(n_calls_a, len(calls), calls)  # the payload-only sibling ran no check of its own
        self.assertEqual(log.count("DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=1 armed"), 1, log)
        self.assertIn("runtime self-check", log)
        self.assertIn("differing DVK8 bytes 0, DQ 0, PARTIAL_COL 0, PARTIAL_AMAX 0, NaN payload/sign DQ 0, "
                      "PARTIAL_COL 0", log)
        self.assertNotIn("declined", log)

    def test_self_check_declines_wrong_profile(self):
        M = _mod()
        from generative_recommenders.ops.triton import hstu_delta_bwd_e5m2_v2_diet as DD
        import triton

        right = DD.profile_candidates(triton.__version__)[0]
        wrong = DD.RED_CHAIN if right != DD.RED_CHAIN else DD.RED_TREE
        orig = DD.profile_candidates
        DD.profile_candidates = lambda version: (wrong,)
        try:
            c = _case([0, 1, 31, 32, 33, 200, 64], 600, self.dev)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                a = _run(c, diet=True)
                b = _run(c, diet=False)
        finally:
            DD.profile_candidates = orig
        log = out.getvalue()
        self.assertIn("DLRM_HSTU_FP8_DELTA_BWD_V2_DIET=1 declined (self-check vs v2", log)
        self.assertNotIn("armed: VALU-diet", log)
        d, nan_only = _diff(a, b)
        self.assertEqual(sum(d.values()) + sum(nan_only.values()), 0, (d, nan_only))

    def test_v2_builds_agree(self):
        # The reference itself: v2's <= 2 GiB and > 2 GiB builds on the same NaN-heavy inputs. Every non-NaN bit
        # must agree; the NaN-payload/sign-only count is printed (informational: whether v2's own cross-row NaN
        # payloads depend on the build).
        lens = [0, 1, 2, 31, 32, 33, 64, 65, 97, 200] + [40] * 10
        arena = _Arena(self.dev)
        out = {}
        for tag, big in (("small", None), ("big", arena)):
            arena.pos = 0
            c = _case(lens, 300, self.dev, big=big, kind="nan")
            out[tag] = _run(c, diet=False)
        d, nan_only = _diff(out["small"], out["big"])
        print(f"[v2 small vs big build] differing {d} NaN payload/sign {nan_only}", flush=True)
        self.assertEqual(sum(d.values()), 0, (d, nan_only))
        del arena

    def test_kill_switch(self):
        M = _mod()
        c = _case([3, 40, 100], 700, self.dev)
        calls = []
        orig = M._v2_diet_probe_run
        M._v2_diet_probe_run = lambda *a, **k: calls.append(1) or orig(*a, **k)
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                _run(c, diet=False)
        finally:
            M._v2_diet_probe_run = orig
        self.assertEqual(calls, [])
        self.assertNotIn("DELTA_BWD_V2_DIET", out.getvalue())


if __name__ == "__main__":
    unittest.main()

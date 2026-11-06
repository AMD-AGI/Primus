"""CPU tests for (119) ACTION_DEDUP (ops/action_dedup.py, DLRM_HSTU_FP8_ACTION_DEDUP).

Knob on vs off on a ContextualPreprocessor at the production action dims (T = 3
action types x 8 -> 24 wide, hidden 256, D = 512): forward BITWISE (fp32,
fp64 module, bf16 autocast), every gradient within the stated tolerance and no
worse than the per-token path against an fp64 reference, d(residual) and every
non-action gradient bitwise, the (65) prefill carrier bitwise vs the inline
plan, all / one / missing codes, empty UIH, no cross-step caching, determinism,
the per-code sum (fp32 chunks and the bmm chunking logic) bitwise vs fp64 on
dyadic data, the hi/lo split bound, declines, the gin binding (exactly once,
code default off, env wins) and the ARMED / OFF lines.

Run: python -m pytest -q generative_recommenders/ops/tests/action_dedup_cpu_test.py
"""

import os
import re
import unittest
from typing import Dict, List, Optional

import fbgemm_gpu  # noqa: F401  registers torch.ops.fbgemm
import torch

from generative_recommenders.common import HammerKernel
from generative_recommenders.modules import action_prefill
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from generative_recommenders.ops import action_dedup

ENV = action_dedup.ENV_VAR
ADDITIONAL = ["artist_id", "album_id"]
ACTION = "action_weight"
N_CTX = 0  # contextual rows are outside the action branch (and fbgemm CPU has no fp64 j2pd)
GIN = os.path.join(
    os.path.dirname(__file__), "..", "..", "dlrm_v4", "train", "gin", "yambda_5b.gin"
)


def _model(
    seed: int, dtype: torch.dtype = torch.float32, weights: Optional[List[int]] = None,
    f1: bool = False, f2: bool = False,
) -> ContextualPreprocessor:
    torch.manual_seed(seed)
    ctx = [f"ctx{i}" for i in range(N_CTX)]
    m = ContextualPreprocessor(
        input_embedding_dim=512,
        hidden_dim=256,
        output_embedding_dim=512,
        contextual_feature_to_max_length={k: 1 for k in ctx},
        contextual_feature_to_min_uih_length={k: 0 for k in ctx},
        action_embedding_dim=8,
        action_feature_name=ACTION,
        action_weights=[1, 2, 4] if weights is None else weights,
        additional_embedding_features=ADDITIONAL,
        action_embedding_init_std=0.1,
        is_inference=False,
        fuse_linear_norm=f1,
        fuse_ln_residual=f2,
    ).to(dtype)
    m.train()
    m.set_hammer_kernel(HammerKernel.PYTORCH)
    return m


def _inputs(
    seed: int, B: int = 48, max_uih: int = 300, dtype: torch.dtype = torch.float32,
    action_hi: int = 8, action_set: Optional[List[int]] = None, uih_zero: bool = False,
) -> Dict[str, object]:
    g = torch.Generator().manual_seed(seed)
    uih = torch.randint(1, max_uih + 1, (B,), generator=g)
    if uih_zero:
        uih = torch.zeros(B, dtype=torch.int64)
    nt = torch.ones(B, dtype=torch.int64)
    T_uih = int(uih.sum())
    M = T_uih + B
    feats = {"seq_embeddings": torch.randn(M, 512, generator=g).to(dtype)}
    for n in ADDITIONAL:
        feats[n] = torch.randn(M, 512, generator=g).to(dtype)
    for i in range(N_CTX):
        feats[f"ctx{i}"] = torch.randn(B, 512, generator=g).to(dtype)
    if action_set is not None:
        idx = torch.randint(0, len(action_set), (T_uih,), generator=g)
        acts = torch.tensor(action_set, dtype=torch.int64)[idx]
    else:
        acts = torch.randint(0, action_hi, (T_uih,), generator=g)
    ints = {ACTION: acts}
    for i in range(N_CTX):
        ints[f"ctx{i}_offsets"] = torch.arange(B + 1, dtype=torch.int64)
    return {
        "uih": uih, "nt": nt, "max_uih": max(int(uih.max()), 0), "B": B,
        "T_uih": T_uih, "ts": torch.randint(0, 1 << 30, (M,), generator=g),
        "feats": feats, "ints": ints,
    }


def _run(
    m: ContextualPreprocessor, inp: Dict[str, object], dedup: Optional[str],
    autocast: Optional[torch.dtype] = None, prefill: bool = False, backward: bool = True,
    loss_seed: int = 7,
) -> Dict[str, torch.Tensor]:
    if dedup is None:
        os.environ.pop(ENV, None)
    else:
        os.environ[ENV] = dedup
    os.environ[action_prefill.ENV] = "1" if prefill else "0"
    for p in m.parameters():
        p.grad = None
    leaves = {k: v.detach().clone().requires_grad_(True) for k, v in inp["feats"].items()}
    payloads: Dict[str, torch.Tensor] = dict(inp["ints"])
    for k, v in leaves.items():
        if k != "seq_embeddings":
            payloads[k] = v
    uih, nt = inp["uih"], inp["nt"]
    with torch.autocast("cpu", dtype=autocast or torch.bfloat16, enabled=autocast is not None):
        if prefill:
            m.prefill_action(
                max_uih_len=inp["max_uih"], max_targets=1, total_targets=inp["B"],
                uih_seq_lengths=uih, num_targets=nt, seq_payloads=payloads,
            )
        out = m(
            max_uih_len=inp["max_uih"], max_targets=1, total_uih_len=inp["T_uih"],
            total_targets=inp["B"], seq_lengths=uih + nt, seq_timestamps=inp["ts"],
            seq_embeddings=leaves["seq_embeddings"], num_targets=nt, seq_payloads=payloads,
        )
    assert action_prefill._pending is None
    res = {"out": out[6].detach()}
    if backward:
        g = torch.Generator().manual_seed(loss_seed)
        w = torch.randn(tuple(out[6].shape), generator=g).to(torch.float64)
        (out[6].to(torch.float64) * w).sum().backward()
        for n, p in m.named_parameters():
            res[f"param:{n}"] = p.grad.detach().clone()
        for k, v in leaves.items():
            res[f"input:{k}"] = v.grad.detach().clone()
    os.environ.pop(ENV, None)
    os.environ.pop(action_prefill.ENV, None)
    return res


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32,
                   torch.float64: torch.int64}[t.dtype])


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    d = (a.double() - ref.double()).norm()
    n = ref.double().norm()
    return float(d / n) if n > 0 else float(d)


ACTION_KEYS = ("_action_embedding_mlp", "_action_encoder")
# Forward tolerance when the 2 * 9-row MLP launch does not reproduce the
# per-token rows' bits (CPU BLAS is M-dependent on ~1e-5..1e-4 of elements):
# |out_on - out_off| <= 2 ulp of max(|out|, max|table|) (fp32: 2^-22 relative,
# bf16: 2^-7), and rel-L2 <= 1e-6 (fp32) / 4e-3 (bf16).
_FWD_TOL = {torch.float32: (2.0 ** -22, 1e-6), torch.bfloat16: (2.0 ** -7, 4e-3),
            torch.float64: (2.0 ** -51, 1e-14)}
FWD_REPORT: List[str] = []


def _fwd_close(tc: unittest.TestCase, m: ContextualPreprocessor, off: torch.Tensor,
               on: torch.Tensor, tag: str) -> None:
    tc.assertEqual((off.dtype, off.shape), (on.dtype, on.shape))
    rel, l2 = _FWD_TOL[off.dtype]
    with torch.no_grad():
        e = action_dedup.table_inputs(m._action_encoder)
        ymax = float(m._action_embedding_mlp(e.to(next(m.parameters()).dtype)).abs().max())
    d = (on.double() - off.double()).abs()
    bound = rel * torch.maximum(off.double().abs(), torch.full_like(d, ymax))
    frac = float((d > 0).double().mean()) if d.numel() else 0.0
    FWD_REPORT.append(f"{tag}: rows={off.shape[0]} bitwise={frac == 0.0} diff_frac={frac:.3g} "
                      f"max_abs={float(d.max()) if d.numel() else 0.0:.3g}")
    tc.assertTrue(bool((d <= bound).all()), FWD_REPORT[-1])
    tc.assertLessEqual(_rel(on, off), l2, FWD_REPORT[-1])


class ActionDedupCpuTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop(ENV, None)
        action_dedup._latched = None

    def test_code_default_off(self) -> None:
        os.environ.pop(ENV, None)
        self.assertFalse(action_dedup.enabled())
        for v in ("0", "false", "off", ""):
            os.environ[ENV] = v
            self.assertFalse(action_dedup.enabled())
        os.environ[ENV] = "1"
        self.assertTrue(action_dedup.enabled())
        os.environ.pop(ENV)

    def test_codes_and_table_rows_match_encoder(self) -> None:
        m = _model(1)
        inp = _inputs(2, action_hi=1 << 12)  # high bits beyond the 3 weights too
        uih, nt = inp["uih"], inp["nt"]
        M = inp["T_uih"] + inp["B"]
        enc = m._action_encoder
        code = action_dedup.codes(enc, inp["ints"], uih + nt, nt, M)
        # Independent reconstruction: per sample, UIH codes then the target code.
        a = inp["ints"][ACTION]
        ref = []
        off = 0
        for b in range(inp["B"]):
            n = int(uih[b])
            for x in a[off:off + n].tolist():
                ref.append((x & 1 != 0) | ((x & 2 != 0) << 1) | ((x & 4 != 0) << 2))
            ref.append(8)
            off += n
        self.assertTrue(torch.equal(code, torch.tensor(ref)))
        # Table rows == the per-token encoder rows, bitwise.
        tso = torch.ops.fbgemm.asynchronous_complete_cumsum(nt)
        so = torch.ops.fbgemm.asynchronous_complete_cumsum(uih + nt)
        rows = enc(
            max_uih_len=inp["max_uih"], max_targets=1, uih_offsets=so - tso,
            target_offsets=tso, seq_embeddings=torch.empty(M, 0),
            seq_payloads=inp["ints"],
        )
        e = action_dedup.table_inputs(enc)
        self.assertTrue(_same(e[code], rows))

    def _fwd_bwd_equal(self, dtype: torch.dtype, autocast: Optional[torch.dtype],
                       f1: bool = False, f2: bool = False, **kw: object) -> None:
        m = _model(3, dtype, f1=f1, f2=f2)
        inp = _inputs(4, dtype=dtype, **kw)
        off = _run(m, inp, None, autocast)
        on = _run(m, inp, "1", autocast)
        _fwd_close(self, m, off["out"], on["out"], f"{dtype} autocast={autocast} {kw}")
        for k in off:
            if k == "out":
                continue
            if any(s in k for s in ACTION_KEYS):
                # d(table) is the per-code sum, reduction order differs.
                tol = 1e-12 if dtype == torch.float64 else (1e-5 if autocast is None else 2e-2)
                self.assertLessEqual(_rel(on[k], off[k]), tol, k)
            else:
                self.assertTrue(_same(off[k], on[k]), f"{k} not bitwise")

    def test_fp32_forward_bitwise_backward_tol(self) -> None:
        self._fwd_bwd_equal(torch.float32, None)

    def test_fp32_f1f2_containers(self) -> None:
        self._fwd_bwd_equal(torch.float32, None, f1=True, f2=True)

    def test_bf16_autocast(self) -> None:
        self._fwd_bwd_equal(torch.float32, torch.bfloat16)

    def test_one_code_missing_codes_empty_uih(self) -> None:
        self._fwd_bwd_equal(torch.float32, None, action_set=[5])
        self._fwd_bwd_equal(torch.float32, None, action_set=[0, 3])
        self._fwd_bwd_equal(torch.float32, None, uih_zero=True)

    def test_bf16_grads_vs_fp64_reference(self) -> None:
        """Both paths under bf16 autocast vs the per-token fp64 module:
        dedup error <= max(2 x per-token error, 1e-5) per action tensor."""
        m = _model(5)
        inp = _inputs(6)
        off = _run(m, inp, None, torch.bfloat16)
        on = _run(m, inp, "1", torch.bfloat16)
        m64 = _model(5, torch.float64)
        m64.load_state_dict({k: v.double() if v.is_floating_point() else v
                             for k, v in m.state_dict().items()})
        inp64 = dict(inp)
        inp64["feats"] = {k: v.double() for k, v in inp["feats"].items()}
        # fp64 reference: the fp64 MLP on the code rows with fp64 per-code
        # sums (the per-token fp64 chain cannot run on CPU: fbgemm's
        # jagged_to_padded_dense has no Double kernel). Exact in fp64.
        ref = _run(m64, inp64, "1", None)
        for k in off:
            if k == "out" or not any(s in k for s in ACTION_KEYS):
                continue
            e_old, e_new = _rel(off[k], ref[k]), _rel(on[k], ref[k])
            self.assertLessEqual(e_new, max(2 * e_old, 1e-5), f"{k} new {e_new} old {e_old}")
            self.assertLessEqual(e_new, 1e-2, k)

    def test_prefill_carrier_bitwise_vs_inline(self) -> None:
        m = _model(7)
        inp = _inputs(8)
        used0 = action_prefill.stats["used"]
        inline = _run(m, inp, "1", None, prefill=False)
        pre = _run(m, inp, "1", None, prefill=True)
        self.assertEqual(action_prefill.stats["used"] - used0, 1)
        for k in inline:
            self.assertTrue(_same(inline[k], pre[k]), k)

    def test_deterministic_and_no_cross_step_cache(self) -> None:
        m = _model(9)
        inp = _inputs(10)
        a = _run(m, inp, "1", None)
        b = _run(m, inp, "1", None)
        for k in a:
            self.assertTrue(_same(a[k], b[k]), k)
        with torch.no_grad():  # an optimizer step on the action branch
            for n, p in m.named_parameters():
                if any(s in n for s in ACTION_KEYS):
                    p.add_(0.01 * torch.sign(p) + 0.003)
        off = _run(m, inp, None, None, backward=False)
        on = _run(m, inp, "1", None, backward=False)
        _fwd_close(self, m, off["out"], on["out"], "after update")
        self.assertGreater(_rel(a["out"], on["out"]), 1e-4)  # recomputed, not cached

    def test_eval_no_grad(self) -> None:
        m = _model(11)
        m.eval()
        inp = _inputs(12)
        with torch.no_grad():
            off = _run(m, inp, None, None, backward=False)
            on = _run(m, inp, "1", None, backward=False)
        _fwd_close(self, m, off["out"], on["out"], "eval no_grad")

    def test_kill_switch_and_decline(self) -> None:
        m = _model(13)
        inp = _inputs(14)
        n0 = action_dedup.stats["forwards"]
        _run(m, inp, "0", None, backward=False)
        self.assertEqual(action_dedup.stats["forwards"], n0)
        # T = 7 > MAX_T: declined, shipped path, one DECLINED line.
        m7 = _model(15, weights=[1, 2, 4, 8, 16, 32, 64])
        d0 = action_dedup.stats["declined"]
        off = _run(m7, inp, None, None, backward=False)
        on = _run(m7, inp, "1", None, backward=False)
        self.assertEqual(action_dedup.stats["declined"], d0 + 1)
        self.assertTrue(_same(off["out"], on["out"]))

    def _dyadic(self, rows: int, k: int):
        i = torch.arange(rows)
        code = (i * 5 + 3) % k
        q = ((i.unsqueeze(1) * 7 + torch.arange(512).unsqueeze(0) * 3) % 17 - 8).float()
        g = (q * 2.0 ** -6).to(torch.bfloat16)
        ref = torch.zeros(k, 512, dtype=torch.float64).index_add_(0, code, g.double())
        return code, g, ref

    def test_segsum_f32_and_bmm_chunking(self) -> None:
        for rows in (0, 7, action_dedup.CHUNK, 3 * action_dedup.CHUNK + 5,
                     action_dedup.F32_CHUNK + 9):
            code, g, ref = self._dyadic(rows, 9)
            s = action_dedup.segsum(code, g, 9, "f32")
            self.assertEqual(s.dtype, torch.float32)
            self.assertTrue(torch.equal(s.double(), ref), rows)
        # The bmm path's chunking / remainder logic, with an fp32-upcast stand-in
        # for bmm(out_dtype=float32) (not implemented on CPU).
        orig = torch.bmm

        def fake_bmm(a, b, out_dtype=None):
            assert out_dtype == torch.float32
            return orig(a.float(), b.float())

        torch.bmm = fake_bmm
        try:
            for rows in (7, action_dedup.CHUNK, 3 * action_dedup.CHUNK + 5):
                code, g, ref = self._dyadic(rows, 9)
                s = action_dedup.segsum(code, g, 9, "bmm")
                self.assertTrue(torch.equal(s.double(), ref), rows)
        finally:
            torch.bmm = orig

    def test_hi_lo_bound(self) -> None:
        g = torch.Generator().manual_seed(3)
        s = torch.randn(9, 512, generator=g) * 37.0
        hi = s.to(torch.bfloat16)
        lo = (s - hi.float()).to(torch.bfloat16)
        err = (s - (hi.float() + lo.float())).abs()
        self.assertTrue(bool((err <= 2.0 ** -17 * s.abs()).all()))

    def test_gather_add_semantics(self) -> None:
        g = torch.Generator().manual_seed(4)
        atab2 = torch.randn(18, 512, generator=g).to(torch.bfloat16).requires_grad_(True)
        res = torch.randn(1000, 512, generator=g).to(torch.bfloat16).requires_grad_(True)
        code = torch.randint(0, 9, (1000,), generator=g)
        out = action_dedup.gather_add(res, atab2, code)
        ref = (res.float() + atab2[:9].float()[code]).to(torch.bfloat16)
        self.assertTrue(_same(out, ref))
        dy = torch.randn(1000, 512, generator=g).to(torch.bfloat16)
        out.backward(dy)
        self.assertTrue(_same(res.grad, dy))  # d(residual) is the incoming gradient
        s = torch.zeros(9, 512, dtype=torch.float64).index_add_(0, code, dy.double())
        got = atab2.grad[:9].double() + atab2.grad[9:].double()
        self.assertLessEqual(float(((got - s).abs() - 2.0 ** -16 * s.abs()).max()), 1e-6)
        empty = action_dedup.gather_add(res[:0], atab2, code[:0])
        self.assertEqual(tuple(empty.shape), (0, 512))

    def test_gin_binding_exactly_once_and_env_wins(self) -> None:
        text = open(GIN).read()
        self.assertEqual(
            len(re.findall(r"^apply_env_bootstrap\.DLRM_HSTU_FP8_ACTION_DEDUP = True$",
                           text, re.M)), 1)
        # One (119) header; its sub-knob blocks ("# (119) sub-knob ...") follow it.
        self.assertEqual(len(re.findall(r"^# \(119\) (?!sub-knob )", text, re.M)), 1)
        self.assertNotIn("ACTION_DEDUP", "".join(
            ln for ln in text.splitlines(True) if ln.startswith("make_model.")))
        import inspect

        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        self.assertIn("DLRM_HSTU_FP8_ACTION_DEDUP",
                      inspect.signature(eb.apply_env_bootstrap).parameters)
        saved = dict(os.environ)
        try:
            os.environ[ENV] = "0"
            eb._bind_bool(ENV, True)
            self.assertEqual(os.environ[ENV], "0")
            os.environ.pop(ENV)
            eb._bind_bool(ENV, True)
            self.assertEqual(os.environ[ENV], "1")
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_armed_and_off_lines(self) -> None:
        import contextlib
        import io

        line = action_dedup._armed_line("bmm", "x")
        self.assertRegex(line, r"armed OK: \[action-dedup\] ARMED \(DLRM_HSTU_FP8_ACTION_DEDUP=1\)")
        self.assertIn("recomputed every forward, exact MLP", line)
        self.assertIn("REDUCTION-ORDER item (119)", line)
        from generative_recommenders.dlrm_v4.train import blas_warmup

        buf = io.StringIO()
        saved = dict(os.environ)
        try:
            with contextlib.redirect_stdout(buf):
                os.environ.pop(ENV, None)
                blas_warmup._prebuild_action_dedup(torch.device("cpu"))
                self.assertEqual(buf.getvalue().strip(), "")
                os.environ[ENV] = "0"
                blas_warmup._prebuild_action_dedup(torch.device("cpu"))
        finally:
            os.environ.clear()
            os.environ.update(saved)
        self.assertIn("[action-dedup] OFF (DLRM_HSTU_FP8_ACTION_DEDUP=0)", buf.getvalue())


def tearDownModule() -> None:
    print("\n[action-dedup test] forward: " + " | ".join(FWD_REPORT))


if __name__ == "__main__":
    unittest.main()

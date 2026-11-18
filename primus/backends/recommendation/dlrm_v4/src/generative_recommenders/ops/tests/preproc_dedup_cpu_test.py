# pyre-strict
"""CPU checks of item (118) PREPROC_DEDUP (ops/preproc_dedup.py; DLRM_HSTU_FP8_PREPROC_DEDUP).

    python -B -m pytest -q generative_recommenders/ops/tests/preproc_dedup_cpu_test.py

The module's device gate is widened to "cpu" per test (production engages on ROCm gfx950 only) and the
ContextualPreprocessor runs its PYTORCH norm kernels (FusedLinearNormMLP falls back to nn.Sequential on CPU, so
the F1 / F2 Triton kernels themselves are GPU-only; the arithmetic is the same row-wise chain).

  * construction (build_triples) == a host torch.unique reference on the worst-case duplicate patterns, stable,
    deterministic, and dispatching no host-reading op; overflow (n_u > M_pad) sets the device flag, the next call
    raises OVERFLOW and the counters line shows it;
  * (a1) LOGIC: dedup vs per-token on the SAME bf16 autocast path (forward: mismatch fraction and max ulp), the
    fp64 wiring (bitwise forward, 1e-12 gradients), and every gradient (MLP parameters and the EC table) vs an fp64
    per-token reference within new <= max(2 x old, 1e-3); the hi-only mutant fails the LN-bias criterion; M_pad =
    n_u and n_u + 512 agree;
  * (a2) PRODUCTION DELTA vs the FP8 per-token path: not runnable on CPU (E-M3a / IX-FP8F are GPU kernels);
  * wiring: the real ContextualPreprocessor.forward with the action MLP: no grouped tail is begun, the action
    branch adds acc_tok in its F2 statement, outputs and gradients match the per-token forward;
  * EC: _merged_construct's grouped branch tags exactly the merged item member and `big`;
  * knob off == no import, no line (child processes), the kill switch, read-once, IGNORED / NOT armed, declines,
    DISABLED, the Indexed-X ledger bump, eval;
  * the gin (118) block, the bootstrap knob, make_model, clean_exit, markers (ARMED_RX on the real prebuild line);
  * KS-K1: armed, eager, untagged calls with T > 0 count untagged / eval_untagged on the exit line; T = 0 is not
    counted and the EC's empty-batch decline is silent.

Opt-ins: PPDD_BASE_TREE=<unpatched tree> (knob-off forward / backward == the unpatched tree, child processes);
PPDD_MARKERS_JSON=<markers_118.json> (the file equals the module constants).
"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from generative_recommenders.common import HammerKernel
from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
from generative_recommenders.ops import preproc_dedup as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_GIN = os.path.join(_ROOT, "generative_recommenders", "dlrm_v4", "train", "gin", "yambda_5b.gin")
CPU = torch.device("cpu")
_LIB: List[Any] = []


def _ensure_cumsum() -> None:
    """fbgemm's asynchronous_complete_cumsum on CPU when fbgemm_gpu is absent (this interpreter)."""
    try:
        torch.ops.fbgemm.asynchronous_complete_cumsum
        return
    except (AttributeError, RuntimeError):
        pass
    lib = torch.library.Library("fbgemm", "DEF")
    lib.define("asynchronous_complete_cumsum(Tensor t) -> Tensor")
    lib.impl("asynchronous_complete_cumsum",
             lambda t: torch.cat([t.new_zeros(1), torch.cumsum(t, 0).to(t.dtype)]), "CompositeExplicitAutograd")
    _LIB.append(lib)


def _arm() -> None:
    pd.reset_for_testing()
    pd._DEVICES = ("cuda", "cpu")
    pd._STATE = "armed"
    pd._ARMED_DEVS.add(("cpu", -1))


def _pre(action: bool = False, dtype: torch.dtype = torch.float32) -> Any:
    from generative_recommenders.modules.preprocessors import ContextualPreprocessor

    torch.manual_seed(1234)
    pre = ContextualPreprocessor(
        input_embedding_dim=512, hidden_dim=256, output_embedding_dim=512, contextual_feature_to_max_length={},
        contextual_feature_to_min_uih_length={}, action_embedding_dim=8, action_feature_name="act",
        action_weights=[1, 2, 4] if action else None, additional_embedding_features=["art", "alb"],
        is_inference=False, fuse_linear_norm=True, fuse_ln_residual=True)
    pre.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
    for m in pre.modules():
        m._f1_reported = True  # quiet the CPU INERT line
    with torch.no_grad():  # nonzero LN affine / biases so every gradient is exercised
        for n, p in pre.named_parameters():
            if p.dim() == 1:
                p.copy_(torch.linspace(-0.5, 0.5, p.numel()) + (1.0 if n.endswith("weight") else 0.0))
    return pre.to(dtype)


def _batch(kind: str, T: int, U: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """(item idx [T], interleaved cat idx [2T]) over a U-row table; artist / album are functions of the item unless
    kind == 'nonfunctional'."""
    g = torch.Generator().manual_seed(seed)
    n_items = max(1, U // 2)
    if kind == "heavy":
        item = torch.randint(0, min(n_items, 40), (T,), generator=g)
    elif kind == "alldup":
        item = torch.full((T,), 3, dtype=torch.int64)
    elif kind == "allunique":
        item = torch.randperm(n_items, generator=g)[:T]
    elif kind == "oneheavy":
        item = torch.randint(0, n_items, (T,), generator=g)
        item[: (T * 3) // 4] = 5
        item = item[torch.randperm(T, generator=g)]
    elif kind == "uniform":
        item = torch.randint(0, n_items, (T,), generator=g)
    elif kind == "nonfunctional":
        item = torch.randint(0, n_items, (T,), generator=g)
        a = n_items + torch.randint(0, (U - n_items) // 2, (T,), generator=g)
        b = n_items + (U - n_items) // 2 + torch.randint(0, (U - n_items) // 2, (T,), generator=g)
        return item, torch.stack([a, b], 1).reshape(-1)
    else:
        raise ValueError(kind)
    span = max(1, (U - n_items) // 2)
    a, b = n_items + (item * 31) % span, n_items + span + (item * 17) % span
    return item.to(torch.int64), torch.stack([a, b], 1).reshape(-1).to(torch.int64)


def _record(emb: torch.Tensor, item: torch.Tensor, cat: torch.Tensor, ledgers: Any = ()) -> pd.PPDDRecord:
    return pd.PPDDRecord(emb, item, cat, ledgers, ebo._expand, ebo._CastExpandFp32Bwd.apply, torch.bfloat16)


def _tokens(emb: torch.Tensor, item: torch.Tensor, cat: torch.Tensor, rec: Optional[pd.PPDDRecord],
            dt: torch.dtype = torch.bfloat16) -> Tuple[torch.Tensor, torch.Tensor]:
    """The EC member outputs (item [T, 512], big [T, 1024]) with the shipped cast-expand; tagged when rec."""
    x = ebo._CastExpandFp32Bwd.apply(emb, item, dt)
    big = ebo._CastExpandFp32Bwd.apply(emb, cat, dt).view(item.numel(), 1024)
    if rec is not None:
        setattr(x, pd.PPDD_ATTR, rec)
        setattr(big, pd.PPDD_ATTR, rec)
    return x, big


def _per_token(pre: Any, x: torch.Tensor, big: torch.Tensor) -> torch.Tensor:
    c = pre._content_embedding_mlp(x)
    return pre._mlp_add(pre._additional_embedding_mlp, big, c)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    def o(t: torch.Tensor) -> torch.Tensor:
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)

    return int((o(a) - o(b)).abs().max()) if a.numel() else 0


def _rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    d = float((a.double() - ref.double()).norm())
    n = float(ref.double().norm())
    return d / n if n > 0 else d


GRAD_REPORT_RATIO = 5.0  # dedup error vs fp64 above 5x the per-token BF16 error: REPORT ONLY (accepted, ruling)
GRAD_HARD_RATIO = 10.0  # above 10x the per-token error (floor GRAD_FLOOR), or any non-finite value: FAIL
GRAD_FLOOR = 1e-5  # where the per-token sum is exact (e.g. LN biases on bf16-exact dy); far below bf16 resolution


def _grad_bug_catcher(tc: unittest.TestCase, new: Dict[str, torch.Tensor], old: Dict[str, torch.Tensor],
                      ref: Dict[str, torch.Tensor], tag: str) -> None:
    """Hard: every dedup gradient finite and within GRAD_HARD_RATIO x the per-token error vs fp64 (floor);
    report (no fail) above GRAD_REPORT_RATIO."""
    for k in sorted(ref):
        tc.assertTrue(bool(torch.isfinite(new[k]).all()), f"{tag} {k}: non-finite gradient")
        e_old, e_new = _rel(old[k], ref[k]), _rel(new[k], ref[k])
        tc.assertLessEqual(e_new, max(GRAD_HARD_RATIO * e_old, GRAD_FLOOR), f"{tag} {k}: {e_new:.2e} vs {e_old:.2e}")
        if e_new > max(GRAD_REPORT_RATIO * e_old, GRAD_FLOOR):
            print(f"[{tag}] REPORT {k}: dedup {e_new:.2e} > {GRAD_REPORT_RATIO:g} x per-token {e_old:.2e}")


class _NoHostRead(TorchDispatchMode):
    BAD = ("_local_scalar_dense", "nonzero", "unique", "masked_select", "item", "equal", "is_nonzero")

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        name = func.__name__.split(".")[0]
        self.seen.append(name)
        if any(name == b or name.startswith(b) for b in self.BAD):
            raise AssertionError(f"host-reading op {func}")
        if name == "index" and any(isinstance(a, torch.Tensor) and a.dtype == torch.bool
                                   for a in (args[1] if len(args) > 1 else [])):
            raise AssertionError("boolean index")
        return func(*args, **(kwargs or {}))


# ------------------------------------------------------------------------------------------------ construction
class ConstructionTest(unittest.TestCase):
    CASES = (("heavy", 5_000, 400), ("alldup", 3_000, 400), ("allunique", 150, 400), ("oneheavy", 12_000, 600),
             ("uniform", 1, 400), ("uniform", 2, 400), ("nonfunctional", 2_500, 3_000), ("uniform", 20_000, 9_000))

    def _check(self, item: torch.Tensor, cat: torch.Tensor, U: int, m_pad: int) -> pd.Triples:
        tri = pd.build_triples(item, cat, U, m_pad)
        ref = pd._host_reference(item, cat, U, m_pad)
        self.assertEqual(int(tri.n_u[0]), int(ref.n_u[0]))
        for got, want in zip((tri.inv, tri.order, tri.offsets) + tri.rows, (ref.inv, ref.order, ref.offsets) + ref.rows):
            self.assertTrue(torch.equal(got, want))
        return tri

    def test_patterns_match_host_reference(self) -> None:
        for kind, T, U in self.CASES:
            item, cat = _batch(kind, T, U, seed=T)
            m_pad = pd.pad_rows(U, T)
            tri = self._check(item, cat, U, m_pad)
            n_u = int(tri.n_u[0])
            self.assertLessEqual(n_u, m_pad, kind)
            if kind == "alldup":
                self.assertEqual(n_u, 1)
            if kind == "allunique":
                self.assertEqual(n_u, T)
            # gathering the unique rows back reproduces every token's triple
            self.assertTrue(torch.equal(tri.rows[0][tri.inv], item))
            self.assertTrue(torch.equal(torch.stack([tri.rows[1][tri.inv], tri.rows[2][tri.inv]], 1).reshape(-1), cat))

    def test_mpad_boundary_exact_and_overflow(self) -> None:
        # n_u == M_pad exactly (all-unique, T = U = 512): fits; T > U with non-functional ids: n_u > M_pad.
        U = 512
        item = torch.arange(512, dtype=torch.int64)
        cat = torch.stack([item, (item + 1) % U], 1).reshape(-1)
        tri = self._check(item, cat, U, pd.pad_rows(U, 512))
        self.assertEqual(int(tri.n_u[0]), pd.pad_rows(U, 512))
        item2 = torch.arange(513, dtype=torch.int64) % U
        cat2 = torch.stack([torch.arange(513) % 7, torch.arange(513) % 11], 1).reshape(-1)
        tri2 = pd.build_triples(item2, cat2, U, pd.pad_rows(U, 513))
        self.assertEqual(int(tri2.n_u[0]), 513)  # > M_pad = 512: indices stay clamped in bounds
        self.assertLess(int(tri2.inv.max()), 512)
        self.assertLessEqual(int(tri2.offsets.max()), 513)

    def test_pad_rows(self) -> None:
        self.assertEqual(pd.pad_rows(95_000, 2_600_000), 95_232)
        self.assertEqual(pd.pad_rows(95_000, 10), 256)
        self.assertEqual(pd.pad_rows(1, 1), 256)
        self.assertEqual(pd.pad_rows(512, 513), 512)

    def test_stable_and_deterministic(self) -> None:
        item, cat = _batch("oneheavy", 9_000, 500)
        a = pd.build_triples(item, cat, 500, pd.pad_rows(500, 9_000))
        b = pd.build_triples(item, cat, 500, pd.pad_rows(500, 9_000))
        for x, y in zip(a, b):
            if isinstance(x, tuple):
                for p, q in zip(x, y):
                    self.assertTrue(torch.equal(p, q))
            else:
                self.assertTrue(torch.equal(x, y))
        seg = a.inv[a.order]
        self.assertTrue(bool((seg[1:] >= seg[:-1]).all()))  # sorted by triple
        for u in range(0, int(a.n_u[0]), 37):  # tokens of a triple keep their order (stable)
            toks = a.order[int(a.offsets[u]):int(a.offsets[u + 1])]
            self.assertTrue(bool((toks[1:] > toks[:-1]).all()))

    def test_no_host_read_op_dispatched(self) -> None:
        _arm()
        pre = _pre()
        U, T = 400, 6_000
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U)
        rec = _record(emb, item, cat)
        mode = _NoHostRead()
        with mode, torch.autocast("cpu", dtype=torch.bfloat16):
            out = pd._run_core(pre, rec, hilo=True)
        with mode:
            out.float().sum().backward()
        for op in ("sort", "cumsum", "searchsorted", "scatter_", "index_select"):
            self.assertIn(op, mode.seen)

    def test_overflow_flag_raises_at_next_call_and_counters(self) -> None:
        _arm()
        pre = _pre()
        U, T = 64, 600  # M_pad = 256 < n_u (non-functional ids)
        emb = torch.randn(U, 512)
        item = torch.arange(T) % U
        cat = torch.stack([torch.arange(T) % 7, torch.arange(T) % 11], 1).reshape(-1)
        rec = _record(emb, item, cat)
        x, big = _tokens(emb, item, cat, rec)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = pd.run(pre, x, big)
        self.assertIsNotNone(out)  # the flag is device-side; never read in the step
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaisesRegex(RuntimeError, "OVERFLOW"):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                pd.run(pre, x, big)
        self.assertIn(pd.OVERFLOW_MARKER, buf.getvalue())
        m = re.search(pd.COUNTERS_RX, pd.counters_line())
        self.assertIsNotNone(m)
        self.assertGreaterEqual(int(m.group(7)), 1)
        self.assertEqual(int(m.group(10)), 600)  # (t % 64, t % 7, t % 11): all 600 tokens are distinct triples


# ------------------------------------------------------------------------------------------------ (a1) logic
class ForwardEqualityTest(unittest.TestCase):
    """(a1): dedup vs per-token, both on the bf16 autocast path. Measured values are printed (-s)."""

    PATTERNS = (("heavy", 6_000, 400), ("alldup", 2_048, 400), ("allunique", 200, 400), ("oneheavy", 10_000, 900),
                ("uniform", 1, 400), ("uniform", 4_096, 512))
    MAX_FRAC, MAX_ULP = 1e-3, 2

    def test_bf16_forward_matches_per_token(self) -> None:
        pre = _pre()
        for kind, T, U in self.PATTERNS:
            _arm()
            torch.manual_seed(T)
            emb = torch.randn(U, 512)
            item, cat = _batch(kind, T, U, seed=7)
            rec = _record(emb, item, cat)
            x, big = _tokens(emb, item, cat, rec)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = pd.run(pre, x, big)
                ref = _per_token(pre, x, big)
            self.assertIsNotNone(out, kind)
            frac = float((out != ref).float().mean())
            ulp = _ulps(out, ref)
            print(f"[a1 fwd] {kind} T={T} U={U}: mismatch fraction {frac:.3e}, max {ulp} ulp")
            self.assertLessEqual(frac, self.MAX_FRAC, kind)
            self.assertLessEqual(ulp, self.MAX_ULP, kind)

    def test_fp64_wiring_bitwise(self) -> None:
        # Plain fp64 torch modules (the repo's PYTORCH LayerNorm reference computes in fp32 internally): the
        # unique / gather / hi-lo / backward wiring itself is then exact to fp64 rounding.
        _arm()

        class _Swish(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln = torch.nn.LayerNorm(256)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * torch.sigmoid(self.ln(x))

        class _Pre64(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                torch.manual_seed(11)
                self._content_embedding_mlp = torch.nn.Sequential(
                    torch.nn.Linear(512, 256), _Swish(), torch.nn.Linear(256, 512), torch.nn.LayerNorm(512))
                self._additional_embedding_mlp = torch.nn.Sequential(
                    torch.nn.Linear(1024, 256), _Swish(), torch.nn.Linear(256, 512), torch.nn.LayerNorm(512))

            def _mlp_add(self, mlp: Any, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
                return residual + mlp(x)

        pre = _Pre64().double()
        U, T = 300, 4_000
        emb = torch.randn(U, 512, dtype=torch.float64, requires_grad=True)
        item, cat = _batch("oneheavy", T, U)
        rec = pd.PPDDRecord(emb, item, cat, (), ebo._expand, ebo._CastExpandFp32Bwd.apply, torch.float64)
        for hilo in (False, True):
            out = pd._run_core(pre, rec, hilo=hilo)
            x, big = _tokens(emb, item, cat, None, torch.float64)
            ref = _per_token(pre, x, big)
            self.assertEqual(float((out - ref).abs().max()), 0.0)
            g = torch.randn(T, 512, dtype=torch.float64)
            ga = torch.autograd.grad(out, [emb] + list(pre.parameters()), g, allow_unused=True)
            gb = torch.autograd.grad(ref, [emb] + list(pre.parameters()), g, allow_unused=True)
            for a, b in zip(ga, gb):
                if b is None:
                    continue
                self.assertLessEqual(_rel(a, b), 1e-12)


class PadSpreadTest(unittest.TestCase):
    """(118) sub-knob PADSPREAD: padding triples name rows u % U; construction == host reference; forward output and
    every gradient bitwise equal to PADSPREAD off (the padding rows' backward is exact zeros)."""

    def tearDown(self) -> None:
        pd._PADSPREAD = None

    def test_knob_reads_only_1(self) -> None:
        for v, want in (("1", True), ("0", False), ("", False), ("true", False)):
            pd._PADSPREAD = None
            with mock.patch.dict(os.environ, {pd.PADSPREAD_ENV: v}):
                self.assertEqual(pd.padspread(), want, v)

    def test_padding_rows_spread_and_match_host_reference(self) -> None:
        pd._PADSPREAD = True
        for kind, T, U in ConstructionTest.CASES:
            item, cat = _batch(kind, T, U, seed=T)
            m_pad = pd.pad_rows(U, T)
            tri = ConstructionTest._check(self, item, cat, U, m_pad)
            n_u = min(int(tri.n_u[0]), m_pad)
            want = torch.arange(n_u, m_pad) % U
            for r in tri.rows:
                self.assertTrue(torch.equal(r[n_u:], want), kind)
            self.assertTrue(torch.equal(tri.rows[0][tri.inv], item))

    def test_bitwise_equal_to_off(self) -> None:
        pre = _pre()
        torch.manual_seed(11)
        U, T = 900, 5_000
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U, seed=9)
        g = torch.randn(T, 512)
        res = []
        for on in (False, True):
            with mock.patch.dict(os.environ, {pd.PADSPREAD_ENV: "1" if on else "0"}):  # _grads -> _arm() re-reads the env
                res.append(BackwardTest._grads(self, pre, emb, item, cat, g, "dedup"))
        for k in res[0]:
            self.assertTrue(torch.equal(res[0][k], res[1][k]), k)


class Hilo2Test(unittest.TestCase):
    """(118) sub-knob HILO2: the two-launch hi / lo split returns exactly the cast / sub / cast / cat grad."""

    def tearDown(self) -> None:
        pd._HILO2 = None

    def test_split_bitwise(self) -> None:
        torch.manual_seed(3)
        s = torch.randn(300, 512) * torch.exp2(torch.randint(-20, 20, (300, 512)).float())
        hi = s.to(torch.bfloat16)
        want = torch.cat([hi, (s - hi.float()).to(torch.bfloat16)], 0)
        d = torch.empty((600, 512), dtype=torch.bfloat16)
        d[:300].copy_(s)
        torch.sub(s, d[:300], out=d[300:])
        self.assertTrue(torch.equal(d.view(torch.int16), want.view(torch.int16)))

    def test_grads_bitwise_equal_to_off(self) -> None:
        pre = _pre()
        torch.manual_seed(12)
        U, T = 900, 5_000
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U, seed=9)
        g = torch.randn(T, 512)
        res = []
        for on in (False, True):
            with mock.patch.dict(os.environ, {pd.HILO2_ENV: "1" if on else "0"}):  # _grads -> _arm() re-reads the env
                res.append(BackwardTest._grads(self, pre, emb, item, cat, g, "dedup"))
        for k in res[0]:
            self.assertTrue(torch.equal(res[0][k], res[1][k]), k)


def _subknobs(lean: bool, pad: bool = False, hilo2: bool = False) -> Dict[str, str]:
    """Explicit values for the three (118) sub-knobs (reset_for_testing / _arm re-read them from the environment)."""
    return {pd.LEAN_ENV: "1" if lean else "0", pd.PADSPREAD_ENV: "1" if pad else "0",
            pd.HILO2_ENV: "1" if hilo2 else "0"}


def _seg_today(skey: torch.Tensor) -> torch.Tensor:
    """build_unique's segment ids as written before LEAN."""
    T = skey.numel()
    new = torch.ones(T, dtype=torch.bool)
    if T > 1:
        new[1:] = skey[1:] != skey[:-1]
    return torch.cumsum(new, 0, dtype=torch.int64) - 1


class LeanTest(unittest.TestCase):
    """(118) sub-knob LEAN: fewer launches, the same tensors. Every construction output (integer index tensors:
    torch.equal + dtype + shape), the unique chain's output and every gradient are bitwise equal to LEAN off, under
    PADSPREAD / HILO2 on and off, training (hilo) and eval rows, and on edge inputs (T = 0 / 1 / 2, all-duplicate,
    all-unique, n_u > M_pad)."""

    PATTERNS = ConstructionTest.CASES + (("alldup", 1, 400), ("allunique", 256, 400), ("allunique", 257, 600))

    def setUp(self) -> None:
        self._patch = mock.patch.dict(os.environ, {})  # restores the environment after the test
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        pd.reset_for_testing()

    def _set(self, lean: bool, pad: bool = False, hilo2: bool = False) -> None:
        os.environ.update(_subknobs(lean, pad, hilo2))
        pd.reset_for_testing()

    def _same(self, a: torch.Tensor, b: torch.Tensor, what: str) -> None:
        self.assertEqual((a.dtype, tuple(a.shape)), (b.dtype, tuple(b.shape)), what)
        self.assertTrue(torch.equal(a, b), what)

    def _triples(self, lean: bool, pad: bool, item: torch.Tensor, cat: torch.Tensor, U: int, m_pad: int
                 ) -> pd.Triples:
        self._set(lean, pad)
        return pd.build_triples(item, cat, U, m_pad)

    def test_knob_reads_only_1_and_once(self) -> None:
        for v, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("true", False), ("2", False)):
            pd._LEAN = None
            os.environ[pd.LEAN_ENV] = v
            self.assertEqual(pd.lean(), want, v)
            os.environ[pd.LEAN_ENV] = "0" if want else "1"  # read once
            self.assertEqual(pd.lean(), want, v)
        os.environ.pop(pd.LEAN_ENV)
        pd.reset_for_testing()
        self.assertFalse(pd.lean())  # unset: off

    def test_segment_ids_equal_today_every_length(self) -> None:
        g = torch.Generator().manual_seed(1)
        cases = [torch.zeros(0, dtype=torch.int64), torch.tensor([7]), torch.tensor([7, 7]), torch.tensor([3, 9]),
                 torch.full((1_000,), 5, dtype=torch.int64), torch.arange(1_000),
                 torch.tensor([-(1 << 62), -1, -1, 0, 0, 1 << 62, 1 << 62])]
        cases += [torch.randint(0, n, (5_000,), generator=g).sort().values for n in (2, 50, 4_000)]
        for skey in cases:
            got = pd._segment_ids(skey)
            self._same(got, _seg_today(skey), f"T={skey.numel()}")

    def test_build_unique_bitwise(self) -> None:
        g = torch.Generator().manual_seed(2)
        cases = [(torch.tensor([5, 3, 5, 9, 3, 3]), 10, None), (torch.tensor([4]), 10, None),
                 (torch.tensor([4, 4]), 10, None), (torch.tensor([4, 2]), 10, None),
                 (torch.full((700,), 11, dtype=torch.int64), 50, None), (torch.randperm(600), 1_000, None),
                 (torch.randint(0, 900, (3_000,), generator=g), 1_000, None),
                 (torch.randint(0, 900, (3_000,), generator=g), 1_000, 256),  # n_u > M_pad: clamped, flagged
                 (torch.randint(0, 1 << 40, (2_000,), generator=g), 1 << 20, 512)]
        for keys, U, mp in cases:
            outs = []
            for lean in (False, True):
                self._set(lean)
                outs.append(pd.build_unique(keys, U, mp))
            self.assertEqual(outs[0][3], outs[1][3])
            for i, name in ((0, "inv"), (1, "order"), (2, "offsets"), (4, "n_u"), (5, "first")):
                self._same(outs[1][i], outs[0][i], f"{name} T={keys.numel()} M_pad={mp}")

    def test_build_triples_bitwise_and_host_reference(self) -> None:
        overflow = (torch.arange(513, dtype=torch.int64) % 512,
                    torch.stack([torch.arange(513) % 7, torch.arange(513) % 11], 1).reshape(-1), 512)
        batches = [_batch(kind, T, U, seed=T) + (U,) for kind, T, U in self.PATTERNS] + [overflow]
        for pad in (False, True):
            for item, cat, U in batches:
                T = item.numel()
                m_pad = pd.pad_rows(U, T)
                off = self._triples(False, pad, item, cat, U, m_pad)
                on = self._triples(True, pad, item, cat, U, m_pad)
                what = f"pad={pad} T={T} U={U}"
                for name in ("inv", "order", "offsets", "n_u"):
                    self._same(getattr(on, name), getattr(off, name), f"{name} {what}")
                for k in range(3):
                    self._same(on.rows[k], off.rows[k], f"rows[{k}] {what}")
                if int(on.n_u[0]) <= m_pad:  # the host reference (padspread() is still LEAN-on's read)
                    ref = pd._host_reference(item, cat, U, m_pad)
                    for k in range(3):
                        self._same(on.rows[k], ref.rows[k], f"ref rows[{k}] {what}")

    def test_hilo_index_rows_bitwise(self) -> None:
        # _run_core's index tensors: today's cat([r] * 2) x 3 + stack(.., 1).reshape(-1) vs LEAN's folded launches.
        for pad in (False, True):
            for kind, T, U in self.PATTERNS:
                item, cat = _batch(kind, T, U, seed=T + 1)
                m_pad = pd.pad_rows(U, T)
                r_i, r_a, r_b = self._triples(False, pad, item, cat, U, m_pad).rows
                self._set(True, pad)
                for copies in (1, 2):
                    want_i = torch.cat([r_i] * copies)
                    want_ab = torch.stack([torch.cat([r_a] * copies), torch.cat([r_b] * copies)], dim=1).reshape(-1)
                    _, got_i, got_ab = pd._lean_triples(item, cat, U, m_pad, copies)
                    self.assertTrue(got_i.is_contiguous() and got_ab.is_contiguous())
                    what = f"pad={pad} {kind} T={T} copies={copies}"
                    self._same(got_i.view(-1), want_i, f"r_i {what}")
                    self._same(got_ab.view(-1), want_ab, f"r_ab {what}")

    def _core(self, lean: bool, pad: bool, hilo2: bool, pre: Any, emb: torch.Tensor, item: torch.Tensor,
              cat: torch.Tensor, g: torch.Tensor, hilo: bool) -> Dict[str, torch.Tensor]:
        self._set(lean, pad, hilo2)
        _arm()  # resets again: the sub-knobs are re-read from the environment set above
        pre.zero_grad(set_to_none=True)
        emb.grad = None
        with torch.autocast("cpu", dtype=torch.bfloat16):
            o = pd._run_core(pre, _record(emb, item, cat), hilo=hilo)
        o.backward(g.to(o.dtype))
        out = {"out": o.detach().clone(), "emb": emb.grad.clone()}
        out.update({n: p.grad.clone() for n, p in pre.named_parameters() if p.grad is not None})
        self.assertEqual(pd.lean(), lean)
        return out

    def test_run_core_output_and_every_gradient_bitwise(self) -> None:
        pre = _pre()
        torch.manual_seed(13)
        for kind, T, U in (("heavy", 4_000, 900), ("oneheavy", 3_000, 600), ("allunique", 300, 700),
                           ("uniform", 1, 400), ("alldup", 2_048, 400)):
            emb = torch.randn(U, 512, requires_grad=True)
            item, cat = _batch(kind, T, U, seed=5)
            g = torch.randn(T, 512)
            for pad, hilo2, hilo in ((False, False, True), (True, False, True), (False, True, True),
                                     (True, True, True), (False, False, False), (True, False, False)):
                a = self._core(False, pad, hilo2, pre, emb, item, cat, g, hilo)
                b = self._core(True, pad, hilo2, pre, emb, item, cat, g, hilo)
                self.assertEqual(set(a), set(b))
                for k in a:
                    self._same(b[k], a[k], f"{k} {kind} pad={pad} hilo2={hilo2} hilo={hilo}")

    def test_no_host_read_op_dispatched(self) -> None:
        self._set(True, True, True)
        _arm()
        pre = _pre()
        U, T = 400, 6_000
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U)
        mode = _NoHostRead()
        with mode, torch.autocast("cpu", dtype=torch.bfloat16):
            out = pd._run_core(pre, _record(emb, item, cat), hilo=True)
        with mode:
            out.float().sum().backward()
        for op in ("sort", "cumsum", "searchsorted", "scatter_", "index_select", "ne", "where"):
            self.assertIn(op, mode.seen)

    NO_LAUNCH = ("view", "_unsafe_view", "slice", "select", "expand", "t", "transpose", "unsqueeze", "squeeze",
                 "as_strided", "alias", "detach", "empty", "empty_like", "lift_fresh")

    def test_construction_dispatches_fewer_ops(self) -> None:
        # Report + guard: dispatched non-view aten ops of the training index construction (rows, [u; u] copies and
        # (a, b) interleave), today's statements vs LEAN; LEAN issues no cat / stack. (Dispatch-level count: ops a
        # kernel calls internally, e.g. cumsum's bool -> int64 cast, are not seen here.)
        item, cat = _batch("heavy", 5_000, 900)
        m_pad = pd.pad_rows(900, 5_000)
        for pad in (False, True):
            seen = []
            for lean in (False, True):
                self._set(lean, pad)
                pd._arange(m_pad + 1, CPU, torch.int64)  # the cache warm, as after prebuild
                mode = _NoHostRead()
                with mode:
                    if lean:
                        pd._lean_triples(item, cat, 900, m_pad, 2)
                    else:
                        r_i, r_a, r_b = pd.build_triples(item, cat, 900, m_pad).rows
                        r_i, r_a, r_b = torch.cat([r_i, r_i]), torch.cat([r_a, r_a]), torch.cat([r_b, r_b])
                        torch.stack([r_a, r_b], dim=1).reshape(-1)
                seen.append([op for op in mode.seen if op not in self.NO_LAUNCH])
            print(f"[lean] pad={pad}: dispatched ops today {len(seen[0])} {seen[0]} -> LEAN {len(seen[1])} {seen[1]}")
            self.assertLessEqual(len(seen[1]), len(seen[0]) - (8 if pad else 5))
            for op in ("cat", "stack", "ones", "arange", "sub", "mul"):
                self.assertNotIn(op, seen[1])

    def test_arange_cache_is_a_constant_prefix(self) -> None:
        self._set(True)
        for n in (1, 2, 257, 4_097, 300, 1):
            a = pd._arange(n, CPU, torch.int64)
            self._same(a, torch.arange(n, dtype=torch.int64), f"n={n}")
        self.assertEqual(list(pd._AR), [("cpu", -1, torch.int64)])  # one entry per (device, dtype)
        self.assertEqual(pd._AR[("cpu", -1, torch.int64)].numel(), 8_192)  # power of two >= 4,097
        self._same(pd._arange(5, CPU, torch.int32), torch.arange(5, dtype=torch.int32), "int32")
        self.assertEqual(len(pd._AR), 2)
        with torch.inference_mode():
            pd._arange(20_000, CPU, torch.int64)  # grows under inference_mode: still a normal tensor
        self.assertFalse(pd._AR[("cpu", -1, torch.int64)].is_inference())
        self._set(True, True)
        item, cat = _batch("heavy", 5_000, 900)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            pd._run_core(_pre(), _record(torch.randn(900, 512), item, cat), hilo=True)  # never written into
        cached = pd._AR[("cpu", -1, torch.int64)]
        self._same(cached, torch.arange(cached.numel()), "after use")
        pd.reset_for_testing()
        self.assertEqual(pd._AR, {})

    def test_overflow_flag_and_counters_line_unchanged(self) -> None:
        U, T = 64, 600  # M_pad = 256 < n_u = 600 (non-functional ids)
        emb = torch.randn(U, 512)
        item = torch.arange(T) % U
        cat = torch.stack([torch.arange(T) % 7, torch.arange(T) % 11], 1).reshape(-1)
        lines = []
        for lean in (False, True):
            self._set(lean)
            _arm()
            pre = _pre()
            rec = _record(emb, item, cat)
            x, big = _tokens(emb, item, cat, rec)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertIsNotNone(pd.run(pre, x, big))
            lines.append(pd.counters_line())
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaisesRegex(RuntimeError, "OVERFLOW"):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    pd.run(pre, x, big)
            self.assertIn(pd.OVERFLOW_MARKER + ": a batch had more unique (item, a, b) triples than M_pad rows "
                          "(max n_u 600)", buf.getvalue())
        self.assertEqual(lines[0], lines[1])  # overflow=1 (not n_u - M_pad), max_n_u=600
        m = re.search(pd.COUNTERS_RX, lines[1])
        self.assertIsNotNone(m)
        self.assertEqual((int(m.group(7)), int(m.group(10))), (1, 600))
        self.assertEqual([int(v) for v in pd._OVF[("cpu", -1)].buf], [600 - 256, 600])
        # no overflow: buf[0] stays 0
        self._set(True)
        _arm()
        item2, cat2 = _batch("heavy", 3_000, 400)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            pd._run_core(_pre(), _record(torch.randn(400, 512), item2, cat2), hilo=True)
        self.assertEqual(int(pd._OVF[("cpu", -1)].buf[0]), 0)
        self.assertEqual(pd._overflow_read()[0], 0)

    def test_mlp_reason_static_part_cached_can_fuse_per_call(self) -> None:
        self._set(True)
        pre = _pre()
        x = torch.zeros(4, 512)
        mlp = pre._content_embedding_mlp
        self._set(False)
        want = pd._mlp_reason(mlp, 512, x), pd._mlp_reason(mlp, 1024, x)
        self._set(True)
        with mock.patch.object(pd, "_mlp_static_reason", wraps=pd._mlp_static_reason) as spy:
            for _ in range(3):
                self.assertEqual((pd._mlp_reason(mlp, 512, x), pd._mlp_reason(mlp, 1024, x)), want)
            self.assertEqual(spy.call_count, 2)  # once per (module, k)
            seq = torch.nn.Sequential(*list(mlp))  # another module: its own entry, today's reason
            self.assertIn("is not FusedLinearNormMLP", pd._mlp_reason(seq, 512, x))
            self.assertEqual(spy.call_count, 3)
        cuda_x = mock.Mock(is_cuda=True)  # _can_fuse still runs per call, after the cached static part
        with mock.patch.object(mlp, "_can_fuse", lambda t: False):
            self.assertEqual(pd._mlp_reason(mlp, 512, cuda_x),
                             "FusedLinearNormMLP._can_fuse declined (non-Triton norm)")
        with mock.patch.object(mlp, "_can_fuse", lambda t: True):
            self.assertIsNone(pd._mlp_reason(mlp, 512, cuda_x))

    def test_declines_and_tracing_unchanged(self) -> None:
        from generative_recommenders.ops.triton import preproc_fp8_snr

        U, T = 300, 3_000
        emb = torch.randn(U, 512)
        item, cat = _batch("heavy", T, U)

        def run(pre: Any, autocast: bool = True) -> Tuple[Optional[torch.Tensor], str]:
            rec = _record(emb, item, cat)
            x, big = _tokens(emb, item, cat, rec)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                out = pd.run(pre, x, big)
            return out, buf.getvalue()

        for lean in (False, True):
            self._set(lean)
            _arm()
            pre = _pre()
            self.assertIsNotNone(run(pre)[0])
            out, log = run(pre, autocast=False)
            self.assertIsNone(out)
            self.assertIn("(autocast)", log)
            old = preproc_fp8_snr.ENABLED
            preproc_fp8_snr.ENABLED = True  # read per call through the cached module
            try:
                out, log = run(pre)
            finally:
                preproc_fp8_snr.ENABLED = old
            self.assertIsNone(out)
            self.assertIn("(snr)", log)
            pre._content_embedding_mlp = torch.nn.Sequential(*list(pre._content_embedding_mlp))
            out, log = run(pre)
            self.assertIsNone(out)
            self.assertIn("(mlp)", log)
            x, _ = _tokens(emb, item, cat, _record(emb, item, cat))
            buf = io.StringIO()
            with mock.patch("torch.fx._symbolic_trace.is_fx_tracing", lambda: True), contextlib.redirect_stdout(buf):
                self.assertIsNone(pd.run(pre, x, None))  # the patched module attribute is still what is read
            self.assertEqual(buf.getvalue(), "")

    def test_self_check_exact_on_cpu_with_lean(self) -> None:
        for pad in (False, True):
            self._set(True, pad)
            self.assertIn("construction == host reference", pd.self_check(CPU))


class BackwardTest(unittest.TestCase):
    """Every gradient vs an fp64 per-token reference: new <= max(2 x old, 1e-3) (old = per-token bf16)."""

    def _grads(self, pre: Any, emb: torch.Tensor, item: torch.Tensor, cat: torch.Tensor, g: torch.Tensor,
               mode: str, hilo: bool = True) -> Dict[str, torch.Tensor]:
        pre.zero_grad(set_to_none=True)
        emb.grad = None
        if mode == "fp64":
            p64 = _pre(dtype=torch.float64)
            p64.load_state_dict({k: v.double() for k, v in pre.state_dict().items()})
            e64 = emb.detach().double().requires_grad_(True)
            x, big = _tokens(e64, item, cat, None, torch.float64)
            _per_token(p64, x, big).backward(g.double())
            out = {"emb": e64.grad}
            out.update({n: p.grad for n, p in p64.named_parameters() if p.grad is not None})
            return out
        with torch.autocast("cpu", dtype=torch.bfloat16):
            if mode == "dedup":
                _arm()
                o = pd._run_core(pre, _record(emb, item, cat), hilo=hilo)
            else:
                x, big = _tokens(emb, item, cat, None)
                o = _per_token(pre, x, big)
        o.backward(g.to(o.dtype))
        out = {"emb": emb.grad.clone()}
        out.update({n: p.grad.clone() for n, p in pre.named_parameters() if p.grad is not None})
        return out

    def test_every_gradient_vs_fp64(self) -> None:
        pre = _pre()
        torch.manual_seed(5)
        U, T = 600, 12_000
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("oneheavy", T, U, seed=3)
        g = torch.randn(T, 512)
        ref = self._grads(pre, emb, item, cat, g, "fp64")
        old = self._grads(pre, emb, item, cat, g, "token")
        new = self._grads(pre, emb, item, cat, g, "dedup")
        self.assertEqual(set(new), set(ref))
        worst = 0.0
        for k in sorted(ref):
            e_old, e_new = _rel(old[k], ref[k]), _rel(new[k], ref[k])
            worst = max(worst, e_new / max(2 * e_old, 1e-3))
            print(f"[a1 bwd] {k}: per-token {e_old:.2e}, dedup {e_new:.2e} (rel-L2 vs fp64)")
            self.assertLessEqual(e_new, max(2 * e_old, 1e-3), k)
        print(f"[a1 bwd] worst new / max(2 old, 1e-3) = {worst:.3f}")
        _grad_bug_catcher(self, new, old, ref, "a1 bwd")

    def test_hi_only_mutant_fails_the_ln_bias_criterion(self) -> None:
        pre = _pre()
        torch.manual_seed(6)
        U, T = 300, 40_000  # ~100 tokens per triple: the bf16-once rounding of the sums shows in column sums
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U, seed=4)
        g = torch.randn(T, 512).to(torch.bfloat16).float()  # bf16-exact dO: isolates the summation (P3's setup)
        ref = self._grads(pre, emb, item, cat, g, "fp64")
        old = self._grads(pre, emb, item, cat, g, "token")
        new = self._grads(pre, emb, item, cat, g, "dedup", hilo=True)
        mut = self._grads(pre, emb, item, cat, g, "dedup", hilo=False)
        k = "_content_embedding_mlp.3.bias"
        e_old, e_new, e_mut = _rel(old[k], ref[k]), _rel(new[k], ref[k]), _rel(mut[k], ref[k])
        print(f"[hilo] {k}: per-token {e_old:.2e}, hi+lo {e_new:.2e}, hi only {e_mut:.2e}")
        self.assertLessEqual(e_new, max(2 * e_old, 1e-5))
        self.assertGreater(e_mut, 10 * max(e_new, e_old))
        _grad_bug_catcher(self, new, old, ref, "hilo")

    def test_mpad_padding_rows_do_not_change_results(self) -> None:
        pre = _pre()
        U, T = 900, 5_000
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U, seed=9)
        n_u = int(pd._host_reference(item, cat, U, 1 << 12).n_u[0])
        g = torch.randn(T, 512)
        res = []
        orig = pd.pad_rows
        try:
            for m in (n_u, n_u + 512):
                pd.pad_rows = lambda U_, T_, m=m: m  # noqa: E731
                res.append(self._grads(pre, emb, item, cat, g, "dedup"))
        finally:
            pd.pad_rows = orig
        for k in res[0]:
            self.assertLessEqual(_rel(res[0][k], res[1][k]), 1e-6, k)


# ------------------------------------------------------------------------------------------------ wiring
class PreprocessorWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        _ensure_cumsum()
        _arm()

    @staticmethod
    def _forward(self: Any, pre: Any, x: torch.Tensor, big: torch.Tensor, T: int) -> torch.Tensor:
        B = 4
        lengths = torch.full((B,), T // B, dtype=torch.int64)
        payloads = {"art": big[:, :512], "alb": big[:, 512:], "act": torch.arange(T, dtype=torch.int64) % 8}
        setattr(payloads["art"], ebo.CATFUSED_ATTR, (big, 0, 2))
        setattr(payloads["alb"], ebo.CATFUSED_ATTR, (big, 1, 2))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = pre(max_uih_len=T // B, max_targets=0, total_uih_len=T, total_targets=0, seq_lengths=lengths,
                      seq_timestamps=torch.zeros(T, dtype=torch.int64), seq_embeddings=x,
                      num_targets=torch.zeros(B, dtype=torch.int64), seq_payloads=payloads)
        return out[6]

    def test_forward_with_action_mlp_matches_and_no_grouped_tail(self) -> None:
        from generative_recommenders.modules import preprocessors as pp

        pre = _pre(action=True)
        U, T = 500, 4_000
        # Stand-in ActionEncoder (its shipped chain needs fbgemm's jagged ops, absent here): a fixed [T, K] input
        # to the action MLP, so the branch runs its real `tail is None` statement _mlp_add(action, x, residual).
        k_in = pre._action_embedding_mlp[0].in_features
        x_act = torch.randn(T, k_in)

        class _Enc(torch.nn.Module):
            def forward(self, **kw: Any) -> torch.Tensor:
                return x_act

        pre._action_encoder = _Enc()
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U)
        calls: List[str] = []
        old_flag, old_begin = pp._PREPROC_GROUPED_TAIL, pre._grouped_tail_begin
        pp._PREPROC_GROUPED_TAIL = True
        pre._grouped_tail_begin = lambda *a, **k: calls.append("tail") or None  # type: ignore[assignment]
        try:
            x, big = _tokens(emb, item, cat, _record(emb, item, cat))
            o_dd = self._forward(self, pre, x, big, T)
            self.assertEqual(calls, [])  # dedup engaged: no grouped tail begun
            self.assertEqual(pd.COUNTS["dedup"], 1)
            g = torch.randn_like(o_dd, dtype=torch.float32)
            gd = torch.autograd.grad(o_dd, [emb] + list(pre.parameters()), g.to(o_dd.dtype))
            x2, big2 = _tokens(emb, item, cat, None)
            o_tok = self._forward(self, pre, x2, big2, T)
            self.assertEqual(calls, ["tail"])  # untagged: today's statements (grouped tail requested)
            gt = torch.autograd.grad(o_tok, [emb] + list(pre.parameters()), g.to(o_tok.dtype))
        finally:
            pp._PREPROC_GROUPED_TAIL = old_flag
            pre._grouped_tail_begin = old_begin  # type: ignore[assignment]
        frac = float((o_dd != o_tok).float().mean())
        print(f"[wiring] full forward with the action MLP: mismatch fraction {frac:.3e}, max {_ulps(o_dd, o_tok)} ulp")
        self.assertLessEqual(frac, 1e-3)
        self.assertLessEqual(_ulps(o_dd, o_tok), 2)
        names = ["emb"] + [n for n, _ in pre.named_parameters()]
        for n, a, b in zip(names, gd, gt):
            self.assertLessEqual(_rel(a, b), 2e-2, n)  # bf16 per-token vs dedup (reduction order); fp64: above

    def test_action_tail_from_prefill_gets_acc_tok(self) -> None:
        pre = _pre(action=True)
        seen = {}

        class _Pre:
            fused, l2 = False, None

        orig = type(pre)._action_tail_from_prefill
        acc = torch.randn(8, 512).to(torch.bfloat16)
        x = torch.randn(8, 256).to(torch.bfloat16)

        def spy(self_: Any, p: Any, xx: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            seen["residual"] = residual
            return orig(self_, p, xx, residual)

        mlp = pre._action_embedding_mlp
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = spy(pre, _Pre(), x, acc)
            ref = mlp[3](mlp[2](x)) + acc  # the F2 statement's value: bf16(LN) + residual
        self.assertIs(seen["residual"], acc)
        self.assertTrue(torch.equal(out, ref))


class ECTagTest(unittest.TestCase):
    """_merged_construct's grouped branch (fake torchrec JaggedTensor; forward only: the grouped EC backward is a
    GPU kernel)."""

    def setUp(self) -> None:
        import types

        _ensure_cumsum()
        mod = types.ModuleType("torchrec.sparse.jagged_tensor")

        class JaggedTensor:
            def __init__(self, lengths: torch.Tensor, values: torch.Tensor, weights: Any = None) -> None:
                self.lengths_, self.values_ = lengths, values

            def values(self) -> torch.Tensor:
                return self.values_

        mod.JaggedTensor = JaggedTensor  # type: ignore[attr-defined]
        self._saved = {k: sys.modules.get(k) for k in ("torchrec", "torchrec.sparse", "torchrec.sparse.jagged_tensor")}
        for k in ("torchrec", "torchrec.sparse"):
            sys.modules.setdefault(k, types.ModuleType(k))
        sys.modules["torchrec.sparse.jagged_tensor"] = mod
        self._g = (ebo._MERGE_PAIRS[:], ebo._CAT_GROUP[:], ebo._SEGMENT_REDUCE, ebo._PPDD_ON, ebo._PPDD_CHAIN,
                   ebo._PPDD_UIH_POST_ID, ebo._IX_ON, ebo._EX2B_ON, os.environ.get(ebo._EC_GROUP_ENV))
        ebo._MERGE_PAIRS[:] = [("item", "c_item"), ("art", "c_art"), ("alb", "c_alb")]
        ebo._CAT_GROUP[:] = ["art", "alb"]
        ebo._SEGMENT_REDUCE, ebo._IX_ON, ebo._EX2B_ON = True, False, False
        os.environ[ebo._EC_GROUP_ENV] = "1"
        ebo.note_preproc_dedup_chain(True, True, True, True, True, "item")

    def tearDown(self) -> None:
        (mp, cg, ebo._SEGMENT_REDUCE, ebo._PPDD_ON, ebo._PPDD_CHAIN, ebo._PPDD_UIH_POST_ID, ebo._IX_ON,
         ebo._EX2B_ON, env) = self._g
        ebo._MERGE_PAIRS[:], ebo._CAT_GROUP[:] = mp, cg
        if env is None:
            os.environ.pop(ebo._EC_GROUP_ENV, None)
        else:
            os.environ[ebo._EC_GROUP_ENV] = env
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        pd.reset_for_testing()

    def _call(self, on: bool) -> Dict[str, Any]:
        ebo._PPDD_ON = on
        names = ["item", "art", "alb", "ctx", "c_item", "c_art", "c_alb"]
        B, U = 3, 50
        L = torch.tensor([[4, 2, 3], [4, 2, 3], [4, 2, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

        class _F:
            def lengths(self) -> torch.Tensor:
                return L.reshape(-1)

            def stride(self) -> int:
                return B

        lpk = [int(r.sum()) for r in L]
        offs = [0]
        for n in lpk[:-1]:
            offs.append(offs[-1] + n)
        ri = torch.arange(sum(lpk)) % U
        emb = torch.randn(U, 512, requires_grad=True)
        return ebo._merged_construct(emb, _F(), names, ri, torch.bfloat16, lpk, offs, ["ctx", "c_item", "c_art", "c_alb"])

    def test_tags_item_member_and_big_only_when_armed(self) -> None:
        _arm()
        out = self._call(True)
        rec = getattr(out["item"].values(), pd.PPDD_ATTR, None)
        self.assertIsInstance(rec, pd.PPDDRecord)
        big = getattr(out["art"].values(), ebo.CATFUSED_ATTR)[0]
        self.assertIs(getattr(big, pd.PPDD_ATTR, None), rec)
        self.assertEqual(rec.T, out["item"].values().shape[0])
        self.assertEqual(rec.idx_cat.numel(), 2 * rec.T)
        for k in ("ctx", "art", "alb"):
            self.assertIsNone(getattr(out[k].values(), pd.PPDD_ATTR, None), k)
        # the record's indices reproduce the member outputs
        self.assertTrue(torch.equal(rec.embeddings.detach().to(torch.bfloat16)[rec.idx_item], out["item"].values()))
        self.assertTrue(torch.equal(rec.embeddings.detach().to(torch.bfloat16)[rec.idx_cat].view(rec.T, 1024), big))

    def test_knob_off_tags_nothing(self) -> None:
        out = self._call(False)
        for k, jt in out.items():
            self.assertIsNone(getattr(jt.values(), pd.PPDD_ATTR, None), k)

    def test_unarmed_prints_not_armed_once_and_tags_nothing(self) -> None:
        pd.reset_for_testing()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self._call(True)
            self._call(True)
        self.assertEqual(buf.getvalue().count(pd.NOT_ARMED_MARKER), 1)
        self.assertIsNone(getattr(out["item"].values(), pd.PPDD_ATTR, None))

    def test_unregistered_prints_ignored(self) -> None:
        pd.reset_for_testing()
        ebo._PPDD_CHAIN = None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._call(True)
        self.assertIn(pd.IGNORED_MARKER, buf.getvalue())
        ebo.note_preproc_dedup_chain(True, False, True, True, True, "item")
        self.assertEqual(ebo._PPDD_CHAIN, "fuse_ln_residual off")
        ebo.note_preproc_dedup_chain(True, True, True, True, False, "item")
        self.assertEqual(ebo._PPDD_CHAIN, "transducer_bf16_dtype off")


# ------------------------------------------------------------------------------------------------ declines etc.
class DeclineTest(unittest.TestCase):
    def setUp(self) -> None:
        _arm()
        self.pre = _pre()
        self.U, self.T = 300, 3_000
        self.emb = torch.randn(self.U, 512, requires_grad=True)
        self.item, self.cat = _batch("heavy", self.T, self.U)

    def _run(self, rec: Optional[pd.PPDDRecord] = None, big_rec: Any = "same", autocast: bool = True
             ) -> Tuple[Optional[torch.Tensor], str]:
        rec = rec if rec is not None else _record(self.emb, self.item, self.cat)
        x, big = _tokens(self.emb, self.item, self.cat, rec)
        if big_rec != "same":
            setattr(big, pd.PPDD_ATTR, big_rec)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            out = pd.run(self.pre, x, big)
        return out, buf.getvalue()

    def test_eval_no_grad_dedups_without_hilo(self) -> None:
        rows: List[int] = []
        h = self.pre._content_embedding_mlp.register_forward_pre_hook(lambda m, a: rows.append(a[0].shape[0]))
        with torch.no_grad():
            out, _ = self._run()
        h.remove()
        self.assertEqual(rows, [pd.pad_rows(self.U, self.T)])  # eval: M_pad rows, no hi / lo copies
        with torch.no_grad():
            x, big = _tokens(self.emb, self.item, self.cat, None)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                ref = _per_token(self.pre, x, big)
        self.assertIsNotNone(out)
        self.assertLessEqual(_ulps(out, ref), 2)
        self.assertEqual((pd.COUNTS["eval"], pd.COUNTS["eval_dedup"], pd.COUNTS["train"]), (1, 1, 0))

    def test_missing_or_foreign_tag_on_big(self) -> None:
        out, log = self._run(big_rec=None)
        self.assertIsNone(out)
        self.assertIn(pd.DECLINED_MARKER + " at run (tag)", log)
        out, _ = self._run(big_rec=_record(self.emb, self.item, self.cat))
        self.assertIsNone(out)

    def test_autocast_off(self) -> None:
        out, log = self._run(autocast=False)
        self.assertIsNone(out)
        self.assertIn("(autocast)", log)

    def test_mlp_not_fused_container(self) -> None:
        mlp = self.pre._content_embedding_mlp
        self.pre._content_embedding_mlp = torch.nn.Sequential(*list(mlp))
        out, log = self._run()
        self.assertIsNone(out)
        self.assertIn("(mlp)", log)

    def test_saved_tensor_hooks(self) -> None:
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            out, log = self._run()
        self.assertIsNone(out)
        self.assertIn("(hooks)", log)

    def test_snr_probe(self) -> None:
        from generative_recommenders.ops.triton import preproc_fp8_snr

        old = preproc_fp8_snr.ENABLED
        preproc_fp8_snr.ENABLED = True
        try:
            out, log = self._run()
        finally:
            preproc_fp8_snr.ENABLED = old
        self.assertIsNone(out)
        self.assertIn("(snr)", log)

    def test_fx_tracing_is_silent(self) -> None:
        from generative_recommenders.modules import preprocessors as pp

        x, _ = _tokens(self.emb, self.item, self.cat, _record(self.emb, self.item, self.cat))
        from unittest import mock

        buf = io.StringIO()
        with mock.patch.object(pp, "is_fx_tracing", lambda: True), \
                mock.patch("torch.fx._symbolic_trace.is_fx_tracing", lambda: True), contextlib.redirect_stdout(buf):
            self.assertIsNone(self.pre._preproc_dedup(x, {}))
            self.assertIsNone(pd.run(self.pre, x, None))
        self.assertEqual(buf.getvalue(), "")
        self.assertTrue(hasattr(pp, "_catfused_base"))

    def test_key_overflow_and_empty_decline_at_ec(self) -> None:
        emb = torch.empty((pd.KEY_U_MAX, 512)).as_strided((pd.KEY_U_MAX, 512), (512, 1))
        why = pd._tag_decline(emb, torch.bfloat16, [self.item, self.cat], 1, 0, 512)
        self.assertEqual(why[0] if why else None, "size")
        why = pd._tag_decline(self.emb, torch.bfloat16, [self.item[:0], self.cat[:0]], 1, 0, 512)
        self.assertEqual(why[0] if why else None, "empty")
        why = pd._tag_decline(self.emb, torch.bfloat16, [self.item, self.cat[:-2]], 1, 0, 512)
        self.assertEqual(why[0] if why else None, "idx")

    def test_disabled_latches_and_later_calls_run_todays(self) -> None:
        orig = pd._run_core
        pd._run_core = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected"))  # type: ignore
        try:
            out, log = self._run()
        finally:
            pd._run_core = orig
        self.assertIsNone(out)
        self.assertIn(pd.DISABLED_MARKER + " (FAILED CLOSED) at run: RuntimeError: injected", log)
        out, log = self._run()
        self.assertIsNone(out)
        self.assertEqual(log, "")
        self.assertEqual(pd.COUNTS["disabled"], 1)

    def test_indexed_x_ledgers_counted_as_consumed(self) -> None:
        from generative_recommenders.ops import ec_indexed_l1 as ixp

        leds = [ixp.Ledger("item"), ixp.Ledger("cat")]
        out, _ = self._run(rec=_record(self.emb, self.item, self.cat, leds))
        self.assertIsNotNone(out)
        self.assertEqual([led.fwd for led in leds], [1, 1])
        # the EC check is then silent for unconsumed (gradient-free) members
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ixp.ec_backward_check([(0, leds[0]), (1, leds[1])], [None, None])
        self.assertNotIn("not consumed", buf.getvalue())


# ------------------------------------------------------------------------------------------------ knob / pins
_CHILD = r"""
import os, sys, io, contextlib, torch
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
    from generative_recommenders.modules import preprocessors as pp
    ebo._MERGE_PAIRS[:] = [("item", "c_item")]; ebo._CAT_GROUP[:] = ["art", "alb"]
    ebo.note_preproc_dedup_chain(True, True, True, True, True, "item")
    for _ in range(2):
        ebo._ppdd_tag(torch.zeros(4, 512), torch.bfloat16, [True, True], [torch.zeros(2, dtype=torch.int64),
                      torch.zeros(4, dtype=torch.int64)], 1, 512, ["item"], [True], {}, {}, None)
    os.environ["DLRM_HSTU_FP8_PREPROC_DEDUP"] = "0" if os.environ.get("DLRM_HSTU_FP8_PREPROC_DEDUP") == "1" else "1"
    ebo._ppdd_tag(torch.zeros(4, 512), torch.bfloat16, [True, True], [torch.zeros(2, dtype=torch.int64),
                  torch.zeros(4, dtype=torch.int64)], 1, 512, ["item"], [True], {}, {}, None)
print("IMPORTED" if "generative_recommenders.ops.preproc_dedup" in sys.modules else "NOT_IMPORTED")
print("LOG:" + buf.getvalue().replace("\n", "|"))
"""


def _child(env_val: Optional[str]) -> Tuple[str, str]:
    env = dict(os.environ)
    env.pop("DLRM_HSTU_FP8_PREPROC_DEDUP", None)
    if env_val is not None:
        env["DLRM_HSTU_FP8_PREPROC_DEDUP"] = env_val
    r = subprocess.run([sys.executable, "-B", "-c", _CHILD], env=env, capture_output=True, text=True, timeout=600)
    lines = r.stdout.strip().splitlines()
    return (lines[-2] if len(lines) >= 2 else r.stdout + r.stderr), (lines[-1] if lines else "")


# ------------------------------------------------------------------------------------------------ KS-K1 (WBND)
class UntaggedTest(unittest.TestCase):
    """KS-K1: an armed, eager preprocessor call with T > 0 and no EC tag runs today's statements and is counted
    (untagged / eval_untagged on the exit line; the kit requires untagged == 0). T = 0 is neither tagged nor
    counted, and the EC's empty-batch decline is silent."""

    def setUp(self) -> None:
        _arm()
        self.pre = _pre()
        self.U, self.T = 300, 3_000
        self.emb = torch.randn(self.U, 512)
        self.item, self.cat = _batch("heavy", self.T, self.U)

    def tearDown(self) -> None:
        pd.reset_for_testing()

    def _untagged(self, T: int) -> Optional[torch.Tensor]:
        x, big = _tokens(self.emb, self.item[:T], self.cat[:2 * T], None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self.pre._preproc_dedup(x, {"art": big[:, :512], "alb": big[:, 512:]})
        self.assertEqual(buf.getvalue(), "")
        return out

    def _fields(self) -> Tuple[int, int]:
        m = re.search(pd.COUNTERS_RX, pd.counters_line())
        self.assertIsNotNone(m)
        return int(m.group(11)), int(m.group(12))

    def test_untagged_train_and_eval_calls_are_counted(self) -> None:
        self.assertIsNone(self._untagged(self.T))
        with torch.no_grad():
            self.assertIsNone(self._untagged(self.T))
            self.assertIsNone(self._untagged(self.T))
        self.assertEqual((pd.COUNTS["untagged"], pd.COUNTS["eval_untagged"]), (1, 2))
        self.assertEqual(self._fields(), (1, 2))
        self.assertEqual((pd.COUNTS["train"], pd.COUNTS["dedup"]), (0, 0))  # run() never saw these calls

    def test_tagged_calls_do_not_count_untagged(self) -> None:
        x, big = _tokens(self.emb, self.item, self.cat, _record(self.emb, self.item, self.cat))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertIsNotNone(pd.run(self.pre, x, big))
        self.assertEqual((pd.COUNTS["dedup"], pd.COUNTS["untagged"], pd.COUNTS["eval_untagged"]), (1, 0, 0))
        self.assertEqual(self._fields(), (0, 0))

    def test_empty_unarmed_and_traced_calls_are_not_counted(self) -> None:
        from unittest import mock

        self.assertIsNone(self._untagged(0))  # B = 0 on this rank: nothing to dedup
        with mock.patch("torch.fx._symbolic_trace.is_fx_tracing", lambda: True):
            pd.note_untagged(_tokens(self.emb, self.item, self.cat, None)[0])
        for state in ("unset", "not_armed", "ignored", "disabled"):
            pd._STATE = state
            self.assertIsNone(self._untagged(self.T))
        self.assertEqual((pd.COUNTS["untagged"], pd.COUNTS["eval_untagged"]), (0, 0))

    def test_module_not_loaded_is_not_imported(self) -> None:
        name = "generative_recommenders.ops.preproc_dedup"
        saved = sys.modules.pop(name)
        try:  # knob off: the module is not loaded and an untagged call imports nothing
            self.assertIsNone(self._untagged(self.T))
            self.assertNotIn(name, sys.modules)
        finally:
            sys.modules[name] = saved
        self.assertEqual(pd.COUNTS["untagged"], 0)

    def test_empty_batch_ec_decline_is_silent(self) -> None:
        item, cat = self.item[:0], self.cat[:0]
        values = torch.empty(0, 512, dtype=torch.bfloat16)
        tail = (["art", "alb"], None, "", "item", ebo._expand, ebo._CastExpandFp32Bwd.apply)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pd._tag(self.emb, torch.bfloat16, [True, True], [item, cat], 1, 512, ["item"], [True], {"item": values},
                    {}, *tail)
        self.assertEqual(buf.getvalue(), "")
        self.assertIsNone(getattr(values, pd.PPDD_ATTR, None))
        self.assertEqual(pd._tag_decline(self.emb, torch.bfloat16, [item, cat], 1, 0, 512)[0], "empty")
        with contextlib.redirect_stdout(buf):  # the other EC declines still print (once per class)
            pd._tag(self.emb, torch.bfloat16, [True, True], [self.item, self.cat[:-2]], 1, 512, ["item"], [True],
                    {"item": values}, {}, *tail)
        self.assertIn(pd.DECLINED_MARKER + " at ec (idx)", buf.getvalue())


class KnobTest(unittest.TestCase):
    def test_knob_off_values_never_import_nor_print(self) -> None:
        for v in (None, "0", "true", "on", " 2"):
            imported, log = _child(v)
            self.assertEqual(imported, "NOT_IMPORTED", v)  # read once: the flip to "1" inside is not seen
            self.assertNotIn("[preproc-dedup]", log, v)

    def test_knob_on_imports_and_is_read_once(self) -> None:
        imported, log = _child("1")
        self.assertEqual(imported, "IMPORTED")
        self.assertEqual(log.count("[preproc-dedup] NOT armed"), 1)  # unarmed (no prebuild), once

    def test_prebuild_on_cpu_prints_armed_line_matching_rx(self) -> None:
        pd.reset_for_testing()
        pd._DEVICES = ("cuda", "cpu")
        saved = (ebo._PPDD_CHAIN, ebo._PPDD_UIH_POST_ID)
        try:
            ebo._PPDD_CHAIN = None
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertFalse(pd.prebuild(CPU, classes=()))
            self.assertIn(pd.IGNORED_MARKER + " (NOT armed): make_model did not register", buf.getvalue())
            pd.reset_for_testing()
            pd._DEVICES = ("cuda", "cpu")
            ebo._PPDD_CHAIN = ""
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertTrue(pd.prebuild(CPU, classes=()))
            line = [ln for ln in buf.getvalue().splitlines() if ln.strip()][-1]
            print(f"[markers] {line}")
            self.assertRegex(line, pd.ARMED_RX)
            self.assertRegex(line, r" \[rank \d+\]$")
            self.assertTrue(line.startswith(pd.ARMED_HEAD))
            self.assertIn(("cpu", -1), pd._ARMED_DEVS)
            # a failing self-check leaves it NOT armed
            pd.reset_for_testing()
            pd._DEVICES = ("cuda", "cpu")
            orig = pd.self_check
            pd.self_check = lambda d: (_ for _ in ()).throw(RuntimeError("bad"))  # type: ignore
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    self.assertFalse(pd.prebuild(CPU, classes=()))
            finally:
                pd.self_check = orig
            self.assertIn(pd.NOT_ARMED_MARKER + ": pre-run_start warm-up / self-check failed", buf.getvalue())
            # device gate: cpu not allowed in production
            pd.reset_for_testing()
            pd._DEVICES = ("cuda",)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertFalse(pd.prebuild(CPU, classes=()))
            self.assertIn(pd.IGNORED_MARKER, buf.getvalue())
        finally:
            ebo._PPDD_CHAIN, ebo._PPDD_UIH_POST_ID = saved
            pd.reset_for_testing()
            import atexit

            atexit.unregister(pd._at_exit)  # this test armed a CPU "device"; no exit line from the test process
            pd._EXIT_REGISTERED = False

    def test_prebuild_imports_no_flydsl_before_the_fork(self) -> None:
        # The pre-RUN_START hook runs before the DataLoader pools fork: no FlyDSL import (child process), and the
        # warm-up never runs a backward through _LinearNormFunction (its LN backward is the FlyDSL kernel).
        import ast
        import inspect

        code = (
            "import sys, torch\n"
            "from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo\n"
            "from generative_recommenders.ops import preproc_dedup as pd\n"
            "before = sorted(m for m in sys.modules if 'flydsl' in m)\n"
            "pd._DEVICES = ('cuda', 'cpu'); ebo._PPDD_CHAIN = ''\n"
            "assert pd.prebuild(torch.device('cpu'))\n"
            "print('FLYDSL', before, sorted(m for m in sys.modules if 'flydsl' in m))\n"
        )
        r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=600)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("FLYDSL")]
        self.assertEqual(line, ["FLYDSL [] []"], r.stdout[-1500:] + r.stderr[-1500:])
        for fn in (pd.warmup, pd._warm_class, pd.prebuild, pd._x_combine_check):
            calls = [n for n in ast.walk(ast.parse(inspect.getsource(fn).strip()))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "backward"]
            self.assertEqual(calls, [], fn.__name__)
            imports = [n for n in ast.walk(ast.parse(inspect.getsource(fn).strip()))
                       if isinstance(n, (ast.Import, ast.ImportFrom))]
            mods = [getattr(n, "module", None) or "" for n in imports] + [a.name for n in imports for a in n.names]
            self.assertFalse(any("flydsl" in m or m == "ln_bwd" for m in mods), (fn.__name__, mods))

    def test_self_check_exact_on_cpu(self) -> None:
        s = pd.self_check(CPU)
        self.assertIn("construction == host reference", s)
        self.assertIn("hi + lo exact", s)


class PinsTest(unittest.TestCase):
    def _gin(self) -> str:
        with open(_GIN) as f:
            return f.read()

    def test_gin_block_placement_and_bindings(self) -> None:
        txt = self._gin()
        i101 = txt.index("# (101) GSPX")
        i118 = txt.index("# (118) PPDD:")
        i39 = re.search(r"^# \(39\) ", txt, re.M).start()  # the next item's header
        self.assertLess(i101, i118)
        self.assertLess(i118, i39)
        block = txt[i118:i39]
        self.assertIn("\napply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP = True\n#\n", block)
        self.assertNotIn("make_model.preproc_dedup", txt)
        self.assertNotIn('"PREPROC_DEDUP"', txt)  # the old env_int key does not come back
        # the (118) binding + its PADSPREAD sub-knob binding (2026-10-24)
        self.assertEqual([ln for ln in block.splitlines() if ln and not ln.startswith("#")],
                         ["apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP = True",
                          "apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP_PADSPREAD = True",
                          "apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP_HILO2 = True",
                          "apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP_LEAN = True"])
        self.assertEqual(txt.count("apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP ="), 1)
        self.assertEqual(txt.count("apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP_PADSPREAD ="), 1)
        self.assertEqual(txt.count("apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_DEDUP_LEAN ="), 1)
        self.assertIn("Kill switch: DLRM_HSTU_FP8_PREPROC_DEDUP_PADSPREAD=0", block)
        self.assertIn("Kill switch: DLRM_HSTU_FP8_PREPROC_DEDUP_LEAN=0", block)
        for s in ("Kill switch: DLRM_HSTU_FP8_PREPROC_DEDUP=0", "DECLARED PRECISION", "REDUCTION-ORDER",
                  "[preproc-dedup] DISABLED", "[preproc-dedup] declined", "[preproc-dedup] NOT armed",
                  "[preproc-dedup] OVERFLOW"):
            self.assertIn(s, block.replace("#     ", "").replace("\n", " ").replace("[preproc- dedup]",
                                                                                     "[preproc-dedup]"))

    def test_bootstrap_bind_and_preset_env_wins(self) -> None:
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb
        import inspect

        src = inspect.getsource(eb)
        self.assertIn('_bind_bool("DLRM_HSTU_FP8_PREPROC_DEDUP", DLRM_HSTU_FP8_PREPROC_DEDUP)', src)
        x2lb = src.index('_bind_bool("DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD"')
        self.assertLess(x2lb, src.index('_bind_bool("DLRM_HSTU_FP8_PREPROC_DEDUP"'))
        old = os.environ.pop("DLRM_HSTU_FP8_PREPROC_DEDUP", None)
        try:
            eb._bind_bool("DLRM_HSTU_FP8_PREPROC_DEDUP", True)
            self.assertEqual(os.environ["DLRM_HSTU_FP8_PREPROC_DEDUP"], "1")
            os.environ["DLRM_HSTU_FP8_PREPROC_DEDUP"] = "0"  # kill switch pre-set: wins
            eb._bind_bool("DLRM_HSTU_FP8_PREPROC_DEDUP", True)
            self.assertEqual(os.environ["DLRM_HSTU_FP8_PREPROC_DEDUP"], "0")
        finally:
            os.environ.pop("DLRM_HSTU_FP8_PREPROC_DEDUP", None)
            if old is not None:
                os.environ["DLRM_HSTU_FP8_PREPROC_DEDUP"] = old
        self.assertIn("DLRM_HSTU_FP8_PREPROC_DEDUP", inspect.signature(eb.apply_env_bootstrap).parameters)

    def test_make_model_param_and_registration(self) -> None:
        import ast

        path = os.path.join(_ROOT, "generative_recommenders", "dlrm_v4", "train", "utils.py")
        with open(path) as f:
            src = f.read()  # parsed, not imported (utils imports torchrec)
        fn = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "make_model"][0]
        names = [a.arg for a in fn.args.args]
        self.assertNotIn("preproc_dedup", names)  # one knob, no make_model name (the reviewer's R8, as ruled)
        body = ast.get_source_segment(src, fn)
        self.assertIn("if _ebo_ppdd.ppdd_enabled():\n        _ebo_ppdd.note_preproc_dedup_chain(", body)
        self.assertLess(body.index("enable_ec_cat_fuse("), body.index("note_preproc_dedup_chain("))
        self.assertLess(body.index("note_preproc_dedup_chain("), body.index("note_ec_indexed_l1_chain("))

    def test_clean_exit_hook(self) -> None:
        from generative_recommenders.dlrm_v4.train import clean_exit as ce

        self.assertIn(("generative_recommenders.ops.preproc_dedup", "_at_exit"), ce.EXIT_HOOKS)
        hooks = [h for h in ce.EXIT_HOOKS if h[0] == "generative_recommenders.ops.preproc_dedup"]
        self.assertEqual(len(hooks), 1)
        self.assertTrue(callable(getattr(pd, hooks[0][1])))

    def test_blas_warmup_hook(self) -> None:
        # The pre-RUN_START hook: after the Indexed-X hook in run_blas_warmup, knob-gated (a literal copy of the
        # env name; not "1": returns before importing / printing anything), and it calls preproc_dedup.prebuild.
        import inspect

        from generative_recommenders.dlrm_v4.train import blas_warmup as bw

        self.assertEqual(bw._PPDD_ENV, pd.ENV_VAR)
        src = inspect.getsource(bw.run_blas_warmup)
        self.assertLess(src.index("_prebuild_ec_indexed_l1(device)"), src.index("_prebuild_preproc_dedup(device)"))
        self.assertIn("ppdd.prebuild(device)", inspect.getsource(bw._prebuild_preproc_dedup))
        old = os.environ.pop(pd.ENV_VAR, None)
        calls: List[Any] = []
        orig = pd.prebuild
        pd.prebuild = lambda d, *a, **k: calls.append(d) or False  # type: ignore[assignment]
        try:
            for v in (None, "0", "true", "on"):
                if v is not None:
                    os.environ[pd.ENV_VAR] = v
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    bw._prebuild_preproc_dedup(CPU)
                self.assertEqual((calls, buf.getvalue()), ([], ""), v)
            os.environ[pd.ENV_VAR] = "1"
            bw._prebuild_preproc_dedup(CPU)
            self.assertEqual(calls, [CPU])
        finally:
            pd.prebuild = orig  # type: ignore[assignment]
            os.environ.pop(pd.ENV_VAR, None)
            if old is not None:
                os.environ[pd.ENV_VAR] = old

    def test_literal_copies(self) -> None:
        import inspect

        from generative_recommenders.modules import preprocessors as pp

        self.assertEqual(ebo._PPDD_ENV, pd.ENV_VAR)
        self.assertEqual(ebo.CATFUSED_ATTR, pd.CATFUSED_ATTR)
        self.assertIn(f'"{pd.PPDD_ATTR}"', inspect.getsource(pp.ContextualPreprocessor._preproc_dedup))

    def test_lane_c_interface(self) -> None:
        self.assertEqual(pd.ENV_VAR, "DLRM_HSTU_FP8_PREPROC_DEDUP")
        self.assertEqual(pd.TAG, "[preproc-dedup]")
        self.assertEqual(pd.ARMED_MARKER, "armed OK: [preproc-dedup] ARMED preproc_dedup (118)")
        self.assertEqual(pd.m_pad(95_000), 95_232)
        self.assertEqual(pd.m_pad(95_000, 1_000), 1_024)
        keys = torch.tensor([5, 3, 5, 9, 3, 3], dtype=torch.int64)
        inv, order, offsets, mp, n_u, first = pd.build_unique(keys, 10)
        self.assertEqual(mp, 256)
        self.assertEqual(inv.tolist(), [1, 0, 1, 2, 0, 0])
        self.assertEqual(order.tolist(), [1, 4, 5, 0, 2, 3])
        self.assertEqual(offsets[:5].tolist(), [0, 3, 5, 6, 6])
        self.assertEqual(int(n_u[0]), 3)
        self.assertEqual(keys[first[:3]].tolist(), [3, 5, 9])
        for v, want in ((None, False), ("0", False), ("true", False), (" 1 ", True), ("1", True)):
            old = os.environ.pop(pd.ENV_VAR, None)
            try:
                if v is not None:
                    os.environ[pd.ENV_VAR] = v
                pd._KNOB, ebo._PPDD_ON = None, None
                self.assertEqual((pd.enabled(), ebo.ppdd_enabled()), (want, want), v)
                os.environ[pd.ENV_VAR] = "0" if want else "1"  # read once
                self.assertEqual((pd.enabled(), ebo.ppdd_enabled()), (want, want), v)
            finally:
                os.environ.pop(pd.ENV_VAR, None)
                if old is not None:
                    os.environ[pd.ENV_VAR] = old
                pd._KNOB, ebo._PPDD_ON = None, None

    def test_markers_json_matches_module(self) -> None:
        path = os.environ.get("PPDD_MARKERS_JSON")
        if not path:
            self.skipTest("PPDD_MARKERS_JSON not set")
        with open(path) as f:
            m = json.load(f)
        row = m["rows"]["PDEDUP"]  # main's ladder row id (SCOPE 6.4 called it PPDD)
        self.assertEqual(row["per_rank"][0]["rx"], pd.ARMED_RX)
        self.assertEqual(m["kit_maintenance"]["armed_rx"], pd.ARMED_RX)
        # every must-be-absent text of the module is caught by a test_absent rx, as the kit matches them
        for marker in (pd.DISABLED_MARKER, pd.DECLINED_MARKER, pd.NOT_ARMED_MARKER, pd.IGNORED_MARKER,
                       pd.OVERFLOW_MARKER):
            self.assertTrue(any(re.search(rx, marker) for rx in row["test_absent"]), marker)
        self.assertTrue(any(re.search(rx, pd.ARMED_MARKER) for rx in row["ctl_absent"]))
        self.assertEqual(row["info_count"][0]["rx"], pd.COUNTERS_RX)
        gate = row["count_ge_R"][0]["rx"]  # the exit-line gate == kit_maintenance.engaged_rx
        self.assertEqual(m["kit_maintenance"]["engaged_rx"], gate)
        _arm()
        try:
            pd.COUNTS.update(train=3, dedup=3, eval=1, eval_dedup=1, eval_untagged=2)
            pd._MAX.update(U=9, T=9, n_u=9)
            line = pd.counters_line()
            self.assertRegex(line, gate)
            self.assertRegex(line, pd.COUNTERS_RX)
            self.assertFalse([rx for rx in row["test_absent"] if re.search(rx, line)])
            pd.COUNTS["untagged"] = 1  # KS-K1: a per-token training call fails the gate and hits test_absent
            line = pd.counters_line()
            self.assertIsNone(re.search(gate, line))
            self.assertTrue([rx for rx in row["test_absent"] if re.search(rx, line)])
        finally:
            pd.reset_for_testing()


class ReviewerItemsTest(unittest.TestCase):
    """The independent reviewer's pre-review items with a CPU check (R3, R4, R5, R7); the rest are stated in the
    row notes."""

    def _full(self, pre: Any, x: torch.Tensor, big: torch.Tensor, T: int) -> torch.Tensor:
        k_in = pre._action_embedding_mlp[0].in_features
        x_act = torch.arange(T * k_in, dtype=torch.float32).view(T, k_in).remainder(7).sub(3).div(4)

        class _Enc(torch.nn.Module):
            def forward(self, **kw: Any) -> torch.Tensor:
                return x_act

        pre._action_encoder = _Enc()
        return PreprocessorWiringTest._forward(self, pre, x, big, T)  # type: ignore[arg-type]

    def test_r3_fp8_l2_and_grouped_tail_never_touched_heavy_duplicates(self) -> None:
        # Main's ruling 2: under dedup the c / a backward never reaches E-M3a's FP8 dh2 path or its scale window,
        # and no grouped tail is begun, even with E-M3a and the grouped tail reported ON.
        from unittest import mock

        from generative_recommenders.modules import preprocessors as pp
        from generative_recommenders.ops import preproc_fp8_l2 as pfl
        from generative_recommenders.ops import preproc_grouped_tail as pgt

        _ensure_cumsum()
        _arm()

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("FP8 L2 / grouped-tail path touched under dedup")

        pre = _pre(action=True)
        U, T = 200, 30_000  # ~300 tokens per triple: summed dh would be far above the per-token amax
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U)
        before = pfl.site_diagnostics()
        with mock.patch.multiple(pfl, run_head=boom, grouped_tail_apply=boom, _site=boom, _quantize_w2=boom,
                                 _grouped_bwd=boom, _wgrad_nn_b=boom, enabled=lambda: True), \
                mock.patch.object(pfl._LinearSwishLNEmitFunction, "apply", boom), \
                mock.patch.object(pfl._GroupedLinearLNTailFP8Function, "apply", boom), \
                mock.patch.object(pgt, "begin_grouped_tail", boom), mock.patch.object(pp, "_PREPROC_GROUPED_TAIL", True):
            x, big = _tokens(emb, item, cat, _record(emb, item, cat))
            out = self._full(pre, x, big, T)
            out.float().square().mean().backward()
        self.assertEqual(pd.COUNTS["dedup"], 1)
        self.assertEqual(pfl.site_diagnostics(), before)  # no dh2.0 / dh2.1 (or a1) site created or fed
        self.assertIsNotNone(emb.grad)

    def _coherent(self) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = _pre()
        torch.manual_seed(21)
        U, T = 300, 40_000
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("heavy", T, U, seed=8)
        base = torch.randn(U, 512)
        g = (base[item] * (1 + 0.01 * torch.randn(T, 512))).to(torch.bfloat16).float()  # near-equal dy per triple
        return pre, emb, item, cat, g

    GRAD_TOL_INCOHERENT = 2.0  # new <= max(2 x old, 1e-3) (the earlier prototype's criterion), random dy
    # coherent dy: main's ruling, report above GRAD_REPORT_RATIO, fail above GRAD_HARD_RATIO (module constants)

    def test_r4_coherent_gradients_tolerance(self) -> None:
        pre, emb, item, cat, g = self._coherent()
        bt = BackwardTest()
        ref = bt._grads(pre, emb, item, cat, g, "fp64")
        old = bt._grads(pre, emb, item, cat, g, "token")
        new = bt._grads(pre, emb, item, cat, g, "dedup")
        worst_ratio = 0.0
        for k in sorted(ref):
            e_old, e_new = _rel(old[k], ref[k]), _rel(new[k], ref[k])
            ratio = e_new / e_old if e_old > 1e-9 else float("nan")  # LN-bias sums can be exact per token
            if e_old > 1e-9:
                worst_ratio = max(worst_ratio, ratio)
            print(f"[R4 coherent] {k}: per-token {e_old:.2e}, dedup {e_new:.2e}, ratio {ratio:.2f}")
        print(f"[R4 coherent] worst dedup / per-token ratio {worst_ratio:.2f} (report > {GRAD_REPORT_RATIO:g}x, "
              f"fail > {GRAD_HARD_RATIO:g}x)")
        _grad_bug_catcher(self, new, old, ref, "R4 coherent")

    def test_r5_rerun_determinism_hot_key(self) -> None:
        pre = _pre()
        U, T = 400, 20_000
        emb = torch.randn(U, 512, requires_grad=True)
        item, cat = _batch("oneheavy", T, U, seed=2)  # one triple with 15,000 tokens (> the 8,192 split)
        g = torch.randn(T, 512)
        runs = [BackwardTest()._grads(pre, emb, item, cat, g, "dedup") for _ in range(2)]
        for k in runs[0]:
            self.assertTrue(torch.equal(runs[0][k], runs[1][k]), k)

    def test_r7_unclaimed_ex2b_ports_are_silent(self) -> None:
        import types

        from generative_recommenders.dlrm_v4.train import ec_dgrad_ports as xp

        ctx = types.SimpleNamespace(ex2b_records=(object(), object()))
        grads: List[Optional[torch.Tensor]] = [None, None]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            seed, out = xp.ec_backward(ctx, grads, [None, None])
        self.assertIsNone(seed)
        self.assertIs(out, grads)
        self.assertEqual(buf.getvalue(), "")


@unittest.skipUnless(os.environ.get("PPDD_BASE_TREE"), "PPDD_BASE_TREE (an unpatched tree) not set")
class KnobOffEqualsUnpatchedTest(unittest.TestCase):
    """Knob unset: the patched preprocessor's forward / backward == the unpatched tree's, bitwise (children)."""

    SCRIPT = r"""
import sys, torch
sys.path.insert(0, sys.argv[1])
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from generative_recommenders.dlrm_v4.train import ec_bf16_output as ebo
try:
    torch.ops.fbgemm.asynchronous_complete_cumsum
except (AttributeError, RuntimeError):
    lib = torch.library.Library("fbgemm", "DEF")
    lib.define("asynchronous_complete_cumsum(Tensor t) -> Tensor")
    lib.impl("asynchronous_complete_cumsum", lambda t: torch.cat([t.new_zeros(1), torch.cumsum(t, 0)]),
             "CompositeExplicitAutograd")
torch.manual_seed(0)
pre = ContextualPreprocessor(input_embedding_dim=512, hidden_dim=256, output_embedding_dim=512,
    contextual_feature_to_max_length={}, contextual_feature_to_min_uih_length={}, action_embedding_dim=8,
    action_feature_name="act", action_weights=[1, 2, 4], additional_embedding_features=["art", "alb"],
    is_inference=False, fuse_linear_norm=True, fuse_ln_residual=True)
pre.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
T, U, B = 2000, 300, 4
k_in = pre._action_embedding_mlp[0].in_features
x_act = (torch.arange(T * k_in, dtype=torch.float32).view(T, k_in) % 7 - 3) / 4


class _Enc(torch.nn.Module):  # stand-in ActionEncoder: its shipped chain needs fbgemm's jagged ops (absent here)
    def forward(self, **kw):
        return x_act


pre._action_encoder = _Enc()
emb = torch.randn(U, 512, requires_grad=True)
item = torch.randint(0, 100, (T,)); cat = torch.randint(100, 300, (2 * T,))
x = ebo._CastExpandFp32Bwd.apply(emb, item, torch.bfloat16)
big = ebo._CastExpandFp32Bwd.apply(emb, cat, torch.bfloat16).view(T, 1024)
pay = {"art": big[:, :512], "alb": big[:, 512:], "act": torch.arange(T) % 8}
setattr(pay["art"], ebo.CATFUSED_ATTR, (big, 0, 2)); setattr(pay["alb"], ebo.CATFUSED_ATTR, (big, 1, 2))
with torch.autocast("cpu", dtype=torch.bfloat16):
    o = pre(max_uih_len=T // B, max_targets=0, total_uih_len=T, total_targets=0,
            seq_lengths=torch.full((B,), T // B), seq_timestamps=torch.zeros(T, dtype=torch.int64),
            seq_embeddings=x, num_targets=torch.zeros(B, dtype=torch.int64), seq_payloads=pay)[6]
o.float().square().sum().backward()
torch.save({"o": o, "emb": emb.grad, **{n: p.grad for n, p in pre.named_parameters()},
            "mods": sorted(k for k in sys.modules if "preproc_dedup" in k)}, sys.argv[2])
"""

    def test_bitwise(self) -> None:
        res = []
        with tempfile.TemporaryDirectory() as d:
            for i, tree in enumerate((os.environ["PPDD_BASE_TREE"], _ROOT)):
                out = os.path.join(d, f"{i}.pt")
                env = {k: v for k, v in os.environ.items() if k != pd.ENV_VAR}
                r = subprocess.run([sys.executable, "-B", "-c", self.SCRIPT, tree, out], env=env,
                                   capture_output=True, text=True, timeout=900)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertNotIn("[preproc-dedup]", r.stdout + r.stderr)
                res.append(torch.load(out))
        self.assertEqual(res[1]["mods"], [])
        for k in res[0]:
            if k != "mods":
                self.assertTrue(torch.equal(res[0][k], res[1][k]), k)


if __name__ == "__main__":
    unittest.main()

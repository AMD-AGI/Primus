"""CPU tests for the (119) ACTION_DEDUP sub-knob LEAN (ops/action_dedup.py, DLRM_HSTU_FP8_ACTION_DEDUP_LEAN).

LEAN only removes host launches; every result must be BITWISE the knob-off result:
  * the knob reads "1" only, once per process (reset_for_testing re-reads it);
  * codes(): on (seq lengths), on (the prefill's UIH lengths) and off are torch.equal int64 on random, high-bit
    and negative action masks, zero-length samples, several targets per sample, an empty UIH, B = 1, int32
    lengths and a watchtime encoder (T = 5);
  * table_inputs(): rows and the gradients to both action tables bitwise for T = 1..6;
  * segsum(): on == off bitwise for every path (f32 / bmm with an fp32 stand-in / fp32 g / fp64 g), every
    chunk boundary and N = 0;
  * CodeGatherAdd backward: the two-launch hi / lo split == cast / sub / cast / cat bitwise (bf16 table, wide
    exponent range, +-0, large values), and the same-dtype / k-row tables still take the shipped statements;
  * the whole ContextualPreprocessor (fp32 and bf16 autocast, with and without the (65) prefill carrier):
    output and every gradient bitwise on vs off;
  * the constants: cached per (kind, device, dtype, size), never data, not inference tensors, and the lean ops
    dispatch no host-reading op and strictly fewer ops (the removed launches are gone);
  * the gin binding and the bootstrap parameter (added with the (119) block).

Run: python -m pytest -q generative_recommenders/ops/tests/action_dedup_lean_test.py
"""

import os
import re
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import fbgemm_gpu  # noqa: F401  registers torch.ops.fbgemm
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from generative_recommenders.modules.action_encoder import ActionEncoder
from generative_recommenders.ops import action_dedup as ad
from generative_recommenders.ops.tests.action_dedup_cpu_test import (
    _inputs,
    _model,
    _run,
    _same,
    GIN,
)

CPU = torch.device("cpu")


def _set_lean(on: Optional[bool]) -> None:
    ad.reset_for_testing()
    ad._LEAN = on


def _enc(T: int, watch: int = 0, seed: int = 0) -> ActionEncoder:
    torch.manual_seed(seed)
    base = T - watch
    return ActionEncoder(
        action_embedding_dim=8,
        action_feature_name="a",
        action_weights=[1 << j for j in range(base)],
        watchtime_feature_name="w" if watch else "",
        watchtime_to_action_thresholds_and_weights=[(30 * (j + 1), 1 << (base + j)) for j in range(watch)]
        if watch else None,
    )


def _code_case(seed: int, B: int, max_uih: int, max_nt: int, zero_frac: float, acts: str,
               ldtype: torch.dtype = torch.int64, watch: bool = False) -> Tuple[Dict[str, torch.Tensor],
                                                                            torch.Tensor, torch.Tensor, int]:
    g = torch.Generator().manual_seed(seed)
    uih = torch.randint(0, max_uih + 1, (B,), generator=g)
    uih = torch.where(torch.rand(B, generator=g) < zero_frac, torch.zeros_like(uih), uih)
    nt = torch.randint(1, max_nt + 1, (B,), generator=g)
    n = int(uih.sum())
    if acts == "small":
        a = torch.randint(0, 8, (n,), generator=g)
    elif acts == "high":
        a = torch.randint(0, 1 << 40, (n,), generator=g)
    else:  # negative masks (two's complement high bits set)
        a = torch.randint(-(1 << 40), 1 << 40, (n,), generator=g)
    pay = {"a": a}
    if watch:
        pay["w"] = torch.randint(0, 200, (n,), generator=g)
    return pay, uih.to(ldtype), nt.to(ldtype), n + int(nt.sum())


class _Ops(TorchDispatchMode):
    BAD = ("_local_scalar_dense", "nonzero", "unique", "masked_select", "item", "equal", "is_nonzero")

    def __init__(self) -> None:
        super().__init__()
        self.seen: List[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        name = func.__name__.split(".")[0]
        self.seen.append(name)
        if any(name == b or name.startswith(b) for b in self.BAD):
            raise AssertionError(f"host-reading op {func}")
        return func(*args, **(kwargs or {}))


class LeanKnobTest(unittest.TestCase):
    def tearDown(self) -> None:
        ad.reset_for_testing()

    def test_knob_reads_only_1_once(self) -> None:
        for v, want in (("1", True), ("0", False), ("", False), ("true", False), ("on", False), (" 1 ", True)):
            ad.reset_for_testing()
            with mock.patch.dict(os.environ, {ad.LEAN_ENV: v}):
                self.assertEqual(ad.lean(), want, v)
        ad.reset_for_testing()
        with mock.patch.dict(os.environ, {ad.LEAN_ENV: "1"}):
            self.assertTrue(ad.lean())
        with mock.patch.dict(os.environ, {ad.LEAN_ENV: "0"}):
            self.assertTrue(ad.lean())  # read once per process
        ad.reset_for_testing()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ad.LEAN_ENV, None)
            self.assertFalse(ad.lean())

    def test_armed_line_names_the_sub_knob(self) -> None:
        for on in (False, True):
            _set_lean(on)
            line = ad._armed_line("bmm", "x")
            self.assertIn(f"lean={int(on)} (kill switch DLRM_HSTU_FP8_ACTION_DEDUP_LEAN=0)", line)
            self.assertIn("REDUCTION-ORDER item (119)", line)


class LeanCodesTest(unittest.TestCase):
    CASES = (
        dict(seed=1, B=48, max_uih=300, max_nt=1, zero_frac=0.0, acts="small"),
        dict(seed=2, B=48, max_uih=300, max_nt=4, zero_frac=0.2, acts="high"),
        dict(seed=3, B=256, max_uih=64, max_nt=3, zero_frac=0.5, acts="neg"),
        dict(seed=4, B=1, max_uih=50, max_nt=2, zero_frac=0.0, acts="small"),
        dict(seed=5, B=33, max_uih=40, max_nt=2, zero_frac=0.0, acts="high", ldtype=torch.int32),
        dict(seed=6, B=17, max_uih=0, max_nt=3, zero_frac=0.0, acts="small"),  # empty UIH
        dict(seed=7, B=20, max_uih=30, max_nt=2, zero_frac=0.3, acts="neg", watch=True),
    )

    def tearDown(self) -> None:
        ad.reset_for_testing()

    def test_codes_bitwise_on_vs_off(self) -> None:
        for kw in self.CASES:
            watch = bool(kw.get("watch", False))
            enc = _enc(5 if watch else 3, watch=2 if watch else 0)
            pay, uih, nt, rows = _code_case(**kw)
            seq = uih + nt
            _set_lean(False)
            off = ad.codes(enc, pay, seq, nt, rows)
            _set_lean(True)
            on_seq = ad.codes(enc, pay, seq, nt, rows)
            on_uih = ad.codes(enc, pay, None, nt, rows, uih_lengths=uih)
            again = ad.codes(enc, pay, None, nt, rows, uih_lengths=uih)  # warm constants
            for got in (on_seq, on_uih, again):
                self.assertEqual((got.dtype, got.shape), (torch.int64, (rows,)), kw)
                self.assertTrue(torch.equal(got, off), kw)

    def test_plan_bitwise_on_vs_off(self) -> None:
        m = _model(21)
        inp = _inputs(22)
        uih, nt = inp["uih"], inp["nt"]
        rows = inp["T_uih"] + inp["B"]
        _set_lean(False)
        c0, t0 = ad.plan(m, inp["ints"], uih + nt, nt, rows)
        _set_lean(True)
        c1, t1 = ad.plan(m, inp["ints"], None, nt, rows, uih_lengths=uih)
        self.assertTrue(torch.equal(c0, c1))
        self.assertTrue(_same(t0.detach(), t1.detach()))


class LeanTableTest(unittest.TestCase):
    def tearDown(self) -> None:
        ad.reset_for_testing()

    def test_table_inputs_and_grads_bitwise(self) -> None:
        for T in range(1, ad.MAX_T + 1):
            enc = _enc(T, seed=T)
            with torch.no_grad():  # signed zeros / large values in the table
                enc._action_embedding_table[0, 0] = -0.0
                enc._action_embedding_table[-1, -1] = -3.0e30
            w = torch.randn(((1 << T) + 1, 8 * T), generator=torch.Generator().manual_seed(T))
            res = []
            for on in (False, True):
                _set_lean(on)
                for _ in range(2):  # cold, then warm constants
                    enc.zero_grad(set_to_none=True)
                    e = ad.table_inputs(enc)
                    (e * w).sum().backward()
                    res.append((e.detach(), enc._action_embedding_table.grad.clone(),
                                enc._target_action_embedding_table.grad.clone()))
            for r in res[1:]:
                for a, b in zip(res[0], r):
                    self.assertTrue(_same(a, b), T)


class LeanSegsumTest(unittest.TestCase):
    ROWS = (0, 1, 7, ad.CHUNK, ad.CHUNK + 3, 3 * ad.CHUNK + 5, ad.F32_CHUNK + 9)

    def tearDown(self) -> None:
        ad.reset_for_testing()

    @staticmethod
    def _data(rows: int, k: int, dtype: torch.dtype, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        code = torch.randint(0, k, (rows,), generator=gen)
        if rows > 3:
            code[: rows // 3] = 0  # one hot code; some codes may be missing
        g = torch.randn(rows, 64, generator=gen) * torch.exp2(
            torch.randint(-30, 30, (rows, 64), generator=gen).float())
        if rows > 0:
            g[0, 0] = -0.0
        return code, g.to(dtype)

    def _both(self, code: torch.Tensor, g: torch.Tensor, k: int, path: str) -> None:
        _set_lean(False)
        off = ad.segsum(code, g, k, path)
        _set_lean(True)
        on = ad.segsum(code, g, k, path)
        warm = ad.segsum(code, g, k, path)
        tag = (g.dtype, g.shape[0], k, path)
        self.assertTrue(_same(off, on), tag)
        self.assertTrue(_same(off, warm), tag)

    def test_f32_path_and_wide_g(self) -> None:
        for k in (9, 17):
            for rows in self.ROWS:
                for dtype, path in ((torch.bfloat16, "f32"), (torch.float32, "f32"), (torch.float64, "f32"),
                                    (torch.float32, "bmm")):
                    code, g = self._data(rows, k, dtype, rows + k)
                    self._both(code, g, k, path)

    def test_bmm_path(self) -> None:
        orig = torch.bmm

        def fake_bmm(a, b, out_dtype=None):  # bmm(out_dtype=float32) is not implemented on CPU
            assert out_dtype == torch.float32
            return orig(a.float(), b.float())

        torch.bmm = fake_bmm
        try:
            for rows in self.ROWS:
                code, g = self._data(rows, 9, torch.bfloat16, rows)
                self._both(code, g, 9, "bmm")
        finally:
            torch.bmm = orig


class LeanBackwardTest(unittest.TestCase):
    def tearDown(self) -> None:
        ad.reset_for_testing()

    def _grads(self, on: bool, tab_dtype: torch.dtype, res_dtype: torch.dtype, rows_mult: int,
               seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _set_lean(on)
        gen = torch.Generator().manual_seed(seed)
        k = 9
        atab = torch.randn(rows_mult * k, 64, generator=gen).to(tab_dtype).requires_grad_(True)
        res = torch.randn(3000, 64, generator=gen).to(res_dtype).requires_grad_(True)
        code = torch.randint(0, k, (3000,), generator=gen)
        dy = torch.randn(3000, 64, generator=gen) * torch.exp2(
            torch.randint(-40, 40, (3000, 64), generator=gen).float())
        dy[0, :3] = torch.tensor([-0.0, 0.0, 3.0e30])  # finite: NaN payloads are not part of the contract
        out = ad.CodeGatherAdd.apply(res, atab, code, k)
        out.backward(dy.to(out.dtype))
        return atab.grad, res.grad

    def test_hilo_bitwise(self) -> None:
        for i, (tab, rd, mult) in enumerate(((torch.bfloat16, torch.bfloat16, 2), (torch.float32, torch.float32, 2),
                                             (torch.bfloat16, torch.bfloat16, 1), (torch.float32, torch.float32, 1),
                                             (torch.bfloat16, torch.float32, 2), (torch.float64, torch.float64, 2))):
            a0, r0 = self._grads(False, tab, rd, mult, 30 + i)
            a1, r1 = self._grads(True, tab, rd, mult, 30 + i)
            self.assertTrue(_same(a0, a1), (tab, rd, mult))
            self.assertTrue(_same(r0, r1), (tab, rd, mult))
            self.assertEqual(a1.shape[0], 9 * mult)


class LeanModelTest(unittest.TestCase):
    """The real ContextualPreprocessor, knob (119) on, LEAN off vs on: output and every gradient bitwise."""

    def tearDown(self) -> None:
        ad.reset_for_testing()
        ad._latched = None

    def _pair(self, autocast: Optional[torch.dtype], prefill: bool, **kw: object) -> None:
        m = _model(41)
        inp = _inputs(42, **kw)
        res = []
        for on in (False, True):
            ad._latched = None
            _set_lean(on)
            res.append(_run(m, inp, "1", autocast, prefill=prefill))
        self.assertIsNone(ad._latched)
        for k in res[0]:
            self.assertTrue(_same(res[0][k], res[1][k]), (k, autocast, prefill, kw))

    def test_fp32(self) -> None:
        self._pair(None, False)
        self._pair(None, True)

    def test_bf16_autocast(self) -> None:
        self._pair(torch.bfloat16, False)
        self._pair(torch.bfloat16, True)

    def test_edges(self) -> None:
        self._pair(torch.bfloat16, True, uih_zero=True)
        self._pair(torch.bfloat16, True, action_set=[5])
        self._pair(None, True, B=1)


class LeanConstTest(unittest.TestCase):
    def tearDown(self) -> None:
        ad.reset_for_testing()

    def test_keys_values_and_no_data(self) -> None:
        _set_lean(True)
        enc = _enc(3)
        for seed in (1, 2, 3):
            pay, uih, nt, rows = _code_case(seed, 30, 40, 2, 0.1, "high")
            ad.codes(enc, pay, None, nt, rows, uih_lengths=uih)
            ad.table_inputs(enc)
        keys = set(ad._CONST)
        self.assertEqual(keys, {("pow2", CPU, torch.int64, 3), ("bits", CPU, torch.bool, 3)})
        p = ad._CONST[("pow2", CPU, torch.int64, 3)]
        self.assertTrue(torch.equal(p, torch.tensor([1, 2, 4])))
        b = ad._CONST[("bits", CPU, torch.bool, 3)]
        want = torch.tensor([[(c >> j) & 1 == 1 for j in range(3)] for c in range(8)])
        self.assertTrue(torch.equal(b, want))
        pay5, uih5, nt5, rows5 = _code_case(9, 10, 20, 2, 0.0, "small", watch=True)
        ad.codes(_enc(5, watch=2), pay5, None, nt5, rows5, uih_lengths=uih5)
        self.assertIn(("pow2", CPU, torch.int64, 5), ad._CONST)  # T is part of the key
        self.assertIs(ad._const("pow2", CPU, torch.int64, 3), p)  # reused, not rebuilt
        ar = ad._const("arange", CPU, torch.int32, 16)
        self.assertEqual((ar.dtype, ar.tolist()), (torch.int32, list(range(16))))

    def test_not_inference_tensors(self) -> None:
        _set_lean(True)
        with torch.inference_mode():
            t = ad._const("bits", CPU, torch.bool, 3)
        self.assertFalse(t.is_inference())
        enc = _enc(3)
        ad.table_inputs(enc).sum().backward()  # mul saves the cached bits for backward
        self.assertIsNotNone(enc._action_embedding_table.grad)

    def test_fewer_ops_and_no_host_read(self) -> None:
        enc = _enc(3)
        pay, uih, nt, rows = _code_case(11, 64, 50, 2, 0.1, "high")
        gen = torch.Generator().manual_seed(12)
        code = torch.randint(0, 9, (5000,), generator=gen)
        g32 = torch.randn(5000, 64, generator=gen)
        seen: Dict[bool, Dict[str, List[str]]] = {}
        for on in (False, True):
            _set_lean(on)
            ad.codes(enc, pay, uih + nt, nt, rows)  # warm the constants
            ad.table_inputs(enc)
            ad.segsum(code, g32, 9, "f32")
            got: Dict[str, List[str]] = {}
            for name, fn in (
                ("codes", lambda: ad.codes(enc, pay, uih + nt, nt, rows) if not on
                 else ad.codes(enc, pay, None, nt, rows, uih_lengths=uih)),
                ("table", lambda: ad.table_inputs(enc)),
                ("segsum", lambda: ad.segsum(code, g32, 9, "f32")),
            ):
                with _Ops() as mode:
                    fn()
                got[name] = mode.seen
            seen[on] = got
        off, on_ = seen[False], seen[True]
        for name in off:
            self.assertLess(len(on_[name]), len(off[name]), (name, off[name], on_[name]))
        self.assertNotIn("bitwise_left_shift", on_["codes"])
        self.assertNotIn("sub", on_["codes"])
        self.assertLess(on_["codes"].count("index_select"), off["codes"].count("index_select"))
        self.assertLess(on_["codes"].count("arange"), off["codes"].count("arange"))
        for op in ("arange", "bitwise_and", "bitwise_left_shift", "gt"):
            self.assertNotIn(op, on_["table"], on_["table"])
        self.assertNotIn("arange", on_["segsum"])


class LeanGinTest(unittest.TestCase):
    """The binding lands with the (119) block (gin + _env_bootstrap)."""

    def test_gin_binding_once_after_119_and_bootstrap_param(self) -> None:
        with open(GIN) as f:
            text = f.read()
        hits = re.findall(r"^apply_env_bootstrap\.DLRM_HSTU_FP8_ACTION_DEDUP_LEAN = True$", text, re.M)
        self.assertEqual(len(hits), 1)
        self.assertLess(text.index("apply_env_bootstrap.DLRM_HSTU_FP8_ACTION_DEDUP = True"),
                        text.index("apply_env_bootstrap.DLRM_HSTU_FP8_ACTION_DEDUP_LEAN = True"))
        self.assertIn("Kill switch: DLRM_HSTU_FP8_ACTION_DEDUP_LEAN=0", text)
        import inspect

        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb

        self.assertIn(ad.LEAN_ENV, inspect.signature(eb.apply_env_bootstrap).parameters)


if __name__ == "__main__":
    unittest.main()

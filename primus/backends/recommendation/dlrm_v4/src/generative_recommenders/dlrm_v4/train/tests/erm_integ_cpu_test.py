#!/usr/bin/env python3
# pyre-unsafe
"""DLRM_HSTU_FP8_BWD_ERM (yambda_5b.gin item (97); r14 E-RM, bottom layer; default off).

CPU-only (no GPU, nothing built or launched). On the REAL modules of this tree, with
their flags / state patched (mock.patch.object restores them):

* the dy_erm protocol: site / claim / ticket / record / take / consumable / finish and
  every decline path D0-D15, D18, D20-D26 of the r14 design (D19, S32, does not exist on
  this tree), each falling back to today's path; unexpected declines latch E-RM only;
  decline lines are rank-tagged single writes; the startup prebuild never maps a block
  of fp8_tm_alloc's B11 pool (it runs before that pool's prewarm; the real allocator
  path, device plumbing faked), and W3's re-lay-out copies never latch that pool;
* the W3 row-major entry (wgrad_ck_w3): rm_inactive_reason per state, the explicit
  layout contract, the first-call BITWISE guard vs x21 (armed / mismatch / launch error /
  x21 failure / re-lay-out OOM), every row-major failure latches the row-major entry only
  (never `_disable`), rm_to_tiled / rm_to_tok byte equality with the dual cast's layouts,
  the E-RM library is built and bound only with the knob on;
* the consumer (bwd_fp8.output_projection_dgrad_operands): a consumable record replaces
  the dual cast's three outputs and nothing else; a declined one runs the cast exactly;
* knob-off identity: the triton_hstu_linear wrappers pass every call through to the
  shipped function with the shipped arguments, the preprocess wrapper claims nothing,
  the W3 prebuild builds and binds exactly the shipped library, every E-RM helper of
  triton_hstu_linear is defined below its last @triton.jit kernel;
* wiring: gin binds False once with the item text, the bootstrap parameter / kill switch,
  the launcher prefix, clean_exit.EXIT_HOOKS, the producer / consumer seams.
The base-vs-composed checks (Triton cache keys, FlyDSL ISA of every shipped LN-bwd variant,
the E-RM ISA == the lab's a8f9fa51) are out of tree: r14 probes/iter62/erm_integ."""

import ast
import contextlib
import os
import pathlib
import re
import types
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import torch

_REPO = pathlib.Path(__file__).resolve().parents[4]
_GR = _REPO / "generative_recommenders"
_GIN = _GR / "dlrm_v4/train/gin/yambda_5b.gin"
_BOOT = _GR / "dlrm_v4/train/_env_bootstrap.py"
_THL = _GR / "ops/triton/triton_hstu_linear.py"
_TPA = _GR / "ops/triton/triton_hstu_preprocess_and_attention.py"
_BLAS = _GR / "dlrm_v4/train/blas_warmup.py"
_STU = _GR / "modules/stu.py"
_LAUNCHER = _REPO / "scripts/launch_slurm.sh"
_KNOB = "DLRM_HSTU_FP8_BWD_ERM"

CUDA0 = torch.device("cuda", 0)  # a device object only; no CUDA call is made with it


def _E():  # pyre-ignore[3]
    from generative_recommenders.ops.triton import dy_erm

    return dy_erm


def _W():  # pyre-ignore[3]
    from generative_recommenders.ops.triton import wgrad_ck_w3

    return wgrad_ck_w3


def _B():  # pyre-ignore[3]
    from generative_recommenders.ops.triton import bwd_fp8

    return bwd_fp8


class _State:
    """A published _SiteScale stand-in (the fields the protocol reads)."""

    def __init__(self) -> None:
        self.site = "output_projection.L0"
        self.published = True
        self.calls = 7
        self.t_calls = 3
        self.t_shift = 5
        self.inv_scale = torch.ones(1)
        self.scale = torch.ones(1)
        self.t_clipped_windows = torch.zeros((), dtype=torch.int64)

    def publish(self) -> None:  # pragma: no cover - not reached at calls 7 / 8
        raise AssertionError("publish")


_WT = torch.zeros((1536, 512), requires_grad=True)  # leaf + requires grad (a Parameter's identity)


def _ops(m: int = 64, wdtype: torch.dtype = torch.bfloat16) -> Dict[str, torch.Tensor]:
    x = torch.zeros((m, 512), dtype=torch.bfloat16)
    return dict(
        x=x, dy=torch.zeros_like(x), res=torch.zeros_like(x),
        w=torch.ones(512, dtype=wdtype), b=torch.zeros(512, dtype=wdtype),
        mean=torch.zeros(m), rstd=torch.ones(m),
    )


class _Harness:
    """Patches the module state the protocol reads; restores everything on exit."""

    def __init__(self, test: unittest.TestCase, **kw: Any) -> None:
        self.t = test
        self.kw = kw
        self.state = kw.get("state") or _State()
        self.disabled: List[str] = []
        self.logs: List[str] = []
        self.wln_disabled: List[str] = []
        self.stack = contextlib.ExitStack()

    def __enter__(self) -> "_Harness":
        E, W, B = _E(), _W(), _B()
        from generative_recommenders.ops.triton import fp8_wgrad_snr as snr
        from generative_recommenders.ops.triton import triton_layer_norm as tln

        kw = self.kw
        p = self.stack.enter_context
        E.reset_for_testing()
        p(mock.patch.object(E, "_override", kw.get("knob", True)))
        p(mock.patch.object(E, "_prebuilt", kw.get("prebuilt", True)))
        p(mock.patch.object(E, "_inert_checked", True))
        p(mock.patch.object(E, "_inert", kw.get("inert")))
        p(mock.patch.object(E, "_atexit_registered", True))  # no real atexit registration in tests
        p(mock.patch.object(E, "M_RM_MAX", kw.get("m_rm_max", E.M_RM_MAX)))
        p(mock.patch.object(E, "_bwd", None))
        p(mock.patch.object(E, "_w3", None))
        p(mock.patch.object(E, "_tln", None))
        p(mock.patch.object(E, "_snr", None))
        p(mock.patch.object(E, "_stream", kw.get("stream_fn", lambda device: "S0")))
        p(mock.patch.object(E, "_print", lambda msg: self.logs.append(f"[hstu-fp8-bwd] {msg}")))
        p(mock.patch.object(E, "_arm_print", lambda msg: self.logs.append(f"[hstu-fp8-bwd] {msg}")))
        # bwd_fp8: the flags of the shipped configuration + the published site
        p(mock.patch.object(B, "scale_controller_enabled", lambda: kw.get("bwd", True)))
        p(mock.patch.object(B, "_OUT_WGRAD_ARMED", True))
        p(mock.patch.object(B, "_OUT_WGRAD_T_E4M3", True))
        p(mock.patch.object(B, "_BWD_HELPERS_FUSED", True))
        p(mock.patch.object(B, "_GRAD_DTYPE", torch.float8_e5m2))
        p(mock.patch.object(B, "_peek_site_state", lambda site, key: self.state))
        # triton_layer_norm: item (29)'s switch / latch / reasons
        p(mock.patch.object(tln, "_WLN_BWD_FLYDSL_ENABLED", kw.get("fly_on", True)))
        p(mock.patch.object(tln, "_WLN_BWD_FLYDSL_BROKEN", kw.get("fly_broken")))
        p(mock.patch.object(tln, "_wln_bwd_flydsl_reason", lambda *a, **k: kw.get("fly_reason")))
        p(mock.patch.object(tln, "_wln_bwd_flydsl_disable", lambda where, e: self.wln_disabled.append(where)))
        p(mock.patch.object(snr, "SNR_ENABLED", kw.get("snr", False)))
        # wgrad_ck_w3: the shipped x21 configuration, armed on cuda:0, xt2 bound
        p(mock.patch.object(W, "_override", kw.get("w3", True)))
        p(mock.patch.object(W, "_tiled_override", kw.get("tiled", True)))
        p(mock.patch.object(W, "_t256s_override", kw.get("t256s", True)))
        p(mock.patch.object(W, "_rm_override", kw.get("rm", True)))
        p(mock.patch.object(W, "_broken", kw.get("w3_broken")))
        p(mock.patch.object(W, "_tiled_declined", kw.get("tiled_declined")))
        p(mock.patch.object(W, "_rm_declined", kw.get("rm_declined")))
        p(mock.patch.object(W, "_tiled_ok", set(kw.get("tiled_ok", {0}))))
        p(mock.patch.object(W, "_armed_s", set(kw.get("armed_s", {0}))))
        p(mock.patch.object(W, "_armed_rm", set(kw.get("armed_rm", set()))))
        p(mock.patch.object(W, "_fn_rm", kw.get("fn_rm", lambda *a: 0)))
        p(mock.patch.object(W, "_disable", lambda reason: self.disabled.append(reason)))
        p(mock.patch.object(W, "log", self.logs.append))
        # CPU tensors stand for cuda:0 in the W3 state predicate (index 0 of _tiled_ok / _armed_s).
        self.real_rm_inactive = W.rm_inactive_reason
        p(mock.patch.object(W, "rm_inactive_reason", lambda device, _f=W.rm_inactive_reason: _f(CUDA0)))
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stack.close()
        _E().reset_for_testing()


def _site(**kw: Any):  # pyre-ignore[3]
    args = dict(fp8_y_saved=True, y_token_major=False, dout_token_major=True, out_wgrad_nn=True,
                need_bwd=True, layer_tag=0, helpers_fused_ok=True)
    args.update(kw)
    return _E().make_site(kw.pop("weight", _WT), **{k: v for k, v in args.items() if k != "weight"})


def _tagged(site, requires_grad: bool = True):  # pyre-ignore[2,3]
    out = torch.zeros((8, 512), requires_grad=requires_grad)
    setattr(out, _E().ERM_SITE_ATTR, site)
    return out


def _tk(site, o, fold: bool = False):  # pyre-ignore[2,3]
    return _E().ticket(site, o["x"], o["dy"], o["res"], o["w"], o["b"], o["mean"], o["rstd"], fold)


def _produce(m: int = 64):  # pyre-ignore[3]
    """site + claim + ticket + a fake E-RM producer run; returns (site, dx, record)."""
    E = _E()
    site = _site()
    assert site is not None
    assert E.claim(_tagged(site), residual_proxy=True, targets_only=False) is site
    o = _ops(m)
    dx = torch.zeros((m, 512), dtype=torch.bfloat16)
    g8 = torch.zeros((m, 512), dtype=torch.float8_e5m2)
    xrm = torch.zeros((m, 512), dtype=torch.float8_e4m3fn)
    part = torch.zeros(8)
    with mock.patch.object(E, "_ln_bwd_erm", lambda: (lambda *a, **k: (dx, torch.zeros(512), torch.zeros(512), g8, xrm, part))):
        out = E.run_producer(site, o["dy"], o["x"], o["w"], o["b"], o["mean"], o["rstd"], o["res"], False)
    assert out is not None and out[0] is dx
    return site, dx, site.record


class ProtocolTest(unittest.TestCase):
    """dy_erm: the r14 iter55 47-case decline matrix, ported to the real modules of this tree."""

    def test_consume_on_the_very_tensor(self) -> None:
        with _Harness(self) as h:
            E = _E()
            site, dx, rec = _produce()
            self.assertIsNotNone(rec)
            got = E.take(site)
            self.assertIs(got, rec)
            self.assertTrue(site.consumed and site.record is None)
            g8 = rec.g8
            pay = E.consumable(rec, h.state, dx, True, True, True)
            self.assertIsNotNone(pay)
            self.assertIs(pay[0], g8)
            self.assertTrue(rec.used and rec.g8 is None and rec.x_rm is None, "payload handed over, record released")
            self.assertIsNone(E.broken())
            self.assertEqual(sum("ARMED erm on" in l for l in h.logs), 1)
            self.assertEqual(sum("erm consumed on" in l for l in h.logs), 1)
            self.assertIsNone(_tk(site, _ops()), "D11: no second ticket after the consume")

    def test_site_declines(self) -> None:
        with _Harness(self):
            E = _E()
            self.assertIsNotNone(_site())
            self.assertIsNone(_site(weight=torch.zeros((1536, 512))), "D6 non-leaf weight")
            self.assertIsNone(_site(helpers_fused_ok=False), "D6 outside the fused helpers")
            self.assertIsNone(_site(fp8_y_saved=False))
            self.assertIsNone(_site(y_token_major=True), "producer (TN) mode")
            self.assertIsNone(_site(out_wgrad_nn=False), "IMPL != nn")
            self.assertIsNone(_site(dout_token_major=False))
            self.assertIsNone(_site(need_bwd=False), "D4")
            self.assertIsNone(_site(layer_tag=1), "D26 top ordinary layer")
            self.assertIsNone(_site(layer_tag=-1), "D26 untagged warmup / direct call")
            self.assertIsNone(_site(layer_tag=2), "D26 fallback-batch ordinary layer 2")
            self.assertEqual(E.counters()["sites"], 1)
        with _Harness(self, knob=False):
            self.assertIsNone(_site(), "D0 knob off")
        with _Harness(self) as h:
            _E().set_broken("x")
            self.assertIsNone(_site(), "D2 latched")
            self.assertIn("erm DISABLED: x", " ".join(h.logs))
        with _Harness(self, inert="DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S") as h:
            self.assertIsNone(_site(), "D1 INERT")
            self.assertEqual([l for l in h.logs if "requested: erm=1" in l],
                             ["[hstu-fp8-bwd] requested: erm=1 (INERT without DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S)"])

    def test_claim_declines(self) -> None:
        with _Harness(self):
            E = _E()
            s = _site()
            self.assertIsNone(E.claim(_tagged(s), residual_proxy=True, targets_only=True), "D3 targets-only")
            self.assertIsNone(E.claim(_tagged(s), residual_proxy=False, targets_only=False), "D3 no proxy")
            with torch.no_grad():
                self.assertIsNone(E.claim(_tagged(s), residual_proxy=True, targets_only=False), "D4 grad mode off")
            self.assertIsNone(E.claim(_tagged(s, requires_grad=False), residual_proxy=True, targets_only=False), "D4")
            self.assertIs(E.claim(_tagged(s), residual_proxy=True, targets_only=False), s)
            self.assertIsNone(E.claim(_tagged(s), residual_proxy=True, targets_only=False), "D11 second claim")
            self.assertIsNone(E.claim(torch.zeros(4, 512), residual_proxy=True, targets_only=False), "untagged x")
            s1 = _site(layer_tag=1)
            self.assertIsNone(E.claim(_tagged(s1), residual_proxy=True, targets_only=False),
                              "fallback batch: ordinary layer 2 finds no site on layer 1's out")

    def _ticket_none(self, label: str, o: Optional[Dict[str, torch.Tensor]] = None, fold: bool = False, **kw: Any) -> List[str]:
        with _Harness(self, **kw) as h:
            E = _E()
            s = _site()
            if s is None:
                self.assertTrue(kw.get("knob") is False, label)
                return h.logs
            E.claim(_tagged(s), residual_proxy=True, targets_only=False)
            self.assertIsNone(_tk(s, o or _ops(), fold), label)
            self.assertEqual(h.disabled, [], f"{label}: W3 never disabled by E-RM")
            self.assertEqual(h.wln_disabled, [], f"{label}: item (29) never latched by E-RM")
            return h.logs

    def test_ticket_declines(self) -> None:
        st = _State()
        st.published = False
        self.assertFalse([l for l in self._ticket_none("D5 warmup", state=st) if "NOT used" in l], "D5 is silent")
        self._ticket_none("D7 fold_dropout", fold=True)
        o = _ops()
        o["res"] = None
        self._ticket_none("D7 residual None", o=o)
        self.assertIn("SNR probe on", " ".join(self._ticket_none("D10", snr=True)))
        self._ticket_none("D1 BWD off", bwd=False)
        self.assertIn("is off", " ".join(self._ticket_none("D18 switch off", fly_on=False)))
        self.assertIn("latched off", " ".join(self._ticket_none("D18 runtime latch", fly_broken="dispatch: RuntimeError")))
        self.assertIn("latched off", " ".join(self._ticket_none("D18 prebuild latch", fly_broken="startup prebuild: X")))
        self.assertIn("padded rows", " ".join(self._ticket_none("D18 reason", fly_reason="padded rows past 32-bit offsets")))
        self.assertIn("bf16 weights only", " ".join(self._ticket_none("D8 fp32 LN weight", o=_ops(wdtype=torch.float32))))
        o = _ops()
        o["dy"] = torch.zeros((64, 512), dtype=torch.float32)
        self._ticket_none("D8 dtype", o=o)
        o = _ops()
        o["x"] = torch.zeros((512, 64), dtype=torch.bfloat16).t()
        self._ticket_none("D8 layout", o=o)
        self.assertIn("W3 off or latched", " ".join(self._ticket_none("D9 W3 latched", w3_broken="guard rejected")))
        self.assertIn("not requested", " ".join(self._ticket_none("D9 tiling off", tiled=False)))
        self.assertIn("x21 (T256S) not requested", " ".join(self._ticket_none("D9 T256S off", t256s=False)))
        self.assertIn("declined earlier", " ".join(self._ticket_none("D20", tiled_declined="M: 0")))
        self.assertIn("x21 not armed", " ".join(self._ticket_none("D21 unarmed", armed_s=set())))
        self.assertIn("tiled_active not yet", " ".join(self._ticket_none("D21 unchecked", tiled_ok=set())))
        self.assertIn("not bound", " ".join(self._ticket_none("D25 unbound", fn_rm=None)))
        self.assertIn("latched off", " ".join(self._ticket_none("D15 -> D9", rm_declined="guard")))
        self.assertIn("32-bit offsets", " ".join(self._ticket_none("D23", m_rm_max=32)))
        self.assertIn("not prebuilt", " ".join(self._ticket_none("D24", prebuilt=False)))
        self._ticket_none("D0", knob=False)
        with _Harness(self) as h:
            s = _site()
            _E().claim(_tagged(s), residual_proxy=True, targets_only=False)
            with mock.patch.object(torch.compiler, "is_compiling", lambda: True):
                self.assertIsNone(_tk(s, _ops()), "D8 compiling")
            self.assertIn("compiling/AOTI", " ".join(h.logs))

    def _consumer_drop(self, label: str, mutate, **kw: Any) -> None:  # pyre-ignore[2]
        with _Harness(self, **kw) as h:
            E = _E()
            site, dx, rec = _produce()
            got = E.take(site)
            st2, g, tiled = mutate(h, dx)
            self.assertIsNone(E.consumable(got, st2, g, True, True, tiled), label)
            self.assertIsNotNone(E.broken(), f"{label}: unexpected decline latches E-RM")
            self.assertTrue(got.g8 is None and not got.used, f"{label}: payload released")
            self.assertTrue(any("erm record dropped" in l for l in h.logs), label)
            self.assertEqual(h.disabled, [], label)

    def test_consumer_declines(self) -> None:
        def gen(h, dx):  # pyre-ignore[2,3]
            h.state.calls += 1
            return h.state, dx, True

        def tsh(h, dx):  # pyre-ignore[2,3]
            h.state.t_shift = 6
            return h.state, dx, True

        self._consumer_drop("D12 generation", gen)
        self._consumer_drop("D12 t_shift", tsh)
        self._consumer_drop("D12 state replaced", lambda h, dx: (_State(), dx, True))
        self._consumer_drop("D13 _version bump", lambda h, dx: (h.state, dx.add_(0), True))
        self._consumer_drop("D13 another tensor", lambda h, dx: (h.state, dx.clone(), True))
        self._consumer_drop("D22 consumer t_tiled False", lambda h, dx: (h.state, dx, False))

        def late_tiled(h, dx):  # pyre-ignore[2,3]
            _W()._tiled_declined = "late"
            return h.state, dx, True

        self._consumer_drop("W3 tiling latched after the emission", late_tiled)
        streams = iter(["S0", "S1"])
        self._consumer_drop("D14 stream", lambda h, dx: (h.state, dx, True), stream_fn=lambda d: next(streams))
        with _Harness(self):
            E = _E()
            site, dx, rec = _produce()
            got = E.take(site)
            got.release()
            self.assertIsNone(E.consumable(got, _State(), dx, True, True, True), "released record")

    def test_finish_and_producer_error(self) -> None:
        with _Harness(self) as h:
            E = _E()
            site, dx, rec = _produce()
            got = E.take(site)
            E.finish(got)
            self.assertTrue(got.g8 is None and E.broken() is not None, "unused record dropped + latched")
        with _Harness(self) as h:
            E = _E()
            site, dx, rec = _produce()
            got = E.take(site)
            self.assertIsNotNone(E.consumable(got, h.state, dx, True, True, True))
            E.finish(got)
            self.assertIsNone(E.broken(), "finish of a consumed record is a no-op")
        with _Harness(self) as h:
            E = _E()
            s = _site()
            E.claim(_tagged(s), residual_proxy=True, targets_only=False)
            o = _ops()

            def boom(*a: Any, **k: Any) -> None:
                raise RuntimeError("flydsl compile error")

            with mock.patch.object(E, "_ln_bwd_erm", lambda: boom):
                self.assertIsNone(E.run_producer(s, o["dy"], o["x"], o["w"], o["b"], o["mean"], o["rstd"], o["res"], False))
            self.assertIsNone(s.record)
            self.assertIn("producer: RuntimeError", E.broken() or "")
            self.assertEqual(h.wln_disabled, [], "the shipped FlyDSL latch is untouched")

    def test_prebuild_and_final(self) -> None:
        with _Harness(self, prebuilt=False) as h:
            E = _E()
            calls = []

            def no_pool(*a: Any, **k: Any) -> None:
                raise AssertionError("the startup prebuild must not allocate from the B11 pool")

            with mock.patch.object(E, "_ln_bwd_erm", lambda: (lambda *a, **k: calls.append((a, k)))), \
                    mock.patch.object(E, "_empty_x_rm", no_pool):
                self.assertEqual(E.prebuild(torch.device("cpu")), "erm=1")
            self.assertTrue(E.prebuilt())
            a, k = calls[0]
            self.assertEqual((tuple(a[1].shape), a[1].dtype, a[8]), ((1024, 512), torch.bfloat16, 5))
            self.assertEqual(float(a[7][0]), 1.0)
            xrm = k["out"]["xrm"]
            self.assertEqual((xrm.dtype, tuple(xrm.shape), xrm.is_contiguous()),
                             (torch.float8_e4m3fn, (1024, 512), True))
        with _Harness(self, prebuilt=False):
            E = _E()

            def boom(*a: Any, **k: Any) -> None:
                raise RuntimeError("no")

            with mock.patch.object(E, "_ln_bwd_erm", lambda: boom):
                self.assertEqual(E.prebuild(torch.device("cpu")), "erm=error")
            self.assertFalse(E.prebuilt())
            self.assertIn("startup prebuild", E.broken() or "")
        with _Harness(self, knob=False):
            self.assertEqual(_E().prebuild(torch.device("cpu")), "")
        with _Harness(self) as h:
            E = _E()
            E._final()
            self.assertFalse([l for l in h.logs if "erm final=1" in l], "no event: no line")
            _site()
            E._final()
            self.assertEqual(len([l for l in h.logs if "erm final=1 sites=1" in l]), 1)
        with _Harness(self, knob=False) as h:
            _E()._counts["sites"] = 1
            _E()._final()
            self.assertFalse(h.logs, "knob off: the exit hook prints nothing")

    def test_prebuild_never_maps_the_b11_pool(self) -> None:
        # The startup prebuild runs BEFORE alloc_pregrow's prewarm of
        # fp8_tm_alloc's private (B11) pool. With that pool forced active (as in
        # production: TM_NONEXP=1 under expandable segments) the REAL empty_tm /
        # _pooled / _alloc_block path runs here; only the device plumbing is
        # faked (allocations land on CPU, 1 MiB blocks, no CUDA call). The
        # prebuild maps no block and prints no pool line; the control proves the
        # harness reaches the pool through produce's allocator (_empty_x_rm).
        from generative_recommenders.ops.triton import fp8_tm_alloc as ta

        real = {n: getattr(torch, n) for n in ("empty", "zeros", "ones", "zeros_like")}

        def cpu(name: str):  # pyre-ignore[3]
            def f(*a: Any, **k: Any) -> torch.Tensor:
                k.pop("device", None)
                return real[name](*a, **k)

            return f

        def run(fn):  # pyre-ignore[2,3]
            logs: List[str] = []
            with contextlib.ExitStack() as st:
                p = st.enter_context
                p(mock.patch.object(ta, "active", lambda: True))
                p(mock.patch.object(ta, "_FLOOR", 1 << 20))
                p(mock.patch.object(ta, "_GRANULE", 1 << 20))
                p(mock.patch.object(ta, "_pool", lambda d: object()))
                p(mock.patch.object(ta, "_set_settings", lambda c: None))
                p(mock.patch.object(ta, "_print", lambda m: logs.append("[hstu-fp8-tm] " + m)))
                p(mock.patch.object(ta, "_block_ptrs", {}))
                p(mock.patch.object(ta, "_prewarmed", set()))
                p(mock.patch.object(ta, "_armed_logged", set()))
                p(mock.patch.object(ta, "_lazy_logged", set()))
                p(mock.patch.object(ta, "_broken", None))
                p(mock.patch.object(torch.cuda, "use_mem_pool",
                                    lambda pool, device=None: contextlib.nullcontext()))
                for n in real:
                    p(mock.patch.object(torch, n, cpu(n)))
                out = fn()
                state = (dict(ta._block_ptrs), set(ta._lazy_logged), set(ta._armed_logged))
            return out, state, logs

        with _Harness(self, prebuilt=False):
            E = _E()
            seen: List[torch.Tensor] = []
            with mock.patch.object(E, "_ln_bwd_erm",
                                   lambda: (lambda *a, **k: seen.append(k["out"]["xrm"]))):
                res, (blocks, lazy, armed), logs = run(lambda: E.prebuild(CUDA0))
            self.assertEqual(res, "erm=1")
            self.assertEqual((blocks, lazy, armed), ({}, set(), set()), "no B11 block mapped")
            self.assertFalse([l for l in logs if "pool created lazily" in l or "ARMED tm_nonexp" in l], logs)
            self.assertEqual((seen[0].dtype, tuple(seen[0].shape)), (torch.float8_e4m3fn, (1024, 512)))
            _, (blocks, lazy, armed), logs = run(lambda: E._empty_x_rm(1024, CUDA0))
            self.assertTrue(blocks and lazy == {0} and armed == {"cuda:0"}, "control: the real pool path ran")
            self.assertTrue(any("pool created lazily" in l for l in logs), logs)

    def test_decline_lines_are_rank_tagged_once(self) -> None:
        with _Harness(self):
            E = _E()
            plain: List[str] = []
            tagged: List[str] = []
            with mock.patch.object(E, "_print", plain.append), mock.patch.object(E, "_arm_print", tagged.append):
                E._note("snr", "erm NOT used (SNR probe on)")
                E._note("snr", "erm NOT used (SNR probe on)")
            self.assertEqual((tagged, plain), (["erm NOT used (SNR probe on)"], []),
                             "one rank-tagged single-write line per key (ops/utils.arm_print_all_ranks)")
            self.assertEqual(E._reasons["snr"], 2)


class W3RowMajorTest(unittest.TestCase):
    """wgrad_ck_w3's E-RM entry: state predicate, guard, latching, layouts."""

    M = 300

    def _run(self, h, launch, x21=None, retile=None, oom=False, xr=None):  # pyre-ignore[2,3]
        W = _W()
        m = self.M
        y8 = types.SimpleNamespace(device=CUDA0)  # run_w3_rm reads only y8.device (launch / contract patched)
        x_rm = xr if xr is not None else torch.zeros((m, 512), dtype=torch.float8_e4m3fn)
        X21 = torch.full((512, 1536), 1.5)
        calls = {"x21": 0, "retile": 0}

        def run_x21(dt):  # pyre-ignore[2,3]
            calls["x21"] += 1
            return X21.clone() if x21 is None else x21(dt)

        def rt(x, mm):  # pyre-ignore[2,3]
            calls["retile"] += 1
            if oom:
                raise MemoryError("HIP out of memory")
            return (retile or (lambda a, b: torch.zeros(3, 512, 128)))(x, mm)

        with mock.patch.object(W, "_launch_rm", launch), \
                mock.patch.object(W, "rm_decline_reason", lambda *a: None):
            out = W.run_w3_rm(y8, x_rm, torch.ones(1), torch.ones(1), m, rt, run_x21,
                              lambda e: isinstance(e, MemoryError))
        return out, X21, calls

    def test_guard_armed_then_xt2_only(self) -> None:
        with _Harness(self) as h:
            W = _W()
            out, X21, c = self._run(h, lambda *a: torch.full((512, 1536), 1.5))
            self.assertTrue(torch.equal(out, X21) and 0 in W._armed_rm and W.rm_declined() is None)
            self.assertEqual(c["x21"], 1)
            self.assertTrue(any("bitwise=x21 -> armed" in l for l in h.logs))
            self.assertTrue(any("ARMED wgrad_ck w3 on cuda:0 (dout_x row-major) entry=wgrad_ck_w3_t256s_xt2" in l for l in h.logs))
            out2, _, c2 = self._run(h, lambda *a: torch.full((512, 1536), 1.5))
            self.assertEqual((c2["x21"], c2["retile"]), (0, 0), "armed: xt2 only, no re-lay-out")

    def test_failures_return_x21_and_latch_rm_only(self) -> None:
        bad = torch.full((512, 1536), 1.5)
        bad[0, 0] = 1.5000001
        with _Harness(self) as h:
            out, X21, c = self._run(h, lambda *a: bad.clone())
            self.assertTrue(torch.equal(out, X21), "guard mismatch: x21's bits")
            self.assertIsNotNone(_W().rm_declined())
            self.assertEqual(h.disabled, [])
            self.assertTrue(any("dout_x row-major OFF" in l for l in h.logs))

        def boom(*a: Any) -> None:
            raise RuntimeError("hipErrorLaunchFailure")

        with _Harness(self) as h:
            out, X21, c = self._run(h, boom)
            self.assertTrue(torch.equal(out, X21) and _W().rm_declined() is not None and h.disabled == [])
        with _Harness(self) as h:
            out, X21, c = self._run(h, lambda *a: torch.full((512, 1536), 1.5), x21=lambda dt: None)
            self.assertIsNone(out, "x21 itself failed: None (caller -> hipBLASLt on rm_to_tok)")
        with _Harness(self) as h:
            out, X21, c = self._run(h, lambda *a: torch.full((512, 1536), 1.5), oom=True)
            self.assertTrue(torch.equal(out, X21) and 0 not in _W()._armed_rm and _W().rm_declined() is None)
            self.assertTrue(any("deferred (re-lay-out copy OOM)" in l for l in h.logs))
        with _Harness(self) as h, mock.patch.object(_W(), "_once", set()):  # fresh once-latch: the line prints
            nan = torch.full((512, 1536), float("nan"))
            out, X21, c = self._run(h, lambda *a: nan.clone(), oom=True)
            self.assertIsNone(out, "non-finite + deferred guard: None (the caller's hipBLASLt on rm_to_tok)")
            self.assertEqual((c["retile"], c["x21"]), (1, 0), "the copy that ran out of memory is not retried")
            self.assertIsNotNone(_W().rm_declined())
            self.assertEqual(h.disabled, [], "W3 itself is not latched")
            self.assertTrue(any("not all-finite -> row-major entry latched" in l for l in h.logs))

        def boom_launch(*a: Any) -> None:
            raise RuntimeError("hipErrorLaunchFailure")

        with _Harness(self) as h:
            # a fallback re-lay-out (after a launch error / decline) has no allocation-free
            # alternative: out of memory there raises (the shipped required-copy policy)
            with self.assertRaises(MemoryError):
                self._run(h, boom_launch, oom=True)
            self.assertIsNotNone(_W().rm_declined())
            self.assertEqual(h.disabled, [])
        with _Harness(self, tiled_declined="late") as h:
            out, X21, c = self._run(h, lambda *a: 1 / 0)
            self.assertTrue(torch.equal(out, X21) and c["x21"] == 1 and h.disabled == [], "D20 at the consumer")

    def test_relayout_copies_never_latch_the_pool(self) -> None:
        # triton_hstu_linear's retile hands rm_to_tiled fp8_tm_alloc.empty_copy (the
        # shipped transient-copy allocator): out of memory reaches run_w3_rm's is_oom
        # deferral and fp8_tm_alloc._broken stays None. Control: empty_tm latches it.
        from generative_recommenders.ops.triton import fp8_tm_alloc as ta
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        def oom(*a: Any, **k: Any) -> None:
            raise RuntimeError("HIP out of memory (simulated)")

        got: Dict[str, Any] = {}

        def fake_rm_to_tiled(x_rm, m, empty=torch.empty):  # pyre-ignore[2,3]
            got["empty"] = empty
            return empty((3, 512, 128), dtype=torch.float8_e4m3fn, device=CUDA0)

        def fake_run_w3_rm(y8, xr, sy, sd, m, retile, run_x21, is_oom):  # pyre-ignore[2,3]
            try:
                retile(xr, m)
            except Exception as error:  # noqa: BLE001
                got["is_oom"] = is_oom(error)
            return torch.zeros(1)

        ctx = types.SimpleNamespace(erm_rec=types.SimpleNamespace(used=True, layout="rm"))
        x_rm = torch.zeros((300, 512), dtype=torch.float8_e4m3fn)
        logs: List[str] = []
        with mock.patch.object(ta, "active", lambda: True), \
                mock.patch.object(ta, "_mapped", lambda d: True), \
                mock.patch.object(ta, "_pooled", oom), \
                mock.patch.object(ta, "_broken", None), \
                mock.patch.object(ta, "_print", logs.append), \
                mock.patch.object(ta.logger, "warning", lambda *a, **k: None), \
                mock.patch.object(thl.wgrad_ck_w3, "rm_to_tiled", fake_rm_to_tiled), \
                mock.patch.object(thl.wgrad_ck_w3, "run_w3_rm", fake_run_w3_rm):
            r = thl._erm_out_wgrad_nn_ck_w3(ctx, torch.zeros((300, 1536)), x_rm, torch.ones(1), torch.ones(1), None)
            self.assertEqual(r[1], "nn_ck")
            self.assertIs(got["empty"], ta.empty_copy)
            self.assertTrue(got.get("is_oom"), "out of memory propagates to run_w3_rm")
            self.assertIsNone(ta._broken, "the B11 pool is not latched")
            with contextlib.suppress(Exception):
                ta.empty_tm((3, 512, 128), torch.float8_e4m3fn, CUDA0)
            self.assertIsNotNone(ta._broken, "control: empty_tm latches the pool on the same error")
        self.assertTrue(any("tm_nonexp DISABLED" in l for l in logs))

    def test_inactive_reason_states(self) -> None:
        cases = [
            (dict(), None),
            (dict(rm=False), "ERM knob off"),
            (dict(rm_declined="x"), "latched off"),
            (dict(w3_broken="y"), "W3 off or latched"),
            (dict(w3=False), "W3 off or latched"),
            (dict(tiled=False), "B13"),
            (dict(tiled_declined="z"), "declined earlier"),
            (dict(t256s=False), "T256S"),
            (dict(tiled_ok=set()), "tiled_active"),
            (dict(armed_s=set()), "x21 not armed"),
            (dict(fn_rm=None), "not bound"),
        ]
        for kw, want in cases:
            with _Harness(self, **kw) as h:
                got = h.real_rm_inactive(CUDA0)
                if want is None:
                    self.assertIsNone(got, kw)
                else:
                    self.assertIn(want, got or "", kw)
        with _Harness(self) as h:
            self.assertIn("device", h.real_rm_inactive(torch.device("cpu")) or "")

    def test_contract_explicit_layout(self) -> None:
        W = _W()
        y8 = torch.zeros((512, 1536), dtype=torch.float8_e4m3fn)
        xr = torch.zeros((512, 512), dtype=torch.float8_e4m3fn)
        one = torch.ones(1)
        # on CPU every in-contract call ends at the device check: "device" = the layout was accepted
        self.assertEqual(W.rm_decline_reason(y8, xr, one, one, 512)[0], "device",
                         "a 2-D [512, 512] x_rm is row-major by the explicit tag, not [512, ldm]")
        self.assertEqual(W.rm_decline_reason(y8, xr.t(), one, one, 512)[0], "x_rm")
        self.assertEqual(W.rm_decline_reason(y8, xr, one, one, 0)[0], "M")
        self.assertEqual(W.rm_decline_reason(y8, xr, one, one, W._M_RM_MAX + 1)[0], "M")
        self.assertEqual(W.rm_decline_reason(y8, xr.view(torch.float8_e5m2), one, one, 512)[0], "dtype")
        self.assertEqual(W.rm_decline_reason(y8[:300], xr, one, one, 512)[0], "y8")
        self.assertEqual(W.rm_decline_reason(y8[:300], xr[:300], one, one, 300)[0], "device")
        self.assertEqual(W.rm_decline_reason(y8[:200], xr[:300], one, one, 300)[0], "y8")

    def test_relayouts_bitwise(self) -> None:
        W = _W()
        from generative_recommenders.ops.triton import wgrad_ck_w3 as w

        g = torch.Generator().manual_seed(7)
        for m in (1, 15, 16, 127, 128, 129, 300, 512, 1000):
            raw = torch.randint(0, 256, (m, 512), dtype=torch.uint8, generator=g)
            x_rm = raw.view(torch.float8_e4m3fn)
            tiled = W.rm_to_tiled(x_rm, m)
            kt = (m + 127) // 128
            self.assertEqual(tuple(tiled.shape), (kt, 512, 128))
            t8 = tiled.view(torch.uint8)
            # independent reference: the dual cast's T_TILED store, token t / column c at (t//128, c, t%128)
            ref = torch.zeros((kt * 128, 512), dtype=torch.uint8)
            ref[:m] = raw
            self.assertTrue(torch.equal(t8, ref.view(kt, 128, 512).transpose(1, 2)), m)
            tok = W.rm_to_tok(x_rm, m)
            self.assertTrue(torch.equal(tok.view(torch.uint8), w.untile_to_row_major(tiled, m).view(torch.uint8)), m)
            mp = (m + 15) // 16 * 16
            self.assertEqual(tuple(tok.shape), (512, mp))
            self.assertTrue(bool((tok.view(torch.uint8)[:, m:] == 0).all()))
        with self.assertRaises(ValueError):
            W.rm_to_tiled(torch.zeros((4, 256), dtype=torch.float8_e4m3fn), 4)

    def test_bind_only_with_the_knob(self) -> None:
        W = _W()
        from generative_recommenders.ops.triton import wgrad_ck

        def run_prebuild(rm: bool, bind_fails: bool = False):  # pyre-ignore[3]
            builds, binds, logs = [], [], []

            class Lib:
                def __getattr__(self, name: str) -> Any:
                    return types.SimpleNamespace(restype=None, argtypes=None, __call__=lambda: 0)

            def build(src, name):  # pyre-ignore[2,3]
                builds.append((pathlib.Path(src).name, name))
                if bind_fails and "erm" in name:
                    raise RuntimeError("hipcc rc=1")
                return pathlib.Path("/nonexistent") / name

            def bind_entry(lib, so, name, nsplit):  # pyre-ignore[2,3]
                binds.append((name, nsplit))
                return lambda *a: 0

            patches = [
                (W, "_override", True), (W, "_tiled_override", True), (W, "_t256s_override", True),
                (W, "_rm_override", rm), (W, "_broken", None), (W, "_rm_declined", None),
                (W, "_fn", None), (W, "_fn_t", None), (W, "_fn_s", None), (W, "_drain", None),
                (W, "_lib", None), (W, "_lib_path", None), (W, "_fn_rm", None), (W, "_drain_rm", None),
                (W, "_lib_rm", None), (W, "_lib_rm_path", None), (W, "_once", set()),
                (W, "log", logs.append), (wgrad_ck, "build", build), (wgrad_ck, "bind_entry", bind_entry),
                (wgrad_ck, "_check_arch", lambda d: None), (torch.cuda, "is_available", lambda: True),
                (torch.cuda, "current_device", lambda: 0),
            ]
            with contextlib.ExitStack() as stack:
                for obj, name, value in patches:
                    stack.enter_context(mock.patch.object(obj, name, value))
                stack.enter_context(mock.patch("ctypes.CDLL", lambda p: Lib()))
                ok = W.prebuild()
                state = (W._fn_rm is not None, W.rm_declined(), W.broken())
            return ok, builds, binds, logs, state

        ok, builds, binds, logs, st = run_prebuild(rm=False)
        self.assertTrue(ok)
        self.assertEqual(builds, [("wgrad_tm_ck_w3.cu", "libwgrad_ck_w3.so")], "knob off: the shipped library only")
        self.assertEqual(binds, [("wgrad_ck_w3", 16), ("wgrad_ck_w3_t", 16), ("wgrad_ck_w3_t256s", 21)])
        self.assertEqual(st, (False, None, None))
        ok, builds, binds, logs, st = run_prebuild(rm=True)
        self.assertEqual(builds[1], ("wgrad_tm_ck_w3_erm.cu", "libwgrad_ck_w3_erm.so"))
        self.assertEqual(binds[-1], ("wgrad_ck_w3_t256s_xt2", 21))
        self.assertTrue(st[0] and st[1] is None and st[2] is None)
        self.assertTrue(any("prebuild: wgrad_ck_w3_t256s_xt2 bound (row-major dout_x, nsplit=21" in l for l in logs))
        ok, builds, binds, logs, st = run_prebuild(rm=True, bind_fails=True)
        self.assertTrue(ok, "W3 itself stays armed")
        self.assertEqual(st[0], False)
        self.assertIn("bind wgrad_ck_w3_t256s_xt2", st[1] or "")
        self.assertIsNone(st[2], "D25: the row-major entry latched, W3 not disabled")

    def test_erm_source_is_shipped_source_plus_block(self) -> None:
        d = _GR / "ops/triton"
        shipped = (d / "wgrad_tm_ck_w3.cu").read_bytes()
        erm = (d / "wgrad_tm_ck_w3_erm.cu").read_bytes()
        self.assertTrue(erm.startswith(shipped), "wgrad_tm_ck_w3_erm.cu = the shipped .cu verbatim + the xt2 block")
        tail = erm[len(shipped):].decode()
        self.assertIn('extern "C" int wgrad_ck_w3_t256s_xt2(', tail)
        self.assertIn('extern "C" int wgrad_ck_w3_t256s_xt2_nsplit(void) { return 21; }', tail)


class ConsumerCastTest(unittest.TestCase):
    """bwd_fp8.output_projection_dgrad_operands with and without a record (patched kernels)."""

    def _call(self, erm=None, state=None, fused_ok=True):  # pyre-ignore[2,3]
        B = _B()
        st = state or _State()
        trace: List[Any] = []
        m = 64
        grad = torch.zeros((m, 512), dtype=torch.bfloat16)
        w = torch.zeros((1536, 512), requires_grad=True)
        part_cast = torch.zeros(4)

        def dual(*a, **k):  # pyre-ignore[2,3]
            trace.append(("dual", a[0] is grad, a[1] is st.inv_scale, tuple(sorted(k.items()))))
            return (torch.zeros((m, 512), dtype=torch.float8_e5m2),
                    torch.zeros((1, 512, 128), dtype=torch.float8_e4m3fn), part_cast)

        def site_launch(state, partial, weight, **k):  # pyre-ignore[2,3]
            trace.append(("site", state.calls, int(partial.numel()), tuple(sorted(k.items()))))
            return torch.zeros(1), torch.ones(1), torch.ones(1)

        with mock.patch.object(B, "_BWD_ENABLED", True), \
                mock.patch.object(B, "_helpers_fused_ok", lambda w: fused_ok), \
                mock.patch.object(B, "scale_controller_enabled", lambda: True), \
                mock.patch.object(B, "_site_state", lambda site, dev: st), \
                mock.patch.object(B, "quantize_row_major_dual", dual), \
                mock.patch.object(B, "_fused_site_launch", site_launch), \
                mock.patch.object(B, "_fused_weight_quant", lambda w, p: (torch.zeros(1), torch.ones(1))), \
                mock.patch.object(B, "quantize_gradient", lambda *a, **k: trace.append(("qg",)) or None), \
                mock.patch.object(B, "record_engaged", lambda s: None), \
                mock.patch.object(B, "_OUT_WGRAD_T_E4M3", True), \
                mock.patch.object(B, "_GRAD_DTYPE", torch.float8_e5m2), \
                mock.patch.object(B, "_GRAD_MAX", B._E5M2_MAX):
            out = B.output_projection_dgrad_operands(grad, w, token_major=True, t_tiled=True, erm=erm(grad, st) if erm else None)
        return out, trace, st

    def test_knob_off_is_the_shipped_trace(self) -> None:
        out, trace, st = self._call()
        self.assertEqual([t[0] for t in trace], ["dual", "site"])
        self.assertTrue(trace[0][1] and trace[0][2], "the cast reads the gradient with the live inv_scale")
        self.assertEqual(st.calls, 8)
        self.assertEqual(trace[1][2], 4, "site program on the cast's partials")

    def test_consumed_record_replaces_the_cast_only(self) -> None:
        held = {}

        def rec(grad, st):  # pyre-ignore[2,3]
            E = _E()
            tk = E.ErmTicket(st, st.calls, st.t_shift, st.inv_scale, "S0")
            r = E.ErmEmit(tk, grad, torch.zeros((64, 512), dtype=torch.float8_e5m2),
                          torch.zeros((64, 512), dtype=torch.float8_e4m3fn), torch.zeros(2048))
            held["g8"], held["x_rm"] = r.g8, r.x_rm
            return r

        with _Harness(self):
            out, trace, st = self._call(erm=rec)
            self.assertEqual([t[0] for t in trace], ["site"], "no dual cast")
            self.assertEqual(trace[0][2], 2048, "site program on the producer's partials")
            off_out, off_trace, _ = self._call()
            self.assertEqual(trace[0][3], off_trace[1][3], "same site-launch flags (snapshot, t_note, shift, snap_t)")
            self.assertEqual(st.calls, 8, "calls += 1 exactly once")
            self.assertIs(out[0], held["g8"])
            self.assertIs(out[2], held["x_rm"], "operand = row-major x_rm")
            self.assertIsNone(_E().broken())

    def test_declined_record_runs_the_cast(self) -> None:
        def rec(grad, st):  # pyre-ignore[2,3]
            E = _E()
            tk = E.ErmTicket(st, st.calls - 1, st.t_shift, st.inv_scale, "S0")  # generation moved
            return E.ErmEmit(tk, grad, torch.zeros((64, 512), dtype=torch.float8_e5m2),
                             torch.zeros((64, 512), dtype=torch.float8_e4m3fn), torch.zeros(2048))

        with _Harness(self):
            out, trace, st = self._call(erm=rec)
            self.assertEqual([t[0] for t in trace], ["dual", "site"], "today's cast, exactly")
            self.assertIsNotNone(_E().broken())
        with _Harness(self):
            out, trace, st = self._call(erm=lambda g, s: _E().ErmEmit(
                _E().ErmTicket(s, s.calls, s.t_shift, s.inv_scale, "S0"), g,
                torch.zeros((64, 512), dtype=torch.float8_e5m2),
                torch.zeros((64, 512), dtype=torch.float8_e4m3fn), torch.zeros(2048)), fused_ok=False)
            self.assertEqual([t[0] for t in trace], ["qg"])
            self.assertIn("did not reach", _E().broken() or "", "finish() drops a record the cast never saw")


class KnobOffIdentityTest(unittest.TestCase):
    """With the knob off the E-RM seams pass every call through unchanged."""

    def test_thl_wrappers_pass_through(self) -> None:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        ctx = types.SimpleNamespace(fp8_y_saved=True, fp8_dout_token_major=True)
        out = torch.zeros(3)
        with mock.patch.object(_E(), "_override", False):
            self.assertIs(thl._erm_forward_tag(ctx, out, _WT, True), out)
            self.assertFalse(hasattr(ctx, "erm_site") or hasattr(out, _E().ERM_SITE_ATTR))
            self.assertIsNone(thl._erm_take(ctx))
            self.assertFalse(hasattr(ctx, "erm_rec"))
        seen: List[Any] = []
        a = (torch.zeros(1), torch.zeros(2), torch.zeros(3), torch.zeros(4), object())
        with mock.patch.object(thl, "_out_wgrad_nn_ck_w3", lambda *x: seen.append(("w3", x)) or "R1"), \
                mock.patch.object(thl, "_dout_row_major", lambda *x: seen.append(("rm", x)) or "R2"), \
                mock.patch.object(thl.wgrad_ck_w3, "untile_to_row_major", lambda *x: seen.append(("un", x)) or "R3"):
            self.assertEqual(thl._erm_out_wgrad_nn_ck_w3(ctx, *a), "R1")
            self.assertEqual(thl._erm_dout_row_major(ctx, a[1], 7, True), "R2")
            self.assertEqual(thl._erm_untile_to_row_major(ctx, a[1], 7), "R3")
        self.assertEqual(seen[0], ("w3", a))
        self.assertEqual(seen[1], ("rm", (a[1], 7, True)))
        self.assertEqual(seen[2], ("un", (a[1], 7)))

    def test_thl_rm_layout_routing(self) -> None:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        rec = types.SimpleNamespace(used=True, layout="rm")
        ctx = types.SimpleNamespace(erm_rec=rec)
        x_rm = torch.zeros((40, 512), dtype=torch.float8_e4m3fn)
        with mock.patch.object(thl.fp8_tm_alloc, "empty_copy", lambda s, dtype, device: torch.empty(s, dtype=dtype)):
            tok = thl._erm_dout_row_major(ctx, x_rm, 40, True)
        self.assertEqual((tuple(tok.shape), rec.layout), ((512, 48), "tok2d"))
        self.assertIs(thl._erm_dout_row_major(ctx, tok, 40, True), tok, "a second reader takes the copy as is")
        rec.layout = "rm"
        got = {}

        def fake_rm(y8, xr, sy, sd, m, retile, run_x21, is_oom):  # pyre-ignore[2,3]
            got.update(m=m, xr=xr, retile=retile, run_x21=run_x21)
            return torch.zeros(1)

        with mock.patch.object(thl.wgrad_ck_w3, "run_w3_rm", fake_rm):
            r = thl._erm_out_wgrad_nn_ck_w3(ctx, torch.zeros((40, 1536)), x_rm, torch.ones(1), torch.ones(1), None)
        self.assertEqual((r[1], got["m"]), ("nn_ck", 40))
        self.assertIs(got["xr"], x_rm)

    def test_claim_with_the_knob_off(self) -> None:
        E = _E()
        x = torch.zeros((4, 512), requires_grad=True)
        with mock.patch.object(E, "_override", False):
            self.assertIsNone(E.claim(x, residual_proxy=True, targets_only=False))
        self.assertFalse(hasattr(x, E.ERM_SITE_ATTR))

    def test_erm_helpers_below_every_triton_kernel(self) -> None:
        # A kernel's starting line enters Triton's cache key: every E-RM addition of
        # triton_hstu_linear sits below its last @triton.jit function, the tpa hooks below
        # its kernels and bwd_fp8's below its kernels.
        for path, prefixes in ((_THL, ("_erm_",)), (_TPA, ("_dy_erm",)),
                               (_GR / "ops/triton/bwd_fp8.py", ("_dy_erm", "_output_projection_dgrad_operands"))):
            tree = ast.parse(path.read_text())
            jit = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and any("triton" in ast.unparse(d) for d in n.decorator_list)]
            mine = [n.lineno for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith(prefixes)]
            self.assertTrue(jit and mine, path.name)
            self.assertGreater(min(mine), max(jit), path.name)


class WiringTest(unittest.TestCase):
    def test_gin_binds_once_with_item_text(self) -> None:
        # Exactly one boolean binding, and the item text names the bound value
        # ("bound False below" / "bound True below"): a gin-on flip edits the gin
        # item only (it never needs to edit this test).
        lines = _GIN.read_text().splitlines()
        hits = [i for i, l in enumerate(lines) if re.match(rf"^apply_env_bootstrap\.{_KNOB}\s*=", l)]
        self.assertEqual(len(hits), 1)
        bound = re.fullmatch(rf"apply_env_bootstrap\.{_KNOB} = (True|False)", lines[hits[0]])
        self.assertIsNotNone(bound, lines[hits[0]])
        start = max(i for i in range(hits[0]) if lines[i].startswith("# ----"))
        block = lines[start: hits[0] + 1]
        self.assertTrue(all(len(l) <= 80 for l in block))
        text = "\n".join(block)
        self.assertIn(f"bound {bound.group(1)} below", text)
        for marker in ("(97) E-RM", "WHAT:", "NUMERICS: BIT-IDENTICAL", "Code default OFF", "kill switch",
                       f"{_KNOB}=0", "ARMED", "Must be ABSENT", "INERT", "erm=1 (INERT", "erm DISABLED",
                       "erm NOT used (", "wgrad_ck_w3 dout_x row-major OFF", "COLD START",
                       '"pool created lazily"', '"pool GREW"', '"tm_nonexp DISABLED"',
                       '"re-laid-out copy of the row-major x_rm"', "without DLRM_HSTU_FP8_BWD, items",
                       "6 production main kernels"):
            self.assertIn(marker, text)
        self.assertNotIn("items (2)", text, "gin item (2) is C4 dropout, not the FP8 backward arm")

    def test_bootstrap_binding_and_kill_switch(self) -> None:
        src = _BOOT.read_text()
        self.assertEqual(src.count(f'_bind_bool("{_KNOB}", {_KNOB})'), 1)
        self.assertIn(f"    {_KNOB}: Optional[bool] = None,", src)
        from generative_recommenders.dlrm_v4.train import _env_bootstrap as eb
        import inspect

        self.assertIn(_KNOB, inspect.signature(eb.apply_env_bootstrap).parameters)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_KNOB, None)
            eb._bind_bool(_KNOB, False)
            self.assertEqual(os.environ[_KNOB], "0", "gin False -> 0 (off)")
            os.environ[_KNOB] = "0"
            eb._bind_bool(_KNOB, True)
            self.assertEqual(os.environ[_KNOB], "0", "a pre-set env value wins (kill switch)")

    def test_launcher_forwards_the_prefix(self) -> None:
        if not _LAUNCHER.is_file():
            self.skipTest("no scripts/launch_slurm.sh in this tree")
        self.assertIn('for v in "${!DLRM_HSTU_FP8_@}"; do', _LAUNCHER.read_text())

    def test_exit_hook_listed(self) -> None:
        from generative_recommenders.dlrm_v4.train import clean_exit as ce

        self.assertIn(("generative_recommenders.ops.triton.dy_erm", "_final"), ce.EXIT_HOOKS)
        src = (_GR / "ops/triton/dy_erm.py").read_text()
        self.assertEqual(src.count("atexit.register(_final)"), 1)

    def test_seams(self) -> None:
        thl = _THL.read_text()
        cls = thl.split("class HSTUComputeOutputFunction")[1].split("def _helion_ln_mul_dropout_fwd")[0]
        self.assertEqual(cls.count("erm=_erm_take(ctx),"), 2, "both dual-cast calls carry the record")
        self.assertEqual(cls.count("return _erm_forward_tag(ctx, out, output_weight, need_bwd)"), 1)
        self.assertEqual(cls.count("_erm_out_wgrad_nn_ck_w3(ctx,"), 1)
        self.assertEqual(cls.count("dout_fp8 = _erm_dout_row_major(ctx,"), 2)
        self.assertEqual(cls.count("dout_fp8=_erm_untile_to_row_major(ctx,"), 1)
        self.assertNotIn("dout_fp8 = _dout_row_major(", cls)
        self.assertNotIn("        return out\n", cls.split("def backward")[0].split("ctx.do8_site = l11_do8.site_of(attn)")[1])
        tpa = ast.parse(_TPA.read_text())
        fn = next(n for n in ast.walk(tpa) if isinstance(n, ast.ClassDef) and n.name == "_HSTUPreprocessAndAttentionFunction")
        fwd = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        impl = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == "_backward_impl")
        n_inputs = len(fwd.args.args) - 1  # minus ctx
        rets = [n for n in ast.walk(impl) if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)]
        self.assertEqual([len(r.value.elts) for r in rets], [n_inputs], "backward returns one value per forward input")
        self.assertEqual(fwd.args.args[-1].arg, "erm_site")
        src = ast.unparse(impl)
        self.assertIn("_dy_erm().run_producer(ctx.erm_site", src)
        # the producer runs only under `if <site>:`, right after the early DO8 release
        gate = [n for n in ast.walk(impl) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "getattr(ctx, 'erm_site', None) is not None"]
        self.assertEqual(len(gate), 1)
        body = [ast.unparse(s) for s in gate[0].body]
        self.assertEqual(body[0], "hstu_do8_registry.release(getattr(ctx, 'do8_site', None))")
        self.assertTrue(body[1].startswith("erm_out = _dy_erm().run_producer(ctx.erm_site"), body)
        self.assertEqual(len(body), 2)
        self.assertEqual(gate[0].orelse, [])
        self.assertEqual(src.count("hstu_do8_registry.release("), 1, "the only release inside _backward_impl")
        self.assertRegex(src, r"erm_out = None\n\s*if getattr\(ctx, 'erm_site', None\) is not None:")
        wrapper = next(n for n in tpa.body if isinstance(n, ast.FunctionDef) and n.name == "triton_hstu_preprocess_and_attention")
        self.assertIn("_dy_erm().claim(x, residual_proxy=return_residual_proxy, targets_only=False)", ast.unparse(wrapper))
        tgt = next(n for n in tpa.body if isinstance(n, ast.FunctionDef) and n.name == "triton_hstu_preprocess_and_attention_targets_only")
        self.assertNotIn("dy_erm", ast.unparse(tgt), "D3: the targets-only Function never claims")

    def test_prebuild_block_never_latches_item_29(self) -> None:
        src = _BLAS.read_text()
        block = src.split("# r14 E-RM (DLRM_HSTU_FP8_BWD_ERM")[1].split("# Not swallowed, as in warmup()")[0]
        self.assertGreater(src.index("# r14 E-RM (DLRM_HSTU_FP8_BWD_ERM"), src.index("# E-M2's FlyDSL grouped"),
                           "appended last: the shipped tokens of the startup line keep their neighbours")
        code = "\n".join(l for l in block.splitlines() if not l.strip().startswith("#"))
        self.assertIn('os.environ.get("DLRM_HSTU_FP8_BWD_ERM", "0") == "1"', code)
        self.assertIn("_tln._WLN_BWD_FLYDSL_BROKEN is None", code)
        self.assertNotIn("_wln_bwd_flydsl_disable", code)
        # the device-fault message names the narrower E-RM switch only when its block ran
        tail = src.split("# r14 E-RM (DLRM_HSTU_FP8_BWD_ERM")[1].split("# Not swallowed, as in warmup()")[1]
        tail = tail.split("return result")[0]
        self.assertIn("DLRM_HSTU_FP8_WLN_BWD_FLYDSL=0 to isolate.", tail)
        self.assertIn("DLRM_HSTU_FP8_BWD_ERM=0 is the", tail)
        self.assertIn('if " erm=" in result', tail)

    def test_layer_tag_set_before_each_compute_output(self) -> None:
        src = _STU.read_text()
        self.assertGreaterEqual(src.count("self._tag_lnmd_layer()"), 2)
        for chunk in src.split("self._tag_lnmd_layer()")[1:3]:
            self.assertIn("hstu_compute_output(", chunk[:200])


if __name__ == "__main__":
    unittest.main()

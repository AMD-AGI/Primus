#!/usr/bin/env python3
# pyre-strict
"""LUT-SiLU fwd (ship17 item (120), knob DLRM_HSTU_FP8_LUT_SILU_FWD, code
default OFF): the forward's arming (hstu_flydsl_mha.fp8_fwd_lut_config: off /
on / invalid / decline without the pv=k128 forward or without mask_arith=3,
one armed line per process), the launcher's table
argument (flydsl_hstu_attention_fwd_fp8 inserts the LUT tensor at positional
index 5, right after SEQ_OFF, only when on), the per-scale table cache
(_fwd_lut_table: one host build and upload per distinct scale, reuse, one
"table built" line per table, one stream wait per launch stream, fail closed
outside the scale contract) and the FP8 warm-up's handling of the latch and
caches (_STDOUT_PASS / _KEEP_GLOBALS).

Ladder chain (v2 tree, item (105) SPF fold fwd bound ON before this item):
LutSpfLauncherTest pins the composition at the launcher: with
DLRM_HSTU_FP8_LAB_FWD_SPF=1 the compile kwargs carry spf=True and lut=13, the
table is the r = 1 table built from the same (on-grid) qkv_scale the kernel
receives, and an off-grid scale is refused by SPF before any table is built.

The gin binding and the bootstrap's fail-closed checks of item (120) are
pinned by fwd_lut_silu_gin_test.py.

CPU-only: nothing here launches a kernel or touches a GPU. The FlyDSL kernel
module is faked (SHIPPED + a recording compile_hstu_fwd) and the CUDA helpers
of _fwd_lut_table (_fwd_lut_upload, _fwd_lut_launch_stream) are replaced by
recording stand-ins. Skips where torch / hstu_flydsl_mha are not importable."""

import contextlib
import io
import os
import types
import unittest
from unittest import mock

_KNOB = "DLRM_HSTU_FP8_LUT_SILU_FWD"
_ALPHA = 1.0 / (128 ** 0.5)  # stu.py: attention_dim 128, attn_alpha unset
# The production forward (gin): FWD_PIPE + MASK_ARITH + FWD_PV + item (100).
_PROD_ENV = {"DLRM_HSTU_FP8_FWD_PIPE": "1", "DLRM_HSTU_FP8_FWD_MASK_ARITH": "1",
             "DLRM_HSTU_FP8_FWD_PV": "1", "DLRM_HSTU_FP8_LAB_FWD_V8SWZ": "1",
             "DLRM_HSTU_FP8_LAB_FWD_RIDER": "1"}
_LATCH_SETS = ("_fp8_fwd_lut_logged", "_fp8_fwd_rider_logged",
               "_fp8_fwd_v8swz_logged", "_fp8_fwd_pv_logged",
               "_fp8_fwd_mask_arith_logged", "_logged_fp8_fwd_layouts")
_LATCH_SCALARS = ("_fp8_fwd_pipe_logged",)
_CACHES = ("_fwd_lut_tables", "_fwd_lut_streams")
_ARMED = (f"[hstu-fwd] {_KNOB}=1 armed: lut=13 (P8 = 8 KB LDS e4m3 table "
          "indexed by f16_rtz(2**j*raw); QK MFMA E8M0 A-scale byte from "
          "score_scale; declared precision change)")
_BUILT = f"[hstu-fwd] {_KNOB} table built: "
# The lut-relevant keys of the production cfg (mask_arith=3: lut's only
# causal mask is mask_arith=3's packed P8 byte mask).
_PV = {"pv": "k128", "pv_k": 3, "epi_fold": True, "pv_s": 10.04,
       "mask_arith": 3}


class _Stream:
    """Launch-stream stand-in: records wait_event calls."""

    def __init__(self, handle: int) -> None:
        self.cuda_stream = handle
        self.waits = []  # pyre-ignore[4]

    def wait_event(self, event) -> None:  # pyre-ignore[2]
        self.waits.append(event)


class _FakeKernel:
    """hstu_fwd stand-in: SHIPPED and a compile_hstu_fwd whose call records
    its positional arguments."""

    SHIPPED = {"BM": 64, "BN": 32, "sched": "iglp1", "waves": 2}

    def __init__(self) -> None:
        self.calls = []  # pyre-ignore[4]
        self.compiles = []  # pyre-ignore[4]

    def compile_hstu_fwd(self, heads, dim, alpha, **kw):  # pyre-ignore[2,3]
        self.compiles.append(dict(kw, heads=heads, dim=dim, alpha=alpha))
        lut = kw.get("lut", 0)
        # hstu_fwd.py's lut contract (asserted there before any MLIR):
        # pv="k128" with epi_fold, and mask_arith=3.
        assert not lut or (kw.get("pv") == "k128" and kw.get("epi_fold")
                           and kw.get("mask_arith") == 3), kw

        def call(*args, **ckw):  # pyre-ignore[2,3]
            self.calls.append((args, ckw))

        call.BM = kw.get("BM", 64)
        call.lut = lut
        return call


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
            from generative_recommenders.ops import fp8_lut_silu
            from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_flydsl_mha / fp8_lut_silu not importable: {error}")
        self.torch = torch
        self.mha = mha
        self.lut = fp8_lut_silu
        # Once-latches and caches are process state: keep this test unleaked.
        saved = {n: set(getattr(mha, n)) for n in _LATCH_SETS}
        scalars = {n: getattr(mha, n) for n in _LATCH_SCALARS}
        caches = {n: dict(getattr(mha, n)) for n in _CACHES}

        def restore() -> None:
            for n, v in saved.items():
                getattr(mha, n).clear()
                getattr(mha, n).update(v)
            for n, v in scalars.items():
                setattr(mha, n, v)
            for n, v in caches.items():
                getattr(mha, n).clear()
                getattr(mha, n).update(v)
        self.addCleanup(restore)
        for n in ("_fp8_fwd_lut_logged",) + _CACHES:
            getattr(mha, n).clear()  # a fresh process
        self.uploads = []  # pyre-ignore[4]
        self.streams = {}  # pyre-ignore[4]
        self.current = 7  # handle of the "current" launch stream

        def upload(host, device):  # pyre-ignore[2,3]
            self.uploads.append((bytes(host), device))
            dev = torch.frombuffer(bytearray(host), dtype=torch.uint8)
            return dev, ("cpu", len(self.uploads)), ("event", len(self.uploads))

        def launch_stream(device):  # pyre-ignore[2,3]
            return self.streams.setdefault(self.current, _Stream(self.current))

        for name, fn in (("_fwd_lut_upload", upload),
                         ("_fwd_lut_launch_stream", launch_stream)):
            p = mock.patch.object(mha, name, fn)
            p.start()
            self.addCleanup(p.stop)

    def _env(self, values):  # pyre-ignore[2,3]
        """Context: the DLRM_* knobs set to `values` exactly."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("DLRM_")}
        env.update({k: v for k, v in values.items() if v is not None})
        return mock.patch.dict(os.environ, env, clear=True)


class LutConfigTest(_Base):
    def _cfg(self, value, cfg):  # pyre-ignore[2,3]
        with self._env({_KNOB: value}):
            return self.mha.fp8_fwd_lut_config(cfg)

    def test_off_values_change_nothing(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for value in (None, "", "0"):
                self.assertEqual(self._cfg(value, dict(_PV)), {}, value)
                self.assertEqual(self._cfg(value, {}), {}, value)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(self.mha._fp8_fwd_lut_logged, set())

    def test_on_arms_once(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for _ in range(3):
                self.assertEqual(self._cfg("1", dict(_PV)), {"lut": 13})
        self.assertEqual(out.getvalue().splitlines(), [_ARMED])
        self.assertEqual(self.mha._fp8_fwd_lut_logged, {"armed"})

    def test_invalid_value_fails_closed(self) -> None:
        for bad in ("2", "13", "true", "on", " 1", "1 ", "01", "1.0", "-1",
                    "yes", "off"):
            for cfg in (dict(_PV), {}):
                with self.assertRaises(ValueError, msg=(bad, cfg)) as cm:
                    self._cfg(bad, cfg)
                self.assertIn(_KNOB, str(cm.exception))
                self.assertIn(repr(bad), str(cm.exception))

    def test_declined_without_pv(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for cfg in ({}, {"pv": ""}, {"pv": "k128"},
                        {"pv": "k128", "epi_fold": False}):
                self.assertEqual(self._cfg("1", cfg), {}, cfg)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith(
            f"[hstu-fwd] {_KNOB}=1 declined: needs the pv=k128 forward "),
            lines[0])
        self.assertEqual(self.mha._fp8_fwd_lut_logged, {"declined"})

    def test_declined_without_mask_arith(self) -> None:
        # lut runs no per-element select: without mask_arith=3 (its packed P8
        # byte mask) the masked tail group would be non-causal, so it
        # declines (one line naming mask_arith) instead of arming.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for cfg in ({"pv": "k128", "epi_fold": True, "pv_k": 3},
                        dict(_PV, mask_arith=0), dict(_PV, mask_arith=None)):
                self.assertEqual(self._cfg("1", cfg), {}, cfg)
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith(
            f"[hstu-fwd] {_KNOB}=1 declined: needs the pv=k128 forward "),
            lines[0])
        self.assertIn("with mask_arith=3 (DLRM_HSTU_FP8_FWD_MASK_ARITH=1)",
                      lines[0])
        self.assertIn("mask_arith=0;", lines[0])
        self.assertEqual(self.mha._fp8_fwd_lut_logged, {"declined"})

    def test_config_adds_lut_only_when_on(self) -> None:
        fake = _FakeKernel()
        with mock.patch.object(self.mha, "_load_fwd_kernel", lambda: fake):
            with self._env(_PROD_ENV):
                base = self.mha._fp8_fwd_config()
            with self._env(dict(_PROD_ENV, **{_KNOB: "0"})):
                off = self.mha._fp8_fwd_config()
            with self._env(dict(_PROD_ENV, **{_KNOB: "1"})):
                on = self.mha._fp8_fwd_config()
            # Not the pv=k128 forward: declines, nothing added.
            with self._env({_KNOB: "1", "DLRM_HSTU_FP8_FWD_PIPE": "1"}):
                bf16p = self.mha._fp8_fwd_config()
            # pv=k128 without mask_arith=3: declines, nothing added.
            with self._env(dict(_PROD_ENV, **{
                    _KNOB: "1", "DLRM_HSTU_FP8_FWD_MASK_ARITH": "0"})):
                noma = self.mha._fp8_fwd_config()
        self.assertEqual(base.get("pv"), "k128")
        self.assertTrue(base.get("epi_fold"))
        self.assertEqual(base.get("mask_arith"), 3)
        self.assertEqual((base.get("v8_swz"), base.get("rider")), (1, 3))
        self.assertNotIn("lut", base)
        self.assertEqual(off, base)
        self.assertEqual(on, dict(base, lut=13))
        self.assertNotIn("lut", bf16p)
        self.assertEqual(noma.get("pv"), "k128")
        self.assertNotEqual(noma.get("mask_arith"), 3)
        self.assertNotIn("lut", noma)


class LutLauncherTest(_Base):
    """flydsl_hstu_attention_fwd_fp8 with the faked kernel module."""

    def _forward(self, value, qkv_scale=2.0 ** -6, env=None):  # pyre-ignore[2,3]
        torch = self.torch
        fake = _FakeKernel()
        q, k, v = (torch.zeros((16, 4, 128), dtype=torch.uint8)
                   for _ in range(3))
        so = torch.tensor([0, 8, 16], dtype=torch.int64)
        with mock.patch.object(self.mha, "_load_fwd_kernel", lambda: fake):
            with self._env(dict(env if env is not None else _PROD_ENV,
                                **{_KNOB: value})):
                self.mha.flydsl_hstu_attention_fwd_fp8(
                    q, k, v, so, 16, _ALPHA, qkv_scale)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(fake.compiles), 1)
        return fake, q, so

    def test_lut_argument_only_when_on(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            for value in (None, "0"):
                fake, q, _so = self._forward(value)
                args, kw = fake.calls[0]
                self.assertEqual(len(args), 9, value)
                self.assertNotIn("lut", fake.compiles[0])
                self.assertEqual(kw, {})
            self.assertEqual(self.uploads, [])
            fake, q, _so = self._forward("1")
        args, kw = fake.calls[0]
        self.assertEqual(fake.compiles[0]["lut"], 13)
        self.assertEqual(len(args), 10)
        self.assertEqual(kw, {})
        lut_t = args[5]
        self.assertEqual(lut_t.dtype, self.torch.uint8)
        self.assertEqual(tuple(lut_t.shape), (self.lut.SIZE,))
        ss = self.lut.score_scale32(2.0 ** -6, _ALPHA)
        want = self.lut.build_table(ss, 10.04, 3)
        self.assertEqual(bytes(lut_t.numpy().tobytes()), want)
        # Every other argument keeps its position and value.
        self.assertEqual(args[6:8], (2, -(-16 // 128)))
        self.assertEqual(args[8:], (1.0 / 16.0, 2.0 ** -6))
        self.assertEqual(args[4].dtype, self.torch.int32)
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(self.uploads[0][0], want)
        self.assertEqual(self.streams[self.current].waits, [("event", 1)])
        lines = out.getvalue().splitlines()
        self.assertEqual(lines.count(_ARMED), 1, lines)
        self.assertEqual(sum(ln.startswith(_BUILT) for ln in lines), 1, lines)

    def test_declined_forward_unchanged(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fake, _q, _so = self._forward(
                "1", env={"DLRM_HSTU_FP8_FWD_PIPE": "1"})
        args, _kw = fake.calls[0]
        self.assertEqual(len(args), 9)
        self.assertNotIn("lut", fake.compiles[0])
        self.assertEqual(self.uploads, [])

    def test_declined_without_mask_arith(self) -> None:
        # The gin's production env with DLRM_HSTU_FP8_FWD_MASK_ARITH=0 and the
        # knob "1": the pv=k128 forward still arms, but without mask_arith=3
        # the lut path would have no causal mask, so the launcher declines
        # and runs the forward unchanged: no lut kwarg, no table.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fake, _q, _so = self._forward("1", env=dict(
                _PROD_ENV, DLRM_HSTU_FP8_FWD_MASK_ARITH="0"))
        args, kw = fake.calls[0]
        self.assertEqual(len(args), 9)
        self.assertEqual(kw, {})
        self.assertEqual(fake.compiles[0].get("pv"), "k128")
        self.assertNotEqual(fake.compiles[0].get("mask_arith"), 3)
        self.assertNotIn("lut", fake.compiles[0])
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.mha._fwd_lut_tables, {})
        lines = out.getvalue().splitlines()
        declined = [ln for ln in lines
                    if ln.startswith(f"[hstu-fwd] {_KNOB}=1 declined: ")]
        self.assertEqual(len(declined), 1, lines)
        self.assertIn("mask_arith=3 (DLRM_HSTU_FP8_FWD_MASK_ARITH=1)",
                      declined[0])
        self.assertNotIn(_ARMED, lines)
        self.assertFalse(any(ln.startswith(_BUILT) for ln in lines), lines)


class LutTableCacheTest(_Base):
    def _table(self, qkv_scale, cfg=None):  # pyre-ignore[2,3]
        return self.mha._fwd_lut_table(
            self.torch.device("cpu"), _ALPHA, qkv_scale,
            dict(_PV) if cfg is None else cfg)

    def test_one_build_per_scale_and_reuse(self) -> None:
        lut = self.lut
        scales = (2.0 ** -6, 0.0146325, 2.0 ** -6, 0.0146325, 0.03, 2.0 ** -6)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            got = [self._table(s) for s in scales]
        self.assertEqual(len(self.uploads), 3)
        self.assertIs(got[0], got[2])
        self.assertIs(got[0], got[5])
        self.assertIs(got[1], got[3])
        self.assertIsNot(got[0], got[1])
        for s, t in zip(scales, got):
            ss = lut.score_scale32(s, _ALPHA)
            self.assertEqual(t.numpy().tobytes(), lut.build_table(ss, 10.04, 3))
        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 3, lines)
        for s, line in zip((2.0 ** -6, 0.0146325, 0.03), lines):
            ss = lut.score_scale32(s, _ALPHA)
            j, r = lut.split_scale(ss)
            tbl = lut.build_table(ss, 10.04, 3)
            self.assertEqual(
                line,
                _BUILT + "score_scale=%.9g (0x%08x) j=%d r=%.9g S=%g "
                "bytes=8192 crc32=%08x" % (ss, lut.bits32(ss), j, r, 10.04,
                                           lut.table_crc32(tbl)))
        keys = sorted(self.mha._fwd_lut_tables, key=repr)
        self.assertEqual(len(keys), 3)
        for key in keys:
            self.assertEqual(key[0], None)  # cpu stand-in: no device index
            self.assertEqual(key[2:], (10.04, 3))

    def test_pv_scale_and_k_key_the_table(self) -> None:
        lut = self.lut
        pow2 = {"pv": "k128", "pv_k": 3, "epi_fold": True}  # pv_s unset = 0.0
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            a = self._table(2.0 ** -6)
            b = self._table(2.0 ** -6, pow2)
        self.assertEqual(len(self.uploads), 2)
        ss = lut.score_scale32(2.0 ** -6, _ALPHA)
        self.assertEqual(b.numpy().tobytes(), lut.build_table(ss, 0.0, 3))
        self.assertNotEqual(a.numpy().tobytes(), b.numpy().tobytes())
        self.assertIn(" S=8 ", out.getvalue().splitlines()[1])

    def test_one_wait_per_launch_stream(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(3):
                self._table(2.0 ** -6)
            self.current = 9  # another launch stream
            for _ in range(2):
                self._table(2.0 ** -6)
            self._table(0.03)  # a new table: its own event
        self.assertEqual(self.streams[7].waits, [("event", 1)])
        self.assertEqual(self.streams[9].waits, [("event", 1), ("event", 2)])

    def test_out_of_contract_fails_closed(self) -> None:
        lut = self.lut
        # score_scale = -(s*s*alpha*log2e): zero, non-finite, f32 overflow,
        # f32 subnormal or j outside [J_MIN, J_MAX].
        bad = [0.0, -0.0, float("nan"), float("inf"), 1e30, 1e-30, 1e17,
               1e-17]
        for s in bad[-2:]:  # j just outside [J_MIN, J_MAX]: a normal f32
            with self.assertRaisesRegex(ValueError, "outside"):
                lut.split_scale(lut.score_scale32(s, _ALPHA))
        for s in bad:
            with self.assertRaises(RuntimeError, msg=repr(s)) as cm:
                self._table(s)
            text = str(cm.exception)
            self.assertIn(_KNOB, text)
            self.assertIn(f"set {_KNOB}=0", text)
            self.assertIn("outside the table contract", text)
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.mha._fwd_lut_tables, {})
        # The edges of the contract are accepted.
        with contextlib.redirect_stdout(io.StringIO()):
            for s in (2.0 ** -6, 4.0):
                j, _r = lut.split_scale(lut.score_scale32(s, _ALPHA))
                self.assertTrue(lut.J_MIN <= j <= lut.J_MAX)
                self._table(s)


class LutSpfLauncherTest(_Base):
    """Ladder chain: item (105) SPF fold fwd ("1") composed with item (120)."""

    _SPF = "DLRM_HSTU_FP8_LAB_FWD_SPF"
    _G15 = 0.015469998121261597  # SPF grid point j = -15: score_scale = -2**-15

    def setUp(self) -> None:
        super().setUp()
        try:
            from generative_recommenders.ops import fp8_qkv_spf  # noqa: F401
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"fp8_qkv_spf not importable here: {error}")
        saved = dict(self.mha._fp8_fwd_spf_logged)

        def restore() -> None:
            self.mha._fp8_fwd_spf_logged.clear()
            self.mha._fp8_fwd_spf_logged.update(saved)
        self.addCleanup(restore)

    def _forward(self, value, qkv_scale):  # pyre-ignore[2,3]
        torch = self.torch
        fake = _FakeKernel()
        q, k, v = (torch.zeros((16, 4, 128), dtype=torch.uint8)
                   for _ in range(3))
        so = torch.tensor([0, 8, 16], dtype=torch.int64)
        with mock.patch.object(self.mha, "_load_fwd_kernel", lambda: fake):
            with self._env(dict(_PROD_ENV, **{self._SPF: "1", _KNOB: value})):
                self.mha.flydsl_hstu_attention_fwd_fp8(
                    q, k, v, so, 16, _ALPHA, qkv_scale)
        return fake

    def test_spf_and_lut_compose_with_the_r1_table(self) -> None:
        lut = self.lut
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            on = self._forward("1", self._G15)
            off = self._forward("0", self._G15)
        self.assertEqual((on.compiles[0].get("spf"), on.compiles[0].get("lut")),
                         (True, 13))
        self.assertTrue(off.compiles[0].get("spf"))
        self.assertNotIn("lut", off.compiles[0])
        self.assertEqual(len(off.calls[0][0]), 9)
        args, kw = on.calls[0]
        self.assertEqual((len(args), kw), (10, {}))
        self.assertEqual(args[9], self._G15)  # the scale the table is built from
        ss = lut.score_scale32(self._G15, _ALPHA)
        self.assertEqual(ss, -(2.0 ** -15))
        self.assertEqual(lut.split_scale(ss), (-15, 1.0))
        self.assertEqual(bytes(args[5].numpy().tobytes()),
                         lut.build_table(ss, 10.04, 3))
        built = [ln for ln in out.getvalue().splitlines()
                 if ln.startswith(_BUILT)]
        self.assertEqual(len(built), 1, built)
        self.assertIn(" j=-15 r=1 S=10.04 ", built[0])

    def test_off_grid_scale_refused_before_any_table(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "not on the SPF grid"):
                self._forward("1", 2.0 ** -6)
        self.assertEqual(self.uploads, [])
        self.assertEqual(self.mha._fwd_lut_tables, {})


class LutWarmupTest(unittest.TestCase):
    def test_warmup_keeps_latch_and_caches(self) -> None:
        try:
            from generative_recommenders.dlrm_v4.train import fp8_warmup as fw
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"fp8_warmup not importable here: {error}")
        self.assertIn("[hstu-fwd] " + _KNOB + "=", fw._STDOUT_PASS)
        # The table-built line is not an arming line: it stays captured.
        self.assertFalse((_BUILT + "score_scale=1").startswith(fw._STDOUT_PASS))
        self.assertTrue(_ARMED.startswith(fw._STDOUT_PASS))
        for name in ("_fp8_fwd_lut_logged", "_fwd_lut_tables",
                     "_fwd_lut_streams"):
            self.assertIn((fw._P_FLYDSL + ".hstu_flydsl_mha", name),
                          fw._KEEP_GLOBALS, name)
            self.assertNotIn(name, fw._MHA_SETS + fw._MHA_DICTS)


class LutHelpersSourceTest(unittest.TestCase):
    """The CUDA helpers' contract, by source (no GPU here)."""

    def test_upload_on_side_stream(self) -> None:
        import inspect

        try:
            from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_flydsl_mha not importable here: {error}")
        src = inspect.getsource(mha._fwd_lut_upload)
        i_with = src.index("with torch.cuda.stream(side):")
        for text in ("dev = torch.empty(", "dev.copy_(cpu, non_blocking=True)",
                     "event.record(side)"):
            self.assertGreater(src.index(text), i_with, text)
        # The host copy is staged in pinned memory (an async H2D copy on any
        # stream or hardware queue), filled from the table bytes.
        i_pin = src.index("pin_memory=True")
        self.assertLess(i_pin, i_with)
        self.assertIn("torch.frombuffer(bytearray(host), dtype=torch.uint8)", src)
        for text in ("synchronize", ".item()", ".cpu()", ".tolist()"):
            self.assertNotIn(text, src)
        side = inspect.getsource(mha._fwd_lut_side_stream)
        # The kernel's own fail-closed check (no flydsl here: by source).
        import pathlib

        fwd = (pathlib.Path(mha.__file__).with_name("hstu_fwd.py")
               .read_text(encoding="utf-8"))
        self.assertIn("assert not lut or mask_arith == 3", fwd)
        self.assertIn("_fwd_lut_streams", side)
        # The CPU fake above stands in for exactly these two helpers.
        self.assertIsInstance(mha._fwd_lut_launch_stream, types.FunctionType)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
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

"""MGPK (DLRM_HSTU_FP8_MASKGEN_PK): _ln_mul_dropout_fwd_maskgen_pk must write the bytes of the shipped
_ln_mul_dropout_fwd_maskgen (item (28)(e)) bit for bit: the int8 [N, 512] dropout mask (bit 0 y, bit 1 x,
bit 2 u; the format the LN*U*dropout backward v2, B1a's e5m2-du producer and L11 read), the e4m3 y
[N, 1536], and the fp32 mean / rstd.

CPU tests (need torch and triton importable, no GPU launch): the fp32 rounding of the kernel's Y_INV_SCALE
argument, the gate's knob / inv_scale / alignment / latch decisions and its printed lines.
GPU tests (gfx950; skipped elsewhere):
  * kernel level: both kernels launched with the production launcher's arguments into output buffers
    pre-filled with two different junk patterns (an unwritten byte cannot match), N % 8 in {0, 5}, three
    i64 seeds, silu_u True / False, Y_INV_SCALE incl. an fp32 subnormal and ~FLT_MAX; inputs with special
    rows (NaN / +-Inf / bf16 subnormal / +-0, the sigmoid fallback, e4m3 overflow, mean = inf / var = NaN,
    var overflow) in the first, a middle and the last (partial) groups, rows of random bf16 bit patterns
    (every NaN payload and sign) in x, in u and in both, and special w / b columns;
  * production entry (_triton_layer_norm_mul_dropout_fwd_impl, gate armed, N >= 2^20): knob on vs knob
    off bitwise, the ACTIVE line only after a real launch, the inv_scale = 1e39 / inf declines (the
    shipped kernel runs), and a raising launch latching the item off with the shipped kernel's bytes.
"""

import contextlib
import io
import math
import os
import unittest
from typing import Dict, Tuple
from unittest import mock

import torch

D = 512
EPS = 1e-6
P_DROP = 0.1
# i64 seeds, as production's torch.randint(0, 2**62) draws
SEEDS = (900450186894289455, 3309065548986932868, 1441151880758558720)
INV = 23.893333029513894
# fp32 subnormal, ~FLT_MAX, 2^-10, 1: every value the gate lets through behaves the same
INV_EXTRA = (1.0e-41, 3.0e38, 2.0**-10, 1.0)
# N % 8 == 0 (and % 16 == 0) and N % 8 == 5 (partial last 8-row group)
NS = (8 * 4096, 8 * 4099 + 5)
_ENV_ARMED = {
    "DLRM_HSTU_FP8_LNMD_FWD_MASKGEN": "1",
    "DLRM_HSTU_FP8_MASK16": "1",
    "DLRM_HSTU_FP8_LNMD_FWD_V2": "1",
    "DLRM_HSTU_FP8_LNMD_FWD_V3": "1",
}
_GATE_GLOBALS = (
    "_LNMD_FWD_MASKGEN_PK_ENABLED",
    "_LNMD_FWD_MASKGEN_PK_ARMED_LOGGED",
    "_LNMD_FWD_MASKGEN_PK_DECLINE_LOGGED",
    "_LNMD_FWD_MASKGEN_PK_BROKEN",
    "_LNMD_FWD_MASKGEN_PK_LAYERS",
    "_LNMD_FWD_MASKGEN_PK_GATED_LOGGED",
    "_LNMD_LAYER_TAG",
    "_LNMD_FWD_MASKGEN_ENABLED",
    "_LNMD_FWD_MASKGEN_ARMED_LOGGED",
    "_LNMD_FWD_MASKGEN_ARCH_LOGGED",
    "_LNMD_FWD_V3_ENABLED",
    "_MASK16_REPORTED",
)


def _thl():
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    return thl


def _importable() -> bool:
    try:
        _thl()
        return True
    except Exception:
        return False


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None or not _importable():
        return False
    try:
        return _thl()._gcn_arch_name(torch.cuda.current_device()).startswith("gfx950")
    except Exception:
        return False


class _GateState:
    """Save / restore the lazy env flags and once-latches the tests touch."""

    def __init__(self) -> None:
        self.thl = _thl()
        self.saved = {k: getattr(self.thl, k) for k in _GATE_GLOBALS if hasattr(self.thl, k)}

    def reset(self) -> None:
        self.thl._LNMD_FWD_MASKGEN_PK_ENABLED = None
        self.thl._LNMD_FWD_MASKGEN_PK_ARMED_LOGGED = False
        self.thl._LNMD_FWD_MASKGEN_PK_DECLINE_LOGGED = False
        self.thl._LNMD_FWD_MASKGEN_PK_BROKEN = None
        self.thl._LNMD_FWD_MASKGEN_PK_LAYERS = None
        self.thl._LNMD_FWD_MASKGEN_PK_GATED_LOGGED = False
        self.thl._LNMD_LAYER_TAG = -1
        self.thl._LNMD_FWD_MASKGEN_ENABLED = None
        self.thl._LNMD_FWD_MASKGEN_ARMED_LOGGED = False

    def restore(self) -> None:
        for k, v in self.saved.items():
            setattr(self.thl, k, v)


# ------------------------------------------------------------------------------------------------ CPU tests
@unittest.skipUnless(_importable(), "triton_hstu_linear not importable (torch / triton missing)")
class MaskgenPkGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.st = _GateState()
        self.st.reset()
        self.thl = self.st.thl

    def tearDown(self) -> None:
        self.st.restore()

    def test_fp32_kernel_arg(self) -> None:
        f = self.thl._fp32_kernel_arg
        flt_max = 3.4028234663852886e38
        self.assertEqual(f(1.0e39), math.inf)
        self.assertEqual(f(-1.0e39), -math.inf)
        self.assertEqual(f(math.inf), math.inf)
        self.assertEqual(f(flt_max), flt_max)
        # halfway between FLT_MAX and 2^128 rounds to even = 2^128 = overflow; just below stays FLT_MAX
        self.assertEqual(f(3.4028235677973366e38), math.inf)
        self.assertEqual(f(3.4028235677973362e38), flt_max)
        self.assertEqual(f(1.0e-46), 0.0)
        self.assertEqual(f(1.0e-45), 2.0**-149)
        self.assertTrue(math.isnan(f(math.nan)))
        self.assertEqual(f(INV), float(torch.tensor(INV, dtype=torch.float32)))
        self.assertEqual(f(1.0e-41), float(torch.tensor(1.0e-41, dtype=torch.float32)))

    def _gate(
        self,
        inv: float,
        knob: str = "1",
        y_stride: int = 3 * D,
        m_stride: int = D,
        layers: "str | None" = None,
    ) -> Tuple[bool, str]:
        y = torch.empty((4, y_stride), dtype=torch.uint8)[:, : 3 * D]
        m = torch.empty((4, m_stride), dtype=torch.int8)[:, :D]
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_MASKGEN_PK": knob}), contextlib.redirect_stdout(buf):
            if layers is None:
                os.environ.pop("DLRM_HSTU_FP8_MASKGEN_PK_LAYERS", None)
            else:
                os.environ["DLRM_HSTU_FP8_MASKGEN_PK_LAYERS"] = layers
            got = self.thl._use_lnmd_fwd_maskgen_pk(inv, y, m)
        return got, buf.getvalue()

    def test_layer_list_parse(self) -> None:
        f = self.thl._lnmd_fwd_maskgen_pk_layers
        for raw, want in (("", ""), (" ", ""), (",", ""), ("1", ",1,"), (" 1 ", ",1,"),
                          ("1,0", ",0,1,"), ("2,1,1", ",1,2,"), ("10", ",10,")):
            self.st.reset()
            with mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_MASKGEN_PK_LAYERS": raw}):
                self.assertEqual(f(), want, raw)
            self.assertIsNone(self.thl._LNMD_FWD_MASKGEN_PK_BROKEN, raw)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DLRM_HSTU_FP8_MASKGEN_PK_LAYERS", None)
            self.assertEqual(f(), "")
        for raw in ("-1", "a", "1;2", "1.0", "x,1", "²"):  # "²".isdigit() is True, int() rejects it
            self.st.reset()
            buf = io.StringIO()
            with mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_MASKGEN_PK_LAYERS": raw}), \
                    contextlib.redirect_stdout(buf):
                self.assertEqual(f(), "", raw)
            self.assertIn("malformed DLRM_HSTU_FP8_MASKGEN_PK_LAYERS", self.thl._LNMD_FWD_MASKGEN_PK_BROKEN, raw)
            self.assertIn("LN*U*dropout fwd maskgen pk DISABLED", buf.getvalue(), raw)
            self.assertIn("[rank ", buf.getvalue(), raw)

    def test_gate_layer_list(self) -> None:
        # unset / empty list: every layer, tagged or not
        for layers in (None, ""):
            for tag in (-1, 0, 1, 2):
                self.st.reset()
                self.thl.set_lnmd_layer_tag(tag)
                got, out = self._gate(INV, layers=layers)
                self.assertTrue(got, (layers, tag))
                self.assertEqual(out, "", (layers, tag))
        # list "1": layer 1 and untagged calls run _pk; layers 0 and 2 the shipped kernel
        self.st.reset()
        for tag, want in ((1, True), (-1, True), (0, False), (2, False), (1, True), (0, False)):
            self.thl.set_lnmd_layer_tag(tag)
            got, out = self._gate(INV, layers="1")
            self.assertEqual(got, want, tag)
            if tag == 0 and not self.thl._LNMD_FWD_MASKGEN_PK_GATED_LOGGED:
                self.fail("layer-gated line not latched")
        # the layer-gated line: once per process, every rank, names the list and the layer
        self.st.reset()
        self.thl.set_lnmd_layer_tag(0)
        got, out = self._gate(INV, layers="1")
        self.assertFalse(got)
        self.assertIn("LN*U*dropout fwd maskgen pk layer-gated (DLRM_HSTU_FP8_MASKGEN_PK_LAYERS=1): STU layer 0", out)
        self.assertIn("[rank ", out)
        self.assertNotIn("declined", out)  # a layer-gated call is not a declined call
        self.assertNotIn("ACTIVE", out)
        got, out = self._gate(INV, layers="1")
        self.assertFalse(got)
        self.assertEqual(out, "")
        # the list is read once with the knob (lazy, like the knob itself)
        self.thl.set_lnmd_layer_tag(1)
        got, _ = self._gate(INV, layers="0")
        self.assertTrue(got)
        # a list on the knob-off path changes nothing
        self.st.reset()
        self.thl.set_lnmd_layer_tag(1)
        got, out = self._gate(INV, knob="0", layers="1")
        self.assertFalse(got)
        self.assertEqual(out, "")
        # the inv_scale / alignment declines still apply on a listed layer
        self.st.reset()
        self.thl.set_lnmd_layer_tag(1)
        got, out = self._gate(math.inf, layers="1")
        self.assertFalse(got)
        self.assertIn("LN*U*dropout fwd maskgen pk declined", out)
        # a malformed list latches the item off (the shipped kernel everywhere)
        self.st.reset()
        self.thl.set_lnmd_layer_tag(1)
        got, out = self._gate(INV, layers="one")
        self.assertFalse(got)
        self.assertIn("LN*U*dropout fwd maskgen pk DISABLED", out)
        got, out = self._gate(INV, layers="1")
        self.assertFalse(got)

    def test_stu_layer_tags(self) -> None:
        """modules/stu.py: each STUStack layer tags the compute output that follows with its index; a layer
        outside a stack tags -1 (every list accepts an untagged call)."""
        from generative_recommenders.modules.stu import STULayer, STULayerConfig, STUStack

        config = STULayerConfig(embedding_dim=32, num_heads=2, hidden_dim=16, attention_dim=16)
        layers = [STULayer(config) for _ in range(3)]
        STUStack(layers)
        for i, layer in enumerate(layers):
            layer._tag_lnmd_layer()
            self.assertEqual(self.thl._LNMD_LAYER_TAG, i)
        STULayer(config)._tag_lnmd_layer()
        self.assertEqual(self.thl._LNMD_LAYER_TAG, -1)

    def test_stu_forward_paths_tag_before_compute_output(self) -> None:
        """The property the layer list relies on: STUStack.forward (plain and GLUE_FOLD3 _g3_forward) and the
        targets-only stack path tag each STU layer's compute output with that layer's index right before
        hstu_compute_output. The attention and compute-output ops are stubbed (CPU); the stub records the tag."""
        from generative_recommenders.modules import stu as stu_mod
        from generative_recommenders.ops import glue_fold3

        E, H, Dh, Da = 32, 2, 16, 16
        config = stu_mod.STULayerConfig(embedding_dim=E, num_heads=H, hidden_dim=Dh, attention_dim=Da)
        layers = [stu_mod.STULayer(config) for _ in range(3)]
        stack = stu_mod.STUStack(layers)
        lengths = torch.tensor([3, 5], dtype=torch.int64)
        offsets = torch.tensor([0, 3, 8], dtype=torch.int64)
        num_targets = torch.ones(2, dtype=torch.int64)
        x = torch.randn(int(offsets[-1]), E)
        seen = []

        def pre(**kw):
            n = kw["x"].shape[0]
            return (torch.zeros(n, H * Dh), torch.zeros(n, H * Dh), torch.zeros(n, H * Da), torch.zeros(n, H * Dh))

        def pre_targets(**kw):
            n = kw["seq_offsets"].numel() - 1
            return (torch.zeros(n, H * Dh), torch.zeros(n, H * Dh))

        def out(**kw):
            seen.append(self.thl._LNMD_LAYER_TAG)
            return torch.zeros(kw["x"].shape[0], E)

        with mock.patch.object(stu_mod, "hstu_preprocess_and_attention", side_effect=pre), \
                mock.patch.object(stu_mod, "hstu_preprocess_and_attention_targets_only", side_effect=pre_targets), \
                mock.patch.object(stu_mod, "hstu_compute_output", side_effect=out), \
                mock.patch.object(stu_mod.STULayer, "supports_targets_only", return_value=True), \
                mock.patch.object(stu_mod, "hstu_fp8_fuse_residual_enabled", return_value=False), \
                mock.patch.object(stu_mod.STUStack, "_g3_begin", lambda *a, **k: None), \
                mock.patch.object(stu_mod.STUStack, "_g3_end", lambda *a, **k: None):
            for pcast in (False, True):
                with mock.patch.object(glue_fold3, "PCAST", pcast):
                    seen.clear()
                    self.thl.set_lnmd_layer_tag(-1)
                    stack(x=x, x_lengths=lengths, x_offsets=offsets, max_seq_len=5, num_targets=num_targets)
                    self.assertEqual(seen, [0, 1, 2], f"forward, PCAST={pcast}")
                    seen.clear()
                    self.thl.set_lnmd_layer_tag(-1)
                    stack.forward_targets_only(
                        x=x, x_lengths=lengths, x_offsets=offsets, max_seq_len=5, num_targets=num_targets
                    )
                    self.assertEqual(seen, [0, 1, 2], f"targets-only, PCAST={pcast}")

    def test_layer_tag_setter(self) -> None:
        self.thl.set_lnmd_layer_tag(3)
        self.assertEqual(self.thl._LNMD_LAYER_TAG, 3)
        self.thl.set_lnmd_layer_tag(-1)
        self.assertEqual(self.thl._LNMD_LAYER_TAG, -1)

    def test_armed_line_names_layers(self) -> None:
        self.thl._LNMD_FWD_MASKGEN_PK_LAYERS = ",1,"
        self.thl._LNMD_LAYER_TAG = 1
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.thl._lnmd_fwd_maskgen_pk_armed()
        self.assertIn("armed OK: LN*U*dropout fwd maskgen pk ACTIVE", buf.getvalue())
        self.assertIn("STU layers 1, first launch on layer tag 1", buf.getvalue())
        self.st.reset()
        self.thl._LNMD_FWD_MASKGEN_PK_LAYERS = ""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.thl._lnmd_fwd_maskgen_pk_armed()
        self.assertIn("STU layers all, first launch on layer tag -1", buf.getvalue())

    def test_gate_knob_off(self) -> None:
        for knob in ("0",):
            self.st.reset()
            got, out = self._gate(INV, knob=knob)
            self.assertFalse(got)
            self.assertEqual(out, "")
        self.st.reset()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DLRM_HSTU_FP8_MASKGEN_PK", None)
            y = torch.empty((4, 3 * D), dtype=torch.uint8)
            m = torch.empty((4, D), dtype=torch.int8)
            self.assertFalse(self.thl._use_lnmd_fwd_maskgen_pk(INV, y, m))  # absent = off

    def test_gate_inv_scale(self) -> None:
        for inv in (INV, 1.0e-41, 3.0e38, 2.0**-10, 1.0, 3.4028234663852886e38):
            self.st.reset()
            got, out = self._gate(inv)
            self.assertTrue(got, inv)
            self.assertEqual(out, "", inv)  # the ACTIVE line belongs to the launch, not the gate
        for inv in (1.0e39, math.inf, -math.inf, math.nan, 0.0, -0.0, -1.0, 1.0e-46):
            self.st.reset()
            got, out = self._gate(inv)
            self.assertFalse(got, inv)
            self.assertIn("LN*U*dropout fwd maskgen pk declined", out)
            self.assertIn("[rank ", out)
            got2, out2 = self._gate(inv)  # printed once per process
            self.assertFalse(got2)
            self.assertEqual(out2, "")

    def test_gate_alignment_and_latch(self) -> None:
        got, out = self._gate(INV, y_stride=3 * D + 1)
        self.assertFalse(got)
        self.assertIn("not dword-aligned", out)
        self.st.reset()
        got, _ = self._gate(INV, m_stride=D + 2)
        self.assertFalse(got)
        self.st.reset()
        self.thl._LNMD_FWD_MASKGEN_PK_BROKEN = "RuntimeError: test"
        got, out = self._gate(INV)
        self.assertFalse(got)
        self.assertEqual(out, "")

    def test_lines(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.thl._lnmd_fwd_maskgen_pk_armed()
            self.thl._lnmd_fwd_maskgen_pk_armed()
        out = buf.getvalue()
        self.assertEqual(out.count("armed OK: LN*U*dropout fwd maskgen pk ACTIVE"), 1)
        self.assertIn("[rank ", out)
        # the shipped line's marker must not match the new line (grep -F markers stay distinct)
        self.assertNotIn("armed OK: LN*U*dropout fwd maskgen ACTIVE", out)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.thl._lnmd_fwd_maskgen_pk_latch(RuntimeError("boom"))
        self.assertIn("LN*U*dropout fwd maskgen pk DISABLED", buf.getvalue())
        self.assertEqual(self.thl._LNMD_FWD_MASKGEN_PK_BROKEN, "RuntimeError: boom")


# ------------------------------------------------------------------------------------------------ GPU helpers
def _bf16_bits(t_int: torch.Tensor) -> torch.Tensor:
    return t_int.to(torch.int16).view(torch.bfloat16)


def _inputs(n: int, special_w: bool = False, gen_seed: int = 1234):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev)
    g.manual_seed(gen_seed)
    x = (torch.randn(n, D, device=dev, generator=g) * 0.5 + 0.05).to(torch.bfloat16)
    u = (torch.randn(n, D, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    b = (0.1 * torch.randn(D, device=dev, generator=g)).to(torch.bfloat16)
    sub = torch.tensor(1e-39, dtype=torch.bfloat16, device=dev)  # bf16 subnormal
    nan_neg = _bf16_bits(torch.tensor(-64, dtype=torch.int32, device=dev))  # 0xFFC0
    snan = _bf16_bits(torch.tensor(0x7F81, dtype=torch.int32, device=dev))
    for base in sorted({0, min(8 * 512 + 64, n - 64), n - 64}):
        u[base + 0, 17] = -100.0  # sigmoid fallback (d > 2^126)
        u[base + 1, 0] = float("nan")
        u[base + 2, 5] = float("-inf")
        u[base + 3, 6] = float("inf")
        x[base + 4, :] = 0.75  # constant row: var = 0
        x[base + 5, 100] = float("inf")
        x[base + 6, 7] = float("nan")
        x[base + 7, :] = -0.0
        u[base + 7, :] = -0.0
        x[base + 8, :] = 0.0
        u[base + 8, :] = -0.0
        u[base + 9, :] = -0.0
        x[base + 10, :64] = 100.0  # e4m3 overflow in the x / y planes
        u[base + 11, :64] = 300.0  # e4m3 overflow in the u plane
        x[base + 12, :64] = -100.0  # negative overflow
        u[base + 13, 9] = -88.0  # just past the fast-sigmoid range
        u[base + 14, 9] = -87.0  # just inside it
        u[base + 15, :] = 88.5  # exp(-u) subnormal, d == 1
        u[base + 16, :] = -87.5  # d just above 2^126
        x[base + 17, :] = sub  # bf16 subnormal rows
        x[base + 18, ::2] = sub
        x[base + 18, 1::2] = -sub
        u[base + 19, :] = sub
        u[base + 20, :] = -1.0e-40
        x[base + 21, :] = 3.0e38  # sum overflows: mean inf, var NaN
        x[base + 22, :256] = 3.0e38
        x[base + 22, 256:] = -3.0e38
        x[base + 23, ::2] = 1.0e20  # var overflows: rstd 0
        x[base + 23, 1::2] = -1.0e20
        u[base + 24, :] = float("nan")
        x[base + 25, :] = 1.0e-30
        x[base + 26, 5] = nan_neg
        u[base + 26, 6] = nan_neg
        x[base + 27, 9] = snan
        u[base + 27, 10] = snan
        x[base + 28, :] = -0.0
        u[base + 28, :] = nan_neg

    def rbits(rows: int) -> torch.Tensor:
        return _bf16_bits(torch.randint(-32768, 32768, (rows, D), generator=g, device=dev, dtype=torch.int32))

    ia = torch.arange(101 % n, n, 997, device=dev)  # x random bit patterns, u normal
    x[ia] = rbits(len(ia))
    ib = torch.arange(211 % n, n, 991, device=dev)  # u in the fast-path range down to bf16 subnormals / zeros
    e = torch.rand(len(ib), D, generator=g, device=dev) * 141.0 - 135.0
    s = torch.where(torch.rand(len(ib), D, generator=g, device=dev) < 0.5, -1.0, 1.0)
    uu = (s * torch.exp2(e) * (1 + torch.rand(len(ib), D, generator=g, device=dev))).clamp(-86.0, 86.0)
    uu[:, ::37] = 0.0
    uu[:, 1::37] = -0.0
    u[ib] = uu.to(torch.bfloat16)
    ic = torch.arange(307 % n, n, 983, device=dev)  # both random bit patterns (every NaN payload / sign)
    x[ic] = rbits(len(ic))
    u[ic] = rbits(len(ic))
    u[n - 1, 3] = -100.0  # fallback in the last (possibly partial) group
    if special_w:
        w[0:8] = 0.0
        b[8:16] = 0.0
        w[16:24] = -1.5
        b[24:32] = -0.0
        w[32] = float("nan")
        w[33] = float("inf")
        b[34] = float("inf")
        b[35] = nan_neg
        w[36] = -0.0
        b[36] = -0.0
    return x, u, w, b


def _bufs(n: int, junk: int) -> Dict[str, torch.Tensor]:
    dev = torch.device("cuda")
    out = dict(
        y=torch.empty((n, 3 * D), dtype=torch.float8_e4m3fn, device=dev),
        mean=torch.empty((n,), dtype=torch.float32, device=dev),
        rstd=torch.empty((n,), dtype=torch.float32, device=dev),
        mask=torch.empty((n, D), dtype=torch.int8, device=dev),
    )
    out["y"].view(torch.uint8).fill_(0x5A + junk)
    out["mask"].fill_(0x7F - junk)
    out["mean"].view(torch.int32).fill_(0x7FC00011 + junk)
    out["rstd"].view(torch.int32).fill_(0x7FC00021 + junk)
    return out


def _launch(kernel, B, x, u, w, b, seed: int, silu_u: bool, inv: float) -> None:
    thl = _thl()
    n = x.shape[0]
    k16 = thl._mask_keep_min16(P_DROP)
    kk2 = (k16 - 1) * 0x10001
    if kk2 >= 2**31:
        kk2 -= 2**32
    kernel[((n + 7) // 8,)](
        x, u, B["y"], w, b, B["mean"], B["rstd"], B["mask"], n, D, EPS, P_DROP,
        x.stride(0), u.stride(0), B["y"].stride(0), B["mask"].stride(0), float(inv), k16, kk2, seed,
        BLOCK_D=D, NUM_MASKS=3, MUL_U_ACTIVATION_TYPE="silu", OUTPUT_FP8=True, SILU_U=silu_u, num_warps=1,
    )


def _diff(a: Dict[str, torch.Tensor], r: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return dict(
        mask=int((a["mask"] != r["mask"]).sum()),
        y=int((a["y"].view(torch.uint8) != r["y"].view(torch.uint8)).sum()),
        mean=int((a["mean"].view(torch.int32) != r["mean"].view(torch.int32)).sum()),
        rstd=int((a["rstd"].view(torch.int32) != r["rstd"].view(torch.int32)).sum()),
    )


def _diff4(a, r) -> Dict[str, int]:
    return _diff(dict(y=a[0], mean=a[1], rstd=a[2], mask=a[3]), dict(y=r[0], mean=r[1], rstd=r[2], mask=r[3]))


# ------------------------------------------------------------------------------------------------ GPU tests
@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU (the kernels' inline asm)")
class MaskgenPkKernelTest(unittest.TestCase):
    def _check(self, n: int, seeds, silus, invs, special_w: bool = False) -> None:
        thl = _thl()
        x, u, w, b = _inputs(n, special_w=special_w)
        for seed in seeds:
            for silu_u in silus:
                for inv in invs:
                    ref = _bufs(n, 0)
                    _launch(thl._ln_mul_dropout_fwd_maskgen, ref, x, u, w, b, seed, silu_u, inv)
                    for junk in (1, 2):  # an unwritten byte cannot equal both junk patterns
                        out = _bufs(n, junk)
                        _launch(thl._ln_mul_dropout_fwd_maskgen_pk, out, x, u, w, b, seed, silu_u, inv)
                        torch.cuda.synchronize()
                        d = _diff(out, ref)
                        self.assertEqual(d, dict(mask=0, y=0, mean=0, rstd=0),
                                         f"n={n} seed={seed} silu_u={silu_u} inv={inv} junk={junk} "
                                         f"special_w={special_w}")
                    # the mask keeps its int8 3-plane format (the readers' contract)
                    self.assertEqual(ref["mask"].dtype, torch.int8)
                    self.assertEqual(int(((ref["mask"].to(torch.int32) & ~7) != 0).sum()), 0)
                    keep = [float(((ref["mask"] >> j) & 1).float().mean()) for j in range(3)]
                    for kf in keep:
                        self.assertAlmostEqual(kf, 1.0 - P_DROP, delta=0.01)

    def test_bitwise_seeds(self) -> None:
        for n in NS:
            self._check(n, SEEDS, (True, False), (INV,))

    def test_bitwise_inv_scales(self) -> None:
        for n in NS:
            self._check(n, SEEDS[:1], (True,), INV_EXTRA)

    def test_bitwise_special_w(self) -> None:
        for n in NS:
            self._check(n, SEEDS[1:2], (True,), (INV, 1.0e-41), special_w=True)


@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU (the kernels' inline asm)")
class MaskgenPkImplTest(unittest.TestCase):
    """The production entry with the item (28)(e) gate armed (N >= 2^20, the v2 row threshold)."""

    def setUp(self) -> None:
        self.st = _GateState()
        self.thl = self.st.thl
        self.env = mock.patch.dict(os.environ, dict(_ENV_ARMED))
        self.env.start()
        self.st.reset()

    def tearDown(self) -> None:
        self.env.stop()
        self.st.restore()

    def _impl(self, x, u, w, b, seed: int, inv: float, knob: str, tag: int = -1):
        self.st.reset()
        self.thl.set_lnmd_layer_tag(tag)
        os.environ["DLRM_HSTU_FP8_MASKGEN_PK"] = knob
        f = self.thl._triton_layer_norm_mul_dropout_fwd_impl  # the custom op (or plain function)
        idx = torch.empty(0, dtype=torch.int64, device="cuda")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = f(x, u, w, b, EPS, P_DROP, True, True, True, True, "silu", seed, idx, True, float(inv))
        torch.cuda.synchronize()
        return out, buf.getvalue()

    def test_impl_knob(self) -> None:
        for n in ((1 << 20) + 8 * 7, (1 << 20) + 8 * 7 + 5):
            x, u, w, b = _inputs(n)
            for seed in SEEDS:
                ref, out0 = self._impl(x, u, w, b, seed, INV, "0")
                self.assertIn("armed OK: LN*U*dropout fwd maskgen ACTIVE", out0)  # the shipped gate armed
                self.assertNotIn("maskgen pk", out0)
                got, out1 = self._impl(x, u, w, b, seed, INV, "1")
                self.assertEqual(_diff4(got, ref), dict(mask=0, y=0, mean=0, rstd=0), f"n={n} seed={seed}")
                self.assertEqual(out1.count("armed OK: LN*U*dropout fwd maskgen pk ACTIVE"), 1)
                self.assertTrue(self.thl._LNMD_FWD_MASKGEN_PK_ARMED_LOGGED)
                self.assertEqual(got[3].dtype, torch.int8)
                self.assertEqual(tuple(got[3].shape), (n, D))
            del x, u, w, b

    def test_impl_layer_list(self) -> None:
        """DLRM_HSTU_FP8_MASKGEN_PK_LAYERS=1: layer 1 (and untagged calls) run _pk, layers 0 / 2 the shipped
        kernel, and every call writes the shipped kernel's bytes."""
        n = (1 << 20) + 8 * 3 + 5
        x, u, w, b = _inputs(n)
        shipped = self.thl._ln_mul_dropout_fwd_maskgen

        class _Counting:
            """Counts the shipped kernel's launches and forwards them unchanged."""

            def __init__(self) -> None:
                self.n = 0

            def __getitem__(self, grid):
                launch = shipped[grid]

                def run(*args, **kwargs):
                    self.n += 1
                    return launch(*args, **kwargs)

                return run

        with mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_MASKGEN_PK_LAYERS": "1"}):
            ref, _ = self._impl(x, u, w, b, SEEDS[1], INV, "0", tag=1)
            # (tag, _pk runs, shipped launches): an untagged call runs both (the warmup compiles both kernels)
            for tag, pk, n_shipped in ((0, False, 1), (1, True, 0), (2, False, 1), (-1, True, 1)):
                counter = _Counting()
                with mock.patch.object(self.thl, "_ln_mul_dropout_fwd_maskgen", counter):
                    got, out = self._impl(x, u, w, b, SEEDS[1], INV, "1", tag=tag)
                self.assertEqual(_diff4(got, ref), dict(mask=0, y=0, mean=0, rstd=0), f"tag={tag}")
                self.assertEqual(counter.n, n_shipped, f"tag={tag}")
                self.assertIn("armed OK: LN*U*dropout fwd maskgen ACTIVE", out)  # (28)(e) armed either way
                self.assertEqual("maskgen pk ACTIVE" in out, pk, f"tag={tag}")
                self.assertEqual("maskgen pk layer-gated" in out, tag >= 0 and not pk, f"tag={tag}")
                self.assertNotIn("maskgen pk declined", out)
                self.assertEqual(self.thl._LNMD_FWD_MASKGEN_PK_ARMED_LOGGED, pk, f"tag={tag}")
            # without a list an untagged call runs _pk only (the pre-list behaviour)
            os.environ.pop("DLRM_HSTU_FP8_MASKGEN_PK_LAYERS")
            counter = _Counting()
            with mock.patch.object(self.thl, "_ln_mul_dropout_fwd_maskgen", counter):
                got, out = self._impl(x, u, w, b, SEEDS[1], INV, "1", tag=-1)
            self.assertEqual(_diff4(got, ref), dict(mask=0, y=0, mean=0, rstd=0))
            self.assertEqual(counter.n, 0)
            self.assertIn("maskgen pk ACTIVE", out)
        del x, u, w, b

    def test_impl_inv_declines(self) -> None:
        n = (1 << 20) + 5
        x, u, w, b = _inputs(n)
        for inv in (1.0e39, math.inf):
            ref, _ = self._impl(x, u, w, b, SEEDS[0], inv, "0")
            got, out = self._impl(x, u, w, b, SEEDS[0], inv, "1")
            self.assertEqual(_diff4(got, ref), dict(mask=0, y=0, mean=0, rstd=0), f"inv={inv}")
            self.assertIn("LN*U*dropout fwd maskgen pk declined", out)
            self.assertNotIn("maskgen pk ACTIVE", out)  # never claims ACTIVE for a declined call
            self.assertFalse(self.thl._LNMD_FWD_MASKGEN_PK_ARMED_LOGGED)

    def test_impl_launch_failure_latches(self) -> None:
        n = (1 << 20) + 8
        x, u, w, b = _inputs(n)
        ref, _ = self._impl(x, u, w, b, SEEDS[2], INV, "0")

        class _Raising:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    raise RuntimeError("synthetic compile failure")

                return launch

        with mock.patch.object(self.thl, "_ln_mul_dropout_fwd_maskgen_pk", _Raising()):
            got, out = self._impl(x, u, w, b, SEEDS[2], INV, "1")
            self.assertEqual(_diff4(got, ref), dict(mask=0, y=0, mean=0, rstd=0))
            self.assertIn("LN*U*dropout fwd maskgen pk DISABLED", out)
            self.assertNotIn("maskgen pk ACTIVE", out)
            self.assertIsNotNone(self.thl._LNMD_FWD_MASKGEN_PK_BROKEN)
            # latched: the next call does not try again and prints nothing new
            os.environ["DLRM_HSTU_FP8_MASKGEN_PK"] = "1"
            buf = io.StringIO()
            f = self.thl._triton_layer_norm_mul_dropout_fwd_impl
            with contextlib.redirect_stdout(buf):
                again = f(x, u, w, b, EPS, P_DROP, True, True, True, True, "silu", SEEDS[2],
                          torch.empty(0, dtype=torch.int64, device="cuda"), True, INV)
            torch.cuda.synchronize()
            self.assertEqual(_diff4(again, ref), dict(mask=0, y=0, mean=0, rstd=0))
            self.assertNotIn("maskgen pk", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

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
"""DO8-ONLY chain test (gin item (88) (DO8-ONLY); B4 v2, r16 iteration 43, R15's review N4): r16's REAL producer through
R15's REAL consumer with the IN-MODEL lifetimes. keep_refs=False drops every Python reference to the attention output
`out` and to the carried view `attn` after the forward (as in the model once hstu_compute's locals die), so
registry.plan() decides from ctx.saved_tensors[0] and the site's weakrefs alone. CPU only; runs wherever the tree's
CPU tests run (PYTHONPATH=<tree>/recommendation, nothing outside the tree).

Ported from R15 iteration 54 do8review tests/do8only_producer_adversarial_test.py
(test_only_equals_plus_twin_real_producer) and its harness (R15 iteration 53 do8v21
tests/hstu_do8_registry_v2_test.py: env, C8_ENV, StubKernel, stub_quantize, _MhaBase).

REAL: l11_do8.site_of / emission_site / run_e5_plus (_plan_only, poison, cancel, latches, _ONLY_CALL, the P1 tag) /
publish (_publish_only); hstu_do8_registry; hstu_flydsl_mha's forward gate (c8_do_blk_planned + stash_plan), the C8
guard and try_flydsl_attention_bwd up to the kernel call; an attention Function stand-in with bind, discard not_taken,
M-E1-style re-runs on the same dO and release in a finally. STUBBED: the DO8 launch (launch_e5_plus: 'plus' writes the
real bf16 dO, 'only' leaves the placeholder bytes 0x7FA5 = "unwritten"; both write the stub quantizer's dO8/E8 of the
real dO), L11's operand checks and dW/dB finalize, the FlyDSL kernel (its outputs are a pure function of the dO8 bytes
+ E8 on the C8 path, of the bf16 dO values otherwise, so a placeholder that reached it shows up as a byte difference)
and the quantizer.
"""

import ast
import contextlib
import io
import os
import unittest
from unittest import mock

import torch

from generative_recommenders.ops.flydsl import hstu_do8_registry as reg
from generative_recommenders.ops.hip import l11_do8 as do8
from generative_recommenders.ops.hip import l11_outproj_lnmd as l11

try:
    from generative_recommenders.ops.flydsl import hstu_do_fp8
    from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
except Exception as error:  # noqa: BLE001 - the chain needs the real consumer module (triton importable)
    hstu_do_fp8 = mha = None
    MHA_ERR = repr(error)
else:
    MHA_ERR = ""

FP8, BF16, E5 = torch.float8_e4m3fn, torch.bfloat16, torch.float8_e5m2
H2, D2 = 4, 128
DW = H2 * D2  # 512 = l11.D
DU_ROW, P_DROP = 2048, 0.1
UNWRITTEN = 0x7FA5  # the stub's "never written" placeholder bytes (a bf16 NaN)
GR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(reg.__file__))))
DEBUG_ENVS = ("DLRM_HSTU_FP8_BWD_FULL_STATS", "DLRM_HSTU_FP8_BWD_DUMP", "DLRM_HSTU_FP8_BWD_DUMP_CALLS",
              "HSTU_ATTN_BWD_CHECK")
C8_ENV = {
    "HSTU_ATTN_BWD": "flydsl",
    "DLRM_HSTU_FP8_BWD_FULL": "1",
    "DLRM_HSTU_FP8_BWD_QK_MFMA": "fp8",
    "DLRM_HSTU_FP8_BWD_K128": None,
    "DLRM_HSTU_FP8_BWD_FULL_HYBRID": None,
    "DLRM_HSTU_FP8_BWD_FULL_DOBLK": None,
    "DLRM_HSTU_FP8_BWD_FULL_SAT": None,
    "DLRM_HSTU_FP8_BWD_DQ_SPLIT": None,
    "DLRM_HSTU_FP8_BWD_DKDV_E5M2": None,
    "DLRM_HSTU_FP8_BWD_CENSUS": None,
    "DLRM_E1_BWD_CENSUS": None,
    **{k: None for k in DEBUG_ENVS},
}
STUB = {"quant": 0}


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def offsets(lens):
    o = [0]
    for n in lens:
        o.append(o[-1] + n)
    return torch.tensor(o, dtype=torch.int64)


def shipped():
    path = os.path.join(GR, "ops", "flydsl", "hstu_bwd_dkdv.py")
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SHIPPED" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no SHIPPED in ops/flydsl/hstu_bwd_dkdv.py")


def stub_quantize(dout, seq_offsets, N):
    """Deterministic stand-in of quantize_do_e4m3_blk (same dtypes / shapes)."""
    f = dout.float()
    STUB["quant"] += 1
    return (f * 64.0).clamp(-448, 448).to(FP8), (f.abs().amax(-1) * 1e4).clamp(0, 255).to(torch.uint8)


class StubKernel:
    """The FlyDSL dk/dv kernel stand-in: C8 (dO8 bytes + E8) and bf16-dO calls, outputs a pure function of them."""

    def __init__(self):
        self.SHIPPED = shipped()
        self.compiled = 0
        self.calls = []

    def dq_acc_dtype(self):
        return torch.bfloat16

    def cvt_scale_is_exact(self, s):
        return False

    def compile_hstu_bwd_dkdv(self, heads, dim, alpha, **kw):
        self.compiled += 1
        kernel = self

        class Fn:
            BN = 128

            def __call__(self, q, k, v, dout, dk, dv, dq, cu, batch, n_blocks, inv_n, qkv_scale, **kw2):
                ds = kw2.get("do_scale")
                kernel.calls.append((dout, ds))
                b = dout.float() if dout.dtype is BF16 else dout.view(torch.uint8).to(torch.float32)
                dk.copy_(b.to(BF16))
                dv.copy_((b * 0.5 + (0 if ds is None else ds.to(torch.float32).unsqueeze(-1))).to(BF16))
                dq.copy_((b * 0.25).to(BF16))
        return Fn()


class _AttnFn(torch.autograd.Function):
    """The attention Function stand-in: the REAL forward gate + site; in the backward the REAL bind /
    try_flydsl_attention_bwd / discard not_taken / M-E1-style re-runs on the same dO / release in a finally; zero-
    initialised dq/dk/dv; when FlyDSL does not serve the call, a deterministic Triton-fallback stand-in that READS the
    bf16 dO (so twins compare defined values)."""

    @staticmethod
    def forward(ctx, x, st):
        q, k, v, offs, N = st["q"], st["k"], st["v"], st["offs"], st["N"]
        out = x.detach().view(-1, H2, D2).clone()
        ctx.st = st
        ctx.do8_site = None
        if (reg.sites_enabled() and bool(ctx.needs_input_grad[0])
                and mha.c8_do_blk_planned(q=q, k=k, v=v, out=out, num_targets=None, max_attn_len=0,
                                          contextual_seq_len=0, num_softmax_heads=0, qkv_scale=st["s"])):
            ctx.do8_site = reg.new_site(out, offs, N)
        return out

    @staticmethod
    def backward(ctx, dout):
        st = ctx.st
        site = ctx.do8_site
        snaps = st.setdefault("snaps", [])
        try:
            reg.bind(site, dout)

            def call():
                dq = torch.zeros(dout.shape, dtype=BF16)
                dk = torch.zeros(dout.shape, dtype=BF16)
                dv = torch.zeros(dout.shape, dtype=BF16)
                served = mha.try_flydsl_attention_bwd(
                    dout=dout, q=st["q"], k=st["k"], v=st["v"], dq=dq, dk=dk, dv=dv,
                    seq_offsets=st["offs"], num_targets=None, N=st["N"], alpha=0.1,
                    max_attn_len=0, contextual_seq_len=0, num_softmax_heads=0, qkv_scale=st["s"])
                if not served:  # the Triton backward reads the bf16 dO
                    dk.copy_(dout)
                    dv.copy_((dout.float() * 0.5).to(BF16))
                    dq.copy_((dout.float() * 0.25).to(BF16))
                snaps.append((served, dq.clone(), dk.clone(), dv.clone()))
                return dq, dk, dv

            dq, dk, dv = call()
            reg.discard(site, "not_taken")
            for _ in range(st.get("reruns", 0)):
                dq, dk, dv = call()
            return (dq.float() + dk.float() + dv.float()).reshape(dout.shape[0], -1).to(BF16), None
        finally:
            reg.release(site)


class _SeamFn(torch.autograd.Function):
    """HSTUComputeOutputFunction stand-in around r16's REAL producer: attn saved unchanged and unpacked as
    ctx.saved_tensors[0]; ctx.do8_site = l11_do8.site_of(attn) in the forward; emission_site + run_e5_plus(x=attn) in
    the backward; None -> L11's own E5 launch writes the bf16 d(attn); else publish after the last write."""

    @staticmethod
    def forward(ctx, attn, slot):
        ctx.save_for_backward(attn)
        ctx.do8_site = do8.site_of(attn)
        ctx.slot = slot
        return attn.float() * 1.0

    @staticmethod
    def backward(ctx, g):
        s = ctx.slot
        attn = ctx.saved_tensors[0]
        site = do8.emission_site(ctx, attn)
        out = run_e5_plus_stubbed(attn, site, s) if site is not None else None
        s["ran_do8"] = out is not None
        if out is None:
            dattn = torch.empty_like(attn)
            dattn.copy_(s["dO"])
            pair = None
        else:
            dattn = out[0]
            pair = (ctx.do8_site, out[5], out[6])
        if pair is not None:
            do8.publish(pair[0], dattn, pair[1], pair[2])
        return dattn, None


def run_e5_plus_stubbed(attn, site, s):
    """l11_do8.run_e5_plus as the seam calls it; the launch, operand checks and finalize stubbed."""
    T = int(attn.shape[0])
    dO = s["dO"]
    s.setdefault("launches", [])

    def fake_launch(g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, dz8, wl, cnt, cap, d8, e8,
                    only=False, dx=None):
        s["launches"].append(only)
        dx_ = torch.empty_like(x) if dx is None else dx
        if only:
            dx_.view(torch.int16).fill_(UNWRITTEN)  # EM = 2: the bf16 d(attn) store is skipped
        else:
            dx_.copy_(dO)  # EM = 1: the bf16 d(attn) is written
        q8, qe8 = stub_quantize(dO.view(-1, H2, D2), None, None)
        STUB["quant"] -= 1  # the emission is not a consumer quantizer call
        d8.copy_(q8.reshape(d8.shape))
        e8.copy_(qe8)
        z = torch.zeros(8, DW)
        return dx_, z, z.clone(), z.clone(), torch.zeros(8), 8

    wl = (torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 20)
    with mock.patch.object(l11, "_operands_ok", return_value=None), \
            mock.patch.object(l11, "e5_decline_reason", return_value=None), \
            mock.patch.object(l11, "dwdb_finalize", side_effect=lambda *a: None), \
            mock.patch.object(do8, "_lib", object()), \
            mock.patch.object(do8, "work_list", return_value=wl), \
            mock.patch.object(do8, "launch_e5_plus", side_effect=fake_launch), \
            mock.patch("generative_recommenders.ops.triton.bwd_fp8.grad_dtype_and_limit", return_value=(E5, 57344.0)), \
            mock.patch("generative_recommenders.ops.triton.bwd_fp8.uvqk_tm_pad", return_value=128):
        return do8.run_e5_plus(torch.empty(T, DW, dtype=E5), "W8", "SA", "SB", attn, torch.zeros(T, DW, dtype=BF16),
                               torch.ones(DW, dtype=BF16), torch.zeros(DW, dtype=BF16), torch.zeros(T), torch.ones(T),
                               DW, 2, P_DROP, True, torch.ones(T, DW, dtype=torch.int8), DU_ROW, 0, torch.ones(1),
                               torch.ones(1), None, site)


def bits(t):
    return t.detach().contiguous().view(torch.int16) if t.dtype in (BF16, torch.float16) else \
        t.detach().contiguous().view(torch.uint8)


@unittest.skipIf(mha is None, f"hstu_flydsl_mha not importable: {MHA_ERR}")
class DO8OnlyChainTest(unittest.TestCase):
    ENVS = (reg.KNOB_ENV, reg.TAKE_ENV, reg.ONLY_ENV, reg.POISON_ENV, l11.VD_KNOB, do8.KNOB + "_SRC",
            do8.KNOB + "_FAIL_CLOSED", "RANK")

    def setUp(self) -> None:
        reg.reset_for_testing()
        self._env = {k: os.environ.pop(k, None) for k in self.ENVS}
        os.environ.update({reg.KNOB_ENV: "1", l11.VD_KNOB: "1", "RANK": "0"})
        self.out = io.StringIO()
        self._redirect = contextlib.redirect_stdout(self.out)
        self._redirect.__enter__()
        self.kernel = StubKernel()
        self._saved = (mha.unsupported_hardware_reason, mha._load_kernel, mha._host_ok, mha._targets_max_memo,
                       hstu_do_fp8.quantize_do_e4m3_blk, mha.try_flydsl_attention_bwd)
        mha.unsupported_hardware_reason = lambda: None
        mha._load_kernel = lambda: self.kernel
        mha._host_ok = True
        mha._targets_max_memo = (None, -1, 0)
        hstu_do_fp8.quantize_do_e4m3_blk = stub_quantize
        STUB["quant"] = 0
        self._cenv = env(**C8_ENV)
        self._cenv.__enter__()
        self.fresh()
        l11.reset_for_testing()
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        self.seen = []  # per consumer call: (dout all 0x7FC0, dout all UNWRITTEN, any NaN)
        orig = self._saved[-1]

        def spy(**kw):
            d = kw["dout"]
            b = d.contiguous().view(torch.int16)
            self.seen.append((bool((b == 0x7FC0).all()), bool((b == UNWRITTEN).all()),
                              bool(torch.isnan(d.float()).any())))
            return orig(**kw)

        mha.try_flydsl_attention_bwd = spy

    def tearDown(self) -> None:
        self._cenv.__exit__(None, None, None)
        (mha.unsupported_hardware_reason, mha._load_kernel, mha._host_ok, mha._targets_max_memo,
         hstu_do_fp8.quantize_do_e4m3_blk, mha.try_flydsl_attention_bwd) = self._saved
        self._redirect.__exit__(None, None, None)
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        do8.set_requested_for_testing(None)
        do8.reset_for_testing()
        l11.set_requested_for_testing(None)
        l11.reset_for_testing()
        reg.reset_for_testing()

    def fresh(self) -> None:
        reg.reset_for_testing()
        do8.reset_for_testing()
        do8.set_requested_for_testing(True)
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        if hasattr(self, "seen"):
            self.seen.clear()

    def operands(self, lens=(5, 17, 64, 1, 41)):
        T2 = sum(lens)
        g = torch.Generator().manual_seed(7)
        q = torch.randn(T2, H2, D2, generator=g).to(FP8)
        k = torch.randn(T2, H2, D2, generator=g).to(FP8)
        v = torch.randn(T2, H2, D2, generator=g).to(FP8)
        return q, k, v, offsets(lens), max(lens), T2

    def chain(self, only, layers=1, reruns=0, poison=False, keep_refs=True):
        """x -> [attention (real consumer) -> carry_site view -> _SeamFn (r16 producer)] x layers -> sum."""
        q, k, v, offs, N, T2 = self.operands()
        g = torch.Generator().manual_seed(11)
        x = torch.randn(T2, DW, generator=g).to(BF16).requires_grad_()
        sts, slots = [], []
        for _ in range(layers):
            sts.append({"q": q, "k": k, "v": v, "offs": offs, "N": N, "s": 0.0173, "reruns": reruns})
            slots.append({"dO": (torch.randn(T2, DW, generator=g) * 1e-2).to(BF16)})
        quant0 = STUB["quant"]
        with env(**{reg.ONLY_ENV: "1" if only else None, reg.POISON_ENV: "1" if poison else None}):
            def fwd():
                h = x
                keep = []
                for li in range(layers):
                    out = _AttnFn.apply(h, sts[li])
                    attn = reg.carry_site(out, out.view(-1, DW))
                    y = _SeamFn.apply(attn, slots[li])
                    keep.append((out, attn))
                    h = y.to(BF16)
                return h, keep  # `out` / `attn` locals die here

            y, keep = fwd()
            if not keep_refs:
                keep = None  # in-model: nothing but autograd holds out / attn
            y.float().sum().backward()
        return {"x_grad": x.grad, "snaps": [s for st in sts for s in st.get("snaps", [])], "slots": slots,
                "quant": STUB["quant"] - quant0, "keep": keep}

    def twin(self, a, b, msg=""):
        self.assertTrue(torch.equal(bits(a["x_grad"]), bits(b["x_grad"])), msg)
        self.assertEqual(len(a["snaps"]), len(b["snaps"]), msg)
        for sa, sb in zip(a["snaps"], b["snaps"]):
            self.assertEqual(sa[0], sb[0], msg)
            for ta, tb in zip(sa[1:], sb[1:]):
                self.assertTrue(torch.equal(bits(ta), bits(tb)), msg)

    def test_only_equals_plus_twin_in_model_lifetimes(self) -> None:
        """R15 iter54 test_only_equals_plus_twin_real_producer, keep_refs=False (the in-model case, R15's N4) and True:
        'only' (POISON off / on) == 'plus' bit for bit through the real consumer, 1-2 layers, 0/1/3 M-E1 re-runs;
        plan() is asked with ctx.saved_tensors[0], which IS the site's carried view (identity and weakrefs intact)
        although the forward's Python references are gone; the consumer saw the placeholder and never quantized."""
        asked = []
        orig_plan = reg.plan

        def plan_spy(site, attn=None):
            view = site.view_ref() if getattr(site, "view_ref", None) is not None else None
            out = site.out_ref() if getattr(site, "out_ref", None) is not None else None
            answer = orig_plan(site, attn)
            asked.append((attn is view and view is not None, out is not None, answer))
            return answer

        with mock.patch.object(reg, "plan", side_effect=plan_spy):
            for keep_refs in (False, True):
                for layers in (1, 2):
                    for reruns in (0, 1, 3):
                        tag = f"layers={layers} reruns={reruns} keep_refs={keep_refs}"
                        self.fresh()
                        plus = self.chain(False, layers, reruns, keep_refs=keep_refs)
                        self.assertTrue(all(sl["launches"] == [False] for sl in plus["slots"]), tag)
                        for poison in (False, True):
                            self.fresh()
                            asked.clear()
                            o = self.chain(True, layers, reruns, poison=poison, keep_refs=keep_refs)
                            t = f"{tag} poison={poison}"
                            self.twin(plus, o, t)
                            self.assertEqual(asked, [(True, True, "only")] * layers, t)
                            self.assertIs(o["keep"] is None, not keep_refs, t)
                            self.assertTrue(all(sl["launches"] == [True] for sl in o["slots"]), t)
                            s = reg.stats()
                            c = do8.only_counters()
                            self.assertEqual(s["only"]["forced_plus"], {}, t)
                            self.assertEqual((s["only"]["planned"], s["only"]["published"], s["only"]["taken"],
                                              s["only"]["retaken"], s["only"]["released"], s["only"]["cancelled"],
                                              s["only"]["released_untaken"]),
                                             (layers, layers, layers, layers * reruns, layers, 0, 0), t)
                            self.assertEqual((c["planned"], c["only_launched"], c["published"], c["cancelled"],
                                              c["fallback"], c["plan_plus"], c["latched"]),
                                             (layers, layers, layers, 0, 0, 0, 0), t)
                            self.assertEqual(o["quant"], 0, t)  # no consumer quantizer call at all
                            self.assertEqual(do8._ONLY_CALL, {}, t)
                            self.assertEqual(len(self.seen), layers * (1 + reruns), t)
                            self.assertTrue(all(sv[0] if poison else sv[1] for sv in self.seen), (t, self.seen))
                            self.assertTrue(torch.isfinite(o["x_grad"].float()).all(), t)


if __name__ == "__main__":
    unittest.main()

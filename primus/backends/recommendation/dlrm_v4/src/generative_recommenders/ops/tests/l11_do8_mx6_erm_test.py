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
"""MX6 (gin item (104)) beside r14's E-RM row (gin item (97), ops/triton/dy_erm.py) through E-RM's REAL consumer and
W3's REAL row-major entry: one simulated training step of the two out-projections at a time, in the STUStack order.
l11_do8_mx6_seam_test.ErmCompositionTest stubs the cast and W3 and uses a record that is never consumed; here:
  forward   layer 0 then layer 1: the REAL _l11_mx6_forward_tag and _erm_forward_tag (the real dy_erm.make_site), then
            the REAL dy_erm.claim (layer 1's preprocess forward);
  backward  the REAL HSTUComputeOutputFunction.backward of the TOP call (layer 1), then the REAL E-RM producer
            (dy_erm.run_producer: ticket + produce; only its _ln_bwd_erm launch faked), then the REAL backward of the
            BOTTOM call (layer 0).
Inside both calls these run for real: the MX6 seam, the shipped block, bwd_fp8.output_projection_dgrad_operands ->
_quantize_gradient_and_weight_fused -> dy_erm.consumable / dy_erm.finish (only the Triton casts faked), the E-RM W3
wrappers and wgrad_ck_w3.run_w3_rm with its BITWISE guard vs x21 (only the xt2 launch and x21's run_w3 faked), with one
scale state per out-proj weight (bwd_fp8 keys them by weight.data_ptr()). The DO8 / dO8 registry / L11 / MX6-launch
stand-ins are the seam test's harness (l11_do8_mx6_seam_test._BackwardBase).
  SHIPPING FORM   E-RM requested, sites=top: the top is MX6 with no _erm_take, the bottom is E-RM's (one take, the
                  real consumer, x_rm to the row-major entry, dW = xt2's); E-RM's counters and lines; no MX6 line
                  in either call (the top's MX6 record is counted only: MX6's lines print at its prebuild).
  KNOB-OFF BOTTOM the same step with MX6 off: the bottom's events, cast keywords, E-RM counters / lines, W3 calls and
                  dW are identical (MX6 at the top moves nothing E-RM reads).
  72925           the xt2 entry's first-call bitwise guard rejects in the bottom call: that call's dW is x21's result on
                  the re-tiled x_rm, the row-major entry latches (E-RM stays enabled), the next step's ticket declines
                  and the bottom runs the dual cast; MX6 at the top is untouched in both steps.
  W3 OPERAND      S1 returning a step-tiled dout_t that is not 16-B aligned is a contract break at the top (fail
                  closed) BEFORE x21 sees it: W3's step tiling stays on and E-RM is consumed at the bottom.
CPU only (no kernel, no device)."""

import contextlib
import os
import re
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import torch

from generative_recommenders.ops.hip import l11_do8 as do8
from generative_recommenders.ops.hip import l11_outproj_lnmd as l11
from generative_recommenders.ops.tests import l11_do8_mx6_seam_test as seam
from generative_recommenders.ops.triton import bwd_fp8, dy_erm, hstu_du_e5m2, wgrad_ck_w3
from generative_recommenders.ops.triton import fp8_wgrad_snr as snr
from generative_recommenders.ops.triton import triton_layer_norm as tln

E4, E5, BF, F32, U8, I8 = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float32, torch.uint8, torch.int8
D, NOUT, T, DU_ROW = seam.D, seam.NOUT, seam.T, seam.DU_ROW
KT = (T + 127) // 128
CUDA0 = torch.device("cuda", 0)  # a device object only (W3's per-device state at index 0)
REAL_CAST = bwd_fp8.output_projection_dgrad_operands
REAL_RM_INACTIVE = wgrad_ck_w3.rm_inactive_reason
REAL_RM_DECLINE = wgrad_ck_w3.rm_decline_reason
REAL_RUN_W3 = wgrad_ck_w3.run_w3
REAL_DECLINE_W3 = wgrad_ck_w3._decline_reason
ERM_BAD = ("erm NOT used", "erm record dropped", "erm DISABLED")
MX6_ORDER = seam.MX6_ORDER


class _State:
    """A published scale-site state stand-in, one per out-proj weight (bwd_fp8 keys the real ones by data_ptr)."""

    def __init__(self, name: str, calls: int = 50) -> None:
        self.site = name
        self.published = True
        self.calls = calls
        self.t_calls = 0
        self.t_shift = 7
        self.inv_scale = torch.full((1,), 2.0**20)
        self.scale = torch.full((1,), 2.0**-20)
        self.t_clipped_windows = torch.zeros((), dtype=torch.int64)
        self.publishes = 0

    def publish(self) -> None:
        self.publishes += 1


class _StepBase(seam._BackwardBase):
    """The seam harness plus E-RM's real protocol and the real cast / W3 wrappers (module docstring)."""

    def setUp(self) -> None:
        super().setUp()  # the seam harness: env, registry, DO8 / L11 / MX6 states (MX6 armed, sites "top")
        self.REAL_ERM_TAKE = self.thl._erm_take
        self.REAL_ERM_DOUT_RM = self.thl._erm_dout_rm
        self.REAL_ERM_W3 = self.thl._erm_out_wgrad_nn_ck_w3
        self.REAL_OUT_W3 = self.thl._out_wgrad_nn_ck_w3
        dy_erm.reset_for_testing()
        dy_erm.set_requested_for_testing(True)
        self.addCleanup(dy_erm.reset_for_testing)
        self.addCleanup(dy_erm.set_requested_for_testing, None)
        self.erm_logs: List[str] = []
        self.w3_logs: List[str] = []
        self.w3_disabled: List[str] = []
        self.rm_launches: List[Tuple[Any, ...]] = []
        self.x21_calls: List[Tuple[Any, ...]] = []
        self.xt2_result = torch.arange(D * NOUT, dtype=F32).view(D, NOUT) * 2.0**-10
        self.x21_result: Optional[torch.Tensor] = self.xt2_result.clone()  # bitwise equal: the guard arms
        self.dual_outs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.fsl_partials: List[torch.Tensor] = []
        self.take_sites: List[Any] = []
        self.erm_dout_rm: List[Tuple[int, bool]] = []
        self.erm_w3: List[Tuple[int, Any, Any]] = []
        self.w3_shipped: List[Tuple[Any, Any]] = []
        self.tiled_live = False  # True: the dout_tiled stand-in follows W3's process-wide tiling latch
        self.W0 = torch.nn.Parameter(torch.randn(NOUT, D) * 0.02)  # layer 0 (bottom): E-RM's site
        self.W1 = torch.nn.Parameter(torch.randn(NOUT, D) * 0.02)  # layer 1 (top)
        self.states = {self.W0.data_ptr(): _State("output_projection.L0"),
                       self.W1.data_ptr(): _State("output_projection.L1")}
        self.by_name = {f"output_projection.{p}": s for p, s in self.states.items()}
        self._keeps: List[Any] = []
        os.environ[do8.ONLY_KNOB] = "1"
        st = contextlib.ExitStack()
        self.addCleanup(st.close)
        W = wgrad_ck_w3
        for target, attr, kw in (
            # E-RM (dy_erm): requested (above); prerequisites configured; prebuilt; lines captured
            (dy_erm, "_prebuilt", {"new": True}),
            (dy_erm, "_inert_checked", {"new": True}),
            (dy_erm, "_inert", {"new": None}),
            (dy_erm, "_atexit_registered", {"new": True}),
            (dy_erm, "_bwd", {"new": None}), (dy_erm, "_w3", {"new": None}),
            (dy_erm, "_tln", {"new": None}), (dy_erm, "_snr", {"new": None}),
            (dy_erm, "_stream", {"new": lambda device: "S0"}),
            (dy_erm, "_print", {"new": lambda m: self.erm_logs.append("[hstu-fp8-bwd] " + m)}),
            (dy_erm, "_arm_print", {"new": lambda m: self.erm_logs.append("[hstu-fp8-bwd] " + m)}),
            # bwd_fp8: the shipped configuration's flags, per-weight published scale states, faked Triton casts
            (bwd_fp8, "scale_controller_enabled", {"return_value": True}),
            (bwd_fp8, "_BWD_ENABLED", {"new": True}),
            (bwd_fp8, "_OUT_WGRAD_ARMED", {"new": True}),
            (bwd_fp8, "_OUT_WGRAD_T_E4M3", {"new": True}),
            (bwd_fp8, "_BWD_HELPERS_FUSED", {"new": True}),
            (bwd_fp8, "_GRAD_DTYPE", {"new": E5}),
            (bwd_fp8, "_GRAD_MAX", {"new": bwd_fp8._E5M2_MAX}),
            (bwd_fp8, "_OUT_WGRAD_DFMT_SHIFT_AUTO", {"new": False}),
            (bwd_fp8, "_peek_site_state", {"new": lambda site, key: self.states.get(key)}),
            (bwd_fp8, "_site_name", {"new": lambda site, disc: f"{site}.{disc}"}),
            (bwd_fp8, "_site_state", {"new": lambda name, device: self.by_name[name]}),
            (bwd_fp8, "quantize_row_major_dual", {"new": self._dual}),
            (bwd_fp8, "_fused_site_launch", {"new": self._fsl}),
            (bwd_fp8, "_fused_weight_quant", {"new": lambda w, wp: (torch.zeros(NOUT, D, dtype=E4),
                                                                    torch.full((1,), 2.0**-12))}),
            (bwd_fp8, "quantize_gradient", {"new": mock.Mock(side_effect=AssertionError("warm-up cast reached"))}),
            (bwd_fp8, "record_engaged", {"new": lambda site: None}),
            (tln, "flydsl_ln_bwd_decline_reason", {"new": lambda *a, **k: None}),
            (snr, "SNR_ENABLED", {"new": False}),
            # W3 (wgrad_ck_w3): x21 armed on cuda:0, the xt2 entry bound, latches clear; lines captured
            (W, "_override", {"new": True}), (W, "_tiled_override", {"new": True}),
            (W, "_t256s_override", {"new": True}), (W, "_rm_override", {"new": True}),
            (W, "_broken", {"new": None}), (W, "_tiled_declined", {"new": None}), (W, "_rm_declined", {"new": None}),
            (W, "_tiled_ok", {"new": {0}}), (W, "_armed_s", {"new": {0}}), (W, "_armed_rm", {"new": set()}),
            (W, "_armed", {"new": {0}}), (W, "_armed_t", {"new": {0}}),
            (W, "_fn_rm", {"new": lambda *a: 0}), (W, "_once", {"new": set()}),
            (W, "_disable", {"new": self.w3_disabled.append}),
            (W, "log", {"new": self.w3_logs.append}),
            (W, "rm_inactive_reason", {"new": lambda device: REAL_RM_INACTIVE(CUDA0)}),
            (W, "rm_decline_reason", {"new": self._rm_decline_reason}),
            (W, "_index", {"new": lambda device: 0}),
            (W, "_launch_rm", {"new": self._launch_rm}),
            (W, "run_w3", {"new": self._run_x21}),
            # the forward tags' inputs (E-RM's FP8 NN wgrad flags)
            (self.thl, "_FP8_OUT_WGRAD", {"new": True}),
            (self.thl, "_FP8_OUT_WGRAD_IMPL", {"new": "nn"}),
            (self.thl, "_fp8_out_wgrad_logged", {"new": True}),
        ):
            st.enter_context(mock.patch.object(target, attr, **kw))

    def tearDown(self) -> None:
        os.environ.pop(do8.ONLY_KNOB, None)
        super().tearDown()

    def fresh_step_state(self) -> None:
        """Per case: the seam harness's _fresh, E-RM reset and prebuilt (every prerequisite configured), W3's
        row-major arming cleared, both scale states at 50 calls."""
        self._fresh()
        dy_erm.reset_for_testing()
        dy_erm._inert_checked = True
        dy_erm._inert = None
        dy_erm.mark_prebuilt()
        wgrad_ck_w3._armed_rm.clear()
        for s in self.states.values():
            s.calls = 50

    # ------------------------------------------------------------------ fakes (device work only)
    def _dual(self, gradient, inv_scale, t_dtype=None, t_inv_mul=1.0, t_limit=None, return_partial=False,
              t_tiled=False):  # pyre-ignore[2,3]
        self.events.append("dual_cast")
        m = int(gradient.shape[0])
        q = torch.zeros(m, D, dtype=E5)
        q_t = torch.zeros(KT, D, 128, dtype=E4) if t_tiled else torch.zeros(D, (m + 15) // 16 * 16, dtype=E4)
        self.dual_outs.append((q, q_t))
        return q, q_t, torch.zeros(8)

    def _fsl(self, state, partial, weight, snapshot, t_note, t_shift, snapshot_t):  # pyre-ignore[2,3]
        self.fsl_partials.append(partial)
        return torch.zeros(8), torch.full((1,), 2.0**-24), torch.full((1,), 2.0**-17)

    def _rm_decline_reason(self, y8, x_rm, scale_y, scale_d, m):  # pyre-ignore[2,3]
        why = REAL_RM_DECLINE(y8, x_rm, scale_y, scale_d, m)
        return None if why is not None and why[0] == "device" else why  # CPU tensors stand for cuda:0

    def _launch_rm(self, y8, x_rm, scale_y, scale_d, m, index):  # pyre-ignore[2,3]
        self.events.append("xt2_launch")
        self.rm_launches.append((x_rm, scale_d, int(m)))
        return self.xt2_result.clone()

    def _run_x21(self, y8, dout_t, scale_y, scale_d, m, reference):  # pyre-ignore[2,3]
        self.events.append("x21")
        self.x21_calls.append((dout_t, scale_d, int(m)))
        return None if self.x21_result is None else self.x21_result.clone()

    def _w3(self, y_fp8, dout_fp8, y_scale, dout_scale, fn):  # pyre-ignore[2,3]: the shipped W3 entry (non-E-RM)
        self.events.append("w3")
        self.w3_operand = (dout_fp8, dout_scale)
        self.w3_shipped.append((dout_fp8, dout_scale))
        return torch.zeros(D, NOUT), "nn_ck"

    # ------------------------------------------------------------------ one call, one step
    def ctx_for(self, site: Any, attn: torch.Tensor, W: torch.nn.Parameter) -> SimpleNamespace:
        mask = torch.ones(T, D, dtype=I8)
        saved = (attn, torch.randn(T, D).to(BF), torch.ones(D, dtype=BF), torch.zeros(D, dtype=BF), torch.zeros(T),
                 torch.ones(T), W, torch.zeros(T, NOUT, dtype=E4), torch.ones(1), mask)
        return SimpleNamespace(
            saved_tensors=saved, recompute_y_in_backward=True, fp8_y_saved=True, has_random_mask=True, BLOCK_D=D,
            num_warps=2, eps=1e-6, seed=0, training=True, concat_u=True, concat_x=True, dropout_ratio=0.1, num_heads=1,
            linear_dim=-1, group_norm=False, silu_u=True, mul_u_activation_type="silu", fp8_forward=True,
            du_row_width=DU_ROW, du_e5m2_site=hstu_du_e5m2.DuE5m2Site(4242), do8_site=site, fp8_y_token_major=False,
            fp8_dout_token_major=True,
        )

    def forward_tags(self, ctx: SimpleNamespace, layer: int) -> torch.Tensor:
        """The two forward hooks of one compute output at STUStack layer `layer`, as the real forward runs them."""
        thl = self.thl
        out = torch.zeros(8, D, requires_grad=True)
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch.object(thl, "_LNMD_LAYER_TAG", layer))
            st.enter_context(mock.patch.object(bwd_fp8, "_helpers_fused_ok", return_value=True))
            st.enter_context(seam._Capture())
            ctx.du_e5m2_site = thl._l11_mx6_forward_tag(ctx, ctx.du_e5m2_site)  # the B1a site line
            ret = thl._erm_forward_tag(ctx, out, ctx.saved_tensors[6], True)  # the forward's return
        self.assertIs(ret, out)
        return out

    def run_call(self, ctx: SimpleNamespace, dout: torch.Tensor) -> Tuple[Any, str]:
        thl = self.thl
        self.state = self.states[ctx.saved_tensors[6].data_ptr()]  # the harness's S1 / rescue stand-ins bump it
        wl = (torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 20)
        real_take = dy_erm.take

        def work_list(*a: Any, **k: Any) -> Any:
            self.events.append("work_list")
            return wl

        def finalize(*a: Any, **k: Any) -> None:
            self.events.append("dwdb_finalize")

        def take_spy(site: Any) -> Any:
            self.take_sites.append(site)
            return real_take(site)

        def dout_rm_spy(c: Any) -> bool:
            r = self.REAL_ERM_DOUT_RM(c)
            self.erm_dout_rm.append((id(c), r))
            return r

        def erm_w3_spy(c, y8, dfp8, ys, ds, fn):  # pyre-ignore[2,3]
            self.erm_w3.append((id(c), dfp8, ds))
            return self.REAL_ERM_W3(c, y8, dfp8, ys, ds, fn)

        with contextlib.ExitStack() as st:
            for target, attr, kw in (
                (l11, "decline_reason", {"new": self.ev("decline_reason", None)}),
                (l11, "run_e5", {"new": self.ev("l11_run_e5", self._l11_e5)}),
                (l11, "run", {"new": self.ev("l11_run", None)}),
                (l11, "dwdb_finalize", {"new": finalize}),
                (thl, "_out_wgrad_dout_tiled",
                 {"new": self.ev("dout_tiled", lambda *a: self.tiled and not (
                     self.tiled_live and wgrad_ck_w3._tiled_declined is not None))}),
                (thl, "_du_e5m2_plan", {"new": self.ev("b1a_plan", lambda *a, **k: self.plan)}),
                (thl, "_triton_layer_norm_mul_dropout_bwd_du_e5m2", {"new": self.ev("b1a_emit", self._b1a_emit)}),
                (thl, "triton_layer_norm_mul_dropout_bwd",
                 {"new": self.ev("ln_bwd", lambda **k: (torch.zeros_like(k["x"]), torch.zeros(T, D, dtype=BF),
                                                        torch.zeros(D, dtype=BF), torch.zeros(D, dtype=BF), None))}),
                (thl, "_out_wgrad_nn_ck_w3", {"new": self._w3}),
                (thl.wgrad_ck_w3, "enabled", {"return_value": True}),
                (bwd_fp8, "output_projection_dgrad_operands", {"new": self.ev("fp8_cast", REAL_CAST)}),
                (bwd_fp8, "output_projection_dgrad_gemm",
                 {"new": self.ev("gemm", lambda *a: torch.zeros(T, NOUT, dtype=BF))}),
                (bwd_fp8, "grad_dtype_and_limit", {"return_value": (E5, 57344.0)}),
                (bwd_fp8, "uvqk_tm_pad", {"return_value": 128}),
                (bwd_fp8, "_helpers_fused_ok", {"return_value": True}),
                (bwd_fp8, "_has_stable_identity", {"return_value": True}),
                (bwd_fp8, "output_projection_dgrad_operands_mx6", {"create": True, "new": self.ev("s1", self._s1)}),
                (bwd_fp8, "output_projection_g8_rescue", {"create": True, "new": self.ev("rescue", self._rescue)}),
                (bwd_fp8, "output_projection_mx6_encode",
                 {"create": True, "new": mock.Mock(side_effect=AssertionError("encode in a training call"))}),
                (do8, "_lib", {"new": object()}),
                (do8, "launch_e5_plus", {"new": self._do8_launch}),
                (do8, "work_list", {"new": work_list}),
                (do8, "_mx6_lib", {"new": object()}),
                (do8, "_impl_mx6", {"new": seam.IMPL_MX6}),
                (do8, "_fn_launch_mx6", {"new": self._mx6_entry}),
                (do8, "_fn_drain", {"new": lambda: 0}),
                (do8, "_fn_plan", {"new": self._fn_plan}),
                (torch.cuda, "current_device", {"return_value": 0}),
                (torch.cuda, "device", {"new": lambda i: contextlib.nullcontext()}),
                (torch.cuda, "current_stream", {"new": lambda d=None: SimpleNamespace(cuda_stream=0)}),
                (thl, "_erm_take", {"new": self.ev("erm_take", self.REAL_ERM_TAKE)}),
                (thl, "_erm_dout_rm", {"new": dout_rm_spy}),
                (thl, "_erm_out_wgrad_nn_ck_w3", {"new": erm_w3_spy}),
                (dy_erm, "take", {"new": take_spy}),
            ):
                st.enter_context(mock.patch.object(target, attr, **kw))
            for target, attr, kw in self.extra:
                st.enter_context(mock.patch.object(target, attr, **kw))
            st.enter_context(mock.patch.dict(do8._plan_cache, clear=True))
            self.wrap(st, do8, "emission_site", "emission_site")
            self.wrap(st, do8, "publish", "publish")
            cap = st.enter_context(seam._Capture())
            st.enter_context(torch.no_grad())
            try:
                out: Any = thl.HSTUComputeOutputFunction.backward(ctx, dout)
            except BaseException as error:  # noqa: BLE001 -- the test inspects it
                out = error
        return out, cap.text

    def produce(self, site: Any) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor]]:
        """Layer 1's preprocess backward: the REAL dy_erm.run_producer (ticket + produce), only _ln_bwd_erm faked.
        Returns (its result, the dY the bottom out-proj receives, the producer's outputs)."""
        x = torch.zeros(T, D, dtype=BF)
        dx = torch.randn(T, D).to(BF)
        g8 = torch.zeros(T, D, dtype=E5)
        part = torch.zeros(8)
        outs: Dict[str, torch.Tensor] = {"dx": dx, "g8": g8, "part": part}

        def ln_bwd_erm(dy, x_, w, b, mean, rstd, res, inv, tshift, out=None):  # pyre-ignore[2,3]
            xrm = out["xrm"]
            xrm.view(U8).fill_(0x38)
            outs["xrm"] = xrm
            return dx, torch.zeros(D), torch.zeros(D), g8, xrm, part

        with mock.patch.object(dy_erm, "_ln_bwd_erm", lambda: ln_bwd_erm), \
                mock.patch.object(dy_erm, "_empty_x_rm", lambda m, dev: torch.empty((m, D), dtype=E4)):
            res = dy_erm.run_producer(site, torch.zeros_like(x), x, torch.ones(D, dtype=BF),
                                      torch.zeros(D, dtype=BF), torch.zeros(T), torch.ones(T), torch.zeros_like(x),
                                      False)
        # without a result the shipped LN backward runs and writes a fresh dY for layer 0
        return res, (dx if res is not None else torch.randn(T, D).to(BF)), outs

    def step(self) -> SimpleNamespace:
        """One training step of the two out-projections (module docstring)."""
        r = SimpleNamespace()
        site0, attn0 = self.graph()
        self._keeps.append(self._keep)
        r.ctx0 = self.ctx_for(site0, attn0, self.W0)
        r.out0 = self.forward_tags(r.ctx0, 0)
        site1, attn1 = self.graph()
        self._keeps.append(self._keep)
        r.ctx1 = self.ctx_for(site1, attn1, self.W1)
        self.forward_tags(r.ctx1, 1)
        r.do8_site0, r.do8_site1 = site0, site1
        r.erm_site = getattr(r.ctx0, "erm_site", None)
        r.claimed = (dy_erm.claim(r.out0, residual_proxy=True, targets_only=False)
                     if isinstance(r.erm_site, dy_erm.ErmSite) else None)
        # backward: the top call first (layer 1), then layer 1's preprocess backward (E-RM's producer), then layer 0
        self.events, self.calls = [], {}
        n_take, n_dual = len(self.take_sites), len(self.dual_outs)
        r.top, r.top_text = self.run_call(r.ctx1, torch.randn(T, D).to(BF))
        r.top_events, r.top_calls = self.events, self.calls
        r.top_takes = self.take_sites[n_take:]
        r.top_dual = self.dual_outs[n_dual:]
        r.top_w3 = list(self.w3_shipped)
        r.prod, r.dout0, r.prod_outs = self.produce(r.claimed)
        r.rec = r.erm_site.record if isinstance(r.erm_site, dy_erm.ErmSite) else None
        self.events, self.calls = [], {}
        n_take, n_dual, n_w3 = len(self.take_sites), len(self.dual_outs), len(self.w3_shipped)
        r.bot, r.bot_text = self.run_call(r.ctx0, r.dout0)
        r.bot_events, r.bot_calls = self.events, self.calls
        r.bot_takes = self.take_sites[n_take:]
        r.bot_dual = self.dual_outs[n_dual:]
        r.bot_w3 = self.w3_shipped[n_w3:]
        return r

    # ------------------------------------------------------------------ shared assertions
    def assert_top_is_mx6_without_erm(self, r: SimpleNamespace) -> None:
        self.assertIsInstance(r.top, tuple, r.top)
        self.assertEqual(r.top_events, MX6_ORDER)  # no erm_take, no fp8_cast
        self.assertEqual(r.top_takes, [])  # dy_erm.take never called at the top
        self.assertFalse(hasattr(r.ctx1, "erm_rec"))
        self.assertIsNone(getattr(r.ctx1, "erm_site", None))
        self.assertIs(r.top_w3[-1][0], self.ops6.dout_fp8)  # W3 on the producer's dout_t, shipped entry
        self.assertIs(r.top_w3[-1][1], self.ops6.dout_fp8_scale)

    def assert_bottom_is_erm(self, r: SimpleNamespace, armed_now: bool = True) -> None:
        self.assertIsInstance(r.bot, tuple, r.bot)
        self.assertEqual(r.bot_events[:3], ["decline_reason", "dout_tiled", "erm_take"])
        self.assertEqual(r.bot_events.count("erm_take"), 1)
        self.assertNotIn("dual_cast", r.bot_events)  # the producer's g8 / x_rm / partials replace the dual cast
        self.assertNotIn("s1", r.bot_events)
        self.assertEqual(r.bot_takes, [r.erm_site])  # one take, of E-RM's site
        self.assertIs(r.ctx0.erm_rec, r.rec)
        self.assertTrue(r.rec.used and r.rec.layout == "rm" and r.rec.g8 is None)
        kw = r.bot_calls["fp8_cast"][0][1]
        self.assertIs(kw["erm"], r.rec)
        self.assertEqual((kw["token_major"], kw["t_tiled"]), (True, True))
        self.assertIs(self.fsl_partials[-1], r.prod_outs["part"])  # the producer's amax partials fed the site program
        self.assertIs(self.erm_w3[-1][1], r.prod_outs["xrm"])  # the E-RM W3 wrapper got x_rm ...
        x_rm, scale_d, m = self.rm_launches[-1]
        self.assertIs(x_rm, r.prod_outs["xrm"])  # ... and the row-major entry launched on it
        self.assertEqual(m, T)
        if armed_now:
            self.assertTrue(torch.equal(r.bot[5], self.xt2_result.t().contiguous()))
        self.assertEqual(r.bot_w3, [])  # the shipped (non-E-RM) W3 entry is not called at the bottom


class TopBesideErmTest(_StepBase):
    """E-RM requested, MX6 armed with the shipping site set "top"."""

    def test_top_is_mx6_and_the_bottom_is_erms_through_its_real_consumer(self) -> None:
        r = self.step()
        self.assertEqual((getattr(r.ctx1, "l11_mx6_layer", None), getattr(r.ctx0, "l11_mx6_layer", None)), (1, None))
        self.assertIsInstance(r.erm_site, dy_erm.ErmSite)
        self.assertIs(r.claimed, r.erm_site)
        self.assertIsNotNone(r.prod)
        self.assert_top_is_mx6_without_erm(r)
        self.assert_bottom_is_erm(r)
        top_id, bot_id = id(r.ctx1), id(r.ctx0)
        self.assertIn((top_id, False), self.erm_dout_rm)  # the top's W3 wrapper takes the shipped path
        self.assertIn((bot_id, True), self.erm_dout_rm)  # the bottom's reads x_rm
        self.assertIs([e for e in self.erm_w3 if e[0] == top_id][0][1], self.ops6.dout_fp8)
        self.assertIs([e for e in self.erm_w3 if e[0] == bot_id][0][1], r.prod_outs["xrm"])
        s0, s1 = self.states[self.W0.data_ptr()], self.states[self.W1.data_ptr()]
        self.assertEqual((s1.calls, s0.calls), (51, 51))  # one bookkeeping per site, each on its own weight's state
        c = dy_erm.counters()
        self.assertEqual(c, {"sites": 1, "claimed": 1, "tickets": 1, "emitted": 1, "consumed": 1, "declined": 0})
        self.assertIsNone(dy_erm.broken())
        logs = "\n".join(self.erm_logs)
        self.assertEqual((logs.count("ARMED erm on"), logs.count("erm consumed on")), (1, 1))
        for bad in ERM_BAD:
            self.assertNotIn(bad, logs)
        self.assertEqual(sum("bitwise=x21 -> armed" in line for line in self.w3_logs), 1)
        self.assertIsNone(wgrad_ck_w3.rm_declined())
        m = do8.mx6_counters()
        self.assertEqual((m["decided"], m["launched"], m["published"], m["rescued"], m["released"], m["latched"],
                          m["declined"]), (1, 1, 1, 0, 0, 0, {}))
        self.assertNotIn("NOT armed", r.top_text + r.bot_text)
        self.assertNotIn("MX6", r.bot_text)
        # re-cut v2: the top's MX6 record is counted (above) and prints no MX6 line (ACTIVE / counters: the prebuild)
        self.assertNotIn(do8.FAMILY_MX6, r.top_text)
        self.assertNotIn("[l11-do8] mx6 counters", r.top_text)
        self.assert_no_open_plan(r.do8_site0)
        self.assert_no_open_plan(r.do8_site1)

    def test_the_bottom_is_the_knob_off_bottom(self) -> None:
        """The same step with MX6 off: the bottom's events, cast keywords, E-RM counters / lines, W3 calls and dW are
        identical (MX6 at the top moves nothing E-RM reads)."""

        def run(mx6: bool) -> Tuple[Any, ...]:
            self.fresh_step_state()
            self.arm_mx6(mx6)
            del self.erm_logs[:], self.w3_logs[:], self.rm_launches[:], self.x21_calls[:]
            r = self.step()
            kw = r.bot_calls["fp8_cast"][0][1]
            erm_lines = [re.sub(r"token \d+", "token #", m) for m in self.erm_logs if "requested" not in m]
            return (r.bot_events, sorted(kw), kw["token_major"], kw["t_tiled"], kw["erm"] is r.rec,
                    dy_erm.counters(), erm_lines,
                    [(m, x is r.prod_outs["xrm"]) for x, _, m in self.rm_launches], len(self.x21_calls),
                    list(self.w3_logs), r.bot[5].clone(), self.states[self.W0.data_ptr()].calls, r.top_events)

        on, off = run(True), run(False)
        for i, (a, b) in enumerate(zip(on[:-1], off[:-1])):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b), i)
            else:
                self.assertEqual(a, b, i)
        self.assertEqual(on[-1], MX6_ORDER)
        self.assertEqual(off[-1][:3], ["decline_reason", "dout_tiled", "erm_take"])  # MX6 off: the top's shipped cast


class TopErmLatchTest(_StepBase):
    def test_the_w3_bitwise_guard_trip_of_job_72925_leaves_mx6_alone(self) -> None:
        """The xt2 entry's first-call BITWISE guard vs x21 rejects in the bottom call. That call's dW is x21's result
        on the re-tiled x_rm (run_w3_rm returns the reference); the row-major entry latches (wgrad_ck_w3._rm_declined;
        dy_erm itself stays enabled); from the next step E-RM still makes and claims its site, the ticket declines (one
        "erm NOT used (W3: ...)" line), the bottom runs the dual cast with the shipped W3 entry. MX6 at the top is
        untouched in both steps; the bottom never falls to MX6."""
        self.x21_result = self.xt2_result + 1.0  # not bitwise equal
        r = self.step()
        self.assert_top_is_mx6_without_erm(r)
        self.assert_bottom_is_erm(r, armed_now=False)
        self.assertTrue(torch.equal(r.bot[5], self.x21_result.t().contiguous()))  # x21's result, not xt2's
        self.assertEqual(r.bot_events.count("xt2_launch"), 1)
        self.assertEqual(r.bot_events.count("x21"), 1)
        dout_t = self.x21_calls[-1][0]
        self.assertEqual(tuple(dout_t.shape), (KT, D, 128))
        self.assertTrue(torch.equal(dout_t.view(U8), wgrad_ck_w3.rm_to_tiled(r.prod_outs["xrm"], T).view(U8)))
        self.assertIn("bitwise=x21 -> rejected", "\n".join(self.w3_logs))
        self.assertIn("output != x21 output (bitwise)", wgrad_ck_w3.rm_declined() or "")
        self.assertIsNone(dy_erm.broken())
        self.assertTrue(dy_erm.enabled())
        self.assertEqual(self.w3_disabled, [])  # x21 / W3 not latched
        r2 = self.step()
        self.assertIsInstance(r2.erm_site, dy_erm.ErmSite)  # E-RM still makes and claims its site
        self.assertIs(r2.claimed, r2.erm_site)
        self.assertIsNone(r2.prod)  # the ticket declined
        self.assertIsNone(r2.rec)
        self.assert_top_is_mx6_without_erm(r2)
        self.assertEqual(r2.bot_takes, [r2.erm_site])
        self.assertIsNone(r2.bot_calls["fp8_cast"][0][1]["erm"])
        self.assertEqual(len(r2.bot_dual), 1)
        self.assertIs(r2.bot_w3[-1][0], r2.bot_dual[0][1])
        self.assertIsNone(getattr(r2.ctx0, "l11_mx6_layer", None))
        logs = "\n".join(self.erm_logs)
        self.assertEqual(logs.count("erm NOT used (W3: row-major entry latched off"), 1)
        c = do8.mx6_counters()
        self.assertEqual((c["published"], c["declined"], c["latched"]), (2, {}, 0))
        self.assertNotIn("NOT armed", r.top_text + r2.top_text + r.bot_text + r2.bot_text)


class TopW3OperandTest(_StepBase):
    def real_top_w3(self) -> None:
        """The shipped W3 entry for real (_out_wgrad_nn_ck_w3 -> run_w3 with its contract check; x21's launch and
        hipBLASLt faked; a CPU tensor stands for cuda:0 in the device check only)."""
        from generative_recommenders.ops import hipblaslt_fp8_nn as hb

        def decline(y8, d, sy, sd, m):  # pyre-ignore[2,3]
            why = REAL_DECLINE_W3(y8, d, sy, sd, m)
            return None if why is not None and why[0] == "device" else why

        self.extra += [
            (self.thl, "_out_wgrad_nn_ck_w3", {"new": self.REAL_OUT_W3}),
            (wgrad_ck_w3, "run_w3", {"new": REAL_RUN_W3}),
            (wgrad_ck_w3, "_decline_reason", {"new": decline}),
            (wgrad_ck_w3, "_load", {"new": lambda: None}),
            (wgrad_ck_w3, "_launch", {"new": lambda *a, **k: self.xt2_result.clone()}),
            (hb, "fp8_wgrad_nn", {"new": lambda y8, d, ys, ds, out_dtype=None: torch.zeros(D, NOUT)}),
        ]
        self.tiled_live = True

    def test_a_misaligned_producer_dout_t_is_a_contract_break_before_x21_and_erm_stays_on(self) -> None:
        """S1 returns a step-tiled, contiguous dout_t that is not 16-B aligned (x21 would decline it at W3, set its
        process-wide tiling latch and with it switch E-RM's row-major entry off): the contract check rejects it at the
        top (Mx6ContractError, fail closed; MX6 latched, plan closed) before W3 runs, so W3's step tiling stays on and
        E-RM is consumed at the bottom of the same step."""
        self.fresh_step_state()
        self.real_top_w3()
        buf = torch.zeros(KT * D * 128 + 16, dtype=U8)
        self.ops6 = seam._ops6(dout_fp8=buf[1:KT * D * 128 + 1].view(E4).view(KT, D, 128))
        self.assertTrue(self.ops6.dout_fp8.is_contiguous() and self.ops6.dout_fp8.data_ptr() % 16 != 0)
        r = self.step()
        self.assertIsInstance(r.top, do8.Mx6ContractError, r.top)
        self.assertIn("a16=0", str(r.top))
        self.assertIsNotNone(do8.mx6_broken())
        self.assertEqual(r.top_events, MX6_ORDER[:5])  # the producer ran; no MX6 launch, no W3 on its operand
        self.assertIsNone(wgrad_ck_w3._tiled_declined)  # x21 never saw the operand: tiling on
        self.assertIsNone(REAL_RM_INACTIVE(CUDA0))
        self.assert_no_open_plan(r.do8_site1)
        self.assert_bottom_is_erm(r)
        c = dy_erm.counters()
        self.assertEqual((c["sites"], c["claimed"], c["tickets"], c["consumed"]), (1, 1, 1, 1))
        self.assertNotIn("erm NOT used", "\n".join(self.erm_logs))


if __name__ == "__main__":
    unittest.main()

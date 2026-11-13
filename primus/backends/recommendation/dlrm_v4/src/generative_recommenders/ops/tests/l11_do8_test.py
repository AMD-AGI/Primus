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
"""Tests of DO8 (gin item (DO8), DLRM_HSTU_FP8_L11_DO8): L11's dO8/E8 emission variant (ops/hip/l11_do8.{cu,py}).

CPU tests (no kernel launches): knob values; knob off = nothing imported, built or printed; DO8's own resource gate;
LATCH ISOLATION (iter-35 review C-I1): a DO8 build / gate / self-check / launch failure latches DO8 only, L11 stays
enabled; device faults in the self-check re-raise; the decide-before-emission reasons and which are silent; publish
counting and the ACTIVE line; the work-list memo key; the seam's wiring (source pins: the DO8 branch sits between L11's
bf16-du launch and its e5 hook, publish after the last write to dattn, site_of only in the forward).
GPU tests (gfx950): the device work list vs a host list (gaps, empty / long sequences, int32 / int64, malformed
offsets); DO8 vs L11's own E5 launch and vs quantize_do_e4m3_blk at small M (edge lengths, zero groups, NaN rows,
saturation, tiny scales); a launch failure falls back; the prebuild self-check; the registry round trip when R15's hook
is in the tree. The large-M matrix and timing run from the lab tasks (r16 probes/iter36/do8e5/staged/lab).
NOT RUN at authoring time (no torch on the authoring host); the lab task t_do8e5_gate.sh runs this file.
DO8-ONLY (r16 iter42, R15 contract v2 'only'): CPU tests against the REAL hstu_do8_registry (plan / publish 'only' /
take / re-take / release / cancel) with the kernel launch and the Triton finalize stubbed: the knob-off path never asks
the registry and publishes 'plus'; an 'only' call launches EM = 2, poisons on request, publishes bf16_written=False;
launch / post-launch failures cancel the plan and latch 'only' resp. DO8; publish errors propagate; the 'only'
prebuild latches 'only' only; the 'only' resource gate, flags and the .cu build switch. GPU tests (gfx950): the
EM = 2 self-check, and EM = 2 vs EM = 1 byte for byte (DX sentinel untouched) on the edge and special-value cases.
B4 v2 (r16 iteration 43, R15's review N1): the REAL launch_e5_plus with only its ctypes entry points stubbed: only the
EM = 2 kernel's own rc > 0 (OnlyLaunchError) latches 'only'; a layout check, plan(cap), the drain or a placeholder /
partials OOM after the plan cancels it and latches DO8 as 'plus' does; a plan left open inside _plan_only is cancelled;
the knob-off failure keeps B3's latch text. R15's keep_refs=False chain (N4) is l11_do8_only_chain_test.py.
DO8 DIET (gin item (89), DLRM_HSTU_FP8_L11_DO8_DIET): CPU: knob values (only "1" arms it, read once), off adds
-DL11DO8_DIET=0 to the 'plus' and 'only' flags (before -DL11DO8_ONLY=1) and changes the cache keys, on leaves the flags
as without the switch, the prebuild lines' OFF note. GPU (gfx950): diet ON vs OFF byte for byte ('plus' and 'only': dO8,
E8, dz8, d(attn) / sentinel, every partial, P) on the edge and special-value cases, and both self-checks on the OFF
build. The other GPU classes pin the diet ON (the gin default). The torch-free gin / .cu switch pins are
dlrm_v4/train/tests/l11_do8_diet_gin_test.py.
"""

import contextlib
import ctypes
import io
import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple
from unittest import mock

import torch

from generative_recommenders.ops.hip import l11_do8 as do8
from generative_recommenders.ops.hip import l11_outproj_lnmd as l11

E4, E5, BF, F32, U8 = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float32, torch.uint8
D, H, HD, DU_ROW = 512, 4, 128, 2048
P_DROP = 0.1
EDGE = (1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128)


def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    return l11.device_arch(torch.cuda.current_device()) == "gfx950"


def _remark(name: str, vgpr: int = 255, spill: int = 0, scratch: int = 0, lds: int = 163840) -> dict:
    return {"Function Name": name, "VGPRs": str(vgpr), "TotalSGPRs": "106", "VGPRs Spill": str(spill),
            "SGPRs Spill": "55", "ScratchSize [bytes/lane]": str(scratch), "LDS Size [bytes/block]": str(lds)}


PROD = "_Z24l11_outproj_lnmd_bwd_do8" + do8._PROD_MANGLE + "vPKhS1_PtPfPKfS5_iiPKtS7_S1_S5_S2_S2_8L11CArgs"
WLB = "_Z16l11_do8_wl_buildIxEvPKT_iixPiS3_"


class _Capture(contextlib.AbstractContextManager):
    def __enter__(self):
        self.buf = io.StringIO()
        self._cm = contextlib.redirect_stdout(self.buf)
        self._cm.__enter__()
        return self

    def __exit__(self, *a):
        self._cm.__exit__(*a)
        return False

    @property
    def text(self) -> str:
        return self.buf.getvalue()


def _host_list(offs: List[int], m: int) -> Tuple[List[Tuple[int, int]], int]:
    """The builder's contract on the host: (entries, status)."""
    b = len(offs) - 1
    ok = offs[0] >= 0 and offs[-1] <= m and all(offs[i + 1] >= offs[i] for i in range(b))
    if not ok:
        return [(64 * i, min(64, m - 64 * i)) for i in range((m + 63) // 64)], 1
    out = [(64 * i, min(64, offs[0] - 64 * i)) for i in range((offs[0] + 63) // 64)]
    for i in range(b):
        for t in range(0, offs[i + 1] - offs[i], 64):
            out.append((offs[i] + t, min(64, offs[i + 1] - offs[i] - t)))
    out += [(offs[-1] + 64 * i, min(64, m - offs[-1] - 64 * i)) for i in range((m - offs[-1] + 63) // 64)]
    return out, 0


class DO8CpuTest(unittest.TestCase):
    def setUp(self) -> None:
        do8.reset_for_testing()
        do8.set_requested_for_testing(True)
        l11.reset_for_testing()

    def tearDown(self) -> None:
        do8.set_requested_for_testing(None)
        do8.reset_for_testing()
        l11.reset_for_testing()

    def test_knob_values(self) -> None:
        do8.set_requested_for_testing(None)
        for value, want in (("1", True), ("0", False), ("", False), ("true", False), ("2", False)):
            with mock.patch.dict(os.environ, {do8.KNOB: value}):
                do8.set_requested_for_testing(None)
                self.assertEqual(do8.requested(), want, value)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(do8.KNOB, None)
            do8.set_requested_for_testing(None)
            self.assertFalse(do8.requested())

    def test_knob_off_imports_builds_prints_nothing(self) -> None:
        do8.set_requested_for_testing(False)
        with _Capture() as cap:
            self.assertIsNone(do8.site_of(torch.empty(4, D)))
            self.assertFalse(do8.prebuild(l11_ready=True, e5_expected=True))
            self.assertIsNone(do8.emission_site(SimpleNamespace(do8_site=object()), torch.empty(4, D)))
            do8.note_bf16_mode(SimpleNamespace(do8_site=object()))
        self.assertEqual(cap.text, "")
        self.assertFalse(do8._registry_tried)
        self.assertIsNone(do8.library_path())
        self.assertEqual(do8.counters()["events"], 0)

    def test_resource_gate(self) -> None:
        ok = do8.check_resources([_remark(PROD), _remark(WLB, vgpr=40, lds=136)])
        self.assertEqual((ok["vgpr"], ok["vgpr_spill"], ok["scratch"], ok["lds"]), (255, 0, 0, 163840))
        bad_cases = [
            [_remark(PROD, spill=1), _remark(WLB, lds=136)],  # V9-like spill
            [_remark(PROD, scratch=8), _remark(WLB, lds=136)],
            [_remark(PROD, lds=98304), _remark(WLB, lds=136)],
            [_remark(PROD), _remark(WLB, scratch=16, lds=136)],  # builder arrays not promoted
            [_remark(PROD)],  # builder missing
            [_remark(PROD), _remark(PROD), _remark(WLB, lds=136)],
            [_remark(PROD.replace("ELi2ELi1ELi0ELi2ELi0EE", "ELi2ELi2ELi0ELi2ELi0EE")), _remark(WLB, lds=136)],
        ]
        for kernels in bad_cases:
            with self.assertRaises(RuntimeError):
                do8.check_resources(kernels)

    def test_build_failure_latches_do8_only(self) -> None:
        """C-I1: a DO8 build failure (compiler crash, spill, gate) latches DO8; L11 stays enabled and prebuilt."""
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                mock.patch.object(do8, "build", side_effect=RuntimeError("hipcc rc=-11 (sched=default): clang crash")), \
                _Capture() as cap:
            self.assertFalse(do8.prebuild(l11_ready=True, e5_expected=True))
        self.assertIsNotNone(do8.broken())
        self.assertIn("L11 DO8 dO8/E8 emission DISABLED: prebuild build/load", cap.text)
        self.assertIn("[rank ", cap.text)
        self.assertIsNone(l11.broken())
        self.assertTrue(l11.enabled() and l11.prebuilt())
        l11.set_requested_for_testing(None)

    def test_self_check_mismatch_latches_do8_only(self) -> None:
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(torch.cuda, "synchronize"), \
                mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                mock.patch.object(do8, "_load"), _Capture() as cap:
            self.assertFalse(do8.prebuild(True, True, self_check_fn=lambda: (False, "dO8 mism 3")))
        self.assertIn("prebuild self-check mismatch: dO8 mism 3", cap.text)
        self.assertFalse(do8.prebuilt())
        self.assertIsNone(l11.broken())
        l11.set_requested_for_testing(None)

    def test_self_check_device_fault_reraises(self) -> None:
        def boom():
            raise RuntimeError("HIP error: an illegal memory access was encountered")

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                mock.patch.object(do8, "_load"), _Capture() as cap:
            with self.assertRaises(RuntimeError):
                do8.prebuild(True, True, self_check_fn=boom)
        self.assertIn("prebuild DEVICE FAULT", cap.text)

    def test_prebuild_inert_lines(self) -> None:
        with _Capture() as cap:
            self.assertFalse(do8.prebuild(l11_ready=False, e5_expected=True))
            self.assertFalse(do8.prebuild(l11_ready=True, e5_expected=False))
        self.assertIn("L11 DO8 dO8/E8 emission INERT: needs item (39)", cap.text)
        self.assertIn("L11 DO8 dO8/E8 emission INERT: needs item (38)(a)", cap.text)

    def _site(self, T: int = 1000, Hs: int = H, Ds: int = HD, offs=None):  # pyre-ignore[2,3]
        if offs is None:
            offs = torch.tensor([0, 400, T], dtype=torch.int64)
        return SimpleNamespace(T=T, H=Hs, D=Ds, N=4096, device=torch.device("cpu"), seq_offsets=offs)

    def test_emission_site_reasons(self) -> None:
        attn = torch.empty((1000, D), dtype=BF)
        reg = SimpleNamespace(take_enabled=lambda: True, site_of=lambda a: None)
        do8.set_prebuilt_for_testing(True)
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        with mock.patch.object(do8, "registry", return_value=reg), _Capture() as cap:
            self.assertIsNone(do8.emission_site(SimpleNamespace(do8_site=None), attn))  # silent, not a call
            site = self._site()
            self.assertIs(do8.emission_site(SimpleNamespace(do8_site=site), attn), site)
            self.assertIsNone(do8.emission_site(SimpleNamespace(do8_site=self._site(T=999)), attn))
            self.assertIsNone(do8.emission_site(SimpleNamespace(do8_site=self._site(Hs=8, Ds=64)), attn))
            self.assertIsNone(do8.emission_site(
                SimpleNamespace(do8_site=self._site(offs=torch.tensor([0, 1000], dtype=torch.int16))), attn))
            self.assertIsNone(do8.emission_site(
                SimpleNamespace(do8_site=self._site(offs=torch.zeros(8194, dtype=torch.int64))), attn))
            reg.take_enabled = lambda: False
            self.assertIsNone(do8.emission_site(SimpleNamespace(do8_site=site), attn))
        lines = [ln for ln in cap.text.splitlines() if "NOT armed" in ln]
        self.assertTrue(all(ln.startswith(do8.FAMILY + " NOT armed: ") and "[rank " in ln for ln in lines))
        self.assertTrue(any("site T=999" in ln for ln in lines))
        self.assertTrue(any("consumer knob DLRM_HSTU_FP8_BWD_DO8 off" in ln for ln in lines))
        self.assertEqual(do8.counters()["calls"], 6)
        # silent expected states: latched
        do8._broken = "x"
        with mock.patch.object(do8, "registry", return_value=reg), _Capture() as cap2:
            self.assertIsNone(do8.emission_site(SimpleNamespace(do8_site=site), attn))
        self.assertNotIn("NOT armed", cap2.text.replace("[l11-do8]", ""))
        l11.set_requested_for_testing(None)

    def test_not_prebuilt_declines(self) -> None:
        attn = torch.empty((1000, D), dtype=BF)
        with _Capture() as cap:
            self.assertIsNone(do8.emission_site(SimpleNamespace(do8_site=self._site()), attn))
        self.assertIn("D11 not prebuilt", cap.text)

    def test_publish_counts_and_active_line_once(self) -> None:
        calls = []

        def pub(site, dx, d8, e8, producer, bf16_written):
            calls.append((producer, bf16_written))
            return None if len(calls) == 1 else object()

        reg = SimpleNamespace(publish=pub)
        dattn = torch.empty((10, D), dtype=BF)
        with mock.patch.object(do8, "registry", return_value=reg), _Capture() as cap:
            self.assertFalse(do8.publish(object(), dattn, torch.empty(0), torch.empty(0)))  # refused
            self.assertTrue(do8.publish(object(), dattn, torch.empty(0), torch.empty(0)))
            self.assertTrue(do8.publish(object(), dattn, torch.empty(0), torch.empty(0)))
        self.assertEqual(calls, [("L11", True)] * 3)  # 'plus' only, contract v1.2
        self.assertEqual(cap.text.count("ACTIVE mode=e5 plus"), 1)
        self.assertEqual(cap.text.count("publish refused"), 1)
        c = do8.counters()
        self.assertEqual((c["published"], c["refused"]), (2, 1))

    def test_work_list_memo_key(self) -> None:
        launches = []

        def fake_build(offs_ptr, i64, b, m, nprog, wl_ptr, cap, cnt_ptr, stream):
            launches.append((i64, b, m, nprog, cap))
            return 0

        stream = SimpleNamespace(cuda_stream=1234)
        offs = torch.tensor([0, 100, 300], dtype=torch.int64)
        with mock.patch.object(do8, "_fn_wl_build", fake_build), \
                mock.patch.object(do8, "_fn_wl_cap", lambda m, b: (m + 63) // 64 + b + 2), \
                mock.patch.object(torch.cuda, "current_stream", return_value=stream):
            wl, cnt, cap = do8.work_list(offs, 300, 4096, torch.device("cpu"))
            self.assertEqual((tuple(wl.shape), cap), ((5 + 2 + 2, 2), 9))
            do8.work_list(offs, 300, 4096, torch.device("cpu"))  # same object, version, stream: reused
            offs.add_(0)  # in-place write: version bump -> rebuild
            do8.work_list(offs, 300, 4096, torch.device("cpu"))
            other = offs.clone()  # equal values, another object -> rebuild
            do8.work_list(other, 300, 4096, torch.device("cpu"))
            stream.cuda_stream = 99  # another stream -> rebuild
            do8.work_list(other, 300, 4096, torch.device("cpu"))
        self.assertEqual(len(launches), 4)
        self.assertEqual((do8.counters()["wl_built"], do8.counters()["wl_reused"]), (4, 1))
        self.assertEqual(launches[0], (1, 2, 300, 64, 9))

    def test_run_failure_latches_and_returns_none(self) -> None:
        """A launch / record failure inside run_e5_plus latches DO8 (only) and returns None: the seam then runs L11's
        own run_e5 on the same operands."""
        do8.set_prebuilt_for_testing(True)
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        with mock.patch.object(l11, "_operands_ok", return_value=None), \
                mock.patch.object(l11, "e5_decline_reason", return_value=None), \
                mock.patch.object(do8, "_lib", object()), \
                mock.patch.object(do8, "work_list", side_effect=RuntimeError("l11do8_wl_build_launch rc=1")), \
                mock.patch("generative_recommenders.ops.triton.bwd_fp8.grad_dtype_and_limit",
                           return_value=(E5, 57344.0)), \
                mock.patch("generative_recommenders.ops.triton.bwd_fp8.uvqk_tm_pad", return_value=128), \
                _Capture() as cap:
            site = self._site()
            args = ["T"] * 20
            args[4] = torch.empty((1000, D), dtype=BF)  # x (M is read before the try)
            out = do8.run_e5_plus(*args, site)
        self.assertIsNone(out)
        self.assertIn("DISABLED: e5 plus: RuntimeError", cap.text)
        self.assertIsNone(l11.broken())
        l11.set_requested_for_testing(None)

    def test_seam_wiring_source_pins(self) -> None:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        src = Path(thl.__file__).read_text()
        i0 = src.index("class HSTUComputeOutputFunction")
        i1 = src.index("    def backward(", i0)
        ends = [j for j in (src.find("\nclass ", i1), src.find("\ndef ", i1)) if j > 0]
        fwd, bwd = src[i0:i1], src[i1:min(ends) if ends else len(src)]
        # the seam's one lookup (forward: ctx.do8_site) ...
        self.assertEqual((fwd + bwd).count("l11_do8.site_of("), 1)
        # ... plus r16 iter54 NIB's (item (94)) read-only probe in the NIB writers' gate, and nowhere else
        self.assertEqual(src.count("l11_do8.site_of("), 2)
        j0 = src.index("\ndef _lnmd_mask_nib_why_not(")
        j1 = src.index("\ndef ", j0 + 1)
        nib_gate = src[j0:j1]
        self.assertEqual(nib_gate.count("l11_do8.site_of("), 1)
        for w in ("publish(", "plan(", "cancel(", "emission_site(", "run_e5_plus(", "launch_e5_plus("):
            self.assertNotIn(w, nib_gate)  # host metadata only: no registry or DO8 state change there
        self.assertIn("ctx.do8_site = l11_do8.site_of(attn)", fwd)
        self.assertNotIn("l11_do8.publish(", fwd)
        # DO8 branch between L11's bf16-du launch and its e5 hook; decided by emission_site before the launch
        i_bf16 = bwd.index("l11_out = l11_outproj_lnmd.run(")
        i_do8 = bwd.index("_l11_do8_e5_or_none(")
        i_site = bwd.index("site=l11_do8.emission_site(ctx, attn)")
        i_e5 = bwd.index("l11_e5 = _l11_du_e5m2_or_none(")
        self.assertTrue(i_bf16 < i_do8 < i_site < i_e5)
        # publish after the last write to dattn (B1a's record attach) and before the wgrad / return
        i_pub = bwd.index("l11_do8.publish(")
        self.assertLess(bwd.index("du_site.emitted = du_emit.token"), i_pub)
        self.assertLess(i_pub, bwd.index("y = saved_y", i_pub))
        self.assertIsNone(re.search(r"^\s*dattn\b[^=\n]*=(?!=)", bwd[i_pub:], re.M))  # no write to dattn after
        self.assertTrue(re.search(r"return \(\s*dattn,", bwd[i_pub:]))
        # the hook returns None without a site or without a prebuilt L11
        with mock.patch.object(l11, "enabled", return_value=True), mock.patch.object(l11, "prebuilt",
                                                                                      return_value=False):
            self.assertIsNone(thl._l11_do8_e5_or_none("G8", "W8", "SA", "SB", site=object(), **{
                k: None for k in ("x", "u", "weight", "bias", "mean", "rstd", "BLOCK_D", "num_warps",
                                  "dropout_ratio", "silu_u", "random_mask", "du_row_width", "site_key", "inv_scale",
                                  "cast_scale", "generation")}))

    def test_l11_e5_census_line_kept(self) -> None:
        """Item (39)'s rung census expects "armed OK: <L11 FAMILY> ACTIVE mode=e5" on every rank; with DO8 emitting from
        the first e5 call, l11.run_e5 never runs, so DO8 prints that line (same text + a DO8 tail) and sets L11's
        once-flag."""
        src = Path(do8.__file__).read_text()
        i = src.index("if not l11._active_e5_logged:")
        tail = src[i:i + 1500]
        self.assertIn("l11._active_e5_logged = True", tail)
        self.assertIn('f"armed OK: {l11.FAMILY} ACTIVE mode=e5 (B1a contract)', tail)
        self.assertTrue(src.index("_count(\"emitted\")") < i)
        # census truth (r16 iter39 PLAN C4): the line names the kernel that ran (DO8's own impl), not L11's .so
        self.assertIn("impl={_impl}", tail)
        self.assertNotIn("impl={l11.impl()}", tail)

    def test_vd_selects_source(self) -> None:
        """DO8-VD (r16 iter41): l11_do8.cu (L11-VD's diet, impl tag "+vd(...)") only for DLRM_HSTU_FP8_L11_VD == "1"
        (L11's own selector); absent / "0" / anything else builds l11_do8_novd.cu (pre-diet); the _SRC override wins;
        distinct sources -> distinct cache keys."""
        self.assertTrue(do8._SOURCE.is_file())
        self.assertTrue(do8._SOURCE_NOVD.is_file())
        self.assertEqual(do8._SOURCE.name, "l11_do8.cu")
        self.assertEqual(do8._SOURCE_NOVD.name, "l11_do8_novd.cu")
        vd, novd = do8._SOURCE.read_bytes(), do8._SOURCE_NOVD.read_bytes()
        self.assertNotEqual(vd, novd)
        self.assertIn(b"+vd(C1,C2,C3,C4,LANEFIX,PIN)", vd)
        self.assertNotIn(b"+vd(", novd)
        guard = b"if (!(dscale > 0.0f && dscale <= 3.40282346638528859811704183484516925e+38f))"
        # the E5 entry and (r16 MX6 row) its MX6 twin l11do8_e5_*_mx6_launch inside #if L11DO8_MX6
        self.assertEqual(vd.count(guard), 2)
        self.assertEqual(vd[:vd.index(b'extern "C" int L11DO8_E5_MX6_LAUNCH(')].count(guard), 1)
        self.assertEqual(novd.count(guard), 0)
        for val, want in (("1", do8._SOURCE), (" 1 ", do8._SOURCE), ("0", do8._SOURCE_NOVD), ("", do8._SOURCE_NOVD),
                          ("true", do8._SOURCE_NOVD), ("2", do8._SOURCE_NOVD), (None, do8._SOURCE_NOVD)):
            env = dict(os.environ)
            env.pop(l11.VD_KNOB, None)
            env.pop(do8.KNOB + "_SRC", None)
            if val is not None:
                env[l11.VD_KNOB] = val
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(do8._source(), want, val)
        with mock.patch.dict(os.environ, {l11.VD_KNOB: "1", do8.KNOB + "_SRC": "/x/y.cu"}):
            self.assertEqual(str(do8._source()), "/x/y.cu")
        try:  # needs hipcc on PATH (the key covers the toolchain), else skip that half
            k_vd, k_novd = do8.cache_key(do8._SOURCE), do8.cache_key(do8._SOURCE_NOVD)
        except (RuntimeError, OSError):
            return
        self.assertNotEqual(k_vd, k_novd)

    def test_lines_carry_family_and_rank(self) -> None:
        with _Capture() as cap:
            do8.note_decline("seq_offsets torch.int16 (2,) contiguous=True")
            do8._disable("test")
        for ln in cap.text.splitlines():
            if ln.startswith("[l11-do8]"):
                self.assertIn("[rank ", ln)
                continue
            self.assertTrue(ln.startswith(do8.FAMILY), ln)
            self.assertIn("[rank ", ln)

    def test_diet_knob_values_flags_and_key(self) -> None:
        """DO8 DIET (gin item (89)): only "1" arms it (read once; absent = off); off adds -DL11DO8_DIET=0 to every
        DO8 build ('plus' and 'only', before the 'only' define), on adds nothing (the flags without the switch)."""
        self.assertEqual(do8.DIET_KNOB, "DLRM_HSTU_FP8_L11_DO8_DIET")
        self.assertEqual(do8._DIET_OFF_DEFINE, "-DL11DO8_DIET=0")
        try:
            for value, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("true", False),
                                ("True", False), ("2", False), (None, False)):
                env = dict(os.environ)
                env.pop(do8.DIET_KNOB, None)
                if value is not None:
                    env[do8.DIET_KNOB] = value
                with mock.patch.dict(os.environ, env, clear=True):
                    do8.set_diet_for_testing(None)
                    self.assertEqual(do8.diet_requested(), want, value)
            with mock.patch.dict(os.environ, {do8.DIET_KNOB: "1"}):
                do8.set_diet_for_testing(None)
                self.assertTrue(do8.diet_requested())
                os.environ[do8.DIET_KNOB] = "0"
                self.assertTrue(do8.diet_requested())  # read once
            do8.set_diet_for_testing(True)
            on, on_only = do8._flags(b"x"), do8._flags(b"x", True)
            self.assertFalse([f for f in on_only if "L11DO8_DIET" in f])
            self.assertEqual(do8._diet_tail(), "")
            do8.set_diet_for_testing(False)
            off, off_only = do8._flags(b"x"), do8._flags(b"x", True)
            self.assertEqual(off, on + ["-DL11DO8_DIET=0"])
            self.assertEqual(off_only, on + ["-DL11DO8_DIET=0", "-DL11DO8_ONLY=1"])
            self.assertIn("diet OFF (DLRM_HSTU_FP8_L11_DO8_DIET != 1: -DL11DO8_DIET=0", do8._diet_tail())
            with mock.patch.object(l11, "_toolchain", return_value="hipcc test"):
                keys = {}
                for diet in (True, False):
                    do8.set_diet_for_testing(diet)
                    keys[diet] = (do8.cache_key(do8._SOURCE), do8.cache_key(do8._SOURCE, only=True))
                self.assertNotEqual(keys[True][0], keys[False][0])
                self.assertNotEqual(keys[True][1], keys[False][1])
        finally:
            do8.set_diet_for_testing(None)

    def test_diet_prebuild_line_note_only_when_off(self) -> None:
        for diet, note in ((True, False), (False, True)):
            do8.reset_for_testing()
            do8.set_diet_for_testing(diet)
            try:
                with mock.patch.object(do8, "_load"), \
                        mock.patch.object(torch.cuda, "is_available", return_value=True), \
                        mock.patch.object(torch.version, "hip", "7.2"), \
                        mock.patch.object(torch.cuda, "current_device", return_value=0), \
                        mock.patch.object(torch.cuda, "synchronize"), \
                        mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                        mock.patch.object(do8, "only_requested", return_value=False), _Capture() as cap:
                    self.assertTrue(do8.prebuild(True, True, self_check_fn=lambda: (True, "ok")))
            finally:
                do8.set_diet_for_testing(None)
            line = [ln for ln in cap.text.splitlines() if "prebuild before run_start" in ln]
            self.assertEqual(len(line), 1, cap.text)
            self.assertEqual("; diet OFF (" in line[0], note, line[0])
            # the logger may append " [rank N]" on a GPU host
            tail = re.sub(r" \[rank \d+\]$", "", line[0].rstrip())
            tail = re.sub(r" mask=\S+$", "", tail)  # NIB (item (94)): the appended " mask=<format>" segment
            self.assertTrue(tail.endswith("ok)") != note, line[0])


@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU")
class DO8GpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("DLRM_HSTU_FP8_MASK16", "1")
        os.environ.setdefault("DLRM_HSTU_FP8_LNMD_BWD_V2", "1")
        do8.set_diet_for_testing(True)  # item (89): the gin default (the shipped diet build); OFF is DO8DietGpuTest's
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(True)
        l11.set_requested_for_testing(True)
        l11._load()
        do8._load()

    @classmethod
    def tearDownClass(cls) -> None:
        do8.set_requested_for_testing(None)
        do8.set_diet_for_testing(None)
        l11.set_requested_for_testing(None)

    def _build(self, offs: List[int], m: int, dtype=torch.int64, n: int = 4096):  # pyre-ignore[2,3]
        t = torch.tensor(offs, dtype=dtype, device="cuda")
        cap = do8.wl_cap(m, len(offs) - 1)
        wl = torch.full((cap, 2), -7, dtype=torch.int32, device="cuda")
        cnt = torch.full((2,), -7, dtype=torch.int32, device="cuda")
        rc = do8._fn_wl_build(t.data_ptr(), int(dtype == torch.int64), len(offs) - 1, m,
                              max(1, min(128, (n + 63) // 64)), wl.data_ptr(), cap, cnt.data_ptr(),
                              torch.cuda.current_stream().cuda_stream)
        self.assertEqual(rc, 0)
        torch.cuda.synchronize()
        return wl.cpu().tolist(), cnt.cpu().tolist(), cap

    def test_work_list_vs_host(self) -> None:
        g = torch.Generator().manual_seed(7)
        cases = []
        for b in (0, 1, 3, 1024, 8192):
            hi = max(2, min(5000, 3_000_000 // max(b, 1)))  # M stays below L11's 4,194,304
            lens = torch.randint(0, hi, (b,), generator=g).tolist()
            if b:
                lens[0] = 0
                lens[-1] = 4097  # longer than N: tiles beyond cdiv(N, 64) are listed too
            offs = [0]
            for n_ in lens:
                offs.append(offs[-1] + n_)
            cases.append((offs, max(offs[-1], 1)))
        cases.append(([5, 100, 300], 400))  # head and tail gaps
        cases.append(([0, 100, 90, 300], 300))  # malformed: negative length -> plain tiles, status 1
        cases.append(([0, 100, 500], 300))  # malformed: o_B > M
        for offs, m in cases:
            for dt in (torch.int64, torch.int32):
                wl, cnt, cap = self._build(offs, m, dt, n=4096)
                ref, status = _host_list(offs, m)
                self.assertEqual(cnt, [len(ref), status], (offs[:4], m, dt))
                self.assertLessEqual(len(ref), cap)
                self.assertEqual([tuple(e) for e in wl[: len(ref)]], ref, (offs[:4], m, dt))
                rows = sorted(r for s, v in ref for r in range(s, s + v))
                self.assertEqual(rows, list(range(m)))  # an exact cover of [0, M)

    def _case(self, lens: List[int], kind: str = "gauss", sa_log2: int = -19, seed: int = 11):  # pyre-ignore[3]
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        M = sum(lens)
        offs = [0]
        for n_ in lens:
            offs.append(offs[-1] + n_)
        g = torch.Generator(device="cuda").manual_seed(seed)
        g8f = torch.randn((M, D), generator=g, device="cuda") * 1024.0
        x = (torch.randn((M, D), generator=g, device="cuda") * 0.5).to(BF)
        if kind == "special":
            for i in range(0, len(lens), 3):  # all-zero first group of every 3rd sequence
                s0 = offs[i]
                for r0 in range(0, min(lens[i], 64), 16):
                    g8f[s0 + r0:s0 + min(r0 + 8, lens[i])] = 0.0
            for i in range(1, len(lens), 5):  # NaN rows (a whole group when long enough)
                s0 = offs[i]
                x[s0 + min(8, lens[i] - 1):s0 + min(16, lens[i])] = float("nan")
        g8 = g8f.clamp_(-57344.0, 57344.0).to(E5)
        w8 = (torch.randn((1536, D), generator=g, device="cuda") * 90.0).clamp_(-448.0, 448.0).to(E4)
        sa = torch.full((1,), 2.0**sa_log2, dtype=F32, device="cuda")
        sb = torch.full((1,), 0.0613 / 448.0, dtype=F32, device="cuda")
        u = torch.randn((M, D), generator=g, device="cuda").to(BF)
        wn = (1.0 + 0.05 * torch.randn((D,), generator=g, device="cuda")).to(BF)
        bn = (0.05 * torch.randn((D,), generator=g, device="cuda")).to(BF)
        xf = x.float()
        mean, rstd = xf.mean(1), torch.rsqrt(xf.var(1, unbiased=False) + 1e-6)
        mask = thl._create_dropout_mask(N=M, D=D, BLOCK_D=D, concat_u=True, concat_x=True, dropout_ratio=P_DROP,
                                        seed=seed, device=torch.device("cuda"))
        return SimpleNamespace(M=M, offs=torch.tensor(offs, dtype=torch.int64, device="cuda"), N=max(lens), g8=g8,
                               w8=w8, sa=sa, sb=sb, x=x, u=u, wn=wn, bn=bn, mean=mean, rstd=rstd, mask=mask)

    def _compare(self, c, inv_v: float = 2.0**-3) -> None:  # pyre-ignore[2]
        from generative_recommenders.ops.flydsl.hstu_do_fp8 import quantize_do_e4m3_blk

        rows_out = (c.M + 127) // 128 * 128
        inv = torch.full((1,), inv_v, dtype=F32, device="cuda")
        dz_s = torch.full((rows_out, DU_ROW), 0x5A, dtype=U8, device="cuda")
        dx_s, _, _, ducs_s, amax_s, _ = l11.launch_e5(c.g8, c.w8, c.sa, c.sb, c.x, c.u, c.mask, c.mean, c.rstd,
                                                      c.wn, c.bn, P_DROP, inv, dz_s.view(E5))
        do8.reset_for_testing()
        do8.set_requested_for_testing(True)
        wl, cnt, cap = do8.work_list(c.offs, c.M, c.N, c.x.device)
        dz_d = torch.full((rows_out, DU_ROW), 0x5A, dtype=U8, device="cuda")
        d8 = torch.full((c.M, D), 0x5A, dtype=U8, device="cuda")
        e8 = torch.full((c.M, H), 0xA5, dtype=U8, device="cuda")
        dx_d, _, _, ducs_d, amax_d, _ = do8.launch_e5_plus(c.g8, c.w8, c.sa, c.sb, c.x, c.u, c.mask, c.mean, c.rstd,
                                                           c.wn, c.bn, P_DROP, inv, dz_d.view(E5), wl, cnt, cap,
                                                           d8.view(E4), e8)
        q8, qe8 = quantize_do_e4m3_blk(dx_s.view(c.M, H, HD), c.offs, c.N)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(dx_d.view(torch.int16), dx_s.view(torch.int16)))
        self.assertTrue(torch.equal(dz_d, dz_s))  # u block, pad rows, untouched guard columns
        self.assertTrue(torch.equal(amax_d.amax().reshape(1).view(torch.int32),
                                    amax_s.amax().reshape(1).view(torch.int32)))
        self.assertTrue(torch.equal(d8.reshape(-1), q8.view(U8).reshape(-1)))
        self.assertTrue(torch.equal(e8, qe8))

    def test_bitwise_edge_lengths(self) -> None:
        self._compare(self._case(list(EDGE) * 40 + [4096, 2288, 333]))

    def test_bitwise_special_values(self) -> None:
        lens = list(EDGE) * 30 + [640, 1000]
        self._compare(self._case(lens, kind="special"))
        self._compare(self._case(lens, kind="special", sa_log2=75))  # saturation (E8 187)
        self._compare(self._case(lens, kind="special", sa_log2=-125))  # tiny groups (e = +60 clamp)

    def test_launch_failure_falls_back(self) -> None:
        c = self._case(list(EDGE) * 10)
        do8.reset_for_testing()
        do8.set_requested_for_testing(True)
        do8.set_prebuilt_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        site = SimpleNamespace(T=c.M, H=H, D=HD, N=c.N, device=c.x.device, seq_offsets=c.offs)
        inv = torch.full((1,), 2.0**-3, dtype=F32, device="cuda")
        with mock.patch.object(do8, "_fn_launch", lambda *a: 1), \
                mock.patch.object(l11, "e5_decline_reason", return_value=None), \
                mock.patch("generative_recommenders.ops.triton.bwd_fp8.grad_dtype_and_limit",
                           return_value=(E5, 57344.0)), \
                mock.patch("generative_recommenders.ops.triton.bwd_fp8.uvqk_tm_pad", return_value=128), \
                _Capture():
            out = do8.run_e5_plus(c.g8, c.w8, c.sa, c.sb, c.x, c.u, c.wn, c.bn, c.mean, c.rstd, D, 2, P_DROP, True,
                                  c.mask, DU_ROW, 0, inv, 1.0 / inv, None, site)
        self.assertIsNone(out)
        self.assertIsNotNone(do8.broken())
        self.assertIsNone(l11.broken())

    def test_prebuild_self_check(self) -> None:
        ok, detail = do8.self_check()
        self.assertTrue(ok, detail)

    def test_registry_round_trip(self) -> None:
        try:
            from generative_recommenders.ops.flydsl import hstu_do8_registry as reg
        except ImportError:
            self.skipTest("R15's hstu_do8_registry not in this tree")
        c = self._case(list(EDGE) * 10)
        with mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_BWD_DO8": "1"}):
            out = torch.empty((c.M, H, HD), dtype=BF, device="cuda")
            site = reg.new_site(out, c.offs, c.N)
            dattn = torch.randn((c.M, D), device="cuda").to(BF)
            d8 = torch.empty((c.M, D), dtype=E4, device="cuda")
            e8 = torch.empty((c.M, H), dtype=U8, device="cuda")
            do8.reset_for_testing()
            do8.set_requested_for_testing(True)
            self.assertTrue(do8.publish(site, dattn, d8, e8))
            dout = dattn.view(c.M, H, HD)
            reg.bind(site, dout)
            got = reg.take(dout, c.offs, c.N)
            self.assertIsNotNone(got)
            self.assertEqual(got[0].data_ptr(), d8.data_ptr())
            self.assertEqual(got[1].data_ptr(), e8.data_ptr())


# ---------------------------------------------------------------------------- DO8-ONLY (r16 iter42, contract v2)
ONLY_PROD = "_Z24l11_outproj_lnmd_bwd_do8" + do8._ONLY_MANGLE + "vPKhS1_PtPfPKfS5_iiPKtS7_S1_S5_S2_S2_8L11CArgs"


class _OnlyFacts:
    """Stand-in of hstu_flydsl_mha._Do8PlanFacts (as in R15's v2 tests): reason() None = the C8 do_blk branch."""

    def __init__(self, why=None) -> None:  # pyre-ignore[2]
        self.why = why

    def reason(self):  # pyre-ignore[3]
        return self.why


class _OnlyLeaf(torch.autograd.Function):
    """The attention output stand-in: [T, H, HD] from a [T, 512] leaf, so plan()'s autograd check sees a real graph."""

    @staticmethod
    def forward(ctx, x):  # pyre-ignore[2,3]
        return x.detach().view(-1, H, HD).clone()

    @staticmethod
    def backward(ctx, g):  # pyre-ignore[2,3]
        return g.reshape(-1, D)


class DO8OnlyCpuTest(unittest.TestCase):
    """DO8-ONLY on CPU through the REAL registry; run_e5_plus with the kernel launch and dwdb finalize stubbed."""

    T = 1000

    def setUp(self) -> None:
        try:
            from generative_recommenders.ops.flydsl import hstu_do8_registry as reg
        except ImportError:
            self.skipTest("R15's hstu_do8_registry not in this tree")
        if not hasattr(reg, "plan"):
            self.skipTest("R15's hook has no contract v2 (plan / cancel)")
        self.reg = reg
        keys = (reg.KNOB_ENV, reg.TAKE_ENV, reg.ONLY_ENV, reg.POISON_ENV, l11.VD_KNOB, do8.KNOB + "_SRC",
                do8.KNOB + "_FAIL_CLOSED")
        self._env = {k: os.environ.pop(k, None) for k in keys}
        os.environ.update({reg.KNOB_ENV: "1", reg.ONLY_ENV: "1", l11.VD_KNOB: "1"})
        reg.reset_for_testing()
        do8.reset_for_testing()
        do8.set_requested_for_testing(True)
        do8.set_prebuilt_for_testing(True)
        l11.reset_for_testing()
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)

    def tearDown(self) -> None:
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        do8.set_requested_for_testing(None)
        do8.reset_for_testing()
        l11.set_requested_for_testing(None)
        l11.reset_for_testing()
        self.reg.reset_for_testing()

    def graph(self, facts=None):  # pyre-ignore[2,3]
        x = torch.randn(self.T, D).to(BF).requires_grad_()
        out = _OnlyLeaf.apply(x)
        offs = torch.tensor([0, 400, self.T], dtype=torch.int64)
        self.reg.stash_plan(out, _OnlyFacts() if facts is None else facts)
        site = self.reg.new_site(out, offs, 4096)
        attn = self.reg.carry_site(out, out.view(-1, D))
        self._keep = (x, out, attn)  # the forward graph stays alive
        return site, attn

    def run_call(self, site, attn, launch=None, finalize=None):  # pyre-ignore[2,3]
        """run_e5_plus as the seam calls it (grad mode off), the launch stubbed: 'plus' writes ones into d(attn),
        both modes write dO8 / E8. Returns (result, the only flag of every launch)."""
        launches = []
        T = int(attn.shape[0])

        def fake_launch(g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, dz8, wl, cnt, cap, d8, e8,
                        only=False, dx=None):  # pyre-ignore[2,3]
            launches.append(only)
            if launch is not None:
                launch(only)
            dx_ = torch.empty_like(x) if dx is None else dx
            if not only:
                dx_.fill_(1.0)
            d8.view(U8).fill_(0x11)
            e8.fill_(0x7F)
            return dx_, torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8), 8

        wl = (torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 20)
        with mock.patch.object(l11, "_operands_ok", return_value=None), \
                mock.patch.object(l11, "e5_decline_reason", return_value=None), \
                mock.patch.object(l11, "dwdb_finalize", side_effect=finalize or (lambda *a: None)), \
                mock.patch.object(do8, "_lib", object()), \
                mock.patch.object(do8, "work_list", return_value=wl), \
                mock.patch.object(do8, "launch_e5_plus", side_effect=fake_launch), \
                mock.patch("generative_recommenders.ops.triton.bwd_fp8.grad_dtype_and_limit",
                           return_value=(E5, 57344.0)), \
                mock.patch("generative_recommenders.ops.triton.bwd_fp8.uvqk_tm_pad", return_value=128), \
                torch.no_grad():
            out = do8.run_e5_plus(torch.empty(T, D, dtype=E5), "W8", "SA", "SB", attn, torch.zeros(T, D, dtype=BF),
                                  torch.ones(D, dtype=BF), torch.zeros(D, dtype=BF), torch.zeros(T), torch.ones(T), D, 2,
                                  P_DROP, True, torch.ones(T, D, dtype=torch.int8), DU_ROW, 0, torch.ones(1),
                                  torch.ones(1), None, site)
        return out, launches

    def test_only_plan_launch_publish_take_retake_release(self) -> None:
        reg = self.reg
        site, attn = self.graph()
        do8.set_only_prebuilt_for_testing(True)
        with _Capture() as cap:
            out, launches = self.run_call(site, attn)
            self.assertEqual(launches, [True])
            self.assertIsNotNone(site.only_plan)  # open from the plan to the publish
            dx, _du, _dw, _db, _emit, d8, e8 = out
            self.assertTrue(do8.publish(site, dx, d8, e8))
        self.assertIsNone(site.only_plan)
        rec = getattr(dx, reg.HSTU_DO8_RECORD_ATTR)
        self.assertFalse(rec.bf16_valid)
        self.assertEqual(rec.producer, "L11")
        dout = dx.view(-1, H, HD)  # autograd's view backward
        self.assertTrue(reg.bind(site, dout))
        first = reg.take(dout, site.seq_offsets, site.N)
        again = reg.take(dout, site.seq_offsets, site.N)  # an M-E1 exact re-run of the same layer call
        self.assertIs(again[0], first[0])
        self.assertIs(again[1], first[1])
        self.assertEqual(first[0].data_ptr(), d8.data_ptr())
        self.assertTrue(reg.release(site))
        s = reg.stats()["only"]
        self.assertEqual((s["planned"], s["published"], s["taken"], s["retaken"], s["released"], s["cancelled"]),
                         (1, 1, 1, 1, 1, 0))
        c = do8.only_counters()
        self.assertEqual((c["planned"], c["only_launched"], c["published"], c["cancelled"], c["fallback"],
                          c["plan_plus"], c["latched"]), (1, 1, 1, 0, 0, 0, 0))
        self.assertEqual((do8.counters()["emitted"], do8.counters()["published"]), (1, 1))
        self.assertEqual(cap.text.count("armed OK: " + do8.FAMILY_ONLY + " ACTIVE mode=e5 only"), 1)
        self.assertNotIn("ACTIVE mode=e5 plus", cap.text)
        self.assertIn("[l11-do8] only counters: planned=1 only_launched=1 published=1", cap.text)

    def test_poison_fills_the_placeholder_with_nan(self) -> None:
        os.environ[self.reg.POISON_ENV] = "1"
        site, attn = self.graph()
        do8.set_only_prebuilt_for_testing(True)
        with _Capture() as cap:
            out, _ = self.run_call(site, attn)
            self.assertTrue(bool((out[0].view(torch.int16) == do8._POISON_BF16).all()))
            self.assertTrue(do8.publish(site, out[0], out[5], out[6]))
        self.assertIn("poison=1", cap.text)

    def test_knob_off_is_plus_and_never_asks_the_registry(self) -> None:
        for val in (None, "0", "", " 0 ", "true", "2"):
            os.environ.pop(self.reg.ONLY_ENV, None)
            if val is not None:
                os.environ[self.reg.ONLY_ENV] = val
            self.reg.reset_for_testing()
            do8.reset_for_testing()
            do8.set_prebuilt_for_testing(True)
            do8.set_only_prebuilt_for_testing(True)
            site, attn = self.graph()  # facts present: plan() would answer "only"
            with mock.patch.object(self.reg, "plan", side_effect=AssertionError("plan() with the knob off")), \
                    mock.patch.object(self.reg, "cancel", side_effect=AssertionError("cancel() with the knob off")), \
                    _Capture() as cap:
                out, launches = self.run_call(site, attn)
                self.assertEqual(launches, [False], val)
                self.assertTrue(do8.publish(site, out[0], out[5], out[6]))
            self.assertTrue(getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid, val)
            self.assertTrue(bool((out[0] == 1.0).all()))
            self.assertEqual(do8.only_counters(), dict.fromkeys(do8._ONLY_COUNTS, 0) | {"latched": 0})
            self.assertNotIn(do8.FAMILY_ONLY, cap.text)
            self.assertNotIn("[l11-do8] only", cap.text)
            self.assertEqual(do8._ONLY_CALL, {})

    def test_only_not_prebuilt_falls_back_without_planning(self) -> None:
        site, attn = self.graph()
        with mock.patch.object(self.reg, "plan", side_effect=AssertionError("plan() without an 'only' build")), \
                _Capture() as cap:
            out, launches = self.run_call(site, attn)
            self.assertEqual(launches, [False])
            self.assertTrue(do8.publish(site, out[0], out[5], out[6]))
        self.assertEqual(do8.only_counters()["fallback"], 1)
        self.assertEqual(cap.text.count("[l11-do8] only counters:"), 1)

    def test_plan_plus_runs_plus(self) -> None:
        site, attn = self.graph(facts=_OnlyFacts("not C8: test reason"))
        do8.set_only_prebuilt_for_testing(True)
        with _Capture():
            out, launches = self.run_call(site, attn)
            self.assertEqual(launches, [False])
            self.assertTrue(do8.publish(site, out[0], out[5], out[6]))
        self.assertTrue(getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)
        self.assertEqual(do8.only_counters()["plan_plus"], 1)
        self.assertEqual(self.reg.stats()["only"]["forced_plus"], {"not C8: test reason": 1})

    def test_only_launch_failure_cancels_and_latches_only(self) -> None:
        site, attn = self.graph()
        do8.set_only_prebuilt_for_testing(True)

        def boom(only: bool) -> None:
            if only:  # launch_e5_plus's raise for the EM = 2 kernel's rc > 0 (b4v2, r15 review N1)
                raise do8.OnlyLaunchError("l11do8_e5_only_launch rc=1 (hipError) M=1000")

        with _Capture() as cap:
            out, launches = self.run_call(site, attn, launch=boom)
        self.assertIsNone(out)  # the seam runs L11's own run_e5 (bf16 d(attn) written)
        self.assertEqual(launches, [True])
        self.assertIsNone(site.only_plan)
        self.assertEqual(self.reg.stats()["only"]["cancelled"], 1)
        self.assertIsNotNone(do8.only_broken())
        self.assertIsNone(do8.broken())
        self.assertTrue(do8.enabled())
        self.assertIn(do8.FAMILY_ONLY + " DISABLED: e5 only launch: OnlyLaunchError", cap.text)
        self.assertFalse(self.reg.bind(site, torch.empty(self.T, H, HD, dtype=BF)))  # no raise, no record
        site2, attn2 = self.graph()
        with mock.patch.object(self.reg, "plan", side_effect=AssertionError("plan() after the 'only' latch")), \
                _Capture():
            _out2, launches2 = self.run_call(site2, attn2)
        self.assertEqual(launches2, [False])
        c = do8.only_counters()
        self.assertEqual((c["planned"], c["cancelled"], c["fallback"], c["latched"]), (1, 1, 1, 1))

    def test_post_launch_failure_cancels_and_latches_do8(self) -> None:
        site, attn = self.graph()
        do8.set_only_prebuilt_for_testing(True)

        def fin(*a) -> None:  # pyre-ignore[2]
            raise RuntimeError("finalize failed")

        with _Capture() as cap:
            out, launches = self.run_call(site, attn, finalize=fin)
        self.assertIsNone(out)
        self.assertEqual(launches, [True])
        self.assertIsNone(site.only_plan)
        self.assertEqual(do8.only_counters()["cancelled"], 1)
        self.assertIsNotNone(do8.broken())
        self.assertIsNone(do8.only_broken())
        self.assertIn(do8.FAMILY + " DISABLED: e5 only call (after the EM = 2 launch): RuntimeError: finalize failed",
                      cap.text)

    def test_only_publish_errors_are_raised_not_latched(self) -> None:
        reg = self.reg
        site, attn = self.graph()
        do8.set_only_prebuilt_for_testing(True)
        with _Capture():
            out, _ = self.run_call(site, attn)
            with self.assertRaisesRegex(RuntimeError, "not the placeholder"):
                do8.publish(site, out[0].clone(), out[5], out[6])
        self.assertIsNotNone(site.only_plan)  # still open: bind would raise (in-model the raise aborts the backward)
        with self.assertRaises(reg.Do8OnlyError):
            reg.bind(site, out[0].view(-1, H, HD))
        site2, attn2 = self.graph()
        with _Capture():
            out2, _ = self.run_call(site2, attn2)
            with self.assertRaises(reg.Do8OnlyError):  # a registry check fails: re-raised, never swallowed
                do8.publish(site2, out2[0], out2[5], out2[6][:, :2].contiguous())
        self.assertIsNone(do8.broken())
        self.assertIsNone(do8.only_broken())
        self.assertEqual(reg.stats()["refused"], {"only_shape": 1})

    def test_interleaved_only_call_never_publishes_a_placeholder_as_plus(self) -> None:
        """review_do8only P1 (hardening): publish() routes 'only' through the single in-flight slot _ONLY_CALL. If any
        other DO8 call planned between an 'only' call's launch and its publish (not reachable in today's seam, where
        run_e5_plus and publish run back to back in one backward), the slot no longer names the site and the
        unwritten placeholder fell through to the 'plus' publish (bf16_written=True), which the registry accepts:
        a later M-E1 re-run would then re-quantize the unwritten bf16 dO. It must fail closed instead."""
        reg = self.reg
        do8.set_only_prebuilt_for_testing(True)
        for second in ("only", "plus"):
            reg.reset_for_testing()
            site_a, attn_a = self.graph()
            keep_a = self._keep
            site_b, attn_b = self.graph(facts=None if second == "only" else _OnlyFacts("not C8: test reason"))
            with _Capture():
                out_a, la = self.run_call(site_a, attn_a)
                out_b, lb = self.run_call(site_b, attn_b)
                self.assertEqual((la, lb), ([True], [second == "only"]), second)
                with self.assertRaisesRegex(RuntimeError, "placeholder", msg=second):
                    do8.publish(site_a, out_a[0], out_a[5], out_a[6])
                self.assertIsNone(getattr(out_a[0], reg.HSTU_DO8_RECORD_ATTR, None), second)  # nothing published
                self.assertIsNotNone(site_a.only_plan, second)  # still open: A's bind would raise (N3 wording)
                with self.assertRaises(reg.Do8OnlyError):
                    reg.bind(site_a, out_a[0].view(-1, H, HD))
                self.assertTrue(do8.publish(site_b, out_b[0], out_b[5], out_b[6]), second)  # B is unaffected
            self.assertEqual(getattr(out_b[0], reg.HSTU_DO8_RECORD_ATTR).bf16_valid, second == "plus", second)
            del keep_a

    # ------------------------------------------------------------ b4v2, r15 review N1: the REAL launch_e5_plus
    def run_call_real(self, site, attn, rc_only=0, rc_plus=0, wl_dtype=torch.int32, plan_rc=0, drain=None,
                      fail_alloc=None):  # pyre-ignore[2,3]
        """run_e5_plus as the seam calls it, with the REAL launch_e5_plus (layout checks, plan(cap), the drain, the
        placeholder / partials allocations, the rc handling). Stubbed: the ctypes entry points (_fn_launch,
        _fn_launch_only, _fn_drain, _fn_plan), the CUDA device / stream calls, L11's operand checks and the dW/dB
        finalize. fail_alloc: "placeholder" (torch.empty_like(attn) raises OOM) or "partials" (the first float32
        [P, 512] partial raises OOM). Returns (result, the entry points the launch called: 'only' / 'plus')."""
        T = int(attn.shape[0])
        calls = []
        oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        real_empty, real_empty_like = torch.empty, torch.empty_like

        def entry(tag: str, rc: int):  # pyre-ignore[3]
            def fn(*args):  # pyre-ignore[2,3]
                calls.append(tag)
                return rc
            return fn

        def fn_plan(cap, addr):  # pyre-ignore[2,3]
            out = ctypes.cast(addr, ctypes.POINTER(ctypes.c_longlong))
            for i, v in enumerate((1, 0, 1, 8)):  # G, fl, nf, P
                out[i] = v
            return plan_rc

        def empty_like(t, *a, **k):  # pyre-ignore[2,3]
            if t is attn:
                raise oom("injected: placeholder d(attn) allocation")
            return real_empty_like(t, *a, **k)

        def empty(*a, **k):  # pyre-ignore[2,3]
            if a and isinstance(a[0], tuple) and a[0] == (8, D) and k.get("dtype") is torch.float32:
                raise oom("injected: DWP partials allocation")
            return real_empty(*a, **k)

        wl = (torch.zeros(20, 2, dtype=wl_dtype), torch.zeros(2, dtype=torch.int32), 20)
        with contextlib.ExitStack() as st:
            for target, attr, kw in (
                (l11, "_operands_ok", {"return_value": None}), (l11, "e5_decline_reason", {"return_value": None}),
                (l11, "dwdb_finalize", {"new": lambda *a: None}), (do8, "_lib", {"new": object()}),
                (do8, "_only_lib", {"new": object()}), (do8, "_fn_launch", {"new": entry("plus", rc_plus)}),
                (do8, "_fn_launch_only", {"new": entry("only", rc_only)}),
                (do8, "_fn_drain", {"new": drain or (lambda: 0)}), (do8, "_fn_plan", {"new": fn_plan}),
                (do8, "work_list", {"return_value": wl}), (torch.cuda, "current_device", {"return_value": 0}),
                (torch.cuda, "device", {"new": lambda i: contextlib.nullcontext()}),
                (torch.cuda, "current_stream", {"new": lambda d=None: SimpleNamespace(cuda_stream=0)}),
            ):
                st.enter_context(mock.patch.object(target, attr, **kw))
            st.enter_context(mock.patch.dict(do8._plan_cache, clear=True))
            if fail_alloc == "placeholder":
                st.enter_context(mock.patch.object(torch, "empty_like", side_effect=empty_like))
            elif fail_alloc == "partials":
                st.enter_context(mock.patch.object(torch, "empty", side_effect=empty))
            st.enter_context(mock.patch("generative_recommenders.ops.triton.bwd_fp8.grad_dtype_and_limit",
                                        return_value=(E5, 57344.0)))
            st.enter_context(mock.patch("generative_recommenders.ops.triton.bwd_fp8.uvqk_tm_pad", return_value=128))
            st.enter_context(torch.no_grad())
            out = do8.run_e5_plus(torch.zeros(T, D, dtype=E5), torch.zeros(1, dtype=E4), torch.ones(1), torch.ones(1),
                                  attn, torch.zeros(T, D, dtype=BF), torch.ones(D, dtype=BF), torch.zeros(D, dtype=BF),
                                  torch.zeros(T), torch.ones(T), D, 2, P_DROP, True, torch.ones(T, D, dtype=torch.int8),
                                  DU_ROW, 0, torch.ones(1), torch.ones(1), None, site)
        return out, calls

    def test_n1_real_launch_routes_only_and_plus(self) -> None:
        """The REAL launch_e5_plus calls l11do8_e5_only_launch for an 'only' plan and l11do8_e5_plus_launch for a
        'plus' one; the results publish 'only' (bf16_written=False) resp. 'plus'."""
        do8.set_only_prebuilt_for_testing(True)
        for facts, want, valid in ((None, ["only"], False), (_OnlyFacts("not C8: test reason"), ["plus"], True)):
            site, attn = self.graph(facts=facts)
            with _Capture():
                out, calls = self.run_call_real(site, attn)
                self.assertEqual(calls, want)
                self.assertTrue(do8.publish(site, out[0], out[5], out[6]))
            self.assertEqual(getattr(out[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid, valid)
            self.assertEqual((out[0].dtype, tuple(out[0].shape)), (attn.dtype, tuple(attn.shape)))
        self.assertIsNone(do8.broken())
        self.assertIsNone(do8.only_broken())

    def test_n1_em2_kernel_failure_latches_only_and_plus_continues(self) -> None:
        """r15 review N1: the EM = 2 kernel's own rc > 0 (OnlyLaunchError, raised by the REAL launch_e5_plus) is the one
        failure that latches 'only' alone: plan cancelled, 'only' latched ("e5 only launch"), DO8 still enabled, and
        the next call launches 'plus' (EM = 1) through the same real launch path."""
        site, attn = self.graph()
        do8.set_only_prebuilt_for_testing(True)
        with _Capture() as cap:
            out, calls = self.run_call_real(site, attn, rc_only=1)
        self.assertIsNone(out)
        self.assertEqual(calls, ["only"])
        self.assertIsNone(site.only_plan)
        self.assertEqual(self.reg.stats()["only"]["cancelled"], 1)
        self.assertEqual(do8.only_broken(),
                         "e5 only launch: OnlyLaunchError: l11do8_e5_only_launch rc=1 (hipError) M=1000")
        self.assertIsNone(do8.broken())
        self.assertEqual(cap.text.count(do8.FAMILY_ONLY + " DISABLED: e5 only launch: OnlyLaunchError"), 1)
        self.assertNotIn(do8.FAMILY + " DISABLED", cap.text)
        self.assertFalse(self.reg.bind(site, torch.empty(self.T, H, HD, dtype=BF)))  # cancelled: no raise, no record
        site2, attn2 = self.graph()
        with _Capture():
            out2, calls2 = self.run_call_real(site2, attn2)
            self.assertEqual(calls2, ["plus"])
            self.assertTrue(do8.publish(site2, out2[0], out2[5], out2[6]))
        self.assertTrue(getattr(out2[0], self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)
        c = do8.only_counters()
        self.assertEqual((c["planned"], c["cancelled"], c["fallback"], c["latched"]), (1, 1, 1, 1))

    def test_n1_failures_before_the_em2_kernel_cancel_and_latch_do8(self) -> None:
        """r15 review N1: a failure between the plan and the EM = 2 kernel that a 'plus' call can have too (a
        launch_e5_plus layout check, plan(cap), the drain, the placeholder or partials allocation incl. OOM) cancels
        the plan and latches DO8, as 'plus' does, with a reason naming the stage; 'only' is NOT latched, no kernel ran,
        bind() sees no open plan, and the next call is declined by DO8's latch (L11's own launches)."""
        oom = getattr(torch.cuda, "OutOfMemoryError", RuntimeError).__name__

        def bad_drain() -> int:
            raise RuntimeError("injected: drain failed")

        cases = (
            ("layout", {"wl_dtype": torch.int64},
             "RuntimeError: work list layout (20, 2) torch.int64 / (2,) torch.int32"),
            ("plan(cap)", {"plan_rc": 3}, "RuntimeError: l11do8_plan(cap=20) rc=3"),
            ("drain", {"drain": bad_drain}, "RuntimeError: injected: drain failed"),
            ("placeholder OOM", {"fail_alloc": "placeholder"}, oom + ": injected: placeholder d(attn) allocation"),
            ("partials OOM", {"fail_alloc": "partials"}, oom + ": injected: DWP partials allocation"),
        )
        for name, kw, want in cases:
            self.reg.reset_for_testing()
            do8.reset_for_testing()
            do8.set_prebuilt_for_testing(True)
            do8.set_only_prebuilt_for_testing(True)
            site, attn = self.graph()
            with _Capture() as cap:
                out, calls = self.run_call_real(site, attn, **kw)
            self.assertIsNone(out, name)
            self.assertEqual(calls, [], name)  # neither kernel ran
            self.assertIsNone(site.only_plan, name)
            self.assertEqual(self.reg.stats()["only"]["cancelled"], 1, name)
            self.assertIsNone(do8.only_broken(), name)
            self.assertEqual(do8.broken(), "e5 only call (before the EM = 2 launch): " + want, name)
            self.assertEqual(cap.text.count(do8.FAMILY + " DISABLED: e5 only call (before the EM = 2 launch): "), 1,
                             name)
            self.assertNotIn(do8.FAMILY_ONLY + " DISABLED", cap.text, name)
            c = do8.only_counters()
            self.assertEqual((c["planned"], c["only_launched"], c["cancelled"], c["latched"]), (1, 0, 1, 0), name)
            self.assertFalse(self.reg.bind(site, torch.empty(self.T, H, HD, dtype=BF)), name)
            with _Capture():
                self.assertIsNone(do8.emission_site(SimpleNamespace(do8_site=site), attn), name)  # DO8 latched

    def test_n1_plan_left_open_inside_plan_only_is_cancelled(self) -> None:
        """r15 review N1 (optional fix 2): a raise inside _plan_only after registry.plan answered "only" (the local
        `only` still False) used to leave the plan open, so bind() raised; the except clause now cancels any plan open
        on the site and latches DO8 (the call never decided 'only': reason "e5 plus")."""
        site, attn = self.graph()
        do8.set_only_prebuilt_for_testing(True)
        orig = do8._only_event

        def event(key: str) -> None:
            if key == "planned":
                raise RuntimeError("injected: _only_event(planned) failed")
            orig(key)

        with mock.patch.object(do8, "_only_event", side_effect=event), _Capture():
            out, launches = self.run_call(site, attn)
        self.assertIsNone(out)
        self.assertEqual(launches, [])
        self.assertIsNone(site.only_plan)
        self.assertEqual(self.reg.stats()["only"]["cancelled"], 1)
        self.assertEqual(do8.only_counters()["cancelled"], 1)
        self.assertEqual(do8.broken(), "e5 plus: RuntimeError: injected: _only_event(planned) failed")
        self.assertFalse(self.reg.bind(site, torch.empty(self.T, H, HD, dtype=BF)))  # B4p1: Do8OnlyError here

    def test_n1_knob_off_real_launch_failure_keeps_b3s_latch(self) -> None:
        """Knob unset or "0", the REAL launch_e5_plus: a 'plus' kernel rc > 0 is a plain RuntimeError and latches DO8
        with B3's exact reason; the registry's plan / cancel are never called and nothing 'only' is printed."""
        for val in (None, "0"):
            os.environ.pop(self.reg.ONLY_ENV, None)
            if val is not None:
                os.environ[self.reg.ONLY_ENV] = val
            self.reg.reset_for_testing()
            do8.reset_for_testing()
            do8.set_prebuilt_for_testing(True)
            do8.set_only_prebuilt_for_testing(True)
            site, attn = self.graph()
            with mock.patch.object(self.reg, "plan", side_effect=AssertionError("plan() with the knob off")), \
                    mock.patch.object(self.reg, "cancel", side_effect=AssertionError("cancel() with the knob off")), \
                    _Capture() as cap:
                out, calls = self.run_call_real(site, attn, rc_plus=1)
            self.assertIsNone(out, val)
            self.assertEqual(calls, ["plus"], val)
            self.assertEqual(do8.broken(), "e5 plus: RuntimeError: l11do8_e5_plus_launch rc=1 (hipError) M=1000", val)
            self.assertIsNone(do8.only_broken(), val)
            self.assertNotIn(do8.FAMILY_ONLY, cap.text, val)
            self.assertEqual(do8._ONLY_CALL, {}, val)

    def test_n1_fail_closed_cancels_before_raising_on_both_latch_routes(self) -> None:
        """r16 iteration 43 review F1 (the body of r16 probes/iter43/b4v2/review/n1_fail_closed_test.py, adopted): with
        DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1 a failed 'only' call cancels its plan BEFORE the latch raises, on BOTH v2
        latch routes: the EM = 2 kernel's rc > 0 (OnlyLaunchError -> _disable_only) and a placeholder OOM before the
        launch (-> _disable). Latching first would raise with the plan still open, and bind() would then raise too."""
        for name, kw, family in (("EM 2 kernel rc", {"rc_only": 1}, do8.FAMILY_ONLY),
                                 ("placeholder OOM", {"fail_alloc": "placeholder"}, do8.FAMILY)):
            self.reg.reset_for_testing()
            do8.reset_for_testing()
            do8.set_prebuilt_for_testing(True)
            do8.set_only_prebuilt_for_testing(True)
            site, attn = self.graph()
            os.environ[do8.KNOB + "_FAIL_CLOSED"] = "1"
            try:
                with _Capture():
                    with self.assertRaisesRegex(RuntimeError, family + " failed with .*FAIL_CLOSED=1", msg=name):
                        self.run_call_real(site, attn, **kw)
            finally:
                os.environ.pop(do8.KNOB + "_FAIL_CLOSED", None)
            self.assertIsNone(site.only_plan, name)  # cancelled before the raise: bind() would not raise
            self.assertEqual(self.reg.stats()["only"]["cancelled"], 1, name)
            self.assertEqual(do8.only_counters()["cancelled"], 1, name)
            self.assertFalse(self.reg.bind(site, torch.empty(self.T, H, HD, dtype=BF)), name)

    def _prebuild_mocks(self):  # pyre-ignore[3]
        stack = contextlib.ExitStack()
        for target, attr, kw in ((torch.cuda, "is_available", {"return_value": True}),
                                 (torch.version, "hip", {"new": "7.2"}),
                                 (torch.cuda, "current_device", {"return_value": 0}),
                                 (torch.cuda, "synchronize", {}),
                                 (l11, "device_arch", {"return_value": "gfx950"}),
                                 (do8, "_load", {})):
            stack.enter_context(mock.patch.object(target, attr, **kw))
        do8.set_prebuilt_for_testing(False)
        return stack

    def test_prebuild_only_failures_latch_only(self) -> None:
        cases = (
            ({"_load_only": {"side_effect": RuntimeError("resource check: x VGPR spill 3, scratch 0 B/lane")}},
             "DISABLED: prebuild build/load: RuntimeError: resource check"),
            ({"_load_only": {}, "self_check_only": {"return_value": (False, "dO8 mism 2")}},
             "DISABLED: prebuild self-check mismatch: dO8 mism 2"),
            ({"_load_only": {}, "self_check_only": {"side_effect": ValueError("bad")}},
             "DISABLED: prebuild self-check: ValueError: bad"),
        )
        for patches, want in cases:
            do8.reset_for_testing()
            with self._prebuild_mocks(), contextlib.ExitStack() as st, _Capture() as cap:
                for attr, kw in patches.items():
                    st.enter_context(mock.patch.object(do8, attr, **kw))
                self.assertTrue(do8.prebuild(True, True, self_check_fn=lambda: (True, "ok")))
            self.assertTrue(do8.prebuilt())
            self.assertIsNone(do8.broken())
            self.assertFalse(do8.only_prebuilt())
            self.assertIsNotNone(do8.only_broken())
            self.assertEqual(cap.text.count(do8.FAMILY_ONLY + " " + want), 1, cap.text)
            self.assertIn("armed OK: " + do8.FAMILY + " prebuild before run_start", cap.text)
            self.assertIn("[l11-do8] only counters:", cap.text)

    def test_prebuild_only_ok_and_device_fault(self) -> None:
        with self._prebuild_mocks(), mock.patch.object(do8, "_load_only"), \
                mock.patch.object(do8, "self_check_only", return_value=(True, "EM=2 vs EM=1 dO8 mism 0")), \
                _Capture() as cap:
            self.assertTrue(do8.prebuild(True, True, self_check_fn=lambda: (True, "ok")))
        self.assertTrue(do8.only_prebuilt())
        self.assertIn("armed OK: " + do8.FAMILY_ONLY + " prebuild before run_start: built+loaded+self-check OK", cap.text)
        do8.reset_for_testing()

        def fault():  # pyre-ignore[3]
            raise RuntimeError("HIP error: an illegal memory access was encountered")

        with self._prebuild_mocks(), mock.patch.object(do8, "_load_only"), \
                mock.patch.object(do8, "self_check_only", side_effect=fault), _Capture() as cap:
            with self.assertRaises(RuntimeError):
                do8.prebuild(True, True, self_check_fn=lambda: (True, "ok"))
        self.assertIn(do8.FAMILY_ONLY + " prebuild DEVICE FAULT", cap.text)

    def test_prebuild_only_inert_without_vd_and_quiet_with_knob_off(self) -> None:
        for env, want in (({l11.VD_KNOB: "0"}, do8.FAMILY_ONLY + " INERT: needs DO8-VD"), ({self.reg.ONLY_ENV: "0"}, None)):
            do8.reset_for_testing()
            with mock.patch.dict(os.environ, env), self._prebuild_mocks(), \
                    mock.patch.object(do8, "_load_only", side_effect=AssertionError("built")), _Capture() as cap:
                self.assertTrue(do8.prebuild(True, True, self_check_fn=lambda: (True, "ok")))
            self.assertFalse(do8.only_prebuilt())
            self.assertIsNone(do8.only_broken())
            if want:
                self.assertEqual(cap.text.count(want), 1)
            else:
                self.assertNotIn(do8.FAMILY_ONLY, cap.text)

    def test_only_resource_gate_flags_and_key(self) -> None:
        ok = do8.check_resources([_remark(ONLY_PROD, vgpr=251), _remark(WLB, vgpr=40, lds=136)], do8._ONLY_MANGLE)
        self.assertEqual((ok["vgpr"], ok["vgpr_spill"], ok["scratch"], ok["lds"]), (251, 0, 0, 163840))
        self.assertEqual(do8._ONLY_MANGLE, do8._PROD_MANGLE.replace("ELi1ELi0ELi2ELi0EE", "ELi2ELi0ELi2ELi0EE"))
        for kernels, mangle in (
            ([_remark(PROD), _remark(WLB, lds=136)], do8._ONLY_MANGLE),  # a 'plus' .so offered as 'only'
            ([_remark(ONLY_PROD), _remark(WLB, lds=136)], do8._PROD_MANGLE),  # and the reverse
            ([_remark(ONLY_PROD, spill=2), _remark(WLB, lds=136)], do8._ONLY_MANGLE),
            ([_remark(ONLY_PROD, scratch=4), _remark(WLB, lds=136)], do8._ONLY_MANGLE),
            ([_remark(ONLY_PROD, lds=98304), _remark(WLB, lds=136)], do8._ONLY_MANGLE),
            ([_remark(ONLY_PROD), _remark(PROD), _remark(WLB, lds=136)], do8._ONLY_MANGLE),
        ):
            with self.assertRaises(RuntimeError):
                do8.check_resources(kernels, mangle)
        self.assertEqual(do8._flags(b"x", True), do8._flags(b"x") + ["-DL11DO8_ONLY=1"])
        self.assertNotIn("-DL11DO8_ONLY=1", do8._flags(b"x"))
        with mock.patch.object(l11, "_toolchain", return_value="hipcc test"):
            self.assertNotEqual(do8.cache_key(do8._SOURCE), do8.cache_key(do8._SOURCE, only=True))

    def test_cu_build_switch_and_dx_store(self) -> None:
        src, novd = do8._SOURCE.read_text(), do8._SOURCE_NOVD.read_text()
        self.assertEqual(src.count("#if L11DO8_ONLY"), 3)  # the impl tag, the launch entry, (r16 MX6) the MX6 entry
        self.assertIn("#define L11DO8_E5_LAUNCH l11do8_e5_only_launch", src)
        self.assertIn("#define L11DO8_E5_LAUNCH l11do8_e5_plus_launch", src)
        self.assertEqual(src.count("(l11_outproj_lnmd_bwd_do8<1, 2, 0, 0, 1, 2, 1, 0, 1, 2, L11DO8_ONLY ? 2 : 1, 0, 2, 0>)"), 1)  # the base's 14-parameter FP8 kernel
        self.assertEqual(src.count("(l11_outproj_lnmd_bwd_do8_mx6<1, 2, 0, 0, 1, 2, 1, 0, 1, 2, L11DO8_ONLY ? 2 : 1, 0, 2, 0>)"), 1)  # r16 MX6: the separately named MX6 kernel
        self.assertIn("EM 2 only no bf16 DX store", src)
        self.assertEqual(src.count("if constexpr (EM != 2) l11_llvm_buffer_store_v4i32(ox, rDX"), 1)
        self.assertIn("static_assert(!(EM == 2 && RET == 2)", src)
        self.assertNotIn("L11DO8_ONLY", novd)  # the pre-diet source has no 'only' build ('plus' only)

    def test_seam_carries_no_only_logic(self) -> None:
        """The 'only' decision, launch, cancel and publish live in l11_do8.py; the B3 seam is unchanged."""
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        src = Path(thl.__file__).read_text()
        for word in ("bf16_written", "_ONLY", ".plan(", "cancel(", "poison"):
            self.assertNotIn(word, src[src.index("class HSTUComputeOutputFunction"):])

    def test_only_lines_carry_family_and_rank(self) -> None:
        with _Capture() as cap:
            do8._only_event("fallback")
            do8._disable_only("test")
        lines = cap.text.splitlines()
        self.assertTrue(lines)
        for ln in lines:
            self.assertIn("[rank ", ln)
            self.assertTrue(ln.startswith(do8.FAMILY_ONLY) or ln.startswith("[l11-do8] only counters:"), ln)


@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU")
class DO8OnlyGpuTest(unittest.TestCase):
    """DO8-ONLY on gfx950: the EM = 2 self-check, and EM = 2 vs EM = 1 byte for byte (dO8, E8, dz8, partials) with
    the EM = 2 d(attn) buffer's sentinel untouched, on the DO8GpuTest cases."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("DLRM_HSTU_FP8_MASK16", "1")
        os.environ.setdefault("DLRM_HSTU_FP8_LNMD_BWD_V2", "1")
        cls._vd = os.environ.get(l11.VD_KNOB)
        os.environ[l11.VD_KNOB] = "1"  # 'only' is built from the DO8-VD source only
        do8.set_diet_for_testing(True)  # item (89): the gin default
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(True)
        l11.set_requested_for_testing(True)
        l11._load()
        do8._load()
        do8._load_only()

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop(l11.VD_KNOB, None)
        if cls._vd is not None:
            os.environ[l11.VD_KNOB] = cls._vd
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(None)
        do8.set_diet_for_testing(None)
        l11.set_requested_for_testing(None)

    def test_only_self_check(self) -> None:
        ok, detail = do8.self_check_only()
        self.assertTrue(ok, detail)

    def _compare_only(self, c, inv_v: float = 2.0**-3) -> None:  # pyre-ignore[2]
        rows_out = (c.M + 127) // 128 * 128
        inv = torch.full((1,), inv_v, dtype=F32, device="cuda")
        wl, cnt, cap = do8.work_list(c.offs, c.M, c.N, c.x.device)
        got = {}
        for only in (False, True):
            dz = torch.full((rows_out, DU_ROW), 0x5A, dtype=U8, device="cuda")
            d8 = torch.full((c.M, D), 0x5A, dtype=U8, device="cuda")
            e8 = torch.full((c.M, H), 0xA5, dtype=U8, device="cuda")
            dxb = torch.full((c.M, D), do8._SENTINEL_BF16, dtype=torch.int16, device="cuda").view(BF) if only else None
            r = do8.launch_e5_plus(c.g8, c.w8, c.sa, c.sb, c.x, c.u, c.mask, c.mean, c.rstd, c.wn, c.bn, P_DROP, inv,
                                   dz.view(E5), wl, cnt, cap, d8.view(E4), e8, only=only, dx=dxb)
            got[only] = (r, dz, d8, e8)
        torch.cuda.synchronize()
        (rp, dzp, d8p, e8p), (ro, dzo, d8o, e8o) = got[False], got[True]

        def same(a: torch.Tensor, b: torch.Tensor) -> bool:
            return torch.equal(a.contiguous().view(U8), b.contiguous().view(U8))

        self.assertEqual(rp[5], ro[5])  # P
        for name, a, b in (("dz8", dzp, dzo), ("dO8", d8p, d8o), ("E8", e8p, e8o), ("DWP", rp[1], ro[1]),
                           ("DBP", rp[2], ro[2]), ("DUCS", rp[3], ro[3]), ("AMAXP", rp[4], ro[4])):
            self.assertTrue(same(a, b), name)
        self.assertTrue(bool((ro[0].view(torch.int16) == do8._SENTINEL_BF16).all()))  # no DX store
        self.assertFalse(bool((rp[0].view(torch.int16) == do8._SENTINEL_BF16).any()))

    def test_only_vs_plus_edge_lengths(self) -> None:
        self._compare_only(DO8GpuTest._case(self, list(EDGE) * 40 + [4096, 2288, 333]))

    def test_only_vs_plus_special_values(self) -> None:
        lens = list(EDGE) * 30 + [640, 1000]
        self._compare_only(DO8GpuTest._case(self, lens, kind="special"))
        self._compare_only(DO8GpuTest._case(self, lens, kind="special", sa_log2=75))
        self._compare_only(DO8GpuTest._case(self, lens, kind="special", sa_log2=-125))


# ---------------------------------------------------------------------------- DO8 DIET (r16 iter52, gin item (89))
@unittest.skipUnless(_gfx950(), "needs a gfx950 GPU")
class DO8DietGpuTest(unittest.TestCase):
    """DO8 DIET on gfx950: the diet build (L11DO8_DIET 1, the gin default) vs the OFF build (-DL11DO8_DIET=0, ship14's
    c68395d kernel body), 'plus' and 'only', byte for byte on every output (dO8, E8, the whole dz8, d(attn) -- the
    sentinel for 'only' --, AMAXP / DWP / DBP / DUCS and P), on the DO8GpuTest cases; and DO8's and DO8-ONLY's
    prebuild self-checks on the OFF build."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("DLRM_HSTU_FP8_MASK16", "1")
        os.environ.setdefault("DLRM_HSTU_FP8_LNMD_BWD_V2", "1")
        cls._vd = os.environ.get(l11.VD_KNOB)
        os.environ[l11.VD_KNOB] = "1"  # the switch lives in the DO8-VD source l11_do8.cu
        l11.set_requested_for_testing(True)
        l11._load()

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop(l11.VD_KNOB, None)
        if cls._vd is not None:
            os.environ[l11.VD_KNOB] = cls._vd
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(None)
        do8.set_diet_for_testing(None)
        l11.set_requested_for_testing(None)

    def _use(self, diet: bool, only: bool) -> None:
        do8.set_diet_for_testing(diet)
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(True)
        do8._load()
        if only:
            do8._load_only()
        self.assertEqual("-DL11DO8_DIET=0" in do8._flags(do8._SOURCE.read_bytes(), only), not diet)

    def _run(self, c, diet: bool, only: bool, inv_v: float = 2.0**-3):  # pyre-ignore[2,3]
        self._use(diet, only)
        lib = do8.only_library_path() if only else do8.library_path()
        rows_out = (c.M + 127) // 128 * 128
        inv = torch.full((1,), inv_v, dtype=F32, device="cuda")
        wl, cnt, cap = do8.work_list(c.offs, c.M, c.N, c.x.device)
        dz = torch.full((rows_out, DU_ROW), 0x5A, dtype=U8, device="cuda")
        d8 = torch.full((c.M, D), 0x5A, dtype=U8, device="cuda")
        e8 = torch.full((c.M, H), 0xA5, dtype=U8, device="cuda")
        dxb = torch.full((c.M, D), do8._SENTINEL_BF16, dtype=torch.int16, device="cuda").view(BF) if only else None
        dx, dwp, dbp, ducs, amaxp, P = do8.launch_e5_plus(c.g8, c.w8, c.sa, c.sb, c.x, c.u, c.mask, c.mean, c.rstd,
                                                          c.wn, c.bn, P_DROP, inv, dz.view(E5), wl, cnt, cap,
                                                          d8.view(E4), e8, only=only, dx=dxb)
        torch.cuda.synchronize()
        out = {"dx": dx, "dz8": dz, "dO8": d8, "E8": e8, "DWP": dwp, "DBP": dbp, "DUCS": ducs, "AMAXP": amaxp}
        return lib, P, {k: v.contiguous().view(U8).reshape(-1).cpu() for k, v in out.items()}

    def _compare_diet(self, c) -> None:  # pyre-ignore[2]
        for only in (False, True):
            lib_on, p_on, on = self._run(c, True, only)
            lib_off, p_off, off = self._run(c, False, only)
            self.assertNotEqual(lib_on, lib_off)  # two builds (own cache keys)
            self.assertEqual(p_on, p_off, only)
            for name in on:
                self.assertTrue(torch.equal(on[name], off[name]), (name, only))
            if only:  # no DX store in either build
                self.assertTrue(bool((on["dx"].view(torch.int16) == do8._SENTINEL_BF16).all()))

    def test_diet_on_vs_off_edge_lengths(self) -> None:
        self._compare_diet(DO8GpuTest._case(self, list(EDGE) * 40 + [4096, 2288, 333]))

    def test_diet_on_vs_off_special_values(self) -> None:
        lens = list(EDGE) * 30 + [640, 1000]
        self._compare_diet(DO8GpuTest._case(self, lens, kind="special"))
        self._compare_diet(DO8GpuTest._case(self, lens, kind="special", sa_log2=75))  # saturation (E8 187)
        self._compare_diet(DO8GpuTest._case(self, lens, kind="special", sa_log2=-125))  # e = +60 clamp groups

    def test_off_build_self_checks(self) -> None:
        self._use(False, True)
        ok, detail = do8.self_check()
        self.assertTrue(ok, detail)
        ok, detail = do8.self_check_only()
        self.assertTrue(ok, detail)


if __name__ == "__main__":
    unittest.main()

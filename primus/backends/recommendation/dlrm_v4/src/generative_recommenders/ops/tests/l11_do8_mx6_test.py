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
"""Tests of MX6's loader rows (gin item (104), DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6; ops/hip/l11_do8.py, module docstring
"MX6"). CPU only (no kernel launch, no hipcc):
  * build: _flags(src, only, mx6) appends -DL11DO8_MX6=1 LAST and leaves the 'plus' / 'only' lists, commands and cache
    keys exactly the base's (a written-out copy of the base formula); mx6 implies only in every entry point; the MX6
    .so's own name, cache entry and gate (build() with a stubbed hipcc fed the production build log), whose SGPR rule
    follows the build's NIB flags;
  * gate: check_resources_mx6 accepts the production (ROCm 7.2.1) remarks of the landing source's MX6 'only' builds
    (int8, NIB with the NIB rule, DIET=0) and rejects mutations, the MX6 'plus' build and the FP8 'only' builds; the
    FP8 check_resources never accepts the MX6 .so and still accepts the landing FP8 builds (fixtures transcribed from
    the kitLB build-only lab job's logs; a provenance test re-parses them where L11DO8_MX6_KITLB_DIR names them);
  * load: ABI 2 authoritative, the impl tags (EM 2 only, the MX6 tag, NIB iff the 'only' .so), 30 argtypes matching
    l11_do8.cu's C signature (the FP8 entry's with G8 -> G6, W8 -> W6 and W6S after W6), RTLD_LOCAL (no mode), the
    FP8 twin's resources equal to the loaded 'only' .so's kernel;
  * knob / INERT matrix / prebuild: nothing read, built or printed with the knob off; one INERT line and nothing built
    per rule (site set, a site set holding layer 0 while r14's E-RM row is requested, DO8 / DO8-ONLY state, DO8-VD, a
    source without the switch, DIET, a NIB 'only' .so with the MX6_NIB_GATED switch off, r14's producer API absent:
    today's tree, also with a NIB 'only' .so); the E-RM rule reads E-RM's request only (a latched E-RM: still INERT),
    after the site-set rule and before the others, also on triton_hstu_linear.prebuild_l11_do8's DO8-off path; the top
    set beside E-RM, and bottom / all without it, arm; with the switch off the NIB rule comes before the producer
    rule; MX6_NIB_GATED pinned True (kitN's N_PASS: with a NIB 'only' .so the INERT rules pass, no INERT line, the
    armed line says mask=nib); a prebuild failure latches MX6 only; a device fault re-raises; FAIL_CLOSED raises;
    prebuild order (DO8, DO8-ONLY, MX6) and the DO8-off path of triton_hstu_linear.prebuild_l11_do8; the prebuild's
    lines (re-cut v2): a success prints, once each and in this order, the prebuild armed line, MX6's ACTIVE line
    (item (104), "before run_start", ending "impl=<impl> device=<device> sites=<set> poison=<0|1>"; the in-model
    per-rank marker regex matches it) and the counters line; a second prebuild_mx6 prints nothing; the INERT /
    DISABLED / self-check-fail / device-fault / knob-off paths print no ACTIVE line (the seam test: an MX6 record
    prints no MX6 line);
  * latch isolation, lines and counters (incl. the exit line through _only_final);
  * the self-check's lattice: exact in e5m2 / e4m3 at power-of-two scales, dot products below 2^15;
  * self_check_mx6 itself on CPU (device, Triton, ctypes and r14's encode faked): every MX6-.so launch gets the keep
    mask in the loaded MX6 .so's format (a NIB .so: lnmd_mask_nib.pack_ref(mask) at all four launches, the round trip
    in the verdict and the detail; an int8 .so: the int8 mask itself);
  * the raw MX6 launch rejects a keep mask the loaded MX6 .so does not read (format or layout) before the entry;
  * the producer-contract and operand checks (the W3 dout_t passes the contract only if x21's own dout_t checks
    accept it, 16-B alignment included), and R2's (mx6_rescue: no scale-state update, the FP8 dgrad operands).
The seam (decide / produce / launch / release / rescue / publish through the REAL dO8 registry and the REAL
HSTUComputeOutputFunction.backward) is l11_do8_mx6_seam_test.py. GPU: the prebuild and self_check_mx6 on gfx950, int8
and NIB builds, once r14's producer API exists (skipped until then)."""

import contextlib
import ctypes
import hashlib
import io
import json
import math
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest import mock

import torch

from generative_recommenders.ops import lnmd_mask_nib
from generative_recommenders.ops.hip import l11_do8 as do8
from generative_recommenders.ops.hip import l11_outproj_lnmd as l11

E4, E5, BF, F32, U8 = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float32, torch.uint8
D = 512

# ------------------------------------------------------------------------------------------------ fixtures
_SIG = "vPKhS1_PtPfPKfS5_iiPKtS7_S1_S5_S2_S2_8L11CArgs"
FP8, MX6 = "_Z24l11_outproj_lnmd_bwd_do8", "_Z28l11_outproj_lnmd_bwd_do8_mx6"
ONLY14, PLUS14, DBG14 = do8._ONLY_MANGLE, do8._PROD_MANGLE, do8._DBG_MANGLE
WLB_X, WLB_I = "_Z16l11_do8_wl_buildIxEvPKT_iixPiS3_", "_Z16l11_do8_wl_buildIiEvPKT_iixPiS3_"
_KEYS = ("Function Name", "VGPRs", "VGPRs Spill", "SGPRs Spill", "ScratchSize [bytes/lane]", "LDS Size [bytes/block]",
         "TotalSGPRs")
_WLB = [(WLB_X, 69, 0, 0, 0, 132, 52), (WLB_I, 69, 0, 0, 0, 132, 52)]
# The kitLB build-only lab job's production remarks (ROCm 7.2.1, clang roc-7.2.1; the landing l11_do8.cu 04942aa1,
# the loader's flags): (name, VGPRs, VGPRs Spill, SGPRs Spill, scratch, LDS, TotalSGPRs) per kernel, in log order.
KITLB: Dict[str, List[Tuple[str, int, int, int, int, int, int]]] = {
    "m_only": _WLB + [(FP8 + ONLY14 + _SIG, 256, 0, 33, 0, 163840, 106), (MX6 + ONLY14 + _SIG, 256, 0, 34, 0, 163840, 106),
                      (MX6 + DBG14 + _SIG, 190, 0, 0, 0, 98304, 58), (FP8 + DBG14 + _SIG, 198, 0, 0, 0, 98304, 56)],
    "m_nib": _WLB + [(FP8 + ONLY14 + _SIG, 256, 0, 34, 0, 163840, 106), (MX6 + ONLY14 + _SIG, 255, 0, 35, 0, 163840, 106),
                     (MX6 + DBG14 + _SIG, 190, 0, 0, 0, 98304, 58), (FP8 + DBG14 + _SIG, 198, 0, 0, 0, 98304, 56)],
    "m_d0": _WLB + [(FP8 + ONLY14 + _SIG, 253, 0, 56, 0, 163840, 106), (MX6 + ONLY14 + _SIG, 251, 0, 57, 0, 163840, 106),
                    (MX6 + DBG14 + _SIG, 190, 0, 0, 0, 98304, 58), (FP8 + DBG14 + _SIG, 198, 0, 0, 0, 98304, 55)],
    "m_plus": _WLB + [(FP8 + PLUS14 + _SIG, 256, 0, 38, 0, 163840, 106), (MX6 + PLUS14 + _SIG, 256, 0, 38, 0, 163840, 106),
                      (MX6 + DBG14 + _SIG, 190, 0, 0, 0, 98304, 58), (FP8 + DBG14 + _SIG, 198, 0, 0, 0, 98304, 56)],
    "l_only": _WLB + [(FP8 + ONLY14 + _SIG, 256, 0, 33, 0, 163840, 106)],
    "l_plus": _WLB + [(FP8 + PLUS14 + _SIG, 256, 0, 38, 0, 163840, 106)],
    "l_nib": _WLB + [(FP8 + ONLY14 + _SIG, 256, 0, 34, 0, 163840, 106)],
    "l_d0": _WLB + [(FP8 + ONLY14 + _SIG, 253, 0, 56, 0, 163840, 106)],
}
# The provenance test re-parses the lab job's logs from the directory this variable names (its build/ directory: one
# subdirectory per build with hipcc.log, and build_summary.json); skipped when unset.
_KITLB_ENV = "L11DO8_MX6_KITLB_DIR"
_KITLB_RAW = os.environ.get(_KITLB_ENV, "").strip()
_KITLB_DIR = Path(_KITLB_RAW) if _KITLB_RAW else None
_SRC_SHA = "04942aa1b6b0e3abd3c2705d2f74e38e64fcc57d517c158f9a2069258e511483"
_IMPL_ONLY = ("l11_outproj_lnmd_bwd_do8 abi=1 variant=e5only(E5 1,WLIST 2 device list,EM 2 only no bf16 DX store,RET 0 "
              "LDS staging,EMV 2 bf16 convert,packed running max)+vd(C1,C2,C3,C4,LANEFIX,PIN)+l11_do8_wl_build")
IMPL_M_ONLY = f"{_IMPL_ONLY} src_sha256={_SRC_SHA}+mx6(e2m3xe2m3,row400,opsel_k)"  # kitLB m_only, ABI 2
IMPL_M_NIB = f"{_IMPL_ONLY}+mkb=nib src_sha256={_SRC_SHA}+mx6(e2m3xe2m3,row400,opsel_k)"  # kitLB m_nib, ABI 2
IMPL_L_ONLY = f"{_IMPL_ONLY} src_sha256={_SRC_SHA}"  # kitLB l_only (the FP8 'only' .so), ABI 1
# Re-cut v2: MX6's ACTIVE line (prebuild_mx6's success, right after the prebuild line, before RUN_START) up to its
# "impl=" field, and the per-rank marker regex an in-model rung greps for (the line + arm_print_all_ranks' " [rank N]").
_ACTIVE_HEAD = ("armed OK: L11 DO8 dO8/E8 emission MX6 ACTIVE mode=e5 only (DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6, "
                "item (104), with items (DO8), (DO8-ONLY), (39), (38)(a) and r14's G6/W6 producer): printed before "
                "run_start (prebuild, once per process); each MX6-site call runs one l11_outproj_lnmd_bwd_do8_mx6 EM=2 "
                "launch (E2M3 x E2M3, E8M0 per 32) that replaces the FP8 'only' launch, counted in the mx6 counters "
                "line (printed again at exit); impl=")
_ACTIVE_RANK_RX = (r"armed OK: L11 DO8 dO8/E8 emission MX6 ACTIVE mode=e5 only \(DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6, "
                   r"item \(104\), .* sites=top poison=0(\s|$)")


def remarks(build: str) -> List[Dict[str, str]]:
    """The fixture as l11_outproj_lnmd.parse_resource_remarks returns it (string values), via a remark-format log."""
    lines = []
    for row in KITLB[build]:
        for key, value in zip(_KEYS, row):
            lines.append(f"remark: l11_do8.cu:714:0: {'' if key == 'Function Name' else '    '}{key}: {value} "
                         "[-Rpass-analysis=kernel-resource-usage]")
    return l11.parse_resource_remarks("\n".join(lines))


def remark_log(build: str) -> str:
    out = []
    for row in KITLB[build]:
        for key, value in zip(_KEYS, row):
            out.append(f"remark: snap/landing/l11_do8.cu:714:0:     {key}: {value} [-Rpass-analysis=kernel-resource-usage]")
    return "\n".join(out) + "\n"


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


def _base_flags(src: bytes, only: bool, diet: bool, nib: bool) -> List[str]:
    """The base tree's _flags, written out (before the MX6 row): the 'plus' / 'only' lists must stay exactly these."""
    sha = hashlib.sha256(src).hexdigest()
    return ([f"--offload-arch={do8._ARCH}", *do8._BASE_FLAGS, *do8._SCHED_FLAGS["default"], f'-DL11DO8_SRC_SHA256="{sha}"']
            + ([] if diet else ["-DL11DO8_DIET=0"]) + (["-DL11DO8_ONLY=1"] if only else [])
            + (["-DL11_MKB_NIB=1"] if only and nib else []))


def _base_key(src: bytes, flags: List[str], toolchain: str) -> str:
    digest = hashlib.sha256()
    digest.update(src)
    digest.update(b"\0" + "\0".join(flags).encode())
    digest.update(b"\0" + do8._ARCH.encode())
    digest.update(b"\0" + toolchain.encode())
    return digest.hexdigest()[:16]


class _Reset(unittest.TestCase):
    _ENV = ("DLRM_HSTU_FP8_L11_DO8_ONLY", "DLRM_HSTU_FP8_L11_VD", "DLRM_HSTU_FP8_L11_DO8_SRC",
            "DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED", "DLRM_HSTU_FP8_L11_DO8_CACHE", "DLRM_HSTU_FP8_L11_DO8_SCHED",
            "DLRM_HSTU_FP8_LNMD_MASK_NIB", do8.MX6_KNOB, do8.MX6_SITES_KNOB)

    def setUp(self) -> None:
        self._saved_env = {k: os.environ.pop(k, None) for k in self._ENV}  # once per test (restored by tearDown)
        self._fresh()

    def _fresh(self) -> None:
        """Per case: the knob variables unset, every DO8 / DO8-ONLY / MX6 / L11 state reset (DO8 requested)."""
        for k in self._ENV:
            os.environ.pop(k, None)
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(True)
        do8.set_mx6_for_testing(None)
        do8.set_mx6_sites_for_testing(None)
        do8.set_diet_for_testing(None)
        l11.set_requested_for_testing(None)
        l11.reset_for_testing()

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        do8.set_requested_for_testing(None)
        do8.set_mx6_for_testing(None)
        do8.set_mx6_sites_for_testing(None)
        do8.set_diet_for_testing(None)
        do8.reset_for_testing(unload=True)
        l11.set_requested_for_testing(None)
        l11.reset_for_testing()


# ------------------------------------------------------------------------------------------------ build
class Mx6BuildFlagsKeyTest(_Reset):
    def _env(self, diet: bool, nib: bool) -> contextlib.ExitStack:
        st = contextlib.ExitStack()
        st.enter_context(mock.patch.object(do8, "diet_requested", return_value=diet))
        st.enter_context(mock.patch.object(do8, "_nib_src", return_value=nib))
        st.enter_context(mock.patch.object(do8, "sched", return_value="default"))
        return st

    def test_mx6_define_last_and_plus_only_lists_unchanged(self) -> None:
        src = b"// any source"
        for diet in (True, False):
            for nib in (False, True):
                with self._env(diet, nib):
                    self.assertEqual(do8._flags(src), _base_flags(src, False, diet, nib))
                    self.assertEqual(do8._flags(src, True), _base_flags(src, True, diet, nib))
                    self.assertEqual(do8._flags(src, False, False), _base_flags(src, False, diet, nib))
                    mx6 = do8._flags(src, True, True)
                    self.assertEqual(mx6, _base_flags(src, True, diet, nib) + ["-DL11DO8_MX6=1"])  # the 'only' NIB call
                    self.assertEqual(mx6[-1], do8._MX6_DEFINE)
                    self.assertNotIn(do8._MX6_DEFINE, do8._flags(src) + do8._flags(src, True))
                    self.assertEqual(do8._flags(src, False, True), mx6)  # mx6 implies only (no MX6 'plus' list)
                    self.assertEqual(do8._flags(src, mx6=True), mx6)

    def test_commands_and_cache_keys(self) -> None:
        src_path = do8._SOURCE
        src = src_path.read_bytes()
        out = Path("/tmp/x.so")
        with self._env(True, False), mock.patch.object(l11, "_hipcc", return_value="/opt/hipcc"), \
                mock.patch.object(l11, "_toolchain", return_value="hipcc test toolchain"):
            for only in (False, True):
                flags = _base_flags(src, only, True, False)
                self.assertEqual(do8.build_command(src_path, out, only),
                                 ["/opt/hipcc", str(src_path.resolve()), *flags, "-o", str(out.resolve())])
                self.assertEqual(do8.cache_key(src_path, only), _base_key(src, flags, "hipcc test toolchain"))
            mx6_flags = _base_flags(src, True, True, False) + ["-DL11DO8_MX6=1"]
            self.assertEqual(do8.build_command(src_path, out, True, True),
                             ["/opt/hipcc", str(src_path.resolve()), *mx6_flags, "-o", str(out.resolve())])
            k_mx6 = do8.cache_key(src_path, only=True, mx6=True)
            self.assertEqual(k_mx6, _base_key(src, mx6_flags, "hipcc test toolchain"))
            self.assertNotIn(k_mx6, (do8.cache_key(src_path), do8.cache_key(src_path, only=True)))
            with mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_L11_DO8_CACHE": "/c"}):
                self.assertEqual(do8._cache_dir(src_path, True, True), Path("/c") / k_mx6)
                self.assertEqual(do8._cache_dir(src_path, mx6=True), Path("/c") / k_mx6)
            # mx6 implies only in every entry point, as build(mx6=True) forces it (a lab tool gets the .so build gates)
            self.assertEqual(do8.build_command(src_path, out, mx6=True), do8.build_command(src_path, out, True, True))
            self.assertEqual(do8.cache_key(src_path, mx6=True), k_mx6)

    def test_build_mx6_gate_takes_the_nib_rule_from_the_build_flags(self) -> None:
        log = {"text": ""}

        def run(cmd, **kw):  # pyre-ignore[2,3]
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x7fELF fake")
            return SimpleNamespace(returncode=0, stderr=log["text"], stdout="")

        spill37 = remark_log("m_nib").replace("SGPRs Spill: 35 ", "SGPRs Spill: 37 ")
        self.assertEqual(spill37.count("SGPRs Spill: 37 "), 1)  # the MX6 'only' kernel's (the twin reads 34)
        for nib, text, want in ((True, remark_log("m_nib"), None), (True, spill37, "need <= 36: the NIB per-variant"),
                                (False, spill37, None)):
            log["text"] = text
            with tempfile.TemporaryDirectory() as tmp, self._env(True, nib), \
                    mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_L11_DO8_CACHE": tmp}), \
                    mock.patch.object(l11, "_hipcc", return_value="/opt/hipcc"), \
                    mock.patch.object(l11, "_toolchain", return_value="t"), \
                    mock.patch.object(do8.subprocess, "run", side_effect=run):
                if want is not None:
                    with self.assertRaisesRegex(RuntimeError, want, msg=(nib, want)):
                        do8.build(mx6=True)
                    continue
                so, summary = do8.build(mx6=True)
                self.assertEqual(summary["nib"], int(nib), nib)  # the gate ran with the build's NIB decision
                self.assertEqual(summary["sgpr_spill"], 35 if text is not spill37 else 37, nib)
                self.assertEqual(do8._NIB_DEFINE in do8._flags(b"", True, True), nib)

    def test_build_mx6_uses_its_own_name_entry_and_gate(self) -> None:
        cmds: List[List[str]] = []
        log = {"text": remark_log("m_only")}

        def run(cmd, **kw):  # pyre-ignore[2,3]
            cmds.append(list(cmd))
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x7fELF fake")
            return SimpleNamespace(returncode=0, stderr=log["text"], stdout="")

        with tempfile.TemporaryDirectory() as tmp, self._env(True, False), \
                mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_L11_DO8_CACHE": tmp}), \
                mock.patch.object(l11, "_hipcc", return_value="/opt/hipcc"), \
                mock.patch.object(l11, "_toolchain", return_value="t"), \
                mock.patch.object(do8.subprocess, "run", side_effect=run):
            so, summary = do8.build(mx6=True)
            self.assertEqual(so.name, "libl11_do8_only_mx6.so")
            self.assertEqual(so.parent.name, do8.cache_key(do8._source(), only=True, mx6=True))
            self.assertEqual(cmds[0][-3:], ["-DL11DO8_MX6=1", "-o", cmds[0][-1]])
            self.assertIn("-DL11DO8_ONLY=1", cmds[0])
            self.assertEqual((summary["sgpr_spill"], summary["fp8_twin_sgpr_spill"], summary["lds"], summary["nib"]),
                             (34, 33, 163840, 0))
            so2, summary2 = do8.build(mx6=True)  # cache hit: no compile, re-gated
            self.assertEqual((so2, len(cmds)), (so, 1))
            self.assertEqual(summary2["sgpr_spill"], 34)
            # the FP8 'only' gate never accepts this log; the MX6 gate never accepts an FP8 'only' log
            with self.assertRaisesRegex(RuntimeError, "need exactly 1"):
                do8.build(only=True)
            log["text"] = remark_log("l_only")
            with tempfile.TemporaryDirectory() as tmp2, mock.patch.dict(os.environ, {"DLRM_HSTU_FP8_L11_DO8_CACHE": tmp2}):
                with self.assertRaisesRegex(RuntimeError, "MX6 resource check: missing"):
                    do8.build(mx6=True)
                self.assertFalse(any(Path(tmp2).rglob("*.so")))  # nothing cached on a failed gate


# ------------------------------------------------------------------------------------------------ gate
class Mx6GateTest(unittest.TestCase):
    def test_production_mx6_builds_are_accepted(self) -> None:
        s = do8.check_resources_mx6(remarks("m_only"))
        self.assertEqual((s["vgpr"], s["vgpr_spill"], s["sgpr_spill"], s["scratch"], s["lds"], s["fp8_twin_vgpr"],
                          s["fp8_twin_sgpr_spill"], s["dbg_lds"], s["kernels"], s["nib"]),
                         (256, 0, 34, 0, 163840, 256, 33, 98304, 6, 0))
        s = do8.check_resources_mx6(remarks("m_nib"), nib=True)
        self.assertEqual((s["vgpr"], s["sgpr_spill"], s["fp8_twin_sgpr_spill"], s["nib"]), (255, 35, 34, 1))
        s = do8.check_resources_mx6(remarks("m_d0"))  # DIET=0 compiles within 63 (the loader keeps it INERT anyway)
        self.assertEqual(s["sgpr_spill"], 57)

    def test_nib_rule(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"SGPR spill 57 \(need <= 36: the NIB per-variant rule\)"):
            do8.check_resources_mx6(remarks("m_d0"), nib=True)
        ks = remarks("m_nib")
        next(k for k in ks if k["Function Name"].startswith(MX6 + ONLY14))["SGPRs Spill"] = "37"
        with self.assertRaisesRegex(RuntimeError, "need <= 36"):
            do8.check_resources_mx6(ks, nib=True)
        self.assertEqual(do8.check_resources_mx6(ks)["sgpr_spill"], 37)  # the int8 bound is 63

    def test_mutations_are_rejected(self) -> None:
        def mx6_only(ks):  # pyre-ignore[2,3]
            return next(k for k in ks if k["Function Name"].startswith(MX6 + ONLY14))

        def mx6_dbg(ks):  # pyre-ignore[2,3]
            return next(k for k in ks if k["Function Name"].startswith(MX6 + DBG14))

        def fp8_only(ks):  # pyre-ignore[2,3]
            return next(k for k in ks if k["Function Name"].startswith(FP8 + ONLY14))

        cases = {
            "MX6 'only' LDS": (lambda ks: mx6_only(ks).__setitem__("LDS Size [bytes/block]", "98304"), "mx6_only LDS"),
            "debug LDS": (lambda ks: mx6_dbg(ks).__setitem__("LDS Size [bytes/block]", "163840"), "mx6_dbg LDS"),
            "SGPR spill 64": (lambda ks: mx6_only(ks).__setitem__("SGPRs Spill", "64"), "need <= 63"),
            "MX6 scratch": (lambda ks: mx6_only(ks).__setitem__("ScratchSize [bytes/lane]", "4"), "scratch 4"),
            "MX6 VGPR spill": (lambda ks: mx6_only(ks).__setitem__("VGPRs Spill", "1"), "VGPR spill 1"),
            "FP8 twin VGPR spill": (lambda ks: fp8_only(ks).__setitem__("VGPRs Spill", "2"), "VGPR spill 2"),
            "list builder scratch": (lambda ks: ks[0].__setitem__("ScratchSize [bytes/lane]", "16"), "scratch 16"),
            "debug kernel missing": (lambda ks: ks.remove(mx6_dbg(ks)), r"missing \['mx6_dbg'\]"),
            "duplicate role": (lambda ks: ks.append(dict(mx6_only(ks))), "instantiated twice"),
            "foreign instantiation": (lambda ks: ks.append({**mx6_only(ks), "Function Name": MX6 + PLUS14 + _SIG}),
                                      "unexpected instantiation"),
            "no list builder": (lambda ks: [ks.remove(k) for k in [k for k in ks if "l11_do8_wl_build" in
                                                                    k["Function Name"]]], "no l11_do8_wl_build"),
        }
        for name, (mutate, want) in cases.items():
            ks = remarks("m_only")
            mutate(ks)
            with self.assertRaisesRegex(RuntimeError, want, msg=name):
                do8.check_resources_mx6(ks)
        with self.assertRaisesRegex(RuntimeError, "unexpected instantiation"):
            do8.check_resources_mx6(remarks("m_plus"))  # the MX6 'plus' build is not this .so
        for b, want in (("l_only", "missing"), ("l_nib", "missing"), ("l_plus", "unexpected instantiation")):
            with self.assertRaisesRegex(RuntimeError, want, msg=b):
                do8.check_resources_mx6(remarks(b))

    def test_fp8_gate_rejects_the_mx6_so_and_keeps_accepting_the_landing_fp8_builds(self) -> None:
        for b in ("m_only", "m_nib"):
            with self.assertRaisesRegex(RuntimeError, "4 l11_outproj_lnmd_bwd_do8 resource remarks .*need exactly 1"):
                do8.check_resources(remarks(b), do8._ONLY_MANGLE)
        for b, mangle, sgpr in (("l_only", do8._ONLY_MANGLE, 33), ("l_nib", do8._ONLY_MANGLE, 34),
                                ("l_d0", do8._ONLY_MANGLE, 56), ("l_plus", do8._PROD_MANGLE, 38)):
            s = do8.check_resources(remarks(b), mangle)
            self.assertEqual((s["lds"], s["vgpr_spill"], s["sgpr_spill"]), (163840, 0, sgpr), b)

    @unittest.skipUnless(_KITLB_DIR is not None and _KITLB_DIR.is_dir(),
                         f"{_KITLB_ENV} does not name the kitLB build-only lab results")
    def test_fixtures_are_the_lab_logs(self) -> None:
        assert _KITLB_DIR is not None
        for build in KITLB:
            got = l11.parse_resource_remarks((_KITLB_DIR / build / "hipcc.log").read_text())
            self.assertEqual([tuple(int(k[x]) if x != "Function Name" else k[x] for x in _KEYS) for k in got],
                             KITLB[build], build)
        summary = json.loads((_KITLB_DIR / "build_summary.json").read_text())["builds"]
        for build, impl, abi in (("m_only", IMPL_M_ONLY, 2), ("m_nib", IMPL_M_NIB, 2), ("l_only", IMPL_L_ONLY, 1)):
            self.assertEqual((summary[build]["interface"]["impl"], summary[build]["interface"]["abi"]), (impl, abi))
            self.assertEqual(summary[build]["lib"], "libl11_do8_only_mx6.so" if build.startswith("m_")
                             else "libl11_do8_only.so")
        self.assertEqual(summary["m_only"]["flags"][-2:], ["-DL11DO8_ONLY=1", "-DL11DO8_MX6=1"])  # MX6 define last


# ------------------------------------------------------------------------------------------------ load / ABI
class _FakeFn:
    def __init__(self, ret: Any = 0) -> None:
        self.ret = ret
        self.restype: Any = None
        self.argtypes: Any = None
        self.calls: List[Tuple[Any, ...]] = []

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.ret


class _FakeLib:
    def __init__(self, abi: int, impl: str) -> None:
        self.l11do8_abi_version = _FakeFn(abi)
        self.l11do8_impl = _FakeFn(impl.encode())
        self.l11do8_e5_only_mx6_launch = _FakeFn(0)
        self.l11do8_e5_only_launch = _FakeFn(0)


def _c_signature(src: str, entry: str) -> List[str]:
    """The ctypes kinds of l11_do8.cu's extern "C" `entry` parameter list ('p' pointer, 'i' int, 'll' long long,
    'f' float)."""
    i = src.index(f'extern "C" int {entry}(')
    params = src[src.index("(", i) + 1:src.index(")", i)]
    out = []
    for p in params.split(","):
        p = " ".join(p.split())
        out.append("p" if "*" in p else "ll" if p.startswith("long long") else "i" if p.startswith("int ")
                   else "f" if p.startswith("float ") else "?" + p)
    return out


_KIND = {ctypes.c_void_p: "p", ctypes.c_int: "i", ctypes.c_longlong: "ll", ctypes.c_float: "f"}


class Mx6LoadTest(_Reset):
    def test_abi_and_impl_rules(self) -> None:
        do8._check_abi_impl_mx6(2, IMPL_M_ONLY, False)
        do8._check_abi_impl_mx6(2, IMPL_M_NIB, True)
        for abi, impl_, nib, want in ((1, IMPL_M_ONLY, False, "abi_version"), (2, IMPL_L_ONLY, False, r"EM 2 \+ MX6"),
                                      (2, IMPL_M_ONLY.replace("EM 2 only", "EM 1 plus"), False, r"EM 2 \+ MX6"),
                                      (2, IMPL_M_ONLY, True, "keep mask format int8"),
                                      (2, IMPL_M_NIB, False, "keep mask format nib")):
            with self.assertRaisesRegex(RuntimeError, want):
                do8._check_abi_impl_mx6(abi, impl_, nib)

    def test_argtypes_are_the_cu_signature(self) -> None:
        src = do8._SOURCE.read_text()
        mx6 = _c_signature(src, "L11DO8_E5_MX6_LAUNCH")
        fp8 = _c_signature(src, "L11DO8_E5_LAUNCH")
        self.assertEqual(len(mx6), 30)
        self.assertEqual([_KIND[t] for t in do8._MX6_LAUNCH_ARGTYPES], mx6)
        self.assertEqual(mx6[:2] + mx6[3:], fp8)  # the FP8 entry's arguments with W6S inserted right after W6
        self.assertEqual(src.count("l11do8_abi_version(void) { return L11DO8_ABI_VERSION; }"), 1)
        self.assertIn('#define L11DO8_MX6_TAG "+mx6(e2m3xe2m3,row400,opsel_k)"', src)
        self.assertIn("#define L11DO8_ABI_VERSION 2", src)

    def _load(self, lib: _FakeLib, only_fmt: str = "int8", twin: Tuple[int, int] = (256, 33),
              only: Any = (256, 33)) -> Tuple[Any, List[Tuple[Any, ...]], List[Dict[str, Any]]]:
        """_load_mx6 with the build and the 'only' load stubbed: the MX6 summary's FP8 twin (VGPRs, SGPR spill) =
        `twin`, the loaded 'only' .so's kernel = `only` (kitLB m_only / l_only by default; None: no resources)."""
        opened: List[Tuple[Any, ...]] = []
        built: List[Dict[str, Any]] = []
        do8._fn_launch = _FakeFn(0)
        do8._fn_launch.argtypes = ["plus-entry argtypes"] * 29
        do8._only_resources.clear()
        if only is not None:
            do8._only_resources.update({"vgpr": only[0], "sgpr_spill": only[1], "lds": 163840})

        def cdll(*a: Any, **k: Any) -> Any:
            opened.append(a + tuple(sorted(k.items())))
            return lib

        def build(**kw: Any) -> Tuple[Path, Dict[str, Any]]:
            built.append(kw)
            return Path("/c/key/libl11_do8_only_mx6.so"), {"kernels_raw": [], "vgpr": 256, "sgpr_spill": 34,
                                                           "fp8_twin_vgpr": twin[0], "fp8_twin_sgpr_spill": twin[1]}

        with mock.patch.object(do8, "_load_only"), mock.patch.object(do8, "build", side_effect=build), \
                mock.patch.object(do8, "only_mask_format", return_value=only_fmt), \
                mock.patch.object(ctypes, "CDLL", side_effect=cdll):
            try:
                do8._load_mx6()
                err = None
            except RuntimeError as e:
                err = e
        return err, opened, built

    def test_load_binds_both_entries_rtld_local(self) -> None:
        lib = _FakeLib(2, IMPL_M_ONLY)
        err, opened, built = self._load(lib)
        self.assertIsNone(err)
        self.assertEqual(built, [{"only": True, "mx6": True}])
        self.assertEqual(opened, [("/c/key/libl11_do8_only_mx6.so",)])  # default mode: RTLD_LOCAL
        self.assertIs(do8._fn_launch_mx6, lib.l11do8_e5_only_mx6_launch)
        self.assertEqual(lib.l11do8_e5_only_mx6_launch.argtypes, list(do8._MX6_LAUNCH_ARGTYPES))
        self.assertEqual(lib.l11do8_e5_only_mx6_launch.restype, ctypes.c_int)
        self.assertIs(do8._fn_launch_mx6_fp8, lib.l11do8_e5_only_launch)
        self.assertEqual(lib.l11do8_e5_only_launch.argtypes, ["plus-entry argtypes"] * 29)
        self.assertEqual((do8.mx6_impl(), do8.mx6_mask_format(), do8.mx6_resources()["sgpr_spill"]),
                         (IMPL_M_ONLY, "int8", 34))
        self.assertEqual(do8.mx6_library_path(), Path("/c/key/libl11_do8_only_mx6.so"))

    def test_load_rejects_and_keeps_nothing(self) -> None:
        for lib, fmt, want in ((_FakeLib(1, IMPL_M_ONLY), "int8", "abi_version"),
                               (_FakeLib(2, IMPL_L_ONLY), "int8", "EM 2 \\+ MX6"),
                               (_FakeLib(2, IMPL_M_ONLY), "nib", "keep mask format")):
            do8.reset_for_testing(unload=True)
            err, _, _ = self._load(lib, fmt)
            self.assertRegex(str(err), want)
            self.assertIsNone(do8.mx6_library_path())
            self.assertEqual(do8.mx6_mask_format(), "int8")
        do8.reset_for_testing(unload=True)
        err, _, _ = self._load(_FakeLib(2, IMPL_M_NIB), "nib")
        self.assertIsNone(err)
        self.assertEqual(do8.mx6_mask_format(), "nib")

    def test_the_fp8_twin_must_be_the_loaded_only_kernel(self) -> None:
        # kitLB: each MX6 build's twin reports its FP8 'only' build's resources (m_only / l_only 256, 33; m_nib / l_nib
        # 256, 34; m_d0 / l_d0 253, 56); a twin that differs no longer vouches for the production 'only' kernel
        for twin, only in (((256, 33), (256, 33)), ((256, 34), (256, 34)), ((253, 56), (253, 56))):
            do8.reset_for_testing(unload=True)
            err, opened, _ = self._load(_FakeLib(2, IMPL_M_ONLY), twin=twin, only=only)
            self.assertIsNone(err, (twin, only))
        for twin, only in (((256, 34), (256, 33)), ((255, 33), (256, 33)), ((256, 200), (256, 33))):
            do8.reset_for_testing(unload=True)
            err, opened, _ = self._load(_FakeLib(2, IMPL_M_ONLY), twin=twin, only=only)
            self.assertRegex(str(err), r"FP8 'only' twin \(VGPRs, SGPR spill\) .* != the loaded 'only' .so's", twin)
            self.assertEqual(opened, [])  # rejected before the .so is opened
            self.assertIsNone(do8.mx6_library_path())
        do8.reset_for_testing(unload=True)  # no 'only' resources (nothing to compare with): never accepted
        err, opened, _ = self._load(_FakeLib(2, IMPL_M_ONLY), only=None)
        self.assertRegex(str(err), r"!= the loaded 'only' .so's \(None, None\)")
        self.assertEqual(opened, [])


# ------------------------------------------------------------------------------------------------ knob / INERT / prebuild
_PRODUCER = ("output_projection_dgrad_operands_mx6", "output_projection_g8_rescue", "output_projection_mx6_encode")


def _bwd_fp8():  # pyre-ignore[3]
    from generative_recommenders.ops.triton import bwd_fp8

    return bwd_fp8


class Mx6PrebuildTest(_Reset):
    """prebuild_mx6 with DO8 and DO8-ONLY prebuilt (the 'only' .so int8), DO8-VD, DIET on and the MX6 knob on, unless
    a case changes one."""

    def _fresh(self) -> None:
        super()._fresh()
        os.environ.update({"DLRM_HSTU_FP8_L11_DO8_ONLY": "1", "DLRM_HSTU_FP8_L11_VD": "1"})
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        do8.set_diet_for_testing(True)
        do8.set_mx6_for_testing(True)

    def producer(self, present: Tuple[str, ...] = _PRODUCER) -> contextlib.ExitStack:
        st = contextlib.ExitStack()
        for name in present:
            st.enter_context(mock.patch.object(_bwd_fp8(), name, create=True, new=lambda *a, **k: None))
        return st

    def prebuild(self, device: int = 0, **patches: Any) -> Tuple[bool, str, mock.MagicMock]:
        load = mock.MagicMock()
        with contextlib.ExitStack() as st, _Capture() as cap:
            st.enter_context(mock.patch.object(do8, "_load_mx6", load))
            st.enter_context(mock.patch.object(torch.cuda, "synchronize"))
            st.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=device))  # ACTIVE's device
            for attr, kw in patches.items():
                st.enter_context(mock.patch.object(do8, attr, **kw))
            ok = do8.prebuild_mx6(self_check_fn=lambda: (True, "lattice MX6 vs FP8 twin dz8 mism 0"))
        return ok, cap.text, load

    def test_knob_off_reads_builds_prints_nothing(self) -> None:
        for value in (None, "0", "", "true", "2"):
            do8.set_mx6_for_testing(None)
            os.environ.pop(do8.MX6_KNOB, None)
            if value is not None:
                os.environ[do8.MX6_KNOB] = value
            with self.producer(), mock.patch.object(do8, "_mx6_inert_reason", side_effect=AssertionError("read")):
                ok, text, load = self.prebuild()
            self.assertFalse(ok, value)
            self.assertEqual(text, "", value)
            load.assert_not_called()
            self.assertFalse(do8.mx6_enabled())
            self.assertFalse(do8._mx6_sites_read)  # the site knob is not even read

    def test_inert_matrix(self) -> None:
        nib = mock.patch.object(do8, "only_mask_format", return_value="nib")
        gate_off = mock.patch.object(do8, "MX6_NIB_GATED", False)  # the switch off: the rule stays covered

        def nib_gate_off() -> None:
            nib.start()
            gate_off.start()

        cases = [
            ("sites", lambda: do8.set_mx6_sites_for_testing("both"), "'both' is not one of top / bottom / all"),
            ("sites empty", lambda: do8.set_mx6_sites_for_testing(""), "is not one of top / bottom / all"),
            ("DO8 not prebuilt", lambda: do8.set_prebuilt_for_testing(False), "needs item (DO8) prebuilt"),
            ("DO8 latched", lambda: setattr(do8, "_broken", "x"), "needs item (DO8) prebuilt"),
            ("DO8 knob off", lambda: do8.set_requested_for_testing(False), "needs item (DO8) prebuilt"),
            ("only knob off", lambda: os.environ.update({"DLRM_HSTU_FP8_L11_DO8_ONLY": "0"}),
             "needs item (DO8-ONLY) prebuilt"),
            ("only not prebuilt", lambda: do8.set_only_prebuilt_for_testing(False), "needs item (DO8-ONLY)"),
            ("only latched", lambda: setattr(do8, "_only_broken", "x"), "needs item (DO8-ONLY)"),
            ("VD off", lambda: os.environ.update({"DLRM_HSTU_FP8_L11_VD": "0"}), "needs DO8-VD"),
            ("DIET off", lambda: do8.set_diet_for_testing(False), "needs item (89) DIET"),
            ("NIB 'only' .so, MX6_NIB_GATED False", nib_gate_off,
             "the 'only' .so reads the NIB keep mask and MX6 + NIB is not gated (MX6_NIB_GATED False: no NIB MX6 build "
             "is allowed)"),
        ]
        for name, arrange, want in cases:
            self._fresh()
            arrange()
            try:
                with self.producer():
                    ok, text, load = self.prebuild()
            finally:
                if arrange == nib_gate_off:
                    gate_off.stop()
                    nib.stop()
            self.assertFalse(ok, name)
            load.assert_not_called()
            self.assertEqual(text.count(do8.FAMILY_MX6 + " INERT: "), 1, (name, text))
            self.assertIn(want, text, name)
            self.assertIn("nothing built -> FP8 dual cast + (DO8-ONLY) [rank ", text, name)
            self.assertNotIn("ACTIVE", text, name)  # INERT: no ACTIVE line
            self.assertEqual(do8.mx6_inert() is not None, True, name)
            self.assertFalse(do8.mx6_enabled(), name)
            self.assertIsNone(do8.mx6_broken(), name)
            with self.producer():
                ok2, text2, _ = self.prebuild()  # once per process: no second line
            self.assertEqual((ok2, text2), (False, ""), name)

    def test_source_without_the_switch_is_inert(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
            f.write("// a candidate .cu that only names L11DO8_MX6 in a comment\n")
        try:
            os.environ["DLRM_HSTU_FP8_L11_DO8_SRC"] = f.name
            with self.producer():
                ok, text, load = self.prebuild()
        finally:
            os.unlink(f.name)
        self.assertFalse(ok)
        load.assert_not_called()
        self.assertIn("has no L11DO8_MX6 switch", text)
        self.assertNotIn("ACTIVE", text)

    def absent(self, names: Tuple[str, ...] = _PRODUCER) -> contextlib.ExitStack:
        """r14's API absent (today's tree; simulated once it lands: the names set to None)."""
        st = contextlib.ExitStack()
        for name in names:
            if hasattr(_bwd_fp8(), name):
                st.enter_context(mock.patch.object(_bwd_fp8(), name, None))
        return st

    def test_producer_api_absent_is_inert(self) -> None:
        with self.absent():
            ok, text, load = self.prebuild()
        self.assertFalse(ok)
        load.assert_not_called()
        self.assertIn("r14's MX6 producer API is not in this tree (ops/triton/bwd_fp8.py has no "
                      "output_projection_dgrad_operands_mx6, output_projection_g8_rescue, output_projection_mx6_encode)",
                      text)
        self.assertNotIn("ACTIVE", text)
        do8._reset_mx6_for_testing(False)
        do8._once.clear()
        with self.absent(_PRODUCER[2:]), self.producer(_PRODUCER[:2]):
            _, text, load = self.prebuild()
        self.assertIn("has no output_projection_mx6_encode)", text)
        load.assert_not_called()

    def test_mx6_nib_gated_is_pinned_true(self) -> None:
        # Provenance: kitN (GPU job 72655, 2026-10-13 20:01-20:11Z; production ROCm 7.2.1 builds of l11_do8.cu 04942aa1;
        # kit seal SHA256SUMS 68af4496; PREREG fc3e9776 + AMEND1 781177c8 + AMEND2 902ae4e2) returned N_PASS, G-N0..G-N4
        # all PASS: resources and FP8-twin identity; MX6 + NIB on pack_ref(mask) == MX6 int8 byte for byte (25 cases);
        # the lattice tie inside MX6 + NIB; FP8 + NIB == FP8 int8; the pack round trip. Its frozen verdict map: N_PASS
        # -> MX6_NIB_GATED True in r16's loader. False again is a reviewed row (test_inert_matrix covers that rule).
        self.assertIs(do8.MX6_NIB_GATED, True)

    def test_nib_gated_builds(self) -> None:
        """The default switch (no patch): a NIB 'only' .so is not an INERT rule (_mx6_inert_reason), so the prebuild
        calls _load_mx6 and the armed line names the loaded MX6 .so's format. Covered here: that rule and the armed
        line only; _load_mx6 is a stub leaving kitLB m_nib's state and the self-check a stub detail. The real NIB load
        chain (want_nib, the NIB agreement check, the NIB SGPR bound) is Mx6LoadTest's / Mx6BuildFlagsKeyTest's, the
        real self-check's NIB branch Mx6SelfCheckCpuTest's, the hardware run Mx6GpuTest's NIB case."""

        def load() -> None:  # what _load_mx6 leaves for a NIB 'only' .so (Mx6LoadTest: IMPL_M_NIB, 'only' "nib")
            do8._mx6_lib = object()
            do8._impl_mx6 = IMPL_M_NIB
            do8._mx6_lib_path = Path("/c/0123456789abcdef/libl11_do8_only_mx6.so")
            do8._mx6_resources.update({"vgpr": 255, "vgpr_spill": 0, "scratch": 0, "lds": 163840, "sgpr_spill": 35,
                                       "fp8_twin_vgpr": 256, "dbg_lds": 98304, "nib": 1})

        loader = mock.MagicMock(side_effect=load)
        detail = "lattice MX6 vs FP8 twin dz8 mism 0; mask nib(uint8[M,256]) unpack(pack_ref) exact 1"
        with contextlib.ExitStack() as st, self.producer(), _Capture() as cap:
            st.enter_context(mock.patch.object(do8, "_load_mx6", loader))
            st.enter_context(mock.patch.object(do8, "only_mask_format", return_value="nib"))
            st.enter_context(mock.patch.object(torch.cuda, "synchronize"))
            st.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=0))
            ok = do8.prebuild_mx6(self_check_fn=lambda: (True, detail))
        text = cap.text
        self.assertTrue(ok, text)
        loader.assert_called_once()
        self.assertNotIn("INERT", text)
        self.assertIsNone(do8.mx6_inert())
        self.assertTrue(do8.mx6_enabled())
        self.assertEqual(do8.mx6_mask_format(), "nib")
        lines = text.splitlines()
        self.assertEqual(len(lines), 3, text)
        self.assertRegex(lines[0], r"^armed OK: L11 DO8 dO8/E8 emission MX6 prebuild before run_start: "
                                   r"built\+loaded\+self-check OK \d+\.\ds \(MX6 'only': VGPR 255, spill 0, scratch 0, "
                                   r"LDS 163840, SGPR spill 35; FP8 twin VGPR 256; debug LDS 98304; ABI 2; so "
                                   r"0123456789abcdef; lattice MX6 vs FP8 twin dz8 mism 0; mask nib\(uint8\[M,256\]\) "
                                   r"unpack\(pack_ref\) exact 1\) mask=nib\(uint8\[M,256\]\) sites=top \[rank \w+\]$")
        # re-cut v2: the ACTIVE line follows the prebuild line and names the loaded (NIB) MX6 impl
        self.assertEqual(re.sub(r" \[rank \w+\]$", "", lines[1]),
                         f"{_ACTIVE_HEAD}{IMPL_M_NIB} device=cuda:0 sites=top poison=0")
        self.assertRegex(lines[1], _ACTIVE_RANK_RX)
        self.assertRegex(lines[2], r"^\[l11-do8\] mx6 counters: decided=0 launched=0 published=0 rescued=0 "
                                   r"released=0 latched=0 \[rank \w+\]$")
        self.assertTrue(do8.enabled() and do8.only_enabled())  # untouched

    def test_nib_only_so_without_the_producer_is_inert_for_the_producer(self) -> None:
        # this tree (r14's API absent) with a NIB 'only' .so and the knob on: still INERT, for the producer only
        with self.absent():
            ok, text, load = self.prebuild(only_mask_format={"return_value": "nib"})
        self.assertFalse(ok)
        load.assert_not_called()
        self.assertEqual(text.count(do8.FAMILY_MX6 + " INERT: "), 1, text)
        self.assertIn("r14's MX6 producer API is not in this tree", text)
        self.assertNotIn("NIB", text)

    def test_switch_off_the_nib_rule_comes_before_the_producer_rule(self) -> None:
        # the documented INERT order (_mx6_inert_reason): with the switch off, a NIB 'only' .so in a tree without r14's
        # producer names the NIB rule, not the producer's
        with self.absent():
            ok, text, load = self.prebuild(only_mask_format={"return_value": "nib"}, MX6_NIB_GATED={"new": False})
        self.assertFalse(ok)
        load.assert_not_called()
        self.assertEqual(text.count(do8.FAMILY_MX6 + " INERT: "), 1, text)
        self.assertIn("the 'only' .so reads the NIB keep mask and MX6 + NIB is not gated (MX6_NIB_GATED False", text)
        self.assertNotIn("producer API", text)

    def erm(self, requested: Any) -> Any:
        """r14's E-RM row requested (True) or not (False) through its test override, reset after the test."""
        from generative_recommenders.ops.triton import dy_erm

        self.addCleanup(dy_erm.reset_for_testing)
        self.addCleanup(dy_erm.set_requested_for_testing, None)
        dy_erm.set_requested_for_testing(requested)
        return dy_erm

    def test_erm_requested_with_a_layer_0_site_set_is_inert(self) -> None:
        """E-RM owns layer 0's out-proj: with E-RM REQUESTED a site set holding layer 0 (bottom, all) is INERT (one
        line, nothing built); the top set beside E-RM, and any set without E-RM, are not (they arm here)."""
        for sites, requested, inert in (("bottom", True, True), ("all", True, True), (" ALL ", True, True),
                                        ("top", True, False), (None, True, False), ("bottom", False, False),
                                        ("all", False, False)):
            self._fresh()
            dy_erm = self.erm(requested)
            do8.set_mx6_sites_for_testing(sites)
            with self.producer():
                ok, text, load = self.prebuild()
            case = (sites, requested)
            if not inert:
                self.assertTrue(ok, (case, text))
                load.assert_called_once()
                self.assertNotIn("INERT", text, case)
                self.assertIn(f" sites={do8.mx6_sites()} [rank ", text, case)
                self.assertEqual(text.count(" ACTIVE mode=e5 only "), 1, case)  # re-cut v2: at the prebuild
                self.assertIn(f" sites={do8.mx6_sites()} poison=0 [rank ", text, case)
                continue
            self.assertFalse(ok, case)
            load.assert_not_called()
            self.assertNotIn("ACTIVE", text, case)
            self.assertEqual(text.count(do8.FAMILY_MX6 + " INERT: "), 1, (case, text))
            self.assertIn(f"{do8.MX6_SITES_KNOB}={sites.strip().lower()} holds layer 0, the out-proj site r14's E-RM "
                          f"row owns ({dy_erm.KNOB}=1 requested): fail closed, MX6 never takes an E-RM site (top only "
                          "beside E-RM); nothing built -> FP8 dual cast + (DO8-ONLY) [rank ", text, case)
            self.assertFalse(do8.mx6_enabled(), case)
            self.assertIsNone(do8.mx6_broken(), case)
            with self.producer():
                self.assertEqual(self.prebuild()[:2], (False, ""), case)  # once per process

    def test_the_erm_rule_reads_the_request_only_and_follows_the_site_rule(self) -> None:
        """The E-RM rule is decided from E-RM's REQUEST, never from its runtime state (latched or INERT E-RM: still
        INERT, and enabled / broken / inert_reason are never asked); it comes after the site-set rule and before every
        other rule (here: DO8 not prebuilt, r14's producer API absent)."""
        dy_erm = self.erm(True)
        with _Capture():
            dy_erm.set_broken("injected: E-RM latched")
        do8.set_mx6_sites_for_testing("all")
        do8.set_prebuilt_for_testing(False)
        asked = {n: mock.patch.object(dy_erm, n, side_effect=AssertionError(f"E-RM {n} read"))
                 for n in ("enabled", "broken", "inert_reason")}
        with contextlib.ExitStack() as st, self.absent():
            for p in asked.values():
                st.enter_context(p)
            ok, text, load = self.prebuild()
        self.assertFalse(ok)
        load.assert_not_called()
        self.assertEqual(text.count(do8.FAMILY_MX6 + " INERT: "), 1, text)
        self.assertIn("holds layer 0, the out-proj site r14's E-RM row owns", text)
        self.assertNotIn("needs item (DO8)", text)
        self.assertNotIn("producer API", text)
        self._fresh()
        self.erm(True)
        do8.set_mx6_sites_for_testing("both")
        with self.producer():
            ok, text, load = self.prebuild()
        self.assertIn("'both' is not one of top / bottom / all", text)  # the site-set rule first
        self.assertNotIn("E-RM", text)

    def test_thl_prebuild_with_do8_off_names_the_erm_rule(self) -> None:
        try:
            from generative_recommenders.ops.triton import triton_hstu_linear as thl
        except Exception as error:  # noqa: BLE001
            self.skipTest(f"triton_hstu_linear not importable: {error}")
        self.erm(True)
        do8.set_requested_for_testing(False)
        do8.set_mx6_sites_for_testing("bottom")
        with _Capture() as cap:
            self.assertFalse(thl.prebuild_l11_do8())
        self.assertEqual(cap.text.count(do8.FAMILY_MX6 + " INERT: " + do8.MX6_SITES_KNOB + "=bottom holds layer 0"), 1,
                         cap.text)

    def test_armed_line_and_counters(self) -> None:
        do8._mx6_resources.update({"vgpr": 256, "vgpr_spill": 0, "scratch": 0, "lds": 163840, "sgpr_spill": 34,
                                   "fp8_twin_vgpr": 256, "dbg_lds": 98304})
        do8._mx6_lib_path = Path("/c/0123456789abcdef/libl11_do8_only_mx6.so")
        do8._impl_mx6 = IMPL_M_ONLY  # what _load_mx6 leaves (Mx6LoadTest); the stub loader keeps it
        with self.producer():
            ok, text, load = self.prebuild()
        self.assertTrue(ok)
        load.assert_called_once()
        self.assertTrue(do8.mx6_enabled())
        lines = text.splitlines()
        self.assertEqual(len(lines), 3, text)  # prebuild line, ACTIVE line, counters line: once each, in this order
        self.assertRegex(lines[0], r"^armed OK: L11 DO8 dO8/E8 emission MX6 prebuild before run_start: "
                                   r"built\+loaded\+self-check OK \d+\.\ds \(MX6 'only': VGPR 256, spill 0, scratch 0, "
                                   r"LDS 163840, SGPR spill 34; FP8 twin VGPR 256; debug LDS 98304; ABI 2; so "
                                   r"0123456789abcdef; lattice MX6 vs FP8 twin dz8 mism 0\) mask=int8 sites=top "
                                   r"\[rank \w+\]$")
        # re-cut v2: MX6's ACTIVE line prints here (before RUN_START), not at the first MX6 record
        self.assertEqual(re.sub(r" \[rank \w+\]$", "", lines[1]),
                         f"{_ACTIVE_HEAD}{IMPL_M_ONLY} device=cuda:0 sites=top poison=0")
        self.assertRegex(lines[1], r" \[rank \w+\]$")
        self.assertIn(" item (104), ", lines[1])
        self.assertIn(" before run_start ", lines[1])
        self.assertRegex(lines[1], r" sites=(top|bottom|all) poison=[01] \[rank \w+\]$")
        self.assertRegex(lines[1], _ACTIVE_RANK_RX)  # the in-model per-rank marker
        self.assertRegex(lines[2], r"^\[l11-do8\] mx6 counters: decided=0 launched=0 published=0 rescued=0 "
                                   r"released=0 latched=0 \[rank \w+\]$")
        self.assertTrue(do8.enabled() and do8.only_enabled())  # untouched

    def test_second_prebuild_prints_nothing_new(self) -> None:
        """Re-cut v2: the prebuild's three lines print once per process; a second prebuild_mx6 (armed: it returns
        mx6_enabled()) builds and prints nothing, also after a latch (then False)."""
        with self.producer():
            ok, text, load = self.prebuild()
        self.assertTrue(ok)
        self.assertEqual(len(text.splitlines()), 3, text)
        with self.producer():
            ok2, text2, load2 = self.prebuild()
        self.assertEqual((ok2, text2), (True, ""))
        load2.assert_not_called()
        self.assertEqual(do8.mx6_counters()["published"], 0)
        with _Capture():
            do8._disable_mx6("injected")
        with self.producer():
            self.assertEqual(self.prebuild()[:2], (False, ""))  # latched: still nothing

    def test_active_line_device_and_poison(self) -> None:
        """Re-cut v2: the ACTIVE line's last fields are the current device, the site set and the registry's poison knob
        (DLRM_HSTU_FP8_L11_DO8_ONLY_POISON, read through registry(); no registry module: poison=0)."""
        reg = do8.registry()
        self.assertIsNotNone(reg)  # this tree has R15's hook (ops/flydsl/hstu_do8_registry.py)
        for poison, sites, registry_kw, want in (
                ("1", None, None, "device=cuda:3 sites=top poison=1"),
                ("0", "all", None, "device=cuda:3 sites=all poison=0"),
                ("1", "bottom", {"return_value": None}, "device=cuda:3 sites=bottom poison=0")):
            self._fresh()
            do8.set_mx6_sites_for_testing(sites)
            patches: Dict[str, Any] = {} if registry_kw is None else {"registry": registry_kw}
            with self.producer(), mock.patch.dict(os.environ, {reg.POISON_ENV: poison}):
                ok, text, _ = self.prebuild(device=3, **patches)
            lines = text.splitlines()
            self.assertTrue(ok, text)
            self.assertEqual(len(lines), 3, text)
            self.assertTrue(lines[1].startswith(_ACTIVE_HEAD), lines[1])
            self.assertTrue(re.sub(r" \[rank \w+\]$", "", lines[1]).endswith(" " + want), (want, lines[1]))
            self.assertEqual(re.search(_ACTIVE_RANK_RX, lines[1]) is not None, want.endswith("sites=top poison=0"))

    def test_failures_latch_mx6_only(self) -> None:
        cases = (
            ({"_load_mx6": {"side_effect": RuntimeError("MX6 resource check: mx6_only SGPR spill 64 (need <= 63)")}},
             None, "DISABLED: prebuild build/load: RuntimeError: MX6 resource check"),
            ({}, (False, "dz8 mism 7"), "DISABLED: prebuild self-check mismatch: dz8 mism 7"),
            ({}, ValueError("encode failed"), "DISABLED: prebuild self-check: ValueError: encode failed"),
        )
        for patches, check, want in cases:
            self._fresh()
            do8.set_only_prebuilt_for_testing(True)

            def fn() -> Tuple[bool, str]:
                if isinstance(check, Exception):
                    raise check
                return check or (True, "ok")

            with contextlib.ExitStack() as st, self.producer(), _Capture() as cap:
                st.enter_context(mock.patch.object(torch.cuda, "synchronize"))
                if "_load_mx6" not in patches:
                    st.enter_context(mock.patch.object(do8, "_load_mx6"))
                for attr, kw in patches.items():
                    st.enter_context(mock.patch.object(do8, attr, **kw))
                self.assertFalse(do8.prebuild_mx6(self_check_fn=fn))
            self.assertEqual(cap.text.count(do8.FAMILY_MX6 + " " + want), 1, cap.text)
            self.assertNotIn("ACTIVE", cap.text)  # DISABLED / self-check failure: no ACTIVE line
            self.assertIn("-> FP8 dual cast + (DO8-ONLY) for this process", cap.text)
            self.assertIn("[l11-do8] mx6 counters: decided=0 launched=0 published=0 rescued=0 released=0 latched=1",
                          cap.text)
            self.assertIsNotNone(do8.mx6_broken())
            self.assertFalse(do8.mx6_enabled())
            self.assertTrue(do8.enabled() and do8.only_enabled())  # DO8 and DO8-ONLY keep running
            self.assertIsNone(do8.broken())
            self.assertIsNone(do8.only_broken())

    def test_device_fault_reraises_and_fail_closed_raises(self) -> None:
        def fault() -> Tuple[bool, str]:
            raise RuntimeError("HIP error: an illegal memory access was encountered")

        with contextlib.ExitStack() as st, self.producer(), _Capture() as cap:
            st.enter_context(mock.patch.object(do8, "_load_mx6"))
            st.enter_context(mock.patch.object(torch.cuda, "synchronize"))
            with self.assertRaises(RuntimeError):
                do8.prebuild_mx6(self_check_fn=fault)
        self.assertIn(do8.FAMILY_MX6 + " prebuild DEVICE FAULT: RuntimeError: HIP error: an illegal memory", cap.text)
        self.assertNotIn("ACTIVE", cap.text)
        self.assertIsNone(do8.mx6_broken())
        self._fresh()
        os.environ["DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED"] = "1"
        with contextlib.ExitStack() as st, self.producer(), _Capture():
            st.enter_context(mock.patch.object(do8, "_load_mx6", side_effect=RuntimeError("hipcc rc=1")))
            with self.assertRaisesRegex(RuntimeError, "L11 DO8 dO8/E8 emission MX6 failed with "
                                                      "DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1: prebuild build/load"):
                do8.prebuild_mx6()
        self.assertIsNone(do8.mx6_broken())

    def test_prebuild_order_do8_only_mx6(self) -> None:
        do8.set_prebuilt_for_testing(False)
        do8.set_only_prebuilt_for_testing(False)
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        with contextlib.ExitStack() as st, self.producer(), _Capture() as cap:
            for target, attr, kw in ((torch.cuda, "is_available", {"return_value": True}),
                                     (torch.version, "hip", {"new": "7.2"}),
                                     (torch.cuda, "current_device", {"return_value": 0}),
                                     (torch.cuda, "synchronize", {}), (l11, "device_arch", {"return_value": "gfx950"}),
                                     (do8, "_load", {}), (do8, "_load_only", {}), (do8, "_load_mx6", {}),
                                     (do8, "self_check_only", {"return_value": (True, "only ok")}),
                                     (do8, "self_check_mx6", {"return_value": (True, "mx6 ok")})):
                st.enter_context(mock.patch.object(target, attr, **kw))
            self.assertTrue(do8.prebuild(True, True, self_check_fn=lambda: (True, "plus ok")))
        text = cap.text
        i_plus = text.index("armed OK: " + do8.FAMILY + " prebuild before run_start")
        i_only = text.index("armed OK: " + do8.FAMILY_ONLY + " prebuild before run_start")
        i_mx6 = text.index("armed OK: " + do8.FAMILY_MX6 + " prebuild before run_start")
        i_act = text.index("armed OK: " + do8.FAMILY_MX6 + " ACTIVE mode=e5 only")  # re-cut v2: in the prebuild
        i_cnt = text.index("[l11-do8] mx6 counters: ")
        self.assertTrue(i_plus < i_only < i_mx6 < i_act < i_cnt)
        self.assertEqual(text.count(" ACTIVE mode=e5 only "), 1)  # MX6's; no DO8 / DO8-ONLY ACTIVE line at prebuild
        self.assertIn("mx6 ok", text)
        self.assertTrue(do8.mx6_enabled())

    def test_knob_off_prebuild_is_unchanged(self) -> None:
        do8.set_mx6_for_testing(False)
        do8.set_prebuilt_for_testing(False)
        do8.set_only_prebuilt_for_testing(False)
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        with contextlib.ExitStack() as st, self.producer(), _Capture() as cap:
            for target, attr, kw in ((torch.cuda, "is_available", {"return_value": True}),
                                     (torch.version, "hip", {"new": "7.2"}),
                                     (torch.cuda, "current_device", {"return_value": 0}),
                                     (torch.cuda, "synchronize", {}), (l11, "device_arch", {"return_value": "gfx950"}),
                                     (do8, "_load", {}), (do8, "_load_only", {}),
                                     (do8, "_load_mx6", {"side_effect": AssertionError("built")}),
                                     (do8, "self_check_only", {"return_value": (True, "only ok")})):
                st.enter_context(mock.patch.object(target, attr, **kw))
            self.assertTrue(do8.prebuild(True, True, self_check_fn=lambda: (True, "plus ok")))
        self.assertNotIn("MX6", cap.text)
        self.assertNotIn("mx6", cap.text)

    def test_thl_prebuild_with_do8_off_says_mx6_is_inert_once(self) -> None:
        try:
            from generative_recommenders.ops.triton import triton_hstu_linear as thl
        except Exception as error:  # noqa: BLE001
            self.skipTest(f"triton_hstu_linear not importable: {error}")
        do8.set_requested_for_testing(False)
        with _Capture() as cap:
            self.assertFalse(thl.prebuild_l11_do8())
        self.assertEqual(cap.text.count(do8.FAMILY_MX6 + " INERT: needs item (DO8) prebuilt and not latched"), 1)
        self.assertNotIn("ACTIVE", cap.text)
        do8.reset_for_testing()
        do8.set_mx6_for_testing(False)
        with _Capture() as cap:
            self.assertFalse(thl.prebuild_l11_do8())
        self.assertEqual(cap.text, "")


# ------------------------------------------------------------------------------------------------ latch / lines
class Mx6LatchLinesTest(_Reset):
    def _fresh(self) -> None:
        super()._fresh()
        os.environ["DLRM_HSTU_FP8_L11_DO8_ONLY"] = "1"
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        do8.set_mx6_prebuilt_for_testing(True)

    def test_latch_isolation_and_line_format(self) -> None:
        self.assertTrue(do8.mx6_enabled())
        with _Capture() as cap:
            do8._disable_mx6("rescue (before the MX6 EM = 2 launch): Mx6LaunchError: rc=1")
            do8._disable_mx6("second reason")  # once
        self.assertFalse(do8.mx6_enabled())
        self.assertEqual(do8.mx6_broken(), "rescue (before the MX6 EM = 2 launch): Mx6LaunchError: rc=1")
        self.assertTrue(do8.enabled() and do8.only_enabled())
        self.assertIsNone(do8.broken())
        self.assertIsNone(do8.only_broken())
        lines = cap.text.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(re.sub(r" \[rank \w+\]$", "", lines[0]),
                         "L11 DO8 dO8/E8 emission MX6 DISABLED: rescue (before the MX6 EM = 2 launch): Mx6LaunchError: "
                         "rc=1 -> FP8 dual cast + (DO8-ONLY) for this process")
        self.assertTrue(lines[1].startswith("[l11-do8] mx6 counters: ") and "latched=1" in lines[1])
        self.assertTrue(all("[rank " in ln for ln in lines))

    def test_fail_closed(self) -> None:
        os.environ["DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED"] = "1"
        with self.assertRaisesRegex(RuntimeError, "MX6 failed with DLRM_HSTU_FP8_L11_DO8_FAIL_CLOSED=1: x"):
            do8._disable_mx6("x")
        self.assertIsNone(do8.mx6_broken())

    def test_counters_and_exit_line(self) -> None:
        with _Capture() as cap:
            do8._mx6_event("decided")
            do8._mx6_event("released")
            do8._mx6_event("released")
        self.assertEqual(cap.text.count("[l11-do8] mx6 counters: decided=1 launched=0 published=0 rescued=0 released=1"
                                        " latched=0"), 1)  # at the first release only
        c = do8.mx6_counters()
        self.assertEqual((c["decided"], c["released"], c["latched"]), (1, 2, 0))
        with _Capture() as cap:
            do8._only_final()
        self.assertEqual(cap.text.count("[l11-do8] mx6 counters: decided=1 launched=0 published=0 rescued=0 "
                                        "released=2 latched=0 final=1"), 1)
        do8.reset_for_testing()
        with _Capture() as cap:
            do8._only_final()  # nothing counted, not prebuilt, not latched: no MX6 line
        self.assertEqual(cap.text, "")

    def test_decline_lines_once_and_quiet(self) -> None:
        with _Capture() as cap:
            do8.mx6_note_decline("C3 the W3 dout_t copy is not e4m3 (x)")
            do8.mx6_note_decline("C3 the W3 dout_t copy is not e4m3 (y)")
            do8.mx6_note_decline("the out-proj dgrad scale is not published yet", quiet=True)
        self.assertEqual(cap.text.count(do8.FAMILY_MX6 + " NOT armed: C3 the W3 dout_t copy is not e4m3 (x) -> FP8 dual "
                                        "cast + (DO8-ONLY) for this call"), 1)
        self.assertNotIn("published yet", cap.text)
        self.assertEqual(do8.mx6_counters()["declined"], {"C3 the": 2, "the out-proj": 1})


# ------------------------------------------------------------------------------------------------ the self-check's lattice
class Mx6LatticeTest(unittest.TestCase):
    def test_lattice_is_exact_in_both_fp8_encodings(self) -> None:
        dout, w = do8.mx6_lattice_operands(2048, do8._MX6_SELF_CHECK_SEED, torch.device("cpu"))
        self.assertEqual((dout.dtype, tuple(dout.shape), w.dtype, tuple(w.shape)), (BF, (2048, D), F32, (1536, D)))
        self.assertEqual(float(dout.float().abs().max()), 7 * 2.0**-14)
        self.assertEqual(float(dout[0, 0]), 7 * 2.0**-14)
        self.assertEqual(float(w.abs().max()), 7 * 2.0**-6)
        sa = do8._MX6_G_SCALE
        self.assertEqual(sa, 7 * 2.0**-14 * 8 / 57344)  # the controller's publish at margin 8 for this amax
        g8 = (dout.float() / sa).clamp(-57344, 57344).to(E5)
        self.assertTrue(torch.equal(g8.float() * sa, dout.float()))
        sb = float(w.abs().max()) / 448.0  # quantize_weight's scale
        self.assertEqual(math.frexp(sb)[0], 0.5)  # a power of two (2^-12)
        w8 = (w / sb).clamp(-448, 448).to(E4)
        self.assertTrue(torch.equal(w8.float() * sb, w))
        # integer parts in [-7, 7] at one power of two per row / per 32-row block, dot products below 2^15
        t = torch.floor(torch.log2(dout.float().abs().amax(dim=1).clamp_min(2.0**-40)))
        n = dout.float() / torch.pow(2.0, t[:, None] - 2)
        self.assertTrue(torch.equal(n, n.round()) and float(n.abs().max()) <= 7 * 4)
        k = (w.reshape(48, 32, D) / w.reshape(48, 32, D).abs().amax(dim=(1, 2), keepdim=True)).reshape(1536, D) * 7
        self.assertLess(float((dout.float() @ w.t()).abs().max() / (2.0**-14 * 2.0**-6)), 2.0**15)
        self.assertTrue(torch.isfinite(k).all())
        again, _ = do8.mx6_lattice_operands(2048, do8._MX6_SELF_CHECK_SEED, torch.device("cpu"))
        other, _ = do8.mx6_lattice_operands(2048, 7, torch.device("cpu"))
        self.assertTrue(torch.equal(again, dout))
        self.assertFalse(torch.equal(other, dout))


class _TorchCpu:
    """l11_do8's `torch` for Mx6SelfCheckCpuTest: torch itself, except that torch.device(...) is the CPU."""

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    @staticmethod
    def device(*a: Any, **k: Any) -> torch.device:
        return torch.device("cpu")


class Mx6SelfCheckCpuTest(_Reset):
    """self_check_mx6 itself on CPU: the device, Triton's mask generator, the work-list builder, r14's encode and both
    MX6-.so entries are stand-ins (the entries record the keep mask they get and return equal outputs, so the verdict
    turns on the mask handling alone; the kernels' numerics are the GPU kits'). The mask each entry gets follows the
    LOADED MX6 .so's format: a NIB .so (MX6 + NIB, MX6_NIB_GATED) gets lnmd_mask_nib.pack_ref(mask) at all four
    launches (FP8 twin, MX6, determinism x2) and the round trip unpack(pack_ref(mask)) == mask is part of the verdict
    and of the detail; an int8 .so gets the int8 mask itself and its detail has no NIB segment."""

    def run_self_check(self, impl: str, poison: bool = False) -> Tuple[bool, str, List[Tuple[str, Any]], Any]:
        try:
            from generative_recommenders.ops.triton import triton_hstu_linear as thl
        except Exception as error:  # noqa: BLE001
            self.skipTest(f"triton_hstu_linear not importable: {error}")
        bwd_fp8 = _bwd_fp8()
        seen: List[Tuple[str, Any]] = []
        made: List[torch.Tensor] = []

        def mask_fn(**k: Any) -> torch.Tensor:
            m = torch.randint(0, 8, (k["N"], k["D"]), generator=torch.Generator().manual_seed(3), dtype=torch.int8)
            if poison:
                m[0, 0] = 8  # bit 3 set: the NIB layout cannot hold it, unpack(pack_ref(m)) != m
            made.append(m)
            return m

        def out(x: Any, d8: Any, e8: Any, dx: Any) -> Tuple[Any, ...]:
            d8.view(torch.uint8).zero_()
            e8.fill_(127)
            z = torch.zeros((8, D))
            return (dx if dx is not None else torch.empty_like(x), z, z.clone(), z.clone(), torch.zeros(8), 8)

        def mx6(g6, w6, w6s, x, u, mask, mean, rstd, wn, bn, p, inv, dz8, wl, cnt, cap, d8, e8, dx=None):  # noqa
            seen.append(("mx6", mask))
            return out(x, d8, e8, dx)

        def twin(fn, head, entry, x, u, mask, mean, rstd, wn, bn, p, inv, dz8, wl, cnt, cap, d8, e8, dx, m6):  # noqa
            seen.append(("twin", mask))
            return out(x, d8, e8, dx)

        def qrm(dout: torch.Tensor, inv: torch.Tensor) -> Tuple[torch.Tensor, None]:
            return (dout.float() * inv).to(E5), None

        def qw(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            sb = torch.full((1,), 2.0**-12)
            return (w / sb).to(E4), sb

        def enc(dout: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            return (torch.zeros((dout.shape[0], do8.G6_ROW), dtype=U8), torch.zeros(do8.W6_BYTES, dtype=U8),
                    torch.zeros(do8.W6S_BYTES, dtype=U8))

        with contextlib.ExitStack() as st:
            for target, attr, kw in (
                (do8, "torch", {"new": _TorchCpu()}),
                (do8, "_fn_launch_mx6", {"new": object()}), (do8, "_fn_launch_mx6_fp8", {"new": object()}),
                (do8, "_fn_wl_build", {"new": lambda *a: 0}),
                (do8, "_fn_wl_cap", {"new": lambda m, b: (m + 63) // 64 + b + 2}),
                (do8, "_mx6_lib", {"new": object()}), (do8, "_impl_mx6", {"new": impl}),
                (do8, "launch_e5_only_mx6", {"new": mx6}), (do8, "_launch_mx6_so", {"new": twin}),
                (torch.cuda, "current_device", {"return_value": 0}),
                (torch.cuda, "current_stream", {"new": lambda d=None: SimpleNamespace(cuda_stream=0)}),
                (torch.cuda, "synchronize", {"new": lambda *a: None}),
                (thl, "_create_dropout_mask", {"new": mask_fn}),
                (bwd_fp8, "grad_dtype_and_limit", {"return_value": (E5, 57344.0)}),
                (bwd_fp8, "uvqk_tm_pad", {"return_value": 128}),
                (bwd_fp8, "quantize_row_major", {"new": qrm}), (bwd_fp8, "quantize_weight", {"new": qw}),
                (bwd_fp8, "output_projection_mx6_encode", {"create": True, "new": enc}),
            ):
                st.enter_context(mock.patch.object(target, attr, **kw))
            ok, detail = do8.self_check_mx6()
        return ok, detail, seen, made[0]

    def test_nib_so_gets_pack_ref_at_every_launch_and_gates_the_round_trip(self) -> None:
        ok, detail, seen, mask = self.run_self_check(IMPL_M_NIB)
        self.assertTrue(ok, detail)
        self.assertEqual([k for k, _ in seen], ["twin", "mx6", "mx6", "mx6"])  # FP8 twin, MX6, determinism x2
        packed = lnmd_mask_nib.pack_ref(mask)
        for kind, m in seen:
            self.assertTrue(lnmd_mask_nib.layout_ok(m, mask.shape[0]) and torch.equal(m, packed), kind)
        self.assertIs(seen[1][1], seen[0][1])  # ONE packed mask for every launch
        self.assertTrue(detail.endswith("; mask nib(uint8[M,256]) unpack(pack_ref) exact 1"), detail)
        ok, detail, seen, mask = self.run_self_check(IMPL_M_NIB, poison=True)
        self.assertFalse(ok, detail)  # the round trip is part of the verdict
        self.assertTrue(detail.endswith("; mask nib(uint8[M,256]) unpack(pack_ref) exact 0"), detail)

    def test_int8_so_gets_the_int8_mask_itself(self) -> None:
        ok, detail, seen, mask = self.run_self_check(IMPL_M_ONLY)
        self.assertTrue(ok, detail)
        self.assertEqual([k for k, _ in seen], ["twin", "mx6", "mx6", "mx6"])
        self.assertTrue(all(m is mask for _, m in seen))
        self.assertNotIn("mask nib", detail)
        ok, detail, seen, mask = self.run_self_check(IMPL_M_ONLY, poison=True)
        self.assertTrue(ok, detail)  # an int8 .so reads the int8 bytes as they are: no round trip asked


# ------------------------------------------------------------------------------------------------ the MX6 launch's mask
class Mx6LaunchMaskTest(_Reset):
    """launch_e5_only_mx6, the raw MX6 launch (self_check_mx6 calls it without the seam's preflight), checks the keep
    mask against the LOADED MX6 .so's format and layout before the kernel entry: a mismatch raises Mx6OperandError
    "mask format: ..." and nothing is launched; a matching mask reaches the entry's MK as the same pointer (never
    unpacked)."""

    M = 300

    def launch(self, mask: torch.Tensor, impl: str, mk: List[int]) -> None:
        def entry(*args: Any) -> int:
            mk.append(int(args[8]))  # G6, W6, W6S, sa, sb, M, X, U, MK, ...
            return 0

        def fn_plan(cap: int, addr: int) -> int:
            out = ctypes.cast(addr, ctypes.POINTER(ctypes.c_longlong))
            for i, v in enumerate((1, 0, 1, 8)):  # G, fl, nf, P
                out[i] = v
            return 0

        m = self.M
        with contextlib.ExitStack() as st:
            for target, attr, kw in (
                (do8, "_mx6_lib", {"new": object()}), (do8, "_impl_mx6", {"new": impl}),
                (do8, "_fn_launch_mx6", {"new": entry}), (do8, "_fn_drain", {"new": lambda: 0}),
                (do8, "_fn_plan", {"new": fn_plan}), (torch.cuda, "current_device", {"return_value": 0}),
                (torch.cuda, "device", {"new": lambda i: contextlib.nullcontext()}),
                (torch.cuda, "current_stream", {"new": lambda d=None: SimpleNamespace(cuda_stream=0)}),
            ):
                st.enter_context(mock.patch.object(target, attr, **kw))
            st.enter_context(mock.patch.dict(do8._plan_cache, clear=True))
            do8.launch_e5_only_mx6(
                torch.zeros((m, do8.G6_ROW), dtype=U8), torch.zeros(do8.W6_BYTES, dtype=U8),
                torch.zeros(do8.W6S_BYTES, dtype=U8), torch.zeros(m, D, dtype=BF), torch.zeros(m, D, dtype=BF), mask,
                torch.zeros(m), torch.ones(m), torch.ones(D, dtype=BF), torch.zeros(D, dtype=BF), 0.1, torch.ones(1),
                torch.zeros((m + 127) // 128 * 128, l11.DU_ROW, dtype=U8).view(E5),
                torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 20,
                torch.zeros(m, D, dtype=U8).view(E4),
                torch.zeros(m, do8.H, dtype=U8),
            )

    def test_the_launch_checks_the_loaded_formats_mask_before_the_entry(self) -> None:
        lnmd_mask_nib.reset_for_testing()
        gen = torch.Generator().manual_seed(5)
        m8 = torch.randint(0, 8, (self.M, D), generator=gen, dtype=torch.int8)
        nib = lnmd_mask_nib.pack_ref(m8)
        strided = lnmd_mask_nib.pack_ref(torch.randint(0, 8, (2 * self.M, D), generator=gen, dtype=torch.int8))[::2]
        for mask, impl in ((nib, IMPL_M_NIB), (m8, IMPL_M_ONLY)):
            mk: List[int] = []
            self.launch(mask, impl, mk)
            self.assertEqual(mk, [mask.data_ptr()], impl)  # the saved tensor itself (NIB: the packed bytes)
        for name, mask, impl in (("int8 mask, NIB .so", m8, IMPL_M_NIB), ("NIB mask, int8 .so", nib, IMPL_M_ONLY),
                                 ("NIB rows - 1", nib[1:], IMPL_M_NIB), ("NIB row stride 512", strided, IMPL_M_NIB)):
            mk = []
            with self.assertRaisesRegex(do8.Mx6OperandError, r"^mask format: keep mask .* \(M=300\)$"):
                self.launch(mask, impl, mk)
            self.assertEqual(mk, [], name)  # raised before the entry
        self.assertEqual(lnmd_mask_nib.counters()["unpack"], 0)


# ------------------------------------------------------------------------------------------------ contract / operand checks
def _ops6(m: int = 300, **over: Any) -> Any:
    f = {
        "g6": torch.zeros((m, 400), dtype=U8),
        "w6": torch.zeros(589824, dtype=U8),
        "w6s": torch.zeros(768, dtype=U8),
        "dout_fp8": torch.zeros(((m + 127) // 128, D, 128), dtype=E4),
        "dout_fp8_scale": torch.ones(1),
        "g_decode_scale": torch.ones(1),
        "inv_scale_used": torch.ones(1),
        "rescue_args": do8.Mx6RescueArgs(True, True, E4, 2.0**-5, 448.0, 5, True),
    }
    f.update(over)
    return do8.Mx6Operands(**f)


def _misaligned_dout_t(m: int = 300) -> torch.Tensor:
    """The step-tiled e4m3 [ceil(m/128), 512, 128] W3 operand, contiguous, at an address that is not 16-B aligned."""
    kt = (m + 127) // 128
    t = torch.zeros(kt * D * 128 + 16, dtype=U8)[1:kt * D * 128 + 1].view(E4).view(kt, D, 128)
    assert t.is_contiguous() and t.data_ptr() % 16 != 0
    return t


class Mx6ChecksTest(unittest.TestCase):
    def test_contract_mismatch(self) -> None:
        cpu = torch.device("cpu")
        self.assertIsNone(do8._mx6_contract_mismatch(_ops6(), 300, cpu))
        bad = {
            "not Mx6Operands": (tuple(_ops6()), "not l11_do8.Mx6Operands"),
            "2-D dout_t": (_ops6(dout_fp8=torch.zeros((D, 384), dtype=E4)), "dout_fp8"),
            "e5m2 dout_t": (_ops6(dout_fp8=torch.zeros((3, D, 128), dtype=E5)), "dout_fp8"),
            "short dout_t": (_ops6(dout_fp8=torch.zeros((2, D, 128), dtype=E4)), "dout_fp8"),
            "misaligned dout_t": (_ops6(dout_fp8=_misaligned_dout_t()), "dout_fp8 torch.float8_e4m3fn (3, 512, 128)"),
            "scale shape": (_ops6(dout_fp8_scale=torch.ones(2)), "dout_fp8_scale"),
            "decode dtype": (_ops6(g_decode_scale=torch.ones(1, dtype=torch.float64)), "g_decode_scale"),
            "inv misaligned": (_ops6(inv_scale_used=torch.ones(2)[1:]), "inv_scale_used"),
            "inv shared": (_ops6(inv_scale_used=torch.ones(5)), "inv_scale_used"),
        }
        for name, (ops, want) in bad.items():
            self.assertIn(want, do8._mx6_contract_mismatch(ops, 300, cpu) or "", name)
        self.assertIn("a16=0", do8._mx6_contract_mismatch(bad["misaligned dout_t"][0], 300, cpu) or "")

    def test_contract_accepts_only_a_dout_t_x21_takes(self) -> None:
        """The W3 operand the contract accepts is one x21's own dout_t checks accept (wgrad_ck_w3._decline_reason on a
        conforming y8 and scales; on CPU everything but the device): a dout_t x21 declines would set W3's process-wide
        tiling latch (and with it switch E-RM's row-major entry off), so the contract rejects it first."""
        from generative_recommenders.ops.triton import wgrad_ck_w3

        cpu, m = torch.device("cpu"), 300
        kt = (m + 127) // 128
        y8, one = torch.zeros((m, 1536), dtype=E4), torch.ones(1)
        cases = {
            "conforming": torch.zeros((kt, D, 128), dtype=E4),
            "misaligned": _misaligned_dout_t(m),
            "non-contiguous": torch.zeros((kt, 128, D), dtype=E4).transpose(1, 2),
            "short": torch.zeros((kt - 1, D, 128), dtype=E4),
            "e5m2": torch.zeros((kt, D, 128), dtype=E5),
        }
        for name, t in cases.items():
            x21 = wgrad_ck_w3._decline_reason(y8, t, one, one, m)
            x21_takes = x21 is None or x21[0] == "device"  # CPU tensors: x21's last check is the CUDA device
            contract_takes = do8._mx6_contract_mismatch(_ops6(m, dout_fp8=t), m, cpu) is None
            self.assertEqual((contract_takes, x21_takes), (name == "conforming",) * 2, (name, x21))

    def test_operand_mismatch(self) -> None:
        cpu = torch.device("cpu")
        g6, w6, w6s = torch.zeros((300, 400), dtype=U8), torch.zeros(589824, dtype=U8), torch.zeros(768, dtype=U8)
        self.assertIsNone(do8._mx6_operand_mismatch(g6, w6, w6s, 300, cpu))
        padded = torch.zeros((300, 416), dtype=U8)[:, :400]
        shifted = torch.zeros(300 * 400 + 16, dtype=U8)[1:300 * 400 + 1].view(300, 400)
        for name, args, want in (
            ("rows", (torch.zeros((301, 400), dtype=U8), w6, w6s), "G6"),
            ("padded row pitch", (padded, w6, w6s), "row stride 400"),
            ("misaligned", (shifted, w6, w6s), "16-B aligned"),
            ("int8", (g6.view(torch.int8), w6, w6s), "G6"),
            ("W6 size", (g6, torch.zeros(589823, dtype=U8), w6s), "W6"),
            ("W6S size", (g6, w6, torch.zeros(767, dtype=U8)), "W6S"),
            ("W6S non-contiguous", (g6, w6, torch.zeros(1536, dtype=U8)[::2]), "W6S"),
        ):
            self.assertIn(want, do8._mx6_operand_mismatch(*args, 300, cpu) or "", name)

    def test_rescue_contract(self) -> None:
        """R2 on the host (mx6_rescue): the out-proj site's calls must not move and the result must be the FP8 dgrad
        operands; a break latches MX6 and raises Mx6ContractError; a conforming rescue is returned as is."""
        cpu = torch.device("cpu")
        m = 300
        good = (torch.zeros((m, D), dtype=E5), torch.ones(1), torch.zeros((1536, D), dtype=E4), torch.ones(1))
        self.assertIsNone(do8._mx6_rescue_mismatch(good, m, cpu))
        for name, out, want in (
            ("list", list(good), "not (g8, g8_scale, w8, w8_scale)"),
            ("5-tuple", good + (good[1],), "not (g8, g8_scale, w8, w8_scale)"),
            ("None in it", (None,) + good[1:], "not (g8, g8_scale, w8, w8_scale)"),
            ("g8 e4m3", (good[0].view(E4),) + good[1:], "g8 "),
            ("g8 bf16", (torch.zeros((m, D), dtype=BF),) + good[1:], "g8 "),
            ("g8 rows", (torch.zeros((m + 1, D), dtype=E5),) + good[1:], "g8 "),
            ("g8 non-contiguous", (torch.zeros((D, m), dtype=E5).t(),) + good[1:], "g8 "),
            ("w8 shape", good[:2] + (torch.zeros((D, 1536), dtype=E4), good[3]), "w8 "),
            ("w8 e5m2", good[:2] + (torch.zeros((1536, D), dtype=E5), good[3]), "w8 "),
            ("g8_scale [2]", (good[0], torch.ones(2)) + good[2:], "g8_scale"),
            ("w8_scale fp64", good[:3] + (torch.ones(1, dtype=torch.float64),), "w8_scale"),
        ):
            self.assertIn(want, do8._mx6_rescue_mismatch(out, m, cpu) or "", name)
        state = SimpleNamespace(calls=51)
        dout, weight, ops6 = torch.zeros((m, D), dtype=BF), torch.zeros((1536, D)), _ops6(m)
        for name, rescue, want in (
            ("conforming", lambda *a: good, None),
            ("second bookkeeping", lambda *a: (setattr(state, "calls", state.calls + 1), good)[1],
             "broke the R2 contract: the out-proj e5m2 site's bookkeeping ran 1 times"),
            ("other operands", lambda *a: good[:3], "broke the R2 contract: the rescue returned tuple"),
        ):
            do8.reset_for_testing()
            state.calls = 51
            seen: List[Tuple[Any, ...]] = []

            def g8_rescue(*a: Any, rescue: Any = rescue) -> Any:
                seen.append(a)
                return rescue(*a)

            with mock.patch.object(_bwd_fp8(), "output_projection_g8_rescue", create=True, new=g8_rescue), \
                    mock.patch.object(_bwd_fp8(), "_peek_site_state", new=lambda site, disc: state), _Capture():
                if want is None:
                    self.assertIs(do8.mx6_rescue(dout, weight, ops6), good)
                    self.assertIsNone(do8.mx6_broken())
                else:
                    with self.assertRaisesRegex(do8.Mx6ContractError, want, msg=name):
                        do8.mx6_rescue(dout, weight, ops6)
                    self.assertIsNotNone(do8.mx6_broken(), name)
            self.assertEqual(len(seen), 1, name)
            self.assertTrue(seen[0][0] is dout and seen[0][1] is weight and seen[0][2] is ops6, name)
        do8.reset_for_testing()

    def test_decision_object(self) -> None:
        d = do8.Mx6Decision("S", 10, 1, True, None, open_=True)
        self.assertTrue(d.taken)
        for only_plan, reason, open_ in ((False, None, True), (None, None, True), (True, "x", True), (True, None, False)):
            self.assertFalse(do8.Mx6Decision("S", 10, 1, only_plan, reason, open_).taken)
        self.assertIn("layer=1", repr(d))

    def test_named_tuples_are_the_contract(self) -> None:
        self.assertEqual(do8.Mx6Operands._fields, ("g6", "w6", "w6s", "dout_fp8", "dout_fp8_scale", "g_decode_scale",
                                                   "inv_scale_used", "rescue_args"))
        self.assertEqual(do8.Mx6RescueArgs._fields, ("token_major", "t_tiled", "t_dtype", "t_inv_mul", "t_limit",
                                                     "t_shift", "helpers_fused"))
        self.assertEqual(do8.MX6_PRODUCER_API, _PRODUCER)
        self.assertEqual((do8.G6_ROW, do8.W6_BYTES, do8.W6S_BYTES), (400, 589824, 768))


# ------------------------------------------------------------------------------------------------ GPU
def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        return l11.device_arch(torch.cuda.current_device()) == "gfx950"
    except Exception:  # noqa: BLE001
        return False


def _producer_present() -> bool:
    try:
        return not do8._mx6_producer_missing()
    except Exception:  # noqa: BLE001
        return False


# Mx6GpuTest's NIB case: the NIB knob and the NIB writers' knobs (l11_do8._nib_writers_off: maskgen, mask16, LNMD fwd
# v2; the out-proj wgrad not in "producer" mode), so DO8-ONLY builds its NIB 'only' .so and MX6 its NIB build.
_GPU_NIB_ENV: Dict[str, str] = {"DLRM_HSTU_FP8_L11_DO8_ONLY": "1", "DLRM_HSTU_FP8_L11_VD": "1",
                                "DLRM_HSTU_FP8_MASK16": "1", "DLRM_HSTU_FP8_BWD_DU_E5M2": "1",
                                "DLRM_HSTU_FP8_LNMD_MASK_NIB": "1", "DLRM_HSTU_FP8_LNMD_FWD_MASKGEN": "1",
                                "DLRM_HSTU_FP8_LNMD_FWD_V2": "1"}


class Mx6GpuNibEnvTest(_Reset):
    def test_the_gpu_nib_case_requests_the_nib_builds(self) -> None:
        # CPU: the NIB case's environment selects the NIB 'only' build of the shipped source (so its MX6 build is NIB)
        with mock.patch.dict(os.environ, _GPU_NIB_ENV):
            os.environ.pop("DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL", None)
            self.assertIsNone(do8._nib_writers_off())
            self.assertTrue(do8.nib_build_requested(do8._SOURCE))
            self.assertIn(do8._NIB_DEFINE, do8._flags(do8._SOURCE.read_bytes(), mx6=True))


@unittest.skipUnless(_gfx950() and _producer_present(), "needs a gfx950 GPU and r14's MX6 producer API in bwd_fp8")
class Mx6GpuTest(unittest.TestCase):
    """The production prebuild chain on hardware: DO8 'plus', DO8-ONLY, then the MX6 .so (build, gate, load) and its
    lattice self-check (MX6 'only' vs the FP8 'only' twin byte for byte, the determinism twin); int8 builds, and NIB
    builds (MX6 + NIB, MX6_NIB_GATED: the self-check feeds both MX6-.so entries pack_ref(mask), the round trip
    gated)."""

    def test_prebuild_and_self_check_nib(self) -> None:
        with mock.patch.dict(os.environ, _GPU_NIB_ENV):
            os.environ.pop("DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL", None)  # not "producer": the NIB writers can arm
            do8.reset_for_testing(unload=True)  # no .so of the other keep-mask format stays loaded
            do8.set_requested_for_testing(True)
            do8.set_diet_for_testing(True)
            do8.set_mx6_for_testing(True)
            l11.set_requested_for_testing(True)
            try:
                l11.set_prebuilt_for_testing(True)
                self.assertTrue(do8.nib_build_requested())
                self.assertTrue(do8.prebuild(True, True))
                self.assertTrue(do8.only_prebuilt(), do8.only_broken())
                self.assertEqual(do8.only_mask_format(), "nib", do8.only_impl())
                self.assertTrue(do8.mx6_enabled(), do8.mx6_broken() or do8.mx6_inert())
                self.assertEqual(do8.mx6_mask_format(), "nib", do8.mx6_impl())
                ok, detail = do8.self_check_mx6()
                self.assertTrue(ok, detail)
                self.assertTrue(detail.endswith("; mask nib(uint8[M,256]) unpack(pack_ref) exact 1"), detail)
            finally:
                do8.set_requested_for_testing(None)
                do8.set_diet_for_testing(None)
                do8.set_mx6_for_testing(None)
                l11.set_requested_for_testing(None)
                do8.reset_for_testing(unload=True)

    def test_prebuild_and_self_check(self) -> None:
        env = {"DLRM_HSTU_FP8_L11_DO8_ONLY": "1", "DLRM_HSTU_FP8_L11_VD": "1", "DLRM_HSTU_FP8_MASK16": "1",
               "DLRM_HSTU_FP8_BWD_DU_E5M2": "1"}
        with mock.patch.dict(os.environ, env):
            do8.reset_for_testing()
            do8.set_requested_for_testing(True)
            do8.set_diet_for_testing(True)
            do8.set_mx6_for_testing(True)
            l11.set_requested_for_testing(True)
            try:
                l11.set_prebuilt_for_testing(True)
                self.assertTrue(do8.prebuild(True, True))
                self.assertTrue(do8.only_prebuilt(), do8.only_broken())
                self.assertTrue(do8.mx6_enabled(), do8.mx6_broken() or do8.mx6_inert())
                ok, detail = do8.self_check_mx6()
                self.assertTrue(ok, detail)
            finally:
                do8.set_requested_for_testing(None)
                do8.set_diet_for_testing(None)
                do8.set_mx6_for_testing(None)
                l11.set_requested_for_testing(None)
                do8.reset_for_testing()


if __name__ == "__main__":
    unittest.main()

# pyre-unsafe
"""CPU tests of r14's MX6 dY producer ops/hip/outproj_dual_mx6.{cu,py} (DESIGN v1 FROZEN, r14 iteration 66, sec. 13:
T11-T16, T18, T19; the seam tests T1-T10 / T17 live in the bwd_fp8 seam suite). No GPU, no launch.

  T11  loader CPU behaviour: import is side-effect free (subprocess + audit hook), not_armed_reason(), prebuild fails
       closed and latches on a host without GPU / toolchain (one DISABLED line, then silent), the self-check outcomes,
       FAIL_CLOSED, the cache key components, dev defines, trust / mkdtemp fallback, the atomic install (temporary
       names, sidecar replaced before the .so, nothing left behind), cache hit / faulty-entry rebuild, the gate called
       with the exact mangled names, killpg on a hipcc timeout (fake hipcc), _Lib's ABI / impl / geometry checks, rc
       handling, S1-E reaching prebuild(); and odm.self_check() (SC0-SC6) run end to end on the CPU with the lane emulator
       as the device (EmuDevice: the .cu's parsed launcher checks + emulated kernels on host memory), plus injected
       device faults (heavy).
  T12  reference parity (no kernel): the loader's in-tree G6 encoder / adversarial rows / W6 references (the ones the
       prebuild self-check uses) and the emulator's ports vs r16 REF bytes (U1's npz30 vectors, copied into
       outproj_dual_mx6_vectors.npz with pinned sha256s; REF pins computed once from mx6_l11_ref.py c49f2659 /
       w6_layout.py 08dc4c64); the W6 index map of the .cu vs the S1 CONTRACT sec. 6 scatter inverse and w6_layout.
  T13  the lane emulator (outproj_dual_mx6_emu.py: DESIGN App. B, every map parsed from the .cu text) vs the in-tree
       G6 encoder and the Triton dual-cast model at the DESIGN sec. 13 M list, npz30, C4 tiles, inv {publish(143),
       NaN, +Inf, 0, -1.0}, t 4..7, modes pk32 and f32dn1; write-once, staging bijection, 0 LDS bank conflicts; source
       edits of the sec. 5 maps fail.
  T14  launcher parity: literals (4194304, the four t bit patterns, 67712, ABI, IMPL, SOURCE_SHA256, variant bits,
       partial count, geometry), SIGNATURES vs the C prototypes, ineligible_reason vs the parsed C checks.
  T15  the resource-gate parser on fixtures in both remark formats (accept the pinned census; reject each violation).
  T16  proxy census + ISA lint (skipped without /opt/COE_modules/_stray_installs/rocm-7.2.0): the loader's own argv,
       census of DESIGN sec. 7, FP6 convert wrapping / disjointness, permlane16_swap hazards, MODE fields, s_setreg,
       the vmcnt rule; the real .so bound by _Lib and its C launchers' host rejections.
  T18  each of DESIGN App. B's 10 mutants fails >= 1 T13 case.
  T19  every line starts with "[outproj-mx6]" and carries none of r16's 5 must-be-ABSENT tokens.

Known defects of the code under test are strict xfails (DEFECTS below; none open). ODM6-LDR-1 (format-B remarks whose
source path holds whitespace parsed to no census) was one; the loader's v1.1 fixed it and its two tests must now pass.
v1.1 regression tests: one physical line per message and a hipcc failure's diagnostic + kept log (MINOR-1), SC1's t
values under a fixed DFMT_SHIFT (CI-3).
v1.2 (r16 iteration 59): R-4, S1C 2.5 as amended (L2(a)): G6L2aRuleTest (odm.g6_mismatch_l2a / g6_l2a_counts vs R4.md
sec. 10's verbatim numpy reference, the 64 x 64 code-pair sweep, exact-count controls, 7 mutants caught), and on the
emulated kernel with G6 converter faults (EmuDevice, G6_L2A_FAULTS): _sc_case, the whole self-check with the 0x00-for-
0x20 quirk (NIT-3: it now passes and reports the codes) and audit_last_bands; the heavy fault test adds a nonzero sign
flip (G6) and per-launch zero signs (twin). R-3: prebuild() re-raises an rc 700 / 719 OutprojMx6Error from its
self-check unchanged (a hook caller must classify by text). The T12 / T13 model-vs-model G6 compares stay strict.

Heavy parametrizations (pytest marker `heavy`) run only with OUTPROJ_MX6_TEST_HEAVY=1. OUTPROJ_MX6_TEST_REPORT=<path>
appends one JSON line per reported result (T13 / T18 / T16 detail) to <path>.
"""

import contextlib
import ctypes
import hashlib
import io
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock

import numpy as np
import torch

from generative_recommenders.ops.hip import outproj_dual_mx6 as odm
from generative_recommenders.ops.tests import outproj_dual_mx6_emu as E

_UNDER_PYTEST = "_pytest" in sys.modules  # collected by pytest (vs python -m unittest)
try:
    import pytest

    warnings.filterwarnings("ignore", message=r".*pytest\.mark\.heavy.*")
    _HEAVY_MARK = pytest.mark.heavy
except Exception:  # noqa: BLE001 -- pytest absent: unittest only
    _HEAVY_MARK = None

HEAVY = os.environ.get("OUTPROJ_MX6_TEST_HEAVY", "0").strip() == "1"
PROXY_ROCM = "/opt/COE_modules/_stray_installs/rocm-7.2.0"
REPO = Path(odm.__file__).resolve().parents[3]


def heavy(obj: Any) -> Any:
    obj = unittest.skipUnless(HEAVY, "heavy: set OUTPROJ_MX6_TEST_HEAVY=1")(obj)
    return _HEAVY_MARK(obj) if _HEAVY_MARK is not None else obj


# Known open defects of the code under test, by id (each test that shows one carries @xfail_defect(id)). None open:
# ODM6-LDR-1 (_REMARK_RE's format-B file group was \S+?, so a source path with whitespace parsed to no census and the
# gate failed closed) is fixed in the loader's v1.1.
DEFECTS: Dict[str, str] = {}


def xfail_defect(defect_id: str) -> Any:
    """A known defect of the code under test (not changed here): strict xfail (an unexpected pass fails)."""
    def deco(fn: Any) -> Any:
        if _UNDER_PYTEST and _HEAVY_MARK is not None:
            return pytest.mark.xfail(strict=True, reason=f"{defect_id}: {DEFECTS[defect_id]}")(fn)
        return unittest.expectedFailure(fn)  # unittest: an unexpected success fails the run
    return deco


def report(tag: str, **kw: Any) -> None:
    path = os.environ.get("OUTPROJ_MX6_TEST_REPORT", "").strip()
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(dict(tag=tag, **kw), default=str) + "\n")


def sha(a: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------------------------------- pins
# DESIGN v1 Appendix A (the frozen kernel text) and its normative census (sec. 7, ROCm 7.2.0 proxy).
CU_SHA256_FROZEN = "e6c885b51cd359e249a350def3a6ce2b620b4752d0c468ab0989f98f290a84b3"
# outproj_dual_mx6_vectors.npz: arrays copied from r14 iter66 U1 spec/g6_vectors_v1.npz (sha256 107ff608...; REF bytes
# g6_ref = r16 mx6_l11_ref.encode_g6(dout_bits), mx6_l11_ref.py sha256 c49f2659...).
NPZ_PINS = {
    "dout_bits": "de3b7ffd059e7a4f318a40c716b5742be97d11c5ec0c61239c7afe63241b2ae8",
    "g6_ref": "da10b923eb1dc95bf611eefdafd5edcd288a58451b50c10d81a4eaa775f347f3",
    "hand_bits": "449633a04f675ad8d915b3bd3c5e9a7c0072227fe3a397401ed4e0df7f2d510f",
    "hand_byte": "438ec80eca9c1b9f9e19dbd1a62f29df0cb3506569284d94521630431a0625bc",
    "hand_group": "8253123a5490ceee1ee40f89487c60acfbc60fb45f695848fa044c3bad11bd6e",
}
# REF pins, computed once (2026-10-13, CPU) by exec-loading r16 REF mx6_l11_ref.py (sha256 c49f2659cda32fcc...) and
# kern/w6_layout.py (sha256 08dc4c645153fc6b...) read-only: (input sha256, REF output sha256).
REF_G6_LAB = {769: ("a122d009d98133573d4665f22a9f2436530b372953caf6a62634fb5cde0f2c27",
                    "cd961273df56f25bc9c6a3fb4f1d8fda1b1e684b50df2a9b54b459a4ad553838"),
              1000: ("14524ed111ca85aa13189425f2b82dcc3bb8f9157370042845fd116c751c8ea4",
                     "799ed9403cd22303ace0e257eef7b22999671f4ca57f7829b2c2eac0bd1d2d64")}
REF_E8M0_TABLE = "ea93faad30263473e563356a2e5085f753f3f72776fdc0fb035d8ff3bba1c1c7"  # e8m0_exp_exact + 127, |bf16|
REF_CODES_TABLE = ("c32c8f702fa7355a9dcca6cdde3d584c2b42d415b094a587d2bb4679f9e974d9", 8412762)  # e2m3_codes pairs
REF_W6 = {  # weight -> (fp32 input bits sha256 or None, REF w6_pack W6 sha256, W6S sha256)
    "edge_fp32": ("83d06986f24fdde44c43b5c8814817982a1be2388d826af435596dcc8dd37716",
                  "41ec1ad77c19b1af72efa15bc9fda582a004dc5157b2781475244307e5d8fc8d",
                  "997e522ea1eacb24138ab0f51d5bb27fd6aa4b1ee1489a0189f001f32b6a6a84"),
    "hash_fp32": ("68dce47ca6bfc509491581d051f91cfc4bc48c53fb9934a271d52b1e61a5807f",
                  "c774f391b26644253660f86be86771752a29ab2b74e585b902e212c46511cd88",
                  "9a458c389f93ca77312ce8550782cd28dd0112cc096fc3a520f0428fa29680dd"),
    "edge_bf16": (None, "fbecfff570667327270fb1b873c1de93664f7ccefe63042eb00a739b950f10c2",
                  "916a6f0981b22a0fdc882e46b3387036618525bcd90f02f21c9e39fc3c81d5e0"),
    "hash_bf16": (None, "b94851747756361d75e71ae128871ec0e30b2dd2f37dbbb3b12fb5165d67017c",
                  "4f91e83f510b2cc5567acdc8c4b8e65a05a1f0d9a2652ca6e382f7335d2bf53a"),
}
W6_SRC_TABLE_SHA = "69cfaef8dc49cb26925c80446f0010eda7e0cf3521a033ecd2fe386bd773290d"  # w6_layout.w6_src_table() int64
ABSENT_TOKENS = ("MX6 DISABLED", "MX6 INERT", "MX6 release", "MX6 rescue", "DEVICE FAULT")  # r16's must-be-ABSENT

# The resource census of the default build (DESIGN sec. 7 [M]; ROCm 7.2.0 proxy remarks, build.log of the loader's own
# argv), as hipcc prints it (key order of the remark block).
REMARK_KEYS = ("TotalSGPRs", "VGPRs", "AGPRs", "ScratchSize [bytes/lane]", "Dynamic Stack", "Occupancy [waves/SIMD]",
               "SGPRs Spill", "VGPRs Spill", "LDS Size [bytes/block]")
CENSUS = {
    "producer": dict(zip(REMARK_KEYS, ("38", "103", "0", "0", "False", "4", "0", "0", "67712"))),
    "w6_bf16": dict(zip(REMARK_KEYS, ("16", "102", "0", "0", "False", "4", "0", "0", "0"))),
    "w6_f32": dict(zip(REMARK_KEYS, ("16", "102", "0", "0", "False", "4", "0", "0", "0"))),
}
GATE_PINNED = {
    "producer": dict(vgpr=103, sgpr=38, vgpr_spill=0, sgpr_spill=0, scratch=0, occupancy=4, lds=67712, agpr=0),
    "w6_f32": dict(vgpr=102, sgpr=16, vgpr_spill=0, sgpr_spill=0, scratch=0, occupancy=4, lds=0, agpr=0),
    "w6_bf16": dict(vgpr=102, sgpr=16, vgpr_spill=0, sgpr_spill=0, scratch=0, occupancy=4, lds=0, agpr=0),
}
# verbatim excerpt of the real --save-temps log (format B) of the default build (producer block)
REAL_FORMAT_B_PRODUCER = """\
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0: Function Name: _ZN4odm623outproj_dual_mx6_kernelEPKhPKffiiPhS4_PfPj [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     TotalSGPRs: 38 [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     VGPRs: 103 [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:213:0:     LDS Size [bytes/block]: 67712 [-Rpass-analysis=kernel-resource-usage]
"""


def remark_log(census: Dict[str, Dict[str, str]], fmt: str, roles: Sequence[str] = ("producer", "w6_bf16", "w6_f32"),
               names: Optional[Dict[str, str]] = None) -> str:
    """A -Rpass-analysis=kernel-resource-usage log in format 'A' ("<file>:<l>:<c>: remark: ...", with clang's source
    snippet lines) or 'B' ("remark: <file>:<l>:<c>: ...", --save-temps)."""
    names = names or odm.KERNELS
    out: List[str] = []
    for role in roles:
        ln = 213 if role == "producer" else 310
        for key in ("Function Name",) + REMARK_KEYS:
            val = names[role] if key == "Function Name" else census[role][key]
            ind = "" if key == "Function Name" else "    "
            tail = " [-Rpass-analysis=kernel-resource-usage]"
            if fmt == "B":
                out.append(f"remark: tree/generative_recommenders/ops/hip/outproj_dual_mx6.cu:{ln}:0: {ind}{key}: {val}"
                           + tail)
            else:
                out.append(f"/x/odm6.cu:{ln}:1: remark: {ind}{key}: {val}" + tail)
                if key == "Function Name":
                    out += [f"  {ln} |                              uint32_t* __restrict__ inv_used) {{", "      | ^"]
    return "\n".join(out) + "\n"


def mutate(census: Dict[str, Dict[str, str]], role: str, **kv: str) -> Dict[str, Dict[str, str]]:
    out = {r: dict(v) for r, v in census.items()}
    out[role].update(kv)
    return out


class FakeCuda(torch.Tensor):
    """A tensor (meta storage: no allocation) that claims cuda:0 and a chosen data_ptr (T14 / T19 metadata checks)."""

    @staticmethod
    def __new__(cls, base: torch.Tensor, ptr: int) -> "FakeCuda":
        t = torch.Tensor._make_subclass(cls, base)
        t._fake_ptr = int(ptr)
        return t

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return torch.device("cuda", 0)

    def data_ptr(self) -> int:  # type: ignore[override]
        return self._fake_ptr


def fake(shape: Sequence[int], dtype: torch.dtype, ptr: int) -> FakeCuda:
    return FakeCuda(torch.empty(tuple(shape), dtype=dtype, device="meta"), ptr)


def launch_operands(m: int, m_pad: Optional[int] = None, base: int = 1 << 24) -> List[Any]:
    """launch() / ineligible_reason() operands for M rows, all 16-B aligned, distinct fake pointers."""
    mp = m_pad if m_pad is not None else (m + 127) // 128 * 128
    step = 1 << 40
    return [fake((m, 512), torch.bfloat16, base), fake((1,), torch.float32, base + step), 2.0 ** -5, m, mp,
            fake((m, 400), torch.uint8, base + 2 * step), fake((max(mp // 128, 1), 512, 128), torch.float8_e4m3fn,
                                                               base + 3 * step),
            fake((odm.partial_count(m),), torch.float32, base + 4 * step), fake((1,), torch.float32, base + 5 * step)]


class Lines:
    """Captures every line the loader prints (arm_print_all_ranks is the only sink) and the real stdout."""

    def __init__(self) -> None:
        self.msgs: List[str] = []
        self.out = io.StringIO()
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> "Lines":
        real = odm.arm_print_all_ranks

        def sink(msg: str) -> None:
            self.msgs.append(msg)
            real(msg)

        self._stack.enter_context(mock.patch.object(odm, "arm_print_all_ranks", side_effect=sink))
        self._stack.enter_context(contextlib.redirect_stdout(self.out))
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stack.close()

    def printed(self) -> List[str]:
        return [ln for ln in self.out.getvalue().splitlines() if ln.strip()]


def assert_line_rules(tc: unittest.TestCase, lines: Sequence[str]) -> None:
    for ln in lines:
        tc.assertTrue(ln.startswith(odm.FAMILY), ln)
        for tok in ABSENT_TOKENS:
            tc.assertNotIn(tok.lower(), ln.lower(), ln)


def c_launch_reject(model: E.CuModel, d: Dict[str, Any]) -> bool:
    """outproj_dual_mx6_launch's argument checks, evaluated from the .cu text (pointers as integers, None = NULL)."""
    env: Dict[str, Any] = {k: E.CV(np.int64(d.get(k) or 0), "i64")
                           for k in ("dout", "inv", "g6", "dout_t", "partial", "inv_used")}
    env.update({"nullptr": E.CV(np.int64(0), "i64"), "M": E.cv_i32(d["M"]), "M_PAD": E.cv_i32(d["M_PAD"]),
                "tb": E.cv_u32(int(np.float32(d["t"]).view(np.uint32))),
                "odm6_mis16": lambda p: E.CV((np.asarray(p.v) & 15) != 0, "i32")})
    return any(E.ev(c, env).scalar() for c in model.launch_checks)


# v1.2 (R-4, L2(a)): G6 converter faults on the records the emulated kernel wrote. The first two are the value-neutral
# quirks the amended rule accepts (deterministic); the rest it must still catch.
G6_L2A_FAULTS: Tuple[str, ...] = ("g6 +0 zero codes", "g6 -0 zero codes", "g6 random zero signs",
                                  "g6 nonzero sign flip", "g6 zero-block e8m0", "g6 0x20 in a byte-255 block")


def g6_fault(rec: np.ndarray, fault: str, launch: int = 0) -> np.ndarray:
    """`rec` (uint8 [M, 400]) as a faulty G6 converter would write it:
      g6 +0 zero codes             every zero-magnitude code of a finite block written as 0x00 (REF writes -0 as 0x20)
      g6 -0 zero codes             ... written as 0x20
      g6 random zero signs         ... with a sign drawn per launch (the two poisoned runs differ: the twin's job)
      g6 nonzero sign flip         the first nonzero code of a finite block with its sign bit flipped
      g6 zero-block e8m0           the first all-zero finite block's E8M0 byte 127 written as 126
      g6 0x20 in a byte-255 block  code 0 of the first byte-255 block written as 0x20 (REF: 24 zero bytes)"""
    codes, byte = odm.g6_codes(rec).copy(), odm.g6_block_bytes(rec).copy()
    fin = (byte != 255)[..., None]
    zero = (codes & 31) == 0
    if fault == "g6 +0 zero codes":
        codes = np.where(zero & fin, 0, codes)
    elif fault == "g6 -0 zero codes":
        codes = np.where(zero & fin, 0x20, codes)
    elif fault == "g6 random zero signs":
        sign = np.random.default_rng(launch).integers(0, 2, size=codes.shape) << 5
        codes = np.where(zero & fin, sign, codes)
    elif fault == "g6 nonzero sign flip":
        r, b, f = np.argwhere(~zero & fin)[0]
        codes[r, b, f] ^= 0x20
    elif fault == "g6 zero-block e8m0":
        r, b = np.argwhere((byte == 127) & zero.all(-1))[0]
        byte[r, b] = 126
    elif fault == "g6 0x20 in a byte-255 block":
        r, b = np.argwhere(byte == 255)[0]
        codes[r, b, 0] = 0x20
    else:
        raise ValueError(fault)
    return odm._assemble_g6(codes, byte)


def c_w6_reject(model: E.CuModel, d: Dict[str, Any]) -> bool:
    env: Dict[str, Any] = {k: E.CV(np.int64(d.get(k) or 0), "i64") for k in ("w", "w6", "w6s")}
    env.update({"nullptr": E.CV(np.int64(0), "i64"), "odm6_mis16": lambda p: E.CV((np.asarray(p.v) & 15) != 0, "i32")})
    return any(E.ev(c, env).scalar() for c in model.w6_launch_checks)


class EmuDevice:
    """The two C launchers of the .so, run on the CPU: the .cu's argument checks (parsed), then the lane emulator on the
    host memory behind the pointers (CPU tensors); exactly the bytes the emulated kernel wrote are stored (guards
    included: the self-check's guard regions are at least the emulator's). fault: an injected device fault."""

    T_OF_BITS = {0x3D800000: 4, 0x3D000000: 5, 0x3C800000: 6, 0x3C000000: 7}

    def __init__(self, emu: E.Emulator, fault: Optional[str] = None) -> None:
        self.emu, self.model, self.fault = emu, emu.m, fault
        self.launches = 0

    @staticmethod
    def _view(ptr: int, n: int) -> np.ndarray:
        return np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(int(ptr)))

    def _put(self, ptr: int, data: np.ndarray, cnt: np.ndarray) -> None:
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        cnt = np.asarray(cnt).reshape(-1)
        w = np.nonzero(cnt > 0)[0]
        if w.size:
            dst = self._view(ptr, int(w.max()) + 1)
            dst[w] = data[w]

    def launch(self, dout: int, inv: int, t: float, M: int, M_PAD: int, g6: int, dout_t: int, partial: int,
               inv_used: Optional[int], stream: Any) -> int:
        d = dict(dout=dout, inv=inv, g6=g6, dout_t=dout_t, partial=partial, inv_used=inv_used, M=M, M_PAD=M_PAD, t=t)
        if c_launch_reject(self.model, d):
            return 1
        self.launches += 1
        bits = self._view(dout, M * 1024).copy().view(np.uint16).reshape(M, 512)
        inv_bits = int(self._view(inv, 4).copy().view(np.uint32)[0])
        r = self.emu.run(bits, np.uint32(inv_bits), self.T_OF_BITS[int(np.float32(t).view(np.uint32))], m_pad=M_PAD)
        if self.fault == "nondeterministic" and self.launches % 2 == 0:
            r["dt"][5] ^= 1
        if self.fault == "g6 overrun":
            r["g6c"][M * 400] = 1
            r["g6"][M * 400] = 0
        if self.fault in G6_L2A_FAULTS:
            r["g6"][:M * 400] = g6_fault(r["g6"][:M * 400].reshape(M, 400), self.fault, self.launches).reshape(-1)
        self._put(g6, r["g6"], r["g6c"])
        self._put(dout_t, r["dt"], r["dtc"])
        self._put(partial, r["part"], np.repeat(r["pc"], 4))
        if inv_used is not None and self.fault != "no INV_USED":
            self._put(inv_used, r["iu"], np.repeat(r["iuc"], 4))
        return 0

    def w6(self, w: int, w_is_bf16: int, w6: int, w6s: int, stream: Any) -> int:
        if c_w6_reject(self.model, dict(w=w, w6=w6, w6s=w6s)):
            return 1
        raw = self._view(w, 1536 * 512 * (2 if w_is_bf16 else 4)).copy()
        arr = raw.view(np.uint16 if w_is_bf16 else np.float32).reshape(1536, 512)
        r = self.emu.run_w6(arr)
        if self.fault == "w6s stale":
            r["w6s"][7] ^= 0x10
        self._put(w6, r["w6"], r["w6c"])
        self._put(w6s, r["w6s"], r["w6sc"])
        return 0

    def lib(self) -> Any:
        return mock.Mock(fn={"outproj_dual_mx6_launch": self.launch, "outproj_w6_shadow_launch": self.w6})


def cpu_dual_reference(dout: torch.Tensor, inv_scale: torch.Tensor, t_shift: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """odm.triton_dual_reference on the CPU: the U2 model of the shipped Triton dual cast (dout_t, PARTIAL)."""
    bits = dout.detach().contiguous().view(torch.int16).numpy().view(np.uint16)
    inv = np.uint32(inv_scale.detach().view(torch.int32).numpy().view(np.uint32)[0])
    r = E.dual_cast(bits, inv, int(t_shift), True)
    return torch.from_numpy(r["dout_t"].copy()), torch.from_numpy(r["partial_bits"].view(np.float32).copy())


class _Env(unittest.TestCase):
    """Isolation for loader tests: a private cache root and temp dir, no dev overrides, a fresh latch."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="odm6_test_"))
        os.chmod(self.tmp, 0o700)
        env = {k: v for k, v in os.environ.items() if not k.startswith("DLRM_HSTU_FP8_OUTPROJ_MX6")}
        env[odm.CACHE_ENV] = str(self.tmp / "cache")
        self._patches = [mock.patch.dict(os.environ, env, clear=True),
                         mock.patch.object(tempfile, "tempdir", str(self.tmp))]
        for p in self._patches:
            p.start()
        odm.reset_for_testing(unload=True)

    def tearDown(self) -> None:
        odm.reset_for_testing(unload=True)
        for p in reversed(self._patches):
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


FAKE_HIPCC = """#!/bin/bash
if [ "$1" = "--version" ]; then echo "HIP version: fake-$ODM6_FAKE_TAG"; echo "fake clang"; exit 0; fi
echo "compile $PWD" >> "$ODM6_FAKE_LOG"
out=""; prev=""
for a in "$@"; do if [ "$prev" = "-o" ]; then out="$a"; fi; prev="$a"; done
case "$ODM6_FAKE_MODE" in
  fail) echo "fake hipcc: error: boom"; exit 1 ;;
  hang) sleep 300 >/dev/null 2>&1 & echo $! > "$ODM6_FAKE_CHILD"; sleep 300; exit 0 ;;
esac
cat "$ODM6_FAKE_REMARKS"
printf '\\t.size\\t%s, .Lfunc_end0-%s\\n\\t; codeLenInByte = 21412\\n' "$ODM6_FAKE_PRODUCER" "$ODM6_FAKE_PRODUCER" \\
  > outproj_dual_mx6-hip-amdgcn-amd-amdhsa-gfx950.s
printf 'fake .so' > "$out"
exit 0
"""


class _FakeHipcc(_Env):
    def setUp(self) -> None:
        super().setUp()
        bindir = self.tmp / "rocm" / "bin"
        bindir.mkdir(parents=True)
        self.hipcc = bindir / "hipcc"
        self.hipcc.write_text(FAKE_HIPCC)
        os.chmod(self.hipcc, 0o755)
        self.log = self.tmp / "hipcc.log"
        self.remarks = self.tmp / "remarks.txt"
        self.remarks.write_text(remark_log(CENSUS, "B"))
        os.environ.update({odm.HIPCC_ENV: str(self.hipcc), "ODM6_FAKE_LOG": str(self.log),
                           "ODM6_FAKE_REMARKS": str(self.remarks), "ODM6_FAKE_MODE": "ok",
                           "ODM6_FAKE_TAG": self.id(), "ODM6_FAKE_CHILD": str(self.tmp / "child.pid"),
                           "ODM6_FAKE_PRODUCER": odm.KERNELS["producer"]})
        odm._l11()._toolchains.pop(str(self.hipcc), None)

    def tearDown(self) -> None:
        odm._l11()._toolchains.pop(str(self.hipcc), None)
        super().tearDown()

    def compiles(self) -> int:
        return len(self.log.read_text().splitlines()) if self.log.exists() else 0


# ===================================================================================================================== T11
CHILD_IMPORT = r'''
import sys, os, json, io, contextlib
sys.dont_write_bytecode = True
import ctypes, subprocess, socket, tempfile, shlex, signal, weakref, numpy, torch
import generative_recommenders.ops.utils
import generative_recommenders.ops.hip
events = []
WATCH = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
         "ctypes.dlopen", "os.mkdir", "os.rmdir", "os.remove", "os.rename", "os.chmod", "shutil.rmtree",
         "socket.connect", "os.symlink", "os.truncate")
def hook(ev, args):
    if ev in WATCH:
        events.append([ev, repr(args)[:300]])
    elif ev == "open":
        a = list(args) + [None, None, None]
        mode, flags = a[1], a[2]
        w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
        if w:
            events.append([ev, repr(args)[:300]])
before = set(sys.modules)
sys.addaudithook(hook)
o, e = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
    import generative_recommenders.ops.hip.outproj_dual_mx6 as odm
new = sorted(m for m in set(sys.modules) - before)
res = dict(events=events, stdout=o.getvalue(), stderr=e.getvalue(), new_modules=new,
           cuda_initialized=bool(torch.cuda.is_initialized()), lib_none=odm._lib is None, prebuilt=odm._prebuilt,
           failure=odm._failure, not_armed=odm.not_armed_reason())
sys.stdout.write("RESULT " + json.dumps(res) + "\n")
'''


class T11LoaderCpuTest(_Env):
    def test_import_is_side_effect_free(self) -> None:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
        proc = subprocess.run([sys.executable, "-B", "-c", CHILD_IMPORT], env=env, capture_output=True, text=True,
                              timeout=300, cwd=str(self.tmp))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        res = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1][7:])
        report("T11 import", **res)
        self.assertEqual(res["events"], [], "importing the module spawned, dlopened, created or wrote something")
        self.assertEqual((res["stdout"], res["stderr"]), ("", ""))
        self.assertFalse(res["cuda_initialized"])
        self.assertTrue(res["lib_none"])
        self.assertFalse(res["prebuilt"])
        self.assertIsNone(res["failure"])
        self.assertTrue(res["not_armed"].startswith("not prebuilt in this process"), res["not_armed"])
        lazy = ("l11_outproj_lnmd", "ec_dgrad_x2h", "bwd_fp8", "l11_do8", "fp8_tm_alloc", "fp8_helpers_fused")
        self.assertEqual([m for m in res["new_modules"] if m.split(".")[-1] in lazy], [])
        self.assertEqual([m for m in res["new_modules"] if m.startswith("generative_recommenders")],
                         ["generative_recommenders.ops.hip.outproj_dual_mx6"])

    def test_not_armed_reason_and_set_prebuilt(self) -> None:
        self.assertEqual(odm.not_armed_reason(), "not prebuilt in this process (outproj_dual_mx6.prebuild() runs "
                                                 "before RUN_START; nothing is built after it)")
        self.assertIsNone(odm.failure())
        odm.set_prebuilt_for_testing(True)
        self.assertIsNone(odm.not_armed_reason())
        odm._failure = "x"
        self.assertEqual(odm.not_armed_reason(), "DISABLED in this process (x)")

    @unittest.skipIf(torch.cuda.is_available(), "this host has a GPU: the no-GPU latch is not reachable")
    def test_prebuild_fails_closed_and_latches_without_gpu(self) -> None:
        with mock.patch.object(odm, "build", side_effect=AssertionError("built")) as build, \
                mock.patch.object(odm.subprocess, "Popen", side_effect=AssertionError("spawned")), Lines() as cap:
            with self.assertRaises(odm.OutprojMx6Error) as ctx:
                odm.prebuild()
            first = cap.msgs[:]
            with self.assertRaises(odm.OutprojMx6Error) as ctx2:
                odm.prebuild()
            with self.assertRaises(odm.OutprojMx6Error):
                odm.launch(*launch_operands(256))
            with self.assertRaises(odm.OutprojMx6Error):
                odm.w6_shadow(fake((1536, 512), torch.float32, 1 << 30))
            with self.assertRaises(odm.OutprojMx6Error):
                odm.w6_encode(fake((1536, 512), torch.float32, 1 << 30))
        build.assert_not_called()
        self.assertEqual(len(first), 1, first)
        self.assertEqual(cap.msgs, first, "a second prebuild() printed again")
        self.assertTrue(first[0].startswith("[outproj-mx6] DISABLED: prebuild build/load: RuntimeError: no ROCm GPU "
                                            "in this process"), first[0])
        self.assertTrue(first[0].endswith(" -> FP8 dual cast for this process"), first[0])
        self.assertEqual(len(cap.printed()), 1)
        self.assertRegex(cap.printed()[0], r"^" + re.escape(first[0]) + r" \[rank \d+\]$")
        reason = odm.failure()
        self.assertTrue(reason.startswith("prebuild build/load: RuntimeError: no ROCm GPU"), reason)
        self.assertEqual(str(ctx.exception), str(ctx2.exception))
        self.assertEqual(str(ctx.exception), f"[outproj-mx6] unavailable: {reason}")
        self.assertEqual(odm.not_armed_reason(), f"DISABLED in this process ({reason})")
        self.assertFalse(odm.prebuilt())

    def _gpu(self) -> contextlib.ExitStack:
        st = contextlib.ExitStack()
        st.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
        st.enter_context(mock.patch.object(torch.version, "hip", "7.2"))
        st.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=0))
        st.enter_context(mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()))
        st.enter_context(mock.patch.object(torch.cuda, "synchronize", return_value=None))
        st.enter_context(mock.patch.object(odm._l11(), "device_arch", return_value="gfx950"))
        return st

    def test_prebuild_no_toolchain_latches(self) -> None:
        os.environ[odm.HIPCC_ENV] = str(self.tmp / "nonexistent" / "hipcc")
        with self._gpu(), Lines() as cap:
            with self.assertRaises(odm.OutprojMx6Error):
                odm.prebuild()
            with self.assertRaises(odm.OutprojMx6Error):
                odm.prebuild()
        self.assertEqual(len(cap.msgs), 1, cap.msgs)
        self.assertIn("DISABLED: prebuild build/load: FileNotFoundError", cap.msgs[0])
        self.assertIsNone(odm._lib)

    def test_prebuild_wrong_arch_latches(self) -> None:
        with self._gpu(), mock.patch.object(odm._l11(), "device_arch", return_value="gfx942"), Lines() as cap, \
                mock.patch.object(odm, "build", side_effect=AssertionError("built")):
            with self.assertRaises(odm.OutprojMx6Error):
                odm.prebuild()
        self.assertEqual(len(cap.msgs), 1)
        self.assertIn("arch gfx942 is not gfx950", cap.msgs[0])

    def test_fail_closed_env_raises_without_latch(self) -> None:
        os.environ[odm.FAIL_CLOSED_ENV] = "1"
        with Lines() as cap:
            with self.assertRaises(RuntimeError) as ctx:
                odm.prebuild()
        self.assertNotIsInstance(ctx.exception, odm.OutprojMx6Error)
        self.assertIn(f"failed with {odm.FAIL_CLOSED_ENV}=1", str(ctx.exception))
        self.assertIsNone(odm.failure())
        self.assertEqual(cap.msgs, [])

    def _fake_lib(self) -> Any:
        side = {"key": "0123456789abcdef", "src_path": str(odm._SOURCE), "src_sha256": odm.SOURCE_SHA256,
                "build_s": 1.0, "summary": {k: dict(v) for k, v in GATE_PINNED.items()},
                "code_bytes": {odm.KERNELS["producer"]: 21412}}
        return mock.Mock(side=side, impl=odm.IMPL + "; src=" + odm.SOURCE_SHA256, variant=40, cached=False,
                         path=self.tmp / "lib.so")

    def test_prebuild_self_check_outcomes(self) -> None:
        lib = self._fake_lib()
        # (a) pass: one armed line, prebuilt, idempotent
        with self._gpu(), mock.patch.object(odm, "_load", return_value=lib), \
                mock.patch.object(odm, "_lib", lib), \
                mock.patch.object(odm, "self_check", return_value=(True, "SC detail")) as sc, Lines() as cap:
            odm.prebuild()
            odm.prebuild()
        self.assertEqual(sc.call_count, 1)
        self.assertTrue(odm.prebuilt())
        self.assertIsNone(odm.not_armed_reason())
        self.assertEqual(len(cap.msgs), 1, cap.msgs)
        armed = cap.msgs[0]
        self.assertTrue(armed.startswith("[outproj-mx6] armed OK: prebuild before run_start: built+loaded+self-check "
                                         "OK "), armed)
        for tok in ("(VGPR 103, SGPR 38, spill 0/0, scratch 0, LDS 67712, occupancy 4, code 21412 B;",
                    "W6 f32/bf16 VGPR 102/102", "so 0123456789abcdef built 1.0s", f"impl={odm.IMPL}; src=",
                    "variant=40", "defines=none", f"@{odm.SOURCE_SHA256[:12]};", "self-check SC detail)"):
            self.assertIn(tok, armed)
        assert_line_rules(self, cap.msgs)
        # (b) mismatch: latch, one DISABLED line, OutprojMx6Error
        odm.reset_for_testing(unload=True)
        with self._gpu(), mock.patch.object(odm, "_load", return_value=lib), \
                mock.patch.object(odm, "self_check", return_value=(False, "G6 3 mismatches")), Lines() as cap:
            with self.assertRaises(odm.OutprojMx6Error):
                odm.prebuild()
        self.assertEqual(cap.msgs, ["[outproj-mx6] DISABLED: prebuild self-check mismatch: G6 3 mismatches -> FP8 "
                                    "dual cast for this process"])
        # (c) a non-device exception: latch
        odm.reset_for_testing(unload=True)
        with self._gpu(), mock.patch.object(odm, "_load", return_value=lib), \
                mock.patch.object(odm, "self_check", side_effect=ValueError("bad shape")), Lines() as cap:
            with self.assertRaises(odm.OutprojMx6Error):
                odm.prebuild()
        self.assertEqual(len(cap.msgs), 1)
        self.assertIn("DISABLED: prebuild self-check: ValueError: bad shape", cap.msgs[0])
        # (d) a device fault: re-raised unchanged after one line; the latch is set (no retry later)
        odm.reset_for_testing(unload=True)
        err = RuntimeError("HIP error: hipErrorIllegalAddress: memory access fault (DEVICE FAULT at 0x0)")
        with self._gpu(), mock.patch.object(odm, "_load", return_value=lib), \
                mock.patch.object(odm, "self_check", side_effect=err), Lines() as cap:
            with self.assertRaises(RuntimeError) as ctx:
                odm.prebuild()
        self.assertIs(ctx.exception, err)
        self.assertEqual(len(cap.msgs), 1, cap.msgs)
        self.assertTrue(cap.msgs[0].startswith("[outproj-mx6] prebuild self-check hit a device-side error; re-raised "
                                               "to abort before RUN_START: "), cap.msgs[0])
        assert_line_rules(self, cap.msgs)
        self.assertIsNotNone(odm.failure())
        with self.assertRaises(odm.OutprojMx6Error):
            odm.prebuild()

    def test_prebuild_self_check_rc_700_719_reraise_unchanged(self) -> None:
        """R-3 (v1.2; r16 iteration 59 R3_R5_R6.md sec. 1.5): a launcher rc 700 / 719 inside r14's self-check is an
        OutprojMx6Error from _check_rc whose "<type>: <text>" matches l11_outproj_lnmd._DEVICE_FAULT, so prebuild()
        re-raises that ORIGINAL exception after the device-side line (latch set, no DISABLED line). A caller of the
        optional hook (bwd_fp8.prebuild_outproj_mx6) must therefore classify by that text, never by isinstance."""
        lib = self._fake_lib()
        for rc, name in ((700, "hipErrorIllegalAddress"), (719, "hipErrorLaunchFailure")):
            with self.subTest(rc=rc):
                odm.reset_for_testing(unload=True)
                try:
                    odm._check_rc(rc, "outproj_dual_mx6_launch", 769)
                except odm.OutprojMx6Error as error:
                    err = error
                self.assertIn(f"rc={rc} ({name})", str(err))
                self.assertTrue(odm._device_fault(f"{type(err).__name__}: {err}"))
                with self._gpu(), mock.patch.object(odm, "_load", return_value=lib), \
                        mock.patch.object(odm, "self_check", side_effect=err), Lines() as cap:
                    with self.assertRaises(odm.OutprojMx6Error) as ctx:
                        odm.prebuild()
                self.assertIs(ctx.exception, err)  # the original object, not a latched "unavailable" error
                self.assertEqual(len(cap.msgs), 1, cap.msgs)
                self.assertTrue(cap.msgs[0].startswith("[outproj-mx6] prebuild self-check hit a device-side error; "
                                                       "re-raised to abort before RUN_START: OutprojMx6Error: "),
                                cap.msgs[0])
                self.assertNotIn("DISABLED", " ".join(cap.msgs))
                assert_line_rules(self, cap.msgs)
                self.assertIsNotNone(odm.failure())

    def test_prebuild_self_check_other_device_codes_latch_today(self) -> None:
        """v1.3 (test-only; the v1.2 test id is kept, although these rows no longer latch): r16's C3-2 v2 row
        (ROW_c3_2_device_fault_v2.patch, sha256 50ce60095a9036ed6b81ff4845c016a6207876e470ac5394eb6bcb9203914ff7)
        adds the five device-fatal hipError names to l11_outproj_lnmd._DEVICE_FAULT, so the loader's _check_rc texts
        for rc 53, 214, 702, 709 and 710 (hipErrorPriorLaunchFailure, hipErrorECCNotCorrectable, hipErrorLaunchTimeOut,
        hipErrorContextIsDestroyed, hipErrorAssert) now match. Inside r14's self-check they re-raise the ORIGINAL
        exception after the device-side line (latch set, no DISABLED line), exactly as rc 700 / 719 do, and r16's
        self-check try aborts before RUN_START. v1.2 pinned the latch of rc 53 / 214 / 702 / 710 (the documented gap:
        the word-bounded ECC term cannot match inside the CamelCase name); rc 709 was not pinned."""
        lib = self._fake_lib()
        for rc, name in ((53, "hipErrorPriorLaunchFailure"), (214, "hipErrorECCNotCorrectable"),
                         (702, "hipErrorLaunchTimeOut"), (709, "hipErrorContextIsDestroyed"),
                         (710, "hipErrorAssert")):
            with self.subTest(rc=rc):
                odm.reset_for_testing(unload=True)
                try:
                    odm._check_rc(rc, "outproj_dual_mx6_launch", 769)
                except odm.OutprojMx6Error as error:
                    err = error
                self.assertIn(f"rc={rc} ({name})", str(err))
                self.assertTrue(odm._device_fault(f"{type(err).__name__}: {err}"))
                with self._gpu(), mock.patch.object(odm, "_load", return_value=lib), \
                        mock.patch.object(odm, "self_check", side_effect=err), Lines() as cap:
                    with self.assertRaises(odm.OutprojMx6Error) as ctx:
                        odm.prebuild()
                self.assertIs(ctx.exception, err)  # the original object, not a latched "unavailable" error
                self.assertEqual(len(cap.msgs), 1, cap.msgs)
                self.assertTrue(cap.msgs[0].startswith("[outproj-mx6] prebuild self-check hit a device-side error; "
                                                       "re-raised to abort before RUN_START: OutprojMx6Error: "),
                                cap.msgs[0])
                self.assertNotIn("DISABLED", " ".join(cap.msgs))
                assert_line_rules(self, cap.msgs)
                self.assertIsNotNone(odm.failure())

    def test_cache_key_components(self) -> None:
        src = odm.source_path()
        with mock.patch.object(odm, "_toolchain", return_value="hipcc A"), \
                mock.patch.object(odm, "hipcc_path", return_value="/opt/rocm/bin/hipcc"):
            k0 = odm.cache_key(src)
            self.assertRegex(k0, r"^[0-9a-f]{16}$")
            self.assertEqual(k0, odm.cache_key(src))
            keys = {"base": k0}
            for var in odm._HIPCC_ENV_KEYS:
                with mock.patch.dict(os.environ, {var: "-O1"}):
                    keys[var] = odm.cache_key(src)
            with mock.patch.dict(os.environ, {odm.DEFINES_ENV: "-DOUTPROJ_MX6_PF=1"}):
                keys["defines"] = odm.cache_key(src)
            edited = self.tmp / "edited.cu"
            edited.write_bytes(src.read_bytes() + b"\n// edit\n")
            keys["source"] = odm.cache_key(edited)
            fl = odm.flags(src.read_bytes())
        with mock.patch.object(odm, "_toolchain", return_value="hipcc B"), \
                mock.patch.object(odm, "hipcc_path", return_value="/opt/rocm/bin/hipcc"):
            keys["toolchain"] = odm.cache_key(src)
        with mock.patch.object(odm, "_toolchain", return_value="hipcc A"), \
                mock.patch.object(odm, "hipcc_path", return_value="/other/rocm/bin/hipcc"):
            keys["hipcc path (include dir)"] = odm.cache_key(src)
        self.assertEqual(len(set(keys.values())), len(keys), keys)
        self.assertEqual(fl, ["--offload-arch=gfx950", "-std=c++20", "-w", "-O3", "-ffp-contract=off",
                              "-fno-slp-vectorize", "-I/opt/rocm/include/hip", "-shared", "-fPIC",
                              "-Rpass-analysis=kernel-resource-usage", "--save-temps",
                              f'-DOUTPROJ_MX6_SRC_SHA256="{hashlib.sha256(src.read_bytes()).hexdigest()}"'])
        # the key hashes exactly: source, "\0".join(flags), arch, toolchain, the HIPCC_* / HIP_* env (DESIGN 12.3)
        h = hashlib.sha256()
        h.update(src.read_bytes())
        h.update(b"\0" + "\0".join(fl).encode())
        h.update(b"\0gfx950")
        h.update(b"\0" + b"hipcc A")
        h.update(b"\0" + "\0".join(f"{k}=" for k in odm._HIPCC_ENV_KEYS).encode())
        self.assertEqual(k0, h.hexdigest()[:16])

    def test_dev_defines(self) -> None:
        good = "-DOUTPROJ_MX6_PF=1 -mllvm -amdgpu-sched-strategy=iterative-ilp -DOUTPROJ_MX6_REVERSE=1 -DOUTPROJ_MX6_DN1"
        with mock.patch.dict(os.environ, {odm.DEFINES_ENV: good}):
            self.assertEqual(odm.dev_defines(), ["-DOUTPROJ_MX6_PF=1", "-mllvm", "-amdgpu-sched-strategy=iterative-ilp",
                                                 "-DOUTPROJ_MX6_REVERSE=1", "-DOUTPROJ_MX6_DN1"])
            self.assertEqual(odm.flags(b"x")[-5:], odm.dev_defines())
        for bad in ("-DFOO=1", "-O0", "-DOUTPROJ_MX6_SRC_SHA256=x", "-mllvm -foo", "-DOUTPROJ_MX6_PF=1;rm",
                    "-mllvm", "-UOUTPROJ_MX6_PF", "-DOUTPROJ_MX6_PF=$(x)"):
            with mock.patch.dict(os.environ, {odm.DEFINES_ENV: bad}):
                with self.assertRaises(ValueError, msg=bad):
                    odm.dev_defines()
        with mock.patch.dict(os.environ, {odm.DEFINES_ENV: "  "}):
            self.assertEqual(odm.dev_defines(), [])

    def test_hipcc_path_rule(self) -> None:
        with mock.patch.dict(os.environ, {"ROCM_PATH": ""}):
            os.environ.pop(odm.HIPCC_ENV, None)
            self.assertEqual(odm.hipcc_path(), "/opt/rocm/bin/hipcc")
        fake_rocm = self.tmp / "r"
        (fake_rocm / "bin").mkdir(parents=True)
        (fake_rocm / "bin" / "hipcc").write_text("#!/bin/sh\n")
        with mock.patch.dict(os.environ, {"ROCM_PATH": str(fake_rocm)}):
            self.assertEqual(odm.hipcc_path(), str(fake_rocm / "bin" / "hipcc"))
            with mock.patch.dict(os.environ, {odm.HIPCC_ENV: "/x/y/hipcc"}):
                self.assertEqual(odm.hipcc_path(), "/x/y/hipcc")

    def test_entry_dir_trust_and_private_fallback(self) -> None:
        root = Path(os.environ[odm.CACHE_ENV])
        d, shared = odm._entry_dir("k1")
        self.assertTrue(shared)
        self.assertEqual(d, root / "k1")
        self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(d.stat().st_mode), 0o700)
        for mode in (0o770, 0o707, 0o777, 0o720):
            os.chmod(root, mode)
            d2, shared2 = odm._entry_dir("k1")
            self.assertFalse(shared2, oct(mode))
            self.assertTrue(d2.name.startswith("outproj_mx6_private_"), d2)
            self.assertEqual(d2.parent, self.tmp)
        os.chmod(root, 0o700)
        os.chmod(root / "k1", 0o775)
        self.assertFalse(odm._entry_dir("k1")[1])
        os.chmod(root / "k1", 0o700)
        with mock.patch.object(odm, "_trusted", return_value=False):  # a root / entry owned by someone else
            self.assertFalse(odm._entry_dir("k1")[1])
        blocker = self.tmp / "file_root"
        blocker.write_text("x")
        with mock.patch.dict(os.environ, {odm.CACHE_ENV: str(blocker)}):  # mkdir fails (OSError): private
            d3, shared3 = odm._entry_dir("k2")
            self.assertFalse(shared3)
            self.assertTrue(d3.is_dir())
        x2h = __import__("generative_recommenders.ops.hip.ec_dgrad_x2h", fromlist=["x"])
        p = self.tmp / "t"
        p.mkdir()
        for mode, want in ((0o700, True), (0o755, True), (0o775, False), (0o757, False)):
            os.chmod(p, mode)
            self.assertEqual(x2h._trusted(p), want, oct(mode))
            self.assertEqual(odm._trusted(p), want, oct(mode))


class T11BuildTest(_FakeHipcc):
    def test_build_install_naming_cache_hit_and_gate_names(self) -> None:
        replaces: List[Tuple[str, str]] = []
        real_replace = os.replace

        def rec_replace(a: Any, b: Any) -> None:
            replaces.append((str(a), str(b)))
            real_replace(a, b)

        gate_args: List[Any] = []
        real_gate = odm.check_resources

        def rec_gate(k: Any) -> Any:
            gate_args.append(sorted(k))
            return real_gate(k)

        with mock.patch.object(odm.os, "replace", side_effect=rec_replace), \
                mock.patch.object(odm, "check_resources", side_effect=rec_gate):
            so, side, hit = odm.build()
        self.assertFalse(hit)
        entry = Path(os.environ[odm.CACHE_ENV]) / odm.cache_key()
        self.assertEqual(so, entry / "liboutproj_dual_mx6.so")
        self.assertEqual(sorted(p.name for p in entry.iterdir()),
                         ["liboutproj_dual_mx6.so", "liboutproj_dual_mx6.so.resources.json"])
        tag = f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
        self.assertEqual(replaces, [(str(entry / f"{tag}-liboutproj_dual_mx6.so.resources.json"),
                                     str(entry / "liboutproj_dual_mx6.so.resources.json")),
                                    (str(entry / f"{tag}-liboutproj_dual_mx6.so"), str(so))])
        self.assertEqual(gate_args, [sorted(odm.KERNELS.values())])
        self.assertEqual(side["summary"], GATE_PINNED)
        self.assertEqual(side["code_bytes"], {odm.KERNELS["producer"]: 21412})
        self.assertEqual(side["remark_format"], {"A": 0, "B": 30})
        self.assertEqual(side["src_sha256"], odm.SOURCE_SHA256)
        self.assertEqual(side["flags"], odm.flags(odm.source_path().read_bytes()))
        on_disk = json.loads((entry / "liboutproj_dual_mx6.so.resources.json").read_text())
        self.assertEqual(on_disk["kernels"], odm.parse_remarks(remark_log(CENSUS, "B")))
        work = [ln.split()[1] for ln in self.log.read_text().splitlines()]
        self.assertEqual(len(work), 1)
        self.assertTrue(Path(work[0]).name.startswith(tag + "-build-"))
        self.assertFalse(Path(work[0]).exists(), "the private build cwd was not removed")
        # cache hit: no compile, same path
        so2, side2, hit2 = odm.build()
        self.assertTrue(hit2)
        self.assertEqual((so2, self.compiles()), (so, 1))
        # a faulty entry (sidecar census now fails the gate) is deleted and rebuilt once, privately
        meta = entry / "liboutproj_dual_mx6.so.resources.json"
        j = json.loads(meta.read_text())
        j["kernels"][odm.KERNELS["producer"]]["VGPRs Spill"] = "3"
        meta.write_text(json.dumps(j))
        so3, _, hit3 = odm.build()
        self.assertFalse(hit3)
        self.assertEqual(self.compiles(), 2)
        self.assertEqual(list(entry.iterdir()), [])
        self.assertTrue(so3.parent.name.startswith("outproj_mx6_private_"), so3)
        self.assertEqual(sorted(p.name for p in so3.parent.iterdir()),
                         ["liboutproj_dual_mx6.so", "liboutproj_dual_mx6.so.resources.json"])

    def test_build_failures_leave_nothing(self) -> None:
        entry = Path(os.environ[odm.CACHE_ENV]) / odm.cache_key()
        for mode, census, names, pat in (
                ("fail", CENSUS, None, r"hipcc rc=1: .*fake hipcc: error: boom"),
                ("ok", mutate(CENSUS, "producer", **{"VGPRs Spill": "2"}), None, r"producer spill 2/0 scratch 0"),
                ("ok", CENSUS, dict(odm.KERNELS, producer=odm.KERNELS["producer"].replace("PfPj", "PfS5_")),
                 r"unexpected kernel\(s\)"),
                ("ok", {r: CENSUS[r] for r in ("w6_bf16", "w6_f32")}, None, r"no resource remarks for the producer")):
            os.environ["ODM6_FAKE_MODE"] = mode
            roles = tuple(census)
            self.remarks.write_text(remark_log(census, "B", roles=roles, names=names))
            with self.assertRaisesRegex(RuntimeError, pat) as ctx:
                odm.build()
            self.assertEqual(list(entry.iterdir()), [], f"{mode}: leftovers {list(entry.iterdir())}")
            if mode == "fail":  # MINOR-1 (v1.1): the diagnostic first, one line, the full log kept (argv first)
                msg = str(ctx.exception)
                self.assertTrue(msg.startswith("hipcc rc=1: fake hipcc: error: boom (full log: "), msg)
                self.assertNotIn("\n", msg)
                self.assertNotIn("--offload-arch", msg)
                logs = sorted(self.tmp.glob("outproj_mx6_hipcc_fail_*/hipcc.log"))
                self.assertEqual(len(logs), 1, logs)
                self.assertTrue(msg.endswith(f"(full log: {logs[0]})"), msg)
                first, rest = logs[0].read_text().split("\n", 1)
                self.assertEqual(first.split(), odm.build_command(out=Path(first.split()[-1])))
                self.assertEqual(rest, "fake hipcc: error: boom\n")
                shutil.rmtree(logs[0].parent)
        self.assertEqual([p for p in self.tmp.iterdir() if p.name.startswith("outproj_mx6_private_")], [])
        self.assertEqual(sorted(self.tmp.glob("outproj_mx6_hipcc_fail_*")), [])  # a gate failure keeps no log

    def test_hipcc_diagnostic_extraction(self) -> None:
        """MINOR-1 (v1.1): the first error line with clang's source line under it (not the caret line), the count of
        further error lines, else the last three lines; one line, at most _DIAG_CHARS characters."""
        log = ("In file included from x.cu:1:\nx.s:12:2: error: invalid instruction\n        v_bad v1, v2\n        ^\n"
               "x.s:40:2: error: invalid instruction\n        v_bad v3, v4\n        ^\n2 errors generated.\n")
        self.assertEqual(odm._hipcc_diagnostic(log),
                         "x.s:12:2: error: invalid instruction | v_bad v1, v2 (+1 more error lines)")
        self.assertEqual(odm._hipcc_diagnostic("clang: error: unknown argument '-foo'\n"),
                         "clang: error: unknown argument '-foo'")
        self.assertEqual(odm._hipcc_diagnostic("a\n\nb\nc\nd\n"), "b | c | d")
        self.assertEqual(odm._hipcc_diagnostic(""), "(no output)")
        long = odm._hipcc_diagnostic("e.cu:1:1: error: " + "x" * 1000)
        self.assertEqual(len(long), odm._DIAG_CHARS)
        self.assertTrue(long.endswith("..."))

    def test_build_timeout_kills_the_process_group(self) -> None:
        os.environ["ODM6_FAKE_MODE"] = "hang"
        t0 = time.time()
        with mock.patch.object(odm, "_BUILD_TIMEOUT_S", 2):
            with self.assertRaisesRegex(RuntimeError, r"hipcc timed out after 2s \(process group killed\)"):
                odm.build()
        self.assertLess(time.time() - t0, 60)
        pid = int((self.tmp / "child.pid").read_text())
        deadline = time.time() + 10
        alive = True
        while time.time() < deadline:
            try:
                with open(f"/proc/{pid}/stat") as f:
                    alive = f.read().split(")")[-1].split()[0] != "Z"
            except FileNotFoundError:
                alive = False
            if not alive:
                break
            time.sleep(0.1)
        self.assertFalse(alive, f"the hipcc grandchild {pid} survived killpg")
        entry = Path(os.environ[odm.CACHE_ENV]) / odm.cache_key()
        self.assertEqual(list(entry.iterdir()), [])

    def test_untrusted_root_builds_privately(self) -> None:
        root = Path(os.environ[odm.CACHE_ENV])
        root.mkdir(mode=0o700)
        os.chmod(root, 0o777)
        so, _, hit = odm.build()
        self.assertFalse(hit)
        self.assertTrue(so.parent.name.startswith("outproj_mx6_private_"))
        self.assertEqual(list(root.rglob("*.so")), [])

    def test_lib_checks_abi_impl_geometry(self) -> None:
        so = self.tmp / "lib.so"
        so.write_text("x")
        side = {"src_sha256": odm.SOURCE_SHA256}

        class Fn:
            def __init__(self, fn: Any) -> None:
                self.fn, self.restype, self.argtypes = fn, None, None

            def __call__(self, *a: Any) -> Any:
                return self.fn(*a)

        def cdll(abi: int = 1, impl: Optional[str] = None, lds: int = 67712, pc: Any = None, variant: int = 40) -> Any:
            impl = odm.IMPL + "; src=" + odm.SOURCE_SHA256 if impl is None else impl
            ns = mock.Mock(spec=list(odm.SIGNATURES))
            fns = {"outproj_dual_mx6_abi_version": lambda: abi, "outproj_dual_mx6_impl": lambda: impl.encode(),
                   "outproj_dual_mx6_lds_bytes": lambda: lds, "outproj_dual_mx6_variant": lambda: variant,
                   "outproj_dual_mx6_partial_count": pc or (lambda m: 4 * ((m + 255) // 256)),
                   "outproj_dual_mx6_launch": lambda *a: 0, "outproj_w6_shadow_launch": lambda *a: 0,
                   "outproj_dual_mx6_drain_last_error": lambda: 0}
            for k, f in fns.items():
                setattr(ns, k, Fn(f))
            return ns

        with mock.patch.object(odm.ctypes, "CDLL", return_value=cdll()):
            lib = odm._Lib(so, side, True)
        self.assertEqual((lib.variant, odm.decode_variant(lib.variant)),
                         (40, {"G6_F32": 0, "DN1": 0, "REVERSE": 0, "WPE": 1, "PF": 2}))
        for name, (res, args) in odm.SIGNATURES.items():
            self.assertEqual((lib.fn[name].restype, lib.fn[name].argtypes), (res, list(args)), name)
        for kw, pat in ((dict(abi=2), "abi_version"), (dict(impl="other impl; src=" + odm.SOURCE_SHA256), "IMPL"),
                        (dict(impl=odm.IMPL + "; src=" + "0" * 64), "not built from the hashed source"),
                        (dict(lds=67713), "lds_bytes"), (dict(pc=lambda m: 4 * (m // 256)), "partial_count")):
            with mock.patch.object(odm.ctypes, "CDLL", return_value=cdll(**kw)):
                with self.assertRaisesRegex(RuntimeError, pat, msg=str(kw)):
                    odm._Lib(so, side, True)

    def test_load_rebuilds_a_faulty_cached_library_once(self) -> None:
        good = mock.Mock(name="lib")
        calls: List[Any] = []

        def fake_build(source: Any = None, fresh: bool = False) -> Any:
            calls.append(fresh)
            return (self.tmp / ("b.so" if fresh else "a.so"), {"x": 1}, not fresh)

        def fake_lib(so: Path, side: Any, cached: bool) -> Any:
            if so.name == "a.so":
                raise OSError("cannot open shared object")
            return good

        with mock.patch.object(odm, "build", side_effect=fake_build), mock.patch.object(odm, "_Lib",
                                                                                        side_effect=fake_lib):
            self.assertIs(odm._load(), good)
            self.assertIs(odm._load(), good)
        self.assertEqual(calls, [False, True])
        odm.reset_for_testing(unload=True)
        with mock.patch.object(odm, "build", return_value=(self.tmp / "a.so", {}, False)), \
                mock.patch.object(odm, "_Lib", side_effect=OSError("bad")):
            with self.assertRaises(OSError):  # a fresh build that does not load is not retried
                odm._load()


class T11LaunchTest(_Env):
    def test_check_rc(self) -> None:
        with Lines() as cap:
            odm._check_rc(0, "w")
            with self.assertRaisesRegex(odm.OutprojMx6Error, r"w rc=1 \(hipErrorInvalidValue\) M=5: not launched"):
                odm._check_rc(1, "w", 5)
            with self.assertRaisesRegex(odm.OutprojMx6Error, r"rc=7 \(hipError 7\)"):
                odm._check_rc(7, "w")
            odm._check_rc(-700, "launch A")
            odm._check_rc(-700, "launch B")
            odm._check_rc(-719, "launch A")
        self.assertEqual(cap.msgs, [
            "[outproj-mx6] stale HIP error 700 (hipErrorIllegalAddress) drained at launch A (left on this thread by an "
            "earlier call, not ours; launched, result used)",
            "[outproj-mx6] stale HIP error 719 (hipErrorLaunchFailure) drained at launch A (left on this thread by an "
            "earlier call, not ours; launched, result used)"])
        assert_line_rules(self, cap.msgs)

    def test_launch_fail_closed_and_argument_order(self) -> None:
        ops = launch_operands(300)
        with self.assertRaisesRegex(odm.OutprojMx6Error, "not loaded"):
            odm.launch(*ops)
        seen: List[Tuple[Any, ...]] = []
        rcs = [0, 1, -53]

        def c_launch(*a: Any) -> int:
            seen.append(a)
            return rcs[len(seen) - 1]

        lib = mock.Mock(fn={"outproj_dual_mx6_launch": c_launch})
        stream = mock.Mock(cuda_stream=0xABC)
        with mock.patch.object(odm, "_lib", lib), \
                mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()), \
                mock.patch.object(torch.cuda, "current_stream", return_value=stream) as cs, Lines() as cap:
            odm.launch(*ops)
            with self.assertRaises(odm.OutprojMx6Error):
                odm.launch(*ops)
            no_iu = list(ops)
            no_iu[8] = None
            odm.launch(*no_iu)
            bad = list(ops)
            bad[2] = 0.03
            with self.assertRaisesRegex(ValueError, "t_inv_mul"):
                odm.launch(*bad)
        self.assertEqual(len(seen), 3)
        p = [t.data_ptr() if isinstance(t, torch.Tensor) else t for t in ops]
        self.assertEqual(seen[0], (p[0], p[1], 2.0 ** -5, 300, 384, p[5], p[6], p[7], p[8], 0xABC))
        self.assertIsNone(seen[2][8])
        self.assertTrue(all(c.args == (0,) for c in cs.call_args_list))
        self.assertEqual(len(cap.msgs), 1)
        self.assertIn("stale HIP error 53 (hipErrorPriorLaunchFailure)", cap.msgs[0])
        odm._failure = "latched"
        with mock.patch.object(odm, "_lib", lib):
            with self.assertRaisesRegex(odm.OutprojMx6Error, "unavailable: latched"):
                odm.launch(*ops)

    @unittest.skipIf(torch.cuda.is_available(), "GPU host")
    def test_s1e_reaches_prebuild_once(self) -> None:
        # S1-E (bwd_fp8.output_projection_mx6_encode) builds through prebuild(): on this host the first call latches
        # with one line and raises the latched reason; a second call raises the same and prints nothing.
        from generative_recommenders.ops.triton import bwd_fp8 as bf

        enc = getattr(bf, "output_projection_mx6_encode", None)
        if enc is None:
            self.skipTest("bwd_fp8 has no S1-E in this tree")
        dout = torch.zeros((4, 512), dtype=torch.bfloat16)
        w = torch.zeros((1536, 512), dtype=torch.float32)
        with mock.patch.object(odm, "_load", side_effect=AssertionError("loaded")), Lines() as cap:
            with self.assertRaises(odm.OutprojMx6Error) as a:
                enc(dout, w)
            with self.assertRaises(odm.OutprojMx6Error) as b:
                enc(dout, w)
        self.assertEqual(str(a.exception), str(b.exception))
        self.assertEqual(len(cap.msgs), 1, cap.msgs)
        assert_line_rules(self, cap.msgs)


class T11SelfCheckOnEmulatorTest(_Env):
    """odm.self_check() (SC0-SC6) end to end on the CPU: the .so's C launchers are EmuDevice (the .cu's parsed argument
    checks + the lane emulator on host memory), the Triton reference is the U2 model; CUDA stream / device calls are
    stubbed and the CUDA-only metadata checks (ineligible_reason, w6_ineligible_reason) bypassed (T14 covers them).
    Pass: every count 0. Injected device faults must make it fail with the matching count."""

    def _emulated(self, dev: EmuDevice) -> contextlib.ExitStack:
        """EmuDevice as the .so, the U2 model as the Triton reference, CUDA stream / device calls stubbed."""
        st = contextlib.ExitStack()
        for p in (mock.patch.object(odm, "_lib", dev.lib()),
                  mock.patch.object(odm, "ineligible_reason", return_value=None),
                  mock.patch.object(odm, "w6_ineligible_reason", return_value=None),
                  mock.patch.object(odm, "triton_dual_reference", side_effect=cpu_dual_reference),
                  mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()),
                  mock.patch.object(torch.cuda, "current_stream", return_value=mock.Mock(cuda_stream=0)),
                  mock.patch.object(torch.cuda, "synchronize", return_value=None),
                  mock.patch.object(torch.cuda, "current_device", return_value=0)):
            st.enter_context(p)
        return st

    def _self_check(self, dev: EmuDevice) -> Tuple[bool, str]:
        with self._emulated(dev):
            return odm.self_check(torch.device("cpu"))

    def test_self_check_passes_on_the_emulated_kernel(self) -> None:
        dev = EmuDevice(E.Emulator(E.CuModel(E.read_cu())))
        t0 = time.time()
        ok, detail = self._self_check(dev)
        report("T11 self-check on emulator", ok=ok, detail=detail, launches=dev.launches, secs=round(time.time() - t0, 1))
        self.assertTrue(ok, detail)
        self.assertRegex(detail, r"^SC0 16/16 bad launcher args rejected; 1[67] producer cases x 2 poisons ")
        self.assertIn("mismatches G6 0, dout_t 0, PARTIAL 0, INV_USED 0, guards 0, twin 0;", detail)
        self.assertIn("4 W6 cases (edge / hash x fp32 / bf16): W6 0, W6S 0, guards 0, twin 0, alloc 0;", detail)
        self.assertIn("adversarial rows sha ok", detail)
        # v1.2 (R-4): the REF-exact emulator writes no value-neutral code either. The L2(a) report closes the detail,
        # after the timing (v1.2 review: r16 re-logs a failure as msg[:600]); the text before it is v1.1's layout.
        self.assertIn("twin 0; 4 W6 cases", detail)
        self.assertRegex(detail, r"; \d+\.\ds; G6 value-neutral zero-sign codes 0 \(S1C 2\.5 L2\(a\), reported, not "
                                 r"gated\): 0x00-for-0x20 0, 0x20-for-0x00 0$")

    def test_sc_case_g6_follows_l2a(self) -> None:
        """R-4 (v1.2, NIT-3): the in-process G6 comparator of SC1 / SC2 / SC3 / SC6 (_sc_case) follows S1C 2.5 as
        amended (L2(a)), on the emulated kernel with G6 converter faults. A converter that writes every E2M3 zero of a
        finite block with one sign passes; its codes are reported per direction: exactly 2 x the REF zero codes of the
        other sign (two poisoned runs). Zero signs that change between launches fail the byte-strict twin. A nonzero
        code's sign, an all-zero block's E8M0 byte and a 0x20 code in a byte-255 block fail G6."""
        emu = E.Emulator(E.CuModel(E.read_cu()))
        bits = odm.self_check_rows(300)  # +-0 rows, the adversarial rows, byte-255 and all-zero blocks
        ref = odm.encode_g6_int(bits)
        codes, fin = odm.g6_codes(ref), (odm.g6_block_bytes(ref) != 255)[..., None]
        n20, n00 = int(((codes == 0x20) & fin).sum()), int(((codes == 0) & fin).sum())
        self.assertTrue(n20 > 0 and n00 > 0)
        rows = (  # fault -> (g6, g6_n00, g6_n20, twin, ok)
            (None, (0, 0, 0, 0, True)),
            ("g6 +0 zero codes", (0, 2 * n20, 0, 0, True)),
            ("g6 -0 zero codes", (0, 0, 2 * n00, 0, True)),
            ("g6 nonzero sign flip", (2, 0, 0, 0, False)),
            ("g6 zero-block e8m0", (2, 0, 0, 0, False)),
            ("g6 0x20 in a byte-255 block", (2, 0, 0, 0, False)),
        )
        for fault, want in rows:
            with self.subTest(fault):
                with self._emulated(EmuDevice(emu, fault=fault)):
                    res = odm._sc_case(f"SC1 M=300 {fault}", bits, 0x424880E5, 5, torch.device("cpu"))
                self.assertEqual((res["g6"], res["g6_n00"], res["g6_n20"], res["twin"], res["ok"]), want, res)
                self.assertEqual(res["g6_neutral"], res["g6_n00"] + res["g6_n20"])
                self.assertEqual((res["dt"], res["p"], res["iu"], res["guard"]), (0, 0, 0, 0), res)
        with self._emulated(EmuDevice(emu, fault="g6 random zero signs")):
            res = odm._sc_case("SC1 M=300 random zero signs", bits, 0x424880E5, 5, torch.device("cpu"))
        self.assertEqual(res["g6"], 0, res)  # each run passes against REF ...
        self.assertGreater(res["g6_n00"], 0)
        self.assertGreater(res["g6_n20"], 0)
        self.assertGreater(res["twin"], 0)  # ... but determinism stays byte-strict
        self.assertFalse(res["ok"])

    def test_self_check_passes_with_the_zero_sign_quirk(self) -> None:
        """R-4 (v1.2, NIT-3): a converter writing 0x00 for REF's -0 code 0x20 failed SC1's strict G6 compare before
        v1.2 and latched MX6 on every rank. Under L2(a) the whole self-check passes and reports the codes: per case
        exactly 2 x REF's 0x20 codes of finite blocks as 0x00-for-0x20, none the other way, the total in the detail."""
        dev = EmuDevice(E.Emulator(E.CuModel(E.read_cu())), fault="g6 +0 zero codes")
        real = odm._sc_case
        seen: List[Tuple[np.ndarray, Dict[str, Any]]] = []

        def spy(label: str, bits: np.ndarray, *a: Any, **k: Any) -> Dict[str, Any]:
            res = real(label, bits, *a, **k)
            seen.append((bits, res))
            return res

        with mock.patch.object(odm, "_sc_case", spy):
            ok, detail = self._self_check(dev)
        report("T11 self-check zero-sign quirk", ok=ok, detail=detail)
        self.assertTrue(ok, detail)
        self.assertIn("mismatches G6 0, dout_t 0, PARTIAL 0, INV_USED 0, guards 0, twin 0;", detail)
        total = 0
        for bits, res in seen:
            ref = odm.encode_g6_int(bits)
            want = 2 * int(((odm.g6_codes(ref) == 0x20) & (odm.g6_block_bytes(ref) != 255)[..., None]).sum())
            self.assertEqual((res["g6"], res["g6_n00"], res["g6_n20"], res["twin"]), (0, want, 0, 0), res["case"])
            total += want
        self.assertGreater(total, 0)
        self.assertIn(f"G6 value-neutral zero-sign codes {total} (S1C 2.5 L2(a), reported, not gated): 0x00-for-0x20 "
                      f"{total}, 0x20-for-0x00 0 (by class, 0x00-for-0x20/0x20-for-0x00: SC1 ", detail)
        # the report closes the detail (after the timing); the counts before it keep v1.1's layout
        self.assertIn("twin 0; 4 W6 cases", detail)
        tail = detail[detail.index("; G6 value-neutral zero-sign codes "):]
        self.assertRegex(detail[:len(detail) - len(tail)], r"; \d+\.\ds$")
        self.assertTrue(tail.endswith(")") and "W6 cases" not in tail and "FAILING" not in tail, tail)

    def test_failing_detail_fits_r16s_relog(self) -> None:
        """v1.2 review: r16's prebuild_mx6 re-logs a failed self-check as f"{type(error).__name__}: {error}"[:600]
        (l11_do8.prebuild_mx6's self-check try). The mismatch counts and the start of the FAILING list must stay in
        that window, so the L2(a) value-neutral report closes the detail. Before the fix it sat after the counts and,
        with its per-class suffix, pushed FAILING out of the window. Synthetic SC results (a zero-sign quirk in every
        case, both directions, and five failing SC1 cases); no kernel or emulator runs."""
        fail = {"g6": 123456, "dt": 7, "p": 98765, "iu": 0, "guard": 4096, "twin": 87505}
        labels: List[str] = []

        def sc_case(label: str, bits: np.ndarray, inv_bits: int, t: int, device: Any, ref_g6: Any = None,
                    inv_used: bool = True, off16: bool = False) -> Dict[str, Any]:
            labels.append(label)
            bad = label.startswith("SC1")
            res = {"case": label, "M": int(bits.shape[0]), "t": t, "inv": f"{inv_bits:#010x}", "g6_neutral": 195570,
                   "g6_n00": 194336, "g6_n20": 1234}
            res.update({k: (v if bad else 0) for k, v in fail.items()})
            res["ok"] = not bad
            return res

        w6_ok = {"w6": 0, "w6s": 0, "guard": 0, "twin": 0, "alloc": 0, "ok": True}
        with mock.patch.object(odm, "_lib_for_launch", return_value=mock.Mock()), \
                mock.patch.object(odm, "_sc_launcher_args", return_value={"n": 16, "bad": [], "ok": True}), \
                mock.patch.object(odm, "sc1_shifts", return_value=(5, 6, 7)), \
                mock.patch.object(odm, "self_check_rows", side_effect=lambda m: np.zeros((m, 512), np.uint16)), \
                mock.patch.object(odm, "encode_g6_int", side_effect=lambda b: np.zeros((b.shape[0], 400), np.uint8)), \
                mock.patch.object(odm, "_sc_case", side_effect=sc_case), \
                mock.patch.object(odm, "_sc_w6_case", side_effect=lambda label, w: dict(w6_ok, case=label)), \
                mock.patch.object(odm, "edge_weight", return_value=np.zeros((2, 2), np.float32)), \
                mock.patch.object(odm, "hash_weight", return_value=np.zeros((2, 2), np.float32)), \
                mock.patch.object(torch.cuda, "synchronize", return_value=None):
            ok, detail = odm.self_check(torch.device("cpu"))
        self.assertFalse(ok)
        # what r16 logs: _disable_mx6(f"prebuild self-check: {msg[:600]}"), msg = the "<type>: <text>" of r14's error
        msg = f"OutprojMx6Error: {odm.FAMILY} unavailable: prebuild self-check mismatch: {detail}"
        window = msg[:600]
        self.assertIn("producer cases x 2 poisons", window)
        self.assertRegex(window, r"mismatches G6 [1-9]\d*, dout_t [1-9]\d*, PARTIAL [1-9]\d*, INV_USED 0, guards "
                                 r"[1-9]\d*, twin [1-9]\d*; ")
        self.assertIn("; FAILING SC1 M=769 t=5: g6 123456, dt 7, p 98765, guard 4096, twin 87505", window)
        n = len(labels)
        self.assertIn(n, (16, 17))  # SC1 x 5 (6 with a fourth shift), SC2 x 6, SC3 x 4, SC6 x 1
        tail = detail[detail.index("; G6 value-neutral zero-sign codes "):]
        self.assertTrue(tail.startswith(f"; G6 value-neutral zero-sign codes {195570 * n} (S1C 2.5 L2(a), reported, "
                                        f"not gated): 0x00-for-0x20 {194336 * n}, 0x20-for-0x00 {1234 * n} (by class, "
                                        "0x00-for-0x20/0x20-for-0x00: SC1 "), tail)
        self.assertNotIn("FAILING", tail)
        self.assertLess(detail.index("; FAILING "), detail.index("; G6 value-neutral"))
        report("T11 failing detail vs r16's 600-char re-log", r16_msg_len=len(msg), failing_at=msg.index("; FAILING"),
               neutral_at=msg.index("; G6 value-neutral"))

    def test_audit_last_bands_g6_follows_l2a(self) -> None:
        """R-4 (v1.2): the dev audit (D16 / lab P9) compares its last two bands' G6 under SC1's L2(a) rule: the
        zero-sign quirk passes with its codes reported; a nonzero code's sign or an all-zero block's E8M0 byte in the
        audited rows fails; rows before the audited bands are not compared."""
        m, t, r0 = 700, 5, 256  # 3 bands: the audit takes bands 1..2, rows [256, 700)
        bits = odm.self_check_rows(m)
        dout = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
        inv = torch.from_numpy(np.array([0x424880E5], dtype=np.uint32).view(np.float32).copy())
        dout_t, partial = cpu_dual_reference(dout, inv, t)  # the full call's dout_t / PARTIAL
        ref = odm.encode_g6_int(bits)

        def audit(g6: np.ndarray) -> Tuple[bool, str]:
            with mock.patch.object(odm, "triton_dual_reference", side_effect=cpu_dual_reference):
                return odm.audit_last_bands(dout, inv, 2.0 ** -t, m, 768, torch.from_numpy(g6), dout_t, partial)

        ok, detail = audit(ref)
        self.assertTrue(ok, detail)
        self.assertIn(f"M={m} rows [{r0}, {m}) bands 1..2 t={t}: G6 mism 0 (L2(a); value-neutral zero-sign codes 0: "
                      "0x00-for-0x20 0, 0x20-for-0x00 0, not gated), dout_t mism 0, PARTIAL mism 0", detail)
        tail = ref[r0:]
        n = int(((odm.g6_codes(tail) == 0x20) & (odm.g6_block_bytes(tail) != 255)[..., None]).sum())
        self.assertGreater(n, 0)
        ok, detail = audit(g6_fault(ref, "g6 +0 zero codes"))
        self.assertTrue(ok, detail)
        self.assertIn(f"G6 mism 0 (L2(a); value-neutral zero-sign codes {n}: 0x00-for-0x20 {n}, 0x20-for-0x00 0, "
                      "not gated)", detail)
        for fault, rows, want in (("g6 nonzero sign flip", slice(r0, m), False),
                                  ("g6 zero-block e8m0", slice(r0, m), False),
                                  ("g6 nonzero sign flip", slice(0, r0), True)):
            with self.subTest(f"{fault} in rows {rows.start}..{rows.stop}"):
                g = ref.copy()
                g[rows] = g6_fault(ref[rows], fault)
                ok, detail = audit(g)
                self.assertEqual(ok, want, detail)
                if not want:
                    self.assertIn("G6 mism 1 (L2(a)", detail)

    def test_sc1_shifts_cover_every_cast_shift(self) -> None:
        """CI-3 (v1.1): SC1 compares dout_t at every t this process can cast at. AUTO casts at FLOOR..7; a FIXED
        DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT casts at that t only (BF: _SiteScale.t_shift never moves)."""
        import inspect

        from generative_recommenders.ops.triton import bwd_fp8 as bf

        rows = [((True, 5, 5), (5, 6, 7)), ((True, 4, 4), (4, 5, 6, 7)), ((True, 6, 6), (5, 6, 7)),
                ((False, 5, 4), (4, 5, 6, 7)), ((False, 5, 7), (5, 6, 7)), ((False, 4, 6), (4, 5, 6, 7)),
                ((False, 5, 5), (5, 6, 7))]
        for (auto, floor, shift), want in rows:
            with mock.patch.object(bf, "_OUT_WGRAD_DFMT_SHIFT_AUTO", auto), \
                    mock.patch.object(bf, "_OUT_WGRAD_DFMT_SHIFT_FLOOR", floor), \
                    mock.patch.object(bf, "_OUT_WGRAD_DFMT_SHIFT", shift):
                got = odm.sc1_shifts()
            self.assertEqual(got, want, (auto, floor, shift))
            reach = set(range(floor, 8)) if auto else {shift}
            self.assertTrue(reach <= set(got), (auto, floor, shift))
        self.assertIn("shifts = sc1_shifts()", inspect.getsource(odm.self_check))

    @heavy
    def test_heavy_self_check_catches_injected_faults(self) -> None:
        src = E.read_cu()
        muts = {name: payload for name, _, payload in E.MUTANTS}
        rows = [("cu", muts["PARTIAL from the NaN-dominant max"], r"PARTIAL [1-9]"),
                ("cu", muts["no byte-255 zeroing"], r"G6 [1-9]"),
                ("cu", (("const float s0 = inv * x0, s1 = inv * x1, s2_ = inv * x2, s3 = inv * x3;",
                         "const float s0 = inv * x0, s1 = inv * x1, s2_ = inv * x2, s3 = inv * x3; "),),
                 None),  # a whitespace-only edit: still passes
                ("fault", "no INV_USED", r"INV_USED [1-9]"), ("fault", "nondeterministic", r"twin [1-9]"),
                ("fault", "g6 overrun", r"guards [1-9]"), ("fault", "w6s stale", r"W6S [1-9]"),
                # v1.2 (R-4, L2(a)): still caught end to end -- a nonzero code's sign (G6) and per-launch zero signs
                # (the byte-strict twin); the one-sign zero quirk passes (the default-run quirk test)
                ("fault", "g6 nonzero sign flip", r"mismatches G6 [1-9]"),
                ("fault", "g6 random zero signs", r"mismatches G6 0, .* twin [1-9]")]
        for kind, what, pat in rows:
            if kind == "cu":
                dev = EmuDevice(E.Emulator(E.CuModel(E.apply_edits(src, what))))
            else:
                dev = EmuDevice(E.Emulator(E.CuModel(src)), fault=what)
            ok, detail = self._self_check(dev)
            # v1.2 review: r16's prebuild_mx6 re-logs a failed self-check as f"{type}: {error}"[:600]
            r16 = f"OutprojMx6Error: {odm.FAMILY} unavailable: prebuild self-check mismatch: {detail}"
            report("T11 self-check fault", fault=str(what)[:80], ok=ok, detail=detail, r16_msg_len=len(r16),
                   r16_window_failing=r16[:600].split("; FAILING ", 1)[1] if "; FAILING " in r16[:600] else None)
            if pat is None:
                self.assertTrue(ok, detail)
            else:
                self.assertFalse(ok, what)
                self.assertRegex(detail, pat)
                # the counts and the start of the FAILING list stay inside r16's window (the L2(a) report is last)
                self.assertRegex(r16[:600], pat)
                self.assertIn("; FAILING ", r16[:600], what)
                self.assertLess(detail.index("; FAILING "), detail.index("; G6 value-neutral zero-sign codes "))


# ===================================================================================================================== R-4
def _ref_g6_mismatch_l2a(got: np.ndarray, ref: np.ndarray) -> Tuple[int, int]:
    """r16 iteration 59 R4.md sec. 10's numpy reference, verbatim: the oracle of odm.g6_mismatch_l2a."""
    m = got.shape[0]
    sg, sr = got[:, 384:400], ref[:, 384:400]
    scale_bad = int((sg != sr).sum())
    br = sr.reshape(m, 4, 4).transpose(0, 2, 1).reshape(m, 16)    # byte 384 + 4q + k -> block b = 4k + q
    ok_blk = ((br != 255) & (sg.reshape(m, 4, 4).transpose(0, 2, 1).reshape(m, 16) == br))[..., None]

    def codes(g):                                                  # [M, 16, 32] in K order
        grp = np.concatenate([g[:, :256].reshape(m, 16, 16), g[:, 256:384].reshape(m, 16, 8)], -1)
        t = grp.reshape(m, 16, 8, 3).astype(np.int64)              # 4 codes per 3 bytes
        v = t[..., 0] | (t[..., 1] << 8) | (t[..., 2] << 16)
        return np.stack([(v >> (6 * j)) & 63 for j in range(4)], -1).reshape(m, 16, 32)

    cg, cr = codes(got), codes(ref)
    same = cg == cr
    neutral = ~same & ((cg & 31) == 0) & ((cr & 31) == 0) & ok_blk
    return scale_bad + int((~same & ~neutral).sum()), int(neutral.sum())


L2A_MUTANTS: Tuple[str, ...] = ("ignores E8M0 equality", "accepts byte 255", "zero test on got only",
                                "zero test on REF only", "masks every sign bit", "strict codes",
                                "E8M0 bytes not compared")


def l2a_mutant(kind: str) -> Any:
    """A wrong L2(a) comparator: the loader's decoders with one rule changed ("none": the rule itself)."""
    def fn(got: np.ndarray, ref: np.ndarray) -> Tuple[int, int]:
        cg, cr = odm.g6_codes(got), odm.g6_codes(ref)
        bg, br = odm.g6_block_bytes(got), odm.g6_block_bytes(ref)
        scale = int((bg != br).sum())
        ok_blk = (br != 255) & (bg == br)
        zg, zr = (cg & 31) == 0, (cr & 31) == 0
        if kind == "ignores E8M0 equality":
            ok_blk = br != 255
        elif kind == "accepts byte 255":
            ok_blk = bg == br
        elif kind == "zero test on got only":
            zr = np.ones_like(zr)
        elif kind == "zero test on REF only":
            zg = np.ones_like(zg)
        elif kind == "masks every sign bit":
            zg = zr = (cg & 31) == (cr & 31)
        elif kind == "strict codes":
            ok_blk = np.zeros_like(ok_blk)
        elif kind == "E8M0 bytes not compared":
            scale = 0
        else:
            assert kind == "none", kind
        differ = cg != cr
        neutral = differ & zg & zr & ok_blk[..., None]
        return scale + int((differ & ~neutral).sum()), int(neutral.sum())
    return fn


class G6L2aRuleTest(unittest.TestCase):
    """R-4 (v1.2; r16 iteration 59 R4.md sec. 9 row 6, sec. 10): the loader's G6 comparator under S1C 2.5 as amended
    (L2(a)) -- odm.g6_mismatch_l2a / g6_l2a_counts, used by _sc_case and audit_last_bands. A code passes iff it equals
    REF's, or both are E2M3 zeros (code & 0x1F == 0) in a block whose E8M0 bytes are equal and not 255; E8M0 bytes are
    strict. Checked against R4.md's verbatim numpy reference, the exhaustive 64 x 64 (REF code, got code) sweep (66
    pairs pass: the 64 equal ones and 0x00 <-> 0x20) and exact-count controls; each of 7 mutants (incl. one that
    ignores E8M0 equality and one that accepts byte 255) fails at least one named check."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = odm.encode_g6_int(odm.self_check_rows(300))
        cls.codes, cls.byte = odm.g6_codes(cls.ref), odm.g6_block_bytes(cls.ref)
        fin = (cls.byte != 255)[..., None]
        zero = (cls.codes & 31) == 0
        cls.nz = tuple(int(i) for i in np.argwhere(~zero & fin)[0])  # (row, block, code): a nonzero code
        cls.z20 = tuple(int(i) for i in np.argwhere((cls.codes == 0x20) & fin)[0])
        cls.z00 = tuple(int(i) for i in np.argwhere((cls.codes == 0x00) & fin)[0])
        cls.b255 = tuple(int(i) for i in np.argwhere(cls.byte == 255)[0])  # (row, block)
        cls.zblk = tuple(int(i) for i in np.argwhere((cls.byte == 127) & zero.all(-1))[0])

    def rec(self, codes: Optional[Dict[Tuple[int, ...], int]] = None,
            byte: Optional[Dict[Tuple[int, ...], int]] = None) -> np.ndarray:
        """self.ref with the given codes ((row, block, code) -> value) and E8M0 bytes ((row, block) -> value)."""
        c, b = self.codes.copy(), self.byte.copy()
        for pos, v in (codes or {}).items():
            c[pos] = v
        for pos, v in (byte or {}).items():
            b[pos] = v
        return odm._assemble_g6(c, b)

    def controls(self) -> List[Tuple[str, np.ndarray, Tuple[int, int]]]:
        """(name, got, want (bad, neutral)) against self.ref (R4_cmp_np.py's negative controls plus exact counts)."""
        z20, z00, nz, b255, zblk = self.z20, self.z00, self.nz, self.b255, self.zblk
        blk = z20[:2]
        e = int(self.byte[blk])
        return [
            ("equal", self.ref, (0, 0)),
            ("zero 0x20 -> 0x00", self.rec({z20: 0x00}), (0, 1)),
            ("zero 0x00 -> 0x20", self.rec({z00: 0x20}), (0, 1)),
            ("nonzero sign flip", self.rec({nz: int(self.codes[nz]) ^ 0x20}), (1, 0)),
            ("zero 0x00 -> 0x01", self.rec({z00: 0x01}), (1, 0)),
            ("zero 0x00 -> 0x21", self.rec({z00: 0x21}), (1, 0)),
            ("zero 0x20 -> 0x01", self.rec({z20: 0x01}), (1, 0)),
            ("nonzero -> 0x20", self.rec({nz: 0x20}), (1, 0)),
            ("E8M0 LSB", self.rec(byte={blk: e ^ 1}), (1, 0)),
            ("E8M0 LSB + a zero-sign flip in that block", self.rec({z20: 0x00}, {blk: e ^ 1}), (2, 0)),
            ("E8M0 255 for a finite block + a zero-sign flip in it", self.rec({z20: 0x00}, {blk: 255}), (2, 0)),
            ("0x20 in a byte-255 block", self.rec({b255 + (0,): 0x20}), (1, 0)),
            ("byte-255 block's E8M0 -> 254", self.rec(byte={b255: 254}), (1, 0)),
            ("all-zero block E8M0 127 -> 126", self.rec(byte={zblk: 126}), (1, 0)),
            ("all-zero block E8M0 127 -> 0", self.rec(byte={zblk: 0}), (1, 0)),
        ]

    def sweep(self, fn: Any, block: int = 0, f: int = 7) -> Tuple[set, Dict[Tuple[int, int], int]]:
        """Every (REF code a, got code g) pair at code f of `block` in a one-row record of a finite block: the pairs
        that pass, and the neutral count of each passing pair."""
        row = next(r for r in range(self.ref.shape[0]) if self.byte[r, block] != 255)
        c, b = self.codes[row:row + 1].copy(), self.byte[row:row + 1]
        passed, neutral = set(), {}
        recs = {}
        for v in range(64):
            c[0, block, f] = v
            recs[v] = odm._assemble_g6(c, b)
        for a in range(64):
            for g in range(64):
                bad, neu = fn(recs[g], recs[a])
                if bad == 0:
                    passed.add((a, g))
                    neutral[(a, g)] = neu
        return passed, neutral

    def variants(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """(got, ref) pairs: random zero-sign flips (finite blocks), plus random code and E8M0 corruptions and zero-sign
        flips inside byte-255 blocks, on the SC rows and on U1's npz30 vectors (byte-255 blocks included)."""
        out = []
        rng = np.random.default_rng(20261014)
        for ref in (self.ref, E.vectors()["g6_ref"]):
            codes, byte = odm.g6_codes(ref), odm.g6_block_bytes(ref)
            zero, fin = (codes & 31) == 0, (byte != 255)[..., None]
            for trial in range(12):
                c, b = codes.copy(), byte.copy()
                c[zero & fin & (rng.random(c.shape) < 0.5)] ^= 0x20
                if trial % 4 == 1:
                    i = rng.integers(0, c.size, 4)
                    c.reshape(-1)[i] = rng.integers(0, 64, 4)
                if trial % 4 == 2:
                    j = rng.integers(0, b.size, 3)
                    b.reshape(-1)[j] = rng.integers(0, 256, 3)
                if trial % 4 == 3:
                    c[zero & ~fin & (rng.random(c.shape) < 0.3)] ^= 0x20
                out.append((odm._assemble_g6(c, b), ref))
        return out

    def battery(self, fn: Any) -> List[str]:
        """The names of the checks `fn` fails (an empty list for a correct comparator)."""
        fails = [f"control {name}" for name, got, want in self.controls() if tuple(fn(got, self.ref)) != want]
        passed, neutral = self.sweep(fn)
        if passed != {(a, a) for a in range(64)} | {(0x00, 0x20), (0x20, 0x00)}:
            fails.append("sweep")
        elif neutral[(0x00, 0x20)] != 1 or neutral[(0x20, 0x00)] != 1 or any(neutral[(a, a)] for a in range(64)):
            fails.append("sweep neutral")
        if any(tuple(fn(g, r)) != _ref_g6_mismatch_l2a(g, r) for g, r in self.variants()):
            fails.append("reference")
        return fails

    def test_decoders_invert_the_assembly(self) -> None:
        rng = np.random.default_rng(66)
        c = rng.integers(0, 64, size=(37, 16, 32))
        b = rng.integers(0, 256, size=(37, 16))
        rec = odm._assemble_g6(c, b)
        np.testing.assert_array_equal(odm.g6_codes(rec), c)
        np.testing.assert_array_equal(odm.g6_block_bytes(rec), b)
        np.testing.assert_array_equal(odm._assemble_g6(odm.g6_codes(self.ref), odm.g6_block_bytes(self.ref)), self.ref)

    def test_the_helper_passes_every_check(self) -> None:
        self.assertEqual(self.battery(odm.g6_mismatch_l2a), [])
        self.assertEqual(self.battery(l2a_mutant("none")), [])  # the mutant builder itself is not the failure
        for block, f in ((0, 0), (0, 21), (0, 31), (15, 13)):  # head, head / tail straddle, tail; the last block
            passed, _ = self.sweep(odm.g6_mismatch_l2a, block, f)
            self.assertEqual(len(passed), 66, (block, f))
        report("R-4 L2(a) controls", controls=[(n, odm.g6_mismatch_l2a(g, self.ref)) for n, g, _ in self.controls()])

    def test_counts_by_direction_and_parts(self) -> None:
        for got, ref in self.variants():
            c = odm.g6_l2a_counts(got, ref)
            self.assertEqual((c["bad"], c["neutral"]), _ref_g6_mismatch_l2a(got, ref))
            self.assertEqual(c["bad"], c["scale"] + c["code"])
            self.assertEqual(c["neutral"], c["neutral_00_for_20"] + c["neutral_20_for_00"])
            cg, cr = odm.g6_codes(got), odm.g6_codes(ref)
            bg, br = odm.g6_block_bytes(got), odm.g6_block_bytes(ref)
            ok = ((br != 255) & (bg == br))[..., None]
            self.assertEqual(c["neutral_00_for_20"], int(((cg == 0x00) & (cr == 0x20) & ok).sum()))
            self.assertEqual(c["neutral_20_for_00"], int(((cg == 0x20) & (cr == 0x00) & ok).sum()))
            self.assertEqual(c["scale"], int((got[:, 384:] != ref[:, 384:]).sum()))
        one = odm.g6_l2a_counts(self.rec({self.z20: 0x00}), self.ref)
        self.assertEqual((one["neutral_00_for_20"], one["neutral_20_for_00"]), (1, 0))
        one = odm.g6_l2a_counts(self.rec({self.z00: 0x20}), self.ref)
        self.assertEqual((one["neutral_00_for_20"], one["neutral_20_for_00"]), (0, 1))

    def test_mutants_are_caught(self) -> None:
        must = {"ignores E8M0 equality": "control E8M0 LSB + a zero-sign flip in that block",
                "accepts byte 255": "control 0x20 in a byte-255 block",
                "zero test on got only": "control nonzero -> 0x20",
                "zero test on REF only": "control zero 0x00 -> 0x21",
                "masks every sign bit": "control nonzero sign flip",
                "strict codes": "control zero 0x20 -> 0x00",
                "E8M0 bytes not compared": "control all-zero block E8M0 127 -> 126"}
        self.assertEqual(sorted(must), sorted(L2A_MUTANTS))
        caught = {}
        for kind in L2A_MUTANTS:
            fails = self.battery(l2a_mutant(kind))
            caught[kind] = fails
            self.assertIn(must[kind], fails, kind)
        report("R-4 L2(a) mutants", caught=caught)

    def test_inputs_and_poisons(self) -> None:
        for got, ref in ((self.ref.astype(np.int64), self.ref), (self.ref[:, :399], self.ref[:, :399]),
                         (self.ref.reshape(-1), self.ref.reshape(-1)), (self.ref[:5], self.ref[:6])):
            with self.assertRaises(ValueError):
                odm.g6_mismatch_l2a(got, ref)
        empty = np.zeros((0, 400), dtype=np.uint8)
        self.assertEqual(odm.g6_mismatch_l2a(empty, empty), (0, 0))
        # A FULLY unwritten code (every byte it touches still poison) never decodes to an E2M3 zero under either
        # self-check poison, so it can never pass as a value-neutral code.
        for p in (0xA5, 0x5A):
            codes = odm.g6_codes(np.full((1, 400), p, dtype=np.uint8))
            self.assertFalse(((codes & 31) == 0).any(), hex(p))
        # v1.2 review: the kernel stores dwords (a block's 24-byte group = head dwords 0-3 + tail dwords 4-5), so a
        # code can also straddle a written and an unwritten dword: codes 5 / 10 / 21 / 26 (bytes 3|4, 7|8, 15|16,
        # 19|20). Enumerated for every value of the written byte: under 0xA5 no code that touches an unwritten dword
        # has zero magnitude; under 0x5A codes 10 / 26 decode to 0x20 when the LATER dword is unwritten and the earlier
        # byte's high nibble is 0, so in that run they could pass as neutral. No gating gap: the fully unwritten
        # siblings 11 / 27 decode to 22 there, the 0xA5 run fails the same codes, and the twin is byte-strict (the
        # two runs differ in every unwritten byte).
        zero: Dict[int, set] = {0xA5: set(), 0x5A: set()}
        for p in (0xA5, 0x5A):
            for f in range(32):
                lo, hi = (6 * f) // 8, (6 * f + 5) // 8
                straddle = lo // 4 != hi // 4  # the code's two bytes lie in different dwords
                for unwritten in ((lo, hi) if straddle else (lo,)):
                    other = hi if unwritten == lo else lo
                    for v in (range(256) if straddle else (p,)):
                        g = bytearray(24)
                        g[lo] = g[hi] = p
                        if straddle:
                            g[other] = v  # the written dword's byte
                        if (int.from_bytes(bytes(g), "little") >> (6 * f)) & 31 == 0:
                            zero[p].add((f, unwritten))
        self.assertEqual(zero[0xA5], set())
        self.assertEqual(zero[0x5A], {(10, 8), (26, 20)})
        full = odm.g6_codes(np.full((1, 400), 0x5A, dtype=np.uint8))[0, 0]
        self.assertEqual((int(full[11]), int(full[27])), (22, 22))
        self.assertNotEqual(bytes([0xA5]), bytes([0x5A]))  # the twin differs on every unwritten byte


# ===================================================================================================================== T12
def bf16_weight(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(bf16 bit patterns, their exact fp32 widening) of an fp32 weight, as the self-check makes them (torch .to)."""
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16).copy(), t.float().numpy()


class T12RefParityTest(unittest.TestCase):
    def test_vectors_file_and_pins(self) -> None:
        v = E.vectors()
        self.assertEqual(sorted(v), sorted(NPZ_PINS))
        for k, want in NPZ_PINS.items():
            self.assertEqual(sha(v[k]), want, k)
        self.assertEqual((v["dout_bits"].shape, v["g6_ref"].shape), ((30, 512), (30, 400)))
        self.assertLess(os.path.getsize(E.VECTORS_PATH), 200 * 1024)
        self.assertEqual(odm.ADV_ROWS_SHA256, NPZ_PINS["dout_bits"])
        self.assertEqual(odm.ADV_G6_SHA256, NPZ_PINS["g6_ref"])

    def test_g6_encoders_vs_ref_npz30(self) -> None:
        v = E.vectors()
        adv = odm.adversarial_rows()  # SC1's ported generator == U1's rows
        self.assertEqual(adv.dtype, np.uint16)
        self.assertEqual(sha(adv), NPZ_PINS["dout_bits"])
        for name, enc in (("loader", odm.encode_g6_int), ("emu", E.encode_g6_int)):
            for chunk in (65536, 7):
                got = enc(v["dout_bits"], chunk=chunk)
                bad = np.argwhere(got != v["g6_ref"])
                self.assertEqual(len(bad), 0, f"{name} chunk {chunk}: first mismatches {bad[:4].tolist()}")
        g = odm.encode_g6_int(v["dout_bits"])
        self.assertEqual(int((g[:, 384:] == 255).sum()) > 0, True)  # the vectors hold byte-255 blocks (strict codes)

    def test_hand_vectors(self) -> None:
        v = E.vectors()
        for i in range(v["hand_bits"].shape[0]):
            row = np.zeros((1, 512), dtype=np.uint16)
            row[0, :32] = v["hand_bits"][i]
            for enc in (odm.encode_g6_int, E.encode_g6_int):
                g = enc(row)[0]
                self.assertEqual(bytes(g[0:16]) + bytes(g[256:264]), bytes(v["hand_group"][i]), i)
                self.assertEqual(int(g[384]), int(v["hand_byte"][i]), i)

    def test_g6_lab_rows_vs_ref_pins(self) -> None:
        for m, (bits_sha, ref_sha) in REF_G6_LAB.items():
            bits = E.lab_rows(m)
            self.assertEqual(sha(bits), bits_sha, f"input generator drift at M={m} (numpy PCG64?)")
            self.assertEqual(sha(odm.encode_g6_int(bits)), ref_sha, f"loader encoder M={m}")
            self.assertEqual(sha(E.encode_g6_int(bits)), ref_sha, f"emu encoder M={m}")

    def test_e8m0_and_code_tables_vs_ref_pins(self) -> None:
        a = np.arange(0x8000, dtype=np.int64)
        self.assertEqual(sha(odm.e8m0_byte(a).astype(np.uint8)), REF_E8M0_TABLE)
        self.assertEqual(sha(E.e8m0_byte_np(a).astype(np.uint8)), REF_E8M0_TABLE)
        allb = np.arange(65536, dtype=np.int64)
        with np.errstate(invalid="ignore"):
            vals = (allb.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            fin = (allb & 0x7F80) != 0x7F80
            sel = [allb[fin & (np.abs(vals) <= 7.5 * 2.0 ** (b - 127))] for b in range(254)]
        for fn in (odm.e2m3_code, E.e2m3_code_np):
            allc = np.concatenate([fn(h, b).astype(np.uint8) for b, h in enumerate(sel)])
            self.assertEqual((sha(allc), int(allc.size)), REF_CODES_TABLE)

    def test_scale_operand_rule(self) -> None:
        b = np.arange(256)
        want = np.where(b == 255, 0x3F800000, np.maximum(b << 23, 0x00400000)).astype(np.uint32)
        np.testing.assert_array_equal(odm.scale_operand_bits(b), want)
        np.testing.assert_array_equal(E.scale_operand_bits(b), want)
        self.assertEqual(int(want[0]), 0x00400000)  # byte 0 -> exactly 2^-127
        self.assertEqual(int(want[127]), 0x3F800000)

    def _w6_threads(self, model: E.CuModel) -> Dict[str, np.ndarray]:
        nthr = model.w6_grid * model.w6_block
        g = np.arange(nthr, dtype=np.int64)
        env: Dict[str, Any] = {"threadIdx.x": E.CV(g % model.w6_block, "u32"),
                               "blockIdx.x": E.CV(g // model.w6_block, "u32")}
        env.update({k: E.cv_i32(v) for k, v in model.consts.items()})
        for nm, ex in model.w_decl + model.w_map:
            env[nm] = E._as(E.ev(ex, env), "i32")
        out = {k: np.broadcast_to(np.asarray(env[k].v, dtype=np.int64), (nthr,)).copy()
               for k in ("lane", "n", "kb", "S", "q", "t", "ws_", "jj", "R")}
        sl = np.broadcast_to(np.asarray(E.ev(model.w_slice, env).v, dtype=np.int64), (nthr,))
        out["head"] = sl + np.asarray(E.ev(model.w_head[0], env).v, dtype=np.int64)
        out["tail"] = sl + np.asarray(E.ev(model.w_tail[0], env).v, dtype=np.int64)
        out["s_on"] = np.broadcast_to(np.asarray(E.ev(model.w_s_cond, env).v) != 0, (nthr,)).copy()
        out["s_idx"] = np.broadcast_to(np.asarray(E.ev(model.w_s_idx, env).v, dtype=np.int64), (nthr,)).copy()
        return out

    def test_w6_index_map_vs_contract_and_w6_layout(self) -> None:
        th = self._w6_threads(E.CuModel(E.read_cu()))
        n, kb = th["n"], th["kb"]
        self.assertEqual(sorted((n * 16 + kb).tolist()), list(range(1536 * 16)))  # every (n, kb) once
        # S1 CONTRACT sec. 6 (r16 probes/iter58/mx6rows/S1_CONTRACT_for_r14.md, sha256 5e865ca3...): the L11 scatter
        # inverse t = n >> 8, w = (n >> 5) & 7, j = (n >> 2) & 1, R = 4((n >> 3) & 3) + (n & 3)
        np.testing.assert_array_equal(th["t"], n >> 8)
        np.testing.assert_array_equal(th["ws_"], (n >> 5) & 7)
        np.testing.assert_array_equal(th["jj"], (n >> 2) & 1)
        np.testing.assert_array_equal(th["R"], 4 * ((n >> 3) & 3) + (n & 3))
        np.testing.assert_array_equal(th["S"], kb >> 2)
        np.testing.assert_array_equal(th["q"], kb & 3)
        # ... it inverts r16 w6_layout.n_of (kern/w6_layout.py sha256 08dc4c64..., copied): n = 256t + 32w + 8(R >> 2) +
        # 4j + (R & 3)
        np.testing.assert_array_equal(256 * th["t"] + 32 * th["ws_"] + 8 * (th["R"] >> 2) + 4 * th["jj"] + (th["R"] & 3),
                                      n)
        # the byte map: the 16 head + 8 tail bytes of every 24-B block land on [0, 589824) exactly once, and the
        # inverse table equals w6_layout.w6_src_table() (src[d] = (n * 16 + kb) * 24 + byte)
        src = np.full(589824, -1, dtype=np.int64)
        cnt = np.zeros(589824, dtype=np.int64)
        blk = (n * 16 + kb) * 24
        for b in range(16):
            src[th["head"] + b] = blk + b
            np.add.at(cnt, th["head"] + b, 1)
        for b in range(8):
            src[th["tail"] + b] = blk + 16 + b
            np.add.at(cnt, th["tail"] + b, 1)
        self.assertTrue((cnt == 1).all())
        self.assertEqual(sha(src), W6_SRC_TABLE_SHA)
        # W6S: the CONTRACT's ws[(n >> 5) * 16 + 4q + S] == W6S 16 nb + 4q + k, one writer lane per (nb, kb)
        on = th["s_on"]
        np.testing.assert_array_equal(th["s_idx"][on], 16 * (n[on] >> 5) + 4 * th["q"][on] + th["S"][on])
        self.assertEqual(sorted(th["s_idx"][on].tolist()), list(range(768)))
        # the loader's SC5 reference scatter (odm.w6_scatter) and the emulator's U1 port follow the same map
        rng = np.random.default_rng(66)
        codes = rng.integers(0, 64, size=(1536, 512))
        e = rng.integers(-127, 127, size=(48, 16))
        e[3, 5] = 128
        grp = E.pack24_np(codes.reshape(1536, 16, 32)).reshape(-1)
        for scat in (odm.w6_scatter, E.w6_scatter_np):
            w6, w6s = scat(codes, e)
            np.testing.assert_array_equal(w6, grp[src])
            want_s = np.zeros(768, dtype=np.uint8)
            nb_, kb_ = np.meshgrid(np.arange(48), np.arange(16), indexing="ij")
            want_s[16 * nb_ + 4 * (kb_ & 3) + (kb_ >> 2)] = np.minimum(e + 127, 255)
            np.testing.assert_array_equal(w6s, want_s)

    def _weights(self) -> Dict[str, Tuple[np.ndarray, Any]]:
        ef, hf = odm.edge_weight(), odm.hash_weight()
        self.assertEqual(sha(ef.view(np.uint32)), REF_W6["edge_fp32"][0], "edge weight generator drift")
        self.assertEqual(sha(hf.view(np.uint32)), REF_W6["hash_fp32"][0], "hash weight generator drift")
        eb, ebf = bf16_weight(ef)
        hb, hbf = bf16_weight(hf)
        return {"edge_fp32": (ef, ef), "hash_fp32": (hf, hf), "edge_bf16": (ebf, eb), "hash_bf16": (hbf, hb)}

    def test_w6_references_vs_ref_pins(self) -> None:
        for k, (wf, _) in self._weights().items():
            for name, qw, sc in (("loader", odm.quant_w_int, odm.w6_scatter), ("emu", E.quant_w_int, E.w6_scatter_np)):
                c, e = qw(wf)
                w6, w6s = sc(c, e)
                self.assertEqual((sha(w6), sha(w6s)), REF_W6[k][1:], f"{name} {k}")

    def test_w6_kernel_emulation_vs_ref_pins(self) -> None:
        emu = E.Emulator(E.CuModel(E.read_cu()))
        for k, (_, kernel_input) in self._weights().items():
            r = emu.run_w6(kernel_input)
            self.assertTrue((r["w6c"][:589824] == 1).all(), k)
            self.assertTrue((r["w6c"][589824:] == 0).all() and (r["w6"][589824:] == 0xEE).all(), k)
            self.assertTrue((r["w6sc"][:768] == 1).all() and (r["w6sc"][768:] == 0).all(), k)
            self.assertEqual((sha(r["w6"][:589824]), sha(r["w6s"][:768])), REF_W6[k][1:], k)


# ===================================================================================================================== T13
class _Models(unittest.TestCase):
    src: str = ""
    models: Dict[str, E.CuModel] = {}
    emus: Dict[str, E.Emulator] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = E.read_cu()
        cls.models = {v: E.CuModel(cls.src, E.VARIANT_DEFINES[v]) for v in ("pk32", "f32dn1")}
        cls.emus = {v: E.Emulator(m) for v, m in cls.models.items()}


def _ok(r: Dict[str, Any]) -> bool:
    return bool(r.get("ok") and r.get("banks_ok"))


class T13EmulatorTest(_Models):
    def test_model_extracts_every_construct(self) -> None:
        pk, l3 = self.models["pk32"], self.models["f32dn1"]
        self.assertEqual((pk.pf, pk.g6_f32, pk.dn1_setreg, l3.pf, l3.g6_f32, l3.dn1_setreg), (2, 0, False, 1, 1, True))
        for m in (pk, l3):
            self.assertTrue(all(m.pins.values()) and len(m.pins) == len(E.HELPER_PINS))
            self.assertTrue(m.fp16_ovfl_setreg)
            self.assertIsNotNone(m.co_swap)
            self.assertEqual([(v, inside) for _, v, _, inside in m.pb_nan], [("v0", True), ("v1", True)])
            self.assertEqual([(v, inside) for _, v, inside in m.pb_zero], [("c0", True), ("c1", True)])
            self.assertEqual(m.consts, {"NT": 512, "BAND": 256, "LDS_T": 67584, "LDS_BYTES": 67712, "W6CH": 24576,
                                        "W6SL": 3072})

        def flat(ops: Sequence[Any]) -> List[str]:
            out = []
            for op in ops:
                if op[0] == "if":
                    out.append(f"if {op[1]} {{ " + " ".join(flat(op[2])) + " }")
                elif op[0] == "call":
                    out.append(op[1] if not op[2] else f"{op[1]}({','.join(op[2][:5])})")
            return out

        self.assertEqual(" ".join(flat(pk.schedule)),
                         "load_pass(PA,QA,rs,0,g0) load_pass(PB,QB,rs,1,g1) pass_body(PA,QA,0,g0,inv) "
                         "load_pass(PA,QA,rs,2,g0) pass_body(PB,QB,1,g1,inv) __syncthreads copy_out(lds,rdt,0u,L.tid) "
                         "load_pass(PB,QB,rs,3,g1) if t1 { __syncthreads pass_body(PA,QA,2,g0,inv) "
                         "pass_body(PB,QB,3,g1,inv) __syncthreads copy_out(lds,rdt,65536u,L.tid) }")
        self.assertEqual(" ".join(flat(l3.schedule)),
                         "load_pass(PA,QA,rs,0,g0) pass_body(PA,QA,0,g0,inv) load_pass(PA,QA,rs,1,g1) "
                         "pass_body(PA,QA,1,g1,inv) __syncthreads copy_out(lds,rdt,0u,L.tid) if t1 { "
                         "load_pass(PA,QA,rs,2,g0) __syncthreads pass_body(PA,QA,2,g0,inv) load_pass(PA,QA,rs,3,g1) "
                         "pass_body(PA,QA,3,g1,inv) __syncthreads copy_out(lds,rdt,65536u,L.tid) }")

    def test_lane_maps_match_design_sec5(self) -> None:
        """DESIGN sec. 5's notation and formulas, restated here, vs the expressions the emulator parsed from the .cu."""
        m, emu = self.models["pk32"], self.emus["pk32"]
        tid = np.arange(512)
        env = emu._fns()
        env.update({k: E.cv_i32(v) for k, v in m.consts.items()})
        env["threadIdx.x"] = E.CV(tid, "u32")
        for nm, ex in m.lane:
            env[nm] = E.ev(ex, env)
        g = lambda nm: np.broadcast_to(np.asarray(env[nm].v, dtype=np.int64), (512,))  # noqa: E731
        wv, l = tid >> 6, tid & 63
        h, hb, x = l >> 5, (l >> 4) & 1, l & 15
        k, q = x & 3, x >> 2
        b = 4 * k + q
        for nm, want in (("L.wv", wv), ("L.lane", l), ("L.h", h), ("L.hb", hb), ("L.k", k), ("L.q", q), ("L.b", b),
                         ("g0", 2 * wv + h), ("g1", 16 + 2 * wv + h), ("L.ld_off", 64 * b + 32 * hb),
                         ("L.wl_base", (33 * b + 16 * hb) * 128), ("L.sw", 2 * (b >> 1) + 16 * hb)):
            np.testing.assert_array_equal(g(nm), want, nm)
        for p in range(4):
            gq = 16 * (p & 1) + 2 * wv + h
            e = dict(env, p=E.cv_i32(p), g=E.CV(gq, "i32"))
            e["r0"] = E._as(E.ev(m.lp_r0, e), "u32")
            e["vo"] = E._as(E.ev(m.lp_vo, e), "i32")
            vo = (128 * (p >> 1) + 4 * gq) * 1024 + 64 * b + 32 * hb
            np.testing.assert_array_equal(np.asarray(e["vo"].v), vo)
            for i in range(4):
                e["i"] = E.cv_i32(i)
                offs = {var: np.asarray(E.ev(off, e).v) for var, (_, off) in m.lp_loads.items()}
                np.testing.assert_array_equal(offs["a"], vo + 1024 * i)
                np.testing.assert_array_equal(offs["c"], vo + 1024 * i + 16)
                self.assertEqual(m.lp_rsel[1] if E.ev(m.lp_rsel[0], e).scalar() else m.lp_rsel[2], "Q" if i & 1 else "P")
                self.assertEqual(E.ev(m.lp_base, e).scalar(), (i >> 1) * 8)
            np.testing.assert_array_equal(np.asarray(E.ev(m.pb_wbase, e).v), (33 * b + 16 * hb) * 128 + 4 * (gq ^ (2 * (b >> 1) + 16 * hb)))
            e["rr"] = E._as(E.ev(m.pb_rr, e), "u32")
            e["ro"] = E._as(E.ev(m.pb_ro, e), "i32")
            ro = 400 * (128 * (p >> 1) + 4 * gq + 2 * hb)
            np.testing.assert_array_equal(np.asarray(e["ro"].v), ro)
            offs = [np.asarray(E.ev(off, e).v) for _, _, _, off, _ in m.pb_stores]
            for got, want in zip(offs, (ro + 16 * b, ro + 256 + 8 * b, ro + 400 + 16 * b, ro + 656 + 8 * b,
                                        ro + 400 * k + 384 + 4 * q)):
                np.testing.assert_array_equal(np.broadcast_to(got, (512,)), want)
        self.assertEqual([(r, ix) for r, ix in m.pb_pick], [("P", "dw"), ("Q", "dw"), ("P", "8 + dw"), ("Q", "8 + dw")])
        self.assertEqual(m.pb_byte_src, [0, 1, 2, 3])
        self.assertEqual([(a, b_) for a, b_, _, _ in m.pb_swaps], [(("P", "d"), ("P", "8 + d")), (("Q", "d"), ("Q", "8 + d"))])
        self.assertEqual((m.dpp["dpp_x1"][0], m.dpp["dpp_x2"][0], m.dpp["dpp_r4"][0], m.dpp["dpp_r8"][0]),
                         (0xB1, 0x4E, 0x124, 0x128))
        # copy-out (it = 0..7): cc = tid >> 3, hp = tid & 7, bhi = cc >> 5, cw = cc & 31, hbw = (cw >> 4) & 1, ...
        e = dict(env, tid=E.CV(tid, "i32"), tile_off=E.cv_u32(65536))
        for nm, ex in m.co_decl:
            e[nm] = E._as(E.ev(ex, e), "i32")
        e["lane_off"] = E._as(E.ev(m.co_lane_off, e), "u32")
        cc, hp = tid >> 3, tid & 7
        bhi, cw = cc >> 5, cc & 31
        hbw = (cw >> 4) & 1
        for it in range(8):
            e["it"] = E.cv_i32(it)
            e["b"] = E._as(E.ev(m.co_b, e), "i32")
            e["grp"] = E._as(E.ev(m.co_grp, e), "i32")
            bb = it + 8 * bhi
            grp = (hp ^ ((bb >> 2) + 4 * hbw)) & 7
            np.testing.assert_array_equal(np.asarray(E.ev(m.co_rd, e).v), (33 * bb + cw) * 128 + 16 * grp)
            np.testing.assert_array_equal(np.asarray(E.ev(m.co_st, e).v),
                                          65536 + 4096 * it + (256 * bhi + cw) * 128 + 16 * hp)
            self.assertEqual(bool(E.ev(m.co_swap[0], e).scalar()), bool((it >> 1) & 1))
        self.assertEqual(m.co_swap[1], [2, 3, 0, 1])  # (z, w, x, y)

    def test_staging_bijection_and_copy_out_inverse(self) -> None:
        """One step tile, analytically from the parsed maps: each (token, column) of the tile is staged at exactly one
        LDS byte (65,536 of the 67,584-B image), and the copy-out moves that byte to dout_t[column * 128 + token]."""
        m, emu = self.models["pk32"], self.emus["pk32"]
        tid = np.arange(512)
        env = emu._fns()
        env.update({k: E.cv_i32(v) for k, v in m.consts.items()})
        env["threadIdx.x"] = E.CV(tid, "u32")
        for nm, ex in m.lane:
            env[nm] = E.ev(ex, env)
        where = np.full(m.consts["LDS_T"], -1, dtype=np.int64)  # LDS byte -> token * 512 + column
        for tile in (0, 1):
            where[:] = -1
            for p in (2 * tile, 2 * tile + 1):
                gq = env["g0"] if p % 2 == 0 else env["g1"]
                e = dict(env, p=E.cv_i32(p), g=gq)
                e["r0"] = E._as(E.ev(m.lp_r0, e), "u32")
                e["vo"] = E._as(E.ev(m.lp_vo, e), "i32")
                regs = {"P": np.zeros((512, 16, 2), dtype=np.int64), "Q": np.zeros((512, 16, 2), dtype=np.int64)}
                for i in m.lp_iters:
                    e["i"] = E.cv_i32(i)
                    tgt = m.lp_rsel[1] if E.ev(m.lp_rsel[0], e).scalar() else m.lp_rsel[2]
                    e["base"] = E.cv_i32(E.ev(m.lp_base, e).scalar())
                    for ix, var, comp in m.lp_fill:
                        off = np.asarray(E.ev(m.lp_loads[var][1], e).v, dtype=np.int64) + 4 * "xyzw".index(comp)
                        regs[tgt][:, E.ev(ix, e).scalar(), 0] = off // 1024 - 128 * tile          # token in tile
                        regs[tgt][:, E.ev(ix, e).scalar(), 1] = (off % 1024) // 2                # low-half column
                wbase = np.broadcast_to(np.asarray(E.ev(m.pb_wbase, e).v, dtype=np.int64), (512,))
                for jj in m.pb_jj:
                    e["jj"] = E.cv_i32(jj)
                    e["dw"] = E.cv_i32(E.ev(m.pb_dw, e).scalar())
                    hi = int(E.ev(m.pb_half, e).scalar() != 0)
                    addr = wbase + np.asarray(E.ev(m.pb_st_off, e).v, dtype=np.int64)
                    for j, src in enumerate(m.pb_byte_src):
                        reg, ix = m.pb_pick[src]
                        cell = regs[reg][:, E.ev(ix, e).scalar()]
                        tok, col = cell[:, 0], cell[:, 1] + hi
                        self.assertTrue((where[addr + j] == -1).all(), "a staging byte written twice")
                        where[addr + j] = tok * 512 + col
            used = where >= 0
            self.assertEqual(int(used.sum()), 65536)
            self.assertEqual(len(np.unique(where[used])), 65536)
            self.assertTrue(((where[used] // 512 >= 0) & (where[used] // 512 < 128)).all())
            # the pad slot of every block (slot 33b + 32) is never written
            for blk in range(16):
                self.assertFalse(used[(33 * blk + 32) * 128:(33 * blk + 33) * 128].any())
            e = dict(env, tid=E.CV(tid, "i32"), tile_off=E.cv_u32(0))
            for nm, ex in m.co_decl:
                e[nm] = E._as(E.ev(ex, e), "i32")
            e["lane_off"] = E._as(E.ev(m.co_lane_off, e), "u32")
            read = np.zeros(m.consts["LDS_T"], dtype=np.int64)
            for it in m.co_iters:
                e["it"] = E.cv_i32(it)
                e["b"] = E._as(E.ev(m.co_b, e), "i32")
                e["grp"] = E._as(E.ev(m.co_grp, e), "i32")
                la = np.asarray(E.ev(m.co_rd, e).v, dtype=np.int64)
                dst = np.asarray(E.ev(m.co_st, e).v, dtype=np.int64)
                order = np.arange(16).reshape(4, 4)
                if E.ev(m.co_swap[0], e).scalar():
                    order = order[m.co_swap[1]]
                src_bytes = la[:, None] + order.reshape(-1)[None, :]
                np.add.at(read, src_bytes.reshape(-1), 1)
                tokcol = where[src_bytes]
                dst_bytes = dst[:, None] + np.arange(16)[None, :]
                np.testing.assert_array_equal(dst_bytes, (tokcol % 512) * 128 + tokcol // 512)
            np.testing.assert_array_equal(read, used.astype(np.int64))

    def _matrix(self, mode: str, offset: int, full_cross: bool = False) -> List[Dict[str, Any]]:
        emu = self.emus[mode] if mode in self.emus else E.Emulator(E.CuModel(self.src, E.VARIANT_DEFINES[mode]))
        cases: List[Tuple[str, np.ndarray]] = [(f"lab M={m}", E.lab_rows(m)) for m in E.T13_M]
        cases += [("npz30", E.npz30_rows())] + E.c4_cases()
        res = []
        for i, (label, bits) in enumerate(cases):
            if full_cross:
                combos = [(inv, t) for inv in E.INV_CASES for t in (4, 5, 6, 7)]
            elif label.startswith("lab"):
                combos = [(E.INV_CASES[(i + offset) % 5], (5, 6, 7, 4)[(i + offset) % 4])]
            else:
                combos = [(("1.0", 0x3F800000), 5), (E.INV_CASES[(i + offset) % 5], (4, 5, 6, 7)[(i + offset) % 4])]
            for (iname, inv), t in combos:
                r = E.compare_safe(emu, bits, inv, t, f"{label} inv={iname}")
                r["mode"] = mode
                res.append(r)
                report("T13", **r)
        return res

    def _assert_all_ok(self, res: List[Dict[str, Any]]) -> None:
        bad = [r for r in res if not _ok(r)]
        self.assertEqual(bad, [], f"{len(bad)} of {len(res)} T13 cases fail")
        invs = {r["inv"] for r in res}
        self.assertTrue({f"{v:#010x}" for _, v in E.INV_CASES} <= invs, invs)
        self.assertEqual({r["t"] for r in res}, {4, 5, 6, 7})

    def test_matrix_pk32(self) -> None:
        self._assert_all_ok(self._matrix("pk32", 0))

    def test_matrix_f32dn1(self) -> None:
        self._assert_all_ok(self._matrix("f32dn1", 2))

    def test_every_inv_and_t_on_one_band(self) -> None:
        emu = self.emus["pk32"]
        for m in (129, 100):
            bits = E.lab_rows(m)
            for iname, inv in E.INV_CASES:
                for t in (4, 5, 6, 7):
                    r = E.compare_safe(emu, bits, inv, t, f"M={m} inv={iname} t={t}")
                    self.assertTrue(_ok(r), r)

    @heavy
    def test_heavy_full_cross_matrix(self) -> None:
        res = self._matrix("pk32", 0, True) + self._matrix("f32dn1", 0, True)
        report("T13 heavy summary", cases=len(res), failing=sum(not _ok(r) for r in res))
        self._assert_all_ok(res)

    @heavy
    def test_heavy_pf1_and_reverse_variants(self) -> None:
        res = []
        for mode in ("pf1", "reverse"):
            emu = E.Emulator(E.CuModel(self.src, E.VARIANT_DEFINES[mode]))
            for m in (1, 128, 129, 257, 383, 1000):
                for iname, inv in E.INV_CASES[:3]:
                    r = E.compare_safe(emu, E.lab_rows(m), inv, 6, f"{mode} M={m} inv={iname}")
                    r["mode"] = mode
                    res.append(r)
                    report("T13", **r)
        self.assertEqual([r for r in res if not _ok(r)], [])

    def test_large_m_offset_arithmetic(self) -> None:
        """DESIGN sec. 4 'Offsets' at M = M_MAX (dout 4 GiB): the descriptor bases (the only 64-bit parts) are exact
        64-bit products for the last band, num_records fit 32 bits, and every per-lane offset of every load / store is
        a non-negative 32-bit value inside its descriptor (<= 262,144 / 102,400 / 131,072 B)."""
        m, emu = self.models["pk32"], self.emus["pk32"]
        M = odm.M_MAX
        nb = E.ev(m.grid_expr, {"M": E.cv_i32(M)}).scalar()
        env = emu._fns()
        env.update({k: E.cv_i32(v) for k, v in m.consts.items()})
        env.update({"blockIdx.x": E.cv_u32(nb - 1), "gridDim.x": E.cv_u32(nb), "M": E.cv_i32(M), "M_PAD": E.cv_i32(M)})
        for nm, ex in (("band", m.band_expr), ("band0", m.band0_expr), ("rvl", m.rvl), ("rpl", m.rpl),
                       ("rows_valid", m.rows_valid), ("rows_pad", m.rows_pad)):
            env[nm] = E._as(E.ev(ex, env), E.decl_type(emu.k_text, nm))
        self.assertEqual(env["band0"].t, "i64")
        bases = {name: (E.ev(off, env).scalar(), E.ev(nrec, env).scalar()) for name, (_, off, nrec) in m.rsrc.items()}
        b0 = (nb - 1) * 256
        self.assertEqual(bases, {"rs": (b0 * 1024, 262144), "rg6": (b0 * 400, 102400), "rdt": (b0 * 512, 131072)})
        self.assertGreater(bases["rs"][0], 2 ** 31)  # an int32 product would wrap
        self.assertEqual(bases["rs"][0] + bases["rs"][1], 2 ** 32)  # the band ends exactly at the 4 GiB dout end
        tid = np.arange(512)
        env["threadIdx.x"] = E.CV(tid, "u32")
        for nm, ex in m.lane:
            env[nm] = E._as(E.ev(ex, env), emu.lane_types.get(nm, E.decl_type(emu.k_text, nm)))
        offs = []
        for p in range(4):
            e = dict(env, p=E.cv_i32(p), g=env["g0"] if p % 2 == 0 else env["g1"])
            e["r0"] = E._as(E.ev(m.lp_r0, e), "u32")
            e["vo"] = E._as(E.ev(m.lp_vo, e), "i32")
            for i in m.lp_iters:
                e["i"] = E.cv_i32(i)
                for _, off in m.lp_loads.values():
                    offs.append(("rs", np.asarray(E.ev(off, e).v) + 16))
            e["rr"] = E._as(E.ev(m.pb_rr, e), "u32")
            e["ro"] = E._as(E.ev(m.pb_ro, e), "i32")
            for kind, _, idx, off, cond in m.pb_stores:
                end = np.broadcast_to(np.asarray(E.ev(off, e).v) + (4 * len(idx) if idx else 4), (512,))
                if cond is not None:  # the E8M0 dword store is predicated (lanes k < 2)
                    end = end[np.broadcast_to(np.asarray(E.ev(cond, e).v) != 0, (512,))]
                offs.append(("rg6", end))
        for tile_off in (0, 65536):
            e = dict(env, tid=E.CV(tid, "i32"), tile_off=E.cv_u32(tile_off))
            for nm, ex in m.co_decl:
                e[nm] = E._as(E.ev(ex, e), "i32")
            e["lane_off"] = E._as(E.ev(m.co_lane_off, e), "u32")
            for it in m.co_iters:
                e["it"] = E.cv_i32(it)
                offs.append(("rdt", np.asarray(E.ev(m.co_st, e).v) + 16))
        for name, end in offs:
            self.assertTrue((end > 0).all() and (end <= bases[name][1]).all() and (end < 2 ** 31).all(), name)

    def test_dn1_changes_no_dout_t_byte(self) -> None:
        """DESIGN 6.3 [M]: under the L3 DN1 MODE (subnormal products flush) no dout_t byte changes: 65,536 patterns x
        11 inv x t 4..7 = 2,883,584 triples."""
        invs = [v for _, v in E.INV_CASES] + [int(E.publish_inv_bits(x)) for x in (0.0, 1.0, 3.0e4)] + [
            0x3F800000, 0x77800000, 0x35800000]
        self.assertEqual(len(invs), 11)
        n = 0
        for inv in invs:
            for t in (4, 5, 6, 7):
                a = E.payload_luts(np.uint32(inv), t)[1]
                b = E.payload_luts(np.uint32(inv), t, ftz_out=True)[1]
                np.testing.assert_array_equal(a, b)
                n += a.size
        self.assertEqual(n, 2883584)

    def test_source_edits_of_the_maps_fail(self) -> None:
        cases = [("lab M=129", E.lab_rows(129), 0x424880E5, 5), ("lab M=257", E.lab_rows(257), 0x00000000, 6),
                 ("npz30", E.npz30_rows(), 0x3F800000, 7)] + [(nm, b, 0x3F800000, 5) for nm, b in E.c4_cases()[1:2]]
        for name, old, new in E.SOURCE_EDITS:
            src = E.apply_edits(self.src, [(old, new)])
            try:
                emu = E.Emulator(E.CuModel(src))
            except E.CuError as err:
                report("T13 source edit", edit=name, caught="model", detail=str(err))
                continue
            caught = None
            for label, bits, inv, t in cases:
                r = E.compare_safe(emu, bits, inv, t, label)
                if not _ok(r):
                    caught = r
                    break
            report("T13 source edit", edit=name, caught=caught and caught["case"], detail=caught)
            self.assertIsNotNone(caught, f"source edit not detected: {name}")

    def test_unmodelled_edits_fail_closed(self) -> None:
        for old, new in (("return a > b ? a : b;", "return a >= b ? a : b;"),          # a pinned helper
                         ("L.sw = 2u * (uint32_t)(L.b >> 1) + 16u * (uint32_t)L.hb;", ""),  # a missing map
                         ("const int ro = (int)(rr * 400u);", "rr = rr + 0u; const int ro = (int)(rr * 400u);"),
                         ("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1", "s_nop 0")):
            with self.assertRaises(E.CuError, msg=old):
                E.CuModel(E.apply_edits(self.src, [(old, new)]))


# ===================================================================================================================== T14
C_TYPES = {"int": ctypes.c_int, "float": ctypes.c_float, "long long": ctypes.c_longlong,
           "const char*": ctypes.c_char_p}


class T14LauncherParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = Path(E.CU_PATH).read_bytes()
        cls.model = E.CuModel(cls.raw.decode())

    def test_source_is_the_frozen_text(self) -> None:
        self.assertEqual(odm._SOURCE.resolve(), Path(E.CU_PATH).resolve())
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), CU_SHA256_FROZEN)
        self.assertEqual(odm.SOURCE_SHA256, CU_SHA256_FROZEN)
        self.assertEqual(self.raw.decode().count("\n"), 390)

    def test_literals(self) -> None:
        m = self.model
        checks = " ".join(m.launch_checks)
        self.assertEqual([int(x) for x in re.findall(r"M > (\d+)", checks)], [odm.M_MAX])
        self.assertEqual(odm.M_MAX, 4194304)
        tb = [int(x, 16) for x in re.findall(r"tb == (0x[0-9a-fA-F]+)u", checks)]
        self.assertEqual(tb, list(odm.T_INV_MUL_BITS))
        self.assertEqual([float(np.uint32(b).view(np.float32)) for b in tb], [2.0 ** -t for t in odm.T_SHIFTS])
        self.assertEqual(odm.T_SHIFTS, (4, 5, 6, 7))
        self.assertEqual((m.consts["LDS_T"], m.consts["LDS_BYTES"], odm.LDS_BYTES), (16 * 33 * 128, 67712, 67712))
        self.assertEqual((m.abi, odm.ABI_VERSION), (1, 1))
        self.assertEqual(m.impl_literal, odm.IMPL + "; src=")
        for mm in (1, 2, 255, 256, 257, 769, 4097, 2342736, 2546840, 2546843, odm.M_MAX):
            self.assertEqual(E.ev(m.partial_count_expr, {"M": E.cv_i32(mm)}).scalar(), odm.partial_count(mm))
            self.assertEqual(E.ev(m.grid_expr, {"M": E.cv_i32(mm)}).scalar(), -(-mm // 256))
        self.assertEqual((m.block_name, m.consts["NT"], m.consts["BAND"]), ("NT", 512, 256))
        self.assertEqual((m.w6_grid, m.w6_block, m.w6_grid * m.w6_block), (96, 256, 1536 * 16))
        for g6f, dn1, rev, wpe, pf in ((0, 0, 0, 1, 2), (1, 1, 0, 1, 1), (0, 0, 1, 0, 2), (1, 0, 0, 1, 1)):
            mac = {"OUTPROJ_MX6_G6_F32": str(g6f), "OUTPROJ_MX6_DN1": str(dn1), "OUTPROJ_MX6_REVERSE": str(rev),
                   "OUTPROJ_MX6_WPE": str(wpe), "OUTPROJ_MX6_PF": str(pf)}
            v = E.pp_value(m.variant_expr, mac)
            self.assertEqual(odm.decode_variant(v), {"G6_F32": g6f, "DN1": dn1, "REVERSE": rev, "WPE": wpe, "PF": pf})
        self.assertEqual(E.pp_value(m.variant_expr, m.macros), 40)
        # the producer's parameter list (the mangled name pinned in KERNELS encodes exactly these types)
        self.assertEqual(m.k_params, ["dout", "INV", "tmul", "M", "M_PAD", "g6", "dout_t", "partial", "inv_used"])
        params = norm(find_params(self.raw.decode(), "outproj_dual_mx6_kernel"))
        self.assertEqual(params, ["const uint8_t* __restrict__ dout", "const float* __restrict__ INV", "float tmul",
                                  "int M", "int M_PAD", "uint8_t* __restrict__ g6", "uint8_t* __restrict__ dout_t",
                                  "float* __restrict__ partial", "uint32_t* __restrict__ inv_used"])
        self.assertEqual(odm.KERNELS["producer"], "_ZN4odm623outproj_dual_mx6_kernelEPKhPKffiiPhS4_PfPj")
        self.assertEqual(norm(find_params(self.raw.decode(), "outproj_w6_shadow_kernel")),
                         ["const void* __restrict__ wv", "uint8_t* __restrict__ w6", "uint8_t* __restrict__ w6s"])
        self.assertEqual((odm.KERNELS["w6_f32"], odm.KERNELS["w6_bf16"]),
                         ("_ZN4odm624outproj_w6_shadow_kernelILb0EEEvPKvPhS3_",
                          "_ZN4odm624outproj_w6_shadow_kernelILb1EEEvPKvPhS3_"))

    def test_signatures_match_the_c_prototypes(self) -> None:
        text = E.strip_comments(self.raw.decode())
        protos = {}
        for mm in re.finditer(r'extern "C" ((?:const )?[a-z ]+?\*?)\s*(outproj_\w+)\(([^)]*)\)\s*\{', text):
            ret, name, params = mm.group(1).strip(), mm.group(2), mm.group(3).strip()
            args = [] if params in ("", "void") else [p.strip() for p in params.split(",")]
            ctypes_args = []
            for a in args:
                ty = re.sub(r"\s*\w+$", "", a).strip()
                ctypes_args.append(ctypes.c_void_p if ty.endswith("*") else C_TYPES[ty])
            protos[name] = (C_TYPES[ret], tuple(ctypes_args))
        self.assertEqual(sorted(protos), sorted(odm.SIGNATURES))
        for name, (res, args) in odm.SIGNATURES.items():
            self.assertIs(res, protos[name][0], name)
            self.assertEqual(len(args), len(protos[name][1]), name)
            for a, b in zip(args, protos[name][1]):
                self.assertIs(a, b, name)

    def _c_reject(self, d: Dict[str, Any]) -> bool:
        return c_launch_reject(self.model, d)

    def _py_reject(self, d: Dict[str, Any]) -> bool:
        m, mp = d["M"], d["M_PAD"]
        sm = m if 1 <= m <= odm.M_MAX else 1
        smp = mp if mp >= 128 and mp % 128 == 0 else 128
        ops = [None if d["dout"] is None else fake((sm, 512), torch.bfloat16, d["dout"]),
               None if d["inv"] is None else fake((1,), torch.float32, d["inv"]), d["t"], m, mp,
               None if d["g6"] is None else fake((sm, 400), torch.uint8, d["g6"]),
               None if d["dout_t"] is None else fake((smp // 128, 512, 128), torch.float8_e4m3fn, d["dout_t"]),
               None if d["partial"] is None else fake((odm.partial_count(sm),), torch.float32, d["partial"]),
               None if d["inv_used"] is None else fake((1,), torch.float32, d["inv_used"])]
        return odm.ineligible_reason(*ops) is not None

    def test_ineligible_reason_mirrors_the_c_checks(self) -> None:
        base = dict(dout=1 << 32, inv=2 << 32, g6=3 << 32, dout_t=4 << 32, partial=5 << 32, inv_used=6 << 32,
                    M=300, M_PAD=384, t=2.0 ** -5)
        cases = [dict(base)]
        for mm in (0, -1, 1, 2, 127, 128, 129, 256, 4194303, 4194304, 4194305, 2 ** 31 - 1):
            mp = (mm + 127) // 128 * 128 if mm > 0 else 0
            cases.append(dict(base, M=mm, M_PAD=mp))
        for mp in (512, 256, 300, 0, 128 * 10 ** 4):
            cases.append(dict(base, M_PAD=mp))
        for t in (2.0 ** -4, 2.0 ** -6, 2.0 ** -7, 1.0, 2.0 ** -3, 2.0 ** -8, 0.03, -(2.0 ** -5), float("nan"),
                  float("inf"), 0.0):
            cases.append(dict(base, t=t))
        for k in ("dout", "inv", "g6", "dout_t", "partial", "inv_used"):
            for off in (4, 8, 12, 16):
                cases.append(dict(base, **{k: base[k] + off}))
            cases.append(dict(base, **{k: None}))
        bad = []
        for d in cases:
            c, p = self._c_reject(d), self._py_reject(d)
            if c != p:
                bad.append((d, c, p))
        self.assertEqual(bad, [])
        # documented asymmetry: Python also rejects a t_inv_mul whose fp32 rounding is a valid pattern but which is not
        # exactly that fp32 value (the C side only sees the rounded float)
        d = dict(base, t=2.0 ** -5 * (1.0 + 2.0 ** -40))
        self.assertEqual((self._c_reject(d), self._py_reject(d)), (False, True))


class T14PythonLayoutChecksTest(unittest.TestCase):
    """ineligible_reason's checks beyond the C launcher's (the layouts the kernel assumes; DESIGN 3.2)."""

    def test_layout_rejections(self) -> None:
        good = launch_operands(300)
        self.assertIsNone(odm.ineligible_reason(*good))
        P = 1 << 44

        def with_(i: int, v: Any) -> List[Any]:
            ops = list(good)
            ops[i] = v
            return ops

        rows = [
            ("inv_used aliases inv_scale", with_(8, fake((1,), torch.float32, good[1].data_ptr())), "aliases"),
            ("inv_used fp64", with_(8, fake((1,), torch.float64, P)), "inv_used"),
            ("inv_used not a tensor", with_(8, 5), "inv_used is int"),
            ("dout fp16", with_(0, fake((300, 512), torch.float16, P)), "dout"),
            ("dout width 256", with_(0, fake((300, 256), torch.bfloat16, P)), "dout"),
            ("dout rows 299", with_(0, fake((299, 512), torch.bfloat16, P)), "dout"),
            ("dout column-major", with_(0, FakeCuda(torch.empty((512, 300), dtype=torch.bfloat16,
                                                                 device="meta").t(), P)), "stride"),
            ("inv_scale 2 elements", with_(1, fake((2,), torch.float32, P)), "inv_scale"),
            ("inv_scale bf16", with_(1, fake((1,), torch.bfloat16, P)), "inv_scale"),
            ("g6 [M, 384]", with_(5, fake((300, 384), torch.uint8, P)), "g6"),
            ("g6 int8", with_(5, fake((300, 400), torch.int8, P)), "g6"),
            ("dout_t fp16", with_(6, fake((3, 512, 128), torch.float16, P)), "dout_t"),
            ("dout_t 2-D", with_(6, fake((512, 384), torch.uint8, P)), "dout_t"),
            ("partial 4 entries short", with_(7, fake((4,), torch.float32, P)), "partial"),
            ("partial int32", with_(7, fake((8,), torch.int32, P)), "partial"),
            ("M a float", with_(3, 300.0), "not ints"),
            ("M a bool", with_(3, True), "not ints"),
            ("t_inv_mul a string", with_(2, "x"), "not a float"),
        ]
        for label, ops, pat in rows:
            why = odm.ineligible_reason(*ops)
            self.assertIsNotNone(why, label)
            self.assertIn(pat, why, label)
        cpu = list(good)
        cpu[0] = torch.empty((300, 512), dtype=torch.bfloat16)
        self.assertIn("needs a CUDA device", odm.ineligible_reason(*cpu))

        class FakeCuda1(FakeCuda):
            @property
            def device(self) -> torch.device:  # type: ignore[override]
                return torch.device("cuda", 1)

        other = with_(5, FakeCuda1(torch.empty((300, 400), dtype=torch.uint8, device="meta"), P))
        self.assertIn("not all on one CUDA device", odm.ineligible_reason(*other))
        # an e4m3 dout_t (the TM pool's dtype) and a uint8 one are both accepted; t_inv_mul as a numpy float32 too
        self.assertIsNone(odm.ineligible_reason(*with_(6, fake((3, 512, 128), torch.uint8, P))))
        self.assertIsNone(odm.ineligible_reason(*with_(2, np.float32(2.0 ** -7))))

    def test_w6_ineligible_reason(self) -> None:
        self.assertIsNone(odm.w6_ineligible_reason(fake((1536, 512), torch.float32, 1 << 30)))
        self.assertIsNone(odm.w6_ineligible_reason(fake((1536, 512), torch.bfloat16, (1 << 30) + 2)))  # copied
        for w, pat in ((fake((1536, 512), torch.float16, 0), "dtype"), (fake((512, 1536), torch.float32, 0), "shape"),
                       (torch.empty((1536, 512)), "CUDA"), (None, "not a tensor")):
            self.assertIn(pat, odm.w6_ineligible_reason(w))


def find_params(text: str, name: str) -> List[str]:
    return E.split_top(E.find_function(E.preprocess(text)[0], name)[0])


def norm(xs: Sequence[str]) -> List[str]:
    return [E.norm_ws(x) for x in xs]


# ===================================================================================================================== T15
class T15GateParserTest(unittest.TestCase):
    def test_fixture_generator_matches_the_real_log(self) -> None:
        self.assertEqual(remark_log(CENSUS, "B", roles=("producer",)), REAL_FORMAT_B_PRODUCER)

    def test_accepts_the_pinned_census_in_both_formats(self) -> None:
        parsed = {}
        for fmt in ("A", "B"):
            log = remark_log(CENSUS, fmt)
            parsed[fmt] = odm.parse_remarks(log)
            self.assertEqual(odm.check_resources(parsed[fmt]), GATE_PINNED, fmt)
            self.assertEqual(odm.remark_formats(log), {"A": 30 if fmt == "A" else 0, "B": 30 if fmt == "B" else 0})
        self.assertEqual(parsed["A"], parsed["B"])
        self.assertEqual(sorted(parsed["A"]), sorted(odm.KERNELS.values()))
        noisy = "warning: something\n" + remark_log(CENSUS, "A") + "remark: unrelated [-Rpass=x]\n"
        self.assertEqual(odm.check_resources(odm.parse_remarks(noisy)), GATE_PINNED)
        dup = remark_log(CENSUS, "B") + remark_log(CENSUS, "B", roles=("producer",))  # an identical repeat is fine
        self.assertEqual(odm.check_resources(odm.parse_remarks(dup)), GATE_PINNED)

    def test_format_b_source_path_with_whitespace(self) -> None:  # ODM6-LDR-1 (fixed in v1.1)
        log = remark_log(CENSUS, "B").replace("remark: tree/", "remark: /home/u/my repo/")
        self.assertEqual(odm.check_resources(odm.parse_remarks(log)), GATE_PINNED)
        self.assertEqual(odm.remark_formats(log), {"A": 0, "B": 30})
        # a path that itself holds ":<l>:<c>:"-like segments still splits at the remark's own line / column
        odd = remark_log(CENSUS, "B").replace("remark: tree/", "remark: /home/u/a:12:34:b/my repo/")
        self.assertEqual(odm.parse_remarks(odd), odm.parse_remarks(remark_log(CENSUS, "B")))
        self.assertEqual(odm.check_resources(odm.parse_remarks(odd)), GATE_PINNED)

    def test_format_a_source_path_with_whitespace(self) -> None:
        log = remark_log(CENSUS, "A").replace("/x/odm6.cu", "/home/u/my repo/odm6.cu")
        self.assertEqual(odm.check_resources(odm.parse_remarks(log)), GATE_PINNED)

    def test_boundaries_that_pass(self) -> None:
        for role, kv in (("producer", {"VGPRs": "128"}), ("producer", {"Occupancy [waves/SIMD]": "5"}),
                         ("w6_f32", {"VGPRs": "200", "Occupancy [waves/SIMD]": "2"}), ("w6_bf16", {"TotalSGPRs": "99"}),
                         ("producer", {"AGPRs": "4"})):
            for fmt in ("A", "B"):
                odm.check_resources(odm.parse_remarks(remark_log(mutate(CENSUS, role, **kv), fmt)))

    def test_rejections(self) -> None:
        rows = [
            ("producer VGPR 129", mutate(CENSUS, "producer", VGPRs="129"), r"producer VGPR 129 > 128"),
            ("producer occupancy 3", mutate(CENSUS, "producer", **{"Occupancy [waves/SIMD]": "3"}), r"occupancy 3 < 4"),
            ("producer LDS 67713", mutate(CENSUS, "producer", **{"LDS Size [bytes/block]": "67713"}),
             r"LDS 67713 != 67712"),
            ("producer LDS 67584", mutate(CENSUS, "producer", **{"LDS Size [bytes/block]": "67584"}), r"LDS 67584"),
            ("producer VGPR spill", mutate(CENSUS, "producer", **{"VGPRs Spill": "1"}), r"producer spill 1/0"),
            ("producer SGPR spill", mutate(CENSUS, "producer", **{"SGPRs Spill": "2"}), r"producer spill 0/2"),
            ("producer scratch", mutate(CENSUS, "producer", **{"ScratchSize [bytes/lane]": "16"}), r"scratch 16"),
            ("producer dynamic stack", mutate(CENSUS, "producer", **{"Dynamic Stack": "True"}), r"Dynamic Stack"),
            ("producer VGPR not a count", mutate(CENSUS, "producer", VGPRs="10x3"), r"not a count"),
            ("w6_f32 VGPR spill", mutate(CENSUS, "w6_f32", **{"VGPRs Spill": "1"}), r"w6_f32 spill 1/0"),
            ("w6_bf16 SGPR spill", mutate(CENSUS, "w6_bf16", **{"SGPRs Spill": "3"}), r"w6_bf16 spill 0/3"),
            ("w6_bf16 scratch", mutate(CENSUS, "w6_bf16", **{"ScratchSize [bytes/lane]": "4"}), r"w6_bf16 .*scratch 4"),
        ]
        for label, census, pat in rows:
            for fmt in ("A", "B"):
                with self.assertRaisesRegex(RuntimeError, pat, msg=f"{label} {fmt}"):
                    odm.check_resources(odm.parse_remarks(remark_log(census, fmt)))
        for fmt in ("A", "B"):
            no_prod = remark_log(CENSUS, fmt, roles=("w6_bf16", "w6_f32"))
            with self.assertRaisesRegex(RuntimeError, "no resource remarks for the producer"):
                odm.check_resources(odm.parse_remarks(no_prod))
            no_w6 = remark_log(CENSUS, fmt, roles=("producer", "w6_bf16"))
            with self.assertRaisesRegex(RuntimeError, "no resource remarks for the w6_f32"):
                odm.check_resources(odm.parse_remarks(no_w6))
            extra = remark_log(CENSUS, fmt) + remark_log(
                CENSUS, fmt, roles=("producer",), names=dict(odm.KERNELS, producer="_ZN4odm63fooEv"))
            with self.assertRaisesRegex(RuntimeError, r"unexpected kernel\(s\)"):
                odm.check_resources(odm.parse_remarks(extra))
            renamed = remark_log(CENSUS, fmt, names=dict(odm.KERNELS, w6_bf16=odm.KERNELS["w6_bf16"] + "x"))
            with self.assertRaisesRegex(RuntimeError, r"unexpected kernel"):
                odm.check_resources(odm.parse_remarks(renamed))
            lines = remark_log(CENSUS, fmt).splitlines()
            missing = "\n".join(ln for ln in lines if not (" Occupancy" in ln and "213:" in ln))
            with self.assertRaisesRegex(RuntimeError, r"Occupancy \[waves/SIMD\].*missing or not a count"):
                odm.check_resources(odm.parse_remarks(missing))
            conflict = remark_log(CENSUS, fmt) + remark_log(mutate(CENSUS, "producer", VGPRs="104"), fmt,
                                                            roles=("producer",))
            with self.assertRaisesRegex(RuntimeError, "two different blocks"):
                odm.parse_remarks(conflict)
            twice = [ln for ln in lines]
            k = next(i for i, ln in enumerate(twice) if "VGPRs: 103" in ln)
            twice.insert(k + 1, twice[k].replace("103", "99"))
            with self.assertRaisesRegex(RuntimeError, "twice"):
                odm.parse_remarks("\n".join(twice))


# ===================================================================================================================== T16
def kernel_lines(asm: str, name: str) -> List[str]:
    """The stripped lines of kernel `name`'s body in a device .s (label to .Lfunc_end), comments kept."""
    lines = asm.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(name + ":"))
    out = []
    for ln in lines[i + 1:]:
        if re.match(r"^\.Lfunc_end\d+:", ln):
            break
        out.append(ln.strip())
    return out


def instructions(body: Sequence[str]) -> List[str]:
    out = []
    for ln in body:
        s = ln.split(";")[0].strip()
        if s and not s.startswith(".") and not s.endswith(":"):
            out.append(s)
    return out


def vregs(op: str) -> set:
    m = re.fullmatch(r"v\[(\d+):(\d+)\]", op.strip())
    if m:
        return set(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.fullmatch(r"v(\d+)", op.strip())
    return {int(m.group(1))} if m else set()


def operands(ins: str) -> List[str]:
    parts = ins.split(None, 1)
    return [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []


def valu_writes(ins: str) -> set:
    op = ins.split()[0]
    if not op.startswith("v_") or op.startswith(("v_cmp", "v_readlane", "v_readfirstlane")):
        return set()
    ops = operands(ins)
    if not ops:
        return set()
    w = vregs(ops[0])
    if op.startswith(("v_permlane16_swap", "v_permlane32_swap")) and len(ops) > 1:
        w |= vregs(ops[1])
    return w


def wait_states(ins: str) -> int:
    m = re.fullmatch(r"s_nop\s+(\d+)", ins)
    return int(m.group(1)) + 1 if m else 1


def amdhsa_fields(asm: str, name: str) -> Dict[str, int]:
    i = asm.index(".amdhsa_kernel " + name)
    blk = asm[i:asm.index(".end_amdhsa_kernel", i)]
    return {k: int(v) for k, v in re.findall(r"\.amdhsa_(\w+)\s+(\d+)", blk)}


def vm_tokens(ins: Sequence[str]) -> List[str]:
    toks = []
    for s in ins:
        op = s.split()[0]
        if op.startswith("buffer_load") or op.startswith("global_load"):
            toks.append("L")
        elif op.startswith(("buffer_store", "global_store")):
            toks.append("S")
        elif op == "s_barrier":
            toks.append("B")
        elif op == "s_waitcnt":
            m = re.search(r"vmcnt\((\d+)\)", s)
            if m:
                toks.append("w" + m.group(1))
    return toks


def rle(toks: Sequence[str]) -> str:
    out: List[List[Any]] = []
    for t in toks:
        if out and t in ("L", "S", "B") and out[-1][0] == t:
            out[-1][1] += 1
        else:
            out.append([t, 1])
    return " ".join((t if n == 1 else f"{t}{n}") if t in ("L", "S", "B") else t for t, n in out)


def isa_lint(asm: str, name: str) -> Dict[str, Any]:
    body = kernel_lines(asm, name)
    ins = instructions(body)
    r: Dict[str, Any] = {"instructions": len(ins)}
    # FP6 converts: inline asm only, "s_nop 1" / convert / "s_nop 4" inside one ;;#ASMSTART..;;#ASMEND, dst disjoint
    fp6 = [k for k, ln in enumerate(body) if re.match(r"v_cvt_scalef32_(pk32_fp6_bf16|2xpk16_fp6_f32)\b", ln)]
    r["fp6"] = len(fp6)
    r["fp6_wrapped"] = sum(1 for k in fp6 if body[k - 2:k + 3][:2] == [";;#ASMSTART", "s_nop 1"]
                           and body[k + 1:k + 3] == ["s_nop 4", ";;#ASMEND"])
    disjoint = 0
    for k in fp6:
        ops = operands(body[k].split(";")[0])
        dst, src = vregs(ops[0]), set().union(*[vregs(o) for o in ops[1:]])
        disjoint += int(bool(dst) and not (dst & src))
    r["fp6_disjoint"] = disjoint
    # v_permlane16_swap: no VALU write of either operand within the 2 preceding wait states
    perm = [k for k, s in enumerate(ins) if s.startswith("v_permlane16_swap")]
    hz = []
    for k in perm:
        regs = set().union(*[vregs(o) for o in operands(ins[k])[:2]])
        ws, j = 0, k - 1
        while j >= 0 and ws < 2:
            if valu_writes(ins[j]) & regs:
                hz.append((ins[j], ins[k], ws))
            ws += wait_states(ins[j])
            j -= 1
    r["permlane16_swap"], r["permlane_hazards"] = len(perm), hz
    r["fp8_pk_cvt"] = sum(s.startswith("v_cvt_scalef32_pk_fp8_f32") for s in ins)
    r["med3"] = sum(s.startswith("v_med3_f32") for s in ins)
    r["s_barrier"] = sum(s == "s_barrier" for s in ins)
    r["setreg"] = [s for s in ins if s.startswith("s_setreg")]
    first_valu = next((k for k, s in enumerate(ins) if s.startswith("v_")), len(ins))
    r["setreg_before_valu"] = all(ins.index(s) < first_valu for s in r["setreg"]) if r["setreg"] else True
    r["setreg_nops"] = [ins[ins.index(s) + 1:ins.index(s) + 3] for s in r["setreg"]]
    r["mode"] = amdhsa_fields(asm, name)
    toks = vm_tokens(ins)
    r["vm_seq"] = rle(toks)
    starts = [k for k, t in enumerate(toks) if t == "L" and (k == 0 or toks[k - 1] != "L")]
    r["load_groups"] = [next((j for j in range(a, len(toks)) if toks[j] != "L"), len(toks)) - a for a in starts]

    def waits_until_barrier(k0: int) -> List[int]:
        ws = []
        for t in toks[k0:]:
            if t == "B":
                break
            if t.startswith("w"):
                ws.append(int(t[1:]))
        return ws

    r["waits_after_group"] = [waits_until_barrier(a) for a in starts]
    # in-order vmcnt model (gfx9: loads and stores in one counter): which load group each wait retires
    out: List[Tuple[str, int]] = []
    retire: List[List[int]] = []
    g = -1
    for k, t in enumerate(toks):
        if t == "L":
            if k in starts:
                g += 1
            out.append(("L", g))
        elif t == "S":
            out.append(("S", -1))
        elif t.startswith("w"):
            n = int(t[1:])
            gone = []
            while len(out) > n:
                kind, gg = out.pop(0)
                if kind == "L":
                    gone.append(gg)
            retire.append([k, sorted(set(gone))])
    r["retire"] = retire
    return r


@unittest.skipUnless(os.path.isfile(PROXY_ROCM + "/bin/hipcc"), "the ROCm 7.2.0 proxy toolchain is absent")
class T16ProxyCensusTest(unittest.TestCase):
    builds: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="odm6_t16_"))
        (cls.tmp / "t.txt").write_text("gfx950\n")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def build(cls, name: str, defines: str = "") -> Dict[str, Any]:
        if name in cls.builds:
            return cls.builds[name]
        work = cls.tmp / name
        work.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith("DLRM_HSTU_FP8_OUTPROJ_MX6")}
        env.update(ROCM_PATH=PROXY_ROCM, HIP_PATH=PROXY_ROCM, ROCM_TARGET_LST=str(cls.tmp / "t.txt"))
        env[odm.HIPCC_ENV] = PROXY_ROCM + "/bin/hipcc"
        if defines:
            env[odm.DEFINES_ENV] = defines
        with mock.patch.dict(os.environ, env, clear=True):
            cmd = odm.build_command(out=work / "liboutproj_dual_mx6.so")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(work), env=env, capture_output=True, text=True, timeout=900)
        s = sorted(work.glob("*amdgcn*gfx950*.s"))
        res = dict(name=name, cmd=cmd, rc=proc.returncode, log=proc.stdout + proc.stderr, secs=round(time.time() - t0, 1),
                   asm=s[0].read_text() if s else "", so=work / "liboutproj_dual_mx6.so")
        cls.builds[name] = res
        return res

    def _census(self, b: Dict[str, Any]) -> Dict[str, Any]:
        self.assertEqual(b["rc"], 0, b["log"][-3000:])
        kern = odm.parse_remarks(b["log"])
        return {"kernels": kern, "formats": odm.remark_formats(b["log"]), "sizes": odm.code_sizes(b["asm"])}

    def test_default_build_census(self) -> None:
        b = self.build("default")
        self.assertEqual(b["cmd"][1], str(odm._SOURCE.resolve()))
        self.assertIn(f'-DOUTPROJ_MX6_SRC_SHA256="{CU_SHA256_FROZEN}"', b["cmd"])
        c = self._census(b)
        self.assertEqual(c["formats"], {"A": 0, "B": 30})
        self.assertEqual(odm.check_resources(c["kernels"]), GATE_PINNED)
        self.assertEqual(c["kernels"][odm.KERNELS["producer"]], dict(CENSUS["producer"]))
        self.assertEqual(c["sizes"][odm.KERNELS["producer"]], 21412)
        report("T16 census", variant="default", secs=b["secs"], gate=odm.check_resources(c["kernels"]),
               code_bytes=c["sizes"])

    def test_default_isa_lint(self) -> None:
        b = self.build("default")
        self.assertEqual(b["rc"], 0, b["log"][-3000:])
        r = isa_lint(b["asm"], odm.KERNELS["producer"])
        report("T16 lint", variant="default", **{k: v for k, v in r.items() if k != "retire"})
        self.assertEqual((r["fp6"], r["fp6_wrapped"], r["fp6_disjoint"]), (8, 8, 8))
        self.assertEqual((r["permlane16_swap"], r["permlane_hazards"]), (64, []))
        self.assertEqual(r["fp8_pk_cvt"], 128)
        self.assertEqual(r["med3"], 2 * r["fp8_pk_cvt"])
        self.assertEqual(r["s_barrier"], 4)
        self.assertEqual(r["setreg"], ["s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1"])
        self.assertTrue(r["setreg_before_valu"])
        self.assertEqual(r["setreg_nops"], [["s_nop 7", "s_nop 7"]])
        for k, v in (("ieee_mode", 1), ("dx10_clamp", 1), ("float_denorm_mode_32", 3), ("float_denorm_mode_16_64", 3),
                     ("float_round_mode_32", 0), ("float_round_mode_16_64", 0), ("group_segment_fixed_size", 67712),
                     ("private_segment_fixed_size", 0)):
            self.assertEqual(r["mode"][k], v, k)
        # the vmcnt rule (DESIGN sec. 7 / P0): PF=2 load groups 16 (passes 0+1), 8 (pass 2), 8 (pass 3); no wait from
        # the pass-2 loads to the end of pass 1 below vmcnt(12); before pass 2 exactly vmcnt(21)
        self.assertEqual(r["load_groups"], [16, 8, 8])
        self.assertGreaterEqual(min(r["waits_after_group"][1]), 12)
        self.assertEqual(min(r["waits_after_group"][2]), 21)
        # in-order model: the pass-2 prefetch (group 1) is not retired before the barrier that ends pass 1, nor the
        # pass-3 prefetch (group 2) before the barrier that precedes pass 2
        toks = vm_tokens(instructions(kernel_lines(b["asm"], odm.KERNELS["producer"])))
        bars = [k for k, t in enumerate(toks) if t == "B"]
        early = [g for k, gs in r["retire"] for g in gs if (g == 1 and k < bars[0]) or (g == 2 and k < bars[1])]
        self.assertEqual(early, [])
        self.assertTrue(r["vm_seq"].startswith("S L16 w15 w14 w13 w12 w10 w8 S5 w5 L8 w12 S5 B S8 L8 w21 B w20"),
                        r["vm_seq"])
        for role in ("w6_f32", "w6_bf16"):
            w = isa_lint(b["asm"], odm.KERNELS[role])
            self.assertEqual((w["fp6"], w["setreg"], w["permlane_hazards"]), (0, [], []), role)
            for k, v in (("ieee_mode", 1), ("dx10_clamp", 1), ("float_denorm_mode_32", 3), ("float_round_mode_32", 0),
                         ("group_segment_fixed_size", 0), ("private_segment_fixed_size", 0)):
                self.assertEqual(w["mode"][k], v, (role, k))

    def test_real_so_binds_and_rejects_bad_arguments(self) -> None:
        b = self.build("default")
        self.assertEqual(b["rc"], 0, b["log"][-3000:])
        child = r'''
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
import generative_recommenders.ops.hip.outproj_dual_mx6 as odm
so, sha = Path(sys.argv[1]), sys.argv[2]
lib = odm._Lib(so, {"src_sha256": sha}, False)
f, w = lib.fn["outproj_dual_mx6_launch"], lib.fn["outproj_w6_shadow_launch"]
p, t5 = 1 << 40, 2.0 ** -5
bad = {"M=0": (p, p, t5, 0, 0, p, p, p, None, None), "M=M_MAX+1": (p, p, t5, 4194305, 4194432, p, p, p, None, None),
       "M_PAD=256 for M=1": (p, p, t5, 1, 256, p, p, p, None, None), "t=1.0": (p, p, 1.0, 1, 128, p, p, p, None, None),
       "t=2^-3": (p, p, 0.125, 1, 128, p, p, p, None, None), "t=2^-8": (p, p, 2.0 ** -8, 1, 128, p, p, p, None, None),
       "DOUT null": (None, p, t5, 1, 128, p, p, p, None, None), "INV+4": (p, p + 4, t5, 1, 128, p, p, p, None, None),
       "G6+8": (p, p, t5, 1, 128, p + 8, p, p, None, None), "DOUT_T+4": (p, p, t5, 1, 128, p, p + 4, p, None, None),
       "PARTIAL+12": (p, p, t5, 1, 128, p, p, p + 12, None, None),
       "INV_USED+4": (p, p, t5, 1, 128, p, p, p, p + 4, None)}
rc = {k: int(f(*a)) for k, a in bad.items()}
rc.update({"W+4": int(w(p + 4, 0, p, p, None)), "W6+8": int(w(p, 1, p + 8, p, None)),
           "W6S null": int(w(p, 0, p, None, None))})
try:
    odm._Lib(so, {"src_sha256": "0" * 64}, False)
    forged = "accepted"
except RuntimeError as e:
    forged = str(e)[-120:]
print("RESULT " + json.dumps(dict(abi=int(lib.fn["outproj_dual_mx6_abi_version"]()), impl=lib.impl, variant=lib.variant,
                                  lds=int(lib.fn["outproj_dual_mx6_lds_bytes"]()), rc=rc, forged=forged)))
'''
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
        proc = subprocess.run([sys.executable, "-B", "-c", child, str(b["so"]), CU_SHA256_FROZEN], env=env,
                              capture_output=True, text=True, timeout=300, cwd=str(self.tmp))
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        res = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1][7:])
        report("T16 so", **res)
        self.assertEqual((res["abi"], res["variant"], res["lds"]), (1, 40, 67712))
        self.assertEqual(res["impl"], odm.IMPL + "; src=" + CU_SHA256_FROZEN)
        self.assertEqual(res["rc"], {k: 1 for k in res["rc"]})
        self.assertIn("not built from the hashed source", res["forged"])

    def test_build_gate_from_a_source_path_with_whitespace(self) -> None:  # ODM6-LDR-1 (fixed in v1.1)
        d = self.tmp / "sp ace"
        d.mkdir(exist_ok=True)
        shutil.copy(odm._SOURCE, d / "outproj_dual_mx6.cu")
        env = {k: v for k, v in os.environ.items() if not k.startswith("DLRM_HSTU_FP8_OUTPROJ_MX6")}
        env.update(ROCM_PATH=PROXY_ROCM, HIP_PATH=PROXY_ROCM, ROCM_TARGET_LST=str(self.tmp / "t.txt"))
        env[odm.HIPCC_ENV] = PROXY_ROCM + "/bin/hipcc"
        env[odm.SRC_ENV] = str(d / "outproj_dual_mx6.cu")
        env[odm.CACHE_ENV] = str(self.tmp / "cache_space")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(tempfile, "tempdir", str(self.tmp)):
            so, side, hit = odm.build()
        self.assertEqual(side["summary"], GATE_PINNED)
        self.assertEqual(side["remark_format"], {"A": 0, "B": 30})

    def test_hipcc_failure_reason_carries_the_diagnostic(self) -> None:
        """MINOR-1 (v1.1): a real proxy compile error through prebuild() (GPU / arch check mocked). The latched reason,
        the DISABLED line and r16's re-logged text (its self-check try: "prebuild self-check: <type>: <msg>"[:600]) name
        the compiler's error and source line, on one physical line; hipcc's full output is kept, argv first."""
        d = self.tmp / "broken"
        d.mkdir(exist_ok=True)
        src = odm._SOURCE.read_text()
        self.assertEqual(src.count("v_cvt_scalef32_pk32_fp6_bf16 %0"), 1)
        (d / "outproj_dual_mx6.cu").write_text(src.replace("v_cvt_scalef32_pk32_fp6_bf16 %0",
                                                           "v_cvt_scalef32_pk32_fp6_bf16_TYPO %0"))
        tmpd = self.tmp / "tmp_broken"
        tmpd.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("DLRM_HSTU_FP8_OUTPROJ_MX6")}
        env.update(ROCM_PATH=PROXY_ROCM, HIP_PATH=PROXY_ROCM, ROCM_TARGET_LST=str(self.tmp / "t.txt"))
        env.update({odm.HIPCC_ENV: PROXY_ROCM + "/bin/hipcc", odm.SRC_ENV: str(d / "outproj_dual_mx6.cu"),
                    odm.CACHE_ENV: str(self.tmp / "cache_broken")})
        odm.reset_for_testing(unload=True)
        try:
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(tempfile, "tempdir", str(tmpd)), \
                    mock.patch.object(odm, "_require_device", return_value=0), \
                    mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()), Lines() as cap:
                with self.assertRaises(odm.OutprojMx6Error) as ctx:
                    odm.prebuild()
            reason = odm.failure() or ""
        finally:
            odm.reset_for_testing(unload=True)
        report("T16 hipcc failure", reason=reason, lines=cap.printed())
        self.assertTrue(reason.startswith("prebuild build/load: RuntimeError: hipcc rc=1: "), reason)
        r16_msg = f"{type(ctx.exception).__name__}: {ctx.exception}"
        for text in (reason, str(ctx.exception), f"prebuild self-check: {r16_msg[:600]}"):
            self.assertIn("error: invalid instruction", text)
            self.assertIn("v_cvt_scalef32_pk32_fp6_bf16_TYPO", text)
            self.assertNotIn("\n", text)
        self.assertNotIn("--offload-arch", reason)  # the argv is in the kept log, not in the reason
        self.assertLessEqual(len(reason), 600)
        self.assertEqual(len(cap.printed()), 1, cap.printed())
        self.assertRegex(cap.printed()[0], r"^\[outproj-mx6\] DISABLED: .* \[rank \d+\]$")
        logs = sorted(tmpd.glob("outproj_mx6_hipcc_fail_*/hipcc.log"))
        self.assertEqual(len(logs), 1, logs)
        self.assertIn(f"(full log: {logs[0]})", reason)
        text = logs[0].read_text()
        self.assertTrue(text.startswith(PROXY_ROCM + "/bin/hipcc "), text[:200])
        self.assertGreaterEqual(text.count("error: invalid instruction"), 2)  # every error, not only the first
        self.assertEqual([p for p in (self.tmp / "cache_broken").rglob("*") if p.is_file()], [])

    @heavy
    def test_heavy_variant_census(self) -> None:
        want = {"pf1": ("-DOUTPROJ_MX6_PF=1", 81, 1), "l3dn1": ("-DOUTPROJ_MX6_G6_F32=1 -DOUTPROJ_MX6_DN1=1", 89, 2),
                "reverse": ("-DOUTPROJ_MX6_REVERSE=1", 103, 1), "wpe0": ("-DOUTPROJ_MX6_WPE=0", 103, 1),
                "ilp": ("-mllvm -amdgpu-sched-strategy=iterative-ilp", 103, 1)}
        for name, (defs, vgpr, nset) in want.items():
            b = self.build(name, defs)
            c = self._census(b)
            gate = odm.check_resources(c["kernels"])
            r = isa_lint(b["asm"], odm.KERNELS["producer"])
            report("T16 variant", variant=name, secs=b["secs"], gate=gate, lint={k: v for k, v in r.items()
                                                                                  if k != "retire"})
            self.assertEqual((gate["producer"]["vgpr"], gate["producer"]["occupancy"], gate["producer"]["lds"]),
                             (vgpr, 4, 67712), name)
            self.assertEqual((r["fp6"], r["fp6_wrapped"], r["fp6_disjoint"], r["permlane_hazards"]), (8, 8, 8, []), name)
            self.assertEqual(len(r["setreg"]), nset, name)
            if name == "l3dn1":
                self.assertIn("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 1", r["setreg"])
        b = self.build("l3pf2", "-DOUTPROJ_MX6_G6_F32=1 -DOUTPROJ_MX6_PF=2")  # DESIGN sec. 7: rejected by the gate
        with self.assertRaisesRegex(RuntimeError, "producer spill"):
            odm.check_resources(self._census(b)["kernels"])


# ===================================================================================================================== T18
class T18MutantTest(_Models):
    def _cases(self) -> List[Tuple[str, np.ndarray, int, int]]:
        c4 = dict(E.c4_cases())
        return [("lab M=129", E.lab_rows(129), 0x3F800000, 5), ("lab M=257", E.lab_rows(257), 0x3F800000, 6),
                ("npz30", E.npz30_rows(), 0x3F800000, 5), ("C4-P2", c4["C4-P2"], 0x3F800000, 5),
                ("C4-P3b M=128", c4["C4-P3b M=128"], 0x3F800000, 5)]

    def _run(self, mode: str) -> Dict[str, int]:
        defines = E.VARIANT_DEFINES[mode]
        out = {}
        for name, kind, payload in E.MUTANTS:
            try:
                if kind == "cu":
                    emu = E.Emulator(E.CuModel(E.apply_edits(self.src, payload), defines))
                else:
                    emu = E.Emulator(E.CuModel(self.src, defines), mutants=(payload,))
            except E.CuError as err:
                out[name] = len(self._cases())
                report("T18", mode=mode, mutant=name, failing=out[name], of=len(self._cases()), detail=str(err))
                continue
            fails = []
            for label, bits, inv, t in self._cases():
                r = E.compare_safe(emu, bits, inv, t, label)
                if not _ok(r):
                    fails.append(label)
            out[name] = len(fails)
            report("T18", mode=mode, mutant=name, kind=kind, failing=len(fails), of=len(self._cases()), cases=fails)
        return out

    def test_each_appendix_b_mutant_fails_a_t13_case(self) -> None:
        got = self._run("pk32")
        self.assertEqual(len(got), 10)
        self.assertEqual([k for k, v in got.items() if v < 1], [])
        # the unmutated emulator passes the same five cases (the failures are the mutants')
        for label, bits, inv, t in self._cases():
            self.assertTrue(_ok(E.compare_safe(self.emus["pk32"], bits, inv, t, label)), label)

    @heavy
    def test_heavy_mutants_in_f32dn1(self) -> None:
        got = self._run("f32dn1")
        self.assertEqual([k for k, v in got.items() if v < 1], [])


# ===================================================================================================================== T19
class T19LineTextTest(_Env):
    def test_loader_source_has_one_sink(self) -> None:
        import ast

        src = Path(odm.__file__).read_text()
        tree = ast.parse(src)
        self.assertEqual(odm._ABSENT_TOKENS, ABSENT_TOKENS)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        named = [n for n in calls if isinstance(n.func, ast.Name)]
        self.assertEqual([n.lineno for n in named if n.func.id == "print"], [])
        self.assertEqual([n.lineno for n in calls if isinstance(n.func, ast.Attribute)
                          and getattr(n.func.value, "id", "") in ("logger", "logging", "sys")
                          and n.func.attr in ("warning", "info", "error", "debug", "write")], [])
        self.assertEqual(len([n for n in named if n.func.id == "arm_print_all_ranks"]), 1)  # inside _say() only
        says = [n for n in named if n.func.id in ("_say", "_say_once")]
        self.assertGreaterEqual(len(says), 6)
        for n in says:
            msg = n.args[-1]
            if isinstance(msg, ast.Name):
                self.assertEqual(msg.id, "msg")
                continue
            self.assertIsInstance(msg, ast.JoinedStr, n.lineno)
            first, second = msg.values[0], msg.values[1]
            self.assertTrue(isinstance(first, ast.FormattedValue) and getattr(first.value, "id", "") == "FAMILY",
                            n.lineno)
            self.assertTrue(isinstance(second, ast.Constant) and second.value.startswith(" "), n.lineno)
            text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in msg.values)
            for tok in ABSENT_TOKENS:
                self.assertNotIn(tok.lower(), text.lower(), n.lineno)

    @unittest.skipIf(torch.cuda.is_available(), "GPU host")
    def test_every_line_kind(self) -> None:
        lines: List[str] = []
        printed: List[str] = []
        # DISABLED (no GPU), with an exception text full of r16 tokens
        with mock.patch.object(odm, "_require_device",
                               side_effect=RuntimeError("MX6 DISABLED mx6 inert Mx6 Release MX6 RESCUE device fault")), \
                Lines() as cap:
            with self.assertRaises(odm.OutprojMx6Error):
                odm.prebuild()
        lines += cap.msgs
        printed += cap.printed()
        self.assertIn("MX6_DISABLED mx6_inert Mx6_Release MX6_RESCUE device_fault", cap.msgs[0])
        # stale HIP error
        odm.reset_for_testing(unload=True)
        with Lines() as cap:
            odm._check_rc(-1, "outproj_dual_mx6_launch")
        lines += cap.msgs
        printed += cap.printed()
        # armed OK, device-side self-check error
        lib = T11LoaderCpuTest._fake_lib(self)
        gpu = T11LoaderCpuTest._gpu(self)
        with gpu, mock.patch.object(odm, "_load", return_value=lib), mock.patch.object(odm, "_lib", lib), \
                mock.patch.object(odm, "self_check", return_value=(True, "ok MX6 rescue DEVICE FAULT")), Lines() as cap:
            odm.prebuild()
        lines += cap.msgs
        printed += cap.printed()
        odm.reset_for_testing(unload=True)
        with T11LoaderCpuTest._gpu(self), mock.patch.object(odm, "_load", return_value=lib), \
                mock.patch.object(odm, "self_check", side_effect=RuntimeError("GPU core dump: MX6 INERT")), \
                Lines() as cap:
            with self.assertRaises(RuntimeError):
                odm.prebuild()
        lines += cap.msgs
        printed += cap.printed()
        # the dev audit line (D16), pass and fail
        for ok in (True, False):
            odm.reset_for_testing(unload=True)
            odm.set_prebuilt_for_testing(True)
            ops = launch_operands(1 << 20)
            c_lib = mock.Mock(fn={"outproj_dual_mx6_launch": lambda *a: 0})
            with mock.patch.dict(os.environ, {odm.AUDIT_ENV: "1"}), mock.patch.object(odm, "_lib", c_lib), \
                    mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()), \
                    mock.patch.object(torch.cuda, "current_stream", return_value=mock.Mock(cuda_stream=0)), \
                    mock.patch.object(odm, "audit_last_bands", return_value=(ok, "G6 mism 0 MX6 release")), \
                    Lines() as cap:
                if ok:
                    odm.launch(*ops)
                    odm.launch(*ops)  # the audit runs once
                else:
                    with self.assertRaises(odm.OutprojMx6Error):
                        odm.launch(*ops)
            lines += cap.msgs
            printed += cap.printed()
            self.assertEqual(len(cap.msgs), 1 if ok else 2, cap.msgs)
        kinds = ("DISABLED: ", "stale HIP error", "armed OK: ", "prebuild self-check hit a device-side error",
                 "dev audit (")
        for k in kinds:
            self.assertTrue(any(k in ln for ln in lines), k)
        assert_line_rules(self, lines)
        self.assertEqual(len(printed), len(lines))
        for ln in printed:
            self.assertTrue(re.fullmatch(r"\[outproj-mx6\] .* \[rank \d+\]", ln), ln)
        assert_line_rules(self, [ln.rsplit(" [rank ", 1)[0] for ln in printed])
        report("T19", lines=lines)

    def test_multiline_texts_print_one_physical_line(self) -> None:
        """MINOR-1 (v1.1): exception texts are multi-line (torch's HIP errors, hipcc logs). Every message still prints
        as ONE physical line with the prefix and the rank tag, and the latched reason / OutprojMx6Error text that r16
        re-logs has no newline: the DISABLED latch, the device-fault line, and the seam's _mx6_line_once."""
        hip = ("HIP error: an illegal memory access was encountered\nHIP kernel errors might be asynchronously "
               "reported at some other API call.\nFor debugging consider passing AMD_SERIALIZE_KERNEL=3\n"
               "Compile with `TORCH_USE_HIP_DSA` to enable device-side assertions.\n")
        printed: List[str] = []
        with mock.patch.object(odm, "_require_device", side_effect=RuntimeError("no ROCm GPU\n  (detail)\n")), \
                Lines() as cap:
            with self.assertRaises(odm.OutprojMx6Error) as ctx:
                odm.prebuild()
        printed += cap.printed()
        self.assertEqual(len(cap.printed()), 1, cap.printed())
        self.assertNotIn("\n", odm.failure())
        self.assertNotIn("\n", str(ctx.exception))
        self.assertIn("no ROCm GPU (detail) -> FP8 dual cast for this process", cap.printed()[0])
        odm.reset_for_testing(unload=True)
        lib = T11LoaderCpuTest._fake_lib(self)
        with T11LoaderCpuTest._gpu(self), mock.patch.object(odm, "_load", return_value=lib), \
                mock.patch.object(odm, "self_check", side_effect=RuntimeError(hip)), Lines() as cap:
            with self.assertRaises(RuntimeError):
                odm.prebuild()
        printed += cap.printed()
        self.assertEqual(len(cap.printed()), 1, cap.printed())
        self.assertNotIn("\n", odm.failure())
        odm.reset_for_testing(unload=True)
        with T11LoaderCpuTest._gpu(self), mock.patch.object(odm, "_load", return_value=lib), \
                mock.patch.object(odm, "self_check", side_effect=ValueError("bad\nshape")), Lines() as cap:
            with self.assertRaises(odm.OutprojMx6Error):
                odm.prebuild()
        printed += cap.printed()
        self.assertEqual(cap.msgs, ["[outproj-mx6] DISABLED: prebuild self-check: ValueError: bad shape -> FP8 dual "
                                    "cast for this process"])
        from generative_recommenders.ops.triton import bwd_fp8 as bf

        if hasattr(bf, "_mx6_line_once"):
            got: List[str] = []
            with mock.patch("generative_recommenders.ops.utils.arm_print_all_ranks", side_effect=got.append):
                bf._MX6_LINES.discard(("t19", "multi"))
                bf._mx6_line_once(("t19", "multi"), "[outproj-mx6] S1 raised before its commit: RuntimeError: " + hip)
                bf._MX6_LINES.discard(("t19", "multi"))
            self.assertEqual(len(got), 1)
            self.assertNotIn("\n", got[0])
            self.assertTrue(got[0].startswith("[outproj-mx6] S1 raised before its commit: RuntimeError: HIP error: an "
                                              "illegal memory access was encountered HIP kernel errors"), got[0])
        for ln in printed:
            self.assertTrue(re.fullmatch(r"\[outproj-mx6\] .* \[rank \d+\]", ln), ln)
        assert_line_rules(self, [ln.rsplit(" [rank ", 1)[0] for ln in printed])

    def test_seam_append_lines(self) -> None:
        """The bwd_fp8 EOF append's line texts (S1 / R2 / S1-E): every literal mentioning outproj-mx6 starts with
        "[outproj-mx6]", none carries an r16 token outside the defusing table, and _mx6_line_once defuses them."""
        import ast

        from generative_recommenders.ops.triton import bwd_fp8 as bf

        if not hasattr(bf, "_mx6_line_once"):
            self.skipTest("no seam append in this tree")
        src = Path(bf.__file__).read_text()
        start = src.index("# r14 iteration 66: the MX6 dY producer seam")
        line0 = src[:start].count("\n") + 1
        tree = ast.parse(src)
        table = next(n for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                     and getattr(n.target, "id", "") == "_MX6_ABSENT_TOKENS")
        skip = set(range(table.lineno, table.end_lineno + 1))
        texts = []
        for node in ast.walk(tree):
            if getattr(node, "lineno", 0) < line0 or getattr(node, "lineno", 0) in skip:
                continue
            if isinstance(node, ast.JoinedStr):
                texts.append("".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                texts.append(node.value)
        mentions = [t for t in texts if "outproj-mx6" in t]
        self.assertTrue(mentions)
        for t in mentions:
            if not t.lstrip().startswith('"[outproj-mx6]"'):  # docstrings quoting the prefix
                self.assertTrue(t.startswith("[outproj-mx6]") or "\n" in t, t[:100])
        for t in texts:
            for tok in ABSENT_TOKENS:
                self.assertNotIn(tok.lower(), t.lower(), t[:100])
        got: List[str] = []
        with mock.patch("generative_recommenders.ops.utils.arm_print_all_ranks", side_effect=got.append):
            bf._MX6_LINES.discard(("t19", "x"))
            bf._mx6_line_once(("t19", "x"), "[outproj-mx6] t19 MX6 DISABLED mx6 inert MX6 Release mx6 rescue Device Fault")
            bf._mx6_line_once(("t19", "x"), "[outproj-mx6] t19 again")
            bf._MX6_LINES.discard(("t19", "x"))
        self.assertEqual(len(got), 1)
        assert_line_rules(self, got)


if __name__ == "__main__":
    unittest.main()

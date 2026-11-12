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
"""Tests of the NIB keep-mask READER side, NIB v2 (r16 iteration 54, gin item (94); knob
DLRM_HSTU_FP8_LNMD_MASK_NIB, only "1" arms, default off): ONLY the DO8-ONLY 'only' build of ops/hip/l11_do8.{cu,py}
(EM 2) gets -DL11_MKB_NIB=1 and reads the LN*U*dropout keep mask as uint8 [M, 256] (ops/lnmd_mask_nib.py, layout A)
instead of int8 [M, 512]; DO8's 'plus' .so is always today's int8 build (r16 Leg L job 70100: the NIB 'plus' build
failed the per-variant SGPR-spill gate). L11's dispatch gate accepts a NIB mask without unpacking it; L11's own int8
launches unpack it exactly.

CPU tests (always run; torch only: no Triton, no GPU, no hipcc, no build; the one test that executes the NIB ACTIVE
line needs triton_hstu_linear importable and skips otherwise):
  * contract: pack_ref / unpack exact on random masks (every byte value 0..7, many shapes, writer-like keep rates), the
    layout-A byte formula, bits 3 and 7 zero, a byte > 7 breaks the round trip (why the self-checks gate it),
    layout_ok;
  * build: _flags carries -DL11_MKB_NIB=1 (last flag) ONLY for the 'only' build, iff the knob is "1" AND the source
    text has the switch "#if L11_MKB_NIB" (a source that only names the macro does not count; review R2-F8) AND the
    NIB writers can arm (review R1-F3: LNMD_FWD_MASKGEN / LNMD_FWD_V2 not "0", MASK16 "1", BWD_OUT_WGRAD_IMPL not
    "producer"); the 'plus' flags, hipcc command and cache key are the knob-off build's for every knob value;
    nib_build_requested follows the L11-VD source selector (the pre-diet l11_do8_novd.cu never builds NIB); the
    'only' cache key separates NIB and int8 builds; the .cu switch pins; the impl strings (NIB off: ship15's bytes;
    NIB on: + "+mkb=nib");
  * state: mask_format() (the 'plus' .so) and only_mask_format() (the 'only' .so) parsing; nib_ready() needs every v2
    condition (knob; DO8 requested / enabled / prebuilt; DO8-ONLY requested, its 'only' .so prebuilt, not latched and
    reading NIB; L11 enabled and prebuilt), reads nothing else with the knob off, and ignores the 'plus' format;
    _load raises on a NIB 'plus' .so; _load_only requires the format its flags requested; the prebuild "armed OK"
    lines keep their text, the 'plus' line ends " mask=int8" always and the 'only' line " mask=nib(uint8[M,256])" or
    " mask=int8"; the NIB-requested-but-inert lines (pre-diet source, writers off, DO8-ONLY off); the exit line
    "[l11-do8] nib counters: ... final=1" after the DO8 counters line, knob on only (review N2);
  * prebuild fallback (review R1-F4, v2): a NIB 'only' build / impl-format / self-check / mismatch failure prints one
    NIB DISABLED line and the int8 'only' build + self-check run instead (nothing latched; 'plus' built once and left
    alone; nib_ready() False); a 'plus' failure latches DO8 exactly as without NIB; an int8 failure, a second
    failure, FAIL_CLOSED and a device fault behave as before;
  * run_e5_plus routing (VERIFY B1: never a latch on a format question): a NIB mask is a counted DECLINE ("E5 mask
    format", one NOT armed line) BEFORE work_list / _plan_only / launch_e5_plus / any torch allocation unless
    nib_ready() and its layout hold, and never calls _disable / _disable_only; with the NIB 'only' .so ready it runs
    'only' with the mask object itself when the plan answers "only" and declines (no latch, no open plan, nothing
    launched) when it answers "plus"; an int8 mask while the NIB 'only' .so is loaded runs 'plus' WITHOUT _plan_only
    (the registry is not asked; counted mask_int8_plus, one line per process; a stale 'only' call entry is dropped
    as _plan_only drops it, DO8-ONLY on only: review N1); everything else (the knob off, an
    int8 'only' .so) is today's path; the same through the REAL registry (plan / publish / bind); the raw
    launch_e5_plus raises on a mismatch before the kernel entry, per .so ('plus' / 'only');
  * self-checks (executed on CPU with the kernels faked): self_check ('plus') launches DO8 with the int8 mask and adds
    no NIB detail; self_check_only feeds a NIB 'only' launch pack_ref(mask) and its EM = 1 reference the int8 mask,
    gates unpack(pack_ref(mask)) == mask, and is unchanged for an int8 'only' .so;
  * L11: the D8 gate accepts a NIB mask without unpacking and still rejects other layouts; run / launch_e5 / run_e5
    hand the int8 kernel the exact unpack of a NIB mask (counted once per mask, cached, one line), int8 masks as is;
    an unpack failure in run / run_e5 raises to the caller and never latches L11 (review R1-F6);
  * thl: the NIB ACTIVE line prints the 'only' impl (l11_do8.only_impl(), "+mkb=nib"); the writers' gate asks
    l11_do8.nib_ready().
GPU tests (gfx950 + hipcc; skipped otherwise): the knob-on 'plus' build is the knob-off build (same cache entry, int8
impl); the NIB 'only' build vs the int8 'only' build byte for byte through launch_e5_plus on several shapes (dz8 incl.
pad rows / guard columns, dO8, E8, AMAXP / DWP / DBP / DUCS, P; the sentinel dx untouched) and vs the int8 'plus'
build (EM 1); resources (0 VGPR spill / 0 scratch from DO8's own gate, 'only' SGPR spill <= the int8 'only' build's +
2: iter53 VERIFY B2); run_e5_plus with the real .so files routes an int8 mask to 'plus', a NIB mask planned "only" to
the NIB 'only' .so and declines a NIB mask planned "plus" without a latch; self_check and self_check_only pass.
"""

import ast
import contextlib
import ctypes
import hashlib
import io
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest import mock

import torch

from generative_recommenders.ops import lnmd_mask_nib as nibm
from generative_recommenders.ops.hip import l11_do8 as do8
from generative_recommenders.ops.hip import l11_outproj_lnmd as l11
from generative_recommenders.ops.triton import bwd_fp8, hstu_du_e5m2

# run_e5_plus / run_e5 import these lazily; loading them here (both are torch-only) keeps that import out of the
# blocks that forbid torch allocations
_PRELOADED = (bwd_fp8, hstu_du_e5m2)

E4, E5, BF, F32, U8, I8 = torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float32, torch.uint8, torch.int8
D, H, HD, DU_ROW = 512, 4, 128, 2048
P_DROP = 0.1
EDGE = (1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128)
KNOB = nibm.KNOB
VD_KNOB = l11.VD_KNOB
SRC_ENV = do8.KNOB + "_SRC"
ONLY = do8.ONLY_KNOB
# the knobs without which the NIB writers can never arm (l11_do8._nib_writers_off; review R1-F3): every test starts with
# them as in production (the yambda_5b gin), and the tests that need them off say so
MASKGEN, MASK16, FWD_V2, OUT_IMPL = ("DLRM_HSTU_FP8_LNMD_FWD_MASKGEN", "DLRM_HSTU_FP8_MASK16",
                                     "DLRM_HSTU_FP8_LNMD_FWD_V2", "DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL")
_WRITERS_ON = {MASKGEN: "1", MASK16: "1", FWD_V2: "1"}
_ENV_KEYS = (KNOB, VD_KNOB, SRC_ENV, ONLY, do8.SCHED_ENV, do8.KNOB + "_FAIL_CLOSED", MASKGEN, MASK16, FWD_V2, OUT_IMPL)
# ship15's (v6x) impl strings with L11DO8_SRC_SHA256 = "x": the NIB-off strings must stay byte-identical
_IMPL_PLUS = ("l11_outproj_lnmd_bwd_do8 abi=1 variant=e5plus(E5 1,WLIST 2 device list,EM 1 plus,RET 0 LDS staging,"
              "EMV 2 bf16 convert,packed running max)+vd(C1,C2,C3,C4,LANEFIX,PIN)+l11_do8_wl_build src_sha256=x")
_IMPL_ONLY = ("l11_outproj_lnmd_bwd_do8 abi=1 variant=e5only(E5 1,WLIST 2 device list,EM 2 only no bf16 DX store,"
              "RET 0 LDS staging,EMV 2 bf16 convert,packed running max)+vd(C1,C2,C3,C4,LANEFIX,PIN)+l11_do8_wl_build "
              "src_sha256=x")
_IMPL_PLUS_NIB = _IMPL_PLUS.replace(" src_sha256=", "+mkb=nib src_sha256=")
_IMPL_ONLY_NIB = _IMPL_ONLY.replace(" src_sha256=", "+mkb=nib src_sha256=")
_INT8_PLUS_LINE = f"{do8.FAMILY}: {nibm.FAMILY}: an int8 keep mask reached DO8 while its 'only' .so reads NIB"
_THL = Path(do8.__file__).resolve().parents[1] / "triton" / "triton_hstu_linear.py"


# ---------------------------------------------------------------------------------------------------- helpers
@contextlib.contextmanager
def _env(values: Dict[str, Optional[str]]) -> Iterator[None]:
    """Set (str) or remove (None) environment variables for the block, then restore them."""
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
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


class _Capture(contextlib.AbstractContextManager):
    def __enter__(self):  # pyre-ignore[3]
        self.buf = io.StringIO()
        self._cm = contextlib.redirect_stdout(self.buf)
        self._cm.__enter__()
        return self

    def __exit__(self, *a):  # pyre-ignore[2,3]
        self._cm.__exit__(*a)
        return False

    @property
    def text(self) -> str:
        return self.buf.getvalue()


class _FakeFn:
    """A ctypes-function stand-in: restype / argtypes are settable, a call returns `value`."""

    def __init__(self, value: Any) -> None:
        self.value = value
        self.restype: Any = None
        self.argtypes: Any = None

    def __call__(self, *args: Any) -> Any:
        return self.value


def _fake_lib(impl: str) -> SimpleNamespace:
    return SimpleNamespace(
        l11do8_abi_version=_FakeFn(do8._ABI_VERSION),
        l11do8_e5_plus_launch=_FakeFn(0),
        l11do8_e5_only_launch=_FakeFn(0),
        l11do8_wl_build_launch=_FakeFn(0),
        l11do8_wl_cap=_FakeFn(1),
        l11do8_plan=_FakeFn(0),
        l11do8_drain_last_error=_FakeFn(0),
        l11do8_impl=_FakeFn(impl.encode()),
    )


def _rand_mask(n: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """int8 [n, 512], every byte uniform in 0..7 (all three keep planes, every combination)."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, 8, (n, D), generator=g, device=device, dtype=I8)


def _keep_mask(n: int, p: float, seed: int) -> torch.Tensor:
    """A writer-like mask: three independent keep planes (keep = u >= p), bits 0 / 1 / 2."""
    g = torch.Generator().manual_seed(seed)
    b = (torch.rand((3, n, D), generator=g) >= p).to(I8)
    return b[0] | (b[1] << 1) | (b[2] << 2)


def _impl_strings_from_source(src: str, nib: bool) -> Dict[str, str]:
    """l11do8_impl()'s two return strings in l11_do8.cu ('only' = the #if L11DO8_ONLY branch, 'plus' = #else), with
    L11DO8_SRC_SHA256 = "x" and L11DO8_MKB_TAG = "+mkb=nib" (nib) or "" (adjacent literals concatenated)."""
    start = src.index('extern "C" const char* l11do8_impl(void) {')
    body = src[start:src.index("\n}\n", start)]
    rets = re.findall(r"return (.*?);", body, re.S)
    assert len(rets) == 2, rets
    vals = []
    for r in rets:
        out = ""
        for tok in re.findall(r'"(?:[^"\\]|\\.)*"|[A-Za-z_]\w*', r):
            if tok.startswith('"'):
                out += ast.literal_eval(tok)
            elif tok == "L11DO8_SRC_SHA256":
                out += "x"
            elif tok == "L11DO8_MKB_TAG":
                out += "+mkb=nib" if nib else ""
            elif tok == "L11DO8_MX6_TAG":  # r16 MX6 row: "" unless -DL11DO8_MX6=1 (the MX6 .so only)
                out += ""
            else:
                raise AssertionError(f"unexpected token {tok!r} in l11do8_impl")
        vals.append(out)
    return {"only": vals[0], "plus": vals[1]}


def _thl_or_none() -> Any:
    try:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl
    except Exception:  # noqa: BLE001 -- no Triton in this interpreter
        return None
    return thl


class _EnvReset(unittest.TestCase):
    """Every test starts with the NIB / VD / ONLY / SRC knobs unset, the NIB writers' knobs on (_WRITERS_ON) and DO8,
    L11 and the contract reset."""

    def setUp(self) -> None:
        self._saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(_WRITERS_ON)
        do8.set_diet_for_testing(True)  # ship15.1 item (89) DO8 DIET at its gin default (on): the build NIB v2 ran on
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(True)
        l11.reset_for_testing()
        l11.set_requested_for_testing(None)
        nibm.reset_for_testing()

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        do8.set_requested_for_testing(None)
        do8.set_diet_for_testing(None)
        do8.reset_for_testing(unload=True)
        l11.set_requested_for_testing(None)
        l11.reset_for_testing()
        nibm.reset_for_testing()


# ---------------------------------------------------------------------------------------------------- CPU: contract
class NibContractCpuTest(unittest.TestCase):
    def test_pack_unpack_exact_on_random_masks(self) -> None:
        for n in (1, 2, 7, 8, 9, 63, 64, 65, 1000):
            for seed in range(4):
                m = _rand_mask(n, 1000 * n + seed)
                p = nibm.pack_ref(m)
                self.assertEqual((p.dtype, tuple(p.shape), tuple(p.stride())), (U8, (n, 256), (256, 1)))
                self.assertTrue(nibm.is_nib(p))
                self.assertEqual(int((p & 0x88).count_nonzero()), 0)  # bits 3 and 7 of every byte are zero
                # layout A: byte 4l+k of a row = col 8l+k's bits (0-2) | col 8l+4+k's bits << 4
                mv = m.view(U8).view(n, 64, 8)
                self.assertTrue(torch.equal(p, (mv[:, :, 0:4] | (mv[:, :, 4:8] << 4)).reshape(n, 256)))
                back = nibm.unpack(p)
                self.assertEqual((back.dtype, tuple(back.shape), tuple(back.stride())), (I8, (n, D), (D, 1)))
                self.assertTrue(torch.equal(back, m), (n, seed))

    def test_writer_like_keep_masks_round_trip(self) -> None:
        for p in (1e-4, 0.1, 0.5):
            for n in (1, 4097):
                m = _keep_mask(n, p, seed=n)
                self.assertTrue(torch.equal(nibm.unpack(nibm.pack_ref(m)), m), (p, n))

    def test_unpack_formula_on_every_byte_value_and_layout(self) -> None:
        """review R1-F6: unpack writes the two nibble planes into the strided halves of a preallocated result; the
        values are the formula's on every byte (bits 3 and 7 dropped), for contiguous, strided and empty inputs."""
        def formula(p: torch.Tensor) -> torch.Tensor:
            n = int(p.shape[0])
            q = p.contiguous().view(n, 64, 4)
            return torch.cat([q & 7, (q >> 4) & 7], dim=2).reshape(n, D).view(I8)

        rows = torch.arange(256, dtype=torch.int32).view(1, 256)
        p = torch.cat([rows, rows.flip(1), (rows * 37 + 11) % 256]).to(U8)
        for src in (p, torch.cat([p, p], dim=1)[:, :256], p[:0]):
            got = nibm.unpack(src)
            self.assertEqual((got.dtype, tuple(got.shape)), (I8, (int(src.shape[0]), D)))
            self.assertTrue(got.is_contiguous())
            self.assertTrue(torch.equal(got, formula(src)))
        self.assertEqual(int((nibm.unpack(p).view(U8) > 7).sum()), 0)

    def test_as_int8_line_is_one_tagged_write(self) -> None:
        """review R2-F3: the first unpack per consumer prints through ops.utils.arm_print_all_ranks (one tagged write
        per rank); the greppable prefix is unchanged; without that module a plain print keeps the line."""
        nibm.reset_for_testing()
        p = nibm.pack_ref(_rand_mask(4, 11))
        with _Capture() as cap:
            nibm.as_int8(p, "unit-a")
        self.assertRegex(cap.text, r"\[lnmd-mask-nib\] LN\*U\*dropout keep mask NIB: exact unpack to int8 for unit-a "
                                   r"\(a non-DO8-NIB consumer read a NIB mask; counted, unpack=1\) \[rank \w+\]")
        with mock.patch.dict(sys.modules, {"generative_recommenders.ops.utils": None}), _Capture() as cap:
            nibm.as_int8(nibm.pack_ref(_rand_mask(4, 12)), "unit-b")
        self.assertIn("exact unpack to int8 for unit-b (a non-DO8-NIB consumer read a NIB mask; counted, unpack=2) "
                      "[rank ", cap.text)
        nibm.reset_for_testing()

    def test_high_bits_break_the_round_trip(self) -> None:
        """Bits 3-7 are not representable (the writers never set them); the DO8 self-checks gate the round trip."""
        m = _rand_mask(4, 3)
        m[2, 17] = 8
        self.assertFalse(torch.equal(nibm.unpack(nibm.pack_ref(m)), m))

    def test_layout_ok(self) -> None:
        n = 9
        p = nibm.pack_ref(_rand_mask(n, 5))
        self.assertTrue(nibm.layout_ok(p, n))
        self.assertFalse(nibm.layout_ok(p, n - 1))
        self.assertFalse(nibm.layout_ok(nibm.pack_ref(_rand_mask(2 * n, 6))[::2], n))  # row stride 512
        buf = torch.zeros(n * 256 + 4, dtype=U8)
        self.assertFalse(nibm.layout_ok(buf[4:].view(n, 256), n))  # base not 16-B aligned
        self.assertFalse(nibm.layout_ok(_rand_mask(n, 5), n))  # today's int8 [n, 512]
        self.assertFalse(nibm.layout_ok(torch.zeros(n, 256, dtype=I8), n))
        self.assertFalse(nibm.layout_ok(None, n))


# ---------------------------------------------------------------------------------------------------- CPU: build
class Do8NibBuildCpuTest(_EnvReset):
    def test_flags_define_only_for_the_only_build(self) -> None:
        """NIB v2: -DL11_MKB_NIB=1 (last flag) only on the 'only' build; the 'plus' flags are today's for every knob
        value and source."""
        def legacy(src: bytes, only: bool) -> List[str]:  # today's _flags, written out
            sha = hashlib.sha256(src).hexdigest()
            return [f"--offload-arch={do8._ARCH}", *do8._BASE_FLAGS, *do8._SCHED_FLAGS[do8.sched()],
                    f'-DL11DO8_SRC_SHA256="{sha}"'] + (["-DL11DO8_ONLY=1"] if only else [])

        switch = b"#ifndef L11_MKB_NIB\n#define L11_MKB_NIB 0\n#endif\n#if L11_MKB_NIB\n#endif\n"
        sources = (
            (b"// a source without the switch", False),
            (b"#ifndef L11_MKB_NIB\n#define L11_MKB_NIB 0\n#endif\n", False),  # the default only, no #if switch
            (b"// mentions L11_MKB_NIB in a comment only\n", False),  # review R2-F8: naming the macro is not enough
            (switch, True),
        )
        for knob in (None, "", "0", "1", " 1 ", "true", "2", "yes"):
            for src, capable in sources:
                for only in (False, True):
                    with _env({KNOB: knob}):
                        got = do8._flags(src, only)
                    want = only and knob is not None and knob.strip() == "1" and capable
                    self.assertEqual(got, legacy(src, only) + (["-DL11_MKB_NIB=1"] if want else []),
                                     (knob, src, only))

    def test_flags_compose_with_the_diet_off(self) -> None:
        """ship15.1 item (89) DO8 DIET off (its kill switch, DLRM_HSTU_FP8_L11_DO8_DIET != 1) with the NIB knob on:
        -DL11DO8_DIET=0 precedes the 'only' define and -DL11_MKB_NIB=1 stays the last flag of the NIB 'only' build;
        the 'plus' build gains the diet define only."""
        switch = b"#ifndef L11_MKB_NIB\n#define L11_MKB_NIB 0\n#endif\n#if L11_MKB_NIB\n#endif\n"
        with _env({KNOB: "1"}):
            on_plus, on_only = do8._flags(switch, False), do8._flags(switch, True)
            do8.set_diet_for_testing(False)
            try:
                off_plus, off_only = do8._flags(switch, False), do8._flags(switch, True)
            finally:
                do8.set_diet_for_testing(True)
        self.assertEqual(on_only, on_plus + ["-DL11DO8_ONLY=1", "-DL11_MKB_NIB=1"])
        self.assertEqual(off_plus, on_plus + ["-DL11DO8_DIET=0"])
        self.assertEqual(off_only, on_plus + ["-DL11DO8_DIET=0", "-DL11DO8_ONLY=1", "-DL11_MKB_NIB=1"])

    def test_nib_build_needs_the_nib_writers_armable(self) -> None:
        """review R1-F3: with a knob that keeps every keep mask int8 the 'only' build stays int8 (read as
        triton_hstu_linear reads it: maskgen / v2 off only on "0" or unset, mask16 on only on "1", the producer
        out-proj wgrad mode); the 'plus' build never gets the define."""
        with _env({KNOB: "1", VD_KNOB: "1"}):
            self.assertIsNone(do8._nib_writers_off())
            self.assertTrue(do8.nib_build_requested())
            for env, armable in (
                ({MASKGEN: None}, False), ({MASKGEN: "0"}, False), ({MASKGEN: "2"}, True), ({MASKGEN: ""}, True),
                ({MASK16: None}, False), ({MASK16: "0"}, False), ({MASK16: "true"}, False), ({MASK16: "1"}, True),
                ({FWD_V2: None}, False), ({FWD_V2: "0"}, False), ({FWD_V2: "yes"}, True),
                ({OUT_IMPL: "producer"}, False), ({OUT_IMPL: "nn"}, True), ({OUT_IMPL: "tn"}, True),
                ({OUT_IMPL: "triton"}, True),
            ):
                with _env(env):
                    self.assertEqual(do8._nib_writers_off() is None, armable, env)
                    self.assertEqual(do8.nib_build_requested(), armable, env)
                    src = do8._SOURCE.read_bytes()
                    self.assertNotIn("-DL11_MKB_NIB=1", do8._flags(src), env)
                    self.assertEqual("-DL11_MKB_NIB=1" in do8._flags(src, True), armable, env)
                    if not armable:
                        self.assertIn(next(iter(env)), do8._nib_writers_off())
        with _env({KNOB: None, VD_KNOB: "1", MASKGEN: None}):
            self.assertFalse(do8.nib_build_requested())  # knob off: False whatever the writers' knobs say

    def test_nib_build_requested_follows_the_source_selector(self) -> None:
        self.assertIn(do8._NIB_MARKER, do8._SOURCE.read_bytes())
        self.assertEqual(do8._NIB_MARKER, b"#if L11_MKB_NIB")
        self.assertNotIn(b"L11_MKB_NIB", do8._SOURCE_NOVD.read_bytes())  # the pre-diet source has no NIB reader
        cases = (
            ({KNOB: "1", VD_KNOB: "1"}, True),
            ({KNOB: "1", VD_KNOB: None}, False),
            ({KNOB: "1", VD_KNOB: "0"}, False),
            ({KNOB: None, VD_KNOB: "1"}, False),
            ({KNOB: "0", VD_KNOB: "1"}, False),
            ({KNOB: "1", VD_KNOB: None, SRC_ENV: str(do8._SOURCE)}, True),  # A/B override that has the switch
            ({KNOB: "1", VD_KNOB: "1", SRC_ENV: str(do8._SOURCE_NOVD)}, False),
            ({KNOB: "1", VD_KNOB: "1", SRC_ENV: "/nonexistent/l11_do8_x.cu"}, False),
        )
        for env, want in cases:
            with _env(env):
                self.assertEqual(do8.nib_build_requested(), want, env)
                src = do8._source().read_bytes() if do8._source().is_file() else b""
                self.assertNotIn("-DL11_MKB_NIB=1", do8._flags(src), env)
                self.assertEqual("-DL11_MKB_NIB=1" in do8._flags(src, True), want, env)

    def test_plus_build_is_the_knob_off_build_only_build_separated(self) -> None:
        """NIB v2: the 'plus' cache key and hipcc command are the knob-off build's (same .so, same cache entry); the
        NIB 'only' build has its own cache key."""
        out = Path("/nonexistent/o.so")
        with mock.patch.object(l11, "_toolchain", return_value="hipcc test"), \
                mock.patch.object(l11, "_hipcc", return_value="/opt/rocm/bin/hipcc"):
            with _env({KNOB: None, VD_KNOB: "1"}):
                k_off = (do8.cache_key(do8._SOURCE), do8.cache_key(do8._SOURCE, only=True))
                k_novd = do8.cache_key(do8._SOURCE_NOVD)
                cmd_off = (do8.build_command(do8._SOURCE, out), do8.build_command(do8._SOURCE, out, only=True))
            with _env({KNOB: "1", VD_KNOB: "1"}):
                k_on = (do8.cache_key(do8._SOURCE), do8.cache_key(do8._SOURCE, only=True))
                self.assertEqual(do8.cache_key(do8._SOURCE_NOVD), k_novd)  # no switch: same flags, same key
                cmd_on = (do8.build_command(do8._SOURCE, out), do8.build_command(do8._SOURCE, out, only=True))
        self.assertEqual(k_off[0], k_on[0])
        self.assertEqual(cmd_off[0], cmd_on[0])
        self.assertNotIn("-DL11_MKB_NIB=1", cmd_on[0])
        self.assertNotEqual(k_off[1], k_on[1])
        self.assertEqual(cmd_on[1], cmd_off[1][:-2] + ["-DL11_MKB_NIB=1"] + cmd_off[1][-2:])  # before "-o <out>"
        self.assertEqual(len({*k_off, *k_on}), 3)

    def test_cu_switch_pins(self) -> None:
        src, novd = do8._SOURCE.read_text(), do8._SOURCE_NOVD.read_text()
        default = "#ifndef L11_MKB_NIB\n#define L11_MKB_NIB 0\n#endif\n"
        self.assertEqual(src.count(default), 1)
        uses = [src.index(t) for t in ("!L11_MKB_NIB", "&& L11_MKB_NIB", "#if L11_MKB_NIB")]
        self.assertLess(src.index(default), min(uses))  # the default precedes every use (incl. the impl tag)
        self.assertEqual(src.count('#if L11_MKB_NIB\n#define L11DO8_MKB_TAG "+mkb=nib"\n#else\n'
                                   '#define L11DO8_MKB_TAG ""\n#endif\n'), 1)
        self.assertIn('"' + do8._NIB_IMPL_TAG + '"', src)
        self.assertIn("pmb[sl] = (uint32_t)l11_llvm_buffer_load_i32(rM, r * 256 + 4 * lane, 0, 0);", src)
        self.assertIn("rM = make_rsrc(ca.MK + (size_t)row0 * 256, (uint32_t)rv * 256u);", src)
        self.assertIn("l11vd_mask_nib<0>(mkn)", src)
        # the int8 code paths are still there, the mangled instantiation and DO8-ONLY switches unchanged
        self.assertIn("pm[sl] = l11_llvm_buffer_load_v2i32(rM, r * 512 + 8 * lane, 0, 0);", src)
        self.assertEqual(src.count("#if L11DO8_ONLY"), 3)  # + (r16 MX6) the MX6 entry's launch name
        self.assertEqual(src.count("(l11_outproj_lnmd_bwd_do8<1, 2, 0, 0, 1, 2, 1, 0, 1, 2, L11DO8_ONLY ? 2 : 1, 0, 2, 0>)"), 1)
        self.assertEqual(src.count("(l11_outproj_lnmd_bwd_do8_mx6<1, 2, 0, 0, 1, 2, 1, 0, 1, 2, L11DO8_ONLY ? 2 : 1, 0, 2, 0>)"), 1)
        self.assertNotIn("L11_MKB_NIB", novd)
        self.assertNotIn("+mkb=nib", novd)

    def test_impl_strings_off_identical_on_tagged(self) -> None:
        src = do8._SOURCE.read_text()
        off, on = _impl_strings_from_source(src, nib=False), _impl_strings_from_source(src, nib=True)
        self.assertEqual(off, {"only": _IMPL_ONLY, "plus": _IMPL_PLUS})  # NIB off: ship15's bytes
        self.assertEqual(on, {"only": _IMPL_ONLY_NIB, "plus": _IMPL_PLUS_NIB})
        for s in off.values():
            self.assertEqual(do8._mask_format_of(s), "int8")
        for s in on.values():
            self.assertEqual(do8._mask_format_of(s), "nib")
        self.assertIn("EM 2 only", on["only"])  # _load_only's EM 2 check still holds for the NIB 'only' build


# ---------------------------------------------------------------------------------------------------- CPU: state
class Do8NibStateCpuTest(_EnvReset):
    def test_mask_format_parsing(self) -> None:
        self.assertEqual((do8.mask_format(), do8.only_mask_format()), ("int8", "int8"))  # nothing loaded
        for impl, want in ((_IMPL_PLUS, "int8"), (_IMPL_PLUS_NIB, "nib"), ("", "int8")):
            with mock.patch.object(do8, "_lib", object()), mock.patch.object(do8, "_impl", impl):
                self.assertEqual(do8.mask_format(), want)
                self.assertEqual(do8.counters()["mask_format"], want)
                self.assertEqual(do8.only_mask_format(), "int8")  # the 'plus' .so says nothing about 'only'
            with mock.patch.object(do8, "_lib", None), mock.patch.object(do8, "_impl", impl):
                self.assertEqual(do8.mask_format(), "int8")  # the impl of a .so that is not loaded does not count
        for impl, want in ((_IMPL_ONLY, "int8"), (_IMPL_ONLY_NIB, "nib"), ("", "int8")):
            with mock.patch.object(do8, "_lib", object()), mock.patch.object(do8, "_impl", _IMPL_PLUS), \
                    mock.patch.object(do8, "_only_lib", object()), mock.patch.object(do8, "_impl_only", impl):
                self.assertEqual(do8.only_mask_format(), want)
                self.assertEqual(do8.counters()["mask_only_format"], want)
                self.assertEqual(do8.mask_format(), "int8")
            with mock.patch.object(do8, "_only_lib", None), mock.patch.object(do8, "_impl_only", impl):
                self.assertEqual(do8.only_mask_format(), "int8")
        c = do8.counters()
        self.assertEqual((c["mask_declined"], c["mask_int8_plus"], c["mask_nib_plus"], c["mask_fallback"]),
                         (0, 0, 0, 0))

    def _ready(self, **over: Any) -> Tuple[bool, Optional[str]]:
        st = dict(knob="1", do8_req=True, do8_pre=True, broken=None, only_knob="1", only_pre=True, only_broken=None,
                  only_impl=_IMPL_ONLY_NIB, only_lib=True, plus_impl=_IMPL_PLUS, l11_req=True, l11_pre=True,
                  l11_broken=None)
        st.update(over)
        do8.set_requested_for_testing(st["do8_req"])
        do8.set_prebuilt_for_testing(st["do8_pre"])
        do8.set_only_prebuilt_for_testing(st["only_pre"])
        l11.set_requested_for_testing(st["l11_req"])
        l11.set_prebuilt_for_testing(st["l11_pre"])
        with _env({KNOB: st["knob"], VD_KNOB: "1", ONLY: st["only_knob"]}), \
                mock.patch.object(do8, "_broken", st["broken"]), \
                mock.patch.object(do8, "_only_broken", st["only_broken"]), \
                mock.patch.object(do8, "_lib", object()), mock.patch.object(do8, "_impl", st["plus_impl"]), \
                mock.patch.object(do8, "_only_lib", object() if st["only_lib"] else None), \
                mock.patch.object(do8, "_impl_only", st["only_impl"]), \
                mock.patch.object(l11, "_broken", st["l11_broken"]):
            return do8.nib_ready(), do8._nib_not_ready()

    def test_nib_ready_needs_every_v2_condition(self) -> None:
        self.assertEqual(self._ready(), (True, None))
        for over, reason in (
            ({"knob": None}, "is not 1"), ({"knob": "0"}, "is not 1"),
            ({"do8_req": False}, "DO8 off"), ({"do8_pre": False}, "DO8 off"), ({"broken": "latched"}, "DO8 off"),
            ({"only_knob": None}, "DO8-ONLY off"), ({"only_knob": "0"}, "DO8-ONLY off"),
            ({"only_pre": False}, "not prebuilt or latched"), ({"only_broken": "latched"}, "not prebuilt or latched"),
            ({"only_impl": _IMPL_ONLY}, "reads the int8"), ({"only_lib": False}, "reads the int8"),
            ({"l11_req": False}, "L11"), ({"l11_pre": False}, "L11"), ({"l11_broken": "latched"}, "L11"),
        ):
            ok, why = self._ready(**over)
            self.assertFalse(ok, over)
            self.assertIn(reason, why or "", over)
        # NIB v2: only the 'only' .so reads NIB; the 'plus' .so's format plays no part
        self.assertEqual(self._ready(plus_impl=_IMPL_PLUS_NIB), (True, None))
        self.assertFalse(self._ready(plus_impl=_IMPL_PLUS_NIB, only_impl=_IMPL_ONLY)[0])

    def test_nib_ready_reads_nothing_else_with_the_knob_off(self) -> None:
        boom = mock.MagicMock(side_effect=AssertionError("read with the knob off"))
        with _env({KNOB: None}), mock.patch.object(do8, "requested", boom), \
                mock.patch.object(do8, "only_requested", boom), mock.patch.object(do8, "only_mask_format", boom), \
                mock.patch.object(l11, "enabled", boom):
            self.assertFalse(do8.nib_ready())
        self.assertEqual(boom.call_count, 0)

    def _load(self, impl: str, env: Dict[str, Optional[str]]) -> Optional[str]:
        do8.reset_for_testing(unload=True)
        with _env(env), \
                mock.patch.object(do8, "build", return_value=(Path("/nonexistent/x/libl11_do8.so"),
                                                               {"kernels_raw": [], "vgpr": 256})), \
                mock.patch.object(ctypes, "CDLL", return_value=_fake_lib(impl)):
            try:
                do8._load()
            except RuntimeError as error:
                return str(error)
        return None

    def test_load_requires_an_int8_plus_so(self) -> None:
        """NIB v2: the 'plus' .so always reads int8; a "+mkb=nib" 'plus' impl raises whatever the knob says (the
        prebuild then latches DO8 exactly as for any 'plus' load failure)."""
        err = "'plus' .so reads keep mask format nib; the 'plus' build is always int8"
        for impl, env, want in (
            (_IMPL_PLUS, {KNOB: "1", VD_KNOB: "1"}, None),
            (_IMPL_PLUS, {KNOB: None, VD_KNOB: "1"}, None),
            (_IMPL_PLUS, {KNOB: "1", VD_KNOB: None}, None),
            (_IMPL_PLUS_NIB, {KNOB: "1", VD_KNOB: "1"}, err),
            (_IMPL_PLUS_NIB, {KNOB: None, VD_KNOB: "1"}, err),
        ):
            got = self._load(impl, env)
            if want is None:
                self.assertIsNone(got, (impl[-30:], env))
                self.assertIsNotNone(do8._lib)
            else:
                self.assertIn(want, got or "", (impl[-30:], env))
                self.assertIsNone(do8._lib)
            self.assertEqual(do8.mask_format(), "int8", (impl[-30:], env))

    def test_load_only_requires_the_requested_format(self) -> None:
        """NIB v2: the 'only' .so must read the format its flags requested (nib_build_requested()); no comparison with
        the 'plus' .so, which is int8 in every case."""
        for env, only, err in (
            ({KNOB: "1", VD_KNOB: "1"}, _IMPL_ONLY_NIB, None),
            ({KNOB: None, VD_KNOB: "1"}, _IMPL_ONLY, None),
            ({KNOB: "1", VD_KNOB: "1", MASKGEN: "0"}, _IMPL_ONLY, None),  # writers off: the int8 'only' is requested
            ({KNOB: "1", VD_KNOB: "1"}, _IMPL_ONLY, "'only' .so reads keep mask format int8, the build requested nib"),
            ({KNOB: None, VD_KNOB: "1"}, _IMPL_ONLY_NIB, "'only' .so reads keep mask format nib, the build requested "
                                                         "int8"),
            ({KNOB: "1", VD_KNOB: "1", MASKGEN: "0"}, _IMPL_ONLY_NIB, "reads keep mask format nib, the build "
                                                                      "requested int8"),
        ):
            self.assertIsNone(self._load(_IMPL_PLUS, env))
            libs = {"/x/only.so": _fake_lib(only)}

            def fake_build(source: Any = None, only: bool = False) -> Tuple[Path, Dict[str, Any]]:
                return Path("/x/only.so"), {"kernels_raw": [], "vgpr": 251}

            with _env(env), mock.patch.object(do8, "build", side_effect=fake_build), \
                    mock.patch.object(ctypes, "CDLL", side_effect=lambda p: libs[p]):
                if err is None:
                    do8._load_only()
                    self.assertEqual(do8.only_impl(), only)
                    self.assertEqual(do8.only_mask_format(), "nib" if "+mkb=nib" in only else "int8")
                else:
                    with self.assertRaises(RuntimeError) as cm:
                        do8._load_only()
                    self.assertIn(err, str(cm.exception))
                    self.assertIsNone(do8._only_lib)
                    self.assertEqual(do8.only_mask_format(), "int8")
            self.assertEqual(do8.mask_format(), "int8", env)

    def _prebuild(self, env: Dict[str, Optional[str]], plus: str, only: str) -> str:
        do8.reset_for_testing(unload=True)
        libs = {"/x/plus/libl11_do8.so": _fake_lib(plus), "/x/only/libl11_do8_only.so": _fake_lib(only)}
        summary = {"kernels_raw": [], "vgpr": 256, "vgpr_spill": 0, "scratch": 0, "lds": 163840, "sgpr_spill": 33,
                   "sched": "default"}

        def fake_build(source: Any = None, only: bool = False) -> Tuple[Path, Dict[str, Any]]:
            return Path("/x/only/libl11_do8_only.so" if only else "/x/plus/libl11_do8.so"), dict(summary)

        with _env(env), mock.patch.object(do8, "build", side_effect=fake_build), \
                mock.patch.object(ctypes, "CDLL", side_effect=lambda p: libs[p]), \
                mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2.1"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(torch.cuda, "synchronize", return_value=None), \
                mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                mock.patch.object(do8, "self_check_only", return_value=(True, "only detail")), \
                _Capture() as cap:
            self.assertTrue(do8.prebuild(True, True, self_check_fn=lambda: (True, "plus detail")))
        return cap.text

    def test_prebuild_lines_keep_their_text_and_end_with_the_mask_format(self) -> None:
        for env, only, want_only in (
            ({KNOB: None, VD_KNOB: "1", ONLY: "1"}, _IMPL_ONLY, "int8"),
            ({KNOB: "1", VD_KNOB: "1", ONLY: "1"}, _IMPL_ONLY_NIB, "nib(uint8[M,256])"),
        ):
            text = self._prebuild(env, _IMPL_PLUS, only)
            pre = re.escape(f"armed OK: {do8.FAMILY} prebuild before run_start: built+loaded+self-check OK ")
            self.assertRegex(text, pre + r"[0-9.]+s \(VGPR 256, spill 0, scratch 0, LDS 163840, SGPR spill 33, "
                             r"sched=default; so plus; plus detail\) mask=int8 \[rank \w+\]")  # 'plus': int8, always
            pre_o = re.escape(f"armed OK: {do8.FAMILY_ONLY} prebuild before run_start: built+loaded+self-check OK ")
            self.assertRegex(text, pre_o + r"[0-9.]+s \(EM 2: VGPR 256, spill 0, scratch 0, LDS 163840, SGPR spill 33, "
                             r"sched=default; so only; only detail\) mask=" + re.escape(want_only) + r" \[rank \w+\]")
            self.assertEqual(text.count(" mask=nib("), int(want_only != "int8"))  # only on the 'only' line
            for inert in ("has no L11_MKB_NIB reader", "NIB writers cannot arm", "DO8-ONLY is off"):
                self.assertNotIn(inert, text)
            if env[KNOB] is None:
                self.assertNotIn("NIB", text)  # knob off: ship15's lines + the " mask=int8" segments only
            self.assertEqual(do8.mask_format(), "int8")
            self.assertEqual(do8.only_mask_format(), want_only.split("(")[0])
            l11.set_requested_for_testing(True)
            l11.set_prebuilt_for_testing(True)
            with _env(env):
                self.assertEqual(do8.nib_ready(), want_only != "int8")
            l11.set_requested_for_testing(None)
            l11.reset_for_testing()

    def test_prebuild_says_when_nib_is_requested_but_the_source_has_no_reader(self) -> None:
        text = self._prebuild({KNOB: "1", VD_KNOB: None, ONLY: None}, _IMPL_PLUS, _IMPL_ONLY)
        self.assertEqual(text.count(f"{do8.FAMILY}: {nibm.FAMILY} requested ({KNOB}=1) but l11_do8_novd.cu has no "
                                    "L11_MKB_NIB reader"), 1)
        self.assertRegex(text, r"plus detail\) mask=int8 \[rank \w+\]")
        self.assertFalse(do8.nib_ready())

    def test_prebuild_says_when_nib_is_requested_but_the_writers_cannot_arm(self) -> None:
        """review R1-F3: the knob on with the diet source, but item (28)(e)'s maskgen off: int8 .so, one line."""
        text = self._prebuild({KNOB: "1", VD_KNOB: "1", ONLY: "1", MASKGEN: "0"}, _IMPL_PLUS, _IMPL_ONLY)
        self.assertEqual(text.count(f"{do8.FAMILY}: {nibm.FAMILY} requested ({KNOB}=1) but the NIB writers cannot "
                                    "arm in this process (DLRM_HSTU_FP8_LNMD_FWD_MASKGEN is off"), 1)
        self.assertNotIn("has no L11_MKB_NIB reader", text)
        self.assertRegex(text, r"plus detail\) mask=int8 \[rank \w+\]")
        self.assertRegex(text, r"only detail\) mask=int8 \[rank \w+\]")
        self.assertEqual((do8.mask_format(), do8.only_mask_format()), ("int8", "int8"))
        self.assertFalse(do8.nib_ready())
        self.assertEqual(do8.counters()["mask_fallback"], 0)

    def test_prebuild_says_when_nib_is_requested_but_do8_only_is_off(self) -> None:
        """NIB v2: only the DO8-ONLY 'only' .so reads NIB; with DO8-ONLY off nothing reads it: one line, no 'only'
        build, nib_ready() False."""
        text = self._prebuild({KNOB: "1", VD_KNOB: "1", ONLY: None}, _IMPL_PLUS, _IMPL_ONLY_NIB)
        self.assertEqual(text.count(f"{do8.FAMILY}: {nibm.FAMILY} requested ({KNOB}=1) but DO8-ONLY is off ({ONLY} "
                                    "!= 1): only its 'only' .so reads NIB"), 1)
        self.assertNotIn(do8.FAMILY_ONLY, text)
        self.assertIsNone(do8._only_lib)
        self.assertRegex(text, r"plus detail\) mask=int8 \[rank \w+\]")
        self.assertFalse(do8.nib_ready())

    def test_exit_nib_counters_line_is_knob_on_only(self) -> None:
        """Review N2: with the knob on, _final (an EXIT_HOOKS entry) adds one "[l11-do8] nib counters: ... final=1" line
        after the DO8 counters line: the loaded 'only' format, int8_plus, nib_plus, every "E5 mask format" decline and
        the prebuild fallback. Knob off: exactly the DO8 counters line, as without NIB. The new line matches none of the
        line patterns the kits judge (the DO8 counters / declined rule, the prebuild ' mask=' segments, NIB lines)."""
        do8._STATE["events"] = 3
        do8._NIB_COUNTS.update(int8_plus=5, nib_plus=2)
        do8._DECLINED.update({"E5 mask": 3})
        with mock.patch.object(do8, "_only_lib", object()), mock.patch.object(do8, "_impl_only", _IMPL_ONLY_NIB):
            with _env({KNOB: None}), _Capture() as off:
                do8._final()
            with _env({KNOB: "0"}), _Capture() as zero:
                do8._final()
            with _env({KNOB: "1"}), _Capture() as on:
                do8._final()
            with _env({KNOB: "1"}), mock.patch.object(do8, "_nib_fallback", "x"), \
                    mock.patch.object(do8, "_only_lib", None), _Capture() as fell:
                do8._final()
        want = ("[l11-do8] nib counters: only_mask_format=nib int8_plus=5 nib_plus=2 mask_declined=3 fallback=0 "
                "final=1")
        self.assertEqual(off.text, zero.text)
        self.assertEqual(off.text.splitlines(), [ln for ln in on.text.splitlines() if "nib counters" not in ln])
        self.assertEqual(len(off.text.splitlines()), 1)
        self.assertIn("[l11-do8] counters: events=3 ", off.text)
        self.assertNotIn("nib", off.text)
        lines = on.text.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith(want), lines[1])
        self.assertIn("[l11-do8] nib counters: only_mask_format=int8 int8_plus=5 nib_plus=2 mask_declined=3 "
                      "fallback=1 final=1", fell.text)
        for rx in (r"\[l11-do8\] counters: .*declined=\{[^}]", r"\[l11-do8\] (only )?counters:", r" mask=",
                   r"keep mask NIB", r"\[lnmd-mask-nib\]", r"\+mkb=nib"):
            self.assertIsNone(re.search(rx, lines[1]), rx)
        do8._STATE["events"] = 0
        with _env({KNOB: "1"}), _Capture() as idle:
            do8._final()
        self.assertEqual(idle.text, "")  # no DO8 event in this process: no line at all, as without NIB


# ---------------------------------------------------------------------------------------------------- CPU: fallback
class Do8NibPrebuildFallbackCpuTest(_EnvReset):
    """review R1-F4, NIB v2: only the 'only' build can be NIB. A failure of the NIB 'only' build at prebuild (build /
    gate, impl format, self-check exception or mismatch) latches nothing: one NIB DISABLED line, then the int8 'only'
    build + self-check (DO8-ONLY on, nib_ready() False, nib_build_requested() False for the process); 'plus' is built
    once and never touched. A 'plus' failure latches DO8 exactly as without NIB. Fake builds: NIB or int8 as the REAL
    _flags say for the mode (so never NIB for 'plus')."""

    FALLBACK = f"{do8.FAMILY}: {nibm.FAMILY} DISABLED ({KNOB}): the NIB reader build failed at prebuild ("
    PLUS_ARMED = "armed OK: L11 DO8 dO8/E8 emission prebuild before run_start"
    ONLY_ARMED = "armed OK: L11 DO8 dO8/E8 emission only prebuild before run_start"

    def _run(self, fail: Any, env: Optional[Dict[str, Optional[str]]] = None,
             check_exc: Optional[BaseException] = None) -> Tuple[Any, str]:
        """prebuild(True, True) with 'only' on. fail: a set of (stage, mode, nib) with stage in build / impl / check /
        mismatch, mode in plus / only, nib True (the NIB build) or False (the int8 build); "impl" flips the impl's NIB
        tag. check_exc: raised by the NIB 'only' self-check instead. Returns (prebuild's result or the exception,
        captured text)."""
        do8.reset_for_testing(unload=True)
        builds: List[Tuple[str, bool]] = []

        def fake_build(source: Any = None, only: bool = False) -> Tuple[Path, Dict[str, Any]]:
            mode = "only" if only else "plus"
            nib = do8._NIB_DEFINE in do8._flags((source or do8._source()).read_bytes(), only)
            builds.append((mode, nib))
            if ("build", mode, nib) in fail:
                raise RuntimeError(f"resource check: {mode} VGPR spill 2, scratch 8 B/lane (need 0/0)")
            tagged = nib != (("impl", mode, nib) in fail)
            name = f"/x/{mode}{'_nib' if nib else ''}/libl11_do8{'_only' if only else ''}.so"
            libs[name] = _fake_lib({("plus", True): _IMPL_PLUS_NIB, ("plus", False): _IMPL_PLUS,
                                    ("only", True): _IMPL_ONLY_NIB, ("only", False): _IMPL_ONLY}[(mode, tagged)])
            return Path(name), {"kernels_raw": [], "vgpr": 256, "vgpr_spill": 0, "scratch": 0, "lds": 163840,
                                "sgpr_spill": 33, "sched": "default"}

        def check(mode: str) -> Tuple[bool, str]:
            nib = (do8.mask_format() if mode == "plus" else do8.only_mask_format()) == "nib"
            if mode == "only" and nib and check_exc is not None:
                raise check_exc
            if ("check", mode, nib) in fail:
                raise ValueError(f"{mode} check blew up")
            if ("mismatch", mode, nib) in fail:
                return False, f"{mode} dO8 mism 3"
            return True, f"{mode} detail nib={int(nib)}"

        libs: Dict[str, Any] = {}
        self.builds = builds
        out: Any = None
        with _env(dict({KNOB: "1", VD_KNOB: "1", ONLY: "1"}, **(env or {}))), \
                mock.patch.object(do8, "build", side_effect=fake_build), \
                mock.patch.object(ctypes, "CDLL", side_effect=lambda p: libs[p]), \
                mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2.1"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(torch.cuda, "synchronize", return_value=None), \
                mock.patch.object(l11, "device_arch", return_value="gfx950"), \
                mock.patch.object(do8, "self_check_only", side_effect=lambda: check("only")), _Capture() as cap:
            try:
                out = do8.prebuild(True, True, self_check_fn=lambda: check("plus"))
            except Exception as error:  # noqa: BLE001
                out = error
        return out, cap.text

    def test_healthy_knob_on_prebuild_builds_plus_int8_and_only_nib(self) -> None:
        out, text = self._run(set())
        self.assertIs(out, True)
        self.assertEqual(self.builds, [("plus", False), ("only", True)])
        self.assertEqual((do8.mask_format(), do8.only_mask_format()), ("int8", "nib"))
        self.assertIn("/x/plus/", str(do8.library_path()))
        self.assertIn("/x/only_nib/", str(do8.only_library_path()))
        self.assertRegex(text, re.escape(self.PLUS_ARMED) + r": .*plus detail nib=0\) mask=int8 \[rank \w+\]")
        self.assertRegex(text, re.escape(self.ONLY_ARMED) + r": .*only detail nib=1\) mask=nib\(uint8\[M,256\]\) "
                               r"\[rank \w+\]")
        self.assertNotIn(self.FALLBACK, text)
        self.assertNotIn("DISABLED", text)
        self.assertEqual(do8.counters()["mask_fallback"], 0)

    def test_plus_failures_latch_as_without_nib(self) -> None:
        """NIB v2: the 'plus' build is int8, so its failures are DO8's own: latch (DISABLED line), no NIB fallback, no
        'only' build, exactly as with the knob off."""
        for fail, why in (
            ({("build", "plus", False)}, "DISABLED: prebuild build/load: RuntimeError: resource check: plus VGPR "
                                         "spill"),
            ({("impl", "plus", False)}, "DISABLED: prebuild build/load: RuntimeError: 'plus' .so reads keep mask "
                                        "format nib; the 'plus' build is always int8"),
            ({("check", "plus", False)}, "DISABLED: prebuild self-check: ValueError: plus check blew up"),
            ({("mismatch", "plus", False)}, "DISABLED: prebuild self-check mismatch: plus dO8 mism 3"),
        ):
            for knob in ("1", None):
                out, text = self._run(fail, env={KNOB: knob})
                self.assertIs(out, False, (fail, knob))
                self.assertEqual(text.count(f"{do8.FAMILY} {why}"), 1, (fail, knob, text[-1500:]))
                self.assertNotIn(self.FALLBACK, text)
                self.assertEqual(self.builds, [("plus", False)], (fail, knob))
                self.assertIsNotNone(do8.broken())
                self.assertIsNone(do8._nib_fallback)
                self.assertFalse(do8.nib_ready())

    def _assert_only_fell_back_to_int8(self, out: Any, text: str, why: str) -> None:
        self.assertIs(out, True, text[-1500:])
        self.assertEqual(text.count(self.FALLBACK), 1, text[-1500:])
        self.assertIn(why, text)
        self.assertNotIn(f"{do8.FAMILY} DISABLED:", text)  # DO8 not latched
        self.assertNotIn(f"{do8.FAMILY_ONLY} DISABLED:", text)  # 'only' not latched
        self.assertTrue(do8.prebuilt() and do8.only_prebuilt())
        self.assertIsNone(do8.broken())
        self.assertIsNone(do8.only_broken())
        self.assertEqual((do8.mask_format(), do8.only_mask_format()), ("int8", "int8"))
        self.assertIn("/x/plus/", str(do8.library_path()))
        self.assertIn("/x/only/", str(do8.only_library_path()))  # the int8 'only' .so, not /x/only_nib/
        self.assertFalse(do8.nib_build_requested())
        self.assertFalse(do8.nib_ready())
        self.assertEqual(do8.counters()["mask_fallback"], 1)
        # 'plus' was built, checked and armed ONCE, before the fallback; 'only' armed once, after it, int8
        first, second = text.split(self.FALLBACK)
        self.assertEqual(text.count(self.PLUS_ARMED), 1)
        self.assertRegex(first, re.escape(self.PLUS_ARMED) + r": .*plus detail nib=0\) mask=int8 \[rank \w+\]")
        self.assertNotIn(self.ONLY_ARMED, first)
        self.assertEqual(second.count(self.ONLY_ARMED), 1)
        self.assertRegex(second, re.escape(self.ONLY_ARMED) + r": .*only detail nib=0\) mask=int8 \[rank \w+\]")
        self.assertIn("DO8-ONLY builds and self-checks its int8 keep-mask 'only' .so instead", text)
        for inert in ("has no L11_MKB_NIB reader", "NIB writers cannot arm", "DO8-ONLY is off"):
            self.assertNotIn(inert, text)

    def test_only_failures_fall_back_to_the_int8_only_build(self) -> None:
        for fail, why in (
            ({("build", "only", True)}, "'only' prebuild build/load: RuntimeError: resource check: only VGPR spill"),
            ({("impl", "only", True)}, "'only' .so reads keep mask format int8, the build requested nib"),
            ({("check", "only", True)}, "'only' prebuild self-check: ValueError: only check blew up"),
            ({("mismatch", "only", True)}, "'only' prebuild self-check mismatch: only dO8 mism 3"),
        ):
            out, text = self._run(fail)
            self._assert_only_fell_back_to_int8(out, text, why)
            self.assertEqual(self.builds, [("plus", False), ("only", True), ("only", False)], fail)

    def test_int8_failures_latch_as_before(self) -> None:
        # knob off: an int8 'plus' failure latches DO8 exactly as without NIB (no fallback line)
        out, text = self._run({("mismatch", "plus", False)}, env={KNOB: None})
        self.assertIs(out, False)
        self.assertIn(f"{do8.FAMILY} DISABLED: prebuild self-check mismatch: plus dO8 mism 3", text)
        self.assertNotIn(self.FALLBACK, text)
        self.assertEqual(self.builds, [("plus", False)])
        # knob off: an int8 'only' failure latches 'only' only
        out, text = self._run({("check", "only", False)}, env={KNOB: None})
        self.assertIs(out, True)
        self.assertEqual(text.count(f"{do8.FAMILY_ONLY} DISABLED: prebuild self-check: ValueError: only check"), 1)
        self.assertNotIn(self.FALLBACK, text)
        self.assertEqual(self.builds, [("plus", False), ("only", False)])
        # knob on, writers off: the 'only' build is int8 -> its failure latches 'only' (no NIB fallback)
        out, text = self._run({("mismatch", "only", False)}, env={MASKGEN: "0"})
        self.assertIs(out, True)
        self.assertIn(f"{do8.FAMILY_ONLY} DISABLED: prebuild self-check mismatch: only dO8 mism 3", text)
        self.assertNotIn(self.FALLBACK, text)
        self.assertIsNone(do8._nib_fallback)
        # NIB 'only', then the int8 'only' retry fails too: 'only' latched (one fallback line, one DISABLED line), DO8
        # 'plus' (int8) on
        out, text = self._run({("check", "only", True), ("check", "only", False)})
        self.assertIs(out, True)
        self.assertEqual(text.count(self.FALLBACK), 1)
        self.assertEqual(text.count(f"{do8.FAMILY_ONLY} DISABLED: prebuild self-check: ValueError: only check"), 1)
        self.assertEqual(self.builds, [("plus", False), ("only", True), ("only", False)])
        self.assertTrue(do8.prebuilt())
        self.assertIsNone(do8.broken())
        self.assertIsNotNone(do8.only_broken())
        self.assertEqual((do8.mask_format(), do8.only_mask_format()), ("int8", "int8"))
        self.assertFalse(do8.nib_ready())

    def test_fail_closed_and_device_fault_are_unchanged(self) -> None:
        out, text = self._run({("mismatch", "only", True)}, env={do8.KNOB + "_FAIL_CLOSED": "1"})
        self.assertIsInstance(out, RuntimeError)
        self.assertIn("FAIL_CLOSED", str(out))
        self.assertNotIn(self.FALLBACK, text)
        self.assertIsNone(do8._nib_fallback)
        out, text = self._run(set(), check_exc=RuntimeError("HIP error: an illegal memory access was encountered"))
        self.assertIsInstance(out, RuntimeError)
        self.assertIn(f"{do8.FAMILY_ONLY} prebuild DEVICE FAULT", text)
        self.assertNotIn(self.FALLBACK, text)
        self.assertIsNone(do8._nib_fallback)

    def test_arch_failure_is_not_a_nib_failure(self) -> None:
        do8.reset_for_testing(unload=True)
        with _env({KNOB: "1", VD_KNOB: "1"}), mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.2.1"), \
                mock.patch.object(torch.cuda, "current_device", return_value=0), \
                mock.patch.object(l11, "device_arch", return_value="gfx942"), \
                mock.patch.object(do8, "build", side_effect=AssertionError("built")), _Capture() as cap:
            self.assertFalse(do8.prebuild(True, True))
        self.assertIn(f"{do8.FAMILY} DISABLED: prebuild build/load: RuntimeError: arch gfx942 is not gfx950", cap.text)
        self.assertNotIn(self.FALLBACK, cap.text)
        self.assertIsNone(do8._nib_fallback)


# ---------------------------------------------------------------------------------------------------- CPU: run_e5_plus
class Do8NibRunCpuTest(_EnvReset):
    T = 1000

    def setUp(self) -> None:
        super().setUp()
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        os.environ.update({KNOB: "1", VD_KNOB: "1", ONLY: "1"})  # a NIB process; tests that need less say so
        T = self.T
        self.args = dict(g8=torch.zeros(T, D, dtype=E5), x=torch.zeros(T, D, dtype=BF), u=torch.zeros(T, D, dtype=BF),
                         wn=torch.ones(D, dtype=BF), bn=torch.zeros(D, dtype=BF), mean=torch.zeros(T),
                         rstd=torch.ones(T), inv=torch.ones(1), cast=torch.ones(1))
        self.site = SimpleNamespace(seq_offsets=torch.tensor([0, T]), N=T, only_plan=None)

    def call(self, mask: Any, only_impl: Optional[str] = _IMPL_ONLY_NIB, allow: bool = False,
             plan: Optional[bool] = None) -> Tuple[Any, List[Tuple[Any, bool]], int]:
        """run_e5_plus as the seam calls it with the int8 'plus' .so and `only_impl` as the loaded 'only' .so (None:
        none loaded). allow=False: every step after the routing decline (work_list, _plan_only, launch_e5_plus,
        _disable, _disable_only, torch allocations) raises if reached. allow=True: work_list / launch_e5_plus stubbed
        and _plan_only answers `plan` (None: raises if called). Returns (result, [(mask, only)] of each launch, the
        number of _plan_only calls)."""
        a, seen, plans = self.args, [], []

        def fake_launch(g8, w8, sa, sb, x, u, mask_, mean, rstd, wn, bn, p, inv, dz8, wl, cnt, cap, d8, e8,
                        only=False, dx=None):  # pyre-ignore[2,3]
            seen.append((mask_, only))
            return torch.empty_like(x), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8), 8

        def fake_plan(site: Any, attn: Any) -> bool:
            plans.append(site)
            if plan is None:
                raise AssertionError("_plan_only reached")
            return plan

        with contextlib.ExitStack() as st:
            for target, attr, kw in (
                (l11, "_operands_ok", {"return_value": None}), (l11, "e5_decline_reason", {"return_value": None}),
                (l11, "dwdb_finalize", {"new": lambda *x: None}), (do8, "_lib", {"new": object()}),
                (do8, "_impl", {"new": _IMPL_PLUS}), (do8, "_only_lib", {"new": object() if only_impl else None}),
                (do8, "_impl_only", {"new": only_impl or ""}),
                (do8, "_registry_mod", {"new": SimpleNamespace(poison_requested=lambda: False)}),
                (bwd_fp8, "grad_dtype_and_limit", {"return_value": (E5, 57344.0)}),
                (bwd_fp8, "uvqk_tm_pad", {"return_value": 128}),
            ):
                st.enter_context(mock.patch.object(target, attr, **kw))
            if allow:
                wl = (torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 20)
                st.enter_context(mock.patch.object(do8, "work_list", return_value=wl))
                st.enter_context(mock.patch.object(do8, "launch_e5_plus", side_effect=fake_launch))
                st.enter_context(mock.patch.object(do8, "_plan_only", side_effect=fake_plan))
            else:
                for name in ("work_list", "_plan_only", "launch_e5_plus", "_disable", "_disable_only"):
                    st.enter_context(mock.patch.object(do8, name, side_effect=AssertionError(f"{name} reached")))
                for fn in ("empty", "empty_like", "full", "zeros", "ones", "tensor"):
                    st.enter_context(mock.patch.object(torch, fn, side_effect=AssertionError(f"torch.{fn} called")))
            out = do8.run_e5_plus(a["g8"], "W8", "SA", "SB", a["x"], a["u"], a["wn"], a["bn"], a["mean"], a["rstd"], D,
                                  2, P_DROP, True, mask, DU_ROW, 0, a["inv"], a["cast"], None, self.site)
        return out, seen, len(plans)

    def _assert_no_latch(self, tag: Any) -> None:
        self.assertIsNone(do8.broken(), tag)
        self.assertIsNone(do8.only_broken(), tag)
        self.assertTrue(do8.enabled(), tag)
        self.assertEqual(do8.counters()["failed"], 0, tag)

    def test_nib_mask_declines_before_any_allocation_without_a_latch(self) -> None:
        """A NIB mask that no ready NIB 'only' .so can read: a counted DECLINE before work_list, _plan_only, the launch
        and every torch allocation; no latch; one NOT armed line per process."""
        T = self.T
        nib = nibm.pack_ref(_rand_mask(T, 1))
        buf = torch.zeros(T * 256 + 4, dtype=U8)
        cases = (
            ("knob off", nib, _IMPL_ONLY_NIB, {KNOB: None}, "is not 1"),
            ("no 'only' .so loaded", nib, None, {}, "reads the int8 keep mask"),
            ("int8 'only' .so", nib, _IMPL_ONLY, {}, "reads the int8 keep mask"),
            ("DO8-ONLY knob off", nib, _IMPL_ONLY_NIB, {ONLY: None}, "DO8-ONLY off"),
            ("DO8-ONLY knob 0", nib, _IMPL_ONLY_NIB, {ONLY: "0"}, "DO8-ONLY off"),
            ("'only' not prebuilt", nib, _IMPL_ONLY_NIB, {"_pre": False}, "not prebuilt or latched"),
            ("'only' latched", nib, _IMPL_ONLY_NIB, {"_broken": "x"}, "not prebuilt or latched"),
            ("other M", nibm.pack_ref(_rand_mask(T + 1, 3)), _IMPL_ONLY_NIB, {}, "the loaded nib .so reads uint8"),
            ("strided", nibm.pack_ref(_rand_mask(2 * T, 4))[::2], _IMPL_ONLY_NIB, {}, "the loaded nib .so reads"),
            ("misaligned", buf[4:].view(T, 256), _IMPL_ONLY_NIB, {}, "a16=0; the loaded nib .so reads"),
        )
        real_note = do8.note_decline
        for i, (name, mask, only_impl, over, why) in enumerate(cases, 1):
            env = {k: v for k, v in over.items() if not k.startswith("_")}
            do8.set_only_prebuilt_for_testing(over.get("_pre", True))
            spy = mock.MagicMock(side_effect=real_note)
            with _env(env), mock.patch.object(do8, "_only_broken", over.get("_broken")), \
                    mock.patch.object(do8, "note_decline", spy), _Capture() as cap:
                out, seen, plans = self.call(mask, only_impl)
            self.assertIsNone(out, name)
            self.assertEqual((seen, plans), ([], 0), name)
            self._assert_no_latch(name)
            self.assertEqual(spy.call_count, 1, name)
            reason = spy.call_args[0][0]
            self.assertTrue(reason.startswith("E5 mask format: keep mask"), (name, reason))
            self.assertIn(why, reason, name)
            c = do8.counters()
            self.assertEqual(c["declined"], {"E5 mask": i}, name)
            self.assertEqual((c["mask_declined"], c["mask_nib_plus"], c["mask_int8_plus"]), (i, 0, 0), name)
            # one NOT armed line per process (the first case printed it)
            self.assertEqual(cap.text.count(f"{do8.FAMILY} NOT armed: E5 mask format: keep mask"), int(i == 1), name)
        self.assertEqual(nibm.counters()["unpack"], 0)

    def test_int8_mask_with_a_nib_only_so_runs_plus_without_a_plan(self) -> None:
        """An int8 mask while the loaded 'only' .so reads NIB: DO8 'plus' with the mask itself, _plan_only never
        called (the registry could answer "only"); counted, one line per process; not a decline, no latch."""
        T = self.T
        int8 = _rand_mask(T, 5)
        l11._active_e5_logged = True  # the L11 ACTIVE line is not under test here
        for i, over in enumerate(({}, {ONLY: None}, {"_broken": "x"}), 1):
            env = {k: v for k, v in over.items() if not k.startswith("_")}
            with _env(env), mock.patch.object(do8, "_only_broken", over.get("_broken")), _Capture() as cap:
                out, seen, plans = self.call(int8, _IMPL_ONLY_NIB, allow=True, plan=None)
            self.assertIsNotNone(out, over)
            self.assertEqual(len(seen), 1, over)
            self.assertIs(seen[0][0], int8, over)  # no unpack, no copy
            self.assertIs(seen[0][1], False, over)  # the int8 'plus' launch
            self.assertEqual(plans, 0, over)
            self.assertEqual(do8.counters()["mask_int8_plus"], i, over)
            self.assertEqual(cap.text.count(_INT8_PLUS_LINE), int(i == 1), over)
            self.assertNotIn("NOT armed", cap.text)
        self.assertEqual(do8.counters()["declined"], {})
        self.assertEqual(do8.counters()["emitted"], 3)
        self._assert_no_latch("int8 -> plus")
        self.assertEqual(nibm.counters()["unpack"], 0)

    def test_int8_route_drops_a_stale_only_call_entry_as_plan_only_does(self) -> None:
        """Review N1: _plan_only clears _ONLY_CALL (the entry of an 'only' call that raised between its launch and its
        publish) at every DO8-ONLY call; the int8 -> 'plus' route skips _plan_only, so it clears the entry itself when
        DO8-ONLY is on, and keeps it with DO8-ONLY off, as _plan_only does."""
        int8 = _rand_mask(self.T, 9)
        l11._active_e5_logged = True  # the L11 ACTIVE line is not under test here
        for over, kept in (({}, False), ({ONLY: None}, True), ({ONLY: "0"}, True)):
            stale = SimpleNamespace(tag="a site whose 'only' call raised before its publish")
            do8._ONLY_CALL.update(site=stale, dx=lambda: None)
            with _env(over), _Capture():
                out, seen, plans = self.call(int8, _IMPL_ONLY_NIB, allow=True, plan=None)
            self.assertIsNotNone(out, over)
            self.assertEqual([(m is int8, o) for m, o in seen], [(True, False)], over)  # the int8 'plus' launch
            self.assertEqual(plans, 0, over)
            self.assertEqual(do8._ONLY_CALL.get("site") is stale, kept, over)
            do8._ONLY_CALL.clear()
        self._assert_no_latch("stale entry")

    def test_nib_mask_runs_only_when_the_plan_answers_only(self) -> None:
        T = self.T
        nib = nibm.pack_ref(_rand_mask(T, 6))
        with _Capture() as cap:
            out, seen, plans = self.call(nib, _IMPL_ONLY_NIB, allow=True, plan=True)
        self.assertIsNotNone(out)
        self.assertEqual(plans, 1)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], nib)  # the NIB mask itself: no unpack, no copy
        self.assertIs(seen[0][1], True)  # the NIB 'only' launch
        self.assertIn(f"impl={_IMPL_ONLY_NIB}", cap.text)  # L11's ACTIVE line names the 'only' build that ran
        c = do8.counters()
        self.assertEqual((c["declined"], c["mask_int8_plus"], c["mask_nib_plus"]), ({}, 0, 0))
        self.assertEqual(do8.only_counters()["only_launched"], 1)
        self._assert_no_latch("nib only")
        self.assertEqual(nibm.counters()["unpack"], 0)
        self.assertIsNone(getattr(nib, nibm._INT8_ATTR, None))

    def test_nib_mask_on_a_plus_planned_call_declines_without_latch_or_launch(self) -> None:
        """The plan answers "plus" (or the 'only' fallback): no 'only' plan is open, and the 'plus' .so reads int8 ->
        the same counted DECLINE (nothing to cancel, nothing launched); L11's own run_e5 then serves the call."""
        T = self.T
        nib = nibm.pack_ref(_rand_mask(T, 7))
        with mock.patch.object(do8, "_cancel_only", side_effect=AssertionError("cancel")), _Capture() as cap:
            out, seen, plans = self.call(nib, _IMPL_ONLY_NIB, allow=True, plan=False)
            out2, seen2, plans2 = self.call(nib, _IMPL_ONLY_NIB, allow=True, plan=False)
        self.assertIsNone(out)
        self.assertIsNone(out2)
        self.assertEqual((seen, plans, seen2, plans2), ([], 1, [], 1))
        c = do8.counters()
        self.assertEqual(c["declined"], {"E5 mask": 2})
        self.assertEqual((c["mask_declined"], c["mask_nib_plus"], c["mask_int8_plus"], c["emitted"]), (2, 2, 0, 0))
        self.assertEqual(cap.text.count(f"{do8.FAMILY} NOT armed: E5 mask format: keep mask NIB on a call DO8-ONLY "
                                        "runs as 'plus'"), 1)
        self._assert_no_latch("nib plus")
        self.assertEqual(nibm.counters()["unpack"], 0)

    def test_int8_only_so_and_knob_off_are_todays_path(self) -> None:
        """An int8 mask with an int8 (or no) 'only' .so, the knob on or off: _plan_only decides as today (its answer
        goes to the launch), no NIB counter, no NIB line."""
        T = self.T
        int8 = _rand_mask(T, 8)
        l11._active_e5_logged = True
        for env, only_impl, plan in (({}, _IMPL_ONLY, True), ({}, _IMPL_ONLY, False), ({KNOB: None}, _IMPL_ONLY, True),
                                     ({KNOB: None, ONLY: None}, None, False), ({}, None, False)):
            with _env(env), _Capture() as cap:
                out, seen, plans = self.call(int8, only_impl, allow=True, plan=plan)
            self.assertIsNotNone(out, env)
            self.assertEqual(plans, 1, env)
            self.assertEqual(len(seen), 1, env)
            self.assertIs(seen[0][0], int8, env)
            self.assertIs(seen[0][1], plan, env)
            self.assertNotIn("NIB", cap.text, env)
            self.assertNotIn("NOT armed", cap.text, env)
        c = do8.counters()
        self.assertEqual((c["declined"], c["mask_int8_plus"], c["mask_nib_plus"]), ({}, 0, 0))
        self._assert_no_latch("today")

    def test_knob_off_run_calls_the_real_plan_only_as_today(self) -> None:
        """Knob off, DO8-ONLY off: the real _plan_only returns False without touching the registry or any state (the
        ship15 B3 + v2.1 behaviour), then the 'plus' launch; nothing NIB is read or printed."""
        T = self.T
        int8 = _rand_mask(T, 9)
        l11._active_e5_logged = True
        spy = mock.MagicMock(wraps=do8._plan_only)
        wl = (torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 20)
        seen: List[Any] = []

        def fake_launch(*a: Any, only: bool = False, dx: Any = None) -> Any:
            seen.append((a[6], only))
            return torch.empty_like(a[4]), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8), 8

        a = self.args
        with _env({KNOB: None, ONLY: None}), mock.patch.object(do8, "_plan_only", spy), \
                mock.patch.object(do8, "registry", side_effect=AssertionError("registry asked")), \
                mock.patch.object(do8, "work_list", return_value=wl), \
                mock.patch.object(do8, "launch_e5_plus", side_effect=fake_launch), \
                mock.patch.object(l11, "_operands_ok", return_value=None), \
                mock.patch.object(l11, "e5_decline_reason", return_value=None), \
                mock.patch.object(l11, "dwdb_finalize", new=lambda *x: None), \
                mock.patch.object(do8, "_lib", object()), \
                mock.patch.object(bwd_fp8, "grad_dtype_and_limit", return_value=(E5, 57344.0)), \
                mock.patch.object(bwd_fp8, "uvqk_tm_pad", return_value=128), _Capture() as cap:
            out = do8.run_e5_plus(a["g8"], "W8", "SA", "SB", a["x"], a["u"], a["wn"], a["bn"], a["mean"], a["rstd"], D,
                                  2, P_DROP, True, int8, DU_ROW, 0, a["inv"], a["cast"], None, self.site)
        self.assertIsNotNone(out)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(seen, [(int8, False)])
        self.assertNotIn("NIB", cap.text)
        self.assertEqual(do8._ONLY_CALL, {})
        self.assertEqual(do8.only_counters(), dict.fromkeys(do8._ONLY_COUNTS, 0) | {"latched": 0})

    def test_non_tensor_stand_in_keeps_todays_path_with_int8_so_only(self) -> None:
        """Stubbed callers pass placeholders for operands they do not use (l11_do8_test.py): with int8 .so only a
        non-tensor mask is not format-checked, so today's control flow (work_list -> its failure latches) is kept;
        while the NIB 'only' .so is loaded it is a counted decline (never a latch on a format question)."""
        a = self.args
        with mock.patch.object(l11, "_operands_ok", return_value=None), \
                mock.patch.object(l11, "e5_decline_reason", return_value=None), \
                mock.patch.object(do8, "_lib", object()), mock.patch.object(do8, "_impl", _IMPL_PLUS), \
                mock.patch.object(do8, "work_list", side_effect=RuntimeError("wl rc=1")), \
                mock.patch.object(bwd_fp8, "grad_dtype_and_limit", return_value=(E5, 57344.0)), \
                mock.patch.object(bwd_fp8, "uvqk_tm_pad", return_value=128), _Capture() as cap:
            out = do8.run_e5_plus(a["g8"], "W8", "SA", "SB", a["x"], a["u"], a["wn"], a["bn"], a["mean"], a["rstd"],
                                  D, 2, P_DROP, True, "MASK", DU_ROW, 0, a["inv"], a["cast"], None, self.site)
        self.assertIsNone(out)
        self.assertEqual(do8.broken(), "e5 plus: RuntimeError: wl rc=1")
        self.assertIn("DISABLED: e5 plus: RuntimeError", cap.text)
        self.assertNotIn("E5 mask", do8.counters()["declined"])
        do8.reset_for_testing()
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        with _Capture() as cap:
            out, seen, plans = self.call("MASK", _IMPL_ONLY_NIB)
        self.assertIsNone(out)
        self.assertEqual((seen, plans), ([], 0))
        self._assert_no_latch("stand-in")
        self.assertEqual(do8.counters()["declined"], {"E5 mask": 1})
        self.assertIn("NOT armed: E5 mask format: keep mask str; the loaded int8 .so reads int8 [1000, 512]", cap.text)

    # ------------------------------------------------------------ the raw entry: launch_e5_plus (ctypes stubbed)
    def raw(self, mask: Any, impl: str, impl_only: str, only: bool) -> List[Tuple[str, int]]:
        T, a, calls = self.T, self.args, []

        def entry(tag: str) -> Any:
            def fn(*args: Any) -> int:
                calls.append((tag, int(args[7])))  # MK is the 8th argument
                return 0
            return fn

        def fn_plan(cap: int, addr: int) -> int:
            out = ctypes.cast(addr, ctypes.POINTER(ctypes.c_longlong))
            for i, v in enumerate((1, 0, 1, 8)):  # G, fl, nf, P
                out[i] = v
            return 0

        rows_out = (T + 127) // 128 * 128
        dz8, d8, e8 = torch.zeros(rows_out, DU_ROW, dtype=U8).view(E5), torch.zeros(T, D, dtype=U8).view(E4), \
            torch.zeros(T, H, dtype=U8)
        wl, cnt = torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)
        with contextlib.ExitStack() as st:
            for target, attr, kw in (
                (do8, "_lib", {"new": object()}), (do8, "_only_lib", {"new": object()}), (do8, "_impl", {"new": impl}),
                (do8, "_impl_only", {"new": impl_only}), (do8, "_fn_launch", {"new": entry("plus")}),
                (do8, "_fn_launch_only", {"new": entry("only")}), (do8, "_fn_drain", {"new": lambda: 0}),
                (do8, "_fn_plan", {"new": fn_plan}), (torch.cuda, "current_device", {"return_value": 0}),
                (torch.cuda, "device", {"new": lambda i: contextlib.nullcontext()}),
                (torch.cuda, "current_stream", {"new": lambda d=None: SimpleNamespace(cuda_stream=0)}),
            ):
                st.enter_context(mock.patch.object(target, attr, **kw))
            st.enter_context(mock.patch.dict(do8._plan_cache, clear=True))
            do8.launch_e5_plus(a["g8"], torch.zeros(1536, D, dtype=E4), torch.ones(1), torch.ones(1), a["x"], a["u"],
                               mask, a["mean"], a["rstd"], a["wn"], a["bn"], P_DROP, a["inv"], dz8, wl, cnt, 20, d8, e8,
                               only=only)
        return calls

    def test_raw_launch_raises_on_a_mismatch_before_the_kernel_entry(self) -> None:
        T = self.T
        nib, int8 = nibm.pack_ref(_rand_mask(T, 7)), _rand_mask(T, 8)
        self.assertEqual(self.raw(nib, _IMPL_PLUS, _IMPL_ONLY_NIB, True), [("only", nib.data_ptr())])
        self.assertEqual(self.raw(int8, _IMPL_PLUS, _IMPL_ONLY_NIB, False), [("plus", int8.data_ptr())])
        self.assertEqual(self.raw(int8, _IMPL_PLUS, _IMPL_ONLY, False), [("plus", int8.data_ptr())])
        self.assertEqual(self.raw(int8, _IMPL_PLUS, _IMPL_ONLY, True), [("only", int8.data_ptr())])
        for mask, impl, impl_only, only in (
            (nib, _IMPL_PLUS, _IMPL_ONLY_NIB, False),  # NIB v2: the 'plus' .so reads int8
            (int8, _IMPL_PLUS, _IMPL_ONLY_NIB, True),
            (nib, _IMPL_PLUS, _IMPL_ONLY, True),  # the 'only' launch checks the 'only' .so's own format
            (nib, _IMPL_PLUS, _IMPL_ONLY, False),
            (int8[:, :256].contiguous(), _IMPL_PLUS, _IMPL_ONLY, False),
            (nibm.pack_ref(_rand_mask(T - 1, 9)), _IMPL_PLUS, _IMPL_ONLY_NIB, True),
        ):
            with self.assertRaisesRegex(RuntimeError, r"^mask format: keep mask .* \(M=1000\)$"):
                self.raw(mask, impl, impl_only, only)
        self.assertEqual(nibm.counters()["unpack"], 0)


# ---------------------------------------------------------------------------------------------------- CPU: registry
class _OnlyFacts:
    """Stand-in of hstu_flydsl_mha._Do8PlanFacts (as in R15's v2 tests): reason() None = the C8 do_blk branch."""

    def __init__(self, why: Optional[str] = None) -> None:
        self.why = why

    def reason(self) -> Optional[str]:
        return self.why


class _OnlyLeaf(torch.autograd.Function):
    """The attention output stand-in: [T, H, HD] from a [T, 512] leaf, so plan()'s autograd check sees a real graph."""

    @staticmethod
    def forward(ctx, x):  # pyre-ignore[2,3]
        return x.detach().view(-1, H, HD).clone()

    @staticmethod
    def backward(ctx, g):  # pyre-ignore[2,3]
        return g.reshape(-1, D)


class Do8NibRegistryCpuTest(_EnvReset):
    """NIB v2 routing through the REAL registry (plan / publish / bind; contract v2), run_e5_plus with the kernel
    launch and the dW/dB finalize stubbed, the NIB 'only' .so faked as loaded and prebuilt."""

    T = 1000

    def setUp(self) -> None:
        super().setUp()
        try:
            from generative_recommenders.ops.flydsl import hstu_do8_registry as reg
        except ImportError:
            self.skipTest("R15's hstu_do8_registry not in this tree")
        if not hasattr(reg, "plan"):
            self.skipTest("R15's hook has no contract v2 (plan / cancel)")
        self.reg = reg
        keys = (reg.KNOB_ENV, reg.TAKE_ENV, reg.ONLY_ENV, reg.POISON_ENV)
        self._reg_env = {k: os.environ.pop(k, None) for k in keys}
        os.environ.update({reg.KNOB_ENV: "1", reg.ONLY_ENV: "1", KNOB: "1", VD_KNOB: "1"})
        reg.reset_for_testing()
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)

    def tearDown(self) -> None:
        for k, v in self._reg_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        self.reg.reset_for_testing()
        super().tearDown()

    def graph(self, facts: Any = None) -> Tuple[Any, torch.Tensor]:
        x = torch.randn(self.T, D).to(BF).requires_grad_()
        out = _OnlyLeaf.apply(x)
        offs = torch.tensor([0, 400, self.T], dtype=torch.int64)
        self.reg.stash_plan(out, _OnlyFacts() if facts is None else facts)
        site = self.reg.new_site(out, offs, 4096)
        attn = self.reg.carry_site(out, out.view(-1, D))
        self._keep = (x, out, attn)  # the forward graph stays alive
        return site, attn

    def run_call(self, site: Any, attn: torch.Tensor, mask: torch.Tensor,
                 only_impl: str = _IMPL_ONLY_NIB) -> Tuple[Any, List[Tuple[Any, bool]]]:
        """run_e5_plus as the seam calls it (grad mode off), the launch stubbed: 'plus' writes ones into d(attn), both
        modes write dO8 / E8. Returns (result, [(mask, only)] of every launch)."""
        launches: List[Tuple[Any, bool]] = []
        T = int(attn.shape[0])

        def fake_launch(g8, w8, sa, sb, x, u, mask_, mean, rstd, wn, bn, p, inv, dz8, wl, cnt, cap, d8, e8,
                        only=False, dx=None):  # pyre-ignore[2,3]
            launches.append((mask_, only))
            dx_ = torch.empty_like(x) if dx is None else dx
            if not only:
                dx_.fill_(1.0)
            d8.view(U8).fill_(0x11)
            e8.fill_(0x7F)
            return dx_, torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8), 8

        wl = (torch.zeros(20, 2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32), 20)
        with mock.patch.object(l11, "_operands_ok", return_value=None), \
                mock.patch.object(l11, "e5_decline_reason", return_value=None), \
                mock.patch.object(l11, "dwdb_finalize", side_effect=lambda *a: None), \
                mock.patch.object(do8, "_lib", object()), mock.patch.object(do8, "_impl", _IMPL_PLUS), \
                mock.patch.object(do8, "_only_lib", object()), mock.patch.object(do8, "_impl_only", only_impl), \
                mock.patch.object(do8, "work_list", return_value=wl), \
                mock.patch.object(do8, "launch_e5_plus", side_effect=fake_launch), \
                mock.patch.object(bwd_fp8, "grad_dtype_and_limit", return_value=(E5, 57344.0)), \
                mock.patch.object(bwd_fp8, "uvqk_tm_pad", return_value=128), torch.no_grad():
            out = do8.run_e5_plus(torch.empty(T, D, dtype=E5), "W8", "SA", "SB", attn, torch.zeros(T, D, dtype=BF),
                                  torch.ones(D, dtype=BF), torch.zeros(D, dtype=BF), torch.zeros(T), torch.ones(T), D,
                                  2, P_DROP, True, mask, DU_ROW, 0, torch.ones(1), torch.ones(1), None, site)
        return out, launches

    def test_nib_mask_planned_only_runs_the_nib_only_so_and_publishes_only(self) -> None:
        site, attn = self.graph()
        nib = nibm.pack_ref(_rand_mask(self.T, 21))
        with _Capture():
            out, launches = self.run_call(site, attn, nib)
            self.assertEqual(len(launches), 1)
            self.assertIs(launches[0][0], nib)
            self.assertIs(launches[0][1], True)
            self.assertIsNotNone(site.only_plan)  # open from the plan to the publish
            dx, _du, _dw, _db, _emit, d8, e8 = out
            self.assertTrue(do8.publish(site, dx, d8, e8))
        self.assertIsNone(site.only_plan)
        self.assertFalse(getattr(dx, self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)
        self.assertTrue(self.reg.bind(site, dx.view(-1, H, HD)))
        c = do8.only_counters()
        self.assertEqual((c["planned"], c["only_launched"], c["published"], c["plan_plus"], c["latched"]),
                         (1, 1, 1, 0, 0))
        self.assertEqual(do8.counters()["declined"], {})
        self.assertEqual(nibm.counters()["unpack"], 0)

    def test_nib_mask_planned_plus_declines_and_leaves_no_open_plan(self) -> None:
        site, attn = self.graph(_OnlyFacts("not the C8 do_blk branch (test)"))  # plan() answers "plus"
        nib = nibm.pack_ref(_rand_mask(self.T, 22))
        with mock.patch.object(self.reg, "cancel", side_effect=AssertionError("cancel() with no plan open")), \
                _Capture() as cap:
            out, launches = self.run_call(site, attn, nib)
        self.assertIsNone(out)
        self.assertEqual(launches, [])
        self.assertIsNone(site.only_plan)  # nothing open: the attention backward binds as for any unpublished call
        self.assertIsNone(site.record)
        self.assertFalse(self.reg.bind(site, attn.view(-1, H, HD)))  # no record, no raise
        s = self.reg.stats()["only"]
        self.assertEqual((s["planned"], s["cancelled"], s["published"]), (0, 0, 0))
        c = do8.counters()
        self.assertEqual((c["declined"], c["mask_nib_plus"], c["emitted"], c["failed"]), ({"E5 mask": 1}, 1, 0, 0))
        self.assertEqual(do8.only_counters()["plan_plus"], 1)
        self.assertIsNone(do8.broken())
        self.assertIsNone(do8.only_broken())
        self.assertIn("NOT armed: E5 mask format: keep mask NIB on a call DO8-ONLY runs as 'plus'", cap.text)

    def test_int8_mask_with_the_nib_only_so_publishes_plus_without_asking_the_registry(self) -> None:
        site, attn = self.graph()  # facts present: plan() WOULD answer "only"
        int8 = _rand_mask(self.T, 23)
        with mock.patch.object(self.reg, "plan", side_effect=AssertionError("plan() asked for an int8 mask")), \
                mock.patch.object(self.reg, "cancel", side_effect=AssertionError("cancel()")), _Capture() as cap:
            out, launches = self.run_call(site, attn, int8)
            self.assertEqual(len(launches), 1)
            self.assertIs(launches[0][0], int8)
            self.assertIs(launches[0][1], False)
            dx, _du, _dw, _db, _emit, d8, e8 = out
            self.assertTrue(do8.publish(site, dx, d8, e8))
        rec = getattr(dx, self.reg.HSTU_DO8_RECORD_ATTR)
        self.assertTrue(rec.bf16_valid)  # a 'plus' record: the bf16 d(attn) was written
        self.assertTrue(bool((dx == 1.0).all()))
        self.assertTrue(self.reg.bind(site, dx.view(-1, H, HD)))
        self.assertIsNone(site.only_plan)
        self.assertEqual(do8.counters()["mask_int8_plus"], 1)
        self.assertEqual(do8.only_counters()["planned"], 0)
        self.assertEqual(cap.text.count(_INT8_PLUS_LINE), 1)
        self.assertEqual(do8._ONLY_CALL, {})

    def test_int8_mask_with_an_int8_only_so_is_todays_only_path(self) -> None:
        site, attn = self.graph()
        int8 = _rand_mask(self.T, 24)
        with _Capture():
            out, launches = self.run_call(site, attn, int8, only_impl=_IMPL_ONLY)
            self.assertEqual([(m is int8, o) for m, o in launches], [(True, True)])
            dx, _du, _dw, _db, _emit, d8, e8 = out
            self.assertTrue(do8.publish(site, dx, d8, e8))
        self.assertFalse(getattr(dx, self.reg.HSTU_DO8_RECORD_ATTR).bf16_valid)
        self.assertEqual(do8.counters()["mask_int8_plus"], 0)


# ---------------------------------------------------------------------------------------------------- CPU: self-checks
class _CpuTorch:
    """l11_do8's `torch` for a CPU run of its self-checks: torch itself, except torch.device (-> cpu) and torch.cuda
    (device 0, a stream stand-in, synchronize no-op). Used only inside a mock.patch of l11_do8.torch."""

    def __init__(self) -> None:
        self.cuda = SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None,
                                    current_stream=lambda d=None: SimpleNamespace(cuda_stream=0))

    def device(self, *a: Any, **k: Any) -> torch.device:
        return torch.device("cpu")

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)


class Do8NibSelfCheckCpuTest(_EnvReset):
    """The self-checks executed on CPU (synthetic inputs as on the GPU; the kernels, L11's E5 launch, the quantizer
    and the mask generator faked): which keep mask each launch receives, the NIB gate and the detail segment."""

    def _fakes(self, mask: torch.Tensor) -> Tuple[contextlib.ExitStack, List[Tuple[Any, bool]]]:
        seen: List[Tuple[Any, bool]] = []
        M = int(mask.shape[0])

        def bits_of(mk: torch.Tensor) -> torch.Tensor:
            return (nibm.unpack(mk) if nibm.is_nib(mk) else mk).view(U8) & 7

        def fake_launch(g8, w8, sa, sb, x, u, mk, mean, rstd, wn, bn, p, inv, dz8, wl, cnt, cap, d8, e8,
                        only=False, dx=None):  # pyre-ignore[2,3]
            seen.append((mk, only))
            b = bits_of(mk)
            d8.view(U8).copy_(b)
            dz8.view(U8)[:M, :D].copy_(b)
            e8.fill_(127)
            dx_ = torch.zeros_like(x) if dx is None else dx
            return dx_, torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8), 8

        def fake_l11_e5(g8, w8, sa, sb, x, u, mk, mean, rstd, wn, bn, p, inv, dz8):  # pyre-ignore[2,3]
            seen.append((mk, "L11"))
            dz8.view(U8)[:M, :D].copy_(bits_of(mk))
            return torch.zeros_like(x), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8), 8

        def fake_quant(dx3: torch.Tensor, offs: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return bits_of(mask).clone(), torch.full((M, H), 127, dtype=U8)

        import generative_recommenders.ops.flydsl as flydsl_pkg
        import generative_recommenders.ops.triton as triton_pkg

        fake_thl = SimpleNamespace(_create_dropout_mask=lambda **kw: mask)
        fake_q = SimpleNamespace(quantize_do_e4m3_blk=fake_quant)
        st = contextlib.ExitStack()
        st.enter_context(mock.patch.dict(sys.modules, {
            "generative_recommenders.ops.triton.triton_hstu_linear": fake_thl,
            "generative_recommenders.ops.flydsl.hstu_do_fp8": fake_q,
        }))
        for target, attr, kw in (
            (triton_pkg, "triton_hstu_linear", {"new": fake_thl, "create": True}),
            (flydsl_pkg, "hstu_do_fp8", {"new": fake_q, "create": True}),
            (do8, "torch", {"new": _CpuTorch()}), (do8, "launch_e5_plus", {"side_effect": fake_launch}),
            (do8, "wl_cap", {"return_value": 2000}), (do8, "_fn_wl_build", {"new": lambda *a: 0}),
            (l11, "launch_e5", {"side_effect": fake_l11_e5}),
            (l11, "dwdb_finalize", {"new": lambda dwp, dbp, dw, db, P: (dw.zero_(), db.zero_())}),
            (bwd_fp8, "grad_dtype_and_limit", {"return_value": (E5, 57344.0)}),
            (bwd_fp8, "uvqk_tm_pad", {"return_value": 128}),
        ):
            st.enter_context(mock.patch.object(target, attr, **kw))
        return st, seen

    def _mask(self, high_bit: bool = False) -> torch.Tensor:
        lens: List[int] = []
        while sum(lens) < 32768:
            lens.extend(do8._SELF_CHECK_LENS)
        m = _keep_mask(sum(lens), P_DROP, seed=do8._SELF_CHECK_SEED)
        if high_bit:
            m[5, 17] |= 8  # not representable in NIB: the round-trip gate must fail
        return m

    def _only(self, only_impl: str, mask: torch.Tensor) -> Tuple[bool, str, List[Tuple[Any, bool]]]:
        st, seen = self._fakes(mask)
        with st, mock.patch.object(do8, "_only_lib", object()), mock.patch.object(do8, "_impl_only", only_impl), \
                _env({KNOB: "1", VD_KNOB: "1", ONLY: "1"}):
            ok, detail = do8.self_check_only()
        return ok, detail, seen

    def test_self_check_only_feeds_the_nib_only_launch_the_packed_mask(self) -> None:
        mask = self._mask()
        ok, detail, seen = self._only(_IMPL_ONLY_NIB, mask)
        self.assertTrue(ok, detail)
        self.assertEqual([o for _, o in seen], [False, True])  # EM = 1 reference first, then EM = 2
        self.assertIs(seen[0][0], mask)  # the 'plus' (int8) reference reads the int8 mask itself
        self.assertTrue(nibm.is_nib(seen[1][0]))
        self.assertTrue(torch.equal(seen[1][0], nibm.pack_ref(mask)))  # the NIB 'only' launch: pack_ref(mask)
        self.assertTrue(detail.endswith("; mask nib(uint8[M,256]) unpack(pack_ref) exact 1"), detail)
        for gate in ("dO8 mism 0", "E8 mism 0", "dz8 mism 0", "DWP mism 0", "DBP mism 0", "DUCS mism 0", "AMAXP mism 0",
                     "sentinel bytes changed 0"):
            self.assertIn(gate, detail)

    def test_self_check_only_gates_the_round_trip(self) -> None:
        ok, detail, seen = self._only(_IMPL_ONLY_NIB, self._mask(high_bit=True))
        self.assertIn("dO8 mism 0", detail)  # every comparison passes (the fake kernels read bits 0-2) ...
        self.assertTrue(detail.endswith("; mask nib(uint8[M,256]) unpack(pack_ref) exact 0"), detail)
        self.assertFalse(ok)  # ... the round-trip gate alone fails it

    def test_self_check_only_with_an_int8_only_so_is_unchanged(self) -> None:
        mask = self._mask()
        ok, detail, seen = self._only(_IMPL_ONLY, mask)
        self.assertTrue(ok, detail)
        self.assertEqual([(m is mask, o) for m, o in seen], [(True, False), (True, True)])
        self.assertNotIn("mask nib", detail)
        self.assertTrue(detail.endswith("E8=127 bytes " + str(int(mask.shape[0]) * H)), detail)

    def test_self_check_plus_reads_the_int8_mask_even_with_a_nib_only_so(self) -> None:
        """self_check ('plus') is ship15's: DO8 'plus' and L11's reference both read the int8 mask itself, no NIB
        detail segment, whatever the knob and the 'only' .so say."""
        mask = self._mask()
        for only_impl in (_IMPL_ONLY_NIB, _IMPL_ONLY):
            st, seen = self._fakes(mask)
            with st, mock.patch.object(do8, "_only_lib", object()), mock.patch.object(do8, "_impl_only", only_impl), \
                    _env({KNOB: "1", VD_KNOB: "1", ONLY: "1"}):
                _ok, detail = do8.self_check()
            self.assertEqual([(m is mask, o) for m, o in seen], [(True, "L11"), (True, False)], only_impl[-30:])
            self.assertNotIn("mask nib", detail)
            self.assertIn("dO8 mism 0, E8 mism 0", detail)


# ---------------------------------------------------------------------------------------------------- CPU: L11
class _FakeCudaDevice:
    """dout's device for decline_reason on a CPU-only host: type cuda, index 0, equal to every tensor's device."""

    type = "cuda"
    index = 0

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 0


class L11NibCpuTest(_EnvReset):
    M = 64

    def setUp(self) -> None:
        super().setUp()
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        m = self.M
        self.t = dict(g8=torch.zeros(m, D, dtype=E5), w8=torch.zeros(1536, D, dtype=E4), sa=torch.ones(1),
                      sb=torch.ones(1), x=torch.zeros(m, D, dtype=BF), u=torch.zeros(m, D, dtype=BF),
                      mean=torch.zeros(m), rstd=torch.ones(m), wn=torch.ones(D, dtype=BF), bn=torch.zeros(D, dtype=BF),
                      inv=torch.ones(1), cast=torch.ones(1))

    def d8(self, mask: Any) -> Optional[str]:
        m, t = self.M, self.t
        ctx = SimpleNamespace(fp8_y_saved=True, group_norm=False, training=True, dropout_ratio=P_DROP,
                              has_random_mask=True, silu_u=True, concat_u=True, concat_x=True,
                              mul_u_activation_type="silu", num_warps=2, BLOCK_D=D, du_row_width=DU_ROW)
        dout = SimpleNamespace(device=_FakeCudaDevice(), dtype=BF, shape=(m, D), dim=lambda: 2)
        with mock.patch.object(torch.version, "hip", "7.2.1"), mock.patch.object(l11, "device_arch",
                                                                                  return_value="gfx950"):
            return l11.decline_reason(ctx, dout, torch.zeros(1536, D, dtype=BF), t["x"], t["u"], t["mean"], t["rstd"],
                                      t["wn"], t["bn"], mask, snr_enabled=False, lnmd_bwd_v2=True)

    def test_d8_gate_accepts_a_nib_mask_without_unpacking(self) -> None:
        m = self.M
        int8 = _rand_mask(m, 1)
        nib = nibm.pack_ref(int8)
        self.assertIsNone(self.d8(int8))
        self.assertIsNone(self.d8(nib))
        self.assertEqual(nibm.counters()["unpack"], 0)
        self.assertIsNone(getattr(nib, nibm._INT8_ATTR, None))
        buf = torch.zeros(m * 256 + 4, dtype=U8)
        for bad in (None, torch.zeros(m, D, dtype=U8), torch.zeros(m, 256, dtype=I8),
                    nibm.pack_ref(_rand_mask(m + 1, 2)), nibm.pack_ref(_rand_mask(2 * m, 3))[::2],
                    buf[4:].view(m, 256), int8[:, :256]):
            self.assertEqual(self.d8(bad), "D8 random_mask layout")

    def _l11_stubs(self, st: contextlib.ExitStack, seen: List[int]) -> None:
        def entry(*args: Any) -> int:
            seen.append(int(args[7]))  # MK is the 8th argument of both L11 launches
            return 0

        for target, attr, kw in (
            (l11, "_lib", {"new": object()}), (l11, "_fn_launch", {"new": entry}),
            (l11, "_fn_launch_e5", {"new": entry}), (l11, "_fn_drain", {"new": lambda: 0}),
            (l11, "plan", {"return_value": (1, 1, 1, 8)}),
            (l11, "_operands_ok", {"return_value": None}), (torch.cuda, "current_device", {"return_value": 0}),
            (torch.cuda, "device", {"new": lambda i: contextlib.nullcontext()}),
            (torch.cuda, "current_stream", {"new": lambda d=None: SimpleNamespace(cuda_stream=0)}),
        ):
            st.enter_context(mock.patch.object(target, attr, **kw))

    def test_launch_e5_and_run_hand_the_kernel_the_exact_unpack(self) -> None:
        m, t, seen = self.M, self.t, []
        int8 = _rand_mask(m, 4)
        nib = nibm.pack_ref(int8)
        dz8 = torch.zeros((m + 127) // 128 * 128, DU_ROW, dtype=U8).view(E5)
        with contextlib.ExitStack() as st, _Capture() as cap:
            self._l11_stubs(st, seen)
            for mask in (int8, nib, nib):  # the second NIB call reuses the cached unpack
                l11.launch_e5(t["g8"], t["w8"], t["sa"], t["sb"], t["x"], t["u"], mask, t["mean"], t["rstd"], t["wn"],
                              t["bn"], P_DROP, t["inv"], dz8)
            out = l11.run(t["g8"], t["w8"], t["sa"], t["sb"], t["x"], t["u"], nib, t["mean"], t["rstd"], t["wn"],
                          t["bn"], P_DROP, lambda u_, w: torch.empty(m, DU_ROW, dtype=BF)[:, :D], log_active=False)
        self.assertIsNotNone(out)
        cached = getattr(nib, nibm._INT8_ATTR)
        self.assertTrue(torch.equal(cached, int8))
        self.assertEqual(seen, [int8.data_ptr()] + [cached.data_ptr()] * 3)
        self.assertEqual(nibm.counters()["unpack"], 1)  # one exact unpack per NIB mask
        self.assertEqual(cap.text.count("exact unpack to int8 for l11_outproj_lnmd.launch_e5"), 1)
        self.assertIsNone(l11.broken())

    def test_run_unpacks_a_fresh_nib_mask_and_leaves_int8_alone(self) -> None:
        m, t, seen = self.M, self.t, []
        int8 = _rand_mask(m, 5)
        nib = nibm.pack_ref(int8)
        with contextlib.ExitStack() as st, _Capture() as cap:
            self._l11_stubs(st, seen)
            for mask in (int8, nib):
                self.assertIsNotNone(l11.run(t["g8"], t["w8"], t["sa"], t["sb"], t["x"], t["u"], mask, t["mean"],
                                             t["rstd"], t["wn"], t["bn"], P_DROP,
                                             lambda u_, w: torch.empty(m, DU_ROW, dtype=BF)[:, :D], log_active=False))
        self.assertEqual(seen, [int8.data_ptr(), getattr(nib, nibm._INT8_ATTR).data_ptr()])
        self.assertEqual(cap.text.count("exact unpack to int8 for l11_outproj_lnmd.run "), 1)
        self.assertEqual(nibm.counters()["unpack"], 1)

    def test_run_e5_unpacks_before_its_launch(self) -> None:
        m, t, seen = self.M, self.t, []
        int8 = _rand_mask(m, 6)
        nib = nibm.pack_ref(int8)

        def fake_launch_e5(g8, w8, sa, sb, x, u, mask, mean, rstd, wn, bn, p, inv, dz8):  # pyre-ignore[2,3]
            seen.append(mask)
            return torch.empty_like(x), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8, D), torch.zeros(8), 8

        with mock.patch.object(l11, "_lib", object()), \
                mock.patch.object(l11, "launch_e5", side_effect=fake_launch_e5), \
                mock.patch.object(l11, "dwdb_finalize", new=lambda *a: None), \
                mock.patch.object(l11, "_operands_ok", return_value=None), \
                mock.patch.object(l11, "e5_decline_reason", return_value=None), \
                mock.patch.object(bwd_fp8, "grad_dtype_and_limit", return_value=(E5, 57344.0)), \
                mock.patch.object(bwd_fp8, "uvqk_tm_pad", return_value=128), _Capture() as cap:
            for mask in (int8, nib):
                out = l11.run_e5(t["g8"], t["w8"], t["sa"], t["sb"], t["x"], t["u"], t["wn"], t["bn"], t["mean"],
                                 t["rstd"], D, 2, P_DROP, True, mask, DU_ROW, 0, t["inv"], t["cast"], None,
                                 log_active=False)
                self.assertIsNotNone(out)
        self.assertIs(seen[0], int8)
        self.assertEqual((seen[1].dtype, tuple(seen[1].shape)), (I8, (m, D)))
        self.assertTrue(torch.equal(seen[1], int8))
        self.assertEqual(cap.text.count("exact unpack to int8 for l11_outproj_lnmd.run_e5"), 1)
        self.assertEqual(nibm.counters()["unpack"], 1)
        self.assertIsNone(l11.broken())

    def test_unpack_failure_raises_without_latching_l11(self) -> None:
        """review R1-F6: run / run_e5 unpack before their try blocks, so an unpack failure (e.g. an OOM) raises to the
        caller instead of latching item (39) for the process."""
        m, t = self.M, self.t
        nib = nibm.pack_ref(_rand_mask(m, 8))
        boom = mock.MagicMock(side_effect=RuntimeError("HIP out of memory (synthetic)"))
        with contextlib.ExitStack() as st, _Capture() as cap:
            self._l11_stubs(st, [])
            st.enter_context(mock.patch.object(nibm, "as_int8", boom))
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                l11.run(t["g8"], t["w8"], t["sa"], t["sb"], t["x"], t["u"], nib, t["mean"], t["rstd"], t["wn"],
                        t["bn"], P_DROP, lambda u_, w: torch.empty(m, DU_ROW, dtype=BF)[:, :D], log_active=False)
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                l11.run_e5(t["g8"], t["w8"], t["sa"], t["sb"], t["x"], t["u"], t["wn"], t["bn"], t["mean"],
                           t["rstd"], D, 2, P_DROP, True, nib, DU_ROW, 0, t["inv"], t["cast"], None, log_active=False)
        self.assertEqual(boom.call_count, 2)
        self.assertIsNone(l11.broken())
        self.assertTrue(l11.enabled())
        self.assertNotIn("DISABLED", cap.text)


# ---------------------------------------------------------------------------------------------------- CPU: thl
class ThlNibActiveLineCpuTest(unittest.TestCase):
    """triton_hstu_linear.py's NIB host code (module end): the ACTIVE line names the build that reads the NIB mask
    (NIB v2: the 'only' .so, l11_do8.only_impl()); the writers' gate keeps asking l11_do8.nib_ready()."""

    def test_source_pins(self) -> None:
        src = _THL.read_text()
        i0 = src.index("\ndef _lnmd_mask_nib_armed(")
        armed = src[i0:src.index("\ndef ", i0 + 1)]
        self.assertIn("(impl={l11_do8.only_impl()})", armed)
        self.assertNotIn("l11_do8.impl()", armed)
        j0 = src.index("\ndef _lnmd_mask_nib_why_not(")
        gate = src[j0:src.index("\ndef ", j0 + 1)]
        self.assertIn('getattr(l11_do8, "nib_ready", None)', gate)
        self.assertNotIn("mask_format", gate)

    def test_active_line_prints_the_only_impl(self) -> None:
        thl = _thl_or_none()
        if thl is None:
            self.skipTest("triton_hstu_linear not importable (no Triton)")
        saved = {k: v for k, v in vars(thl).items() if k.startswith("_LNMD_MASK_NIB_")}
        try:
            thl._LNMD_MASK_NIB_ARMED_LOGGED = False
            thl._LNMD_MASK_NIB_SH_LOGGED = False
            thl._LNMD_MASK_NIB_ATEXIT = True  # no atexit hook from a test
            with mock.patch.object(thl.l11_do8, "only_impl", return_value="ONLY-IMPL+mkb=nib"), \
                    mock.patch.object(thl.l11_do8, "impl", return_value="PLUS-IMPL"), _Capture() as cap:
                thl._lnmd_mask_nib_armed(False, 24)
        finally:
            for k, v in saved.items():
                setattr(thl, k, v)
        self.assertEqual(cap.text.count("armed OK: LN*U*dropout keep mask NIB ACTIVE"), 1)
        self.assertIn("the backward hands it only to DO8's NIB 'only' build (impl=ONLY-IMPL+mkb=nib)", cap.text)
        self.assertNotIn("PLUS-IMPL", cap.text)


# ---------------------------------------------------------------------------------------------------- GPU
def _gfx950() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        return l11.device_arch(torch.cuda.current_device()) == "gfx950"
    except Exception:  # noqa: BLE001
        return False


def _hipcc_ok() -> bool:
    try:
        return os.path.exists(l11._hipcc())
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_gfx950() and _hipcc_ok(), "needs a gfx950 GPU and hipcc (builds l11_do8.cu with and without "
                     "-DL11_MKB_NIB=1)")
class Do8NibGpuTest(_EnvReset):
    def setUp(self) -> None:
        super().setUp()
        os.environ[VD_KNOB] = "1"  # the diet source (the only one with the NIB reader)

    def _load(self, nib: bool, only: bool) -> None:
        do8.reset_for_testing(unload=True)
        do8.set_requested_for_testing(True)
        if nib:
            os.environ[KNOB] = "1"
        else:
            os.environ.pop(KNOB, None)
        do8._load()
        if only:
            do8._load_only()
        self.assertEqual(do8.mask_format(), "int8")  # NIB v2: 'plus' is the int8 build either way
        self.assertNotIn("+mkb=nib", do8.impl())
        if only:
            self.assertEqual(do8.only_mask_format(), "nib" if nib else "int8")
            self.assertEqual("+mkb=nib" in do8.only_impl(), nib)

    def _case(self, lens: List[int], seed: int, kind: str = "gauss") -> SimpleNamespace:
        from generative_recommenders.ops.triton import triton_hstu_linear as thl

        M = sum(lens)
        offs = [0]
        for n_ in lens:
            offs.append(offs[-1] + n_)
        g = torch.Generator(device="cuda").manual_seed(seed)
        g8f = torch.randn((M, D), generator=g, device="cuda") * 1024.0
        x = (torch.randn((M, D), generator=g, device="cuda") * 0.5).to(BF)
        if kind == "special":
            for i in range(0, len(lens), 3):
                g8f[offs[i]:offs[i] + min(8, lens[i])] = 0.0  # zero groups (E8 127)
            for i in range(1, len(lens), 5):
                s0 = offs[i]
                x[s0 + min(8, lens[i] - 1):s0 + min(16, lens[i])] = float("nan")
        g8 = g8f.clamp_(-57344.0, 57344.0).to(E5)
        w8 = (torch.randn((1536, D), generator=g, device="cuda") * 90.0).clamp_(-448.0, 448.0).to(E4)
        u = torch.randn((M, D), generator=g, device="cuda").to(BF)
        wn = (1.0 + 0.05 * torch.randn((D,), generator=g, device="cuda")).to(BF)
        bn = (0.05 * torch.randn((D,), generator=g, device="cuda")).to(BF)
        xf = x.float()
        mean, rstd = xf.mean(1), torch.rsqrt(xf.var(1, unbiased=False) + 1e-6)
        mask = thl._create_dropout_mask(N=M, D=D, BLOCK_D=D, concat_u=True, concat_x=True, dropout_ratio=P_DROP,
                                        seed=seed, device=torch.device("cuda"))
        return SimpleNamespace(M=M, N=max(lens), offs=torch.tensor(offs, dtype=torch.int64, device="cuda"), g8=g8,
                               w8=w8, sa=torch.full((1,), 2.0**-19, dtype=F32, device="cuda"),
                               sb=torch.full((1,), 0.0613 / 448.0, dtype=F32, device="cuda"), x=x, u=u, wn=wn, bn=bn,
                               mean=mean, rstd=rstd, mask=mask)

    def _launch(self, c: SimpleNamespace, only: bool) -> Tuple[Dict[str, torch.Tensor], int]:
        rows_out = (c.M + 127) // 128 * 128
        inv = torch.full((1,), 0.7371 * 2.0**-3, dtype=F32, device="cuda")
        do8._WL_MEMO.clear()
        wl, cnt, cap = do8.work_list(c.offs, c.M, c.N, c.x.device)
        dz = torch.full((rows_out, DU_ROW), 0x5A, dtype=U8, device="cuda")
        d8 = torch.full((c.M, D), 0x5A, dtype=U8, device="cuda")
        e8 = torch.full((c.M, H), 0xA5, dtype=U8, device="cuda")
        dxb = torch.full((c.M, D), do8._SENTINEL_BF16, dtype=torch.int16, device="cuda").view(BF) if only else None
        fmt = do8.only_mask_format() if only else do8.mask_format()
        mask = nibm.pack_ref(c.mask) if fmt == "nib" else c.mask
        dx, dwp, dbp, ducs, amaxp, P = do8.launch_e5_plus(c.g8, c.w8, c.sa, c.sb, c.x, c.u, mask, c.mean, c.rstd, c.wn,
                                                          c.bn, P_DROP, inv, dz.view(E5), wl, cnt, cap, d8.view(E4),
                                                          e8, only=only, dx=dxb)
        torch.cuda.synchronize()
        out = {"dz8": dz, "dO8": d8, "E8": e8, "DWP": dwp, "DBP": dbp, "DUCS": ducs, "AMAXP": amaxp, "dx": dx}
        return {k: v.contiguous().view(U8).reshape(-1).clone() for k, v in out.items()}, int(P)

    def test_plus_build_is_the_knob_off_build(self) -> None:
        self._load(nib=False, only=False)
        off = (do8.library_path(), do8.impl())
        self._load(nib=True, only=False)
        self.assertEqual((do8.library_path(), do8.impl()), off)

    def test_nib_only_build_matches_the_int8_builds_byte_for_byte(self) -> None:
        cases = [self._case(list(EDGE) * 8 + [4096, 333], seed=11), self._case([1, 7, 64, 65, 2288, 0, 640] * 5,
                                                                                 seed=12),
                 self._case(list(EDGE) * 6 + [1000], seed=13, kind="special")]
        self._load(nib=False, only=True)
        ref_only = [self._launch(c, True) for c in cases]
        ref_plus = [self._launch(c, False) for c in cases]
        res_int8 = do8.only_resources()
        self._load(nib=True, only=True)
        res_nib = do8.only_resources()
        for i, (c, (r, p_r), (rp, p_p)) in enumerate(zip(cases, ref_only, ref_plus)):
            got, p_g = self._launch(c, True)
            self.assertEqual(p_g, p_r, i)
            for k in r:  # vs the int8 'only' build: every output byte (the dx sentinel included)
                self.assertTrue(torch.equal(got[k], r[k]), (i, k))
            self.assertTrue(bool((got["dx"].view(torch.int16) == do8._SENTINEL_BF16).all()), i)
            self.assertEqual(p_g, p_p, i)
            for k in ("dz8", "dO8", "E8", "DWP", "DBP", "DUCS", "AMAXP"):  # vs EM = 1 of the int8 'plus' build
                self.assertTrue(torch.equal(got[k], rp[k]), (i, k))
        # DO8's own gate passed both builds (0 VGPR spill, 0 scratch, LDS); iter53 VERIFY B2: SGPR spill <= base + 2
        self.assertEqual((res_nib["vgpr_spill"], res_nib["scratch"]), (0, 0))
        self.assertLessEqual(res_nib["sgpr_spill"], res_int8["sgpr_spill"] + 2, (res_int8, res_nib))

    def test_run_e5_plus_routes_by_mask_format_without_a_latch(self) -> None:
        os.environ[ONLY] = "1"
        self._load(nib=True, only=True)
        c = self._case(list(EDGE) * 4, seed=21)
        do8.set_prebuilt_for_testing(True)
        do8.set_only_prebuilt_for_testing(True)
        do8.registry()  # poison_requested() for the 'only' call
        l11.set_requested_for_testing(True)
        l11.set_prebuilt_for_testing(True)
        site = SimpleNamespace(T=c.M, H=H, D=HD, N=c.N, device=c.x.device, seq_offsets=c.offs, only_plan=None)
        inv = torch.full((1,), 2.0**-3, dtype=F32, device="cuda")
        nib = nibm.pack_ref(c.mask)

        def run(mask: torch.Tensor, plan: Optional[bool]) -> Any:
            side = AssertionError("_plan_only for an int8 mask") if plan is None else None
            with mock.patch.object(do8, "_plan_only", side_effect=side, return_value=plan):
                return do8.run_e5_plus(c.g8, c.w8, c.sa, c.sb, c.x, c.u, c.wn, c.bn, c.mean, c.rstd, D, 2, P_DROP, True,
                                       mask, DU_ROW, 0, inv, 1.0 / inv, None, site)

        with mock.patch.object(bwd_fp8, "grad_dtype_and_limit", return_value=(E5, 57344.0)), \
                mock.patch.object(bwd_fp8, "uvqk_tm_pad", return_value=128), _Capture() as cap:
            plus = run(c.mask, None)  # int8 -> 'plus' (the int8 .so), no plan
            only = run(nib, True)  # NIB planned "only" -> the NIB 'only' .so
            declined = run(nib, False)  # NIB planned "plus" -> decline
            torch.cuda.synchronize()
        self.assertIsNotNone(plus, cap.text[-2000:])
        self.assertIsNotNone(only, cap.text[-2000:])
        self.assertIsNone(declined)
        for k in (4, 5, 6):  # emit / dO8 / E8: dz8's u block + pad rows (torch.empty: DO8 writes [:, :512]), dO8, E8
            a, b = (plus[k].dz8[:, :D], only[k].dz8[:, :D]) if k == 4 else (plus[k], only[k])
            self.assertTrue(torch.equal(a.contiguous().view(U8), b.contiguous().view(U8)), k)
        self.assertIsNone(do8.broken())
        self.assertIsNone(do8.only_broken())
        cnt = do8.counters()
        self.assertEqual((cnt["mask_int8_plus"], cnt["mask_nib_plus"], cnt["mask_declined"]), (1, 1, 1))
        self.assertEqual(nibm.counters()["unpack"], 0)
        self.assertEqual(cap.text.count("NOT armed: E5 mask format"), 1)

    def test_nib_self_checks_pass(self) -> None:
        os.environ[ONLY] = "1"
        self._load(nib=True, only=True)
        ok, detail = do8.self_check()
        self.assertTrue(ok, detail)
        self.assertNotIn("mask nib", detail)  # 'plus' is int8: ship15's self-check
        ok, detail = do8.self_check_only()
        self.assertTrue(ok, detail)
        self.assertIn("mask nib(uint8[M,256]) unpack(pack_ref) exact 1", detail)


if __name__ == "__main__":
    unittest.main()

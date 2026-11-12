# pyre-unsafe
"""outproj_dual_mx6: r14's MX6 dY producer (F1) and the L11 W6 / W6S weight shadow (W1) -- one HIP .so and its loader.

r14 iteration 66, producer track, DESIGN v1 (FROZEN): probes/iter66/producer/design/DESIGN.md sec. 3.1 (C ABI), 3.2
(this module's API), 10 (W1 cache), 12 (build, gate, load, prebuild, self-check, latch, lines). DEFAULT OFF and no knob
of its own: r16's DLRM_HSTU_FP8_BWD_OUTPROJ_MXFP6 (the l11_do8 MX6 row) is the only production switch, and the only
callers are bwd_fp8.py's EOF append (S1 output_projection_dgrad_operands_mx6, S1-E output_projection_mx6_encode, the
optional prebuild hook prebuild_outproj_mx6, which r16 does not call). Importing this module builds, compiles, loads,
launches and prints nothing, and touches no GPU.

WHAT. ops/hip/outproj_dual_mx6.cu (DESIGN v1 Appendix A; SOURCE_SHA256) holds two kernels in one .so.
  outproj_dual_mx6_kernel   one 512-thread workgroup per 256-row band (= the Triton dual cast's pid_m), all 512
      features. ONE read of the bf16 out-proj gradient dout [M, 512] (1 <= M <= M_MAX) writes
        G6        uint8 [M, 400] row records, byte-exact REF mx6_l11_ref.encode_g6 (neg_zero=True): E2M3 heads [0, 256)
                  and tails [256, 384), E8M0 [384, 400) in [q][k] order; exactly M rows (no padding row is written).
                  Byte-exact is the .cu / CPU-model target, pinned strict by T12 / T13 / T18; the in-process gates
                  SC1-SC3 / SC6 and the D16 audit accept S1C 2.5 as amended (L2(a), v1.2: g6_l2a_counts);
        dout_t    e4m3 step-tiled [M_PAD/128, 512, 128] (M_PAD = roundup(M, 128)), bitwise what the shipped
                  bwd_fp8._quantize_amax_dual_kernel writes as DST_T with SEPARATE_T, T_INV_MUL 2^-t (t = 4..7),
                  LIMIT_T 448 and T_TILED; the pad tokens [M, M_PAD) go through the same arithmetic on +0.0;
        PARTIAL   fp32 [4 * ceil(M/256)]: the dual cast's 256 x 128-tile maxnum, entry for entry (NaN only for an
                  all-NaN tile; NaN + Inf -> Inf; masked rows count as +0.0);
        INV_USED  the raw uint32 bits of *INV the kernel multiplied by (nullable; S1 passes NULL, the self-check's
                  SC4 and the D10 form write it).
      INV is a DEVICE fp32 [1], never a host float: S1 passes its own pre-publish clone of bwd_fp8's state.inv_scale
      (the bits R2 re-casts with; v1.1-1 of the seam notes). The kernel writes no g8.
  outproj_w6_shadow_kernel<BF16>   the raw-mode, L11-permutation W6 [589,824 B] / W6S [768 B] image of the [1536, 512]
      out-proj weight (fp32 or bf16 master), REF quant_w (neg_zero=True) + w6_layout.w6_pack byte for byte.

API (DESIGN 3.2). FAMILY, ABI_VERSION, M_MAX, LDS_BYTES, IMPL (the .cu impl literal up to "; src="), KERNELS (the
mangled names the resource gate requires; pinned on the ROCm 7.2.0 proxy, re-pin at lab gate P0).
  prebuild()            idempotent and process-latched: GPU / gfx950 check, build or cache hit, resource gate, load,
                        self-check; raises the latched reason on failure (S1-E relies on that).
  not_armed_reason()    never builds; None iff prebuilt + gated + self-checked + not latched (S1's persistent check).
  failure()             the latched reason, or None.
  launch(dout, inv_scale, t_inv_mul, M, m_pad, g6, dout_t, partial, inv_used)   one producer launch on
                        torch.cuda.current_stream(dev) inside torch.cuda.device(dev); metadata checks first
                        (ineligible_reason: the C launcher's checks plus shapes / dtypes / devices; ValueError);
                        rc > 0: OutprojMx6Error naming the hipError (nothing launched, outputs not written);
                        rc < 0: launched, -rc is a stale error an EARLIER call left on this thread: one
                        "[outproj-mx6] stale HIP error ..." line per process and code, then return (D15).
  w6_shadow(weight)     (w6, w6s), cached (sec. 10): k512_mx6_gemm.mx6_weight's protocol; a refresh is ONE launch into
                        NEW tensors on the current stream; never stale while writers bump _version.
  w6_encode(weight)     (w6, w6s), uncached (S1-E, the self-check). clear_w6_cache(), w6_cache_stats(),
                        reset_for_testing(), set_prebuilt_for_testing().

BUILD (prebuild only; never in the timed region). hipcc = $DLRM_HSTU_FP8_OUTPROJ_MX6_HIPCC, else $ROCM_PATH/bin/hipcc,
else /opt/rocm/bin/hipcc. Flags (flags()): --offload-arch=gfx950 -std=c++20 -w -O3 -ffp-contract=off
-fno-slp-vectorize -I<rocm>/include/hip -shared -fPIC -Rpass-analysis=kernel-resource-usage --save-temps
-DOUTPROJ_MX6_SRC_SHA256="<sha256 of the .cu>" (+ dev defines), run in a private cwd (--save-temps writes its
temporaries there; the device .s gives the code sizes). Cache: /tmp/dlrm_hstu_fp8_outproj_mx6/<key>/
liboutproj_dual_mx6.so + liboutproj_dual_mx6.so.resources.json, key = sha256(source bytes, "\\0".join(flags), "gfx950",
`hipcc --version`, HIPCC_COMPILE_FLAGS_APPEND, HIPCC_LINK_FLAGS_APPEND, HIP_CLANG_PATH, HIP_DEVICE_LIB_PATH)[:16]
(the l11_outproj_lnmd / l11_do8 / ec_dgrad_x2h lineage). Racing ranks compile to per-host/pid/thread temporaries; the
sidecar is os.replace()d before the .so (whoever sees the .so finds its census); temporaries are unlinked in
`finally`. A root or entry directory that is not ours, or is group / world writable, is not trusted: the process
builds into a private mkdtemp directory. A cache entry that fails its sidecar, gate, load, ABI or impl check is
deleted and rebuilt once (privately, so the new library has a path never dlopen()ed). hipcc runs in its own process
group; a build past 600 s kills the whole group. A failed compile raises with the compiler's first error line (and
its source line) up front, and keeps hipcc's full output, argv first, in a private mkdtemp directory
outproj_mx6_hipcc_fail_*/hipcc.log named in the error.

GATE (check_resources; by EXACT mangled name, from the -Rpass-analysis=kernel-resource-usage remarks in any format:
"<file>:<l>:<c>: remark: <Key>: <v>" and, with --save-temps, "remark: <file>:<l>:<c>: <Key>: <v>"; ROCm 10.2 prints
"<file>:<l>:<c>: <Key>: <v>" either way, no "remark:" or suffix, the census twice with --save-temps). Exactly the three
KERNELS blocks must be present (a missing production block fails closed). producer: VGPR <= 128, occupancy >= 4, LDS ==
LDS_BYTES, VGPR spill 0, SGPR spill 0, scratch 0, no dynamic stack. W6 kernels: VGPR / SGPR spill 0, scratch 0, no
dynamic stack. VGPR, SGPR, occupancy, LDS and code size go to the sidecar and the armed line.

LOAD. ctypes.CDLL (RTLD_LOCAL) once per process under a lock; one SIGNATURES table; abi_version() == ABI_VERSION;
impl() starts with IMPL and ends with "; src=<sha256 of the built source>"; lds_bytes() == LDS_BYTES; partial_count()
== 4 * ceil(M/256); variant() recorded (bit0 G6_F32, bit1 DN1, bit2 REVERSE, bit3 WPE, PF << 4).

SELF-CHECK (self_check(); synthetic and deterministic: private numpy generators and integer hashes only, no global RNG
draw, plain torch buffers, no model / data / scale state, no TM pool, no _sites entry). Every producer case runs twice,
on outputs poisoned 0xA5 and then 0x5A, with guards (G6 + 4,096 B, dout_t + one step, PARTIAL + 4 entries, INV_USED + 3
words) that must stay untouched, and the two runs must agree bitwise.
  SC0  the C launchers reject bad arguments with rc 1 (hipErrorInvalidValue) and launch nothing (host only).
  SC1  M = 769 (t = 5 / 6 / 7 plus every other t this process can cast at: bwd_fp8's SHIFT_FLOOR and a fixed
       DFMT_SHIFT; sc1_shifts()) and M = 700 (t = 5; also with 16-B-only aligned
       bases): production-like hash rows (0.5 % all +0 and 0.5 % all -0 rows) with U1's 30 adversarial rows (incl. the
       position row) at [0, 30), across the 128 / 256 / 384 boundaries and in the last 30 rows; inv = publish(143)
       bits 0x424880e5. G6 vs the in-tree integer encoder (encode_g6_int) under S1C 2.5 as amended (L2(a); v1.2,
       r16 iteration 59 R-4; g6_l2a_counts): E8M0 bytes and every code of a byte-255 block strict; a code may differ
       from REF only in the sign bit of an E2M3 zero in a block whose E8M0 bytes are equal (value-neutral: counted per
       direction and reported in the detail, not gated). dout_t (pad tokens included) and PARTIAL (non-NaN bits and
       NaN-ness) vs ONE direct _quantize_amax_dual_kernel[grid](...) launch with quantize_row_major_dual's exact
       arguments (triton_dual_reference); INV_USED bits.
  SC2  critique C4's tiles: P1 / P2 / P4 (M 256), P3 (M 100, INV_USED NULL), P3b (M 128), P3c (M 384).
  SC3  M = 300, inv NaN / +Inf / 0 / -1, t = 5 (pad tokens through the arithmetic).
  SC4  INV_USED bits for inv = publish(143) and 0x7fc00000 (inside SC1 / SC3).
  SC5  W6 / W6S on U3's edge weight and a hash-generated N(0, 0.02)-like weight, fp32 and bf16, vs the in-tree
       quant_w_int + w6_scatter (poisoned twice, guards), and w6_encode's allocating path.
  SC6  a determinism twin at M = 4,097 (bitwise G6, dout_t, PARTIAL).
  The SC1 / SC2 / SC3 / SC6 G6 counts follow SC1's L2(a) rule; the two runs of every case stay byte-strict (twins).
  A G6 count is failing E8M0 bytes plus failing 6-bit codes (v1.1 counted mismatching bytes). The value-neutral codes
  close the detail, after the timing and any FAILING list, so r16's 600-character re-log of a failed self-check keeps
  the counts and the FAILING list.
  Any mismatch or non-device exception latches (one DISABLED line). An exception whose text matches
  l11_outproj_lnmd._DEVICE_FAULT re-raises (abort before RUN_START, r16's prebuild_mx6 policy) after one line.

LATCH / LINES (every rank via ops.utils.arm_print_all_ranks; every line starts with FAMILY and contains none of r16's
must-be-ABSENT tokens "MX6 DISABLED", "MX6 INERT", "MX6 release", "MX6 rescue", "DEVICE FAULT", case-insensitive;
every line is ONE physical line, whitespace runs such as an exception text's newlines collapsed to one space, and so is
the latched reason that OutprojMx6Error and r16's re-logged line carry: _say() / _fail() enforce all three):
  "[outproj-mx6] armed OK: prebuild before run_start: built+loaded+self-check OK <s>s (VGPR .., SGPR .., spill 0/0,
   scratch 0, LDS 67712, occupancy 4, code .. B; ...; so <key> ...; impl=...; variant=..; src=<path>@<sha12>; ...)"
  "[outproj-mx6] DISABLED: <reason> -> FP8 dual cast for this process" (once; the latch)
  "[outproj-mx6] stale HIP error <code> (<name>) drained at <entry> ..." (once per process and code)
  "[outproj-mx6] prebuild self-check hit a device-side error ...; re-raised ..." (then the exception propagates)

DEV-ONLY OVERRIDES (never set in production): DLRM_HSTU_FP8_OUTPROJ_MX6_SRC (another .cu with the same ABI; its bytes
enter the key), _DEFINES (lab variant macros -DOUTPROJ_MX6_<NAME>[=<v>] and the pair "-mllvm
-amdgpu-sched-strategy=iterative-ilp"; they enter the flags, hence the key), _HIPCC, _CACHE (the cache root),
_FAIL_CLOSED=1 (a failure raises without latching), _AUDIT=1 (D16: the first launch after prebuild with M >= 2^20
rows, i.e. the first production-scale S1 call whatever INV_USED form the seam uses (the prebuild and r16 self-checks
run far smaller M), re-checks its last two bands against the in-tree G6 encoder (SC1's L2(a) rule) and a direct
Triton launch, synchronizing; a mismatch latches; audit_last_bands() is also callable directly, lab gate P9).
"""

import ctypes
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from generative_recommenders.ops.utils import arm_print_all_ranks

# ---------------------------------------------------------------------------- public constants (DESIGN 3.2)
FAMILY: str = "[outproj-mx6]"
ABI_VERSION: int = 1  # outproj_dual_mx6.cu ODM6_ABI_VERSION
M_MAX: int = 4_194_304  # outproj_dual_mx6_launch's M bound (= l11_outproj_lnmd.M_MAX)
LDS_BYTES: int = 67_712  # producer static LDS: the 16 x 33 x 128 B dout_t step image + 128 B PARTIAL scratch
# The .cu's impl literal up to "; src=" (outproj_dual_mx6_impl() == IMPL + "; src=" + the source sha256).
IMPL: str = (
    "outproj_dual_mx6 v1 (r14 iter66 DESIGN v1): 8-wave WG per 256-row band; 4x16 load map with block halves 16 lanes "
    "apart + permlane16_swap; pk32_fp6_bf16 G6 (G6_F32=1: f32 pre-scale + 2xpk16); 33-slot LDS dout_t step stage, "
    "whole-line copy-out; +1-encoded maxnum PARTIAL; buffer-OOB masking; W1 raw-mode L11 wshadow fork"
)
# outproj_dual_mx6.cu as frozen in DESIGN v1 Appendix A (CPU tests pin the file to it: an edited kernel needs the GPU
# lab gates again; the lab winner's one-line default edit changes it together with this constant).
SOURCE_SHA256: str = "e6c885b51cd359e249a350def3a6ce2b620b4752d0c468ab0989f98f290a84b3"
# The production instantiations' mangled names (ROCm 7.2.0 proxy; re-pin at lab gate P0). The gate is by exact name.
KERNELS: Dict[str, str] = {
    "producer": "_ZN4odm623outproj_dual_mx6_kernelEPKhPKffiiPhS4_PfPj",
    "w6_f32": "_ZN4odm624outproj_w6_shadow_kernelILb0EEEvPKvPhS3_",
    "w6_bf16": "_ZN4odm624outproj_w6_shadow_kernelILb1EEEvPKvPhS3_",
}
D: int = 512  # dout width = the out-proj K
NOUT: int = 1536  # out-proj weight rows (u | x | y planes)
G6_ROW: int = 400  # bytes per G6 row record
W6_BYTES: int = 589_824  # the pre-swizzled W6 image of the [1536, 512] weight
W6S_BYTES: int = 768  # the W6 E8M0 plane [48][q][k]
BAND_ROWS: int = 256  # rows per producer workgroup (= Triton BLOCK_M; 4 PARTIAL entries)
STEP_TOKENS: int = 128  # dout_t step
STEP_BYTES: int = D * STEP_TOKENS  # 65,536
T_SHIFTS: Tuple[int, ...] = (4, 5, 6, 7)
T_INV_MUL_BITS: Tuple[int, ...] = (0x3D800000, 0x3D000000, 0x3C800000, 0x3C000000)  # fp32 2^-4 .. 2^-7
PRODUCER_VGPR_MAX: int = 128  # above it one workgroup per CU (DESIGN 7: the cliff)
PRODUCER_OCCUPANCY_MIN: int = 4
VARIANT_FLAGS: Tuple[str, ...] = ("G6_F32", "DN1", "REVERSE", "WPE")  # outproj_dual_mx6_variant() bits 0..3; PF << 4

# Dev-only overrides (module docstring): never set in production.
_ENV: str = "DLRM_HSTU_FP8_OUTPROJ_MX6"
SRC_ENV: str = _ENV + "_SRC"
DEFINES_ENV: str = _ENV + "_DEFINES"
HIPCC_ENV: str = _ENV + "_HIPCC"
CACHE_ENV: str = _ENV + "_CACHE"
FAIL_CLOSED_ENV: str = _ENV + "_FAIL_CLOSED"
AUDIT_ENV: str = _ENV + "_AUDIT"

_ARCH: str = "gfx950"
_SOURCE: Path = Path(__file__).with_name("outproj_dual_mx6.cu")
_COMPILE_FLAGS: Tuple[str, ...] = ("-std=c++20", "-w", "-O3", "-ffp-contract=off", "-fno-slp-vectorize")
_LINK_FLAGS: Tuple[str, ...] = ("-shared", "-fPIC", "-Rpass-analysis=kernel-resource-usage", "--save-temps")
# hipcc reads these; they change its output, so they are part of the cache key (ec_dgrad_x2h._HIPCC_ENV)
_HIPCC_ENV_KEYS: Tuple[str, ...] = (
    "HIPCC_COMPILE_FLAGS_APPEND",
    "HIPCC_LINK_FLAGS_APPEND",
    "HIP_CLANG_PATH",
    "HIP_DEVICE_LIB_PATH",
)
_CACHE_ROOT: str = "/tmp/dlrm_hstu_fp8_outproj_mx6"
_LIB_NAME: str = "liboutproj_dual_mx6.so"
_META_NAME: str = _LIB_NAME + ".resources.json"
_BUILD_TIMEOUT_S: int = 600
_DEFINE_RE = re.compile(r"-DOUTPROJ_MX6_[A-Z0-9_]+(=[A-Za-z0-9_.+-]+)?")
_ILP_FLAGS: Tuple[str, str] = ("-mllvm", "-amdgpu-sched-strategy=iterative-ilp")
# r16's must-be-ABSENT tokens (l11_do8 MX6 lines); no line of this module may contain one (DESIGN T19)
_ABSENT_TOKENS: Tuple[str, ...] = ("MX6 DISABLED", "MX6 INERT", "MX6 release", "MX6 rescue", "DEVICE FAULT")


class OutprojMx6Error(RuntimeError):
    """The producer / W6 library is not usable in this process (latched, not loaded) or one of its launches failed
    (rc > 0). bwd_fp8's S1 raises before its commit on it; r16's seam then latches MX6 (FP8 dual cast)."""


# ---------------------------------------------------------------------------- state
_lock = threading.Lock()  # guards _lib (one CDLL per process)
_prebuild_lock = threading.RLock()  # serializes prebuild() (idempotent, process-latched)
_lib: Optional["_Lib"] = None
_failure: Optional[str] = None  # the latch
_prebuilt: bool = False
_armed: Dict[str, Any] = {}  # prebuild summary (tests / diagnostics)
_once_keys: set = set()
_AUDIT: Dict[str, Any] = {"pending": None}
_AUDIT_MIN_M: int = 1 << 20  # the dev audit's trigger: production-scale calls only (self-checks run <= 64K rows)


# ---------------------------------------------------------------------------- lines / latch
def _sanitize(text: str) -> str:
    """No r16 must-be-ABSENT token in any line of this module, whatever an exception text carries (T19)."""
    for tok in _ABSENT_TOKENS:
        text = re.sub(re.escape(tok), lambda m: m.group(0).replace(" ", "_"), text, flags=re.I)
    return text


def _one_line(text: Any) -> str:
    """`text` as ONE physical line: every whitespace run (an exception text's or a hipcc log's newlines) becomes one
    space, so a line keeps its FAMILY prefix and arm_print_all_ranks' " [rank N]" tag on the same line (T19)."""
    return " ".join(str(text).split())


def _say(msg: str) -> None:
    msg = _one_line(msg)
    if not msg.startswith(FAMILY):
        msg = FAMILY + " " + msg
    arm_print_all_ranks(_sanitize(msg))


def _say_once(key: Any, msg: str) -> None:
    if key in _once_keys:
        return
    _once_keys.add(key)
    _say(msg)


def _fail_closed() -> bool:
    """DLRM_HSTU_FP8_OUTPROJ_MX6_FAIL_CLOSED=1 (smoke / debug only): a failure raises without latching."""
    return os.environ.get(FAIL_CLOSED_ENV, "0").strip() == "1"


def _fail(reason: str) -> RuntimeError:
    """Latch the producer off for this process (one DISABLED line) and return the exception to raise. Under
    _FAIL_CLOSED=1: no latch, no line, the error says so. The reason is kept on one line (_one_line): it is printed,
    returned by failure() / not_armed_reason() and carried by OutprojMx6Error, which r16's prebuild_mx6 re-logs."""
    global _failure
    reason = _one_line(reason)
    if _fail_closed():
        return RuntimeError(f"{FAMILY} failed with {FAIL_CLOSED_ENV}=1: {reason}")
    if _failure is None:
        _failure = reason
        _say(f"{FAMILY} DISABLED: {reason} -> FP8 dual cast for this process")
    return OutprojMx6Error(f"{FAMILY} unavailable: {_failure}")


_MODS: Dict[str, Any] = {}


def _l11() -> Any:
    """generative_recommenders.ops.hip.l11_outproj_lnmd, imported at first use (its _toolchain, device_arch, _a16 and
    _DEVICE_FAULT are reused here, as r16's l11_do8 reuses them)."""
    mod = _MODS.get("l11")
    if mod is None:
        from generative_recommenders.ops.hip import l11_outproj_lnmd as mod

        _MODS["l11"] = mod
    return mod


def _device_fault(text: str) -> bool:
    """The same classification as r16's prebuild_mx6 and L11's prebuild (l11_outproj_lnmd._DEVICE_FAULT)."""
    return bool(_l11()._DEVICE_FAULT.search(text))


def not_armed_reason() -> Optional[str]:
    """None iff the library is built, gated, loaded and self-checked in this process and not latched; else why not.
    Host state only: never builds, never touches the GPU (S1's persistent-condition check)."""
    if _failure is not None:
        return f"DISABLED in this process ({_failure})"
    if not _prebuilt:
        return ("not prebuilt in this process (outproj_dual_mx6.prebuild() runs before RUN_START; nothing is built "
                "after it)")
    return None


def failure() -> Optional[str]:
    """The latched reason, or None."""
    return _failure


def prebuilt() -> bool:
    return _prebuilt


def armed_info() -> Dict[str, Any]:
    """The prebuild summary (resources, key, impl, variant, self-check detail); empty until prebuild() passed."""
    return dict(_armed)


# ---------------------------------------------------------------------------- build
def source_path() -> Path:
    override = os.environ.get(SRC_ENV, "").strip()
    return Path(override) if override else _SOURCE


def hipcc_path() -> str:
    """$DLRM_HSTU_FP8_OUTPROJ_MX6_HIPCC, else $ROCM_PATH/bin/hipcc, else /opt/rocm/bin/hipcc (the X2H / L11 rule)."""
    override = os.environ.get(HIPCC_ENV, "").strip()
    if override:
        return shutil.which(override) or override
    rocm = os.environ.get("ROCM_PATH", "").strip()
    if rocm:
        cand = os.path.join(rocm, "bin", "hipcc")
        if os.path.exists(cand):
            return cand
    return "/opt/rocm/bin/hipcc"


def dev_defines() -> List[str]:
    """The lab variant flags of DLRM_HSTU_FP8_OUTPROJ_MX6_DEFINES (dev only): -DOUTPROJ_MX6_<NAME>[=<v>] tokens and the
    pair "-mllvm -amdgpu-sched-strategy=iterative-ilp"; anything else (or a SRC_SHA256 define) raises."""
    raw = os.environ.get(DEFINES_ENV, "").strip()
    if not raw:
        return []
    toks = shlex.split(raw)
    out: List[str] = []
    i = 0
    while i < len(toks):
        if tuple(toks[i:i + 2]) == _ILP_FLAGS:
            out.extend(_ILP_FLAGS)
            i += 2
            continue
        tok = toks[i]
        if not _DEFINE_RE.fullmatch(tok) or tok.startswith("-DOUTPROJ_MX6_SRC_SHA256"):
            raise ValueError(f"{DEFINES_ENV}={raw!r}: {tok!r} is not a -DOUTPROJ_MX6_<NAME>[=<value>] macro or the "
                             "pair -mllvm -amdgpu-sched-strategy=iterative-ilp")
        out.append(tok)
        i += 1
    return out


def flags(src: bytes, hipcc: Optional[str] = None) -> List[str]:
    """The exact hipcc flags (DESIGN 12.2) for source bytes `src`, in order."""
    hipcc = hipcc or hipcc_path()
    sha = hashlib.sha256(src).hexdigest()
    inc = Path(hipcc).parent.parent / "include" / "hip"
    return [f"--offload-arch={_ARCH}", *_COMPILE_FLAGS, f"-I{inc}", *_LINK_FLAGS,
            f'-DOUTPROJ_MX6_SRC_SHA256="{sha}"', *dev_defines()]


def build_command(source: Optional[Path] = None, out: Optional[Path] = None) -> List[str]:
    """The exact hipcc argv build() runs (the lab build-only check P0 reuses it in its own cwd to keep the
    --save-temps ISA)."""
    source = Path(source) if source is not None else source_path()
    hipcc = hipcc_path()
    out = Path(out) if out is not None else Path(_LIB_NAME)
    return [hipcc, str(source.resolve()), *flags(source.read_bytes(), hipcc), "-o", str(out.resolve())]


def _toolchain(hipcc: str) -> str:
    """`hipcc --version`, cached per hipcc path (l11_outproj_lnmd._toolchain: raises if hipcc cannot run)."""
    return _l11()._toolchain(hipcc)


def cache_key(source: Optional[Path] = None) -> str:
    """sha256(source bytes, flags, arch, `hipcc --version`, the HIPCC_* / HIP_* env)[:16]: a .so another hipcc or
    other flags built is never reused, so the resource gate holds for the library that is loaded."""
    source = Path(source) if source is not None else source_path()
    src = source.read_bytes()
    hipcc = hipcc_path()
    h = hashlib.sha256()
    h.update(src)
    h.update(b"\0" + "\0".join(flags(src, hipcc)).encode())
    h.update(b"\0" + _ARCH.encode())
    h.update(b"\0" + _toolchain(hipcc).encode())
    h.update(b"\0" + "\0".join(f"{k}={os.environ.get(k, '')}" for k in _HIPCC_ENV_KEYS).encode())
    return h.hexdigest()[:16]


def cache_root() -> Path:
    return Path(os.environ.get(CACHE_ENV, "").strip() or _CACHE_ROOT)


def _trusted(p: Path) -> bool:
    """Ours and not group / world writable (ec_dgrad_x2h._trusted: a planted .so must never be loaded)."""
    from generative_recommenders.ops.hip import ec_dgrad_x2h as x2h

    return x2h._trusted(p)


def _private_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="outproj_mx6_private_"))


def _entry_dir(key: str) -> Tuple[Path, bool]:
    """(directory, shared): the shared cache entry <root>/<key> (created 0700) when the root and the entry are trusted,
    else a private mkdtemp directory."""
    root = cache_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        d = root / key
        d.mkdir(mode=0o700, exist_ok=True)
        if _trusted(root) and _trusted(d):
            return d, True
    except OSError:
        pass
    return _private_dir(), False


def _hipcc_run(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    """hipcc in its own process group in `cwd` (--save-temps writes there); on the timeout the whole group (clang
    children too) is killed (ec_dgrad_x2h._hipcc_run, plus the private cwd)."""
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=_BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise RuntimeError(f"hipcc timed out after {_BUILD_TIMEOUT_S}s (process group killed): {' '.join(cmd)}")
    return proc.returncode, out or ""


_ERROR_LINE_RE = re.compile(r"\berror\b", re.I)
_DIAG_CHARS: int = 300  # the diagnostic's share of the latched reason (prebuild and r16 keep 600 chars)


def _hipcc_diagnostic(log: str) -> str:
    """The compiler's own words for a failed build, on one line and at most _DIAG_CHARS characters: the first line
    naming an error plus the source line clang prints under it, and how many more error lines follow; with no error
    line, the log's last three non-empty lines."""
    lines = [ln.strip() for ln in log.splitlines()]
    errs = [i for i, ln in enumerate(lines) if _ERROR_LINE_RE.search(ln)]
    if errs:
        i = errs[0]
        text = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if nxt and not set(nxt) <= set("^~ ") and not _ERROR_LINE_RE.search(nxt):
            text += " | " + nxt
        if len(errs) > 1:
            text += f" (+{len(errs) - 1} more error lines)"
    else:
        text = " | ".join([ln for ln in lines if ln][-3:]) or "(no output)"
    text = _one_line(text)
    return text if len(text) <= _DIAG_CHARS else text[:_DIAG_CHARS - 3] + "..."


def _keep_build_log(cmd: List[str], log: str) -> str:
    """A failed build's full hipcc output (argv on the first line) in a private mkdtemp directory: the build cwd and
    the cache entry are cleaned up, the log is what P0 / a production DISABLED line needs. Its path, or why not."""
    try:
        d = Path(tempfile.mkdtemp(prefix="outproj_mx6_hipcc_fail_"))
        p = d / "hipcc.log"
        p.write_text(" ".join(cmd) + "\n" + log)
        return str(p)
    except OSError as error:
        return f"not kept: {type(error).__name__}"


# -Rpass-analysis=kernel-resource-usage remarks, all formats:
#   "<file>:<l>:<c>: remark: <Key>: <v> [-Rpass-analysis=kernel-resource-usage]"   (no --save-temps)
#   "remark: <file>:<l>:<c>: <Key>: <v> [-Rpass-analysis=kernel-resource-usage]"   (--save-temps)
#   "<file>:<l>:<c>: <Key>: <v>"   (format C: ROCm 10.2 / AMD clang 24 either way; no "remark:", no suffix)
# <file> may hold spaces (a checkout under "my repo/"): format B's optional file group takes any characters up to the
# ":<l>:<c>:" that the remark's key follows (v1.1, ODM6-LDR-1; it was \S+?, so such a log parsed to no census and the
# gate failed closed).
_REMARK_RE = re.compile(
    r"remark:(?:\s+.+?:\d+:\d+:)?\s+([A-Za-z][A-Za-z /\[\]]*?):\s+(\S+)\s+\[-Rpass-analysis=kernel-resource-usage\]"
)
# Format C has neither token: anchored on the line's leading "<file>:<l>:<c>:" (no ": " inside <file>), the remark's
# own labels and a value that ends the line, so a warning / note line or other "Key: value" text never matches.
_REMARK_C_RE = re.compile(
    r"^(?:(?!: ).)+?:\d+:\d+:\s+(Function Name|TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|"
    r"Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]):\s+(\S+)\s*$"
)
_FORMAT_A_RE = re.compile(r"^\S+:\d+:\d+:\s+remark:")
_FORMAT_B_RE = re.compile(r"^remark:\s+.+?:\d+:\d+:")


def parse_remarks(log: str) -> Dict[str, Dict[str, str]]:
    """{mangled kernel name: {remark key: value}} from a hipcc log, any remark format. A kernel reported twice
    must report the same values (else RuntimeError: an ambiguous census fails closed)."""
    blocks: List[Tuple[str, Dict[str, str]]] = []
    cur: Optional[Dict[str, str]] = None
    for line in log.splitlines():
        m = _REMARK_RE.search(line) or _REMARK_C_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2)
        if key == "Function Name":
            cur = {}
            blocks.append((val, cur))
            continue
        if cur is None:
            continue
        if key in cur and cur[key] != val:
            raise RuntimeError(f"resource remarks: {blocks[-1][0]} reports {key} twice ({cur[key]} / {val})")
        cur[key] = val
    out: Dict[str, Dict[str, str]] = {}
    for name, kv in blocks:
        prev = out.get(name)
        if prev is not None and prev != kv:
            raise RuntimeError(f"resource remarks: two different blocks for {name}")
        out[name] = kv
    return out


def remark_formats(log: str) -> Dict[str, int]:
    """How many remark lines of each format the log holds (A: '<file>:..: remark:', B: 'remark: <file>:..:'; C, ROCm
    10.2's '<file>:..: <Key>: <v>', is listed only when present, so a ROCm 7.2 log's count reads as before)."""
    a = b = c = 0
    for line in log.splitlines():
        if "-Rpass-analysis=kernel-resource-usage" not in line:
            if _REMARK_C_RE.match(line):  # format C carries no suffix
                c += 1
            continue
        if _FORMAT_A_RE.match(line):
            a += 1
        elif _FORMAT_B_RE.match(line):
            b += 1
    return {"A": a, "B": b, "C": c} if c else {"A": a, "B": b}


def code_sizes(asm: str) -> Dict[str, int]:
    """{kernel: codeLenInByte} from a device .s (the '; Kernel info:' block after each function's .size line)."""
    out: Dict[str, int] = {}
    cur: Optional[str] = None
    for line in asm.splitlines():
        m = re.match(r"\s*\.size\s+(\S+),\s*\.Lfunc_end\d+-", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"\s*;\s*codeLenInByte\s*=\s*(\d+)", line)
        if m and cur is not None:
            out[cur] = int(m.group(1))
            cur = None
    return out


def _code_sizes_in(work: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    # ROCm 10.2 names the device .s "<src>.cu-hip-amdgpu-amd-amdhsa.hipfb.amdgpu9.50.gfx950.img.lto.s" (no "amdgcn")
    for s in sorted({*work.glob("*amdgcn*" + _ARCH + "*.s"), *work.glob("*amdgpu*" + _ARCH + "*.s")}):
        try:
            out.update(code_sizes(s.read_text(errors="replace")))
        except OSError:
            pass
    return out


_GATE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("VGPRs", "vgpr"),
    ("TotalSGPRs", "sgpr"),
    ("VGPRs Spill", "vgpr_spill"),
    ("SGPRs Spill", "sgpr_spill"),
    ("ScratchSize [bytes/lane]", "scratch"),
    ("Occupancy [waves/SIMD]", "occupancy"),
    ("LDS Size [bytes/block]", "lds"),
)


def check_resources(kernels: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """The resource gate (DESIGN 12.4), by exact mangled name: exactly the KERNELS blocks; producer VGPR <= 128,
    occupancy >= 4, LDS == LDS_BYTES; every kernel 0 VGPR spill, 0 SGPR spill, 0 scratch, no dynamic stack. Returns
    {role: {vgpr, sgpr, agpr, vgpr_spill, sgpr_spill, scratch, occupancy, lds}}; raises RuntimeError otherwise (a
    missing block or key fails closed)."""
    names = set(KERNELS.values())
    extra = sorted(n for n in kernels if n not in names)
    if extra:
        raise RuntimeError(f"resource gate: unexpected kernel(s) {extra} in the build log (exactly the {len(names)} "
                           "KERNELS instantiations)")
    out: Dict[str, Dict[str, int]] = {}
    for role, name in KERNELS.items():
        k = kernels.get(name)
        if not k:
            raise RuntimeError(f"resource gate: no resource remarks for the {role} kernel {name} (fail closed: every "
                               "production kernel needs its census)")
        vals: Dict[str, int] = {}
        for key, short in _GATE_KEYS:
            v = k.get(key, "")
            if not re.fullmatch(r"\d+", v):
                raise RuntimeError(f"resource gate: {role} {key!r} = {v!r} (missing or not a count)")
            vals[short] = int(v)
        agpr = k.get("AGPRs", "")
        vals["agpr"] = int(agpr) if re.fullmatch(r"\d+", agpr) else -1
        dyn = k.get("Dynamic Stack", "False")
        if dyn != "False":
            raise RuntimeError(f"resource gate: {role} Dynamic Stack {dyn}")
        out[role] = vals
    bad: List[str] = []
    p = out["producer"]
    if p["vgpr"] > PRODUCER_VGPR_MAX:
        bad.append(f"producer VGPR {p['vgpr']} > {PRODUCER_VGPR_MAX}")
    if p["occupancy"] < PRODUCER_OCCUPANCY_MIN:
        bad.append(f"producer occupancy {p['occupancy']} < {PRODUCER_OCCUPANCY_MIN}")
    if p["lds"] != LDS_BYTES:
        bad.append(f"producer LDS {p['lds']} != {LDS_BYTES}")
    for role, v in out.items():
        if v["vgpr_spill"] or v["sgpr_spill"] or v["scratch"]:
            bad.append(f"{role} spill {v['vgpr_spill']}/{v['sgpr_spill']} scratch {v['scratch']} (need 0/0/0)")
    if bad:
        raise RuntimeError("resource gate: " + "; ".join(bad))
    return out


def build(source: Optional[Path] = None, fresh: bool = False) -> Tuple[Path, Dict[str, Any], bool]:
    """(library path, sidecar, cache hit). Compiles unless a finished, trusted cache entry (.so + sidecar) passes the
    gate again. fresh=True (a faulty entry): deletes the shared entry and builds into a private directory. Raises on
    any failure (no hipcc, compile error or crash, timeout, a missing or failing census)."""
    source = Path(source) if source is not None else source_path()
    src = source.read_bytes()
    hipcc = hipcc_path()
    fl = flags(src, hipcc)
    key = cache_key(source)
    d, shared = _entry_dir(key)
    so, meta = d / _LIB_NAME, d / _META_NAME
    if fresh:
        for p in (so, meta):
            try:
                p.unlink()
            except OSError:
                pass
        if shared:
            d, shared = _private_dir(), False
            so, meta = d / _LIB_NAME, d / _META_NAME
    elif so.is_file() and meta.is_file():
        try:
            side = json.loads(meta.read_text())
            side["summary"] = check_resources(side["kernels"])
            side["key"] = key
            return so, side, True
        except Exception:  # noqa: BLE001 -- a faulty entry: delete it and build once more, privately
            return build(source, fresh=True)
    tag = f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
    tmp, tmp_meta = d / f"{tag}-{_LIB_NAME}", d / f"{tag}-{_META_NAME}"
    work = Path(tempfile.mkdtemp(prefix=f"{tag}-build-", dir=str(d)))
    cmd = [hipcc, str(source.resolve()), *fl, "-o", str(tmp.resolve())]
    t0 = time.perf_counter()
    try:
        rc, log = _hipcc_run(cmd, work)
        secs = time.perf_counter() - t0
        if rc != 0 or not tmp.is_file():  # the diagnostic first: prebuild / r16 keep only 600 chars of this text
            what = f"hipcc rc={rc}" + ("" if rc != 0 else " wrote no library")
            raise RuntimeError(f"{what}: {_hipcc_diagnostic(log)} (full log: {_keep_build_log(cmd, log)})")
        kernels = parse_remarks(log)
        summary = check_resources(kernels)
        side: Dict[str, Any] = {
            "kernels": kernels,
            "summary": summary,
            "code_bytes": _code_sizes_in(work),
            "remark_format": remark_formats(log),
            "src_path": str(source),
            "src_sha256": hashlib.sha256(src).hexdigest(),
            "flags": fl,
            "key": key,
            "hipcc": hipcc,
            "toolchain": _toolchain(hipcc).strip().splitlines()[:2],
            "build_s": round(secs, 2),
            "host": socket.gethostname(),
        }
        tmp_meta.write_text(json.dumps(side, sort_keys=True))
        os.replace(tmp_meta, meta)  # the sidecar first: whoever sees the .so also finds its census
        os.replace(tmp, so)
    finally:
        for p in (tmp, tmp_meta):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(work, ignore_errors=True)
    return so, side, False


# ---------------------------------------------------------------------------- load
_VP, _I, _F, _LL, _CP = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_longlong, ctypes.c_char_p
# The C ABI (outproj_dual_mx6.cu, DESIGN 3.1); a CPU test checks it against the .cu text.
SIGNATURES: Dict[str, Tuple[Any, Tuple[Any, ...]]] = {
    # DOUT, INV, t_inv_mul, M, M_PAD, G6, DOUT_T, PARTIAL, INV_USED (nullable), stream
    "outproj_dual_mx6_launch": (_I, (_VP, _VP, _F, _I, _I, _VP, _VP, _VP, _VP, _VP)),
    # W, w_is_bf16, W6, W6S, stream
    "outproj_w6_shadow_launch": (_I, (_VP, _I, _VP, _VP, _VP)),
    "outproj_dual_mx6_abi_version": (_I, ()),
    "outproj_dual_mx6_drain_last_error": (_I, ()),
    "outproj_dual_mx6_partial_count": (_LL, (_I,)),
    "outproj_dual_mx6_lds_bytes": (_I, ()),
    "outproj_dual_mx6_variant": (_I, ()),
    "outproj_dual_mx6_impl": (_CP, ()),
}


class _Lib:
    """One loaded library (ctypes.CDLL, RTLD_LOCAL): every SIGNATURES entry bound; ABI, impl and geometry checked."""

    def __init__(self, so: Path, side: Dict[str, Any], cached: bool) -> None:
        self.path, self.side, self.cached = Path(so), side, cached
        self.cdll = ctypes.CDLL(str(so))
        self.fn: Dict[str, Any] = {}
        for name, (res, args) in SIGNATURES.items():
            f = getattr(self.cdll, name)
            f.restype = res
            f.argtypes = list(args)
            self.fn[name] = f
        abi = int(self.fn["outproj_dual_mx6_abi_version"]())
        if abi != ABI_VERSION:
            raise RuntimeError(f"{so}: outproj_dual_mx6_abi_version()={abi}, expected {ABI_VERSION}")
        self.impl = (self.fn["outproj_dual_mx6_impl"]() or b"").decode(errors="replace")
        if not self.impl.startswith(IMPL):
            raise RuntimeError(f"{so}: impl {self.impl[:200]!r} does not start with IMPL")
        tail = "; src=" + str(side.get("src_sha256"))
        if not self.impl.endswith(tail):
            raise RuntimeError(f"{so}: impl ends {self.impl[-90:]!r}, not {tail!r} (not built from the hashed source)")
        lds = int(self.fn["outproj_dual_mx6_lds_bytes"]())
        if lds != LDS_BYTES:
            raise RuntimeError(f"{so}: outproj_dual_mx6_lds_bytes()={lds}, expected {LDS_BYTES}")
        for m in (1, 255, 256, 257, 769, 4097, M_MAX):
            got = int(self.fn["outproj_dual_mx6_partial_count"](m))
            if got != partial_count(m):
                raise RuntimeError(f"{so}: outproj_dual_mx6_partial_count({m})={got}, expected {partial_count(m)}")
        self.variant = int(self.fn["outproj_dual_mx6_variant"]())


def _load() -> "_Lib":
    """Build (or find in the cache), gate and load the library once per process. A cached entry that fails to load or
    fails the ABI / impl / geometry checks is deleted and rebuilt once (privately)."""
    global _lib
    lib = _lib
    if lib is not None:
        return lib
    with _lock:
        if _lib is not None:
            return _lib
        so, side, hit = build()
        try:
            lib = _Lib(so, side, hit)
        except Exception:  # noqa: BLE001
            if not hit:
                raise
            so, side, hit = build(fresh=True)
            lib = _Lib(so, side, hit)
        _lib = lib
        return lib


def decode_variant(v: int) -> Dict[str, int]:
    """outproj_dual_mx6_variant() -> {G6_F32, DN1, REVERSE, WPE, PF}."""
    out = {name: (int(v) >> i) & 1 for i, name in enumerate(VARIANT_FLAGS)}
    out["PF"] = int(v) >> 4
    return out


def library_path() -> Optional[Path]:
    return _lib.path if _lib is not None else None


def impl() -> str:
    return _lib.impl if _lib is not None else ""


def variant() -> Optional[int]:
    return _lib.variant if _lib is not None else None


def resources() -> Dict[str, Any]:
    """The loaded library's gate summary {role: {...}} plus code sizes ({} before a load)."""
    if _lib is None:
        return {}
    out: Dict[str, Any] = {k: dict(v) for k, v in _lib.side.get("summary", {}).items()}
    sizes = _lib.side.get("code_bytes", {})
    for role, name in KERNELS.items():
        if role in out:
            out[role]["code_bytes"] = sizes.get(name)
    return out


def _require_device() -> int:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise RuntimeError("no ROCm GPU in this process (torch.cuda.is_available() is False or torch is not a HIP "
                           "build)")
    index = int(torch.cuda.current_device())
    arch = _l11().device_arch(index)
    if arch != _ARCH:
        raise RuntimeError(f"device cuda:{index} arch {arch or 'unknown'} is not {_ARCH} (the .cu is {_ARCH}-only)")
    return index


# ---------------------------------------------------------------------------- prebuild
def prebuild() -> None:
    """Before RUN_START (r16's MX6 prebuild chain, or lazily through S1-E). Idempotent and process-latched:
    GPU / gfx950 check, build or cache hit + resource gate + load (ABI, impl, geometry), then self_check(). Success:
    one armed line. Any failure latches (one DISABLED line) and raises OutprojMx6Error with the reason; a later call
    raises the latched reason again and prints nothing. A self-check exception that is a device fault
    (l11_outproj_lnmd._DEVICE_FAULT) re-raises unchanged after one line (abort before RUN_START)."""
    global _prebuilt, _failure
    if _prebuilt:
        return
    with _prebuild_lock:
        if _prebuilt:
            return
        if _failure is not None:
            raise OutprojMx6Error(f"{FAMILY} unavailable: {_failure}")
        t0 = time.perf_counter()
        try:
            index = _require_device()
            with torch.cuda.device(index):
                lib = _load()
        except Exception as error:  # noqa: BLE001 -- arch / build / gate / load: never a device fault, always latch
            raise _fail(f"prebuild build/load: {type(error).__name__}: {str(error)[:600]}") from error
        try:
            with torch.cuda.device(index):
                ok, detail = self_check()
                torch.cuda.synchronize()
        except Exception as error:  # noqa: BLE001
            msg = f"{type(error).__name__}: {error}"
            if _device_fault(msg):
                if not _fail_closed() and _failure is None:
                    _failure = _one_line(f"prebuild self-check: device-side error: {msg[:600]}")
                _say(f"{FAMILY} prebuild self-check hit a device-side error; re-raised to abort before RUN_START: "
                     f"{msg[:600]}")
                raise
            raise _fail(f"prebuild self-check: {msg[:600]}") from error
        if not ok:
            raise _fail(f"prebuild self-check mismatch: {detail}")
        secs = time.perf_counter() - t0
        res = resources()
        p = res.get("producer", {})
        w0, w1 = res.get("w6_f32", {}), res.get("w6_bf16", {})
        sha = str(lib.side.get("src_sha256", ""))
        pin = "" if sha == SOURCE_SHA256 else " (not SOURCE_SHA256: a dev source)"
        built = "cached" if lib.cached else f"built {lib.side.get('build_s')}s"
        defines = " ".join(dev_defines()) or "none"
        _armed.clear()
        _armed.update(resources=res, key=lib.side.get("key"), impl=lib.impl, variant=lib.variant,
                      variant_flags=decode_variant(lib.variant), src=lib.side.get("src_path"), src_sha256=sha,
                      cached=lib.cached, self_check=detail, seconds=round(secs, 2), path=str(lib.path))
        _prebuilt = True
        _say(
            f"{FAMILY} armed OK: prebuild before run_start: built+loaded+self-check OK {secs:.1f}s (VGPR {p.get('vgpr')}, "
            f"SGPR {p.get('sgpr')}, spill {p.get('vgpr_spill')}/{p.get('sgpr_spill')}, scratch {p.get('scratch')}, "
            f"LDS {p.get('lds')}, occupancy {p.get('occupancy')}, code {p.get('code_bytes')} B; W6 f32/bf16 VGPR "
            f"{w0.get('vgpr')}/{w1.get('vgpr')}, spill 0/0, scratch 0; so {lib.side.get('key')} {built}; "
            f"impl={lib.impl}; variant={lib.variant}; defines={defines}; src={lib.side.get('src_path')}@{sha[:12]}{pin}; "
            f"self-check {detail})"
        )


def set_prebuilt_for_testing(value: bool) -> None:
    """Tests only: mark the prebuild as done / not done without running it."""
    global _prebuilt
    _prebuilt = bool(value)


def reset_for_testing(unload: bool = False) -> None:
    """Tests only: clear the latch, the prebuilt flag, the once-lines, the audit and the W6 cache; unload=True also
    forgets the loaded library (the next prebuild() loads again)."""
    global _failure, _prebuilt, _lib
    with _prebuild_lock:
        _failure = None
        _prebuilt = False
        _armed.clear()
        _once_keys.clear()
        _AUDIT["pending"] = None
        clear_w6_cache()
        if unload:
            with _lock:
                _lib = None


# ---------------------------------------------------------------------------- launch (producer)
_HIP_ERRORS: Dict[int, str] = {
    1: "hipErrorInvalidValue", 2: "hipErrorOutOfMemory", 3: "hipErrorNotInitialized", 4: "hipErrorDeinitialized",
    9: "hipErrorInvalidConfiguration", 17: "hipErrorInvalidDevicePointer", 52: "hipErrorMissingConfiguration",
    53: "hipErrorPriorLaunchFailure", 98: "hipErrorInvalidDeviceFunction", 100: "hipErrorNoDevice",
    101: "hipErrorInvalidDevice", 200: "hipErrorInvalidImage", 201: "hipErrorInvalidContext",
    209: "hipErrorNoBinaryForGpu", 214: "hipErrorECCNotCorrectable", 218: "hipErrorInvalidKernelFile",
    303: "hipErrorSharedObjectInitFailed", 400: "hipErrorInvalidHandle", 401: "hipErrorIllegalState",
    500: "hipErrorNotFound", 600: "hipErrorNotReady", 700: "hipErrorIllegalAddress",
    701: "hipErrorLaunchOutOfResources", 702: "hipErrorLaunchTimeOut", 709: "hipErrorContextIsDestroyed",
    710: "hipErrorAssert", 719: "hipErrorLaunchFailure", 720: "hipErrorCooperativeLaunchTooLarge",
    801: "hipErrorNotSupported", 900: "hipErrorStreamCaptureUnsupported", 901: "hipErrorStreamCaptureInvalidated",
    999: "hipErrorUnknown", 1052: "hipErrorRuntimeMemory", 1053: "hipErrorRuntimeOther",
}


def hip_error_name(code: int) -> str:
    return _HIP_ERRORS.get(int(code), f"hipError {int(code)}")


def partial_count(m: int) -> int:
    """PARTIAL entries for M rows: 4 * ceil(M / 256) (= the dual cast's grid; outproj_dual_mx6_partial_count)."""
    return 4 * ((int(m) + BAND_ROWS - 1) // BAND_ROWS)


def _f32_bits(x: float) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def _a16(t: torch.Tensor) -> bool:
    return bool(_l11()._a16(t))


def _is_int(v: Any) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def ineligible_reason(dout: Any, inv_scale: Any, t_inv_mul: Any, M: Any, m_pad: Any, g6: Any, dout_t: Any,
                      partial: Any, inv_used: Any = None) -> Optional[str]:
    """None when launch() takes these operands, else why not. Host metadata only (no sync, no allocation). It
    mirrors outproj_dual_mx6_launch's own checks (non-null tensors, 1 <= M <= M_MAX, M_PAD == roundup(M, 128),
    t_inv_mul in 2^-4..2^-7, 16-B aligned DOUT / INV / G6 / DOUT_T / PARTIAL / INV_USED) and adds the layouts the
    kernel assumes: dout bf16 [M, 512] row-contiguous; inv_scale fp32 [1]; g6 uint8 [M, 400]; dout_t uint8 or e4m3
    [M_PAD/128, 512, 128]; partial fp32 [4 ceil(M/256)]; inv_used None or fp32 (not inv_scale); one CUDA device."""
    named = (("dout", dout), ("inv_scale", inv_scale), ("g6", g6), ("dout_t", dout_t), ("partial", partial))
    for name, t in named:
        if not isinstance(t, torch.Tensor):
            return f"{name} is {type(t).__name__}, not a tensor"
    if inv_used is not None and not isinstance(inv_used, torch.Tensor):
        return f"inv_used is {type(inv_used).__name__}, not a tensor or None"
    if not _is_int(M) or not _is_int(m_pad):
        return f"M={M!r} / m_pad={m_pad!r} are not ints"
    m, mp = int(M), int(m_pad)
    if not 1 <= m <= M_MAX:
        return f"M={m} outside 1..{M_MAX}"
    if mp != (m + STEP_TOKENS - 1) // STEP_TOKENS * STEP_TOKENS:
        return f"m_pad={mp} != roundup(M={m}, 128)"
    try:
        tf = float(t_inv_mul)
    except (TypeError, ValueError):
        return f"t_inv_mul={t_inv_mul!r} is not a float"
    tb = _f32_bits(tf)
    if tb not in T_INV_MUL_BITS or float(np.float32(tf)) != tf:
        return f"t_inv_mul={tf!r} (fp32 bits {tb:#010x}) is not 2^-4 .. 2^-7"
    dev = dout.device
    if dev.type != "cuda":
        return f"dout on {dev} (needs a CUDA device)"
    tensors = [t for _, t in named] + ([inv_used] if inv_used is not None else [])
    if any(t.device != dev for t in tensors):
        return "operands are not all on one CUDA device: " + ", ".join(str(t.device) for t in tensors)
    if not (dout.dtype == torch.bfloat16 and dout.dim() == 2 and tuple(dout.shape) == (m, D)
            and dout.stride() == (D, 1)):
        return f"dout {dout.dtype} {tuple(dout.shape)} stride {dout.stride()} (needs bf16 [{m}, {D}], row stride {D})"
    if not (inv_scale.dtype == torch.float32 and inv_scale.numel() == 1):
        return f"inv_scale {inv_scale.dtype} {tuple(inv_scale.shape)} (needs fp32 [1], a device tensor)"
    if not (g6.dtype == torch.uint8 and tuple(g6.shape) == (m, G6_ROW) and g6.is_contiguous()):
        return f"g6 {g6.dtype} {tuple(g6.shape)} (needs contiguous uint8 [{m}, {G6_ROW}])"
    want_t = (mp // STEP_TOKENS, D, STEP_TOKENS)
    if not (dout_t.dtype in (torch.uint8, getattr(torch, "float8_e4m3fn", torch.uint8))
            and tuple(dout_t.shape) == want_t and dout_t.is_contiguous()):
        return f"dout_t {dout_t.dtype} {tuple(dout_t.shape)} (needs contiguous e4m3 / uint8 {list(want_t)})"
    if not (partial.dtype == torch.float32 and partial.numel() == partial_count(m) and partial.is_contiguous()):
        return f"partial {partial.dtype} {tuple(partial.shape)} (needs contiguous fp32 [{partial_count(m)}])"
    if inv_used is not None:
        if not (inv_used.dtype == torch.float32 and inv_used.numel() >= 1):
            return f"inv_used {inv_used.dtype} {tuple(inv_used.shape)} (needs fp32 [1] or None)"
        if inv_used.data_ptr() == inv_scale.data_ptr():
            return "inv_used aliases inv_scale (it must be its own allocation)"
    for name, t in named + ((("inv_used", inv_used),) if inv_used is not None else ()):
        if not _a16(t):
            return f"{name} is not 16-B aligned (data_ptr % 16 = {t.data_ptr() % 16})"
    return None


def _lib_for_launch() -> "_Lib":
    if _failure is not None:
        raise OutprojMx6Error(f"{FAMILY} unavailable: {_failure}")
    lib = _lib
    if lib is None:
        raise OutprojMx6Error(f"{FAMILY} unavailable: the library is not loaded (prebuild() has not run in this "
                              "process)")
    return lib


def _check_rc(rc: int, what: str, m: Optional[int] = None) -> None:
    """D15, convention (b): 0 launched; > 0 the hipError of THIS launch (nothing launched, outputs not written) ->
    OutprojMx6Error naming it; < 0 launched, -rc is a stale error an earlier call left on this thread (drained by the
    launcher): one line per process and code, then return. Inside prebuild()'s self-check only an rc whose text matches
    l11_outproj_lnmd._DEVICE_FAULT re-raises (abort before RUN_START): today rc 700 / 719, by their names. rc 53, 214,
    702 and 710 do not match and latch (v1.2 review: a v1.3 / r16 follow-up)."""
    if rc == 0:
        return
    if rc > 0:
        where = "" if m is None else f" M={m}"
        raise OutprojMx6Error(f"{FAMILY} {what} rc={rc} ({hip_error_name(rc)}){where}: not launched, outputs not "
                              "written")
    _say_once(("stale", -rc), f"{FAMILY} stale HIP error {-rc} ({hip_error_name(-rc)}) drained at {what} (left on this "
              "thread by an earlier call, not ours; launched, result used)")


def launch(dout: torch.Tensor, inv_scale: torch.Tensor, t_inv_mul: float, M: int, m_pad: int, g6: torch.Tensor,
           dout_t: torch.Tensor, partial: torch.Tensor, inv_used: Optional[torch.Tensor]) -> None:
    """ONE producer launch (G6, dout_t, PARTIAL, INV_USED) on torch.cuda.current_stream(dev) inside
    torch.cuda.device(dev), dev = dout.device. No host sync. The library must be loaded and not latched (S1 asks
    not_armed_reason() first). Operands failing ineligible_reason(): ValueError (nothing launched). rc > 0:
    OutprojMx6Error naming the hipError; rc < 0: one stale-error line per process and code, then return."""
    lib = _lib_for_launch()
    why = ineligible_reason(dout, inv_scale, t_inv_mul, M, m_pad, g6, dout_t, partial, inv_used)
    if why is not None:
        raise ValueError(f"{FAMILY} launch: {why}")
    dev = dout.device
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        stream = torch.cuda.current_stream(index).cuda_stream
        rc = int(lib.fn["outproj_dual_mx6_launch"](
            dout.data_ptr(), inv_scale.data_ptr(), float(t_inv_mul), int(M), int(m_pad), g6.data_ptr(),
            dout_t.data_ptr(), partial.data_ptr(), None if inv_used is None else inv_used.data_ptr(), stream))
        _check_rc(rc, "outproj_dual_mx6_launch", int(M))
        if _prebuilt and int(M) >= _AUDIT_MIN_M and _audit_pending():
            _AUDIT["pending"] = False
            ok, detail = audit_last_bands(dout, inv_scale, t_inv_mul, M, m_pad, g6, dout_t, partial)
            _say(f"{FAMILY} dev audit ({AUDIT_ENV}=1, the first launch with M >= {_AUDIT_MIN_M}): last bands vs the "
                 f"in-tree G6 encoder and a direct Triton dual cast: {detail}")
            if not ok:
                raise _fail(f"dev audit ({AUDIT_ENV}=1) mismatch: {detail}")


# ---------------------------------------------------------------------------- W1: the W6 / W6S shadow (DESIGN 10)
class _W6Entry:
    __slots__ = ("ref", "version", "data_ptr", "w6", "w6s")

    def __init__(self, ref: Any, version: int, data_ptr: int, w6: torch.Tensor, w6s: torch.Tensor) -> None:
        self.ref, self.version, self.data_ptr, self.w6, self.w6s = ref, version, data_ptr, w6, w6s


_w6_lock = threading.Lock()
_w6_cache: Dict[Tuple[Any, ...], _W6Entry] = {}
_w6_stats: Dict[str, int] = {"hits": 0, "refreshes": 0, "pruned": 0}


def w6_ineligible_reason(weight: Any) -> Optional[str]:
    """None for a weight W1 takes: a CUDA fp32 or bf16 [1536, 512] tensor (any strides / alignment: the launch
    copies those); else why not (a persistent condition for S1)."""
    if not isinstance(weight, torch.Tensor):
        return f"weight is {type(weight).__name__}, not a tensor"
    if weight.dtype not in (torch.float32, torch.bfloat16):
        return f"weight dtype {weight.dtype} (needs float32 or bfloat16)"
    if tuple(weight.shape) != (NOUT, D):
        return f"weight shape {tuple(weight.shape)} (needs ({NOUT}, {D}))"
    if weight.device.type != "cuda":
        return f"weight on {weight.device} (needs a CUDA device)"
    return None


def _w6_launch(weight: torch.Tensor, out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ONE outproj_w6_shadow_launch of `weight` on the current stream of its device into NEW uint8 [589,824] / [768]
    tensors (or `out`: the self-check's poisoned buffers). A non-contiguous or not 16-B aligned weight is copied first
    (stream-ordered). Raises ValueError / OutprojMx6Error (rc > 0)."""
    lib = _lib_for_launch()
    why = w6_ineligible_reason(weight)
    if why is not None:
        raise ValueError(f"{FAMILY} W6: {why}")
    w = weight.detach()
    if not w.is_contiguous() or not _a16(w):
        w = w.clone(memory_format=torch.contiguous_format)
    dev = w.device
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        if out is None:
            w6 = torch.empty((W6_BYTES,), dtype=torch.uint8, device=dev)
            w6s = torch.empty((W6S_BYTES,), dtype=torch.uint8, device=dev)
        else:
            w6, w6s = out
            for name, t, n in (("w6", w6, W6_BYTES), ("w6s", w6s, W6S_BYTES)):
                if not (t.dtype == torch.uint8 and t.numel() == n and t.is_contiguous() and _a16(t)
                        and t.device == dev):
                    raise ValueError(f"{FAMILY} W6: {name} out {t.dtype} {tuple(t.shape)} (needs contiguous 16-B "
                                     f"aligned uint8 [{n}] on {dev})")
        stream = torch.cuda.current_stream(index).cuda_stream
        rc = int(lib.fn["outproj_w6_shadow_launch"](w.data_ptr(), int(w.dtype == torch.bfloat16), w6.data_ptr(),
                                                    w6s.data_ptr(), stream))
    _check_rc(rc, "outproj_w6_shadow_launch")
    return w6, w6s


def w6_encode(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """W6 / W6S of `weight` (fp32 or bf16 [1536, 512]), uncached: ONE launch into new tensors (S1-E, the self-check;
    no cache entry is created or read)."""
    return _w6_launch(weight)


def w6_shadow(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(w6, w6s) of `weight`, cached until `weight` changes (k512_mx6_gemm.mx6_weight's protocol): key = (id, device
    index, shape, stride, dtype, "L11"); a hit needs the weakref alive and the same _version and data_ptr. A miss is
    ONE launch on the current stream into NEW tensors (never into a tensor a queued kernel may still read); entries
    of freed weights are dropped on each refresh. Raises (persistent) for a weight W1 does not take, or when the
    library is not armed."""
    why = not_armed_reason()
    if why is not None:
        raise OutprojMx6Error(f"{FAMILY} W6 shadow unavailable: {why}")
    why = w6_ineligible_reason(weight)
    if why is not None:
        raise ValueError(f"{FAMILY} W6 shadow: {why}")
    index = weight.device.index
    if index is None:
        index = torch.cuda.current_device()
    key = (id(weight), int(index), tuple(weight.shape), tuple(weight.stride()), weight.dtype, "L11")
    version = int(getattr(weight, "_version", 0))
    with _w6_lock:
        entry = _w6_cache.get(key)
        if (entry is not None and entry.ref() is weight and entry.version == version
                and entry.data_ptr == weight.data_ptr()):
            _w6_stats["hits"] += 1
            return entry.w6, entry.w6s
    w6, w6s = _w6_launch(weight)
    with _w6_lock:
        dead = [k for k, e in _w6_cache.items() if e.ref() is None]
        for k in dead:
            del _w6_cache[k]
        _w6_stats["pruned"] += len(dead)
        _w6_cache[key] = _W6Entry(weakref.ref(weight), version, weight.data_ptr(), w6, w6s)
        _w6_stats["refreshes"] += 1
    return w6, w6s


def clear_w6_cache() -> None:
    with _w6_lock:
        _w6_cache.clear()
        for k in _w6_stats:
            _w6_stats[k] = 0


def w6_cache_stats() -> Dict[str, int]:
    with _w6_lock:
        return {"entries": len(_w6_cache), **_w6_stats}


# ---------------------------------------------------------------------------- in-tree references (CPU, numpy)
# Ports of r14 iter66 U1 (spec/g6_spec_helper.py: the REF-equal integer G6 / W6 rules, the adversarial rows) and U3
# (spec/w6_fork_cpu_model.py: the edge weight). No probe-directory import: a CPU test (DESIGN T12) pins these ports
# against r16's REF (mx6_l11_ref.py c49f2659) and the U1 vectors (g6_vectors_v1.npz: ADV_ROWS_SHA256, ADV_G6_SHA256).
ADV_ROWS_SHA256: str = "de3b7ffd059e7a4f318a40c716b5742be97d11c5ec0c61239c7afe63241b2ae8"  # npz dout_bits (uint16)
ADV_G6_SHA256: str = "da10b923eb1dc95bf611eefdafd5edcd288a58451b50c10d81a4eaa775f347f3"  # npz g6_ref (REF bytes)
_BLK, _NBLK, _WNB = 32, 16, 48
_HEAD_OFF, _TAIL_OFF, _SC_OFF = 0, 256, 384
_BYTE_NONFINITE = 255
_BL9 = np.array([int(v).bit_length() for v in range(1 << 9)], dtype=np.int64)


def e8m0_byte(a16: Any) -> np.ndarray:
    """G6 E8M0 byte of the integer block max a16 = max(h & 0x7FFF) (NaN-dominant): 255 if a16 >= 0x7F80, 127 if
    a16 == 0, else max(0, (a16 >> 7) - 2 + ((a16 & 0x7F) > 0x70)) (U1 e8m0_byte_np; == REF e8m0_exp_exact + 127)."""
    a = np.asarray(a16).astype(np.int64)
    b = np.maximum((a >> 7) - 2 + ((a & 0x7F) > 0x70).astype(np.int64), 0)
    b = np.where(a == 0, 127, b)
    return np.where(a >= 0x7F80, _BYTE_NONFINITE, b)


def scale_operand_bits(byte: Any) -> np.ndarray:
    """fp32 bits of the pk32 convert's scale operand for E8M0 `byte` (U1): max(byte << 23, 0x00400000); 1.0 for 255."""
    b = np.asarray(byte).astype(np.int64)
    out = np.maximum(b << 23, 0x00400000)
    return np.where(b == _BYTE_NONFINITE, 0x3F800000, out).astype(np.uint32)


def _bitlen(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.int64)
    if s.size and int(s.max()) < (1 << 9):
        return _BL9[s]
    out = np.zeros(s.shape, dtype=np.int64)
    v = s.copy()
    while np.any(v > 0):
        out += (v > 0)
        v >>= 1
    return out


def _rne_shift(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """RNE(s * 2^u) for integers s >= 0 (s < 2^25), u any integer (ties to even)."""
    s = s.astype(np.int64)
    u = u.astype(np.int64)
    out_pos = np.left_shift(s, np.clip(u, 0, 30))
    sh = np.clip(-u, 1, 62)
    q = np.right_shift(s, sh)
    r = s - np.left_shift(q, sh)
    half = np.left_shift(np.int64(1), sh - 1)
    rnd = (r > half) | ((r == half) & ((q & 1) == 1))
    return np.where(u >= 0, out_pos, q + rnd.astype(np.int64))


def _e2m3_code_from_sig(s: Any, sig: Any, x: Any, e: Any) -> np.ndarray:
    """E2M3 code of (-1)^s * sig * 2^x / 2^e (|y| <= 7.5): the nearest grid index (ties to even), sign bit = s
    (signbit: -0 and negatives rounding to 0 keep 0x20, REF neg_zero=True)."""
    sig = np.asarray(sig, dtype=np.int64)
    t = np.asarray(x, dtype=np.int64) - np.asarray(e, dtype=np.int64) + 3
    lg = np.where(sig > 0, _bitlen(sig) - 1, 0)
    p = lg + t
    if np.any((sig > 0) & (p > 5)):
        raise ValueError("|x / 2^e| > 7.5: not a finite-block element at this E8M0 byte")
    j = np.clip(p - 3, 0, 2)
    c = np.array([0, 8, 16], dtype=np.int64)[j] + _rne_shift(sig, t - j)
    c = np.where(sig == 0, 0, c)
    return c | (np.asarray(s, dtype=np.int64) << 5)


def e2m3_code(h: Any, byte: Any) -> np.ndarray:
    """6-bit G6 code of bf16 pattern h (finite) in a finite block with E8M0 byte `byte` (U1 e2m3_code_np)."""
    h = np.asarray(h).astype(np.int64) & 0xFFFF
    byte = np.asarray(byte).astype(np.int64)
    ex = (h >> 7) & 0xFF
    if np.any(ex == 0xFF):
        raise ValueError("non-finite element: its block's byte is 255 and its codes are 0")
    sig = np.where(ex == 0, h & 0x7F, 128 + (h & 0x7F))
    x = np.where(ex == 0, -133, ex - 134)
    return _e2m3_code_from_sig(h >> 15, sig, x, byte - 127)


def pack24(codes: Any) -> np.ndarray:
    """int [..., 32] codes -> uint8 [..., 24]: code f at bits [6f, 6f + 6) of the little-endian 192-bit group."""
    c = np.asarray(codes).astype(np.int64)
    sh = c.shape[:-1]
    c = c.reshape(*sh, 8, 4)
    v = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    b = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], -1)
    return b.reshape(*sh, 24).astype(np.uint8)


def _assemble_g6(codes: np.ndarray, byte: np.ndarray) -> np.ndarray:
    """codes [R, 16, 32], byte [R, 16] (block b = 4k + q) -> uint8 [R, 400] records."""
    r = codes.shape[0]
    grp = pack24(codes)
    rec = np.empty((r, G6_ROW), dtype=np.uint8)
    rec[:, _HEAD_OFF:_TAIL_OFF] = grp[:, :, 0:16].reshape(r, 256)
    rec[:, _TAIL_OFF:_SC_OFF] = grp[:, :, 16:24].reshape(r, 128)
    rec[:, _SC_OFF:G6_ROW] = byte.reshape(r, 4, 4).transpose(0, 2, 1).reshape(r, 16).astype(np.uint8)
    return rec


def encode_g6_int(bits: Any, chunk: int = 65536) -> np.ndarray:
    """The in-tree integer G6 encoder (U1 encode_g6_int): bf16 bit patterns [M, 512] -> uint8 [M, 400], byte-equal to
    REF encode_g6(neg_zero=True), strict (byte-255 blocks: 24 zero code bytes)."""
    h_all = np.asarray(bits)
    if h_all.ndim != 2 or h_all.shape[1] != D:
        raise ValueError("bits must be [M, 512]")
    m = h_all.shape[0]
    out = np.empty((m, G6_ROW), dtype=np.uint8)
    for r0 in range(0, m, chunk):
        h = h_all[r0:r0 + chunk].astype(np.int64) & 0xFFFF
        r = h.shape[0]
        hb = h.reshape(r, _NBLK, _BLK)
        byte = e8m0_byte((hb & 0x7FFF).max(-1))
        fin = byte != _BYTE_NONFINITE
        codes = e2m3_code(np.where(fin[..., None], hb, 0), np.where(fin, byte, 127)[..., None])
        codes = np.where(fin[..., None], codes, 0)
        out[r0:r0 + r] = _assemble_g6(codes, byte)
    return out


def e8m0_byte_f32(a32: Any) -> np.ndarray:
    """W side: E8M0 byte of the NaN-dominant tile max of fp32 |bits| (255 non-finite, 127 zero, 0 subnormal max)."""
    a = np.asarray(a32).astype(np.int64)
    b = np.maximum((a >> 23) - 2 + ((a & 0x7FFFFF) > 0x700000).astype(np.int64), 0)
    b = np.where(a == 0, 127, b)
    return np.where(a >= 0x7F800000, _BYTE_NONFINITE, b)


def quant_w_int(w: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Integer form of REF quant_w (neg_zero=True), U1: w fp32 [1536, 512] -> (codes int64 [1536, 512], e int64
    [48, 16]; e = 128 marks a non-finite tile: W6S byte 255, codes 0)."""
    u = np.ascontiguousarray(np.asarray(w, dtype=np.float32)).view(np.uint32).astype(np.int64)
    t = u.reshape(_WNB, _BLK, _NBLK, _BLK)
    byte = e8m0_byte_f32((t & 0x7FFFFFFF).max(axis=(1, 3)))
    fin = byte != _BYTE_NONFINITE
    f4 = fin[:, None, :, None]
    ex = (t >> 23) & 0xFF
    mant = t & 0x7FFFFF
    sig = np.where(f4, np.where(ex == 0, mant, (1 << 23) | mant), 0)
    x = np.where(f4, np.where(ex == 0, -149, ex - 150), 0)
    ee = np.where(fin, byte, 127)[:, None, :, None] - 127
    codes = _e2m3_code_from_sig(np.where(f4, t >> 31, 0), sig, x, ee)
    codes = np.where(f4, codes, 0)
    e = np.where(fin, byte - 127, 128)
    return codes.reshape(NOUT, D), e


def w6_scatter(codes: Any, e: Any) -> Tuple[np.ndarray, np.ndarray]:
    """W6 / W6S from (codes [1536, 512], e [48, 16]) with the explicit L11 inverse map (U1 w6_scatter_np):
    t = n >> 8, w = (n >> 5) & 7, j = (n >> 2) & 1, R = 4((n >> 3) & 3) + (n & 3), k = kb >> 2, q = kb & 3; head at
    (4t + k) 24576 + 3072w + 1024j + 64R + 16(q ^ ((n >> 3) & 3)), tail at (4t + k) 24576 + 3072w + 2048 + 512j + 32R
    + 8(q ^ 2((n >> 4) & 1)); W6S[16 nb + 4q + k] = 127 + e(nb, 4k + q) (255 for a non-finite tile)."""
    c = np.asarray(codes).astype(np.int64).reshape(NOUT, _NBLK, _BLK)
    grp = pack24(c)
    w6 = np.full(W6_BYTES, 0xEE, dtype=np.uint8)
    n = np.arange(NOUT)[:, None]
    kb = np.arange(_NBLK)[None, :]
    t, w, j = n >> 8, (n >> 5) & 7, (n >> 2) & 1
    r = 4 * ((n >> 3) & 3) + (n & 3)
    k, q = kb >> 2, kb & 3
    base = (4 * t + k) * 24576 + 3072 * w
    ho = base + 1024 * j + 64 * r + 16 * (q ^ ((n >> 3) & 3))
    to = base + 2048 + 512 * j + 32 * r + 8 * (q ^ (2 * ((n >> 4) & 1)))
    for i in range(16):
        w6[(ho + i).reshape(-1)] = grp[:, :, i].reshape(-1)
    for i in range(8):
        w6[(to + i).reshape(-1)] = grp[:, :, 16 + i].reshape(-1)
    eb = np.minimum(np.asarray(e).astype(np.int64) + 127, 255)
    w6s = np.empty(W6S_BYTES, dtype=np.uint8)
    for nb in range(_WNB):
        for kb_ in range(_NBLK):
            w6s[16 * nb + 4 * (kb_ & 3) + (kb_ >> 2)] = eb[nb, kb_]
    return w6, w6s


# U1's adversarial rows (spec/g6_spec_helper.py adversarial_blocks / position_row / adversarial_rows, ported verbatim)
_GRID: List[float] = [c / 8.0 if c < 16 else ((c - 8) / 4.0 if c < 24 else (c - 16) / 2.0) for c in range(32)]
_MIDS: List[float] = [(_GRID[i] + _GRID[i + 1]) / 2.0 for i in range(31)]
_NAN_P, _NAN_N, _INF_P, _INF_N = 0x7FC0, 0xFFC0, 0x7F80, 0xFF80


def _bf16_bits_of(x: Any) -> int:
    """bf16 bit pattern of a float that is EXACT in bf16 (asserted); NaN -> 0x7FC0 (sign kept)."""
    if isinstance(x, (int, np.integer)):
        return int(x) & 0xFFFF
    if x != x:
        return 0x7FC0 | (0x8000 if np.signbit(x) else 0)
    f = np.float32(x)
    if not np.isinf(x) and float(f) != x:
        raise ValueError(f"{x!r} is not exact in fp32")
    u = int(np.array(f).view(np.uint32))
    if u & 0xFFFF:
        raise ValueError(f"{x!r} is not exact in bf16")
    return u >> 16


def _fill(rng: Any, scale: float, n: int = 32) -> List[int]:
    """n bf16 patterns ~ N(0, 1) * scale (a private generator), RNE-rounded to bf16."""
    v = (rng.standard_normal(n) * scale).astype(np.float32)
    u = v.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    return [int(x) for x in r]


def adversarial_blocks(seed: int = 20261013) -> List[Tuple[str, List[int]]]:
    """U1's deterministic named 32-element blocks (bf16 bit patterns) for the SPEC_G6 sec. 6 classes."""
    rng = np.random.default_rng(seed)
    out: List[Tuple[str, List[int]]] = []

    def add(name: str, blk: List[Any]) -> None:
        out.append((name, [int(v) & 0xFFFF for v in blk]))

    _b = _bf16_bits_of
    add("zero_pos", [0x0000] * 32)
    add("zero_neg", [0x8000] * 32)
    add("zero_mixed", [0x8000 if f & 1 else 0x0000 for f in range(32)])
    add("one_tiny_rest_negzero", [0x0001] + [0x8000] * 31)
    blk = _fill(rng, 1e-3)
    blk[0], blk[31], blk[13] = 0x8000, 0x8000, 0x0000
    add("negzero_in_normal_block", blk)
    add("nan_all", [_NAN_P] * 32)
    add("inf_all", [_INF_P] * 32)
    add("neginf_all", [_INF_N] * 32)
    for f in (0, 5, 10, 21, 26, 31):
        blk = _fill(rng, 1e-2)
        blk[f] = _NAN_P
        add(f"nan_one_f{f}", blk)
    for name, pat, f in (("negnan_one", _NAN_N, 7), ("snan_like_7f81", 0x7F81, 3), ("nan_7fff", 0x7FFF, 30),
                         ("negnan_ffff", 0xFFFF, 16), ("inf_one", _INF_P, 9), ("neginf_one", _INF_N, 22)):
        blk = _fill(rng, 1e-2)
        blk[f] = pat
        add(name, blk)
    blk = _fill(rng, 1e-2)
    blk[2], blk[29] = _NAN_P, _INF_N
    add("nan_and_inf", blk)
    blk = [0x8000] * 32
    blk[17] = _NAN_P
    add("nan_with_negzeros", blk)
    blk = [0x7F7F] * 32
    blk[0] = _INF_P
    add("inf_with_bf16max", blk)
    for k in range(-128, 126):
        pos = (k + 128) % 32
        blk = []
        for i in range(32):
            if i == pos:
                v = 7.5 * 2.0 ** k * (-1.0 if (k & 1) else 1.0)
            else:
                c = (i + k) % 31
                v = _GRID[c] * 2.0 ** k * (-1.0 if ((i + k) & 2) else 1.0)
            blk.append(_b(v))
        add(f"ladder_amax_7.5x2^{k}", blk)
    for m in range(0x70, 0x80):
        a = (254 << 7) | m
        blk = [(a | (0x8000 if (f >> 2) & 1 else 0)) if f % 4 == 0 else _b(1.0 * (-1) ** f) for f in range(32)]
        add(f"topbinade_m{m:#04x}", blk)
    for ex in (1, 2, 3, 60, 126, 127, 128, 200, 253, 254):
        for m in (0x00, 0x01, 0x6F, 0x70, 0x71, 0x7F):
            a = (ex << 7) | m
            blk = _fill(rng, 1.0)
            blk = [(x & 0x8000) | min(x & 0x7FFF, a) for x in blk]
            blk[(ex + m) % 32] = a | (0x8000 if m & 1 else 0)
            add(f"thr_E{ex}_m{m:#04x}", blk)
    for a in (0x0001, 0x0002, 0x0004, 0x0005, 0x0008, 0x0040, 0x0041, 0x007F, 0x0080, 0x00F0, 0x00F1, 0x0100,
              0x0170, 0x0171, 0x01F0, 0x01F1):
        blk = [a if f == 3 else ((f * 37) % (a + 1)) | (0x8000 if f & 1 else 0) for f in range(32)]
        add(f"small_amax_{a:#06x}", blk)
    for k in (-127, -126, -123, -100, -20, -3, 0, 5, 60, 120, 125):
        top = _b(7.5 * 2.0 ** k)
        mids = [_b(v * 2.0 ** k) for v in _MIDS]
        add(f"ties_e{k}", [top] + [mm | (0x8000 if i & 1 else 0) for i, mm in enumerate(mids)])
        add(f"ties_up1ulp_e{k}", [top] + [(mm + 1) | (0x8000 if i & 1 else 0) for i, mm in enumerate(mids)])
        add(f"ties_dn1ulp_e{k}", [top] + [(mm - 1) | (0x8000 if i & 1 else 0) for i, mm in enumerate(mids)])
        grid = [_b(v * 2.0 ** k) for v in _GRID]
        add(f"grid_pos_e{k}", grid)
        add(f"grid_neg_e{k}", [g | 0x8000 for g in grid])
    for lo in (1, 33, 65, 97):
        add(f"subn_all_m{lo}", [((lo + f - 1) % 127 + 1) | (0x8000 if f & 2 else 0) for f in range(32)])
    for k in (-127, -126, -125, -124, -123, -122, -121, -100, 0, 126):
        top = _b(7.5 * 2.0 ** k) if k < 126 else 0x7F7F
        sub = [1, 2, 3, 4, 5, 8, 12, 31, 32, 33, 63, 64, 65, 96, 127]
        blk = ([top] + [s | (0x8000 if i & 1 else 0) for i, s in enumerate(sub)]
               + [s | (0x8000 if not (i & 1) else 0) for i, s in enumerate(sub)] + [0x8001])
        add(f"subn_in_e{k}", blk)
    for top_bits, nm in ((0x7F7F, "max"), (_b(2.0 ** 100), "2^100"), (_b(7.5 * 2.0 ** -100), "7.5x2^-100")):
        smalls = [0xBF80, _b(-(2.0 ** -100)), 0x8080, 0x8001, 0x8000, 0x0001, 0x3F80, _b(2.0 ** -100),
                  _b(-(2.0 ** -126)), 0x80FF, 0x0000]
        blk = [top_bits] + [smalls[i % len(smalls)] for i in range(31)]
        if nm == "7.5x2^-100":
            blk = [top_bits] + [0x8001, 0x8000, 0x0001, 0x0000, _b(-(2.0 ** -126)), 0x80FF] * 5 + [0x8001]
        add(f"underflow_sign_{nm}", blk)
    blk = [_b(-1.0 * 2.0 ** -6 * 0.0625)] * 32
    blk[0] = _b(7.5 * 2.0 ** -6)
    add("neg_tie_to_zero", blk)
    for sh in (6, 12, 20):
        base = _fill(rng, 2.0 ** -sh)
        base[(sh * 3) % 32] = _b(1.5)
        add(f"outlier_2^{sh}", base)
    add("maxrange", [0x7F7F] + [0x0001 | (0x8000 if f & 1 else 0) for f in range(31)])
    for sc in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e3):
        add(f"normal_{sc:g}", _fill(rng, sc))
    return out


def position_row() -> List[int]:
    """One row whose 16 blocks have distinct E8M0 bytes 119 + b and position-dependent codes (U1 position_row)."""
    row: List[int] = []
    for b in range(_NBLK):
        for f in range(_BLK):
            v = 7.5 if f == 31 else _GRID[(f + b) % 31] * (-1.0 if f & 2 else 1.0)
            row.append(_bf16_bits_of(v * 2.0 ** (b - 8)))
    return row


_ADV_CACHE: Dict[int, np.ndarray] = {}


def adversarial_rows(seed: int = 20261013) -> np.ndarray:
    """uint16 [30, 512]: U1's position row, then every adversarial block in order (16 per row; the last row padded
    with production-like filler blocks). Bits == g6_vectors_v1.npz dout_bits (ADV_ROWS_SHA256) under numpy's PCG64."""
    got = _ADV_CACHE.get(seed)
    if got is not None:
        return got.copy()
    rng = np.random.default_rng(seed + 1)
    blocks = adversarial_blocks(seed)
    while len(blocks) % _NBLK:
        blocks.append((f"pad_normal_{len(blocks)}", _fill(rng, 1e-3)))
    rows = [position_row()] + [sum((blk for _, blk in blocks[i:i + _NBLK]), []) for i in range(0, len(blocks), _NBLK)]
    arr = np.array(rows, dtype=np.uint16)
    _ADV_CACHE[seed] = arr
    return arr.copy()


def edge_weight(seed: int = 20261013) -> np.ndarray:
    """U3's edge-case weight (spec/w6_fork_cpu_model.py edge_weight, ported): N(0, 0.02) fp32 [1536, 512] (a private
    generator) with an all-zero, an all -0, a round-to-zero-negatives tile, amax 7.5 * 2^{-3, 0, 5} and one ulp above,
    an E2M3-midpoint ties tile, a subnormal-amax tile, a near-fp32-max, an Inf, a NaN + -Inf and an all-NaN tile."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((NOUT, D)) * 0.02).astype(np.float32)
    blocks = [(nb, kb) for nb in range(48) for kb in range(16)]
    rng.shuffle(blocks)
    it = iter(blocks)

    def blk(nb: int, kb: int) -> Tuple[slice, slice]:
        return (slice(32 * nb, 32 * nb + 32), slice(32 * kb, 32 * kb + 32))

    with np.errstate(all="ignore"):
        nb, kb = next(it)
        w[blk(nb, kb)] = 0.0
        nb, kb = next(it)
        w[blk(nb, kb)] = -0.0
        nb, kb = next(it)
        x = w[blk(nb, kb)]
        x[:] = -1e-6
        x[0, 0] = 1.0
        for k_ in (-3, 0, 5):
            nb, kb = next(it)
            x = w[blk(nb, kb)]
            x[3, 4] = np.float32(7.5 * 2.0 ** k_)
            nb, kb = next(it)
            x = w[blk(nb, kb)]
            x[3, 4] = np.nextafter(np.float32(7.5 * 2.0 ** k_), np.float32(np.inf))
        nb, kb = next(it)
        x = w[blk(nb, kb)]
        x[:] = np.resize(np.array(_MIDS + [-m for m in _MIDS], dtype=np.float32), (32, 32))
        x[0, 0] = 7.5
        nb, kb = next(it)
        x = w[blk(nb, kb)]
        x[:] = np.float32(3e-39) * np.sign(rng.standard_normal((32, 32))).astype(np.float32)
        x[1, 1] = np.float32(1.1e-38)
        nb, kb = next(it)
        x = w[blk(nb, kb)]
        x[5, 6] = np.float32(3.3e38)
        nb, kb = next(it)
        x = w[blk(nb, kb)]
        x[7, 8] = np.inf
        nb, kb = next(it)
        x = w[blk(nb, kb)]
        x[9, 10] = -np.inf
        x[9, 11] = np.nan
        nb, kb = next(it)
        x = w[blk(nb, kb)]
        x[:] = np.nan
    return w


# Deterministic hash inputs (no RNG): ec_dgrad_x2h's 32-bit integer hash (_hashed / _mix32), on the CPU.
_SC_SALT: int = 0x6F646D36  # "odm6"


def _hash_np(n: int, salt: int) -> np.ndarray:
    from generative_recommenders.ops.hip import ec_dgrad_x2h as x2h

    return x2h._hashed(int(n), int(salt), torch.device("cpu")).numpy().astype(np.int64)


def production_like_rows(m: int, salt: int = _SC_SALT) -> np.ndarray:
    """uint16 [m, 512] bf16 patterns from integer hashes: per row a top binade 2^-23 .. 2^0, elements 0..12 binades
    below it with geometric frequency, random 7-bit mantissas and signs, ~0.8 % exact +-0 elements; rows whose hash
    gives kind 0 / 1 of 200 are all +0 / all -0 (0.5 % each)."""
    hr = _hash_np(m, salt)
    he = _hash_np(m * D, salt + 1).reshape(m, D)
    hm = _hash_np(m * D, salt + 2).reshape(m, D)
    top = 104 + (hr % 24)
    v = he | (1 << 12)
    down = np.log2(v & -v).astype(np.int64)
    ex = top[:, None] - down
    sign = (hm >> 31) & 1
    bits = (sign << 15) | (ex << 7) | (hm & 0x7F)
    bits = np.where(((hm >> 8) & 0xFF) < 2, sign << 15, bits)
    kind = (hr >> 8) % 200
    bits[kind == 0] = 0x0000
    bits[kind == 1] = 0x8000
    return bits.astype(np.uint16)


def self_check_rows(m: int, salt: int = _SC_SALT) -> np.ndarray:
    """SC1 / SC3 / SC6 input: production_like_rows with adversarial_rows() written over rows [0, 30), the 4 rows
    straddling 128, 256 and 384 (the step, band and band-1 step boundaries) and the last 30 rows."""
    bits = production_like_rows(m, salt)
    adv = adversarial_rows()
    nr = adv.shape[0]

    def put(r0: int, n: int, a0: int) -> None:
        for i in range(n):
            r = r0 + i
            if 0 <= r < m:
                bits[r] = adv[(a0 + i) % nr]

    put(0, min(nr, m), 0)
    for t, a0 in ((128, 9), (256, 0), (384, 4)):
        put(t - 2, 4, a0)
    put(m - min(nr, m), min(nr, m), 7)
    return bits


def hash_weight(salt: int = _SC_SALT + 16, std: float = 0.02) -> np.ndarray:
    """fp32 [1536, 512] ~ N(0, std) from integer hashes (Irwin-Hall of four 24-bit uniforms; no RNG)."""
    n = NOUT * D
    u = sum(((_hash_np(n, salt + i) & 0xFFFFFF).astype(np.float64) for i in range(4)), np.zeros(n)) / float(1 << 24)
    return ((u - 2.0) * np.sqrt(3.0) * std).astype(np.float32).reshape(NOUT, D)


def triton_dual_reference(dout: torch.Tensor, inv_scale: torch.Tensor, t_shift: int
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(dout_t as uint8 [M_PAD/128, 512, 128], PARTIAL fp32) of ONE direct launch of the shipped
    bwd_fp8._quantize_amax_dual_kernel into plain torch buffers, with exactly the arguments
    quantize_row_major_dual(dout, inv_scale, t_dtype=e4m3fn, t_inv_mul=2^-t_shift, t_limit=448, return_partial=True,
    t_tiled=True) passes (the out-proj FP8 path's call in _quantize_gradient_and_weight_fused). Not
    quantize_row_major_dual itself: it allocates dout_t from fp8_tm_alloc.empty_tm (the TM pool must not be touched
    before RUN_START)."""
    from generative_recommenders.ops.triton import bwd_fp8 as bf

    dtype, limit = bf._GRAD_DTYPE, bf._GRAD_MAX  # quantize_row_major_dual's dtype=None / limit=None
    t_dtype, t_inv_mul, t_limit = bf._FP8_E4M3, 2.0 ** -int(t_shift), bf._E4M3_MAX
    if not bf._TRITON_AVAILABLE or dtype is None or t_dtype is None:
        raise RuntimeError("the Triton dual-cast reference needs Triton and the torch fp8 dtypes")
    source = dout.detach()
    if not source.is_contiguous():
        source = source.contiguous()
    m, n = source.shape
    bm, bn, num_warps = bf._DUAL_CAST_CONFIG
    bn = min(bn, n)
    m_pad = (m + STEP_TOKENS - 1) // STEP_TOKENS * STEP_TOKENS  # t_tiled: pad_to = 128
    out = torch.empty((m, n), dtype=dtype, device=source.device)
    separate_t = t_dtype is not None and (t_dtype != dtype or t_inv_mul != 1.0)
    out_t = torch.empty((m_pad // 128, n, 128), dtype=t_dtype if separate_t else dtype, device=source.device)
    grid = ((m_pad + bm - 1) // bm, n // bn)
    partial = torch.empty(grid[0] * grid[1], dtype=torch.float32, device=source.device)
    bf._quantize_amax_dual_kernel[grid](
        source,
        out,
        out_t,
        partial,
        inv_scale,
        m,
        m_pad,
        n,
        source.stride(0),
        m_pad,
        BLOCK_M=bm,
        BLOCK_N=bn,
        LIMIT=limit,
        SEPARATE_T=separate_t,
        T_INV_MUL=float(t_inv_mul) if separate_t else 1.0,
        LIMIT_T=float(t_limit if t_limit is not None else limit) if separate_t else 1.0,
        T_TILED=True,
        num_warps=num_warps,
    )
    return out_t.view(torch.uint8), partial


def _nan32(u: np.ndarray) -> np.ndarray:
    return ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)


def partial_mismatch(got_bits: np.ndarray, ref_bits: np.ndarray) -> int:
    """PARTIAL entries that differ NaN-aware (NaN-ness equal, non-NaN bits equal); a length mismatch counts all."""
    g = np.asarray(got_bits, dtype=np.uint32).reshape(-1)
    r = np.asarray(ref_bits, dtype=np.uint32).reshape(-1)
    if g.shape != r.shape:
        return max(g.size, r.size, 1)
    gn, rn = _nan32(g), _nan32(r)
    return int(((gn != rn) | (~gn & (g != r))).sum())


# ---------------------------------------------------------------------------- G6 compare: S1C 2.5 as amended (L2(a))
# v1.2, r16 iteration 59 R-4 (R4.md sec. 10: the rule and its numpy reference). A G6 code may differ from REF only in
# the sign bit (0x20) of an E2M3 zero, in a block whose two E8M0 bytes are equal and not 255. Such a difference cannot
# change any output of r16's MX6 L11 (its EPI2 "+ 0.0f" maps -0 to +0; R4.md sec. 1-7), so it is counted and reported
# per direction, never gated. E8M0 bytes stay strict, every code of a byte-255 block stays strict, determinism twins
# stay byte-strict, and the rule covers G6 codes only (dout_t, PARTIAL, INV_USED, W6 and W6S keep their rules). Used by
# _sc_case (SC1 / SC2 / SC3 / SC6; NIT-3) and audit_last_bands (D16 / P9); r16's sealed REF g6_compare is not edited.
_E2M3_SIGN: int = 0x20
_E2M3_MAG: int = 0x1F


def g6_codes(g6: Any) -> np.ndarray:
    """uint8 [M, 400] G6 records -> int64 [M, 16, 32] 6-bit codes. Block b (K elements [32b, 32b + 32)) is the 24-byte
    little-endian group of head bytes 16b .. 16b + 15 and tail bytes 256 + 8b .. 256 + 8b + 7; code f sits at bits
    [6f, 6f + 6) (the inverse of pack24 / _assemble_g6 for the code bytes)."""
    g = np.asarray(g6)
    m = g.shape[0]
    head = g[:, _HEAD_OFF:_TAIL_OFF].reshape(m, _NBLK, 16)
    tail = g[:, _TAIL_OFF:_SC_OFF].reshape(m, _NBLK, 8)
    t = np.concatenate([head, tail], -1).reshape(m, _NBLK, 8, 3).astype(np.int64)
    v = t[..., 0] | (t[..., 1] << 8) | (t[..., 2] << 16)
    return np.stack([(v >> (6 * j)) & 63 for j in range(4)], -1).reshape(m, _NBLK, _BLK)


def g6_block_bytes(g6: Any) -> np.ndarray:
    """uint8 [M, 400] G6 records -> int64 [M, 16] E8M0 bytes in block order: block b = 4k + q is record byte
    384 + 4q + k (the inverse of _assemble_g6 for the scale bytes)."""
    g = np.asarray(g6)
    return g[:, _SC_OFF:G6_ROW].reshape(-1, 4, 4).transpose(0, 2, 1).reshape(-1, _NBLK).astype(np.int64)


def g6_l2a_counts(got: Any, ref: Any) -> Dict[str, int]:
    """G6 records `got` vs `ref` (two uint8 [M, 400] arrays of one shape) under S1C 2.5 as amended (L2(a), the section
    comment above):
      scale    E8M0 bytes [384, 400) that differ (strict, per byte);
      code     6-bit codes that fail. A code passes iff it equals REF's, or both codes have zero magnitude
               (code & 0x1F == 0) in a block whose REF E8M0 byte is not 255 and equals got's;
      bad      scale + code (0 = pass);
      neutral  the codes that differ and pass (value-neutral: report, never gate), split by direction into
               neutral_00_for_20 (got 0x00 where REF has 0x20) and neutral_20_for_00 (got 0x20 where REF has 0x00).
    (bad, neutral) is g6_mismatch_l2a(got, ref), R4.md sec. 10's numpy reference. Equal records return zeros without
    decoding. ValueError for anything but two uint8 [M, 400] arrays of one shape."""
    g, r = np.asarray(got), np.asarray(ref)
    if g.dtype != np.uint8 or r.dtype != np.uint8 or g.ndim != 2 or g.shape != r.shape or g.shape[1] != G6_ROW:
        raise ValueError(f"G6 compare needs two uint8 [M, {G6_ROW}] arrays of one shape; got {g.dtype} "
                         f"{tuple(g.shape)} vs {r.dtype} {tuple(r.shape)}")
    out = {"bad": 0, "scale": 0, "code": 0, "neutral": 0, "neutral_00_for_20": 0, "neutral_20_for_00": 0}
    if np.array_equal(g, r):
        return out
    bg, br = g6_block_bytes(g), g6_block_bytes(r)
    ok_blk = ((br != _BYTE_NONFINITE) & (bg == br))[..., None]
    cg, cr = g6_codes(g), g6_codes(r)
    differ = cg != cr
    neutral = differ & ((cg & _E2M3_MAG) == 0) & ((cr & _E2M3_MAG) == 0) & ok_blk
    out["scale"] = int((bg != br).sum())
    out["code"] = int((differ & ~neutral).sum())
    out["bad"] = out["scale"] + out["code"]
    out["neutral"] = int(neutral.sum())
    out["neutral_00_for_20"] = int((neutral & ((cg & _E2M3_SIGN) == 0)).sum())
    out["neutral_20_for_00"] = out["neutral"] - out["neutral_00_for_20"]
    return out


def g6_mismatch_l2a(got: Any, ref: Any) -> Tuple[int, int]:
    """(bad, neutral) for uint8 [M, 400] G6 records under S1C 2.5 as amended (L2(a)): the signature and result of r16
    R4.md sec. 10's numpy reference. bad == 0 is a pass; neutral is reported, never gated. g6_l2a_counts has the parts
    and the per-direction split."""
    c = g6_l2a_counts(got, ref)
    return c["bad"], c["neutral"]


# ---------------------------------------------------------------------------- self-check (DESIGN 12.7)
_INV_PUBLISH_143: int = 0x424880E5  # _SiteScale.publish() at window amax 143: fl32(1 / fl32(143 * fl32(8 / 57344)))
_INV_ONE: int = 0x3F800000
_INV_NAN: int = 0x7FC00000
_POISONS: Tuple[int, int] = (0xA5, 0x5A)
_G6_GUARD: int = 4096
_P_GUARD: int = 4
_IU_WORDS: int = 4


def _i32(word: int) -> int:
    word &= 0xFFFFFFFF
    return word - (1 << 32) if word >= (1 << 31) else word


def _word(byte: int) -> int:
    return int.from_bytes(bytes([byte]) * 4, "little")


def _bf16_dev(bits: np.ndarray, device: torch.device, off16: bool) -> torch.Tensor:
    h = torch.from_numpy(np.ascontiguousarray(bits, dtype=np.uint16).view(np.int16)).to(device)
    if not off16:
        return h.view(torch.bfloat16)
    m = int(bits.shape[0])
    buf = torch.empty((m * D * 2 + 16,), dtype=torch.uint8, device=device)
    x = buf[16:].view(torch.int16).view(m, D)  # a 16-B (not 32-B) aligned base
    x.copy_(h)
    return x.view(torch.bfloat16)


def _f32_dev(bits: int, device: torch.device, off16: bool) -> torch.Tensor:
    buf = torch.full((8,), _i32(bits), dtype=torch.int32, device=device)
    return (buf[4:5] if off16 else buf[0:1]).view(torch.float32)


def _poisoned(m: int, m_pad: int, poison: int, device: torch.device, off16: bool) -> Dict[str, Any]:
    """Output buffers prefilled with `poison` plus guards (G6 + 4,096 B, dout_t + one step, PARTIAL + 4 entries,
    INV_USED + 3 words); off16: every base 16 B past a fresh allocation (16-B, not 32-B aligned)."""
    off = 16 if off16 else 0
    n_g6, n_dt, n_p = m * G6_ROW, (m_pad // STEP_TOKENS) * STEP_BYTES, partial_count(m)
    w = _i32(_word(poison))
    o: Dict[str, Any] = {"off": off, "poison": poison}
    o["raw_g6"] = torch.full((off + n_g6 + _G6_GUARD,), poison, dtype=torch.uint8, device=device)
    o["raw_dt"] = torch.full((off + n_dt + STEP_BYTES,), poison, dtype=torch.uint8, device=device)
    o["raw_p"] = torch.full((off // 4 + n_p + _P_GUARD,), w, dtype=torch.int32, device=device)
    o["raw_iu"] = torch.full((off // 4 + _IU_WORDS,), w, dtype=torch.int32, device=device)
    o["g6"] = o["raw_g6"][off:off + n_g6].view(m, G6_ROW)
    o["dt"] = o["raw_dt"][off:off + n_dt].view(m_pad // STEP_TOKENS, D, STEP_TOKENS)
    o["p"] = o["raw_p"][off // 4:off // 4 + n_p].view(torch.float32)
    o["iu"] = o["raw_iu"][off // 4:off // 4 + 1].view(torch.float32)
    return o


def _outside(a: np.ndarray, lo: int, hi: int, value: int) -> int:
    """Elements of `a` outside [lo, hi) that differ from `value` (the poison: guards and the bytes before a base)."""
    return int((a[:lo] != value).sum()) + int((a[hi:] != value).sum())


def _sc_case(label: str, bits: np.ndarray, inv_bits: int, t: int, device: torch.device,
             ref_g6: Optional[np.ndarray] = None, inv_used: bool = True, off16: bool = False) -> Dict[str, Any]:
    """One producer case: two poisoned launches vs the in-tree G6 encoder and a direct Triton dual-cast launch. G6
    follows S1C 2.5 as amended (L2(a), g6_l2a_counts; v1.2, NIT-3): "g6" counts the failing E8M0 bytes and codes,
    "g6_neutral" the value-neutral zero-sign codes (not gated), split into "g6_n00" (0x00 for REF's 0x20) and "g6_n20"
    (0x20 for REF's 0x00). "twin" compares the two runs byte for byte (G6 included): determinism stays strict."""
    m = int(bits.shape[0])
    m_pad = (m + STEP_TOKENS - 1) // STEP_TOKENS * STEP_TOKENS
    n_g6, n_dt, n_p = m * G6_ROW, (m_pad // STEP_TOKENS) * STEP_BYTES, partial_count(m)
    x = _bf16_dev(bits, device, off16)
    inv = _f32_dev(inv_bits, device, off16)
    ref_t, ref_p = triton_dual_reference(_bf16_dev(bits, device, False), _f32_dev(inv_bits, device, False), t)
    if ref_g6 is None:
        ref_g6 = encode_g6_int(bits)
    runs = []
    for poison in _POISONS:
        o = _poisoned(m, m_pad, poison, device, off16)
        launch(x, inv, 2.0 ** -t, m, m_pad, o["g6"], o["dt"], o["p"], o["iu"] if inv_used else None)
        runs.append(o)
    ref_dt = ref_t.reshape(-1).cpu().numpy()
    ref_pb = ref_p.cpu().numpy().view(np.uint32)
    res: Dict[str, Any] = {"case": label, "M": m, "t": t, "inv": f"{inv_bits:#010x}", "g6": 0, "g6_neutral": 0,
                           "g6_n00": 0, "g6_n20": 0, "dt": 0, "p": 0, "iu": 0, "guard": 0, "twin": 0}
    prev: Optional[Tuple[np.ndarray, ...]] = None
    for o in runs:
        off, pw = o["off"], np.uint32(_word(o["poison"]))
        g = o["raw_g6"].cpu().numpy()
        dt = o["raw_dt"].cpu().numpy()
        pb = o["raw_p"].cpu().numpy().view(np.uint32)
        iu = o["raw_iu"].cpu().numpy().view(np.uint32)
        o4 = off // 4
        res["guard"] += (_outside(g, off, off + n_g6, o["poison"]) + _outside(dt, off, off + n_dt, o["poison"])
                         + _outside(pb, o4, o4 + n_p, pw) + _outside(iu, o4, o4 + 1, pw))
        gv, dv, pv = g[off:off + n_g6], dt[off:off + n_dt], pb[o4:o4 + n_p]
        c6 = g6_l2a_counts(gv.reshape(m, G6_ROW), ref_g6)  # L2(a): E8M0 / byte-255 blocks strict, zero signs neutral
        res["g6"] += c6["bad"]
        res["g6_neutral"] += c6["neutral"]
        res["g6_n00"] += c6["neutral_00_for_20"]
        res["g6_n20"] += c6["neutral_20_for_00"]
        res["dt"] += int((dv != ref_dt).sum())
        res["p"] += partial_mismatch(pv, ref_pb)
        res["iu"] += int(iu[o4] != (np.uint32(inv_bits) if inv_used else pw))
        cur = (gv, dv, pv) + ((iu[o4:o4 + 1],) if inv_used else ())  # a NULL INV_USED slot keeps its own poison
        if prev is not None:
            res["twin"] += sum(int((a != b).sum()) for a, b in zip(prev, cur))
        prev = cur
    res["ok"] = all(res[k] == 0 for k in ("g6", "dt", "p", "iu", "guard", "twin"))
    return res


def _sc_w6_case(label: str, w: torch.Tensor) -> Dict[str, Any]:
    """One W6 case: two poisoned launches (guards) and the allocating w6_encode vs quant_w_int + w6_scatter."""
    codes, e = quant_w_int(w.float().cpu().numpy())
    ref6, ref6s = w6_scatter(codes, e)
    res: Dict[str, Any] = {"case": label, "w6": 0, "w6s": 0, "guard": 0, "twin": 0, "alloc": 0}
    prev: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for poison in _POISONS:
        b6 = torch.full((W6_BYTES + _G6_GUARD,), poison, dtype=torch.uint8, device=w.device)
        bs = torch.full((W6S_BYTES + 256,), poison, dtype=torch.uint8, device=w.device)
        _w6_launch(w, out=(b6[:W6_BYTES], bs[:W6S_BYTES]))
        h6, hs = b6.cpu().numpy(), bs.cpu().numpy()
        res["guard"] += int((h6[W6_BYTES:] != poison).sum()) + int((hs[W6S_BYTES:] != poison).sum())
        res["w6"] += int((h6[:W6_BYTES] != ref6).sum())
        res["w6s"] += int((hs[:W6S_BYTES] != ref6s).sum())
        if prev is not None:
            res["twin"] += int((prev[0] != h6[:W6_BYTES]).sum()) + int((prev[1] != hs[:W6S_BYTES]).sum())
        prev = (h6[:W6_BYTES], hs[:W6S_BYTES])
    a6, a6s = w6_encode(w)
    res["alloc"] = int((a6.cpu().numpy() != ref6).sum()) + int((a6s.cpu().numpy() != ref6s).sum())
    res["ok"] = all(res[k] == 0 for k in ("w6", "w6s", "guard", "twin", "alloc"))
    return res


def _sc_launcher_args(lib: "_Lib", device: torch.device) -> Dict[str, Any]:
    """SC0 (host only): the C launchers reject bad arguments with rc 1 (hipErrorInvalidValue) BEFORE any HIP call.
    Every pointer lies inside one zeroed 4 MiB scratch allocation, so even a wrongly accepted case could only write
    there, and the scratch must still be all zero afterwards (nothing launched)."""
    scratch = torch.zeros((4 << 20,), dtype=torch.uint8, device=device)
    p = scratch.data_ptr()
    f = lib.fn["outproj_dual_mx6_launch"]
    t5 = 2.0 ** -5
    cases = [
        ("M=0", (p, p, t5, 0, 0, p, p, p, None, None)),
        ("M_PAD=256 for M=1", (p, p, t5, 1, 256, p, p, p, None, None)),
        ("M_PAD=0 for M=1", (p, p, t5, 1, 0, p, p, p, None, None)),
        ("t_inv_mul=1.0", (p, p, 1.0, 1, 128, p, p, p, None, None)),
        ("t_inv_mul=2^-3", (p, p, 2.0 ** -3, 1, 128, p, p, p, None, None)),
        ("t_inv_mul=2^-8", (p, p, 2.0 ** -8, 1, 128, p, p, p, None, None)),
        ("t_inv_mul=0.03", (p, p, 0.03, 1, 128, p, p, p, None, None)),
        ("DOUT+8", (p + 8, p, t5, 1, 128, p, p, p, None, None)),
        ("INV+4", (p, p + 4, t5, 1, 128, p, p, p, None, None)),
        ("G6+4", (p, p, t5, 1, 128, p + 4, p, p, None, None)),
        ("DOUT_T+8", (p, p, t5, 1, 128, p, p + 8, p, None, None)),
        ("PARTIAL+4", (p, p, t5, 1, 128, p, p, p + 4, None, None)),
        ("INV_USED+4", (p, p, t5, 1, 128, p, p, p, p + 4, None)),
    ]
    w6f = lib.fn["outproj_w6_shadow_launch"]
    w6_cases = [("W+4", (p + 4, 0, p, p, None)), ("W6+8", (p, 0, p + 8, p, None)), ("W6S+4", (p, 1, p, p + 4, None))]
    bad: List[str] = []
    for name, args in cases:
        rc = int(f(*args))
        if rc != 1:
            bad.append(f"{name}: rc={rc}")
    for name, args in w6_cases:
        rc = int(w6f(*args))
        if rc != 1:
            bad.append(f"{name}: rc={rc}")
    torch.cuda.synchronize(device)
    written = int(scratch.ne(0).sum().item())
    if written:
        bad.append(f"scratch bytes written {written}")
    return {"n": len(cases) + len(w6_cases), "bad": bad, "ok": not bad}


def _c4_tile(m: int) -> np.ndarray:
    a = np.full((m, D), 0x3F00, dtype=np.uint16)
    a[:, 128:] = 0x3E80
    return a


def sc1_shifts() -> Tuple[int, ...]:
    """SC1's t values (DESIGN 12.7; v1.1): 5 / 6 / 7 plus every other t this process can cast at, i.e. bwd_fp8's
    _OUT_WGRAD_DFMT_SHIFT_FLOOR (where SHIFT=auto starts) and _OUT_WGRAD_DFMT_SHIFT (a FIXED shift, e.g.
    DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT_SHIFT=4, which never moves; under auto it equals the floor), within T_SHIFTS.
    DESIGN v1 added 4 only for FLOOR = 4, so a fixed shift 4 cast at a t the self-check never compared."""
    from generative_recommenders.ops.triton import bwd_fp8 as bf

    extra = set()
    for name in ("_OUT_WGRAD_DFMT_SHIFT_FLOOR", "_OUT_WGRAD_DFMT_SHIFT"):
        try:
            extra.add(int(getattr(bf, name)))
        except (AttributeError, TypeError, ValueError):
            pass
    return tuple(sorted({5, 6, 7} | (extra & set(T_SHIFTS))))


def self_check(device: Optional[torch.device] = None) -> Tuple[bool, str]:
    """DESIGN 12.7 SC0-SC6 on the loaded library (module docstring), on `device` (default: the current CUDA device).
    (ok, detail). Raises on device errors; the caller (prebuild) classifies."""
    lib = _lib_for_launch()
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    t0 = time.perf_counter()
    cases: List[Dict[str, Any]] = []
    sc0 = _sc_launcher_args(lib, dev)
    # SC1 (+ SC4: INV_USED = publish(143) bits): production-like + adversarial rows, the band / tail classes
    shifts = sc1_shifts()
    b769 = self_check_rows(769)
    g769 = encode_g6_int(b769)
    for t in shifts:
        cases.append(_sc_case(f"SC1 M=769 t={t}", b769, _INV_PUBLISH_143, t, dev, g769))
    b700 = self_check_rows(700)
    g700 = encode_g6_int(b700)
    cases.append(_sc_case("SC1 M=700 t=5", b700, _INV_PUBLISH_143, 5, dev, g700))
    cases.append(_sc_case("SC1 M=700 t=5 16-B-only bases", b700, _INV_PUBLISH_143, 5, dev, g700, off16=True))
    # SC2: critique C4's NaN tiles (PARTIAL NaN-aware; G6 and dout_t too)
    c4: List[Tuple[str, np.ndarray, bool]] = []
    a = _c4_tile(256)
    a[5, 7], a[9, 100] = 0x7FC0, 0x4000
    c4.append(("P1", a, True))
    a = _c4_tile(256)
    a[3, 3], a[4, 4] = 0x7FC0, 0x7F80
    c4.append(("P2", a, True))
    a = _c4_tile(100)
    a[:, :128] = 0x7FC0
    c4.append(("P3", a, False))  # also the NULL INV_USED path
    a = _c4_tile(256)
    a[:, :128] = 0x7FC0
    c4.append(("P4", a, True))
    a = _c4_tile(128)
    a[:, :128] = 0x7FC0
    c4.append(("P3b M=128", a, True))
    a = _c4_tile(384)
    a[256:, :128] = 0x7FC1
    c4.append(("P3c M=384", a, True))
    for name, a, iu in c4:
        cases.append(_sc_case(f"SC2 C4-{name}", a, _INV_ONE, 5, dev, inv_used=iu))
    # SC3 (+ SC4: INV_USED = 0x7fc00000): non-finite / zero / negative inv, pad tokens through the arithmetic
    b300 = self_check_rows(300)
    g300 = encode_g6_int(b300)
    for name, ib in (("NaN", _INV_NAN), ("+Inf", 0x7F800000), ("0", 0x00000000), ("-1", 0xBF800000)):
        cases.append(_sc_case(f"SC3 M=300 inv={name}", b300, ib, 5, dev, g300))
    # SC6: determinism twin at M = 4,097 (the two poisoned runs; references too)
    cases.append(_sc_case("SC6 twin M=4097", self_check_rows(4097), _INV_PUBLISH_143, 5, dev))
    # SC5: W6 / W6S
    w6_cases: List[Dict[str, Any]] = []
    for wname, wnp in (("edge", edge_weight()), ("hash N(0,0.02)", hash_weight())):
        wf = torch.from_numpy(wnp).to(dev)
        w6_cases.append(_sc_w6_case(f"SC5 W6 {wname} fp32", wf))
        w6_cases.append(_sc_w6_case(f"SC5 W6 {wname} bf16", wf.to(torch.bfloat16)))
    torch.cuda.synchronize(dev)
    adv_ok = hashlib.sha256(adversarial_rows().tobytes()).hexdigest() == ADV_ROWS_SHA256
    bad = [c for c in cases + w6_cases if not c["ok"]]
    ok = sc0["ok"] and not bad
    keys = ("g6", "dt", "p", "iu", "guard", "twin")
    tot = {k: sum(int(c.get(k, 0)) for c in cases) for k in keys}
    wtot = {k: sum(int(c.get(k, 0)) for c in w6_cases) for k in ("w6", "w6s", "guard", "twin", "alloc")}
    # L2(a) (v1.2): the value-neutral zero-sign G6 codes, per direction and per case class (SC1 / SC2 / SC3 / SC6)
    neu = {k: sum(int(c.get(k, 0)) for c in cases) for k in ("g6_neutral", "g6_n00", "g6_n20")}
    by_cls: Dict[str, List[int]] = {}
    for c in cases:
        if c.get("g6_neutral"):
            acc = by_cls.setdefault(str(c["case"]).split()[0], [0, 0])
            acc[0] += int(c["g6_n00"])
            acc[1] += int(c["g6_n20"])
    per_cls = ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(by_cls.items()))
    neutral = (f"G6 value-neutral zero-sign codes {neu['g6_neutral']} (S1C 2.5 L2(a), reported, not gated): "
               f"0x00-for-0x20 {neu['g6_n00']}, 0x20-for-0x00 {neu['g6_n20']}"
               + (f" (by class, 0x00-for-0x20/0x20-for-0x00: {per_cls})" if per_cls else ""))
    detail = (
        f"SC0 {sc0['n'] - len(sc0['bad'])}/{sc0['n']} bad launcher args rejected; "
        f"{len(cases)} producer cases x 2 poisons (SC1 M=769 t={','.join(str(t) for t in shifts)}, M=700 + 16-B "
        f"bases; SC2 C4 P1-P4/P3b/P3c; SC3 inv NaN/+Inf/0/-1; SC6 twin M=4097): mismatches G6 {tot['g6']}, dout_t "
        f"{tot['dt']}, PARTIAL {tot['p']}, INV_USED {tot['iu']}, guards {tot['guard']}, twin {tot['twin']}; "
        f"{len(w6_cases)} W6 cases (edge / hash x fp32 / bf16): W6 {wtot['w6']}, W6S {wtot['w6s']}, guards "
        f"{wtot['guard']}, twin {wtot['twin']}, alloc {wtot['alloc']}; adversarial rows sha {'ok' if adv_ok else 'DRIFT'}"
        f"; {time.perf_counter() - t0:.1f}s"
    )
    if not ok:
        first = [f"{c['case']}: " + ", ".join(f"{k} {c[k]}" for k in c if k in keys + ("w6", "w6s", "alloc") and c[k])
                 for c in bad[:4]]
        detail += "; FAILING " + " | ".join(first + ([f"SC0 {sc0['bad'][:4]}"] if sc0["bad"] else []))
    # the L2(a) report closes the detail: r16's prebuild_mx6 re-logs a failed self-check as msg[:600], so the counts
    # and the FAILING list above stay inside that window (v1.2 review; the text before it is v1.1's layout)
    return ok, f"{detail}; {neutral}"


# ---------------------------------------------------------------------------- dev audit (D16, lab P9)
_T_OF_BITS: Dict[int, int] = dict(zip(T_INV_MUL_BITS, T_SHIFTS))


def _audit_pending() -> bool:
    if _AUDIT["pending"] is None:
        _AUDIT["pending"] = os.environ.get(AUDIT_ENV, "").strip() == "1"
    return bool(_AUDIT["pending"])


def audit_last_bands(dout: torch.Tensor, inv_scale: torch.Tensor, t_inv_mul: float, M: int, m_pad: int,
                     g6: torch.Tensor, dout_t: torch.Tensor, partial: torch.Tensor, bands: int = 2
                     ) -> Tuple[bool, str]:
    """D16 / lab gate P9 (dev only; synchronizes): the last `bands` 256-row bands of a FINISHED producer call vs the
    in-tree G6 encoder and a direct Triton dual-cast launch on the same rows (the slice starts on a band boundary, so
    its dout_t steps and PARTIAL entries are the tail of the full call's). G6 follows SC1's rule, S1C 2.5 as amended
    (L2(a), g6_l2a_counts; v1.2): value-neutral zero-sign codes are reported, not counted as mismatches."""
    m, mp = int(M), int(m_pad)
    nb = (m + BAND_ROWS - 1) // BAND_ROWS
    b0 = max(0, nb - int(bands))
    r0 = BAND_ROWS * b0
    t = _T_OF_BITS.get(_f32_bits(float(t_inv_mul)))
    if t is None:
        return False, f"t_inv_mul {t_inv_mul!r} is not 2^-4 .. 2^-7"
    xs = dout[r0:m]
    ref_t, ref_p = triton_dual_reference(xs, inv_scale, t)
    bits = xs.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    c6 = g6_l2a_counts(g6[r0:m].cpu().numpy(), encode_g6_int(bits))  # L2(a), as SC1
    g6_bad = c6["bad"]
    got_t = dout_t.reshape(-1).view(torch.uint8)[(r0 // STEP_TOKENS) * STEP_BYTES:(mp // STEP_TOKENS) * STEP_BYTES]
    dt_bad = int((got_t.cpu().numpy() != ref_t.reshape(-1).cpu().numpy()).sum())
    p_bad = partial_mismatch(partial.reshape(-1)[4 * b0:4 * nb].cpu().numpy().view(np.uint32),
                             ref_p.cpu().numpy().view(np.uint32))
    ok = g6_bad == 0 and dt_bad == 0 and p_bad == 0
    return ok, (f"M={m} rows [{r0}, {m}) bands {b0}..{nb - 1} t={t}: G6 mism {g6_bad} (L2(a); value-neutral zero-sign "
                f"codes {c6['neutral']}: 0x00-for-0x20 {c6['neutral_00_for_20']}, 0x20-for-0x00 "
                f"{c6['neutral_20_for_00']}, not gated), dout_t mism {dt_bad}, PARTIAL mism {p_bad}")

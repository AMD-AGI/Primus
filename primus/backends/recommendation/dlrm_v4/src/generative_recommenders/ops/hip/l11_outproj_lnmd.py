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

# pyre-strict
"""L11: the HSTU output-projection FP8 dgrad fused with the LN*U*dropout backward.

Gin item (39) (yambda_5b.gin, DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED; only the
value "1" arms it, the gin binding writes "1", "0" or any other value is off).
Ported from the r16 testbed (its gin item (25)) onto the ship-12 candidate.

TWO MODES, chosen per call. With item (38)(a) B1a (DLRM_HSTU_FP8_BWD_DU_E5M2)
the backward first asks B1a's own plan (triton_hstu_linear._du_e5m2_plan,
evaluated once, between the cast and the GEMM) on every L11-eligible call:
- no plan (B1a off, or B1a declines this call) -> the bf16-du mode (`run`,
  below): du is written in bf16 into the d(uvqk) buffer, as v2 does;
- a plan (inv_scale, cast_scale, generation) (B1a would emit this call's e5m2
  u block) -> the e5m2-du mode (`run_e5`, via the hook
  triton_hstu_linear._l11_du_e5m2_or_none; see E5 MODE below): B1a's emission
  contract, produced without the dgrad GEMM. If the e5 mode declines or fails
  (latched), the hook returns None and the GEMM + B1a's path run on the same
  operands.

WHAT. In HSTUComputeOutputFunction.backward (ops/triton/triton_hstu_linear.py)
of the two ordinary STU layers, today's chain is

    dual cast (bwd_fp8.quantize_gradient, kept) -> W8 cast (kept)
    -> torch._scaled_mm(G8, W8.t(), sa, sb) = hipBLASLt F8B8BS NTB0 stream-K
       dgrad, dy [M, 1536] bf16 written to HBM (7.3 GB at M = 2.36M)
    -> _ln_mul_dropout_bwd_dx_du_rng_v2 (reads dy back)
    -> _ln_mul_dropout_bwd_dwdb finalize (kept).

With the knob on, ONE persistent HIP kernel (l11_outproj_lnmd.cu,
`l11_outproj_lnmd_bwd`) computes the dgrad tile by tile, keeps the bf16 dy rows
in registers, redistributes them through LDS into v2's one-wave-per-row layout
and runs v2's per-element math op for op (FMA contraction pattern and the
tl.sum DPP tree read from v2's gfx950 ISA). dy never touches HBM. The kernel
writes dx, du (cols 0..511 of the [M, 2048] d(uvqk) buffer the preprocess
backward adopts, exactly as v2) and per-wave fp32 dW/dB partials [P, 512]
that the SAME Triton finalize reduces (N = P instead of 16384).

NUMERICS (declared; details in the gin item): same FP8 operands, products,
fp32 accumulation and EPI2 epilogue (RNE_bf16(acc * (sa*sb) + 0.0f)) as
hipBLASLt; dy (never materialized) is bitwise equal to hipBLASLt's dy on its
data-parallel tiles and fp32-order (not bitwise) on the stream-K (SK3) tail
tiles (~0.7% of rows); du/dx are bitwise vs v2 on the same dy; dW/dB are a
declared fp32 reduction-order change (per-wave chains of <= 8*fl rows, P <=
16384 rows, v2-like chain lengths). Deterministic; no atomics.

BUILD / LOAD. hipcc -> .so -> ctypes (r14 wgrad_ck.py pattern), cached under
${DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED_CACHE:-/tmp/dlrm_hstu_fp8_l11}/<key>/
with key = sha256(source, flags, arch, `hipcc --version`)[:16]. Built and
self-checked in the pre-RUN_START prebuild (triton_fp8.prebuild_extensions ->
triton_hstu_linear.prebuild_l11, gin DLRM_HSTU_FP8_PREBUILD). Without a
successful prebuild the backward declines (D11): no hipcc build and no
unchecked kernel inside the timed run (run() still loads lazily for direct
callers: tests, the self-check). Any build / load / ABI / arch / resource
(VGPR spill or scratch) / self-check / launch failure latches the process off
("DISABLED") and the chain runs on the already-cast operands.

Every line is printed once per process on EVERY rank (ops.utils
.arm_print_all_ranks, "[rank N]"), and only when the knob is on. Every
NOT armed / DISABLED / INERT / DEVICE FAULT line starts with the family label
FAMILY ("L11 out-proj dgrad + LN*U*dropout bwd fused"), as the ACTIVE lines
of both modes do, so one census expectation covers all of them.

E5 MODE (r16 iter32 l11e5; item (38)(a) B1a, DLRM_HSTU_FP8_BWD_DU_E5M2).
`run_e5` is the fused replacement of "hipBLASLt out-proj dgrad + B1a's
_triton_layer_norm_mul_dropout_bwd_du_e5m2" (B1a's
_ln_mul_dropout_bwd_dx_du_rng_v3 EMIT + dwdb + amax/colsum finalize): same
inputs except the FP8 dgrad operands (G8, W8, sa, sb) instead of dy, same
return (dx, placeholder du, dweight, dbias, hstu_du_e5m2.DuE5m2Emit). The
kernel's E5 instantiation (l11_outproj_lnmd_bwd<..., E5 = 1>, built into the
same .so as the bf16-du instantiation) writes the e5m2 u block of dz8 with
v3's exact ops (fp32(RNE_bf16(du)) * inv -> v_med3_f32 clamp +-57344 ->
v_cvt_scalef32_pk_bf8_f32), zeroes the pad rows, and emits per-wave amax and
[P, 512] u column-sum partials that are reduced exactly as B1a reduces its
own (partial.amax(), partial.sum(0)). Payload (incl. pad rows), amax and dx:
bitwise vs v3 on the same dy; u colsum: a declared fp32 reduction order
(per-wave chains, as dW/dB). The prebuild self-check covers both modes (the
e5 part when B1a is enabled).

NIB (r16 iteration 54, knob DLRM_HSTU_FP8_LNMD_MASK_NIB, default off; the
contract is ops/lnmd_mask_nib.py). The D8 gate also accepts the packed keep
mask (uint8 [M, 256], lnmd_mask_nib.layout_ok) WITHOUT unpacking it: this gate
also gates DO8, whose NIB build (ops/hip/l11_do8.py) reads it. L11's own
kernels read int8 [M, 512], so run / launch_e5 / run_e5 pass
lnmd_mask_nib.as_int8(mask, ...) to them: the exact unpack of a NIB mask
(cached on the mask, counted, one line per process), int8 masks unchanged.
run / run_e5 unpack BEFORE their try blocks, so an unpack failure (e.g. an
OOM) raises to the caller and never latches item (39).
"""

import ctypes
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from generative_recommenders.ops import lnmd_mask_nib
from generative_recommenders.ops.utils import arm_print_all_ranks

KNOB: str = "DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED"
KERNEL_NAME: str = "l11_outproj_lnmd_bwd"
FAMILY: str = "L11 out-proj dgrad + LN*U*dropout bwd fused"

_SOURCE: Path = Path(__file__).with_name("l11_outproj_lnmd.cu")
# L11-VD (yambda_5b.gin item (75); code default off, gin binds it on): only DLRM_HSTU_FP8_L11_VD=1 builds this
# VALU-diet source instead of _SOURCE (same kernel name, ABI, launch and outputs, bit for bit). Absent, "0" or any
# other value: _SOURCE, exactly as without this item.
VD_KNOB: str = "DLRM_HSTU_FP8_L11_VD"
_SOURCE_VD: Path = Path(__file__).with_name("l11_outproj_lnmd_vd.cu")
_ARCH: str = "gfx950"
# The kernel-lab build flags of the L11-1b builder (probes/iter31/l11b/build.py), INCLUDING --save-temps: with it,
# hipcc runs the classic per-step device compile, which (a) is the flow whose code the builder validated (the
# integrated l11_outproj_lnmd_bwd<1,2,0,0,1,2,1,0> is instruction-identical to its b2 l11c_f1) and (b) is the flow
# that prints the -Rpass-analysis resource remarks the 0-spill / 0-scratch check below parses (ROCm 7.2.1 prints
# none without it). The temporaries go to a private build directory that is deleted after the build.
_FLAGS: Tuple[str, ...] = (
    "-DKITTENS_CDNA4",
    "-std=c++20",
    "-w",
    "-O3",
    "-mllvm",
    "-amdgpu-sched-strategy=iterative-ilp",
    "-ffp-contract=off",
    "--save-temps",
    "-shared",
    "-fPIC",
    "-Rpass-analysis=kernel-resource-usage",
)
_CACHE_ROOT: str = "/tmp/dlrm_hstu_fp8_l11"
# ROCm 10.2 (T14b): the consumer's in-lane row sums (c1 = sum(xhat * wdy), c2 = sum(wdy) over a lane's 8 columns, before
# the DPP tree) follow the Triton that compiles v2 / v3, the kernels L11 and DO8 reproduce bit for bit (the prebuild
# self-check and ops/tests/l11_outproj_lnmd_test.py gate it). Triton 3.6 sums them sequentially; Triton >= 3.9 lowers
# the reduce as a 2-wide tree that LLVM contracts into v_pk_fma_f32 (read from the 3.9 gfx950 ISA of v2 and v3: same
# structure). With 3.9 the 3.6 order breaks that equality on ~1e-5 of the dx elements (self-check "same-dy du/dx mism
# 0/207"), which latches L11, and with it DO8, off. tlsum_tree() -> TLSUM_DEFINE builds the 3.9 order (the .cu note at
# L11_TLSUM_TREE); with Triton 3.6 nothing changes (flags, cache key, code).
TLSUM_DEFINE: str = "-DL11_TLSUM_TREE=1"
_TLSUM_TREE_MIN_TRITON: Tuple[int, int] = (3, 9)
_LIB_NAME: str = "libl11_outproj_lnmd.so"
_BUILD_TIMEOUT_S: int = 300
_ABI_VERSION: int = 2  # 2 = + l11_outproj_lnmd_bwd_e5_launch (r16 iter32 l11e5)

D: int = 512  # LN width (= K of the dgrad)
NOUT: int = 1536  # dgrad output columns (u | x | y planes)
DU_ROW: int = 2048  # du row width (the d(uvqk) buffer)
# Largest M the unit test validated (ops/tests/l11_outproj_lnmd_test.py); the .cu enforces the same bound.
M_MAX: int = 4_194_304

_lock = threading.Lock()
_requested: Optional[bool] = None  # D0, read lazily (the gin env bootstrap runs after import)
_override: Optional[bool] = None  # tests only
_broken: Optional[str] = None
_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None
_fn_launch: Optional[Callable[..., int]] = None
_fn_launch_e5: Optional[Callable[..., int]] = None
_fn_plan: Optional[Callable[..., int]] = None
_fn_dbg: Optional[Callable[..., int]] = None
_fn_drain: Optional[Callable[[], int]] = None
_impl: str = ""
_resources: Dict[str, Any] = {}
_build_secs: float = 0.0
_prebuilt: bool = False
_active_logged: bool = False
_active_e5_logged: bool = False
_once: set = set()
_arch_cache: Dict[int, str] = {}
_toolchains: Dict[str, str] = {}
_plan_cache: Dict[int, Tuple[int, int, int, int]] = {}
_tlsum: Optional[bool] = None  # ROCm 10.2 (T14b): tlsum_tree(), read once


# ---------------------------------------------------------------------------- knob / latch / lines
def requested() -> bool:
    """D0: DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED == "1" (read once, lazily)."""
    global _requested
    if _override is not None:
        return _override
    if _requested is None:
        _requested = os.environ.get(KNOB, "0") == "1"
    return _requested


def enabled() -> bool:
    return requested() and _broken is None


def broken() -> Optional[str]:
    return _broken


def library_path() -> Optional[Path]:
    return _lib_path


def resources() -> Dict[str, Any]:
    return dict(_resources)


def impl() -> str:
    return _impl


def set_requested_for_testing(value: Optional[bool]) -> None:
    """Tests only: override the knob (None restores the lazy env read, re-read on next use)."""
    global _override, _requested
    _override = value
    _requested = None


def reset_for_testing(unload: bool = False) -> None:
    """Tests only: clear the latch, the once-lines, the ACTIVE line and the prebuilt flag; unload=True also forgets
    the library."""
    global _broken, _active_logged, _prebuilt, _lib, _lib_path, _fn_launch, _fn_plan, _fn_dbg, _fn_drain, _impl
    global _fn_launch_e5, _active_e5_logged
    _broken = None
    _active_logged = False
    _active_e5_logged = False
    _prebuilt = False
    _once.clear()
    _plan_cache.clear()
    if unload:
        with _lock:
            _lib = None
            _lib_path = None
            _fn_launch = _fn_plan = _fn_dbg = _fn_drain = _fn_launch_e5 = None
            _impl = ""
            _resources.clear()


def prebuilt() -> bool:
    """True once prebuild() built, loaded and self-checked the kernel in this process (decline D11 otherwise)."""
    return _prebuilt


def set_prebuilt_for_testing(value: bool) -> None:
    """Tests only: mark the prebuild as done / not done (decline D11) without running it."""
    global _prebuilt
    _prebuilt = value


def _fail_closed() -> bool:
    """DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED_FAIL_CLOSED=1 (smoke/debug only, default off): raise instead of latching
    off. Deliberately NOT the global DLRM_HSTU_FP8_FAIL_CLOSED: the FP8 forward-arm preset turns that one on in
    production, where it guards FP8 -> bf16 fallbacks; L11's fallback is the unchanged FP8 chain (same numerics as
    the knob off) and is printed on every rank, like r14's wgrad_ck latch."""
    return os.environ.get(KNOB + "_FAIL_CLOSED", "0") == "1"


def _print_once(key: str, message: str) -> None:
    if key in _once:
        return
    _once.add(key)
    arm_print_all_ranks(message)


def _disable(reason: str) -> None:
    """Latch L11 off for this process (the chain runs from here on) and print one DISABLED line."""
    global _broken
    if _fail_closed():
        raise RuntimeError(f"{FAMILY} failed with {KNOB}_FAIL_CLOSED=1: {reason}")
    if _broken is not None:
        return
    _broken = reason
    arm_print_all_ranks(
        f"{FAMILY} DISABLED: {reason} -> hipBLASLt dgrad + unfused LN*U*dropout bwd for this process"
    )


def decline_key(reason: str) -> str:
    """The once-per-process key of a decline reason: its D-code and first word without values ('D8 M', 'D6 flags')."""
    tokens = reason.split()
    key = tokens[0] if tokens else reason
    if len(tokens) > 1:
        key += " " + re.split(r"[=:(\d]", tokens[1], maxsplit=1)[0]
    return key


def note_decline(reason: str) -> None:
    """One 'NOT armed' line per (reason, process); only reached with the knob on."""
    _print_once(
        "decline:" + decline_key(reason),
        f"{FAMILY} NOT armed: {reason} -> hipBLASLt dgrad + unfused LN*U*dropout bwd",
    )


# ---------------------------------------------------------------------------- build / load
def _hipcc() -> str:
    override = os.environ.get(KNOB + "_HIPCC", "").strip()
    if override:
        return override
    rocm = os.environ.get("ROCM_PATH", "").strip()
    if rocm:
        candidate = os.path.join(rocm, "bin", "hipcc")
        if os.path.exists(candidate):
            return candidate
    return "/opt/rocm/bin/hipcc"


def _toolchain(hipcc: str) -> str:
    """`hipcc --version` (cached per hipcc path; raises if hipcc cannot run)."""
    version = _toolchains.get(hipcc)
    if version is None:
        proc = subprocess.run([hipcc, "--version"], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{hipcc} --version rc={proc.returncode}: "
                f"{((proc.stderr or '') + (proc.stdout or '')).strip()[-400:]}"
            )
        version = proc.stdout
        _toolchains[hipcc] = version
    return version


def vd_requested() -> bool:
    """L11-VD (r16 iter37; default off): DLRM_HSTU_FP8_L11_VD == "1" selects the consumer-VALU-diet source
    l11_outproj_lnmd_vd.cu; absent, "0" or any other value keeps l11_outproj_lnmd.cu."""
    return os.environ.get(VD_KNOB, "0").strip() == "1"


def _source() -> Path:
    """The kernel source; DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED_SRC overrides it (A/B of a candidate .cu only).
    Otherwise l11_outproj_lnmd.cu, or l11_outproj_lnmd_vd.cu when DLRM_HSTU_FP8_L11_VD=1 (L11-VD). Both sources
    produce byte-identical outputs (dx, du, amax, column-sum and dW/dB partials, NaN payloads included); the cache key
    covers the source bytes, so each one builds its own .so."""
    override = os.environ.get(KNOB + "_SRC", "").strip()
    if override:
        return Path(override)
    return _SOURCE_VD if vd_requested() else _SOURCE


def tlsum_tree() -> bool:
    """ROCm 10.2 (T14b): this process's Triton (it compiles v2 / v3) is >= _TLSUM_TREE_MIN_TRITON, whose in-lane tl.sum
    order the L11 / DO8 consumer must reproduce (TLSUM_DEFINE). Read once; False if Triton cannot be imported."""
    global _tlsum
    if _tlsum is None:
        try:
            import triton

            m = re.match(r"(\d+)\.(\d+)", str(triton.__version__))
            _tlsum = m is not None and (int(m.group(1)), int(m.group(2))) >= _TLSUM_TREE_MIN_TRITON
        except Exception:  # noqa: BLE001 - no Triton: no v2 / v3 to match, keep the shipped order
            _tlsum = False
    return _tlsum


def _flags(src: bytes) -> List[str]:
    sha = hashlib.sha256(src).hexdigest()
    # ROCm 10.2 (T14b): + TLSUM_DEFINE with Triton >= 3.9 (the in-lane row-sum order of its v2 / v3)
    return [f"--offload-arch={_ARCH}", *_FLAGS, f'-DL11_SRC_SHA256="{sha}"'] + ([TLSUM_DEFINE] if tlsum_tree() else [])


def cache_key(source: Optional[Path] = None) -> str:
    source = source or _source()
    src = source.read_bytes()
    digest = hashlib.sha256()
    digest.update(src)
    digest.update(b"\0" + "\0".join(_flags(src)).encode())
    digest.update(b"\0" + _ARCH.encode())
    # The toolchain too: a .so another hipcc built is not reused, so the resource check below holds for this one.
    digest.update(b"\0" + _toolchain(_hipcc()).encode())
    return digest.hexdigest()[:16]


def _cache_dir(source: Optional[Path] = None) -> Path:
    root = os.environ.get(KNOB + "_CACHE", "").strip() or _CACHE_ROOT
    return Path(root) / cache_key(source)


# ROCm 10.2 form (no "remark:", no "[-Rpass-analysis=...]"): "<file>:<line>:<col>: <Key>: <value>", known keys only.
_REMARK_ROCM102 = re.compile(
    r"\S+?:\d+:\d+:\s+((?:Function Name|TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|"
    r"Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]):\s*\S+)\s*$"
)


def parse_resource_remarks(log: str) -> List[Dict[str, str]]:
    """-Rpass-analysis=kernel-resource-usage remarks, one dict per kernel (pr7 k512 resource_summary pattern)."""
    kernels: List[Dict[str, str]] = []
    rocm102: List[bool] = []  # per kernel: its "Function Name" line was in the ROCm 10.2 form
    for line in log.splitlines():
        m = re.search(r"remark:\s+\S+?:\d+:\d+:\s+(.*?)\s*\[-Rpass-analysis", line) or _REMARK_ROCM102.match(line)
        if not m:
            continue
        kv = re.match(r"\s*([^:]+):\s*(\S+)", m.group(1))
        if not kv:
            continue
        key, value = kv.group(1).strip(), kv.group(2)
        if key == "Function Name":
            kernels.append({})
            rocm102.append(m.re is _REMARK_ROCM102)
        if kernels:
            kernels[-1][key] = value
    # ROCm 10.2 --save-temps runs the LTO device codegen twice (.lto.s, .lto.o): an identical repeated block counts once
    return [k for i, k in enumerate(kernels) if not (rocm102[i] and k in kernels[:i])]


def check_resources(kernels: List[Dict[str, str]]) -> Dict[str, Any]:
    """Raise unless every l11 kernel reports 0 VGPR spill and 0 scratch (a toolchain drift that reintroduces spills
    latches L11 off); returns the summary of the production kernel."""
    mine = [k for k in kernels if KERNEL_NAME in k.get("Function Name", "")]
    if len(mine) < 1:
        raise RuntimeError(f"no {KERNEL_NAME} resource remarks in the build log")
    for k in mine:
        spill = int(k.get("VGPRs Spill", "-1"))
        scratch = int(k.get("ScratchSize [bytes/lane]", "-1"))
        if spill != 0 or scratch != 0:
            raise RuntimeError(
                f"resource check: {k.get('Function Name')} VGPR spill {spill}, scratch {scratch} B/lane (need 0/0)"
            )
    # production instantiations: <1,2,0,0,1,2,1,0,E5> (bf16 du: E5 = 0; l11e5 e5m2 du: E5 = 1)
    prod = [k for k in mine if "ILi1ELi2ELi0ELi0ELi1ELi2ELi1ELi0ELi0EE" in k.get("Function Name", "")]
    e5 = [k for k in mine if "ILi1ELi2ELi0ELi0ELi1ELi2ELi1ELi0ELi1EE" in k.get("Function Name", "")]
    p = prod[0] if prod else mine[0]
    out = {
        "vgpr": int(p.get("VGPRs", "-1")),
        "sgpr": int(p.get("TotalSGPRs", "-1")),
        "vgpr_spill": int(p.get("VGPRs Spill", "-1")),
        "sgpr_spill": int(p.get("SGPRs Spill", "-1")),
        "scratch": int(p.get("ScratchSize [bytes/lane]", "-1")),
        "lds": int(p.get("LDS Size [bytes/block]", "-1")),
        "kernels": len(mine),
    }
    if e5:
        out.update(
            e5_vgpr=int(e5[0].get("VGPRs", "-1")),
            e5_sgpr=int(e5[0].get("TotalSGPRs", "-1")),
            e5_vgpr_spill=int(e5[0].get("VGPRs Spill", "-1")),
            e5_sgpr_spill=int(e5[0].get("SGPRs Spill", "-1")),
            e5_scratch=int(e5[0].get("ScratchSize [bytes/lane]", "-1")),
        )
    return out


def build(source: Optional[Path] = None) -> Tuple[Path, Dict[str, Any]]:
    """Compile (cache hit: no compile) and return (library path, resource summary). Raises on any failure (missing
    hipcc, compile error, timeout, VGPR spill or scratch)."""
    source = source or _source()
    so = _cache_dir(source) / _LIB_NAME
    meta = so.with_name(_LIB_NAME + ".resources.json")
    if so.is_file() and meta.is_file():
        summary = json.loads(meta.read_text())
        check_resources(summary["kernels_raw"])
        return so, summary
    so.parent.mkdir(parents=True, exist_ok=True)
    # Up to 8 ranks may race here: each compiles to its own file, then renames it into place atomically (identical
    # bytes, last rename wins); the resource sidecar is renamed into place BEFORE the .so (a reader that sees the .so
    # always finds the sidecar).
    tag = f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
    tmp = so.parent / f"{tag}-{_LIB_NAME}"
    tmp_meta = so.parent / f"{tag}-{_LIB_NAME}.resources.json"
    work = so.parent / f"{tag}-build"
    src = source.read_bytes()
    cmd = [_hipcc(), str(source.resolve()), *_flags(src), "-o", str(tmp.resolve())]
    try:
        work.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S, cwd=str(work))
        log = (proc.stderr or "") + (proc.stdout or "")
        if proc.returncode != 0 or not tmp.is_file():
            raise RuntimeError(f"hipcc rc={proc.returncode}: {' '.join(cmd)}: {log.strip()[-1200:]}")
        kernels = parse_resource_remarks(log)
        summary = check_resources(kernels)
        summary["kernels_raw"] = kernels
        summary["src_sha256"] = hashlib.sha256(src).hexdigest()
        tmp_meta.write_text(json.dumps(summary))
        os.replace(tmp_meta, meta)
        os.replace(tmp, so)
    finally:
        for t in (tmp, tmp_meta):
            try:
                t.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(work, ignore_errors=True)
    return so, summary


def _bind(lib: ctypes.CDLL) -> None:
    global _fn_launch, _fn_plan, _fn_dbg, _fn_drain, _impl
    abi = lib.l11_abi_version
    abi.restype = ctypes.c_int
    abi.argtypes = []
    got = int(abi())
    if got != _ABI_VERSION:
        raise RuntimeError(f"l11_abi_version()={got}, expected {_ABI_VERSION}")
    launch = lib.l11_outproj_lnmd_bwd_launch
    launch.restype = ctypes.c_int
    launch.argtypes = (
        [ctypes.c_void_p] * 4  # G8, W8, sa, sb
        + [ctypes.c_int]  # M
        + [ctypes.c_void_p] * 11  # X, U, MK, MEAN, RSTD, WN, BN, DX, DU, DWP, DBP
        + [ctypes.c_longlong]  # P
        + [ctypes.c_float, ctypes.c_float]  # dscale, invD
        + [ctypes.c_void_p]  # stream
    )
    plan = lib.l11_plan
    plan.restype = ctypes.c_int
    plan.argtypes = [ctypes.c_int, ctypes.c_void_p]
    launch_e5 = lib.l11_outproj_lnmd_bwd_e5_launch
    launch_e5.restype = ctypes.c_int
    launch_e5.argtypes = (
        [ctypes.c_void_p] * 4  # G8, W8, sa, sb
        + [ctypes.c_int]  # M
        + [ctypes.c_void_p] * 9  # X, U, MK, MEAN, RSTD, WN, BN, DX, DZ8
        + [ctypes.c_int]  # rows_out
        + [ctypes.c_void_p] * 5  # INVS, AMAXP, DWP, DBP, DUCS
        + [ctypes.c_longlong]  # P
        + [ctypes.c_float, ctypes.c_float]  # dscale, invD
        + [ctypes.c_void_p]  # stream
    )
    dbg = lib.l11_dy_debug
    dbg.restype = ctypes.c_int
    dbg.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    drain = lib.l11_drain_last_error
    drain.restype = ctypes.c_int
    drain.argtypes = []
    tag = lib.l11_impl
    tag.restype = ctypes.c_char_p
    tag.argtypes = []
    _impl = (tag() or b"").decode()
    global _fn_launch_e5
    _fn_launch, _fn_plan, _fn_dbg, _fn_drain, _fn_launch_e5 = launch, plan, dbg, drain, launch_e5


def _load() -> None:
    global _lib, _lib_path, _build_secs
    if _lib is not None:
        return
    with _lock:
        if _lib is not None:
            return
        t0 = time.perf_counter()
        so, summary = build()
        lib = ctypes.CDLL(str(so))
        _bind(lib)
        _resources.clear()
        _resources.update({k: v for k, v in summary.items() if k != "kernels_raw"})
        _build_secs = time.perf_counter() - t0
        _lib_path = so
        _lib = lib


def device_arch(index: int) -> str:
    arch = _arch_cache.get(index)
    if arch is None:
        arch = str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", "") or "").split(":")[0]
        _arch_cache[index] = arch
    return arch


def plan(M: int) -> Tuple[int, int, int, int]:
    """(G, fl, nf, P) from the .so (Python never re-derives the geometry). Needs the library loaded."""
    got = _plan_cache.get(M)
    if got is not None:
        return got
    assert _fn_plan is not None
    out = (ctypes.c_longlong * 4)()
    rc = int(_fn_plan(int(M), ctypes.addressof(out)))
    if rc != 0:
        raise RuntimeError(f"l11_plan(M={M}) rc={rc}")
    got = (int(out[0]), int(out[1]), int(out[2]), int(out[3]))
    if len(_plan_cache) < 4096:
        _plan_cache[M] = got
    return got


# ---------------------------------------------------------------------------- dispatch condition
def _a16(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def _rows512(t: torch.Tensor, dtype: torch.dtype, m: int) -> bool:
    return (
        t.dtype == dtype
        and t.dim() == 2
        and int(t.shape[0]) == m
        and int(t.shape[1]) == D
        and t.stride() == (D, 1)
        and _a16(t)
    )


def decline_reason(
    ctx: Any,
    dout: torch.Tensor,
    output_weight: torch.Tensor,
    attn: torch.Tensor,
    u: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    random_mask: Optional[torch.Tensor],
    snr_enabled: bool,
    lnmd_bwd_v2: bool,
) -> Optional[str]:
    """None = run L11 for this call; otherwise the reason (D0..D9 of the L11-2 plan, D11 not prebuilt). Host metadata
    only: no device sync, no allocation, and evaluated BEFORE the cast, so a decline never follows a cast or an
    emission. The r16 testbed's D10 (decline whenever DLRM_HSTU_FP8_BWD_DU_E5M2 is set) is gone: the backward asks
    B1a's own per-call plan after the cast and routes a B1a-emitting call to the e5m2-du mode (`run_e5`, its own
    E5 checks in `e5_decline_reason`) and every other call to the bf16-du mode (`run`), see the module docstring."""
    if not requested():
        return "D0 knob off"
    if _broken is not None:
        return f"D1 latched off: {_broken}"
    if torch.version.hip is None:
        return "D2 not a ROCm build"
    if dout.device.type != "cuda":
        return f"D2 device {dout.device}"
    index = dout.device.index if dout.device.index is not None else torch.cuda.current_device()
    arch = device_arch(index)
    if arch != _ARCH:
        return f"D2 arch {arch or '<unknown>'} is not {_ARCH}"
    if torch.compiler.is_compiling() or os.environ.get("AOTI_LOWER"):
        return "D3 compiling/AOTI (L11 is eager only)"
    if not getattr(ctx, "fp8_y_saved", False):
        return "D4 FP8 out-proj wgrad branch not active (fp8_y_saved False)"
    if snr_enabled:
        return "D5 DLRM_HSTU_FP8_BWD_OUT_WGRAD_SNR probe on (needs dy)"
    if getattr(ctx, "group_norm", False):
        return "D6 group_norm"
    if not (
        getattr(ctx, "training", False)
        and float(getattr(ctx, "dropout_ratio", 0.0)) > 0.0
        and getattr(ctx, "has_random_mask", False)
        and getattr(ctx, "silu_u", False)
        and getattr(ctx, "concat_u", False)
        and getattr(ctx, "concat_x", False)
        and getattr(ctx, "mul_u_activation_type", "") == "silu"
    ):
        return (
            "D6 flags (need training, dropout>0, saved mask, silu_u, concat_u, concat_x, 'silu'): "
            f"training={getattr(ctx, 'training', None)} p={getattr(ctx, 'dropout_ratio', None)} "
            f"mask={getattr(ctx, 'has_random_mask', None)} silu_u={getattr(ctx, 'silu_u', None)} "
            f"concat_u={getattr(ctx, 'concat_u', None)} concat_x={getattr(ctx, 'concat_x', None)} "
            f"act={getattr(ctx, 'mul_u_activation_type', None)}"
        )
    if not lnmd_bwd_v2:
        return "D7 DLRM_HSTU_FP8_LNMD_BWD_V2 off (reference consumer is v2)"
    if max(1, int(getattr(ctx, "num_warps", 0)) // 2) != 1 or int(getattr(ctx, "BLOCK_D", 0)) != D:
        return f"D7 v2 launch shape num_warps={getattr(ctx, 'num_warps', None)} BLOCK_D={getattr(ctx, 'BLOCK_D', None)}"
    if dout.dtype != torch.bfloat16 or dout.dim() != 2 or int(dout.shape[1]) != D:
        return f"D8 dout {dout.dtype} {tuple(dout.shape)}"
    m = int(dout.shape[0])
    if m < 1 or m > M_MAX:
        return f"D8 M={m} outside [1, {M_MAX}]"
    if tuple(output_weight.shape) != (NOUT, D):
        return f"D8 output_weight {tuple(output_weight.shape)}"
    if not (_rows512(attn, torch.bfloat16, m) and _rows512(u, torch.bfloat16, m)):
        return f"D8 attn/u layout {attn.dtype}{tuple(attn.stride())} {u.dtype}{tuple(u.stride())}"
    # NIB (r16 iter54; iter53 VERIFY B1): the packed keep mask (uint8 [m, 256], lnmd_mask_nib.layout_ok) passes as is,
    # never unpacked here (this gate also gates DO8, whose NIB build reads it); L11's own launches unpack it exactly
    # (lnmd_mask_nib.as_int8 in run / launch_e5 / run_e5)
    if random_mask is None or not (_rows512(random_mask, torch.int8, m) or lnmd_mask_nib.layout_ok(random_mask, m)):
        return "D8 random_mask layout"
    for name, t in (("mean", mean), ("rstd", rstd)):
        if t.dtype != torch.float32 or t.numel() != m or not t.is_contiguous():
            return f"D8 {name} {t.dtype} {tuple(t.shape)}"
    for name, t in (("norm_weight", norm_weight), ("norm_bias", norm_bias)):
        if t.dtype != torch.bfloat16 or tuple(t.shape) != (D,) or not t.is_contiguous() or not _a16(t):
            return f"D8 {name} {t.dtype} {tuple(t.shape)}"
    for name, t in (("attn", attn), ("u", u), ("mask", random_mask), ("mean", mean), ("rstd", rstd),
                    ("norm_weight", norm_weight), ("norm_bias", norm_bias)):
        if t.device != dout.device:
            return f"D8 {name} on {t.device}, dout on {dout.device}"
    if int(getattr(ctx, "du_row_width", 0)) != DU_ROW:
        return f"D9 du_row_width={getattr(ctx, 'du_row_width', None)} (kernel writes du rows of {DU_ROW})"
    if not _prebuilt:
        # Last, so a call outside the contract reports that reason first. train_ranker runs the prebuild
        # (triton_fp8.prebuild_extensions) before RUN_START and before any training step, so with the gin default
        # DLRM_HSTU_FP8_PREBUILD=1 only an opt-out reaches this.
        return "D11 not prebuilt before RUN_START (DLRM_HSTU_FP8_PREBUILD off?): no build/self-check in the timed run"
    return None


# ---------------------------------------------------------------------------- run
def dropout_scale(dropout_ratio: float) -> float:
    """v2's dropout_scale = 1.0 / (1.0 - dropout_ratio), all in fp32 (IEEE), as the Triton kernel computes it."""
    return float(np.float32(1.0) / (np.float32(1.0) - np.float32(dropout_ratio)))


def _operands_ok(g8: torch.Tensor, w8: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, m: int) -> Optional[str]:
    if not (g8.dtype == torch.float8_e5m2 and g8.dim() == 2 and tuple(g8.shape) == (m, D) and g8.stride() == (D, 1)
            and _a16(g8)):
        return f"operand G8 {g8.dtype} {tuple(g8.shape)} {g8.stride()}"
    if not (w8.dtype == torch.float8_e4m3fn and tuple(w8.shape) == (NOUT, D) and w8.stride() == (D, 1) and _a16(w8)):
        return f"operand W8 {w8.dtype} {tuple(w8.shape)} {w8.stride()}"
    for name, t in (("sa", sa), ("sb", sb)):
        if t.dtype != torch.float32 or t.numel() != 1 or t.device != g8.device:
            return f"operand {name} {t.dtype} numel={t.numel()} {t.device}"
    if w8.device != g8.device:
        return "operand W8 device"
    return None


def run(
    g8: torch.Tensor,
    w8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    attn: torch.Tensor,
    u: torch.Tensor,
    random_mask: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    dropout_ratio: float,
    alloc_du: Callable[[torch.Tensor, int], torch.Tensor],
    log_active: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Launch the fused kernel on the already-cast operands. Returns (dx, du, DWP, DBP, P) -- the caller reduces the
    [P, 512] partials with the production finalize -- or None: the caller runs torch._scaled_mm on the SAME operands
    and then v2 (latched off on any failure; FAIL_CLOSED raises instead)."""
    if not enabled():
        return None
    m = int(attn.shape[0])
    why = _operands_ok(g8, w8, sa, sb, m)
    if why is not None:
        note_decline(why)
        return None
    device = attn.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    # NIB (r16 iter54): this kernel reads the int8 [M, 512] keep mask; a NIB uint8 [M, 256] mask (only possible with
    # DLRM_HSTU_FP8_LNMD_MASK_NIB=1) is unpacked exactly here (cached on the mask, counted, one line per process);
    # int8 masks pass unchanged (no work). Before the try (review R1-F6): an unpack failure (e.g. an OOM) raises to
    # the caller instead of latching item (39)
    random_mask = lnmd_mask_nib.as_int8(random_mask, "l11_outproj_lnmd.run")
    try:
        if _lib is None:
            _load()
            _print_once(
                "lazy",
                f"{FAMILY}: library built/loaded at the first call, inside the timed region "
                f"({_build_secs:.1f}s; no prebuild before run_start)",
            )
        assert _fn_launch is not None and _fn_drain is not None
        G, fl, nf, P = plan(m)
        with torch.cuda.device(index):
            stale = int(_fn_drain())
            if stale != 0:
                _print_once(
                    f"stale:{stale}",
                    f"{FAMILY}: stale HIP error {stale} drained before the launch (left on this thread "
                    "by an earlier call, not ours)",
                )
            stream = torch.cuda.current_stream(device).cuda_stream
            dx = torch.empty_like(attn)
            du = alloc_du(u, DU_ROW)
            if du.stride() != (DU_ROW, 1) or tuple(du.shape) != (m, D) or not _a16(du):
                raise RuntimeError(f"du buffer layout {tuple(du.shape)} {du.stride()}")
            dwp = torch.empty((P, D), dtype=torch.float32, device=device)
            dbp = torch.empty((P, D), dtype=torch.float32, device=device)
            rc = int(
                _fn_launch(
                    g8.data_ptr(), w8.data_ptr(), sa.data_ptr(), sb.data_ptr(), m,
                    attn.data_ptr(), u.data_ptr(), random_mask.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
                    norm_weight.data_ptr(), norm_bias.data_ptr(), dx.data_ptr(), du.data_ptr(),
                    dwp.data_ptr(), dbp.data_ptr(), int(P), dropout_scale(dropout_ratio), 1.0 / D, stream,
                )
            )
        if rc > 0:
            _disable(f"launch rc={rc} (hipError) M={m}")
            return None
        if rc < 0:
            _print_once(
                f"stale:{-rc}",
                f"{FAMILY}: stale HIP error {-rc} drained at the launch (not ours; result used)",
            )
        global _active_logged
        if log_active and not _active_logged:
            _active_logged = True
            arm_print_all_ranks(
                f"armed OK: {FAMILY} ACTIVE ({KNOB}, item (39)): one {KERNEL_NAME} launch replaces hipBLASLt "
                f"F8B8BS NTB0 out-proj dgrad + _ln_mul_dropout_bwd_dx_du_rng_v2 (dy never in HBM); impl={_impl} "
                f"M={m} G={G} fl={fl} P={P} device=cuda:{index}"
            )
        return dx, du, dwp, dbp, P
    except Exception as error:  # noqa: BLE001 -- latch and fall back on the same operands
        _disable(f"{type(error).__name__}: {error}")
        return None


# ---------------------------------------------------------------------------- E5 mode (r16 iter32 l11e5)
E5_LIMIT: float = 57344.0  # the e5m2 clamp the kernel hard-codes (bwd_fp8._E5M2_MAX)
E5_MAX_PAD: int = 4096  # rows_out - M bound of the launch contract (B1a: uvqk_tm_pad() = 128)


def e5_decline_reason(
    x: torch.Tensor,
    u: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    silu_u: bool,
    du_row_width: int,
    grad_dtype: Optional[torch.dtype],
    limit: float,
    pad: int,
    inv_scale: torch.Tensor,
) -> Optional[str]:
    """The E5-only part of the contract (on top of decline_reason, which the B1a plan's own checks mirror): None =
    `run_e5` may replace B1a's producer for this call."""
    if grad_dtype is not torch.float8_e5m2 or float(limit) != E5_LIMIT:
        return f"E5 grad dtype {grad_dtype} limit {limit} (kernel: e5m2, {E5_LIMIT})"
    if int(BLOCK_D) != D or max(1, int(num_warps) // 2) != 1 or not silu_u:
        return f"E5 v3 launch shape BLOCK_D={BLOCK_D} num_warps={num_warps} silu_u={silu_u}"
    if int(du_row_width) != DU_ROW:
        return f"E5 du_row_width={du_row_width} (kernel writes dz8 rows of {DU_ROW})"
    if int(pad) < 1 or int(pad) > E5_MAX_PAD:
        return f"E5 pad {pad}"
    if not (inv_scale.dtype == torch.float32 and inv_scale.numel() == 1 and inv_scale.device == x.device
            and inv_scale.data_ptr() % 4 == 0):
        return f"E5 inv_scale {inv_scale.dtype} numel={inv_scale.numel()} {inv_scale.device}"
    return None


def launch_e5(
    g8: torch.Tensor,
    w8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    attn: torch.Tensor,
    u: torch.Tensor,
    random_mask: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    dropout_ratio: float,
    inv_scale: torch.Tensor,
    dz8: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """One E5 kernel launch on the current stream into the caller's dz8 [rows_out, 2048] e5m2 (u block + pad rows
    written, columns 512.. untouched). Returns (dx, DWP, DBP, DUCS, AMAXP, P); raises on a launch error (rc > 0).
    `run_e5` is the production entry; tests call this directly to keep the raw partials."""
    _load()
    assert _fn_launch_e5 is not None and _fn_drain is not None
    m = int(attn.shape[0])
    rows_out = int(dz8.shape[0])
    if not (dz8.dtype == torch.float8_e5m2 and dz8.dim() == 2 and int(dz8.shape[1]) == DU_ROW
            and dz8.stride() == (DU_ROW, 1) and _a16(dz8) and m <= rows_out < m + E5_MAX_PAD):
        raise RuntimeError(f"dz8 layout {dz8.dtype} {tuple(dz8.shape)} {dz8.stride()} (M={m})")
    # NIB (r16 iter54): the int8 [M, 512] keep mask this kernel reads (exact unpack of a NIB mask; int8 unchanged)
    random_mask = lnmd_mask_nib.as_int8(random_mask, "l11_outproj_lnmd.launch_e5")
    G, fl, nf, P = plan(m)
    device = attn.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(index):
        stale = int(_fn_drain())
        if stale != 0:
            _print_once(
                f"stale:{stale}",
                f"{FAMILY}: stale HIP error {stale} drained before the launch (left on this thread "
                "by an earlier call, not ours)",
            )
        stream = torch.cuda.current_stream(device).cuda_stream
        dx = torch.empty_like(attn)
        dwp = torch.empty((P, D), dtype=torch.float32, device=device)
        dbp = torch.empty((P, D), dtype=torch.float32, device=device)
        ducs = torch.empty((P, D), dtype=torch.float32, device=device)
        amaxp = torch.empty((G * 8,), dtype=torch.float32, device=device)
        rc = int(
            _fn_launch_e5(
                g8.data_ptr(), w8.data_ptr(), sa.data_ptr(), sb.data_ptr(), m,
                attn.data_ptr(), u.data_ptr(), random_mask.data_ptr(), mean.data_ptr(), rstd.data_ptr(),
                norm_weight.data_ptr(), norm_bias.data_ptr(), dx.data_ptr(), dz8.data_ptr(), rows_out,
                inv_scale.data_ptr(), amaxp.data_ptr(), dwp.data_ptr(), dbp.data_ptr(), ducs.data_ptr(), int(P),
                dropout_scale(dropout_ratio), 1.0 / D, stream,
            )
        )
    if rc > 0:
        raise RuntimeError(f"l11_outproj_lnmd_bwd_e5_launch rc={rc} (hipError) M={m}")
    if rc < 0:
        _print_once(
            f"stale:{-rc}",
            f"{FAMILY}: stale HIP error {-rc} drained at the launch (not ours; result used)",
        )
    return dx, dwp, dbp, ducs, amaxp, P


def dwdb_finalize(dwp: torch.Tensor, dbp: torch.Tensor, dweight: torch.Tensor, dbias: torch.Tensor, n_rows: int) -> None:
    """dweight/dbias = column sums of the [n_rows, 512] fp32 partials: triton_hstu_linear._lnmd_bwd_finalize, i.e. the
    stock `_ln_mul_dropout_bwd_dwdb` with the BLOCK_D rule of `_triton_layer_norm_mul_dropout_bwd_impl` and of B1a's
    host (the finalize the bf16-du mode's caller runs too)."""
    from generative_recommenders.ops.triton import triton_hstu_linear as thl

    thl._lnmd_bwd_finalize(
        dwp, dbp, dweight, dbias, n_rows, D, torch.cuda.get_device_properties(dwp.device).multi_processor_count
    )


def run_e5(
    g8: torch.Tensor,
    w8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    dropout_ratio: float,
    silu_u: bool,
    random_mask: torch.Tensor,
    du_row_width: int,
    site_key: int,
    inv_scale: torch.Tensor,
    cast_scale: torch.Tensor,
    generation: Optional[Tuple[object, int]],
    log_active: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]]:
    """The fused E5 replacement of `torch._scaled_mm(G8, W8.t(), sa, sb)` + B1a's
    `_triton_layer_norm_mul_dropout_bwd_du_e5m2(dy, ...)` (same arguments with the dgrad operands instead of dy,
    same return): (dx, placeholder du, dweight, dbias, DuE5m2Emit) with the record's dz8 [Mp, 2048] (u block + pad
    rows written), u_amax = per-wave partials .amax(), u_colsum = [P, 512] partials .sum(0). Production caller:
    triton_hstu_linear._l11_du_e5m2_or_none (the seam's hook, only after decline_reason passed and B1a's plan
    returned). None = declined (one NOT armed line per reason) or failed (latched off for the process, DISABLED
    line; FAIL_CLOSED raises): the caller runs `_scaled_mm` on the SAME operands and then B1a's producer. Writes
    only its own fresh buffers; reads no FP8 scale state beyond the inv_scale it is given."""
    if not enabled():
        return None
    m = int(x.shape[0])
    why = _operands_ok(g8, w8, sa, sb, m)
    if why is None:
        # NIB (r16 iter54): L11's E5 launch reads the int8 keep mask; a NIB mask (DO8 declined or latched for this
        # call) is unpacked exactly here (counted, one line per process); int8 masks pass unchanged. Before the try
        # (review R1-F6): an unpack failure (e.g. an OOM) raises to the caller instead of latching item (39)
        random_mask = lnmd_mask_nib.as_int8(random_mask, "l11_outproj_lnmd.run_e5")
    try:
        from generative_recommenders.ops.triton import bwd_fp8, hstu_du_e5m2

        grad_dtype, limit = bwd_fp8.grad_dtype_and_limit()
        pad = int(bwd_fp8.uvqk_tm_pad())
        if why is None:
            why = e5_decline_reason(x, u, BLOCK_D, num_warps, silu_u, du_row_width, grad_dtype, limit, pad, inv_scale)
        if why is not None:
            note_decline(why)
            return None
        if _lib is None:
            _load()
            _print_once(
                "lazy",
                f"{FAMILY}: library built/loaded at the first call, inside the timed region "
                f"({_build_secs:.1f}s; no prebuild before run_start)",
            )
        rows_out = (m + pad - 1) // pad * pad
        dz8 = torch.empty((rows_out, du_row_width), dtype=grad_dtype, device=x.device)
        dx, dwp, dbp, ducs, amaxp, P = launch_e5(
            g8, w8, sa, sb, x, u, random_mask, mean, rstd, weight, bias, dropout_ratio, inv_scale, dz8
        )
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dwdb_finalize(dwp, dbp, dweight, dbias, P)
        du = hstu_du_e5m2.placeholder_du(u, m, D)
        emit = hstu_du_e5m2.DuE5m2Emit(
            dz8=dz8,
            u_cols=(0, D),
            u_amax=amaxp.amax(),
            u_colsum=ducs.sum(dim=0),
            inv_scale=inv_scale,
            cast_scale=cast_scale,
            rows=m,
            width=du_row_width,
            site_key=site_key,
            generation=generation,
            buffer_ptr=0,  # no d(uvqk) buffer handed over (placeholder du), as B1a
        )
    except Exception as error:  # noqa: BLE001 -- latch and fall back on the same operands
        _disable(f"e5 mode: {type(error).__name__}: {str(error)[:600]}")
        return None
    global _active_e5_logged
    if log_active and not _active_e5_logged:
        _active_e5_logged = True
        arm_print_all_ranks(
            f"armed OK: {FAMILY} ACTIVE mode=e5 (B1a contract) ({KNOB}, item (39), with item (38)(a) "
            f"DLRM_HSTU_FP8_BWD_DU_E5M2): one {KERNEL_NAME} E5 launch replaces hipBLASLt F8B8BS NTB0 out-proj dgrad + "
            f"_ln_mul_dropout_bwd_dx_du_rng_v3 (B1a EMIT: e5m2 u block + pad rows of dz8, amax/colsum partials; no "
            f"bf16 du, dy never in HBM); impl={_impl} M={m} Mp={rows_out} P={P} device={x.device}"
        )
    return dx, du, dweight, dbias, emit


def dy_debug(g8: torch.Tensor, w8: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """TESTS ONLY: dy [M, 1536] bf16 from the same mainloop + epilogue (l11_dy_debug)."""
    _load()
    assert _fn_dbg is not None
    m = int(g8.shape[0])
    c = torch.empty((m, NOUT), dtype=torch.bfloat16, device=g8.device)
    rc = int(_fn_dbg(g8.data_ptr(), w8.data_ptr(), sa.data_ptr(), sb.data_ptr(), m, c.data_ptr(),
                     torch.cuda.current_stream(g8.device).cuda_stream))
    if rc > 0:
        raise RuntimeError(f"l11_dy_debug rc={rc}")
    return c


_DEVICE_FAULT = re.compile(
    r"memory access fault|illegal (memory|address|instruction)|unspecified launch failure|device-side assert|"
    r"hipErrorLaunchFailure|hipErrorIllegalAddress|\bECC\b|GPU core dump|HSA_STATUS_ERROR"
    # r16 C3-2 (ship17 MX6; r14 v1.2 open item 2, option (b)): the device-fatal hipError NAMES of rc 53 / 214 / 702 /
    # 709 / 710 (709 = hipErrorContextIsDestroyed, added at r14's request), which r14's outproj_dual_mx6 loader prints
    # (\bECC\b cannot match inside the CamelCase name). Only a text that carries such a name WITHOUT a device-side
    # assertion sentence changes (latch -> abort before RUN_START): r16's launchers raise bare rc numbers, Triton
    # reports the hipGetErrorString description only, and torch's C10_HIP_CHECK text, which does name the hipError,
    # already matched through its "Device-side assert..." sentence. Every shipped path classifies exactly as before.
    r"|hipErrorPriorLaunchFailure|hipErrorECCNotCorrectable|hipErrorLaunchTimeOut|hipErrorContextIsDestroyed"
    r"|hipErrorAssert",
    re.I,
)


def prebuild(
    self_check: Callable[[], Tuple[bool, str]],
    static_ok: bool,
) -> bool:
    """Before RUN_START (train_ranker -> triton_fp8.prebuild_extensions -> triton_hstu_linear.prebuild_l11): arch
    check, build + load (0 VGPR spill / 0 scratch), then `self_check()` (synthetic tensors vs the production chain).
    Knob off: no-op, no line. Not static_ok (DLRM_HSTU_FP8_BWD / _BWD_OUT_WGRAD off): INERT line. Any failure of the
    arch check or the hipcc build / load always latches off (DISABLED line; a hipcc or clang crash log never aborts
    the run). Only an exception raised by the self-check's GPU work (`self_check()` or the synchronize after it)
    whose text matches _DEVICE_FAULT re-raises (aborts before RUN_START rather than inside the timed run); any other
    self-check failure latches off."""
    global _prebuilt
    if not requested():
        return False
    if not static_ok:
        _print_once(
            "inert",
            f"{FAMILY} INERT without DLRM_HSTU_FP8_BWD=1 and DLRM_HSTU_FP8_BWD_OUT_WGRAD=1",
        )
        return False
    if _broken is not None:
        return False
    t0 = time.perf_counter()
    try:
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("no ROCm GPU")
        index = torch.cuda.current_device()
        arch = device_arch(index)
        if arch != _ARCH:
            raise RuntimeError(f"arch {arch or '<unknown>'} is not {_ARCH}")
        _load()
    except Exception as error:  # noqa: BLE001 -- arch / hipcc build / load: never a device fault, always latch
        _disable(f"prebuild build/load: {type(error).__name__}: {str(error)[:600]}")
        return False
    try:
        ok, detail = self_check()
        torch.cuda.synchronize()
    except Exception as error:  # noqa: BLE001
        msg = f"{type(error).__name__}: {error}"
        if _DEVICE_FAULT.search(msg):
            arm_print_all_ranks(f"{FAMILY} prebuild DEVICE FAULT: {msg[:600]}")
            raise
        _disable(f"prebuild self-check: {msg[:600]}")
        return False
    if not ok:
        _disable(f"prebuild self-check mismatch: {detail}")
        return False
    _prebuilt = True
    so_key = _lib_path.parent.name if _lib_path is not None else "?"
    arm_print_all_ranks(
        f"armed OK: L11 prebuild before run_start: built+loaded+self-check OK {time.perf_counter() - t0:.1f}s "
        f"(VGPR {_resources.get('vgpr')}, spill {_resources.get('vgpr_spill')}, scratch {_resources.get('scratch')}; "
        f"e5 VGPR {_resources.get('e5_vgpr')}, spill {_resources.get('e5_vgpr_spill')}, scratch "
        f"{_resources.get('e5_scratch')}; so {so_key}; {detail})"
    )
    return True

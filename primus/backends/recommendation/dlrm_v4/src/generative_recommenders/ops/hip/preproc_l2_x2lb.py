# pyre-unsafe
"""X2L-B (gin item (93)): E-M3a's FP8 L2 dgrad and pair 0's SwishLayerNorm(256) backward as ONE persistent HIP
kernel per FP8 preprocessor MLP (x2lb_pass.cu); the bf16 da1 [M, 256] is never written to HBM.

DEFAULT OFF. Gate: DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1 (1/true/on/yes; gin
apply_env_bootstrap.DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD, a pre-set env var wins; forwarded by launch_slurm.sh's
DLRM_HSTU_FP8_ prefix), read once per process. "0" is the kill switch. It acts only while E-M3a
(DLRM_HSTU_FP8_PREPROC_L2, ops/preproc_fp8_l2.py) runs its FP8 steady state and PR #7
(DLRM_HSTU_FP8_WLN_BWD_FLYDSL, ops/flydsl/ln_bwd.py) owns pair 0's SwishLN(256) backward, and on a device only after
this module's two-phase pre-RUN_START prebuild passed there. Knob off: ops/preproc_fp8_l2.py never imports this
module and runs today's statements.

WHAT. For k = 1 (additional), then k = 0 (content), _GroupedLinearLNTailFP8Function.backward replaces its FP8 dgrad
statement da1 = _scaled_mm(dh8, w8_t.t(), dh_scale, w_scale, bf16) by try_fused():
  x2lb_pass<0, 0> (S1): da1 tile = RNE_bf16(acc * (dh_scale * w_scale) + 0), acc = the in-order 4-step K = 128 MFMA
      chain over e5m2 dh8 [M, 512] x e4m3 w8_t [256, 512] (hipBLASLt F8B8BS's arithmetic), kept in LDS, then PR #7's
      ln_bwd_flydsl_d256_swish_dxsum_x16_c3 per-row sequence on it with pair 0's own saved h1, norm0 w / b, mean1 and
      rstd1 -> dh1 bf16 [M, 256] + PART fp32 [3, G, 256] (dW, dB, dXS), G = x2lb_plan(M, #CU) = min(ceil(M/64), #CU);
  PR #7's finalize ln_bwd_flydsl_fin_d256_n3_wf32 over PART (FIX R14; the same flyc.jit specialization PR #7 uses and
      prebuilds) -> dnorm_w, dnorm_b, db1 (the L1 bias gradient by the F1 rule) fp32 [256].
The results wait in the X2LBHandle; the proxy's gradient edge carries a SENTINEL instead of da1: a per-device bf16
zero scalar made at arm time, expanded to [M, 256] (2 bytes; no per-call allocation or fill). Pair 0's backward
(_LinearSwishLNEmitFunction) takes (dh1, dW, dB, db1) from the handle iff its dy IS that sentinel and skips its
SwishLN backward; E-X2b's port route, X2H, the IX wgrad, the ledger and the returned gradient tuple are unchanged.

PLUMBING (DESIGN_stageC section 3).
  publish  (pair 0's forward, after save_for_backward): X2LBHandle(weakref(ctx), stream, k) into ctx.x2lb and the
           Slot. Silent, no handle: not train, fx / compile tracing, saved-tensor default hooks (activation
           checkpointing), inside a backward (reentrant recompute), k not in _MLPS, device not armed (one NOT armed /
           IGNORED line).
  attach   (the tail's forward, single use like the FP8 payload): the proxy x_k is recorded by weakref; a stream or
           device different from pair 0's declines.
  try_fused (the tail's backward): call_reason() (pure; launches nothing) declines a create_graph backward, anomaly
           mode, pair 0's ctx gone, a proxy that is still referenced (capture, retain_grad, a kept hook), will_run()
           false (pair 0 not executed or only captured: _will_engine_execute_node(pair 0) AND the same query on
           norm0.weight's AccumulateGrad, pair 0's next_functions[3], True without raising; it raises inside every
           torch.autograd.grad), CUDA graph capture and a stranded earlier result. Then pair 0's saved tensors are read
           through p0.saved_tensors (NOT inside a try: a version-counter or freed-buffer error raises exactly as in
           pair 0) and tensor_reason() checks shapes / dtypes / contiguity / 16-B alignment / M in [65,536, 4,200,000]
           / scales, and that PR #7 would own the shipped call (TLN.flydsl_ln_bwd_decline_reason): on a Triton-LN
           fallback X2L-B would change numerics, so it declines. Every decline runs today's statements for that k and
           logs one DECLINED line per reason class (digits normalized).
  take     (pair 0's backward): the results iff dy is the handle's sentinel; a sentinel without results, results
           with a non-sentinel dy, or a sentinel whose storage was written in place (its version counter moved)
           raises "[preproc-l2-x2lb] handshake: ..." (never a silent zero, a silently ignored replacement or a double
           LN backward). Only a gradient ON a sentinel storage (a per-device _ZERO scalar) counts as a sentinel: any
           other stride-0 gradient without results runs the shipped statements.
  TASK IDENTITY (review): a fused result belongs to the autograd graph task that made it
           (torch._C._current_graph_task_id()). Another task (a backward started from a hook while this one is
           pending, or a retain_graph retry after an aborted one) declines without touching it ("a fused result of
           another backward is pending") and runs the shipped statements for its own pair 0, which take() lets
           through; the owning task still consumes it. If the tail's backward raises, its own fused results are
           dropped before the exception propagates (release()).
  Unsupported, documented (the DO8 precedent; absent from production, where the proxy is private to GroupedTail and
  dead before the backward): a tensor hook or retain_grad registered on the proxy and then the proxy dropped, a node
  pre-hook on pair 0's grad_fn, a GradientEdge built from the proxy and kept after the proxy is dropped. These see
  the zero sentinel; one that writes it in place (fill_ / zero_) raises a handshake error and latches X2L-B off.
  A result stranded by a backward that aborts outside the tail (after the tail fused, before pair 0 ran) stays held
  by the graph until the graph is freed; later backwards of that graph decline and run the shipped statements.

NUMERICS: fp32 REDUCTION-ORDER CLASS (DECLARED; no precision change; formats, scales and the bf16 rounding of da1
are unchanged). dh1 is bitwise today's on every row hipBLASLt computes unsplit; on rows in its split stream-K tiles
it is fp32-order (one bf16 ulp class of da1 per element, propagated through the swish row). dW, dB (norm0) and db1
are fp32-order (X2L-B's partition + PR #7's finalize: 1.85 / 1.78 / 2.80 x PR #7's fp64 error at M 2,492,120, CPU,
FIX 5). On a step with any split row, every dW1 element and the EC dX (row reductions over dh1) are fp32-order
too. Every other gradient is bitwise. Deterministic at fixed (M, #CU). No run-time non-finite guard: X2L-B and the
shipped chain give NaN / inf at the same dh1 positions and the same non-finite finals columns (lab gate).
MLPerf: the same FLOPs; da1 is kept in LDS instead of HBM.

BUILD (phase 1 only; never after RUN_START): a port of the lab kit's fail-closed x2lb.py build / load with X2H's
hardening. hipcc (DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD_HIPCC, else $ROCM_PATH/bin/hipcc, else /opt/rocm/bin/hipcc)
compiles x2lb_pass.cu with the kit's (= L11's production) flags verbatim, including -mllvm
-amdgpu-sched-strategy=iterative-ilp and --save-temps (cwd = a private build directory) and
-DX2LB_SRC_SHA256="<sha256 of the source>", into /tmp/dlrm_hstu_preproc_l2_x2lb/<sha16 of source + flags + arch +
`hipcc --version` + the HIPCC_* / HIP_CLANG_PATH / HIP_DEVICE_LIB_PATH env>/libx2lb_pass.so
(DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD_CACHE overrides the root). Racing ranks build to unique temporary files and
os.replace them; an fcntl lock per cache entry lets one rank per node build and gate while the others wait and hit
the cache. The cache root is created 0700 (scripts/launch_slurm.sh runs the workers under umask 000, and
Path.mkdir(parents=True) ignores mode for parents); an existing root of ours that is group / world writable is
tightened to drop those bits. A cache directory that is not ours or is still group / world writable is not trusted
(private mkdtemp build, removed once the library is loaded).
A cached entry that fails to load or its ABI check is deleted and rebuilt once. A build past the timeout kills
hipcc's whole process group. Gates (fail closed, each latches X2L-B off with one FAILED CLOSED line):
  - census (-Rpass-analysis=kernel-resource-usage remarks): every x2lb_pass instance 0 VGPR / SGPR spill, 0 scratch,
    occupancy 2 waves/SIMD, <= 248 unified VGPRs, 0 < LDS <= 163,840 B; x2lb_pass<0,0> LDS == 99,328 B; x2lb_fin
    0 spill, 0 scratch;
  - ISA gate: x2lb_isa/isa_check.py C1-C18 (incl. the C15 dataflow and the C16 / C17 concrete execution;
    isa_dataflow.py, isa_exec.py, emu/x2lb_emu.py; numpy only; verbatim copies of the sealed lab kit, sha256-pinned
    below) on the device assembly THIS toolchain produced, in a child process (its sys.path and bytecode settings
    stay out of the trainer). The verdict is stored with the cache entry and re-validated on a hit. PINNED VERDICT:
    when the device assembly, without its per-compile __hip_cuid_ lines, has exactly the sha256 of the assembly whose
    C1-C18 PASS the lab recorded (job 70428, ROCm 7.2.1), or of the ROCm 10.2 assembly whose full C1-C18 PASS was
    recorded with that toolchain (PINNED_ASM_NORM_SHA256_ROCM102), C15-C17 are not re-run (the verdict is a function
    of those bytes); C1-C14 and C18 always run. ROCm 10.2 (clang 24): C15 accepts one more form of x2lb_pass<1,0>'s
    readlane lanes 16u + 2w (+1), ((x >> 5) & 14) | 16u and (x >> 5) | 1 with x = readfirstlane(v0), equal to ROCm
    7.2.1's ((x >> 6) << 1) + 16u (+1) for every x < 512, i.e. a <= 512-thread workgroup (the extension in
    _ISA_CHILD; the kit files stay verbatim);
  - shipped == gated (ROCm 10.2 only): clang 24 generates the device code inside the LTO device link and --save-temps
    writes the gated .lto.s from a second codegen run, so the build assembles and links that .s and requires the
    .so's gfx950 code object to have the same .text, .rodata, .note, e_flags and dynamic symbols (shipped_vs_gated;
    at build time, stored with the cache entry);
  - ABI: x2lb_abi_version() == 2026101202, x2lb_variant() == 0 (not a lab negative control), x2lb_impl() carries
    this source's sha256.
The .so's gfx950 code object .text / .rodata sha256 is printed next to the lab's (informational at run time; it
decides the UNIT gate's G-a scope).

PREBUILD (two phases, both before RUN_START; MLPerf: library initialization and JIT on synthetic inputs):
  phase 1  dlrm_v4/train/blas_warmup.run_blas_warmup -> _prebuild_preproc_l2_x2lb_build: the build + gates above,
           before the DataLoader pre-start forks; no FlyDSL import.
  phase 2  blas_warmup._prebuild_flydsl_ln_bwd (startup preload, after the FlyDSL prebuild and E-M3a's grp3f8
           block) -> prebuild_check: a self-check against the shipped chain on sentinel-filled synthetic data (no
           RNG draw: integer hashes; no _Site, module or parameter), at M 2,356,560 (hipBLASLt solution A,
           data-parallel grid: torch's da1 gated on ALL rows), 2,424,832 (A, grid 256) and 2,492,120 (B, grid 256;
           both gated on rows < M - 131,072, the data-parallel region for any grid <= 256), the "cancel"
           construction (K step 1 = -(K step 0) bytes, large codes there, small codes in steps 2-3, scales 2^-20 /
           2^-10: an operand K-order change shows), plus finals at M 16,383:
             (a) S1+DA1 da1 == triton_fp8._scaled_mm(dh8, w8_t.t(), ...) (the production callable);
             (b) S1+DA1 dh1 == PR #7 ln_bwd(dy = that da1, x = h1, ..., is_swish, sum_dx) on all rows (ln_bwd is
                 called directly: an error here latches X2L-B only, never PR #7);
             (c) S1 dh1 / PART == S1+DA1 dh1 / PART; (d) S1 at grid 7 gives the same dh1;
             (e) canary rows >= M untouched, every PART element and every finals element written;
             (f) at M 16,383 (a partial last tile): the finals within gamma_h x sum|term| of a torch fp64 oracle.
           Order and frees keep the transient at <= 5 U (U = M x 512 B; 6.38 GB at M 2,492,120) behind a free-memory
           headroom check (5 U + 1 GiB, else FAILED CLOSED (phase-2 headroom)); blas_warmup then releases it
           (torch.cuda.empty_cache) before alloc_pregrow. On success the device's sentinel scalar is made and the
           device is armed.
  Phase 2 is reached only in streaming-train-eval mode with the startup preload on ($BLAS_WARMUP), the DataLoader
  pre-started (item (18)), the FlyDSL preload and PR #7 on, gfx950, and E-M3a on. Otherwise the first publish prints
  "[preproc-l2-x2lb] DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1 IGNORED (NOT armed): phase-2 self-check not run (<why>)"
  and the run keeps the shipped path.

FALLBACKS (every one runs today's dgrad + PR #7 SwishLN backward for that k): knob off; E-M3a or PR #7 off (IGNORED);
not armed (NOT armed / IGNORED); a per-call decline (DECLINED, once per reason class); any host-side error in the
call path, an ABI rc > 0, or a failed prebuild phase: X2L-B latched off for the process (FAILED CLOSED), inputs
intact, the shipped statements run for this and every later call; this module failing to import (ops/preproc_fp8_l2.py
prints "[preproc-l2-x2lb] FAILED CLOSED (import)" once and turns the knob off for the process). Diagnostic env knobs,
never set in a timed run:
DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD_HIPCC, _CACHE, _MLPS ("0,1" default; "0" or "1": which MLPs publish),
_FIN ("pr7" default; "x2lb": the kernel's own sequential x2lb_fin).

ARMED (every rank, once, tagged "[rank N]"):
  "armed OK: [preproc-l2-x2lb] pre-run_start build OK (..."
  "armed OK: [preproc-l2-x2lb] pre-run_start self-check OK (..."
  "armed OK: [preproc-l2-x2lb] fused L2 dgrad + SwishLN256 bwd on cuda:N ..."  (first fused backward)
Must be ABSENT: "[preproc-l2-x2lb] FAILED CLOSED", "[preproc-l2-x2lb] DECLINED", "[preproc-l2-x2lb] NOT armed",
"[preproc-l2-x2lb] DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1 IGNORED", "[preproc-l2-x2lb] handshake".
Informational: "[preproc-l2-x2lb] stale HIP error ... drained" (an earlier call's error left on the thread).

OWNERSHIP: actckpt (item (93) X2L-B). Imported lazily: by ops/preproc_fp8_l2.py only when a handle exists (knob on)
and by blas_warmup's hooks only with the knob on. Importing it builds, loads and launches nothing.
"""

import ctypes
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from generative_recommenders.ops.utils import arm_print_all_ranks

ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD"
HIPCC_ENV_VAR: str = ENV_VAR + "_HIPCC"
CACHE_ENV_VAR: str = ENV_VAR + "_CACHE"
MLPS_ENV_VAR: str = ENV_VAR + "_MLPS"  # diagnostic: "0,1" (default) | "0" | "1"
FIN_ENV_VAR: str = ENV_VAR + "_FIN"  # diagnostic: "pr7" (default) | "x2lb"
TAG: str = "[preproc-l2-x2lb]"
ARMED_PREFIX: str = "armed OK: " + TAG
PASS_NAME: str = "x2lb_pass"
PR7_FIN_NAME: str = "ln_bwd_flydsl_fin_d256_n3_wf32"
SHIPPED: str = "shipped dgrad + SwishLN bwd"

ABI_VERSION: int = 2026101202  # x2lb_pass.cu X2LB_ABI_VERSION
VARIANT: int = 0  # x2lb_variant() of the production kernel (no lab negative-control bit)
# x2lb_pass.cu as sealed by the lab kit (PRED_x2lb.md section 6; kernel-lab job 70428, all correctness gates PASS).
# ops/tests pins the file to it: an edited kernel needs the GPU gates again.
SOURCE_SHA256: str = "a78f4aa3611f6b6211d6f197848cc40661572390c35c6d78853d3064e5eebb55"
# The lab container build (ROCm 7.2.1 hipcc, job 70428, correct/x2lb_cache/1c53548919fa9d49): the device assembly
# without its __hip_cuid_ lines (C1-C18 PASS on all four kernels) and the .so's gfx950 code object sections.
PINNED_ASM_NORM_SHA256: str = "57de20b6ef509eba1429f19a9486e1475c390a2cf9b851f6bda6683960511e50"
# ROCm 10.2 (TheRock 10.2.0a20261015, AMD clang 24.0.0git bdc35841c97f): the gated .lto.s of the same source and flags,
# same normalization (path independent), whose full C1-C18 PASS (C15 with the ROCm 10.2 lane-index extension of
# _ISA_CHILD) was recorded with that toolchain (docs/04-technical-guides/recommendation/dlrm_v4/rocm-port.md T15). Its code object .text is 3ccf287e1e64...
PINNED_ASM_NORM_SHA256_ROCM102: str = "e8726541453c979671b1e4520a1209589e62c52b7a7fafada259e9628eb4668e"
LAB_CODE_OBJECT_SHA256: Dict[str, str] = {
    ".text": "1873c0cbded354588b8f08a384036bb938b63bd7d5d766aebde586472ee39334",
    ".rodata": "c5c7b8872343f81069d9685890e98561bdf0547d78759934a1e54e6ad7cbb484",
}
# The ISA gate's files, verbatim from the sealed kit (PRED_x2lb.md section 6); refused if they differ.
ISA_FILES_SHA256: Dict[str, str] = {
    "isa_check.py": "7680f8138aa0648ccd4ef385779ce3058406592d16870840b4532e593ffec7f2",
    "isa_dataflow.py": "08e7789e15300da4b64aa6ea3b377c944a2ba882c11dd62a060b19280d613384",
    "isa_exec.py": "fd2121ed32c6c2ea2dc25cb7bf3a0ad3f85baca2744537f076617b335df39641",
    "emu/x2lb_emu.py": "468a9abf44fed5f23c8841e2bd9f022a506fde867f0459685d2e2ce900beec1e",
}

HID: int = 256  # norm width = dgrad N
KD: int = 512  # dgrad K (the LN512 width)
BM: int = 64  # x2lb_pass rows per tile
M_MIN: int = 1 << 16  # the FP8 path's own minimum (preproc_fp8_l2._MIN_ROWS)
M_MAX: int = 4_200_000  # the lab's largest verified M (PRED_x2lb G2a / G4); kernel limit 2^29
LIM_VGPR: int = 248
LIM_LDS: int = 163_840
LDS_S1: int = 99_328
STRESS_GRID: int = 7
# phase-2 check points (DESIGN_stageC 5.3): (M, da1 rule)
M_DP: int = 2_356_560  # solution A, data-parallel production grid: torch's da1 gated on all rows
M_A: int = 2_424_832  # solution A, grid 256, split-free
M_B: int = 2_492_120  # solution B (MT256x240 band), grid 256: 144 split tiles
M_FIN: int = 16_383  # finals vs fp64 (a partial last tile)
SPLIT_TAIL: int = 131_072  # rows a <= 256-CTA stream-K grid can split: 511 tiles x 256
CHECK_POINTS: Tuple[Tuple[int, str], ...] = ((M_DP, "all"), (M_A, "dp_region"), (M_B, "dp_region"),
                                             (M_FIN, "report"))
CANARY_ROWS: int = 64
HEADROOM_SLACK: int = 1 << 30

KERNEL_S1: str = "_ZN4x2lb9x2lb_passILi0ELi0EEEvNS_6ParamsE"
KERNEL_EMIT: str = "_ZN4x2lb9x2lb_passILi0ELi1EEEvNS_6ParamsE"
KERNEL_S0: str = "_ZN4x2lb9x2lb_passILi1ELi0EEEvNS_6ParamsE"
KERNEL_FIN: str = "_ZN4x2lb8x2lb_finEPKfiPf"

_ARCH: str = "gfx950"
_HERE: Path = Path(__file__).resolve().parent
_SOURCE: Path = _HERE / "x2lb_pass.cu"
_ISA_DIR: Path = _HERE / "x2lb_isa"
# The lab kit's (= L11's production loader's) flags, verbatim (labkit x2lb.py _FLAGS).
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
# hipcc reads these; they change its output, so they are part of the cache key (X2H)
_HIPCC_ENV: Tuple[str, ...] = ("HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND", "HIP_CLANG_PATH",
                               "HIP_DEVICE_LIB_PATH")
_CACHE_ROOT: str = "/tmp/dlrm_hstu_preproc_l2_x2lb"
_LIB_NAME: str = "libx2lb_pass.so"
_ASM_NAME: str = "x2lb_pass.gfx950.s"
_META_NAME: str = "meta.json"
_PRIVATE_PREFIX: str = "x2lb_private_"  # mkdtemp prefix of a private (untrusted-root or rebuild) build
_BUILD_TIMEOUT_S: int = 600
_ISA_TIMEOUT_S: int = 1200

_REQUESTED: Optional[bool] = None
_MLPS: Optional[Tuple[int, ...]] = None
_FIN: Optional[str] = None
_BROKEN: Optional[str] = None
_lock = threading.Lock()
_lib: Optional["_Lib"] = None
_toolchains: Dict[str, str] = {}
_armed: Set[int] = set()  # device indices whose phase 2 passed
_ONCE: Set[Tuple[str, str]] = set()
_ZERO: Dict[int, torch.Tensor] = {}  # per-device bf16 zero scalar: the sentinel's storage (made at arm time)
_PENDING: "weakref.WeakSet[Any]" = weakref.WeakSet()  # handles holding a fused result (no strong reference)
_P1_STATE: Optional[str] = None  # None (not run) | "ok" | "ignored" | "failed"
_P2_STATE: Optional[str] = None  # None (not run) | "ok" | "failed"
_P2_SKIP: Optional[str] = None  # the first reason phase 2 was not reached (blas_warmup notes it)
_LB: Any = None  # ops/flydsl/ln_bwd (PR #7), imported by phase 2
PHASE1_INFO: Dict[str, Any] = {}
PREBUILD_INFO: Dict[int, Dict[str, Any]] = {}
FUSED_CALLS: Dict[int, int] = {}  # per MLP k (tests / diagnostics)
DECLINES: Dict[str, int] = {}  # per normalized reason (tests / diagnostics)


def _say(msg: str) -> None:
    arm_print_all_ranks(msg)


def _once(site: str, reason: str, msg: str) -> None:
    if (site, reason) not in _ONCE:
        _ONCE.add((site, reason))
        _say(msg)


def _cls(why: str) -> str:
    return re.sub(r"\d+", "N", why)


# ------------------------------------------------------------------------------------------- gate
def requested() -> bool:
    """The knob (read once per process)."""
    global _REQUESTED
    if _REQUESTED is None:
        _REQUESTED = os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "on", "yes")
    return _REQUESTED


def mlps() -> Tuple[int, ...]:
    """The MLPs that publish (DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD_MLPS, read once; default "0,1"). Another value
    raises (phase 1 then fails closed)."""
    global _MLPS
    if _MLPS is None:
        raw = os.environ.get(MLPS_ENV_VAR, "").strip()
        if raw in ("", "0,1", "1,0"):
            _MLPS = (0, 1)
        elif raw in ("0", "1"):
            _MLPS = (int(raw),)
        else:
            raise ValueError(f"{MLPS_ENV_VAR}={raw!r}: must be 0,1 (default), 0 or 1")
    return _MLPS


def fin_mode() -> str:
    """"pr7" (default: PR #7's ln_bwd_flydsl_fin_d256_n3_wf32 over PART, FIX R14) or "x2lb" (diagnostic: the
    kernel's sequential x2lb_fin). Read once; another value raises (phase 1 then fails closed)."""
    global _FIN
    if _FIN is None:
        raw = os.environ.get(FIN_ENV_VAR, "").strip().lower()
        if raw not in ("", "pr7", "x2lb"):
            raise ValueError(f"{FIN_ENV_VAR}={raw!r}: must be pr7 (default) or x2lb")
        _FIN = raw or "pr7"
    return _FIN


def broken() -> Optional[str]:
    return _BROKEN


def latch_off(where: str, exc: BaseException) -> None:
    """X2L-B off for the rest of the process; every later call runs the shipped dgrad + SwishLN backward."""
    global _BROKEN
    if _BROKEN is None:
        _BROKEN = f"{where}: {type(exc).__name__}: {str(exc)[:300]}"
        try:
            _say(f"{TAG} FAILED CLOSED -> {SHIPPED} for the rest of the run ({_BROKEN})")
        except Exception:  # pragma: no cover - a failing print must not escape into the backward
            pass


def ignored(why: str) -> None:
    """The knob is on but X2L-B cannot arm for a configuration reason (once per process)."""
    _once("ignored", _cls(why), f"{TAG} {ENV_VAR}=1 IGNORED (NOT armed): {why}; {SHIPPED} for the whole run")


def phase2_not_run(why: str) -> None:
    """blas_warmup: phase 2 was not reached for `why` (the first reason wins; printed at the first publish)."""
    global _P2_SKIP
    if _P2_SKIP is None:
        _P2_SKIP = str(why)


def _dev_index(dev: torch.device) -> int:
    if dev.index is not None:
        return int(dev.index)
    return int(torch.cuda.current_device()) if dev.type == "cuda" else -1


def active(dev: torch.device) -> bool:
    """X2L-B may run on `dev`: requested, not latched off, phase 2 passed on this device. Never raises."""
    try:
        return requested() and _BROKEN is None and _dev_index(dev) in _armed
    except Exception:  # pragma: no cover - defensive
        return False


def _not_armed_line(dev: torch.device) -> None:
    """The once-line for a publish on a device that is not armed (configuration, not a kernel failure)."""
    if _BROKEN is not None or _P1_STATE in ("ignored", "failed") or _P2_STATE == "failed":
        return  # its IGNORED / FAILED CLOSED line was printed already
    idx = _dev_index(dev)
    if _P1_STATE == "ok" and _P2_STATE is None:
        why = _P2_SKIP or ("the startup preload did not run it (needs mode streaming-train-eval with "
                           "$BLAS_WARMUP and DL_PRESTART)")
        ignored(f"phase-2 self-check not run ({why})")
        return
    _once("not_armed", str(idx), f"{TAG} NOT armed on cuda:{idx}: no pre-run_start prebuild passed on this device "
          f"(BLAS_WARMUP off?) -> {SHIPPED} (nothing is built after RUN_START)")


# ------------------------------------------------------------------------------------------ build
def _hipcc() -> str:
    override = os.environ.get(HIPCC_ENV_VAR, "").strip()
    if override:
        return override
    rocm = os.environ.get("ROCM_PATH", "").strip()
    if rocm:
        cand = os.path.join(rocm, "bin", "hipcc")
        if os.path.exists(cand):
            return cand
    return "/opt/rocm/bin/hipcc"


def _toolchain(hipcc: str) -> str:
    """`hipcc --version` (cached per hipcc path; raises if hipcc cannot run)."""
    v = _toolchains.get(hipcc)
    if v is None:
        proc = subprocess.run([hipcc, "--version"], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"{hipcc} --version rc={proc.returncode}: "
                               f"{((proc.stderr or '') + (proc.stdout or '')).strip()[-400:]}")
        v = proc.stdout
        _toolchains[hipcc] = v
    return v


def build_flags(src: bytes) -> List[str]:
    """The compile flags (labkit x2lb.py _flags for flavor "prod"): arch, the kit's _FLAGS, the source sha."""
    return [f"--offload-arch={_ARCH}", *_FLAGS, f'-DX2LB_SRC_SHA256="{hashlib.sha256(src).hexdigest()}"']


def cache_dir(source: Path = _SOURCE) -> Path:
    root = os.environ.get(CACHE_ENV_VAR, "").strip() or _CACHE_ROOT
    src = source.read_bytes()
    h = hashlib.sha256()
    h.update(src)
    h.update(b"\0" + "\0".join(build_flags(src)).encode())
    h.update(b"\0" + _ARCH.encode())
    # The toolchain too: a .so another hipcc built is not reused (census and ISA verdict hold for this build).
    h.update(b"\0" + _toolchain(_hipcc()).encode())
    h.update(b"\0" + "\0".join(f"{k}={os.environ.get(k, '')}" for k in _HIPCC_ENV).encode())
    return Path(root) / h.hexdigest()[:16]


def _trusted(p: Path) -> bool:
    """Ours and not writable by group / others (a planted .so must not be loaded)."""
    st = p.stat()
    return st.st_uid == os.getuid() and not st.st_mode & 0o022


def _ensure_root(root: Path) -> None:
    """The cache root, created 0700 whatever the umask (launch_slurm.sh runs workers under umask 000, and
    Path.mkdir(parents=True) applies `mode` to the last component only). An existing root of ours that is group /
    world writable loses those bits, so the shared cache is trusted again; a root of another uid is left alone (then
    _trusted refuses it and the build is private)."""
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError:
        pass
    else:
        os.chmod(root, 0o700)  # mkdir's mode is masked by the umask; this sets it exactly
    st = root.stat()
    if st.st_uid == os.getuid() and st.st_mode & 0o022:
        os.chmod(root, st.st_mode & 0o7755)


def _run_group(cmd: List[str], timeout: int, cwd: Optional[str] = None,
               env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run cmd in its own process group; on the timeout the whole group (clang / python children) is killed."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True, cwd=cwd, env=env)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise RuntimeError(f"timed out after {timeout}s: {' '.join(cmd)[:300]}")
    return proc.returncode, out or ""


# ROCm 10.2 form (no "remark:", no "[-Rpass-analysis=...]"): "<file>:<line>:<col>: <Key>: <value>", known keys only.
_REMARK_ROCM102 = re.compile(
    r"\S+?:\d+:\d+:\s+((?:Function Name|TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|"
    r"Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\]):\s*\S+)\s*$"
)


def parse_resource_remarks(log: str) -> List[Dict[str, str]]:
    """-Rpass-analysis=kernel-resource-usage remarks, one dict per kernel (labkit x2lb.py, L11 loader pattern)."""
    kernels: List[Dict[str, str]] = []
    rocm102: List[bool] = []  # per kernel: its "Function Name" line was in the ROCm 10.2 form
    for line in log.splitlines():
        m = re.search(r"remark:\s+\S+?:\d+:\d+:\s+(.*?)\s*\[-Rpass-analysis", line) or _REMARK_ROCM102.match(line)
        if not m:
            continue
        kv = re.match(r"\s*([^:]+):\s*(\S+)", m.group(1))
        if not kv:
            continue
        k, v = kv.group(1).strip(), kv.group(2)
        if k == "Function Name":
            kernels.append({})
            rocm102.append(m.re is _REMARK_ROCM102)
        if kernels:
            kernels[-1][k] = v
    # ROCm 10.2 --save-temps runs the LTO device codegen twice (.lto.s, .lto.o): an identical repeated block counts once
    return [k for i, k in enumerate(kernels) if not (rocm102[i] and k in kernels[:i])]


def check_resources(kernels: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """Raise unless the census passes (module docstring, BUILD): every x2lb_pass instance (all three) 0 VGPR / SGPR
    spill, 0 scratch, occupancy 2, <= 248 unified VGPRs, 0 < LDS <= 163,840 B; x2lb_pass<0,0> LDS == 99,328 B;
    x2lb_fin 0 spill, 0 scratch."""
    out: Dict[str, Dict[str, int]] = {}
    for k in kernels:
        nm = k.get("Function Name", "")
        if PASS_NAME not in nm and nm != KERNEL_FIN:
            continue
        out[nm] = dict(vgpr=int(k.get("VGPRs", "-1")), agpr=int(k.get("AGPRs", "-1")),
                       sgpr=int(k.get("TotalSGPRs", "-1")), vgpr_spill=int(k.get("VGPRs Spill", "-1")),
                       sgpr_spill=int(k.get("SGPRs Spill", "-1")),
                       scratch=int(k.get("ScratchSize [bytes/lane]", "-1")),
                       lds=int(k.get("LDS Size [bytes/block]", "-1")),
                       occupancy=int(k.get("Occupancy [waves/SIMD]", "-1")))
    for nm in (KERNEL_S1, KERNEL_EMIT, KERNEL_S0, KERNEL_FIN):
        if nm not in out:
            raise RuntimeError(f"resource census: no remarks for {nm}")
    for nm, v in out.items():
        if nm == KERNEL_FIN:
            if v["vgpr_spill"] != 0 or v["sgpr_spill"] != 0 or v["scratch"] != 0:
                raise RuntimeError(f"resource census: x2lb_fin {v} (needs 0 spill, 0 scratch)")
            continue
        unified = ((v["vgpr"] + 3) // 4) * 4 + max(v["agpr"], 0)
        v["unified_vgpr"] = unified
        if (v["vgpr"] < 0 or v["vgpr_spill"] != 0 or v["sgpr_spill"] != 0 or v["scratch"] != 0
                or unified > LIM_VGPR or v["occupancy"] != 2 or not (0 < v["lds"] <= LIM_LDS)
                or (nm == KERNEL_S1 and v["lds"] != LDS_S1)):
            raise RuntimeError(f"resource census: {nm} {v} (needs 0 spill, 0 scratch, occupancy 2, <= {LIM_VGPR} "
                               f"unified VGPR, LDS <= {LIM_LDS}; x2lb_pass<0,0> LDS == {LDS_S1})")
    return out


def asm_norm_sha256(text: bytes) -> str:
    """sha256 of a device .s without its __hip_cuid_ lines (the per-compile unique id; labkit x2lb_ab.asm_norm_sha)."""
    h = hashlib.sha256()
    for ln in text.splitlines(keepends=True):
        if b"__hip_cuid_" not in ln:
            h.update(ln)
    return h.hexdigest()


def isa_files_sha256() -> Dict[str, str]:
    return {n: hashlib.sha256((_ISA_DIR / n).read_bytes()).hexdigest() for n in ISA_FILES_SHA256}


# The ISA gate's child process: the kit's isa_check imported by name from x2lb_isa/ (its sys.path inserts and
# sys.dont_write_bytecode stay in the child). Pinned mode stubs C15 (isa_dataflow) and C16 / C17 (isa_exec).
# Full mode extends C15's integer model by exactly one ROCm 10.2 form (clang 24; the kit files stay verbatim). The kit
# models x = readfirstlane(v0) as 64w and accepts x2lb_pass<1,0>'s readlane lanes 16u + 2w (+1) in ROCm 7.2.1's form
# ((x >> 6) << 1) + 16u. clang 24, knowing x < 512 (__launch_bounds__(512)), emits t = (x >> 5) & 14, then t | 16u
# (u = 1..3), t + 16u + 1, t, and (x >> 5) | 1: equal to the 7.2.1 lanes for every x < 512, different from x = 512 on.
# Modelled exactly, nothing more: s_lshr_b32 64w, 5 = 2w + b (b = bit 5 of the first active lane's id; an opaque value
# that only the next two rules read); (2w + b) | 1 = 2w + 1; and, only in a kernel whose code object metadata caps the
# workgroup at <= 512 threads (w <= 7: 2w in {0, 2, .., 14}, bits 1-3), (2w + b) & 14 = 2w and 2w | 16 / 32 / 48 =
# 2w + 16u (bits 4-5, disjoint from 2w's, so | == +). Any other operand, constant or composition keeps the kit's
# semantics (opaque), so a different lane computation still fails C15; C17 checks the lanes concretely on 8 waves.
_ISA_CHILD = r"""
import json, os, re, sys, types
d, asm, pinned = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
sys.dont_write_bytecode = True
sys.path.insert(0, d)
if pinned:
    df = types.ModuleType("isa_dataflow")
    df.check = lambda name, lines: dict(errors=[], info=["C15 not re-run: pinned device assembly"])
    ex = types.ModuleType("isa_exec")
    ex.check = lambda *a, **k: "C16/C17 not re-run: pinned device assembly"
    sys.modules["isa_dataflow"], sys.modules["isa_exec"] = df, ex
import isa_check as IC
assert os.path.realpath(os.path.dirname(IC.__file__)) == os.path.realpath(d), IC.__file__
text = open(asm).read()
kern, desc = IC.split_asm(text)
cur = [None]
if not pinned:
    import isa_dataflow as DF
    assert os.path.realpath(os.path.dirname(DF.__file__)) == os.path.realpath(d), DF.__file__
    md = re.search(r"^\s*\.amdgpu_metadata\s*$(.*?)^\s*\.end_amdgpu_metadata", text, re.M | re.S)
    wg = {}
    for e in re.split(r"\n  - ", md.group(1) if md else ""):
        n = re.search(r"^    \.name:\s+(\S+)\s*$", e, re.M)
        f = re.search(r"^    \.max_flat_workgroup_size:\s+(\d+)\s*$", e, re.M)
        if n and f:
            wg[n.group(1)] = int(f.group(1))
    kit_s_exec = DF.s_exec
    half = DF.mk("x2lb_rocm102_2w_plus_bit5")  # (64w) >> 5 = 2w + b, b unknown

    def s_exec(m, x, line):
        o, r, a = x.ops, None, None
        if x.op in ("s_lshr_b32", "s_and_b32", "s_or_b32") and len(o) == 3:
            r, a = IC.reg(o[0]), IC.reg(o[1])
        if r and a and r[0] == a[0] == "s" and r[1] == r[2] and a[1] == a[2]:
            e, k = m.gets(a[1]), DF.parse_lit(o[2])
            w8 = 0 < wg.get(cur[0], 0) <= 512
            res = (half if x.op == "s_lshr_b32" and DF.as_lin(e) == ("lin", 64, 0) and k == 5 else
                   DF.lin(2, 0) if x.op == "s_and_b32" and w8 and e == half and k == 14 else
                   DF.lin(2, 1) if x.op == "s_or_b32" and e == half and k == 1 else
                   DF.lin(2, k) if (x.op == "s_or_b32" and w8 and DF.as_lin(e) == ("lin", 2, 0)
                                    and k in (16, 32, 48)) else None)
            if res is not None:
                m.s[r[1]] = res
                return
        kit_s_exec(m, x, line)

    DF.s_exec = s_exec
per = {}
for name in list(IC.KERNELS) + [IC.FIN]:
    cur[0] = name
    if name not in kern:
        per[name] = dict(ok=False, failed=["missing"], errors=["kernel not found"])
        continue
    try:
        r = IC.check_fin(kern[name], desc) if name == IC.FIN else IC.check_kernel(name, kern[name], desc)
    except Exception as e:
        per[name] = dict(ok=False, failed=["raised"], errors=[f"{type(e).__name__}: {e}"[:300]])
        continue
    per[name] = dict(label=r["kernel"], ok=bool(r["ok"]), failed=sorted({e.split()[0] for e in r["errors"]}),
                     errors=r["errors"][:6])
print("X2LB_ISA_VERDICT " + json.dumps(per), flush=True)
"""


def isa_gate(asm_path: Path) -> Dict[str, Any]:
    """isa_check C1-C18 on one device assembly (child process). Returns the verdict dict; `ok` False on any
    failure, a missing kernel, a crash or a timeout (fail closed). Pinned assembly: C15-C17 not re-run."""
    t0 = time.perf_counter()
    got = isa_files_sha256()
    if got != ISA_FILES_SHA256:
        bad = sorted(n for n in ISA_FILES_SHA256 if got.get(n) != ISA_FILES_SHA256[n])
        raise RuntimeError(f"ISA gate files differ from the sealed kit: {bad}")
    norm = asm_norm_sha256(asm_path.read_bytes())
    pinned = norm in (PINNED_ASM_NORM_SHA256, PINNED_ASM_NORM_SHA256_ROCM102)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    rc, out = _run_group([sys.executable, "-B", "-c", _ISA_CHILD, str(_ISA_DIR), str(asm_path), "1" if pinned else "0"],
                         _ISA_TIMEOUT_S, env=env)
    line = [ln for ln in out.splitlines() if ln.startswith("X2LB_ISA_VERDICT ")]
    per: Dict[str, Any] = json.loads(line[-1][len("X2LB_ISA_VERDICT "):]) if (line and rc == 0) else {}
    want = (KERNEL_S1, KERNEL_EMIT, KERNEL_S0, KERNEL_FIN)
    ok = bool(per) and all(per.get(k, {}).get("ok") is True for k in want)
    return dict(ok=ok, mode="pinned" if pinned else "full", asm_norm_sha256=norm, kernels=per, rc=rc,
                checker=got, seconds=round(time.perf_counter() - t0, 1),
                tail=None if ok else out.strip()[-1500:])


def _tool(hipcc: str, name: str) -> Optional[str]:
    root = os.path.dirname(os.path.dirname(os.path.realpath(hipcc)))
    for c in (os.path.join(root, "lib", "llvm", "bin", name), os.path.join(root, "llvm", "bin", name),
              os.path.join("/opt/rocm/llvm/bin", name), os.path.join("/opt/rocm/lib/llvm/bin", name)):
        if os.path.isfile(c):
            return c
    return shutil.which(name)


def _code_object(so: Path, objcopy: str, bundler: str, d: str) -> Optional[str]:
    """The .so's gfx950 code object unbundled into d (labkit x2lb_ab.provenance), or None without a gfx950 entry.
    Raises if a step fails."""
    fat, co, so_out = os.path.join(d, "fat.bin"), os.path.join(d, "co.o"), os.path.join(d, "so_out.so")
    subprocess.run([objcopy, "--dump-section", f".hip_fatbin={fat}", str(so), so_out], check=True,
                   capture_output=True, timeout=120)
    lst = subprocess.run([bundler, "--list", "--type=o", f"--input={fat}"], capture_output=True, text=True,
                         timeout=120).stdout.split()
    tgt = next((x for x in lst if x.endswith(_ARCH)), None)
    if tgt is None:
        return None
    subprocess.run([bundler, "--unbundle", "--type=o", f"--targets={tgt}", f"--input={fat}", f"--output={co}"],
                   check=True, capture_output=True, timeout=120)
    return co


def _section_sha256(objcopy: str, obj: str, sec: str, d: str) -> str:
    out = os.path.join(d, "sec" + sec)
    subprocess.run([objcopy, "-O", "binary", f"--only-section={sec}", obj, out], check=True, capture_output=True,
                   timeout=120)
    return hashlib.sha256(Path(out).read_bytes()).hexdigest()


def code_object_sha256(so: Path, hipcc: str) -> Dict[str, str]:
    """sha256 of the .so's gfx950 code object .text / .rodata (labkit x2lb_ab.provenance). Never raises:
    {"status": "unknown (...)"} when a tool is missing or a step fails."""
    objcopy, bundler = _tool(hipcc, "llvm-objcopy"), _tool(hipcc, "clang-offload-bundler")
    if not (objcopy and bundler):
        return {"status": "unknown (llvm-objcopy / clang-offload-bundler not found)"}
    d = tempfile.mkdtemp(prefix="x2lb_co_")
    try:
        co = _code_object(so, objcopy, bundler, d)
        if co is None:
            return {"status": f"unknown (no {_ARCH} bundle entry)"}
        res = {"status": "ok"}
        for sec in (".text", ".rodata"):
            res[sec] = _section_sha256(objcopy, co, sec, d)
        res["lab"] = "identical" if all(res[s] == LAB_CODE_OBJECT_SHA256[s] for s in LAB_CODE_OBJECT_SHA256) \
            else "differs"
        return res
    except Exception as exc:
        return {"status": f"unknown ({type(exc).__name__}: {str(exc)[:160]})"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def shipped_vs_gated(so: Path, asm: Path, hipcc: str) -> Dict[str, Any]:
    """ROCm 10.2 (clang 24's offload driver) generates the device code inside the LTO device link, and --save-temps
    writes the gated .lto.s from a SECOND codegen run of the same bitcode, so the ISA gate reads a twin of what ships.
    This assembles and links that .s (llvm-mc, ld.lld -shared) and requires the .so's gfx950 code object to have the
    same .text, .rodata (kernel descriptors), .note (code object metadata), ELF e_flags (target ID) and dynamic
    symbols (name, value, size, type, binding, visibility, section; .dynstr itself differs only in string order).
    Raises on a difference or a failed step (fail closed). ROCm 7.2.1 does not need it: its driver assembles the very
    .s it saves."""
    tl = {n: _tool(hipcc, n) for n in ("llvm-mc", "ld.lld", "llvm-objcopy", "clang-offload-bundler", "llvm-readelf")}
    if not all(tl.values()):
        raise RuntimeError(f"shipped vs gated: {sorted(n for n, p in tl.items() if not p)} not found")
    d = tempfile.mkdtemp(prefix="x2lb_sg_")
    try:
        g_o, g_co = os.path.join(d, "gated.o"), os.path.join(d, "gated.co")
        for cmd in ([tl["llvm-mc"], "-triple=amdgcn-amd-amdhsa", f"-mcpu={_ARCH}", "-filetype=obj", str(asm), "-o",
                     g_o], [tl["ld.lld"], "-shared", "--no-undefined", g_o, "-o", g_co]):
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if p.returncode != 0:
                raise RuntimeError(f"shipped vs gated: {os.path.basename(cmd[0])} rc={p.returncode}: "
                                   f"{(p.stderr or p.stdout).strip()[-300:]}")
        co = _code_object(so, tl["llvm-objcopy"], tl["clang-offload-bundler"], d)
        if co is None:
            raise RuntimeError(f"shipped vs gated: no {_ARCH} code object in {so.name}")
        res: Dict[str, Any] = {}
        for sec in (".text", ".rodata", ".note"):
            a, b = (_section_sha256(tl["llvm-objcopy"], f, sec, d) for f in (co, g_co))
            if a != b:
                raise RuntimeError(f"shipped code object {sec} {a[:12]} != the gated assembly's {b[:12]}")
            res[sec] = a
        syms, flags = [], []
        for f in (co, g_co):
            out = subprocess.run([tl["llvm-readelf"], "-h", "--dyn-syms", "-W", f], check=True, capture_output=True,
                                 text=True, timeout=120).stdout
            syms.append(sorted((t[7], *t[1:7]) for t in map(str.split, out.splitlines())
                               if len(t) == 8 and t[0][:-1].isdigit()))
            flags.append([ln.split(":", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("Flags:")])
        if not syms[0] or syms[0] != syms[1]:
            raise RuntimeError(f"shipped code object dynamic symbols != the gated assembly's: {syms[0][:3]} / "
                               f"{syms[1][:3]}")
        if len(flags[0]) != 1 or flags[0] != flags[1]:
            raise RuntimeError(f"shipped code object e_flags {flags[0]} != the gated assembly's {flags[1]}")
        res["dynsym"] = len(syms[0])
        res["e_flags"] = flags[0][0]
        return res
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextmanager
def _entry_lock(d: Path) -> Iterator[None]:
    """One builder / ISA gate per cache entry and node (fcntl); the others wait, then hit the cache."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover
        yield
        return
    fd = os.open(str(d / ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_name(f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}-{path.name}")
    try:
        tmp.write_text(json.dumps(obj))
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _verdict_current(meta: Dict[str, Any], asm: Path) -> bool:
    v = meta.get("isa") or {}
    return (v.get("ok") is True and v.get("asm_norm_sha256") == asm_norm_sha256(asm.read_bytes())
            and v.get("checker") == ISA_FILES_SHA256 and meta.get("src_sha256") == SOURCE_SHA256)


def build(source: Path = _SOURCE, fresh: bool = False) -> Tuple[Path, Dict[str, Any], float, bool]:
    """(library path, meta, build seconds, cache hit). Compiles x2lb_pass.cu unless a finished cache entry (.so +
    .s + meta) exists, gates the census and the ISA (a hit re-validates both; the ISA gate re-runs if the stored
    verdict does not match this assembly / these checker files). fresh=True (a bad entry): deletes the shared entry
    and builds into a private directory (load() removes it once the library is loaded). Raises on any failure (fail
    closed)."""
    d = cache_dir(source)
    _ensure_root(d.parent)
    d.mkdir(exist_ok=True, mode=0o700)
    if fresh:
        with _entry_lock(d):  # never while another rank reads the entry under its lock
            for n in (_LIB_NAME, _META_NAME, _ASM_NAME):
                try:
                    (d / n).unlink()
                except OSError:
                    pass
    private = fresh or not (_trusted(d.parent) and _trusted(d))
    if private:
        d = Path(tempfile.mkdtemp(prefix=_PRIVATE_PREFIX))  # a rebuild, or an untrusted shared dir
    with _entry_lock(d):
        so, meta_p, asm = d / _LIB_NAME, d / _META_NAME, d / _ASM_NAME
        meta = None
        if so.is_file() and meta_p.is_file() and asm.is_file():
            try:
                meta = json.loads(meta_p.read_text())
            except (OSError, ValueError):
                meta = None  # unreadable / half-gone entry: a miss, rebuilt below under the lock
        if meta is not None:
            meta["resources"] = check_resources(meta["kernels_raw"])
            if not _verdict_current(meta, asm):
                meta["isa"] = isa_gate(asm)
                if not meta["isa"]["ok"]:
                    raise RuntimeError(f"ISA gate refuses the cached {so}: {_isa_why(meta['isa'])}")
                _write_json(meta_p, meta)
            return so, meta, 0.0, True
        src = source.read_bytes()
        hipcc = _hipcc()
        uniq = f"tmp-{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
        tmp, tmp_asm, work = d / f"{uniq}-{_LIB_NAME}", d / f"{uniq}-{_ASM_NAME}", d / f"{uniq}-build"
        cmd = [hipcc, str(source.resolve()), *build_flags(src), "-o", str(tmp.resolve())]
        t0 = time.perf_counter()
        try:
            work.mkdir(parents=True, exist_ok=True, mode=0o700)
            rc, out = _run_group(cmd, _BUILD_TIMEOUT_S, cwd=str(work))  # --save-temps writes into cwd
            if rc != 0 or not tmp.is_file():
                raise RuntimeError(f"hipcc rc={rc}: {' '.join(cmd)}: {out.strip()[-1200:]}")
            kernels = parse_resource_remarks(out)
            resources = check_resources(kernels)
            dev_s = sorted(work.glob(f"*{_ARCH}*.s"))
            if not dev_s:
                raise RuntimeError("--save-temps produced no gfx950 device assembly: cannot run the ISA gate")
            shutil.copyfile(dev_s[0], tmp_asm)
            isa = isa_gate(tmp_asm)
            if not isa["ok"]:
                raise RuntimeError(f"ISA gate refuses this build: {_isa_why(isa)}")
            # ROCm 10.2: the gated .s is the LTO link's second codegen (".lto.s"), not the assembly of what ships
            ship = shipped_vs_gated(tmp, tmp_asm, hipcc) if ".lto." in dev_s[0].name else None
            build_s = time.perf_counter() - t0
            meta = dict(src_sha256=hashlib.sha256(src).hexdigest(), cmd=cmd, hipcc_version=_toolchain(hipcc),
                        kernels_raw=kernels, resources=resources, isa=isa, shipped_vs_gated=ship,
                        code_object=code_object_sha256(tmp, hipcc), build_s=round(build_s, 1))
            os.replace(tmp_asm, asm)
            _write_json(meta_p, meta)
            os.replace(tmp, so)  # the .so last: whoever sees it also finds its assembly and meta
        finally:
            for p in (tmp, tmp_asm):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            shutil.rmtree(work, ignore_errors=True)
        return so, meta, build_s, False


def _isa_why(isa: Dict[str, Any]) -> str:
    bad = {k: v.get("failed") for k, v in (isa.get("kernels") or {}).items() if not v.get("ok")}
    return (f"mode {isa.get('mode')} rc {isa.get('rc')} failures {bad or '(no verdict)'}"
            + ("" if isa.get("ok") else f" tail {str(isa.get('tail'))[-300:]!r}"))


# ctypes signatures of the C ABI (x2lb_pass.cu section 5; ops/tests checks them against the source).
_VP, _I, _LL, _CP = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_char_p
_LAUNCH_ARGS: Tuple[Any, ...] = (_VP,) * 12 + (_LL,) * 3 + (_VP,)
SIGNATURES: Dict[str, Tuple[Any, Tuple[Any, ...]]] = {
    "x2lb_abi_version": (_I, ()),
    "x2lb_variant": (_I, ()),
    "x2lb_impl": (_CP, ()),
    "x2lb_plan": (_LL, (_LL, _LL)),
    "x2lb_launch": (_I, _LAUNCH_ARGS),
    "x2lb_launch_emit": (_I, _LAUNCH_ARGS),
    "x2lb_launch_s0": (_I, _LAUNCH_ARGS),
    "x2lb_fin_launch": (_I, (_VP, _LL, _VP, _VP)),
    "x2lb_drain_last_error": (_I, ()),
}


class _Lib:
    def __init__(self, so: Path, meta: Dict[str, Any], build_s: float, cached: bool) -> None:
        self.path, self.meta, self.build_s, self.cached = so, meta, build_s, cached
        self.cdll = ctypes.CDLL(str(so))
        for name, (res, args) in SIGNATURES.items():
            fn = getattr(self.cdll, name)
            fn.restype = res
            fn.argtypes = list(args)
            setattr(self, name, fn)
        got = int(self.x2lb_abi_version())
        if got != ABI_VERSION:
            raise RuntimeError(f"{so}: x2lb_abi_version()={got}, expected {ABI_VERSION}")
        var = int(self.x2lb_variant())
        if var != VARIANT:
            raise RuntimeError(f"{so}: x2lb_variant()={var} (a lab negative-control build), expected {VARIANT}")
        impl = (self.x2lb_impl() or b"").decode(errors="replace")
        if f"src_sha256={SOURCE_SHA256}" not in impl:
            raise RuntimeError(f"{so}: x2lb_impl() does not carry src_sha256={SOURCE_SHA256[:16]}...: {impl[-120:]}")
        self.impl = impl

    def plan(self, M: int, num_cu: int) -> int:
        g = int(self.x2lb_plan(int(M), int(num_cu)))
        if g < 1:
            raise ValueError(f"x2lb_plan(M={M}, num_cu={num_cu}) rejected ({g})")
        return g


def load() -> "_Lib":
    """Build (or find in the cache) and load the library. Called by phase 1 only. A cached entry that fails to load
    or its ABI / variant / impl check is deleted and rebuilt once (a stale entry must not disable X2L-B on the node
    for every later job). A private build directory (an untrusted root, a rebuild) is removed once its library is
    loaded or refused: nobody else can reuse it, and the mapped .so stays valid after the unlink."""
    global _lib
    if _lib is None:
        with _lock:
            if _lib is None:
                built = build()
                try:
                    lib = _Lib(*built)
                except Exception:
                    _rm_private(built[0])
                    if not built[3]:
                        raise
                    built = build(fresh=True)
                    try:
                        lib = _Lib(*built)
                    finally:
                        _rm_private(built[0])
                _rm_private(built[0])
                _lib = lib
    return _lib


def _rm_private(so: Path) -> None:
    if so.parent.name.startswith(_PRIVATE_PREFIX):
        shutil.rmtree(so.parent, ignore_errors=True)


def _check_rc(rc: int, what: str) -> None:
    if rc < 0:  # launched; -rc is a stale error an earlier call had left on this thread
        _once("stale", str(-rc), f"{TAG} stale HIP error {-rc} drained before {what} "
              "(left on this thread by an earlier call, not ours)")
    elif rc != 0:
        raise RuntimeError(f"{what} returned hip error {rc}")


# ---------------------------------------------------------------------------------------- runtime
def _ptr(t: Optional[torch.Tensor]) -> Optional[int]:
    return None if t is None else int(t.data_ptr())


def _fin_pr7(part: torch.Tensor, dw: torch.Tensor, db: torch.Tensor, dxs: torch.Tensor, grid: int,
             stream: Any) -> None:
    """PR #7's finalize over X2L-B's PART (FIX R14): the build_finalize(256, 3, True) specialization PR #7 itself
    launches (ln_bwd.py) and the startup preload prebuilds; G is a runtime Int32."""
    lb = _LB
    if lb is None:
        raise RuntimeError("ops/flydsl/ln_bwd not loaded (phase 2 did not run)")
    fin = lb.build_finalize(HID, 3, True)
    fast = getattr(lb, "_fast_call", None)
    if fast is not None:
        # ln_bwd._fast_call, as PR #7's own launches: FlyDSL 0.2.4's plain @flyc.jit call re-derives its cache key
        # and re-snapshots the launcher's globals per call (48 us of host time per finalize, 5 us through
        # _fast_call, 9.5 us under FlyDSL 0.1.1); same kernel and arguments, outputs bit-identical.
        fast(fin, part, dw, db, dxs, int(grid), lb.fx.Stream(stream))
    else:
        fin(part, dw, db, dxs, int(grid), stream=lb.fx.Stream(stream))


class _CudaRuntime:
    """The device side (tests substitute a mock): streams, capture, #CU, the sentinel scalar, the launches."""

    def __init__(self) -> None:
        self._ncu: Dict[int, int] = {}

    @staticmethod
    def stream(dev: torch.device) -> Any:
        return torch.cuda.current_stream(dev)

    @staticmethod
    def capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    def num_cu(self, dev: torch.device) -> int:
        idx = _dev_index(dev)
        n = self._ncu.get(idx)
        if n is None:
            n = int(torch.cuda.get_device_properties(dev).multi_processor_count)
            self._ncu[idx] = n
        return n

    @staticmethod
    def make_zero(dev: torch.device) -> torch.Tensor:
        z = torch.zeros((), dtype=torch.bfloat16, device=dev)
        torch.cuda.current_stream(dev).synchronize()  # made at arm time: every later stream reads a zero
        return z

    @staticmethod
    def launch(lib: "_Lib", mode: str, dh8: Optional[torch.Tensor], w8t: Optional[torch.Tensor],
               sa: Optional[torch.Tensor], sb: Optional[torch.Tensor], h1: torch.Tensor, mean1: torch.Tensor,
               rstd1: torch.Tensor, nw: torch.Tensor, nb: torch.Tensor, dh1: torch.Tensor,
               da1: Optional[torch.Tensor], part: torch.Tensor, M: int, grid: int, stream: Any) -> None:
        fn = {"s1": lib.x2lb_launch, "emit": lib.x2lb_launch_emit, "s0": lib.x2lb_launch_s0}[mode]
        if int(part.shape[1]) != int(grid):
            raise ValueError(f"PART rows {int(part.shape[1])} != grid {grid}")
        rc = fn(_ptr(dh8), _ptr(w8t), _ptr(sa), _ptr(sb), _ptr(h1), _ptr(mean1), _ptr(rstd1), _ptr(nw), _ptr(nb),
                _ptr(dh1), _ptr(da1), _ptr(part), int(grid), int(M), int(grid), int(stream.cuda_stream))
        _check_rc(int(rc), f"x2lb_launch({mode})")

    @staticmethod
    def fin(lib: "_Lib", mode: str, part: torch.Tensor, grid: int, stream: Any,
            out: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(dW, dB, dXS) fp32 [256] from PART [3, grid, 256]. mode "pr7" (default): PR #7's finalize into three
        separate [256] tensors (out: a 3-tuple of them, else fresh); "x2lb" (diagnostic): x2lb_fin into one
        [3, 256] buffer (out: that buffer, else fresh) and its three rows."""
        dev = part.device
        if mode == "x2lb":
            buf = torch.empty((3, HID), dtype=torch.float32, device=dev) if out is None else out
            _check_rc(int(lib.x2lb_fin_launch(part.data_ptr(), int(grid), buf.data_ptr(),
                                              int(stream.cuda_stream))), "x2lb_fin_launch")
            return buf[0], buf[1], buf[2]
        if out is None:
            out = tuple(torch.empty(HID, dtype=torch.float32, device=dev) for _ in range(3))
        dw, db, dxs = out
        _fin_pr7(part, dw, db, dxs, grid, stream)
        return dw, db, dxs

    def fused(self, lib: "_Lib", dh8: torch.Tensor, w8_t: torch.Tensor, dh_scale: torch.Tensor,
              w_scale: torch.Tensor, h1: torch.Tensor, mean1: torch.Tensor, rstd1: torch.Tensor, nw: torch.Tensor,
              nb: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], int, Any]:
        """The production call: S1 into a fresh dh1 / PART, then the finalize, on the current stream of the
        tensors' device. ((dh1, dW, dB, db1), grid, stream)."""
        if lib is None:
            raise RuntimeError("library not loaded (phase 1 did not run)")
        dev = h1.device
        cur = torch.cuda.current_device()
        if dev.index is not None and dev.index != cur:
            raise ValueError(f"tensors on {dev} but the current device is cuda:{cur}")
        M = int(h1.shape[0])
        grid = lib.plan(M, self.num_cu(dev))
        st = torch.cuda.current_stream(dev)
        dh1 = torch.empty((M, HID), dtype=torch.bfloat16, device=dev)
        part = torch.empty((3, grid, HID), dtype=torch.float32, device=dev)
        self.launch(lib, "s1", dh8, w8_t, dh_scale, w_scale, h1, mean1, rstd1, nw, nb, dh1, None, part, M, grid, st)
        mode = fin_mode()
        dw, db, dxs = self.fin(lib, mode, part, grid, st)
        if mode == "x2lb":  # rows of one [3, 256] buffer: separate tensors, as PR #7's finalize writes (no aliasing
            dw, db, dxs = dw.clone(), db.clone(), dxs.clone()  # .grad of norm0 w / b and b1)
        return (dh1, dw, db, dxs), grid, st


_RT: Any = _CudaRuntime()


# ---------------------------------------------------------------------------------------- plumbing
class X2LBHandle:
    """One FP8 MLP's X2L-B plumbing for one forward (DESIGN_stageC 3.1). Strong refs TO it: the pair-0 ctx
    (ctx.x2lb), the Slot until the tail's forward, then the tail ctx (ctx.x2lb[k]). It holds no strong tensor
    reference before the tail's backward, and only weakrefs to the pair-0 ctx and to the proxy (no reference
    cycle: the run gc.freeze()s and relies on refcounting, as E-X2b's PortRecord)."""

    __slots__ = ("k", "p0", "proxy", "stream", "result", "sentinel", "state", "grid", "task", "__weakref__")

    def __init__(self, k: int, p0_ctx: Any, stream: Any) -> None:
        self.k = k
        self.p0 = weakref.ref(p0_ctx)
        self.stream = stream
        self.proxy: Optional[weakref.ref] = None  # set at attach (the proxy does not exist yet at publish)
        self.result: Optional[Tuple[torch.Tensor, ...]] = None  # (dh1, dW, dB, db1) after a fused tail backward
        self.sentinel: Optional[torch.Tensor] = None  # the expanded zero view returned as the proxy's gradient
        self.state = "published"  # published | attached | fused | consumed | declined
        self.grid = 0
        self.task: Optional[int] = None  # the autograd graph task that made `result` (_graph_task_id at fuse time)

    def foreign(self) -> bool:
        """The pending result belongs to another (live or aborted) backward than the one running now."""
        return self.result is not None and self.task is not None and self.task != _graph_task_id()

    def take(self, dy: Optional[torch.Tensor]) -> Optional[Tuple[torch.Tensor, ...]]:
        """Pair 0's backward: the fused (dh1, dW, dB, db1) iff dy is this handle's sentinel; None -> the shipped
        statements. Raises on any mismatch (never a silent zero, never a silently ignored replacement gradient,
        never a double LN backward)."""
        z = _on_zero(dy)
        if z is not None and z._version != 0:  # a hook wrote the sentinel (fill_ / zero_ keep ptr and strides)
            _abort_task(self)
            raise RuntimeError(f"{TAG} handshake: mlp {self.k}: the sentinel was written in place (a hook on the "
                               "proxy's gradient modified it)")
        if self.result is None:
            if z is not None:
                _abort_task(self)
                raise RuntimeError(f"{TAG} handshake: mlp {self.k}: a sentinel gradient without fused results")
            return None
        if self.foreign():
            if z is not None:
                _abort_task(self)
                raise RuntimeError(f"{TAG} handshake: mlp {self.k}: a sentinel gradient in a backward other than the "
                                   "one that fused")
            return None  # another backward's pending result (overlapping backwards): this one runs the shipped path
        s = self.sentinel
        if z is None or s is None or dy.data_ptr() != s.data_ptr() or tuple(dy.shape) != tuple(s.shape):
            _abort_task(self)
            raise RuntimeError(f"{TAG} handshake: mlp {self.k}: fused results but the proxy gradient is not the "
                               "sentinel (another consumer or hook changed it)")
        r = self.result
        self.result, self.sentinel, self.state, self.task = None, None, "consumed", None
        _PENDING.discard(self)
        if r[0].is_cuda:
            cur = torch.cuda.current_stream(r[0].device)
            if cur != self.stream:  # never in production (both nodes on one stream); belt and braces
                cur.wait_stream(self.stream)
                for t in r:
                    t.record_stream(cur)
        return r

    def _drop_own(self) -> None:
        """Nothing of this backward stays behind: clears the result unless another backward owns it."""
        if not self.foreign():
            self.result, self.sentinel, self.state, self.task = None, None, "declined", None
            _PENDING.discard(self)


def _graph_task_id() -> int:
    f = getattr(torch._C, "_current_graph_task_id", None)
    return -1 if f is None else int(f())


def _abort_task(hd: X2LBHandle) -> None:
    """A handshake error aborts the running backward: drop hd's own result and every other result this backward
    fused (the other MLP's too), so nothing of it stays held while the caller keeps the graph."""
    hd._drop_own()
    t = _graph_task_id()
    for h in list(_PENDING):
        if h.task == t:
            h._drop_own()


def _on_zero(dy: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """The per-device zero scalar that dy is an expanded view of (a sentinel: 2-D, strides (0, 0), on a _ZERO
    storage), or None. Any other gradient, stride-0 or not, is not a sentinel."""
    if dy is None or dy.dim() != 2 or dy.stride() != (0, 0):
        return None
    p = dy.data_ptr()
    for z in _ZERO.values():
        if z.data_ptr() == p and z.device == dy.device:
            return z
    return None


def _saved_tensor_hooks_active() -> bool:
    """torch.autograd.graph.saved_tensors_hooks (non-reentrant activation checkpointing, save_on_cpu) is active
    (the IX precedent, ops/ec_indexed_l1.py). Missing API: False."""
    f = getattr(torch._C._autograd, "_top_saved_tensors_default_hooks", None)
    return f is not None and f(False) is not None


def _tracing(x: Any) -> bool:
    """E-X2b's tracing() (fx / compile / an fx.Proxy argument)."""
    from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as xs

    return bool(xs.tracing(x))


def publish_reason(k: int, h: torch.Tensor, train: bool) -> Optional[str]:
    """Why pair 0's forward publishes no handle (silent), or None. Pure."""
    if not train:
        return "not train"
    if _tracing(h):
        return "tracing"
    if _saved_tensor_hooks_active():
        return "saved-tensor hooks active"
    if _graph_task_id() != -1:
        return "inside a backward"
    if int(k) not in mlps():
        return f"mlp {k} not in {MLPS_ENV_VAR}"
    return None


def publish(ctx: Any, k: int, h: torch.Tensor, train: bool) -> Optional[X2LBHandle]:
    """Pair 0's forward (after save_for_backward): a handle for this forward's MLP k, or None (silent decline; a
    device that is not armed prints its NOT armed / IGNORED line once). Never raises."""
    try:
        if not requested() or _BROKEN is not None:
            return None
        if publish_reason(k, h, train) is not None:
            return None
        if not active(h.device):
            _not_armed_line(h.device)
            return None
        return X2LBHandle(int(k), ctx, _RT.stream(h.device))
    except Exception as exc:
        latch_off("publish", exc)
        return None


def _decline(hd: Optional[X2LBHandle], where: str, why: str, say: bool = True) -> None:
    """Count the reason class; one DECLINED line per class (say=False: counted only). A handle whose pending
    result belongs to another backward keeps its state."""
    if hd is not None and hd.result is None:
        hd.state = "declined"
    c = _cls(why)
    DECLINES[c] = DECLINES.get(c, 0) + 1
    if say:
        _once("decline", c, f"{TAG} DECLINED at {where} ({why}) -> {SHIPPED} for such calls")


def attach(hd: Optional[X2LBHandle], a1_rm: torch.Tensor, proxy: torch.Tensor) -> Optional[X2LBHandle]:
    """The tail's forward (the FP8 payload's single-use handoff): the handle for ctx.x2lb[k] with the proxy recorded
    by weakref, or None (declined: a stream / device other than pair 0's, tracing). Never raises."""
    if hd is None:
        return None
    try:
        why = None
        if hd.state != "published":
            why = f"handle state {hd.state}"
        elif hd.p0() is None:
            # Silent (review NIT): pair 0 built no node (no input requires grad), so there is nothing to fuse.
            _decline(hd, "attach", "pair 0's ctx gone", say=False)
            return None
        elif _tracing(proxy):
            why = "tracing"
        elif not active(a1_rm.device):
            why = "device not armed"
        elif _RT.stream(a1_rm.device) != hd.stream:
            why = "the tail's forward stream differs from pair 0's"
        elif proxy.dim() != 2 or int(proxy.shape[1]) != HID or proxy.device != a1_rm.device:
            why = f"proxy {tuple(proxy.shape)} on {proxy.device}"
        if why is not None:
            _decline(hd, "attach", why)
            return None
        hd.proxy = weakref.ref(proxy)
        hd.state = "attached"
        return hd
    except Exception as exc:
        latch_off("attach", exc)
        return None


def will_run(p0: Any) -> bool:
    """This backward EXECUTES pair 0 and is not a capture (DESIGN_stageC 3.5, review A1): _will_engine_execute_node
    on pair 0 AND on norm0.weight's AccumulateGrad (pair 0's 4th tensor input, next_functions[3]); the leaf query
    raises inside every torch.autograd.grad. A missing API, a None node or any exception -> False (decline).
    next_functions lists pair 0's TENSOR inputs (x, lin_w, lin_b, norm_w, norm_b[, ex2b_port]); lin_b is always a
    tensor there (its forward reads lin_b.dtype), so index 3 is norm_w's node (pinned by an ops/tests test)."""
    f = getattr(torch._C, "_will_engine_execute_node", None)
    if f is None or p0 is None:
        return False
    try:
        nf = p0.next_functions
        nw_node = nf[3][0] if len(nf) > 3 else None
        return nw_node is not None and bool(f(p0)) and bool(f(nw_node))
    except Exception:
        return False


def call_reason(hd: X2LBHandle, p0: Any) -> Optional[str]:
    """Why the tail's backward must not fuse this call (DESIGN_stageC 5.4 call-time list, autograd part), or None.
    Pure: launches nothing, never latches."""
    if torch.is_grad_enabled():
        return "create_graph backward"
    if torch.is_anomaly_enabled():
        return "anomaly mode"
    if p0 is None:
        return "pair 0's ctx gone"
    if hd.state not in ("attached", "consumed", "declined", "fused"):
        return f"handle state {hd.state}"
    if hd.foreign():  # kept: its owner (an outer, live backward) still consumes it
        return "a fused result of another backward is pending (overlapping or aborted backward)"
    if hd.result is not None:
        return "an earlier fused result was never consumed"
    if hd.proxy is None or hd.proxy() is not None:
        return "proxy still referenced (capture, retain_grad or hook)"
    if not will_run(p0):
        return "pair 0 does not run in this backward or is only captured (autograd.grad / backward(inputs=))"
    if _RT.capturing():
        return "CUDA graph capture"
    return None


def _a16(t: torch.Tensor) -> bool:
    return int(t.data_ptr()) % 16 == 0


def tensor_reason(dh8: torch.Tensor, w8_t: torch.Tensor, dh_scale: Optional[torch.Tensor],
                  w_scale: Optional[torch.Tensor], h1: torch.Tensor, nw: torch.Tensor, nb: torch.Tensor,
                  mean1: torch.Tensor, rstd1: torch.Tensor, x_dt: torch.dtype) -> Optional[str]:
    """Why these operands are outside x2lb_pass's contract or PR #7 would not own the shipped SwishLN backward, or
    None. Pure (the PR #7 query launches, logs and latches nothing)."""
    dev = dh8.device
    if h1.dim() != 2:
        return f"h1 dim {h1.dim()}"
    M = int(h1.shape[0])
    if not (M_MIN <= M <= M_MAX):
        return f"M {M} outside [{M_MIN}, {M_MAX}]"
    if x_dt != torch.bfloat16:
        return f"proxy dtype {x_dt}"
    if dh8.dtype != torch.float8_e5m2 or tuple(dh8.shape) != (M, KD) or not dh8.is_contiguous() or not _a16(dh8):
        return f"dh8 {dh8.dtype} {tuple(dh8.shape)} stride {tuple(dh8.stride())} (needs e5m2 [M,512] contiguous, 16-B)"
    if (w8_t.dtype != torch.float8_e4m3fn or tuple(w8_t.shape) != (HID, KD) or not w8_t.is_contiguous()
            or not _a16(w8_t)):
        return f"w8_t {w8_t.dtype} {tuple(w8_t.shape)} (needs e4m3 [256,512] contiguous, 16-B)"
    for name, s in (("dh_scale", dh_scale), ("w_scale", w_scale)):
        if s is None:
            return f"{name} missing"
        if s.dtype != torch.float32 or s.numel() != 1 or s.device != dev or int(s.data_ptr()) % 4:
            return f"{name} {s.dtype} {tuple(s.shape)} on {s.device} (needs fp32, one element, on {dev})"
    if h1.dtype != torch.bfloat16 or tuple(h1.shape) != (M, HID) or not h1.is_contiguous() or not _a16(h1):
        return f"h1 {h1.dtype} {tuple(h1.shape)} stride {tuple(h1.stride())} (needs bf16 [M,256] contiguous, 16-B)"
    for name, t in (("mean1", mean1), ("rstd1", rstd1)):
        if t.dtype != torch.float32 or tuple(t.shape) != (M,) or not t.is_contiguous() or int(t.data_ptr()) % 4:
            return f"{name} {t.dtype} {tuple(t.shape)} (needs fp32 [M] contiguous)"
    for name, t in (("norm weight", nw), ("norm bias", nb)):
        if t.dtype != torch.float32 or tuple(t.shape) != (HID,) or not t.is_contiguous() or not _a16(t):
            return f"{name} {t.dtype} {tuple(t.shape)} (needs fp32 [256] contiguous, 16-B)"
    for name, t in (("w8_t", w8_t), ("h1", h1), ("mean1", mean1), ("rstd1", rstd1), ("norm weight", nw),
                    ("norm bias", nb)):
        if t.device != dev:
            return f"{name} on {t.device}, dh8 on {dev}"
    from generative_recommenders.ops.triton import triton_layer_norm as TLN

    why = TLN.flydsl_ln_bwd_decline_reason(h1, h1, nw, nb, mean1, rstd1, is_swish=True, sum_dx=True)
    if why is not None:
        return f"PR #7 would not own the shipped SwishLN backward: {why}"
    return None


def _drop(hd: X2LBHandle) -> None:
    """A call that does not fuse leaves no result of ITS backward behind; a pending result of another backward
    (overlapping backwards, or one that aborted after its tail fused) is left to its owner (take() lets this
    backward's shipped da1 through)."""
    hd._drop_own()


def release(hds: Optional[List[Optional[X2LBHandle]]]) -> None:
    """The tail's backward raised: drop the results it fused (each about M x 512 B) so they do not outlive the
    failed backward while the graph is kept. Never raises (the original exception propagates)."""
    for hd in hds or ():
        if hd is not None:
            try:
                hd._drop_own()
            except Exception:  # pragma: no cover - defensive
                pass


def try_fused(hd: X2LBHandle, dh8: torch.Tensor, w8_t: torch.Tensor, dh_scale: Optional[torch.Tensor],
              w_scale: Optional[torch.Tensor], x_dt: torch.dtype) -> Optional[torch.Tensor]:
    """The tail's backward, in place of the FP8 dgrad statement for MLP hd.k: the sentinel (the proxy's gradient)
    after launching S1 + the finalize into hd, or None -> the caller runs today's _scaled_mm dgrad (knob off,
    declined or latched; inputs intact). A version-counter / freed-buffer error of pair 0's saved tensors raises as
    it would in pair 0."""
    try:
        dev = dh8.device
        z = _ZERO.get(_dev_index(dev))
        if z is not None and z._version != 0:  # never written after arming; a hook wrote it in place
            raise RuntimeError("the per-device zero sentinel was written in place (a hook on a proxy gradient)")
        if not active(dev):
            _drop(hd)
            return None
        p0 = hd.p0()
        why = call_reason(hd, p0)
    except Exception as exc:
        _drop(hd)
        latch_off("x2lb call checks", exc)
        return None
    if why is None:
        sv = p0.saved_tensors  # NOT in a try: version / freed errors raise as in pair 0
        h1, nw, nb, mean1, rstd1 = sv[3], sv[4], sv[5], sv[6], sv[7]
        try:
            why = tensor_reason(dh8, w8_t, dh_scale, w_scale, h1, nw, nb, mean1, rstd1, x_dt)
        except Exception as exc:
            _drop(hd)
            latch_off("x2lb call checks", exc)
            return None
    if why is not None:
        _drop(hd)
        _decline(hd, "call", why)
        return None
    try:
        result, grid, st = _RT.fused(_lib, dh8.detach(), w8_t.detach(), dh_scale.detach(), w_scale.detach(),
                                     h1.detach(), mean1.detach(), rstd1.detach(), nw.detach(), nb.detach())
        M = int(h1.shape[0])
        sentinel = _ZERO[_dev_index(dev)].expand(M, HID)
    except Exception as exc:  # host-side error before / between launches: inputs intact
        _drop(hd)
        latch_off("x2lb call", exc)
        return None
    hd.result, hd.stream, hd.sentinel, hd.state, hd.grid = tuple(result), st, sentinel, "fused", int(grid)
    hd.task = _graph_task_id()
    _PENDING.add(hd)
    FUSED_CALLS[hd.k] = FUSED_CALLS.get(hd.k, 0) + 1
    idx = _dev_index(dev)
    if ("fused", str(idx)) not in _ONCE:
        _once("fused", str(idx),
              f"{ARMED_PREFIX} fused L2 dgrad + SwishLN256 bwd on cuda:{idx} mlp={hd.k} M={M} grid={grid} "
              f"fin={PR7_FIN_NAME if fin_mode() == 'pr7' else 'x2lb_fin (diagnostic)'}; da1 not written; kill "
              f"switch {ENV_VAR}=0")
    return sentinel


# ------------------------------------------------------------------------------------- prebuild phase 1
def _phase1_ignored_reason(device: torch.device) -> Optional[str]:
    from generative_recommenders.ops import preproc_fp8_l2 as l2
    from generative_recommenders.ops.triton import triton_layer_norm as TLN

    if not l2.enabled():
        return "E-M3a (DLRM_HSTU_FP8_PREPROC_L2) is off or IGNORED"
    if not TLN._wln_bwd_flydsl_enabled():
        return "PR #7 (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) is off"
    if device.type != "cuda" or not torch.cuda.is_available():
        return f"device {device} is not a GPU"
    arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
    if arch != _ARCH:
        return f"device arch {arch} is not {_ARCH} (x2lb_pass.cu is {_ARCH}-only)"
    return None


def prebuild_build(device: torch.device) -> bool:
    """Phase 1 (blas_warmup.run_blas_warmup, before the DataLoader forks): build or hit the cache, census, ISA gate,
    ABI. Prints the first armed line; never raises (a failure latches X2L-B off: one FAILED CLOSED line)."""
    global _P1_STATE
    if not requested() or _BROKEN is not None or _P1_STATE is not None:
        return False
    t0 = time.perf_counter()
    try:
        mlps()
        fin_mode()
        why = _phase1_ignored_reason(device)
    except Exception as exc:
        _P1_STATE = "failed"
        latch_off("phase 1 (configuration)", exc)
        return False
    if why is not None:
        _P1_STATE = "ignored"
        ignored(why)
        return False
    try:
        lib = load()
        res = lib.meta["resources"][KERNEL_S1]
        isa = lib.meta["isa"]
        co = lib.meta.get("code_object") or {"status": "unknown (not recorded)"}
    except Exception as exc:
        _P1_STATE = "failed"
        latch_off("phase 1 (build / census / ISA gate / ABI)", exc)
        return False
    _P1_STATE = "ok"
    secs = time.perf_counter() - t0
    if co.get("status") == "ok":
        co_txt = (f"code object .text {co['.text'][:12]} "
                  + ("(== lab)" if co.get("lab") == "identical" else
                     f"(differs from lab {LAB_CODE_OBJECT_SHA256['.text'][:12]})"))
    else:
        co_txt = f"code object {co.get('status')}"
    isa_txt = ("full" if isa.get("mode") == "full" else
               f"pinned .s {str(isa.get('asm_norm_sha256') or PINNED_ASM_NORM_SHA256)[:12]}")
    if lib.meta.get("shipped_vs_gated"):  # ROCm 10.2 (shipped_vs_gated)
        isa_txt += "; code object == the gated .lto.s"
    PHASE1_INFO.clear()
    PHASE1_INFO.update(lib=str(lib.path), cached=lib.cached, build_s=lib.build_s, resources=res, isa=isa,
                       code_object=co, seconds=secs, mlps=mlps(), fin=fin_mode())
    _say(f"{ARMED_PREFIX} pre-run_start build OK ({PASS_NAME}<0,0> VGPR {res.get('unified_vgpr')} / {LIM_VGPR}, "
         f"spill {res.get('vgpr_spill')}, scratch {res.get('scratch')}, LDS {res.get('lds')}, occupancy "
         f"{res.get('occupancy')}; isa_check C1-C18 PASS ({isa_txt}); src {SOURCE_SHA256[:12]}; {co_txt}; "
         f"{lib.path.parent.name}/{_LIB_NAME} "
         f"{'cached' if lib.cached else f'built in {lib.build_s:.1f}s'}; mlps {','.join(map(str, mlps()))}; "
         f"fin {fin_mode()}; phase 1 {secs:.1f}s)")
    return True


# ------------------------------------------------------------------------------------- prebuild phase 2
_M32 = 0xFFFFFFFF
_CH: int = 1 << 16  # rows per generator / compare chunk (bounded transients)


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """32-bit integer hash of int64 x (low 32 bits); stays below 2^59 in int64 (X2H crosscheck_input)."""
    x = x & _M32
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & _M32
    x = ((x ^ (x >> 16)) * 0x45D9F3B) & _M32
    return x ^ (x >> 16)


def _hashed(n: int, salt: int, start: int, device: torch.device) -> torch.Tensor:
    return _mix32(torch.arange(start, start + n, dtype=torch.int64, device=device) * 0x9E3779B1 + salt)


def _unit(h: torch.Tensor) -> torch.Tensor:
    """fp32 in [0, 1) from the low 24 bits of a hash."""
    return (h & 0xFFFFFF).to(torch.float32) * (1.0 / (1 << 24))


class _Live:
    """Phase 2's big allocations with a live-byte account (the peak is the reported transient; tests replace
    alloc to check the frees of DESIGN_stageC 5.3 step 4)."""

    def __init__(self) -> None:
        self.live: Dict[str, int] = {}
        self.peak = 0

    def alloc(self, name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        t = torch.empty(shape, dtype=dtype, device=device)
        self.add(name, t)
        return t

    def add(self, name: str, t: torch.Tensor) -> None:
        self.live[name] = t.numel() * t.element_size()
        self.peak = max(self.peak, sum(self.live.values()))

    def free(self, *names: str) -> None:
        for n in names:
            self.live.pop(n, None)


_BF16_CANARY: int = 0x7FA5  # a quiet-NaN bit pattern
_F32_CANARY: int = 0x7FC0A5A5


def _canary_bf16(live: _Live, name: str, M: int, device: torch.device) -> torch.Tensor:
    t = live.alloc(name, (M + CANARY_ROWS, HID), torch.bfloat16, device)
    t.view(torch.int16).fill_(_BF16_CANARY)
    return t


def _canary_f32(live: _Live, name: str, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    t = live.alloc(name, shape, torch.float32, device)
    t.view(torch.int32).fill_(_F32_CANARY)
    return t


def cancel_data(M: int, device: torch.device, live: _Live, salt: int = 0x2B1D) -> Dict[str, torch.Tensor]:
    """The phase-2 operands (DESIGN_stageC 5.3 step 1, the lab kit's "cancel" construction from integer hashes; no
    RNG state is read or written): dh8 e5m2 [M, 512] with K step 1 = -(K step 0) bytes (large codes 0x50-0x7A
    there, small codes 0x01-0x2F in steps 2-3), w8_t e4m3 [256, 512] with step 1's bytes = step 0's, dh_scale
    2^-20, w_scale 2^-10, h1 bf16 rows with per-row scale / offset, mean1 / rstd1 = fp32 of fp64 statistics
    (eps 1e-5), nw = 1 +- 0.2, nb = +-0.3. Finite everywhere."""
    dh8 = live.alloc("dh8", (M, KD), torch.float8_e5m2, device)
    h1 = live.alloc("h1", (M, HID), torch.bfloat16, device)
    mean1 = live.alloc("mean1", (M,), torch.float32, device)
    rstd1 = live.alloc("rstd1", (M,), torch.float32, device)
    u8 = dh8.view(torch.uint8)
    for s in range(0, M, _CH):
        e = min(M, s + _CH)
        n = e - s
        hb = _hashed(n * 128, salt + 1, s * 128, device).view(n, 128)
        big = (0x50 + hb % 43) | (((hb >> 16) & 1) << 7)
        u8[s:e, 0:128] = big.to(torch.uint8)
        u8[s:e, 128:256] = (big ^ 0x80).to(torch.uint8)
        del hb, big
        hs = _hashed(n * 256, salt + 2, s * 256, device).view(n, 256)
        u8[s:e, 256:512] = ((0x01 + hs % 47) | (((hs >> 16) & 1) << 7)).to(torch.uint8)
        hr = _hashed(n * 256, salt + 3, s * 256, device).view(n, 256)
        rr = _hashed(n, salt + 4, s, device)
        sc = 0.5 + 1.5 * _unit(rr)
        off = _unit(rr >> 8) - 0.5
        hb16 = ((_unit(hr) - 0.5) * 3.4641016 * sc[:, None] + off[:, None]).to(torch.bfloat16)
        h1[s:e] = hb16
        hd = hb16.double()
        mean1[s:e] = hd.mean(1).float()
        rstd1[s:e] = (1.0 / torch.sqrt(hd.var(1, unbiased=False) + 1e-5)).float()
        del hs, hr, rr, sc, off, hb16, hd
    hw = _hashed(HID * KD, salt + 5, 0, device).view(HID, KD)
    w = (0x20 + hw % 0x58) | (((hw >> 16) & 1) << 7)
    w = torch.where(((hw >> 20) & 15) == 0, torch.zeros_like(w), w).to(torch.uint8)
    w[:, 128:256] = w[:, 0:128]
    w8t = w.contiguous().view(torch.float8_e4m3fn)
    hn = _hashed(2 * HID, salt + 6, 0, device)
    nw = (1.0 + 0.2 * (2.0 * _unit(hn[:HID]) - 1.0)).float().contiguous()
    nb = (0.3 * (2.0 * _unit(hn[HID:]) - 1.0)).float().contiguous()
    sa = torch.full((1,), 2.0 ** -20, dtype=torch.float32, device=device)
    sb = torch.full((1,), 2.0 ** -10, dtype=torch.float32, device=device)
    return dict(dh8=dh8, w8t=w8t, sa=sa, sb=sb, h1=h1, mean1=mean1, rstd1=rstd1, nw=nw, nb=nb)


def _diff_rows(a: torch.Tensor, b: torch.Tensor, lo: int, hi: int) -> Tuple[int, Optional[int]]:
    """(differing rows, first differing row) of two 2-D 16-bit tensors over rows [lo, hi), compared as integers
    (-0 != +0, NaN payloads exact), in row chunks."""
    n, first = 0, None
    ai, bi = a.view(torch.int16), b.view(torch.int16)
    for s in range(lo, hi, _CH):
        e = min(hi, s + _CH)
        d = (ai[s:e] != bi[s:e]).any(dim=1)
        c = int(d.sum().item())
        if c and first is None:
            first = s + int(torch.nonzero(d)[0].item())
        n += c
    return n, first


def _n_eq_bits(t: torch.Tensor, bits: int) -> int:
    v = t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)
    return int((v == bits).sum().item())


def _canaries_ok(t: torch.Tensor, M: int, what: str) -> None:
    rows = t[M:]
    bad = int((rows.view(torch.int16) != _BF16_CANARY).sum().item())
    if bad:
        raise RuntimeError(f"{what}: {bad} canary words in rows >= M {M} were written")


def _written(t: torch.Tensor, what: str) -> None:
    left = _n_eq_bits(t, _F32_CANARY)
    if left:
        raise RuntimeError(f"{what}: {left} of {t.numel()} elements never written (sentinel left)")


def _gamma(h: int) -> float:
    u = 2.0 ** -24
    return h * u / (1.0 - h * u)


def finals_bound_check(d: Dict[str, torch.Tensor], da1: torch.Tensor, finals: Tuple[torch.Tensor, ...],
                       grid: int, M: int) -> float:
    """(f): the finals (dW, dB, dXS) vs a torch fp64 oracle of the same per-row terms (PR #7's swish row on
    dy = X2L-B's own bf16 da1), per column within (gamma_h + 2^-18) x S, h = 4 ceil(ntiles / G) + 4 + (G - 1)
    (IMPLEMENTATION 7's defect bound; PR #7's finalize tree is shallower than the G - 1 sequential fold it covers),
    S = the column's sum of an envelope of each row's fp32 sub-terms (e_t = |dy x s| >= |t| covers the (1 - s)
    cancellation; xhat's fma error is covered by rstd (|x| + |mean|); the row sums c1 / c2 by their absolute
    sums), 2^-18 = 64 u for the per-row fp32 evaluation (v_exp / v_rcp, roundings) [H: a canary bound; the UNIT
    gate's G-b applies the sealed G2c rule]. Returns the largest |err| / bound."""
    D = HID
    dy = da1[:M].double()
    x = d["h1"][:M].double()
    mean, rstd = d["mean1"][:M].double()[:, None], d["rstd1"][:M].double()[:, None]
    w, b = d["nw"].double()[None, :], d["nb"].double()[None, :]
    xhat = (x - mean) * rstd
    s = torch.sigmoid(xhat * w + b)
    t = dy * x * s * (1.0 - s)
    xu = w * t
    c1 = (xhat * xu).sum(1, keepdim=True) / D
    c2 = xu.sum(1, keepdim=True) / D
    dx = (xu - (xhat * c1 + c2)) * rstd + dy * s
    e_t = (dy * x * s).abs()
    xhat_env = xhat.abs() + rstd * (x.abs() + mean.abs())
    xu_env = w.abs() * e_t
    a_r = (xhat_env * xu_env).sum(1, keepdim=True) / D
    b_r = xu_env.sum(1, keepdim=True) / D
    refs = ((t * xhat).sum(0), t.sum(0), dx.sum(0))
    mags = ((e_t * xhat_env).sum(0), e_t.sum(0),
            (rstd * (xu_env + xhat_env * a_r + b_r) + (dy * s).abs()).sum(0))
    ntiles = (M + BM - 1) // BM
    h = 4 * ((ntiles + grid - 1) // grid) + 4 + (grid - 1)
    g = _gamma(h) + 2.0 ** -18
    worst = 0.0
    for name, out, ref, mag in zip(("dW", "dB", "dXS"), finals, refs, mags):
        o = out.double()
        if not bool(torch.isfinite(o).all().item()):
            raise RuntimeError(f"finals {name}: non-finite on finite data")
        bound = g * mag + 1e-300
        r = float(((o - ref).abs() / bound).max().item())
        if r > 1.0:
            raise RuntimeError(f"finals {name} outside (gamma_{h} + 2^-18) x sum|term| (use {r:.3g}) at M {M}")
        worst = max(worst, r)
    return worst


class HeadroomError(RuntimeError):
    """Phase 2 would not fit: FAILED CLOSED (phase-2 headroom) instead of an OOM before RUN_START."""


def _headroom(device: torch.device, need: int, M: int) -> None:
    """free = the driver's free bytes + this allocator's reserved-but-unallocated bytes (both usable)."""
    free, _ = torch.cuda.mem_get_info(device)
    free += int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device))
    if free < need:
        raise HeadroomError(f"phase-2 headroom: {free / 2**30:.2f} GiB free < 5U + 1 GiB = {need / 2**30:.2f} GiB "
                          f"at M {M}")


def check_point(lib: "_Lib", M: int, rule: str, device: torch.device, live: _Live) -> Dict[str, Any]:
    """One phase-2 check point in the order of DESIGN_stageC 5.3 step 4 (peak 5 U, U = M x 512 B). Raises on any
    failed check."""
    from generative_recommenders.ops.triton import triton_fp8 as tf8

    rt = _RT
    st = torch.cuda.current_stream(device)
    G = lib.plan(M, rt.num_cu(device))
    out: Dict[str, Any] = dict(M=M, grid=G, rule=rule)
    d = cancel_data(M, device, live)  # 2U
    # 2. torch's da1 through the production callable (use_fast_accum = _HSTU_FP8_FAST_ACCUM)  [3U]
    da1_t = tf8._scaled_mm(d["dh8"], d["w8t"].t(), d["sa"], d["sb"], out_dtype=torch.bfloat16)
    live.add("da1_t", da1_t)
    out["torch_mm"] = "triton_fp8._scaled_mm"
    # 3. S1+DA1 -> da1_x, dh1_x  [5U]; (a)
    da1_x = _canary_bf16(live, "da1_x", M, device)
    dh1_x = _canary_bf16(live, "dh1_x", M, device)
    part_x = _canary_f32(live, "part_x", (3, G, HID), device)
    rt.launch(lib, "emit", d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"], d["nw"], d["nb"],
              dh1_x, da1_x, part_x, M, G, st)
    lim = M if rule == "all" else max(0, M - SPLIT_TAIL) if rule == "dp_region" else 0
    bad, first = _diff_rows(da1_x, da1_t, 0, lim)
    if bad:
        raise RuntimeError(f"(a) S1+DA1 da1 != hipBLASLt (triton_fp8._scaled_mm) at M {M}: {bad} of {lim} gated rows "
                           f"differ (first {first})")
    out["da1_gated_rows"] = lim
    out["da1_reported_diff_rows"] = _diff_rows(da1_x, da1_t, lim, M)[0]
    del da1_t
    live.free("da1_t")  # 4U
    _canaries_ok(da1_x, M, "(e) da1_x")
    _canaries_ok(dh1_x, M, "(e) dh1_x")
    _written(part_x, "(e) S1+DA1 PART")
    # 4. PR #7 on S1+DA1's da1  [5U]; (b)
    dx_p, dw_p, db_p, dxs_p = _LB.ln_bwd(da1_x[:M], d["h1"], d["nw"], d["nb"], d["mean1"], d["rstd1"],
                                         is_swish=True, sum_dx=True)
    live.add("dx_p", dx_p)
    bad, first = _diff_rows(dh1_x, dx_p, 0, M)
    if bad:
        raise RuntimeError(f"(b) S1+DA1 dh1 != PR #7 ln_bwd(da1) at M {M}: {bad} rows differ (first {first})")
    if rule == "report":  # (f) the finals set
        fin_x = rt.fin(lib, fin_mode(), part_x, G, st)
        out["finals_use"] = finals_bound_check(d, da1_x, fin_x, G, M)
        out["finals_vs_pr7_max_abs"] = [float((a.double() - b.double()).abs().max().item())
                                        for a, b in zip(fin_x, (dw_p, db_p, dxs_p))]
        del fin_x
    del dx_p, dw_p, db_p, dxs_p, da1_x
    live.free("dx_p", "da1_x")  # 3U
    # 5. S1 -> dh1_s  [4U]; (c)
    dh1_s = _canary_bf16(live, "dh1_s", M, device)
    part_s = _canary_f32(live, "part_s", (3, G, HID), device)
    rt.launch(lib, "s1", d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"], d["nw"], d["nb"],
              dh1_s, None, part_s, M, G, st)
    bad, first = _diff_rows(dh1_s, dh1_x, 0, M)
    if bad:
        raise RuntimeError(f"(c) S1 dh1 != S1+DA1 dh1 at M {M}: {bad} rows differ (first {first})")
    nbad = int((part_s.view(torch.int32) != part_x.view(torch.int32)).sum().item())
    if nbad:
        raise RuntimeError(f"(c) S1 PART != S1+DA1 PART at M {M}: {nbad} words differ")
    _canaries_ok(dh1_s, M, "(e) dh1_s")
    _written(part_s, "(e) S1 PART")
    # the run's finalize on S1's PART into sentinel-filled outputs
    if fin_mode() == "pr7":
        fins = rt.fin(lib, "pr7", part_s, G, st, out=tuple(_canary_f32(live, f"fin{i}", (HID,), device)
                                                            for i in range(3)))
    else:
        fins = rt.fin(lib, "x2lb", part_s, G, st, out=_canary_f32(live, "fin0", (3, HID), device))
    for nm, t in zip(("dW", "dB", "dXS"), fins):
        _written(t, f"(e) finals {nm}")
    del dh1_x, part_x, fins
    live.free("dh1_x", "part_x", "fin0", "fin1", "fin2")  # 3U
    # 6. S1 at grid 7  [4U]; (d)
    dh1_7 = _canary_bf16(live, "dh1_7", M, device)
    part_7 = _canary_f32(live, "part_7", (3, STRESS_GRID, HID), device)
    rt.launch(lib, "s1", d["dh8"], d["w8t"], d["sa"], d["sb"], d["h1"], d["mean1"], d["rstd1"], d["nw"], d["nb"],
              dh1_7, None, part_7, M, STRESS_GRID, st)
    bad, first = _diff_rows(dh1_7, dh1_s, 0, M)
    if bad:
        raise RuntimeError(f"(d) S1 grid {STRESS_GRID} dh1 != grid {G} dh1 at M {M}: {bad} rows differ "
                           f"(first {first})")
    _canaries_ok(dh1_7, M, "(e) dh1_7")
    _written(part_7, "(e) grid-7 PART")
    del dh1_7, part_7, dh1_s, part_s, d
    live.free(*list(live.live))
    return out


def prebuild_check(device: torch.device) -> str:
    """Phase 2 (blas_warmup._prebuild_flydsl_ln_bwd, after the FlyDSL prebuild): the self-check of DESIGN_stageC 5.3
    on the rank's device, then arm. Returns the startup-preload token "x2lb=1|off|error". Never raises (a failure
    latches X2L-B off: one FAILED CLOSED line; PR #7 is never latched by it)."""
    global _LB, _P2_STATE
    if not requested():
        return ""
    if _BROKEN is not None:
        return "x2lb=off"
    if _P1_STATE != "ok" or _lib is None:
        phase2_not_run(f"phase 1 did not pass ({_P1_STATE or 'not run'})")
        return "x2lb=off"
    from generative_recommenders.ops.triton import triton_layer_norm as TLN

    if not TLN._wln_bwd_flydsl_enabled() or TLN._WLN_BWD_FLYDSL_BROKEN is not None:
        phase2_not_run("PR #7 (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) off or latched off")
        return "x2lb=off"
    t0 = time.perf_counter()
    live = _Live()
    points: List[Dict[str, Any]] = []
    try:
        if device.type != "cuda":
            raise RuntimeError(f"device {device} is not a GPU")
        with torch.cuda.device(device):
            from generative_recommenders.ops.flydsl import ln_bwd as lb

            _LB = lb
            for M, rule in CHECK_POINTS:
                _headroom(device, 5 * M * KD + HEADROOM_SLACK, M)
                points.append(check_point(_lib, M, rule, device, live))
            idx = _dev_index(device)
            _ZERO[idx] = _RT.make_zero(device)
            torch.cuda.synchronize(device)
    except HeadroomError as exc:
        _P2_STATE = "failed"
        latch_off("phase-2 headroom", exc)
        return "x2lb=error"
    except Exception as exc:
        _P2_STATE = "failed"
        latch_off("phase-2 self-check", exc)
        return "x2lb=error"
    _armed.add(idx)
    _P2_STATE = "ok"
    secs = time.perf_counter() - t0
    PREBUILD_INFO[idx] = dict(points=points, peak_bytes=live.peak, seconds=secs)
    big = [p for p in points if p["rule"] != "report"]
    fin = next(p for p in points if p["rule"] == "report")
    dp = ", ".join(f"M {p['M']} grid {p['grid']} beyond {p['da1_reported_diff_rows']} rows differ"
                   for p in big if p["rule"] == "dp_region")
    _say(f"{ARMED_PREFIX} pre-run_start self-check OK (cancel: M {M_DP} da1 == hipBLASLt all rows; M {M_A} / {M_B} "
         f"da1 == hipBLASLt on rows < M-{SPLIT_TAIL} ({dp}); dh1 == PR #7(da1) all rows; S1 == S1+DA1; grids "
         f"{'/'.join(str(p['grid']) for p in big)}/{STRESS_GRID}; canaries; partials within gamma at M {M_FIN} "
         f"(use {fin['finals_use']:.3g}, fin {fin_mode()}); peak transient {live.peak / 1e9:.2f} GB; phase 2 "
         f"{secs:.1f}s; device=cuda:{idx})")
    return "x2lb=1"


def reset_for_testing() -> None:
    """Test-only: forget the gate, the latch, the phases, the arming, the once-lines and the loaded library."""
    global _REQUESTED, _MLPS, _FIN, _BROKEN, _lib, _RT, _P1_STATE, _P2_STATE, _P2_SKIP, _LB
    _REQUESTED = _MLPS = _FIN = _BROKEN = None
    _lib = None
    _RT = _CudaRuntime()
    _P1_STATE = _P2_STATE = _P2_SKIP = None
    _LB = None
    _armed.clear()
    _ONCE.clear()
    _ZERO.clear()
    _PENDING.clear()
    PHASE1_INFO.clear()
    PREBUILD_INFO.clear()
    FUSED_CALLS.clear()
    DECLINES.clear()

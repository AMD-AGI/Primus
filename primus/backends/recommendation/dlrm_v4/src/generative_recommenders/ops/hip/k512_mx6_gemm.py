# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

# pyre-unsafe

"""MXFP6 forward K=512 GEMMs of the HSTU layers and the fused MXFP6 weight shadow, as hipcc-built HIP libraries.

Behind ``DLRM_HSTU_FP8_FWD_MXFP6=1`` (default off; a DECLARED PRECISION CHANGE). The integration adapter
``ops/hip/mx6_fwd.py`` (module ``$DLRM_HSTU_FP8_FWD_MXFP6_GEMM_MODULE``, default this one) calls::

    SUPPORTED_N                       (1024, 1536), plus 512 when track B's U source is present
    kernel(n) -> Kernel               build / load the library for N; RuntimeError when unavailable (latched)
    mx6_weight(weight, layout) -> (w6, ws, sb)
    k512_mx6(x6, xs, w6, ws, bias, sa, sb, M, N, out=None) -> out
    prebuild() -> bool                (PrebuildError on a failed build / self-check)

Credit: the GEMM kernel is Arbor P7's (pr7) iteration-24 MXFP6 prototype (``proto/k512mx6.cu``, pre-tiled "MX6-T"
operands, scaled MFMA; design pr7 DESIGN.md sections 3-9), packaged here by r14 iteration 49 (fp6fwd/gemm track A)
without changing its instruction stream; the fused weight shadow is new (r14). The U GEMM (N=512, bf16 + bias) is
r14 track B's library ``k512_mx6_u.cu`` (C ABI ``k512_mx6_u_bf16_bias``), dispatched from here when its source is
present.

Math (k512_mx6_gemm.cu header)::

    acc[m, n] = sum_k dequant(x6, xs)[m, k] * dequant(w6, ws)[n, k]       fp32 inside the MFMA, K = 512
    N in (1536, 1024): out e4m3 [M, N] = e4m3_rne(sat448(acc * (sa * sb) + float(bias[n])))
    N = 512 (U):       out bf16 [M, 512] = bf16_rne(acc * (sa * sb) + float(bias[n]))     (track B's library)

Operands (uint8, all contiguous): x6 [T, 4, 12288] MX6-T and xs [T, 4, 512] MX6-A-v1 with T = ceil(M / 128) (the
r14 LN producer ``ln_store_mx6`` writes both); w6 [N/256, 2, 4, 12288] MX6-T weight items with the k512 output-column
permutation, ws [N/32, 4, 4] = [N/32][q][S] 2-D 32x32 E8M0 bytes (``mx6_weight`` writes both); bias bf16 [N]; sa / sb
one fp32 value each on the device.

WEIGHT SHADOW (``mx6_weight``). ONE kernel launch per refresh turns the uvqk weight [512, C] (fp32 or bf16, any
strides) and a column layout (``((512, 2048),)`` V/Q/K, ``((512, 1024), (1536, 2048))`` targets V/K, ``((0, 512),)``
U; the ``triton_fp8`` layout convention) into

* ``sb`` fp32 [1, 1]: bitwise ``triton_fp8._get_weight_shadow(weight, layout)[1]`` (calibrated_scale with margin 1.0:
  amax * fp32(1/448), amax clamped at 1e-12; NaN if the slice holds a NaN);
* ``w6`` / ``ws``: byte for byte pr7 ``proto.py`` ``tile_W(quant_W(W^T / sb))`` with ``W^T / sb`` the fp32 IEEE
  quotient the e4m3 shadow quantizes (``quantize_with_scale``). This is the "relative to the FP8 shadow's scale_b"
  contract of pr7 DESIGN section 3 that r14's iter49 e2e check validated end to end (C = acc * (sa * sb) + bias).

The result is cached per (weight identity, device, shape, stride, dtype, layout) and served only while the SAME tensor
is alive with the same ``data_ptr()`` and ``_version``: an optimizer step (any in-place update bumps ``_version``) or
a storage swap refreshes it on the next call. A writer that changes the weight without bumping ``_version`` (raw
pointers, ``.data`` writes) leaves this shadow stale exactly like the FP8 shadows; the adapter's SHADOW_AUDIT covers
that class.

Build: ``/opt/rocm/bin/hipcc`` (the k512_fp8_gemm.py flags), once per library per process, into a per-process directory
under ``/tmp/dlrm_hstu_mx6_k512``: ``libk512_mx6_gemm_N{1536,1024}.so`` (``-DHK_N``), ``libk512_mx6_wshadow.so``
(``-DK512_MX6_WSHADOW``) and, when ``k512_mx6_u.cu`` is present next to this file (or named by
``DLRM_HSTU_FP8_FWD_MXFP6_U_SRC``), ``libk512_mx6_u.so``. ``prebuild()`` (before RUN_START) builds them in parallel and
self-checks them; otherwise each builds on first use. A failed build or self-check is latched for the process.
Optional environment: ``DLRM_HSTU_FP8_FWD_MXFP6_GEMM_SRC`` = another ``.cu`` with this file's ABI (A/B a candidate).

RING6 (``DLRM_HSTU_FP8_FWD_MXFP6_RING6``, yambda_5b.gin item (96); default off; BITWISE vs the shipped kernels). "1"
(read once per process, at the first library key) adds ``-DK512MX6_RING=6`` (``hipcc_command``'s ``extra_flags``) to
the GEMM N=1536 / 1024 and U builds: k512_mx6_gemm.cu / k512_mx6_u.cu then keep six 12 KiB LDS slots per workgroup and
issue the activation item two K steps ahead (the same MFMAs on the same bytes in the same order; r14 iter64 RING6).
Those libraries get their own file names (``libk512_mx6_gemm_N{1536,1024}_ring6.so``, ``libk512_mx6_u_ring6.so``),
keys (``(N, "ring6")`` in the library table, the failure latches and the build locks) and a ``pid<n>_ring6_`` build
directory, so a library built with one setting is never served for the other. The weight shadow is built exactly as
without the knob (its cache entries serve both settings: the GEMM operand formats are unchanged). Routing check, both
ways: with the knob on, each RING6 library must report the RING6 build in ``k512_mx6_gemm_impl()`` /
``k512_mx6_u_impl()`` (the secondary variant RING6S is refused); with it off, a GEMM / U library whose impl string
reports RING6 at all (a define injected outside this loader, e.g. ``HIPCC_COMPILE_FLAGS_APPEND=-DK512MX6_RING=6``, or
a ``*_SRC`` source that defaults to RING6) is refused the same way (latched, never served). prebuild() prints ONE
rank-tagged "[k512-mx6] RING6 ARMED: ..." line when on; in both settings the per-kernel ``lds=`` of its "prebuilt:"
line (each library's ``LDS Size [bytes/block]`` compiler remark) marks the arm: 73728 (six 12 KiB slots) for the RING6
builds, 61440 (five) for the shipped GEMM / U builds. Off (absent or any value but "1"): every hipcc command line, file
name, key and log line is the shipped one.

BSR (``DLRM_HSTU_FP8_FWD_MXFP6_BSR``, yambda_5b.gin item (110); default off; BITWISE vs the shipped kernels). The
persistent MXFP6 K=512 forward GEMM with REGISTER-resident weights (``k512_mx6_bsr.cu`` next to this file, no
environment override; r14 iter68 design O-D2, the lab arm bsr1; the shipped C ABI and impl symbols) for the sites the
knob selects (``parse_bsr``): "1" / "true" every site (N=1536 VQK, N=1024 VK and N=512 U) at every topology; "1n"
every site on a single-node job only (``single_node``: WORLD_SIZE == LOCAL_WORLD_SIZE, else the launcher's
GPUS_PER_NODE, the two agreeing when both are set, cross-checked against the visible device count and an initialized
process group; missing / unparsable / conflicting = multi-node), else off with
one rank-tagged "[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE=<n> (LOCAL_WORLD_SIZE=<m>)" line (at 2n / 4n a
resident RCCL CTA can block the persistent, statically scheduled BSR workgroups); a comma list of 1536 / 1024 / 512
those sites at every topology; unset, "", "0"
or "false" off (no BSR line at all); anything else off plus one rank-tagged WARNING line. U is a BSR site only when
the U site exists (U_N in SUPPORTED_N). Read once per process, by prebuild(). A BSR library is built with the
shipped hipcc_command, ``-DHK_N=<N> -DHK_K=512`` (U: ``-DHK_N=512 -DBSR_U=1``) plus BSR_SWITCH (never RING6's define),
under its own file name ``libk512_mx6_bsr_N<N>.so`` and key ``(N, "bsr")`` (library table, failure latches, build
locks) in the process build directory; the site's shipped library (RING6 per its own knob) is built as without the knob
and stays the site's fallback and bitwise reference. Routing check: a BSR library's impl string must contain "BSR O-D2"
and no "RING6" (RING6's checks do not apply to BSR keys). prebuild() builds the BSR libraries in the same parallel pool
and, after the shipped self-checks, self-checks each one before RUN_START (``_selfcheck_bsr``: zeros into a NaN
prefill, then the full output plus two NaN-prefilled guard rows bitwise equal to the site's shipped library on
deterministic synthetic MX6 operands at every M of BSR_SELFCHECK_M; synthetic only: no data, model state or RNG). A
site that passes is ARMED: kernel(N) serves its BSR library from then on; one rank-tagged "[k512-mx6] BSR ARMED: ..."
line follows the "prebuilt:" line, which lists each armed BSR library's resources (``lds=102400``). Any recoverable
BSR failure (build, load, impl routing, self-check mismatch, a Python exception) latches BSR for that site only, with
one rank-tagged "[k512-mx6] BSR DECLINED N=<n>: <reason> (serving the shipped kernel)" line: the site keeps its shipped
library; BSR never fails the prebuild and never sends a site to the FP8 path. A device fault or a hang inside a BSR
kernel during the self-check aborts or hangs the job before RUN_START, as for any kernel (r14 lab jobs 74953 / 75158
saw neither). The knob, and with it its env kill switch, is read at startup by prebuild(), like every knob; prebuild()
runs once per process, on the main thread before training, so the BSR state takes no lock of its own. Without a
prebuild() nothing is armed and every site serves its shipped library (no BSR build, no BSR line). Off: every hipcc
command line, file name, key, latch and log line OF THIS LOADER is the shipped one (the env bootstrap's own binding
line, "env bootstrap: DLRM_HSTU_FP8_FWD_MXFP6_BSR=<v>", is outside this module).

Every launch is asynchronous on ``torch.cuda.current_stream()``; the scales are read on the device; nothing here
synchronizes the host after the build. Operands are validated on the host from metadata only (dtype, shape,
contiguity, alignment, device): a violation raises ValueError before any launch.
"""

from __future__ import annotations

import concurrent.futures
import ctypes
import hashlib
import os
import re
import subprocess
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

import torch

K: int = 512
TILE: int = 128
KSTEPS: int = K // 128
X6_ITEM_BYTES: int = 12288  # one (128-row tile, K step) MX6-T item: HEAD 8 KiB + TAIL 4 KiB
XS_ITEM_BYTES: int = 512  # one (tile, K step) MX6-A-v1 scale block
GEMM_N: Tuple[int, ...] = (1024, 1536)  # built from k512_mx6_gemm.cu (e4m3 + bias epilogue)
U_N: int = 512  # track B's library (bf16 + bias epilogue)
HIPCC: str = "/opt/rocm/bin/hipcc"
KNOB: str = "DLRM_HSTU_FP8_FWD_MXFP6"
_SRC_ENV: str = KNOB + "_GEMM_SRC"
_U_SRC_ENV: str = KNOB + "_U_SRC"
# RING6 (gin item (96), default off): the knob, the define it adds to the GEMM N=1536 / 1024 and U builds (never to
# the weight shadow), the tag of those libraries' file names, keys and build directory, and what their impl strings
# must contain (k512_mx6_gemm.cu / k512_mx6_u.cu built with K512MX6_RING == 6).
RING6_KNOB: str = KNOB + "_RING6"
RING6_DEFINE: str = "-DK512MX6_RING=6"
RING6_TAG: str = "ring6"
RING6_IMPL: Dict[str, str] = {"gemm": "RING6 6-slot 72 KiB LDS ring", "u": "RING6 6 x 12 KiB LDS ring"}
RING6S_IMPL: str = "RING6S"  # the secondary variant RING6S (its own define): never built by this loader
RING6_ANY: str = "RING6"  # in every RING6 / RING6S impl string and in no shipped one (the knob-off routing check)
# BSR (gin item (110), default off): the knob, the source (next to this file; no environment override), the switch
# defines of every BSR build (lab arm bsr1 of r14 iter68: 16 LDS ring slots, activation scales through the ring, strict
# ring waits), the tag of the BSR library keys, what their impl strings must contain, the sites in knob order (VQK,
# VK, U; U only when the U site exists), and the M and the fixed sa of the bitwise self-check.
BSR_KNOB: str = KNOB + "_BSR"
BSR_MODE_1N: str = "1n"  # every site, single-node jobs only (WORLD_SIZE == LOCAL_WORLD_SIZE); the cand_v1b binding
BSR_SOURCE_NAME: str = "k512_mx6_bsr.cu"
BSR_SWITCH: Tuple[str, ...] = ("-DBSR_S=16", "-DBSR_SCL=1", "-DBSR_VMS=1")
BSR_TAG: str = "bsr"
BSR_IMPL: str = "BSR O-D2"
BSR_SITES: Tuple[int, ...] = (1536, 1024, 512)
BSR_SELFCHECK_M: Tuple[int, ...] = (64, 257, 4096, 4219, 65536, 65665)  # 64-row-aligned and tail sizes
BSR_SELFCHECK_SA: float = 0.8125
U_SOURCE_NAME: str = "k512_mx6_u.cu"
U_SYMBOL: str = "k512_mx6_u_bf16_bias"
WSHADOW_SYMBOL: str = "k512_mx6_wshadow"
WSHADOW_P1: int = 64  # phase-1 workgroups of the shadow kernel (k512mx6w::P1; checked against the library)
WSHADOW_KEY: str = "wshadow"
PERM_K512: int = 0  # w6 output-column permutation of the k512 MX6 kernels (VQK, VK; track B's U uses the same)
_BUILD_ROOT: Path = Path("/tmp/dlrm_hstu_mx6_k512")
_BUILD_TIMEOUT_S: int = 600
_INT32_MAX: int = 2**31 - 1
_STALE_ERROR_BASE: int = 100000  # rc = 100000 + e: the launch was clean, an EARLIER HIP call left error e
_MAX_SHADOW_N: int = 1 << 20
# fp32(1/448): what aten multiplies by for "x / 448.0" on a CUDA tensor (k512_mx6_gemm.cu uses 1.0f / 448.0f)
INV_FP8_MAX_F32: float = float(torch.tensor(1.0 / 448.0, dtype=torch.float32).item())

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)
_Layout = Tuple[Tuple[int, int], ...]


class PrebuildError(RuntimeError):
    """prebuild(): a library failed to build, load or self-check (latched; kernel() then raises for it)."""


@dataclass(frozen=True)
class Kernel:
    """One loaded library: a GEMM for N (1536 / 1024 / 512) or the weight shadow (n = 0)."""

    key: object
    n: int
    path: str
    fn: object  # ctypes function pointer
    impl: str
    source: str
    source_sha256: str
    build_s: float
    resources: str
    ring6: bool = False  # built with -DK512MX6_RING=6 (and its impl string reports the RING6 build)
    bsr: bool = False  # a BSR library (key (N, "bsr"), k512_mx6_bsr.cu; its impl string reports "BSR O-D2")


def source_path() -> Path:
    override = os.environ.get(_SRC_ENV)
    return Path(override) if override else Path(__file__).with_name("k512_mx6_gemm.cu")


def u_source_path() -> Optional[Path]:
    """Track B's U GEMM source (``k512_mx6_u.cu`` next to this file, or ``$DLRM_HSTU_FP8_FWD_MXFP6_U_SRC``); None
    when absent (N = 512 is then not in SUPPORTED_N and U stays on the shipped FP8 GEMM)."""
    override = os.environ.get(_U_SRC_ENV)
    path = Path(override) if override else Path(__file__).with_name(U_SOURCE_NAME)
    return path if path.is_file() else None


def bsr_source_path() -> Path:
    """The BSR source: k512_mx6_bsr.cu next to this file (no environment override)."""
    return Path(__file__).with_name(BSR_SOURCE_NAME)


SUPPORTED_N: Tuple[int, ...] = ((U_N,) if u_source_path() is not None else ()) + GEMM_N

_lock = threading.Lock()  # guards _build_dir and the lock table
_key_locks: Dict[object, threading.Lock] = {}
_kernels: Dict[object, Kernel] = {}
_failures: Dict[object, str] = {}
_build_dir: Optional[Path] = None
_prebuilt: bool = False
_ring6: Optional[bool] = None  # the RING6 knob, read once (lazily, at the first library key)
_ring6_override: Optional[bool] = None  # tests only (set_ring6_for_testing)
_ring6_logged: bool = False  # the RING6 ARMED line: once per process
_bsr: Optional[FrozenSet[int]] = None  # the BSR knob's sites, read once (lazily, by prebuild())
_bsr_raw: Optional[str] = None  # the raw knob value that was parsed (the ARMED line quotes it)
_bsr_note: str = ""  # mode 1n on a single node: the topology the ARMED line reports
_bsr_override: Optional[str] = None  # tests only (set_bsr_for_testing): parsed instead of the environment
_bsr_armed: Dict[int, Kernel] = {}  # N -> the BSR library prebuild() armed for the site (kernel(N) serves it)
_bsr_declined: Dict[int, str] = {}  # N -> why BSR was declined for the site (its DECLINED line: once per process)
_bsr_logged: bool = False  # the BSR ARMED line: once per process


def _print(message: str) -> None:
    print(f"[k512-mx6] {message}", flush=True)


def _rank_print(message: str) -> None:
    """One rank-tagged line on every rank (ops.utils.arm_print_all_ranks): the BSR ARMED / DECLINED / WARNING lines."""
    from generative_recommenders.ops.utils import arm_print_all_ranks

    arm_print_all_ranks(message)


def ring6_requested() -> bool:
    """DLRM_HSTU_FP8_FWD_MXFP6_RING6 == "1" (gin item (96)). Read once, lazily: the first library key is taken after
    the gin env bootstrap, and one process never mixes the two settings. Absent, "0" or any other value: False (every
    library is the shipped build)."""
    global _ring6
    if _ring6_override is not None:
        return _ring6_override
    if _ring6 is None:
        _ring6 = os.environ.get(RING6_KNOB, "0") == "1"
    return _ring6


def set_ring6_for_testing(value: Optional[bool]) -> None:
    """Tests only: force the RING6 knob (None = read the environment again). Libraries already loaded stay in the table
    under their own keys; one built with the other setting is never served."""
    global _ring6, _ring6_override, _ring6_logged
    _ring6_override = value
    _ring6 = None
    _ring6_logged = False


def parse_bsr(raw: Optional[str]) -> Tuple[FrozenSet[int], bool, bool]:
    """(sites, single_node_only, valid) of a raw DLRM_HSTU_FP8_FWD_MXFP6_BSR value (gin item (110); surrounding blanks
    and case ignored): None (unset), "", "0" or "false" -> off; "1" or "true" -> every site of BSR_SITES at every
    topology; "1n" -> every site, single-node jobs only (bsr_sites decides); a comma list of 1536 / 1024 / 512 ->
    those sites at every topology; anything else -> off with valid=False (one WARNING line)."""
    value = (raw or "").strip().lower()
    if value in ("", "0", "false"):
        return frozenset(), False, True
    if value in ("1", "true"):
        return frozenset(BSR_SITES), False, True
    if value == BSR_MODE_1N:
        return frozenset(BSR_SITES), True, True
    items = [item.strip() for item in value.split(",")]
    if all(re.fullmatch(r"[0-9]+", item) and int(item) in BSR_SITES for item in items):
        return frozenset(int(item) for item in items), False, True
    return frozenset(), False, False


def _local_ranks() -> Tuple[Optional[str], str]:
    """(raw value, its source) of the ranks of one node: LOCAL_WORLD_SIZE, else GPUS_PER_NODE (scripts/launch_slurm.sh
    exports GPUS_PER_NODE and WORLD_SIZE = NNODES * GPUS_PER_NODE but no LOCAL_WORLD_SIZE; train_ranker spawns
    GPUS_PER_NODE ranks per node with mp.start_processes, which sets no environment)."""
    if os.environ.get("LOCAL_WORLD_SIZE") is not None:
        return os.environ.get("LOCAL_WORLD_SIZE"), "LOCAL_WORLD_SIZE"
    return os.environ.get("GPUS_PER_NODE"), "GPUS_PER_NODE"


def single_node() -> Tuple[bool, str]:
    """(True, "") for a single-node job: WORLD_SIZE == the ranks of one node (_local_ranks: LOCAL_WORLD_SIZE, else the
    launcher's GPUS_PER_NODE; when both are set they must agree), the visible device count equals them (when a GPU is
    available), and an initialized process group's world size equals them too. Otherwise (False, why) with why "" for
    an ordinary multi-node job; a missing, unparsable, non-positive or conflicting value, a device-count mismatch or a
    disagreeing process group counts as multi-node (why says which)."""
    ws_raw, (lws_raw, source) = os.environ.get("WORLD_SIZE"), _local_ranks()
    lws_env, gpn_env = os.environ.get("LOCAL_WORLD_SIZE"), os.environ.get("GPUS_PER_NODE")
    try:
        ws, lws = int(ws_raw or ""), int(lws_raw or "")
        gpn = int(gpn_env) if lws_env is not None and gpn_env is not None else None
    except ValueError:
        return False, "WORLD_SIZE or LOCAL_WORLD_SIZE / GPUS_PER_NODE missing or unparsable: treated as multi-node"
    if gpn is not None and gpn != lws:  # e.g. a stale LOCAL_WORLD_SIZE next to the launcher's GPUS_PER_NODE
        return False, f"conflicting LOCAL_WORLD_SIZE/GPUS_PER_NODE ({lws} != {gpn}): treated as multi-node"
    if ws < 1 or lws < 1:
        return False, "WORLD_SIZE or LOCAL_WORLD_SIZE / GPUS_PER_NODE not positive: treated as multi-node"
    if ws != lws:
        return False, ""
    if torch.cuda.is_available() and int(torch.cuda.device_count()) != lws:
        return False, (f"torch.cuda.device_count() {int(torch.cuda.device_count())} != {source} {lws}: treated as "
                       "multi-node")
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and int(dist.get_world_size()) != lws:
        return False, (f"torch.distributed world size {int(dist.get_world_size())} != {source} {lws}: treated as "
                       "multi-node")
    return True, ""


def bsr_sites() -> FrozenSet[int]:
    """The sites (N) the BSR knob selects (parse_bsr), read once per process, lazily: prebuild() reads it before its
    parallel builds, after the gin env bootstrap (and the process group). Mode "1n" on a job that is not single-node
    (single_node) selects nothing and prints one rank-tagged "BSR OFF by gin mode 1n" line; an invalid value selects
    nothing and prints one rank-tagged WARNING line; off ("0" / unset) prints nothing."""
    global _bsr, _bsr_raw, _bsr_note
    if _bsr is not None:
        return _bsr
    raw = _bsr_override if _bsr_override is not None else os.environ.get(BSR_KNOB)
    sites, one_node, valid = parse_bsr(raw)
    note, off_line = "", None
    if one_node:
        single, why = single_node()
        lws_raw, source = _local_ranks()
        ws = os.environ.get("WORLD_SIZE", "<unset>")
        lws = ("<unset>" if lws_raw is None else lws_raw) + ("" if source == "LOCAL_WORLD_SIZE" else f" from {source}")
        if single:
            note = f" (mode 1n: single node, WORLD_SIZE={ws} LOCAL_WORLD_SIZE={lws})"
        else:
            sites = frozenset()
            off_line = (f"[k512-mx6] BSR OFF by gin mode 1n at WORLD_SIZE={ws} (LOCAL_WORLD_SIZE={lws})"
                        + (f": {why}" if why else ""))
    with _lock:
        first = _bsr is None
        if first:
            _bsr, _bsr_raw, _bsr_note = sites, raw, note
    if first and not valid:
        _rank_print(f"[k512-mx6] WARNING: {BSR_KNOB}={raw!r} is not 0 / 1 / 1n / false / true or a comma list of "
                    f"{', '.join(str(n) for n in BSR_SITES)}: BSR off (serving the shipped kernels)")
    if first and off_line is not None:
        _rank_print(off_line)
    assert _bsr is not None
    return _bsr


def set_bsr_for_testing(value: Optional[str]) -> None:
    """Tests only: parse `value` as the BSR knob instead of the environment (None = read the environment again).
    Clears the armed / declined sites and the once-per-process ARMED line; libraries already loaded stay in the table
    under their own keys (a site is served BSR again only after the next prebuild() arms it)."""
    global _bsr, _bsr_raw, _bsr_note, _bsr_override, _bsr_logged
    _bsr_override = value
    _bsr = None
    _bsr_raw = None
    _bsr_note = ""
    _bsr_logged = False
    _bsr_armed.clear()
    _bsr_declined.clear()


def _bsr_key(n: int) -> Tuple[int, str]:
    """The key site N's BSR library is built, loaded, latched and served under: (N, "bsr")."""
    return (int(n), BSR_TAG)


def _is_bsr_key(key: object) -> bool:
    return isinstance(key, tuple) and len(key) == 2 and key[1] == BSR_TAG


def _ring6_build(key: object) -> bool:
    """True when the library of `key` is built as RING6: the knob is on and `key` is a GEMM N (1536 / 1024) or U (512).
    The weight shadow never is."""
    return (key in GEMM_N or key == U_N) and ring6_requested()


def _lib_key(key: object) -> object:
    """The key a library is built, loaded, latched and served under (_kernels, _failures, the per-key build locks):
    `key` itself for the shipped build, (key, "ring6") for a RING6 build."""
    return (key, RING6_TAG) if _ring6_build(key) else key


def _lib_name(stem: str, key: object) -> str:
    """The library file name: <stem>.so, or <stem>_ring6.so for a RING6 build."""
    return f"{stem}_{RING6_TAG}.so" if _ring6_build(key) else f"{stem}.so"


def _extra_flags(key: object) -> list:
    """hipcc_command's extra_flags for `key`: [-DK512MX6_RING=6] for a RING6 build; none otherwise (the shipped
    command, unchanged)."""
    return [RING6_DEFINE] if _ring6_build(key) else []


def hipcc_command(source: Path, output: Path, defines: Sequence[str] = (), *, extra_flags: Sequence[str] = ()) -> list:
    """The exact hipcc invocation: k512_fp8_gemm.py's flags (iterative-ilp scheduler, -O3, -ffp-contract=off)."""
    return [
        HIPCC,
        str(source),
        "--offload-arch=gfx950",
        "-std=c++20",
        "-w",
        "-O3",
        "-ffp-contract=off",
        "-I/opt/rocm/include/hip",
        *defines,
        "-mllvm",
        "-amdgpu-sched-strategy=iterative-ilp",
        "-shared",
        "-fPIC",
        "-Rpass-analysis=kernel-resource-usage",
        *extra_flags,
        "-o",
        str(output),
    ]


def _process_build_dir() -> Path:
    global _build_dir
    tag = f"{RING6_TAG}_" if ring6_requested() else ""  # a RING6 process builds into pid<n>_ring6_*
    with _lock:
        if _build_dir is None:
            _BUILD_ROOT.mkdir(parents=True, exist_ok=True)
            _build_dir = Path(tempfile.mkdtemp(prefix=f"pid{os.getpid()}_{tag}", dir=str(_BUILD_ROOT)))
        return _build_dir


def _key_lock(key: object) -> threading.Lock:
    with _lock:
        lk = _key_locks.get(key)
        if lk is None:
            lk = _key_locks[key] = threading.Lock()
        return lk


def resource_summary(build_log: str) -> str:
    """One entry per kernel from the -Rpass-analysis=kernel-resource-usage remarks (k512_fp8_gemm.py's parser)."""
    per_kernel: Dict[str, Dict[str, str]] = {}
    current = None
    location = r"(?:\s+\S+:\d+:\d+:)?"
    # ROCm 10.2 (AMD clang 24), bare: "<loc>: Function Name: X" / "<loc>:     <Key>: <v>" (no "remark:", no "[-Rpass")
    bare = r"\S+:\d+:\d+:\s+"
    bare_key = (r"(TotalSGPRs|VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|Occupancy \[waves/SIMD\]|"
                r"SGPRs Spill|VGPRs Spill|LDS Size \[bytes/block\])")
    for line in build_log.splitlines():
        m = re.search(r"remark:" + location + r"\s+Function Name:\s+(\S+)", line)
        m = m or re.fullmatch(bare + r"Function Name:\s+(\S+)\s*", line)
        if m:
            current = m.group(1)
            per_kernel[current] = {}
            continue
        m = re.search(r"remark:" + location + r"\s+([A-Za-z][A-Za-z /\[\]]*?):\s+(\S+)\s+\[-Rpass", line)
        m = m or re.fullmatch(bare + bare_key + r":\s+(\S+)\s*", line)
        if m and current is not None:
            per_kernel[current][m.group(1).strip()] = m.group(2)
    keep = (("VGPRs", "vgpr"), ("AGPRs", "agpr"), ("VGPRs Spill", "vspill"), ("SGPRs Spill", "sspill"),
            ("ScratchSize [bytes/lane]", "scratch"), ("Occupancy [waves/SIMD]", "occ"), ("LDS Size [bytes/block]", "lds"))
    parts = []
    for name, fields in per_kernel.items():
        m = re.match(r"_Z(\d+)(\w+)", name)
        short = m.group(2)[: int(m.group(1))] if m else name[:40]
        kv = " ".join(f"{label}={fields[k]}" for k, label in keep if k in fields)
        parts.append(f"{short}: {kv}")
    return "; ".join(parts)


def _spec(key: object) -> Tuple[Path, str, list, str, str]:
    """(source, library file name, -D defines, entry symbol, impl symbol) of a library key."""
    if key == WSHADOW_KEY:
        return source_path(), "libk512_mx6_wshadow.so", ["-DK512_MX6_WSHADOW"], WSHADOW_SYMBOL, "k512_mx6_wshadow_impl"
    if key in GEMM_N:
        return (source_path(), _lib_name(f"libk512_mx6_gemm_N{key}", key), [f"-DHK_N={key}", f"-DHK_K={K}"],
                f"k512_mx6_e4m3_bias_N{key}", "k512_mx6_gemm_impl")
    if key == U_N:
        src = u_source_path()
        if src is None:
            raise RuntimeError(
                f"MX6 U library absent: no {U_SOURCE_NAME} next to {Path(__file__).name} and ${_U_SRC_ENV} unset "
                "(U stays on the shipped FP8 GEMM)")
        return src, _lib_name("libk512_mx6_u", key), [], U_SYMBOL, "k512_mx6_u_impl"
    if _is_bsr_key(key) and key[0] in BSR_SITES:  # (N, "bsr"): k512_mx6_bsr.cu, the shipped entry / impl symbols
        n = key[0]
        src = bsr_source_path()
        if not src.is_file():
            raise RuntimeError(f"BSR source absent: no {BSR_SOURCE_NAME} next to {Path(__file__).name}")
        if n == U_N:
            return (src, f"libk512_mx6_bsr_N{n}.so", [f"-DHK_N={n}", "-DBSR_U=1", *BSR_SWITCH], U_SYMBOL,
                    "k512_mx6_u_impl")
        return (src, f"libk512_mx6_bsr_N{n}.so", [f"-DHK_N={n}", f"-DHK_K={K}", *BSR_SWITCH],
                f"k512_mx6_e4m3_bias_N{n}", "k512_mx6_gemm_impl")
    raise RuntimeError(f"N={key} is not built (supported: {SUPPORTED_N})")


def _build(key: object) -> Tuple[Path, Path, float, str]:
    source, libname, defines, _, _ = _spec(key)
    out_dir = _process_build_dir()
    output = out_dir / libname
    temporary = out_dir / f".{libname}.{os.getpid()}.tmp.so"
    log_path = out_dir / f"build_{libname}.log"
    command = hipcc_command(source, temporary, defines, extra_flags=_extra_flags(key))
    t0 = time.perf_counter()
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=_BUILD_TIMEOUT_S, check=False)
    except FileNotFoundError as error:
        raise RuntimeError(f"hipcc not found ({HIPCC})") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"hipcc timed out after {_BUILD_TIMEOUT_S}s") from error
    seconds = time.perf_counter() - t0
    log_path.write_text(" ".join(command) + "\n" + result.stdout)
    if result.returncode != 0 or not temporary.exists():
        lines = result.stdout.strip().splitlines()
        errors = [line.strip() for line in lines if "error" in line.lower()]
        tail = " | ".join((errors or lines)[:3])
        raise RuntimeError(f"hipcc rc={result.returncode} for {libname} (log {log_path}): {tail[:400]}")
    os.replace(temporary, output)
    return source, output, seconds, resource_summary(result.stdout)


def _load(key: object) -> Kernel:
    source, path, seconds, resources = _build(key)
    _, _, _, symbol, impl_symbol = _spec(key)
    blob = source.read_bytes()
    library = ctypes.CDLL(str(path))
    fn = getattr(library, symbol)
    fn.restype = ctypes.c_int
    if key == WSHADOW_KEY:
        fn.argtypes = ([ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
                       + [ctypes.c_int] * 8 + [ctypes.c_int] + [ctypes.c_void_p] * 5)
        p1_fn = getattr(library, "k512_mx6_wshadow_p1", None)
        if p1_fn is not None:
            p1_fn.restype = ctypes.c_int
            p1_fn.argtypes = []
            if int(p1_fn()) != WSHADOW_P1:
                raise RuntimeError(f"{path}: k512_mx6_wshadow_p1() = {int(p1_fn())} != WSHADOW_P1 = {WSHADOW_P1}")
    else:
        # GEMM e4m3 and track B's U share one C ABI: (x6, xs, w6, ws, out, bias, sa, sb, M, stream)
        fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int, ctypes.c_void_p]
    impl = "?"
    impl_fn = getattr(library, impl_symbol, None)
    if impl_fn is not None:
        impl_fn.restype = ctypes.c_char_p
        impl_fn.argtypes = []
        impl = (impl_fn() or b"?").decode(errors="replace")
    ring6 = _ring6_build(key)
    if ring6:  # routing check: the library must BE the RING6 build (a source without the switch ignores the define)
        want = RING6_IMPL["u" if key == U_N else "gemm"]
        if want not in impl or RING6S_IMPL in impl:
            raise RuntimeError(f"RING6 routing check failed: {path.name} {impl_symbol}() = {impl!r} (a RING6 build "
                               f"contains {want!r} and not {RING6S_IMPL!r}; source {source})")
    elif (key in GEMM_N or key == U_N) and RING6_ANY in impl:  # the reverse: knob off, a GEMM / U library is shipped
        raise RuntimeError(f"RING6 routing check failed: {RING6_KNOB} is off but {path.name} {impl_symbol}() = "
                           f"{impl!r} reports a RING6 build (a shipped build contains no {RING6_ANY!r}; "
                           f"-DK512MX6_RING=6 injected outside the loader, e.g. HIPCC_COMPILE_FLAGS_APPEND, or a "
                           f"source defaulting to RING6; source {source})")
    elif _is_bsr_key(key) and (BSR_IMPL not in impl or RING6_ANY in impl):  # the library must BE the BSR build
        raise RuntimeError(f"BSR routing check failed: {path.name} {impl_symbol}() = {impl!r} (a BSR build contains "
                           f"{BSR_IMPL!r} and no {RING6_ANY!r}; source {source})")
    bsr = _is_bsr_key(key)
    n = key if isinstance(key, int) else (key[0] if isinstance(key, tuple) and bsr else 0)
    return Kernel(key=key, n=n, path=str(path), fn=fn, impl=impl, source=str(source),
                  source_sha256=hashlib.sha256(blob).hexdigest(), build_s=seconds, resources=resources, ring6=ring6,
                  bsr=bsr)


def _get(key: object) -> Kernel:
    lkey = _lib_key(key)  # (key, "ring6") for a RING6 build: never the shipped library's entry, and vice versa
    cached = _kernels.get(lkey)
    if cached is not None:
        return cached
    with _key_lock(lkey):  # per library, so prebuild() compiles them in parallel
        cached = _kernels.get(lkey)
        if cached is not None:
            return cached
        if lkey in _failures:
            raise RuntimeError(_failures[lkey])
        try:
            loaded = _load(key)
        except Exception as error:
            what = "weight shadow" if key == WSHADOW_KEY else f"GEMM N={key}"
            if isinstance(key, tuple) and _is_bsr_key(key):
                what = f"BSR GEMM N={key[0]}"
            _failures[lkey] = f"K=512 MXFP6 {what} library unavailable: {error}"
            raise RuntimeError(_failures[lkey]) from error
        _kernels[lkey] = loaded
        return loaded


def kernel(n: int) -> Kernel:
    """The loaded GEMM library for N (built on first use). RuntimeError if N is not supported or the library cannot
    be built / loaded; the failure is latched for the process. BSR (gin item (110)): the site's BSR library once
    prebuild() armed it (built, routed and self-checked; never built here), else the shipped one as above."""
    n = int(n)
    if n not in SUPPORTED_N:
        if n == U_N:
            _spec(U_N)  # raises the "U library absent" reason
        raise RuntimeError(f"N={n} is not built (supported: {SUPPORTED_N})")
    bsr = _bsr_armed.get(n)  # empty unless prebuild() armed BSR for a site the knob selects
    if bsr is not None and n in bsr_sites():
        return bsr
    return _get(n)


def wshadow_kernel() -> Kernel:
    """The loaded weight-shadow library (built on first use; failure latched)."""
    return _get(WSHADOW_KEY)


def failure(n: object) -> Optional[str]:
    return _failures.get(_lib_key(n))


# ----------------------------------------------------------------------------------------------- GEMM launch
def _stream_handle(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _rc_message(rc: int) -> str:
    if rc >= _STALE_ERROR_BASE:
        return f"clean launch, but an EARLIER HIP call left hipError {rc - _STALE_ERROR_BASE} on this thread"
    return f"hipError {rc}"


def ineligible_reason(x6: torch.Tensor, xs: torch.Tensor, w6: torch.Tensor, ws: torch.Tensor, bias: torch.Tensor,
                      sa: torch.Tensor, sb: torch.Tensor, m: int, n: int, out: Optional[torch.Tensor]) -> Optional[str]:
    """None when k512_mx6 can take these operands, else the reason (metadata only; no host sync)."""
    if _FP8_E4M3 is None:
        return "torch has no float8_e4m3fn"
    if not isinstance(m, int) or isinstance(m, bool) or m < 0 or m > _INT32_MAX:
        return f"M={m!r} (needs 0 <= M < 2^31)"
    if n not in SUPPORTED_N:
        return f"N={n} (supported: {SUPPORTED_N})"
    t = (m + TILE - 1) // TILE
    tensors = {"x6": x6, "xs": xs, "w6": w6, "ws": ws, "bias": bias, "sa": sa, "sb": sb}
    for name, ten in tensors.items():
        if not isinstance(ten, torch.Tensor):
            return f"{name} is {type(ten).__name__}, not a tensor"
    for name, ten, numel, align in (("x6", x6, t * KSTEPS * X6_ITEM_BYTES, 16), ("xs", xs, t * KSTEPS * XS_ITEM_BYTES, 8),
                                     ("w6", w6, (n // 256) * 2 * KSTEPS * X6_ITEM_BYTES, 16), ("ws", ws, (n // 32) * 16, 4)):
        if ten.dtype != torch.uint8 or not ten.is_contiguous() or ten.numel() != numel:
            return (f"{name} {ten.dtype} {tuple(ten.shape)} contiguous={ten.is_contiguous()} "
                    f"(needs contiguous uint8 with {numel} bytes for M={m}, N={n})")
        if ten.data_ptr() % align:
            return f"{name} is not {align}-byte aligned"
    if bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,) or bias.stride() != (1,) or bias.data_ptr() % 16:
        return f"bias {bias.dtype} {tuple(bias.shape)} stride {bias.stride()} (needs contiguous 16-B aligned bf16 [{n}])"
    for name, scale in (("sa", sa), ("sb", sb)):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            return f"{name} {scale.dtype} {tuple(scale.shape)} (needs one fp32 value)"
    if out is not None:
        want = torch.bfloat16 if n == U_N else _FP8_E4M3
        if out.dtype != want or tuple(out.shape) != (m, n) or not out.is_contiguous() or out.data_ptr() % 16:
            return (f"out {out.dtype} {tuple(out.shape)} contiguous={out.is_contiguous()} "
                    f"(needs contiguous 16-B aligned {want} [{m}, {n}])")
    device = x6.device
    if device.type != "cuda" or any(ten.device != device for ten in list(tensors.values()) + ([out] if out is not None else [])):
        return "operands are not all on one CUDA device"
    if device.index is not None and device.index != torch.cuda.current_device():
        return f"operands on {device}, current device cuda:{torch.cuda.current_device()}"
    return None


def launch(kern: Kernel, x6: torch.Tensor, xs: torch.Tensor, w6: torch.Tensor, ws: torch.Tensor, bias: torch.Tensor,
           sa: torch.Tensor, sb: torch.Tensor, m: int, out: torch.Tensor) -> torch.Tensor:
    """Launch on operands that passed ineligible_reason() (stream-ordered on the current stream)."""
    if m == 0:
        return out
    rc = kern.fn(x6.data_ptr(), xs.data_ptr(), w6.data_ptr(), ws.data_ptr(), out.data_ptr(), bias.data_ptr(),
                 sa.data_ptr(), sb.data_ptr(), m, _stream_handle(x6.device))
    if rc != 0:
        raise RuntimeError(f"k512_mx6 N={kern.n} M={m} failed: {_rc_message(rc)} (lib={kern.path})")
    return out


def k512_mx6(x6: torch.Tensor, xs: torch.Tensor, w6: torch.Tensor, ws: torch.Tensor, bias: torch.Tensor,
             sa: torch.Tensor, sb: torch.Tensor, M: int, N: int, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """out = the MXFP6 GEMM of (x6, xs) x (w6, ws) with the shipped epilogue (module docstring), written in place into
    ``out`` (allocated when None: e4m3 [M, N], or bf16 [M, 512] for U) and returned. Asynchronous on the current
    stream. ValueError for operands it cannot take; RuntimeError when the library is unavailable or a launch fails."""
    m, n = int(M), int(N)
    reason = ineligible_reason(x6, xs, w6, ws, bias, sa, sb, m, n, out)
    if reason is not None:
        raise ValueError(f"k512_mx6: {reason}")
    kern = kernel(n)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16 if n == U_N else _FP8_E4M3, device=x6.device)
    return launch(kern, x6, xs, w6, ws, bias, sa, sb, m, out)


# ----------------------------------------------------------------------------------------------- weight shadow
@dataclass
class _ShadowEntry:
    ref: "weakref.ReferenceType[torch.Tensor]"
    version: int
    data_ptr: int
    w6: torch.Tensor
    ws: torch.Tensor
    sb: torch.Tensor


_shadow_lock = threading.Lock()
_shadows: Dict[Tuple[Any, ...], _ShadowEntry] = {}
_shadow_stats: Dict[str, int] = {"hits": 0, "refreshes": 0}
_workspaces: Dict[Tuple[int, int], torch.Tensor] = {}


def _normalize_layout(weight: torch.Tensor, layout: Optional[Sequence[Sequence[int]]]) -> _Layout:
    """triton_fp8._normalize_layout, plus int-normalization (a layout is a cache key)."""
    if layout is None:
        return ((0, int(weight.shape[1])),)
    norm = tuple((int(a), int(b)) for a, b in layout)
    if not norm:
        raise ValueError("MX6 weight layout cannot be empty")
    for start, end in norm:
        if start < 0 or end <= start or end > weight.shape[1]:
            raise ValueError(f"invalid MX6 weight column range {(start, end)} for shape {tuple(weight.shape)}")
    return norm


def shadow_ineligible_reason(weight: torch.Tensor, layout: _Layout) -> Optional[str]:
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or weight.shape[0] != K:
        return f"weight {getattr(weight, 'shape', None)} (needs [{K}, C])"
    if weight.dtype not in (torch.float32, torch.bfloat16):
        return f"weight dtype {weight.dtype} (needs float32 or bfloat16)"
    if weight.device.type != "cuda":
        return f"weight on {weight.device} (needs CUDA)"
    if weight.device.index is not None and weight.device.index != torch.cuda.current_device():
        return f"weight on {weight.device}, current device cuda:{torch.cuda.current_device()}"
    if not 1 <= len(layout) <= 4:
        return f"layout {layout} (1..4 column ranges)"
    n = sum(b - a for a, b in layout)
    if n % 256 or n > _MAX_SHADOW_N:
        return f"layout {layout}: N={n} (needs a multiple of 256, <= {_MAX_SHADOW_N})"
    return None


def _workspace(device: torch.device) -> torch.Tensor:
    """The zero-initialized 64-byte barrier workspace of (device, current stream); the kernel leaves it zero."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (int(index), _stream_handle(device))
    ws = _workspaces.get(key)
    if ws is None:
        ws = torch.zeros((16,), dtype=torch.int32, device=device)  # one-time init, ordered on this stream
        _workspaces[key] = ws
    return ws


def refresh_weight_shadow(weight: torch.Tensor, layout: _Layout, *, perm: int = PERM_K512,
                          out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One uncached shadow refresh = ONE launch of k512_mx6_wshadow: (w6 [N/256, 2, 4, 12288] uint8, ws [N/32, 4, 4]
    uint8, sb [1, 1] fp32) of `weight`'s column `layout` (module docstring). `out` reuses buffers (tests, benches)."""
    layout = _normalize_layout(weight, layout)
    reason = shadow_ineligible_reason(weight, layout)
    if reason is not None:
        raise ValueError(f"mx6_weight: {reason}")
    kern = wshadow_kernel()
    n = sum(b - a for a, b in layout)
    device = weight.device
    if out is None:
        w6 = torch.empty((n // 256, 2, KSTEPS, X6_ITEM_BYTES), dtype=torch.uint8, device=device)
        ws = torch.empty((n // 32, 4, 4), dtype=torch.uint8, device=device)
        sb = torch.empty((1, 1), dtype=torch.float32, device=device)
    else:
        w6, ws, sb = out
    ranges = list(layout) + [(0, 0)] * (4 - len(layout))
    args = []
    for a, b in ranges:
        args += [a, b - a]
    wk = _workspace(device)
    rc = kern.fn(weight.data_ptr(), int(weight.dtype == torch.bfloat16), int(weight.stride(0)), int(weight.stride(1)),
                 n, *args, int(perm), w6.data_ptr(), ws.data_ptr(), sb.data_ptr(), wk.data_ptr(),
                 _stream_handle(device))
    if rc != 0:
        raise RuntimeError(f"k512_mx6_wshadow N={n} failed: {_rc_message(rc)} (lib={kern.path})")
    return w6, ws, sb


def mx6_weight(weight: torch.Tensor, layout: Optional[Sequence[Sequence[int]]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(w6, ws, sb) of `weight`'s column `layout`, cached until `weight` changes (its identity, data_ptr or
    _version); a refresh is one kernel launch on the current stream, no host sync. See the module docstring."""
    layout = _normalize_layout(weight, layout)
    index = weight.device.index
    if index is None:
        index = torch.cuda.current_device() if weight.is_cuda else -1
    key = (id(weight), int(index), tuple(weight.shape), tuple(weight.stride()), weight.dtype, layout, PERM_K512)
    version = int(getattr(weight, "_version", 0))
    with _shadow_lock:
        entry = _shadows.get(key)
        if (entry is not None and entry.ref() is weight and entry.version == version
                and entry.data_ptr == weight.data_ptr()):
            _shadow_stats["hits"] += 1
            return entry.w6, entry.ws, entry.sb
    w6, ws, sb = refresh_weight_shadow(weight, layout)
    with _shadow_lock:
        for k in [k for k, e in _shadows.items() if e.ref() is None]:  # drop entries of freed weights
            del _shadows[k]
        _shadows[key] = _ShadowEntry(ref=weakref.ref(weight), version=version, data_ptr=weight.data_ptr(),
                                     w6=w6, ws=ws, sb=sb)
        _shadow_stats["refreshes"] += 1
    return w6, ws, sb


def clear_shadow_cache() -> None:
    with _shadow_lock:
        _shadows.clear()
        _shadow_stats.update(hits=0, refreshes=0)


def shadow_cache_stats() -> Dict[str, int]:
    with _shadow_lock:
        return {"entries": len(_shadows), **_shadow_stats}


# ----------------------------------------------------------------------------------------------- torch reference
# pr7 iteration-24 proto.py (sha256 9edc2647...) q_e2m3 / enc_e2m3 / e8m0_exp / pack_rows / quant_W / lds_map /
# tile_W, unchanged except for the device argument and the explicit N, plus triton_fp8's calibrated_scale chain. Used
# by prebuild()'s self-check and the tests; never on the training path. No MX library.
def _ref_q_e2m3(v: torch.Tensor) -> torch.Tensor:
    a = v.abs()
    step = torch.where(a < 2, torch.full_like(a, 0.125), torch.where(a < 4, torch.full_like(a, 0.25), torch.full_like(a, 0.5)))
    q = torch.clamp(torch.round(a / step) * step, max=7.5)
    return torch.copysign(q, v)


def _ref_enc_e2m3(q: torch.Tensor) -> torch.Tensor:
    a = q.abs()
    s = (q < 0).to(torch.int64)
    e = torch.where(a < 1, torch.zeros_like(a), torch.floor(torch.log2(torch.clamp(a, min=1.0))) + 1)
    m = torch.where(a < 1, a / 0.125, (a / torch.exp2(e - 1) - 1.0) * 8.0)
    return (s << 5) | (e.to(torch.int64) << 3) | torch.round(m).to(torch.int64)


def _ref_e8m0_exp(amax: torch.Tensor, fmax: float = 7.5) -> torch.Tensor:
    e = torch.ceil(torch.log2(torch.clamp(amax, min=1e-38) / fmax))
    for _ in range(2):
        e = torch.where(amax / torch.exp2(e) > fmax, e + 1, e)
        e = torch.where(amax / torch.exp2(e - 1) <= fmax, e - 1, e)
    e = torch.where(amax == 0, torch.zeros_like(e), e)
    return torch.clamp(e, -127, 127)


def _ref_pack_rows(codes: torch.Tensor) -> torch.Tensor:
    r = codes.shape[0]
    c = codes.view(r, 16, 4, 8)
    sh = torch.arange(8, device=codes.device, dtype=torch.int64) * 6
    v = (c << sh).sum(-1)
    by = ((v.unsqueeze(-1) >> (torch.arange(6, device=codes.device, dtype=torch.int64) * 8)) & 255).to(torch.uint8)
    blk = by.view(r, 4, 4, 24)
    head = blk[..., :16].reshape(r, 4, 64)
    tail = blk[..., 16:].reshape(r, 4, 32)
    return torch.cat([head, tail], -1).reshape(r, 384).contiguous()


def _ref_quant_w(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """proto.quant_W: w fp32 [N, 512] -> (MX6 v1 rows [N, 384], ws [N/32][q][S] flat)"""
    n = w.shape[0]
    wb = w.view(n // 32, 32, 16, 32)
    e = _ref_e8m0_exp(wb.abs().amax(dim=(1, 3)))
    q = _ref_q_e2m3(torch.clamp(wb / torch.exp2(e)[:, None, :, None], -7.5, 7.5))
    codes = _ref_enc_e2m3(q).view(n, 16, 32)
    ws = (e.to(torch.int32) + 127).to(torch.uint8).view(n // 32, 4, 4).permute(0, 2, 1).contiguous().view(-1)
    return _ref_pack_rows(codes), ws


def _ref_lds_map(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    d = torch.arange(12288, device=device, dtype=torch.int64)
    b = d % 16
    lane = (d // 16) % 64
    w = (d // 1024) % 4
    i = d // 4096
    j = 4 * i + w
    head = j < 8
    rh = 16 * j + lane // 4
    qh = (lane & 3) ^ ((rh >> 2) & 3)
    rt = 32 * (j - 8) + lane // 2
    pt = (lane & 1) ^ ((rt >> 3) & 1)
    rr = torch.where(head, rh, rt)
    cb = torch.where(head, 16 * qh + b, 64 + 16 * pt + b)
    return rr, cb


def _ref_tile_w(w6_rows: torch.Tensor, n: int) -> torch.Tensor:
    """proto.tile_W with N explicit: MX6 v1 rows [n, 384] -> [n/256, 2, 4, 12288] (k512 output-column permutation)"""
    rr, cb = _ref_lds_map(w6_rows.device)
    perm = lambda h: 64 * (rr >> 5) + 16 * ((rr & 15) >> 2) + 4 * ((rr >> 4) & 1) + (rr & 3) + 8 * h  # noqa: E731
    flat = w6_rows.reshape(-1)
    parts = []
    for tn in range(n // 256):
        for h in (0, 1):
            for s in range(4):
                parts.append(flat[(tn * 256 + perm(h)) * 384 + s * 96 + cb])
    return torch.stack(parts, 0).contiguous().view(n // 256, 2, KSTEPS, X6_ITEM_BYTES)


def mx6_activation_reference(y: torch.Tensor, chunk: int = 65536) -> Tuple[torch.Tensor, torch.Tensor]:
    """proto.quant_A + tile_A + a_scale_native of fp32 y [R, 512] (row 0 = a 128-row tile boundary): (x6 [T, 4, 12288],
    xs [T, 4, 512]), pad rows zero codes / byte 127. The bytes the LN producer must write for y; tests only."""
    r = int(y.shape[0])
    t = (r + TILE - 1) // TILE
    rows = torch.zeros((t * TILE, 384), dtype=torch.uint8, device=y.device)
    ee = torch.zeros((t * TILE, 16), dtype=torch.int32, device=y.device)
    for r0 in range(0, r, chunk):
        xb = y[r0:r0 + chunk].reshape(-1, 16, 32)
        e = _ref_e8m0_exp(xb.abs().amax(-1))
        q = _ref_q_e2m3(torch.clamp(xb / torch.exp2(e).unsqueeze(-1), -7.5, 7.5))
        rows[r0:r0 + xb.shape[0]] = _ref_pack_rows(_ref_enc_e2m3(q))
        ee[r0:r0 + xb.shape[0]] = e.to(torch.int32)
    rr, cb = _ref_lds_map(y.device)
    flat = rows.view(t, TILE * 384)
    x6 = torch.stack([flat[:, rr * 384 + s * 96 + cb] for s in range(KSTEPS)], 1).contiguous()
    xs = (ee + 127).to(torch.uint8).view(t, 8, 16, 4, 4).permute(0, 3, 4, 2, 1).contiguous().view(t, KSTEPS, 512)
    return x6, xs


def mx6_weight_reference(weight: torch.Tensor, layout: Optional[Sequence[Sequence[int]]] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The torch-op chain mx6_weight must equal byte for byte: triton_fp8's FP8 shadow scale (_pack_weight_layout +
    calibrated_scale(margin=1.0)), then proto.quant_W(packed^T / scale) + tile_W. Slow (a few ms); tests only."""
    layout = _normalize_layout(weight, layout)
    w = weight.detach()
    packed = w[:, layout[0][0]:layout[0][1]] if len(layout) == 1 else torch.cat([w[:, a:b] for a, b in layout], 1)
    amax = packed.abs().amax().float().clamp_min(1.0e-12)
    # calibrated_scale's "amax * 1.0 / 448.0" on a CUDA tensor is amax * fp32(1/448) (aten div_true_kernel_cuda with
    # a CPU scalar; triton_fp8_wshadow.py); written out so the reference also holds on the CPU (true division there)
    sb = (amax * torch.tensor(INV_FP8_MAX_F32, dtype=torch.float32, device=amax.device)).reshape(1, 1)
    wnt = packed.t().contiguous() / sb  # fp32 [N, 512] IEEE quotient (bf16 / fp32 promotes to fp32)
    rows, ws = _ref_quant_w(wnt)
    n = wnt.shape[0]
    return _ref_tile_w(rows, n), ws.view(n // 32, 4, 4), sb


# ----------------------------------------------------------------------------------------------- prebuild
def _ramp_weight(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic [512, 2048] weight (no RNG): many exponents per block, zero blocks, a 448 peak (sb = 1 exactly
    is NOT forced), negative values."""
    k = torch.arange(K, device=device, dtype=torch.int64).view(-1, 1)
    c = torch.arange(2048, device=device, dtype=torch.int64).view(1, -1)
    mant = ((k * 131 + c * 29) % 211 - 105).to(torch.float32) / 64.0
    expo = (((k // 32) * 7 + (c // 32) * 3) % 13 - 6).to(torch.float32)
    w = mant * torch.exp2(expo) * 0.01
    w[(k // 32 == 3) & (c // 32 == 5)] = 0.0  # all-zero 32x32 blocks (one per layout)
    w[(k // 32 == 9) & (c // 32 == 37)] = 0.0
    w[(k // 32 == 12) & (c // 32 == 50)] = 0.0
    return w.to(dtype).contiguous()


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same shape and the same bytes (fp32 compared as int32 bits, so -0.0 != +0.0 and NaN payloads count)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.float32:
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    if a.dtype == torch.uint8:
        return bool(torch.equal(a, b))
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def _selfcheck_shadow(device: torch.device) -> Optional[str]:
    """The shadow on deterministic ramps, fp32 and bf16, the three production layouts: every byte of w6 / ws and the
    bits of sb vs mx6_weight_reference."""
    for dtype in (torch.float32, torch.bfloat16):
        w = _ramp_weight(device, dtype)
        for layout in (((512, 2048),), ((512, 1024), (1536, 2048)), ((0, 512),)):
            got = refresh_weight_shadow(w, layout)
            ref = mx6_weight_reference(w, layout)
            bad = [nm for nm, g, r in zip(("w6", "ws", "sb"), got, ref) if not bitwise_equal(g, r)]
            if bad:
                return f"weight shadow self-check {dtype} layout {layout}: {bad} differ from the torch reference"
    return None


def _launch_checked(kern: Kernel, x6: torch.Tensor, xs: torch.Tensor, w6: torch.Tensor, ws: torch.Tensor,
                    bias: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor, m: int, n: int,
                    out: torch.Tensor) -> torch.Tensor:
    """k512_mx6 on the library `kern` instead of the one kernel(N) serves (the BSR self-check): the same operand
    validation (ValueError), then launch()."""
    reason = ineligible_reason(x6, xs, w6, ws, bias, sa, sb, m, n, out)
    if reason is not None:
        raise ValueError(f"k512_mx6: {reason}")
    return launch(kern, x6, xs, w6, ws, bias, sa, sb, m, out)


def _selfcheck_gemm(device: torch.device, n: int, rows: int = 257, kern: Optional[Kernel] = None) -> Optional[str]:
    """GEMM N on zeros (codes 0, E8M0 127) into a 0xFF (e4m3 / bf16 NaN) prefill: every output byte must become +0.
    `kern` (the BSR self-check): launch that library instead of the one kernel(N) serves."""
    u8 = torch.uint8
    t = (rows + TILE - 1) // TILE
    x6 = torch.zeros((t, KSTEPS, X6_ITEM_BYTES), dtype=u8, device=device)
    xs = torch.full((t, KSTEPS, XS_ITEM_BYTES), 127, dtype=u8, device=device)
    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    w6 = torch.zeros(((n // 256), 2, KSTEPS, X6_ITEM_BYTES), dtype=u8, device=device)
    ws = torch.full((n // 32, 4, 4), 127, dtype=u8, device=device)
    bias = torch.zeros((n,), dtype=torch.bfloat16, device=device)
    dtype = torch.bfloat16 if n == U_N else _FP8_E4M3
    width = n * (2 if n == U_N else 1)
    out = torch.full((rows, width), 0xFF, dtype=u8, device=device).view(dtype)
    if kern is None:
        k512_mx6(x6, xs, w6, ws, bias, one, one, rows, n, out=out)
    else:
        _launch_checked(kern, x6, xs, w6, ws, bias, one, one, rows, n, out)
    torch.cuda.synchronize(device)
    nonzero = int(torch.count_nonzero(out.view(u8)).item())
    if nonzero:
        path = kern.path if kern is not None else _kernels[_lib_key(n)].path
        return f"GEMM N={n} self-check on zeros wrote {nonzero} nonzero bytes ({path})"
    return None


# BSR self-check operands (synthetic, deterministic, no RNG): the production weight layout of each site
_BSR_LAYOUT: Dict[int, _Layout] = {1536: ((512, 2048),), 1024: ((512, 1024), (1536, 2048)), 512: ((0, 512),)}


def _bsr_ramp_rows(r0: int, r1: int, device: torch.device) -> torch.Tensor:
    """Rows [r0, r1) of the BSR self-check activation y fp32 [M, 512] (deterministic, no RNG): no two rows of the
    largest self-check are alike after MX6 quantization (32-block b of row r is negated when bit b of r is set), so a
    row, tile, row half or K step read from the wrong place changes the output; many E8M0 block exponents in four row
    magnitude bands (e4m3 outputs from saturation through subnormals to +-0), E2M3 codes of both signs, all-zero
    32-blocks."""
    r = torch.arange(r0, r1, device=device, dtype=torch.int64).view(-1, 1)
    c = torch.arange(K, device=device, dtype=torch.int64).view(1, -1)
    mant = ((r * 37 + c * 11) % 199 - 99).to(torch.float32) / 16.0
    expo = (((r // 7) * 5 + (c // 32) * 3) % 15 - 7 - 6 * ((r // 5) % 4)).to(torch.float32)
    y = mant * torch.exp2(expo) * (1.0 - 2.0 * ((r >> (c // 32)) & 1).to(torch.float32))
    return torch.where(((r + 3 * (c // 32)) % 11) == 0, torch.zeros_like(y), y)


def _bsr_activation(m: int, device: torch.device, block: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
    """(x6, xs) of the BSR self-check activation, M rows: mx6_activation_reference of _bsr_ramp_rows, `block` rows (a
    multiple of 128: whole producer tiles, which are independent) at a time to bound the reference's temporaries."""
    t = (m + TILE - 1) // TILE
    x6 = torch.empty((t, KSTEPS, X6_ITEM_BYTES), dtype=torch.uint8, device=device)
    xs = torch.empty((t, KSTEPS, XS_ITEM_BYTES), dtype=torch.uint8, device=device)
    for r0 in range(0, m, block):
        part6, parts = mx6_activation_reference(_bsr_ramp_rows(r0, min(m, r0 + block), device), chunk=block)
        t0 = r0 // TILE
        x6[t0:t0 + part6.shape[0]] = part6
        xs[t0:t0 + parts.shape[0]] = parts
    return x6, xs


def _bsr_bias(n: int, device: torch.device) -> torch.Tensor:
    """The BSR self-check bias, bf16 [N] (exact, never zero): magnitudes 1/16 .. 29/16 of both signs, in four magnitude
    bands (x 1, 2^-4, 2^-8, 2^-12) across consecutive columns."""
    k = torch.arange(n, device=device, dtype=torch.int64)
    v = (((k * 13) % 29 - 14).to(torch.float32) + 0.5) / 8.0 * torch.exp2(-4.0 * (k % 4).to(torch.float32))
    return v.to(torch.bfloat16).contiguous()


def _selfcheck_bsr(device: torch.device, n: int, bsr: Kernel, ref: Kernel,
                   acts: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
                   ms: Sequence[int] = BSR_SELFCHECK_M) -> Optional[str]:
    """Site N's BSR library `bsr` before it is armed (None = passed, else the reason): (a) _selfcheck_gemm on zeros
    into a NaN prefill; (b) at every M of `ms`, its full output plus two guard rows (they must keep the 0xFF NaN
    prefill) bitwise equal to the site's shipped library `ref` on the same deterministic synthetic MX6 operands:
    activations mx6_activation_reference(_bsr_ramp_rows) (`acts`: {M: (x6, xs)} computed once for every site), weights
    the weight shadow of _ramp_weight in the site's production layout, sb from that shadow, sa = BSR_SELFCHECK_SA, a
    nonzero bf16 bias. Synthetic only: no data, model state or RNG (it runs before RUN_START)."""
    reason = _selfcheck_gemm(device, n, kern=bsr)
    if reason is not None:
        return f"zeros check: {reason}"
    w6, ws, sb = refresh_weight_shadow(_ramp_weight(device, torch.float32), _BSR_LAYOUT[n])
    bias = _bsr_bias(n, device)
    sa = torch.full((1, 1), BSR_SELFCHECK_SA, dtype=torch.float32, device=device)
    u8 = torch.uint8
    dtype = torch.bfloat16 if n == U_N else _FP8_E4M3
    width = n * (2 if n == U_N else 1)
    for m in ms:
        x6, xs = acts[m] if acts is not None and m in acts else _bsr_activation(m, device)
        want = torch.full((m + 2, width), 0xFF, dtype=u8, device=device)
        got = torch.full((m + 2, width), 0xFF, dtype=u8, device=device)
        _launch_checked(ref, x6, xs, w6, ws, bias, sa, sb, m, n, want.view(dtype)[:m])
        _launch_checked(bsr, x6, xs, w6, ws, bias, sa, sb, m, n, got.view(dtype)[:m])
        torch.cuda.synchronize(device)
        guard = int(torch.count_nonzero(got[m:] != 0xFF).item())
        if guard:
            return f"bitwise check M={m}: {guard} guard-row bytes past row M were written ({Path(bsr.path).name})"
        if not bitwise_equal(got, want):
            diff = int(torch.count_nonzero(got != want).item())
            return (f"bitwise check M={m}: {diff} of {got.numel()} output bytes differ from {Path(ref.path).name} "
                    f"({Path(bsr.path).name})")
    return None


def prebuild(warm_rows: int = 257) -> bool:
    """Build and load the N=1536 / 1024 GEMMs and the weight shadow (and track B's U when present) in parallel before
    RUN_START, then self-check: the shadow on deterministic ramps (fp32 / bf16, the three production layouts) against
    the torch reference, byte for byte, and every GEMM on zeros into a NaN prefill. No data, model state or RNG.
    Returns True when 1536 / 1024 / the shadow are ready (U is optional: its failure is latched and logged, and U stays
    FP8), False when skipped (no gfx950 GPU). Raises PrebuildError otherwise (latched). With the RING6 knob on, the
    GEMMs and U built here are the RING6 builds (module docstring) and one rank-tagged "RING6 ARMED" line follows the
    "prebuilt" line; in both settings the per-kernel lds= of the "prebuilt" line marks the arm (73728 RING6, 61440
    shipped). With the BSR knob on (gin item (110); read here, once), the BSR library of every selected site is built
    in the same pool and, after the shipped self-checks, checked by _selfcheck_bsr against the site's shipped library:
    a site that passes is armed (kernel(N) serves BSR; the "prebuilt" line lists the BSR library's resources and one
    rank-tagged "BSR ARMED" line follows it), any other is declined (latched, one rank-tagged "BSR DECLINED" line; the
    site keeps its shipped library). BSR never raises PrebuildError and never changes the return value for a
    recoverable failure (a device fault or a hang inside a BSR kernel aborts or hangs the job, as for any kernel).
    Called once per process, on the main thread before training: the BSR state takes no lock of its own."""
    global _prebuilt
    del warm_rows  # signature kept compatible with k512_fp8_gemm.prebuild
    if not torch.cuda.is_available():
        _print("prebuild skipped: no GPU")
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        _print(f"prebuild skipped: arch {arch or 'unknown'} (built for gfx950)")
        return False
    ring6_requested()  # the RING6 knob is read here, once, before the parallel builds
    keys = list(GEMM_N) + [WSHADOW_KEY] + ([U_N] if U_N in SUPPORTED_N else [])
    # the BSR knob, read here once: its sites that exist (U only with the U site) and are not armed yet
    bsr_ns = [n for n in BSR_SITES if n in bsr_sites() and n in SUPPORTED_N and n not in _bsr_armed]
    bsr_keys = [_bsr_key(n) for n in bsr_ns]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(keys) + len(bsr_keys)) as pool:
        futures = {key: pool.submit(_get, key) for key in keys + bsr_keys}
    errors, u_error = [], None
    bsr_errors: Dict[int, str] = {}  # a BSR build / load / routing failure declines that site only (_arm_bsr)
    for key, future in futures.items():
        try:
            future.result()
        except Exception as error:
            if isinstance(key, tuple) and _is_bsr_key(key):
                bsr_errors[key[0]] = str(error)
            elif key == U_N:
                u_error = str(error)
            else:
                errors.append(f"{key}: {error}")
    if errors:
        _print("prebuild FAILED: " + " | ".join(errors))
        raise PrebuildError("K=512 MXFP6 prebuild failed: " + " | ".join(errors))
    t_build = time.perf_counter() - t0
    reason = _selfcheck_shadow(device)
    for n in GEMM_N:
        reason = reason or _selfcheck_gemm(device, n)
    if reason is not None:
        for key in keys:
            lkey = _lib_key(key)
            with _key_lock(lkey):
                _kernels.pop(lkey, None)
                _failures[lkey] = f"K=512 MXFP6 self-check failed: {reason}"
        _bsr_armed.clear()  # nothing of this module is served after a failed prebuild (no-op with the BSR knob off)
        _print("prebuild FAILED: " + reason)
        raise PrebuildError("K=512 MXFP6 prebuild failed: " + reason)
    u_key = _lib_key(U_N)
    if u_key in _kernels:
        try:
            u_reason = _selfcheck_gemm(device, U_N)
        except RuntimeError as error:  # a failed U launch (rc != 0); a device fault surfaces in the next sync anyway
            u_reason = f"GEMM N={U_N} self-check raised: {error}"
        if u_reason is not None:  # U is optional: latch it, keep VQK / VK
            with _key_lock(u_key):
                _kernels.pop(u_key, None)
                _failures[u_key] = f"K=512 MXFP6 U self-check failed: {u_reason}"
            u_error = u_reason
    if bsr_ns:
        _arm_bsr(device, bsr_ns, bsr_errors)
    loaded = {k: _kernels[_lib_key(k)] for k in keys if _lib_key(k) in _kernels}
    detail = "; ".join(f"{k} {kern.build_s:.1f}s [{kern.resources}]" for k, kern in loaded.items())
    detail += "".join(f"; BSR {n} {_bsr_armed[n].build_s:.1f}s [{_bsr_armed[n].resources}]"
                      for n in BSR_SITES if n in _bsr_armed)  # armed BSR libraries only (nothing with the knob off)
    _print(f"prebuilt: N={','.join(str(n) for n in GEMM_N)} + weight shadow"
           + (f" + U N={U_N}" if u_key in _kernels else f" (U N={U_N}: {u_error or 'source absent'})")
           + f" in {t_build:.1f}s (dir {_process_build_dir()}); self-check (shadow ramps vs torch reference, GEMMs on "
           f"zeros) OK; {detail}")
    _log_ring6_armed(loaded, u_error)
    _log_bsr_armed()
    _prebuilt = True
    return True


def _clip(text: str, limit: int = 1500) -> str:
    """One physical line of at most `limit` characters (a rank-tagged line is written with one pipe-atomic write)."""
    text = " | ".join(part.strip() for part in str(text).splitlines() if part.strip())
    return text if len(text) <= limit else text[:limit] + " ..."


def _arm_bsr(device: torch.device, sites: Sequence[int], build_errors: Dict[int, str]) -> None:
    """prebuild(), BSR knob on, after the shipped self-checks: arm (kernel(N) serves BSR) or decline (_bsr_decline)
    each BSR site of this prebuild. A site is armed only when its BSR library built, loaded and passed the impl routing
    check (_load), its shipped library is loaded (the fallback and bitwise reference; RING6 per its knob), and
    _selfcheck_bsr passed. Any exception declines (never raises: BSR must not fail the prebuild)."""
    checks: Dict[int, Tuple[Kernel, Kernel]] = {}
    for n in sites:
        ref = _kernels.get(_lib_key(n))
        bsr = _kernels.get(_bsr_key(n))
        if n in build_errors:
            _bsr_decline(n, build_errors[n])
        elif ref is None:
            _bsr_decline(n, f"the site's shipped library is not available "
                            f"({_failures.get(_lib_key(n)) or 'not loaded'})")
        elif bsr is None:
            _bsr_decline(n, f"the BSR library is not loaded ({_failures.get(_bsr_key(n)) or 'unknown'})")
        else:
            checks[n] = (bsr, ref)
    if not checks:
        return
    try:  # the activation operands of every M, once for all sites (synthetic, deterministic)
        acts = {m: _bsr_activation(m, device) for m in BSR_SELFCHECK_M}
    except Exception as error:
        for n in checks:
            _bsr_decline(n, f"self-check operands failed: {type(error).__name__}: {error}")
        return
    for n, (bsr, ref) in checks.items():
        try:
            reason = _selfcheck_bsr(device, n, bsr, ref, acts)
        except Exception as error:  # e.g. a failed launch (rc != 0); a device fault surfaces in the next sync anyway
            reason = f"self-check raised {type(error).__name__}: {error}"
        if reason is None:
            _bsr_armed[n] = bsr
        else:
            _bsr_decline(n, f"self-check failed: {reason}")


def _bsr_decline(n: int, reason: str) -> None:
    """Latch BSR for site N (never served; the site keeps its shipped library) and print its one rank-tagged DECLINED
    line (once per process)."""
    key = _bsr_key(n)
    with _key_lock(key):
        _kernels.pop(key, None)
        _failures.setdefault(key, f"K=512 MXFP6 BSR GEMM N={n} declined: {reason}")
    _bsr_armed.pop(n, None)
    if n not in _bsr_declined:
        _bsr_declined[n] = reason
        _rank_print(f"[k512-mx6] BSR DECLINED N={n}: {_clip(reason)} (serving the shipped kernel)")


def _log_bsr_armed() -> None:
    """BSR knob on and at least one site armed: ONE rank-tagged line per process naming the knob value, the armed BSR
    libraries (each passed the impl routing check and both self-checks) and the shipped libraries they were checked
    against. Otherwise nothing (knob off: the shipped log lines)."""
    global _bsr_logged
    if _bsr_logged or not _bsr_armed:
        return
    _bsr_logged = True
    sites = [n for n in BSR_SITES if n in _bsr_armed]
    libs = ", ".join(Path(_bsr_armed[n].path).name for n in sites)
    refs = ", ".join(Path(_kernels[_lib_key(n)].path).name if _lib_key(n) in _kernels else "?" for n in sites)
    note = "" if U_N in SUPPORTED_N or U_N not in bsr_sites() else f"; U N={U_N} not BSR (U site absent)"
    ms = ",".join(str(m) for m in BSR_SELFCHECK_M)
    _rank_print(f"[k512-mx6] BSR ARMED: {BSR_KNOB}={_bsr_raw}{_bsr_note} -> {libs}; impl routing OK ({BSR_IMPL}); "
                f"self-check zeros + bitwise vs shipped ({refs}) at M={ms} OK{note}")


def _log_ring6_armed(loaded: Dict[object, Kernel], u_error: Optional[str]) -> None:
    """RING6 knob on: ONE rank-tagged line per process (ops.utils.arm_print_all_ranks) naming the knob and every library
    built as RING6, each of which passed the impl routing check in _load. Knob off: nothing (the shipped log lines)."""
    global _ring6_logged
    if _ring6_logged or not ring6_requested():
        return
    _ring6_logged = True
    ring6 = [Path(kern.path).name for key, kern in loaded.items() if kern.ring6 and _ring6_build(key)]
    shadow = loaded.get(WSHADOW_KEY)
    impls = "k512_mx6_gemm_impl / k512_mx6_u_impl report" if U_N in loaded else "k512_mx6_gemm_impl reports"
    u_note = "" if U_N in loaded else f"; U N={U_N} not RING6 ({u_error or 'source absent'})"
    from generative_recommenders.ops.utils import arm_print_all_ranks

    arm_print_all_ranks(
        f"[k512-mx6] RING6 ARMED: {RING6_KNOB}=1 -> {RING6_DEFINE} in {', '.join(ring6) or 'NO library'}; impl routing "
        f"OK ({impls} RING6); weight shadow {Path(shadow.path).name if shadow is not None else '?'} as shipped{u_note}")

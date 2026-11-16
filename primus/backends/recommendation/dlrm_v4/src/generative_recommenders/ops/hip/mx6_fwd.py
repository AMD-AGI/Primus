# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

# pyre-unsafe

"""MXFP6 forward K=512 trio of the HSTU layers (U, V/Q/K, targets V/K): integration adapter.

DLRM_HSTU_FP8_FWD_MXFP6=1 (gin ``apply_env_bootstrap.DLRM_HSTU_FP8_FWD_MXFP6``, yambda_5b.gin
item (92); DEFAULT OFF; a DECLARED PRECISION CHANGE). Read once at import, here (``ENABLED``),
in ops/triton/triton_hstu_preprocess_and_attention.py (``_MX6_FWD``, the call-site switch) and in
ops/triton/triton_fp8.py (``_HSTU_FP8_FWD_MXFP6``, the prebuild hook). With the knob off the
training code never imports this module and every call site runs the shipped code (each MX6 arm
is an ``if _MX6_FWD and ...:`` whose ``else`` is the shipped statement, verbatim).

WHAT IT REPLACES (training forwards only; eval / no-grad forwards and the calibration window keep
the shipped FP8 path -- a declared train / eval asymmetry, see DESIGN.md Q-L1 / REVIEW m7):
- ordinary uvqk layers: the LN store_t (``triton_weighted_layer_norm_fwd_fp8_store_t``) + the
  U GEMM (hipBLASLt FP8, bf16 + bias) + the V/Q/K GEMM (k512 FP8, e4m3 out) become the r14 MX6
  LN producer (x6 MX6-T + xs MX6-A-v1, plus the same xT8 / mean / rstd the store_t writes) +
  pr7's MXFP6 k512 GEMMs. U moves only when pr7's N=512 kernel is available; until then the
  producer also writes the e4m3 x8 and U stays on the shipped FP8 call (bitwise).
- the targets-only layer: the LN store_t + the V/K GEMM (k512 FP8, N=1024).
The operands change from e4m3 with one per-tensor scale to MXFP6 E2M3 with one E8M0 scale per
32 K (activations: per (row, 32-K); weights: 2-D 32x32 blocks). The per-tensor fp32 scales,
the output-scale fold, the scaled bias and the epilogue are those of the shipped FP8 call; the
outputs keep their dtype / shape / meaning (u bf16 + bias, vqk / vk e4m3 at the qkv scale).
No K product is skipped.

WHAT THE BACKWARD SEES. Unchanged tensors: x_mean / x_rstd (fp32 [M]) and the token-major e4m3
xT8 (step-tiled [Mp/128, 512, 128], same allocator) come from the producer and must be bitwise
the store_t v2 ones (the producer's contract, checked on the first call per device); the saved u
and the saved / ephemeral vqk (or the targets vk) are the MX6 GEMM outputs. The MX6 arm engages
only on calls whose V/Q/K (V/K) pack is handed to the backward, so the first backward never
recomputes it; a retained-graph second backward recomputes it through MX6 (``recompute_vqk`` /
``recompute_vk``), not FP8.

INTERFACES (imported lazily; anything missing -> DECLINE, one log line, shipped FP8 path, bitwise):
- r14 LN producer, module ``$DLRM_HSTU_FP8_FWD_MXFP6_PRODUCER_MODULE`` (default
  ``generative_recommenders.ops.hip.ln_store_mx6``), its REAL signature::

      ln_store_mx6(x, weight, bias, eps, inv_scale, mp=None, *, want_x8=False, return_y=False,
                   xt_out=None, x6_out=None, xs_out=None, mean_out=None, rstd_out=None,
                   x8_out=None, y_out=None) -> (x6, xs, xT8, mean, rstd, x8 | None)
      ineligible_reason(x, weight, bias) -> Optional[str]    failure() -> Optional[str]
      kernel() -> Kernel(impl, source, source_sha256, ...)    prebuild() -> bool (PrebuildError)
      optional: ready(device) -> Optional[str] (consulted once per device; does not build)

  This adapter passes (x, weight, bias, eps, inv_scale, mp) positionally (mp = roundup(M, 128)),
  ``want_x8=`` and -- when it owns the buffer -- ``xt_out=`` (the fp8_tm_alloc.empty_xt step-tiled
  xT8; the check calls also pass poisoned ``x6_out`` / ``xs_out`` / ``mean_out`` / ``rstd_out`` /
  ``x8_out``); one place builds the call (``_producer_call``) and a contract test binds it against
  the real signature. Without ``xt_out`` the producer allocates AND writes its own xT8 (it has no
  no-xT8 mode): the retained-graph recompute pays that write and drops the tensor.
  x6 uint8 [T, 4, 12288] MX6-T, xs uint8 [T, 4, 512] MX6-A-v1 (pad rows: zero codes / byte 127),
  T = ceil(M / 128); mean / rstd fp32 [M]; x8 e4m3 [M, 512] iff want_x8.
- pr7 GEMMs, module ``$DLRM_HSTU_FP8_FWD_MXFP6_GEMM_MODULE`` (default
  ``generative_recommenders.ops.hip.k512_mx6_gemm``; NOT FINAL, the module does not exist yet)::

      SUPPORTED_N: tuple of int (subset of 512, 1024, 1536)
      kernel(n) -> handle             (loads / builds; raises when unavailable; latched)
      mx6_weight(weight, layout) -> (w6, ws) | (w6, ws, sb)
          w6 uint8 MX6-T [N/256, 2, 4, 12288]; ws uint8 2-D 32x32 E8M0 [N/32, 4, 4];
          sb fp32 per-tensor scale; a 2-tuple means W6 encodes W / scale_b of the FP8 shadow
          (triton_fp8._get_weight_shadow(weight, layout)[1]), which is then passed as sb.
      k512_mx6(x6, xs, w6, ws, bias, sa, sb, M, N, out=out) -> out | None   (writes `out`)
          e4m3 out [M, N] (N 1536 / 1024): e4m3_rne(sat448(acc * (sa * sb) + bias));
          bf16 out [M, N] (N 512): bf16_rne(acc * (sa * sb) + bias); acc = fp32 sum over K of
          dequant(x6) * dequant(w6); current stream, asynchronous, raises on a launch error.
      optional: prebuild() -> bool; PrebuildError.

GUARDS.
(1) Producer check, first producer call per device: one shipped store_t on the same x; mean /
    rstd / xT8 (and x8 when written) must be its bytes; x6 / xs of row slices (the first 65,536
    rows, the rows around each 2^31-byte input / output offset, the last tile) must be the bytes of
    the torch reference (ops/hip/mx6_ref.py: pr7's proto quantizer applied to the exact fp32 y);
    and a second producer call with the OTHER kernel instantiation (x8 written or not) into fresh
    poisoned buffers must give byte-identical x6 / xs / xT8 / mean / rstd (and x8 = the store_t's).
    Any mismatch latches the producer for the device (the shipped LN runs).
(2) GEMM guard, first call per (device, site, weight): the MX6 output must have no non-finite
    value (full output, bit patterns) and, on row windows (the first 65,536 rows, the rows around
    each 2^31-element / -byte output offset, the last tile), match the shipped FP8 GEMM on the same
    rows with rel L2 <= GUARD_REL_L2_MAX, every 128-row tile <= GUARD_TILE_REL_L2_MAX and slope
    <got, ref> / <ref, ref> within GUARD_SLOPE_MAX of 1 (catches a non-power-of-two scale error).
    MX6 and FP8 differ by design: a gross-error guard; full-output correctness belongs to the
    offline fp64 check. A rejection or any exception on that call (also inside the guard) latches
    the site for the process; the call returns the shipped FP8 output of the full call.
(3) Prebuild (before RUN_START): the producer on synthetic zeros and on a deterministic ramp, both
    kernel instantiations, poisoned plain-torch buffers (never the TM_NONEXP pool), every byte vs
    the torch reference; pr7's GEMMs on zeros. A failure latches EVERY site off for the process.

LOG LINES ("[mx6-fwd] ..."): "requested: ..." (import); "prebuilt: ..." | "prebuild DECLINED"
(before RUN_START); "producer check on cuda:<i>: ... -> ok" | "ERROR producer check ...";
"guard <site> on cuda:<i>: ... -> armed" | "ERROR guard ... -> REJECTED"; one
"ARMED on cuda:<i> ..." per device once U / VQK / VK are all decided ("ERROR NOT ARMED ..." if
VQK or VK latched); "DECLINED -> shipped FP8 path" (once per kind); "not engaged (expected): ..."
(calibration window, no-grad forwards); "counters at exit ..." (every process, atexit);
"ERROR OPTION-C U NOT READY: ... -> FALLBACK TO OPTION B ..." (once per process, see below).

OPTION C AND U (r14 iteration 51; review M2 of iteration 50). The GEMM module offers U (N=512 in
its SUPPORTED_N) exactly when its U library source (k512_mx6_u.cu) is present = option C; without
it (option B) U stays on the shipped FP8 GEMM by design. When the module offers U but U cannot run
MX6 -- its build or its self-check failed (the module latches U, and its prebuild() still
succeeds), or the U site latched -- the process would silently run option B. Under ..._STRICT=1
that is fatal: prebuild() raises PrebuildError ("prebuild DECLINED: OPTION-C U NOT READY: ...")
and a uvqk call raises StrictError (a "uvqk:U" decline). Without STRICT the process falls back to
option B with ONE loud "ERROR OPTION-C U NOT READY" line (check_mx6_arming.py rule L8 reads it as
FAIL for an arm that claims option C).

Env-only sub-switches (read at import; forwarded by launch_slurm.sh's DLRM_HSTU_FP8_ prefix):
  DLRM_HSTU_FP8_FWD_MXFP6_STRICT=1        raise instead of declining (prebuild not ready, a
                                         DECLINED call, a guard rejection, a stale shadow, and
                                         option C's U not ready: see OPTION C AND U)
  DLRM_HSTU_FP8_FWD_MXFP6_GEMM_MODULE     pr7 GEMM module name
  DLRM_HSTU_FP8_FWD_MXFP6_PRODUCER_MODULE r14 producer module name
  DLRM_HSTU_FP8_FWD_MXFP6_SHADOW_AUDIT=N  every N-th weight-shadow hit, compare the weight with a
                                         copy taken at the shadow build (a writer that skipped the
                                         _version bump leaves every version-keyed shadow stale)
"""

from __future__ import annotations

import atexit
import importlib
import os
import sys
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

KNOB: str = "DLRM_HSTU_FP8_FWD_MXFP6"
ENABLED: bool = os.environ.get(KNOB, "0") == "1"
STRICT: bool = os.environ.get(KNOB + "_STRICT", "0") == "1"
DEFAULT_GEMM_MODULE: str = "generative_recommenders.ops.hip.k512_mx6_gemm"
DEFAULT_PRODUCER_MODULE: str = "generative_recommenders.ops.hip.ln_store_mx6"
GEMM_MODULE: str = os.environ.get(KNOB + "_GEMM_MODULE", "") or DEFAULT_GEMM_MODULE
PRODUCER_MODULE: str = os.environ.get(KNOB + "_PRODUCER_MODULE", "") or DEFAULT_PRODUCER_MODULE
SHADOW_AUDIT: int = max(0, int(os.environ.get(KNOB + "_SHADOW_AUDIT", "0") or "0"))

K: int = 512  # contraction = LN width D
TILE: int = 128  # rows per MX6-T tile (= the wgrad token pad)
KSTEPS: int = K // 128
X6_ITEM_BYTES: int = 12288  # one (128-row tile, K step) MX6-T item: HEAD 8 KiB + TAIL 4 KiB
XS_ITEM_BYTES: int = 512  # one (tile, K step) MX6-A-v1 scale block
SITE_N: Dict[str, int] = {"U": 512, "VQK": 1536, "VK": 1024}
REQUIRED_N: Tuple[int, ...] = (1536, 1024)  # VQK, VK; U (512) only when the GEMM module offers it
U_NOT_READY_TAG: str = "OPTION-C U NOT READY"  # check_mx6_arming.py rule L8 greps this

# Gross-error guard bounds (first call per device, site and weight). Derivation in DESIGN.md (r14
# iter 49 integ, section 2.5): both outputs carry their own operand-quantization error and the
# e4m3 / bf16 output rounding, so a correct MX6 GEMM differs from the FP8 one by a few percent rel
# L2 with a slope of 1 (RNE noise is unbiased); a layout, permutation, scale-byte or per-tensor-
# scale error is >= 0.5 rel L2, and a non-power-of-two scale error moves the slope. The per-tile
# bound catches a corrupted 128-row tile, which is invisible in the global norm.
GUARD_REL_L2_MAX: float = 0.25
GUARD_TILE_REL_L2_MAX: float = 0.5
GUARD_SLOPE_MAX: float = 0.03
GUARD_CHUNK_ROWS: int = 16384  # multiple of TILE
GUARD_HEAD_ROWS: int = 65536  # multiple of TILE
CHECK_HEAD_ROWS: int = 65536  # producer x6 / xs slice check, multiple of TILE
_TWO31: int = 2**31

_FP8_E4M3: Optional[torch.dtype] = getattr(torch, "float8_e4m3fn", None)
_INT32_MAX: int = 2**31 - 1
_INTERNAL_KINDS: Tuple[str, ...] = ("selfcheck", "check")

_REQUIRED_API: Dict[str, Tuple[str, ...]] = {
    "producer": ("ln_store_mx6", "ineligible_reason", "failure", "kernel"),
    "gemm": ("SUPPORTED_N", "kernel", "mx6_weight", "k512_mx6"),
}
# Every keyword this adapter may pass to the producer's ln_store_mx6 (bound against the real
# signature by dlrm_v4/train/tests/mx6_fwd_integ_test.py).
_PRODUCER_KW: Tuple[str, ...] = ("want_x8", "xt_out", "x6_out", "xs_out", "mean_out", "rstd_out", "x8_out")


class PrebuildError(RuntimeError):
    """prebuild() under DLRM_HSTU_FP8_FWD_MXFP6_STRICT=1: the MX6 path is not ready."""


class StrictError(RuntimeError):
    """DLRM_HSTU_FP8_FWD_MXFP6_STRICT=1: a call would have left the MX6 path."""


_lock = threading.RLock()
_mods: Dict[str, Any] = {}
_mod_error: Dict[str, str] = {}
_n_ok: set = set()
_n_error: Dict[int, str] = {}
_prebuild_error: Optional[str] = None  # a failed prebuild: every site off for the process
_producer_error: Dict[int, str] = {}  # device index -> reason (latched)
_producer_ok: set = set()  # device indices whose producer reported ready (library loaded)
_producer_checked: set = set()  # device indices whose first producer call passed the check
_site_armed: set = set()  # (device index, site, weight key) that passed the guard
_site_latch: Dict[Tuple[int, str], str] = {}  # (device index, site) -> reason (latched to FP8)
_armed_devices: set = set()  # device indices whose ARMED line was printed
_once: set = set()
_counters: Dict[str, int] = {}


def _print(message: str) -> None:
    print(f"[mx6-fwd] {message}", flush=True)


def _count(key: str) -> None:
    _counters[key] = _counters.get(key, 0) + 1


def counters() -> Dict[str, int]:
    return dict(_counters)


_COUNTER_KEYS: Tuple[str, ...] = (
    "planned_uvqk", "planned_vk", "served_U", "served_VQK", "served_VK", "recompute_VQK",
    "recompute_VK", "declined", "skipped",
)


def report_counters(when: str = "exit") -> str:
    """One line with every counter (a bench compares served_* with planned_*), the device states
    and the weight-shadow builds / hits. Printed per process at exit (atexit)."""
    c = counters()
    body = " ".join(f"{key}={c.get(key, 0)}" for key in _COUNTER_KEYS)
    extra = " ".join(f"{key}={c[key]}" for key in sorted(c) if key not in _COUNTER_KEYS)
    line = (
        f"counters at {when} (pid {os.getpid()}): {body}{(' ' + extra) if extra else ''} "
        f"shadows built={_shadow_builds['builds']} hits={_shadow_builds['hits']} "
        f"producer_checked={sorted(_producer_checked)} armed_devices={sorted(_armed_devices)} "
        f"latched={sorted(f'{i}:{s}' for i, s in _site_latch)} prebuild_error={_prebuild_error is not None}"
    )
    _print(line)
    return line


def _at_exit() -> None:
    try:
        report_counters("exit")
    except Exception:  # pragma: no cover - never fail an exit
        pass


def _log_once(key: Any, message: str) -> None:
    if key in _once:
        return
    _once.add(key)
    _print(message)


def _decline(kind: str, detail: str) -> None:
    """A call that the knob asked for leaves the MX6 path: shipped FP8 code, logged once per kind."""
    _count("declined")
    _log_once(("decline", kind), f"DECLINED -> shipped FP8 path ({KNOB}=1): {kind}: {detail}")
    if STRICT:
        raise StrictError(f"{KNOB}_STRICT=1: MX6 declined ({kind}: {detail})")


def _skip(kind: str, detail: str) -> None:
    """An expected non-engagement (calibration window, no-grad forward): shipped path, logged once."""
    _count("skipped")
    _log_once(("skip", kind), f"not engaged (expected): {kind}: {detail}")


def requested() -> bool:
    return ENABLED


def _dev_index(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    return device.index if device.index is not None else torch.cuda.current_device()


# ----------------------------------------------------------------------------- module loading
def _module(role: str) -> Optional[Any]:
    """The producer ("producer") or pr7 GEMM ("gemm") module, imported once; None (latched) when
    it is missing or lacks the API this adapter codes against."""
    mod = _mods.get(role)
    if mod is not None:
        return mod
    if role in _mod_error:
        return None
    name = PRODUCER_MODULE if role == "producer" else GEMM_MODULE
    with _lock:
        if role in _mods:
            return _mods[role]
        if role in _mod_error:
            return None
        try:
            mod = importlib.import_module(name)
        except Exception as error:  # ImportError, or the module failed at import
            _mod_error[role] = f"{role} module {name} not importable ({type(error).__name__}: {error})"
            return None
        missing = [attr for attr in _REQUIRED_API[role] if not hasattr(mod, attr)]
        if missing:
            _mod_error[role] = f"{role} module {name} lacks {missing} (API differs from the assumed one)"
            return None
        _mods[role] = mod
        return mod


def module_error(role: str) -> Optional[str]:
    _module(role)
    return _mod_error.get(role)


def _kernel_reason(n: int) -> Optional[str]:
    """None when pr7's kernel for N is loaded (built lazily on the first ask); latched otherwise."""
    if n in _n_ok:
        return None
    if n in _n_error:
        return _n_error[n]
    gemm = _module("gemm")
    if gemm is None:
        return _mod_error["gemm"]
    with _lock:
        if n in _n_ok:
            return None
        if n in _n_error:
            return _n_error[n]
        try:
            supported = tuple(int(v) for v in getattr(gemm, "SUPPORTED_N"))
        except Exception as error:
            supported = ()
            _n_error[n] = f"{GEMM_MODULE}.SUPPORTED_N unreadable ({error})"
            return _n_error[n]
        if n not in supported:
            _n_error[n] = f"N={n} not in {GEMM_MODULE}.SUPPORTED_N={supported}"
            return _n_error[n]
        try:
            gemm.kernel(n)
        except Exception as error:
            _n_error[n] = f"{GEMM_MODULE}.kernel({n}) failed ({type(error).__name__}: {error})"
            return _n_error[n]
        _n_ok.add(n)
        return None


def _producer_reason(device: torch.device) -> Optional[str]:
    """None when the producer can run on `device`: importable with the API, not latched by itself
    (``failure()``), its library loaded (``kernel()``: built by prebuild, else built here once),
    and not latched by this adapter's producer check."""
    prod = _module("producer")
    if prod is None:
        return _mod_error["producer"]
    index = _dev_index(device)
    latched = _producer_error.get(index)
    if latched is not None:
        return latched
    try:
        failure = prod.failure()
    except Exception as error:
        failure = f"{PRODUCER_MODULE}.failure() raised {type(error).__name__}: {error}"
    if failure:
        _producer_error[index] = f"producer latched off by itself: {failure}"
        return _producer_error[index]
    if index not in _producer_ok:
        ready = getattr(prod, "ready", None)  # optional: device / arch verdict, does not build
        try:
            reason = None if ready is None or device.type != "cuda" else ready(device)
        except Exception as error:
            reason = f"{PRODUCER_MODULE}.ready() raised {type(error).__name__}: {error}"
        if reason:
            _producer_error[index] = f"producer not ready on {device}: {reason}"
            return _producer_error[index]
        try:
            prod.kernel()
        except Exception as error:
            _producer_error[index] = f"{PRODUCER_MODULE}.kernel() failed ({type(error).__name__}: {error})"
            return _producer_error[index]
        _producer_ok.add(index)
    return None


def _producer_desc() -> str:
    """impl / source / source sha256 of the loaded producer library (a DLRM_HSTU_LN_STORE_MX6_SRC
    override shows here)."""
    prod = _mods.get("producer")
    if prod is None:
        return "?"
    try:
        kern = prod.kernel()
    except Exception as error:
        return f"unavailable ({type(error).__name__})"
    impl = getattr(kern, "impl", "?")
    src = os.path.basename(str(getattr(kern, "source", "?")))
    sha = str(getattr(kern, "source_sha256", ""))[:12]
    return f"impl={impl!r} src={src}@{sha}"


def not_ready_reason(device: torch.device, ns: Sequence[int] = REQUIRED_N) -> Optional[str]:
    """None when the producer and pr7's kernels for every N in `ns` are usable on `device`."""
    if not ENABLED:
        return f"{KNOB} is not 1 (read at import of {__name__})"
    if _prebuild_error is not None:
        return f"prebuild failed (latched for this process): {_prebuild_error}"
    reason = _producer_reason(device)
    if reason is not None:
        return reason
    for n in ns:
        reason = _kernel_reason(int(n))
        if reason is not None:
            return reason
    return None


def is_ready(device: torch.device, ns: Sequence[int] = REQUIRED_N) -> bool:
    """Producer ready AND pr7's MX6 GEMM module importable and built for every N in `ns`."""
    return not_ready_reason(device, ns) is None


def site_reason(device: torch.device, site: str) -> Optional[str]:
    """None when `site` (U / VQK / VK) may run MX6 on `device` now (ready, not latched)."""
    latched = _site_latch.get((_dev_index(device), site))
    if latched is not None:
        return f"site {site} latched to FP8 ({latched})"
    return not_ready_reason(device, (SITE_N[site],))


def u_expected() -> bool:
    """True when the loaded GEMM module offers U (N=512 in its SUPPORTED_N: its U library source
    is present = option C), i.e. U is meant to run MX6. False for option B (U stays on the shipped
    FP8 GEMM by design) and when the module is not loaded or its SUPPORTED_N is unreadable (then
    VQK / VK are not ready either: _kernel_reason latches that)."""
    gemm = _mods.get("gemm")
    if gemm is None:
        return False
    try:
        return SITE_N["U"] in tuple(int(v) for v in getattr(gemm, "SUPPORTED_N"))
    except Exception:
        return False


def _u_not_ready(where: str, why: str) -> str:
    """The OPTION-C U NOT READY reason (see the module docstring, OPTION C AND U)."""
    return (f"{U_NOT_READY_TAG}: N={SITE_N['U']} is in {GEMM_MODULE}.SUPPORTED_N (option C: the U "
            f"library source is present) but U cannot run MX6 ({where}: {why})")


def _u_fallback(message: str) -> None:
    """Without STRICT: ONE loud line per process; the process runs option B (U on the shipped FP8
    GEMM, the producer writes the e4m3 x8)."""
    _log_once(("u_fallback",), f"ERROR {message} -> FALLBACK TO OPTION B for this process: U on the "
              f"shipped FP8 GEMM, the producer writes the e4m3 x8 ({KNOB}_STRICT=1 makes this fatal)")


def _weight_key(weight: torch.Tensor) -> Tuple[int, int]:
    return (id(weight), int(weight.data_ptr()))


def _guard_pending(device: torch.device, site: str, wkey: Any) -> bool:
    index = _dev_index(device)
    return (index, site) not in _site_latch and (index, site, wkey) not in _site_armed


def _latch(device: torch.device, site: str, reason: str) -> None:
    _site_latch[(_dev_index(device), site)] = reason


# ----------------------------------------------------------------------------- weight shadows
@dataclass
class _Shadow:
    ref: "weakref.ReferenceType[torch.Tensor]"
    version: int
    data_ptr: int
    shape: torch.Size
    w6: torch.Tensor
    ws: torch.Tensor
    sb: torch.Tensor
    hits: int = 0
    snapshot: Optional[torch.Tensor] = None


_shadows: Dict[Tuple[Any, ...], _Shadow] = {}
_shadow_builds: Dict[str, int] = {"builds": 0, "hits": 0}


def _layout_columns(weight: torch.Tensor, layout: Tuple[Tuple[int, int], ...]) -> torch.Tensor:
    return torch.cat([weight.detach()[:, a:b] for a, b in layout], dim=1)


def weight_shadow(
    weight: torch.Tensor, layout: Tuple[Tuple[int, int], ...]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(w6, ws, sb) of `weight`'s column `layout` through pr7's ``mx6_weight``, cached and keyed
    exactly like triton_fp8._get_weight_shadow: (id, device, shape, stride, layout), valid while
    the same tensor is alive with the same data_ptr and ``weight._version``. A writer that changes
    the weight without bumping ``_version`` (raw-pointer kernels; see dense_adam_fused's
    ``torch.autograd.graph.increment_version``) leaves this shadow stale, like the FP8 ones;
    ``DLRM_HSTU_FP8_FWD_MXFP6_SHADOW_AUDIT=N`` detects that. pr7's ``mx6_weight`` must not serve
    a cached entry of its own across a version bump (DESIGN.md Q-P6)."""
    layout = tuple((int(a), int(b)) for a, b in layout)
    index = _dev_index(weight.device)
    key = (id(weight), index, tuple(weight.shape), tuple(weight.stride()), layout)
    version = int(getattr(weight, "_version", 0))
    entry = _shadows.get(key)
    if (
        entry is not None
        and entry.ref() is weight
        and entry.version == version
        and entry.data_ptr == weight.data_ptr()
        and entry.shape == weight.shape
    ):
        entry.hits += 1
        _shadow_builds["hits"] += 1
        if SHADOW_AUDIT and entry.snapshot is not None and entry.hits % SHADOW_AUDIT == 0:
            _audit(entry, weight, layout)
        return entry.w6, entry.ws, entry.sb
    gemm = _module("gemm")
    if gemm is None:
        raise RuntimeError(_mod_error["gemm"])
    out = gemm.mx6_weight(weight, layout)
    if not isinstance(out, (tuple, list)) or len(out) not in (2, 3):
        raise RuntimeError(f"{GEMM_MODULE}.mx6_weight returned {type(out).__name__} (want 2 or 3 tensors)")
    if len(out) == 3:
        w6, ws, sb = out
    else:
        # 2-tuple: W6 encodes W / scale_b of the FP8 shadow (pr7 DESIGN section 3: the E8M0 bytes
        # are relative to today's per-tensor scales).
        from generative_recommenders.ops.triton import triton_fp8

        w6, ws = out
        sb = triton_fp8._get_weight_shadow(weight, layout)[1]
    _shadows[key] = _Shadow(
        ref=weakref.ref(weight),
        version=version,
        data_ptr=weight.data_ptr(),
        shape=weight.shape,
        w6=w6,
        ws=ws,
        sb=sb,
        snapshot=_layout_columns(weight, layout).clone() if SHADOW_AUDIT else None,
    )
    _shadow_builds["builds"] += 1
    return w6, ws, sb


def _audit(entry: _Shadow, weight: torch.Tensor, layout: Tuple[Tuple[int, int], ...]) -> None:
    assert entry.snapshot is not None
    if torch.equal(_layout_columns(weight, layout), entry.snapshot):  # host sync, audit only
        return
    message = (
        f"STALE weight shadow: weight {tuple(weight.shape)} layout {layout} changed without a "
        f"_version bump (version {entry.version}); every version-keyed shadow (MX6 and the FP8 "
        "triton_fp8._get_weight_shadow) is stale -- the writer must call "
        "torch.autograd.graph.increment_version. MX6 latched off on this device."
    )
    _print("ERROR " + message)
    for site in SITE_N:
        _latch(weight.device, site, "stale weight shadow (audit)")
    if STRICT:
        raise StrictError(message)


def clear_shadows() -> None:
    _shadows.clear()


# ----------------------------------------------------------------------------- producer
@dataclass
class Plan:
    """One LN-producer call and the GEMM decisions of its layer call."""

    kind: str  # "uvqk" | "vk" | "recompute" | "selfcheck" | "check"
    device: torch.device
    m: int
    x6: Optional[torch.Tensor]
    xs: Optional[torch.Tensor]
    xt8: Optional[torch.Tensor]
    mean: torch.Tensor
    rstd: torch.Tensor
    x8: Optional[torch.Tensor]
    u_mx6: bool = False
    guard: Tuple[str, ...] = ()
    wkey: Any = None
    x6_shape: Tuple[int, ...] = ()
    xs_shape: Tuple[int, ...] = ()

    def release(self) -> None:
        """Drop the plan's references to the MX6 operands once the GEMMs consumed them."""
        self.x6 = None
        self.xs = None
        self.xt8 = None
        self.x8 = None

    def meta(self) -> Dict[str, torch.Tensor]:
        """dtype / shape-only stand-ins of x6 / xs (built only for a contract record)."""
        return {
            "x6": torch.empty(self.x6_shape, dtype=torch.uint8, device="meta"),
            "xs": torch.empty(self.xs_shape, dtype=torch.uint8, device="meta"),
        }


def _xt_layout_reason(tiled_xt: bool, mp: int, pad_to: int = TILE) -> Optional[str]:
    """The MX6 producer writes the step-tiled xT8 of Mp = roundup(M, 128) only, i.e. what the
    shipped store_t writes when tiled_xt is active on its v2 kernel with pad_to 128. Anything else
    would change the saved tensor's layout."""
    from generative_recommenders.ops.triton import triton_layer_norm, wgrad_ck

    if pad_to != TILE:
        return f"pad_to={pad_to}: the producer writes Mp = roundup(M, {TILE}) only"
    if not tiled_xt:
        return "tiled xT inactive (wgrad_ck.tiled_active): the store_t writes a row-major xT here"
    if not getattr(triton_layer_norm, "_STORE_T_V2", False):
        return "DLRM_HSTU_FP8_BWD_STORE_T_V2 off: the shipped store_t (v1) writes a row-major xT"
    if wgrad_ck.tiled_shape(K, mp) is None:
        return f"step-tiled xT does not apply (D={K}, Mp={mp})"
    return None


def _producer_call(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    inv_scale: float,
    mp: int,
    *,
    want_x8: bool,
    xt8: Optional[torch.Tensor] = None,
    outs: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """(args, kwargs) of the producer's ln_store_mx6 call -- the only place that builds it. `xt8`:
    the caller-owned step-tiled xT8 (``xt_out``), or None: the keyword is omitted and the producer
    allocates its own. `outs`: preallocated (poisoned) outputs of the check calls."""
    kwargs: Dict[str, Any] = {"want_x8": bool(want_x8)}
    if xt8 is not None:
        kwargs["xt_out"] = xt8
    for key, value in (outs or {}).items():
        kwargs[key] = value
    unknown = sorted(set(kwargs) - set(_PRODUCER_KW))
    if unknown:
        raise RuntimeError(f"adapter bug: producer keywords {unknown} outside _PRODUCER_KW")
    return (x, weight, bias, float(eps), float(inv_scale), int(mp)), kwargs


def _plain_empty(shape: Sequence[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty(tuple(int(d) for d in shape), dtype=dtype, device=device)


def _poisoned_outputs(m: int, mp: int, device: torch.device, want_xt8: bool,
                      want_x8: bool) -> Dict[str, torch.Tensor]:
    """Plain-torch output buffers (never the TM_NONEXP pool) prefilled with 0xA5 bytes / NaN, so an
    unwritten byte cannot pass a check."""
    t = mp // TILE
    u8 = torch.uint8
    outs = {
        "x6_out": torch.full((t, KSTEPS, X6_ITEM_BYTES), 0xA5, dtype=u8, device=device),
        "xs_out": torch.full((t, KSTEPS, XS_ITEM_BYTES), 0xA5, dtype=u8, device=device),
        "mean_out": torch.full((m,), float("nan"), dtype=torch.float32, device=device),
        "rstd_out": torch.full((m,), float("nan"), dtype=torch.float32, device=device),
    }
    if want_xt8:
        outs["xt_out"] = torch.full((t, K, TILE), 0xA5, dtype=u8, device=device).view(_FP8_E4M3)
    if want_x8:
        outs["x8_out"] = torch.full((m, K), 0xA5, dtype=u8, device=device).view(_FP8_E4M3)
    return outs


def ln_store(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    inv_scale: float,
    *,
    pad_to: int = TILE,
    want_xt8: bool = True,
    want_x8: bool = False,
    kind: str = "uvqk",
    xt_empty: Optional[Callable[..., torch.Tensor]] = None,
    poison: bool = False,
) -> Plan:
    """Run the r14 MX6 LN producer: x6 / xs (+ the step-tiled xT8, + mean / rstd, + the e4m3 x8
    when asked). The xT8 comes from `xt_empty` (default fp8_tm_alloc.empty_xt, the store_t's
    allocator); `poison=True` (the self-check and the check call) passes plain-torch outputs
    prefilled with poison instead. Raises on a contract violation."""
    from generative_recommenders.ops.triton import fp8_tm_alloc, wgrad_ck

    prod = _module("producer")
    if prod is None:
        raise RuntimeError(_mod_error["producer"])
    m = int(x.shape[0])
    mp = (m + pad_to - 1) // pad_to * pad_to
    t = (m + TILE - 1) // TILE
    outs: Optional[Dict[str, torch.Tensor]] = None
    xt8: Optional[torch.Tensor] = None
    if poison:
        outs = _poisoned_outputs(m, t * TILE, x.device, want_xt8, want_x8)
        xt8 = outs.pop("xt_out", None)
    elif want_xt8:
        shape = wgrad_ck.tiled_shape(K, mp)
        if shape is None:
            raise RuntimeError(f"step-tiled xT does not apply (D={K}, Mp={mp})")
        xt8 = (fp8_tm_alloc.empty_xt if xt_empty is None else xt_empty)(shape, _FP8_E4M3, x.device)
    args, kwargs = _producer_call(x, weight, bias, eps, inv_scale, mp, want_x8=want_x8, xt8=xt8, outs=outs)
    out = prod.ln_store_mx6(*args, **kwargs)
    if not isinstance(out, (tuple, list)) or len(out) != 6:
        raise RuntimeError(f"{PRODUCER_MODULE}.ln_store_mx6 returned {type(out).__name__} (want 6 items)")
    x6, xs, xt8_out, mean, rstd, x8 = out
    problems = []
    if x6.dtype != torch.uint8 or tuple(x6.shape) != (t, KSTEPS, X6_ITEM_BYTES) or not x6.is_contiguous():
        problems.append(f"x6 {x6.dtype} {tuple(x6.shape)} (want contiguous uint8 [{t}, {KSTEPS}, {X6_ITEM_BYTES}])")
    if xs.dtype != torch.uint8 or tuple(xs.shape) != (t, KSTEPS, XS_ITEM_BYTES) or not xs.is_contiguous():
        problems.append(f"xs {xs.dtype} {tuple(xs.shape)} (want contiguous uint8 [{t}, {KSTEPS}, {XS_ITEM_BYTES}])")
    if xt8 is not None and xt8_out is not xt8:
        problems.append("xT8 is not the caller-allocated tensor")
    for name, value in (("mean", mean), ("rstd", rstd)):
        if value.dtype != torch.float32 or tuple(value.shape) != (m,):
            problems.append(f"{name} {value.dtype} {tuple(value.shape)} (want fp32 [{m}])")
    if want_x8 and (x8 is None or x8.dtype != _FP8_E4M3 or tuple(x8.shape) != (m, K)):
        problems.append(f"x8 {None if x8 is None else (x8.dtype, tuple(x8.shape))} (want e4m3 [{m}, {K}])")
    if outs is not None:
        for name, value in (("x6", x6), ("xs", xs), ("mean", mean), ("rstd", rstd), ("x8", x8)):
            given = outs.get(f"{name}_out")
            if given is not None and value is not given:
                problems.append(f"{name} is not the caller-allocated tensor")
    if problems:
        raise RuntimeError(f"{PRODUCER_MODULE}.ln_store_mx6 broke its contract: " + "; ".join(problems))
    if xt8 is not None and kind not in _INTERNAL_KINDS:
        wgrad_ck.note_tiled_producer("ln_store_mx6", x.device, True, f"M={m} D={K} Mp={mp}")
    return Plan(
        kind=kind,
        device=x.device,
        m=m,
        x6=x6,
        xs=xs,
        xt8=xt8,  # None without want_xt8: the producer's own xT8 is dropped here
        mean=mean,
        rstd=rstd,
        x8=x8 if want_x8 else None,
        x6_shape=tuple(x6.shape),
        xs_shape=tuple(xs.shape),
    )


def _merge_windows(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for a, b in sorted((a, b) for a, b in spans if b > a):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _boundary_windows(m: int, row_bytes: Sequence[int]) -> List[Tuple[int, int]]:
    """Tile-aligned 3-tile windows around every row that holds byte offset k * 2^31 of a tensor
    with `row_bytes` bytes per row (k >= 1, inside M rows)."""
    spans = []
    for per_row in row_bytes:
        k = 1
        while k * _TWO31 < m * per_row:
            row = (k * _TWO31) // per_row
            spans.append((max(0, (row // TILE - 1) * TILE), min(m, (row // TILE + 2) * TILE)))
            k += 1
    return spans


def check_windows(m: int) -> List[Tuple[int, int]]:
    """Row windows of the producer's x6 / xs slice check: the first CHECK_HEAD_ROWS rows, the rows
    around each 2^31-byte offset of x (bf16), x8 / xT8 (e4m3) and x6, and the last tile."""
    if m <= 0:
        return []
    spans = [(0, min(m, CHECK_HEAD_ROWS)), ((m - 1) // TILE * TILE, m)]
    spans += _boundary_windows(m, (2 * K, K, 3 * K // 4))
    return _merge_windows(spans)


def guard_windows(m: int, n: int, itemsize: int) -> List[Tuple[int, int]]:
    """Row windows of the GEMM guard: the first GUARD_HEAD_ROWS rows, the rows around each
    2^31-element and 2^31-byte offset of the [M, N] output, and the last tile."""
    if m <= 0:
        return []
    spans = [(0, min(m, GUARD_HEAD_ROWS)), ((m - 1) // TILE * TILE, m)]
    spans += _boundary_windows(m, (n, n * itemsize))
    return _merge_windows(spans)


def _slice_check(plan: Plan, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                 inv_scale: float, y8: torch.Tensor) -> Dict[str, torch.Tensor]:
    """x6 / xs of the check windows vs the torch reference of the exact fp32 y (mx6_ref), and that
    reference's e4m3 vs the store_t's bytes (a self-test of the emulation). Device counts, no sync."""
    from generative_recommenders.ops.hip import mx6_ref

    assert plan.x6 is not None and plan.xs is not None
    u8 = torch.uint8
    zero = torch.zeros((), dtype=torch.int64, device=x.device)
    bad = {"x6_ref": zero.clone(), "xs_ref": zero.clone(), "ref_vs_store_t": zero.clone()}
    rows = 0
    for r0, r1 in check_windows(plan.m):
        y = mx6_ref.ln_y_exact(x[r0:r1], plan.mean[r0:r1], plan.rstd[r0:r1], weight, bias, inv_scale)
        bad["ref_vs_store_t"] += (mx6_ref.e4m3_sat(y).view(u8) != y8[r0:r1].view(u8)).sum()
        x6r, xsr = mx6_ref.x6_xs_reference(y)
        t0 = r0 // TILE
        t1 = t0 + x6r.shape[0]
        bad["x6_ref"] += (plan.x6[t0:t1] != x6r).sum()
        bad["xs_ref"] += (plan.xs[t0:t1] != xsr).sum()
        rows += r1 - r0
    bad["rows"] = torch.full((), rows, dtype=torch.int64, device=x.device)
    return bad


def _producer_check(plan: Plan, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float,
                    inv_scale: float, pad_to: int, tiled_xt: bool) -> Optional[str]:
    """First producer call per device (see GUARDS (1) in the module docstring). One extra store_t,
    one extra producer call, the reference on row slices and one host sync, once per device.
    None = OK (logged); else the mismatch (the caller latches the producer)."""
    from generative_recommenders.ops.triton.triton_layer_norm import (
        triton_weighted_layer_norm_fwd_fp8_store_t,
    )

    index = _dev_index(x.device)
    u8, i32 = torch.uint8, torch.int32
    y8, mean, rstd, yt = triton_weighted_layer_norm_fwd_fp8_store_t(
        x=x, weight=weight, bias=bias, eps=eps, inv_scale=inv_scale, pad_to=pad_to, tiled_xt=tiled_xt
    )
    if plan.xt8 is None or tuple(yt.shape) != tuple(plan.xt8.shape):
        return f"producer check: xT8 {None if plan.xt8 is None else tuple(plan.xt8.shape)} vs store_t {tuple(yt.shape)}"
    assert plan.x6 is not None and plan.xs is not None
    zero = torch.zeros((), dtype=torch.int64, device=x.device)
    counts: Dict[str, torch.Tensor] = {
        "mean": (plan.mean.view(i32) != mean.view(i32)).sum(),
        "rstd": (plan.rstd.view(i32) != rstd.view(i32)).sum(),
        "xT8": (plan.xt8.view(u8) != yt.view(u8)).sum(),
        "x8": (plan.x8.view(u8) != y8.view(u8)).sum() if plan.x8 is not None else zero,
    }
    del yt, mean, rstd
    counts.update(_slice_check(plan, x, weight, bias, inv_scale, y8))
    # The other kernel instantiation (x8 written or not) on the same x into fresh poisoned buffers:
    # the steady-state calls run the no-x8 kernel once U is MX6 (and VK always after its guard).
    other_x8 = plan.x8 is None
    if not other_x8:
        y8 = None
    other = ln_store(x, weight, bias, eps, inv_scale, pad_to=pad_to, want_xt8=True, want_x8=other_x8,
                     kind="check", poison=True)
    assert other.x6 is not None and other.xs is not None and other.xt8 is not None
    counts["x6_inst"] = (other.x6 != plan.x6).sum()
    counts["xs_inst"] = (other.xs != plan.xs).sum()
    counts["xT8_inst"] = (other.xt8.view(u8) != plan.xt8.view(u8)).sum()
    counts["mean_inst"] = (other.mean.view(i32) != plan.mean.view(i32)).sum()
    counts["rstd_inst"] = (other.rstd.view(i32) != plan.rstd.view(i32)).sum()
    counts["x8_inst"] = ((other.x8.view(u8) != y8.view(u8)).sum()
                         if other_x8 and other.x8 is not None and y8 is not None else zero)
    del other, y8
    values = dict(zip(counts, torch.stack([v.to(torch.int64) for v in counts.values()]).tolist()))
    rows = values.pop("rows")
    detail = (
        f"M={plan.m} x8={'yes' if plan.x8 is not None else 'no'}+{'yes' if other_x8 else 'no'} differing: "
        f"vs store_t mean {values['mean']}, rstd {values['rstd']}, xT8 {values['xT8']} B, "
        f"x8 {values['x8'] if plan.x8 is not None else 'n/a'} B; x6 / xs vs reference on {rows} rows "
        f"{values['x6_ref']} / {values['xs_ref']} B (reference e4m3 vs store_t {values['ref_vs_store_t']} B); "
        f"other instantiation x6 {values['x6_inst']}, xs {values['xs_inst']}, xT8 {values['xT8_inst']} B, "
        f"mean {values['mean_inst']}, rstd {values['rstd_inst']}, x8 {values['x8_inst'] if other_x8 else 'n/a'}"
    )
    if any(values.values()):
        _print(f"ERROR producer check on cuda:{index}: {detail} -> producer latched")
        return f"producer check failed ({detail})"
    _producer_checked.add(index)
    _print(f"producer check on cuda:{index}: {detail} -> ok")
    return None


def _operand_reason(x: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor,
                    uvqk_weight: torch.Tensor, uvqk_bias: torch.Tensor) -> Optional[Tuple[str, str]]:
    if torch.compiler.is_compiling():
        return ("compile", "torch.compile tracing (ctypes launches)")
    if _FP8_E4M3 is None:
        return ("dtype", "torch has no float8_e4m3fn")
    if x.device.type != "cuda":
        return ("device", f"x on {x.device}")
    if x.device.index is not None and x.device.index != torch.cuda.current_device():
        return ("device", f"x on {x.device}, current device cuda:{torch.cuda.current_device()}")
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != K or not x.is_contiguous():
        return ("activation", f"x {x.dtype} {tuple(x.shape)} (needs contiguous bf16 [M, {K}])")
    if (int(x.shape[0]) + TILE - 1) // TILE * TILE > _INT32_MAX:
        return ("rows", f"M={x.shape[0]}: roundup(M, {TILE}) exceeds int32 (the producer's bound)")
    for name, t in (("norm_weight", norm_weight), ("norm_bias", norm_bias)):
        if t.dim() != 1 or t.numel() != K or t.device != x.device:
            return ("norm", f"{name} {t.dtype} {tuple(t.shape)} on {t.device}")
    if uvqk_weight.dim() != 2 or uvqk_weight.shape[0] != K or uvqk_weight.device != x.device:
        return ("weight", f"uvqk_weight {tuple(uvqk_weight.shape)} on {uvqk_weight.device}")
    if uvqk_bias.dim() != 1 or uvqk_bias.numel() != uvqk_weight.shape[1]:
        return ("bias", f"uvqk_bias {tuple(uvqk_bias.shape)} (needs 1-D [{uvqk_weight.shape[1]}])")
    return None


def _producer_operand_reason(prod: Any, x: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor) -> Optional[str]:
    """The producer's own eligibility verdict on these operands (dtype, strides, 16-B alignment,
    row bound, current device): None or a reason. Declines the call, never latches."""
    try:
        reason = prod.ineligible_reason(x, weight, bias)
    except Exception as error:
        return f"{PRODUCER_MODULE}.ineligible_reason raised {type(error).__name__}: {error}"
    return None if not reason else str(reason)


def _scale_mode_reason(device: torch.device) -> Optional[Tuple[str, str]]:
    from generative_recommenders.ops.triton import triton_fp8

    mode = triton_fp8._fp8_output_scale_mode(device)
    if mode != "input_scale":
        return ("mode", f"FP8 output scale mode {mode} (MX6 implements the input_scale fold only)")
    return None


def _plan(
    kind: str,
    sites: Tuple[str, ...],
    *,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    fp8_scale_value: Optional[float],
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    qkv_frozen: bool,
    grad: bool,
    store_t: bool,
    pack_to_backward: bool,
    pad_to: int,
    tiled_xt: bool,
    optional_u: bool,
) -> Optional[Plan]:
    if not ENABLED:
        return None
    # Expected non-engagement first: the calibration window (Q/K/V scale not frozen: the shipped
    # forward runs the bf16-output uvqk GEMM) and no-grad / eval forwards (no store_t).
    if not qkv_frozen or fp8_scale_value is None:
        _skip(f"{kind}:calibration", "Q/K/V output scale not frozen yet (calibration window)")
        return None
    if not grad:
        _skip(f"{kind}:no_grad", "no-grad / eval forward (no store_t): keeps the FP8 recipe")
        return None
    if x.shape[0] == 0:
        _skip(f"{kind}:empty", "M = 0")
        return None
    if not store_t:
        _decline(f"{kind}:store_t", "the shipped call would not take the LN store_t (fp8t wgrad not "
                 "planned for this call, or DLRM_HSTU_FP8_BWD_UVQK_STORE_T off)")
        return None
    if not pack_to_backward:
        _decline(f"{kind}:pack", "the e4m3 pack would not reach the backward (DLRM_HSTU_FP8_SAVE_UVQK"
                 " / _EARLY_RELEASE_UVQK / layer mask): its FP8 recompute would not match an MX6 forward")
        return None
    for role in ("producer", "gemm"):
        if _module(role) is None:
            _decline(f"{role}_module", _mod_error[role])
            return None
    reason = _operand_reason(x, norm_weight, norm_bias, uvqk_weight, uvqk_bias)
    if reason is None:
        reason = _scale_mode_reason(x.device)
    mp = (int(x.shape[0]) + pad_to - 1) // pad_to * pad_to
    if reason is None:
        xt = _xt_layout_reason(tiled_xt, mp, pad_to)
        reason = None if xt is None else ("xt_layout", xt)
    if reason is not None:
        _decline(f"{kind}:{reason[0]}", reason[1])
        return None
    for site in sites:
        why = site_reason(x.device, site)
        if why is not None:
            _decline(f"{kind}:{site}", why)
            return None
    why = _producer_operand_reason(_mods["producer"], x, norm_weight, norm_bias)
    if why is not None:
        _decline(f"{kind}:producer_operands", why)  # this call only; nothing is latched
        return None
    u_mx6 = False
    if optional_u:
        why = site_reason(x.device, "U")
        u_mx6 = why is None
        if not u_mx6:
            if u_expected():  # option C, U not ready: STRICT raises here, before the producer runs
                message = _u_not_ready(f"{kind} call on cuda:{_dev_index(x.device)}", why)
                if STRICT:
                    _decline(f"{kind}:U", message)  # raises StrictError
                _u_fallback(message)
            _log_once(("u_fp8", why), f"U stays on the shipped FP8 GEMM ({why}); the producer also "
                      "writes the e4m3 x8 for it")
    wkey = _weight_key(uvqk_weight)
    active = sites + (("U",) if u_mx6 else ())
    guard = tuple(site for site in active if _guard_pending(x.device, site, wkey))
    # x8: the shipped FP8 U reads it while U is not on MX6, and the guard's FP8 reference needs it.
    want_x8 = (optional_u and not u_mx6) or bool(guard)
    index = _dev_index(x.device)
    checking = index not in _producer_checked
    first = bool(guard) or checking
    try:
        plan = ln_store(x, norm_weight, norm_bias, eps, 1.0 / fp8_scale_value, pad_to=pad_to,
                        want_xt8=True, want_x8=want_x8, kind=kind)
        why = None
        if checking:
            why = _producer_check(plan, x, norm_weight, norm_bias, eps, 1.0 / fp8_scale_value,
                                  pad_to, tiled_xt)
    except Exception as error:
        if not first or STRICT:
            raise
        why = f"producer failed on a first call ({type(error).__name__}: {error})"
    if why is not None:
        # Nothing has consumed the producer's outputs yet: latch and let the shipped LN run.
        _producer_error[index] = why
        _decline(f"{kind}:producer", why)
        return None
    plan.u_mx6 = u_mx6
    plan.guard = guard
    plan.wkey = wkey
    _count(f"planned_{kind}")
    return plan


def uvqk_ln_store(
    *,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    fp8_scale_value: Optional[float],
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    u_columns: int,
    qkv_frozen: bool,
    grad: bool,
    store_t: bool,
    pack_saved: bool,
    pad_to: int,
    tiled_xt: bool,
) -> Optional[Plan]:
    """Ordinary uvqk layer, LN arm: the MX6 producer in place of the store_t, or None (the
    shipped LN branches run; logged once per reason). `store_t` / `pack_saved` are the call
    site's own predicates (the store_t branch would run; the forward saves the e4m3 V/Q/K pack)."""
    if ENABLED and (u_columns != SITE_N["U"] or uvqk_weight.dim() != 2
                    or uvqk_weight.shape[1] - u_columns != SITE_N["VQK"]):
        _decline("uvqk:shape", f"uvqk_weight {tuple(uvqk_weight.shape)} u_columns={u_columns} "
                 f"(MX6 covers U N={SITE_N['U']}, V/Q/K N={SITE_N['VQK']})")
        return None
    return _plan("uvqk", ("VQK",), x=x, norm_weight=norm_weight, norm_bias=norm_bias, eps=eps,
                 fp8_scale_value=fp8_scale_value, uvqk_weight=uvqk_weight, uvqk_bias=uvqk_bias,
                 qkv_frozen=qkv_frozen, grad=grad, store_t=store_t, pack_to_backward=pack_saved,
                 pad_to=pad_to, tiled_xt=tiled_xt, optional_u=True)


def vk_ln_store(
    *,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    fp8_scale_value: Optional[float],
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    vk_layout: Tuple[Tuple[int, int], ...],
    qkv_frozen: bool,
    grad: bool,
    store_t: bool,
    pack_handed: bool,
    pad_to: int,
    tiled_xt: bool,
) -> Optional[Plan]:
    """Targets-only layer, LN arm: the MX6 producer in place of the store_t (no x8 after the
    guard call: V/K is the only reader of the row-major e4m3 normed_x there)."""
    if ENABLED and sum(b - a for a, b in vk_layout) != SITE_N["VK"]:
        _decline("vk:shape", f"vk layout {vk_layout} (MX6 covers V/K N={SITE_N['VK']})")
        return None
    return _plan("vk", ("VK",), x=x, norm_weight=norm_weight, norm_bias=norm_bias, eps=eps,
                 fp8_scale_value=fp8_scale_value, uvqk_weight=uvqk_weight, uvqk_bias=uvqk_bias,
                 qkv_frozen=qkv_frozen, grad=grad, store_t=store_t, pack_to_backward=pack_handed,
                 pad_to=pad_to, tiled_xt=tiled_xt, optional_u=False)


# ----------------------------------------------------------------------------- GEMMs
def _fold_scales(
    scale_a: torch.Tensor,
    output_scale: torch.Tensor,
    scale_a_value: Optional[float],
    output_scale_value: Optional[float],
    bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(effective scale_a, scaled bf16 bias) exactly as scaled_mm_fp8_out_preq builds them in the
    input_scale backend mode (same helpers, same caches, same values)."""
    from generative_recommenders.ops.triton import triton_fp8 as f8

    host = f8._trusted_host_scale_values(scale_a, output_scale, scale_a_value, output_scale_value)
    effective_scale_a = f8._fold_output_scale_into_input(
        scale_a,
        output_scale,
        scale_a_value=None if host is None else host[0],
        output_scale_value=None if host is None else host[1],
    )
    effective_bias = f8._scale_bias_for_fp8_output(
        bias,
        output_scale,
        output_scale_value=None if host is None else host[1],
    )
    return effective_scale_a, effective_bias


def _gemm_call(x6: torch.Tensor, xs: torch.Tensor, w6: torch.Tensor, ws: torch.Tensor, bias: torch.Tensor,
               sa: torch.Tensor, sb: torch.Tensor, m: int, n: int,
               out: torch.Tensor) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """(args, kwargs) of pr7's k512_mx6 call -- the only place that builds it (contract test)."""
    return (x6, xs, w6, ws, bias, sa, sb, int(m), int(n)), {"out": out}


def _launch(
    site: str,
    plan: Plan,
    weight: torch.Tensor,
    layout: Tuple[Tuple[int, int], ...],
    bias: torch.Tensor,
    sa: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    gemm = _mods["gemm"]
    n = SITE_N[site]
    assert plan.x6 is not None and plan.xs is not None
    w6, ws, sb = weight_shadow(weight, layout)
    out = torch.empty((plan.m, n), dtype=out_dtype, device=plan.device)
    args, kwargs = _gemm_call(plan.x6, plan.xs, w6, ws, bias, sa, sb, plan.m, n, out)
    result = gemm.k512_mx6(*args, **kwargs)
    if result is not None and result is not out:
        # A view or a differently strided tensor would change the saved u / vqk / vk.
        raise RuntimeError(
            f"{GEMM_MODULE}.k512_mx6 returned a tensor other than `out` "
            f"({getattr(result, 'dtype', type(result).__name__)} {tuple(getattr(result, 'shape', ()))}): "
            "the adapter requires the output written in place"
        )
    return out


def _nonfinite_count(t: torch.Tensor, chunk_rows: int = GUARD_CHUNK_ROWS) -> torch.Tensor:
    """Device count of NaN / Inf values of a 2-D output, from bit patterns (e4m3fn: S.1111.111,
    bf16: exponent all ones), chunked over rows; no float copy of the output."""
    count = torch.zeros((), dtype=torch.int64, device=t.device)
    for r0 in range(0, int(t.shape[0]), chunk_rows):
        part = t[r0:r0 + chunk_rows]
        if _FP8_E4M3 is not None and t.dtype == _FP8_E4M3:
            count += ((part.view(torch.uint8) & 0x7F) == 0x7F).sum()
        elif t.dtype == torch.bfloat16:
            count += ((part.view(torch.int16) & 0x7F80) == 0x7F80).sum()
        else:
            count += (~torch.isfinite(part)).sum()
    return count


def guard_stats(
    got: torch.Tensor,
    reference_rows: Callable[[int, int], torch.Tensor],
    windows: Sequence[Tuple[int, int]],
    chunk_rows: int = GUARD_CHUNK_ROWS,
) -> Dict[str, float]:
    """rel L2 / max per-128-row-tile rel L2 / slope <got, ref> / <ref, ref> of `got` vs the FP8
    reference on tile-aligned row `windows` (reference_rows(r0, r1) computes them), plus the
    non-finite count of the FULL `got`; fp64 sums, chunked, one host sync."""
    nonfinite = _nonfinite_count(got, chunk_rows)
    step = max(TILE, chunk_rows // TILE * TILE)
    tile_d2: List[torch.Tensor] = []
    tile_r2: List[torch.Tensor] = []
    tile_gr: List[torch.Tensor] = []
    rows = 0
    for w0, w1 in windows:
        ref = reference_rows(w0, w1)
        for r0 in range(w0, w1, step):
            r1 = min(w1, r0 + step)
            a = got[r0:r1].float()
            b = ref[r0 - w0:r1 - w0].float()
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            d = a - b
            # The three sums share one reduction structure, so got == ref gives slope 1 exactly.
            sums = [(d * d).sum(dim=1, dtype=torch.float64), (b * b).sum(dim=1, dtype=torch.float64),
                    (a * b).sum(dim=1, dtype=torch.float64)]
            nt = (r1 - r0 + TILE - 1) // TILE
            pad = nt * TILE - (r1 - r0)
            for target, row_sum in zip((tile_d2, tile_r2, tile_gr), sums):
                if pad:
                    row_sum = torch.cat((row_sum, row_sum.new_zeros(pad)))
                target.append(row_sum.view(nt, TILE).sum(dim=1))
        rows += w1 - w0
        del ref
    if not tile_d2:
        return {"rel_l2": 0.0, "max_tile_rel_l2": 0.0, "slope": 1.0,
                "nonfinite": float(nonfinite.item()), "rows": 0}
    d2 = torch.cat(tile_d2)
    r2 = torch.cat(tile_r2)
    total_d2 = d2.sum()
    total_r2 = r2.sum()
    gr = torch.cat(tile_gr).sum()
    rel = torch.sqrt(total_d2 / total_r2.clamp_min(1e-300))
    floor = (total_r2 / d2.numel()).clamp_min(1e-300) * 1e-6
    tile_rel = torch.sqrt(d2 / torch.maximum(r2, floor)).max()
    slope = torch.where(total_r2 > 0, gr / total_r2.clamp_min(1e-300), torch.ones_like(gr))
    rel_v, tile_v, slope_v, bad_v = torch.stack(
        (rel, tile_rel, slope, nonfinite.to(torch.float64))).tolist()
    return {"rel_l2": float(rel_v), "max_tile_rel_l2": float(tile_v), "slope": float(slope_v),
            "nonfinite": float(bad_v), "rows": rows}


def guard_metrics(
    got: torch.Tensor, ref: torch.Tensor, chunk_rows: int = GUARD_CHUNK_ROWS
) -> Tuple[float, float, bool, float]:
    """Full-tensor form of guard_stats (tests): (rel L2, max tile rel L2, all-finite(got), slope)."""
    m = int(got.shape[0])
    stats = guard_stats(got, lambda a, b: ref[a:b], [(0, m)] if m else [], chunk_rows)
    return stats["rel_l2"], stats["max_tile_rel_l2"], stats["nonfinite"] == 0, stats["slope"]


def guard_accepts(stats: Dict[str, float]) -> bool:
    return (
        stats["nonfinite"] == 0
        and stats["rel_l2"] <= GUARD_REL_L2_MAX
        and stats["max_tile_rel_l2"] <= GUARD_TILE_REL_L2_MAX
        and abs(stats["slope"] - 1.0) <= GUARD_SLOPE_MAX
    )


def _first_call(
    site: str,
    plan: Plan,
    run: Callable[[], torch.Tensor],
    reference_rows: Callable[[int, int], torch.Tensor],
    reference_full: Callable[[], torch.Tensor],
    itemsize: int,
) -> torch.Tensor:
    """The guarded first call per (device, site, weight): accept the MX6 output, or latch the site
    and return the shipped FP8 output of the full call. Any exception (the MX6 call, the guard's
    references) latches too; under STRICT it propagates."""
    index = _dev_index(plan.device)
    n = SITE_N[site]
    try:
        got = run()
    except Exception as error:
        if STRICT:
            raise
        _latch(plan.device, site, f"first call raised {type(error).__name__}: {error}")
        _print(f"ERROR {site} on cuda:{index}: first MX6 call raised {type(error).__name__}: {error} "
               "-> site latched to the shipped FP8 path")
        out = reference_full()
        _log_armed(plan.device)
        return out
    windows = guard_windows(plan.m, n, itemsize)
    try:
        stats = guard_stats(got, reference_rows, windows)
    except Exception as error:
        if STRICT:
            raise
        got = None  # free the MX6 output before the full FP8 call
        _latch(plan.device, site, f"guard raised {type(error).__name__}: {error}")
        _print(f"ERROR guard {site} on cuda:{index}: raised {type(error).__name__}: {error} "
               "-> site latched to the shipped FP8 path")
        out = reference_full()
        _log_armed(plan.device)
        return out
    ok = guard_accepts(stats)
    weight_id = f"{plan.wkey[0]:#x}" if plan.wkey else "?"
    line = (
        f"guard {site} on cuda:{index}: M={plan.m} N={n} weight={weight_id} "
        f"rows={stats['rows']} in {len(windows)} windows rel_l2={stats['rel_l2']:.4g} "
        f"max_tile_rel_l2={stats['max_tile_rel_l2']:.4g} slope={stats['slope']:.5f} "
        f"nonfinite={int(stats['nonfinite'])} (accept: no non-finite output, rel_l2 <= {GUARD_REL_L2_MAX}, "
        f"tile <= {GUARD_TILE_REL_L2_MAX}, |slope - 1| <= {GUARD_SLOPE_MAX}; vs the shipped FP8 GEMM "
        "on the same rows)"
    )
    if ok:
        _site_armed.add((index, site, plan.wkey))
        _print(line + " -> armed")
        _log_armed(plan.device)
        return got
    got = None  # free the MX6 output before the full FP8 call
    _latch(plan.device, site, f"guard rejected: rel_l2={stats['rel_l2']:.4g} tile={stats['max_tile_rel_l2']:.4g} "
           f"slope={stats['slope']:.4g} nonfinite={int(stats['nonfinite'])}")
    _print("ERROR " + line + " -> REJECTED: site latched to the shipped FP8 path for this process")
    if STRICT:
        raise StrictError(f"{KNOB}_STRICT=1: MX6 {site} guard rejected ({line})")
    out = reference_full()
    _log_armed(plan.device)
    return out


def gemm_u(
    plan: Plan,
    *,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    u_columns: int,
    fp8_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """U = bf16(acc * (scale_a * sb) + bias) [M, 512] on pr7's N=512 MX6 kernel. The guard
    reference is the shipped call (triton_hstu_preprocess_and_attention, ordinary forward)."""
    from generative_recommenders.ops.triton.triton_fp8 import scaled_addmm_fp8_preq

    layout = ((0, u_columns),)
    bias = uvqk_bias[:u_columns].to(out_dtype)  # scaled_addmm_fp8_preq: bias_or_residual.to(out_dtype)

    def run() -> torch.Tensor:
        return _launch("U", plan, uvqk_weight, layout, bias, fp8_scale, out_dtype)

    def reference_rows(r0: int, r1: int) -> torch.Tensor:
        assert plan.x8 is not None
        return scaled_addmm_fp8_preq(
            activation_fp8=plan.x8[r0:r1],
            scale_a=fp8_scale,
            weight=uvqk_weight,
            bias_or_residual=uvqk_bias[:u_columns],
            out_dtype=out_dtype,
            weight_layout=layout,
            data_parallel=True,
        )

    def reference_full() -> torch.Tensor:
        return scaled_addmm_fp8_preq(
            activation_fp8=plan.x8,
            scale_a=fp8_scale,
            weight=uvqk_weight,
            bias_or_residual=uvqk_bias[:u_columns],
            out_dtype=out_dtype,
            weight_layout=layout,
            data_parallel=True,
        ).contiguous()

    if "U" in plan.guard:
        out = _first_call("U", plan, run, reference_rows, reference_full, out_dtype.itemsize)
        served = (_dev_index(plan.device), "U", plan.wkey) in _site_armed
    else:
        out = run()
        served = True
    if served:
        _count("served_U")
    return out


def _gemm_e4m3(
    site: str,
    plan: Plan,
    *,
    weight: torch.Tensor,
    layout: Tuple[Tuple[int, int], ...],
    bias: torch.Tensor,
    fp8_scale: torch.Tensor,
    fp8_scale_value: Optional[float],
    fp8_qkv_scale: torch.Tensor,
    qkv_scale_value: Optional[float],
) -> Tuple[torch.Tensor, bool]:
    from generative_recommenders.ops.triton import triton_fp8 as f8

    sa, bias_eff = _fold_scales(fp8_scale, fp8_qkv_scale, fp8_scale_value, qkv_scale_value, bias)

    def run() -> torch.Tensor:
        return _launch(site, plan, weight, layout, bias_eff, sa, _FP8_E4M3)

    def reference(rows: Optional[Tuple[int, int]]) -> torch.Tensor:
        assert plan.x8 is not None
        return f8.scaled_mm_fp8_out_preq(
            activation_fp8=plan.x8 if rows is None else plan.x8[rows[0]:rows[1]],
            scale_a=fp8_scale,
            weight=weight,
            bias=bias,
            weight_layout=layout,
            output_scale=fp8_qkv_scale,
            scale_a_value=fp8_scale_value,
            output_scale_value=qkv_scale_value,
        )

    if site in plan.guard:
        out = _first_call(site, plan, run, lambda r0, r1: reference((r0, r1)),
                          lambda: reference(None).contiguous(), 1)
        served = (_dev_index(plan.device), site, plan.wkey) in _site_armed
    else:
        out = run()
        served = True
    if served:
        # Same telemetry key as scaled_mm_fp8_out_preq's output record (no-op unless
        # DLRM_HSTU_FP8_TELEMETRY_INTERVAL > 0).
        f8.maybe_record_fp8_telemetry(
            f"qkv_output:{id(weight)}:{tuple(weight.shape)}:{layout}",
            out,
            f8.fp8_scale_tensor(fp8_qkv_scale),
        )
        _count(f"served_{site}")
    return out, served


def gemm_vqk(
    plan: Plan,
    *,
    ctx: Any,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    u_columns: int,
    fp8_scale: torch.Tensor,
    fp8_scale_value: Optional[float],
    fp8_qkv_scale: torch.Tensor,
    qkv_scale_value: Optional[float],
) -> torch.Tensor:
    """V/Q/K e4m3 [M, 1536] at the frozen qkv scale; marks ctx so a retained-graph backward
    recomputes it through MX6. The last GEMM of the layer call (U runs first): releases the
    plan's operand references."""
    out, served = _gemm_e4m3(
        "VQK", plan, weight=uvqk_weight, layout=((u_columns, uvqk_weight.shape[1]),),
        bias=uvqk_bias[u_columns:], fp8_scale=fp8_scale, fp8_scale_value=fp8_scale_value,
        fp8_qkv_scale=fp8_qkv_scale, qkv_scale_value=qkv_scale_value,
    )
    ctx._mx6_vqk = served
    _log_armed(plan.device)
    plan.release()
    return out


def gemm_vk(
    plan: Plan,
    *,
    ctx: Any,
    uvqk_weight: torch.Tensor,
    b_vk: torch.Tensor,
    vk_layout: Tuple[Tuple[int, int], ...],
    fp8_scale: torch.Tensor,
    fp8_scale_value: Optional[float],
    fp8_qkv_scale: torch.Tensor,
    qkv_scale_value: Optional[float],
) -> torch.Tensor:
    """Targets V/K e4m3 [M, 1024]; marks ctx for the retained-graph recompute."""
    out, served = _gemm_e4m3(
        "VK", plan, weight=uvqk_weight, layout=tuple(vk_layout), bias=b_vk, fp8_scale=fp8_scale,
        fp8_scale_value=fp8_scale_value, fp8_qkv_scale=fp8_qkv_scale, qkv_scale_value=qkv_scale_value,
    )
    ctx._mx6_vk = served
    _log_armed(plan.device)
    plan.release()
    return out


def _site_state(device: torch.device, site: str) -> Optional[str]:
    """'mx6' (guard passed for at least one weight), 'FP8 (...)' (latched / not available), or
    None while the site's first guard is still pending on this device."""
    index = _dev_index(device)
    latched = _site_latch.get((index, site))
    if latched is not None:
        return f"FP8 (latched: {latched})"
    if any(key[0] == index and key[1] == site for key in _site_armed):
        return "mx6"
    why = not_ready_reason(device, (SITE_N[site],))
    if why is not None:
        return f"FP8 ({why})"
    return None


def _log_armed(device: torch.device) -> None:
    """One device line, once U, VQK and VK are all decided (guard passed, latched or not
    available), naming each site's actual state."""
    index = _dev_index(device)
    if index in _armed_devices:
        return
    states = {site: _site_state(device, site) for site in SITE_N}
    if any(state is None for state in states.values()):
        return
    _armed_devices.add(index)
    head = "ARMED" if states["VQK"] == "mx6" and states["VK"] == "mx6" else "ERROR NOT ARMED"
    _print(
        f"{head} on cuda:{index} ({KNOB}=1, DECLARED PRECISION CHANGE: E2M3 operands, E8M0 per 32 K, "
        f"W 2-D 32x32): producer={PRODUCER_MODULE}.ln_store_mx6 [{_producer_desc()}] "
        f"gemm={GEMM_MODULE}.k512_mx6 sites "
        + " ".join(f"{site}:N={n}:{states[site]}" for site, n in SITE_N.items())
    )


def record_contract(site: str, plan: Plan, **tensors: torch.Tensor) -> None:
    """Capability / contract record of an MX6-served call without an e4m3 normed_x (the shipped
    record needs one): the site plus `<site>_mx6` with the operand formats (dtype / shape only)."""
    from generative_recommenders.ops.triton import triton_fp8 as f8

    f8.record_fp8_capability(site)
    f8.record_fp8_capability(site + "_mx6")
    f8.record_fp8_contract(site + "_mx6", **plan.meta(), **tensors)


# ----------------------------------------------------------------------------- backward recompute
def recompute_vqk(
    ctx: Any,
    *,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    input_scale: torch.Tensor,
    qkv_scale: torch.Tensor,
) -> Optional[torch.Tensor]:
    """A retained-graph backward rebuilding the V/Q/K pack of an MX6 forward: rerun the producer
    (no x8; it writes and drops its own xT8) and the MX6 GEMM. None when the forward did not serve
    MX6 (the shipped FP8 recompute is then the right one) -- or, logged as a deviation, when MX6
    is no longer usable."""
    if not getattr(ctx, "_mx6_vqk", False):
        return None
    return _recompute("VQK", ctx, x=x, norm_weight=norm_weight, norm_bias=norm_bias,
                      weight=uvqk_weight,
                      layout=((ctx.hidden_dim * ctx.num_heads, uvqk_weight.shape[1]),),
                      bias=uvqk_bias[ctx.hidden_dim * ctx.num_heads:],
                      input_scale=input_scale, qkv_scale=qkv_scale)


def recompute_vk(
    ctx: Any,
    *,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    uvqk_weight: torch.Tensor,
    b_vk: torch.Tensor,
    vk_layout: Tuple[Tuple[int, int], ...],
    input_scale: torch.Tensor,
    qkv_scale: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Targets layer twin of recompute_vqk."""
    if not getattr(ctx, "_mx6_vk", False):
        return None
    return _recompute("VK", ctx, x=x, norm_weight=norm_weight, norm_bias=norm_bias,
                      weight=uvqk_weight, layout=tuple(vk_layout), bias=b_vk,
                      input_scale=input_scale, qkv_scale=qkv_scale)


def _recompute(site: str, ctx: Any, *, x, norm_weight, norm_bias, weight, layout, bias,  # pyre-ignore[2]
               input_scale, qkv_scale) -> Optional[torch.Tensor]:
    why = site_reason(x.device, site)
    if why is None and ctx.fp8_input_scale_value is None:
        why = "no frozen input scale"
    if why is None:
        why = _producer_operand_reason(_mods["producer"], x, norm_weight, norm_bias)
    if why is not None:
        _print(f"ERROR retained-graph recompute of {site}: MX6 unavailable ({why}) -> FP8 recompute; "
               "its values differ from this call's MX6 forward")
        return None
    plan = ln_store(x, norm_weight, norm_bias, ctx.norm_eps, 1.0 / ctx.fp8_input_scale_value,
                    want_xt8=False, want_x8=False, kind="recompute")
    sa, bias_eff = _fold_scales(input_scale, qkv_scale, ctx.fp8_input_scale_value,
                                ctx.fp8_qkv_scale_value, bias)
    out = _launch(site, plan, weight, layout, bias_eff, sa, _FP8_E4M3)
    plan.release()
    _count(f"recompute_{site}")
    _log_once(("recompute", site), f"retained-graph backward: {site} recomputed through MX6 "
              "(matches the forward's operands and kernels)")
    return out


# ----------------------------------------------------------------------------- prebuild
def prebuild(warm_rows: int = 257) -> bool:
    """Before RUN_START (triton_fp8.prebuild_extensions, DLRM_HSTU_FP8_PREBUILD=1): import the
    producer and pr7's modules, run their own prebuild(), load the kernels and run the self-checks
    (GUARDS (3) in the module docstring; synthetic data only, plain-torch buffers, no RNG, no model,
    no cache entry, no TM_NONEXP pool block). Returns True when VQK and VK are ready. Not ready:
    one "prebuild DECLINED" line, False, and EVERY site stays on the shipped FP8 path for the
    process (latched); PrebuildError under STRICT. Option C (the GEMM module offers N=512) with U
    not ready counts as not ready under STRICT only; without STRICT VQK / VK stay ready and U stays
    FP8 (option B) after one "ERROR OPTION-C U NOT READY" line (module docstring, OPTION C AND U)."""
    global _prebuild_error
    if not ENABLED:
        return False
    try:
        reason = _prebuild(warm_rows)
    except Exception as error:  # a device fault in the self-check is not swallowed
        if isinstance(error, (PrebuildError, StrictError)):
            raise
        reason = f"{type(error).__name__}: {error}"
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if reason is None:
        return True
    if reason.startswith("skipped:"):
        _print("prebuild " + reason)
        return False
    _prebuild_error = reason
    _print(f"prebuild DECLINED: {reason} -> every MX6 site stays on the shipped FP8 path "
           "(latched for this process)")
    if STRICT:
        raise PrebuildError(f"{KNOB}_STRICT=1: MX6 prebuild not ready: {reason}")
    return False


def _gate_reason() -> Optional[str]:
    """The three import-time gates must agree (an env change between the imports would let the
    prebuild self-check MX6 while the forward never engages, or the reverse)."""
    gates = {"mx6_fwd.ENABLED": ENABLED}
    for module, attr in (("generative_recommenders.ops.triton.triton_fp8", "_HSTU_FP8_FWD_MXFP6"),
                         ("generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention",
                          "_MX6_FWD")):
        mod = sys.modules.get(module)
        if mod is not None and hasattr(mod, attr):
            gates[f"{module.rsplit('.', 1)[-1]}.{attr}"] = bool(getattr(mod, attr))
    if len(set(gates.values())) > 1:
        _print(f"ERROR import-time gates of {KNOB} disagree: {gates}")
        return f"import-time gates disagree {gates}"
    return None


def _ramp_inputs(rows: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Deterministic non-zero LN input (no RNG): many E8M0 exponents and codes per row."""
    r = torch.arange(rows, device=device, dtype=torch.int64).view(-1, 1)
    c = torch.arange(K, device=device, dtype=torch.int64)
    x = ((((r * 37 + c.view(1, -1) * 11) % 199) - 99).to(torch.float32) / 16.0).to(torch.bfloat16)
    w = (1.0 + (((c * 7) % 13) - 6).to(torch.float32) / 64.0).to(torch.bfloat16)
    b = ((((c * 5) % 17) - 8).to(torch.float32) / 32.0).to(torch.bfloat16)
    return x.contiguous(), w, b, 37.5


def _selfcheck_producer(device: torch.device, rows: int) -> Tuple[Optional[str], Optional[Plan]]:
    """The producer on zeros and on the ramp, both instantiations (x8 written or not), into
    poisoned plain-torch buffers; every x6 / xs / xT8 / x8 byte vs the torch reference of the
    exact y (from the producer's own mean / rstd), mean / rstd finite (and 0 on zeros), and the
    two instantiations byte-identical. Returns (reason or None, the zeros plan with x8)."""
    from generative_recommenders.ops.hip import mx6_ref

    u8, i32 = torch.uint8, torch.int32
    zeros = (torch.zeros((rows, K), dtype=torch.bfloat16, device=device),
             torch.ones((K,), dtype=torch.bfloat16, device=device),
             torch.zeros((K,), dtype=torch.bfloat16, device=device), 1.0)
    keep: Optional[Plan] = None
    for name, (x, w, b, inv) in (("zeros", zeros), ("ramp", _ramp_inputs(rows, device))):
        first: Optional[Plan] = None
        for want_x8 in (True, False):
            plan = ln_store(x, w, b, 1e-6, inv, pad_to=TILE, want_xt8=True, want_x8=want_x8,
                            kind="selfcheck", poison=True)
            assert plan.x6 is not None and plan.xs is not None and plan.xt8 is not None
            y = mx6_ref.ln_y_exact(x, plan.mean, plan.rstd, w, b, inv)
            x6r, xsr = mx6_ref.x6_xs_reference(y)
            counts = {
                "x6": (plan.x6 != x6r).sum(),
                "xs": (plan.xs != xsr).sum(),
                "xT8": (plan.xt8.view(u8) != mx6_ref.xt8_reference(y).view(u8)).sum(),
                "mean/rstd non-finite": (~torch.isfinite(plan.mean)).sum() + (~torch.isfinite(plan.rstd)).sum(),
                "xs 255": (plan.xs == 255).sum(),
            }
            if plan.x8 is not None:
                counts["x8"] = (plan.x8.view(u8) != mx6_ref.e4m3_sat(y).view(u8)).sum()
            if name == "zeros":
                counts["mean != 0"] = torch.count_nonzero(plan.mean)
            if first is not None:
                assert first.x6 is not None and first.xs is not None and first.xt8 is not None
                counts["x6 vs x8 leg"] = (plan.x6 != first.x6).sum()
                counts["xs vs x8 leg"] = (plan.xs != first.xs).sum()
                counts["xT8 vs x8 leg"] = (plan.xt8.view(u8) != first.xt8.view(u8)).sum()
                counts["mean vs x8 leg"] = (plan.mean.view(i32) != first.mean.view(i32)).sum()
                counts["rstd vs x8 leg"] = (plan.rstd.view(i32) != first.rstd.view(i32)).sum()
            values = dict(zip(counts, torch.stack([v.to(torch.int64) for v in counts.values()]).tolist()))
            bad = {k: v for k, v in values.items() if v}
            if bad:
                return f"producer self-check on {name} (x8={'yes' if want_x8 else 'no'}, M={rows}): {bad}", None
            if first is None:
                first = plan
        if name == "zeros":
            keep = first
    return None, keep


def _prebuild_device() -> torch.device:
    """The device prebuild() builds and self-checks on: the current CUDA device (one place, so the
    CPU tests can run the prebuild logic on CPU tensors)."""
    return torch.device("cuda", torch.cuda.current_device())


def _prebuild(warm_rows: int) -> Optional[str]:
    reason = _gate_reason()
    if reason is not None:
        return reason
    for role in ("producer", "gemm"):
        if _module(role) is None:
            return _mod_error[role]
    if not torch.cuda.is_available():
        return "skipped: no GPU"
    device = _prebuild_device()
    arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        return f"skipped: arch {arch or 'unknown'} (MX6 kernels are gfx950-only)"
    prod, gemm = _mods["producer"], _mods["gemm"]
    for role, mod in (("producer", prod), ("gemm", gemm)):
        hook = getattr(mod, "prebuild", None)
        if hook is not None:
            try:
                ok = hook()
            except Exception as error:
                return f"{role} prebuild failed ({type(error).__name__}: {error})"
            if ok is False:
                return f"{role} prebuild() returned False on {arch} (not built / self-check skipped)"
    reason = _producer_reason(device)
    if reason is not None:
        return reason
    for n in REQUIRED_N:
        reason = _kernel_reason(n)
        if reason is not None:
            return reason
    u_reason = _kernel_reason(SITE_N["U"])
    if u_reason is not None and u_expected():
        # Option C, and U failed to build or to self-check in the GEMM module's prebuild (latched
        # there): fatal under STRICT (prebuild() raises PrebuildError); otherwise option B, loudly.
        message = _u_not_ready("prebuild", u_reason)
        if STRICT:
            return message + f" ({KNOB}_STRICT=1: fatal)"
        _u_fallback(message)
    reason, zero = _selfcheck_producer(device, warm_rows)
    if reason is not None:
        return reason
    assert zero is not None and zero.x6 is not None and zero.xs is not None
    u8 = torch.uint8
    one = torch.ones((1, 1), dtype=torch.float32, device=device)
    for n in REQUIRED_N + (() if u_reason is not None else (SITE_N["U"],)):
        # Synthetic weight operands in pr7's layouts (zero codes, scale byte 127 = 2^0); no
        # weight-shadow cache entry is created for them.
        w6 = torch.zeros((n // 256, 2, KSTEPS, X6_ITEM_BYTES), dtype=u8, device=device)
        ws = torch.full((n // 32, 4, 4), 127, dtype=u8, device=device)
        bias = torch.zeros((n,), dtype=torch.bfloat16, device=device)
        dtype = torch.bfloat16 if n == SITE_N["U"] else _FP8_E4M3
        width = n * (2 if dtype == torch.bfloat16 else 1)
        # 0xFF prefill: e4m3 / bf16 NaN, so an unwritten byte cannot pass as +0.
        out = torch.full((warm_rows, width), 0xFF, dtype=u8, device=device).view(dtype)
        args, kwargs = _gemm_call(zero.x6, zero.xs, w6, ws, bias, one, one, warm_rows, n, out)
        result = gemm.k512_mx6(*args, **kwargs)
        torch.cuda.synchronize(device)
        if result is not None and result is not out:
            return f"GEMM N={n} self-check: k512_mx6 returned a tensor other than `out`"
        nonzero = int(torch.count_nonzero(out.view(u8)).item())
        if nonzero:
            return f"GEMM N={n} self-check on zeros wrote {nonzero} nonzero bytes"
    zero.release()
    _print(
        f"prebuilt: producer={PRODUCER_MODULE} [{_producer_desc()}] gemm={GEMM_MODULE} "
        f"N={','.join(str(n) for n in REQUIRED_N)}"
        + (f",{SITE_N['U']}" if u_reason is None else f" (U N={SITE_N['U']}: shipped FP8; {u_reason})")
        + f"; self-check M={warm_rows} (zeros + ramp, x8 and no-x8 kernels, every byte vs the torch "
        "reference) OK"
    )
    return None


# ----------------------------------------------------------------------------- tests
def reset_for_testing(
    *, enabled: Optional[bool] = None, gemm_module: Optional[str] = None,
    producer_module: Optional[str] = None, strict: Optional[bool] = None,
    shadow_audit: Optional[int] = None,
) -> None:
    """Test-only: clear every latch / cache / once-set and optionally override the import-time
    settings (None keeps the current value)."""
    global ENABLED, GEMM_MODULE, PRODUCER_MODULE, STRICT, SHADOW_AUDIT, _prebuild_error
    with _lock:
        if enabled is not None:
            ENABLED = enabled
        if gemm_module is not None:
            GEMM_MODULE = gemm_module
        if producer_module is not None:
            PRODUCER_MODULE = producer_module
        if strict is not None:
            STRICT = strict
        if shadow_audit is not None:
            SHADOW_AUDIT = shadow_audit
        _prebuild_error = None
        _mods.clear()
        _mod_error.clear()
        _n_ok.clear()
        _n_error.clear()
        _producer_error.clear()
        _producer_ok.clear()
        _producer_checked.clear()
        _site_armed.clear()
        _site_latch.clear()
        _armed_devices.clear()
        _once.clear()
        _counters.clear()
        _shadows.clear()
        _shadow_builds.update(builds=0, hits=0)


if ENABLED:
    _print(
        f"requested: {KNOB}=1 (DECLARED PRECISION CHANGE; forward U / V/Q/K / targets V/K on MXFP6 "
        f"E2M3 + E8M0/32) producer={PRODUCER_MODULE} gemm={GEMM_MODULE} strict={int(STRICT)} "
        f"shadow_audit={SHADOW_AUDIT}"
    )
    atexit.register(_at_exit)

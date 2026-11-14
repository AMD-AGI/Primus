"""HSTU jagged attention with FlyDSL forward and backward kernels, each optional.

THE TWO HALVES ARE SELECTED INDEPENDENTLY, because HSTU has no softmax: the
forward hands the backward no lse and no rowsum(dO*O), so nothing couples them.
$HSTU_ATTN_BWD defaults to flydsl and is worth -10.77% step time. $HSTU_ATTN_FWD
defaults to **flydsl** in gin (FlyDSL e4m3 forward + fused-silu when
DLRM_HSTU_FP8_FWD is on). $HSTU_ATTN_FWD=triton is the opt-out. The bf16
FlyDSL forward is the same knob with FP8 off:
bit-identical, -19.6% on the traced attention forward, e2e UNRESOLVED --
+0.41% over 3 pairs with a 95% CI of [-2.07%, +2.88%] against a +1.28%
ceiling. See hstu_attention_forward_e2e.md (git history at 33610d2).

That the forward's e2e contribution is smaller and harder to measure than the
backward's is also why its `$HSTU_ATTN_FWD_DUMP_LENS` hook below exists.

`hstu_mha` mirrors the signature of
`generative_recommenders.ops.triton.triton_hstu_attention.triton_hstu_mha`, so
it is a drop-in at the call site. Selection lives in `ops/hstu_attention.py`
behind $HSTU_ATTN_BWD, which **defaults to flydsl** (gin:
`apply_env_bootstrap.HSTU_ATTN_BWD`); set it to "triton" to opt out.

TWO KINDS OF "CAN'T RUN", DELIBERATELY HANDLED DIFFERENTLY.

  * The HOST cannot run it -- not a gfx950 GPU, or flydsl is not installed.
    That is a property of the whole process, it makes every backward ~2x
    slower, and now that flydsl is the default it would otherwise be an
    invisible regression. So it RAISES `FlyDSLUnavailableError` on first use
    rather than degrading. Opt out explicitly with $HSTU_ATTN_BWD=triton.
  * The SHAPE cannot be served -- see SPECIALIZATIONS below. That is per call
    and legitimately varies across configs and tests, so it falls back to the
    Triton backward. The first fallback of each distinct reason is logged, so
    it is quiet but not silent.

SPECIALIZATIONS.  The FlyDSL kernel is not general. It runs only when ALL of the
following hold, and otherwise this falls back to the Triton backward -- a shape
it cannot serve is slower, never wrong:

  * dim == 128           the mfma fragment layout is written around it
  * BM == 64             likewise; asserted inside the kernel builder
  * bf16 q, k, v, dout   the only dtype with an mfma path here
  * causal, max_attn_len == 0, contextual_seq_len == 0
  * num_targets is None, or every entry is <= 1

  The last one deserves its reason. HSTU's target-aware masking clamps positions
  to max_ids = seq_len - n_targets and then applies the ordinary >= mask to the
  clamped values. At n_targets = 1 the only affected position is seq_len-1, which
  clamps to itself, so the clamp is an identity and plain causal is exactly right
  -- measured BIT-IDENTICAL against Triton at num_targets=1, and genuinely
  different at 2 and 8. yambda-5b sets hstu_config.max_num_candidates = 1, so
  this covers production, but a config that raises it must fall back rather than
  quietly compute the wrong function.

  `heads` is a compile-time constant (it keys the kernel cache) -- not a
  correctness limit, just a fresh JIT per distinct head count. max_seq_len is NOT
  compile-time; it is a runtime scalar, which matters because production derives
  it per batch as max_uih_len + num_candidates.

NON-DETERMINISM.  dq is accumulated with global atomics, so the order of the
float adds varies run to run and dq is NOT bitwise reproducible. dk and dv are
deterministic and come back bit-identical (dv) or within a few ulps (dk) of
Triton. Run-to-run dq agreement is ~144 dB, i.e. reassociation only. A caller
needing bitwise reproducibility across runs must use the Triton path.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch

from generative_recommenders.ops.flydsl import (
    DEFAULT_BWD_BACKEND,
    DEFAULT_FWD_BACKEND,
    FLYDSL_ARCH_PIN,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_hstu_attention_bwd,
    triton_hstu_attention_fwd,
)
# R15 A11 hook (contract version: hstu_do8_registry.CONTRACT): torch +
# stdlib only.
from generative_recommenders.ops.flydsl import hstu_do8_registry

logger: logging.Logger = logging.getLogger(__name__)

SUPPORTED_DIM = 128
FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
# The kernel emits gfx950 mfma/ds_read_b64_tr_b16; it is not portable to another
# arch, and there is no meaningful "close enough" here.
SUPPORTED_ARCH = "gfx950"

# Import state for the flydsl kernel, resolved on first backward. Kept lazy so a
# trainer built without flydsl can still import this module (to report why it
# cannot be used, or to run with $HSTU_ATTN_BWD=triton).
_kernel = None
_kernel_error: Optional[BaseException] = None
_backward_calls: int = 0
_logged_fallbacks: set = set()
_logged_layouts: set = set()
_host_ok: bool = False

# The forward kernel is a separate module with its own import latch, so a tree
# that has the backward but not the forward degrades to the Triton forward rather
# than failing the whole attention path.
_fwd_kernel = None
_fwd_kernel_error: Optional[BaseException] = None
_forward_calls: int = 0
_logged_fwd_fallbacks: set = set()
_logged_fwd_layouts: set = set()
_fwd_host_ok: bool = False
# The e4m3 forward keeps its own counters: it is a different kernel behind a
# different knob, and mixing the two would make $HSTU_ATTN_FWD_CHECK's period
# depend on which arm is running.
_fp8_forward_calls: int = 0
_logged_fp8_fwd_layouts: set = set()
_e1_bwd_census: dict = {}
_e1_bwd_calls: int = 0


def _maybe_dump_lengths(seq_offsets: torch.Tensor) -> None:
    """Save one batch's sequence lengths to $HSTU_ATTN_FWD_DUMP_LENS, once.

    The offline benchmark samples lengths from a saved pool, and which pool it is
    decides the measured speedup: the same kernel scored 2.02x against a pool with
    a 3x length spread and 1.19x against production's, because a bound-sized grid
    penalises the two kernels differently when the bound exceeds the work. So the
    pool has to come from here rather than from a plausible-looking file.

    Off unless the variable is set, and writes exactly once per process.
    """
    path = os.environ.get("HSTU_ATTN_FWD_DUMP_LENS", "")
    if not path:
        return
    try:
        lens = (seq_offsets[1:] - seq_offsets[:-1]).to("cpu").numpy()
        # Rank-tagged: every rank sees a different slice of the global batch, and
        # one rank's 1024 sequences is a thin sample of the length distribution.
        out = f"{path}.rank{torch.distributed.get_rank()}.npy" \
            if torch.distributed.is_available() and torch.distributed.is_initialized() \
            else f"{path}.npy"
        np.save(out, lens)
        logger.warning(
            "[hstu-attn] dumped %d sequence lengths (mean %.0f max %d) to %s",
            lens.size, float(lens.mean()), int(lens.max()), out,
        )
    except Exception as exc:  # a benchmarking aid must never break a train step
        logger.warning("[hstu-attn] length dump to %s failed: %r", path, exc)

# Memo for the num_targets support check below: (tensor, version, max).
# A strong reference to the tensor is deliberate -- it is one int per row -- and
# is what makes the `is` test sound, since the storage cannot be recycled under
# a live reference.
_targets_max_memo: Tuple[Optional[torch.Tensor], int, int] = (None, -1, 0)


def _max_num_targets(num_targets: torch.Tensor) -> int:
    """`num_targets.max()`, computed once per batch rather than once per layer.

    Every HSTU layer's backward is handed the same num_targets tensor and asks
    the same question of it, but `.item()` is a device-to-host sync: it drains
    the launch queue, and in the backward the queue is at its deepest. Profiled
    at 31.7 ms per call against 0.6 ms for the same call in the forward, so the
    cost is the wait for queued work, not the 4-byte copy. Memoizing collapses
    the per-step cost from one sync per layer to one.

    Keyed on tensor identity plus `_version` so that a fresh batch or any
    in-place write misses and re-reads.
    """
    global _targets_max_memo
    cached, version, value = _targets_max_memo
    if cached is num_targets and version == num_targets._version:
        return value
    value = int(num_targets.max().item())
    _targets_max_memo = (num_targets, num_targets._version, value)
    return value


def seed_max_num_targets(num_targets: torch.Tensor, value: int) -> None:
    """Record ``num_targets.max() == value`` read by the caller's own sync.

    DlrmHSTU.preprocess reads this max (as max_num_candidates) together with
    its other per-batch length stats in a single device-to-host copy; seeding
    the memo here lets the first _max_num_targets call of the batch skip its
    own `.item()`. Same identity/_version key, so any other tensor, or an
    in-place write to this one, still misses and re-reads.
    """
    global _targets_max_memo
    _targets_max_memo = (num_targets, num_targets._version, int(value))


class FlyDSLUnavailableError(RuntimeError):
    """The FlyDSL backward was selected but this host cannot run it.

    Raised rather than falling back because flydsl is the default: a silent
    downgrade to Triton is a ~2x slower backward that no log line would show.
    """


def _device_arch() -> str:
    """gfx arch of the current device, or "" if it is not a ROCm GPU."""
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
    except BaseException:  # noqa: BLE001 - no device, or a driver that cannot say
        return ""
    # ROCm exposes gcnArchName ("gfx950:sramecc+:xnack-"); CUDA has no such field.
    return str(getattr(props, "gcnArchName", "") or "")


def unsupported_hardware_reason() -> Optional[str]:
    """Why this host cannot run the kernel, or None if it can."""
    if not torch.cuda.is_available():
        return "no GPU is visible to torch"
    arch = _device_arch()
    if not arch:
        return (
            "this is not a ROCm device (no gcnArchName); the kernel is "
            f"{SUPPORTED_ARCH} assembly"
        )
    if not arch.startswith(SUPPORTED_ARCH):
        return f"device arch is {arch!r}, the kernel is built for {SUPPORTED_ARCH}"
    return None


def _flydsl_arch_check(kernel_module) -> Tuple[str, Optional[str]]:
    """(arch FlyDSL builds for, None), or (arch, why the kernel cannot run).

    The device can be gfx950 while FlyDSL compiles for something else (ship13
    K10): its get_rocm_arch() falls back to "gfx942" when $FLYDSL_GPU_ARCH and
    $HSA_OVERRIDE_GFX_VERSION are unset and `rocm_agent_enumerator -name`
    fails or exceeds 5 s, and the first gfx950-only MFMA intrinsic then aborts
    the process (rc 139, no traceback). Read through the kernel module's own
    module-scope import, i.e. the function its builders call. FlyDSL
    lru_caches it per process, so this call fixes the arch the first build
    will use, and repeats are free.
    """
    try:
        get_arch = getattr(kernel_module, "get_rocm_arch", None)
        if get_arch is None:
            from flydsl.runtime.device import get_rocm_arch as get_arch
        arch = str(get_arch())
    except Exception as e:  # noqa: BLE001 - report it, do not mask it
        return "", f"FlyDSL could not resolve its target arch ({type(e).__name__}: {e})"
    if arch.startswith(SUPPORTED_ARCH):
        return arch, None
    return arch, (
        f"FlyDSL resolved its compile target to {arch!r}, not {SUPPORTED_ARCH} "
        f"(FLYDSL_GPU_ARCH={os.environ.get('FLYDSL_GPU_ARCH', '<unset>')}, "
        f"arch pin at import: {FLYDSL_ARCH_PIN[1]}); every FlyDSL kernel would "
        f"be built for it and abort the process at its first "
        f"{SUPPORTED_ARCH}-only MFMA intrinsic (LLVM ERROR: Cannot select, rc "
        f"139). FlyDSL fixes the arch at its first lookup in the process: "
        f"FLYDSL_GPU_ARCH or HSA_OVERRIDE_GFX_VERSION, else "
        f"'rocm_agent_enumerator -name' (5 s timeout, errors swallowed), else "
        f"its 'gfx942' fallback. Fix: export FLYDSL_GPU_ARCH={SUPPORTED_ARCH} "
        f"before the process starts (scripts/launch_slurm.sh does; ship13 K10)"
    )


def _load_kernel():
    """Return the kernel module, or None if flydsl could not be imported.

    The failure is latched so a broken install is diagnosed once, not once per
    backward. Callers that have *selected* flydsl go through `require_flydsl`,
    which turns None into a raise.
    """
    global _kernel, _kernel_error
    if _kernel is not None or _kernel_error is not None:
        return _kernel
    try:
        from generative_recommenders.ops.flydsl import hstu_bwd_dkdv

        _kernel = hstu_bwd_dkdv
    except BaseException as e:  # noqa: BLE001 - report it, do not mask it
        _kernel_error = e
    return _kernel


def require_flydsl() -> None:
    """Raise `FlyDSLUnavailableError` unless the kernel can run on this host.

    Latched on success only: this sits on the per-backward path and neither the
    arch nor the install changes within a process, but an unsupported host must
    keep raising rather than pass on the second call.
    """
    global _host_ok
    if _host_ok:
        return
    reason = unsupported_hardware_reason()
    if reason is None and _load_kernel() is None:
        reason = f"flydsl import failed ({type(_kernel_error).__name__}: {_kernel_error})"
    fly_arch = ""
    if reason is None:
        fly_arch, reason = _flydsl_arch_check(_kernel)
    if reason is None:
        _host_ok = True
        # The one place both call sites pass through, and the only selection
        # record the FUSED path produces -- it does not go through
        # ops/hstu_attention.py's dispatch, so without this a production run
        # says nothing about which backward it ran.
        # The forward half is reported from the env rather than hardcoded: it is
        # separately selectable and now defaults to flydsl too, so a fixed string
        # here would misreport the run it is supposed to be the record of.
        logger.info(
            "[hstu-attn] backward kernel: flydsl (forward: %s) on %s "
            "flydsl_arch=%s pin=%s",
            "flydsl" if flydsl_fwd_selected() else "triton",
            _device_arch(),
            fly_arch,
            FLYDSL_ARCH_PIN[1],
        )
        return
    raise FlyDSLUnavailableError(
        f"the FlyDSL HSTU attention backward is selected but cannot run here: "
        f"{reason}. It is the default; to run the Triton backward on this host "
        f"set HSTU_ATTN_BWD=triton in the environment, or bind "
        f"apply_env_bootstrap.HSTU_ATTN_BWD = 'triton' in the gin config."
    )


def unsupported_shape_reason(
    q, k, v, dout, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0,
) -> Optional[str]:
    """Why the kernel cannot serve this call, or None if it can."""
    if max_attn_len:
        return f"max_attn_len={max_attn_len} (kernel is plain causal)"
    if contextual_seq_len:
        return f"contextual_seq_len={contextual_seq_len} (kernel assumes 0)"
    if num_softmax_heads:
        return f"num_softmax_heads={num_softmax_heads} (kernel assumes 0)"
    if q.dim() != 3 or q.shape[-1] != SUPPORTED_DIM:
        return f"qk dim {tuple(q.shape)} (kernel is written around dim {SUPPORTED_DIM})"
    if v.shape[-1] != SUPPORTED_DIM:
        return f"v dim {v.shape[-1]} (kernel is written around dim {SUPPORTED_DIM})"
    qkv_dtypes = {q.dtype, k.dtype, v.dtype}
    supported_qkv_dtype = qkv_dtypes == {torch.bfloat16} or (
        FP8_E4M3 is not None and qkv_dtypes == {FP8_E4M3}
    ) or (
        FP8_E4M3 is not None
        and q.dtype is torch.bfloat16
        and k.dtype is FP8_E4M3
        and v.dtype is FP8_E4M3
    )
    if not supported_qkv_dtype or dout.dtype is not torch.bfloat16:
        return (
            "dtype "
            f"{tuple(str(t.dtype).replace('torch.', '') for t in (q, k, v, dout))}"
            " (kernel requires BF16 QKV, E4M3 QKV, or BF16 Q with E4M3 K/V)"
        )
    # Layout is deliberately NOT a support condition. On the fused
    # preprocess+attention path q/k/v arrive as column slices of one uvqk buffer
    # and are not contiguous; the kernel's buffer descriptors assume a packed
    # [N, H, D] layout, so those are normalized with .contiguous() below rather
    # than being grounds to fall back.
    if num_targets is not None:
        # Checked here rather than trusted from the caller: that is the
        # difference between falling back and silently computing a different
        # function, which is the exact failure this kernel already shipped once.
        # The sync it costs is amortized to once per batch -- see _max_num_targets.
        n = _max_num_targets(num_targets)
        if n > 1:
            return f"num_targets max={n} (target-aware masking is identity only at <= 1)"
    return None


def flydsl_supports(
    q, k, v, dout, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0,
) -> bool:
    return unsupported_shape_reason(
        q, k, v, dout, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    ) is None


def _common_row_stride(tensors, heads, dim):
    """The shared token-row stride of `tensors`, or None if they cannot share one.

    Every tensor must be [tokens, heads, dim] whose two inner dimensions are
    packed -- stride (rs, dim, 1) -- differing only in `rs`, and every one must
    agree on it, because the kernel carries a single row stride for the whole
    uvqk family. Production satisfies this: q/k/v and dq/dk/dv are all column
    slices of one [tokens, 2048] buffer, so all six have stride (2048, 128, 1).

    rs must also be a multiple of 8 to keep the 8-wide bf16 loads 16 B-aligned
    at every token; the kernel asserts this too, but a caller-side check turns a
    bad layout into a fallback instead of a compile-time crash.
    """
    strides = set()
    for t in tensors:
        if t.dim() != 3 or t.shape[1] != heads or t.shape[2] != dim:
            return None
        rs, sh, sd = t.stride()
        if sd != 1 or sh != dim or rs < heads * dim or rs % 8:
            return None
        strides.add(rs)
    return strides.pop() if len(strides) == 1 else None


def bwd_qk_mfma_mode() -> str:
    """Attention-backward qk GEMM math: 'bf16', 'fp8', or 'fp8_strict'.

    ``DLRM_HSTU_FP8_BWD_QK_MFMA`` default is ``fp8``: k128 fp8 MFMA when this
    call's q and k are already e4m3; calibration stays bf16 math.
    ``bf16`` is the explicit opt-out. ``fp8_strict`` raises instead of skipping.

    Deprecated ``DLRM_HSTU_FP8_BWD_K128=1`` still selects ``fp8_strict``.
    ``K128=0`` is not an opt-out once the named knob defaults to fp8 -- set
    ``QK_MFMA=bf16``.
    """
    aliases = {
        "bf16": "bf16",
        "0": "bf16",
        "off": "bf16",
        "fp8": "fp8",
        "k128": "fp8",
        "auto": "fp8",
        "fp8_strict": "fp8_strict",
        "strict": "fp8_strict",
        "1": "fp8_strict",
    }

    def _parse(name: str, raw: Optional[str]) -> Optional[str]:
        if raw is None or raw.strip() == "":
            return None
        key = raw.strip().lower()
        if key not in aliases:
            raise RuntimeError(
                f"{name}={raw!r} is not one of bf16, fp8, fp8_strict "
                f"(deprecated K128: 0, auto, 1)."
            )
        return aliases[key]

    qk = _parse(
        "DLRM_HSTU_FP8_BWD_QK_MFMA",
        os.environ.get("DLRM_HSTU_FP8_BWD_QK_MFMA"),
    )
    k128 = _parse(
        "DLRM_HSTU_FP8_BWD_K128",
        os.environ.get("DLRM_HSTU_FP8_BWD_K128"),
    )
    if qk == "bf16":
        return "bf16"
    if qk == "fp8_strict" or k128 == "fp8_strict":
        return "fp8_strict"
    return "fp8"


# Removed 2026-10-03: block dO scales (DOBLK) and the saturating cast (SAT)
# are the only C8 path now. An off value must not be silently ignored.
_REMOVED_C8_KNOBS = ("DLRM_HSTU_FP8_BWD_FULL_DOBLK", "DLRM_HSTU_FP8_BWD_FULL_SAT")


def check_removed_c8_knobs() -> None:
    """Raise if a removed C8 sub-knob is set to an off value (an on value is
    the only behaviour and is accepted). Same rule as
    dlrm_v4/train/_env_bootstrap._check_removed_c8_knobs."""
    for name in _REMOVED_C8_KNOBS:
        raw = os.environ.get(name)
        if raw is not None and raw.strip() != "" and raw.strip().lower() \
                not in ("1", "true", "on", "yes"):
            raise RuntimeError(
                f"{name}={raw!r}: this knob was removed; DOBLK+SAT is the only "
                "C8 path. Use DLRM_HSTU_FP8_BWD_FULL=0 to turn C8 off.")


def bwd_full_fp8_enabled() -> bool:
    """$DLRM_HSTU_FP8_BWD_FULL (default OFF): full-fp8 attention backward."""
    if _hybrid_force_off[0]:
        return False
    return os.environ.get("DLRM_HSTU_FP8_BWD_FULL", "0").strip().lower() in (
        "1", "true", "on", "yes")


# $DLRM_HSTU_FP8_BWD_FULL_HYBRID = dq | dkdv (default unset; DIAGNOSTIC ONLY:
# it runs BOTH backward kernels per call, so it is slower than either). With
# the full-fp8 backward armed, the bf16-backward kernel's dq (resp. dk and dv)
# replaces the full-fp8 kernel's. It isolates which gradient's fp8 error owns
# a convergence deficit at L1 without writing a new kernel.
_hybrid_force_off = [False]
_hybrid_inner = [False]


_full_fp8_stats_calls = [0]


def _full_fp8_stats_left() -> bool:
    """$DLRM_HSTU_FP8_BWD_FULL_STATS=<n>: DEBUG ONLY, host-syncs. Log operand
    ranges for the first n armed calls."""
    n = int(os.environ.get("DLRM_HSTU_FP8_BWD_FULL_STATS", "0") or 0)
    return _full_fp8_stats_calls[0] < n


@torch.no_grad()
def _full_fp8_stats(q, k, v, dout8, do_scale, seq_offsets, N, alpha,
                    qkv_scale, config, dq, dk, dv, dout_orig=None):
    """fp32 reference ranges of the full-fp8 kernel's e4m3 tiles on the longest
    of the first 16 sequences (same math as the kernel: silu/N, ds with the
    NORMALIZED dO), plus finiteness of the kernel outputs. Host syncs."""
    _full_fp8_stats_calls[0] += 1
    # Longest WHOLE sequence <= 2048 among the first 64 (a truncated one would
    # give a wrong dk/dv reference: keys see every later query row).
    offs = seq_offsets[:65].tolist()
    lens = [b - a if b - a <= 2048 else 0 for a, b in zip(offs[:-1], offs[1:])]
    i = int(np.argmax(lens))
    st, L = offs[i], lens[i]
    qd = q[st:st + L].float() * qkv_scale
    kd = k[st:st + L].float() * qkv_scale
    vd = v[st:st + L].float() * qkv_scale
    do_n = dout8[st:st + L].float()
    x = torch.einsum("mhd,nhd->hmn", qd, kd) * alpha
    mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
    sg = torch.sigmoid(x)
    siluN = (x * sg).masked_fill(~mask, 0)
    gc = (sg * (1 + x * (1 - sg))).masked_fill(~mask, 0)
    dsN = torch.einsum("mhd,nhd->hmn", do_n, vd) * gc
    xa = float(x.masked_fill(~mask, 0).abs().max())
    sa = float(siluN.abs().max())
    da = float(dsN.abs().max())
    ssc, dsc = config["mu2_ssc"], config["mu2_dsc"]
    s_t = sa / float(N) * ssc
    d_t = da / float(N) * dsc
    # fraction of tile elements that would overflow / underflow e4m3
    s_tile = siluN / float(N) * ssc
    d_tile = dsN / float(N) * dsc
    nz = mask.sum().item() * x.shape[0]
    def frac(t, lo):
        return (float((t.abs() > 448).sum()) / nz,
                float(((t.abs() < lo) & (t != 0)).sum()) / nz)
    fin = [bool(torch.isfinite(t).all()) for t in (dq, dk, dv)]

    def q8snr(t):
        t = t[mask.expand_as(t)]
        g = t.clamp(-448, 448).to(torch.float8_e4m3fn).float()
        return float(10 * torch.log10(t.pow(2).sum()
                                      / (t - g).pow(2).sum().clamp_min(1e-30)))
    c = int(np.ceil(np.log2(max(int(N), 2))))
    sweep = {}
    for nm, t in (("silu", siluN), ("ds", dsN)):
        r = {kk: q8snr(t * 2.0 ** (kk + c) / float(N)) for kk in range(-6, 12)}
        best = max(r, key=r.get)
        sweep[nm] = "best K=%d %.1fdB | cur %.1fdB | %s" % (
            best, r[best], q8snr(t * (ssc if nm == "silu" else dsc) / float(N)),
            " ".join("%d:%.1f" % (kk, r[kk]) for kk in range(-6, 12, 2)))
    gsnr = "n/a"
    if dout_orig is not None:
        with torch.enable_grad():
            qg, kg, vg = (t.detach().clone().requires_grad_() for t in (qd, kd, vd))
            xg = torch.einsum("mhd,nhd->hmn", qg, kg) * alpha
            pg = torch.nn.functional.silu(xg) / float(N) * mask
            og = torch.einsum("hmn,nhd->mhd", pg, vg)
            rq, rk, rv = torch.autograd.grad(
                (og * dout_orig[st:st + L].float()).sum(), (qg, kg, vg))
        def gs(r_, g_):
            g_ = g_[st:st + L].float()
            return float(10 * torch.log10(r_.pow(2).sum()
                                          / (r_ - g_).pow(2).sum().clamp_min(1e-30)))
        gsnr = "dv %.1f dq %.1f dk %.1f" % (gs(rv, dv), gs(rq, dq), gs(rk, dk))
    print("[hstu-attn][full-fp8 stats] call=%d SNR(kernel vs fp32 ref, L=%d) %s"
          " || silu-op %s || ds-op %s" % (_full_fp8_stats_calls[0], L, gsnr,
                                          sweep["silu"], sweep["ds"]),
          flush=True)
    print(
        "[hstu-attn][full-fp8 stats] call=%d N=%d L=%d qkv_scale=%.4g "
        "s_do=%.4g |x|max=%.4g |silu*N|max=%.4g |ds*N|max=%.4g "
        "silu_tile_max=%.4g (ssc=2^%d) ds_tile_max=%.4g (dsc=2^%d) "
        "silu ovf/sub=%.3g/%.3g ds ovf/sub=%.3g/%.3g "
        "|q|max=%.4g |k|max=%.4g |v|max=%.4g finite(dq,dk,dv)=%s" % (
        _full_fp8_stats_calls[0], int(N), L, qkv_scale,
        # s_do: dO scales are per (seq, head, MX group) E8 bytes; no scalar.
        float("nan"), xa, sa, da, s_t, int(np.log2(ssc)), d_t,
        int(np.log2(dsc)), *frac(s_tile, 2.0 ** -9), *frac(d_tile, 2.0 ** -9),
        float(qd.abs().max()), float(kd.abs().max()), float(vd.abs().max()),
        fin,
    ), flush=True)


_bwd_dump_calls = [0]


def _bwd_dump_dir() -> str:
    """$DLRM_HSTU_FP8_BWD_DUMP=<dir> (default OFF; DEBUG ONLY, host-syncs on
    the selected calls). Rank 0 saves real backward operands for offline
    numerics (C8 L0 harness) on the call indices listed in
    $DLRM_HSTU_FP8_BWD_DUMP_CALLS (comma list, 0-based over every backward
    call of this process). Inert when unset."""
    d = os.environ.get("DLRM_HSTU_FP8_BWD_DUMP", "")
    if not d:
        return ""
    rank = (torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else int(os.environ.get("RANK", "0") or 0))
    return d if rank == 0 else ""


@torch.no_grad()
def _bwd_dump(q, k, v, dout, do_scale, seq_offsets, N, alpha, qkv_scale,
              dq, dk, dv, config) -> None:
    i = _bwd_dump_calls[0]
    _bwd_dump_calls[0] += 1
    want = {int(x) for x in os.environ.get(
        "DLRM_HSTU_FP8_BWD_DUMP_CALLS", "").split(",") if x.strip()}
    if i not in want:
        return
    d = _bwd_dump_dir()
    os.makedirs(d, exist_ok=True)
    offs = seq_offsets.to(torch.int64).cpu()
    lens = offs[1:] - offs[:-1]
    B = lens.numel()
    # Whole sequences only (attention is per sequence, so slicing is exact):
    # the 4 longest plus 12 spread evenly over the batch.
    longest = torch.argsort(lens, descending=True)[:4].tolist()
    spread = torch.linspace(0, B - 1, 12).round().long().tolist()
    sel = sorted(set(longest) | set(spread))
    rows = torch.cat([torch.arange(int(offs[s]), int(offs[s + 1]))
                      for s in sel]).to(q.device)

    def g(t):
        return t.index_select(0, rows).contiguous().cpu()

    dof = dout.float()
    payload = {
        "call": i, "N": int(N), "alpha": float(alpha),
        "qkv_scale": float(qkv_scale), "sel": sel, "seq_offsets": offs,
        "q": g(q), "k": g(k), "v": g(v), "dout": g(dout),
        "dq": g(dq), "dk": g(dk), "dv": g(dv),
        "do_scale": (None if do_scale is None or do_scale.dtype == torch.uint8
                     else do_scale.cpu()),
        # do_blk (C8 option B): the per-token group bytes E8 = 127 - e_g.
        "do_e8": (g(do_scale) if do_scale is not None
                  and do_scale.dtype == torch.uint8 else None),
        # Whole-batch dO magnitude census, per token over heads x dim.
        "do_row_amax": dof.abs().amax(dim=(1, 2)).cpu(),
        "do_row_rms": dof.pow(2).mean(dim=(1, 2)).sqrt().cpu(),
        "config": {kk: (vv if isinstance(vv, (int, float, str, bool))
                        else str(vv)) for kk, vv in config.items()},
    }
    p = os.path.join(d, "bwd_call%05d.pt" % i)
    torch.save(payload, p)
    logger.warning("[hstu-attn] BWD DUMP call=%d -> %s (%d seqs, %d rows)",
                   i, p, len(sel), rows.numel())


_DQ_SPLIT_ORDER_LOGGED = set()


def _dq_split_order(N: int) -> str:
    """Grid order of the dq-split kernel ($DLRM_HSTU_FP8_BWD_DQ_SPLIT_ORDER).

    "xcd" (default) groups each (b, h)'s row blocks on one XCD so its K tiles
    stay in that XCD's L2; "bh" is the iter-4 order.  Bitwise-identical dq.
    """
    order = os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_ORDER", "xcd").strip().lower()
    if order not in ("bh", "xcd"):
        raise ValueError(f"DLRM_HSTU_FP8_BWD_DQ_SPLIT_ORDER={order!r}: use bh|xcd")
    if order not in _DQ_SPLIT_ORDER_LOGGED:
        _DQ_SPLIT_ORDER_LOGGED.add(order)
        logger.warning("[hstu-attn] bwd dq SPLIT grid order: %s", order)
    return order


def _dq_split_ld():
    """R15 iter24 ws track ($DLRM_HSTU_FP8_DQ_SPLIT_LD, default unset = {}):
    cache policy of the dq-split kernel's ws loads (read exactly once).
    Comma list of a Triton cache_modifier (.ca|.cg|.cv|.cs) and/or an
    eviction_policy (evict_first|evict_last).  Same bytes loaded: bitwise."""
    v = os.environ.get("DLRM_HSTU_FP8_DQ_SPLIT_LD", "").strip().lower()
    kw = {}
    for t in (x.strip() for x in v.split(",")):
        if t in (".ca", ".cg", ".cv", ".cs"):
            kw["ws_cm"] = t
        elif t in ("evict_first", "evict_last"):
            kw["ws_ev"] = t
        elif t:
            raise ValueError(f"DLRM_HSTU_FP8_DQ_SPLIT_LD={v!r}")
    return kw


_DQ_SPLIT_SCHED_LOGGED = set()


def _dq_split_sched() -> str:
    """XCD-balanced dq-split schedule ($DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED,
    default OFF = ""): "il" interleaves (b, h) units over the 8 XCDs instead of
    giving each XCD a contiguous batch slice (such slices can differ
    in work); "lpt" also walks them longest sequence first.  Bitwise dq."""
    v = os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED", "").strip().lower()
    if v in ("", "0", "off", "none"):
        return ""
    if v not in ("il", "lpt", "ilb", "ilr", "bc"):
        raise ValueError(
            f"DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED={v!r}: use il|lpt|ilb|ilr|bc")
    if v not in _DQ_SPLIT_SCHED_LOGGED:
        _DQ_SPLIT_SCHED_LOGGED.add(v)
        logger.warning("[hstu-attn] bwd dq SPLIT XCD-balanced schedule ARMED: %s", v)
    return v


_DQ_SPLIT_OPT_LOGGED = set()


def _dq_split_opt() -> dict:
    """dq-split load knobs ($DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT, R15 iter24,
    default OFF = no kwargs).  Comma tokens: r1 (mask ws rows past the
    sequence end), sp (scalar per-tile base + int32 offsets: the ws stream
    becomes buffer loads), wcg/wcv/wca (ws load cache_modifier), wef/wel (ws
    eviction_policy evict_first/evict_last), kcg/kca (K cache_modifier),
    kel/kef (K eviction_policy), e<N> (waves_per_eu), s<N> (num_stages).
    Same tiles and dot order: bitwise dq.  Needs a DQ_SPLIT_SCHED."""
    v = os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT", "").strip().lower()
    if v in ("", "0", "off", "none"):
        return {}
    o = {}
    ev = {"ef": "evict_first", "el": "evict_last"}
    for t in (x.strip() for x in v.split(",") if x.strip()):
        if t == "r1":
            o["rm"] = 1
        elif t == "sp":
            o["sptr"] = 1
        elif t == "kb":
            # R15 iter25: constexpr K row stride -> K buffer loads.
            o["rsk"] = 1
        elif t[0] in "wk" and t[1:] in ("cg", "cv", "ca"):
            o["cmw" if t[0] == "w" else "cmk"] = "." + t[1:]
        elif t[0] in "wk" and t[1:] in ev:
            o["evw" if t[0] == "w" else "evk"] = ev[t[1:]]
        elif t[0] == "e" and t[1:].isdigit():
            o["waves_per_eu"] = int(t[1:])
        elif t[0] == "s" and t[1:].isdigit():
            o["stages"] = int(t[1:])
        else:
            raise ValueError(f"DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT token {t!r}")
    if v not in _DQ_SPLIT_OPT_LOGGED:
        _DQ_SPLIT_OPT_LOGGED.add(v)
        logger.warning("[hstu-attn] bwd dq SPLIT load opts ARMED: %s", o)
    return o


# X4C19 intglue ($DLRM_HSTU_FP8_ATTN_INTGLUE, read per call; code default
# OFF: absent or "0" runs the stock `seq_offsets.to(torch.int32)` +
# hstu_dq_split.build_workspace).  On a dq-split call the int32 seq offsets
# (cu) and the ds workspace's tile-prefix header come from ONE Triton launch
# (hstu_dq_split.build_workspace_intglue) instead of 10 tiny torch kernels
# (to, sub, add, floor_divide, fill, mul, add, mul, cumsum, copy).  Index
# bookkeeping only, the same int32 values (bitwise); no memo, no state across
# calls.  Other calls are unchanged.  Arming / declined lines are latched
# once per process in _ATTN_INTGLUE_LOGGED (restored by fp8_warmup, so the
# armed line prints at the first real backward, not inside the warmup).
_ATTN_INTGLUE_LOGGED = set()


def _attn_intglue_reason(seq_offsets: torch.Tensor) -> Optional[str]:
    """None when this dq-split call takes the fused int glue, else why not
    ("off" when the knob is off: the shipped path, nothing logged)."""
    raw = os.environ.get("DLRM_HSTU_FP8_ATTN_INTGLUE", "").strip().lower()
    if raw not in ("1", "true", "on", "yes"):
        return "off"
    from generative_recommenders.ops.flydsl import hstu_dq_split
    why = hstu_dq_split.intglue_unsupported_reason(seq_offsets)
    tag = ("armed",) if why is None else ("declined", why)
    if tag not in _ATTN_INTGLUE_LOGGED:
        _ATTN_INTGLUE_LOGGED.add(tag)
        if why is None:
            logger.warning(
                "[hstu-attn] DLRM_HSTU_FP8_ATTN_INTGLUE=%s armed: dq-split "
                "backward int glue (cu int32 + ds workspace tile-prefix "
                "header) in one Triton launch per call (_intglue_kernel), "
                "replacing 10 torch kernels; bitwise (debug check: "
                "DLRM_HSTU_FP8_ATTN_INTGLUE_CHECK=1)", raw)
        else:
            logger.warning(
                "[hstu-attn] DLRM_HSTU_FP8_ATTN_INTGLUE=%s declined on a "
                "dq-split call (%s); stock int glue for such calls", raw, why)
    return why


# Item (49) vf3p guard: the token is added only when 0 < qkv_scale <= this
# bound (one host float compare per armed call).  It keeps the regions where
# the reassociated chain can differ from the default by more than one e4m3
# code step, or in NaN behaviour, unreachable (R15 iteration-44
# exceptional-value study; user ruling N5 is requested with this guard).
_VF3P_QKV_SCALE_MAX = 0.25


# gin item (80) VF4 range guard on alpha*qkv_scale (on top of vf3p's guard):
# keeps the kernel's runtime 2**round(log2(alpha*s)) and the cast and undo
# scales it derives normal f32 (|ka| <= 50; the kernel asserts
# |log2 mu2_ssc| <= 50, the launcher checks it first).
_VF4_ALPHA_S_MIN = 2.0 ** -50
_VF4_ALPHA_S_MAX = 2.0 ** 50


def _vf4_kernel_ka(alpha: float, qkv_scale: float) -> int:
    """The ka the vf4 kernel computes (hstu_bwd_dkdv._vf4_cast_scale), on
    the host: cx = f32(f32(alpha) * f32(qkv_scale)) (the f32 product of two
    f32 values is exact in a double, so one rounding to f32 is IEEE f32
    multiplication), then the exponent of cx plus 1 when its mantissa is >=
    sqrt(2) (fraction field >= 0x3504F4), which is round(log2(cx)) for every
    positive normal f32 cx (no f32 ties exist); except that for alpha a
    half-integer power of two (f32 fraction 0x3504F3/0x3504F4) and qkv_scale
    a power of two, an exact tie of round(log2(alpha*qkv_scale)), ka is the
    even neighbour (Python round, as the iteration-51 host formula).  For
    the arming log line and tests only; the kernel does not read it."""
    import struct

    def bits(x: float) -> int:
        return struct.unpack("<I", struct.pack("<f", float(x)))[0]

    def f32(x: float) -> float:
        return struct.unpack("<f", struct.pack("<f", float(x)))[0]

    b = bits(f32(f32(alpha) * f32(qkv_scale)))
    if (bits(alpha) & 0x7FFFFF) in (0x3504F3, 0x3504F4) \
            and bits(qkv_scale) & 0x7FFFFF == 0:
        return (((b | 0x800000) & 0x7F800000) >> 23) - 127
    return (((b + 0x4AFB0C) & 0x7F800000) >> 23) - 127


def _bwd_vf4_tokens(vs: set, config: dict, alpha: float,
                    qkv_scale: Optional[float]) -> set:
    """R15 gin item (VF4) ($DLRM_HSTU_FP8_BWD_VF4, read per call; code
    default OFF, the gin binding is True: DECLARED PRECISION CHANGE, a
    pre-set "0" is the kill switch): add the VALU token "vf4" (vf3p minus
    the x' multiply: silu' = qk*r = silu/(alpha*s), 6 pk ops per element
    pair; dS ulp-level vs vf3p, silu re-rounded on a shifted e4m3 grid).
    The token
    "vf4" in $DLRM_HSTU_FP8_BWD_VALU arms it too (the guard still
    applies).  Needs vf3p and msat in the effective set, the f8b=all k128 path,
    0 < qkv_scale <= _VF3P_QKV_SCALE_MAX, _VF4_ALPHA_S_MIN <= alpha*qkv_scale
    <= _VF4_ALPHA_S_MAX and |log2 mu2_ssc| <= 50 (host float compares); a
    call that cannot take it runs without vf4 and logs one "ignored" line
    per reason.  The kernel re-centres the silu' e4m3 cast scale by
    sc = 2**round(log2(alpha*s)) from the RUNTIME qkv_scale it already
    receives (vf2's recipe: cvt_scalef32's scale field is E8M0, pow2 only)
    and its dv epilogue multiplies by 1/(mu2_ssc*sc*mu2_osc)*(alpha*s), so
    the residual alpha*s/sc in [2**-0.5, 2**0.5] is applied exactly in f32.
    config is NOT changed: the token is the only compile-key change, one
    build for every qkv_scale (the R15 iteration-51 probe folded 2**ka into
    mu2_ssc here: one build per alpha*qkv_scale binade)."""
    raw = os.environ.get("DLRM_HSTU_FP8_BWD_VF4", "").strip().lower()
    want = raw in ("1", "true", "on", "yes") or "vf4" in vs
    vs = set(vs) - {"vf4"}
    if not want:
        return vs
    if "vf3p" not in vs:
        why = "needs vf3p in the effective VALU set (valu=%s)" % (
            ",".join(sorted(vs)) or "<none>")
    elif "msat" not in vs:
        why = "needs VALU msat (valu=%s)" % ",".join(sorted(vs))
    elif config.get("f8b") != "all" or config.get("mu2") != "k128":
        why = "needs the f8b=all mu2=k128 path"
    elif not (qkv_scale is not None
              and 0.0 < float(qkv_scale) <= _VF3P_QKV_SCALE_MAX):
        why = "qkv_scale outside 0 < qkv_scale <= %g" % _VF3P_QKV_SCALE_MAX
    elif not (_VF4_ALPHA_S_MIN <= float(alpha) * float(qkv_scale)
              <= _VF4_ALPHA_S_MAX):
        why = "alpha*qkv_scale outside 2^-50..2^50"
    elif abs(math.frexp(float(config["mu2_ssc"]))[1] - 1) > 50:
        why = "mu2_ssc outside 2^-50..2^50"
    else:
        vs = vs | {"vf4"}
        if ("vf4", ",".join(sorted(vs))) not in _logged_layouts:
            _logged_layouts.add(("vf4", ",".join(sorted(vs))))
            ka = _vf4_kernel_ka(alpha, qkv_scale)
            logger.warning(
                "[hstu-attn] DLRM_HSTU_FP8_BWD_VF4=%s armed: bwd_dkdv VALU "
                "token vf4 added (valu=%s); silu e4m3 cast scale re-centred "
                "in the kernel by 2^round(log2(alpha*qkv_scale)) from the "
                "runtime qkv_scale, one build for every qkv_scale (first "
                "call: qkv_scale=%.6g ka=%d, cast scale 2^%d)",
                raw or "token", ",".join(sorted(vs)), float(qkv_scale), ka,
                int(math.frexp(float(config["mu2_ssc"]))[1] - 1) + ka)
        return vs
    if ("vf4_ignored", why) not in _logged_layouts:
        _logged_layouts.add(("vf4_ignored", why))
        logger.warning(
            "[hstu-attn] DLRM_HSTU_FP8_BWD_VF4=%s ignored: %s", raw or "token",
            why)
    return vs


def _bwd_vf3p_tokens(vs: set, off_reason: str = "",
                     qkv_scale: Optional[float] = None) -> set:
    """R15 item (49) ($DLRM_HSTU_FP8_BWD_VF3P, read per call; code default
    OFF, the gin binding is True, a pre-set "0" is
    the kill switch): add the VALU token "vf3p" (compile_hstu_bwd_dkdv(
    valu=): the act chain by f32 reassociation at the default cast scales)
    to the effective token set of a dq-split call.  It needs "ilp", excludes
    vf/vf2/h16/vf3/vf3s and needs 0 < qkv_scale <= _VF3P_QKV_SCALE_MAX (no
    scale = not taken); a call that cannot take it keeps the default chain
    and logs one "ignored" line per reason.  A set that already names vf3p
    (a pre-set DLRM_HSTU_FP8_BWD_VALU) is returned unchanged, so this knob
    cannot remove it and the guard does not apply to it.  The token is part
    of config["valu"], hence of the compile key."""
    raw = os.environ.get("DLRM_HSTU_FP8_BWD_VF3P", "").strip().lower()
    if raw not in ("1", "true", "on", "yes") or "vf3p" in vs:
        return vs
    clash = sorted(vs & {"vf", "vf2", "h16", "vf3", "vf3s"})
    if off_reason:
        why = off_reason
    elif "ilp" not in vs:
        why = "needs VALU ilp (valu=%s)" % (",".join(sorted(vs)) or "<none>")
    elif clash:
        why = "excludes %s (valu=%s)" % (",".join(clash), ",".join(sorted(vs)))
    elif not (qkv_scale is not None
              and 0.0 < float(qkv_scale) <= _VF3P_QKV_SCALE_MAX):
        why = "qkv_scale outside the guard 0 < qkv_scale <= %g" % (
            _VF3P_QKV_SCALE_MAX)
    else:
        vs = vs | {"vf3p"}
        if ("vf3p", ",".join(sorted(vs))) not in _logged_layouts:
            _logged_layouts.add(("vf3p", ",".join(sorted(vs))))
            logger.warning(
                "[hstu-attn] DLRM_HSTU_FP8_BWD_VF3P=%s armed: bwd_dkdv VALU "
                "token vf3p added (valu=%s)", raw, ",".join(sorted(vs)))
        return vs
    if ("vf3p_ignored", why) not in _logged_layouts:
        _logged_layouts.add(("vf3p_ignored", why))
        logger.warning(
            "[hstu-attn] DLRM_HSTU_FP8_BWD_VF3P=%s ignored: %s; default act "
            "chain", raw, why)
    return vs


# DLRM_HSTU_FP8_LAB_BWD_SPF (R15 iter75 tscv2; LAB knob, read per call, code
# default OFF, deliberately NOT bound in gin): the backward half of the SPF
# fold (x4c19 contract v1.1, probes/iter71/spf/CONTRACT.md), bwd_dkdv's VALU
# token "tsc" (R15 iter74 f2). Host side in ops/fp8_qkv_spf_bwd.py.
#   unset / "0"  off: the backward of the stack without this knob. A "tsc"
#                token pre-set in DLRM_HSTU_FP8_BWD_VALU, or an exact "tsc" in
#                config["valu"] by any other path: stripped, one log line per
#                process (other spellings fail the valu assert). Never armed.
#   "1"          needs DLRM_HSTU_FP8_LAB_FWD_SPF in {"snap", "1"} (ValueError
#                at parse). Every call with e4m3 q AND k, before compile and
#                launch: fp8_qkv_spf.arm_check plus the backward range
#                [-40, -7] (RuntimeError; contract section 6: an unsnapped
#                scale, e.g. resumed from a checkpoint written without the
#                knob, fails closed), the vf4 + msat + msplit=1 + k128 body
#                (ValueError), then "tsc" joins config["valu"] (the compile
#                key; j is runtime, one build for every j) and one ARMED line
#                per process names j and the E8M0 byte 127 + j. bwd_dkdv's
#                call() re-checks the scale (tsc_exponent). An e4m3 call the
#                FlyDSL backward does not serve raises ValueError in
#                try_flydsl_attention_bwd. bf16 q/k calls (calibration) are
#                not armed and do not raise.
#   other        ValueError.
# Logging (ship17 row (106)): the once-latch maps each key ("armed", and
# ("strip", source)) to the line it logged. fp8_warmup's F2/F2q/F2e reach
# the hook while _Guards mutes this logger; the latch is deliberately NOT in
# fp8_warmup._MHA_SETS (it is kept: _KEEP_GLOBALS), and fp8_warmup logs the
# latched line after the restore (_spf_bwd_armed_reprint), so the ARMED line
# prints once per process before RUN_START and never in the timed region.
_BWD_SPF_KNOB = "DLRM_HSTU_FP8_LAB_BWD_SPF"
_fp8_bwd_spf_logged = {}


def _bwd_spf_strip_log(source: str, value: str) -> None:
    """One line per process and source for a stripped stray "tsc" token."""
    if ("strip", source) in _fp8_bwd_spf_logged:
        return
    from generative_recommenders.ops import fp8_qkv_spf_bwd as spfb

    line = spfb.strip_line(source, value, os.environ.get(_BWD_SPF_KNOB, ""))
    _fp8_bwd_spf_logged[("strip", source)] = line
    logger.warning("[hstu-attn] %s", line)


def _bwd_spf_tokens(config: dict, kernel, fp8_mfma: str, e4m3_qk: bool,
                    alpha: float, qkv_scale) -> None:
    """$DLRM_HSTU_FP8_LAB_BWD_SPF (read per call) on one backward call's final
    compile config, in place: strips a stray "tsc" from config["valu"]; under
    "1" on an e4m3 q/k call, checks the scale (RuntimeError) and the body
    (ValueError), then adds "tsc". Off without a stray token: returns at once
    (no import, no change)."""
    valu = str(config.get("valu", ""))
    raw = os.environ.get(_BWD_SPF_KNOB, "")
    if raw in ("", "0") and "tsc" not in valu.split(","):
        return
    from generative_recommenders.ops import fp8_qkv_spf as spf
    from generative_recommenders.ops import fp8_qkv_spf_bwd as spfb

    mode = spfb.bwd_spf_mode()
    vs = {t for t in valu.split(",") if t}
    if spfb.TOKEN in vs:
        vs.discard(spfb.TOKEN)
        _bwd_spf_strip_log('config["valu"]', valu)
        if vs:
            config["valu"] = ",".join(sorted(vs))
        else:
            config.pop("valu", None)
    if mode == "off" or not e4m3_qk:
        return
    j = spfb.arm_check_bwd(qkv_scale, alpha)
    why = spfb.body_reason(config, fp8_mfma, vs)
    if why is not None:
        raise ValueError(
            f"{_BWD_SPF_KNOB}=1 needs the vf4 + msat + msplit=1 + k128 "
            f"dq-split backward on every e4m3 call: {why}. Refusing to run "
            "this call unfolded under the fold arm's label.")
    k_rng = (getattr(kernel, "TSC_K_MIN", None),
             getattr(kernel, "TSC_K_MAX", None))
    if k_rng != (spfb.BWD_J_MIN, spfb.BWD_J_MAX):
        raise RuntimeError(
            f"{_BWD_SPF_KNOB}=1: bwd_dkdv's tsc guard range {k_rng} differs "
            f"from the arm range ({spfb.BWD_J_MIN}, {spfb.BWD_J_MAX})")
    vs.add(spfb.TOKEN)
    config["valu"] = ",".join(sorted(vs))
    if "armed" not in _fp8_bwd_spf_logged:
        line = spfb.armed_line(j, float(qkv_scale),
                               os.environ.get(spf.KNOB, ""), config["valu"])
        _fp8_bwd_spf_logged["armed"] = line
        logger.warning("[hstu-attn] %s", line)


def _bwd_spf_refuse_unserved(q, k, why: str) -> None:
    """$DLRM_HSTU_FP8_LAB_BWD_SPF=1: an e4m3 attention backward that the
    FlyDSL backward does not serve (not selected, shape fallback) would run
    the Triton backward, unfolded, under the fold arm's label: ValueError.
    Off: nothing (an invalid value raises ValueError, as in the backward)."""
    if os.environ.get(_BWD_SPF_KNOB, "") in ("", "0"):
        return
    from generative_recommenders.ops import fp8_qkv_spf_bwd as spfb

    if (spfb.bwd_spf_mode() == "fold" and FP8_E4M3 is not None
            and q.dtype is FP8_E4M3 and k.dtype is FP8_E4M3):
        raise ValueError(
            f"{_BWD_SPF_KNOB}=1: this e4m3 attention backward is not served "
            f"by the FlyDSL backward ({why}); the fold needs it. Refusing to "
            "run the Triton backward under the fold arm's label.")


def fp8_bwd_spf_warmup_scale(qkv_scale: float, alpha: float) -> float:
    """The runtime scale fp8_warmup's F2/F2q/F2e hand the FP8 backward:
    `qkv_scale` unchanged unless DLRM_HSTU_FP8_LAB_BWD_SPF=1, else snapped
    onto the SPF grid (the synthetic 2**-6 -> s_-14), so the warmup passes
    the per-call grid check and compiles the arm's tsc build."""
    if os.environ.get(_BWD_SPF_KNOB, "") in ("", "0"):
        return qkv_scale
    from generative_recommenders.ops import fp8_qkv_spf_bwd as spfb

    return spfb.warmup_scale(qkv_scale, alpha)


# ---------------------------------------------------------------------------
# R15 A11 hook (hstu_do8_registry). The C8 do_blk predicate is
# shared by the backward branch in flydsl_hstu_attention_bwd and by
# c8_do_blk_planned, the attention forward's dO8 site gate, so the two
# cannot drift apart.


def _c8_blk_ok(config, heads, dim) -> bool:
    """do_blk's K-block <-> token-group identity needs BM=64 and BN % 128 == 0
    (kernel assert), and quantize_do_e4m3_blk needs power-of-two H and H*D."""
    return (int(config.get("BM", 64)) == 64
            and int(config.get("BN", 128)) % 128 == 0
            and heads > 0 and heads & (heads - 1) == 0
            and (heads * dim) & (heads * dim - 1) == 0)


def _c8_do_blk_selected(f8b_full, fp8_mfma, input_fp8, q_bf16, config,
                        blk_ok) -> bool:
    """The C8 branch condition of flydsl_hstu_attention_bwd: the full-fp8
    backward with the block-scaled e4m3 dO of quantize_do_e4m3_blk."""
    return bool(f8b_full and fp8_mfma == "k128" and input_fp8 and not q_bf16
                and config.get("fused") and config.get("dq16") == "bf16p"
                and blk_ok)


def _targets_max_known(num_targets) -> bool:
    """True when _max_num_targets(num_targets) answers from its memo, i.e.
    without a device-to-host sync."""
    cached, version, _ = _targets_max_memo
    return cached is num_targets and version == num_targets._version


def c8_do_blk_planned(
    *, q, k, v, out, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0, qkv_scale: float = 1.0,
) -> bool:
    """Will this attention call's FlyDSL backward quantize dO on the C8 do_blk
    path? The forward-time gate of the R15 A11 dO8 site (hstu_do8_registry).

    The backward's own predicates through the backward's own functions, on the
    forward's q/k/v (the backward reloads or recomputes exactly these e4m3
    tensors) with `out` standing in for dout (autograd hands the backward a
    gradient of out's dtype and shape):
      try_flydsl_attention_bwd: flydsl_selected(); a usable host
        (unsupported_hardware_reason() / _load_kernel(), never raising here);
        unsupported_shape_reason(q, k, v, out, num_targets, ...);
      flydsl_hstu_attention_bwd: bwd_full_fp8_enabled() and
        check_removed_c8_knobs(); e4m3 q AND k (input_fp8 and not q_bf16);
        qkv_scale > 0; fp8_mfma == "k128" (bwd_qk_mfma_mode() fp8 or
        fp8_strict); _c8_do_blk_selected(...) on kernel.SHIPPED with
        _c8_blk_ok(...).
    It never syncs: a num_targets max that is not memoized yet answers False
    (no site; the backward quantizes as today). The full-length layer only:
    q/k/v/out must share the token count (the targets-only Function, with its
    compact q, never calls this and never reaches this backward). The
    DLRM_HSTU_FP8_BWD_FULL_HYBRID diagnostic answers False (it runs both
    kernels). Known only in the backward, and caught there by
    hstu_do8_registry.take() / discard() with a counter: env changes between
    forward and backward, and a dO that is not the producer's dx (autograd
    accumulation, a hook, a non-contiguous gradient). Any exception: False.
    Contract v2: with DLRM_HSTU_FP8_L11_DO8_ONLY=1 a planned call also keeps
    its plan inputs on `out` (_Do8PlanFacts, picked up by new_site) so that
    hstu_do8_registry.plan() can re-evaluate this predicate at the
    producer's backward.
    """
    try:
        planned = _c8_do_blk_plan_reason(
            q=q, k=k, v=v, out=out, num_targets=num_targets,
            max_attn_len=max_attn_len, contextual_seq_len=contextual_seq_len,
            num_softmax_heads=num_softmax_heads, qkv_scale=qkv_scale,
        ) is None
        if planned and hstu_do8_registry.only_requested():
            hstu_do8_registry.stash_plan(out, _Do8PlanFacts(
                q, k, v, out, num_targets, max_attn_len, contextual_seq_len,
                num_softmax_heads, qkv_scale))
        return planned
    except Exception:  # noqa: BLE001 - a site gate never breaks a forward
        return False


def _c8_do_blk_plan_reason(
    *, q, k, v, out, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0, qkv_scale: float = 1.0,
) -> Optional[str]:
    """c8_do_blk_planned's predicates in its order: None when the call's
    FlyDSL backward takes the C8 do_blk branch, else the first failing one.
    May raise (check_removed_c8_knobs, an invalid QK_MFMA value): callers
    treat that as not planned. q/k/v/out are read for .shape, .dtype and
    .dim() only (contract v2 re-evaluates it from _TSpec stand-ins)."""
    if not flydsl_selected():
        return "HSTU_ATTN_BWD is not flydsl"
    if not bwd_full_fp8_enabled():
        return "DLRM_HSTU_FP8_BWD_FULL off"
    if os.environ.get("DLRM_HSTU_FP8_BWD_FULL_HYBRID", ""):
        return "DLRM_HSTU_FP8_BWD_FULL_HYBRID set (both kernels run)"
    check_removed_c8_knobs()
    if not _host_ok and (unsupported_hardware_reason() is not None
                         or _load_kernel() is None):
        return "host cannot run the FlyDSL backward"
    kernel = _load_kernel()
    if kernel is None:
        return "no FlyDSL backward kernel module"
    input_fp8 = FP8_E4M3 is not None and k.dtype is FP8_E4M3
    q_bf16 = input_fp8 and q.dtype is torch.bfloat16
    if not input_fp8 or q_bf16 or not (qkv_scale > 0.0):
        return ("q and k not both e4m3, or qkv_scale <= 0 (an FP8 "
                "calibration step: the bf16-dO kernel reads the bf16 dO)")
    if (out is None or out.dtype is not torch.bfloat16 or out.dim() != 3
            or tuple(out.shape) != tuple(v.shape)
            or not (q.shape[0] == k.shape[0] == v.shape[0]
                    == out.shape[0])):
        return "not the full-length ordinary layer (q/k/v/dO shapes)"
    if num_targets is not None and not _targets_max_known(num_targets):
        return "num_targets max not memoized (no sync here)"
    reason = unsupported_shape_reason(
        q, k, v, out, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    )
    if reason is not None:
        return "shape fallback: " + reason
    fp8_mfma = ("k128" if bwd_qk_mfma_mode() in ("fp8", "fp8_strict")
                else "")
    config = kernel.SHIPPED
    heads, dim = q.shape[1], q.shape[2]
    if not _c8_do_blk_selected(True, fp8_mfma, input_fp8, q_bf16, config,
                               _c8_blk_ok(config, heads, dim)):
        return "C8 do_blk not selected (qk_mfma or kernel config)"
    return None


# R15 A11 contract v2: the debug readers of the bf16 dO (_full_fp8_stats and
# _bwd_dump read dout_orig, the HSTU_ATTN_BWD_CHECK reference reads dout).
_DO8_ONLY_DEBUG_ENVS = ("DLRM_HSTU_FP8_BWD_FULL_STATS", "DLRM_HSTU_FP8_BWD_DUMP",
                        "DLRM_HSTU_FP8_BWD_DUMP_CALLS", "HSTU_ATTN_BWD_CHECK")


def _do8_only_debug_reason() -> Optional[str]:
    """Contract v2: the first armed debug reader of the bf16 dO, else None.
    DLRM_HSTU_FP8_BWD_DUMP counts on any non-empty raw value, the dump
    reader's own predicate (_bwd_dump_dir: a directory name, "0" and
    whitespace included); the integer knobs on any value but "" or "0"
    (after strip). Stricter than the helpers: rank, call index and
    remaining count are not consulted."""
    for name in _DO8_ONLY_DEBUG_ENVS:
        raw = os.environ.get(name, "")
        if (raw != "" if name == "DLRM_HSTU_FP8_BWD_DUMP"
                else raw.strip() not in ("", "0")):
            return f"debug reader of the bf16 dO armed: {name}"
    return None


class _TSpec:
    """Shape and dtype of a tensor, without its storage: all the C8 plan
    predicates read (contract v2)."""

    __slots__ = ("shape", "dtype")

    def __init__(self, t) -> None:
        self.shape = t.shape
        self.dtype = t.dtype

    def dim(self) -> int:
        return len(self.shape)


class _Do8PlanFacts:
    """Contract v2: one ordinary-layer attention call's C8 plan inputs, kept
    by c8_do_blk_planned (the 'only' knob on) on the site and re-evaluated by
    hstu_do8_registry.plan() at the producer's backward, before its launch,
    with the environment and num_targets memo of that moment. Keeps no
    tensor storage except num_targets (B ints, saved by the attention ctx
    anyway; the memo key is its identity and _version)."""

    __slots__ = ("q", "k", "v", "out", "num_targets", "max_attn_len",
                 "contextual_seq_len", "num_softmax_heads", "qkv_scale")

    def __init__(self, q, k, v, out, num_targets, max_attn_len,
                 contextual_seq_len, num_softmax_heads, qkv_scale) -> None:
        self.q, self.k, self.v, self.out = (
            _TSpec(q), _TSpec(k), _TSpec(v), _TSpec(out))
        self.num_targets = num_targets
        self.max_attn_len = max_attn_len
        self.contextual_seq_len = contextual_seq_len
        self.num_softmax_heads = num_softmax_heads
        self.qkv_scale = float(qkv_scale)

    def reason(self) -> Optional[str]:
        """None: the call still takes the C8 do_blk branch and no debug
        reader of the bf16 dO is armed; else why not ('plus')."""
        try:
            why = _c8_do_blk_plan_reason(
                q=self.q, k=self.k, v=self.v, out=self.out,
                num_targets=self.num_targets, max_attn_len=self.max_attn_len,
                contextual_seq_len=self.contextual_seq_len,
                num_softmax_heads=self.num_softmax_heads,
                qkv_scale=self.qkv_scale)
        except Exception as error:  # noqa: BLE001 - 'plus', never a raise
            why = f"the C8 predicate raised {type(error).__name__}"
        if why is not None:
            return "not C8: " + why
        return _do8_only_debug_reason()


def _do8_only_require_c8(dout, q, k, v, qkv_scale) -> None:
    """Contract v2, top of flydsl_hstu_attention_bwd for a dO that carries an
    'only' record: raise hstu_do8_registry.Do8OnlyError, before anything
    below reads dout, unless this call takes the C8 do_blk branch (whose
    take() hands over the record) with no debug reader of the bf16 dO. The
    branch's own predicates: no HYBRID, DLRM_HSTU_FP8_BWD_FULL, e4m3 q and k,
    qk_mfma fp8/fp8_strict, _c8_do_blk_selected on kernel.SHIPPED (the
    branch's config differs from SHIPPED only in sched/scale_in_cvt, which
    it does not read). try_flydsl_attention_bwd's selection and shape
    returns raise through hstu_do8_registry.discard()."""
    kernel = _load_kernel()
    input_fp8 = FP8_E4M3 is not None and k.dtype is FP8_E4M3
    q_bf16 = input_fp8 and q.dtype is torch.bfloat16
    if os.environ.get("DLRM_HSTU_FP8_BWD_FULL_HYBRID", ""):
        why = "DLRM_HSTU_FP8_BWD_FULL_HYBRID also runs the bf16-dO kernel"
    elif not bwd_full_fp8_enabled():
        why = "DLRM_HSTU_FP8_BWD_FULL is off (bf16-dO kernel)"
    elif kernel is None or not _c8_do_blk_selected(
            True,
            "k128" if (input_fp8 and not q_bf16 and bwd_qk_mfma_mode()
                       in ("fp8", "fp8_strict")) else "",
            input_fp8, q_bf16, kernel.SHIPPED,
            _c8_blk_ok(kernel.SHIPPED, q.shape[1], q.shape[2])):
        why = ("no C8 do_blk branch for this call (q/k dtypes, e.g. an FP8 "
               "calibration step; qk_mfma; kernel config)")
    else:
        why = _do8_only_debug_reason()
        if why is None and not (dout.dtype is torch.bfloat16
                                and dout.is_contiguous()):
            why = "the dO is not a contiguous bf16 tensor"
    if why is not None:
        raise hstu_do8_registry.Do8OnlyError(
            "[hstu-attn] dO8 contract v2 'only': this attention backward "
            f"cannot serve a dO whose bf16 values were never written: {why}")


def flydsl_hstu_attention_bwd(
    dout,
    q,
    k,
    v,
    seq_offsets,
    N,
    alpha,
    out=None,
    qkv_scale: float = 1.0,
    dq_fp8_sink=None,
    dkdv_fp8_sink=None,
):
    """The five-gemm fused backward with BF16 or scaled-E4M3 Q/K/V.

    With `out=(dq, dk, dv)` the gradients are written straight into those
    tensors, which may be strided views; otherwise fresh packed tensors are
    allocated and returned. The in-place form is what avoids the wrapper tax on
    the production path -- see try_flydsl_attention_bwd.

    `dq_fp8_sink` (Arbor R14 M2 hook, default None = no-op): a callable with
    `hstu_dq_split.dq_from_ds`'s signature that the dq-split call uses
    instead (r14's hstu_dq_e5m2.DqE5m2Sink emits e5m2 dz q columns and leaves
    the BF16 dq unwritten). Only the dq-split path calls it.

    `dkdv_fp8_sink` (Arbor R15 M-E1 hook, default None = no-op): a
    hstu_dkdv_e5m2.DkdvE5m2Sink. When its decline_reason() accepts the
    compiled config, bwd_dkdv writes the e5m2 dV|dK columns of the sink's
    dz8 (+ amax/colsum partials) INSTEAD of the BF16 dV/dK; otherwise
    sink.declined is set and the stock kernel runs.

    dq is accumulated across kv blocks with global atomics, so it must start at
    zero -- the zero-fill cannot be folded into the kernel, because nothing
    orders one workgroup's zeroing against another's atomic add. Under the
    shipped dq16 mode dq is bf16 and IS the accumulator, so there is no
    down-convert pass and the zero-fill moves half the bytes.
    """
    # R15 A11 contract v2 ('only'): a dO whose producer skipped the bf16
    # store must take the C8 branch below; raise before anything reads it.
    if hstu_do8_registry.only_record(dout) is not None:
        _do8_only_require_c8(dout, q, k, v, qkv_scale)
    _hy = os.environ.get("DLRM_HSTU_FP8_BWD_FULL_HYBRID", "")
    if _hy and not _hybrid_inner[0] and bwd_full_fp8_enabled():
        if _hy not in ("dq", "dkdv"):
            raise ValueError(
                f"DLRM_HSTU_FP8_BWD_FULL_HYBRID={_hy!r}: expected dq|dkdv")
        _hybrid_inner[0] = True
        try:
            _hybrid_force_off[0] = True
            try:
                ref = flydsl_hstu_attention_bwd(
                    dout, q, k, v, seq_offsets, N, alpha, out=None,
                    qkv_scale=qkv_scale)
            finally:
                _hybrid_force_off[0] = False
            dq, dk, dv = flydsl_hstu_attention_bwd(
                dout, q, k, v, seq_offsets, N, alpha, out=out,
                qkv_scale=qkv_scale)
        finally:
            _hybrid_inner[0] = False
        if _hy == "dq":
            dq.copy_(ref[0])
        else:
            dk.copy_(ref[1])
            dv.copy_(ref[2])
        if "f8b_hybrid" not in _logged_layouts:
            _logged_layouts.add("f8b_hybrid")
            logger.warning(
                "[hstu-attn] FULL-FP8 HYBRID=%s ARMED: %s taken from the "
                "bf16-backward kernel (diagnostic; both kernels run)", _hy,
                "dq" if _hy == "dq" else "dk,dv")
        return dq, dk, dv
    kernel = _load_kernel()
    heads, dim = q.shape[1], q.shape[2]
    acc_dtype = kernel.dq_acc_dtype()
    input_fp8 = FP8_E4M3 is not None and k.dtype is FP8_E4M3
    q_bf16 = input_fp8 and q.dtype is torch.bfloat16
    if input_fp8 and not (qkv_scale > 0.0):
        raise ValueError("FP8 Q/K/V requires a positive dequantization scale")

    # Read q/k/v where they already live when the layout allows it. dO is not
    # part of the uvqk buffer (it is the attention output's gradient) and keeps
    # the packed stride, so .contiguous() stays -- a no-op in production.
    if q_bf16:
        rs_in = _common_row_stride((k, v), heads, dim)
        rs_q = _common_row_stride((q,), heads, dim)
    else:
        rs_in = _common_row_stride((q, k, v), heads, dim)
        rs_q = rs_in
    rs_out = None
    if out is not None:
        dq, dk, dv = out
        # Only when dq IS the accumulator. A narrower accumulator than the
        # caller's dq would need a down-convert, and that has nowhere to land
        # in place, so those modes take the allocate-and-copy path.
        if acc_dtype is torch.bfloat16 and dq.dtype is acc_dtype:
            rs_out = _common_row_stride((dq, dk, dv), heads, dim)

    dout = dout.contiguous()
    if q_bf16:
        if rs_in is None:
            k, v = k.contiguous(), v.contiguous()
            rs_in = heads * dim
        if rs_q is None:
            q = q.contiguous()
            rs_q = heads * dim
    elif rs_in is None:
        q, k, v = (t.contiguous() for t in (q, k, v))
        rs_in = heads * dim
        rs_q = rs_in
    # R15 design B ($DLRM_HSTU_FP8_BWD_DQ_SPLIT=1, default OFF): dq is computed
    # by hstu_dq_split from a stored e4m3 ds workspace with plain stores, so
    # the zero-fill below is deferred until the C8 path is known not to take
    # it.  Unset = the original zeroing at the original point.
    _dqs_req = os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT", "").strip().lower() \
        in ("1", "true", "on", "yes")
    if rs_out is None:
        output_shape = q.shape
        dk = torch.empty(output_shape, dtype=torch.bfloat16, device=q.device)
        dv = torch.empty(output_shape, dtype=torch.bfloat16, device=q.device)
        dq = (torch.empty if _dqs_req else torch.zeros)(
            output_shape, dtype=acc_dtype, device=q.device)
        rs_out = heads * dim
    elif not _dqs_req:
        # Same reasoning: dk/dv are fully overwritten, so only dq needs zeroing,
        # and it is zeroed in place rather than allocated.
        dq.zero_()
    config = dict(kernel.SHIPPED)
    if input_fp8:
        # The packed conversion changes the VALU issue mix enough that the
        # BF16-tuned interleave is no longer optimal on gfx950.
        config["sched"] = "iglp3"
    if q_bf16 and kernel.cvt_scale_is_exact(qkv_scale):
        # ONLY when the scale is an exact power of two. scale_in_cvt folds
        # qkv_scale into cvt_scalef32_pk_bf16_fp8 instead of applying it in the
        # score/epilogue math, and that instruction's scale operand is
        # exponent-only on gfx950 -- see cvt_scale_is_exact. With the calibrated
        # `_fp8_qkv_scale` (amax/448, a power of two with probability zero) the
        # fold silently rescales every gradient: measured dq 3.1 dB against
        # 28.1 dB for the same build with a power-of-two scale.
        #
        # Declining the fold is not a fallback to something worse -- it is the
        # arithmetically correct path, and the only cost is that
        # score_input_scale/value_input_scale no longer collapse to 1.0, so
        # their per-element multiplies stay in the elementwise chain (worth
        # 40.126 -> 34.689 ms on the full-fp8 arm; the q_bf16 arm was never
        # timed correct). The kernel re-checks this at every launch, so a caller
        # that sets the flag by hand gets a ValueError, not wrong numbers.
        config["scale_in_cvt"] = True
    elif q_bf16 and "q_bf16_scale_in_cvt" not in _logged_layouts:
        _logged_layouts.add("q_bf16_scale_in_cvt")
        logger.warning(
            "[hstu-attn] qkv_scale=%r is not an exact power of two, so the "
            "BF16-Q/E4M3-K/V backward applies it in the score math rather than "
            "folding it into the fp8 converts. Correct, and slower than the "
            "fold would be; quantize with a power-of-two scale to get both.",
            qkv_scale,
        )
    # True fp8 MFMA on the qk gemm. $DLRM_HSTU_FP8_BWD_QK_MFMA, DEFAULT fp8.
    #   "fp8"        k128 fp8 MFMA when this call's q AND k are already e4m3;
    #                calibration (bf16 q/k) stays bf16 math.
    #   "bf16"       explicit opt-out -- shipped-before-this-default math.
    #   "fp8_strict" engage, RAISE on any call whose q/k are not both e4m3.
    # Deprecated alias: $DLRM_HSTU_FP8_BWD_K128=0|auto|1 -> bf16|fp8|fp8_strict.
    fp8_mfma = ""
    _qk_mode = bwd_qk_mfma_mode()
    if _qk_mode in ("fp8", "fp8_strict"):
        if not input_fp8 or q_bf16:
            if _qk_mode == "fp8_strict":
                raise RuntimeError(
                    "DLRM_HSTU_FP8_BWD_QK_MFMA=fp8_strict requires E4M3 q AND "
                    "k on every backward call, but this call has "
                    f"q={q.dtype} k={k.dtype} (input_fp8={input_fp8}, "
                    f"q_bf16={q_bf16}). Use QK_MFMA=fp8 so calibration "
                    "steps stay on bf16 math, or QK_MFMA=bf16 to opt out."
                )
        else:
            fp8_mfma = "k128"
            config["sched"] = "iglp2"
    # C8 full-fp8 backward. $DLRM_HSTU_FP8_BWD_FULL (absent = OFF; gin ON).
    # All five gemms on mfma_scale_f32_16x16x128_f8f6f4. dO is quantized here
    # to e4m3 with one pow2 scale per (sequence, head, 32-token MX K-block
    # group) -- quantize_do_e4m3_blk, no host sync -- and the kernel undoes
    # them inside the out-gemm MFMAs (do_blk), with the saturating e4m3 cast
    # of silu/ds (f8_sat). That is the ONLY C8 path: the per-tensor dO scale
    # (do_dyn) flushed small dO rows to zero in-model and is no longer
    # reachable from here, and do_blk without f8_sat NaNs gradients.  A call
    # shape do_blk cannot serve takes the bf16-dO kernel below instead.
    do_scale = None
    f8b_full = bwd_full_fp8_enabled()
    if f8b_full:
        check_removed_c8_knobs()
    # do_blk's K-block <-> token-group identity needs BM=64 and BN % 128 == 0
    # (kernel assert), and quantize_do_e4m3_blk needs power-of-two H and H*D.
    # R15 A11: the same predicates, shared with c8_do_blk_planned.
    _blk_ok = _c8_blk_ok(config, heads, dim)
    if _c8_do_blk_selected(f8b_full, fp8_mfma, input_fp8, q_bf16, config,
                           _blk_ok):
        # silu and ds both carry the kernel's 1/N, so their e4m3 pre-scales
        # are N-RELATIVE: 2**(K + ceil(log2 N)) puts silu*N (resp. ds*N with
        # normalized dO) times 2**K..2**(K+1) on the e4m3 grid, and only one
        # or two kernels compile (ceil(log2 N) is 12 for every in-model N
        # seen in 51366).  In-model 51366: |silu*N| <= ~2 (|x| <= 2.2),
        # |ds*N| 0.5..323 across layers.  An absolute *_LOG2 overrides.
        # f8_sat (always on) clamps before the cast: gfx950's
        # cvt_scalef32_pk_fp8_f32 returns NaN, not 448, on overflow.
        c = int(math.ceil(math.log2(max(int(N), 2))))
        def _sc(name, k_default):
            ab = os.environ.get(f"DLRM_HSTU_FP8_BWD_FULL_{name}_LOG2")
            if ab not in (None, ""):
                return 2.0 ** int(ab)
            return 2.0 ** (c + int(os.environ.get(
                f"DLRM_HSTU_FP8_BWD_FULL_{name}_K", k_default)))
        config.update(
            f8b="all", mu2="k128", do_dyn=False, do_blk=True, f8_sat=True,
            # Defaults sit mid-plateau of the 51381 in-model sweep: operand
            # SNR is flat ~31.5 dB for silu K in [-2, 8] and ds K in [-6, 0]
            # (ds falls off a cliff at K >= 2 on some layers).
            mu2_ssc=_sc("SSC", "3"),
            mu2_dsc=_sc("DSC", "-3"),
            ds8_pad=int(os.environ.get("DLRM_HSTU_FP8_BWD_FULL_DS8_PAD", "16")),
        )
        # The silu clamp is upper-only (silu*N >= -0.279), which stays exact
        # while -0.279 * ssc * 2/2**c > -448, i.e. K <= 9.
        assert config["mu2_ssc"] <= 2.0 ** (c + 9), (
            "DLRM_HSTU_FP8_BWD_FULL ssc too large for the upper-only clamp")
        from generative_recommenders.ops.flydsl.hstu_do_fp8 import (
            quantize_do_e4m3_blk,
        )
        if _dqs_req and dq.dtype is torch.bfloat16:
            config["dqsplit"] = True
        dout_orig = dout
        # One pow2 dO scale per (sequence, head, MX token group); do_scale is
        # E8 [T, H] uint8 (127 - e_g), undone inside the kernel's out-gemm
        # MFMAs (no epilogue rescale).
        # R15 A11 hook (hstu_do8_registry): when the d(attn) producer
        # already emitted these exact bytes for this dO (a record bound to
        # dout, every check passing) they are taken; otherwise -- no record,
        # a failed check, the knob off, a second take -- the quantizer runs
        # as before. dout_orig stays the bf16 dO.
        _do8 = hstu_do8_registry.take(dout, seq_offsets, N)
        if _do8 is None:
            dout, do_scale = quantize_do_e4m3_blk(dout, seq_offsets, N)
        else:
            dout, do_scale = _do8
        _do8 = None
    elif f8b_full:
        # Two once-keys: the expected calibration-step skips (bf16 q/k) must
        # not hide a later block-dO shape fallback, which silently drops a
        # production call from C8.
        _skip_key = "f8b_full_skip" if _blk_ok else "f8b_full_skip_shape"
        if _skip_key not in _logged_layouts:
            _logged_layouts.add(_skip_key)
            logger.warning(
                "[hstu-attn] DLRM_HSTU_FP8_BWD_FULL set but this call cannot "
                "use it (%s; qk_mfma=%r input_fp8=%s q_bf16=%s "
                "do_blk_shape_ok=%s heads=%d dim=%d BM=%s BN=%s); bf16-dO "
                "kernel used.",
                "operand/config" if _blk_ok else "block-dO SHAPE fallback",
                fp8_mfma or "bf16", input_fp8, q_bf16, _blk_ok, heads, dim,
                config.get("BM", 64), config.get("BN", 128),
            )
    if do_scale is None:
        # R15 A11: no C8 dO quantization on this call, so a pending dO8
        # record bound to dout is released here, not pinned.
        hstu_do8_registry.discard(dout, "not_c8")
    # silu/grad-coefficient epilogue form (roadmap R3).
    # $DLRM_HSTU_FP8_BWD_EPILOGUE = legacy (default when absent) | reuse
    # (gin: apply_env_bootstrap.DLRM_HSTU_FP8_BWD_EPILOGUE).
    _epilogue = os.environ.get("DLRM_HSTU_FP8_BWD_EPILOGUE", "legacy") or "legacy"
    if _epilogue not in ("reuse", "legacy", "reuse_pk"):
        raise ValueError(
            f"DLRM_HSTU_FP8_BWD_EPILOGUE={_epilogue!r}: expected "
            "reuse|legacy|reuse_pk"
        )
    # R15 VALU diet: "reuse_pk" runs the "reuse" silu/grad-coefficient chain
    # on packed f32 (v_pk_mul/add/fma_f32, two accumulator elements per issue;
    # exp2/rcp stay scalar).  Same ops in the same order, so the outputs are
    # bitwise equal to "reuse"; only the VALU issue count drops (and the
    # 8-VGPR spill of the scalar build goes away: 484 -> 424 regs).  Default
    # OFF (gin keeps "reuse"); arm with $DLRM_HSTU_FP8_BWD_EPILOGUE=reuse_pk
    # or $DLRM_HSTU_FP8_BWD_PK=1 (upgrades "reuse" to "reuse_pk").
    if (os.environ.get("DLRM_HSTU_FP8_BWD_PK", "").strip().lower()
            in ("1", "true", "on", "yes") and _epilogue == "reuse"):
        _epilogue = "reuse_pk"
    if _epilogue == "reuse_pk" and ("epi", "reuse_pk") not in _logged_layouts:
        _logged_layouts.add(("epi", "reuse_pk"))
        logger.warning(
            "[hstu-attn] bwd_dkdv packed-f32 epilogue ARMED: epilogue=reuse_pk "
            "(DLRM_HSTU_FP8_BWD_PK)")
    # Workgroup grid order of bwd_dkdv (roadmap R15).
    # $DLRM_HSTU_FP8_BWD_GORDER = xcd | xcdn; absent = kernel default "xcd",
    # i.e. the identical compiled kernel. "xcdn" puts n-blocks fastest inside
    # an XCD. xcd was chosen by lab microbench, confirmed by in-model A/B (the
    # FlyDSL backward ship, -10.77% step time); xcdn's lab result changed sign
    # between benchmark inputs, so it needs an in-model A/B (bench.sh ARM_ENV)
    # before any default change. Grid order only: same math; the bf16p dq
    # atomics are already order-nondeterministic.
    _gorder = os.environ.get("DLRM_HSTU_FP8_BWD_GORDER", "").strip().lower()
    if _gorder:
        if _gorder not in ("xcd", "xcdn"):
            raise ValueError(
                f"DLRM_HSTU_FP8_BWD_GORDER={_gorder!r}: expected xcd|xcdn"
            )
        config["gorder"] = _gorder
        if ("gorder", _gorder) not in _logged_layouts:
            _logged_layouts.add(("gorder", _gorder))
            logger.warning(
                "[hstu-attn] bwd_dkdv grid order ARMED: gorder=%s "
                "(DLRM_HSTU_FP8_BWD_GORDER)", _gorder,
            )
    # R15 iter11 intra-pair lookahead ($DLRM_HSTU_FP8_BWD_PIPE = 1..6,
    # default OFF).  dq-split C8 kernel only: both m tiles of an mu2 pair are
    # staged in LDS up front and tile B's qk/dp MFMAs share a scheduling
    # region with tile A's silu/ds VALU chain.  Same math on every element
    # (dk/dv/dq-workspace bitwise equal to the unpiped split kernel).  Unset
    # (or a non-split call) leaves the config dict, hence the compile cache
    # key and the kernel, unchanged.
    _pipe = os.environ.get("DLRM_HSTU_FP8_BWD_PIPE", "").strip()
    if _pipe and _pipe != "0" and config.get("dqsplit"):
        if _pipe not in ("1", "2", "3", "4", "5", "6"):
            raise ValueError(
                f"DLRM_HSTU_FP8_BWD_PIPE={_pipe!r}: expected 1..6")
        config["pipe"] = int(_pipe)
        if ("pipe", _pipe) not in _logged_layouts:
            _logged_layouts.add(("pipe", _pipe))
            logger.warning(
                "[hstu-attn] bwd_dkdv intra-pair lookahead ARMED: pipe=%s "
                "(DLRM_HSTU_FP8_BWD_PIPE; dq-split mu2 pairs staged together)",
                _pipe)
    # R15 VALU diet ($DLRM_HSTU_FP8_BWD_VALU, default OFF).  Comma list of
    # undef|rfl|ilp ("1" = undef,rfl).  All bitwise neutral; see
    # compile_hstu_bwd_dkdv(valu=).  Unset = no config key = the identical
    # kernel and compile-cache key.  dq-split C8 calls only (the measured
    # kernel): on the non-C8 bf16 body "ilp"/"undef" trip an LLVM IGroupLP
    # pre-RA assertion (iter17 verify_integration), so other calls ignore it.
    _valu_env = os.environ.get("DLRM_HSTU_FP8_BWD_VALU", "").strip().lower()
    _vf2_on = False
    if (_valu_env and _valu_env not in ("0", "off", "false", "no")
            and config.get("dqsplit")):
        _vs = set()
        _tsc_stray = False
        for _t in _valu_env.split(","):
            _t = _t.strip()
            if _t in ("1", "on", "true", "yes"):
                _vs |= {"undef", "rfl"}
            elif _t in ("undef", "rfl", "ilp", "nobr", "h16", "vf", "vf2",
                        "msat", "vf3", "vf3s", "vf3p", "vf4"):
                _vs.add(_t)
            elif _t == "tsc":
                # R15 iter75 tscv2: never armed from this list (only
                # DLRM_HSTU_FP8_LAB_BWD_SPF=1 adds it, after its per-call
                # grid check, in _bwd_spf_tokens below).
                _tsc_stray = True
            elif _t:
                raise ValueError(
                    f"DLRM_HSTU_FP8_BWD_VALU={_valu_env!r}: expected a comma "
                    "list of 1|undef|rfl|ilp|nobr|h16|vf|vf2|msat")
        if _tsc_stray:
            _bwd_spf_strip_log("DLRM_HSTU_FP8_BWD_VALU", _valu_env)
        if _epilogue != "reuse_pk":
            _vs.discard("ilp")
        if "vf2" in _vs and "ilp" in _vs and config.get("f8b") == "all":
            # iter22 f16act vf2: ds carries alpha*qkv_scale; re-centre the
            # e4m3 ds range with a pow2 and drop alpha*s from the dq scale.
            config["mu2_ssc"] = config["mu2_ssc"] * 2.0 ** int(round(
                math.log2(alpha * float(qkv_scale) ** 2)))
            config["mu2_dsc"] = config["mu2_dsc"] * 2.0 ** int(round(
                math.log2(float(qkv_scale))))
            _vf2_on = True
        else:
            _vs.discard("vf2")
        # item (49) DLRM_HSTU_FP8_BWD_VF3P (guard 0 < qkv_scale <= 0.25)
        _vs = _bwd_vf3p_tokens(_vs, qkv_scale=qkv_scale)
        # gin item (80) VF4 DLRM_HSTU_FP8_BWD_VF4 (code default OFF, declared
        # precision change; config unchanged: the kernel re-centres the
        # silu' cast scale from the runtime qkv_scale).
        _vs = _bwd_vf4_tokens(_vs, config, alpha, qkv_scale)
        if _vs:
            config["valu"] = ",".join(sorted(_vs))
        if ("valu", config.get("valu")) not in _logged_layouts:
            _logged_layouts.add(("valu", config.get("valu")))
            logger.warning(
                "[hstu-attn] bwd_dkdv VALU diet ARMED: valu=%s "
                "(DLRM_HSTU_FP8_BWD_VALU=%s)", config.get("valu", "<none>"),
                _valu_env)
    elif config.get("dqsplit"):
        _bwd_vf3p_tokens(set(), "DLRM_HSTU_FP8_BWD_VALU=%s is off (vf3p is a "
                         "VALU diet token)" % (_valu_env or "<unset>"))
    # R15 iter17r3 B1 m-split ($DLRM_HSTU_FP8_BWD_MSPLIT = 1|2, default
    # OFF).  dq-split C8 calls only: 8 waves/WG, waves w and w+4 share 32 n
    # rows, m-split score/act + d-split out-gemm, 2 waves/SIMD.  Bitwise
    # dq/dk/dv vs the stack (same MFMA operands and pair order); see
    # compile_hstu_bwd_dkdv(msplit=).  It replaces the pair body, so any
    # DLRM_HSTU_FP8_BWD_PIPE is dropped for these calls.  Unset = no config
    # key = the identical kernel and compile-cache key.
    _msplit = os.environ.get("DLRM_HSTU_FP8_BWD_MSPLIT", "").strip()
    if _msplit and _msplit != "0" and config.get("dqsplit"):
        if _msplit not in ("1", "2"):
            raise ValueError(
                f"DLRM_HSTU_FP8_BWD_MSPLIT={_msplit!r}: expected 1|2")
        config.pop("pipe", None)
        config["msplit"] = int(_msplit)
        if ("msplit", _msplit) not in _logged_layouts:
            _logged_layouts.add(("msplit", _msplit))
            logger.warning(
                "[hstu-attn] bwd_dkdv m-split ARMED: msplit=%s "
                "(DLRM_HSTU_FP8_BWD_MSPLIT; 8 waves, pipe ignored)", _msplit)
        # R15 iter19 ms1tune: $DLRM_HSTU_FP8_BWD_MSPLIT_SCHED = comma tokens
        # (so, anti, prio, iglpN, noiglp, sgK); unset = no config key.
        _mss = os.environ.get("DLRM_HSTU_FP8_BWD_MSPLIT_SCHED", "").strip()
        if _mss:
            config["ms_sched"] = ",".join(t.strip() for t in _mss.split(",")
                                          if t.strip())
    # R15 iter24 ws track: $DLRM_HSTU_FP8_BWD_WS_AUX = cache-policy aux bits
    # of the dq-split ds workspace stores (0/unset = off = no config key).
    _ws_aux = int(os.environ.get("DLRM_HSTU_FP8_BWD_WS_AUX", "0") or 0)
    if _ws_aux and config.get("dqsplit"):
        config["ws_aux"] = _ws_aux
    # $DLRM_HSTU_FP8_BWD_WS_LAYOUT=c64: contiguous 64 B ws store halves (the
    # dq-split kernel reads through the matching permutation).
    _ws_lay = os.environ.get("DLRM_HSTU_FP8_BWD_WS_LAYOUT", "").strip().lower()
    if _ws_lay not in ("", "0", "off", "c64", "l128", "l128t"):
        raise ValueError(
            f"DLRM_HSTU_FP8_BWD_WS_LAYOUT={_ws_lay!r}: use c64|l128|l128t")
    if _ws_lay in ("c64", "l128", "l128t") and config.get("dqsplit"):
        config["ws_" + _ws_lay] = True
    if config.get("dqsplit") and (config.get("ws_aux") or any(
            config.get("ws_" + x) for x in ("c64", "l128", "l128t"))):
        if ("ws_store", _ws_lay, _ws_aux) not in _logged_layouts:
            _logged_layouts.add(("ws_store", _ws_lay, _ws_aux))
            logger.warning(
                "[hstu-attn] bwd dq SPLIT ws store ARMED: layout=%s aux=%s "
                "(DLRM_HSTU_FP8_BWD_WS_LAYOUT / DLRM_HSTU_FP8_BWD_WS_AUX)",
                _ws_lay or "off", _ws_aux)
    # R15 iter75 tscv2 (DLRM_HSTU_FP8_LAB_BWD_SPF, LAB, default off, not in
    # gin): the backward SPF fold, VALU token "tsc", decided per call on the
    # final config (after msplit).  Off: no change (a stray "tsc" in
    # config["valu"] is stripped); "1": grid + range check, body check, token.
    _bwd_spf_tokens(config, kernel, fp8_mfma, input_fp8 and not q_bf16,
                    alpha, qkv_scale)
    # R15 M-E1 (DLRM_HSTU_FP8_BWD_DKDV_E5M2, default OFF): with a
    # DkdvE5m2Sink (hstu_dkdv_e5m2) the dk/dv epilogue writes the e5m2 dV/dK
    # columns of the layer's dz8 + amax/colsum partials instead of BF16
    # dk/dv.  No sink = no config key = the identical kernel and
    # compile-cache key.
    _e8 = None
    if dkdv_fp8_sink is not None:
        _why = dkdv_fp8_sink.decline_reason(config, heads, dim, q.shape[0])
        if _why is None:
            config["e8emit"] = dkdv_fp8_sink.mode
            _e8 = dkdv_fp8_sink
        else:
            dkdv_fp8_sink.declined = _why
    fn = kernel.compile_hstu_bwd_dkdv(
        heads,
        dim,
        alpha,
        epilogue=_epilogue,
        rs_in=rs_in,
        rs_out=rs_out,
        input_fp8=input_fp8,
        q_bf16=q_bf16,
        rs_q=rs_q,
        fp8_mfma=fp8_mfma,
        **config,
    )
    # Logged once per distinct layout: whether the wrapper tax is being paid is
    # invisible in the numerics and costs ~15 ms/step/rank when it is, so it
    # should not take a profile to find out.
    _e1_period = int(
        os.environ.get("DLRM_HSTU_FP8_BWD_CENSUS")
        or os.environ.get("DLRM_E1_BWD_CENSUS", "0")
        or 0
    )
    if _e1_period:
        global _e1_bwd_calls
        _e1_bwd_calls += 1
        vk = (
            "e4m3" if (input_fp8 and not q_bf16)
            else ("bf16q_e4m3kv" if q_bf16 else "bf16"),
            fp8_mfma or "bf16mfma",
        )
        _e1_bwd_census[vk] = _e1_bwd_census.get(vk, 0) + 1
        if _e1_bwd_calls % _e1_period == 0:
            logger.warning(
                "[e1census] calls=%d %s",
                _e1_bwd_calls,
                sorted(_e1_bwd_census.items()),
            )

    key = (rs_in, rs_q, rs_out, input_fp8, q_bf16, heads, dim, out is not None,
           fp8_mfma, do_scale is not None, bool(config.get("do_blk")),
           _epilogue)
    if key not in _logged_layouts:
        _logged_layouts.add(key)
        logger.warning(
            "[e1] bwd_dkdv layout qk_mfma=%r input_fp8=%s q_bf16=%s "
            "rs_in=%d rs_q=%d rs_out=%d heads=%d dim=%d epilogue=%s",
            fp8_mfma or "bf16", input_fp8, q_bf16, rs_in, rs_q, rs_out,
            heads, dim, _epilogue,
        )
        if do_scale is not None:
            logger.warning(
                "[hstu-attn] FULL-FP8 backward ARMED: f8b=all mu2=k128 "
                "do_dyn=%d do_blk=%d ssc=2^%d dsc=2^%d ds8_pad=%d sat=%d "
                "(dO e4m3, runtime pow2 scale per (seq, head, 32-token MX "
                "group); N=%d)",
                # do_dyn is always 0 here; the field stays for log-grep compat.
                int(bool(config.get("do_dyn"))), int(bool(config.get("do_blk"))),
                int(np.log2(config["mu2_ssc"])), int(np.log2(config["mu2_dsc"])),
                config["ds8_pad"], int(config["f8_sat"]), int(N),
            )
        logger.warning(
            "[hstu-attn] FlyDSL backward reading %s at row stride %d "
            "(Q stride %d) and writing BF16 gradients at row stride %d "
            "(heads=%d dim=%d)",
            "BF16 Q + E4M3 K/V"
            if q_bf16
            else ("E4M3 Q/K/V" if input_fp8 else "BF16 Q/K/V"),
            rs_in,
            rs_q,
            rs_out,
            heads,
            dim,
        )

    # The kernel reads seq_offsets through a 32-bit buffer descriptor;
    # production carries it as int64.  With DLRM_HSTU_FP8_ATTN_INTGLUE=1 a
    # dq-split call gets cu from hstu_dq_split.build_workspace_intglue below.
    _intglue = bool(config.get("dqsplit")) and \
        _attn_intglue_reason(seq_offsets) is None
    cu = None if _intglue else seq_offsets.to(torch.int32)
    batch = seq_offsets.numel() - 1
    # N, not seq_offsets.diff().max(): the caller's max_seq_len already bounds
    # every sequence, and reading the real max costs a device-to-host sync in
    # the deepest part of the backward queue (profiled at 31.7 ms a call). The
    # two agreed exactly on 1,600 audited calls across 8 ranks -- production
    # derives N as max_uih_len + max_num_candidates, and with
    # max_num_candidates = 1 the pair can differ by at most one token anyway,
    # which moves n_blocks only when N lands one past a block boundary. A larger
    # N would over-launch, never under-launch, so the failure mode is bounded
    # waste rather than a wrong answer.
    n_blocks = int((int(N) + fn.BN - 1) // fn.BN)

    _dqs = bool(config.get("dqsplit"))
    if _dqs_req and not _dqs:
        dq.zero_()
    dq_slot = dq
    if _dqs:
        # R15 MSPLIT_SCHED dz/dzr (drop the ws stores of padding rows/tiles):
        # the kernel pushes a dead store past the per-(b, h) descriptor end
        # by +2^31, which needs nb*(nb+1)*8 KiB < 2^31, i.e. nb <= 511 blocks
        # of 128 rows.
        _ms_tok = set(str(config.get("ms_sched", "")).split(","))
        if {"dz", "dzr"} & _ms_tok and int(N) > 65408:
            raise AssertionError(
                f"DLRM_HSTU_FP8_BWD_MSPLIT_SCHED dz/dzr: max seq len N={int(N)}"
                " > 65408 (511 blocks of 128); the +2^31 ws store-drop offset"
                " would wrap. Remove dz/dzr from MSPLIT_SCHED for this N.")
        from generative_recommenders.ops.flydsl import hstu_dq_split
        if _intglue:
            ws, cu = hstu_dq_split.build_workspace_intglue(
                seq_offsets, heads, q.shape[0], N,
                check=os.environ.get("DLRM_HSTU_FP8_ATTN_INTGLUE_CHECK",
                                     "").strip() == "1")
        else:
            ws = hstu_dq_split.build_workspace(seq_offsets, heads, q.shape[0],
                                               N)
        dq_slot = ws.view(torch.bfloat16)
        if "dq_split" not in _logged_layouts:
            _logged_layouts.add("dq_split")
            logger.warning(
                "[hstu-attn] bwd dq SPLIT ARMED: e4m3 ds workspace %.2f GB + "
                "f32 dq kernel, no dq atomics/zero-fill "
                "(DLRM_HSTU_FP8_BWD_DQ_SPLIT)", ws.numel() / 1e9)
    # The saved q/k/v still carry requires_grad, and flydsl's dlpack export
    # refuses those.
    fn(
        q.detach(),
        k.detach(),
        v.detach(),
        dout.detach(),
        dk,
        dv,
        dq_slot,
        cu,
        batch,
        n_blocks,
        1.0 / float(N),
        qkv_scale,
        **({"do_scale": do_scale} if do_scale is not None else {}),
        **(_e8.launch_kwargs(seq_offsets, q.shape[0], heads, dim)
           if _e8 is not None else {}),
    )
    if _dqs:
        (hstu_dq_split.dq_from_ds if dq_fp8_sink is None else dq_fp8_sink)(
            ws, k.detach(), do_scale, seq_offsets, dq, N,
            alpha * float(config.get("dq_scale", 1.0)) / config["mu2_dsc"]
            * float(qkv_scale) * (float(qkv_scale) if _vf2_on else 1.0),
            rs_in, rs_out,
            fp8_dot=os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_DOT", "fp8")
            != "bf16",
            bmq=int(os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_BM", "128")),
            num_warps=int(os.environ.get("DLRM_HSTU_FP8_BWD_DQ_SPLIT_NW", "8")),
            **{"stages": int(os.environ.get(
                "DLRM_HSTU_FP8_BWD_DQ_SPLIT_STAGES", "2")),
               **_dq_split_opt()},
            order=_dq_split_order(int(N)),
            sched=_dq_split_sched(),
            **({"ws_c64": True} if config.get("ws_c64") else {}),
            **_dq_split_ld())
        del ws, dq_slot
    if do_scale is not None and _full_fp8_stats_left():
        _full_fp8_stats(q, k, v, dout, do_scale, seq_offsets, N, alpha,
                        qkv_scale, config, dq, dk, dv, dout_orig)
    if _bwd_dump_dir():
        _bwd_dump(q, k, v, dout_orig if do_scale is not None else dout,
                  do_scale, seq_offsets, N, alpha, qkv_scale, dq, dk, dv,
                  config)
    if dq.dtype is not torch.bfloat16:
        dq = dq.to(torch.bfloat16)
    return dq, dk, dv


def flydsl_selected() -> bool:
    """True when $HSTU_ATTN_BWD asks for FlyDSL. Says nothing about whether it can run."""
    return (
        os.environ.get("HSTU_ATTN_BWD", DEFAULT_BWD_BACKEND).strip().lower()
        or DEFAULT_BWD_BACKEND
    ) == "flydsl"


def flydsl_enabled() -> bool:
    """True when FlyDSL is selected; raises if it is selected but unusable.

    Selection and usability are separate questions on purpose -- see the
    module docstring. This never returns False for a host problem.
    """
    if not flydsl_selected():
        return False
    require_flydsl()
    return True


def _log_shape_fallback(reason: str) -> None:
    """Log the first fallback of each distinct reason, then stay quiet."""
    if reason in _logged_fallbacks:
        return
    _logged_fallbacks.add(reason)
    logger.warning(
        "[hstu-attn] FlyDSL backward cannot serve this shape (%s); using the "
        "Triton backward for these calls. This is a per-shape fallback, not a "
        "host problem, and is logged once per reason.",
        reason,
    )


def try_flydsl_attention_bwd(
    *, dout, q, k, v, dq, dk, dv, seq_offsets, num_targets, N, alpha,
    max_attn_len, contextual_seq_len, num_softmax_heads, qkv_scale=1.0,
    dq_fp8_sink=None, dkdv_fp8_sink=None,
) -> bool:
    """Write the attention backward into dq/dk/dv with FlyDSL.

    Returns False without touching the outputs when FlyDSL is not selected or
    cannot serve this shape, so the caller runs its normal Triton backward.

    Mirrors `triton_hstu_attention_bwd`'s convention of writing into caller-owned
    dq/dk/dv rather than returning them: on the fused preprocess+attention path
    those are views into a single `duvqk` buffer that the rest of the backward
    consumes whole, so the gradients have to land in that buffer.

    The kernel is handed those views directly, and reads q/k/v out of `uvqk`
    where they already sit. It used to require packed tensors, which cost three
    pack copies in and three copy-backs out per call -- with a zero-fill for the
    dq accumulator, seven extra kernels per layer. Passing the row stride
    instead costs +0.16% inside the kernel and leaves only dq's zero-fill.
    Layouts that cannot share one row stride still take the copy path.
    """
    if not flydsl_enabled():
        # R15 iter75 tscv2: DLRM_HSTU_FP8_LAB_BWD_SPF=1 fails closed here.
        _bwd_spf_refuse_unserved(q, k, "FlyDSL backward not selected")
        hstu_do8_registry.discard(dout, "flydsl_not_selected")
        if dkdv_fp8_sink is not None:
            dkdv_fp8_sink.declined = "FlyDSL backward not selected"
        return False
    reason = unsupported_shape_reason(
        q, k, v, dout, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    )
    if reason is not None:
        _bwd_spf_refuse_unserved(q, k, f"FlyDSL shape fallback: {reason}")
        _log_shape_fallback(reason)
        hstu_do8_registry.discard(dout, "shape_fallback")
        if dkdv_fp8_sink is not None:
            dkdv_fp8_sink.declined = f"FlyDSL shape fallback: {reason}"
        return False
    global _backward_calls
    out_dq, out_dk, out_dv = flydsl_hstu_attention_bwd(
        dout,
        q,
        k,
        v,
        seq_offsets,
        N,
        alpha,
        out=(dq, dk, dv),
        qkv_scale=qkv_scale,
        **({} if dq_fp8_sink is None else {"dq_fp8_sink": dq_fp8_sink}),
        # R15 M-E1 (default None: the call above is unchanged).
        **({} if dkdv_fp8_sink is None else {"dkdv_fp8_sink": dkdv_fp8_sink}),
    )
    # R15 M-E1: an engaged sink leaves the BF16 dV/dK unwritten, so neither
    # the reference check nor a packed-path copy-back may read them.
    _e8_only = (
        dkdv_fp8_sink is not None
        and dkdv_fp8_sink.declined is None
        and dkdv_fp8_sink.filled
    )

    _backward_calls += 1
    if _check_due(_backward_calls) and _e8_only:
        logger.warning(
            "[hstu-attn] HSTU_ATTN_BWD_CHECK skipped on a dkdv e5m2 sink "
            "call (BF16 dV/dK not written)")
    elif _check_due(_backward_calls):
        # sort_by_length_indices is a scheduling hint only, so the reference does
        # not need the caller's; enable_tma likewise does not change the result.
        _log_reference_snr(
            out_dq, out_dk, out_dv,
            _triton_bwd(
                dout, q, k, v, seq_offsets, num_targets, N, alpha,
                max_attn_len, contextual_seq_len, None, False,
            ),
            q.shape,
        )

    # No-ops when the kernel wrote in place; real copies only when the layout
    # forced the packed path and these are distinct tensors.
    for dst, src in ((dq, out_dq), (dk, out_dk), (dv, out_dv)):
        if _e8_only and dst is not dq:
            continue
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src)
    return True


# ---------------------------------------------------------------------------
# Forward
#
# Selected independently of the backward ($HSTU_ATTN_FWD vs $HSTU_ATTN_BWD),
# because the two kernels are genuinely decoupled: HSTU has no softmax, so there
# is no lse and no rowsum(dO*O) for the forward to hand the backward. Either can
# run while the other stays on Triton, which is also what makes an A/B of one
# half meaningful.
# ---------------------------------------------------------------------------


def _load_fwd_kernel():
    """Return the forward kernel module, or None if it could not be imported."""
    global _fwd_kernel, _fwd_kernel_error
    if _fwd_kernel is not None or _fwd_kernel_error is not None:
        return _fwd_kernel
    try:
        from generative_recommenders.ops.flydsl import hstu_fwd

        _fwd_kernel = hstu_fwd
    except BaseException as e:  # noqa: BLE001 - report it, do not mask it
        _fwd_kernel_error = e
    return _fwd_kernel


def require_flydsl_fwd() -> None:
    """Raise unless the FlyDSL forward can run on this host."""
    global _fwd_host_ok
    if _fwd_host_ok:
        return
    reason = unsupported_hardware_reason()
    if reason is None and _load_fwd_kernel() is None:
        reason = (
            f"flydsl forward import failed "
            f"({type(_fwd_kernel_error).__name__}: {_fwd_kernel_error})"
        )
    fly_arch = ""
    if reason is None:
        fly_arch, reason = _flydsl_arch_check(_fwd_kernel)
    if reason is None:
        _fwd_host_ok = True
        logger.info(
            "[hstu-attn] forward kernel: flydsl on %s flydsl_arch=%s pin=%s",
            _device_arch(), fly_arch, FLYDSL_ARCH_PIN[1],
        )
        return
    raise FlyDSLUnavailableError(
        f"the FlyDSL HSTU attention forward is selected but cannot run here: "
        f"{reason}. It is the DEFAULT, so raising is deliberate -- set "
        f"HSTU_ATTN_FWD=triton to run the Triton forward instead."
    )


def flydsl_fwd_selected() -> bool:
    """True when $HSTU_ATTN_FWD asks for FlyDSL. Says nothing about usability."""
    return (
        os.environ.get("HSTU_ATTN_FWD", DEFAULT_FWD_BACKEND).strip().lower()
        or DEFAULT_FWD_BACKEND
    ) == "flydsl"


def flydsl_fwd_enabled() -> bool:
    """True when the FlyDSL forward is selected; raises if selected but unusable."""
    if not flydsl_fwd_selected():
        return False
    require_flydsl_fwd()
    return True


def unsupported_fwd_shape_reason(
    q, k, v, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0,
) -> Optional[str]:
    """Why the forward kernel cannot serve this call, or None if it can.

    Same specialization set as the backward, minus dO (the forward has none) and
    plus the v head dim, because the forward's out-gemm writes D=128 columns.
    """
    if max_attn_len:
        return f"max_attn_len={max_attn_len} (kernel is plain causal)"
    if contextual_seq_len:
        return f"contextual_seq_len={contextual_seq_len} (kernel assumes 0)"
    if num_softmax_heads:
        return f"num_softmax_heads={num_softmax_heads} (kernel assumes 0)"
    if q.dim() != 3 or q.shape[-1] != SUPPORTED_DIM:
        return f"qk dim {tuple(q.shape)} (kernel is written around dim {SUPPORTED_DIM})"
    if v.shape[-1] != SUPPORTED_DIM:
        return f"v dim {v.shape[-1]} (kernel is written around dim {SUPPORTED_DIM})"
    if any(t.dtype is not torch.bfloat16 for t in (q, k, v)):
        return (
            "dtype "
            f"{tuple(str(t.dtype).replace('torch.', '') for t in (q, k, v))}"
            " (kernel has an mfma path for bf16 only)"
        )
    if num_targets is not None:
        n = _max_num_targets(num_targets)
        if n > 1:
            return f"num_targets max={n} (target-aware masking is identity only at <= 1)"
    return None


def _fwd_seq_offsets_i32(seq_offsets: torch.Tensor) -> torch.Tensor:
    """int32 seq offsets for fwd_kernel_0, padded for its XCD-pinned grid.

    The forward's grid.z is cdiv(batch, 8), so for batch % 8 != 0 the padded
    workgroup slots (batch index >= batch) read seq_offsets[b] and [b + 1]
    past the end of the tensor through a max_size (unchecked) descriptor and
    run with whatever stale bytes follow it: they overwrite real output rows
    or write past `out` (memory access fault). Padding with the total token
    count gives every padded slot seq_len 0. A no-op (no extra kernel) when
    batch % 8 == 0, i.e. always in production (B = 1024, drop_last).
    """
    cu = seq_offsets.to(torch.int32)
    batch = cu.numel() - 1
    padded = -(-batch // 8) * 8
    if padded == batch:
        return cu
    return torch.cat([cu, cu[-1:].expand(padded - batch)])


def flydsl_hstu_attention_fwd(q, k, v, seq_offsets, N, alpha) -> torch.Tensor:
    """The two-gemm forward: out = silu(alpha * q @ k.T)/N masked causally, @ v.

    Returns a freshly allocated packed [tokens, heads, dim] bf16 tensor, matching
    what `triton_hstu_attention_fwd` returns (`torch.empty_like(v)`, which is
    contiguous because v is a non-dense slice of uvqk).

    Unlike the backward there is no accumulator to zero and no atomics: out[m]
    sums over n <= m only, so the m-outer grid gives each workgroup exclusive
    ownership of its output rows. Every token row inside a sequence is written,
    so the output needs no pre-fill.
    """
    kernel = _load_fwd_kernel()
    heads, dim = q.shape[1], q.shape[2]

    # Read q/k/v where they already live. On the fused preprocess+attention path
    # they are column slices of one uvqk buffer at stride 2048, and packing them
    # would cost three full copies of the attention input per layer per call --
    # the same class of wrapper tax the backward measured at ~15 ms/step/rank,
    # which is enough to eat this kernel's entire win.
    rs = _common_row_stride((q, k, v), heads, dim)
    if rs is None:
        q, k, v = (t.contiguous() for t in (q, k, v))
    fn = kernel.compile_hstu_fwd(heads, dim, alpha, rs=(rs or 0),
                                 **kernel.SHIPPED)

    key = (rs, heads, dim)
    if key not in _logged_fwd_layouts:
        _logged_fwd_layouts.add(key)
        _maybe_dump_lengths(seq_offsets)
        # tokens and N are logged because the grid is sized by N (see below) while
        # the work is set by the real lengths, so the ratio of the two is how much
        # of the launch is doing nothing -- and it is not otherwise visible in
        # either a profile or the numerics.
        logger.warning(
            "[hstu-attn] FlyDSL forward %s (heads=%d dim=%d tokens=%d N=%d "
            "mean_len=%.0f)",
            f"reading q/k/v in place at row stride {rs}" if rs
            else "packing q/k/v to contiguous (row stride %d)" % (heads * dim),
            heads, dim, q.shape[0], int(N),
            q.shape[0] / max(1, seq_offsets.numel() - 1),
        )

    # The kernel reads seq_offsets through a 32-bit buffer descriptor;
    # production carries it as int64.
    cu = _fwd_seq_offsets_i32(seq_offsets)
    batch = seq_offsets.numel() - 1
    # N rather than the real max, for the same reason as the backward: reading
    # the true max costs a device-to-host sync deep in the queue, and N already
    # bounds every sequence. Over-launching only wastes workgroups that exit
    # immediately -- it cannot produce a wrong answer. Note this is also why the
    # kernel-level benchmark must derive N the way production does: N sizes this
    # grid, so a padded N would make the comparison meaningless.
    m_blocks = int((int(N) + fn.BM - 1) // fn.BM)

    out = torch.empty((q.shape[0], heads, dim), dtype=q.dtype, device=q.device)
    # q/k/v may still carry requires_grad, and flydsl's dlpack export refuses
    # those.
    fn(q.detach(), k.detach(), v.detach(), out, cu, batch, m_blocks,
       1.0 / float(N))
    return out


def _log_fwd_shape_fallback(reason: str) -> None:
    """Log the first forward fallback of each distinct reason, then stay quiet."""
    if reason in _logged_fwd_fallbacks:
        return
    _logged_fwd_fallbacks.add(reason)
    logger.warning(
        "[hstu-attn] FlyDSL forward cannot serve this shape (%s); using the "
        "Triton forward for these calls. This is a per-shape fallback, not a "
        "host problem, and is logged once per reason.",
        reason,
    )


def try_flydsl_attention_fwd(
    *, q, k, v, seq_offsets, num_targets, N, alpha, max_attn_len,
    contextual_seq_len, num_softmax_heads,
) -> Optional[torch.Tensor]:
    """The attention forward output from FlyDSL, or None to use Triton's.

    Returns None -- without allocating anything -- when FlyDSL is not selected or
    cannot serve this shape, so the caller runs its normal Triton forward.

    Returns a tensor rather than writing into a caller buffer, because that is
    what `triton_hstu_attention_fwd` does: attention's output is its own
    allocation, not a slice of uvqk.
    """
    # e4m3 q/k/v take try_flydsl_attention_fwd_fp8, not this bf16 kernel.
    if os.environ.get("DLRM_HSTU_FP8_FWD", "0") == "1":
        return None
    if not flydsl_fwd_enabled():
        return None
    reason = unsupported_fwd_shape_reason(
        q, k, v, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    )
    if reason is not None:
        _log_fwd_shape_fallback(reason)
        return None
    out = flydsl_hstu_attention_fwd(q, k, v, seq_offsets, N, alpha)

    global _forward_calls
    _forward_calls += 1
    period = int(os.environ.get("HSTU_ATTN_FWD_CHECK", "0") or 0)
    if period > 0 and _forward_calls % period == 0:
        # sort_by_length_indices is a scheduling hint only and does not change
        # the result, so the reference does not need the caller's.
        ref = triton_hstu_attention_fwd(
            N=N, alpha=alpha, q=q, k=k, v=v, seq_offsets=seq_offsets,
            num_targets=num_targets, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=None, enable_tma=False,
            num_softmax_heads=0,
        )
        logger.info(
            "[hstu-fwd-check] call=%d shape=%s snr_db out=%.1f",
            _forward_calls, tuple(q.shape), _snr_db(ref, out),
        )
    return out


# ---------------------------------------------------------------------------
# FP8 forward (Track I1)
#
# The FP8 arm's fused preprocess hands attention q/k/v that are ALREADY e4m3
# column slices of one vqk buffer, plus ONE per-tensor scale. Today that goes to
# the Triton e4m3 forward. This is the FlyDSL alternative for the same contract.
#
# Selected by the same $HSTU_ATTN_FWD=flydsl that picks the bf16 FlyDSL
# forward, once $DLRM_HSTU_FP8_FWD is on. That combination used to raise;
# it now compiles this specialization (fp8_mfma="k128", fp8_v=True) plus the
# fused-silu epilogue (epi_fold / mask_split / iglp0). $HSTU_ATTN_FWD=triton
# with FP8 on keeps the certified Triton e4m3 forward.
#
# CONFIG. `k128` at waves=2, which is `hstu_fwd.SHIPPED` plus the two fp8 flags.
# waves=1 measures ~4% faster standalone and is NOT used: that arm sits at 248 of
# 256 VGPR, eight registers from dropping to one wave per SIMD and off a ~4x
# cliff. waves=2 is the robust point.
#
# fp8_v=True is forced by the contract, not picked by tuning, and it is not an
# extra approximation here even though it is one in the standalone bench.
# Production's V is already e4m3 -- it comes out of the same vqk GEMM as q and
# k, and the shape check below admits only e4m3 q, k and v -- and that e4m3 V
# is exactly what the incumbent Triton e4m3 forward reads.
# The SNR figures are a HISTORICAL equivalence check, not a tuning input: on a
# ragged length pool taken from a benchmark-run length dump (bench/
# fwd_baseline.py, 3 reps; neither the pool nor the script is in this tree),
# SNR_e2e read 26.99 dB for both this arm and triton-fp8, i.e. identical, while
# fp8_v=False read 28.75 dB because it is a DIFFERENT (more accurate) function
# than the one production runs. Matching the incumbent is the point; the
# kernel-vs-oracle SNR was 52.55 dB either way. No value, threshold or choice
# is selected from that check.
# ---------------------------------------------------------------------------

_FP8_FWD_CFG = {"fp8_mfma": "k128", "fp8_v": True}

# Silu-epilogue VALU reductions (hstu_fwd.py's `epi_fold` and `mask_split`)
# are part of this kernel, not a second env var. SHIPPED is iglp1, and
# mask_split is INCOMPATIBLE with it -- the n-loop split trips
# AMDGPUIGroupLP.cpp:2087 -- so the config also moves the scheduling hint to
# iglp0. That is a CONFOUND and it is on the record: the shippable stack is
# (epi_fold, mask_split, iglp0) against (--, --, iglp1). The BASELINE was
# swept across iglp1/iglp0/none at both the bench shape and production's
# (bench/s1_shapes.sh): the whole sched spread is 1.2% at N=4096 and 1.3% at
# N=2994, against a 17% candidate win, so the sched move is not the source.
_FP8_FWD_E2_CFG = {"epi_fold": True, "mask_split": True, "sched": "iglp0"}

# DLRM_HSTU_FP8_FWD_PIPE (X4C19; read per call; code default OFF, gin item
# (26)(a) binds it on): a
# 128-row m-block over 4 waves (MT=2 per wave as before, half the K/V staging
# per output row), the next n-iteration's K/V global loads carried as loop
# iter_args (`kv_prefetch`), and the out-gemm P fragment built from registers
# instead of a p_lds round trip (`p_direct`). Output bit-identical to the
# shipped config for finite V: BM128 adds zero-weight (p=0) MFMAs over V
# rows m0+64..m0+127 for the first half of each 128-row block, so a
# non-finite V value can reach up to 127 (not 63) causally earlier rows of
# its m-block; finite inputs give identical bits. Kernel name unchanged
# (fwd_kernel_0). No causal tile is skipped. DLRM_HSTU_FP8_FWD_PIPE=0 (or
# unset) restores the shipped compile kwargs exactly.
# PROVENANCE: selected by in-model A/B of the benchmark run (bitwise,
# speed-only choice): ship11 ladder rung R2->R3 = gin item (26), this config
# together with DELTA_FWD_BM16 -- equal-work rate +1.44% early (job 64883)
# and +1.42% late (job 64884); in the late traced pair fwd_kernel_0 ran
# 16.26 -> 13.91 ms/rank-step, shorter on 32/32 rank-steps -- and the Arbor
# x4c19 iter-37 1n gate of the bundle that also held DELTA_BWD_DQACC (since
# dropped): job 63517, rate +2.78%, every traced step shorter.
# The iter-36 lab microbench on benchmark-run length pools (since removed
# from the tree) is historical and selects nothing.
_FP8_FWD_PIPE_CFG = {"BM": 128, "waves": 4, "kv_prefetch": True, "p_direct": True}
_fp8_fwd_pipe_logged = False


def fp8_fwd_pipe_enabled() -> bool:
    """$DLRM_HSTU_FP8_FWD_PIPE=1 (read per call; logs one arming line)."""
    global _fp8_fwd_pipe_logged
    on = os.environ.get("DLRM_HSTU_FP8_FWD_PIPE", "0") == "1"
    if on and not _fp8_fwd_pipe_logged:
        _fp8_fwd_pipe_logged = True
        print(
            "[hstu-fwd] DLRM_HSTU_FP8_FWD_PIPE=1 armed: BM128 waves4 "
            "kv_prefetch p_direct (bit-identical to the shipped BM64/waves2 for "
            "finite V)",
            flush=True,
        )
    return on


# DLRM_HSTU_FP8_FWD_MASK_ARITH (X4C19 iter41 f4; read per call, gin item
# (48)(d)): in the diagonal (masked) n-iterations of fwd_kernel_0 the causal
# mask is applied to the packed bf16 P fragment -- one v_and_b32 per bf16 pair,
# with per-lane pair masks from two packed-int16 ops per distinct n-tile/m-tile
# offset -- instead of an f32 compare + select per score element followed by a
# bf16 unpack/repack (hstu_fwd.py mask_arith=3). Every score element is still
# computed and the predicate stays a runtime per-lane value; only the
# instructions that evaluate and apply m >= n change. Bit-identical output for
# all inputs, including non-finite q/k/v (bf16(+0.0) is 0x0000). It needs
# p_direct, so it applies only together with DLRM_HSTU_FP8_FWD_PIPE=1; set
# alone it is ignored with one log line. Kernel name unchanged (fwd_kernel_0).
_FP8_FWD_MASK_ARITH = 3
_fp8_fwd_mask_arith_logged = set()


def fp8_fwd_mask_arith_enabled(pipe_on: bool) -> bool:
    """$DLRM_HSTU_FP8_FWD_MASK_ARITH=1 (read per call; logs one arming line)."""
    on = os.environ.get("DLRM_HSTU_FP8_FWD_MASK_ARITH", "0") == "1"
    if on and pipe_on not in _fp8_fwd_mask_arith_logged:
        _fp8_fwd_mask_arith_logged.add(pipe_on)
        print(
            f"[hstu-fwd] DLRM_HSTU_FP8_FWD_MASK_ARITH=1 armed: mask_arith="
            f"{_FP8_FWD_MASK_ARITH} (causal mask on the packed bf16 P fragment "
            f"in the diagonal n-iterations; bit-identical)" if pipe_on else
            "[hstu-fwd] DLRM_HSTU_FP8_FWD_MASK_ARITH=1 ignored: it needs "
            "DLRM_HSTU_FP8_FWD_PIPE=1 (p_direct)",
            flush=True,
        )
    return on and pipe_on


# DLRM_HSTU_FP8_FWD_PV (X4C19 iter46; read per call; code default OFF, NO gin
# binding): FP8 P@V with a K = 128 retile in fwd_kernel_0 (hstu_fwd.py
# pv="k128"). A DECLARED PRECISION CHANGE, gated on a ruling: the silu output
# P is rounded to e4m3 (static scale, saturating; DLRM_HSTU_FP8_FWD_PV_SCALE
# below) instead of bf16, V is
# consumed as raw e4m3, and P@V runs as one v_mfma_scale_f32_16x16x128_f8f6f4
# per 128 keys instead of four bf16 16x16x32. Output is NOT bit-identical to
# the bf16-P forward. It was built and verified on the FWD_PIPE config only,
# so it arms only when the active config IS that config
# (DLRM_HSTU_FP8_FWD_PIPE=1 and nothing else changed), alone or composed with
# DLRM_HSTU_FP8_FWD_MASK_ARITH=1 (X4C19 iter47: mask_arith=3 then applies the
# causal mask to the packed e4m3 P of the tail group, bit-identical to the
# pv=k128 select); otherwise it declines with one log line and the forward is
# unchanged. Kernel name unchanged (fwd_kernel_0). Unset (or anything but
# "1") = off, no log line.
_FP8_FWD_PV_CFG = {"pv": "k128", "pv_k": 3, "pv_split": True,
                   "pf_late": True, "pv_tg": True}
_fp8_fwd_pv_logged = set()

# DLRM_HSTU_FP8_FWD_PV_SCALE (X4C19 iter50; read per call, used only when
# DLRM_HSTU_FP8_FWD_PV=1 arms): the static scale s of the e4m3 P, P8 =
# e4m3(sat(P' * s)) (hstu_fwd.py pv_s). Unset or "" = 10.04, a NON-power-of-
# two scale folded into the silu chain's fma (it replaces the add; no extra
# hot-loop instruction). So since iter50, DLRM_HSTU_FP8_FWD_PV=1 ALONE means
# s = 10.04. "pow2" or "0" = the iter46 power-of-two scale 2**pv_k (= 8).
# Otherwise a float in [4, 32], canonicalized to 6 decimals (so one accepted
# value = one kernel); any other value RAISES (fail closed: a typo must not
# silently run the bf16-P control). The range keeps s * P' far below the
# e4m3 saturation (448) and keeps small P' (|P'| >= 2**-12) out of the flush.
# Why not a power of two: over negative scores P' = -log2(e) * silu peaks at
# the silu minimum (score -1.27846, P' = 0.40174), where silu is stationary,
# so many P' values pile up there. With s = 8 the pile sits at 3.2139, 1.1%
# below the e4m3 grid point 3.25, and rounds up systematically: a coherent
# bias on the outputs whose keys fall in the negative lobe. Every power of two
# keeps that position inside the binade. s = 10.04 puts the pile 0.84% above
# the grid point 4.0. Provenance of 10.04 (x4c19 iter67 pvscale, PREREG
# sha256 2023c904): located by the analytic route (format + kernel constants;
# mean-rounding-error zero s = 10.0379); not contradicted by a synthetic
# replication of the iter49 rule (non-discriminating on zero-mean V);
# replaces the real-dump emulation. Same declared precision class as FP8 P@V.
_FP8_FWD_PV_SCALE_DEFAULT = 10.04
_FP8_FWD_PV_SCALE_RANGE = (4.0, 32.0)


def _fp8_fwd_pv_scale():
    """(scale, raw): 0.0 for the power-of-two scale, a float in
    _FP8_FWD_PV_SCALE_RANGE rounded to 6 decimals, or None for an invalid
    $DLRM_HSTU_FP8_FWD_PV_SCALE."""
    raw = os.environ.get("DLRM_HSTU_FP8_FWD_PV_SCALE", "")
    v = raw.strip().lower()
    if v == "":
        return _FP8_FWD_PV_SCALE_DEFAULT, raw
    if v == "pow2":
        return 0.0, raw
    try:
        s = float(v)
    except ValueError:
        return None, raw
    if s == 0.0:
        return 0.0, raw
    lo, hi = _FP8_FWD_PV_SCALE_RANGE
    if not (lo <= s <= hi):    # also rejects nan / inf
        return None, raw
    return round(s, 6), raw


def fp8_fwd_pv_config(cfg: dict) -> dict:
    """$DLRM_HSTU_FP8_FWD_PV=1 (read per call): the pv kwargs to add to `cfg`,
    or {} when the knob is off or `cfg` is not the FWD_PIPE config (with or
    without mask_arith=3). Raises ValueError for an invalid
    $DLRM_HSTU_FP8_FWD_PV_SCALE. Logs one decline line per process, and one
    arming line per distinct armed configuration."""
    if os.environ.get("DLRM_HSTU_FP8_FWD_PV", "0") != "1":
        return {}
    want = dict(_load_fwd_kernel().SHIPPED)
    want.update(_FP8_FWD_CFG)
    want.update(_FP8_FWD_E2_CFG)
    want.update(_FP8_FWD_PIPE_CFG)
    # The one composition built and verified with pv=k128 (X4C19 iter47).
    ma = cfg.get("mask_arith") == _FP8_FWD_MASK_ARITH
    if ma:
        want["mask_arith"] = _FP8_FWD_MASK_ARITH
    diff = sorted(k for k in set(want) | set(cfg) if cfg.get(k) != want.get(k))
    if diff:
        if "declined" not in _fp8_fwd_pv_logged:
            _fp8_fwd_pv_logged.add("declined")

            def _kv(d, k):
                return f"{k}={d[k]!r}" if k in d else f"{k} unset"

            print(
                "[hstu-fwd] DLRM_HSTU_FP8_FWD_PV=1 declined: pv=k128 needs the "
                "FWD_PIPE config (DLRM_HSTU_FP8_FWD_PIPE=1, optionally with "
                "DLRM_HSTU_FP8_FWD_MASK_ARITH=1); active config has "
                + ", ".join(_kv(cfg, k) for k in diff) + " (needs "
                + ", ".join(_kv(want, k) for k in diff)
                + "); running the bf16-P forward",
                flush=True,
            )
        return {}
    s, s_raw = _fp8_fwd_pv_scale()
    if s is None:
        lo, hi = _FP8_FWD_PV_SCALE_RANGE
        raise ValueError(
            f"DLRM_HSTU_FP8_FWD_PV_SCALE={s_raw!r}: use unset (10.04), 'pow2', "
            f"'0' or a float in [{lo:g}, {hi:g}]")
    out = dict(_FP8_FWD_PV_CFG)
    if s:
        out["pv_s"] = s
    line = (
        "[hstu-fwd] DLRM_HSTU_FP8_FWD_PV=1 armed: "
        f"pv={_FP8_FWD_PV_CFG['pv']} pv_k={_FP8_FWD_PV_CFG['pv_k']} "
        + (f"pv_s={s!r} (P8 = e4m3(P' * {s!r}), static scale folded "
           "into the silu fma) " if s else
           f"(P8 = e4m3(P' * 2**{_FP8_FWD_PV_CFG['pv_k']})) ")
        + "(e4m3 P, K=128 P@V retile; declared precision change)"
        + (f" with mask_arith={_FP8_FWD_MASK_ARITH} (causal mask on the "
           "packed e4m3 P; bit-identical)" if ma else ""))
    # Latched on the line text: once per distinct armed configuration.
    if line not in _fp8_fwd_pv_logged:
        _fp8_fwd_pv_logged.add(line)
        print(line, flush=True)
    return out


# DLRM_HSTU_FP8_LAB_FWD_V8SWZ (X4C19 iter70; read per call, code default OFF;
# ship17 row (100) RB1 binds it to 1 in yambda_5b.gin, env "0" is the kill
# switch; the LAB_ prefix keeps the name the lab and in-model rung used).
# "1" sets hstu_fwd.py v8_swz=1: the
# pv="k128" V ring keeps rows 16..31 of each slot 128 B further, which removes
# the 2-way LDS bank conflict of the P@V tr8 B-operand read (rows r and r+16
# of the 144 B-stride ring share banks). Same bytes into the same MFMAs:
# bit-identical forward output; +512 B LDS per workgroup. It applies only when
# DLRM_HSTU_FP8_FWD_PV=1 armed pv=k128 (the shipped config); otherwise it logs
# one decline line and changes nothing. Unset or "0" = the shipped kernel (same
# compile kwargs, same symbol, same ISA). Any other value raises ValueError.
_fp8_fwd_v8swz_logged = set()


def fp8_fwd_v8swz_config(cfg: dict) -> dict:
    """$DLRM_HSTU_FP8_LAB_FWD_V8SWZ=1 (read per call): {"v8_swz": 1} when `cfg`
    already carries pv="k128", else {}. Logs one arming or decline line per
    process."""
    raw = os.environ.get("DLRM_HSTU_FP8_LAB_FWD_V8SWZ", "")
    if raw in ("", "0"):
        return {}
    if raw != "1":
        raise ValueError(
            f"DLRM_HSTU_FP8_LAB_FWD_V8SWZ={raw!r}: use unset, '0' or '1'")
    if cfg.get("pv") != "k128":
        if "declined" not in _fp8_fwd_v8swz_logged:
            _fp8_fwd_v8swz_logged.add("declined")
            print(
                "[hstu-fwd] DLRM_HSTU_FP8_LAB_FWD_V8SWZ=1 declined: needs the "
                "pv=k128 forward (DLRM_HSTU_FP8_FWD_PV=1 armed); active pv="
                f"{cfg.get('pv', '')!r}; running the forward unchanged",
                flush=True,
            )
        return {}
    if "armed" not in _fp8_fwd_v8swz_logged:
        _fp8_fwd_v8swz_logged.add("armed")
        print(
            "[hstu-fwd] DLRM_HSTU_FP8_LAB_FWD_V8SWZ=1 armed: v8_swz=1 (V ring "
            "rows 16..31 +128 B; P@V tr8 bank conflict removed; bit-identical)",
            flush=True,
        )
    return {"v8_swz": 1}


# DLRM_HSTU_FP8_LAB_FWD_RIDER (X4C19 iter71; read per call, code default OFF;
# ship17 row (100) RB1 binds it to 1 (rider=3) in yambda_5b.gin, env "0" is
# the kill switch). Bitwise epilogue diets of the pv="k128" forward
# (hstu_fwd.py `rider`); the head loop's IR is not touched. "1" = rider 3:
# R2a (per m-tile wave-uniform branch: full m-tiles store with no per-row exec
# mask) + R2b (one base VGPR per m-tile, the row/column part as MUBUF
# immediate offsets). Same values (same out_scale multiply and RNE bf16
# convert per value) to the same addresses: bit-identical forward output. R3
# (the zero-init peel of the design) is NOT built: every form tried changed
# the head loop. It composes with DLRM_HSTU_FP8_LAB_FWD_V8SWZ. It applies only
# when DLRM_HSTU_FP8_FWD_PV=1 armed pv=k128; otherwise it logs one decline
# line and changes nothing. Unset or "0" = the shipped kernel (same compile
# kwargs, same symbol, same ISA). Any other value raises ValueError, the
# former lab value "2" (R2c, inline asm; removed in row (100) v2) included.
_FP8_FWD_RIDER = {"1": 3}
_fp8_fwd_rider_logged = set()


def fp8_fwd_rider_config(cfg: dict) -> dict:
    """$DLRM_HSTU_FP8_LAB_FWD_RIDER (read per call): {"rider": 3} when `cfg`
    already carries pv="k128", else {}. Raises ValueError for any value but
    unset, "0" or "1". Logs one arming or decline line per process."""
    raw = os.environ.get("DLRM_HSTU_FP8_LAB_FWD_RIDER", "")
    if raw in ("", "0"):
        return {}
    if raw not in _FP8_FWD_RIDER:
        raise ValueError(
            f"DLRM_HSTU_FP8_LAB_FWD_RIDER={raw!r}: use unset, '0' or '1' "
            "(R2a+R2b)")
    if cfg.get("pv") != "k128":
        if "declined" not in _fp8_fwd_rider_logged:
            _fp8_fwd_rider_logged.add("declined")
            print(
                f"[hstu-fwd] DLRM_HSTU_FP8_LAB_FWD_RIDER={raw} declined: needs "
                "the pv=k128 forward (DLRM_HSTU_FP8_FWD_PV=1 armed); active pv="
                f"{cfg.get('pv', '')!r}; running the forward unchanged",
                flush=True,
            )
        return {}
    rider = _FP8_FWD_RIDER[raw]
    if raw not in _fp8_fwd_rider_logged:
        _fp8_fwd_rider_logged.add(raw)
        print(
            f"[hstu-fwd] DLRM_HSTU_FP8_LAB_FWD_RIDER={raw} armed: rider="
            f"{rider} (epilogue: R2a full-m-tile stores without exec mask, "
            "R2b base VGPR + immediate offsets; bit-identical)",
            flush=True,
        )
    return {"rider": rider}


# DLRM_HSTU_FP8_LAB_FWD_SPF (X4C19 iter71; read per call, code default OFF;
# ship17 item (105) "SPF fold fwd" binds it to 1 in yambda_5b.gin; a pre-set
# env value wins: "0" is the kill switch, "snap" the lab's snapped control;
# the LAB_ prefix keeps the name the lab evidence and item (106) use): the
# score-scale power-of-two fold, a DECLARED PRECISION CHANGE through the
# q/k/v scale (contract: x4c19 probes/iter71/spf/CONTRACT.md). Host side in
# ops/fp8_qkv_spf.py.
#   unset / "0"  off: the shipped scale and kernel.
#   "snap"       the SNAPPED CONTROL: calibration publishes the qkv scale
#                snapped UP onto the SPF grid (score_scale = -2**j exactly);
#                the kernel is the shipped one (same compile kwargs, same ISA).
#   "1"          snap + fold: hstu_fwd.py spf=True (the QK MFMA takes the
#                runtime E8M0 byte 127 + j, the per-element score multiply is
#                gone). Needs the pv=k128 forward.
# Under "snap" and "1" every call checks that its qkv_scale IS on the grid
# with j in [J_MIN, J_MAX] and raises otherwise (fail closed: an unsnapped
# scale, e.g. restored from a run without the knob, must not run as either
# arm). Any other value raises ValueError. j is a runtime operand (the E8M0
# byte is read from the runtime score scale), so one build serves every j.
# Logging: ONE armed line per process and mode ("snap" / "fold"): the latch
# is keyed by the mode, not by the scale, and maps it to its line. F1 of
# fp8_warmup compiles the armed build before RUN_START and prints its line
# outside the item's stdout capture; the latch is deliberately NOT in
# fp8_warmup._MHA_SETS (it is kept: _KEEP_GLOBALS), so no SPF arming line
# prints in the timed region. The arming proof of each layer's real snapped
# scale is triton_fp8's per-publication "qkv scale snapped" INFO line.
_fp8_fwd_spf_logged = {}


def fp8_fwd_spf_config(cfg: dict, qkv_scale: float, alpha: float) -> dict:
    """$DLRM_HSTU_FP8_LAB_FWD_SPF (read per call): {"spf": True} when "1" and
    the call is armable, {} for off/"snap". Raises ValueError for an invalid
    value and RuntimeError for an off-grid scale (snap/1) or a non-pv=k128
    `cfg` (1). Prints one armed line per process and mode (_fp8_fwd_spf_logged
    maps the mode to the line it printed)."""
    from generative_recommenders.ops import fp8_qkv_spf as spf

    mode = spf.spf_mode()
    if mode == "off":
        return {}
    j = spf.arm_check(qkv_scale, alpha)
    if mode == "fold" and cfg.get("pv") != "k128":
        raise RuntimeError(
            f"{spf.KNOB}=1 needs the pv=k128 forward (DLRM_HSTU_FP8_FWD_PV=1 "
            f"armed); active pv={cfg.get('pv', '')!r}. Refusing to run the "
            "snapped scale unfolded under the fold arm's label.")
    if mode == "snap":
        line = (
            f"[hstu-fwd] {spf.KNOB}=snap armed: snapped control, kernel "
            f"unchanged; qkv_scale={float(qkv_scale)!r} on the SPF grid, "
            f"score scale = -2**{j} exactly (declared precision change)")
        out = {}
    else:
        line = (
            f"[hstu-fwd] {spf.KNOB}=1 armed: spf=True (QK MFMA E8M0 A-scale "
            f"byte {127 + j} = 2**{j}; score multiply folded); qkv_scale="
            f"{float(qkv_scale)!r} on the SPF grid (declared precision change)")
        out = {"spf": True}
    if mode not in _fp8_fwd_spf_logged:
        _fp8_fwd_spf_logged[mode] = line
        print(line, flush=True)
    return out


def fp8_fwd_spf_warmup_scale(qkv_scale: float, alpha: float) -> float:
    """The runtime scale fp8_warmup's F1 hands the FP8 forward: `qkv_scale`
    unchanged with the SPF knob off, else snapped onto the grid (the F1
    synthetic 2**-6 is off it), so F1 compiles the arm's own build."""
    from generative_recommenders.ops import fp8_qkv_spf as spf

    if spf.spf_mode() == "off":
        return qkv_scale
    return spf.snap_scale(float(qkv_scale), float(alpha))[0]


# DLRM_HSTU_FP8_LUT_SILU_FWD (ship17 gin item (120) LUT-SiLU fwd, x4c19; read
# per call, code default OFF; yambda_5b.gin binds it to 1, a pre-set env "0" is
# the kill switch). DECLARED PRECISION CHANGE. "1" sets hstu_fwd.py lut=13 on
# the pv="k128" forward: the QK MFMA takes the E8M0 A-scale byte of the f32
# score_scale = -2**j * r, and each P8 byte is gathered from an 8 KB LDS copy
# of a table indexed by the top 13 bits of f16_rtz(2**j * raw); the table holds
# r and the e4m3 silu (ops/fp8_lut_silu.py build_table). The table is built on
# the HOST from the math function and the runtime scale (never from data) and
# is a launch argument (one more tensor after SEQ_OFF): a new scale builds a
# table, never a kernel. Parsing: ops/fp8_lut_silu.py lut_mode (unset/"0"
# off, "1" on, anything else ValueError). Not on the pv=k128 forward with
# mask_arith=3 (the lut path's only causal mask is mask_arith=3's packed P8
# byte mask): one decline line, the forward is unchanged.
_fp8_fwd_lut_logged = set()
# (device index, bits32(score_scale), pv_s, pv_k) -> (device uint8[8192],
# host uint8 tensor, upload event, set of stream handles that waited on it).
# Entries are never freed (a handful of scales per run, 8 KB each).
_fwd_lut_tables = {}
# device index -> the side stream the tables are uploaded on.
_fwd_lut_streams = {}


def fp8_fwd_lut_config(cfg: dict) -> dict:
    """$DLRM_HSTU_FP8_LUT_SILU_FWD=1 (read per call): {"lut": 13} when `cfg`
    already carries pv="k128" with epi_fold and mask_arith=3, else {}. Raises
    ValueError for any value but unset, "0" or "1". Logs one arming or decline
    line per process."""
    from generative_recommenders.ops import fp8_lut_silu

    if fp8_lut_silu.lut_mode() == "off":
        return {}
    if (cfg.get("pv") != "k128" or not cfg.get("epi_fold")
            or cfg.get("mask_arith") != 3):
        if "declined" not in _fp8_fwd_lut_logged:
            _fp8_fwd_lut_logged.add("declined")
            print(
                "[hstu-fwd] DLRM_HSTU_FP8_LUT_SILU_FWD=1 declined: needs the "
                "pv=k128 forward (DLRM_HSTU_FP8_FWD_PV=1 armed, epi_fold) with "
                "mask_arith=3 (DLRM_HSTU_FP8_FWD_MASK_ARITH=1); active "
                f"pv={cfg.get('pv', '')!r} epi_fold="
                f"{cfg.get('epi_fold', False)!r} mask_arith="
                f"{cfg.get('mask_arith', 0)!r}; running the forward unchanged",
                flush=True,
            )
        return {}
    if "armed" not in _fp8_fwd_lut_logged:
        _fp8_fwd_lut_logged.add("armed")
        print(
            "[hstu-fwd] DLRM_HSTU_FP8_LUT_SILU_FWD=1 armed: lut=13 (P8 = 8 KB "
            "LDS e4m3 table indexed by f16_rtz(2**j*raw); QK MFMA E8M0 A-scale "
            "byte from score_scale; declared precision change)",
            flush=True,
        )
    return {"lut": 13}


# The CUDA side of _fwd_lut_table, as small helpers (CPU tests replace them).
#
# Why the upload is race-free and never makes the host wait on GPU work:
#  * The table is staged in PINNED host memory, so the host->device copy is
#    asynchronous whatever the side stream holds. torch.cuda.Stream() hands
#    out a pool stream that other work (pipeline copies, process groups) may
#    also use, and on ROCm it may share a hardware queue with the compute
#    stream; neither can make the host wait here (a pageable copy could).
#    The pinned tensor stays referenced by the cache entry, so its memory
#    outlives the copy.
#  * The device buffer is allocated while the side stream is current, so the
#    caching allocator takes it from the side stream's pool: it is never a
#    block the compute stream freed with reads still queued.
#  * The kernel's launch stream waits on the upload event (GPU-side, no host
#    wait) before its first launch with the table; later work on that stream
#    is ordered after it by stream order, so each stream waits once. Work
#    already queued on a shared side stream can only delay the copy, never
#    race it (the event still orders); the cost is at most a GPU-side wait
#    at the first launch with a new table (a handful per run).
#  * An entry is never freed, so the buffer is never handed back to the
#    allocator while a launch may still read it (no record_stream needed).


def _fwd_lut_side_stream(device):  # pyre-ignore[2,3]
    idx = device.index
    side = _fwd_lut_streams.get(idx)
    if side is None:
        side = torch.cuda.Stream(device=device)
        _fwd_lut_streams[idx] = side
    return side


def _fwd_lut_upload(host: bytes, device):  # pyre-ignore[3]
    """(device uint8[len(host)], pinned host uint8 tensor, event): the copy
    and the event are queued on the side stream; nothing waits on the
    host."""
    cpu = torch.empty((len(host),), dtype=torch.uint8, pin_memory=True)
    cpu.copy_(torch.frombuffer(bytearray(host), dtype=torch.uint8))
    side = _fwd_lut_side_stream(device)
    with torch.cuda.stream(side):
        dev = torch.empty((len(host),), dtype=torch.uint8, device=device)
        dev.copy_(cpu, non_blocking=True)
        event = torch.cuda.Event()
        event.record(side)
    return dev, cpu, event


def _fwd_lut_launch_stream(device):  # pyre-ignore[2,3]
    """The stream hstu_fwd.py's call() launches fwd_kernel_0 on: the current
    stream, or the null stream with DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM=0."""
    from generative_recommenders.ops.flydsl import current_stream_launch

    if current_stream_launch():
        return torch.cuda.current_stream(device)
    return torch.cuda.default_stream(device)


def _fwd_lut_table(device, alpha: float, qkv_scale: float, cfg: dict):  # pyre-ignore[2,3]
    """The device LUT of item (120) for this call's f32 score_scale (built,
    uploaded and logged once per (device, scale, pv_s, pv_k)), with the launch
    stream ordered after its upload. Host Python and one 8 KB copy: no compile
    of any kind. A scale outside the contract raises RuntimeError."""
    from generative_recommenders.ops import fp8_lut_silu as lut

    try:
        ss = lut.score_scale32(qkv_scale, alpha)
        j, r = lut.split_scale(ss)
    except (ValueError, OverflowError) as error:
        raise RuntimeError(
            f"{lut.KNOB}=1 (item (120)): the score scale of qkv_scale="
            f"{qkv_scale!r} alpha={alpha!r} is outside the table contract "
            f"({error}); set {lut.KNOB}=0 to run without item (120)"
        ) from error
    # hstu_fwd.compile_hstu_fwd's default pv_s 0.0 = the power-of-two scale.
    pv_s = float(cfg.get("pv_s", 0.0))
    pv_k = int(cfg["pv_k"])
    key = (device.index, lut.bits32(ss), pv_s, pv_k)
    entry = _fwd_lut_tables.get(key)
    if entry is None:
        tbl = lut.build_table(ss, pv_s, pv_k)
        if len(tbl) != lut.SIZE:
            raise RuntimeError(
                f"{lut.KNOB}: build_table returned {len(tbl)} bytes, not "
                f"{lut.SIZE}")
        dev, cpu, event = _fwd_lut_upload(tbl, device)
        entry = (dev, cpu, event, set())
        _fwd_lut_tables[key] = entry
        print(
            "[hstu-fwd] DLRM_HSTU_FP8_LUT_SILU_FWD table built: "
            "score_scale=%.9g (0x%08x) j=%d r=%.9g S=%g bytes=%d crc32=%08x"
            % (ss, lut.bits32(ss), j, r, pv_s if pv_s else 2.0 ** pv_k,
               len(tbl), lut.table_crc32(tbl)),
            flush=True,
        )
    dev, _cpu, event, waited = entry
    stream = _fwd_lut_launch_stream(device)
    handle = stream.cuda_stream
    if handle not in waited:
        stream.wait_event(event)
        waited.add(handle)
    return dev


def _fp8_fwd_config() -> dict:
    """Compile kwargs for the e4m3 FlyDSL forward, including the fused silu epi."""
    cfg = dict(_load_fwd_kernel().SHIPPED)
    cfg.update(_FP8_FWD_CFG)
    cfg.update(_FP8_FWD_E2_CFG)
    pipe = fp8_fwd_pipe_enabled()
    if pipe:
        cfg.update(_FP8_FWD_PIPE_CFG)
    if fp8_fwd_mask_arith_enabled(pipe):
        cfg["mask_arith"] = _FP8_FWD_MASK_ARITH
    cfg.update(fp8_fwd_pv_config(cfg))
    cfg.update(fp8_fwd_v8swz_config(cfg))
    cfg.update(fp8_fwd_rider_config(cfg))
    cfg.update(fp8_fwd_lut_config(cfg))
    return cfg


def flydsl_fp8_fwd_selected() -> bool:
    """True when the FlyDSL e4m3 forward (plus fused-silu epi) is requested.

    Natural selector: ``DLRM_HSTU_FP8_FWD=1`` and ``HSTU_ATTN_FWD=flydsl``.
    Deprecated alias: ``DLRM_HSTU_FP8_FWD_FLYDSL=1`` (same arm; still honored).
    """
    if os.environ.get("DLRM_HSTU_FP8_FWD_FLYDSL", "0") == "1":
        return True
    return (
        os.environ.get("DLRM_HSTU_FP8_FWD", "0") == "1"
        and flydsl_fwd_selected()
    )


def unsupported_fp8_fwd_shape_reason(
    q, k, v, num_targets, max_attn_len, contextual_seq_len,
    num_softmax_heads: int = 0,
) -> Optional[str]:
    """Why the e4m3 forward kernel cannot serve this call, or None if it can.

    The bf16 predicate with the dtype clause inverted: q/k/v must all be e4m3,
    and they must share a row stride, because the fp8 kernel reads the vqk slice
    in place exactly as the bf16 one reads the uvqk slice.
    """
    if max_attn_len:
        return f"max_attn_len={max_attn_len} (kernel is plain causal)"
    if contextual_seq_len:
        return f"contextual_seq_len={contextual_seq_len} (kernel assumes 0)"
    if num_softmax_heads:
        return f"num_softmax_heads={num_softmax_heads} (kernel assumes 0)"
    if q.dim() != 3 or q.shape[-1] != SUPPORTED_DIM:
        return f"qk dim {tuple(q.shape)} (kernel is written around dim {SUPPORTED_DIM})"
    if v.shape[-1] != SUPPORTED_DIM:
        return f"v dim {v.shape[-1]} (kernel is written around dim {SUPPORTED_DIM})"
    fp8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if any(t.dtype not in fp8 for t in (q, k, v)):
        return (
            "dtype "
            f"{tuple(str(t.dtype).replace('torch.', '') for t in (q, k, v))}"
            " (the fp8 path needs e4m3 q, k and v)"
        )
    if num_targets is not None:
        n = _max_num_targets(num_targets)
        if n > 1:
            return f"num_targets max={n} (target-aware masking is identity only at <= 1)"
    return None


def flydsl_hstu_attention_fwd_fp8(
    q, k, v, seq_offsets, N, alpha, qkv_scale: float
) -> torch.Tensor:
    """The e4m3 forward. Returns a packed [tokens, heads, dim] BF16 tensor.

    `qkv_scale` is the single per-tensor scale the fused vqk GEMM quantized with.
    The kernel applies it as qkv_scale**2 on the scores (in f32, on the way into
    the silu) and once more on the out accumulator for V -- not folded into the
    MFMA's scale field, which is E8M0 and could not carry it.

    Output dtype is bf16 and NOT q.dtype: attention's output is a normal
    activation, and `triton_hstu_attention_fwd` returns bf16 here too.
    """
    kernel = _load_fwd_kernel()
    heads, dim = q.shape[1], q.shape[2]

    rs = _common_row_stride((q, k, v), heads, dim)
    if rs is None:
        q, k, v = (t.contiguous() for t in (q, k, v))
    cfg = _fp8_fwd_config()
    # Item (105) SPF fold fwd (gin-bound ON): {} unless
    # DLRM_HSTU_FP8_LAB_FWD_SPF=1 arms it for this call's (grid-checked)
    # runtime scale; "snap" / "1" with an off-grid scale raise.
    cfg.update(fp8_fwd_spf_config(cfg, qkv_scale, alpha))
    fn = kernel.compile_hstu_fwd(heads, dim, alpha, rs=(rs or 0), **cfg)

    key = (rs, heads, dim)
    if key not in _logged_fp8_fwd_layouts:
        _logged_fp8_fwd_layouts.add(key)
        logger.warning(
            "[hstu-attn] FlyDSL FP8 forward %s (heads=%d dim=%d tokens=%d N=%d "
            "mean_len=%.0f scale=%.6g cfg=%s)",
            f"reading q/k/v in place at row stride {rs}" if rs
            else "packing q/k/v to contiguous (row stride %d)" % (heads * dim),
            heads, dim, q.shape[0], int(N),
            q.shape[0] / max(1, seq_offsets.numel() - 1), qkv_scale,
            cfg,
        )

    cu = _fwd_seq_offsets_i32(seq_offsets)
    batch = seq_offsets.numel() - 1
    m_blocks = int((int(N) + fn.BM - 1) // fn.BM)
    out = torch.empty(
        (q.shape[0], heads, dim), dtype=torch.bfloat16, device=q.device
    )
    if cfg.get("lut"):
        # Item (120): the table is a launch argument right after SEQ_OFF.
        lut_t = _fwd_lut_table(q.device, alpha, float(qkv_scale), cfg)
        fn(q.detach(), k.detach(), v.detach(), out, cu, lut_t, batch,
           m_blocks, 1.0 / float(N), float(qkv_scale))
    else:
        fn(q.detach(), k.detach(), v.detach(), out, cu, batch, m_blocks,
           1.0 / float(N), float(qkv_scale))
    return out


def try_flydsl_attention_fwd_fp8(
    *, q, k, v, seq_offsets, num_targets, N, alpha, max_attn_len,
    contextual_seq_len, num_softmax_heads, qkv_scale,
) -> Optional[torch.Tensor]:
    """The e4m3 attention forward from FlyDSL, or None to use Triton's.

    FAIL-CLOSED. Returns None only when the knob is off. Once it is on, every
    other way this could decline -- host cannot run FlyDSL, kernel import
    failed, shape outside the specialization -- RAISES. A knob whose entire
    purpose is to be A/B'd against the incumbent must not be able to silently
    measure the incumbent twice.
    """
    if not flydsl_fp8_fwd_selected():
        return None
    if os.environ.get("DLRM_HSTU_FP8_FWD", "0") != "1":
        raise RuntimeError(
            "HSTU_ATTN_FWD=flydsl (or DLRM_HSTU_FP8_FWD_FLYDSL=1) needs "
            "DLRM_HSTU_FP8_FWD=1 to run the e4m3 FlyDSL forward. It is a "
            "replacement for the Triton e4m3 forward inside the FP8 arm, not "
            "a way to turn the FP8 arm on."
        )
    require_flydsl_fwd()
    reason = unsupported_fp8_fwd_shape_reason(
        q, k, v, num_targets, max_attn_len, contextual_seq_len,
        num_softmax_heads,
    )
    if reason is not None:
        raise RuntimeError(
            f"HSTU_ATTN_FWD=flydsl with DLRM_HSTU_FP8_FWD=1 but the FlyDSL "
            f"e4m3 forward cannot serve this shape ({reason}). Refusing to "
            f"fall back silently -- set HSTU_ATTN_FWD=triton for the Triton "
            f"FP8 forward."
        )
    out = flydsl_hstu_attention_fwd_fp8(
        q, k, v, seq_offsets, N, alpha, qkv_scale
    )

    global _fp8_forward_calls
    _fp8_forward_calls += 1
    period = int(os.environ.get("HSTU_ATTN_FWD_CHECK", "0") or 0)
    if period > 0 and _fp8_forward_calls % period == 0:
        ref = triton_hstu_attention_fwd(
            N=N, alpha=alpha, q=q, k=k, v=v, seq_offsets=seq_offsets,
            num_targets=num_targets, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=None, enable_tma=False,
            num_softmax_heads=0,
            q_scale=qkv_scale, k_scale=qkv_scale, v_scale=qkv_scale,
        )
        logger.info(
            "[hstu-fwd-fp8-check] call=%d shape=%s snr_db out=%.1f",
            _fp8_forward_calls, tuple(q.shape), _snr_db(ref, out),
        )
    return out


def _triton_bwd(
    dout, q, k, v, seq_offsets, num_targets, N, alpha, max_attn_len,
    contextual_seq_len, sort_by_length_indices, enable_tma,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    triton_hstu_attention_bwd(
        dout=dout, q=q, k=k, v=v, dq=dq, dk=dk, dv=dv,
        seq_offsets=seq_offsets, num_targets=num_targets, N=N, alpha=alpha,
        max_attn_len=max_attn_len, contextual_seq_len=contextual_seq_len,
        sort_by_length_indices=sort_by_length_indices, enable_tma=enable_tma,
        num_softmax_heads=0,
    )
    return dq, dk, dv


def _snr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Signal-to-noise in dB of `got` against `ref`; +inf when bit-identical.

    Preferred over a max-abs-diff tolerance because these are whole gradient
    tensors: one denormal-scale element disagreeing in the last ulp would fail an
    elementwise bound while being numerically irrelevant, whereas SNR weighs the
    error against the signal actually present.
    """
    ref32, got32 = ref.float(), got.float()
    noise = torch.linalg.vector_norm(ref32 - got32).item()
    if noise == 0.0:
        return float("inf")
    signal = torch.linalg.vector_norm(ref32).item()
    return 20.0 * torch.log10(torch.tensor(signal / noise)).item()


class _FlyDSLAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, N, alpha, q, k, v, seq_offsets, num_targets, max_attn_len,
        contextual_seq_len, sort_by_length, enable_tma,
    ):
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved.append(num_targets)
        if sort_by_length_indices is not None:
            saved.append(sort_by_length_indices)
        ctx.save_for_backward(*saved)
        ctx.alpha = alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        # $HSTU_ATTN_FWD is honoured here too, so the non-fused dispatch and the
        # fused preprocess+attention path agree on which forward they run.
        out = try_flydsl_attention_fwd(
            q=q, k=k, v=v, seq_offsets=seq_offsets, num_targets=num_targets,
            N=N, alpha=alpha, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len, num_softmax_heads=0,
        )
        if out is not None:
            return out
        return triton_hstu_attention_fwd(
            N=N, alpha=alpha, q=q, k=k, v=v, seq_offsets=seq_offsets,
            num_targets=num_targets, max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=enable_tma, num_softmax_heads=0,
        )

    @staticmethod
    def backward(ctx, dout):
        global _backward_calls
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            num_targets = None
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            sort_by_length_indices = (
                ctx.saved_tensors[idx] if ctx.sort_by_length else None
            )

            dout = dout.contiguous()
            # Reaching this Function at all means flydsl was selected, so a host
            # that cannot run it is an error, not a fallback. Only the shape
            # decides between the two backends here.
            require_flydsl()
            reason = unsupported_shape_reason(
                q, k, v, dout, num_targets, ctx.max_attn_len,
                ctx.contextual_seq_len,
            )
            use_flydsl = reason is None
            if not use_flydsl:
                _log_shape_fallback(reason)
            if use_flydsl:
                dq, dk, dv = flydsl_hstu_attention_bwd(
                    dout, q, k, v, seq_offsets, ctx.N, ctx.alpha
                )
            else:
                dq, dk, dv = _triton_bwd(
                    dout, q, k, v, seq_offsets, num_targets, ctx.N, ctx.alpha,
                    ctx.max_attn_len, ctx.contextual_seq_len,
                    sort_by_length_indices, ctx.enable_tma,
                )

            _backward_calls += 1
            if use_flydsl and _check_due(_backward_calls):
                _log_reference_snr(
                    dq, dk, dv,
                    _triton_bwd(
                        dout, q, k, v, seq_offsets, num_targets, ctx.N,
                        ctx.alpha, ctx.max_attn_len, ctx.contextual_seq_len,
                        sort_by_length_indices, ctx.enable_tma,
                    ),
                    q.shape,
                )
        return (None, None, dq, dk, dv, None, None, None, None, None, None)


def _check_due(call_index: int) -> bool:
    """True when this backward should also run Triton as a numerical reference.

    $HSTU_ATTN_BWD_CHECK is the period in backward calls: 0 (default) disables
    the cross-check, N>0 runs it every Nth call. It doubles the cost of the
    backward on the calls it fires, so it is a validation knob, not a production
    one.
    """
    period = int(os.environ.get("HSTU_ATTN_BWD_CHECK", "0") or 0)
    return period > 0 and call_index % period == 0


def _log_reference_snr(dq, dk, dv, reference, shape) -> None:
    ref_dq, ref_dk, ref_dv = reference
    logger.info(
        "[hstu-bwd-check] call=%d shape=%s snr_db dq=%.1f dk=%.1f dv=%.1f",
        _backward_calls,
        tuple(shape),
        _snr_db(ref_dq, dq),
        _snr_db(ref_dk, dk),
        _snr_db(ref_dv, dv),
    )


def hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
    enable_tma: bool = False,
) -> torch.Tensor:
    """Signature-compatible with triton_hstu_mha."""
    return _FlyDSLAttentionFunction.apply(
        N, alpha, q, k, v, seq_offsets, num_targets, max_attn_len,
        contextual_seq_len, sort_by_length, enable_tma,
    )

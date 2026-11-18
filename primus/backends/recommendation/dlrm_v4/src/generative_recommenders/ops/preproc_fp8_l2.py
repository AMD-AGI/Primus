# pyre-strict
"""E-M3a: FP8 L2 GEMMs for ContextualPreprocessor's content and additional MLPs.

DEFAULT OFF. Gate: DLRM_HSTU_FP8_PREPROC_L2=1, read once per process (forwarded
by launch_slurm.sh's DLRM_HSTU_FP8_ prefix). Absent or "0": GroupedTail
(ops/preproc_grouped_tail.py) never calls into this module, so the shipped E-M1
path runs unchanged. Needs the E-M1 grouped tail (gin default on); when that
tail declines, so does this.

A PRECISION CHANGE (Arbor actckpt E-M3a, redesign_E_M3_fp8.md). Turning it on by
default needs the user's numerics ruling and an equal-sample convergence pair.
Site rule, checked on SYNTHETIC tensors only (Arbor actckpt iteration 71,
pre-registered; fp64 oracle; init-law MLPs, synthetic inputs and upstream
gradients): an FP8 L2-wgrad site qualifies iff none of its 512 dW2 output rows
falls below 10 dB. The content and additional NN-b wgrads keep every row at
21.9 dB or better (pooled 24-40 dB, against ~50 dB for BF16). The action MLP's L2
stays BF16 by structure: its ActionEncoder input has at most 9 distinct rows
(3 action bits plus the target row), so each dW2 row combines 9 group sums and
its SNR has a heavy few-degrees-of-freedom tail; FP8 leaves 1-7 of 512 rows
below 10 dB in 11 of 18 synthetic cases.

WHAT CHANGES, per armed MLP k (0 = content, 1 = additional):
  forward
    * pair 0 (Linear -> SwishLayerNorm(256)): the L1 addmm is the shipped one.
      The norm forward is P2 (triton_preproc_fp8.swishln256_fwd_emit): the
      shipped v2 arithmetic, then a1 rounded to bf16 and cast to e4m3 at a
      delayed per-site scale, written row-major [M, 256] and token-major
      [256, Mp]. No bf16 a1 is written. The pair returns a bf16-typed
      stride-0 PROXY (one storage element, never read). The FP8 operands
      travel to the grouped tail in the GroupedTail's plan, so no FP8 tensor
      crosses an autograd edge (autograd would cast its bf16 grad to it).
    * last pair: h2 = _scaled_mm(e4m3 a1, e4m3 W2^T) + b2 (bf16 out), then the
      unchanged E-M1 grouped LayerNorm(512) tail over all three MLPs.
  backward
    * the LayerNorm(512) dx-sum backward writing e5m2 dh2 at a delayed
      per-site scale (plus per-block amax partials), with the dweight/dbias/
      dxsum of the LayerNorm (dxsum is the Linear's bias gradient, as F1).
      Two producers (see STACKED BACKWARD below):
        - stacked (default, kill switch DLRM_HSTU_FP8_PREPROC_L2_GROUPED):
          ONE FlyDSL launch for all three MLPs of the tail
          (ops/flydsl/ln_bwd_grouped_fp8.py ln_bwd_grouped3_fp8, main kernel
          *grp3f8* + the grouped finalize): e5m2 dh2 + amax partials for
          content and additional, bf16 dh for the action MLP;
        - P3 (triton_preproc_fp8.ln512_bwd_dxsum_emit), one launch per FP8
          MLP, when the stacked kernel does not take the backward.
    * dgrad  da1 = _scaled_mm(e5m2 dh2, e4m3 W2) (bf16 out).
    * wgrad  (NN-b) dW2^T = hipBLASLt FP8 NN(y = e5m2 dh2 row-major,
      dout_t = e4m3 a1 token-major); then a 128k-element transpose. Without a
      hipBLASLt NN algorithm: dh2 transposed + _scaled_mm TN.
    * pair 0's backward is the shipped one (SwishLN backward recomputes from
      h1 with mean/rstd; L1 dgrad/wgrad BF16).

SCALES. a1 is e4m3 and dh2 e5m2, one delayed scale per MLP and operand, under
bwd_fp8's policy and tunables (DLRM_HSTU_FP8_BWD_WARMUP_CALLS, _REFRESH_CALLS,
_MARGIN): the first WARMUP calls run the shipped BF16 path and take a
standalone amax; from then on the producers' amax partials feed the window,
published every REFRESH calls as scale = MARGIN * amax / FORMAT_MAX. FP8 starts
for an MLP once BOTH of its scales are published (training step WARMUP + 1).
The decode scale paired with each FP8 operand is the scale it was cast with:
a1's is copied at cast time (it is saved for the backward); dh2's GEMMs run
right after its cast, so it is copied only on a publishing call. A non-finite
amax is booked as 0 (an inf must not become a window's scale); the casts keep
NaN (a NaN operand gives a NaN loss as on the shipped path). Weights are e4m3
from their own current amax at unit
margin (bwd_fp8.quantize_weight), cast once per forward together with the
transposed bytes the dgrad reads. Eval forwards (no grad) use the published
scales and update nothing. W2's cast is E-M3a's own one-launch kernel
(_quantize_w2 -> triton_preproc_fp8.quantize_w2_dual) whatever
DLRM_HSTU_FP8_BWD_HELPERS_FUSED says: the same bytes and scale as
quantize_weight and as bwd_fp8's fused helper path (ship12 port: one W2 path).

LAUNCHES (iteration 50, value for value). Each site call is one bookkeeping
launch that reads the producer's amax partials (triton_preproc_fp8.site_steady,
was partials.amax() plus 6-7 tiny torch ops), and W2's cast is one launch that
also writes its transposed bytes and, from fp32 master weights under
autocast(bf16), the bf16 b2 (W2's bf16 rounding is applied in registers before
the cast and never stored; quantize_w2_dual, was 2 casts, 9 quantize_weight
launches and a backward transpose). Publishing calls keep _publish's torch ops.

FALLBACKS. A late GroupedTail decline re-runs an FP8 MLP's pair 0 the shipped
way from its saved input, so the queued last pairs run exactly the shipped
computation; the FP8 pair-0 node is then never reached by autograd. A gate that
is on but cannot engage (rows < 2^16, unexpected MLP pairs)
logs one WARNING per process (the v2 EMU mode is not a precondition: unknown,
it defaults to the pinned production mode). The stacked backward's own
fallbacks are listed under STACKED BACKWARD.

STACKED BACKWARD (E-M3a x E-M2, Arbor actckpt iteration 53). Kill switch
DLRM_HSTU_FP8_PREPROC_L2_GROUPED, read once per process and consulted only
while this gate is on: default "1" = on, "0" = the P3 path for every FP8 MLP.
  WHAT CHANGES. In a backward of this node whose content and additional MLPs
    are "fp8" and whose action MLP is BF16 (the FP8 steady state), the three
    LayerNorm(512) dx-sum backwards, which share dy, run as ONE call of
    ln_bwd_grouped_fp8.ln_bwd_grouped3_fp8 ahead of the per-MLP loop. Its main
    kernel is E-M2's FlyDSL grouped kernel (ops/flydsl/ln_bwd_grouped.py, P7
    m1b: dy loaded once per row; the grid, row mapping, per-problem arithmetic
    and grouped finalize of the three PR #7 calls) with per-problem output
    modes: content and additional get e5m2 dh2 = e5m2(clamp(fp32 dx *
    inv_scale, +-57344), NaN kept) at their dh2 site's inv_scale plus amax
    partials of the pre-scale |fp32 dx|; the action MLP gets its bf16 dh. The
    loop then runs every MLP's GEMMs and bookkeeping unchanged: the action
    MLP's dgrad/wgrad from that bf16 dh, and per FP8 MLP dh_scale =
    site.steady(partials) followed by the FP8 dgrad and NN wgrad. Out of
    place only; each MLP's outputs are released after its GEMMs.
  NUMERICS. The action MLP's five gradients are the P3 path's bit for bit:
    its bf16 dh, dweight, dbias and dxsum equal ln_bwd_grouped3's, which equal
    the PR #7 per-pair call's. For content and additional the fp32 dx that is
    cast and summed follows PR #7's operation order instead of P3's, so dh2,
    the LayerNorm weight/bias gradients and the L2 bias gradient are not bit
    for bit the P3 path's (rounding-level differences of that fp32 dx), and
    the delayed scale books the amax of those same fp32 values. Same formats,
    scales and scale policy; compared against an fp64 oracle in the lab.
  WHEN IT RUNS. Exactly three MLPs with modes (fp8, fp8, not fp8); PR #7 owns
    the three per-pair LayerNorm(512) calls by E-M2's own FlyDSL predicate
    (preproc_grouped_tail._flydsl_grouped_reason gives no reason, which also
    needs E-M2's FlyDSL latch _BWD_FLYDSL_BROKEN unset); and
    ln_bwd_grouped_fp8.grouped_fp8_reason accepts the inputs. E-M2's gate
    (DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD) and its FlyDSL sub-switch are not
    consulted. BF16 warmup calls ("warm") and every other mode mix keep the
    per-MLP backward.
  FALLBACKS. Not eligible: the P3 path, silently. An exception from the
    stacked call (import, JIT compile, launch): the stacked path is latched
    off for the process with one WARNING, and this and every later backward
    runs the P3 path. That is safe even after its main kernel was launched:
    every output is a fresh tensor and no input is written. The kernel
    module is imported before E-M2's predicate runs, so an import error (of
    ln_bwd_grouped_fp8 or of what it imports: FlyDSL, ln_bwd_grouped)
    latches only the stacked path, never E-M2's FlyDSL path. Arming: one
    "armed OK: [preproc-fp8-l2-grouped] kernel=<main kernel> N=<rows>
    rank=<r>" line per process (every rank), at the first stacked backward.
  E-M2 INTERPLAY. E-M2's own grouped backward still never runs in this node.
    The stacked path does not depend on it: with E-M2's gate on or off, the
    FP8 steps run E-M2's kernel extended by output modes in place of the
    three per-pair calls, and the warmup calls and the declines above run the
    per-pair backward (PR #7 per call), bitwise equal to E-M2's.
  WARM-UP. warmup_kernels does not touch the stacked path: it runs inside
    run_blas_warmup, before the DataLoader pre-start forks its pools, and
    imports no FlyDSL. The stacked kernels (main + grouped finalize) are
    JIT-built after the forks, by prebuild_grouped from the startup preload
    (dlrm_v4/train/blas_warmup.py run_startup_preload ->
    _prebuild_flydsl_ln_bwd, right after E-M2's grouped prebuild), on
    synthetic zero/one tensors, when this gate and the kill switch are on and
    PR #7 could own the calls; a failure there latches only the stacked path
    off (one WARNING) and the startup goes on. Without DL_PRESTART the
    preload does not run and the stacked kernel compiles at its first use, as
    E-M2's does. The one Triton kernel the stacked path adds to the P3 path's
    set, dh2's site_steady on the stacked kernel's sms*8 amax partials, is the
    specialization that warmup_kernels already compiles for P3 (P3's grid at
    the warm-up rows is sms*8 too; see _WARM_ROWS).

X2L-B (gin item (93); ops/hip/preproc_l2_x2lb.py + x2lb_pass.cu). DEFAULT OFF:
DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD=1, read once per process ("0" = kill
switch); unset or "0": that module is never imported and every statement
below runs as without it (ctx.x2lb is absent on pair 0 and None on the tail).
  WHAT CHANGES. Per FP8 MLP (k = 1, then 0) in the tail's backward, the FP8
    dgrad da1 = _scaled_mm(dh8, w8_t.t(), ...) and pair 0's SwishLN(256)
    backward on that da1 (PR #7's ln_bwd_flydsl_d256_swish_dxsum_x16_c3)
    run as ONE persistent HIP kernel (x2lb_pass) plus PR #7's finalize
    ln_bwd_flydsl_fin_d256_n3_wf32 over its PART; da1 never touches HBM.
    The tail reads pair 0's saved h1, norm0 w/b, mean1, rstd1 through
    pair 0's ctx.saved_tensors (the same version checks) and returns, as the
    proxy's gradient, a sentinel (a per-device bf16 zero scalar expanded to
    [M, 256]); pair 0's backward then takes (dh1, dW, dB, db1) from the
    handle instead of running its SwishLN backward. E-X2b's port route, the
    IX wgrad, the ledger and every returned gradient's node are unchanged.
  PLUMBING. Pair 0's forward publishes an X2LBHandle into ctx.x2lb and the
    Slot (Slot.x2lb); the tail's forward moves it into ctx.x2lb[k] (single
    use, as the payload; attach records the proxy by weakref); the tail's
    backward calls try_fused in place of the dgrad statement; pair 0's
    backward calls handle.take(dy), which raises on any handshake mismatch.
    A tail backward that raises drops the results it fused (release). With
    the knob on, a module that cannot be imported turns X2L-B off for the
    process (one "[preproc-l2-x2lb] FAILED CLOSED (import)" line).
  WHEN IT RUNS. Train FP8 steady state, after the module's pre-RUN_START
    prebuild armed the device; per call it declines (the shipped dgrad and
    SwishLN backward run for that k) under create_graph, anomaly mode, a
    live proxy (capture / retain_grad / hook), a backward that does not
    execute pair 0 or only captures it (autograd.grad, backward(inputs=)),
    CUDA graph capture, an operand outside its contract, and whenever PR #7
    would not own the shipped SwishLN call.
  NUMERICS. DECLARED fp32 reduction-order class: dh1 bitwise on rows
    hipBLASLt computes unsplit, fp32-order on its split stream-K rows; dW, dB
    and db1 fp32-order (X2L-B's partition + PR #7's finalize). Everything the
    tail computes for W2, b2, norm1 and the action MLP is bitwise.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
from generative_recommenders.ops.triton import bwd_fp8, preproc_fp8_snr
from generative_recommenders.ops.triton import triton_ec_dgrad_segreduce as _xs
from generative_recommenders.ops.triton import triton_layer_norm as TLN
from generative_recommenders.ops.triton.triton_layer_norm import (
    compute_BLOCK_D,
    triton_weighted_layer_norm_bwd_dxsum,
)

logger: logging.Logger = logging.getLogger(__name__)

ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_L2"
ARMED_PREFIX: str = "armed OK: [preproc-fp8-l2]"
# GroupedTail order: 0 content, 1 additional, 2 action. Only 0 and 1 go FP8.
FP8_MLPS: Tuple[int, ...] = (0, 1)
# The shipped v2 SwishLN forward (whose arithmetic P2 reproduces) needs >= 2^16 rows.
_MIN_ROWS: int = 1 << 16
_HID: int = 256
_OUT: int = 512
# Indexed-X L1 (ops/ec_indexed_l1.py, DLRM_HSTU_FP8_EC_INDEXED_L1, default off): the tag on an EC placeholder
# (literal: nothing is imported unless a tensor carries it).
_IX_ATTR: str = "_dlrmv4_ec_indexed_l1"


def _ix() -> Any:
    from generative_recommenders.ops import ec_indexed_l1

    return ec_indexed_l1


# X2L-B (gin item (93), module docstring): its knob, read once WITHOUT importing
# ops/hip/preproc_l2_x2lb (a literal copy of that module's ENV_VAR; knob unset
# or "0": the module is never imported).
_X2LB_ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_L2_FUSED_BWD"
_X2LB_REQUESTED: Optional[bool] = None


def _x2lb_requested() -> bool:
    global _X2LB_REQUESTED
    if _X2LB_REQUESTED is None:
        _X2LB_REQUESTED = os.environ.get(_X2LB_ENV_VAR, "").strip().lower() in (
            "1", "true", "on", "yes",
        )
    return _X2LB_REQUESTED


def _x2lb() -> Any:
    from generative_recommenders.ops.hip import preproc_l2_x2lb

    return preproc_l2_x2lb


def _x2lb_off(why: Optional[BaseException] = None) -> None:
    """X2L-B off for the rest of the process: its module cannot be imported. With `why`, one FAILED CLOSED line on
    every rank (blas_warmup's phase-1 hook prints its own line and passes None)."""
    global _X2LB_REQUESTED
    _X2LB_REQUESTED = False
    if why is None:
        return
    msg = (f"[preproc-l2-x2lb] FAILED CLOSED (import): {type(why).__name__}: {str(why)[:300]} -> shipped dgrad + "
           "SwishLN bwd for the rest of the run")
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(msg)
    except Exception:  # never abort a forward over a log line
        logger.warning("%s", msg)


def _x2lb_publish(ctx: Any, k: int, h: torch.Tensor, train: bool) -> Any:
    """X2L-B's handle for this forward, or None. A failed import of ops/hip/preproc_l2_x2lb fails closed (as X2H's
    guarded import): today's statements run for this and every later forward."""
    try:
        mod = _x2lb()
    except Exception as exc:
        _x2lb_off(exc)
        return None
    return mod.publish(ctx, k, h, train)


_ENABLED: Optional[bool] = None
_ARMED_LOGGED: bool = False
_WGRAD_LOGGED: bool = False
_INERT_WARNED: bool = False

# STACKED BACKWARD (module docstring): its kill switch, read once per process
# and only while ENV_VAR is "1" (default "1" = on, "0" = the P3 path).
GROUPED_ENV_VAR: str = "DLRM_HSTU_FP8_PREPROC_L2_GROUPED"
GROUPED_ARMED_PREFIX: str = "armed OK: [preproc-fp8-l2-grouped]"
_GROUPED_ENABLED: Optional[bool] = None
_GROUPED_ARMED_LOGGED: bool = False
# First host-side error of the stacked call (import, JIT compile, launch) or of
# its startup prebuild (prebuild_grouped): the stacked path stays off for the
# process.
_GROUPED_BROKEN: Optional[str] = None
# Why the last FP8-node backward did not take the stacked path (None: it did,
# or no backward ran yet). Diagnostics only; a decline logs nothing.
_GROUPED_LAST_DECLINE: Optional[str] = None


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def enabled() -> bool:
    """DLRM_HSTU_FP8_PREPROC_L2=1 and every precondition this build can check."""
    global _ENABLED
    if _ENABLED is None:
        on = os.environ.get(ENV_VAR, "0") == "1"
        reason: Optional[str] = None
        if on:
            if torch.version.hip is None:
                reason = "not a ROCm/HIP build (verified on gfx950 only)"
            elif getattr(torch, "float8_e4m3fn", None) is None or getattr(
                torch, "float8_e5m2", None
            ) is None:
                reason = "torch has no float8_e4m3fn/float8_e5m2"
            elif preproc_fp8_snr.ENABLED:
                reason = "the FP8 numerics probe (DLRM_HSTU_FP8_PREPROC_SNR) is armed"
            elif not TLN._wln_fwd_v2_enabled():
                reason = "DLRM_HSTU_FP8_WLN_FWD_V2 is off (P2 reproduces the v2 forward)"
        if on and reason is not None:
            logger.warning(
                "[preproc-fp8-l2] %s=1 IGNORED: %s; the shipped BF16 path runs rank=%d",
                ENV_VAR,
                reason,
                _rank(),
            )
            on = False
        if on:
            logger.info(
                "[preproc-fp8-l2] resolved: mlps=content,additional a1=e4m3(rm+tm) "
                "dh2=e5m2 wgrad=nn-b margin=%g warmup=%d refresh=%d rank=%d",
                bwd_fp8._BWD_MARGIN,
                bwd_fp8._BWD_WARMUP_CALLS,
                bwd_fp8._BWD_REFRESH_CALLS,
                _rank(),
            )
        _ENABLED = on
    return _ENABLED


# --------------------------------------------------------------------------- scales
class _Site:
    """One operand's delayed scale, held and published on device (bwd_fp8._SiteScale's
    policy, parameterized by the format's max). scale is the dequantization scale."""

    def __init__(self, name: str, limit: float, device: torch.device) -> None:
        self.name = name
        self.limit = limit
        self.calls = 0
        self.published = False
        self.scale = torch.zeros(1, dtype=torch.float32, device=device)
        self.inv_scale = torch.zeros(1, dtype=torch.float32, device=device)
        self.pending = torch.zeros((), dtype=torch.float32, device=device)
        # Windows whose observed amax exceeded scale * limit (values clamped).
        self.clipped = torch.zeros((), dtype=torch.int64, device=device)

    def _publish(self) -> None:
        self.scale.copy_(
            (self.pending.clamp_min(1.0e-30) * (bwd_fp8._BWD_MARGIN / self.limit)).reshape(1)
        )
        self.inv_scale.copy_(1.0 / self.scale)
        self.pending.zero_()
        self.published = True

    @staticmethod
    def _finite(amax: torch.Tensor) -> torch.Tensor:
        # A non-finite amax (an inf operand; NaN is already ignored by the
        # producers' amax) must not become the next window's scale: an inf scale
        # would cast every later value to zero for a whole window. Treat it as 0,
        # so the window keeps the amax of its finite calls. No host sync.
        return torch.nan_to_num(amax, nan=0.0, posinf=0.0, neginf=0.0)

    def warm(self, amax: torch.Tensor) -> None:
        """A BF16 warmup call: accumulate its standalone amax; publish after WARMUP calls."""
        self.calls += 1
        torch.maximum(self.pending, self._finite(amax), out=self.pending)
        if self.calls >= bwd_fp8._BWD_WARMUP_CALLS:
            self._publish()

    def steady(self, partials: torch.Tensor, snapshot: bool) -> torch.Tensor:
        """An FP8 call cast at inv_scale: returns the decode scale it used, then
        books its amax and publishes at the refresh cadence.

        partials: the producer's per-program amax partials (fp32 [n]). One
        launch (triton_preproc_fp8.site_steady) does what was 7-8 tiny torch
        ops, value for value: amax = partials.amax() (NaN-propagating);
        clipped += amax > scale * limit; pending = max(pending, amax with
        NaN/inf booked as 0); and the decode-scale copy. _publish (every
        REFRESH calls) keeps its torch ops.

        snapshot=True always returns a copy (a1: the decode scale is saved for
        the backward, and another training forward of the site, e.g. under
        gradient accumulation, could publish in place before that backward
        runs). snapshot=False returns the live tensor except on the publishing
        call, which gets a copy taken before the in-place publish (dh2: its
        GEMMs are enqueued right after this call, before the site's next call).
        """
        from generative_recommenders.ops.triton.triton_preproc_fp8 import site_steady

        publish = (self.calls + 1) % bwd_fp8._BWD_REFRESH_CALLS == 0
        decode = site_steady(
            partials,
            self.scale,
            self.pending,
            self.clipped,
            self.limit,
            snap=snapshot or publish,
        )
        self.calls += 1
        if publish:
            self._publish()
        if decode is not None:
            return decode
        return self.scale


_sites: Dict[str, _Site] = {}


def _site(role: str, k: int, device: torch.device) -> _Site:
    name = f"{role}.{k}"
    s = _sites.get(name)
    if s is None:
        limit = bwd_fp8._E4M3_MAX if role == "a1" else bwd_fp8._E5M2_MAX
        s = _Site(name, limit, device)
        _sites[name] = s
    return s


def site_diagnostics() -> Dict[str, Dict[str, float]]:
    """Host copies of every site's state (reads the device: never call in the step)."""
    out: Dict[str, Dict[str, float]] = {}
    for name, s in _sites.items():
        out[name] = {
            "calls": float(s.calls),
            "published": float(s.published),
            "scale": float(s.scale.item()),
            "clipped": float(s.clipped.item()),
        }
    return out


def reset_for_testing() -> None:
    global _ENABLED, _ARMED_LOGGED, _WGRAD_LOGGED, _INERT_WARNED
    global _GROUPED_ENABLED, _GROUPED_ARMED_LOGGED, _GROUPED_BROKEN, _GROUPED_LAST_DECLINE
    global _X2LB_REQUESTED
    _sites.clear()
    _X2LB_REQUESTED = None
    _ENABLED = None
    _ARMED_LOGGED = False
    _WGRAD_LOGGED = False
    _INERT_WARNED = False
    _GROUPED_ENABLED = None
    _GROUPED_ARMED_LOGGED = False
    _GROUPED_BROKEN = None
    _GROUPED_LAST_DECLINE = None


def _warn_inert_once(k: int, reason: str) -> None:
    """The gate is on and the scales are live, but this MLP's head ran the
    shipped path (per rank: emu is read from this rank's autotune state)."""
    global _INERT_WARNED
    if _INERT_WARNED:
        return
    _INERT_WARNED = True
    logger.warning(
        "[preproc-fp8-l2] %s=1 but MLP %d ran the shipped BF16 path (%s) rank=%d",
        ENV_VAR,
        k,
        reason,
        _rank(),
    )


def _amax(t: torch.Tensor) -> torch.Tensor:
    # max |t| as fp32 from two plain reductions (no [M, D] temporary); warmup only.
    t = t.detach()
    return torch.maximum(t.amax().float(), t.amin().float().neg())


# --------------------------------------------------------------------------- plan
class Slot:
    """One MLP's E-M3a plan for one forward: mode is "fp8" (pair 0 ran P2 and
    `payload` = (a1_rm, a1_tm, a1_decode_scale) waits for the tail), "warm"
    (shipped path; a1/dh2 warmup amaxes are collected) or None (shipped path).

    `prefilled` is the entry's modules/action_prefill.py PrefilledLinear (MNC
    item (31): the action MLP's last-pair casts + addmm, issued early) or None.
    A non-FP8 entry's node reuses it exactly as _GroupedLinearLNTailFunction
    does. The Slot list is the ONE per-entry non-tensor argument of
    _GroupedLinearLNTailFP8Function (eps, plans, *flat), so x_k stays at
    ctx.needs_input_grad[2 + 5k] (a second non-tensor argument would shift
    every index by one and drop gradients).

    `x2lb` is X2L-B's X2LBHandle (gin item (93), module docstring) that an
    "fp8" pair 0 published for the tail, or None (always None with that knob
    off). Single use, as `payload`: the tail's forward moves it into its ctx."""

    __slots__ = ("mode", "k", "train", "payload", "prefilled", "x2lb")

    def __init__(self, k: int) -> None:
        self.mode: Optional[str] = None
        self.k = k
        self.train = False
        self.payload: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        self.prefilled: Optional[Any] = None
        self.x2lb: Optional[Any] = None


def prefilled_slot(k: int, prefilled: Optional[Any]) -> Slot:
    """The plan of grouped-tail MLP k whose head already ran the shipped way
    (GroupedTail.head_prefilled, MNC item (31); only the action MLP is ever
    prefilled): BF16 (mode None), carrying its PrefilledLinear or None."""
    slot = Slot(k)
    slot.train = torch.is_grad_enabled()
    slot.prefilled = prefilled
    return slot


def _pairs_ok(mlp: Any) -> bool:
    try:
        pairs = mlp._pairs()
    except Exception:
        return False
    if len(pairs) != 2:
        return False
    (lin0, norm0), (lin1, norm1) = pairs
    return (
        isinstance(lin0, torch.nn.Linear)
        and isinstance(norm0, SwishLayerNorm)
        and isinstance(lin1, torch.nn.Linear)
        and isinstance(norm1, LayerNorm)
        and lin0.bias is not None
        and lin1.bias is not None
        and lin0.out_features == _HID
        and lin1.in_features == _HID
        and lin1.out_features == _OUT
        and tuple(norm0.weight.shape) == (_HID,)
        and tuple(norm0.bias.shape) == (_HID,)
        and norm0.weight.is_contiguous()
        and norm0.bias.is_contiguous()
    )


# P2's fp32 operation order when the shipped launcher's mode is not known: EMU 2
# is what the gin autotune pin for this key ((BLOCK_N 8, num_warps 4)) maps to,
# i.e. what the shipped v2 forward runs in production.
_DEFAULT_EMU: int = 2


def _emu_for(norm0: torch.nn.Module) -> Tuple[int, bool]:
    """(EMU mode for P2, whether it is the shipped launcher's own mode).

    The shipped v2 launcher's key is (h1 dtype, weight dtype, bias dtype); under
    the grouped tail's autocast(bf16) precondition h1 is bf16. The mode only
    decides which fp32 summation order P2 reproduces before rounding a1 to bf16
    (any of them is a valid LayerNorm); taking the shipped one makes P2's a1
    the e4m3 cast of exactly the bf16 a1 the shipped path would compute. When
    the old kernel's autotune pick is outside the v2 class (timing-dependent
    without the gin pins) the mode is unknown and P2 uses _DEFAULT_EMU, so FP8
    engagement does not depend on a per-rank autotune outcome."""
    emu = TLN._WLN_FWD_V2_MODE.get(
        (torch.bfloat16, norm0.weight.dtype, norm0.bias.dtype), -1
    )
    return (emu, True) if emu >= 0 else (_DEFAULT_EMU, False)


def run_head(
    k: int,
    mlp: Any,
    x: torch.Tensor,
    shipped: Callable[[Any, torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, Slot]:
    """Pair 0 of grouped-tail MLP k: through P2 when FP8 is ready, else shipped
    (collecting a1's warmup amax when it is still warming up)."""
    slot = Slot(k)
    slot.train = torch.is_grad_enabled()
    if k not in FP8_MLPS:
        return shipped(mlp, x), slot
    if x.dim() != 2 or x.shape[0] < _MIN_ROWS or not _pairs_ok(mlp):
        _warn_inert_once(
            k,
            f"input {tuple(x.shape)} (needs 2-D, >= {_MIN_ROWS} rows) or MLP pairs "
            "not (Linear->SwishLayerNorm(256), Linear(256->512)->LayerNorm)",
        )
        return shipped(mlp, x), slot
    (lin0, norm0), _ = mlp._pairs()
    a1 = _site("a1", k, x.device)
    dh2 = _site("dh2", k, x.device)
    emu, shipped_emu = _emu_for(norm0)
    if a1.published and dh2.published:
        if not mlp._f1_reported:
            mlp._report_once(True, x)
        slot.mode = "fp8"
        # E-X2b: pair 0 may take x's EC dh port (None, None when off). The
        # "warm" branch below claims inside `shipped` (= _run_head).
        rec, port = _xs.claim(x, lin0.weight)
        y = _LinearSwishLNEmitFunction.apply(
            x,
            lin0.weight,
            lin0.bias,
            norm0.weight,
            norm0.bias,
            norm0._eps,
            k,
            emu,
            slot.train,
            slot,
            rec,
            port,
        )
        _report_armed_once(k, x.shape[0], emu, shipped_emu)
        return y, slot
    y = shipped(mlp, x)
    slot.mode = "warm"
    if slot.train and not a1.published:
        a1.warm(_amax(y))
    return y, slot


def _report_armed_once(k: int, rows: int, emu: int, shipped_emu: bool) -> None:
    global _ARMED_LOGGED
    if _ARMED_LOGGED:
        return
    _ARMED_LOGGED = True
    logger.info(
        "%s first FP8 call: mlp=%d N=%d a1=e4m3(rm+tm) dh2=e5m2 wgrad=nn-b emu=%d%s "
        "a1_scale_calls=%d rank=%d",
        ARMED_PREFIX,
        k,
        rows,
        emu,
        "" if shipped_emu else "(default: shipped mode unknown)",
        _sites["a1.%d" % k].calls,
        _rank(),
    )


def _match(g: Optional[torch.Tensor], dt: torch.dtype) -> Optional[torch.Tensor]:
    return None if g is None else g.to(dt)


def _autocast_casts(
    x: torch.Tensor, lin_w: torch.Tensor, lin_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # == _LinearNormFunction.forward's casts (see its AUTOCAST note).
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        amp_dtype = torch.get_autocast_dtype(dev)
        return x.to(amp_dtype), lin_w.to(amp_dtype), lin_b.to(amp_dtype)
    return x, lin_w, lin_b


def _autocast_weight_casts(
    device: torch.device, lin_w: torch.Tensor, lin_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # The same weight casts, without touching an activation.
    if torch.is_autocast_enabled(device.type):
        amp_dtype = torch.get_autocast_dtype(device.type)
        return lin_w.to(amp_dtype), lin_b.to(amp_dtype)
    return lin_w, lin_b


def _quantize_w2(
    device: torch.device, lin_w: torch.Tensor, lin_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (w8 [512, 256] e4m3, w8_t [256, 512] e4m3, w_scale [1] f32, lin_b_c):
    bwd_fp8.quantize_weight of the autocast-cast W2, those bytes transposed
    (== w8.t().contiguous()) and the autocast-cast b2, bit for bit.

    One launch (triton_preproc_fp8.quantize_w2_dual) when the autocast casts
    are fp32 -> bf16 (fp32 master weights under autocast(bf16), production)
    or W2 already is bf16; otherwise the torch ops plus the transpose."""
    from generative_recommenders.ops.triton.triton_preproc_fp8 import (
        quantize_w2_dual,
    )

    if (
        torch.is_autocast_enabled(device.type)
        and torch.get_autocast_dtype(device.type) == torch.bfloat16
        and lin_w.dtype == torch.float32
        and lin_b.dtype == torch.float32
        and tuple(lin_w.shape) == (_OUT, _HID)
        and lin_w.is_contiguous()
        and lin_b.dim() == 1
        and lin_b.is_contiguous()
    ):
        w8, w8_t, w_scale, lin_b_c = quantize_w2_dual(lin_w, lin_b, cast_bf16=True)
        assert lin_b_c is not None
        return w8, w8_t, w_scale, lin_b_c
    lin_w_c, lin_b_c = _autocast_weight_casts(device, lin_w, lin_b)
    if (
        lin_w_c.dtype == torch.bfloat16
        and tuple(lin_w_c.shape) == (_OUT, _HID)
        and lin_w_c.is_contiguous()
    ):
        w8, w8_t, w_scale, _ = quantize_w2_dual(lin_w_c)
        return w8, w8_t, w_scale, lin_b_c
    w8, w_scale = bwd_fp8.quantize_weight(lin_w_c)
    return w8, w8.t().contiguous(), w_scale, lin_b_c


class _LinearSwishLNEmitFunction(torch.autograd.Function):
    """Pair 0 with the SwishLN forward replaced by P2. Returns a stride-0 bf16
    proxy of a1's shape; the e4m3 a1 goes to `slot.payload`. backward ==
    _LinearNormFunction.backward with is_swish=True (no residual).

    Indexed-X L1 (ops/ec_indexed_l1.py, DLRM_HSTU_FP8_EC_INDEXED_L1, default off):
    when x is an EC placeholder (it carries _IX_ATTR), h = _ecix_fwd(U, idx, W1_c,
    b1_c) reads X through the EC index, the node saves (U, idx) instead of X, and
    the backward's dW1 sums over all M tokens from (U, idx); dX = dh @ W1_c is
    unchanged. Any in-Function decline rebuilds the exact X and runs this body.

    X2L-B (item (93), module docstring; knob off: absent): a train forward
    publishes ctx.x2lb; when the tail fused this MLP, the backward takes dh1,
    dW, dB and db1 from it instead of running the SwishLN backward."""

    @staticmethod
    # pyre-ignore[14]
    def forward(  # pyre-ignore[3]
        ctx,  # pyre-ignore[2]
        x: torch.Tensor,
        lin_w: torch.Tensor,
        lin_b: torch.Tensor,
        norm_w: torch.Tensor,
        norm_b: torch.Tensor,
        eps: float,
        k: int,
        emu: int,
        train: bool,
        slot: Slot,
        ex2b_rec: Optional["_xs.PortRecord"] = None,
        ex2b_port: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from generative_recommenders.ops.triton.triton_preproc_fp8 import (
            swishln256_fwd_emit,
        )

        x_in = x
        ixu = None
        if getattr(x, _IX_ATTR, None) is not None:  # Indexed-X placeholder (never with the knob off)
            w_c, b_c = _autocast_weight_casts(x.device, lin_w, lin_b)  # == the shipped casts
            ixu = _ix().fwd(x, w_c, b_c, lin_w.dtype, "_LinearSwishLNEmitFunction", train=train)
        if ixu is not None and ixu.h is not None:  # indexed: X is never materialized
            # IX-FP8F (train only): table = Ud and w_bwd = Wd, so the saved W1, dX = dh @ W1 and E-X2b's bound W1
            # use the dequantized operands; w_bwd is None on every bf16 path (lin_w_c = w_c, as before).
            x_c, lin_w_c, lin_b_c, h = ixu.table, (w_c if ixu.w_bwd is None else ixu.w_bwd), b_c, ixu.h
        else:
            cw, cb = (lin_w, lin_b) if ixu is None else (w_c, b_c)  # decline: reuse the casts above
            if ixu is not None:
                x = ixu.x  # in-Function decline: the exact rebuilt X
            x = x.contiguous()
            x_c, lin_w_c, lin_b_c = _autocast_casts(x, cw, cb)
            h = torch.addmm(lin_b_c, x_c, lin_w_c.t())
        site = _site("a1", k, h.device)
        # Eval (train=False) skips the token-major copy: only the L2 wgrad reads it.
        # The amax partials go straight to the site's bookkeeping launch (eval
        # books nothing, so its partials are never reduced).
        a1_rm, a1_tm, mean, rstd, partials = swishln256_fwd_emit(
            h, norm_w, norm_b, eps, site.inv_scale, emu, tm=train, return_partials=True
        )
        decode = site.steady(partials, snapshot=True) if train else site.scale
        slot.payload = (a1_rm, a1_tm, decode)
        # Indexed-X: slot 0 is U [U, 512] instead of X, slot 8 is the member's int64 idx.
        ctx.save_for_backward(x_c, lin_w_c, lin_b_c, h, norm_w, norm_b, mean, rstd,
                              *(() if ixu is None or ixu.h is None else (ixu.idx,)))
        if train and _x2lb_requested():
            # X2L-B (item (93)): the handle the tail fuses into (saved[3:8] are
            # read from there); None on a publish-time decline. Knob off: skipped.
            ctx.x2lb = slot.x2lb = _x2lb_publish(ctx, k, h, train)
        ctx.ix = None if ixu is None else ixu.call  # IXCall: metadata + ledger, no tensors
        # E-X2b: as _LinearNormFunction.forward (W1 handed to the EC backward). With
        # Indexed-X the record's tagged tensor is the caller's placeholder.
        ctx.ex2b = None if ex2b_port is None else _xs.bind(ex2b_rec, x_c if ixu is None else x_in, lin_w_c)
        ctx.grad_dtypes = (x.dtype, lin_w.dtype, lin_b.dtype)
        ctx.eps = eps
        ctx.BLOCK_D = compute_BLOCK_D(h)
        # One storage element, zero strides: an autograd placeholder for a1. The
        # grouped tail reads the FP8 payload instead and never this tensor.
        return torch.empty_strided(
            (h.shape[0], _HID), (0, 0), dtype=h.dtype, device=h.device
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(  # pyre-ignore[3]
        ctx, dy: torch.Tensor  # pyre-ignore[2]
    ) -> Tuple[Optional[torch.Tensor], ...]:
        saved = ctx.saved_tensors
        x, lin_w, lin_b, h, norm_w, norm_b, mean, rstd = saved[:8]  # with Indexed-X, x is U [U, 512]
        # X2L-B (item (93)): when the tail fused this MLP, dy is the handle's
        # sentinel and take() returns (dh1, dW, dB, db1); it raises on any
        # handshake mismatch. No handle (knob off, not published): None.
        hd = getattr(ctx, "x2lb", None)
        got = None if hd is None else hd.take(dy)
        if got is not None:
            dh, dnorm_w, dnorm_b, dh_colsum = got
        else:
            dh, dnorm_w, dnorm_b, dh_colsum = triton_weighted_layer_norm_bwd_dxsum(
                dy=dy.contiguous(),
                x=h,
                weight=norm_w,
                bias=norm_b,
                mean=mean,
                rstd=rstd,
                eps=ctx.eps,
                BLOCK_D=ctx.BLOCK_D,
                is_swish=True,
            )
        needs = ctx.needs_input_grad
        # E-X2b: as _LinearNormFunction.backward (dh through the EC port).
        routed = ctx.ex2b is not None and _xs.route(ctx, ctx.ex2b, dh, len(needs) - 1)
        dx = dh @ lin_w if needs[0] and not routed else None
        ix = ctx.ix
        if ix is not None and ix.indexed:
            # Indexed-X: dW1 over ALL M tokens from (U, idx), already in W1's dtype.
            dlin_w = _ix().wgrad(ix, dh, x, saved[8], ctx.grad_dtypes[1]) if needs[1] else None
        else:
            dlin_w = dh.t() @ x if needs[1] else None
        if ix is not None:
            ix.ledger.note_bwd(routed)  # backward-time consumer accounting (EC backward check)
        dlin_b = dh_colsum if needs[2] else None
        x_dt, w_dt, b_dt = ctx.grad_dtypes
        out = (
            _match(dx, x_dt),
            _match(dlin_w, w_dt),
            _match(dlin_b, b_dt),
            _match(dnorm_w, norm_w.dtype),
            _match(dnorm_b, norm_b.dtype),
            None,
            None,
            None,
            None,
            None,
            None,  # ex2b_rec
            dh if routed else None,  # ex2b_port: E-X2b's dh
        )
        return out[: len(needs)]


def _wgrad_nn_b(
    dh8: torch.Tensor,
    a1_tm: torch.Tensor,
    dh_scale: torch.Tensor,
    a1_scale: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    """dW2 [512, 256] fp32 = dh2^T @ a1 from e5m2 dh2 row-major and e4m3 a1 token-major."""
    from generative_recommenders.ops.hipblaslt_fp8_nn import fp8_wgrad_nn
    from generative_recommenders.ops.triton.fp8_wgrad import transpose_fp8_token_major
    from generative_recommenders.ops.triton.triton_fp8 import _scaled_mm

    dw_t = fp8_wgrad_nn(dh8, a1_tm, dh_scale, a1_scale, out_dtype=torch.float32)
    if dw_t is not None:
        return dw_t.t().contiguous(), "nn"
    # No NN algorithm in this library: token-major dh2 [512, Mp] (same 128-token
    # padding as a1_tm, zero tail) and the TN GEMM over K = Mp.
    dh_tm = transpose_fp8_token_major(dh8, pad_to=128)
    assert dh_tm.shape[1] == a1_tm.shape[1], (dh_tm.shape, a1_tm.shape)
    dw = _scaled_mm(
        dh_tm,
        a1_tm.t(),
        dh_scale,
        a1_scale,
        out_dtype=torch.float32,
    )
    return dw, "tn_fallback"


# --------------------------------------------------------------------------- stacked backward
def _grouped_enabled() -> bool:
    """The stacked backward's kill switch GROUPED_ENV_VAR (default on, "0" =
    off), read once per process."""
    global _GROUPED_ENABLED
    if _GROUPED_ENABLED is None:
        _GROUPED_ENABLED = os.environ.get(GROUPED_ENV_VAR, "1") != "0"
    return _GROUPED_ENABLED


def _disable_grouped(where: str, error: BaseException) -> None:
    """Latch the stacked backward off for the process (the first error wins)
    with one WARNING carrying the traceback; the P3 path runs from then on."""
    global _GROUPED_BROKEN
    if _GROUPED_BROKEN is not None:
        return
    lines = str(error).strip().splitlines()
    _GROUPED_BROKEN = (
        f"{where}: {type(error).__name__}: {lines[0][:200] if lines else ''}"
    )
    logger.warning(
        "[preproc-fp8-l2-grouped] DISABLED (%s); the P3 LayerNorm(512) backward "
        "runs for the FP8 MLPs instead rank=%d",
        _GROUPED_BROKEN,
        _rank(),
        exc_info=error,
    )


# Per MLP: (dh, dnorm_w, dnorm_b, dh_colsum, amax partials); dh is e5m2 with
# fp32 partials for an FP8 MLP, bf16 with None for the action MLP.
_GroupedOut = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]


def _grouped_bwd(
    ctx: Any,  # pyre-ignore[2]
    dy: torch.Tensor,
    saved: Tuple[torch.Tensor, ...],
) -> Optional[List[Optional[_GroupedOut]]]:
    """The stacked LayerNorm(512) backward of all three MLPs (module docstring,
    STACKED BACKWARD) as ONE ln_bwd_grouped_fp8.ln_bwd_grouped3_fp8 call, or
    None when it does not take this backward (the caller then runs P3 / the
    per-pair call per MLP). Never raises for a host-side error: the path is
    latched off instead, and every output is fresh, so the caller's fallback
    recomputes from intact inputs."""
    global _GROUPED_ARMED_LOGGED, _GROUPED_LAST_DECLINE
    dy = dy.detach()
    modes = ctx.modes
    if ctx.num_in != 3 or modes[0] != "fp8" or modes[1] != "fp8" or modes[2] == "fp8":
        _GROUPED_LAST_DECLINE = f"modes {tuple(modes)}"
        return None
    if not _grouped_enabled():
        _GROUPED_LAST_DECLINE = f"{GROUPED_ENV_VAR}=0"
        return None
    if _GROUPED_BROKEN is not None:
        _GROUPED_LAST_DECLINE = f"latched off: {_GROUPED_BROKEN}"
        return None
    # Per MLP the saved run is (..., h, norm_w, norm_b, mean, rstd): 9 tensors
    # for an "fp8" entry (h at +4), 8 for any other (h at +3).
    ops: List[List[torch.Tensor]] = [[], [], [], [], []]
    for k in range(3):
        o = ctx.starts[k] + (4 if modes[k] == "fp8" else 3)
        for i, t in enumerate(saved[o : o + 5]):
            ops[i].append(t.detach())
    hs, norm_ws, norm_bs, means, rstds = ops
    # PR #7 off, latched off or not on gfx950 owns none of the three calls
    # (E-M2's predicate below would say so): decline before any FlyDSL import.
    arch = getattr(torch.cuda.get_device_properties(dy.device), "gcnArchName", "")
    if (
        not TLN._wln_bwd_flydsl_enabled()
        or TLN._WLN_BWD_FLYDSL_BROKEN is not None
        or not arch.startswith("gfx950")
    ):
        _GROUPED_LAST_DECLINE = "PR #7 / E-M2 FlyDSL predicate: PR #7 off/latched/not gfx950"
        return None
    try:
        # Before E-M2's predicate, which imports ln_bwd_grouped itself and
        # would latch and log E-M2's FlyDSL path on an import error: this
        # import (FlyDSL, ln_bwd_grouped, ln_bwd_grouped_fp8) latches only
        # the stacked path.
        from generative_recommenders.ops.flydsl import ln_bwd_grouped_fp8 as f8
    except Exception as error:
        _disable_grouped("import", error)
        _GROUPED_LAST_DECLINE = f"latched off: {_GROUPED_BROKEN}"
        return None
    # PR #7 owns the three per-pair calls: E-M2's FlyDSL dispatch predicate
    # itself (no reason <=> PR #7 takes all three, E-M2's FlyDSL latch unset,
    # ln_bwd_grouped.grouped_reason covers them). E-M2's gate is not asked.
    from generative_recommenders.ops import preproc_grouped_tail as pgt

    pr7_owns, why = pgt._flydsl_grouped_reason(dy, hs, norm_ws, norm_bs, means, rstds)
    if not pr7_owns or why is not None or pgt._BWD_FLYDSL_BROKEN is not None:
        _GROUPED_LAST_DECLINE = f"PR #7 / E-M2 FlyDSL predicate: {why}"
        return None
    inv_scales: List[Optional[torch.Tensor]] = [
        _site("dh2", 0, dy.device).inv_scale,
        _site("dh2", 1, dy.device).inv_scale,
        None,
    ]
    name: Optional[str] = None
    try:
        why = f8.grouped_fp8_reason(dy, hs, norm_ws, means, rstds, inv_scales)
        if why is not None:
            _GROUPED_LAST_DECLINE = why
            return None
        if not _GROUPED_ARMED_LOGGED:
            name = f8.stacked_kernel_name()
        outs = f8.ln_bwd_grouped3_fp8(
            dy, hs, norm_ws, norm_bs, means, rstds, inv_scales
        )
    except Exception as error:
        _disable_grouped("dispatch", error)
        _GROUPED_LAST_DECLINE = f"latched off: {_GROUPED_BROKEN}"
        return None
    _GROUPED_LAST_DECLINE = None
    if name is not None:
        _GROUPED_ARMED_LOGGED = True
        logger.info(
            "%s kernel=%s N=%d rank=%d",
            GROUPED_ARMED_PREFIX,
            name,
            dy.shape[0],
            _rank(),
        )
    return list(outs)


class _GroupedLinearLNTailFP8Function(torch.autograd.Function):
    """_GroupedLinearLNTailFunction with E-M3a's FP8 last Linear for the MLPs
    whose plan slot is "fp8". Every other MLP runs that Function's forward and
    per-pair backward statement for statement (and a "warm" one also feeds
    dh2's warmup amax); the opt-in E-M2 grouped backward is not used here.
    In the FP8 steady state the three LayerNorm(512) backwards come from ONE
    stacked call instead (_grouped_bwd; module docstring, STACKED BACKWARD),
    and the per-MLP GEMMs and scale bookkeeping run on its outputs.

    Forward args: eps, plans (one Slot per MLP; it also carries the entry's
    PrefilledLinear, MNC item (31)), then per MLP the five tensors
    (x_k, lin_w_k, lin_b_k, norm_w_k, norm_b_k); for an "fp8" MLP x_k is pair
    0's proxy and the operands come from its slot. x_k's
    ctx.needs_input_grad slot is 2 + 5k, as in _GroupedLinearLNTailFunction.

    X2L-B (item (93), module docstring): ctx.x2lb holds the FP8 MLPs' handles
    (None with the knob off); for a handle the FP8 dgrad statement becomes
    try_fused, whose sentinel is x_k's gradient, else the dgrad runs as before.
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(  # pyre-ignore[3]
        ctx,  # pyre-ignore[2]
        eps: float,
        plans: List[Slot],
        *flat: torch.Tensor,
    ) -> torch.Tensor:
        from generative_recommenders.ops.triton.triton_fp8 import _scaled_mm
        from generative_recommenders.ops.triton.triton_preproc_tail import (
            grouped_ln_residual_fwd,
        )

        num_in = len(flat) // 5
        if num_in not in (2, 3) or len(flat) != 5 * num_in or len(plans) != num_in:
            raise ValueError(
                f"grouped LN tail (FP8): need 2 or 3 x 5 tensors and one plan each, "
                f"got {len(flat)} tensors / {len(plans)} plans"
            )
        saved: List[torch.Tensor] = []
        offsets: List[int] = []
        hs: List[torch.Tensor] = []
        norm_ws: List[torch.Tensor] = []
        norm_bs: List[torch.Tensor] = []
        grad_dtypes: List[Tuple[torch.dtype, torch.dtype, torch.dtype]] = []
        block_ds: List[int] = []
        modes: List[Optional[str]] = []
        x2lbs: Optional[List[Any]] = None  # X2L-B (item (93)): per-MLP handles, or None
        for k in range(num_in):
            x, lin_w, lin_b, norm_w, norm_b = flat[5 * k : 5 * k + 5]
            slot = plans[k]
            offsets.append(len(saved))
            if slot.mode == "fp8":
                payload = slot.payload
                if payload is None:
                    raise RuntimeError(
                        f"grouped LN tail (FP8): MLP {k} planned FP8 without a payload"
                    )
                slot.payload = None
                a1_rm, a1_tm, a1_scale = payload
                del payload
                if slot.x2lb is not None:
                    # X2L-B (item (93)): pair 0's handle, single use as the
                    # payload; x is the proxy. None on an attach decline.
                    if x2lbs is None:
                        x2lbs = [None] * num_in
                    x2lbs[k], slot.x2lb = _x2lb().attach(slot.x2lb, a1_rm, x), None
                # e4m3 W2 [512, 256] from its own amax, plus the transposed bytes
                # the dgrad reads, in one launch (== quantize_weight(lin_w_c) and
                # w8.t().contiguous()).
                w8, w8_t, w_scale, lin_b_c = _quantize_w2(a1_rm.device, lin_w, lin_b)
                h = _scaled_mm(
                    a1_rm,
                    w8.t(),
                    a1_scale,
                    w_scale,
                    out_dtype=torch.bfloat16,
                    bias=lin_b_c,
                )
                del a1_rm, w8
                saved += [a1_tm, w8_t, w_scale, a1_scale, h, norm_w, norm_b]
            else:
                # == _GroupedLinearLNTailFunction.forward's body, including MNC
                # item (31): an entry whose last-pair casts + addmm were issued
                # early (the action MLP) reuses them; take() returns None on any
                # mismatch -> computed here as before.
                x = x.contiguous()
                pf = slot.prefilled
                got = None if pf is None else pf.take(x, lin_w, lin_b)
                if got is not None:
                    x_c, lin_w_c, lin_b_c, h = got
                else:
                    x_c, lin_w_c, lin_b_c = _autocast_casts(x, lin_w, lin_b)
                    h = torch.addmm(lin_b_c, x_c, lin_w_c.t())
                saved += [x_c, lin_w_c, lin_b_c, h, norm_w, norm_b]
            if k > 0 and (h.dtype != hs[0].dtype or h.shape != hs[0].shape):
                raise RuntimeError(
                    f"grouped LN tail: MLP {k} norm input {tuple(h.shape)}/"
                    f"{h.dtype} != MLP 0's {tuple(hs[0].shape)}/{hs[0].dtype}"
                )
            hs.append(h)
            norm_ws.append(norm_w)
            norm_bs.append(norm_b)
            grad_dtypes.append((x.dtype, lin_w.dtype, lin_b.dtype))
            block_ds.append(compute_BLOCK_D(h))
            modes.append(slot.mode)
        out, means, rstds = grouped_ln_residual_fwd(hs, norm_ws, norm_bs, eps)
        packed: List[torch.Tensor] = []
        starts: List[int] = []
        for k in range(num_in):
            end = offsets[k + 1] if k + 1 < num_in else len(saved)
            starts.append(len(packed))
            packed += saved[offsets[k] : end] + [means[k], rstds[k]]
        ctx.save_for_backward(*packed)
        ctx.starts = starts
        ctx.num_in = num_in
        ctx.modes = modes
        ctx.trains = [p.train for p in plans]
        ctx.grad_dtypes = grad_dtypes
        ctx.block_ds = block_ds
        ctx.eps = eps
        ctx.x2lb = x2lbs
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(  # pyre-ignore[3]
        ctx,  # pyre-ignore[2]
        dy: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        if ctx.x2lb is None:  # X2L-B off or no handle (always with the knob off): the body below, nothing else
            return _GroupedLinearLNTailFP8Function._backward(ctx, dy)
        try:
            return _GroupedLinearLNTailFP8Function._backward(ctx, dy)
        except BaseException:
            # X2L-B (item (93)): this backward failed; the dh1 / finals it already
            # fused into the handles must not outlive it while the graph is kept.
            _x2lb().release(ctx.x2lb)
            raise

    @staticmethod
    def _backward(  # pyre-ignore[3]
        ctx,  # pyre-ignore[2]
        dy: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        global _WGRAD_LOGGED
        from generative_recommenders.ops.triton.triton_fp8 import _scaled_mm
        from generative_recommenders.ops.triton.triton_preproc_fp8 import (
            ln512_bwd_dxsum_emit,
        )

        num_in: int = ctx.num_in
        saved = ctx.saved_tensors
        needs = ctx.needs_input_grad
        dy = dy.contiguous()
        grads: List[Optional[torch.Tensor]] = [None] * (5 * num_in)
        # STACKED BACKWARD: the three LayerNorm(512) dx-sum backwards as ONE
        # call (FP8 steady state), or None -> each MLP's own producer below.
        grouped = _grouped_bwd(ctx, dy, saved)
        # Reverse forward order (action, additional, content), as the shipped tail.
        for k in reversed(range(num_in)):
            s = ctx.starts[k]
            n_x, n_w, n_b = needs[2 + 5 * k], needs[3 + 5 * k], needs[4 + 5 * k]
            x_dt, w_dt, b_dt = ctx.grad_dtypes[k]
            if ctx.modes[k] == "fp8":
                a1_tm, w8_t, w_scale, a1_scale, h, norm_w, norm_b, mean, rstd = saved[
                    s : s + 9
                ]
                site = _site("dh2", k, dy.device)
                if grouped is not None:
                    got = grouped[k]
                    # Drop the list's reference: this k's dh2 is freed after
                    # its GEMMs, as the P3 path frees its own.
                    grouped[k] = None
                    assert got is not None and got[4] is not None
                    dh8, dnorm_w, dnorm_b, dh_colsum, partials = got
                    del got
                else:
                    dh8, dnorm_w, dnorm_b, dh_colsum, partials = ln512_bwd_dxsum_emit(
                        dy, h, norm_w, mean, rstd, site.inv_scale, return_partials=True
                    )
                dh_scale = (
                    site.steady(partials, snapshot=False)
                    if ctx.trains[k]
                    else site.scale
                )
                dx = None
                if n_x:
                    hd = None if ctx.x2lb is None else ctx.x2lb[k]
                    if hd is not None:
                        # X2L-B (item (93)): S1 + PR #7's finalize into the
                        # handle; dx = its sentinel. None (declined, latched)
                        # -> the shipped dgrad below.
                        dx = _x2lb().try_fused(hd, dh8, w8_t, dh_scale, w_scale, x_dt)
                    if dx is None:
                        # B = W2 [512, 256] column-major: the e4m3 W2 bytes transposed
                        # (written by the forward's W2 cast).
                        dx = _scaled_mm(
                            dh8,
                            w8_t.t(),
                            dh_scale,
                            w_scale,
                            out_dtype=torch.bfloat16,
                        )
                dlin_w = None
                if n_w:
                    dlin_w, path = _wgrad_nn_b(dh8, a1_tm, dh_scale, a1_scale)
                    if not _WGRAD_LOGGED:
                        _WGRAD_LOGGED = True
                        logger.info(
                            "%s first FP8 backward: mlp=%d N=%d wgrad=%s rank=%d",
                            ARMED_PREFIX,
                            k,
                            dh8.shape[0],
                            path,
                            _rank(),
                        )
                dlin_b = dh_colsum if n_b else None
                del dh8
            else:
                x, lin_w, lin_b, h, norm_w, norm_b, mean, rstd = saved[s : s + 8]
                if grouped is not None:
                    # The stacked call's bf16 dh and sums: bitwise the per-pair
                    # call's below (PR #7 owns it whenever the stack runs).
                    got = grouped[k]
                    grouped[k] = None
                    assert got is not None
                    dh, dnorm_w, dnorm_b, dh_colsum, _ = got
                    del got
                else:
                    # == _GroupedLinearLNTailFunction.backward's per-pair body.
                    dh, dnorm_w, dnorm_b, dh_colsum = triton_weighted_layer_norm_bwd_dxsum(
                        dy=dy,
                        x=h,
                        weight=norm_w,
                        bias=norm_b,
                        mean=mean,
                        rstd=rstd,
                        eps=ctx.eps,
                        BLOCK_D=ctx.block_ds[k],
                        is_swish=False,
                    )
                if ctx.modes[k] == "warm" and ctx.trains[k]:
                    site = _site("dh2", k, dy.device)
                    if not site.published:
                        site.warm(_amax(dh))
                dx = dh @ lin_w if n_x else None
                dlin_w = dh.t() @ x if n_w else None
                dlin_b = dh_colsum if n_b else None
                del dh
            grads[5 * k : 5 * k + 5] = [
                _match(dx, x_dt),
                _match(dlin_w, w_dt),
                _match(dlin_b, b_dt),
                _match(dnorm_w, norm_w.dtype),
                _match(dnorm_b, norm_b.dtype),
            ]
            del dx, dlin_w, dlin_b, dnorm_w, dnorm_b, dh_colsum
        return (None, None, *grads)


def grouped_tail_apply(
    eps: float, plans: List[Slot], flat: List[torch.Tensor]
) -> torch.Tensor:
    return _GroupedLinearLNTailFP8Function.apply(eps, plans, *flat)


# --------------------------------------------------------------------------- pre-RUN_START warm-up
# Rows of warmup_kernels()'s synthetic operands. Triton (3.6) compiles one
# variant per class of each integer argument (== 1, divisible by 16, other),
# per 16-byte pointer alignment and, on AMD, per tensor storage within or
# above 2 GiB (see the P3 block in warmup_kernels). The real token count M
# changes every step, so every class a real armed run reaches is compiled
# here, on fresh (aligned) tensors as in the step:
#   P2 train (a1 token-major copy on): the 64-token kernel (iteration 51) does
#     not specialize N or n_valid, and its other integer arguments are always
#     multiples of 16 (Mp / 4 words with Mp % 128 == 0, a partials grid % 128
#     == 0), so the first row compiles the one variant (per EMU) every M uses
#   P2 eval  (no token-major copy):    N % 16 x n_blocks = cdiv(N, 16)
#   P3: N % 16 (its strides and D are 512).
#          rows   N%16   cdiv(N,16)%16 (eval)
#   2^17          0      0
#   2^17 + 16     0      1
#   2^17 - 1      15     0
#   2^17 + 1      1      1
# All are above 2^16 rows: a1's site_steady then gets more than 4096 P2
# partials (its large-partials config, as real batches) and P3 its full
# sms * 8 grid (dh2's site_steady and the finalize see production counts).
# The stacked backward's amax partials are one per block of ln_bwd's
# persistent grid, also sms * 8 at every FP8 M (>= 2^16 rows), so its dh2
# site_steady is this same specialization (MI350X: 2048 partials, e5m2
# LIMIT, BLOCK 4096) and the stacked path needs no warm-up here.
# The FP8 GEMMs do not need production M: hipBLASLt loads one code object per
# problem type (Arbor actckpt iter 51: after one call at M = 16 or 131089, the
# first call at M = 2.34M opened no library file and took 1.3-2.0 ms, the
# per-new-M heuristic cost; cold it took 2.45 s / 0.59 s / 0.28 s for the
# fwd / dgrad / NN wgrad).
_WARM_BASE: int = 1 << 17
_WARM_ROWS: Tuple[int, ...] = (_WARM_BASE, _WARM_BASE + 16, _WARM_BASE - 1, _WARM_BASE + 1)


def _warm_emus() -> List[int]:
    """The P2 EMU modes this process can run: _DEFAULT_EMU, any mode the
    shipped v2 launcher already recorded, and the mode the autotune pin of
    the shipped SwishLN(256) forward maps to (apply_autotune_pins seeds that
    pick before this runs; with the gin pins it is EMU 2 = _DEFAULT_EMU).
    Reads only."""
    emus = {_DEFAULT_EMU}
    emus.update(e for e in TLN._WLN_FWD_V2_MODE.values() if e >= 0)
    try:
        cache = getattr(TLN._weighted_layer_norm_fwd, "cache", None) or {}
        for key, cfg in cache.items():
            if str(key).startswith("(256,"):
                e = TLN._WLN_FWD_V2_EMU.get(
                    (cfg.kwargs.get("BLOCK_N"), cfg.num_warps), -1
                )
                if e >= 0:
                    emus.add(e)
    except Exception:
        pass
    return sorted(emus)


def _warm_site_ops(
    t: torch.Tensor,
    pending: torch.Tensor,
    scale: torch.Tensor,
    inv_scale: torch.Tensor,
    limit: float,
) -> None:
    """_amax, _Site.warm's booking and _Site._publish, op for op, on private
    tensors: the first use of their torch kernels otherwise lands in the
    armed BF16 steps (iter-51 lab: the publishing step 4 took ~115-125 ms more
    than unarmed)."""
    torch.maximum(pending, _Site._finite(_amax(t)), out=pending)
    scale.copy_((pending.clamp_min(1.0e-30) * (bwd_fp8._BWD_MARGIN / limit)).reshape(1))
    inv_scale.copy_(1.0 / scale)
    pending.zero_()


def warmup_kernels(
    device: torch.device,
    quiet_stderr: Optional[Callable[[], Any]] = None,
) -> Tuple[int, float]:
    """_warmup_kernels_impl (below), plus: ops/hipblaslt_fp8_nn keys its
    heuristic cache by the exact token count by default ("exact"); in its
    non-default "shape" mode an entry made at a warm-up token count would be
    reused by every real M, so the entries the warm-up added are removed again
    in that mode (also when the warm-up fails part way)."""
    try:
        from generative_recommenders.ops import hipblaslt_fp8_nn as _hnn

        nn_keys0: Optional[set] = set(_hnn._algos)
    except Exception:
        _hnn, nn_keys0 = None, None
    try:
        return _warmup_kernels_impl(device, quiet_stderr)
    finally:
        if (
            _hnn is not None
            and nn_keys0 is not None
            and getattr(_hnn, "_ALGO_CACHE_MODE", "exact") != "exact"
        ):
            for key in set(_hnn._algos) - nn_keys0:
                _hnn._algos.pop(key, None)


def _warmup_kernels_impl(
    device: torch.device,
    quiet_stderr: Optional[Callable[[], Any]] = None,
) -> Tuple[int, float]:
    """Launch every kernel an armed E-M3a step and eval forward use, once per
    Triton specialization a real run reaches (see _WARM_ROWS), plus E-M3a's FP8
    GEMMs, on synthetic operands, so their JIT compiles and first-use library
    loads happen before RUN_START (dlrm_v4/train/blas_warmup.py calls this,
    only when enabled()). Returns (calls issued, seconds incl. a final sync).

    Kernels: P2 swishln256_fwd_emit (train: token-major on; eval: off) for
    every EMU of _warm_emus(); site_steady for a1 (e4m3 limit) and dh2 (e5m2
    limit); P3 ln512_bwd_dxsum_emit (+ the shipped finalize it launches);
    _quantize_w2 under autocast(bf16) from fp32 W2/b2 (quantize_w2_dual, as in
    production); _scaled_mm forward (e4m3 x e4m3 + bf16 bias) and dgrad
    (e5m2 x e4m3) through triton_fp8._scaled_mm, and the NN-b wgrad through
    _wgrad_nn_b (hipblaslt_fp8_nn, or its TN fallback), all as the armed path
    issues them; and the torch ops of a site's BF16 warmup calls and publish
    (_warm_site_ops). Nothing of the STACKED BACKWARD: no FlyDSL import here,
    before the DataLoader forks (its kernels: prebuild_grouped, after them;
    its dh2 site_steady specialization is P3's, warmed here).

    TOUCHES NO TRAINING STATE. No _Site is created or advanced: the launchers
    get private throwaway scale/pending/clipped tensors. No module, parameter
    or slot; no RNG draw (torch.full/ones/zeros only); the _ARMED_LOGGED /
    _WGRAD_LOGGED / _INERT_WARNED latches stay unset, so the first real FP8
    step still logs its lines. Caches it does fill, none read differently by
    the real run: Triton's compiled-kernel caches; hipblaslt_fp8_nn's
    per-device handle and workspace (the objects the first armed call would
    create) and, in its default "exact" cache mode, heuristic entries keyed by
    the exact warm-up token counts (valid for exactly those problems; the
    wrapper removes them in any other mode); triton_fp8._quiet_seen (stderr
    silencing of new GEMM shapes only); with TunableOp tuning off a results
    miss records nothing. quiet_stderr: optional context-manager factory around
    the NN wgrad call (its first heuristic query prints hipBLASLt chatter;
    the _scaled_mm calls are silenced by triton_fp8 itself).
    """
    import time

    from generative_recommenders.ops.triton import triton_fp8 as tf8
    from generative_recommenders.ops.triton.triton_preproc_fp8 import (
        E4M3_MAX,
        E5M2_MAX,
        ln512_bwd_dxsum_emit,
        site_steady,
        swishln256_fwd_emit,
    )

    t0 = time.perf_counter()
    bf16, f32 = torch.bfloat16, torch.float32
    emus = _warm_emus()
    calls = 0
    with torch.no_grad(), torch.cuda.device(device):

        def full(shape: Tuple[int, ...], value: float, dtype: torch.dtype) -> torch.Tensor:
            return torch.full(shape, value, dtype=dtype, device=device)

        # Private stand-ins for one site's device state (never a _Site).
        scale = full((1,), 1.0, f32)
        inv_scale = full((1,), 1.0, f32)
        pending = full((), 0.0, f32)
        clipped = torch.zeros((), dtype=torch.int64, device=device)
        # Norm parameters as the modules hold them (fp32 [D]); W2/b2 fp32 masters.
        w256, b256 = full((_HID,), 1.0, f32), full((_HID,), 0.0, f32)
        w512 = full((_OUT,), 1.0, f32)
        w2, b2 = full((_OUT, _HID), 0.0625, f32), full((_OUT,), 0.0625, f32)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            w8, w8_t, w_scale, b2_c = _quantize_w2(device, w2, b2)
        calls += 1
        for rows in _WARM_ROWS:
            h1 = full((rows, _HID), 0.5, bf16)
            if rows == _WARM_ROWS[0]:
                # bf16 a1 [M, 256] / dh2 [M, 512] stand-ins, as the warmup calls book them.
                _warm_site_ops(h1, pending, scale, inv_scale, E4M3_MAX)
                _warm_site_ops(full((rows, _OUT), 0.5, bf16), pending, scale, inv_scale, E5M2_MAX)
                scale.fill_(1.0)
                inv_scale.fill_(1.0)
                calls += 2
            a1_rm: Optional[torch.Tensor] = None
            a1_tm: Optional[torch.Tensor] = None
            for emu in emus:
                for tm in (True, False):
                    rm, tm_t, _, _, part = swishln256_fwd_emit(
                        h1, w256, b256, 1e-6, inv_scale, emu, tm=tm, return_partials=True
                    )
                    calls += 1
                    if tm:
                        site_steady(part, scale, pending, clipped, E4M3_MAX, snap=True)
                        calls += 1
                        a1_rm, a1_tm = rm, tm_t
            del h1
            assert a1_rm is not None and a1_tm is not None
            h2 = tf8._scaled_mm(
                a1_rm, w8.t(), scale, w_scale, out_dtype=bf16, bias=b2_c
            )
            calls += 1
            dy = full((rows, _OUT), 1.0 / 64, bf16)
            mean = full((rows,), 0.0, f32)
            rstd = full((rows,), 1.0, f32)
            dh8, _, _, _, part = ln512_bwd_dxsum_emit(
                dy, h2, w512, mean, rstd, inv_scale, return_partials=True
            )
            calls += 2
            site_steady(part, scale, pending, clipped, E5M2_MAX, snap=False)
            calls += 1
            del dy, h2, mean, rstd
            tf8._scaled_mm(dh8, w8_t.t(), scale, w_scale, out_dtype=bf16)
            calls += 1
            if quiet_stderr is not None:
                with quiet_stderr():
                    _wgrad_nn_b(dh8, a1_tm, scale, scale)
            else:
                _wgrad_nn_b(dh8, a1_tm, scale, scale)
            calls += 1
            del dh8, a1_rm, a1_tm
        # P3 with its [M, 512] bf16 operands' storage above 2 GiB. Triton's AMD
        # backend also specializes every tensor argument on whether its storage
        # fits in 2 GiB (buffer ops; HIPBackend.is_within_2gb), and h2 / the
        # incoming gradient [M, 512] bf16 cross that at M = 2^21 rows, inside
        # the real per-rank range (~1.98-2.78M); the gradient can also be a view
        # of a larger buffer. So P3 is also compiled with (dy, x) storage
        # (> 2 GiB, > 2 GiB) and (> 2 GiB, < 2 GiB) for both N % 16 classes, from
        # views of ONE transient 2 GiB buffer (only the viewed rows are
        # written). Every other E-M3a operand stays below 2 GiB up to M = 2^22.
        big = torch.empty(((1 << 30) + 4096,), dtype=bf16, device=device)
        for rows in (_WARM_ROWS[0], _WARM_ROWS[3]):
            n = rows * _OUT
            x_big = big[:n].view(rows, _OUT).fill_(0.5)
            dy_big = big[n : 2 * n].view(rows, _OUT).fill_(1.0 / 64)
            mean = full((rows,), 0.0, f32)
            rstd = full((rows,), 1.0, f32)
            for x in (x_big, full((rows, _OUT), 0.5, bf16)):
                ln512_bwd_dxsum_emit(
                    dy_big, x, w512, mean, rstd, inv_scale, return_partials=True
                )
                calls += 2
            del x_big, dy_big, mean, rstd
        del big
        torch.cuda.synchronize(device)
    return calls, time.perf_counter() - t0


# --------------------------------------------------------------------------- stacked prebuild
def _grouped_prebuild_wanted(device: torch.device) -> bool:
    """The stacked backward could run in this process: E-M3a on (enabled()),
    its kill switch on and not latched, PR #7's FlyDSL LayerNorm backward on
    and not latched (else PR #7 never owns the calls), E-M2's FlyDSL latch
    unset, gfx950. Reads only (enabled() resolves the gate once, as the first
    forward would)."""
    if not enabled() or not _grouped_enabled() or _GROUPED_BROKEN is not None:
        return False
    if not TLN._wln_bwd_flydsl_enabled() or TLN._WLN_BWD_FLYDSL_BROKEN is not None:
        return False
    from generative_recommenders.ops import preproc_grouped_tail as pgt

    if pgt._BWD_FLYDSL_BROKEN is not None:
        return False
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return arch.startswith("gfx950")


def prebuild_grouped(device: torch.device) -> str:
    """STACKED BACKWARD startup prebuild (module docstring, WARM-UP). Called
    by dlrm_v4/train/blas_warmup.py's startup preload only with ENV_VAR "1"
    (run_startup_preload -> _prebuild_flydsl_ln_bwd, right after E-M2's
    grouped prebuild), i.e. after the DataLoader pre-start forked its pools:
    ln_bwd_grouped_fp8.prebuild_grouped_fp8 JIT-compiles, loads and launches
    once the stacked main kernel and the grouped finalize on synthetic
    zero/one tensors (the flyc.jit key holds no shape, so its M=1024 build is
    the production specialization). Returns "grp3f8=1" (built), "grp3f8=off"
    (the stacked backward cannot run here, _grouped_prebuild_wanted) or
    "grp3f8=error": an import, compile or launch error latches only the
    stacked path off (_disable_grouped, one WARNING; the P3 path runs), never
    PR #7's or E-M2's. Never raises. Touches no training state: no _Site,
    module, parameter or RNG draw (torch.zeros/ones only), and
    _GROUPED_ARMED_LOGGED stays unset."""
    try:
        if not _grouped_prebuild_wanted(device):
            return "grp3f8=off"
        from generative_recommenders.ops.flydsl import ln_bwd_grouped_fp8 as f8

        f8.prebuild_grouped_fp8(device)
        return "grp3f8=1"
    except Exception as error:
        _disable_grouped("startup prebuild", error)
        return "grp3f8=error"

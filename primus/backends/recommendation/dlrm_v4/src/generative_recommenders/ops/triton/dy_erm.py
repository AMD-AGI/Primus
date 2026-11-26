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

"""E-RM: the bottom layer's out-proj dY producer emits the out-proj backward's FP8 operands.

DLRM_HSTU_FP8_BWD_ERM=1 (yambda_5b.gin item (97) binds it; code default OFF;
read once at import; only "1" arms it; Arbor r14 iterations 55 / 62).

What. In layer 1's preprocess backward (ops/triton/triton_hstu_preprocess_and_
attention.py, `_backward_impl`), the input-LayerNorm backward with the residual --
item (29)'s FlyDSL kernel ln_bwd_flydsl_d512_res_x32_c3 -- runs its E-RM variant
ln_bwd_flydsl_d512_res_erm_x32_c3 (ops/flydsl/ln_bwd.py `ln_bwd_erm`): the same
kernel body, row map, dx and dW/dB, plus the e5m2 dgrad operand g8 [M, 512] and
the e4m3 wgrad operand x_rm [M, 512] (both row-major, from the STORED bf16 dx,
cast with layer 0's out-projection scale: the site's live inv_scale read
in-kernel, 2^-t_shift as a runtime argument) and one fp32 amax partial per block.
Layer 0's out-projection backward (ops/triton/triton_hstu_linear.py,
HSTUComputeOutputFunction) then skips its dual cast (bwd_fp8.
_quantize_gradient_and_weight_fused): the site program consumes the producer's
partials, L11 / DO8 / the hipBLASLt dgrad read g8 exactly as before, and W3 reads
x_rm through the row-major entry wgrad_ck_w3_t256s_xt2 (wgrad_ck_w3.run_w3_rm).

Objects (host metadata only, no tensor-to-tensor tag cycles, no device reads):
  ErmSite    made by HSTUComputeOutputFunction.forward of the STACK'S BOTTOM layer
             only (layer tag 0) when its FP8 NN wgrad will run; tagged on that
             forward's output (ERM_SITE_ATTR) and kept on its ctx.
  claim      the next layer's ordinary preprocess wrapper takes the site off its
             input x (the residual proxy makes that Function x's only consumer)
             when a backward through it is possible; the producer's ctx keeps it.
  ErmTicket  issued in the producer's backward BEFORE the LN-bwd launch, only where
             the knob-off call runs the same kernels on the same bytes: item (29)'s
             FlyDSL residual LN backward takes the call
             (triton_layer_norm.flydsl_ln_bwd_decline_reason is None) and the
             knob-off W3 call would run the ARMED x21 entry
             (wgrad_ck_w3.rm_inactive_reason is None); records the consumer
             site's state object, calls, t_shift, live inv_scale and the stream.
  ErmEmit    the record (g8, x_rm, partials + the identity of the stored dx) on
             site.record; the consumer pops it (take) and re-checks everything
             (consumable) BEFORE its `state.calls += 1`; any mismatch releases the
             payload and runs today's dual cast on the stored bf16 dY (exact).

Every decline runs today's exact path. Unexpected declines (a record that cannot
be consumed) and producer host errors latch E-RM off for the process (one
"erm DISABLED" line); no E-RM decline ever sets a shipped latch
(triton_layer_norm._WLN_BWD_FLYDSL_BROKEN, wgrad_ck_w3._broken,
wgrad_ck_w3._tiled_declined). W3's transient re-lay-out copies of x_rm come
from fp8_tm_alloc.empty_copy, as the shipped transient copies: out of memory
there propagates without latching the B11 pool (fp8_tm_alloc._broken).

Numerics: bit-identical to the knob-off path (dx, dW, dB, g8, x_rm = the step-tiled
copy's bytes, the scale state, L11 / DO8 outputs, d_output_weight via xt2 ==
x21), except the actckpt-A5-exempt amax of an all-NaN 256x128 cast tile, and,
until the lab's exhaustive tier passes on GPU, the FP8 bytes of NaN inputs.
"""

import atexit
import itertools
import os
from typing import Any, Dict, Optional, Tuple

import torch

KNOB: str = "DLRM_HSTU_FP8_BWD_ERM"
ERM_SITE_ATTR: str = "_hstu_dy_erm_site"
_REQUESTED: bool = os.environ.get(KNOB, "0") == "1"
_override: Optional[bool] = None
_broken: Optional[str] = None
# Set by dy_erm.prebuild (startup preload, before RUN_START) after the E-RM FlyDSL
# variant compiled and ran once on synthetic zeros; without it the ticket declines
# (D24) rather than compile after RUN_START.
_prebuilt: bool = False
_inert: Optional[str] = None
_inert_checked: bool = False
_tokens = itertools.count(1)
_once: set = set()
_counts: Dict[str, int] = {
    "sites": 0,
    "claimed": 0,
    "tickets": 0,
    "emitted": 0,
    "consumed": 0,
    "declined": 0,
}
_reasons: Dict[str, int] = {}
_atexit_registered: bool = False
D: int = 512
# x_rm offsets: M * 512 <= 2^31 - 1 (defence in depth; the producer's own M*D*2 <
# 2^32 guard and offsets_fit already imply it).
M_RM_MAX: int = (2**31 - 1) // D

# Test seams: the modules the protocol reads (production binds the real ones
# lazily, so importing this module imports nothing of the package).
_bwd = None  # ops/triton/bwd_fp8.py
_w3 = None  # ops/triton/wgrad_ck_w3.py
_tln = None  # ops/triton/triton_layer_norm.py
_snr = None  # ops/triton/fp8_wgrad_snr.py


def _mods() -> Tuple[Any, Any, Any, Any]:
    global _bwd, _w3, _tln, _snr
    if _bwd is None:
        from generative_recommenders.ops.triton import bwd_fp8 as b  # noqa: PLC0415

        _bwd = b
    if _w3 is None:
        from generative_recommenders.ops.triton import wgrad_ck_w3 as w  # noqa: PLC0415

        _w3 = w
    if _tln is None:
        from generative_recommenders.ops.triton import (  # noqa: PLC0415
            triton_layer_norm as t,
        )

        _tln = t
    if _snr is None:
        from generative_recommenders.ops.triton import fp8_wgrad_snr as s  # noqa: PLC0415

        _snr = s
    return _bwd, _w3, _tln, _snr


def _print(msg: str) -> None:
    print(f"[hstu-fp8-bwd] {msg}", flush=True)


def _arm_print(msg: str) -> None:
    """One rank-tagged line (ops/utils.arm_print_all_ranks, as items (39) / (DO8))."""
    try:
        from generative_recommenders.ops.utils import arm_print_all_ranks  # noqa: PLC0415

        arm_print_all_ranks(f"[hstu-fp8-bwd] {msg}")
    except Exception:  # noqa: BLE001  (logging must never break a step)
        _print(msg)


def _note(key: str, msg: str) -> None:
    """A decline line ("erm NOT used (...)", "erm record dropped (...)"), once
    per key and process: rank-tagged and written as ONE line, like the ARMED /
    DISABLED lines (an A/B checks these must-be-absent strings on every rank)."""
    _reasons[key] = _reasons.get(key, 0) + 1
    if key not in _once:
        _once.add(key)
        _arm_print(msg)


def _register_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        _atexit_registered = True
        # clean_exit.EXIT_HOOKS lists ("generative_recommenders.ops.triton.dy_erm",
        # "_final"): the K4 hard exit (os._exit) skips plain atexit callbacks.
        atexit.register(_final)


def requested() -> bool:
    """The knob (or the test-only override)."""
    return _REQUESTED if _override is None else _override


def set_requested_for_testing(value: Optional[bool]) -> None:
    global _override
    _override = value


def broken() -> Optional[str]:
    return _broken


def enabled() -> bool:
    return requested() and _broken is None


def set_broken(reason: str) -> None:
    """Latch E-RM (and only E-RM) off for the process: the shipped FlyDSL LN-bwd
    and W3 latches are never touched by an E-RM failure."""
    global _broken
    if _broken is None:
        _broken = reason
        _arm_print(f"erm DISABLED: {reason}")


def mark_prebuilt() -> None:
    global _prebuilt
    _prebuilt = True


def prebuilt() -> bool:
    return _prebuilt


def inert_reason() -> Optional[str]:
    """None when every prerequisite of E-RM is configured (D1), else the missing
    ones. Import-time flags of the modules E-RM sits between, read lazily."""
    global _inert, _inert_checked
    if _inert_checked:
        return _inert
    missing = []
    try:
        b, w, t, _s = _mods()
        from generative_recommenders.ops.triton import (  # noqa: PLC0415
            triton_fp8 as tf8,
            triton_hstu_linear as thl,
        )

        if not b.scale_controller_enabled():
            missing.append("DLRM_HSTU_FP8_BWD")
        if not (getattr(b, "_OUT_WGRAD_ARMED", False) and getattr(thl, "_FP8_OUT_WGRAD", False)):
            missing.append("DLRM_HSTU_FP8_BWD_OUT_WGRAD")
        if getattr(thl, "_FP8_OUT_WGRAD_IMPL", "nn") != "nn":
            missing.append("DLRM_HSTU_FP8_BWD_OUT_WGRAD_IMPL=nn")
        if not (getattr(b, "_OUT_WGRAD_T_E4M3", False) and b._GRAD_DTYPE is b._FP8_E5M2):
            missing.append("DLRM_HSTU_FP8_BWD_OUT_WGRAD_DFMT=e4m3 with e5m2 grads")
        if not getattr(b, "_BWD_HELPERS_FUSED", False):
            missing.append("DLRM_HSTU_FP8_BWD_HELPERS_FUSED")
        if not t._wln_bwd_flydsl_enabled():
            missing.append("DLRM_HSTU_FP8_WLN_BWD_FLYDSL")
        if not w.requested():
            missing.append("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3")
        if not w.tiled_requested():
            missing.append("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED")
        if not w.t256s_requested():
            missing.append("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S")
        if not tf8.hstu_fp8_fuse_residual_enabled():
            missing.append("DLRM_HSTU_FP8_FUSE_RESIDUAL")
    except Exception as error:  # noqa: BLE001
        missing.append(f"prerequisite check failed: {type(error).__name__}: {error}")
    _inert = ", ".join(missing) if missing else None
    _inert_checked = True
    return _inert


def _requested_line() -> Optional[str]:
    """The one "requested" line per process (D1), printed on first use: the
    startup prebuild (before RUN_START) or the first FP8 out-proj forward."""
    why = inert_reason()
    if "requested" not in _once:
        _once.add("requested")
        _arm_print("requested: erm=1" + (f" (INERT without {why})" if why else ""))
    return why


class ErmSite:
    __slots__ = ("key", "device", "claimed", "consumed", "record")

    def __init__(self, key: int, device: torch.device) -> None:
        self.key = int(key)
        self.device = device
        self.claimed = False
        self.consumed = False
        self.record: Optional["ErmEmit"] = None


class ErmTicket:
    __slots__ = ("state", "calls", "t_shift", "inv_scale", "stream")

    def __init__(
        self, state: Any, calls: int, t_shift: int, inv_scale: torch.Tensor, stream: Any
    ) -> None:
        self.state = state
        self.calls = int(calls)
        self.t_shift = int(t_shift)
        self.inv_scale = inv_scale
        self.stream = stream


class ErmEmit:
    """The emission record: the payload (released as soon as it is consumed or
    declined) and the identity of the stored bf16 dY (metadata only)."""

    def __init__(
        self,
        tk: ErmTicket,
        dx: torch.Tensor,
        g8: torch.Tensor,
        x_rm: torch.Tensor,
        partials: torch.Tensor,
    ) -> None:
        self.ticket = tk
        self.g8: Optional[torch.Tensor] = g8
        self.x_rm: Optional[torch.Tensor] = x_rm
        self.partials: Optional[torch.Tensor] = partials
        self.dx_ptr = int(dx.data_ptr())
        self.dx_version = int(dx._version)
        self.shape = tuple(dx.shape)
        self.stride = tuple(dx.stride())
        self.dtype = dx.dtype
        self.used = False
        # Layout of the wgrad operand the consumer holds once `used`: "rm" (x_rm,
        # [M, 512]) until a hipBLASLt reader re-lays it out ("tok2d").
        self.layout = "rm"
        self.token = next(_tokens)

    def release(self) -> None:
        self.g8 = None
        self.x_rm = None
        self.partials = None


def _stream(device: torch.device) -> Any:
    return torch.cuda.current_stream(device) if device.type == "cuda" else None


def make_site(
    output_weight: torch.Tensor,
    *,
    fp8_y_saved: bool,
    y_token_major: bool,
    dout_token_major: bool,
    out_wgrad_nn: bool,
    need_bwd: bool,
    layer_tag: int,
    helpers_fused_ok: bool,
) -> Optional[ErmSite]:
    """HSTUComputeOutputFunction.forward: the site of this out-projection, or None.
    D0 knob / latch; D1 prerequisites; not the FP8 NN wgrad (no saved e4m3 y,
    producer mode, IMPL != nn); D4 no backward (the forward's need_bwd; the claim
    re-checks grad mode); D26 not the stack's bottom layer (layer_tag =
    triton_hstu_linear._LNMD_LAYER_TAG: 0 = STUStack layer 0, -1 = untagged warmup /
    direct call); D6 weight without a stable identity or outside the fused helpers
    (bwd_fp8._has_stable_identity / _helpers_fused_ok)."""
    if not enabled():
        return None  # D0
    if _requested_line() is not None:
        return None  # D1
    if not (fp8_y_saved and dout_token_major and out_wgrad_nn) or y_token_major:
        return None
    if not need_bwd:
        return None  # D4
    if int(layer_tag) != 0:
        return None  # D26
    if not (output_weight.is_leaf and output_weight.requires_grad and helpers_fused_ok):
        return None  # D6
    _counts["sites"] += 1
    _register_atexit()
    return ErmSite(output_weight.data_ptr(), output_weight.device)


def claim(x: torch.Tensor, *, residual_proxy: bool, targets_only: bool) -> Optional[ErmSite]:
    """The ordinary preprocess wrapper (producer forward): the site tagged on x.
    D3 targets-only Function / no residual proxy (then x has other consumers);
    D4 no backward through this call (grad mode off or x does not require grad:
    eval / no_grad forwards); D11 a second claim."""
    if not enabled():
        return None
    site = getattr(x, ERM_SITE_ATTR, None)
    if not isinstance(site, ErmSite):
        return None
    if targets_only or not residual_proxy:
        return None  # D3
    if not (torch.is_grad_enabled() and x.requires_grad):
        return None  # D4
    if site.claimed:
        return None  # D11
    site.claimed = True
    _counts["claimed"] += 1
    return site


def producer_eligible(
    x: torch.Tensor,
    dy: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> Optional[str]:
    """None iff the knob-OFF call would run item (29)'s FlyDSL residual kernel and
    the E-RM variant covers it. triton_layer_norm.flydsl_ln_bwd_decline_reason is
    the dispatch's own side-effect-free test (switch on, not latched by a runtime
    or startup-prebuild FlyDSL error, no _wln_bwd_flydsl_reason decline)."""
    _, _, t, _ = _mods()
    why = t.flydsl_ln_bwd_decline_reason(
        x, dy, weight, bias, mean, rstd, residual, is_swish=False, sum_dx=False
    )
    if why is not None:
        return f"shipped FlyDSL LN bwd not taken: {why}"  # D18
    if weight.dtype != torch.bfloat16 or int(x.shape[1]) != D:
        return f"ERM variant covers D=512 bf16 weights only ({weight.dtype}, D={int(x.shape[1])})"  # D8
    if int(x.shape[0]) > M_RM_MAX:
        return f"M={int(x.shape[0])} > {M_RM_MAX} (x_rm 32-bit offsets)"  # D23
    if not _prebuilt:
        return "ERM FlyDSL variant not prebuilt before RUN_START"  # D24
    return None


def ticket(
    site: Optional[ErmSite],
    x: torch.Tensor,
    dy: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    fold_dropout: bool,
) -> Optional[ErmTicket]:
    """Producer backward, BEFORE the LN-bwd launch. No state change, no device access."""
    if site is None or not enabled():
        return None
    if site.consumed or site.record is not None:
        return None  # D11 (retained-graph second backward)
    if fold_dropout or residual is None:
        return None  # D7
    if torch.compiler.is_compiling() or os.environ.get("AOTI_LOWER"):
        _note("compile", "erm NOT used (compiling/AOTI)")
        return None  # D8
    b, w, _, s = _mods()
    if not (
        b.scale_controller_enabled()
        and b._OUT_WGRAD_ARMED
        and b._OUT_WGRAD_T_E4M3
        and b._BWD_HELPERS_FUSED
        and b._GRAD_DTYPE is b._FP8_E5M2
    ):
        _note("inert", "erm NOT used (prerequisites: BWD, OUT_WGRAD e4m3 copy, HELPERS_FUSED, e5m2 grads)")
        return None  # D1
    if bool(getattr(s, "SNR_ENABLED", False)):
        _note("snr", "erm NOT used (SNR probe on: fp8_wgrad_snr.SNR_ENABLED)")
        return None  # D10
    st = b._peek_site_state("output_projection", site.key)
    if st is None or not st.published or st.t_clipped_windows is None:
        return None  # D5 (warmup: no emission), silent
    why_w3 = w.rm_inactive_reason(x.device)  # D9 / D20 / D21 / D25
    if why_w3 is not None:
        _note("w3:" + why_w3.split(" (")[0], f"erm NOT used (W3: {why_w3})")
        return None
    if (
        x.dtype != torch.bfloat16
        or dy.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or x.dim() != 2
        or int(x.shape[1]) != D
        or int(x.shape[0]) < 1
        or dy.shape != x.shape
        or residual.shape != x.shape
        or not (x.is_contiguous() and dy.is_contiguous() and residual.is_contiguous())
    ):
        _note("shape", "erm NOT used (producer operands outside the contract)")
        return None  # D8
    why_p = producer_eligible(x, dy, residual, weight, bias, mean, rstd)  # D18 D8 D23 D24
    if why_p is not None:
        _note("producer:" + why_p.split(":")[0].split(" (")[0], f"erm NOT used ({why_p})")
        return None
    _counts["tickets"] += 1
    return ErmTicket(st, st.calls, st.t_shift, st.inv_scale, _stream(x.device))


def _ln_bwd_erm():  # pyre-ignore[3]
    from generative_recommenders.ops.flydsl.ln_bwd import ln_bwd_erm  # noqa: PLC0415

    return ln_bwd_erm


def _empty_x_rm(m: int, device: torch.device) -> torch.Tensor:
    """x_rm [M, 512] e4m3: the B11 pool block the bottom layer's step-tiled dout_t
    no longer takes (fp8_tm_alloc.empty_tm; plain torch.empty when the pool is off).
    For `produce` only (training steps, after alloc_pregrow's prewarm); never for
    the startup prebuild, which runs before that prewarm (see `prebuild`)."""
    from generative_recommenders.ops.triton import fp8_tm_alloc  # noqa: PLC0415

    return fp8_tm_alloc.empty_tm((m, D), torch.float8_e4m3fn, device)


def produce(
    site: ErmSite,
    tk: ErmTicket,
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The E-RM LN backward: (dx, dweight, dbias) of the residual call, with the
    record on site.record; None (nothing recorded) on a host-side error, after
    which today's call runs and writes fresh outputs (E-RM latched off; the
    shipped FlyDSL latch is not touched). Production options (cfg, remap, cache)
    are ln_bwd's own (kernel_options), as for the knob-off call."""
    try:
        x_rm = _empty_x_rm(int(x.shape[0]), x.device)
        dx, dw, db, g8, xrm, amaxp = _ln_bwd_erm()(
            dy, x, weight, bias, mean, rstd, residual, tk.inv_scale, tk.t_shift,
            out={"xrm": x_rm},
        )
    except Exception as error:  # noqa: BLE001
        set_broken(f"producer: {type(error).__name__}: {str(error)[:200]}")
        return None
    rec = ErmEmit(tk, dx, g8, xrm, amaxp)
    site.record = rec
    _counts["emitted"] += 1
    if "armed" not in _once:
        _once.add("armed")
        _arm_print(
            f"ARMED erm on {x.device}: output_projection dY from "
            f"ln_bwd_flydsl_d512_res_erm_x32_c3: e5m2 g8 {list(g8.shape)} + e4m3 x_rm "
            f"{list(xrm.shape)} + {int(amaxp.numel())} amax partials (bottom layer); "
            f"dual cast skipped (token {rec.token}, t_shift {tk.t_shift})"
        )
    return dx, dw, db


def run_producer(
    site: Optional[ErmSite],
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    residual: Optional[torch.Tensor],
    fold_dropout: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The producer hook: ticket + E-RM LN backward, or None (today's call runs)."""
    tk = ticket(site, x, dy, residual, weight, bias, mean, rstd, fold_dropout)
    if tk is None:
        return None
    assert site is not None and residual is not None
    return produce(site, tk, dy, x, weight, bias, mean, rstd, residual)


def take(site: Optional[ErmSite]) -> Optional[ErmEmit]:
    """Consumer backward: pop the record (marks the site consumed, so a
    retained-graph second backward never re-emits)."""
    if site is None:
        return None
    rec = site.record
    site.record = None
    site.consumed = True
    return rec


def _drop(rec: ErmEmit, why: str) -> None:
    rec.release()
    _counts["declined"] += 1
    _note("drop:" + why.split(" (")[0], f"erm record dropped ({why}) -> dual cast")
    set_broken(f"unexpected decline: {why}")


def consumable(
    rec: Optional[ErmEmit],
    state: Any,
    gradient: torch.Tensor,
    token_major: bool,
    t_e4m3: bool,
    t_tiled: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """bwd_fp8._quantize_gradient_and_weight_fused, published branch, BEFORE
    `state.calls += 1`: (g8, x_rm, partials) to use in place of the dual cast, or
    None (the record is released; the caller casts exactly as today). t_tiled is
    the consumer's own _out_wgrad_dout_tiled decision (the knob-off cast would
    write the step-tiled copy x21 reads, D22)."""
    if rec is None:
        return None
    tk = rec.ticket
    why: Optional[str] = None
    if rec.g8 is None or rec.used:
        why = "record released"
    elif tk.state is not state or not state.published:
        why = "site state replaced"  # D12
    elif state.calls != tk.calls:
        why = f"generation moved ({tk.calls} -> {state.calls})"  # D12
    elif state.t_shift != tk.t_shift:
        why = f"t_shift moved ({tk.t_shift} -> {state.t_shift})"  # D12
    elif not (token_major and t_e4m3):
        why = "consumer does not take the e4m3 token-major copy"
    elif not t_tiled:
        why = "consumer would not tile dout_t (knob-off W3 path is not x21)"  # D22
    elif (
        gradient.dtype != rec.dtype
        or int(gradient.data_ptr()) != rec.dx_ptr
        or int(gradient._version) != rec.dx_version
        or tuple(gradient.shape) != rec.shape
        or tuple(gradient.stride()) != rec.stride
    ):
        why = "dout is not the tensor the producer stored"  # D13
    elif _stream(gradient.device) != tk.stream:
        why = "stream differs"  # D14
    else:
        _, w, _, _ = _mods()
        r = w.rm_inactive_reason(gradient.device)
        if r is not None:
            why = f"W3 row-major entry not active since the emission ({r})"  # D9 D20 D21 D25
    if why is not None:
        _drop(rec, why)
        return None
    payload = (rec.g8, rec.x_rm, rec.partials)
    rec.release()
    rec.used = True
    _counts["consumed"] += 1
    if "consumed" not in _once:
        _once.add("consumed")
        _arm_print(
            f"erm consumed on {gradient.device}: {getattr(state, 'site', 'output_projection')} "
            f"(generation {tk.calls}, t_shift {tk.t_shift})"
        )
    return payload  # pyre-ignore[7]


def finish(rec: Optional[ErmEmit]) -> None:
    """bwd_fp8.output_projection_dgrad_operands, on every exit with a record: a
    record the cast did not consume is dropped (payload released, E-RM latched:
    the ticket's checks should have made it consumable)."""
    if rec is None or rec.used:
        return
    if rec.g8 is None:
        return  # already dropped by consumable()
    _drop(rec, "consumer did not reach the published fused cast")


def prebuild(device: torch.device) -> str:
    """Startup preload (dlrm_v4/train/blas_warmup._prebuild_flydsl_ln_bwd, before
    RUN_START): compile + launch the E-RM variant once on synthetic zeros with
    inv_scale = 1 and t_shift = 5 (no site, no scale state, no data: the MLPerf
    L545-549 rule as for the shipped variants). One FlyDSL specialization serves
    every M and shift (M, ITERS, G and 2^-t are runtime arguments). A failure
    latches E-RM only (never item (29)'s FlyDSL path).

    Every buffer here is a plain torch allocation, x_rm included: this runs
    before alloc_pregrow's B11 prewarm (train_ranker run_startup_preload, then
    pregrow_before_run_start), so a fp8_tm_alloc.empty_tm block mapped here
    would print item (38)'s must-be-absent "pool created lazily", spend the
    once-per-device "ARMED tm_nonexp" line on this synthetic operand and map a
    2 GiB block before the main pool's pre-grow sizes "rest" (fp8_warmup holds
    the pool off for the same reason; item (92)'s prebuilds use plain buffers).
    FlyDSL keys a raw tensor by type only, so the specialization is the same."""
    if not requested():
        return ""
    if _broken is not None:
        return "erm=off(latched)"
    why = _requested_line()
    if why is not None:
        return "erm=inert"
    try:
        m = 1024
        x = torch.zeros((m, D), dtype=torch.bfloat16, device=device)
        dy = torch.zeros_like(x)
        res = torch.zeros_like(x)
        w = torch.ones((D,), dtype=torch.bfloat16, device=device)
        b = torch.zeros((D,), dtype=torch.bfloat16, device=device)
        mean = torch.zeros((m,), dtype=torch.float32, device=device)
        rstd = torch.ones((m,), dtype=torch.float32, device=device)
        inv = torch.ones((1,), dtype=torch.float32, device=device)
        # plain torch.empty, NOT _empty_x_rm (the B11 pool; see the docstring)
        x_rm = torch.empty((m, D), dtype=torch.float8_e4m3fn, device=device)
        _ln_bwd_erm()(dy, x, w, b, mean, rstd, res, inv, 5, out={"xrm": x_rm})
    except Exception as error:  # noqa: BLE001
        set_broken(f"startup prebuild: {type(error).__name__}: {str(error)[:200]}")
        return "erm=error"
    mark_prebuilt()
    return "erm=1"


def _final() -> None:
    """clean_exit.EXIT_HOOKS entry: one counter line per process (print only)."""
    if not requested():
        return
    if any(_counts.values()) or _broken is not None:
        _arm_print(
            f"erm final=1 sites={_counts['sites']} claimed={_counts['claimed']} "
            f"tickets={_counts['tickets']} emitted={_counts['emitted']} "
            f"consumed={_counts['consumed']} declined={_counts['declined']} "
            f"latched={int(_broken is not None)} reasons={dict(sorted(_reasons.items()))}"
        )


def counters() -> Dict[str, int]:
    return dict(_counts)


def reset_for_testing() -> None:
    global _broken, _prebuilt, _inert, _inert_checked
    _broken = None
    _prebuilt = False
    _inert = None
    _inert_checked = False
    _once.clear()
    _reasons.clear()
    for k in _counts:
        _counts[k] = 0

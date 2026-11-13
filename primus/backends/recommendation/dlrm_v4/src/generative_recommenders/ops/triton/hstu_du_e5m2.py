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

"""du e5m2 emission from the ordinary layer's LN*U*dropout backward
(Arbor r16 B1a, component B lever L1a-du).

DLRM_HSTU_FP8_BWD_DU_E5M2=1 (default off; read by bwd_fp8 at import). On the
FP8 uvqk backward (DLRM_HSTU_FP8_BWD_UVQK=1, WGRAD=fp8t) the shipped
ordinary-layer path is

    _ln_mul_dropout_bwd_dx_du_rng_v2 -> bf16 du  duvqk[:, 0:512]  (2 B/elem)
    attention bwd                    -> bf16 dv, dk (and dq, or M2's e5m2 q)
    _quantize_amax_colsum(_skip)     -> e5m2 dz8 [Mp, 2048] + amax + colsum
                                        (reads the bf16 u|v|k(|q) columns)

With the knob the compute-output backward runs
`triton_hstu_linear._ln_mul_dropout_bwd_dx_du_rng_v3` instead: the same
per-row fp32 math as v2 (dx and the dW/dB partials bitwise; the stock dW/dB
finalize is unchanged), du rounded to bf16, back to fp32, clamped
du * inv_scale -> e5m2 into dz8[:, 0:512] (the cast's exact ops), rows
[M, Mp) of that block zeroed, per-program fp32 u colsum and int16-pattern
amax partials. The bf16 du is NOT written, and no bf16 d(uvqk) buffer is
allocated there either: the u gradient handed to autograd is a stride-0
placeholder (`placeholder_du`, one NaN element), so the step peak inside the
LN*U*dropout backward holds dz8 instead of the [M, 2048] bf16 buffer (review
fix, iter 28). The preprocess backward takes the emission (`take` +
`check_consumable`), allocates its own d(uvqk) as on the non-adopting path,
never reads the placeholder, and `consume` runs the cast over the remaining
blocks only (v|k with a filled M2 dq sink, else v|q|k) and merges amax /
colsum before `bwd_fp8.uvqk_projection_backward(prequant=...)`.

Contract objects:
  DuE5m2Site  one per preprocess forward call (key = uvqk_weight.data_ptr()),
              tagged on the deferred-SiLU u (HSTU_DUVQK_SITE_ATTR) and kept by
              both autograd contexts. The producer sets `emitted`, so a
              preprocess backward whose incoming gradient is not the emitted
              du view (autograd accumulation, a tensor hook) fails closed
              instead of silently using the placeholder. `take` sets
              `consumed`: a retained-graph second backward of the same call
              (whose preprocess backward recomputes the e4m3 Q/K/V and so
              cannot take the fp8t wgrad) keeps the shipped path.
  DuE5m2Emit  the emission record tagged on the du view (HSTU_DU_E5M2_ATTR).

Fallbacks: a record that cannot be consumed, a failing consumer cast, or an
FP8 projection backward that declines after the emission restores the bf16 u
columns from the e5m2 payload (LOSSY decode with the scale the payload was cast
with -- the item-23 / M2 precedent), logs one line per reason and latches B1a
off for the process (first-call failures only). Producer-side ineligibility
never reaches that path: it runs the shipped v2 kernel and writes bf16 du.

Numerics (DECLARED): e5m2 payload (incl. pad rows) and amax byte-identical to
the shipped cast; dx and dW/dB bitwise; the 512 u-column bias sums are the same
fp32 math in a different reduction order (as M2's q columns).
"""

import itertools
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

from generative_recommenders.ops.triton import bwd_fp8

logger: logging.Logger = logging.getLogger(__name__)

SITE: str = "uvqk_projection"
_tokens = itertools.count(1)
# Module-level latch: set (with the reason) by the first lossy fallback; the
# producer then keeps the shipped path for the rest of the process.
_du_e5m2_broken: Optional[str] = None
# One NaN element per (device, dtype): the base of every placeholder du.
_placeholder_base: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def placeholder_du(like: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The u gradient an emitting LN*U*dropout backward returns to autograd:
    a (rows, cols) stride-0 view of one cached NaN element (no allocation, no
    kernel per call). Its only valid content is the DuE5m2Emit record tagged
    on it; the preprocess backward never reads its values. Anything else that
    does (an observing hook, autograd.grad w.r.t. u) sees NaN instead of
    stale memory, and in-place writes into it raise (internal overlap)."""
    key = (like.device, like.dtype)
    base = _placeholder_base.get(key)
    if base is None:
        base = torch.full((1, 1), float("nan"), dtype=like.dtype, device=like.device)
        _placeholder_base[key] = base
    return base.expand(rows, cols)


def broken() -> Optional[str]:
    return _du_e5m2_broken


def set_broken(reason: str) -> None:
    global _du_e5m2_broken
    if _du_e5m2_broken is None:
        _du_e5m2_broken = reason


def reset_for_testing() -> None:
    global _du_e5m2_broken
    _du_e5m2_broken = None


def diagnostics_on() -> bool:
    """The uvqk backward diagnostics that read the BF16 dz (the preprocess
    module's `_dq_sink_diagnostics_on` set): with any of them on, the u
    columns must stay BF16."""
    return (
        int(os.environ.get("HSTU_ATTN_BWD_CHECK", "0") or 0) > 0
        or int(os.environ.get("DLRM_HSTU_FP8_BWD_FULL_STATS", "0") or 0) > 0
        or bool(os.environ.get("DLRM_HSTU_FP8_BWD_DUMP", ""))
    )


def log_not_used(key: str, reason: str, tail: str) -> None:
    """One 'du_e5m2 NOT used (<reason>)' line per reason key per process."""
    bwd_fp8.log_uvqk_once(
        f"du_e5m2_nu:{key}", f"du_e5m2 NOT used ({reason}) -> {tail}"
    )


class DuE5m2Site:
    """Forward-to-backward handle of one ordinary layer call (no tensors).

    `emitted`: token of the record the producer tagged on the du it returned
    (cleared by `take`). `consumed`: set by the first `take` of this call; a
    retained-graph re-backward of the same call then never emits (its
    preprocess backward recomputes the e4m3 Q/K/V, so the fp8t wgrad that
    could consume the record does not run; review fix, iter 28)."""

    __slots__ = ("key", "emitted", "consumed")

    def __init__(self, key: int) -> None:
        self.key = int(key)
        self.emitted: Optional[int] = None
        self.consumed = False


class DuE5m2Emit:
    """The LN*U*dropout backward's e5m2 u block of one layer's dz8.

    dz8 [Mp, width] e5m2 (only columns u_cols written, rows >= M zero),
    u_amax (0-dim fp32, maxnum semantics like the cast), u_colsum [u width]
    fp32, inv_scale (the site state's device tensor the payload was cast
    with), cast_scale (snapshot of the matching decode scale), the site key
    and scale generation, and the d(uvqk) buffer it belongs to (buffer_ptr;
    0 = none handed over: the du view is `placeholder_du` and the preprocess
    backward allocates its own d(uvqk)). Holds no reference to the du view
    (no tensor reference cycle through its tag).
    """

    def __init__(
        self,
        dz8: torch.Tensor,
        u_cols: Tuple[int, int],
        u_amax: torch.Tensor,
        u_colsum: torch.Tensor,
        inv_scale: torch.Tensor,
        cast_scale: torch.Tensor,
        rows: int,
        width: int,
        site_key: int,
        generation: Optional[Tuple[object, int]],
        buffer_ptr: int,
    ) -> None:
        self.dz8: Optional[torch.Tensor] = dz8
        self.u_cols = (int(u_cols[0]), int(u_cols[1]))
        self.u_amax: Optional[torch.Tensor] = u_amax
        self.u_colsum: Optional[torch.Tensor] = u_colsum
        self.inv_scale: Optional[torch.Tensor] = inv_scale
        self.cast_scale: Optional[torch.Tensor] = cast_scale
        self.rows = int(rows)
        self.width = int(width)
        self.rows_out = int(dz8.shape[0])
        self.site_key = int(site_key)
        self.generation = generation
        self.buffer_ptr = int(buffer_ptr)
        self.device = dz8.device
        self.token = next(_tokens)

    def restore_bf16(self, du: torch.Tensor) -> None:
        """LOSSY fallback: decode the e5m2 u block into the unwritten bf16 u
        columns `du` [M, u width] of the preprocess backward's d(uvqk) (same
        decode as `bwd_fp8.dequantize_prequant` and M2's `restore_bf16`), in
        row chunks to bound the fp32 temporaries. Never the placeholder du."""
        assert self.dz8 is not None and self.cast_scale is not None
        lo, hi = self.u_cols
        assert tuple(du.shape) == (self.rows, hi - lo)
        step = 1 << 18
        for i in range(0, self.rows, step):
            j = min(self.rows, i + step)
            du[i:j].copy_(
                (self.dz8[i:j, lo:hi].float() * self.cast_scale).to(du.dtype)
            )

    def release(self) -> None:
        """Drop the payload (4.8 GB at production M) once consumed."""
        self.dz8 = None
        self.u_amax = None
        self.u_colsum = None
        self.inv_scale = None
        self.cast_scale = None
        self.generation = None


def take(
    site: DuE5m2Site, dsilu_u: Optional[torch.Tensor]
) -> Optional[DuE5m2Emit]:
    """Pop this layer's emission record off the incoming gradient.

    None: the producer took the shipped path (bf16 du written). Raises when
    the producer emitted (it returned `placeholder_du`, no bf16 du) but the
    gradient that arrived is not the tagged du view: something between the
    two autograd nodes already consumed the placeholder, and no fallback can
    repair it. Marks the site consumed either way.
    """
    from generative_recommenders.ops.triton.triton_hstu_linear import (
        HSTU_DU_E5M2_ATTR,
    )

    rec = None
    if dsilu_u is not None:
        rec = getattr(dsilu_u, HSTU_DU_E5M2_ATTR, None)
        if rec is not None:
            delattr(dsilu_u, HSTU_DU_E5M2_ATTR)
    emitted, site.emitted = site.emitted, None
    # Review fix (iter 28): a retained-graph re-backward of this call keeps
    # the shipped path (see DuE5m2Site); set before any raise below.
    site.consumed = True
    if emitted is not None and (rec is None or rec.token != emitted):
        raise RuntimeError(
            "du_e5m2 (DLRM_HSTU_FP8_BWD_DU_E5M2=1): the LN*U*dropout backward "
            "of this layer emitted the e5m2 u columns and returned a "
            "placeholder du, but the gradient reaching the preprocess backward is "
            f"not that tensor (emitted token {emitted}, record "
            f"{None if rec is None else rec.token}); autograd accumulated it "
            "or a hook replaced it. Run without DLRM_HSTU_FP8_BWD_DU_E5M2."
        )
    return rec


def check_consumable(
    rec: DuE5m2Emit,
    duvqk: torch.Tensor,
    du_in_place: bool,
    weight: torch.Tensor,
    u_width: int,
    fp8t_wgrad: bool,
    tm_operand_ok: bool,
) -> Optional[Tuple[str, str]]:
    """(reason key, reason) why the preprocess backward cannot consume `rec`
    this call, or None.

    `duvqk` is the preprocess backward's d(uvqk): with `rec.buffer_ptr == 0`
    (the producer handed over no buffer, only `placeholder_du`) it must be
    the fresh one the preprocess backward allocated (not adopted); with a
    buffer it must be that buffer, adopted in place."""
    if rec.dz8 is None:
        return "buffer", "emission record already released"
    if rec.buffer_ptr == 0:
        if du_in_place:
            return "buffer", (
                "d(uvqk) adopted from a tagged du view, but the emission "
                "handed over no buffer"
            )
    elif not du_in_place or duvqk.data_ptr() != rec.buffer_ptr:
        return "buffer", "d(uvqk) buffer not adopted from the emitting du view"
    if tuple(duvqk.shape) != (rec.rows, rec.width) or not duvqk.is_contiguous():
        return "shape", (
            f"d(uvqk) {tuple(duvqk.shape)} != emitted ({rec.rows}, {rec.width})"
        )
    bn = bwd_fp8._COLSUM_BLOCK_N
    if rec.u_cols != (0, int(u_width)) or u_width % bn or rec.width % bn:
        return "layout", f"u block {rec.u_cols} vs (0, {u_width})"
    if rec.site_key != weight.data_ptr():
        return "site", "site key != uvqk_weight.data_ptr()"
    if not bwd_fp8.producer_generation_matches(rec.generation, SITE, rec.site_key):
        return "generation", "uvqk_projection scale generation moved since emission"
    if not fp8t_wgrad:
        return "fp8t", "fp8t uvqk wgrad not taken this call"
    if not tm_operand_ok:
        return "tm", "token-major E4M3 normed_x operand unavailable"
    return None


def decline(rec: DuE5m2Emit, du: torch.Tensor, key: str, reason: str) -> None:
    """Lossy restore of the bf16 u columns `du` (the u block of the
    preprocess backward's d(uvqk), never the placeholder), one loud line per
    reason, latch."""
    rec.restore_bf16(du)
    rec.release()
    set_broken(reason)
    message = (
        f"du_e5m2 NOT used ({reason}) -> LOSSY e5m2->bf16 restore of the "
        "emitted u columns for this call; du_e5m2 latched OFF for this process"
    )
    bwd_fp8.log_uvqk_once(f"du_e5m2_lossy:{key}", message)
    logger.warning("[hstu-fp8-bwd] %s", message)


def finish_u_only(
    rec: DuE5m2Emit, dz: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """No (filled) M2 dq sink: cast every block but u (the shipped tile, so
    the cast columns' payload / colsum / amax partials are bitwise) and merge
    the producer's u amax and u colsum -> prequant (dz8, amax, colsum, M)."""
    from generative_recommenders.ops.triton.hstu_dq_e5m2 import (
        quantize_amax_colsum_skip,
    )

    assert rec.dz8 is not None and rec.u_amax is not None
    assert dz.shape == (rec.rows, rec.width) and dz.is_contiguous()
    part_amax, part_col = quantize_amax_colsum_skip(
        dz, rec.inv_scale, rec.dz8, rec.u_cols
    )
    amax = torch.maximum(part_amax.amax(), rec.u_amax)
    colsum = part_col.sum(dim=0)
    colsum[rec.u_cols[0] : rec.u_cols[1]] = rec.u_colsum
    return rec.dz8, amax, colsum, rec.rows


def consume(
    rec: DuE5m2Emit,
    dq_sink: Any,
    duvqk: torch.Tensor,
    **projection_kwargs: Any,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Cast the remaining blocks, merge, run the FP8 uvqk projection backward
    on the producer-cast dz. `dq_sink` is a FILLED M2 `DqE5m2Sink` sharing
    rec.dz8 (cast skips u|q) or None (cast skips u). Returns the projection
    result; on None the u columns (and the sink's q columns) of `duvqk` hold
    the lossy bf16 restore, B1a is latched off, and the caller runs BF16."""
    lo, hi = rec.u_cols
    uq = dq_sink is not None
    fp8_grads = None
    reason: Optional[Tuple[str, str]] = None
    try:
        prequant = (
            dq_sink.finish(duvqk, du_emit=rec) if uq else finish_u_only(rec, duvqk)
        )
    except Exception as error:  # first-call failure: never take training down
        prequant = None
        reason = (
            "cast",
            f"consumer cast failed: {type(error).__name__}: {str(error)[:160]}",
        )
    if prequant is not None:
        fp8_grads = bwd_fp8.uvqk_projection_backward(
            dz=None,
            prequant=prequant,
            producer_desc=(
                "LNMD bwd e5m2 u columns + dq-split e5m2 q columns + v|k-only "
                "cast, no BF16 du/dq"
                if uq
                else "LNMD bwd e5m2 u columns + cast of the rest, no BF16 du"
            ),
            **projection_kwargs,
        )
        prequant = None
        if fp8_grads is None:
            reason = ("declined", "FP8 uvqk projection backward declined after emission")
    if fp8_grads is None:
        assert reason is not None
        if uq and getattr(dq_sink, "filled", False):
            dq_sink.restore_bf16(duvqk)
        decline(rec, duvqk[:, lo:hi], reason[0], reason[1])
        return None
    rows, rows_out, device = rec.rows, rec.rows_out, rec.device
    bwd_fp8.log_armed_v2(
        "du_e5m2" if uq else "du_e5m2_u",
        device,
        lambda: (
            f"ARMED du_e5m2 on {device}: _ln_mul_dropout_bwd_dx_du_rng_v3 "
            f"(diet, e5m2 u cols {lo}:{hi} + amax/colsum partials, no bf16 du) "
            f"-> cast skips u{'|q' if uq else ''} blocks; payload/amax bitwise, "
            f"u colsum fp32-order, M={rows} Mp={rows_out}"
        ),
    )
    rec.release()
    return fp8_grads

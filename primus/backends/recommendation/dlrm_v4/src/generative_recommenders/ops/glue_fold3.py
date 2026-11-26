# pyre-strict
"""GLUE_FOLD3: bitwise folds of main-stream casts and small glue kernels
(rccl iteration 68, component C row R-C13 "unassigned model glue").

Master switch ``DLRM_HSTU_FP8_GLUE_FOLD3``: default OFF in code (on only for
an explicit 1/true/on/yes); the gin binding
apply_env_bootstrap.DLRM_HSTU_FP8_GLUE_FOLD3 turns it on, and a pre-set =0 is
the kill switch. Each site also has its own sub-switch, a refinement that is
inert while the master is off and default on with it; an explicit
0/false/off/no on the sub-switch, or the master off, restores the shipped code
path exactly (same kernels, same autograd nodes, same order). Read once at
import, like the other DLRM_HSTU_FP8_ knobs. One "DLRMV4 armed OK: glue_fold3
<site>" line per site and process is logged on rank 0 the first time the
folded path runs.

PCAST -- ``DLRM_HSTU_FP8_GLUE_FOLD3_PCAST``: HSTU-layer parameter casts.
    Shipped: every STU layer casts its fp32 LayerNorm parameters to bf16 at
    the call sites (``self._input_norm_weight.to(x.dtype)`` etc.; the
    targets-only layer also casts ``_output_weight``): 13 tiny
    ``bfloat16_copy_kernel_cuda`` launches per forward at 1n (3 layers), and
    13 ``bfloat16tofloat32_copy_kernel_cuda`` launches (ToCopyBackward0) in
    backward. Folded: ONE Triton launch at the start of the STU stack forward
    casts all of them into one bf16 buffer (``_g3_pcast_fwd_kernel``); each
    call site then creates ONE autograd node per (weight, bias) pair at the
    same position where the two ``.to()`` nodes were created, whose backward
    upcasts both gradients in ONE launch (``_g3_pcast_bwd2_kernel``).
    Per element the math is the shipped cast: fp32->bf16 is the integer RNE
    formula of glue_fusion._f32_to_bf16_bits_rne (exhaustively equal to
    ATen's gfx950 cast, NaN rule included); bf16->fp32 is the exact 16-bit
    shift. Gradient-ready order is preserved: the pair node lists the bias
    first, so AccumulateGrad(bias) runs before AccumulateGrad(weight), as
    with the two ToCopyBackward0 nodes (the later-created bias node had the
    higher sequence number); DDP's rebuilt bucket order is therefore the same.
    Fail closed both ways: an error in the fused forward or in the fused
    backward logs one "DLRMV4 glue_fold3 pcast DISABLED" line and the shipped
    casts run from then on (the backward falls back to ``.to(fp32)``, which is
    the same exact shift). The STU stack clears the shadows in try/finally.
    One-off costs: the source table is built once with a pinned, non-blocking
    H2D copy (no stream synchronize); ``prewarm()`` compiles both kernels for
    the pre-RUN_START precompile list.

LAZY_SORT -- ``DLRM_HSTU_FP8_GLUE_FOLD3_SORT``: the HSTU attention
    ``sort_by_length`` permutation (seq-length diff + aten::sort: sub,
    3 copyBuffer, arange, radixSortKVInPlace per ordinary layer) is consumed
    only by the Triton attention kernels. When the FlyDSL forward/backward
    serve the call it is dead. Folded: the permutation is computed lazily,
    with the same code, only where a Triton attention kernel is about to run
    (forward fallback or backward fallback). Same values when used.
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from generative_recommenders.ops.glue_fusion import (
    _bf16_bits_to_f32,
    _f32_to_bf16_bits_rne,
)

logger: logging.Logger = logging.getLogger(__name__)


def _env_true(name: str) -> bool:
    """Default-off switch: on only for an explicit 1/true/on/yes (the gin
    binding writes "1"); unset, empty, 0 or anything else is off. Same parse
    as glue_fusion._env_true."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


def _env_on(name: str) -> bool:
    """Default-on (off only for an explicit 0/false/off/no), for the two
    per-site refinements only; the master switch is _env_true above."""
    return os.environ.get(name, "").strip().lower() not in ("0", "false", "off", "no")


GLUE_FOLD3: bool = _env_true("DLRM_HSTU_FP8_GLUE_FOLD3")
PCAST: bool = GLUE_FOLD3 and _env_on("DLRM_HSTU_FP8_GLUE_FOLD3_PCAST")
LAZY_SORT: bool = GLUE_FOLD3 and _env_on("DLRM_HSTU_FP8_GLUE_FOLD3_SORT")

_ARMED: "set[str]" = set()


def note_armed(site: str, detail: str) -> None:
    if site in _ARMED:
        return
    _ARMED.add(site)
    rank = 0
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    except Exception:  # pragma: no cover
        rank = 0
    # Ship-12 merge: every rank (tagged rank=R); rank-0-only lines get lost in the
    # merged log pipe, the per-rank count is the arming proof.
    logger.info("DLRMV4 armed OK: glue_fold3 %s ACTIVE -- %s (rank=%d)", site, detail, rank)


_DISABLED: "set[str]" = set()


def _disable(site: str, exc: BaseException) -> None:
    """Fail closed (forward or backward): turn the site off for the rest of the
    process and log ONE "DLRMV4 glue_fold3 <site> DISABLED" line (every rank)."""
    global PCAST
    if site == "pcast":
        PCAST = False
    if site in _DISABLED:
        return
    _DISABLED.add(site)
    logger.warning(
        "DLRMV4 glue_fold3 %s DISABLED for this process (%s: %s); the shipped ops "
        "run from here on",
        site,
        type(exc).__name__,
        str(exc)[:300],
    )


# --------------------------------------------------------------------------
# PCAST kernels
# --------------------------------------------------------------------------
_PCAST_CHUNK = 8192
_PCAST_BLOCK = 1024


@triton.jit
def _g3_pcast_fwd_kernel(Tab, Out, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    """One program per table entry (src address, dst element offset, numel <= CHUNK):
    Out[dst + i] = bf16_bits(src[i]) for i < numel. Out is the int16 view of the
    bf16 shadow buffer."""
    e = tl.program_id(0)
    src = tl.load(Tab + e * 3).to(tl.pointer_type(tl.float32))
    dst = tl.load(Tab + e * 3 + 1)
    n = tl.load(Tab + e * 3 + 2)
    for start in tl.static_range(0, CHUNK, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(src + offs, mask=m, other=0.0)
        tl.store(Out + dst + offs, _f32_to_bf16_bits_rne(v), mask=m)


@triton.jit
def _g3_pcast_bwd2_kernel(G0, G1, O0, O1, N0, N1, BLOCK: tl.constexpr):
    """O_j = float(G_j) for the two bf16 gradients (int16 views) of a pair node."""
    pid = tl.program_id(0)
    which = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if which == 0:
        m = offs < N0
        h = tl.load(G0 + offs, mask=m, other=0)
        tl.store(O0 + offs, _bf16_bits_to_f32(h), mask=m)
    else:
        m = offs < N1
        h = tl.load(G1 + offs, mask=m, other=0)
        tl.store(O1 + offs, _bf16_bits_to_f32(h), mask=m)


def _upcast_pair(g0: torch.Tensor, g1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    o0 = torch.empty_like(g0, dtype=torch.float32)
    o1 = torch.empty_like(g1, dtype=torch.float32)
    n0 = g0.numel()
    n1 = g1.numel()
    grid = (triton.cdiv(max(n0, n1, 1), _PCAST_BLOCK), 2)
    _g3_pcast_bwd2_kernel[grid](
        g0.view(torch.int16), g1.view(torch.int16), o0, o1, n0, n1,
        BLOCK=_PCAST_BLOCK,
    )
    return o0, o1


def _bwd_fusable(g: Optional[torch.Tensor]) -> bool:
    return (
        g is not None
        and g.is_cuda
        and g.dtype == torch.bfloat16
        and g.is_contiguous()
        and not g.requires_grad
    )


class _ShadowCastFunction(torch.autograd.Function):
    """y_i = p_i.to(bf16) where the cast itself was done by the stack-level
    fused launch (``shadows``). Backward: d p_i = d y_i.to(fp32), the two
    gradients of a pair in one launch (exact 16-bit shift)."""

    @staticmethod
    # pyre-ignore[14]
    def forward(ctx, shadows: Tuple[torch.Tensor, ...], *params: torch.Tensor):  # pyre-ignore[2,3]
        ctx.set_materialize_grads(False)
        return tuple(s.detach() for s in shadows)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, *grads: Optional[torch.Tensor]):  # pyre-ignore[2,3]
        if (
            PCAST
            and len(grads) == 2
            and _bwd_fusable(grads[0])
            and _bwd_fusable(grads[1])
            and grads[0].device == grads[1].device  # pyre-ignore[16]
        ):
            try:
                o0, o1 = _upcast_pair(grads[0], grads[1])  # pyre-ignore[6]
                return (None, o0, o1)
            except Exception as exc:  # fail closed: same values via .to(fp32)
                _disable("pcast", exc)
        # Shipped ToCopyBackward0 math (exact; identical bits to the fused kernel).
        return (None,) + tuple(
            None if g is None else g.to(torch.float32) for g in grads
        )


_TAB_CACHE: Dict[Tuple[object, ...], Tuple[torch.Tensor, List[int], int]] = {}


def _pcast_table(params: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, List[int], int]:
    """Device table of (src address, dst offset, numel) chunks. Parameter
    storages are persistent, so the table is built once and re-used while every
    data_ptr and size is unchanged (checked on the host every call). The table
    is a pure function of its key (addresses, sizes, device, stream).

    The one-time build is a PINNED, non-blocking H2D copy on the current stream
    (no stream synchronize, unlike a pageable ``.to(device)``); the caching host
    allocator keeps the pinned block until the copy has completed, and the
    consumer kernel runs on the same stream (the stream is part of the key, so
    a table is never read on a stream that is not ordered after its copy)."""
    dev = params[0].device
    stream = torch.cuda.current_stream(dev).cuda_stream
    key = tuple((p.data_ptr(), p.numel()) for p in params) + (dev, stream)
    hit = _TAB_CACHE.get(key)
    if hit is not None:
        return hit
    rows: List[int] = []
    offsets: List[int] = []
    off = 0
    for p in params:
        offsets.append(off)
        n = p.numel()
        base = p.data_ptr()
        for c in range(0, n, _PCAST_CHUNK):
            rows.extend((base + 4 * c, off + c, min(_PCAST_CHUNK, n - c)))
        off += n
    tab = (
        torch.tensor(rows, dtype=torch.int64)
        .pin_memory()
        .to(dev, non_blocking=True)
    )
    hit = (tab, offsets, off)
    if len(_TAB_CACHE) > 64:
        _TAB_CACHE.clear()
    _TAB_CACHE[key] = hit
    return hit


def stack_param_shadows(
    groups: Sequence[Sequence[torch.Tensor]], dtype: torch.dtype
) -> Optional[List[List[torch.Tensor]]]:
    """ONE launch casting every tensor of every group to bf16. Returns, per
    group, the bf16 tensors (same shapes) or None when not applicable (the
    caller then keeps the shipped ``.to()`` casts)."""
    if not PCAST or dtype != torch.bfloat16 or torch.jit.is_scripting():
        return None
    flat = [p for g in groups for p in g]
    if not flat:
        return None
    dev = flat[0].device
    for p in flat:
        if (
            p.dtype != torch.float32
            or not p.is_cuda
            or p.device != dev
            or not p.is_contiguous()
            or p.numel() == 0
        ):
            return None
    try:
        with torch.no_grad():
            tab, offsets, total = _pcast_table(flat)
            buf = torch.empty((total,), dtype=torch.bfloat16, device=dev)
            n_entries = tab.numel() // 3
            _g3_pcast_fwd_kernel[(n_entries,)](
                tab, buf.view(torch.int16), CHUNK=_PCAST_CHUNK, BLOCK=_PCAST_BLOCK
            )
    except Exception as exc:  # fail closed (forward only)
        _disable("pcast", exc)
        return None
    note_armed(
        "pcast",
        f"{len(flat)} STU parameter casts in 1 launch (_g3_pcast_fwd_kernel); "
        "pair upcasts in backward (_g3_pcast_bwd2_kernel); kill switch "
        "DLRM_HSTU_FP8_GLUE_FOLD3_PCAST=0",
    )
    out: List[List[torch.Tensor]] = []
    k = 0
    for g in groups:
        row = []
        for p in g:
            o = offsets[k]
            row.append(buf[o : o + p.numel()].view(p.shape))
            k += 1
        out.append(row)
    return out


def shadow_cast(
    shadows: Sequence[torch.Tensor], params: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    """Create the autograd node for already-cast ``shadows`` of ``params``."""
    return _ShadowCastFunction.apply(tuple(shadows), *params)


def prewarm(device: Optional[torch.device] = None) -> bool:
    """Compile both PCAST kernels ahead of the first step (entry point for the
    pre-RUN_START precompile list), in their production specializations:
    ``_g3_pcast_fwd_kernel`` (*i64 table, *i16 out, CHUNK=8192, BLOCK=1024) and
    ``_g3_pcast_bwd2_kernel`` (*i16 x2, *fp32 x2, N0/N1 = 512 i.e. divisible by
    16, BLOCK=1024). Synthetic tensors only: touches no module, parameter, FP8
    state or RNG stream, logs no armed line, leaves the table cache as it was.
    Returns False (no-op) when PCAST is off; fails closed like the forward."""
    if not PCAST:
        return False
    try:
        dev = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else device
        )
        before = set(_TAB_CACHE)
        with torch.no_grad():
            src = torch.zeros(512, dtype=torch.float32, device=dev)
            tab, _, total = _pcast_table([src])
            out = torch.empty((total,), dtype=torch.bfloat16, device=dev)
            _g3_pcast_fwd_kernel[(tab.numel() // 3,)](
                tab, out.view(torch.int16), CHUNK=_PCAST_CHUNK, BLOCK=_PCAST_BLOCK
            )
            g = torch.zeros(512, dtype=torch.bfloat16, device=dev)
            _upcast_pair(g, g)
            torch.cuda.current_stream(dev).synchronize()
        for k in set(_TAB_CACHE) - before:
            del _TAB_CACHE[k]
        return True
    except Exception as exc:
        _disable("pcast", exc)
        return False


# --------------------------------------------------------------------------
# LAZY_SORT helper (same code as the shipped eager sort)
# --------------------------------------------------------------------------
def sort_by_length_indices(seq_offsets: torch.Tensor) -> torch.Tensor:
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    _, indices = torch.sort(seq_lengths, descending=True, stable=False)
    return indices

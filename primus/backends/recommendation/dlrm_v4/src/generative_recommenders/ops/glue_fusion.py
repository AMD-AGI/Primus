# pyre-strict
"""Bitwise-exact fusions of small main-stream glue chains (iter30 glue track).

Ported onto the M1 tree in rccl iteration 65. Each site sits behind its own
switch, default OFF in code (``_env_true``: on only for an explicit
1/true/on/yes). The gin bindings apply_env_bootstrap.DLRM_HSTU_FP8_* turn
them on; a pre-set =0 is each site's kill switch and restores the shipped
path. The DLRM_HSTU_FP8_ prefix is what scripts/launch_slurm.sh forwards.
Each fused forward is fail-closed: if a Triton launch raises (compile or
launch), that part is disabled for the rest of the process with one warning,
and the shipped ops take over. The backward kernels of sites 1 and 3 have no
such fallback.

1. ``DLRM_HSTU_FP8_ACTION_ENC_FUSE`` -- ActionEncoder (modules/action_encoder.py).

   ARMED ONLY FOR THE bf16 CONSUMER under bf16 autocast. That is the
   production path: ContextualPreprocessor's action MLP, where
   _LinearNormFunction casts the [N + T, 24] fp32 result to bf16 and returns
   dx.to(fp32) from a bf16 matmul (iter-64 TB1 traces). The fp32 output mode
   is kept for the tests but never armed. Comm's GPU "test_action_fp32"
   failure was a CPU-vs-GPU reduction-order artefact of the test's CPU
   reference. On gfx950 the fp32 fused d_table is bit-identical to the GPU
   shipped graph (probes/iter65/glue/gpu/t0_diag_fp32.log). Restricting to
   the production dtype is still the safe choice.

   Shipped forward, 5 kernels on [N, 3, 8] / [N + T, 24] (1n trace, ship10
   595b51e, ms per rank-step): bitwise_and 0.022, gt 0.012, bool*table mul
   0.197, concat_2D_jagged fp32 0.170, and _LinearNormFunction's fp32->bf16
   cast of the result 0.063. Shipped backward, 4 kernels + the reduction:
   _LinearNormFunctionBackward's bf16->fp32 cast 0.052, zeros 0.034,
   split_2D_jagged 0.153, MulBackward0 mul 0.237, then sum_to 0.089.

   Fused: ONE forward kernel writes the [N + T, 24] rows straight into the
   concat layout in the dtype the consumer casts to (bf16 under autocast),
   and ONE backward kernel reads the bf16 gradient and writes (a) the masked
   fp32 [N, 3, 8] product MulBackward0 would have produced and (b) the fp32
   [T, 24] gradient of the tiled target rows. The reduction is NOT fused: the
   masked product is handed to ``Tensor.sum_to_size`` -- the same
   ``at::sum_to`` autograd's validate_outputs calls for MulBackward0 -- on a
   tensor with the same shape, dtype and strides, so the reduce kernel and its
   summation order are unchanged. The tiled target rows stay an autograd input
   (``tile`` is still called), so RepeatBackward0 runs exactly as shipped.

   Per-element arithmetic is identical to the shipped chain:
     fwd: m = float((a & w) > 0) (0.0 or 1.0); v = m * t (one IEEE fp32 mul,
          same operands); concat is a copy; bf16 = round-to-nearest-even of v
          implemented with the integer formula of
          c10::detail::round_to_nearest_even. There is no hardware cvt, so
          there is no MODE-bit dependence. NaN follows what ATen's
          bfloat16_copy_kernel_cuda does on gfx950 (torch 2.12 + ROCm 7.2): it
          keeps the sign and the upper payload and sets the quiet bit,
          ((u >> 16) | 0x40). That is NOT c10's canonical 0x7FC0. The iter65
          exhaustive 2^32 sweep against torch's own cast shows 0 mismatches
          (probes/iter65/glue/gpu).
     bwd: g = float(bf16) is an exact 16-bit shift; d = g * m (one IEEE fp32
          mul, same operands as MulBackward0's grad * other).
   Exactness assumes every sample's uih/target lengths are <= max_uih_len /
   max_targets, which holds by construction (they are the batch maxima from
   fx_length_stats); the shipped split leaves rows beyond max_seq_len zero /
   uninitialised and so would this fusion's caller-visible contract.

2. ``DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE`` -- ec_bf16_output._merged_index.

   Shipped: per call 2x(fill+cumsum), 3x arange, 2x(cumsum + repeat_interleave
   + gather), 2x index, 3x add, 2x scatter_ (~20 kernels), three calls per
   step plus a torch.stack for the cat-fused pair: 61 kernels, 0.500 ms per
   rank-step at 1n. Fused: one kernel per call, writing into the stacked
   buffer directly (no stack). Pure integer copies; bit-identical by
   construction.
"""

import logging
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger: logging.Logger = logging.getLogger(__name__)


def _env_true(name: str) -> bool:
    """Default-off switch: on only for an explicit 1/true/on/yes (the gin
    binding writes "1"); unset, empty, 0 or anything else is off. Same parse
    as ec_offsets._env_on / triton_fp8._env_true."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


ACTION_ENC_FUSE: bool = _env_true("DLRM_HSTU_FP8_ACTION_ENC_FUSE")
EC_MERGED_INDEX_FUSE: bool = _env_true("DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE")

_ARMED_LOGGED: "set[str]" = set()


def note_armed(part: str, detail: str) -> None:
    """One 'armed OK' line per part per process, emitted by the fused path itself."""
    if part in _ARMED_LOGGED:
        return
    _ARMED_LOGGED.add(part)
    logger.info("DLRMV4 armed OK: glue %s ACTIVE -- %s", part, detail)


def disable(part: str, exc: BaseException) -> None:
    """Fail closed: turn one fused part off for the rest of the process."""
    global ACTION_ENC_FUSE, EC_MERGED_INDEX_FUSE, CONTEXTUAL_FUSE
    global CONCAT_FOLD_UIH_TS, CONCAT_FOLD_CTX_TS
    if part == "action_enc":
        ACTION_ENC_FUSE = False
    elif part == "ec_merged_index":
        EC_MERGED_INDEX_FUSE = False
    elif part == "contextual":
        CONTEXTUAL_FUSE = False
    elif part == "concat_fold_uih_ts":
        CONCAT_FOLD_UIH_TS = False
    elif part == "concat_fold_ctx_ts":
        CONCAT_FOLD_CTX_TS = False
    logger.warning(
        "DLRMV4 glue %s DISABLED for this process (%s: %s); the shipped ops run "
        "from here on",
        part,
        type(exc).__name__,
        str(exc)[:300],
    )


# --------------------------------------------------------------------------
# bf16 bit helpers (integer only: identical on every backend, incl. NaN)
# --------------------------------------------------------------------------
@triton.jit
def _f32_to_bf16_bits_rne(v):
    """fp32 -> bf16 bits exactly as torch's .to(bfloat16) on gfx950: RNE
    (c10::detail::round_to_nearest_even) for every non-NaN input. A NaN keeps
    its sign and upper payload and gets the quiet bit (iter65: ATen's ROCm
    cast does this; c10's canonical 0x7FC0 differs from it)."""
    u = v.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    r = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    r = tl.where(v != v, (u >> 16) | 0x40, r) & 0xFFFF
    return r.to(tl.int16)


@triton.jit
def _bf16_bits_to_f32(h):
    u = (h.to(tl.int32) & 0xFFFF) << 16
    return u.to(tl.float32, bitcast=True)


# --------------------------------------------------------------------------
# 1. ActionEncoder
# --------------------------------------------------------------------------
# Actions is a slice of the batch payload with a data-dependent storage offset:
# no 16-byte alignment specialization, so the kernel compiles once (glue review
# finding 2: each new alignment combination was a ~0.1 s recompile in the run).
@triton.jit(do_not_specialize_on_alignment=["Actions"])
def _action_enc_fwd_kernel(
    Actions,  # int64 [N_left]
    Weights,  # int64 [A]
    Table,  # fp32 [A, D] contiguous
    Right,  # fp32 [N_right, A*D], row stride stride_r
    Out,  # [N_left + N_right, A*D]: int16 view of bf16, or fp32
    OffA,  # int64 [B + 1]
    OffB,  # int64 [B + 1]
    stride_r,
    D: tl.constexpr,
    AD: tl.constexpr,
    BLOCK_AD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    blk = tl.program_id(0)
    b = tl.program_id(1)
    start_a = tl.load(OffA + b)
    len_a = tl.load(OffA + b + 1) - start_a
    start_b = tl.load(OffB + b)
    len_b = tl.load(OffB + b + 1) - start_b
    n = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = n < len_a + len_b
    cols = tl.arange(0, BLOCK_AD)
    cmask = cols < AD
    w = tl.load(Weights + cols // D, mask=cmask, other=0)
    t = tl.load(Table + cols, mask=cmask, other=0.0)
    is_a = valid & (n < len_a)
    is_b = valid & (n >= len_a)
    act = tl.load(Actions + (start_a + n), mask=is_a, other=0)
    m = ((act[:, None] & w[None, :]) > 0).to(tl.float32)
    va = m * t[None, :]
    rb = (start_b + n - len_a).to(tl.int64)
    vb = tl.load(
        Right + rb[:, None] * stride_r + cols[None, :],
        mask=is_b[:, None] & cmask[None, :],
        other=0.0,
    )
    v = tl.where(is_a[:, None], va, vb)
    orow = (start_a + start_b + n).to(tl.int64)
    optr = Out + orow[:, None] * AD + cols[None, :]
    omask = valid[:, None] & cmask[None, :]
    if OUT_BF16:
        tl.store(optr, _f32_to_bf16_bits_rne(v), mask=omask)
    else:
        tl.store(optr, v, mask=omask)


@triton.jit(do_not_specialize_on_alignment=["Actions"])
def _action_enc_bwd_kernel(
    Grad,  # [N_left + N_right, A*D]: int16 view of bf16, or fp32
    Actions,
    Weights,
    DLeft,  # fp32 [N_left, A*D]  (== MulBackward0's grad * mask, viewed [N, A, D])
    DRight,  # fp32 [N_right, A*D]
    OffA,
    OffB,
    D: tl.constexpr,
    AD: tl.constexpr,
    BLOCK_AD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRAD_BF16: tl.constexpr,
    DLEFT_BF16: tl.constexpr = False,
):
    blk = tl.program_id(0)
    b = tl.program_id(1)
    start_a = tl.load(OffA + b)
    len_a = tl.load(OffA + b + 1) - start_a
    start_b = tl.load(OffB + b)
    len_b = tl.load(OffB + b + 1) - start_b
    n = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = n < len_a + len_b
    cols = tl.arange(0, BLOCK_AD)
    cmask = cols < AD
    orow = (start_a + start_b + n).to(tl.int64)
    gmask = valid[:, None] & cmask[None, :]
    graw = tl.load(Grad + orow[:, None] * AD + cols[None, :], mask=gmask, other=0)
    if GRAD_BF16:
        g = _bf16_bits_to_f32(graw)
    else:
        g = graw
    is_a = valid & (n < len_a)
    is_b = valid & (n >= len_a)
    w = tl.load(Weights + cols // D, mask=cmask, other=0)
    act = tl.load(Actions + (start_a + n), mask=is_a, other=0)
    m = ((act[:, None] & w[None, :]) > 0).to(tl.float32)
    ra = (start_a + n).to(tl.int64)
    p = g * m
    if DLEFT_BF16:
        # GLUE_FOLD4 ACTSUM (section 5): g is a bf16 value and m is 0.0 or
        # 1.0, so the fp32 product is exactly bf16-representable (a NaN from
        # +-inf * 0 has zero low bits too): its upper 16 bits are the product.
        p = ((p.to(tl.int32, bitcast=True) >> 16) & 0xFFFF).to(tl.int16)
    tl.store(
        DLeft + ra[:, None] * AD + cols[None, :],
        p,
        mask=is_a[:, None] & cmask[None, :],
    )
    rb = (start_b + n - len_a).to(tl.int64)
    tl.store(
        DRight + rb[:, None] * AD + cols[None, :],
        g,
        mask=is_b[:, None] & cmask[None, :],
    )


# Same (row-block, sample) grid as concat_2D_jagged_multirow; 256-row blocks
# keep the count of empty programs (samples shorter than max_seq_len) down.
_ACT_BLOCK_N: int = 256
_ACT_WARPS: int = 8


class _ActionEncodeConcat(torch.autograd.Function):
    """concat_2D_jagged((float((a & w) > 0)[..., None] * table).view(-1, A*D),
    right) [optionally .to(bf16)] in one kernel; see module docstring."""

    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        actions: torch.Tensor,
        weights: torch.Tensor,
        table: torch.Tensor,
        right: torch.Tensor,
        offsets_a: torch.Tensor,
        offsets_b: torch.Tensor,
        max_seq_len: int,
        out_bf16: bool,
    ) -> torch.Tensor:
        # The caller passes table.unsqueeze(0) ([1, A, D], created before the
        # target-row tile exactly as in the shipped graph) so the table's grad
        # flows through UnsqueezeBackward0 and AccumulateGrad (= DDP grad-ready)
        # order is the shipped one; a 2-D table is accepted too (tests).
        table3 = table.dim() == 3
        table2 = table.squeeze(0) if table3 else table
        A, D = table2.shape
        AD = A * D
        n_left = actions.numel()
        n_right = right.shape[0]
        B = offsets_a.numel() - 1
        table_c = table2.contiguous()
        right_c = right if right.stride(-1) == 1 else right.contiguous()
        out = torch.empty(
            (n_left + n_right, AD),
            device=actions.device,
            dtype=torch.bfloat16 if out_bf16 else table2.dtype,
        )
        BLOCK_AD = triton.next_power_of_2(AD)
        grid = (triton.cdiv(max_seq_len, _ACT_BLOCK_N), B)
        if B > 0 and max_seq_len > 0:
            _action_enc_fwd_kernel[grid](
                actions,
                weights,
                table_c,
                right_c,
                out.view(torch.int16) if out_bf16 else out,
                offsets_a,
                offsets_b,
                right_c.stride(0),
                D=D,
                AD=AD,
                BLOCK_AD=BLOCK_AD,
                BLOCK_N=_ACT_BLOCK_N,
                OUT_BF16=out_bf16,
                num_warps=_ACT_WARPS,
            )
        ctx.save_for_backward(actions, weights, offsets_a, offsets_b)
        ctx.shape = (A, D, n_left, n_right, B, max_seq_len)
        ctx.table3 = table3
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, d_out: torch.Tensor):  # pyre-ignore[3]
        actions, weights, offsets_a, offsets_b = ctx.saved_tensors
        A, D, n_left, n_right, B, max_seq_len = ctx.shape
        AD = A * D
        d_out = d_out.contiguous()
        grad_bf16 = d_out.dtype == torch.bfloat16
        # GLUE_FOLD4 ACTSUM (section 5): the masked product as bf16 bits plus
        # an fp32-accumulating sum (same ATen reduce config, same order).
        use16 = (
            G4_ACTSUM
            and grad_bf16
            and bool(ctx.needs_input_grad[2])
            and n_left >= 2
            and B > 0
            and max_seq_len > 0
            and _g4_actsum_selftest(d_out.device, A, D)
        )
        def _bwd(dleft16):  # pyre-ignore[2,3]
            d_left = torch.empty(
                (n_left, AD),
                device=d_out.device,
                dtype=torch.bfloat16 if dleft16 else torch.float32,
            )
            d_right = torch.empty(
                (n_right, AD), device=d_out.device, dtype=torch.float32
            )
            if B > 0 and max_seq_len > 0:
                grid = (triton.cdiv(max_seq_len, _ACT_BLOCK_N), B)
                _action_enc_bwd_kernel[grid](
                    d_out.view(torch.int16) if grad_bf16 else d_out,
                    actions,
                    weights,
                    d_left.view(torch.int16) if dleft16 else d_left,
                    d_right,
                    offsets_a,
                    offsets_b,
                    D=D,
                    AD=AD,
                    BLOCK_AD=triton.next_power_of_2(AD),
                    BLOCK_N=_ACT_BLOCK_N,
                    GRAD_BF16=grad_bf16,
                    DLEFT_BF16=dleft16,
                    num_warps=_ACT_WARPS,
                )
            return d_left, d_right

        d_table = None
        if use16:
            try:
                d_left, d_right = _bwd(True)
                # ATen's ReduceConfig for this reduction depends on the
                # accumulator type (fp32 in both), the shape and the
                # alignment, not on the input width: same launch config, same
                # fp32 additions in the same order, half the bytes read.
                d_table = torch.sum(
                    d_left.view(n_left, A, D),
                    dim=0,
                    keepdim=True,
                    dtype=torch.float32,
                )
            except Exception as exc:  # noqa: BLE001  fail closed
                # A compile or launch error of the bf16 variant: this call is
                # recomputed on the shipped path below, and the site stays off
                # for the rest of the process.
                g4_disable("actsum", f"{type(exc).__name__}: {exc}")
                use16 = False
                d_table = None
            else:
                note_armed_g4(
                    "actsum",
                    "action-encoder backward writes the masked [N, A * D] "
                    "product as bf16 and reduces it with an fp32-accumulating "
                    "sum (kill switch DLRM_HSTU_FP8_GLUE_FOLD4_ACTSUM=0)",
                )
        if not use16:
            d_left, d_right = _bwd(False)
            if ctx.needs_input_grad[2]:
                # Exactly what autograd does for the shipped graph:
                # MulBackward0 yields grad*mask as [N, A, D]; validate_outputs
                # then calls at::sum_to(.., [1, A, D]) (== Tensor.sum_to_size)
                # and UnsqueezeBackward0 squeezes dim 0 (a view).
                d_table = d_left.view(n_left, A, D).sum_to_size(1, A, D)
        if d_table is not None and not ctx.table3:
            d_table = d_table.squeeze(0)
        d_right_ret = d_right if ctx.needs_input_grad[3] else None
        return None, None, d_table, d_right_ret, None, None, None, None


def action_encode_concat(
    actions: torch.Tensor,
    weights: torch.Tensor,
    table: torch.Tensor,
    right: torch.Tensor,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    max_seq_len: int,
    out_bf16: bool,
) -> torch.Tensor:
    return _ActionEncodeConcat.apply(
        actions, weights, table, right, offsets_a, offsets_b, max_seq_len, out_bf16
    )


def action_encode_fusable(
    actions: torch.Tensor,
    weights: torch.Tensor,
    table: torch.Tensor,
    target_table: torch.Tensor,
    offsets_a: Optional[torch.Tensor],
    offsets_b: Optional[torch.Tensor],
    out_bf16: bool = False,
) -> bool:
    """target_table is the [1, A*D] row the caller tiles into `right`.

    out_bf16 must be True: the fused path arms only for the production
    configuration, a bf16 consumer under bf16 autocast. The fp32 output mode
    (the result used directly as fp32) stays on the shipped ops.
    """
    return (
        ACTION_ENC_FUSE
        and out_bf16
        and actions.is_cuda
        and actions.dim() == 1
        and actions.dtype == torch.int64
        and weights.dtype == torch.int64
        and weights.dim() == 1
        and table.dim() == 2
        and table.dtype == torch.float32
        and weights.numel() == table.shape[0]
        and target_table.dim() == 2
        and target_table.dtype == torch.float32
        and target_table.shape[1] == table.numel()
        and offsets_a is not None
        and offsets_b is not None
        and offsets_a.dtype == torch.int64
        and offsets_b.dtype == torch.int64
        and offsets_a.is_contiguous()
        and offsets_b.is_contiguous()
        and actions.is_contiguous()
        and weights.is_contiguous()
    )


# --------------------------------------------------------------------------
# 2. ec_bf16_output._merged_index
# --------------------------------------------------------------------------
# RiU / RiC are slices of the dedup reverse index with data-dependent storage
# offsets: no alignment specialization (one compiled variant; glue review finding 2).
@triton.jit(do_not_specialize_on_alignment=["RiU", "RiC"])
def _merged_index_kernel(
    RiU,
    RiC,
    LenU,
    LenC,
    Out,
    B,
    out_stride,
    out_offset,
    BLOCK_B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    idx = tl.arange(0, BLOCK_B)
    bm = idx < B
    lu = tl.load(LenU + idx, mask=bm, other=0).to(tl.int64)
    lc = tl.load(LenC + idx, mask=bm, other=0).to(tl.int64)
    off_u = tl.sum(tl.where(idx < b, lu, 0), axis=0)
    off_c = tl.sum(tl.where(idx < b, lc, 0), axis=0)
    len_u = tl.load(LenU + b).to(tl.int64)
    len_c = tl.load(LenC + b).to(tl.int64)
    # uih token j of sample b: src off_u + j  -> dst off_u + off_c + j
    dst_u = off_u + off_c
    for j0 in range(0, len_u, BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        jm = j < len_u
        v = tl.load(RiU + (off_u + j), mask=jm, other=0)
        tl.store(Out + (dst_u + j) * out_stride + out_offset, v, mask=jm)
    # candidate token j of sample b: src off_c + j -> dst off_u[b+1] + off_c + j
    dst_c = off_u + len_u + off_c
    for j0 in range(0, len_c, BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        jm = j < len_c
        v = tl.load(RiC + (off_c + j), mask=jm, other=0)
        tl.store(Out + (dst_c + j) * out_stride + out_offset, v, mask=jm)


def merged_index_fusable(
    ri_u: torch.Tensor, ri_c: torch.Tensor, len_u: torch.Tensor, len_c: torch.Tensor
) -> bool:
    return (
        EC_MERGED_INDEX_FUSE
        and ri_u.is_cuda
        and ri_u.dtype == ri_c.dtype
        and ri_u.dim() == 1
        and ri_c.dim() == 1
        and len_u.dim() == 1
        and len_c.dim() == 1
        and len_u.numel() == len_c.numel()
        and len_u.dtype in (torch.int32, torch.int64)
        and len_c.dtype in (torch.int32, torch.int64)
        and len_u.numel() <= 65536
    )


def merged_index_into(
    out: torch.Tensor,
    ri_u: torch.Tensor,
    ri_c: torch.Tensor,
    len_u: torch.Tensor,
    len_c: torch.Tensor,
    out_stride: int = 1,
    out_offset: int = 0,
) -> None:
    """out[(dst) * out_stride + out_offset] = merged reverse index (1-D out)."""
    B = len_u.numel()
    if B == 0:
        return
    _merged_index_kernel[(B,)](
        ri_u.contiguous(),
        ri_c.contiguous(),
        len_u.contiguous(),
        len_c.contiguous(),
        out,
        B,
        out_stride,
        out_offset,
        BLOCK_B=max(16, triton.next_power_of_2(B)),
        BLOCK=1024,
    )


def merged_index(
    ri_u: torch.Tensor,
    ri_c: torch.Tensor,
    len_u: torch.Tensor,
    len_c: torch.Tensor,
    n_u: int,
    n_c: int,
) -> torch.Tensor:
    out = torch.empty(n_u + n_c, dtype=ri_u.dtype, device=ri_u.device)
    merged_index_into(out, ri_u, ri_c, len_u, len_c)
    return out


def merged_index_stacked(
    parts: "list[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]",
    n: int,
) -> torch.Tensor:
    """torch.stack([merged_index(*p) for p in parts], dim=1).reshape(-1)."""
    k = len(parts)
    out = torch.empty(n * k, dtype=parts[0][0].dtype, device=parts[0][0].device)
    for q, (ri_u, ri_c, len_u, len_c) in enumerate(parts):
        merged_index_into(out, ri_u, ri_c, len_u, len_c, out_stride=k, out_offset=q)
    return out


# --------------------------------------------------------------------------
# 3. Contextual inputs (preprocessors.get_contextual_input_embeddings and
#    DlrmHSTU._construct_payload's per-feature offsets)
#
#    DLRM_HSTU_FP8_CONTEXTUAL_FUSE (default OFF in code; the gin binding turns
#    it on, =0 restores the shipped ops).
#    Shipped, per step at 1n (8 contextual features, max_len 1, 5 fp32 + 3
#    bf16 payloads): 8 x (fill + cumsum) offsets 0.098 ms; fwd 5 casts + 8
#    jagged_to_padded_dense + cat 0.091 ms; bwd 8 x (zeros + j2pd backward) +
#    5 ToCopyBackward0 0.154 ms.
#    Fused: offsets = torch.stack + ONE row-wise complete-cumsum kernel; fwd =
#    one cast+pad kernel per feature writing its column block of the cat
#    output directly; bwd = one kernel per feature producing the jagged
#    gradient in the payload's own dtype (zero for rows beyond max_len, exactly
#    the zeros-initialised j2pd backward; bf16->fp32 widening is exact).
#    Integer scans and copies, RNE casts via the c10 integer formula: bitwise.
#    (iter65 port) Payload rows past offsets[B] are zeroed in the backward,
#    as j2pd's zero-initialised backward does. One extra program does this
#    with no host sync. Production has no such rows: a JaggedTensor's value
#    count is sum(lengths).
# --------------------------------------------------------------------------
CONTEXTUAL_FUSE: bool = _env_true("DLRM_HSTU_FP8_CONTEXTUAL_FUSE")


@triton.jit
def _rowwise_complete_cumsum_kernel(
    Lengths, Out, B, stride_l, stride_o, BLOCK_B: tl.constexpr
):
    k = tl.program_id(0)
    idx = tl.arange(0, BLOCK_B)
    m = idx < B
    x = tl.load(Lengths + k * stride_l + idx, mask=m, other=0).to(tl.int64)
    c = tl.cumsum(x, axis=0)
    tl.store(Out + k * stride_o + 1 + idx, c, mask=m)
    tl.store(Out + k * stride_o + idx, tl.zeros_like(c), mask=idx == 0)


def complete_cumsum_many(lengths: "list[torch.Tensor]") -> "list[torch.Tensor]":
    """[asynchronous_complete_cumsum(l) for l in lengths] (int64, equal sizes)."""
    stacked = torch.stack(lengths, dim=0)
    K, B = stacked.shape
    out = torch.empty((K, B + 1), dtype=stacked.dtype, device=stacked.device)
    _rowwise_complete_cumsum_kernel[(K,)](
        stacked, out, B, stacked.stride(0), out.stride(0),
        BLOCK_B=max(16, triton.next_power_of_2(B)),
    )
    return list(out.unbind(0))


def complete_cumsum_many_fusable(lengths: "list[torch.Tensor]") -> bool:
    return (
        CONTEXTUAL_FUSE
        and len(lengths) > 1
        and all(
            t.is_cuda and t.dim() == 1 and t.dtype == torch.int64
            and t.numel() == lengths[0].numel()
            for t in lengths
        )
        and 0 < lengths[0].numel() <= 65536
    )


@triton.jit
def _ctx_pad_fwd_kernel(
    Values,  # [n, D]: fp32, or int16 view of bf16
    Offsets,  # int64 [B + 1]
    Out,  # int16 view of bf16 [B, W]
    stride_v,
    stride_o,
    col0,
    D: tl.constexpr,
    L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IN_FP32: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    start = tl.load(Offsets + b)
    n = tl.load(Offsets + b + 1) - start
    cols = tl.arange(0, BLOCK_D)
    cm = cols < D
    has = j < n
    src = Values + (start + j).to(tl.int64) * stride_v + cols
    if IN_FP32:
        v = tl.load(src, mask=cm & has, other=0.0)
        bits = tl.where(has, _f32_to_bf16_bits_rne(v), 0).to(tl.int16)
    else:
        bits = tl.load(src, mask=cm & has, other=0)
    tl.store(Out + b.to(tl.int64) * stride_o + col0 + j * D + cols, bits, mask=cm)


@triton.jit
def _ctx_pad_bwd_kernel(
    Grad,  # int16 view of bf16 [B, W]
    Offsets,
    DVals,  # [n, D]: fp32, or int16 view of bf16 (contiguous)
    stride_g,
    col0,
    NB,  # B (programs 0..B-1 own a sample; program B zeroes rows past offsets[B])
    n_rows,  # DVals rows
    D: tl.constexpr,
    L: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    b = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    cm = cols < D
    if b < NB:
        start = tl.load(Offsets + b)
        n = tl.load(Offsets + b + 1) - start
        for j in range(0, n):
            h = tl.load(
                Grad + b.to(tl.int64) * stride_g + col0 + j * D + cols,
                mask=cm & (j < L),
                other=0,
            )
            dst = DVals + (start + j).to(tl.int64) * D + cols
            if OUT_FP32:
                tl.store(dst, _bf16_bits_to_f32(h), mask=cm)
            else:
                tl.store(dst, h, mask=cm)
    else:
        # Rows past offsets[B]: zero, like the zero-initialised j2pd backward.
        # Empty loop in production.
        tail = tl.load(Offsets + NB)
        for k in range(0, n_rows - tail):
            dst = DVals + (tail + k).to(tl.int64) * D + cols
            if OUT_FP32:
                tl.store(dst, tl.zeros([BLOCK_D], dtype=tl.float32), mask=cm)
            else:
                tl.store(dst, tl.zeros([BLOCK_D], dtype=tl.int16), mask=cm)


class _ContextualPadCat(torch.autograd.Function):
    """torch.cat([flatten(j2pd(v.to(bf16), [off], [L], 0.0), 1, 2) ...], dim=1)."""

    @staticmethod
    # pyre-ignore[14]
    def forward(ctx, max_lens, offsets, *values):  # pyre-ignore[2,3]
        B = offsets[0].numel() - 1
        D = values[0].shape[1]
        W = sum(L * D for L in max_lens)
        out = torch.empty((B, W), dtype=torch.bfloat16, device=values[0].device)
        o16 = out.view(torch.int16)
        col0 = 0
        BLOCK_D = triton.next_power_of_2(D)
        done = False
        if G4_CTX1:  # GLUE_FOLD4 (section 5): every feature in one launch
            try:
                done = g4_ctx_fwd(o16, list(values), list(offsets), list(max_lens), D)
            except Exception as exc:  # noqa: BLE001  fail closed
                g4_disable("ctx1", f"{type(exc).__name__}: {exc}")
                done = False
        for L, off, v in zip(max_lens, offsets, values):
            if done:
                break
            fp32 = v.dtype == torch.float32
            vv = v if fp32 else v.view(torch.int16)
            if B > 0 and L > 0:
                _ctx_pad_fwd_kernel[(B, L)](
                    vv, off, o16, vv.stride(0), o16.stride(0), col0,
                    D=D, L=L, BLOCK_D=BLOCK_D, IN_FP32=fp32,
                )
            col0 += L * D
        ctx.save_for_backward(*offsets)
        ctx.meta = (tuple(max_lens), [(v.shape[0], v.dtype) for v in values], B, D)
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, d_out: torch.Tensor):  # pyre-ignore[3]
        offsets = ctx.saved_tensors
        max_lens, vmeta, B, D = ctx.meta
        g = d_out
        if g.dtype != torch.bfloat16:
            raise RuntimeError("contextual fuse: expected a bf16 gradient")
        if g.stride(1) != 1:
            g = g.contiguous()
        g16 = g.view(torch.int16)
        if G4_CTX1:  # GLUE_FOLD4 (section 5): every feature in one launch
            try:
                g4 = g4_ctx_bwd(
                    g16, list(offsets), list(max_lens), list(vmeta),
                    [bool(ctx.needs_input_grad[2 + i]) for i in range(len(vmeta))],
                    B, D,
                )
            except Exception as exc:  # noqa: BLE001  fail closed
                g4_disable("ctx1", f"{type(exc).__name__}: {exc}")
                g4 = None
            if g4 is not None:
                return (None, None, *g4)
        grads = []
        col0 = 0
        BLOCK_D = triton.next_power_of_2(D)
        for i, (L, off, (n, dt)) in enumerate(zip(max_lens, offsets, vmeta)):
            if not ctx.needs_input_grad[2 + i]:
                grads.append(None)
                col0 += L * D
                continue
            fp32 = dt == torch.float32
            dv = torch.empty((n, D), dtype=dt, device=g.device)
            if n > 0:
                _ctx_pad_bwd_kernel[(B + 1,)](
                    g16, off, dv if fp32 else dv.view(torch.int16), g16.stride(0),
                    col0, B, n, D=D, L=L, BLOCK_D=BLOCK_D, OUT_FP32=fp32,
                )
            grads.append(dv)
            col0 += L * D
        return (None, None, *grads)


def contextual_pad_cat_fusable(
    values: "list[torch.Tensor]",
    offsets: "list[torch.Tensor]",
    max_lens: "list[int]",
    dtype: torch.dtype,
) -> bool:
    if not (CONTEXTUAL_FUSE and dtype == torch.bfloat16 and len(values) > 0):
        return False
    D = values[0].shape[-1] if values[0].dim() == 2 else -1
    B = offsets[0].numel()
    return all(
        v.is_cuda and v.dim() == 2 and v.shape[1] == D and v.stride(1) == 1
        and v.dtype in (torch.float32, torch.bfloat16)
        and o.dtype == torch.int64 and o.dim() == 1 and o.is_contiguous()
        and o.numel() == B and L >= 1
        for v, o, L in zip(values, offsets, max_lens)
    )


def contextual_pad_cat(
    values: "list[torch.Tensor]",
    offsets: "list[torch.Tensor]",
    max_lens: "list[int]",
) -> torch.Tensor:
    return _ContextualPadCat.apply(list(max_lens), list(offsets), *values)


# --------------------------------------------------------------------------
# 4. Timestamp concat folds (rccl iteration 68, track "concat")
#
#    DLRM_HSTU_FP8_CONCAT_FOLD (default OFF in code, _env_true like the other
#    switches of this file; the gin binding apply_env_bootstrap.
#    DLRM_HSTU_FP8_CONCAT_FOLD turns it on; a pre-set =0 is the kill switch
#    and restores the shipped ops for both sites). Per-site sub-switches,
#    refinements that are inert while the master is off, each default on and
#    ANDed with it: DLRM_HSTU_FP8_CONCAT_FOLD_UIH_TS (site a),
#    DLRM_HSTU_FP8_CONCAT_FOLD_CTX_TS (site b). Read once at import.
#
#    The M2 tree's two remaining concat_2D_jagged calls on the main stream
#    both concatenate a ONE-column int64 timestamp tensor. concat_2D_jagged_
#    multirow launches a (cdiv(max_seq_len, BLOCK_N <= 8), B) grid, i.e.
#    ~0.4M-3.7M programs of <= 8 single-element rows at D = 1, so each call
#    costs ~0.11 ms for ~38 MB of traffic (iter-66 TB traces, 1n, B = 1024,
#    N ~ 2.37M, max_uih_len ~ 3.3k-3.7k):
#      (a) DlrmHSTU._user_forward: source_timestamps = concat(uih_action_time
#          [N_uih, 1], candidates_querytime [T, 1]), jagged on both sides
#          (0.115 ms per rank-step).
#      (b) ContextualPreprocessor.forward: concat(zeros [B*L, 1] (dense left,
#          L = max_contextual_seq_len), seq_timestamps [N, 1]), followed by
#          lengths + L and complete_cumsum(lengths + L): zeros fill 0.007 +
#          concat 0.110 + add 0.006 + cumsum (fill 0.005 + scan 0.007).
#    Fused:
#      (a) ``_jagged_concat_1d_kernel``: one 1-D jagged concat (same rows, same
#          order).
#      (b) ``_ctx_ts_pad_kernel``: writes the zero-padded timestamps, lengths +
#          L, and the new offsets as off[b] + L*b. That is complete_cumsum(
#          lengths + L) exactly, because the caller's off IS complete_cumsum(
#          lengths) (output_seq_offsets, computed earlier in the same forward,
#          before the contextual block): sum_{i<b}(len_i + L) = off[b] + L*b
#          in int64.
#    Integer copies and integer adds only: bitwise by construction (no float
#    math, no reordering of any reduction). No autograd: the inputs are
#    integer tensors, which the shipped _Concat2DJaggedFunction does not
#    record either. Rows past max_seq_len in a sample (which the batch maxima
#    exclude by construction) are left unwritten by the shipped kernel; the
#    fused kernels loop over the sample and write them.
#    The fused calls are fail-closed (compile/launch errors switch the site
#    off for the process, with one warning) and arm only on the TRITON hammer
#    kernel, outside TorchScript and fx tracing, for 1-D int64 CUDA tensors
#    with int64 contiguous offsets and 1 <= B <= 65535.
#    Fixed configs, no autotune: add both kernels to the pre-RUN_START
#    precompile list.
# --------------------------------------------------------------------------
def _concat_fold_env(name: str) -> bool:
    """Default-on (off only for an explicit 0/false/off/no), for the two
    per-site refinements only; the master switch is _env_true above."""
    return os.environ.get(name, "").strip().lower() not in ("0", "false", "off", "no")


CONCAT_FOLD: bool = _env_true("DLRM_HSTU_FP8_CONCAT_FOLD")
CONCAT_FOLD_UIH_TS: bool = CONCAT_FOLD and _concat_fold_env(
    "DLRM_HSTU_FP8_CONCAT_FOLD_UIH_TS"
)
CONCAT_FOLD_CTX_TS: bool = CONCAT_FOLD and _concat_fold_env(
    "DLRM_HSTU_FP8_CONCAT_FOLD_CTX_TS"
)

# Elements per program and warps (fixed; lab-tuned at the production shape,
# iter68 task 20261009T012024_1586819_19908: 512/2 best of 256..4096 x 2/4/8,
# (a) 15.1 us, (b) 14.4 us; 1024/4 was 17.5 / 17.2 us).
_TS_BLOCK: int = 512
_TS_WARPS: int = 2


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        return (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )
    except Exception:
        return True


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return 0


def note_armed_rank0(part: str, detail: str) -> None:
    """One 'DLRMV4 armed OK: <part>' line per process. Ship-12 merge: printed on
    EVERY rank (tagged rank=R; rank-0-only lines get lost in the merged log pipe,
    the per-rank count is the arming proof). The name is kept for its callers."""
    if part in _ARMED_LOGGED:
        return
    _ARMED_LOGGED.add(part)
    logger.info("DLRMV4 armed OK: %s ACTIVE -- %s (rank=%d)", part, detail, _rank())


# The payload values and offsets are slices of the batch's KJT buffers with
# data-dependent storage offsets: no alignment specialization (one compiled
# variant per kernel; glue review finding 2).
@triton.jit(do_not_specialize_on_alignment=["Left", "Right", "OffL", "OffR"])
def _jagged_concat_1d_kernel(
    Left,  # [N_left] (1-D, contiguous)
    Right,  # [N_right]
    OffL,  # int64 [B + 1]
    OffR,  # int64 [B + 1]
    Out,  # [N_left + N_right]
    BLOCK: tl.constexpr,
):
    blk = tl.program_id(0)
    nblk = tl.num_programs(0)
    b = tl.program_id(1)
    sl = tl.load(OffL + b).to(tl.int64)
    ll = tl.load(OffL + b + 1).to(tl.int64) - sl
    sr = tl.load(OffR + b).to(tl.int64)
    lr = tl.load(OffR + b + 1).to(tl.int64) - sr
    total = ll + lr
    dst = sl + sr
    # Sample b's output rows j in [0, ll + lr): left row sl + j for j < ll,
    # then right row sr + (j - ll). One trip per program when the grid covers
    # max_seq_len; any longer sample is still covered (strided loop).
    for j0 in range(blk.to(tl.int64) * BLOCK, total, nblk.to(tl.int64) * BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        valid = j < total
        is_l = valid & (j < ll)
        is_r = valid & (j >= ll)
        vl = tl.load(Left + (sl + j), mask=is_l, other=0)
        vr = tl.load(Right + (sr + j - ll), mask=is_r, other=0)
        tl.store(Out + (dst + j), tl.where(is_l, vl, vr), mask=valid)


@triton.jit(do_not_specialize_on_alignment=["Ts", "Off", "Lens"])
def _ctx_ts_pad_kernel(
    Ts,  # [N] (1-D, contiguous)
    Off,  # int64 [B + 1] == complete_cumsum(Lens)
    Lens,  # int64 [B]
    OutTs,  # [B * L + N]
    OutLens,  # int64 [B]
    OutOff,  # int64 [B + 1]
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    blk = tl.program_id(0)
    nblk = tl.num_programs(0)
    b = tl.program_id(1)
    b64 = b.to(tl.int64)
    start = tl.load(Off + b).to(tl.int64)
    end = tl.load(Off + b + 1).to(tl.int64)
    total = end - start + L
    dst = start + b64 * L
    # Sample b's output rows j in [0, L + n): zero for j < L (the shipped
    # torch.zeros left values), else Ts[start + j - L].
    for j0 in range(blk.to(tl.int64) * BLOCK, total, nblk.to(tl.int64) * BLOCK):
        j = j0 + tl.arange(0, BLOCK)
        valid = j < total
        v = tl.load(Ts + (start + j - L), mask=valid & (j >= L), other=0)
        tl.store(OutTs + (dst + j), v, mask=valid)
    if blk == 0:
        # lengths + L; complete_cumsum(lengths + L)[b + 1] = off[b + 1] + L*(b + 1)
        tl.store(OutLens + b, tl.load(Lens + b) + L)
        tl.store(OutOff + b + 1, end + (b64 + 1) * L)
        if b == 0:
            tl.store(OutOff, b64 * 0)  # complete_cumsum's leading 0


def _offsets_ok(off: Optional[torch.Tensor], B: int) -> bool:
    return (
        off is not None
        and off.is_cuda
        and off.dtype == torch.int64
        and off.dim() == 1
        and off.is_contiguous()
        and off.numel() == B + 1
    )


def _values_ok(v: torch.Tensor) -> bool:
    return (
        v.is_cuda
        and v.dim() == 1
        and v.dtype == torch.int64
        and v.is_contiguous()
        and not v.requires_grad
    )


def uih_ts_concat_fusable(
    left: torch.Tensor,
    right: torch.Tensor,
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
) -> bool:
    """Site (a): concat_2D_jagged(left[:, None], right[:, None], offsets_left,
    offsets_right).squeeze(-1) with both sides jagged."""
    if not CONCAT_FOLD_UIH_TS or offsets_left is None:
        return False
    B = offsets_left.numel() - 1
    return (
        1 <= B <= 65535
        and _values_ok(left)
        and _values_ok(right)
        and left.device == right.device
        and _offsets_ok(offsets_left, B)
        and _offsets_ok(offsets_right, B)
    )


def jagged_concat_1d(
    left: torch.Tensor,
    right: torch.Tensor,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """== concat_2D_jagged(max_seq_len, left[:, None], right[:, None],
    offsets_left=offsets_left, offsets_right=offsets_right).squeeze(-1)."""
    B = offsets_left.numel() - 1
    out = torch.empty(
        left.numel() + right.numel(), dtype=left.dtype, device=left.device
    )
    if out.numel() == 0:
        return out
    nblk = max(1, triton.cdiv(int(max_seq_len), _TS_BLOCK))
    _jagged_concat_1d_kernel[(nblk, B)](
        left,
        right,
        offsets_left,
        offsets_right,
        out,
        BLOCK=_TS_BLOCK,
        num_warps=_TS_WARPS,
    )
    return out


def ctx_ts_pad_fusable(
    seq_timestamps: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_contextual_seq_len: int,
) -> bool:
    """Site (b): ContextualPreprocessor's timestamp concat + lengths/offsets
    update. seq_offsets must be complete_cumsum(seq_lengths) (the caller's
    contract; see section 4 above)."""
    if not CONCAT_FOLD_CTX_TS:
        return False
    B = seq_lengths.numel()
    return (
        1 <= B <= 65535
        and 1 <= int(max_contextual_seq_len) <= 4096
        and _values_ok(seq_timestamps)
        and seq_lengths.is_cuda
        and seq_lengths.dtype == torch.int64
        and seq_lengths.dim() == 1
        and seq_lengths.is_contiguous()
        and _offsets_ok(seq_offsets, B)
    )


def ctx_ts_pad(
    seq_timestamps: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_contextual_seq_len: int,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(concat_2D_jagged(max_contextual_seq_len + max_seq_len,
         zeros[B * L, 1], seq_timestamps[:, None], max_len_left=L,
         offsets_right=seq_offsets).squeeze(-1),
        seq_lengths + L,
        complete_cumsum(seq_lengths + L)) with L = max_contextual_seq_len."""
    L = int(max_contextual_seq_len)
    B = seq_lengths.numel()
    dev = seq_timestamps.device
    out_ts = torch.empty(
        B * L + seq_timestamps.numel(), dtype=seq_timestamps.dtype, device=dev
    )
    out_lens = torch.empty(B, dtype=seq_lengths.dtype, device=dev)
    out_off = torch.empty(B + 1, dtype=seq_offsets.dtype, device=dev)
    nblk = max(1, triton.cdiv(L + int(max_seq_len), _TS_BLOCK))
    _ctx_ts_pad_kernel[(nblk, B)](
        seq_timestamps,
        seq_offsets,
        seq_lengths,
        out_ts,
        out_lens,
        out_off,
        L=L,
        BLOCK=_TS_BLOCK,
        num_warps=_TS_WARPS,
    )
    return out_ts, out_lens, out_off


# --------------------------------------------------------------------------
# 5. GLUE_FOLD4: main-stream small-kernel tail (rccl iteration 72, track
#    "tail"; component C rows R-C6 / R-C13).
#
#    Master switch DLRM_HSTU_FP8_GLUE_FOLD4: default OFF in code (_env_true:
#    on only for an explicit 1/true/on/yes); the gin binding
#    apply_env_bootstrap.DLRM_HSTU_FP8_GLUE_FOLD4 turns it on and a pre-set
#    =0 is the kill switch. Per-site sub-switches are refinements, inert while
#    the master is off and default on with it (off only for an explicit
#    0/false/off/no). Read once at import. With the master or a site's switch
#    at 0 the shipped code of that site runs (same kernels, same autograd
#    nodes, same order). Each folded path fails closed to the shipped ops for
#    the process with one "DLRMV4 glue_fold4 <site> DISABLED" warning.
#
#    ACTSUM (DLRM_HSTU_FP8_GLUE_FOLD4_ACTSUM): the action-encoder backward
#      (site 1) wrote the masked product g*m as fp32 [N, 24] (227 MB at
#      N = 2.37M) and reduced it with sum_to_size. g is a bf16 value and m is
#      0.0 or 1.0, so every product is exactly representable in bf16. The
#      kernel now stores the product's upper 16 bits (bf16, exact) and the
#      reduction is torch.sum(.., dtype=float32): ATen's ReduceConfig for a
#      reduction with fp32 accumulation depends on the shape, the alignment
#      and the accumulator, not on the input width, so the same reduce_kernel
#      launch config adds the same fp32 values in the same order -- bitwise,
#      from half the bytes. Requires N >= 2 (for N == 1 sum_to_size is a
#      no-op that keeps -0.0, a reduction gives +0.0). A one-time self-test
#      per process and device compares both reductions on synthetic rows
#      (raw bits) before the first use; a mismatch keeps the shipped path.
#    CTX1 (DLRM_HSTU_FP8_GLUE_FOLD4_CTX1): the contextual pad/cat of site 3
#      launched one forward and one backward kernel per feature (8 + 8 per
#      step). Now one launch each for up to 8 features: grid (B, max L, K)
#      forward and (B + 1, K) backward, each program running exactly the
#      per-feature program of the shipped kernels (same loads, RNE casts,
#      stores, tail zeroing) for its feature.
#    OFFS (DLRM_HSTU_FP8_GLUE_FOLD4_OFFS): integer offsets.
#      DlrmHSTU.preprocess: complete_cumsum(uih_seq_lengths) and
#      complete_cumsum(num_candidates) in one launch (g4_offsets
#      preproc=False): 4 -> 1. Integer prefix sums: bitwise. Each returned
#      tensor is a contiguous, 16-byte aligned row of one buffer.
#      g4_offsets(preproc=True) also covers the ContextualPreprocessor's
#      complete_cumsum(num_targets), complete_cumsum(seq_lengths), their
#      difference (uih_offsets) and complete_cumsum(output_seq_lengths) in one
#      launch (7 -> 1; integer subtraction, bitwise). It is NOT wired into
#      modules/preprocessors.py: the action-prefill item of the MNC instance
#      rewrites exactly those lines and moves the first three into
#      DlrmHSTU.preprocess, so a fold there would block that item's merge
#      for little gain. It stays tested here for a re-port onto that code.
#    POST (DLRM_HSTU_FP8_GLUE_FOLD4_POST): the targets-only output
#      postprocessor (hstu_transducer._postprocess + TimestampLayerNorm
#      Postprocessor) ran ~16 tiny kernels before its Linear: complete_cumsum
#      + sub + index_select of the candidate timestamps, the fp32 time
#      features (int64 -> fp32, floor division, remainder, division, two
#      multiplies, ones_like, polar), their bf16 cast, the cat with the
#      embeddings, and a bf16 -> fp32 -> bf16 round trip (.to(fp32) followed
#      by autocast's cast back to bf16 for the Linear; 2 more casts in the
#      backward). Now ONE kernel writes the [B, D + 2F] bf16 Linear input:
#      the same integer row selection, the same fp32 operations in the same
#      order with the same rounding (IEEE division div_rn; fmod, floor,
#      copysign, cos and sin from the same ocml device library ATen's
#      kernels call; the scalars 2 and 3.14 rounded to fp32 as ATen does;
#      1.0f * x == x), the integer RNE bf16 cast, and the embedding copy.
#      Armed only under bf16 autocast with an fp32 Linear weight, where the
#      round trip is the identity. The backward hands the Linear input's
#      bf16 gradient narrowed to the embedding columns to the embeddings, the
#      value and layout CatBackward0 produced after the two identity casts.
#      A one-time self-test per process and device compares the kernel with
#      the shipped torch ops on synthetic timestamps (raw bits); a mismatch
#      keeps the shipped path. Assumes every sample has >= 1 row (the shipped
#      index_select would raise otherwise; the kernel masks such reads).
# --------------------------------------------------------------------------
GLUE_FOLD4: bool = _env_true("DLRM_HSTU_FP8_GLUE_FOLD4")
G4_ACTSUM: bool = GLUE_FOLD4 and _concat_fold_env("DLRM_HSTU_FP8_GLUE_FOLD4_ACTSUM")
G4_CTX1: bool = GLUE_FOLD4 and _concat_fold_env("DLRM_HSTU_FP8_GLUE_FOLD4_CTX1")
G4_OFFS: bool = GLUE_FOLD4 and _concat_fold_env("DLRM_HSTU_FP8_GLUE_FOLD4_OFFS")
G4_POST: bool = GLUE_FOLD4 and _concat_fold_env("DLRM_HSTU_FP8_GLUE_FOLD4_POST")

_G4_LOGGED: "set[str]" = set()


def note_armed_g4(part: str, detail: str) -> None:
    """One 'DLRMV4 armed OK: glue_fold4 <part> ACTIVE' line per part per process."""
    if part in _G4_LOGGED:
        return
    _G4_LOGGED.add(part)
    logger.info("DLRMV4 armed OK: glue_fold4 %s ACTIVE -- %s", part, detail)


def g4_disable(part: str, why: str) -> None:
    """Fail closed: one GLUE_FOLD4 part off for the rest of the process."""
    global G4_ACTSUM, G4_CTX1, G4_OFFS, G4_POST
    if part == "actsum":
        G4_ACTSUM = False
    elif part == "ctx1":
        G4_CTX1 = False
    elif part == "offs":
        G4_OFFS = False
    elif part == "post":
        G4_POST = False
    logger.warning(
        "DLRMV4 glue_fold4 %s DISABLED for this process (%s); the shipped ops "
        "run from here on",
        part,
        str(why)[:300],
    )


def _g4_raw(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


# ---- ACTSUM self-test ------------------------------------------------------
_G4_ACTSUM_OK: "dict" = {}


def _g4_actsum_selftest(device: torch.device, A: int, D: int) -> bool:
    """Once per (device, A, D): sum_to_size of fp32 rows vs torch.sum of the
    same rows in bf16 with fp32 accumulation, raw bits, on synthetic rows that
    include +-0 and a wide exponent range, at a few row counts."""
    key = (str(device), int(A), int(D))
    ok = _G4_ACTSUM_OK.get(key)
    if ok is not None:
        return ok
    ok = False
    why = "mismatch"
    try:
        g = torch.Generator(device=device)
        g.manual_seed(20261009)
        # Small on purpose: it runs inside the first backward, near peak
        # memory (~30 MB transient). 65537 rows already take ATen's
        # multi-block (global) reduce path; production N was swept offline.
        nmax = 65537
        e = torch.randint(-60, 20, (nmax, A, D), device=device, generator=g).float()
        s = torch.where(
            torch.rand(nmax, A, D, device=device, generator=g) < 0.5, -1.0, 1.0
        )
        x16 = (s * torch.exp2(e) * (1 + torch.rand(nmax, A, D, device=device, generator=g))).to(
            torch.bfloat16
        )
        z = torch.rand(nmax, A, D, device=device, generator=g) < 0.4
        x16 = torch.where(z, (s * 0.0).to(torch.bfloat16), x16)
        ok = True
        for n in (2, 3, 17, 1001, 4097, nmax):
            a = x16[:n]
            r32 = a.float().sum_to_size(1, A, D)
            r16 = torch.sum(a, dim=0, keepdim=True, dtype=torch.float32)
            if not torch.equal(_g4_raw(r32), _g4_raw(r16)):
                ok = False
                why = f"raw bits differ at N={n}"
                break
        del e, s, x16, z
    except Exception as exc:  # noqa: BLE001
        ok = False
        why = f"{type(exc).__name__}: {exc}"
    _G4_ACTSUM_OK[key] = ok
    if not ok:
        g4_disable("actsum", "self-test: " + why)
    return ok


# ---- CTX1: contextual pad/cat in one launch per direction -------------------
_G4_CTX_KMAX: int = 8


@triton.jit
def _g4_ctx_fwd_one(
    V, Off, sv, Out, stride_o, b, j, cols, cm,
    COL0: tl.constexpr, D: tl.constexpr, L: tl.constexpr, IN_FP32: tl.constexpr,
):
    # == one program (b, j) of _ctx_pad_fwd_kernel for this feature
    if j < L:
        start = tl.load(Off + b)
        n = tl.load(Off + b + 1) - start
        has = j < n
        src = V + (start + j).to(tl.int64) * sv + cols
        if IN_FP32:
            v = tl.load(src, mask=cm & has, other=0.0)
            bits = tl.where(has, _f32_to_bf16_bits_rne(v), 0).to(tl.int16)
        else:
            bits = tl.load(src, mask=cm & has, other=0)
        tl.store(Out + b.to(tl.int64) * stride_o + COL0 + j * D + cols, bits, mask=cm)


@triton.jit(
    do_not_specialize_on_alignment=[
        "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
        "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
    ]
)
def _g4_ctx_pad_fwd_kernel(
    Out, stride_o,
    V0, V1, V2, V3, V4, V5, V6, V7,
    F0, F1, F2, F3, F4, F5, F6, F7,
    S0, S1, S2, S3, S4, S5, S6, S7,
    D: tl.constexpr, BLOCK_D: tl.constexpr, K: tl.constexpr,
    L0: tl.constexpr, L1: tl.constexpr, L2: tl.constexpr, L3: tl.constexpr,
    L4: tl.constexpr, L5: tl.constexpr, L6: tl.constexpr, L7: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    C4: tl.constexpr, C5: tl.constexpr, C6: tl.constexpr, C7: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    P4: tl.constexpr, P5: tl.constexpr, P6: tl.constexpr, P7: tl.constexpr,
):
    b = tl.program_id(0)
    j = tl.program_id(1)
    k = tl.program_id(2)
    cols = tl.arange(0, BLOCK_D)
    cm = cols < D
    if k == 0:
        _g4_ctx_fwd_one(V0, F0, S0, Out, stride_o, b, j, cols, cm, C0, D, L0, P0)
    if K > 1:
        if k == 1:
            _g4_ctx_fwd_one(V1, F1, S1, Out, stride_o, b, j, cols, cm, C1, D, L1, P1)
    if K > 2:
        if k == 2:
            _g4_ctx_fwd_one(V2, F2, S2, Out, stride_o, b, j, cols, cm, C2, D, L2, P2)
    if K > 3:
        if k == 3:
            _g4_ctx_fwd_one(V3, F3, S3, Out, stride_o, b, j, cols, cm, C3, D, L3, P3)
    if K > 4:
        if k == 4:
            _g4_ctx_fwd_one(V4, F4, S4, Out, stride_o, b, j, cols, cm, C4, D, L4, P4)
    if K > 5:
        if k == 5:
            _g4_ctx_fwd_one(V5, F5, S5, Out, stride_o, b, j, cols, cm, C5, D, L5, P5)
    if K > 6:
        if k == 6:
            _g4_ctx_fwd_one(V6, F6, S6, Out, stride_o, b, j, cols, cm, C6, D, L6, P6)
    if K > 7:
        if k == 7:
            _g4_ctx_fwd_one(V7, F7, S7, Out, stride_o, b, j, cols, cm, C7, D, L7, P7)


@triton.jit
def _g4_ctx_bwd_one(
    Grad, stride_g, Off, DV, NB, n_rows, b, cols, cm,
    COL0: tl.constexpr, D: tl.constexpr, L: tl.constexpr, OUT_FP32: tl.constexpr,
    NEED: tl.constexpr,
):
    # == one program b of _ctx_pad_bwd_kernel for this feature
    if NEED:
        if b < NB:
            start = tl.load(Off + b)
            n = tl.load(Off + b + 1) - start
            for j in range(0, n):
                h = tl.load(
                    Grad + b.to(tl.int64) * stride_g + COL0 + j * D + cols,
                    mask=cm & (j < L),
                    other=0,
                )
                dst = DV + (start + j).to(tl.int64) * D + cols
                if OUT_FP32:
                    tl.store(dst, _bf16_bits_to_f32(h), mask=cm)
                else:
                    tl.store(dst, h, mask=cm)
        else:
            tail = tl.load(Off + NB)
            for k2 in range(0, n_rows - tail):
                dst = DV + (tail + k2).to(tl.int64) * D + cols
                if OUT_FP32:
                    tl.store(dst, tl.zeros_like(cols).to(tl.float32), mask=cm)
                else:
                    tl.store(dst, tl.zeros_like(cols).to(tl.int16), mask=cm)


@triton.jit(
    do_not_specialize_on_alignment=[
        "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7",
    ]
)
def _g4_ctx_pad_bwd_kernel(
    Grad, stride_g, NB,
    F0, F1, F2, F3, F4, F5, F6, F7,
    G0, G1, G2, G3, G4, G5, G6, G7,
    N0, N1, N2, N3, N4, N5, N6, N7,
    D: tl.constexpr, BLOCK_D: tl.constexpr, K: tl.constexpr,
    L0: tl.constexpr, L1: tl.constexpr, L2: tl.constexpr, L3: tl.constexpr,
    L4: tl.constexpr, L5: tl.constexpr, L6: tl.constexpr, L7: tl.constexpr,
    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    C4: tl.constexpr, C5: tl.constexpr, C6: tl.constexpr, C7: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    P4: tl.constexpr, P5: tl.constexpr, P6: tl.constexpr, P7: tl.constexpr,
    Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr, Q3: tl.constexpr,
    Q4: tl.constexpr, Q5: tl.constexpr, Q6: tl.constexpr, Q7: tl.constexpr,
):
    b = tl.program_id(0)
    k = tl.program_id(1)
    cols = tl.arange(0, BLOCK_D)
    cm = cols < D
    if k == 0:
        _g4_ctx_bwd_one(Grad, stride_g, F0, G0, NB, N0, b, cols, cm, C0, D, L0, P0, Q0)
    if K > 1:
        if k == 1:
            _g4_ctx_bwd_one(Grad, stride_g, F1, G1, NB, N1, b, cols, cm, C1, D, L1, P1, Q1)
    if K > 2:
        if k == 2:
            _g4_ctx_bwd_one(Grad, stride_g, F2, G2, NB, N2, b, cols, cm, C2, D, L2, P2, Q2)
    if K > 3:
        if k == 3:
            _g4_ctx_bwd_one(Grad, stride_g, F3, G3, NB, N3, b, cols, cm, C3, D, L3, P3, Q3)
    if K > 4:
        if k == 4:
            _g4_ctx_bwd_one(Grad, stride_g, F4, G4, NB, N4, b, cols, cm, C4, D, L4, P4, Q4)
    if K > 5:
        if k == 5:
            _g4_ctx_bwd_one(Grad, stride_g, F5, G5, NB, N5, b, cols, cm, C5, D, L5, P5, Q5)
    if K > 6:
        if k == 6:
            _g4_ctx_bwd_one(Grad, stride_g, F6, G6, NB, N6, b, cols, cm, C6, D, L6, P6, Q6)
    if K > 7:
        if k == 7:
            _g4_ctx_bwd_one(Grad, stride_g, F7, G7, NB, N7, b, cols, cm, C7, D, L7, P7, Q7)


def _g4_pad8(xs: "list", fill):  # pyre-ignore[2,3]
    return list(xs) + [fill] * (_G4_CTX_KMAX - len(xs))


def g4_ctx_fwd(
    out16: torch.Tensor,
    values: "list[torch.Tensor]",
    offsets: "list[torch.Tensor]",
    max_lens: "list[int]",
    D: int,
) -> bool:
    """All features of _ContextualPadCat.forward in one launch. False: not
    applicable (the caller runs the per-feature launches)."""
    K = len(values)
    B = out16.shape[0]
    if not (G4_CTX1 and 1 <= K <= _G4_CTX_KMAX and B > 0 and min(max_lens) >= 1):
        return False
    vv = [v if v.dtype == torch.float32 else v.view(torch.int16) for v in values]
    cols0, c = [], 0
    for L in max_lens:
        cols0.append(c)
        c += L * D
    Ls = _g4_pad8([int(L) for L in max_lens], 0)
    Cs = _g4_pad8(cols0, 0)
    Ps = _g4_pad8([v.dtype == torch.float32 for v in values], False)
    V = _g4_pad8(vv, out16)
    F = _g4_pad8(list(offsets), offsets[0])
    S = _g4_pad8([int(v.stride(0)) for v in vv], 0)
    _g4_ctx_pad_fwd_kernel[(B, max(max_lens), K)](
        out16, out16.stride(0), *V, *F, *S,
        D=D, BLOCK_D=triton.next_power_of_2(D), K=K,
        L0=Ls[0], L1=Ls[1], L2=Ls[2], L3=Ls[3], L4=Ls[4], L5=Ls[5], L6=Ls[6], L7=Ls[7],
        C0=Cs[0], C1=Cs[1], C2=Cs[2], C3=Cs[3], C4=Cs[4], C5=Cs[5], C6=Cs[6], C7=Cs[7],
        P0=Ps[0], P1=Ps[1], P2=Ps[2], P3=Ps[3], P4=Ps[4], P5=Ps[5], P6=Ps[6], P7=Ps[7],
    )
    note_armed_g4(
        "ctx1",
        f"{K} contextual features padded+cast in 1 forward launch and 1 "
        "backward launch (kill switch DLRM_HSTU_FP8_GLUE_FOLD4_CTX1=0)",
    )
    return True


def g4_ctx_bwd(
    g16: torch.Tensor,
    offsets: "list[torch.Tensor]",
    max_lens: "list[int]",
    vmeta: "list",
    needs: "list[bool]",
    B: int,
    D: int,
) -> "Optional[list]":
    """All features of _ContextualPadCat.backward in one launch: the list of
    gradients (None where not needed), or None if not applicable."""
    K = len(vmeta)
    if not (G4_CTX1 and 1 <= K <= _G4_CTX_KMAX and min(max_lens) >= 1):
        return None
    grads: "list" = []
    G, Ns, Ps, Qs = [], [], [], []
    for (n, dt), need in zip(vmeta, needs):
        if not need:
            grads.append(None)
            G.append(g16)
            Ns.append(0)
            Ps.append(False)
            Qs.append(False)
            continue
        dv = torch.empty((n, D), dtype=dt, device=g16.device)
        grads.append(dv)
        G.append(dv if dt == torch.float32 else dv.view(torch.int16))
        Ns.append(int(n))
        Ps.append(dt == torch.float32)
        Qs.append(int(n) > 0)
    if any(Qs):
        cols0, c = [], 0
        for L in max_lens:
            cols0.append(c)
            c += L * D
        Ls = _g4_pad8([int(L) for L in max_lens], 0)
        Cs = _g4_pad8(cols0, 0)
        F = _g4_pad8(list(offsets), offsets[0])
        G = _g4_pad8(G, g16)
        Ns = _g4_pad8(Ns, 0)
        Ps = _g4_pad8(Ps, False)
        Qs = _g4_pad8(Qs, False)
        _g4_ctx_pad_bwd_kernel[(B + 1, K)](
            g16, g16.stride(0), B, *F, *G, *Ns,
            D=D, BLOCK_D=triton.next_power_of_2(D), K=K,
            L0=Ls[0], L1=Ls[1], L2=Ls[2], L3=Ls[3], L4=Ls[4], L5=Ls[5], L6=Ls[6], L7=Ls[7],
            C0=Cs[0], C1=Cs[1], C2=Cs[2], C3=Cs[3], C4=Cs[4], C5=Cs[5], C6=Cs[6], C7=Cs[7],
            P0=Ps[0], P1=Ps[1], P2=Ps[2], P3=Ps[3], P4=Ps[4], P5=Ps[5], P6=Ps[6], P7=Ps[7],
            Q0=Qs[0], Q1=Qs[1], Q2=Qs[2], Q3=Qs[3], Q4=Qs[4], Q5=Qs[5], Q6=Qs[6], Q7=Qs[7],
        )
    return grads


# ---- OFFS: integer offsets in one launch ------------------------------------
@triton.jit(do_not_specialize_on_alignment=["L0", "L1"])
def _g4_offsets_kernel(
    L0, L1, O0, O1, O2, O3, B, BLOCK_B: tl.constexpr, MODE: tl.constexpr
):
    """MODE 0: O0 = ccs(L0), O1 = ccs(L1).
    MODE 1: also O2 = O1 - O0 and O3 = ccs(L1) (a second, independent scan).
    ccs = fbgemm asynchronous_complete_cumsum (leading 0, int64)."""
    idx = tl.arange(0, BLOCK_B)
    m = idx < B
    a = tl.load(L0 + idx, mask=m, other=0).to(tl.int64)
    ca = tl.cumsum(a, axis=0)
    b = tl.load(L1 + idx, mask=m, other=0).to(tl.int64)
    cb = tl.cumsum(b, axis=0)
    z = ca * 0
    first = idx == 0
    tl.store(O0 + 1 + idx, ca, mask=m)
    tl.store(O0 + idx, z, mask=first)
    tl.store(O1 + 1 + idx, cb, mask=m)
    tl.store(O1 + idx, z, mask=first)
    if MODE == 1:
        tl.store(O2 + 1 + idx, cb - ca, mask=m)
        tl.store(O2 + idx, z, mask=first)
        cb2 = tl.cumsum(b, axis=0)
        tl.store(O3 + 1 + idx, cb2, mask=m)
        tl.store(O3 + idx, z, mask=first)


def _g4_len_ok(t: torch.Tensor) -> bool:
    return (
        t.is_cuda
        and t.dim() == 1
        and t.dtype == torch.int64
        and t.is_contiguous()
        and not t.requires_grad
    )


def g4_offsets(
    a: torch.Tensor, b: torch.Tensor, preproc: bool
) -> "Optional[Tuple[torch.Tensor, ...]]":
    """preproc=False: (ccs(a), ccs(b)). preproc=True (a = num_targets,
    b = seq_lengths): (ccs(a), ccs(b), ccs(b) - ccs(a), ccs(b)). None if not
    armed or not applicable (the caller runs the shipped ops). Every returned
    tensor is a contiguous int64 [B + 1] row of one buffer whose rows start
    16-byte aligned (row length rounded up to even), like a fresh tensor."""
    if not G4_OFFS:
        return None
    B = a.numel()
    if not (
        1 <= B <= 65536
        and b.numel() == B
        and _g4_len_ok(a)
        and _g4_len_ok(b)
        and a.device == b.device
    ):
        return None
    try:
        nout = 4 if preproc else 2
        out = torch.empty(
            (nout, (B + 2) // 2 * 2), dtype=torch.int64, device=a.device
        )
        o = [out[k, : B + 1] for k in range(nout)]
        _g4_offsets_kernel[(1,)](
            a, b, o[0], o[1], o[2 if preproc else 0], o[3 if preproc else 1], B,
            BLOCK_B=max(16, triton.next_power_of_2(B)), MODE=1 if preproc else 0,
        )
    except Exception as exc:  # noqa: BLE001  fail closed
        g4_disable("offs", f"{type(exc).__name__}: {exc}")
        return None
    note_armed_g4(
        "offs",
        "integer offsets (complete_cumsum) of a site in one launch (kill "
        "switch DLRM_HSTU_FP8_GLUE_FOLD4_OFFS=0)",
    )
    return tuple(o)


# ---- POST: targets-only postprocessor time features + cat in one kernel ------
try:  # the backend's device library (ocml on ROCm)
    from triton.language.extra import libdevice as _g4_libdevice
except Exception:  # pragma: no cover
    _g4_libdevice = None  # type: ignore[assignment]


@triton.jit
def _g4_div_floor(a, b):
    """c10::div_floor_floating(a, b) (torch.div(.., rounding_mode='floor'))."""
    mod = _g4_libdevice.fmod(a, b)
    div = tl.math.div_rn(a - mod, b)
    div = tl.where((mod != 0) & ((b < 0) != (mod < 0)), div - 1.0, div)
    fl = _g4_libdevice.floor(div)
    fl = tl.where(div - fl > 0.5, fl + 1.0, fl)
    q = tl.math.div_rn(a, b)
    zero = tl.zeros_like(a)
    r = tl.where(div != 0, fl, _g4_libdevice.copysign(zero, q))
    return tl.where(b == 0, q, r)


@triton.jit
def _g4_remainder(a, b):
    """aten remainder_kernel_cuda for float: fmod, then + b on a sign mismatch."""
    mod = _g4_libdevice.fmod(a, b)
    return tl.where((mod != 0) & ((b < 0) != (mod < 0)), mod + b, mod)


# N (timestamps per step) changes every step: not specialized (one compiled variant).
@triton.jit(do_not_specialize=["N"], do_not_specialize_on_alignment=["Emb", "Lens", "Ts"])
def _g4_time_concat_kernel(
    Emb, Lens, Ts, PU, UPP, Out, B, N, stride_e, stride_o,
    D: tl.constexpr, NF: tl.constexpr, BLOCK_B: tl.constexpr, ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr, C2: tl.constexpr, C314: tl.constexpr,
):
    pid = tl.program_id(0)
    ib = tl.arange(0, BLOCK_B)
    lens_all = tl.load(Lens + ib, mask=ib < B, other=0).to(tl.int64)
    r = pid * ROWS + tl.arange(0, ROWS)
    mr = r < B
    # candidate row of sample r = complete_cumsum(lens)[r + 1] - 1 (integers)
    base = tl.sum(tl.where(ib < pid * ROWS, lens_all, 0), axis=0)
    own = tl.load(Lens + r, mask=mr, other=0).to(tl.int64)
    rows = base + tl.cumsum(own, axis=0) - 1
    ts = tl.load(Ts + rows, mask=mr & (rows >= 0) & (rows < N), other=0)
    a = ts.to(tl.float32)[:, None]  # timestamps.float(): int64 -> fp32, RNE
    f = tl.arange(0, BLOCK_F)
    mf = f < NF
    pu = tl.load(PU + f, mask=mf, other=1.0)[None, :]
    upp = tl.load(UPP + f, mask=mf, other=1.0)[None, :]
    a = tl.broadcast_to(a, [ROWS, BLOCK_F])
    pu = tl.broadcast_to(pu, [ROWS, BLOCK_F])
    upp = tl.broadcast_to(upp, [ROWS, BLOCK_F])
    use = _g4_div_floor(a, pu)
    e = tl.math.div_rn(_g4_remainder(use, upp), upp)
    e = e * C2
    e = e * C314
    cb = _f32_to_bf16_bits_rne(_g4_libdevice.cos(e))
    sb = _f32_to_bf16_bits_rne(_g4_libdevice.sin(e))
    om = mr[:, None] & mf[None, :]
    optr = Out + r.to(tl.int64)[:, None] * stride_o + D + 2 * f[None, :]
    tl.store(optr, cb, mask=om)
    tl.store(optr + 1, sb, mask=om)
    cols = tl.arange(0, BLOCK_D)
    cmk = mr[:, None] & (cols[None, :] < D)
    x = tl.load(Emb + r.to(tl.int64)[:, None] * stride_e + cols[None, :], mask=cmk, other=0)
    tl.store(Out + r.to(tl.int64)[:, None] * stride_o + cols[None, :], x, mask=cmk)


_G4_POST_ROWS: int = 8
_G4_C314: float = float(torch.tensor(3.14, dtype=torch.float32).item())


def _g4_time_concat_launch(
    emb16: torch.Tensor,
    lens: torch.Tensor,
    ts: torch.Tensor,
    pu: torch.Tensor,
    upp: torch.Tensor,
) -> torch.Tensor:
    B, D = emb16.shape
    NF = pu.numel()
    out = torch.empty((B, D + 2 * NF), dtype=torch.int16, device=emb16.device)
    _g4_time_concat_kernel[(triton.cdiv(B, _G4_POST_ROWS),)](
        emb16.view(torch.int16), lens, ts, pu, upp, out, B, ts.numel(), emb16.stride(0),
        out.stride(0), D=D, NF=NF, BLOCK_B=max(16, triton.next_power_of_2(B)),
        ROWS=_G4_POST_ROWS, BLOCK_D=triton.next_power_of_2(D),
        BLOCK_F=max(2, triton.next_power_of_2(NF)),
        # ATen rounds the Python scalars 2 and 3.14 to the fp32 opmath type;
        # pass those exact fp32 values (as doubles that are fp32-exact).
        C2=2.0, C314=_G4_C314,
    )
    return out.view(torch.bfloat16)


def g4_time_features_reference(
    ts: torch.Tensor, pu: torch.Tensor, upp: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """The shipped TimestampLayerNormPostprocessor._concat_time_features time
    part, op for op (for the self-test and the harnesses)."""
    t = ts.unsqueeze(-1)
    use = torch.div(t.float(), pu.float(), rounding_mode="floor")
    el = (torch.remainder(use, upp.float()) / upp.float()) * 2 * 3.14
    el = torch.view_as_real(torch.polar(torch.ones_like(el), el)).flatten(-2, -1)
    return el.to(out_dtype)


_G4_POST_OK: "dict" = {}


def _g4_post_selftest(device: torch.device, pu: torch.Tensor, upp: torch.Tensor) -> bool:
    # Keyed on the buffers' storage and version counters: no device -> host
    # read on the per-step path (the step is host-sync-free); the one-time
    # test below syncs once per key.
    key = (str(device), pu.data_ptr(), pu._version, upp.data_ptr(), upp._version)
    ok = _G4_POST_OK.get(key)
    if ok is not None:
        return ok
    ok = False
    why = "mismatch"
    try:
        g = torch.Generator(device=device)
        g.manual_seed(20261009)
        n = 4096
        ts = torch.cat(
            [
                torch.randint(1_500_000_000, 1_900_000_000, (n - 64,), device=device, generator=g),
                torch.randint(0, 200_000, (32,), device=device, generator=g),
                torch.tensor([0, 1, 3599, 3600, 86399, 86400, 604799, 604800,
                              (1 << 31) - 1, 1 << 31, (1 << 40) + 7, 1_700_000_000,
                              1_700_003_599, 1_700_003_600, 1_699_999_999, 2_000_000_000,
                              123, 4_102_444_800, 999, 3601, 86401, 7 * 86400 + 1,
                              24 * 3600 - 1, 24 * 3600, 1_600_000_000, 1_600_000_128,
                              1_600_000_127, 1_600_000_129, 1_650_000_000, 1_750_000_000,
                              1_800_000_000, 1_850_000_000],
                             device=device, dtype=torch.int64),
            ]
        )
        lens = torch.ones(n, dtype=torch.int64, device=device)
        emb = torch.zeros(n, 8, dtype=torch.bfloat16, device=device)
        got = _g4_time_concat_launch(emb, lens, ts, pu.reshape(-1).float().contiguous(),
                                     upp.reshape(-1).float().contiguous())[:, 8:]
        ref = g4_time_features_reference(ts, pu.reshape(1, -1), upp.reshape(1, -1), torch.bfloat16)
        ok = bool(torch.equal(_g4_raw(got), _g4_raw(ref)))
        if not ok:
            why = f"{int((_g4_raw(got) != _g4_raw(ref)).sum().item())} of {ref.numel()} elements differ"
    except Exception as exc:  # noqa: BLE001
        ok = False
        why = f"{type(exc).__name__}: {exc}"
    _G4_POST_OK[key] = ok
    if not ok:
        g4_disable("post", "self-test: " + why)
    return ok


class _G4TimeConcat(torch.autograd.Function):
    """cat([emb, bf16(time_features(ts[ccs(lens)[1:] - 1]))], -1) in one kernel."""

    @staticmethod
    # pyre-ignore[14]
    def forward(ctx, emb16, lens, ts, pu, upp):  # pyre-ignore[2,3]
        ctx.D = emb16.shape[1]
        return _g4_time_concat_launch(emb16, lens, ts, pu, upp)

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, d_out):  # pyre-ignore[2,3]
        # CatBackward0's narrow of the (identity round-trip) bf16 gradient.
        return d_out.narrow(1, 0, ctx.D), None, None, None, None


def g4_time_concat(
    emb: torch.Tensor,
    lens: torch.Tensor,
    ts: torch.Tensor,
    period_units: torch.Tensor,
    units_per_period: torch.Tensor,
    weight_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """The bf16 [B, D + 2F] input of TimestampLayerNormPostprocessor's Linear
    for the targets-only path, or None (the caller runs the shipped ops)."""
    if not G4_POST or _g4_libdevice is None:
        return None
    dev = emb.device
    if not (
        emb.is_cuda
        and emb.dim() == 2
        and emb.dtype == torch.bfloat16
        and emb.stride(1) == 1
        and weight_dtype == torch.float32
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
        and _g4_len_ok(lens)
        and lens.numel() == emb.shape[0]
        and 1 <= lens.numel() <= 8192
        and ts.is_cuda
        and ts.dim() == 1
        and ts.dtype == torch.int64
        and ts.is_contiguous()
        and period_units.dtype == torch.float32
        and units_per_period.dtype == torch.float32
        and period_units.numel() == units_per_period.numel()
        and 1 <= period_units.numel() <= 4
        and period_units.device == dev
        and units_per_period.device == dev
    ):
        return None
    pu = period_units.reshape(-1).contiguous()
    upp = units_per_period.reshape(-1).contiguous()
    if not _g4_post_selftest(dev, pu, upp):
        return None
    try:
        out = _G4TimeConcat.apply(emb, lens, ts, pu, upp)
    except Exception as exc:  # noqa: BLE001  fail closed
        g4_disable("post", f"{type(exc).__name__}: {exc}")
        return None
    note_armed_g4(
        "post",
        "targets-only postprocessor: candidate timestamps, time features, bf16 "
        "cast and cat in 1 launch; no bf16 -> fp32 -> bf16 round trip "
        "(kill switch DLRM_HSTU_FP8_GLUE_FOLD4_POST=0)",
    )
    return out

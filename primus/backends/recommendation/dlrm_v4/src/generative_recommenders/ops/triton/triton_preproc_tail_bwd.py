# pyre-unsafe
"""Grouped LayerNorm(512) dx-sum backward of the preprocessor tail (E-M2).

Opt-in backward half of gin item (30): env
DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD=1, default off. Caller:
ops/preproc_grouped_tail.py _GroupedLinearLNTailFunction.backward.

WHAT IT REPLACES. The grouped forward tail computes
O = LN_x(h_x) (+) (LN_a(h_a) (+) LN_c(h_c)). Under F2 every last-pair
LayerNorm(512) receives the SAME output gradient dy. The per-pair backward
(E-M1, and the shipped chain) calls triton_weighted_layer_norm_bwd_dxsum once
per MLP. Each call reads dy [N, 512] bf16 and h_k [N, 512] bf16 and writes
dh_k [N, 512] bf16 (3 KB/row per call, 9 KB/row for three). This kernel runs
the three problems in ONE launch: it loads each dy row block once and then
runs the three per-problem bodies on it. That is 7 KB/row, one dy read
instead of three.

BITWISE CONTRACT. dh_k, dweight_k, dbias_k and dxsum_k are bit-identical to
three triton_weighted_layer_norm_bwd_dxsum(dy, h_k, ..., is_swish=False)
calls on the v2 path (DLRM_HSTU_FP8_WLN_BWD_V2, gin item (14)), whose
D = 512 SUM_DX launch is _weighted_layer_norm_bwd_dx_v2 with (tile multiplier
4, BLOCK_N 4, NUM_STAGES 2, strided walk, num_warps 1, EVEN_D). This kernel
keeps all of that:

  * the same grid tile_num = min(sms * 4, cdiv(N, 16)) and the same strided
    block walk (block b goes to program b % tile_num, in increasing order).
    Each program's fp32 dw/db/dxsum partial for problem k therefore sums the
    same rows in the same order;
  * per problem, the v2 non-swish body statement for statement: the same raw
    pointer expressions, masks, fp32 casts and reductions, and num_warps = 1
    (which fixes the tl.sum trees);
  * the unchanged _layer_norm_bwd_dwdbdx finalize per problem, launched as
    triton_weighted_layer_norm_bwd_dxsum launches it.

Each problem's bias partial is reduced from dy * ONE_k, where ONE_k is a
runtime 1.0 per problem. The product is exact, so the bits are those of the
shipped sum(dy). Without it the compiler could merge the three identical
column sums of dy into one, i.e. compute the three (mathematically equal)
LayerNorm bias gradients once. That would be a computation reduction, which
this kernel must not do: every problem's arithmetic runs in full.

PERFORMANCE NOTE (Arbor actckpt iter 45). Isolated, this launch beats the
three v2 launches it replaces: 3.2 vs 4.2 ms at N=2.333M. In the training step
the DDP gradient all-reduce is launched just before the preprocessor backward.
It spins during this launch on every rank except the last to arrive, and
there the kernel runs ~1.8x slower. The v2 launches lose ~5%.
  * The last rank is unaffected and saves ~1.8 ms. The step got shorter at 1n.
  * The per-rank kernel time is mixed, so the caller leaves this off by default.
  * Reproduced by probes/iter45/verify/t7_contention.py (8 ranks, the
    all-reduce issued from a side stream).
  * Neither NUM_STAGES=1 (249 VGPR, occupancy 2, not bitwise), finer contiguous
    partitions nor s_setprio removes it. waves_per_eu=2 spills to scratch.
See ops/preproc_grouped_tail.py, "WHY OFF BY DEFAULT".

IN-PLACE (donation, opt-in). dh_k may be written over h_k: h_k is the
last-pair Linear output that the grouped Function saves only for this
backward. Each program reads a row block of h_k before it writes the same rows
of dh_k, and no other program touches those rows. The caller donates only on
request (DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD_INPLACE=1) and only when
autograd will not run this node's backward again; see
preproc_grouped_tail._grouped_bwd_inplace_ok.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from generative_recommenders.ops.triton.triton_layer_norm import (
    _layer_norm_bwd_dwdbdx,
    _wln_bwd_v2_config,
    _wln_bwd_v2_enabled,
)

# The v2 launch this kernel reproduces: (tile multiplier, BLOCK_N, NUM_STAGES,
# CONTIG, num_warps, FAST_SIGMOID) of _wln_bwd_v2_config(512, is_swish=False,
# add_residual=False, sum_dx=True). grouped_ln_bwd_dxsum_reason() declines if
# that config ever changes, because the partial-sum partition would differ.
_V2_D512_SUMDX_CONFIG: Tuple[int, int, int, bool, int, bool] = (
    4,
    4,
    2,
    False,
    1,
    False,
)
GROUPED_BWD_DIM: int = 512


@triton.jit
def _grp_ln_bwd_one(
    X,
    DX,
    Mean,
    Rstd,
    stride_x,
    stride_dx,
    w,
    one,
    dy_block,
    rows,
    rows64,
    row_mask,
    cols,
    m2,
    D,
    acc_dw,
    acc_db,
    acc_dxsum,
):
    # == _weighted_layer_norm_bwd_dx_v2, IS_SWISH=False, ADD_RESIDUAL=False,
    # SUM_DX=True, EVEN_D=True, MASK_SCALE=False, for one problem; dy_block
    # is the shared fp32 dy row block.
    x_block = tl.load(
        X + rows64[:, None] * stride_x + cols[None, :], mask=m2, other=0.0
    ).to(tl.float32)
    mean = tl.load(Mean + rows, row_mask, other=0.0)
    rstd = tl.load(Rstd + rows, row_mask, other=0.0)
    mean = tl.expand_dims(mean, 1)
    rstd = tl.expand_dims(rstd, 1)

    xhat = (x_block - mean) * rstd
    wdy = w[None, :] * dy_block

    c1 = tl.sum(xhat * wdy, axis=1) / D
    c2 = tl.sum(wdy, axis=1) / D
    c1 = tl.expand_dims(c1, 1)
    c2 = tl.expand_dims(c2, 1)
    dx = (wdy - (xhat * c1 + c2)) * rstd
    partial_dw = tl.sum(dy_block * xhat, axis=0)
    # one == 1.0 at run time: dy * one == dy exactly, and it keeps this
    # problem's bias sum its own computation (see the module docstring).
    partial_db = tl.sum(dy_block * one, axis=0)
    tl.store(
        DX + rows64[:, None] * stride_dx + cols[None, :],
        dx.to(DX.dtype.element_ty),
        mask=m2,
    )
    acc_dw += partial_dw
    acc_db += partial_db
    acc_dxsum += tl.sum(dx, axis=0)
    return acc_dw, acc_db, acc_dxsum


@triton.jit
def _grouped_ln_bwd_dxsum_kernel(
    DY,
    stride_dy,
    X0,
    DX0,
    W0,
    Mean0,
    Rstd0,
    DW0,
    DB0,
    DXSUM0,
    stride_x0,
    stride_dx0,
    one0,
    X1,
    DX1,
    W1,
    Mean1,
    Rstd1,
    DW1,
    DB1,
    DXSUM1,
    stride_x1,
    stride_dx1,
    one1,
    X2,
    DX2,
    W2,
    Mean2,
    Rstd2,
    DW2,
    DB2,
    DXSUM2,
    stride_x2,
    stride_dx2,
    one2,
    D,
    N,
    NUM_IN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Grid and walk == _weighted_layer_norm_bwd_dx_v2 with CONTIG=False.
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    extra = num_blocks % tile_num
    start_block = pid
    step = tile_num
    if pid < extra:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w0 = tl.load(W0 + cols, mask=col_mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=col_mask, other=0.0).to(tl.float32)
    if NUM_IN == 3:
        w2 = tl.load(W2 + cols, mask=col_mask, other=0.0).to(tl.float32)
    else:
        w2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    acc_dw0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dw1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dw2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum2 = tl.zeros([BLOCK_D], dtype=tl.float32)

    for idx in tl.range(0, blocks_per_tile, num_stages=NUM_STAGES):
        current_block = start_block + idx * step
        start_row = current_block * BLOCK_N
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        rows64 = rows.to(tl.int64)
        m2 = row_mask[:, None] & (cols < BLOCK_D)[None, :]
        # The one dy read shared by the three problems.
        dy_block = tl.load(
            DY + rows64[:, None] * stride_dy + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        acc_dw0, acc_db0, acc_dxsum0 = _grp_ln_bwd_one(
            X0,
            DX0,
            Mean0,
            Rstd0,
            stride_x0,
            stride_dx0,
            w0,
            one0,
            dy_block,
            rows,
            rows64,
            row_mask,
            cols,
            m2,
            D,
            acc_dw0,
            acc_db0,
            acc_dxsum0,
        )
        acc_dw1, acc_db1, acc_dxsum1 = _grp_ln_bwd_one(
            X1,
            DX1,
            Mean1,
            Rstd1,
            stride_x1,
            stride_dx1,
            w1,
            one1,
            dy_block,
            rows,
            rows64,
            row_mask,
            cols,
            m2,
            D,
            acc_dw1,
            acc_db1,
            acc_dxsum1,
        )
        if NUM_IN == 3:
            acc_dw2, acc_db2, acc_dxsum2 = _grp_ln_bwd_one(
                X2,
                DX2,
                Mean2,
                Rstd2,
                stride_x2,
                stride_dx2,
                w2,
                one2,
                dy_block,
                rows,
                rows64,
                row_mask,
                cols,
                m2,
                D,
                acc_dw2,
                acc_db2,
                acc_dxsum2,
            )

    poff = pid.to(tl.int64) * D + cols
    tl.store(DW0 + poff, acc_dw0, mask=col_mask)
    tl.store(DB0 + poff, acc_db0, mask=col_mask)
    tl.store(DXSUM0 + poff, acc_dxsum0, mask=col_mask)
    tl.store(DW1 + poff, acc_dw1, mask=col_mask)
    tl.store(DB1 + poff, acc_db1, mask=col_mask)
    tl.store(DXSUM1 + poff, acc_dxsum1, mask=col_mask)
    if NUM_IN == 3:
        tl.store(DW2 + poff, acc_dw2, mask=col_mask)
        tl.store(DB2 + poff, acc_db2, mask=col_mask)
        tl.store(DXSUM2 + poff, acc_dxsum2, mask=col_mask)


def grouped_ln_bwd_dxsum_reason(
    dy: torch.Tensor,
    hs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    means: Sequence[torch.Tensor],
    rstds: Sequence[torch.Tensor],
    block_ds: Sequence[int],
) -> Optional[str]:
    """None iff grouped_ln_bwd_dxsum is proven bitwise for these inputs."""
    if not _wln_bwd_v2_enabled():
        return "DLRM_HSTU_FP8_WLN_BWD_V2 is off (the grouped kernel mirrors v2)"
    if (
        _wln_bwd_v2_config(GROUPED_BWD_DIM, False, False, True)
        != _V2_D512_SUMDX_CONFIG
    ):
        return "the v2 D=512 dx-sum launch config changed"
    if len(hs) not in (2, 3):
        return f"{len(hs)} problems (need 2 or 3)"
    if dy.dim() != 2 or dy.shape[1] != GROUPED_BWD_DIM or dy.stride(1) != 1:
        return f"dy {tuple(dy.shape)}/{dy.stride()} is not row-major [N, 512]"
    if dy.dtype != torch.bfloat16 or not dy.is_cuda:
        return f"dy is {dy.dtype} on {dy.device}"
    for k, h in enumerate(hs):
        if (
            h.shape != dy.shape
            or h.dtype != dy.dtype
            or h.stride(1) != 1
            or h.device != dy.device
        ):
            return f"h_{k} {tuple(h.shape)}/{h.dtype}/{h.stride()} != dy's"
        if block_ds[k] != GROUPED_BWD_DIM:
            return f"BLOCK_D_{k} = {block_ds[k]}"
        if weights[k].dtype != torch.float32 or weights[k].numel() != GROUPED_BWD_DIM:
            return f"weight_{k} is {weights[k].dtype}[{weights[k].numel()}]"
        if means[k].dtype != torch.float32 or rstds[k].dtype != torch.float32:
            return f"mean/rstd_{k} not fp32"
    return None


def grouped_ln_bwd_dxsum(
    dy: torch.Tensor,
    hs: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    biases: Sequence[torch.Tensor],
    means: Sequence[torch.Tensor],
    rstds: Sequence[torch.Tensor],
    inplace: bool = False,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per problem k, (dx_k, dweight_k, dbias_k, dxsum_k), bit-identical to
    triton_weighted_layer_norm_bwd_dxsum(dy, hs[k], weights[k], biases[k],
    means[k], rstds[k], eps, BLOCK_D=512, is_swish=False) on the v2 path.

    Call grouped_ln_bwd_dxsum_reason first; this function assumes it returned
    None. With inplace=True, dx_k is hs[k] (overwritten).
    """
    num_in = len(hs)
    N, D = dy.shape
    out: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    dxs = [h if inplace else torch.empty_like(h) for h in hs]
    finals = []
    for k in range(num_in):
        w = weights[k]
        finals.append(
            (
                torch.empty((D,), dtype=w.dtype, device=dy.device),
                torch.empty((D,), dtype=w.dtype, device=dy.device),
                # The preceding Linear's bias gradient: fp32, as shipped.
                torch.empty((D,), dtype=torch.float32, device=dy.device),
            )
        )
    if N == 0:
        for k in range(num_in):
            dweight, dbias, dxsum = finals[k]
            dweight.zero_()
            dbias.zero_()
            dxsum.zero_()
            out.append((dxs[k], dweight, dbias, dxsum))
        return out

    sms = torch.cuda.get_device_properties(dy.device).multi_processor_count
    mult, block_n, num_stages, _contig, num_warps, _fast = _V2_D512_SUMDX_CONFIG
    tile_num = max(1, min(sms * mult, triton.cdiv(N, 4 * block_n)))
    partials = [
        tuple(
            torch.empty((tile_num, D), dtype=torch.float32, device=dy.device)
            for _ in range(3)
        )
        for _ in range(num_in)
    ]
    # Problem 2's slots alias problem 1's when NUM_IN == 2 (never read or
    # written: every use is under NUM_IN == 3).
    k2 = 2 if num_in == 3 else 1
    _grouped_ln_bwd_dxsum_kernel[(tile_num,)](
        dy,
        dy.stride(0),
        hs[0],
        dxs[0],
        weights[0],
        means[0],
        rstds[0],
        partials[0][0],
        partials[0][1],
        partials[0][2],
        hs[0].stride(0),
        dxs[0].stride(0),
        1.0,
        hs[1],
        dxs[1],
        weights[1],
        means[1],
        rstds[1],
        partials[1][0],
        partials[1][1],
        partials[1][2],
        hs[1].stride(0),
        dxs[1].stride(0),
        1.0,
        hs[k2],
        dxs[k2],
        weights[k2],
        means[k2],
        rstds[k2],
        partials[k2][0],
        partials[k2][1],
        partials[k2][2],
        hs[k2].stride(0),
        dxs[k2].stride(0),
        1.0,
        D,
        N,
        NUM_IN=num_in,
        BLOCK_D=D,
        BLOCK_N=block_n,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )

    # The finalize, launched exactly as triton_weighted_layer_norm_bwd_dxsum
    # launches it (same grid, BLOCK_D and autotune key).
    def grid(META):
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    FINAL_BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
    FINAL_BLOCK_D = min(max(FINAL_BLOCK_D, 4), 128)
    for k in range(num_in):
        _dweight, _dbias, _dxsum = partials[k]
        dweight, dbias, dxsum = finals[k]
        _layer_norm_bwd_dwdbdx[grid](
            _dweight,
            _dbias,
            _dxsum,
            dweight,
            dbias,
            dxsum,
            tile_num,
            D,
            BLOCK_D=FINAL_BLOCK_D,
        )
        out.append((dxs[k], dweight, dbias, dxsum))
    return out

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Weight-gradient GEMM ``dW = X^T @ dY`` for a token-sized reduction.

WHY IT WAS WRITTEN, AND WHY IT IS NOT WIRED IN
----------------------------------------------
Every linear in the HSTU stack computes its weight gradient as ``X^T @ dY``, where X is
``[tokens, in_features]`` and dY is ``[tokens, out_features]``. Because the sequences are
jagged and flattened, ``tokens`` is ~2 million per rank per step while the output is only
weight-sized (256x512 up to 512x2048). That is a reduction, not a square GEMM: it streams
two huge operands into a tiny accumulator, so what matters is HBM traffic and how the
reduction over K is split. Vendor GEMM libraries tune for square-ish problems and treat
this corner as an afterthought.

The pinned torch 2.12 stack prices it badly. Profiling the yambda_5b run (step 52, 8
ranks) against the off-reference torch 2.10 stack, at identical shapes and identical
launch counts:

    role                   2.10    2.12   delta
    mm dW (K reduction)   24.35   35.37  +11.00 ms/rank/step
    addmm forward         56.25   52.12   -4.13   (2.12 is faster here)

dW is the only kernel family where 2.12 loses, and it loses more than the 4.23 ms the
whole step is behind by -- so fixing part of it is enough to put the pinned stack ahead of
the off-reference one. The dW GEMMs are 99% exclusive on the critical path (measured with
KernelForge's step-time-attribution trace analyzers), so the win transfers to step time
almost 1:1.

Both stacks select the *same* Tensile macro-tile (MT256x256x64_MI16x16x1) and differ only
in the generated kernel behind that name, so this is one library version emitting worse
code for the shape class. There is no configuration-level escape:
``TORCH_BLAS_PREFER_HIPBLASLT=1`` is a no-op because hipBLASLt is already the default on
both builds, and ``PYTORCH_TUNABLEOP_ENABLED`` cannot amortise because K is the token count
and changes every step. Pointing 2.12 at the 2.10 wheel's Tensile library was considered
and rejected: it risks a silent format/ABI mismatch between hipBLASLt versions, which is
the class of environment drift the provisioning marker exists to prevent.

WHAT ACTUALLY WON
-----------------
Two replacements were written and measured against each other on MI350X at K=1,982,017,
bf16, interleaved A/B over 5 rounds (interleaving matters: measured back-to-back, ATen's
own time for one shape moved by up to 40% run to run):

    dW shape            2.12 aten   chunked bmm   split-K triton   2.10 aten
    [512,K]x[K,2048]         5.50          4.30             9.25        3.99
    [1536,K]x[K,512]         5.06          3.38             7.21        3.14
    [512,K]x[K,1024]         3.58          1.80             5.45        2.05
    [256,K]x[K,1024]         1.81          1.32             2.92        1.27
    [256,K]x[K,512]          0.95          0.82             1.72        0.59
    [512,K]x[K,256]          0.78          0.80             1.75        0.60

The hand-written split-K Triton kernel below is correct and numerically indistinguishable
from ATen, but it is 1.7-2.2x *slower*, and the autotuner lands on essentially the tiling
rocBLAS already uses (BLOCK 128x256, BLOCK_K 64, SPLIT_K 16-32 against
MT256x256x64 + GSU16). Losing at the same tiling is a codegen and pipelining gap that
Triton is not going to close here, so it is kept only as a measured dead end.

What wins is a reformulation, ``chunked_bmm``: keep using the vendor GEMM but stop handing
it the shape it mishandles. Reshaping the reduction into a batch of medium-K GEMMs and
reducing the partials in fp32 is 1.17-1.99x faster than 2.12's single big-K call on every
shape with N >= 512.

Numerically all three paths are equivalent: against an fp64 reference they land at the
same relative error (3.2e-3), which is the bf16 quantum of the output rather than anything
to do with accumulation order. The two-level sum is better conditioned in principle -- it
measured slightly lower error on an fp64 comparison -- but the difference is below the
output's representable precision, so it should not be claimed as an improvement.

It does not beat 2.10 on every shape -- 2.10's own big-K kernel is good, and chunking is a
pessimisation there (0.75-0.93x), which is itself confirmation that the regression is
specific to 2.12's big-K path.

AND YET IT LOSES END TO END
---------------------------
Measured, 200 steps, one node, 8 ranks, steady state from step 60 on:

    fwd4      2.10 stack                        355.00 ms/step
    refstack  2.12, before the scatter fix      391.43
    gathersel 2.12 + gather-select              359.23
    dwbmm     2.12 + gather-select + bmm dW     370.71   <-- 3.2% WORSE

So ``bmm`` is refuted as an end-to-end win despite being 1.28-1.99x faster on the GEMM in
isolation. Both replacements here are dead ends, so the call sites in ``triton_addmm.py``
and ``triton_hstu_linear.py`` went back to plain ``torch.mm``: nothing in training reaches
this module, and ``HSTU_DW_GEMM`` no longer has anything to switch. It is kept for the
measurement and the two lessons below, reachable by calling ``dw_gemm`` directly (the
original ``dw_gemm_bench.py`` harness is in git history at ``33610d2``).

The trace says where it went (ms/rank/step, gathersel -> dwbmm). The dispatch worked as
intended: dW through ``aten::mm`` fell 177.7 -> 30.9 on rank 0 as the three big shapes
moved to ``aten::bmm``, and total Cijk GEMM time landed at 524.2 against the 2.10 stack's
524.0 -- GEMM parity, exactly the goal. But:

    GEMM (Cijk)     115.16 -> 109.64    -5.52
    collectives      34.79 ->  45.70   +10.91
    GPU idle          4.56 ->   7.56    +3.00  (1.3% -> 2.2% of the step)
    busy union      339.18 -> 338.73    -0.45

Two lessons worth carrying. The GEMM saving realised in situ (-5.52) was well under the
~9.4 projected from isolated timings, because the microbenchmark measured a tight loop over
pre-allocated tensors and not the extra workspace, the extra launches (view, transpose,
bmm, sum, tail mm in place of one mm) or their host-side cost -- which is what the +3.00 ms
of new GPU idle is.

More importantly, the critical-path argument that justified this work was too strong. The
counterfactual "if these kernels vanished the busy union shrinks by 34.96 ms, ratio 0.99"
is a true statement about the union and still a bad predictor of step time, because the
collectives on this path are *wait*-bound: their kernels sit spinning for the slowest rank,
so time freed elsewhere is absorbed by them rather than returned to the step. Union
exclusivity is necessary for a win but not sufficient, and nothing short of an end-to-end
run distinguishes the two.
"""

import os
from typing import List

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

from generative_recommenders.common import backend_allow_tf32, triton_autotune

# Below this reduction length neither replacement is worth it: a short K means the output
# tiles alone already fill the machine and the vendor kernel is fine.
MIN_REDUCTION: int = 65536

# chunked_bmm loses to ATen on narrow outputs ([512,K]x[K,256] and the 24-wide action MLP)
# because the batched GEMM's per-chunk output becomes too small to amortise its own
# launch. Every measured shape with N >= 512 wins.
MIN_BMM_N: int = 512

# Enough chunks to expose parallelism without shrinking each GEMM's own K too far. Swept
# over {4, 8, 16, 32, 64}: 16 is at or near the best on every shape, and accuracy improves
# monotonically with chunk count so there is no accuracy reason to go lower.
BMM_CHUNKS: int = 16


def dw_gemm_mode() -> str:
    """Which dW implementation to use: ``aten`` (default), ``bmm`` or ``triton``."""
    return os.environ.get("HSTU_DW_GEMM", "aten").lower()


def chunked_bmm(
    x: torch.Tensor, dy: torch.Tensor, chunks: int = BMM_CHUNKS
) -> torch.Tensor:
    """``x.t() @ dy`` as a batch of medium-K GEMMs with an fp32 reduction.

    Rather than asking the library for one GEMM with K in the millions -- the shape torch
    2.12's hipBLASLt prices badly -- this asks for ``chunks`` GEMMs with K a few hundred
    thousand, which its heuristics handle well, and reduces the partials afterwards.
    """
    k, m = x.shape
    n = dy.shape[1]
    # Keep each chunk's row count a multiple of 8 so every batch element starts on a
    # well-aligned boundary; the remainder is swept up by a single small mm below.
    per = (k // chunks) // 8 * 8
    used = per * chunks
    partials = torch.bmm(
        x[:used].view(chunks, per, m).transpose(1, 2),
        dy[:used].view(chunks, per, n),
    )
    # sum(dtype=fp32) accumulates in fp32 without materialising an fp32 copy of partials.
    out = partials.sum(0, dtype=torch.float32)
    if used < k:
        out += torch.mm(x[used:].t(), dy[used:]).float()
    return out.to(torch.promote_types(x.dtype, dy.dtype))


def _get_dw_configs() -> List[triton.Config]:
    """Tile candidates for the split-K kernel, biased towards large tiles.

    Operand re-reads dominate: each output tile re-reads its M-slice of X once per N-tile
    and its N-slice of dY once per M-tile, so traffic amplification over a single
    streaming pass is ``M*N*(1/BLOCK_M + 1/BLOCK_N) / (M+N)``. At the hot 512x2048 shape
    that is 6.4x for 128x128 tiles but 3.2x for 256x256, which is why the space leans on
    big tiles even though they spend most of the register file on the fp32 accumulator.

    SPLIT_K then has to be large enough to fill 256 CUs: 512x2048 in 256x256 tiles is only
    16 output tiles, so it takes SPLIT_K=16 to reach one workgroup per CU.
    """
    # (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)
    space = [
        (256, 256, 32, 8, 8, 2),
        (256, 256, 32, 16, 8, 2),
        (256, 256, 64, 8, 8, 2),
        (256, 256, 64, 16, 8, 2),
        (128, 256, 64, 16, 8, 2),
        (256, 128, 64, 16, 8, 2),
        (128, 256, 64, 32, 8, 2),
        (256, 128, 64, 32, 8, 2),
        (128, 128, 64, 32, 8, 2),
        (128, 128, 128, 16, 8, 2),
        (128, 128, 64, 16, 4, 2),
        (256, 256, 64, 8, 8, 3),
        (128, 256, 64, 16, 8, 3),
        (64, 256, 64, 32, 4, 2),
        (256, 64, 64, 32, 4, 2),
        (64, 64, 128, 32, 4, 2),
    ]
    configs = []
    for block_m, block_n, block_k, split_k, num_warps, num_stages in space:
        kwargs = {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "SPLIT_K": split_k,
        }
        if torch.version.hip:
            # kpack is deprecated on gfx950 and silently forced to 1, so it is left out.
            kwargs.update({"matrix_instr_nonkdim": 16, "waves_per_eu": 0})
        configs.append(
            triton.Config(
                kwargs,
                num_stages=num_stages if triton.__version__ >= "3.2.0" else 0,
                num_warps=num_warps,
            )
        )
    return configs


@triton_autotune(
    configs=_get_dw_configs(),
    key=["M", "N"],
    # The output is accumulated into, so every trial has to start from zero or the
    # autotuner's own repetitions would corrupt the result it leaves behind.
    reset_to_zero=["dw_ptr"],
    # K is in the millions, so one trial is milliseconds; the default 25/100 warmup and
    # rep would spend a minute per shape for no extra signal.
    warmup=3,
    rep=10,
)
@triton.jit
def _dw_gemm_split_k(
    x_ptr,
    dy_ptr,
    dw_ptr,
    M,
    N,
    K,
    stride_xk,
    stride_xm,
    stride_dyk,
    stride_dyn,
    stride_dwm,
    stride_dwn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
) -> None:
    """One ``[BLOCK_M, BLOCK_N]`` output tile per program, over a strided slice of K.

    ``SPLIT_K`` programs cooperate per tile so the machine stays full even though the
    output is tiny, and their partials are combined with ``tl.atomic_add`` into an fp32
    buffer. That makes the result nondeterministic in the same way ``scatter_add_``,
    ``index_add_`` and rocBLAS's own split-K reduction already are on this path, and it
    keeps ``SPLIT_K`` free for the autotuner -- a workspace-plus-reduction scheme would
    have to size the workspace before autotuning has chosen it.

    The autotune key above is deliberately ``["M", "N"]`` and *not* K: K is the token
    count and differs on every step (1990209, 2229337, 2284350, 2365056 and 2376002 all
    appear inside a single 4-step window), so keying on it would re-tune continuously and
    never amortise. M and N select the tile; K only sets the trip count.
    """
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K * stride overflows int32 at these token counts (2.4M rows x 2048 cols), so the
    # reduction offsets have to be 64-bit from the start.
    offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x_ptr + offs_k[:, None] * stride_xk + offs_m[None, :] * stride_xm
    dy_ptrs = dy_ptr + offs_k[:, None] * stride_dyk + offs_n[None, :] * stride_dyn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_step = BLOCK_K * SPLIT_K
    for k0 in range(0, tl.cdiv(K, k_step)):
        mask_k = offs_k + k0 * k_step < K
        # X is row-major [K, M], so loading [BLOCK_K, BLOCK_M] is coalesced and the
        # transpose happens in registers. Gathering [BLOCK_M, BLOCK_K] instead would
        # stride by M on the inner axis and destroy coalescing.
        x = tl.load(x_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
        dy = tl.load(dy_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(tl.trans(x), dy, allow_tf32=ALLOW_TF32)
        x_ptrs += k_step * stride_xk
        dy_ptrs += k_step * stride_dyk

    dw_ptrs = (
        dw_ptr + offs_m[:, None].to(tl.int64) * stride_dwm + offs_n[None, :] * stride_dwn
    )
    tl.atomic_add(dw_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_dw_gemm(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Compute ``x.t() @ dy`` with the split-K kernel.

    Args:
        x: ``[K, M]`` activations, row-major (the forward input, *not* pre-transposed).
        dy: ``[K, N]`` output gradients, row-major.

    Returns:
        ``[M, N]``, in the dtype ``torch.mm`` would have produced.
    """
    assert x.dim() == 2 and dy.dim() == 2, f"expected 2D, got {x.shape} and {dy.shape}"
    assert x.shape[0] == dy.shape[0], f"K mismatch: {x.shape} vs {dy.shape}"
    k, m = x.shape
    n = dy.shape[1]
    out_dtype = torch.promote_types(x.dtype, dy.dtype)

    # fp32 because the atomic combine needs a type that can hold a 2M-term sum; the cast
    # back matches torch.mm's output dtype.
    dw = torch.zeros((m, n), device=x.device, dtype=torch.float32)

    def grid(meta):  # pyre-ignore[3]
        return (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            meta["SPLIT_K"],
        )

    _dw_gemm_split_k[grid](
        x,
        dy,
        dw,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        dy.stride(0),
        dy.stride(1),
        dw.stride(0),
        dw.stride(1),
        ALLOW_TF32=backend_allow_tf32(),
    )
    return dw.to(out_dtype)


def dw_gemm(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """``x.t() @ dy``, routed to whichever implementation is the better tool.

    Falls back to ``torch.mm`` whenever the replacements' assumptions do not hold, so call
    sites can use this unconditionally.
    """
    mode = dw_gemm_mode()
    if (
        mode == "aten"
        or x.device.type != "cuda"
        or x.shape[0] < MIN_REDUCTION
        or x.dtype != dy.dtype
        # Non-contiguous operands would turn the coalesced loads into gathers, and the
        # bmm path needs to reshape the token axis.
        or not x.is_contiguous()
        or not dy.is_contiguous()
    ):
        return torch.mm(x.t(), dy)
    if mode == "bmm":
        if dy.shape[1] < MIN_BMM_N:
            return torch.mm(x.t(), dy)
        return chunked_bmm(x, dy)
    if mode == "triton":
        return triton_dw_gemm(x, dy)
    raise ValueError(
        f"HSTU_DW_GEMM={mode!r} is not one of 'aten', 'bmm', 'triton'"
    )

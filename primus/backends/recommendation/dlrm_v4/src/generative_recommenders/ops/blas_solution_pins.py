# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pin known-good hipBLASLt/rocBLAS solutions for the HSTU projection GEMMs.

hipBLASLt's default heuristic leaves a lot on the table for the projection
shapes: sweeping every registered solution, with a numerical check on, finds
1.06x-1.44x per call site across the five that shipped first, worth ~17
ms/step/rank, and a later sweep of the targets-only layer added a sixth at
1.35x-1.63x. Three of the six are bit-identical to the heuristic pick and three
differ by one to two bf16 ULPs; see the per-pin notes in _PINS. The obstacle to
shipping any of it is the cache key. TunableOp keys on the full problem
descriptor, which includes M, the total jagged token count -- and M changes
every step (2229337, 2376002, 2284350, 2365056 across four consecutive traced
steps), so a tuning file collected at one step never hits at the next.

Everything else in the key is M-independent, and the winning solutions proved
stable across token counts, so rather than tuning per step this synthesizes the
cache line for whichever token count is current and merges it via
``tunable.read_file``. Measured cost is ~0.3 ms of CPU per new token count,
which the GPU covers.

A pinned index only means anything for the library build it was measured on, so
the pins refuse to install unless the TunableOp validators match EXPECTED_ENV
exactly. On any mismatch this is a no-op and the default heuristic runs, which
is merely the current behaviour.

Enable with GR_PIN_BF16_BLAS_SOLUTIONS=1, which also requires
PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=0. To enable only some
call sites -- for bisecting which pin is responsible for a numerical problem --
pass a comma-separated list of the short keys in _PINS instead of 1, e.g.
GR_PIN_BF16_BLAS_SOLUTIONS=uvqk_dgrad,out_dgrad.

BF16 is in the name because it is a real limit, not a label. Every key in _PINS
is a BFloat16 TunableOp signature, so a pin fires only when that call site runs
in bf16 -- and with the FP8 forward arm on, FUSE_QKV and FUSE_OUTLN turn the
uvqk and output projections into torch._scaled_mm calls that no key here can
match. Four of the six still fire under FP8 (the backward stays bf16 while
DLRM_HSTU_FP8_BWD=0, and three fire when it is 1); uvqk_fwd and out_fwd do not.
Measured: the pins are worth
-3.66% against a bf16 control and -2.86% against an FP8 one.

Note on when a pin takes effect: TunableOp fixes an operator's solution the
first time it resolves a given problem key, so a pin is only honoured if it is
installed BEFORE the first GEMM at that token count. That is why the hook sits
ahead of the projection rather than anywhere convenient.

PER-STACK TABLES. Each library stack the trainer runs on has its own table,
guarded by its own TunableOp validators (_STACKS): the ROCm 7.2 / torch 2.12
rows of rocm/primus:v26.3 (EXPECTED_ENV, _PINS) are unchanged and install only
there; the ROCm 10.2 / torch 2.16 table (EXPECTED_ENV_ROCM102) installs only on
that stack. The request (GR_PIN_BF16_BLAS_SOLUTIONS) may name keys of any
table; a key of another stack's table is skipped with an info line, a key of no
table still raises. "1"/"all" means every row of the live stack's table.

FP8 SCALEDGEMM ROWS (ROCm 10.2 table, _SCALED_MM_PINS_ROCM102). hipBLASLt 10.2's
heuristic answers the FP8 projection dgrads (torch._scaled_mm, e5m2 gradient x
e4m3 weight -> bf16: uvqk [M,2048]x[2048,512] twice per step, targets-only V/K
[M,1024]x[1024,512] once) with MT256x240x128 (solution 32802) on about one token
count in ten between 1.15M and 2.43M -- 3 of 11 traced steps, 5-6 of 8 ranks,
about +600 us per call -- and with the K-rotated MT256x256x128 (32738) elsewhere.
A third row pins the preprocessor L2 dgrad ([M,512]x[512,256]) to the same
in-order solution so the X2L-B fused backward can match it bitwise (see the row).
Those rows CANNOT be TunableOp lines on torch 2.16: its
ScaledGemmTunableOp<Float8_e5m2, Float8_e4m3fn, BFloat16, TN> describes the
hipBLASLt problem with the A/B types swapped (the heuristic's own 32738/32802
are "unsupported" there, and the 53 solutions it does run all fail the 1e-2
numerics check), so a TunableOp line for them would name a solution the op
refuses. scaled_mm() below therefore runs the pinned solution itself, through
the hipBLASLt C API on torch's own libhipblaslt (the ctypes path of
ops/hipblaslt_fp8_nn.py, with torch._scaled_mm's descriptor: A = weight op T,
B = gradient op N, tensor-wise scale pointers, D bf16), and falls back to
torch._scaled_mm on any refusal. Same library, same kernel object, same
arguments as a torch call that picked that solution.
"""

import ctypes
import logging
import os
import tempfile
import threading
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

# The solution indices below were measured on exactly this stack. They are
# indices into that build's solution registry and are meaningless elsewhere.
EXPECTED_ENV: Dict[str, str] = {
    "PT_VERSION": "2.12.0",
    "HIP_VERSION": "702",
    "HIPBLASLT_VERSION": "100202-dabb6df2b9",
    "GCN_ARCH_NAME": "gfx950:sramecc+:xnack-",
    "ROCBLAS_VERSION": "5.2.0.dabb6df2b9",
}

# (call site, TunableOp op signature, key template on {M}, solution)
#
# These come from a sweep run with PYTORCH_TUNABLEOP_NUMERICAL_CHECK=1e-2_1e-2,
# so every solution here was checked against a reference during tuning and no
# candidate could win on speed alone. That matters: the first version of this
# table came from an unchecked sweep, and the solution it picked for uvqk_dgrad
# (Gemm_Rocblas_439683) computes the wrong answer for that shape. Pinning it
# alone drove eval NE to NaN within one eval, while the other five left the
# training curve bit-for-bit unchanged. Both rocBLAS picks are now gone; every
# entry is hipBLASLt.
#
# These are NOT bit-identical to the heuristic default, and cannot be: tuning
# the same shapes at 1e-30 tolerance returns 'Default' for uvqk_dgrad, i.e. no
# alternative solution reproduces it exactly. Expect differences of one to two
# bf16 ULPs, which is ordinary reassociation for this dtype.
#
# Re-run the sweep (scripts alongside this file) whenever EXPECTED_ENV changes.
#
# The MLP forward GEMM is deliberately absent: tuning picked a solution 1.7x
# SLOWER than the default there, so it keeps the heuristic pick.
_PINS: List[Tuple[str, str, str, str]] = [
    # 437878 is what the checked sweep picked here, but it is a tuning-time
    # measurement error: re-measured against a reference at six token counts it
    # runs 9.82/6.28/9.36/8.31 ms where the default runs 8.36/8.41/7.93/7.02,
    # i.e. mostly slower than doing nothing, against the 4.68 ms tuning claimed.
    # 437615 beats the default at every size (6.12/6.24/5.83/5.24) and produces
    # output identical to 437878's to the last digit at every size, so it is not
    # trading accuracy for speed -- same math, better schedule.
    ("uvqk_fwd", "GemmAndBiasTunableOp_BFloat16_NN",
     "nn_2048_{M}_512_ld_2048_512_2048", "Gemm_Hipblaslt_437615"),
    ("uvqk_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_2048_512_{M}_ld_2048_512_2048", "Gemm_Hipblaslt_436744"),
    # Kept for completeness but not worth enabling: 5.29 ms against a 5.33 ms
    # default is a wash, and it is the shape whose previous pin trained to NaN.
    ("uvqk_dgrad", "GemmTunableOp_BFloat16_TN",
     "tn_512_{M}_2048_ld_2048_2048_512", "Gemm_Hipblaslt_441461"),
    ("out_fwd", "GemmAndBiasTunableOp_BFloat16_NN",
     "nn_512_{M}_1536_ld_512_1536_512", "Gemm_Hipblaslt_437512"),
    ("out_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_512_1536_{M}_ld_512_1536_512", "Gemm_Hipblaslt_437445"),
    ("out_dgrad", "GemmTunableOp_BFloat16_TN",
     "tn_1536_{M}_512_ld_512_512_1536", "Gemm_Hipblaslt_441460"),
    # The targets-only last layer's full-row V/K projection backward
    # (## hstu_targets_full_vk_backward ##). Its N is the 1024-wide V/K pack
    # rather than uvqk's 2048, so it is a key of its own and was not in the
    # original sweep. It wants the same solution the uvqk wgrad does and, like
    # that one, is bit-identical to the default at every production token count
    # -- both run the same GSU16 split, so the reduction order does not move.
    # 1.35x-1.63x over the 1.99M-2.38M token counts the step-52 trace contains,
    # worth 1.0-1.4 ms/step/rank -- an offline figure, not an e2e one, and the
    # spread is the gap between scaling the trace's in-situ 4.05 ms by the
    # measured ratio and scaling by the measured absolute delta. The targets-only
    # forward does not call install_for_tokens; it does not have to, because it
    # is the last layer and the ordinary ones ahead of it register the same
    # count -- the traced M sets are identical. A targets-only-only model would
    # miss this pin silently.
    #
    # Small token counts are mixed rather than uniformly worse: measured at
    # 16K it is 1.5x FASTER (0.034 vs 0.052 ms), at 64K a wash (0.082 vs
    # 0.081), and at 1K it is 8.6x SLOWER (0.080 vs 0.009 ms). Every one of
    # those is under 0.08 ms absolute and two orders of magnitude below the
    # counts a yambda-5b step produces, but it is why this entry claims
    # "production token count" and not "any" -- the solution is tuned for a
    # K=M reduction long enough to amortise its non-temporal schedule.
    #
    # There is no matching dgrad entry. The checked sweep picked
    # Gemm_Hipblaslt_441462 and claimed 1.73 ms, but re-measured it runs
    # 3.57 ms against a 3.32 ms default over twelve paired points -- the same
    # tuning-time measurement error 437878 had on uvqk_fwd. The runner-up,
    # Gemm_Hipblaslt_441024, launches a kernel whose name is identical to the
    # default's, i.e. it is the heuristic pick under another index.
    ("targets_vk_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_1024_512_{M}_ld_1024_512_1024", "Gemm_Hipblaslt_436744"),
    # The item-embedding MLP's second Linear (512 -> 256), DATA-GRADIENT side:
    # grad_in[M,512] = grad_out[M,256] @ w[256,512]. Not an HSTU projection --
    # it is in DlrmHSTU._item_embedding_mlp -- but it is the largest single
    # small-K bf16 GEMM in the step.
    #
    # REACHABILITY IS NOT FREE HERE, and an earlier version of this comment got
    # it wrong. It claimed the dgrad "runs in the backward, after the install,
    # so it is in time". Ordering was never the problem -- the KEY was. The MLP
    # row count is not the HSTU token count: measured on 8 ranks x 5 steps of a
    # profiled run, num_tokens exceeds all_embeddings.shape[0] by exactly 8192
    # (== global_batch_size) at every one of 40 points, because preprocess()
    # appends one token per sample between the MLP and the projection. So the
    # line install_for_tokens() wrote was keyed on an M this GEMM never asks
    # about: 0/40 lookups hit, and TunableOp fell through to the heuristic.
    #
    # This was caught only by a trace, and only because the trace was taken. The
    # pin reported "active on 8/8 ranks" in every driver log and the arm still
    # ran the DEFAULT kernel -- Cijk_..._MT256x256x32_MI32x32x1_... byte for
    # byte identical to the control, 1.5395 vs 1.5875 ms, against a predicted
    # 2.14x. The corresponding 200-step e2e ladder read -0.15% because it was
    # comparing two identical configurations. INSTALLED IS NOT USED, and the
    # "N/8 pins active" line proves only the former.
    #
    # It is reachable now because modules/dlrm_hstu.py:_item_forward calls
    # install_for_tokens(all_embeddings.shape[0]) before the MLP. Remove that
    # call and this entry silently goes inert again.
    #
    # 2.17x, and unusually well-behaved: measured paired against the heuristic
    # at EIGHT token counts spanning 1.98M-2.42M, 3 rounds each (n=24), one
    # process per point. Per-count medians 2.055/2.058/2.084/2.168/2.173/
    # 2.177/2.183/2.208 -- every count wins, and both arms are monotone in M
    # with a within-count spread under 0.04. That across-M stability is not a
    # nicety, it is the precondition this whole file runs on: install_for_tokens
    # synthesises the line for WHATEVER count is current, so a pin fires at
    # every M and a candidate that wins only on a pooled median would be a
    # speedup plus a regression on the steps that land badly.
    #
    # THE SIBLING FORWARD PIN IS DELIBERATELY ABSENT, and not because it is
    # slow: b_fwd screens at 1.06x, which is inside the screen's own noise and
    # not worth a second variable. It is now reachable (the _item_forward
    # install precedes the MLP call), so if it is ever wanted it needs only a
    # measurement, not a plumbing change.
    #
    # The other three small-K candidates were all rejected on measurement:
    #  * d_fwd (multitask head, 1.22x pooled) LOSES at 3 of 8 token counts
    #    (0.872/0.885/0.917). The candidate is smooth and monotone in M; it is
    #    the DEFAULT that is bimodal, dropping to a fast path at those counts.
    #    A static pin gives up that adaptivity, so the pooled median is a
    #    fiction. Pinning is only safe where the default is uniformly mediocre.
    #  * c_fwd's 12.66x screen was an artifact: the pinned arm launches NO
    #    kernel and still "times" 0.0114 ms, implying 2370 TF/s against a
    #    measured 2106 TF/s bf16 peak. Where a kernel really runs it is ~1.05x.
    #  * a_dgrad is a null (2.68-2.81 ms against a 2.05-3.32 ms default):
    #    lower variance, not faster.
    #
    # This is the file's only rocBLAS entry, which the note above says were all
    # removed -- so it gets the scrutiny that history earns. It is NOT the
    # 439683 wrong-answer mode: 99.9928% of elements are bit-identical to the
    # default and the misses are monotone in smallness (elements above 1024 ULP
    # have |ref| 9.3e-10 against a typical 0.0269), max absolute error is one
    # bf16 ULP at the top of range, RMS SNR 94.1 dB. That is ordinary
    # reassociation, not a different answer.
    ("mlp_b_dgrad", "GemmTunableOp_BFloat16_NN",
     "nn_512_{M}_256_ld_512_256_512", "Gemm_Rocblas_437980"),
    # ---- ContextualPreprocessor MLP GEMMs (fused_linear_norm._LinearNormFunction):
    # content (Din=512), additional (Din=1024 = 2 x 512) and action (Din=24) MLPs,
    # each Linear(Din->256), SwishLN(256), Linear(256->512), LN(512). Swept iter29
    # (probes/iter29/gemm): TunableOp exhaustive sweep with a 1e-2 numerics check at
    # M=2334943, then the top 8 re-timed against the heuristic INTERLEAVED on one
    # MI350X (25 reps/arm, 512 MiB cache flush before every call) at six token
    # counts 1.70M-2.42M. Every default-ON entry below wins at all six counts.
    # Bit-identity: the wgrad entry is bit-identical at every count; the other three
    # differ from the heuristic on ~7e-5 of elements by <=2 bf16 ULP (fp32-order
    # differences, identical SNR vs an fp32 reference to 3 decimals).
    #
    # Reachability: these GEMMs run at the preprocessor's row count, which is
    # the item-MLP row count _item_forward already registers; preprocessors.py
    # also registers it itself so the pins do not depend on that ordering.
    #
    # NOTE preproc_dgrad_n512 shares its KEY with mlp_b_dgrad (same dims and layout
    # in another call site). One key can take only one solution, so _decide() refuses
    # to activate both.
    #
    # additional-MLP L1 dgrad dx[M,1024] = dh[M,256] @ w[256,1024]:
    # 4.41 -> 1.93 ms at 2.33M, 2.29x-2.40x over 1.7M-2.42M.
    ("preproc_dgrad_n1024", "GemmTunableOp_BFloat16_NN",
     "nn_1024_{M}_256_ld_1024_256_1024", "Gemm_Rocblas_437863"),
    # content-MLP L1 dgrad dx[M,512] = dh[M,256] @ w[256,512]:
    # 2.19 -> 1.02 ms at 2.33M, 2.04x-2.21x.
    ("preproc_dgrad_n512", "GemmTunableOp_BFloat16_NN",
     "nn_512_{M}_256_ld_512_256_512", "Gemm_Rocblas_437491"),
    # L2 forward h[M,512] = addmm(b, x[M,256], w[512,256].t()), three calls/step:
    # 1.65 -> 1.09 ms at 2.33M, 1.42x-1.62x.
    ("preproc_l2_fwd", "GemmAndBiasTunableOp_BFloat16_TN",
     "tn_512_{M}_256_ld_256_256_512", "Gemm_Hipblaslt_440245"),
    # additional-MLP L1 wgrad dW[256,1024] = dh.t() @ x: BIT-IDENTICAL,
    # 1.95 -> 1.41 ms at 2.33M, 1.26x-1.45x.
    ("preproc_wgrad_n1024", "GemmTunableOp_BFloat16_NT",
     "nt_1024_256_{M}_ld_1024_256_1024", "Gemm_Hipblaslt_436837"),
    # Below the 10% / 0.1 ms-per-call bar; listed, NOT in the gin default.
    # L2 wgrad dW[512,256] (3 calls/step), bit-identical, 1.09x-1.11x (-0.09 ms/call).
    ("preproc_l2_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_256_512_{M}_ld_256_512_256", "Gemm_Hipblaslt_436551"),
    # content L1 wgrad dW[256,512], bit-identical, 1.02x-1.32x.
    ("preproc_wgrad_n512", "GemmTunableOp_BFloat16_NT",
     "nt_512_256_{M}_ld_512_256_512", "Gemm_Hipblaslt_437446"),
    # action L1 fwd addmm [M,24]x[24,256], bit-identical, 1.16x-1.26x (-0.07 ms).
    ("preproc_fwd_k24", "GemmAndBiasTunableOp_BFloat16_TN",
     "tn_256_{M}_24_ld_24_24_256", "Gemm_Hipblaslt_441235"),
    # action L1 wgrad dW[256,24]: 1.36x-1.53x (-0.09 ms) but NOT bit-identical
    # (1.8e-3 of elements differ by 1 bf16 ULP).
    ("preproc_wgrad_k24", "GemmTunableOp_BFloat16_NT",
     "nt_24_256_{M}_ld_24_256_24", "Gemm_Hipblaslt_436761"),
]


# ---- ROCm 10.2 / torch 2.16 (UTD primus_rocm10.2_20261016, TheRock 10.2.0a20261015)
EXPECTED_ENV_ROCM102: Dict[str, str] = {
    "PT_VERSION": "2.16.0",
    "HIP_VERSION": "717",
    "HIPBLASLT_VERSION": "100500-3dc098eb",
    "GCN_ARCH_NAME": "gfx950:sramecc+:xnack-",
    "ROCBLAS_VERSION": "5.8.0.3dc098eb",
}

# No TunableOp rows: the BF16 GEMMs that are pinned on v26.3 run 0.05 ms/step
# FASTER on hipBLASLt 10.2's default heuristics than v26.3 does with its pins
# (traced, job 382863), so re-recording them gains nothing.
_PINS_ROCM102: List[Tuple[str, str, str, str]] = []


class ScaledMmPin(NamedTuple):
    """torch._scaled_mm(a [M, k] row-major, b [k, n] column-major, tensor-wise
    scale_a / scale_b, out_dtype), no bias / scale_result, run as hipBLASLt
    solution `solution` while m_min <= M <= m_max (see scaled_mm())."""

    name: str
    a_dtype: torch.dtype
    b_dtype: torch.dtype
    out_dtype: torch.dtype
    k: int
    n: int
    solution: int
    m_min: int
    m_max: int


# Selected with the numerically checked sweep of every solution hipBLASLt
# 100500 supports for the problem (509 at M=2365018, 476 of them workspace-free),
# run through the C API (scratch: rocm10_work/fp8-gemm-pins, fp8_dgrad_sweep.py)
# because TunableOp's own sweep cannot (module docstring). Each candidate was
# checked against an fp32 reference and the heuristic's output, timed, and its K
# order probed with exact-arithmetic operands (one nonzero product per 128-deep
# K block, cancelling pairs that expose any rotated or split accumulation).
#
# 33058 (Custom_Cijk_Alik_Bljk_F8B8BS_BH_SAB_NTD_UserArgs_shortname0_gfx950) is
# the fastest of all 509 for both projection shapes, needs no workspace, is supported at
# every M probed (1..65537 and every residue mod 256 at 1.0M and 2.3M), and
# accumulates K IN ORDER in every tile. The heuristic's 32738 starts each tile's
# K loop at (tile index mod 16) for K=2048 (mod 8 for K=1024): only 1/16 (1/8)
# of its tiles sum in order. Numerics at M=2365018 on gaussian FP8 operands:
# bit-identical to the heuristic's output (0 of 1.2G elements differ: these
# FP8 products are mostly exact in fp32, so the order rarely shows), SNR vs an
# fp32 reference 55.618 / 55.589 dB, equal to the heuristic's to 3 decimals.
#
# Time, interleaved against torch._scaled_mm at 49 token counts 1.5M-2.6M
# (every traced count, sampled): never slower, worst ratio 0.992 (uvqk) /
# 0.971 (V/K), 0.79 / 0.81 where the heuristic picks the 240 tile, mean
# -0.130 / -0.099 ms per call. At 15 counts 16K-4.19M (powers of two, where the
# heuristic's pick is the 256x256 tile): 0.953-0.982 (uvqk) / 0.882-0.953
# (V/K). The rows apply from 16K to 3M tokens (16384 <= M <= 3 * 2**20), where
# time and bitwise agreement with the heuristic were both measured (a yambda-5b
# step is 1.9M-2.5M per rank); outside it the heuristic runs.
#
# The same in-order solution is what the x2lb lane found makes the preproc
# L2-X2LB fused backward reproduce the shipped [M,512]x[512,256] dgrad bitwise
# (rocm10_work/x2lb, x2lb_korder.py); the third row pins that shape to it.
_E5M2 = torch.float8_e5m2
_E4M3 = torch.float8_e4m3fn
_SCALED_MM_PINS_ROCM102: List[ScaledMmPin] = [
    # hstu_ordinary_uvqk_backward d_normed_x = dz8 @ W8^T (ops/triton/bwd_fp8.py)
    ScaledMmPin("uvqk_dgrad_fp8", _E5M2, _E4M3, torch.bfloat16, 2048, 512,
                33058, 1 << 14, 3 << 20),
    # hstu_targets_full_vk_backward, the same call with the 1024-wide V/K pack
    ScaledMmPin("targets_vk_dgrad_fp8", _E5M2, _E4M3, torch.bfloat16, 1024, 512,
                33058, 1 << 14, 3 << 20),
    # ContextualPreprocessor MLP L2 dgrad da1 = dh2 @ W2 (ops/preproc_fp8_l2.py,
    # [M,512]x[512,256], through triton_fp8._scaled_mm, which also serves the
    # X2L-B phase-2 self-check). Pinned for the ARITHMETIC: the heuristic's
    # 32738 starts each tile's K loop at (tile index mod 4), so the X2L-B fused
    # backward (ops/hip/preproc_l2_x2lb.py), which sums K in order like
    # hipBLASLt 7.2's pick did, cannot reproduce it, and its phase-2 self-check
    # fails closed on 10.2 ("1767248 of 2356560 gated rows differ"). 33058 sums
    # K in order, so the shipped dgrad and X2L-B agree bitwise again and X2L-B
    # arms (8/8 ranks, native, job 383423). Time is a net win but NOT uniform,
    # unlike the two rows above: the heuristic alternates between 32738 and the
    # 240 tile here (240 on 419 of 1,081 counts 1.0-3.0M); 33058 is 0.86-0.88x
    # where it picks 240 and up to 1.047x (+20 us) where it picks 32738, mean
    # -0.019 ms/call over 49 counts 1.5-2.6M. In the yambda-5b config this GEMM
    # is off the training hot path (a trace shows only the three projection
    # dgrads as F8B8BS kernels per rank-step: the preprocessor backward takes the
    # grouped path); it runs in the X2L-B self-check and wherever the per-MLP
    # backward runs (job 383423: 200-step neutral, -0.075 +- 0.15 ms/step vs the
    # two-row table).
    ScaledMmPin("preproc_l2_dgrad_fp8", _E5M2, _E4M3, torch.bfloat16, 512, 256,
                33058, 1 << 14, 3 << 20),
]


# ---- ROCm 10.2 / torch 2.16 (rocm/primus:v26.9, TheRock 10.2.0a20261020). Same hipBLASLt release
# (1.5.0) as the UTD image, another build (cb261897), so a solution index is not guaranteed to name the
# same kernel. The FP8 rows are taken over with three runtime checks of their own: matmulIsAlgoSupported
# per (pin, M) (an unsupported index falls back to torch._scaled_mm), the X2L-B phase-2 bitwise
# self-check against the pinned dgrad (fails closed unless the solution sums K in order), and the
# 200-step throughput A/B (cluster B, 2026-11-06).
EXPECTED_ENV_PRIMUS_V26_9: Dict[str, str] = {
    "PT_VERSION": "2.16.0",
    "HIP_VERSION": "717",
    "HIPBLASLT_VERSION": "100500-cb261897",
    "GCN_ARCH_NAME": "gfx950:sramecc+:xnack-",
    "ROCBLAS_VERSION": "5.8.0.cb261897",
}


class _Stack(NamedTuple):
    name: str
    env: Dict[str, str]
    pins: List[Tuple[str, str, str, str]]  # TunableOp rows
    scaled_mm_pins: List[ScaledMmPin]  # rows run by scaled_mm()


_STACKS: List[_Stack] = [
    _Stack("ROCm 7.2 / torch 2.12 (rocm/primus:v26.3)", EXPECTED_ENV, _PINS, []),
    _Stack("ROCm 10.2 / torch 2.16 (UTD primus_rocm10.2_20261016)",
           EXPECTED_ENV_ROCM102, _PINS_ROCM102, _SCALED_MM_PINS_ROCM102),
    _Stack("ROCm 10.2 / torch 2.16 (rocm/primus:v26.9)",
           EXPECTED_ENV_PRIMUS_V26_9, _PINS_ROCM102, _SCALED_MM_PINS_ROCM102),
]

_active: List[Tuple[str, str, str, str]] = []
_active_smm: Dict[Tuple[torch.dtype, torch.dtype, torch.dtype, int, int], ScaledMmPin] = {}
_stack: Optional[_Stack] = None

_state: Optional[bool] = None  # None = not yet decided, False = disabled
_seen: Set[int] = set()
_path: Optional[str] = None


ENV_VAR: str = "GR_PIN_BF16_BLAS_SOLUTIONS"


def resolve_pin_request() -> str:
    """The requested pin set. Unset or \"0\" means off."""
    return os.environ.get(ENV_VAR, "0")


def _all_pin_names() -> List[str]:
    names: List[str] = []
    for st in _STACKS:
        names += [p[0] for p in st.pins] + [p.name for p in st.scaled_mm_pins]
    return list(dict.fromkeys(names))


def _decide() -> bool:
    """Decide once whether pinning is safe and requested, and which pins."""
    want = resolve_pin_request()
    if want in ("0", ""):
        return False

    global _active, _active_smm, _stack
    keys: Optional[Set[str]] = None
    if want not in ("1", "all"):
        keys = {k.strip() for k in want.split(",") if k.strip()}
        unknown = keys - set(_all_pin_names())
        if unknown:
            # FAIL LOUD. This used to warn and return False, which turned OFF
            # EVERY pin -- not just the unrecognised one -- and cost ~4% of step
            # while the launch log still said pins were requested. A perf
            # regression with no error anywhere is the worst possible outcome:
            # it reads as a result rather than a misconfiguration. A key naming
            # a pin that does not exist is a config error, so raise.
            raise ValueError(
                f"{ENV_VAR}={want!r} names unknown pin key(s) "
                f"{', '.join(sorted(unknown))}. Valid keys: "
                f"{', '.join(_all_pin_names())}. "
                "Refusing to run: an unknown key would silently disable ALL "
                "pins (~4% of step) rather than just the one named."
            )
    # Two pins on one (signature, key) would race: TunableOp keeps the FIRST line it
    # reads for a key, so the second solution would be silently ignored. Checked
    # per stack, on the rows the request names, before the environment checks.
    for st in _STACKS:
        if keys is None:
            # "all" keeps the FIRST pin per (signature, key): mlp_b_dgrad and
            # preproc_dgrad_n512 share one key, and the check below refuses both.
            _first: Dict[Tuple[str, str], str] = {}
            rows = [p for p in st.pins
                    if _first.setdefault((p[1], p[2]), p[0]) == p[0]]
        else:
            rows = [p for p in st.pins if p[0] in keys]
        seen_keys: Dict[Tuple[str, str], str] = {}
        for name, sig, tmpl, _sol in rows:
            other = seen_keys.setdefault((sig, tmpl), name)
            if other != name:
                raise ValueError(
                    f"{ENV_VAR}: pins {other!r} and {name!r} share the key "
                    f"{sig}({tmpl}); enable only one of them."
                )
    # DELIBERATELY SOFT, unlike the checks that raise. This one and the
    # validator match below are ENVIRONMENT differences, not config errors: the
    # same gin default has to be importable on a non-ROCm box and on a library
    # build no table was measured on, where the right behaviour is the
    # documented no-op ("on any mismatch this is a no-op and the default
    # heuristic runs"). Do not convert these two to raises.
    if not torch.cuda.is_available() or not torch.version.hip:
        logger.warning("BLAS solution pins requested but this is not a ROCm build")
        return False

    import torch.cuda.tunable as tunable

    live = dict(tunable.get_validators())
    stack = next((st for st in _STACKS
                  if all(live.get(k) == v for k, v in st.env.items())), None)
    if stack is None:
        logger.warning(
            "BLAS solution pins disabled: no pin table was measured on this "
            "stack, so applying one here could select a different kernel. "
            "Live: %s. Tables: %s. Re-tune and add a table to enable.",
            ", ".join(f"{k}={live.get(k)!r}" for k in EXPECTED_ENV),
            "; ".join(f"{st.name} ({', '.join(f'{k}={v}' for k, v in st.env.items())})"
                      for st in _STACKS),
        )
        return False
    _stack = stack
    if keys is None:
        _first2: Dict[Tuple[str, str], str] = {}
        _active = [p for p in stack.pins
                   if _first2.setdefault((p[1], p[2]), p[0]) == p[0]]
        smm = list(stack.scaled_mm_pins)
    else:
        _active = [p for p in stack.pins if p[0] in keys]
        smm = [p for p in stack.scaled_mm_pins if p.name in keys]
    _active_smm = {(p.a_dtype, p.b_dtype, p.out_dtype, p.k, p.n): p for p in smm}
    if keys is not None:
        other_stack = sorted(keys - {p[0] for p in stack.pins}
                             - {p.name for p in stack.scaled_mm_pins})
        if other_stack:
            # Rows of another stack's table: the gin default names both stacks'
            # keys, and each stack applies its own.
            logger.info("BLAS solution pins not in the %s table (measured on "
                        "another stack), not applied: %s",
                        stack.name, ", ".join(other_stack))

    if _active:
        # Both of the next two are "you asked for pins and will not get them",
        # the same class as an unknown key, so they raise rather than warn.
        # Note apply_env_bootstrap already setdefault()s both of these whenever
        # pins are requested, so reaching either branch means a caller
        # overrode them explicitly -- a config error, not an environment
        # difference. Only the TunableOp rows need TunableOp; the ScaledGemm
        # rows are run by scaled_mm() itself.
        if not tunable.is_enabled():
            raise RuntimeError(
                f"{ENV_VAR}={want!r} requested but TunableOp is OFF, so every pin "
                "would be silently ignored. Set PYTORCH_TUNABLEOP_ENABLED=1 (or "
                f"{ENV_VAR}=0 to disable pinning deliberately)."
            )
        if tunable.tuning_is_enabled():
            raise RuntimeError(
                f"{ENV_VAR}={want!r} requested with TunableOp tuning ON. The sweep "
                "overrides the pins and costs minutes per new token count -- and M "
                "changes every step here. Set PYTORCH_TUNABLEOP_TUNING=0."
            )
        global _path
        _path = os.path.join(tempfile.mkdtemp(prefix="gr_blas_pins_"), "pins.csv")
    n_table = len(stack.pins) + len(stack.scaled_mm_pins)
    logger.info(
        "BLAS solution pins (%s table) active for %d/%d GEMMs: %s",
        stack.name, len(_active) + len(smm), n_table,
        ", ".join([p[0] for p in _active] + [p.name for p in smm]) or "none",
    )
    return bool(_active or smm)


def install_for_tokens(num_tokens: int) -> None:
    """Make the pinned solutions available for this batch's token count.

    Safe to call on every layer of every step: the work happens once per token
    count and is skipped otherwise. Installs the live stack's TunableOp rows;
    its ScaledGemm rows need no install (scaled_mm() runs them per call).
    """
    global _state
    if _state is None:
        _state = _decide()
    if not _state or not _active or num_tokens in _seen:
        return
    _seen.add(num_tokens)

    import torch.cuda.tunable as tunable

    assert _stack is not None
    lines = [f"Validator,{k},{v}" for k, v in _stack.env.items()]
    lines += [
        f"{sig},{tmpl.format(M=num_tokens)},{sol},0.0"
        for _, sig, tmpl, sol in _active
    ]
    assert _path is not None
    try:
        with open(_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        tunable.read_file(_path)
        # GR_PIN_BF16_BLAS_DEBUG=1 reports which token counts were actually
        # registered. "N/8 pins active" is printed once by _decide() and proves
        # only that keys were REQUESTED; it says nothing about the M they were
        # synthesised at. A pin whose GEMM runs at an M that never reaches this
        # function is a silent no-op -- that is exactly how mlp_b_dgrad shipped
        # "active on 8/8 ranks" while running the default kernel. One line per
        # distinct count (so ~1-2 per step), which is why it is off by default.
        if os.environ.get("GR_PIN_BF16_BLAS_DEBUG") == "1":
            logger.info("BLAS pins installed at M=%d for %d key(s): %s",
                        num_tokens, len(_active),
                        ", ".join(p[0] for p in _active))
    except Exception:
        # A failed merge just means the default heuristic runs; never take the
        # training job down over a performance hint.
        logger.warning("failed to install BLAS pins for %d tokens", num_tokens,
                       exc_info=True)


# ---------------------------------------------------------------------------
# ScaledGemm rows: the pinned hipBLASLt solution through the C API.
# ---------------------------------------------------------------------------
# hipBLASLt enum values (hipblaslt.h / hipblas-common.h / library_types.h; the
# same values in ROCm 7.2 and 10.2, see ops/hipblaslt_fp8_nn.py).
_OP_N, _OP_T = 111, 112
_COMPUTE_32F = 2
_R_32F = 0
_DESC_TRANSA, _DESC_TRANSB = 0, 1
_DESC_A_SCALE_POINTER, _DESC_B_SCALE_POINTER = 5, 6
_DESC_A_SCALE_MODE, _DESC_B_SCALE_MODE = 31, 32
_HIP_TYPE: Dict[torch.dtype, int] = {
    torch.float8_e4m3fn: 28, torch.float8_e5m2: 29, torch.bfloat16: 14,
}
# hipblaslt_ext::matmulIsAlgoSupported(handle, desc, alpha, A, B, beta, C, D,
# algo&, workspace&). C++ linkage, but every argument is a pointer or a
# reference, so ctypes can call the exported symbol. hipblasLtMatmul itself does
# NOT check that an algo fits the problem (measured: solution 33058 runs, status
# 0, on a K=1000 problem and on swapped A/B types), so this check is what keeps
# a pinned index off a problem it does not support.
_IS_SUPPORTED = ("_ZN13hipblaslt_ext21matmulIsAlgoSupportedEPvP27hipblasLtMatmulDesc"
                 "Opaque_tPKvP29hipblasLtMatrixLayoutOpaque_tS6_S4_S6_S6_R22_hipblas"
                 "LtMatmulAlgo_tRm")

_smm_lock = threading.Lock()
_smm_fn: Optional[Tuple[ctypes.CDLL, object]] = None
# (device, pin, M) -> (algo bytes, workspace it needs), or None = not supported
_smm_algo: Dict[Tuple[int, str, int], Optional[Tuple[bytes, int]]] = {}
_smm_broken: Dict[str, str] = {}  # pin name -> why it is latched off
_smm_engaged: Set[Tuple[str, int]] = set()


def _smm_library() -> Tuple[ctypes.CDLL, object]:
    global _smm_fn
    if _smm_fn is None:
        from generative_recommenders.ops import hipblaslt_fp8_nn as nn

        lib = nn._library()  # torch's own libhipblaslt, C API argtypes set
        fn = getattr(lib, _IS_SUPPORTED)
        fn.restype = ctypes.c_int
        vp = ctypes.c_void_p
        fn.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp,
                       ctypes.POINTER(nn._Algo), ctypes.POINTER(ctypes.c_size_t)]
        _smm_fn = (lib, fn)
    return _smm_fn


def _latch(pin: ScaledMmPin, why: str) -> None:
    with _smm_lock:
        if pin.name not in _smm_broken:
            _smm_broken[pin.name] = why
            logger.warning("[blas-pins] ScaledGemm pin %s DISABLED (torch._scaled_mm "
                           "from now on): %s", pin.name, why)


def _scaled_mm_pin(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor,
                   scale_b: torch.Tensor, out_dtype: torch.dtype) -> Optional[ScaledMmPin]:
    global _state
    if _state is None:
        _state = _decide()
    if not _state or not _active_smm or a.dim() != 2 or b.dim() != 2:
        return None
    m, k = a.shape
    pin = _active_smm.get((a.dtype, b.dtype, out_dtype, k, b.shape[1]))
    if (
        pin is None
        or pin.name in _smm_broken
        or b.shape[0] != k
        or not (pin.m_min <= m <= pin.m_max)
        # torch's TN call: row-major A, column-major B (both K-contiguous)
        or a.stride() != (k, 1)
        or b.stride() != (1, k)
        or scale_a.numel() != 1
        or scale_b.numel() != 1
        or scale_a.dtype != torch.float32
        or scale_b.dtype != torch.float32
        or not a.is_cuda
        or not (b.device == a.device == scale_a.device == scale_b.device)
        or a.data_ptr() % 16 != 0
        or b.data_ptr() % 16 != 0
        or torch.compiler.is_compiling()
    ):
        return None
    return pin


def _run_scaled_mm_pin(pin: ScaledMmPin, a: torch.Tensor, b: torch.Tensor,
                       scale_a: torch.Tensor, scale_b: torch.Tensor) -> Optional[torch.Tensor]:
    """torch._scaled_mm's hipBLASLt call (TN: A = b = weight [K, N] column-major,
    op T, lda K; B = a = gradient [M, K] row-major, op N, ldb K; D [M, N] row-major
    = [N, M] column-major, ld N; scale A = scale_b, scale B = scale_a, tensor-wise)
    with the pinned solution. None when the library refuses it."""
    from generative_recommenders.ops import hipblaslt_fp8_nn as nn

    try:
        lib, is_supported = _smm_library()
        handle, ws = nn._handle_and_workspace(a.device)
    except Exception as error:
        _latch(pin, f"hipBLASLt C API unavailable: {type(error).__name__}: {error}")
        return None
    m, k = a.shape
    n = pin.n
    vp = ctypes.c_void_p
    desc, la, lb, lc, ld = vp(), vp(), vp(), vp(), vp()
    made: List[Tuple[object, vp]] = []
    try:
        nn._check(lib.hipblasLtMatmulDescCreate(ctypes.byref(desc), _COMPUTE_32F, _R_32F), "DescCreate")
        made.append((lib.hipblasLtMatmulDescDestroy, desc))
        for attr, op in ((_DESC_TRANSA, _OP_T), (_DESC_TRANSB, _OP_N)):
            o = ctypes.c_int(op)
            nn._check(lib.hipblasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(o), 4), "SetTrans")
        sa_ptr, sb_ptr = vp(scale_b.data_ptr()), vp(scale_a.data_ptr())
        nn._check(lib.hipblasLtMatmulDescSetAttribute(desc, _DESC_A_SCALE_POINTER, ctypes.byref(sa_ptr), 8), "ScaleA")
        nn._check(lib.hipblasLtMatmulDescSetAttribute(desc, _DESC_B_SCALE_POINTER, ctypes.byref(sb_ptr), 8), "ScaleB")
        zero = ctypes.c_int(0)  # tensor-wise fp32 scalar, as torch sets it (status ignored, as torch does)
        lib.hipblasLtMatmulDescSetAttribute(desc, _DESC_A_SCALE_MODE, ctypes.byref(zero), 4)
        lib.hipblasLtMatmulDescSetAttribute(desc, _DESC_B_SCALE_MODE, ctypes.byref(zero), 4)
        for lay, (t, rows, cols, lead) in (
            (la, (_HIP_TYPE[b.dtype], k, n, k)),
            (lb, (_HIP_TYPE[a.dtype], k, m, k)),
            (lc, (_HIP_TYPE[pin.out_dtype], n, m, n)),
            (ld, (_HIP_TYPE[pin.out_dtype], n, m, n)),
        ):
            nn._check(lib.hipblasLtMatrixLayoutCreate(ctypes.byref(lay), t, rows, cols, lead), "Layout")
            made.append((lib.hipblasLtMatrixLayoutDestroy, lay))
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        key = (a.device.index if a.device.index is not None else torch.cuda.current_device(), pin.name, m)
        hit = _smm_algo.get(key, ())
        if hit == ():
            # getAlgosFromIndex's record for an index: the index in data[0:4],
            # zeros, max_workspace_bytes = SIZE_MAX (same bytes, measured).
            algo = nn._Algo()
            ctypes.memmove(ctypes.addressof(algo), int(pin.solution).to_bytes(4, "little"), 4)
            algo.max_workspace_bytes = (1 << 64) - 1
            need = ctypes.c_size_t(0)
            st = is_supported(handle, desc, ctypes.byref(alpha), la, lb, ctypes.byref(beta), lc, ld,
                              ctypes.byref(algo), ctypes.byref(need))
            ok = st == 0 and need.value <= ws.numel()
            hit = (ctypes.string_at(ctypes.addressof(algo), ctypes.sizeof(algo)), int(need.value)) if ok else None
            with _smm_lock:
                if len(_smm_algo) > 4096:
                    _smm_algo.clear()
                _smm_algo[key] = hit
            if not ok:
                logger.warning("[blas-pins] ScaledGemm pin %s: hipBLASLt solution %d not supported at "
                               "M=%d K=%d N=%d (status %d, workspace %d) -> torch._scaled_mm",
                               pin.name, pin.solution, m, k, n, st, need.value)
        if hit is None:
            return None
        algo = nn._Algo.from_buffer_copy(hit[0])
        out = torch.empty((m, n), dtype=pin.out_dtype, device=a.device)
        stream = torch.cuda.current_stream(a.device).cuda_stream
        status = lib.hipblasLtMatmul(
            handle, desc, ctypes.byref(alpha), vp(b.data_ptr()), la, vp(a.data_ptr()), lb,
            ctypes.byref(beta), vp(out.data_ptr()), lc, vp(out.data_ptr()), ld,
            ctypes.byref(algo), vp(ws.data_ptr()), ctypes.c_size_t(hit[1]), vp(stream))
        if status != 0:
            _latch(pin, f"hipblasLtMatmul status {status} at M={m}")
            return None
    except Exception as error:  # never take training down over a performance pin
        _latch(pin, f"{type(error).__name__}: {error}")
        return None
    finally:
        for destroy, h in reversed(made):
            destroy(h)
    dev = key[0]
    if (pin.name, dev) not in _smm_engaged:
        with _smm_lock:
            _smm_engaged.add((pin.name, dev))
        # Proof of use, not of install: printed from the first call that ran the
        # pinned solution on this device, as one atomic write per rank (a plain
        # print() was torn by another rank's output in 1 of 24 lines, job 383505).
        from generative_recommenders.ops.utils import arm_print_all_ranks

        arm_print_all_ranks(
            f"[blas-pins] ScaledGemm pin {pin.name} ENGAGED on cuda:{dev}: hipBLASLt "
            f"solution {pin.solution} ({_stack.name if _stack else '?'}), M={m} K={k} N={n}")
    return out


def scaled_mm_pinned(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """scaled_mm() through the live stack's pinned solution, or None when no
    ScaledGemm row applies (or the library refused it): the caller runs its own
    torch._scaled_mm then (triton_fp8._scaled_mm keeps its quiet-stderr call)."""
    pin = _scaled_mm_pin(a, b, scale_a, scale_b, out_dtype)
    return None if pin is None else _run_scaled_mm_pin(pin, a, b, scale_a, scale_b)


def scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """torch._scaled_mm(a, b, scale_a=..., scale_b=..., out_dtype=...) (tensor-wise
    scales, no bias / scale_result), run as the live stack's pinned hipBLASLt
    solution when its table has a ScaledGemm row for this problem, else (and on
    any refusal) as torch._scaled_mm itself."""
    out = scaled_mm_pinned(a, b, scale_a, scale_b, out_dtype=out_dtype)
    if out is not None:
        return out
    return torch._scaled_mm(a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=out_dtype)

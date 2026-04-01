###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Triton FP8 Grouped GEMM for MoE Expert computation on AMD MI355X / gfx950.

Implements:  C[e] = A[tokens_e]  @  B[e]   for each expert e
  - A : [total_M, K]   BF16  – concatenated token activations (sorted by expert)
  - B : [E, K, N]      BF16  – stacked expert weight matrices
  - C : [total_M, N]   BF16  – output

FP8 path (USE_FP8=True):
  * Per-expert dynamic quantisation of A (activation) to float8_e4m3fnuz
  * Per-tensor quantisation of B (weight)  to float8_e4m3fnuz
  * tl.dot with FP8 inputs, FP32 accumulation
  * Dequantisation in-kernel before writing BF16 output

Backward (inside Fp8GroupedGemmFunction):
  * dA and dB (weight grad) use standard BF16 grouped GEMM
    (uses `grouped_gemm` package when available, otherwise torch.bmm fallback)
  * On CUDA, dA and dB run on two streams by default for overlap
    (``PRIMUS_MOE_EXPERT_GEMM_BWD_DUAL_STREAM=0`` to disable).

AMD gfx950 (MI355X) dtype notes
  - triton: tl.float8e4nv  ← OCP E4M3FN, NATIVE on gfx950 (no upcast)
  - torch : torch.float8_e4m3fn
  - max representable value: 448.0
  NOTE: tl.float8e4b8 (AMD fnuz) is gfx942-specific; on gfx950 it is
        upcasted to fp16 → no speedup. Always use tl.float8e4nv on MI355X.
"""

import math
import os
from typing import List, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

try:
    from megatron.core.transformer.moe.grouped_gemm_util import ops as _gg_ops

    HAVE_GG = True
except ImportError:
    HAVE_GG = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_E4M3_MAX: float = 448.0
# OCP E4M3FN — native FP8 on AMD gfx950 (MI355X), matches tl.float8e4nv
_TORCH_FP8 = torch.float8_e4m3fn

# Block size for the per-expert quantization kernels (number of elements per
# Triton program along the K dimension).  Must be a power-of-two.
_QUANT_BLOCK_K: int = 512


# ---------------------------------------------------------------------------
# Triton kernels for fused per-expert activation quantization
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _quant_a_amax_row_kernel(
        X_ptr,            # [total_M, K]  BF16 (row-major)
        row_amax_ptr,     # [total_M]     float32  output: per-row amax
        stride_xm,        # stride along M dim (== K for row-major)
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Pass 1a: compute per-row amax(|X|).

        Grid: (total_M,)  — one program per row.
        Each program reads one row in K-blocks and writes a scalar amax.
        """
        row = tl.program_id(0)
        row_ptr = X_ptr + row * stride_xm

        running_amax = tl.full((), 0.0, dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            mask   = k_offs < K
            x_tile = tl.load(row_ptr + k_offs, mask=mask, other=0.0).to(tl.float32)
            tile_amax = tl.max(tl.abs(x_tile), axis=0)
            running_amax = tl.maximum(running_amax, tile_amax)

        tl.store(row_amax_ptr + row, running_amax)

    @triton.jit
    def _quant_a_expert_amax_kernel(
        row_amax_ptr,     # [total_M]  float32  per-row amax (from pass 1a)
        amax_ptr,         # [E]        float32  output: per-expert amax
        expert_starts_ptr,  # [E+1]    int32
        MAX_ROWS_PER_EXPERT: tl.constexpr,  # upper bound for loop unrolling
    ):
        """Pass 1b: reduce per-row amax to per-expert amax.

        Grid: (E,)  — one program per expert.
        """
        eid = tl.program_id(0)
        e_start = tl.load(expert_starts_ptr + eid)
        e_end   = tl.load(expert_starts_ptr + eid + 1)
        e_M     = e_end - e_start

        running_amax = tl.full((), 0.0, dtype=tl.float32)
        for i in range(MAX_ROWS_PER_EXPERT):
            active = i < e_M
            row_val = tl.load(row_amax_ptr + e_start + i, mask=active, other=0.0)
            running_amax = tl.maximum(running_amax, row_val)

        running_amax = tl.maximum(running_amax, 1e-12)
        tl.store(amax_ptr + eid, running_amax)

    @triton.jit
    def _quant_a_cast_row_kernel(
        X_ptr,            # [total_M, K]  BF16  input
        Xfp8_ptr,         # [total_M, K]  FP8   output
        scale_ptr,        # [total_M]     float32  per-row scale (E4M3_MAX / amax_expert)
        stride_xm,        # stride along M dim
        K: tl.constexpr,
        E4M3_MAX: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Pass 2: scale + clamp + cast BF16 → FP8, one program per row.

        Grid: (total_M,)
        """
        row = tl.program_id(0)
        scale = tl.load(scale_ptr + row)

        in_ptr  = X_ptr    + row * stride_xm
        out_ptr = Xfp8_ptr + row * stride_xm
        for k0 in range(0, K, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            mask   = k_offs < K
            x_tile = tl.load(in_ptr + k_offs, mask=mask, other=0.0).to(tl.float32)
            x_scaled = tl.clamp(x_tile * scale, -E4M3_MAX, E4M3_MAX)
            tl.store(out_ptr + k_offs, x_scaled.to(tl.float8e4nv), mask=mask)


def _per_expert_quant_A_triton(
    x: torch.Tensor,
    expert_starts_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused per-expert activation quantization using row-parallel Triton kernels.

    Three kernel launches:
      1. Per-row amax:   Grid=(total_M,) — fully parallel across all rows
      2. Per-expert amax reduce: Grid=(E,) — reduce row amaxes per expert
      3. Per-row cast:   Grid=(total_M,) — scale+clamp+cast fully parallel

    Args:
        x:                    [total_M, K]  BF16  (must be contiguous)
        expert_starts_tensor: [E+1]         int32 on GPU (cumulative token offsets)

    Returns:
        x_fp8:      [total_M, K]  float8_e4m3fn
        scale_invs: [E]           float32  (scale_inv = amax / E4M3_MAX)
    """
    total_M, K = x.shape
    E = expert_starts_tensor.shape[0] - 1

    x_fp8       = torch.empty_like(x, dtype=_TORCH_FP8)
    row_amax    = torch.empty(total_M, dtype=torch.float32, device=x.device)
    expert_amax = torch.empty(E, dtype=torch.float32, device=x.device)
    scale_invs  = torch.empty(E, dtype=torch.float32, device=x.device)

    # Pass 1a: per-row amax (fully parallel, Grid = total_M)
    _quant_a_amax_row_kernel[(total_M,)](
        x, row_amax,
        x.stride(0),
        K=K,
        BLOCK_K=_QUANT_BLOCK_K,
    )

    # Pass 1b: reduce per-row → per-expert amax (Grid = E)
    # MAX_ROWS_PER_EXPERT must be a constexpr upper bound; use next power-of-2
    # of the actual max tokens per expert (capped at total_M for safety).
    tpe_list = (expert_starts_tensor[1:] - expert_starts_tensor[:-1]).tolist()
    max_rows = max(int(m) for m in tpe_list) if tpe_list else 1
    # round up to power-of-2 for constexpr
    max_rows_pow2 = 1
    while max_rows_pow2 < max_rows:
        max_rows_pow2 *= 2

    _quant_a_expert_amax_kernel[(E,)](
        row_amax, expert_amax, expert_starts_tensor,
        MAX_ROWS_PER_EXPERT=max_rows_pow2,
    )

    # Build per-row scale tensor: scale[row] = E4M3_MAX / expert_amax[expert_of_row]
    # Done on GPU via index_select using a row→expert mapping.
    # Construct row_to_expert mapping from expert_starts_tensor.
    row_to_expert = torch.empty(total_M, dtype=torch.int64, device=x.device)
    es_cpu = expert_starts_tensor.tolist()
    for e in range(E):
        s, t = es_cpu[e], es_cpu[e + 1]
        if s < t:
            row_to_expert[s:t] = e

    row_scale = _E4M3_MAX / expert_amax[row_to_expert]  # [total_M] float32

    # Pass 2: per-row scale + cast (fully parallel, Grid = total_M)
    _quant_a_cast_row_kernel[(total_M,)](
        x, x_fp8, row_scale,
        x.stride(0),
        K=K,
        E4M3_MAX=_E4M3_MAX,
        BLOCK_K=_QUANT_BLOCK_K,
    )

    # scale_invs[e] = expert_amax[e] / E4M3_MAX
    scale_invs = expert_amax / _E4M3_MAX

    return x_fp8, scale_invs


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def _per_expert_quant_A(
    x: torch.Tensor,
    expert_starts: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise activation tensor per expert.

    Uses a fused Triton kernel when Triton is available (fast path), otherwise
    falls back to the original Python for-loop.

    Set ``PRIMUS_QUANT_A_USE_PYTHON=1`` to force the Python fallback (for A/B
    benchmarking or debugging).

    Args:
        x:             [total_M, K]  BF16
        expert_starts: [E+1] Python ints  (cumulative token offsets)

    Returns:
        x_fp8:      [total_M, K]  float8_e4m3fn
        scale_invs: [E]           float32  (scale_inv = amax / E4M3_MAX)
                    Dequantise:  x_bf16 ≈ x_fp8 * scale_invs[e]
    """
    use_python = os.getenv("PRIMUS_QUANT_A_USE_PYTHON", "0") == "1"
    if HAVE_TRITON and x.is_cuda and x.is_contiguous() and not use_python:
        es_tensor = torch.tensor(expert_starts, dtype=torch.int32, device=x.device)
        return _per_expert_quant_A_triton(x, es_tensor)

    # ── Python fallback (original implementation) ──────────────────────────
    E = len(expert_starts) - 1
    x_fp8 = torch.empty_like(x, dtype=_TORCH_FP8)
    scale_invs = torch.ones(E, dtype=torch.float32, device=x.device)

    for e in range(E):
        s, t = expert_starts[e], expert_starts[e + 1]
        if s >= t:
            continue
        chunk = x[s:t].float()
        amax = chunk.abs().max().clamp(min=1e-12)
        scale = _E4M3_MAX / amax
        scale_invs[e] = amax / _E4M3_MAX
        x_fp8[s:t] = chunk.mul_(scale).clamp_(-_E4M3_MAX, _E4M3_MAX).to(_TORCH_FP8)

    return x_fp8, scale_invs


def _per_tensor_quant_A(
    x: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation with a single tensor-wise scale (faster, lower accuracy)."""
    amax = x.float().abs().max().clamp(min=1e-12)
    scale = _E4M3_MAX / amax
    scale_inv = (amax / _E4M3_MAX).float()
    x_fp8 = x.float().mul_(scale).clamp_(-_E4M3_MAX, _E4M3_MAX).to(_TORCH_FP8)
    scale_invs = torch.full((num_experts,), scale_inv, dtype=torch.float32, device=x.device)
    return x_fp8, scale_invs


def _per_tensor_quant_W(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantise weight tensor with a single per-tensor scale.

    Args:
        w: [E, K, N]  BF16

    Returns:
        w_fp8:     [E, K, N]  float8_e4m3fnuz
        scale_inv: scalar float32
    """
    amax = w.float().abs().max().clamp(min=1e-12)
    scale = _E4M3_MAX / amax
    scale_inv = (amax / _E4M3_MAX).float()
    w_fp8 = w.float().mul_(scale).clamp_(-_E4M3_MAX, _E4M3_MAX).to(_TORCH_FP8)
    return w_fp8, scale_inv


# ---------------------------------------------------------------------------
# Tile-mapping (CPU, O(total_tiles))
# ---------------------------------------------------------------------------

def _build_tile_map(
    tokens_per_expert: List[int],
    N: int,
    BLOCK_M: int,
    BLOCK_N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Pre-compute which (expert, pid_m, pid_n) each tile ID belongs to.

    Returns:
        tile_expert: [T] int32
        tile_pid_m:  [T] int32
        tile_pid_n:  [T] int32
        expert_starts: [E+1] Python ints  (token offsets)
    """
    tile_expert: List[int] = []
    tile_pid_m:  List[int] = []
    tile_pid_n:  List[int] = []

    num_n_tiles = math.ceil(N / BLOCK_N)
    expert_starts: List[int] = [0]
    acc = 0
    for m in tokens_per_expert:
        acc += m
        expert_starts.append(acc)

    for e, m in enumerate(tokens_per_expert):
        num_m_tiles = math.ceil(max(m, 1) / BLOCK_M) if m > 0 else 0
        for pm in range(num_m_tiles):
            for pn in range(num_n_tiles):
                tile_expert.append(e)
                tile_pid_m.append(pm)
                tile_pid_n.append(pn)

    dev = torch.device("cuda")
    t_expert = torch.tensor(tile_expert, dtype=torch.int32, device=dev)
    t_pid_m  = torch.tensor(tile_pid_m,  dtype=torch.int32, device=dev)
    t_pid_n  = torch.tensor(tile_pid_n,  dtype=torch.int32, device=dev)
    return t_expert, t_pid_m, t_pid_n, expert_starts


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _fp8_grouped_gemm_kernel(
        # ── data ────────────────────────────────────────────────────────────
        A_ptr, B_ptr, C_ptr,
        # ── dequantisation scales (per-expert for A; single scalar for B) ──
        A_si_ptr,  # [E]  float32  scale_inv_A per expert
        B_si,      # scalar float32  scale_inv_B (passed as constexpr float)
        # ── tile mapping ────────────────────────────────────────────────────
        tile_expert_ptr,  # [T] int32
        tile_pid_m_ptr,   # [T] int32
        tile_pid_n_ptr,   # [T] int32
        # ── expert token starts ─────────────────────────────────────────────
        expert_starts_ptr,  # [E+1] int32
        # ── dimensions ──────────────────────────────────────────────────────
        N: tl.constexpr,
        K: tl.constexpr,
        # ── strides ─────────────────────────────────────────────────────────
        stride_am, stride_ak,
        stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn,
        # ── block sizes ─────────────────────────────────────────────────────
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        One Triton program computes one [BLOCK_M, BLOCK_N] output tile.
        Grid = (total_tiles,).
        """
        tid = tl.program_id(0)

        # ── look up this tile's (expert, pid_m, pid_n) ───────────────────
        eid   = tl.load(tile_expert_ptr + tid)
        pid_m = tl.load(tile_pid_m_ptr  + tid)
        pid_n = tl.load(tile_pid_n_ptr  + tid)

        # ── expert token range ────────────────────────────────────────────
        e_start = tl.load(expert_starts_ptr + eid)
        e_end   = tl.load(expert_starts_ptr + eid + 1)
        e_M     = e_end - e_start

        m0 = pid_m * BLOCK_M
        n0 = pid_n * BLOCK_N
        gm0 = e_start + m0  # global row start in A / C

        # ── accumulator ──────────────────────────────────────────────────
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        A_base = A_ptr + gm0 * stride_am
        B_base = B_ptr + eid * stride_be + n0 * stride_bn

        # ── K-loop ────────────────────────────────────────────────────────
        for k0 in range(0, K, BLOCK_K):
            # A tile [BLOCK_M, BLOCK_K]
            m_offs = tl.arange(0, BLOCK_M)[:, None]
            k_offs = k0 + tl.arange(0, BLOCK_K)[None, :]
            a_mask = (m_offs < e_M - m0) & (k_offs < K)
            a_ptrs = A_base + m_offs * stride_am + k_offs * stride_ak
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # B tile [BLOCK_K, BLOCK_N]
            kb_offs = k0 + tl.arange(0, BLOCK_K)[:, None]
            nb_offs = tl.arange(0, BLOCK_N)[None, :]
            b_mask = (kb_offs < K) & (n0 + nb_offs < N)
            b_ptrs = B_base + kb_offs * stride_bk + nb_offs * stride_bn
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # Cast to FP8 (OCP E4M3FN, native on gfx950) and accumulate in FP32
            acc = tl.dot(
                a_tile.to(tl.float8e4nv),
                b_tile.to(tl.float8e4nv),
                acc,
                out_dtype=tl.float32,
            )

        # ── dequantise ───────────────────────────────────────────────────
        a_si = tl.load(A_si_ptr + eid)
        acc = acc * (a_si * B_si)

        # ── write C ──────────────────────────────────────────────────────
        m_offs = tl.arange(0, BLOCK_M)[:, None]
        n_offs = n0 + tl.arange(0, BLOCK_N)[None, :]
        c_mask = ((gm0 + m_offs) < e_end) & (n_offs < N)
        c_ptrs = C_ptr + (gm0 + m_offs) * stride_cm + n_offs * stride_cn
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# ---------------------------------------------------------------------------
# Python entry point (forward pass only, BF16 I/O)
# ---------------------------------------------------------------------------

def fp8_grouped_gemm_forward(
    A: torch.Tensor,
    B: torch.Tensor,
    tokens_per_expert: List[int],
    *,
    # Best config from 100-round grid search (R093):
    # BM=128 BN=256 BK=128 W=16 S=2 gives 177 TFLOPS / 0.47× vs BF16.
    # NOTE: Speedup target (1.4×) requires FP8 weight storage — see final_summary.md.
    BLOCK_M: int = 128,
    BLOCK_N: int = 256,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    """FP8 grouped GEMM forward.

    Args:
        A:                 [total_M, K]  BF16
        B:                 [E, K, N]    BF16
        tokens_per_expert: [E]          Python int list

    Returns:
        C: [total_M, N]  BF16
    """
    assert HAVE_TRITON, "Triton not available; cannot use FP8 grouped GEMM"
    assert A.dtype == torch.bfloat16, f"A must be BF16, got {A.dtype}"
    assert B.dtype == torch.bfloat16, f"B must be BF16, got {B.dtype}"
    assert A.is_contiguous()
    assert B.is_contiguous()

    # Quantize weight once and delegate the rest (tile map + A quant + kernel launch).
    B_fp8, B_si_scalar = _per_tensor_quant_W(B)
    return fp8_grouped_gemm_forward_prequantized_w(
        A,
        B_fp8,
        B_si_scalar,
        tokens_per_expert,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )


def fp8_grouped_gemm_forward_prequantized_w(
    A: torch.Tensor,
    B_fp8: torch.Tensor,
    B_si_scalar: torch.Tensor,
    tokens_per_expert: List[int],
    *,
    BLOCK_M: int = 128,
    BLOCK_N: int = 256,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    """FP8 grouped GEMM forward with pre-quantized weight.

    Args:
        A:                 [total_M, K]  BF16
        B_fp8:             [E, K, N]     FP8 (torch.float8_e4m3fn)
        B_si_scalar:       scalar float32 tensor (dequant scale_inv for B)
        tokens_per_expert: [E]           Python int list
    """
    assert HAVE_TRITON, "Triton not available; cannot use FP8 grouped GEMM"
    assert A.dtype == torch.bfloat16, f"A must be BF16, got {A.dtype}"
    assert B_fp8.dtype == _TORCH_FP8, f"B_fp8 must be {_TORCH_FP8}, got {B_fp8.dtype}"
    assert A.is_contiguous()
    assert B_fp8.is_contiguous()
    assert B_si_scalar.numel() == 1

    E, K, N = B_fp8.shape
    total_M = A.shape[0]

    # Build tile map on CPU and quantize A.
    t_expert, t_pid_m, t_pid_n, expert_starts = _build_tile_map(
        tokens_per_expert, N, BLOCK_M, BLOCK_N
    )
    total_tiles = t_expert.shape[0]
    if total_tiles == 0:
        return torch.zeros(total_M, N, dtype=torch.bfloat16, device=A.device)

    # expert_starts as int32 tensor (reused by both quant and kernel)
    es_tensor = torch.tensor(expert_starts, dtype=torch.int32, device=A.device)

    # Optional fast mode: tensor-wise A scale to reduce quant/cast overhead.
    # Enable with PRIMUS_TRITON_FP8_A_PER_TENSOR=1 for performance experiments.
    if os.getenv("PRIMUS_TRITON_FP8_A_PER_TENSOR", "0") == "1":
        A_fp8, A_si = _per_tensor_quant_A(A, len(tokens_per_expert))
    else:
        use_python = os.getenv("PRIMUS_QUANT_A_USE_PYTHON", "0") == "1"
        if HAVE_TRITON and A.is_cuda and A.is_contiguous() and not use_python:
            # Fast path: fused Triton kernel, reuse es_tensor already on GPU
            A_fp8, A_si = _per_expert_quant_A_triton(A, es_tensor)
        else:
            A_fp8, A_si = _per_expert_quant_A(A, expert_starts)

    C = torch.zeros(total_M, N, dtype=torch.bfloat16, device=A.device)

    _fp8_grouped_gemm_kernel[(total_tiles,)](
        A_fp8, B_fp8, C,
        A_si,
        B_si_scalar.item(),
        t_expert, t_pid_m, t_pid_n,
        es_tensor,
        N=N, K=K,
        stride_am=A_fp8.stride(0), stride_ak=A_fp8.stride(1),
        stride_be=B_fp8.stride(0), stride_bk=B_fp8.stride(1), stride_bn=B_fp8.stride(2),
        stride_cm=C.stride(0), stride_cn=C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=16,
        num_stages=2,
    )
    return C


# ---------------------------------------------------------------------------
# autograd.Function  (FP8 forward, BF16 backward)
# ---------------------------------------------------------------------------

class Fp8GroupedGemmFunction(torch.autograd.Function):
    """
    Forward:  C[e]  = A[e] @ B[e]           via Triton FP8 kernel
    Backward: dA[e] = grad[e] @ B[e]^T      via BF16 grouped GEMM
              dB[e] = A[e]^T @ grad[e]      via BF16 grouped GEMM
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        tpe_list: List[int] = tokens_per_expert.tolist()
        C = fp8_grouped_gemm_forward(A, B, tpe_list)
        ctx.save_for_backward(A, B)
        ctx.tpe_list = tpe_list
        return C

    @staticmethod
    def backward(
        ctx,
        grad_C: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        A, B = ctx.saved_tensors
        tpe_list = ctx.tpe_list

        dA, dB = _parallel_bfloat16_bwd(grad_C, A, B, tpe_list)
        return dA, dB, None


class Fp8GroupedGemmPrequantWFunction(torch.autograd.Function):
    """Autograd path with cached/pre-quantized weight for forward."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        B_fp8: torch.Tensor,
        B_si_scalar: torch.Tensor,
    ) -> torch.Tensor:
        tpe_list: List[int] = tokens_per_expert.tolist()
        C = fp8_grouped_gemm_forward_prequantized_w(A, B_fp8, B_si_scalar, tpe_list)
        ctx.save_for_backward(A, B)
        ctx.tpe_list = tpe_list
        return C

    @staticmethod
    def backward(
        ctx,
        grad_C: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None, None]:
        A, B = ctx.saved_tensors
        tpe_list = ctx.tpe_list

        dA, dB = _parallel_bfloat16_bwd(grad_C, A, B, tpe_list)
        return dA, dB, None, None, None


def _bfloat16_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    tpe_list: List[int],
) -> torch.Tensor:
    """BF16 grouped GEMM: C[e] = A[tokens_e] @ B[e].

    A: [total_M, K]
    B: [E, K, N]  (B[e] has shape [K, N])
    """
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    if HAVE_GG:
        # grouped_gemm backend requires batch_sizes on CPU.
        tpe_cpu = torch.tensor(tpe_list, dtype=torch.int64, device="cpu")
        return _gg_ops.gmm(A, B, tpe_cpu, trans_b=False)

    # Fallback: torch.bmm over experts
    return _torch_bmm_grouped(A, B, tpe_list, trans_b=False)


def _bfloat16_grouped_gemm_weight_grad(
    A: torch.Tensor,
    grad_C: torch.Tensor,
    tpe_list: List[int],
    B_shape: torch.Size,
) -> torch.Tensor:
    """Accumulate weight gradients per expert.

    dB[e] = A[s:t]^T @ grad_C[s:t]   →   [K, N]
    Returns dB: [E, K, N]
    """
    E, K, N = B_shape
    if not A.is_contiguous():
        A = A.contiguous()
    if not grad_C.is_contiguous():
        grad_C = grad_C.contiguous()
    dB = torch.zeros(B_shape, dtype=A.dtype, device=A.device)
    offset = 0
    for e, m in enumerate(tpe_list):
        if m > 0:
            a_e    = A[offset : offset + m]       # [m, K]
            gc_e   = grad_C[offset : offset + m]  # [m, N]
            dB[e]  = a_e.t().mm(gc_e)             # [K, N]
        offset += m
    return dB


def _use_dual_stream_expert_bwd() -> bool:
    """Overlap activation-grad (dA) and weight-grad (dB) grouped-GEMM on two streams.

    Disable with ``PRIMUS_MOE_EXPERT_GEMM_BWD_DUAL_STREAM=0`` for debugging.
    """
    return os.getenv("PRIMUS_MOE_EXPERT_GEMM_BWD_DUAL_STREAM", "1") == "1"


def _parallel_bfloat16_bwd(
    grad_C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    tpe_list: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """dA = grad_C @ B^T, dB from A^T @ grad_C per expert; dual-stream when on CUDA."""
    B_t = B.transpose(1, 2).contiguous()
    if (
        not _use_dual_stream_expert_bwd()
        or not grad_C.is_cuda
        or not torch.cuda.is_available()
    ):
        dA = _bfloat16_grouped_gemm(grad_C, B_t, tpe_list)
        dB = _bfloat16_grouped_gemm_weight_grad(A, grad_C, tpe_list, B.shape)
        return dA, dB

    dev = grad_C.device
    stream_da = torch.cuda.Stream(device=dev)
    stream_db = torch.cuda.Stream(device=dev)
    current = torch.cuda.current_stream(device=dev)
    dA_holder: List[Optional[torch.Tensor]] = [None]
    dB_holder: List[Optional[torch.Tensor]] = [None]

    with torch.cuda.stream(stream_da):
        dA_holder[0] = _bfloat16_grouped_gemm(grad_C, B_t, tpe_list)
    with torch.cuda.stream(stream_db):
        dB_holder[0] = _bfloat16_grouped_gemm_weight_grad(A, grad_C, tpe_list, B.shape)

    current.wait_stream(stream_da)
    current.wait_stream(stream_db)
    assert dA_holder[0] is not None and dB_holder[0] is not None
    return dA_holder[0], dB_holder[0]


def _torch_bmm_grouped(
    A: torch.Tensor,
    B: torch.Tensor,
    tpe_list: List[int],
    trans_b: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch fallback grouped GEMM."""
    out_parts = []
    offset = 0
    for e, m in enumerate(tpe_list):
        if m > 0:
            a_e = A[offset : offset + m]          # [m, K]
            b_e = B[e].t() if trans_b else B[e]   # [K, N]
            out_parts.append(a_e.mm(b_e))
        elif m == 0:
            out_parts.append(
                torch.zeros(0, B.shape[-1], dtype=A.dtype, device=A.device)
            )
        offset += m
    return torch.cat(out_parts, dim=0) if out_parts else torch.zeros(
        0, B.shape[-1], dtype=A.dtype, device=A.device
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fp8_grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    trans_b: bool = False,
    prequantized_w: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """FP8 expert grouped GEMM with BF16 backward.

    Args:
        A:                 [total_M, K]  BF16
        B:                 [E, K, N]    BF16  (or [E, N, K] if trans_b=True)
        tokens_per_expert: [E]          int32/int64  (tokens routed per expert)
        trans_b:           if True B is [E, N, K] and is transposed to [E, K, N]
        prequantized_w:    Optional cached tuple (B_fp8, B_si_scalar) for forward.
                           Backward still uses BF16 B for correct gradients.

    Returns:
        C: [total_M, N]  BF16
    """
    if not HAVE_TRITON:
        raise RuntimeError(
            "Triton is required for fp8_grouped_gemm. "
            "Install triton (ROCm build) or disable use_triton_fp8_expert_gemm."
        )

    if trans_b:
        B = B.transpose(1, 2).contiguous()

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    if prequantized_w is not None:
        B_fp8, B_si_scalar = prequantized_w
        return Fp8GroupedGemmPrequantWFunction.apply(
            A, B, tokens_per_expert, B_fp8, B_si_scalar
        )

    return Fp8GroupedGemmFunction.apply(A, B, tokens_per_expert)

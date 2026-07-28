###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-7 P45 — multi-tensor BF16 add Triton kernel (prototype).

Replaces the ~743 separate ``vec_elem<add_bf16>`` launches that
dominate the P40 trace's optimizer-step residual (170.99 ms / iter,
32.7 % of step) with a single Triton kernel that processes a
multi-tensor batch in one launch.

This is a **prototype** designed to demonstrate the perf headroom
of multi-tensor fusion against ``torch._foreach_add_`` (the
PyTorch reference path) and the upstream Apex / TE
``multi_tensor_apply`` (the V4-Flash production path).  Production
integration with the Apex / TE optimizer call sites is plan-8
scope — replacing those call sites bit-exactly requires
coordinating with the master-param remainder accumulation logic
that Apex / TE owns.

Gating: import-time only; no env knob (this module is currently
a microbench / unit-test target only).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _multi_tensor_add_per_tensor_kernel(
    OUT_PTR,
    A_PTR,
    B_PTR,
    N,
    SCALE,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-tensor variant.  Each grid program handles ``BLOCK_SIZE``
    elements of one tensor.

    Caller is responsible for issuing the per-tensor launch grid
    and serialising the loops, but the launches are CONCURRENT on
    the GPU (Triton stream stacking).
    """

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A_PTR + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_PTR + offs, mask=mask, other=0.0).to(tl.float32)
    out = a + SCALE * b
    tl.store(OUT_PTR + offs, out, mask=mask)


def multi_tensor_add_triton_per_tensor(
    out_list: List[torch.Tensor],
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    scale: float = 1.0,
    block_size: int = 8192,
) -> None:
    """Compute ``out_i = a_i + scale * b_i`` for each ``i`` in place
    (per-tensor variant).

    All tensors must be contiguous + on CUDA / HIP, same dtype, same
    shape per ``(out_i, a_i, b_i)`` triple.
    """
    assert len(out_list) == len(a_list) == len(b_list)
    for out, a, b in zip(out_list, a_list, b_list):
        assert out.shape == a.shape == b.shape, "shape mismatch"
        n = out.numel()
        grid = (triton.cdiv(n, block_size),)
        _multi_tensor_add_per_tensor_kernel[grid](
            out,
            a,
            b,
            n,
            scale,
            BLOCK_SIZE=block_size,
        )


@triton.jit
def _multi_tensor_add_packed_kernel(
    OUT_PTR_BUF,  # int64 [N_TENSORS] pointer array
    A_PTR_BUF,  # int64 [N_TENSORS]
    B_PTR_BUF,  # int64 [N_TENSORS]
    SIZE_BUF,  # int32 [N_TENSORS] -- element counts
    PID_TO_TID_BUF,  # int32 [N_PROGRAMS] -- program-id -> tensor-id
    PID_TO_LOCAL_BUF,  # int32 [N_PROGRAMS] -- program-id -> local block index
    SCALE,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-kernel multi-tensor add (CPU-side dispatch table variant).

    One program per ``BLOCK_SIZE`` chunk of exactly one tensor.
    The CPU side builds a sorted dispatch table mapping each
    ``program_id`` to ``(tensor_idx, local_block_idx)`` so a single
    grid launch absorbs all N_TENSORS tensors without per-element
    tensor lookups.

    This avoids the cross-tensor-boundary aliasing that a naive
    "concatenated range" partition would create.
    """

    pid = tl.program_id(0)
    tensor_idx = tl.load(PID_TO_TID_BUF + pid)
    local_block = tl.load(PID_TO_LOCAL_BUF + pid)

    offs = local_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    size_i = tl.load(SIZE_BUF + tensor_idx)
    mask = offs < size_i

    out_ptr = tl.load(OUT_PTR_BUF + tensor_idx).to(tl.pointer_type(tl.bfloat16))
    a_ptr = tl.load(A_PTR_BUF + tensor_idx).to(tl.pointer_type(tl.bfloat16))
    b_ptr = tl.load(B_PTR_BUF + tensor_idx).to(tl.pointer_type(tl.bfloat16))

    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = a + SCALE * b
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)


def _build_multi_tensor_dispatch_table(
    out_list: List[torch.Tensor],
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the GPU-side dispatch tables for the packed kernel.

    Returns
    -------
    out_ptrs, a_ptrs, b_ptrs : int64 [N_TENSORS]
    sizes : int32 [N_TENSORS]
    pid_to_tid : int32 [N_PROGRAMS]  -- block-id -> tensor-id
    pid_to_local : int32 [N_PROGRAMS] -- block-id -> local-block-index
        within the tensor it belongs to.
    """
    len(out_list)
    device = out_list[0].device
    out_ptrs = torch.tensor([t.data_ptr() for t in out_list], dtype=torch.int64, device=device)
    a_ptrs = torch.tensor([t.data_ptr() for t in a_list], dtype=torch.int64, device=device)
    b_ptrs = torch.tensor([t.data_ptr() for t in b_list], dtype=torch.int64, device=device)
    sizes = torch.tensor([t.numel() for t in out_list], dtype=torch.int32, device=device)
    # Build pid -> (tid, local) on the host then copy.
    pid_to_tid: List[int] = []
    pid_to_local: List[int] = []
    for t_idx, t in enumerate(out_list):
        n_chunks = (t.numel() + block_size - 1) // block_size
        for local in range(n_chunks):
            pid_to_tid.append(t_idx)
            pid_to_local.append(local)
    pid_to_tid_t = torch.tensor(pid_to_tid, dtype=torch.int32, device=device)
    pid_to_local_t = torch.tensor(pid_to_local, dtype=torch.int32, device=device)
    return out_ptrs, a_ptrs, b_ptrs, sizes, pid_to_tid_t, pid_to_local_t


def multi_tensor_add_triton_packed(
    out_list: List[torch.Tensor],
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    scale: float = 1.0,
    block_size: int = 8192,
) -> None:
    """Compute ``out_i = a_i + scale * b_i`` for each ``i`` in place
    (packed multi-tensor variant; ONE kernel launch total).

    All tensors must be bf16, contiguous, on the same CUDA / HIP
    device, with matching ``(out_i, a_i, b_i)`` shapes.
    """
    assert len(out_list) == len(a_list) == len(b_list) > 0
    for out, a, b in zip(out_list, a_list, b_list):
        assert out.dtype == torch.bfloat16
        assert a.dtype == torch.bfloat16
        assert b.dtype == torch.bfloat16
        assert out.is_contiguous() and a.is_contiguous() and b.is_contiguous()
        assert out.shape == a.shape == b.shape

    out_ptrs, a_ptrs, b_ptrs, sizes, pid_to_tid, pid_to_local = _build_multi_tensor_dispatch_table(
        out_list, a_list, b_list, block_size
    )

    grid = (pid_to_tid.numel(),)
    _multi_tensor_add_packed_kernel[grid](
        out_ptrs,
        a_ptrs,
        b_ptrs,
        sizes,
        pid_to_tid,
        pid_to_local,
        scale,
        BLOCK_SIZE=block_size,
    )


__all__ = [
    "multi_tensor_add_triton_per_tensor",
    "multi_tensor_add_triton_packed",
]

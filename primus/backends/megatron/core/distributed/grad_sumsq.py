###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bandwidth-bound sum of squares over a flat gradient bucket (fp32 accumulation).

``bucket.grad_data.norm()`` (ATen ``reduce_kernel<512,1>`` with ``NormTwoOps``) reads a
1.06 GB bf16 bucket in ~161 us on MI455X (6.6 TB/s, 36% of HBM). This two-stage reduction
(grid-stride partial sums per program, then a tiny ``sum`` of the partials) is
deterministic and streams at close to the HBM rate. Used only for the DDP NaN/Inf grad
check, where the value itself is not consumed.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sumsq_partial_kernel(X_PTR, N, OUT_PTR, BLOCK: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    offs = tl.arange(0, BLOCK)
    for start in range(pid * BLOCK, N, NPROG * BLOCK):
        idx = start + offs
        x = tl.load(X_PTR + idx, mask=idx < N, other=0.0).to(tl.float32)
        acc += x * x
    tl.store(OUT_PTR + pid, tl.sum(acc, axis=0))


def grad_sumsq(x: torch.Tensor, nprog: int = 4096, block: int = 2048) -> torch.Tensor:
    """fp32 sum of squares of a contiguous 1-D CUDA tensor (0-dim fp32 result, no sync)."""
    x = x.reshape(-1)
    n = x.numel()
    if n < (1 << 31):
        nprog = max(1, min(nprog, triton.cdiv(n, block)))
        part = torch.empty((nprog,), dtype=torch.float32, device=x.device)
        _sumsq_partial_kernel[(nprog,)](x, n, part, BLOCK=block, NPROG=nprog)
        return part.sum()
    return x.float().pow(2).sum()


# ---- membound (PRIMUS_MB_GNORM=1): every bucket range of the step in two launches ----
# Each job's ABSOLUTE address comes from an int64 table (no base + offset arithmetic across allocations: the AMD
# backend's buffer loads assume a kernel pointer argument addresses < 2 GB and offsets >= 0, which a cross-
# allocation offset breaks -- measured: wrong sums).
@triton.jit
def _sumsq_multi_partial_kernel(PTRS, NS, POFF, OUT_PTR, J: tl.constexpr, JX: tl.constexpr, CHUNK: tl.constexpr,
                                BLOCK: tl.constexpr):
    """grid (sum_j cdiv(n_j, CHUNK),): program `pid` of job j (POFF[j] <= pid < POFF[j+1]) reduces elements
    [c*CHUNK, (c+1)*CHUNK) of job j to one fp32 value in OUT_PTR[pid]. Jobs j < J - JX are bf16 gradient ranges
    (sum of squares; address and length multiples of 16 elements); the last JX jobs are fp32 sum-of-squares partial
    buffers (plain sum)."""
    pid = tl.program_id(0)
    j = 0
    for k in tl.static_range(1, J):
        j += (pid >= tl.load(POFF + k)).to(tl.int32)
    c = pid - tl.load(POFF + j)
    n0 = tl.load(NS + j)
    a0 = tl.load(PTRS + j)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    r = tl.arange(0, BLOCK)
    base = c.to(tl.int64) * CHUNK
    if j < J - JX:
        n = tl.multiple_of(n0, 16)
        xb_ptr = tl.multiple_of(a0.to(tl.pointer_type(tl.bfloat16)), 16)
        for kk in range(0, CHUNK, BLOCK):
            idx = base + kk + r
            x = tl.load(xb_ptr + idx, mask=idx < n, other=0.0).to(tl.float32)
            acc += x * x
    else:
        xf_ptr = a0.to(tl.pointer_type(tl.float32))
        for kk in range(0, CHUNK, BLOCK):
            idx = base + kk + r
            acc += tl.load(xf_ptr + idx, mask=idx < n0, other=0.0)
    tl.store(OUT_PTR + pid, tl.sum(acc, axis=0))


@triton.jit
def _sumsq_multi_final_kernel(PART, NPART, OUT, CH: tl.constexpr):
    """one program: OUT[0] = sum(PART[:NPART]) (fixed order: per-lane strided sums, then one tree)."""
    r = tl.arange(0, CH)
    s = tl.zeros((CH,), dtype=tl.float32)
    for c in range(0, NPART, CH):
        s += tl.load(PART + c + r, mask=c + r < NPART, other=0.0)
    tl.store(OUT, tl.sum(s, axis=0))


class MultiSumsq:
    """Static plan for a fixed list of 1-D bf16 gradient ranges and fp32 sum-of-squares partial buffers
    (``extras``, read in place each call); built eagerly (H2D copy of the address table), reusable inside CUDA-graph
    capture. __call__ returns a 0-dim fp32 tensor = sum of squares of the ranges + sum of the extras (NaN/Inf
    propagate). Not bit-identical to the per-range grad_sumsq + torch sums (different reduction order). ``ok`` is
    False when a range is not bf16 or not 16-element aligned / sized (the caller then keeps the upstream loop)."""

    CHUNK = 1 << 17

    def __init__(self, ranges, extras, block=2048):
        dev = ranges[0].device
        ptrs, ns, poff = [], [], [0]
        self.ok = all(x.dtype == torch.bfloat16 for x in ranges) and all(e.dtype == torch.float32 for e in extras)
        for x in list(ranges) + list(extras):
            if not x.is_contiguous() or (x.dtype == torch.bfloat16 and (x.data_ptr() % 32 or x.numel() % 16)):
                self.ok = False
            if x.dtype == torch.float32 and x.numel() % 16:
                # the kernel's tl.multiple_of(n0, 16) hint (bf16 branch) also reaches the fp32 branch's mask: an fp32
                # partial buffer whose length is not a multiple of 16 would be over-read -> upstream loop instead
                self.ok = False
            ptrs.append(x.data_ptr())
            ns.append(x.numel())
            poff.append(poff[-1] + triton.cdiv(x.numel(), self.CHUNK))
        self.keep = list(ranges) + list(extras)  # the table holds raw addresses: keep the tensors alive
        self.J, self.JX = len(ptrs), len(extras)
        self.ptrs = torch.tensor(ptrs, dtype=torch.int64, device=dev)
        self.ns = torch.tensor(ns, dtype=torch.int64, device=dev)
        self.poff = torch.tensor(poff, dtype=torch.int32, device=dev)
        self.total = poff[-1]
        self.part = torch.empty((self.total,), dtype=torch.float32, device=dev)
        self.out = torch.empty((1,), dtype=torch.float32, device=dev)
        self.block = block

    def __call__(self):
        _sumsq_multi_partial_kernel[(self.total,)](self.ptrs, self.ns, self.poff, self.part, J=self.J, JX=self.JX,
                                                   CHUNK=self.CHUNK, BLOCK=self.block, num_warps=8)
        _sumsq_multi_final_kernel[(1,)](self.part, self.total, self.out, CH=1024, num_warps=4)
        return self.out[0]

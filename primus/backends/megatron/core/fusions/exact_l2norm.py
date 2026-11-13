###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformer Engine's ROCm ``multi_tensor_l2norm`` (``custom_multi_tensor_l2norm_kernel`` +
``custom_multi_tensor_l2norm_reduce``) for BF16 gradients, bitwise equal and faster.

TE's kernel (read from its gfx1250 ISA; the image ships no source) is apex's L2NormFunctor
with ILP 8, one launch over every (tensor, 64K chunk) block, one partial per block:

* aligned chunks (``(n_rem | chunk) % 8 == 0`` and a 16-B aligned start): thread ``t`` loads
  8 BF16 at ``t*8 + k*BS*8`` (k = 0, 1, ...) and does ``a[j] = fma(x_j, x_j, a[j])``;
  otherwise element ``i0 + t + j*BS`` goes to ``a[j]`` for ``i0 = 0, BS*8, ...``;
* ``v = 0 + a[0] + ... + a[7]``, apex ``reduce_block_into_lanes`` (smem tree down to 64,
  ``x[t] + x[t+32]``, shfl_down 16..1), ``out[block] = v`` (non-finite sets the noop flag);
* one block of the reduce kernel: per-thread sequential sum over ``p[t + k*RBS]``, the same
  block reduction, ``sqrtf``.

TE's loop waits for each 16-B load before issuing the next (11.5 TB/s on 21.8 GB). Here the
per-thread loads of a chunk are issued together before the FMAs, in the same per-accumulator
order, so every partial and the result are bit-identical.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("PRIMUS_EXACT_L2NORM", "1") != "0"
CHUNK = 2048 * 32

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __HIP_DEVICE_COMPILE__
#define L2N_GLOBAL __attribute__((address_space(1)))
#else
#define L2N_GLOBAL
#endif
typedef unsigned int l2n_u4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ void acc8(float (&a)[8], const l2n_u4 q) {
  float e;
  e = bf_lo(q.x); a[0] = __builtin_fmaf(e, e, a[0]);
  e = bf_hi(q.x); a[1] = __builtin_fmaf(e, e, a[1]);
  e = bf_lo(q.y); a[2] = __builtin_fmaf(e, e, a[2]);
  e = bf_hi(q.y); a[3] = __builtin_fmaf(e, e, a[3]);
  e = bf_lo(q.z); a[4] = __builtin_fmaf(e, e, a[4]);
  e = bf_hi(q.z); a[5] = __builtin_fmaf(e, e, a[5]);
  e = bf_lo(q.w); a[6] = __builtin_fmaf(e, e, a[6]);
  e = bf_hi(q.w); a[7] = __builtin_fmaf(e, e, a[7]);
}

// apex reduce_block_into_lanes (lanes = 1) for a 1-D block of BS threads
template <int BS>
__device__ __forceinline__ float reduce_block_into_lanes(float* x, float val) {
  const int tid = threadIdx.x;
  if (BS >= 64) {
    x[tid] = val;
    __syncthreads();
  }
#pragma unroll
  for (int i = (BS >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }
  float final = 0.f;
  if (tid < 32) {
    if (BS >= 64)
      final = x[tid] + x[tid + 32];
    else
      final = val;
#pragma unroll
    for (int i = 16; i >= 1; i >>= 1) final = final + __shfl_down(final, i, 32);
  }
  return final;
}

// Full, 16-B aligned 64K chunks: one wave per chunk. TE's 512-thread workgroup is emulated:
// lane L holds virtual threads t = L + 32*w (w = 0..15); TE's smem stages (x[t] += x[t+256],
// +128, +64) and x[t] + x[t+32] combine values of one lane, in TE's order; the shfl_down tree
// 16..1 is the real cross-lane step. Per virtual thread the 8 accumulators see TE's element
// order (16-B vector k at t*8 + k*4096). No barriers: the 16 per-lane values wait in the
// lane's own LDS slots. Rolled virtual-thread loop + VGPR cap 128 (120 used, no spills); the
// unrolled form hoisted loads across virtual threads (191 VGPRs or spills).
__global__ __launch_bounds__(256) __attribute__((amdgpu_num_vgpr(128))) void l2n_full(
    const int64_t* __restrict__ bptr, const int* __restrict__ bidx, int nfull, float* __restrict__ out, int* noop,
    const int* __restrict__ frows, const unsigned char* __restrict__ rmask) {
  const int i = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (i >= nfull) return;
  // Chunks of the persistent embedding gradient whose rows are all flagged zero hold exactly
  // +0: TE's partial for them is +0 (fma(0, 0, 0) and sums of +0), written without loading.
  if (rmask != nullptr) {
    const int r0 = frows[2 * i], r1 = frows[2 * i + 1];
    if (r0 >= 0) {
      int any = 0;
      for (int r = r0; r <= r1; ++r) any |= rmask[r];
      if (!any) {
        if (lane == 0) out[bidx[i]] = 0.f;
        return;
      }
    }
  }
  // the chunk pointer is wave-uniform: keep it in SGPRs (global_load saddr + 32-bit offset)
  const uint64_t xp = (uint64_t)bptr[i];
  const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t)xp);
  const uint32_t hi = __builtin_amdgcn_readfirstlane((uint32_t)(xp >> 32));
  const uint16_t* x = reinterpret_cast<const uint16_t*>(((uint64_t)hi << 32) | lo);
  __shared__ float s_val[8][16][32];  // [wave][virtual thread][lane]: each lane its own slots
  float (*sv)[32] = s_val[threadIdx.x >> 5];
#pragma unroll 1
  for (int w = 0; w < 16; ++w) {
    const L2N_GLOBAL l2n_u4* gx = (const L2N_GLOBAL l2n_u4*)(x + (lane + 32 * w) * 8);
    l2n_u4 q[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) q[k] = gx[k * 512];
    float a[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc8(a, q[k]);
    float v = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) v = v + a[j];
    sv[w][lane] = v;
  }
  float val[16];
#pragma unroll
  for (int w = 0; w < 16; ++w) val[w] = sv[w][lane];
#pragma unroll
  for (int s = 8; s >= 2; s >>= 1) {
#pragma unroll
    for (int w = 0; w < s; ++w) val[w] = val[w] + val[w + s];
  }
  float final = val[0] + val[1];
#pragma unroll
  for (int s = 16; s >= 1; s >>= 1) final = final + __shfl_down(final, s, 32);
  if (lane == 0) {
    if (!isfinite(final)) *noop = 1;
    out[bidx[i]] = final;
  }
}

// Every other chunk (a tensor's last, partial chunk, or a misaligned tensor): TE's workgroup
// of 512 threads as is.
__global__ __launch_bounds__(512) void l2n_rest(const int64_t* __restrict__ bptr, const int* __restrict__ blim,
                                                const int* __restrict__ bidx, float* __restrict__ out, int* noop) {
  __shared__ float s_vals[512];
  const int i = blockIdx.x;
  const uint16_t* x = reinterpret_cast<const uint16_t*>(bptr[i]);
  const int nrem = blim[2 * i];
  const int lim = blim[2 * i + 1];
  const int chunk = 65536;
  float a[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) a[j] = 0.f;
  const bool aligned = (((nrem | chunk) & 7) == 0) && ((reinterpret_cast<uintptr_t>(x) & 15) == 0);
  if (aligned) {
    for (int idx = threadIdx.x * 8; idx < lim; idx += 512 * 8) acc8(a, *reinterpret_cast<const l2n_u4*>(x + idx));
  } else {
    for (int i0 = 0; i0 < lim; i0 += 512 * 8) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int k = i0 + threadIdx.x + j * 512;
        if (k < lim) {
          const float e = __uint_as_float(((uint32_t)x[k]) << 16);
          a[j] = __builtin_fmaf(e, e, a[j]);
        }
      }
    }
  }
  float val = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) val = val + a[j];
  const float final = reduce_block_into_lanes<512>(s_vals, val);
  if (threadIdx.x == 0) {
    if (!isfinite(final)) *noop = 1;
    out[bidx[i]] = final;
  }
}

// TE's custom_multi_tensor_l2norm_reduce (one workgroup of 512): the same per-thread sequential
// sum over p[t + k*512], loads batched ahead (TE waits for each), the same block reduction, sqrtf.
__global__ __launch_bounds__(512) void l2n_reduce(const float* __restrict__ p, float* __restrict__ ret, int n) {
  __shared__ float s_vals[512];
  float v = 0.f;
  int i = threadIdx.x;
  for (; i + 15 * 512 < n; i += 16 * 512) {
    float q[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) q[k] = p[i + k * 512];
#pragma unroll
    for (int k = 0; k < 16; ++k) v = v + q[k];
  }
  for (; i < n; i += 512) v = v + p[i];
  const float final = reduce_block_into_lanes<512>(s_vals, v);
  if (threadIdx.x == 0) *ret = sqrtf(final);
}

// meta (int64): [full ptrs (nfull) | rest ptrs (nrest) | int32: full idx (nfull), rest idx (nrest),
//                rest (nrem, lim) pairs (2*nrest)]
torch::Tensor l2norm_bf16(torch::Tensor meta, int64_t nfull, int64_t nrest, int64_t nblocks, torch::Tensor noop,
                          torch::Tensor rmask) {
  auto st = at::cuda::getCurrentCUDAStream();
  const int64_t* p64 = reinterpret_cast<const int64_t*>(meta.data_ptr());
  const int* p32 = reinterpret_cast<const int*>(p64 + nfull + nrest);
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(meta.device());
  auto part = torch::empty({nblocks}, opts);
  auto ret = torch::empty({1}, opts);
  int* nz = noop.data_ptr<int>();
  if (nfull > 0)
    hipLaunchKernelGGL(l2n_full, dim3((nfull + 7) / 8), dim3(256), 0, st, p64, p32, (int)nfull, part.data_ptr<float>(), nz,
                       p32 + nfull + 3 * nrest,
                       rmask.numel() > 0 ? reinterpret_cast<const unsigned char*>(rmask.data_ptr()) : nullptr);
  if (nrest > 0)
    hipLaunchKernelGGL(l2n_rest, dim3(nrest), dim3(512), 0, st, p64 + nfull, p32 + nfull + nrest, p32 + nfull,
                       part.data_ptr<float>(), nz);
  hipLaunchKernelGGL(l2n_reduce, dim3(1), dim3(512), 0, st, part.data_ptr<float>(), ret.data_ptr<float>(), (int)nblocks);
  return ret;
}
"""

_MOD = []
_META = {}
_LOGGED = []


def _module():
    if not _MOD:
        from torch.utils.cpp_extension import load_inline

        _MOD.append(
            load_inline(
                "primus_exact_l2norm_v4",
                cpp_sources="torch::Tensor l2norm_bf16(torch::Tensor meta, int64_t nfull, int64_t nrest, int64_t nblocks, torch::Tensor noop, torch::Tensor rmask);",
                cuda_sources=_SRC,
                functions=["l2norm_bf16"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
        )
    return _MOD[0]


def _meta(tensors):
    # Built once per gradient list (the optimizer may norm more than one list per step); a
    # rebuild would cost a host loop and a blocking host-to-device copy.
    zr = [_zero_rows(t) for t in tensors]
    mask = next((z[0] for z in zr if z is not None), None)
    key = tuple((t.data_ptr(), t.numel()) for t in tensors) + (0 if mask is None else mask.data_ptr(),)
    m = _META.get(key)
    if m is None:
        import numpy as np

        fptr, fidx, rptr, ridx, rlim, frow = [], [], [], [], [], []
        b = 0
        for t, z in zip(tensors, zr):
            n, base = t.numel(), t.data_ptr()
            nch = (n + CHUNK - 1) // CHUNK
            c = np.arange(nch, dtype=np.int64)
            ptr = base + 2 * CHUNK * c
            nrem = n - CHUNK * c
            full = (nrem >= CHUNK) & (ptr % 16 == 0)
            fptr.append(ptr[full]); fidx.append(b + c[full])
            if z is not None and z[0] is mask:
                _, row_len, gbase, _ = z
                e = (ptr[full] - gbase) // 2
                frow.append(np.stack([e // row_len, (e + CHUNK - 1) // row_len], 1).reshape(-1))
            else:
                frow.append(np.full(2 * int(full.sum()), -1, np.int64))
            rest = ~full
            rptr.append(ptr[rest]); ridx.append(b + c[rest])
            rlim.append(np.stack([nrem[rest], np.minimum(nrem[rest], CHUNK)], 1).reshape(-1))
            b += nch
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
        p64 = np.concatenate([cat(fptr, np.int64), cat(rptr, np.int64)])
        i32 = np.concatenate([cat(fidx, np.int32), cat(ridx, np.int32), cat(rlim, np.int32), cat(frow, np.int32)])
        if i32.size % 2:
            i32 = np.concatenate([i32, np.zeros(1, np.int32)])
        host = torch.from_numpy(np.concatenate([p64, i32.view(np.int64)]))
        meta = host.to(tensors[0].device)
        if len(_META) >= 8:
            _META.clear()
        nfull = int(sum(x.size for x in fptr))
        m = _META[key] = (meta, nfull, b - nfull, b, mask)
    return m


def _zero_rows(t):
    try:
        from primus.backends.megatron.core.fusions import exact_embedding_grad as eeg
    except Exception:  # pragma: no cover
        return None
    return eeg.zero_row_mask(t)


_EMPTY = {}


def split_pieces(tensors, max_numel: int = 1 << 30):
    """Views of at most ``max_numel`` elements (te_l2norm_large_tensor_patches._pieces)."""
    if all(t.numel() <= max_numel for t in tensors):
        return tensors
    out = []
    for t in tensors:
        out.extend([t] if t.numel() <= max_numel else t.reshape(-1).split(max_numel))
    return out


def supported(tensors) -> bool:
    return (
        ENABLED
        and len(tensors) > 0
        and all(t.is_cuda and t.dtype == torch.bfloat16 and t.is_contiguous() and t.numel() > 0 for t in tensors)
    )


def l2norm(tensors, noop_flag) -> torch.Tensor:
    """TE ``multi_tensor_l2norm(65536, noop_flag, [tensors], False)[0]`` for BF16 tensors."""
    meta, nfull, nrest, nb, mask = _meta(tensors)
    if mask is None:
        dev = tensors[0].device
        mask = _EMPTY.get(dev)
        if mask is None:
            mask = _EMPTY[dev] = torch.empty(0, dtype=torch.uint8, device=dev)
    if not _LOGGED:
        _LOGGED.append(1)
        print(f"[exact_l2norm] {len(tensors)} tensors, {nb} chunks: {nfull} full aligned, {nrest} other, row-skip mask {mask.numel()}", flush=True)
    return _module().l2norm_bf16(meta, nfull, nrest, nb, noop_flag, mask)

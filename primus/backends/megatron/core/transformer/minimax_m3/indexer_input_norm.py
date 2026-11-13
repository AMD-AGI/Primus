###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``rms_norm(x.float(), w32).to(x.dtype)`` in one launch, bit for bit.

The MSA indexer re-normalises the raw residual stream in FP32 (``_indexer_input``): a
BF16->FP32 copy of ``[sq, b, h]``, ATen's ``vectorized_layer_norm_kernel<float, float, rms>``
and an FP32->BF16 copy -- three launches and ~250 MB of traffic per layer at M3's shape.

This HIP kernel is ATen's vectorized RMS path transcribed: the same block shape (one row per
block, ``warp_size x threads_y`` threads with ATen's ``threads_y`` rule), each thread summing
``x*x`` over its 4-wide vectors in the same order, the same shuffle-down warp reduction and
shared-memory tree across warps, ``rsqrtf(mean + eps)`` and ``gamma * (rstd * x)``, compiled
by the same hipcc. It reads the BF16 input directly (the FP32 copy is exact) and rounds the
FP32 result to BF16 once, as ``.to(bf16)`` does. Verified bitwise against ATen on random and
training tensors; any shape/arch it was not verified for (warp size != 32, other block
shapes) falls back to the eager ops. ``PRIMUS_FUSED_INDEXER_INPUT_NORM=0`` turns it off.
"""

from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("PRIMUS_FUSED_INDEXER_INPUT_NORM", "1") != "0"

_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
template <int WARP, int TY>
__global__ void rms_bf16_kernel(const __hip_bfloat16* __restrict__ X, const float* __restrict__ G,
                                __hip_bfloat16* __restrict__ Y, float eps, int N) {
  __shared__ float buf[TY];
  __shared__ float stat;
  const size_t row = blockIdx.x;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int nv = N / 4;
  float s = 0.f;
  for (int i = tid; i < nv; i += blockDim.x * blockDim.y) {
    float v[4];
    const __hip_bfloat16* x = X + row * N + i * 4;
    for (int k = 0; k < 4; k++) v[k] = __bfloat162float(x[k]);
    for (int k = 0; k < 4; k++) s = s + v[k] * v[k];
  }
  for (int off = WARP >> 1; off > 0; off >>= 1) s = __shfl_down(s, off, WARP) + s;
  for (int off = TY / 2; off > 0; off /= 2) {
    if (threadIdx.x == 0 && threadIdx.y >= off && threadIdx.y < 2 * off) buf[threadIdx.y - off] = s;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y < off) s = buf[threadIdx.y] + s;
    __syncthreads();
  }
  if (threadIdx.x == 0 && threadIdx.y == 0) stat = s / float(N);
  __syncthreads();
  const float rstd = rsqrtf(stat + eps);
  for (int i = tid; i < nv; i += blockDim.x * blockDim.y) {
    const __hip_bfloat16* x = X + row * N + i * 4;
    __hip_bfloat16* y = Y + row * N + i * 4;
    for (int k = 0; k < 4; k++) y[k] = __float2bfloat16(G[i * 4 + k] * (rstd * __bfloat162float(x[k])));
  }
}
torch::Tensor rms_bf16(torch::Tensor x, torch::Tensor g, double eps) {
  const int N = x.size(-1);
  const int M = x.numel() / N;
  auto y = torch::empty_like(x);
  hipLaunchKernelGGL((rms_bf16_kernel<32, 8>), dim3(M), dim3(32, 8), 0, at::cuda::getCurrentCUDAStream(),
                     (const __hip_bfloat16*)x.data_ptr(), g.data_ptr<float>(), (__hip_bfloat16*)y.data_ptr(), (float)eps, N);
  return y;
}
"""

_MOD = []


def _module():
    if not _MOD:
        from torch.utils.cpp_extension import load_inline

        _MOD.append(
            load_inline(
                "primus_indexer_input_rms",
                cpp_sources="torch::Tensor rms_bf16(torch::Tensor x, torch::Tensor g, double eps);",
                cuda_sources=_SRC.replace("#include <torch/extension.h>", "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>"),
                functions=["rms_bf16"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
        )
    return _MOD[0]


def _aten_block(N: int):
    """ATen's launch_vectorized_layer_norm_kernel block shape (num_threads() == 256)."""
    warp = torch.cuda.get_device_properties(torch.cuda.current_device()).warp_size
    ty = 256 // warp
    while (ty // 2) * warp >= N // 4 and ty > 1:
        ty //= 2
    return warp, ty


def supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    if not ENABLED or not x.is_cuda or x.dtype != torch.bfloat16 or w.dtype != torch.float32:
        return False
    N = x.shape[-1]
    if not (x.is_contiguous() and w.is_contiguous() and w.numel() == N and N % 4 == 0 and N <= (1 << 24)):
        return False
    if x.data_ptr() % 8 or w.data_ptr() % 16:
        return False
    return _aten_block(N) == (32, 8)


def indexer_input_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``torch.nn.functional.rms_norm(x.float(), (N,), w, eps).to(x.dtype)``, no autograd."""
    if supported(x, w):
        try:
            return _module().rms_bf16(x, w, eps)
        except Exception as e:  # build failure: fall back for the rest of the run
            global ENABLED
            ENABLED = False
            print(f"[indexer_input_norm] disabled: {e!r}")
    return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), w, eps).to(x.dtype)

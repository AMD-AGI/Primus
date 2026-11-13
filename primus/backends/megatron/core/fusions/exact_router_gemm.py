###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bit-exact router-gating backward GEMMs (FP32 router, BF16 input/weight).

Megatron's ``RouterGatingLinearFunction.backward`` (router_dtype FP32) runs, through TE's
``general_gemm``, two FP32 hipBLASLt GEMMs (``Cijk_*_S_B_..._MT32x32x16``: the only FP32
solution hipBLASLt offers for these shapes, ~2 % of peak) plus three casts:

    grad_input  = (W.float()^T-GEMM g).to(bf16)        NN, m=H n=T k=E
    grad_weight = (inp.float() NT-GEMM g).to(w.dtype)  NT, m=H n=E k=T

Both hipBLASLt kernels accumulate every 16x16 output tile with one sequential chain of
``v_wmma_f32_16x16x4_f32`` over K (read from their gfx1250 ISA). The kernels below issue the same
chain with the same K-to-lane-slot assignment, so every FP32 result is bit-identical:

* wgrad (``Ailk_Bjlk`` kernel): WMMA j takes k = 4j..4j+3, lane half h supplying slots 2h, 2h+1.
* dgrad (``Ailk_Bljk`` kernel, B fragments loaded with ds_load_b128): per 8-block kb, the first
  WMMA takes k = {kb+0, kb+1 | kb+4, kb+5}, the second {kb+2, kb+3 | kb+6, kb+7}.

They read the BF16 tensors directly (BF16 -> FP32 is exact) and round to BF16 in the epilogue (RNE,
as ``.to(bf16)``), so the FP32 copies of ``inp``/``W`` and the FP32 GEMM outputs are not materialised.
Microbench (rg/rg.cpp in work_kernel_engineer, T=4096 H=6144 E=128, 4 seeds incl. tiny/huge scales,
0 differing elements vs the pinned hipBLASLt solutions 2876/2877): wgrad 141 + 30 (cast) -> 81 us,
dgrad 98 (+ cast) -> 62 us. Off-switch: ``PRIMUS_EXACT_ROUTER_GEMM=0``.
"""

from __future__ import annotations

import os

import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

typedef float v2f __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned short bf16_t;

__device__ __forceinline__ bf16_t f2bf(float f) {  // RNE; NaN -> quiet (c10::BFloat16 rounding)
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffff) > 0x7f800000) return (bf16_t)((u >> 16) | 0x40);
  return (bf16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// wgrad: out[e][h] = sum_t inp[t][h] * g[t][e], t sequential (WMMA j: t = 4j..4j+3).
// WG 128 threads (4 waves), tile 64h x 32e; wave w: h-subtile w, 2 e-subtiles.
// LDS transposed [h][t], [e][t], 2 stages; global loads 2 chunks ahead.
#define WT 32
#define WS (WT + 4)
#define NQ 2
template <bool OUT_BF16>
__global__ __launch_bounds__(128) void rg_wgrad(const bf16_t* __restrict__ inp, const float* __restrict__ g,
                                                void* __restrict__ out, int T, int H, int E) {
  __shared__ float As[2][64 * WS];
  __shared__ float Bs[2][16 * NQ * WS];
  const int h0 = blockIdx.x * 64, e0 = blockIdx.y * 16 * NQ;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int li = lane & 15, kh = (lane >> 4) * 2;
  constexpr int NB = NQ;
  constexpr int BPR = 4 * NQ;
  uint4 ra[2][2]; float4 rb[2][NB];
  auto gload = [&](int slot, int t0) {
#pragma unroll
    for (int i = 0; i < 2; i++) { int idx = tid + i * 128; int r = idx >> 3, c = (idx & 7) * 8;
      ra[slot][i] = *(const uint4*)(inp + (size_t)(t0 + r) * H + h0 + c); }
#pragma unroll
    for (int i = 0; i < NB; i++) { int idx = tid + i * 128; int r = idx / BPR, c = (idx % BPR) * 4;
      rb[slot][i] = *(const float4*)(g + (size_t)(t0 + r) * E + e0 + c); }
  };
  auto sstore = [&](int slot, int st) {
#pragma unroll
    for (int i = 0; i < 2; i++) { int idx = tid + i * 128; int r = idx >> 3, c = (idx & 7) * 8;
      unsigned u[4] = {ra[slot][i].x, ra[slot][i].y, ra[slot][i].z, ra[slot][i].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        As[st][(c + 2 * j) * WS + r] = __uint_as_float(u[j] << 16);
        As[st][(c + 2 * j + 1) * WS + r] = __uint_as_float(u[j] & 0xffff0000u);
      } }
#pragma unroll
    for (int i = 0; i < NB; i++) { int idx = tid + i * 128; int r = idx / BPR, c = (idx % BPR) * 4;
      Bs[st][(c + 0) * WS + r] = rb[slot][i].x; Bs[st][(c + 1) * WS + r] = rb[slot][i].y;
      Bs[st][(c + 2) * WS + r] = rb[slot][i].z; Bs[st][(c + 3) * WS + r] = rb[slot][i].w; }
  };
  v8f acc[NQ];
#pragma unroll
  for (int q = 0; q < NQ; q++) acc[q] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
  auto compute = [&](int st) {
    const float* A = &As[st][(w * 16 + li) * WS + kh];
    const float* B = &Bs[st][li * WS + kh];
#pragma unroll
    for (int k = 0; k < WT; k += 4) {
      v2f a = *(const v2f*)(A + k);
#pragma unroll
      for (int q = 0; q < NQ; q++) {
        v2f b = *(const v2f*)(B + q * 16 * WS + k);
        acc[q] = __builtin_amdgcn_wmma_f32_16x16x4_f32(false, a, false, b, (short)0, acc[q], false, false);
      }
    }
  };
  const int nch = T / WT;  // even (T % 64 == 0)
  gload(0, 0); gload(1, WT); sstore(0, 0); __syncthreads();
  for (int c = 0; c < nch; c += 2) {
    if (c + 2 < nch) gload(0, (c + 2) * WT);
    compute(0);
    sstore(1, 1);
    __syncthreads();
    if (c + 3 < nch) gload(1, (c + 3) * WT);
    compute(1);
    if (c + 2 < nch) sstore(0, 0);
    __syncthreads();
  }
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    int e = e0 + q * 16 + li;
    int h = h0 + w * 16 + 8 * (lane >> 4);
    if (OUT_BF16) {
      uint4 o; unsigned* op = (unsigned*)&o;
#pragma unroll
      for (int r = 0; r < 4; r++) op[r] = (unsigned)f2bf(acc[q][2 * r]) | ((unsigned)f2bf(acc[q][2 * r + 1]) << 16);
      *(uint4*)((bf16_t*)out + (size_t)e * H + h) = o;
    } else {
#pragma unroll
      for (int r = 0; r < 8; r++) ((float*)out)[(size_t)e * H + h + r] = acc[q][r];
    }
  }
}

// dgrad: out[t][h] = sum_e W[e][h] * g[t][e], E = 128, hipBLASLt NN slot order (see module doc).
// WG 256 threads (8 waves), tile 128h x 128t; wave (wh 0..3, wt 0..1): 32h x 64t = 2x4 accumulators.
#define DWS (128 + 8)
#define DGS (128 + 4)
#define DG_SMEM (128 * DWS * 2 + 128 * DGS * 4)
__global__ __launch_bounds__(256) void rg_dgrad(const bf16_t* __restrict__ W, const float* __restrict__ g,
                                                bf16_t* __restrict__ out, int T, int H) {
  extern __shared__ float smem[];
  bf16_t* Ws = (bf16_t*)smem;
  float* Gs = smem + 128 * DWS / 2;
  const int h0 = blockIdx.x * 128, t0 = blockIdx.y * 128;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, wh = w & 3, wt = w >> 2;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int idx = tid + i * 256; int e = idx >> 4, c = (idx & 15) * 8;
    uint4 v = *(const uint4*)(W + (size_t)e * H + h0 + c);
    const bf16_t* p = (const bf16_t*)&v;
#pragma unroll
    for (int j = 0; j < 8; j++) Ws[(c + j) * DWS + e] = p[j];
  }
#pragma unroll
  for (int i = 0; i < 16; i++) {
    int idx = tid + i * 256; int r = idx >> 5, c = (idx & 31) * 4;
    *(float4*)(Gs + r * DGS + c) = *(const float4*)(g + (size_t)(t0 + r) * 128 + c);
  }
  __syncthreads();
  const int li = lane & 15, hf = lane >> 4;
  v8f acc[2][4];
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int q = 0; q < 4; q++) acc[a][q] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 2
  for (int kb = 0; kb < 128; kb += 8) {
#pragma unroll
    for (int half = 0; half < 2; half++) {
      const int k = kb + hf * 4 + half * 2;
      v2f a[2];
#pragma unroll
      for (int s = 0; s < 2; s++) {
        unsigned u = *(const unsigned*)(Ws + (wh * 32 + s * 16 + li) * DWS + k);
        a[s] = (v2f){__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)};
      }
#pragma unroll
      for (int q = 0; q < 4; q++) {
        v2f b = *(const v2f*)(Gs + (wt * 64 + q * 16 + li) * DGS + k);
#pragma unroll
        for (int s = 0; s < 2; s++)
          acc[s][q] = __builtin_amdgcn_wmma_f32_16x16x4_f32(false, a[s], false, b, (short)0, acc[s][q], false, false);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < 2; s++)
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int t = t0 + wt * 64 + q * 16 + li;
      int h = h0 + wh * 32 + s * 16 + 8 * hf;
      uint4 o; unsigned* op = (unsigned*)&o;
#pragma unroll
      for (int r = 0; r < 4; r++) op[r] = (unsigned)f2bf(acc[s][q][2 * r]) | ((unsigned)f2bf(acc[s][q][2 * r + 1]) << 16);
      *(uint4*)(out + (size_t)t * H + h) = o;
    }
}

torch::Tensor router_wgrad(torch::Tensor inp, torch::Tensor g, bool out_bf16) {
  const int T = inp.size(0), H = inp.size(1), E = g.size(1);
  auto out = torch::empty({E, H}, inp.options().dtype(out_bf16 ? torch::kBFloat16 : torch::kFloat32));
  auto st = at::hip::getCurrentHIPStream();
  dim3 grid(H / 64, E / (16 * NQ));
  if (out_bf16)
    hipLaunchKernelGGL(rg_wgrad<true>, grid, dim3(128), 0, st, (const bf16_t*)inp.data_ptr(), g.data_ptr<float>(), out.data_ptr(), T, H, E);
  else
    hipLaunchKernelGGL(rg_wgrad<false>, grid, dim3(128), 0, st, (const bf16_t*)inp.data_ptr(), g.data_ptr<float>(), out.data_ptr(), T, H, E);
  return out;
}

torch::Tensor router_dgrad(torch::Tensor w, torch::Tensor g) {
  const int H = w.size(1), T = g.size(0);
  static bool attr = false;
  if (!attr) { hipFuncSetAttribute((const void*)rg_dgrad, hipFuncAttributeMaxDynamicSharedMemorySize, DG_SMEM); attr = true; }
  auto out = torch::empty({T, H}, w.options());
  auto st = at::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(rg_dgrad, dim3(H / 128, T / 128), dim3(256), DG_SMEM, st, (const bf16_t*)w.data_ptr(), g.data_ptr<float>(),
                     (bf16_t*)out.data_ptr(), T, H);
  return out;
}
"""

_MOD = []
_LOGGED = []


def enabled() -> bool:
    return os.environ.get("PRIMUS_EXACT_ROUTER_GEMM", "1") != "0"


def _module():
    if not _MOD:
        from torch.utils.cpp_extension import load_inline

        _MOD.append(
            load_inline(
                "primus_exact_router_gemm_v1",
                cpp_sources=(
                    "torch::Tensor router_wgrad(torch::Tensor inp, torch::Tensor g, bool out_bf16);\n"
                    "torch::Tensor router_dgrad(torch::Tensor w, torch::Tensor g);"
                ),
                cuda_sources=_SRC,
                functions=["router_wgrad", "router_dgrad"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
        )
    return _MOD[0]


def eligible(inp2d, weight, bias, grad2d, router_dtype, input_dtype, weight_dtype) -> bool:
    """Shapes/dtypes the kernels reproduce exactly; anything else takes Megatron's path."""
    if router_dtype != torch.float32 or bias is not None:
        return False
    if inp2d.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or grad2d.dtype != torch.float32:
        return False
    if input_dtype != torch.bfloat16 or weight_dtype not in (torch.bfloat16, torch.float32):
        return False
    if not (inp2d.is_cuda and inp2d.is_contiguous() and weight.is_contiguous() and grad2d.is_contiguous()):
        return False
    T, H = inp2d.shape
    E = weight.shape[0]
    return (
        weight.shape[1] == H
        and grad2d.shape == (T, E)
        and E == 128
        and T % 128 == 0
        and H % 128 == 0
        and T * H < 2**31
        and inp2d.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and grad2d.data_ptr() % 16 == 0
    )


def router_backward(inp2d, weight, grad2d, weight_dtype):
    m = _module()
    grad_input = m.router_dgrad(weight, grad2d)
    grad_weight = m.router_wgrad(inp2d, grad2d, weight_dtype == torch.bfloat16)
    return grad_input, grad_weight


def install() -> bool:
    from megatron.core.transformer.moe import moe_utils

    cls = moe_utils.RouterGatingLinearFunction
    if getattr(cls.backward, "_primus_exact", False):
        return False
    original = cls.backward

    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors
        inp2d = inp.view(-1, inp.shape[-1])
        grad2d = grad_output.reshape(-1, grad_output.shape[-1])
        if moe_utils.te_general_gemm is None or not eligible(
            inp2d, weight, bias, grad2d, ctx.router_dtype, ctx.input_dtype, ctx.weight_dtype
        ):
            return original(ctx, grad_output)
        grad_input, grad_weight = router_backward(inp2d, weight, grad2d, ctx.weight_dtype)
        if not _LOGGED:
            _LOGGED.append(1)
            from primus.core.utils.module_utils import log_rank_0

            log_rank_0(f"[exact_router_gemm] in use: T={inp2d.shape[0]} H={inp2d.shape[1]} E={weight.shape[0]} w={ctx.weight_dtype}")
        return grad_input.view(*inp.shape), grad_weight, None, None

    backward._primus_exact = True
    cls.backward = staticmethod(backward)
    return True

/*****************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ****************************************************************************/

#ifndef PRIMUS_FP8BLK_HOST_TEST
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#else
// CPU emulation build of the FP8-block tile path (fp8blk_host_test.cpp): the
// per-thread phase functions below are compiled for the host and driven by a
// sequential thread loop. Kernels, launches and the Python module are excluded.
#include <c10/util/BFloat16.h>
#include <algorithm>
#define __host__
#define __device__
#define __forceinline__ inline
struct float4 { float x, y, z, w; };
struct uint4 { uint32_t x, y, z, w; };
struct uint2 { uint32_t x, y; };
using std::min;
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

constexpr int kBlockSize = 512;
constexpr int kIlp = 8;

__device__ __forceinline__ void unpack_bf16x8(const uint4& packed, float* values) {
  const uint32_t words[kIlp / 2] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int j = 0; j < kIlp / 2; ++j) {
    values[j * 2] = c10::detail::f32_from_bits(
        static_cast<uint16_t>(words[j] & UINT32_C(0xffff)));
    values[j * 2 + 1] = c10::detail::f32_from_bits(
        static_cast<uint16_t>(words[j] >> 16));
  }
}

__device__ __forceinline__ uint4 pack_bf16x8(const float* values) {
  uint32_t words[kIlp / 2];
#pragma unroll
  for (int j = 0; j < kIlp / 2; ++j) {
    words[j] = static_cast<uint32_t>(
                   c10::detail::round_to_nearest_even(values[j * 2])) |
        (static_cast<uint32_t>(
             c10::detail::round_to_nearest_even(values[j * 2 + 1]))
         << 16);
  }
  return {words[0], words[1], words[2], words[3]};
}

#ifndef PRIMUS_FP8BLK_HOST_TEST
template <bool AdamW, typename MomentT, typename GradT>
__global__ __launch_bounds__(kBlockSize) void fused_adam_clip_kernel(
    int chunk_size,
    const int64_t* __restrict__ addresses,
    const int64_t* __restrict__ sizes,
    const int* __restrict__ block_to_tensor,
    const int* __restrict__ chunk_offsets,
    int total_chunks,
    float beta1,
    float beta2,
    float step_size,
    float beta2_corr_inv,
    float epsilon,
    float lr,
    float decay,
    const float* __restrict__ grad_norm,
    float max_norm) {
  const int global_chunk = blockIdx.x;
  if (global_chunk >= total_chunks) return;

  const int tensor_idx = block_to_tensor[global_chunk];
  const int chunk_idx = global_chunk - chunk_offsets[tensor_idx];
  GradT* __restrict__ grad =
      reinterpret_cast<GradT*>(addresses[tensor_idx * 5 + 0]);
  float* __restrict__ param =
      reinterpret_cast<float*>(addresses[tensor_idx * 5 + 1]);
  MomentT* __restrict__ exp_avg =
      reinterpret_cast<MomentT*>(addresses[tensor_idx * 5 + 2]);
  MomentT* __restrict__ exp_avg_sq =
      reinterpret_cast<MomentT*>(addresses[tensor_idx * 5 + 3]);
  c10::BFloat16* __restrict__ model_param =
      reinterpret_cast<c10::BFloat16*>(addresses[tensor_idx * 5 + 4]);

  const int64_t elem_offset = static_cast<int64_t>(chunk_idx) * chunk_size;
  grad += elem_offset;
  param += elem_offset;
  exp_avg += elem_offset;
  exp_avg_sq += elem_offset;
  if (model_param != nullptr) model_param += elem_offset;
  const int n_this = static_cast<int>(
      min(sizes[tensor_idx] - elem_offset, static_cast<int64_t>(chunk_size)));
  __shared__ float grad_scale_shared;
  if (threadIdx.x == 0) {
    const float clip_coeff = max_norm / (*grad_norm + 1.0e-6f);
    // Match Python's min(1.0, coeff), including its NaN behavior.
    grad_scale_shared = clip_coeff < 1.0f ? clip_coeff : 1.0f;
  }
  __syncthreads();
  const float grad_scale = grad_scale_shared;

  for (int i = threadIdx.x * kIlp; i < n_this;
       i += blockDim.x * kIlp) {
    float g[kIlp] = {};
    float p[kIlp] = {};
    float m[kIlp] = {};
    float v[kIlp] = {};
    const bool vectorized = i + kIlp <= n_this;

    if (vectorized) {
      const float4 p0 = *reinterpret_cast<const float4*>(param + i);
      const float4 p1 = *reinterpret_cast<const float4*>(param + i + 4);
      if constexpr (std::is_same_v<GradT, float>) {
        const float4 g0 = *reinterpret_cast<const float4*>(grad + i);
        const float4 g1 = *reinterpret_cast<const float4*>(grad + i + 4);
        g[0] = g0.x; g[1] = g0.y; g[2] = g0.z; g[3] = g0.w;
        g[4] = g1.x; g[5] = g1.y; g[6] = g1.z; g[7] = g1.w;
      } else {
        const uint4 packed_g = *reinterpret_cast<const uint4*>(grad + i);
        unpack_bf16x8(packed_g, g);
      }
      p[0] = p0.x; p[1] = p0.y; p[2] = p0.z; p[3] = p0.w;
      p[4] = p1.x; p[5] = p1.y; p[6] = p1.z; p[7] = p1.w;
      if constexpr (std::is_same_v<MomentT, float>) {
        const float4 m0 = *reinterpret_cast<const float4*>(exp_avg + i);
        const float4 m1 = *reinterpret_cast<const float4*>(exp_avg + i + 4);
        const float4 v0 = *reinterpret_cast<const float4*>(exp_avg_sq + i);
        const float4 v1 = *reinterpret_cast<const float4*>(exp_avg_sq + i + 4);
        m[0] = m0.x; m[1] = m0.y; m[2] = m0.z; m[3] = m0.w;
        m[4] = m1.x; m[5] = m1.y; m[6] = m1.z; m[7] = m1.w;
        v[0] = v0.x; v[1] = v0.y; v[2] = v0.z; v[3] = v0.w;
        v[4] = v1.x; v[5] = v1.y; v[6] = v1.z; v[7] = v1.w;
      } else {
        const uint4 packed_m = *reinterpret_cast<const uint4*>(exp_avg + i);
        const uint4 packed_v = *reinterpret_cast<const uint4*>(exp_avg_sq + i);
        unpack_bf16x8(packed_m, m);
        unpack_bf16x8(packed_v, v);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kIlp; ++j) {
        if (i + j < n_this) {
          g[j] = grad[i + j];
          p[j] = param[i + j];
          m[j] = exp_avg[i + j];
          v[j] = exp_avg_sq[i + j];
        }
      }
    }

#pragma unroll
    for (int j = 0; j < kIlp; ++j) {
      // No FMA contraction: identical rounding in the chunk and the FP8-block tile kernel.
#pragma clang fp contract(off)
      g[j] *= grad_scale;
      if constexpr (!AdamW) g[j] += decay * p[j];
      m[j] = beta1 * m[j] + (1.0f - beta1) * g[j];
      v[j] = beta2 * v[j] + (1.0f - beta2) * g[j] * g[j];
      const float denom = sqrtf(v[j] * beta2_corr_inv) + epsilon;
      if constexpr (AdamW) {
        p[j] = p[j] - step_size * (m[j] / denom) - lr * decay * p[j];
      } else {
        p[j] -= step_size * (m[j] / denom);
      }
    }

    if (vectorized) {
      *reinterpret_cast<float4*>(param + i) = {p[0], p[1], p[2], p[3]};
      *reinterpret_cast<float4*>(param + i + 4) = {p[4], p[5], p[6], p[7]};
      if constexpr (std::is_same_v<MomentT, float>) {
        *reinterpret_cast<float4*>(exp_avg + i) = {m[0], m[1], m[2], m[3]};
        *reinterpret_cast<float4*>(exp_avg + i + 4) = {m[4], m[5], m[6], m[7]};
        *reinterpret_cast<float4*>(exp_avg_sq + i) = {v[0], v[1], v[2], v[3]};
        *reinterpret_cast<float4*>(exp_avg_sq + i + 4) = {v[4], v[5], v[6], v[7]};
      } else {
        *reinterpret_cast<uint4*>(exp_avg + i) = pack_bf16x8(m);
        *reinterpret_cast<uint4*>(exp_avg_sq + i) = pack_bf16x8(v);
      }
      if (model_param != nullptr) {
#pragma unroll
        for (int j = 0; j < kIlp; ++j) {
          model_param[i + j] = static_cast<c10::BFloat16>(p[j]);
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < kIlp; ++j) {
        if (i + j < n_this) {
          param[i + j] = p[j];
          exp_avg[i + j] = m[j];
          exp_avg_sq[i + j] = v[j];
          if (model_param != nullptr) {
            model_param[i + j] = static_cast<c10::BFloat16>(p[j]);
          }
        }
      }
    }
  }
}
#endif  // PRIMUS_FP8BLK_HOST_TEST

// ---------------------------------------------------------------------------
// FP8 128x128-block weights (PRIMUS_FP8BLK_ADAM_QUANT=1).
//
// One 512-thread block per 128x128 tile of a [B, N, K] weight whose FP32 master
// shard is the whole parameter. The block runs the same per-element Adam as
// fused_adam_clip_kernel (statements copied verbatim, same compiler and flags),
// writes the FP32 master, the moments and the BF16 model parameter, and then
// quantizes the *BF16-rounded* updated weight exactly like Turbo's
// quant_fp8_blockwise_for_weight_kernel with POW2_SCALE (UE8M0):
//   s = pow2_floor(448 / max(amax|w_bf16|, 1e-4)), q = e4m3(clamp(w_bf16 * s, +-448)),
//   scale_inv = 1 / s (exact), e8m0 = exponent byte of scale_inv.
// It emits W [B,N,K] fp8, W^T [B,K,N] fp8 (same tile, same scale), scale_inv
// [B,N/128,K/128] fp32 and E8M0 bytes [B,N/128,K/128] and [B,K/128,N/128].
// Partial edge tiles follow the reference (out-of-range elements count as 0).
// ---------------------------------------------------------------------------
constexpr int kTile = 128;
constexpr int kTileColThreads = kTile / kIlp;                      // 16 threads per tile row
constexpr int kTileRowsPerIter = kBlockSize / kTileColThreads;     // 32 rows per pass
constexpr int kTileIters = kTile / kTileRowsPerIter;               // 4 passes
constexpr int kTilePitch = kTile + 4;                              // LDS row pitch (bytes)
constexpr int kTransRows = kTile / (kBlockSize / kTile);           // 32 W rows per W^T thread
constexpr int kFp8BlkAddrs = 10;
constexpr int kFp8BlkDims = 4;
constexpr float kFp8Max = 448.0f;

struct Fp8BlkTile {
  int64_t N, K;      // matrix dims (row-major [N, K] per batch entry)
  int64_t tiles_n, tiles_k;
  int64_t b, tm, tk; // tile coordinates
};

template <typename GradT, typename MomentT>
struct Fp8BlkPtrs {
  GradT* grad;
  float* param;
  MomentT* exp_avg;
  MomentT* exp_avg_sq;
  c10::BFloat16* model_param;
  uint8_t* w;        // [B, N, K]
  uint8_t* wt;       // [B, K, N]
  float* scale_inv;  // [B, tiles_n, tiles_k]
  uint8_t* e8m0;     // [B, tiles_n, tiles_k]
  uint8_t* e8m0_t;   // [B, tiles_k, tiles_n]
};

__host__ __device__ __forceinline__ float fp8blk_pow2_floor(float s) {
  uint32_t u;
  memcpy(&u, &s, 4);
  u &= UINT32_C(0x7F800000);
  float r;
  memcpy(&r, &u, 4);
  return r;
}

__host__ __device__ __forceinline__ uint32_t fp8blk_f32_bits(float s) {
  uint32_t u;
  memcpy(&u, &s, 4);
  return u;
}

#ifdef PRIMUS_FP8BLK_HOST_TEST
// Host model of the E4M3 (OCP, fn) round-to-nearest-even conversion for |x| <= 448.
inline uint8_t fp8blk_e4m3_rne_host(float x) {
  const uint32_t u = fp8blk_f32_bits(x);
  const uint8_t sign = static_cast<uint8_t>((u >> 24) & 0x80);
  const float a = std::fabs(x);
  if (a < 0.015625f) {  // below 2^-6: subnormal grid of 2^-9
    return sign | static_cast<uint8_t>(std::nearbyint(a * 512.0f));
  }
  uint32_t ua = fp8blk_f32_bits(a);
  ua = (ua + UINT32_C(0x7FFFF) + ((ua >> 20) & 1u)) & UINT32_C(0xFFF00000);
  const int e = static_cast<int>((ua >> 23) & 0xFF) - 127 + 7;
  return sign | static_cast<uint8_t>((e << 3) | ((ua >> 20) & 7u));
}
#endif

// 8 clamped values -> 8 E4M3 bytes (little-endian in a uint2). On the device this is
// the same intrinsic Triton emits for Turbo's quant (cvt.scalef32.pk8.fp8.f32, scale 1).
__host__ __device__ __forceinline__ uint2 fp8blk_cvt8(const float* q) {
#ifdef PRIMUS_FP8BLK_HOST_TEST
  uint8_t b[8];
  for (int j = 0; j < 8; ++j) b[j] = fp8blk_e4m3_rne_host(q[j]);
  uint2 r;
  memcpy(&r, b, 8);
  return r;
#else
  typedef float f32x8 __attribute__((ext_vector_type(8)));
  const f32x8 v = {q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]};
  const auto r = __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32(v, 1.0f);
  return {static_cast<uint32_t>(r[0]), static_cast<uint32_t>(r[1])};
#endif
}

__host__ __device__ __forceinline__ Fp8BlkTile fp8blk_tile_coords(
    const int64_t* dims, int tensor_idx, int64_t tile_idx) {
  Fp8BlkTile t;
  t.N = dims[tensor_idx * kFp8BlkDims + 0];
  t.K = dims[tensor_idx * kFp8BlkDims + 1];
  t.tiles_n = dims[tensor_idx * kFp8BlkDims + 2];
  t.tiles_k = dims[tensor_idx * kFp8BlkDims + 3];
  const int64_t per_mat = t.tiles_n * t.tiles_k;
  t.b = tile_idx / per_mat;
  const int64_t r = tile_idx - t.b * per_mat;
  t.tm = r / t.tiles_k;
  t.tk = r - t.tm * t.tiles_k;
  return t;
}

// Phase 1 (per thread): Adam on 4 x 8 elements of the tile, store master, moments and
// the BF16 parameter; keep the BF16-rounded weight in x and return the thread's amax.
template <bool AdamW, typename MomentT, typename GradT>
__host__ __device__ __forceinline__ float fp8blk_adam_phase(
    int tid, const Fp8BlkTile& t, const Fp8BlkPtrs<GradT, MomentT>& ptr,
    float beta1, float beta2, float step_size, float beta2_corr_inv, float epsilon,
    float lr, float decay, float grad_scale, float (&x)[kTileIters][kIlp]) {
  float amax = 0.0f;
  const int col = (tid % kTileColThreads) * kIlp;
  const int64_t k0 = t.tk * kTile + col;
  const bool k_vec_ok = (t.K % kIlp) == 0 && k0 + kIlp <= t.K;
#pragma unroll
  for (int it = 0; it < kTileIters; ++it) {
    const int row = it * kTileRowsPerIter + tid / kTileColThreads;
    const int64_t n = t.tm * kTile + row;
    const bool row_ok = n < t.N;
    const int64_t i = (t.b * t.N + n) * t.K + k0;
    GradT* __restrict__ grad = ptr.grad;
    float* __restrict__ param = ptr.param;
    MomentT* __restrict__ exp_avg = ptr.exp_avg;
    MomentT* __restrict__ exp_avg_sq = ptr.exp_avg_sq;
    float g[kIlp] = {};
    float p[kIlp] = {};
    float m[kIlp] = {};
    float v[kIlp] = {};
    const bool vectorized = row_ok && k_vec_ok;

    if (vectorized) {
      const float4 p0 = *reinterpret_cast<const float4*>(param + i);
      const float4 p1 = *reinterpret_cast<const float4*>(param + i + 4);
      if constexpr (std::is_same_v<GradT, float>) {
        const float4 g0 = *reinterpret_cast<const float4*>(grad + i);
        const float4 g1 = *reinterpret_cast<const float4*>(grad + i + 4);
        g[0] = g0.x; g[1] = g0.y; g[2] = g0.z; g[3] = g0.w;
        g[4] = g1.x; g[5] = g1.y; g[6] = g1.z; g[7] = g1.w;
      } else {
        const uint4 packed_g = *reinterpret_cast<const uint4*>(grad + i);
        unpack_bf16x8(packed_g, g);
      }
      p[0] = p0.x; p[1] = p0.y; p[2] = p0.z; p[3] = p0.w;
      p[4] = p1.x; p[5] = p1.y; p[6] = p1.z; p[7] = p1.w;
      if constexpr (std::is_same_v<MomentT, float>) {
        const float4 m0 = *reinterpret_cast<const float4*>(exp_avg + i);
        const float4 m1 = *reinterpret_cast<const float4*>(exp_avg + i + 4);
        const float4 v0 = *reinterpret_cast<const float4*>(exp_avg_sq + i);
        const float4 v1 = *reinterpret_cast<const float4*>(exp_avg_sq + i + 4);
        m[0] = m0.x; m[1] = m0.y; m[2] = m0.z; m[3] = m0.w;
        m[4] = m1.x; m[5] = m1.y; m[6] = m1.z; m[7] = m1.w;
        v[0] = v0.x; v[1] = v0.y; v[2] = v0.z; v[3] = v0.w;
        v[4] = v1.x; v[5] = v1.y; v[6] = v1.z; v[7] = v1.w;
      } else {
        const uint4 packed_m = *reinterpret_cast<const uint4*>(exp_avg + i);
        const uint4 packed_v = *reinterpret_cast<const uint4*>(exp_avg_sq + i);
        unpack_bf16x8(packed_m, m);
        unpack_bf16x8(packed_v, v);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kIlp; ++j) {
        if (row_ok && k0 + j < t.K) {
          g[j] = grad[i + j];
          p[j] = param[i + j];
          m[j] = exp_avg[i + j];
          v[j] = exp_avg_sq[i + j];
        }
      }
    }

    // --- per-element Adam: verbatim copy of fused_adam_clip_kernel ---
#pragma unroll
    for (int j = 0; j < kIlp; ++j) {
      // No FMA contraction: identical rounding in the chunk and the FP8-block tile kernel.
#pragma clang fp contract(off)
      g[j] *= grad_scale;
      if constexpr (!AdamW) g[j] += decay * p[j];
      m[j] = beta1 * m[j] + (1.0f - beta1) * g[j];
      v[j] = beta2 * v[j] + (1.0f - beta2) * g[j] * g[j];
      const float denom = sqrtf(v[j] * beta2_corr_inv) + epsilon;
      if constexpr (AdamW) {
        p[j] = p[j] - step_size * (m[j] / denom) - lr * decay * p[j];
      } else {
        p[j] -= step_size * (m[j] / denom);
      }
    }

    if (vectorized) {
      *reinterpret_cast<float4*>(param + i) = {p[0], p[1], p[2], p[3]};
      *reinterpret_cast<float4*>(param + i + 4) = {p[4], p[5], p[6], p[7]};
      if constexpr (std::is_same_v<MomentT, float>) {
        *reinterpret_cast<float4*>(exp_avg + i) = {m[0], m[1], m[2], m[3]};
        *reinterpret_cast<float4*>(exp_avg + i + 4) = {m[4], m[5], m[6], m[7]};
        *reinterpret_cast<float4*>(exp_avg_sq + i) = {v[0], v[1], v[2], v[3]};
        *reinterpret_cast<float4*>(exp_avg_sq + i + 4) = {v[4], v[5], v[6], v[7]};
      } else {
        *reinterpret_cast<uint4*>(exp_avg + i) = pack_bf16x8(m);
        *reinterpret_cast<uint4*>(exp_avg_sq + i) = pack_bf16x8(v);
      }
      const uint4 pb = pack_bf16x8(p);  // == static_cast<c10::BFloat16>(p[j]) (RNE)
      *reinterpret_cast<uint4*>(ptr.model_param + i) = pb;
      unpack_bf16x8(pb, x[it]);
    } else {
#pragma unroll
      for (int j = 0; j < kIlp; ++j) {
        x[it][j] = 0.0f;
        if (row_ok && k0 + j < t.K) {
          param[i + j] = p[j];
          exp_avg[i + j] = m[j];
          exp_avg_sq[i + j] = v[j];
          const c10::BFloat16 pb = static_cast<c10::BFloat16>(p[j]);
          ptr.model_param[i + j] = pb;
          x[it][j] = static_cast<float>(pb);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kIlp; ++j) amax = fmaxf(amax, fabsf(x[it][j]));
  }
  return amax;
}

__host__ __device__ __forceinline__ float fp8blk_scale(float amax) {
  return fp8blk_pow2_floor(kFp8Max / fmaxf(amax, 1.0e-4f));
}

// Phase 2 (per thread): quantize the 4 x 8 weights, store W and stage the bytes in LDS.
__host__ __device__ __forceinline__ void fp8blk_quant_phase(
    int tid, const Fp8BlkTile& t, uint8_t* w, float s, const float (&x)[kTileIters][kIlp],
    uint8_t* lds) {
  const int col = (tid % kTileColThreads) * kIlp;
  const int64_t k0 = t.tk * kTile + col;
  const bool k_vec_ok = (t.K % kIlp) == 0 && k0 + kIlp <= t.K;
#pragma unroll
  for (int it = 0; it < kTileIters; ++it) {
    const int row = it * kTileRowsPerIter + tid / kTileColThreads;
    const int64_t n = t.tm * kTile + row;
    float q[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) q[j] = fminf(fmaxf(x[it][j] * s, -kFp8Max), kFp8Max);
    const uint2 bytes = fp8blk_cvt8(q);
    *reinterpret_cast<uint2*>(lds + row * kTilePitch + col) = bytes;
    if (n < t.N) {
      const int64_t i = (t.b * t.N + n) * t.K + k0;
      if (k_vec_ok) {
        *reinterpret_cast<uint2*>(w + i) = bytes;
      } else {
        uint8_t b8[kIlp];
        memcpy(b8, &bytes, kIlp);
        for (int j = 0; j < kIlp; ++j) {
          if (k0 + j < t.K) w[i + j] = b8[j];
        }
      }
    }
  }
}

// Phase 3 (per thread, after the LDS barrier): W^T row k gets tile column k.
__host__ __device__ __forceinline__ void fp8blk_transpose_phase(
    int tid, const Fp8BlkTile& t, uint8_t* wt, const uint8_t* lds) {
  const int kc = tid / (kBlockSize / kTile);
  const int part = tid % (kBlockSize / kTile);
  const int64_t k = t.tk * kTile + kc;
  if (k >= t.K) return;
  uint8_t b32[kTransRows];
#pragma unroll
  for (int r = 0; r < kTransRows; ++r) b32[r] = lds[(part * kTransRows + r) * kTilePitch + kc];
  const int64_t n0 = t.tm * kTile + part * kTransRows;
  const int64_t i = (t.b * t.K + k) * t.N + n0;
  if ((t.N % 16) == 0 && n0 + kTransRows <= t.N) {
    uint4 v0, v1;
    memcpy(&v0, b32, 16);
    memcpy(&v1, b32 + 16, 16);
    *reinterpret_cast<uint4*>(wt + i) = v0;
    *reinterpret_cast<uint4*>(wt + i + 16) = v1;
  } else {
    for (int r = 0; r < kTransRows; ++r) {
      if (n0 + r < t.N) wt[i + r] = b32[r];
    }
  }
}

__host__ __device__ __forceinline__ void fp8blk_scale_store(
    const Fp8BlkTile& t, float s, float* scale_inv, uint8_t* e8m0, uint8_t* e8m0_t) {
  const float inv = 1.0f / s;
  const uint8_t e = static_cast<uint8_t>((fp8blk_f32_bits(inv) >> 23) & 0xFF);
  const int64_t idx = (t.b * t.tiles_n + t.tm) * t.tiles_k + t.tk;
  scale_inv[idx] = inv;
  e8m0[idx] = e;
  e8m0_t[(t.b * t.tiles_k + t.tk) * t.tiles_n + t.tm] = e;
}

#ifndef PRIMUS_FP8BLK_HOST_TEST
template <bool AdamW, typename MomentT, typename GradT>
__global__ __launch_bounds__(kBlockSize) void fused_adam_clip_fp8blk_kernel(
    const int64_t* __restrict__ addresses,
    const int64_t* __restrict__ dims,
    const int* __restrict__ tile_to_tensor,
    const int64_t* __restrict__ tile_offsets,
    int total_tiles,
    float beta1,
    float beta2,
    float step_size,
    float beta2_corr_inv,
    float epsilon,
    float lr,
    float decay,
    const float* __restrict__ grad_norm,
    float max_norm) {
  const int global_tile = blockIdx.x;
  if (global_tile >= total_tiles) return;
  const int tensor_idx = tile_to_tensor[global_tile];
  const Fp8BlkTile t =
      fp8blk_tile_coords(dims, tensor_idx, global_tile - tile_offsets[tensor_idx]);
  const int64_t* a = addresses + tensor_idx * kFp8BlkAddrs;
  Fp8BlkPtrs<GradT, MomentT> ptr;
  ptr.grad = reinterpret_cast<GradT*>(a[0]);
  ptr.param = reinterpret_cast<float*>(a[1]);
  ptr.exp_avg = reinterpret_cast<MomentT*>(a[2]);
  ptr.exp_avg_sq = reinterpret_cast<MomentT*>(a[3]);
  ptr.model_param = reinterpret_cast<c10::BFloat16*>(a[4]);
  ptr.w = reinterpret_cast<uint8_t*>(a[5]);
  ptr.wt = reinterpret_cast<uint8_t*>(a[6]);
  ptr.scale_inv = reinterpret_cast<float*>(a[7]);
  ptr.e8m0 = reinterpret_cast<uint8_t*>(a[8]);
  ptr.e8m0_t = reinterpret_cast<uint8_t*>(a[9]);

  __shared__ float grad_scale_shared;
  __shared__ float wave_amax[kBlockSize / 32];
  __shared__ float tile_scale;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kTile * kTilePitch];
  if (threadIdx.x == 0) {
    const float clip_coeff = max_norm / (*grad_norm + 1.0e-6f);
    // Match Python's min(1.0, coeff), including its NaN behavior.
    grad_scale_shared = clip_coeff < 1.0f ? clip_coeff : 1.0f;
  }
  __syncthreads();
  const float grad_scale = grad_scale_shared;

  float x[kTileIters][kIlp];
  float amax = fp8blk_adam_phase<AdamW, MomentT, GradT>(
      threadIdx.x, t, ptr, beta1, beta2, step_size, beta2_corr_inv, epsilon, lr, decay,
      grad_scale, x);
  // Block amax (max is exact, order-independent).
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
  if ((threadIdx.x & 31) == 0) wave_amax[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = wave_amax[0];
#pragma unroll
    for (int w = 1; w < kBlockSize / 32; ++w) a = fmaxf(a, wave_amax[w]);
    tile_scale = fp8blk_scale(a);
  }
  __syncthreads();
  const float s = tile_scale;
  fp8blk_quant_phase(threadIdx.x, t, ptr.w, s, x, lds);
  __syncthreads();
  fp8blk_transpose_phase(threadIdx.x, t, ptr.wt, lds);
  if (threadIdx.x == 0) fp8blk_scale_store(t, s, ptr.scale_inv, ptr.e8m0, ptr.e8m0_t);
}

void fused_adam_clip(
    torch::Tensor addresses,
    torch::Tensor sizes,
    torch::Tensor block_to_tensor,
    torch::Tensor chunk_offsets,
    int total_chunks,
    int chunk_size,
    double lr,
    double beta1,
    double beta2,
    double epsilon,
    int step,
    int mode,
    int bias_correction,
    double weight_decay,
    int moment_dtype,
    int grad_dtype,
    torch::Tensor grad_norm,
    double max_norm) {
  TORCH_CHECK(addresses.is_cuda() && sizes.is_cuda() &&
              block_to_tensor.is_cuda() && chunk_offsets.is_cuda(),
              "fused_adam_clip metadata must be on GPU");
  TORCH_CHECK(addresses.scalar_type() == at::kLong && sizes.scalar_type() == at::kLong,
              "fused_adam_clip addresses and sizes must be int64");
  TORCH_CHECK(block_to_tensor.scalar_type() == at::kInt &&
              chunk_offsets.scalar_type() == at::kInt,
              "fused_adam_clip maps must be int32");
  TORCH_CHECK(grad_norm.is_cuda() && grad_norm.scalar_type() == at::kFloat &&
              grad_norm.numel() == 1,
              "fused_adam_clip grad_norm must be one FP32 GPU scalar");
  TORCH_CHECK(grad_norm.get_device() == addresses.get_device(),
              "fused_adam_clip grad_norm and optimizer tensors must share a device");
  TORCH_CHECK(mode == 0 || mode == 1, "fused_adam_clip Adam mode must be 0 or 1");
  TORCH_CHECK(moment_dtype == 0 || moment_dtype == 1,
              "fused_adam_clip moment dtype must be 0 (FP32) or 1 (BF16)");
  TORCH_CHECK(grad_dtype == 0 || grad_dtype == 1,
              "fused_adam_clip gradient dtype must be 0 (FP32) or 1 (BF16)");

  float correction1 = 1.0f;
  float correction2 = 1.0f;
  if (bias_correction == 1) {
    correction1 = 1.0f - std::pow(static_cast<float>(beta1), step);
    correction2 = 1.0f - std::pow(static_cast<float>(beta2), step);
  }
  const float step_size = static_cast<float>(lr) / correction1;
  const float beta2_corr_inv = 1.0f / correction2;
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(ADAMW, MOMENT_T, GRAD_T) \
  fused_adam_clip_kernel<ADAMW, MOMENT_T, GRAD_T> \
      <<<total_chunks, kBlockSize, 0, stream>>>( \
      chunk_size, addresses.data_ptr<int64_t>(), sizes.data_ptr<int64_t>(), \
      block_to_tensor.data_ptr<int>(), chunk_offsets.data_ptr<int>(), total_chunks, \
      static_cast<float>(beta1), static_cast<float>(beta2), step_size, \
      beta2_corr_inv, static_cast<float>(epsilon), static_cast<float>(lr), \
      static_cast<float>(weight_decay), grad_norm.data_ptr<float>(), \
      static_cast<float>(max_norm))
  if (mode == 1 && moment_dtype == 0 && grad_dtype == 0) {
    LAUNCH(true, float, float);
  } else if (mode == 1 && moment_dtype == 0) {
    LAUNCH(true, float, c10::BFloat16);
  } else if (mode == 1 && grad_dtype == 0) {
    LAUNCH(true, c10::BFloat16, float);
  } else if (mode == 1) {
    LAUNCH(true, c10::BFloat16, c10::BFloat16);
  } else if (moment_dtype == 0 && grad_dtype == 0) {
    LAUNCH(false, float, float);
  } else if (moment_dtype == 0) {
    LAUNCH(false, float, c10::BFloat16);
  } else if (grad_dtype == 0) {
    LAUNCH(false, c10::BFloat16, float);
  } else {
    LAUNCH(false, c10::BFloat16, c10::BFloat16);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Per tensor: addresses[10] = grad, fp32 master, exp_avg, exp_avg_sq, bf16 param, W fp8,
// W^T fp8, scale_inv fp32, e8m0, e8m0_t; dims[4] = N, K, tiles_n, tiles_k (B is implied
// by the tile count). Scalars are derived exactly as in fused_adam_clip.
void fused_adam_clip_fp8blk(
    torch::Tensor addresses,
    torch::Tensor dims,
    torch::Tensor tile_to_tensor,
    torch::Tensor tile_offsets,
    int total_tiles,
    double lr,
    double beta1,
    double beta2,
    double epsilon,
    int step,
    int mode,
    int bias_correction,
    double weight_decay,
    int moment_dtype,
    int grad_dtype,
    torch::Tensor grad_norm,
    double max_norm) {
  TORCH_CHECK(addresses.is_cuda() && dims.is_cuda() && tile_to_tensor.is_cuda() &&
              tile_offsets.is_cuda(), "fused_adam_clip_fp8blk metadata must be on GPU");
  TORCH_CHECK(addresses.scalar_type() == at::kLong && dims.scalar_type() == at::kLong &&
              tile_offsets.scalar_type() == at::kLong,
              "fused_adam_clip_fp8blk addresses, dims and tile_offsets must be int64");
  TORCH_CHECK(tile_to_tensor.scalar_type() == at::kInt,
              "fused_adam_clip_fp8blk tile_to_tensor must be int32");
  TORCH_CHECK(grad_norm.is_cuda() && grad_norm.scalar_type() == at::kFloat &&
              grad_norm.numel() == 1,
              "fused_adam_clip_fp8blk grad_norm must be one FP32 GPU scalar");
  TORCH_CHECK(mode == 0 || mode == 1, "fused_adam_clip_fp8blk Adam mode must be 0 or 1");
  TORCH_CHECK(moment_dtype == 0 || moment_dtype == 1, "moment dtype must be 0 (FP32) or 1 (BF16)");
  TORCH_CHECK(grad_dtype == 0 || grad_dtype == 1, "gradient dtype must be 0 (FP32) or 1 (BF16)");
  if (total_tiles == 0) return;

  float correction1 = 1.0f;
  float correction2 = 1.0f;
  if (bias_correction == 1) {
    correction1 = 1.0f - std::pow(static_cast<float>(beta1), step);
    correction2 = 1.0f - std::pow(static_cast<float>(beta2), step);
  }
  const float step_size = static_cast<float>(lr) / correction1;
  const float beta2_corr_inv = 1.0f / correction2;
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(ADAMW, MOMENT_T, GRAD_T) \
  fused_adam_clip_fp8blk_kernel<ADAMW, MOMENT_T, GRAD_T> \
      <<<total_tiles, kBlockSize, 0, stream>>>( \
      addresses.data_ptr<int64_t>(), dims.data_ptr<int64_t>(), \
      tile_to_tensor.data_ptr<int>(), tile_offsets.data_ptr<int64_t>(), total_tiles, \
      static_cast<float>(beta1), static_cast<float>(beta2), step_size, \
      beta2_corr_inv, static_cast<float>(epsilon), static_cast<float>(lr), \
      static_cast<float>(weight_decay), grad_norm.data_ptr<float>(), \
      static_cast<float>(max_norm))
  if (mode == 1 && moment_dtype == 0 && grad_dtype == 0) {
    LAUNCH(true, float, float);
  } else if (mode == 1 && moment_dtype == 0) {
    LAUNCH(true, float, c10::BFloat16);
  } else if (mode == 1 && grad_dtype == 0) {
    LAUNCH(true, c10::BFloat16, float);
  } else if (mode == 1) {
    LAUNCH(true, c10::BFloat16, c10::BFloat16);
  } else if (moment_dtype == 0 && grad_dtype == 0) {
    LAUNCH(false, float, float);
  } else if (moment_dtype == 0) {
    LAUNCH(false, float, c10::BFloat16);
  } else if (grad_dtype == 0) {
    LAUNCH(false, c10::BFloat16, float);
  } else {
    LAUNCH(false, c10::BFloat16, c10::BFloat16);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif  // PRIMUS_FP8BLK_HOST_TEST

}  // namespace

#ifndef PRIMUS_FP8BLK_HOST_TEST
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_adam_clip", &fused_adam_clip,
             "Fused Adam/AdamW with FP32 or BF16 moments and in-register clipping");
  module.def("fused_adam_clip_fp8blk", &fused_adam_clip_fp8blk,
             "fused_adam_clip on whole 128x128-tiled weights that also emits FP8 W, W^T and "
             "UE8M0 scales of the updated BF16 weight");
}
#endif

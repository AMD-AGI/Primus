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

// The tensor pointers come from a device array, so the compiler cannot prove they are global
// and emits flat loads/stores (which also count against the LDS wait counter). These helpers
// access them through the global address space explicitly (tile kernel only); the cast goes
// through an integer because a generic->global pointer cast is folded back to flat.
#if defined(__HIP_DEVICE_COMPILE__)
#define FP8BLK_GLOBAL __attribute__((address_space(1)))
#else
#define FP8BLK_GLOBAL
#endif
template <typename V>
__host__ __device__ __forceinline__ V gld(const void* p) {
  return *(const FP8BLK_GLOBAL V*)(reinterpret_cast<uintptr_t>(p));
}
template <typename V>
__host__ __device__ __forceinline__ void gst(void* p, const V& v) {
  *(FP8BLK_GLOBAL V*)(reinterpret_cast<uintptr_t>(p)) = v;
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
//
// Performance: the kernel is latency bound (Adam's IEEE div/sqrt plus the amax and LDS
// barriers per tile), so it needs many resident tiles per CU. The full-tile path keeps its
// VGPR count <= 85 (3 blocks of 16 waves per CU): uniform tile origin + 32-bit offsets, the
// BF16 weights parked in LDS between the Adam and the quant phase, and a lean element-wise
// edge path, since the kernel's VGPR count is the maximum over both paths.
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
  // Tile counts fit in 32 bits (the grid is one block per tile): 32-bit division is far
  // cheaper than the 64-bit software routine.
  const uint32_t per_mat = static_cast<uint32_t>(t.tiles_n * t.tiles_k);
  const uint32_t tiles_k = static_cast<uint32_t>(t.tiles_k);
  const uint32_t idx = static_cast<uint32_t>(tile_idx);
  const uint32_t b = idx / per_mat;
  const uint32_t r = idx - b * per_mat;
  const uint32_t tm = r / tiles_k;
  t.b = b;
  t.tm = tm;
  t.tk = r - tm * tiles_k;
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

// 8 consecutive elements of T as raw 16-byte words (one uint4 for BF16, two for FP32).
template <typename T>
struct Fp8BlkRaw8 {
  static constexpr int kWords = static_cast<int>(sizeof(T)) * kIlp / 16;
  uint4 w[kWords];
};

template <typename T>
__host__ __device__ __forceinline__ void fp8blk_load8(const T* src, Fp8BlkRaw8<T>& r) {
#pragma unroll
  for (int q = 0; q < Fp8BlkRaw8<T>::kWords; ++q) r.w[q] = gld<uint4>(reinterpret_cast<const uint4*>(src) + q);
}

template <typename T>
__host__ __device__ __forceinline__ void fp8blk_unpack8(const Fp8BlkRaw8<T>& r, float* out) {
  if constexpr (std::is_same_v<T, float>) {
    memcpy(out, r.w, sizeof(r.w));
  } else {
    unpack_bf16x8(r.w[0], out);
  }
}

// Full tiles are addressed as a uniform 64-bit tile origin plus a 32-bit per-thread element
// offset (fp8blk_tile_full guarantees 128 * K < 2^31), so the loads and stores use a scalar
// base register instead of a 64-bit VGPR address each.
__host__ __device__ __forceinline__ int64_t fp8blk_tile_origin(const Fp8BlkTile& t) {
  return (t.b * t.N + t.tm * kTile) * t.K + t.tk * kTile;
}

// LDS slot (16 bytes = 8 BF16) of thread tid's row group it in the parked weight tile; the
// tile is row-major [128][128] BF16 (slot = row * 16 + column group).
__host__ __device__ __forceinline__ int fp8blk_xs_slot(int tid, int it) {
  return it * kBlockSize + tid;
}

__host__ __device__ __forceinline__ uint32_t fp8blk_thread_off(int tid, int it, uint32_t K) {
  return static_cast<uint32_t>(it * kTileRowsPerIter + tid / kTileColThreads) * K +
      static_cast<uint32_t>((tid % kTileColThreads) * kIlp);
}

// Phase 1, full-tile fast path (whole 128x128 tile in range): identical math and stores to
// fp8blk_adam_phase. The addresses are a uniform tile origin plus 32-bit offsets and the BF16
// weights are parked in LDS for the quant phase instead of being held in registers: the VGPR
// count sets the occupancy (a 512-thread block is 4 waves per SIMD, 1024 VGPRs per lane per
// SIMD), which is what this latency-bound kernel is limited by.
template <bool AdamW, typename MomentT, typename GradT>
__host__ __device__ __forceinline__ float fp8blk_adam_full(
    int tid, const Fp8BlkTile& t, const Fp8BlkPtrs<GradT, MomentT>& ptr,
    float beta1, float beta2, float step_size, float beta2_corr_inv, float epsilon,
    float lr, float decay, float grad_scale, uint4* xs) {
  float amax = 0.0f;
  const int64_t o = fp8blk_tile_origin(t);
  GradT* __restrict__ grad = ptr.grad + o;
  float* __restrict__ param = ptr.param + o;
  MomentT* __restrict__ exp_avg = ptr.exp_avg + o;
  MomentT* __restrict__ exp_avg_sq = ptr.exp_avg_sq + o;
  c10::BFloat16* __restrict__ model_param = ptr.model_param + o;
  const uint32_t K = static_cast<uint32_t>(t.K);
#pragma unroll
  for (int it = 0; it < kTileIters; ++it) {
    const uint32_t i = fp8blk_thread_off(tid, it, K);
    Fp8BlkRaw8<GradT> rg;
    Fp8BlkRaw8<float> rp;
    Fp8BlkRaw8<MomentT> rm;
    Fp8BlkRaw8<MomentT> rv;
    fp8blk_load8(grad + i, rg);
    fp8blk_load8(param + i, rp);
    fp8blk_load8(exp_avg + i, rm);
    fp8blk_load8(exp_avg_sq + i, rv);
    float g[kIlp];
    float p[kIlp];
    float m[kIlp];
    float v[kIlp];
    fp8blk_unpack8(rg, g);
    fp8blk_unpack8(rp, p);
    fp8blk_unpack8(rm, m);
    fp8blk_unpack8(rv, v);

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

    gst(param + i, float4{p[0], p[1], p[2], p[3]});
    gst(param + i + 4, float4{p[4], p[5], p[6], p[7]});
    if constexpr (std::is_same_v<MomentT, float>) {
      gst(exp_avg + i, float4{m[0], m[1], m[2], m[3]});
      gst(exp_avg + i + 4, float4{m[4], m[5], m[6], m[7]});
      gst(exp_avg_sq + i, float4{v[0], v[1], v[2], v[3]});
      gst(exp_avg_sq + i + 4, float4{v[4], v[5], v[6], v[7]});
    } else {
      gst(exp_avg + i, pack_bf16x8(m));
      gst(exp_avg_sq + i, pack_bf16x8(v));
    }
    const uint4 pb = pack_bf16x8(p);  // == static_cast<c10::BFloat16>(p[j]) (RNE)
    gst(model_param + i, pb);
    xs[fp8blk_xs_slot(tid, it)] = pb;  // parked in LDS until the quant phase
    float x[kIlp];
    unpack_bf16x8(pb, x);
#pragma unroll
    for (int j = 0; j < kIlp; ++j) amax = fmaxf(amax, fabsf(x[j]));
  }
  return amax;
}

__host__ __device__ __forceinline__ bool fp8blk_tile_full(const Fp8BlkTile& t) {
  return (t.K % kIlp) == 0 && (t.tm + 1) * kTile <= t.N && (t.tk + 1) * kTile <= t.K &&
      (t.N % 16) == 0 && t.K * kTile < (INT64_C(1) << 31) && t.N * kTile < (INT64_C(1) << 31);
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
        gst(w + i, bytes);
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
    gst(wt + i, v0);
    gst(wt + i + 16, v1);
  } else {
    for (int r = 0; r < kTransRows; ++r) {
      if (n0 + r < t.N) wt[i + r] = b32[r];
    }
  }
}

// Generic (edge / odd-shape) Adam for the tile kernel: the same per-element math and stores as
// fp8blk_adam_phase, element by element with the BF16 weight (0 out of range) written straight
// to the row-major LDS tile, so this rarely used path holds few registers and does not lower
// the occupancy of the full-tile path.
constexpr int kFp8BlkGenericUnroll = 2;  // elements in flight per thread
template <bool AdamW, typename MomentT, typename GradT>
__host__ __device__ __forceinline__ float fp8blk_adam_generic(
    int tid, const Fp8BlkTile& t, const Fp8BlkPtrs<GradT, MomentT>& ptr,
    float beta1, float beta2, float step_size, float beta2_corr_inv, float epsilon,
    float lr, float decay, float grad_scale, uint4* xs) {
  float amax = 0.0f;
  const int col = (tid % kTileColThreads) * kIlp;
  const int64_t k0 = t.tk * kTile + col;
  GradT* __restrict__ grad = ptr.grad;
  float* __restrict__ param = ptr.param;
  MomentT* __restrict__ exp_avg = ptr.exp_avg;
  MomentT* __restrict__ exp_avg_sq = ptr.exp_avg_sq;
  uint16_t* xs16 = reinterpret_cast<uint16_t*>(xs);
#pragma unroll 1
  for (int it = 0; it < kTileIters; ++it) {
    const int row = it * kTileRowsPerIter + tid / kTileColThreads;
    const int64_t n = t.tm * kTile + row;
    const bool row_ok = n < t.N;
    const int64_t i = (t.b * t.N + n) * t.K + k0;
#pragma unroll kFp8BlkGenericUnroll
    for (int j = 0; j < kIlp; ++j) {
      c10::BFloat16 pb(0.0f);
      if (row_ok && k0 + j < t.K) {
        float g[1] = {grad[i + j]};
        float p[1] = {param[i + j]};
        float m[1] = {exp_avg[i + j]};
        float v[1] = {exp_avg_sq[i + j]};
        {
#pragma clang fp contract(off)
          // --- per-element Adam: verbatim copy of fused_adam_clip_kernel (one element) ---
          const int jj = 0;
          g[jj] *= grad_scale;
          if constexpr (!AdamW) g[jj] += decay * p[jj];
          m[jj] = beta1 * m[jj] + (1.0f - beta1) * g[jj];
          v[jj] = beta2 * v[jj] + (1.0f - beta2) * g[jj] * g[jj];
          const float denom = sqrtf(v[jj] * beta2_corr_inv) + epsilon;
          if constexpr (AdamW) {
            p[jj] = p[jj] - step_size * (m[jj] / denom) - lr * decay * p[jj];
          } else {
            p[jj] -= step_size * (m[jj] / denom);
          }
        }
        param[i + j] = p[0];
        exp_avg[i + j] = m[0];
        exp_avg_sq[i + j] = v[0];
        pb = static_cast<c10::BFloat16>(p[0]);
        ptr.model_param[i + j] = pb;
      }
      xs16[row * kTile + col + j] = pb.x;
      amax = fmaxf(amax, fabsf(static_cast<float>(pb)));
    }
  }
  return amax;
}

// Phase 2 from the parked BF16 weights (same bytes as fp8blk_quant_phase): stage the FP8 tile
// in LDS and store W (full tiles: uniform origin + 32-bit offsets; else edge-checked).
template <bool kFull>
__host__ __device__ __forceinline__ void fp8blk_quant_lds(
    int tid, const Fp8BlkTile& t, uint8_t* w, float s, const uint4* xs, uint8_t* lds) {
  const int col = (tid % kTileColThreads) * kIlp;
  const int64_t k0 = t.tk * kTile + col;
  const bool k_vec_ok = (t.K % kIlp) == 0 && k0 + kIlp <= t.K;
  uint8_t* wo = w + fp8blk_tile_origin(t);
  const uint32_t K = static_cast<uint32_t>(t.K);
#pragma unroll
  for (int it = 0; it < kTileIters; ++it) {
    const int row = it * kTileRowsPerIter + tid / kTileColThreads;
    float x[kIlp];
    unpack_bf16x8(xs[fp8blk_xs_slot(tid, it)], x);
    float q[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) q[j] = fminf(fmaxf(x[j] * s, -kFp8Max), kFp8Max);
    const uint2 bytes = fp8blk_cvt8(q);
    *reinterpret_cast<uint2*>(lds + row * kTilePitch + col) = bytes;
    if constexpr (kFull) {
      gst(wo + fp8blk_thread_off(tid, it, K), bytes);
    } else {
      const int64_t n = t.tm * kTile + row;
      if (n < t.N) {
        const int64_t i = (t.b * t.N + n) * t.K + k0;
        if (k_vec_ok) {
          gst(w + i, bytes);
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
}

// Full-tile phase 3.
__host__ __device__ __forceinline__ void fp8blk_transpose_full(
    int tid, const Fp8BlkTile& t, uint8_t* wt, const uint8_t* lds) {
  // Thread (kq, part) moves the 8x4 byte block rows part*8..+8 x cols 4*kq..+4 of the staged
  // tile to W^T rows 4*kq..+4, cols part*8..+8: 8 dword LDS reads (not 32 byte reads) and 4
  // 8-byte stores; the 16 lanes of a column group cover one 128-byte W^T row segment.
  const int kq = tid / (kTile / kIlp);
  const int part = tid % (kTile / kIlp);
  uint32_t d[kIlp];
#pragma unroll
  for (int r = 0; r < kIlp; ++r) {
    d[r] = *reinterpret_cast<const uint32_t*>(lds + (part * kIlp + r) * kTilePitch + kq * 4);
  }
  uint8_t* wto = wt + (t.b * t.K + t.tk * kTile) * t.N + t.tm * kTile;
  const uint32_t N = static_cast<uint32_t>(t.N);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      lo |= ((d[r] >> (8 * j)) & 0xFFu) << (8 * r);
      hi |= ((d[r + 4] >> (8 * j)) & 0xFFu) << (8 * r);
    }
    gst(wto + static_cast<uint32_t>(kq * 4 + j) * N + static_cast<uint32_t>(part * kIlp),
        uint2{lo, hi});
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

  __shared__ __attribute__((aligned(16))) float wave_amax[kBlockSize / 32];
  __shared__ __attribute__((aligned(16))) uint8_t lds[kTile * kTilePitch];
  // Every thread derives the clip scale itself (uniform scalar load, same arithmetic as the
  // chunk kernel): no barrier in front of the tile loads.
  const float clip_coeff = max_norm / (*grad_norm + 1.0e-6f);
  // Match Python's min(1.0, coeff), including its NaN behavior.
  const float grad_scale = clip_coeff < 1.0f ? clip_coeff : 1.0f;

  __shared__ uint4 xs[kTileIters * kBlockSize];  // parked BF16 weights (32 KB)
  float amax;
  const bool full = fp8blk_tile_full(t);  // block-uniform
  if (full) {
    amax = fp8blk_adam_full<AdamW, MomentT, GradT>(
        threadIdx.x, t, ptr, beta1, beta2, step_size, beta2_corr_inv, epsilon, lr, decay,
        grad_scale, xs);
  } else {
    amax = fp8blk_adam_generic<AdamW, MomentT, GradT>(
        threadIdx.x, t, ptr, beta1, beta2, step_size, beta2_corr_inv, epsilon, lr, decay,
        grad_scale, xs);
  }
  // Block amax (max is exact, order-independent).
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
  if ((threadIdx.x & 31) == 0) wave_amax[threadIdx.x >> 5] = amax;
  __syncthreads();
  // Every thread reduces the 16 wave maxima itself (one barrier instead of two).
  float tile_amax = wave_amax[0];
#pragma unroll
  for (int w = 1; w < kBlockSize / 32; ++w) tile_amax = fmaxf(tile_amax, wave_amax[w]);
  const float s = fp8blk_scale(tile_amax);
  if (full) {
    fp8blk_quant_lds<true>(threadIdx.x, t, ptr.w, s, xs, lds);
  } else {
    fp8blk_quant_lds<false>(threadIdx.x, t, ptr.w, s, xs, lds);
  }
  __syncthreads();
  if (full) {
    fp8blk_transpose_full(threadIdx.x, t, ptr.wt, lds);
  } else {
    fp8blk_transpose_phase(threadIdx.x, t, ptr.wt, lds);
  }
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

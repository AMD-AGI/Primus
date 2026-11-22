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

__host__ __device__ __forceinline__ void unpack_bf16x4(const uint2& packed, float* values) {
  const uint32_t words[2] = {packed.x, packed.y};
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    values[j * 2] = c10::detail::f32_from_bits(static_cast<uint16_t>(words[j] & UINT32_C(0xffff)));
    values[j * 2 + 1] = c10::detail::f32_from_bits(static_cast<uint16_t>(words[j] >> 16));
  }
}

__host__ __device__ __forceinline__ uint2 pack_bf16x4(const float* values) {
  uint32_t words[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    words[j] = static_cast<uint32_t>(c10::detail::round_to_nearest_even(values[j * 2])) |
        (static_cast<uint32_t>(c10::detail::round_to_nearest_even(values[j * 2 + 1])) << 16);
  }
  return {words[0], words[1]};
}

// 4 floats -> 4 BF16 (RNE) with the hardware conversion (v_cvt_pk_bf16_f32 on gfx1250) instead of c10's integer
// rounding (bfe + add3 + NaN compare + select per value). Identical to c10::detail::round_to_nearest_even for every
// one of the 2^32 float inputs once NaN maps to 0x7FC0 like c10 (exhaustive GPU check, adamattn bf16cvt_test.cu).
__host__ __device__ __forceinline__ uint16_t bf16_rne_hw(float f) {
#if defined(__HIP_DEVICE_COMPILE__)
  const uint16_t h = __builtin_bit_cast(uint16_t, static_cast<__bf16>(f));
  return f != f ? UINT16_C(0x7FC0) : h;
#else
  return c10::detail::round_to_nearest_even(f);
#endif
}

#if defined(__HIP_DEVICE_COMPILE__)
// Two floats -> two BF16 in one v_cvt_pk_bf16_f32, then c10's NaN encoding (0x7FC0) for a NaN input.
__device__ __forceinline__ uint32_t bf16x2_rne_hw(float a, float b) {
  typedef float f32x2 __attribute__((ext_vector_type(2)));
  typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
  uint32_t w = __builtin_bit_cast(uint32_t, __builtin_convertvector((f32x2){a, b}, bf16x2));
  w = a != a ? (w & UINT32_C(0xFFFF0000)) | UINT32_C(0x7FC0) : w;
  w = b != b ? (w & UINT32_C(0x0000FFFF)) | UINT32_C(0x7FC00000) : w;
  return w;
}
#endif

__host__ __device__ __forceinline__ uint4 pack_bf16x8_hw(const float* values);
__host__ __device__ __forceinline__ uint2 pack_bf16x4_hw(const float* values) {
#if defined(__HIP_DEVICE_COMPILE__)
  return {bf16x2_rne_hw(values[0], values[1]), bf16x2_rne_hw(values[2], values[3])};
#else
  return {static_cast<uint32_t>(bf16_rne_hw(values[0])) | (static_cast<uint32_t>(bf16_rne_hw(values[1])) << 16),
          static_cast<uint32_t>(bf16_rne_hw(values[2])) | (static_cast<uint32_t>(bf16_rne_hw(values[3])) << 16)};
#endif
}

__host__ __device__ __forceinline__ uint4 pack_bf16x8_hw(const float* values) {
  const uint2 lo = pack_bf16x4_hw(values);
  const uint2 hi = pack_bf16x4_hw(values + 4);
  return {lo.x, lo.y, hi.x, hi.y};
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
  // The BF16 copy goes out as 16-byte stores when its address allows (block-uniform), else per element as before.
  const bool mp_vec = model_param != nullptr && (reinterpret_cast<uintptr_t>(model_param) & 15) == 0;

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
        *reinterpret_cast<uint4*>(exp_avg + i) = pack_bf16x8_hw(m);
        *reinterpret_cast<uint4*>(exp_avg_sq + i) = pack_bf16x8_hw(v);
      }
      if (mp_vec) {
        // One 16-byte store of the hardware-rounded copy (same bits as the per-element c10 cast).
        *reinterpret_cast<uint4*>(model_param + i) = pack_bf16x8_hw(p);
      } else if (model_param != nullptr) {
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
// [B,N/128,K/128] fp32 and E8M0 bytes [B,N/128,K/128] and [B,K/128,N/128]; W^T and the
// transposed E8M0 bytes are skipped for a tensor whose W^T address is 0, and the BF16 model
// parameter store is skipped for a tensor whose BF16 address is 0 (PRIMUS_FP8BLK_ADAM_SKIP_BF16:
// MegaMoE expert weights, whose GEMMs read only the FP8 cache; the caller materializes the BF16
// weight from the FP32 master before any BF16 reader).
// Partial edge tiles follow the reference (out-of-range elements count as 0).
//
// Performance: the kernel streams ~20 B per parameter through HBM with Adam's IEEE div/sqrt and two LDS barriers
// per tile in between. v11 (adamfetch): the FP32->BF16 roundings (moments, BF16 weight) use the hardware converter
// (bf16x2_rne_hw), which cut the integer rounding sequences and took the kernel from 77 to 62 VGPRs (4 blocks per
// CU); the chunk kernel uses it too and stores its BF16 copy 16 bytes at a time. The full-tile path (adamattn, v10):
//  - wave-dense loads/stores: every global access instruction covers one contiguous tile row (512 B FP32, 256 B
//    BF16), not 16-byte pieces at a 32-byte lane stride (an HBM probe of this byte mix: 18.7 vs 13.4 TB/s);
//  - the BF16 store is a compile-time choice (a runtime-conditional store had the compiler keep all four packed
//    rows live: 126 VGPRs, 2 blocks per CU; now 77, 3 blocks per CU);
//  - the barriers order LDS only (no wait for the global stores in flight);
//  - uniform tile origin + 32-bit offsets, the BF16 weights parked in LDS between the Adam and the quant phase,
//    and a lean element-wise edge path, since the kernel's VGPR count is the maximum over both paths.
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
      if (ptr.model_param != nullptr) *reinterpret_cast<uint4*>(ptr.model_param + i) = pb;
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
          if (ptr.model_param != nullptr) ptr.model_param[i + j] = pb;
          x[it][j] = static_cast<float>(pb);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kIlp; ++j) amax = fmaxf(amax, fabsf(x[it][j]));
  }
  return amax;
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

// 4 consecutive elements of T (8 bytes for BF16, 16 for FP32).
template <typename T>
__host__ __device__ __forceinline__ void fp8blk_load4(const T* src, float* out) {
  if constexpr (std::is_same_v<T, float>) {
    const float4 r = gld<float4>(src);
    out[0] = r.x; out[1] = r.y; out[2] = r.z; out[3] = r.w;
  } else {
    unpack_bf16x4(gld<uint2>(src), out);
  }
}

template <typename T>
__host__ __device__ __forceinline__ void fp8blk_store4(T* dst, const float* in) {
  if constexpr (std::is_same_v<T, float>) {
    gst(dst, float4{in[0], in[1], in[2], in[3]});
  } else {
    gst(dst, pack_bf16x4_hw(in));
  }
}

// Phase 1, full-tile fast path (whole 128x128 tile in range): identical per-element math and the same stores as
// fp8blk_adam_phase, with a wave-dense element mapping: in pass `it`, wave w updates tile rows it*32 + 2w and
// it*32 + 2w + 1, lane l columns 4l..4l+3 of each. Every global load and store instruction then covers one
// contiguous tile row (512 B FP32, 256 B BF16) instead of 32-byte lane strides (two half-dense 16-byte FP32
// accesses per lane), which an HBM stream probe of this byte mix measured at 13.4 vs 18.7 TB/s. The BF16 weights
// are parked in LDS (row-major [128][128], the layout the quant phase reads) and the addresses are a uniform tile
// origin plus 32-bit offsets.
template <bool AdamW, typename MomentT, typename GradT, bool kWriteBF16>
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
  // kWriteBF16 false (the caller saw a null BF16 address, per tensor and block-uniform) skips the BF16 parameter
  // store; the BF16 value is still formed, since the FP8 quant reads it. A compile-time flag, not a runtime branch:
  // the compiler merged runtime-conditional stores into one block at the end, which kept every packed BF16 row
  // live through the update (VGPR 126).
  c10::BFloat16* __restrict__ model_param = ptr.model_param + o;
  uint2* xs2 = reinterpret_cast<uint2*>(xs);
  const uint32_t K = static_cast<uint32_t>(t.K);
  const int lane = tid & 31;
  const int wave = tid >> 5;
#pragma unroll
  for (int it = 0; it < kTileIters; ++it) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = it * kTileRowsPerIter + wave * 2 + h;
      const uint32_t i = static_cast<uint32_t>(row) * K + static_cast<uint32_t>(lane * 4);
      float g[4];
      float p[4];
      float m[4];
      float v[4];
      fp8blk_load4(grad + i, g);
      fp8blk_load4(param + i, p);
      fp8blk_load4(exp_avg + i, m);
      fp8blk_load4(exp_avg_sq + i, v);

      // --- per-element Adam: verbatim copy of fused_adam_clip_kernel ---
#pragma unroll
      for (int j = 0; j < 4; ++j) {
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
      fp8blk_store4(exp_avg + i, m);
      fp8blk_store4(exp_avg_sq + i, v);
      const uint2 pb = pack_bf16x4_hw(p);  // == static_cast<c10::BFloat16>(p[j]) (RNE)
      if constexpr (kWriteBF16) gst(model_param + i, pb);
      xs2[row * (kTile / 4) + lane] = pb;  // parked in LDS until the quant phase
      float x[4];
      unpack_bf16x4(pb, x);
#pragma unroll
      for (int j = 0; j < 4; ++j) amax = fmaxf(amax, fabsf(x[j]));
    }
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

// MX-plane mode (MegaMoE fp8_block expert cache, selected per tensor by a null scale_inv
// address): the exponent of Turbo's quant_w_128x128 (quant_blk_gfx1250 _ue8m0_exp ->
// ue8m0_exp_fly(amax, 8)): smallest e with amax <= 448 * 2^e from the amax bits, 0 for
// amax == 0 (no 1e-4 floor), inf/nan -> 126, clamped to [-126, 126]; q = e4m3(w * 2^-e).
__host__ __device__ __forceinline__ int fp8blk_mx_exp(float amax) {
  const uint32_t bits = fp8blk_f32_bits(amax);
  const int fld = static_cast<int>((bits >> 23) & 0xFF);
  int e = fld - (127 + 8) + ((bits & UINT32_C(0x7FFFFF)) > UINT32_C(0x600000) ? 1 : 0);
  if (fld == 0xFF) e = 126;
  if (!(amax > 0.0f)) e = 0;
  return e < -126 ? -126 : (e > 126 ? 126 : e);
}

__host__ __device__ __forceinline__ float fp8blk_pow2_neg(int e) {
  const uint32_t u = static_cast<uint32_t>(127 - e) << 23;
  float r;
  memcpy(&r, &u, 4);
  return r;
}

// MX-plane scale store: S [B, N, K/32] and ST [B, K, N/32] uint8, the tile's E8M0 byte
// (e + 127) repeated 4x per 128 columns, i.e. one int32 word per tile row (S) and per tile
// column (ST), as Turbo's _w_128x128_kernel writes them. Full tiles only (N, K % 128 == 0).
__host__ __device__ __forceinline__ void fp8blk_mx_scale_store(
    int tid, const Fp8BlkTile& t, int e, uint8_t* s, uint8_t* st) {
  const uint32_t word = static_cast<uint32_t>((e + 127) & 0xFF) * UINT32_C(0x01010101);
  if (tid < kTile) {
    uint32_t* s32 = reinterpret_cast<uint32_t*>(s);
    s32[(t.b * t.N + t.tm * kTile + tid) * t.tiles_k + t.tk] = word;
  } else if (tid < 2 * kTile && st != nullptr) {
    uint32_t* st32 = reinterpret_cast<uint32_t*>(st);
    st32[(t.b * t.K + t.tk * kTile + (tid - kTile)) * t.tiles_n + t.tm] = word;
  }
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
        if (ptr.model_param != nullptr) ptr.model_param[i + j] = pb;
      }
      xs16[row * kTile + col + j] = pb.x;
      amax = fmaxf(amax, fabsf(static_cast<float>(pb)));
    }
  }
  return amax;
}

// Phase 2 from the parked BF16 weights (same bytes as fp8blk_quant_phase): store W (full
// tiles: uniform origin + 32-bit offsets; else edge-checked) and, if W^T is wanted, stage the
// FP8 tile in LDS for the transpose.
template <bool kFull>
__host__ __device__ __forceinline__ void fp8blk_quant_lds(
    int tid, const Fp8BlkTile& t, uint8_t* w, float s, const uint4* xs, uint8_t* lds,
    bool stage) {
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
    if (stage) *reinterpret_cast<uint2*>(lds + row * kTilePitch + col) = bytes;
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
  if (e8m0_t != nullptr) e8m0_t[(t.b * t.tiles_k + t.tk) * t.tiles_n + t.tm] = e;
}

#ifndef PRIMUS_FP8BLK_HOST_TEST
// Block barrier that orders LDS only. The tile kernel shares nothing through global memory inside a block (every
// global location is written by exactly one thread and never read back), so the barrier does not need the global
// stores (master, moments, W) to drain first, which __syncthreads()' workgroup fence would wait for.
__device__ __forceinline__ void fp8blk_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
#endif

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
  if (full && ptr.model_param != nullptr) {
    amax = fp8blk_adam_full<AdamW, MomentT, GradT, true>(
        threadIdx.x, t, ptr, beta1, beta2, step_size, beta2_corr_inv, epsilon, lr, decay,
        grad_scale, xs);
  } else if (full) {
    amax = fp8blk_adam_full<AdamW, MomentT, GradT, false>(
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
  fp8blk_lds_barrier();
  // Every thread reduces the 16 wave maxima itself (one barrier instead of two).
  float tile_amax = wave_amax[0];
#pragma unroll
  for (int w = 1; w < kBlockSize / 32; ++w) tile_amax = fmaxf(tile_amax, wave_amax[w]);
  // A null scale_inv address (per tensor, block-uniform) selects the MX-plane mode.
  const bool mx = ptr.scale_inv == nullptr;
  const int mx_e = fp8blk_mx_exp(tile_amax);
  const float s = mx ? fp8blk_pow2_neg(mx_e) : fp8blk_scale(tile_amax);
  // A null W^T address (per tensor) skips the transpose phase: e.g. weights whose GEMMs only
  // read W (grouped expert GEMMs take W with the transpose flag in dgrad).
  const bool want_t = ptr.wt != nullptr;  // block-uniform
  if (full) {
    fp8blk_quant_lds<true>(threadIdx.x, t, ptr.w, s, xs, lds, want_t);
  } else {
    fp8blk_quant_lds<false>(threadIdx.x, t, ptr.w, s, xs, lds, want_t);
  }
  if (want_t) {
    fp8blk_lds_barrier();
    if (full) {
      fp8blk_transpose_full(threadIdx.x, t, ptr.wt, lds);
    } else {
      fp8blk_transpose_phase(threadIdx.x, t, ptr.wt, lds);
    }
  }
  if (mx) {
    fp8blk_mx_scale_store(threadIdx.x, t, mx_e, ptr.e8m0, ptr.e8m0_t);
  } else if (threadIdx.x == 0) {
    fp8blk_scale_store(t, s, ptr.scale_inv, ptr.e8m0, ptr.e8m0_t);
  }
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
// W^T fp8, scale_inv fp32, e8m0, e8m0_t (W^T and e8m0_t may be 0: not written; the bf16 param
// may be 0: not written, for a weight whose only readers take the FP8 outputs). scale_inv 0
// selects the MX-plane mode (MegaMoE expert cache: e8m0 -> S [B,N,K/32], e8m0_t -> ST
// [B,K,N/32], quant_w_128x128 exponent; N, K % 128 == 0 checked by the caller); dims[4] =
// N, K, tiles_n, tiles_k (B is implied by the tile count). Scalars are derived exactly as in
// fused_adam_clip.
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

/*****************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ****************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
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

  // FP32 moments with BF16 gradients (the single-micro-batch direct-gradient path): lane-contiguous
  // accesses. Bitwise equal to the loop below; the FP32-gradient instantiation is not (its codegen
  // contracts differently), so it keeps the original loop.
  if constexpr (std::is_same_v<MomentT, float> && !std::is_same_v<GradT, float>) {
    // Lane-contiguous layout: per step a block covers kBlockSize*8 elements as two
    // halves of kBlockSize*4; thread t owns elements [4t, 4t+4) of each half, so every
    // 16-byte FP32 access and 8-byte BF16 access is contiguous across the wave.
    const int step = blockDim.x * 8;
    for (int base = 0; base < n_this; base += step) {
  #pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int i = base + h * blockDim.x * 4 + threadIdx.x * 4;
        if (i + 4 <= n_this) {
          float g[4], p[4], m[4], v[4];
          const float4 p0 = *reinterpret_cast<const float4*>(param + i);
          const float4 m0 = *reinterpret_cast<const float4*>(exp_avg + i);
          const float4 v0 = *reinterpret_cast<const float4*>(exp_avg_sq + i);
          if constexpr (std::is_same_v<GradT, float>) {
            const float4 g0 = *reinterpret_cast<const float4*>(grad + i);
            g[0] = g0.x; g[1] = g0.y; g[2] = g0.z; g[3] = g0.w;
          } else {
            const uint2 pg = *reinterpret_cast<const uint2*>(grad + i);
            g[0] = c10::detail::f32_from_bits(static_cast<uint16_t>(pg.x & 0xffffu));
            g[1] = c10::detail::f32_from_bits(static_cast<uint16_t>(pg.x >> 16));
            g[2] = c10::detail::f32_from_bits(static_cast<uint16_t>(pg.y & 0xffffu));
            g[3] = c10::detail::f32_from_bits(static_cast<uint16_t>(pg.y >> 16));
          }
          p[0] = p0.x; p[1] = p0.y; p[2] = p0.z; p[3] = p0.w;
          m[0] = m0.x; m[1] = m0.y; m[2] = m0.z; m[3] = m0.w;
          v[0] = v0.x; v[1] = v0.y; v[2] = v0.z; v[3] = v0.w;
  #pragma unroll
          for (int j = 0; j < 4; ++j) {
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
          *reinterpret_cast<float4*>(param + i) = {p[0], p[1], p[2], p[3]};
          *reinterpret_cast<float4*>(exp_avg + i) = {m[0], m[1], m[2], m[3]};
          *reinterpret_cast<float4*>(exp_avg_sq + i) = {v[0], v[1], v[2], v[3]};
          if (model_param != nullptr &&
              (reinterpret_cast<uintptr_t>(model_param + i) & 7) == 0) {
            c10::BFloat16 b[4];
  #pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = static_cast<c10::BFloat16>(p[j]);
            uint2 pb;
            pb.x = static_cast<uint32_t>(b[0].x) | (static_cast<uint32_t>(b[1].x) << 16);
            pb.y = static_cast<uint32_t>(b[2].x) | (static_cast<uint32_t>(b[3].x) << 16);
            *reinterpret_cast<uint2*>(model_param + i) = pb;
          } else if (model_param != nullptr) {
  #pragma unroll
            for (int j = 0; j < 4; ++j) model_param[i + j] = static_cast<c10::BFloat16>(p[j]);
          }
        } else {
          for (int j = 0; j < 4 && i + j < n_this; ++j) {
            float gj = grad[i + j], pj = param[i + j], mj = exp_avg[i + j], vj = exp_avg_sq[i + j];
            gj *= grad_scale;
            if constexpr (!AdamW) gj += decay * pj;
            mj = beta1 * mj + (1.0f - beta1) * gj;
            vj = beta2 * vj + (1.0f - beta2) * gj * gj;
            const float denom = sqrtf(vj * beta2_corr_inv) + epsilon;
            if constexpr (AdamW) {
              pj = pj - step_size * (mj / denom) - lr * decay * pj;
            } else {
              pj -= step_size * (mj / denom);
            }
            param[i + j] = pj; exp_avg[i + j] = mj; exp_avg_sq[i + j] = vj;
            if (model_param != nullptr) model_param[i + j] = static_cast<c10::BFloat16>(pj);
          }
        }
      }
    }
    return;
  }

  // BF16 moments: eight consecutive elements per thread.
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


// ---------------------------------------------------------------------------------------------
// Split FP32 master (PRIMUS_FUSED_ADAM_SPLIT_MASTER): the FP32 master x is held losslessly as
// the BF16 model parameter P = RNE(x) (which the step writes anyway) plus a 16-bit remainder,
//   bits(x) = (bits(P) << 16) + d,  d in [-0x8000, 0x8000],  r = (uint16)d,
// where d = +0x8000 (an exact tie that rounded down to an even P) is flagged in the sign bit of
// exp_avg_sq (v >= 0 always). The step reads P and r instead of x and writes r instead of x:
// 26 instead of 28 bytes per parameter. The Adam arithmetic is the same expression as above,
// on the same x, so P, m, v and the reconstructed x are bit-identical to the FP32-master path.
// addresses slot 1 holds the remainder (uint16) instead of the FP32 master.
__device__ __forceinline__ float split_join(uint32_t P, uint32_t R, uint32_t flag) {
  const int32_t d = flag ? 0x8000 : static_cast<int32_t>(static_cast<int16_t>(R));
  return __uint_as_float((P << 16) + static_cast<uint32_t>(d));
}

__device__ __forceinline__ void split_part(float x, uint32_t& P, uint32_t& R, uint32_t& flag) {
  const uint32_t xb = __float_as_uint(x);
  P = c10::detail::round_to_nearest_even(x);
  const int32_t d = static_cast<int32_t>(xb - (P << 16));
  R = static_cast<uint32_t>(d) & 0xffffu;
  flag = d == 0x8000 ? 1u : 0u;
}

template <bool AdamW>
__global__ __launch_bounds__(kBlockSize) void fused_adam_clip_split_kernel(
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
    float max_norm,
    const uint8_t* __restrict__ row_mask,
    int mask_tensor,
    int row_len) {
  const int global_chunk = blockIdx.x;
  if (global_chunk >= total_chunks) return;

  const int tensor_idx = block_to_tensor[global_chunk];
  const int chunk_idx = global_chunk - chunk_offsets[tensor_idx];
  // Persistent embedding gradient (exact_embedding_grad.ZERO_ROW_MASKS): rows whose flag is 0
  // hold exactly +0, so their gradient is not loaded (row_len % 4 == 0, < 2^31 elements).
  const uint8_t* __restrict__ rmask = (tensor_idx == mask_tensor) ? row_mask : nullptr;
  const uint16_t* __restrict__ grad = reinterpret_cast<const uint16_t*>(addresses[tensor_idx * 5 + 0]);
  uint16_t* __restrict__ rem = reinterpret_cast<uint16_t*>(addresses[tensor_idx * 5 + 1]);
  float* __restrict__ exp_avg = reinterpret_cast<float*>(addresses[tensor_idx * 5 + 2]);
  uint32_t* __restrict__ exp_avg_sq = reinterpret_cast<uint32_t*>(addresses[tensor_idx * 5 + 3]);
  uint16_t* __restrict__ model_param = reinterpret_cast<uint16_t*>(addresses[tensor_idx * 5 + 4]);

  const int64_t elem_offset = static_cast<int64_t>(chunk_idx) * chunk_size;
  grad += elem_offset;
  rem += elem_offset;
  exp_avg += elem_offset;
  exp_avg_sq += elem_offset;
  model_param += elem_offset;
  const int n_this = static_cast<int>(
      min(sizes[tensor_idx] - elem_offset, static_cast<int64_t>(chunk_size)));
  __shared__ float grad_scale_shared;
  if (threadIdx.x == 0) {
    const float clip_coeff = max_norm / (*grad_norm + 1.0e-6f);
    grad_scale_shared = clip_coeff < 1.0f ? clip_coeff : 1.0f;
  }
  __syncthreads();
  const float grad_scale = grad_scale_shared;
  const bool aligned = ((reinterpret_cast<uintptr_t>(model_param) | reinterpret_cast<uintptr_t>(rem) |
                         reinterpret_cast<uintptr_t>(grad)) & 7) == 0;

  const int step = blockDim.x * 8;
  for (int base = 0; base < n_this; base += step) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int i = base + h * blockDim.x * 4 + threadIdx.x * 4;
      if (aligned && i + 4 <= n_this) {
        float g[4], p[4], m[4], v[4];
        uint32_t P[4], R[4], V[4];
        const uint2 pb = *reinterpret_cast<const uint2*>(model_param + i);
        const uint2 rb = *reinterpret_cast<const uint2*>(rem + i);
        const float4 m0 = *reinterpret_cast<const float4*>(exp_avg + i);
        const uint4 vb = *reinterpret_cast<const uint4*>(exp_avg_sq + i);
        uint2 pg = {0u, 0u};
        if (rmask == nullptr ||
            rmask[static_cast<uint32_t>(elem_offset + i) / static_cast<uint32_t>(row_len)])
          pg = *reinterpret_cast<const uint2*>(grad + i);
        g[0] = c10::detail::f32_from_bits(static_cast<uint16_t>(pg.x & 0xffffu));
        g[1] = c10::detail::f32_from_bits(static_cast<uint16_t>(pg.x >> 16));
        g[2] = c10::detail::f32_from_bits(static_cast<uint16_t>(pg.y & 0xffffu));
        g[3] = c10::detail::f32_from_bits(static_cast<uint16_t>(pg.y >> 16));
        P[0] = pb.x & 0xffffu; P[1] = pb.x >> 16; P[2] = pb.y & 0xffffu; P[3] = pb.y >> 16;
        R[0] = rb.x & 0xffffu; R[1] = rb.x >> 16; R[2] = rb.y & 0xffffu; R[3] = rb.y >> 16;
        V[0] = vb.x; V[1] = vb.y; V[2] = vb.z; V[3] = vb.w;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          v[j] = __uint_as_float(V[j] & 0x7fffffffu);
          p[j] = split_join(P[j], R[j], V[j] >> 31);
        }
        m[0] = m0.x; m[1] = m0.y; m[2] = m0.z; m[3] = m0.w;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
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
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t fl;
          split_part(p[j], P[j], R[j], fl);
          V[j] = __float_as_uint(v[j]) | (fl << 31);
        }
        *reinterpret_cast<uint2*>(model_param + i) = {P[0] | (P[1] << 16), P[2] | (P[3] << 16)};
        *reinterpret_cast<uint2*>(rem + i) = {R[0] | (R[1] << 16), R[2] | (R[3] << 16)};
        *reinterpret_cast<float4*>(exp_avg + i) = {m[0], m[1], m[2], m[3]};
        *reinterpret_cast<uint4*>(exp_avg_sq + i) = {V[0], V[1], V[2], V[3]};
      } else {
        for (int j = 0; j < 4 && i + j < n_this; ++j) {
          const uint32_t Vj = exp_avg_sq[i + j];
          float gj = c10::detail::f32_from_bits(grad[i + j]);
          float pj = split_join(model_param[i + j], rem[i + j], Vj >> 31);
          float mj = exp_avg[i + j], vj = __uint_as_float(Vj & 0x7fffffffu);
          gj *= grad_scale;
          if constexpr (!AdamW) gj += decay * pj;
          mj = beta1 * mj + (1.0f - beta1) * gj;
          vj = beta2 * vj + (1.0f - beta2) * gj * gj;
          const float denom = sqrtf(vj * beta2_corr_inv) + epsilon;
          if constexpr (AdamW) {
            pj = pj - step_size * (mj / denom) - lr * decay * pj;
          } else {
            pj -= step_size * (mj / denom);
          }
          uint32_t Pj, Rj, fl;
          split_part(pj, Pj, Rj, fl);
          model_param[i + j] = static_cast<uint16_t>(Pj);
          rem[i + j] = static_cast<uint16_t>(Rj);
          exp_avg[i + j] = mj;
          exp_avg_sq[i + j] = __float_as_uint(vj) | (fl << 31);
        }
      }
    }
  }
}

// x (FP32 master) -> (P, r, v sign flag). P is also written: it equals RNE(x) by construction.
__global__ void split_master_kernel(const float* __restrict__ x, uint16_t* __restrict__ p, uint16_t* __restrict__ r,
                                    uint32_t* __restrict__ v, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    uint32_t P, R, fl;
    split_part(x[i], P, R, fl);
    p[i] = static_cast<uint16_t>(P);
    r[i] = static_cast<uint16_t>(R);
    v[i] = (v[i] & 0x7fffffffu) | (fl << 31);
  }
}

// (P, r, v sign flag) -> x, and clear the flags (v becomes plain FP32 again).
__global__ void join_master_kernel(float* __restrict__ x, const uint16_t* __restrict__ p, const uint16_t* __restrict__ r,
                                   uint32_t* __restrict__ v, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const uint32_t V = v[i];
    x[i] = split_join(p[i], r[i], V >> 31);
    v[i] = V & 0x7fffffffu;
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
    double max_norm,
    bool split_master,
    torch::Tensor row_mask,
    int64_t mask_tensor,
    int64_t row_len) {
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

  if (split_master) {
    TORCH_CHECK(moment_dtype == 0 && grad_dtype == 1,
                "fused_adam_clip split master needs FP32 moments and BF16 gradients");
#define LAUNCH_SPLIT(ADAMW) \
  fused_adam_clip_split_kernel<ADAMW><<<total_chunks, kBlockSize, 0, stream>>>( \
      chunk_size, addresses.data_ptr<int64_t>(), sizes.data_ptr<int64_t>(), \
      block_to_tensor.data_ptr<int>(), chunk_offsets.data_ptr<int>(), total_chunks, \
      static_cast<float>(beta1), static_cast<float>(beta2), step_size, \
      beta2_corr_inv, static_cast<float>(epsilon), static_cast<float>(lr), \
      static_cast<float>(weight_decay), grad_norm.data_ptr<float>(), \
      static_cast<float>(max_norm), mask_ptr, static_cast<int>(mask_idx), static_cast<int>(row_len))
    const uint8_t* mask_ptr = nullptr;
    int64_t mask_idx = -1;
    if (mask_tensor >= 0 && row_mask.defined() && row_mask.numel() > 0) {
      TORCH_CHECK(row_mask.is_cuda() && row_mask.scalar_type() == at::kByte && row_len > 0 && row_len % 4 == 0,
                  "fused_adam_clip row_mask: uint8 GPU tensor, row_len a positive multiple of 4");
      mask_ptr = row_mask.data_ptr<uint8_t>();
      mask_idx = mask_tensor;
    }
    if (mode == 1) {
      LAUNCH_SPLIT(true);
    } else {
      LAUNCH_SPLIT(false);
    }
#undef LAUNCH_SPLIT
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

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

void split_master(torch::Tensor x, torch::Tensor p, torch::Tensor r, torch::Tensor v, bool join) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && v.scalar_type() == at::kFloat &&
              p.scalar_type() == at::kBFloat16 && r.scalar_type() == at::kShort,
              "split_master: x/v FP32, p BF16, r int16 on GPU");
  TORCH_CHECK(x.is_contiguous() && p.is_contiguous() && r.is_contiguous() && v.is_contiguous(),
              "split_master: contiguous tensors");
  const int64_t n = x.numel();
  TORCH_CHECK(p.numel() == n && r.numel() == n && v.numel() == n, "split_master: sizes differ");
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 65536));
  auto* pp = reinterpret_cast<uint16_t*>(p.data_ptr());
  auto* rp = reinterpret_cast<uint16_t*>(r.data_ptr());
  auto* vp = reinterpret_cast<uint32_t*>(v.data_ptr());
  if (join) {
    join_master_kernel<<<blocks, 256, 0, stream>>>(x.data_ptr<float>(), pp, rp, vp, n);
  } else {
    split_master_kernel<<<blocks, 256, 0, stream>>>(x.data_ptr<float>(), pp, rp, vp, n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_adam_clip", &fused_adam_clip,
             "Fused Adam/AdamW with FP32 or BF16 moments and in-register clipping",
             pybind11::arg("addresses"), pybind11::arg("sizes"), pybind11::arg("block_to_tensor"),
             pybind11::arg("chunk_offsets"), pybind11::arg("total_chunks"), pybind11::arg("chunk_size"),
             pybind11::arg("lr"), pybind11::arg("beta1"), pybind11::arg("beta2"), pybind11::arg("epsilon"),
             pybind11::arg("step"), pybind11::arg("mode"), pybind11::arg("bias_correction"),
             pybind11::arg("weight_decay"), pybind11::arg("moment_dtype"), pybind11::arg("grad_dtype"),
             pybind11::arg("grad_norm"), pybind11::arg("max_norm"), pybind11::arg("split_master") = false,
             pybind11::arg("row_mask") = torch::Tensor(), pybind11::arg("mask_tensor") = -1,
             pybind11::arg("row_len") = 0);
  module.def("split_master", &split_master,
             "Split an FP32 master into BF16 + 16-bit remainder + exp_avg_sq sign flag (join=True: inverse)");
}

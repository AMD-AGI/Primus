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
  // Optimizer shards can be views at any element offset of a flat buffer; the 16-byte vector path needs every
  // operand 16-byte aligned (as TE's kernels check), else this chunk takes the per-element path.
  const bool aligned = ((reinterpret_cast<uintptr_t>(grad) | reinterpret_cast<uintptr_t>(param) |
                         reinterpret_cast<uintptr_t>(exp_avg) | reinterpret_cast<uintptr_t>(exp_avg_sq)) &
                        15) == 0;
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
    const bool vectorized = aligned && i + kIlp <= n_this;

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

// TE's store_param_remainders layout: a BF16 param holds the high 16 bits of the FP32 master weight and an int16
// remainder the low 16 bits; a negative remainder means the param was rounded up, so one comes off it first.
__device__ __forceinline__ float master_from_remainder(uint16_t param_bits, int16_t remainder) {
  const uint16_t high = remainder < 0 ? static_cast<uint16_t>(param_bits - 1) : param_bits;
  const uint32_t bits = (static_cast<uint32_t>(high) << 16) | static_cast<uint16_t>(remainder);
  return __uint_as_float(bits);
}

__device__ __forceinline__ void split_to_remainder(float master, uint16_t& param_bits, int16_t& remainder) {
  const uint32_t bits = __float_as_uint(master);
  remainder = static_cast<int16_t>(bits & UINT32_C(0xffff));
  const uint16_t high = static_cast<uint16_t>(bits >> 16);
  param_bits = remainder < 0 ? static_cast<uint16_t>(high + 1) : high;
}

template <bool AdamW, typename MomentT, typename GradT>
__global__ __launch_bounds__(kBlockSize) void fused_adam_clip_param_remainder_kernel(
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
  uint16_t* __restrict__ param =
      reinterpret_cast<uint16_t*>(addresses[tensor_idx * 5 + 1]);
  MomentT* __restrict__ exp_avg =
      reinterpret_cast<MomentT*>(addresses[tensor_idx * 5 + 2]);
  MomentT* __restrict__ exp_avg_sq =
      reinterpret_cast<MomentT*>(addresses[tensor_idx * 5 + 3]);
  int16_t* __restrict__ remainder =
      reinterpret_cast<int16_t*>(addresses[tensor_idx * 5 + 4]);

  const int64_t elem_offset = static_cast<int64_t>(chunk_idx) * chunk_size;
  grad += elem_offset;
  param += elem_offset;
  exp_avg += elem_offset;
  exp_avg_sq += elem_offset;
  remainder += elem_offset;
  const int n_this = static_cast<int>(
      min(sizes[tensor_idx] - elem_offset, static_cast<int64_t>(chunk_size)));
  // Distributed-optimizer shards of BF16 params and grads are views at any element offset of a flat buffer: the
  // 16-byte vector path needs every operand 16-byte aligned (as TE's kernels check), else the per-element path.
  const bool aligned = ((reinterpret_cast<uintptr_t>(grad) | reinterpret_cast<uintptr_t>(param) |
                         reinterpret_cast<uintptr_t>(exp_avg) | reinterpret_cast<uintptr_t>(exp_avg_sq) |
                         reinterpret_cast<uintptr_t>(remainder)) &
                        15) == 0;
  __shared__ float grad_scale_shared;
  if (threadIdx.x == 0) {
    const float clip_coeff = max_norm / (*grad_norm + 1.0e-6f);
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
    const bool vectorized = aligned && i + kIlp <= n_this;

    if (vectorized) {
      const uint4 packed_p = *reinterpret_cast<const uint4*>(param + i);
      const uint4 packed_r = *reinterpret_cast<const uint4*>(remainder + i);
      const uint32_t pw[kIlp / 2] = {packed_p.x, packed_p.y, packed_p.z, packed_p.w};
      const uint32_t rw[kIlp / 2] = {packed_r.x, packed_r.y, packed_r.z, packed_r.w};
#pragma unroll
      for (int j = 0; j < kIlp / 2; ++j) {
        p[j * 2] = master_from_remainder(static_cast<uint16_t>(pw[j] & UINT32_C(0xffff)),
                                         static_cast<int16_t>(rw[j] & UINT32_C(0xffff)));
        p[j * 2 + 1] = master_from_remainder(static_cast<uint16_t>(pw[j] >> 16),
                                             static_cast<int16_t>(rw[j] >> 16));
      }
      if constexpr (std::is_same_v<GradT, float>) {
        const float4 g0 = *reinterpret_cast<const float4*>(grad + i);
        const float4 g1 = *reinterpret_cast<const float4*>(grad + i + 4);
        g[0] = g0.x; g[1] = g0.y; g[2] = g0.z; g[3] = g0.w;
        g[4] = g1.x; g[5] = g1.y; g[6] = g1.z; g[7] = g1.w;
      } else {
        unpack_bf16x8(*reinterpret_cast<const uint4*>(grad + i), g);
      }
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
        unpack_bf16x8(*reinterpret_cast<const uint4*>(exp_avg + i), m);
        unpack_bf16x8(*reinterpret_cast<const uint4*>(exp_avg_sq + i), v);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kIlp; ++j) {
        if (i + j < n_this) {
          g[j] = static_cast<float>(grad[i + j]);
          p[j] = master_from_remainder(param[i + j], remainder[i + j]);
          m[j] = static_cast<float>(exp_avg[i + j]);
          v[j] = static_cast<float>(exp_avg_sq[i + j]);
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

    uint16_t pb[kIlp];
    int16_t rb[kIlp];
#pragma unroll
    for (int j = 0; j < kIlp; ++j) split_to_remainder(p[j], pb[j], rb[j]);

    if (vectorized) {
      uint32_t pw[kIlp / 2];
      uint32_t rw[kIlp / 2];
#pragma unroll
      for (int j = 0; j < kIlp / 2; ++j) {
        pw[j] = static_cast<uint32_t>(pb[j * 2]) | (static_cast<uint32_t>(pb[j * 2 + 1]) << 16);
        rw[j] = static_cast<uint32_t>(static_cast<uint16_t>(rb[j * 2])) |
            (static_cast<uint32_t>(static_cast<uint16_t>(rb[j * 2 + 1])) << 16);
      }
      *reinterpret_cast<uint4*>(param + i) = {pw[0], pw[1], pw[2], pw[3]};
      *reinterpret_cast<uint4*>(remainder + i) = {rw[0], rw[1], rw[2], rw[3]};
      if constexpr (std::is_same_v<MomentT, float>) {
        *reinterpret_cast<float4*>(exp_avg + i) = {m[0], m[1], m[2], m[3]};
        *reinterpret_cast<float4*>(exp_avg + i + 4) = {m[4], m[5], m[6], m[7]};
        *reinterpret_cast<float4*>(exp_avg_sq + i) = {v[0], v[1], v[2], v[3]};
        *reinterpret_cast<float4*>(exp_avg_sq + i + 4) = {v[4], v[5], v[6], v[7]};
      } else {
        *reinterpret_cast<uint4*>(exp_avg + i) = pack_bf16x8(m);
        *reinterpret_cast<uint4*>(exp_avg_sq + i) = pack_bf16x8(v);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kIlp; ++j) {
        if (i + j < n_this) {
          param[i + j] = pb[j];
          remainder[i + j] = rb[j];
          exp_avg[i + j] = static_cast<MomentT>(m[j]);
          exp_avg_sq[i + j] = static_cast<MomentT>(v[j]);
        }
      }
    }
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
    int param_remainder) {
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
  TORCH_CHECK(param_remainder == 0 || param_remainder == 1,
              "fused_adam_clip param_remainder must be 0 (FP32 params) or 1 (BF16 params + int16 remainders)");

  float correction1 = 1.0f;
  float correction2 = 1.0f;
  if (bias_correction == 1) {
    correction1 = 1.0f - std::pow(static_cast<float>(beta1), step);
    correction2 = 1.0f - std::pow(static_cast<float>(beta2), step);
  }
  const float step_size = static_cast<float>(lr) / correction1;
  const float beta2_corr_inv = 1.0f / correction2;
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_KERNEL(KERNEL, ADAMW, MOMENT_T, GRAD_T) \
  KERNEL<ADAMW, MOMENT_T, GRAD_T> \
      <<<total_chunks, kBlockSize, 0, stream>>>( \
      chunk_size, addresses.data_ptr<int64_t>(), sizes.data_ptr<int64_t>(), \
      block_to_tensor.data_ptr<int>(), chunk_offsets.data_ptr<int>(), total_chunks, \
      static_cast<float>(beta1), static_cast<float>(beta2), step_size, \
      beta2_corr_inv, static_cast<float>(epsilon), static_cast<float>(lr), \
      static_cast<float>(weight_decay), grad_norm.data_ptr<float>(), \
      static_cast<float>(max_norm))
#define LAUNCH(ADAMW, MOMENT_T, GRAD_T) \
  if (param_remainder == 1) { \
    LAUNCH_KERNEL(fused_adam_clip_param_remainder_kernel, ADAMW, MOMENT_T, GRAD_T); \
  } else { \
    LAUNCH_KERNEL(fused_adam_clip_kernel, ADAMW, MOMENT_T, GRAD_T); \
  }
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
#undef LAUNCH_KERNEL
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_adam_clip", &fused_adam_clip,
             "Fused Adam/AdamW with FP32 or BF16 moments and in-register clipping; FP32 params, or BF16 params "
             "with int16 remainders (TE store_param_remainders)");
}

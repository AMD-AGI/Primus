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

namespace {

constexpr int kBlockSize = 512;
constexpr int kIlp = 8;

template <bool AdamW>
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
  float* __restrict__ grad =
      reinterpret_cast<float*>(addresses[tensor_idx * 4 + 0]);
  float* __restrict__ param =
      reinterpret_cast<float*>(addresses[tensor_idx * 4 + 1]);
  float* __restrict__ exp_avg =
      reinterpret_cast<float*>(addresses[tensor_idx * 4 + 2]);
  float* __restrict__ exp_avg_sq =
      reinterpret_cast<float*>(addresses[tensor_idx * 4 + 3]);

  const int64_t elem_offset = static_cast<int64_t>(chunk_idx) * chunk_size;
  grad += elem_offset;
  param += elem_offset;
  exp_avg += elem_offset;
  exp_avg_sq += elem_offset;
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
      const float4 g0 = *reinterpret_cast<const float4*>(grad + i);
      const float4 g1 = *reinterpret_cast<const float4*>(grad + i + 4);
      const float4 p0 = *reinterpret_cast<const float4*>(param + i);
      const float4 p1 = *reinterpret_cast<const float4*>(param + i + 4);
      const float4 m0 = *reinterpret_cast<const float4*>(exp_avg + i);
      const float4 m1 = *reinterpret_cast<const float4*>(exp_avg + i + 4);
      const float4 v0 = *reinterpret_cast<const float4*>(exp_avg_sq + i);
      const float4 v1 = *reinterpret_cast<const float4*>(exp_avg_sq + i + 4);
      g[0] = g0.x; g[1] = g0.y; g[2] = g0.z; g[3] = g0.w;
      g[4] = g1.x; g[5] = g1.y; g[6] = g1.z; g[7] = g1.w;
      p[0] = p0.x; p[1] = p0.y; p[2] = p0.z; p[3] = p0.w;
      p[4] = p1.x; p[5] = p1.y; p[6] = p1.z; p[7] = p1.w;
      m[0] = m0.x; m[1] = m0.y; m[2] = m0.z; m[3] = m0.w;
      m[4] = m1.x; m[5] = m1.y; m[6] = m1.z; m[7] = m1.w;
      v[0] = v0.x; v[1] = v0.y; v[2] = v0.z; v[3] = v0.w;
      v[4] = v1.x; v[5] = v1.y; v[6] = v1.z; v[7] = v1.w;
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
      *reinterpret_cast<float4*>(exp_avg + i) = {m[0], m[1], m[2], m[3]};
      *reinterpret_cast<float4*>(exp_avg + i + 4) = {m[4], m[5], m[6], m[7]};
      *reinterpret_cast<float4*>(exp_avg_sq + i) = {v[0], v[1], v[2], v[3]};
      *reinterpret_cast<float4*>(exp_avg_sq + i + 4) = {v[4], v[5], v[6], v[7]};
    } else {
#pragma unroll
      for (int j = 0; j < kIlp; ++j) {
        if (i + j < n_this) {
          param[i + j] = p[j];
          exp_avg[i + j] = m[j];
          exp_avg_sq[i + j] = v[j];
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

  float correction1 = 1.0f;
  float correction2 = 1.0f;
  if (bias_correction == 1) {
    correction1 = 1.0f - std::pow(static_cast<float>(beta1), step);
    correction2 = 1.0f - std::pow(static_cast<float>(beta2), step);
  }
  const float step_size = static_cast<float>(lr) / correction1;
  const float beta2_corr_inv = 1.0f / correction2;
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(ADAMW) \
  fused_adam_clip_kernel<ADAMW><<<total_chunks, kBlockSize, 0, stream>>>( \
      chunk_size, addresses.data_ptr<int64_t>(), sizes.data_ptr<int64_t>(), \
      block_to_tensor.data_ptr<int>(), chunk_offsets.data_ptr<int>(), total_chunks, \
      static_cast<float>(beta1), static_cast<float>(beta2), step_size, \
      beta2_corr_inv, static_cast<float>(epsilon), static_cast<float>(lr), \
      static_cast<float>(weight_decay), grad_norm.data_ptr<float>(), \
      static_cast<float>(max_norm))
  if (mode == 1) {
    LAUNCH(true);
  } else {
    LAUNCH(false);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("fused_adam_clip", &fused_adam_clip,
             "Fused FP32 Adam/AdamW with an in-register gradient clip factor");
}

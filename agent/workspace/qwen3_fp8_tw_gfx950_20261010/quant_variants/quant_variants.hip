// Standalone A/B harness for tensorwise FP8 quantize (no-pad case).
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

using u32x4 = __attribute__((vector_size(16))) uint32_t;
using u32x2 = __attribute__((vector_size(8))) uint32_t;

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

template <bool E5M2, bool WORD_SEL>
__device__ __forceinline__ uint32_t cvt_pk(float a, float b, uint32_t acc) {
    if constexpr (E5M2) {
        return static_cast<uint32_t>(__builtin_amdgcn_cvt_pk_bf8_f32(a, b, static_cast<int>(acc), WORD_SEL));
    } else {
        return static_cast<uint32_t>(__builtin_amdgcn_cvt_pk_fp8_f32(a, b, static_cast<int>(acc), WORD_SEL));
    }
}

template <bool E5M2>
__device__ __forceinline__ float qclip(float x, float scale) {
    constexpr float qmax = E5M2 ? 57344.0f : 448.0f;
    const float v = x * scale;
    return fmaxf(fminf(v, qmax), -qmax);
}

// 8 bf16 (one u32x4) -> 2 dwords of fp8
template <bool E5M2>
__device__ __forceinline__ void quant8(const u32x4 v, float scale, uint32_t &w0, uint32_t &w1) {
    float f[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i]     = bf16_to_f32(static_cast<uint16_t>(v[i] & 0xffffu));
        f[2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(v[i] >> 16));
    }
    uint32_t lo = cvt_pk<E5M2, false>(qclip<E5M2>(f[0], scale), qclip<E5M2>(f[1], scale), 0u);
    w0          = cvt_pk<E5M2, true>(qclip<E5M2>(f[2], scale), qclip<E5M2>(f[3], scale), lo);
    lo          = cvt_pk<E5M2, false>(qclip<E5M2>(f[4], scale), qclip<E5M2>(f[5], scale), 0u);
    w1          = cvt_pk<E5M2, true>(qclip<E5M2>(f[6], scale), qclip<E5M2>(f[7], scale), lo);
}

// ---- V0: copy of the current row-per-block kernel (UNROLL=8, BLOCK=256, no pad) ----
template <bool E5M2>
__launch_bounds__(256) __global__ void v0_row_kernel(const uint16_t *__restrict__ x, uint8_t *__restrict__ y,
                                                     const float *scale_ptr, int32_t N, int32_t K) {
    const float   scale         = scale_ptr[0];
    const int32_t packs_per_row = K / 8;
    for (int32_t r = blockIdx.x; r < N; r += gridDim.x) {
        const uint16_t *xr = x + static_cast<int64_t>(r) * K;
        uint8_t        *yr = y + static_cast<int64_t>(r) * K;
        for (int32_t p = threadIdx.x; p < packs_per_row; p += 256) {
            const u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4 *>(xr + p * 8));
            uint32_t    w0, w1;
            quant8<E5M2>(v, scale, w0, w1);
            *reinterpret_cast<u32x2 *>(yr + p * 8) = u32x2{w0, w1};
        }
    }
}

// ---- V1/V2/V3: flat kernel, PACKS x 16 elements per thread, packs strided by BLOCK ----
template <int BLOCK, int PACKS, bool NT, bool E5M2>
__launch_bounds__(BLOCK) __global__ void flat16_kernel(const uint16_t *__restrict__ x, uint8_t *__restrict__ y,
                                                       const float *scale_ptr, int64_t n16) {
    const float   scale = scale_ptr[0];
    const int64_t base  = static_cast<int64_t>(blockIdx.x) * BLOCK * PACKS + threadIdx.x;
    u32x4         a[PACKS], b[PACKS];
#pragma unroll
    for (int u = 0; u < PACKS; ++u) {
        const int64_t i = base + static_cast<int64_t>(u) * BLOCK;
        if (i < n16) {
            const u32x4 *src = reinterpret_cast<const u32x4 *>(x + i * 16);
            if constexpr (NT) {
                a[u] = __builtin_nontemporal_load(src);
                b[u] = __builtin_nontemporal_load(src + 1);
            } else {
                a[u] = src[0];
                b[u] = src[1];
            }
        }
    }
#pragma unroll
    for (int u = 0; u < PACKS; ++u) {
        const int64_t i = base + static_cast<int64_t>(u) * BLOCK;
        if (i < n16) {
            uint32_t w0, w1, w2, w3;
            quant8<E5M2>(a[u], scale, w0, w1);
            quant8<E5M2>(b[u], scale, w2, w3);
            *reinterpret_cast<u32x4 *>(y + i * 16) = u32x4{w0, w1, w2, w3};
        }
    }
}

// ---- V4: grid-stride persistent flat kernel ----
template <int BLOCK, int PACKS, bool E5M2>
__launch_bounds__(BLOCK) __global__ void gs16_kernel(const uint16_t *__restrict__ x, uint8_t *__restrict__ y,
                                                     const float *scale_ptr, int64_t n16) {
    const float   scale  = scale_ptr[0];
    const int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK * PACKS;
    for (int64_t base = static_cast<int64_t>(blockIdx.x) * BLOCK * PACKS + threadIdx.x; base < n16;
         base += stride) {
        u32x4 a[PACKS], b[PACKS];
#pragma unroll
        for (int u = 0; u < PACKS; ++u) {
            const int64_t i = base + static_cast<int64_t>(u) * BLOCK;
            if (i < n16) {
                const u32x4 *src = reinterpret_cast<const u32x4 *>(x + i * 16);
                a[u]             = __builtin_nontemporal_load(src);
                b[u]             = __builtin_nontemporal_load(src + 1);
            }
        }
#pragma unroll
        for (int u = 0; u < PACKS; ++u) {
            const int64_t i = base + static_cast<int64_t>(u) * BLOCK;
            if (i < n16) {
                uint32_t w0, w1, w2, w3;
                quant8<E5M2>(a[u], scale, w0, w1);
                quant8<E5M2>(b[u], scale, w2, w3);
                *reinterpret_cast<u32x4 *>(y + i * 16) = u32x4{w0, w1, w2, w3};
            }
        }
    }
}

template <bool E5M2>
void launch(int variant, const uint16_t *x, uint8_t *y, const float *s, int64_t N, int64_t K, hipStream_t st) {
    const int64_t n16 = N * K / 16;
    auto blocks       = [&](int64_t per) { return static_cast<unsigned>((n16 + per - 1) / per); };
    const unsigned g0 = static_cast<unsigned>(std::min<int64_t>(N, 1048576));
    switch (variant) {
    case 0: v0_row_kernel<E5M2><<<g0, 256, 0, st>>>(x, y, s, N, K); break;
    case 1: flat16_kernel<512, 1, false, E5M2><<<blocks(512), 512, 0, st>>>(x, y, s, n16); break;
    case 2: flat16_kernel<512, 1, true, E5M2><<<blocks(512), 512, 0, st>>>(x, y, s, n16); break;
    case 3: flat16_kernel<256, 2, true, E5M2><<<blocks(512), 256, 0, st>>>(x, y, s, n16); break;
    case 4: flat16_kernel<256, 4, true, E5M2><<<blocks(1024), 256, 0, st>>>(x, y, s, n16); break;
    case 5: flat16_kernel<256, 1, true, E5M2><<<blocks(256), 256, 0, st>>>(x, y, s, n16); break;
    case 6: gs16_kernel<256, 2, E5M2><<<256 * 8, 256, 0, st>>>(x, y, s, n16); break;
    case 7: gs16_kernel<512, 2, E5M2><<<256 * 4, 512, 0, st>>>(x, y, s, n16); break;
    case 8: flat16_kernel<1024, 1, true, E5M2><<<blocks(1024), 1024, 0, st>>>(x, y, s, n16); break;
    default: TORCH_CHECK(false, "bad variant");
    }
}

void run(int64_t variant, torch::Tensor x, torch::Tensor scale, torch::Tensor y) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2);
    TORCH_CHECK(x.size(1) % 16 == 0);
    auto st      = at::hip::getCurrentHIPStream();
    const bool e5 = y.scalar_type() == at::kFloat8_e5m2;
    auto *xp     = reinterpret_cast<const uint16_t *>(x.data_ptr());
    auto *yp     = reinterpret_cast<uint8_t *>(y.data_ptr());
    if (e5)
        launch<true>(variant, xp, yp, scale.data_ptr<float>(), x.size(0), x.size(1), st);
    else
        launch<false>(variant, xp, yp, scale.data_ptr<float>(), x.size(0), x.size(1), st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }

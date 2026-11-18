// Weighted LayerNorm forward producer for the MXFP6 forward K512 GEMMs (HSTU U / VQK / VK), behind
// DLRM_HSTU_FP8_FWD_MXFP6 (default off; loader generative_recommenders/ops/hip/ln_store_mx6.py). One launch per call:
//
//   x    bf16 [M, 512] row-major (contiguous)          w, b  bf16 [512] (LN weight / bias)
//   y    = fma((x - mean) * rstd, w, b) * inv_scale    fp32, the exact op sequence of the shipped Triton store_t v2
//   mean fp32 [M], rstd fp32 [M]                       bitwise target: _weighted_layer_norm_fwd_fp8_store_t_v2
//   xT8  e4m3 step-tiled [Mp/128, 512, 128]            byte (r/128)*65536 + c*128 + r%128; zero for M <= r < Mp
//   x8   e4m3 [M, 512] row-major (optional)            bitwise target: the Triton kernel's y output
//   x6   MXFP6 E2M3 "MX6-T" [Mp/128, 4, 12288] uint8   pre-tiled LDS image of pr7's k512 MX6 kernel (DESIGN.md s.9)
//   xs   E8M0 "MX6-A-v1" [Mp/128, 4, 512] uint8        byte (T, S, q*16 + r, m) = scale of row 128T+16m+r, K block 4S+q
//   yd   fp32 [M, 512] (debug, optional)               the fp32 y that x8 / xT8 / x6 / xs quantize
//
// Numerics (r14 iteration 49, read from the Triton 3.6.0 AMDGCN of the shipped kernel, see RESULT.md):
//   * lane layout of the statistics: 16 consecutive columns per lane, 32 lanes per row ([1,16] x [2,32]); in-lane sum
//     c0+c1+...+c15 (sequential), then a butterfly over the 32 lanes with xor 16, 8, 4, 2, 1; mean = sum / 512 (IEEE);
//     xm = x - mean; in-lane variance q = xm1*xm1, q = fma(xm0, xm0, q), q = fma(xm2, xm2, q), ..., fma(xm15, xm15, q),
//     same butterfly; var = q / 512; t = var + eps; rstd = t < FLT_MIN ? rsq(t * 2^24) * 2^12 : rsq(t) (OCML rsqrt;
//     the scaling branch is skipped uniformly when eps >= FLT_MIN, where it cannot trigger).
//   * y = fma(xm * rstd, w, b) * inv_scale (xm * rstd rounded, one FMA, one multiply);
//     e4m3 = v_cvt_scalef32_pk_fp8_f32(y, scale 1.0) under MODE.FP16_OVFL (saturating to +-448, RNE).
//   * MXFP6 (pr7 proto.py reference): per (row, 32-column block) amax of |y| -> E8M0 byte by the ceil rule
//     (byte = E - 2 + (mantissa > 0x700000), 0 if negative, 127 for an all-zero block, never 255);
//     codes = E2M3 RNE + saturation of y / 2^e (hardware v_cvt_scalef32_2xpk16_fp6_f32 on y * 2^-e, which is exact);
//     negative-zero codes (0x20) are rewritten to +0 (proto's encoder); pad rows: zero codes, byte 127.
//
// Geometry: workgroup = 4 waves = 32 rows (BLOCK_N and XCD-grouped program -> block map of v26.3's Triton v2 kernel);
// lane (wave w, half h, j = lane & 31) owns rows 4(2w+h) .. +3 and columns 16j .. 16j+15. Per lane: 8 x 16-B loads,
// the statistics (scalar chains; the cross-lane steps as ds_swizzle / DPP-folded adds), y (packed v_pk_*_f32), one e4m3
// dword per column holding its 4 rows (= the xT8 bytes, no transpose arithmetic), one fp6 convert per 2 rows (16 values
// each, pre-scaled). xT8 (17 KiB incl. pad), x6 (12.25 KiB) and xs (512 B) are staged in LDS with immediate-offset
// addressing and copied out to their global byte order with 16-byte (xs: 2-byte) stores after one barrier, so every x6
// and xT8 write is a whole 16-B chunk. Blocks holding rows >= M take a separate (PAD) code path. 64-bit row offsets.
// Build: hipcc --offload-arch=gfx950 -O3 -ffp-contract=off -fno-slp-vectorize (ln_store_mx6.py hipcc_command).
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace lnmx6 {

typedef float v16f __attribute__((ext_vector_type(16)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef uint32_t v6u __attribute__((ext_vector_type(6)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));
typedef short v2s __attribute__((ext_vector_type(2)));

constexpr int D = 512;
constexpr int BN = 32;                         // rows per workgroup (one block)
constexpr int NT = 256;                        // threads per workgroup
constexpr int XCD_GROUP = 8;                   // v26.3's _LN_FWD_STORE_T_V2_XCD_GROUP (this tree's Triton v2: 4)
constexpr int XT_OFF = 0;                      // 32 column groups x (16 columns x 32 rows + 32 B pad), see below
constexpr int XT_GSTR = 17 * 32;               // column group stride (bytes)
constexpr int XT_BYTES = 32 * XT_GSTR;         // 17408
constexpr int X6_SSTR = 32 * 96 + 64;          // per K step: 32 rows x [64 B head | 32 B tail], 64 B pad (bank shift)
constexpr int X6_OFF = XT_OFF + XT_BYTES;
constexpr int X6_BYTES = 4 * X6_SSTR;          // 12544
constexpr int XS_OFF = X6_OFF + X6_BYTES;      // [16 blocks][16 r][2 m] bytes
constexpr int XS_BYTES = 512;
constexpr int LDS_BYTES = XS_OFF + XS_BYTES;   // 30464
// low-5-bit masks of the 6-bit fields of a 96-bit (16-field) group: bit p set iff p % 6 != 5
constexpr uint32_t LOW0 = 0xdf7df7dfu, LOW1 = 0xf7df7df7u, LOW2 = 0x7df7df7du;

__device__ __forceinline__ float lo_f(uint32_t d) { return __uint_as_float(d << 16); }
__device__ __forceinline__ float hi_f(uint32_t d) { return __uint_as_float(d & 0xffff0000u); }

// cross-lane moves of the Triton reduction (data movement only; the adds are commutative)
__device__ __forceinline__ float xor16(float v) {  // ds_swizzle SWAP,16 within 32 lanes
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x401F));
}
__device__ __forceinline__ float xor8(float v) {   // row_ror:8 inside a 16-lane row
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x128, 0xF, 0xF, true));
}
__device__ __forceinline__ float xor4(float v) {   // ds_swizzle xor 4 within 32 lanes
    return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), 0x101F));
}
__device__ __forceinline__ float xor2(float v) {   // quad_perm [2,3,0,1]
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, true));
}
__device__ __forceinline__ float xor1(float v) {   // quad_perm [1,0,3,2]
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, true));
}
__device__ __forceinline__ uint32_t xor1u(uint32_t v) {
    return (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0xB1, 0xF, 0xF, true);
}
// (bound_ctrl = true: every source lane of these patterns is valid, so it changes no value; it lets the compiler fold
// the DPP move into the add.)
__device__ __forceinline__ float butterfly(float s) {
    s = s + xor16(s);
    s = s + xor8(s);
    s = s + xor4(s);
    s = s + xor2(s);
    s = s + xor1(s);
    return s;
}

// e4m3 pair convert exactly as the Triton kernel: v_cvt_scalef32_pk_fp8_f32 with scale 1.0
__device__ __forceinline__ uint32_t fp8x4(float a, float b, float c, float d) {
    v2s z;
    z = __builtin_nondeterministic_value(z);  // high half overwritten below
    v2s lo = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(z, a, b, 1.0f, false);
    v2s hv = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(lo, c, d, 1.0f, true);
    return __builtin_bit_cast(uint32_t, hv);
}

// 32 pre-scaled values -> 6 dwords of E2M3 codes (field 2i <- e[i], 2i+1 <- o[i]).
// Hazards (r14 it49 review m8). LLVM schedules around an asm statement without knowing what it executes, so every wait
// state this convert may need is inside the asm string:
//   * leading s_nop 1 (2 wait states): the compiler writes the sources (e, o and the scale VGPR) with the VALU
//     instructions right before the asm. LLVM inserts no wait state there even for the builtin form (AMD clang 22,
//     r14 it48 mxg.objdump.txt:100-106; AMD clang 23 it49 fix/cpu_isa/hz), so its gfx950 model has no entry for this
//     source path -- but the same model misses this multi-pass, 6-dword-output instruction's destination/scale overlap
//     hazard (r14 it48 acc/RESULT.md 1.4), so it is not relied on. 2 = the largest VALU-write -> VALU-read wait of the
//     gfx9 hazard table (DPP source), at ~2 cycles per convert;
//   * early-clobber "=&v": the destination must not overlap a source (r14 it48);
//   * trailing s_nop 4 (5 wait states) before any consumer of the 6 result dwords (r14 it48).
__device__ __forceinline__ v6u cvt_fp6(v16f e, v16f o) {
    v6u r;
    const float one = 1.0f;
    asm volatile("s_nop 1\n\tv_cvt_scalef32_2xpk16_fp6_f32 %0, %1, %2, %3\n\ts_nop 4"
                 : "=&v"(r)
                 : "v"(e), "v"(o), "v"(one));
    return r;
}

// negative-zero codes (sign set, magnitude 0) -> +0 on one 96-bit group: x & (LOW | ((x & LOW) + LOW))
__device__ __forceinline__ void fix_neg0(uint32_t& d0, uint32_t& d1, uint32_t& d2) {
    const uint64_t a01 = ((uint64_t)(d1 & LOW1) << 32) | (uint64_t)(d0 & LOW0);
    const uint64_t l01 = ((uint64_t)LOW1 << 32) | (uint64_t)LOW0;
    const uint64_t t01 = a01 + l01;
    const uint32_t carry = t01 < a01 ? 1u : 0u;
    const uint32_t t2 = (d2 & LOW2) + LOW2 + carry;
    d0 &= LOW0 | (uint32_t)t01;
    d1 &= LOW1 | (uint32_t)(t01 >> 32);
    d2 &= LOW2 | t2;
}

// max(m, |a|, |b|) in one instruction (m >= 0; finite inputs)
__device__ __forceinline__ float amax3(float m, float a, float b) {
    float r;
    asm("v_max3_f32 %0, %1, |%2|, |%3|" : "=v"(r) : "v"(m), "v"(a), "v"(b));
    return r;
}

// E8M0 byte of a non-negative fp32 amax (bit pattern): smallest e with amax <= 7.5 * 2^e, byte e + 127 >= 0
__device__ __forceinline__ uint32_t e8m0_byte(uint32_t a) {
    int b = (int)(a >> 23) - 2 + (int)((a & 0x7fffffu) > 0x700000u);
    b = b < 0 ? 0 : b;
    return a == 0u ? 127u : (uint32_t)b;
}

// xT LDS staging: column c = 16J + c' (J = column group, c' < 16), row quad g (0..7) -> byte
//   XT_GSTR * J + 32 * c' + 4 * (g ^ (2 * (J >> 3)))
// The 32-byte pad per column group and the row-quad XOR make the per-column dword writes (one wave instruction per c')
// and the 16-byte copy-out reads bank-conflict free, with additive (immediate-offset) addressing per column.

// One 32-row block. PAD: the block holds rows >= M (only the last blocks; uniform per workgroup) -> per-row validity.
// Per lane and row the 16 columns are kept as 8 float2 pairs in the order the fp6 convert takes them:
// E[k] = (column 4k, column 4k+2), O[k] = (column 4k+1, column 4k+3), k = 0..3. The elementwise math runs packed
// (v_pk_*_f32, exact per component); the two reductions run as scalar chains in column order 0, 1, 2, ..., 15.
template <bool X8, bool YD, bool PAD>
__device__ __forceinline__ void block_body(const uint16_t* __restrict__ x, const uint16_t* __restrict__ wt,
                                           const uint16_t* __restrict__ bs, float* __restrict__ mean_out,
                                           float* __restrict__ rstd_out, uint8_t* __restrict__ xt8,
                                           uint8_t* __restrict__ x6, uint8_t* __restrict__ xs,
                                           uint8_t* __restrict__ x8, float* __restrict__ yd, float eps,
                                           float inv_scale, int M, int blk, uint8_t* lds) {
    const int tid = threadIdx.x;
    const int lane = tid & 63, wv = tid >> 6;
    const int j = lane & 31;               // column group: columns 16j .. 16j+15
    const int g = 2 * wv + (lane >> 5);    // row quad: block rows 4g .. 4g+3
    const long long row0 = (long long)blk * BN + 4 * g;
    int nvalid = 4;                        // valid rows of this lane's quad (PAD only)
    if constexpr (PAD) {
        const long long r = (long long)M - row0;
        nvalid = r <= 0 ? 0 : (r >= 4 ? 4 : (int)r);
    }

    // ---- loads: 4 rows x 2 x 16 B (PAD: rows >= M read as zero, like the Triton masked load)
    const uint16_t* xp = x + row0 * D + 16 * j;
    v4u ld[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (!PAD || i < nvalid) {
            ld[i][0] = *reinterpret_cast<const v4u*>(xp + i * D);
            ld[i][1] = *reinterpret_cast<const v4u*>(xp + i * D + 8);
        } else {
            ld[i][0] = (v4u){0u, 0u, 0u, 0u};
            ld[i][1] = (v4u){0u, 0u, 0u, 0u};
        }
    }
    // dword m of a row = columns (2m, 2m+1): E[k] = (lo d[2k], lo d[2k+1]), O[k] = (hi d[2k], hi d[2k+1])
    v2f wE[4], wO[4], bE[4], bO[4];
    {
        const v4u w0 = *reinterpret_cast<const v4u*>(wt + 16 * j), w1 = *reinterpret_cast<const v4u*>(wt + 16 * j + 8);
        const v4u b0 = *reinterpret_cast<const v4u*>(bs + 16 * j), b1 = *reinterpret_cast<const v4u*>(bs + 16 * j + 8);
        const uint32_t wd[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const uint32_t bd[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            wE[k] = (v2f){lo_f(wd[2 * k]), lo_f(wd[2 * k + 1])};
            wO[k] = (v2f){hi_f(wd[2 * k]), hi_f(wd[2 * k + 1])};
            bE[k] = (v2f){lo_f(bd[2 * k]), lo_f(bd[2 * k + 1])};
            bO[k] = (v2f){hi_f(bd[2 * k]), hi_f(bd[2 * k + 1])};
        }
    }
    v2f E[4][4], O[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t d[8] = {ld[i][0].x, ld[i][0].y, ld[i][0].z, ld[i][0].w,
                               ld[i][1].x, ld[i][1].y, ld[i][1].z, ld[i][1].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            E[i][k] = (v2f){lo_f(d[2 * k]), lo_f(d[2 * k + 1])};
            O[i][k] = (v2f){hi_f(d[2 * k]), hi_f(d[2 * k + 1])};
        }
    }

    // ---- statistics (Triton op order; column 4k = E[k].x, 4k+1 = O[k].x, 4k+2 = E[k].y, 4k+3 = O[k].y)
    float mean[4], rstd[4], tv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = E[i][0].x + O[i][0].x;
        a = a + E[i][0].y;
        a = a + O[i][0].y;
#pragma unroll
        for (int k = 1; k < 4; ++k) {
            a = a + E[i][k].x;
            a = a + O[i][k].x;
            a = a + E[i][k].y;
            a = a + O[i][k].y;
        }
        mean[i] = butterfly(a) / 512.0f;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const v2f mu = (v2f){mean[i], mean[i]};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            E[i][k] = E[i][k] - mu;  // xm
            O[i][k] = O[i][k] - mu;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float a = O[i][0].x * O[i][0].x;               // column 1
        a = __builtin_fmaf(E[i][0].x, E[i][0].x, a);   // column 0
        a = __builtin_fmaf(E[i][0].y, E[i][0].y, a);   // column 2
        a = __builtin_fmaf(O[i][0].y, O[i][0].y, a);   // column 3
#pragma unroll
        for (int k = 1; k < 4; ++k) {
            a = __builtin_fmaf(E[i][k].x, E[i][k].x, a);
            a = __builtin_fmaf(O[i][k].x, O[i][k].x, a);
            a = __builtin_fmaf(E[i][k].y, E[i][k].y, a);
            a = __builtin_fmaf(O[i][k].y, O[i][k].y, a);
        }
        tv[i] = butterfly(a) / 512.0f + eps;  // var + eps
    }
    if (eps >= 1.17549435e-38f) {  // uniform: t >= eps >= FLT_MIN, OCML's denormal scaling cannot trigger
#pragma unroll
        for (int i = 0; i < 4; ++i) rstd[i] = __builtin_amdgcn_rsqf(tv[i]);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float t = tv[i];
            const bool sc = t < 1.17549435e-38f;  // OCML rsqrt: t < FLT_MIN ? rsq(t * 2^24) * 2^12 : rsq(t)
            const float r = __builtin_amdgcn_rsqf(sc ? t * 16777216.0f : t);
            rstd[i] = sc ? r * 4096.0f : r;
        }
    }
    if (j == 0) {
        if (!PAD || nvalid == 4) {
            *reinterpret_cast<float4*>(mean_out + row0) = make_float4(mean[0], mean[1], mean[2], mean[3]);
            *reinterpret_cast<float4*>(rstd_out + row0) = make_float4(rstd[0], rstd[1], rstd[2], rstd[3]);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (i < nvalid) {
                    mean_out[row0 + i] = mean[i];
                    rstd_out[row0 + i] = rstd[i];
                }
        }
    }

    // ---- y = fma(xm * rstd, w, b) * inv_scale (pad rows: 0)
    const v2f iv = (v2f){inv_scale, inv_scale};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const v2f rs = (v2f){rstd[i], rstd[i]};
        const bool valid = !PAD || i < nvalid;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const v2f ye = __builtin_elementwise_fma(E[i][k] * rs, wE[k], bE[k]) * iv;
            const v2f yo = __builtin_elementwise_fma(O[i][k] * rs, wO[k], bO[k]) * iv;
            E[i][k] = PAD ? (valid ? ye : (v2f){0.0f, 0.0f}) : ye;  // E / O now hold y
            O[i][k] = PAD ? (valid ? yo : (v2f){0.0f, 0.0f}) : yo;
        }
    }
    if constexpr (X8 || YD) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (!PAD || i < nvalid) {
                if constexpr (X8) {
                    v4u o;
                    o.x = fp8x4(E[i][0].x, O[i][0].x, E[i][0].y, O[i][0].y);
                    o.y = fp8x4(E[i][1].x, O[i][1].x, E[i][1].y, O[i][1].y);
                    o.z = fp8x4(E[i][2].x, O[i][2].x, E[i][2].y, O[i][2].y);
                    o.w = fp8x4(E[i][3].x, O[i][3].x, E[i][3].y, O[i][3].y);
                    *reinterpret_cast<v4u*>(x8 + (row0 + i) * D + 16 * j) = o;
                }
                if constexpr (YD) {
                    float4* p = reinterpret_cast<float4*>(yd + (row0 + i) * D + 16 * j);
#pragma unroll
                    for (int k = 0; k < 4; ++k) p[k] = make_float4(E[i][k].x, O[i][k].x, E[i][k].y, O[i][k].y);
                }
            }
        }
    }

    // ---- xT8: one dword per column = its 4 rows (rows 0,1 -> low half, 2,3 -> high half), straight to LDS
    uint8_t* const xtw = lds + XT_OFF + XT_GSTR * j + 4 * (g ^ (2 * (j >> 3)));
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
        for (int s4 = 0; s4 < 4; ++s4) {  // column 4k + s4
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = s4 == 0 ? E[i][k].x : (s4 == 1 ? O[i][k].x : (s4 == 2 ? E[i][k].y : O[i][k].y));
            v2s z;
            z = __builtin_nondeterministic_value(z);  // high half overwritten by the second convert
            const v2s lo = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(z, v[0], v[1], 1.0f, false);
            const v2s dw = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(lo, v[2], v[3], 1.0f, true);
            *reinterpret_cast<uint32_t*>(xtw + 32 * (4 * k + s4)) = __builtin_bit_cast(uint32_t, dw);
        }
    }

    // ---- MXFP6: per row, block amax over this lane's 16 + the partner lane's 16 values -> E8M0; rows in pairs,
    //      pre-scaled by 2^-e (exact), one convert per pair, negative-zero codes -> +0, then LDS staging
    const int bk = j >> 1, hh = j & 1;    // 32-column block 0..15, half of it
    const int S = bk >> 2, qb = bk & 3;   // K step, block within the step
    // x6 LDS staging: per K step, block row rb at 96*rb: head chunk q at 16*(q ^ ((rb>>2)&3)), tail pair q at
    // 64 + 8*(q ^ (2*((rb>>3)&1))); for this lane's rows rb = 4g + i: (rb>>2)&3 = g&3, (rb>>3)&1 = (g>>1)&1.
    // hh = 0 writes block bytes 0..11 (head 0..11), hh = 1 bytes 12..23 (head 12..15, tail 0..7); row i at +96*i.
    uint8_t* const x6r = lds + X6_OFF + S * X6_SSTR + 96 * 4 * g;
    uint8_t* const hd = x6r + 16 * (qb ^ (g & 3));
    uint8_t* const tl = x6r + 64 + 8 * (qb ^ (2 * ((g >> 1) & 1)));
    uint8_t* const x6w0 = hh ? hd + 12 : hd;
    uint8_t* const x6w1 = hh ? tl : hd + 4;
    uint8_t* const x6w2 = hh ? tl + 4 : hd + 8;
    // xs LDS staging [16 blocks][16 r][2 m]: row rb = 4g + i -> r = 4(g&3) + i, m = g >> 2
    uint8_t* const xsw = lds + XS_OFF + (bk * 16 + 4 * (g & 3)) * 2 + (g >> 2);
    uint32_t sb[4];
    v2f f2[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float m = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            m = amax3(m, E[i][k].x, E[i][k].y);
            m = amax3(m, O[i][k].x, O[i][k].y);
        }
        const uint32_t mu = __float_as_uint(m);
        const uint32_t mp = xor1u(mu);
        sb[i] = e8m0_byte(mu > mp ? mu : mp);
        const float f = __uint_as_float((254u - sb[i]) << 23);  // 2^-e, a normal float for every finite amax
        f2[i] = (v2f){f, f};
    }
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) {
        const int ia = 2 * pr, ib = 2 * pr + 1;
        v2f ea[4], eb[4], oa[4], ob[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            ea[k] = E[ia][k] * f2[ia];
            oa[k] = O[ia][k] * f2[ia];
            eb[k] = E[ib][k] * f2[ib];
            ob[k] = O[ib][k] * f2[ib];
        }
        const v16f ev = (v16f){ea[0].x, ea[0].y, ea[1].x, ea[1].y, ea[2].x, ea[2].y, ea[3].x, ea[3].y,
                               eb[0].x, eb[0].y, eb[1].x, eb[1].y, eb[2].x, eb[2].y, eb[3].x, eb[3].y};
        const v16f ov = (v16f){oa[0].x, oa[0].y, oa[1].x, oa[1].y, oa[2].x, oa[2].y, oa[3].x, oa[3].y,
                               ob[0].x, ob[0].y, ob[1].x, ob[1].y, ob[2].x, ob[2].y, ob[3].x, ob[3].y};
        const v6u cv = cvt_fp6(ev, ov);
        uint32_t c[6] = {cv[0], cv[1], cv[2], cv[3], cv[4], cv[5]};
        fix_neg0(c[0], c[1], c[2]);
        fix_neg0(c[3], c[4], c[5]);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int i = 2 * pr + u;  // block row 4g + i
            *reinterpret_cast<uint32_t*>(x6w0 + 96 * i) = c[3 * u];
            *reinterpret_cast<uint32_t*>(x6w1 + 96 * i) = c[3 * u + 1];
            *reinterpret_cast<uint32_t*>(x6w2 + 96 * i) = c[3 * u + 2];
            xsw[2 * i] = (uint8_t)sb[i];  // both lanes of the block write the same byte
        }
    }
    __syncthreads();

    // ---- copy-out (tile T = blk / 4, quarter blk % 4); 64-bit tile bases, 32-bit in-tile offsets
    const long long T = blk >> 2;
    const int quarter = blk & 3;
    uint8_t* const xtb = xt8 + T * 65536 + quarter * 32;
    uint8_t* const x6b = x6 + T * 49152;
    uint8_t* const xsb = xs + T * 2048 + 2 * quarter;
#pragma unroll
    for (int r = 0; r < 4; ++r) {  // xT8: 512 columns x 2 halves of 16 rows
        const int c = 128 * r + (tid >> 1), h = tid & 1;  // column group c >> 4 has (c >> 7) & 3 = r
        v4u dv = *reinterpret_cast<const v4u*>(lds + XT_OFF + XT_GSTR * (c >> 4) + 32 * (c & 15) + 16 * (h ^ (r >> 1)));
        if (r & 1) dv = (v4u){dv.z, dv.w, dv.x, dv.y};
        *reinterpret_cast<v4u*>(xtb + c * 128 + 16 * h) = dv;
    }
#pragma unroll
    for (int it = 0; it < 3; ++it) {  // x6: per K step 2 KiB head + 1 KiB tail of this quarter
        const int u = tid + NT * it;
        const int st = u / 192, v = u % 192;
        const bool hd = v < 128;
        const int go = st * 12288 + (hd ? quarter * 2048 + 16 * v : 8192 + quarter * 1024 + 16 * (v - 128));
        const int lo = X6_OFF + st * X6_SSTR + (hd ? 96 * (v >> 2) + 16 * (v & 3) : 96 * ((v - 128) >> 1) + 64 + 16 * (v & 1));
        *reinterpret_cast<v4u*>(x6b + go) = *reinterpret_cast<const v4u*>(lds + lo);
    }
    {  // xs: (block, r) -> 2 bytes (m = 2*quarter, 2*quarter+1)
        const uint16_t v = *reinterpret_cast<const uint16_t*>(lds + XS_OFF + 2 * tid);
        *reinterpret_cast<uint16_t*>(xsb + (tid >> 4) * 128 + (tid & 15) * 8) = v;
    }
}

#ifndef LNMX6_WPE
#define LNMX6_WPE 0  // experiment knob: -DLNMX6_WPE=4 asks for 4 waves / SIMD (<= 128 VGPR)
#endif
#if LNMX6_WPE > 0
#define LNMX6_ATTR __attribute__((amdgpu_waves_per_eu(LNMX6_WPE, LNMX6_WPE)))
#else
#define LNMX6_ATTR
#endif

template <bool X8, bool YD>
__global__ __launch_bounds__(NT) LNMX6_ATTR void ln_store_mx6_kernel(const uint16_t* __restrict__ x,
                                                          const uint16_t* __restrict__ wt,
                                                          const uint16_t* __restrict__ bs,
                                                          float* __restrict__ mean_out, float* __restrict__ rstd_out,
                                                          uint8_t* __restrict__ xt8, uint8_t* __restrict__ x6,
                                                          uint8_t* __restrict__ xs, uint8_t* __restrict__ x8,
                                                          float* __restrict__ yd, float eps, float inv_scale, int M,
                                                          int n_blocks) {
    // MODE.FP16_OVFL (bit 23): the fp8 converts saturate to +-448, as in the Triton kernel's prologue.
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n s_nop 7\n s_nop 7" ::: "memory");
    const int k = blockIdx.x;
    const int xcd = k & 7, rr8 = k >> 3;
    const int blk = (((rr8 / XCD_GROUP) * 8 + xcd) * XCD_GROUP) + (rr8 % XCD_GROUP);  // store_t v2 program map
    if (blk >= n_blocks) return;
    __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
    if ((long long)(blk + 1) * BN <= (long long)M)
        block_body<X8, YD, false>(x, wt, bs, mean_out, rstd_out, xt8, x6, xs, x8, yd, eps, inv_scale, M, blk, lds);
    else
        block_body<X8, YD, true>(x, wt, bs, mean_out, rstd_out, xt8, x6, xs, x8, yd, eps, inv_scale, M, blk, lds);
}

template <bool X8, bool YD>
static int launch(const void* x, const void* w, const void* b, void* mean, void* rstd, void* xt8, void* x6, void* xs,
                  void* x8, void* yd, float eps, float inv_scale, int M, int Mp, void* stream) {
    if (M <= 0 || Mp < M || Mp % 128 != 0 || Mp - M >= 128) return (int)hipErrorInvalidValue;
    const int n_blocks = Mp / BN;
    const long long grid = ((long long)n_blocks + 8 * XCD_GROUP - 1) / (8 * XCD_GROUP) * (8 * XCD_GROUP);
    if (grid > 0x7fffffffLL) return (int)hipErrorInvalidValue;
    hipLaunchKernelGGL((ln_store_mx6_kernel<X8, YD>), dim3((unsigned)grid), dim3(NT), 0, (hipStream_t)stream,
                       (const uint16_t*)x, (const uint16_t*)w, (const uint16_t*)b, (float*)mean, (float*)rstd,
                       (uint8_t*)xt8, (uint8_t*)x6, (uint8_t*)xs, (uint8_t*)x8, (float*)yd, eps, inv_scale, M,
                       n_blocks);
    return (int)hipGetLastError();
}

}  // namespace lnmx6

// C ABI. x8 / yd may be NULL (not written). Returns 0 when the launch was issued without error, else the hipError_t of
// this launch (hipGetLastError() right after it). A sticky error that an EARLIER HIP call left on this thread is
// cleared before the launch and, if this launch is clean, returned as LNMX6_STALE_ERROR_BASE + code, so that the caller
// can name it as not this kernel's (r14 it49 review i3). Asynchronous faults are not visible here.
#define LNMX6_STALE_ERROR_BASE 100000
extern "C" int ln_store_mx6_launch(const void* x, const void* w, const void* b, void* mean, void* rstd, void* xt8,
                                   void* x6, void* xs, void* x8, void* yd, float eps, float inv_scale, int M, int Mp,
                                   void* stream) {
    const int stale = (int)hipGetLastError();
    int rc;
    if (x8 != nullptr && yd != nullptr)
        rc = lnmx6::launch<true, true>(x, w, b, mean, rstd, xt8, x6, xs, x8, yd, eps, inv_scale, M, Mp, stream);
    else if (x8 != nullptr)
        rc = lnmx6::launch<true, false>(x, w, b, mean, rstd, xt8, x6, xs, x8, yd, eps, inv_scale, M, Mp, stream);
    else if (yd != nullptr)
        rc = lnmx6::launch<false, true>(x, w, b, mean, rstd, xt8, x6, xs, x8, yd, eps, inv_scale, M, Mp, stream);
    else
        rc = lnmx6::launch<false, false>(x, w, b, mean, rstd, xt8, x6, xs, x8, yd, eps, inv_scale, M, Mp, stream);
    return (rc == 0 && stale != 0) ? LNMX6_STALE_ERROR_BASE + stale : rc;
}
extern "C" const char* ln_store_mx6_impl(void) {
    return "ln_store_mx6 v3 (r14 iter49 + review fixes): 32-row WG, XCD group 8 (v26.3's store_t v2 map; Triton store_t "
           "v2 now 64-token, XCD group 4), Triton-order statistics, e4m3 xT8 dwords from row-pair converts, MX6-T x6 + "
           "MX6-A-v1 xs via one fp6 convert per 2 rows (s_nop 1 before, s_nop 4 after), LDS-staged copy-out";
}

// Custom FP8 K=tokens weight-gradient GEMM ("wgrad_ck") for the HSTU uvqk (W1) and vk (W2) projections, gfx950.
// Arbor r14 iterations 37 (kernel lab, probes/iter37/wgrad, variant "pw1") / 38 (integration). Loaded by
// wgrad_ck.py (DLRM_HSTU_FP8_BWD_WGRAD_CK=1, default off), called from bwd_fp8._run_tm_wgrad.
//
//   dW[kx][n] = (gs*as) * sum_{t<M} xT[kx][t] * G[t][n]      kx < 512, n < N, t < M (tokens)
//   G  : dz8 e5m2 [>=M, N] row-major (ld = N); W1 N=2048 (uvqk), W2 N=1024 (vk). Rows >= M are never read.
//   xT : e4m3 [512, ldm] row-major (tokens contiguous), ldm >= roundup(M,128), ldm % 16 == 0; the tail tokens
//        [M, roundup(M,128)) ARE read and must be finite (the production producers zero-fill them).
//   out: fp32 [512, N] row-major = the bwd_fp8._run_tm_wgrad / hipBLASLt fp8_wgrad_tm orientation-0 output.
//   part: fp32 scratch [NSPLIT, 512, N]; fully overwritten by every call (no stale partial can leak).
//
// Design: split-K over XCD cohorts. NSPLIT * NT WGs of 512 threads (8 waves, 2/SIMD), 1 WG per CU (128 KiB LDS).
// WG b -> XCD b%8 (round-robin dispatch), cohort = (b/8)/NT, tile = (b/8)%NT, split = xcd*(NSPLIT/8) + cohort,
// NT = 2*N/256 output tiles of 256 x 256 (W1: NT=16, NSPLIT=16; W2: NT=8, NSPLIT=32; 256 WGs either way). A cohort
// owns a contiguous token range [KT*s/NSPLIT, KT*(s+1)/NSPLIT) (KT = ceil(M/128) steps of 128 tokens) and computes
// ALL NT tiles over it, so a token chunk of both operands is fetched from HBM once per cohort and re-served from the
// XCD L2 to the cohort's NT WGs. Each WG writes its fp32 tile partial to part[split]; a second kernel sums the NSPLIT
// partials in the fixed order 0..NSPLIT-1 and multiplies by gs*as.
//
// Numerics class: fp32-order relative to hipBLASLt. Each v_mfma_f32_16x16x128_f8f6f4 reduces a 128-token chunk
// internally; that 128-wide dot product, the same instruction on the same chunks as in hipBLASLt, dominates the error
// against an exact fp64 GEMM of the same fp8 operands (rel_l2 ~1.2e-5 Gaussian, ~6e-5 heavy-tailed, independent of M,
// equal for both kernels; r14 iter38 review rv2). Only the fp32 order in which the chunk results are summed (K steps
// within a split, then the fixed-order split sum) differs from hipBLASLt's split-K (GSU16 + PostGSU16).
// Deterministic: no atomics, fixed reduction order, bitwise-reproducible run to run.
//
// WG = 8 waves; wave w owns tile rows [128*(w>>2),+128) x cols [64*(w&3),+64) = 8 x 4 MFMA blocks of
// v_mfma_f32_16x16x128_f8f6f4 issued operand-swapped: A-operand = G fragment (cbsz = 1, bf8/e5m2), B-operand = xT
// fragment (blgp = 0, fp8/e4m3), so a lane holds 4 consecutive n of one kx (P7 k512_fp8_gemm.cu:23-26).
// LDS: A0 | A1 | B0 | B1, 32 KiB each (two stages), filled by buffer_load_dwordx4 ... lds (LDS-DMA,
// P7 k512_fp8_gemm.cu:152-168); every fragment-read offset fits the 16-bit ds immediate.
//   A (xT) tile: row r at r*128, 16-byte chunk c at position c ^ ((r&15)>>1) (HipKittens st_16x128 swizzle; DMA
//     source-offset form and fragment reads borrowed from P7 k512_fp8_gemm.cu:211-217, 230-234, 95-118).
//   B (G) tile: token row t at t*256 (natural layout), chunk c at position c ^ f(t), f(t) = (t&7) | ((t>>4)&1)<<3,
//     read with ds_read_b64_tr_b8 (gfx950 transposing read; lane mapping derived by the r14 iter37 probe
//     probes/iter37/wgrad/probe/tr8_probe.cpp: in each 16-lane group out[L][j] = mem[addr[2j + (L>>3)] + (L&7)]).
//   MFMA K mapping (both operands): lane group g = l>>4 holds tokens 16g+[0,16) in bytes 0-15 and 64+16g+[0,16)
//     in bytes 16-31 (P7/HipKittens layout); B tr read r covers bytes 8r..8r+7.
// Pipeline: one barrier per 128-token step; the step-k+1 DMA goes to the other stage right after the step-k
// barrier; progressive step start (first MFMA after 6 LDS reads, <= 10 LDS ops in flight).
// Token tail: G is read through a buffer resource with num_records = valid_tokens*N per step, so tokens >= M read
// as 0 (0 * finite xT tail = 0).
// Step-tiled xT (template XT=1, entries wgrad_ck_w1_t / wgrad_ck_w2_t; Arbor r14 iteration 39, used with
// DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED=1, whose LN producers write xT in this layout; wgrad_ck.py): xT e4m3
// [>= KT][512][128], element (kx, t) at (t/128)*65536 + kx*128 + t%128, i.e. each 128-token step of all 512 rows is
// one contiguous 64 KiB block (the WG's 256 rows: one contiguous 32 KiB half). The DMA reads the same bytes into the
// same LDS positions as the row-major path (row r at r*128 instead of r*ldm), so the output is bitwise identical.
// The row-major stream (256 rows x 128 B at stride ldm per step) made the kernel stride-sensitive (the W3 lab:
// row-major 1.03-1.24x vs step-tiled 1.32-1.42x over hipBLASLt; r14 probes/iter38/w3). The tail tokens
// [M, KT*128) of the last block are read and must be finite (the producers zero-fill them).
// Resources (hipcc -O3 --offload-arch=gfx950 -mllvm -amdgpu-sched-strategy=iterative-ilp, ROCm 7.2.1): 190 VGPR,
// 0 AGPR, 0 spill, 131072 B LDS for both entries (= the lab pw1 build).
// This file is the lab source probes/iter37/wgrad/wgrad_tm.cu reduced to the production variant
// wgrad_tm_kernel<N, 1, NSPLIT, PRIO=0, PD=0, PW=1, LI=0> (no L2-prefetch / late-issue / SAMEK / ring variants),
// kernels renamed wgrad_ck_kernel / wgrad_ck_reduce; same instruction stream, bitwise-identical output.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

namespace wgck {
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}  // namespace wgck

extern "C" __device__ void wgck_llvm_buffer_load_lds(wgck::i32x4 rsrc, wgck::lds_ptr_t lds, int size, int voffset,
                                                     int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

namespace wgck {

constexpr int KX = 512;          // output rows (activation features)
constexpr int KS = 128;          // tokens per K step
constexpr int A_ST = 256 * KS;   // 32 KiB A tile per stage
constexpr int B_ST = KS * 256;   // 32 KiB B tile per stage
constexpr int B_OFF = 2 * A_ST;  // B stages start at 64 KiB (A0 | A1 | B0 | B1): all read offsets fit 16-bit imm
constexpr int FMT_G = 1;         // MFMA cbsz of the G (dz8) operand: 1 = bf8 (e5m2); xT is blgp 0 = fp8 (e4m3)

// gfx950 buffer resource (P7 k512_fp8_gemm.cu:75-83)
__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}

#define WGCK_FENCE() asm volatile("" ::: "memory")
#define WGCK_SB() __builtin_amdgcn_sched_barrier(0)
#define WGCK_BARRIER()                \
    do {                              \
        WGCK_FENCE();                 \
        __builtin_amdgcn_s_barrier(); \
        WGCK_FENCE();                 \
        WGCK_SB();                    \
    } while (0)

template <int IMM>
__device__ __forceinline__ i32x4 ds_read16(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x4 v;
    asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
template <int IMM>
__device__ __forceinline__ i32x2 ds_tr8(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x2 v;
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}

// xT fragment (16 kx rows x 128 tokens), HipKittens rt_16x128 register layout (P7 k512_fp8_gemm.cu:103-118)
template <int S>
__device__ __forceinline__ i32x8 fragA(uint32_t lp0, uint32_t lp1) {
    const i32x4 lo = ds_read16<S>(lp0);
    const i32x4 hi = ds_read16<S>(lp1);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}
// G fragment (16 n cols x 128 tokens) via 4 transposing reads; read r -> bytes 8r..8r+7 = tokens
// 16g + 8(r&1) + 64(r>>1) + [0,8)  (row offsets 2048*(r&1) + 16384*(r>>1) are immediates)
template <int S>
__device__ __forceinline__ i32x8 fragB(uint32_t q) {
    const i32x2 r0 = ds_tr8<S>(q);
    const i32x2 r1 = ds_tr8<S + 2048>(q);
    const i32x2 r2 = ds_tr8<S + 16384>(q);
    const i32x2 r3 = ds_tr8<S + 16384 + 2048>(q);
    const i32x4 lo = __builtin_shufflevector(r0, r1, 0, 1, 2, 3);
    const i32x4 hi = __builtin_shufflevector(r2, r3, 0, 1, 2, 3);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}

__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    wgck_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, 0);
}

}  // namespace wgck

// Main kernel: one WG computes one 256 x 256 fp32 tile partial of split `split` (see the header).
// XT = 0: xT row-major [512, ldm]; XT = 1: xT step-tiled [>= KT][512][128] (ldm unused; see "Step-tiled xT").
template <int N, int NSPLIT, int XT>
__global__ __launch_bounds__(512, 2) void wgrad_ck_kernel(const uint8_t* __restrict__ G, const uint8_t* __restrict__ X,
                                                          long long ldm, float* __restrict__ part, int M) {
    using namespace wgck;
    static_assert(N % 256 == 0, "N must be a multiple of 256");
    static_assert(NSPLIT % 8 == 0, "NSPLIT must be a multiple of 8 (XCDs)");
    static_assert(XT == 0 || XT == 1, "XT: 0 = row-major xT, 1 = step-tiled xT");
    constexpr int NTN = N / 256;     // n tiles
    constexpr int NT = 2 * NTN;      // tiles (2 kx tiles of 256)
    constexpr int CPX = NSPLIT / 8;  // cohorts per XCD
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;
    const int cohort = loc / NT, tile = loc % NT;
    const int split = xcd * CPX + cohort;
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * 256;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);

    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);

    // ---- DMA (global -> LDS) per-lane source offsets ----
    // A: wave w, instr i, lane l -> LDS A + 8192i + 1024w + 16l = row 64i + 8w + l/8, position l&7
    const uint32_t chunkA = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    // XT=0: row stride ldm (+ 64i*ldm in soffset); XT=1: row stride 128 inside the step's block (+ 64i*128)
    const uint32_t voffA = XT ? (uint32_t)((8 * w + (lane >> 3)) * KS) + chunkA * 16u
                              : (uint32_t)((8 * w + (lane >> 3)) * ldm) + chunkA * 16u;
    // B: wave w, instr i, lane l -> LDS B + 8192i + 1024w + 16l = token 32i + 4w + l/16, position l&15
    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const uint32_t chunkB = (uint32_t)((lane & 15) ^ fB);
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) voffB[i] = (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + chunkB * 16u;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;

    // ---- fragment read addresses ----
    const int r16 = lane & 15, q4 = lane >> 4;
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t rowA = lds0 + 16384u * (uint32_t)(w >> 2) + (uint32_t)r16 * 128u;
    const uint32_t lpA0 = rowA + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lpA1 = rowA + (((uint32_t)(16 * q4 + 64)) ^ swz);
    uint32_t qB[4];
    {
        const int s = r16, g = q4;
        const int t0 = 16 * g + (s >> 1);
        const int f = (s >> 1) + 8 * (g & 1);
        const int cw = 4 * (w & 3);
        const uint32_t q = lds0 + (uint32_t)B_OFF + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) + ((uint32_t)(cw ^ f) << 4);
#pragma unroll
        for (int fb = 0; fb < 4; ++fb) qB[fb] = q ^ ((uint32_t)fb << 4);
    }

    f32x4 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};

    const uint8_t* Xt = X + (long long)kx0 * ldm;
    const uint32_t nrecA = (uint32_t)min((long long)256 * ldm, (long long)0xffffffffLL);

    // Stage toggle: smem is the only LDS allocation (base 0), stage 0 offsets < 32 KiB inside each 64 KiB half,
    // so XOR 0x8000 on any A/B address flips stage 0 <-> 1. One loop body per K step (no 2x unroll, single exit):
    // keeps the accumulators loop-carried in place.
    auto issue = [&](int k, uint32_t st) {
        if constexpr (XT) {
            // step k = one contiguous [512][128] block at k*KX*KS; this WG's 256 rows start at kx0*KS in it
            const i32x4 rA = make_rsrc(X + (long long)k * (KX * KS) + (long long)kx0 * KS, (uint32_t)(256 * KS));
            const int vt = min(KS, M - k * KS);
            const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
            for (int i = 0; i < 4; ++i) g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), voffA, (uint32_t)(64 * i * KS));
#pragma unroll
            for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
        } else {
            const i32x4 rA = make_rsrc(Xt + (long long)k * KS, nrecA);
            const int vt = min(KS, M - k * KS);
            const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
            for (int i = 0; i < 4; ++i) g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), voffA, (uint32_t)(64 * i * ldm));
#pragma unroll
            for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
        }
    };

    uint32_t pA0 = lpA0, pA1 = lpA1;
    uint32_t q0 = qB[0], q1 = qB[1], q2 = qB[2], q3 = qB[3];
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK_SB();
        i32x8 bf[4];
        i32x8 a[2];
        auto mma = [&](int fa, int fb, const i32x8& av) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[fb], av, acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };
        // progressive start: first MFMA after bf0 + a0 (6 reads) instead of all 18; <= 10 LDS ops in flight
        bf[0] = fragB<0>(q0);
        a[0] = fragA<0>(pA0, pA1);
        bf[1] = fragB<0>(q1);
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        bf[2] = fragB<0>(q2);
        WGCK_SB();
        mma(0, 0, a[0]);
        WGCK_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        bf[3] = fragB<0>(q3);
        WGCK_SB();
        mma(0, 1, a[0]);
        WGCK_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        a[1] = fragA<2048>(pA0, pA1);
        WGCK_SB();
        mma(0, 2, a[0]);
        WGCK_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        WGCK_SB();
        mma(0, 3, a[0]);
        WGCK_SB();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        WGCK_SB();
        [&]<int... FB>(std::integer_sequence<int, FB...>) {
            (([&] {
                 constexpr int FA = FB + 1;
                 if constexpr (FA < 7) a[(FA + 1) & 1] = fragA<(FA + 1) * 2048>(pA0, pA1);
                 WGCK_SB();
#pragma unroll
                 for (int fb = 0; fb < 4; ++fb) mma(FA, fb, a[FA & 1]);
                 WGCK_SB();
                 asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                 WGCK_SB();
             }()),
             ...);
        }(std::make_integer_sequence<int, 7>{});
        st ^= 0x8000u;
        pA0 ^= 0x8000u;
        pA1 ^= 0x8000u;
        q0 ^= 0x8000u;
        q1 ^= 0x8000u;
        q2 ^= 0x8000u;
        q3 ^= 0x8000u;
    }

    // ---- partial store: acc[fa][fb] lane l = dW[kx0 + 128(w>>2) + 16fa + (l&15)][n0 + 64(w&3) + 16fb + 4(l>>4) + 0..3]
    // (every WG stores its whole tile, also when its split has no K step: part is fully overwritten every call)
    float* P = part + ((long long)split * KX + kx0 + 128 * (w >> 2) + r16) * N + n0 + 64 * (w & 3) + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < 8; ++fa)
#pragma unroll
        for (int fb = 0; fb < 4; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}

// Fixed-order reduction of the NSPLIT partials (split 0, 1, ..., NSPLIT-1), then x (gs*as). One float4 per thread.
template <int N, int NSPLIT>
__global__ __launch_bounds__(256) void wgrad_ck_reduce(const float* __restrict__ part, float* __restrict__ out,
                                                       const float* __restrict__ gs, const float* __restrict__ as) {
#pragma clang fp contract(off)
    using namespace wgck;
    constexpr long long Q = (long long)KX * N / 4;
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    const f32x4* p = reinterpret_cast<const f32x4*>(part);
    f32x4 s = p[i];
    // partial unroll: a full 32-way unroll (W2, NSPLIT=32) segfaults the ROCm 7.2.1 clang-22 greedy RA
#pragma unroll 8
    for (int k = 1; k < NSPLIT; ++k) s = s + p[k * Q + i];
    const float sc = gs[0] * as[0];
    reinterpret_cast<f32x4*>(out)[i] = s * sc;
}

// 32-bit buffer offsets: the xT resource spans 256 rows of ldm bytes (num_records <= 0xffffffff) and the DMA
// soffset reaches 192*ldm, so ldm must stay below 2^32 / 256 (16.7M tokens per call; production ~3M).
static constexpr long long WGCK_LDM_MAX = 0xffffffffLL / 256;

template <int N, int NSPLIT, int XT>
static int wgrad_ck_launch(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                           const float* as, int M, void* stream) {
    if constexpr (XT) {
        // step-tiled xT [>= ceil(M/128)][512][128]: ldm is its outer (step) stride, which must be 512*128
        if (M <= 0 || ldm != (long long)wgck::KX * wgck::KS) return (int)hipErrorInvalidValue;
    } else {
        if (M <= 0 || ldm < ((M + 127) / 128) * 128LL) return (int)hipErrorInvalidValue;
        if (ldm > WGCK_LDM_MAX || (ldm & 15) != 0) return (int)hipErrorInvalidValue;
    }
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    constexpr int NT = 2 * (N / 256);
    // The HIP last error is sticky per thread (HIP 7.2, as in CUDA): an error that an earlier, unrelated call left on
    // this thread would otherwise read as a failure of these launches. Drain it first, check each launch on its own,
    // and report a drained error as a negative code (both kernels launched; r14 iter38 review S1).
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_kernel<N, NSPLIT, XT>), dim3(NSPLIT * NT), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, ldm, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_reduce<N, NSPLIT>), dim3((wgck::KX * N / 4) / 256), dim3(256), 0, (hipStream_t)stream,
                       (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

// ABI (all entries): G e5m2 [>=M, N] row-major (ld N), X e4m3 [512, ldm] (wgrad_ck_w1 / _w2) or step-tiled
// [>= ceil(M/128)][512][128] with ldm = 65536 = its step stride (wgrad_ck_w1_t / _w2_t), part fp32 [NSPLIT, 512, N]
// scratch, out fp32 [512, N], gs/as device fp32 scalars, stream = hipStream_t. Returns 0 = both kernels launched;
// > 0 = the hipError_t of a rejected argument or a failed launch (out not written); < 0 = both kernels launched, and
// -rc is a stale error an earlier call had left on this thread (drained here, not caused by these launches).
#define WGCK_SIG \
    (const void* G, const void* X, long long ldm, void* part, void* out, const float* gs, const float* as, int M, \
     void* stream)
#define WGCK_ARGS G, X, ldm, part, out, gs, as, M, stream
// W1: uvqk wgrad, N=2048, NSPLIT=16 (2 cohorts of 16 WGs per XCD)
extern "C" int wgrad_ck_w1 WGCK_SIG { return wgrad_ck_launch<2048, 16, 0>(WGCK_ARGS); }
extern "C" int wgrad_ck_w1_nsplit(void) { return 16; }
// W2: vk wgrad, N=1024, NSPLIT=32 (4 cohorts of 8 WGs per XCD)
extern "C" int wgrad_ck_w2 WGCK_SIG { return wgrad_ck_launch<1024, 32, 0>(WGCK_ARGS); }
extern "C" int wgrad_ck_w2_nsplit(void) { return 32; }
// Step-tiled xT (XT=1) entries of W1 / W2: same tiles, splits, reduction and fp32 order -> bitwise equal to the
// row-major entries on the same xT bytes relaid.
extern "C" int wgrad_ck_w1_t WGCK_SIG { return wgrad_ck_launch<2048, 16, 1>(WGCK_ARGS); }
extern "C" int wgrad_ck_w1_t_nsplit(void) { return 16; }
extern "C" int wgrad_ck_w2_t WGCK_SIG { return wgrad_ck_launch<1024, 32, 1>(WGCK_ARGS); }
extern "C" int wgrad_ck_w2_t_nsplit(void) { return 32; }

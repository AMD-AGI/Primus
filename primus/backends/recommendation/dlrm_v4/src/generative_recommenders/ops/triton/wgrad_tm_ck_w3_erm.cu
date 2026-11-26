// Custom FP8 K=tokens weight-gradient GEMM ("wgrad_ck", W3) for the HSTU output projection, gfx950.
// Arbor r14 iterations 37 (kernel lab, probes/iter37/wgrad: the W1/W2 kernel wgrad_tm.cu) / 38 (W3 port, probes/iter38/w3
// w3_v2.cu variant "pw192"; integration). Loaded by wgrad_ck_w3.py (DLRM_HSTU_FP8_BWD_WGRAD_CK_W3=1, default off),
// called from the NN-mode out-proj wgrad in triton_hstu_linear.py (HSTUComputeOutputFunction.backward) in place of
// ops/hipblaslt_fp8_nn.fp8_wgrad_nn(y8, dout_t, scale_y, scale_d, fp32).
//
//   dW[n][c] = (gs*as) * sum_{t<M} X[n][t] * G[t][c]       n < 512, c < 1536, t < M (tokens)
//   G  : y8 e4m3 [>=M, 1536] row-major (ld 1536): the e4m3 LN*U*dropout output the FP8 forward kept. Rows >= M are
//        never read (buffer range check -> 0).
//   X  : dout_t e4m3 [512, ldm] row-major (tokens contiguous): the dual cast's token-major dout copy, ldm % 16 == 0,
//        ldm >= roundup(M,16) (production: ldm = roundup(M,16), zero tail). The kernel reads tokens
//        [0, min(ldm, roundup(M,128))) in 16-token chunks; chunks at token >= ldm are range-masked to 0, so nothing past
//        the row pitch is read. Tokens [M, min(ldm, roundup(M,128))) ARE read and multiplied by the zero G rows: they
//        must be finite (wgrad_ck_w3.py zero-fills them before every launch).
//   out: fp32 [512, 1536] row-major = fp8_wgrad_nn's dw_t (the caller transposes / casts it as before).
//   part: fp32 scratch [16, 512, 1536] (48 MiB); fully overwritten by every call (no stale partial can leak).
//   gs = scale_y, as = scale_d: device fp32 scalars (dequant scales), applied once in the reduction.
//
// Design (the W1 design with a 192-wide n tile): split-K over XCD cohorts. 256 WGs of 512 threads (8 waves, 2/SIMD),
// 1 WG per CU (128 KiB LDS). WG b -> XCD b%8 (round-robin dispatch), cohort = (b/8)/NT, tile = (b/8)%NT,
// split = xcd*(NSPLIT/8) + cohort, NT = 2 * (1536/192) = 16 output tiles of 256 x 192, NSPLIT = 16. A cohort owns a
// contiguous token range [KT*s/16, KT*(s+1)/16) (KT = ceil(M/128) steps of 128 tokens) and computes ALL 16 tiles over
// it, so a token chunk of both operands is fetched from HBM once per cohort and re-served from the XCD L2 (lab EA
// census at M=2650962: 5.43 GB read = 1.000x the minimum bytes vs hipBLASLt's 3.33x). Each WG writes its fp32 tile
// partial to part[split]; a second kernel sums the 16 partials in the fixed order 0..15 and multiplies by gs*as.
//
// Numerics class: fp32-order. e4m3 x e4m3 products are exact in fp32; only the fp32 accumulation order (MFMA K-chunks
// within a split, then the fixed-order split sum) differs from hipBLASLt's. Deterministic: no atomics, fixed reduction
// order, bitwise-reproducible run to run.
//
// WG = 8 waves; wave w owns tile rows [128*(w>>2),+128) x cols [48*(w&3),+48) = 8 x 3 MFMA blocks of
// v_mfma_scale_f32_16x16x128_f8f6f4 issued operand-swapped: A-operand = G fragment (cbsz = 0, fp8/e4m3), B-operand = X
// fragment (blgp = 0, fp8/e4m3), so a lane holds 4 consecutive c of one n (P7 k512_fp8_gemm.cu:23-26).
// LDS: A0 | A1 | B0 | B1, 32 KiB each (two stages), filled by buffer_load_dwordx4 ... lds (LDS-DMA,
// P7 k512_fp8_gemm.cu:152-168); every fragment-read offset fits the 16-bit ds immediate.
//   A (X) tile: row r at r*128, 16-byte chunk c at position c ^ ((r&15)>>1) (HipKittens st_16x128 swizzle; DMA
//     source-offset form and fragment reads borrowed from P7 k512_fp8_gemm.cu:211-217, 230-234, 95-118).
//   B (G) tile: token row t at t*256 (256-B pitch kept from W1; only the 12 valid 16-B chunks of the 192-wide tile are
//     DMA'd, lanes of the 4 other swizzle positions get an out-of-range voffset), chunk c at position c ^ f(t),
//     f(t) = (t&7) | ((t>>4)&1)<<3, read with ds_read_b64_tr_b8 (gfx950 transposing read; lane mapping derived by the
//     r14 iter37 probe probes/iter37/wgrad/probe/tr8_probe.cpp: in each 16-lane group out[L][j] = mem[addr[2j + (L>>3)]
//     + (L&7)]).
//   MFMA K mapping (both operands): lane group g = l>>4 holds tokens 16g+[0,16) in bytes 0-15 and 64+16g+[0,16)
//     in bytes 16-31 (P7/HipKittens layout); B tr read r covers bytes 8r..8r+7.
// Pipeline: one barrier per 128-token step; the step-k+1 DMA goes to the other stage right after the step-k
// barrier; progressive step start (first MFMA after 6 LDS reads).
// Token tails: G is read through a buffer resource with num_records = valid_tokens*1536 per step (tokens >= M read
// as 0); on the last step, X chunks at token >= ldm get voffset 0x80000000 (> num_records <= 0x7fffffff -> 0).
// Resources (hipcc -O3 --offload-arch=gfx950 -mllvm -amdgpu-sched-strategy=iterative-ilp, ROCm 7.2.1): 154 VGPR,
// 0 AGPR, 0 spill, 131072 B LDS (= the lab pw192 build).
// This file is the lab source probes/iter38/w3/w3_v2.cu reduced to the production variant w3_pw192 =
// w3_launch<N=1536, TNW=192, FMT_G=0, NSPLIT=16, RING=0> (w3_pw_kernel + w3_reduce; no ring / pwB / ringB / pw256
// variants), kernels renamed wgrad_ck_w3_kernel / wgrad_ck_w3_reduce; same instruction stream, bitwise-identical output.
// Iteration 39 (B13) adds the step-tiled-X entry wgrad_ck_w3_t (kernel wgrad_ck_w3_t_kernel = the lab's w3_v3.cu
// pwB192_t, see below), used when the dual cast wrote dout_t step-tiled [ceil(M/128)][512][128].
// The W1/W2 (uvqk / vk) kernels of the same design are wgrad_tm_ck.cu (wgrad_ck.py).
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

namespace wgck3 {
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}  // namespace wgck3

extern "C" __device__ void wgck3_llvm_buffer_load_lds(wgck3::i32x4 rsrc, wgck3::lds_ptr_t lds, int size, int voffset,
                                                      int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

namespace wgck3 {

constexpr int KX = 512;                // output rows (dout features)
constexpr int KS = 128;                // tokens per K step
constexpr int A_ST = 256 * KS;         // 32 KiB A tile per stage
constexpr int B_ST = KS * 256;         // 32 KiB B tile per stage (256-B row pitch)
constexpr int B_OFF = 2 * A_ST;        // B stages start at 64 KiB (A0 | A1 | B0 | B1): all read offsets fit 16-bit imm
constexpr int FMT_G = 0;               // MFMA cbsz of the G (y8) operand: 0 = fp8 (e4m3); X is blgp 0 = fp8 (e4m3)
constexpr uint32_t OOR = 0x80000000u;  // voffset that fails the range check (num_records <= 0x7fffffff)

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

#define WGCK3_FENCE() asm volatile("" ::: "memory")
#define WGCK3_SB() __builtin_amdgcn_sched_barrier(0)
#define WGCK3_BARRIER()               \
    do {                              \
        WGCK3_FENCE();                \
        __builtin_amdgcn_s_barrier(); \
        WGCK3_FENCE();                \
        WGCK3_SB();                   \
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
// X fragment (16 n rows x 128 tokens), HipKittens rt_16x128 register layout (P7 k512_fp8_gemm.cu:103-118)
template <int S>
__device__ __forceinline__ i32x8 fragA(uint32_t lp0, uint32_t lp1) {
    const i32x4 lo = ds_read16<S>(lp0);
    const i32x4 hi = ds_read16<S>(lp1);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}
// G fragment (16 c cols x 128 tokens) via 4 transposing reads; read r -> bytes 8r..8r+7 = tokens
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
    wgck3_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, 0);
}

}  // namespace wgck3

// Main kernel: one WG computes one 256 x TNW fp32 tile partial of split `split` (see the header).
template <int N, int TNW, int NSPLIT>
__global__ __launch_bounds__(512, 2) void wgrad_ck_w3_kernel(const uint8_t* __restrict__ G,
                                                             const uint8_t* __restrict__ X, long long ldm,
                                                             float* __restrict__ part, int M) {
    using namespace wgck3;
    static_assert(TNW == 192, "the production W3 tile is 256 x 192");
    static_assert(N % TNW == 0, "N must be a multiple of the tile width");
    static_assert(NSPLIT % 8 == 0, "NSPLIT must be a multiple of 8 (XCDs)");
    constexpr int NCH = TNW / 16;   // valid 16-B chunks per G tile row (12)
    constexpr int NFB = TNW / 64;   // G fragments (16 cols) per wave (3)
    constexpr int NTN = N / TNW;    // c tiles (8)
    constexpr int NT = 2 * NTN;     // tiles (2 n tiles of 256)
    constexpr int CPX = NSPLIT / 8; // cohorts per XCD
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;
    const int cohort = loc / NT, tile = loc % NT;
    const int split = xcd * CPX + cohort;
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * TNW;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);

    // DMA source offsets. A (X): wave w, instr i, lane l -> row 64i + 8w + l/8, position l&7, source chunk chunkA.
    const uint32_t chunkA = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    const uint32_t voffA = (uint32_t)((8 * w + (lane >> 3)) * ldm) + chunkA * 16u;
    // B (G): wave w, instr i, lane l -> token 32i + 4w + l/16, position l&15, source chunk (l&15) ^ fB.
    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const int chunkB = (lane & 15) ^ fB;
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        voffB[i] = chunkB < NCH ? (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + (uint32_t)chunkB * 16u : OOR;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;

    // fragment read addresses
    const int r16 = lane & 15, q4 = lane >> 4;
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t rowA = lds0 + 16384u * (uint32_t)(w >> 2) + (uint32_t)r16 * 128u;
    const uint32_t lpA0 = rowA + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lpA1 = rowA + (((uint32_t)(16 * q4 + 64)) ^ swz);
    uint32_t qB[NFB];
    {
        const int s = r16, g = q4;
        const int t0 = 16 * g + (s >> 1);
        const int f = (s >> 1) + 8 * (g & 1);
        const int cw = NFB * (w & 3);
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb)
            qB[fb] = lds0 + (uint32_t)B_OFF + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) +
                     ((uint32_t)((cw + fb) ^ f) << 4);
    }

    f32x4 acc[8][NFB];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < NFB; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};

    const uint8_t* Xt = X + (long long)kx0 * ldm;
    const uint32_t nrecA = (uint32_t)min((long long)256 * ldm, (long long)0x7fffffffLL);

    // Stage toggle: smem is the only LDS allocation (base 0), stage 0 offsets < 32 KiB inside each 64 KiB half,
    // so XOR 0x8000 on any A/B address flips stage 0 <-> 1.
    auto issue = [&](int k, uint32_t st) {
        const i32x4 rA = make_rsrc(Xt + (long long)k * KS, nrecA);
        uint32_t vA = voffA;
        if ((long long)(k + 1) * KS > ldm) {  // uniform: only the last step of the last split
            if ((long long)k * KS + 16 * (long long)chunkA >= ldm) vA = OOR;
        }
        const int vt = min(KS, M - k * KS);
        const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), vA, (uint32_t)(64 * i * ldm));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
    };

    uint32_t pA0 = lpA0, pA1 = lpA1;
    uint32_t q[NFB];
#pragma unroll
    for (int fb = 0; fb < NFB; ++fb) q[fb] = qB[fb];
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK3_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK3_SB();
        i32x8 bf[NFB];
        i32x8 a[2];
        auto mma = [&](int fa, int fb, const i32x8& av) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[fb], av, acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };
        // progressive start: first MFMA after bf0 + a0 (6 reads)
        bf[0] = fragB<0>(q[0]);
        a[0] = fragA<0>(pA0, pA1);
        bf[1] = fragB<0>(q[1]);
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        bf[2] = fragB<0>(q[2]);
        WGCK3_SB();
        mma(0, 0, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        a[1] = fragA<2048>(pA0, pA1);
        WGCK3_SB();
        mma(0, 1, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        WGCK3_SB();
        mma(0, 2, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        WGCK3_SB();
        [&]<int... FB>(std::integer_sequence<int, FB...>) {
            (([&] {
                 constexpr int FA = FB + 1;
                 if constexpr (FA < 7) a[(FA + 1) & 1] = fragA<(FA + 1) * 2048>(pA0, pA1);
                 WGCK3_SB();
#pragma unroll
                 for (int fb = 0; fb < NFB; ++fb) mma(FA, fb, a[FA & 1]);
                 WGCK3_SB();
                 asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                 WGCK3_SB();
             }()),
             ...);
        }(std::make_integer_sequence<int, 7>{});
        st ^= 0x8000u;
        pA0 ^= 0x8000u;
        pA1 ^= 0x8000u;
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb) q[fb] ^= 0x8000u;
    }

    // partial store: acc[fa][fb] lane l = dW[kx0 + 128(w>>2) + 16fa + (l&15)][n0 + TNW/4*(w&3) + 16fb + 4(l>>4) + 0..3]
    // (every WG stores its whole tile, also when its split has no K step: part is fully overwritten every call)
    float* P = part + ((long long)split * KX + kx0 + 128 * (w >> 2) + r16) * N + n0 + (TNW / 4) * (w & 3) + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < 8; ++fa)
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}

// Step-tiled X variant (Arbor r14 iteration 39, B13; DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED=1): X = dout_t e4m3
// STEP-TILED [KT][512][128], KT = ceil(M/128): step k holds tokens [128k, 128k+128) of all 512 rows, row n at n*128,
// tokens [M, KT*128) zero (the dual cast bwd_fp8._quantize_amax_dual_kernel T_TILED writes them; wgrad_ck_w3.py
// re-zeroes them before every launch). Each K step of a WG is one contiguous 32 KiB block (its 256 rows): no row
// stride (the row-major kernel above is sensitive to ldm, 1.03-1.24x vs hipBLASLt; this layout 1.32-1.42x at
// M 1.5-2.9M in the lab) and no tail masking. Lab source: probes/iter38/w3/w3_v3.cu w3_pwB_kernel<1536, 192, 0, 16,
// XT=1> ("pwB192_t") with XT fixed to 1 (the dead row-major branches and the ldm argument removed): 8 waves as a
// 4 (rows) x 2 (cols) grid, wave w owns rows [64*(w>>1),+64) x cols [96*(w&1),+96) = 4 x 6 MFMA blocks; the 4 X
// fragments stay in VGPRs, the 6 G fragments stream through a 2-deep buffer; same LDS layouts, DMA and
// double-buffered (pw1) schedule as wgrad_ck_w3_kernel. Every accumulator gets exactly one MFMA per K step over the
// same 128 tokens with the same operand bytes and the same split ranges as wgrad_ck_w3_kernel, so the partials and
// the output are bitwise identical to it (fp32 order unchanged).
template <int N, int TNW, int NSPLIT>
__global__ __launch_bounds__(512, 2) void wgrad_ck_w3_t_kernel(const uint8_t* __restrict__ G,
                                                               const uint8_t* __restrict__ X,
                                                               float* __restrict__ part, int M) {
    using namespace wgck3;
    static_assert(TNW == 192, "the production W3 tile is 256 x 192");
    static_assert(N % TNW == 0, "N must be a multiple of the tile width");
    static_assert(NSPLIT % 8 == 0, "NSPLIT must be a multiple of 8 (XCDs)");
    constexpr int NCH = TNW / 16;   // valid 16-B chunks per G tile row (12)
    constexpr int NFB = TNW / 32;   // G fragments (16 cols) per wave (6)
    constexpr int NFA = 4;          // X fragments (16 rows) per wave
    constexpr int NTN = N / TNW;    // c tiles (8)
    constexpr int NT = 2 * NTN;     // tiles (2 n tiles of 256)
    constexpr int CPX = NSPLIT / 8; // cohorts per XCD
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;
    const int cohort = loc / NT, tile = loc % NT;
    const int split = xcd * CPX + cohort;
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * TNW;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);

    // DMA source offsets. A (X step block, row pitch KS): wave w, instr i, lane l -> row 64i + 8w + l/8, position l&7,
    // source chunk chunkA (the LDS layout of wgrad_ck_w3_kernel).
    const uint32_t chunkA = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    const uint32_t voffA = (uint32_t)((8 * w + (lane >> 3)) * KS) + chunkA * 16u;
    // B (G): wave w, instr i, lane l -> token 32i + 4w + l/16, position l&15, source chunk (l&15) ^ fB.
    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const int chunkB = (lane & 15) ^ fB;
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        voffB[i] = chunkB < NCH ? (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + (uint32_t)chunkB * 16u : OOR;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;

    // fragment read addresses: rows 64*(w>>1) + 16*fa (+ r16), G chunks NFB*(w&1) + fb
    const int r16 = lane & 15, q4 = lane >> 4;
    const int rb = w >> 1, cb = w & 1;
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t rowA = lds0 + 8192u * (uint32_t)rb + (uint32_t)r16 * 128u;
    const uint32_t lpA0 = rowA + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lpA1 = rowA + (((uint32_t)(16 * q4 + 64)) ^ swz);
    uint32_t q[NFB];
    {
        const int s = r16, g = q4;
        const int t0 = 16 * g + (s >> 1);
        const int f = (s >> 1) + 8 * (g & 1);
        const int cw = NFB * cb;
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb)
            q[fb] = lds0 + (uint32_t)B_OFF + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) +
                    ((uint32_t)((cw + fb) ^ f) << 4);
    }

    f32x4 acc[NFA][NFB];
#pragma unroll
    for (int i = 0; i < NFA; ++i)
#pragma unroll
        for (int j = 0; j < NFB; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};

    // Stage toggle as in wgrad_ck_w3_kernel (XOR 0x8000 flips stage 0 <-> 1).
    auto issue = [&](int k, uint32_t st) {
        // this WG's 256 rows of step k: one contiguous 32 KiB block, always in range (zero tail tokens are data)
        const i32x4 rA = make_rsrc(X + (long long)k * (KX * KS) + (long long)kx0 * KS, (uint32_t)(256 * KS));
        const int vt = min(KS, M - k * KS);
        const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), voffA, (uint32_t)(64 * i * KS));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
    };

    uint32_t pA0 = lpA0, pA1 = lpA1;
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK3_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK3_SB();
        i32x8 a[NFA];
        i32x8 bf[2];
        auto mma = [&](int fa, int fb, const i32x8& bv) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bv, a[fa], acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };
        a[0] = fragA<0>(pA0, pA1);
        bf[0] = fragB<0>(q[0]);
        a[1] = fragA<2048>(pA0, pA1);
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        a[2] = fragA<4096>(pA0, pA1);
        WGCK3_SB();
        mma(0, 0, bf[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        a[3] = fragA<6144>(pA0, pA1);
        WGCK3_SB();
        mma(1, 0, bf[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        bf[1] = fragB<0>(q[1]);
        WGCK3_SB();
        mma(2, 0, bf[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        WGCK3_SB();
        mma(3, 0, bf[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        WGCK3_SB();
        [&]<int... FBM>(std::integer_sequence<int, FBM...>) {
            (([&] {
                 constexpr int FB = FBM + 1;
                 if constexpr (FB + 1 < NFB) bf[(FB + 1) & 1] = fragB<0>(q[FB + 1]);
                 WGCK3_SB();
#pragma unroll
                 for (int fa = 0; fa < NFA; ++fa) mma(fa, FB, bf[FB & 1]);
                 WGCK3_SB();
                 asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                 WGCK3_SB();
             }()),
             ...);
        }(std::make_integer_sequence<int, NFB - 1>{});
        st ^= 0x8000u;
        pA0 ^= 0x8000u;
        pA1 ^= 0x8000u;
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb) q[fb] ^= 0x8000u;
    }

    // partial store: acc[fa][fb] lane l = dW[kx0 + 64rb + 16fa + (l&15)][n0 + TNW/2*cb + 16fb + 4(l>>4) + 0..3]
    // (every WG stores its whole tile, also when its split has no K step: part is fully overwritten every call)
    float* P = part + ((long long)split * KX + kx0 + 64 * rb + r16) * N + n0 + (TNW / 2) * cb + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < NFA; ++fa)
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}

// Fixed-order reduction of the NSPLIT partials (split 0, 1, ..., NSPLIT-1), then x (gs*as). One float4 per thread.
template <int N, int NSPLIT>
__global__ __launch_bounds__(256) void wgrad_ck_w3_reduce(const float* __restrict__ part, float* __restrict__ out,
                                                          const float* __restrict__ gs, const float* __restrict__ as) {
#pragma clang fp contract(off)
    using namespace wgck3;
    constexpr long long Q = (long long)KX * N / 4;
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    const f32x4* p = reinterpret_cast<const f32x4*>(part);
    f32x4 s = p[i];
#pragma unroll 8
    for (int k = 1; k < NSPLIT; ++k) s = s + p[k * Q + i];
    const float sc = gs[0] * as[0];
    reinterpret_cast<f32x4*>(out)[i] = s * sc;
}


// ---------------------------------------------------------------------------------------------------------------------
// Arbor r14 iteration 41 (ship13 lab scout, probes/iter41/w3; wired in iteration 42 behind
// DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S=1, default off: wgrad_ck_w3.py binds the entry wgrad_ck_w3_t256s (MAP 2) for a
// step-tiled dout_t; wgrad_ck_w3_t256s_m1 (MAP 1, same bits) is kept for the lab only): "t256s" = the W1 8-wave
// 256x256-tile step-tiled kernel (wgrad_tm_ck.cu wgrad_ck_kernel XT=1) at N=1536 with e4m3 G (cbsz 0), NSPLIT=21 splits
// x 12 tiles = 252 WGs. The lab census (iter41) found the 256x192 W3 kernel bound by the per-CU global->LDS DMA path:
// it issues as many 128-B L2 line requests per K step as W1 (A 256 + B 256: the 192-B G rows straddle 2 lines) for 25%
// fewer FLOPs. 256x256 tiles restore W1's request/FLOP ratio; NSPLIT=21 uses 252 of 256 CUs. Cohort map: 2 XCD-local
// cohorts per XCD (splits 0..15) + 5 cohorts spilled over the leftover 7-8 slots of each XCD (MAP 1: tile order e % 12;
// MAP 2: column-major inside a spilled cohort so an XCD part duplicates X (25% of the bytes) instead of G).
// Numerics: same 128-token MFMA chunks, sequential fp32 accumulation within a split, fixed-order 21-way reduce:
// fp32-order class (NOT bitwise to the 16-split kernels; lab rel_l2 vs them 2.3e-7..3.1e-7, fp64 rel_l2 equal).
// Scratch: part fp32 [21, 512, 1536] (63 MiB).
// MAP 0: NSPLIT % 8 == 0, XCD-local cohorts (split = xcd*NSPLIT/8 + cohort). MAP 1 ("spill", any NSPLIT >= 16 with
// NSPLIT*NT <= 256): each XCD (WG b -> XCD b%8) hosts LPX = 32/NT full local cohorts (splits 0..8*LPX-1, XCD-local);
// the remaining splits fill the leftover slots of all XCDs in XCD-major order (a spilled cohort spans 2-3 XCDs, so
// its token chunks are fetched by more than one L2).
template <int N, int NSPLIT, int MAP>
__global__ __launch_bounds__(512, 2) void wgrad_ck_w3_t256s_kernel(const uint8_t* __restrict__ G, const uint8_t* __restrict__ X,
                                                          float* __restrict__ part, int M) {
    using namespace wgck3;
    constexpr int NTN = N / 256;
    constexpr int NT = 2 * NTN;
    constexpr int FMT_G = 0;
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;
    int split, tile;
    if constexpr (MAP == 0) {
        static_assert(NSPLIT % 8 == 0, "MAP 0 needs NSPLIT % 8 == 0");
        constexpr int CPX = NSPLIT / 8;
        split = xcd * CPX + loc / NT;
        tile = loc % NT;
    } else {
        constexpr int NWG = NSPLIT * NT;
        constexpr int LPX = 32 / NT;  // local cohorts per XCD
        static_assert(NWG <= 256 && NSPLIT >= 8 * LPX, "spill map");
        if (loc < LPX * NT) {
            split = xcd * LPX + loc / NT;
            tile = loc % NT;
        } else {
            int e = loc - LPX * NT;
            for (int x = 0; x < xcd; ++x) e += (NWG - x + 7) / 8 - LPX * NT;
            split = 8 * LPX + e / NT;
            if constexpr (MAP == 2) {
                // column-major order inside a spilled cohort (both row tiles of a column tile adjacent): an XCD's part
                // of the cohort then needs only its columns of G (the 75% of the bytes) and all of X (25%), so the
                // cross-XCD duplication is mostly X instead of G. Same (split, tile) set -> bitwise equal to MAP 1.
                const int tp = e % NT;
                tile = (tp & 1) * NTN + (tp >> 1);
            } else {
                tile = e % NT;
            }
        }
    }
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * 256;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);

    const uint32_t chunkA = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    const uint32_t voffA = (uint32_t)((8 * w + (lane >> 3)) * KS) + chunkA * 16u;
    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const uint32_t chunkB = (uint32_t)((lane & 15) ^ fB);
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) voffB[i] = (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + chunkB * 16u;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;

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

    auto issue = [&](int k, uint32_t st) {
        const i32x4 rA = make_rsrc(X + (long long)k * (KX * KS) + (long long)kx0 * KS, (uint32_t)(256 * KS));
        const int vt = min(KS, M - k * KS);
        const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), voffA, (uint32_t)(64 * i * KS));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
    };

    uint32_t pA0 = lpA0, pA1 = lpA1;
    uint32_t q0 = qB[0], q1 = qB[1], q2 = qB[2], q3 = qB[3];
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK3_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK3_SB();
        i32x8 bf[4];
        i32x8 a[2];
        auto mma = [&](int fa, int fb, const i32x8& av) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[fb], av, acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };
        bf[0] = fragB<0>(q0);
        a[0] = fragA<0>(pA0, pA1);
        bf[1] = fragB<0>(q1);
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        bf[2] = fragB<0>(q2);
        WGCK3_SB();
        mma(0, 0, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        bf[3] = fragB<0>(q3);
        WGCK3_SB();
        mma(0, 1, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        a[1] = fragA<2048>(pA0, pA1);
        WGCK3_SB();
        mma(0, 2, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        WGCK3_SB();
        mma(0, 3, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        WGCK3_SB();
        [&]<int... FB>(std::integer_sequence<int, FB...>) {
            (([&] {
                 constexpr int FA = FB + 1;
                 if constexpr (FA < 7) a[(FA + 1) & 1] = fragA<(FA + 1) * 2048>(pA0, pA1);
                 WGCK3_SB();
#pragma unroll
                 for (int fb = 0; fb < 4; ++fb) mma(FA, fb, a[FA & 1]);
                 WGCK3_SB();
                 asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                 WGCK3_SB();
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

    float* P = part + ((long long)split * KX + kx0 + 128 * (w >> 2) + r16) * N + n0 + 64 * (w & 3) + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < 8; ++fa)
#pragma unroll
        for (int fb = 0; fb < 4; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}


// ldm: any multiple of 16 >= roundup(M,16) (X chunks at token >= ldm are masked); 256*ldm must fit the 31-bit
// num_records so the 0x80000000 out-of-range offset stays out of range (ldm < 8388608 tokens per call; production ~3M).
static constexpr long long WGCK3_LDM_MAX = 0x7fffffffLL / 256;

template <int N, int TNW, int NSPLIT>
static int wgrad_ck_w3_launch(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                              const float* as, int M, void* stream) {
    if (M <= 0 || (ldm & 15) != 0 || ldm < ((M + 15) / 16) * 16LL || ldm > WGCK3_LDM_MAX)
        return (int)hipErrorInvalidValue;
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    constexpr int NT = 2 * (N / TNW);
    // The HIP last error is sticky per thread (HIP 7.2, as in CUDA): an error that an earlier, unrelated call left on
    // this thread would otherwise read as a failure of these launches. Drain it first, check each launch on its own,
    // and report a drained error as a negative code (both kernels launched; same contract as wgrad_tm_ck.cu, r14
    // iter38 review S1).
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_w3_kernel<N, TNW, NSPLIT>), dim3(NSPLIT * NT), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, ldm, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_w3_reduce<N, NSPLIT>), dim3((wgck3::KX * N / 4) / 256), dim3(256), 0,
                       (hipStream_t)stream, (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

// ABI (same signature as wgrad_tm_ck.cu's entries): G = y8 e4m3 [>=M, 1536] row-major, X = dout_t e4m3 [512, ldm],
// part fp32 [16, 512, 1536] scratch, out fp32 [512, 1536] (= fp8_wgrad_nn's dw_t), gs = scale_y, as = scale_d
// (device fp32 scalars), stream = hipStream_t. Returns 0 = both kernels launched; > 0 = the hipError_t of a rejected
// argument or a failed launch (out not written); < 0 = both kernels launched, and -rc is a stale error an earlier call
// had left on this thread (drained here, not caused by these launches).
// W3: out-proj wgrad, N=1536 (tile 256 x 192), NSPLIT=16 (2 cohorts of 16 WGs per XCD).
extern "C" int wgrad_ck_w3(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                           const float* as, int M, void* stream) {
    return wgrad_ck_w3_launch<1536, 192, 16>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_nsplit(void) { return 16; }

// Step-tiled X (B13): X = dout_t e4m3 [ceil(M/128)][512][128] (see wgrad_ck_w3_t_kernel); `ldm` must be that
// tensor's step stride 512*128 = 65536 (it is not a row pitch: the tiled kernel has no row stride). Same partials
// workspace, reduction, return codes and stale-error contract as wgrad_ck_w3_launch.
template <int N, int TNW, int NSPLIT>
static int wgrad_ck_w3_t_launch(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                const float* as, int M, void* stream) {
    if (M <= 0 || ldm != (long long)wgck3::KX * wgck3::KS) return (int)hipErrorInvalidValue;
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    constexpr int NT = 2 * (N / TNW);
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_w3_t_kernel<N, TNW, NSPLIT>), dim3(NSPLIT * NT), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_w3_reduce<N, NSPLIT>), dim3((wgck3::KX * N / 4) / 256), dim3(256), 0,
                       (hipStream_t)stream, (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

// Same ABI as wgrad_ck_w3 (X = the step-tiled dout_t, ldm = 65536); bitwise-identical output.
extern "C" int wgrad_ck_w3_t(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                             const float* as, int M, void* stream) {
    return wgrad_ck_w3_t_launch<1536, 192, 16>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t_nsplit(void) { return 16; }
// Drain (return and clear) this thread's sticky HIP last error: wgrad_ck_w3.py calls it before its own torch padding
// fill, so a stale error from an earlier unrelated call is reported once instead of failing that fill.
extern "C" int wgrad_ck_w3_drain_last_error(void) { return (int)hipGetLastError(); }

template <int MAP>
static int wgrad_ck_w3_t256s_launch(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                    const float* as, int M, void* stream) {
    constexpr int N = 1536, NS = 21;
    if (M <= 0 || ldm != (long long)wgck3::KX * wgck3::KS) return (int)hipErrorInvalidValue;
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_w3_t256s_kernel<N, NS, MAP>), dim3(NS * 12), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_w3_reduce<N, NS>), dim3((wgck3::KX * N / 4) / 256), dim3(256), 0, (hipStream_t)stream,
                       (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}
// Same ABI as wgrad_ck_w3_t (X step-tiled, ldm = 65536); part must be fp32 [21, 512, 1536].
extern "C" int wgrad_ck_w3_t256s(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                 const float* as, int M, void* stream) {
    return wgrad_ck_w3_t256s_launch<2>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t256s_nsplit(void) { return 21; }
extern "C" int wgrad_ck_w3_t256s_m1(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                    const float* as, int M, void* stream) {
    return wgrad_ck_w3_t256s_launch<1>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t256s_m1_nsplit(void) { return 21; }


// =====================================================================================================================
// Arbor r14 iteration 53, track erm_w3 (LAB SOURCE, NOT wired into any tree / gin): rccl DY85 route E-RM, r14's half.
// E-RM = the out-proj dY producer writes BOTH FP8 dout copies row-major (whole 512-B rows), which deletes the separate
// dual cast; W3 then reads its X operand (the e4m3 dout copy) ROW-MAJOR. This block adds that entry to x21:
//   wgrad_ck_w3_t256s_xt2_kernel<N=1536, NSPLIT=21, MAP=2, PF=1> + wgrad_ck_w3_reduce<1536, 21>, entry wgrad_ck_w3_t256s_xt2
// = wgrad_ck_w3_t256s_kernel<1536, 21, 2> (x21, above, UNCHANGED) with the B20 "XT=2" operand change (r14 iteration 47,
// probes/iter47/b20/src/wgrad_tm_ck_b20.cu: voffA2 :207-213, pX :228-243, DMA :258-275, step body :310-360, launcher
// :442-447) applied to the X (A) tile:
//   X : dout e4m3 ROW-MAJOR [>= M][512] (token rows of 512 B; the `ldm` argument is its row stride and must be 512).
//       Rows >= M are never read: the buffer range check returns them as 0x00 (out-of-range LDS-DMA lanes write zero,
//       measured for this instruction in B20 lab task 12 with a poisoned LDS), so the producer needs NO zero tail and
//       X may be exactly M rows long.
//   A (X) tile in LDS = the G-tile layout exactly: token row t at t*256 (the WG's 256 bytes kx0..kx0+255 of X row t),
//     16-byte chunk c at position c ^ f(t), f(t) = (t&7) | ((t>>4)&1)<<3; DMA'd like G (wave w, instr i, lane l -> LDS
//     A + 8192i + 1024w + 16l = token 32i + 4w + l/16, position l&15, source chunk (l&15) ^ fB) and read like G (fragB:
//     4 ds_read_b64_tr_b8 per fragment). Fragment fa of wave w = kx chunk 8*(w>>2) + fa, addressed as pX ^ (fa << 4).
//   Token tail: instruction i of a step covers token rows 32i + [0, 32) through its own buffer resource (base row
//     k*128 + 32i, num_records = max(0, vt - 32i) * 512 kept scalar through readfirstlane: no waterfall); the per-lane
//     offset voffA2 is one VGPR for all four instructions (the raw-buffer range check covers voffset + inst_offset, not
//     soffset, so the row offset sits in the resource base).
//   Everything else is x21's: MAP 2 spill map (same (split, tile) per WG), NSPLIT 21, the same [kb, ke) per split, the same
//   per-step MFMA order (fa, fb) = (0,0..3), ..., (7,0..3), the same G DMA / fragments, the same partial store and the same
//   fixed-order 21-way reduce. Every MFMA sees the same operand bytes in the same k slots (lane l: tokens 16g + [0,16) in
//   bytes 0-15 and 64 + 16g + [0,16) in bytes 16-31, kx row l&15; the XT=1 tail tokens are producer zeros, the XT=2 tail
//   reads as zero) -> BITWISE equal to wgrad_ck_w3_t256s on the same e4m3 values (X_rm[t][n] = dout_t_tiled[t/128][n][t%128]).
// Deviations from B20 (all deliberate, see probes/iter53/erm_w3/RESULT.md section 2):
//   D1 a separate kernel template instead of an XT template parameter + `if constexpr` + `continue` in the shared
//      kernel: x21's kernel text, mangled name and ISA stay byte-identical (B20 risk 4 removed by construction) and the
//      XT=2 step body is the loop's only body (no `continue`, single back-edge, as x21).
//   D2 G is e4m3 (FMT_G = 0, cbsz 0), N = 1536, NSPLIT 21, MAP 2: inherited from x21, not from W1.
//   D3 template parameter PF (A-fragment prefetch depth): PF = 1 is B20's step body verbatim (the E-RM entry);
//      PF = 2 is a LAB-ONLY variant arm (entry wgrad_ck_w3_t256s_xt2p2): A fragments fa+1 AND fa+2 in flight (a 3-deep
//      register ring, +8 VGPR), lgkmcnt(4) instead of lgkmcnt(0) inside the fa loop. Same reads, same MFMA order, same
//      operand registers per MFMA -> bitwise equal to PF = 1. Hypothesis under test: B20's +5.7% is A-fragment latency
//      exposed by the 1-deep prefetch (4 tr reads issued only 4 MFMAs before use, each group ends in lgkmcnt(0)).
//   D4 lab-only LDS poison / probe kernels renamed wgck3_lds_poison / wgck3_lds_probe (B20: wgck_lds_*): not for
//      production.
template <int N, int NSPLIT, int MAP, int PF>
__global__ __launch_bounds__(512, 2) void wgrad_ck_w3_t256s_xt2_kernel(const uint8_t* __restrict__ G,
                                                                       const uint8_t* __restrict__ X,
                                                                       float* __restrict__ part, int M) {
    using namespace wgck3;
    static_assert(PF == 1 || PF == 2, "PF: A-fragment prefetch depth 1 (B20) or 2 (lab variant)");
    constexpr int NTN = N / 256;
    constexpr int NT = 2 * NTN;
    constexpr int FMT_G = 0;
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;
    // (split, tile) map: x21's (wgrad_ck_w3_t256s_kernel), verbatim
    int split, tile;
    if constexpr (MAP == 0) {
        static_assert(NSPLIT % 8 == 0, "MAP 0 needs NSPLIT % 8 == 0");
        constexpr int CPX = NSPLIT / 8;
        split = xcd * CPX + loc / NT;
        tile = loc % NT;
    } else {
        constexpr int NWG = NSPLIT * NT;
        constexpr int LPX = 32 / NT;  // local cohorts per XCD
        static_assert(NWG <= 256 && NSPLIT >= 8 * LPX, "spill map");
        if (loc < LPX * NT) {
            split = xcd * LPX + loc / NT;
            tile = loc % NT;
        } else {
            int e = loc - LPX * NT;
            for (int x = 0; x < xcd; ++x) e += (NWG - x + 7) / 8 - LPX * NT;
            split = 8 * LPX + e / NT;
            if constexpr (MAP == 2) {
                const int tp = e % NT;
                tile = (tp & 1) * NTN + (tp >> 1);
            } else {
                tile = e % NT;
            }
        }
    }
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * 256;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);

    // G DMA offsets (x21, verbatim): wave w, instr i, lane l -> token 32i + 4w + l/16, position l&15, chunk (l&15) ^ fB.
    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const uint32_t chunkB = (uint32_t)((lane & 15) ^ fB);
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) voffB[i] = (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + chunkB * 16u;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;
    // X DMA (B20 voffA2): the G DMA with row stride KX instead of N; instruction i's 32i row offset lives in its buffer
    // base (see issue), so the per-lane offset is ONE VGPR for all four instructions.
    const uint32_t voffA2 = (uint32_t)((4 * w + (lane >> 4)) * KX) + chunkB * 16u;

    // fragment read addresses: G (x21, verbatim) and X (B20 pX: lane (s, g) -> token row t0 = 16g + (s>>1), byte half
    // s&1, position ca ^ f(t0), f(t0) = (s>>1) + 8*(g&1), ca = 8*(w>>2) + fa; fragB's immediates +2048 / +16384 / +18432
    // step t0 by 8 / 64 / 72 rows, where f is the same; pX ^ (fa << 4) flips exactly the position bits 4-6 because the row
    // base (lds0 4 KiB-aligned + t0*256) has zero bits 0-7 and 8*(s&1) + (pos << 4) < 256 does not carry)
    const int r16 = lane & 15, q4 = lane >> 4;
    uint32_t qB[4];
    uint32_t pX;
    {
        const int s = r16, g = q4;
        const int t0 = 16 * g + (s >> 1);
        const int f = (s >> 1) + 8 * (g & 1);
        const int cw = 4 * (w & 3);
        const uint32_t q = lds0 + (uint32_t)B_OFF + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) + ((uint32_t)(cw ^ f) << 4);
#pragma unroll
        for (int fb = 0; fb < 4; ++fb) qB[fb] = q ^ ((uint32_t)fb << 4);
        const int ca0 = 8 * (w >> 2);
        pX = lds0 + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) + ((uint32_t)(ca0 ^ f) << 4);
    }

    f32x4 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};

    // Stage toggle as in x21 (XOR 0x8000 flips stage 0 <-> 1 of both tiles).
    auto issue = [&](int k, uint32_t st) {
        // step k = X token rows [k*KS, k*KS + vt), this WG's 256 bytes kx0.. of each 512-byte row. DMA instruction i covers
        // rows 32i + [0, 32) through its own resource (base row k*KS + 32i, num_records max(0, vt - 32i)*KX): rows >= M
        // read as 0 (the range check covers voffset + inst_offset, not soffset).
        const int vt = min(KS, M - k * KS);
        const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            // num_records kept scalar explicitly (B20; S32 v1's max(vt - 64, 0) * N became a VGPR resource and a
            // readfirstlane waterfall around each DMA, r14 iter43 w3_s32v2.cu:848-851); vt is wave-uniform.
            const int vi = vt - 32 * i;
            const uint32_t nrA = (uint32_t)__builtin_amdgcn_readfirstlane((vi > 0 ? vi : 0) * KX);
            const i32x4 rA = make_rsrc(X + ((long long)k * KS + 32 * i) * KX + kx0, nrA);
            g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), voffA2, 0u);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
    };

    uint32_t q0 = qB[0], q1 = qB[1], q2 = qB[2], q3 = qB[3];
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK3_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK3_SB();
        i32x8 bf[4];
        auto mma = [&](int fa, int fb, const i32x8& av) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[fb], av, acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };
        if constexpr (PF == 1) {
            // B20 XT=2 step body, verbatim: x21's step with every fragA replaced by fragB on the X tile at pX ^ (fa << 4);
            // an A fragment is 4 transposing reads, so every wait counts 4 per A fragment: first MFMA after bf0 + a0
            // (8 reads), <= 12 LDS ops in flight (the lgkmcnt field holds 15).
            i32x8 a[2];
            bf[0] = fragB<0>(q0);
            a[0] = fragB<0>(pX);
            bf[1] = fragB<0>(q1);
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            bf[2] = fragB<0>(q2);
            WGCK3_SB();
            mma(0, 0, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            bf[3] = fragB<0>(q3);
            WGCK3_SB();
            mma(0, 1, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            a[1] = fragB<0>(pX ^ 16u);
            WGCK3_SB();
            mma(0, 2, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            WGCK3_SB();
            mma(0, 3, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            WGCK3_SB();
            [&]<int... FB>(std::integer_sequence<int, FB...>) {
                (([&] {
                     constexpr int FA = FB + 1;
                     if constexpr (FA < 7) a[(FA + 1) & 1] = fragB<0>(pX ^ ((uint32_t)(FA + 1) << 4));
                     WGCK3_SB();
#pragma unroll
                     for (int fb = 0; fb < 4; ++fb) mma(FA, fb, a[FA & 1]);
                     WGCK3_SB();
                     asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                     WGCK3_SB();
                 }()),
                 ...);
            }(std::make_integer_sequence<int, 7>{});
        } else {
            // PF = 2 (lab variant): the same reads and MFMAs in the same order; A fragment fa+2 is issued one group
            // earlier (3-deep ring a[fa % 3]) and each fa group waits only for fragment fa+1 (lgkmcnt(4): fragment fa+2's
            // 4 reads stay in flight). <= 12 LDS ops in flight.
            i32x8 a[3];
            bf[0] = fragB<0>(q0);
            a[0] = fragB<0>(pX);
            bf[1] = fragB<0>(q1);
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");  // bf0, a0
            bf[2] = fragB<0>(q2);
            WGCK3_SB();
            mma(0, 0, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");  // bf1
            bf[3] = fragB<0>(q3);
            WGCK3_SB();
            mma(0, 1, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");  // bf2
            a[1] = fragB<0>(pX ^ 16u);
            WGCK3_SB();
            mma(0, 2, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");  // bf3 (a1 in flight)
            a[2] = fragB<0>(pX ^ 32u);
            WGCK3_SB();
            mma(0, 3, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");  // a1 (a2 in flight)
            WGCK3_SB();
            [&]<int... FB>(std::integer_sequence<int, FB...>) {
                (([&] {
                     constexpr int FA = FB + 1;
                     if constexpr (FA + 2 < 8) a[(FA + 2) % 3] = fragB<0>(pX ^ ((uint32_t)(FA + 2) << 4));
                     WGCK3_SB();
#pragma unroll
                     for (int fb = 0; fb < 4; ++fb) mma(FA, fb, a[FA % 3]);
                     WGCK3_SB();
                     if constexpr (FA + 2 < 8) {
                         asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");  // a[FA+1] (a[FA+2] in flight)
                     } else {
                         asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                     }
                     WGCK3_SB();
                 }()),
                 ...);
            }(std::make_integer_sequence<int, 7>{});
        }
        st ^= 0x8000u;
        pX ^= 0x8000u;
        q0 ^= 0x8000u;
        q1 ^= 0x8000u;
        q2 ^= 0x8000u;
        q3 ^= 0x8000u;
    }

    // partial store (x21, verbatim): acc[fa][fb] lane l = dW[kx0 + 128(w>>2) + 16fa + (l&15)][n0 + 64(w&3) + 16fb + 4(l>>4)
    // + 0..3]; every WG stores its whole tile, also when its split has no K step (part fully overwritten every call)
    float* P = part + ((long long)split * KX + kx0 + 128 * (w >> 2) + r16) * N + n0 + 64 * (w & 3) + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < 8; ++fa)
#pragma unroll
        for (int fb = 0; fb < 4; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}

template <int MAP, int PF>
static int wgrad_ck_w3_t256s_xt2_launch(const void* G, const void* X, long long ldm, void* part, void* out,
                                        const float* gs, const float* as, int M, void* stream) {
    constexpr int N = 1536, NS = 21;
    // X row-major e4m3 [>= M][512]: ldm is its row stride, which must be 512 (rows >= M are never read; the
    // per-instruction buffer bases are 64-bit, voffset < 32*512, num_records <= 128*512: no 32-bit limit on M)
    if (M <= 0 || ldm != (long long)wgck3::KX) return (int)hipErrorInvalidValue;
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_w3_t256s_xt2_kernel<N, NS, MAP, PF>), dim3(NS * 12), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_w3_reduce<N, NS>), dim3((wgck3::KX * N / 4) / 256), dim3(256), 0, (hipStream_t)stream,
                       (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}
// E-RM entry: same ABI and return codes as wgrad_ck_w3_t256s, X = row-major e4m3 dout [>= M][512] with ldm = 512; part
// fp32 [21, 512, 1536]. Bitwise equal to wgrad_ck_w3_t256s on the same e4m3 values.
extern "C" int wgrad_ck_w3_t256s_xt2(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                     const float* as, int M, void* stream) {
    return wgrad_ck_w3_t256s_xt2_launch<2, 1>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t256s_xt2_nsplit(void) { return 21; }
// LAB-ONLY variant (D3, PF = 2): same ABI, same bits.
extern "C" int wgrad_ck_w3_t256s_xt2p2(const void* G, const void* X, long long ldm, void* part, void* out,
                                       const float* gs, const float* as, int M, void* stream) {
    return wgrad_ck_w3_t256s_xt2_launch<2, 2>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t256s_xt2p2_nsplit(void) { return 21; }

// ---- LAB ONLY (D4; B20 wgck_lds_poison / wgck_lds_probe renamed): the XT=2 tail relies on the LDS-DMA writing ZERO
// for an out-of-range lane. wgck3_lds_poison fills the whole 160 KiB LDS of the CUs it runs on with `pattern`
// (0x7F7F7F7F = e4m3 NaN bytes); a wgrad call right after it on the same stream then sees that pattern in any LDS byte
// its DMA does not write. wgck3_lds_probe counts the 32-bit LDS words that still hold `pattern` when it starts (it
// writes nothing first), i.e. whether LDS contents survive from one dispatch to the next (if not, the poison is vacuous).
__global__ __launch_bounds__(512) void wgck3_lds_poison_kernel(uint32_t pattern) {
    __shared__ __attribute__((aligned(16))) uint32_t pz[40960];  // 160 KiB: the whole gfx950 LDS, 1 WG per CU
    const uint32_t base = (uint32_t)reinterpret_cast<uintptr_t>(&pz[0]);
    for (int i = (int)threadIdx.x; i < 40960; i += 512)
        asm volatile("ds_write_b32 %0, %1" ::"v"(base + 4u * (uint32_t)i), "v"(pattern) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}
__global__ __launch_bounds__(512) void wgck3_lds_probe_kernel(uint32_t pattern, unsigned int* hits) {
    __shared__ __attribute__((aligned(16))) uint32_t pz[40960];
    const uint32_t base = (uint32_t)reinterpret_cast<uintptr_t>(&pz[0]);
    unsigned int n = 0;
    for (int i = (int)threadIdx.x; i < 40960; i += 512) {
        uint32_t v;
        asm volatile("ds_read_b32 %0, %1\n\ts_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(base + 4u * (uint32_t)i) : "memory");
        n += (v == pattern) ? 1u : 0u;
    }
    atomicAdd(hits, n);
}
extern "C" int wgck3_lds_poison(unsigned int pattern, int nwg, void* stream) {
    const hipError_t stale = hipGetLastError();
    (void)stale;
    hipLaunchKernelGGL(wgck3_lds_poison_kernel, dim3(nwg), dim3(512), 0, (hipStream_t)stream, (uint32_t)pattern);
    return (int)hipGetLastError();
}
extern "C" int wgck3_lds_probe(unsigned int pattern, int nwg, void* hits, void* stream) {
    const hipError_t stale = hipGetLastError();
    (void)stale;
    hipLaunchKernelGGL(wgck3_lds_probe_kernel, dim3(nwg), dim3(512), 0, (hipStream_t)stream, (uint32_t)pattern,
                       (unsigned int*)hits);
    return (int)hipGetLastError();
}

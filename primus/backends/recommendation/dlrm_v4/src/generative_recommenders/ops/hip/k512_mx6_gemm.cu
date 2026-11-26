// MXFP6 forward K=512 projection GEMMs of the HSTU layers (V/Q/K N=1536, targets V/K N=1024) and the fused MXFP6
// weight shadow, behind DLRM_HSTU_FP8_FWD_MXFP6 (default off; a DECLARED PRECISION CHANGE). Python side and dispatch:
// generative_recommenders/ops/hip/k512_mx6_gemm.py (U N=512 is track B's library, dispatched from there).
//
// Credit / provenance. The GEMM kernel is Arbor P7's (pr7) iteration-24 S1/S2 prototype k512mx6_kernel<N, WRONG=0,
// SC=1, TL=1> ("MX6-T" pre-tiled operands, scaled MFMA), /scratch/arbor_sessions/dlrmv4_20261007_pr7/probes/iter24/
// fp6fwd/proto/k512mx6.cu (sha256 a2cf3e5aa341), designed and prototyped by pr7 (DESIGN.md sections 3-9; layout model
// layout_check.py), itself a fork of the in-tree k512_fp8_gemm.cu (P7 iteration 9 "2wg"). Packaged for the tree by r14
// iteration 49 (fp6fwd/gemm track A): the diagnostic variants (row-major operands, unscaled MFMA, wrong-swizzle
// control) are removed; the mainloop / epilogue source is pr7's, statement for statement; the launch wrapper computes
// the tile count in 64 bit and reports a sticky error of an EARLIER HIP call separately; the weight-shadow kernel
// (second half of this file) is new.
//
// Numerics (pr7 DESIGN section 3; r14 iter49 integ DESIGN section 6):
//   A (activation) = MXFP6 E2M3 codes x6 + E8M0 scales xs, one per (row, 32 K)        ("MX6-T" + "MX6-A-v1")
//   B (weight)     = MXFP6 E2M3 codes w6 + E8M0 scales ws, one per 2-D 32x32 block     ("MX6-T" k512 perm + [N/32][q][S])
//   acc[m][n]      = sum_k dequant(A)[m][k] * dequant(B)[n][k]   fp32 in the MFMA (four in-order K=128 steps)
//   C[m][n]        = e4m3_rne( sat448( acc*s + bias[n] ) ),  s = sa*sb (one fp32 multiply), mul then add (no contraction)
//   The E8M0 bytes are RELATIVE to the per-tensor fp32 scales: the activation codes encode x*inv_scale (what the e4m3
//   path quantizes), the weight codes encode W^T/scale_b with scale_b the FP8 shadow's per-tensor scale.
//
// Geometry (pr7 DESIGN 4.3-4.7, section 9): WG = 4 waves, tile 128 (M) x 256 (N), 2 WGs per CU, 5-slot ring of 12 KiB
// slots (HEAD 128 rows x 64 B + TAIL 128 rows x 32 B), 3 LDS-DMA loads per lane per item (every load a linear 1 KiB
// copy of the pre-tiled operand), fragment = ds_read_b128 (head) + ds_read_b64 (tail), MFMA
// v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:2 blgp:2 issued operand-swapped (src A = weight: scale_a = its 2-D block byte,
// op_sel = K step; src B = activation: scale_b = its (row, K-block) byte, op_sel = 16-row subtile), epilogue = the
// k512_fp8_gemm NOCLAMP=1 epilogue (MODE.FP16_OVFL saturating converts), M-tail rows dropped by the store's buffer
// range.
//
// Operand layouts (all uint8, produced by ln_store_mx6 (x6, xs) and by k512_mx6_wshadow below (w6, ws)):
//   x6 [T][4][12288]  T = ceil(M/128); item (T, S) = the 12 KiB slot image of rows 128T..128T+127, K step S:
//                     HEAD row R's 16-B head of K-block q at 64R + 16*(q ^ ((R>>2)&3));
//                     TAIL row R's 8-B tail  of K-block q at 8192 + 32R + 8*(q ^ (2*((R>>3)&1)));
//                     the 24-B MX block of 32 codes: code f at bits [6f, 6f+6) little-endian; head = bytes 0..15.
//                     Rows >= M of the last tile are read (any content: they only feed dropped output rows).
//   xs [T][4][512]    byte (q*16 + r)*8 + m of item (T, S) = E8M0 of row 128T + 16m + r, K-block 4S + q.
//   w6 [N/256][2][4][12288]  item (tile tn, half h, S): slot row R = output column tn*256 + perm(R, h),
//                     perm(R, h) = 64(R>>5) + 16((R&15)>>2) + 4((R>>4)&1) + (R&3) + 8h  (k512 output-column permutation)
//   ws [N/32][4][4]   byte (nb*16 + q*4 + S) = E8M0 of output columns [32nb, +32) x K-block 4S + q.
//
// Build: hipcc --offload-arch=gfx950 -std=c++20 -O3 -ffp-contract=off -mllvm -amdgpu-sched-strategy=iterative-ilp
//        -shared -fPIC, with -DHK_N={1536|1024} for a GEMM library, or -DK512_MX6_WSHADOW for the weight-shadow
//        library (k512_mx6_gemm.py builds both kinds per process).
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

#if !defined(HK_N) && !defined(K512_MX6_WSHADOW)
#error "define HK_N (1024 or 1536) for a GEMM library, or K512_MX6_WSHADOW for the weight-shadow library"
#endif

#define K512MX6_STALE_ERROR_BASE 100000

#if defined(HK_N)
// =====================================================================================================================
// GEMM (pr7 iter24 k512mx6_kernel<N, 0, 1, 1>)
// =====================================================================================================================
// LDS ring switch (r14 iter64 RING6, ledger r14-b-mx6-a-prefetch; lab job 71320: GO_INMODEL, bitwise):
//   K512MX6_RING undefined or 5 = the production 5-slot ring, token for token (the #else branches below are the shipped
//   lines verbatim). K512MX6_RING=6 = RING6: 6 x 12 KiB slots (72 KiB per WG, 144 KiB per CU at 2 WGs) and the
//   activation item A of K step KS+2 issued in step KS (production: A(KS+1) in step KS). Only slot placement, DMA issue
//   points and vmcnt operands change; the MFMAs, every ds_read's logical operand, the scale operands, the accumulation
//   order and the epilogue are unchanged (slot map, waits and proofs: r14 probes/iter64/ring6/RING6.md).
#ifndef K512MX6_RING
#define K512MX6_RING 5
#endif
#if K512MX6_RING != 5 && K512MX6_RING != 6
#error "K512MX6_RING must be 5 (production ring) or 6 (RING6)"
#endif
// SECONDARY variant RING6S (-DK512MX6_RING=6 -DK512MX6_R6_SPIN=1): RING6 with the five scale loads pinned between the
// K-step-0 and K-step-1 prologue items (the compiler otherwise issues them after 17 of the 18 prologue DMAs, so its
// wait before the first MFMA also waits for A(1), B0(1) and two thirds of B1(1)). Same values, same variables.
#ifndef K512MX6_R6_SPIN
#define K512MX6_R6_SPIN 0
#endif
#if K512MX6_R6_SPIN && K512MX6_RING != 6
#error "K512MX6_R6_SPIN requires K512MX6_RING=6"
#endif
namespace k512mx6 {
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}  // namespace k512mx6

extern "C" __device__ void k512mx6_llvm_buffer_load_lds(k512mx6::i32x4 rsrc, k512mx6::lds_ptr_t lds, int size,
                                                        int voffset, int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
extern "C" __device__ void k512mx6_llvm_buffer_store_v4i32(k512mx6::i32x4 vdata, k512mx6::i32x4 rsrc, int voffset,
                                                           int soffset, int aux)
    __asm("llvm.amdgcn.raw.buffer.store.v4i32");

namespace k512mx6 {

constexpr int KD = 512;      // logical K
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int KSTEPS = 4;
#if K512MX6_RING == 6
constexpr int NSLOT = 6;     // RING6
#else
constexpr int NSLOT = 5;
#endif
constexpr int SLOT = 12288;  // HEAD 8192 + TAIL 4096
constexpr int TAILOFF = 8192;
#if K512MX6_RING == 6
// RING6 item -> slot map, K step 0..3 (B0 / B1 = weight halves, A = activation). Prologue: B0(0) B1(0) A(0) A(1) B0(1)
// B1(1) -> slots 0..5. Step KS < 2 issues A(KS+2) into the slot B0(KS) frees at the step's second barrier, and B0(KS+2)
// / B1(KS+2) into the slots B1(KS) / A(KS) free at its third barrier. A never uses slot 5: its half-1 head reads would
// need the ds_read immediate 5*12288 + 7*1024 = 68608 > 65535 (ds_read16 static_assert).
constexpr int R6_SB0[KSTEPS] = {0, 4, 1, 5};
constexpr int R6_SB1[KSTEPS] = {1, 5, 2, 3};
constexpr int R6_SA[KSTEPS] = {2, 3, 0, 4};
static_assert(((1 << R6_SB0[0]) | (1 << R6_SB1[0]) | (1 << R6_SA[0]) | (1 << R6_SA[1]) | (1 << R6_SB0[1]) |
               (1 << R6_SB1[1])) == 63,
              "RING6: the prologue fills six distinct slots");
static_assert(R6_SA[2] == R6_SB0[0] && R6_SA[3] == R6_SB0[1],
              "RING6: A(KS+2) refills the slot B0(KS) frees at the second barrier of step KS");
static_assert(R6_SB0[2] == R6_SB1[0] && R6_SB1[2] == R6_SA[0] && R6_SB0[3] == R6_SB1[1] && R6_SB1[3] == R6_SA[1],
              "RING6: B0(KS+2) / B1(KS+2) refill the slots B1(KS) / A(KS) free at the third barrier of step KS");
static_assert(R6_SA[0] < 5 && R6_SA[1] < 5 && R6_SA[2] < 5 && R6_SA[3] < 5, "RING6: A items never use slot 5");
#endif

// gfx950 buffer resource: base (48 bit), stride 0, num_records (bytes), dword3 as k512_fp8_gemm.cu.
__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}

#define K512MX6_FENCE() asm volatile("" ::: "memory")
#define K512MX6_SB() __builtin_amdgcn_sched_barrier(0)
#define K512MX6_BARRIER()             \
    do {                              \
        K512MX6_FENCE();              \
        __builtin_amdgcn_s_barrier(); \
        K512MX6_FENCE();              \
        K512MX6_SB();                 \
    } while (0)

template <int IMM>
__device__ __forceinline__ i32x4 ds_read16(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x4 v;
    asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
template <int IMM>
__device__ __forceinline__ i32x2 ds_read8(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x2 v;
    asm volatile("ds_read_b64 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}

// One FP6 16x128 fragment: dwords 0..3 = head (ds_read_b128), dwords 4..5 = tail (ds_read_b64), dwords 6..7 undefined
// (ignored by the cbsz/blgp=2 MFMA; x4c19 iter52 mfmamap).
template <int SH, int ST>
__device__ __forceinline__ i32x8 frag6(uint32_t lpH, uint32_t lpT) {
    const i32x4 lo = ds_read16<SH>(lpH);
    const i32x2 hi = ds_read8<ST>(lpT);
    const i32x4 hi4 = __builtin_shufflevector(hi, hi, 0, 1, -1, -1);
    return __builtin_shufflevector(lo, hi4, 0, 1, 2, 3, 4, 5, -1, -1);
}
// A subtiles s = 4H .. 4H+3 (16 rows each): HEAD s*1024, TAIL s*512
template <int SL, int H>
__device__ __forceinline__ void read_A(i32x8 (&a)[4], uint32_t lpH, uint32_t lpT) {
    a[0] = frag6<SL * SLOT + (4 * H + 0) * 1024, SL * SLOT + (4 * H + 0) * 512>(lpH, lpT);
    a[1] = frag6<SL * SLOT + (4 * H + 1) * 1024, SL * SLOT + (4 * H + 1) * 512>(lpH, lpT);
    a[2] = frag6<SL * SLOT + (4 * H + 2) * 1024, SL * SLOT + (4 * H + 2) * 512>(lpH, lpT);
    a[3] = frag6<SL * SLOT + (4 * H + 3) * 1024, SL * SLOT + (4 * H + 3) * 512>(lpH, lpT);
}
// B fragments: slot rows [32w + 16jj, +16) (wave offsets 2048w / 1024w are inside lpBH / lpBT)
template <int SL>
__device__ __forceinline__ void read_B(i32x8 (&b)[2], uint32_t lpBH, uint32_t lpBT) {
    b[0] = frag6<SL * SLOT + 0, SL * SLOT + 0>(lpBH, lpBT);
    b[1] = frag6<SL * SLOT + 1024, SL * SLOT + 512>(lpBH, lpBT);
}

// v_mfma_scale_f32_16x16x128_f8f6f4, E2M3 x E2M3 (cbsz:2 blgp:2); src A = weight fragment, scale_a = its 2-D block
// E8M0 byte, op_sel = K step KS; src B = activation fragment, scale_b = its (row, K-block) byte, op_sel = subtile M.
template <int KS, int M, bool FIRST>
__device__ __forceinline__ f32x4 mf(const i32x8& b, const i32x8& a, const f32x4& c, int sw, int sa) {
    const f32x4 z = {0.f, 0.f, 0.f, 0.f};
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b, a, FIRST ? z : c, 2, 2, KS, sw, M, sa);
}
// acc[n][m] (+)= Bfrag[n] x Afrag[m]  (k512_fp8_gemm.cu order: n outer, m inner); sa = scale dword of this half
template <int KS, bool FIRST>
__device__ __forceinline__ void mma8(f32x4 (&c)[2][4], const i32x8 (&b)[2], const i32x8 (&a)[4], int sw, int sa) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        c[n][0] = mf<KS, 0, FIRST>(b[n], a[0], c[n][0], sw, sa);
        c[n][1] = mf<KS, 1, FIRST>(b[n], a[1], c[n][1], sw, sa);
        c[n][2] = mf<KS, 2, FIRST>(b[n], a[2], c[n][2], sw, sa);
        c[n][3] = mf<KS, 3, FIRST>(b[n], a[3], c[n][3], sw, sa);
    }
}

__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    k512mx6_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, 0);
}
// item (A or B0/B1) of K step KK -> slot SL: 3 x 16 B per lane; per-lane voffsets carry the slot-image offset, soffset
// the K step (one 12 KiB item per K step of the pre-tiled operand)
template <int SL, int KK>
__device__ __forceinline__ void issue(i32x4 r, uint32_t ldsw, const uint32_t (&off)[3]) {
#pragma unroll
    for (int i = 0; i < 3; ++i) g2l(r, ldsw + SL * SLOT + i * 4096, off[i], KK * SLOT);
}

}  // namespace k512mx6

template <int N>
__global__ __launch_bounds__(256, 2) void k512_mx6_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                          uint8_t* __restrict__ C, const uint16_t* __restrict__ bias,
                                                          const float* __restrict__ sa, const float* __restrict__ sb,
                                                          const uint8_t* __restrict__ ascale,
                                                          const uint8_t* __restrict__ wscale, int M) {
#pragma clang fp contract(off)
    using namespace k512mx6;
    static_assert(N % BN == 0, "N must be a multiple of 256");
    constexpr int NT = N / BN;
    __shared__ __attribute__((aligned(2048))) uint8_t smem[NSLOT * SLOT];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));

    int bid = (int)blockIdx.x;
    {  // XCD remap (k512_fp8_gemm.cu): XCD x gets a contiguous tile-id range
        const int G_ = (int)gridDim.x;
        const int per = (G_ + 7) / 8;
        int tall = G_ % 8;
        if (tall == 0) tall = 8;
        const int xcd = bid % 8, loc = bid / 8;
        bid = (xcd < tall) ? xcd * per + loc : tall * per + (xcd - tall) * (per - 1) + loc;
    }
    const int tile_m = bid / NT, tile_n = bid % NT;
    const int row0 = tile_m * BM, col0 = tile_n * BN;
    const int rows_valid = min(BM, M - row0);

    const float sav = sa[0], sbv = sb[0];
    // MODE.FP16_OVFL (bit 23): the fp8 converts saturate to +-448 (k512_fp8_gemm.cu NOCLAMP=1)
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n s_nop 7\n s_nop 7" ::: "memory");

#if K512MX6_R6_SPIN
    // RING6S: the same two scale loads, issued by load_scales() at the pin point in the prologue (below)
    i32x2 sA[KSTEPS];
    int sW;
    auto load_scales = [&]() {
        const i32x2* ap = reinterpret_cast<const i32x2*>(ascale + (size_t)tile_m * (KSTEPS * 512)) + lane;
#pragma unroll
        for (int s = 0; s < KSTEPS; ++s) sA[s] = ap[s * 64];
        sW = reinterpret_cast<const int*>(wscale)[(col0 / 32 + 2 * w + ((lane & 15) >> 3)) * 4 + (lane >> 4)];
    };
#else
    // ---- scales (issued first: oldest VMEM ops; the compiler waits for them before the first MFMA) ----
    // A: 2 dwords per K step at byte 8*lane of the (tile, S) block; dword H = subtiles 4H..4H+3 (op_sel = m)
    i32x2 sA[KSTEPS];
    {
        const i32x2* ap = reinterpret_cast<const i32x2*>(ascale + (size_t)tile_m * (KSTEPS * 512)) + lane;
#pragma unroll
        for (int s = 0; s < KSTEPS; ++s) sA[s] = ap[s * 64];
    }
    // W: [N/32][q][S]: lane's 32-column block = col0/32 + 2w + ((lane&15)>>3), q = lane>>4; byte S = K step (op_sel = S)
    const int sW = reinterpret_cast<const int*>(wscale)[(col0 / 32 + 2 * w + ((lane & 15) >> 3)) * 4 + (lane >> 4)];
#endif

    // ---- buffer resources: one pre-tiled item per K step; the producer pads the last row tile ----
    const i32x4 rA = make_rsrc(A + (size_t)tile_m * (KSTEPS * SLOT), (uint32_t)(KSTEPS * SLOT));
    const i32x4 rB0 = make_rsrc(B + (size_t)(2 * tile_n + 0) * (KSTEPS * SLOT), (uint32_t)(KSTEPS * SLOT));
    const i32x4 rB1 = make_rsrc(B + (size_t)(2 * tile_n + 1) * (KSTEPS * SLOT), (uint32_t)(KSTEPS * SLOT));

    // ---- per-lane DMA source offsets: every LDS-DMA instruction is a linear 1 KiB copy of the slot image ----
    uint32_t offA[3], offB[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) offA[i] = offB[i] = (uint32_t)(i * 4096 + w * 1024 + lane * 16);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);
    const uint32_t ldsw = lds0 + (uint32_t)w * 1024u;

    // ---- per-lane LDS read offsets (pr7 DESIGN 4.5) ----
    const int r16 = lane & 15, q4 = lane >> 4;
    const uint32_t hsw = (uint32_t)((r16 >> 2) & 3);
    const uint32_t tsw = (uint32_t)(2 * ((r16 >> 3) & 1));
    const uint32_t lpH = lds0 + (uint32_t)r16 * 64u + 16u * ((uint32_t)q4 ^ hsw);
    const uint32_t lpT = lds0 + (uint32_t)TAILOFF + (uint32_t)r16 * 32u + 8u * ((uint32_t)q4 ^ tsw);
    const uint32_t lpBH = lpH + (uint32_t)w * 2048u;
    const uint32_t lpBT = lpT + (uint32_t)w * 1024u;

    i32x8 a[4], b0[2], b1[2];
    f32x4 acc[4][2][4];
    i32x4 bq0, bq1;

    auto bias_prefetch = [&]() {  // k512_fp8_gemm.cu: raw bf16 bias of this lane's 16 columns
        const uint16_t* bp = bias + col0 + 64 * w + 16 * (lane >> 4);
        bq0 = *reinterpret_cast<const i32x4*>(bp);
        bq1 = *reinterpret_cast<const i32x4*>(bp + 8);
    };

#if K512MX6_RING == 6
    // RING6: the DMAs issued after the last item a step needs are A, B0, B1 of the step after it = 9 (step 0: the 9
    // prologue DMAs of K step 1 plus any compiler VMEM load issued after A(0)'s; step 3: none -> vmcnt(0))
    auto step0 = [&]<int KS>() {
        constexpr int sB0 = R6_SB0[KS], sB1 = R6_SB1[KS], sAs = R6_SA[KS];
        if constexpr (KS < KSTEPS - 1) asm volatile("s_waitcnt vmcnt(9)" ::: "memory");
        else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#else
    // SCHED 0 (k512_fp8_gemm.cu) with 3-load items: younger loads at a step start = B0/B1 of the next step = 6
    auto step0 = [&]<int KS>() {
        constexpr int sB0 = (3 * KS) % NSLOT, sB1 = (3 * KS + 1) % NSLOT, sAs = (3 * KS + 2) % NSLOT;
        if constexpr (KS < KSTEPS - 1) asm volatile("s_waitcnt vmcnt(6)" ::: "memory");
        else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
        K512MX6_BARRIER();
        read_B<sB0>(b0, lpBH, lpBT);
        read_A<sAs, 0>(a, lpH, lpT);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        K512MX6_BARRIER();
#if K512MX6_RING == 6
        if constexpr (KS + 2 < KSTEPS) issue<R6_SA[KS + 2], KS + 2>(rA, ldsw, offA);  // A two K steps ahead
#else
        if constexpr (KS + 1 < KSTEPS) issue<sB0, KS + 1>(rA, ldsw, offA);
#endif
        if constexpr (KS == KSTEPS - 1) bias_prefetch();
        K512MX6_SB();
        __builtin_amdgcn_s_setprio(1);
        mma8<KS, KS == 0>(acc[0], b0, a, sW, sA[KS].x);
        __builtin_amdgcn_s_setprio(0);
        K512MX6_SB();
        read_B<sB1>(b1, lpBH, lpBT);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        K512MX6_SB();
        __builtin_amdgcn_s_setprio(1);
        mma8<KS, KS == 0>(acc[1], b1, a, sW, sA[KS].x);
        __builtin_amdgcn_s_setprio(0);
        K512MX6_SB();
        read_A<sAs, 1>(a, lpH, lpT);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        K512MX6_BARRIER();
        if constexpr (KS + 2 < KSTEPS) {
#if K512MX6_RING == 6
            issue<R6_SB0[KS + 2], KS + 2>(rB0, ldsw, offB);  // = sB1 (freed above)
            issue<R6_SB1[KS + 2], KS + 2>(rB1, ldsw, offB);  // = sAs (freed above)
#else
            issue<sB1, KS + 2>(rB0, ldsw, offB);
            issue<sAs, KS + 2>(rB1, ldsw, offB);
#endif
        }
        K512MX6_SB();
        __builtin_amdgcn_s_setprio(1);
        mma8<KS, KS == 0>(acc[2], b0, a, sW, sA[KS].y);
        mma8<KS, KS == 0>(acc[3], b1, a, sW, sA[KS].y);
        __builtin_amdgcn_s_setprio(0);
        K512MX6_SB();
    };

#if K512MX6_RING == 6
    issue<R6_SB0[0], 0>(rB0, ldsw, offB);  // slot 0
    issue<R6_SB1[0], 0>(rB1, ldsw, offB);  // slot 1
    issue<R6_SA[0], 0>(rA, ldsw, offA);    // slot 2
#if K512MX6_R6_SPIN
    K512MX6_SB();  // RING6S: scale loads after the K-step-0 items, before the K-step-1 items
    load_scales();
    K512MX6_SB();
#endif
    issue<R6_SA[1], 1>(rA, ldsw, offA);    // slot 3
    issue<R6_SB0[1], 1>(rB0, ldsw, offB);  // slot 4
    issue<R6_SB1[1], 1>(rB1, ldsw, offB);  // slot 5
#else
    issue<0, 0>(rB0, ldsw, offB);
    issue<1, 0>(rB1, ldsw, offB);
    issue<2, 0>(rA, ldsw, offA);
    issue<3, 1>(rB0, ldsw, offB);
    issue<4, 1>(rB1, ldsw, offB);
#endif
    step0.template operator()<0>();
    step0.template operator()<1>();
    step0.template operator()<2>();
    step0.template operator()<3>();

    // ---- epilogue: k512_fp8_gemm.cu NOCLAMP=1 form, verbatim ----
    const float s = sav * sbv;
    const f32x2 s2 = {s, s};
    f32x2 bias2[8];
    {
        const int bw[8] = {bq0.x, bq0.y, bq0.z, bq0.w, bq1.x, bq1.y, bq1.z, bq1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t d = (uint32_t)bw[i];
            bias2[i] = f32x2{__uint_as_float(d << 16), __uint_as_float(d & 0xffff0000u)};
        }
    }
    const uint32_t lane_off = (uint32_t)(lane & 15) * (uint32_t)N + (uint32_t)(64 * w + 16 * (lane >> 4));
    uint8_t* Ct = C + (size_t)row0 * N + col0;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const int h = g >> 2, ms = g & 3;
        const f32x4 v[4] = {acc[2 * h][0][ms], acc[2 * h][1][ms], acc[2 * h + 1][0][ms], acc[2 * h + 1][1][ms]};
        i32x4 d;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            f32x2 lo = __builtin_shufflevector(v[j], v[j], 0, 1);
            f32x2 hi = __builtin_shufflevector(v[j], v[j], 2, 3);
            lo = lo * s2;
            hi = hi * s2;
            lo = lo + bias2[2 * j];
            hi = hi + bias2[2 * j + 1];
            int pk = __builtin_amdgcn_cvt_pk_fp8_f32(lo.x, lo.y, __builtin_bit_cast(int, lo.x), false);
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(hi.x, hi.y, pk, true);
            d[j] = pk;
        }
        int rv = rows_valid - 16 * g;
        rv = rv < 0 ? 0 : (rv > 16 ? 16 : rv);
        const uint32_t nrec = (uint32_t)__builtin_amdgcn_readfirstlane(rv * N);
        const i32x4 rC = make_rsrc(Ct + (size_t)(16 * g) * N, nrec);
        k512mx6_llvm_buffer_store_v4i32(d, rC, (int)lane_off, 0, 0);
        K512MX6_SB();
    }
}

template <int N>
static int k512_mx6_launch(const void* x6, const void* xs, const void* w6, const void* ws, void* out,
                           const void* bias, const float* sa, const float* sb, int M, void* stream) {
    if (M <= 0 || !x6 || !xs || !w6 || !ws || !out || !bias || !sa || !sb) return (int)hipErrorInvalidValue;
    const long long tiles = (((long long)M + 127) / 128) * (N / 256);
    if (tiles > 0x7fffffffLL) return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();  // a sticky error left by an EARLIER call is reported, not hidden
    hipLaunchKernelGGL((k512_mx6_kernel<N>), dim3((unsigned)tiles), dim3(256), 0, (hipStream_t)stream,
                       (const uint8_t*)x6, (const uint8_t*)w6, (uint8_t*)out, (const uint16_t*)bias, sa, sb,
                       (const uint8_t*)xs, (const uint8_t*)ws, M);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) return (int)err;
    if (stale != hipSuccess) return K512MX6_STALE_ERROR_BASE + (int)stale;
    return 0;
}

#define K512MX6_CAT_(a, b) a##b
#define K512MX6_CAT(a, b) K512MX6_CAT_(a, b)

// C ABI (same argument order as track B's k512_mx6_u_bf16_bias):
//   out e4m3 [M, N] = e4m3_rne(sat448(acc * (sa[0] * sb[0]) + float(bias[n]))); rows >= M are not written.
//   Returns 0, this launch's hipError_t, or 100000 + a sticky error an EARLIER HIP call left on the thread.
extern "C" int K512MX6_CAT(k512_mx6_e4m3_bias_N, HK_N)(const void* x6, const void* xs, const void* w6, const void* ws,
                                                       void* out, const void* bias, const float* sa, const float* sb,
                                                       int M, void* stream) {
    return k512_mx6_launch<HK_N>(x6, xs, w6, ws, out, bias, sa, sb, M, stream);
}
extern "C" const char* k512_mx6_gemm_impl(void) {
#if K512MX6_RING == 6
    return "k512mx6 (pr7 iter24 S1/S2, MX6-T pre-tiled, scaled MFMA cbsz:2 blgp:2): 2 x 4-wave WG/CU, 128x256 tile, "
           "RING6 6-slot 72 KiB LDS ring (A two K steps ahead"
#if K512MX6_R6_SPIN
           "; RING6S: scale loads pinned after the K-step-0 items"
#endif
           "), swapped MFMA, FP16_OVFL sat e4m3 epilogue, XCD remap";
#else
    return "k512mx6 (pr7 iter24 S1/S2, MX6-T pre-tiled, scaled MFMA cbsz:2 blgp:2): 2 x 4-wave WG/CU, 128x256 tile, "
           "5-slot 60 KiB LDS ring, swapped MFMA, FP16_OVFL sat e4m3 epilogue, XCD remap";
#endif
}
#endif  // HK_N

#if defined(K512_MX6_WSHADOW)
// =====================================================================================================================
// Fused MXFP6 weight shadow (r14 iter49 fp6fwd/gemm track A). ONE launch per refresh turns the weight slice into
//   sb = the FP8 weight shadow's per-tensor scale, bitwise (triton_fp8._get_weight_shadow -> calibrated_scale(margin
//        1.0): amax = max|w| over the packed slice (NaN if any NaN), clamp_min(1e-12), "* 1.0 / 448.0" which aten runs
//        as a multiply by fp32(1/448) (div_true_kernel_cuda with a CPU scalar; see triton_fp8_wshadow.py)),
//   y  = fp32(w) / sb  (IEEE division: the value the e4m3 shadow quantizes, quantize_with_scale),
//   w6 / ws = pr7 proto.py quant_W(y^T) + tile_W, byte for byte: E8M0 of each 2-D 32x32 block = clamp(ceil(log2(
//        amax_b / 7.5)), -127, 127) (all-zero block: 0), codes = RNE onto the E2M3 grid of y / 2^e, sign only for a
//        nonzero code; packed into the MX6-T k512 weight layout (header) and ws [N/32][q][S].
// Out of the reference's domain (documented, not bitwise): a block with an Inf / NaN y gets E8M0 byte 255 (NaN, so the
// GEMM output is NaN, not garbage); proto's e8m0_exp on fp32-SUBNORMAL block maxima (amax_b < 2^-126, i.e. |w| below
// ~1e-42 relative to scale_b) depends on the device's denormal mode, this kernel returns the exact clamp (-127).
//
// Grid-wide dependency without co-residency assumptions: every workgroup takes a ticket (atomic counter). Tickets
// [0, P1) reduce max|w| over 512/P1 rows each and publish it (atomic max + arrival count) WITHOUT waiting; tickets
// >= P1 (N/16 of them, 4 waves = 4 (64-column, 32-K) items each) load their 32 values per lane, then wait for the P1
// arrivals. A waiting workgroup only waits for workgroups that took a lower ticket, i.e. that are already running and
// never wait themselves, so no grid co-residency is needed. The last phase-2 workgroup to read the result resets the
// 16-byte workspace to zero, so one zero-initialized workspace per (device, stream) serves every later refresh.
// =====================================================================================================================
namespace k512mx6w {
constexpr int K = 512;
constexpr int P1 = 64;            // phase-1 workgroups (k512_mx6_gemm.py WSHADOW_P1)
constexpr int ROWS1 = K / P1;     // weight rows (K) per phase-1 workgroup
constexpr int SLOT = 12288;
constexpr int TAILOFF = 8192;

struct Ranges {
    int s[4];  // first source column of range r
    int w[4];  // width of range r (0 for unused ranges)
    int nr;
};

__device__ __forceinline__ uint32_t abs_bits(float v) { return __float_as_uint(v) & 0x7fffffffu; }
__device__ __forceinline__ uint32_t umax32(uint32_t a, uint32_t b) { return a > b ? a : b; }

template <bool BF16>
__device__ __forceinline__ float ld_w(const void* p, long long idx) {
    if constexpr (BF16) {
        const uint16_t h = reinterpret_cast<const uint16_t*>(p)[idx];
        return __uint_as_float((uint32_t)h << 16);
    } else {
        return reinterpret_cast<const float*>(p)[idx];
    }
}

// E8M0 byte of a block whose max |y| has the bit pattern bm (bits of a non-negative float; NaN sorts above +Inf)
__device__ __forceinline__ int e8m0_byte(uint32_t bm) {
    if (bm == 0u) return 127;            // all-zero block: e = 0 (proto e8m0_exp: amax 0 -> 0)
    if (bm >= 0x7f800000u) return 255;   // Inf / NaN in the block: E8M0 NaN (outside the reference's domain)
    if (bm < 0x00800000u) return 0;      // subnormal amax: ceil(log2(amax / 7.5)) < -127 -> clamp -127
    // amax = g * 2^x, g in [1, 2): smallest e with amax / 2^e <= 7.5 = 1.875 * 2^2 is x - 2 (g <= 1.875) or x - 1
    const int x = (int)(bm >> 23) - 127;
    int e = x - 2 + (((bm & 0x7fffffu) > 0x700000u) ? 1 : 0);
    e = e < -127 ? -127 : e;             // x <= 127 -> e <= 126: no upper clamp needed
    return e + 127;
}

// 6-bit E2M3 code of y / 2^e (pr7 proto q_e2m3 + enc_e2m3); inv2e = 2^-e (normal for e in [-127, 126])
__device__ __forceinline__ uint32_t e2m3_code(float y, float inv2e) {
    const float v = y * inv2e;            // exact power-of-two scaling (== proto's y / exp2(e))
    float a = fabsf(v);
    a = a > 7.5f ? 7.5f : a;              // proto clamps to [-7.5, 7.5] (a no-op for finite blocks)
    uint32_t c;
    if (a < 2.0f) c = (uint32_t)rintf(a * 8.0f);          // step 1/8: codes 0..16 (16 = 2.0 = e2 m0)
    else if (a < 4.0f) c = (uint32_t)rintf(a * 4.0f) + 8u;   // step 1/4: q = k/4, code = k + 8 (k 8..16)
    else c = (uint32_t)rintf(a * 2.0f) + 16u;                // step 1/2: q = k/2, code = k + 16 (k 8..15)
    if (!(a == a)) c = 0u;                // NaN (the block's E8M0 is 255 anyway)
    return (c != 0u && v < 0.0f) ? (c | 0x20u) : c;          // copysign(q, v); (q < 0) is false for q = +-0
}

template <bool BF16>
__global__ __launch_bounds__(256) void k512_mx6_wshadow_kernel(const void* __restrict__ weight, long long ld_k,
                                                               long long ld_n, int n_total, Ranges rg, int perm,
                                                               uint8_t* __restrict__ w6, uint8_t* __restrict__ ws,
                                                               float* __restrict__ sb, uint32_t* __restrict__ work) {
#pragma clang fp contract(off)
    __shared__ uint32_t s_ticket, s_amax, s_red[4];
    const int tid = (int)threadIdx.x, lane = tid & 63, wv = tid >> 6;
    if (tid == 0) s_ticket = __hip_atomic_fetch_add(&work[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    const uint32_t t = s_ticket;

    if (t < (uint32_t)P1) {
        // ---------------- phase 1: max |w| over rows [t*ROWS1, +ROWS1) x every column of the slice ----------------
        // lane j-loop over the slice's columns (coalesced across lanes), the ROWS1 rows unrolled: ROWS1 independent
        // loads per iteration, two iterations unrolled (n_total / 256 = 2..6 iterations for the production layouts)
        uint32_t m = 0u;
        const int k0 = (int)t * ROWS1;
#pragma unroll 2
        for (int j = tid; j < n_total; j += 256) {
            int col = 0, rem = j;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (rem >= 0 && rem < rg.w[r]) col = rg.s[r] + rem;
                rem -= rg.w[r];
            }
            const long long cb = (long long)col * ld_n;
#pragma unroll
            for (int kk = 0; kk < ROWS1; ++kk) m = umax32(m, abs_bits(ld_w<BF16>(weight, (long long)(k0 + kk) * ld_k + cb)));
        }
#pragma unroll
        for (int o = 32; o > 0; o >>= 1) m = umax32(m, (uint32_t)__shfl_xor((int)m, o));
        if (lane == 0) s_red[wv] = m;
        __syncthreads();
        if (tid == 0) {
            const uint32_t mm = umax32(umax32(s_red[0], s_red[1]), umax32(s_red[2], s_red[3]));
            __hip_atomic_fetch_max(&work[3], mm, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_fetch_add(&work[1], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        return;
    }

    // ---------------- phase 2: wave = one (64 output columns, 32-K block) item; lane = one output column n ----------
    const int gw = (int)(t - (uint32_t)P1) * 4 + wv;  // < n_total / 4
    const int kb = gw & 15;                            // K-block: k in [32kb, 32kb + 32); S = kb >> 2, q = kb & 3
    const int n = (gw >> 4) * 64 + lane;               // output column of the shadow (row of W^T)
    int col = 0, rem = n;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (rem >= 0 && rem < rg.w[r]) col = rg.s[r] + rem;
        rem -= rg.w[r];
    }
    float v[32];
    const long long cb = (long long)col * ld_n;
#pragma unroll
    for (int f = 0; f < 32; ++f) v[f] = ld_w<BF16>(weight, (long long)(32 * kb + f) * ld_k + cb);

    if (tid == 0) {
        while (__hip_atomic_load(&work[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < (uint32_t)P1)
            __builtin_amdgcn_s_sleep(1);
        s_amax = __hip_atomic_load(&work[3], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const uint32_t p2 = (uint32_t)(n_total >> 4);
        const uint32_t done = __hip_atomic_fetch_add(&work[2], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (done == p2 - 1u) {  // every ticket is taken, every arrival counted, every phase-2 WG has read the max
            __hip_atomic_store(&work[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&work[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&work[2], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&work[3], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    __syncthreads();

    // scale_b = calibrated_scale(slice, margin=1.0): clamp_min(amax, 1e-12) (NaN kept) * fp32(1/448)
    const float amax = __uint_as_float(s_amax);
    const float amc = (amax < 1.0e-12f) ? 1.0e-12f : amax;
    const float sbv = amc * (1.0f / 448.0f);
    if (t == (uint32_t)P1 && tid == 0) sb[0] = sbv;

    uint32_t bm = 0u;
#pragma unroll
    for (int f = 0; f < 32; ++f) {
        v[f] = v[f] / sbv;  // IEEE fp32 division, as quantize_with_scale's (w^T / scale)
        bm = umax32(bm, abs_bits(v[f]));
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) bm = umax32(bm, (uint32_t)__shfl_xor((int)bm, o));  // 32 columns of the 2-D block
    const int eb = e8m0_byte(bm);
    const int e = eb - 127;
    const float inv2e = __uint_as_float((uint32_t)(127 - (e < -127 ? -127 : (e > 126 ? 126 : e))) << 23);

    uint32_t d[6] = {0u, 0u, 0u, 0u, 0u, 0u};
#pragma unroll
    for (int f = 0; f < 32; ++f) {
        const uint32_t c = e2m3_code(v[f], inv2e);  // code f at bits [6f, 6f + 6) of the 192-bit block
        const int bit = 6 * f, wd = bit >> 5, o = bit & 31;
        d[wd] |= c << o;
        if (o > 26) d[wd + 1] |= c >> (32 - o);
    }

    // ---- scatter into the MX6-T weight layout (inverse of the k512 output-column permutation) ----
    const int S = kb >> 2, q = kb & 3;
    const int tn = n >> 8, cc = n & 255;  // tile, column within the 256-column tile
    int h, R;
    if (perm == 0) {  // k512: perm(R, h) = 64(R>>5) + 16((R&15)>>2) + 4((R>>4)&1) + (R&3) + 8h
        h = (cc >> 3) & 1;
        R = 32 * (cc >> 6) + 16 * ((cc >> 2) & 1) + 4 * ((cc >> 4) & 3) + (cc & 3);
    } else {          // g3a / od STORE=1 (bf16 half-line stores): 64(R>>5) + 8((R&15)>>2) + 4((R>>4)&1) + (R&3) + 32h
        h = (cc >> 5) & 1;
        R = 32 * (cc >> 6) + 16 * ((cc >> 2) & 1) + 4 * ((cc >> 3) & 3) + (cc & 3);
    }
    uint8_t* item = w6 + ((size_t)((tn * 2 + h) * 4 + S)) * SLOT;
    uint4* hp = reinterpret_cast<uint4*>(item + 64 * R + 16 * (q ^ ((R >> 2) & 3)));
    uint2* tp = reinterpret_cast<uint2*>(item + TAILOFF + 32 * R + 8 * (q ^ (2 * ((R >> 3) & 1))));
    *hp = make_uint4(d[0], d[1], d[2], d[3]);
    *tp = make_uint2(d[4], d[5]);
    if ((lane & 31) == 0) ws[(size_t)(n >> 5) * 16 + q * 4 + S] = (uint8_t)eb;
}
}  // namespace k512mx6w

// C ABI. weight: [512, C] fp32 (is_bf16 = 0) or bf16 (1), element (k, col) at weight + k*ld_k + col*ld_n (elements);
// the shadow's columns are the ranges [s_r, s_r + w_r) in order (n_total = sum w_r, a multiple of 256, at most 4
// ranges); perm 0 = k512 output-column permutation (VQK / VK / U), 1 = g3a/od STORE=1 permutation (reserved for a U
// kernel with half-line bf16 stores). Outputs: w6 uint8 [n_total/256][2][4][12288], ws uint8 [n_total/32][4][4],
// sb fp32 [1]; work = 4 uint32 of a ZERO-initialized workspace private to this stream (left zero on return).
// Returns 0, this launch's hipError_t, or 100000 + a sticky error an EARLIER HIP call left on the thread.
extern "C" int k512_mx6_wshadow(const void* weight, int is_bf16, long long ld_k, long long ld_n, int n_total, int s0,
                                int w0, int s1, int w1, int s2, int w2, int s3, int w3, int perm, void* w6, void* ws,
                                void* sb, void* work, void* stream) {
    using namespace k512mx6w;
    if (!weight || !w6 || !ws || !sb || !work) return (int)hipErrorInvalidValue;
    if (n_total <= 0 || n_total % 256 != 0 || n_total > (1 << 20) || (perm != 0 && perm != 1))
        return (int)hipErrorInvalidValue;
    Ranges rg;
    const int ss[4] = {s0, s1, s2, s3}, ww[4] = {w0, w1, w2, w3};
    long long total = 0;
    rg.nr = 0;
    for (int r = 0; r < 4; ++r) {
        if (ww[r] < 0 || ss[r] < 0) return (int)hipErrorInvalidValue;
        if (ww[r] > 0 && r != rg.nr) return (int)hipErrorInvalidValue;  // used ranges first
        rg.s[r] = ss[r];
        rg.w[r] = ww[r];
        if (ww[r] > 0) rg.nr = r + 1;
        total += ww[r];
    }
    if (rg.nr == 0 || total != n_total) return (int)hipErrorInvalidValue;
    const unsigned grid = (unsigned)(P1 + n_total / 16);
    const hipError_t stale = hipGetLastError();
    if (is_bf16)
        hipLaunchKernelGGL((k512_mx6_wshadow_kernel<true>), dim3(grid), dim3(256), 0, (hipStream_t)stream, weight, ld_k,
                           ld_n, n_total, rg, perm, (uint8_t*)w6, (uint8_t*)ws, (float*)sb, (uint32_t*)work);
    else
        hipLaunchKernelGGL((k512_mx6_wshadow_kernel<false>), dim3(grid), dim3(256), 0, (hipStream_t)stream, weight,
                           ld_k, ld_n, n_total, rg, perm, (uint8_t*)w6, (uint8_t*)ws, (float*)sb, (uint32_t*)work);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) return (int)err;
    if (stale != hipSuccess) return K512MX6_STALE_ERROR_BASE + (int)stale;
    return 0;
}
extern "C" int k512_mx6_wshadow_p1(void) { return k512mx6w::P1; }
extern "C" const char* k512_mx6_wshadow_impl(void) {
    return "k512_mx6_wshadow v1 (r14 iter49): one launch, ticketed 2-phase (64 amax WGs + N/16 quantize WGs), "
           "FP8-shadow scale_b, pr7 quant_W + tile_W bytes";
}
#endif  // K512_MX6_WSHADOW

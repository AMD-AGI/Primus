// r14 iter49 fp6fwd/gemm track B: MXFP6 forward U projection GEMM of the HSTU uvqk layer (K = 512, N = 512, bf16 + bias
// output), for DLRM_HSTU_FP8_FWD_MXFP6 (default off; a DECLARED PRECISION CHANGE: E2M3 x E2M3 operands with E8M0 block
// scales instead of e4m3 with per-tensor scales).
//
// Credit / provenance (pr7 designed and prototyped the MX6 k512 family; this file finishes the U member of it):
//   MX6:<line> = pr7 iter24 fp6fwd proto/k512mx6.cu (sha256 a2cf3e5aa341...), k512mx6_kernel<N, WRONG=0, SC=1, TL=1>
//                (pre-tiled "MX6-T" operands, scaled MFMA): constants, buffer resource, barrier macros, ds_read16/8,
//                frag6/read_A/read_B, mf/mma8, g2l/issue, XCD remap, scale loads, pre-tiled resources + offsets, LDS read
//                offsets, SCHED-0 step and prologue. Reused verbatim except where marked [U].
//   G3A:<line> = pr7 iter11 g3a/g3a_gemm.cu (sha256 9b966751afdd...): the U (bf16 + bias) epilogue, K512 column mapping
//                (USTORE = 0), epilogue lane ids from mbcnt, the RNE bf16 pair convert, MODE.FP16_OVFL left at 0.
//   Both derive from the in-tree k512_fp8_gemm.cu (P7 iter9 G1-C); operand register layout and the st_16x128 XOR LDS
//   swizzle follow HipKittens (https://github.com/HazyResearch/HipKittens, MIT License, Copyright (c) 2024 HazyResearch).
//
// Math:
//   out[m][n] bf16 = bf16_rne( fl32( fl32(acc[m][n] * s) + float(bias[n]) ) ),   s = fl32(sa[0] * sb[0])   (mul, then add)
//   acc[m][n]      = fp32 MFMA chain over 4 in-order K = 128 steps of sum_k deq(x6)[m][k] * deq(w6)[n][k]
//                    (v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:2 blgp:2, one per 16 x 16 block per K step, operand-swapped
//                    as in K512), deq(code, e8m0) = E2M3(code) * 2^(e8m0 - 127)
//   = the epilogue of the shipped hipBLASLt U call (torch._scaled_mm(A8, W8.t(), sa, sb, bias = bf16, out_dtype = bf16):
//   mul, then add, two fp32 roundings, RNE to bf16; pr7 iter10 uepi / iter11 g3a) on MXFP6 operands.
//
// Operands (pr7 DESIGN.md section 9 "MX6-T"; the same contract as the VQK / VK MX6 kernels):
//   x6  uint8 [T][4][12288], T = ceil(M / 128): item (128-row tile, K step S) = its 12 KiB LDS image (HEAD 128 rows x 64 B:
//       row R's 16-B head of K block q at 64R + 16*(q ^ ((R>>2)&3)); TAIL 128 x 32 B at 8192: row R's 8-B tail of block q
//       at 32R + 8*(q ^ (2*((R>>3)&1)))). The producer zero-pads the last tile (its scale bytes are 127).
//   xs  uint8 [T][4][512] "MX6-A-v1": byte (q*16 + r)*8 + m of block (T, S) = E8M0 of row 128T + 16m + r, K block 4S + q.
//   w6  uint8 [N/256 = 2][2 halves][4 K steps][12288]: the W rows (= output columns) of B0/B1 of each 256-column tile in
//       slot-row order with the k512 output-column permutation (slot row R of half h holds column
//       64(R>>5) + 16((R&15)>>2) + 4((R>>4)&1) + (R&3) + 8h), each item in the same 12 KiB HEAD/TAIL image (proto tile_W).
//   ws  uint8 [N/32 = 16][q = 4][S = 4]: 2-D 32 x 32 E8M0 (32 output columns x 32 K) of W.
//   out bf16 [M][512] row-major, rows < M written, nothing else; bias bf16 [512]; sa, sb device fp32 scalars.
//
// Geometry (unchanged from K512 / MX6): WG = 4 waves, tile 128 (M) x 256 (N), wave w owns all 128 rows x columns
// [64w, 64w + 64); NT = 2 tiles per row panel (consecutive ids, same XCD after the remap: A from L2 for the second);
// LDS ring of 5 x 12 KiB slots (60 KiB per WG, 2 WGs per CU), 3 LDS-DMA loads per lane per item (linear 1 KiB copies).
// 64-bit addressing: per-tile x6 / xs / w6 bases and per-16-row-group output bases are size_t pointer arithmetic; only
// in-tile offsets (< 64 KiB) are 32-bit (M x N x 2 B passes 2^31 at production M).
// Epilogue per lane per 16 outputs: 8 v_pk_mul_f32 + 8 v_pk_add_f32 + 8 v_cvt_pk_bf16_f32 + 2 buffer_store_dwordx4.
//
// Build (deliverable): /opt/rocm/bin/hipcc k512_mx6_u.cu --offload-arch=gfx950 -std=c++20 -w -O3 -ffp-contract=off
//   -I/opt/rocm/include/hip -mllvm -amdgpu-sched-strategy=iterative-ilp -shared -fPIC
//   -Rpass-analysis=kernel-resource-usage -o libk512_mx6_u.so        (= the in-tree k512_fp8_gemm.py hipcc flags)
// Lab-only experiment / control variants are compiled only with -DK512_MX6_U_VARIANTS.
#include <hip/hip_runtime.h>
#include <stdint.h>

// LDS ring switch (r14 iter64 RING6, ledger r14-b-mx6-a-prefetch; lab job 71320: GO_INMODEL, bitwise; the same switch as
// k512_mx6_gemm.cu): K512MX6_RING undefined or 5 = the production 5-slot ring, token for token (the #else branches below
// are the shipped lines verbatim). K512MX6_RING=6 = RING6: 6 x 12 KiB slots (72 KiB per WG, 144 KiB per CU at 2 WGs) and
// the activation item A of K step KS+2 issued in step KS. Only slot placement, DMA issue points and vmcnt operands
// change (r14 probes/iter64/ring6/RING6.md).
#ifndef K512MX6_RING
#define K512MX6_RING 5
#endif
#if K512MX6_RING != 5 && K512MX6_RING != 6
#error "K512MX6_RING must be 5 (production ring) or 6 (RING6)"
#endif
// SECONDARY variant RING6S (-DK512MX6_RING=6 -DK512MX6_R6_SPIN=1): RING6 with the five scale loads pinned between the
// K-step-0 and K-step-1 prologue items (k512_mx6_gemm.cu). Same values, same variables.
#ifndef K512MX6_R6_SPIN
#define K512MX6_R6_SPIN 0
#endif
#if K512MX6_R6_SPIN && K512MX6_RING != 6
#error "K512MX6_R6_SPIN requires K512MX6_RING=6"
#endif

namespace mx6u {
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}  // namespace mx6u

// MX6:32-35
extern "C" __device__ void mx6u_llvm_buffer_load_lds(mx6u::i32x4 rsrc, mx6u::lds_ptr_t lds, int size, int voffset,
                                                     int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
extern "C" __device__ void mx6u_llvm_buffer_store_v4i32(mx6u::i32x4 vdata, mx6u::i32x4 rsrc, int voffset, int soffset,
                                                        int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

namespace mx6u {

constexpr int N = 512;         // [U] output columns (U slice of the uvqk weight)
constexpr int KD = 512;        // logical K
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int NT = N / BN;     // 2 column tiles per row panel
constexpr int KSTEPS = 4;
#if K512MX6_RING == 6
constexpr int NSLOT = 6;       // RING6
#else
constexpr int NSLOT = 5;
#endif
constexpr int SLOT = 12288;    // HEAD 8192 + TAIL 4096
constexpr int TAILOFF = 8192;
constexpr int XS_STEP = 512;   // bytes of one (tile, K step) MX6-A-v1 scale block
constexpr uint32_t ROWB = (uint32_t)N * 2u;  // [U] bytes per bf16 output row
#if K512MX6_RING == 6
// RING6 item -> slot map, K step 0..3 (k512_mx6_gemm.cu R6_*): prologue B0(0) B1(0) A(0) A(1) B0(1) B1(1) -> slots 0..5;
// step KS < 2 issues A(KS+2) into the slot B0(KS) frees at its second barrier and B0(KS+2) / B1(KS+2) into the slots
// B1(KS) / A(KS) free at its third barrier. A never uses slot 5 (ds_read immediate 5*12288 + 7*1024 > 65535).
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

// MX6:48-56
__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}

// MX6:58-66
#define MX6U_FENCE() asm volatile("" ::: "memory")
#define MX6U_SB() __builtin_amdgcn_sched_barrier(0)
#define MX6U_BARRIER()                \
    do {                              \
        MX6U_FENCE();                 \
        __builtin_amdgcn_s_barrier(); \
        MX6U_FENCE();                 \
        MX6U_SB();                    \
    } while (0)

// MX6:68-81
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

// MX6:83-105. One FP6 16x128 fragment: dwords 0..3 = head (ds_read_b128), dwords 4..5 = tail (ds_read_b64), dwords 6..7
// undefined (ignored by the cbsz/blgp = 2 MFMA, which reads a 6-VGPR operand).
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

// MX6:107-126. Operand-swapped scaled MFMA: src A = weight fragment (scale_a = 2-D W byte, op_sel = K step), src B =
// activation fragment (scale_b = (row, K block) byte, op_sel = subtile m within the half). SC = 0 is a lab-only
// negative control (constant unit scales: the compiler emits the unscaled form; block scales ignored).
template <int KS, int M, bool FIRST, int SC>
__device__ __forceinline__ f32x4 mf(const i32x8& b, const i32x8& a, const f32x4& c, int sw, int sa) {
    const f32x4 z = {0.f, 0.f, 0.f, 0.f};
    if constexpr (SC) {
        return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b, a, FIRST ? z : c, 2, 2, KS, sw, M, sa);
    } else {
        return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b, a, FIRST ? z : c, 2, 2, 0, 0, 0, 0);
    }
}
// acc[n][m] (+)= Bfrag[n] x Afrag[m] (n outer, m inner); sa = scale dword of this A half (op_sel = m)
template <int KS, bool FIRST, int SC>
__device__ __forceinline__ void mma8(f32x4 (&c)[2][4], const i32x8 (&b)[2], const i32x8 (&a)[4], int sw, int sa) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        c[n][0] = mf<KS, 0, FIRST, SC>(b[n], a[0], c[n][0], sw, sa);
        c[n][1] = mf<KS, 1, FIRST, SC>(b[n], a[1], c[n][1], sw, sa);
        c[n][2] = mf<KS, 2, FIRST, SC>(b[n], a[2], c[n][2], sw, sa);
        c[n][3] = mf<KS, 3, FIRST, SC>(b[n], a[3], c[n][3], sw, sa);
    }
}

// MX6:128-136 (TL form: soffset = K step * SLOT). [U] OFFS = 1 moves the per-load 4 KiB step of the linear 1 KiB copies
// from the per-lane voffset (pr7: three per-lane offsets) into the scalar soffset (one per-lane offset); same addresses.
__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    mx6u_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, 0);
}
template <int SL, int KK, int OFFS>
__device__ __forceinline__ void issue(i32x4 r, uint32_t ldsw, const uint32_t (&off)[3]) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        if constexpr (OFFS) g2l(r, ldsw + SL * SLOT + i * 4096, off[0], KK * SLOT + i * 4096);
        else g2l(r, ldsw + SL * SLOT + i * 4096, off[i], KK * SLOT);
    }
}

// G3A:153-156: RNE fp32 -> bf16 pair (gfx950: v_cvt_pk_bf16_f32)
__device__ __forceinline__ uint32_t pk_bf16(f32x2 v) {
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2));
}

}  // namespace mx6u

// SC = 1: scaled MFMA (the deliverable); 0: lab-only negative control. WRONG = 1: lab-only wrong-layout control (LDS
// read swizzle disabled). OFFS: see issue(). REMAP: K512 XCD-aware tile order (1 = deliverable).
template <int SC, int WRONG, int OFFS, int REMAP>
__global__ __launch_bounds__(256, 2) void k512_mx6_u_kernel(const uint8_t* __restrict__ X6,
                                                            const uint8_t* __restrict__ XS,
                                                            const uint8_t* __restrict__ W6,
                                                            const uint8_t* __restrict__ WS,
                                                            uint16_t* __restrict__ Cu,
                                                            const uint16_t* __restrict__ bias,
                                                            const float* __restrict__ sa, const float* __restrict__ sb,
                                                            int M) {
#pragma clang fp contract(off)
    using namespace mx6u;
    __shared__ __attribute__((aligned(2048))) uint8_t smem[NSLOT * SLOT];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));

    int bid = (int)blockIdx.x;
    if constexpr (REMAP) {  // MX6:159-166 (K512 XCD remap)
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
    // [U] no MODE.FP16_OVFL setreg (MX6:172 sets it for the saturating e4m3 convert): the bf16 converts must not
    // saturate (G3A:212-215: U workgroups keep the default MODE).

#if K512MX6_R6_SPIN
    // RING6S: the same two scale loads, issued by load_scales() at the pin point in the prologue (below)
    i32x2 sA[KSTEPS];
    int sW;
    auto load_scales = [&]() {
        const i32x2* ap = reinterpret_cast<const i32x2*>(XS + (size_t)tile_m * (KSTEPS * XS_STEP)) + lane;
#pragma unroll
        for (int s = 0; s < KSTEPS; ++s) sA[s] = ap[s * (XS_STEP / 8)];
        sW = reinterpret_cast<const int*>(WS)[(col0 / 32 + 2 * w + ((lane & 15) >> 3)) * 4 + (lane >> 4)];
    };
#else
    // ---- scales (MX6:174-183; issued first: oldest VMEM ops) ----
    // A: 2 dwords per K step at byte 8*lane of the (tile, S) block; dword H = subtiles 4H..4H+3 (op_sel = m)
    i32x2 sA[KSTEPS];
    {
        const i32x2* ap = reinterpret_cast<const i32x2*>(XS + (size_t)tile_m * (KSTEPS * XS_STEP)) + lane;
#pragma unroll
        for (int s = 0; s < KSTEPS; ++s) sA[s] = ap[s * (XS_STEP / 8)];
    }
    // W: [N/32][q][S]: the lane's 32-column block = col0/32 + 2w + ((lane&15)>>3) (the same for all four B fragments of
    // the k512 permutation), q = lane>>4; byte S = K step (op_sel = S)
    const int sW = reinterpret_cast<const int*>(WS)[(col0 / 32 + 2 * w + ((lane & 15) >> 3)) * 4 + (lane >> 4)];
#endif

    // ---- buffer resources (MX6:188-191, TL = 1: one tile-base resource per operand item stream) ----
    const i32x4 rA = make_rsrc(X6 + (size_t)tile_m * (KSTEPS * SLOT), (uint32_t)(KSTEPS * SLOT));
    const i32x4 rB0 = make_rsrc(W6 + (size_t)(2 * tile_n + 0) * (KSTEPS * SLOT), (uint32_t)(KSTEPS * SLOT));
    const i32x4 rB1 = make_rsrc(W6 + (size_t)(2 * tile_n + 1) * (KSTEPS * SLOT), (uint32_t)(KSTEPS * SLOT));

    // ---- per-lane DMA source offsets (MX6:205-207: every LDS-DMA instruction is a linear 1 KiB copy) ----
    uint32_t off[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) off[i] = (uint32_t)((OFFS ? 0 : i * 4096) + w * 1024 + lane * 16);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);
    const uint32_t ldsw = lds0 + (uint32_t)w * 1024u;

    // ---- per-lane LDS read offsets (MX6:224-231) ----
    const int r16 = lane & 15, q4 = lane >> 4;
    const uint32_t hsw = WRONG ? 0u : (uint32_t)((r16 >> 2) & 3);
    const uint32_t tsw = WRONG ? 0u : (uint32_t)(2 * ((r16 >> 3) & 1));
    const uint32_t lpH = lds0 + (uint32_t)r16 * 64u + 16u * ((uint32_t)q4 ^ hsw);
    const uint32_t lpT = lds0 + (uint32_t)TAILOFF + (uint32_t)r16 * 32u + 8u * ((uint32_t)q4 ^ tsw);
    const uint32_t lpBH = lpH + (uint32_t)w * 2048u;
    const uint32_t lpBT = lpT + (uint32_t)w * 1024u;

    i32x8 a[4], b0[2], b1[2];
    f32x4 acc[4][2][4];  // [b0 x a(rows 0-63), b1 x a(0-63), b0 x a(64-127), b1 x a(64-127)][n][m]
    i32x4 bq0, bq1;      // raw bf16 bias of this lane's 16 columns

    auto bias_prefetch = [&]() {  // MX6:237-241 / G3A:258-262 (K512 mapping: columns [64w + 16*(lane>>4), +16))
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
    // MX6:243-280: SCHED 0 with 3-load items: younger loads at a step start = B0/B1 of the next step = 6
    auto step0 = [&]<int KS>() {
        constexpr int sB0 = (3 * KS) % NSLOT, sB1 = (3 * KS + 1) % NSLOT, sAs = (3 * KS + 2) % NSLOT;
        if constexpr (KS < KSTEPS - 1) asm volatile("s_waitcnt vmcnt(6)" ::: "memory");
        else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#endif
        MX6U_BARRIER();
        read_B<sB0>(b0, lpBH, lpBT);
        read_A<sAs, 0>(a, lpH, lpT);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        MX6U_BARRIER();
#if K512MX6_RING == 6
        if constexpr (KS + 2 < KSTEPS) issue<R6_SA[KS + 2], KS + 2, OFFS>(rA, ldsw, off);  // A two K steps ahead
#else
        if constexpr (KS + 1 < KSTEPS) issue<sB0, KS + 1, OFFS>(rA, ldsw, off);
#endif
        if constexpr (KS == KSTEPS - 1) bias_prefetch();
        MX6U_SB();
        __builtin_amdgcn_s_setprio(1);
        mma8<KS, KS == 0, SC>(acc[0], b0, a, sW, sA[KS].x);
        __builtin_amdgcn_s_setprio(0);
        MX6U_SB();
        read_B<sB1>(b1, lpBH, lpBT);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        MX6U_SB();
        __builtin_amdgcn_s_setprio(1);
        mma8<KS, KS == 0, SC>(acc[1], b1, a, sW, sA[KS].x);
        __builtin_amdgcn_s_setprio(0);
        MX6U_SB();
        read_A<sAs, 1>(a, lpH, lpT);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        MX6U_BARRIER();
        if constexpr (KS + 2 < KSTEPS) {
#if K512MX6_RING == 6
            issue<R6_SB0[KS + 2], KS + 2, OFFS>(rB0, ldsw, off);  // = sB1 (freed above)
            issue<R6_SB1[KS + 2], KS + 2, OFFS>(rB1, ldsw, off);  // = sAs (freed above)
#else
            issue<sB1, KS + 2, OFFS>(rB0, ldsw, off);
            issue<sAs, KS + 2, OFFS>(rB1, ldsw, off);
#endif
        }
        MX6U_SB();
        __builtin_amdgcn_s_setprio(1);
        mma8<KS, KS == 0, SC>(acc[2], b0, a, sW, sA[KS].y);
        mma8<KS, KS == 0, SC>(acc[3], b1, a, sW, sA[KS].y);
        __builtin_amdgcn_s_setprio(0);
        MX6U_SB();
    };

#if K512MX6_RING == 6
    // RING6 prologue: B0_0, B1_0, A_0, A_1, B0_1, B1_1 -> slots 0..5
    issue<R6_SB0[0], 0, OFFS>(rB0, ldsw, off);
    issue<R6_SB1[0], 0, OFFS>(rB1, ldsw, off);
    issue<R6_SA[0], 0, OFFS>(rA, ldsw, off);
#if K512MX6_R6_SPIN
    MX6U_SB();  // RING6S: scale loads after the K-step-0 items, before the K-step-1 items
    load_scales();
    MX6U_SB();
#endif
    issue<R6_SA[1], 1, OFFS>(rA, ldsw, off);
    issue<R6_SB0[1], 1, OFFS>(rB0, ldsw, off);
    issue<R6_SB1[1], 1, OFFS>(rB1, ldsw, off);
#else
    // MX6:282-290: prologue items 0..4 = B0_0, B1_0, A_0, B0_1, B1_1 -> slots 0..4
    issue<0, 0, OFFS>(rB0, ldsw, off);
    issue<1, 0, OFFS>(rB1, ldsw, off);
    issue<2, 0, OFFS>(rA, ldsw, off);
    issue<3, 1, OFFS>(rB0, ldsw, off);
    issue<4, 1, OFFS>(rB1, ldsw, off);
#endif
    step0.template operator()<0>();
    step0.template operator()<1>();
    step0.template operator()<2>();
    step0.template operator()<3>();

    // ---- [U] epilogue, G3A:314-363 with the K512 column mapping (USTORE = 0): lane owns row 16g + l%16 of the tile for
    // g = 0..7 and the 16 consecutive columns [64w + 16*(l/16), +16); out = bf16_rne(f32(f32(acc*s) + bias)) ----
    const float s = sav * sbv;
    const f32x2 s2 = {s, s};
    f32x2 bias2[8];
    {  // MX6:296-303 / G3A:318-325
        const int bw[8] = {bq0.x, bq0.y, bq0.z, bq0.w, bq1.x, bq1.y, bq1.z, bq1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t d = (uint32_t)bw[i];
            bias2[i] = f32x2{__uint_as_float(d << 16), __uint_as_float(d & 0xffff0000u)};
        }
    }
    // G3A:326-331: lane ids re-derived from mbcnt (no lane-id VGPR has to stay live across the mainloop)
    asm volatile("; k512_mx6_u bf16 epilogue" ::: "memory");
    const int le = (int)__builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
    const uint32_t lane_off = (uint32_t)(le & 15) * ROWB + (uint32_t)(64 * w + 16 * (le >> 4)) * 2u;
    uint16_t* Ct = Cu + (size_t)row0 * N + col0;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        const int h = g >> 2, ms = g & 3;
        const f32x4 v[4] = {acc[2 * h][0][ms], acc[2 * h][1][ms], acc[2 * h + 1][0][ms], acc[2 * h + 1][1][ms]};
        uint32_t pk[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            f32x2 lo = __builtin_shufflevector(v[j], v[j], 0, 1);
            f32x2 hi = __builtin_shufflevector(v[j], v[j], 2, 3);
            lo = lo * s2;
            hi = hi * s2;
            lo = lo + bias2[2 * j];
            hi = hi + bias2[2 * j + 1];
            pk[2 * j] = pk_bf16(lo);
            pk[2 * j + 1] = pk_bf16(hi);
        }
        const i32x4 d0 = {(int)pk[0], (int)pk[1], (int)pk[2], (int)pk[3]};  // columns +0..7
        const i32x4 d1 = {(int)pk[4], (int)pk[5], (int)pk[6], (int)pk[7]};  // columns +8..15
        int rv = rows_valid - 16 * g;
        rv = rv < 0 ? 0 : (rv > 16 ? 16 : rv);
        const uint32_t nrec = (uint32_t)__builtin_amdgcn_readfirstlane(rv * (int)ROWB);  // scalar rsrc; rows >= M dropped
        const i32x4 rC = make_rsrc(Ct + (size_t)(16 * g) * N, nrec);
        mx6u_llvm_buffer_store_v4i32(d0, rC, (int)lane_off, 0, 0);
        mx6u_llvm_buffer_store_v4i32(d1, rC, (int)(lane_off + 16u), 0, 0);
        MX6U_SB();  // drain group by group (bounded temporaries; stores start early)
    }
}

template <int SC, int WRONG, int OFFS, int REMAP>
static int k512_mx6_u_launch(const void* x6, const void* xs, const void* w6, const void* ws, void* out, const void* bias,
                             const float* sa, const float* sb, int M, void* stream) {
    if (M <= 0) return (int)hipErrorInvalidValue;
    const long long tiles = (((long long)M + 127) / 128) * mx6u::NT;
    if (tiles * 256LL > 0xffffffffLL) return (int)hipErrorInvalidValue;  // grid work-items must fit in 32 bits
    hipLaunchKernelGGL((k512_mx6_u_kernel<SC, WRONG, OFFS, REMAP>), dim3((unsigned)tiles), dim3(256), 0,
                       (hipStream_t)stream, (const uint8_t*)x6, (const uint8_t*)xs, (const uint8_t*)w6,
                       (const uint8_t*)ws, (uint16_t*)out, (const uint16_t*)bias, sa, sb, M);
    return (int)hipGetLastError();
}

#ifndef K512_MX6_U_OFFS
#define K512_MX6_U_OFFS 0  // ABI default: pr7's per-lane offsets (MX6:205-207)
#endif

// ---------------------------------------------------------------- deliverable C ABI
// x6 uint8 [ceil(M/128), 4, 12288] MX6-T; xs uint8 [ceil(M/128), 4, 512] MX6-A-v1; w6 uint8 [2, 2, 4, 12288] MX6-T weight
// (k512 output-column permutation); ws uint8 [16, 4, 4] 2-D 32x32 E8M0 ([N/32][q][S]); out_bf16 [M, 512] row-major;
// bias_bf16 [512]; sa, sb fp32 device scalars (s = sa[0] * sb[0]); all on the current device, x6 / w6 / bias 16-byte,
// xs 8-byte, ws 4-byte aligned. 1 <= M < 2^31 (and ceil(M/128) * 2 * 256 < 2^32). Rows < M written, nothing else.
// Returns hipGetLastError() of the launch (0 = OK; hipErrorInvalidValue for M <= 0). Asynchronous on `stream`, no host
// sync (scales are read on the device), graph-capturable.
extern "C" int k512_mx6_u_bf16_bias(const void* x6, const void* xs, const void* w6, const void* ws, void* out_bf16,
                                    const void* bias_bf16, const float* sa, const float* sb, int M, void* stream) {
    return k512_mx6_u_launch<1, 0, K512_MX6_U_OFFS, 1>(x6, xs, w6, ws, out_bf16, bias_bf16, sa, sb, M, stream);
}
extern "C" const char* k512_mx6_u_impl(void) {
#if K512MX6_RING == 6
    return "k512_mx6_u: MXFP6 E2M3 x E2M3 (E8M0/32; W 2-D 32x32) U GEMM N=512 K=512, bf16 + bias; pr7 k512mx6 MX6-T "
           "2-WG mainloop (RING6 6 x 12 KiB LDS ring, A two K steps ahead"
#if K512MX6_R6_SPIN
           "; RING6S: scale loads pinned after the K-step-0 items"
#endif
           ", scaled MFMA cbsz:2 blgp:2) + pr7 g3a bf16 epilogue (K512 mapping)";
#else
    return "k512_mx6_u: MXFP6 E2M3 x E2M3 (E8M0/32; W 2-D 32x32) U GEMM N=512 K=512, bf16 + bias; pr7 k512mx6 MX6-T "
           "2-WG mainloop (5 x 12 KiB LDS ring, scaled MFMA cbsz:2 blgp:2) + pr7 g3a bf16 epilogue (K512 mapping)";
#endif
}

#ifdef K512_MX6_U_VARIANTS
// ---------------------------------------------------------------- lab-only variants and controls (same signature)
#define K512_MX6_U_VAR(name, sc, wr, of, rm)                                                                          \
    extern "C" int k512_mx6_u_##name(const void* x6, const void* xs, const void* w6, const void* ws, void* out,       \
                                     const void* bias, const float* sa, const float* sb, int M, void* stream) {       \
        return k512_mx6_u_launch<sc, wr, of, rm>(x6, xs, w6, ws, out, bias, sa, sb, M, stream);                       \
    }
K512_MX6_U_VAR(v_offs0, 1, 0, 0, 1)    // pr7 per-lane offsets (MX6:205-207)
K512_MX6_U_VAR(v_offs1, 1, 0, 1, 1)    // soffset form (one per-lane offset)
K512_MX6_U_VAR(v_remap0, 1, 0, K512_MX6_U_OFFS, 0)  // no XCD remap
K512_MX6_U_VAR(ctl_unscaled, 0, 0, K512_MX6_U_OFFS, 1)  // NEGATIVE CONTROL: block scales ignored
K512_MX6_U_VAR(ctl_wrong, 1, 1, K512_MX6_U_OFFS, 1)     // NEGATIVE CONTROL: LDS read swizzle disabled
#endif  // K512_MX6_U_VARIANTS

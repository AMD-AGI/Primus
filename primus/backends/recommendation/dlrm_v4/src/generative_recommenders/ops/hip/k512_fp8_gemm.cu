// K=512 FP8 forward projection GEMM for the HSTU V/Q/K (N=1536) and targets V/K (N=1024) calls, behind
// DLRM_HSTU_FP8_FWD_K512 (generative_recommenders/ops/hip/k512_fp8_gemm.py; dispatch in
// ops/triton/triton_fp8.scaled_mm_fp8_out_preq). Bit-identical to the hipBLASLt call it replaces
// (ops/tests/k512_fp8_gemm_test.py compares every output byte).
//
// Provenance: Arbor P7 iteration 9, sub-track B7 G1-C ("2wg"), source sha256 1d601231a1aa (probes/iter9/k512/twowg),
// gated at 1.24x (VQK) / 1.23x (VK) over hipBLASLt in sustained 1000 W soaks. Self-contained HIP (raw intrinsics,
// no HipKittens headers); the operand register layout and the st_16x128 XOR LDS swizzle follow HipKittens
// (https://github.com/HazyResearch/HipKittens, MIT License, Copyright (c) 2024 HazyResearch) and r14's 8-wave
// adaptation. Experiment variants of the probe are compiled only with -DK512_EXPERIMENT_VARIANTS.
//
// P7 iter9 B7 variant G1-C "2wg": K=512 FP8 GEMM + scale + bias + sat448 + e4m3 output for the HSTU forward VQK / VK
// projections, TWO resident 4-wave workgroups per CU, so one WG's epilogue (VALU + stores) overlaps the other WG's MFMA
// mainloop (the r14 HipKittens 8-wave kernel is 1 WG/CU with all 8 waves barrier-synchronised: its epilogue is serial).
//
// Math (bit-identical target: hipBLASLt via torch._scaled_mm(A, B.t(), scale_a=sa, scale_b=sb, bias=bf16,
// out_dtype=float8_e4m3fn, use_fast_accum=False), 'input_scale' mode):
//   acc[m][n] = sum_k A[m][k] * B[n][k]   fp32, FOUR IN-ORDER K=128 STEPS, one v_mfma_f32_16x16x128_f8f6f4 per 16x16
//                                         output block per step, HipKittens operand register layout (as r14's kernel)
//   C[m][n]   = e4m3_rne( sat448( acc*s + bias[n] ) ),  s = sa*sb (one fp32 multiply), MUL THEN ADD (no contraction)
//
// Geometry: WG = 4 waves (256 threads), WG tile 128 (M) x 256 (N); wave w owns all 128 rows x 64 columns [64w, 64w+64).
//   * The MFMA is issued operand-SWAPPED (B fragment = 'A' operand, A fragment = 'B' operand): D'[n][m] = C^T, so each
//     lane holds 4 consecutive OUTPUT COLUMNS of one output row per 16x16 block. B rows are permuted on their way into
//     LDS (free: it is the per-lane global source offset of the LDS-DMA load) so that the 4 blocks a lane holds per
//     16-row group are 16 consecutive columns -> one 16-byte buffer_store_dwordx4 per 16 outputs, no lane exchange.
//   * LDS: ring of NSLOT=5 slots x 16 KiB (128 rows x 128 K-bytes, HipKittens st_16x128 XOR swizzle). Per K step the
//     items are B0 (permuted cols 0..127), B1 (128..255), A (128 rows); item j = 3k+t lives in slot j % 5, so every item
//     is prefetched about one K step ahead with 80 KiB of LDS -> 2 WGs/CU fit in the 160 KiB of a gfx950 CU.
//   * M tail: buffer resources with num_records = valid bytes: A loads past M return 0, C stores past M are dropped.
//     64-bit per-tile base pointers (scalar), 32-bit in-tile lane offsets.
//   * XCD-aware tile order (REMAP=1): XCD x runs a contiguous range of tile ids; the N/256 tiles of one A row panel
//     have consecutive ids, so they run close in time on the same XCD (A reuse in its L2).
// Epilogue per 16 outputs (per lane): 8 v_pk_mul_f32 (acc x s) + 8 v_pk_add_f32 (+bias) + 16 v_med3/clamp (NOCLAMP=0)
//   + 8 v_cvt_pk_fp8_f32 + 1 store. NOCLAMP=1 drops the clamp and sets MODE.FP16_OVFL (MODE bit 23), under which the
//   gfx950 fp8 convert saturates to +-448 (r15 iter22 satcvt: exhaustive 2^32 proof; NaN -> 0xff instead of 0x7e;
//   +-inf inputs give NaN instead of +-448 -- which is exactly what hipBLASLt emits for +-inf intermediates, so the
//   NOCLAMP=1 form is the one that is bitwise on every test; the explicit clamp gives 448 there).
// Default ABI = NOCLAMP=1, REMAP=1, PRIO=1, SCHED=0. 256 (N=1536: 244 with NOCLAMP) VGPR, 0 AGPR, 0 spill, 80 KiB LDS,
// 2 waves/SIMD. Sustained (1000 W, ABBA vs hipBLASLt): see RESULT.md.
// Build: hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -DHK_N={1536|1024} -mllvm
//        -amdgpu-sched-strategy=iterative-ilp -ffp-contract=off -shared -fPIC (build.sh).
//
// TILED-A mode (Arbor P7 iteration 20, item B15T(L2); port of the iteration-11 prototype probes/iter11/ytin/k512_ytin_v3.cu
// variant a3g, which was derived from this source at sha256 530a4957ccd2): kernel k512_2wg_kernel_tiled_a<N> (the
// deliverable ABI, body k512_2wg_body<..., TA = true>) reads A from the STEP-TILED token-major yT, e4m3
// [ceil(M/128), 512, 128], token t / feature c at byte ((t / 128) * 512 + c) * 128 + t % 128 (ops/triton/wgrad_ck.py
// "Step-tiled xT"; the LayerNorm store_t writes it for the K=tokens wgrad), instead of the row-major y [M, 512].
// Entry k512_fp8_gemm_e4m3_bias_tiled_a_N{N}, same C signature as k512_fp8_gemm_e4m3_bias_N{N}.
//   * The A rows of one WG tile (tokens row0..row0+127) are tile block row0/128: one contiguous 64 KiB block, K-row stride
//     128 B, K step KK = K-rows [128KK, 128KK+128) at +16384*KK. The A item lands K-major in its LDS slot: K-row kk at
//     kk*128, 16-token chunk c at 16-byte position c ^ f(kk), f(kk) = (kk & 7) ^ ((kk >> 4) & 1), so lane l of wave w,
//     load i fetches chunk (l&7) ^ (l>>3) ^ ((w>>1)&1) of K-row 32i + 8w + l/8 (8 lanes = one 128 B line).
//   * A fragments: 4 ds_read_b64_tr_b8 (gfx950 transposing LDS read) per 16x128 fragment, at +0/+1024/+8192/+9216 from
//     one per-lane address; within a 16-lane group out[L][byte j] = lds[addr[2j + (L>>3)] + (L&7)], so lane 16g+s holds
//     token 16c+s, K bytes 16g+[0,8), 16g+[8,16), 64+16g+[0,8), 64+16g+[8,16) = exactly the bytes of the row-major
//     ds_read_b128 fragment: identical MFMA operands, bit-identical outputs.
//   * Tail: tokens [M, Mp) of the last block are read as stored (the producers zero them); every output row depends only
//     on its own token, and rows >= M are not stored.
//   * Store guard: 2 x s_nop 7 after each buffer_store_dwordx4, so no instruction rewrites store data VGPRs within 16
//     wait states (P7 iter9 G1-A: a 3-wait-state rewrite corrupted bytes; the guard-free yT build had 3).
//   SCHED 0 only. k512_2wg_kernel<...> (every pre-existing instantiation) runs k512_2wg_body<..., TA = false>, which
//   does not instantiate any of this (discarded 'if constexpr' branches): same symbol names and same ISA as before.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

#ifndef HK_N
#error "define HK_N (1024 or 1536)"
#endif

namespace k512 {
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}  // namespace k512

extern "C" __device__ void k512_llvm_buffer_load_lds(k512::i32x4 rsrc, k512::lds_ptr_t lds, int size, int voffset,
                                                     int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
extern "C" __device__ void k512_llvm_buffer_store_v4i32(k512::i32x4 vdata, k512::i32x4 rsrc, int voffset, int soffset,
                                                        int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

namespace k512 {

constexpr int KD = 512;      // K
constexpr int BM = 128;      // WG tile rows (M)
constexpr int BN = 256;      // WG tile cols (N)
constexpr int KSTEPS = 4;    // K / 128
constexpr int NSLOT = 5;     // LDS ring slots
constexpr int SLOT = 16384;  // bytes per slot (128 x 128 fp8)

// gfx950 buffer resource: base (48 bit), stride 0, num_records (bytes), dword3 = HipKittens' value for gfx950.
__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}

#define K512_FENCE() asm volatile("" ::: "memory")
#define K512_SB() __builtin_amdgcn_sched_barrier(0)
#define K512_BARRIER()              \
    do {                            \
        K512_FENCE();               \
        __builtin_amdgcn_s_barrier(); \
        K512_FENCE();               \
        K512_SB();                  \
    } while (0)

template <int IMM>
__device__ __forceinline__ i32x4 ds_read16(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x4 v;
    asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}

// One 16x128 fp8 MFMA operand fragment (HipKittens load_st_to_rt for rt_16x128 from an st_16x128 subtile):
// lane l reads row l%16, K bytes [16*(l/16), +16) -> dwords 0-3 and [64 + 16*(l/16), +16) -> dwords 4-7.
// lp0/lp1 = per-lane swizzled offsets of the two 16-byte chunks (incl. the LDS base and any wave offset);
// S = compile-time byte offset of the 16-row subtile (multiple of 2048).
template <int S>
__device__ __forceinline__ i32x8 frag(uint32_t lp0, uint32_t lp1) {
    i32x4 lo, hi;
    if constexpr (S < 65536) {
        lo = ds_read16<S>(lp0);
        hi = ds_read16<S>(lp1);
    } else {
        lo = ds_read16<S - 65536>(lp0 + 65536u);
        hi = ds_read16<S - 65536>(lp1 + 65536u);
    }
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}

// A fragments: rows [64h, 64h+64) of the A slot (4 subtiles of 16 rows)
template <int SL, int H>
__device__ __forceinline__ void read_A(i32x8 (&a)[4], uint32_t lp0, uint32_t lp1) {
    a[0] = frag<SL * SLOT + H * 8192 + 0 * 2048>(lp0, lp1);
    a[1] = frag<SL * SLOT + H * 8192 + 1 * 2048>(lp0, lp1);
    a[2] = frag<SL * SLOT + H * 8192 + 2 * 2048>(lp0, lp1);
    a[3] = frag<SL * SLOT + H * 8192 + 3 * 2048>(lp0, lp1);
}
// B fragments: rows [32w, 32w+32) of a B slot (wave offset 4096*w is inside lpB0/lpB1)
template <int SL>
__device__ __forceinline__ void read_B(i32x8 (&b)[2], uint32_t lpB0, uint32_t lpB1) {
    b[0] = frag<SL * SLOT + 0>(lpB0, lpB1);
    b[1] = frag<SL * SLOT + 2048>(lpB0, lpB1);
}

// ---- tiled-A mode only (see the header) ----
// gfx950 transposing 8-bit LDS read, 8 bytes per lane
template <int IMM>
__device__ __forceinline__ i32x2 ds_read_tr8(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x2 v;
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
// One 16x128 A fragment from a K-major slot: 4 transposing reads at +0 (K 16g+0..7), +1024 (16g+8..15), +8192
// (64+16g+0..7), +9216 (64+16g+8..15); the same bytes per lane as frag() on the row-major slot.
template <int S>
__device__ __forceinline__ i32x8 fragT(uint32_t tp) {
    static_assert(S + 9216 < 65536, "A items never use slot 4");
    const i32x2 r0 = ds_read_tr8<S>(tp);
    const i32x2 r1 = ds_read_tr8<S + 1024>(tp);
    const i32x2 r2 = ds_read_tr8<S + 8192>(tp);
    const i32x2 r3 = ds_read_tr8<S + 9216>(tp);
    const i32x4 lo = __builtin_shufflevector(r0, r1, 0, 1, 2, 3);
    const i32x4 hi = __builtin_shufflevector(r2, r3, 0, 1, 2, 3);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}
// v ^ IMM as one v_xor per fragment (asm volatile: not hoisted into 8 live address VGPRs)
template <int IMM>
__device__ __forceinline__ uint32_t xor_imm(uint32_t v) {
    if constexpr (IMM == 0) {
        return v;
    } else {
        uint32_t r;
        asm volatile("v_xor_b32 %0, %1, %2" : "=v"(r) : "i"(IMM), "v"(v));
        return r;
    }
}
// A fragments of token subtiles c = 4H + m (rows [64H, 64H+64) of the tile) from a K-major slot; the address of
// subtile c is bfT ^ 16c (bfT: per-lane base, see the kernel)
template <int SL, int H>
__device__ __forceinline__ void read_AT(i32x8 (&a)[4], uint32_t bfT) {
    a[0] = fragT<SL * SLOT>(xor_imm<16 * (4 * H + 0)>(bfT));
    a[1] = fragT<SL * SLOT>(xor_imm<16 * (4 * H + 1)>(bfT));
    a[2] = fragT<SL * SLOT>(xor_imm<16 * (4 * H + 2)>(bfT));
    a[3] = fragT<SL * SLOT>(xor_imm<16 * (4 * H + 3)>(bfT));
}

// acc[n][m] (+)= Bfrag[n] (16 output cols) x Afrag[m] (16 output rows): D'[col][row], lane holds 4 consecutive cols.
template <bool FIRST>
__device__ __forceinline__ void mma8(f32x4 (&c)[2][4], const i32x8 (&b)[2], const i32x8 (&a)[4]) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if constexpr (FIRST) {
                const f32x4 z = {0.f, 0.f, 0.f, 0.f};
                c[n][m] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[n], a[m], z, 0, 0, 0, 0, 0, 0);
            } else {
                c[n][m] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[n], a[m], c[n][m], 0, 0, 0, 0, 0, 0);
            }
        }
    }
}

__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    k512_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, 0);
}

// A item (128 rows x 128 K-bytes of K step KK) -> slot SL. 4 x 16 B per lane; per-lane voffsets carry the row
// (bounds-checked: rows >= M are out of range and read as 0), the K-step column offset goes into soffset.
// KSB = bytes per K step in A: 128 (row-major y), 16384 (tiled A: 128 K-rows of 128 tokens).
template <int SL, int KK, int KSB = 128>
__device__ __forceinline__ void issue_A(i32x4 rA, uint32_t ldsw, const uint32_t (&offA)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) g2l(rA, ldsw + SL * SLOT + i * 4096, offA[i], KK * KSB);
}
// B item: rsrc base already selects B0 (+0) or B1 (+8 rows); per-i row offset 64*i rows = 32768 B in soffset.
template <int SL, int KK>
__device__ __forceinline__ void issue_B(i32x4 rB, uint32_t ldsw, uint32_t offB) {
#pragma unroll
    for (int i = 0; i < 4; ++i) g2l(rB, ldsw + SL * SLOT + i * 4096, offB, i * 32768 + KK * 128);
}

}  // namespace k512

// The kernel body, force-inlined into the two __global__ entries below: k512_2wg_kernel (TA = false: row-major A, every
// pre-existing instantiation, same symbol and ISA as before) and k512_2wg_kernel_tiled_a (TA = true: tiled A, see the
// header). The pointers carry no __restrict__ here: their noalias comes from the kernels' parameters, as before.
template <int N, int NOCLAMP, int REMAP, int PRIO, int SCHED, int SAUX, bool TA>
__device__ __forceinline__ void k512_2wg_body(const uint8_t* A, const uint8_t* B, uint8_t* C, const uint16_t* bias,
                                              const float* sa, const float* sb, int M) {
#pragma clang fp contract(off)
    using namespace k512;
    static_assert(N % BN == 0, "N must be a multiple of 256");
    static_assert(!TA || SCHED == 0, "tiled A: SCHED 0 only");
    constexpr int AKSB = TA ? 128 * BM : 128;  // A bytes per K step: 128 columns (row-major) | 128 K-rows of BM tokens
    constexpr int NT = N / BN;
    __shared__ __attribute__((aligned(2048))) uint8_t smem[NSLOT * SLOT];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));

    int bid = (int)blockIdx.x;
    if constexpr (REMAP) {  // r14/HK XCD remap: XCD x (= bid % 8 under round-robin dispatch) gets a contiguous id range
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
    // MODE.FP16_OVFL (bit 23): fp8 converts saturate to +-448. Set at kernel start: every convert depends (through the
    // MFMA chain) on the asm-volatile LDS reads that follow this asm-volatile s_setreg, so no convert can be scheduled
    // before it (placing it right before the epilogue let the compiler hoist converts above it).
    if constexpr (NOCLAMP) asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n s_nop 7\n s_nop 7" ::: "memory");

    // ---- buffer resources (scalar) ----
    // tiled A: tile block row0/128 = 512 K-rows x 128 tokens, contiguous 64 KiB at byte row0 * KD, all of it in range
    const i32x4 rA = make_rsrc(A + (size_t)row0 * KD, (uint32_t)((TA ? BM : rows_valid) * KD));
    const uint8_t* Bt = B + (size_t)col0 * KD;
    const i32x4 rB0 = make_rsrc(Bt, (uint32_t)((N - col0) * KD));
    const i32x4 rB1 = make_rsrc(Bt + 8 * KD, (uint32_t)((N - col0) * KD - 8 * KD));

    // ---- per-lane global source offsets of the LDS-DMA loads ----
    // LDS destination of lane l, wave w, load i: slot + i*4096 + w*1024 + l*16 -> slot row 32i + 8w + l/8, swizzled
    // 16-byte chunk (l&7) ^ ((4*(w&1) + l/16) & 7)  (inverse of the st_16x128 XOR swizzle; HK global_to_shared).
    const uint32_t chunk = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    uint32_t offA[4];
    if constexpr (TA) {
        // tiled A, K-major slot: K-row 32i + 8w + l/8 (stride 128 B), 16-token chunk (l&7) ^ (l>>3) ^ ((w>>1)&1)
        const uint32_t cT = (uint32_t)((lane & 7) ^ (lane >> 3) ^ ((w >> 1) & 1));
#pragma unroll
        for (int i = 0; i < 4; ++i) offA[i] = (uint32_t)(32 * i + 8 * w + (lane >> 3)) * (uint32_t)BM + cT * 16u;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) offA[i] = (uint32_t)(32 * i + 8 * w + (lane >> 3)) * KD + chunk * 16;
    }
    // B slot row r = 32*i + q (q = 8w + l/8): wave i's fragment row q -> jj = q/16, npos = q%16; output column (within
    // the WG tile) = 64*i + 16*(npos/4) + 4*(2h + jj) + npos%4 for slot h (h=1 adds 8 rows: the rB1 base).
    uint32_t offB;
    {
        const int q = 8 * w + (lane >> 3);
        const int jj = q >> 4, npos = q & 15;
        offB = (uint32_t)(16 * (npos >> 2) + 4 * jj + (npos & 3)) * KD + chunk * 16;
    }

    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);
    const uint32_t ldsw = lds0 + (uint32_t)w * 1024u;  // wave-uniform M0 base for the LDS-DMA loads

    // ---- per-lane LDS read offsets (HK load_st_to_rt): row l%16, chunks 16*(l/16) and 64 + 16*(l/16), XOR swizzle ----
    const int r16 = lane & 15, q4 = lane >> 4;
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t lpA0 = lds0 + (uint32_t)r16 * 128u + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lpA1 = lds0 + (uint32_t)r16 * 128u + (((uint32_t)(16 * q4 + 64)) ^ swz);
    const uint32_t lpB0 = lpA0 + (uint32_t)w * 4096u;
    const uint32_t lpB1 = lpA1 + (uint32_t)w * 4096u;
    // tiled A: per-lane base of the transposing fragment reads, lane l = 16g + s: K-row 16g + s/2, token half 8(s&1),
    // chunk position c ^ f, f = (s>>1) ^ (g&1) (= f(K-row) of all 4 reads); lds0 is 2048-aligned (address bits 4-6
    // clear), so the address of token subtile c is (base ^ 16f) ^ 16c
    uint32_t bfT = 0;
    if constexpr (TA) {
        const uint32_t s = (uint32_t)(lane & 15), g = (uint32_t)(lane >> 4);
        const uint32_t f = (s >> 1) ^ (g & 1u);
        const uint32_t base = lds0 + (16u * g + (s >> 1)) * 128u + 8u * (s & 1u);
        bfT = base ^ (16u * f);
    }

    i32x8 a[4], b0[2], b1[2];
    f32x4 acc[4][2][4];  // [cA: b0 x a(rows 0-63), cB: b1 x a(0-63), cC: b0 x a(64-127), cD: b1 x a(64-127)][n][m]
    i32x4 bq0, bq1;      // raw bf16 bias of this lane's 16 columns

    auto prio_on = [&]() { if constexpr (PRIO) __builtin_amdgcn_s_setprio(1); };
    auto prio_off = [&]() { if constexpr (PRIO) __builtin_amdgcn_s_setprio(0); };
    auto bias_prefetch = [&]() {  // 8 VGPRs, issued at the start of the last K step (no LDS-DMA loads after it)
        const uint16_t* bp = bias + col0 + 64 * w + 16 * (lane >> 4);
        bq0 = *reinterpret_cast<const i32x4*>(bp);
        bq1 = *reinterpret_cast<const i32x4*>(bp + 8);
    };

    // SCHED 0: simple 3-barrier step (items 0..4 in the prologue; B0/B1 of step k+2 issued at the end of step k).
    auto step0 = [&]<int KS>() {
        constexpr int sB0 = (3 * KS) % NSLOT, sB1 = (3 * KS + 1) % NSLOT, sA = (3 * KS + 2) % NSLOT;
        // B0_KS, B1_KS, A_KS of this wave landed (later-issued groups: 8 loads, or none in the last step) ...
        if constexpr (KS < KSTEPS - 1) asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
        else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        K512_BARRIER();  // ... and of every wave
        read_B<sB0>(b0, lpB0, lpB1);
        if constexpr (TA) read_AT<sA, 0>(a, bfT);
        else read_A<sA, 0>(a, lpA0, lpA1);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        K512_BARRIER();  // all waves consumed B0_KS -> its slot takes A_{KS+1}
        if constexpr (KS + 1 < KSTEPS) issue_A<sB0, KS + 1, AKSB>(rA, ldsw, offA);
        if constexpr (KS == KSTEPS - 1) bias_prefetch();
        K512_SB();
        prio_on();
        mma8<KS == 0>(acc[0], b0, a);
        prio_off();
        K512_SB();
        read_B<sB1>(b1, lpB0, lpB1);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        K512_SB();
        prio_on();
        mma8<KS == 0>(acc[1], b1, a);
        prio_off();
        K512_SB();
        if constexpr (TA) read_AT<sA, 1>(a, bfT);
        else read_A<sA, 1>(a, lpA0, lpA1);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        K512_BARRIER();  // all waves consumed B1_KS and A_KS -> their slots take B0_{KS+2}, B1_{KS+2}
        if constexpr (KS + 2 < KSTEPS) {
            issue_B<sB1, KS + 2>(rB0, ldsw, offB);
            issue_B<sA, KS + 2>(rB1, ldsw, offB);
        }
        K512_SB();
        prio_on();
        mma8<KS == 0>(acc[2], b0, a);
        mma8<KS == 0>(acc[3], b1, a);
        prio_off();
        K512_SB();
    };

    // SCHED 1: 2 barriers per step. Step start: own loads landed + barrier (= every wave landed AND finished step
    // KS-1, so the slots of B1_{KS-1} and A_{KS-1} are free) -> issue B0_{KS+1}, B1_{KS+1} into them; read b0, a0
    // and b1; barrier once b0/a0 landed (B0_KS consumed) -> issue A_{KS+1} into its slot; cA; cB interleaved with the
    // a1 reads (a[m] is reloaded as soon as the two cB MFMAs that read it are issued); cC/cD per m as a1[m] lands.
    auto step1 = [&]<int KS>() {
        constexpr int sB0 = (3 * KS) % NSLOT, sB1 = (3 * KS + 1) % NSLOT, sA = (3 * KS + 2) % NSLOT;
        constexpr bool F = (KS == 0);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        K512_BARRIER();
        if constexpr (KS + 1 < KSTEPS) {
            issue_B<(3 * KS + 3) % NSLOT, KS + 1>(rB0, ldsw, offB);
            issue_B<(3 * KS + 4) % NSLOT, KS + 1>(rB1, ldsw, offB);
        }
        if constexpr (KS == KSTEPS - 1) bias_prefetch();
        K512_SB();
        read_B<sB0>(b0, lpB0, lpB1);
        read_A<sA, 0>(a, lpA0, lpA1);
        read_B<sB1>(b1, lpB0, lpB1);
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");  // b0, a0 landed (LDS returns in order)
        K512_BARRIER();
        if constexpr (KS + 1 < KSTEPS) issue_A<sB0, KS + 1>(rA, ldsw, offA);
        K512_SB();
        prio_on();
        mma8<F>(acc[0], b0, a);
        prio_off();
        K512_SB();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");  // b1 landed
        K512_SB();
        [&]<int... M>(std::integer_sequence<int, M...>) {
            (([&] {
                 prio_on();
                 if constexpr (F) {
                     const f32x4 z = {0.f, 0.f, 0.f, 0.f};
                     acc[1][0][M] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b1[0], a[M], z, 0, 0, 0, 0, 0, 0);
                     acc[1][1][M] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b1[1], a[M], z, 0, 0, 0, 0, 0, 0);
                 } else {
                     acc[1][0][M] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b1[0], a[M], acc[1][0][M], 0, 0, 0, 0, 0, 0);
                     acc[1][1][M] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b1[1], a[M], acc[1][1][M], 0, 0, 0, 0, 0, 0);
                 }
                 prio_off();
                 K512_SB();
                 a[M] = frag<sA * SLOT + 8192 + M * 2048>(lpA0, lpA1);
                 K512_SB();
             }()),
             ...);
        }(std::make_integer_sequence<int, 4>{});
        [&]<int... M>(std::integer_sequence<int, M...>) {
            (([&] {
                 if constexpr (M == 0) asm volatile("s_waitcnt lgkmcnt(6)" ::: "memory");
                 else if constexpr (M == 1) asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
                 else if constexpr (M == 2) asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
                 else asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                 K512_SB();
                 prio_on();
#pragma unroll
                 for (int n = 0; n < 2; ++n) {
                     if constexpr (F) {
                         const f32x4 z = {0.f, 0.f, 0.f, 0.f};
                         acc[2][n][M] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b0[n], a[M], z, 0, 0, 0, 0, 0, 0);
                         acc[3][n][M] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b1[n], a[M], z, 0, 0, 0, 0, 0, 0);
                     } else {
                         acc[2][n][M] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b0[n], a[M], acc[2][n][M], 0, 0, 0, 0, 0, 0);
                         acc[3][n][M] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b1[n], a[M], acc[3][n][M], 0, 0, 0, 0, 0, 0);
                     }
                 }
                 prio_off();
                 K512_SB();
             }()),
             ...);
        }(std::make_integer_sequence<int, 4>{});
    };

    if constexpr (SCHED == 0) {
        // prologue: items 0..4 = B0_0, B1_0, A_0, B0_1, B1_1 -> slots 0..4
        issue_B<0, 0>(rB0, ldsw, offB);
        issue_B<1, 0>(rB1, ldsw, offB);
        issue_A<2, 0>(rA, ldsw, offA);
        issue_B<3, 1>(rB0, ldsw, offB);
        issue_B<4, 1>(rB1, ldsw, offB);
        step0.template operator()<0>();
        step0.template operator()<1>();
        step0.template operator()<2>();
        step0.template operator()<3>();
    } else {
        // prologue: items 0..2 = B0_0, B1_0, A_0 -> slots 0..2 (B0_1, B1_1 are issued at the start of step 0)
        issue_B<0, 0>(rB0, ldsw, offB);
        issue_B<1, 0>(rB1, ldsw, offB);
        issue_A<2, 0>(rA, ldsw, offA);
        step1.template operator()<0>();
        step1.template operator()<1>();
        step1.template operator()<2>();
        step1.template operator()<3>();
    }

    // ---- epilogue: lane owns row (16g + l%16), cols [64w + 16*(l/16), +16) of the tile for g = 0..7 ----
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
            float x0 = lo.x, x1 = lo.y, x2 = hi.x, x3 = hi.y;
            if constexpr (!NOCLAMP) {
                x0 = fminf(fmaxf(x0, -448.f), 448.f);
                x1 = fminf(fmaxf(x1, -448.f), 448.f);
                x2 = fminf(fmaxf(x2, -448.f), 448.f);
                x3 = fminf(fmaxf(x3, -448.f), 448.f);
            }
            // 'old' = x0's bits (dead after this op): avoids a v_mov 0 per dword; the high half is overwritten next
            int pk = __builtin_amdgcn_cvt_pk_fp8_f32(x0, x1, __builtin_bit_cast(int, x0), false);
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(x2, x3, pk, true);
            d[j] = pk;
        }
        int rv = rows_valid - 16 * g;
        rv = rv < 0 ? 0 : (rv > 16 ? 16 : rv);
        const uint32_t nrec = (uint32_t)__builtin_amdgcn_readfirstlane(rv * N);  // keep the rsrc scalar (no waterfall)
        const i32x4 rC = make_rsrc(Ct + (size_t)(16 * g) * N, nrec);
        k512_llvm_buffer_store_v4i32(d, rC, (int)lane_off, 0, SAUX);  // SAUX: cache policy (gfx950 bit1 = nt)
        if constexpr (TA) asm volatile("s_nop 7\n\ts_nop 7");  // tiled A: store-data guard (see the header)
        K512_SB();  // drain group by group (bounded temporaries; stores start early)
    }
}

template <int N, int NOCLAMP, int REMAP, int PRIO, int SCHED, int SAUX>
__global__ __launch_bounds__(256, 2) void k512_2wg_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                          uint8_t* __restrict__ C, const uint16_t* __restrict__ bias,
                                                          const float* __restrict__ sa, const float* __restrict__ sb,
                                                          int M) {
    k512_2wg_body<N, NOCLAMP, REMAP, PRIO, SCHED, SAUX, false>(A, B, C, bias, sa, sb, M);
}

// Tiled-A kernel (see the header): the deliverable ABI (NOCLAMP=1, REMAP=1, PRIO=1, SCHED=0, SAUX=0) with A read from
// the step-tiled yT [ceil(M/128), 512, 128].
template <int N>
__global__ __launch_bounds__(256, 2) void k512_2wg_kernel_tiled_a(const uint8_t* __restrict__ A,
                                                                  const uint8_t* __restrict__ B, uint8_t* __restrict__ C,
                                                                  const uint16_t* __restrict__ bias,
                                                                  const float* __restrict__ sa,
                                                                  const float* __restrict__ sb, int M) {
    k512_2wg_body<N, 1, 1, 1, 0, 0, true>(A, B, C, bias, sa, sb, M);
}

template <int N, int NOCLAMP, int REMAP, int PRIO, int SCHED, int SAUX = 0>
static int k512_launch(const void* A, const void* B, void* C, const void* bias, const float* sa, const float* sb, int M,
                       void* stream) {
    if (M <= 0) return (int)hipErrorInvalidValue;
    const long long tiles = (long long)((M + 127) / 128) * (N / 256);
    if (tiles > 0x7fffffffLL) return (int)hipErrorInvalidValue;
    hipLaunchKernelGGL((k512_2wg_kernel<N, NOCLAMP, REMAP, PRIO, SCHED, SAUX>), dim3((unsigned)tiles), dim3(256), 0,
                       (hipStream_t)stream, (const uint8_t*)A, (const uint8_t*)B, (uint8_t*)C, (const uint16_t*)bias, sa,
                       sb, M);
    return (int)hipGetLastError();
}

#define K512_CAT_(a, b) a##b
#define K512_CAT(a, b) K512_CAT_(a, b)

// Deliverable ABI. Default = NOCLAMP=1 (MODE.FP16_OVFL saturating convert, 1.8 VALU/output), XCD remap, s_setprio
// around MFMA blocks, SCHED 0. Bitwise == hipBLASLt on every full-tensor and edge test (incl. +-inf intermediates, where
// the explicit-clamp form gives 448 and hipBLASLt gives NaN); explicit-clamp form = k512_2wg_N{N}_c0_r1_p1_s0.
extern "C" int K512_CAT(k512_fp8_gemm_e4m3_bias_N, HK_N)(const void* A, const void* B, void* C, const void* bias,
                                                          const float* sa, const float* sb, int M, void* stream) {
    return k512_launch<HK_N, 1, 1, 1, 0>(A, B, C, bias, sa, sb, M, stream);
}

// Tiled-A launch (see the header): A = the step-tiled yT, e4m3 [ceil(M/128), 512, 128], 16-byte aligned; the tail tokens
// [M, ceil(M/128)*128) of the last block are read as stored and do not affect the output.
template <int N>
static int k512_launch_tiled_a(const void* A, const void* B, void* C, const void* bias, const float* sa, const float* sb,
                               int M, void* stream) {
    if (M <= 0 || ((uintptr_t)A % 16) != 0) return (int)hipErrorInvalidValue;
    const long long tiles = ((long long)M + 127) / 128 * (N / 256);
    if (tiles > 0x7fffffffLL) return (int)hipErrorInvalidValue;
    hipLaunchKernelGGL((k512_2wg_kernel_tiled_a<N>), dim3((unsigned)tiles), dim3(256), 0, (hipStream_t)stream,
                       (const uint8_t*)A, (const uint8_t*)B, (uint8_t*)C, (const uint16_t*)bias, sa, sb, M);
    return (int)hipGetLastError();
}
// Tiled-A entry (item B15T): the deliverable ABI kernel with A read from the step-tiled yT; bit-identical to
// k512_fp8_gemm_e4m3_bias_N{N} on the row-major y with the same bytes (ops/tests/k512_tiled_a_test.py).
extern "C" int K512_CAT(k512_fp8_gemm_e4m3_bias_tiled_a_N, HK_N)(const void* A_tiled, const void* B, void* C,
                                                                  const void* bias, const float* sa, const float* sb,
                                                                  int M, void* stream) {
    return k512_launch_tiled_a<HK_N>(A_tiled, B, C, bias, sa, sb, M, stream);
}
extern "C" const char* k512_fp8_gemm_impl(void) {
    return "2wg: 2 x 4-wave WG/CU, 128x256 tile, 5-slot 80 KiB LDS ring, swapped MFMA, FP16_OVFL sat, XCD remap (P7 iter9 G1-C)";
}

#ifdef K512_EXPERIMENT_VARIANTS
// Experiment variants, same signature: k512_2wg_N{N}_c{NOCLAMP}_r{REMAP}_p{PRIO}_s{SCHED}
#define K512_VAR(nc, rm, pr, sc)                                                                              \
    extern "C" int K512_CAT(k512_2wg_N, K512_CAT(HK_N, _c##nc##_r##rm##_p##pr##_s##sc))(                     \
        const void* A, const void* B, void* C, const void* bias, const float* sa, const float* sb, int M,      \
        void* stream) {                                                                                        \
        return k512_launch<HK_N, nc, rm, pr, sc>(A, B, C, bias, sa, sb, M, stream);                           \
    }
K512_VAR(0, 1, 1, 0)  // explicit clamp (gated first; ABI default until the FP16_OVFL form was proven)
K512_VAR(1, 1, 1, 0)  // = ABI default (FP16_OVFL no-clamp epilogue)
K512_VAR(0, 0, 1, 0)  // no XCD remap
K512_VAR(0, 1, 0, 0)  // no s_setprio
K512_VAR(0, 1, 0, 1)  // SCHED 1 (2 barriers, interleaved a1 reads), no s_setprio
K512_VAR(1, 1, 0, 1)
// non-temporal output stores (buffer_store ... nt): k512_2wg_N{N}_c{NOCLAMP}_r1_p1_s0_nt -- bitwise, but ~8% SLOWER
// in bursts than plain stores (not promoted)
#define K512_VARNT(nc)                                                                                        \
    extern "C" int K512_CAT(k512_2wg_N, K512_CAT(HK_N, _c##nc##_r1_p1_s0_nt))(                               \
        const void* A, const void* B, void* C, const void* bias, const float* sa, const float* sb, int M,      \
        void* stream) {                                                                                        \
        return k512_launch<HK_N, nc, 1, 1, 0, 2>(A, B, C, bias, sa, sb, M, stream);                           \
    }
K512_VARNT(0)
K512_VARNT(1)
#endif  // K512_EXPERIMENT_VARIANTS

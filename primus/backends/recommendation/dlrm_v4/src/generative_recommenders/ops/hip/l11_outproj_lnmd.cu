// L11 (r16 gin item (25), DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED): ONE persistent HIP kernel = the HSTU output-projection
// FP8 dgrad  dy[M,1536] = RNE_bf16(acc*(sa*sb) + 0.0f), acc = sum_k G8[m,k]*W8[n,k]  (G8 e5m2 [M,512], W8 e4m3 [1536,512];
// == torch._scaled_mm(G8, W8.t(), sa, sb, bf16) bit for bit on hipBLASLt's data-parallel tiles)  fused with the production
// LN*U*dropout backward consumer (_ln_mul_dropout_bwd_dx_du_rng_v2, num_warps=1, op for op incl. its FMA contraction and
// tl.sum DPP tree): dx [M,512], du (cols 0..511 of an [M,2048] row), per-wave fp32 dW/dB partials [P,512] that the
// production Triton finalize _ln_mul_dropout_bwd_dwdb reduces. dy never touches HBM.
//
// Provenance (r16 Arbor session /scratch/arbor_sessions/dlrmv4_20261005_r16):
//   * kernel body = probes/iter31/l11b/build/b2/l11b.cu (L11-1b builder, build b2, variant l11c_f1 = TR 2, PFD 1, SCP 0,
//     PW 1, NBUF 2, persistent; 251 VGPR, 0 VGPR spill, 0 scratch, 98304 B LDS), sha256 e862ae5b9066a19bd0e1541feea1cb7915a9059e1ab06f0545cafea7fcef96c9,
//     lines 42..714 copied VERBATIM except the template name l11_kernel -> l11_outproj_lnmd_bwd (unique trace name).
//     Builder evidence: probes/iter31/l11b/RESULT.md (du/dx bitwise vs v2 on the same dy; DP rows bitwise vs the chain).
//   * which derives from r16 iter30 L11-1a/review (probes/iter30/l11res), r14 iter37 track "od" (od_dgrad.cu), P7 iter9
//     B7 G1-C "2wg" (ops/hip/k512_fp8_gemm.cu) and HipKittens (https://github.com/HazyResearch/HipKittens, MIT License,
//     Copyright (c) 2024 HazyResearch): operand register layout and the st_16x128 XOR LDS swizzle.
//   * host ABI below (l11_plan / l11_outproj_lnmd_bwd_launch / l11_dy_debug / l11_abi_version / l11_drain_last_error /
//     l11_impl) = r16 iter31 L11-2 integration (probes/iter31/l11int), same drain contract as r14 wgrad_tm_ck_w3.cu.
// Built at run time by ops/hip/l11_outproj_lnmd.py (hipcc -> .so -> ctypes, cached under /tmp keyed by source, flags,
// arch and `hipcc --version`), in the pre-RUN_START prebuild; gfx950 only.
//
// E5 emission mode (r16 iter32 track l11e5, probes/iter32/l11e5; ABI 2): template parameter E5 = 1 reproduces the B1a
// producer contract of ship-12 (DLRM_HSTU_FP8_BWD_DU_E5M2, r16's _ln_mul_dropout_bwd_dx_du_rng_v3 EMIT kernel +
// _triton_layer_norm_mul_dropout_bwd_du_e5m2): instead of the bf16 du store it writes, per element of the u block,
//   duf = fp32(RNE_bf16(du))   (the value v2 stores; du = v2's fp32 du, op for op as in the bf16 mode)
//   e5m2 = v_cvt_scalef32_pk_bf8_f32(v_med3_f32(duf * inv, -57344, 57344), scale 1.0)   (v3's exact instructions)
// into DZ8 [rows_out, 2048] columns 0..511 (row stride 2048 B), zeroes rows [M, rows_out) of that block, and emits
// per-wave amax partials AMAXP [G*8] = RNE_bf16(max |du|) with NaN ignored (== v3's int16 max of the bf16 bit patterns
// & 0x7FFF with NaN patterns -> 0: RNE is monotone and odd) and fp32 column-sum partials of duf, DUCS [P,512], flushed
// with the dW/dB partials. The host reduces them as B1a does (partial.amax(), partial.sum(0)). dx and dW/dB are
// unchanged; the bf16 mode (E5 = 0) is unchanged.
//
// ---- original L11-1a/L11-1b design notes (unchanged) ----
// r16 iter30 "L11-1a" RESOURCE-FEASIBILITY PROBE (not a deliverable kernel): the retained-fragment FP8 out-proj dgrad that
// the fused L11 kernel (out-proj dgrad + LN*U*dropout backward, proposals/redesign_L11_outproj_dgrad_lnmd_bwd.md) needs.
//
//   dy[M,1536] (bf16) = RNE_bf16( acc * (sa*sb) + 0.0f ),  acc = sum_k dz8[m,k] * W8[n,k]   (== torch._scaled_mm(dz8, W8.t(),
//                       sa, sb, out_dtype=bf16) == hipBLASLt bit for bit, per r14's od EPI2 evidence)
//   dz8 = A: e5m2 [M,512] row-major; W8 = B: e4m3 [1536,512] row-major (storage of the weight_fp8.t() view).
//
// Shape under test (gpt-6 consult, proposals/dual/dual/20261009T000239Z_gpt6.md, adopted in the L11 proposal section 3):
//   * 512 threads (8 waves) per WG, BM = 64 rows per row block, ALL 1536 output columns per WG, computed as 6 sequential
//     N-tiles of 256 columns; BK = 128, 4 in-order K steps per tile (K = 512).
//   * LDS: A tile 64 x 512 e5m2 (32 KiB) loaded once per row block and reused by the 6 N-tiles; weight K-chunks 256 x 128
//     e4m3 (32 KiB) in an NBUF-deep ring (NBUF = 2 -> 96 KiB total, the consult's shape; NBUF = 3 -> 128 KiB) -> 1 WG/CU.
//   * Wave w owns output columns [32w, 32w+32) x all 64 rows of each 64x256 tile = 8 of its 64 16x16 blocks:
//     acc[2][4] f32x4 = 32 fp32/lane. After each tile: od EPI2 epilogue (acc*(sa*sb) + 0.0f, RNE bf16) packed to bf16 pairs
//     retained in registers: pk[6][4][4] = 96 packed VGPRs/lane after the 6 tiles.
//   * Redistribution: 8 rows at a time (8 x 1536 x 2 B = 24 KiB, row pitch 3088 B) through the weight buffer that held the
//     last K chunk, into the consumer layout of r16's Triton v2 LN*U*dropout backward (num_warps=1, one wave per row,
//     8 contiguous columns per lane): wave w takes row 8*it + w, lane l holds columns [8l, 8l+8) of the u (0..511),
//     x (512..1023) and y (1024..1535) planes = 3 x ds_read_b128.
//   * Consumer = STUB: STUB=0 debug: write the redistributed rows to C [M,1536] bf16 (3 coalesced 16 B stores per lane per
//     row) for the bitwise check; STUB=1 timing stub: per-row fp32 sum of the y plane (in-lane + DPP/readlane reduction)
//     written to R [M] fp32, u/x planes xor-folded into a register consumed only by a never-taken store (keeps the work).
//   * Grid: persistent (min(ceil(M/64), #CU) WGs walking row blocks blk, blk+G, ...; the next block's A tile is
//     LDS-DMA-prefetched during the redistribution; the weight ring runs across row blocks) or non-persistent (ceil(M/64)
//     WGs, same kernel, one block each).
//   * APF=1: the A fragments of K step j+1 are read (into a second register set) right after step j's B reads, before its
//     MFMAs, so after each barrier only the 2 weight fragments are on the critical path.
//
// Derived from r14 iter37 track "od" (/scratch/arbor_sessions/dlrmv4_20261005_r14/probes/iter37/od/od_dgrad.cu, sha256
// 48c85ed2...), itself derived from P7 iter9 B7 G1-C "2wg" (Arbor P7; /scratch/training_dlrmv4_arbor_pr7/recommendation/
// generative_recommenders/ops/hip/k512_fp8_gemm.cu). Operand register layout and the st_16x128 XOR LDS swizzle follow
// HipKittens (https://github.com/HazyResearch/HipKittens, MIT License, Copyright (c) 2024 HazyResearch) as in P7/od.
// Borrowed unchanged from od: buffer resource, barrier macros, ds_read16/frag, the LDS-DMA (buffer_load ... lds) helper,
// the per-lane 16 B swizzle (row R, 16 B position p holds K chunk p ^ ((R>>1)&7)), the fragment read offsets, the operand-
// swapped MFMA (src0 = weight fragment e4m3 cbsz=0, src1 = gradient fragment e5m2 blgp=1, step 0 with inline-0 srcC, K steps
// in order), od's STORE=1 B-row permutation inside each 32-column group (lane holds 8 consecutive output columns), and the
// EPI2 epilogue (v_pk_mul_f32, v_pk_add_f32 +0.0f, v_cvt_pk_bf16_f32). New here: 64x1536 row blocks with 8 waves, the A tile
// resident across 6 N-tiles, the NBUF ring across tiles/row blocks, register retention, the LDS redistribution and stubs.
//
// Build: hipcc -DKITTENS_CDNA4 --offload-arch=gfx950 -std=c++20 -O3 -mllvm -amdgpu-sched-strategy=iterative-ilp
//        -ffp-contract=off -shared -fPIC [-DL11_ONLY=<variant id>] (build.py).
#include <hip/hip_runtime.h>
#include <stdint.h>

#include <utility>

namespace l11 {
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}  // namespace l11

extern "C" __device__ void l11_llvm_buffer_load_lds(l11::i32x4 rsrc, l11::lds_ptr_t lds, int size, int voffset,
                                                    int soffset, int offset, int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
extern "C" __device__ void l11_llvm_buffer_store_v4i32(l11::i32x4 vdata, l11::i32x4 rsrc, int voffset, int soffset,
                                                       int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");
extern "C" __device__ l11::i32x4 l11_llvm_buffer_load_v4i32(l11::i32x4 rsrc, int voffset, int soffset,
                                                            int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
extern "C" __device__ void l11_llvm_buffer_store_f32(float vdata, l11::i32x4 rsrc, int voffset, int soffset,
                                                     int aux) __asm("llvm.amdgcn.raw.buffer.store.f32");
// REVIEW ADDITION (r16 iter30 l11res review, NOT the builder's code): 8-byte buffer load for the consumer-traffic stub
typedef int rv_i32x2 __attribute__((ext_vector_type(2)));
extern "C" __device__ rv_i32x2 l11_llvm_buffer_load_v2i32(l11::i32x4 rsrc, int voffset, int soffset,
                                                          int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
// L11-1b ADDITION (r16 iter31 l11b): 4-byte buffer load (mean / rstd, lane-uniform address)
extern "C" __device__ float l11_llvm_buffer_load_f32(l11::i32x4 rsrc, int voffset, int soffset,
                                                     int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");
// ROCm 10.2 (T14b; ops/hip/l11_outproj_lnmd.py tlsum_tree()): the consumer's in-lane row sums c1 = sum(xhat * wdy),
// c2 = sum(wdy) over a lane's 8 columns follow the Triton that compiles v2 / v3 (whose dx this kernel reproduces bit
// for bit). 0 (default; Triton 3.6): sequential, ((e0 + e1) + e2) + ... + e7. 1 (-DL11_TLSUM_TREE=1, Triton >= 3.9,
// whose reduce lowering is a 2-wide tree that LLVM then contracts): column pairs (0,1) and (4,5) fused with (2,3) and
// (6,7), fma(a, b, RN(a' * b')) per lane (v_pk_fma_f32; a * b = xhat * wdy for c1, dn * w for c2), then (p01 + p45),
// then even + odd column; read from the Triton 3.9 gfx950 ISA of v2 and v3 (identical structure). The DPP tree after
// it (tl_sum_wave) is the same in both.
#ifndef L11_TLSUM_TREE
#define L11_TLSUM_TREE 0
#endif
// l11e5 ADDITION (E5 mode): 8-byte buffer store (8 e5m2 du bytes per lane)
extern "C" __device__ void l11_llvm_buffer_store_v2i32(rv_i32x2 vdata, l11::i32x4 rsrc, int voffset, int soffset,
                                                       int aux) __asm("llvm.amdgcn.raw.buffer.store.v2i32");

// L11-1b ADDITION: arguments of the real LN*U*dropout backward consumer (TR == 2). Production contract of
// _ln_mul_dropout_bwd_dx_du_rng_v2 (triton_hstu_linear.py:1640, launched :3134) with SILU_U, CONCAT_U = CONCAT_X, 'silu',
// COMPUTE_Y=False, TRAINING, D = BLOCK_D = 512: x/u bf16 [M,512] contiguous, mask int8 [M,512] (bit0 dy, bit1 dx, bit2 du
// keep), mean/rstd fp32 [M], norm weight/bias bf16 [512], dx bf16 [M,512], du bf16 rows of an [M,2048] buffer (cols 0..511).
// dW/dB: fp32 per-wave partials, flushed every `fl` row blocks of the WG to row (blockIdx.x*8 + wave)*nf + slot of
// DWP/DBP [P = grid*8*nf, 512] (finalize = _ln_mul_dropout_bwd_dwdb over P rows).
struct L11CArgs {
    const uint16_t* X;
    const uint16_t* U;
    const uint8_t* MK;
    const float* MEAN;
    const float* RSTD;
    const uint16_t* WN;
    const uint16_t* BN;
    uint16_t* DX;
    uint16_t* DU;
    float* DWP;
    float* DBP;
    float dscale;  // fp32 1/(1-p) (IEEE, as v2 computes it)
    float invD;    // fp32 1/D (IEEE) = 2^-9
    int fl;        // row blocks per partial flush
    int nf;        // partial slots per wave
    // l11e5 ADDITION (read only by the E5 = 1 instantiation; unused, null/0, in the bf16 mode): the B1a contract
    uint8_t* DZ8;      // e5m2 [rows_out, 2048]: columns 0..511 written (rows < M payload, rows [M, rows_out) zero)
    const float* INVS; // fp32 [1] device inv_scale the payload is cast with (the uvqk site state's inv_scale)
    float* AMAXP;      // fp32 [G*8] per-wave amax partials
    float* DUCS;       // fp32 [P,512] u column-sum partials (same rows / flush points as DWP/DBP)
    int rows_out;      // padded row count of DZ8 (>= M)
};

namespace l11 {

constexpr int KD = 512;          // K
constexpr int BM = 64;           // rows per row block
constexpr int NOUT = 1536;       // output columns (all owned by the WG)
constexpr int NT = 6;            // N tiles of 256
constexpr int NCH = 24;          // weight K chunks per row block (6 tiles x 4 K steps)
constexpr int ABYTES = 32768;    // A tile: 4 K steps x 64 rows x 128 B
constexpr int WCH = 32768;       // weight chunk: 256 rows (output cols) x 128 B (K)
constexpr uint32_t ROWB = 3072;  // bytes per dy row
constexpr uint32_t SPITCH = 3088;  // staging row pitch (3072 + 16: 4-bank shift per row)

// od / K512:75-83
__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}

// od / K512:85-93
#define L11_FENCE() asm volatile("" ::: "memory")
#define L11_SB() __builtin_amdgcn_sched_barrier(0)
#define L11_BARRIER()                 \
    do {                              \
        L11_FENCE();                  \
        __builtin_amdgcn_s_barrier(); \
        L11_FENCE();                  \
        L11_SB();                     \
    } while (0)

// od / K512:95-133
template <int IMM>
__device__ __forceinline__ i32x4 ds_read16(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x4 v;
    asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
template <int IMM>
__device__ __forceinline__ void ds_write16(uint32_t vaddr, i32x4 v) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    asm volatile("ds_write_b128 %0, %1 offset:%2" : : "v"(vaddr), "v"(v), "i"(IMM) : "memory");
}
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
// A fragments of K step (OFF = k*8192) for the 4 row groups m (16 rows each)
template <int OFF>
__device__ __forceinline__ void read_A4(i32x8 (&a)[4], uint32_t lp0, uint32_t lp1) {
    a[0] = frag<OFF + 0 * 2048>(lp0, lp1);
    a[1] = frag<OFF + 1 * 2048>(lp0, lp1);
    a[2] = frag<OFF + 2 * 2048>(lp0, lp1);
    a[3] = frag<OFF + 3 * 2048>(lp0, lp1);
}

template <int N>
__device__ __forceinline__ void wait_vm() {
    static_assert(N >= 0 && N <= 63, "vmcnt is 6 bit");
    if constexpr (N == 0) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    else if constexpr (N == 4) asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
    else if constexpr (N == 8) asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
    else if constexpr (N == 16) asm volatile("s_waitcnt vmcnt(16)" ::: "memory");
    else if constexpr (N == 24) asm volatile("s_waitcnt vmcnt(24)" ::: "memory");
    else static_assert(N < 0, "add the vmcnt value");
}
template <int N>
__device__ __forceinline__ void wait_lgkm() {
    if constexpr (N == 0) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    else if constexpr (N == 4) asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
    else static_assert(N < 0, "add the lgkmcnt value");
}

// od / K512:135-150 operand-swapped MFMA: src0 = weight fragment (e4m3, cbsz=0), src1 = gradient fragment (e5m2, blgp=1).
// acc[n][m] (+)= Wfrag[n] (16 output cols) x Gfrag[m] (16 output rows); lane holds 4 consecutive output cols of 1 row.
template <bool FIRST, int M0>
__device__ __forceinline__ void mma4(f32x4 (&c)[2][4], const i32x8 (&b)[2], const i32x8 (&a)[4]) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int m = M0; m < M0 + 2; ++m) {
            if constexpr (FIRST) {
                const f32x4 z = {0.f, 0.f, 0.f, 0.f};
                c[n][m] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[n], a[m], z, 0, 1, 0, 0, 0, 0);
            } else {
                c[n][m] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[n], a[m], c[n][m], 0, 1, 0, 0, 0, 0);
            }
        }
    }
}

// od / K512:152-168
__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    l11_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, 0);
}

__device__ __forceinline__ uint32_t pk_bf16(f32x2 v) {  // RNE fp32 -> bf16 pair (gfx950: v_cvt_pk_bf16_f32)
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2));
}

// od EPI2: RNE_bf16(acc*(sa*sb) + 0.0f) (fp-contract off; +0.0f not folded without nsz) -> hipBLASLt's sign of zero
__device__ __forceinline__ uint32_t epi2(f32x2 v, f32x2 s2, f32x2 z2) {
    v = v * s2;
    v = v + z2;
    return pk_bf16(v);
}

__device__ __forceinline__ float bf_lo(uint32_t u) { return __builtin_bit_cast(float, u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __builtin_bit_cast(float, u & 0xffff0000u); }

template <int CTRL>
__device__ __forceinline__ float dpp_sum_step(float v) {
    const int s = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, false);
    return v + __builtin_bit_cast(float, s);
}
// full-wave fp32 sum (stub only; the real consumer replicates Triton's tl.sum tree)
__device__ __forceinline__ float wave_sum(float v) {
    v = dpp_sum_step<0xB1>(v);   // quad_perm [1,0,3,2]
    v = dpp_sum_step<0x4E>(v);   // quad_perm [2,3,0,1]
    v = dpp_sum_step<0x141>(v);  // row_half_mirror
    v = dpp_sum_step<0x140>(v);  // row_mirror -> every lane holds its 16-lane row sum
    const int iv = __builtin_bit_cast(int, v);
    const float r0 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(iv, 0));
    const float r1 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(iv, 16));
    const float r2 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(iv, 32));
    const float r3 = __builtin_bit_cast(float, __builtin_amdgcn_readlane(iv, 48));
    return (r0 + r1) + (r2 + r3);
}

template <int N, typename F>
__device__ __forceinline__ void static_for(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(
        std::make_integer_sequence<int, N>{});
}

// ---- L11-1b ADDITION: v2 consumer helpers. Op order read from the production v2 gfx950 ISA
// (probes/iter30/l11price/out/main_*/isa_prod_v2.s, 760 instructions, 129 VGPR).
__device__ __forceinline__ float bfe8(const i32x4 v, int e) {  // element e (col 8l+e) of 8 packed bf16 -> fp32
    const uint32_t d = (uint32_t)v[e >> 1];
    return (e & 1) ? bf_hi(d) : bf_lo(d);
}
// v_add_f32_dpp v, v, v <ctrl> row_mask:0xf bank_mask:0xf bound_ctrl:1  ==  dpp(v) + v (invalid source lanes read 0)
template <int CTRL>
__device__ __forceinline__ float dpp_add(float v) {
    const int s = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, true);
    return __builtin_bit_cast(float, s) + v;
}
// Triton tl.sum over the 64-lane wave exactly as compiled for v2 (num_warps=1, gfx950): after the in-lane sequential sum,
//   add_dpp row_shr 8,4,2,1 (bound_ctrl), v_mov_dpp row_bcast:15 row_mask:0xa + add, add_dpp row_bcast:31, readlane 63.
__device__ __forceinline__ float tl_sum_wave(float v) {
    v = dpp_add<0x118>(v);  // row_shr:8
    v = dpp_add<0x114>(v);  // row_shr:4
    v = dpp_add<0x112>(v);  // row_shr:2
    v = dpp_add<0x111>(v);  // row_shr:1
    const int t = __builtin_amdgcn_update_dpp(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v), 0x142, 0xA, 0xF,
                                              true);  // row_bcast:15, rows 1 and 3 only (rows 0, 2 keep v)
    v = __builtin_bit_cast(float, t) + v;
    v = dpp_add<0x143>(v);  // row_bcast:31
    return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
}
// tl.sigmoid(u) on gfx950 = IEEE 1/(1 + llvm.exp.f32(-u)); r16 _sigmoid_ieee_fast (triton_hstu_linear.py:617): for
// d = 1 + v_exp(RN(-u*log2e)) in [1, 2^126] the IEEE fdiv reduces to v_rcp + 5 FMAs (bit-identical). Callers take the
// wave-uniform IEEE-division fallback when any d is outside that range (u < -87.3 or NaN).
__device__ __forceinline__ float sig_d(float u) {
    return 1.0f + __builtin_amdgcn_exp2f(-u * __builtin_bit_cast(float, 0x3fb8aa3bu));
}
__device__ __forceinline__ float sig_fast(float d) {
    float r = __builtin_amdgcn_rcpf(d);
    float t = __builtin_fmaf(-d, r, 1.0f);
    r = __builtin_fmaf(t, r, r);
    t = __builtin_fmaf(-d, r, 1.0f);
    float q = __builtin_fmaf(t, r, r);
    t = __builtin_fmaf(-d, q, 1.0f);
    q = __builtin_fmaf(t, r, q);
    return q;
}

// l11e5: 4 fp32 -> 4 e5m2 bytes (a = byte 0 .. d = byte 3) with the instruction Triton emits for v3's
// `scaled.to(tl.float8e5)` on gfx950: v_cvt_scalef32_pk_bf8_f32 (RNE), scale 1.0; word 0 = (a, b), word 1 = (c, d)
// (op_sel dst-high). Inputs are already clamped to [-57344, 57344] by v_med3_f32 (NaN -> -57344, as v3).
typedef short s16x2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ int cvt_pk_e5m2(float a, float b, float c, float d) {
#if __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_bf8_f32)
    // word 0 into a fresh register (its word 1 is don't-care, overwritten next), then word 1
    int w0;
    asm("v_cvt_scalef32_pk_bf8_f32 %0, %1, %2, 1.0" : "=v"(w0) : "v"(a), "v"(b));
    const s16x2 v = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(__builtin_bit_cast(s16x2, w0), c, d, 1.0f, true);
    return __builtin_bit_cast(int, v);
#else
    int v = 0;
    asm("v_cvt_scalef32_pk_bf8_f32 %0, %1, %2, 1.0" : "+v"(v) : "v"(a), "v"(b));
    asm("v_cvt_scalef32_pk_bf8_f32 %0, %1, %2, 1.0 op_sel:[0,0,0,1]" : "+v"(v) : "v"(c), "v"(d));
    return v;
#endif
}

}  // namespace l11

// STUB: 0 = debug store of the redistributed dy rows to C [M,1536] bf16; 1 = y-plane row sum to R [M] fp32 (timing stub).
// NBUF: weight-chunk ring depth (2 = consult shape, 96 KiB LDS; 3 = 128 KiB). APF: prefetch next step's A fragments.
// DIAG (timing attribution only; outputs meaningless for 2/3): 0 normal; 1 = redistribution + stub run twice per row block;
//   2 = no steady-state weight LDS-DMA (fragments read stale LDS); 3 = 2 + fragment reads only in step 0 of a block
//   (MFMA chain + epilogue + redistribution + stub + the A-tile HBM stream).
// PW (r16 addition): 1 = per-wave private weight ring: wave w LDS-DMAs only its own 32 output columns (4 x 1 KiB full-line
//   instructions into its 4 KiB slice of the ring buffer, same row order / swizzle as the shared layout), so the B
//   fragments need no cross-wave sync: no barrier in K steps 1..23 (A tile: barrier at step 0 only). LDS bytes unchanged.
// REVIEW ADDITION: TR = 1 -> consumer-traffic stub on top of the STUB=1 row-sum stub: per redistributed row, wave w loads
// x (bf16 [M,512], 16 B/lane), u (bf16 [M,512], 16 B/lane), mask (int8 [M,512], 8 B/lane), mean/rstd (fp32 [M,2], 8 B,
// lane-uniform) and stores dx (bf16 [M,512], 16 B/lane) and du (bf16 rows of an [M,2048] buffer, 16 B/lane): the real
// consumer's HBM bytes (4616 B/row), no math beyond a bit fold. Loads for row it+PFD are issued in iteration it (register
// prefetch depth PFD; rows 0..PFD-1 right after the A-tile prefetch).
// L11-1b ADDITION: TR = 2 -> the REAL consumer (v2's per-element math, op for op, per redistributed row; see the TR == 2
// block below) with the same loads/stores and addresses as the TR = 1 stub (mean and rstd from separate fp32 [M] arrays),
// no row-sum store; dW/dB per-wave VGPR partials flushed every ca.fl row blocks. SCP: dx/du store cache policy
// (0 = default as the TR=1 stub, 1 = sc0 nt as v2's ".cs").
// l11e5 ADDITION: E5 = 1 (TR == 2 only) -> the B1a e5m2 du emission instead of the bf16 du store (header, "E5 emission
// mode"); E5 = 0 compiles exactly the code above.
template <int STUB, int NBUF, int APF, int DIAG = 0, int PW = 0, int TR = 0, int PFD = 2, int SCP = 0, int E5 = 0>
__global__ __launch_bounds__(512, 2) void l11_outproj_lnmd_bwd(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                     uint16_t* __restrict__ C, float* __restrict__ R,
                                                     const float* __restrict__ sa, const float* __restrict__ sb, int M,
                                                     int nblk, const uint16_t* __restrict__ X,
                                                     const uint16_t* __restrict__ U, const uint8_t* __restrict__ MK,
                                                     const float* __restrict__ MR, uint16_t* __restrict__ DX,
                                                     uint16_t* __restrict__ DU, const L11CArgs ca) {
#pragma clang fp contract(off)
    using namespace l11;
    static_assert(NCH % NBUF == 0, "ring position of a chunk must not depend on the row block");
    static_assert(TR == 0 || (STUB == 1 && DIAG == 0), "traffic stub is built on the row-sum stub");
    static_assert(E5 == 0 || TR == 2, "l11e5: the e5m2 du emission is a mode of the real consumer");
    constexpr int WL = ABYTES;                  // weight buffers start
    constexpr int SBUF = (NCH - 1) % NBUF;      // buffer of the block's last chunk = redistribution staging
    constexpr int STG = WL + SBUF * WCH;
    static_assert(8 * SPITCH <= (uint32_t)WCH, "staging fits one weight buffer");
    // vector-memory stores per redistribution iteration per wave (TR=2: dx + du; the dW/dB flush stores come on top,
    // which only makes the step-0 wait_vm<S_ST> stricter)
    constexpr int NST = STUB == 0 ? 3 : (TR == 2 ? 2 : (TR ? 3 : 1));
    constexpr int NRED = DIAG == 1 ? 2 : 1;     // redistribution passes per row block
    constexpr int S_ST = 8 * NST * NRED;        // ... per row block (all issued after the next block's A prefetch)
    static_assert(DIAG == 0 || (STUB == 1 && APF == 0), "DIAG variants are stub-only, APF=0");
    constexpr int VM_RING = NBUF == 3 ? 4 : 0;  // DMA instructions allowed in flight behind the chunk we wait for
    __shared__ __attribute__((aligned(2048))) uint8_t smem[ABYTES + NBUF * WCH];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int r16 = lane & 15, q4 = lane >> 4;

    const float sav = sa[0], sbv = sb[0];
    float s = sav * sbv;
    asm volatile("" : "+v"(s));  // materialize s (and so retire the scale SMEM loads) before any hand-counted LDS op
    float inv5 = 0.f;  // l11e5: the B1a cast's inv_scale (device scalar, read once like sa/sb)
    if constexpr (E5) {
        // a vector load (the struct pointer is not noalias, so no s_load): wave-uniform -> SGPR, and retired here,
        // before the hand-counted vmcnt/lgkmcnt bookkeeping starts
        inv5 = __builtin_bit_cast(float, __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, ca.INVS[0])));
        asm volatile("" : "+s"(inv5));
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");  // no SMEM in flight once the LDS lgkmcnt bookkeeping starts
    const f32x2 s2 = {s, s}, z2 = {0.f, 0.f};

    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);
    const uint32_t ldsw = lds0 + (uint32_t)w * 1024u;  // this wave's LDS-DMA destination (hardware adds lane*16)

    // ---- per-lane LDS-DMA source offsets: LDS row rr (+64 per instruction), 16 B position p holds K chunk p^((rr>>1)&7)
    const int rr = 8 * w + (lane >> 3);
    const int p = lane & 7;
    const uint32_t kswz = (uint32_t)(16 * (p ^ ((rr >> 1) & 7)));
    const uint32_t voffA = (uint32_t)rr * KD + kswz;
    // weight LDS row Rw = 32g + 16n + r holds output column 32g + 8(r>>2) + 4n + (r&3) of the tile (od STORE=1 permutation)
    const int colw = 32 * (rr >> 5) + 8 * ((rr & 15) >> 2) + 4 * ((rr >> 4) & 1) + (rr & 3);
    const uint32_t voffW = (uint32_t)colw * KD + kswz;
    // PW: instruction i of wave w fills LDS rows 32w + 8i + r8 (r8 = lane>>3) = output columns
    // 32w + cm0 + 16(i&1) + 4(i>>1); K-chunk swizzle p ^ ((r8>>1) | 4(i&1)) (row bits 1..3)
    const int r8 = lane >> 3;
    const int cm0 = 8 * (r8 >> 2) + (r8 & 3);
    const uint32_t voffWe = (uint32_t)(32 * w + cm0) * KD + (uint32_t)(16 * (p ^ (r8 >> 1)));
    const uint32_t voffWo = (uint32_t)(32 * w + cm0) * KD + (uint32_t)(16 * (p ^ ((r8 >> 1) | 4)));
    const i32x4 rW = make_rsrc(B, (uint32_t)(NOUT * KD));

    // ---- fragment read offsets (od / K512:231-236)
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t lp0 = lds0 + (uint32_t)r16 * 128u + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lp1 = lds0 + (uint32_t)r16 * 128u + (((uint32_t)(16 * q4 + 64)) ^ swz);
    const uint32_t lpB0 = lp0 + (uint32_t)WL + (uint32_t)w * 4096u;
    const uint32_t lpB1 = lp1 + (uint32_t)WL + (uint32_t)w * 4096u;
    // ---- redistribution staging: producer (row r16&7 of the 8-row slab, cols 32w+8q4 of each tile), consumer (row w)
    const uint32_t wst = lds0 + (uint32_t)STG + (uint32_t)(r16 & 7) * SPITCH + 64u * (uint32_t)w + 16u * (uint32_t)q4;
    const uint32_t rst = lds0 + (uint32_t)STG + (uint32_t)w * SPITCH + 16u * (uint32_t)lane;
    const uint32_t voffC = (uint32_t)w * ROWB + 16u * (uint32_t)lane;

    auto issue_A = [&](const i32x4 rA) {
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, ldsw + (uint32_t)(i * 8192), voffA, (uint32_t)(i * 128));
    };
    int zs = 0;  // opaque SGPR zero, re-materialized per row block (see the loop head)
    auto issue_W = [&](auto CH_) {  // weight chunk CH (tile CH/4, K step CH%4) -> ring buffer CH % NBUF
        constexpr int CH = decltype(CH_)::value;
        constexpr int t = CH / 4, k = CH % 4, buf = CH % NBUF;
        if constexpr (PW) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                g2l(rW, lds0 + (uint32_t)(WL + buf * WCH + i * 1024) + (uint32_t)w * 4096u, (i & 1) ? voffWo : voffWe,
                    (uint32_t)(zs + ((16 * (i & 1) + 4 * (i >> 1)) * KD + t * 131072 + k * 128)));
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                g2l(rW, ldsw + (uint32_t)(WL + buf * WCH + i * 8192), voffW,
                    (uint32_t)(zs + (i * 32768 + t * 131072 + k * 128)));
        }
    };

    int blk = (int)blockIdx.x;
    const int G = (int)gridDim.x;
    {  // prologue: A tile of the first block, then the first NBUF-1 weight chunks
        const int row0 = blk * BM;
        issue_A(make_rsrc(A + (size_t)row0 * KD, (uint32_t)(min(BM, M - row0) * KD)));
        issue_W(std::integral_constant<int, 0>{});
        if constexpr (NBUF == 3) issue_W(std::integral_constant<int, 1>{});
    }
    bool first = true;
    uint32_t chk = 0;

    // ---- L11-1b ADDITION: consumer state (TR == 2): norm weight/bias of this lane's 8 columns (fp32), dW/dB partials.
    constexpr int NC = TR == 2 ? 8 : 1;
    float pdw[NC], pdb[NC];
    i32x4 wq{}, bq{};  // norm weight / bias of this lane's 8 columns, packed bf16 (8 VGPR), unpacked per use
    int lb = 0, slot = 0;  // row blocks done by this WG, partial slots flushed
    // l11e5 (E5): u column-sum partials of fp32(RNE_bf16(du)) for this lane's 8 columns (same chains as pdw/pdb) and
    // the running max of |du| (maxnum: NaN ignored; RNE_bf16 is applied once to the wave max at the end)
    constexpr int NC5 = (TR == 2 && E5) ? 4 : 1;
    f32x2 pcs[NC5];  // column pairs (2k, 2k+1) of this lane's 8 columns: v_pk_add_f32 (two IEEE fp32 adds)
    float am5 = 0.f;
    if constexpr (TR == 2) {
        wq = *reinterpret_cast<const i32x4*>(ca.WN + 8 * lane);
        bq = *reinterpret_cast<const i32x4*>(ca.BN + 8 * lane);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            pdw[e] = 0.f;
            pdb[e] = 0.f;
        }
        if constexpr (E5) {
#pragma unroll
            for (int e = 0; e < NC5; ++e) pcs[e] = f32x2{0.f, 0.f};
        }
    }
    auto flush = [&](int sl) {  // partial row (blockIdx.x*8 + w)*nf + sl, this lane's 8 columns; then restart the chains
        if constexpr (TR == 2) {
            const size_t off = ((size_t)(blockIdx.x * 8 + w) * (size_t)ca.nf + (size_t)sl) * 512u + 8u * (size_t)lane;
            *reinterpret_cast<f32x4*>(ca.DWP + off) = f32x4{pdw[0], pdw[1], pdw[2], pdw[3]};
            *reinterpret_cast<f32x4*>(ca.DWP + off + 4) = f32x4{pdw[4], pdw[5], pdw[6], pdw[7]};
            *reinterpret_cast<f32x4*>(ca.DBP + off) = f32x4{pdb[0], pdb[1], pdb[2], pdb[3]};
            *reinterpret_cast<f32x4*>(ca.DBP + off + 4) = f32x4{pdb[4], pdb[5], pdb[6], pdb[7]};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pdw[e] = 0.f;
                pdb[e] = 0.f;
            }
            if constexpr (E5) {  // l11e5: the u column-sum partial row, same index as the dW/dB row
                *reinterpret_cast<f32x4*>(ca.DUCS + off) = f32x4{pcs[0].x, pcs[0].y, pcs[1].x, pcs[1].y};
                *reinterpret_cast<f32x4*>(ca.DUCS + off + 4) = f32x4{pcs[2].x, pcs[2].y, pcs[3].x, pcs[3].y};
#pragma unroll
                for (int e = 0; e < NC5; ++e) pcs[e] = f32x2{0.f, 0.f};
            }
        }
    };

    for (; blk < nblk; blk += G) {
        // the 96 distinct weight-DMA soffsets are formed from this loop-variant opaque zero next to their use, instead
        // of being hoisted out of the loop into SGPRs (t1 build: 106 SGPR, 5 SGPR spills in the stub variant)
        asm volatile("s_mov_b32 %0, 0" : "=s"(zs));
        const int row0 = blk * BM;
        const int rv = min(BM, M - row0);
        const bool has_next = blk + G < nblk;

        i32x4 pk[NT][4];  // retained dy: [tile][16-row group] = 8 bf16 (cols 256t+32w+8q4..+8, row 16m+r16)
        f32x4 acc[2][4];
        i32x8 af[APF ? 2 : 1][4];
        i32x8 bf[2];

        static_for<NCH>([&](auto J_) {
            constexpr int J = decltype(J_)::value;
            constexpr int t = J / 4, k = J % 4, buf = J % NBUF;
            constexpr int as = APF ? (J & 1) : 0;
            // chunk J (and at J == 0 the A tile) must have landed: in-order vmcnt bookkeeping (see header of wait_vm users)
            if constexpr (J == 0) {
                if (first) wait_vm<VM_RING>();
                else wait_vm<S_ST>();
            } else {
                wait_vm<VM_RING>();
            }
            if constexpr (!PW || J == 0) {
                L11_BARRIER();  // all waves' DMA parts of chunk J visible; all waves done reading chunk J-1 / old A tile
            } else {
                L11_SB();       // PW: chunk J is this wave's own DMA (vmcnt above); no cross-wave dependency
            }
            if constexpr (DIAG != 2 && DIAG != 3)
                issue_W(std::integral_constant<int, (J + NBUF - 1) % NCH>{});  // always issued (also past the last block)
            L11_SB();
            if constexpr (DIAG != 3 || J == 0) {
                bf[0] = frag<buf * WCH + 0>(lpB0, lpB1);
                bf[1] = frag<buf * WCH + 2048>(lpB0, lpB1);
            }
            if constexpr (DIAG == 3 && J != 0) {  // stale fragments, but opaque: keeps all 192 MFMAs (no tile CSE)
                asm volatile("" : "+v"(bf[0]), "+v"(bf[1]), "+v"(af[as][0]), "+v"(af[as][1]), "+v"(af[as][2]),
                             "+v"(af[as][3]));
            }
            if constexpr (!APF || J == 0) {
                if constexpr (DIAG != 3 || J == 0) read_A4<k * 8192>(af[as], lp0, lp1);
                if constexpr (APF) read_A4<((J + 1) % 4) * 8192>(af[as ^ 1], lp0, lp1);  // J == 0: also step 1's
                wait_lgkm<APF ? 0 : 4>();  // !APF: B frags + A frags m=0,1 landed (LDS returns in order)
                L11_SB();
                if constexpr (APF) {
                    mma4<k == 0, 0>(acc, bf, af[as]);
                    mma4<k == 0, 2>(acc, bf, af[as]);
                } else {
                    mma4<k == 0, 0>(acc, bf, af[as]);
                    L11_SB();
                    wait_lgkm<0>();
                    L11_SB();
                    mma4<k == 0, 2>(acc, bf, af[as]);
                }
            } else {
                wait_lgkm<0>();  // B frags of this step + A frags prefetched during the previous step
                L11_SB();
                if constexpr (J + 1 < NCH) read_A4<((J + 1) % 4) * 8192>(af[as ^ 1], lp0, lp1);  // next step's A
                L11_SB();
                mma4<k == 0, 0>(acc, bf, af[as]);
                mma4<k == 0, 2>(acc, bf, af[as]);
            }
            L11_SB();
            if constexpr (k == 3) {  // tile t done: od EPI2 epilogue, retain packed bf16 (8 consecutive cols per lane/row)
                static_for<4>([&](auto M_) {
                    constexpr int m = decltype(M_)::value;
                    i32x4 q;
                    q.x = (int)epi2(__builtin_shufflevector(acc[0][m], acc[0][m], 0, 1), s2, z2);
                    q.y = (int)epi2(__builtin_shufflevector(acc[0][m], acc[0][m], 2, 3), s2, z2);
                    q.z = (int)epi2(__builtin_shufflevector(acc[1][m], acc[1][m], 0, 1), s2, z2);
                    q.w = (int)epi2(__builtin_shufflevector(acc[1][m], acc[1][m], 2, 3), s2, z2);
                    // Pin the packed tuple here: without this barrier LLVM sinks the whole epilogue into the (divergent)
                    // redistribution blocks that use it and keeps 6 tiles of fp32 accumulators live (t1 build: 256 VGPR
                    // + 12..45 spilled). asm volatile is neither sunk nor reordered against the step's other asm.
                    asm volatile("" : "+v"(q));
                    pk[t][m] = q;
                });
            }
        });

        // ---- all 6 tiles retained. Free the A tile and the last chunk's buffer, prefetch the next block's A tile.
        L11_BARRIER();
        if (has_next) {
            const int nrow0 = (blk + G) * BM;
            issue_A(make_rsrc(A + (size_t)nrow0 * KD, (uint32_t)(min(BM, M - nrow0) * KD)));
        }
        L11_SB();

        const i32x4 rC = make_rsrc(C + (size_t)row0 * NOUT, (uint32_t)rv * ROWB);
        const i32x4 rR = make_rsrc(R + row0, (uint32_t)rv * 4u);
        // REVIEW ADDITION: consumer-traffic stub (TR) resources, per-block 64-bit bases, range-checked like rC
        i32x4 rX{}, rU{}, rM{}, rMR{}, rDX{}, rDU{};
        i32x4 rMn{}, rRs{};  // L11-1b: separate mean / rstd fp32 [M]
        i32x4 px[TR ? PFD + 1 : 1], pu[TR ? PFD + 1 : 1];
        rv_i32x2 pm[TR ? PFD + 1 : 1], pr[TR ? PFD + 1 : 1];
        float pmn[TR == 2 ? PFD + 1 : 1], prs[TR == 2 ? PFD + 1 : 1];
        if constexpr (TR == 1) {
            rX = make_rsrc(X + (size_t)row0 * 512, (uint32_t)rv * 1024u);
            rU = make_rsrc(U + (size_t)row0 * 512, (uint32_t)rv * 1024u);
            rM = make_rsrc(MK + (size_t)row0 * 512, (uint32_t)rv * 512u);
            rMR = make_rsrc(MR + (size_t)row0 * 2, (uint32_t)rv * 8u);
            rDX = make_rsrc(DX + (size_t)row0 * 512, (uint32_t)rv * 1024u);
            rDU = make_rsrc(DU + (size_t)row0 * 2048, (uint32_t)rv * 4096u);
        } else if constexpr (TR == 2) {
            rX = make_rsrc(ca.X + (size_t)row0 * 512, (uint32_t)rv * 1024u);
            rU = make_rsrc(ca.U + (size_t)row0 * 512, (uint32_t)rv * 1024u);
            rM = make_rsrc(ca.MK + (size_t)row0 * 512, (uint32_t)rv * 512u);
            rMn = make_rsrc(ca.MEAN + row0, (uint32_t)rv * 4u);
            rRs = make_rsrc(ca.RSTD + row0, (uint32_t)rv * 4u);
            rDX = make_rsrc(ca.DX + (size_t)row0 * 512, (uint32_t)rv * 1024u);
            if constexpr (E5) {  // l11e5: the e5m2 u block of dz8 (row stride 2048 B, 512 B written per row; the
                                 // launch passes DZ8 in ca.DU as well, so the per-block base needs no extra SGPRs)
                rDU = make_rsrc(reinterpret_cast<const uint8_t*>(ca.DU) + (size_t)row0 * 2048, (uint32_t)rv * 2048u);
            } else {
                rDU = make_rsrc(ca.DU + (size_t)row0 * 2048, (uint32_t)rv * 4096u);
            }
        }
        auto ld_row = [&](auto IT2_) {
            constexpr int it2 = decltype(IT2_)::value;
            constexpr int sl = it2 % (PFD + 1);
            const int r = it2 * 8 + w;
            px[sl] = l11_llvm_buffer_load_v4i32(rX, r * 1024 + 16 * lane, 0, 0);
            pu[sl] = l11_llvm_buffer_load_v4i32(rU, r * 1024 + 16 * lane, 0, 0);
            pm[sl] = l11_llvm_buffer_load_v2i32(rM, r * 512 + 8 * lane, 0, 0);
            if constexpr (TR == 2) {
                pmn[sl] = l11_llvm_buffer_load_f32(rMn, r * 4, 0, 0);
                prs[sl] = l11_llvm_buffer_load_f32(rRs, r * 4, 0, 0);
            } else {
                pr[sl] = l11_llvm_buffer_load_v2i32(rMR, r * 8, 0, 0);
            }
        };
        if constexpr (TR) static_for<PFD>([&](auto I_) { ld_row(I_); });
        static_for<NRED * 8>([&](auto IT_) {
            constexpr int it = decltype(IT_)::value % 8;
            if constexpr (decltype(IT_)::value == 8) L11_BARRIER();  // DIAG 1: second pass
            if constexpr (TR && it + PFD < 8) ld_row(std::integral_constant<int, it + PFD>{});  // REVIEW ADDITION
            constexpr int m = it >> 1;
            if ((r16 >> 3) == (it & 1)) {  // lanes holding rows 8*it .. 8*it+7 write their 6 x 16 B
                ds_write16<0 * 512>(wst, pk[0][m]);
                ds_write16<1 * 512>(wst, pk[1][m]);
                ds_write16<2 * 512>(wst, pk[2][m]);
                ds_write16<3 * 512>(wst, pk[3][m]);
                ds_write16<4 * 512>(wst, pk[4][m]);
                ds_write16<5 * 512>(wst, pk[5][m]);
            }
            wait_lgkm<0>();
            L11_BARRIER();
            const i32x4 du = ds_read16<0>(rst);     // u plane: cols [8l, 8l+8)
            const i32x4 dx = ds_read16<1024>(rst);  // x plane: cols 512 + [8l, 8l+8)
            const i32x4 dyy = ds_read16<2048>(rst); // y plane: cols 1024 + [8l, 8l+8)
            wait_lgkm<0>();
            L11_SB();
            const uint32_t vrow = voffC + (uint32_t)(it * 8) * ROWB;  // full row offset in voffset (range-checked)
            if constexpr (STUB == 0) {
                l11_llvm_buffer_store_v4i32(du, rC, (int)vrow, 0, 0);
                l11_llvm_buffer_store_v4i32(dx, rC, (int)(vrow + 1024u), 0, 0);
                l11_llvm_buffer_store_v4i32(dyy, rC, (int)(vrow + 2048u), 0, 0);
            } else if constexpr (TR == 2) {
                // ---- L11-1b ADDITION: the real consumer, v2 op for op (FMA contraction and tl.sum tree as in its ISA):
                //   du_m/dx_m/dy_m = keep ? RN(s*plane) : 0;  xhat = RN(RN(x-mean)*rstd);  ln = fma(xhat,w,b)
                //   sig = IEEE 1/(1+exp(-u));  silu = RN(u*sig);  dsilu = fma(silu, RN(1-sig), sig)
                //   du  = fma(RN(dy_m*ln), dsilu, RN(du_m*dsilu))                       -> bf16
                //   pdb = fma(dy_m, silu, pdb);  dyn = RN(dy_m*silu);  wdy = RN(dyn*w)
                //   c2 = tl.sum(wdy)*invD, c1 = tl.sum(RN(xhat*wdy))*invD (in-lane sequential, then the DPP tree)
                //   dx  = fma(rstd, fma(dyn, w, -fma(xhat, c1, c2)), dx_m)                 -> bf16
                //   pdw = fma(xhat, dyn, pdw)
                constexpr int sl = it % (PFD + 1);
                const float s = ca.dscale;
                const float mean = pmn[sl], rstd = prs[sl];
                const uint32_t mk0 = (uint32_t)pm[sl].x, mk1 = (uint32_t)pm[sl].y;
                float xh[8], dyn[8], dxm[8], dur[8], sv[8], dd[8];
                bool bad = false;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    dd[e] = sig_d(bfe8(pu[sl], e));
                    bad |= !(dd[e] <= 0x1p126f);
                }
                if (__builtin_expect(__builtin_amdgcn_ballot_w64(bad) != 0, 0)) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) sv[e] = 1.0f / dd[e];  // IEEE division (correctly rounded)
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) sv[e] = sig_fast(dd[e]);
                }
                float c1l = 0.f, c2l = 0.f;
#if L11_TLSUM_TREE
                f32x2 tc1[2], tc2[2];  // ROCm 10.2 (T14b): the fused column-pair partials of each row sum
                float wdyk[8], prk[8];
#endif
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const uint32_t mb = (e < 4 ? (mk0 >> (8 * e)) : (mk1 >> (8 * (e - 4))));
                    const float pdu = bfe8(du, e), pdx = bfe8(dx, e), pdy = bfe8(dyy, e);
                    const float xv = bfe8(px[sl], e), uv = bfe8(pu[sl], e);
                    const float dum = (mb & 4u) ? s * pdu : 0.f;
                    dxm[e] = (mb & 2u) ? s * pdx : 0.f;
                    const float dym = (mb & 1u) ? s * pdy : 0.f;
                    const float xhat = (xv - mean) * rstd;
                    const float ln = __builtin_fmaf(xhat, bfe8(wq, e), bfe8(bq, e));
                    const float sig = sv[e];
                    const float silu = uv * sig;
                    const float oms = 1.0f - sig;
                    const float dsilu = __builtin_fmaf(silu, oms, sig);
                    dur[e] = __builtin_fmaf(dym * ln, dsilu, dum * dsilu);
                    pdb[e] = __builtin_fmaf(dym, silu, pdb[e]);
                    const float dn = dym * silu;
                    const float wdy = dn * bfe8(wq, e);
                    const float pr1 = xhat * wdy;
#if L11_TLSUM_TREE  // ROCm 10.2 (T14b): v2 / v3's in-lane tl.sum as Triton 3.9 compiles it (note at L11_TLSUM_TREE)
                    wdyk[e] = wdy;
                    prk[e] = pr1;
                    if (e == 3 || e == 7) {  // column pairs (e-3, e-2) fused, (e-1, e) rounded: two v_pk_fma_f32
                        const int f = e - 3;
                        tc1[e >> 2] = __builtin_elementwise_fma(f32x2{xh[f], xh[f + 1]}, f32x2{wdyk[f], wdyk[f + 1]},
                                                                f32x2{prk[f + 2], prk[f + 3]});
                        tc2[e >> 2] = __builtin_elementwise_fma(f32x2{dyn[f], dyn[f + 1]},
                                                                f32x2{bfe8(wq, f), bfe8(wq, f + 1)},
                                                                f32x2{wdyk[f + 2], wdyk[f + 3]});
                    }
#else
                    if (e == 0) {
                        c2l = wdy;
                        c1l = pr1;
                    } else {
                        c2l = c2l + wdy;
                        c1l = c1l + pr1;
                    }
#endif
                    xh[e] = xhat;
                    dyn[e] = dn;
                }
#if L11_TLSUM_TREE
                {  // ROCm 10.2 (T14b): (cols 0,1 pair + cols 4,5 pair), then even + odd column (v_pk_add, v_add)
                    const f32x2 s1 = tc1[0] + tc1[1], s2 = tc2[0] + tc2[1];
                    c1l = s1.x + s1.y;
                    c2l = s2.x + s2.y;
                }
#endif
                // du store first (as v2), then the row reductions
                const int r = it * 8 + w;
                i32x4 ou;
                ou.x = (int)pk_bf16(f32x2{dur[0], dur[1]});
                ou.y = (int)pk_bf16(f32x2{dur[2], dur[3]});
                ou.z = (int)pk_bf16(f32x2{dur[4], dur[5]});
                ou.w = (int)pk_bf16(f32x2{dur[6], dur[7]});
                if constexpr (E5) {
                    // l11e5: B1a v3 EMIT ops on duf = fp32(RNE_bf16(du)) (no bf16 du store), one packed bf16 pair at a
                    // time (only ou, as in the bf16 mode, and one pair's temporaries live):
                    //   colsum chain += duf (v_pk_add_f32 = two IEEE fp32 adds);
                    //   running max |du| on the fp32 du (v_max3_f32; RNE_bf16 of the wave max at the end = the max of
                    //   |RNE_bf16(du)|: RNE is monotone and odd; NaN ignored, as maxnum);
                    //   e5m2 = v_cvt_scalef32_pk_bf8_f32(v_med3_f32(RN(duf * inv), -57344, 57344), scale 1.0)
                    //   (duf * inv as v_pk_mul_f32 = two IEEE fp32 multiplies)
                    const f32x2 inv2 = {inv5, inv5};
                    float sc[8];
                    static_for<4>([&](auto K_) {
                        constexpr int k = decltype(K_)::value;
                        am5 = __builtin_fmaxf(__builtin_fmaxf(am5, __builtin_fabsf(dur[2 * k])), __builtin_fabsf(dur[2 * k + 1]));
                        const uint32_t wd = (uint32_t)ou[k];
                        const f32x2 d2 = {bf_lo(wd), bf_hi(wd)};
                        pcs[k] = pcs[k] + d2;
                        const f32x2 p2 = d2 * inv2;
                        sc[2 * k] = __builtin_amdgcn_fmed3f(p2.x, -57344.0f, 57344.0f);
                        sc[2 * k + 1] = __builtin_amdgcn_fmed3f(p2.y, -57344.0f, 57344.0f);
                    });
                    rv_i32x2 q8 = {l11::cvt_pk_e5m2(sc[0], sc[1], sc[2], sc[3]),
                                   l11::cvt_pk_e5m2(sc[4], sc[5], sc[6], sc[7])};
                    asm volatile("" : "+v"(q8));
                    l11_llvm_buffer_store_v2i32(q8, rDU, r * 2048 + 8 * lane, 0, SCP ? 3 : 0);
                    // pin the colsum / amax chains per row like pdw/pdb below
                    asm volatile("" : "+v"(pcs[0]), "+v"(pcs[1]), "+v"(pcs[2]), "+v"(pcs[3]), "+v"(am5));
                } else {
                    l11_llvm_buffer_store_v4i32(ou, rDU, r * 4096 + 16 * lane, 0, SCP ? 3 : 0);
                }
                const float c2 = tl_sum_wave(c2l) * ca.invD;
                const float c1 = tl_sum_wave(c1l) * ca.invD;
                float dxo[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float t = __builtin_fmaf(xh[e], c1, c2);
                    const float v = __builtin_fmaf(dyn[e], bfe8(wq, e), -t);
                    dxo[e] = __builtin_fmaf(rstd, v, dxm[e]);
                    pdw[e] = __builtin_fmaf(xh[e], dyn[e], pdw[e]);
                }
                // pin the partial chains here (else LLVM sinks the 8 rows' FMAs to the end of the block and keeps every
                // row's xhat/dyn live: first t5 build, 103-113 VGPR spilled)
                asm volatile("" : "+v"(pdw[0]), "+v"(pdw[1]), "+v"(pdw[2]), "+v"(pdw[3]), "+v"(pdw[4]), "+v"(pdw[5]),
                             "+v"(pdw[6]), "+v"(pdw[7]));
                asm volatile("" : "+v"(pdb[0]), "+v"(pdb[1]), "+v"(pdb[2]), "+v"(pdb[3]), "+v"(pdb[4]), "+v"(pdb[5]),
                             "+v"(pdb[6]), "+v"(pdb[7]));
                i32x4 ox;
                ox.x = (int)pk_bf16(f32x2{dxo[0], dxo[1]});
                ox.y = (int)pk_bf16(f32x2{dxo[2], dxo[3]});
                ox.z = (int)pk_bf16(f32x2{dxo[4], dxo[5]});
                ox.w = (int)pk_bf16(f32x2{dxo[6], dxo[7]});
                l11_llvm_buffer_store_v4i32(ox, rDX, r * 1024 + 16 * lane, 0, SCP ? 3 : 0);
            } else {
                const uint32_t y0 = (uint32_t)dyy.x, y1 = (uint32_t)dyy.y, y2 = (uint32_t)dyy.z, y3 = (uint32_t)dyy.w;
                float v = ((bf_lo(y0) + bf_hi(y0)) + (bf_lo(y1) + bf_hi(y1))) +
                          ((bf_lo(y2) + bf_hi(y2)) + (bf_lo(y3) + bf_hi(y3)));
                v = wave_sum(v);
                if constexpr (TR) {  // REVIEW ADDITION: byte-touching consumer: dx = x-plane ^ x ^ mean/rstd, du = u-plane ^ u ^ mask
                    constexpr int sl = it % (PFD + 1);
                    i32x4 ox = dx ^ px[sl];
                    ox.x ^= pr[sl].x ^ pr[sl].y;
                    i32x4 ou = du ^ pu[sl];
                    ou.x ^= pm[sl].x;
                    ou.y ^= pm[sl].y;
                    ou.z ^= pm[sl].x;
                    ou.w ^= pm[sl].y;
                    const int r = it * 8 + w;
                    l11_llvm_buffer_store_v4i32(ox, rDX, r * 1024 + 16 * lane, 0, 0);
                    l11_llvm_buffer_store_v4i32(ou, rDU, r * 4096 + 16 * lane, 0, 0);
                } else {
                    chk ^= (uint32_t)(du.x ^ du.y ^ du.z ^ du.w ^ dx.x ^ dx.y ^ dx.z ^ dx.w);
                }
                l11_llvm_buffer_store_f32(v, rR, lane == 0 ? (it * 8 + w) * 4 : 0x40000000, 0, 0);
            }
            L11_SB();
            if constexpr (it < 7) L11_BARRIER();  // slab reused; after it == 7 the next block's step-0 barrier serves
        });
        first = false;
        if constexpr (TR == 2) {  // L11-1b: periodic dW/dB partial flush (chain length fl*8 rows per partial)
            ++lb;
            if (lb % ca.fl == 0) {
                flush(slot);
                ++slot;
            }
        }
    }
    if constexpr (TR == 2) {  // remaining (possibly empty) slots: every partial row of the [P,512] buffer is written
        for (; slot < ca.nf; ++slot) flush(slot);
    }
    if constexpr (TR == 2 && E5) {
        // l11e5: per-wave amax partial = RNE_bf16(max over the wave's lanes of am5) as fp32 (am5 >= 0, never NaN; DPP
        // row_shr 8,4,2,1 with bound_ctrl (invalid lanes read 0) leaves each 16-lane row's max in its lane 15). Equal to
        // v3's max of (bf16 bits & 0x7FFF, NaN -> 0) over the same elements: RNE is monotone and odd.
        float a = am5;
        a = __builtin_fmaxf(a, __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, a), 0x118, 0xF, 0xF, true)));
        a = __builtin_fmaxf(a, __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, a), 0x114, 0xF, 0xF, true)));
        a = __builtin_fmaxf(a, __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, a), 0x112, 0xF, 0xF, true)));
        a = __builtin_fmaxf(a, __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, a), 0x111, 0xF, 0xF, true)));
        const int ia = __builtin_bit_cast(int, a);
        const float m01 = __builtin_fmaxf(__builtin_bit_cast(float, __builtin_amdgcn_readlane(ia, 15)),
                                          __builtin_bit_cast(float, __builtin_amdgcn_readlane(ia, 31)));
        const float m23 = __builtin_fmaxf(__builtin_bit_cast(float, __builtin_amdgcn_readlane(ia, 47)),
                                          __builtin_bit_cast(float, __builtin_amdgcn_readlane(ia, 63)));
        const float wm = __builtin_fmaxf(m01, m23);
        if (lane == 0) ca.AMAXP[blockIdx.x * 8 + w] = bf_lo(pk_bf16(f32x2{wm, wm}));
        // rows [M, rows_out) of the u block: zero (as v3's tail loop and the shipped cast write them); one wave per row
        for (int pr = M + (int)blockIdx.x * 8 + w; pr < ca.rows_out; pr += G * 8)
            *reinterpret_cast<rv_i32x2*>(ca.DZ8 + (size_t)pr * 2048u + 8u * (size_t)lane) = rv_i32x2{0, 0};
    }
    wait_vm<0>();  // drain the trailing ring prefetches before the WG's LDS is released
    if constexpr (STUB == 1) {
        if (sav == 12345.0f) {  // never true for real scales: keeps the u/x-plane consumption alive (diagnostic idiom, r14)
            const i32x4 rC = make_rsrc(C, 64u * 4u);
            l11_llvm_buffer_store_f32(__builtin_bit_cast(float, chk), rC, lane * 4, 0, 0);
        }
    }
}


// ======================================================================================================================
// Production host ABI (r16 iter31 L11-2 integration). All entries are extern "C" and take raw device pointers.
// ======================================================================================================================
#ifndef L11_SRC_SHA256
#define L11_SRC_SHA256 "unknown"
#endif
// ABI 2 (r16 iter32 l11e5): + l11_outproj_lnmd_bwd_e5_launch (the E5 = 1 B1a e5m2 du emission mode).
#define L11_ABI_VERSION 2
// M range validated by the unit test (ops/tests/l11_outproj_lnmd_test.py); the Python gate (D8) enforces the same bound.
static constexpr long long L11_M_MAX = 4194304;

// CU count of the FIRST device this process launches on (cached in a static). Production runs one GPU per process; a
// process that drives GPUs with different CU counts would need a per-device cache (it would only change the grid size
// and P, never correctness, as long as l11_plan and the launch agree -- they share this function).
static int l11_ncu() {
    static int n = 0;
    if (n == 0) {
        int dev = 0, c = 0;
        if (hipGetDevice(&dev) != hipSuccess) return 256;
        if (hipDeviceGetAttribute(&c, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || c <= 0) c = 256;
        n = c;
    }
    return n;
}

// Geometry: persistent grid G = min(nblk, #CU) WGs of 8 waves; per = ceil(nblk / G) row blocks per WG at most; every
// wave flushes its dW/dB fp32 partial chains every fl row blocks (fl = ceil(per / 8), so a chain covers <= 8*fl rows,
// v2-like: <= 152 rows at M = 2.36M vs v2's ~145) into nf = ceil(per / fl) <= 8 slots: P = G*8*nf <= 16384 rows.
static int l11_plan_impl(int M, long long* G, long long* fl, long long* nf) {
    if (M <= 0 || (long long)M > L11_M_MAX) return (int)hipErrorInvalidValue;
    const long long nblk = ((long long)M + 63) / 64;
    const long long ncu = (long long)l11_ncu();
    *G = nblk < ncu ? nblk : ncu;
    const long long per = (nblk + *G - 1) / *G;
    *fl = (per + 7) / 8;
    *nf = (per + *fl - 1) / *fl;
    return 0;
}

extern "C" int l11_abi_version(void) { return L11_ABI_VERSION; }

extern "C" const char* l11_impl(void) {
    return "l11_outproj_lnmd_bwd abi=2 variant=l11c_f1(TR2,PFD1,SCP0,PW1,NBUF2,persistent) modes=bf16du,e5du(B1a v3 EMIT) "
           "src_sha256=" L11_SRC_SHA256;
}

// out4 = {G, fl, nf, P}. Returns 0, or hipErrorInvalidValue for M outside [1, L11_M_MAX].
extern "C" int l11_plan(int M, long long* out4) {
    if (out4 == nullptr) return (int)hipErrorInvalidValue;
    long long G, fl, nf;
    const int rc = l11_plan_impl(M, &G, &fl, &nf);
    if (rc != 0) return rc;
    out4[0] = G;
    out4[1] = fl;
    out4[2] = nf;
    out4[3] = G * 8 * nf;
    return 0;
}

// Drain (return and clear) this thread's sticky HIP last error. The Python wrapper calls it before its own torch output
// allocations, so a stale error from an earlier unrelated call is reported once instead of failing them.
extern "C" int l11_drain_last_error(void) { return (int)hipGetLastError(); }

static inline bool l11_a16(const void* p) { return ((uintptr_t)p & 15u) == 0; }
static inline bool l11_a4(const void* p) { return p != nullptr && ((uintptr_t)p & 3u) == 0; }

// The fused backward. Contract (hard-coded strides; the Python gate D8/D9 enforces them):
//   G8 e5m2 [M,512] row-major; W8 e4m3 [1536,512] row-major (the storage of the W8.t() operand _scaled_mm consumed);
//   sa, sb fp32 [1] device decode scales; X = attn, U = u bf16 [M,512] stride 512; MK int8 [M,512] stride 512 (keep bits
//   1 = y plane, 2 = x plane, 4 = u plane); MEAN, RSTD fp32 [M]; WN, BN bf16 [512]; DX bf16 [M,512] stride 512;
//   DU bf16 rows of an [M,2048] buffer (cols 0..511 written, 512..2047 untouched); DWP, DBP fp32 [P,512] with P from
//   l11_plan(M) (every row written, no zero fill needed); dscale = fp32(1/fp32(1 - p)) as v2; invD = 2^-9.
// Returns 0 = launched; > 0 = hipError_t of a rejected argument or a failed launch (outputs not written);
// < 0 = launched, and -rc is a stale error an earlier call had left on this thread (drained here, not ours).
extern "C" int l11_outproj_lnmd_bwd_launch(const void* G8, const void* W8, const float* sa, const float* sb, int M,
                                           const void* X, const void* U, const void* MK, const float* MEAN,
                                           const float* RSTD, const void* WN, const void* BN, void* DX, void* DU,
                                           void* DWP, void* DBP, long long P, float dscale, float invD, void* stream) {
    long long G, fl, nf;
    if (l11_plan_impl(M, &G, &fl, &nf) != 0) return (int)hipErrorInvalidValue;
    if (P != G * 8 * nf) return (int)hipErrorInvalidValue;
    if (!(l11_a16(G8) && l11_a16(W8) && l11_a16(X) && l11_a16(U) && l11_a16(MK) && l11_a16(WN) && l11_a16(BN) &&
          l11_a16(DX) && l11_a16(DU) && l11_a16(DWP) && l11_a16(DBP)))
        return (int)hipErrorInvalidValue;
    if (G8 == nullptr || W8 == nullptr || X == nullptr || U == nullptr || MK == nullptr || WN == nullptr ||
        BN == nullptr || DX == nullptr || DU == nullptr || DWP == nullptr || DBP == nullptr)
        return (int)hipErrorInvalidValue;
    if (!(l11_a4(sa) && l11_a4(sb) && l11_a4(MEAN) && l11_a4(RSTD))) return (int)hipErrorInvalidValue;
    const long long nblk = ((long long)M + 63) / 64;
    L11CArgs ca{};  // l11e5: value-initialized (the E5 fields stay null/0 in this mode)
    ca.X = (const uint16_t*)X;
    ca.U = (const uint16_t*)U;
    ca.MK = (const uint8_t*)MK;
    ca.MEAN = MEAN;
    ca.RSTD = RSTD;
    ca.WN = (const uint16_t*)WN;
    ca.BN = (const uint16_t*)BN;
    ca.DX = (uint16_t*)DX;
    ca.DU = (uint16_t*)DU;
    ca.DWP = (float*)DWP;
    ca.DBP = (float*)DBP;
    ca.dscale = dscale;
    ca.invD = invD;
    ca.fl = (int)fl;
    ca.nf = (int)nf;
    // The HIP last error is sticky per thread: drain an error an earlier, unrelated call left behind, check our launch
    // on its own, and report a drained error as a negative code (r14 wgrad_tm_ck_w3.cu contract).
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((l11_outproj_lnmd_bwd<1, 2, 0, 0, 1, 2, 1, 0>), dim3((unsigned)G), dim3(512), 0,
                       (hipStream_t)stream, (const uint8_t*)G8, (const uint8_t*)W8, (uint16_t*)nullptr, (float*)nullptr,
                       sa, sb, M, (int)nblk, (const uint16_t*)nullptr, (const uint16_t*)nullptr,
                       (const uint8_t*)nullptr, (const float*)nullptr, (uint16_t*)nullptr, (uint16_t*)nullptr, ca);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

// l11e5 (ABI 2): the E5 = 1 mode = the B1a producer contract (ship-12 DLRM_HSTU_FP8_BWD_DU_E5M2; r16
// _ln_mul_dropout_bwd_dx_du_rng_v3 EMIT). Same contract as l11_outproj_lnmd_bwd_launch except that NO bf16 du is written:
//   DZ8 e5m2 [rows_out, 2048] row-major (16-B aligned): columns 0..511 of rows < M get the e5m2 du payload, columns
//   0..511 of rows [M, rows_out) are zeroed, columns 512..2047 are untouched; M <= rows_out < M + 4096;
//   INVS fp32 [1] device inv_scale (4-B aligned) the payload is cast with (read once by the kernel);
//   AMAXP fp32 [G*8] (G = l11_plan(M)[0]): per-wave RNE_bf16(max |du|), NaN ignored, every entry written (>= 0);
//   DUCS fp32 [P,512] (16-B aligned): u column-sum partials of fp32(RNE_bf16(du)), every row written.
// Record values as B1a: u_amax = AMAXP.amax(), u_colsum = DUCS.sum(0). Same return codes.
extern "C" int l11_outproj_lnmd_bwd_e5_launch(const void* G8, const void* W8, const float* sa, const float* sb, int M,
                                              const void* X, const void* U, const void* MK, const float* MEAN,
                                              const float* RSTD, const void* WN, const void* BN, void* DX, void* DZ8,
                                              int rows_out, const float* INVS, void* AMAXP, void* DWP, void* DBP,
                                              void* DUCS, long long P, float dscale, float invD, void* stream) {
    long long G, fl, nf;
    if (l11_plan_impl(M, &G, &fl, &nf) != 0) return (int)hipErrorInvalidValue;
    if (P != G * 8 * nf) return (int)hipErrorInvalidValue;
    if (rows_out < M || (long long)rows_out >= (long long)M + 4096) return (int)hipErrorInvalidValue;
    if (!(l11_a16(G8) && l11_a16(W8) && l11_a16(X) && l11_a16(U) && l11_a16(MK) && l11_a16(WN) && l11_a16(BN) &&
          l11_a16(DX) && l11_a16(DZ8) && l11_a16(DWP) && l11_a16(DBP) && l11_a16(DUCS)))
        return (int)hipErrorInvalidValue;
    if (G8 == nullptr || W8 == nullptr || X == nullptr || U == nullptr || MK == nullptr || WN == nullptr ||
        BN == nullptr || DX == nullptr || DZ8 == nullptr || DWP == nullptr || DBP == nullptr || DUCS == nullptr)
        return (int)hipErrorInvalidValue;
    if (!(l11_a4(sa) && l11_a4(sb) && l11_a4(MEAN) && l11_a4(RSTD) && l11_a4(INVS) && l11_a4(AMAXP)))
        return (int)hipErrorInvalidValue;
    const long long nblk = ((long long)M + 63) / 64;
    L11CArgs ca{};
    ca.X = (const uint16_t*)X;
    ca.U = (const uint16_t*)U;
    ca.MK = (const uint8_t*)MK;
    ca.MEAN = MEAN;
    ca.RSTD = RSTD;
    ca.WN = (const uint16_t*)WN;
    ca.BN = (const uint16_t*)BN;
    ca.DX = (uint16_t*)DX;
    ca.DU = (uint16_t*)DZ8;  // the E5 kernel addresses the u block through ca.DU (byte rows of 2048), DZ8 for the pad rows
    ca.DWP = (float*)DWP;
    ca.DBP = (float*)DBP;
    ca.dscale = dscale;
    ca.invD = invD;
    ca.fl = (int)fl;
    ca.nf = (int)nf;
    ca.DZ8 = (uint8_t*)DZ8;
    ca.INVS = INVS;
    ca.AMAXP = (float*)AMAXP;
    ca.DUCS = (float*)DUCS;
    ca.rows_out = rows_out;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((l11_outproj_lnmd_bwd<1, 2, 0, 0, 1, 2, 1, 0, 1>), dim3((unsigned)G), dim3(512), 0,
                       (hipStream_t)stream, (const uint8_t*)G8, (const uint8_t*)W8, (uint16_t*)nullptr, (float*)nullptr,
                       sa, sb, M, (int)nblk, (const uint16_t*)nullptr, (const uint16_t*)nullptr,
                       (const uint8_t*)nullptr, (const float*)nullptr, (uint16_t*)nullptr, (uint16_t*)nullptr, ca);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

// TESTS ONLY: the same mainloop + epilogue with the redistributed dy rows stored to C [M,1536] bf16 (STUB 0, per-wave
// weight ring PW 1, persistent; the builder's l11w_s0_b2). Never called by training. Same return contract.
extern "C" int l11_dy_debug(const void* G8, const void* W8, const float* sa, const float* sb, int M, void* C,
                            void* stream) {
    if (M <= 0 || (long long)M > L11_M_MAX || !l11_a16(G8) || !l11_a16(W8) || !l11_a16(C) || C == nullptr ||
        !l11_a4(sa) || !l11_a4(sb))
        return (int)hipErrorInvalidValue;
    const long long nblk = ((long long)M + 63) / 64;
    const long long ncu = (long long)l11_ncu();
    const long long G = nblk < ncu ? nblk : ncu;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((l11_outproj_lnmd_bwd<0, 2, 0, 0, 1>), dim3((unsigned)G), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G8, (const uint8_t*)W8, (uint16_t*)C, (float*)C, sa, sb, M, (int)nblk,
                       (const uint16_t*)nullptr, (const uint16_t*)nullptr, (const uint8_t*)nullptr, (const float*)nullptr,
                       (uint16_t*)nullptr, (uint16_t*)nullptr, L11CArgs{});
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

// BSR (design O-D2): persistent MXFP6 K=512 forward GEMM with REGISTER-RESIDENT weights, bitwise equal to the shipped
// kernels below (verified on gfx950; the loader serves a site's BSR build only after a bitwise prebuild self-check).
// Forked from an unshipped persistent variant with LDS-resident weights, itself a fork of the shipped kernel.
//
// Bitwise reference: the shipped k512_mx6_gemm.cu (VQK N=1536 / VK N=1024, RING6 build -DK512MX6_RING=6) and
// k512_mx6_u.cu (U N=512, bf16 + bias epilogue, MODE left at 0), both in this directory.
//
// VERBATIM from the shipped source: make_rsrc, the barrier / fence macros, ds_read16 / ds_read8 / frag6, mf / mma8, g2l,
// the per-lane LDS offset formulas (lpH / lpT and the wave parts of lpBH / lpBT, used here as the weights' in-item byte
// offsets), the weight-scale (sW) and bias formulas, the activation-scale bytes (dword H of the shipped sA[KS]) and the
// epilogue arithmetic (k512_fp8_gemm NOCLAMP=1 e4m3 form under MODE.FP16_OVFL; U: the k512_mx6_u.cu bf16 form).
//
// Geometry. grid = 256 persistent WGs (one per MI355X CU), 512 threads = 8 waves, __launch_bounds__(512, 1)
// (occupancy 2 waves/SIMD: 256-VGPR budget). WG tile = 64 rows x 512 columns of the CU's N-slice s (fixed per WG).
// Wave w owns the tile's 64 rows x columns [64w, 64w + 64) of the slice: it is the SHIPPED wave wl = w & 3 of the
// 256-column tile tile_n = 2s + (w >> 2), restricted to row half H of the producer's 128-row item (64-row tile m64 ->
// producer tile T = m64 >> 1, H = m64 & 1). Every output element gets the shipped MFMA chain: same fragments (same bytes),
// same four in-order K=128 scaled MFMAs (C = 0 first), same scale bytes, same epilogue.
//
//   weights  the wave's 16 fragments (K step 0..3 x {B0 frag 0, B0 frag 1, B1 frag 0, B1 frag 1}) are loaded ONCE, straight
//            from global w6 at the in-item byte offsets the shipped ds_reads use (the shipped LDS slot is a linear copy of
//            the w6 item): head 16 B (dwordx4) + tail 8 B (dwordx2) per fragment, held as genuine 6-VGPR tuples (i32x4 +
//            i32x2, assembled into the MFMA operand at the call: the cbsz/blgp = 2 MFMA reads 6 VGPRs) = 96 VGPRs.
//            sW (one dword, shipped formula with col0 = tile_n * 256 and w -> wl), the raw bias (2 x dwordx4) and
//            s = sa * sb are formed once per wave. No weight load inside the tile loop.
//   activ.   item (64-row tile, K step) = half H of the producer item: HEAD bytes [4096H, +4096) + TAIL bytes
//            [8192 + 2048H, +2048) = 6 KiB. LDS ring of BSR_S slots (slot = HEAD 4096 + TAIL 2048 B); item i (the WG's
//            running K-step index) lives in slot i % BSR_S. Waves 0..5 each issue ONE linear 1 KiB LDS-DMA per item
//            (wave c < 4: head chunk c; c = 4, 5: tail chunk c - 4). Every wave reads the item's 4 A fragments (subtile
//            i: head i*1024, tail 4096 + i*512 inside the slot, the shipped per-lane swizzled lpH / lpT parts) into ONE
//            register bank a[4] (24 VGPRs).
//   K step   [issuing waves: s_waitcnt vmcnt(own DMA of item k landed)] s_barrier  (proves item k complete AND every wave
//            finished reading item k-1: each did lgkmcnt(0) before arriving) | refill: DMA of item k-1+S into slot
//            (k-1) % S | 4 A fragments (+ the scale dword, SCL=1) | lgkmcnt(0) | 16 MFMAs acc[h][n][i] (+)=
//            mf<KS, i>(W[KS][h][n], a[i]) with the shipped mma8 (h = 0: B0 weights, h = 1: B1). ONE barrier per K step.
//   epilogue the shipped arithmetic for the wave's 4 sixteen-row groups ms of its half: v = {acc[0][0][ms],
//            acc[0][1][ms], acc[1][0][ms], acc[1][1][ms]}, Ct = C + (64 m64 + 16 ms) N + tile_n * 256, shipped lane_off
//            (w -> wl) and num_records row-validity clamp with rows_valid = min(64, M - 64 m64).
//   schedule WG b -> XCD b % 8, local l = b / 8; NS = N / 512 slices (VQK 3, VK 2, U 1); cohort c = l / NS, slice
//            s = l % NS, NCOH = 32 / NS cohorts per XCD; l >= NS * NCOH exits before any barrier (VQK: 2 per XCD). XCD x
//            owns the contiguous range [x*tb + min(x, tr), +tb + (x < tr)) of the T64 = ceil(M/64) 64-row tiles
//            (tb = T64 / 8, tr = T64 % 8); cohort c walks it at c, c + NCOH, ...; a WG with no tile exits at once.
//            Look-ahead items past the cohort's last tile are CLAMPED to its last tile (harmless re-reads, in range: the
//            producer pads to 128 rows, so half H = 1 of the last producer tile exists even past M; drained by the
//            final vmcnt(0)), which keeps every wave's VMEM stream, and hence every vmcnt operand, uniform.
//
// Wave roles (each role's loop is straight-line code, so the ring waits and the compiler's own waits are exact):
//   ROLE_A  waves 0..5: one 1 KiB activation DMA per K step.
//   ROLE_S  wave 6 when BSR_SCL = 1: one 4 B/lane DMA per K step carrying the item's activation-scale dwords (xs bytes
//           8 lane + 4H of block (T, KS) -> scale slot + 4 lane); every wave reads its dword with ds_read_b32.
//   ROLE_N  the remaining waves: no DMA (no ring wait needed); same barriers, MFMAs and stores.
//
// Switches:
//   HK_N     1536 (VQK) / 1024 (VK): e4m3 + bias C ABI k512_mx6_e4m3_bias_N<N>; 512 with BSR_U = 1: U bf16 + bias ABI
//            k512_mx6_u_bf16_bias (k512_mx6_u.cu epilogue).
//   BSR_S    activation ring slots: 16 (4 tiles of look-ahead, 96 KiB) or 8 (48 KiB).
//   BSR_VMS  0: ring waits under the CDNA4 in-order VM_CNT model ("Memory reads and writes return in the order they
//            were issued, including mixing reads and writes": count every younger VMEM op: DMAs, scale loads and the
//            previous tiles' stores; tight); 1: strict (only LDS-DMAs assumed ordered among themselves: vmcnt(S - 2)).
//   BSR_SCL  0: activation-scale dword per K step by a per-wave global load ONE K step ahead (source default);
//            1: the scale dwords travel through the LDS ring with the item (ROLE_S), no VMEM load in the loop.
//            Same scale bytes, same MFMA scale operands in both.
//
// Build: hipcc --offload-arch=gfx950 -std=c++20 -O3 -ffp-contract=off -mllvm -amdgpu-sched-strategy=iterative-ilp
//        -DHK_N={1536|1024} [-DBSR_S=8|16] [-DBSR_VMS=0|1] [-DBSR_SCL=0|1]   (U: -DHK_N=512 -DBSR_U=1)
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

#if !defined(HK_N)
#error "define HK_N (1536 VQK, 1024 VK, or 512 with BSR_U=1)"
#endif
#ifndef BSR_U
#define BSR_U 0
#endif
#if BSR_U && HK_N != 512
#error "BSR_U=1 (U bf16 + bias epilogue) requires HK_N=512"
#endif
#if !BSR_U && HK_N != 1536 && HK_N != 1024
#error "e4m3 builds: HK_N = 1536 (VQK) or 1024 (VK)"
#endif
#ifndef BSR_S
#define BSR_S 16
#endif
#if BSR_S != 8 && BSR_S != 16
#error "BSR_S: activation ring slots, 8 or 16"
#endif
#ifndef BSR_VMS
#define BSR_VMS 0
#endif
#if BSR_VMS != 0 && BSR_VMS != 1
#error "BSR_VMS: 0 = in-order vmcnt model (CDNA4 ISA 4.4, tight), 1 = strict (DMA-only ordering) waits"
#endif
#ifndef BSR_SCL
#define BSR_SCL 0
#endif
#if BSR_SCL != 0 && BSR_SCL != 1
#error "BSR_SCL: 0 = per-wave global scale loads one K step ahead, 1 = scales through the LDS ring"
#endif

#define K512MX6_STALE_ERROR_BASE 100000

// ===================================================================================================================
// shipped helpers (k512_mx6_gemm.cu, verbatim)
// ===================================================================================================================
namespace k512mx6 {
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
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
constexpr int BM = 128;      // producer row tile (x6 / xs items)
constexpr int BN = 256;      // shipped column tile (one w6 item pair)
constexpr int KSTEPS = 4;
constexpr int SLOT = 12288;  // producer item image: HEAD 8192 + TAIL 4096
constexpr int TAILOFF = 8192;

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
// (ignored by the cbsz/blgp=2 MFMA, which reads a 6-VGPR operand).
template <int SH, int ST>
__device__ __forceinline__ i32x8 frag6(uint32_t lpH, uint32_t lpT) {
    const i32x4 lo = ds_read16<SH>(lpH);
    const i32x2 hi = ds_read8<ST>(lpT);
    const i32x4 hi4 = __builtin_shufflevector(hi, hi, 0, 1, -1, -1);
    return __builtin_shufflevector(lo, hi4, 0, 1, 2, 3, 4, 5, -1, -1);
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

// k512_mx6_u.cu's pk_bf16 (verbatim): RNE fp32 -> bf16 pair (gfx950: v_cvt_pk_bf16_f32)
__device__ __forceinline__ uint32_t pk_bf16(f32x2 v) {
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2));
}

// ===================================================================================================================
// BSR additions
// ===================================================================================================================
constexpr int BSR_GRID = 256;          // persistent WGs (MI355X: 256 CUs, 8 XCDs x 32)
constexpr int BSR_WPX = BSR_GRID / 8;  // WGs per XCD
constexpr int BMR = 64;                // BSR tile rows (one half of a producer item)
constexpr int NSL = 512;               // BSR tile columns = the CU's N-slice
constexpr int HSL = 6144;              // ring slot: half item, HEAD 4096 + TAIL 2048
constexpr int HTAIL = 4096;            // tail offset inside a ring slot
constexpr int NSLOT = BSR_S;
constexpr int NGRP = NSLOT / KSTEPS;   // tiles of items in the ring (slot group g = tile index mod NGRP)
constexpr int SCLB = 256;              // SCL=1: one item's scale dwords (64 lanes x 4 B)
constexpr int SCLR0 = NSLOT * HSL;     // SCL=1: scale slots after the activation slots
constexpr int BSR_LDS = NSLOT * HSL + (BSR_SCL ? NSLOT * SCLB : 0);
#if BSR_U
constexpr int STPT = 8;                // stores per wave per tile (U: 2 x dwordx4 per 16-row group)
#else
constexpr int STPT = 4;                // e4m3: 1 x dwordx4 per 16-row group
#endif
static_assert(NGRP * KSTEPS == NSLOT && (NGRP & (NGRP - 1)) == 0, "ring = whole tiles, power of two");
static_assert(BSR_LDS <= 163840, "gfx950 per-workgroup LDS limit (d3 ldsprobe: 163844 B -> 'exceeds limit (163840)')");
static_assert(HSL % 256 == 0 && SCLR0 % 256 == 0, "64-bank phase of every slot = the shipped slots'");
// largest ds_read immediate inside one slot group (4 slots): A tail of K step 3, subtile 3
static_assert((KSTEPS - 1) * HSL + HTAIL + 3 * 512 < 65536 && (KSTEPS - 1) * SCLB < 65536, "ds_read immediates");

constexpr int ROLE_A = 0, ROLE_S = 1, ROLE_N = 2;

// Ring wait at the start of K step KS (issuing roles; first S-1 items are covered by the prologue's vmcnt(0)).
// Per-wave VMEM order of one K step: SCL 0: [scale load G(k+1)] [DMA item k-1+S]; SCL 1: [DMA]; after the K-step-3 MFMAs:
// STPT stores. The wait before reading item k must retire the wave's own DMA of item k, issued S-1 steps earlier as the
// LAST VMEM op of its step (stores of that step follow it). Younger VMEM ops (in-order model) =
//   (S - 2) steps x (1 + (SCL == 0)) + STPT x (number of tile ends among the S - 1 steps k-S+1 .. k-1).
// Strict model (only LDS-DMAs ordered among themselves): younger DMAs = S - 2.
constexpr int ring_wait(int ks) {
#if BSR_VMS
    return NSLOT - 2;
#else
    int n = (NSLOT - 2) * (BSR_SCL == 0 ? 2 : 1);
    for (int d = 1; d <= NSLOT - 1; ++d)
        if ((((ks - d) % KSTEPS) + KSTEPS) % KSTEPS == KSTEPS - 1) n += STPT;
    return n;
#endif
}
static_assert(ring_wait(0) <= 63 && ring_wait(1) <= 63 && ring_wait(2) <= 63 && ring_wait(3) <= 63, "vmcnt is 6 bit");

template <int N>
__device__ __forceinline__ void wait_vm() {
    static_assert(N >= 0 && N <= 63, "vmcnt is 6 bit");
    asm volatile("s_waitcnt vmcnt(%0)" ::"i"(N) : "memory");
}
template <int IMM>
__device__ __forceinline__ int ds_read4(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    int v;
    asm volatile("ds_read_b32 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
// A fragments of the item in ring slot (slot group base + KS): subtile i = HEAD i*1024, TAIL HTAIL + i*512 (the shipped
// read_A<SL, H> offsets with the producer item's half H moved to the slot origin)
template <int KS>
__device__ __forceinline__ void read_A6(i32x8 (&a)[4], uint32_t lpH, uint32_t lpT) {
    a[0] = frag6<KS * HSL + 0 * 1024, KS * HSL + HTAIL + 0 * 512>(lpH, lpT);
    a[1] = frag6<KS * HSL + 1 * 1024, KS * HSL + HTAIL + 1 * 512>(lpH, lpT);
    a[2] = frag6<KS * HSL + 2 * 1024, KS * HSL + HTAIL + 2 * 512>(lpH, lpT);
    a[3] = frag6<KS * HSL + 3 * 1024, KS * HSL + HTAIL + 3 * 512>(lpH, lpT);
}
// register-resident weight fragment: head dwords 0..3 + tail dwords 4..5 (= frag6's tuple), dwords 6..7 undefined
__device__ __forceinline__ i32x8 wfrag(const i32x4& lo, const i32x2& hi) {
    const i32x4 hi4 = __builtin_shufflevector(hi, hi, 0, 1, -1, -1);
    return __builtin_shufflevector(lo, hi4, 0, 1, 2, 3, 4, 5, -1, -1);
}

}  // namespace k512mx6

template <int N>
__global__ __launch_bounds__(512, 1) void k512_mx6_bsr_kernel(const uint8_t* __restrict__ A,
                                                              const uint8_t* __restrict__ B, uint8_t* __restrict__ C,
                                                              const uint16_t* __restrict__ bias,
                                                              const float* __restrict__ sa,
                                                              const float* __restrict__ sb,
                                                              const uint8_t* __restrict__ ascale,
                                                              const uint8_t* __restrict__ wscale, int M) {
#pragma clang fp contract(off)
    using namespace k512mx6;
    static_assert(N % NSL == 0, "N must be a multiple of 512");
    constexpr int NS = N / NSL;            // N-slices
    constexpr int NCOH = BSR_WPX / NS;     // cohorts per XCD (VQK 10, VK 16, U 32)
    static_assert(NCOH >= 1, "one cohort per XCD at least");
    __shared__ __attribute__((aligned(2048))) uint8_t smem[BSR_LDS];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));  // 0..7
    const int wl = w & 3;                                                    // the shipped kernel's wave index

    // ---- persistent cohort schedule (wave-uniform; whole-WG exits only, before any barrier) ----
    const int bid = (int)blockIdx.x;
    const int xcd = bid & 7, loc = bid >> 3;
    if (loc >= NCOH * NS) return;  // VQK: 2 idle WGs per XCD
    const int coh = loc / NS, slc = loc % NS;
    const int T64 = (M - 1) / BMR + 1;  // 64-row tiles (M >= 1)
    const int tb = T64 >> 3, tr = T64 & 7;
    const int mx0 = xcd * tb + min(xcd, tr);
    const int mxn = tb + (xcd < tr ? 1 : 0);
    const int nc = mxn > coh ? (mxn - coh + NCOH - 1) / NCOH : 0;  // the cohort's tiles
    if (nc == 0) return;
    const int tile_n = 2 * slc + (w >> 2);  // the shipped 256-column tile of this wave's 64 columns
    const int col0 = tile_n * BN;
    // the cohort's j-th 64-row tile; j past the end CLAMPED to the last tile (look-ahead re-reads stay in range)
    auto tile_of = [&](int j) { return mx0 + coh + min(j, nc - 1) * NCOH; };

    const float sav = sa[0], sbv = sb[0];
#if !BSR_U
    // MODE.FP16_OVFL (bit 23): the fp8 converts saturate to +-448 (k512_fp8_gemm.cu NOCLAMP=1)
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n s_nop 7\n s_nop 7" ::: "memory");
#endif
    // (U: no MODE.FP16_OVFL setreg -- the bf16 converts must not saturate; k512_mx6_u.cu keeps the default MODE)

    auto run = [&]<int ROLE>() {
        // ---- per-lane constants (shipped formulas) ----
        const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);
        const int r16 = lane & 15, q4 = lane >> 4;
        const uint32_t hsw = (uint32_t)((r16 >> 2) & 3);
        const uint32_t tsw = (uint32_t)(2 * ((r16 >> 3) & 1));
        const uint32_t lpH0 = lds0 + (uint32_t)r16 * 64u + 16u * ((uint32_t)q4 ^ hsw);  // shipped lpH, slot-relative
        const uint32_t lpT0 = lds0 + (uint32_t)r16 * 32u + 8u * ((uint32_t)q4 ^ tsw);   // shipped lpT - TAILOFF
#if BSR_SCL
        const uint32_t lpS0 = lds0 + (uint32_t)SCLR0 + (uint32_t)lane * 4u;            // scale slot, lane's dword
#endif

        // ---- ring refill: item (64-row tile mm, K step kk) -> ring slot sl (wave-uniform scalars) ----
        // ROLE_A wave c (0..5): linear 1 KiB copy, producer-item bytes [c < 4 ? 4096H + 1024c : 8192 + 2048H + 1024(c-4),
        // +1024) -> slot bytes [1024c, +1024) (per-lane voffset 16 lane; soffset = K step item + chunk; shipped TL form)
        const uint32_t cH0 = (w < 4) ? 1024u * (uint32_t)w : (uint32_t)TAILOFF + 1024u * (uint32_t)(w - 4);
        const uint32_t cHs = (w < 4) ? 4096u : 2048u;
        auto dma = [&](int mm, int kk, int sl) {
            if constexpr (ROLE == ROLE_A) {
                const i32x4 r = make_rsrc(A + (size_t)(mm >> 1) * (KSTEPS * SLOT), (uint32_t)(KSTEPS * SLOT));
                g2l(r, lds0 + (uint32_t)sl * HSL + 1024u * (uint32_t)w, (uint32_t)lane * 16u,
                    (uint32_t)kk * SLOT + cH0 + (uint32_t)(mm & 1) * cHs);
            } else if constexpr (ROLE == ROLE_S) {
                // xs [T][4][512]: lane's dword H of block (T, kk) (= shipped sA[kk].x / .y) -> scale slot + 4 lane
                const i32x4 r = make_rsrc(ascale + (size_t)(mm >> 1) * (KSTEPS * 512), (uint32_t)(KSTEPS * 512));
                k512mx6_llvm_buffer_load_lds(r, (lds_ptr_t)(uintptr_t)(lds0 + (uint32_t)SCLR0 + (uint32_t)sl * SCLB), 4,
                                             (int)((uint32_t)lane * 8u), (int)((uint32_t)kk * 512u + 4u * (uint32_t)(mm & 1)),
                                             0, 0);
            }
        };
#if BSR_SCL == 0
        // activation-scale dword of (64-row tile mm, K step kk): byte 8 lane + 4H of xs block (T, kk) = shipped sA[kk].x/.y
        auto load_sA = [&](int mm, int kk) -> int {
            return reinterpret_cast<const int*>(ascale + (size_t)(mm >> 1) * (KSTEPS * 512) + (size_t)kk * 512)[2 * lane + (mm & 1)];
        };
#endif

        // ---- prologue: the ring's first S-1 items, the register-resident weights, sW, bias (and the first scale) ----
        K512MX6_SB();
#pragma unroll
        for (int i = 0; i < NSLOT - 1; ++i) dma(tile_of(i >> 2), i & (KSTEPS - 1), i);
        K512MX6_SB();
        i32x4 wlo[KSTEPS][2][2];  // [K step][B0 / B1 half h][fragment n] head dwords 0..3
        i32x2 whi[KSTEPS][2][2];  //                                      tail dwords 4..5
        {
            // w6 item (2 tile_n + h, kk) at B + ((2 tile_n + h) * 4 + kk) * 12288; in-item offsets of fragment n = the
            // shipped read_B<SL> ds_read addresses minus the slot base: head lpBH - lds0 + 1024 n, tail lpBT - lds0 + 512 n
            const uint8_t* wb = B + (size_t)(2 * tile_n) * (KSTEPS * SLOT);
            const uint32_t woH = (uint32_t)r16 * 64u + 16u * ((uint32_t)q4 ^ hsw) + (uint32_t)wl * 2048u;
            const uint32_t woT = (uint32_t)TAILOFF + (uint32_t)r16 * 32u + 8u * ((uint32_t)q4 ^ tsw) + (uint32_t)wl * 1024u;
#pragma unroll
            for (int kk = 0; kk < KSTEPS; ++kk)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint8_t* it = wb + (size_t)((h * KSTEPS + kk) * SLOT);
#pragma unroll
                    for (int n = 0; n < 2; ++n) {
                        wlo[kk][h][n] = *reinterpret_cast<const i32x4*>(it + woH + 1024u * (uint32_t)n);
                        whi[kk][h][n] = *reinterpret_cast<const i32x2*>(it + woT + 512u * (uint32_t)n);
                    }
                }
        }
        // W scale: [N/32][q][S]: lane's 32-column block = col0/32 + 2 wl + ((lane&15)>>3), q = lane>>4; byte S = K step
        int sW = reinterpret_cast<const int*>(wscale)[(col0 / 32 + 2 * wl + ((lane & 15) >> 3)) * 4 + (lane >> 4)];
        i32x4 bq0, bq1;  // k512_fp8_gemm.cu: raw bf16 bias of this lane's 16 columns (constant per wave: fixed columns)
        {
            const uint16_t* bp = bias + col0 + 64 * wl + 16 * (lane >> 4);
            bq0 = *reinterpret_cast<const i32x4*>(bp);
            bq1 = *reinterpret_cast<const i32x4*>(bp + 8);
        }
#if BSR_SCL == 0
        int sAc = load_sA(tile_of(0), 0);
#endif
        K512MX6_SB();
        wait_vm<0>();  // P0: every prologue VMEM op of this wave landed (ring items 0..S-2, weights, sW, bias, scale)
        // materialize the loaded registers here (the l11_do8.cu idiom): no load is pending on the loop-entry path, so no
        // compiler wait inside the loop depends on a prologue load; the weights stay i32x4 + i32x2 (never an 8-dword value)
#pragma unroll
        for (int kk = 0; kk < KSTEPS; ++kk)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int n = 0; n < 2; ++n) asm volatile("" : "+v"(wlo[kk][h][n]), "+v"(whi[kk][h][n]));
        asm volatile("" : "+v"(sW), "+v"(bq0), "+v"(bq1));
#if BSR_SCL == 0
        asm volatile("" : "+v"(sAc));
#endif
        K512MX6_BARRIER();  // P1: ring items 0..S-2 of every issuing wave visible to all waves

        const float s = sav * sbv;  // epilogue constants (loop-invariant)
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

        i32x8 a[4];          // ONE activation register bank
        f32x4 acc[2][2][4];  // [B0 / B1][n][subtile i] = the shipped acc[2H][..] / acc[2H + 1][..] of this half
        int m = tile_of(0);

#pragma nounroll
        for (int j = 0; j < nc; ++j) {
            const int mn = tile_of(j + 1);         // next tile (clamped)
            const int mA = tile_of(j + NGRP - 1);  // source tile of the K-step-0 refill (its K step 3)
            const int mB = tile_of(j + NGRP);      // source tile of the K-step-1..3 refills (its K steps 0..2)
            const int gc = (j & (NGRP - 1)) * KSTEPS;               // ring slot of item (j, 0)
            const int gp = ((j + NGRP - 1) & (NGRP - 1)) * KSTEPS;  // ring slot of item (j - 1, 0) (= (j + NGRP - 1, 0))
            const uint32_t lpH = lpH0 + (uint32_t)gc * HSL;
            const uint32_t lpT = lpT0 + (uint32_t)gc * HSL;
#if BSR_SCL
            const uint32_t lpS = lpS0 + (uint32_t)gc * SCLB;
#else
            int sAn;
#endif

            auto step = [&]<int KS>() {
                if constexpr (ROLE != ROLE_N) wait_vm<ring_wait(KS)>();  // own DMA of item (j, KS) landed
                K512MX6_BARRIER();  // item (j, KS) complete (all issuers); every wave done reading the previous item
#if BSR_SCL == 0
                // next K step's scale dword, issued BEFORE this step's DMA (the compiler's wait for it then never
                // covers the DMA issued in the same step)
                if constexpr (KS < KSTEPS - 1) sAn = load_sA(m, KS + 1);
                else sAn = load_sA(mn, 0);
                K512MX6_SB();
#endif
                // refill the slot the previous item freed with item (k - 1 + S)
                if constexpr (KS == 0) dma(mA, KSTEPS - 1, gp + KSTEPS - 1);
                else dma(mB, KS - 1, gc + KS - 1);
                K512MX6_SB();
                read_A6<KS>(a, lpH, lpT);
#if BSR_SCL
                const int sx = ds_read4<KS * SCLB>(lpS);
#else
                const int sx = sAc;
#endif
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                K512MX6_SB();
                __builtin_amdgcn_s_setprio(1);
                {
                    const i32x8 b0[2] = {wfrag(wlo[KS][0][0], whi[KS][0][0]), wfrag(wlo[KS][0][1], whi[KS][0][1])};
                    mma8<KS, KS == 0>(acc[0], b0, a, sW, sx);
                }
                {
                    const i32x8 b1[2] = {wfrag(wlo[KS][1][0], whi[KS][1][0]), wfrag(wlo[KS][1][1], whi[KS][1][1])};
                    mma8<KS, KS == 0>(acc[1], b1, a, sW, sx);
                }
                __builtin_amdgcn_s_setprio(0);
                K512MX6_SB();
#if BSR_SCL == 0
                sAc = sAn;
#endif
            };
            step.template operator()<0>();
            step.template operator()<1>();
            step.template operator()<2>();
            step.template operator()<3>();

            // ---- epilogue of 64-row tile m (shipped arithmetic, verbatim) ----
            const int row0 = m * BMR;
            const int rows_valid = min(BMR, M - row0);
#if !BSR_U
            // k512_fp8_gemm.cu NOCLAMP=1 form: e4m3 = cvt_pk_fp8(sat448(acc * s + bias)) under MODE.FP16_OVFL
            const uint32_t lane_off = (uint32_t)(lane & 15) * (uint32_t)N + (uint32_t)(64 * wl + 16 * (lane >> 4));
            uint8_t* Ct = C + (size_t)row0 * N + col0;
#pragma unroll
            for (int ms = 0; ms < 4; ++ms) {
                const f32x4 v[4] = {acc[0][0][ms], acc[0][1][ms], acc[1][0][ms], acc[1][1][ms]};
                i32x4 d;
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    f32x2 lo = __builtin_shufflevector(v[jj], v[jj], 0, 1);
                    f32x2 hi = __builtin_shufflevector(v[jj], v[jj], 2, 3);
                    lo = lo * s2;
                    hi = hi * s2;
                    lo = lo + bias2[2 * jj];
                    hi = hi + bias2[2 * jj + 1];
                    int pk = __builtin_amdgcn_cvt_pk_fp8_f32(lo.x, lo.y, __builtin_bit_cast(int, lo.x), false);
                    pk = __builtin_amdgcn_cvt_pk_fp8_f32(hi.x, hi.y, pk, true);
                    d[jj] = pk;
                }
                int rv = rows_valid - 16 * ms;
                rv = rv < 0 ? 0 : (rv > 16 ? 16 : rv);
                const uint32_t nrec = (uint32_t)__builtin_amdgcn_readfirstlane(rv * N);
                const i32x4 rC = make_rsrc(Ct + (size_t)(16 * ms) * N, nrec);
                k512mx6_llvm_buffer_store_v4i32(d, rC, (int)lane_off, 0, 0);
                K512MX6_SB();
            }
#else
            // k512_mx6_u.cu's U epilogue (K512 column mapping): out = bf16_rne(f32(f32(acc * s) + bias)), lane ids from mbcnt
            constexpr uint32_t ROWB = (uint32_t)N * 2u;
            asm volatile("; k512_mx6_u bf16 epilogue" ::: "memory");
            const int le = (int)__builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u));
            const uint32_t lane_off = (uint32_t)(le & 15) * ROWB + (uint32_t)(64 * wl + 16 * (le >> 4)) * 2u;
            uint16_t* Ct = reinterpret_cast<uint16_t*>(C) + (size_t)row0 * N + col0;
#pragma unroll
            for (int ms = 0; ms < 4; ++ms) {
                const f32x4 v[4] = {acc[0][0][ms], acc[0][1][ms], acc[1][0][ms], acc[1][1][ms]};
                uint32_t pk[8];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    f32x2 lo = __builtin_shufflevector(v[jj], v[jj], 0, 1);
                    f32x2 hi = __builtin_shufflevector(v[jj], v[jj], 2, 3);
                    lo = lo * s2;
                    hi = hi * s2;
                    lo = lo + bias2[2 * jj];
                    hi = hi + bias2[2 * jj + 1];
                    pk[2 * jj] = pk_bf16(lo);
                    pk[2 * jj + 1] = pk_bf16(hi);
                }
                const i32x4 d0 = {(int)pk[0], (int)pk[1], (int)pk[2], (int)pk[3]};  // columns +0..7
                const i32x4 d1 = {(int)pk[4], (int)pk[5], (int)pk[6], (int)pk[7]};  // columns +8..15
                int rv = rows_valid - 16 * ms;
                rv = rv < 0 ? 0 : (rv > 16 ? 16 : rv);
                const uint32_t nrec = (uint32_t)__builtin_amdgcn_readfirstlane(rv * (int)ROWB);
                const i32x4 rC = make_rsrc(Ct + (size_t)(16 * ms) * N, nrec);
                k512mx6_llvm_buffer_store_v4i32(d0, rC, (int)lane_off, 0, 0);
                k512mx6_llvm_buffer_store_v4i32(d1, rC, (int)(lane_off + 16u), 0, 0);
                K512MX6_SB();
            }
#endif
            m = mn;
        }
        wait_vm<0>();  // the clamped trailing ring refills land before the WG's LDS is released
    };

    if (w < 6) {
        run.template operator()<ROLE_A>();
#if BSR_SCL
    } else if (w == 6) {
        run.template operator()<ROLE_S>();
#endif
    } else {
        run.template operator()<ROLE_N>();
    }
}

template <int N>
static int k512_mx6_bsr_launch(const void* x6, const void* xs, const void* w6, const void* ws, void* out,
                               const void* bias, const float* sa, const float* sb, int M, void* stream) {
    if (M <= 0 || !x6 || !xs || !w6 || !ws || !out || !bias || !sa || !sb) return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();  // a sticky error left by an EARLIER call is reported, not hidden
    hipLaunchKernelGGL((k512_mx6_bsr_kernel<N>), dim3((unsigned)k512mx6::BSR_GRID), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)x6, (const uint8_t*)w6, (uint8_t*)out, (const uint16_t*)bias, sa, sb,
                       (const uint8_t*)xs, (const uint8_t*)ws, M);
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) return (int)err;
    if (stale != hipSuccess) return K512MX6_STALE_ERROR_BASE + (int)stale;
    return 0;
}

#define K512MX6_CAT_(a, b) a##b
#define K512MX6_CAT(a, b) K512MX6_CAT_(a, b)

#if !BSR_U
// C ABI (shipped k512_mx6_gemm.cu, same argument order):
//   out e4m3 [M, N] = e4m3_rne(sat448(acc * (sa[0] * sb[0]) + float(bias[n]))); rows >= M are not written.
//   Returns 0, this launch's hipError_t, or 100000 + a sticky error an EARLIER HIP call left on the thread.
extern "C" int K512MX6_CAT(k512_mx6_e4m3_bias_N, HK_N)(const void* x6, const void* xs, const void* w6, const void* ws,
                                                       void* out, const void* bias, const float* sa, const float* sb,
                                                       int M, void* stream) {
    return k512_mx6_bsr_launch<HK_N>(x6, xs, w6, ws, out, bias, sa, sb, M, stream);
}
extern "C" const char* k512_mx6_gemm_impl(void) {
    return "k512mx6 BSR O-D2: persistent 256 x 8-wave WGs, 64x512 tile, weights "
           "REGISTER-resident (96 VGPR/wave, loaded once), 6 KiB half-item LDS ring, 1 barrier per K step, cohort "
           "schedule per XCD";
}
#else
// C ABI (shipped k512_mx6_u.cu): out bf16 [M, 512] = bf16_rne(f32(f32(acc * s) + bias)), s = sa[0] * sb[0]; rows < M
// written. Returns hipGetLastError() of the launch (hipErrorInvalidValue for M <= 0).
extern "C" int k512_mx6_u_bf16_bias(const void* x6, const void* xs, const void* w6, const void* ws, void* out_bf16,
                                    const void* bias_bf16, const float* sa, const float* sb, int M, void* stream) {
    if (M <= 0) return (int)hipErrorInvalidValue;
    hipLaunchKernelGGL((k512_mx6_bsr_kernel<512>), dim3((unsigned)k512mx6::BSR_GRID), dim3(512), 0,
                       (hipStream_t)stream, (const uint8_t*)x6, (const uint8_t*)w6, (uint8_t*)out_bf16,
                       (const uint16_t*)bias_bf16, sa, sb, (const uint8_t*)xs, (const uint8_t*)ws, M);
    return (int)hipGetLastError();
}
extern "C" const char* k512_mx6_u_impl(void) {
    return "k512_mx6_u BSR O-D2: persistent 256 x 8-wave WGs, 64x512 tile, weights "
           "REGISTER-resident, 6 KiB half-item LDS ring, bf16 + bias epilogue (k512_mx6_u.cu)";
}
#endif

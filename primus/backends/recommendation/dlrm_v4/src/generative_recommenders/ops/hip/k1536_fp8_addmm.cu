// K=1536 FP8 output-projection forward GEMM with the fused bf16 residual, behind DLRM_HSTU_FP8_FWD_K1536
// (generative_recommenders/ops/hip/k1536_fp8_addmm.py; dispatch in ops/triton/triton_fp8.scaled_addmm_fp8_preq, residual
// branch). Bit-identical to the residual-fused hipBLASLt call it replaces (ops/triton/hipblaslt_scaled_addmm.cpp,
// algorithm index 1, 64 MiB workspace): ops/tests/k1536_fp8_addmm_test.py compares every output word.
//
// Provenance: Arbor P7 iteration 11, sub-track B7 G3b (probes/iter11/g3b, source k1536_res.cu sha256 3828858edf45),
// gated at 1.08-1.10x over hipBLASLt in sustained 1000 W soaks (3 GPUs, 2 nodes) and 1.13x when interleaved with other
// GEMM work. Production default RWAIT = 0 (one vmcnt(0) at the epilogue start, before any store: no assumption about
// the retire order of interleaved VMEM loads and stores). Experiment / diagnostic variants are compiled only with
// -DK1536_EXPERIMENT_VARIANTS.
//
// P7 iter11 track g3b: HSTU output-projection FORWARD FP8 GEMM + residual (component B), K=1536, N=512, bf16 out.
//
//   D[M,512] (bf16) = production residual-fused hipBLASLt call (testbed ops/triton/triton_fp8.py:1262-1282 ->
//                     ops/triton/hipblaslt_scaled_addmm.cpp): activation e4m3 [M,1536] row-major, weight e4m3 storage
//                     [512,1536] row-major (= the [1536,512] column-major view production passes), residual C bf16
//                     [M,512], alpha = beta = 1, scaleA = weight scale, scaleB = activation scale, compute 32F.
//   acc[m][n] = sum_k A[m][k] * W[n][k]      fp32, TWELVE IN-ORDER K=128 STEPS (one MFMA chain per 16x16 block)
//   D[m][n]   = bf16_rne( fp32( acc * s ) + fp32(C[m][n]) ),  s = s_w * s_act (one fp32 multiply), MUL THEN ADD
//               (= the production epilogue: exact-integer test g3b_test.py uepi, 0 mismatches over 1.2e9 outputs; FMA,
//               (acc*s_w)*s_act and bf16-then-add rejected). The production SK3 stream-K kernels keep this single
//               in-order chain: bitwise at 26 M values from 17 to 2.8M (out/bc_*, out/msweep_*).
//
// Derived from P7's in-tree K=512 kernel (/scratch/training_dlrmv4_arbor_pr7/recommendation/generative_recommenders/
// ops/hip/k512_fp8_gemm.cu, cited K512:<line>) and r14's bf16-epilogue fork (/scratch/arbor_sessions/
// dlrmv4_20261005_r14/probes/iter37/od/od_dgrad.cu, cited OD:<line>). Operand register layout and the st_16x128 XOR
// LDS swizzle follow HipKittens (https://github.com/HazyResearch/HipKittens, MIT License, Copyright (c) 2024 HazyResearch).
//
// Borrowed unchanged: buffer resource (K512:75-83), barrier macros (K512:85-93), ds_read16/frag/read_A/read_B
// (K512:95-133), mma8 (K512:136-150), LDS-DMA issue (K512:152-168, B row offset generalised to 64*KD), XCD remap + tile
// decode (K512:186-197), LDS-DMA source offsets / swizzle and LDS read offsets (K512:211-236 with r14's STORE=1 B-row
// permutation OD:193-216), the SCHED0 3-barrier 5-slot ring step (K512:250-288) and prologue (K512:362-372), the bf16
// epilogue packing (OD:150-158, 284-327: v_cvt_pk_bf16_f32 RNE, 2 x dwordx4 stores per 16-row group, per-group scalar
// buffer resource whose num_records drops rows >= M, inline-0 soffset).
//
// New for g3b:
//   * K = 1536: 12 K steps through the same 5-slot ring (item j = 3k+t in slot j % 5).
//   * Residual: loaded by LDS-DMA (no VGPRs) into ring slots that the last two K steps free (step 10's 3rd barrier frees
//     2 slots -> rows 0-63; step 11's 2nd barrier frees 1 slot -> rows 64-95; its 3rd barrier -> rows 96-127), laid out
//     PER WAVE and per lane exactly as the epilogue consumes it (16 B per lane, consecutive lanes -> conflict-free
//     ds_read_b128, no cross-wave dependency). Rows >= M read as 0 (num_records), never used.
//   * RWAIT 0: s_waitcnt vmcnt(0) once at the epilogue start (no load/store ordering assumption); RWAIT 1 (ABI): per-group
//     in-order vmcnt (VMEM loads and stores retire in issue order on one counter, as LLVM assumes for gfx9 without vscnt).
//   * Load cache policy variants APOL/BPOL (aux bits of the A / B LDS-DMA loads; 0 = default = ABI). DIAG 6 / 7: A-only /
//     B-only operand loads (diagnostic).
//   * DIAG (diagnostic builds only, NOT bitwise): 1 stores dropped (num_records 0), 2 no store instruction, 3 no residual
//     load (residual = 0), 4 no operand global->LDS loads, 5 = 2 + 3.
// Build: hipcc --offload-arch=gfx950 -std=c++20 -O3 -mllvm -amdgpu-sched-strategy=iterative-ilp -ffp-contract=off
//        -shared -fPIC (build.sh).
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

#ifndef G3_N
#define G3_N 512
#endif
#ifndef G3_K
#define G3_K 1536
#endif

namespace g3 {
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}  // namespace g3

extern "C" __device__ void g3_llvm_buffer_load_lds(g3::i32x4 rsrc, g3::lds_ptr_t lds, int size, int voffset,
                                                   int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
extern "C" __device__ void g3_llvm_buffer_store_v4i32(g3::i32x4 vdata, g3::i32x4 rsrc, int voffset, int soffset,
                                                      int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

namespace g3 {

constexpr int KD = G3_K;          // K
constexpr int BM = 128;           // WG tile rows (M)
constexpr int BN = 256;           // WG tile cols (N)
constexpr int KSTEPS = KD / 128;  // 12
constexpr int NSLOT = 5;          // LDS ring slots
constexpr int SLOT = 16384;       // bytes per slot (128 x 128 fp8)
static_assert(KD % 128 == 0 && KSTEPS >= 3, "K must be a multiple of 128, >= 384");

// K512:75-83
__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}

// K512:85-93
#define G3_FENCE() asm volatile("" ::: "memory")
#define G3_SB() __builtin_amdgcn_sched_barrier(0)
#define G3_BARRIER()                  \
    do {                              \
        G3_FENCE();                   \
        __builtin_amdgcn_s_barrier(); \
        G3_FENCE();                   \
        G3_SB();                      \
    } while (0)

// K512:95-101
template <int IMM>
__device__ __forceinline__ i32x4 ds_read16(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x4 v;
    asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}

// any LDS byte offset (ds immediates are 16 bit): K512:110-116 split
template <int OFF>
__device__ __forceinline__ i32x4 ds_read16_any(uint32_t vaddr) {
    if constexpr (OFF < 65536) return ds_read16<OFF>(vaddr);
    else return ds_read16<OFF - 65536>(vaddr + 65536u);
}

// K512:107-118
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

// K512:121-133
template <int SL, int H>
__device__ __forceinline__ void read_A(i32x8 (&a)[4], uint32_t lp0, uint32_t lp1) {
    a[0] = frag<SL * SLOT + H * 8192 + 0 * 2048>(lp0, lp1);
    a[1] = frag<SL * SLOT + H * 8192 + 1 * 2048>(lp0, lp1);
    a[2] = frag<SL * SLOT + H * 8192 + 2 * 2048>(lp0, lp1);
    a[3] = frag<SL * SLOT + H * 8192 + 3 * 2048>(lp0, lp1);
}
template <int SL>
__device__ __forceinline__ void read_B(i32x8 (&b)[2], uint32_t lpB0, uint32_t lpB1) {
    b[0] = frag<SL * SLOT + 0>(lpB0, lpB1);
    b[1] = frag<SL * SLOT + 2048>(lpB0, lpB1);
}

// K512:136-150
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

// K512:152-168 (B per-i row offset 64 rows = 64*KD bytes)
// AUX = cache policy bits of the load (gfx950: 1 = sc0, 2 = nt, 16 = sc1); 0 = default
template <int AUX = 0>
__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    g3_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, AUX);
}
template <int SL, int KK, int AUX = 0>
__device__ __forceinline__ void issue_A(i32x4 rA, uint32_t ldsw, const uint32_t (&offA)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) g2l<AUX>(rA, ldsw + SL * SLOT + i * 4096, offA[i], KK * 128);
}
template <int SL, int KK, int AUX = 0>
__device__ __forceinline__ void issue_B(i32x4 rB, uint32_t ldsw, uint32_t offB) {
#pragma unroll
    for (int i = 0; i < 4; ++i) g2l<AUX>(rB, ldsw + SL * SLOT + i * 4096, offB, i * (64 * KD) + KK * 128);
}

// Residual group G (tile rows [16G, 16G+16)): slot that holds it and the byte offset of (G, half HF) inside the slot.
// Groups 0-1 -> slot of B1_{KSTEPS-2}, 2-3 -> A_{KSTEPS-2}, 4-5 -> B0_{KSTEPS-1}, 6-7 -> B1_{KSTEPS-1}.
template <int G>
constexpr int rslot() {
    constexpr int k2 = KSTEPS - 2, k1 = KSTEPS - 1;
    return G < 2 ? (3 * k2 + 1) % NSLOT : G < 4 ? (3 * k2 + 2) % NSLOT : G < 6 ? (3 * k1) % NSLOT : (3 * k1 + 1) % NSLOT;
}
template <int G, int HF>
constexpr int roff() {
    return rslot<G>() * SLOT + ((G & 1) * 2 + HF) * 4096;
}
// LDS-DMA of residual group G, both halves, this wave's 64 columns: lane l -> row 16G + l%16, 16 B at column
// 64w + 32*HF + 8*(l/16) (the STORE=1 output mapping); LDS destination = roff + w*1024 + l*16 (ldsw carries w*1024).
template <int G>
__device__ __forceinline__ void issue_R(i32x4 rC, uint32_t ldsw, uint32_t offR) {
    constexpr int ROWB = G3_N * 2;
    g2l(rC, ldsw + roff<G, 0>(), offR, G * 16 * ROWB);
    g2l(rC, ldsw + roff<G, 1>(), offR, G * 16 * ROWB + 64);
}

}  // namespace g3

// RWAIT: 0 = vmcnt(0) at the epilogue start, 1 = per-group in-order vmcnt.  PRIO: s_setprio around MFMA blocks.
// DIAG: diagnostic variants (see header), 0 = deliverable.
template <int N, int RWAIT, int PRIO, int DIAG, int APOL = 0, int BPOL = 0>
__global__ __launch_bounds__(256, 2) void k1536_res_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                     const uint16_t* __restrict__ C, uint16_t* __restrict__ D,
                                                     const float* __restrict__ s_act, const float* __restrict__ s_w,
                                                     int M) {
#pragma clang fp contract(off)
    using namespace g3;
    static_assert(N % BN == 0, "N must be a multiple of 256");
    constexpr int NT = N / BN;
    constexpr uint32_t ROWB = (uint32_t)N * 2u;  // bytes per bf16 row (output and residual)
    constexpr bool OPLOADS = (DIAG != 4);
    constexpr bool ALOADS = OPLOADS && DIAG != 7;  // DIAG 7: B operand loads only
    constexpr bool BLOADS = OPLOADS && DIAG != 6;  // DIAG 6: A operand loads only
    constexpr bool RLOADS = (DIAG != 3 && DIAG != 5);
    __shared__ __attribute__((aligned(2048))) uint8_t smem[NSLOT * SLOT];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));

    // K512:186-197 (XCD-aware tile order: the NT=2 tiles of one A row panel get consecutive ids on one XCD)
    int bid = (int)blockIdx.x;
    {
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

    const float sav = s_act[0], swv = s_w[0];

    // ---- buffer resources (scalar); K512:206-209 with r14's STORE=1 B1 item at +32 rows (OD:193-197) ----
    constexpr int B1ROWS = 32;
    const i32x4 rA = make_rsrc(A + (size_t)row0 * KD, (uint32_t)(rows_valid * KD));
    const uint8_t* Bt = B + (size_t)col0 * KD;
    const i32x4 rB0 = make_rsrc(Bt, (uint32_t)((N - col0) * KD));
    const i32x4 rB1 = make_rsrc(Bt + B1ROWS * KD, (uint32_t)((N - col0) * KD - B1ROWS * KD));
    const i32x4 rC = make_rsrc(C + (size_t)row0 * N + col0, (uint32_t)rows_valid * ROWB);

    // ---- per-lane global source offsets of the LDS-DMA loads (K512:211-225, OD:199-216 STORE=1) ----
    const uint32_t chunk = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    uint32_t offA[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) offA[i] = (uint32_t)(32 * i + 8 * w + (lane >> 3)) * KD + chunk * 16;
    uint32_t offB;
    {
        const int q = 8 * w + (lane >> 3);
        const int jj = q >> 4, npos = q & 15;
        offB = (uint32_t)(8 * (npos >> 2) + 4 * jj + (npos & 3)) * KD + chunk * 16;
    }
    const int r16 = lane & 15, q4 = lane >> 4;
    // residual: row r16 of the group, columns 64w + 8*q4 (+32 for the second half: +64 B in soffset)
    const uint32_t offR = (uint32_t)r16 * ROWB + (uint32_t)(64 * w + 8 * q4) * 2u;

    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);
    const uint32_t ldsw = lds0 + (uint32_t)w * 1024u;  // wave-uniform M0 base for the LDS-DMA loads

    // K512:231-236
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t lpA0 = lds0 + (uint32_t)r16 * 128u + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lpA1 = lds0 + (uint32_t)r16 * 128u + (((uint32_t)(16 * q4 + 64)) ^ swz);
    const uint32_t lpB0 = lpA0 + (uint32_t)w * 4096u;
    const uint32_t lpB1 = lpA1 + (uint32_t)w * 4096u;

    i32x8 a[4], b0[2], b1[2];
    f32x4 acc[4][2][4];  // [b0 x a(rows 0-63), b1 x a(0-63), b0 x a(64-127), b1 x a(64-127)][n][m]

    auto prio_on = [&]() { if constexpr (PRIO) __builtin_amdgcn_s_setprio(1); };
    auto prio_off = [&]() { if constexpr (PRIO) __builtin_amdgcn_s_setprio(0); };
    auto iA = [&]<int SL, int KK>() { if constexpr (ALOADS) issue_A<SL, KK, APOL>(rA, ldsw, offA); };
    auto iB = [&]<int SL, int KK>(const i32x4& r) { if constexpr (BLOADS) issue_B<SL, KK, BPOL>(r, ldsw, offB); };
    auto iR = [&]<int G>() { if constexpr (RLOADS) issue_R<G>(rC, ldsw, offR); };
    // number of residual LDS-DMA loads issued after this step's items (vmcnt bookkeeping)
    constexpr int NR_K2 = RLOADS ? 8 : 0;  // rows 0-63, issued at step KSTEPS-2's 3rd barrier

    // K512:250-288 (SCHED 0). Last two steps: the freed slots take the residual instead of B0/B1_{KS+2} / A_{KS+1}.
    auto step0 = [&]<int KS>() {
        constexpr int sB0 = (3 * KS) % NSLOT, sB1 = (3 * KS + 1) % NSLOT, sA = (3 * KS + 2) % NSLOT;
        if constexpr (KS < KSTEPS - 1) {
            // younger than A_KS: B0/B1_{KS+1} (4 loads each when issued)
            if constexpr (BLOADS) asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
            else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        } else {  // younger than A_{KS}: the 8 residual loads of rows 0-63 (if any)
            if constexpr (NR_K2 == 8) asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
            else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
        G3_BARRIER();
        read_B<sB0>(b0, lpB0, lpB1);
        read_A<sA, 0>(a, lpA0, lpA1);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        G3_BARRIER();  // all waves consumed B0_KS -> its slot takes A_{KS+1} (or residual rows 64-95 in the last step)
        if constexpr (KS + 1 < KSTEPS) iA.template operator()<sB0, KS + 1>();
        else { iR.template operator()<4>(); iR.template operator()<5>(); }
        G3_SB();
        prio_on();
        mma8<KS == 0>(acc[0], b0, a);
        prio_off();
        G3_SB();
        read_B<sB1>(b1, lpB0, lpB1);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        G3_SB();
        prio_on();
        mma8<KS == 0>(acc[1], b1, a);
        prio_off();
        G3_SB();
        read_A<sA, 1>(a, lpA0, lpA1);
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        G3_BARRIER();  // all waves consumed B1_KS and A_KS -> their slots take B0/B1_{KS+2} (or residual rows)
        if constexpr (KS + 2 < KSTEPS) {
            iB.template operator()<sB1, KS + 2>(rB0);
            iB.template operator()<sA, KS + 2>(rB1);
        } else if constexpr (KS + 2 == KSTEPS) {
            iR.template operator()<0>(); iR.template operator()<1>(); iR.template operator()<2>(); iR.template operator()<3>();
        } else {
            iR.template operator()<6>(); iR.template operator()<7>();
        }
        G3_SB();
        prio_on();
        mma8<KS == 0>(acc[2], b0, a);
        mma8<KS == 0>(acc[3], b1, a);
        prio_off();
        G3_SB();
    };

    // K512:362-372
    iB.template operator()<0, 0>(rB0);
    iB.template operator()<1, 0>(rB1);
    iA.template operator()<2, 0>();
    iB.template operator()<3, 1>(rB0);
    iB.template operator()<4, 1>(rB1);
    [&]<int... KS>(std::integer_sequence<int, KS...>) {
        (step0.template operator()<KS>(), ...);
    }(std::make_integer_sequence<int, KSTEPS>{});

    // ---- epilogue: lane owns row (16g + l%16) of the tile, columns [64w + 8*q4, +8) and [64w + 32 + 8*q4, +8) ----
    const float s = swv * sav;  // scaleA * scaleB (fp32, commutative)
    const f32x2 s2 = {s, s};
    const uint32_t colb = (uint32_t)(64 * w + 8 * q4);
    const uint32_t lane_off = (uint32_t)r16 * ROWB + colb * 2u;
    const uint32_t ldsRl = lds0 + (uint32_t)w * 1024u + (uint32_t)lane * 16u;  // + roff<G,HF>
    uint16_t* Dt = D + (size_t)row0 * N + col0;
    if constexpr (RLOADS && RWAIT == 0) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    G3_SB();

    i32x4 rr[8][2];
    auto rread = [&]<int G>() {
        if constexpr (RLOADS) {
            if constexpr (RWAIT == 1) {
                // residual loads of groups <= G landed; younger: loads of groups > G and the stores of groups < G-1
                // (2 per group) -> 16 - 2(G+1) + 2*max(G-1, 0) ... = 14 (G = 0) / 12 (G >= 1) under in-order retire
                if constexpr (G == 0) asm volatile("s_waitcnt vmcnt(14)" ::: "memory");
                else asm volatile("s_waitcnt vmcnt(12)" ::: "memory");
            }
            rr[G][0] = ds_read16_any<roff<G, 0>()>(ldsRl);
            rr[G][1] = ds_read16_any<roff<G, 1>()>(ldsRl);
        } else {
            rr[G][0] = i32x4{0, 0, 0, 0};
            rr[G][1] = i32x4{0, 0, 0, 0};
        }
    };
    rread.template operator()<0>();
    [&]<int... G>(std::integer_sequence<int, G...>) {
        (([&] {
             if constexpr (G + 1 < 8) {
                 rread.template operator()<G + 1>();
                 if constexpr (RLOADS) asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
             } else {
                 if constexpr (RLOADS) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
             }
             G3_SB();  // no use of rr[G] may be scheduled above the wait (asm outputs carry no wait dependency)
             constexpr int h = G >> 2, ms = G & 3;
             const f32x4 v[4] = {acc[2 * h][0][ms], acc[2 * h][1][ms], acc[2 * h + 1][0][ms], acc[2 * h + 1][1][ms]};
             const int rw[8] = {rr[G][0].x, rr[G][0].y, rr[G][0].z, rr[G][0].w,
                                rr[G][1].x, rr[G][1].y, rr[G][1].z, rr[G][1].w};
             uint32_t pk[8];
#pragma unroll
             for (int j = 0; j < 4; ++j) {
                 f32x2 lo = __builtin_shufflevector(v[j], v[j], 0, 1);
                 f32x2 hi = __builtin_shufflevector(v[j], v[j], 2, 3);
                 const uint32_t d0 = (uint32_t)rw[2 * j], d1 = (uint32_t)rw[2 * j + 1];
                 const f32x2 c0 = {__uint_as_float(d0 << 16), __uint_as_float(d0 & 0xffff0000u)};
                 const f32x2 c1 = {__uint_as_float(d1 << 16), __uint_as_float(d1 & 0xffff0000u)};
                 lo = lo * s2;
                 hi = hi * s2;
                 lo = lo + c0;
                 hi = hi + c1;
                 uint32_t r0, r1;
                 asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(r0) : "v"(lo.x), "v"(lo.y));
                 asm("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(r1) : "v"(hi.x), "v"(hi.y));
                 pk[2 * j] = r0;
                 pk[2 * j + 1] = r1;
             }
             const i32x4 o0 = {(int)pk[0], (int)pk[1], (int)pk[2], (int)pk[3]};
             const i32x4 o1 = {(int)pk[4], (int)pk[5], (int)pk[6], (int)pk[7]};
             int rv = rows_valid - 16 * G;
             rv = rv < 0 ? 0 : (rv > 16 ? 16 : rv);
             uint32_t nrec = (uint32_t)__builtin_amdgcn_readfirstlane(rv * (int)ROWB);  // K512:425 (scalar rsrc)
             if constexpr (DIAG == 1) nrec = 0;
             const i32x4 rD = make_rsrc(Dt + (size_t)(16 * G) * N, nrec);
             if constexpr (DIAG == 2 || DIAG == 5) {
                 if (M < 0) {  // never true: keeps the math, no store instruction executed
                     g3_llvm_buffer_store_v4i32(o0, rD, (int)lane_off, 0, 0);
                     g3_llvm_buffer_store_v4i32(o1, rD, (int)(lane_off + 64u), 0, 0);
                 }
             } else {
                 g3_llvm_buffer_store_v4i32(o0, rD, (int)lane_off, 0, 0);
                 g3_llvm_buffer_store_v4i32(o1, rD, (int)(lane_off + 64u), 0, 0);
             }
             G3_SB();
         }()),
         ...);
    }(std::make_integer_sequence<int, 8>{});
}

template <int N, int RWAIT, int PRIO, int DIAG, int APOL = 0, int BPOL = 0>
static int g3b_launch(const void* A, const void* B, const void* C, void* D, const float* s_act, const float* s_w, int M,
                      void* stream) {
    if (M <= 0) return (int)hipErrorInvalidValue;
    const long long tiles = (long long)((M + 127) / 128) * (N / 256);
    if (tiles > 0x7fffffffLL) return (int)hipErrorInvalidValue;
    hipLaunchKernelGGL((k1536_res_kernel<N, RWAIT, PRIO, DIAG, APOL, BPOL>), dim3((unsigned)tiles), dim3(256), 0, (hipStream_t)stream,
                       (const uint8_t*)A, (const uint8_t*)B, (const uint16_t*)C, (uint16_t*)D, s_act, s_w, M);
    return (int)hipGetLastError();
}

#define G3_CAT_(a, b) a##b
#define G3_CAT(a, b) G3_CAT_(a, b)
#define G3_STR_(x) #x
#define G3_STR(x) G3_STR_(x)

#ifndef G3_ABI_RWAIT
#define G3_ABI_RWAIT 0  // 0 = vmcnt(0) at the epilogue start (production). 1 = per-group vmcnt that assumes VMEM loads and
                        // stores retire in issue order (experiment only; LLVM treats a counter with mixed pending load
                        // and store events as out of order).
#endif

// ABI: A = activation e4m3 [M,1536] row-major; Wst = weight e4m3 [512,1536] row-major (storage of the column-major
// [1536,512] view production passes); C = residual bf16 [M,512] row-major; D = output bf16 [M,512] row-major (rows < M
// written only, D != C); s_act, s_w = device fp32 scalars; M >= 1.
extern "C" int G3_CAT(k1536_fp8_addmm_res_bf16_N, G3_N)(const void* A, const void* Wst, const void* C, void* D,
                                                         const float* s_act, const float* s_w, int M, void* stream) {
    return g3b_launch<G3_N, G3_ABI_RWAIT, 1, 0>(A, Wst, C, D, s_act, s_w, M, stream);
}
extern "C" const char* k1536_fp8_addmm_impl(void) {
    return "g3b: P7 2wg family, K=1536 (12 K steps, 5-slot ring), 128x256 tile, swapped MFMA, residual via LDS-DMA into "
           "freed ring slots, bf16 RNE(acc*s + C) epilogue, XCD remap, RWAIT " G3_STR(G3_ABI_RWAIT) " (P7 iter11 g3b)";
}

#ifdef K1536_EXPERIMENT_VARIANTS
// Experiment / diagnostic variants: g3b_N{N}_w{RWAIT}_p{PRIO}_d{DIAG}
#define G3_VAR(rw, pr, dg)                                                                                         \
    extern "C" int G3_CAT(g3b_N, G3_CAT(G3_N, _w##rw##_p##pr##_d##dg))(const void* A, const void* B, const void* C, \
                                                                        void* D, const float* sa, const float* sb,  \
                                                                        int M, void* stream) {                      \
        return g3b_launch<G3_N, rw, pr, dg>(A, B, C, D, sa, sb, M, stream);                                          \
    }
G3_VAR(0, 1, 0)
G3_VAR(1, 1, 0)
G3_VAR(0, 0, 0)
G3_VAR(0, 1, 1)
G3_VAR(0, 1, 2)
G3_VAR(0, 1, 3)
G3_VAR(0, 1, 4)
G3_VAR(0, 1, 5)
G3_VAR(1, 1, 6)
G3_VAR(1, 1, 7)
G3_VAR(1, 0, 0)
// load cache-policy variants: g3b_N{N}_w{RWAIT}_p{PRIO}_a{APOL}_b{BPOL}
#define G3_VARP(rw, pr, ap, bp)                                                                                    \
    extern "C" int G3_CAT(g3b_N, G3_CAT(G3_N, _w##rw##_p##pr##_a##ap##_b##bp))(                                    \
        const void* A, const void* B, const void* C, void* D, const float* sa, const float* sb, int M, void* stream) { \
        return g3b_launch<G3_N, rw, pr, 0, ap, bp>(A, B, C, D, sa, sb, M, stream);                                   \
    }
G3_VARP(1, 1, 2, 0)
G3_VARP(1, 1, 0, 2)
G3_VARP(1, 1, 1, 0)
#endif  // K1536_EXPERIMENT_VARIANTS

// x2lb_pass.cu -- X2L-B v0 (actckpt iteration 70, track x2lb/build). ONE persistent HIP kernel per FP8 preprocessor MLP
// k in {content, additional} (component E, gfx950 / MI350X) that fuses
//   (1) the L2 FP8 dgrad  da1[M,256] = RNE_bf16(acc*(dh_scale*w_scale) + 0.0f),  acc = sum_k dh8[t,k]*w8_t[n,k]
//       (dh8 e5m2 [M,512] row-major, w8_t e4m3 [256,512] row-major; == torch._scaled_mm(dh8, w8_t.t(), dh_scale,
//       w_scale, out_dtype=bf16) = hipBLASLt F8B8BS ..._SK3 bit for bit on every row outside its split stream-K tiles,
//       CONTRACT_dgrad.md 6.3), and
//   (2) PR #7's SwishLN256 backward ln_bwd_flydsl_d256_swish_dxsum_x16_c3 (op for op, CONTRACT_swish.md 2-3):
//       dy = da1 (bf16-rounded), x = h1 bf16 [M,256], mean1/rstd1 fp32 [M], w/b fp32 [256]
//       -> dh1 bf16 [M,256] + per-WG fp32 partial rows PART [3, G, 256] (dW, dB, dXS = db1), reduced by x2lb_fin.
// da1 never touches HBM (S1). Numerics class: fp32-order (da1 / dh1 bitwise targets; partials = a fixed fp32 order).
//
// Contract documents (read only): probes/iter70/x2lb/contract_dgrad/CONTRACT_dgrad.md ("[CD x]"),
//   probes/iter70/x2lb/contract_swish/CONTRACT_swish.md ("[CS x]"); design: probes/iter69/xl/design/DESIGN_X2L.md 3.1
//   and 7, corrections probes/iter69/xl/critique/CRITIQUE.md 2.3, 2.4, 3.1. Build notes: IMPLEMENTATION.md (this dir).
//
// Provenance / attribution:
//   * The FP8 dgrad helpers are copied VERBATIM (only the l11_/L11_ name prefixes changed to x2lb_/X2LB_) from the
//     shipped L11 kernel RT ops/hip/l11_outproj_lnmd.cu (sha256 543478a3...; RT = the run tree
//     proposals/ship15/x2h/rung/tree/recommendation/generative_recommenders): buffer resource (150-158), barrier macros
//     (161-169), ds_read16 / ds_write16 / frag (172-195), the operand-
//     swapped MFMA call of mma4 (222-238: src0 = weight fragment e4m3 cbsz 0, src1 = gradient fragment e5m2 blgp 1,
//     step 0 with an inline-0 C), g2l LDS-DMA (240-243), pk_bf16 / epi2 (245-254), bf_lo / bf_hi (256-257), dpp_add
//     (290-295), static_for (278-282), the scale prologue (396-407), the A-tile DMA swizzle kswz / voffA (412-416), the
//     fragment read offsets lp0 / lp1 (428-431) and the epilogue pin (579-593).
//   * L11 itself derives from r16 iter30/31 (L11-1a/1b), r14 iter37 track "od", P7 iter9 B7 G1-C (ops/hip/
//     k512_fp8_gemm.cu) and HipKittens (https://github.com/HazyResearch/HipKittens, MIT License, Copyright (c) 2024
//     HazyResearch): the MFMA operand register layout and the st_16x128 XOR LDS swizzle used here.
//   * Persistent chassis idioms (static tile sequence blockIdx.x + j*grid, explicit wait + barrier per phase with
//     compiler memory fences and sched_barriers, compiler-visible __builtin_amdgcn_s_waitcnt for vmcnt, C ABI with the
//     drain contract) follow X2H (RT ops/hip/x2h_pass.cu, sha256 d50b03f3...) and probes/iter68/x2h/facts/FACTS.md
//     (LDS-DMA wait rules b.4, b.6). Unlike X2H/L11 the loop is NOT unrolled by buffer parity (one body, runtime
//     A-buffer offset), so the loop-carried prefetch registers keep one register assignment.
//   * The consumer is PR #7's per-row sequence (RT ops/flydsl/ln_bwd.py:394-492, sha256 be4a626f...) written out in
//     HIP with the operand order and rounding points of [CS 2] and the row-sum tree of [CS 3].
//
// Kernels
//   x2lb_pass<S0, EMIT_DA1>   S0 = 0: the fused kernel (S1); EMIT_DA1 = 1 additionally stores the bf16 da1 tile to a
//                             global DA1 [M,256] (debug: the GPU bitwise gate G2 vs hipBLASLt). S0 = 1: diagnostic
//                             "chassis only" arm: no dh8 load, no MFMA, no epilogue; the consumer reads da1 rows from a
//                             global DA1 [M,256] instead of LDS; everything else (grid, tiles, barriers, LDS footprint,
//                             consumer, partials) identical.
//   x2lb_fin                  out[o][c] = sum_{g = 0..G-1} PART[o][g][c] in ascending g, fp32, from +0.
// C ABI (extern "C", bottom): x2lb_abi_version, x2lb_variant, x2lb_impl, x2lb_plan, x2lb_launch (S1),
//   x2lb_launch_emit (S1 + DA1 out), x2lb_launch_s0 (S0), x2lb_fin_launch, x2lb_drain_last_error.
//
// Build (production flags of L11's loader, RT ops/hip/l11_outproj_lnmd.py:124-140):
//   hipcc --offload-arch=gfx950 -DKITTENS_CDNA4 -std=c++20 -w -O3 -mllvm -amdgpu-sched-strategy=iterative-ilp
//         -ffp-contract=off -shared -fPIC   (x2lb.py; CPU census: build_cpu.sh)
//
// Fix pass (2026-10-12, task F; IMPLEMENTATION.md "fix pass", ../FIX_REPORT.md; pre-fix copy x2lb_pass.prefix.cu):
//   * X2LB_INDEX_ONLY=1 compiles sections 0-1 alone on the host (idx_harness.cpp: the drift check of the emulator's
//     mirror of the index functions, build_cpu.sh) [R12]; the device code is not affected by the guards.
//   * the partial-tile zero-fill guard is the index function zfill_lane (mirrored by emu E6) [race F1];
//   * a tile with 0 rows (the one-past-the-end prefetch / DMA) gets its tensor's tile-0 base [R9];
//   * lab-only variant knobs X2LB_G3_FMAEPI (contracted epilogue) and X2LB_PROBE_NOFILL (no zero fill: the FACTS f
//     probe) next to X2LB_G3_KPERM; all of them and the cache-policy knobs are reported by x2lb_impl() /
//     x2lb_variant() [contract F4, race F1, R5];
//   * the pass checks grid == part_rows (PART is exactly [3, grid, 256]); ABI version 2026101202 [R6];
//   * comments only: F6 (hand-written LDS waits must stay lgkmcnt(0)), O4 / R14 (padded-row and zero-fill wording).
#ifndef X2LB_INDEX_ONLY
#define X2LB_INDEX_ONLY 0
#endif
#if !X2LB_INDEX_ONLY
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>

#include <utility>

#if !X2LB_INDEX_ONLY
namespace x2lb {
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}  // namespace x2lb

// LLVM buffer intrinsics with the v4i32 resource (L11:88-102, names prefixed x2lb)
extern "C" __device__ void x2lb_llvm_buffer_load_lds(x2lb::i32x4 rsrc, x2lb::lds_ptr_t lds, int size, int voffset,
                                                     int soffset, int offset, int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
extern "C" __device__ void x2lb_llvm_buffer_store_v4i32(x2lb::i32x4 vdata, x2lb::i32x4 rsrc, int voffset, int soffset,
                                                        int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");
extern "C" __device__ x2lb::i32x4 x2lb_llvm_buffer_load_v4i32(x2lb::i32x4 rsrc, int voffset, int soffset,
                                                             int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
extern "C" __device__ float x2lb_llvm_buffer_load_f32(x2lb::i32x4 rsrc, int voffset, int soffset,
                                                      int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");
#endif  // !X2LB_INDEX_ONLY

// Cache policies of the streamed tensors (aux bits of the buffer instructions on gfx950: 0 default, 2 nt, 3 sc0 nt).
// Defaults = PR #7's ln_bwd_flydsl_d256_swish_dxsum_x16_c3 (CACHE 3 / STORE_CACHE 3: loads of x, dy and the dx store
// `sc0 nt`, [CS 1] "options"); the dh8 LDS-DMA keeps L11's 0. Lab knobs only (no numerical effect).
#ifndef X2LB_LD_CP
#define X2LB_LD_CP 3
#endif
#ifndef X2LB_ST_CP
#define X2LB_ST_CP 3
#endif
#ifndef X2LB_DMA_CP
#define X2LB_DMA_CP 0
#endif
// G3 negative control of the GPU gate (DESIGN_X2L 7 stage B, [CD 8] "negative controls"): X2LB_G3_KPERM=1 builds the
// SAME kernel with the K steps of every MFMA chain in the order 0, 2, 1, 3. It must MISMATCH hipBLASLt's da1 on
// exact-cancellation data (and isa_check C3 must reject it). Never set it outside that lab arm.
#ifndef X2LB_G3_KPERM
#define X2LB_G3_KPERM 0
#endif
// Second G3 negative control ([CD 8]; contract review F4): X2LB_G3_FMAEPI=1 contracts the epilogue to
// fma(acc, s, +0). It differs from hipBLASLt's mul-then-add only where acc*s is negative and underflows to zero
// (fma keeps -0 -> bf16 0x8000; mul gives -0, then -0 + +0 = +0 -> 0x0000), so the lab arm needs the underflow vector
// of x2lb.py fmaepi_vector(). isa_check C4 / C15 must reject it. Never set it outside that lab arm.
#ifndef X2LB_G3_FMAEPI
#define X2LB_G3_FMAEPI 0
#endif
// Lab probe of FACTS (f) (race review F1): X2LB_PROBE_NOFILL=1 disables the zero fill of partial tiles (zfill_lane is
// false). That kernel is correct only if an out-of-range LDS-DMA lane writes zeros: the G2c informational arm compares
// its dXS with the production build's at a partial-tile M. isa_check C16 must reject it. Never set it outside that arm.
#ifndef X2LB_PROBE_NOFILL
#define X2LB_PROBE_NOFILL 0
#endif

namespace x2lb {

// ---------------------------------------------------------------------------------------------------------------------
// 0. Geometry
// ---------------------------------------------------------------------------------------------------------------------
constexpr int KD = 512;           // dgrad K = bytes per e5m2 dh8 row
constexpr int NOUT = 256;         // da1 / h1 / dh1 width (bf16 elements)
constexpr int BM = 64;            // rows per tile
constexpr int NWAVE = 8;          // 512 threads = 8 wave64, 1 WG per CU (LDS > 80 KiB), 2 waves per SIMD
constexpr int NTHREADS = 512;
constexpr int NSLOT = 4;          // consumer slots per wave per tile: 64 rows / (8 waves x 2 rows)
constexpr int ABYTES = 32768;     // A tile: 4 K steps x 64 rows x 128 B (L11 layout, st_16x128 XOR swizzle)
constexpr int RBASE = 2 * ABYTES; // da1 redistribution buffer after the two A buffers
constexpr int RPITCH = 528;       // R row pitch: 512 B of bf16 + 16 B (4-bank shift per row)
constexpr int RBYTES = BM * RPITCH;              // 33,792 B
constexpr int LDS_BYTES = RBASE + RBYTES;        // 99,328 B (S0 allocates the same: identical occupancy)
constexpr uint32_t ROWB = 512;    // bytes per bf16 row of h1 / da1 / dh1 (and per e5m2 row of dh8)
constexpr uint32_t SLOT_STRIDE = 16 * ROWB;      // global byte step between consumer slots (16 rows)
constexpr int RED_FLOATS = 3 * 2 * NWAVE * NOUT; // end-of-kernel partial staging [3][16][256] fp32 = 49,152 B
static_assert(RED_FLOATS * 4 <= RBASE, "partial staging fits the (then free) A buffers");
static_assert(LDS_BYTES <= 163840, "LDS budget");

// s_waitcnt counts at the top of an iteration (S1): VMEM instructions this wave issues AFTER its LDS-DMA of the next
// tile's A in the previous iteration = (EMIT_DA1: 4 da1 stores) + 2 mean/rstd prefetches + 4 dh1 stores + 4 h1
// prefetches. vmcnt(N) with N = that count retires the DMA (VM_CNT is in order [FACTS b.4]) and leaves the stores and
// prefetches in flight. isa_check.py C13 re-derives this on the compiled CFG.
template <int EMIT>
constexpr int top_vmcnt() { return (EMIT ? 4 : 0) + NSLOT + NSLOT + 2; }

// ---------------------------------------------------------------------------------------------------------------------
// 1. Index functions (mirrored 1:1 by emu/x2lb_emu.py; every one is a clause of the design, see IMPLEMENTATION.md 2).
//    build_cpu.sh compiles this section on the host (X2LB_INDEX_ONLY, idx_harness.cpp) and diffs its table against the
//    mirror's (emu/x2lb_emu.py --dump-index): a drift between the two fails the CPU gate.
// ---------------------------------------------------------------------------------------------------------------------
#if X2LB_INDEX_ONLY
#define X2LB_FN inline constexpr
#else
#define X2LB_FN __host__ __device__ __forceinline__ constexpr
#endif
// tiles: tile t = rows [64t, 64t + 64) clipped to M; WG b runs t = b, b + G, b + 2G, ... (static, no counter)
X2LB_FN int static_tile(int b, int j, int G) { return b + j * G; }
X2LB_FN int tile_rows(int t, int M) { return M - t * BM < 0 ? 0 : (M - t * BM > BM ? BM : M - t * BM); }
// A tile LDS-DMA (L11:412-416, 439-442): instruction i (= K step i) of wave w writes LDS bytes
//   buf*ABYTES + i*8192 + w*1024 + 16*lane (hardware adds 16*lane); lane loads tile row rr = 8w + lane/8, global bytes
//   [128i + kswz, +16) of that row with kswz = 16 * (p ^ ((rr >> 1) & 7)), p = lane & 7.
X2LB_FN int a_dma_row(int w, int lane) { return 8 * w + (lane >> 3); }
X2LB_FN uint32_t a_dma_kswz(int w, int lane) {
    return (uint32_t)(16 * ((lane & 7) ^ ((a_dma_row(w, lane) >> 1) & 7)));
}
X2LB_FN uint32_t a_dma_voff(int w, int lane) { return (uint32_t)a_dma_row(w, lane) * KD + a_dma_kswz(w, lane); }
X2LB_FN uint32_t a_dma_lds(int buf, int w, int i) { return (uint32_t)(buf * ABYTES + i * 8192 + w * 1024); }
X2LB_FN uint32_t a_dma_soff(int i) { return (uint32_t)(i * 128); }
// zero fill of a partial tile (0 < rows < 64): lane (w, lane) ds_writes zeros over the 4 x 16 B its own DMA targets iff
//   its DMA row a_dma_row(w, lane) is >= rows (out of range of the per-tile resource). Full tiles (rows = 64) and the
//   one-past-the-end tile (rows = 0, never read) are not filled. Used by issue_A; emu E6 and isa_check C16 check it.
X2LB_FN bool zfill_lane(int rows, int w, int lane) {
    return !X2LB_PROBE_NOFILL && rows > 0 && rows < BM && a_dma_row(w, lane) >= rows;
}
// tile whose 64-bit base a per-tile buffer resource uses: the tile itself, or tile 0 of the same tensor when the tile
//   has no valid row (rows = 0: the one-past-the-end prefetch / DMA tile, num_records 0, every lane out of range).
//   The base then always lies inside the tensor [R9].
X2LB_FN int rsrc_tile(int t, int rows) { return rows > 0 ? t : 0; }
// fragment reads (L11:428-431): lane (q4 = lane/16, r16 = lane%16) reads LDS row 16m + r16 of K step k at
//   lp0 + buf*ABYTES + k*8192 + m*2048 (bytes 0..15) and lp1 + ... (bytes 16..31)
X2LB_FN uint32_t frag_lp0(int lane) {
    return (uint32_t)(lane & 15) * 128u + (((uint32_t)(16 * (lane >> 4))) ^ ((uint32_t)((lane & 15) >> 1) << 4));
}
X2LB_FN uint32_t frag_lp1(int lane) {
    return (uint32_t)(lane & 15) * 128u + (((uint32_t)(16 * (lane >> 4) + 64)) ^ ((uint32_t)((lane & 15) >> 1) << 4));
}
X2LB_FN int frag_off(int buf, int k, int m) { return buf * ABYTES + k * 8192 + m * 2048; }
// resident W fragment (wave w, column block n, K step k, half h = bytes 0..15 / 16..31) of lane (q4, r16): output
//   column w_col = 32w + 8(r16>>2) + 4n + (r16&3) (L11's od STORE=1 row order, L11:417-418), bytes
//   w8_t[w_col][128k + 64h + 16 q4, +16) [CD 4, 6.2]
X2LB_FN int w_col(int w, int n, int lane) { return 32 * w + 8 * ((lane & 15) >> 2) + 4 * n + (lane & 3); }
X2LB_FN uint32_t w_voff(int w, int lane) { return (uint32_t)w_col(w, 0, lane) * KD + (uint32_t)(16 * (lane >> 4)); }
X2LB_FN uint32_t w_imm(int n, int k, int h) { return (uint32_t)(4 * n * KD + 128 * k + 64 * h); }
// MFMA result: acc[n] element i of lane (q4, r16) in row group m = da1[16m + r16][32w + 8q4 + 4n + i]
X2LB_FN int d_row(int m, int lane) { return 16 * m + (lane & 15); }
X2LB_FN int d_col(int w, int n, int i, int lane) { return 32 * w + 8 * (lane >> 4) + 4 * n + i; }
// R layout: tile row r, column c (bf16) at RBASE + r*RPITCH + 2c. Producer write of pk (8 bf16 = cols 32w+8q4..+8 of
//   row 16m + r16): rw_base + m*16*RPITCH. Consumer read (row c_row, cols 8*(lane%32)..+8): rr_base + u*16*RPITCH.
X2LB_FN uint32_t rw_base(int w, int lane) {
    return (uint32_t)(RBASE + (lane & 15) * RPITCH + 64 * w + 16 * (lane >> 4));
}
X2LB_FN int c_row(int w, int u, int lane) { return 16 * u + 2 * w + (lane >> 5); }  // consumer row of slot u
X2LB_FN int c_col0(int lane) { return 8 * (lane & 31); }                              // first of the lane's 8 columns
X2LB_FN uint32_t rr_base(int w, int lane) {
    return (uint32_t)(RBASE + (2 * w + (lane >> 5)) * RPITCH + 2 * c_col0(lane));
}
// global byte offset (within the tile) of the lane's 16 B in consumer slot u: row c_row, cols c_col0..+8
X2LB_FN uint32_t c_voff(int w, int u, int lane) { return (uint32_t)c_row(w, u, lane) * ROWB + 2u * (uint32_t)c_col0(lane); }
// EMIT_DA1 store of pk (row 16m + r16, cols 32w + 8q4 .. +8)
X2LB_FN uint32_t e_voff(int w, int m, int lane) {
    return (uint32_t)d_row(m, lane) * ROWB + 2u * (uint32_t)(32 * w + 8 * (lane >> 4));
}
// partial slot of a lane accumulator (16 per WG) and its staging address [o][slot][c]
X2LB_FN int acc_slot(int w, int lane) { return 2 * w + (lane >> 5); }
X2LB_FN int red_index(int o, int slot, int c) { return (o * 2 * NWAVE + slot) * NOUT + c; }

#if !X2LB_INDEX_ONLY
// ---------------------------------------------------------------------------------------------------------------------
// 2. Device building blocks
// ---------------------------------------------------------------------------------------------------------------------
// L11:150-158 (od / K512:75-83)
__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}

// L11:161-169 (od / K512:85-93)
#define X2LB_FENCE() asm volatile("" ::: "memory")
#define X2LB_SB() __builtin_amdgcn_sched_barrier(0)
#define X2LB_BARRIER()                \
    do {                              \
        X2LB_FENCE();                 \
        __builtin_amdgcn_s_barrier(); \
        X2LB_FENCE();                 \
        X2LB_SB();                    \
    } while (0)

// L11:172-195 (od / K512:95-133)
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

// Waits. L11's wait_vm / wait_lgkm (L11:205-220) are inline asm; X2L-B keeps inline-asm lgkmcnt(0) waits only where
// they are TIED to the registers of inline-asm LDS reads ("+v" operands, below), and issues every vmcnt wait (and the
// barrier waits) as vmcnt waits that the COMPILER must see (X2H idiom, x2h_pass.cu:57-68; FACTS b.6): __builtin_amdgcn_s_waitcnt with the
// gfx9 simm16 (vmcnt[3:0] = [3:0], expcnt [6:4] = 7, lgkmcnt [11:8], vmcnt[5:4] = [15:14]) between memory fences.
// SIInsertWaitcnts only ever tightens such a "hard" wait, and it accounts for it: an inline-asm vmcnt wait is invisible
// to it, so after the prologue's inline drain it assumed the prologue loads were still in flight and forced
// vmcnt(0)/vmcnt(1) on the loop-carried prefetch registers in every iteration (first R720 build).
constexpr int wenc(int vm, int lgkm) { return (vm & 15) | ((vm >> 4) << 14) | (7 << 4) | (lgkm << 8); }
template <int VM, int LGKM>
__device__ __forceinline__ void waitcnt() {
    static_assert(VM >= 0 && VM <= 63 && LGKM >= 0 && LGKM <= 15, "simm16 fields");
    X2LB_SB();
    X2LB_FENCE();
    __builtin_amdgcn_s_waitcnt(wenc(VM, LGKM));
    X2LB_FENCE();
    X2LB_SB();  // nothing is scheduled between this wait and the s_barrier that follows it (isa_check C9)
}
// top-of-iteration wait: vmcnt(N) and lgkmcnt(0) in ONE instruction, so that the s_barrier that follows is immediately
// preceded by an lgkmcnt(0) wait (isa_check C9)
template <int N>
__device__ __forceinline__ void wait_vm_lgkm0() { waitcnt<N, 0>(); }

// The operand-swapped FP8 MFMA of L11's mma4 (L11:222-238), for one 16-row group: src0 = weight fragment (e4m3,
// cbsz = 0), src1 = gradient fragment (e5m2, blgp = 1), K step 0 with an inline-0 C, chained in place. acc[n] (+)=
// Wfrag[n] (16 output cols) x Gfrag (16 output rows); lane (q4, r16) holds 4 consecutive output cols of row r16.
// The builtin call and its arguments are L11's verbatim; L11 loops m over 2 row groups per call, X2L-B over 1.
template <bool FIRST>
__device__ __forceinline__ void mma2(f32x4 (&c)[2], const i32x8 (&b)[2], const i32x8& a) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        if constexpr (FIRST) {
            const f32x4 z = {0.f, 0.f, 0.f, 0.f};
            c[n] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[n], a, z, 0, 1, 0, 0, 0, 0);
        } else {
            c[n] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[n], a, c[n], 0, 1, 0, 0, 0, 0);
        }
    }
}

// L11:240-243
__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    x2lb_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, X2LB_DMA_CP);
}

// L11:245-254
__device__ __forceinline__ uint32_t pk_bf16(f32x2 v) {  // RNE fp32 -> bf16 pair (gfx950: v_cvt_pk_bf16_f32)
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2));
}
// od EPI2: RNE_bf16(acc*(sa*sb) + 0.0f) (fp-contract off; +0.0f not folded without nsz) -> hipBLASLt's sign of zero
__device__ __forceinline__ uint32_t epi2(f32x2 v, f32x2 s2, f32x2 z2) {
#if X2LB_G3_FMAEPI
    v = __builtin_elementwise_fma(v, s2, z2);  // G3 negative control only: one rounding (see X2LB_G3_FMAEPI)
#else
    v = v * s2;
    v = v + z2;
#endif
    return pk_bf16(v);
}

// L11:256-257
__device__ __forceinline__ float bf_lo(uint32_t u) { return __builtin_bit_cast(float, u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __builtin_bit_cast(float, u & 0xffff0000u); }
// a packed bf16 pair (low half = even column) -> exact fp32 pair
__device__ __forceinline__ f32x2 bf2(int d) { return f32x2{bf_lo((uint32_t)d), bf_hi((uint32_t)d)}; }

// L11:290-295: v_add_f32_dpp v, v, v <ctrl> row_mask:0xf bank_mask:0xf bound_ctrl:1 == dpp(v) + v
template <int CTRL>
__device__ __forceinline__ float dpp_add(float v) {
    const int s = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, true);
    return __builtin_bit_cast(float, s) + v;
}

// L11:278-282
template <int N, typename F>
__device__ __forceinline__ void static_for(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(
        std::make_integer_sequence<int, N>{});
}

// ---- PR #7 consumer pieces [CS 2, 3] ----
// fused multiply-add of two independent fp32 lanes (llvm.fma.v2f32: one rounding per lane; v_pk_fma_f32 on gfx950)
__device__ __forceinline__ f32x2 fma2(f32x2 a, f32x2 b, f32x2 c) { return __builtin_elementwise_fma(a, b, c); }
// R1/R2 in-lane fold [CS 3.2]: lane columns v0..v7 held as p0 = (v0,v1), p1 = (v2,v3), p2 = (v4,v5), p3 = (v6,v7);
//   H = ((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))  (ln_bwd.py:329-335 hsum8: halves 8 -> 4 -> 2 -> 1)
__device__ __forceinline__ float hsum8(f32x2 p0, f32x2 p1, f32x2 p2, f32x2 p3) {
    const f32x2 h4a = p0 + p2;   // (v0+v4, v1+v5)
    const f32x2 h4b = p1 + p3;   // (v2+v6, v3+v7)
    const f32x2 h2 = h4a + h4b;  // ((v0+v4)+(v2+v6), (v1+v5)+(v3+v7))
    return h2.x + h2.y;
}
// R1/R2 across the row's 32 lanes [CS 3.3] (ln_bwd.py:231-259 _xor_sum(width 32, dpp)): pair distances 1, 2, 4, 8 by
//   DPP quad_perm [1,0,3,2], quad_perm [2,3,0,1], row_half_mirror, row_mirror; 16 by ds_swizzle SWAP16 (0x401F).
//   Every lane of the 32-lane half ends with the same bits = the xor butterfly 1, 2, 4, 8, 16.
__device__ __forceinline__ float row_tree32(float s) {
    s = dpp_add<0xB1>(s);
    s = dpp_add<0x4E>(s);
    s = dpp_add<0x141>(s);
    s = dpp_add<0x140>(s);
    const int o = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x401F);
    return s + __builtin_bit_cast(float, o);
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Kernel arguments
// ---------------------------------------------------------------------------------------------------------------------
struct Params {
    const uint8_t* dh8;   // e5m2 [M, 512] row-major (S1)
    const uint8_t* w8t;   // e4m3 [256, 512] row-major: the storage of the w8_t.t() operand _scaled_mm consumes (S1)
    const float* sa;      // dh_scale fp32 [1] (scale_a of the dgrad _scaled_mm) (S1)
    const float* sb;      // w_scale fp32 [1] (scale_b) (S1)
    const uint16_t* h1;   // bf16 [M, 256] (x of the SwishLN backward)
    const float* mean1;   // fp32 [M]
    const float* rstd1;   // fp32 [M]
    const float* nw;      // fp32 [256] SwishLN norm weight
    const float* nb;      // fp32 [256] SwishLN norm bias
    uint16_t* dh1;        // bf16 [M, 256] out
    uint16_t* da1;        // bf16 [M, 256]: out (EMIT_DA1) / in (S0) / unused
    float* part;          // fp32 [3, G, 256] out: one row per WG per output (dW, dB, dXS)
    int M;
    int ntiles;           // ceil(M / 64)
};

// ---------------------------------------------------------------------------------------------------------------------
// 4. The kernel
// ---------------------------------------------------------------------------------------------------------------------
template <int S0, int EMIT>
__global__ __launch_bounds__(512) void x2lb_pass(const Params p) {
#pragma clang fp contract(off)
    static_assert(!(S0 && EMIT), "S0 has no da1 to emit");
    __shared__ __attribute__((aligned(2048))) uint8_t smem[LDS_BYTES];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int G = (int)gridDim.x;
    const int B = (int)blockIdx.x;
    const int M = p.M;
    const int ntiles = p.ntiles;

    // ---- scale s = fl32(dh_scale * w_scale), once (L11:396-407; [CD 0] "one IEEE multiply per call"); S0 reads none
    float s = 0.f;
    if constexpr (!S0) {
        const float sav = p.sa[0], sbv = p.sb[0];
        s = sav * sbv;
        asm volatile("" : "+v"(s));  // materialize s (and so retire the scale SMEM loads) before any hand-counted LDS op
    }
    // Retires the SMEM issued so far. It does NOT keep SMEM out of flight later: the compiler may still issue kernarg
    // s_load after this point (the production-flag build issues the nb/dh1 s_load_dwordx4 after it; race review F6),
    // and SMEM returns out of order. Hence the rule: every hand-written LDS wait of this kernel is lgkmcnt(0) (never
    // lgkmcnt(N > 0), e.g. L11's wait_lgkm<4>); isa_check C18 rejects any inline-asm s_waitcnt other than lgkmcnt(0).
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    const f32x2 s2 = {s, s}, z2 = {0.f, 0.f};

    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);
    // per-lane LDS addresses (all loop invariant, kept in VGPRs)
    const uint32_t voffA = a_dma_voff(w, lane);
    const uint32_t ldsw = lds0 + a_dma_lds(0, w, 0);            // wave's DMA block of buffer 0, K step 0 (SGPR)
    const uint32_t ldsz = ldsw + 16u * (uint32_t)lane;          // the same block, this lane's 16 B (zero fill)
    const uint32_t lp0 = lds0 + frag_lp0(lane);
    const uint32_t lp1 = lds0 + frag_lp1(lane);
    const uint32_t wR = lds0 + rw_base(w, lane);
    const uint32_t rR = lds0 + rr_base(w, lane);
    const uint32_t vC = c_voff(w, 0, lane);                     // + u * SLOT_STRIDE per slot (full offset in voffset)
    const uint32_t vE = e_voff(w, 0, lane);                     // + m * SLOT_STRIDE per row group (EMIT_DA1)
    const bool hi_half = lane >= 32;

    // ---- resident W fragments: wave w, column blocks n = 0, 1, K steps k = 0..3: 8 x i32x8 = 64 VGPR for the
    // kernel's lifetime, gathered once per WG straight from w8_t (no prepack) [CD 6.2]
    i32x8 wf[2][4];
    if constexpr (!S0) {
        const i32x4 rW = make_rsrc(p.w8t, (uint32_t)(NOUT * KD));
        const uint32_t vw = w_voff(w, lane);
        static_for<2>([&](auto N_) {
            constexpr int n = decltype(N_)::value;
            static_for<4>([&](auto K_) {
                constexpr int k = decltype(K_)::value;
                const i32x4 lo = x2lb_llvm_buffer_load_v4i32(rW, (int)(vw + w_imm(n, k, 0)), 0, 0);
                const i32x4 hi = x2lb_llvm_buffer_load_v4i32(rW, (int)(vw + w_imm(n, k, 1)), 0, 0);
                wf[n][k] = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            });
        });
    }

    // ---- consumer constants of this lane's 8 columns c0 .. c0+7 (pairs (2i, 2i+1)): w, w2 = RN(w*NL2E),
    // b2 = RN(b*NL2E), NL2E = fp32(-log2 e) = 0xBFB8AA3B [CS 2 C1, C2]
    f32x2 wv[4], w2v[4], b2v[4];
    {
        const int c0 = c_col0(lane);
        const f32x4 wa = *reinterpret_cast<const f32x4*>(p.nw + c0);
        const f32x4 wb = *reinterpret_cast<const f32x4*>(p.nw + c0 + 4);
        const f32x4 ba = *reinterpret_cast<const f32x4*>(p.nb + c0);
        const f32x4 bb = *reinterpret_cast<const f32x4*>(p.nb + c0 + 4);
        const f32x2 nl2e = {__builtin_bit_cast(float, 0xBFB8AA3Bu), __builtin_bit_cast(float, 0xBFB8AA3Bu)};
        wv[0] = f32x2{wa.x, wa.y};
        wv[1] = f32x2{wa.z, wa.w};
        wv[2] = f32x2{wb.x, wb.y};
        wv[3] = f32x2{wb.z, wb.w};
        const f32x2 bv[4] = {f32x2{ba.x, ba.y}, f32x2{ba.z, ba.w}, f32x2{bb.x, bb.y}, f32x2{bb.z, bb.w}};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w2v[i] = wv[i] * nl2e;
            b2v[i] = bv[i] * nl2e;
        }
    }
    // lane accumulators (dW, dB, dXS) of the lane's 8 columns, slot acc_slot(w, lane), from +0 [CS 4.1]
    f32x2 adw[4], adb[4], adx[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        adw[i] = f32x2{0.f, 0.f};
        adb[i] = f32x2{0.f, 0.f};
        adx[i] = f32x2{0.f, 0.f};
    }

    // ---- consumer prefetch registers (one tile ahead, rotated per slot): h1 (and S0: da1) 16 B per slot, the tile's
    // 64 mean / rstd values (lane l = tile row l; a slot's two rows are read back with v_readlane)
    i32x4 hx[NSLOT], dv[S0 ? NSLOT : 1];
    float mv, rv;
    // valid rows of tile t, wave-uniform in an SGPR (the clamp of tile_rows selects to a VALU v_med3_i32; without the
    // readfirstlane every buffer instruction on a resource built from it becomes a readfirstlane waterfall loop)
    auto rows_of = [&](int t) { return __builtin_amdgcn_readfirstlane(tile_rows(t, M)); };
    // resource of tile t: 64-bit base per tile, num_records = valid rows * row bytes (rows >= M read 0 / are dropped;
    // the whole row offset is passed in voffset, which the range check covers, never in soffset). A tile with no valid
    // row (the one-past-the-end prefetch / DMA tile) uses the tensor's tile-0 base (rsrc_tile) [R9].
    auto rs_rows = [&](const void* base, int t, int rows, uint32_t rowb) {
        return make_rsrc(reinterpret_cast<const uint8_t*>(base) + (size_t)rsrc_tile(t, rows) * BM * rowb,
                         (uint32_t)rows * rowb);
    };
    auto prefetch_slot = [&](int t, int rows, auto U_) {  // h1 (+ S0: da1) of slot u of tile t
        constexpr int u = decltype(U_)::value;
        hx[u] = x2lb_llvm_buffer_load_v4i32(rs_rows(p.h1, t, rows, ROWB), (int)(vC + (uint32_t)u * SLOT_STRIDE), 0,
                                            X2LB_LD_CP);
        if constexpr (S0)
            dv[u] = x2lb_llvm_buffer_load_v4i32(rs_rows(p.da1, t, rows, ROWB), (int)(vC + (uint32_t)u * SLOT_STRIDE),
                                                0, X2LB_LD_CP);
    };
    auto prefetch_stats = [&](int t, int rows) {  // mean / rstd of tile t: lane l <- row 64t + l
        mv = x2lb_llvm_buffer_load_f32(rs_rows(p.mean1, t, rows, 4u), lane * 4, 0, 0);
        rv = x2lb_llvm_buffer_load_f32(rs_rows(p.rstd1, t, rows, 4u), lane * 4, 0, 0);
    };

    // ---- A tile of tile t (S1): LDS-DMA with L11's swizzle into the A buffer at byte offset bofs (0 / ABYTES).
    // The 4 DMA instructions are ALWAYS issued by every lane (also for the one-past-the-end tile, rows = 0, whose
    // buffer is never read): the VMEM count after them is then the same on every path, which the waits rely on
    // (isa_check C13; a conditional DMA made the compiler's own waits assume the shorter path and wait on the DMA).
    // Rows >= M are out of range of the per-tile resource. Partial last tile (0 < rows < 64): the lanes of rows >= M
    // (zfill_lane) also write zeros to the same LDS bytes with ds_write_b128, so the MFMA sees zeros. Premise: an
    // out-of-range LDS-DMA lane writes NOTHING or ZEROS to its LDS bytes (FACTS f: which one is unsure; the ISA rule
    // that out-of-range buffer loads return 0 makes "garbage" implausible); under that premise both writers can only
    // leave zeros, so their order is immaterial (R14). Then da1 = fl(+-0 * s) + 0 = +0 for every padded row, and with
    // x = mean = rstd = 0 the padded rows contribute exactly +0 to the partials [CS 4.1], PROVIDED s is finite and the
    // W column holds no NaN e4m3 byte; otherwise that column is already NaN / non-finite from the real rows, so no
    // observable output changes (emulator O4).
    auto issue_A = [&](int t, int rows, uint32_t bofs) {
        const i32x4 rA = rs_rows(p.dh8, t, rows, (uint32_t)KD);
        const uint32_t ldsb = ldsw + bofs;
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, ldsb + (uint32_t)(i * 8192), voffA, a_dma_soff(i));
        if (zfill_lane(rows, w, lane)) {
            const i32x4 zq = {0, 0, 0, 0};
            const uint32_t za = ldsz + bofs;
            static_for<4>([&](auto I_) {
                constexpr int i = decltype(I_)::value;
                ds_write16<i * 8192>(za, zq);
            });
        }
    };

    // ---- prologue: W, norm params, the first tile's consumer data and A tile, then drain
    int t = static_tile(B, 0, G);
    {
        const int rows = rows_of(t);
        static_for<NSLOT>([&](auto U_) { prefetch_slot(t, rows, U_); });
        prefetch_stats(t, rows);
        if constexpr (!S0) issue_A(t, rows, 0u);
    }
    waitcnt<0, 15>();  // compiler-visible vmcnt(0): W, norm params, prefetches and the first A tile landed
    X2LB_SB();

    // One iteration = one tile t of this WG; bofs = byte offset of its A buffer (alternates 0, ABYTES). Phases:
    //   (1) wait: A(t) landed in every wave (vmcnt(N) + lgkmcnt(0) + s_barrier); R and the other A buffer are free
    //   (2) LDS-DMA A(t + G) -> the other buffer (prefetch, overlaps (3)-(5))
    //   (3) MFMA phase: 4 row groups m x (8 frag reads, 2 chains of 4 MFMAs, EPI2, ds_write of the bf16 da1 rows to R,
    //       EMIT_DA1: store); the frags of m+1 are read while the MFMAs of m run
    //   (4) lgkmcnt(0) + s_barrier: R complete
    //   (5) consumer: the 4 slots' da1 rows from R and the wave's 8 row statistics (v_readlane -> SGPRs), after which
    //       the mean/rstd registers are refilled with tile t+G's; then 4 slots of 2 rows per wave (PR #7's layout and
    //       math), dh1 store, partials; after each slot its h1 (S0: da1) registers are refilled with tile t+G's.
    // ONE loop body (no unroll by parity): the loop-carried prefetch registers then keep one physical assignment (an
    // unrolled pair made the register allocator copy them across the halves, i.e. wait for the prefetches).
    uint32_t bofs = 0;
    for (; t < ntiles; t += G) {
        const int tn = t + G;  // static_tile(B, j + 1, G)
        const int rows = rows_of(t), rows_n = rows_of(tn);
        // (1)
        if constexpr (!S0) wait_vm_lgkm0<top_vmcnt<EMIT>()>();
        else waitcnt<63, 0>();  // lgkmcnt(0) only (vmcnt field 63 = no wait)
        X2LB_BARRIER();
        // (2)
        if constexpr (!S0) issue_A(tn, rows_n, bofs ^ (uint32_t)ABYTES);
        X2LB_SB();
        const i32x4 rDH1 = rs_rows(p.dh1, t, rows, ROWB);
        // (3)
        if constexpr (!S0) {
            const i32x4 rDA1 = rs_rows(p.da1, t, rows, ROWB);
            const uint32_t la0 = lp0 + bofs, la1 = lp1 + bofs;
            i32x8 af[2][4];  // A fragments of a row group, K steps 0..3, double buffered across row groups
            static_for<4>([&](auto K_) {
                constexpr int k = decltype(K_)::value;
                af[0][k] = frag<frag_off(0, k, 0)>(la0, la1);
            });
            asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(af[0][0]), "+v"(af[0][1]), "+v"(af[0][2]), "+v"(af[0][3]));
            X2LB_SB();
            static_for<4>([&](auto M_) {
                constexpr int m = decltype(M_)::value;
                constexpr int cur = m & 1;
                if constexpr (m < 3) {  // next row group's fragments, in flight during this group's MFMAs
                    static_for<4>([&](auto K_) {
                        constexpr int k = decltype(K_)::value;
                        af[cur ^ 1][k] = frag<frag_off(0, k, m + 1)>(la0, la1);
                    });
                }
                X2LB_SB();
                const i32x8 bw0[2] = {wf[0][0], wf[1][0]};
                const i32x8 bw1[2] = {wf[0][1], wf[1][1]};
                const i32x8 bw2[2] = {wf[0][2], wf[1][2]};
                const i32x8 bw3[2] = {wf[0][3], wf[1][3]};
                f32x4 acc[2];
                mma2<true>(acc, bw0, af[cur][0]);   // K step 0, inline-0 C
                if constexpr (!X2LB_G3_KPERM) {
                    mma2<false>(acc, bw1, af[cur][1]);  // K steps 1, 2, 3 in order, in place [CD 0]
                    mma2<false>(acc, bw2, af[cur][2]);
                } else {
                    mma2<false>(acc, bw2, af[cur][2]);  // G3 negative control only: K order 0, 2, 1, 3
                    mma2<false>(acc, bw1, af[cur][1]);
                }
                mma2<false>(acc, bw3, af[cur][3]);
                // EPI2 (L11:579-593): 8 consecutive bf16 of row 16m + r16, columns 32w + 8q4 .. +8
                i32x4 q;
                q.x = (int)epi2(__builtin_shufflevector(acc[0], acc[0], 0, 1), s2, z2);
                q.y = (int)epi2(__builtin_shufflevector(acc[0], acc[0], 2, 3), s2, z2);
                q.z = (int)epi2(__builtin_shufflevector(acc[1], acc[1], 0, 1), s2, z2);
                q.w = (int)epi2(__builtin_shufflevector(acc[1], acc[1], 2, 3), s2, z2);
                asm volatile("" : "+v"(q));  // pin the packed tuple here (L11:587-590)
                ds_write16<m * 16 * RPITCH>(wR, q);
                if constexpr (EMIT)
                    x2lb_llvm_buffer_store_v4i32(q, rDA1, (int)(vE + (uint32_t)m * SLOT_STRIDE), 0, X2LB_ST_CP);
                X2LB_SB();
                if constexpr (m < 3) {  // next group's fragments landed (and this group's R write retired)
                    asm volatile("s_waitcnt lgkmcnt(0)"
                                 : "+v"(af[cur ^ 1][0]), "+v"(af[cur ^ 1][1]), "+v"(af[cur ^ 1][2]),
                                   "+v"(af[cur ^ 1][3]));
                    X2LB_SB();
                }
            });
        }
        // (4)
        waitcnt<63, 0>();
        X2LB_BARRIER();
        // (5) consumer
        i32x4 dq[NSLOT];
        if constexpr (!S0) {
            static_for<NSLOT>([&](auto U_) {
                constexpr int u = decltype(U_)::value;
                dq[u] = ds_read16<u * 16 * RPITCH>(rR);
            });
            asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(dq[0]), "+v"(dq[1]), "+v"(dq[2]), "+v"(dq[3]));
        } else {
            static_for<NSLOT>([&](auto U_) { dq[decltype(U_)::value] = dv[decltype(U_)::value]; });
        }
        // the statistics of this wave's 8 rows (slot u: tile rows 16u + 2w and +1 = c_row(w, u, 0 / 1)) -> SGPRs, then
        // the mean / rstd registers are refilled with tile t+G's (a whole iteration ahead of their use)
        float smv[NSLOT][2], srv[NSLOT][2];
        static_for<NSLOT>([&](auto U_) {
            constexpr int u = decltype(U_)::value;
            const int rl0 = 16 * u + 2 * w;
            smv[u][0] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, mv), rl0));
            smv[u][1] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, mv), rl0 + 1));
            srv[u][0] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, rv), rl0));
            srv[u][1] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, rv), rl0 + 1));
        });
        X2LB_SB();
        prefetch_stats(tn, rows_n);
        X2LB_SB();
        static_for<NSLOT>([&](auto U_) {
            constexpr int u = decltype(U_)::value;
            // Pin the slot's inputs here: the slot's math depends on these asm outputs, so it is not computed before
            // the previous slot's store (the SDK10 DAG scheduler otherwise hoisted slot 3's dy-independent sigmoid
            // into slot 0: bit-identical, but every hoisted value stays live across the slots in between).
            asm volatile("" : "+v"(hx[u]), "+v"(dq[u]));
            // the slot's row statistics, lane-uniform per 32-lane half (lanes 0-31: row c_row(w,u,0); 32-63: +1)
            const float mean = hi_half ? smv[u][1] : smv[u][0];
            const float rstd = hi_half ? srv[u][1] : srv[u][0];
            // ---- PR #7's per-row sequence [CS 2], 8 columns as 4 pairs (2i, 2i+1); every op one IEEE rounding
            const f32x2 rs2 = {rstd, rstd};
            const float nmr = -(mean * rstd);           // S1
            const f32x2 nmr2 = {nmr, nmr};
            f32x2 xh[4], dy[4], sg[4], g[4], qv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const f32x2 x = bf2(hx[u][i]);
                dy[i] = bf2(dq[u][i]);
                xh[i] = fma2(x, rs2, nmr2);             // E1  xhat = fma(x, rstd, nmr)
                const f32x2 z = fma2(xh[i], w2v[i], b2v[i]);  // E2  z = fma(xhat, w2, b2)
                const float e0 = __builtin_amdgcn_exp2f(z.x);  // E3  v_exp_f32
                const float e1 = __builtin_amdgcn_exp2f(z.y);
                const float d0 = e0 + 1.0f;             // E4
                const float d1 = e1 + 1.0f;
                sg[i] = f32x2{__builtin_amdgcn_rcpf(d0), __builtin_amdgcn_rcpf(d1)};  // E5  v_rcp_f32
                const f32x2 der = fma2(-sg[i], sg[i], sg[i]);  // E6  deriv = fma(-sig, sig, sig)
                const f32x2 pp = dy[i] * x;             // E7  p = dy * x
                const f32x2 tt = pp * der;              // E8  t = p * deriv
                g[i] = wv[i] * tt;                      // E9  g = w * t
                adw[i] = fma2(tt, xh[i], adw[i]);       // E10 acc_dw = fma(t, xhat, acc_dw)
                adb[i] = adb[i] + tt;                   // E11 acc_db = acc_db + t
                qv[i] = xh[i] * g[i];                   // E12 q = xhat * g
            }
            const float s1 = row_tree32(hsum8(qv[0], qv[1], qv[2], qv[3]));  // R1
            const float s2r = row_tree32(hsum8(g[0], g[1], g[2], g[3]));     // R2
            const float c1 = s1 * 0x1p-8f;              // R3  (1/D = 2^-8, 0x3B800000)
            const float c2 = s2r * 0x1p-8f;
            const float a1 = c1 * rstd;                 // R4
            const float a2 = c2 * rstd;
            const f32x2 a12 = {a1, a1}, a22 = {a2, a2};
            i32x4 o;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const f32x2 inner = fma2(xh[i], a12, a22);   // E13 inner = fma(xhat, a1, a2)
                const f32x2 v = fma2(g[i], rs2, -inner);     // E14 v = fma(g, rstd, -inner)
                const f32x2 dx = fma2(dy[i], sg[i], v);      // E15 dx = fma(dy, sig, v)
                o[i] = (int)pk_bf16(dx);                     // E16 dh1 = bf16_rne(dx)
                adx[i] = adx[i] + dx;                        // E17 acc_dxs = acc_dxs + dx (fp32 dx)
            }
            // pin the partial chains per slot (L11:776-781: else LLVM sinks the slots' updates and keeps their
            // temporaries live)
            asm volatile("" : "+v"(adw[0]), "+v"(adw[1]), "+v"(adw[2]), "+v"(adw[3]));
            asm volatile("" : "+v"(adb[0]), "+v"(adb[1]), "+v"(adb[2]), "+v"(adb[3]));
            asm volatile("" : "+v"(adx[0]), "+v"(adx[1]), "+v"(adx[2]), "+v"(adx[3]));
            x2lb_llvm_buffer_store_v4i32(o, rDH1, (int)(vC + (uint32_t)u * SLOT_STRIDE), 0, X2LB_ST_CP);
            X2LB_SB();
            prefetch_slot(tn, rows_n, U_);  // the slot's registers now carry tile t+G's h1 (S0: and da1)
            X2LB_SB();
        });
        bofs ^= (uint32_t)ABYTES;
    }

    // ---- WG partials: drain (all DMAs / prefetches / stores), stage the 16 slot partials per column in the free A
    // buffers, reduce each (o, c) over the 16 slots with a fixed balanced tree, write PART[o][B][c]
    waitcnt<0, 0>();
    X2LB_BARRIER();
    {
        float* red = reinterpret_cast<float*>(smem);
        const int sl = acc_slot(w, lane), c0 = c_col0(lane);
        *reinterpret_cast<f32x4*>(red + red_index(0, sl, c0)) = f32x4{adw[0].x, adw[0].y, adw[1].x, adw[1].y};
        *reinterpret_cast<f32x4*>(red + red_index(0, sl, c0 + 4)) = f32x4{adw[2].x, adw[2].y, adw[3].x, adw[3].y};
        *reinterpret_cast<f32x4*>(red + red_index(1, sl, c0)) = f32x4{adb[0].x, adb[0].y, adb[1].x, adb[1].y};
        *reinterpret_cast<f32x4*>(red + red_index(1, sl, c0 + 4)) = f32x4{adb[2].x, adb[2].y, adb[3].x, adb[3].y};
        *reinterpret_cast<f32x4*>(red + red_index(2, sl, c0)) = f32x4{adx[0].x, adx[0].y, adx[1].x, adx[1].y};
        *reinterpret_cast<f32x4*>(red + red_index(2, sl, c0 + 4)) = f32x4{adx[2].x, adx[2].y, adx[3].x, adx[3].y};
        __syncthreads();
        for (int pi = (int)threadIdx.x; pi < 3 * NOUT; pi += NTHREADS) {
            const int o = pi / NOUT, c = pi % NOUT;
            float v[2 * NWAVE];
#pragma unroll
            for (int k = 0; k < 2 * NWAVE; ++k) v[k] = red[red_index(o, k, c)];
            // fixed tree: ((((v0+v1)+(v2+v3))+((v4+v5)+(v6+v7)))+(((v8+v9)+(v10+v11))+((v12+v13)+(v14+v15))))
#pragma unroll
            for (int h = 1; h < 2 * NWAVE; h *= 2)
#pragma unroll
                for (int k = 0; k < 2 * NWAVE; k += 2 * h) v[k] = v[k] + v[k + h];
            p.part[((size_t)o * (size_t)G + (size_t)B) * NOUT + c] = v[0];
        }
    }
}

// Final reduction of the per-WG partial rows: out[o][c] = (((+0 + P[o][0][c]) + P[o][1][c]) + ...) + P[o][G-1][c],
// o = 0 dW, 1 dB, 2 dXS (= db1). One block per o, one thread per column.
// The loads are issued in batches of FINB (independent, all in flight), the adds stay strictly sequential in ascending
// g (no fast-math: clang cannot reassociate them); a plain loop compiled to one load -> wait -> add per g (latency bound).
constexpr int FINB = 32;
__global__ __launch_bounds__(256) void x2lb_fin(const float* __restrict__ part, int G, float* __restrict__ out) {
#pragma clang fp contract(off)
    const int o = (int)blockIdx.x, c = (int)threadIdx.x;
    const float* pp = part + (size_t)o * (size_t)G * NOUT + c;
    float acc = 0.0f;
    int g = 0;
    for (; g + FINB <= G; g += FINB) {
        float v[FINB];
#pragma unroll
        for (int i = 0; i < FINB; ++i) v[i] = pp[(size_t)(g + i) * NOUT];
#pragma unroll
        for (int i = 0; i < FINB; ++i) acc = acc + v[i];
    }
    for (; g < G; ++g) acc = acc + pp[(size_t)g * NOUT];
    out[o * NOUT + c] = acc;
}
#endif  // !X2LB_INDEX_ONLY

}  // namespace x2lb
#if !X2LB_INDEX_ONLY

// =====================================================================================================================
// 5. C ABI (ctypes; L11 / X2H pattern). Return 0 = launched; > 0 = hipError_t of a rejected argument or a failed
// launch (outputs not written); < 0 = launched, and -rc is a stale error an earlier call had left on this thread.
// =====================================================================================================================
#ifndef X2LB_SRC_SHA256
#define X2LB_SRC_SHA256 "unknown"
#endif
// 2026101202 (fix pass): x2lb_launch* take part_rows (= PART.shape[1]) and require grid == part_rows; x2lb_variant().
#define X2LB_ABI_VERSION 2026101202
#define X2LB_STR2(x) #x
#define X2LB_STR(x) X2LB_STR2(x)
// largest M the kernel's int32 tile arithmetic supports (64 * (t + grid) < 2^31 for every tile index the static
// schedule forms, incl. the one-past-the-end prefetch tile); the lab validates its own range
static constexpr long long X2LB_M_MAX = 1LL << 29;

extern "C" int x2lb_abi_version(void) { return X2LB_ABI_VERSION; }

// The build variant, so that a negative-control / probe library can never pass for the production one (R5): bit 0 =
// X2LB_G3_KPERM, bit 1 = X2LB_G3_FMAEPI, bit 2 = X2LB_PROBE_NOFILL. 0 = the production kernel.
extern "C" int x2lb_variant(void) {
    return (X2LB_G3_KPERM ? 1 : 0) | (X2LB_G3_FMAEPI ? 2 : 0) | (X2LB_PROBE_NOFILL ? 4 : 0);
}

extern "C" const char* x2lb_impl(void) {
    return "x2lb_pass v0.1 (S1 / S1+EMIT_DA1 / S0) BM64 8 waves persistent static tiles, resident W, lockstep; "
           "g3_kperm=" X2LB_STR(X2LB_G3_KPERM) " g3_fmaepi=" X2LB_STR(X2LB_G3_FMAEPI) " probe_nofill="
           X2LB_STR(X2LB_PROBE_NOFILL) " ld_cp=" X2LB_STR(X2LB_LD_CP) " st_cp=" X2LB_STR(X2LB_ST_CP)
           " dma_cp=" X2LB_STR(X2LB_DMA_CP) " src_sha256=" X2LB_SRC_SHA256;
}

// Grid of the persistent kernel = number of PART rows: G = min(ceil(M / 64), num_cu). Returns G >= 1, or
// -hipErrorInvalidValue for M outside [1, X2LB_M_MAX] or num_cu < 1. M = 0 is not launched: the caller returns an
// empty dh1 and zero dW / dB / dXS, as PR #7 does (ln_bwd.py:783-786; x2lb.py run()) [R7].
extern "C" long long x2lb_plan(long long M, long long num_cu) {
    if (M < 1 || M > X2LB_M_MAX || num_cu < 1) return -(long long)hipErrorInvalidValue;
    const long long nt = (M + x2lb::BM - 1) / x2lb::BM;
    return nt < num_cu ? nt : num_cu;
}

extern "C" int x2lb_drain_last_error(void) { return (int)hipGetLastError(); }

static inline bool x2lb_a16(const void* q) { return q != nullptr && ((uintptr_t)q & 15u) == 0; }
static inline bool x2lb_a4(const void* q) { return q != nullptr && ((uintptr_t)q & 3u) == 0; }

// Contract of every launch (hard-coded strides; the Python wrapper checks the same):
//   dh8 e5m2 [M,512] row-major; w8t e4m3 [256,512] row-major (the storage of the w8_t.t() operand); sa = dh_scale,
//   sb = w_scale fp32 [1] device; h1, dh1, da1 bf16 [M,256] row-major; mean1, rstd1 fp32 [M]; nw, nb fp32 [256];
//   part fp32 [3, part_rows, 256] with part_rows == grid (every element written; the pass's o-stride is grid*256, so a
//   PART of any other row count would be misread) [R6]; grid in [1, ceil(M/64)] (x2lb_plan: min(ceil(M/64), #CU)).
//   x2lb_launch: S1 (da1 unused, may be null). x2lb_launch_emit: S1 + da1 out. x2lb_launch_s0: S0, da1 in, dh8 / w8t /
//   sa / sb unused (may be null).
template <int S0, int EMIT>
static int x2lb_launch_t(const void* dh8, const void* w8t, const void* sa, const void* sb, const void* h1,
                         const void* mean1, const void* rstd1, const void* nw, const void* nb, void* dh1, void* da1,
                         void* part, long long part_rows, long long M, long long grid, void* stream) {
    if (M < 1 || M > X2LB_M_MAX) return (int)hipErrorInvalidValue;
    const long long nt = (M + x2lb::BM - 1) / x2lb::BM;
    if (grid < 1 || grid > nt || grid > 0x7FFFFFFFLL) return (int)hipErrorInvalidValue;
    if (part_rows != grid) return (int)hipErrorInvalidValue;
    if (!(x2lb_a16(h1) && x2lb_a16(nw) && x2lb_a16(nb) && x2lb_a16(dh1) && x2lb_a16(part))) return (int)hipErrorInvalidValue;
    if (!(x2lb_a4(mean1) && x2lb_a4(rstd1))) return (int)hipErrorInvalidValue;
    if (!S0 && !(x2lb_a16(dh8) && x2lb_a16(w8t) && x2lb_a4(sa) && x2lb_a4(sb))) return (int)hipErrorInvalidValue;
    if ((S0 || EMIT) && !x2lb_a16(da1)) return (int)hipErrorInvalidValue;
    x2lb::Params prm{};
    prm.dh8 = (const uint8_t*)dh8;
    prm.w8t = (const uint8_t*)w8t;
    prm.sa = (const float*)sa;
    prm.sb = (const float*)sb;
    prm.h1 = (const uint16_t*)h1;
    prm.mean1 = (const float*)mean1;
    prm.rstd1 = (const float*)rstd1;
    prm.nw = (const float*)nw;
    prm.nb = (const float*)nb;
    prm.dh1 = (uint16_t*)dh1;
    prm.da1 = (uint16_t*)da1;
    prm.part = (float*)part;
    prm.M = (int)M;
    prm.ntiles = (int)nt;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((x2lb::x2lb_pass<S0, EMIT>), dim3((unsigned)grid), dim3(x2lb::NTHREADS), 0,
                       (hipStream_t)stream, prm);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

extern "C" int x2lb_launch(const void* dh8, const void* w8t, const void* sa, const void* sb, const void* h1,
                           const void* mean1, const void* rstd1, const void* nw, const void* nb, void* dh1, void* da1,
                           void* part, long long part_rows, long long M, long long grid, void* stream) {
    return x2lb_launch_t<0, 0>(dh8, w8t, sa, sb, h1, mean1, rstd1, nw, nb, dh1, da1, part, part_rows, M, grid, stream);
}
extern "C" int x2lb_launch_emit(const void* dh8, const void* w8t, const void* sa, const void* sb, const void* h1,
                                const void* mean1, const void* rstd1, const void* nw, const void* nb, void* dh1,
                                void* da1, void* part, long long part_rows, long long M, long long grid, void* stream) {
    return x2lb_launch_t<0, 1>(dh8, w8t, sa, sb, h1, mean1, rstd1, nw, nb, dh1, da1, part, part_rows, M, grid, stream);
}
extern "C" int x2lb_launch_s0(const void* dh8, const void* w8t, const void* sa, const void* sb, const void* h1,
                              const void* mean1, const void* rstd1, const void* nw, const void* nb, void* dh1, void* da1,
                              void* part, long long part_rows, long long M, long long grid, void* stream) {
    return x2lb_launch_t<1, 0>(dh8, w8t, sa, sb, h1, mean1, rstd1, nw, nb, dh1, da1, part, part_rows, M, grid, stream);
}

// out fp32 [3, 256] (dW, dB, dXS) from part fp32 [3, part_rows, 256]; part_rows = PART.shape[1] = the grid of the pass
// that wrote it (the pass enforces grid == part_rows, so pass and finalize cannot disagree on G) [R6]
extern "C" int x2lb_fin_launch(const void* part, long long part_rows, void* out, void* stream) {
    if (part_rows < 1 || part_rows > 0x7FFFFFFFLL || !x2lb_a16(part) || !x2lb_a16(out)) return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL(x2lb::x2lb_fin, dim3(3), dim3(x2lb::NOUT), 0, (hipStream_t)stream, (const float*)part,
                       (int)part_rows, (float*)out);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}
#endif  // !X2LB_INDEX_ONLY

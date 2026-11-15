// =====================================================================================================================
// DO8 (r16 iteration 36, track DO8E5; gin item (DO8), DLRM_HSTU_FP8_L11_DO8): the A11 attention-backward dO quantization
// folded into L11's E5 (B1a e5m2-du) mode. A SEPARATE translation unit and .so (ops/hip/l11_do8.py builds THIS file
// only, with its own flags, cache, resource gate, self-check and latch), so a DO8 build / gate / self-check / launch
// failure latches DO8 off and never L11: ops/hip/l11_outproj_lnmd.cu and its .so are byte-identical to ship 12.
//
// What the production instantiation l11_outproj_lnmd_bwd_do8<1,2,0,0,1,2,1,0, E5=1, WLIST=2, EM=1, RET=0, EMV=2, TM=0>
// computes = L11's E5 launch (l11_outproj_lnmd_bwd with E5 = 1: same dgrad, same per-row v2/v3 math, the bf16 d(attn) DX,
// B1a's e5m2 u block of dz8 + pad rows, per-wave amax / u-colsum / dW / dB partials) on SEQUENCE-aligned 64-token tiles
// read from a device work list, PLUS the C8 dO quantizer contract of hstu_do_fp8.quantize_do_e4m3_blk on that d(attn):
//   group G(p) = (p // 64, (p % 16) // 8) on sequence-relative token p, per head h (64 lanes x 8 columns: h = lane >> 4);
//   amax_g = RNE_bf16(maxnum |d(attn)|) over the group's valid rows (NaN ignored; all-NaN group -> 0 -> E8 127);
//   e_g = _grp_exp(amax_g) (fp32 bit pattern, clamp [-60, 60], subnormal -> 60, zero -> 0); E8 = 127 - e_g;
//   dO8 = v_cvt_scalef32_pk_fp8_bf16(x_bf16x2, 2^-e_g) under MODE[23] = 1 (byte-identical to v_mul_f32(x, 2^e) +
//   v_cvt_scalef32_pk_fp8_f32(.., 1.0), the stock _blk_kernel ISA: 0 of 7,929,856 words differ, all bf16 x all e).
// 'plus' only (EM = 1): the bf16 d(attn) is still stored (R15 hstu_do8_registry contract v1.1 accepts 'plus' only).
// DO8-ONLY (r16 iteration 42, track do8only; gin item (88) (DO8-ONLY), DLRM_HSTU_FP8_L11_DO8_ONLY; R15 contract v2 'only'):
// the SAME file built with -DL11DO8_ONLY=1 is a second .so whose launch entry l11do8_e5_only_launch runs the EM = 2
// instantiation <1,2,0,0,1,2,1,0, E5=1, WLIST=2, EM=2, RET=0, EMV=2, TM=0>: everything the 'plus' launch writes (dO8,
// E8, the e5m2 u block + pad rows of DZ8, AMAXP / DWP / DBP / DUCS) except the bf16 d(attn) DX, which is not written
// (no DX store; with RET = 0 the emission reads the LDS staging copy of each bf16 row, never DX). Without the define
// this file compiles exactly the code it compiled before (the 'plus' .so, EM = 1).
// NIB (r16 iteration 54, ship16 candidate; knob DLRM_HSTU_FP8_LNMD_MASK_NIB, default OFF; edits 1-7 of r16
// probes/iter53/nib/make_do8_nib.py + the impl tag): built with -DL11_MKB_NIB=1 (l11_do8.py _flags; the 'plus' AND the
// 'only' .so), the TR == 2 consumer reads the LN*U*dropout keep mask as uint8 [M,256] "layout A" (ops/lnmd_mask_nib.py:
// lane l's dword at byte 4l, byte k = col 8l+k's keep bits in bits 0-2 | col 8l+4+k's in bits 4-6) with ONE
// buffer_load_dword per row-lane and builds the C1 keep masks with l11vd_mask_nib<P> (same keep decisions, half the mask
// bytes); impl tag "+mkb=nib". Without the define (L11_MKB_NIB = 0) every line compiles the v6x code (int8 [M,512] mask).
// MX6 (r16 iteration 56, kernel track; design r16 probes/iter53/mx6/DESIGN_r16.md sec 1.3, 1.4, 2.1-2.3 + VERIFY.md):
// MX = 0 / 1: a constexpr of the kernel body (r16 iter57 landing: the FP8 kernel keeps the base's name
// l11_outproj_lnmd_bwd_do8<14 parameters>; the MX6 kernel is l11_outproj_lnmd_bwd_do8_mx6<the same 14>).
// MX = 0 compiles exactly the code above (every MX line is under
// if constexpr (MX) or in an unused constexpr helper; L11CArgs and the kernel arguments are unchanged). MX = 1 computes the
// out-proj dgrad as MXFP6 E2M3 x E2M3 with per-32 E8M0 block scales (v_mfma_scale_f32_16x16x128_f8f6f4 cbsz:2 blgp:2,
// k512_mx6_gemm.cu's call pattern) on the same tiles, accumulators, EPI2 epilogue (host passes sa = sb = 1.0f), consumer
// and DO8 emission:
//   A = G6 uint8 [M,400] row records (heads [0,256) | tails [256,384) | E8M0 [384,400) in [q][k] order, byte
//       384 + 4q + k = K block 4k + q), LDS-DMA'd per tile into pr7's frag6 image + a [q][r][m] gradient-scale sub-plane;
//   B = W6 589,824-B pre-swizzled image (chunk (t,k) at (4t+k)*24576, wave slice w at +3072w, copied linearly);
//   W6S = the 768-B weight scale plane [48][q][k], passed in the TR == 1 stub's MR argument (unused when TR != 1), DMA'd
//       once per kernel by wave 0 (published by the first J == 0 vmcnt(0) + barrier).
// Built only with -DL11DO8_MX6=1 (the MX = 1 'only' / 'plus' entries, the dy debug launches, ABI 2, impl tag
// "+mx6(e2m3xe2m3,row400,opsel_k)"); without it the exported symbols, ABI version and impl string are the base's.
//
// Provenance: r16 iter35 track DO8, /scratch/arbor_sessions/dlrmv4_20261005_r16/probes/iter35/do8/kern/l11_do8.cu
// (sha256 71c7a110...; its V20 = E5 + WLIST + EM 1 + RET 0 + EMV 2 + packed running max, built with the DEFAULT
// scheduler: 255 VGPR, 0 VGPR spill, 0 scratch, 163,840 B LDS; bitwise vs the shipped E5 launch and vs
// quantize_do_e4m3_blk in lab task e5r). This file is that prototype through staged/tools/mk_l11_do8_cu.py
// (probes/iter36/do8e5): edits E1..E8 listed there; every other line of the kernel body is the prototype's.
// NOT YET BUILT in this form (no hipcc on the authoring host): the lab task staged/lab/t_do8e5_build.sh builds it with
// the loader's exact flags, checks the resource gate and the MODE[23] window (CFG scan of the ISA), and diffs the
// normalized ISA of the production instantiation against e5r's v20.s.
//
// Built at run time by ops/hip/l11_do8.py: hipcc --offload-arch=gfx950 -DKITTENS_CDNA4 -std=c++20 -w -O3
// -ffp-contract=off --save-temps -shared -fPIC -Rpass-analysis=kernel-resource-usage -DL11DO8_SRC_SHA256="<sha>"
// (= L11's flags WITHOUT -mllvm -amdgpu-sched-strategy=iterative-ilp: the E5 + emission instantiations crash the
// roc-7.2.1 clang-22 greedy register allocator under iterative-ilp, iter35 e5a..e5f).
// =====================================================================================================================
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
// VALU diet (r16 iter37 track l11vd, probes/iter37/l11vd: census + proto): the consumer is VALU/power bound at the cap, so
// five bitwise-neutral instruction deletions (each proven exhaustively on gfx950 hardware, see the census):
//   C1  keep masks built in the bf16 domain (two 64-bit shifts + 4 v_perm per plane) and ANDed into the packed planes,
//       dropping 24 (v_and, v_cmp, v_cndmask) selects per row (needs a finite dscale > 0: launch guard);
//   C2  one Newton step in sig_fast (== the 6-FMA chain on the whole fast-path domain);
//   C3  the fallback predicate as an unsigned max of the d bit patterns (1 compare instead of 8);
//   C4  the per-wave dW/dB partial chains as f32x2 column pairs: 8 v_pk_fma_f32 (two IEEE fp32 FMAs each, same chains)
//       instead of 16 v_fmac_f32 per row;
//   LANEFIX  the lane id rematerialized after the loop (the VGPR headroom C1 needs at E5 = 1).
// The 14 in-lane row-sum adds are pinned to the l11c_f1 operand order: with two NaN operands v_add_f32 returns src0's
// payload, and the C1 build otherwise commutes them. dx, du (bf16 / e5m2 + pad rows), amax, column-sum and dW/dB
// partials are byte-identical to the l11c_f1 kernel, NaN payloads included (proto gate: 25 data classes + 7 NaN-payload
// stress cases, E5 and bf16-du modes). NOTE: the 64-bit shifts are left to LLVM on purpose; pinning them as inline-asm
// v_lshlrev_b64 (C1P) is ~0.05 ms/call faster but makes LLVM commute dy_m * silu, which changes dx / dW NaN payloads
// when NaN dy meets NaN u, and pinning that multiply too costs more than C1P gains (proto rounds 4-5).
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
#ifndef L11DO8_BODY_PASS  // r16 iter57: see the l11_outproj_lnmd_bwd_do8 kernel below
#include <hip/hip_runtime.h>
#include <stdint.h>

#include <utility>
// >>> item (89) DIET switch (r16 iter52 L11 DO8 VALU/SGPR diet; gin item (89), DLRM_HSTU_FP8_L11_DO8_DIET,
// ops/hip/l11_do8.py). L11DO8_DIET 1 (the default, also when the define is absent) compiles the diet: O1 packed
// v_pk_max_u16 group maxima, O7(i) one SGPR nit per tile for the row tests, O11 no E == 0 select in grp_exp, v3b
// carry form of the exponent, O2 + E8 window (straight-line 4+4 rolling staging reads, E8 in the reads' latency) and
// the combine v6x exchange tree. L11DO8_DIET 0 (l11_do8.py passes -DL11DO8_DIET=0 when the knob is not "1") compiles
// exactly ship14's (c68395d) kernel body in every "#if ... // (89) DIET" region: the "#else" / "#if !" branches are
// c68395d's lines verbatim. Both settings produce the same bytes (dO8, E8, dz8, d(attn), partials); only the ISA
// (VGPR/SGPR use, instruction count) differs. Applies to the 'plus' and the 'only' (-DL11DO8_ONLY=1) builds alike.
#ifndef L11DO8_DIET
#define L11DO8_DIET 1
#endif
// <<< item (89) DIET switch

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
// r16 iter53 NIB (S/probes/iter53/nib): 4-byte buffer load of the packed keep mask (one dword per row-lane)
extern "C" __device__ int l11_llvm_buffer_load_i32(l11::i32x4 rsrc, int voffset, int soffset,
                                                   int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
#ifndef L11_MKB_NIB
#define L11_MKB_NIB 0
#endif
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
    // r16 iter35 DO8 ADDITION (read only by the WL / EM instantiations; null otherwise)
    const int* WLP;    // WL: work list, int32 pairs (start row, valid rows 1..64) of SEQUENCE-aligned 64-token tiles
    uint8_t* DO8;      // EM: e4m3 dO [M,512] (== quantize_do_e4m3_blk's dO8 of the bf16 d(attn) this kernel computes)
    uint8_t* E8;       // EM: E8 [M,4] uint8 = 127 - e_g
    // r16 iter36 do8e5 ADDITION (read only by WLIST = 2; appended last so the fields above keep their kernarg offsets)
    const int* WLCNT;  // WLIST 2: int32 [2] = {list entries (<= the host capacity passed as nblk), status} (device)
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

// ---- r16 iter56 MX6 (MX = 1 only; unused for MX = 0). LDS map (DESIGN_r16 2.1; total LDS unchanged, 163,840 B):
//   [0, 24576)      A codes: K step k at 6144k: heads [4 subtiles m][16 rows][64 B] at +1024m, tails [4][16][32 B] at
//                   +4096 + 512m (pr7 frag6 swizzles: head chunk q of row R at 16*(q ^ ((R>>2)&3)), tail chunk q at
//                   8*(q ^ (2*((R>>3)&1))));
//   [24576, 25600)  gradient-scale sub-plane [q][r][m] dwords (dword m of lane 16q + r = row 16m + r's E8M0 dword q);
//   [25600, 26368)  W scale plane [48][q][k] (= W6S), written once per kernel; [26368, 26624) zero fill (lanes 48..63);
//   [32768, 98304)  weight ring, 2 x 32 KiB slots (slot geometry kept: slot 1 doubles as the redistribution staging +
//                   DO8 exchange, 24,960 B > a 24 KiB MX6 chunk); a W6 chunk uses 8 wave slices x 3 KiB = 24 KiB.
// Every MX6 LDS-DMA source offset / LDS destination and LDS read offset below is a single-expression constexpr function
// (non-negative int arithmetic): sim_kernel_reads.py (r16 iter56 mx6/kern) extracts and evaluates these expressions.
constexpr int A6ROW = 400;     // G6 row record bytes (16 blocks x 24 B codes + 16 E8M0)
constexpr int A6SC = 24576;    // LDS gradient-scale sub-plane
constexpr int A6IMG = 25600;   // LDS A image bytes per tile (codes + scales)
constexpr int W6SC = 25600;    // LDS W scale plane (written once per kernel)
constexpr int W6SB = 768;      // W6S bytes = 48 x 16
constexpr int W6SL = 3072;     // W6 per-wave slice of a chunk: heads [32][64 B] then tails [32][32 B]
constexpr int W6CH = 24576;    // W6 chunk (t, k) = 8 wave slices
constexpr int W6IMG = 589824;  // W6 bytes = 24 chunks
// A side: wave w LDS-DMAs K step mx6_wk(w), subtile pair mx6_wh(w) (heads of subtiles 2h, 2h+1 and their tails); waves
// mx6_as_on(w) also DMA one dword of every row's E8M0 (q = w). Source offsets are bytes from the tile's first row. The
// ROW part is in the VGPR voffset, which the buffer range check (num_records = rv * 400) covers; soffset is NOT
// range-checked (CDNA4 ISA 9.1.5.1; x2h_pass.cu:457-458), so it carries only the in-row (K step / q) column offset.
// Rows >= rv therefore read 0 (codes 0, E8M0 0: products exactly 0) and nothing past the tile is fetched.
constexpr int mx6_wk(int w) { return w >> 1; }
constexpr int mx6_wh(int w) { return w & 1; }
constexpr bool mx6_as_on(int w) { return w < 4; }
constexpr int mx6_ah_v(int l, int h, int j) { return (32 * h + 16 * j + (l >> 2)) * A6ROW + 16 * ((l & 3) ^ ((l >> 4) & 3)); }
constexpr int mx6_ah_s(int k) { return 64 * k; }
constexpr int mx6_ah_d(int k, int h, int j) { return 6144 * k + 1024 * (2 * h + j); }
constexpr int mx6_at_v(int l, int h) { return (32 * h + (l >> 1)) * A6ROW + 256 + 16 * ((l & 1) ^ ((l >> 4) & 1)); }
constexpr int mx6_at_s(int k) { return 32 * k; }
constexpr int mx6_at_d(int k, int h) { return 6144 * k + 4096 + 1024 * h; }
constexpr int mx6_as_v(int l) { return (16 * (l & 3) + (l >> 2)) * A6ROW + 384; }
constexpr int mx6_as_s(int w) { return 4 * w; }
constexpr int mx6_as_d(int w) { return A6SC + 256 * w; }
// W side: chunk (t, k), wave w: 3 linear 1 KiB LDS-DMAs (i = 0..2) of the wave's 3 KiB slice into ring slot buf; the W
// scale plane: one 16 B-per-lane LDS-DMA by wave 0 in the prologue (num_records W6SB: lanes 48..63 write zeros).
constexpr int mx6_w_v(int l, int w) { return 16 * l + W6SL * w; }
constexpr int mx6_w_s(int t, int k, int i) { return (4 * t + k) * W6CH + 1024 * i; }
constexpr int mx6_w_d(int buf, int w, int i) { return ABYTES + buf * WCH + W6SL * w + 1024 * i; }
constexpr int mx6_wp_v(int l) { return 16 * l; }
// readers (lane l = 16 q4 + r16): frag6 head / tail byte offset of the lane inside a 16-row subtile image; A subtile m of
// K step k; B subtile j (16 output columns) of the wave's slice in ring slot buf (wave offset in the base address)
constexpr int mx6_fh(int l) { return 64 * (l & 15) + 16 * ((l >> 4) ^ (((l & 15) >> 2) & 3)); }
constexpr int mx6_ft(int l) { return 32 * (l & 15) + 8 * ((l >> 4) ^ (2 * (((l & 15) >> 3) & 1))); }
constexpr int mx6_afh(int k, int m) { return 6144 * k + 1024 * m; }
constexpr int mx6_aft(int k, int m) { return 6144 * k + 4096 + 512 * m; }
constexpr int mx6_bh0(int w) { return ABYTES + W6SL * w; }
constexpr int mx6_bt0(int w) { return ABYTES + 2048 + W6SL * w; }
constexpr int mx6_bfh(int buf, int j) { return buf * WCH + 1024 * j; }
constexpr int mx6_bft(int buf, int j) { return buf * WCH + 512 * j; }
// scale reads: sA = ONE ds_read_b128 per lane per tile (dword m = subtile m; byte k = K step: op_sel_b = k);
// sW = one ds_read_b32 per lane per N tile t (W block 8t + w, q = q4; byte k = K step: op_sel_a = k)
constexpr int mx6_sa(int l) { return A6SC + 16 * l; }
constexpr int mx6_sw0(int w, int l) { return W6SC + 16 * w + 4 * (l >> 4); }
constexpr int mx6_swt(int t) { return 128 * t; }

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
// ---- r16 iter56 MX6 fragments (pr7 k512_mx6_gemm.cu:105-141 = r14 iter51 proto :213-253). One 16x128 FP6 fragment = head
// 16 B (ds_read_b128) + tail 8 B (ds_read_b64) per lane = dwords 0..5; dwords 6..7 are not read by the cbsz/blgp = 2 MFMA
// (left undefined). LDS returns in order, so a fragment stays 2 LGKM ops (as FP8's 2 x b128).
template <int IMM>
__device__ __forceinline__ rv_i32x2 ds_read8(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    rv_i32x2 v;
    asm volatile("ds_read_b64 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
template <int IMM>
__device__ __forceinline__ int ds_read4(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    int v;
    asm volatile("ds_read_b32 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
template <int SH, int ST>
__device__ __forceinline__ i32x8 frag6(uint32_t lpH, uint32_t lpT) {
    static_assert(SH >= 0 && SH < 65536 && ST >= 0 && ST < 65536, "MX6 fragment offsets fit the 16-bit ds offset");
    const i32x4 lo = ds_read16<SH>(lpH);
    const rv_i32x2 hi = ds_read8<ST>(lpT);
    const i32x4 hi4 = __builtin_shufflevector(hi, hi, 0, 1, -1, -1);
    return __builtin_shufflevector(lo, hi4, 0, 1, 2, 3, 4, 5, -1, -1);
}
// A fragments of K step K for the 4 subtiles m (16 rows each)
template <int K>
__device__ __forceinline__ void read_A4_mx6(i32x8 (&a)[4], uint32_t lpH, uint32_t lpT) {
    a[0] = frag6<mx6_afh(K, 0), mx6_aft(K, 0)>(lpH, lpT);
    a[1] = frag6<mx6_afh(K, 1), mx6_aft(K, 1)>(lpH, lpT);
    a[2] = frag6<mx6_afh(K, 2), mx6_aft(K, 2)>(lpH, lpT);
    a[3] = frag6<mx6_afh(K, 3), mx6_aft(K, 3)>(lpH, lpT);
}

template <int N>
__device__ __forceinline__ void wait_vm() {
    static_assert(N >= 0 && N <= 63, "vmcnt is 6 bit");
    if constexpr (N == 0) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    else if constexpr (N == 4) asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
    else if constexpr (N == 8) asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
    else if constexpr (N == 16) asm volatile("s_waitcnt vmcnt(16)" ::: "memory");
    else if constexpr (N == 24) asm volatile("s_waitcnt vmcnt(24)" ::: "memory");
    else if constexpr (N == 17) asm volatile("s_waitcnt vmcnt(17)" ::: "memory");  // DO8 'only': 8 du + 9 emit stores
    else if constexpr (N == 25) asm volatile("s_waitcnt vmcnt(25)" ::: "memory");  // DO8 'plus': 16 + 9 emit stores
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
// r16 iter56 MX6: the same operand-swapped MFMA as E2M3 x E2M3 (cbsz:2 blgp:2) with runtime E8M0 scales, k512_mx6_gemm.cu:
// 144-150's call pattern: src0 = weight fragment, scale_a = sw (W block 8t + w, q = lane>>4; bytes = K steps),
// op_sel_a = K step KS; src1 = gradient fragment, scale_b = sg[m] (row 16m + (lane&15), q = lane>>4; bytes = K steps),
// op_sel_b = KS (the same immediate: DESIGN_r16 1.3 fix). Same (n, m) order as mma4: (0,M0) (0,M0+1) (1,M0) (1,M0+1).
template <bool FIRST, int KS>
__device__ __forceinline__ f32x4 mf6(const i32x8& b, const i32x8& a, const f32x4& c, int sw, int sg) {
    const f32x4 z = {0.f, 0.f, 0.f, 0.f};
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b, a, FIRST ? z : c, 2, 2, KS, sw, KS, sg);
}
template <bool FIRST, int M0, int KS>
__device__ __forceinline__ void mma4_mx6(f32x4 (&c)[2][4], const i32x8 (&b)[2], const i32x8 (&a)[4], int sw,
                                         const i32x4 sg) {
    static_assert(KS >= 0 && KS < 4 && (M0 == 0 || M0 == 2), "op_sel is a 2-bit byte select");
    c[0][M0] = mf6<FIRST, KS>(b[0], a[M0], c[0][M0], sw, sg[M0]);
    c[0][M0 + 1] = mf6<FIRST, KS>(b[0], a[M0 + 1], c[0][M0 + 1], sw, sg[M0 + 1]);
    c[1][M0] = mf6<FIRST, KS>(b[1], a[M0], c[1][M0], sw, sg[M0]);
    c[1][M0 + 1] = mf6<FIRST, KS>(b[1], a[M0 + 1], c[1][M0 + 1], sw, sg[M0 + 1]);
}

// od / K512:152-168
__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    l11_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, 0);
}
// r16 iter56 MX6: dword LDS-DMA (hardware writes lane l's 4 B at lds_addr + 4*l): the gradient-scale sub-plane
__device__ __forceinline__ void g2l4(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    l11_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 4, (int)voff, (int)soff, 0, 0);
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
// r16 iter37 l11vd (C2): ONE Newton step. On the fast-path domain (u an exact bf16 with d <= 2^126: 49712 values) it
// equals the former 6-FMA chain and the IEEE quotient 1/d bit for bit (exhaustive on gfx950: probes/iter37/l11vd/census
// verify.cpp test A, 0 mismatches; v_rcp alone mismatches 425). Out-of-domain d still takes the IEEE fallback.
__device__ __forceinline__ float sig_fast(float d) {
    const float r = __builtin_amdgcn_rcpf(d);
    return __builtin_fmaf(__builtin_fmaf(-d, r, 1.0f), r, r);
}
// r16 iter37 l11vd (LANEFIX): the lane id recomputed (opaque v_mbcnt pair, not CSE'd with threadIdx & 63) at the flush
// and post-loop use sites, so no lane VGPR is live across the row-block loop (frees the VGPRs the mask change needs).
__device__ __forceinline__ int l11vd_lane() {
    int l;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\tv_mbcnt_hi_u32_b32 %0, -1, %0" : "=v"(l));
    return l;
}
// r16 iter37 l11vd (C1): plane keep masks in the bf16 domain (MGPK sign-broadcast v_perm idiom on the (mk1:mk0) pair):
// dword k of the result = (keep(2k) ? 0xffff : 0) | (keep(2k+1) ? 0xffff0000 : 0) for plane bit P (exhaustive on gfx950:
// census verify C/F, 0 mismatches). Used as RN(s * (plane & mask)) == keep ? RN(s * plane) : +0, exact for every finite
// s > 0 (census verify D; the launches reject other dscale values).
template <int P>
__device__ __forceinline__ i32x4 l11vd_mask(uint64_t pr) {
    const uint64_t t1 = pr << (7 - P), t2 = pr << (15 - P);
    i32x4 m;
    m.x = (int)__builtin_amdgcn_perm((uint32_t)t2, (uint32_t)t1, 0x08080A0Au);
    m.y = (int)__builtin_amdgcn_perm((uint32_t)t2, (uint32_t)t1, 0x09090B0Bu);
    m.z = (int)__builtin_amdgcn_perm((uint32_t)(t2 >> 32), (uint32_t)(t1 >> 32), 0x08080A0Au);
    m.w = (int)__builtin_amdgcn_perm((uint32_t)(t2 >> 32), (uint32_t)(t1 >> 32), 0x09090B0Bu);
    return m;
}
// r16 iter53 NIB (R2): the same masks from ONE packed dword m (col 8l+k plane P at bit 8k+P, col 8l+4+k at bit 8k+4+P).
// Sources a, b = l11vd_mask's low halves (m << (15-P), m << (7-P)); c, d replace the high halves of the 64-bit shifts
// (bit 15 / 31 of c = m bits 4+P / 20+P = cols 4 / 6, of d = m bits 12+P / 28+P = cols 5 / 7). Bit-for-bit
// l11vd_mask<P>((mk1:mk0)) of the unpacked int8 bytes (S/probes/iter53/nib/nib_proof.py); no mk1, 12 v_lshlrev_b32 +
// 12 v_perm_b32 per row for the 3 planes (today: 6 v_lshlrev_b32 + 6 v_alignbit_b32 + 12 v_perm_b32).
template <int P>
__device__ __forceinline__ i32x4 l11vd_mask_nib(uint32_t mn) {
    const uint32_t a = mn << (15 - P), b = mn << (7 - P), c = mn << (11 - P), d = mn << (3 - P);
    i32x4 m;
    m.x = (int)__builtin_amdgcn_perm(a, b, 0x08080A0Au);
    m.y = (int)__builtin_amdgcn_perm(a, b, 0x09090B0Bu);
    m.z = (int)__builtin_amdgcn_perm(c, d, 0x08080A0Au);
    m.w = (int)__builtin_amdgcn_perm(c, d, 0x09090B0Bu);
    return m;
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

// ---- r16 iter35 DO8 ADDITION: the C8 dO quantizer contract (hstu_do_fp8.py _grp_exp / _blk_kernel), op for op.
// _grp_exp(a): e from the fp32 bit pattern of the group amax a (a >= 0, never NaN here).
__device__ __forceinline__ int grp_exp(float a) {
    const int bits = __builtin_bit_cast(int, a);
    const int E = (bits >> 23) & 0xFF;
    const int mant = bits & 0x7FFFFF;
#if L11DO8_DIET  // (89) DIET
    // item (89) DIET (r16 iter52) v3b (iter-50 REVIEW item 6): the carry/literal form of the contract's
    //   (mant <= 0x600000) ? 135 - E : 134 - E  ==  134 - E + [mant <= 0x600000]
    // (an integer identity for every E and mant; with the clamp and the zero select below: 0 mismatches over all 2^32
    // fp32 patterns, v3/proof_grpexp.py). Written as the select, LLVM forms v_cndmask_b32(134, 135) - E with BOTH
    // constants in VGPRs (VOP2 cndmask src1 must be a VGPR; gfx9 VOP3 takes no literal) and hoists the two v_mov_b32
    // out of the persistent loop: 2 loop-resident VGPRs for 2 tail-only uses. The empty asm keeps 134 - E opaque, so
    // InstCombine cannot fold "+ zext(mant <= 0x600000)" back into select(.., 135, 134): the add stays an add of the
    // compare bit (carry-in add, or an inline 0/1 select + add) and 134 is a VOP2 literal; no constant VGPR.
    int e = 134 - E;
    asm("" : "+v"(e));
    e += (mant <= 0x600000) ? 1 : 0;
    e = e < -60 ? -60 : (e > 60 ? 60 : e);
    // r16 iter52 DIET O11: the contract's "E == 0 (fp32 subnormal amax) -> 60" select is implied by the clamp above
    // (E == 0 gives 135 or 134, both clamped to 60), so it is not emitted; every input gives the same e.
#else  // (89) DIET off: ship14 (c68395d)
    int e = (mant <= 0x600000) ? 135 - E : 134 - E;
    e = e < -60 ? -60 : (e > 60 ? 60 : e);
    e = (E == 0) ? 60 : e;        // fp32 subnormal amax
#endif  // (89) DIET
    return (a == 0.0f) ? 0 : e;   // all-zero group
}
// 4 fp32 -> 4 e4m3 bytes with the instruction _blk_kernel's `y.to(tl.float8e4nv)` compiles to on gfx950 (stock ISA:
// v_cvt_scalef32_pk_fp8_f32 ..., 1.0 and ... 1.0 op_sel:[0,0,0,1]); the caller holds MODE[23] = 1 around it exactly as
// _blk_kernel's prologue `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1` does.
__device__ __forceinline__ int cvt_pk_e4m3(float a, float b, float c, float d) {
    int v;
    asm volatile("v_cvt_scalef32_pk_fp8_f32 %0, %1, %2, 1.0" : "=v"(v) : "v"(a), "v"(b));
    asm volatile("v_cvt_scalef32_pk_fp8_f32 %0, %1, %2, 1.0 op_sel:[0,0,0,1]" : "+v"(v) : "v"(c), "v"(d));
    return v;
}
template <int CTRL>
__device__ __forceinline__ float dpp_max(float v) {  // bound_ctrl: invalid source lanes read 0 (max with 0: v >= 0)
    return __builtin_fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL,
                                                                                    0xF, 0xF, true)));
}
#if L11DO8_DIET  // (89) DIET
// r16 iter52 DIET O1: per-half unsigned max of two packed bf16 bit-pattern pairs (v_pk_max_u16). On non-negative, non-NaN
// bf16 patterns (0x0000..0x7F80) u16 order is value order, so this is the value max, exactly.
__device__ __forceinline__ uint32_t pkmax_u16(uint32_t a, uint32_t b) {
    typedef unsigned short u16x2_t __attribute__((ext_vector_type(2)));
    return __builtin_bit_cast(uint32_t, __builtin_elementwise_max(__builtin_bit_cast(u16x2_t, a), __builtin_bit_cast(u16x2_t, b)));
}
template <int CTRL>
__device__ __forceinline__ uint32_t dpp_pkmax_u16(uint32_t v) {  // bound_ctrl: invalid source lanes read 0 (u16 max identity)
    return pkmax_u16(v, (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, CTRL, 0xF, 0xF, true));
}
#endif  // (89) DIET

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
// r16 iter35 DO8 ADDITION (WLIST / EM / RET; all 0 compiles the code above):
//   WLIST = 1: the persistent loop walks ca.WLP, a device work list of SEQUENCE-aligned 64-token tiles (start row, valid
//     rows 1..64); nblk = number of tiles. A partial tile's rows >= valid are range-clamped exactly like the shipped
//     M tail (buffer resources sized to the valid rows) and skip the consumer math (wave-uniform branch).
//   EM = 1 ('plus'): + the C8 dO quantizer contract (hstu_do_fp8.quantize_do_e4m3_blk) on the bf16 d(attn) rows (DX)
//     this kernel computes: per (tile, head, group g = (r % 16) // 8) amax over the valid rows (NaN ignored, as maxnum),
//     e_g = _grp_exp(RNE_bf16 amax), dO8 = e4m3(x_bf16 * 2^e_g) (v_mul_f32 + v_cvt_scalef32_pk_fp8_f32 under MODE[23]=1),
//     E8 = 127 - e_g. bf16 DX still stored. EM = 2 ('only'): the same without the bf16 DX store. Needs WLIST = 1.
//     (r16 iter42 do8only: WLIST 2 = WLIST 1 + a device tile count; RET 0 keeps the LDS staging write and the emission
//     reads that copy; RET 2 re-reads DX and is static_asserted out; the only other DX store is the TR = 1 stub's.
//     Instantiated only by the L11DO8_ONLY .so.)
//   RET = 0: the wave's 8 bf16 d(attn) rows wait for the group scale in an extra 64 KiB LDS region (LDS 160 KiB);
//   RET = 1: in registers (8 x 4 VGPR); RET = 2 ('plus' only): re-read from the bf16 DX rows just stored (L2-hot).
// r16 iter36 do8e5: WLIST = 2 = WLIST 1 with the tile count read from the device (ca.WLCNT[0], written by
//   l11_do8_wl_build on the same stream) instead of the host nblk (the host passes the list capacity there). No host sync.
// r16 iter56 MX6: MX = 1 -> the GEMM half in MXFP6 (header "MX6"): A = G6 row records, B = W6 image, MR = W6S (the scale
//   plane; MR is the TR == 1 stub's argument, unused otherwise), sa / sb = device 1.0f. PW 1, NBUF 2, APF 0, DIAG 0 only.
#endif  // !L11DO8_BODY_PASS (r16 iter57: everything above is skipped when this file is self-included for the body)
#ifndef L11DO8_BODY_PASS
// r16 iter57 MX6 landing (symbol-preserving R1 fix, self-include form): the FP8 kernel keeps the base's 14-parameter
// template l11_outproj_lnmd_bwd_do8<...> (same mangled name, kernarg layout and body text, MX = 0); the MX6 kernel is the
// separately named l11_outproj_lnmd_bwd_do8_mx6<the same 14> (MX = 1; instantiated only by the -DL11DO8_MX6=1 host
// entries). Both bodies are the text of the #else region below, included from this same file (__FILE_NAME__) so it
// stays lexically inside each kernel, exactly as in the base: no shared device function is optimized standalone.
template <int STUB, int NBUF, int APF, int DIAG = 0, int PW = 0, int TR = 0, int PFD = 2, int SCP = 0, int E5 = 0,
          int WLIST = 0, int EM = 0, int RET = 0, int EMV = 0, int TM = 0>
__global__ __launch_bounds__(512, 2) void l11_outproj_lnmd_bwd_do8(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                     uint16_t* __restrict__ C, float* __restrict__ R,
                                                     const float* __restrict__ sa, const float* __restrict__ sb, int M,
                                                     int nblk, const uint16_t* __restrict__ X,
                                                     const uint16_t* __restrict__ U, const uint8_t* __restrict__ MK,
                                                     const float* __restrict__ MR, uint16_t* __restrict__ DX,
                                                     uint16_t* __restrict__ DU, const L11CArgs ca) {
#pragma clang fp contract(off)
    constexpr int MX = 0;  // FP8 (e5m2 x e4m3): every MX line below compiles away
#define L11DO8_BODY_PASS 1
#include __FILE_NAME__  // this file again: only its kernel-body region below (L11DO8_BODY_PASS)
#undef L11DO8_BODY_PASS
}
template <int STUB, int NBUF, int APF, int DIAG = 0, int PW = 0, int TR = 0, int PFD = 2, int SCP = 0, int E5 = 0,
          int WLIST = 0, int EM = 0, int RET = 0, int EMV = 0, int TM = 0>
__global__ __launch_bounds__(512, 2) void l11_outproj_lnmd_bwd_do8_mx6(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                     uint16_t* __restrict__ C, float* __restrict__ R,
                                                     const float* __restrict__ sa, const float* __restrict__ sb, int M,
                                                     int nblk, const uint16_t* __restrict__ X,
                                                     const uint16_t* __restrict__ U, const uint8_t* __restrict__ MK,
                                                     const float* __restrict__ MR, uint16_t* __restrict__ DX,
                                                     uint16_t* __restrict__ DU, const L11CArgs ca) {
#pragma clang fp contract(off)
    constexpr int MX = 1;  // MXFP6 (e2m3 x e2m3, E8M0 per 32)
#define L11DO8_BODY_PASS 1
#include __FILE_NAME__  // this file again: only its kernel-body region below (L11DO8_BODY_PASS)
#undef L11DO8_BODY_PASS
}
#else  // L11DO8_BODY_PASS: the kernel body (unchanged text), included once per kernel above
    using namespace l11;
    static_assert(NCH % NBUF == 0, "ring position of a chunk must not depend on the row block");
    static_assert(TR == 0 || (STUB == 1 && DIAG == 0), "traffic stub is built on the row-sum stub");
    static_assert(E5 == 0 || TR == 2, "l11e5: the e5m2 du emission is a mode of the real consumer");
    static_assert(EM == 0 || (TR == 2 && WLIST >= 1), "DO8: emission needs the real consumer + work list");
    static_assert(WLIST >= 0 && WLIST <= 2, "DO8: WLIST 1 = host tile count, 2 = device tile count (ca.WLCNT)");
    static_assert(!(EM == 2 && RET == 2), "DO8: the global re-read needs the bf16 store ('plus')");
    // r16 iter56 MX6: the per-wave ring (each wave reads only its own W6 slice: no barrier for K steps 1..23), VM_RING 0
    // (every J > 0 wait and the first J == 0 wait are vmcnt(0)), the APF 0 schedule, and MR free for W6S
    static_assert(MX == 0 || MX == 1, "MX: 0 = FP8 (e5m2 x e4m3), 1 = MXFP6 (e2m3 x e2m3, E8M0 per 32)");
    static_assert(MX == 0 || (PW == 1 && NBUF == 2 && APF == 0 && DIAG == 0 && TR != 1),
                  "MX6: PW 1, NBUF 2, APF 0, DIAG 0, TR != 1 only");
    constexpr int WL = ABYTES;                  // weight buffers start
    constexpr int SBUF = (NCH - 1) % NBUF;      // buffer of the block's last chunk = redistribution staging
    constexpr int STG = WL + SBUF * WCH;
    static_assert(8 * SPITCH <= (uint32_t)WCH, "staging fits one weight buffer");
    // vector-memory stores per redistribution iteration per wave (TR=2: dx + du; the dW/dB flush stores come on top,
    // which only makes the step-0 wait_vm<S_ST> stricter)
    constexpr int NST = STUB == 0 ? 3 : (TR == 2 ? (EM == 2 ? 1 : 2) : (TR ? 3 : 1));
    constexpr int NRED = DIAG == 1 ? 2 : 1;     // redistribution passes per row block
    // DO8: + 8 dO8 row stores + 1 E8 store per wave per tile, all younger than the next tile's A prefetch (the 8 * 5
    // consumer loads are always issued, so the A tile stays older than the S_ST youngest ops)
    constexpr int S_ST = 8 * NST * NRED + (EM ? 9 : 0);
    static_assert(DIAG == 0 || (STUB == 1 && APF == 0), "DIAG variants are stub-only, APF=0");
    constexpr int VM_RING = NBUF == 3 ? 4 : 0;  // DMA instructions allowed in flight behind the chunk we wait for
    constexpr int XDXB = ABYTES + NBUF * WCH;   // DO8 RET 0: bf16 d(attn) tile staging (64 rows x 1 KiB)
    constexpr int XCHB = STG + 8 * (int)SPITCH; // DO8: per-wave group-max exchange (2 x 4 x 8 fp32) in the staging tail
    static_assert(8 * SPITCH + 256 <= (uint32_t)WCH, "DO8 exchange fits the staging buffer tail");
    __shared__ __attribute__((aligned(2048))) uint8_t smem[ABYTES + NBUF * WCH + ((EM && RET == 0) ? 65536 : 0)];

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
    // DO8 WLIST: the first tile's (start row, valid rows), read before the hand-counted bookkeeping starts
    int cur0 = 0, curv = 0;
    if constexpr (WLIST) {
        if constexpr (WLIST == 2) {
            // r16 iter36 do8e5: the device-built list's entry count replaces the host capacity in nblk before any use
            // (a vector load: the struct is not noalias; wave-uniform -> SGPR; retired by the vmcnt(0) below)
            nblk = __builtin_amdgcn_readfirstlane(ca.WLCNT[0]);
            asm volatile("" : "+s"(nblk));
        }
        if ((int)blockIdx.x < nblk) {
            const rv_i32x2 e0 = *reinterpret_cast<const rv_i32x2*>(ca.WLP + 2 * (int)blockIdx.x);
            cur0 = __builtin_amdgcn_readfirstlane(e0.x);
            curv = __builtin_amdgcn_readfirstlane(e0.y);
        }
        asm volatile("" : "+s"(cur0), "+s"(curv));
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
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
    // DO8 RET 0: this wave's row slot in the d(attn) tile staging (row it*8 + w at + it * 8 KiB)
    const uint32_t xdxw = lds0 + (uint32_t)XDXB + (uint32_t)w * 1024u + 16u * (uint32_t)lane;
    // DO8 TM: the same address re-derived at each use from rst (+ a wave-uniform SGPR offset) inside an asm, so no VGPR
    // holds it across the persistent loop (the E5 + emission builds crash the greedy RA once they spill)
    const int xoffs = (int)XDXB - (int)STG + w * (1024 - (int)SPITCH);
    auto xdx_addr = [&]() {
        uint32_t a;
        asm volatile("v_add_u32 %0, %1, %2" : "=v"(a) : "s"(xoffs), "v"(rst));
        return a;
    };
    // ---- r16 iter56 MX6 (MX = 1; for MX = 0 these are constant 0 / unused): fragment, scale and W6-DMA lane offsets
    const uint32_t lpH = MX ? lds0 + (uint32_t)mx6_fh(lane) : 0u;                    // A heads (+ mx6_afh(k, m))
    const uint32_t lpT = MX ? lds0 + (uint32_t)mx6_ft(lane) : 0u;                    // A tails (+ mx6_aft(k, m))
    const uint32_t lpBH = MX ? lds0 + (uint32_t)(mx6_bh0(w) + mx6_fh(lane)) : 0u;   // B heads (+ mx6_bfh(buf, j))
    const uint32_t lpBT = MX ? lds0 + (uint32_t)(mx6_bt0(w) + mx6_ft(lane)) : 0u;   // B tails (+ mx6_bft(buf, j))
    const uint32_t lpWS = MX ? lds0 + (uint32_t)mx6_sw0(w, lane) : 0u;              // W scale dword (+ mx6_swt(t))
    const uint32_t voffW6 = MX ? (uint32_t)mx6_w_v(lane, w) : 0u;                   // W6 DMA voffset (loop-invariant)
    const i32x4 rW6 = make_rsrc(B, (uint32_t)W6IMG);
    // A tile of one work-list entry (rows [row0, row0 + rvv)): 3 x 16 B LDS-DMAs per wave (+ 1 dword DMA, waves 0..3).
    // The voffsets are formed here from an OPAQUE lane id (l11vd_lane, the LANEFIX idiom): nothing is held across the
    // persistent loop. Rows >= rvv are out of range (row offset in voffset) and write zeros.
    auto issue_A6 = [&](const uint8_t* ab, int rvv) {
        if constexpr (MX) {
            const i32x4 rA6 = make_rsrc(ab, (uint32_t)(rvv * A6ROW));
            const int l = l11vd_lane();
            const int kk = mx6_wk(w), hh = mx6_wh(w);
            g2l(rA6, lds0 + (uint32_t)mx6_ah_d(kk, hh, 0), (uint32_t)mx6_ah_v(l, hh, 0), (uint32_t)mx6_ah_s(kk));
            g2l(rA6, lds0 + (uint32_t)mx6_ah_d(kk, hh, 1), (uint32_t)mx6_ah_v(l, hh, 1), (uint32_t)mx6_ah_s(kk));
            g2l(rA6, lds0 + (uint32_t)mx6_at_d(kk, hh), (uint32_t)mx6_at_v(l, hh), (uint32_t)mx6_at_s(kk));
            if (mx6_as_on(w)) g2l4(rA6, lds0 + (uint32_t)mx6_as_d(w), (uint32_t)mx6_as_v(l), (uint32_t)mx6_as_s(w));
        }
    };

    auto issue_A = [&](const i32x4 rA) {
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, ldsw + (uint32_t)(i * 8192), voffA, (uint32_t)(i * 128));
    };
    int zs = 0;  // opaque SGPR zero, re-materialized per row block (see the loop head)
    auto issue_W = [&](auto CH_) {  // weight chunk CH (tile CH/4, K step CH%4) -> ring buffer CH % NBUF
        constexpr int CH = decltype(CH_)::value;
        constexpr int t = CH / 4, k = CH % 4, buf = CH % NBUF;
        if constexpr (MX) {
            // r16 iter56 MX6: wave w's 3 KiB slice of W6 chunk (t, k) = 3 linear 1 KiB copies into its ring-slot slice
            // (72 distinct soffsets per row block instead of 96, formed from zs at use as today)
#pragma unroll
            for (int i = 0; i < 3; ++i)
                g2l(rW6, lds0 + (uint32_t)mx6_w_d(buf, w, i), voffW6, (uint32_t)(zs + mx6_w_s(t, k, i)));
        } else if constexpr (PW) {
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
        const int row0 = WLIST ? cur0 : blk * BM;
        if constexpr (MX) {
            issue_A6(A + (size_t)row0 * A6ROW, WLIST ? curv : min(BM, M - row0));
            // the W scale plane, once per kernel: one LDS-DMA by wave 0 (DESIGN_r16 1.3 defect 1 fix): the first J == 0
            // vmcnt(0) + barrier publishes it to every wave (no compiler-visible LDS write racing that barrier)
            if (w == 0)
                g2l(make_rsrc(reinterpret_cast<const uint8_t*>(MR), (uint32_t)W6SB), lds0 + (uint32_t)W6SC,
                    (uint32_t)mx6_wp_v(lane), 0u);
        } else {
        issue_A(make_rsrc(A + (size_t)row0 * KD, (uint32_t)((WLIST ? curv : min(BM, M - row0)) * KD)));
        }
        issue_W(std::integral_constant<int, 0>{});
        if constexpr (NBUF == 3) issue_W(std::integral_constant<int, 1>{});
    }
    bool first = true;
    uint32_t chk = 0;

    // ---- L11-1b ADDITION: consumer state (TR == 2): norm weight/bias of this lane's 8 columns (fp32), dW/dB partials.
    constexpr int NC = TR == 2 ? 8 : 1;
    constexpr int NC2 = NC > 1 ? NC / 2 : 1;
    f32x2 pdw2[NC2], pdb2[NC2];  // r16 iter37 l11vd (C4): column pairs (2k, 2k+1), v_pk_fma_f32 chains
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
            pdw2[e >> 1][e & 1] = 0.f;
            pdb2[e >> 1][e & 1] = 0.f;
        }
        if constexpr (E5) {
#pragma unroll
            for (int e = 0; e < NC5; ++e) pcs[e] = f32x2{0.f, 0.f};
        }
    }
    auto flush = [&](int sl) {  // partial row (blockIdx.x*8 + w)*nf + sl, this lane's 8 columns; then restart the chains
        if constexpr (TR == 2) {
            const size_t off = ((size_t)(blockIdx.x * 8 + w) * (size_t)ca.nf + (size_t)sl) * 512u + 8u * (size_t)l11vd_lane();
            *reinterpret_cast<f32x4*>(ca.DWP + off) = f32x4{pdw2[0].x, pdw2[0].y, pdw2[1].x, pdw2[1].y};
            *reinterpret_cast<f32x4*>(ca.DWP + off + 4) = f32x4{pdw2[2].x, pdw2[2].y, pdw2[3].x, pdw2[3].y};
            *reinterpret_cast<f32x4*>(ca.DBP + off) = f32x4{pdb2[0].x, pdb2[0].y, pdb2[1].x, pdb2[1].y};
            *reinterpret_cast<f32x4*>(ca.DBP + off + 4) = f32x4{pdb2[2].x, pdb2[2].y, pdb2[3].x, pdb2[3].y};
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pdw2[e >> 1][e & 1] = 0.f;
                pdb2[e >> 1][e & 1] = 0.f;
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
        const int row0 = WLIST ? cur0 : blk * BM;
        const int rv = WLIST ? curv : min(BM, M - row0);
        // DO8 WLIST: the next tile's entry, loaded now (one uniform vector load), consumed after this tile's mainloop
        rv_i32x2 nxte = {0, 0};
        if constexpr (WLIST) {
            if (blk + G < nblk) nxte = *reinterpret_cast<const rv_i32x2*>(ca.WLP + 2 * (blk + G));
        }
        // DO8 EM: per-lane running |d(attn)| max of the tile's two token groups (this lane's head = lane >> 4)
        float amg0 = 0.f, amg1 = 0.f;
        // DO8 E5 instantiations (PKA): the two running group maxima as ONE packed pair of bf16 bit patterns (v_pk_max_u16
        // on RNE_bf16 of the row max: RNE is monotone, so the final group max is identical; saves a VGPR)
        constexpr bool PKA = (E5 != 0);
        uint32_t amgp = 0u;
        i32x4 oxr[(EM && RET == 1) ? 8 : 1];  // DO8 RET 1: the wave's 8 bf16 d(attn) rows
        const bool has_next = blk + G < nblk;

        i32x4 pk[NT][4];  // retained dy: [tile][16-row group] = 8 bf16 (cols 256t+32w+8q4..+8, row 16m+r16)
        f32x4 acc[2][4];
        i32x8 af[APF ? 2 : 1][4];
        i32x8 bf[2];
        i32x4 sA6;  // r16 iter56 MX6: this tile's gradient E8M0 dwords (dword m = row 16m + r16, q4; byte k = K step)
        int sW6;    // r16 iter56 MX6: the N tile's weight E8M0 dword (block 8t + w, q4; byte k = K step)

        static_for<NCH>([&](auto J_) {
            constexpr int J = decltype(J_)::value;
            constexpr int t = J / 4, k = J % 4, buf = J % NBUF;
            constexpr int as = APF ? (J & 1) : 0;
            // chunk J (and at J == 0 the A tile) must have landed: in-order vmcnt bookkeeping (see header of wait_vm users)
            if constexpr (J == 0) {
                if (first) wait_vm<VM_RING>();
                // r16 iter56 MX6 fix (REVIEW_kernel R8): the untimed MX = 1 dy debug instantiation (STUB 0: S_ST = 24 =
                // exactly its 24 always-issued C stores, zero margin) waits vmcnt(0) instead; every other instantiation,
                // MX = 0 included, keeps S_ST (same template argument -> same code)
                else wait_vm<(MX && STUB == 0) ? 0 : S_ST>();
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
            if constexpr (MX) {
                // r16 iter56 MX6 step (DESIGN_r16 2.2 edit 10). LGKM issue order: [sA6 b128 at J == 0] [sW6 b32 at k == 0]
                // B 2 x (b128 + b64), A 4 x (b128 + b64): at most 1 + 1 + 4 + 8 = 14 in flight (lgkmcnt is 4 bit: <= 15).
                // wait_lgkm<4> leaves exactly A m = 2, 3 outstanding (LDS returns in order), as FP8's.
                // J == 0 first drains LGKM (REVIEW_kernel R3): an SMEM load the compiler sinks into the loop preheader
                // (s_load_dwordx8 of ca.X..MEAN in the proxy ISA) would otherwise make it 14 + 1 = 15 on a WG's first tile,
                // the counter's maximum; with the drain the count is <= 14 for any SMEM placement. After a WG's first tile
                // no LGKM op is outstanding here (isa_wait_check.py on the proxy ISA), so the wait costs one issue slot.
                if constexpr (J == 0) {
                    wait_lgkm<0>();
                    sA6 = ds_read16<0>(lds0 + (uint32_t)mx6_sa(lane));  // ONE b128 per lane per tile
                }
                if constexpr (k == 0) sW6 = ds_read4<mx6_swt(t)>(lpWS);
                bf[0] = frag6<mx6_bfh(buf, 0), mx6_bft(buf, 0)>(lpBH, lpBT);
                bf[1] = frag6<mx6_bfh(buf, 1), mx6_bft(buf, 1)>(lpBH, lpBT);
                read_A4_mx6<k>(af[0], lpH, lpT);
                wait_lgkm<4>();  // sA6, sW6, B frags + A frags m = 0, 1 landed
                L11_SB();
                mma4_mx6<k == 0, 0, k>(acc, bf, af[0], sW6, sA6);
                L11_SB();
                wait_lgkm<0>();
                L11_SB();
                mma4_mx6<k == 0, 2, k>(acc, bf, af[0], sW6, sA6);
            } else {
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
            }  // MX
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
        int nx0 = 0, nxv = 0;  // DO8 WLIST: next tile (carried to the next iteration)
        if constexpr (WLIST) {
            nx0 = __builtin_amdgcn_readfirstlane(nxte.x);
            nxv = __builtin_amdgcn_readfirstlane(nxte.y);
        }
        if (has_next) {
            const int nrow0 = WLIST ? nx0 : (blk + G) * BM;
            if constexpr (MX) {
                issue_A6(A + (size_t)nrow0 * A6ROW, WLIST ? nxv : min(BM, M - nrow0));  // r16 iter56 MX6 (edit 11)
            } else {
            issue_A(make_rsrc(A + (size_t)nrow0 * KD, (uint32_t)((WLIST ? nxv : min(BM, M - nrow0)) * KD)));
            }
        }
        L11_SB();
#if L11DO8_DIET  // (89) DIET
        // r16 iter52 DIET O7(i): row it*8 + w of this wave is a valid tile row (it*8 + w < rv) iff it < nit, with ONE
        // wave-uniform SGPR per tile nit = (rv - w + 7) >> 3 = ceil((rv - w) / 8) (exhaustive: rv 1..64 x w 0..7 x it 0..7,
        // 0 mismatches; and for every int rv, w, it without overflow: x < rv <=> x <= rv - 1 and floor). The former test
        // compared rv with 7 loop-invariant SGPR constants it*8 + w (it = 1..7) that LLVM hoisted out of the persistent
        // loop and spilled (production: 7 v_readlane reloads per tile-wave, v252 lanes 27-33). The pin keeps the shift
        // form: no InstCombine re-association back to "w + 8*it < rv"; the tests compare nit with inline constants.
        int nit = 0;
        if constexpr (WLIST) {
            nit = (rv - w + 7) >> 3;
            asm volatile("" : "+s"(nit));
        }
#endif  // (89) DIET

        const i32x4 rC = make_rsrc(C + (size_t)row0 * NOUT, (uint32_t)rv * ROWB);
        const i32x4 rR = make_rsrc(R + row0, (uint32_t)rv * 4u);
        // REVIEW ADDITION: consumer-traffic stub (TR) resources, per-block 64-bit bases, range-checked like rC
        i32x4 rX{}, rU{}, rM{}, rMR{}, rDX{}, rDU{};
        i32x4 rMn{}, rRs{};  // L11-1b: separate mean / rstd fp32 [M]
        i32x4 px[TR ? PFD + 1 : 1], pu[TR ? PFD + 1 : 1];
        rv_i32x2 pm[(TR == 1 || (TR == 2 && !L11_MKB_NIB)) ? PFD + 1 : 1], pr[TR ? PFD + 1 : 1];
        uint32_t pmb[(TR == 2 && L11_MKB_NIB) ? PFD + 1 : 1];  // r16 iter53 NIB: packed keep-mask dword per slot
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
#if L11_MKB_NIB
            rM = make_rsrc(ca.MK + (size_t)row0 * 256, (uint32_t)rv * 256u);  // r16 iter53 NIB: uint8 [M,256]
#else
            rM = make_rsrc(ca.MK + (size_t)row0 * 512, (uint32_t)rv * 512u);
#endif
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
            if constexpr (TR == 2 && L11_MKB_NIB) {
                pmb[sl] = (uint32_t)l11_llvm_buffer_load_i32(rM, r * 256 + 4 * lane, 0, 0);  // r16 iter53 NIB
            } else {
                pm[sl] = l11_llvm_buffer_load_v2i32(rM, r * 512 + 8 * lane, 0, 0);
            }
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
#if L11DO8_DIET  // (89) DIET
              if (WLIST == 0 || it < nit) {  // DO8 WLIST: rows past the tile's valid rows skip the math (uniform; O7(i))
#else  // (89) DIET off: ship14 (c68395d)
              if (WLIST == 0 || it * 8 + w < rv) {  // DO8 WLIST: rows past the tile's valid rows skip the math (uniform)
#endif  // (89) DIET
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
#if L11_MKB_NIB
                const uint32_t mkn = pmb[sl];  // r16 iter53 NIB
#else
                const uint32_t mk0 = (uint32_t)pm[sl].x, mk1 = (uint32_t)pm[sl].y;
#endif
                float xh[8], dyn[8], dxm[8], dur[8], sv[8], dd[8];
                uint32_t dmx = 0u;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    dd[e] = sig_d(bfe8(pu[sl], e));
                    const uint32_t b = __builtin_bit_cast(uint32_t, dd[e]);
                    dmx = dmx > b ? dmx : b;
                }
                // r16 iter37 l11vd (C3): d = 1 + exp(.) is >= 1 or NaN, so uint32 order == !(d <= 2^126) per element
                // (exhaustive over all bf16 u: census verify2 test G); one compare instead of 8 v_cmp + s_or
                const bool bad = dmx > 0x7E800000u;
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
                float dymv[8], silv[8];
                // r16 iter37 l11vd (C1): keep ? RN(s*x) : +0 == RN(s * (x & mask)) for finite s > 0 (census verify D,
                // exhaustive); the masks are applied to the packed bf16 planes before the unpack
#if L11_MKB_NIB
                const i32x4 dyM = dyy & l11vd_mask_nib<0>(mkn), dxM = dx & l11vd_mask_nib<1>(mkn),
                            duM = du & l11vd_mask_nib<2>(mkn);  // r16 iter53 NIB (R2)
#else
                const uint64_t mpr = ((uint64_t)mk1 << 32) | (uint64_t)mk0;
                const i32x4 dyM = dyy & l11vd_mask<0>(mpr), dxM = dx & l11vd_mask<1>(mpr), duM = du & l11vd_mask<2>(mpr);
#endif
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float pdu = bfe8(duM, e), pdx = bfe8(dxM, e), pdy = bfe8(dyM, e);
                    const float xv = bfe8(px[sl], e), uv = bfe8(pu[sl], e);
                    const float dum = s * pdu;
                    dxm[e] = s * pdx;
                    const float dym = s * pdy;
                    const float xhat = (xv - mean) * rstd;
                    const float ln = __builtin_fmaf(xhat, bfe8(wq, e), bfe8(bq, e));
                    const float sig = sv[e];
                    const float silu = uv * sig;
                    const float oms = 1.0f - sig;
                    const float dsilu = __builtin_fmaf(silu, oms, sig);
                    dur[e] = __builtin_fmaf(dym * ln, dsilu, dum * dsilu);
                    dymv[e] = dym;
                    silv[e] = silu;
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
                        // r16 iter37 l11vd: operand order pinned (accumulator = src0, as compiled in the l11c_f1 build).
                        // With two NaN operands v_add_f32 returns src0's payload, and the C1 build otherwise commutes
                        // these adds (dx NaN payloads differed on rows with two NaN u in one lane: proto gate, enumu).
                        asm("v_add_f32_e32 %0, %1, %2" : "=v"(c2l) : "v"(c2l), "v"(wdy));
                        asm("v_add_f32_e32 %0, %1, %2" : "=v"(c1l) : "v"(c1l), "v"(pr1));
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
#pragma unroll
                for (int k = 0; k < 4; ++k)  // pdb = fma(dy_m, silu, pdb) as two IEEE fp32 FMAs per v_pk_fma_f32
                    pdb2[k] = __builtin_elementwise_fma(f32x2{dymv[2 * k], dymv[2 * k + 1]},
                                                        f32x2{silv[2 * k], silv[2 * k + 1]}, pdb2[k]);
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
                }
#pragma unroll
                for (int k = 0; k < 4; ++k)  // pdw = fma(xhat, dyn, pdw)
                    pdw2[k] = __builtin_elementwise_fma(f32x2{xh[2 * k], xh[2 * k + 1]}, f32x2{dyn[2 * k], dyn[2 * k + 1]},
                                                        pdw2[k]);
                // pin the partial chains here (else LLVM sinks the 8 rows' FMAs to the end of the block and keeps every
                // row's xhat/dyn live: first t5 build, 103-113 VGPR spilled)
                asm volatile("" : "+v"(pdw2[0]), "+v"(pdw2[1]), "+v"(pdw2[2]), "+v"(pdw2[3]));
                asm volatile("" : "+v"(pdb2[0]), "+v"(pdb2[1]), "+v"(pdb2[2]), "+v"(pdb2[3]));
                i32x4 ox;
                ox.x = (int)pk_bf16(f32x2{dxo[0], dxo[1]});
                ox.y = (int)pk_bf16(f32x2{dxo[2], dxo[3]});
                ox.z = (int)pk_bf16(f32x2{dxo[4], dxo[5]});
                ox.w = (int)pk_bf16(f32x2{dxo[6], dxo[7]});
                if constexpr (EM != 2) l11_llvm_buffer_store_v4i32(ox, rDX, r * 1024 + 16 * lane, 0, SCP ? 3 : 0);
                if constexpr (EM && TM) {
                    // DO8 TM: no running max in the consumer (register pressure); only stage the bf16 row
                    ds_write16<it * 8192>(xdx_addr(), ox);
                } else if constexpr (EM) {
                    // DO8: running group max of |d(attn)| on the fp32 dx (maxnum, NaN ignored; RNE_bf16 of the final
                    // group max == the max of |RNE_bf16(dx)|: RNE is monotone and odd), and keep the bf16 row
                    float a = PKA ? 0.f : ((it & 1) ? amg1 : amg0);
                    a = __builtin_fmaxf(__builtin_fmaxf(a, __builtin_fabsf(dxo[0])), __builtin_fabsf(dxo[1]));
                    a = __builtin_fmaxf(__builtin_fmaxf(a, __builtin_fabsf(dxo[2])), __builtin_fabsf(dxo[3]));
                    a = __builtin_fmaxf(__builtin_fmaxf(a, __builtin_fabsf(dxo[4])), __builtin_fabsf(dxo[5]));
                    a = __builtin_fmaxf(__builtin_fmaxf(a, __builtin_fabsf(dxo[6])), __builtin_fabsf(dxo[7]));
                    if constexpr (PKA) {
                        typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));
                        const unsigned short bb = (unsigned short)(pk_bf16(f32x2{a, a}) & 0xFFFFu);
                        const u16x2 nb = (it & 1) ? u16x2{(unsigned short)0, bb} : u16x2{bb, (unsigned short)0};
                        amgp = __builtin_bit_cast(uint32_t, __builtin_elementwise_max(__builtin_bit_cast(u16x2, amgp), nb));
                    } else if constexpr (it & 1) {
                        amg1 = a;
                    } else {
                        amg0 = a;
                    }
                    if constexpr (RET == 1) {
                        oxr[it] = ox;
                        asm volatile("" : "+v"(oxr[it]));
                    } else if constexpr (RET == 0) {
                        ds_write16<it * 8192>(xdxw, ox);  // row it*8 + w of the tile: xdxw + it * 8 KiB
                    }
                    if constexpr (PKA) asm volatile("" : "+v"(amgp));
                    else asm volatile("" : "+v"(amg0), "+v"(amg1));
                }
              }
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
        if constexpr (EM) {
            // ---- DO8 emission. Tile row r of wave w is it*8 + w, so group g = (r % 16) // 8 = it & 1 and every wave
            // holds 4 rows of each group: per (group, head) max = 16-lane DPP max (lane 15 of each 16-lane row = head
            // lane >> 4) -> per-wave partial in LDS -> one barrier -> 8-wave max.
#if L11DO8_DIET  // (89) DIET
            // r16 iter52 DIET O7(i): the tail's row tests (it < nitc) use an opaque copy of the tile's nit, so LLVM cannot
            // CSE them with the consumer's tests and keep 8 row-valid lane-mask pairs (s_cselect_b64, 16 SGPRs) live
            // across the row loop: each convert row re-tests nit against an inline constant instead (same value, same
            // branch outcome; no instruction is emitted for the copy).
            int nitc = nit;
            asm volatile("" : "+s"(nitc));
            const int hh = lane >> 4;
            float m0, m1;
            if constexpr (PKA && !TM) {
                // r16 iter52 DIET O1: the two group maxima stay ONE packed pair of bf16 bit patterns (lo = group 0,
                // hi = group 1) through the 16-lane DPP max, the LDS exchange and the 8-wave max (v_pk_max_u16 only).
                // Bitwise vs the fp32 path below: both halves are non-negative, non-NaN bf16 patterns in 0x0000..0x7F80
                // (RNE_bf16 of a maxnum of |x| that starts at +0, so never -0 and never NaN), on which u16 order is value
                // order; bound_ctrl's 0 fill is the max identity as it is for the fp32 max of v >= 0; and the fp32 path's
                // final RNE_bf16 is the identity on these bf16-exact values. One packed lane-15 LDS write per head (was 2
                // fp32 writes) and 8 packed reads (was 16 fp32); no fp32 max canonicalizations, no tail RNE converts.
                uint32_t ap = amgp;
                ap = dpp_pkmax_u16<0x118>(ap);
                ap = dpp_pkmax_u16<0x114>(ap);
                ap = dpp_pkmax_u16<0x112>(ap);
                ap = dpp_pkmax_u16<0x111>(ap);
                const uint32_t xch = lds0 + (uint32_t)XCHB;  // u32 [h][w]: packed (group 0, group 1) bf16 maxima
                if ((lane & 15) == 15) {
                    const uint32_t va = xch + (uint32_t)(hh * 8 + w) * 4u;
                    asm volatile("ds_write_b32 %0, %1" : : "v"(va), "v"(ap) : "memory");
                }
                wait_lgkm<0>();
                L11_BARRIER();
                // compiler-visible LDS reads after the barrier (as the fp32 path below; see its note on asm reads)
                const uint32_t* xcu = reinterpret_cast<const uint32_t*>(smem + XCHB) + hh * 8;
                // r16 iter52 DIET (combine v6x): the 8 packed words (2 x ds_read_b128) reduced as a 3-level tree whose
                // first level pairs word k of the first read with word k of the second, so no max can start before BOTH
                // reads returned and they issue back to back before ONE wait (V1..v6 compile read 1, wait, 3 maxes, read 2,
                // wait, 4 maxes: two serialised LDS round trips). Exact for every input: u16 max is associative and
                // commutative, and the chain's initial 0 is its identity.
                uint32_t xv[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) xv[k] = xcu[k];
                const uint32_t mp = pkmax_u16(pkmax_u16(pkmax_u16(xv[0], xv[4]), pkmax_u16(xv[1], xv[5])),
                                              pkmax_u16(pkmax_u16(xv[2], xv[6]), pkmax_u16(xv[3], xv[7])));
                L11_SB();
                m0 = bf_lo(mp);  // the same bf16 value the fp32 path's RNE_bf16 yields, exact in fp32
                m1 = bf_hi(mp);
            } else {
#endif  // (89) DIET
            float a0 = PKA ? bf_lo(amgp) : amg0, a1 = PKA ? bf_hi(amgp) : amg1;
            if constexpr (TM) {
                // DO8 TM: per-lane group maxima from this wave's staged bf16 rows (valid rows only; the stale rows of a
                // partial tile are skipped), maxnum on the bf16 values exactly as _blk_kernel's tl.max(|x|)
                static_assert(RET == 0, "TM reads the LDS staging");
                const i32x4* xrw = reinterpret_cast<const i32x4*>(smem + XDXB) + (w * 64 + lane);  // tail: short-lived
                a0 = 0.f;
                a1 = 0.f;
                static_for<8>([&](auto IT_) {
                    constexpr int it = decltype(IT_)::value;
#if L11DO8_DIET  // (89) DIET
                    if (it < nitc) {  // r16 iter52 DIET O7(i): == it * 8 + w < rv
#else  // (89) DIET off: ship14 (c68395d)
                    if (it * 8 + w < rv) {
#endif  // (89) DIET
                        const i32x4 x = xrw[it * 512];
                        float a = (it & 1) ? a1 : a0;
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            const uint32_t wd = (uint32_t)x[k];
                            a = __builtin_fmaxf(__builtin_fmaxf(a, __builtin_fabsf(bf_lo(wd))), __builtin_fabsf(bf_hi(wd)));
                        }
                        if constexpr (it & 1) a1 = a;
                        else a0 = a;
                    }
                });
            }
            a0 = dpp_max<0x118>(a0);
            a1 = dpp_max<0x118>(a1);
            a0 = dpp_max<0x114>(a0);
            a1 = dpp_max<0x114>(a1);
            a0 = dpp_max<0x112>(a0);
            a1 = dpp_max<0x112>(a1);
            a0 = dpp_max<0x111>(a0);
            a1 = dpp_max<0x111>(a1);
#if !L11DO8_DIET  // (89) DIET off: ship14 (c68395d)
            const int hh = lane >> 4;
#endif  // (89) DIET
            const uint32_t xch = lds0 + (uint32_t)XCHB;  // fp32 [g][h][w]
            if ((lane & 15) == 15) {
                const uint32_t va = xch + (uint32_t)(hh * 8 + w) * 4u;
                asm volatile("ds_write_b32 %0, %1" : : "v"(va), "v"(a0) : "memory");
                asm volatile("ds_write_b32 %0, %1 offset:128" : : "v"(va), "v"(a1) : "memory");
            }
            wait_lgkm<0>();
            L11_BARRIER();
            // (compiler-visible LDS reads: the bt2 build read these with 4 asm ds_read_b128 and LLVM kept only element
            // .x of each -- wrong group max; plain loads after the barrier fences instead)
            const float* xcf = reinterpret_cast<const float*>(smem + XCHB) + hh * 8;
#if L11DO8_DIET  // (89) DIET
            float fm0 = 0.f, fm1 = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                fm0 = __builtin_fmaxf(fm0, xcf[k]);
                fm1 = __builtin_fmaxf(fm1, xcf[32 + k]);
            }
            L11_SB();
            m0 = bf_lo(pk_bf16(f32x2{fm0, fm0}));  // RNE_bf16 of the group max (exact back to fp32)
            m1 = bf_lo(pk_bf16(f32x2{fm1, fm1}));
            }
            int e0 = grp_exp(m0), e1 = grp_exp(m1);  // (non-const: pinned below, iter36 C-B4)
            // r16 iter52 DIET O2 + E8 window (combine v6 = v5 + O2 in iter52 v2b's rolling 4+4 reads). For RET 0 / EMV 2 / TM 0
            // (the production family, O2SL) the 8-row convert loop below is straight-line, with no per-row validity
            // test. Its staging reads are COMPILER-VISIBLE LDS loads (plain loads, as the exchange reads above), so the
            // waitcnt pass places the progressive lgkmcnt(N) waits before each row's converts itself.
            // Rows 0-3 are read HERE; row k + 4 is read right after row k's 4 converts, into the 4 VGPRs row k just
            // released (each late read has 3 convert rounds of cover and sits inside the MODE[23] = 1 window, as all 8
            // reads of the guarded loop always did: a ds_read_b128 has no FP semantics). At most 16 VGPRs of staging
            // data are in flight.
            // Every LGKM op in flight here is compiler-visible: the last asm LDS op (the lane-15 exchange write) was
            // drained by wait_lgkm<0> before the exchange barrier. The LDS pointer is formed from xdxw itself (the
            // consumer's loop-resident staging-write VGPR), so no second copy of the address stays live.
            // Rows >= rv: the read returns a stale staging row (the 64 KiB staging region is always in bounds) and the dO8
            // store offset r * 512 + 8 * lane >= rv * 512 = rD8's num_records, so the buffer range check drops it (as RET 2
            // and E8 already rely on); rows < rv: the same bytes, scales, converts and addresses as before. The 9 emission
            // stores are now always issued, which is what S_ST counts.
            // E8 (O2SL only) is built and stored right after the staging reads ISSUE and before the scale pin and the
            // MODE[23] = 1 setreg, so its readlane / SALU chain and its store ride the reads' LDS latency (iter50 REVIEW
            // item 5, iter52 census what-if e8win). Same bytes and addresses as base's E8 after the converts: it reads
            // e0/e1 before the empty pin asm, which does not change them, and it is integer-only (MODE[23] cannot apply).
            // Every other family keeps E8 after the converts, as base.
            constexpr bool O2SL = (RET == 0 && EMV == 2 && !TM);
            i32x4 xs[O2SL ? 4 : 1];
            typedef __attribute__((address_space(3))) const i32x4 lds_ci32x4_t;
            const lds_ci32x4_t* xrw = (const lds_ci32x4_t*)(uintptr_t)xdxw;  // row it at xrw[it * 512] = +it * 8 KiB
            if constexpr (O2SL) {
                static_for<4>([&](auto IT_) {
                    constexpr int it = decltype(IT_)::value;
                    xs[it] = xrw[it * 512];  // row it*8 + w
                });
                L11_SB();  // the 4 reads issue before the E8 chain below, which then rides their latency
                // E8 [M,4]: row r -> bytes (127 - e_g(h)) for h = 0..3, g = (r % 16) // 8; lane j < 8 stores row 8j + w
                const uint32_t b0 = (uint32_t)((127 - __builtin_amdgcn_readlane(e0, 0)) & 0xFF) |
                                    ((uint32_t)((127 - __builtin_amdgcn_readlane(e0, 16)) & 0xFF) << 8) |
                                    ((uint32_t)((127 - __builtin_amdgcn_readlane(e0, 32)) & 0xFF) << 16) |
                                    ((uint32_t)((127 - __builtin_amdgcn_readlane(e0, 48)) & 0xFF) << 24);
                const uint32_t b1 = (uint32_t)((127 - __builtin_amdgcn_readlane(e1, 0)) & 0xFF) |
                                    ((uint32_t)((127 - __builtin_amdgcn_readlane(e1, 16)) & 0xFF) << 8) |
                                    ((uint32_t)((127 - __builtin_amdgcn_readlane(e1, 32)) & 0xFF) << 16) |
                                    ((uint32_t)((127 - __builtin_amdgcn_readlane(e1, 48)) & 0xFF) << 24);
                const i32x4 rE8 = make_rsrc(ca.E8 + (size_t)row0 * 4, (uint32_t)rv * 4u);
                const float bv = __builtin_bit_cast(float, (lane & 1) ? b1 : b0);
                l11_llvm_buffer_store_f32(bv, rE8, lane < 8 ? (8 * lane + w) * 4 : 0x40000000, 0, 0);
            }
#else  // (89) DIET off: ship14 (c68395d)
            float m0 = 0.f, m1 = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                m0 = __builtin_fmaxf(m0, xcf[k]);
                m1 = __builtin_fmaxf(m1, xcf[32 + k]);
            }
            L11_SB();
            m0 = bf_lo(pk_bf16(f32x2{m0, m0}));  // RNE_bf16 of the group max (exact back to fp32)
            m1 = bf_lo(pk_bf16(f32x2{m1, m1}));
            int e0 = grp_exp(m0), e1 = grp_exp(m1);  // (non-const: pinned below, iter36 C-B4)
#endif  // (89) DIET
            const f32x2 s20 = {__builtin_bit_cast(float, (e0 + 127) << 23), __builtin_bit_cast(float, (e0 + 127) << 23)};
            const f32x2 s21 = {__builtin_bit_cast(float, (e1 + 127) << 23), __builtin_bit_cast(float, (e1 + 127) << 23)};
            const i32x4 rD8 = make_rsrc(ca.DO8 + (size_t)row0 * 512, (uint32_t)rv * 512u);
            i32x4 xg[(RET == 2) ? 8 : 1];
            if constexpr (RET == 2) {  // re-read the bf16 rows this wave just stored (same lanes, same addresses; L2-hot)
                static_for<8>([&](auto I_) {
                    constexpr int i = decltype(I_)::value;
                    xg[i] = l11_llvm_buffer_load_v4i32(rDX, (i * 8 + w) * 1024 + 16 * lane, 0, 0);
                });
            }
            // EMV >= 1: MODE[23] = 1 once for the tile's whole row loop (only the e4m3 converts below run inside it);
#if L11DO8_DIET  // (89) DIET
            // the per-row branch structure of EMV 0 is kept (straight-line variants crashed the greedy RA in bc4/bct5),
            // except for the O2SL family (r16 iter52 DIET O2, combine v6: straight-line, rolling 4-row read window)
#else  // (89) DIET off: ship14 (c68395d)
            // the per-row branch structure of EMV 0 is kept (straight-line variants crashed the greedy RA in bc4/bct5)
#endif  // (89) DIET
            //   EMV 1: x_bf16 -> fp32, x * 2^e (v_pk_mul_f32), v_cvt_scalef32_pk_fp8_f32 (scale 1.0)
            //   EMV 2/3: v_cvt_scalef32_pk_fp8_bf16 straight from the packed bf16 words, scale 2^-e / 2^+e
            float cs0 = __builtin_bit_cast(float, (127 - e0) << 23);  // EMV 2/3: 2^-e (the convert divides)
            float cs1 = __builtin_bit_cast(float, (127 - e1) << 23);
            // r16 iter36 do8e5 (iter-35 review C-B4): pin the group exponents and scales BEFORE MODE[23] is set. In V9's
            // and V20's ISA the compiler had placed group 1's v_max3_f32 x4 and its RNE v_cvt_pk_bf16_f32 after the
            // setreg (harmless today: e clamps to -60 either way, but compiler placement, not construction). The asm
            // statements keep their order and their outputs are opaque, so the maxima, the RNE and _grp_exp execute
            // with MODE[23] = 0 and nothing after the pin recomputes them.
            if constexpr (EMV >= 1) asm volatile("" : "+v"(e0), "+v"(e1), "+v"(cs0), "+v"(cs1));
            if constexpr (EMV >= 1) asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n\ts_nop 3" ::: "memory");
            static_for<8>([&](auto IT_) {
                constexpr int it = decltype(IT_)::value;
                const int r = it * 8 + w;
#if L11DO8_DIET  // (89) DIET
                // RET 2: straight-line (rows >= rv read 0 and their stores are range-dropped). O7(i): it < nitc == r < rv.
                // LLVM proves !(it < nitc) => !(it + 1 < nitc) and turns the 8 tests into an early-exit chain; without
                // the hint each exit is 50/50, so row k's block looks 2^-k cold and the register allocator split and
                // spilled a dO8 store offset there (SDK10 noSLP 'only': 1 VGPR spill + a vmcnt(0) reload in row 5,
                // r16 iter52 v3 ablation v3h0). Valid rows are the common case (63.2 of 64 rows per tile at ts73).
                // O2SL (combine v6/v6f): straight-line (rows >= rv convert a stale staging row; their stores are range-dropped)
                if (RET == 2 || O2SL || __builtin_expect(it < nitc, 1)) {
#else  // (89) DIET off: ship14 (c68395d)
                if (RET == 2 || r < rv) {  // RET 2: straight-line (rows >= rv read 0 and their stores are range-dropped)
#endif  // (89) DIET
                    i32x4 x;
                    if constexpr (RET == 1) {
                        x = oxr[it];
                    } else if constexpr (RET == 2) {
                        x = xg[it];
#if L11DO8_DIET  // (89) DIET
                    } else if constexpr (O2SL) {
                        x = xs[it & 3];  // row it: read before the setreg (it < 4) or after row it - 4's converts
#endif  // (89) DIET
                    } else {
                        x = ds_read16<it * 8192>(TM ? xdx_addr() : xdxw);
                        wait_lgkm<0>();
                    }
                    int q0, q1;
                    if constexpr (EMV == 2) {
                        // bct6: the scale operand DIVIDES (2^+e gave ~0) and op_sel:[0,0,0,1] did not select the high
                        // destination word of this 2-source VOP3 -> 2^-e and op_sel:[0,0,1] (dst bit is the 3rd entry)
                        const float cs = (it & 1) ? cs1 : cs0;
                        asm volatile(
                            "v_cvt_scalef32_pk_fp8_bf16 %0, %2, %6\n\t"
                            "v_cvt_scalef32_pk_fp8_bf16 %0, %3, %6 op_sel:[0,0,1]\n\t"
                            "v_cvt_scalef32_pk_fp8_bf16 %1, %4, %6\n\t"
                            "v_cvt_scalef32_pk_fp8_bf16 %1, %5, %6 op_sel:[0,0,1]"
                            : "=&v"(q0), "=&v"(q1)
                            : "v"(x.x), "v"(x.y), "v"(x.z), "v"(x.w), "v"(cs));
#if L11DO8_DIET  // (89) DIET
                        // v6: row it + 4 into the slot row it just released (after its converts: an asm volatile is a
                        // memory barrier for the scheduler, so the read cannot be hoisted above them)
                        if constexpr (O2SL && it < 4) xs[it] = xrw[(it + 4) * 512];
#endif  // (89) DIET
                    } else if constexpr (EMV == 3) {
                        // the same through the clang builtin (encoding by the compiler), scale 2^-e
                        const float cs = (it & 1) ? cs1 : cs0;
                        s16x2 w0 = {0, 0}, w1 = {0, 0};
                        w0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(w0, __builtin_bit_cast(bf16x2, x.x), cs, false);
                        w0 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(w0, __builtin_bit_cast(bf16x2, x.y), cs, true);
                        w1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(w1, __builtin_bit_cast(bf16x2, x.z), cs, false);
                        w1 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(w1, __builtin_bit_cast(bf16x2, x.w), cs, true);
                        asm volatile("" : "+v"(w0), "+v"(w1));
                        q0 = __builtin_bit_cast(int, w0);
                        q1 = __builtin_bit_cast(int, w1);
                    } else {
                    const f32x2 sg = (it & 1) ? s21 : s20;
                    f32x2 f[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t wd = (uint32_t)x[k];
                        f[k] = f32x2{bf_lo(wd), bf_hi(wd)} * sg;  // x * 2^e (IEEE fp32 multiplies, as _blk_kernel)
                    }
                    if constexpr (EMV == 1) {
                        asm volatile(
                            "v_cvt_scalef32_pk_fp8_f32 %0, %2, %3, 1.0\n\t"
                            "v_cvt_scalef32_pk_fp8_f32 %0, %4, %5, 1.0 op_sel:[0,0,0,1]\n\t"
                            "v_cvt_scalef32_pk_fp8_f32 %1, %6, %7, 1.0\n\t"
                            "v_cvt_scalef32_pk_fp8_f32 %1, %8, %9, 1.0 op_sel:[0,0,0,1]"
                            : "=&v"(q0), "=&v"(q1)
                            : "v"(f[0].x), "v"(f[0].y), "v"(f[1].x), "v"(f[1].y), "v"(f[2].x), "v"(f[2].y),
                              "v"(f[3].x), "v"(f[3].y));
                    } else
                    // MODE[23] = 1 only around the e4m3 converts (as _blk_kernel's prologue sets it for its whole body)
                    asm volatile(
                        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n\t"
                        "s_nop 3\n\t"
                        "v_cvt_scalef32_pk_fp8_f32 %0, %2, %3, 1.0\n\t"
                        "v_cvt_scalef32_pk_fp8_f32 %0, %4, %5, 1.0 op_sel:[0,0,0,1]\n\t"
                        "v_cvt_scalef32_pk_fp8_f32 %1, %6, %7, 1.0\n\t"
                        "v_cvt_scalef32_pk_fp8_f32 %1, %8, %9, 1.0 op_sel:[0,0,0,1]\n\t"
                        "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\n\t"
                        "s_nop 3"
                        : "=&v"(q0), "=&v"(q1)
                        : "v"(f[0].x), "v"(f[0].y), "v"(f[1].x), "v"(f[1].y), "v"(f[2].x), "v"(f[2].y), "v"(f[3].x),
                          "v"(f[3].y));
                    }
                    l11_llvm_buffer_store_v2i32(rv_i32x2{q0, q1}, rD8, r * 512 + 8 * lane, 0, 0);
                }
            });
            if constexpr (EMV >= 1) asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\n\ts_nop 3" ::: "memory");
#if L11DO8_DIET  // (89) DIET
            if constexpr (!O2SL) {  // r16 iter52 DIET (combine): O2SL stored E8 in the reads' window above
#endif  // (89) DIET
            // E8 [M,4]: row r -> bytes (127 - e_g(h)) for h = 0..3, g = (r % 16) // 8; lane j < 8 stores row 8j + w
            const uint32_t b0 = (uint32_t)((127 - __builtin_amdgcn_readlane(e0, 0)) & 0xFF) |
                                ((uint32_t)((127 - __builtin_amdgcn_readlane(e0, 16)) & 0xFF) << 8) |
                                ((uint32_t)((127 - __builtin_amdgcn_readlane(e0, 32)) & 0xFF) << 16) |
                                ((uint32_t)((127 - __builtin_amdgcn_readlane(e0, 48)) & 0xFF) << 24);
            const uint32_t b1 = (uint32_t)((127 - __builtin_amdgcn_readlane(e1, 0)) & 0xFF) |
                                ((uint32_t)((127 - __builtin_amdgcn_readlane(e1, 16)) & 0xFF) << 8) |
                                ((uint32_t)((127 - __builtin_amdgcn_readlane(e1, 32)) & 0xFF) << 16) |
                                ((uint32_t)((127 - __builtin_amdgcn_readlane(e1, 48)) & 0xFF) << 24);
            const i32x4 rE8 = make_rsrc(ca.E8 + (size_t)row0 * 4, (uint32_t)rv * 4u);
            const float bv = __builtin_bit_cast(float, (lane & 1) ? b1 : b0);
            l11_llvm_buffer_store_f32(bv, rE8, lane < 8 ? (8 * lane + w) * 4 : 0x40000000, 0, 0);
#if L11DO8_DIET  // (89) DIET
            }
#endif  // (89) DIET
        }
        if constexpr (WLIST) {
            cur0 = nx0;
            curv = nxv;
        }
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
        if (l11vd_lane() == 0) ca.AMAXP[blockIdx.x * 8 + w] = bf_lo(pk_bf16(f32x2{wm, wm}));
        // rows [M, rows_out) of the u block: zero (as v3's tail loop and the shipped cast write them); one wave per row
        for (int pr = M + (int)blockIdx.x * 8 + w; pr < ca.rows_out; pr += G * 8)
            *reinterpret_cast<rv_i32x2*>(ca.DZ8 + (size_t)pr * 2048u + 8u * (size_t)l11vd_lane()) = rv_i32x2{0, 0};
    }
    wait_vm<0>();  // drain the trailing ring prefetches before the WG's LDS is released
    if constexpr (STUB == 1) {
        if (sav == 12345.0f) {  // never true for real scales: keeps the u/x-plane consumption alive (diagnostic idiom, r14)
            const i32x4 rC = make_rsrc(C, 64u * 4u);
            l11_llvm_buffer_store_f32(__builtin_bit_cast(float, chk), rC, l11vd_lane() * 4, 0, 0);
        }
    }
#endif  // L11DO8_BODY_PASS (end of the kernel body)
#ifndef L11DO8_BODY_PASS



// =====================================================================================================================
// r16 iter36 do8e5: DO8 production host ABI. This .so carries ONLY the DO8 kernels below; L11's own .so
// (l11_outproj_lnmd.cu) and its ABI are untouched. extern "C", raw device pointers, L11's return convention:
// 0 = launched; > 0 = hipError_t of a rejected argument or a failed launch (outputs not written); < 0 = launched, and -rc
// is a stale error an earlier call left on this thread (drained here, not ours).
// =====================================================================================================================
#ifndef L11DO8_SRC_SHA256
#define L11DO8_SRC_SHA256 "unknown"
#endif
// r16 iter42 do8only: 1 = the 'only' .so (l11do8_e5_only_launch, EM = 2); 0 (default) = the 'plus' .so, unchanged
#ifndef L11DO8_ONLY
#define L11DO8_ONLY 0
#endif
// r16 iter56 MX6: 1 = this .so also carries the MX = 1 kernels (ABI 2: + l11do8_e5_{only,plus}_mx6_launch per L11DO8_ONLY,
// l11do8_mx6_dy_debug_launch, l11do8_fp8_dy_debug_launch; every ABI-1 export unchanged); 0 (default) = exactly the base
#ifndef L11DO8_MX6
#define L11DO8_MX6 0
#endif
#if L11DO8_MX6
#define L11DO8_ABI_VERSION 2
#define L11DO8_MX6_TAG "+mx6(e2m3xe2m3,row400,opsel_k)"
#else
#define L11DO8_ABI_VERSION 1
#define L11DO8_MX6_TAG ""
#endif
static constexpr long long L11DO8_M_MAX = 4194304;  // = L11's validated M bound (l11_outproj_lnmd.py M_MAX)
static constexpr int L11DO8_B_MAX = 8192;           // sequences per list build: 1024 threads x 8
static constexpr int L11DO8_WL_THREADS = 1024;
static constexpr int L11DO8_WL_KMAX = 8;

// ---- DO8 device work list (the pattern of x4c19's _blk_wl_build_kernel, ops/flydsl/hstu_do_fp8_wl.py: programs over
// the tile index t, every program scans all B offsets (a block prefix sum), sequence-major entries, no host sync, no
// atomics, deterministic). Entries are int32 pairs (start row, valid rows 1..64), in this order:
//   [0, H0)            head gap: rows [0, o_0) in 64-row tiles (none when o_0 == 0, the production case);
//   [H0, H0 + S)       sequence b's tiles t = 0 .. cdiv(L_b, 64) - 1 at H0 + excl_b + t (b ascending):
//                      (o_b + 64 t, min(64, L_b - 64 t)) = the 64-token tiles of quantize_do_e4m3_blk's group rule, for
//                      EVERY tile of the sequence (also t >= cdiv(N, 64), rows the stock quantizer leaves unwritten);
//   [H0 + S, total)    tail gap: rows [o_B, M) in 64-row tiles (none when o_B == M, the production case).
// Every row of [0, M) is in exactly one entry, so the DO8 kernel writes all of L11's outputs for all M rows whatever the
// offsets. CNT = {total, 0}, total <= cap. Malformed offsets (o_0 < 0, a negative length, o_B > M, total > cap): the
// plain 64-row tiles of [0, M) instead, CNT = {cdiv(M, 64), 1}: L11's outputs stay exact; dO8/E8 then follow those tiles
// (the attention reads the same malformed offsets, so no consumer result is defined there).
template <typename IT>
__global__ __launch_bounds__(1024) void l11_do8_wl_build(const IT* __restrict__ offs, int B, int M, long long cap,
                                                         int* __restrict__ wl, int* __restrict__ cnt) {
    constexpr int NTH = L11DO8_WL_THREADS, KMAX = L11DO8_WL_KMAX, NWV = L11DO8_WL_THREADS / 64;
    __shared__ int s_sum[NWV];
    __shared__ int s_max[NWV];
    __shared__ int s_bad;
    const int tid = (int)threadIdx.x, lane = tid & 63, wv = tid >> 6;
    if (tid == 0) s_bad = 0;
    __syncthreads();
    const int K = (B + NTH - 1) / NTH;  // sequences per thread (B <= 8192 -> K <= 8), one contiguous chunk per thread
    const int b0 = tid * K;
    int ost[KMAX], len[KMAX], ntl[KMAX];
    int loc = 0, mx = 0, bad = 0;
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        ost[k] = 0;
        len[k] = 0;
        ntl[k] = 0;
        const int b = b0 + k;
        if (k < K && b < B) {
            const long long lo = (long long)offs[b], hi = (long long)offs[b + 1];
            if (lo < 0 || hi < lo || hi > (long long)M) {
                bad = 1;
            } else {
                ost[k] = (int)lo;
                len[k] = (int)(hi - lo);
                ntl[k] = (len[k] + 63) >> 6;
            }
            loc += ntl[k];
            mx = max(mx, ntl[k]);
        }
    }
    // block exclusive prefix of the per-thread tile counts (wave Hillis-Steele scan + the 16 wave totals) and the block
    // maximum of the per-sequence tile count
    int inc = loc;
#pragma unroll
    for (int d = 1; d < 64; d <<= 1) {
        const int y = __shfl_up(inc, (unsigned int)d, 64);
        if (lane >= d) inc += y;
    }
    int wmx = mx;
#pragma unroll
    for (int d = 32; d >= 1; d >>= 1) wmx = max(wmx, __shfl_xor(wmx, d, 64));
    if (lane == 63) s_sum[wv] = inc;
    if (lane == 0) s_max[wv] = wmx;
    if (bad) s_bad = 1;
    __syncthreads();
    int wpre = 0, tot = 0, tmax = 0;
#pragma unroll
    for (int i = 0; i < NWV; ++i) {
        const int s = s_sum[i];
        wpre += i < wv ? s : 0;
        tot += s;
        tmax = max(tmax, s_max[i]);
    }
    const long long o0 = (long long)offs[0], oB = (long long)offs[B];
    const bool gbad = s_bad != 0 || o0 < 0 || oB < o0 || oB > (long long)M;
    const int H0 = gbad ? 0 : (int)((o0 + 63) >> 6);
    const int T1 = gbad ? 0 : (int)(((long long)M - oB + 63) >> 6);
    const long long total = (long long)H0 + (long long)tot + (long long)T1;
    rv_i32x2* out = reinterpret_cast<rv_i32x2*>(wl);
    if (gbad || total > cap) {
        const int np = (M + 63) >> 6;  // <= cap (the launcher requires cap >= cdiv(M, 64) + B + 2)
        for (int i = (int)blockIdx.x * NTH + tid; i < np; i += (int)gridDim.x * NTH)
            out[i] = rv_i32x2{64 * i, min(64, M - 64 * i)};
        if (blockIdx.x == 0 && tid == 0) {
            cnt[0] = np;
            cnt[1] = 1;
        }
        return;
    }
    const int excl = wpre + inc - loc;  // tiles of all sequences before this thread's chunk
    for (int t = (int)blockIdx.x; t < tmax; t += (int)gridDim.x) {
        int pre = H0 + excl;
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            if (k < K && ntl[k] > t) out[pre + t] = rv_i32x2{ost[k] + 64 * t, min(64, len[k] - 64 * t)};
            pre += ntl[k];
        }
    }
    if (blockIdx.x == 0) {
        for (int i = tid; i < H0; i += NTH) out[i] = rv_i32x2{64 * i, min(64, (int)o0 - 64 * i)};
        for (int i = tid; i < T1; i += NTH) {
            const int st = (int)oB + 64 * i;
            out[H0 + tot + i] = rv_i32x2{st, min(64, M - st)};
        }
        if (tid == 0) {
            cnt[0] = (int)total;
            cnt[1] = 0;
        }
    }
}

// CU count of the first device this process launches on (as l11_ncu: one GPU per process in production).
static int l11do8_ncu() {
    static int n = 0;
    if (n == 0) {
        int dev = 0, c = 0;
        if (hipGetDevice(&dev) != hipSuccess) return 256;
        if (hipDeviceGetAttribute(&c, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || c <= 0) c = 256;
        n = c;
    }
    return n;
}

// List capacity for M rows and B sequences: head gap + sum_b cdiv(L_b, 64) + tail gap
//   <= (o_0 + 63)/64 + sum_b (L_b + 63)/64 + (M - o_B + 63)/64 <= M/64 + 63 B/64 + 2 <= cdiv(M, 64) + B + 2.
static long long l11do8_cap_impl(long long M, long long B) { return (M + 63) / 64 + B + 2; }

// Persistent grid for a list of at most `cap` tiles (L11's rule with nblk -> cap): G = min(cap, #CU) WGs of 8 waves,
// per = ceil(cap / G) tiles per WG at most, a dW / dB / u-colsum partial flush every fl = ceil(per / 8) tiles, nf =
// ceil(per / fl) <= 8 slots per wave: P = G*8*nf <= 16384 partial rows. A device count below cap only leaves trailing
// slots zero (the kernel writes every partial row) and WGs without a tile write zero partials.
static int l11do8_plan_impl(long long cap, long long* G, long long* fl, long long* nf) {
    if (cap <= 0 || cap > (L11DO8_M_MAX + 63) / 64 + L11DO8_B_MAX + 2) return (int)hipErrorInvalidValue;
    const long long ncu = (long long)l11do8_ncu();
    *G = cap < ncu ? cap : ncu;
    const long long per = (cap + *G - 1) / *G;
    *fl = (per + 7) / 8;
    *nf = (per + *fl - 1) / *fl;
    return 0;
}

static inline bool l11do8_a16(const void* p) { return p != nullptr && ((uintptr_t)p & 15u) == 0; }
static inline bool l11do8_a8(const void* p) { return p != nullptr && ((uintptr_t)p & 7u) == 0; }
static inline bool l11do8_a4(const void* p) { return p != nullptr && ((uintptr_t)p & 3u) == 0; }

extern "C" int l11do8_abi_version(void) { return L11DO8_ABI_VERSION; }

// r16 iter54 NIB: the packed keep-mask build (L11_MKB_NIB = 1) tags BOTH impl strings "+mkb=nib" (l11_do8.py
// mask_format() reads it); with L11_MKB_NIB = 0 the tag is empty and both strings are byte-identical to the v6x ones.
#if L11_MKB_NIB
#define L11DO8_MKB_TAG "+mkb=nib"
#else
#define L11DO8_MKB_TAG ""
#endif
extern "C" const char* l11do8_impl(void) {
#if L11DO8_ONLY
    return "l11_outproj_lnmd_bwd_do8 abi=1 variant=e5only(E5 1,WLIST 2 device list,EM 2 only no bf16 DX store,"
           "RET 0 LDS staging,EMV 2 bf16 convert,packed running max)+vd(C1,C2,C3,C4,LANEFIX,PIN)+l11_do8_wl_build"
           L11DO8_MKB_TAG " src_sha256=" L11DO8_SRC_SHA256 L11DO8_MX6_TAG;
#else
    return "l11_outproj_lnmd_bwd_do8 abi=1 variant=e5plus(E5 1,WLIST 2 device list,EM 1 plus,RET 0 LDS staging,"
           "EMV 2 bf16 convert,packed running max)+vd(C1,C2,C3,C4,LANEFIX,PIN)+l11_do8_wl_build"
           L11DO8_MKB_TAG " src_sha256=" L11DO8_SRC_SHA256 L11DO8_MX6_TAG;
#endif
}

extern "C" int l11do8_drain_last_error(void) { return (int)hipGetLastError(); }

// The list capacity (entries) for M rows and B sequences, or -1 outside [1, M_MAX] x [0, B_MAX].
extern "C" long long l11do8_wl_cap(long long M, long long B) {
    if (M <= 0 || M > L11DO8_M_MAX || B < 0 || B > L11DO8_B_MAX) return -1;
    return l11do8_cap_impl(M, B);
}

// out4 = {G, fl, nf, P} for a list of at most cap tiles. Returns 0, or hipErrorInvalidValue.
extern "C" int l11do8_plan(long long cap, long long* out4) {
    if (out4 == nullptr) return (int)hipErrorInvalidValue;
    long long G, fl, nf;
    const int rc = l11do8_plan_impl(cap, &G, &fl, &nf);
    if (rc != 0) return rc;
    out4[0] = G;
    out4[1] = fl;
    out4[2] = nf;
    out4[3] = G * 8 * nf;
    return 0;
}

// Build the work list of M rows from OFFS [B+1] (int64 if offs_i64 else int32; the attention's seq_offsets tensor) into
// WL int32 [cap, 2] and CNT int32 [2], on `stream`, with nprog programs of 1024 threads (program p takes tile indices
// p, p + nprog, ...). cap must be >= l11do8_wl_cap(M, B).
extern "C" int l11do8_wl_build_launch(const void* OFFS, int offs_i64, int B, int M, int nprog, void* WL, long long cap,
                                      void* CNT, void* stream) {
    if (M <= 0 || (long long)M > L11DO8_M_MAX || B < 0 || B > L11DO8_B_MAX) return (int)hipErrorInvalidValue;
    if (cap < l11do8_cap_impl(M, B) || nprog < 1 || nprog > 1024) return (int)hipErrorInvalidValue;
    if (!(offs_i64 ? l11do8_a8(OFFS) : l11do8_a4(OFFS)) || !l11do8_a8(WL) || !l11do8_a4(CNT))
        return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();
    if (offs_i64) {
        hipLaunchKernelGGL((l11_do8_wl_build<long long>), dim3((unsigned)nprog), dim3(L11DO8_WL_THREADS), 0,
                           (hipStream_t)stream, (const long long*)OFFS, B, M, cap, (int*)WL, (int*)CNT);
    } else {
        hipLaunchKernelGGL((l11_do8_wl_build<int>), dim3((unsigned)nprog), dim3(L11DO8_WL_THREADS), 0,
                           (hipStream_t)stream, (const int*)OFFS, B, M, cap, (int*)WL, (int*)CNT);
    }
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

// The DO8 E5 'plus' launch. Contract = l11_outproj_lnmd_bwd_e5_launch (L11's E5 mode: the same operands, DZ8 / INVS /
// AMAXP / DWP / DBP / DUCS / dscale / invD meaning, AMAXP [G*8] and the [P,512] partials sized from l11do8_plan(cap))
// plus: WL / CNT = the device work list of l11do8_wl_build_launch (same stream, cap entries); DO8 e4m3 [M,512] and E8
// uint8 [M,4] (16-B aligned): dO8 / E8 of every listed tile (= quantize_do_e4m3_blk(DX, seq_offsets, N) on every row the
// quantizer writes). DX bf16 [M,512] is written ('plus').
// r16 iter42 do8only: the 'only' .so (L11DO8_ONLY = 1) exports this same entry as l11do8_e5_only_launch, which launches
// the EM = 2 instantiation: the same checks and outputs except DX, which is NOT written (the caller passes its bf16
// placeholder, 16-B aligned as for 'plus'; its bytes are left as they are).
#if L11DO8_ONLY
#define L11DO8_E5_LAUNCH l11do8_e5_only_launch
#else
#define L11DO8_E5_LAUNCH l11do8_e5_plus_launch
#endif
extern "C" int L11DO8_E5_LAUNCH(const void* G8, const void* W8, const float* sa, const float* sb, int M,
                                     const void* X, const void* U, const void* MK, const float* MEAN, const float* RSTD,
                                     const void* WN, const void* BN, void* DX, void* DZ8, int rows_out,
                                     const float* INVS, void* AMAXP, void* DWP, void* DBP, void* DUCS, long long P,
                                     float dscale, float invD, const void* WL, const void* CNT, long long cap,
                                     void* DO8, void* E8, void* stream) {
    long long G, fl, nf;
    if (M <= 0 || (long long)M > L11DO8_M_MAX || cap < ((long long)M + 63) / 64) return (int)hipErrorInvalidValue;
    if (l11do8_plan_impl(cap, &G, &fl, &nf) != 0) return (int)hipErrorInvalidValue;
    if (P != G * 8 * nf) return (int)hipErrorInvalidValue;
    // r16 iter37 l11vd (C1): the masked-before-scaling dropout is exact only for a finite dscale > 0 (prod: 1/(1-p))
    if (!(dscale > 0.0f && dscale <= 3.40282346638528859811704183484516925e+38f)) return (int)hipErrorInvalidValue;
    if (rows_out < M || (long long)rows_out >= (long long)M + 4096) return (int)hipErrorInvalidValue;
    if (!(l11do8_a16(G8) && l11do8_a16(W8) && l11do8_a16(X) && l11do8_a16(U) && l11do8_a16(MK) && l11do8_a16(WN) &&
          l11do8_a16(BN) && l11do8_a16(DX) && l11do8_a16(DZ8) && l11do8_a16(DWP) && l11do8_a16(DBP) &&
          l11do8_a16(DUCS) && l11do8_a16(DO8) && l11do8_a16(E8) && l11do8_a8(WL)))
        return (int)hipErrorInvalidValue;
    if (!(l11do8_a4(sa) && l11do8_a4(sb) && l11do8_a4(MEAN) && l11do8_a4(RSTD) && l11do8_a4(INVS) &&
          l11do8_a4(AMAXP) && l11do8_a4(CNT)))
        return (int)hipErrorInvalidValue;
    L11CArgs ca{};
    ca.X = (const uint16_t*)X;
    ca.U = (const uint16_t*)U;
    ca.MK = (const uint8_t*)MK;
    ca.MEAN = MEAN;
    ca.RSTD = RSTD;
    ca.WN = (const uint16_t*)WN;
    ca.BN = (const uint16_t*)BN;
    ca.DX = (uint16_t*)DX;
    ca.DU = (uint16_t*)DZ8;  // the E5 kernel addresses the u block through ca.DU (byte rows of 2048), DZ8 for pad rows
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
    ca.WLP = (const int*)WL;
    ca.DO8 = (uint8_t*)DO8;
    ca.E8 = (uint8_t*)E8;
    ca.WLCNT = (const int*)CNT;
    const hipError_t stale = hipGetLastError();
    // nblk = cap: a placeholder the kernel replaces with the device count (WLIST = 2) before any use
    // (EM = 1 'plus'; EM = 2 'only' in the L11DO8_ONLY build)
    hipLaunchKernelGGL((l11_outproj_lnmd_bwd_do8<1, 2, 0, 0, 1, 2, 1, 0, 1, 2, L11DO8_ONLY ? 2 : 1, 0, 2, 0>), dim3((unsigned)G), dim3(512),
                       0, (hipStream_t)stream, (const uint8_t*)G8, (const uint8_t*)W8, (uint16_t*)nullptr,
                       (float*)nullptr, sa, sb, M, (int)cap, (const uint16_t*)nullptr, (const uint16_t*)nullptr,
                       (const uint8_t*)nullptr, (const float*)nullptr, (uint16_t*)nullptr, (uint16_t*)nullptr, ca);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

#if L11DO8_MX6
// =====================================================================================================================
// r16 iter56 MX6 host entries (ABI 2; this block is compiled only with -DL11DO8_MX6=1, so without it the .so is the base's).
//   G6  uint8 [M,400] row records (16-B aligned, row pitch 400, no padding rows); W6 uint8 [589,824] pre-swizzled weight
//   image; W6S uint8 [768] weight E8M0 plane [48][q][k] (r16 iter56 mx6/kern/LAYOUT_W6.txt, w6_layout.py). sa / sb:
//   device fp32 1.0 (the EPI2 scale s = sa * sb stays in the epilogue; E8M0 bytes carry every scale).
// =====================================================================================================================
// The MX6 E5 launch: L11DO8_E5_LAUNCH's contract and checks with G8 -> G6, W8 -> W6, + W6S (16-B aligned like G6 / W6).
// The 'only' .so (L11DO8_ONLY = 1) exports l11do8_e5_only_mx6_launch (EM = 2, MX = 1); the 'plus' .so exports
// l11do8_e5_plus_mx6_launch (EM = 1, MX = 1: the cancel path keeps the bf16 d(attn) DX store).
#if L11DO8_ONLY
#define L11DO8_E5_MX6_LAUNCH l11do8_e5_only_mx6_launch
#else
#define L11DO8_E5_MX6_LAUNCH l11do8_e5_plus_mx6_launch
#endif
extern "C" int L11DO8_E5_MX6_LAUNCH(const void* G6, const void* W6, const void* W6S, const float* sa, const float* sb,
                                    int M, const void* X, const void* U, const void* MK, const float* MEAN,
                                    const float* RSTD, const void* WN, const void* BN, void* DX, void* DZ8, int rows_out,
                                    const float* INVS, void* AMAXP, void* DWP, void* DBP, void* DUCS, long long P,
                                    float dscale, float invD, const void* WL, const void* CNT, long long cap, void* DO8,
                                    void* E8, void* stream) {
    long long G, fl, nf;
    if (M <= 0 || (long long)M > L11DO8_M_MAX || cap < ((long long)M + 63) / 64) return (int)hipErrorInvalidValue;
    if (l11do8_plan_impl(cap, &G, &fl, &nf) != 0) return (int)hipErrorInvalidValue;
    if (P != G * 8 * nf) return (int)hipErrorInvalidValue;
    // r16 iter37 l11vd (C1): the masked-before-scaling dropout is exact only for a finite dscale > 0 (prod: 1/(1-p))
    if (!(dscale > 0.0f && dscale <= 3.40282346638528859811704183484516925e+38f)) return (int)hipErrorInvalidValue;
    if (rows_out < M || (long long)rows_out >= (long long)M + 4096) return (int)hipErrorInvalidValue;
    if (!(l11do8_a16(G6) && l11do8_a16(W6) && l11do8_a16(W6S) && l11do8_a16(X) && l11do8_a16(U) && l11do8_a16(MK) &&
          l11do8_a16(WN) && l11do8_a16(BN) && l11do8_a16(DX) && l11do8_a16(DZ8) && l11do8_a16(DWP) &&
          l11do8_a16(DBP) && l11do8_a16(DUCS) && l11do8_a16(DO8) && l11do8_a16(E8) && l11do8_a8(WL)))
        return (int)hipErrorInvalidValue;
    if (!(l11do8_a4(sa) && l11do8_a4(sb) && l11do8_a4(MEAN) && l11do8_a4(RSTD) && l11do8_a4(INVS) &&
          l11do8_a4(AMAXP) && l11do8_a4(CNT)))
        return (int)hipErrorInvalidValue;
    L11CArgs ca{};
    ca.X = (const uint16_t*)X;
    ca.U = (const uint16_t*)U;
    ca.MK = (const uint8_t*)MK;
    ca.MEAN = MEAN;
    ca.RSTD = RSTD;
    ca.WN = (const uint16_t*)WN;
    ca.BN = (const uint16_t*)BN;
    ca.DX = (uint16_t*)DX;
    ca.DU = (uint16_t*)DZ8;  // the E5 kernel addresses the u block through ca.DU (byte rows of 2048), DZ8 for pad rows
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
    ca.WLP = (const int*)WL;
    ca.DO8 = (uint8_t*)DO8;
    ca.E8 = (uint8_t*)E8;
    ca.WLCNT = (const int*)CNT;
    const hipError_t stale = hipGetLastError();
    // nblk = cap: a placeholder the kernel replaces with the device count (WLIST = 2) before any use; MR = W6S
    hipLaunchKernelGGL((l11_outproj_lnmd_bwd_do8_mx6<1, 2, 0, 0, 1, 2, 1, 0, 1, 2, L11DO8_ONLY ? 2 : 1, 0, 2, 0>),
                       dim3((unsigned)G), dim3(512), 0, (hipStream_t)stream, (const uint8_t*)G6, (const uint8_t*)W6,
                       (uint16_t*)nullptr, (float*)nullptr, sa, sb, M, (int)cap, (const uint16_t*)nullptr,
                       (const uint16_t*)nullptr, (const uint8_t*)nullptr, (const float*)W6S, (uint16_t*)nullptr,
                       (uint16_t*)nullptr, ca);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

// dy debug (the correctness gate's dy): the instantiation <STUB 0, NBUF 2, APF 0, DIAG 0, PW 1, TR 0, PFD 1, SCP 0, E5 0,
// WLIST 2, EM 0, RET 0, EMV 0, TM 0, MX> runs the same tiles, mainloop and EPI2 epilogue and stores the redistributed dy
// rows (the bf16 values the consumer reads) to C bf16 [M,1536] row-major for every row of every work-list tile (rows of
// [row0, row0 + rv) only: C's buffer range drops the rest). WL / CNT / cap as for the E5 launch (l11do8_wl_build_launch).
static int l11do8_dy_debug_impl(int mx, const void* A, const void* B, const void* W6S, const float* sa, const float* sb,
                                int M, void* C, const void* WL, const void* CNT, long long cap, void* stream) {
    long long G, fl, nf;
    if (M <= 0 || (long long)M > L11DO8_M_MAX || cap < ((long long)M + 63) / 64) return (int)hipErrorInvalidValue;
    if (l11do8_plan_impl(cap, &G, &fl, &nf) != 0) return (int)hipErrorInvalidValue;
    if (!(l11do8_a16(A) && l11do8_a16(B) && (!mx || l11do8_a16(W6S)) && l11do8_a16(C) && l11do8_a8(WL)))
        return (int)hipErrorInvalidValue;
    if (!(l11do8_a4(sa) && l11do8_a4(sb) && l11do8_a4(CNT))) return (int)hipErrorInvalidValue;
    L11CArgs ca{};
    ca.fl = 1;  // TR 0: no consumer partials
    ca.nf = 1;
    ca.WLP = (const int*)WL;
    ca.WLCNT = (const int*)CNT;
    const hipError_t stale = hipGetLastError();
    if (mx) {
        hipLaunchKernelGGL((l11_outproj_lnmd_bwd_do8_mx6<0, 2, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 0>), dim3((unsigned)G),
                           dim3(512), 0, (hipStream_t)stream, (const uint8_t*)A, (const uint8_t*)B, (uint16_t*)C,
                           (float*)nullptr, sa, sb, M, (int)cap, (const uint16_t*)nullptr, (const uint16_t*)nullptr,
                           (const uint8_t*)nullptr, (const float*)W6S, (uint16_t*)nullptr, (uint16_t*)nullptr, ca);
    } else {
        hipLaunchKernelGGL((l11_outproj_lnmd_bwd_do8<0, 2, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 0>), dim3((unsigned)G),
                           dim3(512), 0, (hipStream_t)stream, (const uint8_t*)A, (const uint8_t*)B, (uint16_t*)C,
                           (float*)nullptr, sa, sb, M, (int)cap, (const uint16_t*)nullptr, (const uint16_t*)nullptr,
                           (const uint8_t*)nullptr, (const float*)nullptr, (uint16_t*)nullptr, (uint16_t*)nullptr, ca);
    }
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}
// MX6 dy (G6 / W6 / W6S, sa = sb = device 1.0)
extern "C" int l11do8_mx6_dy_debug_launch(const void* G6, const void* W6, const void* W6S, const float* sa, const float* sb,
                                          int M, void* C, const void* WL, const void* CNT, long long cap, void* stream) {
    return l11do8_dy_debug_impl(1, G6, W6, W6S, sa, sb, M, C, WL, CNT, cap, stream);
}
// FP8 dy, the gate's positive control (production-recipe G8 e5m2 [M,512] / W8 e4m3 [1536,512] / sa / sb)
extern "C" int l11do8_fp8_dy_debug_launch(const void* G8, const void* W8, const float* sa, const float* sb, int M,
                                          void* C, const void* WL, const void* CNT, long long cap, void* stream) {
    return l11do8_dy_debug_impl(0, G8, W8, nullptr, sa, sb, M, C, WL, CNT, cap, stream);
}
#endif  // L11DO8_MX6
#endif  // !L11DO8_BODY_PASS

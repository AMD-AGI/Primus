// outproj_dual_mx6.cu -- r14 iter66 DESIGN v1 (frozen sketch; compile-checked with the ROCm 7.2.0 proxy, NOT GPU-run).
// One read of bf16 dout [M,512] -> G6 uint8 [M,400] (REF encode_g6, neg_zero=True), dout_t e4m3 step-tiled
// [M_PAD/128,512,128] (bitwise _quantize_amax_dual_kernel DST_T, SEPARATE_T + T_TILED), PARTIAL fp32 [4*ceil(M/256)]
// (Triton's 256x128 tile maxnum, NaN-aware), INV_USED (the INV bits read). WG = 8 waves = one 256-row band (= Triton
// pid_m) x all 512 features. Base: DESIGN_C (odm6_c5.cu 7fd591ba) + judge fixes F1-F6 (DESIGN.md sec. 2).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define ODM6_ABI_VERSION 1
#ifndef OUTPROJ_MX6_G6_F32
#define OUTPROJ_MX6_G6_F32 0   // 0: v_cvt_scalef32_pk32_fp6_bf16 on the raw bf16 (primary); 1: f32 pre-scale + 2xpk16 (rung L3)
#endif
#ifndef OUTPROJ_MX6_DN1
#define OUTPROJ_MX6_DN1 0      // 1: MODE.FP_DENORM f32 = 1 (keep input denormals, flush output denormals); only with G6_F32=1
#endif
#ifndef OUTPROJ_MX6_REVERSE
#define OUTPROJ_MX6_REVERSE 0  // 1: lab variant V-R (band = nb - 1 - blockIdx.x)
#endif
#ifndef OUTPROJ_MX6_WPE
#define OUTPROJ_MX6_WPE 1      // amdgpu_waves_per_eu(4, 4) on the producer (the occupancy the LDS allows anyway)
#endif
#ifndef OUTPROJ_MX6_PF
#define OUTPROJ_MX6_PF (OUTPROJ_MX6_G6_F32 ? 1 : 2)   // register sets: 2 = double-buffered passes; 1 = one set (rung L3)
#endif
#if OUTPROJ_MX6_DN1 && !OUTPROJ_MX6_G6_F32
#error "OUTPROJ_MX6_DN1 is a rung-L3 option"
#endif

namespace odm6 {
typedef uint32_t v16u __attribute__((ext_vector_type(16)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef uint32_t v6u __attribute__((ext_vector_type(6)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef short v2s __attribute__((ext_vector_type(2)));
typedef unsigned short us2 __attribute__((ext_vector_type(2)));
typedef short s2 __attribute__((ext_vector_type(2)));
}  // namespace odm6
extern "C" __device__ odm6::i32x4 odm6_buf_ld4(odm6::i32x4 rsrc, int voff, int soff, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
extern "C" __device__ void odm6_buf_st4(odm6::i32x4 v, odm6::i32x4 rsrc, int voff, int soff, int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");
extern "C" __device__ void odm6_buf_st2(odm6::i32x2 v, odm6::i32x4 rsrc, int voff, int soff, int aux) __asm("llvm.amdgcn.raw.buffer.store.v2i32");
extern "C" __device__ void odm6_buf_st1(int v, odm6::i32x4 rsrc, int voff, int soff, int aux) __asm("llvm.amdgcn.raw.buffer.store.i32");

namespace odm6 {
constexpr int NT = 512, BAND = 256;
constexpr int LDS_T = 16 * 33 * 128;    // 67,584: dout_t step image, 16 blocks x 33 slots x 128 B (1 pad slot per block)
constexpr int LDS_BYTES = LDS_T + 128;  // + PARTIAL scratch [8 waves][4 strips] u32 = 67,712

__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {  // l11_do8.cu:221-229 (raw, stride 0)
  const uint64_t a = reinterpret_cast<uint64_t>(p);
  i32x4 r; r.x = (int)(uint32_t)a; r.y = (int)(uint32_t)(a >> 32); r.z = (int)nrec; r.w = 0x00110000; return r;
}
__device__ __forceinline__ uint32_t pkmax(uint32_t a, uint32_t b) {
  return __builtin_bit_cast(uint32_t, __builtin_elementwise_max(__builtin_bit_cast(us2, a), __builtin_bit_cast(us2, b)));
}
__device__ __forceinline__ uint32_t pkadd(uint32_t a, uint32_t b) {
  return __builtin_bit_cast(uint32_t, __builtin_bit_cast(us2, a) + __builtin_bit_cast(us2, b));
}
__device__ __forceinline__ uint32_t pkasr15(uint32_t a) {
  return __builtin_bit_cast(uint32_t, __builtin_bit_cast(s2, a) >> (s2){15, 15});
}
__device__ __forceinline__ float lo_f(uint32_t d) { return __uint_as_float(d << 16); }
__device__ __forceinline__ float hi_f(uint32_t d) { return __uint_as_float(d & 0xffff0000u); }
__device__ __forceinline__ float med448(float v) { return __builtin_amdgcn_fmed3f(v, -448.0f, 448.0f); }  // v_med3_f32 v, s(-448), v(448)
__device__ __forceinline__ uint32_t e4x4(float a, float b, float c, float d) {    // ln_store_mx6.cu:95-101
  v2s z; z = __builtin_nondeterministic_value(z);
  const v2s lo = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(z, a, b, 1.0f, false);
  const v2s hv = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(lo, c, d, 1.0f, true);
  return __builtin_bit_cast(uint32_t, hv);
}
// FP6 converts: inline asm only (early clobber; s_nop 1 before, s_nop 4 after inside the string) -- SPEC_G6 5.3, STP 1.5.3
__device__ __forceinline__ v6u cvt_pk32(v16u x, uint32_t sbits) {
  v6u r;
  asm volatile("s_nop 1\n\tv_cvt_scalef32_pk32_fp6_bf16 %0, %1, %2\n\ts_nop 4" : "=&v"(r) : "v"(x), "v"(__uint_as_float(sbits)));
  return r;
}
__device__ __forceinline__ v6u cvt_2xpk16(v16f e, v16f o) {   // ln_store_mx6.cu:114-121 verbatim: field 2i <- e[i], 2i+1 <- o[i]
  v6u r; const float one = 1.0f;
  asm volatile("s_nop 1\n\tv_cvt_scalef32_2xpk16_fp6_f32 %0, %1, %2, %3\n\ts_nop 4" : "=&v"(r) : "v"(e), "v"(o), "v"(one));
  return r;
}
__device__ __forceinline__ uint32_t dpp_x1(uint32_t v) { return (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0xB1, 0xF, 0xF, true); }   // quad_perm [1,0,3,2]
__device__ __forceinline__ uint32_t dpp_x2(uint32_t v) { return (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0x4E, 0xF, 0xF, true); }   // quad_perm [2,3,0,1]
__device__ __forceinline__ uint32_t dpp_r4(uint32_t v) { return (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0x124, 0xF, 0xF, true); }  // row_ror:4
__device__ __forceinline__ uint32_t dpp_r8(uint32_t v) { return (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0x128, 0xF, 0xF, true); }  // row_ror:8
__device__ __forceinline__ uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }

// G6 codes of one row's 32-column block R (16 raw bf16 dwords in element order). bs = the E8M0 byte for a finite
// non-zero block, 127 for an all-zero or a non-finite block (scale 1.0: SPEC_G6 5.1; the byte-255 codes are zeroed later).
__device__ __forceinline__ v6u g6_codes(const v16u& R, uint32_t bs) {
#if OUTPROJ_MX6_G6_F32
  const float f = __uint_as_float((254u - bs) << 23);          // 2^-e, a normal float for bs in 0..253 (exact pre-scale)
  v16f e, o;
#pragma unroll
  for (int i = 0; i < 16; ++i) { e[i] = lo_f(R[i]) * f; o[i] = hi_f(R[i]) * f; }
  return cvt_2xpk16(e, o);
#else
  return cvt_pk32(R, umax(bs << 23, 0x00400000u));              // byte 0 -> 0x00400000 = 2^-127 exactly
#endif
}
// rare path: NaN-ignoring max of |h| + 1 over the block (0: all 32 NaN)
__device__ __forceinline__ uint32_t nanfree_max1(const v16u& X) {
  uint32_t m = 0u;
#pragma unroll
  for (int d = 0; d < 16; ++d) {
    uint32_t xd = X[d];
    asm volatile("" : "+v"(xd));                                 // opaque: no CSE with the hot path's |h| (C lesson 5.2a)
    const uint32_t a = xd & 0x7fff7fffu;
    const uint32_t nan = pkasr15(pkadd(a, 0x007f007fu));         // 0xFFFF where h > 0x7F80
    m = pkmax(m, pkadd(a, 0x00010001u) & ~nan);
  }
  return umax(m & 0xffffu, m >> 16);
}

struct Lane { int tid, wv, lane, h, hb, k, q, b; uint32_t ld_off, wl_base, sw; };

__device__ __forceinline__ void load_pass(v16u& P, v16u& Q, i32x4 rs, int p, int g, const Lane& L) {
  const uint32_t r0 = 128u * (uint32_t)(p >> 1) + 4u * (uint32_t)g;
  const int vo = (int)(r0 * 1024u + L.ld_off);
#pragma unroll
  for (int i = 0; i < 4; ++i) {   // rows >= rows_valid are outside num_records -> zeros (Triton other=0.0)
    const i32x4 a = odm6_buf_ld4(rs, vo + 1024 * i, 0, 0), c = odm6_buf_ld4(rs, vo + 1024 * i + 16, 0, 0);
    v16u& R = (i & 1) ? Q : P;
    const int base = (i >> 1) * 8;
    R[base + 0] = (uint32_t)a.x; R[base + 1] = (uint32_t)a.y; R[base + 2] = (uint32_t)a.z; R[base + 3] = (uint32_t)a.w;
    R[base + 4] = (uint32_t)c.x; R[base + 5] = (uint32_t)c.y; R[base + 6] = (uint32_t)c.z; R[base + 7] = (uint32_t)c.w;
  }
}

__device__ __forceinline__ void pass_body(v16u& P, v16u& Q, int p, int g, float inv, float tmul, i32x4 rg6,
                                          uint8_t* lds, uint32_t& acc, const Lane& L) {
  // (1) dout_t: Triton's sequence per element, on the load layout (column 32b+16hb+jj, tokens 4g..4g+3)
  uint8_t* const wbase = lds + L.wl_base + 4u * ((uint32_t)g ^ L.sw);
#pragma unroll
  for (int jj = 0; jj < 16; ++jj) {
    const int dw = jj >> 1;
    const uint32_t d0 = P[dw], d1 = Q[dw], d2 = P[8 + dw], d3 = Q[8 + dw];
    float x0, x1, x2, x3;
    if (jj & 1) { x0 = hi_f(d0); x1 = hi_f(d1); x2 = hi_f(d2); x3 = hi_f(d3); }
    else        { x0 = lo_f(d0); x1 = lo_f(d1); x2 = lo_f(d2); x3 = lo_f(d3); }
    const float s0 = inv * x0, s1 = inv * x1, s2_ = inv * x2, s3 = inv * x3;      // v_mul_f32 v, inv, x
    const float t0 = tmul * s0, t1 = tmul * s1, t2 = tmul * s2_, t3 = tmul * s3;  // v_mul_f32 v, 2^-t, s (2nd rounding)
    *reinterpret_cast<uint32_t*>(wbase + 128 * jj) = e4x4(med448(t0), med448(t1), med448(t2), med448(t3));
  }
  // (2) G6 map: 16 permlane16_swap -> lane holds row slot 0 (P) and slot 1 (Q) of block b, whole, in element order
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    const auto r = __builtin_amdgcn_permlane16_swap(P[d], P[8 + d], false, false); P[d] = r[0]; P[8 + d] = r[1];
    const auto s = __builtin_amdgcn_permlane16_swap(Q[d], Q[8 + d], false, false); Q[d] = s[0]; Q[8 + d] = s[1];
  }
  // (3) E8M0 (SPEC_G6 3.1, NaN-dominant u16 max), codes, byte-255 zeroing, PARTIAL key (+1 encoded maxnum)
  uint32_t m0 = 0u, m1 = 0u;
#pragma unroll
  for (int d = 0; d < 16; ++d) { m0 = pkmax(m0, P[d] & 0x7fff7fffu); m1 = pkmax(m1, Q[d] & 0x7fff7fffu); }
  const uint32_t a0 = umax(m0 & 0xffffu, m0 >> 16), a1 = umax(m1 & 0xffffu, m1 >> 16);
  int b0 = (int)(a0 >> 7) - 2 + (int)((a0 & 0x7fu) > 0x70u); b0 = b0 < 0 ? 0 : b0;
  int b1 = (int)(a1 >> 7) - 2 + (int)((a1 & 0x7fu) > 0x70u); b1 = b1 < 0 ? 0 : b1;
  const uint32_t bs0 = (a0 - 1u) >= 0x7f7fu ? 127u : (uint32_t)b0;   // judge fix F1: zero / non-finite -> scale 1.0
  const uint32_t bs1 = (a1 - 1u) >= 0x7f7fu ? 127u : (uint32_t)b1;
  const uint32_t e0 = a0 >= 0x7f80u ? 255u : (a0 == 0u ? 127u : (uint32_t)b0);
  const uint32_t e1 = a1 >= 0x7f80u ? 255u : (a1 == 0u ? 127u : (uint32_t)b1);
  uint32_t v0 = a0 + 1u, v1 = a1 + 1u;
  // wave-uniform rare branch (some Inf / NaN in the wave's 2 x 8 row-blocks); taken BEFORE the converts so that the
  // raw dwords die at their convert (judge fix F4: keeps rung L3 inside 128 VGPR)
  const bool nf = __builtin_amdgcn_ballot_w64(a0 >= 0x7f80u || a1 >= 0x7f80u) != 0ull;
  if (nf) {
    if (a0 > 0x7f80u) v0 = nanfree_max1(P);
    if (a1 > 0x7f80u) v1 = nanfree_max1(Q);
  }
  v6u c0 = g6_codes(P, bs0);
#if OUTPROJ_MX6_G6_F32
  asm volatile("" : "+v"(c0), "+v"(Q));          // L3: finish row slot 0's unpack/pre-scale/convert before slot 1's
#endif
  v6u c1 = g6_codes(Q, bs1);
  if (nf) {                                      // byte-255 blocks: 24 zero code bytes (SPEC_G6 4.7, strict)
    if (a0 >= 0x7f80u) c0 = (v6u){0u, 0u, 0u, 0u, 0u, 0u};
    if (a1 >= 0x7f80u) c1 = (v6u){0u, 0u, 0u, 0u, 0u, 0u};
  }
  acc = umax(acc, umax(v0, v1));
  // (4) G6 stores: rows >= rows_valid are outside num_records -> dropped (G6 has exactly M rows)
  const uint32_t rr = 128u * (uint32_t)(p >> 1) + 4u * (uint32_t)g + 2u * (uint32_t)L.hb;
  const int ro = (int)(rr * 400u);
  odm6_buf_st4((i32x4){(int)c0[0], (int)c0[1], (int)c0[2], (int)c0[3]}, rg6, ro + 16 * L.b, 0, 0);
  odm6_buf_st2((i32x2){(int)c0[4], (int)c0[5]}, rg6, ro + 256 + 8 * L.b, 0, 0);
  odm6_buf_st4((i32x4){(int)c1[0], (int)c1[1], (int)c1[2], (int)c1[3]}, rg6, ro + 400 + 16 * L.b, 0, 0);
  odm6_buf_st2((i32x2){(int)c1[4], (int)c1[5]}, rg6, ro + 656 + 8 * L.b, 0, 0);
  uint32_t u0 = e0 << (8 * L.k), u1 = e1 << (8 * L.k);              // E8M0 dword q = [k0 k1 k2 k3] over the DPP quad
  u0 |= dpp_x1(u0); u0 |= dpp_x2(u0);
  u1 |= dpp_x1(u1); u1 |= dpp_x2(u1);
  if (L.k < 2) odm6_buf_st1((int)(L.k ? u1 : u0), rg6, ro + 400 * L.k + 384 + 4 * L.q, 0, 0);
}

__device__ __forceinline__ void copy_out(const uint8_t* lds, i32x4 rdt, uint32_t tile_off, int tid) {
  const int cc = tid >> 3, hp = tid & 7, bhi = cc >> 5, cw = cc & 31, hbw = (cw >> 4) & 1;
  const uint32_t lane_off = (uint32_t)(256 * bhi + cw) * 128u + 16u * (uint32_t)hp;
#pragma unroll
  for (int it = 0; it < 8; ++it) {   // 8 lanes per 128-B column line: every store instruction writes 8 whole lines
    const int b = it + 8 * bhi;
    const int grp = (hp ^ ((b >> 2) + 4 * hbw)) & 7;
    v4u v = *reinterpret_cast<const v4u*>(lds + (33 * b + cw) * 128 + 16 * grp);
    if ((it >> 1) & 1) v = (v4u){v.z, v.w, v.x, v.y};
    odm6_buf_st4(__builtin_bit_cast(i32x4, v), rdt, (int)(tile_off + 4096u * (uint32_t)it + lane_off), 0, 0);  // judge F3
  }
}

__global__ __launch_bounds__(NT)
#if OUTPROJ_MX6_WPE
__attribute__((amdgpu_waves_per_eu(4, 4)))
#endif
void outproj_dual_mx6_kernel(const uint8_t* __restrict__ dout, const float* __restrict__ INV, float tmul, int M, int M_PAD,
                             uint8_t* __restrict__ g6, uint8_t* __restrict__ dout_t, float* __restrict__ partial,
                             uint32_t* __restrict__ inv_used) {
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n s_nop 7\n s_nop 7" ::: "memory");   // FP16_OVFL = 1, as Triton
#if OUTPROJ_MX6_DN1
  asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 1\n s_nop 7\n s_nop 7" ::: "memory");    // f32 denorm: in keep, out flush
#endif
  __shared__ __attribute__((aligned(16))) uint8_t lds[LDS_BYTES];
#if OUTPROJ_MX6_REVERSE
  const int band = (int)(gridDim.x - 1u - blockIdx.x);
#else
  const int band = (int)blockIdx.x;
#endif
  const long long band0 = (long long)band * BAND;
  const float inv = INV[0];                                                     // device fp32 (never a host float)
  if (band == 0 && threadIdx.x == 0 && inv_used != nullptr) inv_used[0] = __float_as_uint(inv);   // judge F2: raw bits
  const long long rvl = (long long)M - band0, rpl = (long long)M_PAD - band0;
  const int rows_valid = rvl < BAND ? (int)rvl : BAND, rows_pad = rpl < BAND ? (int)rpl : BAND;
  const i32x4 rs = make_rsrc(dout + band0 * 1024, (uint32_t)rows_valid * 1024u);   // dout rows >= M read 0
  const i32x4 rg6 = make_rsrc(g6 + band0 * 400, (uint32_t)rows_valid * 400u);      // G6 rows >= M dropped
  const i32x4 rdt = make_rsrc(dout_t + band0 * 512, (uint32_t)rows_pad * 512u);    // steps >= M_PAD/128 never written
  Lane L;
  L.tid = (int)threadIdx.x; L.wv = L.tid >> 6; L.lane = L.tid & 63; L.h = L.lane >> 5; L.hb = (L.lane >> 4) & 1;
  const int x = L.lane & 15; L.k = x & 3; L.q = x >> 2; L.b = 4 * L.k + L.q;
  L.ld_off = 64u * L.b + 32u * L.hb;
  L.wl_base = (uint32_t)(33 * L.b + 16 * L.hb) * 128u;
  L.sw = 2u * (uint32_t)(L.b >> 1) + 16u * (uint32_t)L.hb;
  const int g0 = 2 * L.wv + L.h, g1 = 16 + g0;
  uint32_t acc = rows_valid < BAND ? 1u : 0u;   // masked rows of the 256-row tile contribute +0.0 (C4)
  const bool t1 = rows_pad > 128;
#if OUTPROJ_MX6_PF == 2
  v16u PA, QA, PB, QB;                           // set A: passes 0, 2; set B: passes 1, 3 (one pass of loads in flight)
  load_pass(PA, QA, rs, 0, g0, L);
  load_pass(PB, QB, rs, 1, g1, L);
  pass_body(PA, QA, 0, g0, inv, tmul, rg6, lds, acc, L);
  load_pass(PA, QA, rs, 2, g0, L);               // judge fix F6: UNCONDITIONAL prefetch (no CFG join before the waits:
  pass_body(PB, QB, 1, g1, inv, tmul, rg6, lds, acc, L);   // exact vmcnt); without tile 1 these rows lie past num_records
  __syncthreads();                               // and read 0 with no memory traffic
  copy_out(lds, rdt, 0u, L.tid);
  load_pass(PB, QB, rs, 3, g1, L);               // F6, idem
  if (t1) {
    __syncthreads();                             // copy-out reads of tile 0 done before pass 2 rewrites the staging
    pass_body(PA, QA, 2, g0, inv, tmul, rg6, lds, acc, L);
    pass_body(PB, QB, 3, g1, inv, tmul, rg6, lds, acc, L);
    __syncthreads();
    copy_out(lds, rdt, 65536u, L.tid);
  }
#else
  v16u PA, QA;                                   // one register set: load, then compute (Triton's own MLP pattern)
  load_pass(PA, QA, rs, 0, g0, L);
  pass_body(PA, QA, 0, g0, inv, tmul, rg6, lds, acc, L);
  load_pass(PA, QA, rs, 1, g1, L);
  pass_body(PA, QA, 1, g1, inv, tmul, rg6, lds, acc, L);
  __syncthreads();
  copy_out(lds, rdt, 0u, L.tid);
  if (t1) {
    load_pass(PA, QA, rs, 2, g0, L);
    __syncthreads();
    pass_body(PA, QA, 2, g0, inv, tmul, rg6, lds, acc, L);
    load_pass(PA, QA, rs, 3, g1, L);
    pass_body(PA, QA, 3, g1, inv, tmul, rg6, lds, acc, L);
    __syncthreads();
    copy_out(lds, rdt, 65536u, L.tid);
  }
#endif
  uint32_t a = acc;                              // PARTIAL strip = k: row_ror 4 / 8 keep x & 3; then xor 16, xor 32
  a = umax(a, dpp_r4(a)); a = umax(a, dpp_r8(a));
  a = umax(a, (uint32_t)__shfl_xor((int)a, 16));
  a = umax(a, (uint32_t)__shfl_xor((int)a, 32));
  uint32_t* red = reinterpret_cast<uint32_t*>(lds + LDS_T);
  if (L.lane < 4) red[L.wv * 4 + L.lane] = a;
  __syncthreads();
  if (L.tid < 4) {
    uint32_t m = red[L.tid];
#pragma unroll
    for (int w = 1; w < 8; ++w) m = umax(m, red[w * 4 + L.tid]);
    partial[4LL * band + L.tid] = __uint_as_float(m == 0u ? 0x7fc00000u : (m - 1u) << 16);
  }
}

// ---- W1: raw-mode, L11-permutation fork of k512_mx6_wshadow (U3 w6_fork_sketch.cu + bf16 master; DESIGN_B App. A)
constexpr int W6CH = 24576, W6SL = 3072;
__device__ __forceinline__ int w_e8m0(uint32_t bm) {   // k512_mx6_gemm.cu:420-429 verbatim (on the RAW weight)
  if (bm == 0u) return 127;
  if (bm >= 0x7f800000u) return 255;
  if (bm < 0x00800000u) return 0;
  int e = (int)(bm >> 23) - 127 - 2 + (((bm & 0x7fffffu) > 0x700000u) ? 1 : 0);
  e = e < -127 ? -127 : e;
  return e + 127;
}
__device__ __forceinline__ uint32_t w_code(float y, float inv2e) {   // E2M3 RNE, sign = signbit (REF neg_zero=True)
  const float v = y * inv2e;
  float a = fabsf(v); a = a > 7.5f ? 7.5f : a;
  uint32_t c;
  if (a < 2.0f) c = (uint32_t)rintf(a * 8.0f); else if (a < 4.0f) c = (uint32_t)rintf(a * 4.0f) + 8u; else c = (uint32_t)rintf(a * 2.0f) + 16u;
  if (!(a == a)) c = 0u;
  return c | ((__float_as_uint(v) >> 26) & 0x20u);
}
template <bool BF16>
__global__ __launch_bounds__(256) void outproj_w6_shadow_kernel(const void* __restrict__ wv, uint8_t* __restrict__ w6, uint8_t* __restrict__ w6s) {
#pragma clang fp contract(off)
  const int tid = threadIdx.x, lane = tid & 63, gw = (int)blockIdx.x * 4 + (tid >> 6);
  const int kb = gw & 15, n = (gw >> 4) * 64 + lane;
  float v[32];
  if constexpr (BF16) {
    const v4u* src = reinterpret_cast<const v4u*>(reinterpret_cast<const uint16_t*>(wv) + (size_t)n * 512 + 32 * kb);
#pragma unroll
    for (int i = 0; i < 4; ++i) { const v4u t = src[i]; const uint32_t d[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) { v[8 * i + 2 * k] = lo_f(d[k]); v[8 * i + 2 * k + 1] = hi_f(d[k]); } }
  } else {
    const float4* src = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(wv) + (size_t)n * 512 + 32 * kb);
#pragma unroll
    for (int i = 0; i < 8; ++i) { const float4 t = src[i]; v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w; }
  }
  uint32_t bm = 0u;
#pragma unroll
  for (int x = 0; x < 32; ++x) bm = umax(bm, __float_as_uint(v[x]) & 0x7fffffffu);   // NaN-dominant
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) bm = umax(bm, (uint32_t)__shfl_xor((int)bm, o));   // the 32 columns n of the 32x32 tile
  const int eb = w_e8m0(bm), e = eb - 127;
  const float inv2e = __uint_as_float((uint32_t)(127 - (e < -127 ? -127 : (e > 126 ? 126 : e))) << 23);
  uint32_t d[6] = {0u, 0u, 0u, 0u, 0u, 0u};
#pragma unroll
  for (int x = 0; x < 32; ++x) {
    const uint32_t c = w_code(v[x], inv2e); const int bit = 6 * x, wd = bit >> 5, o = bit & 31;
    d[wd] |= c << o; if (o > 26) d[wd + 1] |= c >> (32 - o);
  }
  if (eb == 255) { for (int i = 0; i < 6; ++i) d[i] = 0u; }                          // non-finite tile: zero codes
  const int S = kb >> 2, q = kb & 3, t = n >> 8, ws_ = (n >> 5) & 7, jj = (n >> 2) & 1, R = 4 * ((n >> 3) & 3) + (n & 3);
  uint8_t* const slice = w6 + (size_t)(4 * t + S) * W6CH + W6SL * ws_;
  *reinterpret_cast<v4u*>(slice + 1024 * jj + 64 * R + 16 * (q ^ ((R >> 2) & 3))) = (v4u){d[0], d[1], d[2], d[3]};
  *reinterpret_cast<uint2*>(slice + 2048 + 512 * jj + 32 * R + 8 * (q ^ (2 * ((R >> 3) & 1)))) = make_uint2(d[4], d[5]);
  if ((lane & 31) == 0) w6s[(size_t)(n >> 5) * 16 + q * 4 + S] = (uint8_t)eb;
}
}  // namespace odm6

// ---- C ABI, convention (b) as l11_do8 / x2h: 0 = launched; > 0 = hipError of a rejected argument or a failed launch
// (outputs not written); < 0 = launched, and -rc is a stale error an EARLIER call left on this thread (drained here).
#ifndef OUTPROJ_MX6_SRC_SHA256
#define OUTPROJ_MX6_SRC_SHA256 "unknown"
#endif
static inline bool odm6_mis16(const void* q) { return ((uintptr_t)q & 15u) != 0; }
extern "C" int outproj_dual_mx6_launch(const void* dout, const float* inv, float t_inv_mul, int M, int M_PAD, void* g6,
                                       void* dout_t, float* partial, float* inv_used, void* stream) {
  const uint32_t tb = __builtin_bit_cast(uint32_t, t_inv_mul);
  if (dout == nullptr || inv == nullptr || g6 == nullptr || dout_t == nullptr || partial == nullptr) return (int)hipErrorInvalidValue;
  if (M < 1 || M > 4194304 || M_PAD != (M + 127) / 128 * 128) return (int)hipErrorInvalidValue;
  if (!(tb == 0x3d800000u || tb == 0x3d000000u || tb == 0x3c800000u || tb == 0x3c000000u)) return (int)hipErrorInvalidValue;  // 2^-4..2^-7
  if (odm6_mis16(dout) || odm6_mis16(g6) || odm6_mis16(dout_t) || odm6_mis16(partial) || odm6_mis16(inv) ||
      (inv_used != nullptr && odm6_mis16(inv_used))) return (int)hipErrorInvalidValue;
  const hipError_t stale = hipGetLastError();
  hipLaunchKernelGGL(odm6::outproj_dual_mx6_kernel, dim3((unsigned)((M + 255) / 256)), dim3(odm6::NT), 0, (hipStream_t)stream,
                     (const uint8_t*)dout, inv, t_inv_mul, M, M_PAD, (uint8_t*)g6, (uint8_t*)dout_t, partial, (uint32_t*)inv_used);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) return (int)e;
  return stale == hipSuccess ? 0 : -(int)stale;
}
extern "C" int outproj_w6_shadow_launch(const void* w, int w_is_bf16, void* w6, void* w6s, void* stream) {
  if (w == nullptr || w6 == nullptr || w6s == nullptr || odm6_mis16(w) || odm6_mis16(w6) || odm6_mis16(w6s)) return (int)hipErrorInvalidValue;
  const hipError_t stale = hipGetLastError();
  if (w_is_bf16) hipLaunchKernelGGL(odm6::outproj_w6_shadow_kernel<true>, dim3(96), dim3(256), 0, (hipStream_t)stream, w, (uint8_t*)w6, (uint8_t*)w6s);
  else hipLaunchKernelGGL(odm6::outproj_w6_shadow_kernel<false>, dim3(96), dim3(256), 0, (hipStream_t)stream, w, (uint8_t*)w6, (uint8_t*)w6s);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) return (int)e;
  return stale == hipSuccess ? 0 : -(int)stale;
}
extern "C" int outproj_dual_mx6_abi_version(void) { return ODM6_ABI_VERSION; }
extern "C" int outproj_dual_mx6_drain_last_error(void) { return (int)hipGetLastError(); }
extern "C" long long outproj_dual_mx6_partial_count(int M) { return 4LL * ((M + 255) / 256); }
extern "C" int outproj_dual_mx6_lds_bytes(void) { return odm6::LDS_BYTES; }
extern "C" int outproj_dual_mx6_variant(void) {
  return OUTPROJ_MX6_G6_F32 | (OUTPROJ_MX6_DN1 << 1) | (OUTPROJ_MX6_REVERSE << 2) | (OUTPROJ_MX6_WPE << 3) | (OUTPROJ_MX6_PF << 4);
}
extern "C" const char* outproj_dual_mx6_impl(void) {
  return "outproj_dual_mx6 v1 (r14 iter66 DESIGN v1): 8-wave WG per 256-row band; 4x16 load map with block halves 16 lanes "
         "apart + permlane16_swap; pk32_fp6_bf16 G6 (G6_F32=1: f32 pre-scale + 2xpk16); 33-slot LDS dout_t step stage, "
         "whole-line copy-out; +1-encoded maxnum PARTIAL; buffer-OOB masking; W1 raw-mode L11 wshadow fork; src="
         OUTPROJ_MX6_SRC_SHA256;
}

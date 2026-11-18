// x2h_pass.cu -- X2H (actckpt iter68): a persistent HIP kernel for gfx950 (MI350X) that replaces the Triton
// _ecx_dgrad_pass launches of E-X2b (DLRMv4) and is meant to be BITWISE identical to them.
//
// Normative inputs (all under probes/iter68/x2h/):
//   contract/CONTRACT.md section 1  the bitwise contract ([C1.1] main dot, [C1.2] indicator, [C1.3] semantics,
//                                   [C1.4] deviations)
//   DESIGN_v1.md D1-D8              the build decisions ([D1] .. [D8])
//   facts/FACTS.md                  gfx950 layouts, builtins, LDS-DMA, waits, occupancy ([F a.4] etc.)
//   critique/CRITIQUE.md 3.3        v1-R ([CR B3] = deferred stores, ...)
// Production reference (READ ONLY): ship14_wt/ladder3_R11/recommendation/generative_recommenders/ops/triton/
//   triton_ec_dgrad_segreduce.py, _ecx_tile lines 346-424 ([src:N] = line N of that file).
//
// Kernels
//   x2h_wpack                   W1 [256, SW] bf16 -> fragment-order Wp [D2]
//   x2h_pass<ROLE_SWAP, EMIT>   the pass; IS_CAT is a runtime argument. ROLE_SWAP=1: main dot srcA = A (H-swap,
//                               needs GPU gate G1); ROLE_SWAP=0: Triton's roles + wave-private LDS transpose.
// C ABI (extern "C", bottom of the file): x2h_abi_version, x2h_lds_bytes, x2h_wp_elems, x2h_wpack_launch,
//   x2h_pass_grid, x2h_pass_launch, x2h_drain_last_error.
//
// Every index computation is a small named __device__ function in section 1, commented with the clause it implements;
// the CPU emulator ../emu/x2h_emu.py mirrors them 1:1 (its [cu:N] citations name lines of the pre-fix source, sha256
// c45f517a; fix/FIX_REPORT.md maps them to this file). Section 2 holds the LDS layout, section 3 the device building
// blocks, section 4 the kernel, section 5 the launchers.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define X2H_ABI_VERSION 2026101202  // 2026101202: + max_tiles (x2h_pass_launch), x2h_pass_grid (fix stage)

namespace x2h {

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float f32x16 __attribute__((ext_vector_type(16)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) bf16x4 lds_bf16x4_t;
typedef __attribute__((address_space(3))) uint32_t* lds_u32_ptr_t;
}  // namespace x2h

// LLVM buffer intrinsics with the v4i32 resource (the in-tree W3 idiom, wgrad_tm_ck_w3.cu:69-71); available in both
// toolchains. buffer.load.lds: M0 = the (wave-uniform) LDS pointer, lane l writes 16 B at M0 + 16 l.
extern "C" __device__ void x2h_llvm_buffer_load_lds(x2h::i32x4 rsrc, x2h::lds_u32_ptr_t lds, int size, int voffset,
                                                    int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");
extern "C" __device__ x2h::i32x4 x2h_llvm_buffer_load_v4i32(x2h::i32x4 rsrc, int voffset, int soffset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");

namespace x2h {
#define X2H_AS1(p) ((__attribute__((address_space(1))) void*)(p))
#define X2H_AS3(p) ((__attribute__((address_space(3))) void*)(p))
#define X2H_SB() __builtin_amdgcn_sched_barrier(0)
// Workgroup barrier with its explicit wait, bracketed by compiler memory fences (FIX M1-sem, fix/FIX_REPORT.md):
// s_barrier, s_waitcnt and sched_barrier are IntrNoMem in LLVM IR (IntrinsicsAMDGPU.td:282-283, 390-391, 367-369), so
// without a fence IR passes may move or forward LDS accesses across the barrier. The empty asm with a "memory" clobber
// is the in-tree W3 idiom (wgrad_tm_ck_w3.cu:94-101 WGCK3_FENCE); it emits no instruction. The first fence precedes the
// wait, so no store can sink below the wait that must cover it.
#define X2H_FENCE() asm volatile("" ::: "memory")
#define X2H_WAIT_BARRIER(WAITCNT)          \
  do {                                     \
    X2H_FENCE();                           \
    __builtin_amdgcn_s_waitcnt(WAITCNT);   \
    __builtin_amdgcn_s_barrier();          \
    X2H_FENCE();                           \
  } while (0)
#define X2H_FN __host__ __device__ __forceinline__ constexpr  // index functions (mirrored by the emulator)

constexpr int BM = 128;       // positions per tile [D1] = Triton BM [src:114]
constexpr int KD = 256;       // K of the main dot (dh width) [src:110]
constexpr int NOUT = 512;     // columns per W half / EC row width [src:111]
constexpr int S = 16;         // slots per pass [src:114]
constexpr int NWAVE = 8;      // 512 threads = 8 wave64 [D1]
constexpr int NTHREADS = 512;
constexpr int APITCH = 528;   // A tile row pitch in LDS [D3]
constexpr int ABUF = BM * APITCH;  // 67,584 B per A buffer [D3]
constexpr int NSTATIC = 6;    // tile-sequence entries 0..5 of every WG are static (blockIdx + j * grid) [D6]
constexpr int INVALID_TILE = 0x7FFFFFFF;
constexpr int ORDER_WAVE = 0;   // ORDER DMA, tile claim, done protocol
constexpr int IDX_WAVE = 1;     // IDX DMA
constexpr int OFF_WAVE = 2;     // OFF DMA
constexpr int DERIVE_WAVE = 7;  // per-tile derive + pass tables
constexpr int XS_BYTES = 1280;  // ROLE_SWAP=0 wave-private transpose scratch: 32 rows x 40 B [D4]

// s_waitcnt simm16 encodings (gfx9) [F b.6]
constexpr int WAIT_VM0_LGKM0 = 0x0070;  // vmcnt(0) expcnt(7) lgkmcnt(0)
constexpr int WAIT_LGKM0 = 0xC07F;      // lgkmcnt(0) only
constexpr int WAIT_VM0 = 0x0F70;        // vmcnt(0) only

// =====================================================================================================================
// 1. Index functions (each mirrored 1:1 by the CPU emulator)
// =====================================================================================================================

// ---- tile geometry [C1 notation; src:356-363] ----------------------------------------------------------------------
// s0 = 128 * tile; row r of the tile is sorted position s0 + r.
X2H_FN int tile_s0(int tile) { return tile * BM; }
// tile_end = min(s0 + 128, T)  [C1.3; src:363]
X2H_FN int tile_end_of(int s0, int npos) { return s0 + BM < npos ? s0 + BM : npos; }
// valid = s < T  [src:362]
X2H_FN bool pos_valid(int s, int npos) { return s < npos; }
// number of tiles = cdiv(T, 128)  [src:526]
X2H_FN int ntiles_of(int npos) { return (npos + BM - 1) / BM; }
// a tile index is valid iff 0 <= tile < ntiles (INVALID_TILE = 0x7FFFFFFF never is; unsigned compare, so a corrupt
// negative claim is invalid too)
X2H_FN bool tile_valid(int tile, int ntiles) { return (uint32_t)tile < (uint32_t)ntiles; }

// ---- per-position metadata [C1 notation; src:364-376, 394-398] ----------------------------------------------------
// t = ORDER[s] (item) or ORDER[s] >> 1 (cat), from the low dword of the int64 ORDER value (ORDER is a permutation of
// [0, T), T < 2^31, so the high dword is 0)  [src:366, 372]
X2H_FN int order_to_t(int o_lo, int is_cat) {
  return is_cat ? (int)((uint32_t)o_lo >> 1) : o_lo;
}
// half = ORDER[s] & 1 (cat), 0 (item)  [src:367]
X2H_FN int order_to_half(int o_lo, int is_cat) { return is_cat ? (o_lo & 1) : 0; }
// start flag: valid & (r == 0 | lo == s), lo = OFF[u] as int32  [src:377, 398]
X2H_FN bool start_flag(bool valid, int r, int lo, int s) {
  return valid && (r == 0 || lo == s);
}
// kr = inclusive prefix count of start flags - 1  [src:398]
//   computed from two 64-bit ballots m0 (rows 0..63), m1 (rows 64..127); below = popcount of the rows < r.
X2H_FN int kr_of(int below, bool start) { return below + (start ? 1 : 0) - 1; }
// key: one 16-bit word per position that encodes sel's inputs (derived, [D5]):
//   valid -> (half << 8) | kr (kr <= 127), invalid -> 0xFFFF.
// sel(pass g, h; slot j)[pos] = (kr == 16g + j) & rmask, rmask = valid (item) / valid & half == h (cat)
//   <=> key == pass_target(g, h, j)  [src:401-406]
X2H_FN uint32_t pos_key(bool valid, int half, int kr) {
  return valid ? (((uint32_t)half << 8) | (uint32_t)kr) : 0xFFFFu;
}
X2H_FN uint32_t pass_target(int g, int h, int slot) {
  return ((uint32_t)h << 8) | (uint32_t)(S * g + slot);
}
// rmask of the pass with half h at a position: (key >> 8) == h  [src:401, 404] (0xFFFF >> 8 = 0xFF never matches)
X2H_FN bool key_rmask(uint32_t key, int h) { return (key >> 8) == (uint32_t)h; }
// slot of a position in pass g (valid only if 0 <= j < 16 and key_rmask):  j = kr - 16g  [src:406]
X2H_FN int key_slot(uint32_t key, int g) { return (int)(key & 0xFFu) - S * g; }

// ---- passes [src:368-370, 390-392, 400] ---------------------------------------------------------------------------
// hmin = min(where(valid, half, 1)), hmax = max(where(valid, half, 0)), nh = hmax - hmin + 1 (item: hmin 0, nh 1)
X2H_FN int hmin_of(bool any_valid_half0) { return any_valid_half0 ? 0 : 1; }
X2H_FN int hmax_of(bool any_valid_half1) { return any_valid_half1 ? 1 : 0; }
// npass = cdiv(nseg, 16) * nh  [src:390]
X2H_FN int npass_of(int nseg, int nh) { return ((nseg + S - 1) / S) * nh; }
// g = pz // nh  [src:392];  h = hmin + pz % nh  [src:400]
X2H_FN int pass_g(int pz, int nh) { return pz / nh; }
X2H_FN int pass_h(int pz, int nh, int hmin) { return hmin + pz % nh; }
// hoff = h * 512 (cat), 0 (item)  [src:402, 405]
X2H_FN int pass_hoff(int h, int is_cat) { return is_cat ? h * NOUT : 0; }

// ---- slots and fragment rows [C1.3; src:379-386, 408-412] ----------------------------------------------------------
// act = su >= 0; closed = act & shi <= tile_end; open = act & shi > tile_end; FRAG_V store only if last_open.
// flags bit0 = ACC store (closed), bit1 = FRAG_V store (open & last_open)
X2H_FN int slot_flags(int su, int shi, int tend, bool last_open) {
  const bool act = su >= 0;
  return ((act && shi <= tend) ? 1 : 0) | ((act && shi > tend && last_open) ? 2 : 0);
}
// last_open = hi_l > tile_end  [src:383]
X2H_FN bool last_open_of(int hi_l, int tend) { return hi_l > tend; }
// FRAG_U = last_open ? u_l : -1  [src:384]
X2H_FN int frag_u_of(bool last_open, int u_l) { return last_open ? u_l : -1; }
// FRAG_N = (last_open & lo_l >= s0) ? (hi_l - 1) // 128 - tile : 0  [src:386] (hi_l >= 2 whenever last_open)
X2H_FN int frag_n_of(bool last_open, int lo_l, int s0, int hi_l, int tile) {
  return (last_open && lo_l >= s0) ? (hi_l - 1) / BM - tile : 0;
}
// frow = frag_base + tile  [src:432]
X2H_FN int frag_row(int frag_base, int tile) { return frag_base + tile; }

// ---- W fragments [D2; C1.1 src0] ----------------------------------------------------------------------------------
// Wp[half][wave][cb][kb][lane] (16 B = 8 bf16); element e = W1[wp_k(kb, lane, e)][wp_col(half, w, cb, lane)]
X2H_FN int wp_index(int half, int w, int cb, int kb, int lane) {
  return (((half * NWAVE + w) * 2 + cb) * 16 + kb) * 64 + lane;
}
X2H_FN int wp_k(int kb, int lane, int e) { return 16 * kb + 8 * (lane >> 5) + e; }
X2H_FN int wp_col(int half, int w, int cb, int lane) {
  return half * NOUT + 64 * w + 32 * cb + (lane & 31);
}

// ---- A tile in LDS [D3; C1.1 src1] --------------------------------------------------------------------------------
// row r at byte r * 528; its 512-B dh row is DMA'd by lanes 0..31 (lane c -> 16-B chunk c at r * 528 + 16c)
X2H_FN int a_row_off(int r) { return r * APITCH; }
// rows gathered by wave w: 16w + j, j = 0..15  [D3]
X2H_FN int dma_row(int w, int j) { return 16 * w + j; }
// A fragment (rb, kb), lane l: row 32rb + l%32, k = 16kb + 8(l/32) .. +7 = bytes 32kb + 16(l/32) .. +15 of the row
//   byte offset = a_frag_lane_off(l) + a_frag_imm(rb, kb)  [D3]
X2H_FN int a_frag_lane_off(int lane) { return (lane & 31) * APITCH + 16 * (lane >> 5); }
X2H_FN int a_frag_imm(int rb, int kb) { return rb * 32 * APITCH + 32 * kb; }
// K block used at chain step s: s (contract: K ascending [C1.1]). X2H_PERTURB_KORDER=1 reverses it (15 - s) for A and W
// alike: a deliberately non-bitwise build for CONTRACT gate G3 (sensitivity). Never set it in production.
#ifndef X2H_PERTURB_KORDER
#define X2H_PERTURB_KORDER 0
#endif
X2H_FN int kstep(int s) { return X2H_PERTURB_KORDER ? 15 - s : s; }

// ---- main dot result layouts [C1.1 result; F a.1] -----------------------------------------------------------------
// accumulator item v of lane l for the 32x32 block (rb, q) of wave w:
//   ROLE_SWAP=1 (srcA = A, srcB = W):    position 32rb + 8(v/4) + 4(l/32) + v%4, column 64w + 32q + l%32
//   ROLE_SWAP=0 (srcA = W, srcB = A):    position 32rb + l%32, column 64w + 32q + 8(v/4) + 4(l/32) + v%4
X2H_FN int d32_row(int lane, int v) { return 8 * (v >> 2) + 4 * (lane >> 5) + (v & 3); }
X2H_FN int d32_col(int lane) { return lane & 31; }

// ---- indicator operands / result [C1.2; D5] -----------------------------------------------------------------------
// xb operand (srcA) of N-block nb, step rb: lane l, element e = xb[32rb + 8(l/16) + e][64w + 16nb + l%16]
// P operand (srcB), step rb: lane l, element e = P[slot l%16][32rb + 8(l/16) + e]
X2H_FN int p_pos(int rb, int lane, int e) { return 32 * rb + 8 * (lane >> 4) + e; }
X2H_FN int p_slot(int lane) { return lane & 15; }
// P tile in LDS: [rb][lane] x 16 B (one ds_read_b128 per rb)  [D5]
X2H_FN int pt_index(int rb, int lane) { return rb * 64 + lane; }
// result item r of lane l of N-block nb: val[slot l%16][column ind_col(w, nb, l, r)] (column inside the half)
X2H_FN int ind_col(int w, int nb, int lane, int r) {
  return 64 * w + 16 * nb + 4 * (lane >> 4) + r;
}

// ---- ROLE_SWAP=0 wave-private transpose scratch [D4; C1.2 remap] --------------------------------------------------
// Logical [32 positions][16 columns] bf16 of one N-block, row p at p * 40 B (40-B pitch: the 32 lanes of a 64-bit write
// hit 64 distinct banks), 8-B unit u (columns 4u..4u+3) at + 8u.
constexpr int XS_PITCH = 40;
X2H_FN int xs_off(int p, int unit) { return p * XS_PITCH + 8 * unit; }
// writer: lane l holds position l%32 and, for group g = 2h + gg of its 32x32 block, columns 8g + 4(l/32) + 0..3,
//   i.e. local unit 2gg + l/32 of N-block nb = 2q + h
X2H_FN int xs_write_unit(int gg, int lane) { return 2 * gg + (lane >> 5); }
// reader: ds_read_b64_tr_b16 #m (m = 0, 1): lane L reads row 8(L/16) + 4m + (L%16)/4, unit (L%16)%4. Within each 16-lane
// group, output lane i element r = element (i%4) of the 64 bits read by lane 4r + (i/4)%4 (Triton's model, CONTRACT
// 2.4 / MemoryOpToLLVM.cpp:124-176) -> lane L element e (= 4m + r) = xb[position 8(L/16) + e][column L%16].
X2H_FN int xs_tr_row(int lane, int m) { return 8 * (lane >> 4) + 4 * m + ((lane & 15) >> 2); }
X2H_FN int xs_tr_unit(int lane) { return lane & 3; }

// ---- global stores [C1.3; src:413, 419-424] -----------------------------------------------------------------------
// ACC[su][col] fp32 (64-bit offset), FRAG_V[frow][col], DX[t][hoff + col] bf16 (EMIT)
X2H_FN long long acc_off(int su, int col) { return (long long)su * NOUT + col; }
X2H_FN long long dx_byte_off(int t, uint32_t dx_row_bytes, int hoff, int col) {
  return (long long)((unsigned long long)(uint32_t)t * dx_row_bytes) + 2LL * (hoff + col);
}

// ---- metadata rings and the tile sequence [D6] --------------------------------------------------------------------
// Tile sequence of a WG: T_j = blockIdx + j * grid (j < 6), then 6 * grid + c (c = claim from the atomic counter).
X2H_FN int static_tile(int b, int j, int grid) { return b + j * grid; }
X2H_FN int dynamic_tile(int claim, int grid) { return NSTATIC * grid + claim; }
// Bounded residency (optional, sync review F4; compiled in only with -DX2H_BOUNDED_RESIDENCY=1, see claim_allowed):
// entry j >= NSTATIC of a WG's sequence is claimed only while j < max_seq; the WG then exits and frees its CU.
// max_seq >= NSTATIC (the launcher clamps it and raises the grid to cdiv(ntiles, max_seq) so that grid * max_seq >=
// ntiles covers every tile); 0x7FFFFFFF = unbounded (default).
X2H_FN bool seq_claimable(int j, int max_seq) { return j < max_seq; }
// launcher: max_seq of a launch (max_tiles <= 0: unbounded; else at least NSTATIC) and its grid: capped at ntiles and,
// when bounded, at least cdiv(ntiles, max_seq) (no overflow: max_seq < ntiles < 2^24 in that branch)
X2H_FN int launch_max_seq(int max_tiles) {
  return max_tiles <= 0 ? INVALID_TILE : (max_tiles < NSTATIC ? NSTATIC : max_tiles);
}
X2H_FN int launch_grid(int ntiles, int grid, int max_seq) {
  const int g = grid < ntiles ? grid : ntiles;
  const int need = max_seq >= ntiles ? 1 : (ntiles + max_seq - 1) / max_seq;
  return g > need ? g : need;
}
// W half loaded in the prologue: production cat members sort every album (half 1) position before the artist ones
// [M emu real_early: one parity switch, tile 18410], so a cat WG starts with half 1; item has half 0 only. Any start
// half is correct: a pass whose half differs reloads W (compute_tile).
X2H_FN int prologue_half(int is_cat) { return is_cat ? 1 : 0; }
// ring slot of T_j: ORDER j % 4, IDX j % 3, OFF / DER / PT / A j % 2, TIDX j % 8
X2H_FN int ord_slot(int j) { return j & 3; }
X2H_FN int idx_slot(int j) { return j % 3; }
X2H_FN int off_slot(int j) { return j & 1; }
X2H_FN int tidx_slot(int j) { return j & 7; }
// ORDER DMA source row for ring position r of tile s0: clamp to the last valid position (deterministic, in bounds)
X2H_FN int meta_src_row(int r, int s0, int npos) {
  return r < npos - 1 - s0 ? r : npos - 1 - s0;
}

// =====================================================================================================================
// 2. LDS layout
// =====================================================================================================================
// Distinct __shared__ objects per LDS-DMA target (A0, A1, ORD, IDX, OFL, OFH) so the compiler's LDS-DMA alias tracking
// never makes a ds_read of one wait for a DMA into another [F b.4]; DER / PT / TIDX / XS are written by ds_write only.
enum { SC_TILE = 0, SC_S0, SC_TEND, SC_NSEG, SC_HMIN, SC_NH, SC_NPASS, SC_LOPEN, SC_FU, SC_FN, SC_COUNT = 16 };
struct Der {                 // derived per-tile tables (written by DERIVE_WAVE one iteration ahead)
  uint16_t key[BM];          // pos_key per position
  int u[BM];                 // u = IDX[o] (int32) per position
  int hi[BM];                // OFF[u + 1] (int32) per position
  int t[BM];                 // dh row t per position (EMIT)
  int2 sl[S];                // slot table of the current pass: {su, shi} = {max u, max hi} over sel  [src:408-409]
  int sc[SC_COUNT];          // tile scalars
};
static_assert(sizeof(Der) == 1984, "Der layout");
constexpr int LDS_A = 2 * ABUF;                                  // 135,168
constexpr int LDS_META = (4 + 3 + 2 + 2) * BM * 4;               // ORD 4 + IDX 3 + OFL 2 + OFH 2 slots = 5,632
constexpr int LDS_DER = 2 * (int)sizeof(Der);                    // 3,968
constexpr int LDS_PT = 2 * 4 * 64 * 16;                          // 8,192
constexpr int LDS_TIDX = 8 * 4;                                  // 32
constexpr int LDS_XS = NWAVE * XS_BYTES;                         // 10,240 (ROLE_SWAP=0 only)
constexpr int lds_bytes(int role_swap) {
  return LDS_A + LDS_META + LDS_DER + LDS_PT + LDS_TIDX + (role_swap ? 0 : LDS_XS);
}
static_assert(lds_bytes(1) <= 163840 && lds_bytes(0) <= 163840, "LDS budget [D1]");

struct Lds {
  uint8_t* a0;
  uint8_t* a1;
  int* ord;     // [4][128] low dword of ORDER[s]
  int* idx;     // [3][128] low dword of IDX[o]
  int* ofl;     // [2][128] low dword of OFF[u]
  int* ofh;     // [2][128] low dword of OFF[u + 1]
  Der* der;     // [2]
  uint4* pt;    // [2][4][64]
  int* tidx;    // [8]
  uint8_t* xs;  // [8][1024] (ROLE_SWAP=0)
};

struct Params {
  const uint8_t* dh;           // [M, sdh] bf16 (bytes)
  const uint4* wp;             // prepacked W fragments [D2]
  const long long* idx;        // [T] int64
  const long long* order;      // [T] int64
  const long long* off;        // [U + 1] int64
  float* acc;                  // [U, 512] fp32
  int* frag_u;                 // [nt] int32
  int* frag_n;                 // [nt] int32
  float* frag_v;               // [nt, 512] fp32
  uint8_t* dx;                 // EMIT: DX [M, SW] bf16 (bytes)
  const uint8_t* zrow;         // >= 512 B of zero bytes (invalid rows) [D3]
  int* counters;               // [0] tile claims, [1] finished WGs (both self-reset to 0 by the last WG) [D6]
  uint32_t dx_row_bytes;       // SW * 2 (EMIT)
  uint32_t dh_row_bytes;       // sdh * 2 (< 2^31, multiple of 16)
  int npos;                    // T
  int ntiles;                  // cdiv(T, 128)
  int frag_base;
  int is_cat;
  int max_seq;                 // tiles per WG sequence (seq_claimable; read only with X2H_BOUNDED_RESIDENCY=1)
};

// Bounded residency is compiled in only with -DX2H_BOUNDED_RESIDENCY=1 (the lab's persistent-vs-bounded A/B, CRITIQUE
// 3.1 / 5.5; sync review F4). The per-iteration check costs R720 4-6 SGPR lane spills per kernel at the SGPR ceiling
// [M fix/out/probes], so the default build has no check (a constant true) and its launcher rejects max_tiles > 0.
#ifndef X2H_BOUNDED_RESIDENCY
#define X2H_BOUNDED_RESIDENCY 0
#endif
__device__ __forceinline__ bool claim_allowed(const Params& p, int j) {
#if X2H_BOUNDED_RESIDENCY
  return seq_claimable(j, p.max_seq);
#else
  (void)p;
  (void)j;
  return true;
#endif
}

// Loop-carried per-wave state. Wave-uniform fields live in SGPRs; st_su / st_flags are per lane.
struct Ctx {
  int w, B, G;
  int i;                          // iteration index: iteration i computes T_i (starts at -4: pipeline fill)
  int Tm1, T0, T1, T2, T3;        // T_{i-1} .. T_{i+3}  (T_{i+4} is read at the start of iteration i)
  int end_seen;                   // sticky: an invalid tile index was seen
  int cl_T5;                      // ORDER_WAVE: T_{i+5}
  int cur_half;                   // W half currently in Wr
  int st_valid, st_frow, st_fu, st_fn;  // deferred stores of T_{i-1}
  int st_su, st_flags;            // per lane
};

// =====================================================================================================================
// 3. Device building blocks
// =====================================================================================================================
__device__ __forceinline__ bf16x8 lds_ld16(const uint8_t* p) {
  return *reinterpret_cast<const bf16x8*>(__builtin_assume_aligned(p, 16));
}

// v_cvt_pk_bf16_f32 (hardware RNE) [C0.4; F b.3]
__device__ __forceinline__ unsigned cvt_pk(float lo, float hi) {
  const f32x2 v = {lo, hi};
  return __builtin_bit_cast(unsigned, __builtin_convertvector(v, bf16x2));
}
__device__ __forceinline__ void cvt8(const f32x16& C, unsigned (&d)[8]) {
#pragma unroll
  for (int j = 0; j < 8; ++j) d[j] = cvt_pk(C[2 * j], C[2 * j + 1]);
}

// Main dot step [C1.1; D4]: v_mfma_f32_32x32x16_bf16, modifiers 0.
template <int ROLE_SWAP>
__device__ __forceinline__ f32x16 mfma_main(const bf16x8& a, const bf16x8& wf, const f32x16& c) {
  if constexpr (ROLE_SWAP) return __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, wf, c, 0, 0, 0);  // srcA = A (H-swap)
  else return __builtin_amdgcn_mfma_f32_32x32x16_bf16(wf, a, c, 0, 0, 0);                      // srcA = W (Triton)
}
// Indicator step [C1.2; D5]: v_mfma_f32_16x16x32_bf16, srcA = xb, srcB = P (Triton's roles in both variants).
__device__ __forceinline__ f32x4 mfma_ind(const bf16x8& x, const bf16x8& p, const f32x4& c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_bf16(x, p, c, 0, 0, 0);
}

// ROLE_SWAP=1: one 32x32 C block -> two 16x16x32 operands (positions along K), 8 cvt + 4 permlane32 + 4 permlane16
// [F a.4; facts/permlane_model.py c32_to_b16_transpose]. X0 = columns 0..15 of the block, X1 = 16..31.
__device__ __forceinline__ void t4(unsigned& a, unsigned& b, unsigned& c, unsigned& d) {
  u32x2 r;
  r = __builtin_amdgcn_permlane32_swap(a, c, false, false); a = r.x; c = r.y;
  r = __builtin_amdgcn_permlane32_swap(b, d, false, false); b = r.x; d = r.y;
  r = __builtin_amdgcn_permlane16_swap(a, b, false, false); a = r.x; b = r.y;
  r = __builtin_amdgcn_permlane16_swap(c, d, false, false); c = r.x; d = r.y;
}
__device__ __forceinline__ void xb_permlane(unsigned (&d)[8], bf16x8& X0, bf16x8& X1) {
  t4(d[0], d[2], d[4], d[6]);
  t4(d[1], d[3], d[5], d[7]);
  const u32x4 h0 = {d[0], d[1], d[4], d[5]};
  const u32x4 h1 = {d[2], d[3], d[6], d[7]};
  X0 = __builtin_bit_cast(bf16x8, h0);
  X1 = __builtin_bit_cast(bf16x8, h1);
}

// ROLE_SWAP=0: D (Triton roles) -> the two 16x16x32 srcA operands of one 32x32 block through the wave-private scratch
// (32 rows x 40 B) [D4; C1.2 remap]. ONE asm statement: for N-block h = 0, then h = 1, 2 ds_write_b64 (groups 2h, 2h+1)
// + 2 ds_read_b64_tr_b16 (m = 0, 1); then `s_waitcnt lgkmcnt(0)` INSIDE the statement.
// Why the wait is inside (FIX B1, fix/FIX_REPORT.md): the compiler does not see LDS operations inside inline asm
// (SIInsertWaitcnts adds no wait for them), and `__builtin_amdgcn_s_waitcnt` is IntrNoMem (IntrinsicsAMDGPU.td:390-391),
// so with a separate wait builtin the scheduler hoisted the (pure) indicator MFMA above the wait and it read the tr-read
// registers before the LDS data had returned (pre-fix ISA: 24-28 of 64 tr reads per ROLE_SWAP=0 kernel). Here the
// outputs exist, for the compiler, only where the statement ends, i.e. after the wait: neither a consumer nor a
// register-allocator copy can read them earlier. "=&v" (early clobber): no output shares a register with an input, so
// a tr read in flight cannot overwrite the address or the h = 1 write data that later instructions of the statement read.
// LDS order: the h = 1 writes reuse the h = 0 slots after the h = 0 tr reads were issued; LDS operations of one wave are
// performed and returned in issue order (CDNA4 ISA 4.4; AMDGPUUsage GFX942 memory model), so no wait is needed between
// them (IMPLEMENTATION 7.3). Inline asm rather than the compiler-visible tr intrinsic on purpose: the compiler attaches
// no LDS alias info to that intrinsic, so a compiler-visible read got an `s_waitcnt vmcnt(0)` for the in-flight A DMA
// inside the MFMA loop (measured on both toolchains). G0 C8 (isa_check.py) checks the compiled waits.
//   xs_w: LDS byte address of this lane's write slot  xs_off(lane%32, xs_write_unit(0, lane))  (gg = 1 at +16)
//   xs_r: LDS byte address of this lane's tr-read slot xs_off(xs_tr_row(lane, 0), xs_tr_unit(lane)) (m = 1 at +160)
static_assert(xs_off(0, xs_write_unit(1, 0)) - xs_off(0, xs_write_unit(0, 0)) == 16, "gg immediate");
static_assert(xs_off(xs_tr_row(0, 1), 0) - xs_off(xs_tr_row(0, 0), 0) == 160, "m immediate");
__device__ __forceinline__ void xb_lds_tr2(const unsigned (&d)[8], uint32_t xs_w, uint32_t xs_r, bf16x8& X0,
                                           bf16x8& X1) {
  const u32x2 a0 = {d[0], d[1]};  // h = 0, group g = 0 (columns  0 + 4(l/32) + 0..3)
  const u32x2 a1 = {d[2], d[3]};  // h = 0, group g = 1 (columns  8 + 4(l/32) + 0..3)
  const u32x2 b0 = {d[4], d[5]};  // h = 1, group g = 2 (columns 16 + 4(l/32) + 0..3)
  const u32x2 b1 = {d[6], d[7]};  // h = 1, group g = 3 (columns 24 + 4(l/32) + 0..3)
  u32x2 r0, r1, r2, r3;
  asm volatile(
      "ds_write_b64 %4, %6\n\t"
      "ds_write_b64 %4, %7 offset:16\n\t"
      "ds_read_b64_tr_b16 %0, %5\n\t"
      "ds_read_b64_tr_b16 %1, %5 offset:160\n\t"
      "ds_write_b64 %4, %8\n\t"
      "ds_write_b64 %4, %9 offset:16\n\t"
      "ds_read_b64_tr_b16 %2, %5\n\t"
      "ds_read_b64_tr_b16 %3, %5 offset:160\n\t"
      "s_waitcnt lgkmcnt(0)"
      : "=&v"(r0), "=&v"(r1), "=&v"(r2), "=&v"(r3)
      : "v"(xs_w), "v"(xs_r), "v"(a0), "v"(a1), "v"(b0), "v"(b1));
  const u32x4 x0 = {r0.x, r0.y, r1.x, r1.y};  // N-block h = 0
  const u32x4 x1 = {r2.x, r2.y, r3.x, r3.y};  // N-block h = 1
  X0 = __builtin_bit_cast(bf16x8, x0);
  X1 = __builtin_bit_cast(bf16x8, x1);
}

// An SGPR 0 the compiler cannot see through. Added to LDS bases, the W base and a few constants once per iteration so
// that loop-invariant addresses / constants are not hoisted out of the persistent loop into VGPRs or SGPRs held for
// the whole kernel (they would otherwise eat the VGPR budget). Pointers keep their underlying __shared__ object, so
// LDS alias scopes are unchanged.
__device__ __forceinline__ int opaque_zero() {
  int z;
  asm volatile("s_mov_b32 %0, 0" : "=s"(z));
  return z;
}

// gfx950 buffer resource (W3 make_rsrc, wgrad_tm_ck_w3.cu:84-92): raw buffer, range check = voffset + inst_offset <
// num_records (soffset is not range-checked, CDNA4 ISA 9.1.5.1).
__device__ __forceinline__ i32x4 make_rsrc(const void* q, uint32_t nrec) {
  const uint64_t a = reinterpret_cast<uint64_t>(q);
  i32x4 r;
  r.x = (int)(uint32_t)a;
  r.y = (int)(uint32_t)(a >> 32);
  r.z = (int)nrec;
  r.w = 0x00110000;
  return r;
}

// A VGPR 0 the compiler must treat as divergent. Used as the claim atomic's address offset so LLVM's AMDGPU atomic
// optimizer (uniform address -> wave-level rewrite with a readfirstlane right after the atomic, which forces an early
// `s_waitcnt vmcnt(0)`) leaves the lane-0 atomic alone; its result is read only after the per-tile barrier.
__device__ __forceinline__ int opaque_vzero() {
  int z;
  asm volatile("v_mov_b32 %0, 0" : "=v"(z));
  return z;
}

// lane id recomputed per phase from threadIdx.x + an opaque zero: lane-derived addresses are then rebuilt inside
// each phase instead of being hoisted out of the persistent loop (and held in VGPRs across the compute).
__device__ __forceinline__ int lane_id() { return ((int)threadIdx.x + opaque_zero()) & 63; }

// W fragments of half h into registers [D2]: 32 x buffer_load_dwordx4, voffset = lane*16 (VGPR), soffset = byte offset
// of fragment (half, w, cb, kb) (SGPR). Byte offset of (half, w, cb, kb, lane) = 16 * wp_index.
__device__ __forceinline__ void load_w(bf16x8 (&Wr)[2][16], const uint4* wp, int half, int w, int lane) {
  const i32x4 rs = make_rsrc(wp, 0x7FFFFFFFu);
  const int sbase = wp_index(half, w, 0, 0, 0) * 16 + opaque_zero();
  const int lane16 = lane * 16;
#pragma unroll
  for (int cb = 0; cb < 2; ++cb)
#pragma unroll
    for (int kb = 0; kb < 16; ++kb)
      Wr[cb][kb] = __builtin_bit_cast(
          bf16x8, x2h_llvm_buffer_load_v4i32(rs, lane16, sbase + wp_index(0, 0, cb, kb, 0) * 16, 0));
}

// P-operand dword from two keys: element = (key == target) ? bf16 1.0 (0x3F80) : +0.0  [C1.2 src1]
// one_lo = 0x3F80, one_hi = 0x3F800000 (passed in SGPRs).
__device__ __forceinline__ unsigned p_pair(unsigned keypair, unsigned tgt, unsigned one_lo, unsigned one_hi) {
  return ((keypair & 0xFFFFu) == tgt ? one_lo : 0u) | ((keypair >> 16) == tgt ? one_hi : 0u);
}

// Slot tables (su / shi) and P tile of pass (g, h) from the Der key / u / hi tables. Called by one wave (in-order LDS:
// init -> atomics -> reads). [src:406-409; D5]
__device__ __forceinline__ void build_pass_tables(Der& der, uint4* pt, int g, int h, int lane) {
  const int z = opaque_zero();
  const unsigned one_lo = 0x3F80u + (unsigned)z, one_hi = 0x3F800000u + (unsigned)z;
  if (lane < S) der.sl[lane] = int2{-1 + z, -1 + z};
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int r = lane + 64 * k;
    const uint32_t key = der.key[r];
    const int j = key_slot(key, g);
    if (key_rmask(key, h) && j >= 0 && j < S) {
      atomicMax(&der.sl[j].x, der.u[r]);
      atomicMax(&der.sl[j].y, der.hi[r]);
    }
  }
  const uint32_t tgt = pass_target(g, h, p_slot(lane));
#pragma unroll
  for (int rb = 0; rb < 4; ++rb) {
    const uint4 kk = *reinterpret_cast<const uint4*>(&der.key[p_pos(rb, lane, 0)]);
    uint4 o;
    o.x = p_pair(kk.x, tgt, one_lo, one_hi);
    o.y = p_pair(kk.y, tgt, one_lo, one_hi);
    o.z = p_pair(kk.z, tgt, one_lo, one_hi);
    o.w = p_pair(kk.w, tgt, one_lo, one_hi);
    pt[pt_index(rb, lane)] = o;
  }
}

struct DerIn {
  int o[2], u[2], lo[2], hi[2];  // positions lane and lane + 64
};

__device__ __forceinline__ unsigned mbcnt64(uint64_t m) {
  return __builtin_amdgcn_mbcnt_hi((uint32_t)(m >> 32), __builtin_amdgcn_mbcnt_lo((uint32_t)m, 0u));
}

// Per-tile derive (DERIVE_WAVE, one iteration ahead): start flags, kr, nseg, keys, hmin / nh / npass, fragment
// scalars, then the pass-0 slot tables and P tile. [src:356-390, 398-409]
__device__ __forceinline__ void derive_tile(const Params& p, int is_cat, int tile, int lane, const DerIn& d, Der& der,
                                            uint4* pt) {
  const int s0 = tile_s0(tile);
  const int tend = tile_end_of(s0, p.npos);
  bool vv[2], st[2];
  int half[2];
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int r = lane + 64 * k;
    const int s = s0 + r;
    vv[k] = pos_valid(s, p.npos);
    half[k] = order_to_half(d.o[k], is_cat);
    st[k] = start_flag(vv[k], r, d.lo[k], s);
  }
  const uint64_t m0 = __builtin_amdgcn_ballot_w64(st[0]);
  const uint64_t m1 = __builtin_amdgcn_ballot_w64(st[1]);
  const int c0 = __builtin_popcountll(m0);
  const int kr0 = kr_of((int)mbcnt64(m0), st[0]);
  const int kr1 = kr_of(c0 + (int)mbcnt64(m1), st[1]);
  const int nseg = c0 + __builtin_popcountll(m1);
  const uint64_t h0 = __builtin_amdgcn_ballot_w64(vv[0] && half[0] == 0) | __builtin_amdgcn_ballot_w64(vv[1] && half[1] == 0);
  const uint64_t h1 = __builtin_amdgcn_ballot_w64(vv[0] && half[0] == 1) | __builtin_amdgcn_ballot_w64(vv[1] && half[1] == 1);
  const int hmin = hmin_of(h0 != 0);
  const int hmax = hmax_of(h1 != 0);
  const int nh = hmax - hmin + 1;
  const int npass = npass_of(nseg, nh);
  const int rl = tend - 1 - s0;  // row of the tile's last valid position (wave-uniform)
  const int u_l = rl < 64 ? __builtin_amdgcn_readlane(d.u[0], rl) : __builtin_amdgcn_readlane(d.u[1], rl - 64);
  const int lo_l = rl < 64 ? __builtin_amdgcn_readlane(d.lo[0], rl) : __builtin_amdgcn_readlane(d.lo[1], rl - 64);
  const int hi_l = rl < 64 ? __builtin_amdgcn_readlane(d.hi[0], rl) : __builtin_amdgcn_readlane(d.hi[1], rl - 64);
  const bool lopen = last_open_of(hi_l, tend);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int r = lane + 64 * k;
    der.key[r] = (uint16_t)pos_key(vv[k], half[k], k ? kr1 : kr0);
    der.u[r] = d.u[k];
    der.hi[r] = d.hi[k];
    der.t[r] = order_to_t(d.o[k], is_cat);
  }
  if (lane == 0) {  // tile scalars: 3 x ds_write_b128 from one base
    int4* sc4 = reinterpret_cast<int4*>(der.sc);
    sc4[0] = int4{tile, s0, tend, nseg};                                              // SC_TILE .. SC_NSEG
    sc4[1] = int4{hmin, nh, npass, lopen ? 1 : 0};                                    // SC_HMIN .. SC_LOPEN
    sc4[2] = int4{frag_u_of(lopen, u_l), frag_n_of(lopen, lo_l, s0, hi_l, tile), 0, 0};  // SC_FU, SC_FN
  }
  build_pass_tables(der, pt, 0, hmin, lane);
}

// Metadata LDS-DMA (4-B dword per lane, M0 = ring slot base + 256k, lane l -> +4l) [D6]
__device__ __forceinline__ void order_dma(const Params& p, int* slot, int tile, int lane) {
  const int s0 = tile_s0(tile);
  const uint8_t* base = reinterpret_cast<const uint8_t*>(p.order) + (long long)s0 * 8;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int r = meta_src_row(lane + 64 * k, s0, p.npos);
    __builtin_amdgcn_global_load_lds(X2H_AS1(base + (uint32_t)r * 8u), X2H_AS3(slot + 64 * k), 4, 0, 0);
  }
}
__device__ __forceinline__ void idx_dma(const Params& p, int* slot, const int (&o)[2]) {
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const uint8_t* g = reinterpret_cast<const uint8_t*>(p.idx) + (unsigned long long)(uint32_t)o[k] * 8ull;
    __builtin_amdgcn_global_load_lds(X2H_AS1(g), X2H_AS3(slot + 64 * k), 4, 0, 0);
  }
}
__device__ __forceinline__ void off_dma(const Params& p, int* slot_lo, int* slot_hi, const int (&u)[2]) {
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const uint8_t* g = reinterpret_cast<const uint8_t*>(p.off) + (long long)u[k] * 8;
    __builtin_amdgcn_global_load_lds(X2H_AS1(g), X2H_AS3(slot_lo + 64 * k), 4, 0, 0);
    __builtin_amdgcn_global_load_lds(X2H_AS1(g + 8), X2H_AS3(slot_hi + 64 * k), 4, 0, 0);
  }
}

// byte offset of dh row t: t * row_bytes (32-bit; the host guarantees M * row_bytes < 2^31)  [D3; src:374]
X2H_FN int dh_row_soff(int t, uint32_t row_bytes) { return (int)((uint32_t)t * row_bytes); }

// A gather of the next tile [D3]: wave w DMAs rows 16w..16w+15, one buffer_load_dwordx4 ... lds per row with
// EXEC = lanes 0..31 (32 x 16 B = the 512-B dh row): voffset = 16 * lane, soffset = t * row_bytes (SGPR),
// M0 = A buffer + r * 528. Invalid rows (only in the last tile) read the 512-B zero row instead.
// t_lane: lane j (< 16) holds t of row 16w + j.
__device__ __forceinline__ void a_dma(const Params& p, uint8_t* abuf, int s0n, int w, int lane, int t_lane) {
  const int lane16 = lane * 16;
  uint8_t* awave = abuf + opaque_zero() + a_row_off(dma_row(w, 0));
  const i32x4 rs = make_rsrc(p.dh, 0x7FFFFFFFu);
  const bool full = s0n + BM <= p.npos;
  if (lane < 32) {
    if (full) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int t = __builtin_amdgcn_readlane(t_lane, j);
        x2h_llvm_buffer_load_lds(rs, (lds_u32_ptr_t)X2H_AS3(awave + a_row_off(j)), 16, lane16,
                                 dh_row_soff(t, p.dh_row_bytes), 0, 0);
      }
    } else {
      const i32x4 rz = make_rsrc(p.zrow, 0x7FFFFFFFu);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const int t = __builtin_amdgcn_readlane(t_lane, j);
        if (pos_valid(s0n + dma_row(w, j), p.npos))
          x2h_llvm_buffer_load_lds(rs, (lds_u32_ptr_t)X2H_AS3(awave + a_row_off(j)), 16, lane16,
                                   dh_row_soff(t, p.dh_row_bytes), 0, 0);
        else
          x2h_llvm_buffer_load_lds(rz, (lds_u32_ptr_t)X2H_AS3(awave + a_row_off(j)), 16, lane16, 0, 0, 0);
      }
    }
  }
}

// ACC / FRAG_V stores of one pass from the indicator accumulators (lane l: slot l%16, 4 columns per N-block) [C1.3]
__device__ __forceinline__ void store_pass(const Params& p, int w, int lane, int su, int flags, int frow,
                                           const f32x4 (&ind)[4]) {
  const int col0 = ind_col(w, 0, lane, 0);
  if (flags & 1) {
    float* pa = p.acc + acc_off(su, col0);
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) *reinterpret_cast<f32x4*>(pa + 16 * nb) = ind[nb];
  }
  if (flags & 2) {
    float* pf = p.frag_v + acc_off(frow, col0);
#pragma unroll
    for (int nb = 0; nb < 4; ++nb) *reinterpret_cast<f32x4*>(pf + 16 * nb) = ind[nb];
  }
}

// EMIT (debug): DX[t][hoff + col] = xb for rmask rows [D7; src:418-420].
// ROLE_SWAP=0 from the D dwords (lane: one position, 4 consecutive columns per group g).
__device__ __forceinline__ void emit_r0(const Params& p, const Der& der, int rb, int q, int w, int lane, int h,
                                        int hoff, const unsigned (&d)[8]) {
  const int pos = 32 * rb + (lane & 31);
  const uint32_t key = der.key[pos];
  if (key_rmask(key, h)) {
    uint8_t* base = p.dx + dx_byte_off(der.t[pos], p.dx_row_bytes, hoff, 64 * w + 32 * q + 4 * (lane >> 5));
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const u32x2 v = {d[2 * g], d[2 * g + 1]};
      *reinterpret_cast<u32x2*>(base + 16 * g) = v;
    }
  }
}
// ROLE_SWAP=1 from the two xb operands of block q (lane L: 8 positions, one column per operand).
__device__ __forceinline__ void emit_r1(const Params& p, const Der& der, int rb, int q, int w, int lane, int h,
                                        int hoff, const bf16x8& X0, const bf16x8& X1) {
  const u32x4 x0 = __builtin_bit_cast(u32x4, X0);
  const u32x4 x1 = __builtin_bit_cast(u32x4, X1);
  const int col = 64 * w + 32 * q + (lane & 15);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int pos = p_pos(rb, lane, e);
    const uint32_t key = der.key[pos];
    if (key_rmask(key, h)) {
      uint8_t* base = p.dx + dx_byte_off(der.t[pos], p.dx_row_bytes, hoff, col);
      const unsigned w0 = x0[e >> 1], w1 = x1[e >> 1];
      *reinterpret_cast<uint16_t*>(base) = (uint16_t)((e & 1) ? (w0 >> 16) : (w0 & 0xFFFFu));
      *reinterpret_cast<uint16_t*>(base + 32) = (uint16_t)((e & 1) ? (w1 >> 16) : (w1 & 0xFFFFu));
    }
  }
}

// One pass over the tile: 4 row blocks x (32 main MFMAs + xb conversion + 4 indicator MFMAs). [C1.1, C1.2; D4, D5]
template <int ROLE_SWAP, int EMIT>
__device__ __forceinline__ void compute_pass(const Params& p, const uint8_t* abuf, const uint4* pt, const Der& der,
                                             uint8_t* xs, int w, int lane, int h, int hoff,
                                             const bf16x8 (&Wr)[2][16], f32x4 (&ind)[4]) {
  const uint8_t* ap = abuf + a_frag_lane_off(lane);
  // ROLE_SWAP=0 scratch slots of this lane (LDS byte addresses for the asm in xb_lds_tr2)
  const uint32_t xs_w = (uint32_t)(uintptr_t)xs + (uint32_t)xs_off(lane & 31, xs_write_unit(0, lane));
  const uint32_t xs_r = (uint32_t)(uintptr_t)xs + (uint32_t)xs_off(xs_tr_row(lane, 0), xs_tr_unit(lane));
  // A fragments (rb, 0) and (rb, 1) are prefetched before the previous row block's conversion phase, so the next
  // row block's chain heads do not wait for LDS (load timing only; operand values unchanged).
  bf16x8 a_k0 = lds_ld16(ap + a_frag_imm(0, kstep(0)));
  bf16x8 a_k1 = lds_ld16(ap + a_frag_imm(0, kstep(1)));
#pragma unroll
  for (int rb = 0; rb < 4; ++rb) {
    f32x16 C0, C1;
    const bf16x8 P = *reinterpret_cast<const bf16x8*>(&pt[pt_index(rb, lane)]);  // P operand of step rb [D5]
    bf16x8 a = a_k0;
    {  // K step 0: chain heads with C = +0.0 [C1.1]
      C0 = mfma_main<ROLE_SWAP>(a, Wr[0][kstep(0)], f32x16{});
      C1 = mfma_main<ROLE_SWAP>(a, Wr[1][kstep(0)], f32x16{});
      X2H_SB();
      a = a_k1;
    }
#pragma unroll
    for (int kb = 1; kb < 16; ++kb) {  // K ascending [C1.1]
      bf16x8 an = a;
      if (kb < 15) an = lds_ld16(ap + a_frag_imm(rb, kstep(kb + 1)));
      C0 = mfma_main<ROLE_SWAP>(a, Wr[0][kstep(kb)], C0);
      C1 = mfma_main<ROLE_SWAP>(a, Wr[1][kstep(kb)], C1);
      X2H_SB();
      a = an;
    }
    if (rb < 3) {
      a_k0 = lds_ld16(ap + a_frag_imm(rb + 1, kstep(0)));
      a_k1 = lds_ld16(ap + a_frag_imm(rb + 1, kstep(1)));
    }
    X2H_SB();  // keep the prefetch ahead of the conversion phase
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      unsigned d[8];
      cvt8(q ? C1 : C0, d);  // xb = bf16_RNE(acc) [C1.1 rounding]
      bf16x8 X0, X1;
      if constexpr (ROLE_SWAP) {
        xb_permlane(d, X0, X1);
      } else {
        xb_lds_tr2(d, xs_w, xs_r, X0, X1);  // the lgkmcnt(0) for the tr reads is inside the asm (FIX B1)
      }
      // indicator chains: N-blocks 2q, 2q+1; step rb; head (rb == 0) with C = +0.0 [C1.2]
      ind[2 * q] = mfma_ind(X0, P, rb ? ind[2 * q] : f32x4{});
      ind[2 * q + 1] = mfma_ind(X1, P, rb ? ind[2 * q + 1] : f32x4{});
      if constexpr (EMIT) {  // debug addends, fenced off from the compute schedule
        X2H_SB();
        if constexpr (ROLE_SWAP) emit_r1(p, der, rb, q, w, lane, h, hoff, X0, X1);
        else emit_r0(p, der, rb, q, w, lane, h, hoff, d);
        X2H_SB();
      }
    }
    X2H_SB();
  }
}

__device__ __forceinline__ int rfl(int v) { return __builtin_amdgcn_readfirstlane(v); }

// Compute tile T_i (A buffer / DER / PT of parity PAR): all passes in Triton's pz order; the last pass's stores are
// deferred past the end-of-tile barrier [CR B3; D6], earlier passes (slow path) store at once.
template <int ROLE_SWAP, int EMIT, int PAR>
__device__ __forceinline__ void compute_tile(const Params& p, Ctx& c, const Lds& L, int w, int is_cat,
                                             bf16x8 (&Wr)[2][16], f32x4 (&ind)[4]) {
  Der& der = L.der[PAR];
  uint4* pt = L.pt + PAR * 256;
  const uint8_t* abuf = PAR ? L.a1 : L.a0;
  uint8_t* xs = L.xs + (ROLE_SWAP ? 0 : w * XS_BYTES);
  const int lane = lane_id();
  const int tile = c.T0;
  const int4* sc4 = reinterpret_cast<const int4*>(der.sc);  // 3 x ds_read_b128 from one base
  const int4 sa = sc4[0], sb = sc4[1], sf = sc4[2];
  const int tend = rfl(sa.z);           // SC_TEND
  const int hmin = rfl(sb.x);           // SC_HMIN
  const int nh = rfl(sb.y);             // SC_NH
  const int npass = rfl(sb.z);          // SC_NPASS
  const bool lopen = rfl(sb.w) != 0;    // SC_LOPEN
  const int fu = rfl(sf.x);             // SC_FU
  const int fn = rfl(sf.y);             // SC_FN
  const int frow = frag_row(p.frag_base, tile);
  // Passes in Triton's pz order [src:391-392, 400]. Pass 0's tables were built by the derive one iteration ahead; the
  // slow path (> 16 segments or both cat halves) rebuilds them at the bottom of the body between two barriers.
  int pz = 0;
#pragma clang loop unroll(disable)
  for (;;) {
    const int h = pass_h(pz, nh, hmin);
    if (h != c.cur_half) {  // slow path: W half switch (cat) [D2]
      load_w(Wr, p.wp, h, w, lane);
      __builtin_amdgcn_s_waitcnt(WAIT_VM0);
      c.cur_half = h;
      X2H_SB();
    }
    compute_pass<ROLE_SWAP, EMIT>(p, abuf, pt, der, xs, w, lane, h, pass_hoff(h, is_cat), Wr, ind);
    const int2 sl = der.sl[p_slot(lane)];  // one ds_read_b64
    const int su = sl.x;
    const int flags = slot_flags(su, sl.y, tend, lopen);
    if (pz + 1 >= npass) {  // last pass: stores deferred past the end-of-tile barrier
      c.st_su = su;
      c.st_flags = flags;
      break;
    }
    store_pass(p, w, lane, su, flags, frow, ind);
    ++pz;
    X2H_WAIT_BARRIER(WAIT_LGKM0);  // every wave has read this pass's P tile / slot table (WAR)
    if (w == DERIVE_WAVE) build_pass_tables(der, pt, pass_g(pz, nh), pass_h(pz, nh, hmin), lane);
    X2H_WAIT_BARRIER(WAIT_LGKM0);  // the next pass's tables are visible (RAW)
    X2H_SB();
  }
  c.st_valid = 1;
  c.st_frow = frow;
  c.st_fu = fu;
  c.st_fn = fn;
}

// One iteration of the persistent loop (PAR = i & 1). Returns false when the WG has no more tiles.
//   (1) deferred stores of T_{i-1}  (2) T_{i+4}  (3) publish T_{i+5}  (4) ring reads  (5) derive T_{i+1}
//   (6) DMAs: ORDER T_{i+4}, IDX T_{i+3}, OFF T_{i+2}, A T_{i+1}, claim T_{i+6}  (7) compute T_i
//   (8) s_waitcnt vmcnt(0) lgkmcnt(0) + s_barrier (the only per-tile wait)  (9) claim result  (10) shift.
// Program order matters for the compiler's LDS-DMA wait analysis: every LDS access that may lack alias info (ring
// reads, merged derive writes, LDS atomics) precedes the first DMA of the iteration, so no `s_waitcnt vmcnt` is
// inserted before the per-tile one (isa_check.py verifies this).
template <typename T>
__device__ __forceinline__ T* lds_plus(T* q, int z) {
  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(q) + z);
}
// The iteration's view of the LDS: every base + an opaque SGPR zero (see opaque_zero).
__device__ __forceinline__ Lds lds_view(const Lds& L, int z) {
  return Lds{lds_plus(L.a0, z),  lds_plus(L.a1, z),  lds_plus(L.ord, z), lds_plus(L.idx, z),
             lds_plus(L.ofl, z), lds_plus(L.ofh, z), lds_plus(L.der, z), lds_plus(L.pt, z),
             lds_plus(L.tidx, z), lds_plus(L.xs, z)};
}

template <int ROLE_SWAP, int EMIT, int PAR>
__device__ __forceinline__ bool iteration(const Params& p, Ctx& c, const Lds& L0, bf16x8 (&Wr)[2][16],
                                          f32x4 (&ind)[4]) {
  const int i = c.i;
  const int z = opaque_zero();
  const Lds L = lds_view(L0, z);
  const int w = c.w + z;            // wave role tests and is_cat are re-derived per iteration (not hoisted as masks)
  const int is_cat = p.is_cat + z;
  // (1)
  if (c.st_valid) {
    const int lane1 = lane_id();
    store_pass(p, w, lane1, c.st_su, c.st_flags, c.st_frow, ind);
    if (w == ORDER_WAVE && lane1 == 0) {
      p.frag_u[c.st_frow] = c.st_fu;
      p.frag_n[c.st_frow] = c.st_fn;
    }
    c.st_valid = 0;
  }
  X2H_SB();
  if (i >= 0 && !tile_valid(c.T0, p.ntiles)) return false;
  // (2)
  int T4;
  if (i + 4 < NSTATIC) T4 = static_tile(c.B, i + 4, c.G);
  else T4 = rfl(L.tidx[tidx_slot(i + 4)]);
  if (c.end_seen || !tile_valid(T4, p.ntiles)) {
    T4 = INVALID_TILE;
    c.end_seen = 1;
  }
  const bool v1 = tile_valid(c.T1, p.ntiles), v2 = tile_valid(c.T2, p.ntiles);
  const bool v3 = tile_valid(c.T3, p.ntiles), v4 = tile_valid(T4, p.ntiles);
  const int lane = lane_id();
  // (3)
  if (w == ORDER_WAVE && i + 5 >= NSTATIC && lane == 0) L.tidx[tidx_slot(i + 5)] = c.cl_T5;
  // (4) every read of a DMA-target ring precedes this iteration's DMAs
  int t_lane = 0;
  if (v1) t_lane = order_to_t(L.ord[ord_slot(i + 1) * BM + dma_row(w, lane & 15)], is_cat);
  int ord3[2] = {0, 0}, idx2[2] = {0, 0};
  DerIn dv = {};
  if (w == IDX_WAVE && v3) {
#pragma unroll
    for (int k = 0; k < 2; ++k) ord3[k] = L.ord[ord_slot(i + 3) * BM + lane + 64 * k];
  }
  if (w == OFF_WAVE && v2) {
#pragma unroll
    for (int k = 0; k < 2; ++k) idx2[k] = L.idx[idx_slot(i + 2) * BM + lane + 64 * k];
  }
  if (w == DERIVE_WAVE && v1) {
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int r = lane + 64 * k;
      dv.o[k] = L.ord[ord_slot(i + 1) * BM + r];
      dv.u[k] = L.idx[idx_slot(i + 1) * BM + r];
      dv.lo[k] = L.ofl[off_slot(i + 1) * BM + r];
      dv.hi[k] = L.ofh[off_slot(i + 1) * BM + r];
    }
  }
  X2H_SB();
  // (5)
  if (w == DERIVE_WAVE && v1) derive_tile(p, is_cat, c.T1, lane, dv, L.der[PAR ^ 1], L.pt + (PAR ^ 1) * 256);
  X2H_SB();
  // (6)
  int v_claim = 0;
  if (w == ORDER_WAVE) {
    if (v4) order_dma(p, L.ord + ord_slot(i + 4) * BM, T4, lane);
    if (i >= 0 && tile_valid(c.cl_T5, p.ntiles) && claim_allowed(p, i + 6) && lane == 0)
      v_claim = __hip_atomic_fetch_add(p.counters + opaque_vzero(), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  if (w == IDX_WAVE && v3) idx_dma(p, L.idx + idx_slot(i + 3) * BM, ord3);
  if (w == OFF_WAVE && v2) off_dma(p, L.ofl + off_slot(i + 2) * BM, L.ofh + off_slot(i + 2) * BM, idx2);
  if (v1) a_dma(p, PAR ? L.a0 : L.a1, tile_s0(c.T1), w, lane, t_lane);
  X2H_SB();
  // (7)
  if (tile_valid(c.T0, p.ntiles)) compute_tile<ROLE_SWAP, EMIT, PAR>(p, c, L, w, is_cat, Wr, ind);
  X2H_SB();
  // (8)
  X2H_WAIT_BARRIER(WAIT_VM0_LGKM0);
  X2H_SB();
  // (9) ORDER_WAVE: T_{i+6}. The asm pins the use of the claim below the barrier (otherwise its readfirstlane is
  // hoisted next to the atomic and forces an early vmcnt(0) on wave 0).
  asm volatile("" : "+v"(v_claim));
  {
    int nxt = INVALID_TILE;
    if (i + 6 < NSTATIC) nxt = static_tile(c.B, i + 6, c.G);
    else if (i >= 0 && tile_valid(c.cl_T5, p.ntiles) && claim_allowed(p, i + 6))
      nxt = dynamic_tile(rfl(v_claim), c.G);
    c.cl_T5 = tile_valid(nxt, p.ntiles) ? nxt : INVALID_TILE;
  }
  // (10)
  c.Tm1 = c.T0;
  c.T0 = c.T1;
  c.T1 = c.T2;
  c.T2 = c.T3;
  c.T3 = T4;
  c.i = i + 1;
  return true;
}

// =====================================================================================================================
// 4. Kernels
// =====================================================================================================================
template <int ROLE_SWAP, int EMIT>
__global__ __launch_bounds__(NTHREADS) void x2h_pass(Params p) {
  __shared__ __attribute__((aligned(16))) uint8_t sA0[ABUF];
  __shared__ __attribute__((aligned(16))) uint8_t sA1[ABUF];
  __shared__ __attribute__((aligned(16))) int sORD[4 * BM];
  __shared__ __attribute__((aligned(16))) int sIDX[3 * BM];
  __shared__ __attribute__((aligned(16))) int sOFL[2 * BM];
  __shared__ __attribute__((aligned(16))) int sOFH[2 * BM];
  __shared__ __attribute__((aligned(16))) Der sDER[2];
  __shared__ __attribute__((aligned(16))) uint4 sPT[2 * 4 * 64];
  __shared__ __attribute__((aligned(16))) int sTIDX[8];
  __shared__ __attribute__((aligned(16))) uint8_t sXS[ROLE_SWAP ? 16 : NWAVE * XS_BYTES];
  const Lds L = {sA0, sA1, sORD, sIDX, sOFL, sOFH, sDER, sPT, sTIDX, sXS};

  Ctx c;
  c.w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
  c.B = (int)blockIdx.x;
  c.G = (int)gridDim.x;
  if (tile_valid(c.B, p.ntiles)) {
    c.i = -4;
    c.Tm1 = c.T0 = c.T1 = c.T2 = c.T3 = INVALID_TILE;
    c.end_seen = 0;
    c.cl_T5 = tile_valid(static_tile(c.B, 1, c.G), p.ntiles) ? static_tile(c.B, 1, c.G) : INVALID_TILE;
    c.cur_half = prologue_half(p.is_cat);
    c.st_valid = 0;
    c.st_frow = c.st_fu = c.st_fn = 0;
    c.st_su = -1;
    c.st_flags = 0;
    bf16x8 Wr[2][16];
    load_w(Wr, p.wp, c.cur_half, c.w, lane_id());
    __builtin_amdgcn_s_waitcnt(WAIT_VM0);  // nothing outstanding enters the loop (compiler waitcnt analysis)
    f32x4 ind[4] = {f32x4{}, f32x4{}, f32x4{}, f32x4{}};
    X2H_SB();
    for (;;) {
      if (!iteration<ROLE_SWAP, EMIT, 0>(p, c, L, Wr, ind)) break;
      if (!iteration<ROLE_SWAP, EMIT, 1>(p, c, L, Wr, ind)) break;
    }
  }
  // Self-resetting counters [D6]: the last WG to finish resets both, so no memset launch is needed.
  if (threadIdx.x == 0) {
    const int old = __hip_atomic_fetch_add(p.counters + 1, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (old == c.G - 1) {
      __hip_atomic_store(p.counters, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(p.counters + 1, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// W prepack [D2]: one thread per 16-B fragment element, gid = wp_index(half, w, cb, kb, lane).
__global__ __launch_bounds__(256) void x2h_wpack(const uint16_t* __restrict__ w1, int sw, uint4* __restrict__ wp,
                                                 int nhalf) {
  const int gid = (int)(blockIdx.x * 256 + threadIdx.x);
  if (gid >= nhalf * NWAVE * 2 * 16 * 64) return;
  const int lane = gid & 63, kb = (gid >> 6) & 15, cb = (gid >> 10) & 1, w = (gid >> 11) & 7, half = gid >> 14;
  const int col = wp_col(half, w, cb, lane);
  uint32_t v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t lo = w1[(long long)wp_k(kb, lane, 2 * i) * sw + col];
    const uint32_t hi = w1[(long long)wp_k(kb, lane, 2 * i + 1) * sw + col];
    v[i] = lo | (hi << 16);
  }
  wp[gid] = uint4{v[0], v[1], v[2], v[3]};
}

}  // namespace x2h

// =====================================================================================================================
// 5. C ABI (ctypes; pattern of ops/triton/wgrad_tm_ck_w3.cu). Return 0 = launched; > 0 = hipError_t of a rejected
// argument or a failed launch; < 0 = launched, and -rc is a stale error an earlier call had left on this thread.
// =====================================================================================================================
extern "C" int x2h_abi_version(void) { return X2H_ABI_VERSION; }
extern "C" int x2h_lds_bytes(int role_swap) { return x2h::lds_bytes(role_swap ? 1 : 0); }
extern "C" int x2h_wp_elems(int nhalf) { return nhalf * x2h::NWAVE * 2 * 16 * 64 * 8; }  // bf16 elements of Wp
extern "C" int x2h_drain_last_error(void) { return (int)hipGetLastError(); }

static inline bool x2h_misaligned16(const void* q) { return ((uintptr_t)q & 15u) != 0; }

extern "C" int x2h_wpack_launch(const void* w1, int sw, void* wp, int nhalf, void* stream) {
  if (w1 == nullptr || wp == nullptr || (nhalf != 1 && nhalf != 2) || sw != nhalf * x2h::NOUT)
    return (int)hipErrorInvalidValue;
  if (x2h_misaligned16(w1) || x2h_misaligned16(wp)) return (int)hipErrorInvalidValue;
  const hipError_t stale = hipGetLastError();
  const int total = nhalf * x2h::NWAVE * 2 * 16 * 64;
  hipLaunchKernelGGL(x2h::x2h_wpack, dim3((total + 255) / 256), dim3(256), 0, (hipStream_t)stream,
                     (const uint16_t*)w1, sw, (uint4*)wp, nhalf);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) return (int)e;
  return stale == hipSuccess ? 0 : -(int)stale;
}

// The grid x2h_pass_launch uses for (npos, grid, max_tiles); 0 = x2h_pass_launch would reject these arguments
// (including max_tiles > 0 in a build without -DX2H_BOUNDED_RESIDENCY=1).
extern "C" int x2h_pass_grid(int npos, int grid, int max_tiles) {
  if (npos <= 0 || grid <= 0 || max_tiles < 0 || (max_tiles > 0 && !X2H_BOUNDED_RESIDENCY)) return 0;
  return x2h::launch_grid(x2h::ntiles_of(npos), grid, x2h::launch_max_seq(max_tiles));
}

// dh: [M, sdh] bf16 (sdh = row stride in elements, multiple of 8, >= 256); wp: x2h_wpack output for this member;
// dx / sw_dx: EMIT only (DX [M, sw_dx] bf16); zrow: >= 512 zero bytes; counters: 2 int32, zero before the first
// launch on a stream (self-reset afterwards); grid: number of workgroups (persistent; capped at the tile count);
// max_tiles: 0 = persistent (each WG claims until the tiles run out), K > 0 = bounded residency (each WG computes at
// most max(K, 6) tiles and exits; the grid is raised to cdiv(ntiles, max(K, 6)); x2h_pass_grid gives the result;
// only in a build with -DX2H_BOUNDED_RESIDENCY=1, otherwise rejected).
extern "C" int x2h_pass_launch(int role_swap, int emit, const void* dh, long long sdh, const void* wp,
                               const void* idx, const void* order, const void* off, void* acc, void* frag_u,
                               void* frag_n, void* frag_v, void* dx, long long sw_dx, const void* zrow,
                               void* counters, int npos, int frag_base, int is_cat, int grid, int max_tiles,
                               void* stream) {
  if (npos <= 0 || grid <= 0 || frag_base < 0 || (is_cat != 0 && is_cat != 1) || max_tiles < 0 ||
      (max_tiles > 0 && !X2H_BOUNDED_RESIDENCY))
    return (int)hipErrorInvalidValue;
  if (sdh < x2h::KD || (sdh & 7) != 0 || sdh * 2 >= 0x80000000LL) return (int)hipErrorInvalidValue;
  const void* need[] = {dh, wp, idx, order, off, acc, frag_u, frag_n, frag_v, zrow, counters};
  for (const void* q : need)
    if (q == nullptr || x2h_misaligned16(q)) return (int)hipErrorInvalidValue;
  if (emit && (dx == nullptr || x2h_misaligned16(dx) || sw_dx != (is_cat ? 2 : 1) * x2h::NOUT))
    return (int)hipErrorInvalidValue;
  x2h::Params p;
  p.dh = (const uint8_t*)dh;
  p.wp = (const uint4*)wp;
  p.idx = (const long long*)idx;
  p.order = (const long long*)order;
  p.off = (const long long*)off;
  p.acc = (float*)acc;
  p.frag_u = (int*)frag_u;
  p.frag_n = (int*)frag_n;
  p.frag_v = (float*)frag_v;
  p.dx = (uint8_t*)dx;
  p.zrow = (const uint8_t*)zrow;
  p.counters = (int*)counters;
  p.dx_row_bytes = emit ? (uint32_t)(sw_dx * 2) : 0u;
  p.dh_row_bytes = (uint32_t)(sdh * 2);
  p.npos = npos;
  p.ntiles = x2h::ntiles_of(npos);
  p.frag_base = frag_base;
  p.is_cat = is_cat;
  p.max_seq = x2h::launch_max_seq(max_tiles);
  const int g = x2h::launch_grid(p.ntiles, grid, p.max_seq);
  const hipError_t stale = hipGetLastError();
  if (role_swap) {
    if (emit) hipLaunchKernelGGL((x2h::x2h_pass<1, 1>), dim3(g), dim3(x2h::NTHREADS), 0, (hipStream_t)stream, p);
    else hipLaunchKernelGGL((x2h::x2h_pass<1, 0>), dim3(g), dim3(x2h::NTHREADS), 0, (hipStream_t)stream, p);
  } else {
    if (emit) hipLaunchKernelGGL((x2h::x2h_pass<0, 1>), dim3(g), dim3(x2h::NTHREADS), 0, (hipStream_t)stream, p);
    else hipLaunchKernelGGL((x2h::x2h_pass<0, 0>), dim3(g), dim3(x2h::NTHREADS), 0, (hipStream_t)stream, p);
  }
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) return (int)e;
  return stale == hipSuccess ? 0 : -(int)stale;
}

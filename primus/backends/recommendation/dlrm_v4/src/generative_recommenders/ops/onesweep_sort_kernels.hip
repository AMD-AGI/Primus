// onesweep_sort_kernels.hip -- stable LSD radix sort pairs for DLRMv4's bounded integer sort keys
// (ops/onesweep_sort.py; yambda_5b.gin item (M4)).
//
// Device code: rocPRIM 4.x's onesweep kernels (rocprim::detail::onesweep_histograms / onesweep_scan_histograms /
// onesweep_iteration -- the code torch.sort runs on gfx950). Host orchestration (this file) differs from
// torch.sort -> hipcub::DeviceRadixSort::SortPairs:
//   * bits [0, end_bit) = the caller's static key bound, not the key width;
//   * digit width per caller (int16 posenc ts keys: 6 bits, 2 passes for 12-bit keys; int32 EC joint keys: 9 bits,
//     2 passes for 18-bit keys) -- MI350X lab sweep, iter-72 rccl probes/iter72/sorts;
//   * the values are a counting iterator (no iota kernel, no index-tensor read in the first pass);
//   * ONE fill per sort (the digit histogram); every pass's lookback states and ordered-block-id counter are zeroed
//     by the histogram kernel (rocPRIM's host code issues 1 + 2 fills per pass on gfx942/gfx950);
//   * EC joint: keys read straight from the two members' int64 id tensors (no int32 key build), values = member-
//     relative positions (no "joint[ta:] - ta" kernel), member b's order 16-byte aligned in the shared output, and
//     both members' segment offsets from the sorted keys in one streaming kernel plus one gap-fill kernel for
//     the long runs of absent keys (no binary searches).
// Stability: an LSD radix sort is stable and the stable sorting permutation of a key sequence is unique, so with
// every key in [0, 2^end_bit) the permutation equals torch.sort(stable=True)'s, which torch runs for every 1-D CUDA
// sort above 4096 elements. No torch headers here (they pull thrust/complex into HIP mode); the torch binding is
// onesweep_sort_bind.cpp.
#include <cstring>  // host memset for rocprim's texture_cache_iterator
#include <hip/hip_runtime.h>

#include <rocprim/rocprim.hpp>
#include <rocprim/device/detail/device_radix_sort.hpp>
#include <rocprim/device/detail/ordered_block_id.hpp>

#include <algorithm>
#include <cstdint>

namespace rd = rocprim::detail;
using Off = unsigned int;
using LState = rd::onesweep_lookback_state;
using Bid = rd::block_id_wrapper<unsigned int, true>;

// ------------------------------------------------------------------------------------------ kernels
// Histogram of every digit place + zeroing of all passes' lookback states and block-id counters.
template <unsigned BS, unsigned IPT, unsigned RB, class KIt>
__global__ __launch_bounds__(BS) void dlrm_onesweep_hist(KIt keys, Off* counts, Off size, Off full_blocks,
                                                         unsigned begin_bit, unsigned end_bit, uint32_t* zero_words,
                                                         Off zero_n) {
  const Off stride = gridDim.x * BS;
  for (Off i = blockIdx.x * BS + threadIdx.x; i < zero_n; i += stride) zero_words[i] = 0u;
  rd::onesweep_histograms<BS, IPT, RB, false>(keys, counts, size, full_blocks, rocprim::identity_decomposer{},
                                              begin_bit, end_bit);
}

template <unsigned BS, unsigned RB>
__global__ __launch_bounds__(BS) void dlrm_onesweep_scan(Off* counts) {
#if ROCPRIM_VERSION >= 400800  // TargetWaveSize template parameter, as in dlrm_onesweep_pass below
  rd::onesweep_scan_histograms<BS, RB, ::rocprim::arch::wavefront::get_target()>(counts);
#else
  rd::onesweep_scan_histograms<BS, RB>(counts);
#endif
}

template <unsigned BS, unsigned IPT, unsigned RB, class KIn, class KOut, class VIn, class VOut>
__global__ __launch_bounds__(BS) void dlrm_onesweep_pass(KIn kin, KOut kout, VIn vin, VOut vout, Off size, Off* gin,
                                                         Off* gout, LState* states, unsigned bit, unsigned cur_bits,
                                                         Off full_blocks, Bid bid) {
#if ROCPRIM_VERSION >= 400800
  // rocPRIM 4.8 (ROCm 10.2) adds the TargetWaveSize template parameter; rocPRIM passes its config's wavefront,
  // which for this device-side call is the compile target's (size64 on gfx950). 4.3-4.7 are untested.
  rd::onesweep_iteration<BS, IPT, RB, false, rocprim::block_radix_rank_algorithm::match,
                         ::rocprim::arch::wavefront::get_target()>(
      kin, kout, vin, vout, size, gin, gout, states, rocprim::identity_decomposer{}, bit, cur_bits, full_blocks, bid);
#else
  rd::onesweep_iteration<BS, IPT, RB, false, rocprim::block_radix_rank_algorithm::match>(
      kin, kout, vin, vout, size, gin, gout, states, rocprim::identity_decomposer{}, bit, cur_bits, full_blocks, bid);
#endif
}

// joint offsets J[k] = #{i : sk[i] < k}, k in [0, 2U]; off_a[k] = J[k] (k <= U), off_b[k - U] = J[k] - ta (k >= U).
// Position i (0 <= i <= n) owns the keys (sk[i-1], sk[i]] (sk[-1] = -1, sk[n] = 2U), all with J = i: every key is
// written exactly once. Real EC members use disjoint id sub-ranges, so some ranges span ~10^5 absent keys. A range of
// at most kShortRange keys is written by its thread; longer ones are appended, in chunks of at most kGapChunk keys,
// to a list (counter zeroed by the sort's histogram kernel) that dlrm_ec_offsets_fill_gaps writes one block per chunk.
constexpr int64_t kShortRange = 32;
constexpr int64_t kGapChunk = 4096;

__device__ __forceinline__ void dlrm_ec_put(int64_t k, int64_t i, int64_t* __restrict__ off_a,
                                            int64_t* __restrict__ off_b, int64_t U, int64_t ta) {
  if (k <= U) off_a[k] = i;
  if (k >= U) off_b[k - U] = i - ta;
}

__global__ __launch_bounds__(256) void dlrm_ec_offsets_from_sorted(const int32_t* __restrict__ sk, Off n,
                                                                   int64_t* __restrict__ off_a,
                                                                   int64_t* __restrict__ off_b, int32_t U32,
                                                                   int64_t ta, unsigned int* __restrict__ gap_count,
                                                                   int64_t* __restrict__ gaps) {
  const int64_t U = U32;
  const int64_t i = (int64_t)blockIdx.x * 256 + threadIdx.x;
  if (i > (int64_t)n) return;
  const int64_t lo = ((i == 0) ? -1 : (int64_t)sk[i - 1]) + 1;
  const int64_t hi = (i == (int64_t)n) ? 2 * U : (int64_t)sk[i];
  const int64_t len = hi - lo + 1;
  if (len <= kShortRange) {
    for (int64_t k = lo; k <= hi; ++k) dlrm_ec_put(k, i, off_a, off_b, U, ta);
    return;
  }
  const int64_t chunks = (len + kGapChunk - 1) / kGapChunk;
  const int64_t base = (int64_t)atomicAdd(gap_count, (unsigned int)chunks);
  for (int64_t c = 0; c < chunks; ++c) {
    int64_t* e = gaps + 3 * (base + c);
    e[0] = lo + c * kGapChunk;
    e[1] = std::min(hi, lo + (c + 1) * kGapChunk - 1);
    e[2] = i;
  }
}

__global__ __launch_bounds__(256) void dlrm_ec_offsets_fill_gaps(const unsigned int* __restrict__ gap_count,
                                                                 const int64_t* __restrict__ gaps,
                                                                 int64_t* __restrict__ off_a,
                                                                 int64_t* __restrict__ off_b, int32_t U32,
                                                                 int64_t ta) {
  const int64_t U = U32;
  const unsigned int cnt = *gap_count;
  for (unsigned int e = blockIdx.x; e < cnt; e += gridDim.x) {
    const int64_t lo = gaps[3 * (int64_t)e], hi = gaps[3 * (int64_t)e + 1], v = gaps[3 * (int64_t)e + 2];
    for (int64_t k = lo + threadIdx.x; k <= hi; k += 256) dlrm_ec_put(k, v, off_a, off_b, U, ta);
  }
}

// list capacity (chunks) that every key layout fits: long ranges are disjoint and longer than kShortRange
int64_t dlrm_onesweep_gap_capacity(int64_t U) {
  const int64_t keys = 2 * U + 1;
  return keys / (kShortRange + 1) + keys / kGapChunk + 2;
}

// ------------------------------------------------------------------------------------------ iterators
struct ECKeyF {  // joint key of position i: member a's id, or member b's id + U (int32)
  const int64_t* a;
  const int64_t* b;
  int64_t sa, sb;
  Off ta;
  int32_t U;
  __host__ __device__ int32_t operator()(Off i) const {
    return i < ta ? (int32_t)a[(int64_t)i * sa] : (int32_t)(b[(int64_t)(i - ta) * sb] + U);
  }
};
struct ECValF {  // member-relative position of joint position i
  Off ta;
  __host__ __device__ int64_t operator()(Off i) const { return i < ta ? (int64_t)i : (int64_t)(i - ta); }
};
using CountIt = rocprim::counting_iterator<Off>;
using ECKeyIt = rocprim::transform_iterator<CountIt, ECKeyF, int32_t>;
using ECValIt = rocprim::transform_iterator<CountIt, ECValF, int64_t>;
using IotaIt = rocprim::counting_iterator<int64_t>;

// final-pass value output: joint rank r >= ta (member b) lands `shift` slots later (16-byte alignment of b)
struct ShiftOut {
  int64_t* p;
  Off ta;
  Off shift;
  __host__ __device__ int64_t& operator[](Off r) const { return p[(size_t)r + (r >= ta ? shift : 0u)]; }
};

// ------------------------------------------------------------------------------------------ host driver
static inline Off cdiv(Off a, Off b) { return (a + b - 1) / b; }

struct Plan {
  Off places, radix, hist_blocks, hist_full, sort_blocks, sort_full;
  size_t counts_words, gtmp_words, state_words, bid_words;
};

template <unsigned HBS, unsigned HIPT, unsigned SBS, unsigned SIPT, unsigned RB>
static Plan make_plan(Off n, unsigned end_bit) {
  Plan p;
  p.radix = 1u << RB;
  p.places = cdiv(end_bit, RB);
  const Off hipb = HBS * HIPT, sipb = SBS * SIPT;
  p.hist_blocks = cdiv(n, hipb);
  p.hist_full = (n % hipb == 0) ? p.hist_blocks : p.hist_blocks - 1;
  p.sort_blocks = cdiv(n, sipb);
  p.sort_full = (n % sipb == 0) ? p.sort_blocks : p.sort_blocks - 1;
  p.counts_words = (size_t)p.places * p.radix;
  p.gtmp_words = p.radix;
  p.state_words = (size_t)p.places * p.radix * p.sort_blocks;
  p.bid_words = 64;  // one ordered-block-id counter per pass (places <= 64)
  return p;
}

template <unsigned HBS, unsigned HIPT, unsigned SBS, unsigned SIPT, unsigned RB>
static int64_t workspace_bytes_t(int64_t n, int64_t end_bit) {
  Plan p = make_plan<HBS, HIPT, SBS, SIPT, RB>((Off)n, (unsigned)end_bit);
  return (int64_t)((p.counts_words + p.gtmp_words + p.state_words + p.bid_words) * 4 + 256);
}

// Sort n keys (KIt) carrying values (VIt) over bits [0, end_bit). Pass p writes to (P-1-p) even ? out : tmp, so the
// last pass lands in kout / vout_final. Requires 2 <= P <= 63. Returns 0 or a hipError_t.
template <unsigned HBS, unsigned HIPT, unsigned SBS, unsigned SIPT, unsigned RB, class KeyT, class KIt, class VIt,
          class VOutFinal>
static int run_sort(KIt kin, VIt vin, Off n, unsigned end_bit, KeyT* kout, VOutFinal vout_final, int64_t* vout_plain,
                    KeyT* ktmp, int64_t* vtmp, uint32_t* ws, hipStream_t stream) {
  Plan p = make_plan<HBS, HIPT, SBS, SIPT, RB>(n, end_bit);
  if (p.places < 2 || p.places > 63) return (int)hipErrorInvalidValue;
  Off* counts = reinterpret_cast<Off*>(ws);
  Off* gtmp = counts + p.counts_words;
  uint32_t* zero_base = reinterpret_cast<uint32_t*>(gtmp + p.gtmp_words);  // states then bids, zeroed in hist
  LState* states = reinterpret_cast<LState*>(zero_base);
  unsigned int* bids = reinterpret_cast<unsigned int*>(zero_base + p.state_words);
  const size_t zero_n = p.state_words + p.bid_words;
  hipError_t e = hipMemsetAsync(counts, 0, p.counts_words * sizeof(Off), stream);
  if (e != hipSuccess) return (int)e;
  hipLaunchKernelGGL((dlrm_onesweep_hist<HBS, HIPT, RB, KIt>), dim3(p.hist_blocks), dim3(HBS), 0, stream, kin, counts,
                     n, p.hist_full, 0u, end_bit, zero_base, (Off)zero_n);
  hipLaunchKernelGGL((dlrm_onesweep_scan<HBS, RB>), dim3(p.places), dim3(HBS), 0, stream, counts);
  const size_t state_per_pass = (size_t)p.radix * p.sort_blocks;
  {  // pass 0: iterators in
    const bool to_out = ((p.places - 1) % 2) == 0;
    KeyT* ko = to_out ? kout : ktmp;
    int64_t* vo = to_out ? vout_plain : vtmp;
    Bid bid = Bid::create(bids + 0);
    hipLaunchKernelGGL((dlrm_onesweep_pass<SBS, SIPT, RB, KIt, KeyT*, VIt, int64_t*>), dim3(p.sort_blocks), dim3(SBS),
                       0, stream, kin, ko, vin, vo, n, counts, gtmp, states, 0u, std::min((unsigned)RB, end_bit),
                       p.sort_full, bid);
  }
  for (Off place = 1; place < p.places; ++place) {
    const unsigned bit = place * RB;
    const unsigned cur = std::min((unsigned)RB, end_bit - bit);
    const bool src_out = ((p.places - 1 - (place - 1)) % 2) == 0;
    const KeyT* ki = src_out ? kout : ktmp;
    const int64_t* vi = src_out ? vout_plain : vtmp;
    Bid bid = Bid::create(bids + place);
    LState* st = states + state_per_pass * place;
    if (place == p.places - 1) {
      hipLaunchKernelGGL((dlrm_onesweep_pass<SBS, SIPT, RB, const KeyT*, KeyT*, const int64_t*, VOutFinal>),
                         dim3(p.sort_blocks), dim3(SBS), 0, stream, ki, kout, vi, vout_final, n,
                         counts + place * p.radix, gtmp, st, bit, cur, p.sort_full, bid);
    } else {
      const bool to_out = ((p.places - 1 - place) % 2) == 0;
      KeyT* ko = to_out ? kout : ktmp;
      int64_t* vo = to_out ? vout_plain : vtmp;
      hipLaunchKernelGGL((dlrm_onesweep_pass<SBS, SIPT, RB, const KeyT*, KeyT*, const int64_t*, int64_t*>),
                         dim3(p.sort_blocks), dim3(SBS), 0, stream, ki, ko, vi, vo, n, counts + place * p.radix, gtmp,
                         st, bit, cur, p.sort_full, bid);
    }
  }
  return (int)hipGetLastError();
}

// ------------------------------------------------------------------------------------------ C++ API
// (HBS, HIPT, SBS, SIPT, RB). int16 ts keys: rocPRIM's gfx950 <short, 8-byte value> hist/sort configs, RB 6.
// int32 EC joint keys: rocPRIM's gfx950 <int, 8-byte value> configs, RB 9.
#define TS16_CFG 1024, 32, 1024, 6, 6
#define EC32_CFG 1024, 8, 1024, 8, 9

int64_t dlrm_onesweep_radix_bits16() { return 6; }
int64_t dlrm_onesweep_radix_bits32() { return 9; }
int64_t dlrm_onesweep_ws16(int64_t n, int64_t end_bit) { return workspace_bytes_t<TS16_CFG>(n, end_bit); }
int64_t dlrm_onesweep_ws32(int64_t n, int64_t end_bit) { return workspace_bytes_t<EC32_CFG>(n, end_bit); }

int dlrm_onesweep_sort16(const int16_t* keys, int64_t n, int64_t end_bit, int16_t* kout, int64_t* vout, int16_t* ktmp,
                         int64_t* vtmp, void* ws, hipStream_t s) {
  return run_sort<TS16_CFG, int16_t>(keys, IotaIt(0), (Off)n, (unsigned)end_bit, kout, vout, vout, ktmp, vtmp,
                                     reinterpret_cast<uint32_t*>(ws), s);
}

int dlrm_onesweep_ec_joint(const int64_t* a, int64_t ta, int64_t sa, const int64_t* b, int64_t tb, int64_t sb,
                           int64_t U, int64_t end_bit, int32_t* kout, int64_t* vout, int32_t* ktmp, int64_t* vtmp,
                           void* ws, int64_t* off_a, int64_t* off_b, int64_t* gaps, hipStream_t s) {
  const int64_t n = ta + tb;
  const Off shift = (Off)(ta & 1);
  ECKeyIt kit(CountIt(0), ECKeyF{a, b, sa, sb, (Off)ta, (int32_t)U});
  ECValIt vit(CountIt(0), ECValF{(Off)ta});
  ShiftOut so{vout, (Off)ta, shift};
  int rc = run_sort<EC32_CFG, int32_t>(kit, vit, (Off)n, (unsigned)end_bit, kout, so, vout, ktmp, vtmp,
                                       reinterpret_cast<uint32_t*>(ws), s);
  if (rc != 0) return rc;
  // gap-list counter: the last ordered-block-id word, unused by the <= 63 passes and zeroed by the histogram kernel
  Plan p = make_plan<EC32_CFG>((Off)n, (unsigned)end_bit);
  unsigned int* gap_count = reinterpret_cast<unsigned int*>(ws) + p.counts_words + p.gtmp_words + p.state_words + 63;
  hipLaunchKernelGGL(dlrm_ec_offsets_from_sorted, dim3((unsigned)((n + 1 + 255) / 256)), dim3(256), 0, s, kout, (Off)n,
                     off_a, off_b, (int32_t)U, ta, gap_count, gaps);
  hipLaunchKernelGGL(dlrm_ec_offsets_fill_gaps, dim3(1024), dim3(256), 0, s, gap_count, gaps, off_a, off_b,
                     (int32_t)U, ta);
  return (int)hipGetLastError();
}

#ifdef DLRM_ONESWEEP_EC_RB10
// yambda_5b.gin item (102) EC_JOINT_ONESWEEP_RB10: the same EC joint sort with 10-bit digits, compiled only into
// the separate "_rb10" build of this extension (ops/onesweep_sort.py; the default build never sees this block).
// The caller uses it only for 19- and 20-bit keys (2U - 1 in [2^18, 2^20), i.e. 2^17 < U <= 2^19): two passes
// (10 + 9 or 10 + 10 bits) instead of three 9-bit ones (the third sorting 1-2 bits). Same device code (rocPRIM
// onesweep, radix 1024 <= block size 1024), same key/value iterators, same offsets kernels, so the same stable
// permutation and the same integer offsets (bitwise): only the digit split of the key bits changes.
#define EC32W_CFG 1024, 8, 1024, 8, 10

int64_t dlrm_onesweep_radix_bits32w() { return 10; }
int64_t dlrm_onesweep_ws32w(int64_t n, int64_t end_bit) { return workspace_bytes_t<EC32W_CFG>(n, end_bit); }

int dlrm_onesweep_ec_joint_w(const int64_t* a, int64_t ta, int64_t sa, const int64_t* b, int64_t tb, int64_t sb,
                             int64_t U, int64_t end_bit, int32_t* kout, int64_t* vout, int32_t* ktmp, int64_t* vtmp,
                             void* ws, int64_t* off_a, int64_t* off_b, int64_t* gaps, hipStream_t s) {
  const int64_t n = ta + tb;
  const Off shift = (Off)(ta & 1);
  ECKeyIt kit(CountIt(0), ECKeyF{a, b, sa, sb, (Off)ta, (int32_t)U});
  ECValIt vit(CountIt(0), ECValF{(Off)ta});
  ShiftOut so{vout, (Off)ta, shift};
  int rc = run_sort<EC32W_CFG, int32_t>(kit, vit, (Off)n, (unsigned)end_bit, kout, so, vout, ktmp, vtmp,
                                        reinterpret_cast<uint32_t*>(ws), s);
  if (rc != 0) return rc;
  // gap-list counter: the last ordered-block-id word of THIS config's workspace layout (zeroed by its histogram)
  Plan p = make_plan<EC32W_CFG>((Off)n, (unsigned)end_bit);
  unsigned int* gap_count = reinterpret_cast<unsigned int*>(ws) + p.counts_words + p.gtmp_words + p.state_words + 63;
  hipLaunchKernelGGL(dlrm_ec_offsets_from_sorted, dim3((unsigned)((n + 1 + 255) / 256)), dim3(256), 0, s, kout, (Off)n,
                     off_a, off_b, (int32_t)U, ta, gap_count, gaps);
  hipLaunchKernelGGL(dlrm_ec_offsets_fill_gaps, dim3(1024), dim3(256), 0, s, gap_count, gaps, off_a, off_b,
                     (int32_t)U, ta);
  return (int)hipGetLastError();
}
#endif  // DLRM_ONESWEEP_EC_RB10

// ------------------------------------------------------------------------------------------ int64 keys (zero-fills)
// ops/preproc_dedup.py build_unique: the stable sort of the int64 triple keys b + U * (a + U * item) < U^3 (valid ids
// in [0, U)), ~2.4M keys per step at U ~ 54k (48-bit keys). torch.sort runs rocPRIM's onesweep over all 64 bits:
// 8 passes plus 17 hipMemsetAsync fills (1 + 2 per pass; torch 2.16 / ROCm 10.2: rocclr fillBufferUnAligned blits on
// the compute stream). Here: rocPRIM's own gfx950 <int64_t, int64_t> config (hist 1024 x 8, sort 1024 x 8, 8-bit
// digits; the kernels torch.sort runs), bits [0, end_bit) only (6 passes at 48 bits), one fill per sort, and the
// values from a counting iterator (no iota kernel). Same stable permutation, so the same (keys, order) bits.
// Appended after every earlier line so the existing kernels' and checks' source lines are unchanged.
#define K64_CFG 1024, 8, 1024, 8, 8

int64_t dlrm_onesweep_radix_bits64() { return 8; }
int64_t dlrm_onesweep_ws64(int64_t n, int64_t end_bit) { return workspace_bytes_t<K64_CFG>(n, end_bit); }

int dlrm_onesweep_sort64(const int64_t* keys, int64_t n, int64_t end_bit, int64_t* kout, int64_t* vout, int64_t* ktmp,
                         int64_t* vtmp, void* ws, hipStream_t s) {
  return run_sort<K64_CFG, int64_t>(keys, IotaIt(0), (Off)n, (unsigned)end_bit, kout, vout, vout, ktmp, vtmp,
                                    reinterpret_cast<uint32_t*>(ws), s);
}

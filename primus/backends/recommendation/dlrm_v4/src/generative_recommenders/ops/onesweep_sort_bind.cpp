// onesweep_sort_bind.cpp -- torch binding of onesweep_sort_kernels.hip (host C++ only; ops/onesweep_sort.py).
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>

int64_t dlrm_onesweep_radix_bits16();
int64_t dlrm_onesweep_radix_bits32();
int64_t dlrm_onesweep_ws16(int64_t n, int64_t end_bit);
int64_t dlrm_onesweep_ws32(int64_t n, int64_t end_bit);
int dlrm_onesweep_sort16(const int16_t* keys, int64_t n, int64_t end_bit, int16_t* kout, int64_t* vout, int16_t* ktmp,
                         int64_t* vtmp, void* ws, hipStream_t s);
int dlrm_onesweep_ec_joint(const int64_t* a, int64_t ta, int64_t sa, const int64_t* b, int64_t tb, int64_t sb,
                           int64_t U, int64_t end_bit, int32_t* kout, int64_t* vout, int32_t* ktmp, int64_t* vtmp,
                           void* ws, int64_t* off_a, int64_t* off_b, int64_t* gaps, hipStream_t s);
int64_t dlrm_onesweep_gap_capacity(int64_t U);

static hipStream_t cur_stream() { return c10::hip::getCurrentHIPStream().stream(); }

static void chk(bool ok, const char* what) { TORCH_CHECK(ok, "onesweep_sort: ", what); }
static void chk_rc(int rc) { TORCH_CHECK(rc == 0, "onesweep_sort: HIP error ", rc); }

static void vec(const torch::Tensor& t, c10::ScalarType dt, int64_t n, const char* what) {
  chk(t.is_cuda() && t.scalar_type() == dt && t.is_contiguous() && t.dim() == 1 && t.numel() == n, what);
}

int64_t radix_bits16() { return dlrm_onesweep_radix_bits16(); }
int64_t radix_bits32() { return dlrm_onesweep_radix_bits32(); }
int64_t ws16(int64_t n, int64_t end_bit) { return dlrm_onesweep_ws16(n, end_bit); }
int64_t ws32(int64_t n, int64_t end_bit) { return dlrm_onesweep_ws32(n, end_bit); }
int64_t gap_capacity(int64_t U) { return dlrm_onesweep_gap_capacity(U); }

// sorted keys + int64 permutation (== torch.sort(keys) for keys in [0, 2^end_bit) and n > 4096)
void sort16(torch::Tensor keys, int64_t end_bit, torch::Tensor kout, torch::Tensor vout, torch::Tensor ktmp,
            torch::Tensor vtmp, torch::Tensor ws) {
  const int64_t n = keys.numel();
  vec(keys, torch::kInt16, n, "keys: int16 1-D contiguous CUDA");
  // one onesweep batch: rocPRIM's lookback state keeps 30 bits of count (its host code batches < 2^30 items)
  chk(n > 0 && n < (int64_t(1) << 30) && end_bit > dlrm_onesweep_radix_bits16() && end_bit <= 15, "n / end_bit");
  vec(kout, torch::kInt16, n, "kout");
  vec(ktmp, torch::kInt16, n, "ktmp");
  vec(vout, torch::kInt64, n, "vout");
  vec(vtmp, torch::kInt64, n, "vtmp");
  vec(ws, torch::kUInt8, dlrm_onesweep_ws16(n, end_bit), "workspace");
  chk(keys.device() == kout.device() && keys.device() == ws.device(), "one device");
  chk_rc(dlrm_onesweep_sort16(keys.data_ptr<int16_t>(), n, end_bit, kout.data_ptr<int16_t>(),
                              vout.data_ptr<int64_t>(), ktmp.data_ptr<int16_t>(), vtmp.data_ptr<int64_t>(),
                              ws.data_ptr(), cur_stream()));
}

// EC joint: member ids a [ta] / b [tb] (int64, any stride), valid ids in [0, U). Writes the joint sorted keys
// (kout int32 [ta+tb]), the member orders into vout (int64 [ta + (ta&1) + tb]: a's at [0, ta), b's at
// [ta + (ta&1), ...)) and both members' offsets (int64 [U+1] each); gaps = int64 [3 * gap_capacity(U)] scratch.
void ec_joint(torch::Tensor a, torch::Tensor b, int64_t U, int64_t end_bit, torch::Tensor kout, torch::Tensor vout,
              torch::Tensor ktmp, torch::Tensor vtmp, torch::Tensor ws, torch::Tensor off_a, torch::Tensor off_b,
              torch::Tensor gaps) {
  chk(a.is_cuda() && b.is_cuda() && a.scalar_type() == torch::kInt64 && b.scalar_type() == torch::kInt64 &&
          a.dim() == 1 && b.dim() == 1 && a.device() == b.device(),
      "ids: int64 1-D CUDA on one device");
  const int64_t ta = a.numel(), tb = b.numel(), n = ta + tb;
  chk(ta > 0 && tb > 0 && n < (int64_t(1) << 30) && U > 0 && end_bit > dlrm_onesweep_radix_bits32() &&
          end_bit <= 31 && 2 * U <= (int64_t(1) << end_bit),
      "sizes / U / end_bit");
  vec(kout, torch::kInt32, n, "kout");
  vec(ktmp, torch::kInt32, n, "ktmp");
  vec(vout, torch::kInt64, n + (ta & 1), "vout");
  vec(vtmp, torch::kInt64, n, "vtmp");
  vec(ws, torch::kUInt8, dlrm_onesweep_ws32(n, end_bit), "workspace");
  vec(off_a, torch::kInt64, U + 1, "off_a");
  vec(off_b, torch::kInt64, U + 1, "off_b");
  vec(gaps, torch::kInt64, 3 * dlrm_onesweep_gap_capacity(U), "gaps");
  chk(a.device() == kout.device() && a.device() == ws.device() && a.device() == off_a.device() &&
          a.device() == gaps.device(),
      "one device");
  chk_rc(dlrm_onesweep_ec_joint(a.data_ptr<int64_t>(), ta, a.stride(0), b.data_ptr<int64_t>(), tb, b.stride(0), U,
                                end_bit, kout.data_ptr<int32_t>(), vout.data_ptr<int64_t>(), ktmp.data_ptr<int32_t>(),
                                vtmp.data_ptr<int64_t>(), ws.data_ptr(), off_a.data_ptr<int64_t>(),
                                off_b.data_ptr<int64_t>(), gaps.data_ptr<int64_t>(), cur_stream()));
}

#ifdef DLRM_ONESWEEP_EC_RB10
// yambda_5b.gin item (102): only in the "_rb10" variant build (ops/onesweep_sort.py). Kept after every default-build
// line so the default build's source lines (TORCH_CHECK embeds __LINE__) are unchanged.
int64_t dlrm_onesweep_radix_bits32w();
int64_t dlrm_onesweep_ws32w(int64_t n, int64_t end_bit);
int dlrm_onesweep_ec_joint_w(const int64_t* a, int64_t ta, int64_t sa, const int64_t* b, int64_t tb, int64_t sb,
                             int64_t U, int64_t end_bit, int32_t* kout, int64_t* vout, int32_t* ktmp, int64_t* vtmp,
                             void* ws, int64_t* off_a, int64_t* off_b, int64_t* gaps, hipStream_t s);

int64_t radix_bits32w() { return dlrm_onesweep_radix_bits32w(); }
int64_t ws32w(int64_t n, int64_t end_bit) { return dlrm_onesweep_ws32w(n, end_bit); }

// item (102): ec_joint with 10-bit digits; same arguments and outputs as ec_joint, for 2-pass key widths only
// (radix_bits32w() < end_bit <= 2 * radix_bits32w(), 2U <= 2^end_bit).
void ec_joint_w(torch::Tensor a, torch::Tensor b, int64_t U, int64_t end_bit, torch::Tensor kout, torch::Tensor vout,
                torch::Tensor ktmp, torch::Tensor vtmp, torch::Tensor ws, torch::Tensor off_a, torch::Tensor off_b,
                torch::Tensor gaps) {
  chk(a.is_cuda() && b.is_cuda() && a.scalar_type() == torch::kInt64 && b.scalar_type() == torch::kInt64 &&
          a.dim() == 1 && b.dim() == 1 && a.device() == b.device(),
      "ids: int64 1-D CUDA on one device");
  const int64_t ta = a.numel(), tb = b.numel(), n = ta + tb;
  const int64_t rb = dlrm_onesweep_radix_bits32w();
  chk(ta > 0 && tb > 0 && n < (int64_t(1) << 30) && U > 0 && end_bit > rb && end_bit <= 2 * rb &&
          2 * U <= (int64_t(1) << end_bit),
      "sizes / U / end_bit (wide)");
  vec(kout, torch::kInt32, n, "kout");
  vec(ktmp, torch::kInt32, n, "ktmp");
  vec(vout, torch::kInt64, n + (ta & 1), "vout");
  vec(vtmp, torch::kInt64, n, "vtmp");
  vec(ws, torch::kUInt8, dlrm_onesweep_ws32w(n, end_bit), "workspace (wide)");
  vec(off_a, torch::kInt64, U + 1, "off_a");
  vec(off_b, torch::kInt64, U + 1, "off_b");
  vec(gaps, torch::kInt64, 3 * dlrm_onesweep_gap_capacity(U), "gaps");
  chk(a.device() == kout.device() && a.device() == ws.device() && a.device() == off_a.device() &&
          a.device() == gaps.device(),
      "one device");
  chk_rc(dlrm_onesweep_ec_joint_w(a.data_ptr<int64_t>(), ta, a.stride(0), b.data_ptr<int64_t>(), tb, b.stride(0), U,
                                  end_bit, kout.data_ptr<int32_t>(), vout.data_ptr<int64_t>(),
                                  ktmp.data_ptr<int32_t>(), vtmp.data_ptr<int64_t>(), ws.data_ptr(),
                                  off_a.data_ptr<int64_t>(), off_b.data_ptr<int64_t>(), gaps.data_ptr<int64_t>(),
                                  cur_stream()));
}
#endif

// zero-fills: int64 keys (ops/preproc_dedup.py build_unique); appended after every earlier function so their
// TORCH_CHECK source lines are unchanged.
int64_t dlrm_onesweep_radix_bits64();
int64_t dlrm_onesweep_ws64(int64_t n, int64_t end_bit);
int dlrm_onesweep_sort64(const int64_t* keys, int64_t n, int64_t end_bit, int64_t* kout, int64_t* vout, int64_t* ktmp,
                         int64_t* vtmp, void* ws, hipStream_t s);

int64_t radix_bits64() { return dlrm_onesweep_radix_bits64(); }
int64_t ws64(int64_t n, int64_t end_bit) { return dlrm_onesweep_ws64(n, end_bit); }

// sorted keys + int64 permutation (== torch.sort(keys, stable=True) for keys in [0, 2^end_bit) and n > 4096)
void sort64(torch::Tensor keys, int64_t end_bit, torch::Tensor kout, torch::Tensor vout, torch::Tensor ktmp,
            torch::Tensor vtmp, torch::Tensor ws) {
  const int64_t n = keys.numel();
  vec(keys, torch::kInt64, n, "keys: int64 1-D contiguous CUDA");
  chk(n > 0 && n < (int64_t(1) << 30) && end_bit > dlrm_onesweep_radix_bits64() && end_bit <= 62, "n / end_bit (64)");
  vec(kout, torch::kInt64, n, "kout");
  vec(ktmp, torch::kInt64, n, "ktmp");
  vec(vout, torch::kInt64, n, "vout");
  vec(vtmp, torch::kInt64, n, "vtmp");
  vec(ws, torch::kUInt8, dlrm_onesweep_ws64(n, end_bit), "workspace (64)");
  chk(keys.device() == kout.device() && keys.device() == ws.device(), "one device");
  chk_rc(dlrm_onesweep_sort64(keys.data_ptr<int64_t>(), n, end_bit, kout.data_ptr<int64_t>(),
                              vout.data_ptr<int64_t>(), ktmp.data_ptr<int64_t>(), vtmp.data_ptr<int64_t>(),
                              ws.data_ptr(), cur_stream()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("radix_bits16", &radix_bits16);
  m.def("radix_bits32", &radix_bits32);
  m.def("ws16", &ws16);
  m.def("ws32", &ws32);
  m.def("gap_capacity", &gap_capacity);
  m.def("sort16", &sort16);
  m.def("ec_joint", &ec_joint);
  m.def("radix_bits64", &radix_bits64);
  m.def("ws64", &ws64);
  m.def("sort64", &sort64);
#ifdef DLRM_ONESWEEP_EC_RB10
  m.def("radix_bits32w", &radix_bits32w);
  m.def("ws32w", &ws32w);
  m.def("ec_joint_w", &ec_joint_w);
#endif
}

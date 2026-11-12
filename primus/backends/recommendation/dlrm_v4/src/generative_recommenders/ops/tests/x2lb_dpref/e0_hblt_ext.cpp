// e0_hblt_ext.cpp -- E0 lab kit (actckpt iteration 69): a HOST-ONLY shim over the hipBLASLt ext C++ API.
//
// No device code and no new kernel: every GEMM it runs is a hipBLASLt library solution. It exists because a few ext
// entry points are C++ (std::string / std::vector / classes) and cannot be called safely through ctypes:
//   getIndexFromAlgo, getKernelNameFromAlgo, getSolutionNameFromAlgo   (record each algo's index / name)
//   getAllAlgos + matmulIsAlgoSupported                                  (the full supported solution set)
//   getAlgosFromIndex                                                    (rebuild an algo from its index)
//   Gemm(handle, matmulDesc, ...) + GemmTuning{splitK, wgm}              (split-K / workgroup-mapping variants)
// The descriptors (hipblasLtMatmulDesc_t / hipblasLtMatrixLayout_t) are created by the ctypes path in e0_hblt.py, so a
// tuned run sees exactly the problem the C-API run sees (types, ops, leading dimensions, scale pointers).
// Built at run time in the lab container (e0_hblt.py build_ext): host compiler, -D__HIP_PLATFORM_AMD__, linked against
// the libhipblaslt.so.1 torch already loaded (same SONAME, so the process keeps ONE hipBLASLt copy). Every entry point
// catches C++ exceptions and returns a status; nothing here aborts the Python process on a library error.
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#define E0X_ABI 1

namespace {

void copy_str(const std::string& s, char* buf, int len) {
  if (buf == nullptr || len <= 0) {
    return;
  }
  const size_t n = s.size() < static_cast<size_t>(len - 1) ? s.size() : static_cast<size_t>(len - 1);
  std::memcpy(buf, s.data(), n);
  buf[n] = '\0';
}

hipblasLtMatmulAlgo_t load_algo(const void* p) {
  hipblasLtMatmulAlgo_t a;
  std::memcpy(&a, p, sizeof(a));
  return a;
}

// The Gemm object is placement-constructed into a buffer with 1 KiB of slack: the header this shim is compiled against
// and the library torch loaded are the same minor release, but the slack keeps a slightly larger library-side layout
// from writing past the allocation.
constexpr size_t kSlack = 1024;

struct E0Gemm {
  hipblaslt_ext::Gemm* g = nullptr;
  void* mem = nullptr;
  float alpha = 1.0f;
  float beta = 0.0f;
};

void drop(E0Gemm* h) {
  if (h == nullptr) {
    return;
  }
  if (h->g != nullptr) {
    h->g->~Gemm();
  }
  if (h->mem != nullptr) {
    ::operator delete(h->mem);
  }
  delete h;
}

}  // namespace

extern "C" {

int e0x_abi(void) { return E0X_ABI; }

// out[0] = sizeof(hipblasLtMatmulAlgo_t), out[1] = sizeof(hipblasLtMatmulHeuristicResult_t): e0_hblt.py checks them
// against its ctypes structures before using any other entry point.
int e0x_sizes(int* out) {
  out[0] = static_cast<int>(sizeof(hipblasLtMatmulAlgo_t));
  out[1] = static_cast<int>(sizeof(hipblasLtMatmulHeuristicResult_t));
  return 2;
}

int e0x_index(const void* algo) {
  try {
    hipblasLtMatmulAlgo_t a = load_algo(algo);
    return hipblaslt_ext::getIndexFromAlgo(a);
  } catch (...) {
    return -2;
  }
}

int e0x_names(void* handle, const void* algo, char* kname, int klen, char* sname, int slen) {
  try {
    hipblasLtMatmulAlgo_t a = load_algo(algo);
    copy_str(hipblaslt_ext::getKernelNameFromAlgo(static_cast<hipblasLtHandle_t>(handle), a), kname, klen);
    copy_str(hipblaslt_ext::getSolutionNameFromAlgo(static_cast<hipblasLtHandle_t>(handle), a), sname, slen);
    return 0;
  } catch (...) {
    return -1;
  }
}

// Every algorithm of this type combination (GEMM, not grouped). Copies up to max_n heuristic results into out
// (array of hipblasLtMatmulHeuristicResult_t); *n_total = the library's count. Returns the hipblasStatus_t.
int e0x_all_algos(void* handle, int opA, int opB, int tA, int tB, int tC, int tD, int compute, void* out, int max_n,
                  int* n_total) {
  try {
    std::vector<hipblasLtMatmulHeuristicResult_t> r;
    hipblasStatus_t st = hipblaslt_ext::getAllAlgos(
        static_cast<hipblasLtHandle_t>(handle), hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        static_cast<hipblasOperation_t>(opA), static_cast<hipblasOperation_t>(opB), static_cast<hipDataType>(tA),
        static_cast<hipDataType>(tB), static_cast<hipDataType>(tC), static_cast<hipDataType>(tD),
        static_cast<hipblasComputeType_t>(compute), r);
    *n_total = static_cast<int>(r.size());
    const size_t n = r.size() < static_cast<size_t>(max_n) ? r.size() : static_cast<size_t>(max_n);
    if (n > 0) {
      std::memcpy(out, r.data(), n * sizeof(hipblasLtMatmulHeuristicResult_t));
    }
    return static_cast<int>(st);
  } catch (...) {
    return -1;
  }
}

// matmulIsAlgoSupported for the C-API problem (desc + layouts); updates the algo in place (the library stores the
// problem inside it) and returns the workspace it needs.
int e0x_is_supported(void* handle, void* desc, const void* alpha, void* la, void* lb, const void* beta, void* lc,
                     void* ld, void* algo_inout, size_t* ws) {
  try {
    hipblasLtMatmulAlgo_t a = load_algo(algo_inout);
    size_t w = 0;
    hipblasStatus_t st = hipblaslt_ext::matmulIsAlgoSupported(
        static_cast<hipblasLtHandle_t>(handle), static_cast<hipblasLtMatmulDesc_t>(desc), alpha,
        static_cast<hipblasLtMatrixLayout_t>(la), static_cast<hipblasLtMatrixLayout_t>(lb), beta,
        static_cast<hipblasLtMatrixLayout_t>(lc), static_cast<hipblasLtMatrixLayout_t>(ld), a, w);
    std::memcpy(algo_inout, &a, sizeof(a));
    *ws = w;
    return static_cast<int>(st);
  } catch (...) {
    return -1;
  }
}

int e0x_algos_from_index(void* handle, const int* idx, int n, void* out, int max_n, int* n_out) {
  try {
    std::vector<int> want(idx, idx + n);
    std::vector<hipblasLtMatmulHeuristicResult_t> r;
    hipblasStatus_t st = hipblaslt_ext::getAlgosFromIndex(static_cast<hipblasLtHandle_t>(handle), want, r);
    const size_t m = r.size() < static_cast<size_t>(max_n) ? r.size() : static_cast<size_t>(max_n);
    if (m > 0) {
      std::memcpy(out, r.data(), m * sizeof(hipblasLtMatmulHeuristicResult_t));
    }
    *n_out = static_cast<int>(m);
    return static_cast<int>(st);
  } catch (...) {
    return -1;
  }
}

// A tuned instance (GemmTuning splitK / wgm; 0 = the solution's own value) of `algo` on the C-API problem. Returns an
// opaque handle or nullptr; *status = hipblasStatus_t of the failing step (-100: constructor threw, -101: needs more
// workspace than ws_bytes, -102: an exception after construction); *ws_need = the workspace the tuned algo needs.
void* e0x_gemm_create(void* handle, void* desc, const void* A, void* la, const void* B, void* lb, const void* C,
                      void* lc, void* D, void* ld, const void* algo, int splitk, int wgm, void* ws, size_t ws_bytes,
                      size_t* ws_need, int* status, char* kname, int klen) {
  E0Gemm* h = new E0Gemm();
  *ws_need = 0;
  try {
    h->mem = ::operator new(sizeof(hipblaslt_ext::Gemm) + kSlack);
    std::memset(h->mem, 0, sizeof(hipblaslt_ext::Gemm) + kSlack);
    h->g = new (h->mem) hipblaslt_ext::Gemm(
        static_cast<hipblasLtHandle_t>(handle), static_cast<hipblasLtMatmulDesc_t>(desc), &h->alpha, A,
        static_cast<hipblasLtMatrixLayout_t>(la), B, static_cast<hipblasLtMatrixLayout_t>(lb), &h->beta, C,
        static_cast<hipblasLtMatrixLayout_t>(lc), D, static_cast<hipblasLtMatrixLayout_t>(ld));
  } catch (...) {
    *status = -100;
    h->g = nullptr;
    drop(h);
    return nullptr;
  }
  try {
    hipblasLtMatmulAlgo_t a = load_algo(algo);
    hipblaslt_ext::GemmTuning tuning;
    tuning.setSplitK(static_cast<uint16_t>(splitk));
    tuning.setWgm(static_cast<int16_t>(wgm));
    h->g->setMaxWorkspaceBytes(ws_bytes);
    size_t need = 0;
    hipblasStatus_t st = h->g->isAlgoSupported(a, tuning, need);
    *ws_need = need;
    if (st != HIPBLAS_STATUS_SUCCESS) {
      *status = static_cast<int>(st);
      drop(h);
      return nullptr;
    }
    if (need > ws_bytes) {
      *status = -101;
      drop(h);
      return nullptr;
    }
    st = h->g->initialize(a, tuning, ws, false, nullptr);
    if (st != HIPBLAS_STATUS_SUCCESS) {
      *status = static_cast<int>(st);
      drop(h);
      return nullptr;
    }
    copy_str(h->g->getKernelName(), kname, klen);
    *status = 0;
    return h;
  } catch (...) {
    *status = -102;
    drop(h);
    return nullptr;
  }
}

int e0x_gemm_run(void* inst, void* stream) {
  try {
    return static_cast<int>(static_cast<E0Gemm*>(inst)->g->run(static_cast<hipStream_t>(stream)));
  } catch (...) {
    return -1;
  }
}

void e0x_gemm_destroy(void* inst) { drop(static_cast<E0Gemm*>(inst)); }

}  // extern "C"

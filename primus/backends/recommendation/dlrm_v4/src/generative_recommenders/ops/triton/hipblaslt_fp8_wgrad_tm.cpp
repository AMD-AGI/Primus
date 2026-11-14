// FP8 token-major weight gradient for the HSTU uvqk / vk projections
// (DLRM_HSTU_FP8_BWD_UVQK_WGRAD=fp8t, ops/triton/bwd_fp8.py). Default off: this
// module is only built and loaded when that mode is requested.
//
// Ported from the Arbor R14 testbed, where these entries lived in
// hipblaslt_scaled_addmm.cpp. They are a separate extension here so the
// shipped residual-fused extension (scaled_addmm) is neither changed nor
// rebuilt by this feature.
#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

void check_hipblaslt(hipblasStatus_t status, const char* operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed with status " +
        std::to_string(static_cast<int>(status)));
  }
}

hipblasLtHandle_t handle() {
  static hipblasLtHandle_t value = [] {
    hipblasLtHandle_t result;
    check_hipblaslt(hipblasLtCreate(&result), "hipblasLtCreate");
    return result;
  }();
  return value;
}

// Explicit hipBLASLt solution index -> algo. Resolved once per index; per-call
// support is re-checked against the exact descriptor by the caller.
bool solution_algo(int64_t index, hipblasLtMatmulAlgo_t* algo) {
  static std::mutex mutex;
  static std::unordered_map<int64_t, std::pair<bool, hipblasLtMatmulAlgo_t>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(index);
  if (it == cache.end()) {
    std::vector<int> indices{static_cast<int>(index)};
    std::vector<hipblasLtMatmulHeuristicResult_t> results;
    bool ok = hipblaslt_ext::getAlgosFromIndex(handle(), indices, results) ==
            HIPBLAS_STATUS_SUCCESS &&
        results.size() == 1;
    hipblasLtMatmulAlgo_t value{};
    if (ok) {
      value = results[0].algo;
    }
    it = cache.emplace(index, std::make_pair(ok, value)).first;
  }
  if (!it->second.first) {
    return false;
  }
  *algo = it->second.second;
  return true;
}

// FP8 weight gradient with a TOKEN-MAJOR activation.
//
//   dW[Kx, Ng] = act[K, Kx]^T @ grad[K, Ng]          (K = k_tokens tokens)
//
// grad   : e5m2 [>=K, Ng] row-major (the payload quantize_amax_colsum already
//          writes for the dgrad; may be a row slice of a taller buffer, ld =
//          grad.stride(0) >= Ng). Only its first K rows are read.
// act_t  : e4m3 [Kx, >=K] row-major, i.e. the activation stored token-major
//          (ld = act_t.stride(0) = ldM >= K). Only its first K columns are read.
// torch._scaled_mm only accepts TN, which would need a transposed copy of the
// gradient; keeping both operands row-major maps onto the library's slower NT
// kernels. Here only the activation is token-major, which maps onto the
// library's fast layouts:
//   orientation 0 ("NN"): C'[Ng, Kx] = A[Ng, K] (grad, op N, ld ldg)
//                                     @ B[K, Kx] (act_t, op N, ld ldM)
//                         -> returns dW   [Kx, Ng] row-major.
//   orientation 1 ("TT"): C [Kx, Ng] = A[Kx, K] (act_t, op T, ld ldM)
//                                     @ B[K, Ng] (grad, op T, ld ldg)
//                         -> returns dW^T [Ng, Kx] row-major (caller .t()).
// Scales are dequantization scales by device pointer (no sync).
struct WgradTm {
  hipblasLtMatrixLayout_t mat_a{}, mat_b{}, mat_c{}, mat_d{};
  hipblasLtMatmulDesc_t matmul{};
  const void* a_ptr = nullptr;
  const void* b_ptr = nullptr;
  int64_t out_rows = 0, out_cols = 0;
  hipDataType out_type = HIP_R_32F;

  WgradTm(
      const torch::Tensor& grad,
      const torch::Tensor& act_t,
      const torch::Tensor& grad_scale,
      const torch::Tensor& act_scale,
      bool out_fp32,
      int64_t k_tokens,
      int64_t orientation) {
    TORCH_CHECK(grad.is_cuda() && act_t.is_cuda());
    TORCH_CHECK(grad.dim() == 2 && act_t.dim() == 2);
    TORCH_CHECK(grad.scalar_type() == at::kFloat8_e5m2, "grad must be e5m2");
    TORCH_CHECK(
        act_t.scalar_type() == at::kFloat8_e4m3fn, "act_t must be e4m3fn");
    TORCH_CHECK(orientation == 0 || orientation == 1, "orientation 0|1");
    const int64_t ng = grad.size(1);
    const int64_t kx = act_t.size(0);
    const int64_t k = k_tokens > 0 ? k_tokens : grad.size(0);
    TORCH_CHECK(k <= grad.size(0), "k_tokens exceeds grad rows");
    TORCH_CHECK(k <= act_t.size(1), "k_tokens exceeds act_t columns");
    TORCH_CHECK(grad.stride(1) == 1 && grad.stride(0) >= ng, "grad layout");
    TORCH_CHECK(
        act_t.stride(1) == 1 && act_t.stride(0) >= act_t.size(1),
        "act_t layout");
    const int64_t ldg = grad.stride(0);
    const int64_t ldm = act_t.stride(0);
    out_type = out_fp32 ? HIP_R_32F : HIP_R_16BF;
    hipblasOperation_t trans_a, trans_b;
    const void* scale_a_ptr;
    const void* scale_b_ptr;
    if (orientation == 0) {
      check_hipblaslt(
          hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_8F_E5M2, ng, k, ldg),
          "tm layout A");
      check_hipblaslt(
          hipblasLtMatrixLayoutCreate(&mat_b, HIP_R_8F_E4M3, k, kx, ldm),
          "tm layout B");
      out_rows = kx;
      out_cols = ng;
      trans_a = HIPBLAS_OP_N;
      trans_b = HIPBLAS_OP_N;
      a_ptr = grad.data_ptr();
      b_ptr = act_t.data_ptr();
      scale_a_ptr = grad_scale.data_ptr();
      scale_b_ptr = act_scale.data_ptr();
    } else {
      check_hipblaslt(
          hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_8F_E4M3, k, kx, ldm),
          "tm layout A");
      check_hipblaslt(
          hipblasLtMatrixLayoutCreate(&mat_b, HIP_R_8F_E5M2, ng, k, ldg),
          "tm layout B");
      out_rows = ng;
      out_cols = kx;
      trans_a = HIPBLAS_OP_T;
      trans_b = HIPBLAS_OP_T;
      a_ptr = act_t.data_ptr();
      b_ptr = grad.data_ptr();
      scale_a_ptr = act_scale.data_ptr();
      scale_b_ptr = grad_scale.data_ptr();
    }
    // Column-major C is [out_cols, out_rows] == row-major [out_rows, out_cols].
    check_hipblaslt(
        hipblasLtMatrixLayoutCreate(
            &mat_c, out_type, out_cols, out_rows, out_cols),
        "tm layout C");
    check_hipblaslt(
        hipblasLtMatrixLayoutCreate(
            &mat_d, out_type, out_cols, out_rows, out_cols),
        "tm layout D");
    check_hipblaslt(
        hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F),
        "tm matmul descriptor");
    check_hipblaslt(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
        "tm transpose A");
    check_hipblaslt(
        hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
        "tm transpose B");
    check_hipblaslt(
        hipblasLtMatmulDescSetAttribute(
            matmul,
            HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &scale_a_ptr,
            sizeof(scale_a_ptr)),
        "tm scale A");
    check_hipblaslt(
        hipblasLtMatmulDescSetAttribute(
            matmul,
            HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &scale_b_ptr,
            sizeof(scale_b_ptr)),
        "tm scale B");
  }
  ~WgradTm() {
    hipblasLtMatmulDescDestroy(matmul);
    hipblasLtMatrixLayoutDestroy(mat_a);
    hipblasLtMatrixLayoutDestroy(mat_b);
    hipblasLtMatrixLayoutDestroy(mat_c);
    hipblasLtMatrixLayoutDestroy(mat_d);
  }
  bool supported(hipblasLtMatmulAlgo_t& algo, size_t& needed) {
    float alpha = 1.0f;
    float beta = 0.0f;
    needed = 0;
    return hipblaslt_ext::matmulIsAlgoSupported(
               handle(),
               matmul,
               &alpha,
               mat_a,
               mat_b,
               &beta,
               mat_c,
               mat_d,
               algo,
               needed) == HIPBLAS_STATUS_SUCCESS;
  }
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(
      uint64_t workspace_bytes,
      int requested) {
    hipblasLtMatmulPreference_t preference;
    check_hipblaslt(
        hipblasLtMatmulPreferenceCreate(&preference), "tm preference");
    check_hipblaslt(
        hipblasLtMatmulPreferenceSetAttribute(
            preference,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes,
            sizeof(workspace_bytes)),
        "tm workspace preference");
    std::vector<hipblasLtMatmulHeuristicResult_t> results(requested);
    int returned = 0;
    hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(
        handle(),
        matmul,
        mat_a,
        mat_b,
        mat_c,
        mat_d,
        preference,
        requested,
        results.data(),
        &returned);
    hipblasLtMatmulPreferenceDestroy(preference);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      returned = 0;
    }
    results.resize(returned);
    return results;
  }
};

torch::Tensor fp8_wgrad_tm(
    const torch::Tensor& grad,
    const torch::Tensor& act_t,
    const torch::Tensor& grad_scale,
    const torch::Tensor& act_scale,
    const torch::Tensor& workspace,
    bool out_fp32,
    int64_t algorithm_index,
    int64_t solution_index,
    int64_t k_tokens,
    int64_t orientation,
    bool strict_solution) {
  WgradTm problem(
      grad, act_t, grad_scale, act_scale, out_fp32, k_tokens, orientation);
  const uint64_t workspace_bytes = workspace.numel() * workspace.element_size();
  hipblasLtMatmulAlgo_t chosen{};
  bool have_chosen = false;
  if (solution_index >= 0 && solution_algo(solution_index, &chosen)) {
    size_t needed = 0;
    have_chosen =
        problem.supported(chosen, needed) && needed <= workspace_bytes;
  }
  if (!have_chosen) {
    TORCH_CHECK(
        !(strict_solution && solution_index >= 0),
        "fp8_wgrad_tm: solution ",
        solution_index,
        " unsupported for this problem");
    auto heuristics = problem.heuristic(workspace_bytes, 16);
    TORCH_CHECK(
        algorithm_index >= 0 &&
            algorithm_index < static_cast<int64_t>(heuristics.size()),
        "fp8_wgrad_tm algorithm index ",
        algorithm_index,
        " is outside [0, ",
        heuristics.size(),
        ")");
    chosen = heuristics[algorithm_index].algo;
  }
  auto output = torch::empty(
      {problem.out_rows, problem.out_cols},
      grad.options().dtype(out_fp32 ? at::kFloat : at::kBFloat16));
  float alpha = 1.0f;
  float beta = 0.0f;
  check_hipblaslt(
      hipblasLtMatmul(
          handle(),
          problem.matmul,
          &alpha,
          problem.a_ptr,
          problem.mat_a,
          problem.b_ptr,
          problem.mat_b,
          &beta,
          output.data_ptr(),
          problem.mat_c,
          output.data_ptr(),
          problem.mat_d,
          &chosen,
          workspace.data_ptr(),
          workspace_bytes,
          at::hip::getCurrentHIPStream()),
      "tm hipblasLtMatmul");
  return output;
}

// Enumerate candidate solutions for the fp8_wgrad_tm problem. all=false: the
// heuristic list in rank order (top `requested`); all=true: every registered
// solution of this type/op signature (getAllAlgos). Each entry is
// (solution index, kernel name, supported for this exact descriptor,
//  workspace bytes needed, heuristic rank or -1).
std::vector<std::tuple<int64_t, std::string, bool, int64_t, int64_t>>
fp8_wgrad_tm_algos(
    const torch::Tensor& grad,
    const torch::Tensor& act_t,
    const torch::Tensor& grad_scale,
    const torch::Tensor& act_scale,
    const torch::Tensor& workspace,
    bool out_fp32,
    int64_t k_tokens,
    int64_t orientation,
    bool all,
    int64_t requested) {
  WgradTm problem(
      grad, act_t, grad_scale, act_scale, out_fp32, k_tokens, orientation);
  const uint64_t workspace_bytes = workspace.numel() * workspace.element_size();
  std::vector<std::tuple<int64_t, std::string, bool, int64_t, int64_t>> out;
  auto heuristics =
      problem.heuristic(workspace_bytes, static_cast<int>(requested));
  if (!all) {
    for (size_t i = 0; i < heuristics.size(); ++i) {
      size_t needed = 0;
      bool ok = problem.supported(heuristics[i].algo, needed);
      out.emplace_back(
          hipblaslt_ext::getIndexFromAlgo(heuristics[i].algo),
          hipblaslt_ext::getKernelNameFromAlgo(handle(), heuristics[i].algo),
          ok,
          static_cast<int64_t>(needed),
          static_cast<int64_t>(i));
    }
    return out;
  }
  std::unordered_map<int, int64_t> rank;
  for (size_t i = 0; i < heuristics.size(); ++i) {
    rank[hipblaslt_ext::getIndexFromAlgo(heuristics[i].algo)] =
        static_cast<int64_t>(i);
  }
  hipblasOperation_t op = orientation == 0 ? HIPBLAS_OP_N : HIPBLAS_OP_T;
  const hipDataType type_a = orientation == 0 ? HIP_R_8F_E5M2 : HIP_R_8F_E4M3;
  const hipDataType type_b = orientation == 0 ? HIP_R_8F_E4M3 : HIP_R_8F_E5M2;
  std::vector<hipblasLtMatmulHeuristicResult_t> results;
  check_hipblaslt(
      hipblaslt_ext::getAllAlgos(
          handle(),
          hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
          op,
          op,
          type_a,
          type_b,
          problem.out_type,
          problem.out_type,
          HIPBLAS_COMPUTE_32F,
          results),
      "tm getAllAlgos");
  for (auto& result : results) {
    size_t needed = 0;
    bool ok = problem.supported(result.algo, needed);
    int index = hipblaslt_ext::getIndexFromAlgo(result.algo);
    auto it = rank.find(index);
    out.emplace_back(
        index,
        ok ? hipblaslt_ext::getKernelNameFromAlgo(handle(), result.algo)
           : std::string(),
        ok,
        static_cast<int64_t>(needed),
        it == rank.end() ? -1 : it->second);
  }
  return out;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fp8_wgrad_tm",
      &fp8_wgrad_tm,
      pybind11::arg("grad"),
      pybind11::arg("act_t"),
      pybind11::arg("grad_scale"),
      pybind11::arg("act_scale"),
      pybind11::arg("workspace"),
      pybind11::arg("out_fp32") = true,
      pybind11::arg("algorithm_index") = 0,
      pybind11::arg("solution_index") = -1,
      pybind11::arg("k_tokens") = -1,
      pybind11::arg("orientation") = 0,
      pybind11::arg("strict_solution") = false);
  module.def(
      "fp8_wgrad_tm_algos",
      &fp8_wgrad_tm_algos,
      pybind11::arg("grad"),
      pybind11::arg("act_t"),
      pybind11::arg("grad_scale"),
      pybind11::arg("act_scale"),
      pybind11::arg("workspace"),
      pybind11::arg("out_fp32") = true,
      pybind11::arg("k_tokens") = -1,
      pybind11::arg("orientation") = 0,
      pybind11::arg("all") = false,
      pybind11::arg("requested") = 16);
}

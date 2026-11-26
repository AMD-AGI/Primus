#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <stdexcept>
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

torch::Tensor scaled_addmm(
    const torch::Tensor& activation,
    const torch::Tensor& weight,
    const torch::Tensor& residual,
    const torch::Tensor& activation_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& workspace,
    int64_t algorithm_index) {
  TORCH_CHECK(activation.is_cuda() && weight.is_cuda() && residual.is_cuda());
  TORCH_CHECK(activation.dim() == 2 && weight.dim() == 2);
  TORCH_CHECK(residual.dim() == 2);
  const int64_t m = activation.size(0);
  const int64_t k = activation.size(1);
  const int64_t n = weight.size(1);
  TORCH_CHECK(weight.size(0) == k);
  TORCH_CHECK(residual.size(0) == m && residual.size(1) == n);
  TORCH_CHECK(activation.stride(1) == 1 && activation.stride(0) == k);
  TORCH_CHECK(weight.stride(0) == 1 && weight.stride(1) == k);
  TORCH_CHECK(residual.stride(1) == 1 && residual.stride(0) == n);

  auto output = torch::empty_like(residual);
  hipblasLtMatrixLayout_t mat_a;
  hipblasLtMatrixLayout_t mat_b;
  hipblasLtMatrixLayout_t mat_c;
  hipblasLtMatrixLayout_t mat_d;
  hipblasLtMatmulDesc_t matmul;
  hipblasLtMatmulPreference_t preference;

  // Evaluate row-major [M,K] @ column-major [K,N] as the equivalent
  // column-major [N,K] @ [K,M] -> [N,M].
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&mat_a, HIP_R_8F_E4M3, k, n, k),
      "layout A");
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&mat_b, HIP_R_8F_E4M3, k, m, k),
      "layout B");
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&mat_c, HIP_R_16BF, n, m, n),
      "layout C");
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&mat_d, HIP_R_16BF, n, m, n),
      "layout D");
  check_hipblaslt(
      hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F),
      "matmul descriptor");
  hipblasOperation_t trans_a = HIPBLAS_OP_T;
  hipblasOperation_t trans_b = HIPBLAS_OP_N;
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(
          matmul,
          HIPBLASLT_MATMUL_DESC_TRANSA,
          &trans_a,
          sizeof(trans_a)),
      "transpose A");
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(
          matmul,
          HIPBLASLT_MATMUL_DESC_TRANSB,
          &trans_b,
          sizeof(trans_b)),
      "transpose B");
  const void* scale_a_ptr = weight_scale.data_ptr();
  const void* scale_b_ptr = activation_scale.data_ptr();
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(
          matmul,
          HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
          &scale_a_ptr,
          sizeof(scale_a_ptr)),
      "scale A");
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(
          matmul,
          HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,
          &scale_b_ptr,
          sizeof(scale_b_ptr)),
      "scale B");

  check_hipblaslt(
      hipblasLtMatmulPreferenceCreate(&preference), "preference create");
  const uint64_t workspace_bytes = workspace.numel() * workspace.element_size();
  check_hipblaslt(
      hipblasLtMatmulPreferenceSetAttribute(
          preference,
          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_bytes,
          sizeof(workspace_bytes)),
      "workspace preference");

  constexpr int requested_algorithms = 32;
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristics(
      requested_algorithms);
  int returned = 0;
  check_hipblaslt(
      hipblasLtMatmulAlgoGetHeuristic(
          handle(),
          matmul,
          mat_a,
          mat_b,
          mat_c,
          mat_d,
          preference,
          requested_algorithms,
          heuristics.data(),
          &returned),
      "algorithm heuristic");
  TORCH_CHECK(
      algorithm_index >= 0 && algorithm_index < returned,
      "scaled addmm algorithm index ",
      algorithm_index,
      " is outside [0, ",
      returned,
      ")");

  float alpha = 1.0f;
  float beta = 1.0f;
  hipStream_t stream = at::hip::getCurrentHIPStream();
  check_hipblaslt(
      hipblasLtMatmul(
          handle(),
          matmul,
          &alpha,
          weight.data_ptr(),
          mat_a,
          activation.data_ptr(),
          mat_b,
          &beta,
          residual.data_ptr(),
          mat_c,
          output.data_ptr(),
          mat_d,
          &heuristics[algorithm_index].algo,
          workspace.data_ptr(),
          workspace_bytes,
          stream),
      "hipblasLtMatmul");

  check_hipblaslt(hipblasLtMatmulPreferenceDestroy(preference), "destroy pref");
  check_hipblaslt(hipblasLtMatmulDescDestroy(matmul), "destroy matmul");
  check_hipblaslt(hipblasLtMatrixLayoutDestroy(mat_a), "destroy A");
  check_hipblaslt(hipblasLtMatrixLayoutDestroy(mat_b), "destroy B");
  check_hipblaslt(hipblasLtMatrixLayoutDestroy(mat_c), "destroy C");
  check_hipblaslt(hipblasLtMatrixLayoutDestroy(mat_d), "destroy D");
  return output;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("scaled_addmm", &scaled_addmm);
}

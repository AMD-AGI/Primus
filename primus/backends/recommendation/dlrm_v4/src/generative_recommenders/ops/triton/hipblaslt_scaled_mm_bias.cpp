// FP8 GEMM + bias through hipBLASLt with a caller-sized workspace.
//
// This reproduces, call for call, what torch._scaled_mm does on ROCm for a
// tensor-wise scaled e4m3 x e4m3 + bf16 bias -> bf16 product (torch 2.12,
// aten/src/ATen/native/cuda/ScaledBlas.cpp _scaled_gemm and
// aten/src/ATen/cuda/CUDABlas.cpp scaled_gemm): the row-major [M,K] x
// column-major [K,N] product is evaluated as the column-major
// D[N,M] = W^T (op T) x X (op N), compute HIPBLAS_COMPUTE_32F / HIP_R_32F,
// A scale = weight scale, B scale = activation scale, bias epilogue in bf16,
// C = D layout, alpha 1, beta 0, one heuristic candidate requested, current
// stream. The only difference is the workspace: torch passes its process-wide
// hipBLASLt workspace (HIPBLASLT_WORKSPACE_SIZE, 76 MiB by default) to both the
// heuristic and the call; this passes the caller's tensor. hipBLASLt's
// stream-K grid selection falls back to the data-parallel grid when the
// partial-tile buffer of the persistent grid does not fit in that workspace,
// so a workspace below 63.75 MiB keeps the K=512 U projection on the
// data-parallel grid at every M without moving any other GEMM of the process.
#include <torch/extension.h>

#include <ATen/hip/HIPContext.h>
#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace {

void check_hipblaslt(hipblasStatus_t status, const char* operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string("hipblaslt_scaled_mm_bias: ") + operation +
        " failed with status " + std::to_string(static_cast<int>(status)));
  }
}

// One handle per device, created on first use with that device current.
hipblasLtHandle_t handle_for_current_device() {
  static std::mutex mutex;
  static std::unordered_map<int, hipblasLtHandle_t> handles;
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    throw std::runtime_error("hipblaslt_scaled_mm_bias: hipGetDevice failed");
  }
  std::lock_guard<std::mutex> lock(mutex);
  auto found = handles.find(device);
  if (found != handles.end()) {
    return found->second;
  }
  hipblasLtHandle_t created;
  check_hipblaslt(hipblasLtCreate(&created), "hipblasLtCreate");
  handles.emplace(device, created);
  return created;
}

struct LayoutDeleter {
  void operator()(std::remove_pointer_t<hipblasLtMatrixLayout_t>* p) const {
    hipblasLtMatrixLayoutDestroy(p);
  }
};
struct DescDeleter {
  void operator()(std::remove_pointer_t<hipblasLtMatmulDesc_t>* p) const {
    hipblasLtMatmulDescDestroy(p);
  }
};
struct PrefDeleter {
  void operator()(std::remove_pointer_t<hipblasLtMatmulPreference_t>* p) const {
    hipblasLtMatmulPreferenceDestroy(p);
  }
};
using Layout =
    std::unique_ptr<std::remove_pointer_t<hipblasLtMatrixLayout_t>, LayoutDeleter>;
using Desc =
    std::unique_ptr<std::remove_pointer_t<hipblasLtMatmulDesc_t>, DescDeleter>;
using Pref = std::unique_ptr<
    std::remove_pointer_t<hipblasLtMatmulPreference_t>,
    PrefDeleter>;

Layout make_layout(hipDataType type, uint64_t rows, uint64_t cols, int64_t ld) {
  hipblasLtMatrixLayout_t raw = nullptr;
  check_hipblaslt(
      hipblasLtMatrixLayoutCreate(&raw, type, rows, cols, ld), "layout create");
  return Layout(raw);
}

template <typename T>
void set_desc(hipblasLtMatmulDesc_t desc, hipblasLtMatmulDescAttributes_t attr,
              const T value, const char* what) {
  check_hipblaslt(
      hipblasLtMatmulDescSetAttribute(desc, attr, &value, sizeof(value)), what);
}

// activation: e4m3 [M,K] row-major contiguous; weight: e4m3 [K,N] column-major
// (stride (1,K), the layout triton_fp8._get_weight_shadow returns); bias: bf16
// [N] contiguous; scales: fp32 single-element device tensors; workspace: uint8
// device tensor whose byte size is both the heuristic's MAX_WORKSPACE_BYTES and
// the call's workspace size. Returns bf16 [M,N] row-major.
torch::Tensor scaled_mm_bias(
    const torch::Tensor& activation,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& activation_scale,
    const torch::Tensor& weight_scale,
    const torch::Tensor& workspace) {
  TORCH_CHECK(
      activation.is_cuda() && weight.is_cuda() && bias.is_cuda() &&
          activation_scale.is_cuda() && weight_scale.is_cuda() &&
          workspace.is_cuda(),
      "scaled_mm_bias: every operand must be a GPU tensor");
  const auto device = activation.device();
  TORCH_CHECK(
      weight.device() == device && bias.device() == device &&
          activation_scale.device() == device &&
          weight_scale.device() == device && workspace.device() == device,
      "scaled_mm_bias: operands must share one device");
  TORCH_CHECK(
      activation.scalar_type() == at::kFloat8_e4m3fn &&
          weight.scalar_type() == at::kFloat8_e4m3fn,
      "scaled_mm_bias: activation and weight must be float8_e4m3fn");
  TORCH_CHECK(activation.dim() == 2 && weight.dim() == 2, "scaled_mm_bias: 2-D");
  const int64_t m = activation.size(0);
  const int64_t k = activation.size(1);
  const int64_t n = weight.size(1);
  TORCH_CHECK(weight.size(0) == k, "scaled_mm_bias: K mismatch");
  TORCH_CHECK(activation.is_contiguous(), "scaled_mm_bias: activation must be row-major contiguous");
  TORCH_CHECK(
      weight.stride(0) == 1 && weight.stride(1) == k,
      "scaled_mm_bias: weight must be column-major [K,N] (stride (1,K))");
  TORCH_CHECK(
      bias.scalar_type() == at::kBFloat16 && bias.dim() == 1 &&
          bias.size(0) == n && bias.is_contiguous(),
      "scaled_mm_bias: bias must be contiguous bf16 [N]");
  TORCH_CHECK(
      activation_scale.scalar_type() == at::kFloat &&
          activation_scale.numel() == 1 &&
          weight_scale.scalar_type() == at::kFloat && weight_scale.numel() == 1,
      "scaled_mm_bias: scales must be single-element fp32 tensors");
  TORCH_CHECK(
      workspace.scalar_type() == at::kByte && workspace.is_contiguous(),
      "scaled_mm_bias: workspace must be a contiguous uint8 tensor");

  const c10::DeviceGuard guard(device);
  auto output = torch::empty({m, n}, activation.options().dtype(at::kBFloat16));
  if (m == 0 || n == 0) {
    return output;
  }

  hipblasLtMatmulDesc_t raw_desc = nullptr;
  check_hipblaslt(
      hipblasLtMatmulDescCreate(&raw_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
      "matmul descriptor");
  Desc desc(raw_desc);
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_TRANSA, HIPBLAS_OP_T, "transpose A");
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_TRANSB, HIPBLAS_OP_N, "transpose B");
  const void* scale_a_ptr = weight_scale.data_ptr();
  const void* scale_b_ptr = activation_scale.data_ptr();
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, scale_a_ptr, "scale A");
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, scale_b_ptr, "scale B");
  const void* bias_ptr = bias.data_ptr();
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr, "bias pointer");
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, HIPBLASLT_EPILOGUE_BIAS, "epilogue");
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, HIP_R_16BF, "bias type");
#if defined(HIPBLASLT_OUTER_VEC)
  // torch sets the tensor-wise scale mode explicitly when its hipBLASLt has
  // the SCALE_MODE attributes (value 0 = scalar fp32, also the default).
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, int{0}, "scale mode A");
  set_desc(raw_desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, int{0}, "scale mode B");
#endif

  // Column-major view: A = W stored [K,N] ld K (op T), B = X stored [K,M]
  // ld K (op N), C = D stored [N,M] ld N.
  Layout mat_a = make_layout(HIP_R_8F_E4M3, k, n, k);
  Layout mat_b = make_layout(HIP_R_8F_E4M3, k, m, k);
  Layout mat_c = make_layout(HIP_R_16BF, n, m, n);
  Layout mat_d = make_layout(HIP_R_16BF, n, m, n);

  hipblasLtMatmulPreference_t raw_pref = nullptr;
  check_hipblaslt(hipblasLtMatmulPreferenceCreate(&raw_pref), "preference create");
  Pref pref(raw_pref);
  const size_t workspace_bytes =
      static_cast<size_t>(workspace.numel()) * workspace.element_size();
  check_hipblaslt(
      hipblasLtMatmulPreferenceSetAttribute(
          raw_pref,
          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &workspace_bytes,
          sizeof(workspace_bytes)),
      "workspace preference");

  hipblasLtHandle_t lt = handle_for_current_device();
  hipblasLtMatmulHeuristicResult_t heuristic = {};
  int returned = 0;
  check_hipblaslt(
      hipblasLtMatmulAlgoGetHeuristic(
          lt,
          raw_desc,
          mat_a.get(),
          mat_b.get(),
          mat_c.get(),
          mat_d.get(),
          raw_pref,
          1,
          &heuristic,
          &returned),
      "algorithm heuristic");
  TORCH_CHECK(
      returned > 0,
      "scaled_mm_bias: hipBLASLt heuristic returned no algorithm for M=", m,
      " N=", n, " K=", k, " workspace=", workspace_bytes);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  hipStream_t stream = at::hip::getCurrentHIPStream();
  check_hipblaslt(
      hipblasLtMatmul(
          lt,
          raw_desc,
          &alpha,
          weight.data_ptr(),
          mat_a.get(),
          activation.data_ptr(),
          mat_b.get(),
          &beta,
          output.data_ptr(),
          mat_c.get(),
          output.data_ptr(),
          mat_d.get(),
          &heuristic.algo,
          workspace.data_ptr(),
          workspace_bytes,
          stream),
      "hipblasLtMatmul");
  return output;
}

bool scale_mode_attribute_set() {
#if defined(HIPBLASLT_OUTER_VEC)
  return true;
#else
  return false;
#endif
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("scaled_mm_bias", &scaled_mm_bias);
  module.def("scale_mode_attribute_set", &scale_mode_attribute_set);
}

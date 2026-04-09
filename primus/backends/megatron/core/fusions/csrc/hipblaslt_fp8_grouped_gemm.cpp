/*
 * FP8 Grouped GEMM via C++ loop over torch._scaled_mm.
 *
 * Eliminates Python dispatch overhead by looping in C++ over experts,
 * calling at::_scaled_mm for each expert, then concatenating results.
 *
 * Build: torch.utils.cpp_extension.load(...)
 */

#include <torch/extension.h>
#include <ATen/Functions.h>
#include <vector>

/*
 * Forward: D[total_M, N] = A_fp8[total_M, K] @ W_fp8_NK[E, N, K]^T
 */
torch::Tensor fp8_grouped_gemm_fwd(
    torch::Tensor A_fp8,       // [total_M, K] FP8 contiguous
    torch::Tensor W_fp8_NK,    // [E, N, K]   FP8 contiguous
    torch::Tensor scale_a,     // scalar float32 on GPU
    torch::Tensor scale_b,     // scalar float32 on GPU
    torch::Tensor tpe)         // [E] int64
{
  TORCH_CHECK(A_fp8.is_contiguous(), "A_fp8 must be contiguous");
  TORCH_CHECK(W_fp8_NK.is_contiguous(), "W_fp8_NK must be contiguous");

  const int64_t E       = W_fp8_NK.size(0);
  const int64_t N       = W_fp8_NK.size(1);
  const int64_t total_M = A_fp8.size(0);

  auto tpe_cpu = tpe.to(torch::kCPU, torch::kInt64).contiguous();
  auto* tpe_ptr = tpe_cpu.data_ptr<int64_t>();

  std::vector<torch::Tensor> chunks;
  chunks.reserve(E);

  int64_t offset = 0;
  for (int64_t e = 0; e < E; e++) {
    int64_t m = tpe_ptr[e];
    if (m <= 0) continue;

    auto A_e = A_fp8.narrow(0, offset, m);
    auto W_e_t = W_fp8_NK[e].t();

    auto C_e = at::_scaled_mm(
        A_e, W_e_t,
        scale_a, scale_b,
        /*bias=*/c10::nullopt,
        /*scale_result=*/c10::nullopt,
        /*out_dtype=*/torch::kBFloat16,
        /*use_fast_accum=*/false);

    chunks.push_back(C_e);
    offset += m;
  }

  if (chunks.empty())
    return torch::empty({0, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_fp8.device()));

  return torch::cat(chunks, 0);
}


/*
 * dA backward: dA[total_M, K] = grad_fp8[total_M, N] @ W_fp8_KN[E, K, N]^T
 * = grad[m, N] @ W_NK[N, K]  via  _scaled_mm(grad, W_KN.t())
 */
torch::Tensor fp8_grouped_gemm_dA(
    torch::Tensor grad_fp8,     // [total_M, N] FP8 contiguous
    torch::Tensor W_fp8_KN,     // [E, K, N]   FP8 contiguous
    torch::Tensor scale_grad,   // scalar float32
    torch::Tensor scale_w,      // scalar float32
    torch::Tensor tpe)          // [E] int64
{
  TORCH_CHECK(grad_fp8.is_contiguous(), "grad_fp8 must be contiguous");
  TORCH_CHECK(W_fp8_KN.is_contiguous(), "W_fp8_KN must be contiguous");

  const int64_t E       = W_fp8_KN.size(0);
  const int64_t K_dim   = W_fp8_KN.size(1);
  const int64_t total_M = grad_fp8.size(0);

  auto tpe_cpu = tpe.to(torch::kCPU, torch::kInt64).contiguous();
  auto* tpe_ptr = tpe_cpu.data_ptr<int64_t>();

  std::vector<torch::Tensor> chunks;
  chunks.reserve(E);

  int64_t offset = 0;
  for (int64_t e = 0; e < E; e++) {
    int64_t m = tpe_ptr[e];
    if (m <= 0) continue;

    auto g_e = grad_fp8.narrow(0, offset, m);
    auto W_e_t = W_fp8_KN[e].t();

    auto dA_e = at::_scaled_mm(
        g_e, W_e_t,
        scale_grad, scale_w,
        /*bias=*/c10::nullopt,
        /*scale_result=*/c10::nullopt,
        /*out_dtype=*/torch::kBFloat16,
        /*use_fast_accum=*/false);

    chunks.push_back(dA_e);
    offset += m;
  }

  if (chunks.empty())
    return torch::empty({0, K_dim},
        torch::TensorOptions().dtype(torch::kBFloat16).device(grad_fp8.device()));

  return torch::cat(chunks, 0);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_grouped_gemm_fwd", &fp8_grouped_gemm_fwd,
        "FP8 Grouped GEMM Forward (C++ _scaled_mm loop)");
  m.def("fp8_grouped_gemm_dA", &fp8_grouped_gemm_dA,
        "FP8 Grouped GEMM dA backward (C++ _scaled_mm loop)");
}

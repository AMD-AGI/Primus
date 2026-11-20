// CPU emulation of fused_adam_clip_fp8blk_kernel: the kernel's own per-thread phase
// functions (fused_adam_clip_kernel.cu, compiled for the host) driven by a sequential
// loop over the 512 threads of each 128x128 tile. Used by test_fp8blk_adam_cpu.py.
#define PRIMUS_FP8BLK_HOST_TEST 1
#include "../../primus/backends/megatron/patches/te_patches/fused_adam_clip_kernel.cu"
#include <torch/extension.h>
#include <vector>

namespace {

template <bool AdamW, typename MomentT, typename GradT>
void run_tiles(torch::Tensor grad, torch::Tensor param, torch::Tensor m, torch::Tensor v,
               torch::Tensor model, torch::Tensor w, torch::Tensor wt, torch::Tensor scale_inv,
               torch::Tensor e8m0, torch::Tensor e8m0_t, int64_t B, int64_t N, int64_t K,
               float beta1, float beta2, float step_size, float beta2_corr_inv, float eps,
               float lr, float decay, float grad_scale) {
  const int64_t tn = (N + kTile - 1) / kTile, tk = (K + kTile - 1) / kTile;
  const int64_t dims[4] = {N, K, tn, tk};
  Fp8BlkPtrs<GradT, MomentT> ptr;
  ptr.grad = reinterpret_cast<GradT*>(grad.data_ptr());
  ptr.param = param.data_ptr<float>();
  ptr.exp_avg = reinterpret_cast<MomentT*>(m.data_ptr());
  ptr.exp_avg_sq = reinterpret_cast<MomentT*>(v.data_ptr());
  ptr.model_param = reinterpret_cast<c10::BFloat16*>(model.data_ptr());
  ptr.w = reinterpret_cast<uint8_t*>(w.data_ptr());
  ptr.wt = reinterpret_cast<uint8_t*>(wt.data_ptr());
  ptr.scale_inv = scale_inv.data_ptr<float>();
  ptr.e8m0 = e8m0.data_ptr<uint8_t>();
  ptr.e8m0_t = e8m0_t.data_ptr<uint8_t>();
  std::vector<uint8_t> lds(kTile * kTilePitch);
  std::vector<float> xs(static_cast<size_t>(kBlockSize) * kTileIters * kIlp);
  for (int64_t tile = 0; tile < B * tn * tk; ++tile) {
    const Fp8BlkTile t = fp8blk_tile_coords(dims, 0, tile);
    float amax = 0.0f;
    for (int tid = 0; tid < kBlockSize; ++tid) {
      auto& x = *reinterpret_cast<float(*)[kTileIters][kIlp]>(&xs[tid * kTileIters * kIlp]);
      amax = fmaxf(amax, fp8blk_adam_phase<AdamW, MomentT, GradT>(
                             tid, t, ptr, beta1, beta2, step_size, beta2_corr_inv, eps, lr,
                             decay, grad_scale, x));
    }
    const float s = fp8blk_scale(amax);
    std::fill(lds.begin(), lds.end(), 0xAB);
    for (int tid = 0; tid < kBlockSize; ++tid) {
      auto& x = *reinterpret_cast<float(*)[kTileIters][kIlp]>(&xs[tid * kTileIters * kIlp]);
      fp8blk_quant_phase(tid, t, ptr.w, s, x, lds.data());
    }
    for (int tid = 0; tid < kBlockSize; ++tid) fp8blk_transpose_phase(tid, t, ptr.wt, lds.data());
    fp8blk_scale_store(t, s, ptr.scale_inv, ptr.e8m0, ptr.e8m0_t);
  }
}

std::vector<double> run(torch::Tensor grad, torch::Tensor param, torch::Tensor m, torch::Tensor v,
                        torch::Tensor model, torch::Tensor w, torch::Tensor wt,
                        torch::Tensor scale_inv, torch::Tensor e8m0, torch::Tensor e8m0_t,
                        int64_t B, int64_t N, int64_t K, double lr, double beta1, double beta2,
                        double eps, int64_t step, int64_t mode, double decay, double grad_scale) {
  // Same scalar derivation as the fused_adam_clip_fp8blk launcher.
  float correction1 = 1.0f - std::pow(static_cast<float>(beta1), static_cast<int>(step));
  float correction2 = 1.0f - std::pow(static_cast<float>(beta2), static_cast<int>(step));
  const float step_size = static_cast<float>(lr) / correction1;
  const float beta2_corr_inv = 1.0f / correction2;
  const bool g32 = grad.scalar_type() == at::kFloat, m32 = m.scalar_type() == at::kFloat;
#define RUN(AW, MT, GT)                                                                     \
  run_tiles<AW, MT, GT>(grad, param, m, v, model, w, wt, scale_inv, e8m0, e8m0_t, B, N, K, \
                        (float)beta1, (float)beta2, step_size, beta2_corr_inv, (float)eps,  \
                        (float)lr, (float)decay, (float)grad_scale)
  if (mode == 1) {
    if (m32 && g32) RUN(true, float, float);
    else if (m32) RUN(true, float, c10::BFloat16);
    else if (g32) RUN(true, c10::BFloat16, float);
    else RUN(true, c10::BFloat16, c10::BFloat16);
  } else {
    if (m32 && g32) RUN(false, float, float);
    else if (m32) RUN(false, float, c10::BFloat16);
    else if (g32) RUN(false, c10::BFloat16, float);
    else RUN(false, c10::BFloat16, c10::BFloat16);
  }
#undef RUN
  return {step_size, beta2_corr_inv};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run); }

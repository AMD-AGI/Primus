import torch
from primus_turbo.pytorch.kernels.quantization.quantization_impl import quantize_fp8_tensorwise_pad_impl
def bench(fn, iters=50):
    for _ in range(5): fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters
for m, k in [(262144, 2048), (262144, 768), (262144, 1536), (32768, 2048), (32768, 4096), (32768, 5120)]:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    p = x.abs().amax().float().reshape(1)
    full = bench(lambda: quantize_fp8_tensorwise_pad_impl(x, torch.float8_e4m3fn, k_align=128))
    qonly = bench(lambda: quantize_fp8_tensorwise_pad_impl(x, torch.float8_e4m3fn, k_align=128, amax_partials=p))
    print(f"{m}x{k}: full {full:6.1f} us   quant-only(+scale) {qonly:6.1f} us  {m*k*3/qonly/1e6:4.2f} TB/s")

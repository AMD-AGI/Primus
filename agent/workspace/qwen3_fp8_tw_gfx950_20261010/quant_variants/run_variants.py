import os
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(
    name="quant_variants",
    sources=[os.path.join(HERE, "quant_variants.hip")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-std=c++20"],
    build_directory=os.path.join(HERE, "build"),
    verbose=False,
)

from primus_turbo.pytorch.kernels.quantization.quantization_impl import (  # noqa: E402
    quantize_fp8_tensorwise_pad_impl,
)


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


SHAPES = [(262144, 2048), (262144, 768), (262144, 1536), (32768, 2048), (32768, 4096), (32768, 5120), (4096, 768)]
VARIANTS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(9))
for dt in (torch.float8_e4m3fn, torch.float8_e5m2):
    print(f"== {dt}")
    print(f"{'shape':>14s} " + " ".join(f"{'v' + str(v):>12s}" for v in VARIANTS))
    for m, k in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 3
        ref, sinv = quantize_fp8_tensorwise_pad_impl(x, dt, k_align=128)
        scale = (1.0 / sinv).reshape(1).float().contiguous()
        cells = []
        for v in VARIANTS:
            y = torch.empty(m, k, device="cuda", dtype=dt)
            ext.run(v, x, scale, y)
            ok = torch.equal(y.view(torch.uint8), ref.view(torch.uint8))
            us = bench(lambda: ext.run(v, x, scale, y))
            bw = m * k * 3 / us / 1e6
            cells.append(f"{us:6.1f}/{bw:4.2f}{'' if ok else '!'}")
        print(f"{m:7d}x{k:<6d} " + " ".join(f"{c:>12s}" for c in cells))
        del x, ref

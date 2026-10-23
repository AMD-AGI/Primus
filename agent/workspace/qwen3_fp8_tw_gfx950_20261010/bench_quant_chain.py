"""Item 1 microbench: tensorwise FP8 quant chain at Qwen3-30B-A3B (EP8, MBS8) shapes.

Per-kernel time comes from torch.profiler so amax / scale / quant / transpose are
separated; bandwidth = bytes moved / kernel time.
"""

import argparse
import collections
import re

import torch
from torch.profiler import ProfilerActivity, profile

from primus_turbo.pytorch.kernels.quantization.quantization_impl import (
    quantize_fp8_tensorwise_pad_impl,
)

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2

ROWS_GG = 262144
TOK = 32768
# (tag, rows, cols, fp8 dtype, calls per layer-microbatch)
QUANT_CASES = [
    ("gg_fwd_fc1_in", ROWS_GG, 2048, E4M3),
    ("gg_fwd_fc2_in", ROWS_GG, 768, E4M3),
    ("gg_bwd_fc2_gout", ROWS_GG, 2048, E5M2),
    ("gg_bwd_fc1_gout", ROWS_GG, 1536, E5M2),
    ("dense_fwd_qkv_in", TOK, 2048, E4M3),
    ("dense_fwd_proj_in", TOK, 4096, E4M3),
    ("dense_bwd_qkv_gout", TOK, 5120, E5M2),
    ("dense_bwd_proj_gout", TOK, 2048, E5M2),
]
TRANSPOSE_CASES = [
    ("fwd_a_t_qkv", TOK, 2048),
    ("fwd_a_t_proj", TOK, 4096),
    ("fwd_w_t_qkv", 5120, 2048),
    ("fwd_w_t_proj", 2048, 4096),
    ("bwd_gout_t_qkv", TOK, 5120),
    ("bwd_gout_t_proj", TOK, 2048),
]


def kernel_times(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    agg = collections.Counter()
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            name = re.sub(r"<.*", "", e.name).replace("void ", "").replace("primus_turbo::", "")
            agg[name] += e.device_time
    return {k: v / iters for k, v in agg.items()}  # us per call


def short(name):
    for k in ("amax_partial", "amax_scale", "pad_row", "transpose_2d", "elementwise", "reduce", "copy"):
        if k in name:
            return k
    return name[:40]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"

    print("== HBM reference (torch)")
    x = torch.randn(ROWS_GG, 2048, device=dev, dtype=torch.bfloat16)
    nbytes = x.numel() * 2
    t = kernel_times(lambda: x.clone(), args.iters)
    tt = sum(t.values())
    print(f"  clone 1.07GB bf16 : {tt:8.1f} us  {2 * nbytes / tt / 1e6:6.2f} TB/s (r+w)")
    t = kernel_times(lambda: x.abs().amax(), args.iters)
    rd = max(t.values())
    print(f"  amax  (largest kernel): {rd:8.1f} us  {nbytes / rd / 1e6:6.2f} TB/s (r)")
    del x

    print("== quantize_fp8_tensorwise (amax partial + scale + quant)")
    print(f"  {'case':22s} {'shape':>14s} {'amax us':>8s} {'TB/s':>5s} {'quant us':>8s} {'TB/s':>5s} {'scale us':>8s} {'total us':>8s}")
    for tag, m, k, dt in QUANT_CASES:
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        t = kernel_times(lambda: quantize_fp8_tensorwise_pad_impl(x, dt, k_align=128), args.iters)
        d = collections.Counter()
        for n, v in t.items():
            d[short(n)] += v
        n = m * k
        amax_bw = n * 2 / d["amax_partial"] / 1e6 if d["amax_partial"] else 0
        q_bw = n * 3 / d["pad_row"] / 1e6 if d["pad_row"] else 0
        print(
            f"  {tag:22s} {m:7d}x{k:<6d} {d['amax_partial']:8.1f} {amax_bw:5.2f} {d['pad_row']:8.1f} {q_bw:5.2f}"
            f" {d['amax_scale']:8.1f} {sum(t.values()):8.1f}"
        )
        del x

    print("== transpose_2d<uint8>")
    for tag, m, k in TRANSPOSE_CASES:
        x = torch.randint(0, 255, (m, k), device=dev, dtype=torch.uint8).view(E4M3)
        t = kernel_times(lambda: torch.ops.primus_turbo_cpp_extension.transpose_2d(x, 0, 1), args.iters)
        tt = sum(t.values())
        print(f"  {tag:22s} {m:7d}x{k:<6d} {tt:8.1f} us {2 * m * k / tt / 1e6:5.2f} TB/s")


if __name__ == "__main__":
    main()

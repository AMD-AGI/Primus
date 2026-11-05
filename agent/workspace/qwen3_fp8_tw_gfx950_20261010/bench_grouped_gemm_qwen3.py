"""Item 2 microbench: FlyDSL FP8 tensorwise grouped GEMM at Qwen3-30B-A3B EP8 shapes.

16 local experts, 262144 routed rows per rank per microbatch. Reports the GEMM
kernels only (fwd nt / dgrad nn / wgrad tn); quantization is excluded.
"""

import argparse
import collections
import re

import torch
from torch.profiler import ProfilerActivity, profile

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

G, M_TOTAL = 16, 262144
SHAPES = {"fc1": (1536, 2048), "fc2": (2048, 768)}  # (N, K) with trans_b


def group_lens_for(dist, gen):
    if dist == "even":
        return torch.full((G,), M_TOTAL // G, dtype=torch.int64)
    if dist == "uniform":  # each row routed uniformly at random
        idx = torch.randint(0, G, (M_TOTAL,), generator=gen)
        return torch.bincount(idx, minlength=G).to(torch.int64)
    if dist == "skew":  # bench_grouped_gemm_turbo's unbalanced generator
        d = 0.2 + 0.8 * torch.rand(G, generator=gen)
        lens = (d / d.sum() * M_TOTAL).to(torch.int64)
        lens[-1] += M_TOTAL - lens.sum()
        return lens
    if dist == "hot":  # one expert takes half the rows, two experts empty
        lens = torch.zeros(G, dtype=torch.int64)
        lens[0] = M_TOTAL // 2
        rest = M_TOTAL - lens[0]
        lens[1:14] = rest // 13
        lens[13] = rest - lens[1:13].sum()
        return lens
    raise ValueError(dist)


def classify(name):
    if "grouped_nt" in name:
        return "fwd"
    if "grouped_nn" in name:
        return "dgrad"
    if "wgrad" in name or "grouped_tn" in name:
        return "wgrad"
    return None


def run_case(name, dist, iters, gen):
    N, K = SHAPES[name]
    lens = group_lens_for(dist, gen).cuda()
    M = int(lens.sum())
    cfg = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(G, N, K, device="cuda", dtype=torch.bfloat16) * 0.02)
    w.main_grad = torch.zeros(G, N, K, device="cuda", dtype=torch.float32)
    w.grad_added_to_main_grad = False
    go = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    def step():
        out = turbo.ops.grouped_gemm_fp8(a, w, lens, trans_b=True, config=cfg, fuse_bgrad_accum_pattern="megatron")
        out.backward(go)
        a.grad = None

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
    t = collections.Counter()
    names = {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            c = classify(e.name)
            if c:
                t[c] += e.device_time
                names[c] = re.sub(r"\.kd$", "", e.name)
    flops = 2 * M * N * K
    res = {c: (t[c] / iters, flops / (t[c] / iters) / 1e6) for c in ("fwd", "dgrad", "wgrad")}
    tot = sum(v[0] for v in res.values())
    return M, res, 3 * flops / tot / 1e6, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dists", default="even,uniform,skew,hot")
    args = p.parse_args()
    gen = torch.Generator().manual_seed(1234)
    print(f"{'case':14s} {'M':>7s} | {'fwd us':>7s} {'TF/s':>5s} | {'dgrad us':>8s} {'TF/s':>5s} | {'wgrad us':>8s} {'TF/s':>5s} | step TF/s")
    for dist in args.dists.split(","):
        for name in SHAPES:
            M, r, step, names = run_case(name, dist, args.iters, gen)
            print(
                f"{name + '/' + dist:14s} {M:7d} | {r['fwd'][0]:7.1f} {r['fwd'][1]:5.0f} | {r['dgrad'][0]:8.1f} {r['dgrad'][1]:5.0f} |"
                f" {r['wgrad'][0]:8.1f} {r['wgrad'][1]:5.0f} | {step:6.0f}"
            )
    print("kernels:", names)


if __name__ == "__main__":
    main()

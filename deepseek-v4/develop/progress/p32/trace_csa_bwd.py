"""Trace CSA BWD with PyTorch profiler to see per-kernel time.

Run as: PYTHONPATH=. python deepseek-v4/develop/progress/p32/trace_csa_bwd.py
"""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton import (
    v4_csa_attention_bwd as _csa,
)


def _build_inputs(B=1, HQ=64, Sq=4096, D=512, P=1024, K_topk=512, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    k_local = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    v_local = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    pool = torch.randn(B, P, D, device=device, dtype=dtype) * 0.1
    topk = torch.randint(0, P, (B, Sq, K_topk), device=device, dtype=torch.int32)
    sink = torch.zeros(HQ, device=device, dtype=dtype)
    out = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    dout = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    lse = torch.randn(B, HQ, Sq, device=device, dtype=torch.float32)
    return q, k_local, v_local, pool, topk, sink, out, dout, lse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-out", default="/tmp/csa_bwd_trace.json")
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    device = "cuda"
    q, k_local, v_local, pool, topk, sink, out, dout, lse = _build_inputs(device=device)
    swa_window = 128
    scale = 1.0 / (q.shape[-1] ** 0.5)

    for _ in range(3):
        _csa._launch_v4_csa_attention_pool_bwd(
            q,
            k_local,
            v_local,
            pool,
            topk,
            out,
            dout,
            lse,
            sink=sink,
            swa_window=swa_window,
            scale=scale,
        )
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=1, active=args.iters, repeat=1),
        with_stack=False,
    ) as prof:
        for i in range(args.iters + 1):
            torch.cuda.synchronize()
            with torch.profiler.record_function(f"iter_{i}"):
                _csa._launch_v4_csa_attention_pool_bwd(
                    q,
                    k_local,
                    v_local,
                    pool,
                    topk,
                    out,
                    dout,
                    lse,
                    sink=sink,
                    swa_window=swa_window,
                    scale=scale,
                )
            torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(args.trace_out)
    print(f"Trace written to {args.trace_out}")


if __name__ == "__main__":
    main()

"""Quick CSA BWD component-level profiler (P32 debug aid).

Times each sub-step of ``_launch_v4_csa_attention_pool_bwd`` in
isolation by mocking out the others or by manually wrapping each
Triton kernel launch in CUDA events. Run on a single GPU.
"""

from __future__ import annotations

import argparse
import os

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton import (
    v4_csa_attention_bwd as _csa,
)


def _build_inputs(
    B=1, HQ=64, Sq=4096, D=512, P=1024, K_topk=512, swa_window=128, dtype=torch.bfloat16, device="cuda"
):
    torch.manual_seed(0)
    q = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    k_local = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    v_local = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    pool = torch.randn(B, P, D, device=device, dtype=dtype) * 0.1
    topk = torch.randint(0, P, (B, Sq, K_topk), device=device, dtype=torch.int32)
    sink = torch.zeros(HQ, device=device, dtype=dtype)

    # Build out / dout / lse / d via a fake forward (dimensions only)
    out = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    dout = torch.randn(B, HQ, Sq, D, device=device, dtype=dtype) * 0.1
    lse = torch.randn(B, HQ, Sq, device=device, dtype=torch.float32)
    return q, k_local, v_local, pool, topk, sink, out, dout, lse


def _time_block(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2], min(times), max(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args()

    device = "cuda"
    q, k_local, v_local, pool, topk, sink, out, dout, lse = _build_inputs(device=device)
    B, HQ, Sq, D = q.shape
    P = pool.shape[1]
    K_topk = topk.shape[-1]
    scale = 1.0 / (D**0.5)
    swa_window = 128

    # Full BWD baseline
    def full_bwd():
        return _csa._launch_v4_csa_attention_pool_bwd(
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

    med, mn, mx = _time_block(full_bwd, args.warmup, args.iters)
    print(f"[full BWD]                  median={med:.3f}ms min={mn:.3f}ms max={mx:.3f}ms")

    # Sort + searchsorted (inverse index) only
    def sort_only():
        MK = Sq * K_topk
        flat_topk = topk.contiguous().view(B, MK).to(torch.int32)
        sentinel = torch.full_like(flat_topk, P)
        masked = torch.where((flat_topk >= 0) & (flat_topk < P), flat_topk, sentinel)
        sorted_topk, perm = torch.sort(masked, dim=1, stable=True)
        queries = torch.arange(P + 1, device=q.device, dtype=torch.int32)
        queries = queries.unsqueeze(0).expand(B, -1).contiguous()
        bin_ptr = torch.searchsorted(sorted_topk, queries, right=False).to(torch.int32)
        return perm, bin_ptr

    med, mn, mx = _time_block(sort_only, args.warmup, args.iters)
    print(f"[sort + searchsorted only]  median={med:.3f}ms min={mn:.3f}ms max={mx:.3f}ms")

    # Allocate dpool_partial only (empty)
    def alloc_partial():
        return torch.empty((B, Sq, K_topk, D), device=q.device, dtype=torch.float32)

    med, mn, mx = _time_block(alloc_partial, args.warmup, args.iters)
    print(f"[alloc dpool_partial 4 GB]  median={med:.3f}ms min={mn:.3f}ms max={mx:.3f}ms")

    # dq sum
    a = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    b = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)

    def dq_sum():
        return a + b

    med, mn, mx = _time_block(dq_sum, args.warmup, args.iters)
    print(f"[dq fp32 sum 128 MB]        median={med:.3f}ms min={mn:.3f}ms max={mx:.3f}ms")

    # Test with gather path (atomic_add)
    os.environ["PRIMUS_V4_CSA_BWD_SEGREDUCE"] = "0"
    med, mn, mx = _time_block(full_bwd, args.warmup, args.iters)
    print(f"[gather BWD (atomic)]       median={med:.3f}ms min={mn:.3f}ms max={mx:.3f}ms")

    # Re-enable segreduce
    os.environ["PRIMUS_V4_CSA_BWD_SEGREDUCE"] = "1"


if __name__ == "__main__":
    main()

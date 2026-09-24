###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused AdamW step microbenchmark on Llama 3.1 8B parameter shapes.

Compares the optimizers Megatron can end up with on ROCm:
  te      TE FusedAdam (Megatron default; multi_tensor_apply, grid capped at 320)
  torch   torch.optim.AdamW(fused=True) (same 320 cap in ATen)
  triton  Primus TritonFusedAdam, swept over --grids (0 = auto)

All states are FP32 (4R3W, 28 B/param), matching Megatron's BF16 + FP32 main
params path. Optimizers run one at a time; peak memory is ~4 x 4 B x params.

  python3 benchmark/kernel/optimizer/bench_fused_adamw.py --peak-bw-tbs 20
  python3 benchmark/kernel/optimizer/bench_fused_adamw.py --layers 8 --grids 320,1024,2048,4096,0
  # Many small tensors (MoE-like): split every 2D weight into 64 row slices.
  python3 benchmark/kernel/optimizer/bench_fused_adamw.py --layers 8 --split 64 --grids 0
"""

import argparse
import gc

import torch

BYTES_PER_PARAM = 7 * 4
HPARAMS = dict(lr=7.2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def llama31_8b_shapes(n_layers, split=1):
    d, kv, ffn, vocab = 4096, 1024, 14336, 128256
    shapes = [(vocab, d), (d,)]
    for _ in range(n_layers):
        shapes += [(d,), (d, d), (kv, d), (kv, d), (d, d), (d,), (ffn, d), (d, ffn), (ffn, d)]
    shapes.append((vocab, d))
    return [piece for s in shapes for piece in ([(s[0] // split, s[1])] * split if len(s) == 2 else [s])]


def build_optimizer(name, params, grid):
    if name == "te":
        from transformer_engine.pytorch.optimizers import FusedAdam

        return FusedAdam(params, adam_w_mode=True, **HPARAMS)
    if name == "torch":
        return torch.optim.AdamW(params, fused=True, **HPARAMS)
    from primus.backends.megatron.core.optimizer.triton_fused_adam import (
        TritonFusedAdam,
    )

    return TritonFusedAdam(params, adam_w_mode=True, grid_size=grid, **HPARAMS)


def bench(name, shapes, grid, warmup, iters):
    params = [torch.nn.Parameter(torch.randn(s, device="cuda")) for s in shapes]
    for p in params:
        p.grad = torch.randn_like(p)
    opt = build_optimizer(name, params, grid)
    for _ in range(warmup):
        opt.step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        opt.step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    del opt, params
    gc.collect()
    torch.cuda.empty_cache()
    return sorted(times)[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32, help="transformer layers (32 = full 8B)")
    ap.add_argument("--split", type=int, default=1, help="split every 2D weight into N row slices")
    ap.add_argument("--impls", default="te,torch,triton")
    ap.add_argument("--grids", default="320,1024,2048,3712,0", help="Triton grid sizes, 0 = auto")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--peak-bw-tbs", type=float, default=None)
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    shapes = llama31_8b_shapes(args.layers, args.split)
    n = sum(torch.Size(s).numel() for s in shapes)
    traffic = n * BYTES_PER_PARAM
    print(
        f"device: {props.name} arch={getattr(props, 'gcnArchName', '?')} CUs={props.multi_processor_count}\n"
        f"tensors: {len(shapes)}  params: {n:,}  traffic: {traffic / 1e9:.1f} GB/step\n"
    )
    print(f"{'impl':>8} {'grid':>6} {'median ms':>10} {'TB/s':>7} {'%peak':>6} {'vs 1st':>6}")

    base = None
    for impl in args.impls.split(","):
        grids = [int(g) for g in args.grids.split(",")] if impl == "triton" else [None]
        for grid in grids:
            ms = bench(impl, shapes, grid, args.warmup, args.iters)
            tbs = traffic / (ms / 1e3) / 1e12
            base = base or ms
            peak = f"{100 * tbs / args.peak_bw_tbs:.1f}" if args.peak_bw_tbs else "-"
            label = "-" if grid is None else ("auto" if grid == 0 else str(grid))
            print(f"{impl:>8} {label:>6} {ms:>10.2f} {tbs:>7.2f} {peak:>6} {base / ms:>5.2f}x")


if __name__ == "__main__":
    main()

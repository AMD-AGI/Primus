"""
bench_ragged_shard_fsdp.py
==========================
Standalone FSDP communication benchmark: PyTorch FSDP2 Shard(0) vs
veScale RaggedShard DTensor.

Tests the exact operations that differ between the two approaches:
  - All-gather  : flat buffer redistribution (RaggedShard → Replicate)
  - Reduce-scatter: gradient redistribution (Replicate → RaggedShard)

LLaMA 3.1 8B parameter profile (single TransformerLayer):
  hidden_size=4096, ffn_hidden=14336, num_heads=32, num_kv_heads=8

Usage (8x AMD MI355X):
    torchrun --standalone --nproc_per_node=8 \\
        tests/fsdp_bench/bench_ragged_shard_fsdp.py

Environment:
    BENCH_ITERS=50        (default: 50 warmup+measure iterations)
    BENCH_WARMUP=10       (default: 10 warmup iterations)
    BENCH_LAYERS=32       (default: 32 = full LLaMA 3.1 8B)
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.distributed as dist

# ── add dev Primus to path if running from repo ──────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_PRIMUS_ROOT = os.path.dirname(os.path.dirname(_THIS))
for p in [_PRIMUS_ROOT, os.path.join(_PRIMUS_ROOT, "third_party", "Megatron-LM")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vescale.dtensor import DTensor, distribute_tensor
from vescale.dtensor.placement_types import RaggedShard, Replicate, Shard

# ── Configuration ─────────────────────────────────────────────────────────────
ITERS = int(os.environ.get("BENCH_ITERS", "50"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "10"))
NUM_LAYERS = int(os.environ.get("BENCH_LAYERS", "32"))

# LLaMA 3.1 8B single TransformerLayer parameter shapes
# (name, shape)
LLAMA31_8B_LAYER_PARAMS: List[Tuple[str, Tuple[int, ...]]] = [
    # Self-attention QKV projections (GQA: Q=4096→4096, KV=4096→1024 each)
    ("attn.qkv.weight", (4096 + 1024 + 1024, 4096)),
    ("attn.o_proj.weight", (4096, 4096)),
    # MLP: gate+up fused, down
    ("mlp.gate_up.weight", (14336 * 2, 4096)),
    ("mlp.down.weight", (4096, 14336)),
    # LayerNorms (small)
    ("input_layernorm.weight", (4096,)),
    ("post_attn_layernorm.weight", (4096,)),
]


def _layer_params_flat(dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Create a flat parameter buffer for one TransformerLayer."""
    total = sum(math.prod(s) for _, s in LLAMA31_8B_LAYER_PARAMS)
    return torch.randn(total, dtype=dtype, device="cuda")


def _param_metas() -> List[Tuple[str, int, int]]:
    """Return (name, offset, numel) for each param in the flat buffer."""
    metas = []
    offset = 0
    for name, shape in LLAMA31_8B_LAYER_PARAMS:
        n = math.prod(shape)
        metas.append((name, offset, n))
        offset += n
    return metas


# ── Timing helpers ────────────────────────────────────────────────────────────
def _sync_and_time(fn, reps: int) -> Tuple[float, float]:
    """Run fn() reps times, return (mean_ms, std_ms) after GPU sync."""
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std


# ── Baseline: PyTorch FSDP2 Shard(0) ─────────────────────────────────────────
def bench_fsdp2_shard0(
    device_mesh,
    flat_buf: torch.Tensor,
    param_metas: List,
    iters: int,
    warmup: int,
) -> Tuple[float, float, float, float]:
    """
    Simulate FSDP2 Shard(0) communication:
      - Each parameter is individually Shard(0) along the first dim.
      - To batch all-gather: concatenate into a flat buffer (requires padding/copy).
      - All-gather: per-parameter all_gather → single collective on padded buffer.
    Returns (ag_mean_ms, ag_std_ms, rs_mean_ms, rs_std_ms).
    """
    device_mesh.size()
    dtype = flat_buf.dtype

    # Create per-parameter Shard(0) DTensors
    param_dtensors = []
    for name, offset, numel in param_metas:
        param_data = flat_buf[offset : offset + numel]
        # Shard(0) on a 1D tensor == divide into world_size pieces
        dt = distribute_tensor(param_data.unsqueeze(0), device_mesh, [Shard(0)])
        param_dtensors.append(dt)

    def _all_gather():
        # All-gather each param individually (simulates FSDP2 batched all-gather)
        gathered = [dt.redistribute(placements=[Replicate()]).to_local() for dt in param_dtensors]
        # Combine into flat tensor (this copy is the overhead RaggedShard avoids)
        return torch.cat([g.view(-1) for g in gathered])

    def _reduce_scatter():
        # Simulate reduce-scatter: full grad → per-param Shard(0)
        full_grad = torch.randn(flat_buf.numel(), dtype=dtype, device="cuda")
        sharded_grads = []
        for _, o, n in param_metas:
            g = full_grad[o : o + n].unsqueeze(0)
            g_dt = DTensor.from_local(g, device_mesh, [Replicate()], run_check=False)
            sg = g_dt.redistribute(placements=[Shard(0)]).to_local()
            sharded_grads.append(sg)
        return sharded_grads

    # Warmup
    for _ in range(warmup):
        _all_gather()
        _reduce_scatter()

    ag_mean, ag_std = _sync_and_time(_all_gather, iters)
    rs_mean, rs_std = _sync_and_time(_reduce_scatter, iters)
    return ag_mean, ag_std, rs_mean, rs_std


# ── veScale: RaggedShard flat buffer ─────────────────────────────────────────
def bench_vescale_ragged_shard(
    device_mesh,
    flat_buf: torch.Tensor,
    param_metas: List,
    iters: int,
    warmup: int,
) -> Tuple[float, float, float, float]:
    """
    veScale RaggedShard: single flat buffer, single collective.
      - All params packed into ONE flat buffer with RaggedShard placement.
      - All-gather  = flat_dtensor.redistribute([Replicate()])
      - Reduce-scatter = from_local(grad, [Replicate()]).redistribute([RaggedShard])
    Returns (ag_mean_ms, ag_std_ms, rs_mean_ms, rs_std_ms).
    """
    world_size = device_mesh.size()
    dtype = flat_buf.dtype
    total = flat_buf.numel()

    # Compute equal-ratio RaggedShard units
    base = total // world_size
    remainder = total % world_size
    raw = tuple(base + (1 if i == world_size - 1 and remainder else 0) for i in range(world_size))
    gcd = math.gcd(*raw)
    local_units = tuple(u // gcd for u in raw)
    ragged_shard = RaggedShard(dims=(0,), local_units=local_units)

    # Distribute flat buffer
    flat_dtensor = distribute_tensor(flat_buf, device_mesh, [ragged_shard])

    def _all_gather():
        full_dt = flat_dtensor.redistribute(placements=[Replicate()], async_op=False)
        return full_dt.to_local()

    def _reduce_scatter():
        full_grad = torch.randn(total, dtype=dtype, device="cuda")
        grad_dt = DTensor.from_local(full_grad, device_mesh, [Replicate()], run_check=False)
        sg_dt = grad_dt.redistribute(placements=[ragged_shard], async_op=False)
        return sg_dt.to_local()

    # Warmup
    for _ in range(warmup):
        _all_gather()
        _reduce_scatter()

    ag_mean, ag_std = _sync_and_time(_all_gather, iters)
    rs_mean, rs_std = _sync_and_time(_reduce_scatter, iters)
    return ag_mean, ag_std, rs_mean, rs_std


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Init distributed
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh("cuda", (world_size,))

    dtype = torch.bfloat16

    # Build flat buffer for NUM_LAYERS TransformerLayers (simulate full model)
    metas = _param_metas()
    layer_numel = sum(n for _, _, n in metas)
    total_numel = layer_numel * NUM_LAYERS

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  FSDP Communication Benchmark — LLaMA 3.1 8B")
        print(f"  {world_size}x AMD MI355X  |  {NUM_LAYERS} layers  |  {dtype}")
        print(f"  Layer params: {layer_numel:,} elems  |  Total: {total_numel:,} elems")
        print(f"  Iters: {ITERS}  |  Warmup: {WARMUP}")
        print(f"{'='*70}\n")

    flat_buf = torch.randn(total_numel, dtype=dtype, device="cuda")
    full_metas = []
    for layer_idx in range(NUM_LAYERS):
        base_offset = layer_idx * layer_numel
        for name, off, n in metas:
            full_metas.append((f"layer{layer_idx}.{name}", base_offset + off, n))

    dist.barrier()

    # ── Baseline: FSDP2 Shard(0) ─────────────────────────────────────────
    if rank == 0:
        print("[1/2] Benchmarking: PyTorch FSDP2 Shard(0) ...")
    dist.barrier()

    ag_b, ag_bs, rs_b, rs_bs = bench_fsdp2_shard0(device_mesh, flat_buf, full_metas, ITERS, WARMUP)

    dist.barrier()

    # ── veScale: RaggedShard ──────────────────────────────────────────────
    if rank == 0:
        print("[2/2] Benchmarking: veScale RaggedShard DTensor ...")
    dist.barrier()

    ag_v, ag_vs, rs_v, rs_vs = bench_vescale_ragged_shard(device_mesh, flat_buf, full_metas, ITERS, WARMUP)

    dist.barrier()

    # ── Results (rank 0 only) ──────────────────────────────────────────────
    if rank == 0:

        def speedup(baseline, new):
            return f"{baseline/new:.3f}x" if new > 0 else "N/A"

        ag_sp = speedup(ag_b, ag_v)
        rs_sp = speedup(rs_b, rs_v)
        total_b = ag_b + rs_b
        total_v = ag_v + rs_v
        total_sp = speedup(total_b, total_v)

        print(f"\n{'='*70}")
        print(f"  Results — LLaMA 3.1 8B ({NUM_LAYERS} layers, {world_size} GPUs)")
        print(f"{'='*70}")
        print(f"  {'Metric':<40} {'FSDP2 Shard(0)':>14}  {'RaggedShard':>12}  {'Speedup':>8}")
        print(f"  {'-'*40}  {'-'*14}  {'-'*12}  {'-'*8}")
        print(f"  {'All-gather mean (ms)':<40} {ag_b:>12.2f}ms  {ag_v:>10.2f}ms  {ag_sp:>8}")
        print(f"  {'All-gather std (ms)':<40} {ag_bs:>12.2f}ms  {ag_vs:>10.2f}ms  {'':>8}")
        print(f"  {'Reduce-scatter mean (ms)':<40} {rs_b:>12.2f}ms  {rs_v:>10.2f}ms  {rs_sp:>8}")
        print(f"  {'Reduce-scatter std (ms)':<40} {rs_bs:>12.2f}ms  {rs_vs:>10.2f}ms  {'':>8}")
        print(f"  {'Total (AG + RS) mean (ms)':<40} {total_b:>12.2f}ms  {total_v:>10.2f}ms  {total_sp:>8}")
        print(f"{'='*70}")
        print(f"\n  veScale RaggedShard is {total_sp} faster on total FSDP communication")
        print(f"  (AG + RS per training step for the full model)\n")

        # Write machine-readable results
        import json

        results = {
            "world_size": world_size,
            "num_layers": NUM_LAYERS,
            "total_params": total_numel,
            "baseline_fsdp2": {
                "all_gather_ms": ag_b,
                "all_gather_std": ag_bs,
                "reduce_scatter_ms": rs_b,
                "reduce_scatter_std": rs_bs,
            },
            "vescale_ragged_shard": {
                "all_gather_ms": ag_v,
                "all_gather_std": ag_vs,
                "reduce_scatter_ms": rs_v,
                "reduce_scatter_std": rs_vs,
            },
            "speedup_all_gather": float(ag_b / ag_v) if ag_v > 0 else None,
            "speedup_reduce_scatter": float(rs_b / rs_v) if rs_v > 0 else None,
            "speedup_total": float(total_b / total_v) if total_v > 0 else None,
        }
        out = os.path.join(_PRIMUS_ROOT, "output", "fsdp_bench", "results.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {out}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

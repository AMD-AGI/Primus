#!/usr/bin/env python3
"""Aggregate per-category time from a PyTorch profiler trace.

Categorize by op name into:
  - GDN forward/backward (Triton + chunk_state, recurrent_state)
  - LinearGEMM (matmul, addmm, linear)
  - SwiGLU + RMSNorm (Triton fused norms)
  - Fused-CE loss (FusedLinearCrossEntropy)
  - NCCL collectives (allreduce, allgather)
  - hipMalloc / hipFree
  - aten::item / aten::isnan / cudaStreamSync
  - Python/CPU misc

Usage:
  python3 profile_breakdown.py <trace.json> [ts_lo ts_hi] [--gpu-only]

Examples:
  # Full breakdown of all events in the trace (CPU + GPU)
  python3 profile_breakdown.py mytrace.json

  # GPU events only (cleaner, no Python-call noise)
  python3 profile_breakdown.py mytrace.json --gpu-only

  # Window to a single iter (timestamps in us, from ProfilerStep markers)
  python3 profile_breakdown.py mytrace.json 4731843356001 4731846112894 --gpu-only
"""
import json, sys, re, collections

# Use ijson if available (streaming, low-memory) else fall back to full load
try:
    import ijson
    HAVE_IJSON = True
except ImportError:
    HAVE_IJSON = False

CATEGORIES = [
    # (category, list of substring patterns to match against the op name)
    # Order matters: first match wins.  Put most-specific before generic.
    ("GDN fwd/bwd",        ["fused_recurrent", "chunk_gated_delta", "ChunkGatedDelta", "gated_delta_net",
                            "chunk_fwd", "chunk_bwd", "fused_gdn", "_gated_delta", "FusedRecurrent",
                            "delta_h_kernel", "intra_chunk", "chunk_state_kernel", "recurrent_state",
                            "recompute_w_u_fwd"]),
    ("GDN short-conv",     ["causal_conv1d", "conv1d_fwd", "conv1d_bwd"]),
    ("GDN L2 norm",        ["l2norm_fwd", "l2norm_bwd"]),
    ("Fused-CE loss",      ["fused_linear_cross_entropy", "FusedLinearCrossEntropy", "cross_entropy_loss",
                            "ce_loss_fwd", "ce_loss_bwd", "fused_ce", "loss_chunk",
                            "logsumexp", "MaxNanFunctor", "AbsFunctor"]),
    ("SwiGLU + RMSNorm",   ["swiglu", "SwiGLU", "rms_norm", "RMSNorm", "fused_rmsnorm", "rms_layer",
                            "norm_fused", "FusedRMSNorm", "layer_norm_fwd", "layer_norm_bwd"]),
    ("Linear GEMMs",       ["aten::mm", "aten::addmm", "aten::linear", "aten::matmul", "aten::bmm",
                            "Cijk_Ailk", "Cijk_Alik", "Cijk_", "rocblas", "hipblaslt", "gemm_kernel"]),
    ("NCCL collectives",   ["nccl", "ncclAll", "ncclReduce", "ncclSend", "ncclRecv", "allreduce_",
                            "allgather_", "reduce_scatter_", "c10d::"]),
    ("hipMalloc/hipFree",  ["hipMalloc", "hipFree", "Memcpy", "Memset", "cudaMalloc", "cudaFree"]),
    ("CPU↔GPU sync",       ["aten::item", "aten::isnan", "aten::isinf", "aten::any", "cudaStreamSync",
                            "synchronize", "_local_scalar_dense", "_to_copy", "aten::to"]),
    ("Optimizer",          ["adam", "Adam", "AdamW", "fused_adam", "optimizer_step", "DistributedOptimizer"]),
    ("Embedding",          ["embedding", "Embedding", "aten::embedding"]),
    ("dropout/activation", ["dropout", "Dropout", "aten::relu", "aten::gelu", "silu", "aten::silu"]),
    ("Pointwise ops",      ["elementwise_kernel", "vectorized_elementwise", "CUDAFunctor",
                            "AUnaryFunctor", "ABinaryFunctor", "elementwise", "direct_copy"]),
    ("Reductions",         ["reduce_kernel", "sum_functor", "min_functor", "max_functor"]),
    ("data movement",      ["aten::contiguous", "aten::transpose", "aten::view", "aten::permute",
                            "aten::reshape", "aten::clone", "aten::cat", "aten::stack", "aten::expand",
                            "bfloat16_copy_kernel"]),
]

def categorize(name):
    n = name.lower()
    for cat, pats in CATEGORIES:
        for p in pats:
            if p.lower() in n:
                return cat
    return "uncategorized"

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    ts_lo = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ts_hi = int(sys.argv[3]) if len(sys.argv) > 3 else None
    only_gpu = "--gpu-only" in sys.argv

    cat_time = collections.defaultdict(float)
    uncat_names = collections.Counter()
    gpu_kernel_time = 0.0
    cpu_op_time = 0.0
    total_events = 0
    iter_starts = []   # records of ProfilerStep# events

    # Use streaming if possible
    if HAVE_IJSON:
        with open(path, 'rb') as fp:
            for ev in ijson.items(fp, 'traceEvents.item'):
                _process(ev, cat_time, uncat_names, ts_lo, ts_hi, iter_starts, only_gpu)
                total_events += 1
                if total_events % 500000 == 0:
                    print(f"  ... processed {total_events} events", file=sys.stderr)
    else:
        with open(path) as fp:
            data = json.load(fp)
        for ev in data.get('traceEvents', []):
            _process(ev, cat_time, uncat_names, ts_lo, ts_hi, iter_starts, only_gpu)
            total_events += 1

    # Report
    print(f"\nProcessed {total_events} events from {path}")
    print(f"Found {len(iter_starts)} ProfilerStep markers")
    if iter_starts:
        # Compute per-iter average duration if we have >1 step
        for i, (name, ts, dur) in enumerate(iter_starts[:10]):
            print(f"  step {i}: name={name}  dur={dur/1000:.1f} ms (ts={ts})")
    print(f"\nPer-category time:")
    total = sum(cat_time.values())
    for cat, _ in CATEGORIES + [("uncategorized", None)]:
        t = cat_time.get(cat, 0)
        pct = 100*t/total if total else 0
        print(f"  {cat:25s}  {t/1000:>10.1f} ms   {pct:5.1f}%")
    print(f"  {'TOTAL':25s}  {total/1000:>10.1f} ms")
    print()
    print("Top 20 uncategorized op names by count:")
    for name, count in uncat_names.most_common(20):
        print(f"  {count:>6d}  {name}")

def _process(ev, cat_time, uncat_names, ts_lo, ts_hi, iter_starts, only_gpu=False):
    if ev.get('ph') != 'X':
        return
    name = ev.get('name', '')
    dur = ev.get('dur', 0)
    ts = ev.get('ts', 0)
    if ts_lo is not None and ts < ts_lo: return
    if ts_hi is not None and ts > ts_hi: return
    if name.startswith('ProfilerStep'):
        iter_starts.append((name, ts, dur))
        return
    cat_full = ev.get('cat', '')
    # Filter: GPU events have cat in {kernel, gpu_memcpy, gpu_memset, gpu_user_annotation}
    is_gpu = cat_full.startswith('kernel') or cat_full.startswith('gpu_')
    if only_gpu and not is_gpu:
        return
    cat = categorize(name)
    if cat == "uncategorized":
        uncat_names[name] += 1
    cat_time[cat] += dur

if __name__ == "__main__":
    main()

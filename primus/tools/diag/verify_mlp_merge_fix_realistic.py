"""Realistic-scale sanity check for the mlp.py fix under a true 70B load.

Differs from verify_mlp_merge_fix.py: this version preallocates ~80 GB of
"context" tensors (proxy for attention / embed / optimizer state) and only
then runs the 160 sh_ten_merge_fn calls, so we can see whether the peak and
fragmentation actually push against the GPU limit.
"""

from __future__ import annotations

import gc
import time

import torch

NUM_LAYERS = 80
SHARDS_PER_MERGE = 8
PER_SHARD_SHAPE = (3584, 8192)
DTYPE = torch.bfloat16
DEVICE = "cuda"
CONTEXT_TENSOR_SHAPE = (6144, 8192)  # ~96 MB bf16 each
CONTEXT_TENSOR_COUNT = 800  # ~75 GB total


def merge_old(sub_state_dict):
    """Original mlp.py:430-442 behavior: cat then keep sub_state_dict refs alive."""
    with torch.no_grad():
        try:
            return torch.cat(sub_state_dict)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            merged = torch.cat([t.cpu() for t in sub_state_dict])
            gc.collect()
            torch.cuda.empty_cache()
            return merged


def merge_new(sub_state_dict):
    """Post-fix behavior: del sub_state_dict elements once cat succeeds."""
    with torch.no_grad():
        try:
            merged = torch.cat(sub_state_dict)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            cpu_tensors = [t.cpu() for t in sub_state_dict]
            del sub_state_dict[:]
            gc.collect()
            torch.cuda.empty_cache()
            return torch.cat(cpu_tensors)
        else:
            del sub_state_dict[:]
            return merged


def run(merge_fn, label, context_tensors):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    merged_keep = []
    oom_at = None
    t0 = time.time()

    print(
        f"  [{label}] context preloaded: "
        f"alloc={torch.cuda.memory_allocated() / 2**30:.2f} GB, "
        f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GB"
    )

    try:
        for layer in range(NUM_LAYERS):
            for which in ("fc1", "fc2"):
                shards = [
                    torch.empty(PER_SHARD_SHAPE, dtype=DTYPE, device=DEVICE) for _ in range(SHARDS_PER_MERGE)
                ]
                merged_keep.append(merge_fn(shards))
                if (layer + 1) % 20 == 0 and which == "fc2":
                    a = torch.cuda.memory_allocated() / 2**30
                    r = torch.cuda.memory_reserved() / 2**30
                    print(
                        f"  [{label}] layer {layer + 1:3d}/{NUM_LAYERS} "
                        f"alloc={a:7.2f} GB  reserved={r:7.2f} GB  "
                        f"frag={r - a:.2f} GB"
                    )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        oom_at = (len(merged_keep), str(e)[:140])
        print(f"  [{label}] OOM at merge #{oom_at[0]}: {oom_at[1]}")

    elapsed = time.time() - t0
    pa = torch.cuda.max_memory_allocated() / 2**30
    pr = torch.cuda.max_memory_reserved() / 2**30

    n_done = len(merged_keep)
    del merged_keep
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "done": n_done,
        "elapsed_s": elapsed,
        "peak_alloc_gb": pa,
        "peak_reserved_gb": pr,
        "frag_gb": pr - pa,
        "oom": oom_at,
    }


def main():
    assert torch.cuda.is_available()
    print(
        f"Simulating realistic 70B load: {CONTEXT_TENSOR_COUNT} context tensors "
        f"({CONTEXT_TENSOR_COUNT * CONTEXT_TENSOR_SHAPE[0] * CONTEXT_TENSOR_SHAPE[1] * 2 / 2**30:.0f} GB) "
        f"+ {NUM_LAYERS} layers × 2 MLPs × {SHARDS_PER_MERGE} shards merge"
    )
    print()

    for label, fn in [("OLD (no del)", merge_old), ("NEW (del sub_state_dict)", merge_new)]:
        print(f"===== {label} =====")
        context = [
            torch.empty(CONTEXT_TENSOR_SHAPE, dtype=DTYPE, device=DEVICE) for _ in range(CONTEXT_TENSOR_COUNT)
        ]
        r = run(fn, label, context)
        del context
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"  RESULT: done={r['done']}/{NUM_LAYERS * 2}, elapsed={r['elapsed_s']:.2f}s, "
            f"peak_alloc={r['peak_alloc_gb']:.2f} GB, peak_reserved={r['peak_reserved_gb']:.2f} GB, "
            f"frag={r['frag_gb']:.2f} GB"
        )
        if r["oom"]:
            print(f"  OOM: at merge #{r['oom'][0]}: {r['oom'][1]}")
        print()


if __name__ == "__main__":
    main()

"""Sanity check for mlp.py:sh_ten_merge_fn fragmentation fix.

Simulates Llama2-70B's 80 layers x (fc1 + fc2) = 160 sharded MLP weight
cat-merges and compares the HIP allocator's peak / fragmentation footprint
before vs after the fix.

Usage (single GPU is sufficient -- only single-rank allocator behavior is
validated)::

    python -m primus.tools.diag.verify_mlp_merge_fix

Pass criteria:
    * "after-fix peak allocated"  noticeably lower than "before-fix peak allocated"
    * "after-fix max reserved"    noticeably lower than "before-fix max reserved"
    * No ``RuntimeError`` (the pre-fix path frequently OOMs around merge 80-120)
"""

from __future__ import annotations

import gc
import os
import time

import torch

# 70B Llama2 SwiGLU MLP shape: fc1 weight = (28672, 8192) bf16 ~ 470 MB.
# At load time it is normally split into 8 shards of ~58 MB each;
# fc2 weight = (8192, 28672) is comparable in size.
NUM_LAYERS = 80
SHARDS_PER_MERGE = 8
PER_SHARD_SHAPE = (28672 // SHARDS_PER_MERGE, 8192)  # (3584, 8192) bf16 ≈ 58.7 MB
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _build_shards() -> list[torch.Tensor]:
    """Simulate one dist_ckpt sub_state_dict load (list[Tensor] on GPU)."""
    return [torch.empty(PER_SHARD_SHAPE, dtype=DTYPE, device=DEVICE) for _ in range(SHARDS_PER_MERGE)]


def merge_old(sub_state_dict: list[torch.Tensor]) -> torch.Tensor:
    """Original mlp.py:430-442 behavior: cat then keep sub_state_dict refs alive."""
    with torch.no_grad():
        try:
            return torch.cat(sub_state_dict)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            merged = torch.cat([t.cpu() for t in sub_state_dict])
            gc.collect()
            torch.cuda.empty_cache()
            return merged


def merge_new(sub_state_dict: list[torch.Tensor]) -> torch.Tensor:
    """Post-fix behavior: del sub_state_dict on success; on OOM fallback also
    del + empty_cache before retrying on CPU."""
    with torch.no_grad():
        force_cpu = os.environ.get("MEGATRON_CKPT_CPU_MERGE", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        if force_cpu:
            cpu_tensors = [t.cpu() for t in sub_state_dict]
            del sub_state_dict[:]
            gc.collect()
            torch.cuda.empty_cache()
            merged = torch.cat(cpu_tensors)
            del cpu_tensors
            return merged
        try:
            merged = torch.cat(sub_state_dict)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            cpu_tensors = [t.cpu() for t in sub_state_dict]
            del sub_state_dict[:]
            gc.collect()
            torch.cuda.empty_cache()
            merged = torch.cat(cpu_tensors)
            del cpu_tensors
            return merged
        else:
            del sub_state_dict[:]
            return merged


def run(merge_fn, label: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    merged_holder: list[torch.Tensor] = []
    t0 = time.time()
    oom_at = None
    try:
        for layer in range(NUM_LAYERS):
            for which in ("fc1", "fc2"):
                shards = _build_shards()
                merged = merge_fn(shards)
                merged_holder.append(merged)
                if (layer + 1) % 20 == 0 and which == "fc2":
                    alloc = torch.cuda.memory_allocated() / 2**30
                    reserved = torch.cuda.memory_reserved() / 2**30
                    print(
                        f"  [{label}] layer {layer + 1:3d} allocated={alloc:7.2f} GB"
                        f" reserved={reserved:7.2f} GB"
                    )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        oom_at = (len(merged_holder), str(e)[:120])
    elapsed = time.time() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    n_done = len(merged_holder)
    del merged_holder
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "merges_done": n_done,
        "elapsed_s": elapsed,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
        "fragmentation_gb": peak_reserved - peak_alloc,
        "oom": oom_at,
    }


def fmt(r: dict) -> str:
    lines = [
        f"  merges_done      = {r['merges_done']}/{NUM_LAYERS * 2}",
        f"  elapsed          = {r['elapsed_s']:.2f} s",
        f"  peak allocated   = {r['peak_alloc_gb']:.2f} GB",
        f"  peak reserved    = {r['peak_reserved_gb']:.2f} GB",
        f"  fragmentation    = {r['fragmentation_gb']:.2f} GB",
    ]
    if r["oom"]:
        lines.append(f"  OOM at merge #{r['oom'][0]}: {r['oom'][1]}")
    return "\n".join(lines)


def main():
    assert torch.cuda.is_available(), "GPU required"
    print(
        f"Simulating {NUM_LAYERS} layers × 2 MLPs × {SHARDS_PER_MERGE} shards"
        f" = {NUM_LAYERS * 2 * SHARDS_PER_MERGE} bf16 tensors"
    )
    print(
        f"Per-shard shape = {PER_SHARD_SHAPE},"
        f" merged tensor ≈ {SHARDS_PER_MERGE * PER_SHARD_SHAPE[0] * PER_SHARD_SHAPE[1] * 2 / 2**20:.0f} MB\n"
    )

    print("===== run: BEFORE-FIX (no del sub_state_dict) =====")
    old_r = run(merge_old, "old")
    print(fmt(old_r))
    print()

    print("===== run: AFTER-FIX (del sub_state_dict + escape hatch) =====")
    new_r = run(merge_new, "new")
    print(fmt(new_r))
    print()

    print("===== run: FORCE CPU (MEGATRON_CKPT_CPU_MERGE=1) =====")
    os.environ["MEGATRON_CKPT_CPU_MERGE"] = "1"
    cpu_r = run(merge_new, "cpu")
    print(fmt(cpu_r))
    os.environ.pop("MEGATRON_CKPT_CPU_MERGE", None)
    print()

    saved_alloc = old_r["peak_alloc_gb"] - new_r["peak_alloc_gb"]
    saved_frag = old_r["fragmentation_gb"] - new_r["fragmentation_gb"]
    print("===== SUMMARY =====")
    print(
        f"  peak alloc reduced by  : {saved_alloc:+.2f} GB"
        f"  ({saved_alloc / max(old_r['peak_alloc_gb'], 1e-9) * 100:+.1f} %)"
    )
    print(
        f"  fragmentation reduced  : {saved_frag:+.2f} GB"
        f"  ({saved_frag / max(old_r['fragmentation_gb'], 1e-9) * 100:+.1f} %)"
    )
    print(f"  old OOM                : {old_r['oom']}")
    print(f"  new OOM                : {new_r['oom']}")
    print(f"  force-cpu OOM          : {cpu_r['oom']}")


if __name__ == "__main__":
    main()

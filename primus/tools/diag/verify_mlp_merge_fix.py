"""Sanity check for mlp.py:sh_ten_merge_fn fragmentation fix.

模拟 Llama2-70B 80 层 × (fc1+fc2) = 160 次 sharded MLP weight cat-merge，
对比修复前/后 HIP allocator 的峰值显存与碎片占用。

跑法（单卡足够，因为只验证单 rank 的 allocator 行为）：

    docker exec -it sft_primus_0512 bash -lc \
        'cd /home/botahu/sft_primus_0507/Primus && \
         export PYTORCH_ALLOC_CONF=expandable_segments:True && \
         python examples/megatron/configs/MI355X/verify_mlp_merge_fix.py'

判定：
    * "after-fix peak allocated"  显著低于 "before-fix peak allocated"
    * "after-fix max reserved"    显著低于 "before-fix max reserved"
    * 没有 ``RuntimeError``      （修复前在 80~120 次 cat 时常 OOM）
"""

from __future__ import annotations

import gc
import os
import time

import torch


# 70B Llama2 SwiGLU MLP shape: fc1 weight = (28672, 8192) bf16 ≈ 470 MB
# 加载时通常拆成 8 个 chunks，每个 ~58 MB；fc2 weight = (8192, 28672) 同量级
NUM_LAYERS = 80
SHARDS_PER_MERGE = 8
PER_SHARD_SHAPE = (28672 // SHARDS_PER_MERGE, 8192)  # (3584, 8192) bf16 ≈ 58.7 MB
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _build_shards() -> list[torch.Tensor]:
    """模拟一次 dist_ckpt 加载得到的 sub_state_dict (list[Tensor on GPU])."""
    return [
        torch.empty(PER_SHARD_SHAPE, dtype=DTYPE, device=DEVICE)
        for _ in range(SHARDS_PER_MERGE)
    ]


def merge_old(sub_state_dict: list[torch.Tensor]) -> torch.Tensor:
    """原始 mlp.py:430-442 行为：cat 后不释放 sub_state_dict 引用。"""
    with torch.no_grad():
        try:
            return torch.cat(sub_state_dict)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            merged = torch.cat([t.cpu() for t in sub_state_dict])
            gc.collect()
            torch.cuda.empty_cache()
            return merged


def merge_new(sub_state_dict: list[torch.Tensor]) -> torch.Tensor:
    """修复后行为：cat 成功立即 del list 元素，失败路径也 del + empty_cache。"""
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
    assert torch.cuda.is_available(), "需要 GPU"
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
    print(f"  peak alloc reduced by  : {saved_alloc:+.2f} GB"
          f"  ({saved_alloc / max(old_r['peak_alloc_gb'], 1e-9) * 100:+.1f} %)")
    print(f"  fragmentation reduced  : {saved_frag:+.2f} GB"
          f"  ({saved_frag / max(old_r['fragmentation_gb'], 1e-9) * 100:+.1f} %)")
    print(f"  old OOM                : {old_r['oom']}")
    print(f"  new OOM                : {new_r['oom']}")
    print(f"  force-cpu OOM          : {cpu_r['oom']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Component breakdown of GPU time in a stitched dlrm_v4 trace, and A/B two of them.

Every GPU event (kernel / memcpy / memset) is attributed to:

* a **module** — the innermost recognized annotation on its CPU launch stack
  (``## stu_preprocess_and_attention ##``, ``hstu_input_preprocessor``,
  ``[lookup]_[_embedding_collection]_...``, autograd node names, ...);
* an **op kind** — attention, GEMM, HSTU Triton pointwise, aten elementwise,
  reduction, collective, memcpy, ...;
* a **phase** — forward / backward / optimizer / other, from whether the launch
  stack passes through the autograd engine.

Attribution is exact rather than name-heuristic: GPU events carry a
``correlation`` id shared with the ``cuda_runtime`` launch call that issued them,
and that launch call sits inside the CPU op/annotation tree, so walking the
enclosing intervals recovers the real caller. Correlation ids are only unique
within a rank, so everything is scoped per rank (``pid // 1_000_000``, the
stride ``stitch_traces.py`` uses).

Reported time is **summed kernel time per rank per step** ("work"), which can
exceed step wall time because compute, collectives and the pipeline's copy
streams overlap. Occupancy is reported separately as the per-rank union of GPU
busy intervals.

Usage
-----
Single trace::

    python scripts/analyze_component_breakdown.py traces/mi350x/trace_step52_pipeline.json.gz

Compare two (second is the baseline, first is "A")::

    python scripts/analyze_component_breakdown.py \\
        traces/b200/trace_step52_pipeline.json.gz \\
        traces/mi350x/trace_step52_pipeline.json.gz \\
        --labels B200 MI350X
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

_PID_STRIDE = 1_000_000
_GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
_CPU_TREE_CATS = {"cpu_op", "user_annotation"}
_LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}

# ---------------------------------------------------------------- op kinds
# Matched against the lowercased kernel name, first hit wins. ROCm decorates
# names with ".kd" / "[clone .kd]"; both sides are matched by substring so the
# same rule covers CUDA and HIP.
_OP_KIND_RULES = [
    ("attention fwd",      ["_hstu_attn_fwd", "fwd_sm100", "fmhafwd"]),
    ("attention bwd",      ["_hstu_attn_bwd", "bwd_sm100", "bwd_preprocess",
                            "bwd_postprocess", "fmhabwd"]),
    ("collective",         ["ncclkernel", "nccldevkernel", "mscclkernel", "rccl"]),
    ("GEMM",               ["nvjet", "cutlass", "cijk_", "rocblas", "splitkreduce",
                            "gemm"]),
    ("HSTU Triton p'wise", ["_weighted_layer_norm", "_ln_mul_dropout",
                            "_generate_random_mask", "_add_timestamp_position",
                            "_add_embeddings", "_jagged", "concat_2d_jagged",
                            "split_2d_jagged", "_norm_mul_dropout"]),
    ("embedding scatter",  ["indexfunclargeindex", "index_add", "embedding_dense",
                            "split_embedding", "batched_embedding"]),
    ("gather / index",     ["vectorized_gather", "gather_kernel", "index_elementwise",
                            "index_select", "indexselect"]),
    ("sort / scan",        ["radixsort", "radix_sort", "devicescan", "scan_kernel",
                            "reduce_by_key", "runs_config", "lookback_scan",
                            "unique", "encode::policy"]),
    ("reduction",          ["reduce_kernel", "devicereduce"]),
    ("cat / permute",      ["catarraybatchedcopy", "permute_2d", "invert_permute",
                            "linearize_index", "_cat_"]),
    ("copy / fill",        ["copybuffer", "fillbuffer", "direct_copy", "copy_kernel",
                            "fillfunctor", "_copy_kernel"]),
    ("elementwise",        ["elementwise_kernel", "silu", "dropout_kernel",
                            "masked_scale", "cudafunctor", "multi_tensor_apply"]),
    ("memcpy",             ["memcpy"]),
    ("memset",             ["memset"]),
]

# ---------------------------------------------------------------- modules
# Matched against launch-stack frame names, innermost frame first, first hit
# wins. Forward frames are record_function annotations; backward frames are
# autograd node names, so both spellings map to one module.
_MODULE_RULES = [
    # --- dense HSTU path ---
    ("HSTU attention block",   ["## stu_preprocess_and_attention ##",
                                "HSTUPreprocessAndAttentionFunction"]),
    ("HSTU output projection", ["## stu_compute_output ##",
                                "HSTUComputeOutputFunction"]),
    ("HSTU input preprocess",  ["hstu_input_preprocessor", "hstu_positional_encoder",
                                "_AddTimestampPositionEmbeddingsFunction",
                                "_Concat2DJaggedFunction", "_Split2DJaggedFunction"]),
    ("HSTU output postproc",   ["hstu_output_postprocessor"]),
    ("multitask head (MLP)",   ["## multitask_module ##", "AddmmBackward0",
                                "SwishLayerNormFunctionBackward",
                                "LayerNormFunctionBackward"]),
    # --- sparse / embedding path ---
    ("emb input dist (a2a)",   ["## dedup_ec_indices ##", "[splits_dist]_",
                                "[input_dist]_", "KJTAllToAll", "KJTList",
                                "FusedKJTListSplits", "SplitsAllToAll",
                                "## start_sparse_data_dist",
                                "## wait_sparse_data_dist"]),
    ("emb lookup (TBE)",       ["[lookup]_", "SplitNoBagLookupFunction",
                                "split_embedding"]),
    ("emb output dist (a2a)",  ["[output_dist", "SequenceEmbeddingsAwaitable",
                                "All2All_Seq", "## all2all_data",
                                "## decoder", "## encoder"]),
    ("emb jagged regather",    ["## construct_jagged_tensors ##",
                                "IndexSelectBackward0", "index_select_backward",
                                "JaggedToPaddedDenseOp",
                                "EmbeddingCollectionAwaitable"]),
    # --- everything else ---
    ("dense grad all-reduce",  ["nccl:all_reduce", "Reducer"]),
    ("optimizer",              ["Optimizer.step", "AccumulateGrad"]),
    ("H2D batch copy",         ["## copy_batch_to_gpu"]),
    ("loss / metrics",         ["## gauc"]),
    ("autograd glue",          ["ToCopyBackward0", "SplitWithSizesBackward0",
                                "CatBackward0", "NativeDropoutBackward0",
                                "MulBackward0", "SumBackward"]),
    # Outer-scope fallbacks — only reached when nothing more specific matched.
    ("sparse preprocess misc", ["## preprocess ##"]),
    ("HSTU body misc",         ["## user_forward ##"]),
    ("model glue misc",        ["## main_forward ##", "DistributedDataParallel"]),
]

_HSTU_MODULES = ("HSTU attention block", "HSTU output projection",
                 "HSTU input preprocess", "HSTU output postproc",
                 "multitask head (MLP)", "HSTU body misc")
_SPARSE_MODULES = ("emb input dist (a2a)", "emb lookup (TBE)",
                   "emb output dist (a2a)", "emb jagged regather",
                   "sparse preprocess misc")


def _load(path: str) -> tuple[list[dict], dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data["traceEvents"], data
    return data, {}


def _device_name(meta: dict) -> str:
    for dev in meta.get("deviceProperties", []):
        n = dev.get("name")
        if n:
            return n
    return "unknown"


def _op_kind(name: str) -> str:
    low = name.lower()
    for kind, pats in _OP_KIND_RULES:
        if any(p in low for p in pats):
            return kind
    return "other"


def _module(stack: tuple[str, ...]) -> str:
    for frame in reversed(stack):
        for mod, pats in _MODULE_RULES:
            if any(p in frame for p in pats):
                return mod
    return "unattributed"


def _phase(stack: tuple[str, ...], module: str) -> str:
    if module == "optimizer":
        return "optimizer"
    for frame in stack:
        if "autograd::engine" in frame or "## backward" in frame:
            return "bwd"
    for frame in stack:
        if "forward" in frame or "## " in frame:
            return "fwd"
    return "other"


def _launch_stacks(events: list[dict]) -> dict[int, tuple[str, ...]]:
    """correlation id -> enclosing CPU frame names (outermost first), one rank."""
    by_tid: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        cat = e.get("cat")
        if cat in _CPU_TREE_CATS or (cat in _LAUNCH_CATS
                                     and "correlation" in e.get("args", {})):
            if e.get("ph") == "X" and "ts" in e:
                by_tid[e.get("tid")].append(e)

    out: dict[int, tuple[str, ...]] = {}
    for evs in by_tid.values():
        # Parent-first ordering: earlier start wins, longer span wins on ties.
        evs.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
        stack: list[tuple[float, str]] = []
        for e in evs:
            ts = e["ts"]
            while stack and ts >= stack[-1][0]:
                stack.pop()
            if e.get("cat") in _LAUNCH_CATS:
                out[e["args"]["correlation"]] = tuple(n for _, n in stack)
            else:
                stack.append((ts + e.get("dur", 0), e.get("name", "")))
    return out


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and coalesce overlapping intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for s, e in intervals[1:]:
        if s > out[-1][1]:
            out.append((s, e))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], e))
    return out


def _measure(merged: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in merged) / 1000


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    return _measure(_merge(intervals))


def _subtract_ms(a: list[tuple[float, float]],
                 b: list[tuple[float, float]]) -> float:
    """Length of merged ``a`` with merged ``b`` removed — i.e. the part of a
    that is not covered by b. Used for *exposed* communication: comm intervals
    with no compute kernel running underneath them."""
    total = 0.0
    j = 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                total += min(b[k][0], e) - cur
            cur = max(cur, b[k][1])
            if cur >= e:
                break
            k += 1
        if cur < e:
            total += e - cur
    return total / 1000


def _clip(intervals: list[tuple[float, float]],
          lo: float, hi: float) -> list[tuple[float, float]]:
    out = []
    for s, e in intervals:
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > s2:
            out.append((s2, e2))
    return out


def analyze(path: str) -> dict:
    events, meta = _load(path)
    by_rank: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        pid = e.get("pid")
        if isinstance(pid, int) and pid >= 0:
            by_rank[pid // _PID_STRIDE].append(e)
    # Only ranks that actually ran kernels; stray metadata pids are not ranks.
    real_ranks = {r for r, revs in by_rank.items()
                  if any(e.get("cat") == "kernel" for e in revs)}
    by_rank = {r: v for r, v in by_rank.items() if r in real_ranks}

    # count unique ProfilerStep names.
    steps = {e["name"] for e in events
             if "ProfilerStep" in str(e.get("name", "")) and e.get("ph") == "X"}
    n_steps = len(steps) or 1

    # (module, kind, phase) -> [total_ms, count]
    cells: dict[tuple[str, str, str], list] = defaultdict(lambda: [0.0, 0])
    kernels: dict[tuple[str, str], list] = defaultdict(lambda: [0.0, 0])
    # (shape label, module, phase) -> [ms, n_kernels, kernel, flops, tokens, n_ops]
    gemm_shapes: dict[tuple, list] = defaultdict(lambda: [0.0, 0, "", 0.0, 0, 0])
    unattributed: dict[tuple[str, tuple], list] = defaultdict(lambda: [0.0, 0])
    per_rank_busy: dict[int, float] = {}
    per_rank_work: dict[int, float] = defaultdict(float)
    step_spans: list[float] = []
    n_ranks = 0
    tl: dict[str, float] = defaultdict(float)
    tl_buckets: dict[str, float] = defaultdict(float)

    for rank, revs in sorted(by_rank.items()):
        n_ranks += 1
        stacks = _launch_stacks(revs)

        # GPU-side step spans for this rank (one per ProfilerStep name).
        rank_steps: dict[int, list[tuple[float, float]]] = {rank: []}
        seen_steps: set = set()
        for e in revs:
            if (e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"
                    and "ProfilerStep" in str(e.get("name", ""))):
                if e["name"] not in seen_steps:
                    seen_steps.add(e["name"])
                    rank_steps[rank].append((e["ts"], e["ts"] + e.get("dur", 0)))

        # (M,N,K) per External id, for GEMM shape attribution.
        shape_by_ext: dict[int, tuple[int, int, int]] = {}
        for e in revs:
            if e.get("cat") != "cpu_op":
                continue
            a = e.get("args", {})
            dims, ext = a.get("Input Dims"), a.get("External id")
            name = e.get("name", "")
            if not dims or ext is None:
                continue
            try:
                if name == "aten::addmm" and len(dims) >= 3:
                    (M, K), (K2, N) = dims[1], dims[2]
                elif name == "aten::mm" and len(dims) >= 2:
                    (M, K), (K2, N) = dims[0], dims[1]
                elif name == "aten::bmm" and len(dims) >= 2:
                    continue
                else:
                    continue
            except (ValueError, TypeError):
                continue
            if K == K2:
                shape_by_ext[ext] = (M, N, K)

        intervals: list[tuple[float, float]] = []
        comm_iv: list[tuple[float, float]] = []
        compute_iv: list[tuple[float, float]] = []
        # Component intervals, for the timeline decomposition (HSTU attention,
        # GEMM, ... as fractions of the step rather than as summed kernel time).
        comp_iv: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # One aten::mm/addmm can emit several kernels (a splitK main kernel plus
        # its reduction) under a single External id. Time counts every kernel;
        # FLOPs must count the op once.
        flops_counted: set = set()
        for e in revs:
            if e.get("cat") not in _GPU_CATS or e.get("ph") != "X":
                continue
            dur = e.get("dur", 0)
            if dur <= 0:
                continue
            name = e.get("name", "")
            corr = e.get("args", {}).get("correlation")
            stack = stacks.get(corr, ())
            kind = _op_kind(name)
            mod = _module(stack)
            ph = _phase(stack, mod)
            if kind in ("attention fwd", "attention bwd"):
                ph = "fwd" if kind == "attention fwd" else "bwd"
            if kind == "collective" and mod == "unattributed":
                mod = "collective (unattributed)"

            c = cells[(mod, kind, ph)]
            c[0] += dur / 1000
            c[1] += 1
            k = kernels[(_short(name), f"{mod} / {ph}")]
            k[0] += dur / 1000
            k[1] += 1
            per_rank_work[rank] += dur / 1000
            iv = (e["ts"], e["ts"] + dur)
            intervals.append(iv)
            if kind == "collective":
                comm_iv.append(iv)
            else:
                compute_iv.append(iv)
            comp_iv[_timeline_bucket(mod, kind)].append(iv)

            if mod == "unattributed":
                u = unattributed[(_short(name), stack[-4:])]
                u[0] += dur / 1000
                u[1] += 1

            if kind == "GEMM":
                ext = e.get("args", {}).get("External id")
                shp = shape_by_ext.get(ext)
                if shp:
                    label, tokens = _gemm_label(*shp)
                    g = gemm_shapes[(label, mod, ph)]
                    g[0] += dur / 1000
                    g[1] += 1
                    if not g[2]:
                        g[2] = _short(name)
                    if ext not in flops_counted:
                        flops_counted.add(ext)
                        g[3] += 2 * shp[0] * shp[1] * shp[2]
                        g[4] += tokens
                        g[5] += 1

        per_rank_busy[rank] = _union_ms(intervals)

        # --- timeline decomposition, per (rank, step) ---
        # step = compute-busy + exposed comm + idle, all measured as wall time on
        # the GPU rather than as summed kernel time, so the three are additive.
        cmp_m, cmm_m = _merge(compute_iv), _merge(comm_iv)
        buckets_m = {k: _merge(v) for k, v in comp_iv.items()}
        for lo, hi in rank_steps.get(rank, []):
            span = (hi - lo) / 1000
            c = _clip(cmp_m, lo, hi)
            m = _clip(cmm_m, lo, hi)
            compute_busy = _measure(c)
            exposed = _subtract_ms(m, c)
            busy = _union_ms(c + m)
            tl["n"] += 1
            tl["step"] += span
            tl["compute"] += compute_busy
            tl["comm_total"] += _measure(m)
            tl["comm_exposed"] += exposed
            tl["idle"] += span - busy
            for k, mv in buckets_m.items():
                tl_buckets[k] += _measure(_clip(mv, lo, hi))

    for e in events:
        if ("ProfilerStep" in str(e.get("name", "")) and e.get("ph") == "X"
                and e.get("cat") == "gpu_user_annotation"):
            step_spans.append(e.get("dur", 0) / 1000)

    norm = n_ranks * n_steps
    return {
        "path": path,
        "device": _device_name(meta),
        "n_ranks": n_ranks,
        "n_steps": n_steps,
        "norm": norm,
        "cells": {k: (v[0] / norm, v[1] / norm) for k, v in cells.items()},
        "kernels": {k: (v[0] / norm, v[1] / norm) for k, v in kernels.items()},
        "unattributed": {k: (v[0] / norm, v[1] / norm) for k, v in unattributed.items()},
        # ms/step/rank, kernels/step/rank, kernel, TFLOP/s, mean T, ops/step/rank
        "gemm_shapes": {
            k: (v[0] / norm, v[1] / norm, v[2],
                (v[3] / (v[0] / 1000) / 1e12) if v[0] > 0 else 0.0,
                v[4] / v[5] if v[5] else 0, v[5] / norm)
            for k, v in gemm_shapes.items()},
        "work_ms_step": sum(per_rank_work.values()) / norm,
        "busy_ms_step": sum(per_rank_busy.values()) / norm,
        "step_ms": (sum(step_spans) / len(step_spans)) if step_spans else 0.0,
        # wall-time decomposition, averaged per (rank, step)
        "timeline": {k: v / tl["n"] for k, v in tl.items() if k != "n"} if tl["n"] else {},
        "tl_buckets": {k: v / tl["n"] for k, v in tl_buckets.items()} if tl["n"] else {},
    }


_TOKEN_DIM_MIN = 100_000


def _gemm_label(M: int, N: int, K: int) -> tuple[str, int]:
    """Canonical shape label with the jagged token dim abstracted as ``T``.

    ``M`` (or ``K``, for a weight-gradient GEMM) is the total token count of the
    batch, which drifts every step, so keying on the raw triple would scatter one
    logical GEMM across a row per step. The fixed weight dims are what identify
    it. Returns the label and the token count for averaging.
    """
    dims = (M, N, K)
    big = [i for i, d in enumerate(dims) if d >= _TOKEN_DIM_MIN]
    if len(big) != 1:
        return f"[{M},{K}]x[{K},{N}]", 0
    if big[0] == 0:      # activation GEMM: [T,K] x [K,N]
        return f"[T,{K}]x[{K},{N}]", M
    if big[0] == 2:      # weight-gradient GEMM: [M,T] x [T,N]
        return f"[{M},T]x[T,{N}]", K
    return f"[{M},{K}]x[{K},T]", N


def _timeline_bucket(mod: str, kind: str) -> str:
    """Coarse bucket for the wall-time decomposition of a step."""
    if kind in ("attention fwd", "attention bwd"):
        return f"HSTU attention {kind.split()[1]}"
    if kind == "collective":
        return "collective (total)"
    if kind == "GEMM":
        return "GEMM"
    if kind == "HSTU Triton p'wise":
        return "HSTU Triton pointwise"
    if kind in ("embedding scatter", "gather / index"):
        return "embedding gather/scatter"
    return "other compute"


def _short(name: str, n: int = 58) -> str:
    name = re.sub(r"\s*\[clone \.kd\]$", "", name)
    name = re.sub(r"\.kd$", "", name)
    name = re.sub(r"^void ", "", name)
    name = name.split("(")[0].split("<")[0]
    return name[:n]


def _fmt(x: float) -> str:
    return f"{x:8.2f}"


def _rollup(res: dict, keyfn) -> dict[str, tuple[float, float]]:
    out: dict[str, list] = defaultdict(lambda: [0.0, 0.0])
    for k, (ms, cnt) in res["cells"].items():
        key = keyfn(k)
        if key is None:
            continue
        out[key][0] += ms
        out[key][1] += cnt
    return {k: tuple(v) for k, v in out.items()}


def _print_compare(a: dict, b: dict, la: str, lb: str) -> None:
    w = 26
    print(f"\n{'='*104}\nGPU work per rank per step — {la} vs {lb}\n{'='*104}")
    for res, lab in ((a, la), (b, lb)):
        print(f"  {lab:<8} {res['device']:<22} {res['n_ranks']} ranks x "
              f"{res['n_steps']} steps | step {res['step_ms']:.1f} ms | "
              f"kernel work {res['work_ms_step']:.1f} ms | "
              f"GPU busy (union) {res['busy_ms_step']:.1f} ms "
              f"({100*res['busy_ms_step']/res['step_ms']:.0f}% occupancy)"
              if res["step_ms"] else f"  {lab}: {res['device']}")

    def table(title, keyfn, sort_by_a=True):
        ra, rb = _rollup(a, keyfn), _rollup(b, keyfn)
        keys = set(ra) | set(rb)
        rows = sorted(keys, key=lambda k: -(ra.get(k, (0, 0))[0] if sort_by_a
                                            else rb.get(k, (0, 0))[0]))
        print(f"\n{title}")
        print(f"  {'component':<{w}} {la:>9} {lb:>9} {'gap':>9} {'gap %':>8}"
              f" {'kn/step ' + la:>12} {'kn/step ' + lb:>12}")
        print("  " + "-" * (w + 66))
        ta = tb = 0.0
        for k in rows:
            ma, ca = ra.get(k, (0.0, 0.0))
            mb, cb = rb.get(k, (0.0, 0.0))
            ta += ma
            tb += mb
            gap = mb - ma
            pct = (gap / ma * 100) if ma > 0 else float("nan")
            pcts = f"{pct:+7.1f}%" if ma > 0 else "      —"
            print(f"  {k:<{w}} {_fmt(ma)} {_fmt(mb)} {gap:+9.2f} {pcts}"
                  f" {ca:12.0f} {cb:12.0f}")
        print("  " + "-" * (w + 66))
        print(f"  {'TOTAL':<{w}} {_fmt(ta)} {_fmt(tb)} {tb-ta:+9.2f} "
              f"{(tb-ta)/ta*100:+7.1f}%")

    table("By module (all phases) — ms/step/rank",
          lambda k: k[0])
    table("By op kind (all modules) — ms/step/rank",
          lambda k: k[1])
    table("By phase — ms/step/rank",
          lambda k: k[2])
    table("HSTU only, by module x phase — ms/step/rank",
          lambda k: f"{k[0]} [{k[2]}]" if k[0] in _HSTU_MODULES else None)
    table("Sparse path + collectives, by module x kind — ms/step/rank",
          lambda k: f"{k[0]} [{k[1]}]"
          if (k[0] in _SPARSE_MODULES or "collective" in k[0]
              or "all-reduce" in k[0]) else None)
    table("GEMM only, by module x phase — ms/step/rank",
          lambda k: f"{k[0]} [{k[2]}]" if k[1] == "GEMM" else None)


def _print_single(res: dict, label: str) -> None:
    """Summed-kernel-time breakdown for one trace (no A/B column)."""
    step = res["step_ms"]
    for title, keyfn in (
        ("By module — ms/step/rank", lambda k: k[0]),
        ("By op kind — ms/step/rank", lambda k: k[1]),
        ("By phase — ms/step/rank", lambda k: k[2]),
    ):
        roll = _rollup(res, keyfn)
        print(f"\n{title}  ({label})")
        print(f"  {'component':<28} {'ms':>9} {'% work':>8} {'% step':>8} {'kn/step':>9}")
        print("  " + "-" * 66)
        tot = sum(v[0] for v in roll.values())
        for k, (ms, cnt) in sorted(roll.items(), key=lambda x: -x[1][0]):
            print(f"  {k:<28} {ms:9.2f} {100*ms/tot:7.1f}% "
                  f"{100*ms/step:7.1f}% {cnt:9.0f}")
        print("  " + "-" * 66)
        print(f"  {'TOTAL':<28} {tot:9.2f} {100.0:7.1f}% {100*tot/step:7.1f}%")


def _print_timeline(res: dict, label: str) -> None:
    """Wall-time decomposition of a step: where the GPU's time actually goes."""
    t = res.get("timeline") or {}
    if not t:
        print(f"\n  {label}: no GPU step spans found; cannot decompose")
        return
    step = t["step"]
    print(f"\n{'='*104}\nStep wall-time decomposition — {label} ({res['device']})"
          f"\n{'='*104}")
    print(f"  averaged over {res['n_ranks']} ranks x {res['n_steps']} steps; "
          f"these three are additive and sum to the step")
    print(f"\n  {'component':<34} {'ms/step':>9} {'% step':>8}")
    print("  " + "-" * 54)
    rows = [
        ("compute busy (any compute kernel)", t["compute"]),
        ("exposed comm (comm, no compute)", t["comm_exposed"]),
        ("idle (no GPU work at all)", t["idle"]),
    ]
    for n_, v in rows:
        print(f"  {n_:<34} {v:9.2f} {100*v/step:7.1f}%")
    print("  " + "-" * 54)
    print(f"  {'STEP':<34} {step:9.2f} {100.0:7.1f}%")

    print(f"\n  communication detail")
    print(f"    {'collective wall time (union)':<32} {t['comm_total']:9.2f} ms"
          f" {100*t['comm_total']/step:6.1f}% of step")
    print(f"    {'-> overlapped with compute':<32} "
          f"{t['comm_total']-t['comm_exposed']:9.2f} ms"
          f" {100*(t['comm_total']-t['comm_exposed'])/t['comm_total']:6.1f}% of comm")
    print(f"    {'-> exposed (on critical path)':<32} {t['comm_exposed']:9.2f} ms"
          f" {100*t['comm_exposed']/t['comm_total']:6.1f}% of comm")

    buckets = dict(res["tl_buckets"])
    # Collectives are mostly hidden behind compute, so their total wall time
    # overstates what they cost the step. Report only the exposed part, which is
    # what the step actually pays for them.
    if "collective (total)" in buckets:
        del buckets["collective (total)"]
        buckets["collective (exposed)"] = t["comm_exposed"]

    print(f"\n  compute occupancy by bucket (wall time, overlaps possible;"
          f" collectives counted as exposed only)")
    print(f"    {'bucket':<28} {'ms/step':>9} {'% step':>8}")
    print("    " + "-" * 48)
    for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"    {k:<28} {v:9.2f} {100*v/step:7.1f}%")


def _print_top_kernels(res: dict, label: str, n: int = 25) -> None:
    print(f"\n{'-'*104}\nTop {n} kernels — {label} ({res['device']}), ms/step/rank")
    print(f"  {'ms':>8} {'n':>6}  {'module / phase':<34} kernel")
    rows = sorted(res["kernels"].items(), key=lambda x: -x[1][0])[:n]
    for (kn, modph), (ms, cnt) in rows:
        print(f"  {ms:8.2f} {cnt:6.1f}  {modph:<34} {kn}")


def _print_gemms(a: dict, b: dict, la: str, lb: str, n: int = 16) -> None:
    print(f"\n{'='*124}\nGEMM shapes (T = jagged token count of the batch) — "
          f"ms/step/rank and achieved TFLOP/s\n{'='*124}")
    keys = set(a["gemm_shapes"]) | set(b["gemm_shapes"])
    ranked = sorted(keys, key=lambda k: -(a["gemm_shapes"].get(k, (0,))[0]
                                          + b["gemm_shapes"].get(k, (0,))[0]))[:n]
    print(f"  {'shape':<24} {'mean T':>9} {'op/st':>5} | "
          f"{la+' ms':>9} {lb+' ms':>9} {'gap %':>8} | "
          f"{la+' TF/s':>10} {lb+' TF/s':>10} {'ratio':>6} | {'module':<24} phase")
    print("  " + "-" * 138)
    for k in ranked:
        shape, mod, ph = k
        ga = a["gemm_shapes"].get(k, (0.0, 0.0, "", 0.0, 0, 0.0))
        gb = b["gemm_shapes"].get(k, (0.0, 0.0, "", 0.0, 0, 0.0))
        gap = (gb[0] - ga[0]) / ga[0] * 100 if ga[0] else float("nan")
        ratio = ga[3] / gb[3] if gb[3] else 0
        print(f"  {shape:<24} {max(ga[4], gb[4]):>9.0f} {max(ga[5], gb[5]):>5.1f} | "
              f"{ga[0]:9.2f} {gb[0]:9.2f} {gap:+7.1f}% | "
              f"{ga[3]:10.1f} {gb[3]:10.1f} {ratio:6.2f}x | {mod:<24} {ph}")
    print("  " + "-" * 138)
    for res, lab in ((a, la), (b, lb)):
        tot = sum(v[0] for v in res["gemm_shapes"].values())
        # v[3] is TFLOP/s for that row and v[0] its seconds, so the product is
        # TFLOP of real work in that row.
        tflop = sum(v[3] * v[0] / 1000 for v in res["gemm_shapes"].values())
        print(f"  {lab:<8} shape-attributed GEMM {tot:6.2f} ms/step/rank, "
              f"{tflop:5.2f} TFLOP/step/rank, aggregate "
              f"{tflop/(tot/1000):6.1f} TFLOP/s")


def _print_unattributed(res: dict, label: str, n: int = 10) -> None:
    print(f"\n  {label} ({res['device']}) — top unattributed kernels + launch stack")
    rows = sorted(res["unattributed"].items(), key=lambda x: -x[1][0])[:n]
    for (kn, stack), (ms, cnt) in rows:
        print(f"    {ms:8.2f} ms {cnt:6.1f}x  {kn}")
        print(f"             stack: {' > '.join(s[:46] for s in stack) or '(empty)'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="+")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--top-kernels", type=int, default=25)
    p.add_argument("--debug-unattributed", action="store_true",
                   help="print the CPU launch stack of the largest unattributed kernels")
    args = p.parse_args()

    labels = args.labels or [Path(t).parent.name for t in args.traces]
    results = [analyze(t) for t in args.traces]

    if len(results) == 2:
        _print_compare(results[0], results[1], labels[0], labels[1])
        _print_gemms(results[0], results[1], labels[0], labels[1])
    for res, lab in zip(results, labels):
        _print_timeline(res, lab)
    if len(results) == 1:
        _print_single(results[0], labels[0])
    for res, lab in zip(results, labels):
        _print_top_kernels(res, lab, args.top_kernels)

    print(f"\n{'='*104}\nUnattributed check\n{'='*104}")
    for res, lab in zip(results, labels):
        un = sum(ms for (mod, _, _), (ms, _) in res["cells"].items()
                 if mod.startswith("unattributed"))
        tot = sum(ms for (ms, _) in res["cells"].values())
        print(f"  {lab:<8} unattributed {un:.2f} / {tot:.2f} ms/step/rank "
              f"({100*un/tot:.1f}%)")
    if args.debug_unattributed:
        for res, lab in zip(results, labels):
            _print_unattributed(res, lab)


if __name__ == "__main__":
    main()

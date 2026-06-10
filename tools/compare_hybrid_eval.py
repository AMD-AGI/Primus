#!/usr/bin/env python3
"""Side-by-side compare lm-eval results from two output dirs.

Produces the table the team uses for parity reports:

    Task             Metric    Primus  ±stderr   FLA    ±stderr   Δ abs   σ (|Δ|/√(σ1²+σ2²))
    arc_easy         acc       ...     ±...      ...    ±...     -0.008   0.58
    arc_easy         acc_norm  ...     ±...      ...    ±...      0.006   0.44
    ...
    Mean of N metrics                                             ...     ...
    Mean |Δ|                                                      ...     ...
    Max  |Δ|                                                       ... (which task)

`mmlu` is reported as a single line averaged over its 57 sub-tasks
(matches FLA's `mmlu (avg of 57)` row).
"""
import argparse
import json
import math
import sys
from pathlib import Path

# (display_name, lm-eval task id, [metrics])  — order = display order
DEFAULT_REPORT = [
    ("arc_easy",         "arc_easy",       ["acc", "acc_norm"]),
    ("arc_challenge",    "arc_challenge",  ["acc", "acc_norm"]),
    ("hellaswag",        "hellaswag",      ["acc", "acc_norm"]),
    ("openbookqa",       "openbookqa",     ["acc", "acc_norm"]),
    ("piqa",             "piqa",           ["acc", "acc_norm"]),
    ("winogrande",       "winogrande",     ["acc"]),
    ("mmlu (avg of 57)", "mmlu",           ["acc"]),
    ("race",             "race",           ["acc"]),
]


def _find_results_json(out_dir: Path):
    candidates = sorted(out_dir.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _metric_lookup(task_results, metric):
    """Return (value, stderr) for `metric` in lm-eval's results dict for one task."""
    if task_results is None:
        return None, None
    val = task_results.get(f"{metric},none", task_results.get(metric))
    err = task_results.get(f"{metric}_stderr,none", task_results.get(f"{metric}_stderr"))
    if val is None:
        return None, None
    if err is None or (isinstance(err, str) and err.upper() == "N/A"):
        err = 0.0
    return float(val), float(err)


def _mmlu_aggregate(results):
    """lm-eval reports mmlu as 57 leaf tasks (mmlu_abstract_algebra, ...) plus
    parent rollup keys (mmlu, mmlu_humanities, mmlu_stem, ...).
    Prefer the parent 'mmlu' if present; otherwise average the 57 leaves
    (acc) and propagate stderr via √(Σσ²)/N."""
    if "mmlu" in results:
        return results["mmlu"]
    leaves = []
    for k, v in results.items():
        if not k.startswith("mmlu_"):
            continue
        # parent groups have no `acc,none`, skip
        if not isinstance(v, dict):
            continue
        if any(kk.startswith("acc") for kk in v):
            leaves.append(v)
    if not leaves:
        return None
    accs = [_metric_lookup(v, "acc")[0] for v in leaves]
    errs = [_metric_lookup(v, "acc")[1] for v in leaves]
    accs = [a for a in accs if a is not None]
    errs = [e for e in errs if e is not None]
    if not accs:
        return None
    return {
        "acc,none": sum(accs) / len(accs),
        "acc_stderr,none": math.sqrt(sum(e * e for e in errs)) / len(accs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primus-dir", required=True)
    ap.add_argument("--fla-dir", required=True)
    ap.add_argument("--name-primus", default="Primus")
    ap.add_argument("--name-fla", default="FLA")
    args = ap.parse_args()

    p_json = _find_results_json(Path(args.primus_dir))
    f_json = _find_results_json(Path(args.fla_dir))
    if not p_json or not f_json:
        print(f"ERROR: missing results json. primus={p_json}, fla={f_json}")
        sys.exit(1)

    with open(p_json) as f:
        primus_results = json.load(f).get("results", {})
    with open(f_json) as f:
        fla_results = json.load(f).get("results", {})

    # Materialize mmlu rollup if it's split across 57 leaf tasks
    if "mmlu" not in primus_results:
        agg = _mmlu_aggregate(primus_results)
        if agg is not None:
            primus_results = {**primus_results, "mmlu": agg}
    if "mmlu" not in fla_results:
        agg = _mmlu_aggregate(fla_results)
        if agg is not None:
            fla_results = {**fla_results, "mmlu": agg}

    print(f"{args.name_primus:8} results: {p_json}")
    print(f"{args.name_fla:8} results: {f_json}")
    print()

    HDR = f"{'Task':22} {'Metric':9} {args.name_primus:>8} {'±stderr':>9}  {args.name_fla:>8} {'±stderr':>9}  {'Δ abs':>8}  {'σ-units':>8}"
    print(HDR)
    print("-" * len(HDR))

    abs_deltas = []
    z_scores = []
    pvals = []
    fvals = []
    rows_for_max = []

    for display_name, task_id, metrics in DEFAULT_REPORT:
        ptask = primus_results.get(task_id)
        ftask = fla_results.get(task_id)
        for m_i, metric in enumerate(metrics):
            pv, pe = _metric_lookup(ptask, metric)
            fv, fe = _metric_lookup(ftask, metric)
            if pv is None or fv is None:
                print(f"{(display_name if m_i==0 else ''):22} {metric:9}    (missing)")
                continue
            d = pv - fv
            denom = math.sqrt(pe * pe + fe * fe) if (pe or fe) else 0.0
            z = (abs(d) / denom) if denom > 0 else 0.0
            label_left = display_name if m_i == 0 else ""
            print(
                f"{label_left:22} {metric:9} {pv:8.4f} ±{pe:7.4f}  {fv:8.4f} ±{fe:7.4f}  "
                f"{d:+8.4f}  {z:8.2f}"
            )
            abs_deltas.append(abs(d))
            z_scores.append(z)
            pvals.append(pv)
            fvals.append(fv)
            rows_for_max.append((abs(d), f"{display_name} {metric}", d))

    print("-" * len(HDR))
    n = len(abs_deltas)
    if n:
        mean_p = sum(pvals) / n
        mean_f = sum(fvals) / n
        mean_d = mean_p - mean_f
        mean_abs_d = sum(abs_deltas) / n
        mean_z = sum(z_scores) / n
        max_abs_d, max_label, max_d_signed = max(rows_for_max, key=lambda r: r[0])
        max_z = max(z_scores)
        print(
            f"{'Mean of '+str(n)+' metrics':22} {'':9} {mean_p:8.4f} {'':>8}  {mean_f:8.4f} {'':>8}  "
            f"{mean_d:+8.4f}  {mean_z:8.2f}"
        )
        print(
            f"{'Mean |Δ|':22} {'':9} {'':>8} {'':>8}  {'':>8} {'':>8}  "
            f"{mean_abs_d:8.4f}  {'':>8}"
        )
        print(
            f"{'Max  |Δ|':22} {'':9} {'':>8} {'':>8}  {'':>8} {'':>8}  "
            f"{max_d_signed:+8.4f}  {max_z:8.2f}    ({max_label})"
        )


if __name__ == "__main__":
    main()

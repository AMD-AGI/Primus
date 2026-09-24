###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Turn a Primus training log into a loss curve, a CSV, and a health summary.

Primus emits Megatron's per-iteration line at DEBUG, so the numbers live in
``<workspace>/<group>/<user>/<exp>/logs/pre_trainer/rank-<last>/debug.log``
rather than on the console. Point this at the experiment directory and it finds
the right rank for you.

Re-running an experiment appends to the same debug.log, so the file usually
holds several runs. Runs are split on the iteration counter resetting and the
last one is used by default.

Examples:
    # Most recent run in an experiment directory
    python3 tools/convergence_test/plot_loss.py output/amd/root/my-exp

    # Overlay two runs, x-axis in tokens
    python3 tools/convergence_test/plot_loss.py runA.log runB.log \
        --labels "rocm 7.14" "rocm 7.15" --x tokens --out compare
"""

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path

ITER_RE = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
VALID_RE = re.compile(r"validation loss at iteration (\d+) \| lm loss value:\s*([\d.Ee+-]+)")
MEM_RE = re.compile(r"max mem usage/usage_ratio:\s*([\d.]+)GB/([\d.]+)%")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")

# Megatron's log line is a pipe-separated list of "key: value" pairs. Only these
# are kept; everything else in the line is log-framing noise.
FIELDS = {
    "lm loss": "loss",
    "learning rate": "lr",
    "grad norm": "grad_norm",
    "global batch size": "global_batch_size",
    "consumed samples": "consumed_samples",
    "elapsed time per iteration (ms)": "elapsed_ms",
    "throughput per GPU (TFLOP/s/GPU)": "tflops",
    "compute per GPU (TFLOP/s/GPU)": "tflops",
    "tokens/s/GPU inst/harmonic mean": "tokens_s_gpu",
    "load_balancing_loss": "aux_loss",
    "number of skipped iterations": "skipped",
    "number of nan iterations": "nan",
}


def resolve_log(path):
    """Accept a log file, an experiment directory, or a run directory."""
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(glob.glob(str(path / "**" / "rank-*" / "debug.log"), recursive=True))
    if not candidates:
        candidates = sorted(glob.glob(str(path / "**" / "*.log"), recursive=True))
    if not candidates:
        raise SystemExit(f"no log files found under {path}")
    # Megatron logs iterations from the last pipeline rank only.
    best, best_hits = None, -1
    for candidate in candidates:
        with open(candidate, errors="ignore") as handle:
            hits = sum(1 for line in handle if "lm loss:" in line)
        if hits > best_hits:
            best, best_hits = candidate, hits
    if best_hits <= 0:
        raise SystemExit(f"no iteration lines found under {path}")
    return Path(best)


def parse_line(line):
    match = ITER_RE.search(line)
    if not match or "lm loss:" not in line:
        return None
    record = {"iteration": int(match.group(1)), "total_iters": int(match.group(2))}
    for segment in line.split("|"):
        if ":" not in segment:
            continue
        key, _, value = segment.partition(":")
        name = FIELDS.get(key.strip())
        if not name:
            continue
        number = NUM_RE.search(value)
        if number:
            record[name] = float(number.group())
    mem = MEM_RE.search(line)
    if mem:
        record["peak_mem_gb"] = float(mem.group(1))
        record["peak_mem_pct"] = float(mem.group(2))
    return record


def parse_log(path):
    """Return a list of runs; each run is (train_records, valid_records)."""
    runs = []
    train, valid = [], []
    last_iteration = 0

    with open(path, errors="ignore") as handle:
        for line in handle:
            record = parse_line(line)
            if record:
                # A non-increasing iteration counter means a new run was appended.
                if record["iteration"] <= last_iteration and train:
                    runs.append((train, valid))
                    train, valid = [], []
                last_iteration = record["iteration"]
                train.append(record)
                continue
            match = VALID_RE.search(line)
            if match:
                valid.append({"iteration": int(match.group(1)), "loss": float(match.group(2))})
    if train:
        runs.append((train, valid))
    return runs


def steady_state_ms(train):
    """Median step time over the second half of the run, ignoring iteration 1.

    Iteration 1 carries all of the kernel autotuning, and for MoE the step time
    keeps falling for dozens of iterations as the router balances and the
    per-expert buffers shrink. Returns None if there is nothing usable.
    """
    samples = [r["elapsed_ms"] for r in train if "elapsed_ms" in r and r["iteration"] > 1]
    if not samples:
        return None
    second_half = sorted(samples[len(samples) // 2 :])
    return second_half[len(second_half) // 2]


def summarise(label, train, valid, vocab_size=None):
    first, last = train[0], train[-1]
    gbs = last.get("global_batch_size")
    median_ms = steady_state_ms(train)

    print(f"\n{label}\n" + "-" * len(label))
    print(f"  iterations        : {first['iteration']} -> {last['iteration']} (of {last['total_iters']})")
    print(f"  lm loss           : {first.get('loss'):.4f} -> {last.get('loss'):.4f}")
    if valid:
        print(f"  validation loss   : {valid[0]['loss']:.4f} -> {valid[-1]['loss']:.4f}")
    if vocab_size:
        expected = math.log(vocab_size)
        delta = first.get("loss", 0) - expected
        verdict = "as expected" if abs(delta) < 0.6 else "SUSPICIOUS (tokenizer/data mismatch?)"
        print(f"  initial vs ln(V)  : {first.get('loss'):.2f} vs {expected:.2f} -- {verdict}")
    if median_ms:
        print(f"  median s/iter     : {median_ms/1000:.1f}")
    if "tokens_s_gpu" in last:
        print(f"  tokens/s/GPU      : {last['tokens_s_gpu']:.0f} (final instantaneous)")
    if gbs:
        print(f"  global batch size : {int(gbs)}")
    nan_total = max((r.get("nan", 0) for r in train), default=0)
    skipped = max((r.get("skipped", 0) for r in train), default=0)
    flag = "" if nan_total == 0 and skipped == 0 else "   <-- investigate"
    print(f"  nan / skipped     : {int(nan_total)} / {int(skipped)}{flag}")
    if "peak_mem_pct" in last:
        risky = "   <-- little headroom" if last["peak_mem_pct"] > 90 else ""
        print(f"  peak memory       : {last['peak_mem_gb']:.1f} GB ({last['peak_mem_pct']:.1f}%){risky}")
    if "aux_loss" in last:
        print(f"  moe aux loss      : {train[0].get('aux_loss', float('nan')):.3f} -> {last['aux_loss']:.3f}")
    if "grad_norm" in last:
        print(f"  final grad norm   : {last['grad_norm']:.3f}")


def recommend_iterations(train, budget_hours, startup_min, seq_length, wall_clock_min=None):
    """How many iterations fit in a wall-clock budget, from measured step time.

    If the caller measured the wall clock of the probe, start-up is derived
    (wall clock minus time actually spent stepping) instead of guessed.
    """
    median_ms = steady_state_ms(train)
    if median_ms is None:
        print("\n[budget] not enough timing data; probe for more iterations")
        return
    median_s = median_ms / 1000.0
    samples = [r for r in train if "elapsed_ms" in r and r["iteration"] > 1]

    source = "assumed"
    if wall_clock_min is not None:
        stepping_min = sum(r["elapsed_ms"] for r in train if "elapsed_ms" in r) / 60000.0
        startup_min = max(0.0, wall_clock_min - stepping_min)
        source = "measured"

    budget_s = budget_hours * 3600 - startup_min * 60
    # Reserve ~5% for periodic validation and teardown.
    iterations = int(budget_s * 0.95 / median_s)
    gbs = train[-1].get("global_batch_size")

    print(f"\nbudget for {budget_hours:g} h\n" + "-" * 20)
    print(f"  median step time  : {median_s:.1f} s (from {len(samples)} samples, iteration 1 excluded)")
    print(f"  start-up          : {startup_min:.1f} min ({source})")
    print(f"  recommended iters : {iterations}")
    if gbs and seq_length:
        print(f"  that is           : {iterations*gbs*seq_length/1e6:.0f}M tokens")
    print("  remember to set lr_decay_iters to the same value")
    if len(samples) < 10:
        print("  NOTE: very few timing samples; probe at least 20 iterations")
    if any("aux_loss" in r for r in train):
        print("  NOTE: MoE step time keeps falling for ~50 iterations as the router")
        print("        balances, so a short probe under-estimates the iteration count")


def write_csv(path, train, valid):
    valid_by_iter = {v["iteration"]: v["loss"] for v in valid}
    columns = [
        "iteration",
        "loss",
        "valid_loss",
        "lr",
        "grad_norm",
        "elapsed_ms",
        "tflops",
        "tokens_s_gpu",
        "aux_loss",
        "consumed_samples",
        "peak_mem_pct",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in train:
            row = dict(record)
            row["valid_loss"] = valid_by_iter.get(record["iteration"], "")
            writer.writerow(row)


def x_values(records, mode, seq_length):
    if mode == "tokens" and seq_length:
        return [r.get("consumed_samples", r["iteration"]) * seq_length / 1e9 for r in records]
    return [r["iteration"] for r in records]


def plot(runs, labels, out_prefix, mode, seq_length, detailed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 4 if detailed else 1
    fig, axes = plt.subplots(panels, 1, figsize=(10, 4.2 * panels), squeeze=False)
    axes = axes[:, 0]
    xlabel = "tokens (B)" if mode == "tokens" and seq_length else "iteration"

    for index, ((train, valid), label) in enumerate(zip(runs, labels)):
        colour = f"C{index}"
        axes[0].plot(
            x_values(train, mode, seq_length),
            [r.get("loss") for r in train],
            lw=1.3,
            color=colour,
            label=f"{label} train",
        )
        if valid:
            by_iter = {r["iteration"]: r for r in train}
            vx = [by_iter[v["iteration"]] for v in valid if v["iteration"] in by_iter]
            if vx:
                axes[0].plot(
                    x_values(vx, mode, seq_length),
                    [v["loss"] for v in valid if v["iteration"] in by_iter],
                    "o--",
                    ms=3.5,
                    lw=1.0,
                    color=colour,
                    alpha=0.75,
                    label=f"{label} valid",
                )
        if detailed:
            for axis, key, title in (
                (axes[1], "lr", "learning rate"),
                (axes[2], "grad_norm", "grad norm"),
                (axes[3], "tokens_s_gpu", "tokens/s/GPU"),
            ):
                series = [(r, r[key]) for r in train if key in r]
                if series:
                    axis.plot(
                        x_values([r for r, _ in series], mode, seq_length),
                        [v for _, v in series],
                        lw=1.2,
                        color=colour,
                        label=label,
                    )
                axis.set_ylabel(title)

    axes[0].set_ylabel("lm loss")
    axes[0].set_title("Primus convergence test")
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    png = f"{out_prefix}.png"
    fig.savefig(png, dpi=130)
    return png


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("logs", nargs="+", help="Log files or experiment directories")
    parser.add_argument("--labels", nargs="*", help="Legend label per input")
    parser.add_argument("--out", default="loss_curve", help="Output prefix for .png/.csv")
    parser.add_argument("--x", choices=["iterations", "tokens"], default="iterations")
    parser.add_argument("--seq-length", type=int, help="Needed for the tokens x-axis")
    parser.add_argument("--vocab-size", type=int, help="Sanity-check initial loss against ln(V)")
    parser.add_argument("--dataset-info", help="dataset_info.json to read vocab_size from")
    parser.add_argument(
        "--run-index",
        type=int,
        default=-1,
        help="Which run in the log to use when it holds several (default: last)",
    )
    parser.add_argument("--detailed", action="store_true", help="Also plot lr, grad norm and throughput")
    parser.add_argument("--no-plot", action="store_true", help="Summary and CSV only")
    parser.add_argument(
        "--budget-hours", type=float, help="Report the train_iters that fit in this wall-clock budget"
    )
    parser.add_argument(
        "--startup-min",
        type=float,
        default=5.0,
        help="Start-up overhead to reserve from the budget (default 5 min)",
    )
    parser.add_argument(
        "--wall-clock-min", type=float, help="Measured wall clock of this run; derives start-up exactly"
    )
    args = parser.parse_args()

    vocab_size = args.vocab_size
    if args.dataset_info:
        vocab_size = json.loads(Path(args.dataset_info).read_text()).get("vocab_size")

    labels = args.labels or [Path(p).name for p in args.logs]
    if len(labels) != len(args.logs):
        raise SystemExit("--labels must give one label per log")

    selected = []
    for source, label in zip(args.logs, labels):
        path = resolve_log(source)
        runs = parse_log(path)
        if not runs:
            raise SystemExit(f"no iteration lines parsed from {path}")
        if len(runs) > 1:
            print(f"[plot-loss] {path} holds {len(runs)} runs; using index {args.run_index}")
        train, valid = runs[args.run_index]
        print(f"[plot-loss] {label}: {path} ({len(train)} points, {len(valid)} validations)")
        selected.append((train, valid))
        summarise(label, train, valid, vocab_size)
        if args.budget_hours:
            recommend_iterations(
                train, args.budget_hours, args.startup_min, args.seq_length, args.wall_clock_min
            )

    for (train, valid), label in zip(selected, labels):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        csv_path = f"{args.out}_{safe}.csv" if len(selected) > 1 else f"{args.out}.csv"
        write_csv(csv_path, train, valid)
        print(f"\n[plot-loss] wrote {csv_path}")

    if not args.no_plot:
        png = plot(selected, labels, args.out, args.x, args.seq_length, args.detailed)
        print(f"[plot-loss] wrote {png}")


if __name__ == "__main__":
    main()

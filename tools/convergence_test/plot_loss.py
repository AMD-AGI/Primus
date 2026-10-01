###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Turn a Primus training log into a loss curve, a CSV, and a health summary.

Reads Megatron and MaxText runs, and the CSVs this script writes, so an archived
run can be compared against a new one without its logs.

Megatron: Primus emits the per-iteration line at DEBUG, and on a single node
forwards it to the console, so a console log holds it with
``stderr_sink_level: DEBUG`` (every bundled config sets it). Evaluations are not
forwarded. Both are always in
``<workspace>/<group>/<user>/<exp>/logs/pre_trainer/rank-<last>/debug.log``;
point this at the experiment directory and it finds the right rank for you, or
plot a console log with ``--validation-from <experiment directory>``.

MaxText: the per-step line (``completed step: N ...``) is at INFO on rank 0. It
carries the loss to three decimals and no learning rate or grad norm; pass
``--tensorboard <base_output_directory>/<run_name>`` to take those, and
full-precision losses, from MaxText's TensorBoard file. MaxText counts steps
from 0; they are reported as iterations from 1 so both backends line up.

Re-running an experiment appends to the same log, so the file usually holds
several runs. Runs are split on the iteration counter resetting and the last one
is used by default.

``--baseline`` compares against a reference run and exits with status 3 if the
loss moved by more than ``--tolerance``, for release gating. ``--expect-iters``
exits with status 4 if the run logged fewer iterations: MaxText treats a failing
data iterator as a graceful stop and exits 0.

Examples:
    # Most recent run in an experiment directory
    python3 tools/convergence_test/plot_loss.py output/amd/root/my-exp

    # Overlay two runs, x-axis in tokens
    python3 tools/convergence_test/plot_loss.py runA.log runB.log \
        --labels "rocm 7.14" "rocm 7.15" --x tokens --seq-length 4096 --out compare

    # Gate a new run on an archived one
    python3 tools/convergence_test/plot_loss.py output/amd/root/my-exp \
        --baseline baselines/my-exp-v26.7.csv --tolerance 0.05
"""

import argparse
import bisect
import csv
import glob
import json
import math
import re
import struct
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Megatron
ITER_RE = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
VALID_RE = re.compile(
    r"validation loss at iteration (\d+)(?: on validation set)? \| lm loss value:\s*([\d.Ee+-]+)"
)
MEM_RE = re.compile(r"max mem usage/usage_ratio:\s*([\d.]+)GB/([\d.]+)%")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")

# MaxText
MT_TRAIN_RE = re.compile(r"completed (?:profiler activation/deactivation )?step: (\d+),(.*)")
MT_EVAL_RE = re.compile(r"Completed eval after train step (\d+), loss=([^,\s]+)")
MT_STEPS_RE = re.compile(r"Config param steps: (\d+)")

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

# MaxText's line is a comma-separated list of "key: value" pairs: (field, scale).
# lm_loss is the pure cross-entropy, comparable with Megatron's "lm loss"; loss
# also carries the MoE balancing term.
MT_FIELDS = {
    "seconds": ("elapsed_ms", 1000.0),
    "TFLOP/s/device": ("tflops", 1.0),
    "Tokens/s/device": ("tokens_s_gpu", 1.0),
    "total_weights": ("step_tokens", 1.0),
    "loss": ("total_loss", 1.0),
    "lm_loss": ("loss", 1.0),
    "moe_lb_loss": ("aux_loss", 1.0),
}

# MaxText TensorBoard tags merged into the log records: tag -> field. MaxText's
# learning/grad_norm is measured after clipping, so it sits at the threshold
# for most of a run; raw_grad_norm is the pre-clip norm Megatron reports.
TB_TRAIN_TAGS = {
    "learning/lm_loss": "loss",
    "learning/current_learning_rate": "lr",
    "learning/grad_norm": "grad_norm",
    "learning/raw_grad_norm": "grad_norm",
    "learning/moe_lb_loss": "aux_loss",
}
TB_EVAL_TAG = "eval/avg_loss"

CSV_COLUMNS = [
    "iteration",
    "loss",
    "valid_loss",
    "lr",
    "grad_norm",
    "elapsed_ms",
    "tflops",
    "tokens_s_gpu",
    "step_tokens",
    "aux_loss",
    "consumed_samples",
    "peak_mem_pct",
    "peak_mem_gb",
]


class NoIterations(Exception):
    """The input holds no training iterations at all."""


def _count_iteration_lines(path):
    with open(path, errors="ignore") as handle:
        return sum(1 for line in handle if "lm loss:" in line or "completed step:" in line)


def resolve_log(path):
    """Accept a log file, a CSV, an experiment directory, or a run directory."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.exists():
        raise NoIterations(f"{path} does not exist")
    candidates = sorted(glob.glob(str(path / "**" / "rank-*" / "debug.log"), recursive=True))
    if not candidates:
        candidates = sorted(glob.glob(str(path / "**" / "*.log"), recursive=True))
    if not candidates:
        raise NoIterations(f"no log files found under {path}")
    # Megatron logs iterations from the last pipeline rank only, MaxText from rank 0.
    best, best_hits = None, -1
    for candidate in candidates:
        hits = _count_iteration_lines(candidate)
        if hits > best_hits:
            best, best_hits = candidate, hits
    if best_hits <= 0:
        raise NoIterations(f"no iteration lines found under {path}")
    return Path(best)


def _float(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse_megatron_line(line):
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


def parse_maxtext_line(line):
    match = MT_TRAIN_RE.search(line)
    if not match:
        return None
    record = {"iteration": int(match.group(1)) + 1}
    for segment in match.group(2).split(","):
        key, _, value = segment.partition(":")
        field = MT_FIELDS.get(key.strip())
        number = _float(value.strip()) if field else None
        if number is not None:
            record[field[0]] = number * field[1]
    if "loss" not in record and "total_loss" in record:
        record["loss"] = record["total_loss"]
    return record if "loss" in record else None


def parse_log(path):
    """Return a list of runs; each run is (train_records, valid_records).

    A run ends when the iteration counter goes backwards. MaxText logs an
    evaluation *before* the step it follows, so an evaluation that lands behind
    the current run's last step also starts a new run.
    """
    runs = []
    train, valid = [], []
    last_iteration = 0
    total_iters = None

    def flush():
        nonlocal train, valid
        if train:
            runs.append((train, valid))
        train, valid = [], []

    with open(path, errors="ignore") as handle:
        for raw in handle:
            line = ANSI_RE.sub("", raw)
            record = parse_megatron_line(line) or parse_maxtext_line(line)
            if record:
                if record["iteration"] <= last_iteration and train:
                    flush()
                if "total_iters" not in record and total_iters:
                    record["total_iters"] = total_iters
                last_iteration = record["iteration"]
                train.append(record)
                continue
            match = VALID_RE.search(line)
            if match:
                iteration = int(match.group(1))
                # Megatron evaluates once more after the last iteration; when the
                # run also evaluated at that iteration, keep the in-run point.
                if not (valid and valid[-1]["iteration"] == iteration):
                    valid.append({"iteration": iteration, "loss": float(match.group(2))})
                continue
            match = MT_EVAL_RE.search(line)
            if match:
                iteration = int(match.group(1)) + 1
                if iteration < last_iteration and train:
                    flush()
                    last_iteration = 0
                loss = _float(match.group(2))
                if loss is not None:
                    valid.append({"iteration": iteration, "loss": loss})
                continue
            match = MT_STEPS_RE.search(line)
            if match:
                total_iters = int(match.group(1))
    flush()
    return runs


def parse_csv(path):
    """Read back a CSV written by write_csv as a single run."""
    train, valid = [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            record = {"iteration": int(float(row["iteration"]))}
            for key in CSV_COLUMNS[1:]:
                value = _float(row.get(key) or "")
                if value is not None:
                    record[key] = value
            if "valid_loss" in record:
                valid.append({"iteration": record["iteration"], "loss": record.pop("valid_loss")})
            if "loss" in record:
                train.append(record)
    return [(train, valid)] if train else []


def load_runs(source):
    path = resolve_log(source)
    runs = parse_csv(path) if path.suffix == ".csv" else parse_log(path)
    if not runs:
        raise NoIterations(f"no iteration lines parsed from {path}")
    return path, runs


def stopped_early(train, expect_iters):
    """Iterations missing from the end of a run, allowing for the logging stride."""
    last = train[-1]["iteration"]
    stride = last - train[-2]["iteration"] if len(train) > 1 else 1
    return expect_iters - last if expect_iters - last >= stride else 0


# ---------------------------------------------------------------------------
# TensorBoard (MaxText). A few dozen lines of TFRecord/protobuf decoding keep
# the host free of a tensorboard/tensorflow dependency. Only scalar summaries
# (Summary.Value.simple_value) are read, which is what tensorboardX writes.
# ---------------------------------------------------------------------------


def _varint(buf, pos):
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _proto_fields(buf):
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        elif wire == 2:
            size, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def read_tensorboard(path):
    """{tag: {step: value}} from the newest event file under ``path``."""
    files = sorted(glob.glob(str(Path(path) / "**" / "events.out.tfevents.*"), recursive=True))
    if not files:
        return None, None
    newest = max(files, key=lambda f: Path(f).stat().st_mtime)
    data = Path(newest).read_bytes()
    scalars = {}
    pos = 0
    # TFRecord framing: uint64 length, uint32 crc, payload, uint32 crc.
    while pos + 12 <= len(data):
        (length,) = struct.unpack_from("<Q", data, pos)
        if pos + 12 + length + 4 > len(data):
            break  # a record still being written
        payload = data[pos + 12 : pos + 12 + length]
        pos += 12 + length + 4
        step, summary = 0, None
        for number, wire, value in _proto_fields(payload):
            if number == 2 and wire == 0:  # Event.step
                step = value
            elif number == 5 and wire == 2:  # Event.summary
                summary = value
        if summary is None:
            continue
        for number, _, value in _proto_fields(summary):
            if number != 1:  # Summary.value
                continue
            tag = scalar = None
            for inner, inner_wire, inner_value in _proto_fields(value):
                if inner == 1:
                    tag = inner_value.decode(errors="replace")
                elif inner == 2 and inner_wire == 5:
                    scalar = struct.unpack("<f", inner_value)[0]
            if tag is not None and scalar is not None:
                scalars.setdefault(tag, {})[step] = scalar
    return scalars, newest


def merge_tensorboard(train, valid, path):
    """Fill in lr/grad norm and full-precision losses from MaxText's event file."""
    scalars, source = read_tensorboard(path)
    if not scalars:
        print(f"[plot-loss] no TensorBoard scalars under {path}")
        return
    losses = scalars.get("learning/lm_loss") or scalars.get("learning/loss") or {}
    # The log prints losses to 3 decimals. Refuse a file from a different run.
    probe = [r for r in train[:20] if r["iteration"] - 1 in losses and math.isfinite(r["loss"])]
    if not probe or any(abs(losses[r["iteration"] - 1] - r["loss"]) > 2e-3 for r in probe):
        print(f"[plot-loss] {source} does not match this run's losses; ignoring it")
        return
    for record in train:
        step = record["iteration"] - 1
        for tag, field in TB_TRAIN_TAGS.items():
            # MaxText writes a zero balancing loss for dense models too; only
            # refine it where the log shows the model has one.
            if field == "aux_loss" and "aux_loss" not in record:
                continue
            if step in scalars.get(tag, {}):
                record[field] = scalars[tag][step]
    evals = scalars.get(TB_EVAL_TAG, {})
    for record in valid:
        if record["iteration"] - 1 in evals:
            record["loss"] = evals[record["iteration"] - 1]
    print(f"[plot-loss] merged TensorBoard scalars from {source}")


def merge_validation(train, valid, source):
    """Take this run's validation points from another log of the same run.

    On a single node Primus forwards Megatron's per-iteration line to the
    console, but not its evaluations: those reach only the last rank's log,
    which also holds every earlier run of the experiment. The run there whose
    losses match this one iteration for iteration is this run.
    """
    try:
        path, runs = load_runs(source)
    except NoIterations as exc:
        print(f"[plot-loss] no validation points taken: {exc}")
        return
    for other_train, other_valid in reversed(runs):
        losses = {r["iteration"]: r["loss"] for r in other_train}
        if all(losses.get(r["iteration"]) == r["loss"] for r in train):
            merged = {v["iteration"]: v for v in other_valid}
            merged.update({v["iteration"]: v for v in valid})
            valid[:] = [merged[i] for i in sorted(merged)]
            print(f"[plot-loss] took {len(other_valid)} validation points from {path}")
            return
    print(f"[plot-loss] no run in {path} matches this one; no validation points taken from it")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


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


def summarise(label, train, valid, vocab_size=None, seq_length=None, global_batch_size=None):
    first, last = train[0], train[-1]
    gbs = last.get("global_batch_size") or global_batch_size
    median_ms = steady_state_ms(train)
    total = last.get("total_iters")

    print(f"\n{label}\n" + "-" * len(label))
    of_total = f" (of {int(total)})" if total else ""
    print(f"  iterations        : {first['iteration']} -> {last['iteration']}{of_total}")
    if total and stopped_early(train, total):
        print(f"  WARNING           : the run stopped {int(total) - last['iteration']} iterations early")
    print(f"  lm loss           : {first.get('loss'):.4f} -> {last.get('loss'):.4f}")
    if valid:
        print(f"  validation loss   : {valid[0]['loss']:.4f} -> {valid[-1]['loss']:.4f}")
    if vocab_size:
        expected = math.log(vocab_size)
        delta = first.get("loss", 0) - expected
        verdict = (
            "as expected" if abs(delta) < 0.6 else "SUSPICIOUS (tokenizer/data mismatch, or a large init?)"
        )
        print(f"  initial vs ln(V)  : {first.get('loss'):.2f} vs {expected:.2f} -- {verdict}")
    if median_ms:
        print(f"  median s/iter     : {median_ms/1000:.2f}")
    if "tokens_s_gpu" in last:
        print(f"  tokens/s/GPU      : {last['tokens_s_gpu']:.0f} (final instantaneous)")
    if gbs:
        print(f"  global batch size : {int(gbs)}")
    step_tokens = [r["step_tokens"] for r in train if "step_tokens" in r]
    if len(step_tokens) == len(train):
        trained = sum(step_tokens)
        fill = ""
        if gbs and seq_length:
            fill = f" ({trained / (len(train) * gbs * seq_length):.1%} of the token slots)"
        print(f"  tokens trained    : {trained/1e6:.1f}M non-padding{fill}")
    if any("nan" in r or "skipped" in r for r in train):
        nan_total = max((r.get("nan", 0) for r in train), default=0)
        skipped = max((r.get("skipped", 0) for r in train), default=0)
    else:
        nan_total, skipped = sum(1 for r in train if not math.isfinite(r["loss"])), 0
    flag = "" if nan_total == 0 and skipped == 0 else "   <-- investigate"
    print(f"  nan / skipped     : {int(nan_total)} / {int(skipped)}{flag}")
    if "peak_mem_pct" in last:
        risky = "   <-- little headroom" if last["peak_mem_pct"] > 90 else ""
        size = f"{last['peak_mem_gb']:.1f} GB " if "peak_mem_gb" in last else ""
        print(f"  peak memory       : {size}({last['peak_mem_pct']:.1f}%){risky}")
    if "aux_loss" in last:
        print(f"  moe aux loss      : {train[0].get('aux_loss', float('nan')):.3f} -> {last['aux_loss']:.3f}")
    if "grad_norm" in last:
        print(f"  final grad norm   : {last['grad_norm']:.3f}")


def recommend_iterations(
    train, budget_hours, startup_min, seq_length, wall_clock_min=None, global_batch_size=None
):
    """How many iterations fit in a wall-clock budget, from measured step time.

    If the caller measured the wall clock of the probe, start-up is derived
    instead of guessed: everything that was not steady-state stepping, which
    includes the compilation hidden inside the first iterations.
    """
    median_ms = steady_state_ms(train)
    if median_ms is None:
        print("\n[budget] not enough timing data; probe for more iterations")
        return
    median_s = median_ms / 1000.0
    samples = [r for r in train if "elapsed_ms" in r and r["iteration"] > 1]

    source = "assumed"
    if wall_clock_min is not None:
        stepping_min = train[-1]["iteration"] * median_s / 60.0
        startup_min = max(0.0, wall_clock_min - stepping_min)
        source = "measured"

    budget_s = budget_hours * 3600 - startup_min * 60
    # Reserve ~5% for periodic validation and teardown.
    iterations = int(budget_s * 0.95 / median_s)
    gbs = train[-1].get("global_batch_size") or global_batch_size

    print(f"\nbudget for {budget_hours:g} h\n" + "-" * 20)
    print(f"  median step time  : {median_s:.2f} s (from {len(samples)} samples, iteration 1 excluded)")
    print(f"  start-up          : {startup_min:.1f} min ({source})")
    print(f"  recommended iters : {iterations}")
    if gbs and seq_length:
        print(f"  that is           : {iterations*gbs*seq_length/1e6:.0f}M tokens")
    print("  remember to scale the learning-rate schedule to the same length")
    if len(samples) < 10:
        print("  NOTE: very few timing samples; probe at least 20 iterations")
    if any("aux_loss" in r for r in train):
        print("  NOTE: MoE step time keeps falling for ~50 iterations as the router")
        print("        balances, so a short probe under-estimates the iteration count")


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def _window_mean(train, lo, hi):
    values = [r["loss"] for r in train if lo <= r["iteration"] <= hi and math.isfinite(r["loss"])]
    return sum(values) / len(values) if values else None


def compare_to_baseline(train, valid, base_train, base_valid, tolerance):
    """PASS if the end-of-run losses agree with the baseline to within tolerance.

    Single-step training losses are noisy, so the training loss is compared as
    a mean over the last 5% of the common iteration range (at least 10
    iterations). Validation loss is compared at the last common evaluation.
    """
    print("\nbaseline comparison\n" + "-" * 19)
    failures = []
    run_end, base_end = train[-1]["iteration"], base_train[-1]["iteration"]
    if run_end < base_end:
        failures.append(f"the run stopped at iteration {run_end}, the baseline reached {base_end}")
    end = min(run_end, base_end)
    start = max(train[0]["iteration"], base_train[0]["iteration"])
    window = max(10, (end - start) // 20)
    lo = max(start, end - window + 1)
    ours, theirs = _window_mean(train, lo, end), _window_mean(base_train, lo, end)
    if ours is None or theirs is None:
        failures.append(f"no finite training losses in iterations {lo}-{end} to compare")
    else:
        delta = ours - theirs
        print(f"  train loss, iters {lo}-{end}: {ours:.4f} vs {theirs:.4f} (delta {delta:+.4f})")
        if abs(delta) > tolerance:
            failures.append(f"training loss moved by {delta:+.4f}")

    common = sorted({v["iteration"] for v in valid} & {v["iteration"] for v in base_valid})
    if common:
        at = common[-1]
        ours = next(v["loss"] for v in valid if v["iteration"] == at)
        theirs = next(v["loss"] for v in base_valid if v["iteration"] == at)
        delta = ours - theirs
        print(f"  valid loss at iteration {at}: {ours:.4f} vs {theirs:.4f} (delta {delta:+.4f})")
        if abs(delta) > tolerance:
            failures.append(f"validation loss moved by {delta:+.4f}")
    elif valid and base_valid:
        print("  valid loss: no common evaluation iterations")

    verdict = "FAIL" if failures else "PASS"
    print(f"  verdict           : {verdict} (tolerance {tolerance:g})")
    for failure in failures:
        print(f"    - {failure}")
    return not failures


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path, train, valid):
    valid_by_iter = {v["iteration"]: v["loss"] for v in valid}
    iterations = sorted({r["iteration"] for r in train} | set(valid_by_iter))
    train_by_iter = {r["iteration"]: r for r in train}
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for iteration in iterations:
            row = dict(train_by_iter.get(iteration, {"iteration": iteration}))
            row["valid_loss"] = valid_by_iter.get(iteration, "")
            writer.writerow(row)


def token_axis(train, seq_length):
    """(iterations, cumulative tokens) when every record carries its token count.

    MaxText reports the non-padding tokens of every step, which is the honest
    x-axis for a packed or padded run; Megatron's batch is all real tokens.
    """
    if not train or any("step_tokens" not in r for r in train):
        return None
    iterations, cumulative, total = [], [], 0.0
    for record in train:
        total += record["step_tokens"]
        iterations.append(record["iteration"])
        cumulative.append(total)
    return iterations, cumulative


def x_values(iterations, mode, seq_length, gbs, axis=None):
    """Map iteration numbers onto the chosen x-axis.

    Deliberately derived from the iteration number rather than looked up in the
    train records: validation happens at iterations that are not necessarily
    logged (eval_interval need not be a multiple of log_interval), and those
    points would otherwise be silently dropped.
    """
    if mode == "tokens" and axis:
        known, cumulative = axis
        out = []
        for i in iterations:
            index = bisect.bisect_right(known, i) - 1
            out.append(cumulative[max(index, 0)] / 1e9)
        return out
    if mode == "tokens" and seq_length and gbs:
        return [i * gbs * seq_length / 1e9 for i in iterations]
    return list(iterations)


def plot(runs, labels, out_prefix, mode, seq_length, detailed, global_batch_size=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 4 if detailed else 1
    fig, axes = plt.subplots(panels, 1, figsize=(10, 4.2 * panels), squeeze=False)
    axes = axes[:, 0]
    xlabel = "tokens (B)" if mode == "tokens" else "iteration"

    for index, ((train, valid), label) in enumerate(zip(runs, labels)):
        colour = f"C{index}"
        gbs = train[-1].get("global_batch_size") or global_batch_size
        axis = token_axis(train, seq_length)
        axes[0].plot(
            x_values([r["iteration"] for r in train], mode, seq_length, gbs, axis),
            [r.get("loss") for r in train],
            lw=1.3,
            color=colour,
            label=f"{label} train",
        )
        if valid:
            axes[0].plot(
                x_values([v["iteration"] for v in valid], mode, seq_length, gbs, axis),
                [v["loss"] for v in valid],
                "o--",
                ms=3.5,
                lw=1.0,
                color=colour,
                alpha=0.75,
                label=f"{label} valid",
            )
        if detailed:
            for panel, key, title in (
                (axes[1], "lr", "learning rate"),
                (axes[2], "grad_norm", "grad norm"),
                (axes[3], "tokens_s_gpu", "tokens/s/GPU"),
            ):
                series = [(r["iteration"], r[key]) for r in train if key in r]
                if series:
                    panel.plot(
                        x_values([i for i, _ in series], mode, seq_length, gbs, axis),
                        [v for _, v in series],
                        lw=1.2,
                        color=colour,
                        label=label,
                    )
                panel.set_ylabel(title)

    axes[0].set_ylabel("lm loss")
    axes[0].set_title("Primus convergence test")
    for panel in axes:
        panel.set_xlabel(xlabel)
        panel.grid(alpha=0.3)
        if panel.get_legend_handles_labels()[0]:
            panel.legend(fontsize=8)
    fig.tight_layout()
    png = f"{out_prefix}.png"
    fig.savefig(png, dpi=130)
    return png


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("logs", nargs="+", help="Log files, experiment directories, or CSVs from this script")
    parser.add_argument("--labels", nargs="*", help="Legend label per input")
    parser.add_argument("--out", default="loss_curve", help="Output prefix for .png/.csv")
    parser.add_argument("--x", choices=["iterations", "tokens"], default="iterations")
    parser.add_argument("--seq-length", type=int, help="Needed for the tokens x-axis on Megatron runs")
    parser.add_argument(
        "--global-batch-size", type=int, help="For runs whose log does not print it (MaxText)"
    )
    parser.add_argument("--vocab-size", type=int, help="Sanity-check initial loss against ln(V)")
    parser.add_argument("--dataset-info", help="dataset_info.json to read vocab_size from")
    parser.add_argument(
        "--tensorboard",
        nargs="*",
        default=[],
        help="MaxText TensorBoard directory per input (<base_output_directory>/<run_name>)",
    )
    parser.add_argument(
        "--validation-from",
        nargs="*",
        default=[],
        help="Per input, a log or experiment directory of the same run to take validation points from "
        "(Megatron prints them only in the last rank's log)",
    )
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
    parser.add_argument(
        "--baseline", help="Reference run (log, directory or CSV) to compare the first input against"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.05, help="Largest loss difference from --baseline that passes"
    )
    parser.add_argument(
        "--expect-iters", type=int, help="Exit with status 4 if the first input logged fewer iterations"
    )
    args = parser.parse_args()

    vocab_size = args.vocab_size
    if args.dataset_info and not vocab_size:
        vocab_size = json.loads(Path(args.dataset_info).read_text()).get("vocab_size")

    labels = args.labels or [Path(p).name for p in args.logs]
    if len(labels) != len(args.logs):
        raise SystemExit("--labels must give one label per log")
    if len(args.tensorboard) > len(args.logs):
        raise SystemExit("--tensorboard takes at most one directory per log")
    if len(args.validation_from) > len(args.logs):
        raise SystemExit("--validation-from takes at most one source per log")
    if args.x == "tokens" and not args.seq_length:
        print("[plot-loss] --x tokens without --seq-length only works for runs that log their token counts")

    selected = []
    for index, (source, label) in enumerate(zip(args.logs, labels)):
        try:
            path, runs = load_runs(source)
        except NoIterations as exc:
            if index == 0 and args.expect_iters:
                print(f"[plot-loss] FAIL: {exc}; the run trained no iterations")
                raise SystemExit(4) from exc
            raise SystemExit(str(exc)) from exc
        if len(runs) > 1:
            print(f"[plot-loss] {path} holds {len(runs)} runs; using index {args.run_index}")
        train, valid = runs[args.run_index]
        print(f"[plot-loss] {label}: {path} ({len(train)} points, {len(valid)} validations)")
        if index < len(args.tensorboard) and args.tensorboard[index]:
            merge_tensorboard(train, valid, args.tensorboard[index])
        if index < len(args.validation_from) and args.validation_from[index]:
            merge_validation(train, valid, args.validation_from[index])
        selected.append((train, valid))
        summarise(label, train, valid, vocab_size, args.seq_length, args.global_batch_size)
        if args.budget_hours:
            recommend_iterations(
                train,
                args.budget_hours,
                args.startup_min,
                args.seq_length,
                args.wall_clock_min,
                args.global_batch_size,
            )

    passed = True
    plotted, plot_labels = list(selected), list(labels)
    if args.baseline:
        try:
            base_path, base_runs = load_runs(args.baseline)
        except NoIterations as exc:
            raise SystemExit(f"baseline: {exc}") from exc
        print(f"[plot-loss] baseline: {base_path}")
        passed = compare_to_baseline(*selected[0], *base_runs[-1], args.tolerance)
        plotted.append(base_runs[-1])
        plot_labels.append("baseline")

    for (train, valid), label in zip(selected, labels):
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        csv_path = f"{args.out}_{safe}.csv" if len(selected) > 1 else f"{args.out}.csv"
        write_csv(csv_path, train, valid)
        print(f"\n[plot-loss] wrote {csv_path}")

    # The checks above must reach the caller even when the picture cannot be drawn.
    plot_failed = False
    if not args.no_plot:
        try:
            png = plot(
                plotted, plot_labels, args.out, args.x, args.seq_length, args.detailed, args.global_batch_size
            )
            print(f"[plot-loss] wrote {png}")
        except ImportError:
            print("[plot-loss] matplotlib is not installed; wrote the CSV only")
        except Exception as exc:  # noqa: BLE001 - reported through the exit status
            print(f"[plot-loss] plotting failed: {type(exc).__name__}: {exc}")
            plot_failed = True

    # Exit statuses are distinct from 1, which any uncaught error produces.
    missing = stopped_early(selected[0][0], args.expect_iters) if args.expect_iters else 0
    if missing:
        print(
            f"\n[plot-loss] FAIL: the run stopped at iteration {selected[0][0][-1]['iteration']} "
            f"of {args.expect_iters}"
        )
        raise SystemExit(4)
    if not passed:
        raise SystemExit(3)
    if plot_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from statistics import mean

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMUS_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

WARMUP_SKIP = 5

ERROR_STATUS = "Error found - check log"
NONFINITE_STATUS = "Diverged (non-finite loss) - invalid"
OK_STATUS = "OK"

# Per-step metric lines, one pattern per backend log format:
#   torchtitan: "step:  1  loss: 12.78468  grad_norm:     nan  memory: ..."
#   megatron:   "iteration  3/  3 | ... | lm loss: 1.17E+01 | ... | grad norm: 5.885 | ..."
# Same parsing as tests/utils.py:assert_finite_training_metrics.
STEP_METRIC_REGEXES = (
    re.compile(r"\bstep:\s*(?P<step>\d+)\b.*?\bloss:\s*(?P<loss>\S+).*?\bgrad_norm:\s*(?P<grad>\S+)"),
    re.compile(
        r"\biteration\s*(?P<step>\d+)\s*/.*?\blm loss:\s*(?P<loss>\S+).*?\bgrad norm:\s*(?P<grad>\S+)"
    ),
)

ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[0-9;]*m")
LOG_EXIT_CODE_REGEX = re.compile(
    r"(?:AUTO_BENCHMARK_EXIT_CODE=|(?:primus launcher|\[direct\].*torchrun|torchrun) exited with code )(\d+)",
    re.IGNORECASE,
)

LOG_ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"\[ERROR\]", re.IGNORECASE),
    re.compile(r"\bRuntimeError:", re.IGNORECASE),
    re.compile(r"\bOutOfMemoryError:", re.IGNORECASE),
    re.compile(r"\bCUDA out of memory\b", re.IGNORECASE),
    re.compile(r"\bHIP out of memory\b", re.IGNORECASE),
    re.compile(r"\bNCCL error\b", re.IGNORECASE),
    re.compile(r"\bfatal error\b", re.IGNORECASE),
)

LOG_ERROR_EXCLUSIONS = (
    re.compile(r"error_injection", re.IGNORECASE),
    re.compile(r"\[SKIP\].*Import failed", re.IGNORECASE),
    re.compile(r"avoid ImportError", re.IGNORECASE),
    re.compile(r"TORCH_NCCL_ASYNC_ERROR_HANDLING", re.IGNORECASE),
    re.compile(r"destroy_process_group\(\)", re.IGNORECASE),
    re.compile(r"\[CK-JIT\] ERROR:", re.IGNORECASE),
    re.compile(r"\[Patch\].*failed due to missing dependency", re.IGNORECASE),
    re.compile(r"\[Patch\].*Patch '.*' failed", re.IGNORECASE),
    # The triton buffer-store patch logs a caught traceback when triton is
    # absent. Training continues. The header and the exception line are not a
    # failed run; [ERROR] on that patch line is already excluded above.
    re.compile(r"^Traceback \(most recent call last\):"),
    re.compile(r"No module named ['\"]triton['\"]", re.IGNORECASE),
    # primus-cli logs the docker command it would run. That command contains the
    # literal text "[ERROR]: Primus not found" inside a shell test, which is not
    # a failed run. Matching it blanked every successful CSV row.
    re.compile(r"Would execute:.*\[ERROR\]: Primus not found", re.IGNORECASE),
)

# First matching category wins. Patterns stay specific so a normal log line
# (RCCL settings, a caught import, a quoted launcher string) does not become
# the label for a different failure.
ERROR_CLASSIFIERS = (
    (
        "huggingface_auth",
        re.compile(
            r"GatedRepoError|401 Client Error|use_auth_token|huggingface-cli login",
            re.IGNORECASE,
        ),
    ),
    (
        "out_of_memory",
        re.compile(
            r"OutOfMemoryError|\bCUDA out of memory\b|\bHIP out of memory\b|cannot allocate memory",
            re.IGNORECASE,
        ),
    ),
    (
        "distributed_or_network",
        re.compile(
            r"NCCL error|RCCL operation|no socket interface found|connection (?:reset|refused)",
            re.IGNORECASE,
        ),
    ),
    (
        "dataset_or_checkpoint",
        re.compile(
            r"FileNotFoundError|No such file or directory|dataset.*(?:missing|not found)|"
            r"checkpoint.*(?:missing|not found|invalid)",
            re.IGNORECASE,
        ),
    ),
    (
        "dependency_or_import",
        re.compile(r"ModuleNotFoundError|ImportError:|No module named|command not found", re.IGNORECASE),
    ),
    (
        "jax_or_compilation",
        re.compile(r"JaxRuntimeError|LLVM ERROR|hipErrorNoBinary", re.IGNORECASE),
    ),
    (
        "configuration",
        re.compile(r"ValidationError|Config(?:uration)?Error|unrecognized arguments", re.IGNORECASE),
    ),
    ("permission", re.compile(r"PermissionError|Permission denied|Operation not permitted", re.IGNORECASE)),
    ("timeout", re.compile(r"TimeoutError|\btimed out\b|watchdog.*(?:abort|timeout)", re.IGNORECASE)),
)

RUN_LABEL_REGEX = re.compile(r"_run(\d+)\.log$", re.IGNORECASE)
LEGACY_TIMESTAMP_REGEX = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.log$")
LEGACY_SUFFIX_REGEX = re.compile(r"_MI\d+X?_(.+)\.log$", re.IGNORECASE)

ENV_DEFAULT_REGEX = re.compile(r"\$\{[A-Z0-9_]+:(\d+)\}")
PLAIN_INT_REGEX = re.compile(r"^\d+$")

MEGATRON_BS_KEYS = ("micro_batch_size",)
MEGATRON_SEQ_KEYS = ("seq_length",)
MEGATRON_GBS_KEYS = ("global_batch_size",)

TORCHTITAN_BS_KEYS = ("local_batch_size",)
TORCHTITAN_SEQ_KEYS = ("seq_len", "seq_length")
TORCHTITAN_GBS_KEYS = ("global_batch_size",)

MEGATRON_NOTE_TEXT = """
NOTE:
- Results are saved to results/metrics_megatron.csv (latest) and a timestamped snapshot.
- "Run" is run1, run2, ... per model and device, ordered oldest to newest.
- "BS", "Seq", and "GBS" come from the benchmark yaml when available.
- Megatron logs often print each iteration twice (Primus log forwarding); metrics
  deduplicate by iteration number before averaging.
- Timing/throughput fields use the current value before "/" (e.g. 5896.1/5913.1 -> 5896.1).
- Warm-up: the first five iterations are excluded; throughput uses the harmonic mean.
- "Status" is "Error found - check log" when the log contains errors or metrics could not be computed.
- "Status" is "Diverged (non-finite loss) - invalid" when any step logged a NaN/Inf loss or grad
  norm. Such a run exits 0 and still prints throughput, but the numbers are meaningless.
- Numeric fields may contain commas in logs; commas are removed before averaging.
"""

TORCHTITAN_NOTE_TEXT = """
NOTE:
- Results are saved to results/metrics_torchtitan.csv (latest) and a timestamped snapshot.
- "Run" is run1, run2, ... per model and device, ordered oldest to newest.
- "BS", "Seq", and "GBS" come from the benchmark yaml when available.
- "Steps" is the number of training steps used after dropping the first five warm-up steps.
- TPS and TFLOPS use the post-warm-up harmonic mean; memory and MFU use arithmetic means.
- "Status" is "Error found - check log" when the log contains errors or metrics could not be computed.
- "Status" is "Diverged (non-finite loss) - invalid" when any step logged a NaN/Inf loss or grad
  norm. Such a run exits 0 and still prints throughput, but the numbers are meaningless.
"""

MEGATRON_BRIDGE_NOTE_TEXT = """
NOTE:
- Results are saved to results/metrics_megatron_bridge.csv (latest) and a timestamped snapshot.
- "Run" is run1, run2, ... per model and device, ordered oldest to newest.
- "BS", "Seq", and "GBS" come from the benchmark yaml when available.
- Megatron-Bridge logs split iteration timing and MODEL_TFLOP/s/GPU across lines; both are merged.
- Warm-up: the first five iterations are excluded; throughput uses the harmonic mean.
- "Status" is "Error found - check log" when metrics could not be computed from the log.
"""

MLPERF_NOTE_TEXT = """
NOTE:
- Results are saved to results/metrics_mlperf.csv (latest) and a timestamped snapshot.
- Metrics are parsed from :::MLLOG lines emitted at the end of MLPerf timed runs.
- "Run Duration", "Throughput", and "Samples" come from run_duration / overall_throughput / throughput keys.
- llama3.1_8b, gpt_oss_20b, and flux1 use overall_throughput; llama2_70b uses throughput.
- "Status" is "Error found - check log" when MLLOG metrics could not be parsed from the log.
"""

JAX_NOTE_TEXT = """
NOTE:
- Results are saved to results/metrics_<backend>.csv (latest) and a timestamped snapshot.
- Metrics come from MaxText/MaxDiffusion "completed step:" lines and are deduplicated by step.
- The first five steps are excluded as warm-up.
- TFLOPS, Tokens/s, Samples/s, and Frames/s use the harmonic mean when logged.
- "Status" is "Error found - check log" when the run failed or metrics could not be computed.
- "Status" is "Diverged (non-finite loss) - invalid" when a step logged NaN/Inf loss.
"""

MEGATRON_NUM = r"[\d,]+(?:\.\d+)?"
MEGATRON_METRIC_VALUE = rf"({MEGATRON_NUM})(?:\s*/\s*{MEGATRON_NUM})?"

MEGATRON_ITERATION_REGEX = re.compile(
    rf"iteration\s+(\d+)/\s*\d+.*?" rf"elapsed time per iteration \(ms\):\s*{MEGATRON_METRIC_VALUE}.*?"
    # Compute (TFLOP/s/GPU) accepts both the current "compute per GPU (...)" label
    # and the legacy "throughput per GPU (...)" label. Only the instantaneous value
    # is captured; the "(avg Y)" / "/Y" suffix is absorbed by the trailing ".*?".
    rf"(?:compute|throughput) per GPU \(TFLOP/s/GPU\):\s*{MEGATRON_METRIC_VALUE}.*?"
    # Tokens accepts both the current "tokens/s/GPU inst/harmonic mean:" label and
    # the legacy "tokens per GPU (tokens/s/GPU):" label.
    rf"(?:(?:tokens per GPU \(tokens/s/GPU\)|tokens/s/GPU inst/harmonic mean):\s*{MEGATRON_METRIC_VALUE}.*?)?"
    rf"global batch size:\s*(\d+)",
    re.IGNORECASE,
)

MEGATRON_BRIDGE_ITERATION_REGEX = re.compile(
    rf"iteration\s+(\d+)/\s*\d+.*?"
    rf"elapsed time per iteration \(ms\):\s*{MEGATRON_METRIC_VALUE}.*?"
    rf"global batch size:\s*(\d+)",
    re.IGNORECASE,
)

MEGATRON_BRIDGE_TFLOPS_REGEX = re.compile(
    rf"Step Time\s*:\s*[\d.]+\s*s\s+GPU utilization:\s*{MEGATRON_METRIC_VALUE}MODEL_TFLOP/s/GPU",
    re.IGNORECASE,
)

MEGATRON_FILENAME_REGEX = re.compile(r"(?P<model>.+?)_megatron_(?P<device>MI\d+X?)", re.IGNORECASE)
MEGATRON_BRIDGE_FILENAME_REGEX = re.compile(
    r"(?P<model>.+?)_megatron_bridge_(?P<device>MI\d+X?)", re.IGNORECASE
)

TORCHTITAN_STEP_REGEX = re.compile(
    r"step:\s*(\d+).*?"
    r"memory:\s*([\d.]+)GiB.*?"
    r"tps:\s*([\d,]+(?:\.\d+)?).*?"
    r"tflops:\s*([\d,]+(?:\.\d+)?).*?"
    r"mfu:\s*([\d.]+)%"
)

TORCHTITAN_BS_REGEX = re.compile(r"training\.local_batch_size\s*\.{2,}\s*(\d+)")
TORCHTITAN_SEQ_REGEX = re.compile(r"training\.seq_len\s*\.{2,}\s*(\d+)")

TORCHTITAN_FILENAME_REGEX = re.compile(r"(?P<model>.+?)_torchtitan_(?P<device>MI\d+X?)", re.IGNORECASE)

MLPERF_FILENAME_REGEX = re.compile(r"(?P<model>.+?)_mlperf_(?P<device>MI\d+X?)", re.IGNORECASE)
JAX_FILENAME_REGEX_TEMPLATE = r"(?P<model>.+?)_{backend}_(?P<device>MI\d+X?)"
MLPERF_MLLOG_PREFIX = ":::MLLOG"
MLPERF_SFT_SOURCE = "mlperf_sft.py"
MLPERF_PRETRAIN_SOURCE = "mlperf_pretrain_trainer.py"

PRECISION_REGEX = re.compile(r"(BF16|FP8)", re.IGNORECASE)


def _parse_scalar(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value
    if isinstance(value, str):
        stripped = value.strip()
        if PLAIN_INT_REGEX.match(stripped):
            return int(stripped)
        m = ENV_DEFAULT_REGEX.search(stripped)
        if m:
            return int(m.group(1))
    return None


def _deep_find(node, keys):
    if isinstance(node, dict):
        for key in keys:
            if key in node:
                parsed = _parse_scalar(node[key])
                if parsed is not None:
                    return parsed
        for child in node.values():
            found = _deep_find(child, keys)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _deep_find(item, keys)
            if found is not None:
                return found
    return None


def _regex_find(text, keys):
    for key in keys:
        m = re.search(
            rf"^\s*{re.escape(key)}:\s*(.+?)(?:\s+#.*)?$",
            text,
            re.MULTILINE,
        )
        if not m:
            continue
        parsed = _parse_scalar(m.group(1).strip())
        if parsed is not None:
            return parsed
    return None


def load_yaml_dict(path):
    try:
        import yaml
    except ImportError:
        return None

    try:
        with open(path, "r", errors="ignore") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None

    return data if isinstance(data, dict) else None


def resolve_config_yaml(log_fname, backend, device, model, log_dir):
    base = log_fname[:-4] if log_fname.endswith(".log") else log_fname

    for suffix in ("_working", "_override", "_edited", ""):
        candidate = os.path.join(log_dir, f"{base}{suffix}.yaml")
        if os.path.isfile(candidate):
            return candidate

    config_dir = os.path.join(PRIMUS_ROOT, "examples", backend, "configs", device)
    return os.path.join(config_dir, f"{model}.yaml")


def load_training_params(backend, device, model, log_fname, log_dir):
    path = resolve_config_yaml(log_fname, backend, device, model, log_dir)
    if backend in ("megatron", "megatron_bridge"):
        bs_keys, seq_keys, gbs_keys = MEGATRON_BS_KEYS, MEGATRON_SEQ_KEYS, MEGATRON_GBS_KEYS
    else:
        bs_keys, seq_keys, gbs_keys = TORCHTITAN_BS_KEYS, TORCHTITAN_SEQ_KEYS, TORCHTITAN_GBS_KEYS

    bs = seq = gbs = None

    data = load_yaml_dict(path)
    if data is not None:
        bs = _deep_find(data, bs_keys)
        seq = _deep_find(data, seq_keys)
        gbs = _deep_find(data, gbs_keys)

    if os.path.isfile(path) and (bs is None or seq is None or gbs is None):
        with open(path, "r", errors="ignore") as f:
            text = f.read()
        if bs is None:
            bs = _regex_find(text, bs_keys)
        if seq is None:
            seq = _regex_find(text, seq_keys)
        if gbs is None:
            gbs = _regex_find(text, gbs_keys)

    return (
        bs if bs is not None else "-",
        seq if seq is not None else "-",
        gbs if gbs is not None else "-",
    )


def log_chronological_key(fname, path):
    m = RUN_LABEL_REGEX.search(fname)
    if m:
        return (0, int(m.group(1)), fname)

    m = LEGACY_TIMESTAMP_REGEX.search(fname)
    if m:
        return (1, m.group(1), fname)

    m = LEGACY_SUFFIX_REGEX.search(fname)
    if m and m.group(1):
        return (1, m.group(1), fname)

    return (2, os.path.getmtime(path), fname)


def assign_run_labels(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["model"], entry["device"])].append(entry)

    for group in grouped.values():
        group.sort(key=lambda entry: log_chronological_key(entry["fname"], entry["path"]))
        for index, entry in enumerate(group, start=1):
            entry["run"] = f"run{index}"

    return entries


def run_sort_key(run_label):
    m = re.fullmatch(r"run(\d+)", run_label)
    if m:
        return int(m.group(1))
    return 0


def terminal_width(default=120):
    try:
        return shutil.get_terminal_size().columns
    except OSError:
        return default


def print_table(headers, rows, max_col_width=36):
    if not rows:
        return

    widths = []
    for i, header in enumerate(headers):
        col_width = max(len(str(header)), *(len(str(row[i])) for row in rows))
        widths.append(min(col_width, max_col_width))

    def fmt(row):
        cells = []
        for i, value in enumerate(row):
            text = str(value)
            if len(text) > widths[i]:
                text = text[: widths[i] - 3] + "..."
            cells.append(text.ljust(widths[i]))
        return "| " + " | ".join(cells) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))
    print(sep)


SETUP_HEADERS = [
    "Image Digest",
    "Primus Commit",
    "Submodule Pins",
    "GPU",
    "ROCm",
    "Framework Versions",
]
SETUP_KEYS = [
    "image_digest",
    "primus_commit",
    "submodule_pins",
    "gpu",
    "rocm_version",
    "framework_versions",
]


def environment_from_log(path):
    blank = ["-"] * len(SETUP_KEYS)
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            handle.seek(max(0, size - 65536))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return blank
    marker = tail.rfind("===== AUTO BENCHMARK ENVIRONMENT =====")
    if marker < 0:
        return blank
    found = {}
    for line in tail[marker:].splitlines():
        if "=" not in line or line.startswith("="):
            continue
        key, value = line.split("=", 1)
        found[key.strip()] = value.strip() or "-"
    return [found.get(key, "-") for key in SETUP_KEYS]


def save_csv(headers, rows, backend):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    latest_path = os.path.join(RESULTS_DIR, f"metrics_{backend}.csv")
    snapshot_path = os.path.join(RESULTS_DIR, f"metrics_{backend}_{timestamp}.csv")

    for path in (latest_path, snapshot_path):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)

    return latest_path, snapshot_path


def strip_ansi(text):
    return ANSI_ESCAPE_REGEX.sub("", text)


def harmonic_mean(values):
    positive = [value for value in values if value is not None and value > 0]
    if not positive:
        return None
    return len(positive) / sum(1.0 / value for value in positive)


def find_nonfinite_metric(plain):
    if "loss:" not in plain:  # the vast majority of log lines; skip the regexes
        return None
    for pattern in STEP_METRIC_REGEXES:
        m = pattern.search(plain)
        if m is None:
            continue
        for field in ("loss", "grad"):
            raw = m.group(field).rstrip("|,")
            try:
                value = float(raw)
            except ValueError:
                continue
            if not math.isfinite(value):
                return f"step {m.group('step')} {field}={raw}"
        return None
    return None


def scan_log(path):
    """Single pass over a run log for launcher errors and non-finite metrics.

    A diverged run still exits 0 and still prints throughput, so without the
    second check a NaN run reports a (inflated) tps as if it were a valid result.
    Returns (has_error, nonfinite) where nonfinite lists the offending steps.
    """
    exit_code = None
    saw_error_line = False
    nonfinite = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            plain = strip_ansi(line)

            m = LOG_EXIT_CODE_REGEX.search(plain)
            if m:
                exit_code = int(m.group(1))

            bad = find_nonfinite_metric(plain)
            if bad:
                nonfinite.append(bad)

            if any(pattern.search(plain) for pattern in LOG_ERROR_EXCLUSIONS):
                continue

            if any(pattern.search(plain) for pattern in LOG_ERROR_PATTERNS):
                saw_error_line = True

    has_error = True if exit_code not in (None, 0) else saw_error_line
    return has_error, nonfinite, exit_code


def metrics_are_reportable(has_error, nonfinite, stats, exit_code):
    """Keep parsed throughput when the run exited 0.

    A completed run can still contain a caught patch traceback or a quoted
    launcher error string. Those lines must not wipe the CSV. A non-zero exit,
    a missing exit plus an error line, or non-finite loss still blanks the row.
    """
    if nonfinite or not stats:
        return False
    if exit_code not in (None, 0):
        return False
    if has_error and exit_code != 0:
        return False
    return True


def classify_log_error(path):
    matches = set()
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            plain = strip_ansi(line)
            if any(pattern.search(plain) for pattern in LOG_ERROR_EXCLUSIONS):
                continue
            for category, pattern in ERROR_CLASSIFIERS:
                if pattern.search(plain):
                    matches.add(category)
    for category, _ in ERROR_CLASSIFIERS:
        if category in matches:
            return category
    return "runtime_or_incomplete"


def failed_status(path, has_error, nonfinite, has_metrics, exit_code=None):
    if nonfinite and not has_error:
        return NONFINITE_STATUS
    if exit_code == 137:
        return f"{ERROR_STATUS} (killed_sigkill)"
    if exit_code == 143:
        return f"{ERROR_STATUS} (killed_sigterm)"
    if has_error:
        return f"{ERROR_STATUS} ({classify_log_error(path)})"
    if not has_metrics:
        return f"{ERROR_STATUS} (incomplete_metrics)"
    return OK_STATUS


def render_results(backend, headers, rows, note_text):
    if not rows:
        print(f"No {backend} logs found.")
        return None

    latest_path, snapshot_path = save_csv(headers, rows, backend)
    print_table(headers, rows)

    error_rows = [row for row in rows if len(row) > 2 and str(row[2]).startswith(ERROR_STATUS)]
    if error_rows:
        print(
            f"\n{len(error_rows)} run(s) reported errors or incomplete metrics."
            " Open the corresponding log file for details."
        )

    diverged_rows = [row for row in rows if len(row) > 2 and str(row[2]).startswith(NONFINITE_STATUS)]
    if diverged_rows:
        labels = ", ".join(f"{row[0]} {row[1]}" for row in diverged_rows)
        print(
            f"\n{len(diverged_rows)} run(s) logged a non-finite loss or grad norm and are"
            f" reported without throughput: {labels}"
        )

    table_width = sum(len(str(header)) for header in headers) + 3 * len(headers)
    if table_width > terminal_width():
        print("\n(Table may be wider than the terminal. Open the CSV for the full view.)")

    print(note_text)
    print("\nMetrics saved to:")
    print(f"  Latest:   {latest_path}")
    print(f"  Snapshot: {snapshot_path}")

    return latest_path


def is_megatron(filename):
    name = filename.lower()
    if "megatron_bridge" in name:
        return False
    return "megatron" in name or "megatorn" in name


def is_megatron_bridge(filename):
    return "megatron_bridge" in filename.lower()


def megatron_parse_filename(filename):
    m = MEGATRON_FILENAME_REGEX.search(filename)
    if not m:
        return None

    p = PRECISION_REGEX.search(filename)
    precision = p.group(1).upper() if p else "-"

    return {
        "model": m.group("model"),
        "device": m.group("device"),
        "precision": precision,
    }


def megatron_bridge_parse_filename(filename):
    m = MEGATRON_BRIDGE_FILENAME_REGEX.search(filename)
    if not m:
        return None

    p = PRECISION_REGEX.search(filename)
    precision = p.group(1).upper() if p else "-"

    return {
        "model": m.group("model"),
        "device": m.group("device"),
        "precision": precision,
    }


def megatron_to_float(num_str):
    return float(num_str.replace(",", ""))


def megatron_parse_log_file(path):
    records_by_iter = {}

    with open(path, "r", errors="ignore") as f:
        for line in f:
            plain = strip_ansi(line)
            if "iteration" not in plain:
                continue

            m = MEGATRON_ITERATION_REGEX.search(plain)
            if not m:
                continue

            iter_num = int(m.group(1))
            if iter_num in records_by_iter:
                continue

            records_by_iter[iter_num] = {
                "iter": iter_num,
                "elapsed_ms": megatron_to_float(m.group(2)),
                "tflops_gpu": megatron_to_float(m.group(3)),
                "tokens_gpu": megatron_to_float(m.group(4)) if m.group(4) else None,
                "gbs": int(m.group(5)),
            }

    return [records_by_iter[i] for i in sorted(records_by_iter)]


def megatron_bridge_parse_log_file(path):
    records_by_iter = {}
    pending_tflops = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            plain = strip_ansi(line)

            tm = MEGATRON_BRIDGE_TFLOPS_REGEX.search(plain)
            if tm:
                pending_tflops.append(megatron_to_float(tm.group(1)))
                continue

            if "iteration" not in plain:
                continue

            m = MEGATRON_BRIDGE_ITERATION_REGEX.search(plain)
            if not m:
                continue

            iter_num = int(m.group(1))
            if iter_num in records_by_iter:
                continue

            tflops_gpu = pending_tflops.pop(0) if pending_tflops else None
            records_by_iter[iter_num] = {
                "iter": iter_num,
                "elapsed_ms": megatron_to_float(m.group(2)),
                "tflops_gpu": tflops_gpu,
                "tokens_gpu": None,
                "gbs": int(m.group(3)),
            }

    return [records_by_iter[i] for i in sorted(records_by_iter)]


def megatron_compute_averages(records):
    if len(records) <= WARMUP_SKIP:
        return None

    records = records[WARMUP_SKIP:]
    token_values = [r["tokens_gpu"] for r in records if r["tokens_gpu"] is not None]
    tflops_values = [r["tflops_gpu"] for r in records if r["tflops_gpu"] is not None]

    return {
        "count": len(records),
        "elapsed_ms": mean(r["elapsed_ms"] for r in records),
        "tflops_gpu": harmonic_mean(tflops_values),
        "tokens_gpu": harmonic_mean(token_values),
        "gbs": records[0]["gbs"],
    }


def megatron_collect_rows(
    backend="megatron",
    log_subdir=None,
    parse_filename=None,
    filename_filter=None,
    parse_log_file=None,
    include_tokens=True,
):
    if log_subdir is None:
        log_subdir = f"logs_{backend}"
    if parse_filename is None:
        parse_filename = megatron_parse_filename
    if filename_filter is None:
        filename_filter = is_megatron
    if parse_log_file is None:
        parse_log_file = megatron_parse_log_file

    log_dir = os.path.join(RESULTS_DIR, log_subdir)
    entries = []

    if not os.path.isdir(log_dir):
        return []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log") or not filename_filter(fname):
            continue

        meta = parse_filename(fname)
        if not meta:
            continue

        path = os.path.join(log_dir, fname)
        has_error, nonfinite, exit_code = scan_log(path)
        records = parse_log_file(path)
        stats = megatron_compute_averages(records)

        bs, seq, gbs = load_training_params(backend, meta["device"], meta["model"], fname, log_dir)

        if not metrics_are_reportable(has_error, nonfinite, stats, exit_code):
            if gbs == "-" and records:
                gbs = records[0]["gbs"]
            values = [
                failed_status(path, has_error, nonfinite, bool(stats), exit_code),
                backend,
                meta["device"],
                bs,
                seq,
                gbs,
                meta["precision"],
                "-",
                "-",
                "-",
            ]
            if include_tokens:
                values.append("-")
        else:
            if gbs == "-":
                gbs = stats["gbs"]

            tflops_gpu = f"{stats['tflops_gpu']:.2f}" if stats["tflops_gpu"] is not None else "-"
            values = [
                OK_STATUS,
                backend,
                meta["device"],
                bs,
                seq,
                gbs,
                meta["precision"],
                stats["count"],
                f"{stats['elapsed_ms']:.2f}",
                tflops_gpu,
            ]
            if include_tokens:
                tokens_gpu = f"{stats['tokens_gpu']:.2f}" if stats["tokens_gpu"] is not None else "-"
                values.append(tokens_gpu)

        entries.append(
            {
                "model": meta["model"],
                "device": meta["device"],
                "fname": fname,
                "path": path,
                "values": values,
            }
        )

    assign_run_labels(entries)

    rows = [
        [entry["model"], entry["run"], *entry["values"], *environment_from_log(entry["path"])]
        for entry in entries
    ]
    rows.sort(key=lambda row: (row[0], run_sort_key(row[1]), row[7]))
    return rows


def megatron_main():
    headers = [
        "Model",
        "Run",
        "Status",
        "Backend",
        "Device",
        "BS",
        "Seq",
        "GBS",
        "Precision",
        "Iterations",
        "Iter Time (ms)",
        "HMean TFLOPS/GPU",
        "HMean Tokens/GPU",
        *SETUP_HEADERS,
    ]
    render_results("megatron", headers, megatron_collect_rows(), MEGATRON_NOTE_TEXT)


def megatron_bridge_main():
    headers = [
        "Model",
        "Run",
        "Status",
        "Backend",
        "Device",
        "BS",
        "Seq",
        "GBS",
        "Precision",
        "Iterations",
        "Iter Time (ms)",
        "HMean TFLOPS/GPU",
        *SETUP_HEADERS,
    ]
    render_results(
        "megatron_bridge",
        headers,
        megatron_collect_rows(
            backend="megatron_bridge",
            log_subdir="logs_megatron_bridge",
            parse_filename=megatron_bridge_parse_filename,
            filename_filter=is_megatron_bridge,
            parse_log_file=megatron_bridge_parse_log_file,
            include_tokens=False,
        ),
        MEGATRON_BRIDGE_NOTE_TEXT,
    )


def is_mlperf(filename):
    return "_mlperf_" in filename.lower()


def mlperf_parse_filename(filename):
    m = MLPERF_FILENAME_REGEX.search(filename)
    if not m:
        return None

    return {
        "model": m.group("model"),
        "device": m.group("device"),
    }


def mlperf_parse_mllog_line(line):
    idx = line.find(MLPERF_MLLOG_PREFIX)
    if idx == -1:
        return None

    payload = line[idx + len(MLPERF_MLLOG_PREFIX) :].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def mlperf_event_source(event):
    metadata = event.get("metadata") or {}
    file_path = metadata.get("file") or ""
    if MLPERF_SFT_SOURCE in file_path:
        return "sft"
    if MLPERF_PRETRAIN_SOURCE in file_path:
        return "pretrain"
    return None


def mlperf_model_metric_profile(model):
    model_name = (model or "").lower()
    if "llama2" in model_name:
        return "sft"
    if "llama3" in model_name or "gpt_oss" in model_name or "flux" in model_name:
        return "pretrain"
    return None


def mlperf_parse_log_file(path, model=None):
    profile = mlperf_model_metric_profile(model)
    buckets = {
        "sft": {"run_duration": None, "throughput": None, "samples": None},
        "pretrain": {"run_duration": None, "throughput": None, "samples": None},
    }

    with open(path, "r", errors="ignore") as f:
        for line in f:
            plain = strip_ansi(line)
            event = mlperf_parse_mllog_line(plain)
            if not event:
                continue

            source = mlperf_event_source(event)
            if source not in buckets:
                if profile in buckets:
                    source = profile
                else:
                    continue

            bucket = buckets[source]
            key = event.get("key")
            value = event.get("value")
            metadata = event.get("metadata") or {}

            if key == "run_duration":
                bucket["run_duration"] = value
            elif key == "throughput" and source == "sft":
                bucket["throughput"] = value
            elif key == "overall_throughput" and source == "pretrain":
                bucket["throughput"] = value

            if key in ("throughput", "overall_throughput", "run_duration"):
                if metadata.get("samples") is not None:
                    bucket["samples"] = metadata.get("samples")

    if profile == "sft":
        chosen = buckets["sft"]
    elif profile == "pretrain":
        chosen = buckets["pretrain"]
    elif buckets["sft"]["throughput"] is not None:
        chosen = buckets["sft"]
    else:
        chosen = buckets["pretrain"]

    if chosen["run_duration"] is None and chosen["throughput"] is None and chosen["samples"] is None:
        return None

    return {
        "run_duration": chosen["run_duration"] if chosen["run_duration"] is not None else "-",
        "throughput": chosen["throughput"] if chosen["throughput"] is not None else "-",
        "samples": chosen["samples"] if chosen["samples"] is not None else "-",
    }


def mlperf_collect_rows():
    log_dir = os.path.join(RESULTS_DIR, "logs_mlperf")
    entries = []

    if not os.path.isdir(log_dir):
        return []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log") or not is_mlperf(fname):
            continue

        meta = mlperf_parse_filename(fname)
        if not meta:
            continue

        path = os.path.join(log_dir, fname)
        has_error, nonfinite, exit_code = scan_log(path)
        stats = mlperf_parse_log_file(path, model=meta["model"])

        if not metrics_are_reportable(has_error, nonfinite, stats, exit_code):
            values = [
                failed_status(path, has_error, nonfinite, bool(stats), exit_code),
                "mlperf",
                meta["device"],
                "-",
                "-",
                "-",
            ]
        else:
            values = [
                OK_STATUS,
                "mlperf",
                meta["device"],
                stats["samples"],
                stats["run_duration"],
                stats["throughput"],
            ]

        entries.append(
            {
                "model": meta["model"],
                "device": meta["device"],
                "fname": fname,
                "path": path,
                "values": values,
            }
        )

    assign_run_labels(entries)

    rows = [
        [entry["model"], entry["run"], *entry["values"], *environment_from_log(entry["path"])]
        for entry in entries
    ]
    rows.sort(key=lambda row: (row[0], run_sort_key(row[1]), str(row[5])))
    return rows


def mlperf_main():
    headers = [
        "Model",
        "Run",
        "Status",
        "Backend",
        "Device",
        "Samples",
        "Run Duration",
        "Throughput",
        *SETUP_HEADERS,
    ]
    render_results("mlperf", headers, mlperf_collect_rows(), MLPERF_NOTE_TEXT)


def is_torchtitan(filename):
    return "torchtitan" in filename.lower()


def torchtitan_parse_filename(filename):
    m = TORCHTITAN_FILENAME_REGEX.search(filename)
    if not m:
        return None

    p = PRECISION_REGEX.search(filename)
    precision = p.group(1).upper() if p else "-"

    return {
        "model": m.group("model"),
        "device": m.group("device"),
        "precision": precision,
    }


def torchtitan_parse_log_file(path):
    steps_by_num = {}

    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = TORCHTITAN_STEP_REGEX.search(line)
            if not m:
                continue

            step_num = int(m.group(1))
            if step_num in steps_by_num:
                continue

            steps_by_num[step_num] = {
                "step": step_num,
                "memory": float(m.group(2)),
                "tps": float(m.group(3).replace(",", "")),
                "tflops": float(m.group(4).replace(",", "")),
                "mfu": float(m.group(5)),
            }

    return [steps_by_num[i] for i in sorted(steps_by_num)]


def torchtitan_parse_log_training_fallback(path):
    bs = None
    seq = None

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if bs is None:
                m = TORCHTITAN_BS_REGEX.search(line)
                if m:
                    bs = int(m.group(1))
            if seq is None:
                m = TORCHTITAN_SEQ_REGEX.search(line)
                if m:
                    seq = int(m.group(1))
            if bs is not None and seq is not None:
                break

    return bs, seq


def torchtitan_compute_averages(steps):
    if len(steps) <= WARMUP_SKIP:
        return None

    steps = steps[WARMUP_SKIP:]

    return {
        "count": len(steps),
        "memory": mean(s["memory"] for s in steps),
        "tps": harmonic_mean([s["tps"] for s in steps]),
        "tflops": harmonic_mean([s["tflops"] for s in steps]),
        "mfu": mean(s["mfu"] for s in steps),
    }


def torchtitan_collect_rows():
    log_dir = os.path.join(RESULTS_DIR, "logs_torchtitan")
    entries = []

    if not os.path.isdir(log_dir):
        return []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log") or not is_torchtitan(fname):
            continue

        meta = torchtitan_parse_filename(fname)
        if not meta:
            continue

        path = os.path.join(log_dir, fname)
        has_error, nonfinite, exit_code = scan_log(path)
        steps = torchtitan_parse_log_file(path)
        stats = torchtitan_compute_averages(steps)

        bs, seq, gbs = load_training_params("torchtitan", meta["device"], meta["model"], fname, log_dir)
        if bs == "-":
            log_bs, _ = torchtitan_parse_log_training_fallback(path)
            if log_bs is not None:
                bs = log_bs
        if seq == "-":
            _, log_seq = torchtitan_parse_log_training_fallback(path)
            if log_seq is not None:
                seq = log_seq

        if not metrics_are_reportable(has_error, nonfinite, stats, exit_code):
            values = [
                failed_status(path, has_error, nonfinite, bool(stats), exit_code),
                "torchtitan",
                meta["device"],
                bs,
                seq,
                gbs,
                meta["precision"],
                "-",
                "-",
                "-",
                "-",
                "-",
            ]
        else:
            values = [
                OK_STATUS,
                "torchtitan",
                meta["device"],
                bs,
                seq,
                gbs,
                meta["precision"],
                stats["count"],
                f"{stats['memory']:.2f}",
                f"{stats['tps']:.2f}",
                f"{stats['tflops']:.2f}",
                f"{stats['mfu']:.2f}",
            ]

        entries.append(
            {
                "model": meta["model"],
                "device": meta["device"],
                "fname": fname,
                "path": path,
                "values": values,
            }
        )

    assign_run_labels(entries)

    rows = [
        [entry["model"], entry["run"], *entry["values"], *environment_from_log(entry["path"])]
        for entry in entries
    ]
    rows.sort(key=lambda row: (row[0], run_sort_key(row[1]), row[7]))
    return rows


def torchtitan_main():
    headers = [
        "Model",
        "Run",
        "Status",
        "Backend",
        "Device",
        "BS",
        "Seq",
        "GBS",
        "Precision",
        "Steps",
        "Mem(GiB)",
        "HMean TPS",
        "HMean TFLOPS",
        "MFU(%)",
        *SETUP_HEADERS,
    ]
    render_results("torchtitan", headers, torchtitan_collect_rows(), TORCHTITAN_NOTE_TEXT)


def jax_parse_filename(filename, backend):
    pattern = re.compile(JAX_FILENAME_REGEX_TEMPLATE.format(backend=re.escape(backend)), re.IGNORECASE)
    match = pattern.search(filename)
    if not match:
        return None
    return {"model": match.group("model"), "device": match.group("device")}


def jax_parse_log_file(path):
    records_by_step = {}
    nonfinite = []

    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            plain = strip_ansi(line)
            if "completed step:" not in plain or "TFLOP/s/device" not in plain:
                continue

            step_match = re.search(r"completed step:\s*(\d+)", plain, re.IGNORECASE)
            seconds_match = re.search(r"seconds:\s*([\d.]+)", plain, re.IGNORECASE)
            tflops_match = re.search(r"TFLOP/s/device:\s*([\d.]+)", plain, re.IGNORECASE)
            loss_match = re.search(r"loss:\s*([^,\s]+)", plain, re.IGNORECASE)
            if not (step_match and seconds_match and tflops_match):
                continue

            step = int(step_match.group(1))
            record = {
                "step": step,
                "seconds": float(seconds_match.group(1)),
                "tflops": float(tflops_match.group(1)),
            }

            optional_fields = (
                ("tokens", r"Tokens/s/device:\s*([\d,.]+)"),
                ("samples", r"Samples/s/device:\s*([\d,.]+)"),
                ("frames", r"Frames/s/device:\s*([\d,.]+)"),
            )
            for field, pattern in optional_fields:
                match = re.search(pattern, plain, re.IGNORECASE)
                if match:
                    record[field] = float(match.group(1).replace(",", ""))

            if loss_match:
                raw_loss = loss_match.group(1).rstrip("|,")
                try:
                    record["loss"] = float(raw_loss)
                    if not math.isfinite(record["loss"]):
                        nonfinite.append(f"step {step} loss={raw_loss}")
                except ValueError:
                    pass

            records_by_step[step] = record

    return [records_by_step[step] for step in sorted(records_by_step)], nonfinite


def jax_compute_averages(records):
    if len(records) <= WARMUP_SKIP:
        return None

    records = records[WARMUP_SKIP:]
    result = {
        "count": len(records),
        "seconds": mean(record["seconds"] for record in records),
        "tflops": mean(record["tflops"] for record in records),
    }
    for field in ("tokens", "samples", "frames", "loss"):
        values = [record[field] for record in records if field in record]
        if field in ("tokens", "samples", "frames"):
            result[field] = harmonic_mean(values)
        else:
            result[field] = mean(values) if values else None
    return result


def jax_collect_rows(backend):
    log_dir = os.path.join(RESULTS_DIR, f"logs_{backend}")
    entries = []
    if not os.path.isdir(log_dir):
        return []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log") or f"_{backend}_" not in fname.lower():
            continue

        meta = jax_parse_filename(fname, backend)
        if not meta:
            continue

        path = os.path.join(log_dir, fname)
        has_error, scanned_nonfinite, exit_code = scan_log(path)
        records, jax_nonfinite = jax_parse_log_file(path)
        nonfinite = scanned_nonfinite + jax_nonfinite
        stats = jax_compute_averages(records)
        precision_match = PRECISION_REGEX.search(meta["model"])
        precision = precision_match.group(1).upper() if precision_match else "-"

        if not metrics_are_reportable(has_error, nonfinite, stats, exit_code):
            values = [
                failed_status(path, has_error, nonfinite, bool(stats), exit_code),
                backend,
                meta["device"],
                precision,
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
            ]
        else:

            def metric(field, digits):
                value = stats[field]
                return f"{value:.{digits}f}" if value is not None else "-"

            values = [
                OK_STATUS,
                backend,
                meta["device"],
                precision,
                stats["count"],
                metric("seconds", 3),
                metric("tflops", 2),
                metric("tokens", 2),
                metric("samples", 4),
                metric("frames", 2),
                metric("loss", 4),
            ]

        entries.append(
            {
                "model": meta["model"],
                "device": meta["device"],
                "fname": fname,
                "path": path,
                "values": values,
            }
        )

    assign_run_labels(entries)
    rows = [
        [entry["model"], entry["run"], *entry["values"], *environment_from_log(entry["path"])]
        for entry in entries
    ]
    rows.sort(key=lambda row: (row[0], run_sort_key(row[1]), row[4]))
    return rows


def jax_main(backend):
    headers = [
        "Model",
        "Run",
        "Status",
        "Backend",
        "Device",
        "Precision",
        "Steps",
        "Step Time(s)",
        "HMean TFLOPS/Device",
        "HMean Tokens/s/Device",
        "HMean Samples/s/Device",
        "HMean Frames/s/Device",
        "Loss",
        *SETUP_HEADERS,
    ]
    render_results(backend, headers, jax_collect_rows(backend), JAX_NOTE_TEXT)


def maxtext_main():
    jax_main("maxtext")


def maxdiffusion_main():
    jax_main("maxdiffusion")


BACKENDS = {
    "maxdiffusion": maxdiffusion_main,
    "maxtext": maxtext_main,
    "megatron": megatron_main,
    "megatron_bridge": megatron_bridge_main,
    "mlperf": mlperf_main,
    "torchtitan": torchtitan_main,
}


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark metrics tables and CSVs.")
    parser.add_argument(
        "backend",
        choices=sorted(BACKENDS),
        help="Training backend to process logs for",
    )
    args = parser.parse_args()
    BACKENDS[args.backend]()


if __name__ == "__main__":
    main()

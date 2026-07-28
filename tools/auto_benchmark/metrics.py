#!/usr/bin/env python3

import argparse
import csv
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
OK_STATUS = "OK"

ANSI_ESCAPE_REGEX = re.compile(r"\x1b\[[0-9;]*m")
LOG_EXIT_CODE_REGEX = re.compile(r"primus launcher exited with code (\d+)", re.IGNORECASE)

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
- Warm-up: the first five iterations are excluded before averaging.
- "Status" is "Error found - check log" when the log contains errors or metrics could not be computed.
- Numeric fields may contain commas in logs; commas are removed before averaging.
"""

TORCHTITAN_NOTE_TEXT = """
NOTE:
- Results are saved to results/metrics_torchtitan.csv (latest) and a timestamped snapshot.
- "Run" is run1, run2, ... per model and device, ordered oldest to newest.
- "BS", "Seq", and "GBS" come from the benchmark yaml when available.
- "Steps" is the number of training steps used after dropping the first five warm-up steps.
- TPS and TFLOPS values may contain commas in logs; commas are removed before averaging.
- "Status" is "Error found - check log" when the log contains errors or metrics could not be computed.
"""

MEGATRON_NUM = r"[\d,]+(?:\.\d+)?"
MEGATRON_METRIC_VALUE = rf"({MEGATRON_NUM})(?:\s*/\s*{MEGATRON_NUM})?"

MEGATRON_ITERATION_REGEX = re.compile(
    rf"iteration\s+(\d+)/\s*\d+.*?"
    rf"elapsed time per iteration \(ms\):\s*{MEGATRON_METRIC_VALUE}.*?"
    rf"throughput per GPU \(TFLOP/s/GPU\):\s*{MEGATRON_METRIC_VALUE}.*?"
    rf"(?:tokens per GPU \(tokens/s/GPU\):\s*{MEGATRON_METRIC_VALUE}.*?)?"
    rf"global batch size:\s*(\d+)",
    re.IGNORECASE,
)

MEGATRON_FILENAME_REGEX = re.compile(r"(?P<model>.+?)_megatron_(?P<device>MI\d+X?)", re.IGNORECASE)

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

    for suffix in ("_override", "_edited", ""):
        candidate = os.path.join(log_dir, f"{base}{suffix}.yaml")
        if os.path.isfile(candidate):
            return candidate

    config_dir = os.path.join(PRIMUS_ROOT, "examples", backend, "configs", device)
    return os.path.join(config_dir, f"{model}.yaml")


def load_training_params(backend, device, model, log_fname, log_dir):
    path = resolve_config_yaml(log_fname, backend, device, model, log_dir)
    if backend == "megatron":
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


def log_has_error(path):
    exit_code = None
    saw_error_line = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            plain = strip_ansi(line)

            m = LOG_EXIT_CODE_REGEX.search(plain)
            if m:
                exit_code = int(m.group(1))

            if any(pattern.search(plain) for pattern in LOG_ERROR_EXCLUSIONS):
                continue

            if any(pattern.search(plain) for pattern in LOG_ERROR_PATTERNS):
                saw_error_line = True

    if exit_code not in (None, 0):
        return True
    return saw_error_line


def render_results(backend, headers, rows, note_text):
    if not rows:
        print(f"No {backend} logs found.")
        return None

    latest_path, snapshot_path = save_csv(headers, rows, backend)
    print_table(headers, rows)

    error_rows = [row for row in rows if len(row) > 2 and row[2] == ERROR_STATUS]
    if error_rows:
        print(
            f"\n{len(error_rows)} run(s) reported errors or incomplete metrics."
            " Open the corresponding log file for details."
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
    return "megatron" in name or "megatorn" in name


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


def megatron_to_float(num_str):
    return float(num_str.replace(",", ""))


def megatron_parse_log_file(path):
    records_by_iter = {}

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "iteration" not in line:
                continue

            m = MEGATRON_ITERATION_REGEX.search(line)
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


def megatron_compute_averages(records):
    if len(records) <= WARMUP_SKIP:
        return None

    records = records[WARMUP_SKIP:]
    token_values = [r["tokens_gpu"] for r in records if r["tokens_gpu"] is not None]

    return {
        "count": len(records),
        "elapsed_ms": mean(r["elapsed_ms"] for r in records),
        "tflops_gpu": mean(r["tflops_gpu"] for r in records),
        "tokens_gpu": mean(token_values) if token_values else None,
        "gbs": records[0]["gbs"],
    }


def megatron_collect_rows():
    log_dir = os.path.join(RESULTS_DIR, "logs_megatron")
    entries = []

    if not os.path.isdir(log_dir):
        return []

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log") or not is_megatron(fname):
            continue

        meta = megatron_parse_filename(fname)
        if not meta:
            continue

        path = os.path.join(log_dir, fname)
        has_error = log_has_error(path)
        records = megatron_parse_log_file(path)
        stats = megatron_compute_averages(records)

        bs, seq, gbs = load_training_params("megatron", meta["device"], meta["model"], fname, log_dir)

        if has_error or not stats:
            if gbs == "-" and records:
                gbs = records[0]["gbs"]
            values = [
                ERROR_STATUS,
                "megatron",
                meta["device"],
                bs,
                seq,
                gbs,
                meta["precision"],
                "-",
                "-",
                "-",
                "-",
            ]
        else:
            if gbs == "-":
                gbs = stats["gbs"]

            tokens_gpu = f"{stats['tokens_gpu']:.2f}" if stats["tokens_gpu"] is not None else "-"
            values = [
                OK_STATUS,
                "megatron",
                meta["device"],
                bs,
                seq,
                gbs,
                meta["precision"],
                stats["count"],
                f"{stats['elapsed_ms']:.2f}",
                f"{stats['tflops_gpu']:.2f}",
                tokens_gpu,
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

    rows = [[entry["model"], entry["run"], *entry["values"]] for entry in entries]
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
        "TFLOPS/GPU",
        "Tokens/GPU",
    ]
    render_results("megatron", headers, megatron_collect_rows(), MEGATRON_NOTE_TEXT)


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
        "tps": mean(s["tps"] for s in steps),
        "tflops": mean(s["tflops"] for s in steps),
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
        has_error = log_has_error(path)
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

        if has_error or not stats:
            values = [
                ERROR_STATUS,
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

    rows = [[entry["model"], entry["run"], *entry["values"]] for entry in entries]
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
        "TPS",
        "TFLOPS",
        "MFU(%)",
    ]
    render_results("torchtitan", headers, torchtitan_collect_rows(), TORCHTITAN_NOTE_TEXT)


BACKENDS = {
    "megatron": megatron_main,
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

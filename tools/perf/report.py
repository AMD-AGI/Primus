#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Turn perf batch results into an Excel report.

Usage::

    python3 tools/perf/report.py RESULT_DIR [RESULT_DIR ...] [-o report.xlsx]
                                 [--baseline BASELINE_DIR ...] [--threshold 5]

Reads each directory's ``results.json`` (or, when that is missing, older than
the logs or from an older parser, parses the logs itself with
extract_results.py, without writing anything) and its
``batch_summary_<stamp>.txt`` files, and writes ``report.xlsx`` (into the
result directory when there is one) with these sheets:

* Overview     per batch: outcome, image, code, the platform (a one-line
               system description, then server, BIOS, CPU, memory, GPU and
               driver details), host settings, run settings and run counts.
* Performance  one row per config (backend, model, precision, config hash,
               sequence length, MBS, GBS, nodes, GPUs, image), with the
               averages over its successful runs of TFLOP/s per GPU and
               throughput (tokens, samples or frames per second, or DLRM
               recommendations). With --baseline, the baseline's numbers and
               the change.
* Runs         every run, as an Excel table: the source for PivotTables.
* Chart        TFLOP/s per GPU for each config.

Needs pandas and XlsxWriter. A copy installed in ~/.cache/primus-perf/
python<X.Y>-<platform> is used first; when one is missing or unusable, it is
installed there (no root needed, nothing else changes) unless --no-install
is given.

Exit codes: 0 written, 1 error, 3 no runs to report, 4 pandas or XlsxWriter
missing.
"""

# Keeps the annotations below unevaluated, for hosts whose python3 predates 3.10.
from __future__ import annotations

import argparse
import glob
import importlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import sysconfig
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Per Python version and platform: pandas and numpy are compiled for one, and
# a home directory can be shared by hosts of different kinds.
PRIVATE_SITE = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "primus-perf",
    f"python{sys.version_info[0]}.{sys.version_info[1]}-{sysconfig.get_platform()}",
)
REQUIREMENTS = (("pandas", "pandas>=1.3"), ("xlsxwriter", "XlsxWriter>=3.0.2,<4"))
MIN_PANDAS = (1, 3)
# Set when the script re-runs itself after installing over a half-imported library.
REEXEC_ENV = "PRIMUS_PERF_REPORT_REEXEC"

EXIT_OK, EXIT_ERROR, EXIT_NO_RUNS, EXIT_NO_DEPENDENCIES = 0, 1, 3, 4

BACKEND_ORDER = (
    "megatron",
    "megatron_bridge",
    "torchtitan",
    "maxtext",
    "maxdiffusion",
    "diffusion",
    "dlrm",
)
PRECISION_ORDER = ("BF16", "FP16", "FP8", "NANOO_FP8", "MXFP8", "FP4", "MXFP4", "TF32", "FP32")
SECTION_ORDER = ("Batch", "Image and code", "Hardware", "Run settings", "Runs")
# Run-to-run variation above this is highlighted.
CV_WARN = 0.03

# Set by load_dependencies.
pd = None
xlsxwriter = None


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _version(text: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", text)[:2])


def _import_all() -> list[str]:
    """Import the requirements into the module globals; return the ones that
    are missing, broken (e.g. pandas built against another numpy) or too old."""
    missing = []
    for module, requirement in REQUIREMENTS:
        try:
            imported = importlib.import_module(module)
        except Exception:
            missing.append(requirement)
            continue
        if module == "pandas" and _version(getattr(imported, "__version__", "0")) < MIN_PANDAS:
            missing.append(requirement)
            continue
        globals()["pd" if module == "pandas" else module] = imported
    return missing


def _use_private_site():
    if PRIVATE_SITE not in sys.path:
        sys.path.insert(0, PRIVATE_SITE)
    importlib.invalidate_caches()


def load_dependencies(allow_install: bool) -> bool:
    """Import pandas and XlsxWriter, the private copy first, installing it
    when something is missing and that is allowed."""
    if os.path.isdir(PRIVATE_SITE):
        _use_private_site()
    missing = _import_all()
    if not missing:
        return True
    if not allow_install or os.environ.get(REEXEC_ENV):
        return False
    print(f"[INFO] Installing {', '.join(missing)} into {PRIVATE_SITE}", file=sys.stderr)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "--no-input",
        "--upgrade",
        "--target",
        PRIVATE_SITE,
        *missing,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[ERROR] could not install {', '.join(missing)} ({e}).", file=sys.stderr)
        print(
            f"        Install them with pip ({' '.join(m.split('>')[0] for m in missing)}) into a Python of your own,\n"
            "        e.g. python3 -m venv ~/venv-perf && ~/venv-perf/bin/pip install pandas XlsxWriter,\n"
            "        then run this script with ~/venv-perf/bin/python.",
            file=sys.stderr,
        )
        return False
    # A library half-imported before the install (an unusable system copy, and
    # the numpy under it) cannot be unloaded; start again, so the fresh copy
    # is the one imported.
    stale = [
        module for module, requirement in REQUIREMENTS if requirement in missing and module in sys.modules
    ]
    if "pandas>=1.3" in missing and "numpy" in sys.modules:
        stale.append("numpy")
    if stale:
        os.environ[REEXEC_ENV] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    _use_private_site()
    return not _import_all()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

_extract_module = None


def _extractor():
    """extract_results.py, loaded once for its parser."""
    global _extract_module
    if _extract_module is None:
        spec = importlib.util.spec_from_file_location(
            "perf_extract_results", os.path.join(SCRIPT_DIR, "extract_results.py")
        )
        _extract_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_extract_module)
    return _extract_module


def _log_files(result_dir: str) -> list[str]:
    """What extract_results.py reads: every *.txt and *.log in the directory."""
    return sorted(glob.glob(os.path.join(result_dir, "*.txt")) + glob.glob(os.path.join(result_dir, "*.log")))


def _newest_run_log(result_dir: str) -> float:
    # Not the batch_*.txt snapshots: the summary is appended to after extraction.
    paths = [p for p in _log_files(result_dir) if not os.path.basename(p).startswith("batch_")]
    return max((os.path.getmtime(p) for p in paths), default=0.0)


def load_runs(result_dir: str) -> list[dict]:
    """The result rows of a directory: results.json when it is current,
    otherwise the logs parsed here, so nothing is written into the directory
    (a baseline may not be yours). Each row gets ``result_dir``."""
    path = os.path.join(result_dir, "results.json")
    runs = None
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        current = data.get("parser_version") == _extractor().PARSER_VERSION
        if current and os.path.getmtime(path) >= _newest_run_log(result_dir):
            runs = data.get("runs", [])
    if runs is None:
        logs = _log_files(result_dir)
        if logs:
            print(
                f"[INFO] {result_dir}: results.json is missing, older than the logs or from an older "
                "parser; parsing the logs",
                file=sys.stderr,
            )
        extract = _extractor()
        runs = []
        for log in logs:
            try:
                rows = extract.parse_log(log)
            except Exception as e:  # one bad log must not lose the others
                print(f"[WARN] could not parse {log}: {e}", file=sys.stderr)
                continue
            runs += [{field: r.get(field) for field in extract.CSV_FIELDS} for r in rows]
    for run in runs:
        run["result_dir"] = result_dir
    return runs


# "# Key : value" lines of batch_summary_<stamp>.txt. A three-space indent marks
# a sub-field of the line above it (the diffusion_dlrm block); "#   - <hash> <path>"
# lines (the config list) start with a dash and are skipped.
SUMMARY_KV_RE = re.compile(r"^#( {1,3})([^:#\s-][^:]*?)\s*:\s*(.*?)\s*$")
OUTCOME_RE = re.compile(r"^# Primus (?:Multi-Node )?Benchmark Batch (.+?)\s*$")


def parse_batch_summary(path: str) -> dict:
    """Fields of one batch summary, by section: ``system`` (the environment
    snapshot), ``config`` (the batch configuration) and ``final`` (the closing
    block), plus ``outcome`` (Complete, STOPPED: ..., INTERRUPTED, REFUSED: ...)
    and ``errors`` (its [ERROR] lines)."""
    info: dict = {"system": {}, "config": {}, "final": {}, "outcome": None, "errors": []}
    section = "system"
    parent = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[ERROR]"):
                info["errors"].append(line[len("[ERROR]") :].strip())
                continue
            if line.startswith("# Batch Configuration"):
                section = "config"
                continue
            m = OUTCOME_RE.match(line)
            if m:
                section = "final"
                info["outcome"] = m.group(1)
                continue
            m = SUMMARY_KV_RE.match(line)
            if not m:
                continue
            indent, key, value = m.groups()
            if len(indent) == 1:
                parent = key
            elif parent:
                key = f"{parent} {key}"
            info[section].setdefault(key, value)
    return info


def load_batches(result_dir: str) -> dict[str, dict]:
    batches = {}
    for path in sorted(glob.glob(os.path.join(result_dir, "batch_summary_*.txt"))):
        stamp = os.path.basename(path)[len("batch_summary_") : -len(".txt")]
        batches[stamp] = parse_batch_summary(path)
    return batches


# ---------------------------------------------------------------------------
# Data (pandas)
# ---------------------------------------------------------------------------

# Fields the report reads; results.json files from older parser versions lack
# some of them.
RUN_FIELDS = (
    "filename",
    "backend",
    "model_name",
    "precision",
    "seq_len",
    "micro_batch_size",
    "global_batch_size",
    "nnodes",
    "gpus_per_node",
    "world_size",
    "repeat",
    "total_steps",
    "num_post_warmup",
    "hmean_tps_per_gpu",
    "hmean_tflops",
    "hmean_samples_per_gpu",
    "hmean_frames_per_gpu",
    "amean_memory_usage",
    "amean_memory_usage_pct",
    "exit_code",
    "elapsed_sec",
    "docker_image",
    "image_digest",
    "primus_commit",
    "submodule_pins",
    "gpu_model",
    "rocm_version",
    "torch_version",
    "jax_version",
    "config_hash",
    "host",
    "slurm_job_id",
    "nodelist",
    "timestamp",
    "status",
    "failure_reason",
    "failure_detail",
    "attempt",
    "failed_attempt",
    "config_path",
    "batch_stamp",
    "gpu_check",
    "tflops_cv",
    "version_warnings",
    "image_id",
    "container_rocm_version",
    "container_hip_version",
    "cpu_governor",
    "numa_balancing",
    "transparent_hugepages",
    "iommu",
    "amdgpu_version",
    "rec_per_s",
    "step_log",
    "result_dir",
    "system_description",
    "host_check",
    "server",
    "bios",
    "os",
    "kernel",
    "kernel_cmdline",
    "cpu_model",
    "cpu_topology",
    "cpu_frequency",
    "cpu_idle",
    "host_memory",
    "gpu_vram",
    "gpu_power_cap",
    "gpu_partition",
    "gpu_clocks",
    "gpu_vbios",
    "gpu_driver",
    "rdma_nics",
    "hmean_nominal_tflops",
    "throughput_note",
)

# One Performance row per distinct combination; the image is its ID (its tag
# when the ID is unknown). Replicates with different step counts still share
# a row.
GROUP_FIELDS = (
    "backend",
    "model_name",
    "precision",
    "config_hash",
    "seq_len",
    "micro_batch_size",
    "global_batch_size",
    "nnodes",
    "world_size",
    "image",
)

THROUGHPUT_FIELDS = (
    ("rec_per_s", "recommendations/s"),
    ("hmean_frames_per_gpu", "frames/s"),
    ("hmean_samples_per_gpu", "samples/s"),
    ("hmean_tps_per_gpu", "tokens/s"),
)


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _plain(value) -> str:
    """13.0 as 13, for text."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def throughput(run: dict) -> tuple[float | None, str | None]:
    """Per-GPU throughput and its unit: recommendations (DLRM), frames (video
    diffusion), samples (image diffusion) or tokens (language models)."""
    for field, unit in THROUGHPUT_FIELDS:
        value = run.get(field)
        if not _missing(value) and isinstance(value, (int, float)) and value > 0:
            return float(value), unit
    return None, None


def short_image(run_or_entry: dict) -> str:
    """``tag (short id)``, readable where the full ID is 71 characters."""
    tag = run_or_entry.get("docker_image") or ""
    image_id = run_or_entry.get("image_id") or ""
    short = image_id.split(":", 1)[-1][:12] if image_id else ""
    if tag and short:
        return f"{tag} ({short})"
    return tag or short


def runs_frame(runs: list[dict]):
    """One row per run, with the columns the report derives: throughput and
    unit, image (key and display), and ``counts`` (whether the run enters the
    averages)."""
    df = pd.DataFrame(runs)
    for field in RUN_FIELDS:
        if field not in df:
            df[field] = None
    df = df.astype(object).where(df.notna(), None)
    records = df.to_dict("records")
    pairs = [throughput(r) for r in records]
    df["throughput"] = [p[0] for p in pairs]
    df["unit"] = [p[1] for p in pairs]
    df["image"] = [r.get("image_id") or r.get("docker_image") or "" for r in records]
    df["image_name"] = [short_image(r) for r in records]
    df["result_dir_name"] = [os.path.basename(r.get("result_dir") or "") for r in records]
    # A failed multi-node attempt was retried; its log is kept for debugging only.
    df["counts"] = (df["status"] == "ok") & df["failed_attempt"].isna()
    return df


def _rank(value, order: tuple) -> int:
    return order.index(value) if value in order else len(order)


def sort_frame(df):
    """Backends in BACKEND_ORDER, then model, then precisions in PRECISION_ORDER."""
    keys = pd.DataFrame(
        {
            "_backend": [_rank(b, BACKEND_ORDER) for b in df["backend"]],
            "_backend_name": df["backend"].fillna("").astype(str),
            "_model": df["model_name"].fillna("").astype(str).str.lower(),
            "_precision": [_rank(str(p).upper() if p else "", PRECISION_ORDER) for p in df["precision"]],
            "_precision_name": df["precision"].fillna("").astype(str),
            "_gpus": pd.to_numeric(df["world_size"], errors="coerce").fillna(0),
            "_image": df["image"].fillna("").astype(str),
            "_hash": df["config_hash"].fillna("").astype(str),
            "_steps": pd.to_numeric(df["total_steps"], errors="coerce").fillna(0),
        },
        index=df.index,
    )
    if "batch_stamp" in df:
        keys["_batch"] = [("" if _missing(v) else str(v)) for v in df["batch_stamp"]]
    for c in ("repeat", "attempt"):
        if c in df:
            keys["_" + c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    order = keys.sort_values(list(keys.columns), kind="stable").index
    return df.loc[order].reset_index(drop=True)


def _common(values):
    """The value when all agree, otherwise the distinct values joined."""
    distinct = []
    for v in values:
        if not _missing(v) and v != "" and v not in distinct:
            distinct.append(v)
    if not distinct:
        return None
    return distinct[0] if len(distinct) == 1 else "; ".join(_plain(v) for v in distinct)


def _cv(values) -> float | None:
    values = values.dropna().astype(float)
    if len(values) < 2 or not values.mean():
        return None
    return float(values.std(ddof=1) / values.mean())


def _summarize(group) -> dict:
    good = group[group["counts"]]
    skipped = group[~group["counts"]]
    units = good["unit"].dropna()
    unit = units.iloc[0] if len(units) else None
    per_gpu = (
        good.loc[good["unit"] == unit, "throughput"].astype(float) if unit else good["throughput"].iloc[0:0]
    )
    tflops = pd.to_numeric(good["hmean_tflops"], errors="coerce")

    def mean(series):
        series = pd.to_numeric(series, errors="coerce").dropna()
        return float(series.mean()) if len(series) else None

    return {
        "runs_ok": int(len(good)),
        "runs_total": int(len(group)),
        # Retries of multi-node attempts that then succeeded, and nothing else.
        "only_retries": bool(len(good) and len(skipped) and skipped["failed_attempt"].notna().all()),
        "tflops": mean(tflops),
        "nominal_tflops": mean(good["hmean_nominal_tflops"]),
        "throughput": mean(per_gpu),
        "unit": unit,
        # Spread between repetitions; of TFLOP/s where there is one.
        "cv": _cv(tflops) if tflops.notna().any() else _cv(per_gpu),
        "memory_gb": mean(good["amean_memory_usage"]),
        "memory_pct": mean(good["amean_memory_usage_pct"]),
        "minutes": mean(pd.to_numeric(good["elapsed_sec"], errors="coerce") / 60),
        # Every step count the replicates ran, e.g. "10; 50".
        "total_steps": _common(group["total_steps"]),
        "config_path": _common(group["config_path"]),
        "docker_image": _common(group["docker_image"]),
        "image_name": _common(group["image_name"]),
    }


def aggregate(df) -> list[dict]:
    """One entry per config (GROUP_FIELDS), with the means over its
    successful runs only; the others are only counted.
    Setup and download rows of a diffusion_dlrm run (backend diffusion_dlrm)
    are not configs and only appear in Runs."""
    configs = df[df["backend"].notna() & (df["backend"] != "diffusion_dlrm")]
    if configs.empty:
        return []
    entries = []
    for key, group in configs.groupby(list(GROUP_FIELDS), dropna=False, sort=False):
        entry = {f: (None if _missing(v) else v) for f, v in zip(GROUP_FIELDS, key)}
        entry.update(_summarize(group))
        world_size = entry["world_size"]
        entry["throughput_total"] = (
            entry["throughput"] * world_size if entry["throughput"] is not None and world_size else None
        )
        entries.append(entry)
    frame = sort_frame(pd.DataFrame(entries)).astype(object)
    return frame.where(frame.notna(), None).to_dict("records")


def match_baseline(entry: dict, baseline: list[dict]) -> tuple[dict | None, str | None]:
    """The baseline config to compare with: same backend, model, precision and
    GPU count, preferring the same config file, then the same batch sizes."""
    candidates = [
        b
        for b in baseline
        if (b["backend"], b["model_name"], b["precision"], b["world_size"])
        == (entry["backend"], entry["model_name"], entry["precision"], entry["world_size"])
    ]
    if not candidates:
        return None, None
    for b in candidates:
        if b["config_hash"] and b["config_hash"] == entry["config_hash"]:
            return b, "same"
    for b in candidates:
        if (b["micro_batch_size"], b["global_batch_size"], b["seq_len"]) == (
            entry["micro_batch_size"],
            entry["global_batch_size"],
            entry["seq_len"],
        ):
            return b, "edited (same batch sizes)"
    b = candidates[0]
    return b, f"changed (MBS {_plain(b['micro_batch_size'])}, GBS {_plain(b['global_batch_size'])})"


def _change(current: float | None, base: float | None) -> float | None:
    if current is None or not base:
        return None
    return current / base - 1


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def _hms(seconds) -> str | None:
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return None
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _seconds_between(start: str | None, end: str | None) -> float | None:
    """Seconds between two summary times like ``2026-10-15 00:27:31 CDT``."""
    try:
        a, b = (datetime.strptime(t[:19], "%Y-%m-%d %H:%M:%S") for t in (start, end))
    except (TypeError, ValueError):
        return None
    return (b - a).total_seconds()


def overview_rows(stamp: str, batch: dict, runs, keep_empty: bool = False) -> list[tuple[str, str, object]]:
    """``(section, label, value)`` rows describing one batch; ``runs`` is the
    batch's slice of the runs frame. Every batch yields the same rows in the
    same order; ``keep_empty`` keeps those without a value."""
    system, config, final = batch.get("system", {}), batch.get("config", {}), batch.get("final", {})

    def from_runs(field: str):
        return _common(runs[field]) if field in runs else None

    statuses = runs["status"].fillna("unknown").value_counts().to_dict() if len(runs) else {}
    git = system.get("Git commit")
    if git and system.get("Git dirty?", "").startswith("yes"):
        git += f"  ({system['Git dirty?']})"
    container = from_runs("container_rocm_version") or config.get("ROCm (image)")
    if from_runs("container_hip_version"):
        container = f"{container}  (HIP {from_runs('container_hip_version')})"
    dd_config = config.get("diffusion_dlrm")
    start = config.get("Start time") or system.get("Date")
    end = final.get("End time") or final.get("Interrupted at")
    elapsed = final.get("Elapsed (sec)") or _seconds_between(start, end)
    good, target = final.get("Good runs"), final.get("Target good")
    rows = [
        ("Batch", "Batch stamp", stamp),
        ("Batch", "Outcome", batch.get("outcome") or "no closing block (still running, or killed)"),
        ("Batch", "Errors", "; ".join(batch.get("errors", [])[:5])),
        ("Batch", "Start time", start),
        ("Batch", "End time", end),
        ("Batch", "Elapsed (h:mm:ss)", _hms(elapsed)),
        ("Batch", "Result directory", config.get("RESULT_DIR") or from_runs("result_dir")),
        ("Batch", "Config source", config.get("Config source")),
        ("Batch", "Runner host", system.get("Hostname") or from_runs("host")),
        ("Batch", "User", system.get("User")),
        ("Batch", "SLURM job", system.get("SLURM job") or from_runs("slurm_job_id")),
        ("Batch", "SLURM nodes", system.get("SLURM nodes") or from_runs("nodelist")),
        ("Image and code", "Docker image", config.get("Docker image") or from_runs("docker_image")),
        ("Image and code", "Image digest", config.get("Image digest") or from_runs("image_digest")),
        ("Image and code", "Image ID", from_runs("image_id") or config.get("Image ID")),
        ("Image and code", "Container ROCm", container),
        ("Image and code", "Torch", from_runs("torch_version")),
        ("Image and code", "JAX", from_runs("jax_version")),
        ("Image and code", "Primus commit", from_runs("primus_commit") or git),
        ("Image and code", "Git branch", system.get("Git branch")),
        ("Image and code", "Submodule pins", from_runs("submodule_pins")),
        ("Hardware", "System", from_runs("system_description") or config.get("System")),
        ("Hardware", "Host check", from_runs("host_check") or config.get("Host check")),
        ("Hardware", "GPU model", from_runs("gpu_model") or system.get("GPU model")),
        ("Hardware", "Nodes", from_runs("nnodes")),
        ("Hardware", "GPUs per node", from_runs("gpus_per_node")),
        ("Hardware", "World size (GPUs)", from_runs("world_size")),
        ("Hardware", "Server", from_runs("server")),
        ("Hardware", "BIOS", from_runs("bios")),
        ("Hardware", "CPU", from_runs("cpu_model")),
        ("Hardware", "CPU topology", from_runs("cpu_topology")),
        ("Hardware", "CPU frequency", from_runs("cpu_frequency")),
        ("Hardware", "CPU governor", from_runs("cpu_governor")),
        ("Hardware", "CPU idle states", from_runs("cpu_idle")),
        ("Hardware", "Host memory", from_runs("host_memory")),
        ("Hardware", "NUMA balancing", from_runs("numa_balancing")),
        ("Hardware", "Transparent huge pages", from_runs("transparent_hugepages")),
        ("Hardware", "IOMMU", from_runs("iommu")),
        ("Hardware", "GPU VRAM", from_runs("gpu_vram")),
        ("Hardware", "GPU power cap", from_runs("gpu_power_cap")),
        ("Hardware", "GPU partition", from_runs("gpu_partition")),
        ("Hardware", "GPU clocks", from_runs("gpu_clocks")),
        ("Hardware", "GPU VBIOS", from_runs("gpu_vbios")),
        ("Hardware", "Host GPU driver", from_runs("gpu_driver")),
        ("Hardware", "amdgpu driver", from_runs("amdgpu_version")),
        ("Hardware", "ROCm (host)", from_runs("rocm_version") or system.get("ROCm (host)")),
        ("Hardware", "RDMA NICs", from_runs("rdma_nics")),
        ("Hardware", "OS", system.get("OS") or from_runs("os")),
        ("Hardware", "Kernel", system.get("Kernel") or from_runs("kernel")),
        ("Hardware", "Kernel cmdline", from_runs("kernel_cmdline")),
        ("Hardware", "GPU check", config.get("GPU check") or from_runs("gpu_check")),
        ("Hardware", "NCCL interface", config.get("NCCL iface")),
        (
            "Run settings",
            "Repetitions per config",
            config.get("Reps/config") or config.get("NUM_REPS (target)"),
        ),
        ("Run settings", "TRAIN_STEPS", config.get("TRAIN_STEPS")),
        ("Run settings", "Run limits", config.get("Run limits")),
        ("Run settings", "Max attempts per config", config.get("MAX_ATTEMPTS_PER_CFG")),
        ("Run settings", "FAIL_PATTERN", config.get("FAIL_PATTERN")),
        ("Run settings", "Launcher", config.get("Launcher mode") or config.get("PRIMUS_MODE")),
        ("Run settings", "diffusion_dlrm config", dd_config),
        ("Run settings", "diffusion_dlrm models", config.get("diffusion_dlrm Models")),
        ("Run settings", "diffusion_dlrm HF cache", config.get("diffusion_dlrm HF cache")),
        ("Run settings", "diffusion_dlrm limits", config.get("diffusion_dlrm Run limits")),
        ("Runs", "Catalog configs", config.get("Total configs")),
        (
            "Runs",
            "Runs planned",
            (
                f"{config['Total runs']} (diffusion_dlrm counts as one)"
                if dd_config and config.get("Total runs")
                else config.get("Total runs")
            ),
        ),
        ("Runs", "Runs started", final.get("Total runs") or final.get("Runs started")),
        ("Runs", "Total attempts", final.get("Total attempts")),
        ("Runs", "Good runs", f"{good} of {target}" if good and target else good),
        ("Runs", "Failed runs", final.get("Failed runs") or final.get("Runs failed")),
        ("Runs", "Runs not started", final.get("Not started") if final.get("Not started") != "0" else None),
        ("Runs", "Configs skipped", final.get("Skipped cfgs") if final.get("Skipped cfgs") != "0" else None),
        (
            "Runs",
            "In-flight run when stopped",
            (
                os.path.basename(final["In-flight run"])
                if final.get("In-flight run", "<none>") != "<none>"
                else None
            ),
        ),
        (
            "Runs",
            "Result rows",
            f"{len(runs)} (one per diffusion/DLRM model)" if dd_config else len(runs),
        ),
    ]
    for status in ("ok", "failed", "interrupted", "incomplete", "unknown"):
        rows.append(("Runs", f"Rows {status}", int(statuses[status]) if statuses.get(status) else None))
    if keep_empty:
        return rows
    return [(section, label, value) for section, label, value in rows if not _missing(value) and value != ""]


def overview_labels(per_batch: list[list[tuple]]) -> list[tuple[str, str]]:
    """The (section, label) rows to show for these batches (from
    overview_rows with keep_empty): in overview_rows' order, so each section
    is one block whatever fields each batch has, and only those some batch
    has a value for."""
    order: list[tuple[str, str]] = []
    filled: set[tuple[str, str]] = set()
    for rows in per_batch:
        for section, label, value in rows:
            if (section, label) not in order:
                order.append((section, label))
            if not _missing(value) and value != "":
                filled.add((section, label))
    return sorted(
        (sl for sl in order if sl in filled), key=lambda sl: (_rank(sl[0], SECTION_ORDER), order.index(sl))
    )


# ---------------------------------------------------------------------------
# Workbook (XlsxWriter)
# ---------------------------------------------------------------------------


class Formats:
    def __init__(self, wb, out_dir: str):
        # Logs are linked relative to the report, so they open wherever the folder goes.
        self.out_dir = out_dir
        add = wb.add_format
        self.title = add({"bold": True, "font_size": 16})
        self.subtitle = add({"italic": True, "font_color": "#595959"})
        self.header = add(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "text_wrap": True,
                "valign": "top",
                "border": 1,
            }
        )
        self.section = add({"bold": True, "bg_color": "#D9E1F2", "border": 1})
        self.label = add({"bold": True, "valign": "top", "border": 1})
        self.text = add({"valign": "top", "border": 1, "text_wrap": True})
        self.wrap = add({"valign": "top", "text_wrap": True})
        self.int = add({"num_format": "#,##0", "valign": "top"})
        self.tflops = add({"num_format": "#,##0.0", "valign": "top"})
        self.pct = add({"num_format": "0.0%", "valign": "top"})
        self.change = add({"num_format": "+0.0%;-0.0%;0.0%", "valign": "top"})
        self.minutes = add({"num_format": "0.0", "valign": "top"})
        self.by_magnitude = {
            0: add({"num_format": "#,##0", "valign": "top"}),
            1: add({"num_format": "#,##0.0", "valign": "top"}),
            2: add({"num_format": "0.00", "valign": "top"}),
            3: add({"num_format": "0.000", "valign": "top"}),
        }
        self.bad = add({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        self.good = add({"bg_color": "#C6EFCE", "font_color": "#006100"})
        self.warn = add({"bg_color": "#FFEB9C", "font_color": "#9C5700"})
        self.bad_cell = add({"bg_color": "#FFC7CE", "font_color": "#9C0006", "valign": "top"})
        self.warn_cell = add({"bg_color": "#FFEB9C", "font_color": "#9C5700", "valign": "top"})
        self.note = add({"italic": True, "font_color": "#595959"})

    def number(self, value: float):
        """A number format that suits the value: 38,414 tokens/s and 0.624 frames/s."""
        return self.by_magnitude[0 if value >= 1000 else 1 if value >= 100 else 2 if value >= 1 else 3]


def _native(value):
    """numpy scalars as plain Python values; NaN as None."""
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        value = value.item()
    return None if _missing(value) else value


def _write(ws, row: int, col: int, value, fmt=None, fmts: Formats | None = None):
    """Write by type; without ``fmt``, whole numbers get thousands separators
    and fractional ones the decimals their size calls for."""
    value = _native(value)
    if value is None or value == "":
        ws.write_blank(row, col, None, fmt)
    elif isinstance(value, bool):
        ws.write_string(row, col, "yes" if value else "no", fmt)
    elif isinstance(value, (int, float)):
        if fmt is None and fmts is not None:
            whole = isinstance(value, int) or float(value).is_integer()
            fmt = fmts.int if whole else fmts.number(value)
        ws.write_number(row, col, value, fmt)
    elif isinstance(value, list):
        ws.write_string(row, col, " | ".join(str(v) for v in value), fmt)
    else:
        ws.write_string(row, col, str(value), fmt)


def _link(ws, row: int, col: int, target: str | None, out_dir: str):
    """A relative link to a log, so it opens from wherever the folder is copied."""
    if not target or not os.path.exists(target):
        ws.write_blank(row, col, None)
        return
    rel = os.path.relpath(target, out_dir)
    ws.write_url(row, col, "external:" + rel, string=os.path.basename(target))


def _table(
    ws, first_row: int, columns: list[dict], rows: list[list], fmts: Formats, name: str, freeze_cols: int = 0
) -> int:
    """An Excel table (filter buttons, banded rows) with per-cell formats.
    ``columns``: {"header", "width", "fmt" (a Formats attribute name), "link"
    (True to write the value as a log link)}. Returns the last row."""
    last_row = first_row + max(len(rows), 1)
    ws.add_table(
        first_row,
        0,
        last_row,
        len(columns) - 1,
        {
            "name": name,
            "style": "Table Style Medium 2",
            "columns": [{"header": c["header"], "header_format": fmts.header} for c in columns],
        },
    )
    for i, c in enumerate(columns):
        ws.set_column(i, i, c.get("width", 12))
    for r, values in enumerate(rows, start=first_row + 1):
        for i, (c, value) in enumerate(zip(columns, values)):
            fmt = getattr(fmts, c["fmt"]) if c.get("fmt") else None
            if c.get("link"):
                _link(ws, r, i, value, fmts.out_dir)
            else:
                _write(ws, r, i, value, fmt, fmts)
    ws.freeze_panes(first_row + 1, freeze_cols)
    return last_row


def _print_setup(ws, title: str, context: str):
    """Landscape, one page wide, with the title and the images in the page header."""
    ws.set_landscape()
    ws.fit_to_pages(1, 0)
    ws.set_header(f"&L{title.replace('&', '&&')}&R{context.replace('&', '&&')}"[:250])


def write_overview(
    wb,
    fmts: Formats,
    title: str,
    inputs: list[str],
    baselines: list[tuple[str, str]],
    batches: list[tuple[str, dict, object]],
):
    ws = wb.add_worksheet("Overview")
    ws.write(0, 0, title, fmts.title)
    ws.write(
        1,
        0,
        f"Generated {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} by tools/perf/report.py",
        fmts.subtitle,
    )
    row = 3
    for v in inputs:
        ws.write(row, 0, "Results", fmts.label)
        ws.write(row, 1, v, fmts.text)
        row += 1
    for path, description in baselines:
        ws.write(row, 0, "Baseline", fmts.label)
        ws.write(row, 1, f"{path}\n{description}", fmts.text)
        row += 1
    row += 1

    per_batch = [overview_rows(label, info, runs, keep_empty=True) for label, info, runs in batches]
    values = [{(s, l): v for s, l, v in rows} for rows in per_batch]
    current_section = None
    for section, label in overview_labels(per_batch):
        if section != current_section:
            ws.write(row, 0, section, fmts.section)
            for i in range(len(batches)):
                ws.write_blank(row, i + 1, None, fmts.section)
            current_section = section
            row += 1
        ws.write(row, 0, label, fmts.label)
        for i, v in enumerate(values):
            _write(ws, row, i + 1, v.get((section, label)), fmts.text)
        row += 1

    row += 1
    notes = [
        "How to read the numbers:",
        (
            "- Performance has one row per config and image, and averages its successful runs (status ok), "
            "replicates with different step counts included; failed runs and failed multi-node attempts are "
            "left out before averaging and counted in 'Runs ok / total' (the Runs sheet says why)."
        ),
        "- Config hash: the first 8 hex digits of the config file's SHA-256, so an edited file shows up as a new row.",
        (
            "- Within a run, Primus throughput is the harmonic mean over the steps after warmup "
            "(at least the first three logged steps are dropped)."
        ),
        "- Diffusion and DLRM numbers are diffusion_bench's own averages (they match its summary.md).",
        (
            "- Throughput is per GPU: tokens/s (language models), samples/s or frames/s (diffusion), "
            "recommendations/s (DLRM). 'All GPUs' multiplies it by the GPU count."
        ),
        (
            "- Run-to-run CV is the spread between repetitions (stdev / mean), shown from two successful runs; "
            f"above {CV_WARN:.0%} it is highlighted. Step-to-step CV (Runs sheet) is the spread within one run."
        ),
        "- GPU memory is as the backend logs it: GB for Megatron, GiB for TorchTitan.",
        (
            "- Runs lists every run as an Excel table, with the failure reason and detail of those that did not "
            "pass; Insert > PivotTable on it for other views."
        ),
    ]
    for line in notes:
        # Unwrapped, so the line runs on across the empty cells to its right.
        ws.write_string(row, 0, line, fmts.note)
        row += 1
    ws.set_column(0, 0, 26)
    ws.set_column(1, max(len(batches), 1), 70)


def performance_columns(with_image: bool, with_baseline: bool, with_nominal: bool = False) -> list[dict]:
    columns = [
        {"header": "Backend", "width": 14},
        {"header": "Model", "width": 22},
        {"header": "Precision", "width": 10},
        {"header": "Config hash", "width": 11},
        {"header": "Seq len", "width": 8, "fmt": "int"},
        {"header": "MBS", "width": 7},
        {"header": "GBS", "width": 8},
        {"header": "Steps", "width": 7},
        {"header": "Nodes", "width": 7, "fmt": "int"},
        {"header": "GPUs", "width": 6, "fmt": "int"},
    ]
    if with_image:
        columns.append({"header": "Image", "width": 34})
    columns += [
        {"header": "Runs ok / total", "width": 9},
        {"header": "TFLOP/s per GPU", "width": 10, "fmt": "tflops"},
    ]
    if with_nominal:
        columns.append({"header": "Nominal TFLOP/s per GPU", "width": 11, "fmt": "tflops"})
    columns += [
        {"header": "Throughput per GPU", "width": 12},
        {"header": "Throughput unit", "width": 17},
        {"header": "Throughput, all GPUs", "width": 13},
        {"header": "Run-to-run CV", "width": 9, "fmt": "pct"},
        {"header": "GPU memory", "width": 9, "fmt": "tflops"},
        {"header": "GPU memory (%)", "width": 9, "fmt": "tflops"},
        {"header": "Minutes per run", "width": 9, "fmt": "minutes"},
    ]
    if with_baseline:
        columns += [
            {"header": "Baseline TFLOP/s per GPU", "width": 11, "fmt": "tflops"},
            {"header": "TFLOP/s change", "width": 9, "fmt": "change"},
            {"header": "Baseline throughput per GPU", "width": 12},
            {"header": "Throughput change", "width": 10, "fmt": "change"},
            {"header": "Baseline config", "width": 22},
        ]
    columns.append({"header": "Config", "width": 60})
    return columns


def write_performance(
    wb,
    fmts: Formats,
    entries: list[dict],
    baseline: list[dict] | None,
    threshold: float,
    title: str,
    context: str,
):
    ws = wb.add_worksheet("Performance")
    with_image = len({e["image"] for e in entries}) > 1
    with_nominal = any(e["nominal_tflops"] is not None for e in entries)
    columns = performance_columns(with_image, baseline is not None, with_nominal)
    rows = []
    for e in entries:
        values = [
            e["backend"],
            e["model_name"],
            e["precision"],
            e["config_hash"],
            e["seq_len"],
            e["micro_batch_size"],
            e["global_batch_size"],
            e["total_steps"],
            e["nnodes"],
            e["world_size"],
        ]
        if with_image:
            values.append(e["image_name"])
        values += [f"{e['runs_ok']} / {e['runs_total']}", e["tflops"]]
        if with_nominal:
            values.append(e["nominal_tflops"])
        values += [
            e["throughput"],
            e["unit"],
            e["throughput_total"],
            e["cv"],
            e["memory_gb"],
            e["memory_pct"],
            e["minutes"],
        ]
        if baseline is not None:
            base, how = match_baseline(e, baseline)
            same_unit = base is not None and base["unit"] == e["unit"]
            values += [
                base["tflops"] if base else None,
                _change(e["tflops"], base["tflops"] if base else None),
                base["throughput"] if same_unit else None,
                _change(e["throughput"], base["throughput"]) if same_unit else None,
                how or "not in baseline",
            ]
        values.append(e["config_path"])
        rows.append(values)
    last = _table(ws, 0, columns, rows, fmts, "PerformanceTable", freeze_cols=3)

    headers = [c["header"] for c in columns]
    col = headers.index("Run-to-run CV")
    ws.conditional_format(
        1, col, last, col, {"type": "cell", "criteria": ">", "value": CV_WARN, "format": fmts.warn}
    )
    # A config with runs that did not count stands out: amber when they were
    # multi-node attempts that a retry made good, red otherwise.
    col = headers.index("Runs ok / total")
    for r, e in enumerate(entries, start=1):
        if e["runs_ok"] < e["runs_total"]:
            fmt = fmts.warn_cell if e["only_retries"] else fmts.bad_cell
            ws.write_string(r, col, f"{e['runs_ok']} / {e['runs_total']}", fmt)
    if baseline is not None:
        for name in ("TFLOP/s change", "Throughput change"):
            col = headers.index(name)
            ws.conditional_format(
                1, col, last, col, {"type": "cell", "criteria": "<", "value": -threshold, "format": fmts.bad}
            )
            ws.conditional_format(
                1, col, last, col, {"type": "cell", "criteria": ">", "value": threshold, "format": fmts.good}
            )
    ws.set_row(0, 45)
    _print_setup(ws, title, context)


RUN_COLUMNS = [
    {"header": "Batch", "width": 16, "key": "batch_stamp"},
    {"header": "Backend", "width": 14, "key": "backend"},
    {"header": "Model", "width": 22, "key": "model_name"},
    {"header": "Precision", "width": 10, "key": "precision"},
    {"header": "Config hash", "width": 11, "key": "config_hash"},
    {"header": "Rep", "width": 5, "key": "repeat", "fmt": "int"},
    {"header": "Attempt", "width": 7, "key": "attempt", "fmt": "int"},
    {"header": "Status", "width": 11, "key": "status"},
    {"header": "Failure reason", "width": 14, "key": "failure_reason"},
    {"header": "Failure detail", "width": 40, "key": "failure_detail"},
    {"header": "TFLOP/s per GPU", "width": 10, "key": "hmean_tflops", "fmt": "tflops"},
    {"header": "Nominal TFLOP/s per GPU", "width": 11, "key": "hmean_nominal_tflops", "fmt": "tflops"},
    {"header": "Throughput per GPU", "width": 12, "key": "throughput"},
    {"header": "Throughput unit", "width": 17, "key": "unit"},
    {"header": "Throughput note", "width": 30, "key": "throughput_note"},
    {"header": "Seq len", "width": 8, "key": "seq_len", "fmt": "int"},
    {"header": "MBS", "width": 7, "key": "micro_batch_size"},
    {"header": "GBS", "width": 8, "key": "global_batch_size"},
    {"header": "Nodes", "width": 7, "key": "nnodes", "fmt": "int"},
    {"header": "GPUs", "width": 6, "key": "world_size", "fmt": "int"},
    {"header": "Steps", "width": 7, "key": "total_steps", "fmt": "int"},
    {"header": "Steps measured", "width": 9, "key": "num_post_warmup", "fmt": "int"},
    {"header": "Step-to-step CV", "width": 9, "key": "tflops_cv", "fmt": "pct"},
    {"header": "GPU memory", "width": 9, "key": "amean_memory_usage", "fmt": "tflops"},
    {"header": "Elapsed (s)", "width": 9, "key": "elapsed_sec", "fmt": "int"},
    {"header": "Exit code", "width": 6, "key": "exit_code", "fmt": "int"},
    {"header": "Image", "width": 34, "key": "image_name"},
    {"header": "Host", "width": 16, "key": "host"},
    {"header": "System", "width": 40, "key": "system_description"},
    {"header": "Started", "width": 22, "key": "timestamp"},
    {"header": "Result dir", "width": 24, "key": "result_dir_name"},
    {"header": "Log", "width": 40, "key": "_log", "link": True},
    {"header": "Step log", "width": 30, "key": "_step_log", "link": True},
    {"header": "Config", "width": 60, "key": "config_path"},
]


def _run_values(run: dict, columns: list[dict]) -> list:
    extra = {
        "_log": os.path.join(run["result_dir"], run["filename"]) if run.get("filename") else None,
        "_step_log": os.path.join(run["result_dir"], run["step_log"]) if run.get("step_log") else None,
    }
    return [extra[c["key"]] if c["key"] in extra else run.get(c["key"]) for c in columns]


def write_runs(wb, fmts: Formats, runs: list[dict]):
    ws = wb.add_worksheet("Runs")
    last = _table(
        ws, 0, RUN_COLUMNS, [_run_values(r, RUN_COLUMNS) for r in runs], fmts, "RunsTable", freeze_cols=4
    )
    col = [c["header"] for c in RUN_COLUMNS].index("Status")
    ws.conditional_format(
        1, col, last, col, {"type": "cell", "criteria": "!=", "value": '"ok"', "format": fmts.bad}
    )
    ws.set_row(0, 30)


def chart_labels(entries: list[dict]) -> list[str]:
    """``backend: model precision``, with the GPU count when not 8, the image
    when there are several, and the config hash when that still repeats."""
    several_images = len({e["image"] for e in entries}) > 1
    labels = []
    for e in entries:
        label = f"{e['backend']}: " + " ".join(str(v) for v in (e["model_name"], e["precision"]) if v)
        if e["world_size"] and e["world_size"] != 8:
            label += f" ({_plain(e['world_size'])} GPUs)"
        if several_images:
            label += f" [{e['docker_image'] or e['image_name']}]"
        labels.append(label)
    repeated = {l for l in labels if labels.count(l) > 1}
    return [
        (
            f"{l} #{e['config_hash']}" + (f" {_plain(e['total_steps'])} steps" if e["total_steps"] else "")
            if l in repeated
            else l
        )
        for l, e in zip(labels, entries)
    ]


def write_chart(wb, fmts: Formats, entries: list[dict], baseline: list[dict] | None):
    plotted = [e for e in entries if e["tflops"] is not None]
    if not plotted:
        return
    data = []
    for label, e in zip(chart_labels(plotted), plotted):
        base = match_baseline(e, baseline)[0] if baseline is not None else None
        data.append((label, e["tflops"], base["tflops"] if base else None))
    ws = wb.add_worksheet("Chart")
    ws.write_row(
        0, 0, ["Config", "TFLOP/s per GPU"] + (["Baseline"] if baseline is not None else []), fmts.header
    )
    for i, (label, value, base) in enumerate(data, start=1):
        ws.write_string(i, 0, label)
        ws.write_number(i, 1, value, fmts.tflops)
        if baseline is not None:
            _write(ws, i, 2, base, fmts.tflops)
    ws.set_column(0, 0, 48)
    ws.set_column(1, 2, 14)

    chart = wb.add_chart({"type": "bar"})
    n = len(data)
    chart.add_series(
        {
            "name": "This run",
            "categories": ["Chart", 1, 0, n, 0],
            "values": ["Chart", 1, 1, n, 1],
            "data_labels": {"value": True, "num_format": "#,##0"},
            "fill": {"color": "#1F4E78"},
        }
    )
    if baseline is not None:
        chart.add_series(
            {
                "name": "Baseline",
                "categories": ["Chart", 1, 0, n, 0],
                "values": ["Chart", 1, 2, n, 2],
                "fill": {"color": "#A6A6A6"},
            }
        )
    chart.set_title({"name": "TFLOP/s per GPU"})
    # XlsxWriter swaps the axes of a bar chart: x is the (horizontal) value
    # axis, y the category axis, reversed so configs run top to bottom in
    # table order.
    chart.set_x_axis({"num_format": "#,##0", "major_gridlines": {"visible": True}})
    chart.set_y_axis({"reverse": True})
    chart.set_legend({"position": "bottom" if baseline is not None else "none"})
    chart.set_size({"width": 900, "height": max(320, 28 * n * (2 if baseline is not None else 1) + 120)})
    ws.insert_chart(0, 4, chart)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _describe_baseline(runs: list[dict]) -> str:
    """Image, code and batches of a baseline, which the change columns depend on."""
    images = _common(short_image(r) for r in runs) or "image unknown"
    commits = _common(r.get("primus_commit") for r in runs) or "commit unknown"
    stamps = _common(r.get("batch_stamp") for r in runs) or "no batch stamp"
    return f"{images}; Primus {commits}; batch {stamps}"


def _unique(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        p = os.path.realpath(p)
        if p not in out:
            out.append(p)
    return out


def build_report(
    out_path: str, inputs: list[str], baselines: list[str], threshold: float, title: str | None
) -> tuple[int, int]:
    all_runs: list[dict] = []
    batches = []
    for d in inputs:
        dir_runs = load_runs(d)
        all_runs += dir_runs
        summaries = load_batches(d)
        stamps = list(summaries) + sorted(
            {r.get("batch_stamp") for r in dir_runs if r.get("batch_stamp")} - set(summaries)
        )
        for stamp in stamps:
            label = stamp if len(inputs) == 1 else f"{stamp}  ({os.path.basename(d)})"
            batches.append((label, summaries.get(stamp, {}), d, stamp))
    if not all_runs:
        return 0, 0
    df = sort_frame(runs_frame(all_runs))
    batch_rows = []
    for label, info, d, stamp in batches:
        batch_rows.append((label, info, df[(df["result_dir"] == d) & (df["batch_stamp"] == stamp)]))
    orphans = df[df["batch_stamp"].isna()]
    if len(orphans):
        batch_rows.append(("(no batch stamp)", {}, orphans))

    baseline_entries = None
    baseline_notes = []
    if baselines:
        baseline_runs = []
        for d in baselines:
            runs = load_runs(d)
            baseline_runs += runs
            baseline_notes.append((d, _describe_baseline(runs) if runs else "no runs found"))
        baseline_entries = aggregate(runs_frame(baseline_runs)) if baseline_runs else []
    entries = aggregate(df)
    runs = df.to_dict("records")
    title = title or "Primus performance report"
    context = _common(df["image_name"]) or ""

    # Written beside the final name and renamed over it, so an interrupted
    # run never leaves a truncated report in place of the previous one.
    tmp = out_path + ".tmp.xlsx"
    wb = xlsxwriter.Workbook(tmp)
    fmts = Formats(wb, os.path.dirname(out_path))
    try:
        write_overview(wb, fmts, title, inputs, baseline_notes, batch_rows)
        write_performance(wb, fmts, entries, baseline_entries, threshold, title, str(context))
        write_runs(wb, fmts, runs)
        write_chart(wb, fmts, entries, baseline_entries)
        wb.close()
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return len(entries), len(runs)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an Excel report of perf batch results.")
    parser.add_argument(
        "result_dirs", nargs="+", metavar="RESULT_DIR", help="Result directories to report on"
    )
    parser.add_argument(
        "-o", "--output", help="Report file (default: RESULT_DIR/report.xlsx for one directory)"
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="DIR",
        help="Result directory to compare with, e.g. the previous release; repeatable",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Highlight changes from the baseline beyond this many percent (default: 5)",
    )
    parser.add_argument("--title", help="Title on the Overview sheet")
    parser.add_argument(
        "--no-install", action="store_true", help="Do not install pandas or XlsxWriter when they are missing"
    )
    args = parser.parse_args(argv)

    inputs = _unique(args.result_dirs)
    baselines = _unique(args.baseline)
    for d in inputs + baselines:
        if not os.path.isdir(d):
            print(f"[ERROR] not a directory: {d}", file=sys.stderr)
            return EXIT_ERROR
    if args.output:
        out_path = os.path.abspath(args.output)
    elif len(inputs) == 1:
        out_path = os.path.join(inputs[0], "report.xlsx")
    else:
        print("[ERROR] with several result directories, name the report with -o FILE.xlsx", file=sys.stderr)
        return EXIT_ERROR
    if not os.path.isdir(os.path.dirname(out_path)):
        print(f"[ERROR] the report's directory does not exist: {os.path.dirname(out_path)}", file=sys.stderr)
        return EXIT_ERROR

    if not load_dependencies(allow_install=not args.no_install):
        print(
            f"[INFO] pandas or XlsxWriter is missing or unusable; run report.py once without --no-install "
            f"to install them into {PRIVATE_SITE}.",
            file=sys.stderr,
        )
        return EXIT_NO_DEPENDENCIES

    # XlsxWriter reports an unwritable file with its own exception, not OSError.
    write_errors = (OSError,)
    exceptions = getattr(xlsxwriter, "exceptions", None)
    if exceptions is not None and hasattr(exceptions, "XlsxWriterException"):
        write_errors += (exceptions.XlsxWriterException,)
    try:
        configs, rows = build_report(out_path, inputs, baselines, args.threshold / 100, args.title)
    except write_errors as e:
        print(f"[ERROR] could not write {out_path}: {e}", file=sys.stderr)
        return EXIT_ERROR
    if not rows:
        print(f"No runs to report in {', '.join(inputs)}")
        return EXIT_NO_RUNS
    print(f"Report written to: {out_path}  ({_plural(configs, 'config')}, {_plural(rows, 'result row')})")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

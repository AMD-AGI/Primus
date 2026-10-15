#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Extract benchmarking results from Primus training log files.

Usage::

    python3 tools/perf/extract_results.py [--quiet] <result_dir>

Writes ``results.csv`` and ``results.json`` into ``<result_dir>``, replacing
any previous ones: both are derived from every log in the directory, so
re-running is always safe. The batch runners call this automatically at the
end of a batch. ``results.json`` carries the same rows plus the parser
version, and keeps list-valued fields (warnings) as lists.

Every row has a ``status``: ``ok``, ``failed``, ``interrupted``,
``incomplete`` (no footer: still running, or the batch was killed) or
``unknown`` (legacy logs without a footer). Rows that are not ``ok`` also get
a ``failure_reason`` and the log line behind it (``failure_detail``). The
runner's own verdict (``timeout``, ``idle``, ``interrupted``) is taken from
the log footer; otherwise the run output is searched for the cause:
``nic_error``, ``oom``, ``gpu_contended`` (an out-of-memory error while
PyTorch itself held less than half the GPU, i.e. another process had it),
``segfault``, ``nccl_error`` or ``python_error``; failing that ``killed``
(SIGKILL with no error in the log), ``signal``, ``fail_pattern`` or ``exit``.

Supported header formats:

1. Batch-runner format - logs start with a structured ``# Primus Benchmark
   Run`` banner (``run_batch.sh``, single node) or ``# Primus Multi-Node
   Benchmark Run`` (``run_batch_multinode.sh``), followed by the full YAML
   config dump and the run output. The two share the same key/value layout,
   so provenance fields (image digest, submodule pins, world size, ...) are
   read the same way from both. Older logs that predate a given field simply
   leave its column empty.
2. Legacy formats (``EXP=examples/...`` and ``print config file:`` headers)
   produced by older runners. Kept so historical result directories still
   parse.

Supported Megatron per-iteration layouts. The format has changed twice, and
all three spellings still appear -- sometimes within a single run, because
the first few warmup iterations are emitted in the legacy form::

    throughput per GPU (TFLOP/s/GPU): 346.3 | tokens per GPU (tokens/s/GPU): 20502.6
    throughput per GPU (TFLOP/s/GPU): 346.3/340.8 | tokens/s/GPU inst/harmonic mean: 20502.6/20151.0
    compute per GPU (TFLOP/s/GPU): 615.5 (avg 615.5) | tokens/s/GPU inst/harmonic mean: 11941.8/11941.8

In the latter two the first number is the instantaneous per-step value and
the second is the backend's running average; the instantaneous value is what
gets captured, so this script computes its own harmonic mean over the
post-warmup window.

Per-iteration metrics are parsed for megatron, megatron_bridge, torchtitan,
maxtext and maxdiffusion backends. Megatron-Bridge only logs TFLOP/s/GPU,
as the number immediately before ``MODEL_TFLOP/s/GPU`` (no space between
the value and the unit). At least the first three unique logged steps are
always dropped before aggregates (compile / autotune / ramp); a larger
configured warmup is still honored when it leaves some steps behind. For
megatron the warmup prefers ``log_avg_skip_iterations`` (the steps the
throughput patch itself skips) over ``lr_warmup_iters``. Throughput metrics
use the harmonic mean, and memory metrics use the arithmetic mean. Results
are written to results.csv / results.json in the input directory. MaxDiffusion also
reports sample/s and (WAN) frame/s; those extra columns are empty for other
backends.

For multi-node runs every rank's stdout is interleaved into a single log
file. Different backends behave differently:

* torchtitan emits per-step lines from every rank (so step 1 appears in
  dozens of lines on an 8-node job),
* Primus-patched megatron uses ``print_rank_last`` and therefore only
  emits per-iteration lines from the last rank in the world (e.g.
  ``rank-63/64``), not ``rank-0``,
* maxtext and maxdiffusion log only from JAX process 0.

To keep ``num_iterations`` and the throughput aggregates honest, the
parsers filter to a single rank: ``rank-0`` when it emits any iteration
lines, otherwise the lowest rank that does. Legacy logs that have no
``[rank-N/M]`` tag are not filtered.

A ``diffusion_dlrm`` log (one diffusion_bench run covering several models)
becomes one row per diffusion model and DLRM precision, with backend
``diffusion`` or ``dlrm``. Their numbers come from diffusion_bench's own
parser (``diffusion_bench/parse_results.py``), so they match its
``summary.md``: TFLOP/s and samples/s per GPU are AMDiffusionBenchmark's
averages rather than this script's harmonic means, and ``rec_per_s`` is
DLRMBenchmark's recommendations/s. ``exit_code``, ``elapsed_sec``, status and
failure reason are the step's, and ``step_log`` points at its log.
"""

# Keeps the annotations below unevaluated, so the batch runners can call this
# script on hosts whose python3 predates 3.10.
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import signal
import statistics
import subprocess
import sys
from datetime import datetime

# Bumped whenever a column is added or its meaning changes.
PARSER_VERSION = 4

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BANNER_KV_RE = re.compile(r"^#\s*([^:]+?)\s*:\s*(.*?)\s*$")
PLACEHOLDER_RE = re.compile(r"^\$\{[^:}]+:([^}]*)\}$")
RANK_TAG_RE = re.compile(r"\[rank-(\d+)/\d+\]")

# Minimum number of unique logged steps dropped before harmonic/arithmetic
# means. Step 0 (and the next couple) are typically JIT/compile + autotune
# and would otherwise poison the aggregates, including for maxtext /
# maxdiffusion where there is no configured perf-warmup.
MIN_PERF_SKIP_STEPS = 3


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _detect_iter_logging_rank(lines: list[str], iter_marker: str) -> int | None:
    """Pick which rank's per-step lines to keep in a multi-rank log.

    Scans ``lines`` for occurrences of ``iter_marker`` (a backend-specific
    substring that uniquely identifies per-iteration log lines, e.g.
    ``"elapsed time per iteration"`` or ``"tflops:"``) and returns:

    * ``0`` if any matching line is tagged ``[rank-0/N]``,
    * otherwise the lowest rank that emits such lines (so Primus-patched
      megatron, where only ``rank-N-1`` prints, still works),
    * ``None`` if no matching line carries a ``[rank-X/Y]`` tag (legacy
      single-rank logs).

    A ``None`` return tells the per-backend parser not to filter.
    """
    has_tag = False
    ranks: set[int] = set()
    for line in lines:
        clean = strip_ansi(line)
        if iter_marker not in clean:
            continue
        m = RANK_TAG_RE.search(clean)
        if not m:
            continue
        has_tag = True
        rank = int(m.group(1))
        if rank == 0:
            return 0
        ranks.add(rank)
    if not has_tag:
        return None
    return min(ranks) if ranks else None


def _line_matches_rank(clean_line: str, keep_rank: int | None) -> bool:
    """Return True if ``clean_line`` should be kept for ``keep_rank``.

    If ``keep_rank`` is None, every line passes. Otherwise lines tagged
    ``[rank-X/Y]`` are kept only when ``X == keep_rank``; lines without
    a rank tag are always kept (they're not multi-rank duplicates).
    """
    if keep_rank is None:
        return True
    m = RANK_TAG_RE.search(clean_line)
    if m is None:
        return True
    return int(m.group(1)) == keep_rank


def harmonic_mean(values: list[float]) -> float | None:
    if not values or any(v <= 0 for v in values):
        return None
    return len(values) / sum(1.0 / v for v in values)


def arithmetic_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _resolve_placeholder(value: str | None) -> str | None:
    """Resolve a ``${VAR:default}`` shell-style placeholder to its default.

    Returns the value unchanged if it's not a placeholder. Used because some
    YAML fields (e.g. ``seq_length: ${PRIMUS_SEQ_LENGTH:4096}``) are dumped
    into logs without env-substitution.
    """
    if value is None:
        return None
    m = PLACEHOLDER_RE.match(value.strip())
    return m.group(1) if m else value


def _maybe_int(value: str | None) -> int | None:
    if value is None:
        return None
    resolved = _resolve_placeholder(value)
    if resolved is None or resolved == "" or resolved.upper() in ("N/A", "NA", "NULL", "NONE"):
        return None
    try:
        return int(resolved)
    except (TypeError, ValueError):
        # Banner fields often look like ``20 (CLI override; yaml=50)``.
        m = re.match(r"^\s*(-?\d+)", resolved)
        if m:
            return int(m.group(1))
        return None


def _maybe_number(value: str | None) -> int | float | None:
    """Like ``_maybe_int``, but keeps a fractional value such as MaxText's
    ``per_device_batch_size: 0.5`` instead of truncating it to 0."""
    if value is None:
        return None
    resolved = _resolve_placeholder(value)
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:$|\()", resolved or "")
    if not m:
        return _maybe_int(value)
    number = float(m.group(1))
    return int(number) if number.is_integer() else number


def parse_config_section(lines: list[str]) -> str:
    """Extract the YAML config dump from a log file.

    Handles both the new ``########## Begin Config File Dump ##########`` /
    ``########## End Config File Dump ##########`` markers and the older
    ``=====`` / ``-----`` separator style anchored on a
    ``print config file:`` line.
    """
    begin_re = re.compile(r"Begin Config File Dump")
    end_re = re.compile(r"End Config File Dump")
    in_block = False
    out: list[str] = []
    for line in lines:
        clean = strip_ansi(line)
        if not in_block:
            if begin_re.search(clean):
                in_block = True
            continue
        if end_re.search(clean):
            return "".join(out)
        out.append(line)
    if out:
        return "".join(out)

    sep_re = re.compile(r"^[=-]{5,}$")
    print_cfg_re = re.compile(r"print (?:the )?config file:", re.IGNORECASE)
    found_print_config = False
    config_start = False
    config_lines: list[str] = []
    for line in lines:
        stripped = strip_ansi(line.strip())
        if not found_print_config:
            if print_cfg_re.search(stripped):
                found_print_config = True
            continue
        if sep_re.match(stripped):
            if not config_start:
                config_start = True
                continue
            else:
                break
        if config_start:
            config_lines.append(line)
    return "".join(config_lines)


def parse_cmd_params(cmd_line: str) -> dict:
    """Extract key-value pairs from a CMD: line (--key value)."""
    params = {}
    for m in re.finditer(r"--(\S+)\s+(\S+)", cmd_line):
        params[m.group(1)] = m.group(2)
    return params


def parse_run_footer(lines: list[str]) -> dict:
    """Extract ``exit_code`` / ``elapsed_sec`` from the new format footer.

    The batch runner appends a small ``# Run finished at ...`` footer to every
    log. We scan from the bottom up so we don't accidentally parse other
    ``Exit code:`` strings that may appear earlier in the run output.
    """
    out: dict = {}
    for line in reversed(lines[-50:]):
        clean = strip_ansi(line.rstrip())
        if clean.startswith("# Failure reason") and "failure_reason" not in out:
            m = re.match(r"# Failure reason\s*:\s*(.*)$", clean)
            if m:
                out["failure_reason"] = m.group(1).strip()
        elif clean.startswith("# Failure pattern?") and "failure_pattern_flag" not in out:
            # Footer of the multi-node runner before "Failure reason" existed.
            out["failure_pattern_flag"] = clean.split(":", 1)[-1].strip().lower()
        elif "Exit code" in clean and "exit_code" not in out:
            m = re.search(r"Exit code\s*:\s*(-?\d+)", clean)
            if m:
                out["exit_code"] = int(m.group(1))
        elif "Elapsed (sec)" in clean and "elapsed_sec" not in out:
            m = re.search(r"Elapsed \(sec\)\s*:\s*(\d+)", clean)
            if m:
                out["elapsed_sec"] = int(m.group(1))
        if "exit_code" in out and "elapsed_sec" in out:
            break
    return out


def parse_new_banner(lines: list[str]) -> dict | None:
    """Parse the new batch-runner banner.

    The banner is a contiguous block of ``# key : value`` lines bracketed by
    ``###...`` separator rows, starting on line 1 with either
    ``# Primus Benchmark Run`` (single-node, ``run_batch.sh``) or
    ``# Primus Multi-Node Benchmark Run`` (multi-node,
    ``run_batch_multinode.sh``), and ending at the
    ``Begin Config File Dump`` marker. Returns the collected key/value pairs,
    or ``None`` if this log is not in the new format.
    """
    if len(lines) < 3:
        return None
    if not lines[0].lstrip().startswith("#"):
        return None
    # Accept both single-node ("Primus Benchmark Run") and multi-node
    # ("Primus Multi-Node Benchmark Run") banner titles. The two batch
    # runners share the same banner layout otherwise, so anything that
    # contains both "Primus" and "Benchmark Run" on line 2 is treated as
    # a Primus run log.
    header = lines[1]
    if "Primus" not in header or "Benchmark Run" not in header:
        return None

    banner: dict[str, str] = {}
    for line in lines[1:200]:
        clean = strip_ansi(line.rstrip())
        if "Begin Config File Dump" in clean:
            break
        if not clean.startswith("#"):
            continue
        m = BANNER_KV_RE.match(clean)
        if not m:
            continue
        key = m.group(1).strip()
        val = m.group(2).strip()
        if not key or key.startswith("---") or key.startswith("==="):
            continue
        banner[key] = val
    return banner or None


def parse_megatron_iterations(lines: list[str], keep_rank: int | None = None) -> list[dict]:
    """Parse Primus-megatron per-iteration lines.

    Three on-disk formats are supported:

    * Legacy single-value fields, e.g.::

        throughput per GPU (TFLOP/s/GPU): 346.3 | tokens per GPU (tokens/s/GPU): 20502.6

    * ``inst/harmonic mean`` dual-value fields, e.g.::

        elapsed time per iteration (ms): 1997.8/2032.7 |
        throughput per GPU (TFLOP/s/GPU): 346.3/340.8 |
        tokens/s/GPU inst/harmonic mean: 20502.6/20151.0

    * Current ``print_rank_last`` throughput patch, which renamed the TFLOP
      field to ``compute per GPU`` and switched it to an ``X (avg Y)``
      layout, e.g.::

        elapsed time per iteration (ms): 10975.9/10975.9 |
        compute per GPU (TFLOP/s/GPU): 615.5 (avg 615.5) |
        tokens/s/GPU inst/harmonic mean: 11941.8/11941.8

    In the two newer formats the first number is the instantaneous per-step
    value and the second is the backend's running average. We capture the
    instantaneous value (the leading number) so the script can compute its
    own harmonic mean over the post-warmup window, matching legacy
    behaviour. The tokens field was also renamed from
    ``tokens per GPU (tokens/s/GPU)`` to ``tokens/s/GPU inst/harmonic mean``.

    Note the very first ``log_avg_skip_iterations`` steps are still logged in
    the legacy single-value form (no tokens field, warmup/compile TFLOPs), so
    multiple patterns can appear within one run.
    """
    iterations = []
    for line in lines:
        clean = strip_ansi(line)
        if "iteration" not in clean or "elapsed time per iteration" not in clean:
            continue
        if not _line_matches_rank(clean, keep_rank):
            continue

        it = {}
        m = re.search(r"iteration\s+(\d+)/", clean)
        if m:
            it["step"] = int(m.group(1))

        # Capture the leading number, so "346.3", "346.3/340.8" and
        # "346.3 (avg 340.8)" all yield the instantaneous per-step value.
        m = re.search(r"(?:compute|throughput) per GPU \(TFLOP/s/GPU\):\s*([\d.]+)", clean)
        if m:
            it["tflops"] = float(m.group(1))

        # Legacy key: "tokens per GPU (tokens/s/GPU): X".
        # New key:    "tokens/s/GPU inst/harmonic mean: X/Y".
        m = re.search(r"tokens per GPU \(tokens/s/GPU\):\s*([\d.]+)", clean)
        if not m:
            m = re.search(r"tokens/s/GPU inst/harmonic mean:\s*([\d.]+)", clean)
        if m:
            it["tps_per_gpu"] = float(m.group(1))

        m = re.search(
            r"hip mem usage/free/total/usage_ratio:\s*([\d.]+)GB/[\d.]+GB/[\d.]+GB/([\d.]+)%",
            clean,
        )
        if m:
            it["memory_usage"] = float(m.group(1))
            it["memory_usage_pct"] = float(m.group(2))

        if "step" in it:
            iterations.append(it)
    return iterations


def parse_megatron_bridge_iterations(lines: list[str], keep_rank: int | None = None) -> list[dict]:
    """Parse Megatron-Bridge per-step TFLOP lines.

    The value is glued to the unit, with no separator::

        Step Time : 0.79s GPU utilization: 9013.4MODEL_TFLOP/s/GPU

    These lines carry no step index, so steps are numbered in log order
    starting at 1. That lets the shared warmup skip drop the first compile
    steps before the harmonic mean. Tokens and memory are not logged here.
    """
    iterations = []
    step = 0
    for line in lines:
        clean = strip_ansi(line)
        if "MODEL_TFLOP/s/GPU" not in clean:
            continue
        if not _line_matches_rank(clean, keep_rank):
            continue

        m = re.search(r"([\d.]+)\s*MODEL_TFLOP/s/GPU", clean)
        if not m:
            continue
        step += 1
        iterations.append({"step": step, "tflops": float(m.group(1))})
    return iterations


def parse_torchtitan_iterations(lines: list[str], keep_rank: int | None = None) -> list[dict]:
    iterations = []
    for line in lines:
        clean = strip_ansi(line)
        if not re.search(r"step:\s*\d+", clean) or "tflops:" not in clean:
            continue
        if not _line_matches_rank(clean, keep_rank):
            continue

        it = {}
        m = re.search(r"step:\s*(\d+)", clean)
        if m:
            it["step"] = int(m.group(1))

        m = re.search(r"memory:\s*([\d.]+)GiB\(([\d.]+)%\)", clean)
        if m:
            it["memory_usage"] = float(m.group(1))
            it["memory_usage_pct"] = float(m.group(2))

        m = re.search(r"tps:\s*([\d,]+)", clean)
        if m:
            it["tps_per_gpu"] = float(m.group(1).replace(",", ""))

        m = re.search(r"tflops:\s*([\d.]+)", clean)
        if m:
            it["tflops"] = float(m.group(1))

        if "step" in it:
            iterations.append(it)
    return iterations


def parse_maxtext_iterations(lines: list[str], keep_rank: int | None = None) -> list[dict]:
    """Parse MaxText / MaxDiffusion ``completed step:`` lines.

    MaxText example (ANSI-stripped)::

        completed step: 0, seconds: 18.903, TFLOP/s/device: 89.225,
            Tokens/s/device: 1733.519, total_weights: 262144, loss: 12.262

    MaxDiffusion (Primus throughput patch) adds sample/frame rates::

        completed step: 0, seconds: 82.797, TFLOP/s/device: 49.347,
            Tokens/s/device: 956.555, Samples/s/device: 0.0121,
            Frames/s/device: 1.027, loss: 3.018

    Flux logs omit ``Frames/s/device``. Neither backend reports per-step
    memory on this line, so those CSV cells stay empty.
    """
    iterations = []
    for line in lines:
        clean = strip_ansi(line)
        if "completed step:" not in clean or "TFLOP/s/device" not in clean:
            continue
        if not _line_matches_rank(clean, keep_rank):
            continue

        it = {}
        m = re.search(r"completed step:\s*(\d+)", clean)
        if m:
            it["step"] = int(m.group(1))

        m = re.search(r"seconds:\s*([\d.]+)", clean)
        if m:
            it["seconds"] = float(m.group(1))

        m = re.search(r"TFLOP/s/device:\s*([\d.]+)", clean)
        if m:
            it["tflops"] = float(m.group(1))

        m = re.search(r"Tokens/s/device:\s*([\d.,]+)", clean)
        if m:
            it["tps_per_gpu"] = float(m.group(1).replace(",", ""))

        m = re.search(r"loss:\s*([\d.]+)", clean)
        if m:
            it["loss"] = float(m.group(1))

        m = re.search(r"Samples/s/device:\s*([\d.]+)", clean)
        if m:
            it["samples_per_gpu"] = float(m.group(1))

        m = re.search(r"Frames/s/device:\s*([\d.]+)", clean)
        if m:
            it["frames_per_gpu"] = float(m.group(1))

        if "step" in it:
            iterations.append(it)
    return iterations


# Precision tokens as they appear in config names. Order matters: the longer
# spellings must be tried first so ``nanoo_fp8`` is not truncated to ``fp8``
# and ``mxfp8`` / ``mxfp4`` not to ``fp8`` / ``fp4``.
PRECISION_TOKENS = (
    "nanoo_fp8",
    "mxfp8",
    "mxfp4",
    "bf16",
    "fp16",
    "fp8",
    "fp4",
    "bf8",
    "int8",
)

# Only the train-suite suffix is stripped. A ``_sft`` / ``_lora`` marker is
# kept, because it distinguishes two different workloads on the same model
# (qwen3_32b_sft_posttrain vs qwen3_32b_lora_posttrain).
_SUITE_SUFFIX_RE = re.compile(r"[-_](?:pre|post)train$", re.IGNORECASE)


def _split_model_precision(config_name: str) -> tuple[str, str | None]:
    """Split a config name into model and precision.

    The precision token is not always fenced by hyphens. Real examples::

        llama3.1_8B-BF16-pretrain     -> (llama3.1_8B, BF16)
        gdn_1B_BF16-pretrain          -> (gdn_1B,      BF16)
        llama3.1_405B-pretrain-FP8    -> (llama3.1_405B, FP8)
        llama3_8B-nanoo_fp8-pretrain  -> (llama3_8B,  NANOO_FP8)
        mamba_370M-pretrain           -> (mamba_370M, None)

    The previous implementation required the ``-<PREC>-pretrain`` shape, so
    the very common ``<model>_BF16-pretrain`` spelling left precision empty
    and glued into the model name, making BF16 and FP8 rows of the same model
    indistinguishable in the CSV.
    """
    if "/" in config_name:
        config_name = config_name.rsplit("/", 1)[1]

    stem = config_name
    precision: str | None = None

    for token in PRECISION_TOKENS:
        m = re.search(rf"(?:^|[-_]){re.escape(token)}(?=[-_]|$)", stem, re.IGNORECASE)
        if m:
            precision = token.upper()
            stem = stem[: m.start()] + stem[m.end() :]
            break

    # Strip the suite suffix after the precision token has been removed, so
    # "llama3.1_405B-pretrain-FP8" reduces cleanly to "llama3.1_405B".
    stem = _SUITE_SUFFIX_RE.sub("", stem).strip("-_")
    return (stem or config_name), precision


def _precision_from_config(config_text: str) -> str | None:
    """Infer precision from the config body when the name does not say.

    Megatron marks low precision with ``fp8: hybrid``; MaxText uses
    ``quantization: "nanoo_fp8"``. Absence of both is left as unknown rather
    than assumed to be BF16, since the experiment YAML does not always carry
    the dtype -- it can come from the merged model config.
    """
    m = re.search(r"^\s*quantization:\s*[\"']?([A-Za-z0-9_]+)", config_text, re.MULTILINE)
    if m and m.group(1).lower() not in ("none", "null", ""):
        return m.group(1).upper()
    if re.search(r"^\s*fp8:\s*\S", config_text, re.MULTILINE):
        return "FP8"
    return None


def _banner_value(banner: dict, key: str) -> str | None:
    """Read a banner field, mapping the not-available markers to empty.

    The runner writes human-readable placeholders such as
    ``unknown (image not present locally)`` or ``<unset; ...>`` when a value
    could not be determined. Those read fine in a log but are noise in a CSV
    cell, so they become empty here.
    """
    value = (banner.get(key) or "").strip()
    if not value or value.startswith("<") or value.lower().startswith(("unknown", "n/a", "none")):
        return None
    return value


def _populate_from_new_banner(result: dict, banner: dict, config_text: str) -> None:
    """Fill ``result`` from a parsed new-format banner + YAML dump fallback."""
    result["backend"] = banner.get("Framework") or None
    result["config_hash"] = banner.get("Config Hash", "").split()[0] or None
    result["host"] = banner.get("Hostname") or None
    result["timestamp"] = banner.get("Timestamp") or None

    config_name = banner.get("Config Name") or ""
    model_name, precision = _split_model_precision(config_name)
    result["model_name"] = model_name or None
    result["precision"] = precision or _precision_from_config(config_text)
    result["config_path"] = config_name or None
    result["batch_stamp"] = banner.get("Batch Stamp") or None
    result["gpu_check"] = banner.get("GPU check") or None
    attempt = banner.get("Attempt", "")
    result["attempt"] = _maybe_int(attempt.split("/")[0]) if attempt else None

    # Provenance emitted by tools/perf/lib/common.sh. Absent from older logs,
    # which simply leave these columns empty.
    result["docker_image"] = _banner_value(banner, "Docker image")
    result["image_digest"] = _banner_value(banner, "Image digest")
    result["primus_commit"] = _banner_value(banner, "Primus commit")
    result["submodule_pins"] = _banner_value(banner, "Submodule pins")
    result["gpu_model"] = _banner_value(banner, "GPU model")
    # The host's ROCm; logs before the container's was recorded too called it
    # "ROCm version".
    result["rocm_version"] = _banner_value(banner, "ROCm (host)") or _banner_value(banner, "ROCm version")
    result["torch_version"] = _banner_value(banner, "Torch version")
    result["jax_version"] = _banner_value(banner, "JAX version")
    result["slurm_job_id"] = _banner_value(banner, "SLURM job")
    result["nodelist"] = _banner_value(banner, "SLURM nodes")
    result["image_id"] = _banner_value(banner, "Image ID")
    result["container_rocm_version"] = _banner_value(banner, "Container ROCm")
    result["container_hip_version"] = _banner_value(banner, "Container HIP")
    result["cpu_governor"] = _banner_value(banner, "CPU governor")
    result["numa_balancing"] = _banner_value(banner, "NUMA balancing")
    result["transparent_hugepages"] = _banner_value(banner, "Transparent HP")
    result["iommu"] = _banner_value(banner, "IOMMU")
    result["amdgpu_version"] = _banner_value(banner, "amdgpu driver")

    # Cluster shape. "World size" is emitted as "8  (NNODES=1 x GPUS_PER_NODE=8)";
    # the Cluster line is the older format and is parsed as a fallback so
    # existing logs still yield a world size.
    world = banner.get("World size", "")
    result["world_size"] = _maybe_int(world.split()[0]) if world else None

    cluster = banner.get("Cluster", "")
    m = re.search(r"NNODES=(\d+)", cluster) or re.search(r"NNODES=(\d+)", world)
    if m:
        result["nnodes"] = int(m.group(1))
    m = re.search(r"GPUS_PER_NODE=(\d+)", cluster) or re.search(r"GPUS_PER_NODE=(\d+)", world)
    if m:
        result["gpus_per_node"] = int(m.group(1))
    if result["world_size"] is None and result["nnodes"] and result["gpus_per_node"]:
        result["world_size"] = result["nnodes"] * result["gpus_per_node"]

    result["micro_batch_size"] = _maybe_number(banner.get("Micro Batch Size"))
    result["global_batch_size"] = _maybe_number(banner.get("Global Batch Size"))
    result["seq_len"] = _maybe_int(banner.get("Sequence Length"))
    result["total_steps"] = _maybe_int(banner.get("Train Steps/Iters"))

    rep = banner.get("Repetition", "")
    m = re.match(r"\s*(\d+)\s*(?:/\s*(\d+))?", rep)
    if m:
        result["repeat"] = int(m.group(1))
        if m.group(2):
            result["num_reps"] = int(m.group(2))

    backend = result["backend"]
    if backend == "megatron":
        if result["micro_batch_size"] is None:
            mm = re.search(r"micro_batch_size:\s*(\S+)", config_text)
            if mm:
                result["micro_batch_size"] = _maybe_int(mm.group(1))
        if result["global_batch_size"] is None:
            mm = re.search(r"global_batch_size:\s*(\S+)", config_text)
            if mm:
                result["global_batch_size"] = _maybe_int(mm.group(1))
        if result["total_steps"] is None:
            mm = re.search(r"train_iters:\s*(\S+)", config_text)
            if mm:
                result["total_steps"] = _maybe_int(mm.group(1))
        if result["seq_len"] is None:
            mm = re.search(r"seq_length:\s*(\S+)", config_text)
            if mm:
                result["seq_len"] = _maybe_int(mm.group(1))
        # Perf warmup: the updated throughput patch skips the first
        # ``log_avg_skip_iterations`` steps (compile/tuning) from its running
        # harmonic mean, so prefer that value. Fall back to lr_warmup_iters
        # for older logs that don't expose the skip setting.
        mm = re.search(r"log_avg_skip_iterations:\s*(\d+)", config_text)
        if mm:
            result["warmup_steps"] = int(mm.group(1))
        else:
            mm = re.search(r"lr_warmup_iters:\s*(\d+)", config_text)
            if mm:
                result["warmup_steps"] = int(mm.group(1))

    elif backend == "torchtitan":
        if result["micro_batch_size"] is None:
            mm = re.search(r"local_batch_size:\s*(\S+)", config_text)
            if mm:
                result["micro_batch_size"] = _maybe_int(mm.group(1))
        if result["total_steps"] is None:
            mm = re.search(r"\bsteps:\s*(\S+)", config_text)
            if mm:
                result["total_steps"] = _maybe_int(mm.group(1))
        if result["seq_len"] is None:
            mm = re.search(r"seq_len:\s*(\S+)", config_text)
            if mm:
                result["seq_len"] = _maybe_int(mm.group(1))
        mm = re.search(r"warmup_steps:\s*(\d+)", config_text)
        if mm:
            result["warmup_steps"] = int(mm.group(1))

    elif backend == "maxtext":
        # MaxText keys differ from the others:
        #   per_device_batch_size -> micro_batch_size (per-device batch)
        #   max_target_length     -> seq_len
        #   steps                 -> total_steps
        # GBS is computed by the batch runner as MBS * NNODES * GPUS_PER_NODE
        # and lives in the banner; the YAML itself has no global_batch_size
        # field so there's no fallback for it here.
        if result["micro_batch_size"] is None:
            mm = re.search(r"per_device_batch_size:\s*(\S+)", config_text)
            if mm:
                result["micro_batch_size"] = _maybe_number(mm.group(1))
        if result["total_steps"] is None:
            mm = re.search(r"\bsteps:\s*(\S+)", config_text)
            if mm:
                result["total_steps"] = _maybe_int(mm.group(1))
        if result["seq_len"] is None:
            mm = re.search(r"max_target_length:\s*(\S+)", config_text)
            if mm:
                result["seq_len"] = _maybe_int(mm.group(1))
        # MaxText has no explicit perf-warmup setting. The first
        # MIN_PERF_SKIP_STEPS unique logged steps (JIT + autotune) are
        # dropped downstream; leave result["warmup_steps"] as None here so
        # the CSV records the skip that was actually applied.

    elif backend == "maxdiffusion":
        # Diffusion configs have no sequence length / GBS in the banner
        # (logged as N/A). Train length is ``max_train_steps``; MBS is
        # ``per_device_batch_size``. Same minimum step skip as maxtext.
        if result["micro_batch_size"] is None:
            mm = re.search(r"per_device_batch_size:\s*(\S+)", config_text)
            if mm:
                result["micro_batch_size"] = _maybe_number(mm.group(1))
        if result["total_steps"] is None:
            mm = re.search(r"max_train_steps:\s*(\S+)", config_text)
            if mm:
                result["total_steps"] = _maybe_int(mm.group(1))


def _populate_from_legacy_header(result: dict, lines: list[str], config_text: str) -> bool:
    """Fill ``result`` from a legacy-format header. Return ``False`` if unrecognised."""
    if len(lines) < 6:
        return False

    line1 = strip_ansi(lines[0].strip())
    line3 = strip_ansi(lines[2].strip())
    line4 = strip_ansi(lines[3].strip()) if len(lines) > 3 else ""

    cmd_params: dict = {}
    config_name = ""

    if re.match(r"EXP=examples/", line4):
        m = re.search(r"seq_len=(\d+)", line3)
        if m:
            result["seq_len"] = int(m.group(1))

        exp_match = re.match(r"EXP=examples/(\w+)/configs/[^/]+/(.+)\.yaml", line4)
        if not exp_match:
            return False
        result["backend"] = exp_match.group(1)
        config_name = exp_match.group(2)

        line5 = strip_ansi(lines[4].strip())
        cmd_params = parse_cmd_params(line5)

        line6 = strip_ansi(lines[5].strip())
        m = re.search(r"repeat:\s*(\d+)", line6)
        if m:
            result["repeat"] = int(m.group(1))

    elif re.search(r"print (?:the )?config file:\s*examples/", line3):
        cfg_match = re.search(
            r"print (?:the )?config file:\s*examples/(\w+)/configs/[^/]+/(.+)\.yaml",
            line3,
        )
        if not cfg_match:
            return False
        result["backend"] = cfg_match.group(1)
        config_name = cfg_match.group(2)

        m = re.search(r"for rep\s+(\d+)", line1)
        if m:
            result["repeat"] = int(m.group(1))

    else:
        return False

    model_name, precision = _split_model_precision(config_name)
    result["model_name"] = model_name
    result["precision"] = precision

    backend = result["backend"]
    if backend == "megatron":
        result["micro_batch_size"] = _int_from_cmd_or_config(
            cmd_params, "micro_batch_size", config_text, r"micro_batch_size:\s*(\d+)"
        )
        result["global_batch_size"] = _int_from_cmd_or_config(
            cmd_params, "global_batch_size", config_text, r"global_batch_size:\s*(\d+)"
        )
        result["total_steps"] = _int_from_cmd_or_config(
            cmd_params, "train_iters", config_text, r"train_iters:\s*(\d+)"
        )
        if result["seq_len"] is None:
            m = re.search(r"seq_length:\s*(\d+)", config_text)
            if m:
                result["seq_len"] = int(m.group(1))
        m = re.search(r"lr_warmup_iters:\s*(\d+)", config_text)
        if m:
            result["warmup_steps"] = int(m.group(1))

    elif backend == "torchtitan":
        local_bs = cmd_params.get("training.local_batch_size")
        if local_bs and local_bs.isdigit():
            result["micro_batch_size"] = int(local_bs)
        else:
            m = re.search(r"local_batch_size:\s*(\d+)", config_text)
            if m:
                result["micro_batch_size"] = int(m.group(1))

        steps = cmd_params.get("training.steps")
        if steps and steps.isdigit():
            result["total_steps"] = int(steps)
        else:
            m = re.search(r"\bsteps:\s*(\d+)", config_text)
            if m:
                result["total_steps"] = int(m.group(1))

        if result["seq_len"] is None:
            m = re.search(r"seq_len:\s*(\d+)", config_text)
            if m:
                result["seq_len"] = int(m.group(1))

        m = re.search(r"warmup_steps:\s*(\d+)", config_text)
        if m:
            result["warmup_steps"] = int(m.group(1))

    return True


def _int_from_cmd_or_config(
    cmd_params: dict, cmd_key: str, config_text: str, config_pattern: str
) -> int | None:
    """Get an integer from CMD params first, falling back to config regex."""
    val = cmd_params.get(cmd_key)
    if val and val.isdigit():
        return int(val)
    m = re.search(config_pattern, config_text)
    if m:
        return int(m.group(1))
    return None


# --------------------------------------------------------------------------
# Why a run failed, and warnings worth surfacing
# --------------------------------------------------------------------------

# Searched in this order; the first category found anywhere in the run output
# wins. A bad NIC or an OOM usually also produces NCCL errors and tracebacks
# further down, so the specific causes come before the generic ones.
FAILURE_PATTERNS = (
    ("nic_error", re.compile(r"no socket interface found", re.IGNORECASE)),
    ("oom", re.compile(r"out of memory|OutOfMemoryError|RESOURCE_EXHAUSTED", re.IGNORECASE)),
    # Not a bare "SIGSEGV": Primus logs "... to avoid SIGSEGV ..." when it
    # applies its dataloader patch.
    (
        "segfault",
        re.compile(r"Signal 11 \(SIGSEGV\)|Segmentation fault|signal SIGSEGV|SIGSEGV received|\*\*\* SIGSEGV"),
    ),
    (
        "nccl_error",
        re.compile(
            r"ncclRemoteError|ncclInternalError|ncclSystemError|ncclUnhandledCudaError|"
            r"NCCL error|RCCL error|Watchdog caught collective operation timeout"
        ),
    ),
)
# Cheap first pass, so a multi-million-line log only pays for the regexes
# above on the few lines that can match.
# Applied to the lower-cased line: re.IGNORECASE over these alternatives is
# several times slower than lower() plus a case-sensitive search.
FAILURE_PREFILTER_RE = re.compile(
    r"socket interface|out of memory|outofmemory|resource_exhausted|sigsegv|"
    r"segmentation fault|nccl|rccl|watchdog caught|traceback|error|exception"
)
# Primus's own patch log lines quote error and version strings they guard
# against; they are never the failure.
PRIMUS_NOISE = "[Patch"
TRACEBACK_MARKER = "Traceback (most recent call last)"
# An exception line once its log prefix is stripped, e.g. "ValueError: bad".
PY_EXCEPTION_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception))(?::|$)")
PY_EXCEPTION_RE = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception)):\s")
GENERIC_EXCEPTIONS = ("ChildFailedError",)
# PyTorch's OOM message says how much of the GPU this process held. When that
# is a small part of the capacity, something else was occupying the GPU.
_SIZE = r"([\d.]+) (bytes|B|KiB|MiB|GiB)"
OOM_TOTAL_RE = re.compile(r"total capacity of " + _SIZE, re.IGNORECASE)
OOM_PROCESS_RE = re.compile(r"this process has " + _SIZE + r" memory in use", re.IGNORECASE)
OOM_ALLOCATED_RE = re.compile(_SIZE + r" is allocated by PyTorch", re.IGNORECASE)
OOM_RESERVED_RE = re.compile(_SIZE + r" is reserved by PyTorch but unallocated", re.IGNORECASE)
_GIB_PER_UNIT = {"bytes": 1.0 / 2**30, "b": 1.0 / 2**30, "kib": 1.0 / 2**20, "mib": 1.0 / 2**10, "gib": 1.0}
VERSION_WARNING_RE = re.compile(r"([\w.\-]*\s*version mismatch\b.*)", re.IGNORECASE)
# Leading "[timestamp][rank-N/M][LEVEL] [file:line] :" decorations.
LOG_PREFIX_RE = re.compile(r"^(?:\s*\[[^\]]*\])+\s*(?::\s*)?")
MAX_DETAIL_CHARS = 300


def _clean_detail(line: str) -> str:
    text = LOG_PREFIX_RE.sub("", strip_ansi(line).strip()).strip()
    return text[:MAX_DETAIL_CHARS]


def run_output_lines(lines: list[str]) -> list[str]:
    """The launcher's output, without banner, config dump or footer."""
    start = end = None
    for i, line in enumerate(lines):
        if start is None and "Begin Run Output" in line:
            start = i + 1
        elif start is not None and "End Run Output" in line:
            end = i
    if start is None:
        return lines
    return lines[start:end]


def _gib(match: re.Match | None) -> float | None:
    # The size regexes ignore case, so the unit lookup must too.
    return float(match.group(1)) * _GIB_PER_UNIT[match.group(2).lower()] if match else None


def _oom_is_contended(oom_lines: list[str]) -> bool:
    """True when a PyTorch OOM message shows this process held less than half
    of the GPU: what it holds is "this process has X memory in use" when
    present, otherwise allocated plus reserved-but-unallocated."""
    for line in oom_lines:
        total = _gib(OOM_TOTAL_RE.search(line))
        if not total:
            continue
        held = _gib(OOM_PROCESS_RE.search(line))
        if held is None:
            allocated = _gib(OOM_ALLOCATED_RE.search(line))
            if allocated is None:
                continue
            held = allocated + (_gib(OOM_RESERVED_RE.search(line)) or 0.0)
        return held < 0.5 * total
    return False


def _scan_failure_text(run_lines: list[str]) -> tuple[str | None, str | None]:
    first: dict[str, str] = {}
    oom_lines: list[str] = []
    traceback_seen = False
    loose_exception = None
    for line in run_lines:
        if not FAILURE_PREFILTER_RE.search(line.lower()) or PRIMUS_NOISE in line:
            continue
        clean = strip_ansi(line)
        for category, pattern in FAILURE_PATTERNS:
            if pattern.search(clean):
                if category not in first:
                    first[category] = clean
                if category == "oom":
                    oom_lines.append(clean)
        # The root-cause exception is the last line of the first traceback,
        # not whichever "XError:" happens to be logged first (a benign import
        # warning, for instance).
        if "python_error" not in first:
            if TRACEBACK_MARKER in clean:
                traceback_seen = True
            elif traceback_seen:
                m = PY_EXCEPTION_LINE_RE.match(_clean_detail(clean))
                if m and not m.group(1).endswith(GENERIC_EXCEPTIONS):
                    first["python_error"] = clean
            elif loose_exception is None:
                m = PY_EXCEPTION_RE.search(clean)
                if m and not m.group(1).endswith(GENERIC_EXCEPTIONS):
                    loose_exception = clean[m.start() :]
    if "python_error" not in first and loose_exception:
        first["python_error"] = loose_exception
    for category in [c for c, _ in FAILURE_PATTERNS] + ["python_error"]:
        if category not in first:
            continue
        if category == "oom" and _oom_is_contended(oom_lines):
            category = "gpu_contended"
            return category, _clean_detail(next(l for l in oom_lines if OOM_TOTAL_RE.search(l)))
        return category, _clean_detail(first[category])
    return None, None


def _last_output_line(run_lines: list[str]) -> str | None:
    for line in reversed(run_lines):
        text = _clean_detail(line)
        if text and not text.startswith(">>>"):
            return text
    return None


def legacy_footer_reason(footer: dict, run_lines: list[str], exit_code: int | None) -> str | None:
    """Rebuild a ``Failure reason`` for logs written before the footer had one.

    Older single-node runners wrote exit 130 on Ctrl+C; the older multi-node
    runner marked watchdog kills with ``Failure pattern? : true`` and a
    ``>>> ... TIMEOUT`` line, whatever the exit code.
    """
    if footer.get("failure_pattern_flag") == "true":
        if any(">>>" in line and "TIMEOUT" in line for line in run_lines[-200:]):
            return "timeout: stopped by the older runner's MAX_RUN_SECONDS"
        return "fail_pattern: failure text matched the older runner's FAIL_PATTERN"
    if exit_code is None or exit_code == 0:
        return None
    if exit_code == 130:
        return "interrupted: exit 130 from an older runner"
    if 128 < exit_code < 160:
        try:
            name = signal.Signals(exit_code - 128).name
        except ValueError:
            name = f"signal {exit_code - 128}"
        return f"signal: killed by {name} (exit {exit_code})"
    return f"exit: launcher exited with code {exit_code}"


def classify_failure(
    run_lines: list[str], footer_reason: str | None, exit_code: int | None
) -> tuple[str | None, str | None]:
    """Return ``(failure_reason, failure_detail)`` for a run that did not pass.

    The runner's own verdict wins when it is definitive (it stopped a hang,
    or the batch was interrupted). Otherwise the cause in the output is more
    useful than the bare exit status, which comes last. A failed run always
    gets a reason, ``unknown`` at worst.
    """
    footer_cat, _, footer_detail = (footer_reason or "").partition(":")
    footer_cat = footer_cat.strip()
    footer_detail = footer_detail.strip() or None

    # "steps": the diffusion_dlrm run listing the diffusion_bench steps that failed.
    if footer_cat in ("timeout", "idle", "interrupted", "steps"):
        return footer_cat, footer_detail
    text_cat, text_detail = _scan_failure_text(run_lines)
    if text_cat:
        return text_cat, text_detail
    if footer_cat == "signal":
        return ("killed" if "SIGKILL" in (footer_detail or "") else "signal"), footer_detail
    if footer_cat in ("fail_pattern", "exit"):
        return footer_cat, _last_output_line(run_lines) or footer_detail
    return "unknown", _last_output_line(run_lines)


def version_warnings(run_lines: list[str], limit: int = 10) -> list[str]:
    """Distinct ``... version mismatch ...`` messages (e.g. aiter installed vs
    expected), which affect performance but never fail a run."""
    seen: list[str] = []
    for line in run_lines:
        if "mismatch" not in line or PRIMUS_NOISE in line:
            continue
        m = VERSION_WARNING_RE.search(strip_ansi(line))
        if m:
            text = m.group(1).strip()[:MAX_DETAIL_CHARS]
            if text not in seen:
                seen.append(text)
                if len(seen) >= limit:
                    break
    return seen


def _spread(values: list[float]) -> dict:
    """min / max / sample stdev / coefficient of variation of per-step values."""
    if not values:
        return {"min": None, "max": None, "stdev": None, "cv": None}
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    mean = statistics.fmean(values)
    return {"min": min(values), "max": max(values), "stdev": stdev, "cv": stdev / mean if mean else None}


FAILED_ATTEMPT_RE = re.compile(r"-failed-attempt(\d+)_")


def parse_log_file(filepath: str) -> dict | None:
    """Parse a single log file and return extracted metadata + statistics."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if len(lines) < 6:
        return None

    result: dict = {
        "filename": os.path.basename(filepath),
        "backend": None,
        "model_name": None,
        "precision": None,
        "seq_len": None,
        "micro_batch_size": None,
        "global_batch_size": None,
        "warmup_steps": None,
        "repeat": None,
        "total_steps": None,
        "config_hash": None,
        "host": None,
        "timestamp": None,
        "nnodes": None,
        "gpus_per_node": None,
        "world_size": None,
        "docker_image": None,
        "image_digest": None,
        "primus_commit": None,
        "submodule_pins": None,
        "gpu_model": None,
        "rocm_version": None,
        "torch_version": None,
        "jax_version": None,
        "slurm_job_id": None,
        "nodelist": None,
        "exit_code": None,
        "elapsed_sec": None,
        "num_iterations": 0,
        "num_post_warmup": 0,
        "hmean_tps_per_gpu": None,
        "hmean_tflops": None,
        "hmean_samples_per_gpu": None,
        "hmean_frames_per_gpu": None,
        "amean_memory_usage": None,
        "amean_memory_usage_pct": None,
        "status": None,
        "failure_reason": None,
        "failure_detail": None,
        "attempt": None,
        "failed_attempt": None,
        "config_path": None,
        "batch_stamp": None,
        "num_reps": None,
        "gpu_check": None,
        "tps_per_gpu_min": None,
        "tps_per_gpu_max": None,
        "tps_per_gpu_stdev": None,
        "tps_per_gpu_cv": None,
        "tflops_min": None,
        "tflops_max": None,
        "tflops_stdev": None,
        "tflops_cv": None,
        "version_warnings": [],
        "parser_version": PARSER_VERSION,
        "image_id": None,
        "container_rocm_version": None,
        "container_hip_version": None,
        "cpu_governor": None,
        "numa_balancing": None,
        "transparent_hugepages": None,
        "iommu": None,
        "amdgpu_version": None,
        "rec_per_s": None,
        "rec_per_s_cv": None,
        "step_log": None,
    }

    config_text = parse_config_section(lines)

    banner = parse_new_banner(lines)
    if banner:
        _populate_from_new_banner(result, banner, config_text)
    else:
        if not _populate_from_legacy_header(result, lines, config_text):
            return None

    footer = parse_run_footer(lines)
    result["exit_code"] = footer.get("exit_code")
    result["elapsed_sec"] = footer.get("elapsed_sec")

    m = FAILED_ATTEMPT_RE.search(result["filename"])
    if m:
        result["failed_attempt"] = int(m.group(1))

    run_lines = run_output_lines(lines)
    exit_code = result["exit_code"]
    footer_reason = footer.get("failure_reason") or None
    if footer_reason is None:
        footer_reason = legacy_footer_reason(footer, run_lines, exit_code)
    if exit_code is None:
        result["status"] = "incomplete" if banner else "unknown"
    elif footer_reason and footer_reason.startswith("interrupted"):
        result["status"] = "interrupted"
    elif exit_code == 0 and footer_reason in (None, "none") and result["failed_attempt"] is None:
        result["status"] = "ok"
    else:
        result["status"] = "failed"
    if result["status"] in ("failed", "interrupted"):
        result["failure_reason"], result["failure_detail"] = classify_failure(
            run_lines, footer_reason, exit_code
        )
    elif result["status"] == "incomplete":
        # Still running, or the batch was killed: only a cause visible in the
        # output is worth reporting. Legacy logs without a footer ("unknown")
        # are not scanned at all, since they may well have succeeded.
        result["failure_reason"], result["failure_detail"] = _scan_failure_text(run_lines)
    result["version_warnings"] = version_warnings(run_lines)

    backend = result["backend"]
    if backend == "megatron":
        keep_rank = _detect_iter_logging_rank(lines, "elapsed time per iteration")
        iterations = parse_megatron_iterations(lines, keep_rank=keep_rank)
    elif backend == "megatron_bridge":
        keep_rank = _detect_iter_logging_rank(lines, "MODEL_TFLOP/s/GPU")
        iterations = parse_megatron_bridge_iterations(lines, keep_rank=keep_rank)
    elif backend == "torchtitan":
        keep_rank = _detect_iter_logging_rank(lines, "tflops:")
        iterations = parse_torchtitan_iterations(lines, keep_rank=keep_rank)
    elif backend in ("maxtext", "maxdiffusion"):
        keep_rank = _detect_iter_logging_rank(lines, "TFLOP/s/device")
        iterations = parse_maxtext_iterations(lines, keep_rank=keep_rank)
    elif backend == DIFFUSION_DLRM_BACKEND:
        # Expanded into one row per model by parse_log.
        return result
    else:
        return None

    result["num_iterations"] = len(iterations)
    if not iterations:
        return result

    warmup = result["warmup_steps"] or 0

    steps_present = [it.get("step", 0) for it in iterations]
    max_step = max(steps_present) if steps_present else 0
    sorted_steps = sorted(set(steps_present))

    # Drop at least the first MIN_PERF_SKIP_STEPS unique logged steps.
    # ``step > threshold`` with threshold = sorted_steps[N-1] excludes those
    # N steps whether numbering is 0-based (maxtext) or 1-based (megatron).
    if len(sorted_steps) > MIN_PERF_SKIP_STEPS:
        min_threshold = sorted_steps[MIN_PERF_SKIP_STEPS - 1]
    elif len(sorted_steps) >= 2:
        min_threshold = sorted_steps[-2]
    else:
        min_threshold = -1

    # A configured warmup can be unrelated to perf measurement and larger than
    # the (often short) benchmark run -- e.g. torchtitan reports the LR-schedule
    # ``lr_scheduler.warmup_steps`` (commonly 200) while the benchmark runs only
    # ~20 steps. When it would exclude every logged step, fall back to the
    # minimum skip instead of dropping everything.
    if warmup >= max_step:
        warmup = min_threshold
    else:
        warmup = max(warmup, min_threshold)

    post_warmup = [it for it in iterations if it.get("step", 0) > warmup]
    if not post_warmup:
        post_warmup = iterations

    # Reflect the perf-warmup actually applied so the CSV stays self-consistent
    # with ``num_post_warmup`` (the configured LR warmup may have been clamped).
    result["warmup_steps"] = warmup

    tps_vals = [it["tps_per_gpu"] for it in post_warmup if it.get("tps_per_gpu", 0) > 0]
    tflops_vals = [it["tflops"] for it in post_warmup if it.get("tflops", 0) > 0]
    samples_vals = [it["samples_per_gpu"] for it in post_warmup if it.get("samples_per_gpu", 0) > 0]
    frames_vals = [it["frames_per_gpu"] for it in post_warmup if it.get("frames_per_gpu", 0) > 0]
    mem_vals = [it["memory_usage"] for it in post_warmup if "memory_usage" in it]
    mem_pct_vals = [it["memory_usage_pct"] for it in post_warmup if "memory_usage_pct" in it]

    result["num_post_warmup"] = len(post_warmup)
    result["hmean_tps_per_gpu"] = harmonic_mean(tps_vals)
    result["hmean_tflops"] = harmonic_mean(tflops_vals)
    result["hmean_samples_per_gpu"] = harmonic_mean(samples_vals)
    result["hmean_frames_per_gpu"] = harmonic_mean(frames_vals)
    result["amean_memory_usage"] = arithmetic_mean(mem_vals)
    result["amean_memory_usage_pct"] = arithmetic_mean(mem_pct_vals)

    # Spread across post-warmup steps: a low mean from a few slow steps reads
    # very differently from a uniformly slow run.
    for prefix, values in (("tps_per_gpu", tps_vals), ("tflops", tflops_vals)):
        for stat, value in _spread(values).items():
            result[f"{prefix}_{stat}"] = value

    return result


# ---------------------------------------------------------------------------
# diffusion_dlrm: one run log, one row per diffusion model / DLRM precision
# ---------------------------------------------------------------------------

DIFFUSION_DLRM_BACKEND = "diffusion_dlrm"
# run_benchmarks.sh logs the results folder it creates, then its plan:
#   [diffusion_bench 2026-10-14 18:24:21] Results  : /path/to/folder
#   [diffusion_bench 2026-10-14 18:24:21] Diffusion: flux-dev  batch size 16  ...
#   [diffusion_bench 2026-10-14 18:24:21] DLRM     : TF32[  batch size 32768]
DIFFUSION_BENCH_RESULTS_RE = re.compile(r"^\[diffusion_bench [^\]]*\] Results\s+:\s+(\S.*?)\s*$")
DIFFUSION_BENCH_PLAN_DIFFUSION_RE = re.compile(r"^\[diffusion_bench [^\]]*\] Diffusion: (\S+)\s+batch size (\d+)")
DIFFUSION_BENCH_PLAN_DLRM_RE = re.compile(r"^\[diffusion_bench [^\]]*\] DLRM\s+: (\S+)(?:\s+batch size (\d+))?")
_diffusion_bench_parser = None


def _load_diffusion_bench_parser():
    """``diffusion_bench/parse_results.py``, which knows its folder layout."""
    global _diffusion_bench_parser
    if _diffusion_bench_parser is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusion_bench", "parse_results.py")
        spec = importlib.util.spec_from_file_location("diffusion_bench_parse_results", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _diffusion_bench_parser = module
    return _diffusion_bench_parser


def _diffusion_bench_run_dir(run_lines: list[str], log_dir: str) -> str | None:
    for line in run_lines:
        m = DIFFUSION_BENCH_RESULTS_RE.match(strip_ansi(line).strip())
        if not m:
            continue
        path = m.group(1)
        # Next to the log first, so a moved or copied result directory reads
        # its own folder rather than the original's.
        local = os.path.join(log_dir, DIFFUSION_DLRM_BACKEND, os.path.basename(path.rstrip("/")))
        if os.path.isdir(local):
            return local
        return path if os.path.isdir(path) else None
    return None


def _diffusion_bench_plan(run_lines: list[str]) -> list[dict]:
    """The steps diffusion_bench announced, in order, as parse_results-style rows."""
    plan = []
    for line in run_lines:
        clean = strip_ansi(line).strip()
        m = DIFFUSION_BENCH_PLAN_DIFFUSION_RE.match(clean)
        if m:
            plan.append({"suite": "diffusion", "benchmark": m.group(1), "batch_size": m.group(2), "precision": "bf16"})
            continue
        m = DIFFUSION_BENCH_PLAN_DLRM_RE.match(clean)
        if m:
            plan.append({"suite": "dlrm", "benchmark": "dlrm-v2", "batch_size": m.group(2), "precision": m.group(1)})
    return plan


def _diffusion_bench_step(r: dict) -> str:
    """diffusion_bench's step name (status.tsv, logs/) for a result row."""
    if r["suite"] == "diffusion":
        return f"diffusion_{r['benchmark']}_bs{r['batch_size']}"
    return f"dlrm_{r['precision']}"


def _row_from_step_name(step: str) -> dict:
    if step.startswith("dlrm_"):
        return {"suite": "dlrm", "benchmark": "dlrm-v2", "batch_size": None, "precision": step[len("dlrm_") :]}
    model, _, bs = step[len("diffusion_") :].rpartition("_bs")
    return {"suite": "diffusion", "benchmark": model, "batch_size": bs or None, "precision": "bf16"}


def _runs_summary_iterations(run_dir: str, r: dict) -> int | None:
    path = os.path.join(run_dir, "diffusion", f"{r['benchmark']}_bs{r['batch_size']}", "runs_summary.csv")
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    return _maybe_int(rows[-1].get("num_iterations")) if rows else None


def _read_lines(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except OSError:
        return []


def _diffusion_bench_outcome(
    step_status: str | None, step_log: str, exit_code: int | None, run: dict
) -> tuple[str, str | None, str | None]:
    """``(status, failure_reason, failure_detail)`` of one diffusion_bench step."""
    if step_status == "success":
        return "ok", None, None
    if step_status in (None, "", "unknown"):
        # No outcome recorded: in flight when the run ended, or never started.
        if run["status"] in ("interrupted", "incomplete"):
            return run["status"], run["failure_reason"], run["failure_detail"]
        return "failed", "not_run", "diffusion_bench recorded no outcome for this step"
    if step_status == "timeout":
        return "failed", "timeout", "stopped by diffusion_bench's step_timeout"
    if step_status == "no-metrics":
        return "failed", "no_metrics", "the step succeeded but logged no throughput"
    # Everything else failed, including not_run: the runner never asks for a
    # dry run, so a launcher still at not_run crashed before training.
    lines = _read_lines(step_log)
    category, detail = _scan_failure_text(lines)
    if step_status in ("oom", "oom-skipped"):
        if category in ("oom", "gpu_contended"):
            return "failed", category, detail
        return "failed", "oom", f"diffusion_bench reported {step_status}"
    if category:
        return "failed", category, detail
    return "failed", "exit", _last_output_line(lines) or f"step exited with code {exit_code}"


def diffusion_dlrm_rows(run: dict, filepath: str, lines: list[str]) -> list[dict]:
    """Rows for the steps of a diffusion_dlrm run; the run's own row when
    diffusion_bench never got as far as creating its results folder.

    Every step diffusion_bench planned, recorded in status.tsv or left a
    folder for gets a row, so a step that failed before writing its folder
    (a DLRM step stopped by step_timeout, say) or never started still shows.
    """
    log_dir = os.path.dirname(os.path.abspath(filepath))
    run_lines = run_output_lines(lines)
    run_dir = _diffusion_bench_run_dir(run_lines, log_dir)
    if run_dir is None:
        return [run]
    parser = _load_diffusion_bench_parser()
    steps = parser.read_status(run_dir)
    measured = {
        _diffusion_bench_step(r): r for r in parser.parse_diffusion(run_dir, steps) + parser.parse_dlrm(run_dir, steps)
    }
    planned = {_diffusion_bench_step(r): r for r in _diffusion_bench_plan(run_lines)}
    order = list(planned)
    for step in list(steps) + list(measured):
        if step.startswith(("diffusion_", "dlrm_")) and step not in order:
            order.append(step)

    rows = []
    for step in order:
        step_row = steps.get(step, {})
        r = measured.get(step) or planned.get(step) or _row_from_step_name(step)
        suite = r["suite"]
        step_log = os.path.join(run_dir, "logs", f"{step}.log")
        bs = _maybe_int(str(r["batch_size"])) if r.get("batch_size") is not None else None
        gpus = r.get("num_gpus") or run["gpus_per_node"]
        iterations = _runs_summary_iterations(run_dir, r) if suite == "diffusion" else None

        row = dict(run)
        row["version_warnings"] = list(run["version_warnings"])
        row.update(
            {
                "backend": suite,
                "model_name": r["benchmark"],
                "precision": str(r.get("precision") or "").upper() or None,
                "seq_len": None,
                "micro_batch_size": bs,
                "global_batch_size": bs * gpus if bs and gpus else None,
                "nnodes": 1,
                "gpus_per_node": gpus,
                "world_size": gpus,
                "warmup_steps": None,
                # TRAIN_STEPS sets the diffusion models' num_iterations only.
                "total_steps": iterations or (run["total_steps"] if suite == "diffusion" else None),
                "num_iterations": None,
                "num_post_warmup": None,
                "hmean_tflops": r.get("tflops_per_gpu"),
                "hmean_samples_per_gpu": r.get("samples_per_s_per_gpu"),
                # DLRMBenchmark's figure: iterations/s x the per-GPU BATCH_SIZE.
                "rec_per_s": r.get("rec_per_s"),
                "rec_per_s_cv": r.get("rec_per_s_cv"),
                "exit_code": _maybe_int(step_row.get("exit_code")),
                "elapsed_sec": _maybe_int(step_row.get("duration_s")),
                "step_log": os.path.relpath(step_log, log_dir),
            }
        )
        # A step missing from status.tsv was in flight when the run ended, even
        # though AMDiffusionBenchmark's runs_summary.csv already says not_run.
        # parse_results refines a recorded status from that file.
        if not step_row:
            step_status = None
        elif step in measured:
            step_status = r.get("status")
        else:
            step_status = step_row.get("status")
        row["status"], row["failure_reason"], row["failure_detail"] = _diffusion_bench_outcome(
            step_status, step_log, row["exit_code"], run
        )
        rows.append(row)
    return rows or [run]


def parse_log(filepath: str) -> list[dict]:
    """The result rows of one log: one for a Primus run, one per model for a
    diffusion_dlrm run, none for a file that is not a run log."""
    result = parse_log_file(filepath)
    if result is None:
        return []
    if result["backend"] == DIFFUSION_DLRM_BACKEND:
        return diffusion_dlrm_rows(result, filepath, _read_lines(filepath))
    return [result]


CSV_FIELDS = [
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
    "warmup_steps",
    "repeat",
    "total_steps",
    "num_iterations",
    "num_post_warmup",
    "hmean_tps_per_gpu",
    "hmean_tflops",
    "hmean_samples_per_gpu",
    "hmean_frames_per_gpu",
    "amean_memory_usage",
    "amean_memory_usage_pct",
    "exit_code",
    "elapsed_sec",
    # Provenance: which build, which code, which hardware produced the row.
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
    # Added in parser version 2; kept at the end so positional readers of
    # older CSVs keep working.
    "status",
    "failure_reason",
    "failure_detail",
    "attempt",
    "failed_attempt",
    "config_path",
    "batch_stamp",
    "num_reps",
    "gpu_check",
    "tps_per_gpu_min",
    "tps_per_gpu_max",
    "tps_per_gpu_stdev",
    "tps_per_gpu_cv",
    "tflops_min",
    "tflops_max",
    "tflops_stdev",
    "tflops_cv",
    "version_warnings",
    "parser_version",
    # Added in parser version 3.
    "image_id",
    "container_rocm_version",
    "container_hip_version",
    "cpu_governor",
    "numa_balancing",
    "transparent_hugepages",
    "iommu",
    "amdgpu_version",
    # Added in parser version 4, for diffusion_dlrm rows.
    "rec_per_s",
    "rec_per_s_cv",
    "step_log",
]


CSV_NAME = "results.csv"
JSON_NAME = "results.json"


def tools_commit() -> str:
    """Commit of the checkout this parser runs from, for the JSON metadata."""
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "describe", "--always", "--dirty", "--abbrev=12"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# Columns that predate parser version 2 keep their four-decimal format. A
# coefficient of variation is often 0.002, and a fractional batch size is a
# plain 0.5, so these use significant digits instead.
SIG_DIGIT_FIELDS = {"micro_batch_size", "global_batch_size", "tps_per_gpu_cv", "tflops_cv", "rec_per_s_cv"}


def _csv_cell(field: str, val):
    if isinstance(val, float):
        return f"{val:.6g}" if field in SIG_DIGIT_FIELDS else f"{val:.4f}"
    if val is None:
        return ""
    if isinstance(val, list):
        return " | ".join(val)
    return val


def write_outputs(results: list[dict], input_dir: str) -> tuple[str, str]:
    """Write results.csv and results.json into ``input_dir``; return their paths.

    Each file is written beside its final name and then renamed over it, so
    an interrupted run never leaves a truncated file in place of the
    previous one.
    """
    csv_path = os.path.join(input_dir, CSV_NAME)
    json_path = os.path.join(input_dir, JSON_NAME)

    with open(csv_path + ".tmp", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({field: _csv_cell(field, r.get(field)) for field in CSV_FIELDS})

    with open(json_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(
            {
                "parser_version": PARSER_VERSION,
                "tools_commit": tools_commit(),
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "result_dir": input_dir,
                "runs": [{field: r.get(field) for field in CSV_FIELDS} for r in results],
            },
            f,
            indent=2,
        )
        f.write("\n")

    os.replace(csv_path + ".tmp", csv_path)
    os.replace(json_path + ".tmp", json_path)
    return csv_path, json_path


def _describe(result: dict) -> str:
    text = (
        f"{result['backend']} | {result['model_name']} | precision={result['precision']} | "
        f"repeat={result['repeat']} | status={result['status']}"
    )
    if result["failure_reason"]:
        text += f" ({result['failure_reason']})"
    if result["backend"] == "dlrm":
        rec = result["rec_per_s"]
        return f"{text} | rec_per_s={f'{rec:.0f}' if rec else 'N/A'}"
    if result["backend"] == "diffusion":
        tflops = result["hmean_tflops"]
        samples = result["hmean_samples_per_gpu"]
        return (
            f"{text} | tflops={f'{tflops:.2f}' if tflops else 'N/A'} | "
            f"samples_per_gpu={f'{samples:.3f}' if samples else 'N/A'}"
        )
    if result["num_iterations"] == 0:
        return text + " | no iteration metrics found"
    tflops = result["hmean_tflops"]
    tps = result["hmean_tps_per_gpu"]
    return (
        f"{text} | warmup={result['warmup_steps']} | "
        f"hmean_tflops={f'{tflops:.2f}' if tflops else 'N/A'} | "
        f"hmean_tps={f'{tps:.2f}' if tps else 'N/A'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract benchmarking results from Primus training log files."
    )
    parser.add_argument("input_dir", help="Directory containing .txt / .log log files")
    parser.add_argument(
        "--quiet", action="store_true", help="Print only the totals and the output paths."
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        return 1

    log_files = sorted(
        glob.glob(os.path.join(input_dir, "*.txt")) + glob.glob(os.path.join(input_dir, "*.log"))
    )
    # Exit codes: 0 written, 1 a log could not be parsed (results are still
    # written for the others), 3 no run logs here yet. 2 is left to Python
    # and argparse, so the batch runners can tell these apart.
    if not log_files:
        print(f"No .txt or .log files found in {input_dir}")
        return 3

    results = []
    errors = 0
    for fpath in log_files:
        fname = os.path.basename(fpath)
        try:
            rows = parse_log(fpath)
        except Exception as e:
            errors += 1
            print(f"Processing: {fname}\n  -> Error: {e}")
            continue
        if rows:
            results.extend(rows)
            if not args.quiet:
                print(f"Processing: {fname}")
                for row in rows:
                    print(f"  -> {_describe(row)}")
        elif not args.quiet:
            print(f"Processing: {fname}\n  -> Skipped (not a Primus run log)")

    if not results:
        print("No results extracted.")
        return 1 if errors else 3

    csv_path, json_path = write_outputs(results, input_dir)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    totals = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    print(f"Extracted {len(results)} runs ({totals}) from {len(log_files)} files.")
    print(f"Results written to: {csv_path}")
    print(f"                    {json_path}")
    if errors:
        print(f"{errors} log(s) could not be parsed and are missing from the results.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

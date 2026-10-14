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
are written to a timestamped CSV in the input directory. MaxDiffusion also
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
"""

# Keeps the annotations below unevaluated, so the batch runners can call this
# script on hosts whose python3 predates 3.10.
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime

# Bumped whenever a column is added or its meaning changes.
PARSER_VERSION = 2

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
    ``# Primus Benchmark Run`` (single-node, ``primus-bench-batch.sh``) or
    ``# Primus Multi-Node Benchmark Run`` (multi-node,
    ``primus-bench-batch-multinode.sh``), and ending at the
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
    result["rocm_version"] = _banner_value(banner, "ROCm version")
    result["torch_version"] = _banner_value(banner, "Torch version")
    result["jax_version"] = _banner_value(banner, "JAX version")
    result["slurm_job_id"] = _banner_value(banner, "SLURM job")
    result["nodelist"] = _banner_value(banner, "SLURM nodes")

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
    ("segfault", re.compile(r"Signal 11 \(SIGSEGV\)|Segmentation fault|SIGSEGV")),
    (
        "nccl_error",
        re.compile(
            r"ncclRemoteError|ncclInternalError|ncclSystemError|ncclUnhandledCudaError|"
            r"NCCL error|RCCL error|Watchdog caught collective operation timeout"
        ),
    ),
)
# First Python exception that is not torchrun's generic wrapper.
PY_EXCEPTION_RE = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception)):\s")
GENERIC_EXCEPTIONS = ("ChildFailedError",)
# PyTorch's OOM message says how much of the GPU PyTorch itself held. When that
# is a small part of the capacity, something else was occupying the GPU.
OOM_HOLDINGS_RE = re.compile(
    r"total capacity of ([\d.]+) GiB.*?([\d.]+) (GiB|MiB) is allocated by PyTorch", re.IGNORECASE
)
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


def _scan_failure_text(run_lines: list[str]) -> tuple[str | None, str | None]:
    first: dict[str, str] = {}
    for line in run_lines:
        clean = strip_ansi(line)
        for category, pattern in FAILURE_PATTERNS:
            if category not in first and pattern.search(clean):
                first[category] = clean
        if "python_error" not in first:
            m = PY_EXCEPTION_RE.search(clean)
            if m and not m.group(1).endswith(GENERIC_EXCEPTIONS):
                first["python_error"] = clean[m.start() :]
    for category in [c for c, _ in FAILURE_PATTERNS] + ["python_error"]:
        if category not in first:
            continue
        line = first[category]
        if category == "oom":
            m = OOM_HOLDINGS_RE.search(line)
            if m:
                total = float(m.group(1))
                held = float(m.group(2)) / (1024.0 if m.group(3) == "MiB" else 1.0)
                if total > 0 and held < 0.5 * total:
                    category = "gpu_contended"
        return category, _clean_detail(line)
    return None, None


def _last_output_line(run_lines: list[str]) -> str | None:
    for line in reversed(run_lines):
        text = _clean_detail(line)
        if text and not text.startswith(">>>"):
            return text
    return None


def classify_failure(
    run_lines: list[str], footer_reason: str | None, exit_code: int | None
) -> tuple[str | None, str | None]:
    """Return ``(failure_reason, failure_detail)`` for a run that did not pass.

    The runner's own verdict wins when it is definitive (it stopped a hang,
    or the batch was interrupted). Otherwise the cause in the output is more
    useful than the bare exit status, which comes last.
    """
    footer_cat, _, footer_detail = (footer_reason or "").partition(":")
    footer_cat = footer_cat.strip()
    footer_detail = footer_detail.strip() or None
    if not footer_cat and exit_code is not None:
        # Logs written before the footer recorded a reason.
        if exit_code > 128:
            footer_cat = "signal"
            footer_detail = "SIGKILL" if exit_code == 137 else f"exit {exit_code}"
        elif exit_code != 0:
            footer_cat = "exit"

    if footer_cat in ("timeout", "idle", "interrupted"):
        return footer_cat, footer_detail
    text_cat, text_detail = _scan_failure_text(run_lines)
    if text_cat:
        return text_cat, text_detail
    if footer_cat == "signal":
        return ("killed" if "SIGKILL" in (footer_detail or "") else "signal"), footer_detail
    if footer_cat in ("fail_pattern", "exit"):
        return footer_cat, _last_output_line(run_lines) or footer_detail
    return None, None


def version_warnings(run_lines: list[str], limit: int = 10) -> list[str]:
    """Distinct ``... version mismatch ...`` messages (e.g. aiter installed vs
    expected), which affect performance but never fail a run."""
    seen: list[str] = []
    for line in run_lines:
        if "mismatch" not in line:
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
    mean = statistics.fmean(values) if hasattr(statistics, "fmean") else statistics.mean(values)
    return {"min": min(values), "max": max(values), "stdev": stdev, "cv": stdev / mean if mean else None}


FAILED_ATTEMPT_RE = re.compile(r"-failed-attempt(\d+)_")


def parse_log_file(filepath: str) -> dict | None:
    """Parse a single log file and return extracted metadata + statistics."""
    with open(filepath, "r", errors="replace") as f:
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
    footer_reason = footer.get("failure_reason")
    if exit_code is None:
        result["status"] = "incomplete" if banner else "unknown"
    elif footer_reason and footer_reason.startswith("interrupted"):
        result["status"] = "interrupted"
    elif exit_code == 0 and footer_reason in (None, "none") and result["failed_attempt"] is None:
        result["status"] = "ok"
    else:
        result["status"] = "failed"
    if result["status"] != "ok":
        result["failure_reason"], result["failure_detail"] = classify_failure(
            run_lines, footer_reason, exit_code
        )
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
]


CSV_NAME = "results.csv"
JSON_NAME = "results.json"


def tools_commit() -> str:
    """Commit of the checkout this parser runs from, for the JSON metadata."""
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_outputs(results: list[dict], input_dir: str) -> tuple[str, str]:
    """Write results.csv and results.json into ``input_dir``; return their paths."""
    csv_path = os.path.join(input_dir, CSV_NAME)
    json_path = os.path.join(input_dir, JSON_NAME)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {}
            for field in CSV_FIELDS:
                val = r.get(field)
                if isinstance(val, float):
                    row[field] = f"{val:.4f}"
                elif val is None:
                    row[field] = ""
                elif isinstance(val, list):
                    row[field] = " | ".join(val)
                else:
                    row[field] = val
            writer.writerow(row)

    with open(json_path, "w") as f:
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
    return csv_path, json_path


def _describe(result: dict) -> str:
    text = (
        f"{result['backend']} | {result['model_name']} | precision={result['precision']} | "
        f"repeat={result['repeat']} | status={result['status']}"
    )
    if result["failure_reason"]:
        text += f" ({result['failure_reason']})"
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
    if not log_files:
        print(f"No .txt or .log files found in {input_dir}")
        return 1

    results = []
    for fpath in log_files:
        fname = os.path.basename(fpath)
        try:
            result = parse_log_file(fpath)
        except Exception as e:
            print(f"Processing: {fname}\n  -> Error: {e}")
            continue
        if result:
            results.append(result)
            if not args.quiet:
                print(f"Processing: {fname}\n  -> {_describe(result)}")
        elif not args.quiet:
            print(f"Processing: {fname}\n  -> Skipped (not a Primus run log)")

    if not results:
        print("No results extracted.")
        return 1

    csv_path, json_path = write_outputs(results, input_dir)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    totals = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    print(f"Extracted {len(results)} runs ({totals}) from {len(log_files)} files.")
    print(f"Results written to: {csv_path}")
    print(f"                    {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

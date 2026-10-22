###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Why a Primus benchmark run failed, read from its log.

The failure analysis of extract_results.py (and through it report.py): the
``failure_reason`` and ``failure_detail`` of a run that did not pass, whether
it got past setup (``failed_at``), and ``no_metrics`` for a training run that
exited 0 without logging a step. See extract_results.py's docstring and the
README's "Extracting results" for what each reason means.

Callers pass the run output (the lines between ``Begin Run Output`` and
``End Run Output`` of a log) or a diffusion_bench step's own log.
"""

# Keeps the annotations below unevaluated, so the batch runners can call this
# on hosts whose python3 predates 3.10.
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import signal
import socket
import sys
from datetime import datetime, timedelta, timezone

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


# Searched in this order; the first category found anywhere in the run output
# wins. A bad NIC or an OOM usually also produces NCCL errors and tracebacks
# further down, so the specific causes come before the generic ones. The
# detail of nccl_error and python_error is the exception that stopped the run
# (see _fatal_exception) when there is one.
FAILURE_PATTERNS = (
    ("nic_error", re.compile(r"no socket interface found", re.IGNORECASE)),
    ("oom", re.compile(r"out of memory|OutOfMemoryError|RESOURCE_EXHAUSTED", re.IGNORECASE)),
    # Not a bare "SIGSEGV": Primus logs "... to avoid SIGSEGV ..." when it
    # applies its dataloader patch.
    (
        "segfault",
        re.compile(
            r"Signal 11 \(SIGSEGV\)|Segmentation fault|signal SIGSEGV|SIGSEGV received|\*\*\* SIGSEGV"
        ),
    ),
    (
        "nccl_error",
        re.compile(
            r"ncclRemoteError|ncclInternalError|ncclSystemError|ncclUnhandledCudaError|"
            r"NCCL error|RCCL error|Watchdog caught collective operation timeout"
        ),
    ),
)
NCCL_ERROR_RE = dict(FAILURE_PATTERNS)["nccl_error"]
# Cheap first pass, so a multi-million-line log only pays for the regexes
# above on the few lines that can match.
# Applied to the lower-cased line: re.IGNORECASE over these alternatives is
# several times slower than lower() plus a case-sensitive search.
FAILURE_PREFILTER_RE = re.compile(
    r"socket interface|out of memory|outofmemory|resource_exhausted|sigsegv|"
    r"segmentation fault|nccl|rccl|watchdog caught|traceback|error|exception"
)
# Primus's own patch log lines quote error and version strings they guard
# against, and a patch skipped for a missing dependency logs its ImportError's
# traceback; they are never the failure.
PRIMUS_NOISE = "[Patch"
# Primus's log prefix: "[20261021 05:06:54][rank-0/8][ERROR]    [---file.py:12] : "
# on the console ("(PrimusMaster  pid=N) " and node-<n> on the master), and
# "[20261021 05:06:54][root/amd][pre_trainer][ip-...][rank-0/8][ERROR] ..." in
# its per-rank log files. A multi-line message continues on unprefixed lines.
PRIMUS_RECORD_RE = re.compile(
    r"^(?:\(PrimusMaster[^)]*\)\s*)?\[\d{8} \d{2}:\d{2}:\d{2}\](?:\[[^\]]*\])*?"
    r"\[(?:rank|node)-(\d+)/\d+\]\[([A-Z]+)\]\s*(?:\[-*[^\]\s]+:\d+\] : )?"
)
ANSI_RESET = "\x1b[0m"
PATCH_SKIPPED = "failed due to missing dependency"
# PyTorch prefixes each line of an uncaught exception with "[rank3]: ".
TORCH_RANK_PREFIX_RE = re.compile(r"^\[rank\d+\]: ?")
# Lines another process can put in the middle of a traceback: torchrun's glog
# lines and the launcher's own log lines.
FOREIGN_LINE_RE = re.compile(
    r"^(?:\[?[IWEF]\d{4} \d{2}:\d{2}:\d{2}|\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[)"
)
TRACEBACK_MARKER = "Traceback (most recent call last):"
MAX_TRACEBACK_LINES = 2000
CHAIN_LINKS = {
    "The above exception was the direct cause of the following exception:": "cause",
    "During handling of the above exception, another exception occurred:": "context",
}
# loguru marks the frame that caught the exception with ">".
FRAME_RE = re.compile(r'^\s*>?\s*File "([^"]+)", line (\d+), in ')
CARET_RE = re.compile(r"^\s*[\^~]+\s*$")
# The last line of a traceback, e.g. "ValueError: bad" or a bare "AssertionError".
EXCEPTION_RE = re.compile(r"^((?:[A-Za-z_]\w*\.)*[A-Z]\w*)(?::(.*))?$")
# An exception line once its log prefix is stripped, for tracebacks too
# mangled to parse.
PY_EXCEPTION_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception))(?::|$)")
PY_EXCEPTION_RE = re.compile(r"\b([A-Za-z_][\w.]*(?:Error|Exception)):\s")
# torchrun's summary, and Primus's wrapper around the training error.
GENERIC_EXCEPTIONS = ("ChildFailedError",)
TRAINING_WRAPPER = "RuntimeError: Training execution failed"
# A chain ending in one of these was stopped from outside (a signal, an exit
# call); the runner's verdict says more than the traceback.
NOT_A_FAILURE = ("KeyboardInterrupt", "SystemExit", "GeneratorExit")
PRIMUS_ERROR = "[Primus] Error:"
# PyTorch's OOM message says how much of the GPU this process held. When that
# is a small part of the capacity, something else was occupying the GPU.
_SIZE = r"([\d.]+) (bytes|B|KiB|MiB|GiB)"
OOM_TOTAL_RE = re.compile(r"total capacity of " + _SIZE, re.IGNORECASE)
OOM_PROCESS_RE = re.compile(r"this process has " + _SIZE + r" memory in use", re.IGNORECASE)
OOM_ALLOCATED_RE = re.compile(_SIZE + r" is allocated by PyTorch", re.IGNORECASE)
OOM_RESERVED_RE = re.compile(_SIZE + r" is reserved by PyTorch but unallocated", re.IGNORECASE)
_GIB_PER_UNIT = {"bytes": 1.0 / 2**30, "b": 1.0 / 2**30, "kib": 1.0 / 2**20, "mib": 1.0 / 2**10, "gib": 1.0}
# Leading "[timestamp][rank-N/M][LEVEL] [file:line] :" decorations.
LOG_PREFIX_RE = re.compile(r"^(?:\s*\[[^\]]*\])+\s*(?::\s*)?")
MAX_DETAIL_CHARS = 300


def clean_detail(line: str) -> str:
    text = LOG_PREFIX_RE.sub("", strip_ansi(line).strip()).strip()
    return text[:MAX_DETAIL_CHARS]


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


class _OutputTexts:
    """The run output as failure analysis reads it: ``texts[i]`` is line ``i``
    without colour codes or a log prefix, None for the lines of a [Patch]
    message (with the traceback a patch logs when it is skipped for a missing
    dependency). Made on demand, since a log can have millions of lines; ``raw``
    is there for cheap substring checks."""

    def __init__(self, run_lines: list[str]):
        self.raw = run_lines
        self.noise: set[int] = set()
        for i, line in enumerate(run_lines):
            if PRIMUS_NOISE in line and i not in self.noise:
                self.noise.update(range(i, self._patch_message_end(i) + 1))

    def __len__(self) -> int:
        return len(self.raw)

    def _clean(self, i: int) -> str:
        line = self.raw[i].rstrip("\r\n")
        return strip_ansi(line) if "\x1b" in line else line

    def _patch_message_end(self, start: int) -> int:
        """Last line of the [Patch] message on line ``start``. Only a message
        that goes on to the traceback of a patch skipped for a missing
        dependency has more: with colours, up to the line with its colour
        reset; without, through that traceback (chained ones included)."""
        n = len(self.raw)
        end = min(n, start + MAX_TRACEBACK_LINES)
        line = self.raw[start].rstrip("\r\n")
        if "\x1b" in line:
            if line.endswith(ANSI_RESET):
                return start
            for j in range(start + 1, end):
                if self.raw[j].rstrip("\r\n").endswith(ANSI_RESET):
                    return j
            return start
        if PATCH_SKIPPED not in line:
            return start
        last, j = start, start + 1
        while j < end and self._clean(j).strip() == TRACEBACK_MARKER:
            j += 1
            while j < end and (not self._clean(j).strip() or self._clean(j)[0].isspace()):
                j += 1
            last = j  # the exception line
            j += 1
            while j < end and (not self._clean(j).strip() or self._clean(j).strip() in CHAIN_LINKS):
                j += 1
        return min(last, n - 1)

    def __getitem__(self, i: int) -> str | None:
        if i in self.noise:
            return None
        clean = self._clean(i)
        if clean.startswith(("[", "(")):
            m = PRIMUS_RECORD_RE.match(clean) or TORCH_RANK_PREFIX_RE.match(clean)
            if m:
                return clean[m.end() :]
        return clean

    def tag(self, i: int) -> tuple[str, str] | None:
        """``(rank, level)`` of a Primus record line, None for any other line."""
        clean = self._clean(i)
        m = PRIMUS_RECORD_RE.match(clean) if clean.startswith(("[", "(")) else None
        return (m.group(1), m.group(2)) if m else None


def _read_traceback(texts: _OutputTexts, start: int) -> tuple[dict | None, int]:
    """The traceback whose header is on line ``start``, and the index after it.
    The exception is None when the block does not end in an exception line.
    Indentation counts from the header's, as torchrun's summary indents the
    traceback it got from a rank's error file. Records of another rank or log
    level, and torchrun's and the launcher's own lines, are skipped."""
    frames: list[dict] = []
    header = texts[start] or ""
    base = len(header) - len(header.lstrip())
    owner = texts.tag(start)
    i = start + 1
    end = min(len(texts), start + MAX_TRACEBACK_LINES)
    while i < end:
        text = texts[i]
        i += 1
        if text is None or not text.strip() or CARET_RE.match(text) or FOREIGN_LINE_RE.match(text):
            continue
        tag = texts.tag(i - 1)
        if tag is not None and tag != owner:
            continue
        frame = FRAME_RE.match(text)
        if frame:
            frames.append({"file": frame.group(1), "line": int(frame.group(2)), "source": None})
        elif len(text) - len(text.lstrip()) > base:
            if frames and frames[-1]["source"] is None:
                frames[-1]["source"] = text.strip()
        else:
            m = EXCEPTION_RE.match(text.strip())
            if not m:
                return None, i - 1
            exception = {
                "type": m.group(1),
                "message": (m.group(2) or "").strip(),
                "text": text.strip(),
                "frames": frames,
                "index": i - 1,
            }
            return exception, i
    return None, i


def _traceback_chains(texts: _OutputTexts) -> list[list[dict]]:
    """Every traceback in the output, chained as Python prints them: each chain
    lists its exceptions cause first, and an exception's ``link`` ("cause" or
    "context") says how it relates to the one before it."""
    chains: list[list[dict]] = []
    linkable = False  # an exception was read, so a link line can follow it
    link = None
    i = 0
    while i < len(texts):
        if TRACEBACK_MARKER in texts.raw[i] and (texts[i] or "").strip().endswith(TRACEBACK_MARKER):
            exception, i = _read_traceback(texts, i)
            if exception is None:
                linkable, link = False, None
                continue
            exception["link"] = link
            if link and chains:
                chains[-1].append(exception)
            else:
                chains.append([exception])
            linkable, link = True, None
            continue
        # The rest of a multi-line exception message may come before the link
        # line; after it, only blank lines before the next traceback.
        if linkable:
            if link is None:
                if "exception" in texts.raw[i]:
                    link = CHAIN_LINKS.get((texts[i] or "").strip())
            elif (texts[i] or "").strip():
                linkable, link = False, None
        i += 1
    return chains


def _is_wrapper(exception: dict) -> bool:
    return exception["type"].endswith(GENERIC_EXCEPTIONS) or exception["text"].startswith(TRAINING_WRAPPER)


def _root_exception(chain: list[dict]) -> dict | None:
    """The exception a chain started from: back along "raise ... from" links and
    past wrappers. None for a chain of torchrun's summary alone."""
    i = len(chain) - 1
    while i > 0 and (chain[i]["link"] == "cause" or _is_wrapper(chain[i])):
        i -= 1
    return None if chain[i]["type"].endswith(GENERIC_EXCEPTIONS) else chain[i]


def _fatal_exception(texts: _OutputTexts) -> tuple[dict | None, bool]:
    """The exception that stopped the run: the root of the last chain printed
    before Primus's first "[Primus] Error:" line, else of the last chain. An
    earlier traceback is usually one that was caught and logged. Also whether
    the run was stopped from outside: that chain ends in an interrupt or exit."""
    chains = _traceback_chains(texts)
    anchor = next(
        (
            i
            for i, line in enumerate(texts.raw)
            if PRIMUS_ERROR in line and (texts[i] or "").lstrip().startswith(PRIMUS_ERROR)
        ),
        None,
    )
    if anchor is not None and any(chain[-1]["index"] < anchor for chain in chains):
        chains = [chain for chain in chains if chain[-1]["index"] < anchor]
    for chain in reversed(chains):
        if chain[-1]["type"].endswith(NOT_A_FAILURE):
            return None, True
        root = _root_exception(chain)
        if root is not None:
            return root, False
    return None, False


def _exception_detail(exception: dict) -> str:
    """The exception line. A bare one gets the location of the frame that
    raised it, and a bare AssertionError the failed assert as well."""
    text = exception["text"]
    if not exception["message"] and exception["frames"]:
        frame = exception["frames"][-1]
        where = f"{os.path.basename(frame['file'])}:{frame['line']}"
        if frame["source"] and exception["type"].endswith("AssertionError"):
            text = f"{exception['type']}: {frame['source']} ({where})"
        else:
            text = f"{exception['type']} ({where})"
    return text[:MAX_DETAIL_CHARS]


def scan_failure_text(run_lines: list[str]) -> tuple[str | None, str | None]:
    texts = _OutputTexts(run_lines)
    first: dict[str, str] = {}
    oom_lines: list[str] = []
    traceback_seen = False
    tail_exception = None  # last exception line after a traceback header
    loose_exception = None  # first "XError: ..." anywhere
    for i, line in enumerate(run_lines):
        if not FAILURE_PREFILTER_RE.search(line.lower()):
            continue
        text = texts[i]
        if text is None:
            continue
        for category, pattern in FAILURE_PATTERNS:
            if pattern.search(text):
                first.setdefault(category, text)
                if category == "oom":
                    oom_lines.append(text)
        if TRACEBACK_MARKER in text:
            traceback_seen = True
        elif traceback_seen:
            m = PY_EXCEPTION_LINE_RE.match(text.strip())
            if m and not m.group(1).endswith(GENERIC_EXCEPTIONS):
                tail_exception = text
        if loose_exception is None:
            m = PY_EXCEPTION_RE.search(text)
            if m and not m.group(1).endswith(GENERIC_EXCEPTIONS):
                loose_exception = text[m.start() :]
    for category in ("nic_error", "oom", "segfault"):
        if category not in first:
            continue
        if category == "oom" and _oom_is_contended(oom_lines):
            return "gpu_contended", clean_detail(next(l for l in oom_lines if OOM_TOTAL_RE.search(l)))
        return category, clean_detail(first[category])
    fatal, stopped = _fatal_exception(texts)
    if "nccl_error" in first:
        if fatal is not None and NCCL_ERROR_RE.search(fatal["text"]):
            return "nccl_error", _exception_detail(fatal)
        return "nccl_error", clean_detail(first["nccl_error"])
    if fatal is not None:
        return "python_error", _exception_detail(fatal)
    # Stopped by a signal or an exit call: an exception line elsewhere is not why.
    for exception in () if stopped else (tail_exception, loose_exception):
        if exception:
            return "python_error", clean_detail(exception)
    return None, None


def fatal_exception_detail(lines: list[str]) -> str | None:
    """The detail scan_failure_text gives the exception that stopped the run,
    when ``lines`` hold one in a traceback it could read."""
    fatal, _ = _fatal_exception(_OutputTexts(lines))
    return _exception_detail(fatal) if fatal is not None else None


def last_output_line(run_lines: list[str]) -> str | None:
    for line in reversed(run_lines):
        text = clean_detail(line)
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
    run_lines: list[str],
    footer_reason: str | None,
    exit_code: int | None,
    rank_lines: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """Return ``(failure_reason, failure_detail)`` for a run that did not pass.

    The runner's own verdict wins when it is definitive (it stopped a hang,
    or the batch was interrupted). Otherwise the cause in the failing rank's
    own log (``rank_lines``, see failing_rank_output), then in the output, is
    more useful than the bare exit status, which comes last. A failed run
    always gets a reason, ``unknown`` at worst.
    """
    footer_cat, _, footer_detail = (footer_reason or "").partition(":")
    footer_cat = footer_cat.strip()
    footer_detail = footer_detail.strip() or None

    # "steps": the diffusion_dlrm run listing the diffusion_bench steps that failed.
    if footer_cat in ("timeout", "idle", "interrupted", "steps"):
        return footer_cat, footer_detail
    rank = scan_failure_text(rank_lines) if rank_lines else (None, None)
    output = scan_failure_text(run_lines)
    # Only a specific cause in the output outranks the failing rank's own: the
    # NCCL errors of the ranks it left behind are what the output usually has.
    if output[0] in SPECIFIC_REASONS and (
        rank[0] not in SPECIFIC_REASONS or SPECIFIC_REASONS.index(output[0]) < SPECIFIC_REASONS.index(rank[0])
    ):
        return output
    for text_cat, text_detail in (rank, output):
        if text_cat:
            return text_cat, text_detail
    if footer_cat == "signal":
        return ("killed" if "SIGKILL" in (footer_detail or "") else "signal"), footer_detail
    if footer_cat in ("fail_pattern", "exit"):
        return footer_cat, last_output_line(run_lines) or footer_detail
    return "unknown", last_output_line(run_lines)


def settle_outcome(result: dict, iterations: list[dict]) -> None:
    """What only the parsed steps of a training run tell: a run that exited 0
    without logging a step failed (no_metrics), and how far a run that did not
    pass got. A no_metrics run finished, so it has no failed_at."""
    if result["status"] == "ok" and not iterations:
        result["status"] = "failed"
        result["failure_reason"] = "no_metrics"
        result["failure_detail"] = "the run exited 0 but logged no training step"
    elif result["status"] in ("failed", "interrupted", "incomplete"):
        result["failed_at"] = (
            f"after step {max(it.get('step', 0) for it in iterations)}" if iterations else "setup"
        )


# --------------------------------------------------------------------------
# The failing rank's own log
#
# For the PyTorch backends the console shows local rank 0 only (and the last
# local rank of the last node), so when another rank fails first, its error is
# not in the run log. torchrun's summary names that rank ("Root Cause (first
# observed failure)"), and Primus writes every rank's ERROR records to
# <workspace>/<team>/<user>/<exp>/logs/<module>/rank-<N>/error.log. After a
# failed run, the runner copies that rank's records from this run into the
# log, between "End Run Output" and the footer.
# --------------------------------------------------------------------------

ROOT_CAUSE_MARKER = "Root Cause (first observed failure):"
SUMMARY_FIELD_RE = re.compile(r"^\s*(time|host|rank|exitcode)\s*:\s*(.*?)\s*$")
SUMMARY_RANK_RE = re.compile(r"^(\d+)(?:\s*\(local_rank:\s*(\d+)\))?")
PRIMUS_TIMESTAMP_RE = re.compile(r"^\[(\d{8} \d{2}:\d{2}:\d{2})\]")
PRIMUS_TIMESTAMP_FORMAT = "%Y%m%d %H:%M:%S"
PLACEHOLDER_RE = re.compile(r"^\$\{([^:}]+):([^}]*)\}$")
# Of the runner's own environment, these reach the container without EXTRA_ENV.
FORWARDED_PREFIXES = ("PRIMUS_",)
FAILING_RANK_BEGIN = "Begin Failing Rank Output"
FAILING_RANK_END = "End Failing Rank Output"
FAILING_RANK_FIELD_RE = re.compile(
    r"^# Failing rank\s*:\s*(\d+)(?: \(local rank (\d+)\))?(?: on ([^\s,]+))?(?:, exit code (-?\d+))?"
)
MAX_SECTION_LINES = 200
MAX_TAIL_BYTES = 4 << 20
# Before the run's start, for other nodes' clocks; a run of the same
# experiment just before this one is further back than that.
CLOCK_SLACK = timedelta(seconds=5)
OWN_RECORDS = "its ERROR records from this run"
# Most specific first. Of the run output's reasons, only these outrank the
# failing rank's own.
SPECIFIC_REASONS = ("nic_error", "oom", "gpu_contended", "segfault")


def failing_rank(run_lines: list[str]) -> dict | None:
    """The rank torchrun's summary names as the first to fail, as
    ``{"rank", "local_rank", "host", "exitcode", "time"}``: the earliest when
    several nodes print one. None when there is no summary."""
    found = []
    for i, line in enumerate(run_lines):
        if ROOT_CAUSE_MARKER not in line:
            continue
        fields: dict[str, str] = {}
        for follow in run_lines[i + 1 : i + 12]:
            clean = strip_ansi(follow)
            if clean.startswith("====="):
                break
            m = SUMMARY_FIELD_RE.match(clean)
            if m:
                fields.setdefault(m.group(1), m.group(2))
        rank = SUMMARY_RANK_RE.match(fields.get("rank", ""))
        if not rank:
            continue
        code = re.match(r"-?\d+", fields.get("exitcode", ""))
        found.append(
            {
                "rank": int(rank.group(1)),
                "local_rank": int(rank.group(2)) if rank.group(2) else None,
                "host": fields.get("host") or None,
                "exitcode": int(code.group(0)) if code else None,
                "time": fields.get("time", ""),
            }
        )
    if not found:
        return None
    # torchrun stops the other ranks with a signal (a negative exit code).
    return min(found, key=lambda f: (f["time"], (f["exitcode"] or 0) < 0))


def _between(lines: list[str], begin: str, end: str) -> list[str]:
    start = next((i for i, line in enumerate(lines) if begin in line), None)
    if start is None:
        return []
    stop = next((i for i in range(start + 1, len(lines)) if end in lines[i]), len(lines))
    return lines[start + 1 : stop]


def _resolve(value: str, env: dict[str, str]) -> str:
    """A config value with its ``${VAR:default}`` placeholder resolved."""
    value = value.strip().strip("'\"")
    m = PLACEHOLDER_RE.match(value)
    return (env.get(m.group(1)) or m.group(2)) if m else value


def failing_rank_spec(log_path: str, primus_path: str, since: float = 0) -> dict | None:
    """Where to look for the failing rank's log of the run in ``log_path``,
    with the rank as failing_rank gives it; None when torchrun named none.
    ``since`` is the run's start (epoch seconds, the runner's clock)."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    run_lines = _between(lines, "Begin Run Output", "End Run Output") or lines
    who = failing_rank(run_lines)
    if who is None:
        return None
    banner: dict[str, str] = {}
    for line in lines:
        if "Begin Config File Dump" in line:
            break
        m = re.match(r"^#\s*([^:]+?)\s*:\s*(.*?)\s*$", line)
        if m:
            banner.setdefault(m.group(1), m.group(2))
    config = "".join(_between(lines, "Begin Config File Dump", "End Config File Dump"))

    # The container's view: EXTRA_ENV, then the runner's forwarded variables.
    env = {k: v for k, v in os.environ.items() if k.startswith(FORWARDED_PREFIXES)}
    extra = banner.get("Extra env (--env)", "")
    if extra and extra != "<none>":
        for token in shlex.split(extra):
            key, sep, value = token.partition("=")
            if sep:
                env[key] = value

    def top_level(key: str, default: str) -> str:
        m = re.search(rf"^{key}:\s*(.+?)\s*$", config, re.MULTILINE)
        return _resolve(m.group(1), env) if m else default

    workspace = top_level("workspace", "./output")
    if not os.path.isabs(workspace):
        workspace = os.path.normpath(os.path.join(primus_path, workspace))
    first = next((m.group(1) for m in map(PRIMUS_TIMESTAMP_RE.match, map(strip_ansi, run_lines)) if m), None)
    return {
        **who,
        "workspace": workspace,
        "exp_name": top_level("exp_name", ""),
        "module": (banner.get("Train Module") or "").split(" ")[0],
        "since": since,
        "window_start": first,
    }


def _find_rank_dir(spec: dict) -> str | None:
    """The directory of the rank's logs for this run's experiment, written to
    during the run; any experiment's only when the run's is not known."""
    rank = f"rank-{spec['rank']}"
    exp = glob.escape(spec["exp_name"]) if spec.get("exp_name") else "*"
    module = glob.escape(spec["module"]) if spec.get("module") else "*"
    since = float(spec.get("since") or 0) - CLOCK_SLACK.total_seconds()

    def written(path: str) -> float:
        files = [os.path.join(path, name) for name in ("error.log", "debug.log")]
        return max((os.path.getmtime(f) for f in files if os.path.isfile(f)), default=0)

    for pattern in ((exp, module), (exp, "*")):
        dirs = [
            d
            for d in glob.glob(
                os.path.join(spec["workspace"], "*", "*", pattern[0], "logs", pattern[1], rank)
            )
        ]
        dirs = [d for d in dirs if written(d) >= since]
        if dirs:
            return max(dirs, key=written)
    return None


def _run_cut(spec: dict) -> str | None:
    """The time of this run's first possible record, in the container's clock
    as Primus writes it: the run's start (epoch seconds) moved by the
    container's time zone. The output's first Primus record, shortly after the
    start, gives that zone to the quarter hour, rounded down (with two minutes
    of slack for another node's clock), so the cut stays before that record
    even when it comes late."""
    since = float(spec.get("since") or 0)
    first = spec.get("window_start")
    if since <= 0:
        if first is None:
            return None
        return (datetime.strptime(first, PRIMUS_TIMESTAMP_FORMAT) - CLOCK_SLACK).strftime(
            PRIMUS_TIMESTAMP_FORMAT
        )
    start = datetime.fromtimestamp(since, timezone.utc).replace(tzinfo=None)
    if first:
        quarter = 15 * 60
        seen = (datetime.strptime(first, PRIMUS_TIMESTAMP_FORMAT) - start).total_seconds()
        start += timedelta(seconds=((seen + 120) // quarter) * quarter)
    # With no Primus record to tell the zone from, containers run in UTC.
    return (start - CLOCK_SLACK).strftime(PRIMUS_TIMESTAMP_FORMAT)


def _records_from_run(path: str, cut: str | None) -> list[str]:
    """The records of ``path`` from ``cut`` on: Primus appends every run of an
    experiment to the same files. Read from the end, as they can be large."""
    if cut is None or not os.path.isfile(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - MAX_TAIL_BYTES))
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    kept, keep = [], False
    for line in lines:
        m = PRIMUS_TIMESTAMP_RE.match(line)
        if m:
            keep = m.group(1) >= cut
        if keep and line.strip():
            kept.append(line)
    return kept


def _short_host(host: str) -> str:
    return host.split(".")[0]


def failing_rank_section(spec: dict, this_host_only: bool = False) -> list[str] | None:
    """The lines that go into the run log: the failing rank, where its records
    come from, and the records. None when ``this_host_only`` and the rank ran
    on another node."""
    host = spec.get("host") or ""
    if this_host_only and _short_host(host) != _short_host(socket.gethostname()):
        return None
    who = f"{spec['rank']}"
    if spec.get("local_rank") is not None:
        who += f" (local rank {spec['local_rank']})"
    if host:
        who += f" on {host}"
    if spec.get("exitcode") is not None:
        who += f", exit code {spec['exitcode']}"
    body: list[str] = []
    rank_dir = _find_rank_dir(spec)
    cut = _run_cut(spec)
    if rank_dir is None:
        source = f"no rank-{spec['rank']} log under {spec['workspace']} written during this run"
    elif cut is None:
        source = f"{rank_dir}: no time to tell this run's records from earlier runs' by"
    else:
        error_log, debug_log = (os.path.join(rank_dir, name) for name in ("error.log", "debug.log"))
        body = _records_from_run(error_log, cut)
        source = f"{error_log} ({OWN_RECORDS})"
        if not body:
            body = _records_from_run(debug_log, cut)[-40:]
            source = f"{error_log} has no ERROR records from this run; the last lines of {debug_log}"
    if len(body) > MAX_SECTION_LINES:
        source += f", the last {MAX_SECTION_LINES} of {len(body)} lines"
        body = body[-MAX_SECTION_LINES:]
    return (
        [
            f" {FAILING_RANK_BEGIN} ".center(80, "#"),
            f"# Failing rank     : {who}",
            f"# Source           : {source}",
        ]
        + body
        + [f" {FAILING_RANK_END} ".center(80, "#")]
    )


def failing_rank_output(lines: list[str]) -> dict | None:
    """The failing-rank section of a log as ``{"rank", "local_rank", "host",
    "exitcode", "lines", "span"}``: ``lines`` are its ERROR records (none when
    it shows debug.log lines instead, which are not errors), ``span`` the first
    and last line index of the section. None when the log has none."""
    start = next((i for i, line in enumerate(lines) if FAILING_RANK_BEGIN in line), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if FAILING_RANK_END in lines[i]), len(lines) - 1)
    section = lines[start + 1 : end]
    m = next(
        (FAILING_RANK_FIELD_RE.match(strip_ansi(line)) for line in section[:2] if "Failing rank" in line),
        None,
    )
    if not m:
        return None
    source = next((line for line in section[:3] if line.startswith("# Source")), "")
    records = [line for line in section if not line.startswith(("# Failing rank", "# Source"))]
    return {
        "rank": int(m.group(1)),
        "local_rank": int(m.group(2)) if m.group(2) else None,
        "host": m.group(3),
        "exitcode": int(m.group(4)) if m.group(4) else None,
        "lines": records if OWN_RECORDS in source else [],
        "span": (start, end),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The failing rank's own log, for the batch runners.")
    sub = parser.add_subparsers(dest="command", required=True)
    spec_cmd = sub.add_parser("failing-rank", help="Print where the failing rank of a run logged, as JSON.")
    spec_cmd.add_argument("--log", required=True, help="The run's log.")
    spec_cmd.add_argument("--primus-path", required=True, help="The Primus checkout the run used.")
    spec_cmd.add_argument("--since", type=float, default=0, help="When the run started, in epoch seconds.")
    section_cmd = sub.add_parser("rank-log", help="Print the failing-rank section for a failing-rank spec.")
    section_cmd.add_argument("--spec", required=True, help="The JSON failing-rank printed.")
    section_cmd.add_argument(
        "--this-host-only", action="store_true", help="Print nothing unless the rank ran on this node."
    )
    args = parser.parse_args(argv)
    if args.command == "failing-rank":
        spec = failing_rank_spec(args.log, args.primus_path, args.since)
        if spec:
            print(json.dumps(spec))
    else:
        lines = failing_rank_section(json.loads(args.spec), args.this_host_only)
        if lines:
            print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

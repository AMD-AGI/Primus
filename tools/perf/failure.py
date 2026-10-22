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

import os
import re
import signal

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
    text_cat, text_detail = scan_failure_text(run_lines)
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

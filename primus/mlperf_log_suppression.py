###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# ---------------------------------------------------------------------------
# Non-MLLOG log suppression for MLPerf submission runs.
#
# The MLPerf reference/submission logs (":::MLLOG ...") plus the run banners
# emitted by the launch script are the only required stdout lines for a
# submission run. All other noisy framework output (Primus loguru banners,
# Megatron deprecations, aiter JIT build chatter, TE-RoPE, hipify, torchrun
# warnings, UserWarnings, ...) is suppressed when this module is enabled, to
# keep the run logs clean.
#
# Activation is GATED so that importing this module is a complete no-op for
# normal (non-MLPerf) Primus runs:
#
#   PRIMUS_LOG_SUPPRESSION=1  -> enable the suppression (MLPerf launch sets it)
#   PRIMUS_LOG_SUPPRESSION=0  -> default: install() is a no-op
#
# When enabled, verbosity is further controlled by:
#
#   MLPERF_VERBOSE_LOGS=1  -> restore the full verbose output (old behaviour),
#                            i.e. enabled-but-not-quiet.
#   MLPERF_VERBOSE_LOGS=0  -> quiet mode (default when enabled): only MLLOG +
#                            training timing/result lines are emitted.
#
# Strategy (strongly preferred over stdout scraping):
#   1. Export env vars that the noisy libraries honour (AITER_LOG_LEVEL,
#      PYTHONWARNINGS, TRANSFORMERS_VERBOSITY, HF_HUB_DISABLE_PROGRESS_BARS).
#   2. Raise Python ``logging`` levels for the actual logger names emitting
#      noise (``aiter``, ``megatron`` families, ``torch.distributed``, ...).
#      Safe to call multiple times via ``reapply_quiet_logger_levels``.
#   3. Silence ``warnings`` category to match ``PYTHONWARNINGS=ignore``.
#
# A handful of sources (unconditional C++ ``std::cout`` in aiter / hipify;
# Primus ``loguru`` sinks that bypass ``logging``; raw ``print()`` in TE
# ``rope.py``; ...) cannot be controlled via the above. For those a narrow
# FD-level line filter is installed on stdout so even native writes get
# matched. Stderr carries no useful signal in quiet mode so FD 2 is
# redirected wholesale to ``/dev/null``. If you need stderr back (e.g. to
# debug a crash) re-run with ``MLPERF_VERBOSE_LOGS=1``.
#
# IMPORTANT: This module must be imported BEFORE any other import that may
# log/print at import time (Megatron, TE, aiter, ...). For the Primus CLI it
# is imported at the very top of ``primus/cli/main.py``. Because it lives
# directly under the ``primus`` package, importing it only triggers the light
# ``primus/__init__.py`` (config/logging utilities) and pulls in none of the
# heavy native libraries that emit the banners we want to hide.
# ---------------------------------------------------------------------------
import logging as _logging
import os as _os

ENABLED = _os.environ.get("PRIMUS_LOG_SUPPRESSION", "0") == "1"
VERBOSE_LOGS = _os.environ.get("MLPERF_VERBOSE_LOGS", "0") == "1"

# Logger names that emit the non-MLLOG noise we want to silence. Safe to
# set even when a given logger is not present in the process.
QUIET_LOGGER_NAMES = (
    # aiter: Primus also raises this logger to ERROR inside the trainer,
    # but we do it earlier (before the aiter JIT build banner fires).
    "aiter",
    # Megatron-LM deprecation + rerun-state-machine warnings and the
    # "Setting RerunStateMachine mode" warning from rerun_state_machine.py.
    "megatron",
    "megatron.core",
    "megatron.core.utils",
    "megatron.core.rerun_state_machine",
    "megatron.core.pipeline_parallel",
    # Primus / Primus-Turbo stdlib loggers (Primus routes most output
    # through loguru instead, which we drop at the FD level below).
    "primus",
    "primus_turbo",
    "primus_mllog",
    # TransformerEngine import / RoPE banners that use stdlib logging.
    "transformer_engine",
    # torch.distributed elastic / launcher chatter.
    "torch.distributed",
    "torch.distributed.run",
    "torch.distributed.elastic",
    "torch.distributed.elastic.multiprocessing",
    "torch.distributed.launcher.api",
    # HuggingFace Hub / transformers progress + warnings.
    "transformers",
    "huggingface_hub",
    # General misc.
    "filelock",
    "urllib3",
)


def reapply_quiet_logger_levels() -> None:
    """Raise levels on noisy Python loggers. Safe to call multiple times.

    No-op unless suppression is enabled and quiet mode is active.
    """
    if not ENABLED or VERBOSE_LOGS:
        return
    for _logger_name in QUIET_LOGGER_NAMES:
        _logging.getLogger(_logger_name).setLevel(_logging.ERROR)


def _configure_non_mllog_logs_quiet() -> None:
    """Silence every non-MLLOG log source we can control via env vars or
    the standard ``logging`` / ``warnings`` modules. :::MLLOG output is
    untouched because ``mlperf_logging`` has its own stdout handler.
    """
    _os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")
    _os.environ.setdefault("AITER_LOG_MORE", "0")
    _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _os.environ.setdefault("PYTHONWARNINGS", "ignore")
    # Tell Primus-Turbo / GPT-OSS helpers to keep JIT tuning traces off.
    _os.environ.setdefault("AITER_LOG_TUNED_CONFIG", "0")

    for _logger_name in QUIET_LOGGER_NAMES:
        _logging.getLogger(_logger_name).setLevel(_logging.ERROR)

    # ``warnings.filterwarnings("ignore")`` covers cases where
    # ``PYTHONWARNINGS=ignore`` has no effect because Python was started
    # before we ran (e.g. re-exec'd by a launcher).
    try:
        import warnings as _warnings

        _warnings.filterwarnings("ignore")
    except Exception:
        pass


def _install_fd_level_fallback_filter() -> None:
    """Last-resort line filter for logs that bypass Python ``logging``
    entirely: unconditional ``std::cout`` writes in aiter / hipify C++,
    raw ``print()`` statements in TE / Megatron, and Primus loguru output
    that landed on stdout.

    Design:
      * FD 2 (stderr) is redirected straight to ``/dev/null``. In quiet
        mode the only stderr sources observed in practice are Primus's
        ``loguru`` sink, ``warnings.warn`` output, the hipify
        ``"Successfully preprocessed all matching files."`` banner, and
        torchrun's elastic SIGTERM chatter -- none of which carry a
        training or MLLOG signal.
      * FD 1 (stdout) is routed through an ``os.pipe`` + reader thread.
        The thread runs each line through the suppression regexes and
        forwards surviving lines to the saved original stdout FD.
      * ``:::MLLOG`` lines always pass through.
    """
    import re
    import sys
    import threading

    ansi_re = re.compile(r"\x1b\[[0-9;]*[ -/]*[@-~]")

    suppress_patterns = tuple(
        re.compile(p)
        for p in (
            # aiter JIT module-load / build / baton-wait banners.
            r"^\[aiter\] ",
            # TE-RoPE banner from transformer_engine rope.py (raw print).
            r"^\[TE-RoPE\] ",
            # "[MLPerf Train] ..." status prints -- all informational, all
            # emitted N times in distributed mode. Re-run verbose to see them.
            r"^\[MLPerf Train\] ",
            # Primus runtime/CLI/patch informational prints that bypass loguru
            # (e.g. "[Primus:Runtime] ...", "[Primus:Env] ...", "[Primus] ...",
            # "[Primus CLI] ...", "[PrimusPatch] ..."). Covers the new-arch
            # runtime/launcher print() calls. Re-run verbose to see them.
            r"^\[Primus",
            # Gloo C++ peer-connect banner ("[Gloo] Rank N is connected
            # to 7 peer ranks. Expected number of connected peer ranks
            # is : 7") -- native std::cout from libgloo.
            r"^\[Gloo\] Rank \d+ is connected to ",
            r"^Expected number of connected peer ranks is\s*:",
            # hipify preprocessing banner (mostly fires on stderr, listed
            # here as a safety net for any leak to stdout).
            r"^Successfully preprocessed all matching files\.",
            # Primus loguru lines, in case they end up on stdout for any
            # reason (e.g. ``colorize=False`` configurations). The Primus
            # format always starts with ``[YYYYMMDD HH:MM:SS]``.
            r"^\[\d{8} \d{2}:\d{2}:\d{2}\]\[(?:rank|node)-\d+/",
            # "Setting RerunStateMachine mode RerunMode.DISABLED" fires
            # from Megatron's rerun_state_machine.py as a root-logger
            # warning (the default Python logging handler dumps the message
            # unformatted to stderr, but belt-and-braces: cover stdout too).
            r"^Setting RerunStateMachine mode ",
        )
    )

    def _should_suppress(line: str) -> bool:
        stripped = ansi_re.sub("", line)
        if ":::MLLOG" in stripped:
            return False
        for pat in suppress_patterns:
            if pat.search(stripped):
                return True
        return False

    def _start_reader(read_fd: int, out_fd: int) -> None:
        def _run() -> None:
            buf = b""
            try:
                while True:
                    chunk = _os.read(read_fd, 4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", errors="replace")
                        if not _should_suppress(line):
                            _os.write(out_fd, raw + b"\n")
            except Exception:
                # Never let the filter thread bring down the training run.
                pass
            finally:
                if buf:
                    line = buf.decode("utf-8", errors="replace")
                    if not _should_suppress(line):
                        try:
                            _os.write(out_fd, buf)
                        except Exception:
                            pass

        threading.Thread(target=_run, daemon=True).start()

    # Flush any buffered Python stdout/stderr output before we steal the FDs.
    sys.stdout.flush()
    sys.stderr.flush()

    orig_stdout_fd = _os.dup(1)

    # Drop stderr unconditionally: nothing useful lands on FD 2 in quiet
    # mode (Primus loguru, warnings, hipify). Users who need stderr back
    # (e.g. crash debug) can re-run with ``MLPERF_VERBOSE_LOGS=1``.
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(devnull_fd, 2)
    _os.close(devnull_fd)

    stdout_r, stdout_w = _os.pipe()
    _os.dup2(stdout_w, 1)
    _os.close(stdout_w)

    _start_reader(stdout_r, orig_stdout_fd)

    sys.stdout = _os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stderr = _os.fdopen(2, "w", buffering=1, closefd=False)


_INSTALLED = False


def install() -> None:
    """Install the quiet-mode suppression exactly once per process.

    No-op unless ``PRIMUS_LOG_SUPPRESSION=1``. Calling this a second time is
    also a no-op.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    if not ENABLED:
        return
    if VERBOSE_LOGS:
        return
    _configure_non_mllog_logs_quiet()
    _install_fd_level_fallback_filter()


# Auto-install on import (gated by PRIMUS_LOG_SUPPRESSION).
install()

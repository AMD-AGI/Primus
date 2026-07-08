# ---------------------------------------------------------------------------
# Non-MLLOG log suppression for the Primus/Megatron training pipeline.
#
# Adapted from small_llm_pretraining/nemo/src/_log_suppression.py for use
# with the Primus stack (Megatron-LM + TransformerEngine + AITER).
#
# The MLPerf reference logs (":::MLLOG ...") plus the pretraining
# STARTING/ENDING/RESULT banners are the only lines we want on stdout.
# All other framework output (Megatron deprecations, TE warnings,
# AITER/hipblaslt/gloo C++ writes, torch.distributed noise, ...) is
# suppressed by default to keep CI and submission logs clean.
#
# Control:
#   MLPERF_VERBOSE_LOGS=1  -> restore the full verbose output
#   MLPERF_VERBOSE_LOGS=0  -> quiet mode (default): only MLLOG + training
#                             timing/result lines are emitted.
#
# Strategy:
#   1. Export env vars that noisy libraries honour (AITER_LOG_LEVEL,
#      PYTHONWARNINGS, TRANSFORMERS_VERBOSITY, ...).
#   2. Raise Python ``logging`` levels for the logger names emitting the
#      noise (megatron.*, transformer_engine.*, torch.distributed, ...).
#   3. Install an FD-level line filter on stdout so native C++ writes
#      (hipModuleLoad, hipblaslt latency, gloo peer-connect) are caught.
#      Stderr is redirected to /dev/null in quiet mode.
#
# This module must be imported BEFORE any other import that may log/print
# at import time (TE, Megatron, aiter, ...).
# ---------------------------------------------------------------------------
import os as _os
import logging as _logging

VERBOSE_LOGS = _os.environ.get("MLPERF_VERBOSE_LOGS", "0") == "1"

QUIET_LOGGER_NAMES = (
    "megatron",
    "megatron.core.utils",
    "megatron.core.rerun_state_machine",
    "transformer_engine",
    "transformer_engine.aiter_rope",
    "torch.distributed",
    "torch.distributed.c10d_logger",
)


def reapply_quiet_logger_levels() -> None:
    """Raise levels on noisy Python loggers.  Safe to call multiple times."""
    for name in QUIET_LOGGER_NAMES:
        _logging.getLogger(name).setLevel(_logging.ERROR)


def _configure_env_and_loggers() -> None:
    """Silence non-MLLOG sources controllable via env vars or logging."""
    _os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")
    _os.environ.setdefault("AITER_LOG_MORE", "0")
    _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _os.environ.setdefault("PYTHONWARNINGS", "ignore")

    reapply_quiet_logger_levels()


def _install_fd_level_fallback_filter() -> None:
    """Last-resort line filter for logs that bypass Python ``logging``.

    * FD 2 (stderr) -> /dev/null (only torchrun OMP_NUM_THREADS and hipify
      banners land there; no MLLOG signal).
    * FD 1 (stdout) -> pipe + reader thread that applies regex suppression.
      ``:::MLLOG`` lines always pass through unconditionally.
    """
    import re
    import sys
    import threading

    ansi_re = re.compile(r"\x1b\[[0-9;]*[ -/]*[@-~]")

    suppress_patterns = tuple(
        re.compile(p)
        for p in (
            # aiter C++ hipModuleLoad / hipModuleGetFunction banners
            r"\[aiter\] hipModule",
            r"\[aiter\] hipModuleGetFunction:",
            r"^\[aiter\] import \[",
            # hipblaslt latency warnings (ROCm)
            r"Warning: Latency not found for MI_M=",
            r"Returning latency value of 32 \(really slow\)",
            r"^(?:\s*(?:BFloat8Float8_fnuz|Float8_fnuz|,\s*(?:MI_[MNK]|mi_input_type)=\d*|\d+)\s*)+\.?\s*$",
            # Gloo C++ peer-connect banners
            r"\[Gloo\] Rank \d+ is connected to ",
            r"^Expected number of connected peer ranks is\s*:",
            # PyTorch AccumulateGrad stream mismatch warning
            r"AccumulateGrad node's stream does not match",
            r"^\s*Variable\._execution_engine\.run_backward\(",
            # torch.distributed c10d barrier warning
            r"barrier\(\): using the device under current context",
            # hipify preprocessed banner
            r"^(?:\x1b\[[0-9;]*m)?Successfully preprocessed all matching files\.",
            # [MLPerf Train] framework startup banners (rpdTracer, HF_HOME, config, etc.)
            r"^\[MLPerf Train\]",
            # Created path banners from multi-rank output directory creation
            r"^Created path:",
            # Orphan fragments from interleaved C++ writes
            r"^\s*Success\s*$",
            r"^\s*\d+\s*$",
            # torchrun OMP_NUM_THREADS banner (usually on stderr, but catch
            # it on stdout too in case it leaks through)
            r"Setting OMP_NUM_THREADS environment variable for each process",
            r"^\*{5,}$",
            r"^W\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\d+\s+torch/distributed/run\.py",
        )
    )

    def _should_suppress(line: str) -> bool:
        stripped = ansi_re.sub("", line)
        if ":::MLLOG" in stripped:
            return False
        if not stripped.strip():
            return True
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

    sys.stdout.flush()
    sys.stderr.flush()

    orig_stdout_fd = _os.dup(1)

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
    """Install quiet-mode suppression exactly once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    if VERBOSE_LOGS:
        return
    _configure_env_and_loggers()
    _install_fd_level_fallback_filter()


# Auto-install on import.
install()
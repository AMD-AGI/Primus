###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import math
import os
import re
import subprocess
import sys
import time
import unittest
from typing import Optional

from primus.core.utils import logger

TRAINING_COMPLETED_MARKER = "Training completed."

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Per-step metric lines, one pattern per backend log format:
#   torchtitan: "step:  1  loss: 12.78468  grad_norm:     nan  memory: ..."
#   megatron:   "iteration  3/  3 | ... | lm loss: 1.17E+01 | ... | grad norm: 5.885 | ..."
_STEP_METRIC_RES = (
    re.compile(r"\bstep:\s*(?P<step>\d+)\b.*?\bloss:\s*(?P<loss>\S+).*?\bgrad_norm:\s*(?P<grad>\S+)"),
    re.compile(
        r"\biteration\s*(?P<step>\d+)\s*/.*?\blm loss:\s*(?P<loss>\S+).*?\bgrad norm:\s*(?P<grad>\S+)"
    ),
)


def assert_finite_training_metrics(tag: str, log_text: str) -> int:
    """Fail when any logged per-step loss or grad norm is NaN/Inf.

    A training run that diverges numerically still exits 0 and still prints the
    "Training completed." marker, so without this check such a run passes as a
    green test. Returns the number of steps that were checked; 0 means the log
    had no recognizable metric lines, which is not treated as a failure so that
    backends with other log formats keep working.
    """
    plain = _ANSI_ESCAPE_RE.sub("", log_text)

    checked = 0
    bad: list[str] = []
    for line in plain.splitlines():
        if "loss:" not in line:  # the vast majority of log lines; skip the regexes
            continue
        for pattern in _STEP_METRIC_RES:
            match = pattern.search(line)
            if match is None:
                continue
            checked += 1
            for field in ("loss", "grad"):
                raw = match.group(field).rstrip("|,")
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not math.isfinite(value):
                    bad.append(f"step {match.group('step')}: {field}={raw}")
            break

    if bad:
        raise AssertionError(
            f"[{tag}] Training reported non-finite metrics, so the run diverged even "
            f"though the process exited 0: {', '.join(bad[:8])}"
        )

    return checked


def skip_if_no_cuda(reason: str = "requires GPU (primus_turbo initializes CUDA at import)") -> None:
    """Skip the calling test module at collection time when CUDA is unavailable.

    Several Flux/diffusion test modules import ``primus_turbo`` (directly or
    transitively), which initializes CUDA at import and raises on CPU-only
    hosts. Call this at module scope *before* those imports so collection
    succeeds without a GPU. It is a no-op when CUDA is present, so GPU CI still
    runs every test.
    """
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip(reason, allow_module_level=True)


def install_aiter_deepbind_hook() -> None:
    """Install Primus' production RTLD_DEEPBIND import hook for aiter's mha kernels.

    This invokes the exact mechanism the ``megatron.turbo.aiter_deepbind``
    before_train patch uses in real training: it wraps ``importlib.import_module``
    so aiter's pinned mha extensions bind their own ``aiter::mha_bwd`` instead of
    transformer_engine's stale vendored ``libmha`` (ROCm/aiter#1332). On
    gfx942/gfx950 that stale-symbol interposition otherwise makes Turbo's hd128
    backward launch with an invalid grid config and crash the process.

    Unit tests call ``flash_attn_func`` directly and never run the before_train
    phase, so without this hook they hit the same crash the production patch
    prevents. Call this at diffusion ``conftest`` import time so the hook is in
    place before any test first imports the aiter mha modules (which happens
    lazily on the first attention op). No-op when CUDA is unavailable or the hook
    cannot be installed (e.g. CPU-only host).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        from primus.backends.megatron.patches.turbo.aiter_deepbind_patches import (
            _install_deepbind_import_hook,
        )
    except Exception:
        return

    _install_deepbind_import_hook()


class PrimusUT(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        ut_log_path = os.environ.get("UT_LOG_PATH", "ut_out")
        logger_cfg = logger.LoggerConfig(
            exp_root_path=ut_log_path,
            work_group="develop",
            user_name="root",
            exp_name="unittest",
            module_name=f"UT-{cls.__name__}",
            file_sink_level="DEBUG",
            stderr_sink_level="INFO",
            node_ip="localhost",
            rank=os.environ.get("RANK", 0),
            world_size=os.environ.get("WORLD_SIZE", 1),
        )
        logger.setup_logger(logger_cfg, is_head=False)

    def setUp(self):
        pass

    def tearDown(self):
        pass


def run_training_script(
    tag: str,
    cmd: list[str],
    train_log_path: str,
    env: Optional[dict] = None,
    check_metrics: bool = True,
) -> tuple[str, str]:
    """Execute a training command and validate that training completed successfully.

    Runs the command via subprocess, streams output to console, then reads the
    training log file and asserts that the PrimusRuntime "Training completed."
    marker is present. This catches silent failures where the process exits 0
    but training did not actually finish.

    Args:
        tag: Human-readable label for log messages (e.g. "llama3_8B").
        cmd: Command to execute (passed to subprocess.run).
        train_log_path: Path to the training log file written by the launcher.
        env: Environment variables for the subprocess.
        check_metrics: Also require every logged loss and grad norm to be finite.

    Returns:
        (stdout_output, stderr_output) tuple where stdout_output is the
        content of train_log_path.

    Raises:
        AssertionError: If training did not complete successfully.
    """
    # Short-circuit Python / torchrun teardown on successful runs to save
    # ~20s per end-to-end test. The marker we rely on ("Training completed.")
    # is emitted well before cleanup(), so this does not affect assertions.
    # Developers can opt out locally via PRIMUS_EXIT_FAST=0.
    if env is not None:
        env.setdefault("PRIMUS_EXIT_FAST", "1")

    try:
        logger.info(f"[{tag}] Begin run...")
        start = time.time()
        subprocess.run(
            cmd,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            env=env,
        )
        logger.info(f"[{tag}] End run, time={time.time() - start:.3f} s")
        logger.info(f"[{tag}] Training log: {train_log_path}")

        stdout_output = ""
        if os.path.exists(train_log_path):
            with open(train_log_path, "r") as f:
                stdout_output = f.read()

        if TRAINING_COMPLETED_MARKER not in stdout_output:
            raise AssertionError(
                f"[{tag}] Process exited with code 0 but '{TRAINING_COMPLETED_MARKER}' "
                f"not found in log output. Training may have failed silently.\n"
                f"Log file: {train_log_path}"
            )

        if check_metrics:
            assert_finite_training_metrics(tag, stdout_output)

        return stdout_output, ""

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr or ""
        stdout_output = e.stdout or ""

        if os.path.exists(train_log_path):
            try:
                with open(train_log_path, "r") as f:
                    stdout_output = f.read()
            except Exception as log_err:
                logger.warning(f"[{tag}] Failed to read train log: {log_err}")

        if TRAINING_COMPLETED_MARKER in stdout_output:
            logger.warning(f"[{tag}] Training likely succeeded despite return code != 0.")
            logger.warning(f"stderr excerpt:\n{stderr_output[:1000]}")
        else:
            raise AssertionError(f"[{tag}] Shell script failed: {stderr_output.strip()}")

    return stdout_output, stderr_output

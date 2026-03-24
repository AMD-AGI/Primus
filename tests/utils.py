###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
import subprocess
import sys
import time
import unittest
from typing import Optional

from primus.core.utils import logger

TRAINING_COMPLETED_MARKER = "Training completed."


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

    Returns:
        (stdout_output, stderr_output) tuple where stdout_output is the
        content of train_log_path.

    Raises:
        AssertionError: If training did not complete successfully.
    """
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

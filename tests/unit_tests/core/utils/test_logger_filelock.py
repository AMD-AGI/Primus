###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import primus
from primus.core.utils.logger import quiet_filelock_logger


@pytest.fixture
def filelock_logger(monkeypatch):
    monkeypatch.delenv("PRIMUS_VERBOSE_FILELOCK", raising=False)
    logger = logging.getLogger("filelock")
    level = logger.level
    yield logger
    logger.setLevel(level)


def test_raises_unset_level_to_info(filelock_logger):
    filelock_logger.setLevel(logging.NOTSET)
    quiet_filelock_logger()
    assert filelock_logger.level == logging.INFO


def test_keeps_a_stricter_level(filelock_logger):
    """MLPerf quiet mode sets ERROR before the Primus logger is set up."""
    filelock_logger.setLevel(logging.ERROR)
    quiet_filelock_logger()
    assert filelock_logger.level == logging.ERROR


def test_verbose_env_keeps_debug(filelock_logger, monkeypatch):
    monkeypatch.setenv("PRIMUS_VERBOSE_FILELOCK", "1")
    filelock_logger.setLevel(logging.DEBUG)
    quiet_filelock_logger()
    assert filelock_logger.level == logging.DEBUG


def test_setup_logger_quiets_filelock(tmp_path):
    """setup_logger runs once per process, so check it in a fresh one."""
    code = (
        "import logging\n"
        "from primus.core.utils import logger\n"
        f"logger.setup_logger(logger.LoggerConfig({str(tmp_path)!r}, 'ut', 'ut', 'ut', 'ut'))\n"
        "print(logging.getLogger('filelock').level)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(primus.__file__).resolve().parents[1]))
    env.pop("PRIMUS_VERBOSE_FILELOCK", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split()[-1] == str(logging.INFO)

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import logging

import pytest

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

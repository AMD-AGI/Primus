###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pytest fixtures for the SpecForge backend tests."""

import logging

import pytest

from primus.core.utils import logger as primus_logger


@pytest.fixture(autouse=True)
def primus_logging():
    """Give Primus a real logger for the duration of each test.

    ``primus.core.utils.logger._logger`` is None until the worker logger is
    initialized by the runtime, so any ``log_rank_0`` call raises
    ``AttributeError`` in a bare unit-test process.
    """
    original = primus_logger._logger
    primus_logger._logger = logging.getLogger("primus.unit_tests.specforge")
    try:
        yield
    finally:
        primus_logger._logger = original

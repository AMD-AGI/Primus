# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Pytest fixtures for diffusion training tests.

Sets up the Primus logger (required by log_rank_0) and re-exports
the parallel state fixture from the parent conftest.
"""

import os

import pytest

from primus.core.utils import logger
from tests.unit_tests.backends.megatron.conftest import (  # noqa: F401
    init_parallel_state,
)


@pytest.fixture(autouse=True, scope="session")
def setup_logger():
    """Initialize Primus logger for tests that use log_rank_0."""
    logger_cfg = logger.LoggerConfig(
        exp_root_path=os.environ.get("UT_LOG_PATH", "ut_out"),
        work_group="develop",
        user_name="root",
        exp_name="unittest",
        module_name="UT-training",
        file_sink_level="DEBUG",
        stderr_sink_level="INFO",
        node_ip="localhost",
        rank=os.environ.get("RANK", 0),
        world_size=os.environ.get("WORLD_SIZE", 1),
    )
    logger.setup_logger(logger_cfg, is_head=False)

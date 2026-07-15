# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Shared pytest fixtures for Megatron backend tests.

Provides reusable fixtures for parallel state initialization used across
optimizer, diffusion, and other Megatron test suites.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from primus.core.utils import logger


@pytest.fixture(autouse=True, scope="session")
def setup_logger():
    """Initialize the Primus logger for megatron tests that use log_rank_0.

    The Primus ``_logger`` global is ``None`` until configured, so tests that
    log (e.g. test_warmup_convergence.py) raise ``AttributeError`` when run
    without a prior logger-init. This session-scoped autouse fixture removes
    that ordering dependency. Subdirectories may override it with an identically
    named fixture (see diffusion/training/conftest.py).
    """
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


def _is_mxfp4_supported():
    if not torch.cuda.is_available():
        return False
    try:
        from primus_turbo.pytorch.core.low_precision import check_mxfp4_support

        supported, _ = check_mxfp4_support()
        return supported
    except ImportError:
        return False


requires_mxfp4 = pytest.mark.skipif(
    not _is_mxfp4_supported(),
    reason="Requires gfx950+ (MI355X) for MXFP4 support",
)


@pytest.fixture(scope="function")
def init_parallel_state():
    """
    Initialize Megatron parallel state for tests that need it.

    This fixture initializes the parallel state with no actual parallelism
    (tensor_model_parallel_size=1), which allows tensor parallel layers to
    function in single-GPU unit tests.

    The fixture is function-scoped so each test gets a clean parallel state.
    It uses dynamic ports to avoid conflicts when running tests in parallel.

    Yields:
        None

    Example:
        @pytest.fixture(autouse=True)
        def setup_parallel(self, init_parallel_state):
            '''Auto-use parallel state for this test class.'''
            pass
    """
    from megatron.core import parallel_state as ps

    # Initialize torch.distributed if not already initialized
    if not torch.distributed.is_initialized():
        import os
        import socket

        # Use OS-assigned ephemeral port to avoid TIME_WAIT conflicts
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))

        try:
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
        except Exception as e:
            pytest.skip(f"Could not initialize distributed: {e}")

    # Check if model parallel already initialized
    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()

    # Initialize with minimal parallelism (TP=1, PP=1, EP=1, CP=1)
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )

    # Initialize RNG states for tensor parallel operations
    # This is required for:
    # 1. ColumnParallelLinear and other tensor parallel layers that use get_cuda_rng_tracker().fork()
    # 2. CUDA graph support in layers (enable_cuda_graph=True)
    from megatron.core.tensor_parallel import random as tp_random

    if torch.cuda.is_available():
        # Initialize RNG tracker with CUDA graph support BEFORE calling model_parallel_cuda_manual_seed
        # Try Transformer Engine RNG tracker first (best for CUDA graphs, used by Megatron's own tests)
        try:
            tp_random.initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        except (ImportError, AssertionError):
            # Fallback to native PyTorch CUDA graph RNG support if TE not available
            tp_random.initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)

        tp_random.model_parallel_cuda_manual_seed(42)

    # Initialize Megatron global args with minimal defaults required by
    # PrimusTurboLocalAttention and PrimusTorchFullyShardedDataParallel
    from megatron.training.global_vars import set_args

    set_args(
        SimpleNamespace(
            enable_turbo_attention_float8=False,
            data_parallel_replicate_degree=1,
        )
    )

    yield

    # Cleanup after test
    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()

    # Reset Megatron global args
    import megatron.training.global_vars as gvars

    gvars._GLOBAL_ARGS = None

    # Cleanup torch.distributed for single-process tests
    # (In multi-process torchrun tests, the process group persists across tests)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

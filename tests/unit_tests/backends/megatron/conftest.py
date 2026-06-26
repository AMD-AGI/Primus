###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Pytest fixtures for Megatron backend tests.

This conftest provides fixtures for tests that need Megatron parallel state
initialization, such as tests for Primus Turbo layers and other Megatron-Core
components that require distributed setup.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist


def _find_free_port() -> int:
    """Ask the kernel for a free TCP port on localhost.

    Binding to port 0 lets the OS pick a port that is guaranteed free at this
    instant, which avoids the EADDRINUSE failures seen when guessing a random
    port out of a fixed range that overlaps the ephemeral port range.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
        def setup_parallel(self, init_parallel_state, monkeypatch):
            '''Auto-use parallel state for this test class.'''
            pass
    """
    # Only import after sys.path is set up (which happens in pytest_configure)
    from megatron.core import parallel_state as ps

    # Initialize torch.distributed if not already initialized
    if not dist.is_initialized():
        # Use a kernel-assigned free port to avoid conflicts with other running
        # tests. Set MASTER_PORT explicitly (not setdefault) so a stale value
        # left in the environment cannot diverge from the port we actually use.
        port = _find_free_port()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=1,
            rank=0,
        )

    # Check if model parallel already initialized and destroy if so
    # This ensures clean state for each test
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
    if torch.cuda.is_available():
        try:
            from megatron.core.tensor_parallel import random as tp_random

            # Initialize RNG tracker with CUDA graph support BEFORE calling model_parallel_cuda_manual_seed
            # Try Transformer Engine RNG tracker first (best for CUDA graphs, used by Megatron's own tests)
            try:
                tp_random.initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
            except (ImportError, AssertionError):
                # Fallback to native PyTorch CUDA graph RNG support if TE not available
                tp_random.initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)

            tp_random.model_parallel_cuda_manual_seed(42)
        except ImportError:
            # RNG tracker initialization is optional - skip if not available
            pass

    yield

    # Cleanup after test
    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()

    # Cleanup torch.distributed for single-process tests
    # (In multi-process torchrun tests, the process group persists across tests)
    if dist.is_initialized():
        dist.destroy_process_group()

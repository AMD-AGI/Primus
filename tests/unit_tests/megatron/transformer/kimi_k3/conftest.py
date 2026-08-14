###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fixtures for the Kimi K3 / KDA unit tests.

Unlike the DeepSeek-V4 directory these tests are **not** hardware-gated:
the eager KDA reference is pure PyTorch and its numerics must hold
everywhere. Tests that need a specific backend (``fla``'s Triton kernels)
skip themselves individually.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def kda_device() -> str:
    """``"cuda"`` when an accelerator is visible, else ``"cpu"``.

    The eager reference is device-agnostic; running it on GPU when one is
    present keeps the longer shapes fast.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Seed every test so a failure is reproducible from its node id alone."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield


@pytest.fixture(autouse=True)
def _release_gpu_cache_per_test():
    """Free the caching allocator after each test.

    The eager reference materialises ``O(C)`` decay tensors per chunk, so
    the long-sequence cases hold a lot of transient memory; releasing it
    per test keeps allocator pressure test-local.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

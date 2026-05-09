###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-test cleanup hooks for plan-4 V4 attention tests.

The plan-4 release-tier shape gates (``pytest.mark.slow``, see G28 in
``deepseek-v4/develop/plan-4/02-phase-details.md``) allocate large
eager-reference tensors at production V4 dimensions
(``head_dim=512``, MQA / MHA, ``K_topk`` up to 1024). Specifically the
CSA path's ``torch.einsum("bhsd,bhskd->bhsk", q, gathered.unsqueeze(1)
.expand(...))`` materialises a ``[B, H, Sq, K, D]`` intermediate that
is up to ~64 GiB at fp32 V4-Flash release / ~128 GiB at fp32 V4-Pro
release. PyTorch's caching allocator otherwise holds those tensors
across parametrised tests in the same process, so the second / third
release-tier test in a pytest session ends up OOM-ing well before
exhausting the MI355 287 GiB HBM budget.

This conftest installs a function-scoped autouse fixture that empties
the CUDA / HIP cache after every test to keep the allocator pressure
test-local. The cost (~milliseconds) is negligible for the fast tier
and an outright requirement for the release tier.
"""

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:  # pragma: no cover — pytest collects on torchless envs
    torch = None  # type: ignore


@pytest.fixture(autouse=True)
def _release_gpu_cache_per_test():
    """Free PyTorch's CUDA / HIP cache after each V4 attention test."""
    yield
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

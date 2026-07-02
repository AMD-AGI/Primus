# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Parity test for the batched grouped-expert Newton-Schulz fast path in
:meth:`TensorParallelMuon.orthogonalize`.

At TP=1 the consolidated ``[E, N, K]`` expert momentum is orthogonalized in a
single batched Newton-Schulz call (``PRIMUS_MUON_BATCHED_NS`` default-on)
instead of an E-long per-expert Python loop. This asserts the batched path is
numerically equivalent to the loop it replaces (the loop is still used for the
partitioned TP>1 case, and is reachable via ``PRIMUS_MUON_BATCHED_NS=0``).
"""
from __future__ import annotations

import pytest
import torch

from primus.backends.megatron.core.optimizer.moun import (
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelMuon,
)

pytestmark = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS,
    reason="needs emerging_optimizers>=0.4.0a0 (batched 3-D newton_schulz)",
)


@pytest.mark.parametrize("shape", [(8, 64, 32), (6, 32, 48)])
def test_batched_3d_orthogonalize_matches_loop(shape, monkeypatch):
    """Batched 3-D orthogonalize == per-expert loop (CPU, fp32)."""
    torch.manual_seed(0)
    E, N, K = shape

    # fp32 + "high" matmul precision so the comparison isn't masked by the
    # bf16 ("medium") Newton-Schulz path; both branches use the same precision,
    # so the only difference under test is batched-vs-loop reduction order.
    prev_prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        p = torch.nn.Parameter(torch.zeros(E, N, K, dtype=torch.float32))
        opt = TensorParallelMuon([p], lr=1e-3, coefficient_type="deepseekv4", extra_scale_factor=0.18)
        grad = torch.randn(E, N, K, dtype=torch.float32)

        monkeypatch.setenv("PRIMUS_MUON_BATCHED_NS", "1")
        batched = opt.orthogonalize(p, grad.clone())

        monkeypatch.setenv("PRIMUS_MUON_BATCHED_NS", "0")
        loop = opt.orthogonalize(p, grad.clone())

        assert batched.shape == (E, N, K)
        torch.testing.assert_close(batched, loop, rtol=1e-4, atol=1e-4)
    finally:
        torch.set_float32_matmul_precision(prev_prec)

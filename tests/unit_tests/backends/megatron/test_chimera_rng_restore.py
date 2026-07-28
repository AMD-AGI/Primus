# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for `_restore_chimera_rng_state` defensive RNG restoration.

Validates Fix 2: when Megatron's private `_set_random_seed` API drifts
(signature change or removal), the helper falls back to a manual restore
that covers all three RNG generators — CPU default, CUDA default, and the
model-parallel tracker — so chimera training does not fail at startup.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.flux_pretrain_trainer import _restore_chimera_rng_state


def _canonical_args(seed=42):
    """Build the args namespace the helper reads."""
    return SimpleNamespace(
        seed=seed,
        data_parallel_random_init=False,
        te_rng_tracker=False,
        inference_rng_tracker=False,
        enable_cuda_graph=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestChimeraRngRestoreFallback:
    """The fallback path: when `_set_random_seed` raises TypeError or
    ImportError, the helper must restore CPU, CUDA default, and the
    model-parallel tracker generators manually so training continues."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state — required because the fallback's
        tp_random.model_parallel_cuda_manual_seed needs the tracker set up."""

    def _verify_all_rngs_match_seed(self, seed):
        """Sample from each generator and compare to a freshly-seeded
        reference. If the helper's fallback fully restored the state,
        the two samples must be bitwise identical for every generator."""
        from megatron.core.tensor_parallel import random as tp_random

        # CPU default generator.
        cpu_after = torch.randn(8)
        torch.manual_seed(seed)
        cpu_ref = torch.randn(8)
        assert torch.equal(cpu_after, cpu_ref), "CPU default generator was not restored to seed"

        # CUDA default generator.
        cuda_after = torch.randn(8, device="cuda")
        torch.cuda.manual_seed(seed)
        cuda_ref = torch.randn(8, device="cuda")
        assert torch.equal(cuda_after, cuda_ref), "CUDA default generator was not restored to seed"

        # Model-parallel tracker.
        with tp_random.get_cuda_rng_tracker().fork():
            tracker_after = torch.randn(8, device="cuda")
        tp_random.model_parallel_cuda_manual_seed(seed)
        with tp_random.get_cuda_rng_tracker().fork():
            tracker_ref = torch.randn(8, device="cuda")
        assert torch.equal(tracker_after, tracker_ref), "Model-parallel tracker was not restored to seed"

    def _contaminate_rngs(self, contamination_seed):
        """Set all three generators to a non-canonical seed so the test
        observes the helper actively restoring rather than no-op-passing."""
        from megatron.core.tensor_parallel import random as tp_random

        torch.manual_seed(contamination_seed)
        torch.cuda.manual_seed(contamination_seed)
        tp_random.model_parallel_cuda_manual_seed(contamination_seed)

    def test_typeerror_fallback_restores_all_rngs(self):
        """Megatron signature drift (TypeError) must trigger manual restore."""
        seed = 42
        self._contaminate_rngs(contamination_seed=999)

        args = _canonical_args(seed=seed)
        with patch(
            "megatron.training.initialize._set_random_seed",
            side_effect=TypeError("signature changed"),
        ):
            _restore_chimera_rng_state(args)

        self._verify_all_rngs_match_seed(seed)

    def test_importerror_fallback_restores_all_rngs(self):
        """Megatron module removal (ImportError) must trigger manual restore."""
        seed = 42
        self._contaminate_rngs(contamination_seed=999)

        args = _canonical_args(seed=seed)
        with patch(
            "megatron.training.initialize._set_random_seed",
            side_effect=ImportError("module gone"),
        ):
            _restore_chimera_rng_state(args)

        self._verify_all_rngs_match_seed(seed)

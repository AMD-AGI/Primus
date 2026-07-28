# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests that torch.compile does not alter sharded_state_dict keys.

The "whole_model" compile strategy compiles self.forward without wrapping
submodules, so state_dict keys must remain identical before and after
compilation. This test guards against regressions (e.g., _orig_mod prefixes
leaking into keys).
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from tests.utils import PrimusUT


class TestFluxCompileCheckpointKeys(PrimusUT):
    """Verify torch.compile does not change sharded_state_dict keys."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for model tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_compile_does_not_change_sharded_state_dict_keys(self):
        """Keys from sharded_state_dict() must be identical before and after compile_model().

        Compile must actually run (enable_torch_compile=True), otherwise
        compile_model() returns early and the test is vacuous. 'per_block'
        rebinds each layer's .forward (no module re-wrap), so a regression that
        wraps the module instead would leak '_orig_mod' prefixes into the keys
        and fail this test.
        """
        config = FluxConfig.flux_535m()
        config.enable_torch_compile = True
        config.torch_compile_strategy = "per_block"
        model = Flux(config).cuda()

        keys_before = set(model.sharded_state_dict().keys())

        model.compile_model()

        keys_after = set(model.sharded_state_dict().keys())

        added = keys_after - keys_before
        removed = keys_before - keys_after

        assert not added, f"Keys added after compile: {added}"
        assert not removed, f"Keys removed after compile: {removed}"

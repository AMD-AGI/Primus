# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for diffusion model normalization layers.

Tests RMSNorm, AdaLN, and AdaLNContinuous.

NOTE: These tests require CUDA and Megatron parallel state initialization.
"""

import pytest
import torch
import torch.nn as nn
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.models.diffusion.common.normalization import (
    AdaLN,
    AdaLNContinuous,
)
from tests.unit_tests.backends.megatron.diffusion.constants import (
    ATTENTION_SEQ_LEN,
    BATCH_SIZE_QUAD,
    HIDDEN_DIM_FLUX,
    NUM_ATTENTION_HEADS_FLUX,
)
from tests.utils import PrimusUT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA (uses ColumnParallelLinear)")
class TestAdaLN(PrimusUT):
    """Tests for AdaLN class."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """
        Initialize parallel state for AdaLN tests.

        AdaLN uses ColumnParallelLinear which requires Megatron's RNG tracker.
        The init_parallel_state fixture handles initialization and cleanup.
        """

    def test_forward_output_chunks(self):
        """Test that AdaLN produces correct number of chunks and numeric behavior."""
        config = TransformerConfig(
            hidden_size=HIDDEN_DIM_FLUX,
            num_attention_heads=NUM_ATTENTION_HEADS_FLUX,
            num_layers=1,
        )
        n_chunks = 9
        adaln = AdaLN(config, n_adaln_chunks=n_chunks).cuda()

        timestep_emb = torch.randn(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX).cuda()
        chunks = adaln(timestep_emb)

        assert len(chunks) == n_chunks, f"Expected {n_chunks} chunks, got {len(chunks)}"
        for chunk in chunks:
            assert chunk.shape == (BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX)

        # Numeric: gate=0 should zero out the contribution in scale_add
        x = torch.randn(16, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX).cuda()
        residual = torch.randn(16, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX).cuda()
        zero_gate = torch.zeros(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX).cuda()
        result = adaln.scale_add(residual, x, zero_gate)
        assert torch.allclose(result, residual, atol=1e-6), "gate=0 should leave residual unchanged"

    def test_default_init_method_zeros_modulation_weight(self):
        """The default init_method is nn.init.zeros_; modulation weight comes up zero.

        Guards the post-Flux-PR contract that AdaLN's default init is
        observably zero, so downstream callers don't accidentally pick up
        the NeMo-aligned normal_ RNG draw without opting in.
        """
        config = TransformerConfig(
            hidden_size=HIDDEN_DIM_FLUX,
            num_attention_heads=NUM_ATTENTION_HEADS_FLUX,
            num_layers=1,
        )
        adaln = AdaLN(config, n_adaln_chunks=6).cuda()
        weight = adaln.adaLN_modulation[-1].weight
        assert torch.equal(
            weight, torch.zeros_like(weight)
        ), "Default AdaLN init_method should produce a zero modulation weight"

    def test_normal_init_method_produces_nonzero_modulation_weight(self):
        """Passing nn.init.normal_ produces a nonzero pre-init_weights() draw.

        This is the call sites used by Flux's layer_spec.py to match NeMo's
        RNG sequence. Flux's init_weights() immediately re-zeroes these
        weights, so the only observable effect is RNG advancement.
        """
        config = TransformerConfig(
            hidden_size=HIDDEN_DIM_FLUX,
            num_attention_heads=NUM_ATTENTION_HEADS_FLUX,
            num_layers=1,
        )
        torch.manual_seed(0)
        adaln = AdaLN(config, n_adaln_chunks=6, init_method=nn.init.normal_).cuda()
        weight = adaln.adaLN_modulation[-1].weight
        assert not torch.equal(
            weight, torch.zeros_like(weight)
        ), "init_method=normal_ should draw a nonzero modulation weight"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="AdaLNContinuous forward dispatches to primus::fused_ln_modulate, "
    "which has no CPU kernel registered.",
)
class TestAdaLNContinuous(PrimusUT):
    """Tests for AdaLNContinuous class."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """AdaLNContinuous uses RowParallelLinear which requires Megatron's
        RNG tracker / parallel state. Mirror TestAdaLN."""

    def test_output_shape(self):
        """Test that AdaLNContinuous produces correct output shape."""
        config = TransformerConfig(
            hidden_size=HIDDEN_DIM_FLUX,
            num_attention_heads=NUM_ATTENTION_HEADS_FLUX,
            num_layers=1,
        )
        adaln = AdaLNContinuous(config, conditioning_embedding_dim=HIDDEN_DIM_FLUX).cuda()

        # Use sequence-first format: [seq_len, batch, hidden]
        x = torch.randn(ATTENTION_SEQ_LEN, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX).cuda()
        cond = torch.randn(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX).cuda()

        output = adaln(x, cond)
        assert output.shape == x.shape

    def test_invalid_norm_type(self):
        """Test that invalid norm type raises error (Primus validation)."""
        config = TransformerConfig(
            hidden_size=HIDDEN_DIM_FLUX,
            num_attention_heads=NUM_ATTENTION_HEADS_FLUX,
            num_layers=1,
        )

        with pytest.raises(ValueError, match="Unknown normalization type"):
            AdaLNContinuous(config, conditioning_embedding_dim=HIDDEN_DIM_FLUX, norm_type="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

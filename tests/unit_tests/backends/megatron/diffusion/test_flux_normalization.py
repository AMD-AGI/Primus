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

    def test_fused_forward_matches_plain_ops_formula(self):
        """Default fused path must still equal norm(x)*(1+scale)+shift (NeMo chunk order)."""
        config = TransformerConfig(
            hidden_size=HIDDEN_DIM_FLUX,
            num_attention_heads=NUM_ATTENTION_HEADS_FLUX,
            num_layers=1,
        )
        torch.manual_seed(0)
        adaln = AdaLNContinuous(config, conditioning_embedding_dim=HIDDEN_DIM_FLUX).cuda()
        x = torch.randn(ATTENTION_SEQ_LEN, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX).cuda()
        cond = torch.randn(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX).cuda()

        output = adaln(x, cond)
        emb = adaln.adaLN_modulation(cond)
        scale, shift = torch.chunk(emb, 2, dim=1)
        expected = adaln.norm(x) * (1 + scale) + shift
        assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)

    def test_fused_forward_backward_reaches_input_and_conditioning(self):
        """Gradients must flow to both x and cond through the fused CUDA custom op
        (primus::fused_ln_modulate), not just the CPU plain-ops branch."""
        config = TransformerConfig(
            hidden_size=HIDDEN_DIM_FLUX,
            num_attention_heads=NUM_ATTENTION_HEADS_FLUX,
            num_layers=1,
        )
        adaln = AdaLNContinuous(config, conditioning_embedding_dim=HIDDEN_DIM_FLUX).cuda()
        x = torch.randn(
            ATTENTION_SEQ_LEN, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX, device="cuda", requires_grad=True
        )
        cond = torch.randn(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX, device="cuda", requires_grad=True)

        adaln(x, cond).sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert cond.grad is not None and torch.isfinite(cond.grad).all() and cond.grad.abs().sum() > 0


class TestAdaLNContinuousForwardPlainOps(PrimusUT):
    """CPU-only tests for AdaLNContinuous.forward numerics.

    These construct AdaLNContinuous with config.adaln_plain_ops=True, which
    routes forward() through the plain `self.norm(x) * (1 + scale) + shift`
    branch instead of the primus::fused_ln_modulate custom op (registered
    only for device_types="cuda"). AdaLNContinuous itself only uses
    nn.Linear/nn.LayerNorm (no tensor-parallel layers), so no CUDA or
    Megatron parallel state is required for this path.
    """

    @staticmethod
    def _make_config():
        config = TransformerConfig(
            hidden_size=HIDDEN_DIM_FLUX,
            num_attention_heads=NUM_ATTENTION_HEADS_FLUX,
            num_layers=1,
        )
        config.adaln_plain_ops = True
        return config

    def test_forward_matches_manual_layernorm_and_modulate(self):
        """forward() should equal norm(x) * (1 + scale) + shift for the given conditioning."""
        config = self._make_config()
        torch.manual_seed(0)
        adaln = AdaLNContinuous(config, conditioning_embedding_dim=HIDDEN_DIM_FLUX, modulation_bias=False)
        x = torch.randn(ATTENTION_SEQ_LEN, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX)
        cond = torch.randn(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX)

        output = adaln(x, cond)

        emb = adaln.adaLN_modulation(cond)
        scale, shift = torch.chunk(emb, 2, dim=1)
        expected = adaln.norm(x) * (1 + scale) + shift

        assert output.shape == x.shape
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_zero_modulation_weight_is_pure_layernorm(self):
        """Zeroed modulation weight/no bias => scale=shift=0, so forward reduces to plain LayerNorm."""
        config = self._make_config()
        adaln = AdaLNContinuous(config, conditioning_embedding_dim=HIDDEN_DIM_FLUX, modulation_bias=False)
        nn.init.zeros_(adaln.adaLN_modulation[-1].weight)

        x = torch.randn(ATTENTION_SEQ_LEN, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX)
        cond = torch.randn(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX)

        output = adaln(x, cond)
        expected = torch.nn.functional.layer_norm(x, [HIDDEN_DIM_FLUX], eps=1e-6)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_forward_scale_shift_chunk_order(self):
        """First half of the modulation output is scale, second half is shift (NeMo convention)."""
        config = self._make_config()
        adaln = AdaLNContinuous(config, conditioning_embedding_dim=HIDDEN_DIM_FLUX, modulation_bias=True)

        with torch.no_grad():
            adaln.adaLN_modulation[-1].weight.zero_()
            bias = adaln.adaLN_modulation[-1].bias
            bias[:HIDDEN_DIM_FLUX] = 1.0  # scale half -> scale=1 everywhere
            bias[HIDDEN_DIM_FLUX:] = 5.0  # shift half -> shift=5 everywhere

        x = torch.randn(ATTENTION_SEQ_LEN, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX)
        cond = torch.randn(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX)

        output = adaln(x, cond)
        expected = adaln.norm(x) * 2.0 + 5.0
        assert torch.allclose(output, expected, atol=1e-5)

    def test_forward_backward_reaches_input_and_modulation(self):
        config = self._make_config()
        adaln = AdaLNContinuous(config, conditioning_embedding_dim=HIDDEN_DIM_FLUX, modulation_bias=False)
        x = torch.randn(ATTENTION_SEQ_LEN, BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX, requires_grad=True)
        cond = torch.randn(BATCH_SIZE_QUAD, HIDDEN_DIM_FLUX, requires_grad=True)

        adaln(x, cond).sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        assert cond.grad is not None and torch.isfinite(cond.grad).all() and cond.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for CLIP text encoder implementations.

Tests CLIPLEncoder with mocked models for fast unit testing.

NOTE: Common wrapper tests (initialization, from_pretrained, device handling, etc.)
have been moved to test_encoder_wrappers_consolidated.py. This file contains only
CLIP-specific tests that verify Primus's CLIP integration logic.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from primus.backends.megatron.data.diffusion.encoders.config import CLIPLConfig


class TestCLIPLEncoder:
    """Tests for CLIPLEncoder Primus-specific logic."""

    @patch("primus.backends.megatron.data.diffusion.encoders.text.clip_l.CLIPTextModel")
    @patch("primus.backends.megatron.data.diffusion.encoders.text.clip_l.CLIPTokenizer")
    def test_pooled_embeddings_extraction(
        self, mock_tokenizer_cls, mock_model_cls, mock_clip_model, mock_clip_tokenizer
    ):
        """Test that pooled embeddings are correctly extracted (Primus-specific CLIP integration)."""
        from primus.backends.megatron.data.diffusion.encoders.text import CLIPLEncoder

        mock_model_cls.from_pretrained.return_value = mock_clip_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_clip_tokenizer

        # Setup mock tokenizer
        mock_clip_tokenizer.return_value = {
            "input_ids": torch.randint(0, 49408, (2, 77)),
            "attention_mask": torch.ones(2, 77),
        }

        # Setup mock model output with specific pooled embeddings
        expected_pooled = torch.randn(2, 768)
        mock_clip_model.return_value = MagicMock(
            last_hidden_state=torch.randn(2, 77, 768), pooler_output=expected_pooled
        )

        config = CLIPLConfig(type="clip_l", model_path="/path/to/clip")
        encoder = CLIPLEncoder.from_pretrained("/path/to/clip", config=config)

        sequence_embeds, pooled_embeds = encoder.encode(["Text 1", "Text 2"])

        # Must return the model's pooler_output verbatim (not e.g. a mean-pool of
        # last_hidden_state, which would have the same shape but wrong values).
        assert pooled_embeds.shape == expected_pooled.shape
        assert torch.equal(pooled_embeds, expected_pooled), "pooled_embeds must equal pooler_output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

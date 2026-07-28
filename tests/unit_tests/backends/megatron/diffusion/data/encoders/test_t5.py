# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for T5 text encoder implementations.

Tests T5XXLEncoder with mocked models for fast unit testing.

NOTE: Common wrapper tests (initialization, from_pretrained, device handling, etc.)
have been moved to test_encoder_wrappers_consolidated.py. This file contains only
T5-specific tests that verify Primus's T5 integration logic.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from primus.backends.megatron.data.diffusion.encoders.config import T5XXLConfig


class TestT5XXLEncoder:
    """Tests for T5XXLEncoder Primus-specific logic."""

    @patch("primus.backends.megatron.data.diffusion.encoders.text.t5_xxl.T5EncoderModel")
    @patch("primus.backends.megatron.data.diffusion.encoders.text.t5_xxl.T5Tokenizer")
    def test_encode_with_padding_truncation(
        self, mock_tokenizer_cls, mock_model_cls, mock_t5_model, mock_t5_tokenizer
    ):
        """Test that tokenizer is called with correct padding/truncation (Primus tokenization logic)."""
        from primus.backends.megatron.data.diffusion.encoders.text import T5XXLEncoder

        mock_model_cls.from_pretrained.return_value = mock_t5_model
        mock_tokenizer_cls.from_pretrained.return_value = mock_t5_tokenizer

        # Setup mock tokenizer
        mock_t5_tokenizer.return_value = {
            "input_ids": torch.randint(0, 32000, (1, 512)),
            "attention_mask": torch.ones(1, 512),
        }

        # Setup mock model output
        mock_t5_model.return_value = MagicMock(last_hidden_state=torch.randn(1, 512, 4096))

        config = T5XXLConfig(type="t5_xxl", model_path="/path/to/t5")
        encoder = T5XXLEncoder.from_pretrained("/path/to/t5", config=config)

        encoder.encode("Test caption")

        # Verify tokenizer was called with correct Primus padding/truncation params
        call_kwargs = mock_t5_tokenizer.call_args[1]
        assert call_kwargs["padding"] == "max_length"
        assert call_kwargs["truncation"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

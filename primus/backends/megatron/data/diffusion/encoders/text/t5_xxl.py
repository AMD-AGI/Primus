# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Adapted from NeMo's text encoder implementation

"""
T5-XXL text encoder implementation for Flux.

This module provides a wrapper around the transformers T5EncoderModel
for encoding text prompts into embeddings for Flux diffusion models.
"""

import logging
from typing import List, Optional, Union

import torch

try:
    from transformers import T5EncoderModel, T5Tokenizer
except ImportError:
    T5EncoderModel = None
    T5Tokenizer = None

from primus.backends.megatron.data.diffusion.encoders.base import (
    BaseTextEncoder,
    get_torch_dtype,
    load_pretrained_with_subfolder_fallback,
)
from primus.backends.megatron.data.diffusion.encoders.config import T5XXLConfig

logger = logging.getLogger(__name__)


class T5XXLEncoder(BaseTextEncoder):
    """
    T5-XXL text encoder for Flux diffusion models.

    This encoder uses the T5-v1.1-XXL model from HuggingFace to encode
    text prompts into embeddings. For Flux, the default configuration is:
        - max_length: 512 tokens
        - embedding_dim: 4096 (T5-XXL hidden size)
        - precision: bf16

    The output embeddings have shape (batch_size, seq_len, 4096).
    """

    def __init__(self, config: T5XXLConfig):
        """
        Initialize T5-XXL encoder.

        Args:
            config: T5XXLConfig with model_path, max_length, etc.
        """
        super().__init__(config)

        if T5EncoderModel is None or T5Tokenizer is None:
            raise ImportError(
                "transformers library is required for T5XXLEncoder. " "Install with: pip install transformers"
            )

        self.transformer = None  # Will be loaded in from_pretrained
        self._tokenizer = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[T5XXLConfig] = None,
        subfolder: Optional[str] = None,
    ) -> "T5XXLEncoder":
        """
        Load T5-XXL encoder from pretrained weights.

        Args:
            model_path: Path to pretrained model (local path or HuggingFace repo)
            config: Optional T5XXLConfig. If None, uses defaults.
            subfolder: Subfolder for model weights. Priority: param > config.subfolder

        Returns:
            Loaded T5XXLEncoder instance

        Examples:
            >>> # Using config (recommended)
            >>> config = T5XXLConfig(
            ...     model_path="black-forest-labs/FLUX.1-dev",
            ...     subfolder="text_encoder_2",
            ...     tokenizer_subfolder="tokenizer_2"
            ... )
            >>> encoder = T5XXLEncoder.from_pretrained("black-forest-labs/FLUX.1-dev", config=config)

            >>> # Using method parameter (backward compatible)
            >>> encoder = T5XXLEncoder.from_pretrained("google/t5-v1_1-xxl", subfolder=None)
        """
        if config is None:
            config = T5XXLConfig(
                type="t5_xxl",
                model_path=model_path,
                precision="bf16",
            )

        instance = cls(config)

        # Prepare kwargs for from_pretrained calls
        pretrained_kwargs = {}
        if config.cache_dir:
            pretrained_kwargs["cache_dir"] = config.cache_dir
            logger.info(f"Using cache directory: {config.cache_dir}")

        # Resolve model subfolder with priority: param > config.subfolder > error
        model_subfolder = subfolder if subfolder is not None else getattr(config, "subfolder", None)

        # Require explicit configuration for model subfolder
        # NOTE: config.subfolder can legitimately be None for standalone models like google/t5-v1_1-xxl
        # but it must be explicitly set in the config. If neither param nor config specify it, that's an error.
        if model_subfolder is None and not hasattr(config, "subfolder"):
            raise ValueError(
                f"subfolder must be specified for T5XXLEncoder with model_path='{model_path}'. "
                f"For FLUX models (e.g., black-forest-labs/FLUX.1-dev), use subfolder='text_encoder_2'. "
                f"For standalone T5 models (e.g., google/t5-v1_1-xxl), use subfolder=None. "
                f"Set it via config.subfolder or the subfolder parameter."
            )

        # Resolve tokenizer subfolder with priority: config.tokenizer_subfolder > model_subfolder
        tokenizer_subfolder = getattr(config, "tokenizer_subfolder", None)
        if tokenizer_subfolder is None:
            tokenizer_subfolder = model_subfolder  # Use model subfolder as fallback

        # Load tokenizer
        tokenizer_path = config.tokenizer_path or model_path
        logger.info(f"Loading T5 tokenizer from {tokenizer_path} (subfolder={tokenizer_subfolder})")

        # Load tokenizer from subfolder
        import os

        tokenizer_kwargs = {
            "token": os.environ.get("HF_TOKEN"),
            "trust_remote_code": getattr(config, "trust_remote_code", False),
            **pretrained_kwargs,  # Merge cache_dir if provided
        }
        if tokenizer_subfolder:
            tokenizer_kwargs["subfolder"] = tokenizer_subfolder

        instance._tokenizer = T5Tokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )

        # Load model
        torch_dtype = get_torch_dtype(config.precision)

        logger.info(f"Loading T5-XXL model from {model_path} (subfolder={model_subfolder})")
        instance.transformer = load_pretrained_with_subfolder_fallback(
            T5EncoderModel,
            model_path,
            subfolder=model_subfolder,
            torch_dtype=torch_dtype,
            **pretrained_kwargs,
        )

        instance.transformer.to(instance.device)

        if config.freeze_weights:
            instance.freeze()
            instance.transformer.eval()

        logger.info(
            f"Loaded T5-XXL: max_length={instance.max_length}, "
            f"embedding_dim={instance.embedding_dim}, dtype={torch_dtype}"
        )

        return instance

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        max_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text string or list of text strings
            max_sequence_length: Optional max length override. If None, uses config max_length.

        Returns:
            Text embeddings tensor of shape (batch_size, seq_len, 4096)
            Sequence length is padded to max_length.

        Example:
            >>> texts = ["A beautiful sunset over the ocean", "A cat sitting on a mat"]
            >>> embeddings = encoder.encode(texts)
            >>> embeddings.shape
            torch.Size([2, 512, 4096])
        """
        if self.transformer is None or self._tokenizer is None:
            raise RuntimeError("T5 encoder not loaded. Call from_pretrained() first.")

        texts = self._prepare_texts(texts)
        max_len = max_sequence_length if max_sequence_length is not None else self.max_length

        # Tokenize
        batch_encoding = self._tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)

        # Encode
        outputs = self.transformer(input_ids=tokens, output_hidden_states=None)
        embeddings = outputs.last_hidden_state

        return embeddings

    def forward(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(texts, **kwargs)


# Register encoder in registry
from primus.backends.megatron.data.diffusion.encoders import register_encoder

register_encoder("t5_xxl", T5XXLEncoder)

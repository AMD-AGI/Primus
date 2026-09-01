# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
UMT5 text encoder implementation for Wan.

Wan models condition on UMT5 features (a multilingual T5 variant) instead of
Flux's T5-XXL + CLIP-L pair. The interface mirrors ``T5XXLEncoder``: load
weights with HuggingFace transformers, tokenize, and run the encoder.

Reference:
    https://huggingface.co/docs/diffusers/en/api/pipelines/wan
    Wan technical report: https://arxiv.org/abs/2503.20314
"""

import logging
import os
from typing import List, Optional, Union

import torch

try:
    # UMT5 weights are stored as a T5-style encoder, so the generic
    # T5EncoderModel loads them; some transformers releases additionally expose
    # an explicit UMT5EncoderModel, which is preferred when available.
    from transformers import AutoTokenizer, T5EncoderModel

    try:
        from transformers import UMT5EncoderModel
    except ImportError:
        UMT5EncoderModel = None
except ImportError:
    AutoTokenizer = None
    T5EncoderModel = None
    UMT5EncoderModel = None

from primus.backends.megatron.data.diffusion.encoders.base import (
    BaseTextEncoder,
    get_torch_dtype,
    load_pretrained_with_subfolder_fallback,
)
from primus.backends.megatron.data.diffusion.encoders.config import UMT5Config

logger = logging.getLogger(__name__)


class UMT5Encoder(BaseTextEncoder):
    """
    UMT5 text encoder for Wan diffusion models.

    This encoder uses the multilingual UMT5-XXL model from HuggingFace to
    encode text prompts into embeddings. For Wan, the default configuration is:
        - max_length: 512 tokens
        - embedding_dim: 4096 (UMT5-XXL hidden size)
        - precision: bf16

    The output embeddings have shape (batch_size, seq_len, 4096).
    """

    def __init__(self, config: UMT5Config):
        """
        Initialize UMT5 encoder.

        Args:
            config: UMT5Config with model_path, max_length, etc.
        """
        super().__init__(config)

        if T5EncoderModel is None or AutoTokenizer is None:
            raise ImportError(
                "transformers library is required for UMT5Encoder. "
                "Install with: pip install -U transformers"
            )

        self.transformer = None  # Will be loaded in from_pretrained
        self._tokenizer = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[UMT5Config] = None,
        subfolder: Optional[str] = None,
    ) -> "UMT5Encoder":
        """
        Load UMT5 encoder from pretrained weights.

        Args:
            model_path: Path to pretrained model (local path or HuggingFace repo)
            config: Optional UMT5Config. If None, uses defaults.
            subfolder: Subfolder for model weights. Priority: param > config.subfolder

        Returns:
            Loaded UMT5Encoder instance

        Examples:
            >>> config = UMT5Config(
            ...     model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            ...     subfolder="text_encoder",
            ...     tokenizer_subfolder="tokenizer",
            ... )
            >>> encoder = UMT5Encoder.from_pretrained(
            ...     "Wan-AI/Wan2.1-T2V-14B-Diffusers", config=config
            ... )
        """
        if config is None:
            config = UMT5Config(
                type="umt5",
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

        if model_subfolder is None and not hasattr(config, "subfolder"):
            raise ValueError(
                f"subfolder must be specified for UMT5Encoder with model_path='{model_path}'. "
                f"For Wan diffusers repos (e.g., Wan-AI/Wan2.1-T2V-14B-Diffusers), use "
                f"subfolder='text_encoder'. For standalone UMT5 checkpoints, use subfolder=None. "
                f"Set it via config.subfolder or the subfolder parameter."
            )

        # Resolve tokenizer subfolder with priority: config.tokenizer_subfolder > model_subfolder
        tokenizer_subfolder = getattr(config, "tokenizer_subfolder", None)
        if tokenizer_subfolder is None:
            tokenizer_subfolder = model_subfolder  # Use model subfolder as fallback

        # Load tokenizer
        tokenizer_path = config.tokenizer_path or model_path
        logger.info(f"Loading UMT5 tokenizer from {tokenizer_path} (subfolder={tokenizer_subfolder})")

        tokenizer_kwargs = {
            "token": os.environ.get("HF_TOKEN"),
            "trust_remote_code": getattr(config, "trust_remote_code", False),
            **pretrained_kwargs,  # Merge cache_dir if provided
        }
        if tokenizer_subfolder:
            tokenizer_kwargs["subfolder"] = tokenizer_subfolder

        instance._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **tokenizer_kwargs,
        )

        # Load model
        torch_dtype = get_torch_dtype(config.precision)

        logger.info(f"Loading UMT5 encoder from {model_path} (subfolder={model_subfolder})")
        loader_class = UMT5EncoderModel if UMT5EncoderModel is not None else T5EncoderModel
        instance.transformer = load_pretrained_with_subfolder_fallback(
            loader_class,
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
            f"Loaded UMT5: max_length={instance.max_length}, "
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
            >>> texts = ["A cat walking through tall grass"]
            >>> embeddings = encoder.encode(texts)
            >>> embeddings.shape
            torch.Size([1, 512, 4096])
        """
        if self.transformer is None or self._tokenizer is None:
            raise RuntimeError("UMT5 encoder not loaded. Call from_pretrained() first.")

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
        attention_mask = batch_encoding["attention_mask"].to(self.device, non_blocking=True)

        # Encode
        outputs = self.transformer(
            input_ids=tokens,
            attention_mask=attention_mask,
            output_hidden_states=None,
        )
        embeddings = outputs.last_hidden_state

        return embeddings

    def forward(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(texts, **kwargs)


# Register encoder in registry
from primus.backends.megatron.data.diffusion.encoders import register_encoder

register_encoder("umt5", UMT5Encoder)

# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Adapted from NeMo's text encoder implementation

"""
CLIP-L text encoder implementation for Flux.

This module provides a wrapper around the transformers CLIPTextModel
for encoding text prompts into embeddings for Flux diffusion models.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch

try:
    from transformers import CLIPTextModel, CLIPTokenizer
except ImportError:
    CLIPTextModel = None
    CLIPTokenizer = None

from primus.backends.megatron.data.diffusion.encoders.base import (
    BaseTextEncoder,
    get_torch_dtype,
    load_pretrained_with_subfolder_fallback,
)
from primus.backends.megatron.data.diffusion.encoders.config import CLIPLConfig

logger = logging.getLogger(__name__)


class CLIPLEncoder(BaseTextEncoder):
    """
    CLIP-L text encoder for Flux diffusion models.

    This encoder uses the CLIP ViT-L/14 text encoder from HuggingFace to encode
    text prompts into embeddings. For Flux, the default configuration is:
        - max_length: 77 tokens (CLIP default)
        - embedding_dim: 768 (CLIP-L hidden size)
        - pooled_dim: 768 (CLIP-L pooled embedding size)
        - precision: bf16

    The encoder returns both sequence embeddings (B, seq_len, 768) and
    pooled embeddings (B, 768) which are used differently in Flux.

    Note: Sequence length is padded to a multiple of 8 for Flux compatibility.
    """

    def __init__(self, config: CLIPLConfig):
        """
        Initialize CLIP-L encoder.

        Args:
            config: CLIPLConfig with model_path, max_length, etc.
        """
        super().__init__(config)
        self.pooled_dim = config.pooled_dim if hasattr(config, "pooled_dim") else 768

        if CLIPTextModel is None or CLIPTokenizer is None:
            raise ImportError(
                "transformers library is required for CLIPLEncoder. " "Install with: pip install transformers"
            )

        self.transformer = None  # Will be loaded in from_pretrained
        self._tokenizer = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[CLIPLConfig] = None,
        subfolder: Optional[str] = None,
    ) -> "CLIPLEncoder":
        """
        Load CLIP-L encoder from pretrained weights.

        Args:
            model_path: Path to pretrained model (local path or HuggingFace repo)
            config: Optional CLIPLConfig. If None, uses defaults.
            subfolder: Subfolder for model weights. Priority: param > config.subfolder

        Returns:
            Loaded CLIPLEncoder instance

        Examples:
            >>> # Using config (recommended)
            >>> config = CLIPLConfig(
            ...     model_path="black-forest-labs/FLUX.1-dev",
            ...     subfolder="text_encoder",
            ...     tokenizer_subfolder="tokenizer"
            ... )
            >>> encoder = CLIPLEncoder.from_pretrained("black-forest-labs/FLUX.1-dev", config=config)

            >>> # Using method parameter (backward compatible)
            >>> encoder = CLIPLEncoder.from_pretrained("openai/clip-vit-large-patch14", subfolder=None)
        """
        if config is None:
            config = CLIPLConfig(
                type="clip_l",
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
        # NOTE: config.subfolder can legitimately be None for standalone models like openai/clip-vit-large-patch14
        # but it must be explicitly set in the config. If neither param nor config specify it, that's an error.
        if model_subfolder is None and not hasattr(config, "subfolder"):
            raise ValueError(
                f"subfolder must be specified for CLIPLEncoder with model_path='{model_path}'. "
                f"For FLUX models (e.g., black-forest-labs/FLUX.1-dev), use subfolder='text_encoder'. "
                f"For standalone CLIP models (e.g., openai/clip-vit-large-patch14), use subfolder=None. "
                f"Set it via config.subfolder or the subfolder parameter."
            )

        # Resolve tokenizer subfolder with priority: config.tokenizer_subfolder > model_subfolder
        tokenizer_subfolder = getattr(config, "tokenizer_subfolder", None)
        if tokenizer_subfolder is None:
            tokenizer_subfolder = model_subfolder  # Use model subfolder as fallback

        # Load tokenizer
        tokenizer_path = config.tokenizer_path or model_path
        logger.info(f"Loading CLIP tokenizer from {tokenizer_path} (subfolder={tokenizer_subfolder})")
        try:
            instance._tokenizer = load_pretrained_with_subfolder_fallback(
                CLIPTokenizer,
                tokenizer_path,
                subfolder=tokenizer_subfolder,
                **pretrained_kwargs,
            )
        except Exception as e:
            # Try standard CLIP tokenizer if neither subfolder nor root work
            logger.info(f"Failed loading from {tokenizer_path}, trying standard CLIP tokenizer: {e}")
            instance._tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                **pretrained_kwargs,
            )

        # Load model
        torch_dtype = get_torch_dtype(config.precision)

        logger.info(f"Loading CLIP-L model from {model_path} (subfolder={model_subfolder})")
        instance.transformer = load_pretrained_with_subfolder_fallback(
            CLIPTextModel,
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
            f"Loaded CLIP-L: max_length={instance.max_length}, "
            f"embedding_dim={instance.embedding_dim}, pooled_dim={instance.pooled_dim}, "
            f"dtype={torch_dtype}"
        )

        return instance

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        max_sequence_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text string or list of text strings
            max_sequence_length: Optional max length override. If None, uses config max_length.

        Returns:
            Tuple of (sequence_embeddings, pooled_embeddings):
                - sequence_embeddings: (batch_size, padded_seq_len, 768)
                  Sequence length is padded to a multiple of 8 for Flux
                - pooled_embeddings: (batch_size, 768)
                  Pooled representation from CLIP's [CLS] token

        Example:
            >>> texts = ["A beautiful sunset over the ocean", "A cat sitting on a mat"]
            >>> seq_embeds, pooled_embeds = encoder.encode(texts)
            >>> seq_embeds.shape
            torch.Size([2, 80, 768])  # Padded to multiple of 8
            >>> pooled_embeds.shape
            torch.Size([2, 768])
        """
        if self.transformer is None or self._tokenizer is None:
            raise RuntimeError("CLIP encoder not loaded. Call from_pretrained() first.")

        texts = self._prepare_texts(texts)
        max_len = max_sequence_length if max_sequence_length is not None else self.max_length

        # Tokenize
        batch_encoding = self._tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)

        # Encode
        outputs = self.transformer(input_ids=tokens, output_hidden_states=False)

        # Get sequence embeddings (last hidden state)
        sequence_embeddings = outputs.last_hidden_state

        # Pad sequence length to multiple of 8 (required for Flux)
        seq_len = sequence_embeddings.shape[1]
        padded_seq_len = ((seq_len + 7) // 8) * 8  # Round up to nearest multiple of 8

        if padded_seq_len > seq_len:
            sequence_embeddings = torch.nn.functional.pad(
                sequence_embeddings,
                (0, 0, 0, padded_seq_len - seq_len),  # Pad on sequence dimension
                value=0.0,
            )

        # Get pooled embeddings (from pooler output)
        pooled_embeddings = outputs.pooler_output

        return sequence_embeddings, pooled_embeddings

    def forward(self, texts: Union[str, List[str]], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (alias for encode)."""
        return self.encode(texts, **kwargs)


# Register encoder in registry
from primus.backends.megatron.data.diffusion.encoders import register_encoder

register_encoder("clip_l", CLIPLEncoder)

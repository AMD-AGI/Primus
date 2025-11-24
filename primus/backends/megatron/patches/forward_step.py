###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Forward step function provider for Megatron training.

Provides model-specific forward step functions.
"""

from types import SimpleNamespace


def get_forward_step(args: SimpleNamespace):
    """
    Get the forward step function based on model configuration.

    Args:
        args: Megatron argument namespace

    Returns:
        Forward step function compatible with Megatron's training loop
    """
    # Get model type
    model_type = getattr(args, "model_type", "GPT").upper()

    if "GPT" in model_type or "DECODER" in model_type:
        return _get_gpt_forward_step()
    elif "BERT" in model_type or "ENCODER" in model_type:
        return _get_bert_forward_step()
    elif "T5" in model_type:
        return _get_t5_forward_step()
    else:
        # Default to GPT-style forward step
        return _get_gpt_forward_step()


def _get_gpt_forward_step():
    """Get forward step for GPT-style decoder models."""
    try:
        # Try to import from pretrain_gpt
        from pretrain_gpt import forward_step  # type: ignore

        return forward_step
    except ImportError:
        # Fallback: use Megatron's default
        return None


def _get_bert_forward_step():
    """Get forward step for BERT-style encoder models."""
    try:
        from pretrain_bert import forward_step  # type: ignore

        return forward_step
    except ImportError:
        return None


def _get_t5_forward_step():
    """Get forward step for T5-style encoder-decoder models."""
    try:
        from pretrain_t5 import forward_step  # type: ignore

        return forward_step
    except ImportError:
        return None

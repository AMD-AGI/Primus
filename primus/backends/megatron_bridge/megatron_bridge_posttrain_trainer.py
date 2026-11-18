###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridgePosttrainTrainer: Primus wrapper for Megatron-Bridge post-training.

This trainer bridges Primus configuration system with Megatron-Bridge's
post-training framework. Post-training includes supervised fine-tuning (SFT),
instruction tuning, and other post-pretraining tasks.

The trainer follows the same pattern as other Megatron-Bridge trainers but is
optimized for post-training workflows with smaller datasets and specialized
training objectives.
"""

import functools
from typing import Any

from primus.backends.megatron_bridge.config_utils import load_recipe_config
from primus.backends.megatron_bridge.megatron_bridge_base_trainer import (
    MegatronBridgeBaseTrainer,
)
from primus.core.utils.module_utils import log_dict_aligned, log_rank_0


def _is_vlm_model(model_cfg: Any) -> bool:
    # Ministral3 and Nemotron-VL have no vision_config, but every VLM provider
    # (and no text-only one) defines freeze_vision_model.
    return getattr(model_cfg, "vision_config", None) is not None or hasattr(model_cfg, "freeze_vision_model")


class _DefaultAttentionMask:
    """Adds ``attention_mask=None`` to batches that omit it.

    gpt_step.get_batch indexes ``batch["attention_mask"]``, but the chat SFT collate
    (with the default get_attention_mask_from_fusion=True) leaves the key out.
    """

    def __init__(self, data_iterator):
        self._data_iterator = data_iterator

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._data_iterator)
        if isinstance(batch, dict):
            batch.setdefault("attention_mask", None)
        return batch


def _with_default_attention_mask(forward_step):
    @functools.wraps(forward_step)
    def wrapped(state, data_iterator, model, *args, **kwargs):
        if data_iterator is not None:
            data_iterator = _DefaultAttentionMask(data_iterator)
        return forward_step(state, data_iterator, model, *args, **kwargs)

    return wrapped


def _select_forward_step(model_cfg: Any):
    # vlm_step pads tokens without updating cu_seqlens, which breaks packed text-only batches.
    if _is_vlm_model(model_cfg):
        from megatron.bridge.training.vlm_step import forward_step

        return forward_step
    from megatron.bridge.training.gpt_step import forward_step

    return _with_default_attention_mask(forward_step)


class MegatronBridgePosttrainTrainer(MegatronBridgeBaseTrainer):
    """
    Trainer class for Megatron-Bridge post-training (SFT, instruction tuning, etc.).

    This trainer handles:
        - Supervised fine-tuning (SFT) workflows
        - Instruction tuning with specialized datasets
        - Recipe-based configuration for post-training
        - HuggingFace model conversion (bidirectional)
        - Integration with Megatron-Core training infrastructure
        - Support for multiple model architectures (Llama, GPT, Mistral, etc.)
        - LoRA and other parameter-efficient fine-tuning methods

    Inherits from MegatronBridgeBaseTrainer which provides:
        - Common Megatron-Bridge initialization and logging
        - Version detection
        - Unified training workflow and patch management
    """

    # Task type identifier for logging
    TASK_TYPE = "Post-training (SFT/Instruction Tuning)"

    def __init__(self, backend_args: Any = None, **kwargs):
        """
        Initialize Megatron-Bridge posttrain trainer.

        Args:
            backend_args: Megatron-Bridge argument namespace (from MegatronBridgeArgBuilder)
            **kwargs: Runtime context kwargs forwarded to BaseTrainer for filtering.
        """
        # Initialize MegatronBridgeBaseTrainer (which initializes BaseTrainer)
        super().__init__(backend_args=backend_args, **kwargs)

    def setup(self):
        """
        Setup phase for Megatron-Bridge post-training.
        """
        log_rank_0("MegatronBridgePosttrainTrainer.setup()")

    def init(self):
        """
        Initialize Megatron-Bridge post-training components.

        This includes:
            - Loading pretrained model checkpoint
            - Model initialization (with or without recipe)
            - LoRA/PEFT adapter initialization if configured
            - Optimizer setup (often with lower learning rate)
            - Data pipeline initialization for instruction/SFT data
            - Distributed training setup
        """
        log_rank_0("Initializing Megatron-Bridge post-training components...")

        self.cfg_container = load_recipe_config(self.backend_args)
        self._apply_nested_overrides()

        log_rank_0("Post-training initialization completed")

    def train(self):
        """
        Execute Megatron-Bridge post-training (SFT, instruction tuning).

        This method is called by BaseTrainer.run() after applying patches.
        It executes the main fine-tuning loop using Megatron-Bridge's infrastructure.

        Post-training typically involves:
            - Loading pretrained checkpoint
            - Training on instruction/SFT dataset
            - Regular evaluation on validation set
            - Saving fine-tuned checkpoints
            - Optional conversion to HuggingFace format
        """
        log_rank_0("Executing Megatron-Bridge post-train...")

        try:
            # Execute post-training based on configuration
            from megatron.bridge.training.config import runtime_config_update
            from megatron.bridge.training.finetune import finetune

            forward_step = _select_forward_step(self.cfg_container.model)
            log_rank_0(f"Using forward step: {forward_step.__module__}.forward_step")

            # log_rank_0(f"ConfigContainer: {self.cfg_container}")
            runtime_config_update(self.cfg_container)

            log_dict_aligned("ConfigContainer", self.cfg_container.to_dict())
            finetune(self.cfg_container, forward_step_func=forward_step)

        except Exception as e:
            log_rank_0(f"Error during post-training: {e}")
            raise

        log_rank_0("Megatron-Bridge post-train execution completed.")

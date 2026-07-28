# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Base diffusion module with Megatron-Core integration.

This module provides common infrastructure for all diffusion models (Flux, DiT, etc.),
following the architecture pattern established by LanguageModule in Megatron-LM.
"""

import os
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none


class DiffusionModule(MegatronModule):
    """
    Base diffusion module with Megatron-Core integration.

    Provides common infrastructure for all diffusion models (Flux, DiT, MovieGen, etc.):
    - Process group management (TP, PP, CP, DP)
    - Attention backend configuration
    - Distributed checkpointing support
    - Encoder management (VAE, T5, CLIP, etc.)
    - Common loss computation utilities

    This class follows the architecture pattern of LanguageModule from Megatron-LM,
    adapted for diffusion model requirements.

    Args:
        config (TransformerConfig): Transformer config with diffusion-specific parameters
        pg_collection (Optional[ProcessGroupCollection]): Model communication process groups.
            Defaults to None (uses MPU process groups).
        encoder_configs (Optional[Dict[str, Any]]): Optional encoder configurations.
            Defaults to None.
            Format: {'encoder_name': EncoderConfig, ...}
            Example: {'vae': VAEConfig(...), 't5': T5XXLConfig(...), 'clip': CLIPLConfig(...)}

    Example:
        >>> from primus.backends.megatron.core.models.diffusion.flux import FluxConfig
        >>> config = FluxConfig.flux_12b()
        >>> model = Flux(config=config)

    Reference:
        Megatron-LM LanguageModule: megatron/core/models/common/language_module/language_module.py
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        encoder_configs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config=config)

        # Configure attention backend
        self._set_attention_backend()

        # Setup process groups for distributed training
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.tp_group = get_tensor_model_parallel_group_if_none(pg_collection.tp)
        self.pp_group = pg_collection.pp

        # Note: Diffusion models typically don't need embd_group since they don't share
        # embeddings across pipeline stages like language models do

        # Virtual pipeline parallelism
        self.vp_stage = None
        self.vp_size = self.config.virtual_pipeline_model_parallel_size

        # Encoder management (VAE, T5, CLIP, etc.)
        self.encoder_configs = encoder_configs or {}
        self.encoders = nn.ModuleDict()

    def _set_attention_backend(self):
        """
        Set attention backend for TransformerEngine.

        Configures environment variables to control which attention implementation
        TransformerEngine uses. Options:
        - flash: Flash Attention (fastest, requires Ampere+ GPUs)
        - fused: Fused attention kernels
        - unfused: Standard unfused attention
        - auto: Let TE choose automatically
        - local: Disable TE attention, use PyTorch native

        TransformerEngine works on opt-out basis. By default all three attention
        backend flags are set to 1. If the user chooses a particular attention
        backend, we set the other two to 0. If the user chooses local, we set
        all 3 TE env variables to 0.

        Reference:
            LanguageModule._set_attention_backend() in Megatron-LM
        """

        def check_and_set_env_variable(
            env_variable_name: str, expected_value: int, attn_type: AttnBackend
        ) -> None:
            current_value = os.getenv(env_variable_name)
            if current_value is not None and current_value != str(expected_value):
                raise ValueError(
                    f"{env_variable_name} set to {current_value}, but expected {expected_value} "
                    f"for attention backend type {attn_type.name}. Unset NVTE_FLASH_ATTN, "
                    f"NVTE_FUSED_ATTN and NVTE_UNFUSED_ATTN. Use the --attention-backend argument "
                    f"if you want to choose between (flash/fused/unfused/auto/local). Default is auto."
                )
            os.environ[env_variable_name] = str(expected_value)

        if self.config.attention_backend == AttnBackend.local:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 0, AttnBackend.local)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 0, AttnBackend.local)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 0, AttnBackend.local)
        elif self.config.attention_backend == AttnBackend.flash:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 1, AttnBackend.flash)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 0, AttnBackend.flash)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 0, AttnBackend.flash)
        elif self.config.attention_backend == AttnBackend.fused:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 0, AttnBackend.fused)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 1, AttnBackend.fused)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 0, AttnBackend.fused)
        elif self.config.attention_backend == AttnBackend.unfused:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 0, AttnBackend.unfused)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 0, AttnBackend.unfused)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 1, AttnBackend.unfused)
        elif self.config.attention_backend == AttnBackend.auto:
            check_and_set_env_variable("NVTE_FLASH_ATTN", 1, AttnBackend.auto)
            check_and_set_env_variable("NVTE_FUSED_ATTN", 1, AttnBackend.auto)
            check_and_set_env_variable("NVTE_UNFUSED_ATTN", 1, AttnBackend.auto)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass through diffusion model.

        This method must be implemented by all subclasses to define the
        model's forward computation.

        Returns:
            Model prediction (noise, velocity, or other target type depending on model)
        """

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """
        Generate sharded state dictionary for distributed checkpointing.

        Subclasses should override this to handle model-specific checkpointing logic
        (e.g., weight tying, custom sharding strategies).

        Args:
            prefix: Prefix for state dict keys (e.g., 'module.')
            sharded_offsets: Pipeline parallel offsets
            metadata: Optional metadata for checkpoint conversion

        Returns:
            Dictionary mapping state dict keys to ShardedTensor objects

        Reference:
            LanguageModule.sharded_state_dict() in Megatron-LM
        """
        return super().sharded_state_dict(prefix, sharded_offsets, metadata)

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegatronSFTTrainer: Megatron-LM based supervised fine-tuning trainer."""

from typing import Any

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronSFTTrainer(MegatronBaseTrainer):
    """
    Trainer class for Megatron-LM based supervised fine-tuning.
    
    This trainer handles:
        - SFT workflows with HuggingFace datasets
        - Instruction tuning with proper loss masking
        - Multiple conversation formats (extensible)
        - Direct Megatron-LM integration
        - Support for various model architectures (Llama, GPT, etc.)
        - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    
    Inherits from MegatronBaseTrainer which provides:
        - Argument injection into Megatron runtime
        - ROCm compatibility patches
        - Common Megatron initialization patterns
    """

    def __init__(self, backend_args: Any):
        """
        Initialize Megatron SFT trainer.

        Args:
            backend_args: Megatron-LM argument namespace (from MegatronArgBuilder)
        """
        super().__init__(backend_args=backend_args)

        # Initialize LoRA if enabled
        self.peft = None
        self._init_lora()

        self.model_type = getattr(self.backend_args, "model_type", "gpt")
        log_rank_0(f"Initialized MegatronSFTTrainer for model_type: {self.model_type}")
    
    def _init_lora(self):
        """Initialize LoRA (Low-Rank Adaptation) if enabled in config."""
        lora_config = getattr(self.backend_args, "lora", None)

        if lora_config is None or not getattr(lora_config, "enabled", False):
            log_rank_0("LoRA disabled, using full fine-tuning")
            return

        from primus.backends.megatron.peft import LoRA

        # Convert config to dict, excluding 'enabled' field
        lora_kwargs = {k: v for k, v in vars(lora_config).items() if k != "enabled"}

        log_rank_0(f"Initializing LoRA with config: {lora_kwargs}")

        self.peft = LoRA(**lora_kwargs)

    def setup(self):
        """
        Setup phase for Megatron SFT training.

        Can be used for pre-initialization setup if needed.
        """
        super().setup()
        log_rank_0("MegatronSFTTrainer.setup()")

    def init(self):
        """
        Initialize Megatron SFT training components.

        Note:
            Argument injection is handled during setup().
            This method can be used for trainer-specific initialization.
        """
        super().init()
        log_rank_0(f"Initializing Megatron SFT training for model_type: {self.model_type}")

    def _create_model_provider_with_lora(self, base_model_provider):
        """
        Wrap the model provider to apply LoRA after model creation.
        
        Args:
            base_model_provider: Original model provider function
            
        Returns:
            Wrapped model provider that applies LoRA to the created model
        """
        peft = self.peft

        def model_provider_with_lora(*args, **kwargs):
            """
            Model provider that applies LoRA after model creation.

            Newer Megatron versions may pass extra keywords like `config` or
            `pg_collection`, so we forward the full call signature unchanged.
            """
            # Create the base model
            model = base_model_provider(*args, **kwargs)

            # Apply LoRA if enabled
            if peft is not None:
                log_rank_0("=" * 60)
                log_rank_0("Applying LoRA to model...")
                model = peft(model, training=True)
                
                # Log trainable parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                frozen_params = total_params - trainable_params
                
                log_rank_0(f"LoRA Summary:")
                log_rank_0(f"  - Total parameters:     {total_params:,}")
                log_rank_0(f"  - Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
                log_rank_0(f"  - Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
                log_rank_0("=" * 60)
            
            return model
        
        return model_provider_with_lora

    def train(self):
        """
        Execute Megatron SFT training.

        This method is called by the runtime-owned trainer lifecycle and executes
        the main SFT training loop using Megatron-LM's infrastructure.
        """
        log_rank_0("Executing Megatron SFT training...")

        from megatron.training import pretrain  # type: ignore

        from primus.core.utils.import_utils import get_model_provider

        from .sft.forward_step import create_sft_forward_step
        from .sft.runtime import create_sft_datasets_provider, run_sft_pretrain

        train_valid_test_datasets_provider = create_sft_datasets_provider()
        forward_step = create_sft_forward_step()

        # Keep model-provider behavior aligned with pretrain trainer.
        if self.model_type != "gpt":
            base_model_provider = get_model_provider(model_type=self.model_type)
        else:
            base_model_provider = get_model_provider()

        if self.peft is not None:
            model_provider = self._create_model_provider_with_lora(base_model_provider)
            log_rank_0("Using LoRA-enabled model provider")
        else:
            model_provider = base_model_provider

        run_sft_pretrain(
            pretrain_fn=pretrain,
            datasets_provider=train_valid_test_datasets_provider,
            model_provider=model_provider,
            forward_step=forward_step,
        )

        log_rank_0("Megatron SFT training execution completed.")

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronSFTTrainer: Primus wrapper for native Megatron-LM supervised fine-tuning.

This trainer bridges Primus configuration system with Megatron-LM's native SFT
(Supervised Fine-Tuning) capabilities. It uses the same pretrain() entry point
as pretraining but with SFT-specific dataset providers and forward steps.

Features:
    - Full fine-tuning and LoRA (Low-Rank Adaptation) support
    - SFT dataset loading (JSON/JSONL/HuggingFace datasets)
    - Chat template application and conversation handling
    - Answer-only loss masking for instruction tuning

The trainer inherits from MegatronBaseTrainer which handles:
    - Argument injection and Megatron runtime initialization
    - Patch management (before_train, after_train) via template method
    - Common Megatron setup patterns
"""

from typing import Any, Optional

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.backends.megatron.sft.lora_utils import LoraConfig
from primus.modules.module_utils import log_rank_0


class MegatronSFTTrainer(MegatronBaseTrainer):
    """
    Trainer class for native Megatron-LM supervised fine-tuning (SFT).

    This trainer handles:
        - Supervised fine-tuning workflows using native Megatron-LM
        - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
        - SFT dataset loading (JSON/JSONL/HuggingFace datasets)
        - Chat template application and conversation handling
        - Answer-only loss masking for instruction tuning
        - Integration with Megatron-LM's pretrain() infrastructure

    Inherits from MegatronBaseTrainer which handles:
        - Argument injection into Megatron runtime
        - Patch management via template method pattern
        - Common Megatron initialization patterns
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize Megatron SFT trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-LM argument namespace (from MegatronArgBuilder)
        """
        # Initialize base class (handles argument injection)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # Training components (will be set during training)
        self.model = None
        self.optimizer = None
        self.opt_param_scheduler = None

        # Parse LoRA configuration
        self.lora_config = self._parse_lora_config()

        log_rank_0(f"MegatronSFTTrainer initialized for model: {module_config.model or 'custom'}")
        if self.lora_config.enabled:
            log_rank_0(f"  LoRA enabled: r={self.lora_config.r}, alpha={self.lora_config.alpha}")

    def _parse_lora_config(self) -> LoraConfig:
        """
        Parse LoRA configuration from backend_args.

        Returns:
            LoraConfig instance
        """
        args = self.backend_args

        # Check if LoRA is enabled
        lora_enabled = getattr(args, "lora_enable", False) or getattr(args, "peft", None) == "lora"

        if not lora_enabled:
            return LoraConfig(enabled=False)

        # Build LoRA config from arguments
        return LoraConfig(
            enabled=True,
            r=getattr(args, "lora_r", 16),
            alpha=getattr(args, "lora_alpha", 32),
            dropout=getattr(args, "lora_dropout", 0.05),
            target_modules=getattr(
                args,
                "lora_target_modules",
                ["query", "key", "value", "dense", "linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            ),
            bias=getattr(args, "lora_bias", "none"),
        )

    def setup(self):
        """
        Setup phase for SFT training.

        Can be used for pre-initialization setup if needed, such as:
            - Validating SFT-specific configuration
            - Setting up data paths
            - Checking pretrained checkpoint availability
        """
        log_rank_0("MegatronSFTTrainer.setup()")

        # Validate SFT-specific configuration
        self._validate_sft_config()

    def _validate_sft_config(self):
        """
        Validate SFT-specific configuration requirements.

        SFT typically requires:
            - A pretrained checkpoint (load path)
            - Training data path (JSON/JSONL format or HF dataset)
            - HuggingFace tokenizer (for chat template support)
        """
        args = self.backend_args

        # Check for pretrained checkpoint
        if hasattr(args, "load") and args.load:
            log_rank_0(f"  Pretrained checkpoint: {args.load}")
        else:
            log_rank_0("  ⚠️  No pretrained checkpoint specified (--load)")

        # Check for training data
        if hasattr(args, "train_data_path") and args.train_data_path:
            log_rank_0(f"  Training data: {args.train_data_path}")
        elif hasattr(args, "finetune_hf_dataset") and args.finetune_hf_dataset:
            log_rank_0(f"  HuggingFace dataset: {args.finetune_hf_dataset}")
        else:
            log_rank_0("  ⚠️  No training data specified (--train-data-path or --finetune-hf-dataset)")

        # Check tokenizer type (SFT requires HuggingFaceTokenizer for chat templates)
        if hasattr(args, "tokenizer_type"):
            if args.tokenizer_type != "HuggingFaceTokenizer":
                log_rank_0(
                    f"  ⚠️  SFT typically requires HuggingFaceTokenizer, "
                    f"but got: {args.tokenizer_type}"
                )

        # Log LoRA configuration
        if self.lora_config.enabled:
            log_rank_0("  LoRA Configuration:")
            log_rank_0(f"    - Rank (r): {self.lora_config.r}")
            log_rank_0(f"    - Alpha: {self.lora_config.alpha}")
            log_rank_0(f"    - Dropout: {self.lora_config.dropout}")
            log_rank_0(f"    - Target modules: {self.lora_config.target_modules}")

    def init(self):
        """
        Initialize Megatron SFT training components.

        Note:
            Argument injection is already done by MegatronBaseTrainer.__init__()
            This method can be used for SFT-specific initialization.
        """
        log_rank_0("Initializing Megatron SFT training...")
        log_rank_0(f"  Model: {self.module_config.model or 'custom'}")
        log_rank_0(f"  Framework: {self.module_config.framework}")
        log_rank_0(f"  Task: Supervised Fine-Tuning (SFT)")
        if self.lora_config.enabled:
            log_rank_0(f"  Mode: LoRA (r={self.lora_config.r}, alpha={self.lora_config.alpha})")
        else:
            log_rank_0("  Mode: Full Fine-tuning")

    def run_train(self):
        """
        Execute Megatron SFT using the standard Megatron calling pattern.

        This method is called by MegatronBaseTrainer.run() after applying patches.
        It uses pretrain() with SFT-specific dataset provider and forward step.

        The SFT workflow:
            1. Load pretrained checkpoint
            2. Apply LoRA if enabled
            3. Initialize SFT dataset (with chat templates, packing, etc.)
            4. Train with answer-only loss masking
            5. Save fine-tuned checkpoints
        """
        log_rank_0("Executing Megatron SFT...")

        import inspect

        # Import Megatron components
        from megatron.core.enums import ModelType
        from megatron.training import pretrain  # type: ignore

        # Import SFT-specific components
        from primus.backends.megatron.sft.sft_utils import (
            forward_step,
            train_valid_test_sft_datasets_provider,
        )

        # Get model provider (with LoRA support if enabled)
        model_provider = self._get_model_provider_with_lora()

        # Megatron versions differ:
        # - v0.12.0: calls `pretrain(...)` directly (no inprocess_restart wrapper).
        # - newer: wraps pretrain via inprocess_restart and may pass `store=...`.
        wrapped_pretrain = pretrain
        store = None
        try:
            from megatron.training import inprocess_restart  # type: ignore

            if hasattr(inprocess_restart, "maybe_wrap_for_inprocess_restart"):
                wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
        except Exception:
            pass

        # Execute SFT training
        sig = inspect.signature(wrapped_pretrain)
        kwargs = {}

        # SFT requires HuggingFaceTokenizer for chat template support
        if "args_defaults" in sig.parameters:
            kwargs["args_defaults"] = {"tokenizer_type": "HuggingFaceTokenizer"}
        if "extra_args_provider" in sig.parameters:
            kwargs["extra_args_provider"] = self._add_sft_args
        if "store" in sig.parameters:
            kwargs["store"] = store

        wrapped_pretrain(
            train_valid_test_sft_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            **kwargs,
        )

        log_rank_0("Megatron SFT execution completed.")

    def _get_model_provider_with_lora(self):
        """
        Get model provider function that applies LoRA if enabled.

        Returns:
            Model provider function compatible with Megatron's pretrain()
        """
        from primus.core.utils.import_utils import get_model_provider

        if not self.lora_config.enabled:
            # No LoRA, use standard model provider
            return get_model_provider()

        # Create LoRA-aware model provider
        base_model_provider = get_model_provider()
        lora_config = self.lora_config

        def lora_model_provider(pre_process=True, post_process=True):
            """Model provider that applies LoRA after model creation."""
            from primus.backends.megatron.sft.lora_utils import apply_lora_to_model

            # Create base model
            model = base_model_provider(pre_process=pre_process, post_process=post_process)

            # Apply LoRA
            model, wrapped_modules = apply_lora_to_model(model, lora_config)

            return model

        return lora_model_provider

    def _add_sft_args(self, parser):
        """
        Add SFT and LoRA arguments to Megatron's argument parser.

        Args:
            parser: argparse.ArgumentParser instance

        Returns:
            Modified parser with SFT/LoRA arguments added
        """
        # SFT arguments
        sft_group = parser.add_argument_group(title="SFT")

        sft_group.add_argument(
            "--finetune-hf-dataset",
            type=str,
            default=None,
            help="HuggingFace dataset name for SFT (e.g., Open-Orca/OpenOrca)",
        )

        sft_group.add_argument(
            "--train-data-path",
            nargs="*",
            default=None,
            help="Path to training data file(s) in JSON/JSONL format",
        )

        sft_group.add_argument(
            "--valid-data-path",
            nargs="*",
            default=None,
            help="Path to validation data file(s) in JSON/JSONL format",
        )

        sft_group.add_argument(
            "--test-data-path",
            nargs="*",
            default=None,
            help="Path to test data file(s) in JSON/JSONL format",
        )

        # LoRA arguments
        lora_group = parser.add_argument_group(title="LoRA")

        lora_group.add_argument(
            "--lora-enable",
            action="store_true",
            default=False,
            help="Enable LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning",
        )

        lora_group.add_argument(
            "--peft",
            type=str,
            default=None,
            choices=["none", "lora"],
            help="PEFT method to use ('none' or 'lora')",
        )

        lora_group.add_argument(
            "--lora-r",
            type=int,
            default=16,
            help="LoRA rank (dimension of low-rank matrices)",
        )

        lora_group.add_argument(
            "--lora-alpha",
            type=int,
            default=32,
            help="LoRA scaling factor (alpha/r is the actual scaling)",
        )

        lora_group.add_argument(
            "--lora-dropout",
            type=float,
            default=0.05,
            help="Dropout probability for LoRA layers",
        )

        lora_group.add_argument(
            "--lora-target-modules",
            nargs="*",
            default=None,
            help="Module name patterns to apply LoRA (e.g., query value dense)",
        )

        lora_group.add_argument(
            "--lora-bias",
            type=str,
            default="none",
            choices=["none", "all", "lora_only"],
            help="Bias configuration for LoRA",
        )

        return parser

    def cleanup(self, on_error: bool = False):
        """
        Cleanup after SFT training.

        Args:
            on_error: Whether cleanup is being called due to an error
        """
        if on_error:
            log_rank_0("MegatronSFTTrainer cleanup (on error)")
        else:
            log_rank_0("MegatronSFTTrainer cleanup")

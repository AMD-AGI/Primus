###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronPretrainTrainer(MegatronBaseTrainer):
    """Trainer for Megatron-LM pre-training."""

    def train(self):
        """Execute Megatron pre-training."""
        log_rank_0("Executing Megatron pretrain...")

        import inspect

        from megatron.core.enums import ModelType
        from megatron.training import pretrain  # type: ignore
        from pretrain_gpt import (  # type: ignore
            forward_step,
            train_valid_test_datasets_provider,
        )

        from primus.core.utils.import_utils import get_model_provider

        if hasattr(train_valid_test_datasets_provider, "is_distributed"):
            train_valid_test_datasets_provider.is_distributed = True

        # Handle Megatron version differences (v0.12.0 vs newer with inprocess_restart)
        wrapped_pretrain = pretrain
        store = None
        try:
            from megatron.training import inprocess_restart  # type: ignore

            if hasattr(inprocess_restart, "maybe_wrap_for_inprocess_restart"):
                wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
        except Exception:
            pass

        sig = inspect.signature(wrapped_pretrain)
        kwargs = {}
        if "args_defaults" in sig.parameters:
            kwargs["args_defaults"] = {"tokenizer_type": "GPT2BPETokenizer"}
        if "extra_args_provider" in sig.parameters:
            kwargs["extra_args_provider"] = None
        if "store" in sig.parameters:
            kwargs["store"] = store

        wrapped_pretrain(
            train_valid_test_datasets_provider,
            get_model_provider(),
            ModelType.encoder_or_decoder,
            forward_step,
            **kwargs,
        )

        log_rank_0("Megatron pretrain execution completed.")

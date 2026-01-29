###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer
from primus.core.backend.backend_registry import BackendRegistry

# Register megatron backend (pretrain)
BackendRegistry.register_path_name("megatron", "Megatron-LM")
BackendRegistry.register_adapter("megatron", MegatronAdapter)
BackendRegistry.register_trainer_class("megatron", MegatronPretrainTrainer)

# Register megatron_sft backend (supervised fine-tuning)
# Uses the same adapter as megatron but with SFT-specific trainer
BackendRegistry.register_path_name("megatron_sft", "Megatron-LM")
BackendRegistry.register_adapter("megatron_sft", MegatronAdapter)
BackendRegistry.register_trainer_class("megatron_sft", MegatronSFTTrainer)

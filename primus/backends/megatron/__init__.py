###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_path_name("megatron", "Megatron-LM")

# Register adapter
BackendRegistry.register_adapter("megatron", MegatronAdapter)

# Register default trainer (pretrain)
BackendRegistry.register_trainer_class("megatron", MegatronPretrainTrainer)

# Register additional trainers with specific names
BackendRegistry.register_trainer_class("megatron_sft", MegatronSFTTrainer)

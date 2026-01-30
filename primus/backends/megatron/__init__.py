###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.megatron_adapter import MegatronAdapter
from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer
from primus.core.backend.backend_registry import BackendRegistry

# Register megatron backend
BackendRegistry.register_path_name("megatron", "Megatron-LM")
BackendRegistry.register_adapter("megatron", MegatronAdapter)

# Register trainer classes by stage
# Usage: framework=megatron with stage: pretrain / stage: sft in config
BackendRegistry.register_trainer_class(MegatronPretrainTrainer, "megatron")
BackendRegistry.register_trainer_class(MegatronSFTTrainer, "megatron", "sft")

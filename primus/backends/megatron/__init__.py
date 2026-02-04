###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.megatron_adapter import MegatronAdapter

# from primus.backends.megatron.megatron_pretrain_trainer import MegatronPretrainTrainer
from primus.core.backend.backend_registry import BackendRegistry

# Register megatron backend
# BackendRegistry.register_path_name("megatron", "Megatron-LM")
BackendRegistry.register_adapter("megatron", MegatronAdapter)

# Register trainer classes by stage
# Usage: framework=megatron with module name "pre_trainer" or "sft_trainer"
# Or explicitly set stage in config: stage: pretrain / stage: sft
# BackendRegistry.register_trainer_class(MegatronPretrainTrainer, "megatron")

# Note: To add SFT support, import and register the SFT trainer:
# from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer
# BackendRegistry.register_trainer_class(MegatronSFTTrainer, "megatron", stage="sft")

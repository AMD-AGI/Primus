###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron_bridge.megatron_bridge_adapter import (
    MegatronBridgeAdapter,
)
from primus.backends.megatron_bridge.megatron_bridge_posttrain_trainer import (
    MegatronBridgePosttrainTrainer,
)
from primus.core.backend.backend_registry import BackendRegistry

# Register adapter
BackendRegistry.register_adapter("megatron_bridge", MegatronBridgeAdapter)

# Register posttrain trainer as the default trainer
# Megatron-Bridge is designed for post-training tasks (SFT, instruction tuning, LoRA)
# BackendRegistry.register_trainer_class(MegatronBridgePosttrainTrainer, "megatron_bridge", "sft")

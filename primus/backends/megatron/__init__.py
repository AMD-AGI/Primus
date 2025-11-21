###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.megatron.adapters.megatron_adapter import MegatronAdapter
from primus.backends.megatron.trainers import MegatronPretrainTrainer
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_path_name("megatron", "Megatron-LM")

# Register adapter
BackendRegistry.register_adapter("megatron", MegatronAdapter)

# Register trainer
BackendRegistry.register_trainer_class("megatron", MegatronPretrainTrainer)

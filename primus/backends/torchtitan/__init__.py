###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.torchtitan.torchtitan_adapter import TorchTitanAdapter
from primus.backends.torchtitan.torchtitan_pretrain_trainer import (
    TorchTitanPretrainTrainer,
)
from primus.core.backend.backend_registry import BackendRegistry

# Register backend path name
BackendRegistry.register_path_name("torchtitan", "torchtitan")

# Register adapter
BackendRegistry.register_adapter("torchtitan", TorchTitanAdapter)

# Register trainer
BackendRegistry.register_trainer_class("torchtitan", TorchTitanPretrainTrainer)

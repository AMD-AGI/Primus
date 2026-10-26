###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.automodel.automodel_adapter import AutoModelAdapter
from primus.backends.automodel.automodel_pretrain_trainer import (
    AutoModelPretrainTrainer,
)
from primus.core.backend.backend_registry import BackendRegistry

# Register adapter
BackendRegistry.register_adapter("automodel", AutoModelAdapter)

# Register trainer
BackendRegistry.register_trainer_class(AutoModelPretrainTrainer, "automodel")

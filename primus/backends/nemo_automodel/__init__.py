###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.nemo_automodel.nemo_automodel_adapter import NemoAutomodelAdapter
from primus.backends.nemo_automodel.nemo_automodel_pretrain_trainer import (
    NemoAutomodelPretrainTrainer,
)
from primus.core.backend.backend_registry import BackendRegistry

# Register adapter
BackendRegistry.register_adapter("nemo_automodel", NemoAutomodelAdapter)

# Register trainer
BackendRegistry.register_trainer_class(NemoAutomodelPretrainTrainer, "nemo_automodel")

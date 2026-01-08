###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Backend Registration

Register MaxText backend adapter and trainer classes.
"""

from primus.backends.maxtext.maxtext_adapter import MaxTextAdapter
from primus.backends.maxtext.maxtext_pretrain_trainer import MaxTextPretrainTrainer
from primus.core.backend.backend_registry import BackendRegistry

# Register MaxText backend adapter
BackendRegistry.register_backend("maxtext", MaxTextAdapter)

# Register MaxText pretrain trainer
BackendRegistry.register_trainer("maxtext", MaxTextPretrainTrainer)

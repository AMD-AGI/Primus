###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.backends.hummingbirdxt.hummingbirdxt_adapter import HummingbirdXTAdapter
from primus.backends.hummingbirdxt.hummingbirdxt_posttrain_trainer import (
    HummingbirdXTPosttrainTrainer,
)
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_path_name("hummingbirdxt", "HummingbirdXT")

# Register adapter
BackendRegistry.register_adapter("hummingbirdxt", HummingbirdXTAdapter)

# Register posttrain trainer as the default trainer
# HummingbirdXT is designed for self-forcing DMD post-training tasks based on the Wan Models
# blog : https://rocm.blogs.amd.com/artificial-intelligence/hummingbirdxt/README.html
# code : https://github.com/AMD-AGI/HummingbirdXT
BackendRegistry.register_trainer_class("hummingbirdxt", HummingbirdXTPosttrainTrainer)

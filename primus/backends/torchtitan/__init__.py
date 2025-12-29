###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan backend integration for Primus.

This package wires TorchTitan components (models, protocols, Primus-Turbo
extensions) into the Primus stack.

On import, it also triggers registration of TorchTitan-specific patches so
that they can be managed via the unified Primus patch system.
"""

# IMPORTANT: Patch TorchTitan's logger BEFORE importing any TorchTitan modules
# This ensures TorchTitan uses a named logger instead of root logger,
# which allows proper source file/line tracking in Primus logs.
from primus.backends.torchtitan.patches.logger_patch import patch_torchtitan_logger

patch_torchtitan_logger()

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

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
import logging

import torchtitan.tools.logging as titan_logging

# Replace TorchTitan's root logger with a properly-named logger
# This allows InterceptHandler to extract correct module names
titan_logging.logger = logging.getLogger("torchtitan")

# Disable TorchTitan's logger initialization (Primus manages logging)
titan_logging.init_logger = lambda: None

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

# Trigger registration of all TorchTitan patches (classic attention, etc.)
import primus.backends.torchtitan.patches  # noqa: F401

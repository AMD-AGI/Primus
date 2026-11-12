###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus-owned vision backend for multi-view geometry models."""

from primus.backends.vision.vision_adapter import VisionAdapter
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_adapter("vision", VisionAdapter)

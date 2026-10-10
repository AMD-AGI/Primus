###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus backend for HunyuanWorld-Mirror training and post-training."""

from primus.backends.worldmirror.worldmirror_adapter import WorldMirrorAdapter
from primus.core.backend.backend_registry import BackendRegistry

BackendRegistry.register_adapter("worldmirror", WorldMirrorAdapter)

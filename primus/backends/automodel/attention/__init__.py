###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Attention backend overrides for the AutoModel diffusion path.

Each override is a module of its own that supplies a kernel to the shared
rebinding machinery in ``_backend_registry.py``. No re-exports here: adding an
override means adding a file, not editing this one.
"""

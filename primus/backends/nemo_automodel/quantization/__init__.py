###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Low-precision linear swaps for the AutoModel diffusion path.

Each precision is a module of its own that registers itself with the selector in
``_common.py``. No re-exports and no list of contents here: adding a precision
means adding a file, not editing this one.
"""

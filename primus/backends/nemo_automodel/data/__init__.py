###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Model-agnostic data plumbing for the AutoModel diffusion backend.

Currently just the cache-builder registry that ``primus data automodel-cache``
dispatches through. The per-model builders themselves live with their model, in
``models/<name>/data/``.
"""

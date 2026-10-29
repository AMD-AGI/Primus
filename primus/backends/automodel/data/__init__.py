###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Model-agnostic data plumbing for the AutoModel diffusion backend.

Currently just the registry that ``primus data automodel-cache`` dispatches
through. The per-model builders live with their model, in ``models/<name>/data/``,
because what a cache contains is a property of the model that reads it.
"""

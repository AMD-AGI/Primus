###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Model-agnostic distributed/sharding hooks for the AutoModel diffusion backend.

Mirrors ``nemo_automodel.components.distributed`` on the Primus side. Code here
applies to every diffusion model, so it must not import anything under
``models/``. Re-exports nothing on purpose: hooks are imported lazily by dotted
path so a missing optional dependency degrades to a skipped hook.
"""

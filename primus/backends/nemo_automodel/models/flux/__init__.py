###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""FLUX-specific hooks for the AutoModel diffusion backend.

``parallelize.py`` registers a FLUX parallelization strategy (real activation
checkpointing + FSDP2 sharding), replacing the default strategy whose AC path is
a silent no-op for FLUX blocks.
"""

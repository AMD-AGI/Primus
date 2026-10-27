###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark Open text backbone for the Primus Megatron backend.

The model is upstream :class:`GPTModel` with Muse-specific layer, attention,
router and MoE subclasses swapped into the stock GPT decoder spec. Import
``muse_spark_builders`` for the training entry point. This package init stays
empty so the math tests do not import Megatron.
"""

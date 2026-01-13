# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hybrid Mamba+MLA layer specifications for Megatron-LM."""

from .hybrid_mamba_mla_layer_specs import hybrid_inference_stack_spec, hybrid_stack_spec

__all__ = ["hybrid_stack_spec", "hybrid_inference_stack_spec"]


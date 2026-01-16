###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo Patches Module

This package contains all PrimusTurbo backend related patches for Megatron.
Each patch is organized in its own file for better maintainability.

Patches included:
  - te_spec_provider_patches: Replace TESpecProvider with PrimusTurboSpecProvider
  - gpt_output_layer_patches: Replace GPT ColumnParallelLinear with PrimusTurbo implementation
  - moe_dispatcher_patches: Replace MoE token dispatcher with PrimusTurbo DeepEP implementation
  - rms_norm_patches: Replace RMSNorm with PrimusTurbo implementation

Patch modules are discovered and imported automatically by
``primus.backends.megatron.patches``; no explicit imports are required here.
"""

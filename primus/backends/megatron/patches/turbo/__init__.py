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
  - turbo_attn_hd128_te_fallback_patches: On gfx942, route known-bad attention shapes
    (MHA hd128/hd256, MLA hd128) to TEDotProductAttention at spec-build time, avoiding the
    aiter FMHA v3/CK backward crash; other shapes keep PrimusTurboAttention. Off by default;
    CI enables it via PRIMUS_TURBO_ATTN_HD128_FALLBACK_TE=1. Temporary workaround (ROCm/aiter#1332)

Patch modules are discovered and imported automatically by
``primus.backends.megatron.patches``; no explicit imports are required here.
"""

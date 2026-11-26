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
  - aiter_deepbind_patches: Install the aiter mha RTLD_DEEPBIND isolation hook (gfx942/gfx950)
    so the Turbo attention backward binds the pinned aiter::mha_bwd, not TE's stale libmha
  - dense_mlp_fp4_patches: Route dense SwiGLU MLP through FlyDSL dense MXFP4 mlp_fp4
  - fused_qkv_rope_patches: Route Megatron fused packed-QKV RoPE through FlyDSL fused_qkv_rope
  - qk_rmsnorm_rope_patches: Fuse GPT-OSS packed QKV, Q/K RMSNorm and RoPE with FlyDSL

Patch modules are discovered and imported automatically by
``primus.backends.megatron.patches``; no explicit imports are required here.
"""

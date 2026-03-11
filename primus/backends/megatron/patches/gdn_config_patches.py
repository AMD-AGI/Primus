###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
GatedDeltaNet / KimiDeltaAttention Configuration Patches

Monkey-patch TransformerConfig with linear-attention fields required by
GatedDeltaNet and KimiDeltaAttention layers, so that no changes are needed
in the third-party Megatron-LM codebase.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

_GDN_CONFIG_FIELDS = {
    "linear_conv_kernel_dim": None,
    "linear_key_head_dim": None,
    "linear_value_head_dim": None,
    "linear_num_key_heads": None,
    "linear_num_value_heads": None,
}


def _has_any_gdn_field(args) -> bool:
    return any(
        getattr(args, name, None) is not None for name in _GDN_CONFIG_FIELDS
    )


@register_patch(
    "megatron.transformer.gdn_config",
    backend="megatron",
    phase="before_train",
    description=(
        "Monkey-patch TransformerConfig with linear-attention fields "
        "(linear_conv_kernel_dim, linear_key_head_dim, etc.) required by "
        "GatedDeltaNet and KimiDeltaAttention without modifying third-party code."
    ),
    condition=lambda ctx: _has_any_gdn_field(get_args(ctx)),
)
def patch_gdn_config(ctx: PatchContext):
    args = get_args(ctx)

    import megatron.core.transformer.transformer_config as config_mod

    for field_name, default in _GDN_CONFIG_FIELDS.items():
        value = getattr(args, field_name, default)
        setattr(config_mod.TransformerConfig, field_name, value)
        log_rank_0(
            f"[Patch:megatron.transformer.gdn_config] "
            f"TransformerConfig.{field_name} = {value}"
        )

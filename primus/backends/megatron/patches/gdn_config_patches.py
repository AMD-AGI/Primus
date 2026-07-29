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

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_GDN_CONFIG_FIELDS = {
    "linear_conv_kernel_dim": None,
    "use_short_conv": True,
    "linear_key_head_dim": None,
    "linear_value_head_dim": None,
    "linear_num_key_heads": None,
    "linear_num_value_heads": None,
    "use_fla_triton_kda": False,
    "use_fla_triton_kda_hybrid": False,
    # When True (default) and use_fla_triton_kda is also True, chunk_kda is
    # called with use_gate_in_kernel=True (gate fused inside the Triton
    # kernel). Set to False to materialize the gate up-front via
    # fused_kda_gate() — bit-identical to FLA's pre-fusion path and to the
    # tw006-validated numerics (loss=4.7281 @ iter 500 vs FLA/8=4.7350).
    "use_fla_kda_in_kernel_gate": True,
    # When True (default when use_fla_triton_kda=True), the output norm is
    # replaced by fla.modules.FusedRMSNormGated (RMSNorm + sigmoid-gate +
    # multiply in one Triton kernel). Set to False to use the unfused
    # _apply_gated_norm path with explicit fp32 sigmoid (tw006 numerics).
    "use_fla_fused_norm_gated": None,
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
        "(linear_conv_kernel_dim, linear_key_head_dim, etc.) and FLA Triton flags "
        "required by GatedDeltaNet and KimiDeltaAttention without modifying third-party code."
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


# -----------------------------------------------------------------------------
# Hybrid-model output-layer init method
# -----------------------------------------------------------------------------
#
# Upstream TransformerConfig.__post_init__ scales the output-layer init std
# by depth (scaled_init_method_normal) or by mu-P width/depth
# (mup_scaled_init_method_normal) -- both appropriate for pure transformers,
# but not for hybrid models (GDN/KDA/Mamba), which FLA initializes with a
# plain uniform `initializer_range` (init_method_normal, no depth scaling).
# Without this, hybrid runs start from a different output-layer init than
# FLA's reference training and the early loss curve doesn't line up.

_HYBRID_INIT_PATCH_KEY = "megatron.transformer.hybrid_output_init"


def _is_hybrid_model_run(args) -> bool:
    """True for any HybridStack-based model (GDN/KDA/Mamba, pure or mixed
    with MLA). Primus YAMLs set ``is_hybrid_model`` directly (it's a plain
    ``TransformerConfig`` field, generically copied from ``args`` by
    ``core_transformer_config_from_args``); fall back to the
    ``hybrid_override_pattern`` / ``hybrid_layer_pattern`` derivation
    upstream uses (see ``megatron.training.utils.is_hybrid_model``) in case
    a config relies on that instead."""
    return bool(
        getattr(args, "is_hybrid_model", False)
        or getattr(args, "hybrid_override_pattern", None)
        or getattr(args, "hybrid_layer_pattern", None)
    )


def _install_hybrid_output_init_patch() -> None:
    import megatron.core.transformer.transformer_config as config_mod
    from megatron.core.utils import init_method_normal

    TransformerConfig = config_mod.TransformerConfig
    if is_patched(TransformerConfig, _HYBRID_INIT_PATCH_KEY):
        log_rank_0(f"[Patch:{_HYBRID_INIT_PATCH_KEY}] TransformerConfig already patched; skipping.")
        return

    original_post_init = TransformerConfig.__post_init__

    def patched_post_init(self):
        # Only overriding output_layer_init_method's own None-guard mirrors
        # upstream exactly: if the caller already set it explicitly, leave
        # it untouched (same as upstream's `if self.output_layer_init_method
        # is None:` gate).
        was_none = self.output_layer_init_method is None
        original_post_init(self)
        if was_none and getattr(self, "is_hybrid_model", False):
            self.output_layer_init_method = init_method_normal(self.init_method_std)

    TransformerConfig.__post_init__ = patched_post_init
    mark_patched(TransformerConfig, _HYBRID_INIT_PATCH_KEY)
    log_rank_0(
        f"[Patch:{_HYBRID_INIT_PATCH_KEY}] Patched TransformerConfig.__post_init__: "
        "hybrid models now use uniform init_method_normal for the output layer "
        "(matching FLA's initializer_range) instead of depth/mu-P-scaled init."
    )


@register_patch(
    _HYBRID_INIT_PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description=(
        "Hybrid models (GDN, KDA, Mamba) use uniform init_method_normal for the "
        "output layer, matching FLA's initializer_range, instead of the "
        "depth/mu-P-scaled init appropriate only for pure transformers."
    ),
    condition=lambda ctx: _is_hybrid_model_run(get_args(ctx)),
)
def patch_hybrid_output_init(ctx: PatchContext):
    _install_hybrid_output_init_patch()

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Argument checks for the Granite builder (pure Python, no ``megatron.training`` import)."""

__all__ = ["reject_unsupported_args"]


def reject_unsupported_args(args):
    """Refuse upstream ``gpt_builder`` options the dense Granite builder does not implement.

    Raises ``ValueError`` naming every option that is set. Primus Turbo attention
    (``enable_primus_turbo`` + ``use_turbo_attention``) is allowed: it receives
    ``config.softmax_scale`` (= ``attention_multiplier``) like TE attention, and the
    builder reads the scale back from the built attention modules and refuses a mismatch.
    """
    unsupported = {
        "use_legacy_models": bool(getattr(args, "use_legacy_models", False)),
        "yaml_cfg": getattr(args, "yaml_cfg", None) is not None,
        "experimental_attention_variant": getattr(args, "experimental_attention_variant", None) is not None,
        "heterogeneous_layers_config_path": getattr(args, "heterogeneous_layers_config_path", None)
        is not None,
        "transformer_impl=inference_optimized": getattr(args, "transformer_impl", None)
        == "inference_optimized",
        "num_experts": bool(getattr(args, "num_experts", None)),
        "mtp_num_layers": getattr(args, "mtp_num_layers", None) is not None,
        "multi_latent_attention": bool(getattr(args, "multi_latent_attention", False)),
        # Local DotProductAttention then stores softmax_scale / layer_number and multiplies
        # the layer number back inside the softmax, so the stored value is not the scale
        # applied; the builder's read-back of the explicit attention_multiplier scale
        # (softmax_scale) would be wrong, so it is refused.
        "apply_query_key_layer_scaling": bool(getattr(args, "apply_query_key_layer_scaling", False)),
        # Granite 4.1 uses plain RoPE (HF rope_scaling: null; the converter refuses any
        # rope_scaling). Upstream's use_rope_scaling is Llama-3.x scaling (factor 8.0 by
        # default) and would make the RoPE differ from HF without an error.
        "use_rope_scaling": bool(getattr(args, "use_rope_scaling", False)),
        # Granite has no learned position embeddings; anything other than RoPE (or unset)
        # builds a different model.
        "position_embedding_type!=rope": getattr(args, "position_embedding_type", "rope")
        not in (None, "rope"),
    }
    bad = [name for name, is_set in unsupported.items() if is_set]
    if bad:
        raise ValueError(f"the dense Granite builder does not support: {', '.join(bad)}")

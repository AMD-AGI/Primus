###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron GPT Decoder Layer Specs Patches

Patch Megatron's get_gpt_decoder_layer_specs to use Primus implementation.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.gpt.decoder_layer_specs",
    backend="megatron",
    phase="before_train",
    description=(
        "Monkey patch get_gpt_decoder_layer_specs to use Primus implementation "
        "when lfm_layer_types is configured for LFM model."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "lfm_layer_types", None) is not None,
    priority=42,  # must patch after te_spec_provider_patches
)
def patch_gpt_decoder_layer_specs(ctx: PatchContext):
    """
    Patch Megatron GPT decoder layer spec builder to use Primus implementation.
    """
    import megatron.core.extensions as meg_ext

    assert (
        meg_ext.transformer_engine.HAVE_TE
    ), "patch_gpt_decoder_layer_specs patch failed, can't find transformer_engine"

    from megatron.core.models.gpt import gpt_layer_specs as megatron_gpt_layer_specs

    from primus.backends.megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_layer_specs as primus_get_gpt_decoder_layer_specs,
    )

    megatron_gpt_layer_specs.get_gpt_decoder_layer_specs = primus_get_gpt_decoder_layer_specs
    log_rank_0(
        "[Patch:megatron.gpt.decoder_layer_specs]   Patched "
        "megatron.core.models.gpt.gpt_layer_specs.get_gpt_decoder_layer_specs "
        f"-> {primus_get_gpt_decoder_layer_specs.__name__}"
    )

    # gpt_builders imports get_gpt_decoder_layer_specs directly; force import and
    # patch its local binding.
    try:
        import gpt_builders as gpt_builders_module  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        log_rank_0(
            "[Patch:megatron.gpt.decoder_layer_specs]   Failed to import gpt_builders; "
            "cannot patch local get_gpt_decoder_layer_specs binding."
        )
        raise RuntimeError("Failed to import required module gpt_builders") from exc

    gpt_builders_module.get_gpt_decoder_layer_specs = primus_get_gpt_decoder_layer_specs
    log_rank_0(
        "[Patch:megatron.gpt.decoder_layer_specs]   Patched "
        "gpt_builders.get_gpt_decoder_layer_specs "
        f"-> {primus_get_gpt_decoder_layer_specs.__name__}"
    )

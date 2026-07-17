###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan GPT-OSS Primus-Turbo Sink Attention Patch
====================================================

GPT-OSS uses FlexAttention with learnable per-head attention sinks and a
sliding-window mask on even layers. The module-form ``TurboAttention`` used by
the other models' turbo path does NOT expose ``sink`` / ``window_size``, so it
cannot model GPT-OSS attention. However, the Primus-Turbo *functional* kernel
``primus_turbo.pytorch.ops.flash_attn_func`` does support both (this is the same
API the Megatron ``PrimusTurboAttention`` uses for GPT-OSS sink attention).

This setup patch swaps the upstream GPT-OSS ``Attention`` and ``TransformerBlock``
classes for Primus mirrors that route attention through ``flash_attn_func`` with
``sink=self.sinks`` and the per-layer ``window_size``. It is gated on
``primus_turbo.enable_primus_turbo`` + ``primus_turbo.use_turbo_attention`` and
only applies to ``gpt_oss`` runs, so existing GPT-OSS FlexAttention configs
(turbo off) are unaffected.
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.core.utils.module_utils import log_rank_0


def _gptoss_turbo_enabled(ctx: PatchContext) -> bool:
    return (
        get_param(ctx, "model.name", None) == "gpt_oss"
        and get_param(ctx, "primus_turbo.enable_primus_turbo", False)
        and get_param(ctx, "primus_turbo.use_turbo_attention", False)
    )


@register_patch(
    "torchtitan.primus_turbo.gptoss_sink_attention",
    backend="torchtitan",
    phase="setup",
    description="Use Primus-Turbo functional flash_attn_func (sink + sliding window) for GPT-OSS",
    condition=_gptoss_turbo_enabled,
)
def patch_gptoss_sink_attention(ctx: PatchContext) -> None:
    """Install the Primus-Turbo sink-attention mirror for GPT-OSS."""
    log_rank_0(
        "[Patch:torchtitan.primus_turbo.gptoss_sink_attention] "
        "Enabling Primus-Turbo sink attention (flash_attn_func) for GPT-OSS...",
    )

    import torchtitan.models.gpt_oss.model.model as gptoss_model_mod

    from primus.backends.torchtitan.models.gpt_oss.model.model import (
        Attention as GptOssTurboAttention,
    )
    from primus.backends.torchtitan.models.gpt_oss.model.model import (
        TransformerBlock as GptOssTurboBlock,
    )

    # Replace the module-level classes so GptOssModel builds the mirror instances
    # (the mirror TransformerBlock also constructs the mirror Attention, and both
    # resolve these names from this module's globals at build time).
    gptoss_model_mod.Attention = GptOssTurboAttention
    gptoss_model_mod.TransformerBlock = GptOssTurboBlock

    log_rank_0(
        "[Patch:torchtitan.primus_turbo.gptoss_sink_attention] "
        "GPT-OSS Primus-Turbo sink attention successfully installed.",
    )

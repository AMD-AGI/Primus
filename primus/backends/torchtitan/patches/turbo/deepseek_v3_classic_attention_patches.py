###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan DeepSeek-V3 Classic Attention Patch (Primus-Turbo extension)

This patch switches DeepSeek-V3 to use the "classic" attention implementation
and argument structure when ``primus_turbo.use_classic_attention`` is enabled
in the TorchTitan job config.

The original logic lived inside ``TorchTitanPretrainTrainer.patch_classic_attention``.
It is now expressed as a backend patch so it can be managed via the Primus
patch system.
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "torchtitan.primus_turbo.deepseek_v3_classic_attention",
    backend="torchtitan",
    phase="setup",
    description="Use classic DeepSeek-V3 attention and args when requested",
    condition=lambda ctx: (
        get_param(ctx, "primus_turbo.enable_primus_turbo", False)
        and get_param(ctx, "primus_turbo.use_classic_attention", False)
    ),
)
def patch_deepseek_v3_classic_attention(ctx: PatchContext) -> None:
    """
    Monkey patch DeepSeek-V3 to use the classic attention implementation.
    """
    log_rank_0(
        "[Patch:torchtitan.primus_turbo.deepseek_v3_classic_attention] "
        "Enabling classic DeepSeek-V3 attention...",
    )

    import torchtitan.models.deepseek_v3

    from primus.backends.torchtitan.models.deepseek_v3 import classic_deepseekv3_args
    from primus.backends.torchtitan.models.deepseek_v3.model.args import (
        DeepSeekV3ClassicModelArgs,
    )
    from primus.backends.torchtitan.models.deepseek_v3.model.model import (
        MultiHeadAttention,
    )

    # Swap argument helpers to the classic variants
    torchtitan.models.deepseek_v3.deepseekv3_args = classic_deepseekv3_args
    torchtitan.models.deepseek_v3.DeepSeekV3ModelArgs = DeepSeekV3ClassicModelArgs

    # Swap the Attention class inside the model module
    import torchtitan.models.deepseek_v3.model.model as ds_model_mod

    ds_model_mod.Attention = MultiHeadAttention

    log_rank_0(
        "[Patch:torchtitan.primus_turbo.deepseek_v3_classic_attention] "
        "DeepSeek-V3 classic Attention successfully installed."
    )

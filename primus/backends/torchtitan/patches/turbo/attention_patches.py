###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Primus-Turbo Attention Patch

This patch switches selected models (LLaMA3, LLaMA4, DeepSeek-V3) to use the
Primus-Turbo Attention implementation when ``primus_turbo.use_turbo_attention``
is enabled in the TorchTitan job config.

The original logic lives inside ``TorchTitanPretrainTrainer``. It is now also
expressed as a backend patch so it can be managed via the Primus patch system.
"""

from primus.backends.torchtitan.patches.turbo.utils import get_primus_turbo_config
from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "torchtitan.primus_turbo.turbo_attention",
    backend="torchtitan",
    phase="before_train",
    description="Use Primus-Turbo Attention kernels for supported models",
    condition=lambda ctx: (
        (cfg := get_primus_turbo_config(ctx)) is not None
        and getattr(cfg, "enable_primus_turbo", False)
        and getattr(cfg, "use_turbo_attention", False)
    ),
)
def patch_turbo_attention(ctx: PatchContext) -> None:
    """
    Monkey patch LLaMA3, LLaMA4 and DeepSeek-V3 to use Primus-Turbo Attention.
    """
    from primus.core.utils.logger import _logger as primus_logger

    primus_logger.info(
        "[Patch:torchtitan.primus_turbo.turbo_attention] "
        "Enabling Primus-Turbo Attention for LLaMA3/LLaMA4/DeepSeek-V3..."
    )

    # ******* LLaMA3 Attention Model *******
    import torchtitan.models.llama3.model.model

    from primus.backends.torchtitan.models.llama3.model.model import (
        Attention as Llama3Attention,
    )

    torchtitan.models.llama3.model.model.Attention = Llama3Attention

    # ******* LLaMA4 Attention Model *******
    import torchtitan.models.llama4.model.model

    from primus.backends.torchtitan.models.llama4.model.model import (
        Attention as Llama4Attention,
    )

    torchtitan.models.llama4.model.model.Attention = Llama4Attention

    # ******* DeepSeek-V3 Attention Model *******
    import torchtitan.models.deepseek_v3.model.model

    from primus.backends.torchtitan.models.deepseek_v3.model.model import (
        Attention as DeepSeekV3Attention,
    )

    torchtitan.models.deepseek_v3.model.model.Attention = DeepSeekV3Attention

    log_rank_0(
        "[Patch:torchtitan.primus_turbo.turbo_attention] "
        "Primus-Turbo Attention successfully installed "
        "for LLaMA3, LLaMA4 and DeepSeek-V3.",
    )

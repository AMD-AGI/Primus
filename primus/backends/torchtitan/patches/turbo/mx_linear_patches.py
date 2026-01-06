###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Primus-Turbo MXLinear Patch

This patch switches the MXLinear implementation and model converter to the
Primus-Turbo versions when ``primus_turbo.use_turbo_mx_linear`` is enabled
in the TorchTitan job config.

The original logic lives inside ``TorchTitanPretrainTrainer``. It is now also
expressed as a backend patch so it can be managed via the Primus patch system.
"""

from primus.backends.torchtitan.patches.turbo.utils import get_primus_turbo_config
from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "torchtitan.primus_turbo.turbo_mx_linear",
    backend="torchtitan",
    phase="before_train",
    description="Use Primus-Turbo MXLinear and model converter",
    condition=lambda ctx: (
        (cfg := get_primus_turbo_config(ctx)) is not None
        and getattr(cfg, "enable_primus_turbo", False)
        and getattr(cfg, "use_turbo_mx_linear", False)
    ),
)
def patch_turbo_mx_linear(ctx: PatchContext) -> None:
    """
    Monkey patch MXLinear and its converter to use Primus-Turbo implementations.
    """
    from primus.core.utils.logger import _logger as primus_logger

    primus_logger.info(
        "[Patch:torchtitan.primus_turbo.turbo_mx_linear] "
        "Enabling Primus-Turbo MXLinear and model converter..."
    )

    # ******* MXLinear *******
    import torchtitan.components.quantization.mx
    from torchtitan.protocols.model_converter import _registry_model_converter_cls

    from primus.backends.torchtitan.components.quantization.mx import (
        PrimusTubroMXConverter,
    )

    _registry_model_converter_cls["mx"] = PrimusTubroMXConverter
    torchtitan.components.quantization.mx.MXLinearConverter = PrimusTubroMXConverter

    log_rank_0(
        "[Patch:torchtitan.primus_turbo.turbo_mx_linear] " "Primus-Turbo MXLinear successfully installed.",
    )

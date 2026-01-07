###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Primus-Turbo FP8Linear Patch

This patch switches the FP8Linear implementation and model converter to the
Primus-Turbo versions when ``primus_turbo.use_turbo_float8_linear`` is
enabled in the TorchTitan job config.

The original logic lives inside ``TorchTitanPretrainTrainer``. It is now also
expressed as a backend patch so it can be managed via the Primus patch system.
"""

from primus.core.patches import PatchContext, get_param, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "torchtitan.primus_turbo.turbo_float8_linear",
    backend="torchtitan",
    phase="before_train",
    description="Use Primus-Turbo FP8Linear and model converter",
    condition=lambda ctx: (
        get_param(ctx, "primus_turbo.enable_primus_turbo", False)
        and get_param(ctx, "primus_turbo.use_turbo_float8_linear", False)
    ),
)
def patch_turbo_fp8_linear(ctx: PatchContext) -> None:
    """
    Monkey patch FP8Linear and its converter to use Primus-Turbo implementations.
    """
    from primus.core.utils.logger import _logger as primus_logger

    primus_logger.info(
        "[Patch:torchtitan.primus_turbo.turbo_float8_linear] "
        "Enabling Primus-Turbo FP8Linear and model converter...",
    )

    # ******* FP8Linear *******
    import torchtitan.components.quantization.float8
    from torchtitan.protocols.model_converter import _registry_model_converter_cls

    from primus.backends.torchtitan.components.quantization.float8 import (
        PrimusTubroFP8Converter,
    )

    _registry_model_converter_cls["turbo_fp8_linear"] = PrimusTubroFP8Converter
    torchtitan.components.quantization.float8.Float8LinearConverter = PrimusTubroFP8Converter

    log_rank_0(
        "[Patch:torchtitan.primus_turbo.turbo_float8_linear] "
        "Primus-Turbo FP8Linear successfully installed.",
    )

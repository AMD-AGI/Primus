###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.nn as nn

# Compatibility for different primus_turbo versions
try:
    from primus_turbo.pytorch.core.float8 import Float8QuantConfig, ScalingGranularity
except ImportError:
    from primus_turbo.pytorch.core.low_precision import Float8QuantConfig, ScalingGranularity

from primus_turbo.pytorch.modules.linear_fp8 import Float8Linear
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger


def module_filter_fn(mod: nn.Module, fqn: str, filter_fqns: list[str]) -> bool:
    """
    Filter function to determine which modules should be converted.
    For both Float8 and MXFP8, we only convert Linear modules
    with dimensions divisible by 16 and not matching any filtered FQNs.
    """
    if not isinstance(mod, nn.Linear):
        return False

    # All dims must be divisible by 16 due to float8 tensorcore hardware requirements.
    dims_multiples_of_128 = mod.weight.shape[0] % 128 == 0 and mod.weight.shape[1] % 128 == 0

    # If the fqn matches any filtered fqn, then we should not convert this module.
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)

    return dims_multiples_of_128 and not is_filtered_fqn


def replace_turbo_fp8linear_modules(model: nn.Module, config: Float8QuantConfig):
    filter_fqns = ["gate", "output"]
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and not isinstance(module, Float8Linear):
            if module_filter_fn(module, name, filter_fqns):
                fp8_linear = Float8Linear.from_float(module, config)
                logger.info(f"module {name} shape {module.weight.shape}, replaced to FP8Linear")
                setattr(model, name, fp8_linear)
            else:
                logger.info(f"module {name} shape {module.weight.shape}, cannot be replaced to FP8Linear")
        else:
            replace_turbo_fp8linear_modules(module, config)


class PrimusTubroFP8Converter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = True
        self.config = Float8QuantConfig(granularity=ScalingGranularity.TENSORWISE)

    def convert(self, model: nn.Module):
        if not self.enabled:
            return

        replace_turbo_fp8linear_modules(model, self.config)

        logger.info("Swapped to FP8Linear layers")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        FP8 doesn't require any post-optimizer hooks at the moment
        """
        return


register_model_converter(PrimusTubroFP8Converter, "primus_turbo_fp8")

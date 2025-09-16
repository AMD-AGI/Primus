###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.nn as nn
from primus_turbo.pytorch.core.float8 import MXQuantConfig
from primus_turbo.pytorch.modules import MXLinear
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)
from torchtitan.tools.logging import logger


def replace_turbo_mxlinear_modules(model: nn.Module, config: MXQuantConfig):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and not isinstance(module, MXLinear):
            mx_linear = MXLinear.from_float(module, config)
            setattr(model, name, mx_linear)
        else:
            replace_turbo_mxlinear_modules(module, config)


class PrimusTubroMXConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = True
        # TODO: quant config
        self.config = MXQuantConfig()

    def convert(self, model: nn.Module):
        if not self.enabled:
            return

        replace_turbo_mxlinear_modules(model, self.config)

        logger.info("Swapped to MXLinear layers")

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        """
        MXFP8 doesn't require any post-optimizer hooks at the moment
        """
        return


register_model_converter(PrimusTubroMXConverter, "primus_turbo_mx")

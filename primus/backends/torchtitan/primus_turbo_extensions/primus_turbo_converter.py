###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.attention import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
)
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)


def replace_turbo_attention_modules(model: torch.nn.Module, fp8_config):
    from primus_turbo.pytorch.modules import TurboAttention  # TODO: import Check

    for name, module in model.named_children():
        if isinstance(module, (FlexAttentionWrapper, ScaledDotProductAttentionWrapper)):
            setattr(
                model,
                name,
                TurboAttention(causal=True, fp8_config=fp8_config),
            )
        else:
            replace_turbo_attention_modules(module, fp8_config)


class PrimusTurboConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.primus_turbo_config = job_config.primus_turbo
        # Model configs list this converter unconditionally, but TurboAttention only matches
        # the (q, k, v) Attention.forward installed by the turbo_attention patch, which is
        # gated on the same two flags.
        self.enabled = bool(
            self.primus_turbo_config.enable_primus_turbo and self.primus_turbo_config.use_turbo_attention
        )
        self.fp8_config = None
        if self.enabled and self.primus_turbo_config.enable_attention_float8:
            from primus_turbo.pytorch.core.low_precision import (
                Float8QuantConfig,
                ScalingGranularity,
            )

            self.fp8_config = Float8QuantConfig(
                granularity=ScalingGranularity.BLOCKWISE,
                block_size=64,
            )

    def convert(self, model: torch.nn.Module):
        if not self.enabled:
            return model

        replace_turbo_attention_modules(model, self.fp8_config)
        return model

    def post_optimizer_hook(self, model: torch.nn.Module | list[torch.nn.Module]):
        return


register_model_converter(PrimusTurboConverter, "primus_turbo")

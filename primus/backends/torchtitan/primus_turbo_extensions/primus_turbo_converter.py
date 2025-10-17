###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.attention import FlexAttention, ScaledDotProductAttention
from torchtitan.protocols.model_converter import (
    ModelConverter,
    register_model_converter,
)


def replace_turbo_attention_modules(model: torch.nn.Module, backend_type: str, use_fp8: bool):
    from primus_turbo.pytorch.modules import TurboAttention  # TODO: import Check

    for name, module in model.named_children():
        if isinstance(module, (FlexAttention, ScaledDotProductAttention)):
            setattr(
                model,
                name,
                TurboAttention(causal=True, backend_type=backend_type, use_fp8=use_fp8),
            )
        else:
            replace_turbo_attention_modules(module, backend_type, use_fp8)


def set_turbo_config_on_modules(model: torch.nn.Module, config: object):
    """
    Set Primus Turbo configuration flags on Attention and FeedForward modules.
    This allows the patched modules to access the config and conditionally use turbo features.
    """
    from torchtitan.models.llama3.model import Attention, FeedForward
    
    for module in model.modules():
        if isinstance(module, (Attention, FeedForward)):
            # Set config flags as module attributes
            module.use_turbo_fp8_gemm = config.use_turbo_fp8_gemm
            module.use_turbo_attention = config.use_turbo_attention


class PrimusTubroConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.enabled = True
        self.primus_turbo_config = job_config.primus_turbo
        self.enabled_attn_fp8 = self.primus_turbo_config.enable_attention_float8
        self.attn_backend_type = "triton" if self.enabled_attn_fp8 else "ck"

    def convert(self, model: torch.nn.Module):
        if self.enabled == False:
            return

        replace_turbo_attention_modules(model, self.attn_backend_type, self.enabled_attn_fp8)
        
        # Set config flags on Attention and FeedForward modules
        set_turbo_config_on_modules(model, self.primus_turbo_config)
        
        return model

    def post_optimizer_hook(self, model: torch.nn.Module | list[torch.nn.Module]):
        return


register_model_converter(PrimusTubroConverter, "primus_turbo")

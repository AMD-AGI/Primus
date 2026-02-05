###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import primus_turbo.pytorch as turbo
import torch
import torch.nn.functional as F
import torchtitan.models.gpt_oss.model.moe
from torchtitan.models.gpt_oss.model.moe import swiglu, ScaleBiasForward


try:
    from primus_turbo.pytorch.core.float8 import (
        Float8QuantConfig,
        Format,
        ScalingGranularity,
    )
except ImportError:
    from primus_turbo.pytorch.core.low_precision import (
        Float8QuantConfig,
        Format,
        ScalingGranularity,
    )


def _run_experts_grouped_mm(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
    use_fp8: bool = True,
) -> torch.Tensor:
    """
    Primus Turbo implementation of grouped_mm for GPT-OSS MoE.

    """
    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64).to(x.device)
    num_tokens_per_expert_long = num_tokens_per_expert.to(torch.long)
    
    if use_fp8:
        fp8_cfg = Float8QuantConfig(
            format=Format.E4M3,
            granularity=ScalingGranularity.TENSORWISE,
        )
        
        h = turbo.ops.grouped_gemm_fp8(
            x.bfloat16(), mlp1_weight.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True, config=fp8_cfg
        )
    else:
        h = turbo.ops.grouped_gemm(
            x.bfloat16(), mlp1_weight.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True
        )
    
    b1 = mlp1_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
    tail_slack = x.shape[0] - int(num_tokens_per_expert.sum())
    if tail_slack:
        b1 = torch.cat([b1, b1.new_zeros((tail_slack, b1.shape[-1]))], dim=0)
    h = h + b1.to(h.dtype)
    h = swiglu(h, limit=swiglu_limit)
    
    if use_fp8:
        h = turbo.ops.grouped_gemm_fp8(
            h, mlp2_weight.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True, config=fp8_cfg
        )
    else:
        h = turbo.ops.grouped_gemm(
            h, mlp2_weight.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True
        )
    
    b2_base = mlp2_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
    b2 = ScaleBiasForward.apply(b2_base, tp_degree)
    tail_slack = x.shape[0] - int(num_tokens_per_expert.sum())
    if tail_slack:
        b2 = torch.cat([b2, b2.new_zeros((tail_slack, b2.shape[-1]))], dim=0)
    h = h + b2.to(h.dtype)
    
    return h

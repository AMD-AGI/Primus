###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.nn.functional as F


def _primus_run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor | None,
) -> torch.Tensor:

    import primus_turbo.pytorch as pt

    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64).to(x.device)
    h = F.silu(
        pt.ops.grouped_gemm(
            x.bfloat16(), w1.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True
        )
    )
    h = h * pt.ops.grouped_gemm(
        x.bfloat16(), w3.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True
    )

    out = pt.ops.grouped_gemm(
        h, w2.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True
    ).type_as(x)

    return out


def _primus_run_experts_grouped_mm_fp8(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor | None,
) -> torch.Tensor:

    import primus_turbo.pytorch as pt
    from primus_turbo.pytorch.core.float8 import (
        Float8QuantConfig,
        Format,
        ScalingGranularity,
    )

    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64).to(x.device)
    fp8_cfg = Float8QuantConfig(
        format=Format.E4M3,
        granularity=ScalingGranularity.TENSORWISE,  # or ROWWISE ,TENSORWISE
    )
    
    h = F.silu(
        pt.ops.grouped_gemm_fp8(
            x.bfloat16(), w1.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True,
            config=fp8_cfg
        )
    )
    h = h * pt.ops.grouped_gemm_fp8(
        x.bfloat16(), w3.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True,
        config=fp8_cfg
    )

    out = pt.ops.grouped_gemm_fp8(
        h, w2.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True,
        config=fp8_cfg
    ).type_as(x)

    return out

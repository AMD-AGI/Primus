

import torch
import torch.nn.functional as F
import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    use_deepep: bool = False,
    use_fp8: bool = True,
) -> torch.Tensor:
    assert x.dim() == 2
    num_tokens_per_expert = num_tokens_per_expert.to(torch.int64).to(x.device)
    if use_fp8:
        fp8_cfg = Float8QuantConfig(
            format=Format.E4M3,
            granularity=ScalingGranularity.TENSORWISE,  # or ROWWISE ,TENSORWISE
        )
    
        h = F.silu(
            turbo.ops.grouped_gemm_fp8(
                x.bfloat16(), w1.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True,
                config=fp8_cfg
            )
        )
        h = h * turbo.ops.grouped_gemm_fp8(
            x.bfloat16(), w3.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True,
            config=fp8_cfg
        )
    
        out = turbo.ops.grouped_gemm_fp8(
            h, w2.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True,
            config=fp8_cfg
        ).type_as(x)
    else:
        h = F.silu(
            turbo.ops.grouped_gemm(
                x.bfloat16(), w1.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True
            )
        )
        h = h * turbo.ops.grouped_gemm(
            x.bfloat16(), w3.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True
        )

        out = turbo.ops.grouped_gemm(
            h, w2.bfloat16(), group_lens=num_tokens_per_expert, trans_b=True
        ).type_as(x)
 
    return out

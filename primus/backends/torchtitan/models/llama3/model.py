###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch

from torchtitan.models.llama3.model import Attention as TTAttention
from torchtitan.models.llama3.model import apply_rotary_emb
from torchtitan.models.llama3.model import FeedForward as TTFeedForward
from torch.nn import functional as F

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.core.float8 import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
)

from primus.backends.torchtitan.models.llama3.rope import TEFusedRoPEFunc

def precompute_freqs_cis_for_te(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute frequency angles for TE's FusedRoPEFunc (real-valued tensor).
    Output shape: [S, 1, 1, D]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # shape: [D/2]
    t = torch.arange(end).float()  # shape: [S]
    angles = torch.outer(t, freqs)  # shape: [S, D/2]
    freqs_full = torch.cat([angles, angles], dim=-1)  # → shape: [S, D]
    return freqs_full.view(end, 1, 1, dim).to(torch.float32)

class Attention(TTAttention):
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        bs, seqlen, _ = x.shape
        if self.use_turbo_fp8_gemm:
            fp8_cfg = Float8QuantConfig(
                format=Format.E4M3,
                granularity=ScalingGranularity.TENSORWISE,
            )
            # Reshape input from 3D (bs, seqlen, dim) to 2D (bs*seqlen, dim) for gemm_fp8
            x_2d = x.view(-1, x.size(-1))
            
            xq_2d = turbo.ops.gemm_fp8(x_2d, self.wq.weight, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=fp8_cfg)
            xk_2d = turbo.ops.gemm_fp8(x_2d, self.wk.weight, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=fp8_cfg)
            xv_2d = turbo.ops.gemm_fp8(x_2d, self.wv.weight, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=fp8_cfg)
            
            # Reshape back to 4D (bs, seqlen, n_heads, head_dim) directly
            xq = xq_2d.view(bs, seqlen, -1, self.head_dim)
            xk = xk_2d.view(bs, seqlen, -1, self.head_dim)
            xv = xv_2d.view(bs, seqlen, -1, self.head_dim)
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

            # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
            # local heads from sizes of xq, xk, and xv as TP may have sharded them
            # after the above linear ops.
            xq = xq.view(bs, seqlen, -1, self.head_dim)
            xk = xk.view(bs, seqlen, -1, self.head_dim)
            xv = xv.view(bs, seqlen, -1, self.head_dim)

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq = TEFusedRoPEFunc.apply(xq, freqs_cis)
        xk = TEFusedRoPEFunc.apply(xk, freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        # xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.view(bs, seqlen, -1)

        if self.use_turbo_fp8_gemm:
            fp8_cfg = Float8QuantConfig(
                format=Format.E4M3,
                granularity=ScalingGranularity.TENSORWISE,
            )
            # Reshape output from 3D (bs, seqlen, dim) to 2D (bs*seqlen, dim) for gemm_fp8
            output_2d = output.view(-1, output.size(-1))
            result_2d = turbo.ops.gemm_fp8(output_2d, self.wo.weight, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=fp8_cfg)
            # Reshape back to 3D (bs, seqlen, dim)
            result = result_2d.view(bs, seqlen, -1)
            return result
        else:
            return self.wo(output)

class FeedForward(TTFeedForward):
    def forward(self, x: torch.Tensor):
        if self.use_turbo_fp8_gemm:
            fp8_cfg = Float8QuantConfig(
                format=Format.E4M3,
                granularity=ScalingGranularity.TENSORWISE,
            )
            
            # Reshape input from 3D (bs, seqlen, dim) to 2D (bs*seqlen, dim) for gemm_fp8
            original_shape = x.shape
            x_2d = x.view(-1, x.size(-1))
            
            # SwiGLU: w2(silu(w1(x)) * w3(x))
            w1_output = turbo.ops.gemm_fp8(x_2d, self.w1.weight, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=fp8_cfg)
            w1_activated = F.silu(w1_output)
            w3_output = turbo.ops.gemm_fp8(x_2d, self.w3.weight, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=fp8_cfg)
            gated = w1_activated * w3_output
            output = turbo.ops.gemm_fp8(gated, self.w2.weight, trans_a=False, trans_b=True, out_dtype=torch.bfloat16, config=fp8_cfg)
            
            # Reshape output back to original 3D shape
            return output.view(original_shape)
        else:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
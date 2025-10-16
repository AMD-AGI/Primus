###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from torchtitan.models.llama3.model import Attention as TTAttention
from torchtitan.models.llama3.model import apply_rotary_emb

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
        return self.wo(output)

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from torch.nn.attention.flex_attention import BlockMask
from torchtitan.models.qwen3.model.model import Attention as TTAttention
from torchtitan.models.qwen3.model.model import apply_rotary_emb

AttentionMasksType = dict[str, BlockMask] | BlockMask


class Attention(TTAttention):
    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)

        # Primus-Turbo path: inner_attention is replaced by TurboAttention, which
        # consumes the (bs, seqlen, n_heads, head_dim) layout and handles GQA /
        # causal masking internally, so we skip the transpose and block_mask.
        output = self.inner_attention(xq, xk, xv)

        output = output.contiguous().view(bs, seqlen, -1)
        return self.wo(output)

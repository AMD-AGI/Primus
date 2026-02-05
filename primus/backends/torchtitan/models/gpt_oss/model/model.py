###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
import torch.nn as nn
from torchtitan.models.gpt_oss.model.model import (
    Attention as TTGptOssAttention, 
    repeat_kv,
    apply_rotary_emb,
)


class Attention(TTGptOssAttention):
    
    def forward(self, x, rope_cache, attention_masks):
        """
        Forward pass that handles both FlexAttention and TurboAttention.
        
        """
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.wq(x).view(hidden_shape)
        k = self.wk(x).view(hidden_shape)
        v = self.wv(x).view(hidden_shape)

        q, k = apply_rotary_emb(q, k, rope_cache)

        keys = repeat_kv(k, self.n_rep)
        values = repeat_kv(v, self.n_rep)

        xq = q.transpose(1, 2).contiguous()
        xk = keys.transpose(1, 2).contiguous()
        xv = values.transpose(1, 2).contiguous()

        if hasattr(self.inner_attention, '__class__') and 'TurboAttention' in self.inner_attention.__class__.__name__:
            # TurboAttention: doesn't accept block_mask, doesn't return lse
            output = self.inner_attention(xq, xk, xv)
            # Create dummy lse for Attention Sink (will have no effect with zeros)
            lse = torch.zeros(bsz, self.n_heads, seqlen, device=output.device, dtype=output.dtype)
        else:
            # FlexAttention: uses block_mask and returns lse
            output, lse = self.inner_attention(
                xq, xk, xv, block_mask=attention_masks, scale=None, return_lse=True
            )

        sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1)).unsqueeze(-1)
        output = output * sink_scale.to(output.dtype)

        output = output.transpose(1, 2).contiguous()

        output = output.reshape(bsz, seqlen, -1).contiguous()
        output = self.wo(output)

        return output
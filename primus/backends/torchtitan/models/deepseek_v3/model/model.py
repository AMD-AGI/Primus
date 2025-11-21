###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from torch.nn.attention.flex_attention import BlockMask
from torchtitan.models.deepseek_v3.model.model import Attention as TTAttention
from torchtitan.models.deepseek_v3.model.model import apply_rotary_emb

# Import the Attention class from llama4 as new option for DeepSeekV3
from torchtitan.models.llama4.model.model import Attention as Llama4Attention
from torchtitan.models.llama4.model.model import (
    precompute_freqs_cis as llama4_precompute_freqs_cis,
)

from .args import DeepSeekV3ClassicModelArgs

AttentionMasksType = dict[str, BlockMask] | BlockMask


class Attention(TTAttention):
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of q and kv as TP may have sharded them after
        # the above linear ops.
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)

        # Key-value projection
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # (bsz, seqlen, 1, qk_rope_head_dim)

        kv = self.wkv_b(self.kv_norm(kv))  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1
        )  # (bsz, seqlen, n_heads, qk_head_dim)

        # q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        # k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        # v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

        output = self.inner_attention(q, k, v)

        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        return self.wo(output)  # (bsz, seqlen, dim)


class MultiHeadAttention(Llama4Attention):
    """
    Multi-head attention module for DeepSeekV3, inheriting from llama4's Attention class.

    This class adapts the llama4 Attention class to work with DeepSeekV3ModelArgs
    instead of TransformerModelArgs.
    """

    def __init__(
        self,
        model_args: DeepSeekV3ClassicModelArgs,
        use_rope: bool = True,
        fixed_block_size: int | None = None,
    ):
        # Convert DeepSeekV3ModelArgs to a format compatible with llama4's Attention
        # Create a mock TransformerModelArgs-like object
        class MockTransformerModelArgs:
            def __init__(self, deepseek_args: DeepSeekV3ClassicModelArgs):
                self.n_heads = deepseek_args.q_head
                self.n_kv_heads = deepseek_args.n_kv_heads
                self.dim = deepseek_args.dim
                self.head_dim = deepseek_args.head_dim

                self.use_flex_attn = deepseek_args.use_flex_attn

        # Initialize the parent class with the mock args
        super().__init__(
            MockTransformerModelArgs(model_args), use_rope=use_rope, fixed_block_size=fixed_block_size
        )
        self.rope_theta = model_args.rope_theta

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        # Always use llama4-style freqs_cis for this attention, regardless of input
        seqlen = x.shape[1]
        freqs_llama4 = llama4_precompute_freqs_cis(self.head_dim, seqlen, self.rope_theta)
        # Ensure freqs are on the same device as activations
        freqs_llama4 = freqs_llama4.to(x.device, dtype=x.dtype)
        return super().forward(x, freqs_llama4, None)

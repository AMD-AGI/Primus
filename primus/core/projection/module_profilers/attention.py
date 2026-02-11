###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

import torch

from primus.core.projection.base_module_profiler import BaseModuleProfiler

from .utils import benchmark_layer


class AttentionProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.module = None  # Will be set during benchmarking
        self._cached_results = (
            None  # Cache for (forward_time, backward_time, activation_memory)
        )
        self._cache_key = None  # Cache key (batch_size, seq_len)

    def set_module(self, module):
        """Set the actual attention module for benchmarking."""
        self.module = module
        # Invalidate cache when module changes
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        args = self.config.model_config
        # Group-query & multi-latent attention support.
        # If GQA not enabled, fall back to per-head queries.
        num_query_groups = (
            args.num_query_groups
            if args.group_query_attention and args.num_query_groups
            else args.num_attention_heads
        )

        # Projection ratio: (kv_channels * n_heads) / hidden_size
        query_proj_to_hidden = (
            args.kv_channels * args.num_attention_heads
        ) / args.hidden_size

        if args.multi_latent_attention:
            # q_term: either dense or LoRA factored Q with RoPE/Q-norm
            if args.q_lora_rank is None:
                q_term = (
                    args.hidden_size
                    * args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                )
            else:
                q_term = args.q_lora_rank * (
                    args.hidden_size
                    + args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                    + 1
                )
            attn = (
                q_term
                # kv lora + rope + kv norm
                + args.kv_lora_rank
                * (
                    args.hidden_size
                    + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
                    + 1
                )
                # pos emb
                + args.hidden_size * args.qk_pos_emb_head_dim
                # out proj
                + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
            )
            return attn

        # Standard attention path (Q,K,V,O projections)
        return (
            2
            * args.hidden_size
            * args.hidden_size
            * (
                (1 + (num_query_groups / args.num_attention_heads))
                * query_proj_to_hidden
            )
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        args = self.config.model_config
        mp = self.config.model_parallel_config

        tp_size = max(1, mp.tensor_model_parallel_size)
        cp_size = max(1, mp.context_model_parallel_size)

        tokens_per_rank = batch_size * seq_len // tp_size // cp_size
        if tokens_per_rank == 0:
            return 0

        bytes_per_value = 2  # assume bf16 activations

        def _num_query_groups() -> int:
            if args.group_query_attention and args.num_query_groups:
                return args.num_query_groups
            return args.num_attention_heads

        ln_width = 0

        if args.multi_latent_attention:
            # MLA uses separate latent dimensions for Q/K and V plus optional LoRA ranks.
            heads = args.num_attention_heads
            q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
            v_head_dim = args.v_head_dim

            q_width = heads * q_head_dim
            k_width = q_width  # key stores the same latent + positional dims
            v_width = heads * v_head_dim
            context_width = v_width  # attention output before the final projection
            query_projection_size = q_width  # For softmax width calculation

            if args.qk_layernorm:
                ln_width += q_width
                ln_width += k_width

            activation_width = q_width + k_width + v_width + context_width
        else:
            query_projection_size = args.kv_channels * args.num_attention_heads
            kv_projection_size = args.kv_channels * _num_query_groups()

            # Need to retain Q, K, V as well as the projected context/output.
            activation_width = (
                query_projection_size + 2 * kv_projection_size + args.hidden_size
            )

            if args.qk_layernorm:
                ln_width += kv_projection_size * 2

        heads_per_partition = max(1, args.num_attention_heads // tp_size)
        seqlen_per_cp = max(1, (seq_len + cp_size - 1) // cp_size)
        if getattr(args, "use_flash_attn", False):
            softmax_width = query_projection_size
        else:
            softmax_width = heads_per_partition * seqlen_per_cp
        activation_width += softmax_width

        return tokens_per_rank * (activation_width + ln_width) * bytes_per_value

    def _get_benchmark_results(
        self, batch_size: int, seq_len: int
    ) -> tuple[float, float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len)

        if self._cached_results is None or self._cache_key != cache_key:
            # Context parallel / Sequence parallel adjustment
            cp_size = self.config.model_parallel_config.context_model_parallel_size
            # Effective sequence length per rank if CP is used
            slen_per_cp = seq_len // cp_size

            self._cached_results = benchmark_layer(
                self.module,
                [
                    (seq_len, batch_size, self.config.model_config.hidden_size),
                    ((1, 1, slen_per_cp, seq_len), torch.bool),
                ],
            )
            self._cache_key = cache_key
        return self._cached_results

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        forward_time, _, _ = self._get_benchmark_results(batch_size, seq_len)
        return forward_time

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        _, backward_time, _ = self._get_benchmark_results(batch_size, seq_len)
        return backward_time

    def measured_activation_memory(self, batch_size: int, seq_len: int) -> int:
        _, _, activation_memory = self._get_benchmark_results(batch_size, seq_len)
        return activation_memory

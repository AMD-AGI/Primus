###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.projection.base_module_profiler import BaseModuleProfiler


class AttentionProfiler(BaseModuleProfiler):
    def estimated_num_params(self, rank: int | None = None) -> int:
        args = self.config.model_config
        # Group-query & multi-latent attention support.
        # If GQA not enabled, fall back to per-head queries.
        num_query_groups = args.num_query_groups if args.group_query_attention and args.num_query_groups else args.num_attention_heads

        # Projection ratio: (kv_channels * n_heads) / hidden_size
        query_proj_to_hidden = (args.kv_channels * args.num_attention_heads) / args.hidden_size

        if args.multi_latent_attention:
            # q_term: either dense or LoRA factored Q with RoPE/Q-norm
            if args.q_lora_rank is None:
                q_term = args.hidden_size * args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
            else:
                q_term = args.q_lora_rank * (
                    args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim) + 1
                )
            attn = (
                q_term
                # kv lora + rope + kv norm
                + args.kv_lora_rank * (
                    args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim) + 1
                )
                # pos emb
                + args.hidden_size * args.qk_pos_emb_head_dim
                # out proj
                + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
            )
            return attn

        # Standard attention path (Q,K,V,O projections)
        return (
            2 * args.hidden_size * args.hidden_size *
            ((1 + (num_query_groups / args.num_attention_heads)) * query_proj_to_hidden)
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        multiplier = 4  # for Q, K, V, O
        return (batch_size * seq_len //
                self.config.model_parallel_config.tensor_model_parallel_size //
                self.config.model_parallel_config.context_model_parallel_size *
                self.config.model_config.hidden_size * multiplier * 2)  # bf16

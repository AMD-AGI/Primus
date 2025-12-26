###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FLOPs Calculation Patches

Patches for FLOPs estimation and performance profiling in Megatron.
"""

from primus.core.patches import PatchContext
from primus.modules.module_utils import log_rank_0

# ============================================================================
# FLOPs Calculator Implementation
# ============================================================================


def num_floating_point_operations(args, batch_size):
    """
    Calculate floating-point operations for Transformer and hybrid models.

    Supports:
        - Standard Transformer architectures
        - Hybrid architectures (Attention + Mamba + MLP)
        - MoE (Mixture of Experts) models
        - GQA (Grouped Query Attention)
        - SwiGLU activation

    Args:
        args: Model configuration (Megatron args namespace)
        batch_size: Training batch size

    Returns:
        Total FLOPs for forward + backward pass
    """

    def calculate_layer_counts():
        """Calculate the number of attention, Mamba, and MLP layers."""
        if args.hybrid_override_pattern:
            counts = {"M": 0, "*": 0, "-": 0}
            for layer_type in args.hybrid_override_pattern:
                if layer_type in counts:
                    counts[layer_type] += 1
            return counts["*"], counts["M"], counts["-"]
        else:
            num_attn_layers = round(args.num_layers * args.hybrid_attention_ratio)
            num_mlp_layers = round(args.num_layers * args.hybrid_mlp_ratio)
            num_mamba_layers = args.num_layers - num_attn_layers - num_mlp_layers
            return num_attn_layers, num_mamba_layers, num_mlp_layers

    def mlp_layer_flops(batch_size, seq_len, hidden_size, expansion=4.0, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        return 4 * expansion * scale_factor * batch_size * seq_len * hidden_size**2

    def attn_layer_flops(
        batch_size, seq_len, hidden_size, num_heads, gqa=True, gqa_groups=8, kv_channels=None
    ):
        """Calculate FLOPs for an attention layer."""
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups if gqa else num_heads
        return (
            4
            * batch_size
            * seq_len
            * hidden_size
            * p
            * (hidden_size + (hidden_size * (g / num_heads)) + (seq_len / 2))
        )

    def mamba_layer_flops(batch_size, seq_len, hidden_size, state_dim=16, head_dim=64, num_groups=1):
        """Calculate FLOPs for a Mamba layer."""
        # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
        # but small percent of overall layer flops
        d_in = 2 * hidden_size
        nheads = d_in // head_dim
        return (
            (
                2 * batch_size * seq_len * hidden_size * (2 * d_in + 2 * num_groups * state_dim + nheads)
            )  # in_proj
            + (7 * batch_size * seq_len * d_in * state_dim)  # scan
            + (2 * batch_size * seq_len * d_in * hidden_size)  # out_proj
        )

    def hybrid_flops(
        batch_size,
        seq_len,
        hidden_size,
        num_attn_layers,
        num_mamba_layers,
        num_mlp_layers,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        num_attn_heads=32,
        gqa=True,
        gqa_groups=8,
        kv_channels=None,
        mlp_expansion=4.0,
        swiglu=False,
        vocab_size=256000,
    ):
        """Calculate total FLOPs for the hybrid model."""
        flops_fwd = (
            num_attn_layers
            * attn_layer_flops(batch_size, seq_len, hidden_size, num_attn_heads, gqa, gqa_groups, kv_channels)
            + num_mlp_layers * mlp_layer_flops(batch_size, seq_len, hidden_size, mlp_expansion, swiglu)
            + num_mamba_layers
            * mamba_layer_flops(
                batch_size, seq_len, hidden_size, mamba_state_dim, mamba_head_dim, mamba_num_groups
            )
            + (2 * batch_size * seq_len * hidden_size * vocab_size)  # logits computation
        )
        return flops_fwd * 3

    def transformer_flops():
        """Calculate FLOPs for a standard Transformer model."""
        # TODO(helenn/dnarayanan): Refactor this to reuse the helper methods.
        # Attention projection size.
        query_projection_size = args.kv_channels * args.num_attention_heads
        query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # MoE.
        if args.num_experts is None:
            # Every Transformer MLP is dense.
            num_dense_layers = args.num_layers
            num_moe_layers = 0
            num_experts_routed_to = 0
        else:
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.moe_layer_freq, int):
                moe_layer_pattern = [
                    1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
                ]
            elif isinstance(args.moe_layer_freq, list):
                moe_layer_pattern = args.moe_layer_freq
            else:
                raise RuntimeError("Illegal --moe-layer-freq argument provided!")
            assert len(moe_layer_pattern) == args.num_layers
            # Number of 1s in `moe_layer_pattern`.
            num_moe_layers = sum(moe_layer_pattern)
            num_dense_layers = args.num_layers - num_moe_layers
            num_experts_routed_to = args.moe_router_topk

        moe_ffn_hidden_size = (
            args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
        )
        shared_expert_ffn_hidden_size = (
            0
            if args.moe_shared_expert_intermediate_size is None
            else args.moe_shared_expert_intermediate_size
        )
        # SwiGLU.
        gated_linear_multiplier = 3 / 2 if args.swiglu else 1

        # The 12x term below comes from the following factors; for more details, see
        # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
        #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
        #       in MLP layer).
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        expansion_factor = 3 * 2 * 2

        return (
            expansion_factor
            * batch_size
            * args.seq_length
            * args.num_layers
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                (
                    (
                        1
                        + (args.num_query_groups / args.num_attention_heads)
                        # Only half of the attention matrix is non-zero and needs to be multiplied with V.
                        + (args.seq_length / args.hidden_size)
                    )
                    * query_projection_to_hidden_size_ratio
                )
                # MLP.
                + (
                    (
                        # Dense.
                        (args.ffn_hidden_size * num_dense_layers)
                        +
                        # MoE.
                        (
                            (
                                # Routed experts.
                                moe_ffn_hidden_size * num_experts_routed_to
                                +
                                # Shared experts.
                                shared_expert_ffn_hidden_size
                            )
                            * num_moe_layers
                        )
                    )
                    * gated_linear_multiplier
                    / (args.num_layers * args.hidden_size)
                )
                # Logit.
                + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
            )
        )

    # Main entrypoint for FLOPs calculation.
    if args.is_hybrid_model:
        # Calculate the number of each type of layer.
        num_attn_layers, num_mamba_layers, num_mlp_layers = calculate_layer_counts()

        # Compute hybrid model FLOPs.
        return hybrid_flops(
            batch_size=batch_size,
            seq_len=args.seq_length,
            hidden_size=args.hidden_size,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            num_mlp_layers=num_mlp_layers,
            mamba_state_dim=args.mamba_state_dim,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            num_attn_heads=args.num_attention_heads,
            gqa=args.group_query_attention,
            gqa_groups=args.num_query_groups,
            kv_channels=args.kv_channels,
            mlp_expansion=args.ffn_hidden_size / args.hidden_size,
            swiglu=args.swiglu,
            vocab_size=args.padded_vocab_size,
        )
    else:
        # Compute standard Transformer model FLOPs.
        return transformer_flops()


# ============================================================================
# FLOPs Calculator Patches
# ============================================================================


# @register_patch(
#     "megatron.flops.use_primus_estimator",
#     backend="megatron",
#     phase="before_train",
#     description="Replace Megatron FLOPs calculator with Primus implementation",
# )
def use_primus_flops_estimator(ctx: PatchContext):
    """
    Replace Megatron's FLOPs calculator with Primus implementation.

    Primus's FLOPs estimator supports:
        - Standard Transformer architectures
        - Hybrid architectures (Attention + Mamba + MLP)
        - MoE (Mixture of Experts) models
        - GQA (Grouped Query Attention)
        - SwiGLU activation

    The Primus estimator provides more accurate FLOPs calculation for
    hybrid models that combine different layer types.

    Benefits:
        - Accurate FLOPs reporting for hybrid architectures
        - Better performance profiling
        - Support for custom layer combinations
    """
    import megatron.training.training as megatron_training

    # Replace Megatron's FLOPs calculator with our implementation
    megatron_training.num_floating_point_operations = num_floating_point_operations

    log_rank_0(
        "[Patch:megatron.flops.use_primus_estimator] Replaced FLOPs calculator with Primus implementation"
    )

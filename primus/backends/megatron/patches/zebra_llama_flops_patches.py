###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Zebra-Llama FLOPs Patch

Monkey-patches megatron.training.training.num_floating_point_operations so that
hybrid models using Multi-Latent Attention (MLA) get an accurate FLOPs estimate
instead of falling through to the GQA-based hybrid_flops path.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def zebra_llama_flops(args, batch_size):
    """Calculate FLOPs for the hybrid model with MLA attention."""

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

    def mamba_layer_flops(
        batch_size, seq_len, hidden_size, state_dim=16, head_dim=64, num_groups=1, num_heads=128
    ):
        """Calculate FLOPs for a Mamba layer."""
        # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
        # but small percent of overall layer flops
        d_in = 2 * hidden_size
        if num_heads:
            nheads = num_heads
        else:
            nheads = d_in // head_dim
        return (
            (
                2 * batch_size * seq_len * hidden_size * (2 * d_in + 2 * num_groups * state_dim + nheads)
            )  # in_proj
            + (7 * batch_size * seq_len * d_in * state_dim)  # scan
            + (2 * batch_size * seq_len * d_in * hidden_size)  # out_proj
        )

    def mla_attn_layer_flops(
        batch_size,
        seq_len,
        hidden_size,
        num_heads,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
    ):
        """Calculate FLOPs for an MLA (Multi-Latent Attention) layer."""
        """
        Basic arithmetic
        let B is batch size, s is seq_len, h is embedding dim,
        for one self_attnetion block (prenorm is not included)
        qkv projection:  6Bsh^2
        attn:            2Bs^2h
        attn over value: 2Bs^2h
        oproj:           2Bsh^2

        references
        https://arxiv.org/abs/2305.10403
        https://arxiv.org/abs/2205.05198
        """
        ## MLA
        if q_lora_rank is None:
            q_term = hidden_size * num_heads * (qk_head_dim + qk_pos_emb_head_dim)
        else:
            q_term = q_lora_rank * (hidden_size + num_heads * (qk_head_dim + qk_pos_emb_head_dim) + 1)
        return (
            (
                2  # FMA
                * (
                    ## q lora + rope + q norm
                    q_term
                    ## kv lora + rope + kv norm
                    + kv_lora_rank * (hidden_size + num_heads * (qk_head_dim + v_head_dim) + 1)
                    + hidden_size * qk_pos_emb_head_dim
                    ## o proj
                    + (num_heads * v_head_dim) * hidden_size
                    ## core attn
                    + seq_len * (num_heads * (qk_head_dim + qk_pos_emb_head_dim)) / 2
                    + seq_len * num_heads * v_head_dim / 2
                )
            )
            * batch_size
            * seq_len
        )

    num_attn_layers, num_mamba_layers, num_mlp_layers = calculate_layer_counts()

    flops_fwd = (
        num_attn_layers
        * mla_attn_layer_flops(
            batch_size,
            args.seq_length,
            args.hidden_size,
            args.num_attention_heads,
            args.q_lora_rank,
            args.kv_lora_rank,
            args.qk_head_dim,
            args.qk_pos_emb_head_dim,
            args.v_head_dim,
        )
        + num_mlp_layers
        * mlp_layer_flops(
            batch_size,
            args.seq_length,
            args.hidden_size,
            args.ffn_hidden_size / args.hidden_size,
            args.swiglu,
        )
        + num_mamba_layers
        * mamba_layer_flops(
            batch_size,
            args.seq_length,
            args.hidden_size,
            args.mamba_state_dim,
            args.mamba_head_dim,
            args.mamba_num_groups,
            args.mamba_num_heads,
        )
        + (2 * batch_size * args.seq_length * args.hidden_size * args.padded_vocab_size)  # logits computation
    )
    return flops_fwd * 3


# ---------------------------------------------------------------------------
# Patch registration
# ---------------------------------------------------------------------------


@register_patch(
    "megatron.training.zebra_llama_flops",
    backend="megatron",
    phase="before_train",
    description="Use MLA-aware FLOPs estimate for hybrid (Zebra-Llama) models.",
    condition=lambda ctx: (
        getattr(get_args(ctx), "is_hybrid_model", False)
        and getattr(get_args(ctx), "multi_latent_attention", False)
    ),
)
def patch_zebra_llama_flops(ctx: PatchContext):
    import megatron.training.training as orig_training

    _orig_num_flops = orig_training.num_floating_point_operations

    def _patched_num_floating_point_operations(args, batch_size):
        if getattr(args, "is_hybrid_model", False) and getattr(args, "multi_latent_attention", False):
            return zebra_llama_flops(args, batch_size)
        return _orig_num_flops(args, batch_size)

    orig_training.num_floating_point_operations = _patched_num_floating_point_operations
    log_rank_0("MegatronPatches: using zebra_llama_flops for hybrid + MLA FLOPs.")

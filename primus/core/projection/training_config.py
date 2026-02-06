###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from dataclasses import dataclass, fields


@dataclass
class RuntimeConfig:
    global_batch_size: int = 1
    micro_batch_size: int = 1
    sequence_length: int = 0
    data_parallel_size: int = 1


@dataclass
class ModelParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int = 1
    context_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    use_torch_fsdp2: bool = False
    use_distributed_optimizer: bool = False
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = False
    # Recomputation settings
    recompute_granularity: str = None  # "full" or "selective"
    recompute_num_layers: int = 0


@dataclass
class ModelConfig:
    num_layers: int = 0
    hidden_size: int = 0
    padded_vocab_size: int = 0
    ffn_hidden_size: int = 0
    # attention
    num_attention_heads: int = 0
    kv_channels: int = 0
    group_query_attention: bool = False
    num_query_groups: int = 0
    qk_layernorm: bool = False
    multi_latent_attention: bool = False
    use_flash_attn: bool = False
    qk_head_dim: int = 0
    qk_pos_emb_head_dim: int = 0
    v_head_dim: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    # FFN & MoE
    swiglu: bool = False
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_pattern: list = None
    moe_router_topk: int = 0
    moe_shared_expert_intermediate_size: int = 0
    # Misc
    share_embeddings_and_output_weights: bool = False


@dataclass
class TrainingConfig:
    """
    Configuration for training the profiler models.
    """

    model_config: ModelConfig
    runtime_config: RuntimeConfig
    model_parallel_config: ModelParallelConfig


def update_config_from_args(config, args):
    for field in fields(config):
        if hasattr(args, field.name):
            setattr(config, field.name, getattr(args, field.name))
    return config


def megatron_derive_default_args(args):
    world_size = int(os.getenv("NNODES", "1")) * int(os.getenv("GPUS_PER_NODE", "8"))
    if args.kv_channels is None:
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if not args.group_query_attention:
        # If GQA not set, treat as per-head queries
        args.num_query_groups = args.num_attention_heads

    if not hasattr(args, "data_parallel_size") or args.data_parallel_size is None:
        args.data_parallel_size = world_size // (
            args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
        )
    if not hasattr(args, "virtual_pipeline_model_parallel_size"):
        args.virtual_pipeline_model_parallel_size = None
    if (
        args.num_layers_per_virtual_pipeline_stage is None
        and args.virtual_pipeline_model_parallel_size is None
    ):
        args.virtual_pipeline_model_parallel_size = 1
    elif args.num_layers_per_virtual_pipeline_stage is not None:
        args.virtual_pipeline_model_parallel_size = args.num_layers // (
            args.num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size
        )

    args.share_embeddings_and_output_weights = not args.untie_embeddings_and_output_weights

    if args.num_experts is None:
        args.moe_pattern = [0] * args.num_layers
    else:
        if isinstance(args.moe_layer_freq, int):
            args.moe_pattern = [1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)]
        elif isinstance(args.moe_layer_freq, list):
            args.moe_pattern = args.moe_layer_freq
        elif isinstance(args.moe_layer_freq, str):
            try:
                parsed = eval(args.moe_layer_freq)
            except Exception:
                raise ValueError(f"Invalid moe_layer_freq format: {args.moe_layer_freq}")
            
            # Handle case where eval returns an int (e.g., "1" -> 1 means all layers are MoE)
            if isinstance(parsed, int):
                if parsed == 1:
                    # All layers are MoE
                    args.moe_pattern = [1] * args.num_layers
                else:
                    # Every Nth layer is MoE
                    args.moe_pattern = [1 if (i % parsed == 0) else 0 for i in range(args.num_layers)]
            elif isinstance(parsed, list):
                args.moe_pattern = parsed
                assert (
                    len(args.moe_pattern) == args.num_layers
                ), f"Invalid moe_layer_freq length: {len(args.moe_pattern)} (expected {args.num_layers})"
            else:
                raise ValueError(f"Invalid moe_layer_freq format after eval: {type(parsed)}")

    # naming conversion
    args.sequence_length = args.seq_length
    args.context_model_parallel_size = args.context_parallel_size

    # Use model's vocab size if set, otherwise default to 100352
    if not hasattr(args, 'padded_vocab_size') or args.padded_vocab_size is None:
        args.padded_vocab_size = 100352

    return args


def convert_primus_config_to_projection_config(primus_config) -> TrainingConfig:
    args = primus_config.get_module_config("pre_trainer")
    framework = getattr(args, "framework", "")
    if framework == "megatron":
        args = megatron_derive_default_args(args)
    else:
        raise NotImplementedError(f"Unsupported framework: {framework}")

    model_config = update_config_from_args(ModelConfig(), args)
    runtime_config = update_config_from_args(RuntimeConfig(), args)
    model_parallel_config = update_config_from_args(ModelParallelConfig(), args)

    training_config = TrainingConfig(
        model_config=model_config,
        runtime_config=runtime_config,
        model_parallel_config=model_parallel_config,
    )

    return training_config

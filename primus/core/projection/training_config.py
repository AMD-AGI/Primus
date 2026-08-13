###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from dataclasses import dataclass, fields
from typing import List, Optional


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
    # Pipeline stage layer distribution
    decoder_first_pipeline_num_layers: int = None
    decoder_last_pipeline_num_layers: int = None
    pipeline_model_parallel_layout: str = None
    # Recomputation settings
    recompute_granularity: str = None  # "full" or "selective"
    recompute_num_layers: int = 0
    recompute_method: str = None  # "uniform" (recompute every layer) or "block" (first N)
    # Megatron selective block recompute: global transformer layer indices (0..num_layers-1)
    recompute_layer_ids: Optional[List[int]] = None
    # Precision-aware optimizer (Megatron `--use-precision-aware-optimizer`).
    # When enabled the optimizer state dtypes follow the *_dtype fields below;
    # the projection's bytes-per-param formula uses these to size the static
    # block correctly instead of assuming default fp32 main params + fp32 m + fp32 v.
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: str = "fp32"  # fp32 | bf16 | fp16
    exp_avg_dtype: str = "fp32"  # 1st moment dtype (fp32 | bf16 | fp16)
    exp_avg_sq_dtype: str = "fp32"  # 2nd moment dtype (fp32 | bf16 | fp16)


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
    # DeepSeek-V4 custom attention (compressed/hierarchical/sliding-window)
    compress_ratios: object = None  # per-layer schedule (list or "[...]" string)
    hc_mult: int = 1  # multi-hyper-connection stream count
    index_topk: int = 0  # CSA per-query selected keys
    index_head_dim: int = 0  # indexer head dim
    index_n_heads: int = 0  # indexer heads
    attn_sliding_window: int = 0  # SWA local window
    o_lora_rank: int = 0  # output-projection LoRA rank
    o_groups: int = 1  # output-projection groups
    # DeepSeek-V4 fused-attention kernel flags.  The production run scripts flip
    # these on (run_deepseek_v4.sh / proj_dsv4.sh: --use_v4_triton_csa_attention),
    # so the projection MUST see them to know the CSA top-K gather is fused
    # in-kernel (roofline max(compute, memory)) rather than a separate additive
    # pre-pass.  update_config_from_args copies them from args by field name;
    # without these fields the CLI values were silently dropped and csa_fused
    # was stuck at False -> gather over-counted on every CSA layer.
    use_v4_triton_attention: bool = False
    use_v4_triton_csa_attention: bool = False
    use_v4_tilelang_attention: bool = False
    use_v4_tilelang_csa_attention: bool = False
    use_v4_fp8_indexer: bool = False
    # FFN & MoE
    swiglu: bool = False
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_pattern: list = None
    moe_router_topk: int = 0
    moe_shared_expert_intermediate_size: int = 0
    # Optimizer (Muon adds Newton-Schulz orthogonalization compute on 2D weights)
    optimizer: str = ""
    muon_num_ns_steps: int = 0
    # Misc
    share_embeddings_and_output_weights: bool = False
    # Precision – None means bf16, "hybrid" means FP8-hybrid (linear GEMMs in FP8)
    fp8: str = None
    # FP8 quantization recipe (e.g. "tensorwise", "blockwise", "mxfp8").  The
    # microscaled recipe ("mxfp8") runs GEMMs in MX8 compute (scale 3/3).
    fp8_recipe: str = None

    # Primus Turbo flags — used to select the grouped-GEMM performance model
    enable_primus_turbo: bool = False
    use_turbo_grouped_gemm: bool = False
    # Megatron/Primus names the turbo grouped-MLP flag ``use_turbo_grouped_mlp``;
    # carry it (and the legacy opt-in) so the profiler can pick the batched
    # (near-ideal) grouped-GEMM model instead of the pessimistic sequential one.
    use_turbo_grouped_mlp: bool = False
    moe_use_legacy_grouped_gemm: bool = False
    use_turbo_deepep: bool = False  # DeepEP enables async A2A with compute overlap
    turbo_sync_free_moe_stage: int = 0  # 0=off, 1=fused router, 2=+DeepEP+grouped, 3=+fused act

    # Loss fusion – fuses cross-entropy with output layer avoiding full logits materialisation
    cross_entropy_loss_fusion: bool = False


@dataclass
class TrainingConfig:
    """
    Configuration for training the profiler models.
    """

    model_config: ModelConfig
    runtime_config: RuntimeConfig
    model_parallel_config: ModelParallelConfig


def gemm_dtype_from_config(config) -> str:
    """Return the GEMM compute-dtype string for the model's precision recipe.

    Mirrors the reference GEMM path: FP8 with the microscaled recipe ("mxfp8")
    runs MX8 compute (scale 3/3); other FP8 recipes run FP8; otherwise BF16.
    Accepts anything exposing ``fp8`` / ``fp8_recipe`` attributes (ModelConfig
    or raw args).
    """
    if not getattr(config, "fp8", None):
        return "bf16"
    recipe = (getattr(config, "fp8_recipe", None) or "").lower()
    return "mx8" if recipe == "mxfp8" else "fp8"


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
                # Handle list-based moe_layer_freq pattern
                if len(parsed) > args.num_layers:
                    # Truncate to first num_layers elements (for proxy models with fewer layers)
                    # This is safe: we're using a subset of the pattern for faster profiling
                    args.moe_pattern = parsed[: args.num_layers]
                elif len(parsed) < args.num_layers:
                    # If the pattern is shorter than num_layers, this is likely an error
                    # (config specifies fewer layers than requested)
                    raise ValueError(
                        f"moe_layer_freq pattern has {len(parsed)} elements but num_layers={args.num_layers}. "
                        f"The pattern length must match or exceed num_layers. "
                        f"Pattern: {parsed}"
                    )
                else:
                    # Exact match - use as-is (normal case for full model)
                    args.moe_pattern = parsed
            else:
                raise ValueError(f"Invalid moe_layer_freq format after eval: {type(parsed)}")

    # naming conversion
    args.sequence_length = args.seq_length
    args.context_model_parallel_size = args.context_parallel_size

    # Use model's vocab size if set, otherwise default to 100352
    if not hasattr(args, "padded_vocab_size") or args.padded_vocab_size is None:
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

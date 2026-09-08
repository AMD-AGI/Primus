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

    # ------------------------------------------------------------------
    # Sparse-embedding + HSTU (DLRM-v4 / TorchRec ranker) fields.
    #
    # These are additive and default to the inert values used by the
    # language-model path (num_embedding_tables=0 -> no sparse arch), so an
    # LLM config is unaffected.  A DLRM workload populates them and the
    # ``torchrec_dlrm`` workload spec assembles the sparse-embedding + HSTU
    # profilers from them.
    # ------------------------------------------------------------------
    # Sparse embeddings (TorchRec/DMP EmbeddingBagCollection).
    num_embedding_tables: int = 0
    # Per-table row (cardinality) counts; if unset, ``embedding_total_rows`` is
    # split evenly across ``num_embedding_tables``.
    embedding_table_rows: object = None  # list[int] | "[...]" | None
    embedding_total_rows: int = 0
    embedding_dim: int = 0  # per-table embedding width (D); defaults to hidden_size
    # Average number of lookups per sample per table (pooling factor); a scalar
    # or a per-table list.  Drives the sparse gather byte-traffic and the
    # embedding all-to-all message size.
    embedding_pooling_factor: object = None  # int | list[int] | None
    embedding_default_pooling_factor: int = 1
    # Sharding of the embedding tables across the model-parallel mesh.
    embedding_sharding: str = "row"  # row | column | table | data
    # Stored precision of the embedding parameters (fp32 tables are common).
    embedding_param_bytes: int = 4
    # Memory tiering: fraction of embedding parameters resident in HBM; the
    # remainder lives on DDR/UVM (host) and is streamed over the host link.
    embedding_hbm_fraction: float = 1.0
    # Optimizer state carried per embedding parameter.  Sparse tables usually
    # use a *row-wise* optimizer (one fp32 scalar per row, ~4/dim bytes/param),
    # not a full per-element moment -- charging a full fp32 moment doubles the
    # dominant memory term (see get_num_bytes_per_param).
    #   rowwise_adagrad -> 1 fp32 / row     (default; Yambda uses this)
    #   adagrad         -> 1 fp32 / element
    #   adam            -> 2 fp32 / element
    embedding_optimizer: str = "rowwise_adagrad"
    # Effective fraction of peak HBM sustained by the backward embedding
    # gradient scatter-add (at::indexFuncLargeIndex).  This is an fp32 atomic
    # read-modify-write over randomly addressed rows, so it sustains only a few
    # percent of peak -- far below the (coalesced, bf16) forward gather -- and
    # is the single largest embedding kernel in DLRM-v4 traces.  It is neither a
    # GEMM nor a collective, so nothing else in the model prices it.
    embedding_grad_scatter_efficiency: float = 0.06
    # Per-GPU HBM capacity (GB) available to embedding tables.  0 = unknown
    # (embedding_hbm_fraction must then be supplied directly).
    embedding_hbm_capacity_gb: int = 0

    # HSTU (Hierarchical Sequential Transduction Unit) attention block.
    hstu_num_heads: int = 0
    hstu_qk_dim: int = 0  # per-head query/key dim (d_qk)
    hstu_v_dim: int = 0  # per-head value dim (d_v)
    hstu_max_seq_len: int = 0
    # Jagged-sequence fill factor: mean valid tokens / padded max_seq_len
    # (HSTU sequences are variable length; ~0.4 is typical for Yambda-5B).
    hstu_fill_factor: float = 1.0
    # Std-dev of the fill factor across the batch.  Attention cost goes as
    # E[L^2] = mean^2 + std^2, which is > (mean)^2, so squaring the mean fill
    # systematically under-counts.  0 = use the mean only (backward compatible).
    hstu_fill_factor_std: float = 0.0
    # Efficiency of the ragged-HSTU attention kernel relative to the FAv3
    # roofline the SDPA backend prices.  ragged_hstu fuses gating and handles
    # jagged sequences, so it sustains a lower fraction of peak than FAv3;
    # <1.0 derates the attention-core time.  1.0 = price as FAv3 (default).
    hstu_attn_efficiency: float = 1.0
    # Separate efficiency for the attention *backward* kernel.  Traces show the
    # unautotuned Triton _hstu_attn_bwd runs ~1.8x less efficient than forward
    # (e.g. ~6% vs ~11% of peak) on top of doing ~2.5x the FLOPs, which a single
    # fwd=bwd efficiency cannot capture.  0 = reuse hstu_attn_efficiency.
    hstu_attn_bwd_efficiency: float = 0.0
    # HSTU attention is a *gated jagged* attention (SiLU gate + relative bias,
    # not softmax) whose cost the FAv3 SDPA roofline does not model (measured
    # traces scale as L^2 while the FAv3 backend scales ~L^1.4 and under-predicts
    # by >30x).  When > 0, the attention core is priced directly from its FLOPs
    #   flop_per_layer = B * heads * E[L^2] * (d_qk + d_v)   (causal-adjusted)
    # divided by (realizable peak flops x this efficiency), which is the measured
    # achieved fraction of matmul peak for the ragged-HSTU kernel (~0.25 fwd on
    # MI350X).  0 = keep the SDPA-roofline path (backward compatible).
    hstu_attn_flop_efficiency: float = 0.0
    # Attention-core cost model selector.  "" / "flop" (see above) or SDPA
    # roofline (default legacy paths), or "fav3_hstu" for the tile-level FAv3
    # simulator specialised for HSTU: it prices QKᵀ/A·V (and the 5 backward
    # sub-GEMMs) per-tile on 1 CU via origami, then adds the HSTU pointwise
    # epilogue (SiLU gate + relative bias + U gate) as a physical throughput.
    # This replaces the single opaque efficiency with a shape/arch-portable model.
    hstu_attn_model: str = ""
    # Fused-epilogue throughput (Gelem/s, chip-wide) for the "fav3_hstu" model:
    # score-matrix elements the SiLU-gate/relative-bias/U-gate epilogue processes
    # per second.  0 = use the calibrated defaults in the HSTU simulator.
    hstu_attn_epilogue_gelem_fwd: float = 0.0
    hstu_attn_epilogue_gelem_bwd: float = 0.0
    # Attention backward / forward wall-time ratio, used with the FLOP model.
    # The bwd_dkdv kernel measures ~2.0x the forward on MI350X.
    hstu_attn_bwd_ratio: float = 2.0
    # HSTU applies selective activation recomputation to the attention block:
    # the fused UVQK projection is *recomputed* in the backward to regenerate
    # Q/K/V for the attention-backward instead of stashing the large [T, 2048]
    # UVQK activation (measured traces show the UVQK GEMM running 2x -- once in
    # forward, once in backward).  True adds the UVQK forward GEMM to the bwd.
    hstu_recompute_attn: bool = False
    # Input width of the HSTU output projection.  The gated attention output is
    # concatenated with residual/gate streams before the output GEMM, so the
    # measured input width is 3*D (=1536 for D=512), not H*d_v (=512).  0 = auto
    # (H*d_v, backward compatible).
    hstu_output_input_dim: int = 0
    # Number of read+write elementwise passes over the block activation
    # footprint (input/linear dropout w/ masks, layer norms, SiLU gate, jagged
    # pack/unpack).  The naive model charges a single pass and under-counts.
    hstu_elementwise_passes: float = 6.0
    # DLRM dense (bottom) and interaction (top/over) MLP layer widths.
    dlrm_bottom_mlp: object = None  # list[int] | None
    dlrm_over_mlp: object = None  # list[int] | None
    dense_input_dim: int = 0  # width of the dense (continuous) feature vector
    # HSTU input preprocessor: a per-TOKEN MLP that fuses the content embedding
    # with the contextual/action/position features before the HSTU stack.  It is
    # a large GEMM (M = valid tokens, not batch) and is a dominant slice of the
    # "dense GEMM" trace bucket that a UVQK+output-only model omits.  Widths are
    # the output dims of each linear; input width defaults to (num contextual
    # features + 1) x D.  None = no preprocessor (backward compatible).
    dlrm_preprocessor_mlp: object = None  # list[int] | None
    dlrm_preprocessor_input_dim: int = 0  # 0 -> (num_contextual+1) x D
    # Multitask prediction head: a per-SAMPLE MLP tower replicated per task
    # (ranking models predict several targets).  Runs once per sample on the
    # pooled sequence output.  Widths are per-tower; num_tasks replicates it.
    dlrm_prediction_head_mlp: object = None  # list[int] | None
    dlrm_num_tasks: int = 1
    # Fraction of the embedding all-to-all that is *exposed* (not overlapped
    # behind compute).  Traces show a large share hidden; 1.0 = fully exposed
    # (conservative default).
    dlrm_comm_exposed_fraction: float = 1.0
    # Exposed collective *synchronization* time per step (ms), added on top of
    # the bandwidth estimate.  Traces show the sparse collectives are dominated
    # by peer-wait, not data movement: identical fwd/bwd payloads differ >10x in
    # kernel time and a few-KiB int64 splits exchange costs >15 ms.  That barrier
    # cost is load-imbalance driven and not derivable from bytes, so it is a
    # measured knob (0 = bandwidth-only, first-principles default).
    dlrm_collective_sync_ms: float = 0.0
    # Optional host->device (H2D) input-copy time per step (ms).  Not derivable
    # first-principles without the input pipeline; inject a measured value here
    # to include it in the projected step (0 = omit).
    dlrm_h2d_ms: float = 0.0
    # Optional device memcpy (jagged KJT pack/unpack + a2a staging) and local
    # reduce kernel time per step (ms).  These are framework-internal glue whose
    # byte traffic is not recoverable from the model config, so they are measured
    # inputs like dlrm_h2d_ms (0 = omit).
    dlrm_memcpy_ms: float = 0.0
    dlrm_reduce_ms: float = 0.0
    # Explicit per-token input-preprocessor GEMMs as [[in, out], ...] pairs (one
    # entry per branch/linear, each run over the valid-token count).  When set,
    # this overrides the collapsed dlrm_preprocessor_mlp chain so the multi-branch
    # ContextualPreprocessor (content 512->256, additional 1024->256, action
    # 24->256, fuse 256->512) is priced at its true FLOPs.  None = use the chain.
    dlrm_preprocessor_gemms: object = None  # list[[in,out], ...] | None
    # Framework "glue" data-movement per valid token (bytes): jagged KJT
    # pack/unpack, bf16<->fp32 activation casts/copies, and bias/norm reduction
    # kernels.  These are memory-bound streaming ops whose per-token byte traffic
    # is a structural property of the implementation (measured ~1.6e5 B/token for
    # the HSTU ranker); glue_ms = tokens * bytes_per_token / (peak_hbm * frac).
    # It scales linearly with the token count, unlike the fixed dlrm_memcpy_ms /
    # dlrm_reduce_ms knobs it supersedes.  0 = omit.
    dlrm_glue_bytes_per_token: float = 0.0
    dlrm_glue_hbm_fraction: float = 0.6  # streaming efficiency for the glue traffic


@dataclass
class TrainingConfig:
    """
    Configuration for training the profiler models.
    """

    model_config: ModelConfig
    runtime_config: RuntimeConfig
    model_parallel_config: ModelParallelConfig
    # Workload framework this config describes.  Read by the workload registry
    # (``resolve_top_level_spec``) to pick the top-level profiler tree.  Defaults
    # to ``megatron`` so the language-model path is unchanged.
    framework: str = "megatron"


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


# Frameworks whose profiler tree is a sparse-embedding + HSTU recommender
# (DLRM-v4).  They share the ``dlrm`` config-derivation path and are resolved to
# the ``torchrec_dlrm`` workload spec by the workload registry.
_DLRM_FRAMEWORKS = ("torchrec_dlrm", "torchrec", "dlrm", "dlrm_v4")


def dlrm_derive_default_args(args):
    """Fill in derived defaults for a DLRM-v4 (TorchRec/HSTU) config.

    Kept intentionally light: it only normalises the naming differences the
    dataclass copy (``update_config_from_args``) relies on and derives the data
    parallel size the same way the Megatron path does.  Everything else is
    copied straight from the workload args by field name.
    """
    world_size = int(os.getenv("NNODES", "1")) * int(os.getenv("GPUS_PER_NODE", "8"))

    # HSTU sequence length maps onto the generic ``sequence_length`` slot so the
    # activation/attention machinery sees it without special-casing.
    if getattr(args, "sequence_length", None) in (None, 0) and getattr(args, "hstu_max_seq_len", 0):
        args.sequence_length = args.hstu_max_seq_len

    # DLRM sparse embeddings dominate params; hidden_size defaults to the
    # embedding width when the ranker doesn't carry a separate hidden dim.
    if getattr(args, "hidden_size", 0) in (None, 0) and getattr(args, "embedding_dim", 0):
        args.hidden_size = args.embedding_dim
    if getattr(args, "embedding_dim", 0) in (None, 0) and getattr(args, "hidden_size", 0):
        args.embedding_dim = args.hidden_size

    tp = getattr(args, "tensor_model_parallel_size", 1) or 1
    pp = getattr(args, "pipeline_model_parallel_size", 1) or 1
    cp = getattr(args, "context_parallel_size", 1) or 1
    if not getattr(args, "data_parallel_size", None):
        args.data_parallel_size = max(1, world_size // (tp * pp * cp))
    if getattr(args, "context_parallel_size", None) is not None:
        args.context_model_parallel_size = args.context_parallel_size

    return args


def convert_primus_config_to_projection_config(primus_config) -> TrainingConfig:
    args = primus_config.get_module_config("pre_trainer")
    framework = getattr(args, "framework", "") or ""
    fw = framework.lower().strip()
    if fw == "megatron":
        args = megatron_derive_default_args(args)
    elif fw in _DLRM_FRAMEWORKS:
        args = dlrm_derive_default_args(args)
    else:
        raise NotImplementedError(f"Unsupported framework: {framework}")

    model_config = update_config_from_args(ModelConfig(), args)
    runtime_config = update_config_from_args(RuntimeConfig(), args)
    model_parallel_config = update_config_from_args(ModelParallelConfig(), args)

    training_config = TrainingConfig(
        model_config=model_config,
        runtime_config=runtime_config,
        model_parallel_config=model_parallel_config,
        framework=fw or "megatron",
    )

    return training_config

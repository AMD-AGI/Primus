###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional


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
    # FFN & MoE
    swiglu: bool = False
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_pattern: list = None
    moe_router_topk: int = 0
    moe_shared_expert_intermediate_size: int = 0
    # Misc
    share_embeddings_and_output_weights: bool = False
    # Precision – None means bf16, "hybrid" means FP8-hybrid (linear GEMMs in FP8)
    fp8: str = None

    # Primus Turbo flags — used to select the grouped-GEMM performance model
    enable_primus_turbo: bool = False
    use_turbo_grouped_mlp: bool = False
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


# ─────────────────────────────────────────────────────────────────────────────
# Inference / serving configuration
# ─────────────────────────────────────────────────────────────────────────────


def dtype_num_bytes(dtype: Optional[str]) -> float:
    """Return the byte width of a (loosely-named) tensor dtype.

    Accepts the informal names used throughout the projection layer
    (``bf16``, ``fp16``, ``fp8``, ``int8``, ``fp32``).  Unknown / ``None``
    values fall back to bf16 (2 bytes) which is the projection default.
    """
    if dtype is None:
        return 2.0
    key = str(dtype).lower().strip()
    return {
        "fp32": 4.0,
        "float32": 4.0,
        "bf16": 2.0,
        "bfloat16": 2.0,
        "fp16": 2.0,
        "float16": 2.0,
        "half": 2.0,
        "fp8": 1.0,
        "fp8_e4m3": 1.0,
        "fp8_e5m2": 1.0,
        "int8": 1.0,
        "uint8": 1.0,
        "fp4": 0.5,
        "int4": 0.5,
    }.get(key, 2.0)


@dataclass
class InferenceRequestConfig:
    """Describes the *serving* workload an inference projection targets.

    Unlike :class:`RuntimeConfig` (which models a training microbatch /
    global-batch / gradient-accumulation pipeline) this captures the
    request profile that drives prefill + autoregressive decode.
    """

    # Prompt (prefill) and generation (decode) lengths, in tokens.
    input_seq_len: int = 1024
    output_seq_len: int = 128
    # Number of sequences processed together in one forward (a decode batch).
    batch_size: int = 1
    # Max number of sequences whose KV cache is resident at once (for memory
    # sizing / continuous batching).  Defaults to ``batch_size``.
    max_concurrency: Optional[int] = None
    # Largest context (prompt + generated) any sequence can reach.  Drives
    # KV-cache capacity.  Defaults to ``input_seq_len + output_seq_len``.
    max_context_len: Optional[int] = None
    # Fraction of per-GPU HBM the serving engine may use (vLLM
    # ``gpu_memory_utilization`` / SGLang ``mem_fraction_static``).  Bounds the
    # usable HBM for weights + KV + activations and therefore the max concurrent
    # sequences.  ``None`` = use the full HBM capacity (legacy behaviour).
    kv_cache_memory_fraction: Optional[float] = None

    # ---- Precision ----
    weight_dtype: str = "bf16"     # weights kept resident (bf16 | fp8 | ...)
    kv_cache_dtype: str = "bf16"   # KV cache precision (bf16 | fp8 | int8 | ...)

    # ---- Inference features ----
    # Chunked prefill: split a long prompt into chunks of this many tokens
    # (0 disables; the whole prompt is one forward).
    chunked_prefill_size: int = 0
    # Speculative decoding: number of draft tokens proposed per verify step
    # (0 disables) and the expected acceptance rate in [0, 1].
    speculative_num_tokens: int = 0
    speculative_acceptance_rate: float = 0.0

    # ---- Serving / continuous-batching dynamics ----
    # How decode latency is modelled:
    #   "continuous" — continuous batching with mixed prefill+decode steps
    #                  (models the TPOT "pollution" real servers like vLLM see).
    #   "static"     — an idealized batch doing pure decode (legacy behaviour;
    #                  prefill is charged once as TTFT only).
    serving_model: str = "continuous"
    # Fixed per-decode-step host/launch overhead (microseconds). At low decode
    # batch the step is launch-bound; CUDA-graph capture shrinks this. Added to
    # every decode/mixed step. 0 = ignore (pure kernel-compute model).
    decode_step_overhead_us: float = 0.0
    # Extra cost fraction applied to a *mixed* (prefill+decode) step to model
    # vLLM's less-efficient PIECEWISE CUDA-graph path vs the FULL graph used for
    # uniform pure-decode steps. 0 = no penalty.
    mixed_batch_penalty: float = 0.0
    # CUDA-graph capture strategy. A friendly preset over the two low-level
    # knobs above:
    #   "none"      — eager, launch-bound per step (high per-step overhead).
    #   "piecewise" — piecewise graphs; mixed prefill+decode steps fall off the
    #                 captured graph (moderate overhead + a mixed-step penalty).
    #   "full"      — one full graph capture (minimal overhead, no penalty).
    # ``None`` leaves ``decode_step_overhead_us`` / ``mixed_batch_penalty`` at
    # their explicit values (legacy behaviour). An explicit non-zero value of
    # either low-level knob overrides the preset.
    cudagraph_mode: Optional[str] = None

    def resolved_max_context_len(self) -> int:
        if self.max_context_len is not None:
            return int(self.max_context_len)
        return int(self.input_seq_len) + int(self.output_seq_len)

    def resolved_max_concurrency(self) -> int:
        if self.max_concurrency is not None:
            return int(self.max_concurrency)
        return int(self.batch_size)

    def resolved_decode_step_overhead_us(self) -> float:
        """Per-decode-step launch overhead, honoring the cudagraph preset.

        An explicit non-zero ``decode_step_overhead_us`` always wins; otherwise
        the ``cudagraph_mode`` preset (if any) supplies a representative value.
        """
        if self.decode_step_overhead_us:
            return float(self.decode_step_overhead_us)
        return _CUDAGRAPH_PRESETS.get(self.cudagraph_mode, (0.0, 0.0))[0]

    def resolved_mixed_batch_penalty(self) -> float:
        """Mixed-step penalty fraction, honoring the cudagraph preset."""
        if self.mixed_batch_penalty:
            return float(self.mixed_batch_penalty)
        return _CUDAGRAPH_PRESETS.get(self.cudagraph_mode, (0.0, 0.0))[1]


# CUDA-graph presets → (decode_step_overhead_us, mixed_batch_penalty).
# Representative, ROCm-order-of-magnitude values; override with the explicit
# low-level knobs for a measured number.
_CUDAGRAPH_PRESETS = {
    "none": (40.0, 0.0),
    "piecewise": (8.0, 0.15),
    "full": (3.0, 0.0),
}


@dataclass
class InferenceCollectiveConfig:
    """Knobs for the explicit inference communication model (feature B).

    When :attr:`enabled` the inference performance projector replaces the
    layer profiler's *implicit* TP-AllReduce / EP-AllToAll cost with an
    explicit, reportable communication model.  At default values it
    reproduces the implicit cost; the knobs below let a user model
    **custom collective ops** — forcing a specific algorithm, hiding comm
    behind compute (overlap), or applying a fused-op speedup (e.g.
    AllReduce+RMSNorm fusion, DeepEP-style overlapped dispatch/combine).
    """

    enabled: bool = True
    # Algorithm override for the TP AllReduce / EP AllToAll. ``auto`` lets the
    # collective model pick the fastest; otherwise force a specific algorithm.
    tp_allreduce_algo: str = "auto"  # auto | ring | one_shot | two_shot | hierarchical
    ep_a2a_algo: str = "auto"  # auto | direct | single_shot | hierarchical
    # Fraction of communication hidden behind compute (0 = none, fully
    # exposed; 1 = fully overlapped). Set per phase.
    prefill_overlap: float = 0.0
    decode_overlap: float = 0.0
    # Custom fused-op efficiency multipliers applied to comm time
    # (<1.0 = faster, models kernel fusion / better algorithms). 1.0 = none.
    tp_allreduce_efficiency: float = 1.0
    ep_a2a_efficiency: float = 1.0
    # Whether to charge pipeline-stage P2P (send/recv) latency. Only nonzero
    # when pipeline_model_parallel_size > 1.
    include_pp_p2p: bool = True
    # Optional hardware overrides forwarded to ``get_default_args`` (node_bw,
    # pod_bw, bw_eff, latencies, ...). ``None`` uses the model defaults.
    hardware_config: Optional[Dict] = None


@dataclass
class DisaggregationConfig:
    """Prefill/decode disaggregation (feature A).

    Models separate prefill and decode worker pools, each with its own
    parallelism, plus the KV-cache transfer cost incurred when a request
    migrates from a prefill worker to a decode worker.  When disabled the
    projector runs the standard colocated two-phase model.
    """

    enabled: bool = False
    # Per-pool parallelism overrides. ``None`` falls back to the shared
    # ``model_parallel_config`` values.
    prefill_tp: Optional[int] = None
    prefill_pp: Optional[int] = None
    prefill_ep: Optional[int] = None
    decode_tp: Optional[int] = None
    decode_pp: Optional[int] = None
    decode_ep: Optional[int] = None
    # Number of replicas in each pool (for aggregate-throughput / GPU split).
    prefill_replicas: int = 1
    decode_replicas: int = 1
    # KV-cache transfer link. ``None`` bw uses the inter-node (pod) bandwidth
    # from the collective model; latency is a fixed per-transfer overhead (us).
    kv_transfer_bw_gbps: Optional[float] = None
    kv_transfer_latency_us: float = 0.0
    # Friendly preset over the two link knobs above, naming the KV-transfer
    # engine: "nixl", "mooncake", or "mori".  ``None`` leaves the explicit link
    # values untouched.  An explicit non-zero/non-None link knob overrides the
    # preset value for that field.
    transfer_backend: Optional[str] = None

    def resolved_kv_transfer_bw_gbps(self) -> Optional[float]:
        if self.kv_transfer_bw_gbps:
            return float(self.kv_transfer_bw_gbps)
        return _TRANSFER_BACKEND_PRESETS.get(self.transfer_backend, (None, 0.0))[0]

    def resolved_kv_transfer_latency_us(self) -> float:
        if self.kv_transfer_latency_us:
            return float(self.kv_transfer_latency_us)
        return _TRANSFER_BACKEND_PRESETS.get(self.transfer_backend, (None, 0.0))[1]

    def prefill_parallel(self, mp: ModelParallelConfig) -> "ModelParallelConfig":
        return _override_parallel(mp, self.prefill_tp, self.prefill_pp, self.prefill_ep)

    def decode_parallel(self, mp: ModelParallelConfig) -> "ModelParallelConfig":
        return _override_parallel(mp, self.decode_tp, self.decode_pp, self.decode_ep)


# KV-transfer engine presets → (kv_transfer_bw_gbps, kv_transfer_latency_us).
# Representative effective point-to-point KV link numbers; override with the
# explicit link knobs for a measured fabric.
_TRANSFER_BACKEND_PRESETS = {
    "nixl": (400.0, 5.0),
    "mooncake": (200.0, 10.0),
    "mori": (300.0, 7.0),
}


def _override_parallel(
    mp: ModelParallelConfig, tp: Optional[int], pp: Optional[int], ep: Optional[int]
) -> ModelParallelConfig:
    """Return a copy of ``mp`` with TP/PP/EP optionally overridden."""
    from copy import copy

    out = copy(mp)
    if tp is not None:
        out.tensor_model_parallel_size = int(tp)
    if pp is not None:
        out.pipeline_model_parallel_size = int(pp)
    if ep is not None:
        out.expert_model_parallel_size = int(ep)
    return out


@dataclass
class InferenceConfig:
    """Configuration for inference / serving projection.

    Reuses the same :class:`ModelConfig` / :class:`ModelParallelConfig` as
    training (so the existing profiler tree can estimate forward compute and
    parameter counts) but swaps the training :class:`RuntimeConfig` for an
    :class:`InferenceRequestConfig`.
    """

    model_config: ModelConfig
    request_config: InferenceRequestConfig
    model_parallel_config: ModelParallelConfig
    collective_config: InferenceCollectiveConfig = field(
        default_factory=InferenceCollectiveConfig
    )
    disaggregation_config: DisaggregationConfig = field(
        default_factory=DisaggregationConfig
    )

    def as_training_config(self, *, batch_size: int, seq_len: int) -> TrainingConfig:
        """Build a throwaway :class:`TrainingConfig` view for a given
        (batch, seq_len) so the existing profiler tree can be reused for
        forward-only compute estimation.
        """
        runtime = RuntimeConfig(
            global_batch_size=batch_size,
            micro_batch_size=batch_size,
            sequence_length=seq_len,
            data_parallel_size=1,
        )
        return TrainingConfig(
            model_config=self.model_config,
            runtime_config=runtime,
            model_parallel_config=self.model_parallel_config,
        )


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


def convert_primus_config_to_inference_config(
    primus_config,
    *,
    inference_overrides: Optional[dict] = None,
) -> InferenceConfig:
    """Build an :class:`InferenceConfig` from a primus config.

    Reuses :func:`convert_primus_config_to_projection_config` for the model
    and parallelism config, then layers an :class:`InferenceRequestConfig`
    on top.  ``inference_overrides`` (typically parsed from CLI flags) takes
    precedence over any ``inference:`` block embedded in the YAML.
    """
    training_config = convert_primus_config_to_projection_config(primus_config)

    # Allow an optional ``inference:`` block in the pre_trainer module config
    # (so a single workload YAML can carry a default serving profile).
    args = primus_config.get_module_config("pre_trainer")
    yaml_inf = getattr(args, "inference", None) or {}
    if not isinstance(yaml_inf, dict):
        yaml_inf = {}

    request = InferenceRequestConfig()
    # 1) seed from the training seq_length so a bare config still works.
    if getattr(training_config.runtime_config, "sequence_length", 0):
        request.input_seq_len = int(training_config.runtime_config.sequence_length)
    # 2) apply YAML inference block, then 3) CLI overrides.
    overrides = inference_overrides or {}
    for source in (yaml_inf, overrides):
        for f in fields(request):
            if f.name in source and source[f.name] is not None:
                setattr(request, f.name, source[f.name])

    # ---- Collective (feature B) + disaggregation (feature A) configs ----
    collective = InferenceCollectiveConfig()
    disagg = DisaggregationConfig()
    # YAML may carry nested ``collective:`` / ``disaggregation:`` blocks; CLI
    # overrides arrive flattened (e.g. ``collective_*`` / ``disagg_*`` keys).
    yaml_coll = yaml_inf.get("collective") if isinstance(yaml_inf.get("collective"), dict) else {}
    yaml_disagg = (
        yaml_inf.get("disaggregation") if isinstance(yaml_inf.get("disaggregation"), dict) else {}
    )
    _apply_fields(collective, yaml_coll)
    _apply_fields(disagg, yaml_disagg)
    _apply_prefixed(collective, overrides, prefix="collective_")
    _apply_prefixed(disagg, overrides, prefix="disagg_")

    return InferenceConfig(
        model_config=training_config.model_config,
        request_config=request,
        model_parallel_config=training_config.model_parallel_config,
        collective_config=collective,
        disaggregation_config=disagg,
    )


def _apply_fields(target, source: dict) -> None:
    """Set dataclass fields on ``target`` from matching keys in ``source``."""
    if not source:
        return
    valid = {f.name for f in fields(target)}
    for key, val in source.items():
        if key in valid and val is not None:
            setattr(target, key, val)


def _apply_prefixed(target, source: dict, *, prefix: str) -> None:
    """Set dataclass fields from ``source`` keys of the form ``<prefix><field>``."""
    if not source:
        return
    valid = {f.name for f in fields(target)}
    for key, val in source.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if name in valid and val is not None:
            setattr(target, name, val)

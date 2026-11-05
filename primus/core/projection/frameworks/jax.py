###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""JAX (MaxText) config adapter for the projection tool.

MaxText describes a job in the vocabulary of JAX meshes -- ``ici_*`` degrees
inside a node, ``dcn_*`` degrees across nodes, ``per_device_batch_size``,
``remat_policy`` -- and a Primus MaxText experiment is a thin overlay on top of
MaxText's own ``base.yml``, carrying only the keys the experiment changes.  This
module resolves that overlay into the projection's flat fields, after which a
MaxText run projects through the same profiler tree as a Megatron one.

Two things have to be recovered before the workload can be described at all:

* **The architecture**, which the experiment names (``model_name: llama3-8b``)
  but does not spell out.  It is read from MaxText's own model YAML when a
  MaxText checkout is present, and otherwise from the transcribed specs in
  :mod:`primus.core.projection.frameworks.model_specs`.  Architecture keys set
  directly in the Primus YAML win over both, which is how the shipped Grok-1
  preset describes a model MaxText has no config for.
* **The mesh**, because MaxText writes ``-1`` for "give this axis whatever ranks
  are left" and resolves it against the real device count.  The same fill is
  applied here, separately for the intra-node (ICI) and inter-node (DCN) groups,
  exactly as ``fill_unspecified_mesh_axes`` does.

The mesh axes then fold into the projection's parallelism the way they already
overlap in meaning: tensor and tensor-sequence shard a layer's tensors (TP),
sequence and context shard the sequence (CP), stage is pipeline (PP), expert is
EP, and data/FSDP replicate or shard the batch (DP).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from primus.core.projection.frameworks import is_normalized, mark_normalized
from primus.core.projection.frameworks.model_specs import (
    ModelSpec,
    maxtext_builtin_spec,
    spec_to_projection_fields,
)

# MaxText mesh axes, grouped by the projection dimension they shard.  Each name
# is the suffix of an ``ici_<name>_parallelism`` / ``dcn_<name>_parallelism`` key.
_TENSOR_AXES = ("tensor", "tensor_transpose", "tensor_sequence")
_CONTEXT_AXES = ("sequence", "context", "context_usp_ulysses", "context_autoregressive")
_PIPELINE_AXES = ("pipeline",)
_EXPERT_AXES = ("expert",)
_DATA_AXES = ("data", "fsdp", "fsdp_transpose", "diloco", "autoregressive")

_ALL_AXES = _TENSOR_AXES + _CONTEXT_AXES + _PIPELINE_AXES + _EXPERT_AXES + _DATA_AXES

# MaxText base.yml defaults for the axes it auto-shards; every other axis
# defaults to 1.
_AXIS_DEFAULTS = {"ici_fsdp": -1, "dcn_data": -1}

# MaxText base.yml defaults for the architecture keys this adapter reads, used
# when the experiment sets neither them nor a resolvable ``model_name``.
_MAXTEXT_ARCH_DEFAULTS: Dict[str, object] = {
    "base_num_decoder_layers": 16,
    "base_emb_dim": 2048,
    "base_mlp_dim": 7168,
    "base_num_query_heads": 16,
    "base_num_kv_heads": 16,
    "base_moe_mlp_dim": -1,
    "head_dim": 128,
    "vocab_size": 32000,
    "num_experts": 1,
    "num_experts_per_tok": 1,
    "shared_experts": 0,
    "first_num_dense_layers": 0,
    "logits_via_embedding": False,
    "use_qk_norm": False,
    "sliding_window_size": 0,
    "attention_type": "global",
    "q_lora_rank": 0,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mlp_activations": ["silu", "linear"],
}

# The architecture keys an experiment may set directly (Grok-1 does), read off
# the merged namespace in preference to anything the model registry says.
_ARCH_KEYS = tuple(_MAXTEXT_ARCH_DEFAULTS)

# Remat policies that keep some activations; MaxText's other policies all
# recompute selectively rather than wholesale.
_FULL_REMAT = "full"
_NO_REMAT = frozenset({"", "none", "save_all"})


def _maxtext_config_dir() -> Optional[Path]:
    """Locate MaxText's ``configs`` directory, or ``None`` if there is none."""
    candidates: List[Path] = []
    env_path = os.getenv("PRIMUS_MAXTEXT_PATH") or os.getenv("BACKEND_PATH")
    if env_path:
        candidates.append(Path(env_path))

    primus_root = Path(__file__).resolve().parents[4]
    candidates.append(primus_root / "third_party" / "maxtext")
    tp_root = os.getenv("PRIMUS_THIRDPARTY_DIR") or str(Path.home() / ".cache" / "Primus" / "third_party")
    candidates.append(Path(tp_root) / "maxtext")

    try:
        import maxtext  # noqa: F401

        candidates.append(Path(maxtext.__file__).resolve().parent)
    except Exception:
        pass

    for root in candidates:
        # MaxText moved from ``MaxText/`` to ``src/maxtext/``; accept both, and
        # accept being pointed straight at the package.
        for relative in ("src/maxtext/configs", "src/MaxText/configs", "MaxText/configs", "configs"):
            config_dir = root / relative
            if (config_dir / "models").is_dir():
                return config_dir
    return None


def _read_maxtext_model_config(model_name: str) -> Optional[Dict[str, object]]:
    """Read MaxText's own YAML for *model_name*, or ``None`` if unavailable."""
    if not model_name or model_name.lower().strip() == "default":
        return None
    config_dir = _maxtext_config_dir()
    if config_dir is None:
        return None
    for suffix in (".yml", ".yaml"):
        path = config_dir / "models" / f"{model_name}{suffix}"
        if path.is_file():
            import yaml

            with open(path, "r") as handle:
                loaded = yaml.safe_load(handle)
            return loaded if isinstance(loaded, dict) else None
    return None


def _spec_from_maxtext_keys(mx: Dict[str, object]) -> ModelSpec:
    """Translate MaxText architecture keys into a :class:`ModelSpec`."""
    num_layers = int(mx["base_num_decoder_layers"])
    hidden_size = int(mx["base_emb_dim"])
    num_heads = int(mx["base_num_query_heads"])
    num_kv_heads = int(mx.get("base_num_kv_heads") or num_heads)
    head_dim = int(mx.get("head_dim") or (hidden_size // num_heads))

    num_experts = int(mx.get("num_experts") or 1)
    # MaxText spells "dense" as a single expert; the projection spells it as none.
    num_experts = 0 if num_experts <= 1 else num_experts
    moe_ffn = int(mx.get("base_moe_mlp_dim") or -1)
    if num_experts and moe_ffn <= 0:
        moe_ffn = int(mx["base_mlp_dim"])

    is_mla = str(mx.get("attention_type") or "global").lower() == "mla"
    activations = mx.get("mlp_activations") or []

    return ModelSpec(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=int(mx["base_mlp_dim"]),
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        kv_channels=int(mx.get("qk_nope_head_dim") or head_dim) if is_mla else head_dim,
        vocab_size=int(mx.get("vocab_size") or 0),
        # A gated MLP ("linear" alongside the activation) is the three-matrix
        # shape the projection prices as SwiGLU, whichever activation it gates.
        swiglu="linear" in [str(a).lower() for a in activations],
        tie_embeddings=bool(mx.get("logits_via_embedding", False)),
        qk_layernorm=bool(mx.get("use_qk_norm", False)) or is_mla,
        num_experts=num_experts,
        moe_ffn_hidden_size=moe_ffn if num_experts else 0,
        moe_router_topk=int(mx.get("num_experts_per_tok") or 0) if num_experts else 0,
        num_shared_experts=int(mx.get("shared_experts") or 0) if num_experts else 0,
        num_dense_layers=int(mx.get("first_num_dense_layers") or 0),
        multi_latent_attention=is_mla,
        q_lora_rank=int(mx.get("q_lora_rank") or 0) if is_mla else 0,
        kv_lora_rank=int(mx.get("kv_lora_rank") or 0) if is_mla else 0,
        qk_head_dim=int(mx.get("qk_nope_head_dim") or 0) if is_mla else 0,
        qk_pos_emb_head_dim=int(mx.get("qk_rope_head_dim") or 0) if is_mla else 0,
        v_head_dim=int(mx.get("v_head_dim") or 0) if is_mla else 0,
        attn_sliding_window=int(mx.get("sliding_window_size") or 0),
    )


def resolve_model_spec(args) -> ModelSpec:
    """Return the architecture a MaxText experiment describes."""
    model_name = str(getattr(args, "model_name", "") or "")
    explicit = {key: getattr(args, key) for key in _ARCH_KEYS if getattr(args, key, None) is not None}

    from_maxtext = _read_maxtext_model_config(model_name)
    if from_maxtext is not None:
        merged = dict(_MAXTEXT_ARCH_DEFAULTS)
        merged.update({k: v for k, v in from_maxtext.items() if k in _MAXTEXT_ARCH_DEFAULTS})
        merged.update(explicit)
        return _spec_from_maxtext_keys(merged)

    if "base_num_decoder_layers" in explicit and "base_emb_dim" in explicit:
        # The experiment spells the architecture out itself (Grok-1 does, because
        # MaxText ships no config for it).
        merged = dict(_MAXTEXT_ARCH_DEFAULTS)
        merged.update(explicit)
        return _spec_from_maxtext_keys(merged)

    spec = maxtext_builtin_spec(model_name)
    if spec is not None:
        return spec

    raise ValueError(
        f"Cannot project MaxText model '{model_name}': no MaxText checkout was found to "
        "read its model config from, the model is not transcribed in "
        "primus/core/projection/frameworks/model_specs.py, and the experiment does not "
        "set base_num_decoder_layers / base_emb_dim itself. Point PRIMUS_MAXTEXT_PATH at "
        "a MaxText checkout, or spell the architecture out in the experiment YAML."
    )


def _resolve_axis_group(args, prefix: str, target_product: int) -> Dict[str, int]:
    """Resolve one mesh group's degrees, filling a single ``-1`` axis.

    Mirrors MaxText's ``fill_unspecified_mesh_axes``: at most one axis may be
    unspecified, and it absorbs whatever ranks the others leave.
    """
    degrees = {}
    for axis in _ALL_AXES:
        key = f"{prefix}_{axis}_parallelism"
        default = _AXIS_DEFAULTS.get(f"{prefix}_{axis}", 1)
        degrees[axis] = int(getattr(args, key, default) or default)

    unspecified = [axis for axis, value in degrees.items() if value < 0]
    if len(unspecified) > 1:
        raise ValueError(
            f"More than one {prefix.upper()} parallelism axis is unspecified (-1): "
            f"{', '.join(unspecified)}. MaxText allows at most one."
        )
    if unspecified:
        specified_product = 1
        for axis, value in degrees.items():
            if value > 0:
                specified_product *= value
        filled, remainder = divmod(target_product, specified_product)
        if remainder or filled < 1:
            raise ValueError(
                f"Cannot resolve {prefix.upper()} axis '{unspecified[0]}': "
                f"{target_product} devices do not divide evenly by the specified "
                f"degrees (product {specified_product})."
            )
        degrees[unspecified[0]] = filled
    return degrees


def _product(degrees: Dict[str, int], axes) -> int:
    result = 1
    for axis in axes:
        result *= degrees.get(axis, 1)
    return result


def _resolve_parallelism(args) -> Dict[str, int]:
    num_nodes = int(os.getenv("NNODES", "1"))
    gpus_per_node = int(os.getenv("GPUS_PER_NODE", "8"))

    ici = _resolve_axis_group(args, "ici", gpus_per_node)
    dcn = _resolve_axis_group(args, "dcn", num_nodes)

    def combined(axes):
        return _product(ici, axes) * _product(dcn, axes)

    return {
        "tp": combined(_TENSOR_AXES),
        "cp": combined(_CONTEXT_AXES),
        "pp": combined(_PIPELINE_AXES),
        "ep": combined(_EXPERT_AXES),
        # MaxText shards the batch over data, FSDP and expert alike, so the
        # projection's DP -- which also contains the EP ranks -- is everything
        # that is not TP, CP or PP.
        "dp": combined(_DATA_AXES) * combined(_EXPERT_AXES),
        "fsdp": ici.get("fsdp", 1) * dcn.get("fsdp", 1),
        "world_size": num_nodes * gpus_per_node,
    }


def _resolve_recompute(args) -> Dict[str, object]:
    policy = str(getattr(args, "remat_policy", _FULL_REMAT) or "").lower().strip()
    if policy == _FULL_REMAT:
        return {"recompute_granularity": "full", "recompute_method": "uniform"}
    if policy in _NO_REMAT:
        return {"recompute_granularity": None, "recompute_method": None}
    # Every other MaxText policy names a set of tensors to keep and recomputes
    # the rest, which is what the projection calls selective.
    return {"recompute_granularity": "selective", "recompute_method": None}


def _resolve_precision(args) -> Dict[str, Optional[str]]:
    quantization = str(getattr(args, "quantization", "") or "").lower().strip()
    if "fp8" not in quantization:
        return {"fp8": None, "fp8_recipe": None}
    # MaxText's nanoo_fp8 is the OCP-variant FP8 AMD GPUs run; both spellings are
    # tensor-scaled FP8 GEMMs as far as the projection's GEMM model is concerned.
    return {"fp8": "hybrid", "fp8_recipe": "tensorwise"}


def _normalize(args) -> None:
    """Write the projection's field names onto a MaxText trainer namespace."""
    if int(getattr(args, "global_parameter_scale", 1) or 1) != 1:
        raise ValueError(
            "MaxText global_parameter_scale != 1 rescales the architecture by rules the "
            "projection does not model. Set the architecture explicitly (base_emb_dim, "
            "base_num_decoder_layers, ...) in the experiment YAML instead."
        )

    spec = resolve_model_spec(args)
    flat: Dict[str, object] = dict(spec_to_projection_fields(spec))

    dims = _resolve_parallelism(args)
    flat.update(
        {
            "tensor_model_parallel_size": dims["tp"],
            "pipeline_model_parallel_size": dims["pp"],
            "context_parallel_size": dims["cp"],
            "expert_model_parallel_size": dims["ep"],
            "data_parallel_size": dims["dp"],
            # A non-trivial FSDP axis shards parameters, gradients and optimizer
            # state per parameter, which is the FSDP2 model the projection has.
            "use_torch_fsdp2": dims["fsdp"] > 1,
            "use_distributed_optimizer": False,
            "overlap_grad_reduce": True,
            "overlap_param_gather": dims["fsdp"] > 1,
            "virtual_pipeline_model_parallel_size": max(
                1, int(getattr(args, "num_pipeline_repeats", 1) or 1)
            ),
            "pipeline_model_parallel_layout": None,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
            "num_layers_per_virtual_pipeline_stage": None,
            "enable_zero_bubble": False,
        }
    )

    # MaxText's per_device_batch_size counts every device, while the batch is
    # only sharded across the data/FSDP/expert axes -- so one data-parallel rank
    # carries the batch of all the TP/CP/PP ranks that mirror it.
    per_device_batch_size = float(getattr(args, "per_device_batch_size", 1.0) or 1.0)
    step_batch = per_device_batch_size * dims["world_size"]
    accumulation_steps = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    flat.update(
        {
            "seq_length": int(getattr(args, "max_target_length", 0) or 0),
            "micro_batch_size": max(1, int(step_batch / max(1, dims["dp"]))),
            "global_batch_size": max(1, int(step_batch * accumulation_steps)),
        }
    )

    recompute = _resolve_recompute(args)
    flat.update(recompute)
    flat["recompute_num_layers"] = (
        int(flat["num_layers"]) if recompute["recompute_granularity"] == "full" else 0
    )
    flat["recompute_layer_ids"] = None

    flat.update(_resolve_precision(args))

    attention = str(getattr(args, "attention", "") or "").lower()
    flat.update(
        {
            "use_flash_attn": "flash" in attention or attention == "autoselected",
            "optimizer": str(getattr(args, "opt_type", "adamw") or "adamw").lower(),
            "enable_primus_turbo": False,
            "use_turbo_grouped_gemm": False,
            "use_turbo_grouped_mlp": False,
            "use_turbo_deepep": False,
            "turbo_sync_free_moe_stage": 0,
            "cross_entropy_loss_fusion": False,
        }
    )

    for key, value in flat.items():
        setattr(args, key, value)


def maxtext_derive_default_args(args):
    """Normalize a MaxText trainer config into the projection's fields.

    Idempotent for the same reason the TorchTitan adapter is: the performance
    driver edits the normalized namespace and re-converts it, and a second pass
    that went back to the mesh keys would discard those edits.
    """
    from primus.core.projection.training_config import megatron_derive_default_args

    if not is_normalized(args):
        _normalize(args)
        mark_normalized(args)
    return megatron_derive_default_args(args)


# ---------------------------------------------------------------------------
# Benchmark write-back: flat projection fields -> MaxText's own config
# ---------------------------------------------------------------------------
#
# The performance driver shrinks the model onto the bench node by editing the
# flat fields -- capping the stack at one or two layers, reducing expert
# parallelism, flattening the pipeline.  MaxText builds from its own mesh and
# architecture keys, so without the reverse translation a benchmark would
# resolve the full-size model from ``model_name`` and try to build all of it.


def _clear_axis_group(args, axes, exclude=()) -> None:
    """Set every ICI and DCN axis in *axes* to 1, leaving *exclude* alone."""
    for axis in axes:
        if axis in exclude:
            continue
        for scope in ("ici", "dcn"):
            setattr(args, f"{scope}_{axis}_parallelism", 1)


def maxtext_apply_bench_overrides(args) -> None:
    """Push the driver's flat bench edits back into MaxText's config.

    Mutates *args* in place.  MaxText spreads one projection dimension across
    several mesh axes -- tensor parallelism over ``tensor``,
    ``tensor_transpose`` and ``tensor_sequence``, for instance -- and there is
    no way to know which the user meant, so each group is collapsed onto a
    single named axis carrying the whole degree and its siblings are set to 1.
    That preserves the product the projection cares about, which is what decides
    the shapes the benchmark measures.
    """
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    if num_layers:
        args.base_num_decoder_layers = num_layers

    num_experts = getattr(args, "num_experts", None)
    if num_experts:
        args.num_experts = int(num_experts)

    # Intra-node (ICI) carries the whole degree; inter-node (DCN) goes to 1,
    # because the bench runs inside one node by construction.
    _clear_axis_group(args, _TENSOR_AXES)
    args.ici_tensor_parallelism = int(getattr(args, "tensor_model_parallel_size", 1) or 1)

    _clear_axis_group(args, _CONTEXT_AXES)
    args.ici_context_parallelism = int(getattr(args, "context_model_parallel_size", 1) or 1)

    _clear_axis_group(args, _EXPERT_AXES)
    args.ici_expert_parallelism = int(getattr(args, "expert_model_parallel_size", 1) or 1)

    _clear_axis_group(args, _PIPELINE_AXES)
    args.ici_pipeline_parallelism = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)

    # Leave data parallelism unsharded rather than -1: the projection models
    # data-parallel gradient reduction analytically when it scales up to the
    # target cluster, and an auto-filled FSDP axis would shard the very
    # parameters the layer benchmark needs whole.
    _clear_axis_group(args, _DATA_AXES)

    # MaxText sizes the batch per device, and the bench measures one microbatch.
    args.per_device_batch_size = float(int(getattr(args, "micro_batch_size", 1) or 1))
    seq_length = int(getattr(args, "seq_length", 0) or 0)
    if seq_length:
        args.max_target_length = seq_length

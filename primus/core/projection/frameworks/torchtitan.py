###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TorchTitan config adapter for the projection tool.

A TorchTitan experiment describes its job the way TorchTitan's ``JobConfig``
does -- ``model.flavor``, ``training.local_batch_size``,
``parallelism.data_parallel_shard_degree`` -- and keeps the architecture itself
out of the YAML entirely, in a flavor table inside the ``torchtitan`` package.
This module recovers the architecture and rewrites the job in the projection's
vocabulary, after which a TorchTitan run projects through exactly the same
profiler tree as a Megatron one.

Architecture resolution prefers the installed ``torchtitan``, which is the
authority on its own flavors and covers flavors newer than this file.  When
TorchTitan is not importable -- the normal case for projection, which is meant to
size a cluster without a GPU or a training checkout -- it falls back to the
transcribed specs in :mod:`primus.core.projection.frameworks.model_specs`.
Explicit fields under ``model:`` in the experiment YAML win over both, which is
how the shipped 405B preset shortens the stack for debugging.
"""

import math
import os
from types import SimpleNamespace
from typing import Dict, List, Optional

from primus.core.projection.frameworks import is_normalized, mark_normalized
from primus.core.projection.frameworks.model_specs import (
    ModelSpec,
    llama4_moe_ffn_hidden_size,
    llama_ffn_hidden_size,
    spec_to_projection_fields,
    torchtitan_builtin_spec,
)

# torch.distributed.pipelining schedules that run one stage per rank.  Anything
# else is looped and defaults to two virtual stages per rank, which is how
# TorchTitan picks its stage count when the config does not say.
_SINGLE_STAGE_SCHEDULES = frozenset({"1f1b", "gpipe"})

# Schedule names that split the backward into B/W passes.
_ZERO_BUBBLE_MARKERS = ("zerobubble", "zbv", "dualpipe")

# ``model:`` keys that override the resolved architecture, under the names
# TorchTitan's ModelArgs uses.
_MODEL_ARG_OVERRIDES = {
    "n_layers": "num_layers",
    "dim": "hidden_size",
    "n_heads": "num_attention_heads",
    "n_kv_heads": "num_query_groups",
    "vocab_size": "padded_vocab_size",
    "moe_inter_dim": "moe_ffn_hidden_size",
}


def _ns_get(obj, path: str, default=None):
    """Read a dotted path out of a nested namespace, tolerating missing levels."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return default if cur is None else cur


def _torchtitan_model_args(name: str, flavor: str):
    """Return the installed TorchTitan ModelArgs for a flavor, or ``None``."""
    try:
        import torchtitan.models  # noqa: F401  (registers the built-in train specs)
        from torchtitan.protocols.train_spec import get_train_spec
    except Exception:
        return None
    try:
        return get_train_spec(name).model_args[flavor]
    except Exception:
        return None


def _spec_from_model_args(name: str, model_args) -> Optional[ModelSpec]:
    """Translate a TorchTitan ModelArgs dataclass into a :class:`ModelSpec`.

    Only attributes TorchTitan defines with a stable meaning are read; a flavor
    whose FFN width cannot be determined returns ``None`` so the caller falls
    back to a transcribed spec rather than projecting an invented shape.
    """
    dim = getattr(model_args, "dim", 0)
    n_layers = getattr(model_args, "n_layers", 0)
    n_heads = getattr(model_args, "n_heads", 0)
    if not (dim and n_layers and n_heads):
        return None
    n_kv_heads = getattr(model_args, "n_kv_heads", None) or n_heads
    head_dim = getattr(model_args, "head_dim", None) or (dim // n_heads)

    moe_args = getattr(model_args, "moe_args", None)
    # Qwen 3 ships dense and MoE flavors off one dataclass and gates on a flag;
    # DeepSeek / Llama 4 / GPT-OSS are MoE whenever they carry moe_args.
    moe_enabled = moe_args is not None and getattr(model_args, "moe_enabled", True)
    num_experts = getattr(moe_args, "num_experts", 0) if moe_enabled else 0
    top_k = getattr(moe_args, "top_k", 0) if moe_enabled else 0
    num_shared = getattr(moe_args, "num_shared_experts", 0) if moe_enabled else 0

    multiple_of = getattr(model_args, "multiple_of", None)
    ffn_dim_multiplier = getattr(model_args, "ffn_dim_multiplier", None)
    if multiple_of is not None:
        # Llama 3 / Llama 4 derive the dense width rather than storing it.
        ffn_hidden_size = llama_ffn_hidden_size(dim, multiple_of, ffn_dim_multiplier)
    else:
        ffn_hidden_size = getattr(model_args, "hidden_dim", 0) or getattr(model_args, "inter_dim", 0)

    moe_ffn_hidden_size = getattr(model_args, "moe_inter_dim", 0)
    if moe_enabled and not moe_ffn_hidden_size and multiple_of is not None:
        moe_ffn_hidden_size = llama4_moe_ffn_hidden_size(
            dim,
            multiple_of,
            ffn_dim_multiplier,
            top_k,
            num_shared,
            bool(getattr(model_args, "auto_scale_hidden_dim", True)),
        )
    if not ffn_hidden_size:
        # GPT-OSS is MoE in every layer and has no dense width at all.
        ffn_hidden_size = moe_ffn_hidden_size
    if not ffn_hidden_size:
        return None

    kv_lora_rank = getattr(model_args, "kv_lora_rank", 0) or 0
    is_mla = bool(kv_lora_rank) and hasattr(model_args, "qk_nope_head_dim")

    return ModelSpec(
        num_layers=n_layers,
        hidden_size=dim,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=n_heads,
        num_query_groups=n_kv_heads,
        kv_channels=getattr(model_args, "qk_nope_head_dim", head_dim) if is_mla else head_dim,
        vocab_size=getattr(model_args, "vocab_size", 0),
        tie_embeddings=bool(getattr(model_args, "enable_weight_tying", False)),
        qk_layernorm=bool(getattr(model_args, "qk_norm", False)) or is_mla,
        num_experts=num_experts,
        moe_ffn_hidden_size=moe_ffn_hidden_size if moe_enabled else 0,
        moe_router_topk=top_k,
        num_shared_experts=num_shared,
        num_dense_layers=getattr(model_args, "n_dense_layers", 0) or 0,
        moe_layer_step=getattr(model_args, "interleave_moe_layer_step", 1) or 1,
        multi_latent_attention=is_mla,
        q_lora_rank=getattr(model_args, "q_lora_rank", 0) or 0,
        kv_lora_rank=kv_lora_rank,
        qk_head_dim=getattr(model_args, "qk_nope_head_dim", 0) or 0,
        qk_pos_emb_head_dim=getattr(model_args, "qk_rope_head_dim", 0) or 0,
        v_head_dim=getattr(model_args, "v_head_dim", 0) or 0,
        attn_sliding_window=getattr(model_args, "sliding_window_size", 0) or 0,
    )


def resolve_model_spec(name: str, flavor: str) -> ModelSpec:
    """Return the architecture for a TorchTitan ``(name, flavor)`` pair."""
    # TorchTitan's registry is keyed the way it writes flavors ("8B", "30B-A3B");
    # an experiment that spells one in lower case should still reach it rather
    # than silently falling through to the transcribed table.
    model_args = _torchtitan_model_args(name, flavor) or _torchtitan_model_args(name, flavor.upper())
    if model_args is not None:
        spec = _spec_from_model_args(name, model_args)
        if spec is not None:
            return spec

    spec = torchtitan_builtin_spec(name, flavor)
    if spec is not None:
        return spec

    raise ValueError(
        f"Cannot project TorchTitan model '{name}' flavor '{flavor}': the flavor is "
        "neither known to the installed torchtitan nor transcribed in "
        "primus/core/projection/frameworks/model_specs.py. Install torchtitan "
        "(primus-cli deps sync) or add the flavor to BUILTIN_MODEL_SPECS."
    )


def _stage_layer_counts(num_stages: int, num_layers: int, input_weight: int, output_weight: int) -> List[int]:
    """Per-stage decoder-layer counts, matching TorchTitan's own split.

    TorchTitan spreads ``num_layers + input_weight + output_weight`` evenly over
    the stages and then charges the embedding and the output head against the
    first and last stage's share, which is what makes those stages hold fewer
    transformer layers than the middle ones.
    """
    if num_stages <= 1:
        return [num_layers]

    effective = num_layers + input_weight + output_weight
    per_stage, extra = divmod(effective, num_stages)

    counts = []
    for stage_idx in range(num_stages):
        share = per_stage + (1 if stage_idx < extra else 0)
        if stage_idx == 0:
            share -= input_weight
        elif stage_idx == num_stages - 1:
            share -= output_weight
        counts.append(max(0, share))

    # The weighted split can leave the total off by the rounding it applied to
    # the embedding/output charge; put any difference on the middle stages.
    drift = num_layers - sum(counts)
    idx = 1 if num_stages > 2 else 0
    while drift > 0:
        counts[idx % num_stages] += 1
        idx += 1
        drift -= 1
    return counts


def _layout_string(counts: List[int]) -> str:
    """Encode per-stage layer counts as a Megatron pipeline layout string."""
    stages = [f"t*{count}" for count in counts]
    stages[0] = "E" + stages[0]
    stages[-1] = stages[-1] + ",L"
    return "|".join(stages)


def _resolve_parallelism(args, world_size: int) -> Dict[str, int]:
    par = getattr(args, "parallelism", SimpleNamespace())
    tp = int(_ns_get(par, "tensor_parallel_degree", 1) or 1)
    pp = int(_ns_get(par, "pipeline_parallel_degree", 1) or 1)
    cp = int(_ns_get(par, "context_parallel_degree", 1) or 1)
    ep = int(_ns_get(par, "expert_parallel_degree", 1) or 1)
    dp_replicate = int(_ns_get(par, "data_parallel_replicate_degree", 1) or 1)
    dp_shard = int(_ns_get(par, "data_parallel_shard_degree", -1))

    if dp_shard < 0:
        # TorchTitan's "-1 means the leftover ranks", which it resolves against
        # the real world size.
        dp_shard = max(1, world_size // max(1, dp_replicate * cp * tp * pp))

    return {
        "tp": tp,
        "pp": pp,
        "cp": cp,
        "ep": ep,
        "dp_replicate": dp_replicate,
        "dp_shard": dp_shard,
        "dp": dp_replicate * dp_shard,
    }


def _resolve_pipeline(args, dims: Dict[str, int], num_layers: int) -> Dict[str, object]:
    par = getattr(args, "parallelism", SimpleNamespace())
    pp = dims["pp"]
    schedule = str(_ns_get(par, "pipeline_parallel_schedule", "1F1B") or "1F1B")
    normalized_schedule = schedule.lower().replace("_", "").replace("-", "")
    single_stage = normalized_schedule in _SINGLE_STAGE_SCHEDULES

    input_weight = int(_ns_get(par, "pipeline_parallel_first_stage_less_layers", 1) or 0)
    output_weight = int(_ns_get(par, "pipeline_parallel_last_stage_less_layers", 1) or 0)
    layers_per_stage = _ns_get(par, "pipeline_parallel_layers_per_stage", None)

    if pp <= 1:
        vpp = 1
    elif layers_per_stage:
        num_virtual_stages = math.ceil((num_layers + input_weight + output_weight) / int(layers_per_stage))
        vpp = max(1, num_virtual_stages // pp)
    else:
        vpp = 1 if single_stage else 2

    layout = None
    total_stages = pp * vpp
    if total_stages > 1:
        counts = _stage_layer_counts(total_stages, num_layers, input_weight, output_weight)
        # An even split is what the projection already assumes, so only spell the
        # layout out when the embedding/output charge actually skews it.
        if len(set(counts)) > 1 and all(counts):
            layout = _layout_string(counts)

    return {
        "virtual_pipeline_model_parallel_size": vpp,
        "pipeline_model_parallel_layout": layout,
        "enable_zero_bubble": any(m in normalized_schedule for m in _ZERO_BUBBLE_MARKERS),
    }


def _resolve_precision(args) -> Dict[str, Optional[str]]:
    """Map TorchTitan's model converters onto the projection's FP8 recipe."""
    converters = [str(c).lower() for c in (_ns_get(args, "model.converters", []) or [])]
    uses_mx = any("mx" in c for c in converters)
    uses_float8 = any("float8" in c for c in converters)

    if uses_mx:
        return {"fp8": "hybrid", "fp8_recipe": "mxfp8"}
    if uses_float8:
        recipe = _ns_get(args, "quantize.linear.float8.recipe_name", None)
        return {"fp8": "hybrid", "fp8_recipe": str(recipe or "tensorwise")}
    return {"fp8": None, "fp8_recipe": None}


def _resolve_recompute(args) -> Dict[str, object]:
    mode = str(_ns_get(args, "activation_checkpoint.mode", "none") or "none").lower()
    if mode == "full":
        # TorchTitan's full AC wraps every transformer block, which is Megatron's
        # uniform method rather than its first-N-of-the-stage block method.
        return {"recompute_granularity": "full", "recompute_method": "uniform"}
    if mode in ("selective", "memory_budget"):
        return {"recompute_granularity": "selective", "recompute_method": None}
    return {"recompute_granularity": None, "recompute_method": None}


def _normalize(args) -> None:
    """Write the projection's field names onto a TorchTitan trainer namespace."""
    name = str(_ns_get(args, "model.name", "") or "")
    flavor = str(_ns_get(args, "model.flavor", "") or "")
    spec = resolve_model_spec(name, flavor)

    flat: Dict[str, object] = dict(spec_to_projection_fields(spec))

    # Explicit ModelArgs overrides in the experiment YAML beat the flavor table.
    model_ns = getattr(args, "model", None)
    for titan_name, projection_name in _MODEL_ARG_OVERRIDES.items():
        override = getattr(model_ns, titan_name, None)
        if override is not None:
            flat[projection_name] = override
    if flat["num_layers"] != spec.num_layers:
        # A shortened stack invalidates the per-layer MoE pattern derived above.
        flat["moe_layer_freq"] = spec.moe_layer_pattern()[: int(flat["num_layers"])]

    world_size = int(os.getenv("NNODES", "1")) * int(os.getenv("GPUS_PER_NODE", "8"))
    dims = _resolve_parallelism(args, world_size)
    flat.update(
        {
            "tensor_model_parallel_size": dims["tp"],
            "pipeline_model_parallel_size": dims["pp"],
            "context_parallel_size": dims["cp"],
            "expert_model_parallel_size": dims["ep"],
            "data_parallel_size": dims["dp"],
            # dp_shard is FSDP2 per-parameter sharding, which is the model the
            # projection already has for ``use_torch_fsdp2``.
            "use_torch_fsdp2": dims["dp_shard"] > 1,
            "use_distributed_optimizer": False,
            "overlap_grad_reduce": True,
            "overlap_param_gather": dims["dp_shard"] > 1,
        }
    )
    flat.update(_resolve_pipeline(args, dims, int(flat["num_layers"])))

    local_batch_size = int(_ns_get(args, "training.local_batch_size", 1) or 1)
    global_batch_size = int(_ns_get(args, "training.global_batch_size", -1) or -1)
    pp_microbatch_size = int(_ns_get(args, "parallelism.pipeline_parallel_microbatch_size", 1) or 1)
    flat.update(
        {
            "seq_length": int(_ns_get(args, "training.seq_len", 0) or 0),
            # Without PP the local batch runs as one forward; with PP it is split
            # into microbatches of pipeline_parallel_microbatch_size.
            "micro_batch_size": pp_microbatch_size if dims["pp"] > 1 else local_batch_size,
            "global_batch_size": (
                global_batch_size if global_batch_size > 0 else local_batch_size * dims["dp"]
            ),
        }
    )

    recompute = _resolve_recompute(args)
    flat.update(recompute)
    flat["recompute_num_layers"] = (
        int(flat["num_layers"]) if recompute["recompute_granularity"] == "full" else 0
    )
    flat["recompute_layer_ids"] = None

    flat.update(_resolve_precision(args))

    deepep = str(_ns_get(args, "parallelism.expert_parallel_comm_backend", "") or "").lower()
    turbo_grouped_gemm = bool(_ns_get(args, "primus_turbo.use_turbo_grouped_gemm", False))
    flat.update(
        {
            "use_flash_attn": True,
            "optimizer": str(_ns_get(args, "optimizer.name", "adamw") or "adamw").lower(),
            "enable_primus_turbo": bool(_ns_get(args, "primus_turbo.enable_primus_turbo", False)),
            "use_turbo_grouped_gemm": turbo_grouped_gemm,
            "use_turbo_grouped_mlp": turbo_grouped_gemm,
            "use_turbo_deepep": deepep == "deepep",
            "turbo_sync_free_moe_stage": 0,
            "cross_entropy_loss_fusion": False,
            "num_layers_per_virtual_pipeline_stage": None,
            "decoder_first_pipeline_num_layers": None,
            "decoder_last_pipeline_num_layers": None,
        }
    )

    for key, value in flat.items():
        setattr(args, key, value)


def torchtitan_derive_default_args(args):
    """Normalize a TorchTitan trainer config into the projection's fields.

    Idempotent: after the first pass the namespace carries the flat fields, and
    later passes only re-derive from those, so the performance driver's edits to
    the normalized config (layer limiting, EP rescale, PP flattening) survive the
    re-conversion it does after making them.
    """
    from primus.core.projection.training_config import megatron_derive_default_args

    if not is_normalized(args):
        _normalize(args)
        mark_normalized(args)
    return megatron_derive_default_args(args)


# ---------------------------------------------------------------------------
# Benchmark write-back: flat projection fields -> TorchTitan's own config
# ---------------------------------------------------------------------------
#
# Normalization runs one way -- TorchTitan's nested config in, flat
# Megatron-spelled fields out -- and the performance driver then edits the flat
# side: it caps the stack at one or two layers, shrinks expert parallelism onto
# the bench node, and flattens the pipeline.  Simulation reads those flat fields
# directly, so for simulate-only backends that is the end of it.
#
# A benchmark has to *build* the model those edits describe, and TorchTitan
# builds from its own namespaces, so the edits have to be written back.  Whatever
# is not written back is silently ignored: a run that believes it profiled two
# layers at EP=1 would in fact have profiled all 61 at EP=8, which is the
# difference between a benchmark that finishes and one that will not fit.


def _ns_set(obj, path: str, value) -> None:
    """Write a dotted path into a nested namespace, creating levels as needed."""
    parts = path.split(".")
    for part in parts[:-1]:
        child = getattr(obj, part, None)
        if child is None:
            child = SimpleNamespace()
            setattr(obj, part, child)
        obj = child
    setattr(obj, parts[-1], value)


def _leading_dense_layers(moe_layer_freq) -> int:
    """Count the dense layers the projection put at the front of the stack."""
    if not isinstance(moe_layer_freq, (list, tuple)):
        return 0
    dense = 0
    for flag in moe_layer_freq:
        if flag:
            break
        dense += 1
    return dense


def torchtitan_apply_bench_overrides(args) -> None:
    """Push the driver's flat bench edits back into TorchTitan's config.

    Mutates *args* in place.  Parallelism degrees, batch shape and activation
    checkpointing live in TorchTitan's own namespaces; the layer count and expert
    count live on the flavor's ``ModelArgs``, which no config field reaches, so
    those are staged under ``primus_projection.model_args_overrides`` for
    :func:`primus.backends.torchtitan.model_builder.build_model_only` to apply
    after ``update_from_config``.
    """
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    moe_layer_freq = getattr(args, "moe_layer_freq", None)

    _ns_set(args, "parallelism.tensor_parallel_degree", int(getattr(args, "tensor_model_parallel_size", 1) or 1))
    _ns_set(args, "parallelism.context_parallel_degree", int(getattr(args, "context_model_parallel_size", 1) or 1))
    _ns_set(args, "parallelism.expert_parallel_degree", int(getattr(args, "expert_model_parallel_size", 1) or 1))
    _ns_set(
        args, "parallelism.pipeline_parallel_degree", int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    )

    # Build the bench model unsharded across data parallelism. FSDP2 and DDP
    # both wrap parameters in DTensors, which the layer benchmark cannot feed
    # plain tensors to, and the projection already models data-parallel gradient
    # reduction analytically when it scales back up to the target cluster. This
    # mirrors the Megatron path, which turns off use_torch_fsdp2 for the same
    # reason.
    _ns_set(args, "parallelism.data_parallel_shard_degree", 1)
    _ns_set(args, "parallelism.data_parallel_replicate_degree", 1)

    micro_batch_size = int(getattr(args, "micro_batch_size", 1) or 1)
    _ns_set(args, "training.local_batch_size", micro_batch_size)
    _ns_set(args, "training.seq_len", int(getattr(args, "seq_length", 0) or 0))
    # One gradient accumulation step: the benchmark measures a single
    # microbatch and the projection composes the rest.
    _ns_set(args, "training.global_batch_size", micro_batch_size)

    granularity = getattr(args, "recompute_granularity", None)
    if granularity == "full":
        _ns_set(args, "activation_checkpoint.mode", "full")
    elif granularity == "selective":
        _ns_set(args, "activation_checkpoint.mode", "selective")

    model_args_overrides = {}
    if num_layers:
        model_args_overrides["n_layers"] = num_layers
    num_experts = getattr(args, "num_experts", None)
    if num_experts:
        model_args_overrides["moe_args.num_experts"] = int(num_experts)
    if moe_layer_freq is not None:
        dense = _leading_dense_layers(moe_layer_freq)
        # DeepSeek V3 counts dense layers from the front; Llama 4 interleaves on
        # a stride. Both are set because only the resolved flavor's ModelArgs
        # knows which one it has, and the builder drops the ones it does not.
        model_args_overrides["n_dense_layers"] = dense
        model_args_overrides["interleave_moe_layer_step"] = 1 if dense == 0 else 2

    _ns_set(args, "primus_projection.model_args_overrides", model_args_overrides)

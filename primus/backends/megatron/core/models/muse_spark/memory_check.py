###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fail fast when the Muse Spark model states cannot fit the GPUs (e.g. the full 816B
model with the single-node example defaults) instead of an out-of-memory error later.

Pure Python: an approximate estimate of the model states on the most loaded GPU:
weights (``params_dtype``: bf16 / fp16 2 bytes, fp32 4), main grads (as Megatron keeps
them: fp32 unless ``main_grads_dtype`` is 16-bit, bf16 with ``grad_reduce_in_bf16`` or
fp16 without fp32 accumulation) and optimizer
states (master params + Adam moments or one SGD momentum, dtypes from the
precision-aware optimizer args, divided over data parallel ranks with the distributed
optimizer, minus the CPU-offloaded fraction). Activations, the TE workspace, RCCL
buffers and allocator fragmentation are not counted, and the parameter count is
approximate (norms and biases are left out). The most loaded pipeline stage is used:
the embedding is on the first stage, the output layer and MTP blocks on the last. It is
an estimate, not a strict bound:
passing does not guarantee a fit, and a layout slightly over the limit can still fit.
"""

import copy
import math
import os

from primus.backends.megatron.core.models.muse_spark.model_types import bool_flag
from primus.backends.megatron.core.models.muse_spark.number_utils import (
    parse_positive_int,
)

__all__ = [
    "DEFAULT_GPUS_PER_NODE",
    "GPUS_PER_NODE_ENV",
    "gpus_per_node",
    "DEFAULT_BUDGET",
    "DEFAULT_MAX_NODES",
    "DEFAULT_VOCAB_SIZE",
    "RAISE_FRACTION",
    "WARN_FRACTION",
    "param_parts",
    "param_split",
    "stage_layers",
    "state_bytes_per_gpu",
    "suggest_layout",
    "vocab_size_of",
    "expert_ffn_size",
    "required_state_bytes_or_raise",
    "skip_requested",
]

# GPUs per node of the MI300X / MI355X nodes (8 OAM GPUs): gpus_per_node()'s default.
DEFAULT_GPUS_PER_NODE = 8
# Where gpus_per_node() looks, in order: the Primus launcher's GPU count, then torchrun's
# per-node process count (a run without the launcher). GPUS_PER_NODE comes first: with
# fewer processes than GPUs (--nproc_per_node=1 debug runs, a single-process conversion)
# LOCAL_WORLD_SIZE is the process count, not the node size.
GPUS_PER_NODE_ENV = ("GPUS_PER_NODE", "LOCAL_WORLD_SIZE")


def gpus_per_node(default: int = DEFAULT_GPUS_PER_NODE) -> int:
    """GPUs per node: the first of ``$GPUS_PER_NODE``, ``$LOCAL_WORLD_SIZE`` that is set
    (not blank), parsed by ``number_utils.parse_positive_int``; ``default`` when neither
    is set.

    A set value that is not a positive int (``"abc"``, ``"8x"``, ``"0"``) raises
    ``ValueError`` naming the variable: it does not fall through to the next variable
    (a malformed ``GPUS_PER_NODE`` under ``--nproc_per_node=1`` would otherwise give the
    process count, 1, and a wrong node-span decision).

    Unlike ``BasePlatform.get_gpus_per_node`` this needs no platform object (it is read
    in arg patches, before the platform is set up)."""
    for name in GPUS_PER_NODE_ENV:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return parse_positive_int(
                raw, default, f"{name}={raw!r} must be a positive integer (GPUs per node)"
            )
    return default


# HF ``vocab_size`` of the released model: used only when the args carry no vocabulary.
DEFAULT_VOCAB_SIZE = 202048
# Model states above this fraction of a GPU leave little room for activations: warn.
WARN_FRACTION = 0.8
# Model states above this fraction of a GPU cannot fit with the TE workspaces, RCCL
# buffers and the smallest activations: raise.
RAISE_FRACTION = 0.95
# suggest_layout keeps the model states within this fraction of a GPU (rest: activations).
DEFAULT_BUDGET = 0.65
# suggest_layout gives up above this node count.
DEFAULT_MAX_NODES = 128
# suggest_layout's search space: EP a power of two up to 256, PP up to 8 (TP is 1 or a node).
EP_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PP_CANDIDATES = (1, 2, 4, 8)


_HALF = {"bf16", "fp16", "bfloat16", "float16", "half"}
# 8-bit optimizer state dtypes (Megatron --exp-avg-dtype / --exp-avg-sq-dtype fp8 keep the
# moments as torch.uint8).
_BYTE = {"fp8", "uint8", "int8", "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"}


def vocab_size_of(a, vocab_size=None) -> int:
    """``vocab_size`` if given, else Megatron's ``padded_vocab_size`` (the rows actually
    allocated, set by the muse_spark_vocab patch), else ``vocab_size``,
    else the released model's 202048."""
    for v in (vocab_size, getattr(a, "padded_vocab_size", None), getattr(a, "vocab_size", None)):
        if v:
            return int(v)
    return DEFAULT_VOCAB_SIZE


def expert_ffn_size(a) -> int:
    """Expert FFN width: ``moe_ffn_hidden_size``, else ``ffn_hidden_size`` (Megatron's own
    fallback). An MoE model with neither raises: a zero width would make the check pass
    without counting the experts."""
    v = getattr(a, "moe_ffn_hidden_size", None) or getattr(a, "ffn_hidden_size", None)
    if not v and int(getattr(a, "num_experts", None) or 0):
        raise ValueError("Muse Spark memory check: moe_ffn_hidden_size (or ffn_hidden_size) is not set")
    return int(v or 0)


def expert_mlp_matrices(a) -> int:
    """Weight matrices per expert MLP: 3 (gate, up, down) for a gated (SwiGLU) MLP, 2 (up,
    down) otherwise. Read from ``swiglu`` (Megatron args) or ``gated_linear_unit``
    (TransformerConfig); with neither set it is the Muse Spark expert, SwiGLU
    (``muse_spark_base.yaml``: ``swiglu: true``)."""
    gated = getattr(a, "swiglu", None)
    if gated is None:
        gated = getattr(a, "gated_linear_unit", None)
    return 2 if gated is False else 3


def param_parts(a, vocab_size=None):
    """Parameter counts by placement from Megatron args ``a`` (linear weights only).

    Keys: ``layer_dense`` / ``layer_expert`` (one decoder layer), ``embedding`` (word
    embedding, first pipeline stage), ``output`` (untied output layer, last stage; with
    tied weights Megatron keeps a copy of the embedding there when PP > 1), ``mtp``
    (``mtp_num_layers`` MTP blocks on the last stage: a layer plus the 2h -> h
    projection each) and ``mtp_expert`` (their experts).

    The latent projections are counted only when ``moe_latent_size`` is set, the output
    gate only with ``attention_output_gate``. An expert has :func:`expert_mlp_matrices`
    weight matrices (3 for SwiGLU).
    """
    h = int(a.hidden_size)
    kv = int(a.kv_channels or h // a.num_attention_heads)
    q = int(a.num_attention_heads) * kv
    # Like core_transformer_config_from_args: without GQA, K/V have one group per head
    # (Megatron's num_query_groups default is ignored then). Unset (None) keeps GQA's count.
    groups = a.num_query_groups if getattr(a, "group_query_attention", None) is not False else None
    g = int(groups or a.num_attention_heads) * kv
    gate = q if getattr(a, "attention_output_gate", False) else 0
    attn = h * (q + gate + 2 * g) + q * h
    latent_size = getattr(a, "moe_latent_size", None)
    n_exp = int(a.num_experts or 0)
    layer_dense = attn + (2 * h * int(latent_size) if latent_size and n_exp else 0) + n_exp * h
    layer_expert = n_exp * expert_mlp_matrices(a) * int(latent_size or h) * expert_ffn_size(a)
    vocab = vocab_size_of(a, vocab_size)
    untied = getattr(a, "untie_embeddings_and_output_weights", False)  # Megatron default: tied
    n_mtp = int(getattr(a, "mtp_num_layers", None) or 0)
    return {
        "layer_dense": layer_dense,
        "layer_expert": layer_expert,
        "embedding": vocab * h,
        "output": vocab * h if untied else 0,
        "mtp": n_mtp * (layer_dense + 2 * h * h),
        "mtp_expert": n_mtp * layer_expert,
    }


def param_split(a, vocab_size=None):
    """(dense, expert) parameter counts of the whole model (one copy of each weight)."""
    p = param_parts(a, vocab_size)
    L = int(a.num_layers)
    dense = L * p["layer_dense"] + p["embedding"] + p["output"] + p["mtp"]
    return dense, L * p["layer_expert"] + p["mtp_expert"]


def stage_layers(a):
    """Decoder layers on each pipeline stage, as Megatron splits them.

    Follows ``decoder_first_pipeline_num_layers`` / ``decoder_last_pipeline_num_layers``
    (the rest spread evenly over the middle stages) and
    ``account_for_embedding_in_pipeline_split`` / ``account_for_loss_in_pipeline_split``
    (the embedding / loss count as one layer of the first / last stage). An uneven split
    rounds up, so the estimate errs high.
    """
    L = int(a.num_layers)
    pp = int(getattr(a, "pipeline_model_parallel_size", 1) or 1)
    if pp == 1:
        return [L]
    first = getattr(a, "decoder_first_pipeline_num_layers", None)
    last = getattr(a, "decoder_last_pipeline_num_layers", None)
    if first is not None or last is not None:
        rest = L - int(first or 0) - int(last or 0)
        middle = pp - (first is not None) - (last is not None)
        if rest < 0 or (middle <= 0 and rest != 0):
            raise ValueError(
                f"Muse Spark memory check: decoder_first/last_pipeline_num_layers "
                f"({first}, {last}) do not split num_layers={L} over pipeline size {pp}"
            )
        each = math.ceil(rest / middle) if middle > 0 else 0
        out = [each] * pp
        if first is not None:
            out[0] = int(first)
        if last is not None:
            out[-1] = int(last)
        return out
    emb = int(bool(getattr(a, "account_for_embedding_in_pipeline_split", False)))
    loss = int(bool(getattr(a, "account_for_loss_in_pipeline_split", False)))
    each = math.ceil((L + emb + loss) / pp)
    out = [each] * pp
    out[0] -= emb
    out[-1] -= loss
    return out


def _stage_params(a, parts):
    """(dense, expert) parameters held by each pipeline stage (whole tensors, before TP /
    EP splits)."""
    pp = len(layers := stage_layers(a))
    out = []
    for i, n in enumerate(layers):
        dense, expert = n * parts["layer_dense"], n * parts["layer_expert"]
        if i == 0:
            dense += parts["embedding"]
        if i == pp - 1:
            # untied: the output layer; tied with PP > 1: a copy of the embedding
            dense += parts["output"] or (parts["embedding"] if pp > 1 else 0)
            dense += parts["mtp"]
            expert += parts["mtp_expert"]
        out.append((dense, expert))
    return out


def _dtype_bytes(v):
    """1 for an 8-bit dtype (``"fp8"``, ``torch.uint8``, ...), 2 for a 16-bit dtype
    (``"bf16"``, ``torch.bfloat16``, ...), 4 for fp32 / unset."""
    if v is None:
        return 4
    name = str(v).replace("torch.", "").lower()
    if name in _BYTE:
        return 1
    return 2 if name in _HALF else 4


def _weight_and_grad_bytes(a):
    """(weight, main grad) bytes per parameter, as Megatron's ``validate_args`` and DDP set them.

    Weight: ``params_dtype`` when set; else 2 under ``--bf16`` / ``--fp16`` or when neither
    flag is present (the Muse Spark default, bf16); 4 when neither flag is true and at
    least one is explicitly False (fp32 params). Main grad: a 16-bit ``main_grads_dtype`` is 2 bytes; fp32 params keep
    fp32 grads; ``--bf16`` keeps fp32 grads unless ``grad_reduce_in_bf16`` (``validate_args``
    turns ``accumulate_allreduce_grads_in_fp32`` on for bf16, so its argparse default False
    is not read here); ``--fp16`` keeps fp32 grads only with
    ``accumulate_allreduce_grads_in_fp32``; with no precision flag the main grad is fp32
    unless ``accumulate_allreduce_grads_in_fp32`` is explicitly False.
    """
    bf16, fp16 = getattr(a, "bf16", None), getattr(a, "fp16", None)
    params_dtype = getattr(a, "params_dtype", None)
    if params_dtype is not None:
        weight = _dtype_bytes(params_dtype)
    elif bf16 or fp16 or (bf16 is None and fp16 is None):
        weight = 2  # 16-bit params; no precision args = the Muse Spark default, bf16
    else:
        weight = 4  # neither flag true, at least one explicitly False: fp32 params
    if _dtype_bytes(getattr(a, "main_grads_dtype", None)) < 4:
        return weight, 2
    if weight == 4:
        return weight, 4
    if fp16 and not bf16:
        return weight, 4 if getattr(a, "accumulate_allreduce_grads_in_fp32", False) else 2
    if bf16:
        return weight, 2 if getattr(a, "grad_reduce_in_bf16", False) else 4
    # 16-bit params with no flag to say which: the fp32-accumulation flag, if any
    return weight, 2 if getattr(a, "accumulate_allreduce_grads_in_fp32", None) is False else 4


def _bytes_per_param(a):
    """(weight + main grad, optimizer states on the GPU) bytes per parameter from the args.

    Defaults (bf16, Adam, fp32 master / moments / main grads, no offload): (2 + 4, 12).
    fp32 params keep no separate master copy: (4 + 4, 8).
    """
    weight, grad = _weight_and_grad_bytes(a)
    pa = bool(getattr(a, "use_precision_aware_optimizer", False))

    def state_bytes(key):  # fp32 (unset) unless the precision-aware optimizer sets a dtype
        return _dtype_bytes(getattr(a, key, None) if pa else None)

    # fp32 params are their own main copy (FP32Optimizer, or the distributed optimizer's
    # fp32 shard groups): no separate master weights
    master = 0 if weight == 4 else state_bytes("main_params_dtype")
    m1, m2 = state_bytes("exp_avg_dtype"), state_bytes("exp_avg_sq_dtype")
    opt = master + (m1 if getattr(a, "optimizer", "adam") == "sgd" else m1 + m2)
    if getattr(a, "optimizer_cpu_offload", False):
        frac = getattr(a, "optimizer_offload_fraction", None)
        opt *= 1.0 - float(1.0 if frac is None else frac)
    return weight + grad, float(opt)


def state_bytes_per_gpu(a, world_size, vocab_size=None, parts=None, per_param=None, stage_params=None):
    """Bytes of weights + grads + optimizer states on the most loaded GPU (approximate).

    Takes the most loaded pipeline stage: the word embedding sits on the first stage,
    the output layer and the MTP blocks on the last (not spread over the stages).

    ``parts``: a precomputed :func:`param_parts` result (it does not depend on the
    layout unless the vocab padding does). ``per_param``: a precomputed
    :func:`_bytes_per_param` result (it does not depend on the layout).
    ``stage_params``: a precomputed :func:`_stage_params` result for ``a``'s pipeline
    layout (it depends on PP, not on TP / EP / the world size).
    """
    tp = int(getattr(a, "tensor_model_parallel_size", 1) or 1)
    pp = int(getattr(a, "pipeline_model_parallel_size", 1) or 1)
    ep = int(getattr(a, "expert_model_parallel_size", 1) or 1)
    etp = int(getattr(a, "expert_tensor_parallel_size", 1) or 1)
    if stage_params is None:
        stage_params = _stage_params(a, parts if parts is not None else param_parts(a, vocab_size))
    # Megatron shards dense optimizer state / FSDP over the data-parallel group *with*
    # context parallel (get_data_parallel_group(with_context_parallel=True)): DP x CP.
    dp = max(1, world_size // (tp * pp))
    edp = max(1, world_size // (ep * etp * pp))
    fsdp = bool(getattr(a, "use_torch_fsdp2", False) or getattr(a, "use_megatron_fsdp", False))
    dist_opt = fsdp or bool(getattr(a, "use_distributed_optimizer", False))
    wg, opt = per_param if per_param is not None else _bytes_per_param(a)  # FSDP shards weights + grads too
    per_dense = (wg / dp if fsdp else wg) + (opt / dp if dist_opt else opt)
    per_expert = (wg / edp if fsdp else wg) + (opt / edp if dist_opt else opt)
    return max(dense / tp * per_dense + expert / (ep * etp) * per_expert for dense, expert in stage_params)


def suggest_layout(
    a, total_bytes, n_gpus_per_node=DEFAULT_GPUS_PER_NODE, budget=DEFAULT_BUDGET, max_nodes=DEFAULT_MAX_NODES
):
    """The smallest node count (and a TP / PP / EP for it) whose model states fit in
    ``budget`` x ``total_bytes`` per GPU (the rest is left for activations), or None.

    Each candidate uses an even pipeline split (the user's first/last-stage layer counts
    and pipeline-split options are dropped: they belong to the user's PP).
    TP in (1, ``n_gpus_per_node``) (within a node), PP dividing ``num_layers`` (up to 8), EP a power of two
    dividing ``num_experts`` and ``world / PP``; the distributed optimizer is assumed. At the
    smallest node count TP 1 is preferred, then the lowest memory.
    """
    L = int(a.num_layers)
    n_exp = int(a.num_experts or 0)
    eps = [e for e in EP_CANDIDATES if not n_exp or n_exp % e == 0]
    pps = [p for p in PP_CANDIDATES if L % p == 0]
    tps = sorted({1, int(n_gpus_per_node)})
    parts = param_parts(a)
    limit = budget * total_bytes
    # one copy with the overrides that do not depend on the candidate; the user's uneven
    # pipeline split is for the user's PP only: each candidate PP uses Megatron's even split
    b = copy.copy(a)
    for k, v in dict(
        expert_tensor_parallel_size=1,
        use_distributed_optimizer=True,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
    ).items():
        setattr(b, k, v)
    per_param = _bytes_per_param(b)  # dtypes / optimizer only: the same for every candidate
    stage_p = {}  # per-stage (dense, expert) params: depend on PP only
    for pp in pps:
        b.pipeline_model_parallel_size = pp
        stage_p[pp] = _stage_params(b, parts)
    for nodes in range(1, max_nodes + 1):
        world = nodes * n_gpus_per_node
        best = None
        for tp in tps:
            for pp in pps:
                if world % (tp * pp):
                    continue
                per_stage = world // pp
                b.tensor_model_parallel_size, b.pipeline_model_parallel_size = tp, pp
                for ep in eps:
                    if per_stage % ep:
                        continue
                    b.expert_model_parallel_size = ep
                    need = state_bytes_per_gpu(b, world, per_param=per_param, stage_params=stage_p[pp])
                    if need <= limit and (best is None or (tp, need) < best[:2]):
                        best = (tp, need, dict(nodes=nodes, world=world, tp=tp, pp=pp, ep=ep))
        if best:
            return {**best[2], "bytes": best[1]}
    return None


# ---- the fail-fast check (the before_train patch and the conversion hook call it) ----
GiB = 2**30  # bytes per GiB, for messages
SKIP_ARG = "muse_spark_skip_memory_check"  # config / args key (muse_spark_base.yaml: false)


# Alternative to more GPUs: cut the model on the command line.
CUT_MODEL_HINT = "cut the model (--num_layers / --num_experts)"


def _layout_hint(args, total_bytes, n_gpus_per_node):
    """A layout for these GPUs, from :func:`suggest_layout` (states within ``DEFAULT_BUDGET``
    of a GPU)."""
    hint = suggest_layout(args, total_bytes, n_gpus_per_node=n_gpus_per_node)
    if hint is None:
        return (
            f"No layout up to {DEFAULT_MAX_NODES} nodes keeps the model states under "
            f"{DEFAULT_BUDGET:.0%} of a GPU: {CUT_MODEL_HINT}"
        )
    return (
        f"Use more GPUs, e.g. >= {hint['nodes']} node(s) of {n_gpus_per_node} such GPUs: PRIMUS_TP={hint['tp']} "
        f"PRIMUS_PP={hint['pp']} PRIMUS_EP={hint['ep']} (~{hint['bytes'] / GiB:.0f} GiB of states per GPU; "
        f"GBS a multiple of DP; recompute helps the activations), or {CUT_MODEL_HINT}"
    )


def skip_requested(args) -> bool:
    """``args.muse_spark_skip_memory_check`` as a bool (:func:`model_types.bool_flag`); an
    unrecognised value (e.g. a typo) raises ValueError."""
    value = getattr(
        args, "muse_spark_skip_memory_check", False
    )  # SKIP_ARG; a literal for the config-schema scan
    return bool_flag(value, SKIP_ARG)


def required_state_bytes_or_raise(args, world_size, total_bytes, n_gpus_per_node=DEFAULT_GPUS_PER_NODE):
    """The estimated weights + grads + optimizer states per GPU (bytes); raise ValueError when
    they exceed ``RAISE_FRACTION`` of ``total_bytes``. The rest is a fixed headroom for the
    TE GEMM workspaces, RCCL buffers and activations, which are not estimated, so only
    layouts that cannot fit raise.

    Uses ``args.padded_vocab_size`` (the allocated embedding / head rows). The layout in
    the error is computed for this GPU's memory (MI300X and MI355X differ).
    """
    need = state_bytes_per_gpu(args, world_size)
    if need > RAISE_FRACTION * total_bytes:
        raise ValueError(
            f"Muse Spark: weights + grads + optimizer states need ~{need / GiB:.0f} GiB per GPU "
            f"(num_layers={args.num_layers}, num_experts={args.num_experts}, "
            f"TP={args.tensor_model_parallel_size}, PP={args.pipeline_model_parallel_size}, "
            f"EP={args.expert_model_parallel_size}, world={world_size}) but a GPU has "
            f"{total_bytes / GiB:.0f} GiB (limit {RAISE_FRACTION:.0%} of it). {_layout_hint(args, total_bytes, n_gpus_per_node)}. "
            f"{SKIP_ARG}: true skips this check."
        )
    return need

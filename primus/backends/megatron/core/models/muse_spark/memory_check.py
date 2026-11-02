###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fail fast when the Muse Spark model states cannot fit the GPUs (e.g. the full 816B
model with the single-node example defaults) instead of an out-of-memory error later.

Pure Python: an approximate estimate of the model states on the most loaded GPU:
weights (bf16), main grads (fp32 unless ``main_grads_dtype`` is 16-bit or
``accumulate_allreduce_grads_in_fp32`` is False) and optimizer
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

__all__ = [
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
]


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
# GPUs per node of the MI300X / MI355X nodes these presets target (8 OAM GPUs).
DEFAULT_GPUS_PER_NODE = 8
# Where gpus_per_node() looks, in order: torchrun's per-node process count, then the
# Primus launcher's GPUS_PER_NODE.
GPUS_PER_NODE_ENV = ("LOCAL_WORLD_SIZE", "GPUS_PER_NODE")
# suggest_layout's search space: EP a power of two up to 256, PP up to 8 (TP is 1 or a node).
EP_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PP_CANDIDATES = (1, 2, 4, 8)


def gpus_per_node(default=DEFAULT_GPUS_PER_NODE, warn=None) -> int:
    """GPUs per node: the first of ``$LOCAL_WORLD_SIZE``, ``$GPUS_PER_NODE`` that is a
    positive int, else ``default``. The one reader for the Muse Spark patches (the DeepEP
    node-span check and the memory-check layout hint). A set but malformed value is
    skipped; ``warn`` (if given) is called with a message naming it."""
    for name in GPUS_PER_NODE_ENV:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
        if warn is not None:
            warn(f"ignoring {name}={raw!r} (not a positive int)")
    return int(default)


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


def param_parts(a, vocab_size=None):
    """Parameter counts by placement from Megatron args ``a`` (linear weights only).

    Keys: ``layer_dense`` / ``layer_expert`` (one decoder layer), ``embedding`` (word
    embedding, first pipeline stage), ``output`` (untied output layer, last stage; with
    tied weights Megatron keeps a copy of the embedding there when PP > 1), ``mtp``
    (``mtp_num_layers`` MTP blocks on the last stage: a layer plus the 2h -> h
    projection each) and ``mtp_expert`` (their experts).

    The latent projections are counted only when ``moe_latent_size`` is set, the output
    gate only with ``attention_output_gate``.
    """
    h = int(a.hidden_size)
    kv = int(a.kv_channels or h // a.num_attention_heads)
    q = int(a.num_attention_heads) * kv
    g = int(a.num_query_groups or a.num_attention_heads) * kv
    gate = q if getattr(a, "attention_output_gate", False) else 0
    attn = h * (q + gate + 2 * g) + q * h
    latent_size = getattr(a, "moe_latent_size", None)
    n_exp = int(a.num_experts or 0)
    layer_dense = attn + (2 * h * int(latent_size) if latent_size and n_exp else 0) + n_exp * h
    layer_expert = n_exp * 3 * int(latent_size or h) * expert_ffn_size(a)
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


def _bytes_per_param(a):
    """(weight + main grad, optimizer states on the GPU) bytes per parameter from the args.

    Defaults (Adam, fp32 master / moments / main grads, no offload): (2 + 4, 12).
    """
    grad = _dtype_bytes(getattr(a, "main_grads_dtype", None))
    if getattr(a, "accumulate_allreduce_grads_in_fp32", None) is False and grad == 4:
        grad = 2  # grads kept in the bf16 param dtype
    pa = bool(getattr(a, "use_precision_aware_optimizer", False))
    master = _dtype_bytes(getattr(a, "main_params_dtype", None)) if pa else 4
    m1 = _dtype_bytes(getattr(a, "exp_avg_dtype", None)) if pa else 4
    m2 = _dtype_bytes(getattr(a, "exp_avg_sq_dtype", None)) if pa else 4
    opt = master + (m1 if getattr(a, "optimizer", "adam") == "sgd" else m1 + m2)
    if getattr(a, "optimizer_cpu_offload", False):
        frac = getattr(a, "optimizer_offload_fraction", None)
        opt *= 1.0 - float(1.0 if frac is None else frac)
    return 2 + grad, float(opt)


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
    if parts is None:
        parts = param_parts(a, vocab_size)
    # Megatron shards dense optimizer state / FSDP over the data-parallel group *with*
    # context parallel (get_data_parallel_group(with_context_parallel=True)): DP x CP.
    dp = max(1, world_size // (tp * pp))
    edp = max(1, world_size // (ep * etp * pp))
    fsdp = bool(getattr(a, "use_torch_fsdp2", False) or getattr(a, "use_megatron_fsdp", False))
    dist_opt = fsdp or bool(getattr(a, "use_distributed_optimizer", False))
    wg, opt = per_param if per_param is not None else _bytes_per_param(a)  # FSDP shards weights + grads too
    per_dense = (wg / dp if fsdp else wg) + (opt / dp if dist_opt else opt)
    per_expert = (wg / edp if fsdp else wg) + (opt / edp if dist_opt else opt)
    return max(
        dense / tp * per_dense + expert / (ep * etp) * per_expert
        for dense, expert in (stage_params if stage_params is not None else _stage_params(a, parts))
    )


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
                    need = state_bytes_per_gpu(
                        b, world, parts=parts, per_param=per_param, stage_params=stage_p[pp]
                    )
                    if need <= limit and (best is None or (tp, need) < (best[1]["tp"], best[0])):
                        best = (need, dict(nodes=nodes, world=world, tp=tp, pp=pp, ep=ep))
        if best:
            return {**best[1], "bytes": best[0]}
    return None

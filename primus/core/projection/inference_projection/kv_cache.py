###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""KV-cache memory model for inference projection.

The KV cache is the defining memory term of LLM serving: every resident
sequence stores per-layer Key/Value tensors that grow with context length.
This module sizes that cache analytically, accounting for:

  * GQA / MQA  — fewer KV heads than query heads.
  * MLA        — a single compressed latent KV per token (DeepSeek V2/V3).
  * Tensor parallelism — KV heads shard across TP ranks.
  * Pipeline parallelism — only the layers on a rank store cache.
  * KV-cache quantization — fp8 / int8 halve (or quarter) the footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

from primus.core.projection.training_config import InferenceConfig, dtype_num_bytes


@dataclass
class KVCacheBreakdown:
    """Per-rank KV-cache sizing for a single sequence and for the batch."""

    bytes_per_token_per_layer: float
    layers_on_rank: int
    bytes_per_token: float          # across all layers on this rank
    bytes_per_sequence: float       # at max_context_len
    bytes_total: float              # across resident concurrency
    max_context_len: int
    concurrency: int
    kv_cache_dtype: str


def _num_kv_heads(model_config) -> int:
    if model_config.group_query_attention and model_config.num_query_groups:
        return int(model_config.num_query_groups)
    return int(model_config.num_attention_heads)


def kv_bytes_per_token_per_layer(inference_config: InferenceConfig) -> float:
    """Bytes of KV cache stored per token, per transformer layer, *per rank*."""
    mc = inference_config.model_config
    mp = inference_config.model_parallel_config
    tp = max(1, mp.tensor_model_parallel_size)
    kv_bytes = dtype_num_bytes(inference_config.request_config.kv_cache_dtype)

    if mc.multi_latent_attention:
        # MLA caches a single compressed latent per token (kv_lora_rank) plus
        # the decoupled RoPE key (qk_pos_emb_head_dim).  This latent is shared
        # across heads, so it is *not* sharded by TP heads.
        latent = int(mc.kv_lora_rank or 0) + int(mc.qk_pos_emb_head_dim or 0)
        return latent * kv_bytes

    # Standard MHA / GQA: K and V each store (kv_heads_per_rank * head_dim).
    kv_heads = _num_kv_heads(mc)
    kv_heads_per_rank = max(1, kv_heads // tp)
    head_dim = int(mc.kv_channels)
    return 2.0 * kv_heads_per_rank * head_dim * kv_bytes


def estimate_kv_cache(
    inference_config: InferenceConfig,
    layers_on_rank: int,
    *,
    concurrency: int | None = None,
    context_len: int | None = None,
) -> KVCacheBreakdown:
    """Size the KV cache on one rank.

    Args:
        inference_config: The serving configuration.
        layers_on_rank: Number of transformer layers hosted by this rank
            (``num_layers / pp`` for an even split).
        concurrency: Number of resident sequences.  Defaults to the request
            config's ``max_concurrency`` (which itself defaults to
            ``batch_size``).
        context_len: Context length to size each sequence at.  Defaults to
            ``max_context_len``.
    """
    req = inference_config.request_config
    if concurrency is None:
        concurrency = req.resolved_max_concurrency()
    if context_len is None:
        context_len = req.resolved_max_context_len()

    per_token_per_layer = kv_bytes_per_token_per_layer(inference_config)
    per_token = per_token_per_layer * max(1, layers_on_rank)
    per_sequence = per_token * context_len
    total = per_sequence * concurrency

    return KVCacheBreakdown(
        bytes_per_token_per_layer=per_token_per_layer,
        layers_on_rank=int(layers_on_rank),
        bytes_per_token=per_token,
        bytes_per_sequence=per_sequence,
        bytes_total=total,
        max_context_len=int(context_len),
        concurrency=int(concurrency),
        kv_cache_dtype=req.kv_cache_dtype,
    )


def max_concurrent_sequences(
    inference_config: InferenceConfig,
    layers_on_rank: int,
    free_bytes_for_kv: float,
    *,
    context_len: int | None = None,
) -> int:
    """How many sequences fit in ``free_bytes_for_kv`` at ``context_len``.

    Useful for answering "how many concurrent requests can this config
    serve?" given HBM left over after weights + activation working set.
    """
    one = estimate_kv_cache(
        inference_config, layers_on_rank, concurrency=1, context_len=context_len
    )
    if one.bytes_per_sequence <= 0:
        return 0
    return int(max(0, free_bytes_for_kv) // one.bytes_per_sequence)

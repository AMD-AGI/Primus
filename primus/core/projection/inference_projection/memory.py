###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Inference memory projection.

Per-rank HBM for serving is dominated by two terms that training does *not*
share:

  * **Weights only** — no gradients, no fp32 master copy, no Adam moments.
  * **KV cache** — grows with resident concurrency × context length.

plus a (comparatively small) **forward activation working set** for the
in-flight batch.  We also report how many concurrent sequences fit in the
remaining HBM, which is the headline capacity number for a serving config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from primus.core.projection.module_profilers.language_model import (
    build_profiler,
    get_language_model_profiler_spec,
)
from primus.core.projection.training_config import InferenceConfig, dtype_num_bytes

from .kv_cache import KVCacheBreakdown, estimate_kv_cache, max_concurrent_sequences


@dataclass
class InferenceMemoryResult:
    rank: int
    num_params: int
    weight_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    total_bytes: int
    kv: KVCacheBreakdown
    layers_on_rank: int
    hbm_capacity_bytes: Optional[int] = None
    max_concurrent_sequences: Optional[int] = None
    fits: Optional[bool] = None


_GB = 1024.0 ** 3


def _layers_on_rank(inference_config: InferenceConfig) -> int:
    mc = inference_config.model_config
    pp = max(1, inference_config.model_parallel_config.pipeline_model_parallel_size)
    return max(1, (mc.num_layers + pp - 1) // pp)


def _forward_activation_bytes(inference_config: InferenceConfig, profiler) -> int:
    """A bounded estimate of the transient forward activation working set.

    Inference keeps no activations for backward, so only a few in-flight
    layers' worth of hidden states exist at once.  We proxy this with one
    transformer layer's activation footprint at the prefill token count,
    scaled by a small number of in-flight layers.
    """
    req = inference_config.request_config
    batch = max(1, req.batch_size)
    chunk = int(req.chunked_prefill_size or 0)
    tokens = chunk if 0 < chunk < req.input_seq_len else req.input_seq_len

    layer = profiler.sub_profilers.get("dense_transformer_layer")
    moe_layer = profiler.sub_profilers.get("moe_transformer_layer")
    chosen = layer
    # Prefer whichever layer type actually exists in the model.
    if (inference_config.model_config.num_experts or 0) and moe_layer is not None:
        chosen = moe_layer
    if chosen is None:
        return 0
    per_layer = chosen.estimated_activation_memory(batch, tokens)
    inflight_layers = min(_layers_on_rank(inference_config), 2)
    return int(per_layer * inflight_layers)


def project_inference_memory(
    inference_config: InferenceConfig,
    *,
    rank: Optional[int] = None,
    hbm_capacity_gb: Optional[float] = None,
    verbose: bool = True,
) -> InferenceMemoryResult:
    eff_rank = int(os.getenv("RANK", "0")) if rank is None else int(rank)

    view = inference_config.as_training_config(
        batch_size=inference_config.request_config.batch_size,
        seq_len=inference_config.request_config.input_seq_len,
    )
    profiler = build_profiler(get_language_model_profiler_spec(view))

    num_params = profiler.estimated_num_params(rank=eff_rank)
    weight_bytes = int(num_params * dtype_num_bytes(inference_config.request_config.weight_dtype))

    layers_on_rank = _layers_on_rank(inference_config)
    kv = estimate_kv_cache(inference_config, layers_on_rank)
    activation_bytes = _forward_activation_bytes(inference_config, profiler)

    total = weight_bytes + int(kv.bytes_total) + activation_bytes

    hbm_bytes = int(hbm_capacity_gb * _GB) if hbm_capacity_gb else None
    max_conc = None
    fits = None
    if hbm_bytes is not None:
        # Serving engines only hand a fraction of HBM to the runtime (vLLM
        # gpu_memory_utilization / SGLang mem_fraction_static); the rest is
        # reserved for the driver/CUDA context and fragmentation headroom.
        fraction = inference_config.request_config.kv_cache_memory_fraction
        usable_bytes = int(hbm_bytes * float(fraction)) if fraction else hbm_bytes
        free_for_kv = usable_bytes - weight_bytes - activation_bytes
        max_conc = max_concurrent_sequences(inference_config, layers_on_rank, free_for_kv)
        fits = total <= usable_bytes

    result = InferenceMemoryResult(
        rank=eff_rank,
        num_params=int(num_params),
        weight_bytes=weight_bytes,
        kv_cache_bytes=int(kv.bytes_total),
        activation_bytes=activation_bytes,
        total_bytes=total,
        kv=kv,
        layers_on_rank=layers_on_rank,
        hbm_capacity_bytes=hbm_bytes,
        max_concurrent_sequences=max_conc,
        fits=fits,
    )

    if verbose:
        _print_memory(inference_config, result)
    return result


def _print_memory(inference_config: InferenceConfig, r: InferenceMemoryResult) -> None:
    req = inference_config.request_config
    print("\n" + "=" * 100)
    print(f"[Primus:Inference] Memory Projection (Rank {r.rank})")
    print("=" * 100)
    print(f"  Params (this rank):       {r.num_params / 1e9:.4f} B")
    print(f"  Weights ({req.weight_dtype}):           {r.weight_bytes / _GB:.4f} GB")
    print(
        f"  KV cache ({req.kv_cache_dtype}):         {r.kv_cache_bytes / _GB:.4f} GB "
        f"(concurrency={r.kv.concurrency}, ctx={r.kv.max_context_len}, "
        f"layers/rank={r.layers_on_rank})"
    )
    print(f"    KV per sequence:        {r.kv.bytes_per_sequence / _GB:.4f} GB")
    print(f"  Activation working set:   {r.activation_bytes / _GB:.4f} GB")
    print(f"  Projected Total Memory:   {r.total_bytes / _GB:.4f} GB")
    if r.hbm_capacity_bytes is not None:
        print(f"  HBM capacity:             {r.hbm_capacity_bytes / _GB:.4f} GB")
        if req.kv_cache_memory_fraction:
            usable = r.hbm_capacity_bytes * float(req.kv_cache_memory_fraction)
            print(
                f"  Usable HBM (frac={req.kv_cache_memory_fraction:.2f}): {usable / _GB:.4f} GB"
            )
        print(f"  Fits:                     {r.fits}")
        print(f"  Max concurrent sequences: {r.max_concurrent_sequences}")
    print("=" * 100)

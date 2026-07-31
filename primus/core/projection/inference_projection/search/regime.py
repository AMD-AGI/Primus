###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Regime signature + config keying — the canonical parameter split.

Every inference recipe parameter is one of two kinds (see the design note §2):

* **regime-defining** — swaps the kernel/execution regime (dtype → different
  GEMM/attention kernels, backend, cudagraph/AITER path).  Two recipes that
  differ on any of these are *not* transportable from one another; each regime
  needs its own measured anchor.
* **transportable** — moves cost analytically from an existing measurement
  (layers, TP/EP/PP, batch, sequence length, concurrency, arrival rate).

This module is the single source of truth for that split.  It is deliberately
**stdlib-only** so the dependency-light ``benchmark_vllm.py`` (which runs inside
a bare vLLM container without Primus) can import it to key its result cache with
the exact same scheme the anchor store uses.

Public API
----------
``regime_signature(recipe)``  -> 16-hex hash over the regime axes only.
``config_key(recipe, extra)`` -> 16-hex hash over regime + transport (+extra);
                                 an exact-run identity (used by the bench cache).
``regime_distance(a, b)``     -> Hamming distance over the regime axes.
``recipe_from_meta(meta)``    -> canonical recipe dict from an artifact ``meta``.
``recipe_from_bench_args(args, env)`` -> canonical recipe from benchmark CLI args.
``recipe_from_inference_config(cfg)`` -> canonical recipe from an InferenceConfig.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Optional

# Regime-defining axes: differ on any of these → a different measured anchor.
REGIME_AXES = (
    "model",
    "weight_dtype",
    "kv_cache_dtype",
    "moe_expert_dtype",
    "attention_backend",
    "cudagraph",
    "aiter",
)

# Transportable axes: reconstructed analytically from an anchor in the same
# regime (the projector's restore + interpolation already implement these).
TRANSPORT_AXES = (
    "tp",
    "pp",
    "ep",
    "num_layers",
    "batch",
    "input_len",
    "output_len",
    "concurrency",
    "request_rate",
)


def _canon(v: Any) -> str:
    """Canonical, comparison-stable string for a single axis value."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        # Trim trailing zeros so 8.0 and 8 compare equal.
        if v == int(v):
            return str(int(v))
        return repr(round(v, 6))
    if isinstance(v, (int,)):
        return str(v)
    return str(v).strip().lower()


def _sig(recipe: Dict[str, Any], axes: Iterable[str]) -> str:
    payload = {k: _canon(recipe.get(k)) for k in axes}
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def regime_signature(recipe: Dict[str, Any]) -> str:
    """Hash over the regime-defining axes only.  Two recipes with the same
    signature are mutually transportable (same kernels/regime)."""
    return _sig(recipe, REGIME_AXES)


def config_key(recipe: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    """Exact-run identity: hash over regime + transport axes, plus any ``extra``
    (measurement knobs that change the number but not the regime, e.g.
    decode-steps).  Used by the benchmark result cache."""
    payload = {k: _canon(recipe.get(k)) for k in (*REGIME_AXES, *TRANSPORT_AXES)}
    if extra:
        for k, v in extra.items():
            payload[f"x_{k}"] = _canon(v)
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def regime_distance(
    a: Dict[str, Any], b: Dict[str, Any], *, ignore_missing: bool = True
) -> int:
    """Hamming distance over the regime axes.  0 => same regime (fully
    transportable).  When ``ignore_missing`` (default), an axis absent/None on
    *either* side is not counted — so a partially-specified target still matches
    an anchor on the axes both actually pin (e.g. an anchor with no
    ``attention_backend`` is not penalised against a target that sets one)."""
    d = 0
    for k in REGIME_AXES:
        av, bv = a.get(k), b.get(k)
        if ignore_missing and (av is None or bv is None or av == "" or bv == ""):
            continue
        if _canon(av) != _canon(bv):
            d += 1
    return d


# --------------------------------------------------------------------------
# Adapters: build a canonical recipe from the three sources that produce one.
# --------------------------------------------------------------------------

def recipe_from_meta(meta: Dict[str, Any], *, model: Optional[str] = None) -> Dict[str, Any]:
    """Canonical recipe from a benchmark artifact's ``meta`` block.  The
    *benchmark* parallelism (what it actually ran at) is recorded on the
    transport axes so the anchor's coverage is described in benchmark space;
    the restore to the target happens at reconstruction time."""
    quant = meta.get("quantization")
    return {
        "model": model or meta.get("model"),
        # A null quantization / kv-dtype means "model default" which is bf16 for
        # an un-quantized checkpoint; canonicalize to "bf16" so it matches a
        # config that spells the same regime as bf16 (avoids false mismatches).
        "weight_dtype": quant if quant else "bf16",
        "kv_cache_dtype": meta.get("kv_cache_dtype") or "bf16",
        "moe_expert_dtype": meta.get("moe_expert_dtype"),
        "attention_backend": meta.get("attention_backend"),
        "cudagraph": "eager" if meta.get("enforce_eager") else "graph",
        "aiter": bool(meta.get("use_aiter")),
        # transport (benchmark space)
        "tp": meta.get("benchmark_tp") or meta.get("tp"),
        "pp": meta.get("benchmark_pp") or meta.get("pp"),
        "ep": meta.get("benchmark_ep") or meta.get("ep"),
        "num_layers": meta.get("num_hidden_layers"),
        "batch": meta.get("batch"),
        "input_len": meta.get("input_len"),
        "output_len": meta.get("output_len"),
    }


def recipe_from_bench_args(args: Any, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Canonical recipe from ``benchmark_vllm.py`` CLI args (stdlib-only path).

    Also returns nothing extra here — measurement-only knobs are passed as the
    ``extra`` dict to :func:`config_key` by the caller so the cache stays exact
    without polluting the regime/transport axes."""
    env = env or {}
    aiter_on = env.get("VLLM_ROCM_USE_AITER", "0") == "1" and not getattr(args, "no_aiter", False)
    ep = int(getattr(args, "tp", 1) or 1) if getattr(args, "enable_expert_parallel", False) else 1
    return {
        "model": getattr(args, "model", None),
        "weight_dtype": getattr(args, "quantization", None) or "bf16",
        "kv_cache_dtype": getattr(args, "kv_cache_dtype", None) or "bf16",
        "moe_expert_dtype": None,
        "attention_backend": None,
        "cudagraph": "eager" if getattr(args, "enforce_eager", False) else "graph",
        "aiter": aiter_on,
        "tp": getattr(args, "tp", 1),
        "pp": getattr(args, "pp", 1),
        "ep": ep,
        "num_layers": getattr(args, "num_hidden_layers", None),
        "batch": getattr(args, "batch", None),
        "input_len": getattr(args, "input_len", None),
        "output_len": getattr(args, "output_len", None),
    }


def recipe_from_inference_config(cfg: Any) -> Dict[str, Any]:
    """Canonical recipe from an ``InferenceConfig`` (the reconstruction target).

    Structural configs carry no HF model *name*, so ``model`` is left ``None``
    and matching is expected within a per-model anchor store (the common case:
    you harvest anchors for the model you are tuning).  ``num_layers`` is the
    full target depth (restore extrapolates the anchor's reduced depth to it)."""
    req = getattr(cfg, "request_config", None)
    mc = getattr(cfg, "model_config", None)
    mp = getattr(cfg, "model_parallel_config", None)

    def g(o, name, default=None):
        return getattr(o, name, default) if o is not None else default

    tp = int(g(mp, "tensor_model_parallel_size", 1) or 1)
    ep = int(g(mp, "expert_model_parallel_size", 1) or 1)
    return {
        "model": None,
        "weight_dtype": g(req, "weight_dtype", "bf16"),
        "kv_cache_dtype": g(req, "kv_cache_dtype", "bf16"),
        "moe_expert_dtype": g(req, "moe_expert_dtype"),
        "attention_backend": g(req, "attention_backend"),
        "cudagraph": _cudagraph_from_mode(g(req, "cudagraph_mode")),
        "aiter": None,  # not represented on the config side; ignored in distance
        "tp": tp,
        "pp": int(g(mp, "pipeline_model_parallel_size", 1) or 1),
        "ep": ep,
        "num_layers": g(mc, "num_layers"),
        "batch": g(req, "batch_size"),
        "input_len": g(req, "input_seq_len"),
        "output_len": g(req, "output_seq_len"),
        "concurrency": g(req, "max_concurrency"),
        "request_rate": g(req, "request_rate"),
    }


def _cudagraph_from_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    return "eager" if str(mode).lower() in ("none", "off", "eager") else "graph"

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Simulated cost of the sequence mixer in a hybrid (``M``) layer.

Hybrid stacks (Zebra-Llama and friends) replace softmax attention with a
linear-time mixer in most layers:

* ``gdn``   -- Gated DeltaNet (FLA ``chunk_gated_delta_rule``)
* ``kda``   -- Kimi Delta Attention (FLA delta rule with per-channel decay)
* ``mamba`` -- Mamba2 SSD (``mamba_ssm`` chunked scan)

Each mixer is priced as three parts:

1. Projection GEMMs (fused norm + ``in_proj``, ``out_proj`` and KDA's low-rank
   gate projections), fwd + dgrad + wgrad, from the GEMM backend.
2. Element-wise streaming (short causal conv, activations, q/k L2 norm, gates,
   gated output norm) at the same sustained HBM fraction as the other
   element-wise terms.  The layer's pre-norm and residual are priced by the
   caller, as for every other hybrid layer kind.
3. The chunked scan.  Its FLOPs are counted per chunk (intra-chunk ``C x C``
   products, WY / SSD transforms, state read and update) and run at a sustained
   fraction of the fp32 vector peak.  These Triton kernels are latency- and
   vector-bound far below the matrix peak; the fractions are fitted to the
   per-layer MI325X benchmarks of zebra_llama GDN-1B, KDA-1B and Mamba-1B/3B/8B
   (each within ~9%) and scale with CU count and clock on other parts.
"""

import warnings
from typing import Tuple

from primus.core.projection.ceiling import at_ceiling
from primus.core.projection.training_config import gemm_dtype_from_config

from .moe_mlp import _ACTIVATION_BW_FRACTION, _FALLBACK_HBM_BW_GBPS

_CHUNK = 64
_ACT_BYTES = 2  # bf16 activations
# Used only when the GEMM backend reports no vector peak (no arch profile):
# MI300X/MI325X, 304 CUs x 2.1 GHz x 256 fp32 FLOP/clk.
_FALLBACK_VECTOR_FLOPS = 304 * 2.1e9 * 256
# Dense matrix peak (TFLOP/s) for the roofline used when there is no GEMM backend.
_FALLBACK_PEAK_TFLOPS = {"bf16": 1307.0, "fp8": 2615.0, "mx8": 2615.0}

# Effective scan FLOP rate as a multiple of the fp32 vector peak, (fwd, bwd),
# fitted to the MI325X per-layer benchmarks named in the module docstring. The
# delta-rule kernels run on the vector units well below peak. The SSD forward
# runs its chunk products on the matrix cores, so its rate exceeds the vector
# peak (3.18x); its backward is vector-bound again.
_SCAN_EFF = {
    "delta": (0.153, 0.175),  # FLA chunk delta rule (GDN, KDA)
    "ssd": (3.18, 0.685),  # mamba_ssm SSD
}
_MIXERS = ("gdn", "kda", "mamba")
# Backward recomputes the forward chunk states, then runs the VJP.
_SCAN_BWD_FLOPS = 3.0

_DEFAULTS = {
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "mamba_state_dim": 128,
    "mamba_head_dim": 64,
    "mamba_num_groups": 8,
    "mamba_expand": 2,
}


def _cfg(model_config, name):
    value = getattr(model_config, name, None)
    return _DEFAULTS[name] if value in (None, 0) else int(value)


def mixer_kind(spec, explicit=None) -> str:
    """``gdn`` / ``kda`` / ``mamba``: *explicit* if set, else from the hybrid stack spec name."""
    if explicit:
        kind = str(explicit).lower()
        if kind not in _MIXERS:
            raise ValueError(f"hybrid_mixer must be one of {_MIXERS}, got {explicit!r}")
        return kind
    text = " ".join(spec) if isinstance(spec, (list, tuple)) else str(spec or "")
    text = text.lower()
    if "gdn" in text or "gated_delta" in text:
        return "gdn"
    if "kda" in text or "kimi" in text:
        return "kda"
    if text and not any(w in text for w in ("mamba", "hybrid", "ssm")):
        warnings.warn(
            f"hybrid stack spec {text!r} names no known mixer; pricing it as Mamba2 (set hybrid_mixer to override)",
            stacklevel=2,
        )
    return "mamba"


def _per_rank(n: int, tp: int) -> int:
    """Heads (or groups) on one TP rank; a rank always holds at least one."""
    return max(1, -(-int(n) // max(1, int(tp))))


def _shapes(model_config, tp: int):
    """(in_proj width, extra GEMMs as (n, k), element-wise width, scan family, scan FLOPs / token)."""
    hidden = model_config.hidden_size
    kind = getattr(model_config, "hybrid_mixer", None) or "mamba"
    if kind in ("gdn", "kda"):
        dk, dv = _cfg(model_config, "linear_key_head_dim"), _cfg(model_config, "linear_value_head_dim")
        nk = _per_rank(_cfg(model_config, "linear_num_key_heads"), tp)
        nv = _per_rank(_cfg(model_config, "linear_num_value_heads"), tp)
        qk, v = nk * dk, nv * dv
        conv = 2 * qk + v
        if kind == "gdn":
            n_in = 2 * qk + 2 * v + 2 * nv
            extra = [(hidden, v)]
        else:
            n_in = 2 * qk + v + 2 * dv + nv
            extra = [(hidden, v), (nv * dk, dv), (v, dv)]
        # Per head and token: K K^T, Q K^T, W = T K (3 x C*dk), T V and the
        # intra-chunk output (2 x C*dv), state read for q / w and update (3 x dk*dv).
        flops = nv * 2 * (3 * _CHUNK * dk + 2 * _CHUNK * dv + 3 * dk * dv)
        return n_in, extra, n_in + 2 * conv + 3 * v, "delta", flops
    head = _cfg(model_config, "mamba_head_dim")
    heads = _per_rank(_cfg(model_config, "mamba_expand") * hidden // head, tp)
    d_inner = heads * head
    state = _cfg(model_config, "mamba_state_dim")
    groups = _per_rank(_cfg(model_config, "mamba_num_groups"), tp)
    n_in = 2 * d_inner + 2 * groups * state + heads
    conv = d_inner + 2 * groups * state
    # Per head and token: C B^T, chunk output, B X (2 x C*N, 2 x C*P), state pass (3 x N*P).
    flops = heads * 2 * (2 * _CHUNK * state + 2 * _CHUNK * head + 3 * state * head)
    return n_in, [(hidden, d_inner)], n_in + 2 * conv + 3 * d_inner, "ssd", flops


def _one_gemm_ms(gemm_backend, m: int, n: int, k: int, dtype: str) -> float:
    if gemm_backend is not None:
        return gemm_backend.simulate_gemm(m, n, k, dtype).forward_time_ms
    bpe = 1 if dtype in ("fp8", "mx8") else 2
    flops_ms = 2.0 * m * n * k / (_FALLBACK_PEAK_TFLOPS.get(dtype, 1307.0) * 1e9)
    bytes_ms = (m * k + k * n + m * n) * bpe / (_FALLBACK_HBM_BW_GBPS * 1e6)
    return max(flops_ms, bytes_ms)


def _gemm_ms(gemm_backend, m: int, n: int, k: int, dtype: str = "bf16") -> Tuple[float, float]:
    """A linear layer: forward, and dgrad + wgrad backward."""
    fwd = _one_gemm_ms(gemm_backend, m, n, k, dtype)
    dgrad = _one_gemm_ms(gemm_backend, m, k, n, dtype)
    wgrad = _one_gemm_ms(gemm_backend, k, n, m, dtype)
    return fwd, dgrad + wgrad


def mixer_time_ms(model_config, tokens: int, gemm_backend, tp: int = 1) -> Tuple[float, float]:
    """(fwd_ms, bwd_ms) of one hybrid mixer layer over ``tokens`` tokens on one rank.

    At the fusion ceiling the element-wise passes (conv, activations, gates,
    gated norm) are fused into the neighbouring kernels and cost nothing.
    """
    hidden = model_config.hidden_size
    n_in, extra, ew_width, family, flops_per_token = _shapes(model_config, tp)
    dtype = gemm_dtype_from_config(model_config)

    fwd, bwd = _gemm_ms(gemm_backend, tokens, n_in, hidden, dtype)
    for n, k in extra:
        f, b = _gemm_ms(gemm_backend, tokens, n, k, dtype)
        fwd, bwd = fwd + f, bwd + b

    if not at_ceiling("fusion"):
        hbm = getattr(gemm_backend, "hbm_bandwidth_gbps", None) or _FALLBACK_HBM_BW_GBPS
        ew_fwd = tokens * _ACT_BYTES * ew_width / (hbm * _ACTIVATION_BW_FRACTION * 1e6)
        fwd += ew_fwd
        bwd += 2.0 * ew_fwd

    vector = gemm_backend.vector_flops() if hasattr(gemm_backend, "vector_flops") else None
    vector = vector or _FALLBACK_VECTOR_FLOPS
    eff_fwd, eff_bwd = _SCAN_EFF[family]
    scan_flops = flops_per_token * tokens
    fwd += scan_flops / (eff_fwd * vector) * 1e3
    bwd += _SCAN_BWD_FLOPS * scan_flops / (eff_bwd * vector) * 1e3
    return fwd, bwd


def mixer_activation_bytes(model_config, tokens: int, tp: int = 1) -> int:
    """Saved activations: the in_proj output, the conv output and the mixer output."""
    n_in, _extra, _ew, _family, _flops = _shapes(model_config, tp)
    return int(tokens * _ACT_BYTES * (2 * n_in + model_config.hidden_size))

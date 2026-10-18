###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``flash-linear-attention`` (``fla``) backend for Kimi Delta Attention.

Thin adapter over ``fla.ops.kda.chunk_kda`` that pins the argument
contract to the one :mod:`.._eager.reference` implements, so the two are
interchangeable at a call site.

Two version hazards are handled here rather than at the call site.

``use_beta_sigmoid_in_kernel``
    The upstream HF ``modeling_kimi_linear.py`` passes
    ``use_beta_sigmoid_in_kernel=True`` and hands ``chunk_kda`` a **raw**
    ``b_proj(x)``. No released ``fla`` (0.4.2 included) declares that
    keyword: ``chunk_kda`` swallows it via ``**kwargs`` and silently
    leaves ``beta`` un-activated. Passing an un-sigmoided ``beta`` is not
    a tolerance-level difference — it changes the write strength's range
    from ``(0, 1)`` to all of ``R``. This adapter therefore takes
    ``beta`` **already sigmoid-activated** (as the eager reference does)
    and never forwards the flag.

``transpose_state_layout``
    A deprecated alias for ``state_v_first`` that emits a
    ``DeprecationWarning``. Not forwarded; the returned state keeps
    ``chunk_kda``'s documented ``[N, H, K, V]`` layout, matching the
    eager reference.

The gate (``A_log`` / ``dt_bias`` / bound) is likewise applied by the
caller, so ``g`` arrives as a plain log-decay tensor. That keeps the gate
math in one auditable place — :func:`..._eager.reference.kda_gate` —
instead of being duplicated in every backend.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from fla.ops.kda import chunk_kda as _fla_chunk_kda

__all__ = ["fla_chunk_kda"]

logger = logging.getLogger(__name__)

# MI300X (gfx942) has 64 KB of LDS. FLA's CP kernels autotune num_stages in
# {2, 3, 4} and BV in {32, 64}, and the first of those asks for 80 KB.
_CP_LDS_PATCHED = False


def _clamp_fla_cp_lds() -> None:
    """Keep FLA context-parallel kernels inside 64 KB of shared memory."""
    global _CP_LDS_PATCHED
    if _CP_LDS_PATCHED:
        return
    _CP_LDS_PATCHED = True

    import triton
    from fla.ops.cp import chunk_delta_h

    clamped = []
    for name, wrapped in vars(chunk_delta_h).items():
        # @heuristics wraps @autotune, so the configs live on the inner tuner.
        fn = wrapped
        configs = None
        for _ in range(4):
            found = getattr(fn, "configs", None)
            if isinstance(found, list) and found:
                configs = found
                break
            fn = getattr(fn, "fn", None)
            if fn is None:
                break
        if configs is None:
            continue
        safe = []
        for cfg in configs:
            stages = int(getattr(cfg, "num_stages", 1) or 1)
            kwargs = dict(getattr(cfg, "kwargs", {}) or {})
            bv = int(kwargs.get("BV", 32))
            if stages <= 1 and bv <= 32:
                safe.append(cfg)
        if not safe:
            sample = configs[0]
            kwargs = dict(getattr(sample, "kwargs", {}) or {})
            if "BV" in kwargs:
                kwargs["BV"] = 32
            warps = int(getattr(sample, "num_warps", 4) or 4)
            safe = [triton.Config(kwargs, num_warps=min(warps, 4), num_stages=1)]
        try:
            configs[:] = safe
        except TypeError:
            fn.configs = safe
        cache = getattr(fn, "cache", None)
        if isinstance(cache, dict):
            cache.clear()
        clamped.append(name)
    if clamped:
        logger.info("Clamped FLA CP kernels to 64 KB LDS: %s", ", ".join(sorted(clamped)))


def fla_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.Tensor] = None,
    cp_context: Optional[object] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run KDA through ``fla``'s fused Triton chunk kernel.

    Signature-compatible with
    :func:`..._eager.reference.eager_chunk_kda`. ``g`` must already be a
    log-decay tensor and ``beta`` must already be sigmoid-activated.

    ``chunk_size`` is accepted for signature parity and validated only:
    ``fla`` fixes its own chunk tiling internally.
    """
    if chunk_size != 64:
        raise ValueError(
            f"fla's chunk_kda fixes its chunk tiling internally; chunk_size={chunk_size} "
            "cannot be honoured. Use the eager backend to vary chunk_size."
        )
    packed = {}
    if cu_seqlens is not None:
        packed["cu_seqlens"] = cu_seqlens
    if cp_context is not None:
        _clamp_fla_cp_lds()
        packed["cp_context"] = cp_context
    o, final_state = _fla_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None if initial_state is None else initial_state.float(),
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=False,
        **packed,
    )
    return o, final_state

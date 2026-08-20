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

from typing import Optional, Tuple

import torch
from fla.ops.kda import chunk_kda as _fla_chunk_kda

__all__ = ["fla_chunk_kda"]


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
    )
    return o, final_state

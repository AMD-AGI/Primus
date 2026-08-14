###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The chunk's decay-weighted operand preparation, as one kernel plus its adjoint.

Profiling left the production forward at 2749 µs on device with **half of
it torch elementwise glue**. The largest connected
piece was six ops producing the sweep's state-independent operands:

    Γ = exp(cg);  QΓ = q ⊙ Γ;  KΓ = k ⊙ Γ;  KG = k ⊙ exp(ct − cg);  dec = exp(ct)

each a full HBM round trip on a 100–200 MB fp32 tensor for one ``exp`` of real
arithmetic. :mod:`.kda_chunk_prep_kernel` does all of it in one coalesced pass.

Autograd cannot see through a custom kernel, so the adjoint is written out here
rather than recovered by recomputation. With ``ct[d] = cg[C−1, d]``,
``G = exp(cg)`` and ``E = exp(ct − cg)``, and writing ``A = dKΓ ⊙ G``,
``B = dKG ⊙ E``::

    dq  = dQΓ ⊙ G
    dk  = A + B
    dcg = q ⊙ dq + k ⊙ (A − B)
    dct = Σ_r (k ⊙ B)[r] + d_dec ⊙ dec        and  dcg[C−1] += dct

Each line is the chain rule on one of the four outputs: ``∂QΓ/∂cg = QΓ``,
``∂KΓ/∂cg = KΓ`` and ``∂KG/∂cg = −KG`` because the exponent enters with the
opposite sign there, and ``ct`` is a *row* of ``cg`` rather than an input of its
own, so its gradient lands back on the last row.

``G`` and ``E`` are recomputed in the backward rather than saved. Saving them
would be 400 MB of extra residency per layer, and recovering them by division
(``G = KΓ / k``) is exactly the kind of thing this codebase does not do — ``k``
has zeros and ``G`` underflows to zero at ``cg = −320``.

:func:`chunk_prep_torch` is the pure-torch twin: same operations in the same
order, so it is both the unit-test oracle for the kernel and the fallback for
geometries the kernel cannot take. Same discipline as
:func:`..ops.decay_scores_torch` and :func:`..sweep.state_sweep_torch`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import torch

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1._compile import (
    compiled,
)
from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.kda_chunk_prep_kernel import (  # noqa: E501
    build_kda_chunk_prep,
    supports_prep_geometry,
)

__all__ = [
    "chunk_prep",
    "chunk_prep_torch",
    "supports_prep",
    "supports_prep_geometry",
]

_KERNEL_CACHE: Dict[Tuple[Any, ...], Any] = {}
_KERNEL_LOCK = threading.Lock()

_OP_DTYPES = (torch.bfloat16, torch.float32)


def supports_prep(chunk_size: int, k_dim: int, op_dtype: torch.dtype) -> Optional[str]:
    """``None`` when the kernel can run this configuration, else why it cannot."""
    if op_dtype not in _OP_DTYPES:
        return f"operand dtype {op_dtype} is not one of {list(_OP_DTYPES)}"
    return supports_prep_geometry(chunk_size, k_dim)


def _get_kernel(chunk_size: int, k_dim: int, op_bf16: bool):
    key = (int(chunk_size), int(k_dim), bool(op_bf16))
    with _KERNEL_LOCK:
        launch = _KERNEL_CACHE.get(key)
        if launch is None:
            launch = build_kda_chunk_prep(chunk_size=key[0], k_dim=key[1], op_bf16=key[2])
            _KERNEL_CACHE[key] = launch
        return launch


def chunk_prep_torch(
    qf: torch.Tensor, kf: torch.Tensor, cg: torch.Tensor, op_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch twin. ``qf, kf, cg: [NB, C, K]`` fp32.

    Returns ``(qg, kgam, kg, dec)`` with ``qg``/``kg`` at ``op_dtype``, exactly
    the four tensors the kernel writes and rounded at exactly the same points.
    """
    gamma = cg.exp()
    chunk_total = cg[:, -1:, :]
    e_fac = (chunk_total - cg).exp()
    return (
        (qf * gamma).to(op_dtype),
        kf * gamma,
        (kf * e_fac).to(op_dtype),
        chunk_total.reshape(cg.shape[0], cg.shape[-1]).exp(),
    )


@compiled
def _chunk_prep_bwd(qf, kf, cg, d_qw, d_kgam, d_kg, d_dec, chunk: int, want_cg: bool):
    """The adjoint of :func:`chunk_prep`, as one Inductor-fused region.

    Fourteen elementwise ops on ``[NB, C, K]``, which is 200 MB at production
    geometry — measured at **17 launches and ~7.4 GB of HBM traffic** as an eager
    chain, the single largest block of glue in the whole backward. Inductor fuses
    it to **3 launches** (`bench/probe_torch_tricks.py`), because every line is
    elementwise in the same index space and only ``d_ct`` reduces.

    ``d_qw`` and ``d_kg`` arrive at the operand dtype (bf16 in production) while
    ``gamma``/``e_fac`` are fp32, so the multiplies promote and the ``.float()``
    an earlier version spelled out is dead weight. ``d_kgam`` is fp32 already.
    """
    gamma = cg.exp()
    chunk_total = cg[:, -1:, :]
    e_fac = (chunk_total - cg).exp()

    d_qf = d_qw[:, :chunk] * gamma
    a = d_kgam * gamma
    b = d_kg * e_fac
    d_kf = a + b

    if not want_cg:
        return d_qf, d_kf, None

    d_cg = torch.addcmul(qf * d_qf, kf, a - b)
    # `ct` is a *row* of `cg`, not an input of its own, so everything that flows
    # through it lands back on the last row.
    d_ct = (kf * b).sum(dim=-2)
    if d_dec is not None:
        d_ct = d_ct + d_dec * chunk_total.reshape(d_ct.shape).exp()
    # `index_copy` on the last row rather than an in-place `+=`: the in-place
    # write on a freshly produced tensor is correct but forces Inductor to
    # materialise `d_cg` before mutating it, which costs the fusion.
    d_cg = torch.cat((d_cg[:, : chunk - 1], (d_cg[:, chunk - 1] + d_ct).unsqueeze(1)), dim=1)
    return d_qf, d_kf, d_cg


class _ChunkPrep(torch.autograd.Function):
    """The decay chain and its analytic adjoint.

    ``qw`` comes back as the full ``[NB, 2C, K]`` operand buffer with **only its
    top half written**: the sweep wants ``[QG; W]`` stacked, ``W`` is a GEMM the
    caller issues next, and having the kernel store straight into the top half
    is what removes the separate concatenate. The caller must fill
    ``qw[:, C:]`` before using it, and the gradient routes correctly either way
    because ``copy_`` into the bottom slice zeroes that region of ``qw``'s own
    gradient.
    """

    @staticmethod
    def forward(ctx, qf, kf, cg, op_dtype, use_kernel):  # type: ignore[override]
        nb, chunk, k_dim = qf.shape
        dev = qf.device
        qw = torch.empty((nb, 2 * chunk, k_dim), dtype=op_dtype, device=dev)
        kgam = torch.empty((nb, chunk, k_dim), dtype=torch.float32, device=dev)
        kg = torch.empty((nb, chunk, k_dim), dtype=op_dtype, device=dev)
        dec = torch.empty((nb, k_dim), dtype=torch.float32, device=dev)

        if use_kernel:
            _get_kernel(chunk, k_dim, op_dtype is torch.bfloat16)(
                qf.reshape(-1),
                kf.reshape(-1),
                cg.reshape(-1),
                qw.reshape(-1),
                kgam.reshape(-1),
                kg.reshape(-1),
                dec.reshape(-1),
                int(nb),
            )
        else:
            qg_t, kgam_t, kg_t, dec_t = chunk_prep_torch(qf, kf, cg, op_dtype)
            qw[:, :chunk].copy_(qg_t)
            kgam.copy_(kgam_t)
            kg.copy_(kg_t)
            dec.copy_(dec_t)

        ctx.save_for_backward(qf, kf, cg)
        ctx.chunk = chunk
        return qw, kgam, kg, dec

    @staticmethod
    def backward(ctx, d_qw, d_kgam, d_kg, d_dec):  # type: ignore[override]
        qf, kf, cg = ctx.saved_tensors
        needs = ctx.needs_input_grad
        d_qf, d_kf, d_cg = _chunk_prep_bwd(
            qf, kf, cg, d_qw, d_kgam, d_kg, d_dec, ctx.chunk, bool(needs[2])
        )
        return (
            d_qf if needs[0] else None,
            d_kf if needs[1] else None,
            d_cg,
            None,
            None,
        )


def chunk_prep(
    qf: torch.Tensor,
    kf: torch.Tensor,
    cg: torch.Tensor,
    op_dtype: torch.dtype,
    use_kernel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(qw, kgam, kg, dec)``; see :class:`_ChunkPrep` for ``qw``'s contract."""
    return _ChunkPrep.apply(qf, kf, cg, op_dtype, use_kernel)

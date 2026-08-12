###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Torch-facing FlyDSL attention-residual mixer: one launch each way.

:func:`flydsl_attn_res_mix` has the same signature as
:func:`..._eager.reference.eager_attn_res_mix` and is interchangeable with it.

Division of labour, following the KDA precedent:

* the **rank-1 scorer is folded outside the kernel**, with plain
  autograd-visible torch ops, so the kernel takes one ``[hidden]`` vector and
  there is exactly one auditable copy of the ``norm_weight ⊙ proj_weight``
  factorisation. Gradients reach both factors through ordinary autograd;
* the **``dW`` token reduction is finished outside the kernel**. The kernel
  emits ``dw_tok[n, d]`` and this module does the ``sum(0)``, which keeps that
  gradient fp32 and deterministic (see the backward kernel's docstring for why
  a bf16 GEMV against ``v`` was rejected);
* **one ``torch.autograd.Function`` for the kernel pair**, taking the launchers
  from a per-geometry cache rather than rebuilding them.

Anything the kernel cannot run raises :class:`ValueError` naming the fallback
rather than degrading quietly — the rule the KDA kernel already follows.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._eager.reference import (
    accum_dtype,
    eager_attn_res_mix,
    fused_score_weight,
)
from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1.attn_res_mixer_bwd_kernel import (
    build_attn_res_mixer_bwd,
)
from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1.attn_res_mixer_kernel import (
    HIDDEN_ALIGN,
    MAX_CANDIDATES,
    build_attn_res_mixer_fwd,
    supports_mixer_geometry,
)

__all__ = [
    "flydsl_attn_res_mix",
    "flydsl_mix_kernel",
    "supports_mixer_inputs",
    "inject_defect",
]

#: The kernel's element types. Everything else (fp16, fp64) goes to eager,
#: because the kernel's accumulators are fp32 and a wider input would be
#: silently narrowed.
_ELEM_DTYPES = {torch.float32: "f32", torch.bfloat16: "bf16"}

_FWD_CACHE: Dict[Tuple[int, int, str, float, str], object] = {}
_BWD_CACHE: Dict[Tuple[int, int, str, str], object] = {}
_CACHE_LOCK = threading.Lock()

#: Test-only build-time defects for the current process, ``(fwd, bwd)``.
#: Set by :func:`inject_defect`; ``("", "")`` in production.
_INJECT: Tuple[str, str] = ("", "")


def inject_defect(fwd: str = "", bwd: str = "") -> None:
    """**Test only.** Make subsequent launches use a deliberately broken kernel.

    A parity test that cannot fail proves nothing, so the suite builds each
    named defect and requires the very same assertions to reject it. The defect
    is applied in the emitted MLIR — not in the torch glue — because the glue is
    not the thing under test. Names are validated by the kernel builders, so a
    typo raises instead of quietly meaning "no defect".

    Call with no arguments to restore the correct kernels.
    """
    global _INJECT
    _INJECT = (str(fwd or ""), str(bwd or ""))


def _get_fwd(hidden: int, num_blocks: int, elem: str, eps: float):
    key = (int(hidden), int(num_blocks), elem, float(eps), _INJECT[0])
    with _CACHE_LOCK:
        launch = _FWD_CACHE.get(key)
        if launch is None:
            launch = build_attn_res_mixer_fwd(
                hidden=key[0], num_blocks=key[1], elem_dtype=elem, eps=key[3], inject=key[4]
            )
            _FWD_CACHE[key] = launch
        return launch


def _get_bwd(hidden: int, num_blocks: int, elem: str):
    key = (int(hidden), int(num_blocks), elem, _INJECT[1])
    with _CACHE_LOCK:
        launch = _BWD_CACHE.get(key)
        if launch is None:
            launch = build_attn_res_mixer_bwd(
                hidden=key[0], num_blocks=key[1], elem_dtype=elem, inject=key[3]
            )
            _BWD_CACHE[key] = launch
        return launch


def supports_mixer_inputs(prefix_sum: Tensor, block_residual: Tensor) -> Optional[str]:
    """``None`` when the kernel can run these inputs, else why it cannot."""
    if not prefix_sum.is_cuda:
        return "the kernel is a GPU kernel and the inputs are on CPU"
    if prefix_sum.dtype != block_residual.dtype:
        return (
            f"prefix_sum is {prefix_sum.dtype} but block_residual is "
            f"{block_residual.dtype}; the kernel reads one element type"
        )
    if prefix_sum.dtype not in _ELEM_DTYPES:
        return f"dtype {prefix_sum.dtype} is not one of {list(_ELEM_DTYPES)}"
    hidden = prefix_sum.shape[-1]
    num_blocks = block_residual.shape[-2]
    return supports_mixer_geometry(int(hidden), int(num_blocks) + 1)


class _FlydslMix(torch.autograd.Function):
    """The kernel pair as one autograd node.

    ``score_weight`` arrives already fused and fp32; the kernel never sees the
    two factors, and their gradients are produced by the ordinary autograd graph
    that built it.
    """

    @staticmethod
    def forward(ctx, prefix_sum: Tensor, block_residual: Tensor, score_weight: Tensor, eps: float):
        num_tokens, hidden = prefix_sum.shape
        num_blocks = block_residual.shape[-2]
        elem = _ELEM_DTYPES[prefix_sum.dtype]

        out = torch.empty_like(prefix_sum)
        r_sav = torch.empty(num_tokens, num_blocks + 1, dtype=torch.float32, device=prefix_sum.device)
        dot_sav = torch.empty_like(r_sav)

        _get_fwd(hidden, num_blocks, elem, eps)(
            block_residual.reshape(-1),
            prefix_sum.reshape(-1),
            score_weight.reshape(-1),
            out.reshape(-1),
            r_sav.reshape(-1),
            dot_sav.reshape(-1),
            int(num_tokens),
        )

        ctx.save_for_backward(prefix_sum, block_residual, score_weight, r_sav, dot_sav)
        ctx.elem = elem
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        prefix_sum, block_residual, score_weight, r_sav, dot_sav = ctx.saved_tensors
        num_tokens, hidden = prefix_sum.shape
        num_blocks = block_residual.shape[-2]

        grad_out = grad_out.contiguous()
        d_ps = torch.empty_like(prefix_sum)
        d_br = torch.empty_like(block_residual)
        # dw_tok is the per-token partial of the scorer gradient; see the
        # backward kernel's docstring for why the token reduction is finished
        # here in fp32 rather than inside the kernel.
        dw_tok = torch.empty(num_tokens, hidden, dtype=torch.float32, device=prefix_sum.device)
        # d_dot is the per-candidate scorer adjoint. Nothing here consumes it --
        # dw_tok already folds it against `v` -- but the kernel emits it because
        # it is `[N, C]` (64 KB at the scaled shape, against dw_tok's 33 MB) and
        # it is the one intermediate that localises a softmax-adjoint bug to a
        # candidate. The buffer therefore has to exist for the launch.
        d_dot = torch.empty(num_tokens, num_blocks + 1, dtype=torch.float32, device=prefix_sum.device)

        _get_bwd(hidden, num_blocks, ctx.elem)(
            block_residual.reshape(-1),
            prefix_sum.reshape(-1),
            score_weight.reshape(-1),
            grad_out.reshape(-1),
            r_sav.reshape(-1),
            dot_sav.reshape(-1),
            d_br.reshape(-1),
            d_ps.reshape(-1),
            dw_tok.reshape(-1),
            d_dot.reshape(-1),
            int(num_tokens),
        )
        d_sw = dw_tok.sum(0).to(score_weight.dtype)
        return d_ps, d_br, d_sw, None


def flydsl_mix_kernel(prefix_sum: Tensor, block_residual: Tensor, score_weight: Tensor, eps: float) -> Tensor:
    """The kernel pair on already-flattened, already-fused inputs.

    Args:
        prefix_sum: ``[num_tokens, hidden]``, contiguous.
        block_residual: ``[num_tokens, num_blocks, hidden]``, contiguous.
        score_weight: ``[hidden]`` fp32 — the fused rank-1 scorer.
        eps: RMSNorm epsilon.

    Exposed separately from :func:`flydsl_attn_res_mix` so tests can drive the
    kernel without the reshape / fold layer around it.
    """
    return _FlydslMix.apply(prefix_sum, block_residual, score_weight, eps)


def flydsl_attn_res_mix(
    prefix_sum: Tensor,
    block_residual: Tensor,
    norm_weight: Tensor,
    proj_weight: Tensor,
    eps: float,
) -> Tensor:
    """Fused-kernel ``_apply_attn_res``, interchangeable with the eager entry.

    Args:
        prefix_sum: ``[*, hidden]`` — the running residual stream.
        block_residual: ``[*, num_blocks, hidden]`` — the checkpoints.
        norm_weight: ``[hidden]`` RMSNorm gain.
        proj_weight: ``[1, hidden]`` projection row.
        eps: RMSNorm epsilon.

    Returns:
        ``[*, hidden]``, in ``prefix_sum``'s dtype.

    Raises:
        ValueError: when the geometry or dtype is outside what the kernel
            supports, naming the fallback rather than degrading quietly.
    """
    if block_residual.shape[-1] != prefix_sum.shape[-1]:
        raise ValueError(
            f"block_residual hidden {block_residual.shape[-1]} != "
            f"prefix_sum hidden {prefix_sum.shape[-1]}"
        )

    # num_blocks == 0 makes the softmax a single-candidate no-op returning
    # prefix_sum itself. The kernel handles C == 1 correctly, but the eager path
    # is one op and needs no launch, so route it there. The caller normally
    # skips the call entirely in that case (kimi_k3_block.py).
    if block_residual.shape[-2] == 0:
        return eager_attn_res_mix(prefix_sum, block_residual, norm_weight, proj_weight, eps)

    reason = supports_mixer_inputs(prefix_sum, block_residual)
    if reason is not None:
        raise ValueError(
            f"the FlyDSL attention-residual kernel cannot run these inputs: {reason}. It "
            f"needs a CUDA tensor, hidden % {HIDDEN_ALIGN} == 0, at most "
            f"{MAX_CANDIDATES} candidates, and bf16 or fp32. Select "
            "attn_res_backend: eager."
        )

    hidden = prefix_sum.shape[-1]
    num_blocks = block_residual.shape[-2]
    lead = prefix_sum.shape[:-1]

    # The scorer is folded with plain torch ops so autograd reaches both factors.
    score_weight = fused_score_weight(norm_weight, proj_weight, accum_dtype(proj_weight.dtype))
    if score_weight.dtype != torch.float32:
        score_weight = score_weight.float()

    ps_flat = prefix_sum.reshape(-1, hidden).contiguous()
    br_flat = block_residual.reshape(-1, num_blocks, hidden).contiguous()
    mixed = flydsl_mix_kernel(ps_flat, br_flat, score_weight.contiguous(), float(eps))
    return mixed.reshape(*lead, hidden)

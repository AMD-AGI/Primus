###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Opt-in fused RMSNorm for Gemma 4.

Bridge implements Gemma 4's norm as the literal Hugging Face expression::

    normed = hidden_states.float() * torch.pow(
        hidden_states.float().pow(2).mean(-1, keepdim=True) + eps, -0.5
    )
    normed = normed * weight.float()
    return normed.type_as(hidden_states)

Every ``Gemma4RMSNorm`` in both the dense and the MoE stack routes through it, so
in eager mode a single norm costs two full fp32 casts of the activation plus a
chain of unfused elementwise kernels. On the dense 31B config that is 7 norms per
layer x 60 layers per micro-batch, and a profile of a 4-micro-batch step shows
13,442 ``aten::pow`` and 3,360 ``aten::mean`` launches with elementwise work at
17.8% of GPU time -- pure memory traffic, since at hidden 5376 / seq 4096 /
mbs 4 each fp32 temporary is ~350 MB.

``PRIMUS_GEMMA4_FUSED_NORMS=compile`` hands the same expression to
``torch.compile``, so Inductor collapses the cast/square/mean/rsqrt/scale chain
into a single kernel that reads and writes the activation once. The arithmetic is
unchanged, which keeps the Bridge parity tests meaningful.

``PRIMUS_GEMMA4_FUSED_NORMS=te`` instead routes the scaled norms to Transformer
Engine's fused RMSNorm. That is faster still, but TE normalizes in the input
dtype with fp32 accumulation rather than promoting the whole expression to fp32,
so it is not bit-comparable with Hugging Face.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_ENV = "PRIMUS_GEMMA4_FUSED_NORMS"
_PATCHED_ATTR = "_primus_gemma4_fused_norm"


def _fused_rms_norm(hidden_states: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """Hugging Face Gemma 4 RMSNorm, written so a compiler can fuse it.

    ``pow(x, -0.5)`` is kept as ``rsqrt`` (identical for the strictly positive
    mean-square) and the input is promoted once instead of twice.
    """
    input_dtype = hidden_states.dtype
    x = hidden_states.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        normed = normed * weight.float()
    return normed.to(input_dtype)


def _build_compiled_norm() -> Optional[Callable]:
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return None
    try:
        # Leave `dynamic` at the default so the first trace is shape-specialized
        # and only the axes that actually move get generalized. Gemma 4 calls the
        # norm on both hidden_size and head_dim tails, so pinning `dynamic=False`
        # would recompile per shape.
        return compile_fn(_fused_rms_norm)
    except Exception as exc:  # pragma: no cover - depends on host inductor
        log_rank_0(f"[Patch:gemma4.norms.fused] torch.compile unavailable ({exc}); norms left eager")
        return None


def _build_te_norm() -> Optional[Callable]:
    """Return a callable with the ``_gemma4_rms_norm`` signature backed by TE.

    TE exposes RMSNorm as a module rather than a function, and it has to be
    constructed per (hidden_size, dtype). The weight lives on the caller's
    ``Gemma4RMSNorm``, so the module here is only a kernel holder: its own
    parameter is swapped out for the caller's on every call.
    """
    try:
        import transformer_engine.pytorch as te
    except Exception:
        return None

    rms_norm_cls = getattr(te, "RMSNorm", None)
    if rms_norm_cls is None:
        return None

    cache: dict = {}

    def te_rms_norm(hidden_states: torch.Tensor, weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
        # The scaleless variant (MoE router norm) has no TE equivalent that skips
        # the weight, and it is a negligible share of the norm traffic.
        if weight is None:
            return _fused_rms_norm(hidden_states, weight, eps)

        key = (weight.shape[-1], hidden_states.dtype, eps)
        module = cache.get(key)
        if module is None:
            module = rms_norm_cls(
                weight.shape[-1], eps=eps, sequence_parallel=False, params_dtype=weight.dtype
            ).to(device=weight.device)
            cache[key] = module
        module.weight = weight
        return module(hidden_states)

    return te_rms_norm


@register_patch(
    "gemma4.norms.fused",
    backend="megatron_bridge",
    phase="setup",
    description="Opt-in: fuse the Gemma 4 RMSNorm elementwise chain via torch.compile or Transformer Engine",
)
def patch_gemma4_fused_norms(ctx: PatchContext) -> None:
    backend = os.environ.get(_ENV, "").strip().lower()
    if backend in ("", "0", "off", "none", "eager"):
        return

    try:
        from megatron.bridge.models.gemma import modeling_gemma4
    except Exception:
        return

    original = getattr(modeling_gemma4, "_gemma4_rms_norm", None)
    if original is None or getattr(original, _PATCHED_ATTR, False):
        return

    if backend in ("1", "on", "compile", "inductor"):
        fused = _build_compiled_norm()
        label = "torch.compile"
    elif backend == "te":
        fused = _build_te_norm()
        label = "transformer_engine.pytorch.RMSNorm"
    else:
        log_rank_0(f"[Patch:gemma4.norms.fused] Unknown {_ENV}={backend!r}; norms left eager")
        return

    if fused is None:
        log_rank_0(f"[Patch:gemma4.norms.fused] {label} unavailable; norms left eager")
        return

    setattr(fused, _PATCHED_ATTR, True)

    # Gemma4RMSNorm.forward resolves the helper from module globals, so rebinding
    # it here covers every norm instance in the dense and MoE stacks -- input,
    # q/k, post-attention, pre/post-MLP, per-layer-input and router norms alike.
    modeling_gemma4._gemma4_rms_norm = fused
    log_rank_0(f"[Patch:gemma4.norms.fused] Gemma 4 RMSNorm -> {label}")

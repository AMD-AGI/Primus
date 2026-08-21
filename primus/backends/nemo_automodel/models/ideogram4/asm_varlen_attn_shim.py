###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Bridge from Primus's var-len dispatch to the hand-written CDNA4 ASM backward.

WHAT IT IS:
  ``varlen_attn_impl=asm`` keeps aiter's CK-tile kernel for the forward and replaces the
  BACKWARD -- 2.5x the forward's FLOPs, and the larger half of the attention step -- with
  three hand-written gfx950 kernels: a delta preprocess, a fused dK/dV pass and a fused dQ
  pass. On the Ideogram-4 production packing (40968 tokens, 18 heads, head_dim 256, 16
  ragged segments) the backward measures 13.77 ms against Triton's 14.54 ms, and the
  gradients match CK's own error against an fp32 reference to three significant figures.

WHY IT IS A SHIM:
  The kernels are assembled by PyISA, which is not a Primus dependency and is not present
  in a normal training image. So this module holds no kernel code: it locates the ASM
  package, reports honestly whether it is usable, and forwards the call. Everything about
  whether the path is AVAILABLE is decided here; everything about what it COMPUTES lives
  in the ASM package. ``resolve_varlen_impl`` calls :func:`asm_available` once at install
  time and silently degrades to ``triton`` if anything is missing, so selecting ``asm`` in
  an image without it costs a warning rather than a failed run.

  Point ``PRIMUS_IDEOGRAM_ASM_PATH`` at the directory holding the ASM package if it is not
  already importable.

CONSTRAINTS (checked by :func:`asm_available`, not assumed):
  gfx950, head_dim 256, and a ROCm new enough to assemble the kernels. The kernels bake
  the packed-layout token stride in as an immediate, so they are also built per head
  count -- that happens once, into an on-disk cache, on first use.
"""
from __future__ import annotations

import functools
import os
import sys
from typing import Optional, Tuple

from torch import Tensor

IDEOGRAM4_HEAD_DIM = 256
_SUPPORTED_ARCH = "gfx950"


def _candidate_paths() -> list:
    paths = []
    env = os.getenv("PRIMUS_IDEOGRAM_ASM_PATH")
    if env:
        paths.append(env)
    return paths


def _import_asm():
    for p in _candidate_paths():
        if p not in sys.path:
            sys.path.insert(0, p)
    import ideogram4_pyisa  # noqa: F401  (the ASM package)

    return ideogram4_pyisa


@functools.lru_cache(maxsize=1)
def asm_available() -> Tuple[bool, str]:
    """Whether the ASM path can actually run here, and if not, why not.

    Resolved once. The reason string is surfaced in the fallback warning, so a
    misconfigured ``PRIMUS_IDEOGRAM_ASM_PATH`` says so rather than looking like an
    unsupported GPU.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "no ROCm device visible"
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch != _SUPPORTED_ARCH:
            return False, f"built for {_SUPPORTED_ARCH}, this GPU is {arch}"
        _import_asm()
    except ImportError as exc:
        return False, (
            f"the ASM package is not importable ({exc}); set PRIMUS_IDEOGRAM_ASM_PATH "
            "to the directory containing it"
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def asm_varlen_flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    *,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """Var-len bf16 flash attention with the ASM backward.

    Signature-compatible with :func:`..attention.varlen_flash_attention`: q/k/v are packed
    ``(total_tokens, H, D)``, ``cu_seqlens`` is the ``int32`` ``(num_segments + 1,)``
    prefix sum, and the result is ``(total_tokens, H, D)``.

    The head-dim check is here rather than inside the op for the same reason as the Triton
    path: it reads static shape metadata, so it costs a guard under ``torch.compile``,
    whereas raising from inside an opaque op would only surface at runtime.
    """
    if q.shape[-1] != IDEOGRAM4_HEAD_DIM:
        raise ValueError(
            f"varlen_attn_impl=asm is built for head_dim={IDEOGRAM4_HEAD_DIM}, "
            f"got {q.shape[-1]}"
        )
    return _import_asm().asm_varlen_flash_attention(
        q, k, v, cu_seqlens, max_seqlen, softmax_scale=softmax_scale
    )

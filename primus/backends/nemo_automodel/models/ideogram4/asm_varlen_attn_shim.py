###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Bridge from Primus's var-len dispatch to the hand-written CDNA4 ASM attention.

WHAT IT IS:
  ``varlen_attn_impl=asm`` replaces BOTH halves of var-len attention with hand-written
  gfx950 kernels: a fused forward, and a backward made of a delta preprocess, a fused
  dK/dV pass and a fused dQ pass. The gradients match the CK path's own error against an
  fp32 reference. An earlier revision of this shim swapped only the backward and left the
  forward on CK; that is no longer what the ``asm`` arm does.

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
  gfx950 and head_dim 256, which is what the tile sizes were tuned for. The kernels take
  the packed-layout token stride in the parameter block rather than baking it in, so ONE
  build serves every head count -- the code objects are a pure function of (arch, head
  dim) and can therefore ship prebuilt. PyISA is needed to produce them, not to run them;
  an image carrying a prebuilt cache needs no assembler.
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
    """Var-len bf16 flash attention on the ASM forward and backward.

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

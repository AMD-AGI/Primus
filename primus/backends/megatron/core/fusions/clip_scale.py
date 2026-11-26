###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The gradient-clip scale ``g = bf16(f32(g) * coeff)`` over a list of BF16 tensors, bit for bit.

Megatron's ``clip_grad_by_total_norm_fp32`` keeps the clip coefficient ``min(1, max_norm / norm)``
on the device and scales every gradient by it on every step through Transformer Engine's
``multi_tensor_scale_tensor``. On ROCm that kernel runs at about a fifth of the memory
bandwidth (TE 2.19.0.dev0 on gfx1250: 13.1 ms for 10.2e9 BF16 elements, 9.4 ms of it with the
host blocked in the applier). ``scale_tensors`` is the same elementwise operation as one
Triton launch per tensor: widen to FP32, multiply by the FP32 coefficient read from the device,
round once to BF16 (2.5 ms for the same list).

Transcribed from TE's observable behaviour, so a TE change can break it silently:
``bitwise_check`` compares it, before its first use in a process, with the installed
``multi_tensor_scale_tensor`` on every BF16 bit pattern (all finite values, the subnormals,
+-0, +-inf and the NaNs) for a set of coefficients that includes the first call's own, and
keeps TE on a mismatch. TE's kernel also raises its ``noop_flag`` buffer on a non-finite input;
Megatron passes a throwaway buffer there and never reads it, so this path leaves it untouched.
"""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check

_BLOCK = 4096
# Elements per launch. Triton types the element count as int32 below 2**31 and as int64 from
# there on, so one launch over a larger tensor would run a kernel variant the self-check (small
# probes) never ran. Larger tensors (the expert weights) take several launches over consecutive
# slices instead: an elementwise kernel, so the result is the same.
MAX_ELEMS = 2**30
# probe coefficients: no clipping (1.0), just below it, round and unround fractions, a tiny
# one that drives results into the BF16 subnormals, and zero
_PROBE_COEFFS = (1.0, 0.99999994, 0.75, 0.5, 0.3333333, 0.159744, 1.0e-3, 7.1e-5, 1.0e-30, 0.0)


@triton.jit
def _scale(G, C, n, BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    c = tl.load(C)
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
    tl.store(G + off, (g * c).to(G.dtype.element_ty), mask=m)


def eligible(tensor_lists, coeff) -> bool:
    """Whether a ``multi_tensor_scale_tensor`` call is the in-place BF16 scale this module covers."""
    if len(tensor_lists) != 2 or len(tensor_lists[0]) != len(tensor_lists[1]) or not tensor_lists[0]:
        return False
    if not (torch.is_tensor(coeff) and coeff.is_cuda and coeff.dtype == torch.float32 and coeff.numel() == 1):
        return False
    for src, dst in zip(*tensor_lists):
        # in place only: same storage, offset and layout (Megatron passes [grads, grads])
        if src is not dst and not (
            src.data_ptr() == dst.data_ptr() and src.shape == dst.shape and src.stride() == dst.stride()
        ):
            return False
        if not (dst.is_cuda and dst.dtype == torch.bfloat16 and dst.is_contiguous()):
            return False
        if dst.device != coeff.device:
            return False
    return True


def scale_tensors(tensors: Sequence[torch.Tensor], coeff: torch.Tensor) -> None:
    """``t = bf16(f32(t) * coeff)`` in place for contiguous BF16 CUDA tensors and a one-element
    FP32 ``coeff`` on the same device. One launch per ``MAX_ELEMS`` elements of a tensor."""
    for t in tensors:
        flat = t.view(-1)
        for start in range(0, flat.numel(), MAX_ELEMS):
            part = flat[start : start + MAX_ELEMS]
            n = part.numel()
            _scale[(triton.cdiv(n, _BLOCK),)](part, coeff, n, BLOCK=_BLOCK, num_warps=4)


def _probe_tensors(device) -> list:
    """Every BF16 bit pattern, in lengths that end inside, at and one past a kernel block."""
    bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
    return [
        bits.clone(),
        bits[: 3 * _BLOCK + 1].clone(),
        bits[32000 : 32000 + _BLOCK - 1].clone(),
        bits[-7:].clone(),
    ]


def _probe(scale, coeff: torch.Tensor) -> list:
    """``scale(tensors, coeff_tensor)`` on fresh probe tensors for every probe coefficient."""
    outs = []
    coeffs = [coeff.detach().reshape(1).clone()]
    coeffs += [torch.tensor([c], dtype=torch.float32, device=coeff.device) for c in _PROBE_COEFFS]
    for c in coeffs:
        tensors = _probe_tensors(coeff.device)
        scale(tensors, c)
        outs += tensors
    return outs


def checked(original_scale, coeff: torch.Tensor) -> bool:
    """The self-check's verdict. ``original_scale(tensors, coeff)`` must run the installed TE path
    (``multi_tensor_applier`` with ``multi_tensor_scale_tensor``) in place. Elementwise, so one
    check per process covers every tensor length; the probes are built only when it runs."""
    return bitwise_check.passed(
        "bitwise_clip_scale",
        lambda: _probe(scale_tensors, coeff),
        lambda: _probe(original_scale, coeff),
    )

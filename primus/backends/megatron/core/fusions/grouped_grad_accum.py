###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``main_grad += g_1; main_grad += g_2; ...`` for BF16 gradients as one pass, bit for bit.

Megatron DDP's backward hook runs ``param.main_grad.add_(param.grad)`` once per parameter and
micro-batch: with a BF16 gradient buffer, 2 B + 2 B read and 2 B written per element, each time.
``accumulate_`` adds up to ``MAX_GRADS`` held gradients in one Triton launch,
``main = bf16(...bf16(bf16(main + g_1) + g_2)... + g_k)``: every addition in FP32 on the
widened operands and rounded to the buffer's dtype before the next one, in the same order, as
``k`` calls of ``add_`` do. One pass reads ``main`` and writes it once: ``2k + 4`` instead of
``6k`` bytes per element (FP32 buffer: ``2k + 8`` instead of ``10k``; the rounding step is then
the identity).

Depends on ATen's BF16 addition (FP32 add, round to nearest even), so a torch change can break
it silently: ``bitwise_check`` compares it, before its first use for a group size and buffer
dtype, with the ``add_`` loop on probes holding every BF16 bit pattern and sums whose
intermediate rounding matters, and the caller keeps ``add_`` on a mismatch.

What the caller that holds gradients (``grouped_grad_accum_patches.py``) assumes beyond that:
a held gradient is not a tensor that a later backward node of the same micro-batch overwrites in
place (the hook's ``data_ptr`` guard only sees a storage that comes back in a later micro-batch;
the per-step gradient hashes of a model cover the rest), and a parameter whose gradient route
changes to a fused write into ``main_grad`` (``grad_added_to_main_grad``) inside a step gets its
held gradients added after that write unless the fused path flushes them first, as
``embedding_rows_grad`` does (``gradient_accumulation_fusion`` is all or nothing per parameter).
"""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check

MAX_GRADS = 8
_BLOCK = 4096
# Elements per launch. Triton types the element count as int32 below 2**31 and as int64 from
# there on, so one launch over a larger tensor would run a kernel variant the self-check (small
# probes) never ran. Larger tensors (the expert weights) take several launches over consecutive
# slices instead: an elementwise kernel, so the result is the same.
MAX_ELEMS = 2**30
# Attribute of a parameter: the list of its gradients held for a later pass (the DDP hook of
# grouped_grad_accum_patches.py owns it).
HELD = "_primus_held_grads"


@triton.jit
def _add1(acc, G, off, m, out_ty: tl.constexpr):
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
    return (acc + g).to(out_ty).to(tl.float32)


@triton.jit
def _accum(MAIN, G0, G1, G2, G3, G4, G5, G6, G7, n, K: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    ty: tl.constexpr = MAIN.dtype.element_ty
    acc = tl.load(MAIN + off, mask=m, other=0.0).to(tl.float32)
    acc = _add1(acc, G0, off, m, ty)
    if K > 1:
        acc = _add1(acc, G1, off, m, ty)
    if K > 2:
        acc = _add1(acc, G2, off, m, ty)
    if K > 3:
        acc = _add1(acc, G3, off, m, ty)
    if K > 4:
        acc = _add1(acc, G4, off, m, ty)
    if K > 5:
        acc = _add1(acc, G5, off, m, ty)
    if K > 6:
        acc = _add1(acc, G6, off, m, ty)
    if K > 7:
        acc = _add1(acc, G7, off, m, ty)
    tl.store(MAIN + off, acc.to(ty), mask=m)


def can_fuse(main: torch.Tensor, grad: torch.Tensor) -> bool:
    """A contiguous BF16 CUDA gradient for a contiguous BF16 or FP32 buffer view of its size."""
    return (
        main.is_cuda
        and grad.device == main.device
        and grad.dtype == torch.bfloat16
        and main.dtype in (torch.bfloat16, torch.float32)
        and main.numel() == grad.numel()
        and main.numel() > 0
        and main.is_contiguous()
        and grad.is_contiguous()
    )


def _kernel(main: torch.Tensor, grads: Sequence[torch.Tensor]) -> None:
    k = len(grads)
    flat = main.view(-1)
    flat_grads = [g.view(-1) for g in grads]
    for start in range(0, flat.numel(), MAX_ELEMS):
        part = flat[start : start + MAX_ELEMS]
        n = part.numel()
        ptrs = [g[start : start + MAX_ELEMS] for g in flat_grads]
        ptrs = ptrs + [ptrs[0]] * (MAX_GRADS - k)
        _accum[(triton.cdiv(n, _BLOCK),)](part, *ptrs, n, K=k, BLOCK=_BLOCK, num_warps=4)


def _add_loop(main: torch.Tensor, grads: Sequence[torch.Tensor]) -> None:
    for g in grads:
        main.add_(g)  # the DDP hook's statement


def _probe(k: int, main_dtype, device):
    """``main`` and ``k`` gradients: every BF16 bit pattern against partners of nearby and of
    distant exponents (ties and sticky bits of the intermediate roundings, cancellations to
    +-0, overflow to inf, inf - inf, NaN), plus plain normal samples."""
    gen = torch.Generator(device=device)
    gen.manual_seed(4321 + k)
    bits = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
    reps = 6
    n = 65536 * reps + 4097

    def sample(i):
        r = torch.randn(n, generator=gen, device=device)
        # exponents spread over +-12 binades so sums land on and off rounding ties
        r = r * torch.exp2(torch.randint(-12, 13, (n,), generator=gen, device=device).float())
        t = r.to(torch.bfloat16)
        t[i * 65536 : (i + 1) * 65536] = bits  # one operand sweeps every pattern
        return t

    main = sample(0).to(main_dtype)
    main[-4097::2] = 0.0  # a freshly zeroed buffer
    grads = [sample(1 + (i % (reps - 1))) for i in range(k)]
    if k > 1:
        grads[1][: 65536 * 2] = -grads[0][: 65536 * 2]  # exact cancellation, then +-0 sums
    return main, grads


def _checked(k: int, main_dtype, device) -> bool:
    def run(fn):
        main, grads = _probe(k, main_dtype, device)
        fn(main, grads)
        return [main]

    return bitwise_check.passed(
        "grouped_grad_accum", lambda: run(_kernel), lambda: run(_add_loop), key=(k, str(main_dtype))
    )


def accumulate_(main: torch.Tensor, grads: Sequence[torch.Tensor]) -> None:
    """``for g in grads: main.add_(g)``, in one pass per ``MAX_GRADS`` gradients where every
    pair passes ``can_fuse`` and the kernel has matched the loop for that count."""
    grads = list(grads)
    while grads:
        chunk, grads = grads[:MAX_GRADS], grads[MAX_GRADS:]
        if all(can_fuse(main, g) for g in chunk) and _checked(len(chunk), main.dtype, main.device):
            _kernel(main, chunk)
        else:
            _add_loop(main, chunk)


def flush_held(param: torch.Tensor) -> None:
    """Add ``param``'s held gradients into ``param.main_grad`` now, in the order they arrived.
    For code that writes into ``main_grad`` itself between two micro-batches (a later addition
    must not land before the held ones)."""
    held = getattr(param, HELD, None)
    if held:
        accumulate_(param.main_grad, held)
        held.clear()

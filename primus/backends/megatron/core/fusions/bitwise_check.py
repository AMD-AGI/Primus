###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One-time check that a bitwise replacement still matches the path it replaces.

The bitwise kernels reproduce the roundings of the torch / Inductor / Primus-Turbo / TE code
they stand in for, which a version bump can change without any error. Before its first use in
a process, each replacement runs itself and the original path once, forward and backward, and
compares every bit: on probe inputs of the training shapes (random, with the scalars and
weights the kernels will see and the edge values that matter to them), or on the real inputs
of that first call. On a mismatch (or an error) every rank that sees it warns, and the caller
keeps the original path for the rest of the run.

Probe inputs and output gradients come from private generators, so the check does not move the
global RNG; gradients are taken with ``torch.autograd.grad`` with respect to detached copies
only, so no parameter's ``.grad``, ``main_grad`` or hook is touched. Each rank decides on its
own, so neither path of a checked pair may run a collective.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Optional, Sequence, Tuple

import torch

from primus.core.utils.module_utils import log_rank_0, warning_rank_all

_results: Dict[Tuple[str, Hashable], bool] = {}


def randn(shape, device, seed: int = 0) -> torch.Tensor:
    """FP32 normal samples from a private generator."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(tuple(shape), generator=gen, device=device)


def outputs_and_grads(fn: Callable, *inputs: torch.Tensor) -> list:
    """``fn``'s outputs on detached copies of ``inputs``, then the gradients of the floating-point
    copies (None where unused) for random output gradients."""
    leaves = [t.detach().clone().requires_grad_(t.is_floating_point()) for t in inputs]
    with torch.enable_grad():
        outs = fn(*leaves)
    outs = list(outs) if isinstance(outs, (tuple, list)) else [outs]
    diff = [o for o in outs if o.requires_grad]
    wrt = [t for t in leaves if t.requires_grad]
    grads = []
    if diff and wrt:
        grad_outs = [randn(o.shape, o.device, seed=1 + i).to(o.dtype) for i, o in enumerate(diff)]
        grads = torch.autograd.grad(diff, wrt, grad_outs, allow_unused=True)
    return [o.detach() for o in outs] + list(grads)


def same_bits(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    """Equal shape, dtype and bits; any NaN matches any NaN."""
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if not a.is_floating_point():
        return torch.equal(a, b)
    nan_a, nan_b = a.isnan(), b.isnan()
    if not torch.equal(nan_a, nan_b):
        return False
    a, b = a.masked_fill(nan_a, 0), b.masked_fill(nan_b, 0)
    as_int = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.view(as_int), b.view(as_int))


def _device_error(e: Exception) -> bool:
    """A GPU fault, after which the context cannot be trusted (unlike running out of memory)."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return False
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if accelerator_error is not None and isinstance(e, accelerator_error):
        return True
    return isinstance(e, RuntimeError) and ("HIP error" in str(e) or "CUDA error" in str(e))


def passed(
    name: str, run: Callable[[], Sequence], reference: Callable[[], Sequence], key: Hashable = None
) -> bool:
    """Whether ``run()`` and ``reference()`` return the same bits, decided on the first call for
    ``(name, key)`` in this process. ``key`` holds what changes the kernels' launch or reduction
    order (row counts, widths, scalars), so each such case is checked once.

    Callers run on every forward: build probe inputs inside ``run`` / ``reference``, not before
    the call, so a decided check costs only the lookup. Nothing is decided (False, not cached)
    during CUDA graph capture, under inference mode, or when there is nothing to compare. A GPU
    fault is raised, not taken for a mismatch."""
    if (name, key) in _results:
        return _results[(name, key)]
    if torch.is_inference_mode_enabled() or (
        torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    ):
        return False
    try:
        got, want = list(run()), list(reference())
    except Exception as e:  # noqa: BLE001 - any failure but a GPU fault keeps the original path
        if _device_error(e):
            raise
        ok, reason = False, f"{type(e).__name__}: {e}"
    else:
        if not any(t is not None and t.numel() for t in want):
            return False
        ok = len(got) == len(want) and all(same_bits(a, b) for a, b in zip(got, want))
        reason = "outputs or gradients differ"
    _results[(name, key)] = ok
    label = name if key is None else f"{name} {key}"
    if ok:
        log_rank_0(f"[{label}] bit-identical to the path it replaces with the installed versions; in use")
    else:
        warning_rank_all(
            f"[{label}] not bit-identical to the path it replaces with the installed versions "
            f"({reason}); keeping that path"
        )
    return ok

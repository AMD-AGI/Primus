###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Forward + backward equivalence harness (plan-4 P24).

Every plan-4 test that compares an eager-Python reference against a
candidate (P25 / P26 Triton kernels, or the new P24 reference op
itself) runs through :func:`compare_fwd_bwd`. The harness:

* clones every input that ``requires_grad`` so reference and candidate
  run on independent autograd graphs (one ``.grad`` accumulator each);
* invokes the reference and candidate;
* asserts forward output matches within the supplied tolerance;
* calls ``.sum().backward()`` on each (with ``retain_graph=False``);
* asserts every cloned-and-grad-bearing input's gradient matches
  within the supplied tolerance;
* on mismatch, prints a structured diff (max abs / max rel error per
  leaf, with the eight worst entries by magnitude).

Tolerance budget defaults follow the plan-4 P24 design:

* ``fp32`` — ``atol=1e-5, rtol=1e-5`` for forward, same for backward.
* ``bf16`` — ``atol=2e-2, rtol=2e-2`` for forward, ``atol=5e-2,
  rtol=5e-2`` for backward (the looser bwd budget absorbs MQA
  atomic-add reordering and other accumulator-order delta).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

# ---------------------------------------------------------------------------
# Tolerance budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tol:
    """Numerical tolerance pair for ``torch.allclose``-style comparisons."""

    atol: float
    rtol: float

    def as_kwargs(self) -> dict:
        return {"atol": self.atol, "rtol": self.rtol}


FP32_TOL = Tol(atol=1e-5, rtol=1e-5)
BF16_FWD_TOL = Tol(atol=2e-2, rtol=2e-2)
BF16_BWD_TOL = Tol(atol=5e-2, rtol=5e-2)


def default_tols(dtype: torch.dtype) -> tuple[Tol, Tol]:
    """Return ``(fwd_tol, bwd_tol)`` for the supplied input dtype."""
    if dtype == torch.float32:
        return FP32_TOL, FP32_TOL
    if dtype == torch.bfloat16:
        return BF16_FWD_TOL, BF16_BWD_TOL
    if dtype == torch.float16:
        # Treat fp16 like bf16 for tolerance purposes (slightly tighter
        # mantissa, slightly tighter exponent — same ballpark).
        return BF16_FWD_TOL, BF16_BWD_TOL
    raise ValueError(f"No default tolerance preset for dtype {dtype!r}.")


# ---------------------------------------------------------------------------
# Input cloning + gradient comparison helpers
# ---------------------------------------------------------------------------


def _clone_for_grad(inputs: Mapping[str, Any]) -> dict:
    """Return a shallow-copied mapping with leaf tensors deep-cloned.

    Tensors that ``requires_grad`` are replaced with detached clones
    that re-enable ``requires_grad`` so the autograd graph for
    reference / candidate is independent. Non-tensor values pass
    through unchanged. Tensors that do NOT require grad pass through
    by reference (so the caller can share read-only tensors like masks
    across reference / candidate cheaply).
    """
    out: dict = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.requires_grad:
            out[k] = v.detach().clone().requires_grad_(True)
        else:
            out[k] = v
    return out


def _diag(name: str, ref: torch.Tensor, cand: torch.Tensor, tol: Tol) -> str:
    """Format a structured diff for an off-tolerance ``(ref, cand)`` pair."""
    if ref.shape != cand.shape:
        return f"{name}: shape mismatch — ref={tuple(ref.shape)} " f"vs cand={tuple(cand.shape)}"
    diff = (ref.float() - cand.float()).abs()
    # rel uses ref magnitude as denominator (matches torch.allclose
    # semantics: atol + rtol * |ref|).
    rel = diff / (ref.float().abs() + 1e-12)
    max_abs = diff.max().item()
    max_rel = rel.max().item()

    flat_diff = diff.flatten()
    k = min(8, flat_diff.numel())
    topk = torch.topk(flat_diff, k=k).indices.tolist()
    worst = []
    flat_ref = ref.float().flatten()
    flat_cand = cand.float().flatten()
    for idx in topk:
        worst.append(
            f"  [{idx}]: ref={flat_ref[idx].item():+.6g} "
            f"cand={flat_cand[idx].item():+.6g} "
            f"abs={(flat_ref[idx] - flat_cand[idx]).abs().item():+.6g}"
        )
    worst_block = "\n".join(worst)
    return (
        f"{name}: max_abs={max_abs:.6g} (atol={tol.atol:.6g}), "
        f"max_rel={max_rel:.6g} (rtol={tol.rtol:.6g}); 8 worst entries:\n" + worst_block
    )


def _assert_close(
    name: str,
    ref: torch.Tensor,
    cand: torch.Tensor,
    tol: Tol,
) -> None:
    if ref is None and cand is None:
        return
    if ref is None or cand is None:
        raise AssertionError(
            f"{name}: one side is None (ref is None: {ref is None}, " f"cand is None: {cand is None})"
        )
    # Match dtype before comparing — gradient dtypes can differ
    # slightly across paths (one side accumulates fp32, the other
    # bf16), so coerce to fp32 for the comparison itself.
    ref32 = ref.detach().float()
    cand32 = cand.detach().float()
    if ref32.shape != cand32.shape:
        raise AssertionError(_diag(name, ref32, cand32, tol))
    if torch.allclose(ref32, cand32, **tol.as_kwargs()):
        return
    raise AssertionError(_diag(name, ref32, cand32, tol))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compare_fwd_bwd(
    *,
    reference: Callable[..., torch.Tensor],
    candidate: Callable[..., torch.Tensor],
    inputs: Mapping[str, Any],
    fwd_tol: Tol,
    bwd_tol: Tol,
    grad_keys: Optional[Iterable[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run reference + candidate on independent input clones; assert match.

    Args:
        reference: callable taking ``**inputs`` and returning a single
            tensor (the "eager truth").
        candidate: callable with the same signature (the path under
            test).
        inputs: kwargs forwarded to both. Tensor values that
            ``requires_grad`` are independently cloned so the two
            backward passes do not interfere.
        fwd_tol: forward output tolerance.
        bwd_tol: backward gradient tolerance.
        grad_keys: optional override for which input keys to assert
            gradients on. Defaults to every tensor in ``inputs`` that
            ``requires_grad``.

    Returns:
        ``(out_ref, out_cand)`` for follow-up inspection by the caller
        (e.g., asserting non-NaN, asserting shape, etc.).

    Raises:
        AssertionError: on first forward or backward mismatch, with a
        structured diff in the message.
    """
    ref_inputs = _clone_for_grad(inputs)
    cand_inputs = _clone_for_grad(inputs)

    out_ref = reference(**ref_inputs)
    out_cand = candidate(**cand_inputs)

    _assert_close("forward output", out_ref, out_cand, fwd_tol)

    out_ref.sum().backward()
    out_cand.sum().backward()

    if grad_keys is None:
        grad_keys = [k for k, v in inputs.items() if isinstance(v, torch.Tensor) and v.requires_grad]
    for k in grad_keys:
        ref_t = ref_inputs[k]
        cand_t = cand_inputs[k]
        if not isinstance(ref_t, torch.Tensor) or not isinstance(cand_t, torch.Tensor):
            raise AssertionError(
                f"grad_keys[{k!r}] points at a non-tensor input "
                f"(ref={type(ref_t).__name__}, cand={type(cand_t).__name__})"
            )
        _assert_close(f"gradient[{k}]", ref_t.grad, cand_t.grad, bwd_tol)

    return out_ref, out_cand


__all__ = [
    "Tol",
    "FP32_TOL",
    "BF16_FWD_TOL",
    "BF16_BWD_TOL",
    "compare_fwd_bwd",
    "default_tols",
]

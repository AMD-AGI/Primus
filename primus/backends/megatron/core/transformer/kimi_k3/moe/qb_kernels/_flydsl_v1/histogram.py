###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Torch-facing FlyDSL Quantile Balancing statistic.

:func:`flydsl_compute_margin_histogram` has the same signature and the same
return types as :func:`..._eager.reference.compute_margin_histogram` and is
interchangeable with it. No autograd wrapper: the statistic runs under
``@torch.no_grad()`` inside ``QuantileBalancingMixin._accumulate_margin_histogram``
and the report freezes the bias at inference, so this computation has no
backward and never needs one. That is a property of the algorithm, not an
omission.

What is fused, and what is deliberately not
-------------------------------------------
The kernel takes over everything downstream of the cutoff. The **cutoff itself
stays in torch**, and that is a measured decision rather than a shortcut:

===========================  ============  =========================
op (4096 tokens x 32 experts)  median µs    where it goes
===========================  ============  =========================
``sigmoid``                          7.5   torch (bit-exactness)
``add_bias``                         7.9   torch (bit-exactness)
``topk`` (k+1)                      13.6   **torch** — library selection
``margin_sub``                       7.1   kernel
``scale_floor``                     14.7   kernel
``clamp_count`` (x2)                36.1   kernel
``clamp_to_int64``                  11.3   kernel
``offset_reshape``                   8.0   kernel
``bincount``                       102.2   kernel
===========================  ============  =========================

The six stages the kernel absorbs are 179.4 µs of the 208.4 µs total, i.e.
**86 %**; ``torch.topk`` is 6.5 %. Re-implementing a top-(k+1) selection in
FlyDSL would be the largest and least certain part of the work for the smallest
share of the win — and ``torch.topk`` is a tuned library kernel that a
hand-rolled ``O(E²)`` rank count (1024 comparisons per token at ``E = 32``,
802 816 at ``E = 896``) would not approach. ``sigmoid`` and ``add_bias`` stay in
torch for a different reason: the bin index is a ``floor``, so any last-bit
difference upstream of it moves a count into the neighbouring bin, and matching
``torch.sigmoid`` bit-for-bit in the kernel is not something to bet parity on.
The same discontinuity is why the kernel scales by ``1/width`` rather than
dividing — see the kernel module's docstring; matching the oracle beats being
more accurate than it.

So the honest description of this backend is: **one fused kernel replacing six
launches, with three cheap torch ops kept in front of it on purpose.**

The kernel is slower above ~6 K tokens, and that is guarded
-----------------------------------------------------------
The kernel's cost is **global-atomic contention**, which scales with tokens per
bin. ``torch.bincount`` privatises or sorts and does not, so the eager path barely
grows with the token count while the kernel grows linearly. Measured at 32
experts / 1024 bins:

===============  ==========  ==========  ========
tokens           eager µs    kernel µs   speedup
===============  ==========  ==========  ========
2 048                 246.5        68.2    3.61x
6 144                 233.0       107.9    2.16x
16 384                231.6       233.3    0.99x
32 768                290.8       466.2    **0.62x**
===============  ==========  ==========  ========

So selecting ``flydsl`` at a large micro-batch would make training *slower* while
looking like an optimisation, which is worse than having no kernel at all.
:func:`kernel_beats_eager` therefore gates the launch on the token count and
falls back to the eager path outside the measured win, logging once so the
decision is visible in the run's log rather than only in this docstring. The
threshold is :data:`KERNEL_MAX_TOKENS`; ``quantile_balancing_kernel_max_tokens``
overrides it, and ``0`` disables the guard entirely.

The threshold is a stopgap, not the answer. The fix is an LDS-privatised
histogram, which needs the grid re-mapped from ``(token, expert)`` pairs to
``(expert, token-slice)``; see ``flydsl_modules/FINDINGS.md`` §6.2.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

import torch

from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._eager.reference import (
    compute_margin_histogram,
    margin_cutoff,
)
from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1.qb_margin_histogram_kernel import (
    MAX_TOKENS,
    build_qb_margin_histogram,
)

logger = logging.getLogger(__name__)

__all__ = [
    "flydsl_compute_margin_histogram",
    "supports_histogram_inputs",
    "kernel_beats_eager",
    "KERNEL_MAX_TOKENS",
    "inject_defect",
]

#: Largest token count at which the kernel is measured faster than eager **at
#: both measured expert counts**. From the ladder in :func:`kernel_beats_eager`:
#: 32 experts cross over at ~16 384 and 896 experts at ~8 192, so 6 144 is the
#: last rung that wins on both. Deliberately the conservative choice — a guard
#: that can never make a run slower is worth more than one tuned to the shape in
#: front of it. ``quantile_balancing_kernel_max_tokens`` raises it for anyone who
#: wants the 1.2-2.2x still on the table at 32 experts.
KERNEL_MAX_TOKENS = 6144

_CACHE: Dict[Tuple[int, int, float, float, str], object] = {}
_CACHE_LOCK = threading.Lock()
#: One warning per process per reason, not one per MoE layer per microbatch.
_WARNED: set = set()

#: Test-only build-time defect for the current process; ``""`` in production.
_INJECT = ""


def inject_defect(name: str = "") -> None:
    """**Test only.** Make subsequent launches use a deliberately broken kernel.

    See :func:`..._flydsl_v1.mixer.inject_defect` for the rationale. The name is
    validated by the kernel builder, so a typo raises rather than quietly
    meaning "no defect". Call with no arguments to restore the correct kernel.
    """
    global _INJECT
    _INJECT = str(name or "")


def _get_kernel(num_experts: int, num_bins: int, lo: float, hi: float):
    key = (int(num_experts), int(num_bins), float(lo), float(hi), _INJECT)
    with _CACHE_LOCK:
        launch = _CACHE.get(key)
        if launch is None:
            launch = build_qb_margin_histogram(
                num_experts=key[0],
                num_bins=key[1],
                margin_min=key[2],
                margin_max=key[3],
                inject=key[4],
            )
            _CACHE[key] = launch
        return launch


def kernel_beats_eager(num_tokens: int, num_experts: int, max_tokens: int = KERNEL_MAX_TOKENS) -> bool:
    """Whether the kernel is expected to be faster than eager at this shape.

    Args:
        num_tokens: tokens in this microbatch.
        num_experts: routed experts. Accepted and deliberately unused — see below.
        max_tokens: threshold; ``0`` disables the guard.

    The measured ladder, 1024 bins, median of 30 (``bench_mem_and_ladder.py``):

    ==========  ====================  =====================
    tokens      speedup, 32 experts   speedup, 896 experts
    ==========  ====================  =====================
    2 048                     3.61x                  1.34x
    4 096                     2.84x                  1.13x
    **6 144**                 2.16x                  1.03x
    8 192                     1.74x                  **0.97x**
    12 288                    1.23x                  0.92x
    16 384                    **0.99x**              0.81x
    24 576                    0.81x                  0.71x
    32 768                    0.62x                  0.77x
    ==========  ====================  =====================

    Two things the ladder settles. The **cause** is atomic contention: the loss
    tracks counts per bin (``num_tokens / num_bins``), and both columns degrade
    monotonically in it while the eager column barely moves — ``torch.bincount``
    privatises or sorts and the kernel does not. The **crossover depends on
    ``num_experts``** after all, and not through contention: each expert's
    histogram row receives exactly ``num_tokens`` increments regardless of how
    many experts there are, so contention per address is identical. What differs
    is the eager baseline — ``torch.topk`` grows from 6.6 % of it at 32 experts to
    17 % at 896, so at 896 the kernel has less to win and crosses over sooner.

    The threshold is the last rung that wins in **both** columns, i.e. 6 144. It
    is a single number rather than a per-expert-count table because two points per
    column is enough to place a conservative bound and not enough to fit a curve.
    """
    if int(max_tokens) <= 0:
        return True
    return int(num_tokens) <= int(max_tokens)


def supports_histogram_inputs(scores: torch.Tensor, num_bins: int) -> Optional[str]:
    """``None`` when the kernel can run these inputs, else why it cannot."""
    if not scores.is_cuda:
        return "the kernel is a GPU kernel and the scores are on CPU"
    if scores.dtype != torch.float32:
        return f"scores dtype {scores.dtype} is not fp32"
    if scores.dim() != 2:
        return f"scores must be [num_tokens, num_experts], got {tuple(scores.shape)}"
    if scores.shape[0] > MAX_TOKENS:
        return f"num_tokens={scores.shape[0]} exceeds the int32 counter bound {MAX_TOKENS}"
    if int(num_bins) < 2:
        return f"num_bins={num_bins} must be >= 2"
    return None


def flydsl_compute_margin_histogram(
    scores: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    topk: int,
    num_bins: int,
    margin_min: float,
    margin_max: float,
    max_tokens: int = KERNEL_MAX_TOKENS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused-kernel margin histogram, interchangeable with the eager entry.

    Args / returns: see :func:`..._eager.reference.compute_margin_histogram`.
    ``max_tokens`` is the shape guard; see :func:`kernel_beats_eager`.

    Above the guard this **runs the eager path instead**, and warns once. It is a
    fallback to the *faster* implementation, not a silent degradation: selecting
    the kernel at 32 768 tokens would cost 1.6x and look like an optimisation,
    which is the one outcome worse than not having a kernel.

    Raises:
        ValueError: when the inputs are outside what the kernel supports at all
            (dtype, device, rank), naming the config fallback.
    """
    num_tokens, num_experts = scores.shape
    if num_tokens == 0:
        return (
            torch.zeros(num_experts, num_bins, dtype=torch.int64, device=scores.device),
            torch.zeros(2, dtype=torch.int64, device=scores.device),
        )

    if not kernel_beats_eager(num_tokens, num_experts, max_tokens):
        key = ("slow_regime", int(num_tokens))
        if key not in _WARNED:
            _WARNED.add(key)
            logger.warning(
                "[Kimi K3] quantile_balancing_backend='flydsl' but this microbatch has "
                "%d tokens, above the measured crossover of %d, so the EAGER path is "
                "being used. The kernel is atomic-contention bound and was measured at "
                "0.61x of eager at 32768 tokens (2.58x at 4096). Set "
                "quantile_balancing_kernel_max_tokens=0 to force the kernel anyway.",
                num_tokens,
                max_tokens,
            )
        return compute_margin_histogram(
            scores,
            expert_bias,
            topk=topk,
            num_bins=num_bins,
            margin_min=margin_min,
            margin_max=margin_max,
        )

    reason = supports_histogram_inputs(scores, num_bins)
    if reason is not None:
        raise ValueError(
            f"the FlyDSL Quantile Balancing histogram kernel cannot run these inputs: "
            f"{reason}. Select quantile_balancing_backend: eager."
        )

    # The cutoff stays in torch: bit-exact, and only 6.5 % of the cost. See the
    # module docstring for the per-op measurement behind that split.
    tau = margin_cutoff(scores, expert_bias, topk)

    # [E, B+2]: the bins, then the below-range and above-range counters.
    buf = torch.zeros(num_experts, int(num_bins) + 2, dtype=torch.int32, device=scores.device)
    _get_kernel(num_experts, num_bins, margin_min, margin_max)(
        scores.contiguous().reshape(-1),
        tau.contiguous().reshape(-1),
        buf.reshape(-1),
        int(num_tokens * num_experts),
    )

    hist = buf[:, :num_bins].to(torch.int64)
    clamped = buf[:, num_bins:].sum(dim=0).to(torch.int64)
    return hist, clamped

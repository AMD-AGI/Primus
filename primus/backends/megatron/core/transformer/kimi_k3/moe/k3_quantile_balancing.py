###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Quantile Balancing — Kimi K3's MoE load-balancing rule.

Kimi K3 does **not** use DeepSeek-V3's ``b <- b + u * sign(violation)`` bias
update. Tech report §2.3.3 replaces it wholesale: the expert bias is *set*
from a per-expert quantile of the routing margin. The report's own framing of
why (verbatim from the extracted text):

    The original method updates b with the fixed-step rule
    ``b_j^(t+1) = b_j^(t) + u * sign(...)`` for which u trades off slow
    adaptation against load oscillation. Maintaining balanced loads becomes
    more challenging as LatentMoE increases the routed expert pool to 896 per
    layer.

The rule
--------
With ``m`` tokens, ``n`` routed experts, top-``k`` selection, and target load
``q := m*k/n``:

.. code-block:: text

    s_i        = sigmoid(W_r x_i)                       raw router scores  [n]
    tau_i      = (k+1)-th largest of (s_i + b^(t))      the token's cutoff
    margin_ij  = s_ij - tau_i                            RAW score - BIASED cutoff
    b_hat_j    = -quantile_{1-k/n}( margin_{:,j} )       Eq. 14, line 1
    b^(t+1)    = b_hat - mean(b_hat)                     Eq. 14, line 2

Three details that are easy to get wrong, all forced by the report's prose
rather than by the (glyph-lossy) equation image:

1. **The cutoff is taken from the *biased* scores, the margin from the *raw*
   ones.** "The margins subtract the biased cutoff tau_i^(t) from the raw score
   s_ij, so the old bias enters the update only through the cutoffs." So this
   is a *set*, not an increment: ``b^(t+1)`` contains no additive ``b^(t)``.
2. **The sign.** The PDF drops the leading minus in Eq. 14, but the derivation
   fixes it: "the token count routed to expert j under a candidate bias
   b_hat_j is ``sum_i 1[s_ij + b_hat_j > tau_i]``, which is monotonically
   decreasing in the threshold ``-b_hat_j``. Assuming no ties, setting this
   count to q makes ``-b_hat_j`` the (q+1)-th largest margin." Hence the
   negation.
3. **Mean-centring is cosmetic, not corrective.** "the second line removes a
   common offset that leaves Top-k selection unchanged" — adding a constant to
   every ``b_j`` cannot change ``argtopk(s_i + b)``. It only stops the bias
   drifting. Configurable, on by default.

Taking the cutoff from a Top-(k+1) pass is the report's own trick: "Taking the
cutoff from Top-(k+1) routing avoids a separate token-side quantile." Upstream
Megatron routes with Top-k and there is no clean seam to widen it to Top-(k+1),
so this module runs its own ``torch.topk(..., k+1)`` over the biased scores.
That is *mathematically identical* — the first k entries of a Top-(k+1) are the
Top-k — at the cost of one extra top-k per MoE layer per microbatch.

Histogram estimation
--------------------
Also §2.3.3, and this is what settles the batch-scope question:

    At scale, the quantile in Eq. 14 spans **the full global batch**, whose
    margins number in the millions and are spread across **ranks and
    accumulation steps**, so gathering them for an exact quantile is not viable
    at training time. We instead read each expert's quantile from a histogram
    of its margins: **a single all-reduce** sums the per-rank bin counts, and
    the quantile is recovered from the pooled counts.

So the statistic is per global batch, accumulated over microbatches, reduced
once across ranks, and is **not** an EMA. That is exactly the cadence and the
collective at which Megatron already runs the sign-based rule, in
``_update_router_expert_bias`` (``finalize_model_grads.py:293-319``, called at
``:480-481``) — which is why :data:`k3_stable_latent_moe.QUANTILE_BALANCING_HOOK_SITE`
points there. The one thing upstream does not have is the *statistic*: it
gathers ``local_tokens_per_expert``, and Quantile Balancing needs routing
scores. :class:`QuantileBalancingMixin` adds a ``local_margin_histogram``
buffer beside it.

What the report does **not** say, and is therefore configurable
--------------------------------------------------------------
* **The binning.** Range and bin count are not stated anywhere in any
  extraction we have; the sentence that would have said so is truncated. The
  defaults here are ``[-1, 1]`` over 1024 uniform bins, chosen because
  ``sigmoid`` confines the raw score to ``(0, 1)`` and a mean-centred bias
  keeps ``tau`` close to that interval, and because the resulting bin width
  (1.95e-3) is the same order as DeepSeek-V3's ``moe_router_bias_update_rate``
  of 1e-3, i.e. the resolution the old rule was happy with. Out-of-range
  margins are clamped into the end bins and counted, so saturation is visible
  rather than silent.
* **Whether TP replication should be de-duplicated.** It does not matter: the
  all-reduce runs over TP x CP x DP (matching upstream's rule), so under TP the
  same token is counted once per TP rank — but a quantile is invariant to
  uniform duplication of the whole sample, so the estimate is unchanged.
* **The update cadence** is stated indirectly ("QB derives the next bias from a
  single forward pass" + global batch), so the default is every optimizer step.
  ``quantile_balancing_update_interval`` widens it, and
  ``quantile_balancing_ema_decay`` turns the set into an EMA, both off by
  default, so the two readings the report leaves open are reachable without
  editing code.

Memory: the histogram is ``num_experts x num_bins`` int64 **per MoE layer**
(8 MB per layer at 896 experts / 1024 bins), and the all-reduce sees all layers
stacked. Reduce ``quantile_balancing_num_bins`` if that matters.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import torch  # noqa: F401  # must precede any transformer_engine import

from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels import (
    QB_BACKENDS,
    compute_margin_histogram,
    margin_cutoff,
    resolve_qb_backend,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MARGIN_HISTOGRAM_BUFFER",
    "QuantileBalancingMixin",
    "make_quantile_balancing_router",
    "resolve_quantile_balancing_router",
    "compute_margin_histogram",
    "margin_cutoff",
    "quantile_from_histogram",
    "compute_quantile_bias",
    "collect_quantile_balancing_routers",
    "update_router_expert_bias_quantile",
    "quantile_balancing_enabled",
]

MARGIN_HISTOGRAM_BUFFER = "local_margin_histogram"

#: Value of ``moe_router_bias_update_rule`` that selects this module.
QUANTILE_RULE = "quantile"
#: Value that keeps DeepSeek-V3's sign-based rule (upstream's behaviour).
SIGN_RULE = "sign"


def quantile_balancing_enabled(config) -> bool:
    """Whether this config asks for Quantile Balancing.

    Requires the expert bias itself to be on: QB *is* the bias update rule, so
    without ``moe_router_enable_expert_bias`` there is nothing to update.
    """
    if not bool(getattr(config, "moe_router_enable_expert_bias", False)):
        return False
    return str(getattr(config, "moe_router_bias_update_rule", SIGN_RULE)) == QUANTILE_RULE


def _binning(config) -> Tuple[int, float, float]:
    num_bins = int(getattr(config, "quantile_balancing_num_bins", 1024))
    lo = float(getattr(config, "quantile_balancing_margin_min", -1.0))
    hi = float(getattr(config, "quantile_balancing_margin_max", 1.0))
    if num_bins < 2:
        raise ValueError(f"quantile_balancing_num_bins must be >= 2, got {num_bins}")
    if not hi > lo:
        raise ValueError(
            f"quantile_balancing_margin_max ({hi}) must exceed quantile_balancing_margin_min ({lo})"
        )
    return num_bins, lo, hi


# ---------------------------------------------------------------------------
# The statistic: a per-expert histogram of routing margins
# ---------------------------------------------------------------------------


# :func:`compute_margin_histogram` now lives in
# :mod:`.qb_kernels._eager.reference` and is re-exported above, unchanged. It is
# the eager backend *and* the oracle the fused FlyDSL kernel in
# :mod:`.qb_kernels._flydsl_v1` is checked against; every existing importer
# (including ``test_quantile_balancing.py``) still finds it at this name.


def quantile_from_histogram(
    hist: torch.Tensor, q: float, *, margin_min: float, margin_max: float
) -> torch.Tensor:
    """Recover the ``q``-quantile of each row's distribution from bin counts.

    Args:
        hist: ``[..., num_bins]`` non-negative counts over uniform bins
            spanning ``[margin_min, margin_max]``.
        q: quantile in ``[0, 1]``.

    Returns:
        ``[...]`` estimated quantile values.

    The value is linearly interpolated inside the bin that straddles the
    target rank. The report does not say how the quantile is recovered from
    the pooled counts, only that it is; linear interpolation is the standard
    choice and is exact for a uniform within-bin distribution. Bin-lower-edge
    would bias every bias term low by up to one bin width, i.e. by the same
    order as DeepSeek-V3's whole update step.
    """
    num_bins = hist.shape[-1]
    cum = hist.to(torch.float64).cumsum(dim=-1)
    total = cum[..., -1:]
    target = total * float(q)

    idx = torch.searchsorted(cum.contiguous(), target.contiguous(), right=False)
    idx = idx.clamp_(max=num_bins - 1)

    cum_prev = torch.gather(cum, -1, (idx - 1).clamp_(min=0))
    cum_prev = torch.where(idx == 0, torch.zeros_like(cum_prev), cum_prev)
    bin_count = torch.gather(hist.to(torch.float64), -1, idx)

    frac = ((target - cum_prev) / bin_count.clamp_(min=1.0)).clamp_(0.0, 1.0)
    width = (margin_max - margin_min) / num_bins
    value = margin_min + width * (idx.to(torch.float64) + frac)
    # An all-zero row has no distribution; report the midpoint so the caller's
    # "did anything happen" check (total == 0) is the thing that decides.
    value = torch.where(total > 0, value, torch.full_like(value, 0.5 * (margin_min + margin_max)))
    return value.squeeze(-1)


def compute_quantile_bias(
    hist: torch.Tensor,
    *,
    topk: int,
    num_experts: int,
    margin_min: float,
    margin_max: float,
    center: bool = True,
) -> torch.Tensor:
    """Eq. 14: turn pooled margin histograms into the next expert bias.

    Args:
        hist: ``[..., num_experts, num_bins]`` pooled counts.
        topk / num_experts: ``k`` and ``n``; the quantile is ``1 - k/n``.
        center: subtract the mean, i.e. Eq. 14's second line.

    Returns:
        ``[..., num_experts]`` float64 bias.
    """
    q = 1.0 - float(topk) / float(num_experts)
    tau_q = quantile_from_histogram(hist, q, margin_min=margin_min, margin_max=margin_max)
    bias = -tau_q
    if center:
        bias = bias - bias.mean(dim=-1, keepdim=True)
    return bias


# ---------------------------------------------------------------------------
# Router side: stash the statistic next to local_tokens_per_expert
# ---------------------------------------------------------------------------


class QuantileBalancingMixin:
    """Adds a per-expert margin histogram to a :class:`TopKRouter`.

    Mixed in *ahead* of the router base class, so ``routing`` runs the base
    implementation and then accumulates. It is deliberately a mixin rather
    than a subclass of a specific router: Primus swaps upstream
    ``TopKRouter`` for ``PrimusTopKRouter`` at ``build_args`` time
    (``moe_patches/topk_router_patches.py:57-75``), and a hard-coded base
    would either lose those features in production or drag
    ``megatron.training.get_args()`` into unit tests that have no args.
    :func:`resolve_quantile_balancing_router` picks the base up from the same
    dataclass slot the patch rewrites.

    The buffer mirrors ``local_tokens_per_expert`` (``router.py:172-181``):
    non-persistent, allocated on the current device in ``__init__``, zeroed by
    whoever consumes it.
    """

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.qb_num_bins, self.qb_margin_min, self.qb_margin_max = _binning(config)
        self.qb_enabled = quantile_balancing_enabled(config)

        # Resolve the statistic's kernel once, at construction, the same way
        # KimiDeltaAttention and AttentionResidualMixer resolve theirs: a missing
        # optional dependency then surfaces while the model is being built rather
        # than on the first forward.
        self.qb_backend_name = str(getattr(config, "quantile_balancing_backend", "eager") or "eager")
        if self.qb_backend_name not in QB_BACKENDS:
            raise ValueError(
                f"quantile_balancing_backend must be one of {list(QB_BACKENDS)}; "
                f"got {self.qb_backend_name!r}."
            )
        # The shape guard is bound at resolve time, so the hot path calls a
        # single-signature callable. The FlyDSL histogram is measured SLOWER than
        # eager above ~8k tokens per microbatch, so selecting it at a large
        # micro-batch without this would silently cost throughput.
        self.qb_histogram = resolve_qb_backend(
            self.qb_backend_name,
            max_tokens=getattr(config, "quantile_balancing_kernel_max_tokens", None),
        )

        if not self.enable_expert_bias:
            raise ValueError(
                "Quantile Balancing is a rule for updating e_score_correction_bias, "
                "so moe_router_enable_expert_bias must be True. Set "
                "moe_router_bias_update_rule: sign to use a router without it."
            )

        device = self.expert_bias.device
        self.register_buffer(
            MARGIN_HISTOGRAM_BUFFER,
            torch.zeros(self.config.num_moe_experts, self.qb_num_bins, dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "local_margin_clamped",
            torch.zeros(2, dtype=torch.int64, device=device),
            persistent=False,
        )

    def routing(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, **kwargs):
        """Base routing, then accumulate this microbatch's margins.

        The ``torch.is_grad_enabled()`` half of the gate has to be read *here*.
        ``_accumulate_margin_histogram`` runs under ``@torch.no_grad()``, so the
        same call inside it would always return ``False`` and the histogram
        would silently stay empty. Upstream's ``_apply_expert_bias``
        (``router.py:580-584``) gets away with the inline check only because it
        opens the ``no_grad`` block *after* testing.
        """
        grad_enabled = torch.is_grad_enabled()
        probs, routing_map = super().routing(logits, padding_mask=padding_mask, **kwargs)
        if self.qb_enabled and self.training and grad_enabled:
            self._accumulate_margin_histogram(logits, padding_mask)
        return probs, routing_map

    @torch.no_grad()
    def _accumulate_margin_histogram(
        self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor]
    ) -> None:
        """Add this microbatch's margins to the running histogram.

        Gated by the caller exactly like ``_apply_expert_bias``
        (``router.py:580``): training mode and grad enabled, so evaluation
        passes contribute nothing — the report freezes the bias at inference —
        and the two statistics stay in step.
        """
        num_experts = self.config.num_moe_experts
        flat = logits.detach().reshape(-1, num_experts)

        if padding_mask is not None:
            keep = ~padding_mask.reshape(-1)
            if keep.numel() == flat.shape[0]:
                flat = flat[keep]
            if flat.shape[0] == 0:
                return

        # Same score as topk_routing_with_score_function's sigmoid branch
        # (moe_utils.py:773): fp32 sigmoid of the gating output, no
        # normalisation. KimiMoEGate does the same (modeling_kimi_linear.py:712).
        scores = torch.sigmoid(flat.float())
        hist, clamped = self.qb_histogram(
            scores,
            self.expert_bias,
            topk=self.topk,
            num_bins=self.qb_num_bins,
            margin_min=self.qb_margin_min,
            margin_max=self.qb_margin_max,
        )
        getattr(self, MARGIN_HISTOGRAM_BUFFER).add_(hist)
        self.local_margin_clamped.add_(clamped)


_ROUTER_CACHE: dict = {}


def make_quantile_balancing_router(base_cls: type) -> type:
    """Return ``base_cls`` with :class:`QuantileBalancingMixin` mixed in.

    Cached, so repeated calls return the *same* class object — Megatron
    compares router classes by identity in places
    (``topk_router_patches.py:181``), and a fresh type per MoE layer would
    also defeat any ``isinstance`` check.
    """
    if issubclass(base_cls, QuantileBalancingMixin):
        return base_cls
    cached = _ROUTER_CACHE.get(base_cls)
    if cached is None:
        cached = type(
            f"QuantileBalancing{base_cls.__name__}",
            (QuantileBalancingMixin, base_cls),
            {"__doc__": QuantileBalancingMixin.__doc__},
        )
        _ROUTER_CACHE[base_cls] = cached
    return cached


def resolve_quantile_balancing_router() -> type:
    """Build the QB router on top of whichever router class is in force.

    Reads ``MoESubmodules.router``'s dataclass default, which is upstream
    ``TopKRouter`` normally and ``PrimusTopKRouter`` once
    ``megatron.moe.primus_topk_router`` has been applied
    (``topk_router_patches.py:80-96``). Composing rather than hard-coding is
    what lets the same class work in a unit test with no Megatron args and in
    a real run with the Primus router's softcap / fused paths.
    """
    from megatron.core.transformer.moe.moe_layer import MoESubmodules

    field = MoESubmodules.__dataclass_fields__["router"]
    base = field.default
    if not isinstance(base, type):  # pragma: no cover - upstream contract
        from megatron.core.transformer.moe.router import TopKRouter

        base = TopKRouter
    return make_quantile_balancing_router(base)


# ---------------------------------------------------------------------------
# Global-batch side: the replacement for _update_router_expert_bias
# ---------------------------------------------------------------------------


def collect_quantile_balancing_routers(model: Sequence[torch.nn.Module]) -> List[torch.nn.Module]:
    """Every training-mode router in ``model`` that carries a margin histogram.

    Mirrors ``_update_router_expert_bias``'s own walk
    (``finalize_model_grads.py:300-308``), including its ``module.training``
    filter — online distillation puts the teacher in eval mode and its bias
    must not move.
    """
    from megatron.core.utils import get_attr_wrapped_model

    routers: List[torch.nn.Module] = []
    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, "modules")():
            if not hasattr(module, MARGIN_HISTOGRAM_BUFFER):
                continue
            if getattr(module, "expert_bias", None) is None:
                continue
            if not module.training:
                continue
            routers.append(module)
    return routers


def update_router_expert_bias_quantile(
    model: Sequence[torch.nn.Module],
    config,
    *,
    group: Optional["torch.distributed.ProcessGroup"] = None,
    step: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Set every router's ``expert_bias`` from its pooled margin histogram.

    Drop-in replacement for ``_update_router_expert_bias``
    (``finalize_model_grads.py:293-319``), called at the same point and with
    the same collective group.

    Args:
        model: the model chunks, as ``finalize_model_grads`` passes them.
        config: the transformer config.
        group: all-reduce group. Defaults to TP x CP x DP, matching
            ``get_updated_expert_bias`` (``moe_utils.py:1134-1138``).
        step: optimizer step, used only for
            ``quantile_balancing_update_interval``. ``None`` means "update now".

    Returns:
        The new bias, ``[num_routers, num_experts]``, or ``None`` if nothing
        was updated. Returned rather than only written so tests can inspect it.
    """
    routers = collect_quantile_balancing_routers(model)
    if not routers:
        return None

    interval = max(1, int(getattr(config, "quantile_balancing_update_interval", 1)))
    if step is not None and interval > 1 and (step % interval) != 0:
        # Deliberately do NOT zero the histograms: a longer interval is meant
        # to widen the sample window, not to throw the extra samples away.
        return None

    num_bins, lo, hi = _binning(config)
    stacked = torch.stack([getattr(r, MARGIN_HISTOGRAM_BUFFER) for r in routers], dim=0)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if group is None:
            from megatron.core import parallel_state

            group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
        # The single all-reduce of §2.3.3. Counts are additive, so the pooled
        # histogram is the global batch's histogram; and a quantile is
        # invariant to the uniform duplication TP introduces.
        torch.distributed.all_reduce(stacked, group=group)

    new_bias = compute_quantile_bias(
        stacked,
        topk=int(config.moe_router_topk),
        num_experts=int(config.num_moe_experts),
        margin_min=lo,
        margin_max=hi,
        center=bool(getattr(config, "quantile_balancing_center_bias", True)),
    )

    empty = stacked.sum(dim=(-1, -2)) == 0
    ema = getattr(config, "quantile_balancing_ema_decay", None)

    for i, router in enumerate(routers):
        if bool(empty[i]):
            # No tokens reached this router this global batch (e.g. a PP stage
            # that ran no microbatches). Leave the bias alone.
            continue
        target = new_bias[i].to(router.expert_bias.dtype)
        if ema is not None:
            decay = float(ema)
            router.expert_bias.mul_(decay).add_(target, alpha=1.0 - decay)
        else:
            router.expert_bias.copy_(target)

    for router in routers:
        getattr(router, MARGIN_HISTOGRAM_BUFFER).zero_()
        router.local_margin_clamped.zero_()

    return new_bias


def iter_margin_histograms(model: Iterable[torch.nn.Module]):
    """Yield ``(router, histogram)`` for logging/diagnostics."""
    for router in collect_quantile_balancing_routers(list(model)):
        yield router, getattr(router, MARGIN_HISTOGRAM_BUFFER)

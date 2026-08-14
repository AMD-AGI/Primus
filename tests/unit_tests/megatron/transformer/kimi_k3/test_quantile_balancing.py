###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for Quantile Balancing (WP10), Kimi K3's MoE balancing rule.

Reference: tech report §2.3.3 / Eq. 14, transcribed and argued in
``primus/backends/megatron/core/transformer/kimi_k3/moe/k3_quantile_balancing.py``.

What each group pins down:

exactness against a brute-force quantile
    The histogram is an *approximation*, so the tests compare it against an
    exact ``torch.quantile`` over the same margins and require agreement to
    within one bin width. That is the only honest tolerance, and it also means
    a binning bug (off-by-one edge, wrong width, wrong interpolation) fails
    rather than passes with a fudge factor.

the sign and the reference point
    ``b_j = -quantile(...)``, and the margin is ``raw score - biased cutoff``.
    Both are things the PDF extraction lost and the prose recovered, so both
    get a test that fails if the sign flips or the cutoff is taken from the
    raw scores instead of the biased ones.

it actually balances
    Given a deliberately imbalanced score matrix, one QB update must move the
    resulting top-k load towards uniform. This is the property the whole work
    package exists for and it is checked directly, not via a proxy.

it is off by default
    ``moe_router_bias_update_rule`` defaults to ``sign``; the spec factory must
    then build a stock router with no histogram, so nothing in the tree changes
    behaviour until the rule is selected.
"""

from __future__ import annotations

import math
import os
import socket

import pytest
import torch  # noqa: F401  # must precede any transformer_engine import
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_quantile_balancing import (
    QuantileBalancingMixin,
    collect_quantile_balancing_routers,
    compute_margin_histogram,
    compute_quantile_bias,
    make_quantile_balancing_router,
    quantile_balancing_enabled,
    quantile_from_histogram,
    update_router_expert_bias_quantile,
)

HIDDEN = 64
NUM_EXPERTS = 16
TOPK = 4
NUM_TOKENS = 512


@pytest.fixture(autouse=True, scope="module")
def _primus_logger(tmp_path_factory):
    """The patch body calls ``log_rank_0``, which needs Primus's logger global.

    Same remedy as ``tests/unit_tests/configs/test_kimi_k3_yaml.py:106-133``.
    ``setup_logger`` is ``@call_once``, so this is a no-op if another suite got
    there first.
    """
    from primus.core.utils import logger

    logger.setup_logger(
        logger.LoggerConfig(
            exp_root_path=str(tmp_path_factory.mktemp("kimi_k3_qb_logs")),
            work_group="develop",
            user_name="root",
            exp_name="unittest",
            module_name="UT-kimi-k3-qb",
            file_sink_level="DEBUG",
            stderr_sink_level="INFO",
            node_ip="localhost",
            rank=os.environ.get("RANK", 0),
            world_size=os.environ.get("WORLD_SIZE", 1),
        ),
        is_head=False,
    )


NUM_BINS = 1024
MARGIN_MIN = -1.0
MARGIN_MAX = 1.0
BIN_WIDTH = (MARGIN_MAX - MARGIN_MIN) / NUM_BINS


# ---------------------------------------------------------------------------
# Pure-tensor reference: the rule written out directly from Eq. 14
# ---------------------------------------------------------------------------


def reference_quantile_bias(scores: torch.Tensor, expert_bias: torch.Tensor, topk: int):
    """Eq. 14 with an exact quantile, no histogram.

    ``tau_i`` = (k+1)-th largest of ``s_i + b``; ``margin_ij = s_ij - tau_i``;
    ``b_hat_j = -quantile_{1-k/n}(margin_{:,j})``; then mean-centre.
    """
    num_experts = scores.shape[1]
    biased = scores + expert_bias
    tau = torch.topk(biased, k=topk + 1, dim=1).values[:, -1:]
    margins = scores - tau
    q = 1.0 - topk / num_experts
    # torch.quantile's default 'linear' interpolation over the empirical CDF is
    # the continuous analogue of the histogram's within-bin interpolation.
    tau_q = torch.quantile(margins.double(), q, dim=0)
    bias = -tau_q
    return bias - bias.mean(), margins


def random_scores(num_tokens=NUM_TOKENS, num_experts=NUM_EXPERTS, *, skew=0.0, device="cpu"):
    """Sigmoid scores, optionally skewed so expert 0 is over-subscribed."""
    g = torch.Generator(device="cpu").manual_seed(1234)
    logits = torch.randn(num_tokens, num_experts, generator=g).to(device)
    if skew:
        logits[:, 0] += skew
        logits[:, 1] += 0.6 * skew
    return torch.sigmoid(logits.double()).float()


# ---------------------------------------------------------------------------
# quantile_from_histogram
# ---------------------------------------------------------------------------


def test_quantile_histogram_matches_exact_quantile_on_uniform_data() -> None:
    """A uniform sample's q-quantile is q, up to one bin."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(200_000, generator=g) * 2 - 1  # U[-1, 1)
    idx = ((x - MARGIN_MIN) / BIN_WIDTH).floor().clamp(0, NUM_BINS - 1).long()
    hist = torch.bincount(idx, minlength=NUM_BINS).view(1, NUM_BINS)

    for q in (0.5, 0.9, 0.982, 0.99):
        got = quantile_from_histogram(hist, q, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX)
        exact = torch.quantile(x.double(), q)
        assert abs(float(got) - float(exact)) <= BIN_WIDTH, (
            f"q={q}: histogram {float(got):.6f} vs exact {float(exact):.6f}, " f"bin width {BIN_WIDTH:.6f}"
        )


def test_quantile_histogram_is_monotone_in_q() -> None:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(50_000, generator=g).clamp(-0.99, 0.99)
    idx = ((x - MARGIN_MIN) / BIN_WIDTH).floor().clamp(0, NUM_BINS - 1).long()
    hist = torch.bincount(idx, minlength=NUM_BINS).view(1, NUM_BINS)

    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.982]
    vals = [float(quantile_from_histogram(hist, q, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX)) for q in qs]
    assert vals == sorted(vals)


def test_quantile_histogram_handles_an_empty_row() -> None:
    hist = torch.zeros(3, NUM_BINS, dtype=torch.int64)
    out = quantile_from_histogram(hist, 0.982, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX)
    assert out.shape == (3,)
    assert torch.allclose(out, torch.full((3,), 0.5 * (MARGIN_MIN + MARGIN_MAX), dtype=out.dtype))


def test_quantile_histogram_is_batched_over_leading_dims() -> None:
    """``[num_layers, num_experts, num_bins]`` is the shape the hook uses."""
    hist = torch.zeros(2, 3, NUM_BINS, dtype=torch.int64)
    # Row (l, e) puts all its mass in bin 100 * (l + 1) + 10 * e.
    for lay in range(2):
        for exp in range(3):
            hist[lay, exp, 100 * (lay + 1) + 10 * exp] = 7
    out = quantile_from_histogram(hist, 0.5, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX)
    assert out.shape == (2, 3)
    for lay in range(2):
        for exp in range(3):
            b = 100 * (lay + 1) + 10 * exp
            lo_edge = MARGIN_MIN + BIN_WIDTH * b
            assert lo_edge <= float(out[lay, exp]) <= lo_edge + BIN_WIDTH


# ---------------------------------------------------------------------------
# compute_margin_histogram -- the statistic itself
# ---------------------------------------------------------------------------


def test_margin_histogram_uses_the_biased_cutoff_and_raw_scores() -> None:
    """§2.3.3: "the margins subtract the biased cutoff from the raw score".

    Recomputes the margins by hand and checks the histogram is exactly their
    binning. A version that used the *biased* score as the numerator, or the
    unbiased cutoff, disagrees.
    """
    scores = random_scores()
    bias = torch.linspace(-0.2, 0.2, NUM_EXPERTS)

    hist, clamped = compute_margin_histogram(
        scores, bias, topk=TOPK, num_bins=NUM_BINS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
    )

    tau = torch.topk(scores + bias, k=TOPK + 1, dim=1).values[:, -1:]
    margins = scores - tau
    idx = ((margins - MARGIN_MIN) / BIN_WIDTH).floor().clamp(0, NUM_BINS - 1).long()
    expected = torch.zeros(NUM_EXPERTS, NUM_BINS, dtype=torch.int64)
    for e in range(NUM_EXPERTS):
        expected[e] = torch.bincount(idx[:, e], minlength=NUM_BINS)

    assert torch.equal(hist, expected)
    assert hist.sum().item() == NUM_TOKENS * NUM_EXPERTS
    assert clamped.tolist() == [0, 0]

    # A wrong reference point must not produce the same histogram.
    wrong_tau = torch.topk(scores, k=TOPK + 1, dim=1).values[:, -1:]
    wrong_idx = ((scores - wrong_tau - MARGIN_MIN) / BIN_WIDTH).floor().clamp(0, NUM_BINS - 1).long()
    wrong = torch.stack([torch.bincount(wrong_idx[:, e], minlength=NUM_BINS) for e in range(NUM_EXPERTS)])
    assert not torch.equal(hist, wrong), (
        "the unbiased cutoff produced the same histogram; the test cannot tell "
        "the two reference points apart on this data"
    )


def test_margin_histogram_counts_exactly_topk_positive_margins_per_token() -> None:
    """The cutoff is the (k+1)-th score, so exactly k margins are positive.

    That identity is the whole reason the report can take the cutoff from a
    Top-(k+1) pass instead of a separate token-side quantile.
    """
    scores = random_scores()
    bias = torch.zeros(NUM_EXPERTS)
    tau = torch.topk(scores + bias, k=TOPK + 1, dim=1).values[:, -1:]
    margins = scores - tau
    assert torch.all((margins > 0).sum(dim=1) == TOPK)


def test_margin_histogram_reports_clamping() -> None:
    """Margins outside the range must be counted, not silently absorbed."""
    scores = random_scores()
    bias = torch.zeros(NUM_EXPERTS)
    hist, clamped = compute_margin_histogram(
        scores, bias, topk=TOPK, num_bins=64, margin_min=-0.05, margin_max=0.05
    )
    assert hist.sum().item() == NUM_TOKENS * NUM_EXPERTS
    assert clamped.sum().item() > 0, "a deliberately narrow range clamped nothing"


# ---------------------------------------------------------------------------
# compute_quantile_bias -- Eq. 14 end to end
# ---------------------------------------------------------------------------


def test_quantile_bias_matches_the_exact_rule() -> None:
    """Histogram-estimated Eq. 14 agrees with an exact-quantile Eq. 14."""
    scores = random_scores(num_tokens=8192)
    bias = torch.zeros(NUM_EXPERTS)

    hist, _ = compute_margin_histogram(
        scores, bias, topk=TOPK, num_bins=NUM_BINS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
    )
    got = compute_quantile_bias(
        hist,
        topk=TOPK,
        num_experts=NUM_EXPERTS,
        margin_min=MARGIN_MIN,
        margin_max=MARGIN_MAX,
        center=True,
    )
    expected, _ = reference_quantile_bias(scores.double(), bias.double(), TOPK)

    assert got.shape == (NUM_EXPERTS,)
    assert torch.max(torch.abs(got - expected)).item() <= BIN_WIDTH, (
        f"max deviation {torch.max(torch.abs(got - expected)).item():.6f} exceeds one "
        f"bin width {BIN_WIDTH:.6f}\n got={got}\n exp={expected}"
    )


def test_quantile_bias_is_the_negation_of_the_quantile() -> None:
    """The sign the PDF dropped. An over-subscribed expert gets a lower bias.

    Skewing expert 0's scores upward raises its margins, so its
    ``1-k/n`` quantile is higher, so its bias must go *down* relative to the
    others. If the negation were missing this test reads the opposite way.
    """
    scores = random_scores(num_tokens=8192, skew=2.5)
    bias = torch.zeros(NUM_EXPERTS)
    hist, _ = compute_margin_histogram(
        scores, bias, topk=TOPK, num_bins=NUM_BINS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
    )
    new_bias = compute_quantile_bias(
        hist, topk=TOPK, num_experts=NUM_EXPERTS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
    )

    load = torch.zeros(NUM_EXPERTS)
    top = torch.topk(scores, k=TOPK, dim=1).indices
    load.scatter_add_(0, top.reshape(-1), torch.ones(top.numel()))

    busiest = int(torch.argmax(load))
    quietest = int(torch.argmin(load))
    assert float(new_bias[busiest]) < float(new_bias[quietest]), (
        f"bias for the busiest expert ({busiest}, load {load[busiest]}) is not below "
        f"the quietest ({quietest}, load {load[quietest]}); the sign of Eq. 14 is wrong"
    )


def test_quantile_bias_centers_to_zero_mean_when_asked() -> None:
    scores = random_scores(num_tokens=4096, skew=1.5)
    bias = torch.zeros(NUM_EXPERTS)
    hist, _ = compute_margin_histogram(
        scores, bias, topk=TOPK, num_bins=NUM_BINS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
    )
    centered = compute_quantile_bias(
        hist, topk=TOPK, num_experts=NUM_EXPERTS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX, center=True
    )
    raw = compute_quantile_bias(
        hist, topk=TOPK, num_experts=NUM_EXPERTS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX, center=False
    )
    assert abs(float(centered.mean())) < 1e-12
    # Eq. 14 line 2 is a *uniform* shift, i.e. it must not change any pairwise
    # difference -- that is precisely why it "leaves Top-k selection unchanged".
    assert torch.allclose(centered - centered[0], raw - raw[0], atol=1e-12)


def _load_under_bias(scores, bias):
    top = torch.topk(scores + bias, k=TOPK, dim=1).indices
    out = torch.zeros(NUM_EXPERTS)
    out.scatter_add_(0, top.reshape(-1), torch.ones(top.numel()))
    return out


def _max_relative_imbalance(load):
    target = load.sum() / load.numel()
    return float(torch.max(torch.abs(load - target)) / target)


def _load_entropy(load):
    p = load / load.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def test_quantile_bias_balances_an_imbalanced_router() -> None:
    """QB drives the top-k load towards uniform. The point of WP10.

    One step is derived to hit the target load *exactly* only if the cutoffs
    ``tau_i`` stay put, and they do not: ``tau_i`` is the (k+1)-th largest
    **biased** score, so applying the new bias moves it. The rule is therefore
    a fixed-point iteration, which is consistent with the report applying it
    once per global batch throughout training rather than once. So this test
    asserts a large single-step improvement *and* convergence over a handful
    of iterations, which is what the mechanism actually claims.
    """
    scores = random_scores(num_tokens=8192, skew=3.0)
    bias = torch.zeros(NUM_EXPERTS)

    trajectory = [_max_relative_imbalance(_load_under_bias(scores, bias))]
    entropies = [_load_entropy(_load_under_bias(scores, bias))]
    for _ in range(5):
        hist, _ = compute_margin_histogram(
            scores, bias, topk=TOPK, num_bins=NUM_BINS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
        )
        bias = compute_quantile_bias(
            hist, topk=TOPK, num_experts=NUM_EXPERTS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
        ).float()
        load = _load_under_bias(scores, bias)
        trajectory.append(_max_relative_imbalance(load))
        entropies.append(_load_entropy(load))

    assert (
        trajectory[1] < 0.5 * trajectory[0]
    ), f"one QB step barely moved the load: imbalance trajectory {trajectory}"
    assert trajectory[-1] < 0.05, f"QB did not converge: imbalance trajectory {trajectory}"
    assert entropies[-1] > entropies[0], f"entropy trajectory {entropies}"
    # Uniform load over n experts has entropy ln(n); QB should get very close.
    assert entropies[-1] > math.log(NUM_EXPERTS) - 1e-3, f"entropy trajectory {entropies}"


def test_quantile_bias_beats_one_sign_step_from_the_same_state() -> None:
    """QB is a *set*; the sign rule is a fixed step of ``moe_router_bias_update_rate``.

    From the same imbalanced state and in a single update, QB must get closer
    to the target load. This is the report's stated motivation ("u trades off
    slow adaptation against load oscillation") reduced to a test.
    """
    scores = random_scores(num_tokens=8192, skew=3.0)
    bias = torch.zeros(NUM_EXPERTS)
    before = _load_under_bias(scores, bias)

    hist, _ = compute_margin_histogram(
        scores, bias, topk=TOPK, num_bins=NUM_BINS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
    )
    qb_bias = compute_quantile_bias(
        hist, topk=TOPK, num_experts=NUM_EXPERTS, margin_min=MARGIN_MIN, margin_max=MARGIN_MAX
    ).float()

    # get_updated_expert_bias (moe_utils.py:1119-1142) all-reduces before doing
    # its arithmetic, so reproduce lines 1139-1141 locally rather than standing
    # up a process group just to compare one step.
    average = before.sum(dim=-1, keepdim=True) / before.shape[-1]
    sign_bias = bias + torch.sign(average - before) * 1e-3

    qb_err = _max_relative_imbalance(_load_under_bias(scores, qb_bias))
    sign_err = _max_relative_imbalance(_load_under_bias(scores, sign_bias))
    base_err = _max_relative_imbalance(before)

    assert qb_err < sign_err, (
        f"one QB step ({qb_err:.4f}) did not beat one sign step " f"({sign_err:.4f}); baseline {base_err:.4f}"
    )


# ---------------------------------------------------------------------------
# Router integration
# ---------------------------------------------------------------------------


def _k3_config(rule: str = "quantile", **kw):
    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    base = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        kv_channels=8,
        ffn_hidden_size=HIDDEN,
        moe_ffn_hidden_size=HIDDEN,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=1.0,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_force_load_balancing=False,
        moe_router_bias_update_rule=rule,
        quantile_balancing_num_bins=NUM_BINS,
        quantile_balancing_margin_min=MARGIN_MIN,
        quantile_balancing_margin_max=MARGIN_MAX,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        params_dtype=torch.float32,
        sequence_parallel=False,
    )
    base.update(kw)
    return KimiK3TransformerConfig(**base)


def test_quantile_balancing_enabled_predicate() -> None:
    assert quantile_balancing_enabled(_k3_config("quantile"))
    assert not quantile_balancing_enabled(_k3_config("sign"))


def test_quantile_config_rejects_a_bad_rule() -> None:
    with pytest.raises(ValueError, match="moe_router_bias_update_rule"):
        _k3_config("ema")


def test_quantile_config_rejects_quantile_without_expert_bias() -> None:
    with pytest.raises(ValueError, match="moe_router_enable_expert_bias"):
        _k3_config("quantile", moe_router_enable_expert_bias=False)


def test_quantile_router_factory_is_cached_and_idempotent() -> None:
    from megatron.core.transformer.moe.router import TopKRouter

    a = make_quantile_balancing_router(TopKRouter)
    b = make_quantile_balancing_router(TopKRouter)
    assert a is b, "a fresh class per call would break identity checks upstream"
    assert make_quantile_balancing_router(a) is a
    assert issubclass(a, QuantileBalancingMixin)
    assert issubclass(a, TopKRouter)


_GPU_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TopKRouter allocates its buffers on torch.cuda.current_device() "
    "in __init__ (router.py:172-189).",
)


@pytest.fixture()
def moe_parallel_state():
    """TP=PP=EP=CP=1; same shape as ``test_stable_latent_moe.py``."""
    from megatron.core import parallel_state as ps

    if not torch.distributed.is_initialized():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        try:
            torch.distributed.init_process_group(
                backend="nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0
            )
        except Exception as exc:  # pragma: no cover - environment guard
            pytest.skip(f"could not initialize torch.distributed: {exc}")

    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )
    from megatron.core.tensor_parallel import random as tp_random

    try:
        tp_random.initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    except (ImportError, AssertionError):  # pragma: no cover - environment guard
        tp_random.initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)
    tp_random.model_parallel_cuda_manual_seed(42)

    yield

    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()


def _build_qb_router(config):
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.router import TopKRouter

    cls = make_quantile_balancing_router(TopKRouter)
    return cls(config=config, pg_collection=get_default_pg_collection()).cuda()


@_GPU_ONLY
def test_quantile_router_accumulates_across_microbatches(moe_parallel_state) -> None:
    """The histogram is a *global-batch* statistic, so it must accumulate.

    §2.3.3: the margins are "spread across ranks and accumulation steps".
    """
    router = _build_qb_router(_k3_config("quantile"))
    hidden = torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda")

    assert router.local_margin_histogram.sum().item() == 0
    router.train()
    router(hidden)
    after_one = int(router.local_margin_histogram.sum().item())
    assert after_one == NUM_TOKENS * NUM_EXPERTS

    router(hidden)
    assert int(router.local_margin_histogram.sum().item()) == 2 * after_one


@_GPU_ONLY
def test_quantile_router_is_frozen_at_inference(moe_parallel_state) -> None:
    """ "The final bias is frozen at inference" (§2.3.3).

    Two things must hold: eval-mode forwards contribute nothing to the
    histogram, and an eval-mode router is not collected for update.
    """
    router = _build_qb_router(_k3_config("quantile"))
    hidden = torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda")

    router.eval()
    with torch.no_grad():
        router(hidden)
    assert router.local_margin_histogram.sum().item() == 0

    holder = torch.nn.Module()
    holder.router = router
    holder.eval()
    assert collect_quantile_balancing_routers([holder]) == []
    holder.train()
    assert collect_quantile_balancing_routers([holder]) == [router]


@_GPU_ONLY
def test_quantile_router_histogram_matches_a_direct_recomputation(moe_parallel_state) -> None:
    """What the router accumulates is the same thing the pure function computes."""
    router = _build_qb_router(_k3_config("quantile"))
    with torch.no_grad():
        router.expert_bias.copy_(torch.linspace(-0.1, 0.1, NUM_EXPERTS, device="cuda"))
    hidden = torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda")

    bias_before = router.expert_bias.clone()
    router.train()
    router(hidden)

    logits = router.gating(hidden).reshape(-1, NUM_EXPERTS)
    scores = torch.sigmoid(logits.float())
    expected, _ = compute_margin_histogram(
        scores,
        bias_before,
        topk=TOPK,
        num_bins=NUM_BINS,
        margin_min=MARGIN_MIN,
        margin_max=MARGIN_MAX,
    )
    assert torch.equal(router.local_margin_histogram, expected)


@_GPU_ONLY
def test_quantile_update_writes_the_bias_and_clears_the_histogram(moe_parallel_state) -> None:
    """The global-batch hook: set the bias, then reset the statistic."""
    config = _k3_config("quantile")
    router = _build_qb_router(config)
    holder = torch.nn.Module()
    holder.router = router
    holder.train()

    hidden = torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda")
    for _ in range(3):  # three microbatches in one global batch
        router(hidden)

    hist = router.local_margin_histogram.clone()
    assert (
        hist.sum().item() == 3 * NUM_TOKENS * NUM_EXPERTS
    ), "nothing accumulated, so the rest of this test would pass vacuously"

    new_bias = update_router_expert_bias_quantile([holder], config)

    assert new_bias is not None and new_bias.shape == (1, NUM_EXPERTS)
    assert float(new_bias.abs().max()) > 0.0, "the update produced an all-zero bias"
    assert router.local_margin_histogram.sum().item() == 0, "histogram not reset after update"

    expected = compute_quantile_bias(
        hist,
        topk=TOPK,
        num_experts=NUM_EXPERTS,
        margin_min=MARGIN_MIN,
        margin_max=MARGIN_MAX,
        center=True,
    )
    assert torch.allclose(router.expert_bias.double(), expected, atol=1e-6)
    assert abs(float(router.expert_bias.mean())) < 1e-6


@_GPU_ONLY
def test_quantile_update_respects_the_ema_option(moe_parallel_state) -> None:
    config = _k3_config("quantile", quantile_balancing_ema_decay=0.9)
    router = _build_qb_router(config)
    holder = torch.nn.Module()
    holder.router = router
    holder.train()

    with torch.no_grad():
        router.expert_bias.fill_(0.5)
    old = router.expert_bias.clone()

    router(torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda"))
    target = update_router_expert_bias_quantile([holder], config)[0]

    expected = 0.9 * old.double() + 0.1 * target
    assert torch.allclose(router.expert_bias.double(), expected, atol=1e-6)


@_GPU_ONLY
def test_quantile_update_is_a_noop_with_no_statistic(moe_parallel_state) -> None:
    """No routers with histograms -> no update, no crash."""
    config = _k3_config("quantile")
    holder = torch.nn.Module()
    holder.linear = torch.nn.Linear(4, 4)
    assert update_router_expert_bias_quantile([holder], config) is None


# ---------------------------------------------------------------------------
# Default-off, and the spec factory
# ---------------------------------------------------------------------------


@_GPU_ONLY
def test_quantile_spec_factory_wires_the_router_only_when_selected(moe_parallel_state) -> None:
    from megatron.core.transformer.spec_utils import build_module

    from primus.backends.megatron.core.transformer.kimi_k3.moe import (
        build_stable_latent_moe_spec,
    )

    for rule, expect_hist in (("sign", False), ("quantile", True)):
        config = _k3_config(
            rule,
            routed_expert_hidden_size=None,
            latent_moe_use_norm=False,
            moe_token_dispatcher_type="alltoall",
            moe_grouped_gemm=False,
            moe_permute_fusion=False,
            moe_shared_expert_intermediate_size=HIDDEN,
        )
        spec = build_stable_latent_moe_spec(config=config)
        moe = build_module(spec, config=config)
        moe.set_layer_number(1)
        moe = moe.cuda()
        has_hist = hasattr(moe.router, "local_margin_histogram")
        assert has_hist is expect_hist, (
            f"rule={rule}: router {type(moe.router).__name__} "
            f"{'has' if has_hist else 'has no'} margin histogram"
        )
        if expect_hist:
            assert isinstance(moe.router, QuantileBalancingMixin)


def test_quantile_balancing_is_off_by_default_in_the_yaml() -> None:
    """Selecting the faithful rule must be an explicit, reviewable diff."""
    from pathlib import Path

    from primus.core.config.yaml_loader import parse_yaml

    repo = Path(__file__).resolve().parents[5]
    parsed = parse_yaml(str(repo / "primus/configs/models/megatron/kimi_k3_base.yaml"))
    assert parsed["moe_router_bias_update_rule"] == "sign"
    assert parsed["quantile_balancing_num_bins"] == NUM_BINS
    assert float(parsed["quantile_balancing_margin_min"]) == MARGIN_MIN
    assert float(parsed["quantile_balancing_margin_max"]) == MARGIN_MAX
    assert parsed["quantile_balancing_center_bias"] is True
    assert parsed["quantile_balancing_ema_decay"] is None


def test_quantile_target_quantile_is_one_minus_k_over_n() -> None:
    """`k=16`, `n=896` -> 0.98214..., the number §2.3.3 implies."""
    assert math.isclose(1.0 - 16 / 896, 0.9821428571428571, rel_tol=0, abs_tol=1e-15)


# ---------------------------------------------------------------------------
# The patch site
# ---------------------------------------------------------------------------


def test_quantile_patch_target_is_the_module_not_the_reexported_function() -> None:
    """``megatron.core.distributed.finalize_model_grads`` is ambiguous.

    ``megatron/core/distributed/__init__.py:10`` does
    ``from .finalize_model_grads import finalize_model_grads``, so the package
    attribute of that name is the **function**, and
    ``from megatron.core.distributed import finalize_model_grads`` silently
    returns a callable with no ``_update_router_expert_bias`` on it. The patch
    goes through ``importlib.import_module`` for exactly this reason; if
    upstream ever stops re-exporting, this test says the workaround can go.
    """
    import importlib

    from megatron.core import distributed

    module = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    assert hasattr(module, "_update_router_expert_bias")
    assert callable(distributed.finalize_model_grads)
    assert not hasattr(
        distributed.finalize_model_grads, "_update_router_expert_bias"
    ), "the package attribute is no longer the shadowing function; simplify the patch"


def test_quantile_patch_condition_is_off_unless_selected() -> None:
    """The patch must be inert for every model that has not opted in."""
    from types import SimpleNamespace

    from primus.backends.megatron.patches.kimi_k3_quantile_balancing_patches import (
        _wants_quantile_balancing,
    )

    def wants(**kw):
        # get_args reads ctx.extra["module_config"].params (context.py:106-110),
        # i.e. the merged pre_trainer namespace.
        ctx = SimpleNamespace(extra={"module_config": SimpleNamespace(params=SimpleNamespace(**kw))})
        return _wants_quantile_balancing(ctx)

    assert not wants(moe_router_enable_expert_bias=False, moe_router_bias_update_rule="quantile")
    assert not wants(moe_router_enable_expert_bias=True, moe_router_bias_update_rule="sign")
    assert not wants(moe_router_enable_expert_bias=True)  # key absent entirely
    assert wants(moe_router_enable_expert_bias=True, moe_router_bias_update_rule="quantile")


@_GPU_ONLY
def test_quantile_patch_rebinds_the_update_and_the_rebound_function_works(
    moe_parallel_state,
) -> None:
    """Apply the patch for real, then drive it through one global batch.

    The experiment harness calls ``update_router_expert_bias_quantile``
    directly, so without this the patch itself — the thing a production run
    actually depends on — would only ever be tested for its *condition*.
    """
    import importlib
    from types import SimpleNamespace

    from primus.backends.megatron.patches.kimi_k3_quantile_balancing_patches import (
        patch_quantile_balancing,
    )

    fmg = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    original = fmg._update_router_expert_bias

    config = _k3_config("quantile")
    router = _build_qb_router(config)
    holder = torch.nn.Module()
    holder.router = router
    holder.train()

    try:
        patch_quantile_balancing(SimpleNamespace(extra={"module_config": SimpleNamespace(params=config)}))
        assert fmg._update_router_expert_bias is not original, "the patch did not rebind"

        router(torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda"))
        assert router.local_margin_histogram.sum().item() > 0

        fmg._update_router_expert_bias([holder], config)

        assert float(router.expert_bias.abs().max()) > 0.0, "the rebound update did nothing"
        assert router.local_margin_histogram.sum().item() == 0
    finally:
        fmg._update_router_expert_bias = original


@_GPU_ONLY
def test_quantile_patch_falls_back_loudly_without_a_histogram(moe_parallel_state) -> None:
    """Selected but not wired -> use the sign rule, do not pretend to work.

    A silent no-op here would look like "Quantile Balancing balances badly"
    rather than "Quantile Balancing never ran".
    """
    import importlib
    from types import SimpleNamespace

    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.router import TopKRouter

    from primus.backends.megatron.patches.kimi_k3_quantile_balancing_patches import (
        patch_quantile_balancing,
    )

    fmg = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    original = fmg._update_router_expert_bias

    config = _k3_config("quantile")
    plain = TopKRouter(config=config, pg_collection=get_default_pg_collection()).cuda()
    holder = torch.nn.Module()
    holder.router = plain
    holder.train()

    try:
        patch_quantile_balancing(SimpleNamespace(extra={"module_config": SimpleNamespace(params=config)}))
        plain(torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda"))
        assert float(plain.local_tokens_per_expert.sum()) > 0
        fmg._update_router_expert_bias([holder], config)
        # The sign rule moves the bias by exactly moe_router_bias_update_rate.
        assert torch.allclose(
            plain.expert_bias.abs(),
            torch.full_like(plain.expert_bias, config.moe_router_bias_update_rate),
        ), "fell through to neither rule"
    finally:
        fmg._update_router_expert_bias = original


def test_quantile_hook_site_constant_still_points_at_the_sign_rule() -> None:
    """``QUANTILE_BALANCING_HOOK_SITE`` was recorded by WP5; keep it honest."""
    import importlib
    import inspect

    from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_stable_latent_moe import (
        QUANTILE_BALANCING_HOOK_SITE,
    )

    path, _, line_range = QUANTILE_BALANCING_HOOK_SITE.rpartition(":")
    assert path.endswith("finalize_model_grads.py")
    module = importlib.import_module("megatron.core.distributed.finalize_model_grads")
    src, start = inspect.getsourcelines(module._update_router_expert_bias)
    lo, hi = (int(x) for x in line_range.split("-"))
    assert start <= lo <= hi <= start + len(src), (
        f"{QUANTILE_BALANCING_HOOK_SITE} no longer lands inside "
        f"_update_router_expert_bias (lines {start}..{start + len(src)})"
    )
    assert "get_updated_expert_bias" in "".join(src)

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The fused FlyDSL Quantile Balancing histogram, against the eager oracle.

The acceptance bar here is different from every other kernel in this tree: the
statistic is a **count**, so the only tolerance that means anything is zero.
``torch.equal`` on the whole ``[num_experts, num_bins]`` table, and on the
``(below, above)`` saturation pair, or the test fails.

That bar is achievable only because of a deliberate design decision recorded in
the kernel: the bin index is a ``floor`` of a **true IEEE division**, not of a
reciprocal multiply. ``floor`` is discontinuous, so a last-bit difference moves a
count into the neighbouring bin. ``test_injected_defect_is_caught`` includes
``reciprocal_multiply`` precisely so that claim is tested rather than asserted —
it is the optimisation a reviewer would suggest, and this is the evidence for
turning it down.

There is no backward. ``QuantileBalancingMixin._accumulate_margin_histogram``
runs under ``@torch.no_grad()`` and the report freezes the bias at inference, so
the statistic has no adjoint by construction rather than by omission.
"""

from __future__ import annotations

import os

import pytest
import torch  # noqa: F401  # must precede any transformer_engine import

for _var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
    os.environ.pop(_var, None)

flydsl = pytest.importorskip(
    "flydsl", reason="the FlyDSL Quantile Balancing backend needs the flydsl package"
)

from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels import (  # noqa: E402
    QB_BACKENDS,
    compute_margin_histogram,
    resolve_qb_backend,
)


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    return arch.startswith("gfx950")


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the FlyDSL Quantile Balancing kernel is built for gfx950 (CDNA4)"
)

BINNING = dict(num_bins=1024, margin_min=-1.0, margin_max=1.0)


def _make_inputs(num_tokens, num_experts, seed=0, spread=1.0, device="cuda"):
    """Router scores and an expert bias, shaped like a real microbatch.

    ``scores`` is a sigmoid, matching ``topk_routing_with_score_function``'s
    sigmoid branch (``moe_utils.py:773``) and ``KimiMoEGate``
    (``modeling_kimi_linear.py:712``). ``spread`` widens the logits, which pushes
    the sigmoid towards its rails and manufactures exact ties — the case where a
    top-(k+1) selection is least well-determined.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        num_tokens, num_experts, generator=gen, device=device, dtype=torch.float32
    )
    scores = torch.sigmoid(logits * spread)
    expert_bias = (
        torch.randn(num_experts, generator=gen, device=device, dtype=torch.float32) * 0.05
    )
    return scores, expert_bias


def _assert_identical(got, ref, what=""):
    hist_got, clamped_got = got
    hist_ref, clamped_ref = ref
    assert hist_got.dtype == hist_ref.dtype == torch.int64
    assert hist_got.shape == hist_ref.shape
    mismatched = int((hist_got != hist_ref).sum())
    assert mismatched == 0, (
        f"{what}: {mismatched} of {hist_ref.numel()} bins differ; "
        f"totals {int(hist_got.sum())} vs {int(hist_ref.sum())}"
    )
    assert torch.equal(clamped_got, clamped_ref), (
        f"{what}: saturation counters {clamped_got.tolist()} vs {clamped_ref.tolist()}"
    )


# ---------------------------------------------------------------------------
# parity -- bit-identical or nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [
        (1, 32, 4),  # one token
        (17, 8, 2),  # ragged: 17*8 = 136 is not a multiple of the 256-thread block
        (256, 8, 2),  # the debug config's expert pool
        (4096, 32, 4),  # the scaled config, per microbatch
        (4096, 896, 16),  # the 93-layer release's expert pool
        (2048, 33, 5),  # a prime-ish expert count, so E does not divide the block
    ],
)
def test_matches_eager_bit_for_bit(num_tokens, num_experts, topk):
    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(num_tokens, num_experts, seed=num_experts)
    kw = dict(topk=topk, **BINNING)
    _assert_identical(
        backend(scores, expert_bias, **kw),
        compute_margin_histogram(scores, expert_bias, **kw),
        what=f"N={num_tokens} E={num_experts}",
    )


@pytest.mark.parametrize("num_bins", [2, 64, 256, 4096])
def test_matches_eager_across_bin_counts(num_bins):
    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(1024, 32, seed=num_bins)
    kw = dict(topk=4, num_bins=num_bins, margin_min=-1.0, margin_max=1.0)
    _assert_identical(
        backend(scores, expert_bias, **kw),
        compute_margin_histogram(scores, expert_bias, **kw),
        what=f"B={num_bins}",
    )


def test_saturating_range_counts_both_tails_exactly():
    """A deliberately narrow range, so ``(below, above)`` are large and checkable.

    The production default is ``+-1.0``, which the A/B measured as wide enough
    for real data (zero clamped margins at every step of every quantile arm), so
    the counters would otherwise never be exercised at all.
    """
    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(4096, 32, seed=7)
    kw = dict(topk=4, num_bins=1024, margin_min=-0.01, margin_max=0.01)
    ref = compute_margin_histogram(scores, expert_bias, **kw)
    assert int(ref[1][0]) > 0 and int(ref[1][1]) > 0, "fixture does not saturate both tails"
    _assert_identical(backend(scores, expert_bias, **kw), ref, what="saturating")


def test_matches_eager_when_scores_are_tied():
    """Wide logits push the sigmoid to its rails, manufacturing exact ties.

    Ties matter because the cutoff is a top-(k+1) selection: with ties the
    *identity* of the selected element is not unique, only its value. Both paths
    call the same ``torch.topk``, so this pins that the kernel consumes the
    library's answer rather than re-deriving one.
    """
    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(4096, 32, seed=11, spread=25.0)
    kw = dict(topk=4, num_bins=256, margin_min=-1.0, margin_max=1.0)
    _assert_identical(
        backend(scores, expert_bias, **kw),
        compute_margin_histogram(scores, expert_bias, **kw),
        what="tied",
    )


def test_topk_larger_than_the_expert_pool_is_clamped():
    """``k + 1 > num_experts`` must take the eager path's ``min`` clamp, not crash."""
    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(64, 4, seed=3)
    kw = dict(topk=8, **BINNING)
    _assert_identical(
        backend(scores, expert_bias, **kw),
        compute_margin_histogram(scores, expert_bias, **kw),
        what="k>=E",
    )


def test_zero_tokens_returns_zeros_without_launching():
    backend = resolve_qb_backend("flydsl")
    scores = torch.zeros(0, 32, device="cuda")
    expert_bias = torch.zeros(32, device="cuda")
    hist, clamped = backend(scores, expert_bias, topk=4, **BINNING)
    assert hist.shape == (32, 1024) and int(hist.sum()) == 0
    assert clamped.tolist() == [0, 0]


def test_every_token_is_counted_exactly_once():
    """A property the histogram must have and that a bin-by-bin diff can miss."""
    backend = resolve_qb_backend("flydsl")
    num_tokens, num_experts = 4096, 32
    scores, expert_bias = _make_inputs(num_tokens, num_experts, seed=5)
    hist, _ = backend(scores, expert_bias, topk=4, **BINNING)
    assert int(hist.sum()) == num_tokens * num_experts
    # and per expert, since the atomics are per-expert rows
    assert torch.equal(
        hist.sum(dim=1),
        torch.full((num_experts,), num_tokens, dtype=torch.int64, device=hist.device),
    )


# ---------------------------------------------------------------------------
# bug injection
# ---------------------------------------------------------------------------


def _injection_names():
    from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1 import (
        INJECTIONS,
    )

    return INJECTIONS


@pytest.fixture
def clean_kernel():
    from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1 import (
        inject_defect,
    )

    yield inject_defect
    inject_defect()


@pytest.mark.parametrize("defect", _injection_names())
def test_injected_defect_is_caught(defect, clean_kernel):
    """Every named defect must fail the exact comparison that passes clean.

    ``trunc_not_floor`` is the interesting one: it is the bug you get from C
    habits, it agrees with ``floor`` for every positive value, and most margins
    are *negative* — a margin is a score minus the ``(k+1)``-th largest — which
    is exactly where the two part company.
    """
    backend = resolve_qb_backend("flydsl")
    # A narrow range so the saturation counters are live too, otherwise
    # `no_clamp_count` has nothing to get wrong.
    kw = dict(topk=4, num_bins=512, margin_min=-0.02, margin_max=0.02)
    scores, expert_bias = _make_inputs(4096, 32, seed=13)
    ref = compute_margin_histogram(scores, expert_bias, **kw)
    assert int(ref[1].sum()) > 0, "fixture must saturate for no_clamp_count to be a defect"

    _assert_identical(backend(scores, expert_bias, **kw), ref, what="clean")  # sanity

    clean_kernel(defect)
    with pytest.raises(AssertionError):
        _assert_identical(backend(scores, expert_bias, **kw), ref, what=defect)


def test_torch_scalar_division_is_itself_a_reciprocal_multiply():
    """The premise the kernel's scaling rests on, checked directly.

    ``compute_margin_histogram`` divides an fp32 tensor by a Python float, and
    PyTorch compiles that to a multiply by the reciprocal. Since the bin index is
    a ``floor``, the kernel has to do the same thing to match — being *more*
    accurate than the oracle is a defect here, which is the opposite of the usual
    rule and is the reason this test exists rather than a comment.
    """
    x = torch.randn(1 << 20, device="cuda")
    lo, width = -1.0, 2.0 / 1000  # deliberately NOT a power of two
    scalar_div = ((x - lo) / width).floor()
    scalar_recip = ((x - lo) * (1.0 / width)).floor()
    tensor_div = ((x - lo) / torch.tensor([width], device="cuda")).floor()

    assert torch.equal(scalar_div, scalar_recip), (
        "torch's tensor-by-scalar division is expected to be a reciprocal "
        "multiply; if this ever changes, the kernel's scaling must change with it"
    )
    assert not torch.equal(scalar_div, tensor_div), (
        "dividing by a one-element tensor is a true division and must differ, "
        "otherwise this fixture proves nothing"
    )


def test_scaling_choice_is_invisible_at_the_shipped_binning(clean_kernel):
    """At ``±1.0`` over a power-of-two bin count the choice cannot matter.

    ``width = 2/1024 = 2**-9`` is an exact power of two, so its reciprocal is
    exact and every spelling agrees bit-for-bit. Worth pinning so the reciprocal
    multiply is not credited with fixing the shipped default — it is not; it fixes
    the arbitrary binning the config permits.
    """
    import math

    width = (1.0 - -1.0) / 1024
    assert math.frexp(width)[0] == 0.5, "the shipped width must be a power of two"

    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(4096, 32, seed=13)
    kw = dict(topk=4, **BINNING)
    ref = compute_margin_histogram(scores, expert_bias, **kw)

    clean_kernel("true_division")
    _assert_identical(backend(scores, expert_bias, **kw), ref, what="truediv@pow2")


def test_a_true_division_diverges_at_a_non_power_of_two_width(clean_kernel):
    """...and at 1000 bins it moves counts, which is what makes this a decision.

    Measured: 4 of 32 000 bins. Small, and entirely sufficient — a histogram
    parity test with a nonzero tolerance would not be a parity test.
    """
    import math

    kw = dict(topk=4, num_bins=1000, margin_min=-1.0, margin_max=1.0)
    width = (kw["margin_max"] - kw["margin_min"]) / kw["num_bins"]
    assert math.frexp(width)[0] != 0.5, "this width must NOT be a power of two"

    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(4096, 32, seed=13)
    ref = compute_margin_histogram(scores, expert_bias, **kw)
    _assert_identical(backend(scores, expert_bias, **kw), ref, what="clean")  # sanity

    clean_kernel("true_division")
    with pytest.raises(AssertionError):
        _assert_identical(backend(scores, expert_bias, **kw), ref, what="truediv@1000")


def test_an_unknown_injection_name_raises():
    from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1.qb_margin_histogram_kernel import (
        build_qb_margin_histogram,
    )

    with pytest.raises(ValueError, match="unknown injection"):
        build_qb_margin_histogram(
            num_experts=8, num_bins=64, margin_min=-1.0, margin_max=1.0, inject="floor_not_trunc"
        )


# ---------------------------------------------------------------------------
# the shape guard: the kernel is SLOWER above ~8k tokens and must not be picked
# ---------------------------------------------------------------------------


def test_kernel_beats_eager_predicate():
    """The guard's decision function, including the disable escape hatch."""
    from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1 import (
        KERNEL_MAX_TOKENS,
        kernel_beats_eager,
    )

    assert kernel_beats_eager(4096, 32) is True
    assert kernel_beats_eager(KERNEL_MAX_TOKENS, 32) is True
    assert kernel_beats_eager(KERNEL_MAX_TOKENS + 1, 32) is False
    assert kernel_beats_eager(32768, 32) is False
    # num_experts does not enter the decision: contention is counts-per-bin,
    # which is num_tokens/num_bins, and each expert's row gets exactly
    # num_tokens increments regardless of how many experts there are.
    assert kernel_beats_eager(32768, 896) is False
    assert kernel_beats_eager(4096, 896) is True
    # 0 disables the guard, for re-measuring or for forcing the kernel
    assert kernel_beats_eager(32768, 32, max_tokens=0) is True


def test_above_the_guard_the_kernel_is_not_launched_at_all():
    """Structural, not statistical: break the launcher and see who notices.

    Above the threshold the entry must run the eager path, so a kernel that
    cannot even be built must not matter. Below it, the same broken launcher must
    surface — otherwise this test would pass on an entry that never launches
    anything.
    """
    import primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1.histogram as H

    scores_small, bias_small = _make_inputs(64, 8, seed=1)
    scores_big, bias_big = _make_inputs(9000, 8, seed=1)
    kw = dict(topk=2, **BINNING)

    original = H._get_kernel

    def boom(*a, **k):
        raise AssertionError("the kernel launcher must not be reached here")

    H._get_kernel = boom
    try:
        # above the guard: eager path, so the broken launcher is never touched
        hist, clamped = H.flydsl_compute_margin_histogram(
            scores_big, bias_big, max_tokens=8192, **kw
        )
        _assert_identical(
            (hist, clamped), compute_margin_histogram(scores_big, bias_big, **kw), what="fallback"
        )
        # below the guard: it must be touched, or the guard is doing nothing
        with pytest.raises(AssertionError, match="must not be reached"):
            H.flydsl_compute_margin_histogram(scores_small, bias_small, max_tokens=8192, **kw)
    finally:
        H._get_kernel = original


def test_the_fallback_warns_once_and_says_why(caplog):
    """A silent fallback is a fallback nobody knows happened."""
    import logging

    import primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1.histogram as H

    H._WARNED.clear()
    scores, bias = _make_inputs(9000, 8, seed=2)
    kw = dict(topk=2, **BINNING)
    with caplog.at_level(logging.WARNING, logger=H.logger.name):
        H.flydsl_compute_margin_histogram(scores, bias, max_tokens=8192, **kw)
        H.flydsl_compute_margin_histogram(scores, bias, max_tokens=8192, **kw)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, "one warning per process, not one per microbatch per layer"
    msg = warnings[0].getMessage()
    assert "0.61x" in msg and "EAGER" in msg, msg
    H._WARNED.clear()


def test_forcing_the_kernel_above_the_guard_still_matches():
    """`max_tokens=0` runs the kernel in the losing regime; it must still be exact.

    Slower is a performance property. Wrong would be a correctness property, and
    the guard must not be load-bearing for correctness.
    """
    backend = resolve_qb_backend("flydsl", max_tokens=0)
    scores, bias = _make_inputs(16384, 32, seed=4)
    kw = dict(topk=4, **BINNING)
    _assert_identical(
        backend(scores, bias, **kw),
        compute_margin_histogram(scores, bias, **kw),
        what="forced above guard",
    )


def test_resolve_binds_the_guard_and_eager_ignores_it():
    from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels import (
        resolve_qb_backend as resolve,
    )

    # eager takes no guard and must be handed back untouched, not wrapped
    assert resolve("eager", max_tokens=123) is compute_margin_histogram
    bound = resolve("flydsl", max_tokens=123)
    assert getattr(bound, "keywords", {}).get("max_tokens") == 123
    # None leaves the module default in place rather than binding anything
    assert not hasattr(resolve("flydsl"), "keywords")


# ---------------------------------------------------------------------------
# refusals and dispatch
# ---------------------------------------------------------------------------


def test_rejects_non_fp32_scores():
    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(64, 8, seed=1)
    with pytest.raises(ValueError, match="quantile_balancing_backend: eager"):
        backend(scores.bfloat16(), expert_bias, topk=2, **BINNING)


def test_rejects_a_cpu_tensor():
    backend = resolve_qb_backend("flydsl")
    scores, expert_bias = _make_inputs(64, 8, seed=1)
    with pytest.raises(ValueError, match="quantile_balancing_backend: eager"):
        backend(scores.cpu(), expert_bias.cpu(), topk=2, **BINNING)


def test_backend_registry_and_unknown_name():
    assert set(QB_BACKENDS) == {"eager", "flydsl"}
    assert resolve_qb_backend("eager") is compute_margin_histogram
    with pytest.raises(ValueError, match="Unknown Quantile Balancing backend"):
        resolve_qb_backend("triton_v99")


def test_the_eager_entry_is_still_importable_from_its_old_home():
    """``compute_margin_histogram`` moved into the backend tree; the name did not.

    ``test_quantile_balancing.py`` and the patch module both import it from
    ``k3_quantile_balancing``, so this pins that the re-export is the same object
    and not a copy that could drift from the oracle.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.moe import k3_quantile_balancing

    assert k3_quantile_balancing.compute_margin_histogram is compute_margin_histogram


@pytest.fixture()
def moe_parallel_state():
    """TP=PP=EP=CP=1. Same bring-up as ``test_quantile_balancing.py``.

    ``TopKRouter.__init__`` allocates its expert-bias buffers on
    ``torch.cuda.current_device()`` and needs Megatron's parallel state, so a
    router cannot be constructed without this.
    """
    import socket

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


def test_router_selects_the_kernel_from_the_config(moe_parallel_state):
    """``quantile_balancing_backend: flydsl`` must reach the router's statistic."""
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.router import TopKRouter

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.moe.k3_quantile_balancing import (
        make_quantile_balancing_router,
    )

    def _config(backend, **over):
        return KimiK3TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=8,
            kv_channels=64,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rule="quantile",
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1e-3,
            params_dtype=torch.float32,
            activation_func=torch.nn.functional.silu,
            init_method_std=0.02,
            quantile_balancing_backend=backend,
            **over,
        )

    cls = make_quantile_balancing_router(TopKRouter)
    pg = get_default_pg_collection()
    eager_router = cls(config=_config("eager"), pg_collection=pg).cuda()
    fly_router = cls(config=_config("flydsl"), pg_collection=pg).cuda()
    assert eager_router.qb_backend_name == "eager"
    assert eager_router.qb_histogram is compute_margin_histogram
    assert fly_router.qb_backend_name == "flydsl"
    assert fly_router.qb_histogram is not compute_margin_histogram

    # and the shape guard is bound from the config, not left to the module default
    guarded = cls(
        config=_config("flydsl", quantile_balancing_kernel_max_tokens=4096), pg_collection=pg
    ).cuda()
    assert getattr(guarded.qb_histogram, "keywords", {}).get("max_tokens") == 4096

    # A live forward, swapping the backend on ONE router rather than comparing
    # two. Two routers differ in more than their backend -- each draws its own
    # gating weight, and copying it across still leaves the aux-loss tracker and
    # any other per-instance state to diverge. One router with the same weight,
    # the same bias and the same input isolates the backend and nothing else.
    fly_router.train()
    hidden = torch.randn(64, 1, 64, device="cuda")

    fly_router(hidden)
    fly_hist = fly_router.local_margin_histogram.clone()
    assert int(fly_hist.sum()) == 64 * 8, "the kernel must count every (token, expert) pair"

    fly_router.local_margin_histogram.zero_()
    fly_router.qb_histogram = compute_margin_histogram
    fly_router(hidden)
    eager_hist = fly_router.local_margin_histogram.clone()

    assert torch.equal(fly_hist, eager_hist), (
        "the same router with the same inputs must accumulate the identical "
        "histogram under either backend"
    )

    with pytest.raises(ValueError, match="quantile_balancing_backend must be one of"):
        cls(config=_config("triton_v99"), pg_collection=pg)


def test_importing_qb_kernels_does_not_import_flydsl():
    """Checked in a subprocess; this process has already imported flydsl."""
    import subprocess
    import sys

    code = (
        "import sys, torch; "
        "import primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels as m; "
        "assert 'flydsl' not in sys.modules, sorted(k for k in sys.modules if 'flydsl' in k); "
        "print('ok')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ok" in out.stdout

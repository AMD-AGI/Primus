###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-6 P39 G42 — V4 router post-logits Triton parity.

Asserts that :class:`V4RouterPostFn` (Triton kernel from
``primus.backends.megatron.core.transformer.moe._triton.v4_router_post``)
produces ``(probs, routing_map)`` bit-equal to the eager body of
:func:`primus.backends.megatron.core.transformer.moe.v4_topk_router._compute_route`
across the 3 score functions × {with, without bias} × {hash router,
learned router}.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "v4_router_post Triton kernel requires CUDA / HIP",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.moe._triton.v4_router_post import (  # noqa: E402
    V4RouterPostFn,
    is_triton_path_enabled,
)
from primus.backends.megatron.core.transformer.moe.v4_hash_router import (  # noqa: E402
    DeepseekV4HashRouter,
)
from primus.backends.megatron.core.transformer.moe.v4_topk_router import (  # noqa: E402
    DeepseekV4LearnedRouter,
)


@contextmanager
def _env(key: str, value: str | None):
    prev = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _eager_v4_score(logits: torch.Tensor, *, score_function: str):
    if score_function == "softmax":
        return torch.softmax(logits, dim=-1)
    if score_function == "sigmoid":
        return torch.sigmoid(logits)
    if score_function == "sqrtsoftplus":
        return torch.sqrt(torch.nn.functional.softplus(logits))
    raise ValueError(score_function)


def _eager_post(
    logits: torch.Tensor,
    indices: torch.Tensor,
    *,
    score_function: str,
    topk_scaling_factor: float,
    out_dtype: torch.dtype,
):
    scores = _eager_v4_score(logits.float(), score_function=score_function)
    weights = scores.gather(1, indices)
    if score_function != "softmax":
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-12)
        weights = weights / denom
    if topk_scaling_factor != 1.0:
        weights = weights * float(topk_scaling_factor)
    N, E = logits.shape
    probs = torch.zeros(N, E, dtype=out_dtype, device=logits.device)
    probs.scatter_(1, indices, weights.to(out_dtype))
    rmap = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
    rmap.scatter_(1, indices, True)
    return probs, rmap


def _build_inputs(*, N: int, E: int, K: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((N, E), dtype=torch.float32, device="cuda", generator=gen)
    indices = torch.stack(
        [torch.randperm(E, generator=gen, device="cuda")[:K] for _ in range(N)],
        dim=0,
    ).to(torch.int64)
    return logits, indices


# ---------------------------------------------------------------------------
# G42: FWD parity vs eager (3 score fns × {fp32, bf16} × small / release)
# ---------------------------------------------------------------------------


class TestG42ForwardParity:
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
    @pytest.mark.parametrize("scale", [1.0, 2.5])
    @pytest.mark.parametrize("out_dtype", [torch.float32])
    def test_fast_tier_fwd_eager_parity(self, score_function, scale, out_dtype):
        N, E, K = 64, 32, 4
        logits, indices = _build_inputs(N=N, E=E, K=K, seed=2000)
        probs_t, rmap_t = V4RouterPostFn.apply(logits, indices, score_function, scale, out_dtype)
        probs_e, rmap_e = _eager_post(
            logits, indices, score_function=score_function, topk_scaling_factor=scale, out_dtype=out_dtype
        )
        torch.testing.assert_close(probs_t, probs_e, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rmap_t, rmap_e)
        torch.testing.assert_close(probs_t.nonzero(), probs_e.nonzero(), check_dtype=False)

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
    @pytest.mark.parametrize("shape", [(64, 256, 6), (64, 7, 3), (64, 100, 5)])
    def test_non_pow2_e_k_fwd_parity(self, score_function, shape):
        """Arbitrary (non-power-of-2) E / K — the production V4-Flash router
        runs E=256, K=6, so K=6 is the load-bearing case.  Also cover a
        non-power-of-2 E (7, 100)."""
        N, E, K = shape
        logits, indices = _build_inputs(N=N, E=E, K=K, seed=2500)
        probs_t, rmap_t = V4RouterPostFn.apply(logits, indices, score_function, 2.5, torch.float32)
        probs_e, rmap_e = _eager_post(
            logits, indices, score_function=score_function, topk_scaling_factor=2.5, out_dtype=torch.float32
        )
        torch.testing.assert_close(probs_t, probs_e, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rmap_t, rmap_e)

    @pytest.mark.slow
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
    def test_release_tier_fwd_eager_parity(self, score_function):
        # V4-Flash production widths: E=256, K=6 (non-power-of-2 topk).
        N, E, K = 4096, 256, 6
        logits, indices = _build_inputs(N=N, E=E, K=K, seed=8000)
        probs_t, rmap_t = V4RouterPostFn.apply(logits, indices, score_function, 2.5, torch.float32)
        probs_e, rmap_e = _eager_post(
            logits, indices, score_function=score_function, topk_scaling_factor=2.5, out_dtype=torch.float32
        )
        torch.testing.assert_close(probs_t, probs_e, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rmap_t, rmap_e)


# ---------------------------------------------------------------------------
# G42: BWD parity
# ---------------------------------------------------------------------------


class TestG42BackwardParity:
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
    @pytest.mark.parametrize("scale", [1.0, 2.5])
    @pytest.mark.parametrize("shape", [(64, 32, 4), (64, 256, 6), (64, 100, 5)])
    def test_fast_tier_bwd_eager_parity(self, score_function, scale, shape):
        N, E, K = shape
        logits_e, indices = _build_inputs(N=N, E=E, K=K, seed=3100)
        logits_e = logits_e.requires_grad_(True)
        logits_t = logits_e.detach().clone().requires_grad_(True)

        probs_t, _ = V4RouterPostFn.apply(logits_t, indices, score_function, scale, torch.float32)
        probs_e, _ = _eager_post(
            logits_e,
            indices,
            score_function=score_function,
            topk_scaling_factor=scale,
            out_dtype=torch.float32,
        )

        g = torch.randn_like(probs_t)
        (probs_t * g).sum().backward()
        (probs_e * g).sum().backward()

        torch.testing.assert_close(logits_t.grad, logits_e.grad, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# G42: composed routers
# ---------------------------------------------------------------------------


class TestG42ComposedRouters:
    """Both routers (`DeepseekV4LearnedRouter` and `DeepseekV4HashRouter`)
    must produce bit-equal (probs, routing_map) between the
    PRIMUS_V4_ROUTER_TRITON=1 and =0 paths.  This is the load-bearing
    integration test for downstream MoEDispatch.
    """

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
    def test_learned_router_env_toggle_parity(self, score_function):
        torch.manual_seed(123)
        N, D, E, K = 64, 32, 32, 4
        router = DeepseekV4LearnedRouter(
            hidden_size=D, num_experts=E, topk=K, score_function=score_function
        ).to("cuda")
        hidden = torch.randn((1, N, D), dtype=torch.float32, device="cuda")

        with _env("PRIMUS_V4_ROUTER_TRITON", "1"):
            assert is_triton_path_enabled()
            probs_t, rmap_t = router(hidden)
        with _env("PRIMUS_V4_ROUTER_TRITON", "0"):
            assert not is_triton_path_enabled()
            probs_e, rmap_e = router(hidden)

        torch.testing.assert_close(probs_t, probs_e, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(rmap_t, rmap_e)

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
    def test_hash_router_env_toggle_parity(self, score_function):
        torch.manual_seed(124)
        N, D, E, K = 64, 32, 32, 4
        V = 1000
        router = DeepseekV4HashRouter(
            hidden_size=D,
            num_experts=E,
            topk=K,
            vocab_size=V,
            score_function=score_function,
        ).to("cuda")
        hidden = torch.randn((1, N, D), dtype=torch.float32, device="cuda")
        token_ids = torch.randint(0, V, (1, N), dtype=torch.long, device="cuda")

        with _env("PRIMUS_V4_ROUTER_TRITON", "1"):
            probs_t, rmap_t = router(hidden, token_ids)
        with _env("PRIMUS_V4_ROUTER_TRITON", "0"):
            probs_e, rmap_e = router(hidden, token_ids)

        torch.testing.assert_close(probs_t, probs_e, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(rmap_t, rmap_e)


# ---------------------------------------------------------------------------
# G42: edge cases
# ---------------------------------------------------------------------------


class TestG42EdgeCases:
    def test_unknown_score_function_raises(self):
        logits = torch.randn((4, 8), dtype=torch.float32, device="cuda")
        indices = torch.zeros((4, 2), dtype=torch.int64, device="cuda")
        with pytest.raises(ValueError, match="score_function"):
            V4RouterPostFn.apply(logits, indices, "tanh", 1.0, torch.float32)

    def test_cpu_tensor_asserts(self):
        # v4_router_post_triton asserts CUDA tensors (no support predicate).
        from primus.backends.megatron.core.transformer.moe._triton.v4_router_post import (
            v4_router_post_triton,
        )

        cpu_logits = torch.randn((4, 8), dtype=torch.float32)
        cpu_idx = torch.zeros((4, 2), dtype=torch.int64)
        with pytest.raises(AssertionError, match="CUDA"):
            v4_router_post_triton(
                cpu_logits,
                cpu_idx,
                score_function="softmax",
                topk_scaling_factor=1.0,
                out_dtype=torch.float32,
            )

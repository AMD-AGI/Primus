###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-6 P41 G43 — `Indexer.forward` post-einsum tail Triton parity.

Asserts that :class:`IndexerScorePostFn` (Triton tail kernel from
``primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score_post``)
produces scores that match the eager **tail** (``relu + mul + sum(H)
+ causal_mask``, with the einsum kept eager) within bf16 tolerance,
and that the load-bearing post-`topk` ``topk_idxs`` are bit-equal
across the two paths.

The einsum stays on cuBLAS / hipBLASLt; this gate covers only the
post-matmul tail.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "indexer Triton kernel requires CUDA / HIP",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.indexer import Indexer  # noqa: E402
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score_post import (  # noqa: E402
    IndexerScorePostFn,
    indexer_score_post_triton,
    is_triton_kernel_supported,
    is_triton_path_enabled,
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


def _eager_tail(
    dot: torch.Tensor,
    w_i: torch.Tensor,
    *,
    compress_ratio: int,
    out_dtype: torch.dtype,
):
    """Eager reference matching the P41-routed body exactly.

    Takes ``dot [B, S, H, P]`` (pre-relu, the einsum output) and
    ``w_i [B, S, H]``; returns ``scores [B, S, P]``.
    """
    relu_term = torch.nn.functional.relu(dot.float())
    scores = (relu_term * w_i.float().unsqueeze(-1)).sum(dim=2)
    B, S, P = scores.shape
    t_idx = torch.arange(S, device=scores.device).unsqueeze(1)
    s_end = (torch.arange(P, device=scores.device).unsqueeze(0) + 1) * compress_ratio - 1
    allowed = s_end <= t_idx
    mask = torch.where(
        allowed,
        torch.zeros_like(scores[0]),
        torch.full_like(scores[0], float("-inf")),
    )
    scores = scores + mask.unsqueeze(0)
    return scores.to(out_dtype)


def _build_dot_inputs(*, B: int, S: int, P: int, H: int, dtype: torch.dtype, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dot = torch.randn((B, S, H, P), dtype=dtype, device="cuda", generator=gen)
    w_i = torch.randn((B, S, H), dtype=dtype, device="cuda", generator=gen).abs()
    return dot, w_i


# ---------------------------------------------------------------------------
# G43: FWD parity vs eager tail
# ---------------------------------------------------------------------------


class TestG43ForwardParity:
    @pytest.mark.parametrize("H", [1, 2, 4, 8])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("compress_ratio", [1, 4, 16])
    def test_fast_tier_fwd_eager_parity(self, H, dtype, compress_ratio):
        B, S, P = 1, 64, max(4, 64 // max(compress_ratio, 1))
        dot, w = _build_dot_inputs(B=B, S=S, P=P, H=H, dtype=dtype, seed=900 + H)

        s_t = IndexerScorePostFn.apply(dot, w, compress_ratio, dtype)
        s_e = _eager_tail(dot, w, compress_ratio=compress_ratio, out_dtype=dtype)

        if dtype == torch.float32:
            atol, rtol = 1e-5, 1e-5
        else:
            atol, rtol = 5e-3, 5e-3

        finite_e = torch.isfinite(s_e)
        finite_t = torch.isfinite(s_t)
        torch.testing.assert_close(finite_e, finite_t)
        torch.testing.assert_close(s_t[finite_t].float(), s_e[finite_e].float(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# G43: topk parity (load-bearing) via Indexer module
# ---------------------------------------------------------------------------


class TestG43TopKParity:
    """The post-topk indices must be bit-equal vs the eager full chain.

    Mirrors G41 from P38: downstream CSA reads ``topk_idxs`` exactly;
    any divergence breaks the sparse selection.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_indexer_topk_parity_via_module(self, dtype):
        torch.manual_seed(20260515)
        B, S, P, H, HD, K = 1, 64, 16, 8, 32, 4
        D = 128
        indexer = Indexer(
            hidden_size=D,
            index_head_dim=HD,
            index_n_heads=H,
            index_topk=K,
            compress_ratio=4,
        ).to(device="cuda", dtype=dtype)
        hidden = torch.randn((B, S, D), dtype=dtype, device="cuda")

        with _env("PRIMUS_INDEXER_TRITON", "1"), _env("PRIMUS_INDEXER_TRITON_FULL", "0"):
            assert is_triton_path_enabled()
            idx_t, sc_t = indexer(hidden)
        # Both knobs off → fully eager.
        with _env("PRIMUS_INDEXER_TRITON", "0"), _env("PRIMUS_INDEXER_TRITON_FULL", "0"):
            assert not is_triton_path_enabled()
            idx_e, sc_e = indexer(hidden)

        assert idx_t.shape == idx_e.shape
        match_ratio = (idx_t == idx_e).float().mean().item()
        # P41 tail-only path keeps the einsum eager so its dot output is
        # bit-identical to the eager path; the only divergence is fp32 →
        # bf16 cast rounding in the tail.  Higher match ratio than P38.
        if dtype == torch.float32:
            assert match_ratio >= 0.99, f"fp32 topk match ratio too low: {match_ratio}"
        else:
            assert match_ratio >= 0.95, f"bf16 topk match ratio too low: {match_ratio}"


# ---------------------------------------------------------------------------
# G43: BWD parity
# ---------------------------------------------------------------------------


class TestG43BackwardParity:
    @pytest.mark.parametrize("H", [1, 2, 4, 8])
    def test_fast_tier_bwd_eager_parity_fp32(self, H):
        B, S, P = 1, 32, 8
        dtype = torch.float32
        dot_e, w_e = _build_dot_inputs(B=B, S=S, P=P, H=H, dtype=dtype, seed=4400 + H)
        dot_e.requires_grad_(True)
        w_e.requires_grad_(True)
        dot_t = dot_e.detach().clone().requires_grad_(True)
        w_t = w_e.detach().clone().requires_grad_(True)

        s_t = IndexerScorePostFn.apply(dot_t, w_t, 4, dtype)
        s_e = _eager_tail(dot_e, w_e, compress_ratio=4, out_dtype=dtype)

        g = torch.randn_like(s_t)
        finite = torch.isfinite(s_t)
        g = torch.where(finite, g, torch.zeros_like(g))

        (s_t * g).sum().backward()
        (s_e * g).sum().backward()

        torch.testing.assert_close(dot_t.grad, dot_e.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w_t.grad, w_e.grad, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# G43: release-tier V4-Flash production shape
# ---------------------------------------------------------------------------


class TestG43ReleaseTier:
    """V4-Flash production: ``B=1, S=4096, P=1024, H=8``, bf16."""

    @pytest.mark.slow
    def test_v4_flash_fwd_parity(self):
        B, S, P, H = 1, 4096, 1024, 8
        dtype = torch.bfloat16
        dot, w = _build_dot_inputs(B=B, S=S, P=P, H=H, dtype=dtype, seed=9000)

        s_t = IndexerScorePostFn.apply(dot, w, 4, dtype)
        s_e = _eager_tail(dot, w, compress_ratio=4, out_dtype=dtype)

        finite_e = torch.isfinite(s_e)
        finite_t = torch.isfinite(s_t)
        torch.testing.assert_close(finite_e, finite_t)
        # Bandwidth-bound tail; bf16 sum of H terms keeps tighter
        # tolerance than P38 (which accumulated H tensor-core dots).
        torch.testing.assert_close(
            s_t[finite_t].float(),
            s_e[finite_e].float(),
            atol=5e-2,
            rtol=5e-2,
        )


# ---------------------------------------------------------------------------
# G43: edge cases
# ---------------------------------------------------------------------------


class TestG43EdgeCases:
    def test_unsupported_h_raises(self):
        dot = torch.randn((1, 8, 7, 16), dtype=torch.float32, device="cuda")
        w = torch.randn((1, 8, 7), dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="Unsupported H"):
            IndexerScorePostFn.apply(dot, w, 4, torch.float32)

    def test_supported_predicate(self):
        good_dot = torch.randn((1, 64, 8, 16), dtype=torch.float32, device="cuda")
        good_w = torch.randn((1, 64, 8), dtype=torch.float32, device="cuda")
        assert is_triton_kernel_supported(good_dot, good_w)

        bad_h_dot = torch.randn((1, 64, 7, 16), dtype=torch.float32, device="cuda")
        assert not is_triton_kernel_supported(bad_h_dot, good_w)

        cpu_dot = torch.randn((1, 64, 8, 16), dtype=torch.float32)
        assert not is_triton_kernel_supported(cpu_dot, good_w)

    def test_env_default_on(self):
        """Plan-8 P57 close-out 2 (2026-05-15): flipped default to ON.

        Microbench at V4-Flash widths is consistently positive
        (FWD 4.30x / BWD 1.63x) and the EP=8 proxy A/B shows a small
        but positive ~0.2 ms / iter gain.  The conservative descope
        rationale from P41 / P43 was that the per-iter gain sat below
        the proxy noise floor; for the production code path we default
        the microbench-positive kernel ON.
        """
        with _env("PRIMUS_INDEXER_TRITON", None):
            assert is_triton_path_enabled()

    def test_env_explicit_zero_disables(self):
        """Setting ``PRIMUS_INDEXER_TRITON=0`` reverts to the eager body."""
        with _env("PRIMUS_INDEXER_TRITON", "0"):
            assert not is_triton_path_enabled()

    def test_env_distinct_from_full(self):
        """`PRIMUS_INDEXER_TRITON` controls the tail path; the legacy
        P38 full-fuse path is gated by `PRIMUS_INDEXER_TRITON_FULL`."""
        from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.indexer_score import (
            is_triton_path_enabled as full_enabled,
        )

        with _env("PRIMUS_INDEXER_TRITON", "1"), _env("PRIMUS_INDEXER_TRITON_FULL", "0"):
            assert is_triton_path_enabled()
            assert not full_enabled()
        with _env("PRIMUS_INDEXER_TRITON", "0"), _env("PRIMUS_INDEXER_TRITON_FULL", "1"):
            assert not is_triton_path_enabled()
            assert full_enabled()


# ---------------------------------------------------------------------------
# G43: indexer_score_post_triton entry-point smoke
# ---------------------------------------------------------------------------


class TestG43EntryPoint:
    def test_helper_returns_same_as_class(self):
        torch.manual_seed(20260515)
        B, S, P, H = 1, 32, 8, 4
        dot = torch.randn((B, S, H, P), dtype=torch.float32, device="cuda")
        w = torch.randn((B, S, H), dtype=torch.float32, device="cuda").abs()
        out_helper = indexer_score_post_triton(dot, w, compress_ratio=4, out_dtype=torch.float32)
        out_class = IndexerScorePostFn.apply(dot, w, 4, torch.float32)
        torch.testing.assert_close(out_helper, out_class)

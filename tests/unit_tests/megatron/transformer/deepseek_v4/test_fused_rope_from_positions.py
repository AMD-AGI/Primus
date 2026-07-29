###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Small-kernel-fusion 2026-07-03 — fused RoPE-from-positions parity.

:class:`RoPEFromPositionsFn` computes cos/sin in-kernel from
``(position_ids, inv_freq)`` (instead of consuming precomputed cos/sin).
Pins it FWD + BWD against the eager ``cos = pos*inv_freq -> rotate`` path
AND against :class:`RoPEInterleavedPartialFn` (which already matches eager),
plus an end-to-end check through :meth:`DualRoPE.apply_rope`.

GPU-only; CPU runs are skipped at collection time.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("fused RoPE kernel requires CUDA / HIP", allow_module_level=True)

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.dual_rope import DualRoPE  # noqa: E402
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rope_interleaved_partial import (  # noqa: E402
    RoPEFromPositionsFn,
    apply_rope_from_positions,
    eager_apply_interleaved_partial_rope,
)


def _tol(dtype):
    return {torch.float32: (1e-5, 1e-5), torch.bfloat16: (1e-2, 1e-2)}[dtype]


def _inv_freq(rotary_dim, theta=10000.0):
    i = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
    return 1.0 / (theta ** (i / rotary_dim))


def _eager(x, position_ids, inv_freq, rotary_dim, inverse=False):
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    if inverse:
        freqs = -freqs
    return eager_apply_interleaved_partial_rope(x, freqs.cos(), freqs.sin(), rotary_dim=rotary_dim)


class TestFwd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("H", [1, 64])
    @pytest.mark.parametrize("rotary_dim", [64, 128])
    def test_parity(self, dtype, H, rotary_dim):
        B, S, D = 2, 128, 512
        gen = torch.Generator(device="cuda").manual_seed(1)
        x = torch.randn((B, S, H, D), dtype=dtype, device="cuda", generator=gen)
        pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, S)
        inv = _inv_freq(rotary_dim)
        got = RoPEFromPositionsFn.apply(x, pos.broadcast_to(B, S).reshape(-1), inv, rotary_dim)
        ref = _eager(x, pos, inv, rotary_dim)
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)

    def test_broadcast_positions_1d(self):
        B, S, H, D, rd = 3, 64, 8, 512, 64
        x = torch.randn((B, S, H, D), dtype=torch.float32, device="cuda")
        pos1d = torch.arange(S, device="cuda")  # [S] -> broadcast to [B, S]
        inv = _inv_freq(rd)
        got = apply_rope_from_positions(x, pos1d, inv, rotary_dim=rd)
        ref = _eager(x, pos1d.unsqueeze(0).expand(B, S), inv, rd)
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


class TestInverse:
    """``inverse=True`` is the de-rotation the V4 attention output needs.

    It exists so the ``R(-t)`` before the O projection reuses this fused path
    (cos/sin generated in-kernel) instead of building cos/sin eagerly on all 43
    layers.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rotary_dim", [64, 128])
    def test_matches_eager_negated_angle(self, dtype, rotary_dim):
        B, S, H, D = 2, 128, 8, 512
        gen = torch.Generator(device="cuda").manual_seed(3)
        x = torch.randn((B, S, H, D), dtype=dtype, device="cuda", generator=gen)
        pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, S)
        inv = _inv_freq(rotary_dim)

        got = apply_rope_from_positions(x, pos, inv, rotary_dim=rotary_dim, inverse=True)
        ref = _eager(x, pos, inv, rotary_dim, inverse=True)
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)

    def test_round_trips_with_the_forward_rotation(self):
        B, S, H, D, rd = 2, 64, 4, 512, 64
        x = torch.randn((B, S, H, D), dtype=torch.float32, device="cuda")
        pos = torch.arange(S, device="cuda")
        inv = _inv_freq(rd)

        rotated = apply_rope_from_positions(x, pos, inv, rotary_dim=rd)
        back = apply_rope_from_positions(rotated, pos, inv, rotary_dim=rd, inverse=True)
        torch.testing.assert_close(back, x, atol=1e-5, rtol=1e-5)

        # Guard against a vacuous round trip.
        assert not torch.allclose(rotated, x, atol=1e-3, rtol=1e-3)

    def test_is_out_of_place(self):
        """The attention backward retains the raw output, so it must not be mutated."""
        B, S, H, D, rd = 1, 32, 4, 512, 64
        x = torch.randn((B, S, H, D), dtype=torch.float32, device="cuda")
        before = x.detach().clone()
        out = apply_rope_from_positions(x, torch.arange(S, device="cuda"), _inv_freq(rd), rotary_dim=rd, inverse=True)

        assert out.data_ptr() != x.data_ptr()
        torch.testing.assert_close(x, before, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_bwd_matches_eager(self, dtype):
        B, S, H, D, rd = 2, 64, 8, 512, 64
        gen = torch.Generator(device="cuda").manual_seed(4)
        xb = torch.randn((B, S, H, D), dtype=dtype, device="cuda", generator=gen)
        pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, S)
        inv = _inv_freq(rd)

        xt = xb.detach().clone().requires_grad_(True)
        xe = xb.detach().clone().requires_grad_(True)
        out_t = apply_rope_from_positions(xt, pos, inv, rotary_dim=rd, inverse=True)
        out_e = _eager(xe, pos, inv, rd, inverse=True)
        g = torch.randn_like(out_t)
        out_t.backward(g)
        out_e.backward(g.detach().clone())

        atol, rtol = _tol(dtype)
        torch.testing.assert_close(xt.grad, xe.grad, atol=atol, rtol=rtol)

    def test_inverse_differs_from_forward(self):
        """Sanity: the flag actually changes the result."""
        B, S, H, D, rd = 1, 32, 4, 512, 64
        x = torch.randn((B, S, H, D), dtype=torch.float32, device="cuda")
        pos = torch.arange(1, S + 1, device="cuda")  # avoid position 0 (identity)
        inv = _inv_freq(rd)

        fwd = apply_rope_from_positions(x, pos, inv, rotary_dim=rd)
        inv_out = apply_rope_from_positions(x, pos, inv, rotary_dim=rd, inverse=True)
        assert not torch.allclose(fwd, inv_out, atol=1e-4, rtol=1e-4)


class TestBwd:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("H", [1, 64])
    def test_bwd_parity(self, dtype, H):
        B, S, D, rd = 2, 64, 512, 64
        gen = torch.Generator(device="cuda").manual_seed(2)
        xb = torch.randn((B, S, H, D), dtype=dtype, device="cuda", generator=gen)
        pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, S)
        inv = _inv_freq(rd)

        xt = xb.detach().clone().requires_grad_(True)
        xe = xb.detach().clone().requires_grad_(True)
        out_t = RoPEFromPositionsFn.apply(xt, pos.broadcast_to(B, S).reshape(-1), inv, rd)
        out_e = _eager(xe, pos, inv, rd)
        g = torch.randn_like(out_t)
        out_t.backward(g)
        out_e.backward(g.detach().clone())
        atol, rtol = _tol(dtype)
        torch.testing.assert_close(xt.grad, xe.grad, atol=atol, rtol=rtol)


class TestDualRopeIntegration:
    @pytest.mark.parametrize("compress_ratio", [0, 4])
    def test_apply_rope_matches_eager(self, compress_ratio):
        rope = DualRoPE(
            rotary_dim=64,
            rope_theta=10000.0,
            compress_rope_theta=160000.0,
            yarn_factor=16.0,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            original_max_position_embeddings=65536,
        ).cuda()
        B, S, H, D = 2, 128, 8, 512
        x = torch.randn((B, S, H, D), dtype=torch.bfloat16, device="cuda")
        pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, S)

        got = rope.apply_rope(x, position_ids=pos, compress_ratio=compress_ratio)
        cache = rope.get_rope(compress_ratio=compress_ratio)
        ref = _eager(x, pos, cache.inv_freq, 64)
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)

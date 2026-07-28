###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Plan-6 P35 G38 — `apply_interleaved_partial_rope` Triton FWD/BWD parity.

Asserts that :class:`RoPEInterleavedPartialFn` (Triton kernel from
``primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rope_interleaved_partial``)
produces the same output as the eager
:func:`apply_interleaved_partial_rope` body in
``primus.backends.megatron.core.transformer.dual_rope``, FWD **and**
BWD, at two tiers:

* fast tier — small shapes (``B=2, S=8, H=4, head_dim=16, rd=8``)
  exercising every code path (nope copy, interleaved pair rotation,
  cos/sin broadcast across heads) in milliseconds; parametrised over
  ``{fp32, fp16, bf16}`` and ``rotary_dim ∈ {0, 4, 8, 16}``;
* release tier — Q shape (``B=1, S=4096, H=64, head_dim=512, rd=64``)
  and K shape (``B=1, S=4096, H=1, head_dim=64, rd=64``) mirroring the
  V4-Flash EP=8 proxy widths, behind ``pytest.mark.slow``.

`gradcheck` is run at the fast tier in fp32 to catch any analytic-VJP
bug in the BWD kernel; the FWD has a clean closed form so its forward
parity assertion is the load-bearing test.

GPU-only; CPU runs are ``pytest.skip``-ed at module collection time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip(
        "rope_interleaved_partial Triton kernel requires CUDA / HIP",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton not installed")

from primus.backends.megatron.core.transformer.dual_rope import (  # noqa: E402
    apply_interleaved_partial_rope,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rope_interleaved_partial import (  # noqa: E402
    RoPEInterleavedPartialFn,
    apply_rope_interleaved_partial,
    eager_apply_interleaved_partial_rope,
    is_triton_path_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _env(key: str, value: str | None):
    """Temporarily set / unset ``os.environ[key]``."""
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


def _build_cos_sin(
    *,
    leading_shape: tuple[int, ...],
    rotary_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos / sin with shape ``leading_shape + (rotary_dim // 2,)``.

    Mimics the way :class:`RoPECache` produces cos/sin: build
    ``position_ids.float() * inv_freq`` then take ``cos`` / ``sin``.
    For the test we just generate random freqs to exercise more values.
    """
    rd_half = rotary_dim // 2
    gen = torch.Generator(device="cuda").manual_seed(seed)
    freqs = torch.randn((*leading_shape, rd_half), dtype=torch.float32, device="cuda", generator=gen)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def _build_x(
    *,
    leading_shape: tuple[int, ...],
    H: int,
    head_dim: int,
    dtype: torch.dtype,
    requires_grad: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (*leading_shape, H, head_dim),
        dtype=dtype,
        device="cuda",
        generator=gen,
    )
    if requires_grad:
        x.requires_grad_(True)
    return x


def _dtype_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-6, 1e-6
    if dtype == torch.float16:
        return 1e-3, 1e-3
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    raise ValueError(dtype)


# ---------------------------------------------------------------------------
# G38: FWD parity vs eager
# ---------------------------------------------------------------------------


_FAST_LEADING = (2, 8)  # B=2, S=8
_FAST_H = 4
_FAST_HEAD_DIM = 16


class TestG38ForwardParity:
    """FWD output matches eager :func:`apply_interleaved_partial_rope`
    body within dtype tolerance.

    The Triton kernel does its arithmetic in the caller's dtype (bf16 /
    fp16 / fp32), matching the plan-5 P32 RoPE bf16 cast contract.  So
    the tolerance is the same as if we cast cos/sin to ``x.dtype`` and
    ran the eager body — which is exactly what
    :func:`eager_apply_interleaved_partial_rope` does.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("rotary_dim", [0, 4, 8, 16])
    def test_fast_tier_fwd_parity(self, dtype, rotary_dim):
        if rotary_dim > _FAST_HEAD_DIM:
            pytest.skip("rotary_dim must be <= head_dim")
        x = _build_x(
            leading_shape=_FAST_LEADING,
            H=_FAST_H,
            head_dim=_FAST_HEAD_DIM,
            dtype=dtype,
            seed=42,
        )
        cos, sin = _build_cos_sin(
            leading_shape=_FAST_LEADING,
            rotary_dim=max(rotary_dim, 2),
            dtype=dtype,
            seed=43,
        )
        # For rd=0 the kernel ignores cos/sin; pass any shape.
        cos_use = cos[..., : max(rotary_dim // 2, 1)]
        sin_use = sin[..., : max(rotary_dim // 2, 1)]

        out_triton = RoPEInterleavedPartialFn.apply(x, cos_use, sin_use, rotary_dim)
        out_eager = eager_apply_interleaved_partial_rope(x, cos_use, sin_use, rotary_dim=rotary_dim)

        atol, rtol = _dtype_tolerance(dtype)
        torch.testing.assert_close(out_triton, out_eager, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_uneven_block_h(self, dtype):
        """H=5 is not a multiple of BLOCK_H=8 -> masked load / store paths."""
        leading = (1, 7)  # also a prime sequence length
        H = 5
        head_dim = 12
        rotary_dim = 4
        x = _build_x(leading_shape=leading, H=H, head_dim=head_dim, dtype=dtype, seed=1234)
        cos, sin = _build_cos_sin(leading_shape=leading, rotary_dim=rotary_dim, dtype=dtype, seed=1235)
        out_triton = RoPEInterleavedPartialFn.apply(x, cos, sin, rotary_dim)
        out_eager = eager_apply_interleaved_partial_rope(x, cos, sin, rotary_dim=rotary_dim)
        atol, rtol = _dtype_tolerance(dtype)
        torch.testing.assert_close(out_triton, out_eager, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# G38: BWD parity vs eager + gradcheck
# ---------------------------------------------------------------------------


class TestG38BackwardParity:
    """BWD parity vs eager autograd graph, plus a small fp64 gradcheck."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("rotary_dim", [4, 8, 16])
    def test_fast_tier_bwd_parity(self, dtype, rotary_dim):
        if rotary_dim > _FAST_HEAD_DIM:
            pytest.skip("rotary_dim must be <= head_dim")

        x_t = _build_x(
            leading_shape=_FAST_LEADING,
            H=_FAST_H,
            head_dim=_FAST_HEAD_DIM,
            dtype=dtype,
            requires_grad=True,
            seed=44,
        )
        x_e = x_t.detach().clone().requires_grad_(True)
        cos, sin = _build_cos_sin(
            leading_shape=_FAST_LEADING,
            rotary_dim=rotary_dim,
            dtype=dtype,
            seed=45,
        )

        out_triton = RoPEInterleavedPartialFn.apply(x_t, cos, sin, rotary_dim)
        out_eager = eager_apply_interleaved_partial_rope(x_e, cos, sin, rotary_dim=rotary_dim)

        gen = torch.Generator(device="cuda").manual_seed(46)
        grad = torch.randn_like(out_triton, dtype=dtype)
        grad_e = grad.detach().clone()
        gen.manual_seed(46)
        out_triton.backward(grad)
        out_eager.backward(grad_e)

        atol, rtol = _dtype_tolerance(dtype)
        assert x_t.grad is not None
        assert x_e.grad is not None
        torch.testing.assert_close(x_t.grad, x_e.grad, atol=atol, rtol=rtol)

    def test_gradcheck_fast_tier(self):
        """`torch.autograd.gradcheck` at fp64 + tiny shape catches any
        analytic-VJP bug in :func:`_apply_rope_bwd_kernel`.
        """
        leading = (1, 4)
        H = 2
        head_dim = 8
        rotary_dim = 4
        x = _build_x(
            leading_shape=leading,
            H=H,
            head_dim=head_dim,
            dtype=torch.float64,
            requires_grad=True,
            seed=100,
        )
        cos, sin = _build_cos_sin(
            leading_shape=leading,
            rotary_dim=rotary_dim,
            dtype=torch.float64,
            seed=101,
        )

        def fn(xx):
            return RoPEInterleavedPartialFn.apply(xx, cos, sin, rotary_dim)

        torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5)


# ---------------------------------------------------------------------------
# G38: release-tier V4-Flash widths (Q + K)
# ---------------------------------------------------------------------------


class TestG38ReleaseTier:
    """V4-Flash EP=8 widths: Q (H=64, head_dim=512, rd=64) and K (H=1).

    Marked ``slow``; bf16 only (the production dtype).  FWD + BWD parity
    within the elementwise attention tolerance from plan-5 P32.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "spec",
        [
            ("Q", (1, 4096), 64, 512, 64),
            ("K", (1, 4096), 1, 64, 64),
        ],
        ids=["Q-shape", "K-shape"],
    )
    def test_v4_flash_proxy_parity(self, spec):
        name, leading, H, head_dim, rotary_dim = spec
        dtype = torch.bfloat16

        x_t = _build_x(
            leading_shape=leading,
            H=H,
            head_dim=head_dim,
            dtype=dtype,
            requires_grad=True,
            seed=200,
        )
        x_e = x_t.detach().clone().requires_grad_(True)
        cos, sin = _build_cos_sin(leading_shape=leading, rotary_dim=rotary_dim, dtype=dtype, seed=201)

        out_triton = RoPEInterleavedPartialFn.apply(x_t, cos, sin, rotary_dim)
        out_eager = eager_apply_interleaved_partial_rope(x_e, cos, sin, rotary_dim=rotary_dim)

        atol, rtol = _dtype_tolerance(dtype)
        torch.testing.assert_close(out_triton, out_eager, atol=atol, rtol=rtol)

        grad = torch.randn_like(out_triton)
        out_triton.backward(grad)
        out_eager.backward(grad.detach().clone())
        torch.testing.assert_close(x_t.grad, x_e.grad, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# G38: rd == 0 early-return + error paths
# ---------------------------------------------------------------------------


class TestG38EdgeCases:
    """Cover :class:`RoPEInterleavedPartialFn`'s defensive validation."""

    def test_rd_zero_returns_input_contiguous(self):
        x = _build_x(
            leading_shape=(2, 4),
            H=2,
            head_dim=8,
            dtype=torch.float32,
            seed=1,
        )
        cos, sin = _build_cos_sin(leading_shape=(2, 4), rotary_dim=2, dtype=torch.float32, seed=2)
        out = RoPEInterleavedPartialFn.apply(x, cos, sin, 0)
        assert torch.equal(out, x.contiguous())

    def test_odd_rotary_dim_raises(self):
        x = _build_x(
            leading_shape=(1, 4),
            H=2,
            head_dim=8,
            dtype=torch.float32,
            seed=1,
        )
        cos, sin = _build_cos_sin(leading_shape=(1, 4), rotary_dim=4, dtype=torch.float32, seed=2)
        with pytest.raises(ValueError, match="rotary_dim must be even"):
            RoPEInterleavedPartialFn.apply(x, cos, sin, 3)

    def test_rotary_dim_exceeds_head_dim_raises(self):
        x = _build_x(
            leading_shape=(1, 4),
            H=2,
            head_dim=8,
            dtype=torch.float32,
            seed=1,
        )
        cos, sin = _build_cos_sin(leading_shape=(1, 4), rotary_dim=4, dtype=torch.float32, seed=2)
        with pytest.raises(ValueError, match="must be <= head_dim"):
            RoPEInterleavedPartialFn.apply(x, cos, sin, 16)

    def test_cos_shape_mismatch_raises(self):
        x = _build_x(
            leading_shape=(2, 4),
            H=2,
            head_dim=8,
            dtype=torch.float32,
            seed=1,
        )
        cos, sin = _build_cos_sin(leading_shape=(2, 4), rotary_dim=2, dtype=torch.float32, seed=2)
        with pytest.raises(ValueError, match="last dim must be rotary_dim"):
            RoPEInterleavedPartialFn.apply(x, cos, sin, 4)


# ---------------------------------------------------------------------------
# G38: env-flag dispatch through dual_rope.apply_interleaved_partial_rope
# ---------------------------------------------------------------------------


class TestG38EnvFlagDispatch:
    """`PRIMUS_ROPE_TRITON` env knob flips the dispatcher behaviour
    inside :func:`apply_interleaved_partial_rope`.

    The Triton path is bit-equivalent to the eager body within bf16
    tolerance; this test pins both code paths to make sure neither is
    silently broken.
    """

    def test_env_on_uses_triton(self):
        x = _build_x(
            leading_shape=_FAST_LEADING,
            H=_FAST_H,
            head_dim=_FAST_HEAD_DIM,
            dtype=torch.bfloat16,
            seed=11,
        )
        cos, sin = _build_cos_sin(
            leading_shape=_FAST_LEADING,
            rotary_dim=8,
            dtype=torch.bfloat16,
            seed=12,
        )
        with _env("PRIMUS_ROPE_TRITON", "1"):
            assert is_triton_path_enabled()
            out_on = apply_interleaved_partial_rope(x, cos, sin, rotary_dim=8)
        with _env("PRIMUS_ROPE_TRITON", "0"):
            assert not is_triton_path_enabled()
            out_off = apply_interleaved_partial_rope(x, cos, sin, rotary_dim=8)
        atol, rtol = _dtype_tolerance(torch.bfloat16)
        torch.testing.assert_close(out_on, out_off, atol=atol, rtol=rtol)

    def test_apply_rope_interleaved_partial_dispatcher(self):
        """The dispatcher in the kernel module mirrors the dual_rope
        wiring -- pin both paths return the same answer.
        """
        x = _build_x(
            leading_shape=(2, 16),
            H=2,
            head_dim=16,
            dtype=torch.bfloat16,
            seed=21,
        )
        cos, sin = _build_cos_sin(
            leading_shape=(2, 16),
            rotary_dim=8,
            dtype=torch.bfloat16,
            seed=22,
        )
        with _env("PRIMUS_ROPE_TRITON", "1"):
            out_triton = apply_rope_interleaved_partial(x, cos, sin, rotary_dim=8)
        with _env("PRIMUS_ROPE_TRITON", "0"):
            out_eager = apply_rope_interleaved_partial(x, cos, sin, rotary_dim=8)
        atol, rtol = _dtype_tolerance(torch.bfloat16)
        torch.testing.assert_close(out_triton, out_eager, atol=atol, rtol=rtol)

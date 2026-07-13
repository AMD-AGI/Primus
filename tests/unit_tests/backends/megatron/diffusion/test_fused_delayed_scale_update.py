# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Validate the fused Triton kernel for delayed FP8 scale update.

Tests:
  1. Correctness: fused kernel matches reference Python-loop implementation
  2. Edge cases: zero amaxes, NaN amaxes, very small amaxes
  3. History rollover: circular index wraps correctly
  4. Algorithm variants: 'max' vs 'most_recent'

Run:
    python -m pytest tests/unit_tests/backends/megatron/diffusion/test_fused_delayed_scale_update.py -v
or standalone:
    python tests/unit_tests/backends/megatron/diffusion/test_fused_delayed_scale_update.py
"""

import pytest
import torch

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

DEVICE = "cuda:0"
FP8_FWD_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
FP8_BWD_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0
FP32_MAX = torch.finfo(torch.float32).max


def _reference_update(amax_history, staged_amaxes, scales, fp8_maxes, history_idx, use_max_algo):
    """Pure-PyTorch reference matching the fused delayed-scale update logic."""
    T, N, H = amax_history.shape
    for t in range(T):
        for m in range(N):
            new_amax = staged_amaxes[t, m].item()
            amax_history[t, m, history_idx] = new_amax

            if use_max_algo:
                amax = amax_history[t, m, :].max().item()
            else:
                amax = new_amax

            fp8_max = fp8_maxes[t].item()
            old_scale = scales[t, m].item()

            if amax > 0.0 and amax == amax:  # positive and not NaN
                sf = fp8_max / max(amax, 1e-12)
                sf = min(sf, FP32_MAX)
            else:
                sf = old_scale

            scales[t, m] = sf


def _run_fused_kernel(amax_history, staged_amaxes, scales, fp8_maxes, history_idx, use_max_algo):
    """Call the actual fused Triton kernel."""
    from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
        _fused_delayed_scale_update_kernel,
    )

    _, N, H = amax_history.shape
    BLOCK_H = triton.next_power_of_2(H) if H <= 1024 else 1024

    _fused_delayed_scale_update_kernel[(N, 3)](
        amax_history,
        staged_amaxes,
        scales,
        fp8_maxes,
        history_idx,
        N=N,
        H=H,
        use_max_algo=use_max_algo,
        BLOCK_H=BLOCK_H,
        FP32_MAX=FP32_MAX,
    )


@pytest.mark.parametrize(
    "N,H",
    [
        (1, 1),
        (4, 16),
        (57, 1024),
        (10, 128),
    ],
)
@pytest.mark.parametrize("algo", ["max", "most_recent"])
def test_correctness(N, H, algo):
    """Fused kernel must produce identical scales and history as reference."""
    use_max_algo = algo == "max"
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)

    torch.manual_seed(42)
    amax_history_ref = torch.rand(3, N, H, dtype=torch.float32, device=DEVICE) * 10.0
    staged_amaxes = torch.rand(3, N, dtype=torch.float32, device=DEVICE) * 5.0
    scales_ref = torch.ones(3, N, dtype=torch.float32, device=DEVICE)
    history_idx = H // 3

    amax_history_fused = amax_history_ref.clone()
    scales_fused = scales_ref.clone()

    _reference_update(amax_history_ref, staged_amaxes, scales_ref, fp8_maxes, history_idx, use_max_algo)
    _run_fused_kernel(amax_history_fused, staged_amaxes, scales_fused, fp8_maxes, history_idx, use_max_algo)

    torch.testing.assert_close(
        amax_history_fused,
        amax_history_ref,
        atol=0,
        rtol=0,
        msg=f"amax_history mismatch for N={N}, H={H}, algo={algo}",
    )
    torch.testing.assert_close(
        scales_fused, scales_ref, atol=1e-5, rtol=1e-5, msg=f"scales mismatch for N={N}, H={H}, algo={algo}"
    )


def test_zero_amaxes():
    """When all staged amaxes are zero and history is zero, scales should stay at old_scale."""
    N, H = 8, 16
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    staged_amaxes = torch.zeros(3, N, dtype=torch.float32, device=DEVICE)
    old_scales = torch.full((3, N), 42.0, dtype=torch.float32, device=DEVICE)
    scales = old_scales.clone()

    _run_fused_kernel(amax_history, staged_amaxes, scales, fp8_maxes, history_idx=0, use_max_algo=True)

    torch.testing.assert_close(
        scales, old_scales, atol=0, rtol=0, msg="Scales should be unchanged when all amaxes are zero"
    )


def test_nan_amaxes():
    """NaN amaxes should leave scales unchanged (NaN guard)."""
    N, H = 4, 8
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    staged_amaxes = torch.full((3, N), float("nan"), dtype=torch.float32, device=DEVICE)
    old_scales = torch.full((3, N), 7.0, dtype=torch.float32, device=DEVICE)
    scales = old_scales.clone()

    _run_fused_kernel(amax_history, staged_amaxes, scales, fp8_maxes, history_idx=0, use_max_algo=True)

    torch.testing.assert_close(
        scales, old_scales, atol=0, rtol=0, msg="Scales should be unchanged when amaxes are NaN"
    )


def test_very_small_amaxes():
    """Very small amaxes should produce large scales clamped to FP32_MAX."""
    N, H = 4, 8
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    staged_amaxes = torch.full((3, N), 1e-40, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(3, N, dtype=torch.float32, device=DEVICE)

    _run_fused_kernel(amax_history, staged_amaxes, scales, fp8_maxes, history_idx=0, use_max_algo=False)

    assert not scales.isinf().any(), "Scales should not be inf (should be clamped)"
    assert not scales.isnan().any(), "Scales should not be NaN"
    assert (scales <= FP32_MAX).all(), "Scales should be <= FP32_MAX"


@pytest.mark.parametrize("H", [4, 16, 64])
def test_history_rollover(H):
    """Circular index should write to correct position across multiple steps."""
    N = 3
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    scales = torch.ones(3, N, dtype=torch.float32, device=DEVICE)

    for step in range(H + 5):
        idx = step % H
        staged = torch.full((3, N), float(step + 1), dtype=torch.float32, device=DEVICE)
        _run_fused_kernel(amax_history, staged, scales, fp8_maxes, history_idx=idx, use_max_algo=True)

        assert amax_history[0, 0, idx].item() == float(
            step + 1
        ), f"History not written at idx={idx} on step={step}"


def test_multi_step_vs_reference():
    """Run 50 steps of both fused and reference, verify they stay in sync."""
    N, H = 10, 32
    use_max_algo = True
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)

    torch.manual_seed(123)
    amax_history_ref = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    amax_history_fused = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    scales_ref = torch.ones(3, N, dtype=torch.float32, device=DEVICE)
    scales_fused = torch.ones(3, N, dtype=torch.float32, device=DEVICE)

    for step in range(50):
        idx = step % H
        staged = torch.rand(3, N, dtype=torch.float32, device=DEVICE) * (10.0 + step)

        _reference_update(amax_history_ref, staged, scales_ref, fp8_maxes, idx, use_max_algo)
        _run_fused_kernel(amax_history_fused, staged, scales_fused, fp8_maxes, idx, use_max_algo)

    torch.testing.assert_close(
        amax_history_fused, amax_history_ref, atol=0, rtol=0, msg="amax_history diverged over 50 steps"
    )
    torch.testing.assert_close(
        scales_fused, scales_ref, atol=1e-5, rtol=1e-5, msg="scales diverged over 50 steps"
    )


def test_registry_integration():
    """Validate _fast_update_scales_with_history via _DelayedScalingRegistry."""
    from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
        _DelayedScalingRegistry,
        _fast_update_scales_with_history,
    )

    N, H = 8, 16

    class _FakeModule(torch.nn.Module):
        """Minimal stand-in for Float8*ParallelLinear with real tensors."""

        def __init__(self, weight_data):
            super().__init__()
            self._fp8_fwd_max = FP8_FWD_MAX
            self._fp8_bwd_max = FP8_BWD_MAX
            self._amax_compute_algo = "max"
            self._use_delayed_scaling = True
            self._first_delayed_step = True
            self._history_idx = 0
            self.weight = torch.nn.Parameter(weight_data)
            self.register_buffer("amax_history_input", torch.zeros(H, dtype=torch.float32, device=DEVICE))
            self.register_buffer("amax_history_weight", torch.zeros(H, dtype=torch.float32, device=DEVICE))
            self.register_buffer("amax_history_grad", torch.zeros(H, dtype=torch.float32, device=DEVICE))
            self.register_buffer("scale_input", torch.tensor(1.0, dtype=torch.float32, device=DEVICE))
            self.register_buffer("scale_weight", torch.tensor(1.0, dtype=torch.float32, device=DEVICE))
            self.register_buffer("scale_grad", torch.tensor(1.0, dtype=torch.float32, device=DEVICE))
            self.register_buffer("staged_input_amax", torch.tensor(0.0, dtype=torch.float32, device=DEVICE))
            self.register_buffer("staged_weight_amax", torch.tensor(0.0, dtype=torch.float32, device=DEVICE))
            self.register_buffer("staged_grad_amax", torch.tensor(0.0, dtype=torch.float32, device=DEVICE))

    torch.manual_seed(99)
    weights = [torch.randn(64, 64, dtype=torch.bfloat16, device=DEVICE) for _ in range(N)]
    modules = [_FakeModule(weights[i]) for i in range(N)]

    registry = _DelayedScalingRegistry(modules)

    # Staging is now done per-module: each m owns scalar buffers m.staged_input_amax,
    # m.staged_grad_amax. _fast_update_scales_with_history stacks them into
    # registry.staged_amaxes_3n[0/2] before launching the Triton kernel; scales
    # land in registry.scales_3n and are scattered back to m.scale_*.
    # Parallel pure-Python reference, driven by the EXACT amaxes the kernel used.
    fp8_maxes = torch.tensor([FP8_FWD_MAX, FP8_FWD_MAX, FP8_BWD_MAX], dtype=torch.float32, device=DEVICE)
    amax_history_ref = torch.zeros(3, N, H, dtype=torch.float32, device=DEVICE)
    scales_ref = torch.ones(3, N, dtype=torch.float32, device=DEVICE)
    history_idx_ref = 0
    use_max_algo = True  # _FakeModule sets _amax_compute_algo = "max"

    torch.manual_seed(77)
    for step in range(5):
        for i in range(N):
            modules[i].staged_input_amax.fill_(torch.rand(1, device=DEVICE).item() * 10)
            modules[i].staged_grad_amax.fill_(torch.rand(1, device=DEVICE).item() * 5)

        _fast_update_scales_with_history(registry)

        # The fused kernel only reads registry.staged_amaxes_3n (rows input/weight/grad)
        # and writes amax_history/scales, so reading it back yields the exact amaxes the
        # kernel consumed -- including the first-step weight bootstrap from ||weight||_inf
        # that is staged inside the call. Drive the reference with the same inputs and the
        # same history index it used this step.
        staged_used = registry.staged_amaxes_3n.clone()
        _reference_update(amax_history_ref, staged_used, scales_ref, fp8_maxes, history_idx_ref, use_max_algo)
        history_idx_ref = (history_idx_ref + 1) % H

    # Registry results must match the reference: history bit-for-bit, scales within fp32 tol.
    torch.testing.assert_close(
        registry.amax_history,
        amax_history_ref,
        atol=0,
        rtol=0,
        msg="registry amax_history diverged from reference loop",
    )
    torch.testing.assert_close(
        registry.scales_3n,
        scales_ref,
        atol=1e-5,
        rtol=1e-5,
        msg="registry scales diverged from reference loop",
    )

    # Per-module buffers must reflect the registry-batched results (rows input/weight/grad).
    for i, m in enumerate(modules):
        torch.testing.assert_close(m.scale_input, scales_ref[0, i], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(m.scale_weight, scales_ref[1, i], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(m.scale_grad, scales_ref[2, i], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(m.amax_history_input, amax_history_ref[0, i], atol=0, rtol=0)
        torch.testing.assert_close(m.amax_history_weight, amax_history_ref[1, i], atol=0, rtol=0)
        torch.testing.assert_close(m.amax_history_grad, amax_history_ref[2, i], atol=0, rtol=0)

    # Sanity: scales finite and positive.
    assert not registry.scales_3n.isnan().any(), "scales have NaN"
    assert not registry.scales_3n.isinf().any(), "scales have Inf"
    assert (registry.scales_3n[0] > 0).all(), "scale_input should be positive"
    assert (registry.scales_3n[1] > 0).all(), "scale_weight should be positive"

    assert registry._history_idx == 5 % H, f"history_idx should be {5 % H}, got {registry._history_idx}"
    for m in modules:
        assert m._history_idx == registry._history_idx, "Per-module _history_idx not synced with registry"


if __name__ == "__main__":
    print("=== Test 1: Correctness (max, various sizes) ===")
    for N, H in [(1, 1), (4, 16), (57, 1024), (10, 128)]:
        for algo in ["max", "most_recent"]:
            test_correctness(N, H, algo)
            print(f"  N={N}, H={H}, algo={algo}: PASS")

    print("\n=== Test 2: Zero amaxes ===")
    test_zero_amaxes()
    print("  PASS")

    print("\n=== Test 3: NaN amaxes ===")
    test_nan_amaxes()
    print("  PASS")

    print("\n=== Test 4: Very small amaxes ===")
    test_very_small_amaxes()
    print("  PASS")

    print("\n=== Test 5: History rollover ===")
    for H in [4, 16, 64]:
        test_history_rollover(H)
        print(f"  H={H}: PASS")

    print("\n=== Test 6: Multi-step vs reference ===")
    test_multi_step_vs_reference()
    print("  PASS")

    print("\n=== Test 7: Registry integration ===")
    test_registry_integration()
    print("  PASS")

    print("\nAll tests passed!")

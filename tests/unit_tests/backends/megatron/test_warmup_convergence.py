###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Single-GPU end-to-end test proving that warmup is transparent.

A model that undergoes warmup steps (with param snapshot, optimizer neutering,
FP8 reset) followed by a full state restore produces the **exact same
training trajectory** as a model that trains from scratch.

Parametrised over both FP8 spec paths (local delayed-scaling and
TransformerEngine).  The trainable modules are ``torch.nn.Linear`` subclasses
decorated with FP8 metadata; the forward pass runs standard bf16 matmuls.
Loss-matching validates param snapshot/restore + optimizer reset.  FP8 state
health (no NaN scales, registry reinit, amax seeding) is checked separately.

Run:
    python -m pytest tests/unit_tests/backends/megatron/test_warmup_convergence.py -v
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

DEVICE = "cuda:0"
FP8_FWD_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_BWD_MAX = torch.finfo(torch.float8_e5m2).max

WARMUP_STEPS = 3
TRAIN_STEPS = 20
IN_FEATURES = 32
HIDDEN = 64
OUT_FEATURES = 16
BATCH = 8


# ---------------------------------------------------------------------------
# Trainable FP8 module helpers
# ---------------------------------------------------------------------------


class _DelayedScalingLinear(torch.nn.Linear):
    """``torch.nn.Linear`` with local-spec delayed-scaling FP8 metadata."""

    def __init__(self, in_features, out_features, history_len=1):
        super().__init__(in_features, out_features)
        self._use_delayed_scaling = True
        self._fp8_fwd_dtype = torch.float8_e4m3fn
        self._fp8_bwd_dtype = torch.float8_e5m2
        self._fp8_fwd_max = FP8_FWD_MAX
        self._fp8_bwd_max = FP8_BWD_MAX
        self._amax_compute_algo = "most_recent"
        self._first_delayed_step = True
        self._history_idx = 0
        self.config = SimpleNamespace(
            fp8_amax_history_len=history_len,
            fp8_amax_compute_algo="most_recent",
        )

        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            _init_delayed_scaling_state,
        )

        _init_delayed_scaling_state(self)

    def _move_fp8_buffers(self, device):
        for name in list(self._buffers):
            buf = self._buffers[name]
            if buf is not None and buf.device != device:
                self._buffers[name] = buf.to(device)


class _TEScalingLinear(torch.nn.Linear):
    """``torch.nn.Linear`` with TransformerEngine-style FP8 metadata."""

    def __init__(self, in_features, out_features, device=DEVICE):
        super().__init__(in_features, out_features)
        self.fp8_initialized = True
        self.fp8_meta = {
            "scaling_fwd": SimpleNamespace(
                amax_history=torch.zeros(16, 1, device=device),
                scale=torch.ones(1, device=device),
                scale_inv=torch.ones(1, device=device),
            ),
            "scaling_bwd": SimpleNamespace(
                amax_history=torch.zeros(16, 1, device=device),
                scale=torch.ones(1, device=device),
                scale_inv=torch.ones(1, device=device),
            ),
        }


def _make_model(fp8_spec, seed, device=DEVICE):
    """Create a 3-layer MLP with the requested FP8 metadata, on ``device``."""
    torch.manual_seed(seed)
    if fp8_spec == "local":
        model = torch.nn.Sequential(
            _DelayedScalingLinear(IN_FEATURES, HIDDEN),
            torch.nn.ReLU(),
            _DelayedScalingLinear(HIDDEN, OUT_FEATURES),
        ).to(device)
        for m in model.modules():
            if isinstance(m, _DelayedScalingLinear):
                m._move_fp8_buffers(torch.device(device))
    elif fp8_spec == "te":
        model = torch.nn.Sequential(
            _TEScalingLinear(IN_FEATURES, HIDDEN, device=device),
            torch.nn.ReLU(),
            _TEScalingLinear(HIDDEN, OUT_FEATURES, device=device),
        ).to(device)
    else:
        raise ValueError(f"Unknown fp8_spec: {fp8_spec}")
    return model


def _get_delayed_modules(model):
    return [m for m in model.modules() if getattr(m, "_use_delayed_scaling", False)]


def _make_registry(model):
    from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
        _DelayedScalingRegistry,
    )

    modules = _get_delayed_modules(model)
    assert modules, "No delayed-scaling modules found"
    return _DelayedScalingRegistry(modules)


def _generate_fixed_data(num_batches, seed=99, device=DEVICE):
    torch.manual_seed(seed)
    return [
        (
            torch.randn(BATCH, IN_FEATURES, device=device),
            torch.randn(BATCH, OUT_FEATURES, device=device),
        )
        for _ in range(num_batches)
    ]


def _run_warmup(model, adam_opt, fp8_spec, registry=None):
    """Execute the full warmup pipeline: snapshot, neuter, SGD steps, restore.

    Returns the registry (possibly reinitialised for local spec).
    """
    from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
        _fast_update_scales,
    )
    from primus.backends.megatron.patches.mlperf_warmup_patches import (
        _neuter_optimizer,
        _reset_fp8_local_spec,
        _reset_fp8_te_spec,
        _reset_optimizer_state,
        _restore_optimizer,
        _seed_fp8_amax,
    )

    wrapper = SimpleNamespace(optimizer=adam_opt)
    device = next(model.parameters()).device

    # 1. Snapshot params to CPU
    saved_params = {name: p.data.to("cpu", non_blocking=True) for name, p in model.named_parameters()}
    torch.cuda.synchronize()

    # 2. Neuter Adam
    saved_hyp = _neuter_optimizer(wrapper)

    # 3-4. Throwaway SGD warmup steps
    warmup_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(WARMUP_STEPS):
        if fp8_spec == "local" and registry is not None:
            _fast_update_scales(registry)
        x = torch.randn(BATCH, IN_FEATURES, device=device)
        y = torch.randn(BATCH, OUT_FEATURES, device=device)
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        warmup_sgd.step()
        warmup_sgd.zero_grad()

    # 5. Restore Adam hyperparams
    _restore_optimizer(wrapper, saved_hyp)

    # 6. Reset Adam step counters
    _reset_optimizer_state(wrapper)

    # 7. Restore params from CPU snapshot
    for name, p in model.named_parameters():
        if name in saved_params:
            p.data.copy_(saved_params[name])
    del saved_params

    # 8. Reset FP8 state
    if fp8_spec == "local":
        _reset_fp8_local_spec([model])
    elif fp8_spec == "te":
        _reset_fp8_te_spec([model])
        _seed_fp8_amax([model])

    # 9. Synchronize
    torch.cuda.synchronize()

    # 10. Zero gradients
    adam_opt.zero_grad(set_to_none=True)

    return registry


def _train_loop(model, optimizer, data, fp8_spec, registry=None):
    """Run training and return per-step losses."""
    from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
        _fast_update_scales,
    )

    losses = []
    for x, y in data:
        if fp8_spec == "local" and registry is not None:
            _fast_update_scales(registry)
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach())
    return losses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fp8_spec", ["local", "te"])
class TestWarmupConvergence:

    def test_loss_matches_baseline(self, fp8_spec):
        """After warmup + full reset, loss curve is identical to no-warmup baseline."""
        seed = 42

        baseline_model = _make_model(fp8_spec, seed)
        warmup_model = _make_model(fp8_spec, seed)

        # Verify identical starting params
        for (n1, p1), (n2, p2) in zip(baseline_model.named_parameters(), warmup_model.named_parameters()):
            assert torch.equal(p1.data, p2.data), f"Init mismatch on {n1}"

        baseline_opt = torch.optim.Adam(
            baseline_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01
        )
        warmup_opt = torch.optim.Adam(
            warmup_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01
        )

        baseline_reg = _make_registry(baseline_model) if fp8_spec == "local" else None
        warmup_reg = _make_registry(warmup_model) if fp8_spec == "local" else None

        # Run warmup on warmup_model
        _run_warmup(warmup_model, warmup_opt, fp8_spec, registry=warmup_reg)

        # Generate fixed training data
        data = _generate_fixed_data(TRAIN_STEPS)

        # Train both
        baseline_losses = _train_loop(baseline_model, baseline_opt, data, fp8_spec, registry=baseline_reg)
        warmup_losses = _train_loop(warmup_model, warmup_opt, data, fp8_spec, registry=warmup_reg)

        for i, (lb, lw) in enumerate(zip(baseline_losses, warmup_losses)):
            if i < 5:
                assert torch.equal(
                    lb, lw
                ), f"Step {i}: bitwise mismatch baseline={lb.item()}, warmup={lw.item()}"
            else:
                assert abs(lb.item() - lw.item()) < 1e-5, (
                    f"Step {i}: baseline={lb.item()}, warmup={lw.item()}, "
                    f"diff={abs(lb.item() - lw.item())}"
                )

    def test_fp8_state_healthy_after_reset(self, fp8_spec):
        """FP8 metadata has no NaN/Inf after warmup + reset."""
        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            _fast_update_scales,
        )

        seed = 42
        model = _make_model(fp8_spec, seed)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        registry = _make_registry(model) if fp8_spec == "local" else None

        _run_warmup(model, opt, fp8_spec, registry=registry)

        if fp8_spec == "local":
            _fast_update_scales(registry)

            assert (
                registry._first_step is False
            ), "_first_step should be consumed after post-reset _fast_update_scales"
            assert torch.isfinite(
                registry.scales_3n
            ).all(), f"Non-finite scales after reset: {registry.scales_3n}"

            for i, m in enumerate(_get_delayed_modules(model)):
                expected_amax = m.weight.data.abs().amax().float().item()
                actual_amax = m.staged_weight_amax.item()
                assert (
                    abs(actual_amax - expected_amax) < 1e-3
                ), f"Module {i}: weight amax {actual_amax} != expected {expected_amax}"

        elif fp8_spec == "te":
            for m in model.modules():
                if hasattr(m, "fp8_initialized"):
                    assert m.fp8_initialized is False, "fp8_initialized should be False after TE reset"
                if hasattr(m, "fp8_meta"):
                    meta = m.fp8_meta
                    for key in ("scaling_fwd", "scaling_bwd"):
                        if key not in meta:
                            continue
                        tm = meta[key]
                        if hasattr(tm, "amax_history"):
                            assert (tm.amax_history == 1.0).all(), (
                                f"{key}.amax_history should be seeded to 1.0, " f"got {tm.amax_history}"
                            )
                        if hasattr(tm, "scale"):
                            assert torch.isfinite(tm.scale).all(), f"{key}.scale has non-finite values"

    def test_optimizer_step_parity(self, fp8_spec):
        """Optimizer step counters match baseline after same number of real steps."""
        seed = 42
        train_steps = 10

        baseline_model = _make_model(fp8_spec, seed)
        warmup_model = _make_model(fp8_spec, seed)

        baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        warmup_opt = torch.optim.Adam(warmup_model.parameters(), lr=1e-3, betas=(0.9, 0.999))

        baseline_reg = _make_registry(baseline_model) if fp8_spec == "local" else None
        warmup_reg = _make_registry(warmup_model) if fp8_spec == "local" else None

        _run_warmup(warmup_model, warmup_opt, fp8_spec, registry=warmup_reg)

        data = _generate_fixed_data(train_steps, seed=77)

        _train_loop(baseline_model, baseline_opt, data, fp8_spec, registry=baseline_reg)
        _train_loop(warmup_model, warmup_opt, data, fp8_spec, registry=warmup_reg)

        for (p_b, state_b), (p_w, state_w) in zip(baseline_opt.state.items(), warmup_opt.state.items()):
            step_b = state_b["step"]
            step_w = state_w["step"]
            if isinstance(step_b, torch.Tensor):
                step_b = step_b.item()
            if isinstance(step_w, torch.Tensor):
                step_w = step_w.item()
            assert step_b == step_w == train_steps, (
                f"Step count mismatch: baseline={step_b}, warmup={step_w}, " f"expected={train_steps}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

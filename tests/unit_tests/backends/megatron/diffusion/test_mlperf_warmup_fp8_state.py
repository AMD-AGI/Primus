###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
GPU unit tests for FP8 delayed-scaling state reset during MLPerf warmup.

Validates that ``_reset_fp8_local_spec`` correctly breaks buffer pointers,
triggering ``_DelayedScalingRegistry`` reinitialisation with
``_first_step = True`` so that the first real step bootstraps weight amaxes
from the restored (pre-warmup) weights.

Requires a CUDA device; skipped automatically when unavailable.

Run:
    python -m pytest tests/unit_tests/backends/megatron/diffusion/test_mlperf_warmup_fp8_state.py -v
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

DEVICE = "cuda:0"
FP8_FWD_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_BWD_MAX = torch.finfo(torch.float8_e5m2).max


class _FakeDelayedModule(torch.nn.Module):
    """Minimal stand-in for a Float8*ParallelLinear with delayed scaling."""

    def __init__(self, in_features=64, out_features=64, history_len=1):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16, device=DEVICE)
        )
        self._use_delayed_scaling = True
        self._fp8_fwd_dtype = torch.float8_e4m3fn
        self._fp8_bwd_dtype = torch.float8_e5m2
        self._fp8_fwd_max = FP8_FWD_MAX
        self._fp8_bwd_max = FP8_BWD_MAX
        self._amax_compute_algo = "most_recent"
        self._first_delayed_step = True
        self._history_idx = 0
        self.config = type(
            "Config",
            (),
            {
                "fp8_amax_history_len": history_len,
                "fp8_amax_compute_algo": "most_recent",
            },
        )()

        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            _init_delayed_scaling_state,
        )

        _init_delayed_scaling_state(self)
        for name in list(self._buffers):
            buf = self._buffers[name]
            if buf is not None and buf.device.type == "cpu":
                self._buffers[name] = buf.to(DEVICE)


class TestResetFp8LocalSpecOnDevice:
    """Verify _reset_fp8_local_spec creates new buffers on the correct device."""

    def test_buffers_on_device_after_reset(self):
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_local_spec,
        )

        module = _FakeDelayedModule()
        model = torch.nn.Sequential(module)

        count = _reset_fp8_local_spec([model])
        assert count == 1

        for buf_name in (
            "scale_input",
            "scale_weight",
            "scale_grad",
            "amax_history_input",
            "amax_history_weight",
            "amax_history_grad",
            "staged_input_amax",
            "staged_grad_amax",
            "staged_weight_amax",
        ):
            buf = module._buffers.get(buf_name)
            assert buf is not None, f"Buffer {buf_name} missing after reset"
            assert (
                buf.device == module.weight.device
            ), f"Buffer {buf_name} on {buf.device}, expected {module.weight.device}"

    def test_first_delayed_step_reset(self):
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_local_spec,
        )

        module = _FakeDelayedModule()
        module._first_delayed_step = False

        _reset_fp8_local_spec([torch.nn.Sequential(module)])

        assert module._first_delayed_step is True


class TestRegistryPointerBreak:
    """Verify that reset breaks pointer identity with registry global tensors."""

    def test_pointer_mismatch_triggers_reinit(self):
        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            _DelayedScalingRegistry,
        )
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_local_spec,
        )

        modules = [_FakeDelayedModule() for _ in range(4)]
        model = torch.nn.Sequential(*modules)
        registry = _DelayedScalingRegistry(modules)

        # The detection-via-aliasing check in _fast_update_scales_with_history
        # is keyed on amax_history (the only registry tensor that the modules
        # hold views into); per-module scale_* buffers have always been
        # independent scalar tensors. Verify the alias holds pre-reset and is
        # broken post-reset.
        old_registry_amax_ptr = registry.amax_history.untyped_storage().data_ptr()
        old_mod0_amax_ptr = modules[0].amax_history_input.untyped_storage().data_ptr()
        assert (
            old_registry_amax_ptr == old_mod0_amax_ptr
        ), "Sanity: registry.amax_history must alias each module's amax_history_input"

        _reset_fp8_local_spec([model])

        new_mod0_amax_ptr = modules[0].amax_history_input.untyped_storage().data_ptr()
        assert new_mod0_amax_ptr != old_registry_amax_ptr, (
            "After reset, module amax_history_input should have different storage "
            "than the original registry.amax_history (so the next "
            "_fast_update_scales* call triggers registry.__init__ via the "
            "data_ptr mismatch check)"
        )


class TestRegistryReinitialisationOnFirstStep:
    """End-to-end: reset → _fast_update_scales detects mismatch → re-creates registry."""

    def test_fast_path_reinit(self):
        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            _DelayedScalingRegistry,
            _fast_update_scales,
        )
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_local_spec,
        )

        modules = [_FakeDelayedModule(history_len=1) for _ in range(4)]
        model = torch.nn.Sequential(*modules)
        registry = _DelayedScalingRegistry(modules)

        _fast_update_scales(registry)
        assert registry._first_step is False

        new_weights = [torch.randn_like(m.weight.data) for m in modules]
        for m, w in zip(modules, new_weights):
            m.weight.data.copy_(w)

        _reset_fp8_local_spec([model])
        _fast_update_scales(registry)

        assert registry._first_step is False, "Should have consumed _first_step"
        for i, m in enumerate(modules):
            expected_amax = new_weights[i].abs().amax().float().item()
            # _fast_update_scales writes per-module weight amaxes to
            # m.staged_weight_amax in the _first_step bootstrap path; the
            # registry-batched staged_amaxes_3n is only populated by
            # _fast_update_scales_with_history, so read from the module here.
            actual_amax = m.staged_weight_amax.item()
            assert (
                abs(actual_amax - expected_amax) < 1e-3
            ), f"Module {i}: weight amax {actual_amax} != expected {expected_amax}"

    def test_history_path_reinit(self):
        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            _DelayedScalingRegistry,
            _fast_update_scales_with_history,
        )
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_local_spec,
        )

        H = 16
        modules = [_FakeDelayedModule(history_len=H) for _ in range(4)]
        model = torch.nn.Sequential(*modules)
        registry = _DelayedScalingRegistry(modules)

        _fast_update_scales_with_history(registry)
        assert registry._first_step is False

        new_weights = [torch.randn_like(m.weight.data) for m in modules]
        for m, w in zip(modules, new_weights):
            m.weight.data.copy_(w)

        _reset_fp8_local_spec([model])
        _fast_update_scales_with_history(registry)

        assert registry._first_step is False
        for i, m in enumerate(modules):
            expected_amax = new_weights[i].abs().amax().float().item()
            # _fast_update_scales_with_history mirrors the staged amaxes into
            # registry.staged_amaxes_3n (rows 0/1/2 = input/weight/grad), so
            # both registry.staged_amaxes_3n[1, i] and m.staged_weight_amax
            # carry the value. Read from the module for consistency with the
            # fast-path test above.
            actual_amax = m.staged_weight_amax.item()
            assert (
                abs(actual_amax - expected_amax) < 1e-3
            ), f"Module {i}: weight amax {actual_amax} != expected {expected_amax}"


class TestScaleLeakagePrevention:
    """Verify that warmup-phase scales do not leak into post-reset state."""

    def test_scales_recomputed_from_clean_state(self):
        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            _DelayedScalingRegistry,
            _fast_update_scales,
        )
        from primus.backends.megatron.patches.mlperf_warmup_patches import (
            _reset_fp8_local_spec,
        )

        modules = [_FakeDelayedModule(history_len=1) for _ in range(4)]
        model = torch.nn.Sequential(*modules)
        registry = _DelayedScalingRegistry(modules)

        # Staging is now per-module: registry batches them into
        # registry.staged_amaxes_3n inside _fast_update_scales_with_history,
        # but _fast_update_scales reads directly from the per-module buffers.
        # _fast_update_scales also scatters new scales back to per-module
        # m.scale_* buffers (registry.scales_3n is no longer the
        # source-of-truth for the fast path), so we read scales from there.
        for _ in range(5):
            for i, m in enumerate(modules):
                m.staged_input_amax.fill_(100.0 + i)
                m.staged_grad_amax.fill_(50.0 + i)
            _fast_update_scales(registry)

        warmup_scales = torch.stack(
            [
                torch.stack([m.scale_input.clone() for m in modules]),
                torch.stack([m.scale_weight.clone() for m in modules]),
                torch.stack([m.scale_grad.clone() for m in modules]),
            ]
        )
        assert (warmup_scales != 1.0).any(), "Scales should have changed during warmup"

        new_weights = [torch.randn_like(m.weight.data) * 0.01 for m in modules]
        for m, w in zip(modules, new_weights):
            m.weight.data.copy_(w)

        _reset_fp8_local_spec([model])
        _fast_update_scales(registry)

        post_reset_scales = torch.stack(
            [
                torch.stack([m.scale_input.clone() for m in modules]),
                torch.stack([m.scale_weight.clone() for m in modules]),
                torch.stack([m.scale_grad.clone() for m in modules]),
            ]
        )
        assert not torch.equal(
            warmup_scales, post_reset_scales
        ), "Scales should differ after reset with new weights"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

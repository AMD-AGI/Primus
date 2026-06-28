# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for FSDP2 FP32 optimizer patches.

Tests:
- FSDP2 FP32 optimizer patch condition evaluation and application
- Float16Module skip patch condition evaluation and application
- _FSDP2PassthroughModule preserves FP32 weights and passes through forward
- Mutual exclusivity of bf16 and fsdp2_fp32 optimizer patches
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from primus.core.patches.context import PatchContext
from primus.core.patches.patch_runner import run_patches


def _build_patch_context(backend_args=None):
    if backend_args is None:
        backend_args = SimpleNamespace()
    module_config = SimpleNamespace(params=backend_args)
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"backend_args": backend_args, "module_config": module_config},
    )


def _install_fake_megatron_modules(monkeypatch):
    training_mod = types.ModuleType("megatron.training.training")
    training_pkg = types.ModuleType("megatron.training")
    training_pkg.training = training_mod

    core_pkg = types.ModuleType("megatron.core")
    optimizer_mod = types.ModuleType("megatron.core.optimizer")
    core_pkg.optimizer = optimizer_mod

    megatron_pkg = types.ModuleType("megatron")
    megatron_pkg.training = training_pkg
    megatron_pkg.core = core_pkg

    monkeypatch.setitem(sys.modules, "megatron", megatron_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training", training_pkg)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_pkg)
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer", optimizer_mod)

    return training_mod, optimizer_mod


def _install_fsdp2_fp32_stub(monkeypatch, optimizer_fn):
    stub = types.ModuleType("primus.backends.megatron.core.optimizer.fsdp2_fp32_optimizer")
    stub.get_fsdp2_fp32_optimizer = optimizer_fn
    monkeypatch.setitem(
        sys.modules,
        "primus.backends.megatron.core.optimizer.fsdp2_fp32_optimizer",
        stub,
    )


class TestFSDP2FP32OptimizerPatch:
    """Tests for the FSDP2 FP32 optimizer patch."""

    def test_patch_applies_when_all_conditions_met(self, monkeypatch):
        from primus.backends.megatron.patches.optimizer_patches import (
            patch_fsdp2_fp32_optimizer,
        )

        backend_args = SimpleNamespace(
            use_fsdp2_fp32_param_optimizer=True,
            use_torch_fsdp2=True,
            bf16=True,
        )
        ctx = _build_patch_context(backend_args)

        training_mod, optimizer_mod = _install_fake_megatron_modules(monkeypatch)
        original = lambda *args, **kwargs: Mock()
        training_mod.get_megatron_optimizer = original
        optimizer_mod.get_megatron_optimizer = original

        mock_optimizer = Mock()
        _install_fsdp2_fp32_stub(monkeypatch, lambda *args, **kwargs: mock_optimizer)

        monkeypatch.setattr(
            "primus.backends.megatron.patches.optimizer_patches.log_rank_0",
            lambda *args, **kwargs: None,
        )

        patch_fsdp2_fp32_optimizer(ctx)

        assert training_mod.get_megatron_optimizer is not original
        assert optimizer_mod.get_megatron_optimizer is not original

        result = training_mod.get_megatron_optimizer(Mock(), [Mock()])
        assert result is mock_optimizer

    @pytest.mark.parametrize(
        "backend_args",
        [
            SimpleNamespace(use_fsdp2_fp32_param_optimizer=False, use_torch_fsdp2=True, bf16=True),
            SimpleNamespace(use_fsdp2_fp32_param_optimizer=True, use_torch_fsdp2=False, bf16=True),
            SimpleNamespace(use_fsdp2_fp32_param_optimizer=True, use_torch_fsdp2=True, bf16=False),
        ],
    )
    def test_patch_skipped_when_any_condition_false(self, monkeypatch, backend_args):
        import primus.backends.megatron.patches.optimizer_patches  # noqa: F401

        monkeypatch.setattr(
            "primus.core.patches.patch_runner.log_rank_0",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "primus.core.patches.patch_runner.error_rank_0",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "primus.core.patches.patch.log_rank_0",
            lambda *args, **kwargs: None,
        )

        training_mod, optimizer_mod = _install_fake_megatron_modules(monkeypatch)
        original = lambda *args, **kwargs: Mock()
        training_mod.get_megatron_optimizer = original
        optimizer_mod.get_megatron_optimizer = original

        count = run_patches(
            backend="megatron",
            phase="before_train",
            extra={"module_config": SimpleNamespace(params=backend_args)},
            enabled_ids=["megatron.optimizer.fsdp2_fp32_param"],
        )

        assert count == 0
        assert training_mod.get_megatron_optimizer is original

    def test_patched_optimizer_filters_megatron_kwargs(self, monkeypatch):
        from primus.backends.megatron.patches.optimizer_patches import (
            patch_fsdp2_fp32_optimizer,
        )

        backend_args = SimpleNamespace(
            use_fsdp2_fp32_param_optimizer=True,
            use_torch_fsdp2=True,
            bf16=True,
        )
        ctx = _build_patch_context(backend_args)

        training_mod, optimizer_mod = _install_fake_megatron_modules(monkeypatch)
        training_mod.get_megatron_optimizer = lambda *a, **kw: Mock()
        optimizer_mod.get_megatron_optimizer = lambda *a, **kw: Mock()

        received_kwargs = {}

        def tracking_factory(**kwargs):
            received_kwargs.update(kwargs)
            return Mock()

        _install_fsdp2_fp32_stub(monkeypatch, lambda *args, **kwargs: tracking_factory(**kwargs))

        monkeypatch.setattr(
            "primus.backends.megatron.patches.optimizer_patches.log_rank_0",
            lambda *args, **kwargs: None,
        )

        patch_fsdp2_fp32_optimizer(ctx)

        training_mod.get_megatron_optimizer(
            Mock(),
            [Mock()],
            config_overrides={"x": 1},
            use_gloo_process_groups=True,
            no_weight_decay_cond=Mock(),
        )

        assert "config_overrides" not in received_kwargs
        assert "use_gloo_process_groups" not in received_kwargs
        assert "no_weight_decay_cond" in received_kwargs


class TestSkipFloat16ModulePatch:
    """Tests for the Float16Module skip patch."""

    def test_patch_replaces_float16_module(self, monkeypatch):
        from primus.backends.megatron.patches.build_model_patches import (
            patch_skip_float16_module,
        )

        backend_args = SimpleNamespace(
            use_fsdp2_fp32_param_optimizer=True,
            use_torch_fsdp2=True,
            bf16=True,
        )
        ctx = _build_patch_context(backend_args)

        monkeypatch.setattr(
            "primus.backends.megatron.patches.build_model_patches.log_rank_0",
            lambda *args, **kwargs: None,
        )

        _install_fake_megatron_modules(monkeypatch)

        transformer_pkg = types.ModuleType("megatron.core.transformer")
        transformer_module_mod = types.ModuleType("megatron.core.transformer.module")

        class FakeFloat16Module:
            pass

        class FakeMegatronModule(torch.nn.Module):
            def __init__(self, config=None):
                super().__init__()

        transformer_module_mod.Float16Module = FakeFloat16Module
        transformer_module_mod.MegatronModule = FakeMegatronModule
        transformer_pkg.module = transformer_module_mod
        monkeypatch.setitem(
            sys.modules,
            "megatron.core.transformer",
            transformer_pkg,
        )
        monkeypatch.setitem(
            sys.modules,
            "megatron.core.transformer.module",
            transformer_module_mod,
        )
        sys.modules["megatron.core"].transformer = transformer_pkg

        patch_skip_float16_module(ctx)

        import megatron.training.training as training_mod

        assert training_mod.Float16Module is not FakeFloat16Module
        assert training_mod.Float16Module.__name__ == "_FSDP2PassthroughModule"

    def test_passthrough_module_preserves_fp32_weights(self, monkeypatch):
        """Verify _FSDP2PassthroughModule does NOT convert parameters to BF16
        and that forward() passes through correctly.

        This is the core contract: Float16Module converts to BF16, but the
        passthrough replacement must keep FP32 so FSDP2 MixedPrecisionPolicy
        can handle the casting.
        """
        from primus.backends.megatron.patches.build_model_patches import (
            patch_skip_float16_module,
        )

        backend_args = SimpleNamespace(
            use_fsdp2_fp32_param_optimizer=True,
            use_torch_fsdp2=True,
            bf16=True,
        )
        ctx = _build_patch_context(backend_args)

        monkeypatch.setattr(
            "primus.backends.megatron.patches.build_model_patches.log_rank_0",
            lambda *args, **kwargs: None,
        )

        _install_fake_megatron_modules(monkeypatch)

        transformer_pkg = types.ModuleType("megatron.core.transformer")
        transformer_module_mod = types.ModuleType("megatron.core.transformer.module")

        class FakeMegatronModule(torch.nn.Module):
            def __init__(self, config=None):
                super().__init__()

        class FakeFloat16Module:
            pass

        transformer_module_mod.Float16Module = FakeFloat16Module
        transformer_module_mod.MegatronModule = FakeMegatronModule
        transformer_pkg.module = transformer_module_mod
        monkeypatch.setitem(
            sys.modules,
            "megatron.core.transformer",
            transformer_pkg,
        )
        monkeypatch.setitem(
            sys.modules,
            "megatron.core.transformer.module",
            transformer_module_mod,
        )
        sys.modules["megatron.core"].transformer = transformer_pkg

        patch_skip_float16_module(ctx)

        import megatron.training.training as training_mod

        PassthroughCls = training_mod.Float16Module

        inner_model = torch.nn.Linear(4, 4, bias=True).to(dtype=torch.float32)
        config = SimpleNamespace(virtual_pipeline_model_parallel_size=None)
        wrapper = PassthroughCls(config, inner_model)

        # Parameters must remain FP32
        for name, p in wrapper.named_parameters():
            assert (
                p.dtype == torch.float32
            ), f"Parameter '{name}' was converted to {p.dtype}, expected float32"

        # wrapper.module must be the original model
        assert wrapper.module is inner_model

        # Forward must pass through correctly
        x = torch.randn(2, 4, dtype=torch.float32)
        expected = inner_model(x)
        actual = wrapper(x)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        "backend_args",
        [
            SimpleNamespace(use_fsdp2_fp32_param_optimizer=False, use_torch_fsdp2=True, bf16=True),
            SimpleNamespace(use_fsdp2_fp32_param_optimizer=True, use_torch_fsdp2=False, bf16=True),
            SimpleNamespace(use_fsdp2_fp32_param_optimizer=True, use_torch_fsdp2=True, bf16=False),
        ],
    )
    def test_patch_skipped_when_conditions_not_met(self, monkeypatch, backend_args):
        import primus.backends.megatron.patches.build_model_patches  # noqa: F401

        monkeypatch.setattr(
            "primus.core.patches.patch_runner.log_rank_0",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "primus.core.patches.patch_runner.error_rank_0",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "primus.core.patches.patch.log_rank_0",
            lambda *args, **kwargs: None,
        )

        count = run_patches(
            backend="megatron",
            phase="before_train",
            extra={"module_config": SimpleNamespace(params=backend_args)},
            enabled_ids=["megatron.training.training.skip_float16_module"],
        )

        assert count == 0


class TestFSDP2BF16MasterWeightOptimizerPatch:
    """Tests for the FSDP2 BF16 master weight optimizer patch."""

    def test_patch_applies_when_all_conditions_met(self, monkeypatch):
        from primus.backends.megatron.patches.optimizer_patches import (
            patch_fsdp2_bf16_master_weight_optimizer,
        )

        backend_args = SimpleNamespace(
            use_fsdp2_bf16_master_weight_optimizer=True,
            use_torch_fsdp2=True,
            bf16=True,
        )
        ctx = _build_patch_context(backend_args)

        training_mod, optimizer_mod = _install_fake_megatron_modules(monkeypatch)
        original = lambda *args, **kwargs: Mock()
        training_mod.get_megatron_optimizer = original
        optimizer_mod.get_megatron_optimizer = original

        mock_optimizer = Mock()
        bf16_mw_stub = types.ModuleType(
            "primus.backends.megatron.core.optimizer.fsdp2_bf16_master_weight_optimizer"
        )
        bf16_mw_stub.get_fsdp2_bf16_master_weight_optimizer = lambda *args, **kwargs: mock_optimizer
        monkeypatch.setitem(
            sys.modules,
            "primus.backends.megatron.core.optimizer.fsdp2_bf16_master_weight_optimizer",
            bf16_mw_stub,
        )

        monkeypatch.setattr(
            "primus.backends.megatron.patches.optimizer_patches.log_rank_0",
            lambda *args, **kwargs: None,
        )

        patch_fsdp2_bf16_master_weight_optimizer(ctx)

        assert training_mod.get_megatron_optimizer is not original
        assert optimizer_mod.get_megatron_optimizer is not original

        result = training_mod.get_megatron_optimizer(Mock(), [Mock()])
        assert result is mock_optimizer

    @pytest.mark.parametrize(
        "backend_args",
        [
            SimpleNamespace(
                use_fsdp2_bf16_master_weight_optimizer=False,
                use_torch_fsdp2=True,
                bf16=True,
            ),
            SimpleNamespace(
                use_fsdp2_bf16_master_weight_optimizer=True,
                use_torch_fsdp2=False,
                bf16=True,
            ),
            SimpleNamespace(
                use_fsdp2_bf16_master_weight_optimizer=True,
                use_torch_fsdp2=True,
                bf16=False,
            ),
        ],
    )
    def test_patch_skipped_when_any_condition_false(self, monkeypatch, backend_args):
        import primus.backends.megatron.patches.optimizer_patches  # noqa: F401

        monkeypatch.setattr(
            "primus.core.patches.patch_runner.log_rank_0",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "primus.core.patches.patch_runner.error_rank_0",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "primus.core.patches.patch.log_rank_0",
            lambda *args, **kwargs: None,
        )

        training_mod, optimizer_mod = _install_fake_megatron_modules(monkeypatch)
        original = lambda *args, **kwargs: Mock()
        training_mod.get_megatron_optimizer = original
        optimizer_mod.get_megatron_optimizer = original

        count = run_patches(
            backend="megatron",
            phase="before_train",
            extra={"module_config": SimpleNamespace(params=backend_args)},
            enabled_ids=["megatron.optimizer.fsdp2_bf16_master_weight"],
        )

        assert count == 0
        assert training_mod.get_megatron_optimizer is original


class TestOptimizerPatchMutualExclusivity:
    """Tests for mutual exclusivity of the FSDP2 custom optimizer patches."""

    def test_two_conflicting_optimizer_flags_raise(self):
        """Two FSDP2 custom optimizer flags set must fail loudly: the flags all
        patch get_megatron_optimizer at the same priority, so enabling more than
        one is ambiguous. validate_fsdp2_optimizer_exclusivity enforces this at
        arg-validation time."""
        from primus.modules.trainer.megatron.utils import (
            validate_fsdp2_optimizer_exclusivity,
        )

        args = SimpleNamespace(
            use_fsdp2_fp32_param_optimizer=True,
            use_fsdp2_bf16_master_weight_optimizer=True,
        )

        with pytest.raises(ValueError, match="Conflicting FSDP2 optimizer selection"):
            validate_fsdp2_optimizer_exclusivity(args)

    def test_single_optimizer_flag_is_allowed(self):
        """Exactly one FSDP2 optimizer flag is the supported case (no raise)."""
        from primus.modules.trainer.megatron.utils import (
            validate_fsdp2_optimizer_exclusivity,
        )

        for flag in (
            "use_fsdp2_fp32_param_optimizer",
            "use_fsdp2_bf16_master_weight_optimizer",
        ):
            args = SimpleNamespace(**{flag: True})
            # Should not raise.
            validate_fsdp2_optimizer_exclusivity(args)

    def test_no_optimizer_flag_is_allowed(self):
        """No FSDP2 optimizer flag set (default Megatron optimizer) must not raise."""
        from primus.modules.trainer.megatron.utils import (
            validate_fsdp2_optimizer_exclusivity,
        )

        validate_fsdp2_optimizer_exclusivity(SimpleNamespace())

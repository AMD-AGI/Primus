###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Hardware-independent contracts for the AutoModel MXFP6 path."""

import sys
import types

import pytest

from primus.backends.automodel.quantization import (
    _common,
    _fp4_common,
    _fp6_common,
    fp8_linear,
    te_mxfp4_linear,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="needs torch")


class TestSelection:
    def test_mxfp6_outranks_other_precisions(self):
        by_name = {entry.name: entry for entry in _common.registered_backends()}
        assert by_name[_fp6_common.BACKEND_NAME].precedence > by_name[_fp4_common.BACKEND_NAME].precedence
        assert by_name[_fp6_common.BACKEND_NAME].precedence > by_name[fp8_linear.BACKEND_NAME].precedence

    def test_mxfp6_wins_a_conflicting_request(self, set_option):
        set_option("primus_turbo.mxfp6_linear", True)
        set_option("primus_turbo.mxfp4_linear", True)
        set_option("primus_turbo.fp8_linear", True)
        assert _common.active_backend().name == _fp6_common.BACKEND_NAME

    def test_te_mxfp4_remains_the_most_specific_request(self, set_option):
        set_option("primus_turbo.mxfp6_linear", True)
        set_option("primus_te.mxfp4_linear", True)
        assert _common.active_backend().name == te_mxfp4_linear.BACKEND_NAME

    def test_swap_is_off_by_default(self):
        assert _fp6_common.is_enabled() is False

    def test_sensitive_layers_stay_bf16_by_default(self):
        assert _fp6_common.keep_sensitive_bf16() is True


class TestEligibility:
    @staticmethod
    def _install_automodel_predicate(monkeypatch):
        pipeline = types.ModuleType("nemo_automodel._diffusers.auto_diffusion_pipeline")
        pipeline._is_fp8_training_safe_linear = lambda name, linear: not name.startswith("unsafe.")
        for name in ("nemo_automodel", "nemo_automodel._diffusers"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, pipeline.__name__, pipeline)

    @pytest.mark.parametrize(
        "name,shape,eligible",
        [
            ("blocks.0.attn.to_q", (5120, 5120), True),
            ("blocks.0.ff.net.2", (5120, 13824), True),
            ("blocks.0.ff.net.0", (13824, 5120), True),
            ("blocks.0.proj", (5120, 4992), False),
            ("blocks.0.proj", (4992, 5120), False),
            ("condition_embedder.time_embedder.linear_1", (5120, 5120), False),
            ("unsafe.proj", (5120, 5120), False),
        ],
    )
    def test_model_policy_and_a6w6_alignment(self, monkeypatch, name, shape, eligible):
        self._install_automodel_predicate(monkeypatch)
        linear = types.SimpleNamespace(weight=types.SimpleNamespace(shape=shape))
        assert _fp6_common.is_fp6_training_safe_linear(name, linear) is eligible

    def test_sensitive_layers_can_be_enabled_explicitly(self, monkeypatch, set_option):
        self._install_automodel_predicate(monkeypatch)
        set_option("primus_turbo.mxfp6_keep_sensitive", False)
        linear = types.SimpleNamespace(weight=types.SimpleNamespace(shape=(5120, 5120)))
        assert _fp6_common.is_fp6_training_safe_linear("condition_embedder.proj", linear) is True


class TestTokenPadding:
    @pytest.mark.parametrize(
        "tokens,padded",
        [(1, 256), (255, 256), (256, 256), (257, 512), (75600, 75776)],
    )
    def test_rounds_to_an_a6w6_tile(self, tokens, padded):
        assert _fp6_common.pad_multiple(tokens) == padded

    @requires_torch
    def test_padding_adds_zero_rows_and_preserves_gradients(self):
        from primus.backends.automodel.quantization.mxfp6_gemm import _pad_tokens

        x = torch.randn(257, 8, requires_grad=True)
        padded, real_tokens = _pad_tokens(x)
        assert real_tokens == 257
        assert padded.shape == (512, 8)
        assert torch.count_nonzero(padded[real_tokens:]) == 0
        padded.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x))


@requires_torch
class TestHybridRoute:
    def test_forward_is_fp6_and_both_gradients_are_fp4(self, monkeypatch):
        """The Task 6 recipe is a routing contract, not just a config label."""
        calls = []
        fp6_impl_module = types.ModuleType("primus_turbo.pytorch.kernels.gemm.gemm_fp6_impl")
        fp6_pack_module = types.ModuleType("primus_turbo.pytorch.kernels.quantization.mxfp6_pack")
        backend_module = types.ModuleType("primus_turbo.pytorch.core.backend")
        precision_module = types.ModuleType("primus_turbo.pytorch.core.low_precision")
        fp4_impl_module = types.ModuleType("primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl")
        quant_module = types.ModuleType("primus_turbo.pytorch.ops.quantization")

        def fake_fp6(a, a_scale, b, b_scale, m, n, k, *args, **kwargs):
            del a_scale, b_scale, args, kwargs
            calls.append(("fp6", False, True, (m, k), (n, k)))
            return a @ b.T

        def fake_fp6_pack(x, block_size):
            del block_size
            return x, torch.ones(1)

        def fake_fp4_impl(
            a,
            a_scale,
            trans_a,
            b,
            b_scale,
            trans_b,
            **kwargs,
        ):
            calls.append(("fp4", trans_a, trans_b, tuple(a.shape), tuple(b.shape)))
            return (a.T if trans_a else a) @ (b.T if trans_b else b)

        def fake_quantize(x, *args, **kwargs):
            scale = torch.ones(1)
            return x, scale, x.T.contiguous(), scale

        class FakeBackend:
            def __init__(self, value):
                self.value = value

        class FakeScalingRecipe:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fp6_impl_module.gemm_fp6_impl = fake_fp6
        fp6_pack_module.check_mxfp6_support = lambda device: (True, "")
        fp6_pack_module.quantize_mxfp6_row = fake_fp6_pack
        backend_module.BackendType = types.SimpleNamespace(
            AITER=FakeBackend(1),
            HIPBLASLT=FakeBackend(2),
        )
        precision_module.ScalingRecipe = FakeScalingRecipe
        fp4_impl_module.gemm_fp4_impl = fake_fp4_impl
        quant_module.quantize_fp4_with_trans = fake_quantize
        for name in (
            "primus_turbo",
            "primus_turbo.pytorch",
            "primus_turbo.pytorch.core",
            "primus_turbo.pytorch.kernels",
            "primus_turbo.pytorch.kernels.gemm",
            "primus_turbo.pytorch.kernels.quantization",
            "primus_turbo.pytorch.ops",
        ):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, fp6_impl_module.__name__, fp6_impl_module)
        monkeypatch.setitem(sys.modules, fp6_pack_module.__name__, fp6_pack_module)
        monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
        monkeypatch.setitem(sys.modules, precision_module.__name__, precision_module)
        monkeypatch.setitem(sys.modules, fp4_impl_module.__name__, fp4_impl_module)
        monkeypatch.setitem(sys.modules, quant_module.__name__, quant_module)

        from primus.backends.automodel.quantization.mxfp6_gemm import (
            mxfp6_fwd_mxfp4_bwd,
        )

        x = torch.randn(257, 256, requires_grad=True)
        weight = torch.randn(768, 256, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        fp4_config = types.SimpleNamespace(
            use_preshuffle=True,
            use_gradient_sr=False,
            granularity=types.SimpleNamespace(value=3),
            block_size=32,
        )
        fp6_config = types.SimpleNamespace(
            block_size=32,
            granularity=types.SimpleNamespace(value=3),
        )
        out = mxfp6_fwd_mxfp4_bwd(x, weight, fp6_config, fp4_config)
        grad = torch.randn_like(out)
        out.backward(grad)

        out_ref = x_ref @ weight_ref.T
        out_ref.backward(grad)

        torch.testing.assert_close(out, out_ref, rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=1e-5, atol=2e-5)
        assert calls == [
            ("fp6", False, True, (512, 256), (768, 256)),
            ("fp4", False, True, (512, 768), (256, 768)),
            ("fp4", False, True, (768, 512), (256, 512)),
        ]


class TestPatchRegistration:
    @staticmethod
    def _patches():
        import primus.backends.automodel.patches  # noqa: F401
        from primus.core.patches.patch_registry import PatchRegistry

        return {
            patch.id: patch for patch in PatchRegistry.iter_patches(backend="automodel", phase="before_train")
        }

    def test_patch_is_registered_and_off(self):
        patch = self._patches()["automodel.quantization.mxfp6_linear"]
        assert patch.condition(None) is False

    def test_patch_turns_on_when_requested(self, set_option):
        set_option("primus_turbo.mxfp6_linear", True)
        assert self._patches()["automodel.quantization.mxfp6_linear"].condition(None) is True

    def test_only_mxfp6_patch_is_active_under_conflict(self, set_option):
        set_option("primus_turbo.mxfp6_linear", True)
        set_option("primus_turbo.mxfp4_linear", True)
        set_option("primus_turbo.fp8_linear", True)
        patches = self._patches()
        candidates = (
            "automodel.quantization.mxfp6_linear",
            "automodel.quantization.mxfp4_linear",
            "automodel.quantization.fp8_linear",
        )
        assert [patch_id for patch_id in candidates if patches[patch_id].condition(None)] == [
            "automodel.quantization.mxfp6_linear"
        ]

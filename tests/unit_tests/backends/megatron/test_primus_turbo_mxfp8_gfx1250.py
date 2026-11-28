###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MXFP8 on gfx1250 in the Primus-Turbo extension: TransformerEngine's autocast stays off (fp8_utils), and the
MXFP8 wgrad adds straight into main_grad (beta=1) where Primus-Turbo says its GEMM can. The decisions run on any
GPU with Primus-Turbo's answers stubbed; the fused wgrad itself is checked on gfx1250."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus_turbo.pytorch as primus_turbo_torch  # noqa: E402  isort:skip
from primus_turbo.pytorch.core.low_precision import check_mxfp8_support  # noqa: E402  isort:skip

import primus.backends.megatron.core.extensions.primus_turbo as primus_turbo_ext  # noqa: E402  isort:skip
import primus.backends.megatron.core.fp8_utils as fp8_utils  # noqa: E402  isort:skip
from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: E402  isort:skip
    Format,
    PrimusTurboLowPrecisionGlobalStateManager,
    PrimusTurboQuantConfig,
    PrimusTurboQuantizedTensorPair,
    ScaleDtype,
    ScalingGranularity,
    _bridge_weight_grad,
    _fuse_wgrad_accum_pattern,
    _maybe_create_quantized_weight_buffers,
    float8_e4m3,
)

FUSION = SimpleNamespace(gradient_accumulation_fusion=True)
MXFP8 = PrimusTurboQuantConfig(
    format=Format.E4M3,
    granularity=ScalingGranularity.MX_BLOCKWISE,
    block_size=32,
    scale_dtype=ScaleDtype.E8M0,
)


@pytest.mark.parametrize(
    "recipe, gfx1250, te_enabled",
    [("mxfp8", True, False), ("mxfp8", False, True), ("tensorwise", True, True)],
)
def test_te_autocast_off_for_mxfp8_on_gfx1250(monkeypatch, recipe, gfx1250, te_enabled):
    """TransformerEngine's MXFP8 recipe does not run on gfx1250: only Primus-Turbo quantizes there. Other archs and
    recipes keep TE's autocast on."""
    from megatron.core.enums import Fp8Recipe

    seen = {}
    monkeypatch.setattr(
        primus_turbo_ext, "primus_turbo_fp8_autocast", lambda **kwargs: seen.update(kwargs) or nullcontext()
    )
    monkeypatch.setattr(fp8_utils, "is_gfx1250", lambda: gfx1250)
    config = SimpleNamespace(
        fp8="e4m3",
        fp8_recipe=Fp8Recipe(recipe),
        fp8_param=False,
        fp8_wgrad=True,
        fp8_dot_product_attention=False,
        first_last_layers_bf16=False,
        num_layers=1,
        num_layers_at_start_in_bf16=0,
        num_layers_at_end_in_bf16=0,
        tp_only_amax_red=False,
    )
    fp8_utils.get_fp8_context(config)
    assert seen["enabled"] is te_enabled
    assert seen["enabled_turbo"] is True


@pytest.fixture
def turbo(monkeypatch):
    """The extension under the Turbo MXFP8 autocast on gfx1250, with a Primus-Turbo that says its MXFP8 wgrad
    accumulates; returns the questions it was asked. ``turbo.answer`` changes what it says."""
    manager = PrimusTurboLowPrecisionGlobalStateManager
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP4_ENABLED", False)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP8_ENABLED", True)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_QUANT_CONFIG", MXFP8)
    monkeypatch.setattr(primus_turbo_ext, "_is_gfx1250", lambda: True)
    asked = SimpleNamespace(calls=[], answer=True)

    def supported(grouped, main_grad):
        asked.calls.append((grouped, main_grad))
        return asked.answer

    monkeypatch.setattr(primus_turbo_torch.ops, "mx_wgrad_accum_supported", supported, raising=False)
    caches = (primus_turbo_ext._turbo_mx_wgrad_accum_supported, primus_turbo_ext._warn_mxfp8_wgrad_not_fused)
    for cache in caches:
        cache.cache_clear()
    yield asked
    for cache in caches:
        cache.cache_clear()


def _weight(shape=(8, 8), main_grad_dtype=torch.bfloat16, device="cpu"):
    weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16, device=device))
    weight.main_grad = torch.zeros(shape, dtype=main_grad_dtype, device=device)
    weight.grad_added_to_main_grad = False
    return weight


@pytest.mark.parametrize("flag, fusion", [(True, False), (False, True)])
def test_mxfp8_wgrad_follows_the_switches(turbo, monkeypatch, flag, fusion):
    """The experts' fusion follows turbo_grouped_gemm_wgrad_accum_fusion under MXFP8 as under BF16, the dense
    linears gradient_accumulation_fusion. Primus-Turbo is asked about the fused weight's own main_grad."""
    args = SimpleNamespace(turbo_grouped_gemm_wgrad_accum_fusion=flag)
    monkeypatch.setattr(primus_turbo_ext, "get_args", lambda: args)
    config = SimpleNamespace(gradient_accumulation_fusion=fusion)
    experts, dense = _weight((2, 8, 8)), _weight()
    assert _fuse_wgrad_accum_pattern(config, experts, grouped=True) == ("megatron" if flag else None)
    assert _fuse_wgrad_accum_pattern(config, dense) == ("megatron" if fusion else None)
    ((grouped, main_grad),) = turbo.calls  # the switched-off one was not asked
    assert grouped is flag and main_grad is (experts if flag else dense).main_grad


@pytest.mark.parametrize("case", ["other_arch", "turbo_says_no", "turbo_cannot_be_asked", "no_main_grad"])
def test_mxfp8_wgrad_keeps_the_separate_add(turbo, monkeypatch, case):
    weight = _weight()
    if case == "other_arch":  # gfx950's MXFP8 GEMMs have the epilogue too, but the fusion is gfx1250's
        monkeypatch.setattr(primus_turbo_ext, "_is_gfx1250", lambda: False)
    elif case == "turbo_says_no":
        turbo.answer = False
    elif case == "turbo_cannot_be_asked":
        monkeypatch.delattr(primus_turbo_torch.ops, "mx_wgrad_accum_supported")
    else:
        del weight.main_grad
    assert _fuse_wgrad_accum_pattern(FUSION, weight) is None
    assert len(turbo.calls) == (1 if case == "turbo_says_no" else 0)


def test_moe_permute_hands_off_under_the_mxfp8_autocast(turbo, monkeypatch):
    """The MoE permute + quantization fusion defers its gather to the experts' quantizer under the Turbo MXFP8
    autocast, as under MXFP4, and stays eager under any other FP8 recipe."""
    from primus.backends.megatron.patches.moe_patches.gather_quant_config import (
        _uses_mx,
    )

    assert _uses_mx() is True
    tensorwise = PrimusTurboQuantConfig()
    monkeypatch.setattr(PrimusTurboLowPrecisionGlobalStateManager, "PRIMUS_TURBO_QUANT_CONFIG", tensorwise)
    assert _uses_mx() is False


def _accumulate_mxfp8_wgrad(weight, inputs, grad_outputs, *, fused, group_lens=None):
    """Micro-batches through the cached quantized weight and _bridge_weight_grad, as the Turbo linears run them."""
    quantized_weight, quantized_weight_trans = _maybe_create_quantized_weight_buffers(
        weight, float8_e4m3, MXFP8, disable_parameter_transpose_cache=False
    )
    op = primus_turbo_torch.ops.gemm_fp8 if group_lens is None else primus_turbo_torch.ops.grouped_gemm_fp8
    for x, grad_output in zip(inputs, grad_outputs):
        bridged_x, bridged_weight = _bridge_weight_grad(
            x.detach().clone().requires_grad_(True),
            weight,
            PrimusTurboQuantizedTensorPair(data=quantized_weight, data_t=quantized_weight_trans),
            fuse_wgrad_accum=fused,
        )
        output = op(
            bridged_x,
            bridged_weight,
            *(() if group_lens is None else (group_lens,)),
            trans_b=True,
            config=MXFP8.data(),
            fuse_bgrad_accum_pattern="megatron" if fused else None,
        )
        output.backward(grad_output)
    return weight.main_grad


@pytest.mark.skipif(
    not check_mxfp8_support()[0] or not hasattr(primus_turbo_torch.ops, "mx_wgrad_accum_supported"),
    reason="needs MXFP8 and a Primus-Turbo that reports its MXFP8 wgrad accumulation",
)
@pytest.mark.parametrize("grouped", [False, True])
def test_mxfp8_fused_wgrad_matches_unfused(grouped, main_grad_dtype=torch.bfloat16):
    """The beta=1 GEMM against the framework's add, on top of an existing gradient, two micro-batches."""
    torch.manual_seed(42)
    if grouped:
        G, m, n, k = 4, 256, 512, 384
        shape, rows = (G, n, k), G * m
        group_lens = torch.full((G,), m, dtype=torch.int64, device="cuda")
    else:
        n, k, rows = 512, 384, 512
        shape, group_lens = (n, k), None
    if not primus_turbo_ext._mxfp8_wgrad_can_accumulate(_weight(shape, main_grad_dtype, "cuda"), grouped):
        pytest.skip("no MXFP8 beta=1 wgrad into this main_grad here")
    weight_data = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 16
    inputs = [torch.randn(rows, k, dtype=torch.bfloat16, device="cuda") / 16 for _ in range(2)]
    grad_outputs = [torch.randn(rows, n, dtype=torch.bfloat16, device="cuda") / 16 for _ in range(2)]
    initial_main_grad = torch.randn(shape, dtype=main_grad_dtype, device="cuda") / 16

    results = {}
    for fused in (True, False):
        weight = torch.nn.Parameter(weight_data.clone())
        weight.main_grad = initial_main_grad.clone()
        weight.grad_added_to_main_grad = False
        results[fused] = _accumulate_mxfp8_wgrad(
            weight, inputs, grad_outputs, fused=fused, group_lens=group_lens
        ).clone()
        assert weight.grad_added_to_main_grad

    assert not torch.equal(results[True], initial_main_grad)
    torch.testing.assert_close(results[True], results[False], rtol=5e-3, atol=5e-3)

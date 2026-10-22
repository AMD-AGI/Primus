# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The multi-micro-batch FP8 weight cache must match the layout Turbo's ops use."""

from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus.backends.megatron.core.extensions.primus_turbo as pt  # noqa: E402  isort:skip

_TURBO_PADS = pt._turbo_quantize_supports_padding()
TW = pt.ScalingGranularity.TENSORWISE
ALIGN = pt._TURBO_FP8_PAD_ALIGN

# GPT-OSS experts: hidden 2880 (not 128-aligned), intermediate 2880, GLU fc1 = 2 * 2880.
GPT_OSS_FC1 = (4, 5760, 2880)
GPT_OSS_FC2 = (4, 2880, 2880)
# DeepSeek-V2-Lite experts: hidden 2048, intermediate 1408 (both 128-aligned).
DSV2_LITE_FC1 = (8, 2816, 2048)
DSV2_LITE_FC2 = (8, 2048, 1408)


@pytest.fixture(autouse=True)
def _turbo_pads(monkeypatch):
    monkeypatch.setattr(pt, "_turbo_quantize_supports_padding", lambda: True)


@pytest.mark.parametrize(
    "op,role,shape,expected",
    [
        ("grouped_mlp", "fc1", GPT_OSS_FC1, {"pad_align_last": ALIGN}),
        ("grouped_mlp", "fc2", GPT_OSS_FC2, {"pad_align_last": ALIGN, "pad_align_penultimate": ALIGN}),
        # N = 5760 is aligned, so only K is padded.
        ("grouped_gemm", "b", GPT_OSS_FC1, {"pad_align_last": ALIGN}),
        ("grouped_gemm", "b", GPT_OSS_FC2, {"pad_align_last": ALIGN, "pad_align_penultimate": ALIGN}),
        ("grouped_mlp", "fc1", DSV2_LITE_FC1, {}),
        ("grouped_mlp", "fc2", DSV2_LITE_FC2, {}),
        ("grouped_gemm", "b", DSV2_LITE_FC2, {}),
    ],
)
def test_tensorwise_pads_only_unaligned_dims(op, role, shape, expected):
    assert pt._turbo_fp8_weight_pad(op, role, TW, shape) == expected


@pytest.mark.parametrize(
    "granularity",
    [
        pt.ScalingGranularity.MX_BLOCKWISE,
        pt.ScalingGranularity.BLOCKWISE,
        pt.ScalingGranularity.ROWWISE,
    ],
)
@pytest.mark.parametrize(
    "op,role,shape",
    [
        ("grouped_mlp", "fc1", GPT_OSS_FC1),
        ("grouped_mlp", "fc2", GPT_OSS_FC2),
        ("grouped_gemm", "b", GPT_OSS_FC2),
    ],
)
def test_non_tensorwise_cache_is_never_padded(granularity, op, role, shape):
    assert pt._turbo_fp8_weight_pad(op, role, granularity, shape) == {}


def test_turbo_without_padding_support_keeps_unpadded_cache(monkeypatch):
    monkeypatch.setattr(pt, "_turbo_quantize_supports_padding", lambda: False)
    assert pt._turbo_fp8_weight_pad("grouped_mlp", "fc2", TW, GPT_OSS_FC2) == {}


def _tensorwise_quant_config():
    return pt.PrimusTurboQuantConfig(granularity=TW)


def _mxfp8_quant_config():
    return pt.PrimusTurboQuantConfig(
        granularity=pt.ScalingGranularity.MX_BLOCKWISE, block_size=32, scale_dtype=pt.ScaleDtype.E8M0
    )


@pytest.mark.parametrize(
    "quant_config,role,shape,qdata_shape",
    [
        (_tensorwise_quant_config, "fc1", (2, 384, 192), (2, 384, 256)),
        (_tensorwise_quant_config, "fc2", (2, 192, 192), (2, 256, 256)),
        (_tensorwise_quant_config, "fc2", (2, 256, 128), (2, 256, 128)),
        # None: the cache create_quantized_weight builds (Turbo pads MXFP8 storage itself).
        pytest.param(
            _mxfp8_quant_config,
            "fc2",
            (2, 192, 192),
            None,
            marks=pytest.mark.skipif(not pt.check_mxfp8_support()[0], reason="Requires MXFP8 support"),
        ),
    ],
)
def test_prepare_weights_caches_in_the_ops_layout(monkeypatch, quant_config, role, shape, qdata_shape):
    _skip_without_padding_turbo()
    manager = pt.PrimusTurboLowPrecisionGlobalStateManager
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP8_ENABLED", True)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_FP4_ENABLED", False)
    monkeypatch.setattr(manager, "PRIMUS_TURBO_QUANT_CONFIG", quant_config())
    monkeypatch.setattr(pt, "get_num_microbatches", lambda: 4)
    linear = SimpleNamespace(
        weights=torch.nn.Parameter(torch.randn(shape, device="cuda", dtype=torch.bfloat16)),
        is_first_microbatch=True,
        config=SimpleNamespace(gradient_accumulation_fusion=False),
        disable_parameter_transpose_cache=True,
        _ensure_weight_views=lambda: None,
    )
    x = torch.randn(8, shape[-1], device="cuda", dtype=torch.bfloat16)

    pt.PrimusTurboGroupedLinear.prepare_weights(linear, x, "grouped_mlp", role)

    cached = linear.quantized_weight_buffer
    assert tuple(cached.shape) == shape
    if qdata_shape is None:
        reference, _ = pt.create_quantized_weight(
            linear.weights.detach(), pt.float8_e4m3, manager.PRIMUS_TURBO_QUANT_CONFIG.data()
        )
        assert cached.qdata.shape == reference.qdata.shape
        assert torch.equal(cached.qdata.view(torch.uint8), reference.qdata.view(torch.uint8))
    else:
        assert tuple(cached.qdata.shape) == qdata_shape


def _cached(weight, op, role, quant_config):
    pad = pt._turbo_fp8_weight_pad(op, role, quant_config.data().granularity, weight.shape)
    rowwise, _ = pt._maybe_create_quantized_weight_buffers(
        weight.detach(), pt.float8_e4m3, quant_config, disable_parameter_transpose_cache=True, weight_pad=pad
    )
    rowwise.requires_grad_(True)
    return rowwise


def _skip_without_padding_turbo():
    if not _TURBO_PADS:
        pytest.skip("installed primus_turbo does not pad FP8 weights")


@pytest.mark.parametrize("hidden", [192, 256])
def test_padded_cache_matches_self_quantized_grouped_mlp(hidden):
    _skip_without_padding_turbo()
    from primus_turbo.pytorch.ops.grouped_mlp_fp8 import grouped_mlp_fp8

    torch.manual_seed(0)
    experts, inter, tokens = 2, 192, 96
    quant_config = _tensorwise_quant_config()
    x = torch.randn(experts * tokens, hidden, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(experts * tokens, device="cuda", dtype=torch.float32)
    group_lens = torch.full((experts,), tokens, device="cuda", dtype=torch.int64)
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.05
    grad_out = torch.randn(experts * tokens, hidden, device="cuda", dtype=torch.bfloat16)

    def run(a, b):
        x_ = x.clone().requires_grad_(True)
        out = grouped_mlp_fp8(
            x_,
            a,
            b,
            group_lens,
            probs=probs,
            trans_w1=True,
            trans_w2=True,
            config=quant_config.data(),
            activation="silu",
        )
        out.backward(grad_out)
        return out, x_.grad, a.grad, b.grad

    ref = run(w1.clone().requires_grad_(True), w2.clone().requires_grad_(True))
    got = run(
        _cached(w1, "grouped_mlp", "fc1", quant_config),
        _cached(w2, "grouped_mlp", "fc2", quant_config),
    )
    for name, r, g in zip(("out", "grad_x", "grad_w1", "grad_w2"), ref, got):
        assert g.shape == r.shape, name
        torch.testing.assert_close(g, r, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("k,n", [(192, 320), (256, 256)])
def test_padded_cache_matches_self_quantized_grouped_gemm(k, n):
    _skip_without_padding_turbo()
    import primus_turbo.pytorch as turbo

    torch.manual_seed(0)
    experts, tokens = 2, 96
    quant_config = _tensorwise_quant_config()
    a = torch.randn(experts * tokens, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    group_lens = torch.full((experts,), tokens, device="cuda", dtype=torch.int64)
    grad_out = torch.randn(experts * tokens, n, device="cuda", dtype=torch.bfloat16)

    def run(weight):
        a_ = a.clone().requires_grad_(True)
        out = turbo.ops.grouped_gemm_fp8(a_, weight, group_lens, trans_b=True, config=quant_config.data())
        out.backward(grad_out)
        return out, a_.grad, weight.grad

    ref = run(b.clone().requires_grad_(True))
    got = run(_cached(b, "grouped_gemm", "b", quant_config))
    for name, r, g in zip(("out", "grad_a", "grad_b"), ref, got):
        assert g.shape == r.shape, name
        torch.testing.assert_close(g, r, rtol=0, atol=0, msg=name)

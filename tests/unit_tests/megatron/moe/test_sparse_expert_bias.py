###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("primus_turbo")
pytest.importorskip("megatron.core")
from primus.backends.megatron.core.extensions import mega_moe


@pytest.mark.parametrize(
    "enabled,training,supported,with_bias,uses_sparse",
    [
        (False, True, True, True, False),
        (True, False, True, True, False),
        (True, True, False, True, False),
        (True, True, True, False, False),
        (True, True, True, True, True),
    ],
)
def test_stage2_and_bias_run_once_in_parameter_gather_order(
    monkeypatch, enabled, training, supported, with_bias, uses_sparse
):
    trace = []
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    ids = torch.zeros(2, 4, dtype=torch.long)
    probs = torch.rand(2, 4)
    b1, b2 = object(), object() if with_bias else None
    group = object()

    def gather(name, bias):
        trace.append(name)
        return object(), bias

    def first(*args):
        trace.append("stage1")
        assert args[-1] is group
        return object(), object(), object(), object()

    def second(*args, fc1_bias, fc2_bias):
        trace.append("stage2")
        assert args[-2] is group and fc1_bias is b1
        assert fc2_bias is (None if uses_sparse else b2)
        return x

    def sparse(y, i, p, bias, ep):
        trace.append("bias")
        assert y is x and i is ids and p is probs and bias is b2 and ep is group
        return y + 1

    monkeypatch.setattr(mega_moe, "fused_mega_moe_fp8_stage1", first)
    monkeypatch.setattr(mega_moe, "fused_mega_moe_fp8_stage2", second)
    messages = []
    monkeypatch.setattr(mega_moe, "_log_once", lambda message: None)
    monkeypatch.setattr(mega_moe, "log_rank_all", messages.append)
    mega_moe._log_sparse_bias_fallback.cache_clear()
    layer = SimpleNamespace(
        ep_group=group,
        training=training,
        activation=object(),
        fc1_weight=lambda: gather("fc1", b1),
        fc2_weight=lambda: gather("fc2", b2),
        _sparse_bias_ops=(
            (lambda *a: None if supported and a[3] is not None else "runtime guard", sparse)
            if enabled
            else None
        ),
    )
    out = mega_moe.MegaMoEFP8Experts.forward(layer, x, ids, probs)
    assert torch.equal(out, x + 1 if uses_sparse else x)
    expected = ["fc1", "stage1", "fc2", "stage2"] + (["bias"] if uses_sparse else [])
    assert trace == expected
    mega_moe.MegaMoEFP8Experts.forward(layer, x, ids, probs)
    assert trace == expected * 2
    assert len(messages) == int(enabled and not uses_sparse)
    if messages:
        assert ("runtime guard" if training else "evaluation mode") in messages[0]
    mega_moe._log_sparse_bias_fallback.cache_clear()


def test_sparse_bias_constructor_selects_compatible_stage(monkeypatch):
    import megatron.training

    from primus.backends.megatron.core.extensions.sparse_expert_bias import (
        SparseExpertBias,
        sparse_expert_bias_fallback_reason,
    )

    cfg = mega_moe.TransformerConfig(
        num_layers=1,
        hidden_size=2880,
        num_attention_heads=8,
        num_moe_experts=128,
        expert_model_parallel_size=8,
        moe_ffn_hidden_size=2880,
        moe_router_topk=4,
        gated_linear_unit=True,
        activation_func=mega_moe.quick_gelu,
        glu_linear_offset=1.0,
        activation_func_clamp_value=7.0,
        bias_activation_fusion=False,
        add_bias_linear=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
        perform_initialization=False,
    )
    args = SimpleNamespace(turbo_mega_moe_sparse_fc2_bias=True, turbo_mega_moe_wgrad_accum_fusion=False)
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    # Exercise the actual constructor and both weight modules without allocating 760 MiB.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("meta"))
    messages = []
    monkeypatch.setattr(mega_moe, "log_rank_all", messages.append)
    mega_moe._log_sparse_bias_fallback.cache_clear()

    def build():
        return mega_moe.MegaMoEFP8Experts(cfg, 16, 2880, 2880, SimpleNamespace(size=lambda: 8))

    # Exercise the installed public ABI when present. Also qualify the negative path on older releases.
    bias_arg = inspect.signature(mega_moe.fused_mega_moe_fp8_stage2).parameters.get("fc2_bias")
    if (
        bias_arg is not None
        and bias_arg.default is None
        and bias_arg.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ):
        assert build()._sparse_bias_ops == (sparse_expert_bias_fallback_reason, SparseExpertBias.apply)
        assert not messages
    monkeypatch.setattr(mega_moe, "fused_mega_moe_fp8_stage2", lambda *args, fc2_bias: None)
    assert build()._sparse_bias_ops is None
    assert "optional keyword fc2_bias=None" in messages[-1]
    cfg.glu_linear_offset = 0.0
    assert build()._sparse_bias_ops is None
    assert "biased quick-GeGLU" in messages[-1]
    count = len(messages)
    args.turbo_mega_moe_sparse_fc2_bias = False
    assert build()._sparse_bias_ops is None
    assert len(messages) == count
    mega_moe._log_sparse_bias_fallback.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="requires ROCm")
def test_runtime_fallbacks():
    from primus.backends.megatron.core.extensions.sparse_expert_bias import (
        sparse_expert_bias_fallback_reason,
    )

    x = torch.empty(7, 2880, dtype=torch.bfloat16, device="cuda")
    p = torch.empty(7, 4, device="cuda")
    i = torch.zeros_like(p, dtype=torch.int64)
    bias = torch.empty(16, 2880, dtype=x.dtype, device=x.device)
    group = SimpleNamespace(size=lambda: 8)
    arch = getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(":")[0]
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type="cuda", enabled=False):
            reason = sparse_expert_bias_fallback_reason(x, i, p, bias, group)
            assert reason is None if arch == "gfx950" else reason == "requires gfx950"
    finally:
        torch.set_float32_matmul_precision(precision)
    assert sparse_expert_bias_fallback_reason(x, i.to(torch.int32), p, bias, group)
    assert sparse_expert_bias_fallback_reason(x, i, p.bfloat16(), bias, group)
    assert sparse_expert_bias_fallback_reason(x, i, p, None, group)
    assert sparse_expert_bias_fallback_reason(x, i, p, bias, SimpleNamespace(size=lambda: 4))
    assert sparse_expert_bias_fallback_reason(x, i[:, :2], p[:, :2], bias, group)
    assert sparse_expert_bias_fallback_reason(x[:0], i[:0], p[:0], bias, group)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        assert sparse_expert_bias_fallback_reason(x, i, p, bias, group)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="requires ROCm")
@pytest.mark.parametrize("exceptional", [False, True])
def test_real_sparse_bias_matches_dense_first_derivatives(exceptional):
    from primus.backends.megatron.core.extensions.sparse_expert_bias import (
        SparseExpertBias,
    )

    if (
        getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "gcnArchName", "").split(":")[
            0
        ]
        != "gfx950"
    ):
        pytest.skip("requires a qualified gfx950 device")
    if dist.is_initialized() and dist.get_world_size() != 1:
        pytest.skip("requires an isolated one-rank group")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        generator = torch.Generator(device="cuda").manual_seed(10942)
        tokens, hidden, experts = 73, 257, 8
        y = torch.randn(hidden, tokens, generator=generator, device="cuda", dtype=torch.bfloat16).T
        dy = torch.randn(hidden, tokens, generator=generator, device="cuda", dtype=torch.bfloat16).T
        bias = torch.randn(experts, hidden, generator=generator, device="cuda", dtype=y.dtype)
        scores = torch.randn(tokens, experts, generator=generator, device="cuda")
        ids = scores.topk(4).indices
        ids[::3, 1] = ids[::3, 0]
        probs = scores.gather(1, ids).softmax(1)
        # Both calls scatter independently. Equal duplicate weights make the
        # reference independent of which writer the device chooses.
        probs[::3, :2] = 0.625
        probs[::4, 2] = 0.0
        if exceptional:
            bias[0, 0], bias[1, 1], bias[2, 2] = float("nan"), float("inf"), -float("inf")
        y.requires_grad_()
        probs.requires_grad_()
        bias.requires_grad_()
        out = SparseExpertBias.apply(y, ids, probs, bias, dist.group.WORLD)
        grad_y, grad_p, grad_bias = torch.autograd.grad(out, (y, probs, bias), dy)
        with torch.no_grad():
            dense = torch.zeros_like(scores).scatter_(1, ids, probs)
            wanted_y = torch.addmm(y.float(), dense, bias.float()).to(y.dtype)
            wanted_p = (dy.float() @ bias.float().T).gather(1, ids)
            wanted_bias = (dense.T @ dy.float()).to(bias.dtype)
        torch.testing.assert_close(grad_y, dy, atol=0, rtol=0)
        torch.testing.assert_close(grad_bias, wanted_bias, atol=0, rtol=0)
        torch.testing.assert_close(grad_p, wanted_p, atol=5e-5, rtol=2e-5, equal_nan=True)
        torch.testing.assert_close(grad_p[::3, 0], grad_p[::3, 1], atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(out, wanted_y, atol=2e-5, rtol=0.008, equal_nan=True)
        keep = torch.isfinite(wanted_y)
        ref = wanted_y[keep].float()
        assert (out[keep].float() - ref).norm() <= 5e-4 * ref.norm()
    finally:
        torch.set_float32_matmul_precision(precision)
        if owns_group:
            dist.destroy_process_group()

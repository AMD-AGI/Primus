###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from primus.backends.megatron.patches.turbo import mega_ulysses_patches
from primus.backends.megatron.patches.turbo.mega_ulysses_patches import (
    _GQA_OPS,
    _MLA_OPS,
    _contiguous_cp_split,
    _input_norm,
    _is_turbo_mega_ulysses_can_patch,
    _missing_ops,
)
from primus.core.patches import PatchContext


def _params(**overrides):
    params = dict(
        use_turbo_mega_ulysses=True,
        turbo_mega_ulysses_impl="fused",
        tensor_model_parallel_size=1,
        context_parallel_size=4,
        normalization="RMSNorm",
        multi_latent_attention=False,
    )
    params.update(overrides)
    return SimpleNamespace(**params)


def _ctx(params):
    return PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=params)}
    )


@pytest.fixture
def eligible_env(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(mega_ulysses_patches, "is_primus_turbo_can_patch", lambda _ctx: True)
    monkeypatch.setattr(mega_ulysses_patches, "_missing_ops", lambda _args: None)


def test_enabled_when_eligible(eligible_env):
    assert _is_turbo_mega_ulysses_can_patch(_ctx(_params())) is True


def test_mla_enabled_when_eligible(eligible_env):
    params = _params(multi_latent_attention=True, q_lora_rank=1536, turbo_mega_ulysses_impl="per_head")
    assert _is_turbo_mega_ulysses_can_patch(_ctx(params)) is True


def test_disabled_without_flag(eligible_env):
    assert _is_turbo_mega_ulysses_can_patch(_ctx(_params(use_turbo_mega_ulysses=False))) is False


def test_disabled_without_turbo(eligible_env, monkeypatch):
    monkeypatch.setattr(mega_ulysses_patches, "is_primus_turbo_can_patch", lambda _ctx: False)
    assert _is_turbo_mega_ulysses_can_patch(_ctx(_params())) is False


@pytest.mark.parametrize(
    "overrides",
    [
        dict(tensor_model_parallel_size=2),
        dict(context_parallel_size=1),
        dict(context_parallel_size=16),
        dict(context_parallel_size=3),
        dict(add_qkv_bias=True),
        dict(add_bias_linear=True),
        dict(attention_output_gate=True),
        dict(window_size=(128, 0)),
        dict(attention_dropout=0.1),
        dict(experimental_attention_variant="dsa"),
        dict(use_turbo_fused_qk_rmsnorm_rope=True),
        dict(normalization="L2Norm"),
        dict(multi_latent_attention=True, q_lora_rank=None),
        dict(multi_latent_attention=True, q_lora_rank=1536, cache_mla_latents=True),
        dict(multi_latent_attention=True, q_lora_rank=1536, recompute_modules=["mla_up_proj"]),
    ],
)
def test_unsupported_settings_keep_stock_path(eligible_env, overrides):
    assert _is_turbo_mega_ulysses_can_patch(_ctx(_params(**overrides))) is False


@pytest.mark.parametrize(
    "mla, missing, expected",
    [
        (False, set(), True),
        (False, {_GQA_OPS}, False),
        (True, set(), True),
        (True, {_MLA_OPS}, False),
    ],
)
def test_missing_turbo_ops_keep_stock_path(eligible_env, monkeypatch, mla, missing, expected):
    monkeypatch.setattr(mega_ulysses_patches, "_missing_ops", _missing_ops)
    monkeypatch.setattr(
        mega_ulysses_patches.importlib.util,
        "find_spec",
        lambda name: None if name in missing else SimpleNamespace(name=name),
    )
    params = _params(multi_latent_attention=mla, q_lora_rank=1536 if mla else None)
    assert _is_turbo_mega_ulysses_can_patch(_ctx(params)) is expected


@pytest.mark.parametrize("mla, impl", [(False, "per_head"), (False, "unfused"), (True, "unfused")])
def test_invalid_impl_raises(eligible_env, mla, impl):
    params = _params(multi_latent_attention=mla, q_lora_rank=1536, turbo_mega_ulysses_impl=impl)
    with pytest.raises(AssertionError, match="turbo_mega_ulysses_impl"):
        _is_turbo_mega_ulysses_can_patch(_ctx(params))


@pytest.mark.parametrize("cp_rank", [0, 1, 3])
def test_contiguous_cp_split(monkeypatch, cp_rank):
    cp, s, b = 4, 16, 2
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: cp)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group: cp_rank)
    tokens = torch.arange(b * s).view(b, s)
    mask = torch.ones(b, 1, s, s)
    cu_seqlens = torch.tensor([0, s])
    batch = {
        "tokens": tokens,
        "labels": tokens + 1,
        "attention_mask": mask,
        "cu_seqlens": cu_seqlens,
        "x": None,
    }

    out = _contiguous_cp_split(batch, cp_group=None)

    chunk = slice(cp_rank * s // cp, (cp_rank + 1) * s // cp)
    torch.testing.assert_close(out["tokens"], tokens[:, chunk])
    torch.testing.assert_close(out["labels"], tokens[:, chunk] + 1)
    assert out["attention_mask"].shape == (b, 1, s // cp, s)
    assert out["cu_seqlens"] is cu_seqlens
    assert out["x"] is None


def _norm_linear(normalization, zero_centered_gamma=False, skip=False):
    h = 8
    linear = SimpleNamespace(
        layer_norm_weight=torch.rand(h) + 0.5,
        layer_norm_bias=torch.rand(h),
        normalization=normalization,
        zero_centered_gamma=zero_centered_gamma,
        eps=1e-6,
    )
    if skip:
        linear._skip_fused_norm = True
    return linear


@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_input_norm_rmsnorm(zero_centered_gamma):
    linear = _norm_linear("RMSNorm", zero_centered_gamma)
    x = torch.randn(4, 2, 8)
    weight = linear.layer_norm_weight + (1 if zero_centered_gamma else 0)
    torch.testing.assert_close(_input_norm(linear, x), F.rms_norm(x, (8,), weight, 1e-6))


def test_input_norm_layernorm():
    linear = _norm_linear("LayerNorm")
    x = torch.randn(4, 2, 8)
    expected = F.layer_norm(x, (8,), linear.layer_norm_weight, linear.layer_norm_bias, 1e-6)
    torch.testing.assert_close(_input_norm(linear, x), expected)


def test_input_norm_passthrough():
    x = torch.randn(4, 2, 8)
    assert _input_norm(SimpleNamespace(), x) is x
    assert _input_norm(_norm_linear("RMSNorm", skip=True), x) is x

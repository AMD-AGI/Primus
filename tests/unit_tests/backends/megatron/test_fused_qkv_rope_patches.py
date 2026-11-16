###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches.turbo import fused_qkv_rope_patches as mod
from primus.core.patches import PatchContext


def _install_fakes(monkeypatch, shape_error=None):
    """Fake Megatron's attention module and Primus-Turbo's RoPE op and shape check."""
    calls = []

    def te_fn(qkv, q_freqs, k_freqs, split, **kwargs):
        calls.append(("te", kwargs))
        return "te"

    def fused_qkv_rope(qkv, q_freqs, k_freqs, split):
        calls.append(("flydsl", split))
        return "flydsl"

    modules = {}
    for name in (
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.attention",
        "primus_turbo",
        "primus_turbo.pytorch",
        "primus_turbo.pytorch.ops",
        "primus_turbo.pytorch.ops.rope",
        "primus_turbo.pytorch.kernels",
        "primus_turbo.pytorch.kernels.rope",
        "primus_turbo.pytorch.kernels.rope.rope_impl",
    ):
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, modules[name])
    attention = modules["megatron.core.transformer.attention"]
    attention.apply_fused_qkv_rotary_pos_emb = te_fn
    attention.HAVE_FUSED_QKV_ROPE = False
    modules["primus_turbo.pytorch.ops.rope"].fused_qkv_rope = fused_qkv_rope
    modules["primus_turbo.pytorch.kernels.rope.rope_impl"].rope_shape_error = lambda *a: shape_error
    warnings = []
    monkeypatch.setattr(mod, "warning_rank_0", warnings.append)
    return attention, calls, warnings


def _args(freqs_dtype=torch.float32):
    freqs = torch.zeros(4, 1, 1, 128, dtype=freqs_dtype)
    return torch.zeros(4, 2, 1, 384, dtype=torch.bfloat16), freqs, freqs, [128, 128, 128]


@pytest.mark.parametrize(
    ("env", "can_patch", "expected"),
    [("1", True, True), ("true", True, True), ("0", True, False), ("1", False, False)],
)
def test_enabled_needs_env_and_turbo(monkeypatch, env, can_patch, expected):
    monkeypatch.setenv("PRIMUS_TURBO_FLYDSL_ROPE", env)
    monkeypatch.setattr(mod, "is_primus_turbo_can_patch", lambda _ctx: can_patch)
    assert mod._enabled(PatchContext(backend="megatron", phase="before_train")) is expected


def test_installs_once_and_routes_llama_shape_to_flydsl(monkeypatch):
    attention, calls, warnings = _install_fakes(monkeypatch)
    mod.patch_fused_qkv_rope(SimpleNamespace())
    patched = attention.apply_fused_qkv_rotary_pos_emb
    assert attention.HAVE_FUSED_QKV_ROPE
    mod.patch_fused_qkv_rope(SimpleNamespace())
    assert attention.apply_fused_qkv_rotary_pos_emb is patched

    assert patched(*_args()) == "flydsl"
    assert calls == [("flydsl", [128, 128, 128])] and not warnings


@pytest.mark.parametrize(
    ("args", "kwargs", "shape_error"),
    [
        (_args(torch.bfloat16), {}, None),
        (_args(), {"interleaved": True}, None),
        (_args(), {"cp_size": 2}, None),
        (_args(), {"cu_seqlens": torch.zeros(2)}, None),
        (_args(), {}, "qkv must be bfloat16"),
    ],
    ids=["bf16_freqs", "interleaved", "context_parallel", "cu_seqlens", "shape_error"],
)
def test_unsupported_calls_fall_back_to_te(monkeypatch, args, kwargs, shape_error):
    attention, calls, warnings = _install_fakes(monkeypatch, shape_error=shape_error)
    mod.patch_fused_qkv_rope(SimpleNamespace())
    patched = attention.apply_fused_qkv_rotary_pos_emb

    assert patched(*args, **kwargs) == "te"
    assert patched(*args, **kwargs) == "te"
    assert [c[0] for c in calls] == ["te", "te"]
    assert all(calls[0][1][k] is v or calls[0][1][k] == v for k, v in kwargs.items())
    assert len(warnings) == 1


def test_unsupported_call_without_te_raises(monkeypatch):
    attention, _, _ = _install_fakes(monkeypatch, shape_error="qkv must be bfloat16")
    attention.apply_fused_qkv_rotary_pos_emb = None
    mod.patch_fused_qkv_rope(SimpleNamespace())
    with pytest.raises(RuntimeError, match="TE has none"):
        attention.apply_fused_qkv_rotary_pos_emb(*_args())

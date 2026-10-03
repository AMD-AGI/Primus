###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the MLA decoupled-RoPE QK-Norm patch.

Verifies:
  1. The source anchor still matches upstream ``get_query_key_value_tensors``.
  2. The patch rewrites the method body (on the defining base class) and is idempotent.
  3. The enable condition honors the flag/env var and the fused-RoPE guard.
"""

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")

from megatron.core.transformer.multi_latent_attention import MLASelfAttention

import primus.backends.megatron.patches.mla_rope_qk_norm_patches as patch_mod
from primus.backends.megatron.patches._patch_guard import is_patched
from primus.core.patches.context import PatchContext

_METHOD = "get_query_key_value_tensors"


def _defining_cls():
    return next(c for c in MLASelfAttention.__mro__ if _METHOD in c.__dict__)


@pytest.fixture
def pristine_method():
    """Restore the patched method and patch-guard state after each test."""
    cls = _defining_cls()
    original = cls.__dict__[_METHOD]
    yield
    setattr(cls, _METHOD, original)
    if hasattr(cls, "_primus_applied_patch_keys"):
        cls._primus_applied_patch_keys.discard(patch_mod._PATCH_KEY)


def test_anchor_matches_upstream():
    source = inspect.getsource(getattr(_defining_cls(), _METHOD))
    assert (
        patch_mod._ORI in source
    ), "Upstream get_query_key_value_tensors changed; update mla_rope_qk_norm anchor."


def test_new_differs_and_normalizes_rotary():
    assert patch_mod._NEW != patch_mod._ORI
    assert "torch.rsqrt" in patch_mod._NEW
    assert "q_pos_emb" in patch_mod._NEW and "k_pos_emb" in patch_mod._NEW


def test_install_rewrites_and_is_idempotent(pristine_method, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    cls = _defining_cls()
    original = cls.__dict__[_METHOD]

    patch_mod._install_mla_rope_qk_norm_patch()
    assert is_patched(cls, patch_mod._PATCH_KEY)
    first = cls.__dict__[_METHOD]
    assert first is not original

    patch_mod._install_mla_rope_qk_norm_patch()  # idempotent
    assert cls.__dict__[_METHOD] is first


def _ctx(**args):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**args))},
    )


@pytest.mark.parametrize(
    "env, flag, fusion, expected",
    [
        ("1", False, False, True),  # env var enables
        (None, True, False, True),  # config flag enables
        (None, False, False, False),  # off by default
        ("1", False, True, False),  # fused RoPE -> skip (guarded)
    ],
)
def test_enabled_condition(env, flag, fusion, expected, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    if env is None:
        monkeypatch.delenv("PRIMUS_MLA_ROPE_QK_NORM", raising=False)
    else:
        monkeypatch.setenv("PRIMUS_MLA_ROPE_QK_NORM", env)
    ctx = _ctx(mla_rope_qk_norm=flag, apply_rope_fusion=fusion)
    assert patch_mod._enabled(ctx) is expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

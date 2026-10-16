###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the MLA decoupled-RoPE QK-Norm patch.

Verifies:
  1. The source anchor still matches upstream ``get_query_key_value_tensors``.
  2. The patch rewrites the method body (on the defining base class) and is idempotent.
  3. The enable condition honors the flag/env var and the fused-RoPE,
     cached-latent and NoPE-MLA guards.
  4. An install failure aborts the run instead of silently training without QK-Norm.
  5. On a real (tiny) MLA module, the patched method feeds unit-RMS q/k into
     the rotary embedding and leaves the content q/k and the value unchanged.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")

import megatron.core.transformer.multi_latent_attention as mla_mod
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
    "env, args, expected",
    [
        ("1", {}, True),  # env var enables
        (None, {"mla_rope_qk_norm": True}, True),  # config flag enables
        (None, {}, False),  # off by default
        ("1", {"apply_rope_fusion": True}, False),  # fused RoPE -> skip
        ("1", {"cache_mla_latents": True}, False),  # cached latents -> skip
        ("1", {"mla_use_nope": True}, False),  # NoPE MLA -> skip
        ("1", {"model_type": "kimi_k3"}, False),  # K3 defaults to NoPE -> skip
        ("1", {"model_type": "kimi_k3", "mla_use_nope": False}, True),  # K3 with RoPE
        ("1", {"model_type": "gpt", "mla_use_nope": False}, True),
    ],
)
def test_enabled_condition(env, args, expected, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    if env is None:
        monkeypatch.delenv("PRIMUS_MLA_ROPE_QK_NORM", raising=False)
    else:
        monkeypatch.setenv("PRIMUS_MLA_ROPE_QK_NORM", env)
    ctx = _ctx(**{"mla_rope_qk_norm": False, "apply_rope_fusion": False, **args})
    assert patch_mod._enabled(ctx) is expected


def test_install_failure_is_fatal(pristine_method, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    monkeypatch.setattr(patch_mod, "_ORI", "# anchor that is not in the upstream source\n")
    with pytest.raises(SystemExit, match="QK-Norm patch installation failed"):
        patch_mod.patch_mla_rope_qk_norm(_ctx(mla_rope_qk_norm=True))
    assert not is_patched(_defining_cls(), patch_mod._PATCH_KEY)


class _UnusedCoreAttention(torch.nn.Module):
    """Stand-in core attention: the test never runs attention itself."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def _tiny_mla():
    import dataclasses

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    config = MLATransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        q_lora_rank=None,
        kv_lora_rank=32,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        apply_rope_fusion=False,
        normalization="RMSNorm",
        params_dtype=torch.float32,
    )
    spec = get_gpt_layer_local_spec(multi_latent_attention=True).submodules.self_attention
    submodules = dataclasses.replace(spec.submodules, core_attention=_UnusedCoreAttention)
    attn = MLASelfAttention(config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal)
    with torch.no_grad():
        # Inflate the projections so the unpatched rotary q/k are far from unit RMS.
        attn.linear_q_proj.weight.mul_(20)
        attn.linear_kv_down_proj.weight.mul_(20)
    return attn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MLA builds its rotary embedding on the GPU")
def test_patched_forward_normalizes_only_rotary(init_parallel_state, pristine_method, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    torch.manual_seed(0)
    attn = _tiny_mla()
    device = attn.linear_q_proj.weight.device
    hidden = torch.randn(6, 2, attn.config.hidden_size, device=device)
    d_nope = attn.config.qk_head_dim

    rotary_inputs = []
    real_rope = mla_mod.apply_rotary_pos_emb

    def recording_rope(t, *args, **kwargs):
        rotary_inputs.append(t.detach().float())
        return real_rope(t, *args, **kwargs)

    monkeypatch.setattr(mla_mod, "apply_rotary_pos_emb", recording_rope)

    with torch.no_grad():
        q0, k0, v0 = attn.get_query_key_value_tensors(hidden, None)[:3]
        before = list(rotary_inputs)
        rotary_inputs.clear()
        patch_mod._install_mla_rope_qk_norm_patch()
        q1, k1, v1 = attn.get_query_key_value_tensors(hidden, None)[:3]

    # q_pos_emb and k_pos_emb each pass through the rotary embedding once.
    assert len(before) == 2 and len(rotary_inputs) == 2
    for t in before:
        assert not torch.allclose(t.pow(2).mean(-1).sqrt(), torch.ones(1, device=device), atol=0.1)
    for t in rotary_inputs:
        rms = t.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=0)

    torch.testing.assert_close(q1[..., :d_nope], q0[..., :d_nope], atol=0, rtol=0)
    torch.testing.assert_close(k1[..., :d_nope], k0[..., :d_nope], atol=0, rtol=0)
    torch.testing.assert_close(v1, v0, atol=0, rtol=0)
    assert not torch.allclose(q1[..., d_nope:], q0[..., d_nope:])
    assert not torch.allclose(k1[..., d_nope:], k0[..., d_nope:])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the FLA fused-SwiGLU MLP patch.

Verifies:
  1. The source anchors still match upstream ``MLP.__init__``/``MLP.forward``.
  2. The patch rewrites both methods in one pass and is idempotent.
  3. ``_plain_swiglu_reject`` accepts only plain, unclamped, zero-offset SwiGLU.
  4. ``_fused_fc2_reject`` refuses every path that ``linear_fc2`` would own
     (TP/SP/PP, bias, FP8, deferred wgrad, overlapped param gather, experts).
  5. ``_resolve_fla_swiglu`` reports a rejected opt-in instead of silently
     falling back to the unfused path.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch.nn.functional as F

pytest.importorskip("megatron")

from megatron.core.transformer.mlp import MLP

import primus.backends.megatron.patches.mlp_fla_swiglu_patches as patch_mod
from primus.backends.megatron.patches._patch_guard import is_patched


def _config(**overrides):
    """A config the fused fc2 path accepts; override one field per test."""
    config = SimpleNamespace(
        gated_linear_unit=True,
        activation_func=F.silu,
        use_te_activation_func=False,
        activation_func_clamp_value=None,
        glu_linear_offset=0,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        add_bias_linear=False,
        pipeline_model_parallel_size=1,
        fp8=None,
        delay_wgrad_compute=False,
        activation_func_fp8_input_store=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _args(**overrides):
    args = SimpleNamespace(
        use_fla_fused_swiglu=True,
        use_fla_fused_swiglu_linear=False,
        overlap_param_gather=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def pristine_mlp():
    """Restore ``MLP`` methods and patch-guard state after each test."""
    original_init, original_forward = MLP.__init__, MLP.forward
    yield
    MLP.__init__, MLP.forward = original_init, original_forward
    if hasattr(MLP, "_primus_applied_patch_keys"):
        MLP._primus_applied_patch_keys.discard(patch_mod._PATCH_KEY)


@pytest.fixture(autouse=True)
def _reset_decision_log():
    patch_mod._DECISIONS_LOGGED.clear()
    yield
    patch_mod._DECISIONS_LOGGED.clear()


def test_init_anchor_matches_upstream():
    source = inspect.getsource(MLP.__init__)
    assert (
        patch_mod._INIT_ORI in source
    ), "Upstream MLP.__init__ changed; update mlp_fla_swiglu_patches anchor."


def test_forward_anchors_match_upstream():
    source = inspect.getsource(MLP.forward)
    for anchor in (patch_mod._FWD_FUSED_ORI, patch_mod._FORWARD_ORI):
        assert anchor in source, f"Upstream MLP.forward changed; anchor missing: {anchor!r}"


def test_fused_forward_rechecks_lora_wrapper():
    # PEFT swaps linear_fc2 for a wrapper after __init__, so the __init__-time
    # decision cannot be trusted at call time.
    assert "hasattr(self.linear_fc2, 'weight')" in patch_mod._FWD_FUSED_NEW


def test_install_patch_rewrites_both_methods(pristine_mlp, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    original_init, original_forward = MLP.__init__, MLP.forward

    patch_mod._install_mlp_fla_swiglu_patch()

    assert is_patched(MLP, patch_mod._PATCH_KEY)
    assert MLP.__init__ is not original_init
    assert MLP.forward is not original_forward


def test_install_patch_is_idempotent(pristine_mlp, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)

    patch_mod._install_mlp_fla_swiglu_patch()
    first_init, first_forward = MLP.__init__, MLP.forward
    patch_mod._install_mlp_fla_swiglu_patch()

    assert MLP.__init__ is first_init
    assert MLP.forward is first_forward


def test_plain_swiglu_accepts_plain_config():
    assert patch_mod._plain_swiglu_reject(_config()) == ""


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"gated_linear_unit": False}, "gated_linear_unit is off"),
        ({"activation_func": F.gelu}, "activation_func is not F.silu"),
        # Kimi K3 keeps activation_func=F.silu but routes the real, clamped
        # activation through the TE module slot.
        ({"use_te_activation_func": True}, "TE module slot"),
        ({"activation_func_clamp_value": 7.0}, "activation_func_clamp_value is set"),
        ({"glu_linear_offset": 1}, "glu_linear_offset is non-zero"),
    ],
)
def test_plain_swiglu_rejects(overrides, expected):
    reason = patch_mod._plain_swiglu_reject(_config(**overrides))
    assert expected in reason


def test_fused_fc2_accepts_plain_config():
    assert patch_mod._fused_fc2_reject(_config(), _args(), is_expert=False) == ""


@pytest.mark.parametrize(
    "config_overrides, args_overrides, is_expert, expected",
    [
        ({}, {}, True, "expert MLP"),
        ({"tensor_model_parallel_size": 2}, {}, False, "tensor_model_parallel_size > 1"),
        ({"sequence_parallel": True}, {}, False, "sequence_parallel is on"),
        ({"add_bias_linear": True}, {}, False, "add_bias_linear is on"),
        ({"pipeline_model_parallel_size": 2}, {}, False, "pipeline_model_parallel_size > 1"),
        ({"fp8": "e4m3"}, {}, False, "fp8 is enabled"),
        ({"delay_wgrad_compute": True}, {}, False, "delay_wgrad_compute is on"),
        (
            {"activation_func_fp8_input_store": True},
            {},
            False,
            "activation_func_fp8_input_store is on",
        ),
        ({}, {"overlap_param_gather": True}, False, "overlap_param_gather is on"),
        # A config the plain kernel already refuses must not reach the fc2 path.
        ({"use_te_activation_func": True}, {}, False, "TE module slot"),
    ],
)
def test_fused_fc2_rejects(config_overrides, args_overrides, is_expert, expected):
    reason = patch_mod._fused_fc2_reject(_config(**config_overrides), _args(**args_overrides), is_expert)
    assert expected in reason


def test_resolve_declines_fused_fc2_when_not_opted_in(monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    monkeypatch.setattr("megatron.training.get_args", _args, raising=False)

    _, swiglu_linear_fn = patch_mod._resolve_fla_swiglu(_config(), is_expert=False)

    assert swiglu_linear_fn is None


def test_resolve_reports_rejected_opt_in(monkeypatch):
    messages = []
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda msg, *a, **k: messages.append(msg))
    monkeypatch.setattr(
        "megatron.training.get_args",
        lambda: _args(use_fla_fused_swiglu_linear=True),
        raising=False,
    )

    _, swiglu_linear_fn = patch_mod._resolve_fla_swiglu(
        _config(tensor_model_parallel_size=2), is_expert=False
    )

    assert swiglu_linear_fn is None
    assert any("disabled: tensor_model_parallel_size > 1" in m for m in messages)


def test_resolve_declines_plain_swiglu_for_clamped_activation(monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    monkeypatch.setattr("megatron.training.get_args", _args, raising=False)

    swiglu_fn, _ = patch_mod._resolve_fla_swiglu(_config(activation_func_clamp_value=7.0), is_expert=False)

    assert swiglu_fn is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

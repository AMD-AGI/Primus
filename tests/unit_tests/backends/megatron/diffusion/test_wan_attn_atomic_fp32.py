# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Wan's attention-backward kernel selection.

The ``attn_atomic_fp32`` model key exists because both attention backends
re-read their fp32-atomic setting from the environment on every call, and
both arrive at the trainer already forced to "0" -- Primus' launchers export
Primus-Turbo's variable that way, and the release image pins TE's. Reading
either variable back therefore cannot distinguish a deliberate opt-out from
that blanket default, so the trainer writes both from the YAML key instead.

These tests pin that plumbing, which is otherwise only observable by reading
kernel names out of a profile.
"""

import os

import pytest


def _trainer_module():
    """Import lazily so collection does not depend on the trainer's imports."""
    from primus.backends.megatron import wan_pretrain_trainer

    return wan_pretrain_trainer


@pytest.fixture(autouse=True)
def isolate_attention_env(monkeypatch):
    """Give every variable the helper writes a restore point for teardown."""
    module = _trainer_module()
    for name in (*module.ATTN_ATOMIC_FP32_ENVS, module.TE_CK_BWD_V3_ENV):
        monkeypatch.delenv(name, raising=False)


def test_both_backend_variables_are_covered():
    """One key has to reach Primus-Turbo and TransformerEngine alike.

    A run uses the local spec or te_spec, not both, so the trainer sets both
    variables rather than working out which backend is live. Dropping either
    name silently leaves that spec on ck_tile.
    """
    module = _trainer_module()

    assert set(module.ATTN_ATOMIC_FP32_ENVS) == {
        "PRIMUS_TURBO_ATTN_V3_ATOMIC_FP32",
        "NVTE_CK_IS_V3_ATOMIC_FP32",
    }


def test_enabling_sets_both_backend_variables():
    module = _trainer_module()

    module._apply_atomic_fp32_backward(True)

    for name in module.ATTN_ATOMIC_FP32_ENVS:
        assert os.environ[name] == "1", name


def test_enabling_forces_the_te_v3_backward():
    """TE only reaches a v3 backward with this on, so enabling pins it.

    Without it, a launcher that disabled v3 would quietly undo the atomics.
    """
    module = _trainer_module()

    module._apply_atomic_fp32_backward(True)

    assert os.environ[module.TE_CK_BWD_V3_ENV] == "1"


def test_enabling_overrides_the_launcher_blanket_default():
    """The pre-set "0" is exactly the state this helper exists to correct."""
    module = _trainer_module()
    for name in module.ATTN_ATOMIC_FP32_ENVS:
        os.environ[name] = "0"

    module._apply_atomic_fp32_backward(True)

    for name in module.ATTN_ATOMIC_FP32_ENVS:
        assert os.environ[name] == "1", name


def test_disabling_sets_both_backend_variables_to_zero():
    module = _trainer_module()

    module._apply_atomic_fp32_backward(False)

    for name in module.ATTN_ATOMIC_FP32_ENVS:
        assert os.environ[name] == "0", name


def test_disabling_leaves_the_te_v3_setting_alone():
    """Opting out of the atomics is not a request to disable v3 as well."""
    module = _trainer_module()
    os.environ[module.TE_CK_BWD_V3_ENV] = "1"

    module._apply_atomic_fp32_backward(False)

    assert os.environ[module.TE_CK_BWD_V3_ENV] == "1"


@pytest.mark.parametrize("enabled", [True, False])
def test_helper_is_idempotent(enabled):
    module = _trainer_module()

    module._apply_atomic_fp32_backward(enabled)
    first = {name: os.environ[name] for name in module.ATTN_ATOMIC_FP32_ENVS}
    module._apply_atomic_fp32_backward(enabled)

    assert {name: os.environ[name] for name in module.ATTN_ATOMIC_FP32_ENVS} == first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

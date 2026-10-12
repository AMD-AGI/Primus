###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the declarative backend env registry.

The focus is the ``XLA_FLAGS`` layering contract. It gets its own tests because it
is the only variable that packs many independent settings into one string, so it
cannot use plain ``setdefault`` semantics:

    XLA_FLAGS_APPEND  >  per-config ``env:``  >  backend defaults  >  inherited
"""

from __future__ import annotations

import os

import pytest

from primus.core.backend.env_registry import (
    MODE_XLA_APPEND,
    XLA_FLAGS_APPEND,
    EnvVar,
    append_xla_flags,
    apply_env_defaults,
    apply_xla_flags_append,
    clear_config_owned,
    mark_config_owned,
)

# An image-baked value carrying the autotune level that NaNs fp8 MoE runs, plus a
# flag Primus does not manage (which must survive).
BAKED = "--xla_gpu_autotune_level=0 --xla_gpu_baked_only=7"
MANAGED = "--xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true"


def _effective(flags: str, name: str) -> str | None:
    """Value XLA would use for ``name``, i.e. the last occurrence."""
    value = None
    for token in flags.split():
        if token.split("=", 1)[0] == name:
            value = token.split("=", 1)[1] if "=" in token else ""
    return value


def _managed_entry() -> EnvVar:
    return EnvVar("XLA_FLAGS", MANAGED, mode=MODE_XLA_APPEND)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv(XLA_FLAGS_APPEND, raising=False)
    clear_config_owned()
    yield
    clear_config_owned()


# ------------------------------ append_xla_flags ------------------------------


def test_append_puts_addition_last():
    assert append_xla_flags("--a=1", "--b=2") == "--a=1 --b=2"


@pytest.mark.parametrize(
    "existing, addition, expected",
    [
        ("", "--b=2", "--b=2"),
        ("--a=1", "", "--a=1"),
        ("", "", ""),
        ("  --a=1  ", " --b=2 ", "--a=1 --b=2"),
    ],
)
def test_append_handles_empty_and_padded_operands(existing, addition, expected):
    assert append_xla_flags(existing, addition) == expected


def test_append_preserves_values_containing_spaces():
    """The superseded per-flag merge had to assume no value ever contained a space."""
    out = append_xla_flags("--xla_dump_hlo_pipeline_re='(?i) gpu'", "--b=2")
    assert out.endswith("--b=2")
    assert "'(?i) gpu'" in out


# -------------------- backend defaults vs inherited value ---------------------


def test_managed_defaults_override_baked_autotune(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", BAKED)

    applied = apply_env_defaults([_managed_entry()], "maxtext")

    assert applied == ["XLA_FLAGS"]
    flags = os.environ["XLA_FLAGS"]
    assert _effective(flags, "--xla_gpu_autotune_level") == "4"
    assert _effective(flags, "--xla_gpu_baked_only") == "7", "unmanaged baked flags must survive"


def test_managed_defaults_apply_when_nothing_inherited():
    apply_env_defaults([_managed_entry()], "maxtext")

    assert _effective(os.environ["XLA_FLAGS"], "--xla_gpu_autotune_level") == "4"


# ---------------------- per-config env: owns the variable ---------------------


def test_config_owned_xla_flags_is_not_overridden(monkeypatch):
    """Regression: managed defaults used to silently beat the config's own value."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=5")
    mark_config_owned("XLA_FLAGS")

    applied = apply_env_defaults([_managed_entry()], "maxtext")

    assert applied == []
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_autotune_level=5"


def test_ownership_of_other_vars_does_not_protect_xla_flags(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", BAKED)
    mark_config_owned("XLA_PYTHON_CLIENT_MEM_FRACTION")

    apply_env_defaults([_managed_entry()], "maxtext")

    assert _effective(os.environ["XLA_FLAGS"], "--xla_gpu_autotune_level") == "4"


# ------------------------------ XLA_FLAGS_APPEND ------------------------------


def test_flags_append_beats_managed_defaults(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", BAKED)
    monkeypatch.setenv(XLA_FLAGS_APPEND, "--xla_gpu_autotune_level=5")

    apply_env_defaults([_managed_entry()], "maxtext")
    assert apply_xla_flags_append() is True

    flags = os.environ["XLA_FLAGS"]
    assert _effective(flags, "--xla_gpu_autotune_level") == "5"
    assert (
        _effective(flags, "--xla_gpu_enable_latency_hiding_scheduler") == "true"
    ), "managed knobs the user did not override must remain"


def test_flags_append_applies_without_any_backend_defaults(monkeypatch):
    """MaxDiffusion declares no defaults, but must still honour the override."""
    monkeypatch.setenv("XLA_FLAGS", BAKED)
    monkeypatch.setenv(XLA_FLAGS_APPEND, "--xla_gpu_autotune_level=5")

    assert apply_env_defaults([], "maxdiffusion") == []
    assert apply_xla_flags_append() is True
    assert _effective(os.environ["XLA_FLAGS"], "--xla_gpu_autotune_level") == "5"


def test_flags_append_is_consumed_so_repeat_calls_are_noops(monkeypatch):
    monkeypatch.setenv(XLA_FLAGS_APPEND, "--b=2")

    assert apply_xla_flags_append() is True
    first = os.environ["XLA_FLAGS"]

    assert apply_xla_flags_append() is False
    assert os.environ["XLA_FLAGS"] == first


def test_flags_append_absent_is_noop():
    assert apply_xla_flags_append() is False
    assert "XLA_FLAGS" not in os.environ


# ---------------------------- end-to-end layering -----------------------------


def test_full_layering_matches_documented_precedence(monkeypatch):
    """Replay the real pipeline: image ENV, config env:, adapter, then append."""
    monkeypatch.setenv("XLA_FLAGS", BAKED)  # 1) image-baked

    # 2) TrainRuntime._apply_config_env
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=5 --xla_gpu_from_config=1")
    mark_config_owned("XLA_FLAGS")

    # 3) BackendAdapter.prepare_backend
    apply_env_defaults([_managed_entry()], "maxtext")

    # 4) final override layer
    monkeypatch.setenv(XLA_FLAGS_APPEND, "--xla_gpu_autotune_level=6")
    apply_xla_flags_append()

    flags = os.environ["XLA_FLAGS"]
    assert _effective(flags, "--xla_gpu_autotune_level") == "6"
    assert _effective(flags, "--xla_gpu_from_config") == "1"


# ------------------------ unchanged setdefault behaviour ----------------------


def test_setdefault_entry_respects_existing_value(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.96")

    applied = apply_env_defaults([EnvVar("XLA_PYTHON_CLIENT_MEM_FRACTION", ".97")], "maxtext")

    assert applied == []
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.96"


def test_setdefault_entry_fills_when_unset(monkeypatch):
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)

    applied = apply_env_defaults([EnvVar("XLA_PYTHON_CLIENT_MEM_FRACTION", ".97")], "maxtext")

    assert applied == ["XLA_PYTHON_CLIENT_MEM_FRACTION"]
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == ".97"


def test_empty_entries_is_noop():
    assert apply_env_defaults([], "megatron") == []

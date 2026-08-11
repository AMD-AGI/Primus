###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the aiter RTLD_DEEPBIND gating logic
(``primus/backends/megatron/patches/turbo/aiter_deepbind_patches.py``).

These verify the patch self-detects whether the installed transformer_engine
already fixed the stale-vendored-libmha issue (ROCm/aiter#1332), without
needing a GPU. The actual RTLD_DEEPBIND import hook installation is exercised
separately in the diffusion GPU tests via ``tests.utils.install_aiter_deepbind_hook``.
"""

import sys
import types

import pytest

import primus.backends.megatron.patches.turbo.aiter_deepbind_patches as patch_mod


def _fake_te_module(version: str) -> types.ModuleType:
    module = types.ModuleType("transformer_engine")
    module.__version__ = version
    return module


@pytest.mark.parametrize(
    "version, expected",
    [
        ("2.14.0", (2, 14, 0)),
        ("2.14.1", (2, 14, 1)),
        ("1.11.0+abcdef0", (1, 11, 0)),
        ("2.14.0.dev0+rocm", (2, 14, 0)),
    ],
)
def test_te_version_tuple_parses_leading_semver(monkeypatch, version, expected):
    monkeypatch.setitem(sys.modules, "transformer_engine", _fake_te_module(version))
    assert patch_mod._te_version_tuple() == expected


def test_te_version_tuple_none_when_unimportable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformer_engine", None)  # forces ImportError on import
    assert patch_mod._te_version_tuple() is None


def test_te_version_tuple_none_when_unparseable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformer_engine", _fake_te_module("not-a-version"))
    assert patch_mod._te_version_tuple() is None


@pytest.mark.parametrize(
    "version, expected_fixed",
    [
        ("2.13.9", False),  # pre-fix: still vendors the stale libmha_bwd.so
        ("2.14.0", True),  # fixed: renamed to te_libmha_*.so / qola::te::
        ("2.15.0", True),
        ("1.11.0+abcdef0", False),  # e.g. rocm/primus:v26.3's TE
    ],
)
def test_te_already_fixed_thresholds_on_fix_version(monkeypatch, version, expected_fixed):
    monkeypatch.setitem(sys.modules, "transformer_engine", _fake_te_module(version))
    assert patch_mod._te_already_fixed() is expected_fixed


def test_te_already_fixed_fails_safe_towards_unfixed_when_unknown(monkeypatch):
    # An unparseable/missing version must NOT be treated as "fixed": skipping the
    # hook on an actually-unfixed TE reintroduces the hd128 backward crash.
    monkeypatch.setitem(sys.modules, "transformer_engine", None)
    assert patch_mod._te_already_fixed() is False


def _condition_ctx(monkeypatch, *, use_turbo_attention=True, turbo_can_patch=True, affected_arch=True):
    args = types.SimpleNamespace(use_turbo_attention=use_turbo_attention)
    monkeypatch.setattr(patch_mod, "get_args", lambda ctx: args)
    monkeypatch.setattr(patch_mod, "is_primus_turbo_can_patch", lambda ctx: turbo_can_patch)
    monkeypatch.setattr(patch_mod, "_is_affected_arch", lambda: affected_arch)
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    return object()  # PatchContext stand-in; unused by the stubbed collaborators above


def test_can_install_skips_when_te_already_fixed(monkeypatch):
    ctx = _condition_ctx(monkeypatch)
    monkeypatch.setitem(sys.modules, "transformer_engine", _fake_te_module("2.14.0"))
    assert patch_mod._can_install_aiter_deepbind(ctx) is False


def test_can_install_applies_when_te_unfixed_on_affected_arch(monkeypatch):
    ctx = _condition_ctx(monkeypatch)
    monkeypatch.setitem(sys.modules, "transformer_engine", _fake_te_module("1.11.0+abcdef0"))
    assert patch_mod._can_install_aiter_deepbind(ctx) is True


def test_can_install_skips_when_arch_unaffected_regardless_of_te(monkeypatch):
    ctx = _condition_ctx(monkeypatch, affected_arch=False)
    monkeypatch.setitem(sys.modules, "transformer_engine", _fake_te_module("1.11.0+abcdef0"))
    assert patch_mod._can_install_aiter_deepbind(ctx) is False


def test_can_install_skips_when_turbo_attention_disabled(monkeypatch):
    ctx = _condition_ctx(monkeypatch, use_turbo_attention=False)
    monkeypatch.setitem(sys.modules, "transformer_engine", _fake_te_module("1.11.0+abcdef0"))
    assert patch_mod._can_install_aiter_deepbind(ctx) is False

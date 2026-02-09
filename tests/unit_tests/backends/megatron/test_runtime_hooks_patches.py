###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

import types

import pytest

from primus.core.patches.context import PatchContext


def _install_fake_megatron_initialize(monkeypatch: pytest.MonkeyPatch):
    import sys

    megatron_mod = types.ModuleType("megatron")
    training_pkg = types.ModuleType("megatron.training")
    init_mod = types.ModuleType("megatron.training.initialize")

    def _orig_compile():
        return "orig"

    init_mod._compile_dependencies = _orig_compile

    training_pkg.initialize = init_mod
    megatron_mod.training = training_pkg

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.training"] = training_pkg
    sys.modules["megatron.training.initialize"] = init_mod
    return init_mod, _orig_compile


def test_patch_skip_compile_dependencies_is_idempotent(monkeypatch: pytest.MonkeyPatch):
    init_mod, orig = _install_fake_megatron_initialize(monkeypatch)

    # Avoid real logging during unit tests.
    monkeypatch.setattr(
        "primus.backends.megatron.patches.runtime_hooks_patches.log_rank_0",
        lambda *a, **k: None,
    )

    from primus.backends.megatron.patches.runtime_hooks_patches import (
        patch_skip_compile_dependencies,
    )

    ctx = PatchContext(backend="megatron", phase="before_train")

    patch_skip_compile_dependencies(ctx)
    assert init_mod._compile_dependencies is not orig
    patched = init_mod._compile_dependencies
    assert getattr(patched, "_primus_patched", False) is True

    # Apply again should keep the same function.
    patch_skip_compile_dependencies(ctx)
    assert init_mod._compile_dependencies is patched

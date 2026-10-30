###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""megatron.args.checkpoint_path: where a non-null save goes, and the keep_explicit_save_path opt-out."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.args import checkpoint_path_patches
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry


def _ctx(save, keep=None, exp_root="/exp"):
    params = SimpleNamespace(save=save)
    if keep is not None:
        params.keep_explicit_save_path = keep
    return PatchContext(
        backend="megatron",
        phase="build_args",
        extra={
            "module_config": SimpleNamespace(params=params),
            "backend_args": SimpleNamespace(save=save),
            "primus_config": SimpleNamespace(exp_root_path=exp_root),
        },
    )


_PATCHES = {
    "megatron.args.checkpoint_path": checkpoint_path_patches.patch_checkpoint_path,
    "megatron.args.checkpoint_path_kept": checkpoint_path_patches.patch_keep_explicit_save_path,
}


def _run(ctx):
    """Run the registered patches as run_patches does: each handler only when its condition holds."""
    for patch_id, handler in _PATCHES.items():
        patch = PatchRegistry.get(patch_id)
        assert patch is not None and patch.handler is handler
        if patch.condition is None or patch.condition(ctx):
            patch.handler(ctx)
    return ctx.extra["backend_args"].save


@pytest.mark.parametrize("keep", [None, False, "false", "0", "False", 0, ""])
def test_default_routes_save_to_exp_root(keep):
    assert _run(_ctx("/tmp/ckpt", keep)) == "/exp/checkpoints"


@pytest.mark.parametrize("keep", [None, True])
def test_null_save_stays_null(keep):
    assert _run(_ctx(None, keep)) is None


@pytest.mark.parametrize("keep", [True, "true", "1", 1])
def test_keep_explicit_save_path_uses_save_as_given(keep):
    assert _run(_ctx("/tmp/ckpt", keep=keep)) == "/tmp/ckpt"


def test_keep_explicit_save_path_makes_a_relative_save_absolute_and_logs_it(monkeypatch, tmp_path):
    logged = []
    monkeypatch.setattr(checkpoint_path_patches, "log_rank_0", logged.append)
    monkeypatch.chdir(tmp_path)
    assert _run(_ctx("out/ckpt", keep=True)) == str(tmp_path / "out" / "ckpt")
    assert any("keep_explicit_save_path: true" in m and str(tmp_path / "out" / "ckpt") in m for m in logged)


@pytest.mark.parametrize("keep", ["maybe", 2, 0.5])
def test_keep_explicit_save_path_refuses_a_non_boolean(keep):
    with pytest.raises(ValueError, match="true or false"):
        _run(_ctx("/tmp/ckpt", keep=keep))


def test_condition_has_no_side_effects(monkeypatch):
    logged = []
    monkeypatch.setattr(checkpoint_path_patches, "log_rank_0", logged.append)
    assert checkpoint_path_patches._routes_save(_ctx(None, keep=True)) is False
    assert logged == []

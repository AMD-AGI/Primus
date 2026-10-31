###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""megatron.args.checkpoint_path and its model-independent keep_absolute_save_path switch."""

from types import SimpleNamespace

import pytest


def test_generic_checkpoint_path_keeps_an_absolute_save_when_asked():
    """megatron.args.checkpoint_path moves ``save`` to ``<exp_root>/checkpoints``; with the
    model-independent switch keep_absolute_save_path and an absolute ``save`` the patch does not run
    (its condition), so a run with ``load`` at the same path finds it. A relative save, or the
    switch off or unset (the same unset values as log_reset_first_iteration), keeps the Primus
    location. The switch on with a save that is not absolute is refused. No model-specific save patch exists.
    """
    from primus.backends.megatron.patches.args import checkpoint_path_patches as cp
    from primus.core.patches.patch_registry import PatchRegistry

    def run(save, model_type="gpt", **params):
        backend = SimpleNamespace(save=save)
        mc = SimpleNamespace(params=SimpleNamespace(model_type=model_type, save=save, **params))
        ctx = SimpleNamespace(
            extra={
                "backend_args": backend,
                "module_config": mc,
                "primus_config": SimpleNamespace(exp_root_path="/exp"),
            }
        )
        if cp.routes_save(ctx):
            cp.patch_checkpoint_path(ctx)
        return backend.save

    assert run("/tmp/ckpt") == "/exp/checkpoints"  # unchanged default, every model
    assert run("/tmp/ckpt", model_type="olmo3") == "/exp/checkpoints"
    for model_type in ("gpt", "olmo3"):
        assert run("/tmp/ckpt", model_type, keep_absolute_save_path=True) == "/tmp/ckpt"
        assert run("/tmp/ckpt", model_type, keep_absolute_save_path="off") == "/exp/checkpoints"
    for unset in (None, "", "null", "~"):
        assert run("/tmp/ckpt", keep_absolute_save_path=unset) == "/exp/checkpoints"
    for relative in ("ckpt", "~/ckpt"):  # the opt-in cannot be honoured: refused, not rerouted
        assert run(relative) == "/exp/checkpoints"
        with pytest.raises(ValueError, match="not an absolute path"):
            run(relative, keep_absolute_save_path=True)
    from pathlib import Path

    assert run(Path("/tmp/ckpt"), keep_absolute_save_path=True) == Path("/tmp/ckpt")
    assert run(None, keep_absolute_save_path=True) is None  # save: null still opts out of saving
    assert run(None) is None
    for save in (None, "/tmp/ckpt", "ckpt"):  # an unreadable switch is refused whatever the save
        with pytest.raises(ValueError, match="keep_absolute_save_path"):
            run(save, keep_absolute_save_path="bogus")
    # backend_args.save None is the opt-out even if the module param names an absolute path: the
    # condition does not skip the patch, so its opt-out message runs and nothing is saved
    backend = SimpleNamespace(save=None)
    ctx = SimpleNamespace(
        extra={
            "backend_args": backend,
            "module_config": SimpleNamespace(
                params=SimpleNamespace(save="/ckpt", keep_absolute_save_path=True)
            ),
            "primus_config": SimpleNamespace(exp_root_path="/exp"),
        }
    )
    assert cp.routes_save(ctx)
    cp.patch_checkpoint_path(ctx)
    assert backend.save is None
    # an unreadable switch value: the condition does not raise (patch selection goes on); the
    # patch body refuses it
    with pytest.raises(ValueError, match="keep_absolute_save_path"):
        run("/tmp/ckpt", keep_absolute_save_path="maybe")
    assert PatchRegistry.get("megatron.olmo3.save_path") is None


def test_routed_save_hint_only_for_an_absolute_save(monkeypatch):
    """The WARN suggests keep_absolute_save_path only for an absolute save; with a relative save the
    switch would refuse the run, so no hint."""
    from primus.backends.megatron.patches.args import checkpoint_path_patches as cp

    logs = []
    monkeypatch.setattr(cp, "log_rank_0", logs.append)

    def warn(save):
        logs.clear()
        ctx = SimpleNamespace(
            extra={
                "backend_args": SimpleNamespace(save=save),
                "module_config": SimpleNamespace(params=SimpleNamespace(save=save)),
                "primus_config": SimpleNamespace(exp_root_path="/exp"),
            }
        )
        cp.patch_checkpoint_path(ctx)
        return next(m for m in logs if "[WARN]" in m)

    assert "keep_absolute_save_path: true" in warn("/tmp/ckpt")
    assert "keep_absolute_save_path" not in warn("ckpt")


def test_save_switch_and_reset_switch_read_unset_alike():
    """keep_absolute_save_path and log_reset_first_iteration use one unset rule (is_unset)."""
    from primus.backends.megatron.patches.args.checkpoint_path_patches import (
        keep_absolute_save,
    )
    from primus.backends.megatron.patches.training_log import (
        first_iteration_reset_patches as fr,
    )

    for value in (None, "", "null", "~", True, "true", False, "off"):
        ctx = SimpleNamespace(
            extra={
                "module_config": SimpleNamespace(
                    params=SimpleNamespace(keep_absolute_save_path=value, log_reset_first_iteration=value)
                )
            }
        )
        assert keep_absolute_save(ctx) is fr.reset_enabled(ctx), value

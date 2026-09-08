###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the indexer-distillation-loss logging patch.

The patch reaches into ``track_moe_metrics``, which every MoE model calls, so
the tests that matter are the negative ones: it must not install for anything
but a V4 run that has the loss switched on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.deepseek_v4_indexer_loss_patches import (
    _indexer_distill_enabled,
    _make_tracked_with_indexer_loss,
)

_LOSS_NAME = "indexer_distill_loss"


class _Ctx:
    """Stand-in for ``PatchContext``; only ``get_args`` reads from it."""

    def __init__(self, **args):
        self.args = SimpleNamespace(**args)


@pytest.fixture(autouse=True)
def _args_from_ctx(monkeypatch):
    """Point the patch module's ``get_args`` at the fake context."""
    import primus.backends.megatron.patches.deepseek_v4_indexer_loss_patches as mod

    monkeypatch.setattr(mod, "get_args", lambda ctx: ctx.args)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, expected, why",
    [
        ({"model_type": "deepseek_v4", "v4_indexer_distill_loss_coeff": 1e-2}, True, "V4, loss on"),
        ({"model_type": "deepseek_v4", "v4_indexer_distill_loss_coeff": "1e-2"}, True, "coeff as str"),
        ({"model_type": "deepseek_v4", "v4_indexer_distill_loss_coeff": 0.0}, False, "V4, loss off"),
        ({"model_type": "deepseek_v4", "v4_indexer_distill_loss_coeff": None}, False, "coeff unset"),
        ({"model_type": "deepseek_v4"}, False, "no coeff attribute"),
        ({"model_type": "gpt", "v4_indexer_distill_loss_coeff": 1e-2}, False, "not V4"),
        ({"model_type": "deepseek_v3", "v4_indexer_distill_loss_coeff": 1e-2}, False, "V3, not V4"),
        ({}, False, "no model_type at all"),
        ({"model_type": "deepseek_v4", "v4_indexer_distill_loss_coeff": "not-a-number"}, False, "junk"),
    ],
)
def test_patch_only_installs_for_v4_with_the_loss_on(args, expected, why):
    assert _indexer_distill_enabled(_Ctx(**args)) is expected, why


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


def _recorder():
    seen = {}

    def fake_track_moe_metrics(*args, track_names=None, **kwargs):
        seen["args"] = args
        seen["track_names"] = track_names
        seen["kwargs"] = kwargs
        return "sentinel"

    return fake_track_moe_metrics, seen


def test_wrapper_appends_the_key_to_an_explicit_list():
    fake, seen = _recorder()
    wrapped = _make_tracked_with_indexer_loss(fake)

    assert wrapped(0.5, 7, track_names=["load_balancing_loss"]) == "sentinel"
    assert seen["track_names"] == ["load_balancing_loss", _LOSS_NAME]
    # Positional and keyword arguments must pass through untouched.
    assert seen["args"] == (0.5, 7)


def test_wrapper_does_not_duplicate_the_key():
    fake, seen = _recorder()
    wrapped = _make_tracked_with_indexer_loss(fake)

    wrapped(track_names=["z_loss", _LOSS_NAME])
    assert seen["track_names"].count(_LOSS_NAME) == 1


def test_wrapper_leaves_none_alone():
    """``None`` already means every key in the tracker, ours included."""
    fake, seen = _recorder()
    wrapped = _make_tracked_with_indexer_loss(fake)

    wrapped(track_names=None)
    assert seen["track_names"] is None


def test_wrapper_does_not_mutate_the_caller_list():
    """``training_log`` reuses its list; appending in place would grow it."""
    fake, _ = _recorder()
    wrapped = _make_tracked_with_indexer_loss(fake)

    caller_list = ["load_balancing_loss"]
    wrapped(track_names=caller_list)
    wrapped(track_names=caller_list)
    assert caller_list == ["load_balancing_loss"]


def test_wrapper_is_idempotent():
    """Re-running the patch must not stack wrappers."""
    fake, _ = _recorder()
    once = _make_tracked_with_indexer_loss(fake)
    assert getattr(once, "_v4_indexer_loss_patched", False) is True


def test_patch_skips_when_already_installed(monkeypatch):
    """A second install leaves the first wrapper in place."""
    import megatron.training.training as training_module

    import primus.backends.megatron.patches.deepseek_v4_indexer_loss_patches as mod

    fake, _ = _recorder()
    already = _make_tracked_with_indexer_loss(fake)
    monkeypatch.setattr(training_module, "track_moe_metrics", already, raising=False)

    mod.patch_indexer_distill_loss_logging(_Ctx(model_type="deepseek_v4"))
    assert training_module.track_moe_metrics is already

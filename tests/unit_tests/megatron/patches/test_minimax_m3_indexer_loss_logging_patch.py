###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the MiniMax-M3 MSA indexer-loss logging patch.

The patch reaches into ``track_moe_metrics``, which every MoE model calls, so
the gating tests matter most: it must install only for an MSA run with the loss
on -- including when the preset leaves the coefficient to the config default.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import primus.backends.megatron.patches.minimax_m3_indexer_loss_patches as mod

_LOSS_NAME = "msa_indexer_loss"


class _Ctx:
    """Stand-in for ``PatchContext``; only ``get_args`` reads from it."""

    def __init__(self, **args):
        self.args = SimpleNamespace(**args)


@pytest.fixture(autouse=True)
def _args_from_ctx(monkeypatch):
    monkeypatch.setattr(mod, "get_args", lambda ctx: ctx.args)
    # The real default lives on MSATransformerConfig, which needs Megatron.
    monkeypatch.setattr(mod, "_default_coeff", lambda: 1.0e-2)


@pytest.mark.parametrize(
    "args, expected, why",
    [
        ({"minimax_sparse_attention": True, "sparse_indexer_loss_coeff": 1e-2}, True, "MSA, loss on"),
        ({"minimax_sparse_attention": True, "sparse_indexer_loss_coeff": "1e-2"}, True, "coeff as str"),
        ({"minimax_sparse_attention": True}, True, "coeff unset -> config default"),
        ({"minimax_sparse_attention": True, "sparse_indexer_loss_coeff": None}, True, "coeff None"),
        ({"minimax_sparse_attention": True, "sparse_indexer_loss_coeff": 0.0}, False, "MSA, loss off"),
        ({"minimax_sparse_attention": False, "sparse_indexer_loss_coeff": 1e-2}, False, "MSA off"),
        ({"sparse_indexer_loss_coeff": 1e-2}, False, "not an MSA model"),
        ({}, False, "nothing set"),
        ({"minimax_sparse_attention": True, "sparse_indexer_loss_coeff": "junk"}, False, "junk"),
    ],
)
def test_patch_only_installs_for_msa_with_the_loss_on(args, expected, why):
    assert mod._indexer_loss_enabled(_Ctx(**args)) is expected, why


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
    wrapped = mod._make_tracked_with_indexer_loss(fake)

    assert wrapped(0.5, 7, track_names=["load_balancing_loss"], force_initialize=True) == "sentinel"
    assert seen["track_names"] == ["load_balancing_loss", _LOSS_NAME]
    assert seen["args"] == (0.5, 7)
    assert seen["kwargs"] == {"force_initialize": True}


def test_wrapper_does_not_duplicate_the_key():
    fake, seen = _recorder()
    wrapped = mod._make_tracked_with_indexer_loss(fake)

    wrapped(track_names=["z_loss", _LOSS_NAME])
    assert seen["track_names"].count(_LOSS_NAME) == 1


def test_wrapper_leaves_none_alone():
    fake, seen = _recorder()
    wrapped = mod._make_tracked_with_indexer_loss(fake)

    wrapped(track_names=None)
    assert seen["track_names"] is None


def test_wrapper_does_not_mutate_the_caller_list():
    fake, _ = _recorder()
    wrapped = mod._make_tracked_with_indexer_loss(fake)

    caller_list = ["load_balancing_loss"]
    wrapped(track_names=caller_list)
    wrapped(track_names=caller_list)
    assert caller_list == ["load_balancing_loss"]


def test_loss_name_matches_the_attention_module():
    pytest.importorskip("megatron")
    from primus.backends.megatron.core.transformer.minimax_m3.indexer_loss import (
        MSA_INDEXER_LOSS_NAME,
    )

    assert mod._LOSS_NAME == MSA_INDEXER_LOSS_NAME


def test_patch_skips_when_already_installed(monkeypatch):
    pytest.importorskip("megatron")
    import megatron.training.training as training_module

    fake, _ = _recorder()
    already = mod._make_tracked_with_indexer_loss(fake)
    monkeypatch.setattr(training_module, "track_moe_metrics", already, raising=False)

    mod.patch_minimax_m3_indexer_loss_logging(_Ctx(minimax_sparse_attention=True))
    assert training_module.track_moe_metrics is already

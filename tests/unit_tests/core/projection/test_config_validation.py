###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for projection config guards (``core/projection/config_validation.py``).

These guards reject combinations that produce misleading memory projections —
chiefly activation recomputation combined with split-wgrad pipeline schedules
(zero-bubble / ZBV family), whose deferred W nodes pin recomputed linear
inputs.  The module is pure logic (no GPU), so we test each predicate plus the
``check_*`` / ``assert_*`` entry points directly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("primus.core.projection.config_validation")

from primus.core.projection.config_validation import (  # noqa: E402
    SPLIT_WGRAD_PIPELINE_SCHEDULES,
    assert_recompute_pipeline_compat,
    check_recompute_pipeline_compat,
    recompute_is_enabled,
    resolve_pipeline_schedule,
    uses_split_wgrad_schedule,
)

# ─────────────────────────────────────────────────────────────────────────────
# resolve_pipeline_schedule
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_pipeline_schedule_cli_overrides_yaml():
    # An explicit (non-auto) CLI algorithm wins over the YAML pp_algorithm.
    assert (
        resolve_pipeline_schedule(
            pipeline_schedule_algorithm="zbv-formatted",
            pp_algorithm="zero-bubble",
        )
        == "zbv-formatted"
    )


def test_resolve_pipeline_schedule_falls_back_to_pp_algorithm():
    assert resolve_pipeline_schedule(pipeline_schedule_algorithm="auto", pp_algorithm="v-half") == "v-half"


def test_resolve_pipeline_schedule_auto_with_zero_bubble():
    # auto + enable_zero_bubble resolves to the zerobubble schedule.
    assert resolve_pipeline_schedule(enable_zero_bubble=True) == "zerobubble"


def test_resolve_pipeline_schedule_defaults_to_auto():
    assert resolve_pipeline_schedule() == "auto"


# ─────────────────────────────────────────────────────────────────────────────
# uses_split_wgrad_schedule
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sched", sorted(SPLIT_WGRAD_PIPELINE_SCHEDULES))
def test_uses_split_wgrad_schedule_true_for_known(sched):
    assert uses_split_wgrad_schedule(sched) is True


def test_uses_split_wgrad_schedule_false_for_inline():
    assert uses_split_wgrad_schedule("1f1b") is False
    assert uses_split_wgrad_schedule("auto") is False


def test_uses_split_wgrad_schedule_auto_with_zero_bubble():
    assert uses_split_wgrad_schedule("auto", enable_zero_bubble=True) is True


# ─────────────────────────────────────────────────────────────────────────────
# recompute_is_enabled
# ─────────────────────────────────────────────────────────────────────────────


def test_recompute_is_enabled_via_granularity():
    assert recompute_is_enabled(SimpleNamespace(recompute_granularity="full")) is True
    assert recompute_is_enabled(SimpleNamespace(recompute_granularity="selective")) is True


def test_recompute_is_enabled_via_num_layers_or_ids():
    assert recompute_is_enabled(SimpleNamespace(recompute_num_layers=4)) is True
    assert recompute_is_enabled(SimpleNamespace(recompute_layer_ids=[0, 3, 7])) is True


def test_recompute_is_disabled_when_all_off():
    mp = SimpleNamespace(
        recompute_granularity=None,
        recompute_num_layers=0,
        recompute_layer_ids=None,
    )
    assert recompute_is_enabled(mp) is False


def test_recompute_is_enabled_falls_back_to_module_cfg():
    mp = SimpleNamespace(recompute_granularity=None, recompute_num_layers=0, recompute_layer_ids=None)
    module_cfg = SimpleNamespace(recompute_granularity="full")
    assert recompute_is_enabled(mp, module_cfg=module_cfg) is True


# ─────────────────────────────────────────────────────────────────────────────
# check_recompute_pipeline_compat
# ─────────────────────────────────────────────────────────────────────────────


def test_check_compat_ok_when_recompute_off():
    ok, reason = check_recompute_pipeline_compat(
        recompute_granularity=None,
        recompute_num_layers=0,
        pipeline_schedule="zbv-formatted",  # split-wgrad but recompute is off
    )
    assert ok is True
    assert reason == ""


def test_check_compat_ok_with_inline_schedule():
    ok, reason = check_recompute_pipeline_compat(
        recompute_granularity="full",
        recompute_num_layers=4,
        pipeline_schedule="1f1b",
    )
    assert ok is True
    assert reason == ""


def test_check_compat_rejects_recompute_with_split_wgrad():
    ok, reason = check_recompute_pipeline_compat(
        recompute_granularity="full",
        recompute_num_layers=4,
        pipeline_schedule="zbv-formatted",
    )
    assert ok is False
    assert "split-wgrad" in reason


def test_check_compat_rejects_recompute_with_auto_zero_bubble():
    ok, reason = check_recompute_pipeline_compat(
        recompute_granularity="selective",
        pipeline_schedule="auto",
        enable_zero_bubble=True,
    )
    assert ok is False
    assert "split-wgrad" in reason


def test_check_compat_triggers_on_layer_ids():
    ok, reason = check_recompute_pipeline_compat(
        recompute_layer_ids=[0, 1, 2],
        pipeline_schedule="zerobubble",
    )
    assert ok is False
    assert "split-wgrad" in reason


# ─────────────────────────────────────────────────────────────────────────────
# assert_recompute_pipeline_compat
# ─────────────────────────────────────────────────────────────────────────────


def _training_config(**mp_kwargs):
    mp = SimpleNamespace(
        recompute_granularity=mp_kwargs.get("recompute_granularity"),
        recompute_num_layers=mp_kwargs.get("recompute_num_layers", 0),
        recompute_layer_ids=mp_kwargs.get("recompute_layer_ids"),
    )
    return SimpleNamespace(model_parallel_config=mp)


def test_assert_compat_passes_when_recompute_off():
    tc = _training_config(recompute_granularity=None, recompute_num_layers=0)
    # No exception expected.
    assert_recompute_pipeline_compat(tc, pipeline_schedule_algorithm="zbv-formatted")


def test_assert_compat_passes_with_inline_schedule():
    tc = _training_config(recompute_granularity="full", recompute_num_layers=4)
    assert_recompute_pipeline_compat(tc, pipeline_schedule_algorithm="1f1b")


def test_assert_compat_raises_on_conflict():
    tc = _training_config(recompute_granularity="full", recompute_num_layers=4)
    with pytest.raises(AssertionError, match="split-wgrad"):
        assert_recompute_pipeline_compat(tc, pipeline_schedule_algorithm="zbv-greedy-half")

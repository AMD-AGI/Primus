###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the feature-owned patch idempotency guard.

These validate that:
  - a guarded "wrapping" patch applies exactly once even if its handler runs
    twice (e.g. before_train re-invoked), and
  - distinct patches still compose (each wraps once), which is required for the
    stacked train_step wrappers (FP8 cache, delayed scaling, wall-clock timer).
"""

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched


class _FakeModule:
    """Stand-in for ``megatron.training.training`` (a module object)."""


def _apply_wrapping_patch(module, key):
    """Mimic the guarded train_step wrap pattern used by the FP8/timer patches."""
    if is_patched(module, key):
        return
    original = module.train_step

    def wrapped(*args, **kwargs):
        module.call_log.append(key)
        return original(*args, **kwargs)

    module.train_step = wrapped
    mark_patched(module, key)


def test_is_patched_false_before_mark():
    module = _FakeModule()
    assert not is_patched(module, "k")


def test_mark_then_is_patched():
    module = _FakeModule()
    mark_patched(module, "k")
    assert is_patched(module, "k")
    assert not is_patched(module, "other")


def test_guarded_wrap_applies_once_on_reapply():
    module = _FakeModule()
    module.call_log = []
    module.train_step = lambda: "base"

    # Apply the same patch twice (simulating before_train running twice).
    _apply_wrapping_patch(module, "megatron.train_step.demo")
    _apply_wrapping_patch(module, "megatron.train_step.demo")

    module.train_step()
    # Wrapped exactly once -> the side effect is recorded once per call.
    assert module.call_log == ["megatron.train_step.demo"]


def test_distinct_patches_still_compose():
    module = _FakeModule()
    module.call_log = []
    module.train_step = lambda: "base"

    _apply_wrapping_patch(module, "patch.a")
    _apply_wrapping_patch(module, "patch.b")
    # Re-run both: idempotent, so still only one layer each.
    _apply_wrapping_patch(module, "patch.a")
    _apply_wrapping_patch(module, "patch.b")

    module.train_step()
    assert sorted(module.call_log) == ["patch.a", "patch.b"]

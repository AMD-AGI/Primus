###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the CudaPrefetchIterator-cache invalidation that the MLPerf
warmup hook performs in its epilogue.

Background
----------
``patch_grad_zero_and_data_prefetch`` (priority 41) lazily wraps the current
``data_iterator`` in a ``CudaPrefetchIterator`` on its first invocation and
caches the result in a closure-local ``_prefetch_state["iter"]``.  The
MLPerf one-shot warmup hook (priority 95) drives the first invocation of
that inner ``_patched_train_step`` with ``data_iterator = synthetic_iter``
for the synthetic warmup steps, which means the cached prefetch wrapper
ends up bound to the synthetic iterator -- and ``MegatronDataloaderWrapper``
is cyclic (never raises ``StopIteration``).

Without invalidation, every subsequent real training step's
``_synced_prefetch_fwd_bwd`` substitutes the cached
``CudaPrefetchIterator(synthetic_iter)`` for the incoming real
``data_iterator`` argument, so the model trains forever on the synthetic
mock dataset and val_loss on real data stays stuck near 1.38.

The fix exposes a ``_PREFETCH_HANDLE`` plus a ``reset_prefetch_state``
helper on ``delayed_fp8_scaling_patches`` and calls it from the warmup
epilogue.  These tests cover the helper contract, the closure-handle
binding done at patch-installation time, and the integration semantics
that "after warmup epilogue, the next train_step rebuilds the prefetcher".

Level 1: Pure Python, no GPU.

Run:
    python -m pytest \
      tests/unit_tests/backends/megatron/test_warmup_prefetch_cache.py -v
"""

import pytest

from primus.backends.megatron.patches import delayed_fp8_scaling_patches as dfp


@pytest.fixture(autouse=True)
def _isolate_prefetch_handle():
    """Snapshot+restore the module-level ``_PREFETCH_HANDLE`` so tests can
    mutate it without bleeding into each other or into a real patch install.
    """
    saved = dict(dfp._PREFETCH_HANDLE)
    yield
    dfp._PREFETCH_HANDLE.clear()
    dfp._PREFETCH_HANDLE.update(saved)


# ---------------------------------------------------------------------------
# Helper contract
# ---------------------------------------------------------------------------


class _FakePrefetchIter:
    """Stand-in for ``CudaPrefetchIterator`` -- the helper only inspects the
    type name, never the behaviour."""

    def __next__(self):
        raise StopIteration


def test_get_prefetch_state_none_when_unbound():
    """Before any patch installation, the handle points at None."""
    dfp._PREFETCH_HANDLE["state"] = None
    assert dfp.get_prefetch_state() is None


def test_get_prefetch_state_returns_closure_dict_when_bound():
    """``patch_grad_zero_and_data_prefetch`` binds its closure-local
    ``_prefetch_state`` here at install time; the getter must return it."""
    bound = {"iter": _FakePrefetchIter()}
    dfp._PREFETCH_HANDLE["state"] = bound
    assert dfp.get_prefetch_state() is bound


def test_reset_prefetch_state_noop_when_unbound():
    """No patch installed -> reset is a no-op returning None."""
    dfp._PREFETCH_HANDLE["state"] = None
    assert dfp.reset_prefetch_state() is None


def test_reset_prefetch_state_noop_when_bound_but_empty():
    """Patch installed but no iterator yet cached -> nothing to evict."""
    bound = {}
    dfp._PREFETCH_HANDLE["state"] = bound
    assert dfp.reset_prefetch_state() is None
    assert bound == {}  # state dict untouched


def test_reset_prefetch_state_evicts_cached_iter():
    """The cached iterator is returned AND popped from the state dict, so
    the closure's ``if "iter" not in _prefetch_state`` guard will rebuild
    on the next train_step."""
    fake_iter = _FakePrefetchIter()
    bound = {"iter": fake_iter}
    dfp._PREFETCH_HANDLE["state"] = bound

    evicted = dfp.reset_prefetch_state()

    assert evicted is fake_iter
    assert "iter" not in bound, "iter key must be removed after reset"


def test_reset_prefetch_state_idempotent():
    """Calling reset twice doesn't crash and the second call returns None."""
    bound = {"iter": _FakePrefetchIter()}
    dfp._PREFETCH_HANDLE["state"] = bound

    first = dfp.reset_prefetch_state()
    second = dfp.reset_prefetch_state()

    assert first is not None
    assert second is None


def test_reset_prefetch_state_preserves_other_keys():
    """Reset must only pop the ``iter`` key; any other state (e.g. metadata
    a future patch revision might add) should survive."""
    fake_iter = _FakePrefetchIter()
    bound = {"iter": fake_iter, "other_metadata": 42}
    dfp._PREFETCH_HANDLE["state"] = bound

    evicted = dfp.reset_prefetch_state()

    assert evicted is fake_iter
    assert bound == {"other_metadata": 42}


# ---------------------------------------------------------------------------
# Integration semantics: warmup epilogue + next-train_step rebuild
# ---------------------------------------------------------------------------


def test_warmup_epilogue_makes_next_step_rebuild_prefetcher():
    """End-to-end behavioural test of the contract we rely on.

    Simulates:
      1. Patch installation publishes a closure dict via ``_PREFETCH_HANDLE``.
      2. Warmup step 1 lazily caches a prefetch wrapper around the SYNTHETIC
         iterator.
      3. Warmup epilogue calls ``reset_prefetch_state()``.
      4. The next ``_patched_train_step`` call sees ``"iter" not in
         _prefetch_state`` and rebuilds the wrapper around the REAL
         iterator.

    What we assert: step (4) sees an empty / iter-free state dict, which is
    the precondition for the closure's lazy-rebuild guard to fire.
    """

    # Step 1: patch install publishes the closure dict.
    closure_state: dict = {}
    dfp._PREFETCH_HANDLE["state"] = closure_state

    # Step 2: warmup step 1 caches the prefetcher around synthetic_iter.
    synthetic_iter_marker = object()
    closure_state["iter"] = synthetic_iter_marker
    assert dfp.get_prefetch_state() is closure_state
    assert dfp.get_prefetch_state()["iter"] is synthetic_iter_marker

    # Step 3: warmup epilogue invalidates.
    evicted = dfp.reset_prefetch_state()
    assert evicted is synthetic_iter_marker

    # Step 4: next train_step's lazy-build guard now fires -- i.e. "iter"
    # is not in the state dict, so the closure will construct a fresh
    # CudaPrefetchIterator around the real data_iterator argument.
    assert "iter" not in closure_state, (
        "After warmup epilogue invalidates the cache, the closure's "
        "_prefetch_state['iter'] must be absent so the next train_step "
        "rebuilds the prefetch wrapper around the real data_iterator."
    )

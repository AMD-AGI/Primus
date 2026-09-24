###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Regression tests for ``grad_buffer_ownership_patches``.

Covers the pure, megatron-independent helpers that decide which slices of a
``_ParamAndGradBuffer.grad_data`` may skip zeroing (``_owned_slices``,
``_reset_complement``) and the registration gate (``_is_enabled``). These run
on plain CPU tensors and fake ``SimpleNamespace`` buffers/params -- no GPU and
no live Megatron classes required.

The monkey-patched ``zero_grad_buffer`` / ``_ParamAndGradBuffer.reset``
wrappers themselves are intentionally out of scope here: they are closures
built inside ``patch_grad_buffer_ownership()``, which patches real Megatron
classes, so exercising them needs an actual (or end-to-end) Megatron
environment rather than a unit test.
"""

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches.turbo import grad_buffer_ownership_patches as gbo
from primus_turbo.pytorch.core import grad_ownership


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset both this module's and grad_ownership's module-level state
    around every test."""
    gbo._state = {"owned": frozenset(), "seen": set(), "logged": False}
    gbo._POISON = False
    grad_ownership._written = set()
    grad_ownership._skipped = set()
    grad_ownership._previous = frozenset()
    grad_ownership._ever_recorded = False
    yield
    gbo._state = {"owned": frozenset(), "seen": set(), "logged": False}
    gbo._POISON = False
    grad_ownership._written = set()
    grad_ownership._skipped = set()
    grad_ownership._previous = frozenset()
    grad_ownership._ever_recorded = False


def _slice_of(tensor: torch.Tensor):
    return (tensor.data_ptr(), tensor.numel())


class _FakeParam:
    """A hashable (identity-based, like ``nn.Parameter``) stand-in for a
    ``param_index_map`` key. ``types.SimpleNamespace`` is *not* usable here:
    it defines value-based ``__eq__`` without ``__hash__``, which makes
    instances unhashable and breaks use as a dict key."""

    def __init__(self, main_grad):
        self.main_grad = main_grad


def _fake_param(main_grad):
    return _FakeParam(main_grad)


class TestOwnedSlices:
    def test_returns_nothing_when_owned_set_is_empty(self):
        grad = torch.zeros(30, dtype=torch.float32)
        param = _fake_param(grad[10:20])
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (10, 20, 0)})

        assert gbo._owned_slices(buffer) == []

    def test_matches_a_slice_by_exact_address_dtype_and_numel(self):
        grad = torch.zeros(30, dtype=torch.float32)
        main_grad = grad[10:20]
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (10, 20, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        assert gbo._owned_slices(buffer) == [(10, 20, _slice_of(main_grad))]

    def test_excludes_a_param_whose_key_is_not_in_the_owned_set(self):
        """A param may match address/dtype/numel exactly, but if its
        (data_ptr, numel) key was never recorded as a beta=0 write, it must
        not be treated as owned."""
        grad = torch.zeros(30, dtype=torch.float32)
        main_grad = grad[10:20]
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (10, 20, 0)})
        gbo._state["owned"] = frozenset({(0xDEAD, 10)})  # unrelated key

        assert gbo._owned_slices(buffer) == []

    def test_excludes_dtype_mismatch(self):
        grad = torch.zeros(30, dtype=torch.float32)
        # main_grad lives elsewhere and merely has the same numel; dtype differs.
        main_grad = torch.zeros(10, dtype=torch.bfloat16)
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (10, 20, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        assert gbo._owned_slices(buffer) == []

    def test_excludes_numel_mismatch(self):
        grad = torch.zeros(30, dtype=torch.float32)
        main_grad = grad[10:15]  # 5 elements, but the index map claims [10, 20) = 10
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (10, 20, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        assert gbo._owned_slices(buffer) == []

    def test_excludes_non_contiguous_main_grad(self):
        grad = torch.zeros(30, dtype=torch.float32)
        main_grad = grad[0:20:2]  # 10 elements, starts at offset 0, but strided
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (0, 10, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        assert not main_grad.is_contiguous()
        assert gbo._owned_slices(buffer) == []

    def test_excludes_address_mismatch_against_the_index_maps_claimed_start(self):
        """``main_grad`` is a real, contiguous, correctly-sized, correctly-typed
        view into the buffer -- just not at the offset the index map claims for
        this param. Must not be honoured."""
        grad = torch.zeros(30, dtype=torch.float32)
        main_grad = grad[5:15]  # real slice of the buffer, but not at claimed [10, 20)
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (10, 20, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        assert gbo._owned_slices(buffer) == []

    def test_raises_on_overlapping_owned_slices(self):
        grad = torch.zeros(30, dtype=torch.float32)
        main_grad_a = grad[0:10]
        main_grad_b = grad[5:15]
        param_a = _fake_param(main_grad_a)
        param_b = _fake_param(main_grad_b)
        buffer = SimpleNamespace(
            grad_data=grad,
            param_index_map={param_a: (0, 10, 0), param_b: (5, 15, 0)},
        )
        gbo._state["owned"] = frozenset({_slice_of(main_grad_a), _slice_of(main_grad_b)})

        with pytest.raises(RuntimeError, match="overlapping"):
            gbo._owned_slices(buffer)


class TestResetComplement:
    def test_declines_and_returns_false_when_nothing_is_owned(self):
        grad = torch.full((30,), 7.0, dtype=torch.float32)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={})

        assert gbo._reset_complement(buffer) is False
        # Declining must not have touched the buffer at all.
        assert torch.equal(grad, torch.full((30,), 7.0, dtype=torch.float32))

    def test_zeroes_the_complement_and_leaves_owned_slices_untouched(self):
        grad = torch.full((30,), 7.0, dtype=torch.float32)
        main_grad = grad[10:20]
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (10, 20, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        assert gbo._reset_complement(buffer) is True

        assert torch.equal(grad[0:10], torch.zeros(10))
        assert torch.equal(grad[10:20], torch.full((10,), 7.0))  # owned: untouched
        assert torch.equal(grad[20:30], torch.zeros(10))

    def test_zeroes_only_the_tail_when_the_owned_slice_starts_at_zero(self):
        """Sanity check for the tail branch: when the owned slice already
        starts at offset 0, only the trailing complement should be zeroed."""
        grad = torch.full((10,), 3.0, dtype=torch.float32)
        main_grad = grad[0:4]
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (0, 4, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        gbo._reset_complement(buffer)

        assert torch.equal(grad[0:4], torch.full((4,), 3.0))
        assert torch.equal(grad[4:10], torch.zeros(6))

    def test_records_skipped_slices_with_grad_ownership(self):
        grad = torch.zeros(20, dtype=torch.float32)
        main_grad = grad[0:10]
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (0, 10, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        gbo._reset_complement(buffer)

        assert _slice_of(main_grad) in grad_ownership._skipped

    def test_poison_mode_fills_owned_slices_with_nan_instead_of_preserving_them(self):
        gbo._POISON = True
        grad = torch.full((20,), 7.0, dtype=torch.float32)
        main_grad = grad[0:10]
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (0, 10, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        gbo._reset_complement(buffer)

        assert torch.isnan(grad[0:10]).all(), "poison mode must overwrite the skipped region with NaN"
        assert torch.equal(grad[10:20], torch.zeros(10)), "the real complement clear must be unaffected"

    def test_begin_step_raises_when_a_claimed_slice_is_never_overwritten(self):
        """End-to-end through the real grad_ownership consumer: this module
        predicts (via ``_reset_complement``) that a slice will be rewritten,
        but the producer never calls ``record_overwrite`` for it. The next
        rotation must raise -- proving the safety net is wired all the way
        through from this patch into ``grad_ownership.begin_step``."""
        grad = torch.zeros(20, dtype=torch.float32)
        main_grad = grad[0:10]
        param = _fake_param(main_grad)
        buffer = SimpleNamespace(grad_data=grad, param_index_map={param: (0, 10, 0)})
        gbo._state["owned"] = frozenset({_slice_of(main_grad)})

        gbo._reset_complement(buffer)  # predicts + skips zeroing main_grad's slice

        with pytest.raises(RuntimeError, match="left unzeroed"):
            grad_ownership.begin_step()


class TestIsEnabled:
    def test_disabled_short_circuits_before_reading_args(self, monkeypatch):
        monkeypatch.setattr(gbo, "_DISABLED", True)
        monkeypatch.setattr(gbo, "get_args", lambda ctx: pytest.fail("must not read args when disabled"))

        assert gbo._is_enabled(ctx=None) is False

    def test_false_when_gradient_accumulation_fusion_is_off(self, monkeypatch):
        monkeypatch.setattr(gbo, "_DISABLED", False)
        monkeypatch.setattr(gbo, "get_args", lambda ctx: SimpleNamespace(gradient_accumulation_fusion=False))
        monkeypatch.setattr(
            gbo, "is_primus_turbo_can_patch", lambda ctx: pytest.fail("must short-circuit before this")
        )

        assert gbo._is_enabled(ctx=None) is False

    def test_false_when_turbo_cannot_patch(self, monkeypatch):
        monkeypatch.setattr(gbo, "_DISABLED", False)
        monkeypatch.setattr(gbo, "get_args", lambda ctx: SimpleNamespace(gradient_accumulation_fusion=True))
        monkeypatch.setattr(gbo, "is_primus_turbo_can_patch", lambda ctx: False)

        assert gbo._is_enabled(ctx=None) is False

    def test_true_when_all_conditions_hold(self, monkeypatch):
        monkeypatch.setattr(gbo, "_DISABLED", False)
        monkeypatch.setattr(gbo, "get_args", lambda ctx: SimpleNamespace(gradient_accumulation_fusion=True))
        monkeypatch.setattr(gbo, "is_primus_turbo_can_patch", lambda ctx: True)

        assert gbo._is_enabled(ctx=None) is True

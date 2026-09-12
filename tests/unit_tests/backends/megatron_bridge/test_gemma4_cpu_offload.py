###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the Gemma 4 CPU optimizer offload patch.

The behaviour worth pinning down is that enabling offload must not quietly put
the run back on TE's FusedAdam, because on this device that is a hang rather
than a slowdown -- a failure mode no assertion would catch and no log line would
explain. These tests need no GPU, no megatron-core and no checkpoint: the patch
operates on a config container and on ``sys.modules``, so both are substitutable.
"""

import sys
from types import SimpleNamespace

import pytest


class _FusedAdam:
    """Stand-in for transformer_engine's FusedAdam, which we must not end up on."""


def _make_optimizer_cfg(
    *,
    precision_aware=False,
    distributed=True,
    decoupled_wd=True,
):
    return SimpleNamespace(
        optimizer_cpu_offload=False,
        optimizer_offload_fraction=0.0,
        use_torch_optimizer_for_cpu_offload=False,
        use_precision_aware_optimizer=precision_aware,
        use_distributed_optimizer=distributed,
        decoupled_weight_decay=decoupled_wd,
    )


@pytest.fixture(autouse=True)
def logged(monkeypatch):
    """Capture log_rank_0; Primus's logger is None outside a real run."""
    from primus.backends.megatron_bridge.patches.gemma4 import gemma4_cpu_offload as mod

    messages: list[str] = []
    monkeypatch.setattr(mod, "log_rank_0", lambda msg, *a, **k: messages.append(str(msg)))
    return messages


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PRIMUS_GEMMA4_CPU_OFFLOAD", raising=False)


@pytest.fixture
def fake_optimizer_module(monkeypatch):
    """Put a substitute megatron.core.optimizer in sys.modules, Adam bound to FusedAdam."""
    module = SimpleNamespace(Adam=_FusedAdam, USING_PYTORCH_OPTIMIZER=False)
    monkeypatch.setitem(sys.modules, "megatron.core.optimizer", module)
    return module


# -----------------------------------------------------------------------------
# Reading the request
# -----------------------------------------------------------------------------


def test_offload_not_requested_by_default():
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        offload_fraction,
    )

    assert offload_fraction() is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", 1.0),  # the convenience spelling of "offload everything"
        ("1.0", 1.0),
        ("0.5", 0.5),
        ("  0.25  ", 0.25),
    ],
)
def test_offload_fraction_is_parsed(monkeypatch, raw, expected):
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        offload_fraction,
    )

    monkeypatch.setenv("PRIMUS_GEMMA4_CPU_OFFLOAD", raw)
    assert offload_fraction() == expected


@pytest.mark.parametrize("raw", ["", "   ", "0", "0.0"])
def test_empty_or_zero_means_not_requested(monkeypatch, raw):
    """Zero offload is indistinguishable from no offload, so do not touch anything."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        offload_fraction,
    )

    monkeypatch.setenv("PRIMUS_GEMMA4_CPU_OFFLOAD", raw)
    assert offload_fraction() is None


@pytest.mark.parametrize("raw", ["yes", "1,0", "1.5", "-0.1"])
def test_unusable_values_are_declined_loudly(monkeypatch, logged, raw):
    """A typo here would silently cost the whole reason for the run."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        offload_fraction,
    )

    monkeypatch.setenv("PRIMUS_GEMMA4_CPU_OFFLOAD", raw)
    assert offload_fraction() is None
    assert any(raw in m for m in logged), logged


# -----------------------------------------------------------------------------
# The GPU half must not be FusedAdam -- the reason this patch exists
# -----------------------------------------------------------------------------


def test_gpu_optimizer_is_rebound_to_torch(fake_optimizer_module):
    """megatron reads the module-level Adam at line 516 regardless of any flag."""
    import torch

    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        _use_torch_gpu_optimizer,
    )

    assert fake_optimizer_module.Adam is _FusedAdam
    _use_torch_gpu_optimizer()
    assert fake_optimizer_module.Adam is torch.optim.AdamW


def test_rebinding_is_idempotent(fake_optimizer_module, logged):
    import torch

    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        _use_torch_gpu_optimizer,
    )

    _use_torch_gpu_optimizer()
    logged.clear()
    _use_torch_gpu_optimizer()

    assert fake_optimizer_module.Adam is torch.optim.AdamW
    assert logged == []


def test_missing_optimizer_module_says_a_hang_is_coming(monkeypatch, logged):
    """There is no partial action available here, so the log is the whole mitigation."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        _use_torch_gpu_optimizer,
    )

    monkeypatch.delitem(sys.modules, "megatron.core.optimizer", raising=False)
    _use_torch_gpu_optimizer()

    assert any("hang" in m for m in logged), logged


# -----------------------------------------------------------------------------
# Config the offload path asserts on
# -----------------------------------------------------------------------------


def test_enabling_offload_sets_the_fields(fake_optimizer_module):
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        _enable_offload,
    )

    container = SimpleNamespace(optimizer=_make_optimizer_cfg())
    _enable_offload(container, 0.75)

    opt = container.optimizer
    assert opt.optimizer_cpu_offload is True
    assert opt.optimizer_offload_fraction == 0.75
    assert opt.use_torch_optimizer_for_cpu_offload is True


def test_precision_aware_is_turned_on_not_off(fake_optimizer_module):
    """The opposite of the non-offload path: HDO supplies the master weights."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        _enable_offload,
    )

    container = SimpleNamespace(optimizer=_make_optimizer_cfg(precision_aware=False))
    _enable_offload(container, 1.0)

    assert container.optimizer.use_precision_aware_optimizer is True


def test_downstream_assertions_are_satisfied(fake_optimizer_module, logged):
    """Both of these are bare asserts a long way from their cause."""
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        _enable_offload,
    )

    container = SimpleNamespace(
        optimizer=_make_optimizer_cfg(distributed=False, decoupled_wd=False),
    )
    _enable_offload(container, 1.0)

    opt = container.optimizer
    assert opt.use_distributed_optimizer is True
    assert opt.decoupled_weight_decay is True
    assert any("use_distributed_optimizer" in m for m in logged), logged
    assert any("decoupled_weight_decay" in m for m in logged), logged


def test_missing_optimizer_section_is_harmless(fake_optimizer_module):
    from primus.backends.megatron_bridge.patches.gemma4.gemma4_cpu_offload import (
        _enable_offload,
    )

    _enable_offload(SimpleNamespace(), 1.0)  # must not raise


# -----------------------------------------------------------------------------
# The coupling with gemma4_local_spec, which clears the same flag
# -----------------------------------------------------------------------------


def test_local_spec_leaves_precision_aware_alone_when_offloading(monkeypatch):
    """Clearing it here would push the fp32 master weights back onto the GPU."""
    from primus.backends.megatron_bridge.patches.gemma4 import gemma4_local_spec as mod

    monkeypatch.setattr(mod, "log_rank_0", lambda msg, *a, **k: None)
    monkeypatch.setenv("PRIMUS_GEMMA4_CPU_OFFLOAD", "1.0")

    container = SimpleNamespace(optimizer=_make_optimizer_cfg(precision_aware=True))
    mod._use_torch_optimizer(container)

    assert container.optimizer.use_precision_aware_optimizer is True


def test_local_spec_still_clears_precision_aware_without_offload(monkeypatch):
    """The original behaviour has to survive: bare torch Adam has no master weights."""
    from primus.backends.megatron_bridge.patches.gemma4 import gemma4_local_spec as mod

    monkeypatch.setattr(mod, "log_rank_0", lambda msg, *a, **k: None)
    monkeypatch.delenv("PRIMUS_GEMMA4_CPU_OFFLOAD", raising=False)

    container = SimpleNamespace(optimizer=_make_optimizer_cfg(precision_aware=True))
    mod._use_torch_optimizer(container)

    assert container.optimizer.use_precision_aware_optimizer is False

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend import / default / hardware-gating guarantees for V4 attention.

These are the regression gates for the "gluon must not be a hard dependency"
contract:

* the shared config default is ``triton_v1`` (arch-portable), NOT the
  gfx950-only ``gluon``;
* importing ``v4_attention_kernels`` (and ``deepseek_v4_attention``) must NOT
  eagerly import the gluon backend (``_gluon_dsa`` /
  ``triton.experimental.gluon``), so ``eager`` / ``triton_v1`` / ``triton_v2``
  work on any Triton build / GPU arch;
* the gluon backend is loaded lazily via ``load_gluon_attention_backends`` and,
  when selected, ``_require_gfx950`` rejects non-gfx950 devices.

All tests here are hardware-independent (no CUDA required).
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_config_defaults_are_triton_v1():
    """Shared config default must be the arch-portable triton_v1 (not gluon)."""
    from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
        DeepSeekV4TransformerConfig,
    )

    fields = DeepSeekV4TransformerConfig.__dataclass_fields__
    assert fields["use_v4_attention_backend"].default == "triton_v1"
    assert fields["use_v4_csa_attention_backend"].default == "triton_v1"


def test_kernels_package_exposes_lazy_gluon_loader():
    """The package exposes the lazy loader and does NOT eagerly bind gluon entries."""
    pkg = importlib.import_module("primus.backends.megatron.core.transformer.v4_attention_kernels")
    assert hasattr(pkg, "load_gluon_attention_backends")
    # gluon entries must NOT be module-level attributes (would mean eager import).
    assert not hasattr(pkg, "v4_attention_gluon")
    assert not hasattr(pkg, "v4_csa_attention_gluon")


def test_importing_kernels_package_does_not_pull_gluon():
    """Reloading the kernels package must not import ``_gluon_dsa`` as a side effect."""
    # Purge any previously-imported gluon modules so this asserts the *package*
    # import path, independent of test ordering (e.g. the gfx950 gluon UT).
    for name in [m for m in sys.modules if "_gluon_dsa" in m]:
        del sys.modules[name]
    pkg = importlib.import_module("primus.backends.megatron.core.transformer.v4_attention_kernels")
    importlib.reload(pkg)
    assert not any(
        "_gluon_dsa" in m for m in sys.modules
    ), "importing v4_attention_kernels must not eagerly import the gluon backend"


def test_attention_module_uses_lazy_gluon_helpers():
    """``deepseek_v4_attention`` imports the lazy loader + arch guard, not gluon entries."""
    mod = importlib.import_module("primus.backends.megatron.core.transformer.deepseek_v4_attention")
    assert hasattr(mod, "load_gluon_attention_backends")
    assert hasattr(mod, "_require_gfx950")
    # no eager module-level gluon entry bindings
    assert not hasattr(mod, "v4_attention_gluon")
    assert not hasattr(mod, "v4_csa_attention_gluon")


def test_require_gfx950_rejects_non_gfx950(monkeypatch):
    """``_require_gfx950`` raises a clear error on a non-gfx950 device."""
    torch = pytest.importorskip("torch")
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        _require_gfx950,
    )

    class _FakeProps:
        gcnArchName = "gfx942:sramecc+:xnack-"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _FakeProps())
    with pytest.raises(RuntimeError, match="gfx950"):
        _require_gfx950()


def test_require_gfx950_rejects_no_device(monkeypatch):
    """``_require_gfx950`` raises when no accelerator is available."""
    torch = pytest.importorskip("torch")
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        _require_gfx950,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="gfx950"):
        _require_gfx950()


def test_require_gfx950_accepts_gfx950(monkeypatch):
    """``_require_gfx950`` passes on a gfx950 device."""
    torch = pytest.importorskip("torch")
    from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
        _require_gfx950,
    )

    class _FakeProps:
        gcnArchName = "gfx950:sramecc+:xnack-"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _FakeProps())
    _require_gfx950()  # must not raise

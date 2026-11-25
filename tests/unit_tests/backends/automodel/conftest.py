###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
import pkgutil
from collections import defaultdict
from types import SimpleNamespace

import pytest

from primus.backends.automodel import options
from primus.core.patches.patch_registry import PatchRegistry


def _registry_state():
    return PatchRegistry._patches_by_backend_phase, PatchRegistry._all_patches


def _set_registry_state(state):
    by_phase, all_patches = state
    PatchRegistry._patches_by_backend_phase = by_phase
    PatchRegistry._all_patches = all_patches


def _copy_registry_state(state):
    by_phase, all_patches = state
    copied = defaultdict(lambda: defaultdict(list))
    for backend, phases in by_phase.items():
        for phase, patches in phases.items():
            copied[backend][phase] = list(patches)
    return copied, list(all_patches)


def production_registry():
    """The registry as this backend's patch modules fill it, built from scratch.

    The patch modules register at import, and Python imports a module once, so
    a test elsewhere that empties the global registry (``main``'s
    ``tests/unit_tests/core/patches`` do, without restoring it) would leave
    every later test here without them. Each module is re-executed against an
    empty registry instead of trusting whatever the session left behind.
    """
    import primus.backends.automodel.patches as package

    saved = _registry_state()
    _set_registry_state((defaultdict(lambda: defaultdict(list)), []))
    try:
        for info in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + "."):
            if info.name.endswith(("_patches", "_patch")):
                importlib.reload(importlib.import_module(info.name))
        return _copy_registry_state(_registry_state())
    finally:
        _set_registry_state(saved)


@pytest.fixture(scope="session")
def _production_registry():
    return production_registry()


@pytest.fixture
def build_production_registry():
    return production_registry


@pytest.fixture(autouse=True)
def _patch_registry(_production_registry):
    """Every test sees this backend's patches registered, whatever ran before it."""
    saved = _registry_state()
    _set_registry_state(_copy_registry_state(_production_registry))
    yield
    _set_registry_state(saved)


@pytest.fixture(autouse=True)
def _default_options():
    """Every test starts from the defaults: settings are process-global."""
    options.load({})
    yield
    options.load({})


@pytest.fixture
def set_option():
    """``set_option("primus_turbo.fp8_linear", True)``; ``None`` unsets."""
    sections = {}

    def _set(path, value):
        section, key = path.split(".", 1)
        sections.setdefault(section, {})[key] = value
        options.load(sections)

    return _set


@pytest.fixture
def flux_fused_patches(monkeypatch):
    """The FLUX fused-op patch modules, with every class attribute their installs set restored afterwards."""
    transformer_flux = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    from diffusers.models.attention import FeedForward

    from primus.backends.automodel.models.flux import (
        adaln,
        adaln_dual,
        mlp_gelu,
        qk_norm_rope,
    )
    from primus.backends.automodel.models.flux.attn_processor import (
        PrimusFluxAttnProcessor,
    )

    single, dual = transformer_flux.FluxSingleTransformerBlock, transformer_flux.FluxTransformerBlock
    attention = transformer_flux.FluxAttention
    for attr in ("_default_processor_cls", "_available_processors"):
        monkeypatch.setattr(attention, attr, getattr(attention, attr))
    monkeypatch.setattr(PrimusFluxAttnProcessor, "dit_qk_norm_rope", None)
    monkeypatch.setattr(mlp_gelu, "_single_forward", None)
    for cls in (FeedForward, single, dual, transformer_flux.FluxTransformer2DModel):
        monkeypatch.setattr(cls, "forward", cls.forward)
    for cls, attr in (
        (FeedForward, mlp_gelu._PATCHED_ATTR),
        (single, mlp_gelu._PATCHED_ATTR),
        (attention, qk_norm_rope._PATCHED_ATTR),
        (single, adaln._PATCHED_ATTR),
        (dual, adaln_dual._PATCHED_ATTR),
    ):
        monkeypatch.setattr(cls, attr, False, raising=False)
    return SimpleNamespace(
        adaln_dual=adaln_dual,
        adaln=adaln,
        mlp_gelu=mlp_gelu,
        qk_norm_rope=qk_norm_rope,
    )

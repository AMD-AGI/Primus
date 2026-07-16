###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the ROCm-safe attention_backend patch.

These verify the patched ``LanguageModule._set_attention_backend`` reconciles the
selected backend with the image's baked ``NVTE_*_ATTN`` env vars the way ROCm
needs (override/respect rather than assert-crash), without needing a GPU.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import AttnBackend

import primus.backends.megatron.patches.attention_backend_patches as patch_mod

_NVTE = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")


@pytest.fixture
def patched_set_attention_backend(monkeypatch):
    """Apply the patch onto a pristine LanguageModule and restore it afterwards."""
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    original = LanguageModule.__dict__.get("_set_attention_backend")
    monkeypatch.delattr(LanguageModule, "_primus_attention_backend_patched", raising=False)

    patch_mod.patch_attention_backend(None)
    try:
        yield LanguageModule._set_attention_backend
    finally:
        if original is not None:
            LanguageModule._set_attention_backend = original
        if hasattr(LanguageModule, "_primus_attention_backend_patched"):
            delattr(LanguageModule, "_primus_attention_backend_patched")


def _run(monkeypatch, backend, baked):
    # Simulate the ROCm image's baked NVTE_*_ATTN before model construction.
    for name in _NVTE:
        monkeypatch.delenv(name, raising=False)
    for name, value in baked.items():
        monkeypatch.setenv(name, value)

    dummy = SimpleNamespace(config=SimpleNamespace(attention_backend=backend))
    LanguageModule._set_attention_backend(dummy)

    import os

    return tuple(os.environ.get(name) for name in _NVTE)


# The Primus ROCm images bake NVTE_FLASH_ATTN=0 / NVTE_FUSED_ATTN=1.
_BAKED = {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1"}


def test_auto_respects_baked_flash_off(patched_set_attention_backend, monkeypatch):
    # "auto" must NOT force NVTE_FLASH_ATTN=1 (that is what crashes stock
    # megatron); it fills only the unset var, leaving the baked FLASH=0.
    assert _run(monkeypatch, AttnBackend.auto, _BAKED) == ("0", "1", "1")


def test_unfused_overrides_baked_fused(patched_set_attention_backend, monkeypatch):
    # An explicit backend wins over the baked defaults (FUSED 1 -> 0), where
    # stock megatron would assert-crash.
    assert _run(monkeypatch, AttnBackend.unfused, _BAKED) == ("0", "0", "1")


def test_fused_sets_expected_combination(patched_set_attention_backend, monkeypatch):
    assert _run(monkeypatch, AttnBackend.fused, _BAKED) == ("0", "1", "0")


def test_local_disables_all(patched_set_attention_backend, monkeypatch):
    assert _run(monkeypatch, AttnBackend.local, _BAKED) == ("0", "0", "0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

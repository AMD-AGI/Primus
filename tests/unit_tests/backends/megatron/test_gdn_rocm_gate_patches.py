###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the Megatron upstream GatedDeltaNet ROCm gate patch.

Verifies:
  1. The source anchor still matches upstream ``GatedDeltaNet.forward``.
  2. The patch rewrites the forward body and is idempotent.
  3. Registration condition requires both ROCm and ``gated_delta_net``.
"""

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")

from megatron.core.ssm.gated_delta_net import GatedDeltaNet

import primus.backends.megatron.patches.gdn_rocm_gate_patches as patch_mod
from primus.backends.megatron.patches._patch_guard import is_patched
from primus.core.patches.context import PatchContext


def _clear_patch_state(cls):
    if hasattr(cls, "_primus_applied_patch_keys"):
        cls._primus_applied_patch_keys.discard(patch_mod._PATCH_KEY)


@pytest.fixture
def pristine_gdn_forward():
    """Restore ``GatedDeltaNet.forward`` and patch-guard state after each test."""
    original = GatedDeltaNet.forward
    yield
    GatedDeltaNet.forward = original
    _clear_patch_state(GatedDeltaNet)


def test_gate_anchor_matches_upstream_forward():
    source = inspect.getsource(GatedDeltaNet.forward)
    assert (
        patch_mod._GATE_ORI in source
    ), "Upstream GatedDeltaNet.forward changed; update gdn_rocm_gate_patches anchor."


def test_gate_new_differs_from_anchor():
    assert patch_mod._GATE_NEW != patch_mod._GATE_ORI
    assert "use_gate_in_kernel=False" in patch_mod._GATE_NEW
    assert "A_log.float().exp()" in patch_mod._GATE_NEW


def test_install_patch_rewrites_forward(pristine_gdn_forward, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    original = GatedDeltaNet.forward
    original_source = inspect.getsource(original)

    patch_mod._install_gdn_rocm_gate_patch()

    assert is_patched(GatedDeltaNet, patch_mod._PATCH_KEY)
    assert GatedDeltaNet.forward is not original

    # exec()-patched methods are not inspectable; verify the expected rewrite.
    rewritten = original_source.replace(patch_mod._GATE_ORI, patch_mod._GATE_NEW)
    assert "use_gate_in_kernel=False" in rewritten
    assert "A_log.float().exp()" in rewritten


def test_install_patch_is_idempotent(pristine_gdn_forward, monkeypatch):
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)

    patch_mod._install_gdn_rocm_gate_patch()
    first = GatedDeltaNet.forward
    patch_mod._install_gdn_rocm_gate_patch()

    assert GatedDeltaNet.forward is first


@pytest.mark.parametrize(
    "variant, expected",
    [
        ("gated_delta_net", True),
        ("dsa", False),
        (None, False),
    ],
)
def test_uses_gated_delta_net(variant, expected):
    args = SimpleNamespace(experimental_attention_variant=variant)
    assert patch_mod._uses_gated_delta_net(args) is expected


@pytest.mark.parametrize(
    "hip, variant, expected",
    [
        ("6.2.41133-65d174c3", "gated_delta_net", True),
        ("6.2.41133-65d174c3", "dsa", False),
        (None, "gated_delta_net", False),
    ],
)
def test_should_patch_gdn_rocm_gate(hip, variant, expected, monkeypatch):
    monkeypatch.setattr(patch_mod.torch.version, "hip", hip, raising=False)
    args = SimpleNamespace(experimental_attention_variant=variant)
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=args)},
    )
    assert patch_mod._should_patch_gdn_rocm_gate(ctx) is expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Primus-Turbo MoE grouped_mm patch.

Regression focus: the replacement on
``torchtitan.models.moe.moe._run_experts_grouped_mm`` must expose a
``__qualname__`` containing the sentinel ``_run_experts_grouped_mm_dynamic`` so
upstream ``apply_compile`` treats it as already patched and skips
``torch.compile`` over the ``torch.compiler.disable``'d turbo kernels (dynamo
gb0098). A ``functools.partial`` (no ``__qualname__``) would crash that read.
"""

import functools
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import primus.backends.torchtitan.patches.turbo.moe_grouped_mm_patches as moe_patch
from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry

PATCH_ID = "torchtitan.primus_turbo.moe_grouped_mm"


def _ctx(enable_turbo=True, use_turbo_grouped_mm=True, use_moe_fp8=False):
    params = SimpleNamespace(
        primus_turbo=SimpleNamespace(
            enable_primus_turbo=enable_turbo,
            use_turbo_grouped_mm=use_turbo_grouped_mm,
            use_moe_fp8=use_moe_fp8,
        ),
    )
    module_config = SimpleNamespace(params=params)
    return PatchContext(backend="torchtitan", phase="setup", extra={"module_config": module_config})


class TestMoeGroupedMmPatch:
    def test_patch_registered(self):
        assert PATCH_ID in PatchRegistry.list_ids()
        p = PatchRegistry.get(PATCH_ID)
        assert p is not None and p.backend == "torchtitan" and p.phase == "setup"

    def test_condition_gates_on_turbo_flags(self):
        p = PatchRegistry.get(PATCH_ID)
        assert p.condition(_ctx(True, True)) is True
        assert p.condition(_ctx(False, True)) is False
        assert p.condition(_ctx(True, False)) is False

    def _run_patch(self, use_moe_fp8, stub_impl):
        """Apply the real patch against the real torchtitan moe module.

        Saves/restores torchtitan.models.moe.moe._run_experts_grouped_mm and
        stubs the Primus grouped_mm impl so no GPU kernel runs.
        """
        import torchtitan.models.moe.moe as tt_moe

        import primus.backends.torchtitan.models.moe.moe as primus_moe

        orig_tt = tt_moe._run_experts_grouped_mm
        orig_primus = primus_moe._run_experts_grouped_mm
        try:
            primus_moe._run_experts_grouped_mm = stub_impl
            with patch.object(moe_patch, "log_rank_0"):
                moe_patch.patch_torchtitan_moe(_ctx(use_moe_fp8=use_moe_fp8))
            return tt_moe._run_experts_grouped_mm
        finally:
            tt_moe._run_experts_grouped_mm = orig_tt
            primus_moe._run_experts_grouped_mm = orig_primus

    def test_replacement_has_qualname_not_partial(self):
        pytest.importorskip("torchtitan")

        def _stub(w1, w2, w3, x, num_tokens_per_expert, use_fp8=True):
            return use_fp8

        _stub.__qualname__ = "_run_experts_grouped_mm"

        replacement = self._run_patch(use_moe_fp8=True, stub_impl=_stub)
        assert not isinstance(replacement, functools.partial)
        assert hasattr(replacement, "__qualname__")
        # Must match upstream's already_patched guard to skip torch.compile.
        assert "_run_experts_grouped_mm_dynamic" in replacement.__qualname__

    def test_replacement_binds_use_fp8(self):
        pytest.importorskip("torchtitan")
        captured = {}

        def _stub(w1, w2, w3, x, num_tokens_per_expert, use_fp8=True):
            captured["use_fp8"] = use_fp8
            return "ok"

        _stub.__qualname__ = "_run_experts_grouped_mm"

        replacement = self._run_patch(use_moe_fp8=True, stub_impl=_stub)
        # use_fp8 must be injected by the wrapper (upstream calls positionally).
        assert replacement(1, 2, 3, 4, 5) == "ok"
        assert captured["use_fp8"] is True

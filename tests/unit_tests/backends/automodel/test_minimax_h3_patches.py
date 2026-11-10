###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for how the MiniMax-H3 patches are gated and ordered.

The wrappers are additive, so gating is hygiene rather than correctness: a patch that only
appears in the runs that use it is much easier to reason about when something goes wrong.
The asymmetry that matters is the default. Registering for a run that does not want it
changes nothing; skipping for a run that does want it fails later, at adapter or model
resolution, with an error that says nothing about this patch.
"""

import subprocess
import sys
import types

import pytest

PIPELINE_ID = "automodel.models.minimax_h3.pipeline"
ADAPTER_ID = "automodel.models.minimax_h3.adapter"
MODULATION_ID = "automodel.models.minimax_h3.modulation"


def _patches():
    import primus.backends.automodel.patches  # noqa: F401
    from primus.core.patches.patch_registry import PatchRegistry

    return {p.id: p for p in PatchRegistry.iter_patches(backend="automodel", phase="before_train")}


def _ctx(adapter_type):
    """The context the trainer builds: the module config's params, as YAML."""
    from primus.core.patches import PatchContext

    params = types.SimpleNamespace(flow_matching=types.SimpleNamespace(adapter_type=adapter_type))
    return PatchContext(
        backend="automodel",
        phase="before_train",
        extra={"module_config": types.SimpleNamespace(params=params)},
    )


class TestRegistration:
    def test_both_patches_are_registered_for_before_train(self):
        patches = _patches()
        assert PIPELINE_ID in patches and ADAPTER_ID in patches and MODULATION_ID in patches

    def test_the_pipeline_is_in_place_before_the_adapter(self):
        """Both have to precede the recipe building the model; the order between them is by priority."""
        patches = _patches()
        assert patches[PIPELINE_ID].priority < patches[ADAPTER_ID].priority

    def test_the_modulation_patch_is_in_place_before_the_model_is_built(self):
        """It swaps class methods, so it only has to precede the first forward; it runs early anyway."""
        patches = _patches()
        assert patches[MODULATION_ID].priority < patches["automodel.models.minimax_h3.parallelize"].priority


class TestGating:
    @pytest.mark.parametrize("patch_id", [PIPELINE_ID, ADAPTER_ID, MODULATION_ID])
    def test_a_minimax_h3_run_registers_both(self, patch_id):
        assert _patches()[patch_id].condition(_ctx("minimax_h3")) is True

    @pytest.mark.parametrize("patch_id", [PIPELINE_ID, ADAPTER_ID, MODULATION_ID])
    @pytest.mark.parametrize(
        "other", ["flux", "flux2", "wan", "hunyuan", "qwen_image", "simple", "ideogram4"]
    )
    def test_another_model_does_not(self, patch_id, other):
        assert _patches()[patch_id].condition(_ctx(other)) is False

    @pytest.mark.parametrize("patch_id", [PIPELINE_ID, ADAPTER_ID, MODULATION_ID])
    def test_an_unreadable_config_defaults_to_registering(self, patch_id):
        patch = _patches()[patch_id]
        assert patch.condition(types.SimpleNamespace(extra={})) is True
        assert patch.condition(None) is True


class TestImportWeight:
    def test_importing_the_patches_imports_neither_torch_nor_diffusers(self):
        """Discovery runs before anything has decided a run needs them, so a condition that reached
        for the implementation would make an unrelated diffusion job fail at startup on an import
        it never needed."""
        code = (
            "import sys\n"
            "import primus.backends.automodel.patches.minimax_h3_patches\n"
            "assert 'torch' not in sys.modules, 'torch'\n"
            "assert 'diffusers' not in sys.modules, 'diffusers'\n"
            "assert 'nemo_automodel' not in sys.modules, 'nemo_automodel'\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

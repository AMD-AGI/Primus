###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import primus.backends.torchtitan.patches.fsdp_weight_tying_patches as wt_patch
from primus.core.patches.patch_registry import PatchRegistry

PATCH_ID = "torchtitan.fsdp.weight_tying"


class TestFsdpWeightTyingPatchRegistration:
    def test_patch_registered(self):
        assert PATCH_ID in PatchRegistry.list_ids()
        patch = PatchRegistry.get(PATCH_ID)
        assert patch is not None
        assert patch.backend == "torchtitan"
        assert patch.phase == "setup"


class TestFsdpWeightTyingProxy:
    def test_groups_tied_embedding_and_output(self):
        tok = object()
        output = object()
        norm = object()
        model = SimpleNamespace(
            enable_weight_tying=True,
            tok_embeddings=tok,
            output=output,
            norm=norm,
        )

        calls = []

        def orig_fully_shard(target, **kwargs):
            calls.append((_normalize := wt_patch._normalize_modules(target), kwargs))
            return f"wrapped:{_normalize}"

        parallelize_mod = ModuleType("parallelize")
        parallelize_mod.fully_shard = orig_fully_shard

        def orig_apply_fsdp(model, *args, **kwargs):
            parallelize_mod.fully_shard(model.tok_embeddings, mesh="dp")
            parallelize_mod.fully_shard([model.norm, model.output], mesh="dp", reshard=True)
            return "done"

        wrapped = wt_patch._make_apply_fsdp_wrapper(orig_apply_fsdp, parallelize_mod)
        assert wrapped(model) == "done"
        assert calls[0][0] == [tok, output]
        assert calls[1][0] == [norm]

    def test_passthrough_without_weight_tying(self):
        model = SimpleNamespace(enable_weight_tying=False)
        orig_apply_fsdp = MagicMock(return_value="orig")
        parallelize_mod = ModuleType("parallelize")
        parallelize_mod.fully_shard = MagicMock()

        wrapped = wt_patch._make_apply_fsdp_wrapper(orig_apply_fsdp, parallelize_mod)
        assert wrapped(model, mesh="dp") == "orig"
        orig_apply_fsdp.assert_called_once_with(model, mesh="dp")
        parallelize_mod.fully_shard.assert_not_called()


class TestFsdpWeightTyingPatchModules:
    def test_patches_llama4_and_qwen3_without_qwen3_fully_shard(self, monkeypatch):
        llama4_mod = ModuleType("llama4.parallelize")
        qwen3_mod = ModuleType("qwen3.parallelize")

        llama4_mod.apply_fsdp = MagicMock(return_value="orig")
        llama4_mod.fully_shard = MagicMock()
        qwen3_mod.apply_fsdp = llama4_mod.apply_fsdp

        def fake_import(name):
            if name == "torchtitan.models.llama4.infra.parallelize":
                return llama4_mod
            if name == "torchtitan.models.qwen3.infra.parallelize":
                return qwen3_mod
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        monkeypatch.setattr(wt_patch, "log_rank_0", lambda *args, **kwargs: None)

        orig = llama4_mod.apply_fsdp
        wt_patch.patch_torchtitan_fsdp_weight_tying(None)
        assert llama4_mod.apply_fsdp is qwen3_mod.apply_fsdp
        assert llama4_mod.apply_fsdp is not orig
        assert not hasattr(qwen3_mod, "fully_shard")

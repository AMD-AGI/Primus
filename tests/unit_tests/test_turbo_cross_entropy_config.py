# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""CPU dispatch tests; numerical GPU coverage lives in the integration suite."""

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from primus.core.patches import PatchContext
from primus.core.patches.patch_registry import PatchRegistry


def context(**kwargs):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**kwargs))},
    )


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(PatchRegistry, "_all_patches", [])
    monkeypatch.setattr(PatchRegistry, "_patches_by_backend_phase", defaultdict(lambda: defaultdict(list)))
    path = Path(__file__).parents[2] / "primus/backends/megatron/patches/turbo/cross_entropy_patches.py"
    spec = importlib.util.spec_from_file_location("ce_config_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "log_rank_0", lambda *args: None)
    original = Mock(return_value="fallback")
    original._primus_turbo_ce = False
    cls = type("LanguageModule", (), {"compute_language_model_loss": original})
    lm = ModuleType("megatron.core.models.common.language_module.language_module")
    lm.LanguageModule = cls
    monkeypatch.setitem(sys.modules, lm.__name__, lm)
    ce = ModuleType("primus_turbo.pytorch.ops.cross_entropy")
    ce.cross_entropy = Mock(side_effect=lambda logits, target, **kw: torch.zeros_like(target))
    monkeypatch.setitem(sys.modules, ce.__name__, ce)
    return mod, cls, original, ce.cross_entropy


def test_explicit_arguments_override_legacy_environment(patched, monkeypatch):
    mod, *_ = patched
    monkeypatch.setenv("PRIMUS_TURBO_CROSS_ENTROPY", "1")
    assert not mod._enabled(context())
    assert not mod._enabled(context(use_turbo_cross_entropy=False))
    monkeypatch.setenv("PRIMUS_TURBO_CROSS_ENTROPY", "0")
    assert mod._enabled(context(use_turbo_cross_entropy=True))


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_overwrite_argument_and_layout(patched, monkeypatch, overwrite, contiguous):
    mod, cls, original, ce = patched
    monkeypatch.setenv("PRIMUS_TURBO_CE_OVERWRITE_INPUT", str(int(not overwrite)))
    mod.patch_cross_entropy(context(turbo_ce_overwrite_input=overwrite))
    installed = cls.compute_language_model_loss
    mod.patch_cross_entropy(context(turbo_ce_overwrite_input=overwrite))
    assert cls.compute_language_model_loss is installed
    logits = torch.empty(7, 3, 11) if contiguous else torch.empty(3, 7, 11).transpose(0, 1)
    labels = torch.zeros(3, 7, dtype=torch.long)
    model = SimpleNamespace(
        config=SimpleNamespace(tensor_model_parallel_size=1, cross_entropy_loss_fusion=True)
    )
    assert cls.compute_language_model_loss(model, labels, logits).shape == labels.shape
    assert ce.call_args.kwargs["overwrite_input"] == (overwrite and contiguous)
    torch.testing.assert_close(ce.call_args.args[1], labels.T.contiguous())
    original.assert_not_called()


@pytest.mark.parametrize(
    "tp,fused,dtype", [(2, True, torch.float32), (1, False, torch.float32), (1, True, torch.float16)]
)
def test_fallback(patched, tp, fused, dtype):
    mod, cls, original, ce = patched
    mod.patch_cross_entropy(context())
    model = SimpleNamespace(
        config=SimpleNamespace(tensor_model_parallel_size=tp, cross_entropy_loss_fusion=fused)
    )
    assert (
        cls.compute_language_model_loss(model, torch.zeros(3, 7), torch.empty(7, 3, 11, dtype=dtype))
        == "fallback"
    )
    original.assert_called_once()
    ce.assert_not_called()

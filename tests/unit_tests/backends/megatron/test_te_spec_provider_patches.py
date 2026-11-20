###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the Megatron ``megatron.turbo.te_spec_provider`` patch."""

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("megatron")

from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
    PrimusTurboSpecProvider,
)
from primus.backends.megatron.patches.turbo.te_spec_provider_patches import (
    patch_te_spec_provider,
)
from primus.core.patches import PatchContext

# Every Megatron module that binds ``TESpecProvider`` by name.
_PROVIDER_MODULES = [
    "megatron.core.extensions.transformer_engine_spec_provider",
    "megatron.core.models.gpt.gpt_layer_specs",
    "megatron.core.models.gpt.moe_module_specs",
    "megatron.core.transformer.multi_token_prediction",
    "megatron.core.models.gpt.experimental_attention_variant_module_specs",
]


@pytest.fixture
def patched_modules(monkeypatch):
    modules = {name: importlib.import_module(name) for name in _PROVIDER_MODULES}
    for module in modules.values():
        monkeypatch.setattr(module, "TESpecProvider", module.TESpecProvider)  # restored on teardown

    args = SimpleNamespace(enable_primus_turbo=True, use_turbo_grouped_gemm=True)
    ctx = PatchContext(
        backend="megatron", phase="before_train", extra={"module_config": SimpleNamespace(params=args)}
    )
    patch_te_spec_provider(ctx)
    return modules


@pytest.mark.parametrize("module_name", _PROVIDER_MODULES)
def test_every_spec_module_uses_primus_provider(patched_modules, module_name):
    assert patched_modules[module_name].TESpecProvider is PrimusTurboSpecProvider


def test_gated_delta_net_layer_specs_use_primus_provider(patched_modules):
    """Qwen3.5-style hybrid specs build their backend inside this module."""
    module = patched_modules["megatron.core.models.gpt.experimental_attention_variant_module_specs"]
    config = SimpleNamespace(transformer_impl="transformer_engine", use_kitchen=False)
    assert isinstance(module._get_backend_spec_provider(config), PrimusTurboSpecProvider)

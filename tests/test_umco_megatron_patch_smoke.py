from __future__ import annotations

import importlib
import types


def _install_fake_megatron_modules(monkeypatch) -> None:
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    transformer = types.ModuleType("megatron.core.transformer")
    moe = types.ModuleType("megatron.core.transformer.moe")
    token_dispatcher = types.ModuleType("megatron.core.transformer.moe.token_dispatcher")
    moe_layer = types.ModuleType("megatron.core.transformer.moe.moe_layer")

    class FakeDispatcher:
        def __init__(self, *args, **kwargs):
            class Cfg:
                expert_model_parallel_size = 1
                expert_tensor_parallel_size = 1
                pipeline_model_parallel_size = 1

            self.config = Cfg()

        def token_permutation(self, hidden_states, probs, routing_map):
            return hidden_states, [0]

        def token_unpermutation(self, hidden_states, bias=None):
            return hidden_states, bias

    token_dispatcher.MoEFlexTokenDispatcher = FakeDispatcher
    moe_layer.MoEFlexTokenDispatcher = FakeDispatcher

    monkeypatch.setitem(__import__("sys").modules, "megatron", megatron)
    monkeypatch.setitem(__import__("sys").modules, "megatron.core", core)
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.transformer", transformer)
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.transformer.moe", moe)
    monkeypatch.setitem(
        __import__("sys").modules, "megatron.core.transformer.moe.token_dispatcher", token_dispatcher
    )
    monkeypatch.setitem(__import__("sys").modules, "megatron.core.transformer.moe.moe_layer", moe_layer)


def test_apply_umco_patch_smoke(monkeypatch):
    _install_fake_megatron_modules(monkeypatch)
    monkeypatch.setenv("PRIMUS_UMCO_ENABLE", "1")

    patch_mod = importlib.import_module("primus.backends.megatron.moe_umco_patch")
    patch_mod._PATCHED = False
    patch_mod.apply_umco_patches(exp_config=None)

    token_dispatcher = importlib.import_module("megatron.core.transformer.moe.token_dispatcher")
    wrapped_cls = token_dispatcher.MoEFlexTokenDispatcher
    assert wrapped_cls.__name__.startswith("UmcoWrapped")

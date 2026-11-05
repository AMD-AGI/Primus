# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""``megatron.moe.primus_topk_router``: its router swap rule, and the GPT MoE spec on Megatron-core 0.19+.

From core 0.19 the MoE spec is ``functools.partial(MoELayer, submodules=MoESubmodules(...))`` with the
router left at the dataclass default, which the patch's field-default edit does not reach.
"""

import functools
import sys
from types import SimpleNamespace

import pytest

# Imported before any test runs the patch, so they keep Megatron's own spec builders and TopKRouter.
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt import gpt_layer_specs, moe_module_specs  # noqa: F401
from megatron.core.transformer.moe import moe_layer
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec

from primus.backends.megatron.core.transformer.moe.router import (
    PrimusTopKRouter,
    primus_topk_router_for,
)
from primus.backends.megatron.patches.moe_patches import topk_router_patches
from primus.core.patches.context import PatchContext

# Megatron's own class: the patch rebinds the name ``TopKRouter`` inside Megatron's modules.
TopKRouter = PrimusTopKRouter.__bases__[0]


def test_swap_rule():
    assert TopKRouter.__name__ == "TopKRouter"
    assert primus_topk_router_for(TopKRouter) is PrimusTopKRouter
    for router in (PrimusTopKRouter, InferenceTopKRouter, functools.partial(TopKRouter)):
        assert primus_topk_router_for(router) is router
    # matched by name, so a TopKRouter some module bound before the patch ran is caught too
    assert primus_topk_router_for(type("TopKRouter", (), {})) is PrimusTopKRouter


@pytest.mark.parametrize("router", [TopKRouter, InferenceTopKRouter], ids=lambda r: r.__name__)
@pytest.mark.parametrize("as_partial", [True, False], ids=["partial_core_v0.19", "module_spec"])
def test_spec_router(router, as_partial):
    submodules = moe_layer.MoESubmodules(experts=None, router=router)
    if as_partial:
        spec = functools.partial(moe_layer.MoELayer, submodules=submodules)
    else:
        spec = ModuleSpec(module=moe_layer.MoELayer, submodules=submodules)
    assert topk_router_patches._use_primus_topk_router_in_moe_spec(spec) is spec
    assert submodules.router is (PrimusTopKRouter if router is TopKRouter else router)


def _run_patch(mp):
    """Run the patch under ``mp``, which restores every global the patch rebinds.

    The real ``MoELayer.__init__`` needs process groups, so the patch wraps a stand-in that records
    the router it is handed; the recorded routers are returned.
    """
    mp.setattr(topk_router_patches, "log_rank_0", lambda *args, **kwargs: None)
    router_module = sys.modules["megatron.core.transformer.moe.router"]
    mp.setattr(router_module, "TopKRouter", router_module.TopKRouter)
    mp.setattr(moe_layer, "TopKRouter", moe_layer.TopKRouter)
    field = moe_layer.MoESubmodules.__dataclass_fields__["router"]
    mp.setattr(field, "default", field.default)
    mp.setattr(field, "default_factory", field.default_factory)
    for name in ("get_moe_module_spec_for_backend", "get_moe_module_spec"):
        mp.setattr(moe_module_specs, name, getattr(moe_module_specs, name))
    routers = []
    mp.setattr(
        moe_layer.MoELayer,
        "__init__",
        lambda self, config, submodules=None, *args, **kwargs: routers.append(submodules.router),
    )
    module_config = SimpleNamespace(params=SimpleNamespace(disable_primus_topk_router=False))
    topk_router_patches.patch_primus_topk_router(
        PatchContext(backend="megatron", phase="build_args", extra={"module_config": module_config})
    )
    return routers


@pytest.fixture
def patched(monkeypatch):
    return _run_patch(monkeypatch)


def test_gpt_moe_spec_gets_primus_router(patched):
    for spec in (
        moe_module_specs.get_moe_module_spec_for_backend(backend=LocalSpecProvider(), num_experts=4),
        moe_module_specs.get_moe_module_spec(use_te=False, num_experts=4),
    ):
        submodules = spec.keywords["submodules"] if isinstance(spec, functools.partial) else spec.submodules
        assert submodules.router is PrimusTopKRouter


def test_moe_layer_init_swaps_only_megatron_topk_router(patched):
    for router in (TopKRouter, InferenceTopKRouter, PrimusTopKRouter):
        moe_layer.MoELayer(SimpleNamespace(), moe_layer.MoESubmodules(experts=None, router=router))
    assert patched == [PrimusTopKRouter, InferenceTopKRouter, PrimusTopKRouter]


def test_patch_globals_restored():
    # other tests build plain Megatron MoE layers in the same process
    field = moe_layer.MoESubmodules.__dataclass_fields__["router"]
    before = (
        moe_layer.MoELayer.__init__,
        moe_layer.TopKRouter,
        field.default,
        moe_module_specs.get_moe_module_spec_for_backend,
        moe_module_specs.get_moe_module_spec,
    )
    with pytest.MonkeyPatch.context() as mp:
        _run_patch(mp)
        assert moe_layer.TopKRouter is PrimusTopKRouter
    assert before == (
        moe_layer.MoELayer.__init__,
        moe_layer.TopKRouter,
        field.default,
        moe_module_specs.get_moe_module_spec_for_backend,
        moe_module_specs.get_moe_module_spec,
    )

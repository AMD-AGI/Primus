###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Applies the deferred output-layer wgrad patch to the real Megatron modules."""

import ast
import inspect
import sys
from types import ModuleType

import pytest

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import megatron.core.pipeline_parallel.schedules as schedules  # noqa: E402
from megatron.core.distributed.param_and_grad_buffer import (  # noqa: E402
    _ParamAndGradBucketGroup,
)

import primus.backends.megatron.core.pipeline_parallel.primuspipe.pipeline_launcher as primuspipe  # noqa: E402

# Importing the package also imports every *_patches module, which the wrapper-order test scans.
from primus.backends.megatron.patches import (  # noqa: E402
    defer_embedding_wgrad_patches as patch_mod,
)

_SCHEDULE_HELPERS = ("clear_embedding_activation_buffer", "finish_embedding_wgrad_compute")
_module_dict = ModuleType.__dict__["__dict__"].__get__


def _modules_binding(name, value):
    return [
        mod
        for mod in list(sys.modules.values())
        if issubclass(type(mod), ModuleType) and _module_dict(mod).get(name) is value
    ]


@pytest.fixture
def originals(monkeypatch):
    """Snapshot every binding the patch touches so monkeypatch restores them."""
    saved = {name: getattr(schedules, name) for name in _SCHEDULE_HELPERS}
    for name, value in saved.items():
        for mod in _modules_binding(name, value):
            monkeypatch.setattr(mod, name, value)
    saved["start_grad_sync"] = _ParamAndGradBucketGroup.start_grad_sync
    monkeypatch.setattr(_ParamAndGradBucketGroup, "start_grad_sync", saved["start_grad_sync"])
    monkeypatch.setattr(patch_mod, "log_rank_0", lambda *a, **k: None)
    return saved


def test_upstream_helpers_keep_the_patched_signatures():
    assert list(inspect.signature(schedules.clear_embedding_activation_buffer).parameters) == [
        "config",
        "model",
        "is_last_stage",
    ]
    assert list(inspect.signature(schedules.finish_embedding_wgrad_compute).parameters) == [
        "config",
        "embedding_module",
        "is_last_stage",
        "tp_group",
    ]
    assert "self.params = set()" in inspect.getsource(_ParamAndGradBucketGroup.__init__)


def test_patch_rebinds_the_real_modules(originals):
    patch_mod.patch_defer_embedding_wgrad_grad_sync(None)

    for name in _SCHEDULE_HELPERS:
        assert getattr(schedules, name).__wrapped__ is originals[name]
        assert getattr(primuspipe, name) is getattr(schedules, name)
        assert _modules_binding(name, originals[name]) == []
    start_grad_sync = _ParamAndGradBucketGroup.start_grad_sync
    assert start_grad_sync.__wrapped__ is originals["start_grad_sync"]
    assert getattr(start_grad_sync, patch_mod._PATCHED_ATTR)

    patch_mod.patch_defer_embedding_wgrad_grad_sync(None)
    assert _ParamAndGradBucketGroup.start_grad_sync is start_grad_sync


def _value(module, node):
    return getattr(module, node.id) if isinstance(node, ast.Name) else ast.literal_eval(node)


def _start_grad_sync_patches():
    """(id, priority) of every Megatron before_train patch whose module touches start_grad_sync.

    Read from the ``@register_patch`` calls rather than PatchRegistry, which other
    tests clear and fill with throwaway patches.
    """
    found = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("primus.backends.megatron.patches.") or not issubclass(
            type(module), ModuleType
        ):
            continue
        source = inspect.getsource(module)
        if "start_grad_sync" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_patch":
                kwargs = {kw.arg: _value(module, kw.value) for kw in node.keywords}
                if kwargs.get("backend") == "megatron" and kwargs.get("phase") == "before_train":
                    found[_value(module, node.args[0])] = kwargs.get("priority", 50)
    return found


def test_wraps_outside_every_other_start_grad_sync_patch():
    patches = _start_grad_sync_patches()
    ours = patches.pop(patch_mod._PATCH_KEY)
    assert set(patches) >= {
        "megatron.distributed.rccl_sdma_grad_reduce_scatter",
        "megatron.core.distributed.grad_buffer_ownership",
    }
    assert all(priority < ours for priority in patches.values()), patches

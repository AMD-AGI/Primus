###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from collections import OrderedDict

import pytest

from primus.backends.megatron.patches.turbo.qk_rmsnorm_rope_patches import (
    _DDP_PARAM_GATHER_HOOK_ATTR,
    _install_ddp_param_gather_hook_marker,
    _run_forward_pre_hooks,
    _supported_forward_pre_hooks,
)


class _FakeModule:
    def __init__(self):
        self._forward_pre_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._backward_pre_hooks = OrderedDict()
        self._backward_hooks = OrderedDict()


def test_ddp_param_gather_hook_is_tagged_and_replayed():
    calls = []

    class FakeDDP:
        def _make_forward_pre_hook(self):
            def hook(module, inputs):
                calls.append((module, inputs))

            return hook

    _install_ddp_param_gather_hook_marker(FakeDDP)
    _install_ddp_param_gather_hook_marker(FakeDDP)

    module = _FakeModule()
    hook = FakeDDP()._make_forward_pre_hook()
    module._forward_pre_hooks[1] = hook

    assert getattr(hook, _DDP_PARAM_GATHER_HOOK_ATTR) is True
    hooks = _supported_forward_pre_hooks(module)
    assert hooks == (hook,)

    _run_forward_pre_hooks(module, hooks)
    assert calls == [(module, ())]


@pytest.mark.parametrize(
    "hook_collection",
    ["_forward_hooks", "_backward_pre_hooks", "_backward_hooks"],
)
def test_non_pre_hooks_disable_fusion(hook_collection):
    module = _FakeModule()
    getattr(module, hook_collection)[1] = lambda *args: None

    assert _supported_forward_pre_hooks(module) is None


def test_unmarked_forward_pre_hook_disables_fusion():
    module = _FakeModule()
    module._forward_pre_hooks[1] = lambda *args: None

    assert _supported_forward_pre_hooks(module) is None


def test_ddp_param_gather_hook_must_not_modify_inputs():
    module = _FakeModule()

    def hook(_module, _inputs):
        return ("modified",)

    setattr(hook, _DDP_PARAM_GATHER_HOOK_ATTR, True)

    with pytest.raises(RuntimeError, match="unexpectedly modified"):
        _run_forward_pre_hooks(module, (hook,))

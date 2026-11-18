###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""run_forward_pre_hooks matches nn.Module.__call__ for forward pre-hooks."""

import pytest
import torch

from primus.backends.megatron.core.transformer.muse_spark.forward_pre_hooks import (
    run_forward_pre_hooks,
)


class _Echo(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return args, kwargs


def _hooked():
    m = _Echo()
    calls = []

    def plain(mod, args):
        calls.append("plain")
        return (args[0] + 1,)

    def with_kw(mod, args, kwargs):
        calls.append("kw")
        return (args[0] * 2,), {**kwargs, "tag": 1}

    def none_hook(mod, args):
        calls.append("none")

    m.register_forward_pre_hook(plain)
    m.register_forward_pre_hook(with_kw, with_kwargs=True)
    m.register_forward_pre_hook(none_hook)
    return m, calls


def test_matches_module_call():
    x = torch.ones(2)
    m, calls = _hooked()
    ref_args, ref_kwargs = m(x)
    ref_calls = list(calls)
    calls.clear()
    args, kwargs = run_forward_pre_hooks(m, (x,))
    assert calls == ref_calls == ["plain", "kw", "none"]
    assert torch.equal(args[0], ref_args[0]) and kwargs == ref_kwargs == {"tag": 1}


def test_single_tensor_args_and_no_hooks():
    x = torch.zeros(3)
    args, kwargs = run_forward_pre_hooks(_Echo(), x)
    assert args[0] is x and kwargs == {}


def test_bad_kwargs_hook_return_raises():
    m = _Echo()
    m.register_forward_pre_hook(lambda mod, a, k: a, with_kwargs=True)
    with pytest.raises(RuntimeError):
        run_forward_pre_hooks(m, (torch.ones(1), torch.ones(1), torch.ones(1)))


def test_muse_spark_post_expert_norm_hooks_return_rewritten_input():
    pytest.importorskip("megatron")
    from primus.backends.megatron.core.transformer.muse_spark.moe import (
        _run_forward_pre_hooks,
    )

    m, _ = _hooked()
    with pytest.raises(RuntimeError):  # the kwargs hook adds a kwarg
        _run_forward_pre_hooks(m, torch.ones(1))
    n = _Echo()
    n.register_forward_pre_hook(lambda mod, a: (a[0] + 1,))
    assert torch.equal(_run_forward_pre_hooks(n, torch.ones(1)), torch.full((1,), 2.0))

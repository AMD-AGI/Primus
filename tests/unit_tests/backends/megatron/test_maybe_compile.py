###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``maybe_compile``: which errors fall back to eager, and that the fallback is logged."""

import pytest
import torch

from primus.backends.megatron.core import utils


def _compile_that_raises(exc):
    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            raise exc

        return compiled

    return fake_compile


@pytest.mark.parametrize("exc_name", ["TorchDynamoException", "FailOnRecompileLimitHit"])
def test_compile_errors_fall_back_to_eager_and_are_logged_once(monkeypatch, exc_name):
    exc_type = getattr(torch._dynamo.exc, exc_name, None)
    if exc_type is None:
        pytest.skip(f"this torch has no {exc_name}")
    monkeypatch.setattr(torch, "compile", _compile_that_raises(exc_type("cannot compile")))
    logged = []
    monkeypatch.setattr(utils, "warning_rank_0", logged.append)

    def add_one(x):
        return x + 1

    wrapped = utils.maybe_compile(add_one)
    assert wrapped.compiled_ok()
    torch.testing.assert_close(wrapped(torch.ones(2)), torch.full((2,), 2.0))
    torch.testing.assert_close(wrapped(torch.ones(2)), torch.full((2,), 2.0))
    assert not wrapped.compiled_ok()
    assert len(logged) == 1 and "add_one" in logged[0] and "cannot compile" in logged[0]


def test_other_errors_propagate_and_keep_the_compiled_path(monkeypatch):
    monkeypatch.setattr(torch, "compile", _compile_that_raises(torch.OutOfMemoryError("out of memory")))
    logged = []
    monkeypatch.setattr(utils, "warning_rank_0", logged.append)
    wrapped = utils.maybe_compile(lambda x: x)
    with pytest.raises(torch.OutOfMemoryError):
        wrapped(torch.ones(2))
    assert wrapped.compiled_ok() and not logged


def test_a_real_dynamo_failure_falls_back_to_eager(monkeypatch):
    logged = []
    monkeypatch.setattr(utils, "warning_rank_0", logged.append)

    def breaks_the_graph(x):
        torch._dynamo.graph_break()
        return x * 2

    wrapped = utils.maybe_compile(breaks_the_graph)
    torch.testing.assert_close(wrapped(torch.ones(3)), torch.full((3,), 2.0))
    assert not wrapped.compiled_ok() and len(logged) == 1

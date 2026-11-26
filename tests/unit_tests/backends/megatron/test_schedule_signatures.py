###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Primus replacements of ``get_forward_backward_func`` must take whatever upstream passes."""

import inspect
from types import SimpleNamespace

import megatron.core.pipeline_parallel as pipeline_parallel
import pytest

from primus.backends.megatron.patches.deepseek_v4_get_batch_patches import (
    _make_v4_get_fbf,
)
from primus.backends.megatron.patches.pp_dump_data_patches import (
    _make_wrapped_get_forward_backward_func,
)

_UPSTREAM = inspect.signature(pipeline_parallel.get_forward_backward_func)
_UPSTREAM_KWARGS = {name: object() for name in _UPSTREAM.parameters}


def _primus_schedule_factories():
    from primus.backends.megatron.core.pipeline_parallel import schedules

    return [
        schedules.get_forward_backward_func_zbpp,
        schedules.get_primus_pipeline_parallel_fwd_backward_func,
    ]


@pytest.mark.parametrize("factory", _primus_schedule_factories(), ids=lambda f: f.__qualname__)
def test_factory_takes_upstream_arguments(factory):
    inspect.signature(factory).bind(**_UPSTREAM_KWARGS)


_WRAPPERS = {
    "pp_dump": _make_wrapped_get_forward_backward_func,
    "deepseek_v4": lambda original: _make_v4_get_fbf(original, original_get_batch=None),
}


@pytest.mark.parametrize("name", sorted(_WRAPPERS))
def test_wrapper_forwards_upstream_arguments(name, monkeypatch):
    import megatron.training

    monkeypatch.setattr(megatron.training, "get_args", lambda: SimpleNamespace(dump_pp_data=False))
    calls = []

    def schedule(*args, **kwargs):
        return None

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return schedule

    func = _WRAPPERS[name](original)(**_UPSTREAM_KWARGS)

    assert calls == [((), _UPSTREAM_KWARGS)]
    assert callable(func)


def test_primuspipe_takes_custom_process_groups():
    """Megatron-core 0.19 has no ModelType.encoder_and_decoder to check a custom pg_collection against."""
    from primus.backends.megatron.core.pipeline_parallel.primuspipe.pipeline_launcher import (
        PrimusPipelineParallelLauncher,
    )

    groups = SimpleNamespace(tp=None, cp=None, embd=None, pos_embd=None, pp=None, dp_cp=None)
    assert PrimusPipelineParallelLauncher._init_check_pg_collection(None, groups) is groups

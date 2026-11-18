###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V-schedule ``get_batch_on_this_tp_rank`` wrapper: PP rank 0 also runs the loss chunk."""

from primus.backends.megatron.patches.parallelism.v_schedule_patches import (
    _make_v_schedule_get_batch_on_this_tp_rank,
)


def _recorder():
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "batch"

    return original, calls


def test_first_stage_keeps_every_field():
    original, calls = _recorder()
    wrapped = _make_v_schedule_get_batch_on_this_tp_rank(original)

    assert wrapped({}, pipeline_model_parallel_size=4, is_pipeline_first_stage=True) == "batch"
    assert calls == [(({},), {"pipeline_model_parallel_size": 1, "is_pipeline_first_stage": True})]


def test_other_stages_pass_through():
    original, calls = _recorder()
    wrapped = _make_v_schedule_get_batch_on_this_tp_rank(original)

    kwargs = {"pipeline_model_parallel_size": 4, "is_pipeline_first_stage": False}
    wrapped({}, **kwargs)
    wrapped({})
    assert calls == [(({},), kwargs), (({},), {})]


def test_wrapper_is_marked():
    original, _ = _recorder()
    assert _make_v_schedule_get_batch_on_this_tp_rank(original)._primus_v_schedule is True

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.training import evaluator
from primus.backends.megatron.training.evaluator import primus_evaluate


def test_accepts_every_upstream_evaluate_argument():
    import megatron.training.training as megatron_training

    upstream = getattr(megatron_training.evaluate, "__wrapped__", megatron_training.evaluate)
    if upstream is primus_evaluate:
        return
    ours = inspect.signature(primus_evaluate).parameters
    missing = [name for name in inspect.signature(upstream).parameters if name not in ours]
    assert not missing, f"primus_evaluate does not accept {missing}"


class _ScheduleRan(Exception):
    pass


class _Timers:
    def __call__(self, name, log_level=0):
        return SimpleNamespace(start=lambda barrier=False: None, stop=lambda: None)

    def log(self, names):
        pass


def test_full_iteration_cuda_graph_wraps_the_eval_schedule(monkeypatch):
    """Megatron-core 0.19 selects full-iteration CUDA graphs with cuda_graph_impl."""
    seen = {}

    def get_forward_backward_func(**kwargs):
        seen["schedule_kwargs"] = kwargs
        return "schedule"

    class FullCudaGraphWrapper:
        def __init__(self, forward_backward_func, cuda_graph_warmup_steps, use_single_mempool):
            seen["wrapped"] = (forward_backward_func, cuda_graph_warmup_steps, use_single_mempool)

        def __call__(self, **kwargs):
            raise _ScheduleRan

    # validate_args deletes enable_cuda_graph, and 0.19 has no cuda_graph_scope.
    args = SimpleNamespace(
        cuda_graph_impl="full_iteration",
        cuda_graph_warmup_steps=3,
        vision_pretraining=False,
        global_batch_size=2,
        micro_batch_size=1,
        data_parallel_size=1,
        eval_iters=1,
        seq_length=8,
        decoder_seq_length=None,
    )
    rerun_state_machine = SimpleNamespace(get_mode=lambda: None, set_mode=lambda mode: None)
    monkeypatch.setattr(evaluator, "get_args", lambda: args)
    monkeypatch.setattr(evaluator, "get_timers", _Timers)
    monkeypatch.setattr(evaluator, "get_rerun_state_machine", lambda: rerun_state_machine)
    monkeypatch.setattr(evaluator, "get_forward_backward_func", get_forward_backward_func)
    monkeypatch.setattr(evaluator, "FullCudaGraphWrapper", FullCudaGraphWrapper)
    monkeypatch.setattr(
        evaluator,
        "ft_integration",
        SimpleNamespace(on_eval_step_start=lambda: None, on_eval_step_end=lambda: None),
    )

    pg_collection = object()
    config = SimpleNamespace(cuda_graph_use_single_mempool=True, timers=None)
    with pytest.raises(_ScheduleRan):
        primus_evaluate(None, None, [torch.nn.Linear(1, 1)], None, config, pg_collection=pg_collection)

    assert seen == {
        "schedule_kwargs": {"schedule_pg_collection": pg_collection},
        "wrapped": ("schedule", 3, True),
    }

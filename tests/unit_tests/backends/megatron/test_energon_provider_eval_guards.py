# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests that the Energon provider's evaluation guards refuse before building.

The provider wraps validation-dataloader construction in a bare
``except Exception`` that logs, sets ``eval_iters = 0`` and carries on, so a
check that runs inside it does not refuse the run -- it turns the run into one
that evaluates nothing and exits 0. That is the same swallow that left
``eval_iters`` at 0 when the ``build_args`` patch raised. The guards therefore
have to run ahead of that block, and this pins that they do.
"""

from types import SimpleNamespace

import pytest

import primus.backends.megatron.data.energon_dataset_provider as provider_module
from primus.backends.megatron.data.energon_dataset_provider import (
    EnergonDatasetProvider,
)
from primus.backends.megatron.training.eval_budget import EvalCoverageError

# The MLPerf Flux shape: 58 iterations x 1 microbatch x 64 x DP 8 = 29,696.
DATA_PARALLEL_SIZE = 8


def _args(**overrides):
    base = dict(
        skip_train=True,
        eval_iters=58,
        eval_samples=None,
        micro_batch_size=64,
        global_batch_size=512,
        data_parallel_size=DATA_PARALLEL_SIZE,
        val_num_workers=0,
        num_workers=4,
        prefetch_factor=2,
        mlperf_mode=True,
        eval_timestep_source="dataset",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def build_dataloaders(monkeypatch):
    """Drive create_dataloaders far enough to reach the guards.

    Everything stubbed here is either rank plumbing or an Energon entry point;
    the argument handling and the guards themselves are the real code.
    """
    import megatron.training

    val_dataset_calls = []

    def _stub(args, val_datasets_result=None):
        monkeypatch.setattr(megatron.training, "get_args", lambda: args)
        monkeypatch.setattr(EnergonDatasetProvider, "_is_dataloader_rank", lambda self: True)
        monkeypatch.setattr(
            EnergonDatasetProvider, "_create_worker_config", lambda self, args, num_workers=None: object()
        )
        monkeypatch.setattr(EnergonDatasetProvider, "_get_data_path", lambda self, args: "/dev/null/dataset")
        monkeypatch.setattr(
            provider_module.parallel_state, "get_data_parallel_world_size", lambda: DATA_PARALLEL_SIZE
        )

        def _get_val_datasets(*call_args, **call_kwargs):
            val_dataset_calls.append(call_kwargs)
            if isinstance(val_datasets_result, Exception):
                raise val_datasets_result
            return val_datasets_result or []

        monkeypatch.setattr(provider_module, "get_val_datasets", _get_val_datasets)

        provider = EnergonDatasetProvider(task_encoder_factory=lambda: object())
        return provider.create_dataloaders(trainer_config=None, train_val_test_num_samples=[0, 0, 0])

    _stub.val_dataset_calls = val_dataset_calls
    return _stub


class TestTimestepSourceGuard:
    def test_an_mlperf_run_on_injected_timesteps_never_reaches_construction(self, build_dataloaders):
        args = _args(eval_timestep_source="equidistant")

        with pytest.raises(EvalCoverageError, match="eval_timestep_source is 'equidistant'"):
            build_dataloaders(args)

        assert build_dataloaders.val_dataset_calls == []
        # The refusal has to leave the budget alone. Zeroing it here is what
        # the swallowing handler does, and it is what turns a refusal into a
        # run that reports nothing and exits 0.
        assert args.eval_iters == 58

    def test_dataset_timesteps_proceed_to_construction(self, build_dataloaders):
        _, valid_dataloaders, _ = build_dataloaders(_args())

        assert len(build_dataloaders.val_dataset_calls) == 1
        assert valid_dataloaders == []


class TestGuardsRunAheadOfTheSwallow:
    def test_a_construction_failure_is_still_swallowed(self, build_dataloaders):
        """Not a defect to fix here, but the reason guard placement matters.

        A dataset with no validation split has to degrade rather than abort,
        which is why the handler exists. It cannot tell that case apart from
        a misconfiguration, so anything raised inside it disappears.
        """
        args = _args()

        _, valid_dataloaders, _ = build_dataloaders(args, val_datasets_result=RuntimeError("no val split"))

        assert valid_dataloaders is None
        assert args.eval_iters == 0

    def test_the_zero_eval_iters_guard_also_refuses_before_construction(self, build_dataloaders):
        """The guard this PR added for a swallowed build_args patch failure."""
        args = _args(eval_samples=29696, eval_iters=0)

        with pytest.raises(EvalCoverageError, match="no evaluation would run"):
            build_dataloaders(args)

        assert build_dataloaders.val_dataset_calls == []

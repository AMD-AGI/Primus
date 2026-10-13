# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
The training dataset is built without an EpochizeDataset unless a recipe asks for one.

A nonzero virtual_epoch_length makes every data-loader worker build
list(range(virtual_epoch_length // num_workers)) before its first sample, so a large
"never ends" value costs worker start-up time and memory while changing nothing about the
samples: the stream repeats either way. The provider therefore falls back to Energon's own
default, 0, and still passes an explicit recipe value through.
"""

from types import SimpleNamespace

import pytest

import primus.backends.megatron.data.energon_dataset_provider as provider_module
from primus.backends.megatron.data.energon_dataset_provider import (
    EnergonDatasetProvider,
)


def _args(**overrides):
    base = dict(
        skip_train=False,
        eval_iters=0,
        eval_samples=None,
        micro_batch_size=32,
        global_batch_size=1024,
        data_parallel_size=32,
        val_num_workers=0,
        num_workers=4,
        prefetch_factor=2,
        mlperf_mode=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def train_dataset_kwargs(monkeypatch):
    """Run create_dataloaders with Energon stubbed; return the kwargs get_train_dataset received."""
    import megatron.training

    calls = []

    def _run(args):
        monkeypatch.setattr(megatron.training, "get_args", lambda: args)
        monkeypatch.setattr(EnergonDatasetProvider, "_is_dataloader_rank", lambda self: True)
        monkeypatch.setattr(
            EnergonDatasetProvider, "_create_worker_config", lambda self, args, **kw: object()
        )
        monkeypatch.setattr(EnergonDatasetProvider, "_get_data_path", lambda self, args: "/dev/null/dataset")
        monkeypatch.setattr(provider_module.parallel_state, "get_data_parallel_world_size", lambda: 32)
        monkeypatch.setattr(provider_module, "get_val_datasets", lambda *a, **k: [])

        def _get_train_dataset(*a, **kw):
            calls.append(kw)
            return object()

        monkeypatch.setattr(provider_module, "get_train_dataset", _get_train_dataset)
        monkeypatch.setattr(provider_module, "get_savable_loader", lambda dataset, **kw: object())
        provider = EnergonDatasetProvider(task_encoder_factory=lambda: object())
        provider.create_dataloaders(trainer_config=None, train_val_test_num_samples=[0, 0, 0])
        assert len(calls) == 1
        return calls[0]

    return _run


def test_default_is_energon_default_without_epochize(train_dataset_kwargs):
    assert train_dataset_kwargs(_args())["virtual_epoch_length"] == 0


def test_an_explicit_recipe_value_is_passed_through(train_dataset_kwargs):
    assert train_dataset_kwargs(_args(virtual_epoch_length=4096))["virtual_epoch_length"] == 4096

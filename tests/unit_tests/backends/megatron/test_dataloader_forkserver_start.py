# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
DataLoader workers under forkserver: concurrent start and an eagerly started daemon.

Stock popen_forkserver.Popen._launch returns only after the new child has read (that is,
unpickled) its process object, so DataLoader's start loop unpickles its workers one after
another. The patch moves that write to a background thread. These tests pin that the
workers still receive their datasets intact and deliver every batch, that their unpickling
now overlaps, and that the forkserver daemon is running once the preload has been set.
"""

import multiprocessing
import time

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from primus.backends.megatron.patches import dataloader_patch

UNPICKLE_SECONDS = 1.0
NUM_WORKERS = 4


class _SlowToUnpickle(IterableDataset):
    """Carries a payload larger than a pipe buffer and takes UNPICKLE_SECONDS to unpickle."""

    def __init__(self, n_items):
        self.n_items = n_items
        self.payload = list(range(200_000))

    def __setstate__(self, state):
        time.sleep(UNPICKLE_SECONDS)
        self.__dict__.update(state)

    def __iter__(self):
        info = get_worker_info()
        for i in range(info.id, self.n_items, info.num_workers):
            yield torch.tensor([i, len(self.payload)])


def _loader(dataset):
    return DataLoader(dataset, batch_size=None, num_workers=NUM_WORKERS, multiprocessing_context="forkserver")


@pytest.fixture(scope="module", autouse=True)
def concurrent_launch():
    from multiprocessing import popen_forkserver

    original = popen_forkserver.Popen._launch
    # As in training: the daemon preloads torch, so a worker's own start-up is short next to UNPICKLE_SECONDS.
    dataloader_patch._preload_forkserver_torch()
    assert dataloader_patch._install_concurrent_forkserver_launch()
    yield
    popen_forkserver.Popen._launch = original


def test_every_batch_arrives_with_the_dataset_intact():
    items = sorted(int(t[0]) for t in _loader(_SlowToUnpickle(40)))
    assert items == list(range(40))
    assert all(int(t[1]) == 200_000 for t in _loader(_SlowToUnpickle(8)))


def test_workers_unpickle_concurrently():
    multiprocessing.get_context("forkserver")  # the forkserver itself is warm after the first test
    start = time.perf_counter()
    iterator = iter(_loader(_SlowToUnpickle(NUM_WORKERS)))
    next(iterator)
    elapsed = time.perf_counter() - start
    # One after another this takes at least NUM_WORKERS * UNPICKLE_SECONDS.
    assert elapsed < (NUM_WORKERS - 1) * UNPICKLE_SECONDS
    del iterator


def test_install_is_idempotent():
    from multiprocessing import popen_forkserver

    installed = popen_forkserver.Popen._launch
    assert dataloader_patch._install_concurrent_forkserver_launch()
    assert popen_forkserver.Popen._launch is installed


def test_preload_starts_the_forkserver_daemon():
    from multiprocessing import forkserver

    dataloader_patch._preload_forkserver_torch()
    assert forkserver._forkserver._forkserver_pid is not None

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import functools
import pickle
from types import SimpleNamespace

import megatron.training as megatron_training
import pytest

from primus.backends.megatron.patches import dataloader_patch


def build_pretraining_data_loader():
    def worker_init_fn(_):
        pass

    return worker_init_fn


@pytest.mark.parametrize("handler, signal", [(False, None), (True, 15)])
def test_megatron_worker_init_fn_becomes_picklable(monkeypatch, handler, signal):
    args = SimpleNamespace(exit_signal_handler=handler, exit_signal=15)
    monkeypatch.setattr(megatron_training, "get_args", lambda: args)
    local = build_pretraining_data_loader()
    with pytest.raises((AttributeError, pickle.PicklingError)):
        pickle.dumps(local)

    replaced = dataloader_patch._picklable_worker_init_fn(local)

    assert isinstance(replaced, functools.partial)
    assert replaced.func is dataloader_patch._megatron_worker_init
    assert replaced.keywords == {"exit_signal": signal}
    pickle.dumps(replaced)


def test_other_worker_init_fns_are_kept():
    def worker_init_fn(_):
        pass

    assert dataloader_patch._picklable_worker_init_fn(worker_init_fn) is worker_init_fn
    assert dataloader_patch._picklable_worker_init_fn(None) is None

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""light_log_sync_events: the last deferred training_log is written however train() ends (return, sys.exit, an
exception), and a failing flush does not hide the exception that ended training."""

import sys

import pytest

from primus.backends.megatron.patches.light_log_sync_events_patches import (
    _flushing_train,
)


class _Deferred:
    def __init__(self, fail=False):
        self.flushes, self.fail = 0, fail

    def flush(self):
        self.flushes += 1
        if self.fail:
            raise RuntimeError("flush failed")


def test_flush_on_return():
    d = _Deferred()
    assert _flushing_train(lambda x: x + 1, d)(1) == 2
    assert d.flushes == 1


def test_flush_on_sys_exit():
    d = _Deferred()
    with pytest.raises(SystemExit):
        _flushing_train(lambda: sys.exit(0), d)()
    assert d.flushes == 1


def test_flush_on_exception_keeps_the_exception():
    d = _Deferred(fail=True)

    def boom():
        raise ValueError("training failed")

    with pytest.raises(ValueError, match="training failed"):
        _flushing_train(boom, d)()
    assert d.flushes == 1

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""light_log_sync_events: the last deferred training_log is written when train() returns or exits through sys.exit
(paths every rank takes together); on any other exception it is dropped, never flushed (its collectives would hold the
failing rank until the distributed timeout), and a failing flush does not hide the exit."""

import sys

import pytest

from primus.backends.megatron.patches.light_log_sync_events_patches import (
    _flushing_train,
)


class _Deferred:
    def __init__(self, fail=False):
        self.flushes, self.fail = 0, fail
        self.pending = object()

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


def test_failing_flush_keeps_the_exit():
    d = _Deferred(fail=True)
    with pytest.raises(SystemExit):
        _flushing_train(lambda: sys.exit(3), d)()
    assert d.flushes == 1


def test_no_flush_on_a_rank_local_exception():
    d = _Deferred()

    def boom():
        raise ValueError("training failed")

    with pytest.raises(ValueError, match="training failed"):
        _flushing_train(boom, d)()
    assert d.flushes == 0 and d.pending is None

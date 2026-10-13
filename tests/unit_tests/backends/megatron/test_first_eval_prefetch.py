# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
The first evaluation gets its first batch prefetched as training starts.

Later evaluations get theirs from the end of the previous evaluation. The first one used to
start its loader and read its first batch inside the evaluation. The train() wrapper installed
with the evaluate patch starts the same prefetch for it, and the evaluation then takes the batch
through _take_val_prefetch, so it sees the stream exactly as before.
"""

import megatron.training.training as mt
import pytest

from primus.backends.megatron.training import evaluator


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    evaluator._val_prefetch_state.clear()
    monkeypatch.setattr(evaluator, "_val_prefetch_enabled", lambda: True)
    yield
    evaluator._val_prefetch_state.clear()


def test_prefetched_batch_is_the_first_one_the_evaluation_reads():
    stream = iter(range(10))
    evaluator.start_first_eval_prefetch(stream)
    taken, _ = evaluator._take_val_prefetch(stream)
    assert [next(taken) for _ in range(3)] == [0, 1, 2]


def test_no_iterator_is_a_no_op():
    evaluator.start_first_eval_prefetch(None)
    assert evaluator._val_prefetch_state == {}


def test_train_wrapper_prefetches_the_validation_iterator(monkeypatch):
    from primus.backends.megatron.patches import evaluate_patches

    seen = {}

    def fake_train(*args, **kwargs):
        seen["state"] = dict(evaluator._val_prefetch_state)
        return "trained"

    # Both module attributes are restored by monkeypatch after the test.
    monkeypatch.setattr(mt, "train", fake_train)
    monkeypatch.setattr(mt, "evaluate", mt.evaluate)
    evaluate_patches.patch_evaluate(ctx=None)

    valid = iter(range(5))
    assert mt.train(None, None, None, None, iter([]), valid, None, None, None, None) == "trained"
    assert id(valid) in seen["state"]
    taken, _ = evaluator._take_val_prefetch(valid)
    assert next(taken) == 0

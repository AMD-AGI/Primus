###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""flux_prep_ahead: the next step's inputs are prepared only when the loop's next forward is that training step's --
never before an evaluation, a checkpoint (regular or non-persistent), a phase transition, an exit or the last step,
nor with the rerun state machine on."""

from types import SimpleNamespace

import pytest

from primus.backends.megatron.patches.flux_prep_ahead_patches import (
    _next_is_training_forward,
)


def _args(**kw):
    base = dict(train_iters=100, eval_interval=None, do_valid=False, save=None, save_interval=None,
                non_persistent_save_interval=None, phase_transition_iterations=None, exit_interval=None,
                rerun_mode="disabled")
    base.update(kw)
    return SimpleNamespace(**base)


def test_plain_step_prepares():
    assert _next_is_training_forward(_args(), 7)


@pytest.mark.parametrize(
    "kw,iteration",
    [
        (dict(), 100),  # last step
        (dict(eval_interval=10, do_valid=True), 20),
        (dict(save="/ckpt", save_interval=5), 15),
        (dict(non_persistent_save_interval=4), 8),
        (dict(phase_transition_iterations=[30, 60]), 60),
        (dict(exit_interval=9), 18),
        (dict(rerun_mode="validate_results"), 7),
    ],
    ids=["last", "eval", "save", "non_persistent_save", "phase_transition", "exit_interval", "rerun"],
)
def test_skips(kw, iteration):
    assert not _next_is_training_forward(_args(**kw), iteration)


def test_intervals_only_at_their_iterations():
    a = _args(eval_interval=10, do_valid=True, save="/ckpt", save_interval=5, non_persistent_save_interval=4,
              phase_transition_iterations=[30], exit_interval=9)
    assert _next_is_training_forward(a, 7)

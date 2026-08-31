###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Where one evaluation ends and the next begins.

Consumers that derive per-microbatch state -- the diffusion trainer's eval RNG
index is the only one today -- need a signal for the start of an evaluation.
The training step is not that signal: two evaluations can run at the same step,
the one inside the training loop and the one ``pretrain`` runs after the loop
exits, and keying off the step alone would let the second continue the first's
counter instead of reproducing it.

Deliberately free of imports. The producer is the generic evaluator and the
consumer is a diffusion trainer, so anywhere else this could live would either
drag the Megatron evaluation stack into a per-microbatch code path or point the
dependency from generic code at diffusion code.
"""

_eval_session = 0


def begin_eval_session() -> int:
    """Mark the start of one evaluation. Called on every rank.

    Only ``primus_evaluate`` calls this, and the patch that installs it also
    requires ``eval_interval > 0`` while ``do_valid`` does not. A configuration
    with ``eval_iters > 0`` and ``eval_interval == 0`` therefore still reaches
    pretrain's post-training evaluation through stock Megatron ``evaluate``,
    where no session opens and consumers fall back to a free-running counter.
    No shipped recipe is in that state.

    ``multiple_validation_sets`` would open one session per set rather than per
    evaluation, since ``evaluate_and_print_results`` calls ``evaluate`` once per
    iterator. Consumers keying on (step, session) would then hand the same index
    to the first microbatch of every set. Primus defaults the flag off in
    trainer_base.yaml and no recipe turns it on; enabling it means revisiting
    this.
    """
    global _eval_session
    _eval_session += 1
    return _eval_session


def current_eval_session() -> int:
    """Identify the evaluation in progress.

    The value is not reproducible across runs and carries no meaning beyond
    changing when an evaluation starts, which is all a consumer needs to know
    to reset per-evaluation state.
    """
    return _eval_session

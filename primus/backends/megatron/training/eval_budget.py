###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
One definition of how large an evaluation is, shared by the dataloader
provider and the evaluation loop.

Previously the two disagreed: the Energon provider sized the validation
LimitDataset with ``get_num_microbatches()`` while the evaluator recomputed
``global_batch_size // (micro_batch_size * data_parallel_size)`` for itself.
Those diverge under batch-size ramp-up, and the evaluator's version silently
floors to zero when the global batch is smaller than one microbatch per rank,
which turns evaluation into a no-op that still reports a loss.

This module also owns the validation worker count, because worker count is
what actually decides how many samples an evaluation reads. Energon splits
the *data* across ``dp_size * num_workers`` workers but splits the *batch
quota* across ``num_workers`` alone, and those two divisions only agree when
each worker's slice is a whole number of full batches. When they disagree,
short-quota workers silently leave the tail of their slice unread and the
evaluation reports the count it intended rather than the count it achieved.
"""

import json
from pathlib import Path
from typing import Optional

__all__ = [
    "DEFAULT_VAL_NUM_WORKERS",
    "EvalCoverageError",
    "assert_val_worker_divisibility",
    "get_eval_num_microbatches",
    "get_val_num_workers",
    "read_energon_split_sample_count",
    "resolve_eval_iters",
]

# Energon clamps the worker count to at least one when sharding samples
# (sharder.py, ``max(1, worker_config.num_workers)``) and short-circuits the
# quota split entirely when ``num_workers <= 1`` (limit_dataset.py). At 0 or 1
# each rank therefore holds a single contiguous slice and takes the whole
# quota from it, so coverage is exact whenever the per-rank slice divides into
# whole microbatches. That makes 0 the only default that cannot silently
# under-read, which matters more for a metric that gates convergence than the
# throughput of a loader that runs a few dozen times per job.
DEFAULT_VAL_NUM_WORKERS = 0


class EvalCoverageError(ValueError):
    """Raised when an evaluation would not read the samples it claims to."""


def get_eval_num_microbatches(args) -> int:
    """Microbatches per evaluation iteration.

    Uses the same global batch as training so that ``eval_iters`` counts in
    global batches, matching Megatron's convention.
    """
    dp_size = args.data_parallel_size
    micro_batch_size = args.micro_batch_size
    global_batch_size = args.global_batch_size

    samples_per_microbatch = micro_batch_size * dp_size
    if samples_per_microbatch <= 0:
        raise EvalCoverageError(
            f"micro_batch_size ({micro_batch_size}) * data_parallel_size ({dp_size}) " f"must be positive."
        )

    num_microbatches = global_batch_size // samples_per_microbatch
    if num_microbatches <= 0:
        raise EvalCoverageError(
            f"global_batch_size ({global_batch_size}) is smaller than one microbatch "
            f"across data parallelism (micro_batch_size {micro_batch_size} x "
            f"data_parallel_size {dp_size} = {samples_per_microbatch}). Evaluation "
            f"would run zero microbatches per iteration and report a loss computed "
            f"from no samples."
        )
    if num_microbatches * samples_per_microbatch != global_batch_size:
        raise EvalCoverageError(
            f"global_batch_size ({global_batch_size}) is not divisible by "
            f"micro_batch_size ({micro_batch_size}) x data_parallel_size ({dp_size}) "
            f"= {samples_per_microbatch}. Evaluation cannot cover a whole number of "
            f"global batches."
        )
    return num_microbatches


def get_val_num_workers(args) -> int:
    """Dataloader worker count to use for validation.

    Falls back to ``DEFAULT_VAL_NUM_WORKERS`` rather than to the training
    ``num_workers``: sharing the training value is exactly what produces the
    silent under-read this module exists to prevent.
    """
    val_num_workers = getattr(args, "val_num_workers", None)
    if val_num_workers is None:
        return DEFAULT_VAL_NUM_WORKERS
    if val_num_workers < 0:
        raise EvalCoverageError(f"val_num_workers must be >= 0, got {val_num_workers}.")
    return val_num_workers


def assert_val_worker_divisibility(args, eval_samples: int) -> None:
    """Fail loudly when the configured shape cannot read every eval sample.

    Coverage is exact when each global worker's slice is a whole number of
    full microbatches, i.e. when::

        eval_samples % (dp_size * max(1, val_num_workers) * micro_batch_size) == 0

    The ``max(1, ...)`` mirrors Energon's own clamping, and is also what keeps
    this from dividing by zero at the default worker count of 0.
    """
    dp_size = args.data_parallel_size
    micro_batch_size = args.micro_batch_size
    val_num_workers = get_val_num_workers(args)

    divisor = dp_size * max(1, val_num_workers) * micro_batch_size
    remainder = eval_samples % divisor
    if remainder == 0:
        return

    per_global_worker = eval_samples // (dp_size * max(1, val_num_workers))
    suggestions = [
        w
        for w in range(0, min(64, eval_samples) + 1)
        if eval_samples % (dp_size * max(1, w) * micro_batch_size) == 0
    ]
    raise EvalCoverageError(
        f"Validation would silently read fewer than {eval_samples} samples.\n"
        f"  eval_samples          = {eval_samples}\n"
        f"  data_parallel_size    = {dp_size}\n"
        f"  micro_batch_size      = {micro_batch_size}\n"
        f"  val_num_workers       = {val_num_workers}\n"
        f"Energon shards samples across dp_size x max(1, val_num_workers) = "
        f"{dp_size * max(1, val_num_workers)} global workers, giving "
        f"{per_global_worker} samples each, which is not a whole number of "
        f"{micro_batch_size}-sample batches ({remainder} sample(s) over). Workers "
        f"whose batch quota is short leave the tail of their slice unread, so the "
        f"reported sample count would exceed the count actually evaluated.\n"
        f"Valid val_num_workers for this shape: {suggestions}"
    )


def read_energon_split_sample_count(data_path, split: str = "val") -> Optional[int]:
    """Total samples in an Energon split, from the dataset's own index.

    Lets ``full_validation`` mean "all of it" without the count being written
    into a recipe by hand, and gives the evaluation loop a third, independent
    number to check the configured and observed counts against.

    Returns None when the path is not an Energon dataset (mock data, an
    unprepared directory), so callers can fall back rather than fail.
    """
    if not data_path:
        return None
    if isinstance(data_path, (list, tuple)):
        if not data_path:
            return None
        data_path = data_path[0]

    info = Path(str(data_path)) / ".nv-meta" / ".info.json"
    try:
        shard_counts = json.loads(info.read_text())["shard_counts"]
    except (OSError, ValueError, KeyError):
        return None

    prefix = f"{split}/"
    total = sum(count for shard, count in shard_counts.items() if shard.startswith(prefix))
    return total or None


def resolve_eval_iters(args) -> Optional[int]:
    """Derive ``eval_iters`` from ``eval_samples`` when the latter is set.

    ``eval_samples`` says what the evaluation is meant to measure -- coverage
    of a dataset -- whereas ``eval_iters`` only means that at one particular
    global batch size. Returns the derived value, or None when ``eval_samples``
    is unset and ``eval_iters`` should be left alone.
    """
    eval_samples = getattr(args, "eval_samples", None)
    if eval_samples is None:
        return None

    if eval_samples <= 0:
        raise EvalCoverageError(f"eval_samples must be positive, got {eval_samples}.")

    global_batch_size = args.global_batch_size
    if eval_samples % global_batch_size != 0:
        raise EvalCoverageError(
            f"eval_samples ({eval_samples}) is not divisible by global_batch_size "
            f"({global_batch_size}), so evaluation cannot cover it in whole global "
            f"batches. Either adjust global_batch_size or accept a different "
            f"eval_samples; {global_batch_size * (eval_samples // global_batch_size)} "
            f"and {global_batch_size * (eval_samples // global_batch_size + 1)} are "
            f"the nearest reachable counts."
        )

    derived = eval_samples // global_batch_size

    # eval_samples deliberately wins over eval_iters rather than conflicting
    # with it. There is no way to tell an eval_iters a recipe chose from one it
    # inherited: trainer_base.yaml always supplies a value, and Megatron's
    # parser defaults it besides, so treating a mismatch as an error would
    # reject every recipe that opts into eval_samples at all.
    assert_val_worker_divisibility(args, eval_samples)
    return derived

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
import os
from pathlib import Path
from typing import Optional

__all__ = [
    "DEFAULT_VAL_NUM_WORKERS",
    "EvalCoverageError",
    "assert_mlperf_timestep_source",
    "assert_val_worker_divisibility",
    "get_data_parallel_size",
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
    """Raised when an evaluation would not measure what it reports.

    Covers both reading fewer samples than the configuration claims and
    evaluating the samples it does read at timesteps the dataset did not pair
    them with.
    """


def get_data_parallel_size(args) -> int:
    """Data-parallel width, usable before Megatron has computed it.

    Megatron only sets ``data_parallel_size`` while initialising process
    groups, but the evaluation budget has to be resolved earlier than that --
    in the ``build_args`` phase, so the Energon provider and the evaluator
    both see the corrected ``eval_iters``. Reading the attribute directly
    there raises ``AttributeError``, which the patch runner logs and swallows,
    leaving ``eval_iters`` at 0; the job then runs to completion having
    evaluated nothing. So derive the value the same way Megatron does instead.
    """
    dp_size = getattr(args, "data_parallel_size", None)
    if dp_size:
        return dp_size

    world_size = getattr(args, "world_size", None) or int(os.environ.get("WORLD_SIZE", 0))
    if not world_size:
        raise EvalCoverageError(
            "Cannot size the evaluation: data_parallel_size is not set yet and "
            "neither args.world_size nor the WORLD_SIZE environment variable is "
            "available to derive it from."
        )

    divisor = 1
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
    ):
        divisor *= getattr(args, name, 1) or 1

    if world_size % divisor != 0:
        raise EvalCoverageError(
            f"world_size ({world_size}) is not divisible by "
            f"tensor x pipeline x context parallel size ({divisor}), so "
            f"data_parallel_size cannot be derived."
        )
    return world_size // divisor


def get_eval_num_microbatches(args) -> int:
    """Microbatches per evaluation iteration.

    Uses the same global batch as training so that ``eval_iters`` counts in
    global batches, matching Megatron's convention.
    """
    dp_size = get_data_parallel_size(args)
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
    dp_size = get_data_parallel_size(args)
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


def assert_mlperf_timestep_source(args) -> None:
    """Refuse injected validation timesteps in an MLPerf run.

    ``resolve_validation_timesteps`` takes the dataset's ``timestep`` column
    whenever the batch carries one, so a split ingested with that column is
    evaluated correctly under either source setting. One combination is left
    unsafe: shards ingested before the column was carried through, read under
    ``eval_timestep_source='equidistant'``. There the positional fallback
    injects ``t = index % 8``, which does not reproduce the pairing of image
    to timestep the published split defines, and the run reports a val_loss
    indistinguishable in the logs from a correct one.

    The MLPerf recipes set ``dataset`` and close that cell for themselves, but
    ``trainer_base.yaml`` defaults to ``equidistant`` for the diffusion
    recipes that have no annotated split, so the protection would otherwise
    rest on every future submission recipe remembering to override it. A
    submission has no legitimate use for injected timesteps, so make the
    combination unreachable rather than merely avoidable.

    Belongs at validation-dataloader-build time and not in the ``build_args``
    patch that sizes the eval budget: a failure raised there is logged and
    swallowed by the patch runner, which is how ``eval_iters`` silently stayed
    at 0. ``eval_timestep_source`` is also a Primus-only key, and those are
    merged onto ``args`` only after the ``build_args`` phase has run.
    """
    if not getattr(args, "mlperf_mode", False):
        return

    # Imported here rather than at module scope to keep this module free of
    # the diffusion forward step, which pulls in torch and the Flux model
    # utilities; the eval budget is also resolved from a build_args patch.
    from primus.backends.megatron.training.diffusion.forward_step import (
        DATASET_TIMESTEPS,
    )

    source = getattr(args, "eval_timestep_source", None)
    if source == DATASET_TIMESTEPS:
        return

    raise EvalCoverageError(
        f"mlperf_mode is set but eval_timestep_source is {source!r}, which lets "
        f"validation fall back to injecting t = index % 8 when the shards carry "
        f"no per-sample 'timestep' column. That reproduces neither the timesteps "
        f"nor the image-to-timestep pairing of the published val split, and it "
        f"fails silently: the loss it reports looks exactly like a correct one. "
        f"Set eval_timestep_source='{DATASET_TIMESTEPS}' in the recipe, and point "
        f"the run at a val split ingested with the timestep column "
        f"(primus/configs/data/megatron/diffusion/preprocessing/mlperf_flux1_val.yaml) "
        f"so the requirement is met rather than merely asserted."
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

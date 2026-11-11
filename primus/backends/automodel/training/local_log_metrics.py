###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Compute the diffusion train loop's logged step metrics locally, without a device sync or collectives.

WHY:
  The AutoModel diffusion loop's logging path costs a full device
  synchronization and three scalar collectives on every step it logs. No
  compute overlaps those collectives, because the point of the
  synchronization is that nothing is in flight, and the GPU idles in the
  same tail of the step while the CPU refills the queue.

  Three separate things are paid per step, all inside the `should_log` branch
  (and with `log_remote_every_steps: 1`, that is every step):

    _elapsed_seconds_since   torch.cuda.synchronize() + all_reduce(MAX, f64)
    _count_global_samples    all_reduce(SUM, i64)
    _get_memory_metrics      all_reduce(MAX, f64[4])

  The synchronize() is the expensive one. It does not merely cost its own wall
  time: it drains the queue, so the CPU cannot run ahead and every following
  kernel launch is exposed until the queue refills.

WHAT:
  Replaces the three methods on ``TrainDiffusionRecipe``, each with a cheaper
  equivalent:

    _elapsed_seconds_since   local wall clock, no sync and no reduction
    _count_global_samples    local_samples * dp_size, no reduction
    _get_memory_metrics      local allocator counters, no reduction

  Model-agnostic: ``TrainDiffusionRecipe`` is the shared diffusion recipe and
  none of this touches a model definition. No AutoModel/diffusers edit --
  methods are swapped on the class.

  Independent of ``primus_train.defer_host_sync``, which moves the loop's loss
  ``.item()`` and ``float(grad_norm)`` to logging steps; with both on, a logging
  step waits only for those values.

  Off by default; idempotent; a default run is unaffected.

WHAT IS GIVEN UP (read before enabling on a reporting run):
  * ``step_time`` becomes a CPU-side measurement. In steady state the CPU cannot
    outrun the GPU, so the *rate* over many steps stays correct, but an
    individual step's time is now "time to enqueue", not "time to execute", and
    per-step jitter goes up. Validate throughput against the log's wall-clock
    timestamps over many steps rather than trusting a single step.
  * ``samples_per_sec`` is derived from a count that assumes every rank
    contributed the same number of samples. True with ``drop_last: true`` and
    ``dynamic_batch_size: false``; the patch falls back to the original
    reduction when it cannot confirm that.
  * ``mem`` becomes this rank's peak rather than the max across ranks. These
    are within noise of each other on a uniformly sharded FSDP2 job.

Setting (``primus_train:`` section, off by default):
    local_log_metrics: false
"""
from __future__ import annotations

import logging
import time

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_INSTALLED_FLAG = "_primus_local_log_metrics_installed"
_SECTION = "primus_train"


def is_enabled() -> bool:
    """Whether the logged step metrics should be computed locally."""
    return options.flag(f"{_SECTION}.local_log_metrics")


def _uniform_batch_across_ranks(recipe) -> bool:
    """Whether every rank is guaranteed the same per-step sample count.

    Only then can the global count be computed instead of reduced. Unknown
    counts as False, so an exotic loader keeps the original reduction.
    """
    try:
        loader_cfg = recipe.cfg.get("data", {}).get("dataloader", {})
        get = loader_cfg.get if hasattr(loader_cfg, "get") else (lambda k, d=None: getattr(loader_cfg, k, d))
        return bool(get("drop_last", False)) and not bool(get("dynamic_batch_size", False))
    except Exception:
        return False


def install() -> bool:
    """Swap the three synchronizing methods on the diffusion recipe class.

    Returns True if the replacements are in place (including when they already were).
    """
    import torch
    import torch.distributed as dist
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

    if getattr(TrainDiffusionRecipe, _INSTALLED_FLAG, False):
        return True

    orig_count = TrainDiffusionRecipe._count_global_samples

    def elapsed_seconds_since(self, start_time: float):
        """Local elapsed wall time: no device sync, no MAX reduction.

        The original takes the max across ranks so the slowest rank defines the
        step. Ranks are already kept in lockstep by the gradient reduce-scatter,
        so the spread between ranks is small compared with what the sync costs.
        """
        end_time = time.perf_counter()
        return max(end_time - start_time, 1e-12), end_time

    def count_global_samples(self, local_samples: int) -> int:
        """dp_size * local count, when every rank is known to match."""
        if not getattr(self, "_primus_uniform_batch", False):
            return orig_count(self, local_samples)
        try:
            if dist.is_initialized():
                return int(local_samples) * dist.get_world_size(self._get_dp_group())
        except Exception:
            return orig_count(self, local_samples)
        return int(local_samples)

    def get_memory_metrics(self) -> dict:
        """This rank's allocator counters, with no MAX reduction."""
        if not torch.cuda.is_available():
            zero = dict.fromkeys(
                (
                    "mem",
                    "memory_allocated_gb",
                    "memory_reserved_gb",
                    "max_memory_allocated_gb",
                    "max_memory_reserved_gb",
                ),
                0.0,
            )
            return zero
        scale = 1024**3
        allocated = torch.cuda.memory_allocated(self.device) / scale
        reserved = torch.cuda.memory_reserved(self.device) / scale
        max_allocated = torch.cuda.max_memory_allocated(self.device) / scale
        max_reserved = torch.cuda.max_memory_reserved(self.device) / scale
        return {
            "mem": max_allocated,
            "memory_allocated_gb": allocated,
            "memory_reserved_gb": reserved,
            "max_memory_allocated_gb": max_allocated,
            "max_memory_reserved_gb": max_reserved,
        }

    # Decided once per process rather than per step: the loader config cannot
    # change mid-run, and this runs before the first logged step.
    orig_setup = TrainDiffusionRecipe.setup

    def setup(self, *args, **kwargs):
        ret = orig_setup(self, *args, **kwargs)
        self._primus_uniform_batch = _uniform_batch_across_ranks(self)
        if not self._primus_uniform_batch:
            logger.warning(
                "[PrimusLocalLogMetrics] dataloader is not known to give every rank the "
                "same sample count (needs drop_last and no dynamic_batch_size); keeping "
                "the sample-count all-reduce."
            )
        return ret

    TrainDiffusionRecipe._elapsed_seconds_since = elapsed_seconds_since
    TrainDiffusionRecipe._count_global_samples = count_global_samples
    TrainDiffusionRecipe._get_memory_metrics = get_memory_metrics
    TrainDiffusionRecipe.setup = setup
    setattr(TrainDiffusionRecipe, _INSTALLED_FLAG, True)
    logger.info(
        "[PrimusLocalLogMetrics] removed the per-step device sync and the 3 logging "
        "collectives. NOTE: step_time is now CPU-side; judge throughput over a "
        "window, not per step."
    )
    return True

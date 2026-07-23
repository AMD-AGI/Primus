###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Env-gated ``torch.profiler`` wrapper for the AutoModel diffusion train loop —
the no-fork vehicle for Phase-C+ Phase-2 (b): the real 8-GPU FSDP2 COMMS-share
profile (``09_PHASE_C_SWEEP_DESIGN.md`` §4b).

WHY:
  Profile (a) (single-GPU ``scripts/attn_profile.py``) isolates the intra-step
  COMPUTE op-share. To size the FSDP collective (all-gather + reduce-scatter)
  share — the top remaining lever now that throughput saturates at a low mbs knee
  (`08`/`09` §1) — we must profile the ACTUAL distributed step. This hook captures
  a per-rank ``torch.profiler`` Chrome trace of a few steady steps of the real
  ``TrainDiffusionRecipe`` loop, which TraceLens turns into a gpu-timeline
  comm/compute split + a per-collective (NcclAnalyser) report.

WHAT (NO Automodel/diffusers fork):
  ``install()`` class-patches ``TrainDiffusionRecipe.run_train_validation_loop`` so
  that, when ``PRIMUS_IDEOGRAM_PROFILE=1``, it runs the ORIGINAL loop inside a
  ``torch.profiler.profile`` context. The per-step boundary needed to drive
  ``prof.step()`` (for the wait/warmup/active schedule) is obtained by wrapping the
  recipe's ``self.optimizer.step`` (called exactly once per optimization step),
  with NO edit to the submodule loop body. Each rank exports its own trace so the
  multi-rank collective report can correlate collectives across ranks.

  Env-gated by ``PRIMUS_IDEOGRAM_PROFILE=1`` (default off). Installed via the
  trainer's optional-hooks list. Idempotent; a default run is unaffected.

Activation / knobs (env):
    PRIMUS_IDEOGRAM_PROFILE=1        enable the profiler wrapper
    PRIMUS_PROFILE_DIR=<dir>         per-rank trace output dir (default
                                     /mnt/m2m_nobackup/ideogram_profile/comms)
    PRIMUS_PROFILE_TAG=<tag>         subdir/prefix for this point (e.g. 1024_m32)
    PRIMUS_PROFILE_WAIT=3            steps skipped before profiling (skip warmup/
                                     epoch-boundary data tax)
    PRIMUS_PROFILE_WARMUP=1          profiler warmup steps (recorded then dropped)
    PRIMUS_PROFILE_ACTIVE=3          steps actually captured
    PRIMUS_PROFILE_WITH_STACK=0      include python call stacks (bigger trace;
                                     needed for nn.Module attribution / recompute)
    PRIMUS_PROFILE_RECORD_SHAPES=1   record op input shapes (GEMM M/N/K + coll size)
    PRIMUS_PROFILE_MEMORY=0          record allocator events
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}


def is_profile_enabled() -> bool:
    return os.getenv("PRIMUS_IDEOGRAM_PROFILE", "0") in _TRUTHY


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) in _TRUTHY


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _rank() -> int:
    for key in ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.getenv(key)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def install() -> bool:
    """Wrap the AutoModel diffusion train loop in ``torch.profiler`` (no-fork).

    No-op (returns False) unless ``PRIMUS_IDEOGRAM_PROFILE`` is set.
    """
    if not is_profile_enabled():
        return False

    import torch

    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

    if getattr(TrainDiffusionRecipe, "_primus_profile_installed", False):
        return True

    orig_loop = TrainDiffusionRecipe.run_train_validation_loop

    out_dir = os.getenv("PRIMUS_PROFILE_DIR", "/mnt/m2m_nobackup/ideogram_profile/comms")
    tag = os.getenv("PRIMUS_PROFILE_TAG", "run")
    wait = _int("PRIMUS_PROFILE_WAIT", 3)
    warmup = _int("PRIMUS_PROFILE_WARMUP", 1)
    active = _int("PRIMUS_PROFILE_ACTIVE", 3)
    with_stack = _flag("PRIMUS_PROFILE_WITH_STACK", "0")
    record_shapes = _flag("PRIMUS_PROFILE_RECORD_SHAPES", "1")
    profile_memory = _flag("PRIMUS_PROFILE_MEMORY", "0")

    def patched_loop(self):
        rank = _rank()
        point_dir = os.path.join(out_dir, tag)
        os.makedirs(point_dir, exist_ok=True)
        trace_path = os.path.join(point_dir, f"rank{rank}.pt.trace.json")

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        sched = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)

        def _on_ready(prof):
            try:
                prof.export_chrome_trace(trace_path)
                logger.info("[PrimusIdeogramProfile] rank%s wrote trace -> %s", rank, trace_path)
            except Exception as exc:  # pragma: no cover
                logger.error("[PrimusIdeogramProfile] rank%s trace export failed: %s", rank, exc)

        prof = torch.profiler.profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=_on_ready,
            record_shapes=record_shapes,
            with_stack=with_stack,
            profile_memory=profile_memory,
        )

        # Drive prof.step() from the recipe's per-step optimizer.step (called once
        # per optimization step) without editing the submodule loop body.
        orig_opt_step = self.optimizer.step

        def opt_step_and_prof(*args, **kwargs):
            ret = orig_opt_step(*args, **kwargs)
            prof.step()
            return ret

        logger.info(
            "[PrimusIdeogramProfile] rank%s profiling '%s' (wait=%d warmup=%d active=%d "
            "with_stack=%s record_shapes=%s) -> %s",
            rank, tag, wait, warmup, active, with_stack, record_shapes, point_dir,
        )
        self.optimizer.step = opt_step_and_prof
        prof.start()
        try:
            orig_loop(self)
        finally:
            try:
                prof.stop()
            except Exception:
                pass
            self.optimizer.step = orig_opt_step

    patched_loop._primus_profile_wrapped = True
    TrainDiffusionRecipe.run_train_validation_loop = patched_loop
    TrainDiffusionRecipe._primus_profile_installed = True
    logger.info("[PrimusIdeogramProfile] Installed torch.profiler train-loop wrapper (no-fork).")
    return True

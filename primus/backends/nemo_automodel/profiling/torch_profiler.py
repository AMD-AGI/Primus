###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Env-gated ``torch.profiler`` wrapper for the AutoModel diffusion train loop.

WHY:
  A single-GPU profile shows the intra-step compute op mix, but says nothing about
  the FSDP2 collectives (all-gather + reduce-scatter), which is usually what one
  wants to size once compute is understood. Those only appear in a real
  distributed step. This captures a per-rank Chrome trace of a few steady-state
  steps of the actual ``TrainDiffusionRecipe`` loop, so collectives can be
  correlated across ranks by any standard trace viewer.

WHAT (no AutoModel/diffusers fork):
  Class-patches ``TrainDiffusionRecipe.run_train_validation_loop`` so that the
  original loop runs inside a ``torch.profiler.profile`` context. The per-step
  boundary needed to drive ``prof.step()`` (for the wait/warmup/active schedule)
  comes from wrapping the recipe's ``optimizer.step``, which is called exactly
  once per optimization step -- so no edit to the loop body is required. Each rank
  exports its own trace.

  Model-agnostic: ``TrainDiffusionRecipe`` is the shared diffusion recipe and the
  step boundary is found through the recipe's optimizer, so nothing here is
  specific to any one model.

  Env-gated and off by default; idempotent; a default run is unaffected.

Knobs (env):
    PRIMUS_DIFFUSION_PROFILE=1       enable the profiler wrapper
    PRIMUS_PROFILE_DIR=<dir>         per-rank trace output dir
                                     (default ./output/diffusion_profile)
    PRIMUS_PROFILE_TAG=<tag>         subdir/prefix for this run
    PRIMUS_PROFILE_WAIT=3            steps skipped before profiling, to get past
                                     warmup and the epoch-boundary data cost
    PRIMUS_PROFILE_WARMUP=1          profiler warmup steps (recorded then dropped)
    PRIMUS_PROFILE_ACTIVE=3          steps actually captured
    PRIMUS_PROFILE_WITH_STACK=0      include python call stacks
    PRIMUS_PROFILE_WITH_MODULES=0    include the nn.Module hierarchy per op
    PRIMUS_PROFILE_RECORD_SHAPES=1   record op input shapes
    PRIMUS_PROFILE_MEMORY=0          record allocator events

Note on ``WITH_STACK``/``WITH_MODULES``: these are what let a reader say *which part
of the model* a kernel belongs to. Without them a trace still answers "what ran, on
what shapes" but not "who called it", which is usually the question being asked of a
trace. They are off by default because both inflate trace size and CPU-side step
time; turn them on deliberately when the trace is for someone to read rather than to
time.
"""
from __future__ import annotations

import logging
import os

from primus.backends.nemo_automodel._env import current_rank, env_flag, env_int, env_str

logger = logging.getLogger(__name__)

_INSTALLED_FLAG = "_primus_profile_installed"


def is_enabled() -> bool:
    """Whether the profiler wrapper should be applied."""
    return env_flag("PRIMUS_DIFFUSION_PROFILE")


def install() -> bool:
    """Wrap the AutoModel diffusion train loop in ``torch.profiler``.

    Returns True if the wrapper is in place (including when it already was).
    """
    import torch
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

    if getattr(TrainDiffusionRecipe, _INSTALLED_FLAG, False):
        return True

    orig_loop = TrainDiffusionRecipe.run_train_validation_loop

    out_dir = env_str("PRIMUS_PROFILE_DIR", "./output/diffusion_profile")
    tag = env_str("PRIMUS_PROFILE_TAG", "run")
    wait = env_int("PRIMUS_PROFILE_WAIT", 3)
    warmup = env_int("PRIMUS_PROFILE_WARMUP", 1)
    active = env_int("PRIMUS_PROFILE_ACTIVE", 3)
    with_stack = env_flag("PRIMUS_PROFILE_WITH_STACK", False)
    with_modules = env_flag("PRIMUS_PROFILE_WITH_MODULES", False)
    record_shapes = env_flag("PRIMUS_PROFILE_RECORD_SHAPES", True)
    profile_memory = env_flag("PRIMUS_PROFILE_MEMORY", False)

    def patched_loop(self):
        rank = current_rank()
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
                logger.info("[PrimusDiffusionProfile] rank%s wrote trace -> %s", rank, trace_path)
            except Exception as exc:  # pragma: no cover
                logger.error("[PrimusDiffusionProfile] rank%s trace export failed: %s", rank, exc)

        prof = torch.profiler.profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=_on_ready,
            record_shapes=record_shapes,
            with_stack=with_stack,
            with_modules=with_modules,
            profile_memory=profile_memory,
        )

        # Drive prof.step() from the recipe's per-step optimizer.step (called once
        # per optimization step) without editing the loop body.
        #
        # ``self.optimizer`` is a LIST: OptimizerConfig.build returns one optimizer per
        # model part, and the recipe steps each in turn. Hook only the LAST one, so the
        # profiler schedule advances exactly once per optimization step -- hooking all of
        # them would advance it once per model part and capture the wrong steps.
        optimizers = self.optimizer if isinstance(self.optimizer, (list, tuple)) else [self.optimizer]
        if not optimizers:
            logger.error("[PrimusDiffusionProfile] recipe has no optimizer to hook; not profiling.")
            return orig_loop(self)
        target_optimizer = optimizers[-1]
        orig_opt_step = target_optimizer.step

        def opt_step_and_prof(*args, **kwargs):
            ret = orig_opt_step(*args, **kwargs)
            prof.step()
            return ret

        logger.info(
            "[PrimusDiffusionProfile] rank%s profiling '%s' (wait=%d warmup=%d active=%d "
            "with_stack=%s with_modules=%s record_shapes=%s) -> %s",
            rank,
            tag,
            wait,
            warmup,
            active,
            with_stack,
            with_modules,
            record_shapes,
            point_dir,
        )
        target_optimizer.step = opt_step_and_prof
        prof.start()
        try:
            orig_loop(self)
        finally:
            try:
                prof.stop()
            except Exception:
                pass
            target_optimizer.step = orig_opt_step

    patched_loop._primus_profile_wrapped = True
    TrainDiffusionRecipe.run_train_validation_loop = patched_loop
    setattr(TrainDiffusionRecipe, _INSTALLED_FLAG, True)
    logger.info("[PrimusDiffusionProfile] installed torch.profiler train-loop wrapper.")
    return True

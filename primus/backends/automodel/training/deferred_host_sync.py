###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Defer the diffusion train loop's per-step host syncs to logging steps.

WHY:
  ``TrainDiffusionRecipe.run_train_validation_loop`` blocks the CPU on the GPU
  twice every step: ``float(grad_norm)`` right after clipping, before
  ``optimizer.step()``, and ``.item()`` on the micro-batch loss mean after it.
  The CPU therefore cannot queue the optimizer and the next forward ahead of the
  GPU, and each step starts with a GPU bubble while it catches up: 45-60 ms on
  FLUX.1-schnell FP8, of which deferring recovered 15 ms per step.

WHAT:
  The loss mean stays a GPU tensor and is accumulated as one, in float64 like the
  Python float it replaces. ``group_loss_mean``, ``epoch_loss`` and ``grad_norm``
  become Python floats only on logging steps
  (``step_scheduler.log_remote_every_steps``), where the timing code synchronizes
  anyway, and where a checkpoint or the epoch summary needs the loss. Logged
  values are unchanged to the bit; only the point at which the CPU waits for them
  moves.

  The edits are source rewrites of the upstream method (``_source_patch``): they
  sit in the middle of a long loop that no wrapper can reach. An anchor that no
  longer matches raises, naming it.

Setting (``primus_train:`` section, off by default):
  defer_host_sync: false   also builds the FLUX latent ids on device (``models/flux/latent_ids``)
"""
from __future__ import annotations

import logging

from primus.backends.automodel import options
from primus.backends.automodel._source_patch import patch_method_source

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_primus_deferred_host_sync"

# At the absolute indentation of nemo_automodel/recipes/diffusion/train.py.
REPLACEMENTS = [
    (
        "                grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm\n",
        "",
    ),
    (
        "                group_loss_mean = float(torch.stack(micro_losses).mean().item())\n"
        "                epoch_loss += group_loss_mean\n",
        "                group_loss_mean = torch.stack(micro_losses).mean().detach()\n"
        "                epoch_loss = epoch_loss + group_loss_mean.double()\n",
    ),
    (
        "                if should_log:\n" "                    elapsed_seconds, perf_window_end_time = ",
        "                if should_log:\n"
        "                    group_loss_mean = float(group_loss_mean)\n"
        "                    epoch_loss = float(epoch_loss)\n"
        "                    grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else grad_norm\n"
        "                    elapsed_seconds, perf_window_end_time = ",
    ),
    (
        "self.save_checkpoint(epoch, global_step, epoch_loss / num_steps)",
        "self.save_checkpoint(epoch, global_step, float(epoch_loss) / num_steps)",
    ),
    (
        "            avg_loss = epoch_loss / num_steps\n            logging.info(",
        "            avg_loss = float(epoch_loss) / num_steps\n            logging.info(",
    ),
]


def is_enabled() -> bool:
    return options.flag("primus_train.defer_host_sync")


def install() -> bool:
    """Rewrite ``TrainDiffusionRecipe.run_train_validation_loop``."""
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

    return patch_recipe(TrainDiffusionRecipe)


def patch_recipe(recipe_cls) -> bool:
    """Apply the rewrite to ``recipe_cls``; True once it is in place."""
    if getattr(recipe_cls, _PATCH_FLAG, False):
        return True

    loop = recipe_cls.__dict__.get("run_train_validation_loop")
    if getattr(loop, "_primus_profile_wrapped", False):
        raise RuntimeError(
            "[PrimusDeferredHostSync] run_train_validation_loop is already wrapped by the profiler; "
            "this rewrite must be installed first (lower patch priority)."
        )

    patch_method_source(recipe_cls, "run_train_validation_loop", REPLACEMENTS)
    setattr(recipe_cls, _PATCH_FLAG, True)
    logger.info(
        "[PrimusDeferredHostSync] train-loop host syncs (loss .item(), float(grad_norm)) deferred to "
        "logging steps"
    )
    return True

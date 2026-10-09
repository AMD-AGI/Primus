###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Prepare the next Flux training step's inputs at the end of the current one (``flux_prep_ahead``).

A Flux forward step starts with a stretch of eager host work before the first transformer block can be launched: the
batch fetch, the VAE resample, the noise / timestep / CFG-dropout draws, latent packing and the position IDs. At the
step boundary that work sits between the optimizer and the next forward, and the GPU waits for it.

With this gate, after a training step's ``train_step`` returns (backward and optimizer step launched), the trainer's
``prepare_next_training_step`` runs the next forward step's input preparation right away; the forward step then uses
the prepared inputs. Preparation draws from the same data iterator and the same CUDA generator in the same order --
nothing else draws from either between the two points -- so the run is bitwise unchanged.

It is done only when the next forward is known to be the next training step's:

* inside Megatron's ``train()`` loop only, so never around the MLPerf warmup (which runs before ``train()``, or
  inside the first ``train_step`` call below this wrapper) and its restore;
* not when an evaluation, a checkpoint save or an exit-interval stop follows the step, nor after the last step
  (evaluation draws from the same generator; a checkpoint records the RNG and data position).

An exit by signal or wall-clock duration after a prepared step leaves one batch and one step's draws consumed but
unused, which only matters if the RNG state is checkpointed at that exit.

Off by default. Enable with ``flux_prep_ahead: true``.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _next_is_training_forward(args, iteration) -> bool:
    """After train_step(iteration=i) the loop's iteration becomes i + 1; is its next forward a training step's?"""
    nxt = iteration + 1
    train_iters = getattr(args, "train_iters", None)
    if train_iters is None or nxt >= train_iters:
        return False
    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval and nxt % eval_interval == 0 and getattr(args, "do_valid", False):
        return False
    save_interval = getattr(args, "save_interval", None)
    if getattr(args, "save", None) and save_interval and nxt % save_interval == 0:
        return False
    exit_interval = getattr(args, "exit_interval", None)
    if exit_interval and nxt % exit_interval == 0:
        return False
    return True


@register_patch(
    "megatron.training.flux_prep_ahead",
    backend="megatron",
    phase="before_train",
    description="Prepare the next Flux training step's inputs at the end of the current train_step.",
    # Outermost train_step wrapper (after the warmup hook, the data prefetch (42) and the wall-clock timer (90)).
    priority=97,
    condition=lambda ctx: bool(getattr(get_args(ctx), "flux_prep_ahead", False)),
)
def patch_flux_prep_ahead(ctx: PatchContext) -> None:
    import megatron.training.training as MT

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    key = "megatron.training.flux_prep_ahead"
    if is_patched(MT, key):
        return
    args = get_args(ctx)
    in_train = [False]
    warned = [False]

    inner_train_step = MT.train_step

    def train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        result = inner_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
            iteration=iteration,
        )
        if not in_train[0] or iteration is None:
            return result
        should_exit = isinstance(result, tuple) and len(result) > 3 and bool(result[3])
        if should_exit or not _next_is_training_forward(args, iteration):
            return result
        prepare = getattr(forward_step_func, "prepare_next_training_step", None)
        if prepare is None:
            if not warned[0]:
                warning_rank_0("[Patch:flux_prep_ahead] the forward step has no prepare_next_training_step; inactive")
                warned[0] = True
            return result
        prepare()
        return result

    MT.train_step = train_step

    inner_train = MT.train

    def train(*a, **k):
        in_train[0] = True
        try:
            return inner_train(*a, **k)
        finally:
            in_train[0] = False

    MT.train = train
    mark_patched(MT, key)
    log_rank_0("[Patch:megatron.training.flux_prep_ahead] next step's inputs prepared at the end of train_step")

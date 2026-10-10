###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Prepare the next Flux training step's inputs at the end of the current one (``flux_prep_ahead``).

A Flux forward step starts with a stretch of eager host work before the first transformer block can be launched: the
batch fetch, the VAE resample, the noise / timestep / CFG-dropout draws, latent packing and the position IDs. At the
step boundary that work sits between the optimizer and the next forward, and the GPU waits for it.

With this gate the trainer's ``prepare_next_training_step`` runs the next forward step's input preparation as soon as
a training step has been launched (backward and optimizer step), and the forward step then uses the prepared inputs.
Preparation draws from the same data iterator and the same CUDA generator in the same order -- nothing else draws from
either between the two points -- so the run is bitwise unchanged.

The trigger is Megatron's ``training_log`` call, which ``train()`` makes right after every ``train_step`` (before
logging, evaluation or checkpointing). Hooking it rather than ``train_step`` keeps it off the warmup steps: both MLPerf
warmup paths replace ``training_log`` with a no-op while they run (and the in-step one rewrites ``train_step`` when it
removes itself). Preparation is skipped when the next forward is not the next training step's:

* an evaluation, a checkpoint save (regular or non-persistent), a phase transition or an exit-interval stop follows
  the step, or it was the last step (evaluation draws from the same generator; a checkpoint records the RNG and data
  position);
* the rerun state machine is enabled (it may replay a step with its own data and draws).

An exit by signal or wall-clock duration after a prepared step leaves one batch and one step's draws consumed but
unused, which only matters if the RNG state is checkpointed at that exit.

Off by default. Enable with ``flux_prep_ahead: true``.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _next_is_training_forward(args, iteration) -> bool:
    """training_log(iteration=i) follows step i; is the loop's next forward step i + 1's training forward?"""
    train_iters = getattr(args, "train_iters", None)
    if train_iters is None or iteration >= train_iters:
        return False
    eval_interval = getattr(args, "eval_interval", None)
    if eval_interval and iteration % eval_interval == 0 and getattr(args, "do_valid", False):
        return False
    save_interval = getattr(args, "save_interval", None)
    if getattr(args, "save", None) and save_interval and iteration % save_interval == 0:
        return False
    non_persistent = getattr(args, "non_persistent_save_interval", None)
    if non_persistent and iteration % non_persistent == 0:
        return False
    if iteration in (getattr(args, "phase_transition_iterations", None) or ()):
        return False  # Megatron saves and exits at a phase transition
    exit_interval = getattr(args, "exit_interval", None)
    if exit_interval and iteration % exit_interval == 0:
        return False
    if str(getattr(args, "rerun_mode", "disabled")) not in ("disabled", "RerunMode.DISABLED"):
        return False  # the rerun state machine may replay a step with its own data and draws
    return True


@register_patch(
    "megatron.training.flux_prep_ahead",
    backend="megatron",
    phase="before_train",
    description="Prepare the next Flux training step's inputs right after the current step is launched.",
    # Outermost training_log wrapper, so preparation is launched before any logging work.
    priority=97,
    condition=lambda ctx: bool(getattr(get_args(ctx), "flux_prep_ahead", False)),
)
def patch_flux_prep_ahead(ctx: PatchContext) -> None:
    import megatron.training.training as MT

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
    from primus.backends.megatron.patches.training_log.training_log_args import (
        bind_training_log_args,
    )

    key = "megatron.training.flux_prep_ahead"
    if is_patched(MT, key):
        return
    args = get_args(ctx)
    state = {"prepare": None}
    warned = [False]

    inner_training_log = MT.training_log

    def training_log(*a, **k):
        prepare = state["prepare"]
        if prepare is not None:
            bound = bind_training_log_args(a, k)
            if bound is not None and _next_is_training_forward(args, bound.arguments["iteration"]):
                prepare()
        return inner_training_log(*a, **k)

    MT.training_log = training_log

    inner_train = MT.train

    def train(*a, **k):
        forward_step_func = a[0] if a else k.get("forward_step_func")
        prepare = getattr(forward_step_func, "prepare_next_training_step", None)
        if prepare is None and not warned[0]:
            warning_rank_0("[Patch:flux_prep_ahead] the forward step has no prepare_next_training_step; inactive")
            warned[0] = True
        # Read through the function at call time, so a later wrapper of the attribute is honoured.
        state["prepare"] = (lambda: forward_step_func.prepare_next_training_step()) if prepare is not None else None
        try:
            return inner_train(*a, **k)
        finally:
            state["prepare"] = None

    MT.train = train
    mark_patched(MT, key)
    log_rank_0("[Patch:megatron.training.flux_prep_ahead] next step's inputs prepared right after each training step")

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Wall-clock timer patch for Megatron train_step.

Wraps ``megatron.training.training.train_step`` with ``time.perf_counter()``
to measure the actual forward + backward + optimizer wall-clock duration,
independent of Megatron's ``interval-time`` timer which includes collective
barriers, CUDA synchronizes, and logging overhead.

The measured duration is stored on ``runtime_state.last_metrics`` so that
downstream log extensions (e.g. ``DiffusionMetricsExtension``) can surface
it alongside existing throughput metrics.
"""

import time

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _wall_clock_timer_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return args is not None and getattr(args, "wall_clock_step_timer", False)


@register_patch(
    "megatron.training.wall_clock_step_timer",
    backend="megatron",
    phase="before_train",
    description="Wrap train_step with wall-clock timer for NeMo-comparable throughput measurement",
    condition=_wall_clock_timer_enabled,
    priority=90,
)
def patch_wall_clock_timer(ctx: PatchContext):
    import megatron.training.training as megatron_training

    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    _PATCH_KEY = "megatron.training.train_step_wall_clock_timer"
    if is_patched(megatron_training, _PATCH_KEY):
        log_rank_0("[Patch:wall_clock_step_timer] Already applied; skipping re-wrap.")
        return

    runtime_state = ctx.extra.get("runtime_state")

    _original_train_step = megatron_training.train_step

    def _patched_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        t0 = time.perf_counter()
        result = _original_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
            iteration=iteration,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if runtime_state is not None:
            runtime_state.update_metrics({"wall_clock_step_ms": dt_ms})
        return result

    megatron_training.train_step = _patched_train_step
    mark_patched(megatron_training, _PATCH_KEY)
    log_rank_0(
        "[Patch:wall_clock_step_timer] "
        "Patched train_step with wall-clock timer "
        f"(runtime_state={'available' if runtime_state else 'unavailable'})"
    )

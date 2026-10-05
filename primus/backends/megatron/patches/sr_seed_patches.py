###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Per-step stochastic-rounding seed for the Primus-Turbo SR quantizers.

Primus-Turbo's stochastic-rounding quantizers (the ``quantize_mx*`` packers of the MXFP6 / MXFP4 recipes and the MXFP4
quantizer) seed each launch from a base seed and a launch counter. Before every ``train_step`` this sets the base to a
hash of the run seed, the global rank and the iteration (``sr_step_seed``), which resets the counters. So the SR bits
of a step are:

* different per data-parallel rank -- the ranks round different gradients at the same positions, and shared bits
  would correlate their rounding errors instead of letting them average out in the gradient all-reduce;
* different per run seed;
* the same again when a run resumes at that iteration, and independent of anything quantized before the step.

Applied only to MXFP-quantized jobs (``fp6`` or ``fp4`` set) on a Primus-Turbo with ``set_sr_seed``.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _sr_seed_ops():
    try:
        from primus_turbo.pytorch.ops.quantization import set_sr_seed, sr_step_seed
    except ImportError:
        return None
    import torch

    if not hasattr(getattr(torch.ops, "primus_turbo_cpp_extension", None), "set_sr_seed"):
        return None
    return set_sr_seed, sr_step_seed


def _is_sr_seed_patch_needed(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    quantized = bool(getattr(args, "fp6", None) or getattr(args, "fp4", None))
    return quantized and _sr_seed_ops() is not None


@register_patch(
    "megatron.training.sr_seed.wrap_train_step",
    backend="megatron",
    phase="before_train",
    description="Seed Primus-Turbo stochastic rounding per step from (run seed, global rank, iteration).",
    condition=_is_sr_seed_patch_needed,
)
def patch_train_step_with_sr_seed(ctx: PatchContext) -> None:
    import megatron.training.training as training  # type: ignore
    import torch
    from megatron.training.global_vars import get_args as get_megatron_args

    original_train_step = training.train_step
    if getattr(original_train_step, "_primus_sr_seed_wrapped", False):
        return
    set_sr_seed, sr_step_seed = _sr_seed_ops()
    state = {"logged": False}

    def _train_step_with_sr_seed(*args, **kwargs):
        mg_args = get_megatron_args()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # curr_iteration is the step about to run (set by Megatron's train loop before train_step); the base is set
        # here, outside any compiled region, before the step's first SR pack.
        iteration = int(getattr(mg_args, "curr_iteration", 0))
        set_sr_seed(sr_step_seed(int(mg_args.seed), rank, iteration))
        if not state["logged"]:
            log_rank_0(
                f"[Patch:megatron.sr_seed] SR base seed per step = hash(seed={mg_args.seed}, rank, iteration)"
            )
            state["logged"] = True
        return original_train_step(*args, **kwargs)

    setattr(_train_step_with_sr_seed, "_primus_sr_seed_wrapped", True)
    training.train_step = _train_step_with_sr_seed

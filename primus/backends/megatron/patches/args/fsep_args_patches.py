###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
FSEP (Fully Sharded Expert Parallel) argument patches.

Injects FSEP-specific arguments that are not part of Megatron's argparse
into the args namespace, and validates their constraints at startup.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.args.fsep_defaults",
    backend="megatron",
    phase="build_args",
    description="Inject FSEP argument defaults that are not in Megatron's argparse",
)
def patch_fsep_args_defaults(ctx: PatchContext):
    """
    Inject default values for FSEP-related args that Megatron's argparse
    does not define. This prevents AttributeError when get_args() is called.
    """
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    if not hasattr(args, "moe_fsep_sharding_degree"):
        args.moe_fsep_sharding_degree = 0

    if not hasattr(args, "moe_log_expert_load"):
        args.moe_log_expert_load = False


@register_patch(
    "megatron.args.fsep_validation",
    backend="megatron",
    phase="before_train",
    description="Validate FSEP configuration constraints before training starts",
    condition=lambda ctx: getattr(get_args(ctx), "moe_fsep_sharding_degree", 0) > 1,
)
def patch_fsep_args_validation(ctx: PatchContext):
    """
    Validate FSEP argument constraints.

    Requirements:
      - moe_fsep_sharding_degree must be power of 2
      - moe_fsep_sharding_degree must divide expert_model_parallel_size
      - moe_fsep_sharding_degree must equal expert_tensor_parallel_size
        (initial constraint to reuse Expert TP process group and sharded_state_dict)
      - moe_pad_expert_input_to_capacity must be False
      - enable_primus_turbo and use_turbo_deepep must be True
    """
    args = get_args(ctx)
    S = args.moe_fsep_sharding_degree
    ep = getattr(args, "expert_model_parallel_size", 1)
    etp = getattr(args, "expert_tensor_parallel_size", None) or 1

    assert S in [2, 4, 8, 16], (
        f"moe_fsep_sharding_degree must be one of [2, 4, 8, 16], got {S}"
    )
    assert ep % S == 0, (
        f"expert_model_parallel_size ({ep}) must be divisible by "
        f"moe_fsep_sharding_degree ({S})"
    )
    assert S <= ep, (
        f"moe_fsep_sharding_degree ({S}) cannot exceed "
        f"expert_model_parallel_size ({ep})"
    )
    assert etp == S, (
        f"expert_tensor_parallel_size ({etp}) must equal "
        f"moe_fsep_sharding_degree ({S}) for checkpoint compatibility. "
        f"Set expert_tensor_parallel_size: {S} in your config."
    )
    assert not getattr(args, "moe_pad_expert_input_to_capacity", False), (
        "FSEP does not support moe_pad_expert_input_to_capacity=True"
    )
    assert getattr(args, "enable_primus_turbo", False), (
        "FSEP requires enable_primus_turbo: true"
    )
    assert getattr(args, "use_turbo_deepep", False), (
        "FSEP requires use_turbo_deepep: true"
    )

    log_rank_0(
        f"[FSEP] Validation passed: "
        f"sharding_degree={S}, EP={ep}, ETP={etp}"
    )

###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_kv_rank_0, log_rank_0


@register_patch(
    "megatron.args.hsdp",
    backend="megatron",
    phase="build_args",
    description=(
        "Map data_parallel_replicate_degree to Megatron's "
        "num_distributed_optimizer_instances so that initialize_model_parallel "
        "creates the HSDP process groups (intra-partial shard + inter-instance replicate)."
    ),
)
def patch_hsdp_args(ctx: PatchContext):
    """
    Propagate data_parallel_replicate_degree through to Megatron.

    data_parallel_replicate_degree is a Primus-only config key that is not
    recognized by MegatronArgBuilder and would be silently dropped.  This
    patch reads it from module_config.params, injects it into backend_args,
    and -- when > 1 -- maps it to num_distributed_optimizer_instances so that
    Megatron's parallel_state.initialize_model_parallel creates the required
    intra-partial (shard) and inter-instance (replicate) process groups.
    """
    args = ctx.extra.get("backend_args")
    module_config = ctx.extra.get("module_config")

    if not args or not module_config:
        return

    replicate_degree = getattr(module_config.params, "data_parallel_replicate_degree", 1)
    args.data_parallel_replicate_degree = replicate_degree

    if replicate_degree > 1:
        if not getattr(args, "use_torch_fsdp2", False):
            raise ValueError("data_parallel_replicate_degree > 1 requires use_torch_fsdp2=True")

        ckpt_fmt = getattr(args, "ckpt_format", "torch_dist")
        if ckpt_fmt != "torch_dcp":
            raise ValueError(
                f"HSDP (data_parallel_replicate_degree={replicate_degree}) requires "
                f"ckpt_format='torch_dcp', got '{ckpt_fmt}'. The torch_dist format's "
                f"checkpoint offset logic assumes flat DP and is incompatible with a 2D mesh."
            )

        args.num_distributed_optimizer_instances = replicate_degree
        log_rank_0(
            f"[Patch:megatron.args.hsdp] HSDP enabled: "
            f"data_parallel_replicate_degree={replicate_degree} "
            f"-> num_distributed_optimizer_instances={replicate_degree}"
        )

    log_kv_rank_0("[Patch:megatron.args.hsdp] data_parallel_replicate_degree", str(replicate_degree))

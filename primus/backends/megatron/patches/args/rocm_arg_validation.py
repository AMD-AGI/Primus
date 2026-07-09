###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ROCm-specific Megatron argument validation.

``validate_args_on_rocm`` is invoked by the ``megatron.validate_args`` patch
after Megatron's own ``validate_args`` runs. It enforces ROCm/Primus-Turbo
constraints (deterministic-mode env vars, Turbo FP8/FP4 recipes, sync-free MoE
auto-config, DeepEP restrictions, ...).
"""

import inspect
import os

import torch

from primus.core.utils import logger


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(msg):
    """If distributed is initialized, print only on last rank."""
    log_func = logger.info_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if torch.distributed.is_initialized():
        if is_last_rank():
            log_func(msg, module_name, function_name, line)
    else:
        log_func(msg, module_name, function_name, line)


def _get_sync_free_moe_options(args) -> dict:
    stage = args.turbo_sync_free_moe_stage

    if stage > 3 or stage < 0:
        raise ValueError("turbo_sync_free_moe_stage only support [0-3]")

    sync_free_moe = {
        1: {
            "moe_use_fused_router_with_aux_score": True,
            "moe_permute_fusion": True,
            "moe_router_padding_for_quantization": True if args.fp8 or args.fp4 else False,
        },
        2: {
            "moe_use_fused_router_with_aux_score": True,
            "use_turbo_deepep": True,
            "moe_permute_fusion": True,
            "use_turbo_grouped_gemm": True,
            "moe_router_padding_for_quantization": True if args.fp8 or args.fp4 else False,
        },
        3: {
            "moe_use_fused_router_with_aux_score": True,
            "use_turbo_deepep": True,
            "moe_permute_fusion": True,
            "use_turbo_grouped_gemm": True,
            "moe_router_padding_for_quantization": True if args.fp8 or args.fp4 else False,
            "use_turbo_fused_act_with_probs": True,
        },
    }

    return sync_free_moe[stage]


# FSDP2 custom optimizer selection flags. Each one monkeypatches
# get_megatron_optimizer at the same priority (50), so at most one may be set.
_FSDP2_OPTIMIZER_FLAGS = (
    "use_fsdp2_fp32_param_optimizer",
    "use_fsdp2_bf16_master_weight_optimizer",
)


def validate_fsdp2_optimizer_exclusivity(args) -> None:
    """Ensure at most one FSDP2 custom optimizer flag is enabled.

    Enabling more than one would silently let whichever optimizer patch applies
    last win the monkeypatch, so raise ValueError to fail loudly at
    arg-validation time (before training starts).
    """
    enabled = [flag for flag in _FSDP2_OPTIMIZER_FLAGS if getattr(args, flag, False)]
    if len(enabled) > 1:
        raise ValueError(
            "Conflicting FSDP2 optimizer selection: at most one of "
            f"{list(_FSDP2_OPTIMIZER_FLAGS)} may be enabled, but got {enabled}. "
            "Enable exactly one."
        )


def validate_args_on_rocm(args):
    # Deterministic mode
    if args.deterministic_mode:
        # NOTE: Some environment variables affect deterministic mode on ROCm. Need to do extra check.
        NON_DETERMINISTIC_ENVS = {
            "TORCH_COMPILE_DISABLE": "1",
            "ROCBLAS_DEFAULT_ATOMICS_MODE": "0",
            "PRIMUS_TURBO_AUTO_TUNE": "0",
            "PRIMUS_DETERMINISTIC": "1",
        }
        # NOTE: Some version triton compile exist potential racing condition issue.
        for env, value in NON_DETERMINISTIC_ENVS.items():
            assert (
                os.environ.get(env, None) == value
            ), f"{env} must be set to {value} in deterministic mode but got {os.environ.get(env, None)} instead."

        # Set fill_uninitialized_memory to False to avoid calling extra fill kernel in deterministic mode.
        torch.utils.deterministic.fill_uninitialized_memory = False

    assert not getattr(
        args, "use_turbo_parallel_linear", False
    ), "use_turbo_parallel_linear has been removed; please use use_turbo_gemm instead."

    validate_fsdp2_optimizer_exclusivity(args)

    use_turbo_gemm = getattr(args, "use_turbo_gemm", False)
    # Turbo FP8 linear check
    if args.fp8 and use_turbo_gemm:
        support_fp8_recipe = ["tensorwise", "blockwise", "mxfp8"]
        assert (
            args.fp8_recipe in support_fp8_recipe
        ), f"{args.fp8_recipe} recipe is not support when enable `use_turbo_gemm`."

    # Turbo FP4 linear check
    if args.fp4 and use_turbo_gemm:
        support_fp4_recipe = ["mxfp4"]
        assert (
            args.fp4_recipe in support_fp4_recipe
        ), f"{args.fp4_recipe} recipe is not support when enable `use_turbo_gemm`."

    # NOTE: mxfp8 environment variable must be set to 1 to enable mxfp8 recipe on ROCm.
    if args.fp8_recipe == "mxfp8":
        assert (
            os.getenv("NVTE_ROCM_ENABLE_MXFP8", "0") == "1"
        ), "Please set `NVTE_ROCM_ENABLE_MXFP8=1` to enable `mxfp8` recipe."

    # dump pp data
    if args.dump_pp_data and args.pipeline_model_parallel_size == 1:
        args.dump_pp_data = False
        print_rank_last(f"Disable args.dump_pp_data since args.pipeline_model_parallel_size=1")

    # PrimusTurboGroupedMLP no longer depends on legacy GroupedMLP; the two
    # flags are mutually exclusive when turbo is enabled.
    assert not getattr(
        args, "use_turbo_grouped_mlp", False
    ), "use_turbo_grouped_mlp has been removed; please use use_turbo_grouped_gemm instead."
    use_turbo_grouped_gemm = getattr(args, "use_turbo_grouped_gemm", False)
    if use_turbo_grouped_gemm:
        if getattr(args, "moe_use_legacy_grouped_gemm", False):
            raise ValueError(
                "use_turbo_grouped_gemm=True is incompatible with moe_use_legacy_grouped_gemm=True. "
                "please set moe_use_legacy_grouped_gemm=False."
            )

    # sync-free MoE
    if args.turbo_sync_free_moe_stage > 0:
        assert args.enable_primus_turbo, "Please set `enable_primus_turbo=True` to enable sync-free MoE."

        if args.turbo_sync_free_moe_stage > 1 and not use_turbo_grouped_gemm:
            raise ValueError(
                "Sync-Free MoE stage 2 or 3 require PrimusTurboGroupedLinear, please set `use_turbo_grouped_gemm=True`"
            )
        options = _get_sync_free_moe_options(args)
        print_rank_last(
            f"========== Enable Sync-Free MoE Stage {args.turbo_sync_free_moe_stage} (Auto-Enabled Options) =========="
        )
        for flag, value in options.items():
            dots = "." * (73 - len(flag) - len(str(value)))
            print_rank_last(f"{flag}{dots}{value}")
            setattr(args, flag, value)
        print_rank_last(
            f"========== Enable Sync-Free MoE Stage {args.turbo_sync_free_moe_stage} (Auto-Enabled Options) =========="
        )

    # turbo deepep
    if args.use_turbo_deepep:
        assert (
            not args.moe_shared_expert_overlap
        ), "DeepEP not support moe_shared_expert_overlap, please set `moe_shared_expert_overlap=False`."
        assert (
            args.moe_router_dtype == "fp32"
        ), "DeepEP only supports float32 probs, please set `moe_router_dtype=fp32`"
        if (
            args.expert_model_parallel_size >= 16
            and os.getenv("PRIMUS_TURBO_MOE_DISPATCH_COMBINE_BACKEND", "TURBO") == "TURBO"
        ):
            # Turbo DeepEP is not supported for CUs > 32 when using internode dispatch/combine.
            assert args.turbo_deepep_num_cu <= 32, "Set `turbo_deepep_num_cu<=32` when using ep_size >= 16."

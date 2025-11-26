###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
ROCm Memory Monitoring Patches

Patches Megatron's training_log function to add ROCm-specific memory monitoring
and logging for AMD GPUs.
"""

import torch

from primus.core.patches import PatchContext, register_patch


@register_patch(
    "megatron.rocm.memory_monitoring",
    backend="megatron",
    phase="before_train",
    description="Add ROCm memory monitoring to Megatron training_log",
)
def patch_training_log_for_rocm_memory(ctx: PatchContext):
    """
    Patch Megatron's training_log function to add ROCm memory monitoring.

    This patch wraps the original training_log function and enhances the
    log string output with AMD GPU memory statistics.

    Strategy:
        1. Check if ROCm monitoring is enabled at patch time (not runtime)
        2. If disabled, skip patching entirely (zero overhead)
        3. If enabled, wrap training_log to inject memory stats

    Memory stats provided:
        - HIP memory (torch.cuda.mem_get_info): Fast, always available
        - ROCm memory (rocm-smi): More accurate, optional (controlled by args)

    The patch only applies when:
        - args.log_throughput is True
        - args.use_rocm_mem_info is True OR args.use_rocm_mem_info_iters is set
    """
    try:
        import megatron.training.training as megatron_training  # type: ignore
        from megatron.training.global_vars import get_args  # type: ignore
        from megatron.training.utils import (
            print_rank_0 as original_print_rank_0,  # type: ignore
        )

        from primus.core.utils.distributed_logging import log_rank_0
        from primus.core.utils.rocm_mem_info import get_rocm_smi_mem_info

        # Check if ROCm monitoring is enabled at patch time
        args = get_args()
        should_enable_patch = (
            hasattr(args, "log_throughput")
            and args.log_throughput
            and (
                getattr(args, "use_rocm_mem_info", False)
                or (
                    hasattr(args, "use_rocm_mem_info_iters")
                    and len(getattr(args, "use_rocm_mem_info_iters", [])) > 0
                )
            )
        )

        if not should_enable_patch:
            # ROCm monitoring disabled, skip patching entirely
            log_rank_0(
                "[Patch:megatron.rocm.memory_monitoring][SKIP] ROCm memory monitoring disabled "
                "(log_throughput=False or use_rocm_mem_info not configured)"
            )
            return

        # Store the original training_log function
        original_training_log = megatron_training.training_log

        def patched_training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
        ):
            """
            Patched training_log with ROCm memory monitoring.

            This wraps the original Megatron training_log and injects ROCm
            memory statistics into the throughput logging section.
            """
            # Capture the log string from original function
            captured_lines = []

            def capturing_print_rank_0(message):
                """Temporary replacement for print_rank_0 that captures output."""
                captured_lines.append(message)

            # Replace print_rank_0 temporarily
            import megatron.training.utils as megatron_utils

            original_utils_print = megatron_utils.print_rank_0
            megatron_utils.print_rank_0 = capturing_print_rank_0
            megatron_training.print_rank_0 = capturing_print_rank_0

            try:
                # Call original function (it will call our capturing print)
                result = original_training_log(
                    loss_dict,
                    total_loss_dict,
                    learning_rate,
                    decoupled_learning_rate,
                    iteration,
                    loss_scale,
                    report_memory_flag,
                    skipped_iter,
                    grad_norm,
                    params_norm,
                    num_zeros_in_grad,
                )
            finally:
                # Always restore original print_rank_0
                megatron_utils.print_rank_0 = original_utils_print
                megatron_training.print_rank_0 = original_utils_print

            # Process captured lines and inject ROCm memory stats
            for line in captured_lines:
                # Check if this is the throughput line (contains "throughput per GPU")
                if "throughput per GPU" in line:
                    try:
                        # Get memory stats
                        hip_mem_str = ""
                        rocm_mem_str = ""

                        # Get HIP memory info (unless ROCm SMI is primary)
                        if not getattr(args, "use_rocm_mem_info", False):
                            hip_free_mem, hip_total_mem = torch.cuda.mem_get_info()
                            hip_used_mem = hip_total_mem - hip_free_mem
                            hip_mem_usage = hip_used_mem / hip_total_mem
                            hip_mem_str = (
                                f" hip mem usage/free/total/usage_ratio: {hip_used_mem/1024/1024/1024:.2f}GB/"
                                f"{hip_free_mem/1024/1024/1024:.2f}GB/"
                                f"{hip_total_mem/1024/1024/1024:.2f}GB/{hip_mem_usage*100:.2f}% |"
                            )

                        # Get ROCm memory info if requested
                        if getattr(args, "use_rocm_mem_info", False) or iteration in getattr(
                            args, "use_rocm_mem_info_iters", []
                        ):
                            local_rank = torch.cuda.current_device()
                            rocm_total_mem, rocm_used_mem, rocm_free_mem = get_rocm_smi_mem_info(local_rank)
                            rocm_mem_usage = rocm_used_mem / rocm_total_mem
                            rocm_mem_str = (
                                f" rocm mem usage/free/total/usage_ratio: {rocm_used_mem/1024/1024/1024:.2f}GB/"
                                f"{rocm_free_mem/1024/1024/1024:.2f}GB/"
                                f"{rocm_total_mem/1024/1024/1024:.2f}GB/{rocm_mem_usage*100:.2f}% |"
                            )

                        # Inject memory stats before "throughput per GPU"
                        line = line.replace(
                            " throughput per GPU", f"{hip_mem_str}{rocm_mem_str} throughput per GPU"
                        )

                    except Exception:
                        # Silently fail memory monitoring to avoid breaking training
                        pass

                # Print the (possibly modified) line
                original_print_rank_0(line)

            return result

        # Replace Megatron's training_log with our patched version
        megatron_training.training_log = patched_training_log

        log_rank_0("[Patch:megatron.rocm.memory_monitoring] Added ROCm memory monitoring to training_log")

    except ImportError as e:
        from primus.core.utils.distributed_logging import log_rank_0

        log_rank_0(f"[Patch:megatron.rocm.memory_monitoring][SKIP] Failed to import modules: {e}")
    except AttributeError as e:
        from primus.core.utils.distributed_logging import log_rank_0

        log_rank_0(
            f"[Patch:megatron.rocm.memory_monitoring][WARN] "
            f"Megatron version may not have training_log: {e}"
        )

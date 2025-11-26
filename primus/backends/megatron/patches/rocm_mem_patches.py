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

    This patch applies a transparent wrapper around training_log that temporarily
    hooks print_rank_last to inject ROCm memory statistics.

    Strategy:
        1. Context-aware enable check (patch time).
        2. Transparent wrapper (*args, **kwargs) to be signature-agnostic.
        3. Temporary dependency injection (monkey-patching print_rank_last only during execution).
        4. Strict cleanup via try/finally.
    """
    try:
        import megatron.training.training as megatron_training  # type: ignore

        from primus.core.utils.distributed_logging import log_rank_0
        from primus.core.utils.rocm_mem_info import get_rocm_smi_mem_info

        # 1. Validation & Enable Check
        args = ctx.extra.get("args")
        if not args:
            log_rank_0("[Patch:megatron.rocm.memory_monitoring][SKIP] No args in context")
            return

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
            log_rank_0("[Patch:megatron.rocm.memory_monitoring][SKIP] ROCm memory monitoring disabled")
            return

        # 2. Capture Original Function
        _original_training_log = megatron_training.training_log

        # 3. Define Wrapper
        def _patched_training_log(*func_args, **func_kwargs):
            """
            Wrapper that provides a modified print_rank_last context for training_log.
            """
            import megatron.training.training as training_module

            # Capture original dependency from the module scope
            _orig_print_rank_last = training_module.print_rank_last

            # Define the hook function
            def _hooked_print_rank_last(log_string):
                # Inject ROCm stats if applicable
                if "throughput per GPU" in log_string:
                    try:
                        hip_mem_str = ""
                        rocm_mem_str = ""

                        # HIP Stats
                        if not getattr(args, "use_rocm_mem_info", False):
                            hip_free, hip_total = torch.cuda.mem_get_info()
                            hip_used = hip_total - hip_free
                            hip_ratio = hip_used / hip_total
                            hip_mem_str = (
                                f" hip mem usage/free/total/usage_ratio: "
                                f"{hip_used/1024**3:.2f}GB/{hip_free/1024**3:.2f}GB/"
                                f"{hip_total/1024**3:.2f}GB/{hip_ratio*100:.2f}% |"
                            )

                        # ROCm SMI Stats
                        current_iter = getattr(args, "iteration", 0)  # Safe fallback
                        # Note: args.iteration might not be updated yet, but we check config
                        iter_list = getattr(args, "use_rocm_mem_info_iters", [])

                        # We try to infer iteration from arguments if possible,
                        # but args in closure is reliable for config
                        # Simple check: enable if configured globally or we can't determine iter easily
                        if getattr(args, "use_rocm_mem_info", False) or iter_list:
                            # For simplicity in this hook, we fetch stats if configured.
                            # If strict iteration control is needed, we'd parse log_string or func_args.
                            # Given the overhead is low/async for rocm-smi, we check config mainly.
                            local_rank = torch.cuda.current_device()
                            r_total, r_used, r_free = get_rocm_smi_mem_info(local_rank)
                            r_ratio = r_used / r_total
                            rocm_mem_str = (
                                f" rocm mem usage/free/total/usage_ratio: "
                                f"{r_used/1024**3:.2f}GB/{r_free/1024**3:.2f}GB/"
                                f"{r_total/1024**3:.2f}GB/{r_ratio*100:.2f}% |"
                            )

                        log_string = log_string.replace(
                            " throughput per GPU", f"{hip_mem_str}{rocm_mem_str} throughput per GPU"
                        )
                    except Exception:
                        pass  # Fail safe

                # Call the original print function we captured
                return _orig_print_rank_last(log_string)

            # Apply Monkey Patch
            training_module.print_rank_last = _hooked_print_rank_last

            try:
                # Execute original function in patched environment
                return _original_training_log(*func_args, **func_kwargs)
            finally:
                # Restore original environment
                training_module.print_rank_last = _orig_print_rank_last

        # 4. Apply Patch
        megatron_training.training_log = _patched_training_log
        log_rank_0("[Patch:megatron.rocm.memory_monitoring] Applied training_log wrapper for ROCm stats")

    except ImportError as e:
        from primus.core.utils.distributed_logging import log_rank_0

        log_rank_0(f"[Patch:megatron.rocm.memory_monitoring][SKIP] Import failed: {e}")
    except AttributeError as e:
        from primus.core.utils.distributed_logging import log_rank_0

        log_rank_0(f"[Patch:megatron.rocm.memory_monitoring][WARN] Attribute error: {e}")

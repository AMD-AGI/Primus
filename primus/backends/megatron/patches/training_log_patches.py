###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Training Log Patches

This module provides an extensible patching mechanism for Megatron's training_log function.
It allows multiple extensions (ROCm monitoring, MLflow, etc.) to hook into the
training_log execution flow without coupling logic.

Architecture:
    - Extensions are implemented as Context Managers.
    - A unified wrapper uses contextlib.ExitStack to manage multiple extensions dynamically.
"""

import contextlib
from typing import Any, Dict, List, Protocol

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.distributed_logging import log_rank_0
from primus.core.utils.rocm_mem_info import get_rocm_smi_mem_info

# =============================================================================
# Extension Protocol
# =============================================================================


class TrainingLogExtension(Protocol):
    """Protocol for training_log extensions."""

    def update_context(self, func_args: tuple, func_kwargs: dict) -> None:
        """Optional: Update extension state based on training_log arguments."""
        ...

    def __enter__(self):
        """Prepare environment (e.g., swap functions, log metrics)."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore environment."""
        ...


# =============================================================================
# ROCm Memory Monitoring Extension
# =============================================================================


class RocmMonitorExtension:
    """
    Extension to inject ROCm memory statistics into training logs.

    It works by temporarily replacing 'megatron.training.training.print_rank_last'
    during the execution of training_log.
    """

    def __init__(self, args: Any, config: Dict[str, Any]):
        self.args = args
        self.config = config
        self.original_print = None
        self._megatron_training_module = None

    def __enter__(self):
        """Swap print_rank_last with our enhanced version."""
        try:
            import megatron.training.training as megatron_training

            self._megatron_training_module = megatron_training
            self.original_print = megatron_training.print_rank_last

            # Inject the hooked function
            megatron_training.print_rank_last = self._hooked_print_rank_last
        except (ImportError, AttributeError):
            pass  # Fail safe if module structure changes

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original print_rank_last."""
        if self._megatron_training_module and self.original_print:
            self._megatron_training_module.print_rank_last = self.original_print

    def _hooked_print_rank_last(self, log_string: str):
        """
        Intercepts the log string and injects memory stats if applicable.
        """
        # Only inject stats if this looks like a training iteration log (has throughput)
        if "throughput per GPU" in log_string:
            try:
                mem_stats = self._get_memory_stats()
                if mem_stats:
                    # Append memory stats to the log string
                    log_string += mem_stats
            except Exception:
                pass  # Do not crash training for logging issues

        # Delegate to the original print function
        if self.original_print:
            self.original_print(log_string)

    def _get_memory_stats(self) -> str:
        """Collect HIP and ROCm-SMI memory stats."""
        hip_mem_str = ""
        rocm_mem_str = ""

        # 1. HIP Stats (Always available on ROCm)
        # We assume that if this extension is active, we want to see memory stats.
        # HIP stats are cheap and always available via PyTorch.
        hip_free, hip_total = torch.cuda.mem_get_info()
        hip_used = hip_total - hip_free
        hip_ratio = hip_used / hip_total
        hip_mem_str = (
            f" hip mem usage/free/total/usage_ratio: "
            f"{hip_used / 1024 ** 3:.2f}GB/{hip_free / 1024 ** 3:.2f}GB/"
            f"{hip_total / 1024 ** 3:.2f}GB/{hip_ratio * 100:.2f}% |"
        )

        # 2. ROCm SMI Stats (If configured)
        # Check config from self.config (Primus specific parameters)
        use_rocm_mem = self.config.get("use_rocm_mem_info", False)
        rocm_iters = self.config.get("use_rocm_mem_info_iters", [])

        if use_rocm_mem or rocm_iters:
            # Note: We use config to decide enablement. For strict iteration control,
            # we would need to parse iteration from log_string or args, but
            # config-based check is sufficient and low-overhead here.
            local_rank = torch.cuda.current_device()
            r_total, r_used, r_free = get_rocm_smi_mem_info(local_rank)
            r_ratio = r_used / r_total
            rocm_mem_str = (
                f" rocm mem usage/free/total/usage_ratio: "
                f"{r_used / 1024 ** 3:.2f}GB/{r_free / 1024 ** 3:.2f}GB/"
                f"{r_total / 1024 ** 3:.2f}GB/{r_ratio * 100:.2f}% |"
            )

        return f"{hip_mem_str}{rocm_mem_str}"


# =============================================================================
# Main Patch Logic
# =============================================================================


@register_patch(
    "megatron.training_log.unified_patch",
    backend="megatron",
    phase="before_train",
    description="Extensible wrapper for Megatron training_log (ROCm stats, MLflow, etc.)",
)
def patch_training_log_unified(ctx: PatchContext):
    """
    Applies an extensible wrapper to Megatron's training_log.

    This implementation uses a Chain of Responsibility pattern via Context Managers.
    Each feature (ROCm stats, MLflow) is an independent 'Extension' that manages
    its own setup and teardown around the original function call.
    """
    try:
        import megatron.training.training as megatron_training  # type: ignore

        # 1. Get Configuration
        args = ctx.extra.get("args")
        config = ctx.extra.get("config", {})

        if not args:
            log_rank_0("[Patch:megatron.training_log][SKIP] No args in context")
            return

        # 2. Register Extensions
        extensions: List[Any] = []

        # -> Check ROCm Monitoring
        # Primus specific parameters are in config, not args
        use_rocm_mem = config.get("use_rocm_mem_info", False)
        rocm_iters = config.get("use_rocm_mem_info_iters", [])

        enable_rocm_stats = (
            hasattr(args, "log_throughput")
            and args.log_throughput
            and (use_rocm_mem or (rocm_iters and len(rocm_iters) > 0))
        )

        if enable_rocm_stats:
            extensions.append(RocmMonitorExtension(args, config))

        # -> Check MLflow (Placeholder for future implementation)
        # if enable_mlflow:
        #     extensions.append(MlflowExtension(ctx, args))

        if not extensions:
            log_rank_0("[Patch:megatron.training_log][SKIP] No active extensions configured")
            return

        # 3. Apply Wrapper
        _original_training_log = megatron_training.training_log

        def _patched_training_log(*func_args, **func_kwargs):
            # Use ExitStack to manage multiple extensions dynamically
            with contextlib.ExitStack() as stack:
                for ext in extensions:
                    # Allow extension to read current function args if needed
                    if hasattr(ext, "update_context"):
                        ext.update_context(func_args, func_kwargs)
                    # Enter the extension's context
                    stack.enter_context(ext)

                # Execute original function within the combined context
                return _original_training_log(*func_args, **func_kwargs)

        megatron_training.training_log = _patched_training_log
        log_rank_0(f"[Patch:megatron.training_log] Applied wrapper with {len(extensions)} extensions")

    except ImportError as e:
        log_rank_0(f"[Patch:megatron.training_log][SKIP] Import failed: {e}")
    except AttributeError as e:
        log_rank_0(f"[Patch:megatron.training_log][WARN] Attribute error: {e}")

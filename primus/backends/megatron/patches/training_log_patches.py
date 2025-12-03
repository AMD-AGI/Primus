###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Training Log Patches (with stacking)

This module provides an extensible AND stackable patching mechanism for
Megatron's training_log function.

Features
--------
- Extensions are implemented as Context Managers (e.g., ROCm monitoring, MLflow).
- A unified wrapper uses contextlib.ExitStack to manage multiple extensions.
- Multiple patches on training_log will be *stacked* instead of overwriting
  each other. Each new patch can append its own extensions to the wrapper.

Design
------
We attach metadata to the patched function:

    training_log._primus_training_log_wrapper = True
    training_log._primus_original_training_log = <original_fn>
    training_log._primus_extensions = [ext1, ext2, ...]

When a new patch runs:
    - If training_log is already wrapped, we re-use the original_fn and
      existing extensions, then append new ones.
    - If not, we wrap the current training_log and register our extensions.

This avoids the "last patch wins" problem and allows multiple modules to
contribute logging features safely.
"""

import contextlib
from typing import Any, Dict, List, Protocol

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.rocm_mem_info import get_rocm_smi_mem_info
from primus.modules.module_utils import log_rank_0, log_rank_all, warning_rank_0

# =============================================================================
# Extension Protocol
# =============================================================================


class TrainingLogExtension(Protocol):
    """Protocol for training_log extensions."""

    def update_context(self, func_args: tuple, func_kwargs: dict) -> None:
        """Optional: Update extension state based on training_log arguments."""

    def __enter__(self):
        """Prepare environment (e.g., swap functions, log metrics)."""

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore environment."""


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
        self.call_count = 0
        # Cache Primus-specific ROCm config to avoid repeated dict lookups
        self.use_rocm_mem: bool = bool(config.get("use_rocm_mem_info", False))
        self.rocm_iters = config.get("use_rocm_mem_info_iters", [])
        # Cache last successful ROCm SMI stats string so we can reuse it on
        # iterations where we intentionally skip expensive SMI queries.
        self._last_rocm_mem_str: str = ""

    def update_context(self, func_args: tuple, func_kwargs: dict) -> None:
        """
        Update the context before entering.
        We maintain a simple internal counter for logging steps to avoid
        fragile dependency on 'training_log' argument positions.
        """
        self.call_count += 1

    def __enter__(self):
        """Swap print_rank_last with our enhanced version."""
        try:
            import megatron.training.training as megatron_training

            self._megatron_training_module = megatron_training
            self.original_print = megatron_training.print_rank_last

            # Inject the hooked function
            megatron_training.print_rank_last = self._hooked_print_rank_last
        except (ImportError, AttributeError):
            # Fail safe if module structure changes or in CPU-only UTs.
            self._megatron_training_module = None
            self.original_print = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original print_rank_last."""
        if self._megatron_training_module and self.original_print:
            self._megatron_training_module.print_rank_last = self.original_print
        # Do not suppress exceptions from training_log.
        return False

    def _hooked_print_rank_last(self, log_string: str):
        """
        Intercepts the log string and injects memory stats if applicable.
        """
        try:
            mem_stats = self._get_memory_stats()
            if mem_stats:
                # Append memory stats to the log string
                log_string = f"{log_string} {mem_stats}"
        except Exception as e:
            # Logging must never break training; emit a warning and continue.
            warning_rank_0(f"[Patch:megatron.training_log] Failed to append memory stats: {e}")

        log_rank_all(f"{log_string}")

    def _get_memory_stats(self) -> str:
        """Collect HIP and ROCm-SMI memory stats."""
        hip_mem_str = ""
        rocm_mem_str = ""

        # 1. HIP Stats (Always available on ROCm)
        # We assume that if this extension is active, we want to see memory stats.
        # HIP stats are cheap and always available via PyTorch.
        try:
            hip_free, hip_total = torch.cuda.mem_get_info()
            hip_used = hip_total - hip_free
            hip_ratio = hip_used / hip_total
            hip_mem_str = (
                f" hip mem usage/free/total/usage_ratio: "
                f"{hip_used / 1024 ** 3:.2f}GB/"
                f"{hip_free / 1024 ** 3:.2f}GB/"
                f"{hip_total / 1024 ** 3:.2f}GB/"
                f"{hip_ratio * 100:.2f}%"
            )
        except Exception:
            # CUDA/ROCm may not be initialized (e.g., CPU-only UT).
            hip_mem_str = ""

        # 2. ROCm SMI Stats (Only if configured and iteration matches)
        # Only call expensive SMI if globally enabled OR current iteration is in list.
        # If we decide not to collect on this iteration but have a previously
        # collected value, reuse the last known ROCm SMI stats to keep the log
        # informative without incurring per-step overhead.
        should_collect_smi = self.use_rocm_mem or (self.call_count in self.rocm_iters)

        if should_collect_smi:
            try:
                local_rank = torch.cuda.current_device()
                r_total, r_used, r_free = get_rocm_smi_mem_info(local_rank)
                r_ratio = r_used / r_total
                rocm_mem_str = (
                    f" rocm mem usage/free/total/usage_ratio: "
                    f"{r_used / 1024 ** 3:.2f}GB/"
                    f"{r_free / 1024 ** 3:.2f}GB/"
                    f"{r_total / 1024 ** 3:.2f}GB/"
                    f"{r_ratio * 100:.2f}%"
                )
                # Cache for reuse on non-sampled iterations
                self._last_rocm_mem_str = rocm_mem_str
            except Exception:
                # If SMI fails, fall back to last known value (if any)
                rocm_mem_str = self._last_rocm_mem_str
        else:
            # Not a sampling iteration; reuse last successful SMI stats if available.
            rocm_mem_str = self._last_rocm_mem_str

        combined = " ".join(s for s in [hip_mem_str, rocm_mem_str] if s)
        return combined.strip()


# =============================================================================
# Helper: Training Log Wrapper with Stacking
# =============================================================================


def _wrap_training_log_with_extensions(
    original_fn,
    existing_extensions: List[Any],
    new_extensions: List[Any],
):
    """
    Create a new training_log wrapper that stacks existing + new extensions.

    This function:
        - Combines extension lists.
        - Defines a new wrapper that uses ExitStack to enter all extensions.
        - Attaches Primus metadata to enable future stacking.
    """
    all_extensions = list(existing_extensions) + list(new_extensions)

    def _patched_training_log(*func_args, **func_kwargs):
        # Use ExitStack to manage multiple extensions dynamically
        with contextlib.ExitStack() as stack:
            for ext in all_extensions:
                # Allow extension to read current function args if needed
                if hasattr(ext, "update_context"):
                    try:
                        ext.update_context(func_args, func_kwargs)
                    except Exception:
                        # Logging must not break training
                        pass
                # Enter the extension's context
                stack.enter_context(ext)
            # Execute original function within the combined context
            return original_fn(*func_args, **func_kwargs)

    # Attach metadata for stacking
    setattr(_patched_training_log, "_primus_training_log_wrapper", True)
    setattr(_patched_training_log, "_primus_original_training_log", original_fn)
    setattr(_patched_training_log, "_primus_extensions", all_extensions)

    return _patched_training_log


def _get_training_log_wrapper_state(training_log_fn):
    """
    Inspect current training_log function and return:

        is_wrapped: bool
        original_fn: callable
        existing_extensions: List[Any]

    If not wrapped by Primus, original_fn is training_log_fn itself
    and existing_extensions is an empty list.
    """
    is_wrapped = getattr(training_log_fn, "_primus_training_log_wrapper", False)
    if is_wrapped:
        original_fn = getattr(training_log_fn, "_primus_original_training_log", training_log_fn)
        existing_extensions = getattr(training_log_fn, "_primus_extensions", [])
        return True, original_fn, list(existing_extensions)
    else:
        return False, training_log_fn, []


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

    Additionally, this patch is *stackable*:
        - If training_log is already patched by Primus, we keep its original_fn
          and existing extensions, and simply append our new extensions.
        - If not, we wrap the existing training_log for the first time.
    """
    try:
        import megatron.training.training as megatron_training  # type: ignore

        # 1. Get Configuration
        args = ctx.extra.get("args")

        # Try to get config dict from 'config' key (Adapter style)
        config: Dict[str, Any] = ctx.extra.get("config", {}) or {}

        # If not found, try to get from 'module_config' object (BaseTrainer style)
        if not config and "module_config" in ctx.extra:
            module_config = ctx.extra["module_config"]
            if hasattr(module_config, "params"):
                config = module_config.params  # type: ignore[assignment]

        if not args:
            log_rank_0("[Patch:megatron.training_log][SKIP] No args in context")
            return

        # 2. Decide which extensions to enable for this patch
        new_extensions: List[Any] = []

        # -> ROCm Memory Monitoring
        use_rocm_mem = config.get("use_rocm_mem_info", False)
        rocm_iters = config.get("use_rocm_mem_info_iters", [])

        enable_rocm_stats = (
            hasattr(args, "log_throughput")
            and bool(getattr(args, "log_throughput"))
            and (use_rocm_mem or (rocm_iters and len(rocm_iters) > 0))
        )

        if enable_rocm_stats:
            new_extensions.append(RocmMonitorExtension(args, config))

        # -> MLflow / other extensions can be added here in the future:
        # if enable_mlflow:
        #     new_extensions.append(MlflowExtension(ctx, args))

        # 3. Inspect current training_log to see if it's already wrapped by Primus
        is_wrapped, original_fn, existing_extensions = _get_training_log_wrapper_state(
            megatron_training.training_log
        )

        if not new_extensions:
            if is_wrapped:
                # We don't remove existing extensions; just log and keep them.
                log_rank_0(
                    "[Patch:megatron.training_log][INFO] "
                    "No new extensions; keeping existing wrapper "
                    f"({len(existing_extensions)} existing extensions)"
                )
            else:
                log_rank_0(
                    "[Patch:megatron.training_log][SKIP] " "No active extensions configured for this patch"
                )
            return

        # 4. Create a new wrapper that stacks old + new extensions
        patched_fn = _wrap_training_log_with_extensions(
            original_fn=original_fn,
            existing_extensions=existing_extensions,
            new_extensions=new_extensions,
        )

        # 5. Install the new wrapper
        megatron_training.training_log = patched_fn

        log_rank_0(
            "[Patch:megatron.training_log] Applied wrapper: "
            f"{len(existing_extensions)} existing extensions + "
            f"{len(new_extensions)} new = "
            f"{len(existing_extensions) + len(new_extensions)} total"
        )

        # NOTE:
        # If Primus's PatchContext adds support for register_cleanup in the future,
        # we could register a cleanup handler here to restore
        # training_log = original_fn.
        # For now, we keep the wrapper active for the lifetime of the process.

    except ImportError as e:
        log_rank_0(f"[Patch:megatron.training_log][SKIP] Import failed: {e}")
    except AttributeError as e:
        log_rank_0(f"[Patch:megatron.training_log][WARN] Attribute error: {e}")
    except Exception as e:
        # Catch-all to make sure patch does not crash training.
        log_rank_0(f"[Patch:megatron.training_log][ERROR] Unexpected error: {e}")

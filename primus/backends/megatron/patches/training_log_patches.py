###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Training Log Patches

This module consolidates all patches related to Megatron's training_log function.
It currently includes:
1. MLflow logging integration
2. ROCm memory monitoring integration

Consolidating these patches is necessary because they both wrap the same
training_log function. Applying them separately would cause conflicts
(one wrapper overwriting the other).
"""

import torch

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.distributed_logging import log_rank_0
from primus.core.utils.rocm_mem_info import get_rocm_smi_mem_info


@register_patch(
    "megatron.training_log.unified_patch",
    backend="megatron",
    phase="before_train",
    description="Add MLflow logging and ROCm memory monitoring to Megatron training_log",
)
def patch_training_log_unified(ctx: PatchContext):
    """
    Unified patch for Megatron's training_log function.

    This patch applies a single transparent wrapper around training_log that:
    1. Extracts metrics for MLflow logging (if enabled).
    2. Temporarily hooks print_rank_last to inject ROCm memory statistics (if enabled).

    Strategy:
        - Single wrapper to avoid conflicts between multiple patches on the same function.
        - Feature flags (args) determine which enhancements are active.
        - Uses dependency injection pattern for ROCm memory stats (print_rank_last hook).
        - Uses argument inspection for MLflow metrics.
    """
    try:
        import megatron.training.training as megatron_training  # type: ignore

        # 1. Validation & Context Setup
        args = ctx.extra.get("args")
        if not args:
            log_rank_0("[Patch:megatron.training_log] No args in context, skipping")
            return

        # Check feature flags
        enable_mlflow = getattr(args, "enable_mlflow", False) or ctx.extra.get("mlflow_writer")

        enable_rocm_stats = (
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

        if not enable_mlflow and not enable_rocm_stats:
            log_rank_0("[Patch:megatron.training_log][SKIP] No training_log enhancements enabled")
            return

        # 2. Capture Original Function
        _original_training_log = megatron_training.training_log

        # 3. Define Unified Wrapper
        def _patched_training_log(*func_args, **func_kwargs):
            """
            Unified wrapper for training_log.
            """
            # --- MLflow Logic ---
            if enable_mlflow:
                try:
                    _log_to_mlflow(ctx, args, *func_args)
                except Exception as e:
                    # Don't fail training if MLflow logging fails
                    print(f"[Patch] MLflow logging failed: {e}")

            # --- ROCm Stats Logic ---
            if enable_rocm_stats:
                import megatron.training.training as training_module

                # Capture original dependency
                _orig_print_rank_last = training_module.print_rank_last

                # Define hook
                def _hooked_print_rank_last(log_string):
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
                            iter_list = getattr(args, "use_rocm_mem_info_iters", [])
                            if getattr(args, "use_rocm_mem_info", False) or iter_list:
                                local_rank = torch.cuda.current_device()
                                r_total, r_used, r_free = get_rocm_smi_mem_info(local_rank)
                                r_ratio = r_used / r_total
                                rocm_mem_str = (
                                    f" rocm mem usage/free/total/usage_ratio: "
                                    f"{r_used/1024**3:.2f}GB/{r_free/1024**3:.2f}GB/"
                                    f"{r_total/1024**3:.2f}GB/{r_ratio*100:.2f}% |"
                                )

                            log_string = log_string.replace(
                                " throughput per GPU",
                                f"{hip_mem_str}{rocm_mem_str} throughput per GPU",
                            )
                        except Exception:
                            pass

                    return _orig_print_rank_last(log_string)

                # Swap dependency
                training_module.print_rank_last = _hooked_print_rank_last

                try:
                    return _original_training_log(*func_args, **func_kwargs)
                finally:
                    training_module.print_rank_last = _orig_print_rank_last

            else:
                # Just run original if ROCm stats disabled but MLflow enabled
                return _original_training_log(*func_args, **func_kwargs)

        # 4. Apply Patch
        megatron_training.training_log = _patched_training_log
        log_rank_0("[Patch:megatron.training_log] Applied unified wrapper (MLflow + ROCm stats)")

    except ImportError as e:
        log_rank_0(f"[Patch:megatron.training_log][SKIP] Import failed: {e}")
    except AttributeError as e:
        log_rank_0(f"[Patch:megatron.training_log][WARN] Attribute error: {e}")


def _log_to_mlflow(ctx, args, *func_args):
    """Helper to extract metrics and log to MLflow."""
    # Unpack arguments based on Megatron signature
    # def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate,
    #                  iteration, loss_scale, report_memory_flag, skipped_iter,
    #                  grad_norm, params_norm, num_zeros_in_grad)
    if len(func_args) < 11:
        return

    (
        loss_dict,
        _,  # total_loss_dict
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        _,  # report_memory_flag
        _,  # skipped_iter
        grad_norm,
        params_norm,
        num_zeros_in_grad,
    ) = func_args[:11]

    # Check logging interval
    if iteration % getattr(args, "tensorboard_log_interval", 1) != 0:
        return

    mlflow_writer = _get_mlflow_writer(ctx)
    if not mlflow_writer:
        return

    # Log metrics
    mlflow_writer.log_metric("samples_vs_steps", args.consumed_train_samples, step=iteration)

    if learning_rate is not None:
        mlflow_writer.log_metric("learning_rate", learning_rate, step=iteration)

    if decoupled_learning_rate is not None:
        mlflow_writer.log_metric("decoupled_learning_rate", decoupled_learning_rate, step=iteration)

    batch_size = args.micro_batch_size * args.data_parallel_size * getattr(args, "num_microbatches", 1)
    mlflow_writer.log_metric("batch_size", batch_size, step=iteration)

    for key, value in loss_dict.items():
        metric_name = key.replace(" ", "_").replace("-", "_")
        mlflow_writer.log_metric(metric_name, value, step=iteration)

    if grad_norm is not None:
        mlflow_writer.log_metric("grad_norm", grad_norm, step=iteration)

    if params_norm is not None:
        mlflow_writer.log_metric("params_norm", params_norm, step=iteration)

    if num_zeros_in_grad is not None:
        mlflow_writer.log_metric("num_zeros_in_grad", num_zeros_in_grad, step=iteration)

    if loss_scale is not None:
        mlflow_writer.log_metric("loss_scale", loss_scale, step=iteration)

    if getattr(args, "log_memory_to_tensorboard", False):
        mem_stats = torch.cuda.memory_stats()
        mlflow_writer.log_metric(
            "mem_reserved_bytes", mem_stats["reserved_bytes.all.current"], step=iteration
        )
        mlflow_writer.log_metric(
            "mem_allocated_bytes", mem_stats["allocated_bytes.all.current"], step=iteration
        )
        mlflow_writer.log_metric(
            "mem_max_allocated_bytes", mem_stats["allocated_bytes.all.peak"], step=iteration
        )


def _get_mlflow_writer(ctx: PatchContext):
    """Get or create MLflow writer."""
    mlflow_writer = ctx.extra.get("mlflow_writer")
    if mlflow_writer:
        return mlflow_writer

    try:
        from primus.core.utils.logger import get_mlflow_writer

        return get_mlflow_writer()
    except (ImportError, AttributeError):
        return None

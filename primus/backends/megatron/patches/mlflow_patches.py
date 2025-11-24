###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MLflow Integration Patches

Patches Megatron's training_log function to add MLflow logging support.
"""

from primus.core.patches import PatchContext, register_patch


@register_patch(
    "megatron.mlflow.training_log",
    backend="megatron",
    phase="before_train",
    description="Add MLflow logging to Megatron training_log function",
)
def patch_training_log_for_mlflow(ctx: PatchContext):
    """
    Patch Megatron's training_log function to add MLflow logging.

    This patch wraps the original training_log function and adds MLflow
    metric logging alongside the existing TensorBoard and W&B logging.

    The patch adds logging for:
        - samples vs steps
        - learning-rate
        - batch-size
        - loss metrics
        - grad-norm
        - params-norm
        - memory stats (if enabled)
    """
    try:
        import megatron.training.training as megatron_training  # type: ignore

        # Store the original function
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
            model=None,
            optimizer=None,
            noise_scale_logger=None,
        ):
            """Patched training_log with MLflow support."""
            # Call the original function first
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
                model,
                optimizer,
                noise_scale_logger,
            )

            # Add MLflow logging
            try:
                # Get args from Megatron's global state
                from megatron.training import get_args  # type: ignore

                args = get_args()

                # Check if we should log (same interval as TensorBoard)
                if iteration % args.tensorboard_log_interval == 0:
                    # Try to get MLflow writer from context or create one
                    mlflow_writer = _get_mlflow_writer(ctx)

                    if mlflow_writer:
                        # Log samples vs steps
                        mlflow_writer.log_metric(
                            "samples_vs_steps", args.consumed_train_samples, step=iteration
                        )

                        # Log learning rate
                        if learning_rate is not None:
                            mlflow_writer.log_metric("learning_rate", learning_rate, step=iteration)

                        # Log decoupled learning rate if applicable
                        if decoupled_learning_rate is not None:
                            mlflow_writer.log_metric(
                                "decoupled_learning_rate", decoupled_learning_rate, step=iteration
                            )

                        # Log batch size
                        batch_size = (
                            args.micro_batch_size
                            * args.data_parallel_size
                            * getattr(args, "num_microbatches", 1)
                        )
                        mlflow_writer.log_metric("batch_size", batch_size, step=iteration)

                        # Log all loss metrics
                        for key, value in loss_dict.items():
                            # Replace spaces with underscores for MLflow
                            metric_name = key.replace(" ", "_").replace("-", "_")
                            mlflow_writer.log_metric(metric_name, value, step=iteration)

                        # Log gradient norm
                        if grad_norm is not None:
                            mlflow_writer.log_metric("grad_norm", grad_norm, step=iteration)

                        # Log params norm
                        if params_norm is not None:
                            mlflow_writer.log_metric("params_norm", params_norm, step=iteration)

                        # Log num zeros in gradient
                        if num_zeros_in_grad is not None:
                            mlflow_writer.log_metric("num_zeros_in_grad", num_zeros_in_grad, step=iteration)

                        # Log loss scale
                        if loss_scale is not None:
                            mlflow_writer.log_metric("loss_scale", loss_scale, step=iteration)

                        # Log memory stats if enabled
                        if getattr(args, "log_memory_to_tensorboard", False):
                            import torch

                            mem_stats = torch.cuda.memory_stats()
                            mlflow_writer.log_metric(
                                "mem_reserved_bytes",
                                mem_stats["reserved_bytes.all.current"],
                                step=iteration,
                            )
                            mlflow_writer.log_metric(
                                "mem_allocated_bytes",
                                mem_stats["allocated_bytes.all.current"],
                                step=iteration,
                            )
                            mlflow_writer.log_metric(
                                "mem_max_allocated_bytes",
                                mem_stats["allocated_bytes.all.peak"],
                                step=iteration,
                            )

            except Exception as e:
                # Don't fail training if MLflow logging fails
                print(f"[Patch] MLflow logging failed: {e}")

            return result

        # Replace the original function with the patched version
        megatron_training.training_log = patched_training_log
        print("[Patch] Successfully patched training_log for MLflow support")

    except ImportError as e:
        print(f"[Patch] Failed to patch training_log: {e}")


def _get_mlflow_writer(ctx: PatchContext):
    """
    Get or create MLflow writer.

    This function tries to get the MLflow writer from:
    1. Context extra data
    2. Primus global state
    3. Create a new one if needed

    Returns:
        MLflow writer instance or None
    """
    # Try to get from context
    mlflow_writer = ctx.extra.get("mlflow_writer")
    if mlflow_writer:
        return mlflow_writer

    # Try to get from Primus config
    try:
        from primus.core.utils.logger import get_mlflow_writer

        mlflow_writer = get_mlflow_writer()
        if mlflow_writer:
            return mlflow_writer
    except (ImportError, AttributeError):
        pass

    # If no MLflow writer is available, return None
    # (MLflow logging will be skipped)
    return None

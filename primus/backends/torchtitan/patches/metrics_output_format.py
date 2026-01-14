###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Metrics Output Format Patch

This patch customizes the console output format of training metrics in
TorchTitan's MetricsProcessor.log() method, using a surgical approach that
only intercepts the logger.info() call without modifying any other logic.
"""

from typing import Any

from primus.core.patches import PatchContext


# @register_patch(
#     "torchtitan.metrics.output_format",
#     backend="torchtitan",
#     phase="setup",
#     description="Customize TorchTitan metrics console output format",
# )
def patch_metrics_output_format(ctx: PatchContext) -> None:
    """
    Customize the console output format of training metrics.

    This patch only intercepts the logger.info() call in MetricsProcessor.log()
    without modifying the metric calculation, TensorBoard/WandB logging, or
    counter reset logic.

    Original format:
        step: 1  loss: 10.1234  grad_norm: 1.2345  memory: 5.23GiB(45.67%)  ...

    Custom format (Megatron-style single-line):
        iteration        1/      50 | consumed samples:          128 | elapsed time per iteration (ms): 19392.0 |
        memory usage/free/total: 145.40GiB/142.58GiB(50.49%) | throughput per GPU (TFLOP/s/GPU): 347.9 |
        tokens per GPU (tokens/s/GPU): 6759.1 | global batch size:   128 | loss: 1.189155E+01 |
        grad norm: 7.872 | mfu: 45.67% |
    """
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.tools import logging as titan_logging

    # Store original method for potential restoration
    original_log = MetricsProcessor.log

    def wrapped_log(
        self,
        step: int,
        global_avg_loss: float,
        global_max_loss: float,
        grad_norm: float,
        extra_metrics: dict[str, Any] | None = None,
    ):
        """
        Thin wrapper that intercepts logger.info() to customize output format.

        All original logic (metric calculation, backend logging, counter resets)
        runs unchanged. Only the console output format is customized.
        """
        import time

        # Pre-calculate metrics ONCE (same calculations as in original method at line 398-408)
        # This avoids duplicate calculation - we compute them here and use in custom_info
        time_delta = time.perf_counter() - self.time_last_log
        tps = self.ntokens_since_last_log / (time_delta * self.parallel_dims.non_data_parallel_size)
        mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
        tflops = self.num_flops_per_token * tps / 1e12
        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        # Extract additional info from job_config and state
        total_steps = self.job_config.training.steps

        # Handle global_batch_size calculation (same logic as train.py:199-202)
        # If global_batch_size < 0, it defaults to local_batch_size * dp_degree
        global_batch_size = self.job_config.training.global_batch_size
        if global_batch_size < 0:
            # Calculate actual global batch size from local batch size and data-parallel degree
            # dp_degree = dp_replicate * dp_shard (see parallel_dims.py:245)
            dp_degree = self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard
            global_batch_size = self.job_config.training.local_batch_size * dp_degree

        consumed_samples = step * global_batch_size

        # Calculate average elapsed time (ms)
        elapsed_time_ms = time_delta * 1000

        # Save original logger.info
        original_logger_info = titan_logging.logger.info

        # Track if we've intercepted the metrics output
        metrics_call_intercepted = False

        def custom_info(msg: str, *args, **kwargs):
            """
            Intercept logger.info and customize metrics output.

            Detects the metrics output call by checking for key indicators
            (step:, loss:, mfu:) and replaces it with custom format.
            Non-metrics calls pass through unchanged.

            Note: Uses pre-calculated metrics from outer scope (time_delta, tps, etc.)
            to avoid duplicate calculation.
            """
            nonlocal metrics_call_intercepted

            # Detect metrics output by checking for characteristic keywords
            if not metrics_call_intercepted and "step:" in msg and "loss:" in msg and "mfu:" in msg:
                metrics_call_intercepted = True

                # Use pre-calculated metrics from outer scope (no re-calculation!)
                # Format: Megatron-style single-line output with all metrics
                output_parts = [
                    f"iteration {step:8d}/{total_steps:8d}",
                    f"consumed samples: {consumed_samples:12d}",
                    f"elapsed time per iteration (ms): {elapsed_time_ms:.1f}",
                    f"memory usage/free/total: {device_mem_stats.max_reserved_gib:.2f}GiB/"
                    f"{device_mem_stats.max_active_gib:.2f}GiB/"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)",
                    f"throughput per GPU (TFLOP/s/GPU): {tflops:.1f}",
                    f"tokens per GPU (tokens/s/GPU): {tps:.1f}",
                    f"global batch size: {global_batch_size:5d}",
                    f"loss: {global_avg_loss:.6E}",
                    f"grad norm: {grad_norm:.3f}",
                    f"mfu: {mfu:.2f}%",
                ]

                # Join with " | " separator for single-line output
                formatted_msg = " | ".join(output_parts) + " |"
                original_logger_info(formatted_msg)
            else:
                # Not the metrics call, pass through
                original_logger_info(msg, *args, **kwargs)

        # Temporarily replace logger.info
        titan_logging.logger.info = custom_info

        try:
            # Call original method - all logic runs normally
            # Note: The original method will re-calculate the same metrics,
            # but that's unavoidable without modifying TorchTitan's code
            original_log(
                self,
                step=step,
                global_avg_loss=global_avg_loss,
                global_max_loss=global_max_loss,
                grad_norm=grad_norm,
                extra_metrics=extra_metrics,
            )
        finally:
            # Always restore original logger.info
            titan_logging.logger.info = original_logger_info

    # Replace the method with our wrapper
    MetricsProcessor.log = wrapped_log

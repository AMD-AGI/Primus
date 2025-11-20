###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Profiling utilities for TorchTitan with Primus extensions.
This module provides patching functions to enhance TorchTitan's profiling capabilities.
"""

import contextlib
import os
import time


def patch_maybe_enable_profiling(profiling_module, primus_logger):
    """
    Patch torchtitan profiling to support profile_ranks parameter and json.gz format.
    
    This function:
    1. Adds the ability to specify which ranks should be profiled via profile_ranks
    2. Saves profiling traces in compressed json.gz format instead of json
    
    Args:
        profiling_module: The torchtitan.tools.profiling module to patch
        primus_logger: Logger instance for logging patch status
    
    Returns:
        None
    """
    import torch
    from torchtitan.tools.logging import logger
    
    # Save the original function
    original_maybe_enable_profiling = profiling_module.maybe_enable_profiling
    
    @contextlib.contextmanager
    def patched_maybe_enable_profiling(profiling_config, *, global_step=0, base_folder="", leaf_folder=""):
        """
        Patched version of maybe_enable_profiling that checks profile_ranks and saves as json.gz.
        
        Args:
            profiling_config: Profiling configuration object
            global_step: Current training step number
            base_folder: Base directory for profiling outputs
            leaf_folder: Subdirectory within the base folder
            
        Yields:
            torch_profiler: PyTorch profiler object or None
        """
        enable_profiling = profiling_config.enable_profiling
        
        if enable_profiling:
            # Check if profile_ranks attribute exists and current rank should be profiled
            if hasattr(profiling_config, 'profile_ranks'):
                rank = torch.distributed.get_rank()
                if rank not in profiling_config.profile_ranks:
                    primus_logger.info(f"Rank {rank} not in profile_ranks {profiling_config.profile_ranks}, skipping profiling")
                    yield None
                    return
            
            # Create custom trace handler that saves as json.gz
            trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
            profile_freq, warmup, active = (
                profiling_config.profile_freq,
                profiling_config.profiler_warmup,
                profiling_config.profiler_active,
            )
            rank = torch.distributed.get_rank()
            
            def trace_handler(prof):
                curr_trace_dir_name = "iteration_" + str(prof.step_num)
                curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name, leaf_folder)
                if not os.path.exists(curr_trace_dir):
                    os.makedirs(curr_trace_dir, exist_ok=True)

                logger.info(f"Dumping profiler traces at step {prof.step_num}")
                begin = time.monotonic()

                # Save as .json.gz instead of .json
                output_file = os.path.join(curr_trace_dir, f"rank{rank}_trace.json")
                prof.export_chrome_trace(output_file)
                logger.info(
                    f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
                )
            
            logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

            if not os.path.exists(trace_dir):
                os.makedirs(trace_dir, exist_ok=True)

            wait = profile_freq - (active + warmup)
            assert (
                wait >= 0
            ), "profile_freq must be greater than or equal to warmup + active"
            
            # Always profile CPU activity
            activities = [torch.profiler.ProfilerActivity.CPU]
            
            # Add GPU profiling if available
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
                logger.info("Profiling CPU and CUDA activities")
            elif torch.xpu.is_available():
                activities.append(torch.profiler.ProfilerActivity.XPU)
                logger.info("Profiling CPU and XPU activities")
            else:
                logger.info("Profiling CPU activity only (no GPU detected)")
            
            with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
                on_trace_ready=trace_handler,
                record_shapes=True,
                with_stack=True,  # Enable stack traces for better CPU profiling
                profile_memory=True,  # Track memory allocations/deallocations
                with_flops=True,  # Estimate FLOPs for operations
            ) as torch_profiler:
                torch_profiler.step_num = global_step
                yield torch_profiler
        else:
            yield None
    
    # Replace the function
    profiling_module.maybe_enable_profiling = patched_maybe_enable_profiling
    primus_logger.info("Successfully patched torchtitan profiling with profile_ranks support and json.gz format")


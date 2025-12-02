###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from typing import List, Tuple, Union

import torch


def benchmark_layer(
    layer_module: torch.nn.Module,
    input_shapes: List[Union[Tuple[int, ...], Tuple[Tuple[int, ...], torch.dtype]]],
    num_iterations: int = 10,
) -> tuple[float, float, int]:
    """
    Benchmark both forward and backward passes of a transformer layer using CUDA events.
    Also measures activation memory used by the forward pass.

    Args:
        layer_module: The transformer layer module
        input_shapes: List of input shapes. Each element can be:
                      - A tuple of integers (shape), defaults to bfloat16 and requires_grad=True
                      - A tuple of ((shape), dtype), defaults to requires_grad=True if float, False otherwise
        num_iterations: Number of iterations to average over

    Returns:
        Tuple of (average forward time in ms, average backward time in ms, activation memory in bytes)
    """
    # Get device from module parameters
    try:
        device = next(layer_module.parameters()).device
    except StopIteration:
        # Fallback if module has no parameters (unlikely for profiled modules)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_input(spec):
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], torch.dtype):
            shape, dtype = spec
        else:
            shape = spec
            dtype = torch.bfloat16

        requires_grad = dtype in (torch.float16, torch.float32, torch.bfloat16)

        if dtype == torch.bool:
            return torch.randint(0, 2, shape, device=device, dtype=dtype)
        elif dtype in (torch.int32, torch.int64):
            return torch.randint(0, 100, shape, device=device, dtype=dtype)
        else:
            return torch.randn(
                *shape,
                device=device,
                dtype=dtype,
                requires_grad=requires_grad,
            )

    inputs = [create_input(spec) for spec in input_shapes]

    # Warm-up: forward and backward passes
    for _ in range(3):
        outputs = layer_module(*inputs)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        # Backward on all outputs that require grad
        grad_outputs = []
        valid_outputs = []
        for out in outputs:
            if isinstance(out, torch.Tensor) and out.requires_grad:
                grad_outputs.append(torch.randn_like(out))
                valid_outputs.append(out)

        if valid_outputs:
            torch.autograd.backward(valid_outputs, grad_outputs)

        layer_module.zero_grad()
        for inp in inputs:
            if inp.requires_grad:
                inp.grad = None

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Measure forward and backward passes using CUDA events
    forward_times = []
    backward_times = []
    activation_memories = []

    # Clear cache and reset memory stats before measuring forward pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)
    outputs_cache = []

    # First loop: Measure forward pass only
    for _ in range(num_iterations):
        # Measure forward pass
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)

        forward_start.record()
        outputs = layer_module(*inputs)
        outputs_cache.append(outputs)
        forward_end.record()

        # Wait for forward pass to complete
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # Record forward time
        forward_times.append(forward_start.elapsed_time(forward_end))

    # Measure activation memory (peak memory during forward - baseline)
    mem_after_forward = torch.cuda.max_memory_allocated(device)
    activation_memory = (mem_after_forward - mem_before) // num_iterations
    activation_memories.append(activation_memory)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    del outputs_cache

    # Second loop: Measure backward pass only
    for _ in range(num_iterations):
        # Forward pass (not timed)
        outputs = layer_module(*inputs)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        grad_outputs = []
        valid_outputs = []
        for out in outputs:
            if isinstance(out, torch.Tensor) and out.requires_grad:
                grad_outputs.append(torch.randn_like(out))
                valid_outputs.append(out)

        # Measure backward pass
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)

        backward_start.record()
        if valid_outputs:
            torch.autograd.backward(valid_outputs, grad_outputs)
        backward_end.record()

        # Wait for backward pass to complete
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Record backward time
        backward_times.append(backward_start.elapsed_time(backward_end))

        # Clear gradients for next iteration
        layer_module.zero_grad()
        for inp in inputs:
            if inp.requires_grad:
                inp.grad = None
        del outputs, grad_outputs, valid_outputs

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    avg_activation_memory = (
        int(sum(activation_memories) / len(activation_memories)) if activation_memories else 0
    )

    return avg_forward_time, avg_backward_time, avg_activation_memory

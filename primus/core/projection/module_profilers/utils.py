###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from contextlib import nullcontext
from typing import List, Tuple, Union

import torch


class _FP8ContextFactory:
    """Factory that creates fresh FP8 contexts on each `with` statement."""

    def __init__(self, transformer_config):
        self.transformer_config = transformer_config
        self.fp8_enabled = (
            getattr(transformer_config, "fp8", None) if transformer_config else None
        )
        self._printed = False

    def __enter__(self):
        if not self.fp8_enabled:
            self._ctx = nullcontext()
        else:
            try:
                from primus.backends.megatron.core.fp8_utils import get_fp8_context

                self._ctx = get_fp8_context(self.transformer_config, layer_no=-1)
                if not self._printed:
                    print(
                        f"  [FP8] Using FP8 autocast context for benchmarking (fp8={self.fp8_enabled})"
                    )
                    self._printed = True
            except Exception as e:
                try:
                    import transformer_engine.pytorch as te

                    self._ctx = te.fp8_autocast(enabled=True)
                    if not self._printed:
                        print("  [FP8] Using TE fp8_autocast fallback for benchmarking")
                        self._printed = True
                except Exception:
                    if not self._printed:
                        print(f"  [FP8] Warning: Could not enable FP8 context: {e}")
                        self._printed = True
                    self._ctx = nullcontext()
        return self._ctx.__enter__()

    def __exit__(self, *args):
        return self._ctx.__exit__(*args)


def _get_fp8_context_for_benchmark(transformer_config):
    """Get FP8 context factory for benchmarking if FP8 is enabled."""
    return _FP8ContextFactory(transformer_config)


def benchmark_layer(
    layer_module: torch.nn.Module,
    input_shapes: List[Union[Tuple[int, ...], Tuple[Tuple[int, ...], torch.dtype]]],
    num_iterations: int = 64,  # Match typical microbatch count
    transformer_config=None,  # Optional: pass config to enable FP8 context
) -> tuple[float, float, int]:
    """
    Benchmark both forward and backward passes of a transformer layer using CUDA events.

    Optimizations for accurate timing:
    1. Warmup (20 iterations) to fully warm GPU caches and JIT
    2. Many benchmark iterations (64) for stable steady-state measurement
    3. Separate forward/backward timing with CUDA events for accurate splits
    4. Pre-allocated grad_outputs tensors reused across iterations
    5. FP8 autocast context when enabled (critical for accurate FP8 timing!)

    Args:
        layer_module: The transformer layer module
        input_shapes: List of input shapes. Each element can be:
                      - A tuple of integers (shape), defaults to bfloat16 and requires_grad=True
                      - A tuple of ((shape), dtype), defaults to requires_grad=True if float, False otherwise
        num_iterations: Number of iterations to average over
        transformer_config: Optional TransformerConfig to enable FP8 autocast

    Returns:
        Tuple of (average forward time in ms, average backward time in ms, activation memory in bytes)
    """
    # Get device from module parameters
    try:
        device = next(layer_module.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_input(spec):
        if (
            isinstance(spec, tuple)
            and len(spec) == 2
            and isinstance(spec[1], torch.dtype)
        ):
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

    # ===========================================================================
    # Get FP8 context - CRITICAL for accurate FP8 timing!
    # ===========================================================================
    fp8_context = _get_fp8_context_for_benchmark(transformer_config)

    # ===========================================================================
    # WARMUP (20 iterations) - fully warm GPU caches, JIT, and allocators
    # This matches the "warmed up" state of actual training after many iterations
    # ===========================================================================
    grad_outputs = None
    output_indices = []
    num_warmup = 20  # Many warmup iterations to match sustained training state

    with fp8_context:
        for _ in range(num_warmup):
            outputs = layer_module(*inputs)
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)

            # Create grad_outputs once during warmup (reused for all subsequent iterations)
            if grad_outputs is None:
                grad_outputs = []
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        grad_outputs.append(torch.randn_like(out))
                        output_indices.append(i)

            valid_outputs = [outputs[i] for i in output_indices]
            if valid_outputs:
                torch.autograd.backward(valid_outputs, grad_outputs)

            layer_module.zero_grad(set_to_none=True)
            for inp in inputs:
                if inp.requires_grad:
                    inp.grad = None

    # Synchronize after warmup - GPU is now in "hot" state
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # --- Measure activation memory (forward-only loop) ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mem_before = torch.cuda.memory_allocated(device)

    with fp8_context:
        for _ in range(num_iterations):
            outputs = layer_module(*inputs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mem_after_forward = torch.cuda.max_memory_allocated(device)
    activation_memory = (mem_after_forward - mem_before) // num_iterations

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    del outputs

    # ===========================================================================
    # BENCHMARK: Measure forward and backward separately with CUDA events
    # ===========================================================================
    forward_times = []
    backward_times = []

    with fp8_context:
        for _ in range(num_iterations):
            # --- Forward pass ---
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)

            forward_start.record()
            outputs = layer_module(*inputs)
            forward_end.record()

            # --- Backward pass ---
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)

            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            valid_outputs = [outputs[i] for i in output_indices]

            backward_start.record()
            if valid_outputs:
                torch.autograd.backward(valid_outputs, grad_outputs)
            backward_end.record()

            torch.cuda.synchronize(device)

            forward_times.append(forward_start.elapsed_time(forward_end))
            backward_times.append(backward_start.elapsed_time(backward_end))

            layer_module.zero_grad(set_to_none=True)
            for inp in inputs:
                if inp.requires_grad:
                    inp.grad = None

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)

    return avg_forward_time, avg_backward_time, int(activation_memory)

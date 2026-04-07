###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch


def _backward_autograd_debug_enabled() -> bool:
    if int(os.getenv("RANK", "0")) != 0:
        return False
    v = os.getenv("PRIMUS_PROJECTION_BACKWARD_AUTOGRAD_DEBUG", "1").lower()
    return v not in ("0", "false", "no", "off")


def _print_backward_autograd_enter(
    label: str,
    args_line: Optional[str],
    module: torch.nn.Module,
) -> None:
    if not _backward_autograd_debug_enabled() or not label:
        return
    print(f"  [autograd.backward] {label}  (nn.Module: {module.__class__.__name__})")
    if args_line:
        print(f"    config: {args_line}")


def _maybe_cuda_backward_kernel_table(
    layer_module: torch.nn.Module,
    inputs: list,
    output_indices: list,
    grad_outputs: list,
    fp8_context,
) -> None:
    """Optional one-shot torch.profiler after timed benchmark (does not affect reported ms)."""
    if os.getenv("PRIMUS_PROJECTION_BACKWARD_CUDA_PROFILER", "").lower() not in ("1", "true", "yes"):
        return
    if int(os.getenv("RANK", "0")) != 0:
        return
    try:
        _dev = next(layer_module.parameters()).device
    except StopIteration:
        return
    if _dev.type != "cuda":
        return
    try:
        from torch.profiler import ProfilerActivity, profile
    except ImportError:
        print("  [autograd.backward] torch.profiler unavailable; skipping CUDA kernel table.")
        return
    device = next(layer_module.parameters()).device
    print("  [autograd.backward] CUDA kernel summary (one untimed fwd+bwd, PRIMUS_PROJECTION_BACKWARD_CUDA_PROFILER=1):")
    with fp8_context:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
            outputs = layer_module(*inputs)
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            valid_outputs = [outputs[i] for i in output_indices]
            if valid_outputs:
                torch.autograd.backward(valid_outputs, grad_outputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    layer_module.zero_grad(set_to_none=True)
    for inp in inputs:
        if getattr(inp, "requires_grad", False) and inp.grad is not None:
            inp.grad = None
    print(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    )


class _FP8ContextFactory:
    """Factory that creates fresh FP8 contexts on each `with` statement."""

    def __init__(self, transformer_config):
        self.transformer_config = transformer_config
        self.fp8_enabled = getattr(transformer_config, "fp8", None) if transformer_config else None
        self._printed = False

    def __enter__(self):
        if not self.fp8_enabled:
            self._ctx = nullcontext()
        else:
            try:
                from primus.backends.megatron.core.fp8_utils import get_fp8_context

                self._ctx = get_fp8_context(self.transformer_config, layer_no=-1)
                if not self._printed:
                    print(f"  [FP8] Using FP8 autocast context for benchmarking (fp8={self.fp8_enabled})")
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
    backward_autograd_label: Optional[str] = None,
    backward_autograd_args: Optional[str] = None,
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
        backward_autograd_label: If set (rank 0), print this label when ``torch.autograd.backward``
            runs in the timed loop (first iteration only).  Disable prints with
            ``PRIMUS_PROJECTION_BACKWARD_AUTOGRAD_DEBUG=0``.
        backward_autograd_args: Optional one-line config string (e.g. from
            :func:`training_config_debug_one_line`).

    Returns:
        Tuple of (average forward time in ms, average backward time in ms, activation memory in bytes)
    """
    # Get device from module parameters
    try:
        device = next(layer_module.parameters()).device
    except StopIteration:
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
    _first_timed_backward = True

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

            if _first_timed_backward and backward_autograd_label:
                _print_backward_autograd_enter(
                    backward_autograd_label, backward_autograd_args, layer_module
                )
                _first_timed_backward = False

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

    _maybe_cuda_backward_kernel_table(
        layer_module, inputs, output_indices, grad_outputs, fp8_context
    )

    return avg_forward_time, avg_backward_time, int(activation_memory)


def benchmark_moe_layer_decomposed(
    moe_module: torch.nn.Module,
    input_shapes: List[Union[Tuple[int, ...], Tuple[Tuple[int, ...], torch.dtype]]],
    num_iterations: int = 64,
    transformer_config=None,
    backward_autograd_label: Optional[str] = None,
    backward_autograd_args: Optional[str] = None,
) -> tuple[float, float, int, float, float]:
    """
    Benchmark an MoE layer with decomposed A2A timing.

    This function works exactly like ``benchmark_layer`` but additionally
    measures the All-to-All dispatch and combine times separately by
    monkey-patching the MoE module's ``dispatch`` and ``combine`` methods
    with CUDA event timing.

    The A2A times can then be used for accurate EP scaling: when EP changes,
    the measured compute (total - A2A) is kept, and the A2A is replaced with
    an analytical estimate for the target EP.

    Args:
        moe_module: The MoE layer module (``MoELayer``).
        input_shapes: List of input shapes (same as ``benchmark_layer``).
        num_iterations: Number of benchmark iterations.
        transformer_config: Optional config for FP8 context.

    Returns:
        Tuple of (avg_forward_ms, avg_backward_ms, activation_memory_bytes,
                  avg_a2a_forward_ms, avg_a2a_backward_ms)
        where a2a_forward/backward is the dispatch+combine time per direction.
    """
    # Get device from module parameters
    try:
        device = next(moe_module.parameters()).device
    except StopIteration:
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
    fp8_context = _get_fp8_context_for_benchmark(transformer_config)

    # =========================================================================
    # WARMUP (20 iterations) - same as benchmark_layer
    # =========================================================================
    grad_outputs = None
    output_indices = []
    num_warmup = 20

    with fp8_context:
        for _ in range(num_warmup):
            outputs = moe_module(*inputs)
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)

            if grad_outputs is None:
                grad_outputs = []
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        grad_outputs.append(torch.randn_like(out))
                        output_indices.append(i)

            valid_outputs = [outputs[i] for i in output_indices]
            if valid_outputs:
                torch.autograd.backward(valid_outputs, grad_outputs)

            moe_module.zero_grad(set_to_none=True)
            for inp in inputs:
                if inp.requires_grad:
                    inp.grad = None

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
            outputs = moe_module(*inputs)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mem_after_forward = torch.cuda.max_memory_allocated(device)
    activation_memory = (mem_after_forward - mem_before) // num_iterations

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    del outputs

    # =========================================================================
    # BENCHMARK with decomposed A2A timing
    # =========================================================================
    # Monkey-patch dispatch() and combine() to insert CUDA events.
    # MoELayer.forward() calls self.dispatch(...) and self.combine(...)
    # so instance-attribute patches are picked up by Python's MRO.
    original_dispatch = moe_module.dispatch
    original_combine = moe_module.combine

    # Accumulate (start_event, end_event) pairs per iteration
    _dispatch_events = []  # one (start, end) per iteration
    _combine_events = []

    def timed_dispatch(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_dispatch(*args, **kwargs)
        end.record()
        _dispatch_events.append((start, end))
        return result

    def timed_combine(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_combine(*args, **kwargs)
        end.record()
        _combine_events.append((start, end))
        return result

    moe_module.dispatch = timed_dispatch
    moe_module.combine = timed_combine

    forward_times = []
    backward_times = []
    _first_timed_backward = True

    try:
        with fp8_context:
            for _ in range(num_iterations):
                # --- Forward pass ---
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)

                forward_start.record()
                outputs = moe_module(*inputs)
                forward_end.record()

                # --- Backward pass ---
                backward_start = torch.cuda.Event(enable_timing=True)
                backward_end = torch.cuda.Event(enable_timing=True)

                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)
                valid_outputs = [outputs[i] for i in output_indices]

                if _first_timed_backward and backward_autograd_label:
                    _print_backward_autograd_enter(
                        backward_autograd_label, backward_autograd_args, moe_module
                    )
                    _first_timed_backward = False

                backward_start.record()
                if valid_outputs:
                    torch.autograd.backward(valid_outputs, grad_outputs)
                backward_end.record()

                torch.cuda.synchronize(device)

                forward_times.append(forward_start.elapsed_time(forward_end))
                backward_times.append(backward_start.elapsed_time(backward_end))

                moe_module.zero_grad(set_to_none=True)
                for inp in inputs:
                    if inp.requires_grad:
                        inp.grad = None
    finally:
        # Always restore original methods
        moe_module.dispatch = original_dispatch
        moe_module.combine = original_combine

    _maybe_cuda_backward_kernel_table(
        moe_module, inputs, output_indices, grad_outputs, fp8_context
    )

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)

    # Compute average A2A forward time (dispatch + combine)
    # Each forward iteration produces one dispatch event and one combine event.
    a2a_fwd_times = []
    for (ds, de), (cs, ce) in zip(_dispatch_events, _combine_events):
        dispatch_ms = ds.elapsed_time(de)
        combine_ms = cs.elapsed_time(ce)
        a2a_fwd_times.append(dispatch_ms + combine_ms)

    avg_a2a_fwd = sum(a2a_fwd_times) / len(a2a_fwd_times) if a2a_fwd_times else 0.0

    # Backward A2A is approximately equal to forward A2A (same message sizes).
    # We don't instrument the backward autograd graph, so we use this assumption.
    avg_a2a_bwd = avg_a2a_fwd

    is_rank_0 = int(os.getenv("RANK", "0")) == 0
    if is_rank_0:
        compute_fwd = avg_forward_time - avg_a2a_fwd
        compute_bwd = avg_backward_time - avg_a2a_bwd
        print(f"  [MoE Decomposed] Total fwd: {avg_forward_time:.2f} ms, bwd: {avg_backward_time:.2f} ms")
        print(f"  [MoE Decomposed] A2A   fwd: {avg_a2a_fwd:.2f} ms, bwd(est): {avg_a2a_bwd:.2f} ms")
        print(f"  [MoE Decomposed] Compute fwd: {compute_fwd:.2f} ms, bwd: {compute_bwd:.2f} ms")

    return avg_forward_time, avg_backward_time, int(activation_memory), avg_a2a_fwd, avg_a2a_bwd

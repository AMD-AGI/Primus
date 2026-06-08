###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


import os
from contextlib import nullcontext
from typing import List, Tuple, Union

import torch


def _bench_iter_count(default_warmup: int, default_iters: int) -> Tuple[int, int]:
    """Look up bench iteration counts, allowing runtime override.

    Env vars (intended for debugging slow MoE benches that hit NCCL
    collective watchdog timeouts):

    * ``PRIMUS_BENCH_WARMUP_ITERS``    — replaces hardcoded warmup count
    * ``PRIMUS_BENCH_NUM_ITERATIONS``  — replaces ``num_iterations`` default

    Defaults preserve original measurement fidelity (20 warmup, 64 iters).
    """
    try:
        warmup = int(os.environ.get("PRIMUS_BENCH_WARMUP_ITERS", default_warmup))
    except ValueError:
        warmup = default_warmup
    try:
        iters = int(os.environ.get("PRIMUS_BENCH_NUM_ITERATIONS", default_iters))
    except ValueError:
        iters = default_iters
    return max(1, warmup), max(1, iters)


def _bench_central_value(times: list[float]) -> float:
    """Compute the per-iteration central value used by the bench reporters.

    Defaults to *median* so a single slow first-iter (e.g. a backward JIT
    compile that wasn't covered by warmup) does not poison the average.
    Set ``PRIMUS_BENCH_AGGREGATE=mean`` to recover the historical mean.
    """
    if not times:
        return 0.0
    mode = os.environ.get("PRIMUS_BENCH_AGGREGATE", "median").strip().lower()
    if mode == "mean":
        return sum(times) / len(times)
    sorted_times = sorted(times)
    n = len(sorted_times)
    mid = n // 2
    if n % 2 == 1:
        return sorted_times[mid]
    return 0.5 * (sorted_times[mid - 1] + sorted_times[mid])


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
    num_warmup, num_iterations = _bench_iter_count(20, num_iterations)

    with fp8_context:
        for _ in range(num_warmup):
            outputs = layer_module(*inputs)
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)

            # Create grad_outputs once during warmup (reused for all subsequent iterations)
            if grad_outputs is None:
                grad_outputs = []
                for i, out in enumerate(outputs):
                    # Filter on ``grad_fn is not None`` rather than
                    # ``requires_grad`` so we only pick outputs that are
                    # actually nodes in the autograd graph.  ``requires_grad``
                    # alone is True for leaf parameters and for tensors that
                    # came out of a custom autograd Function whose backward
                    # link was dropped -- calling ``torch.autograd.backward``
                    # on those does NOT walk the forward graph, which means
                    # saved-for-backward activations are never released and
                    # get destructed inside the *next* forward call (showing
                    # up as a 100-1000x inflation of forward time on MoE
                    # transformer layers under ``rec=none``).
                    if isinstance(out, torch.Tensor) and out.requires_grad and out.grad_fn is not None:
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

    avg_forward_time = _bench_central_value(forward_times)
    avg_backward_time = _bench_central_value(backward_times)

    return avg_forward_time, avg_backward_time, int(activation_memory)


# ─────────────────────────────────────────────────────────────────────────────
# MoE kernel-shape padding (deterministic balanced routing for benchmarking)
# ─────────────────────────────────────────────────────────────────────────────
#
# Production MoE forward passes use stochastic top-k routing, which means the
# per-expert token count (the ``M`` dimension of each expert's grouped GEMM)
# varies from one iteration to the next.  On ROCm composable_kernel the
# FP8 grouped-GEMM JIT compiles a fresh kernel for every unique ``M`` tile it
# observes, so a 64-iteration MoE benchmark with O(256) experts and topk=8 can
# easily trigger 100+ JIT compilations -- pushing per-iter time to >20 s and
# making whole-bench runtimes impractical for models like DeepSeek v3.
#
# The fix below pins ``M`` by replacing the router's ``routing()`` with a
# deterministic round-robin assignment (token i, slot j → expert
# ``(i * topk + j) % num_experts``).  Every expert receives exactly the same
# number of tokens on every iteration, so the JIT cache hits after a single
# compile.  This is the same load-balancing technique Megatron already uses
# under ``moe_router_force_load_balancing`` for benchmarking, but installed
# directly from the projection bench code so it works regardless of which
# router class (upstream / Primus / Primus-Turbo) the model happens to use.
#
# The patch is enabled by default during projection benches and can be
# disabled by setting ``PRIMUS_BENCH_MOE_KERNEL_PAD=0``.


def _kernel_pad_enabled() -> bool:
    raw = os.environ.get("PRIMUS_BENCH_MOE_KERNEL_PAD", "1").strip().lower()
    return raw not in {"", "0", "false", "no", "off"}


def _looks_like_router(submodule: torch.nn.Module) -> bool:
    """Heuristic: does ``submodule`` look like a TopK MoE router?"""
    if not callable(getattr(submodule, "routing", None)):
        return False
    cfg = getattr(submodule, "config", None)
    if cfg is None or not hasattr(cfg, "num_moe_experts"):
        return False
    if getattr(submodule, "topk", None) is None and not hasattr(cfg, "moe_router_topk"):
        return False
    return True


def _install_balanced_routing_patches(moe_module: torch.nn.Module):
    """Replace ``routing()`` on every router-like submodule with a deterministic
    round-robin assignment so the per-expert ``M`` dimension is constant
    across bench iterations.

    Returns a tuple ``(restore_callables, descriptors)``.  ``descriptors`` is a
    list of ``(qualified_name, topk, num_experts)`` entries suitable for
    logging.  ``restore_callables`` should be invoked (in any order) to undo
    the patch -- typically inside a ``finally`` block.
    """
    restores = []
    descriptors = []
    seen = set()

    for name, sub in moe_module.named_modules():
        if id(sub) in seen or not _looks_like_router(sub):
            continue
        seen.add(id(sub))

        cfg = sub.config
        topk = int(getattr(sub, "topk", None) or cfg.moe_router_topk)
        num_experts = int(cfg.num_moe_experts)
        if topk <= 0 or num_experts <= 0:
            continue

        original_routing = sub.routing

        def make_balanced(k=topk, e=num_experts):
            def balanced_routing(logits, *args, **kwargs):
                flat_logits = logits.reshape(-1, e)
                num_tokens = flat_logits.size(0)
                device = flat_logits.device

                positions = torch.arange(num_tokens * k, device=device)
                row = positions // k
                col = positions % e
                routing_map = torch.zeros(num_tokens, e, dtype=torch.bool, device=device)
                routing_map.index_put_(
                    (row, col),
                    torch.ones(1, device=device, dtype=torch.bool),
                )

                # Probs come from softmax-of-logits so the gating linear's
                # backward path is still exercised; mask + renormalize so each
                # row sums to 1 over its k assigned experts.
                probs = torch.softmax(flat_logits, dim=-1)
                probs = probs * routing_map.to(probs.dtype)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                return probs, routing_map

            return balanced_routing

        sub.routing = make_balanced()

        def make_restore(s=sub, orig=original_routing):
            def _restore():
                s.routing = orig

            return _restore

        restores.append(make_restore())
        descriptors.append((name or sub.__class__.__name__, topk, num_experts))

    return restores, descriptors


def _uninstall_routing_patches(restores) -> None:
    for restore in restores:
        try:
            restore()
        except Exception as exc:
            # Best-effort cleanup: do not fail teardown if one restore callback fails.
            print(f"[primus] warning: failed to restore routing patch: {exc}")


def benchmark_moe_layer_decomposed(
    moe_module: torch.nn.Module,
    input_shapes: List[Union[Tuple[int, ...], Tuple[Tuple[int, ...], torch.dtype]]],
    num_iterations: int = 64,
    transformer_config=None,
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

    By default this function also installs a deterministic round-robin
    routing patch on the MoE module's router (see
    :func:`_install_balanced_routing_patches`) so the per-expert ``M``
    dimension is constant across iterations.  Without this, ROCm
    composable_kernel JIT recompiles the FP8 grouped GEMM for every unique
    expert load count, which dominates iteration time for high-expert MoE
    models (Qwen3 235B, DeepSeek v3).  Set
    ``PRIMUS_BENCH_MOE_KERNEL_PAD=0`` to opt out and recover stochastic
    routing.

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

    is_rank_0 = int(os.getenv("RANK", "0")) == 0

    # Install kernel-shape-padding (balanced round-robin routing) so the
    # per-expert M dimension is constant across iterations.  This eliminates
    # per-shape JIT recompiles in the FP8 grouped GEMM and is the dominant
    # cost on high-expert MoE benches.
    routing_restores = []
    if _kernel_pad_enabled():
        routing_restores, routing_descriptors = _install_balanced_routing_patches(moe_module)
        if routing_descriptors and is_rank_0:
            _, topk, num_experts = routing_descriptors[0]
            # The bench feeds a single [seq_len, batch_size, hidden] input,
            # so the per-rank token count is the product of the first two
            # dims (TP/SP sharding is already reflected in the supplied
            # shape).
            num_tokens = 1
            for spec in input_shapes:
                shape = (
                    spec[0]
                    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], torch.dtype)
                    else spec
                )
                # [seq, batch, hidden]
                if len(shape) >= 2:
                    num_tokens = int(shape[0]) * int(shape[1])
                break
            m_per_expert = (num_tokens * topk) // max(num_experts, 1)
            print(
                f"  [MoE Decomposed] Kernel-shape padding ON: balanced "
                f"round-robin routing on {len(routing_descriptors)} router(s) "
                f"(topk={topk}, num_experts={num_experts}, "
                f"M_per_expert={m_per_expert}). "
                f"Set PRIMUS_BENCH_MOE_KERNEL_PAD=0 to disable."
            )
        elif is_rank_0:
            print(
                "  [MoE Decomposed] Kernel-shape padding requested but no "
                "router-like submodule found; falling back to stochastic routing."
            )

    try:
        # =====================================================================
        # WARMUP (20 iterations) - same as benchmark_layer
        # =====================================================================
        grad_outputs = None
        output_indices = []
        num_warmup, num_iterations = _bench_iter_count(20, num_iterations)

        with fp8_context:
            for _ in range(num_warmup):
                outputs = moe_module(*inputs)
                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)

                if grad_outputs is None:
                    grad_outputs = []
                    for i, out in enumerate(outputs):
                        # See note in ``benchmark_layer``: filter on
                        # ``grad_fn is not None`` so backward actually
                        # walks the autograd graph (and releases saved
                        # tensors), instead of being a no-op on detached
                        # custom-autograd outputs / leaf parameters.
                        if isinstance(out, torch.Tensor) and out.requires_grad and out.grad_fn is not None:
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

        # =====================================================================
        # BENCHMARK with decomposed A2A timing
        # =====================================================================
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
            moe_module.dispatch = original_dispatch
            moe_module.combine = original_combine
    finally:
        _uninstall_routing_patches(routing_restores)

    avg_forward_time = _bench_central_value(forward_times)
    avg_backward_time = _bench_central_value(backward_times)

    # Compute average A2A forward time (dispatch + combine)
    # Each forward iteration produces one dispatch event and one combine event.
    a2a_fwd_times = []
    for (ds, de), (cs, ce) in zip(_dispatch_events, _combine_events):
        dispatch_ms = ds.elapsed_time(de)
        combine_ms = cs.elapsed_time(ce)
        a2a_fwd_times.append(dispatch_ms + combine_ms)

    avg_a2a_fwd = _bench_central_value(a2a_fwd_times)

    # Backward A2A is approximately equal to forward A2A (same message sizes).
    # We don't instrument the backward autograd graph, so we use this assumption.
    avg_a2a_bwd = avg_a2a_fwd

    if is_rank_0:
        compute_fwd = avg_forward_time - avg_a2a_fwd
        compute_bwd = avg_backward_time - avg_a2a_bwd
        print(f"  [MoE Decomposed] Total fwd: {avg_forward_time:.2f} ms, bwd: {avg_backward_time:.2f} ms")
        print(f"  [MoE Decomposed] A2A   fwd: {avg_a2a_fwd:.2f} ms, bwd(est): {avg_a2a_bwd:.2f} ms")
        print(f"  [MoE Decomposed] Compute fwd: {compute_fwd:.2f} ms, bwd: {compute_bwd:.2f} ms")

    return avg_forward_time, avg_backward_time, int(activation_memory), avg_a2a_fwd, avg_a2a_bwd

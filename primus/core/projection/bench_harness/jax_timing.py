###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Measuring a JAX module the way the torch harness measures a torch one.

The torch path times a module with CUDA events around ``module(*inputs)`` and
``torch.autograd.backward(outputs, grad_outputs)``, and reads activation memory
off the caching allocator's peak.  None of those three exist under JAX, so the
equivalents are:

**Backward.** ``jax.vjp`` is the direct analogue of
``torch.autograd.backward(outputs, cotangents)``: both push a fixed cotangent
back through the graph.  Using it rather than ``grad`` of a summed loss matters
for more than symmetry -- a cotangent of all ones lets XLA constant-fold parts
of the backward that a real training step would actually run.

**Timing.** There is no event to record inside an XLA executable, and dispatch
is asynchronous, so a call is timed on the host with ``block_until_ready``
forcing completion.  Forward and backward cannot be split inside one executable,
so they are compiled and timed separately and backward is reported as the
difference -- the same decomposition the torch path gets from two event pairs.

**Memory.** ``device.memory_stats()`` is not usable as an activation measure
here: XLA preallocates most of the device by default, so the allocator's peak
reflects the preallocation rather than the workload, and on CPU the dict is
empty.  XLA's own compile-time accounting is both available and more precise --
``memory_analysis()`` reports the scratch an executable needs beyond its
arguments, which is what "activation memory" means for a compiled layer.

Compilation is excluded from every number: each executable is lowered and
compiled up front, then warmed, then timed.
"""

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

# Kept in step with the torch harness (20 warmup / 64 timed) so a JAX-measured
# layer and a torch-measured one are averaged the same way.  XLA executables are
# far more deterministic run to run, so fewer timed iterations suffice.
_DEFAULT_WARMUP = 5
_DEFAULT_ITERS = 20


@dataclass(frozen=True)
class JaxMeasurement:
    """One module's measured cost, in the units the artifact records."""

    forward_ms: float
    backward_ms: float
    activation_bytes: int


def _iter_counts() -> Tuple[int, int]:
    """Warmup and timed iteration counts, overridable for quick runs."""
    warmup = int(os.environ.get("PRIMUS_BENCH_JAX_WARMUP", _DEFAULT_WARMUP))
    iters = int(os.environ.get("PRIMUS_BENCH_JAX_ITERS", _DEFAULT_ITERS))
    return max(1, warmup), max(1, iters)


def _median(values: Sequence[float]) -> float:
    """Median, matching the torch harness's central-value choice.

    The median rather than the mean because a stray host-side hiccup between
    dispatch and ``block_until_ready`` shows up as one long sample, and should
    not move the number.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _time_compiled(compiled: Callable[[], Any], warmup: int, iters: int) -> float:
    """Median wall time of *compiled*, in ms, with dispatch forced to complete."""
    import jax

    for _ in range(warmup):
        jax.block_until_ready(compiled())

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(compiled())
        samples.append((time.perf_counter() - start) * 1e3)
    return _median(samples)


def _activation_bytes(lowered_forward) -> int:
    """Bytes a forward pass needs beyond its arguments.

    ``temp_size_in_bytes`` is XLA's scratch for the executable and
    ``output_size_in_bytes`` is the result the next layer consumes; together
    they are the footprint a training step carries forward, which is what the
    torch harness's allocator delta captures.
    """
    try:
        stats = lowered_forward.memory_analysis()
    except Exception:
        # Not every backend implements the analysis; a zero here makes the
        # projection fall back to its analytical activation estimate rather
        # than reporting a wrong measurement.
        return 0
    if stats is None:
        return 0
    temp = getattr(stats, "temp_size_in_bytes", 0) or 0
    output = getattr(stats, "output_size_in_bytes", 0) or 0
    return int(temp + output)


def benchmark_jax_module(
    forward: Callable[..., Any],
    params: Any,
    inputs: Sequence[Any],
    *,
    cotangent: Optional[Any] = None,
    num_warmup: Optional[int] = None,
    num_iters: Optional[int] = None,
) -> JaxMeasurement:
    """Time *forward*'s forward and backward pass and size its activations.

    Args:
        forward: called as ``forward(params, *inputs)``, returning one array.
        params: the module's parameter pytree, differentiated against.
        inputs: positional activations, also differentiated against so the
            measurement includes the input-gradient work a real layer does.
        cotangent: gradient seed for the backward pass.  Defaults to a fixed
            normal draw shaped like the output, mirroring the torch harness's
            ``randn_like`` grad outputs.
        num_warmup, num_iters: override the iteration counts.

    Returns:
        A :class:`JaxMeasurement`.  ``backward_ms`` is the forward+backward
        executable's time minus the forward's, and is clamped at zero.
    """
    import jax

    warmup, iters = _iter_counts()
    warmup = num_warmup if num_warmup is not None else warmup
    iters = num_iters if num_iters is not None else iters

    inputs = tuple(inputs)

    forward_jit = jax.jit(lambda p, xs: forward(p, *xs))
    forward_compiled = forward_jit.lower(params, inputs).compile()

    if cotangent is None:
        out_shape = jax.eval_shape(lambda p, xs: forward(p, *xs), params, inputs)
        cotangent = jax.tree.map(
            lambda spec: jax.random.normal(jax.random.key(0), spec.shape, spec.dtype),
            out_shape,
        )

    def _forward_and_backward(p, xs, ct):
        # The direct analogue of torch.autograd.backward(outputs, grad_outputs):
        # differentiate against both parameters and activations.
        primal, vjp_fn = jax.vjp(lambda p_, xs_: forward(p_, *xs_), p, xs)
        return primal, vjp_fn(ct)

    fwd_bwd_compiled = jax.jit(_forward_and_backward).lower(params, inputs, cotangent).compile()

    forward_ms = _time_compiled(lambda: forward_compiled(params, inputs), warmup, iters)
    total_ms = _time_compiled(lambda: fwd_bwd_compiled(params, inputs, cotangent), warmup, iters)

    return JaxMeasurement(
        forward_ms=forward_ms,
        backward_ms=max(0.0, total_ms - forward_ms),
        activation_bytes=_activation_bytes(forward_compiled),
    )


def device_memory_snapshot(label: str) -> dict:
    """Capture a JAX device memory snapshot in the harness's snapshot shape.

    Mirrors :func:`primus.core.projection.memory_capture.capture_memory_snapshot`
    so a JAX run's ``_memory_benchmark`` payload reads the same as a torch run's
    and the memory extrapolator needs no backend-specific branch.  Fields XLA
    does not report come back zero.
    """
    import jax

    snapshot = {
        "label": label,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "max_allocated_bytes": 0,
        "max_reserved_bytes": 0,
        "free_bytes": 0,
        "total_bytes": 0,
    }

    devices = jax.local_devices()
    if not devices:
        return snapshot

    stats = getattr(devices[0], "memory_stats", lambda: None)() or {}
    snapshot.update(
        allocated_bytes=int(stats.get("bytes_in_use", 0) or 0),
        reserved_bytes=int(stats.get("bytes_reserved", stats.get("pool_bytes", 0)) or 0),
        max_allocated_bytes=int(stats.get("peak_bytes_in_use", 0) or 0),
        max_reserved_bytes=int(stats.get("peak_pool_bytes", stats.get("largest_alloc_size", 0)) or 0),
        total_bytes=int(stats.get("bytes_limit", 0) or 0),
    )
    snapshot["free_bytes"] = max(0, snapshot["total_bytes"] - snapshot["allocated_bytes"])
    return snapshot

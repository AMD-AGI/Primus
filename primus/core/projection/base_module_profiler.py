from abc import ABC
from typing import Optional

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


class BaseModuleProfiler(ABC):
    """Abstract base class for transformer-like module profiler.
    Provides both estimated and measured statistics.
    """

    def __init__(self, config, sub_profilers=None):
        self.config = config
        self.sub_profilers = sub_profilers

    # -------- Benchmark backend --------
    #: Set by :meth:`set_bench_adapter` when benchmarking a non-Megatron
    #: backend.  ``None`` means Megatron's conventions, which is what the
    #: profilers were written against.
    _bench_adapter = None

    def set_bench_adapter(self, adapter):
        """Point this profiler and its children at *adapter*'s conventions.

        Propagated down the whole tree so the layer, attention, MLP, embedding
        and output profilers all build inputs the same backend's ``forward``
        will accept.
        """
        self._bench_adapter = adapter
        for sub in (self.sub_profilers or {}).values():
            if sub is not None and hasattr(sub, "set_bench_adapter"):
                sub.set_bench_adapter(adapter)

    def bench_adapter(self):
        """Return the active bench adapter, defaulting to Megatron's."""
        if self._bench_adapter is None:
            from primus.core.projection.bench_harness.megatron import MegatronBenchAdapter

            self._bench_adapter = MegatronBenchAdapter()
        return self._bench_adapter

    def bench_inputs(self, kind: str, module, batch_size: int, seq_len: int):
        """Ask the active adapter what to call *module* with.

        Returns ``None`` when the backend cannot construct a valid call, which
        the caller must treat as "do not benchmark this module".
        """
        from primus.core.projection.bench_harness.base import BenchContext

        ctx = BenchContext.build(self.config, batch_size, seq_len)
        return self.bench_adapter().inputs(kind, module, ctx)

    def require_bench_inputs(self, kind: str, module, batch_size: int, seq_len: int):
        """Like :meth:`bench_inputs`, but as ``(args, kwargs)`` and never ``None``.

        A benchmark run was asked for measurements, so a backend that cannot
        construct a valid call fails loudly rather than quietly substituting an
        analytical estimate and reporting it as measured.
        """
        inputs = self.bench_inputs(kind, module, batch_size, seq_len)
        if inputs is None:
            adapter = self.bench_adapter()
            raise RuntimeError(
                f"{type(adapter).__name__} cannot build benchmark inputs for "
                f"'{kind}' module {type(module).__name__}. Run this projection with "
                "--profiling-mode simulate to use analytical estimates instead."
            )
        return list(inputs.args), (dict(inputs.kwargs) or None)

    # -------- Parameter related --------
    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        """Return estimated parameter count (based on formula).
        If rank is provided, return the parameter count for the given rank,
        otherwise return the total parameter count for the entire model.
        """
        raise NotImplementedError

    def measured_num_params(self) -> int:
        """Return measured parameter count (from real tensors)."""
        raise NotImplementedError

    # -------- Memory related --------
    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        """Return estimated memory usage in bytes (activations)."""
        raise NotImplementedError

    def measured_activation_memory(self, batch_size: int, seq_len: int) -> int:
        """Return measured memory usage in bytes (via profiler/runtime stats)."""
        raise NotImplementedError

    # -------- Performance related --------
    def estimated_forward_time(self, batch_size: int, seq_len: int) -> int:
        """Return estimated forward latency for forward pass in milliseconds."""
        raise NotImplementedError

    def estimated_backward_time(self, batch_size: int, seq_len: int) -> int:
        """Return estimated latency for backward pass in milliseconds."""
        raise NotImplementedError

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        """Return measured forward latency in milliseconds."""
        raise NotImplementedError

    def measured_backward_time(self, batch_size: int, seq_len: int) -> float:
        """Return measured backward latency in milliseconds."""
        raise NotImplementedError

    # -------- Debugging / summary --------
    def __repr__(self):
        return f"{self.__class__.__name__}"

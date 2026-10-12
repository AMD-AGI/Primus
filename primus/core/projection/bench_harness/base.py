###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The seam between a trained model and the projection's layer benchmark.

Benchmark-anchored projection measures one representative layer of a real model
and extrapolates from it.  Finding that layer means knowing where the backend
keeps it -- Megatron nests the stack under ``language_model.decoder.layers`` and
names a block's halves ``self_attention`` / ``mlp``; TorchTitan keeps a
``ModuleDict`` at ``layers`` and names them ``attention`` / ``feed_forward`` or
``moe`` -- and feeding it means knowing what its ``forward`` wants, which is
``[S, B, D]`` plus a boolean mask for Megatron and ``[B, S, D]`` plus a rope
cache for TorchTitan.

A :class:`BenchModelAdapter` is the one place those two facts live.  Everything
downstream of it -- the CUDA-event timing loop, the MoE dispatch/combine
decomposition, the activation-memory capture, the artifact -- is already
backend-neutral, so a backend becomes benchmarkable by describing its module
tree rather than by forking the harness.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# What a profiler is asking to be fed.  Kept as plain strings rather than an
# enum because adapters are matched on them in dict literals.
LAYER = "layer"
ATTENTION = "attention"
MLP = "mlp"
MOE = "moe"
EMBEDDING = "embedding"
OUTPUT = "output"

ALL_KINDS = (LAYER, ATTENTION, MLP, MOE, EMBEDDING, OUTPUT)


@dataclass(frozen=True)
class BenchContext:
    """The shape of the workload one benchmark call is measuring."""

    batch_size: int
    seq_len: int
    hidden_size: int
    # Sequence length actually resident on this rank; context parallelism shards
    # it, and the benchmark has to feed the per-rank slice rather than the whole.
    seq_len_per_cp: int
    config: Any = None

    @classmethod
    def build(cls, config, batch_size: int, seq_len: int) -> "BenchContext":
        cp_size = getattr(config.model_parallel_config, "context_model_parallel_size", 1) or 1
        return cls(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=config.model_config.hidden_size,
            seq_len_per_cp=seq_len // cp_size,
            config=config,
        )


@dataclass(frozen=True)
class BenchInputs:
    """What to call a module with.

    Each entry is either a shape tuple, a ``(shape, dtype)`` pair, an already
    built tensor, or ``None``.  Shapes are filled with random data by
    :func:`~primus.core.projection.module_profilers.utils.benchmark_layer`;
    tensors are passed straight through, which is how a backend supplies
    something it cannot describe as a shape -- TorchTitan's precomputed rope
    cache, say, whose values decide whether the kernel is even valid.
    """

    args: Sequence[Any] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelParts:
    """Where a backend keeps the pieces the benchmark needs to reach."""

    layers: List[Any] = field(default_factory=list)
    embedding: Optional[Any] = None
    output_layer: Optional[Any] = None


@dataclass
class LayerParts:
    """The halves of a transformer block, as the profiler tree names them."""

    attention: Optional[Any] = None
    mlp: Optional[Any] = None


class BenchModelAdapter(ABC):
    """Describes one backend's model tree and forward conventions."""

    #: Framework name this adapter is registered under.
    framework: str = ""

    @abstractmethod
    def discover(self, model) -> ModelParts:
        """Locate the transformer layers, embedding and output head in *model*.

        *model* is whatever the trainer built -- a module, or a list of pipeline
        chunks.  Layers must come back in global order so the projection's
        layer indices select the same layer the config says they do.
        """

    @abstractmethod
    def layer_submodules(self, layer) -> LayerParts:
        """Split one transformer block into its attention and MLP halves."""

    @abstractmethod
    def inputs(self, kind: str, module, ctx: BenchContext) -> Optional[BenchInputs]:
        """Return what to call *module* with, or ``None`` to skip benchmarking it.

        ``None`` means this adapter cannot construct a valid call -- the caller
        then leaves the measurement to the analytical estimate rather than
        feeding the module something that would crash or measure the wrong
        kernel.
        """

    def supports_moe_decomposition(self, module) -> bool:
        """Whether *module* exposes ``dispatch``/``combine`` to time separately.

        The MoE all-to-all is decomposed out of the expert compute so that
        shrinking EP onto the bench node can be corrected analytically.  A
        backend whose MoE does not expose those two calls returns ``False`` and
        gets a whole-layer measurement instead.
        """
        return hasattr(module, "dispatch") and hasattr(module, "combine")

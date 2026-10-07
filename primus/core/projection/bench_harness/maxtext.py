###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MaxText/JAX benchmark runner.

Megatron and TorchTitan share a harness because they share a substrate: torch
modules, autograd, CUDA events, a caching allocator.  MaxText shares none of it.
Its layers are Flax modules whose parameters live in a pytree rather than on the
module, its backward is a ``vjp`` through a traced graph rather than a walk over
recorded operations, and its memory is XLA's to account for.  So rather than
adapting the torch harness, this runs its own measurement (see
:mod:`~primus.core.projection.bench_harness.jax_timing`) and emits the same
artifact, which is the only thing the rest of the projection reads.

What it measures, and how:

**Layers.** ``Decoder.get_decoder_layers()`` is MaxText's own answer to "what
block does this model use", and for the architectures that mix them -- DeepSeek,
for instance -- it returns the dense and MoE blocks as separate classes.  That
lines up exactly with the projection's dense/MoE representatives, so each is
built and timed once and the result is spread over the layers of that type.

**Embedding.** Built the way ``Transformer.setup`` builds it, and timed the same
way.

**Attention and MLP within a layer.** Reported, but apportioned rather than
measured.  A MaxText decoder layer is one Flax module with one traced graph;
there is no sub-module to time in isolation the way ``layer.self_attention`` can
be on the torch side.  The measured layer total is therefore split using the
analytical model's own forward-time ratio, and the entries are marked
``"estimated_split": True`` so nothing downstream mistakes them for measurements.

**Output head and loss.** Left to the analytical model.  MaxText's head is
applied by a method on a bound ``Decoder`` scope rather than by a standalone
module, and a reconstruction of it that was subtly wrong would be worse than a
number that is honestly labelled an estimate.  The head is a small fraction of
step time, and the projection already models it.

**Expert all-to-all.** Not decomposed.  MaxText expresses expert parallelism as
sharding over the mesh, so there is no dispatch/combine pair to time; at the EP
the bench node can hold the all-to-all is near zero anyway, and the projection
restores it analytically when scaling to the target -- the same treatment
TorchTitan's MoE and Megatron's DeepSeek-V4 layers already get.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from primus.core.projection.bench_harness.jax_timing import (
    JaxMeasurement,
    benchmark_jax_module,
    device_memory_snapshot,
)


def _resolve_maxtext_layers():
    """Import MaxText's decoder/embedding/mesh helpers across its two layouts.

    MaxText v26.4+ ships as the lowercase ``maxtext`` package; v26.3 and earlier
    as ``MaxText``.  Primus supports both, so resolve the same way the trainer
    does rather than pinning one.
    """
    try:
        from maxtext.common_types import MODEL_MODE_TRAIN
        from maxtext.layers.decoders import Decoder
        from maxtext.layers.embeddings import embed_as_linen
        from maxtext.layers.quantizations import configure_quantization
        from maxtext.utils.maxtext_utils import get_mesh_from_config
    except ImportError:
        from MaxText.common_types import MODEL_MODE_TRAIN
        from MaxText.layers.decoders import Decoder
        from MaxText.layers.embeddings import embed_as_linen
        from MaxText.layers.quantizations import configure_quantization
        from MaxText.utils.maxtext_utils import get_mesh_from_config

    return {
        "MODEL_MODE_TRAIN": MODEL_MODE_TRAIN,
        "Decoder": Decoder,
        "embed_as_linen": embed_as_linen,
        "configure_quantization": configure_quantization,
        "get_mesh_from_config": get_mesh_from_config,
    }


def _unwrap_layer_output(output):
    """A layer returns ``(out, None)`` under ``scan_layers`` and ``out`` otherwise."""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _rank() -> int:
    return int(os.getenv("RANK", "0"))


def _log(message: str) -> None:
    if _rank() == 0:
        print(f"[Primus:Performance Projection] {message}")


class MaxTextLayerBench:
    """Measures a MaxText model's layers and emits the standard bench artifact."""

    framework = "maxtext"

    def run(
        self,
        *,
        trainer: Any,
        training_config: Any,
        batch_size: int,
        seq_len: int,
        profiler: Any = None,
    ) -> Dict[str, Any]:
        """Benchmark this rank's layers and return ``profiling_results``.

        Args:
            trainer: the MaxText trainer after ``setup_model_only``; its
                ``train_config`` is MaxText's resolved ``HyperParameters``.
            training_config: the projection's ``TrainingConfig``, which says
                which layers this rank owns and which of them are MoE.
            batch_size, seq_len: the bench microbatch shape.
            profiler: the analytical profiler tree, used only to apportion the
                measured layer total across attention and MLP.

        Returns:
            The same dict shape the torch harness returns: integer layer keys,
            plus ``"embedding"`` and ``"_memory_benchmark"``.
        """
        config = getattr(trainer, "train_config", None)
        if config is None:
            raise RuntimeError(
                "MaxText benchmark needs the trainer's resolved config; "
                "setup_model_only() must run before the layer benchmark."
            )

        import jax
        import jax.numpy as jnp
        from flax import linen as nn

        mt = _resolve_maxtext_layers()

        snapshots = [device_memory_snapshot("post_setup")]

        mesh = mt["get_mesh_from_config"](config)
        quant = mt["configure_quantization"](config)
        model_mode = mt["MODEL_MODE_TRAIN"]

        # Every MaxText module places its arrays through logical axis names, which
        # only resolve inside the mesh and the config's rule set. Building or
        # applying one outside both contexts either fails or silently measures an
        # unsharded layer, which at TP>1 is a different kernel.
        with mesh, nn.logical_axis_rules(config.logical_axis_rules):
            layer_classes = mt["Decoder"](
                config=config, mesh=mesh, quant=quant, model_mode=model_mode
            ).get_decoder_layers()

            layer_results = self._benchmark_layers(
                layer_classes=layer_classes,
                config=config,
                mesh=mesh,
                quant=quant,
                model_mode=model_mode,
                training_config=training_config,
                batch_size=batch_size,
                seq_len=seq_len,
                profiler=profiler,
                jax=jax,
                jnp=jnp,
            )

            embedding_result = self._benchmark_embedding(
                embed_as_linen=mt["embed_as_linen"],
                config=config,
                mesh=mesh,
                model_mode=model_mode,
                batch_size=batch_size,
                seq_len=seq_len,
                jax=jax,
                jnp=jnp,
            )

        snapshots.append(device_memory_snapshot("post_layer_benchmark"))

        results: Dict[str, Any] = dict(layer_results)
        if embedding_result is not None:
            results["embedding"] = embedding_result
        results["_memory_benchmark"] = self._memory_payload(snapshots)
        return results

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def _benchmark_layers(
        self,
        *,
        layer_classes,
        config,
        mesh,
        quant,
        model_mode,
        training_config,
        batch_size,
        seq_len,
        profiler,
        jax,
        jnp,
    ) -> Dict[int, Any]:
        moe_pattern = training_config.model_config.moe_pattern
        layers = getattr(profiler, "layers", None) or list(range(len(moe_pattern)))

        dense_cls, moe_cls = self._classify_layer_classes(layer_classes)

        measured: Dict[str, JaxMeasurement] = {}
        results: Dict[int, Any] = {}

        for layer_idx in layers:
            if layer_idx >= len(moe_pattern):
                continue
            is_moe = bool(moe_pattern[layer_idx])
            layer_type = "moe" if is_moe else "dense"

            if layer_type not in measured:
                layer_cls = moe_cls if is_moe else dense_cls
                if layer_cls is None:
                    continue
                _log(f"Benchmarking MaxText {layer_type} decoder layer...")
                measured[layer_type] = self._measure_layer(
                    layer_cls=layer_cls,
                    config=config,
                    mesh=mesh,
                    quant=quant,
                    model_mode=model_mode,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    jax=jax,
                    jnp=jnp,
                )
                m = measured[layer_type]
                _log(
                    f"  {layer_type} layer -> fwd: {m.forward_ms:.2f} ms, "
                    f"bwd: {m.backward_ms:.2f} ms, "
                    f"act: {m.activation_bytes / (1024 ** 2):.2f} MB"
                )

            if layer_type in measured:
                results[layer_idx] = self._layer_entry(
                    measured[layer_type], layer_type, profiler, batch_size, seq_len
                )

        return results

    def _classify_layer_classes(self, layer_classes) -> Tuple[Optional[Any], Optional[Any]]:
        """Split MaxText's decoder-layer classes into the dense and MoE blocks.

        Architectures that mix layer types return both -- DeepSeek returns its
        dense block then its MoE block, in that order.  A single class serves
        every layer of that model, whatever the projection's MoE pattern says.
        """
        classes = list(layer_classes or [])
        if not classes:
            return None, None
        if len(classes) == 1:
            return classes[0], classes[0]
        return classes[0], classes[1]

    def _measure_layer(
        self, *, layer_cls, config, mesh, quant, model_mode, batch_size, seq_len, jax, jnp
    ) -> JaxMeasurement:
        layer = layer_cls(config=config, mesh=mesh, quant=quant, model_mode=model_mode)

        hidden = self._hidden_states(config, batch_size, seq_len, jax, jnp)
        segment_ids, positions = self._token_metadata(batch_size, seq_len, jnp)

        # deterministic=True: dropout is off in every pretraining config the
        # projection targets, and leaving it on would make the timing depend on
        # a sampled mask rather than on the layer.
        call_args = (segment_ids, positions, True, model_mode)
        rngs = self._rngs(jax)
        variables = layer.init(rngs, hidden, *call_args)

        def forward(params, activations):
            return _unwrap_layer_output(layer.apply(params, activations, *call_args))

        return benchmark_jax_module(forward, variables, [hidden])

    def _layer_entry(self, measurement, layer_type, profiler, batch_size, seq_len):
        attention_share, mlp_share = self._analytical_split(profiler, layer_type, batch_size, seq_len)

        attention_fwd = measurement.forward_ms * attention_share
        mlp_fwd = measurement.forward_ms * mlp_share
        attention_bwd = measurement.backward_ms * attention_share
        mlp_bwd = measurement.backward_ms * mlp_share
        attention_mem = int(measurement.activation_bytes * attention_share)
        mlp_mem = int(measurement.activation_bytes * mlp_share)

        return {
            "type": layer_type,
            "compress_ratio": None,
            "forward_time_ms": measurement.forward_ms,
            "backward_time_ms": measurement.backward_ms,
            "activation_memory_bytes": measurement.activation_bytes,
            # A MaxText layer is one traced graph; these halves are the measured
            # total apportioned by the analytical model, not separate timings.
            "estimated_split": True,
            "attention": {
                "forward_time_ms": attention_fwd,
                "backward_time_ms": attention_bwd,
                "activation_memory_bytes": attention_mem,
            },
            "mlp": {
                "forward_time_ms": mlp_fwd,
                "backward_time_ms": mlp_bwd,
                "activation_memory_bytes": mlp_mem,
                # Expert communication is sharding here, not a timed call; the
                # projection restores it analytically for the target EP.
                "a2a_forward_time_ms": 0.0,
                "a2a_backward_time_ms": 0.0,
            },
        }

    def _analytical_split(self, profiler, layer_type, batch_size, seq_len) -> Tuple[float, float]:
        """Attention/MLP fractions of a layer's forward time, per the analytical model.

        Falls back to an even split if the profiler tree cannot supply them, so a
        missing estimate degrades the reported breakdown rather than the
        measurement it is dividing.
        """
        default = (0.5, 0.5)
        if profiler is None:
            return default

        key = "moe_transformer_layer" if layer_type == "moe" else "dense_transformer_layer"
        layer_profiler = (getattr(profiler, "sub_profilers", None) or {}).get(key)
        if layer_profiler is None:
            return default

        try:
            attention = layer_profiler.get_sub_profiler("self_attention").estimated_forward_time(
                batch_size, seq_len
            )
            mlp = layer_profiler.get_sub_profiler("mlp").estimated_forward_time(batch_size, seq_len)
        except Exception:
            return default

        total = float(attention) + float(mlp)
        if total <= 0:
            return default
        return float(attention) / total, float(mlp) / total

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _benchmark_embedding(
        self, *, embed_as_linen, config, mesh, model_mode, batch_size, seq_len, jax, jnp
    ):
        _log("Benchmarking MaxText embedding...")
        embedding = embed_as_linen(
            num_embeddings=config.vocab_size,
            num_features=config.emb_dim,
            dtype=config.dtype,
            attend_dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
            embedding_init=jax.nn.initializers.normal(stddev=1.0),
            name="token_embedder",
            config=config,
            mesh=mesh,
        )

        tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        variables = embedding.init(self._rngs(jax), tokens, model_mode=model_mode)

        # The token ids are integers, so there is nothing to differentiate them
        # against; the gradient the backward pass computes is the table's.
        def forward(params, _unused):
            return embedding.apply(params, tokens, model_mode=model_mode)

        dummy = jnp.zeros((1,), dtype=jnp.float32)
        measurement = benchmark_jax_module(forward, variables, [dummy])
        _log(
            f"  Embedding -> fwd: {measurement.forward_ms:.2f} ms, "
            f"bwd: {measurement.backward_ms:.2f} ms, "
            f"act: {measurement.activation_bytes / (1024 ** 2):.2f} MB"
        )
        return {
            "type": "embedding",
            "forward_time_ms": measurement.forward_ms,
            "backward_time_ms": measurement.backward_ms,
            "activation_memory_bytes": measurement.activation_bytes,
        }

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _hidden_states(self, config, batch_size, seq_len, jax, jnp):
        dtype = jnp.dtype(config.dtype) if isinstance(config.dtype, str) else config.dtype
        return jax.random.normal(jax.random.key(0), (batch_size, seq_len, config.emb_dim), dtype=dtype)

    def _token_metadata(self, batch_size, seq_len, jnp):
        # Segment id 1 marks a real token; 0 would mark padding, which attention
        # masks out and would leave part of the kernel unmeasured.
        segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
        return segment_ids, positions

    def _rngs(self, jax) -> Dict[str, Any]:
        """The rng streams MaxText modules draw from during init and apply."""
        key = jax.random.key(0)
        return {"params": key, "dropout": key, "aqt": key}

    def _memory_payload(self, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble a ``_memory_benchmark`` payload in the torch harness's shape."""
        deltas = []
        for before, after in zip(snapshots, snapshots[1:]):
            deltas.append(
                {
                    "from": before["label"],
                    "to": after["label"],
                    "allocated_delta_bytes": after["allocated_bytes"] - before["allocated_bytes"],
                    "reserved_delta_bytes": after["reserved_bytes"] - before["reserved_bytes"],
                }
            )

        return {
            "captured_on_rank": _rank(),
            "snapshots": snapshots,
            "phase_deltas": deltas,
            "global_peak_allocated_bytes": max((s["max_allocated_bytes"] for s in snapshots), default=0),
            "global_peak_reserved_bytes": max((s["max_reserved_bytes"] for s in snapshots), default=0),
        }

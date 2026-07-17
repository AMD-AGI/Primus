###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Real-GPU (benchmark) mode for inference projection.

Training projection has a ``--profiling-mode benchmark`` path that builds the
real model and times each transformer layer on silicon.  This module provides
the **forward-only** analogue for serving:

  * a *worker* (``run_inference_benchmark_worker``) that runs under
    ``torch.distributed.run``, builds the real model via the Megatron trainer
    (exactly like the training layer benchmark), then times **forward-only**
    passes of a chained stack of ``capture_layers`` dense (and, for MoE models,
    MoE) layers at the **prefill** shape ``(batch, input_len)`` and the
    **decode** shape ``(batch, 1)``; the per-layer time is the stack time
    divided by the layer count.  Rank 0 writes the measured times to a JSON
    file.  ``capture_layers`` defaults to 4 (``--inference-bench-layers``).
  * a *parent* (``spawn_inference_benchmark``) that — from the normal
    ``projection inference`` process — spawns the torchrun worker, polls for
    the JSON, and returns the measured per-layer-type forward times.

The measured times are consumed by
:class:`~primus.core.projection.inference_projection.performance.InferencePerformanceProjector`
as a **per-phase calibration** of the analytical layer-compute time (so the
absolute scale is anchored to real silicon while the context/sequence-length
scaling still comes from the simulator).  See ``performance.py``.

NOTE: the worker requires GPUs + a distributed launch; it cannot run in a plain
CPU process.  It mirrors the proven training benchmark harness.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from typing import Dict, Optional


# Keys in the saved JSON.
_DENSE = "dense"
_MOE = "moe"

# Default number of same-type layers to build and time as a chained stack.
# Timing a stack (rather than a single layer) averages out per-layer jitter and
# captures inter-layer effects; the reported per-layer time is stack / N.
_INFERENCE_CAPTURE_LAYERS = 4


def _unwrap_layers(model):
    """Extract the transformer layer modules from a (possibly wrapped) model."""

    def unwrap(m):
        return unwrap(m.module) if hasattr(m, "module") else m

    layers = []
    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        u = unwrap(chunk)
        lm = getattr(u, "language_model", None)
        target = lm if lm is not None else u
        if hasattr(target, "decoder") and hasattr(target.decoder, "layers"):
            layers.extend(list(target.decoder.layers))
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layers"):
            layers.extend(list(target.encoder.layers))
        elif hasattr(target, "layers"):
            layers.extend(list(target.layers))
    return layers


def _make_layer_stack(layers):
    """Wrap ``layers`` into a single module that chains their forward passes.

    Each transformer layer returns ``(hidden_states, context)``; the stack feeds
    the hidden-state output of one layer into the next so a single timed forward
    covers the whole chain.
    """
    import torch

    class _LayerStack(torch.nn.Module):
        def __init__(self, modules):
            super().__init__()
            self.stack = torch.nn.ModuleList(modules)

        def forward(self, hidden_states):
            out = hidden_states
            for layer in self.stack:
                res = layer(out)
                out = res[0] if isinstance(res, (tuple, list)) else res
            return out

    return _LayerStack(layers)


def run_inference_benchmark_worker(primus_config, unknown_overrides, args) -> Optional[dict]:
    """Build the real model and benchmark forward-only prefill/decode layers.

    Must be invoked under ``torch.distributed.run``.  Rank 0 writes the result
    JSON to ``args.save_profiling`` and returns the dict; other ranks return
    ``None``.
    """
    import torch

    from primus.core.projection.module_profilers.utils import benchmark_layer
    from primus.core.projection.performance_projection.projection import (
        _limit_layers_for_projection,
        _rescale_expert_parallelism,
    )
    from primus.core.projection.training_config import (
        convert_primus_config_to_projection_config,
    )
    from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

    module_config = primus_config.get_module_config("pre_trainer")
    # Reduce the stack to just the distinct layer types (1 dense +/- 1 MoE) and
    # fit it on the available GPUs.
    _limit_layers_for_projection(module_config)
    _rescale_expert_parallelism(module_config)

    # Build (and later time) a chain of ``capture_layers`` same-type layers per
    # phase instead of a single layer: this averages out per-layer jitter and
    # captures inter-layer effects.  Expand the reduced 1-2 layer pattern into
    # ``capture_layers`` copies of each distinct type (dense block, then MoE).
    capture_layers = max(1, int(getattr(args, "inference_bench_layers", None) or _INFERENCE_CAPTURE_LAYERS))
    reduced_pattern = list(getattr(module_config, "moe_layer_freq", None) or [0])
    types_needed = sorted(set(reduced_pattern))  # 0 (dense) before 1 (MoE)
    expanded_pattern = [t for t in types_needed for _ in range(capture_layers)]
    module_config.num_layers = len(expanded_pattern)
    module_config.moe_layer_freq = expanded_pattern

    # Derive layer metadata (hidden size) from the projection model config.
    proj_model = convert_primus_config_to_projection_config(primus_config).model_config
    hidden = int(proj_model.hidden_size)
    # Classify built layers by the (expanded) pattern we just set, so the type
    # of each constructed layer is known regardless of the full-model pattern.
    moe_pattern = list(expanded_pattern)
    batch = int(getattr(args, "inference_batch_size", None) or 1)
    input_len = int(getattr(args, "input_len", None) or module_config.seq_length or 1024)

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    # Disable training-only overlap / FSDP features for isolated layer timing.
    module_config.overlap_grad_reduce = False
    module_config.overlap_param_gather = False
    module_config.use_torch_fsdp2 = False

    trainer = MegatronPretrainTrainer(
        module_name="pre_trainer",
        primus_config=primus_config,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
        extra_args=unknown_overrides,
    )
    trainer.init()

    # Forward-only benchmark: build ONLY the model. ``trainer.setup()`` would
    # also construct the optimizer (optimizer-state GB), gradient buffers and
    # datasets -- none of which a forward-only timing needs -- and that extra
    # ~28 GB is what OOMs on a shared/loaded GPU.  Build the bare model instead
    # (no DDP wrap → no grad buffers) so the footprint is just weights +
    # activations.
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model

    try:
        model = get_model(trainer.model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    except TypeError:
        model = get_model(trainer.model_provider, ModelType.encoder_or_decoder)
    layers = _unwrap_layers(model)
    if not layers:
        raise RuntimeError("[Primus:Inference:Benchmark] No transformer layers found in model.")

    # Serving runs in inference (eval) mode: no dropout. This both matches the
    # real forward pass and removes the per-forward RNG-tracker ``set_current_seed``
    # calls that otherwise make the layer un-capturable by a CUDA/HIP graph.
    for _layer in layers:
        _layer.eval()

    # Swap in Megatron's inference RNG tracker: its ``fork()`` is a no-op
    # (nullcontext), so the transformer layers no longer set the CUDA RNG seed
    # per forward -- which is what makes them capturable by a CUDA/HIP graph.
    try:
        from megatron.core.tensor_parallel.random import initialize_rng_tracker

        initialize_rng_tracker(inference_rng_tracker=True, force_reset=True)
    except Exception as _exc:  # noqa: BLE001 - non-fatal; capture will just fall back
        if rank == 0:
            print(f"[Primus:Inference:Benchmark] could not set inference RNG tracker: {_exc}")

    if not moe_pattern:
        moe_pattern = [0] * len(layers)

    # Time each step under a CUDA/HIP graph by default so the measured decode
    # step reflects graph-replayed serving (no per-kernel host launch overhead),
    # matching how vLLM/SGLang execute decode.  Disable with
    # ``PRIMUS_INF_BENCH_CUDA_GRAPH=0`` (falls back to eager timing).
    use_cuda_graph = os.environ.get("PRIMUS_INF_BENCH_CUDA_GRAPH", "1").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }

    def _bench_stack(stack_module, cfg, q_len) -> float:
        fwd_ms, _, _ = benchmark_layer(
            stack_module,
            [(q_len, batch, hidden)],
            transformer_config=cfg,
            forward_only=True,
            use_cuda_graph=use_cuda_graph,
        )
        return float(fwd_ms)

    # Group the constructed layers by type, then time a chained stack of each so
    # the reported per-layer time is the stack time divided by the layer count.
    layers_by_type: Dict[str, list] = {_DENSE: [], _MOE: []}
    for idx, layer in enumerate(layers):
        is_moe = bool(moe_pattern[idx]) if idx < len(moe_pattern) else False
        layers_by_type[_MOE if is_moe else _DENSE].append(layer)

    measured: Dict[str, Dict[str, float]] = {}
    for ltype in (_DENSE, _MOE):
        ls = layers_by_type[ltype]
        if not ls:
            continue
        n = len(ls)
        cfg = getattr(ls[0], "config", None)
        stack = _make_layer_stack(ls)
        prefill_ms = _bench_stack(stack, cfg, input_len) / n
        decode_ms = _bench_stack(stack, cfg, 1) / n
        measured[ltype] = {"prefill_ms": prefill_ms, "decode_ms": decode_ms}
        if rank == 0:
            print(
                f"[Primus:Inference:Benchmark] {ltype} layer (avg of {n} chained)  "
                f"prefill(q={input_len})={prefill_ms:.3f} ms  decode(q=1)={decode_ms:.3f} ms"
            )

    result = {
        "measured": measured,
        "meta": {
            "batch": batch,
            "input_len": input_len,
            "hidden": hidden,
            "capture_layers": capture_layers,
        },
    }

    if rank == 0 and getattr(args, "save_profiling", None):
        with open(args.save_profiling, "w") as f:
            json.dump(result, f)
        print(f"[Primus:Inference:Benchmark] wrote {args.save_profiling}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return result if rank == 0 else None


def spawn_inference_benchmark(args, overrides=None) -> Optional[dict]:
    """Spawn the torchrun worker and return the measured forward times.

    ``overrides`` are the raw trailing CLI override tokens (e.g. tokenizer /
    mock-data settings) which are forwarded verbatim to the worker so it can
    build the real model.  Returns the parsed result dict, or ``None`` on
    failure (caller should fall back to simulation).
    """
    benchmark_gpus = int(getattr(args, "benchmark_gpus", None) or 1)
    master_port = os.getenv("MASTER_PORT", "29500")
    save_path = getattr(args, "save_profiling", None) or (
        f"/tmp/primus_inf_bench_{master_port}_{os.getpid()}.json"
    )
    if os.path.exists(save_path):
        os.remove(save_path)
    failed_marker = save_path + ".failed"
    if os.path.exists(failed_marker):
        os.remove(failed_marker)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        free_port = s.getsockname()[1]

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={benchmark_gpus}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=localhost",
        f"--master_port={free_port}",
        "-m",
        "primus.cli.main",
        "projection",
        "inference",
        "--config",
        str(args.config),
        "--inference-bench-worker",
        "--save-profiling",
        save_path,
    ]
    # Forward the request-shape knobs the worker needs.
    for attr, flag in [
        ("input_len", "--input-len"),
        ("output_len", "--output-len"),
        ("inference_batch_size", "--inference-batch-size"),
        ("inference_bench_layers", "--inference-bench-layers"),
        ("gpu_arch", "--gpu-arch"),
        ("gpu_clock_mhz", "--gpu-clock-mhz"),
        ("gemm_backend", "--gemm-backend"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            cmd.extend([flag, str(val)])

    # Forward arbitrary trailing overrides (tokenizer, mock_data, parallelism…).
    if overrides:
        cmd.extend([str(o) for o in overrides])

    # Primus' RemotePlatform needs NNODES / NODE_RANK; torchrun only sets the
    # master/rank vars. Default to single-node for the local benchmark.
    env = os.environ.copy()
    env.setdefault("NNODES", "1")
    env.setdefault("NODE_RANK", "0")

    print(
        f"[Primus:Inference:Benchmark] launching forward-only layer benchmark on "
        f"{benchmark_gpus} GPU(s)..."
    )
    timeout_s = int(getattr(args, "benchmark_timeout_s", None) or 1800)
    try:
        result = subprocess.run(cmd, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print("[Primus:Inference:Benchmark] worker timed out; falling back to simulation.")
        return None

    if result.returncode != 0:
        print(
            f"[Primus:Inference:Benchmark] worker failed (exit {result.returncode}); "
            "falling back to simulation."
        )
        return None

    # Poll briefly for the result file (rank 0 writes it just before exit).
    for _ in range(120):
        if os.path.exists(save_path) and os.path.getsize(save_path) > 2:
            break
        time.sleep(0.5)
    if not (os.path.exists(save_path) and os.path.getsize(save_path) > 2):
        print("[Primus:Inference:Benchmark] no result file produced; falling back to simulation.")
        return None

    with open(save_path) as f:
        return json.load(f)

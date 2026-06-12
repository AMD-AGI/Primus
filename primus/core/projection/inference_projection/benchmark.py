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
    passes of one dense + one MoE layer at the **prefill** shape
    ``(batch, input_len)`` and the **decode** shape ``(batch, 1)``; rank 0
    writes the measured times to a JSON file.
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
    # Keep the benchmarked model tiny (1 dense + 1 MoE layer is enough for a
    # per-layer-type measurement) and fit it on the available GPUs.
    _limit_layers_for_projection(module_config)
    _rescale_expert_parallelism(module_config)

    # Derive layer metadata (hidden size + which layers are MoE) from the
    # projection model config — the raw module config has no ``moe_pattern``.
    proj_model = convert_primus_config_to_projection_config(primus_config).model_config
    hidden = int(proj_model.hidden_size)
    moe_pattern = list(proj_model.moe_pattern or [])
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
    trainer.setup()

    model = getattr(trainer, "model", None)
    layers = _unwrap_layers(model)
    if not layers:
        raise RuntimeError("[Primus:Inference:Benchmark] No transformer layers found in model.")

    if not moe_pattern:
        moe_pattern = [0] * len(layers)

    def _bench(layer_module, q_len) -> float:
        cfg = getattr(layer_module, "config", None)
        fwd_ms, _, _ = benchmark_layer(
            layer_module,
            [(q_len, batch, hidden)],
            transformer_config=cfg,
            forward_only=True,
        )
        return float(fwd_ms)

    measured: Dict[str, Dict[str, float]] = {}
    done_types = set()
    for idx, layer in enumerate(layers):
        is_moe = bool(moe_pattern[idx]) if idx < len(moe_pattern) else False
        ltype = _MOE if is_moe else _DENSE
        if ltype in done_types:
            continue
        prefill_ms = _bench(layer, input_len)
        decode_ms = _bench(layer, 1)
        measured[ltype] = {"prefill_ms": prefill_ms, "decode_ms": decode_ms}
        done_types.add(ltype)
        if rank == 0:
            print(
                f"[Primus:Inference:Benchmark] {ltype} layer  "
                f"prefill(q={input_len})={prefill_ms:.3f} ms  decode(q=1)={decode_ms:.3f} ms"
            )

    result = {
        "measured": measured,
        "meta": {"batch": batch, "input_len": input_len, "hidden": hidden},
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

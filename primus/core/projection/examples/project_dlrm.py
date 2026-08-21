###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Run a first-principles projection of a DLRM-v4 (TorchRec / HSTU) ranker.

This is a self-contained entry point that does NOT require a full primus
launcher config: it builds a projection ``TrainingConfig`` directly, resolves
the ``torchrec_dlrm`` workload through the workload registry, builds the
profiler tree, and prints the parameter / memory breakdown plus a first-cut
throughput estimate priced with the shared GEMM/SDPA simulation backends.

Defaults reproduce the Yambda-5B DLRM-v4 ranker from the gap report:
  5 HSTU layers, D=512, 4 heads, d_qk=d_v=128, max_seq_len=4096, jagged ~0.4,
  11 sparse tables (~560 GB fp32 params), 8x GPU single node.

Usage:
  python -m primus.core.projection.examples.project_dlrm
  python -m primus.core.projection.examples.project_dlrm --gpu-arch mi300x --nnodes 1 --gpus-per-node 8
"""

import argparse
import os

from primus.core.projection.module_profilers.language_model import build_profiler
from primus.core.projection.simulation_backends.factory import (
    get_gemm_simulation_backend,
    get_sdpa_simulation_backend,
)
from primus.core.projection.training_config import (
    ModelConfig,
    ModelParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from primus.core.projection.workload_registry import resolve_top_level_spec


def build_yambda_config(args) -> TrainingConfig:
    total_embed_bytes = args.embedding_gb * (1024**3)
    total_rows = int(total_embed_bytes / args.embedding_param_bytes / args.embedding_dim)

    model = ModelConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        # sparse embeddings
        num_embedding_tables=args.num_tables,
        embedding_total_rows=total_rows,
        embedding_dim=args.embedding_dim,
        embedding_default_pooling_factor=args.pooling_factor,
        embedding_sharding="row",
        embedding_param_bytes=args.embedding_param_bytes,
        embedding_hbm_fraction=args.hbm_fraction,
        # HSTU
        hstu_num_heads=args.num_heads,
        hstu_qk_dim=args.qk_dim,
        hstu_v_dim=args.v_dim,
        hstu_max_seq_len=args.max_seq_len,
        hstu_fill_factor=args.fill_factor,
        num_attention_heads=args.num_heads,
        kv_channels=args.qk_dim,
        # dense MLPs
        dense_input_dim=args.dense_input_dim,
        dlrm_bottom_mlp=args.bottom_mlp,
        dlrm_over_mlp=args.over_mlp,
    )
    runtime = RuntimeConfig(
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.max_seq_len,
        data_parallel_size=args.gpus_per_node * args.nnodes // args.tp,
    )
    mp = ModelParallelConfig(tensor_model_parallel_size=args.tp)
    return TrainingConfig(
        model_config=model,
        runtime_config=runtime,
        model_parallel_config=mp,
        framework="torchrec_dlrm",
    )


def _gb(x):
    return f"{x / (1024**3):.2f} GB"


def main():
    p = argparse.ArgumentParser(description="Project a DLRM-v4 (HSTU) ranker.")
    p.add_argument("--num-layers", type=int, default=5)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--qk-dim", type=int, default=128)
    p.add_argument("--v-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--fill-factor", type=float, default=0.4)
    p.add_argument("--num-tables", type=int, default=11)
    p.add_argument("--embedding-gb", type=float, default=560.0, help="total embedding param bytes (GB)")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--embedding-param-bytes", type=int, default=4)
    p.add_argument("--pooling-factor", type=int, default=20)
    p.add_argument("--hbm-fraction", type=float, default=1.0)
    p.add_argument("--dense-input-dim", type=int, default=512)
    p.add_argument("--bottom-mlp", type=int, nargs="*", default=[512, 512])
    p.add_argument("--over-mlp", type=int, nargs="*", default=[512, 256, 1])
    p.add_argument("--global-batch-size", type=int, default=8192)
    p.add_argument("--micro-batch-size", type=int, default=1024)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--gpu-arch", type=str, default="mi300x")
    args = p.parse_args()

    # World size is read from the environment by the profilers/collectives.
    os.environ["NNODES"] = str(args.nnodes)
    os.environ["GPUS_PER_NODE"] = str(args.gpus_per_node)
    world = args.nnodes * args.gpus_per_node

    config = build_yambda_config(args)

    spec = resolve_top_level_spec(config)
    profiler = build_profiler(spec)

    gemm = get_gemm_simulation_backend(gpu_arch=args.gpu_arch)
    sdpa = get_sdpa_simulation_backend(gpu_arch=args.gpu_arch)
    profiler.set_simulation_backends(gemm_backend=gemm, sdpa_backend=sdpa)

    emb = profiler.sub_profilers["sparse_embedding"]
    total_params = profiler.estimated_num_params(None)
    per_rank_params = profiler.estimated_num_params(0)
    hbm_bytes, ddr_bytes = emb.param_bytes_by_tier(0)
    bpp = profiler.get_num_bytes_per_param()

    print("=" * 78)
    print(
        f"DLRM-v4 (HSTU) projection  --  world={world} ({args.nnodes}x{args.gpus_per_node}), arch={args.gpu_arch}"
    )
    print("=" * 78)
    print("[Model]")
    print(
        f"  HSTU layers          : {args.num_layers}  (D={args.hidden_size}, heads={args.num_heads}, "
        f"d_qk={args.qk_dim}, d_v={args.v_dim})"
    )
    print(
        f"  seq_len (padded)     : {args.max_seq_len}  fill={args.fill_factor}  "
        f"(effective ~{int(args.max_seq_len * args.fill_factor)})"
    )
    print(
        f"  sparse tables        : {args.num_tables}  dim={args.embedding_dim}  "
        f"pooling={args.pooling_factor}  ({args.embedding_param_bytes}B/param)"
    )
    print("[Parameters]")
    print(f"  total params         : {total_params / 1e9:.2f} B")
    print(f"  per-rank params      : {per_rank_params / 1e9:.2f} B   (row-sharded across {world} ranks)")
    print(f"  embedding HBM tier   : {_gb(hbm_bytes)}   DDR/UVM tier: {_gb(ddr_bytes)}")
    print(f"  bytes/param (static) : {bpp:.2f}  ->  ~{_gb(per_rank_params * bpp)} static/rank")
    print("[Throughput]")
    step = profiler.project_step()
    print(f"  forward              : {step['forward_ms']:.2f} ms")
    print(f"  backward             : {step['backward_ms']:.2f} ms")
    print(f"  embedding all-to-all : {step['comm_ms']:.2f} ms")
    print(f"  step time            : {step['step_ms']:.2f} ms")
    print(
        f"  per-HSTU-layer       : fwd {step['hstu_layer_fwd_ms']:.3f} ms / bwd {step['hstu_layer_bwd_ms']:.3f} ms"
    )
    print(f"  samples/s (global)   : {step['samples_per_s']:,.0f}")
    print(f"  samples/s per GPU    : {step['samples_per_s_per_gpu']:,.0f}")
    print("=" * 78)


if __name__ == "__main__":
    main()

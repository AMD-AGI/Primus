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
  5 HSTU layers, D=512, 4 heads, d_qk=d_v=128, max_seq_len=4096, jagged fill
  ~0.425 (std ~0.079), 11 sparse tables (~560 GB fp32 params) with row-wise
  Adagrad state, per-token pooling (~1741 for the 3 sequence features, 1 for
  the 8 contextual), RowWiseAdagrad sparse optimizer, 8x GPU single node.

The default GEMM/SDPA backend is ``gemmologist`` (overridable via
``--gemm-backend`` or ``PRIMUS_GEMM_BACKEND``); ``origami`` raises on
MI450-class parts. ``gemmologist`` ships as an out-of-tree plugin -- register
it with ``PRIMUS_GEMM_BACKEND_PLUGINS`` before running on those parts.

Usage:
  python -m primus.core.projection.examples.project_dlrm --gpu-arch mi355x
  PRIMUS_GEMM_BACKEND=gemmologist \\
    python -m primus.core.projection.examples.project_dlrm --gpu-arch mi450x
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
        embedding_pooling_factor=(args.pooling_factors or None),
        embedding_default_pooling_factor=args.pooling_factor,
        embedding_sharding="row",
        embedding_param_bytes=args.embedding_param_bytes,
        embedding_hbm_fraction=args.hbm_fraction,
        embedding_optimizer=args.embedding_optimizer,
        # HSTU
        hstu_num_heads=args.num_heads,
        hstu_qk_dim=args.qk_dim,
        hstu_v_dim=args.v_dim,
        hstu_max_seq_len=args.max_seq_len,
        hstu_fill_factor=args.fill_factor,
        hstu_fill_factor_std=args.fill_factor_std,
        hstu_attn_efficiency=args.attn_efficiency,
        hstu_elementwise_passes=args.elementwise_passes,
        num_attention_heads=args.num_heads,
        kv_channels=args.qk_dim,
        # dense MLPs
        dense_input_dim=args.dense_input_dim,
        dlrm_bottom_mlp=args.bottom_mlp,
        dlrm_over_mlp=args.over_mlp,
        dlrm_comm_exposed_fraction=args.comm_exposed_fraction,
        dlrm_h2d_ms=args.h2d_ms,
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
    p.add_argument("--fill-factor", type=float, default=0.425)
    p.add_argument(
        "--fill-factor-std",
        type=float,
        default=0.079,
        help="std-dev of jagged fill; attention cost ~ E[L^2] = mean^2 + std^2",
    )
    p.add_argument(
        "--attn-efficiency",
        type=float,
        default=1.0,
        help="ragged_hstu kernel efficiency vs the FAv3 roofline (<1 derates attn)",
    )
    p.add_argument(
        "--elementwise-passes",
        type=float,
        default=6.0,
        help="read+write elementwise passes (dropout/norm/gate/pack) per direction",
    )
    p.add_argument("--num-tables", type=int, default=11)
    p.add_argument("--embedding-gb", type=float, default=560.0, help="total embedding param bytes (GB)")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--embedding-param-bytes", type=int, default=4)
    p.add_argument("--pooling-factor", type=int, default=1, help="default per-table pooling (fallback)")
    p.add_argument(
        "--pooling-factors",
        type=int,
        nargs="*",
        # Yambda-5B: 3 sequence features (item/artist/album) looked up ~once per
        # valid position (~fill*seq), 8 contextual features looked up once.
        default=[1741, 1741, 1741, 1, 1, 1, 1, 1, 1, 1, 1],
        help="per-table pooling factors (lookups per sample)",
    )
    p.add_argument(
        "--embedding-optimizer",
        type=str,
        default="rowwise_adagrad",
        help="sparse optimizer: rowwise_adagrad | adagrad | adam",
    )
    p.add_argument("--hbm-fraction", type=float, default=1.0)
    p.add_argument(
        "--comm-exposed-fraction",
        type=float,
        default=1.0,
        help="fraction of embedding a2a not overlapped behind compute",
    )
    p.add_argument("--h2d-ms", type=float, default=0.0, help="optional measured H2D copy time (ms)")
    p.add_argument("--dense-input-dim", type=int, default=512)
    p.add_argument("--bottom-mlp", type=int, nargs="*", default=[512, 512])
    p.add_argument("--over-mlp", type=int, nargs="*", default=[512, 256, 1])
    p.add_argument("--global-batch-size", type=int, default=8192)
    p.add_argument("--micro-batch-size", type=int, default=1024)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--gpu-arch", type=str, default="mi355x")
    p.add_argument(
        "--gemm-backend",
        type=str,
        default=os.getenv("PRIMUS_GEMM_BACKEND", "gemmologist"),
        help="GEMM/SDPA simulation backend (origami raises on MI450-class archs)",
    )
    args = p.parse_args()

    # Guard the --pooling-factors / --num-tables consistency.
    if args.pooling_factors and len(args.pooling_factors) != args.num_tables:
        p.error(
            f"--pooling-factors has {len(args.pooling_factors)} entries but "
            f"--num-tables is {args.num_tables}"
        )

    # World size + arch are read from the environment by the profilers and the
    # collective-arg fabric resolver.
    os.environ["NNODES"] = str(args.nnodes)
    os.environ["GPUS_PER_NODE"] = str(args.gpus_per_node)
    os.environ.setdefault("PRIMUS_GPU_ARCH", args.gpu_arch)
    world = args.nnodes * args.gpus_per_node

    config = build_yambda_config(args)

    spec = resolve_top_level_spec(config)
    profiler = build_profiler(spec)

    # origami raises "Unknown GPU architecture" on MI450-class parts; the
    # gemmologist backend prices them, so it is the default here.
    gemm = get_gemm_simulation_backend(backend_name=args.gemm_backend, gpu_arch=args.gpu_arch)
    sdpa = get_sdpa_simulation_backend(backend_name=args.gemm_backend, gpu_arch=args.gpu_arch)
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
        f"  seq_len (padded)     : {args.max_seq_len}  fill={args.fill_factor}"
        f"+-{args.fill_factor_std}  (effective ~{int(args.max_seq_len * args.fill_factor)})"
    )
    pooling_repr = args.pooling_factors if args.pooling_factors else args.pooling_factor
    print(
        f"  sparse tables        : {args.num_tables}  dim={args.embedding_dim}  "
        f"pooling={pooling_repr}  opt={args.embedding_optimizer}  ({args.embedding_param_bytes}B/param)"
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
    print(
        f"  embedding all-to-all : {step['comm_ms']:.2f} ms exposed "
        f"({step['comm_ms_unoverlapped']:.2f} ms raw)"
    )
    if step.get("h2d_ms"):
        print(f"  host->device copy    : {step['h2d_ms']:.2f} ms")
    print(f"  step time            : {step['step_ms']:.2f} ms")
    print(
        f"  per-HSTU-layer       : fwd {step['hstu_layer_fwd_ms']:.3f} ms / bwd {step['hstu_layer_bwd_ms']:.3f} ms"
    )
    print(f"  samples/s (global)   : {step['samples_per_s']:,.0f}")
    print(f"  samples/s per GPU    : {step['samples_per_s_per_gpu']:,.0f}")
    peak = step.get("peak_tflops_per_gpu")
    mfu = step.get("mfu")
    if mfu is not None and peak:
        print(
            f"  MFU / HFU            : {mfu * 100:.1f}% / {step['hfu'] * 100:.1f}%  "
            f"({step['achieved_tflops_per_gpu']:.0f} / {peak:.0f} TFLOPS per GPU)"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()

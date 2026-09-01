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

Defaults reproduce the measured Yambda-5B run (chriscai-amd/training
yambda_5b.gin): 3 HSTU layers, D=512, 4 heads, d_qk=d_v=128, max_seq_len=4096,
jagged fill ~0.42 (std ~0.079), a per-token ContextualPreprocessor and a single
listen_plus prediction tower (512->512->1), 11 sparse tables (~560 GB fp32
params) with RowWiseAdagrad, per-token pooling (~1741 for the 3 sequence
features, 1 for the 8 contextual), 8x GPU single node.

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
        embedding_grad_scatter_efficiency=args.grad_scatter_efficiency,
        # HSTU
        hstu_num_heads=args.num_heads,
        hstu_qk_dim=args.qk_dim,
        hstu_v_dim=args.v_dim,
        hstu_max_seq_len=args.max_seq_len,
        hstu_fill_factor=args.fill_factor,
        hstu_fill_factor_std=args.fill_factor_std,
        hstu_attn_efficiency=args.attn_efficiency,
        hstu_attn_bwd_efficiency=args.attn_bwd_efficiency,
        hstu_attn_flop_efficiency=(0.0 if args.attn_model == "fav3_hstu" else args.attn_flop_efficiency),
        hstu_attn_model=args.attn_model,
        hstu_attn_epilogue_gelem_fwd=args.attn_epilogue_gelem_fwd,
        hstu_attn_epilogue_gelem_bwd=args.attn_epilogue_gelem_bwd,
        hstu_attn_bwd_ratio=args.attn_bwd_ratio,
        hstu_recompute_attn=args.recompute_attn,
        hstu_output_input_dim=args.output_input_dim,
        hstu_elementwise_passes=args.elementwise_passes,
        num_attention_heads=args.num_heads,
        kv_channels=args.qk_dim,
        # dense MLPs
        dense_input_dim=args.dense_input_dim,
        dlrm_bottom_mlp=args.bottom_mlp,
        dlrm_over_mlp=args.over_mlp,
        dlrm_preprocessor_mlp=args.preprocessor_mlp,
        dlrm_preprocessor_input_dim=args.preprocessor_input_dim,
        dlrm_preprocessor_gemms=args.preprocessor_gemms,
        dlrm_prediction_head_mlp=args.prediction_head_mlp,
        dlrm_num_tasks=args.num_tasks,
        dlrm_comm_exposed_fraction=args.comm_exposed_fraction,
        dlrm_collective_sync_ms=args.collective_sync_ms,
        dlrm_h2d_ms=args.h2d_ms,
        dlrm_memcpy_ms=args.memcpy_ms,
        dlrm_reduce_ms=args.reduce_ms,
        dlrm_glue_bytes_per_token=args.glue_bytes_per_token,
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
    # Defaults reproduce the measured yambda_5b.gin submission (chriscai-amd/
    # training): HSTU_NUM_LAYERS=3, MAX_SEQ_LEN=4096, listen_plus single task.
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--qk-dim", type=int, default=128)
    p.add_argument("--v-dim", type=int, default=128)
    # Measured from the MI350X step-52 (flydsl) trace: max_seq_len=3650, total
    # valid tokens T=2,229,337 over 1024 local sequences -> mean fill ~0.596.
    p.add_argument("--max-seq-len", type=int, default=3650)
    p.add_argument("--fill-factor", type=float, default=0.6)
    p.add_argument(
        "--fill-factor-std",
        type=float,
        default=0.0,
        help="std-dev of jagged fill; attention cost ~ E[L^2] = mean^2 + std^2",
    )
    p.add_argument(
        "--attn-efficiency",
        type=float,
        default=1.0,
        help="ragged_hstu kernel efficiency vs the FAv3 roofline (used only when --attn-flop-efficiency=0)",
    )
    p.add_argument(
        "--attn-bwd-efficiency",
        type=float,
        default=0.0,
        help="separate efficiency for the (unautotuned) attention backward; 0 reuses --attn-efficiency",
    )
    p.add_argument(
        "--attn-flop-efficiency",
        type=float,
        # Measured: attn fwd flops (14.9 TF over 3 layers) at 43.23 ms vs the
        # 1404 TF/s realizable peak -> 0.246 achieved fraction for gated-jagged
        # HSTU attention on MI350X.  >0 selects the direct-FLOP attention model.
        default=0.246,
        help="achieved fraction of matmul peak for the gated-jagged HSTU attention core (0 = use SDPA roofline)",
    )
    p.add_argument(
        "--attn-bwd-ratio",
        type=float,
        default=2.025,  # measured bwd_dkdv (87.55 ms) / attn_fwd (43.23 ms)
        help="attention backward/forward wall-time ratio (FLOP model)",
    )
    p.add_argument(
        "--attn-model",
        type=str,
        default="flop",
        choices=["flop", "fav3_hstu"],
        help="attention cost model: 'flop' (direct-FLOP roofline, default) or "
        "'fav3_hstu' (FAv3 tile-level matmuls via origami 1-CU + HSTU pointwise epilogue)",
    )
    p.add_argument(
        "--attn-epilogue-gelem-fwd",
        type=float,
        default=0.0,
        help="fav3_hstu fused-epilogue fwd throughput (Gelem/s); 0 = calibrated default",
    )
    p.add_argument(
        "--attn-epilogue-gelem-bwd",
        type=float,
        default=0.0,
        help="fav3_hstu fused-epilogue bwd throughput (Gelem/s); 0 = calibrated default",
    )
    p.add_argument(
        "--recompute-attn",
        action="store_true",
        default=True,
        help="HSTU recomputes UVQK in the backward (selective activation recompute); measured on",
    )
    p.add_argument("--no-recompute-attn", dest="recompute_attn", action="store_false")
    p.add_argument(
        "--output-input-dim",
        type=int,
        default=1536,  # measured HSTU output projection input width (3*D)
        help="HSTU output-projection input width; 0 -> H*d_v",
    )
    p.add_argument(
        "--elementwise-passes",
        type=float,
        # Measured: layernorm/dropout (53.6 ms) + elementwise (19.9 ms) = 73.4 ms
        # of HBM-streaming glue over the [T, uvqk+D] footprint across 3 layers.
        default=4.63,
        help="read+write elementwise passes (dropout/norm/gate) per direction",
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
        # Yambda-5B: 3 sequence features (item/artist/album) looked up once per
        # valid position (mean ~2177 = fill*seq), 8 contextual features once.
        # Measured: 3 x 2177 x 1024 ~= 6.69M gathered rows (trace: 6.67M).
        default=[2177, 2177, 2177, 1, 1, 1, 1, 1, 1, 1, 1],
        help="per-table pooling factors (lookups per sample)",
    )
    p.add_argument(
        "--embedding-optimizer",
        type=str,
        default="rowwise_adagrad",
        help="sparse optimizer: rowwise_adagrad | adagrad | adam",
    )
    p.add_argument(
        "--grad-scatter-efficiency",
        type=float,
        # Measured: 48.54 ms fp32 atomic scatter over 6.67M rows x 512 x 4B
        # (read+accumulate) at 7.2 TB/s -> 0.078 achieved fraction.
        default=0.078,
        help="effective peak-HBM fraction for the fp32 atomic embedding grad scatter-add",
    )
    p.add_argument("--hbm-fraction", type=float, default=1.0)
    p.add_argument(
        "--comm-exposed-fraction",
        type=float,
        # Measured: the embedding a2a is fully overlapped behind compute (only
        # 0.86 ms of collectives is exposed, carried by --collective-sync-ms).
        default=0.0,
        help="fraction of embedding a2a not overlapped behind compute",
    )
    p.add_argument(
        "--collective-sync-ms",
        type=float,
        # Measured (flydsl trace): collective kernels total 3.55 ms and only
        # 0.86 ms is exposed (not overlapped by compute) -- communication is
        # effectively free in this run, not the 32-85 ms of the Triton run.
        default=0.86,
        help="measured exposed collective peer-wait/barrier time per step (ms); not bandwidth-derivable",
    )
    p.add_argument("--h2d-ms", type=float, default=0.0, help="optional measured H2D copy time (ms)")
    p.add_argument(
        "--memcpy-ms",
        type=float,
        default=0.0,
        help="fixed device memcpy time per step (ms); superseded by --glue-bytes-per-token",
    )
    p.add_argument(
        "--reduce-ms",
        type=float,
        default=0.0,
        help="fixed local reduce-kernel time per step (ms); superseded by --glue-bytes-per-token",
    )
    p.add_argument(
        "--glue-bytes-per-token",
        type=float,
        # Measured: memcpy (47.2 ms) + reduce (22.6 ms) + jagged pack/unpack
        # (15.2 ms) = 85.0 ms of HBM streaming over 2.229M tokens at 7.2 TB/s x
        # 0.6 -> ~1.65e5 bytes/token of framework data-movement glue.
        default=164693.0,
        help="framework glue (jagged/cast/reduce) HBM bytes moved per valid token; scales with tokens",
    )
    p.add_argument(
        "--trace-roles",
        action="store_true",
        help="print the projected per-role breakdown alongside the MI350X step-52 trace",
    )
    p.add_argument("--dense-input-dim", type=int, default=512)
    p.add_argument("--bottom-mlp", type=int, nargs="*", default=[512, 512])
    p.add_argument("--over-mlp", type=int, nargs="*", default=[512, 256, 1])
    p.add_argument(
        "--preprocessor-mlp",
        type=int,
        nargs="*",
        # ContextualPreprocessor per-token MLPs (content 512->256->512, additional
        # 1024->256->512, action 8->256->512) collapsed to a FLOP-equivalent single
        # chain 512->768->512 (2*(512*768+768*512) ~= sum of the three branches).
        default=[768, 512],
        help="per-token input-preprocessor MLP output widths (fuses contextual features)",
    )
    p.add_argument(
        "--preprocessor-input-dim",
        type=int,
        default=512,
        help="preprocessor input width; 0 -> (num_contextual+1) x D (used only if --preprocessor-gemms unset)",
    )
    p.add_argument(
        "--preprocessor-gemms",
        type=str,
        # Measured ContextualPreprocessor branches (per valid token):
        # content 512->256, additional 1024->256, action 24->256, fuse 256->512.
        default="[[512,256],[1024,256],[24,256],[256,512]]",
        help="explicit per-token preprocessor GEMMs '[[in,out],...]' (overrides --preprocessor-mlp)",
    )
    p.add_argument(
        "--prediction-head-mlp",
        type=int,
        nargs="*",
        default=[512, 1],  # listen_plus tower: 512 -> 512 -> 1
        help="per-sample multitask prediction-head tower widths",
    )
    p.add_argument("--num-tasks", type=int, default=1, help="number of multitask prediction heads")
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
        f"({step['comm_ms_unoverlapped']:.2f} ms raw bw"
        + (f" + {step['comm_sync_ms']:.2f} ms sync)" if step.get("comm_sync_ms") else ")")
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
    roles = step.get("roles")
    if roles and args.trace_roles:
        # Measured GPU-busy kernel time by role, rank 0, ProfilerStep#52 of the
        # MI350X flydsl trace (trace_step52_flydsl.json): GPU busy 467.4 ms of a
        # 471.7 ms wall step (99% busy), collectives 0.86 ms exposed.
        measured = {
            "dense_gemm": 132.29,
            "attention": 130.78,
            "elementwise": 73.43,  # layernorm/dropout (53.56) + elementwise (19.87)
            "embedding": 51.93,
            "glue": 84.99,  # memcpy (47.19) + reduce (22.62) + jagged (15.18)
            "collectives": 0.86,  # exposed (3.55 ms total kernel time)
        }
        # Fold the model's fixed memcpy/reduce knobs into the glue role for the
        # side-by-side (they default to 0 now that glue is token-scaled).
        proj = dict(roles)
        proj["glue"] = roles.get("glue", 0.0) + roles.get("memcpy", 0.0) + roles.get("reduce", 0.0)
        print("[Role breakdown vs measured (MI350X flydsl step-52, rank 0)]")
        print(f"  {'role':<14}{'projected':>12}{'measured':>12}{'ratio':>9}")
        p_tot = m_tot = 0.0
        for r in ("dense_gemm", "attention", "elementwise", "embedding", "glue", "collectives"):
            pj = proj.get(r, 0.0)
            ms = measured[r]
            ratio = (pj / ms) if ms else 0.0
            p_tot += pj
            m_tot += ms
            print(f"  {r:<14}{pj:>11.2f} {ms:>11.2f} {ratio:>8.2f}x")
        print(f"  {'TOTAL':<14}{p_tot:>11.2f} {m_tot:>11.2f} {(p_tot / m_tot if m_tot else 0):>8.2f}x")
        print("  (measured GPU-busy union = 467.44 ms; step wall = 471.71 ms)")
    print("=" * 78)


if __name__ == "__main__":
    main()

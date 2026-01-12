###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

from primus.tools.preflight.global_vars import (
    ITERATION,
    LOCAL_RANK,
    LOCAL_WORLD_SIZE,
    RANK,
    WARMUP,
    WORLD_SIZE,
    get_hostnames,
)
from primus.tools.preflight.utility import create_dir, log


def run_square_gemm(args):
    sizes = [1024, 2048, 4096, 8192, 10240]
    latency_results = {}
    flops_results = {}
    for size in sizes:
        a = torch.randn((size, size), device=f"cuda:{LOCAL_RANK}", dtype=torch.bfloat16)
        b = torch.randn((size, size), device=f"cuda:{LOCAL_RANK}", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERATION):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        t = (end - start) / ITERATION
        tflops = 2 * size * size * size / (t * 1e12)
        latency_results[f"{size}x{size}x{size}"] = t
        flops_results[f"{size}x{size}x{size}"] = tflops
    all_latency_results = [None for _ in range(WORLD_SIZE)]
    all_tflops_results = [None for _ in range(WORLD_SIZE)]
    dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
    dist.gather_object(flops_results, all_tflops_results if RANK == 0 else None, dst=0)

    if RANK == 0:
        max_len = max(len(s) for s in get_hostnames()) + 2
        sizes_sorted = flops_results.keys()
        formatted_sizes = [f"{size:<14}" for size in sizes_sorted]

        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"# Square Gemm Perf\n\n")
            f.write(f"=======Square GEMM Latency (us)=======\n")
            log("=======Square GEMM Latency (us)=======")

            # f.write(f"| Hostname | Node | Rank |\n")
            # f.write(f"|----------|----------|----------|\n")
            f.write(f"| Hostname | Node | Rank | {' | '.join(formatted_sizes)}|\n")
            f.write(f"|----------|----------|----------{'|----------' * len(formatted_sizes)}|\n")
            log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
            for rank, result in enumerate(all_latency_results):
                hostname = get_hostnames()[rank]
                node_id = rank // LOCAL_WORLD_SIZE
                formatted_values = [f"{result[size]*1000000:<14.2f}" for size in sizes_sorted]
                log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
            f.write(f"\n")

            f.write(f"=======Square GEMM TFLOPS =======\n")
            log("=======Square GEMM TFLOPS=======")

            f.write(f"| Hostname | Node | Rank | {' | '.join(formatted_sizes)}|\n")
            f.write(f"|----------|----------|----------{'|----------' * len(formatted_sizes)}|\n")
            log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_sizes)}")
            for rank, result in enumerate(all_tflops_results):
                hostname = get_hostnames()[rank]
                node_id = rank // LOCAL_WORLD_SIZE
                formatted_values = [f"{result[size]:<14.2f}" for size in sizes_sorted]
                log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
            f.write(f"\n")

        if not args.plot:
            return

        log("=======Plot Square GEMM TFLOPS=======")
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"=======Plot Square GEMM TFLOPS=======\n")
        plot_case = f"square_gemm_tflops"
        dump_path = f"{args.dump_path}/{plot_case}"
        create_dir(dump_path)
        for size_key in sizes_sorted:
            values = [r[size_key] for r in all_tflops_results]
            plt.figure(figsize=(10, 4))
            bars = plt.bar(range(WORLD_SIZE), values)
            plt.xlabel("Rank")
            plt.ylabel("TFLOPS")
            plt.title(f"Square GEMM TFLOPS for {size_key}")
            plt.xticks(range(WORLD_SIZE))
            plt.grid(True, axis="y")

            # plt value
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                )

            png_file = f"square_gemm_tflops_{size_key.replace('x', '_')}.png"
            plt.tight_layout()
            plt.savefig(f"{dump_path}/{png_file}")
            plt.close()
            with open(args.markdown_file, "a", encoding="utf-8") as f:
                f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
        # Bar chart visualization for rank 0
        rank_0_values = [all_tflops_results[0][size_key] for size_key in sizes_sorted]
        plt.figure(figsize=(10, 4))
        bars = plt.bar(sizes_sorted, rank_0_values)
        plt.xlabel("Size")
        plt.ylabel("TFLOPS")
        plt.title("Square GEMM TFLOPS for Rank 0")
        plt.grid(True, axis="y")

        # plt value
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")

        png_file = f"square_gemm_tflops_rank_0.png"
        plt.tight_layout()
        plt.savefig(f"{dump_path}/{png_file}.png")
        plt.close()
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
            f.write(f"\n")
        log(f"")

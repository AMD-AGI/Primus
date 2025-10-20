###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os
import subprocess
from pathlib import Path

CONFIG_DIR = Path("examples/megatron/configs/MI300X")


def find_all_model_configs():
    return sorted(CONFIG_DIR.glob("*.yaml"))


def run_benchmark(config_path, log_dir: Path, nnodes: int):
    model_name = config_path.name.replace("-pretrain.yaml", "")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{model_name}.log"

    print(f"\nRunning benchmark for: {model_name} → log: {log_file}")

    env = os.environ.copy()
    env["EXP"] = str(config_path)
    env["NNODES"] = str(nnodes)
    env["BACKEND"] = "megatron"

    with open(log_file, "w") as f:
        subprocess.run(
            ["bash", "examples/run_slurm_pretrain.sh"],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Primus Benchmark Runner")
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "Specify a model name (without -pretrain.yaml). "
            "For example, for config 'examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml', use: --model llama2_7B"
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("output/benchmarks"),
        help="Directory to store benchmark logs (default: output/benchmarks)",
    )
    parser.add_argument(
        "--nnodes", type=int, default=1, help="Number of nodes to run training on (default: 1)"
    )
    args = parser.parse_args()

    if args.model:
        config_file = CONFIG_DIR / f"{args.model}-pretrain.yaml"
        if not config_file.exists():
            print(f"Config not found: {config_file}")
            return
        run_benchmark(config_file, args.log_dir, args.nnodes)
    else:
        for config_file in find_all_model_configs():
            run_benchmark(config_file, args.log_dir, args.nnodes)


if __name__ == "__main__":
    main()

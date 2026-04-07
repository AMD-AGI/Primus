#!/bin/bash
sbatch -w slurm-compute-node-[0-1] run_bench_gb200.sh
sbatch -w slurm-compute-node-[2-3] run_bench_gb200.sh
sbatch -w slurm-compute-node-[4-5] run_bench_gb200.sh
sbatch -w slurm-compute-node-[6-7] run_bench_gb200.sh
sbatch -w slurm-compute-node-[8-9] run_bench_gb200.sh
sbatch -w slurm-compute-node-[10-11] run_bench_gb200.sh
sbatch -w slurm-compute-node-[12-13] run_bench_gb200.sh
sbatch -w slurm-compute-node-[14-15] run_bench_gb200.sh
sbatch -w slurm-compute-node-[16-17] run_bench_gb200.sh

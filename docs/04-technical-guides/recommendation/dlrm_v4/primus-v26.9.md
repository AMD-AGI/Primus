# DLRMv4 on rocm/primus:v26.9 (Primus)

How the trainer runs in Primus on the published `rocm/primus:v26.9` image, what
differed from the validated ROCm 10.2 stack ([rocm-port.md](rocm-port.md): the UTD
`primus_rocm10.2_20261016` image), and how the Primus recipe was brought to the
same performance. All measurements: one MI355X node (cluster B, b003),
same node and data for every comparison, 2026-11-06.

## Result

| | validated stack (UTD image, original repo) | Primus + rocm/primus:v26.9 |
|---|---|---|
| 200-step median `global_sps` | 84,177 | 84,390-84,455 (+0.3-0.5%) |
| time to target, seed 731542219 | 917.0 s, 5 evals | 917.1 s, 5 evals |
| time to target, seed 1234567 | 967.5 s, 6 evals (other node) | 961.2 s, 6 evals |
| time to target, seed 20261103 | 1054.2 s, 8 evals; eval 7 at 1009.4 s (other node) | 1050.4 s, 8 evals; eval 7 at 1006.0 s |
| `run_with_docker.sh`, seed 20261103 | | 1000.2 s, 7 evals |
| MLPerf 6.1.0 compliance | SUCCESS | SUCCESS (all four runs) |

Seed 20261103's seventh eval lands within 0.0003 of the 0.75 target, so it needs a
seventh or an eighth eval from run to run on either stack.

Before the changes below, the Primus run took 939.6-941.8 s for seed 731542219
(-0.45% throughput, five 3-6 s stalls).

## Stack differences that mattered

| Difference | Effect | What the recipe does |
|---|---|---|
| FlyDSL 0.3.4.1 (image) vs 0.2.4: `flydsl.expr.buffer_ops` / `vector` removed, raw ROCDL builders changed | the FlyDSL attention forward is fail-closed: training stopped at step 1 | `ops/flydsl/_compat.py` gives the kernels the 0.2 API on both; FlyDSL 0.2.4 installed into a private directory (`PRIMUS_RECOMMENDATION_SITE`) that only the trainer puts first on its path, because the kernels built by 0.3.4.1's newer LLVM are 0.19% slower |
| hipBLASLt 100500-cb261897 vs -3dc098eb | every stack-gated table and pin turned itself off: FP8 ScaledGemm pins (and with them the X2L-B fused backward, which fails closed), BLAS replay table, HIP preload table | v26.9 stack added to `ops/blas_solution_pins.py` (solution 33058 verified at run time by X2L-B's bitwise self-check), tables re-recorded (`tools/recommendation/dlrm_v4/record_stack_tables.sh`) |
| Primus `base_env.sh`: `GPU_MAX_HW_QUEUES=2` (HIP default 4) | -0.44% throughput | experiment yaml `env: GPU_MAX_HW_QUEUES: "4"` |
| image `TORCH_DIST_USE_NCCL2=0` (legacy c10d `ProcessGroupNCCL`; the UTD image used nccl2) | none: nccl2 + the deferred-wait DDP hook (T23) is 0.12% slower here | keeps the image's legacy process group |
| image `GPU_USE_DEVICE_QUEUE=1`, `DEBUG_CLR_AQL_DEV_QUEUE=1` | (as validated: set to 0) | `config_MI355X_1x8x1.sh` sets 0 |

200-step A/B (2 runs each, interleaved; on the FP8 pins + HIP table baseline):

| Variant | samples/s | vs baseline |
|---|---|---|
| baseline (Primus env: `GPU_MAX_HW_QUEUES=2`, `TORCH_NCCL_HIGH_PRIORITY=1`) | 84,019-84,049 | - |
| FP8 pins off | 83,749 | -0.36% |
| `TORCH_DIST_USE_NCCL2=1` | 83,944 | -0.12% |
| FlyDSL 0.3.4.1 (the image's) | 83,886 | -0.19% |
| `GPU_MAX_HW_QUEUES=4` | 84,390 | +0.44% |
| `TORCH_NCCL_HIGH_PRIORITY=0` with 2 queues | 79,717 | **-5.1%** |
| `GPU_MAX_HW_QUEUES=4` + `TORCH_NCCL_HIGH_PRIORITY=0` | 84,455 | +0.52% |

## One-time costs in the timed region (T26 / T12 on v26.9)

Without a BLAS replay table for the stack, five windows (43, 50, 51, 53, 63) each
had one step 16-28x the median (m4 speed estimator, step max/median over ranks):
hipBLASLt loading kernel libraries the first time a GEMM size falls into their
region, ~19 s in total. Recording the table on v26.9 found a second kind of cost
the validated stack did not have: in some regions ((1024, 256) dgrad at
M >= 0.76M, (256, 512) forward near 1.1M / 1.5M / 1.9M rows) the first call costs
2-3 s at *every* new M, so no replay can remove it; the dedup chain never reaches
those sizes in production. `gen_blas_replay_table.py filter` keeps only rows whose
cost does not recur at neighbouring M after a replay (42 rows, 139 s of first
uses, paid during init). The HIP preload table is small on this stack (13 rows,
~0.09 s).

## Re-recording after an image update

```bash
# inside the new image, on one MI355X node (the tables check their stack and ignore others)
bash tools/recommendation/dlrm_v4/record_stack_tables.sh blas <label>      # ~12 min on 8 GPUs
bash tools/recommendation/dlrm_v4/record_stack_tables.sh shim /tmp/shim
mkdir -p /tmp/coload
LD_PRELOAD=/tmp/shim/coload_shim.so COLOAD_LOG_DIR=/tmp/coload <one full training run>
bash tools/recommendation/dlrm_v4/record_stack_tables.sh hip <label> /tmp/coload <that run's mlperf.log>
```

Then add the files to `blas_warmup._BLAS_REPLAY_TABLES` / `hip_preload.STACK_TABLES`,
and check the FP8 pins (`ops/blas_solution_pins.py`) for the new hipBLASLt build:
a pin only installs on the stack it lists. A full run's m4 lines
(`[m4] factors ... maxmed=[...]`) show whether any window still stalls.

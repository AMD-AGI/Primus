# DLRMv4: HSTU ranker on Yambda-5b (MLPerf Training `dlrmv4`)

DLRMv4 is the MLPerf Training recommendation benchmark. It trains an HSTU
sequential ranking model on the
[Yambda-5b](https://huggingface.co/datasets/yandex/yambda) music-listening
dataset. The model trains on the data's daily windows in time order. Every 0.1%
of the training data, it is evaluated on a fixed held-out day that training
never sees. The run stops when the eval AUC reaches **0.75**.

In Primus it is the `recommendation` backend with `model_family: dlrm_v4`. The
recipe is [examples/mlperf/dlrm_v4](../../../../examples/mlperf/dlrm_v4/README.md).

- This page: requirements, running, configuration, results, multi-node,
  updating the image and troubleshooting.
- [Benchmark reference](benchmark.md): the dataset, data order, model, loss,
  optimizer, quality target and eval schedule.

## Requirements

| | |
|---|---|
| GPUs | 1, 2 or 4 nodes of 8× AMD Instinct MI355X (`gfx950`). The FP8 and FlyDSL kernels are gfx950-only. |
| Image | `rocm/primus:v26.9` |
| Dataset | 227 GB, on node-local NVMe on every node (training reads it with random page faults) |
| Host memory | The data loaders pin about 180 GiB per node before `run_start`. Each NUMA node must have its share of that (about 22 GiB with 8 NUMA nodes) plus `DL_PIN_PREWARM_HEADROOM_GIB` (default 512 GiB) free, or the run stops before `run_start`. On hosts with less free memory per NUMA node, lower the headroom; the 2- and 4-node results used 128. |
| Container | Host network and IPC, `--ulimit memlock=-1:-1`. `--privileged` is not needed. `run_with_docker.sh` sets all of this. |

At the first launch in a container, the `recommendation` install hook:

- installs `polars-u64-idx`, `gin-config` and the MLPerf logging package, all
  with `--no-deps`. The image's torch, FBGEMM, torchrec, Triton and FlyDSL stay
  untouched, and the hook fails if any of them would change;
- installs FlyDSL 0.2.4 into a private directory,
  `/opt/primus-recommendation/site-packages`, which only the trainer puts first
  on its path. The trainer's kernels also run on the image's FlyDSL 0.3.4.1
  (set `PRIMUS_RECOMMENDATION_SITE=` to empty to use it), but 0.19% slower;
- links `/opt/rocm` to the image's ROCm. The trainer builds HIP kernels at run
  time with `/opt/rocm/bin/hipcc`.

## Quick start

On the host, from the Primus checkout:

```bash
# 1. Download the dataset (227 GB, ready to train) and verify it
export DLRM_DATA_PATH=/data/mlperf_dlrm_v4
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d "$DLRM_DATA_PATH" https://training.mlcommons-storage.org/metadata/dlrmv4_dataset.uri
bash examples/mlperf/dlrm_v4/data/verify_dataset.sh

# 2. If it is on shared storage, copy it to node-local NVMe
DLRM_DATA_SRC=$DLRM_DATA_PATH bash examples/mlperf/dlrm_v4/data/stage_data_local.sh /mnt/nvme/dlrm_data 4086

# 3. One timed MLPerf run on one node
DATADIR=/mnt/nvme/dlrm_data LOGDIR=/mnt/nvme/dlrm_results \
    bash examples/mlperf/dlrm_v4/run_with_docker.sh
```

`run_with_docker.sh` starts a `rocm/primus:v26.9` container and runs
`examples/mlperf/dlrm_v4/run_and_time.sh` in it. Each trial prints
`RESULT,DLRM_V4,<seed>,<seconds>,...` and writes three files to `LOGDIR`: the
MLPerf log (`mlperf_<date>_<n>.log`), the train log (`train_<date>_<n>.log`)
and the console log. The recipe README covers the remaining options: number of
trials, fixed seeds, the compliance checker, short throughput runs, and running
inside an existing container. To build the dataset from the HuggingFace release
instead of downloading it, see
[Benchmark reference: Dataset](benchmark.md#getting-the-data).

To run without the Docker launcher, from inside a container:

```bash
DLRM_DATA_PATH=/data ./primus-cli direct -- train pretrain \
    --config examples/mlperf/dlrm_v4/configs/MI355X/dlrm_v4-yambda5b-mlperf.yaml
```

The trainer starts one process per GPU itself. `primus-cli` therefore runs one
Primus process per node, not a torchrun fan-out.

The first run in a container builds the trainer's HIP extensions under
`/tmp/dlrm_hstu_*` before `run_start`, which makes initialization take 3–7
minutes. Later runs reuse them. At the end of a run, the train log prints
`[Primus:dlrm_v4] MLPerf summary:` with the initialization time (`init_s`), the
time to train (`run_s`), the status, the number of evals and the last AUC.

## Configuration

Settings come from three layers, each overriding the one before it:

1. **The trainer's gin file**
   (`primus/backends/recommendation/dlrm_v4/src/generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin`).
   It sets the model, kernels and precision, and the defaults of the run
   parameters. Each run parameter is read from an environment variable when that
   variable is set.
2. **The Primus config.** Each run parameter in the table below that is set
   (not `null`) is written to its environment variable before the trainer
   starts, replacing any exported value. `null` keeps the gin value.
   All parameters are listed with their meaning in
   [`primus/configs/modules/recommendation/pre_trainer.yaml`](../../../../primus/configs/modules/recommendation/pre_trainer.yaml).
   The MLPerf experiment,
   [`dlrm_v4-yambda5b-mlperf.yaml`](../../../../examples/mlperf/dlrm_v4/configs/MI355X/dlrm_v4-yambda5b-mlperf.yaml),
   fills them from environment variables of the same name, so an exported
   `BATCH_SIZE` still applies.
3. **`gin_bindings`**: a list of gin bindings that the backend appends after the
   gin file. Later gin bindings win, so a binding overrides the file's binding
   for that parameter, environment lookup included. Use it for settings that are
   not exposed as parameters.

| Primus parameter | Environment variable | Default | Meaning |
|---|---|---|---|
| `data_path` | `DLRM_DATA_PATH` | none (`/data` in the recipe) | Dataset root, holding `processed_5b/hstu_cache_L4086/` and `shared_metadata/`. The run stops before training if they are missing. |
| `batch_size` | `BATCH_SIZE` | 1024 | Batch per GPU; the global batch is `BATCH_SIZE × GPUs` |
| `dense_lr`, `sparse_lr` | `DENSE_LR`, `SPARSE_LR` | 1e-6 | Learning rate after warmup (dense Adam, sparse row-wise Adagrad) |
| `lr_warmup_steps` | `LR_WARMUP_STEPS` | 24000 | Linear warmup from 0 |
| `seed` | `SEED` | -1 | -1: a new random seed per run |
| `auc_threshold` | `AUC_THRESHOLD` | 0.75 | Stop at the first eval that reaches it; 1.0 trains through every window |
| `eval_every_data_pct` | `EVAL_EVERY_DATA_PCT` | 0.001 | Evaluate every 0.1% of the training data |
| `skip_eval_epoch_pct` | `SKIP_EVAL_EPOCH_PCT` | -1 | -1: first eval by the MLPerf rule for the run's global batch; 0: evaluate from the start |
| `start_ts`, `num_train_ts` | `START_TS`, `NUM_TRAIN_TS` | 0, 299 | First daily window and number of training windows (0–298: all of them) |
| `history_length` | `HISTORY_LENGTH` | 4086 | User history length; selects `hstu_cache_L<n>/` (only 4086 is distributed) |
| `metric_log_freq` | `METRIC_LOG_FREQ` | 0 | Steps between throughput and loss lines in the train log (0: none) |
| `ckpt_path` | `CKPT_PATH` | none | Directory for checkpoints; unset disables them (see below) |
| `clear_caches` | | `false` (`true` in the recipe) | Evict the dataset from the page cache before INIT and verify it (MLPerf rule) |
| `warm_cache_after_run_start` | | `false` (`true` in the recipe) | Stream each node's dataset copy back into the page cache once that node's ranks pass `run_start` |
| `mlperf_log_path` | `MLPERF_LOG_PATH` | `<output_dir>/mlperf_dlrm_v4.log` | MLPerf log (`/results/mlperf_<date>_<n>.log` with `run_with_docker.sh`) |
| `output_dir` | | `<workspace>/<work_group>/<user_name>/<exp_name>/dlrm_v4` | Generated gin file and default MLPerf log |

In the MLPerf experiment, `clear_caches`, `warm_cache_after_run_start` and
`output_dir` are filled from `CLEAR_CACHES`, `WARM_CACHE_AFTER_RUN_START` and
`DLRM_OUTPUT_DIR`. Other environment variables reach the trainer through the
experiment's `env:` block, or through `EXTRA_ENV="K=V ..."` with
`run_with_docker.sh`.

**MLPerf rules.** For a submission, only the global batch size and the
post-warmup learning rate may be tuned. The reference points are 1e-6 at a
global batch of 8192 (1 node), 2e-6 at 16384 and 4e-6 at 32768. Warmup (24000
steps from 0) and the eval schedule are fixed. See
[Benchmark reference: Quality](benchmark.md#quality).

### Precision

Training is bf16 mixed precision. On top of it, the recipe runs most HSTU GEMMs
in FP8, and this is **on by default**:

- **Forward:** the HSTU projections and attention use `e4m3` operands. Their
  per-tensor scales are calibrated over the first 4 steps and then frozen.
- **Backward:** the attention backward GEMMs, the output projection's input
  gradient, and the UVQK projection's input and weight gradients. Gradients are
  `e5m2` or block-scaled `e4m3`; weights and activations are `e4m3`. Backward
  scales are recomputed during training, not frozen.
- **Preprocessor:** the second-layer GEMMs of the contextual preprocessor MLPs.

FP8 is folded into the kernels that produce each tensor (LayerNorm, projection,
SiLU, residual add), never added as a separate cast. GEMMs accumulate in fp32.
Embeddings, the optimizer, the loss, the metrics and the collectives do not use
FP8.

The FP8 paths **fail closed**: if one cannot run, the run stops instead of
silently falling back to bf16. The train log shows `[hstu-fp8] resolved:` and
`[hstu-fp8] engaged:` lines for the forward paths in use, and `armed OK:` lines
for the other optional paths.

The [Results](#results) are with FP8 on, and the recipe is validated only in
that configuration. Each FP8 path has its own off switch, documented with the
path in the gin file. The trainer refuses an FP8 backward with a bf16 forward:
setting only `DLRM_HSTU_FP8_FWD=0` stops at startup, and the error lists the
other switches to set. They are `DLRM_HSTU_FP8_BWD=0`,
`DLRM_HSTU_FP8_BWD_UVQK=0` and `DLRM_HSTU_FP8_PREPROC_L2=0`.

### Checkpointing

Checkpointing is off unless `ckpt_path` (`CKPT_PATH`) is set. A checkpoint
holds the full sharded model and is hundreds of GB. With the path set:

- the run saves after every 100th window, after the last window, and every
  hour (`CKPT_TIME_INTERVAL_S`, default 3600). `CKPT_STEP_FREQ` and
  `IN_WINDOW_CKPT_FREQ` add step-based saves;
- it keeps the newest `KEEP_LAST_N` checkpoints (default 1);
- on start, it resumes from the highest-numbered checkpoint under the path.

The MLPerf run state is saved too, so the MLPerf log continues across a resume.
The MLPerf recipe leaves checkpointing off.

## Results

One MI355X node, `rocm/primus:v26.9`, the MLPerf recipe (global batch 8192, LR
1e-6), measured 2026-11-06:

| | |
|---|---|
| Throughput, median `global_sps` over steps 20–200 | 84,506–84,533 samples/s |
| Time to target, seed 731542219 | 913.3 s, 5 evals |
| Time to target, seed 1234567 | 955.1 s, 6 evals |
| Time to target, seed 20261103 | 1041.0 s, 8 evals |
| MLPerf 6.1.0 compliance checker | SUCCESS on every run |

Seed 20261103's 7th eval lands within 0.0003 of the target, so this seed needs
7 or 8 evals from run to run (another run: 1000.2 s, 7 evals).

Time to target is `run_start` to `run_stop`, which excludes initialization.
Each eval takes about 13 s.

## Multi-node

The recipe has validated 2- and 4-node configurations:
`examples/mlperf/dlrm_v4/config_MI355X_2x8x1.sh` and `config_MI355X_4x8x1.sh`. Each
one sources the 1-node config and then changes what depends on the node count:

| Nodes | Config | Global batch | `DENSE_LR` / `SPARSE_LR` | `TORCH_NCCL_HIGH_PRIORITY` |
|---|---|---|---|---|
| 1 | `config_MI355X_1x8x1.sh` | 8192 | 1e-6 | 1 (Primus' default) |
| 2 | `config_MI355X_2x8x1.sh` | 16384 | 2e-6 | 0 |
| 4 | `config_MI355X_4x8x1.sh` | 32768 | 4e-6 | 0 |

`SKIP_EVAL_EPOCH_PCT` stays unset, so the first eval follows the global batch.

**Running.** `run_with_docker.sh` launches one node. For 2 or 4 nodes, start one
task per node, each in a `rocm/primus:v26.9` container with the Primus checkout
at `/workspace/Primus` and the dataset on the node's local NVMe. In each task:

```bash
source examples/mlperf/dlrm_v4/config_MI355X_2x8x1.sh   # or config_MI355X_4x8x1.sh
bash examples/mlperf/dlrm_v4/run_and_time.sh
```

- **Node layout.** Under Slurm, `run_and_time.sh` and `primus-cli` take `NNODES`
  and `NODE_RANK` from `SLURM_*`. Inside a container without `scontrol`, pass
  `MASTER_ADDR` in as well. Without Slurm, set `NODE_RANK` and `MASTER_ADDR` (the
  first node) on each node before sourcing the config. Values set before
  sourcing are kept.
- **CPUs.** Give each task all of the node's CPUs (`srun --ntasks-per-node=1
  --cpus-per-task=$SLURM_CPUS_ON_NODE --cpu-bind=none`). Otherwise a Slurm step
  can get one CPU per task, and RCCL runs far below line rate.
- **Network.** Multi-node runs over RDMA (RoCE or InfiniBand), with the usual
  RCCL settings: the routable network interface for the bootstrap
  (`NCCL_SOCKET_IFNAME`), the RDMA devices (`NCCL_IB_HCA`), and your fabric's GID
  index and traffic class. See
  [Multi-node networking](../../multi-node-networking.md). The container needs
  `/dev/infiniband` and `--ulimit memlock=-1:-1`.

**Why `TORCH_NCCL_HIGH_PRIORITY=0` at 2+ nodes.** Under the HIP in v26.9, all
streams share one pool of `GPU_MAX_HW_QUEUES` hardware queues per GPU. With 1,
the NCCL stream is pinned to the same queue as the train pipeline's data-dist
stream. Every step, a small copy of the next batch's all-to-all split sizes then
waits behind the backward collectives: the GPU idles 4–6 ms per step, and time to
target rises by about 3%. With 0, the streams land on separate queues. On one node,
1 shows no stall and stays. The full analysis is in the comment above the
setting in `config_MI355X_2x8x1.sh`.

**Results** (OLCF Lux, MI355X, 8 × 400 Gb/s RoCE per node, `rocm/primus:v26.9`,
2 runs per seed, run with `DL_PIN_PREWARM_HEADROOM_GIB=128`):

| Nodes | Seed 731542219 | Seed 1234567 | Seed 20261103 |
|---|---|---|---|
| 2 | 577.6–579.9 s, 6 evals | 626.2–626.9 s, 8 evals | 638.4–639.4 s, 9 evals |
| 4 | 376.3–376.7 s, 5 evals | 426.7–427.3 s, 9 evals | 436.8–437.1 s, 10 evals |

Every run passed the MLPerf 6.1.0 compliance checker. These times are 1.7% (2
nodes) and 2.1% (4 nodes) below the MLPerf submission version (`training_dlrmv4`
on `rocm/primus:v26.3`) on the same nodes.

## Updating the image

Some of the trainer's startup work is driven by tables recorded for one exact
software stack. On any other stack, the trainer ignores them:

| Table | Purpose | Log line when ignored |
|---|---|---|
| BLAS replay table | Loads hipBLASLt's kernel libraries before `run_start` | `[blas-replay] no table for this stack` |
| HIP preload table | Loads deferred HIP code objects before `run_start` | `[hip-preload] NOT armed:` |
| FP8 GEMM solution pins | Pin the FP8 GEMM kernels some fused paths need (`ops/blas_solution_pins.py`) | |

The Triton kernel replay table is keyed by Triton version instead and covers
Triton 3.6 and 3.9.

Without these tables the run still trains correctly, but slower. Before the
v26.9 tables were recorded, a full run had five 3–6 s stalls, about 19 s in
total, and throughput without the FP8 pins was 0.36% lower.

After moving to a new image, re-record the tables inside the new image, on one
node:

```bash
bash tools/recommendation/dlrm_v4/record_stack_tables.sh blas <label>      # ~12 min on 8 GPUs
bash tools/recommendation/dlrm_v4/record_stack_tables.sh shim /tmp/shim
mkdir -p /tmp/coload
LD_PRELOAD=/tmp/shim/coload_shim.so COLOAD_LOG_DIR=/tmp/coload <one full training run>
bash tools/recommendation/dlrm_v4/record_stack_tables.sh hip <label> /tmp/coload <that run's MLPerf log>
```

Then register the new files and the new hipBLASLt build in the trainer
package:

- add the table files to `_BLAS_REPLAY_TABLES` in
  `dlrm_v4/train/blas_warmup.py` and to `STACK_TABLES` in
  `dlrm_v4/train/hip_preload.py`;
- add the hipBLASLt build to the FP8 pins in `ops/blas_solution_pins.py`. A pin
  applies only on the stacks it lists.

These are edits to the trainer package, so run its tests afterwards. See the
[backend README](../../../../primus/backends/recommendation/README.md#origin).
To find windows that still stall, read the `[m4] factors ... maxmed=` lines of a
full run: `maxmed` is a window's slowest step divided by its median step.

Use a fresh container after an image change. The HIP extension builds cached
in `/tmp` are reused whenever the sources match, even on a different stack.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `dataset not ready (see the recipe README ...)` before training | `DLRM_DATA_PATH` (in the container: the mount) must hold `processed_5b/hstu_cache_L4086/` and `shared_metadata/{album,artist}_item_mapping.parquet`. |
| `[pin-prewarm] NUMA node(s) ... have too little FREE memory` before `run_start` | The run first reclaims page cache by itself; what remains is held by processes. Free that memory, or lower `DL_PIN_PREWARM_HEADROOM_GIB` (default 512; the 2- and 4-node results used 128 on hosts with 300–350 GiB free per NUMA node). `DL_PIN_PREWARM=0` skips the pre-grow; the pools then grow in the timed region (about 3 s). |
| All ranks stall at random, worse with node uptime | Transparent huge-page compaction. Keep `NUMPY_MADVISE_HUGEPAGE=0`, which the recipe and the backend set. |
| Slow steps throughout, high I/O wait | The data is on network storage, or the page cache is cold. Stage the data to local NVMe and keep `WARM_CACHE_AFTER_RUN_START=1`. Without `--privileged`, `CLEAR_CACHES=1` evicts only the dataset's own pages and checks that less than 1% stays resident. `PRIVILEGED=1` also drops the global page cache. |
| A few windows have a step 16–28× the median | The per-stack tables do not match the image. See [Updating the image](#updating-the-image). |
| `FlyDSLUnavailableError` | No usable FlyDSL. The kernels need FlyDSL 0.2 or ≥ 0.3. The train log line `[Primus:dlrm_v4] PYTHONPATH front: ...; flydsl from ...` shows which copy was loaded. Check that the install hook ran. |
| One node: throughput about 5% below the results | Primus' default `GPU_MAX_HW_QUEUES=2` with `TORCH_NCCL_HIGH_PRIORITY=0`. Keep the experiment's `GPU_MAX_HW_QUEUES=4`, and `TORCH_NCCL_HIGH_PRIORITY=1` on one node. |
| 2+ nodes: every step slow, time to target about 3% high | `TORCH_NCCL_HIGH_PRIORITY=1` (see [Multi-node](#multi-node)). Use the 2- or 4-node config, which sets 0. The runner's env dump shows the value. The median of rank 0's `[boundary] train_ts=N train first-batch data-wait=` lines is 27–37 ms when healthy, and 106–119 ms with the stall. |
| A CUDA or CPU torch replaced the image's torch | Something was installed with dependencies. Never reinstall torch, FBGEMM or torchrec; install extra packages with `--no-deps`. |
| Triton cannot find `libamdhip64` | Do not set `TRITON_LIBHIP_PATH`. Triton needs a library name ending in `.so`, and the image ships `.so.7`. |

## For developers

[`primus/backends/recommendation/README.md`](../../../../primus/backends/recommendation/README.md)
describes the backend layout, the trainer's origin (the MLPerf reference fork
`training_dlrmv4`) and the trainer's test suite.

---

[← Recommendation models](../README.md)

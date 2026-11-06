# DLRMv4 MLPerf Training (HSTU on yambda-5b)

MLPerf Training `dlrmv4` on one MI355X node (8 GPUs) with Primus: an HSTU
sequential ranking model trained in streaming order on the
[Yambda-5b](https://huggingface.co/datasets/yandex/yambda) music dataset until the
eval AUC reaches 0.75.

## Setup

### Configuration

- **Model**: HSTU ranker (generative recommenders), FP8 + BF16, Triton / FlyDSL / HIP kernels
- **Training**: streaming train-on-window-T / eval-on-T+1, global batch 8192 (1024 per GPU), LR 1e-6, 24k-step warmup
- **Eval**: the full eval set every 0.1% of the data, first eval by the MLPerf rule; target AUC 0.75
- **Backend**: `recommendation`, model family `dlrm_v4`
  (`primus/backends/recommendation/`). The model, kernel and precision settings
  are the trainer's gin file
  `primus/backends/recommendation/dlrm_v4/src/generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin`.
  Benchmark description and engineering notes:
  [docs/04-technical-guides/recommendation](../../../docs/04-technical-guides/recommendation/README.md).
- **Image**: `rocm/primus:v26.9`. The `recommendation` hook adds `polars-u64-idx`,
  `gin-config` and the MLPerf logging package (all `--no-deps`) and links
  `/opt/rocm` to the image's ROCm, at the first launch in a container.

### Data

227 GB, ready to train (no preprocessing), from MLCommons:

```bash
export DLRM_DATA_PATH=/data/mlperf_dlrm_v4
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d "$DLRM_DATA_PATH" https://training.mlcommons-storage.org/metadata/dlrmv4_dataset.uri
DLRM_DATA_PATH=$DLRM_DATA_PATH bash examples/mlperf/dlrm_v4/data/verify_dataset.sh
```

The trainer reads `processed_5b/hstu_cache_L4086/` and `shared_metadata/`.
Keep the dataset on node-local NVMe: training reads it with random page faults.
To copy it from shared storage to a node:
`DLRM_DATA_SRC=/shared/mlperf_dlrm_v4 bash examples/mlperf/dlrm_v4/data/stage_data_local.sh /mnt/nvme/dlrm_data 4086`.
Building the dataset from the HuggingFace release instead:
`examples/mlperf/dlrm_v4/data/download_dataset.sh` (see the benchmark page).

## Run with Docker (recommended)

```bash
cd /path/to/Primus

export DATADIR=/mnt/nvme/dlrm_data      # mounted read-only at /data
export LOGDIR=/mnt/nvme/dlrm_results    # mounted at /results

# Optional; these are the defaults.
export CONT=rocm/primus:v26.9
export DGXSYSTEM=MI355X_1x8x1
export NEXP=1
# export SEED=731542219                 # default: a random seed per trial
# export CHECK_COMPLIANCE=1             # MLPerf compliance checker per trial

bash examples/mlperf/dlrm_v4/run_with_docker.sh
```

Each trial writes `LOGDIR/mlperf_<date>_<n>.log` (the MLPerf log),
`LOGDIR/train_<date>_<n>.log` and the console log `LOGDIR/<date>_<n>.log`, and
prints `RESULT,DLRM_V4,<seed>,<seconds>,...`.

The container gets no `--privileged` by default (some clusters forbid it). The
cache flush then evicts the dataset's own pages and verifies with `mincore`
that less than 1% is resident; `PRIVILEGED=1` also drops the global page cache.

### Shorter runs

The run shape comes from environment variables (`EXTRA_ENV` passes them in):

```bash
# 200-step throughput check (median global_sps over steps 20-200 in the train log)
EXTRA_ENV="NUM_TRAIN_TS=31 METRIC_LOG_FREQ=20 CLEAR_CACHES=0" bash examples/mlperf/dlrm_v4/run_with_docker.sh
# a few windows with early evals
EXTRA_ENV="NUM_TRAIN_TS=4 METRIC_LOG_FREQ=20 SKIP_EVAL_EPOCH_PCT=0 EVAL_EVERY_DATA_PCT=0.00005 CLEAR_CACHES=0" \
    bash examples/mlperf/dlrm_v4/run_with_docker.sh
```

## Run inside an existing container

```bash
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    -v /path/to/Primus:/workspace/Primus -v /mnt/nvme/dlrm_data:/data:ro -v /mnt/nvme/dlrm_results:/results \
    rocm/primus:v26.9

cd /workspace/Primus
source examples/mlperf/dlrm_v4/config_MI355X_1x8x1.sh
bash examples/mlperf/dlrm_v4/run_and_time.sh
```

or directly: `./primus-cli direct -- train pretrain --config examples/mlperf/dlrm_v4/configs/MI355X/dlrm_v4-yambda5b-mlperf.yaml`.

### Key files

- `configs/MI355X/dlrm_v4-yambda5b-mlperf.yaml`: the experiment (backend `dlrm_v4`; run shape from the environment)
- `config_MI355X_1x8x1.sh`: system configuration and environment
- `run_and_time.sh`: timed run inside the container
- `run_with_docker.sh`: host launcher (container, trials, compliance check)
- `data/`: dataset download, verification and staging scripts, checksums
- `primus/configs/modules/recommendation/pre_trainer.yaml`: every backend parameter, with its meaning

## Notes

- **Process model**: the trainer starts one process per GPU itself, so
  `primus-cli direct` runs one Primus process per node (the `recommendation`
  prepare hook selects single mode), not a torchrun fan-out.
- **Page cache (C1)**: with `CLEAR_CACHES=1` the dataset is evicted and verified
  before INIT; with `WARM_CACHE_AFTER_RUN_START=1` a background reader streams it
  back into the page cache once `run_start` is logged, so all of that I/O is in
  the timed region. Cold random reads cannot feed 8 MI355X GPUs from
  network-attached NVMe. Disclose the in-run reader in a submission.
- **Transparent huge pages**: `NUMPY_MADVISE_HUGEPAGE=0` (set by the config and
  the backend) keeps numpy from asking for huge pages; with THP `defrag=madvise`
  a fragmented host otherwise stalls all ranks at once.
- **Image stack**: never reinstall torch, FBGEMM or torchrec, and install
  extras with `--no-deps`; otherwise pip pulls a CUDA / CPU torch over the ROCm
  build. Stock `polars` silently overflows on yambda-5b: the hook installs
  `polars-u64-idx`. Do not set `TRITON_LIBHIP_PATH` (Triton needs a name ending
  in `.so`; TheRock ships `.so.7`).
- **First run in a container** builds the trainer's HIP extensions under
  `/tmp/dlrm_hstu_*` before `run_start` (init ~3-7 min); later runs reuse them.
  Use a fresh container (or `/tmp`) after changing the image: builds are reused
  when the sources match, even across stacks.
- **Per-stack tables**: the Triton spec replay, BLAS replay and HIP preload
  tables in the trainer are recorded for one software stack and turn themselves
  off on another (first-use compiles and library loads then happen inside the
  timed region). Re-record them after an image change
  (`tools/recommendation/dlrm_v4/gen_blas_replay_table.py`,
  `gen_hip_preload_table.py`).
- **Multi-node**: the trainer supports it (`NNODES`, `NODE_RANK`, `MASTER_ADDR`,
  `MASTER_PORT`; MLPerf RCP LR 2e-6 at 2 nodes, 4e-6 at 4 nodes); this recipe has
  been validated on one node.

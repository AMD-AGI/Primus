# FLUX.1-Schnell MLPerf on MI355X (Primus)

MLPerf Training Flux.1-Schnell on **MI355X** via the in-tree diffusion backend and `primus-cli`.

Precision defaults are in `config_common.sh`: MXFP4 `dgrad,wgrad` with A6W4 forward, fused norm/RoPE, FlyDSL attention, local AdamW, and the SDMA all-gather. The 6.1 disclosure defaults are mxfp4 linear, bfloat16 attention, and mxfp4 communication. Target validation loss is `0.586`. An evaluation runs every 262,144 training samples.

## Key files

- `flux.1_schnell_t2i-pretrain.yaml` — pretrain config (`EXP` inside the container)
- `config_common.sh` — shared Flux flags
- `config_MI355X_1x8x32.sh` — 1 node
- `config_MI355X_4x8x32.sh` — 4 nodes
- `config_MI355X_8x8x32.sh` — 8 nodes
- `run_with_docker.sh` — single-node host launcher (N timed trials)
- `run_with_docker_slurm.sh` — one container per node, used by `job.sh`
- `run_and_time.sh` — timed run inside the container
- `job.sh` — `sbatch` wrapper for the multinode launcher
- `runtime_tunables.sh` — host CPU/THP/cache tuning
- `scripts/download_ref_data.sh` — optional download of the preprocessed dataset
- `scripts/stage_dataset_local.sh` — copy the dataset onto node-local disk

## Configurations

`DGXSYSTEM` selects `config_${DGXSYSTEM}.sh`. Each file sets the node count and the global batch, then sources `config_common.sh`.

| Config | Nodes × GPUs | Local batch | Grad accum | Global batch | LR | Eval every |
|--------|-------------:|------------:|-----------:|-------------:|----|-----------:|
| `config_MI355X_1x8x32.sh` | 1 × 8 | 32 | 2 | 512 | 0.0002 | 512 steps |
| `config_MI355X_4x8x32.sh` | 4 × 8 | 32 | 1 | 1024 | 0.00025 | 256 steps |
| `config_MI355X_8x8x32.sh` | 8 × 8 | 32 | 1 | 2048 | 0.00027 | 128 steps |

Eval step count is `262144 / global batch`. The 4-node config switches the forward to FP8 at step 7325. The 8-node config switches at step 4150 and defaults `DP_REPLICATE=2`.

## Data

`DATADIR` must contain the three preprocessed directories:

```text
$DATADIR/
├── cc12m_preprocessed/
├── coco_preprocessed/
└── empty_encodings/
```

Inside the container they are `/dataset/cc12m_preprocessed`, `/dataset/coco_preprocessed`, and `/dataset/empty_encodings`.

Downloading is optional when those directories are already on shared storage. The preprocessed set is about 2.5 TB. To fetch it once:

```bash
cd examples/mlperf/flux1
PREPROCESSED_PATH=/path/to/flux1/data ./scripts/download_ref_data.sh
```

Point `DATADIR` at that path. Preprocessing details are in the [MLPerf text-to-image instructions](https://github.com/mlcommons/training/tree/master/text_to_image#preprocessing).

## Single node

From the Primus repo root, on the compute node:

```bash
export CONT=rocm/primus:v26.9
export DGXSYSTEM=MI355X_1x8x32
export DATADIR=/path/to/flux1/data
export LOGDIR=/path/to/results
export NEXP=1
bash examples/mlperf/flux1/run_with_docker.sh
```

`NEXP=10` is the submission trial count. A short smoke test:

```bash
export MAX_STEPS=6
export NEXP=1
bash examples/mlperf/flux1/run_with_docker.sh
```

Each trial writes `${LOGDIR}/<DATESTAMP>_<N>.log`. The MLPerf log is `${LOGDIR}/mlperf_logging.out`.

| Host | Container |
|------|-----------|
| this example directory | `/workspace/code` |
| `DATADIR` | `/dataset` |
| `LOGDIR` | `/results` |

## Multiple nodes

Stage the dataset onto each node's local disk, then submit one task per node. `job.sh` checks that the allocation size matches `NNODES` from the sourced config.

```bash
cd examples/mlperf/flux1
export CONT=rocm/primus:v26.9
export DATADIR=/path/to/flux1/data
export LOGDIR=/path/to/results
source config_MI355X_4x8x32.sh
sbatch --nodes=$NNODES job.sh
```

Eight nodes is the same command after `source config_MI355X_8x8x32.sh`.

From inside an existing allocation, skip `sbatch` and run the per-node script directly:

```bash
export CONT=rocm/primus:v26.9
export DGXSYSTEM=MI355X_4x8x32
export DATADIR=/path/to/flux1/data
export LOGDIR=/path/to/results
srun --ntasks-per-node=1 bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

`run_with_docker_slurm.sh` starts one container per node. `torchrun` inside `primus-cli` creates the eight GPU workers. Rank 0 is the first host in the allocation. Set `FLUX_STAGE_LOCAL=0` to read `DATADIR` directly instead of the node-local copy.

## Inside an existing container

```bash
export DGXSYSTEM=MI355X_1x8x32
export SEED=10007
source /workspace/code/config_${DGXSYSTEM}.sh
bash /workspace/code/run_and_time.sh
```

`run_and_time.sh` prints `RESULT,FLUX1,,<seconds>,AMD,<start time>` when the trial finishes.

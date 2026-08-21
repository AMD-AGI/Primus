# FLUX.1-Schnell FP8 MLPerf Training

This recipe runs FLUX.1-Schnell on MI355X GPUs with the in-tree `diffusion`
backend. The defaults enable tensorwise FP8 and target the MLPerf validation
loss threshold of `0.586`.

## Docker image

The launch scripts use `zirui3/primus-v26.3-flux:v0.4` by default. This image
contains the selective FlyDSL and natural-backward optimizations from
`docker/flux-fp8/Dockerfile.v26.3-natural-bwd`.

```bash
docker pull zirui3/primus-v26.3-flux:v0.4
```

## Prepare data

Approximately 2.5 TB of storage is required. Download the preprocessed MLPerf
TorchTitan datasets:

```bash
mkdir -p /path/to/data
cd /path/to/data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri
```

The resulting data root must contain:

```text
/path/to/data/
├── cc12m_preprocessed/
├── coco_preprocessed/
└── empty_encodings/
```

See the [MLPerf FLUX preprocessing instructions](https://github.com/mlcommons/training/tree/master/text_to_image#preprocessing)
for details.

## Run on one node

`OUTPUT_ROOT` must be writable from the compute node. A full `dtcp_full`
checkpoint is approximately 93 GB, so use shared storage with enough space.

```bash
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker.sh
```

For a short training smoke test without saving a checkpoint:

```bash
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/output \
MAX_STEPS=1 \
SAVE_STRATEGY=none \
COMPILE_TRANSFORMER_BLOCKS=false \
bash examples/mlperf/flux1/run_with_docker.sh
```

## Run on four Slurm or Spur nodes

Submit one four-node allocation; the script uses one `srun` task per node and
starts eight training processes in each container. Set `NODELIST_ARG` to
`--nodelist=node1,node2,node3,node4` only when specific idle nodes are needed.

```bash
REPO=/shared_nfs/zirui/code/primus-compile
DATA_ROOT=/shared_nfs/zirui/data
OUTPUT_ROOT=/shared_nfs/zirui/runs/flux-fp8-4n
NODELIST_ARG=

sbatch -A amd-spur -p amd-spur --qos=amd-spur-qos \
  -N4 --ntasks-per-node=1 --exclusive --gpus-per-node=8 -t 04:00:00 \
  $NODELIST_ARG \
  --output="$OUTPUT_ROOT-%j.slurm.log" \
  --wrap="cd '$REPO' && env DATA_ROOT='$DATA_ROOT' OUTPUT_ROOT='$OUTPUT_ROOT'-\$SLURM_JOB_ID \
    DOCKER_IMAGE=zirui3/primus-v26.3-flux:v0.4 \
    DP_REPLICATE=4 LOCAL_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=1 \
    LR=0.00025 WARMUP_STEPS=800 GRADIENT_CHECKPOINTING_RATIO=0 \
    FSDP2_REDUCE_DTYPE=bf16 FLUX_FP8_GEMM_BACKEND=selective_flydsl \
    TORCHINDUCTOR_BENCHMARK_FUSION=1 PRIMUS_FLUX_AITER_ATOMIC_FP32=0 \
    PRIMUS_FLUX_REUSE_FP8_INPUT=1 \
    bash examples/mlperf/flux1/run_with_docker_slurm.sh"
```

This gives 32 GPUs, HSDP `dp_replicate=4`, `dp_shard=8`, MBS 32, GA 1, and
GBS 1024. Verify that the selected account and QOS are allowed before
submitting. For nodes split across independent allocations, follow
[`luanch-multi-nodes.md`](../../../luanch-multi-nodes.md).

From an existing four-node allocation, run only the command wrapped above:

```bash
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
DP_REPLICATE=4 LOCAL_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=1 \
LR=0.00025 WARMUP_STEPS=800 GRADIENT_CHECKPOINTING_RATIO=0 \
FSDP2_REDUCE_DTYPE=bf16 FLUX_FP8_GEMM_BACKEND=selective_flydsl \
TORCHINDUCTOR_BENCHMARK_FUSION=1 PRIMUS_FLUX_AITER_ATOMIC_FP32=0 \
PRIMUS_FLUX_REUSE_FP8_INPUT=1 \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

Common overrides include `GPUS_PER_NODE`, `MAX_STEPS`, `SEED`, `MASTER_PORT`,
`SAVE_STRATEGY`, `SAVE_STEPS`, `RESUME_FROM_CHECKPOINT`, and
`MLPERF_CLEAR_CACHES=false`.

## Files

```text
examples/mlperf/flux1/
├── README.md
├── flux.1_schnell_t2i-pretrain.yaml
├── requirements.txt
├── run_with_docker.sh
└── run_with_docker_slurm.sh
```

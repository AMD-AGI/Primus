# FLUX.1-Schnell FP8 MLPerf Training

This recipe runs FLUX.1-Schnell on MI355X GPUs with the in-tree `diffusion`
backend. The defaults enable tensorwise FP8 and target the MLPerf validation
loss threshold of `0.586`.

## Docker image

The launch scripts use `zirui3/primus-v26.3-flux:v0.1` by default. Pull it
before starting training:

```bash
docker pull zirui3/primus-v26.3-flux:v0.1
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

## Run with Slurm or Spur

From an existing allocation, launch one Docker container per node and one
training process per GPU. Spur exposes the Slurm environment and `srun`
interface used by this script, so both schedulers use the same entry point:

```bash
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker_slurm.sh
```

Common overrides include `GPUS_PER_NODE`, `MAX_STEPS`, `LOCAL_BATCH_SIZE`,
`SEED`, `MASTER_PORT`, `SAVE_STRATEGY`, `SAVE_STEPS`,
`RESUME_FROM_CHECKPOINT`, and `MLPERF_CLEAR_CACHES=false`.

## Files

```text
examples/mlperf/flux1/
├── Dockerfile
├── README.md
├── flux.1_schnell_t2i-pretrain.yaml
├── requirements.txt
├── run_with_docker.sh
└── run_with_docker_slurm.sh
```

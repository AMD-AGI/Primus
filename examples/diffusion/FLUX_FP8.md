# FLUX.1-Schnell FP8 MLPerf Training

This recipe targets the `feat/zirui/flux-fp8` baseline on one node with eight
MI355X GPUs.

## Environment

Docker image: `zirui3/primus-v26.3-flux:v0.1`

## Data

Download the preprocessed embeddings published for the MLPerf TorchTitan
reference. Approximately 2.5 TB of storage is required.

```bash
mkdir -p /path/to/data
cd /path/to/data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri
```

See the [MLPerf FLUX preprocessing instructions](https://github.com/mlcommons/training/tree/master/text_to_image#preprocessing).

## Launch

Run from the repository root:

```bash
DATA_ROOT=/path/to/data
OUTPUT_ROOT=/path/to/output

docker run --rm --init --privileged \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host --shm-size=20G \
  -v "$PWD:/workspace/Primus" \
  -v "$DATA_ROOT:/data" \
  -v "$OUTPUT_ROOT:/output" \
  -w /workspace/Primus \
  zirui3/primus-v26.3-flux:v0.1 \
  bash examples/diffusion/run_flux_mlperf.sh
```

Override launcher defaults such as `MAX_STEPS`, `LOCAL_BATCH_SIZE`, or `SEED`
with `docker run --env NAME=value`.

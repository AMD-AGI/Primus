# FLUX.1-Schnell FP8 MLPerf Training

This recipe selects the legacy MBS32 FP8 profile on one node with eight MI355X
GPUs.

## Environment

Use the [maintained runtime instructions](../mlperf/flux1/README.md#runtime-image)
for the pinned image and prerequisites.

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

Use the single maintained entry point documented in
[`examples/mlperf/flux1/README.md`](../mlperf/flux1/README.md):

```bash
FLUX_CONFIG=config_1n_gbs1024.sh \
DATA_ROOT=/path/to/data OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker.sh
```

This explicit profile selects legacy MBS32 FP8 rather than the launcher's
default single-node A6W4 path.

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

Use the single maintained entry point documented in
[`examples/mlperf/flux1/README.md`](../mlperf/flux1/README.md):

```bash
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/output \
bash examples/mlperf/flux1/run_with_docker.sh
```

The natural-backward stack is `zirui3/primus-v26.3-flux:v0.4`, built from
`examples/mlperf/flux1/Dockerfile`. It pins FlyDSL 0.2.4 and Primus-Turbo
`a14521b8`, and enables qualified natural-layout wgrads and forward-input FP8
reuse for the MBS32 recipe.

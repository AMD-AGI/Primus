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

For MBS32 comparison runs, build
`docker/flux-fp8/Dockerfile.v26.3-selective-gemm` and select one backend:

```bash
-e FLUX_FP8_GEMM_BACKEND=selective_triton
-e FLUX_FP8_GEMM_BACKEND=selective_flydsl
```

Both policies replace the two affected contraction forward shapes. The
selective FlyDSL path also handles the MBS32 single-down wgrad directly from
natural-layout operands, avoiding TorchAO's runtime canonical transpose/copy.
An empty value retains the normal TorchAO/Inductor autotuning policy.

`local_runs/run_flux_mlperf.sh` disables activation checkpointing by default
when `LOCAL_BATCH_SIZE=32`; this removes recompute kernels and fits in 151.78 GB
per MI355X in the one-node GA4 proxy. MBS64 retains checkpoint ratio `0.25`.
Set `GRADIENT_CHECKPOINTING_RATIO` explicitly to override either default. The
selective-gemm image also enables MBS32-only Inductor fusion benchmarking and
the faster AITER hd128 backward conversion path; MBS64 retains the prior AITER
accumulation policy.

The production MBS32/MBS64 stack is `zirui3/primus-v26.3-flux:v0.3`, built
from `docker/flux-fp8/Dockerfile.v26.3-combined`. It pins FlyDSL 0.2.4 and
Primus-Turbo commit `a14521b8`, retaining the MBS32 optimizations while adding
the two qualified MBS64 natural-layout TN wgrads. Use
`FLUX_FP8_GEMM_BACKEND=selective_flydsl`; exact-shape dispatch keeps the MBS32
and MBS64 paths independent. `Dockerfile.v26.3-pr453-tn-wgrad` remains available
to reproduce the earlier MBS64-only qualification image.

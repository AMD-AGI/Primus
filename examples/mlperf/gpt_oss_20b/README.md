# GPT-OSS-20B MLPerf pretraining

GPT-OSS 20B on one MI355X node with 8 GPUs and global batch size 32.

## Build

The Dockerfile builds the complete runtime, including the GPT-OSS
Primus-Turbo test branch, TransformerEngine, and the attention ASM kernels.
The v26.5 image reuses the base image's Triton 3.7 compiler; v26.3 upgrades its
older base Triton for compatibility with the same Turbo branch.

```bash
cd examples/mlperf/gpt_oss_20b
docker build --network host \
  -f Dockerfile.runtime-v26.5 \
  -t primus:gpt-oss-20b-mlperf-v26.5 .
```

Use `Dockerfile.runtime-v26.3` and a v26.3 tag for the compatibility stack.
Push a shared tag with `docker push <image>`.

## Data

```bash
mkdir -p /data/gpt_oss_20b
cd /data/gpt_oss_20b
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d data \
  https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
```

Training uses the `c4-train.en_6_text_document` prefix and validation uses
`c4-validation-91205-samples.en_text_document`.

## Run

```bash
docker run -it --rm \
  --privileged --network host --ipc host --shm-size 128g \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  -v /path/to/Primus:/workspace/Primus \
  -v /path/to/data:/data \
  -v /path/to/model:/model \
  -v /path/to/results:/results \
  primus:gpt-oss-20b-mlperf-v26.5 bash
```

Inside the container:

```bash
cd /workspace/Primus/examples/mlperf/gpt_oss_20b
source config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh
./run_and_time.sh
```

That is the complete long-run entry. The config is the single source of
submission defaults: MLPerf trainer, 1.2M iteration ceiling, 128-step warmup,
FP8 Triton grouped GEMM, fused wgrad accumulation, and disabled profiling.
Short diagnostics and backend ablations should override environment variables
outside the checked-in submission config.

## v26.5 attention prewarm

The v26.5 TE stack lazily compiles two attention variants. Starting eight
torchrun ranks against an empty cache can race while writing the same blobs.
`run_and_time.sh` therefore runs `prewarm_attention.py` once before timing; the
helper only populates the sliding-window and full-attention cache entries.
The v26.3 TE/AITER stack does not exhibit this cache race, so the prewarm is
skipped automatically for v26.3.

## Key files

- `Dockerfile.runtime-v26.3`, `Dockerfile.runtime-v26.5`: complete runtime builds
- `config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh`: submission defaults
- `run_and_time.sh`: benchmark entry
- `prewarm_attention.py`: v26.5-only attention cache prewarm

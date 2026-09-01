# GPT-OSS-20B MLPerf Pretraining

MLPerf-compliant GPT-OSS 20B pretraining on one MI355X node (8 GPUs, GBS=32)
using Primus. The layout matches `examples/mlperf/llama3.1_8b`.

## Setup

### Configuration

- **Model**: GPT-OSS 20B (2880 hidden, 24 layers, 64 heads, 32 experts)
- **Training**: 1.2M iteration ceiling, GBS=32, MBS=4, LR=8e-4
- **Default precision**: FP8 + Turbo attention (`gpt_oss_20B-FP8-turbo-attn-mlperf-pretrain.yaml`)
- **Optional precision**: MXFP4 grouped GEMM, QKVO BF16, weight de-oscillation
- **Data**: C4 dataset (tokenized, same as Llama 3.1 8B)

### Data

Download the preprocessed C4 dataset and tokenizer:

```bash
mkdir -p /data/gpt_oss_20b
cd /data/gpt_oss_20b

# data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# model
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

Training uses the `c4-train.en_6_text_document` prefix and validation uses
`c4-validation-91205-samples.en_text_document`.

## Run with Docker (recommended)

Run the launcher from the host. It starts the container, mounts the Primus
checkout and data directories, loads the selected system configuration, and
runs the requested number of experiments.

```bash
cd /path/to/Primus

export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results

# Optional; these are the defaults.
export CONT=rocm/primus:v26.5
export DGXSYSTEM=MI355X_1x8x1
export NEXP=1

# Optional host runtime tunables before each trial (cpupower, THP, drop_caches; see runtime_tunables.sh):
# export RUN_RUNTIME_TUNABLES=1

bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

`DATADIR` must contain the preprocessed C4 dataset. If `MODELDIR` exists and
is nonempty, it is mounted at `/model` and used as the local tokenizer. If a
local tokenizer is unavailable, omit `MODELDIR` (or point it to an empty
directory) and export a Hugging Face token:

```bash
export HF_TOKEN=<your_huggingface_token>
bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

### MXFP4 recipe

Override `EXP` to switch from the default FP8 Turbo-attention yaml:

```bash
export EXP=/workspace/Primus/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml
export MLLOG_LOWEST_NUMERICAL_PRECISION_LINEAR=mxfp4
# Optional scale rounding for Turbo MXFP4 quant: 0=RTE, 1=RZ, 2=stochastic
# export PRIMUS_TURBO_MXFP4_SCALE_ROUNDING=0
bash examples/mlperf/gpt_oss_20b/run_with_docker.sh
```

## Run inside an existing container

### Start Docker Image

```bash
docker run -it     --device /dev/dri     --device /dev/kfd     --device /dev/infiniband     --network host --ipc host     --group-add video     --cap-add SYS_PTRACE     --security-opt seccomp=unconfined     --privileged     -v $HOME:$HOME   --shm-size 128G     --name primus_training_env rocm/primus:v26.5

cd /workspace/Primus
```

### Key Files

- `configs/MI355/gpt_oss_20B-FP8-turbo-attn-mlperf-pretrain.yaml` — default FP8 + Turbo attention
- `configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml` — MXFP4 grouped GEMM, QKVO BF16, de-oscillation
- `configs/MI355/gpt_oss_20B-MXFP4-qkvo-bf16.yaml` — TE precision matcher (not a runnable experiment)
- `config_MI355X_1x8x1.sh` — system config and env vars
- `run_and_time.sh` — timed training entry
- `tune_gemm_results-v26.5.txt` — optional hipBLASLt replay for TE QKV GEMMs

```bash
export HF_TOKEN=<your_huggingface_token>
source config_MI355X_1x8x1.sh
# optional MXFP4:
# export EXP=${PRIMUS_PATH}/examples/mlperf/gpt_oss_20b/configs/MI355/gpt_oss_20B-MXFP4-deosc-mlperf-pretrain.yaml
bash run_and_time.sh
```

## Notes

- `log_interval: 999999` suppresses regular Primus logs
- Grouped GEMM backend is set in the yaml (`turbo_grouped_gemm_backend: flydsl`), not in the config shell
- `RUN_RUNTIME_TUNABLES` defaults to `0`; set `RUN_RUNTIME_TUNABLES=1` to run `runtime_tunables.sh` on the host before each trial (some steps require `sudo`)

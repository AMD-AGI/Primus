# GPT-OSS-20B Pretraining Benchmark

GPT-OSS 20B (Mixture of Experts)


## Setup

### Start Docker Image

```bash
docker run -it     --device /dev/dri     --device /dev/kfd     --device /dev/infiniband     --network host --ipc host     --group-add video     --cap-add SYS_PTRACE     --security-opt seccomp=unconfined     --privileged     -v $HOME:$HOME   --shm-size 128G     --name primus_training_env rocm/primus:v26.5

cd /workspace/Primus
```


### Configuration

This benchmark trains a 20B parameter GPT model with Mixture of Experts (MoE) architecture using the Primus framework on AMD GPUs.

**Key Features:**
- 20B parameter MoE model
- Expert Parallelism (EP=8)
- FP8 hybrid precision training
- Primus Turbo optimizations (DeepEP, sync-free MoE)

## Key Files

- `configs/MI355/gpt_oss_20B-FP8-mlperf-pretrain.yaml` - Model and training config
    - Update `train_data_path` and `train_data_path` to your local downloaded location
- `config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh` - System config and env vars
   - Update `PRIMUS_PATH` to clone Primus Repo
   - Update `EXP`to `<PRIMUS_PATH>/examples/mlperf/configs/MI355/gpt_oss_20B-FP8-mlperf-pretrain.yaml`
- `run_and_time.sh` - Run script

### Data

Download preprocessed C4 dataset:

```bash
mkdir -p /data/gpt_oss_20b
cd /data/gpt_oss_20b

# Download training and validation data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
```

After download, you should see files with the following naming conventions:
- Training: `c4-train.en_6_text_document.bin` and `.idx`
- Validation: `c4-validation-91205-samples.en_text_document.bin` and `.idx`

The data directory is approximately **80 GB** and model directory is approximately **30 GB**.

### How to run 

```bash
export HF_TOKEN=<your_huggingface_token>
source config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh
bash run_and_time.sh
```
## Notes

- `log_interval: 99999999` suppresses regular Primus logs

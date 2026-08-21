# LLama3.1 8B MLPerf Pretraining

MLPerf-compliant LLama3.1 8B pretraining using Primus

## Setup

### Configuration

- **Model**: LLama3.1 8B (4096 hidden, 32 layers, 32 attention heads)
- **Training**: 1.2M iterations, GBS=32, MBS=2, LR=8e-4
- **Precision**: MXFP4
- **Data**: C4 dataset (tokenized)


### Data

Download preprocessed C4 dataset:

```bash
mkdir -p /data/mlperf_llama31_8b
cd /data/mlperf_llama31_8b

# data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# model
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

```

## Run with Docker (recommended)

Run the launcher from the host. It starts the container, mounts the Primus
checkout and data directories, loads the selected system configuration, and
runs the requested number of experiments.

```bash
cd /path/to/Primus

export DATADIR=/data/mlperf_llama31_8b/data
export MODELDIR=/data/mlperf_llama31_8b/model
export LOGDIR=/data/mlperf_llama31_8b/results

# Optional; these are the defaults.
export CONT=rocm/primus:v26.5
export DGXSYSTEM=MI355X_1x8x1
export NEXP=1

bash examples/mlperf/llama3.1_8b/run_with_docker.sh
```

`DATADIR` must contain the preprocessed C4 dataset. If `MODELDIR` exists and
is nonempty, it is mounted at `/model` and used as the local tokenizer. If a
local tokenizer is unavailable, omit `MODELDIR` (or point it to an empty
directory) and export a Hugging Face token:

```bash
export HF_TOKEN=<your_huggingface_token>
bash examples/mlperf/llama3.1_8b/run_with_docker.sh
```


## Run inside an existing container

### Start Docker Image

```bash
docker run -it     --device /dev/dri     --device /dev/kfd     --device /dev/infiniband     --network host --ipc host     --group-add video     --cap-add SYS_PTRACE     --security-opt seccomp=unconfined     --privileged     -v $HOME:$HOME   --shm-size 128G     --name primus_training_env rocm/primus:v26.5

cd /workspace/Primus
```

### Key Files

- `configs/MI355X/llama3.1_8B-pretrain-FP4.yaml` - Model and training config
    - Update `train_data_path` and `train_data_path` to your local downloaded location
- `config_MI355X_1x8x1.sh` - System config and env vars
   - Update `PRIMUS_PATH` to clone Primus Repo
   - Update `EXP`to `<PRIMUS_PATH>/examples/mlperf/configs/MI355X/llama3.1_8B-pretrain-FP4.yaml`
- `run_and_time.sh` - Run script

```bash
export HF_TOKEN=<your_huggingface_token>
source config_MI355X_1x8x1.sh
bash run_and_time.sh
```
## Notes

- `log_interval: 99999999` suppresses regular Primus logs

# LLama3.1 8B MLPerf Pretraining

MLPerf-compliant LLama3.1 8B pretraining using Primus

## Setup

### Start Docker Image

```bash
export MLPERF_PAT=<your_github_pat>
docker run -it     --device /dev/dri     --device /dev/kfd     --device /dev/infiniband     --network host --ipc host     --group-add video     --cap-add SYS_PTRACE     --security-opt seccomp=unconfined     --privileged     -v $HOME:$HOME   --shm-size 128G     --name primus_training_env rocm/primus:v26.2


git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
```


### Configuration

- **Model**: LLama3.1 8B (4096 hidden, 32 layers, 32 attention heads)
- **Training**: 1.2M iterations, GBS=32, MBS=2, LR=8e-4
- **Precision**: FP8 hybrid
- **Data**: C4 dataset (tokenized)

## Key Files

- `configs/MI355X/llama3.1_8B-pretrain.yaml` - Model and training config
    - Update `train_data_path` and `train_data_path` to your local downloaded location
- `config_MI355X_1x8x1.sh` - System config and env vars
   - Update `PRIMUS_PATH` to clone Primus Repo
   - Update `EXP`to `<PRIMUS_PATH>/examples/mlperf/configs/MI355X/llama3.1_8B-pretrain-FP8.yaml`
- `src/train.py` - Training entry point
- `run_and_time.sh` - Run script

### Data

Download preprocessed C4 dataset:

```bash
cd /data/mlperf_llama31_8b
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
```

### How to run 

```bash
export HF_TOKEN=<your_huggingface_token>
source config_MI355X_1x8x1.sh
bash run_and_time.sh
```
## Notes

- `log_interval: 99999999` suppresses regular Primus logs

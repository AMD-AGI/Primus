# LLama3.1 8B MLPerf Pretraining

MLPerf-compliant LLama3.1 8B pretraining using Primus

## Setup

### Configuration

- **Model**: LLama3.1 8B (4096 hidden, 32 layers, 32 attention heads)
- **Training**: 1.2M iterations, GBS=32, MBS=4, LR=7.6e-4
- **Precision**: MXFP4
- **Data**: C4 dataset (tokenized)
- **Kernels**: Primus-Turbo FlyDSL attention, MXFP4 GEMMs and fused dense MLP, Turbo RMSNorm,
  FlyDSL fused QKV RoPE, Turbo fused cross entropy and TE FusedAdam

On 8x MI355X with `rocm/primus:v26.9` and Primus-Turbo `fda6df0b`, this recipe reaches eval
loss 3.289 at 196,608 samples, the MLPerf v6.0 submission's sample count, in 78.0 minutes
(761 ms per step).

### Image

Use `rocm/primus:v26.9` with Primus-Turbo at or after `54d628c9`. The Primus-Turbo that `v26.9`
ships (`de8fa6ca`) lacks the gfx950 dense-GEMM race fixes of Primus-Turbo#570.
`run_with_docker.sh` builds Primus-Turbo `PRIMUS_TURBO_REF` (default `fda6df0b`) into the
container; set `PRIMUS_TURBO_REF=` to keep the image's. After changing Primus-Turbo, clear
FlyDSL's compile cache (`~/.flydsl/cache`), whose key does not cover every kernel change.

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
export CONT=rocm/primus:v26.9
export DGXSYSTEM=MI355X_1x8x1
export NEXP=1

# Optional host runtime tunables before each trial (cpupower, THP, drop_caches; see runtime_tunables.sh):
# export RUN_RUNTIME_TUNABLES=1

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
docker run -it     --device /dev/dri     --device /dev/kfd     --device /dev/infiniband     --network host --ipc host     --group-add video     --cap-add SYS_PTRACE     --security-opt seccomp=unconfined     --privileged     --ulimit nofile=1048576:1048576     -v $HOME:$HOME   --shm-size 128G     --name primus_training_env rocm/primus:v26.9

cd /workspace/Primus
```

Then rebuild Primus-Turbo (see [Image](#image)):

```bash
git clone https://github.com/AMD-AGI/Primus-Turbo.git /tmp/primus_turbo_src
cd /tmp/primus_turbo_src && git checkout fda6df0b && git submodule update --init --recursive
pip uninstall -y primus_turbo && GPU_ARCHS=gfx950 pip install --no-build-isolation --no-deps .
rm -rf ~/.flydsl/cache
cd /workspace/Primus
```

### Key Files

- `configs/MI355X/llama3.1_8B-pretrain-FP4.yaml` - Model and training config
    - `train_data_path` and `valid_data_path` point under `/data`
- `config_MI355X_1x8x1.sh` - System config and env vars
    - `PRIMUS_PATH` is the Primus checkout (default `/workspace/Primus`); `EXP` is the yaml above
- `run_and_time.sh` - Run script

```bash
export HF_TOKEN=<your_huggingface_token>
source config_MI355X_1x8x1.sh
bash run_and_time.sh
```

## Notes

- `log_interval: 999999` suppresses regular Primus logs.
- `RUN_RUNTIME_TUNABLES=1` runs `runtime_tunables.sh` on the host before each trial (some steps need `sudo`).
- `run_and_time.sh` sets `HIPBLASLT_TUNING_OVERRIDE_FILE` to the file in `hipblaslt_tuning/` for the image's hipBLASLt build (`HIPBLASLT_VERSION_TWEAK`), if there is one. It tunes the output layer's weight-gradient GEMM (25.8 to 24.0 ms on `v26.9`). Set the variable yourself, empty for hipBLASLt's own choice, to override.
- `MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma` (default) runs the parameter all-gather on the SDMA copy engines. It needs `--ulimit nofile=1048576:1048576`, `NCCL_CTA_POLICY` unset and `GPU_MAX_HW_QUEUES=4` (set in the config). Set it to `nccl` for RCCL's all-gather.
- `MEGATRON_GRAD_REDUCE_BACKEND` runs the gradient reduce-scatter on the same communicator. It defaults to `MEGATRON_PARAM_GATHER_BACKEND`, so `rccl_sdma` unless the all-gather is switched off. Set it to `disabled` for Megatron's reduce-scatter. With the RCCL 2.32.3 in `rocm/primus:v26.9` it runs on symmetric-memory kernels.
- `PRIMUS_REBUILD_RCCL=1` builds an RCCL with copy-engine reduce-scatter (ROCm/rocm-systems#15042, `PRIMUS_RCCL_REF`, default `96020c58`) through `runner/helpers/patches/12_rebuild_rccl.sh` at the first trial's launch, and the reduce-scatter then runs on the copy engines. It is off by default: with that RCCL, about one step in 2,000 gets garbage gradients (values up to 1e37) on one GPU, which Megatron's clipping usually absorbs but which can turn the loss into NaN for the rest of the run (seed 731542219 did once on 8x MI355X). The same RCCL with the reduce-scatter on Megatron's path, and the image's RCCL, showed no such steps. The build takes about 70 s and needs network access.
- On 8x MI355X (seed 1234) the reduce-scatter is 6.4 ms per step faster with the rebuilt RCCL (760.1 ms) and 8.4 ms faster with the image's RCCL (758.1 ms) than with Megatron's (766.5 ms).
- `PRIMUS_TURBO_CROSS_ENTROPY=true` (default) uses Primus-Turbo's fused cross entropy. Set it to `false` for TE's; `PRIMUS_DEFERRED_CROSS_ENTROPY=true` then defers TE's gradient to backward. The two cannot both be on.
- For an older Primus-Turbo without `mlp_fp4` or `rope` ops, set `PRIMUS_TURBO_FUSED_DENSE_MLP=false` and `PRIMUS_TURBO_FLYDSL_ROPE=0`.

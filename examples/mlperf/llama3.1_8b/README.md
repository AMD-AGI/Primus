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

### Image

Use `rocm/primus:v26.9` with a Primus-Turbo at or after `d04ffd9b`. The Primus-Turbo that
`v26.9` ships (`de8fa6ca`) has every op this recipe uses but predates the gfx950 dense-GEMM race
fixes of Primus-Turbo#570, so rebuild Primus-Turbo from `main` in the container. After changing
Primus-Turbo, clear FlyDSL's compile cache (`FLYDSL_RUNTIME_CACHE_DIR`, `~/.flydsl/cache` by
default): its key does not cover every kernel change, so stale binaries can be reused.
The fused dense MLP needs `primus_turbo.pytorch.ops.mlp_fp4` and the fused QKV RoPE needs
`primus_turbo.pytorch.ops.rope` (`v26.9` has both); set
`PRIMUS_TURBO_FUSED_DENSE_MLP=false` and `PRIMUS_TURBO_FLYDSL_ROPE=0` for an older Primus-Turbo.
RoPE calls outside the FlyDSL kernel's shapes go to TE's, with a warning on the first one.

On 8x MI355X with `rocm/primus:v26.9`, Primus-Turbo `main` (`fda6df0b`) and seed 1234, this
recipe reached eval loss 3.289 at 196,608 samples, the same sample count as the MLPerf v6.0
submission, in 77.96 minutes at 761.3 ms per step. On the same node the submission
(`rocm/primus:v26.6`, Primus-Turbo `728e15ff`, SDMA off) took 79.86 minutes at 779.7 ms per step.


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
docker run -it     --device /dev/dri     --device /dev/kfd     --device /dev/infiniband     --network host --ipc host     --group-add video     --cap-add SYS_PTRACE     --security-opt seccomp=unconfined     --privileged     -v $HOME:$HOME   --shm-size 128G     --name primus_training_env rocm/primus:v26.9

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
- `RUN_RUNTIME_TUNABLES` defaults to `0`; set `RUN_RUNTIME_TUNABLES=1` to run `runtime_tunables.sh` on the host before each trial (some steps require `sudo`).
- `run_and_time.sh` sets `HIPBLASLT_TUNING_OVERRIDE_FILE` to the file in `hipblaslt_tuning/` named after the image's hipBLASLt build (`HIPBLASLT_VERSION_TWEAK`), if there is one; hipBLASLt solution indices differ between builds. Set the variable yourself (empty for hipBLASLt's own choices) to override. The files tune the output layer's weight-gradient GEMM, accumulated in place by Apex, with output identical to hipBLASLt's choice. `main_grad` is FP32 unless `grad_reduce_in_bf16: true`, which makes it BF16:

  | Image (hipBLASLt) | `main_grad`, MBS | hipBLASLt's choice | Tuned |
  |---|---|---|---|
  | `rocm/primus:v26.9` (`cb261897`) | FP32, 2 | 23202, 16.5 ms | 23309, 13.1 ms |
  | | FP32, 4 | 23202, 32.6 ms | 23309, 27.0 ms |
  | | BF16, 4 (this config) | 12094, 25.8 ms | 12115, 24.0 ms |
  | `rocm/primus:v26.6` (`bbb68174`) | BF16, 4 | 12088, 25.6 ms | 12109, 24.3 ms |

  On `v26.6` hipBLASLt's choice for FP32 `main_grad` is already the fastest. The files were made with `examples/megatron/guides/offline_tune/offline_tune_gemm_ext.py` from the GEMMs' `HIPBLASLT_LOG_MASK=32` lines; another image needs its own file.
- `PRIMUS_TURBO_CROSS_ENTROPY` (default `true` in `config_MI355X_1x8x1.sh`) replaces TE's fused cross entropy with Primus-Turbo's, which writes the logits' gradient over the logits. Loss and gradient are bit-identical to TE's; on `rocm/primus:v26.9` forward plus backward takes 2.90 ms instead of 3.65 ms per micro-batch (MBS 2). The image's Primus-Turbo must have `primus_turbo.pytorch.ops.cross_entropy` (`v26.9` has it, `v26.6` does not); set it to `false` for TE's.
- `PRIMUS_DEFERRED_CROSS_ENTROPY=true`, with `PRIMUS_TURBO_CROSS_ENTROPY=false`, keeps TE's cross entropy but builds its gradient in backward with the double BF16 rounding of the TE in `rocm/primus:v26.6`. It also converged at 196,608 samples, about 1.8 ms per step slower than Turbo's.
- `MEGATRON_PARAM_GATHER_BACKEND=rccl_sdma` (the default in `config_MI355X_1x8x1.sh`) runs the distributed optimizer's parameter all-gather on the SDMA copy engines. It needs an open-file limit of 1048576 in the container (`run_with_docker.sh` passes `--ulimit=nofile=1048576:1048576`) and `NCCL_CTA_POLICY` unset. Set it to `nccl` for RCCL's kernel all-gather. The config sets `GPU_MAX_HW_QUEUES=4` (the runner's `base_env.sh` defaults to 2): on the same node SDMA was 14 ms per step faster than `nccl` with 4 queues and 10 ms per step slower with 2.
- `PRIMUS_TRITON_FUSED_ADAM` (default `false`) selects TE's FusedAdam over the Triton fused Adam that `trainer_base.yaml` defaults to; TE's is 1.2-1.9 ms per step faster here.

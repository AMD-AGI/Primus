# Llama2-70B LoRA MLPerf on MI355X (Primus)

MLPerf Training 6.0 Llama2-70B LoRA on **MI355X** (8× GPU, 1 node) via Megatron-Bridge and `primus-cli`.

Dataset: [GovReport](https://gov-report-data.github.io/) (SCROLLS `gov_report`), packed to **8192** tokens.  
Model: **meta-llama/Llama-2-70b-hf** with LoRA (rank 16, alpha 32).  
Precision: **MXFP4** + BF16; **FP8 delayed scaling** after healing at step 340.

---

## Prerequisites

- 8× MI355X GPUs on one node
- Hugging Face access to `meta-llama/Llama-2-70b-hf` (`HF_TOKEN`)
- ~300 GB disk for packed data + Megatron checkpoint
- Docker with ROCm (`/dev/kfd`, `/dev/dri`)

---

## 1. Launch container

```bash
docker pull rocm/primus:v26.4

docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add 44 \
  --group-add 109 \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --memory=0 \
  --memory-swap=0 \
  --privileged \
  --ulimit nofile=65535:65535 \
  -v /home/kgoginen@amd.com/Primus:/workspace/Primus \
  rocm/primus:v26.4
```

Change the `-v` host path to your Primus checkout. Repo is at `/workspace/Primus` inside the container.

Optional: mount a data volume if data/checkpoints live on the host:

```bash
  -v /path/on/host/data:/data \
```

---

## 2. Set data and checkpoint paths

Inside the container:

```bash
cd /workspace/Primus

export HF_TOKEN=hf_...   # required on first run (hooks download model + dataset)

# Packed GovReport .npy files (train.npy, validation.npy, packed_metadata.jsonl)
export PACKED_DATA_DIR=/data

# Megatron checkpoint root (must contain latest_train_state.pt, not iter_0000000/)
export PRETRAINED_CHECKPOINT=/data/megatron_checkpoints/Llama-2-70b-hf
```

Hooks create these under `/data` on first run if missing. Point `PACKED_DATA_DIR` and `PRETRAINED_CHECKPOINT` at existing paths to skip re-download.

---

## 3. Run training

```bash
bash examples/megatron_bridge/llama2_70b_lora/run_mlperf_cli.sh
```

Hooks run automatically: patches → deps → dataset → HF→Megatron checkpoint.

Equivalent:

```bash
source examples/megatron_bridge/llama2_70b_lora/config_MI355X_1x8x1.sh
./runner/primus-cli direct train posttrain \
  --config examples/megatron_bridge/configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml
```

---

## 4. MLPerf experiment configuration

### Config files

| File | Role |
|------|------|
| `configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml` | Post-train overrides |
| `primus/configs/models/megatron_bridge/llama2_70b_lora_mxfp4.yaml` | Model recipe |
| `primus/recipes/llama2_custom.py` | `llama2_70b_lora_mxfp4_config` |
| `config_MI355X_1x8x1.sh` | MLPerf env (MXFP4, AITER, NCCL, MLLOG) |

### Training schedule

| Parameter | Value |
|-----------|-------|
| `train_iters` | 550 |
| `global_batch_size` | 8 |
| `micro_batch_size` | 1 |
| `seq_length` | 8192 |
| `lr` | 0.0006 |
| `eval_interval` / `eval_iters` | 48 / 24 |
| Quality target | eval loss **< 0.925** |

### Precision

**MXFP4 (steps 0–339):** `fp4=mxfp4`, `fp8=None`, `PRE_QUANTIZED_MODEL=True`, fused attention, AITER A4W4 GEMMs (`a4w4_tuned_gemms.csv`).

**FP8 healing (step 340+):** `HEALING_ITER=340`, delayed scaling via `FP8_*` env vars in `config_MI355X_1x8x1.sh`.

### LoRA

Targets `linear_qkv`, `linear_proj` (dim 16, alpha 32). `stable_lora_with_te_op_fuser=True` (unfused `LoRALinear` adapters).

### Parallelism

TP=1, PP=1, CP=1, 8 GPUs data parallel.

### Patches (`primus/recipes/patches/`)

`megatron_nemo_lora_only`, `megatron_bridge_validation_consumed_samples`, `megatron_bridge_deterministic_eval`, `sft_attention_mask_cache`, `megatron_lm_mxfp4_recipe`.

---

## 5. Logging

Bring-up defaults (`config_MI355X_1x8x1.sh`): `log_interval=10`, `PRIMUS_LOG_GPU_MEM=1`, `VERBOSE_TRAINING_LOG=1`.

MLPerf submission (quiet):

```bash
export PRIMUS_LOG_GPU_MEM=0
export VERBOSE_TRAINING_LOG=0
# yaml: log_interval: 99999, stderr_sink_level: ERROR
```

### Common issues

| Symptom | Fix |
|---------|-----|
| NCCL hang, 0% GPU | `NCCL_IB_DISABLE=1` (default in config) |
| Invalid pretrained checkpoint | Point at checkpoint **root**, not `iter_0000000` |
| Long silence at start | Pre-quantize + warmup + AITER JIT (normal) |

---

## 6. Optional overrides

```bash
export PRIMUS_TRAIN_ITERS=550
export SEED=1234
export SKIP_PATCHES=1
export SYNTH_WARMUP_STEPS=0
export NCCL_IB_DISABLE=0    # if RDMA works on your system
```

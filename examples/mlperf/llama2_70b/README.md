# Llama2-70B LoRA MLPerf on MI355X (Primus)

MLPerf Training 6.0 Llama2-70B LoRA on **MI355X** (8× GPU, 1 node) via Megatron-Bridge and `primus-cli`.

Dataset: [GovReport](https://gov-report-data.github.io/) (SCROLLS `gov_report`), packed to **8192** tokens.  
Model: **meta-llama/Llama-2-70b-hf** with LoRA (rank 16, alpha 32).  
Precision: **MXFP4** + BF16; **FP8 delayed scaling** after healing at step 340.

## Key files

- `configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml` — post-train overrides
- `config_MI355X_1x8x1.sh` — system config and env vars (set `PRIMUS_PATH` to your Primus clone)
- `run_and_time.sh` — one-shot MLPerf run via `primus-cli`
- `a4w4_tuned_gemms.csv` — tuned AITER A4W4 GEMM configs

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
bash examples/mlperf/llama2_70b/run_and_time.sh
```

Hooks run automatically: pip deps → dataset → HF→Megatron checkpoint.

Equivalent:

```bash
source examples/mlperf/llama2_70b/config_MI355X_1x8x1.sh
./runner/primus-cli direct train posttrain \
  --config examples/mlperf/llama2_70b/configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml
```

---

## 4. MLPerf experiment configuration

### Config files

| File | Role |
|------|------|
| `examples/mlperf/llama2_70b/configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml` | Post-train overrides |
| `examples/mlperf/llama2_70b/config_MI355X_1x8x1.sh` | MLPerf env (MXFP4, AITER, NCCL, MLLOG) |
| `examples/mlperf/llama2_70b/a4w4_tuned_gemms.csv` | Tuned AITER A4W4 GEMM configs |
| `primus/configs/models/megatron_bridge/llama2_70b_lora_mxfp4.yaml` | Model recipe |
| `primus/backends/megatron_bridge/recipes/mlperf_llama2_70b/llama2_custom.py` | `llama2_70b_lora_mxfp4_config` |

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

### MLPerf overrides (Primus-side, no third_party git patches)

Runtime patches under `primus/backends/megatron_bridge/patches/mlperf_llama2_70b/` apply only when
`llama2_70b_lora_mxfp4` / `llama2_70b_lora_mlperf_posttrain.yaml` is selected.

Recipe code lives under `primus/backends/megatron_bridge/recipes/mlperf_llama2_70b/`.

| File | Role |
|------|------|
| `lora.py` | NeMo-stable LoRA (`use_te_fused_lora=False`) |
| `resettable_data_iterator.py` | Deterministic validation iterator |
| `bridge_patches.py` | Data loaders, eval reset, SFT mask cache, NeMo timing |
| `megatron_patches.py` | MXFP4 recipe + optional TE SwiGLU |
| `conditions.py` | Scopes patches to MLPerf Llama2-70B only |

One-time cleanup if you previously applied git patches to submodules:

```bash
git -C third_party/Megatron-Bridge checkout -- .
git -C third_party/Megatron-Bridge/3rdparty/Megatron-LM checkout -- .
```

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
export SYNTH_WARMUP_STEPS=0
export NCCL_IB_DISABLE=0    # if RDMA works on your system
```

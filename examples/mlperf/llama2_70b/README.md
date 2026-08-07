# Llama2-70B LoRA MLPerf on MI355X (Primus)

MLPerf Training 6.0 Llama2-70B LoRA on **MI355X** (8× GPU, 1 node) via Megatron-Bridge and `primus-cli`.

Dataset: [GovReport](https://gov-report-data.github.io/) (SCROLLS `gov_report`), packed to **8192** tokens.
Model: **meta-llama/Llama-2-70b-hf** with LoRA (rank 16, alpha 32).
Precision: **MXFP4** + BF16; **FP8 delayed scaling** after healing at step 340.

## Key files

- `configs/MI355X/llama2_70b_lora_mlperf_posttrain.yaml` — post-train overrides
- `config_MI355X_1x8x1.sh` — system config and env vars (set `PRIMUS_PATH` to your Primus clone)
- `run_with_docker.sh` — **recommended**: host-side Docker orchestration (N timed trials)
- `runtime_tunables.sh` — host CPU/THP/cache tuning before each trial
- `run_and_time.sh` — timed MLPerf run inside the container (called by `run_with_docker.sh`)
- `a4w4_tuned_gemms.csv` — tuned AITER A4W4 GEMM configs

---

## Prerequisites

- 8× MI355X GPUs on one node
- Hugging Face access to `meta-llama/Llama-2-70b-hf` (`HF_TOKEN`)
- ~300 GB disk for packed data + Megatron checkpoint
- Docker with ROCm (`/dev/kfd`, `/dev/dri`)

---

## 1. Manual container (optional)

Use this only if you are **not** using **`run_with_docker.sh`** and want an interactive shell inside the image.

```bash
docker pull rocm/primus:v26.5

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
  -v /path/to/Primus:/workspace/Primus \
  -v /path/on/host/data:/data \
  rocm/primus:v26.5
```

Repo is at **`/workspace/Primus`** inside the container. Then follow [§4](#4-run-training-inside-an-existing-container).

## 2. Data layout (host `DATADIR` / container `/data`)

```text
${DATADIR}/
├── train.npy
├── validation.npy
├── packed_metadata.jsonl
├── megatron_checkpoints/Llama-2-70b-hf/   # checkpoint root (not iter_0000000/)
└── .cache/huggingface/                    # optional HF cache
```

**`run_with_docker.sh`** sets **`PACKED_DATA_DIR=/data`**, **`PRETRAINED_CHECKPOINT=/data/megatron_checkpoints/Llama-2-70b-hf`**, and **`HF_HOME=/data/.cache/huggingface`** automatically. On first run, posttrain hooks populate missing data/checkpoints when **`HF_TOKEN`** is set.

For a **manual** container (§1), export the same paths inside the shell before **`run_and_time.sh`**:

```bash
export HF_TOKEN=hf_...
export PACKED_DATA_DIR=/data
export PRETRAINED_CHECKPOINT=/data/megatron_checkpoints/Llama-2-70b-hf
```

Checkpoint root must contain **`latest_train_state.pt`**, not only **`iter_0000000/`**.

---

## 3. Run with `run_with_docker.sh` (recommended)

Use this when you want the **MLPerf submission-style flow** from the **host**: one command pulls up Docker, mounts data and results, applies host runtime tunables, runs **N** timed trials, and collects logs under **`LOGDIR`**. You do **not** need to `docker run -it` manually unless you are debugging inside the container (see [§4](#4-run-training-inside-an-existing-container)).

### 3.1 Quick start (non-interactive)

From your **Primus repo root** on an 8× MI355X node with Docker + ROCm devices:

```bash
cd /path/to/Primus

export HF_TOKEN=hf_...                      # required: Llama-2-70b + GovReport hooks
export DATADIR=/data/mlperf_llama2          # writable: dataset + checkpoint (host)
export LOGDIR=/data/mlperf_llama2/results   # writable: trial logs + MLLOG (host)
export CONT=rocm/primus:v26.5               # Primus ROCm image
export DGXSYSTEM=MI355X_1x8x1               # selects config_${DGXSYSTEM}.sh
export NEXP=1                               # use 10 for MLPerf submission runs

# Optional: quiet stdout (:::MLLOG + timing banners only)
export SUBMISSION_QUIET=1

bash examples/mlperf/llama2_70b/run_with_docker.sh
```

`PRIMUS_HOST` defaults to the repo root (three levels above `examples/mlperf/llama2_70b/`). Override if your checkout is elsewhere:

```bash
export PRIMUS_HOST=/other/path/Primus
bash examples/mlperf/llama2_70b/run_with_docker.sh
```

### 3.2 Interactive mode

Prompts for **`HF_TOKEN`**, image, **`NEXP`**, **`DATADIR`**, **`LOGDIR`**, and **`DGXSYSTEM`** when values are missing:

```bash
cd /path/to/Primus
export SUBMISSION_QUIET=1   # optional
INTERACTIVE=1 bash examples/mlperf/llama2_70b/run_with_docker.sh
```

### 3.3 What the script does

1. **Validates** `config_${DGXSYSTEM}.sh` exists and **`HF_TOKEN`** is set.
2. **Removes** any existing container named **`CONT_NAME`** (default `mlperf_llama2_70b_lora_primus`), then starts a **detached** container (`sleep infinity`) with ROCm devices and mounts below.
3. **Installs** editable Primus from the mount (`pip install -e /workspace/Primus --no-deps`); Python/torchrun come from the image venv **`/opt/venv/bin`** (not the host `PATH`).
4. For each trial **`1..NEXP`**:
   - Runs **`runtime_tunables.sh`** on the **host** (unless disabled).
   - Optionally drops host page cache if **`CLEAR_CACHES=1`**.
   - **`docker exec`** → **`bash /workspace/code/run_and_time.sh`** with a new **`SEED=$RANDOM`** and env from **`config_MI355X_1x8x1.sh`** (+ MLLOG paths under **`/results`**).
   - Streams container stdout to **`${LOGDIR}/${DATESTAMP}_<trial>.log`** on the host.
   - Copies **`mlperf_logging.out`** and **`train.mlperfposttrain.exp.log`** into **`${LOGDIR}/artifacts/`** after each successful trial.
5. **Removes** the container on exit (success or failure).

First run can take a long time: posttrain hooks may download the HF model, convert checkpoints, and pack GovReport under **`DATADIR`**. Later runs reuse **`${DATADIR}/train.npy`**, **`validation.npy`**, and **`megatron_checkpoints/`**.

### 3.4 Volume and path map

| Host | Container | Purpose |
|------|-----------|---------|
| **`PRIMUS_HOST`** (Primus repo) | `/workspace/Primus` | Code; editable install; `PRIMUS_PATH` |
| **`examples/mlperf/llama2_70b/`** (this example dir) | `/workspace/code` | `run_and_time.sh`, `config_*.sh` |
| **`DATADIR`** | `/data` | Packed `.npy`, HF cache, Megatron checkpoint |
| **`LOGDIR`** | `/results` | MLLOG, primus-cli logs, timed run artifacts |

Inside the container, **`run_and_time.sh`** uses **`/results`** for:

| File | Description |
|------|-------------|
| `mlperf_logging.out` | `:::MLLOG` log (`ENABLE_MLLOG=1`) |
| `train.mlperfposttrain.exp.log` | Full timed-run stdout |
| `logs/log_*.txt` | `primus-cli direct` log |
| `RESULT,LLAMA2_70B_LORA,,<sec>,AMD,<time>` | MLPerf wall-clock line |

On the **host**, after trial *k*:

| File | Description |
|------|-------------|
| `${LOGDIR}/${DATESTAMP}_k.log` | Tee of that trial’s container output |
| `${LOGDIR}/artifacts/mlperf_logging_${DATESTAMP}_k.out` | Copy of MLLOG |
| `${LOGDIR}/artifacts/train_${DATESTAMP}_k.log` | Copy of training log |

### 3.5 Environment variables

Variables you set most often:

| Variable | Default | Meaning |
|----------|---------|---------|
| **`HF_TOKEN`** | *(none)* | **Required.** Hugging Face token for Llama-2-70b + dataset hooks |
| **`DATADIR`** | `$HOME/data/mlperf_llama2` | Host data + checkpoint tree (mounted at `/data`) |
| **`LOGDIR`** | `${DATADIR}/results` | Host results (mounted at `/results`) |
| **`CONT`** | `rocm/primus:v26.5` | Docker image |
| **`DGXSYSTEM`** | `MI355X_1x8x1` | Suffix for `config_${DGXSYSTEM}.sh` |
| **`NEXP`** | `1` | Number of timed trials (submission often uses **10**) |
| **`SUBMISSION_QUIET`** | `0` | `1` → sets `PRIMUS_LOG_SUPPRESSION=1`, `MLPERF_VERBOSE_LOGS=0`, `PRIMUS_LOG_GPU_MEM=0`, `VERBOSE_TRAINING_LOG=0` |
| **`INTERACTIVE`** | `0` | `1` → prompt for missing settings |
| **`PRIMUS_HOST`** | auto (repo root) | Host path to Primus checkout |

Optional tuning:

| Variable | Default | Meaning |
|----------|---------|---------|
| **`CONT_NAME`** | `mlperf_llama2_70b_lora_primus` | Docker container name |
| **`RUN_RUNTIME_TUNABLES`** | `1` | `0` → skip host **`runtime_tunables.sh`** |
| **`CLEAR_CACHES`** | `0` | `1` → extra host page-cache drop before each trial (needs sudo) |
| **`CHECK_COMPLIANCE`** | `0` | `1` → run `mlperf_logging.compliance_checker` after each trial (non-blocking on failure) |
| **`MLPERF_RULESET`** | `6.0.0` | Ruleset passed to compliance checker |
| **`DATESTAMP`** | `date +%y%m%d%H%M%S` | Prefix for host trial log filenames |

The script also sets **`PRIMUS_GPU_MODEL`** from **`DGXSYSTEM`** (e.g. `MI355X`) so **`primus-env.sh`** works when **`rocm-smi`** is missing inside **`docker exec`**.

Config env from **`config_MI355X_1x8x1.sh`** (NCCL, MXFP4, MLLOG, 550 iters, etc.) is forwarded into the container; host **`PATH`** is **not** forwarded (container uses **`/opt/venv/bin`**).

### 3.6 Submission-style run (10 trials, quiet)

```bash
export HF_TOKEN=hf_...
export DATADIR=/data/mlperf_llama2
export LOGDIR=/data/mlperf_llama2/results
export NEXP=10
export SUBMISSION_QUIET=1
bash examples/mlperf/llama2_70b/run_with_docker.sh
```

Each trial gets a random **`SEED`**. Collect **`${LOGDIR}/${DATESTAMP}_*.log`** and **`${LOGDIR}/artifacts/`** for your submission bundle.

### 3.7 Host runtime tunables

Before each trial, **`runtime_tunables.sh`** (when **`RUN_RUNTIME_TUNABLES=1`**) adjusts cpupower governor, transparent huge pages, and may drop caches—matching the MLPerf reference host setup. Some steps need **root** (`sudo` for `cpupower`, `/proc/sys/vm/drop_caches`). If tunables fail on your cluster policy, set **`RUN_RUNTIME_TUNABLES=0`** and apply equivalent settings via your admin workflow.

### 3.8 Troubleshooting (`run_with_docker.sh`)

| Symptom | What to check |
|---------|----------------|
| `HF_TOKEN is required` | Export token or use **`INTERACTIVE=1`** |
| `Config not found: config_*.sh` | Set **`DGXSYSTEM=MI355X_1x8x1`** (or add a matching config file) |
| `torchrun: command not found` / `No module named 'yaml'` | Usually host **`PATH`** leaked into the container; use current script (container **`PATH=/opt/venv/bin:...`**) |
| `rocm-smi not found` warning | Expected in minimal **`docker exec`**; set **`PRIMUS_GPU_MODEL=MI355X`** or rely on auto-map from **`DGXSYSTEM`** |
| Permission denied under **`LOGDIR`** / **`DATADIR`** | Ensure host paths exist and are writable by your user |
| Trial fails immediately | Inspect **`${LOGDIR}/${DATESTAMP}_1.log`** and **`/results/logs/log_*.txt`** inside the mount |
| Stuck on first run | Hooks downloading/converting; verify **`HF_TOKEN`** and disk space (~300 GB under **`DATADIR`**) |

To debug inside the same mounts without re-running the full script, start a shell in the running container (while a trial is not active):

```bash
docker exec -it mlperf_llama2_70b_lora_primus bash -lc 'source /opt/venv/bin/activate && cd /workspace/Primus && bash /workspace/code/run_and_time.sh'
```

(Use your **`CONT_NAME`** if overridden.)

---

## 4. Run training inside an existing container

```bash
export HF_TOKEN=hf_...
source examples/mlperf/llama2_70b/config_MI355X_1x8x1.sh
bash examples/mlperf/llama2_70b/run_and_time.sh
```

Timed run writes under **`/results`**:

- `train.mlperfposttrain.exp.log` — full stdout from the timed wrapper
- `logs/log_*.txt` — primus-cli direct log
- `mlperf_logging.out` — `:::MLLOG` submission log (`ENABLE_MLLOG=1`)
- `RESULT,LLAMA2_70B_LORA,,<seconds>,AMD,<start time>` — wall-clock line for MLPerf timing

One-shot without sourcing config first (env + hooks still run):

```bash
export HF_TOKEN=hf_...
bash examples/mlperf/llama2_70b/run_and_time.sh
```

Equivalent manual launch (same as `run_and_time.sh` without timing/`tee`):

```bash
cd /workspace/Primus   # or: cd "${PRIMUS_PATH}"

export HF_TOKEN=hf_...
source examples/mlperf/llama2_70b/config_MI355X_1x8x1.sh

mkdir -p /results/logs
cd "${PRIMUS_PATH}"

./primus-cli direct \
  --log_file "/results/logs/log_$(date +%Y%m%d_%H%M%S).txt" \
  -- \
  train posttrain \
  --config "${EXP}"
```

Notes:

- The **`--`** separates launcher flags from the Primus Python CLI (`train posttrain ...`).
- Run from **`${PRIMUS_PATH}`** with **`--log_file` under `/results/logs`** so a read-only Primus bind mount does not fail on `logs/`.
- `EXP` is set by `config_MI355X_1x8x1.sh`; override with `export EXP=...` if needed.

---

## 5. MLPerf experiment configuration

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

## 6. Logging

Bring-up defaults (`config_MI355X_1x8x1.sh`): `log_interval=10`, `PRIMUS_LOG_GPU_MEM=1`, `VERBOSE_TRAINING_LOG=1`.

MLPerf submission (quiet):

```bash
export PRIMUS_LOG_GPU_MEM=0
export VERBOSE_TRAINING_LOG=0
# Or use run_with_docker.sh: SUBMISSION_QUIET=1 (also sets PRIMUS_LOG_SUPPRESSION + MLPERF_VERBOSE_LOGS)
# yaml: log_interval: 99999, stderr_sink_level: ERROR
```

### Common issues

| Symptom | Fix |
|---------|-----|
| NCCL hang, 0% GPU | `NCCL_IB_DISABLE=1` (default in config) |
| Invalid pretrained checkpoint | Point at checkpoint **root**, not `iter_0000000` |
| Long silence at start | Pre-quantize + warmup + AITER JIT (normal) |

---

## 7. Optional overrides

```bash
export PRIMUS_TRAIN_ITERS=550
export SEED=1234
export SYNTH_WARMUP_STEPS=0
export NCCL_IB_DISABLE=0    # if RDMA works on your system
```

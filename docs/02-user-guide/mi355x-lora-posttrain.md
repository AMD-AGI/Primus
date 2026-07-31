# MI355X native LoRA post-training (Megatron)

Run **parameter-efficient LoRA** fine-tuning on a single **8× MI355X** node using Primus’s **native Megatron** post-trainer (`framework: megatron`). These recipes complement Megatron Bridge configs in [Post-training workflows](./posttraining.md).

| Model | Config | Dataset | Seq length | Parallelism |
|-------|--------|---------|------------|-------------|
| Llama 2 70B | [`llama2_70B-BF16-lora-128k-longalign.yaml`](../../examples/megatron/configs/MI355X/llama2_70B-BF16-lora-128k-longalign.yaml) | [LongAlign-10k](https://huggingface.co/datasets/THUDM/LongAlign-10k) (packed) | 131072 | TP=2, CP=4 |
| Qwen2.5 72B | [`qwen2.5_72B-BF16-lora-128k-longalign.yaml`](../../examples/megatron/configs/MI355X/qwen2.5_72B-BF16-lora-128k-longalign.yaml) | [LongAlign-10k](https://huggingface.co/datasets/THUDM/LongAlign-10k) (packed) | 32768 (default; 8k/128k via env) | TP=1, CP=8 @ 32k |
| Qwen3-235B-A22B MoE | [`qwen3_235B_A22B-BF16-lora-alpaca.yaml`](../../examples/megatron/configs/MI355X/qwen3_235B_A22B-BF16-lora-alpaca.yaml) | [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (packed) | 4096 | TP=1, EP=8 |
| Qwen3-235B-A22B MoE | [`qwen3_235B_A22B-BF16-lora-govreport.yaml`](../../examples/megatron/configs/MI355X/qwen3_235B_A22B-BF16-lora-govreport.yaml) | [SCROLLS GovReport](https://huggingface.co/datasets/tau/scrolls) (local JSONL + `hf_chat`) | 8192 | TP=1, EP=8 |

All recipes use BF16 base weights, LoRA adapters, and `use_packed_attention: false` (ROCm-safe implicit packing). Implementation details: [Native SFT and LoRA](../04-technical-guides/native-sft-lora.md).

For **MLPerf Llama 2 70B LoRA** at 8192 with pre-built `.npy` packs, use [`llama2_70B-BF16-sft-packed-mlperf_aligned.yaml`](../../examples/megatron/configs/MI355X/llama2_70B-BF16-sft-packed-mlperf_aligned.yaml) instead of the 128k LongAlign recipe.

---

## Prerequisites

- One node with **8× MI355X** GPUs (288 GB HBM per GPU)
- Docker with ROCm device access (`/dev/kfd`, `/dev/dri`) if using containers
- Hugging Face token with access to gated models when needed:
  - `meta-llama/Llama-2-70b-hf`
  - `Qwen/Qwen3-235B-A22B`
- Megatron **torch_dist** checkpoints converted for Primus ([Checkpoint management](../04-technical-guides/checkpoint-management.md))
- Network egress to Hugging Face on first dataset download

---

## Base Megatron checkpoints

Native Megatron LoRA **loads base weights from a Megatron `torch_dist` tree** before adapters are applied. The HF model id in each yaml (`hf_path` / `tokenizer_model`) is used for the **tokenizer** and for **optional** checkpoint conversion — not as the direct weight source at train time unless you have already converted HF weights into Megatron format.

**Typical layout on disk:**

- `latest_checkpointed_iteration.txt` (often contains `release`)
- A shard directory such as `release/` (or `iter_*`) with distributed checkpoint files

**Two ways to satisfy the base checkpoint:**

1. **Reuse an existing Megatron checkpoint** — Bind-mount or copy the tree onto the path your yaml expects (see the table below). Override with `PRETRAINED_CHECKPOINT=/path/to/checkpoint` if your mount differs.

2. **Convert on first launch** — The posttrain hook `runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py` can HF→Megatron convert using `hf_path` / `tokenizer_model` and write under `${DATA_PATH}/megatron_checkpoints/<HF-repo-name>/`. That hook runs only when **`pretrained_checkpoint` (and `load`) are not set** in the yaml. The MI355X LoRA yamls **default** `pretrained_checkpoint` to `/data/megatron_checkpoints/...`, so if that directory is missing the hook **skips** conversion and training fails at load time. For a greenfield setup, either mount a real checkpoint at the default path or temporarily **omit** `pretrained_checkpoint` in the yaml so the hook can convert (requires HF access, large disk, and time for big models such as Qwen3-235B).

**Quick sanity check inside the container:**

```bash
CKPT="${PRETRAINED_CHECKPOINT:-/data/megatron_checkpoints/Qwen3-235B-A22B}"
test -d "$CKPT/release" -o -d "$CKPT/iter_0000000" && echo "checkpoint shards OK" || echo "missing or incomplete checkpoint"
```

Logs may show `[PEFT pre-wrap] Loading base model weights from: ...` followed by a clear error if the directory does not exist. MoE models (Qwen3-235B) use `MoELayer` blocks; debug “canary” lines about missing dense `linear_fc1` on layer 0 are expected and not a training failure by themselves.

**Disk space (Qwen3-235B HF→Megatron conversion):** The convert hook downloads the full HF model into `HF_HOME` (under `${DATA_PATH}/huggingface` by default). Qwen3-235B-A22B is on the order of **~470 GB** of HF shards (119 files, ~4 GB each) plus additional space for the Megatron `torch_dist` output. A full home-directory or repo `Primus/data` mount is often too small. Use a **large bind mount**, set `DATA_PATH` (and thus `HF_HOME`) explicitly before launch, and prefer a **pre-converted Megatron checkpoint** with `PRETRAINED_CHECKPOINT` instead of on-node conversion when disk is tight.

**Why caches land under `Primus/data`:** If you do not set `DATA_PATH` and there is no writable `/data` in the container, Primus defaults to **`${PRIMUS_PATH}/data`** (`runner/helpers/envs/base_env.sh`) — e.g. `/workspace/Primus/data` when the repo is bind-mounted. That is intentional for local dev but not enough for 235B HF downloads. After a failed partial download, remove incomplete blobs under `.../huggingface/hub/models--Qwen--Qwen3-235B-A22B/` before retrying on a larger filesystem.

---

## Environment

From the Primus repository root (host or container at `/workspace/Primus`):

```bash
export HF_TOKEN=<your_hf_token>
# Required on most clusters: large disk for HF + Megatron artifacts (not ~/Primus/data unless it is huge).
export DATA_PATH=/data   # or /mnt/your_scratch/primus_data — must exist and be writable
export HF_HOME="${DATA_PATH}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${DATA_PATH}/megatron_checkpoints" "${HF_HOME}"

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export GPUS_PER_NODE=8
export PRIMUS_EXP_NAME=my_lora_run_$(date +%Y%m%d_%H%M%S)
```

On ROCm, set **`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`** before **`primus-cli direct`** (especially Qwen2.5-72B @ 32k). PyTorch OOM messages may reference `PYTORCH_CUDA_ALLOC_CONF`; that name is CUDA-only — use the HIP variable on MI355X.

The MI355X LoRA yamls default to container paths under **`/data`** (no `PRETRAINED_CHECKPOINT` / dataset env vars required if you bind-mount host storage there). See [Base Megatron checkpoints](#base-megatron-checkpoints) for mount vs first-run conversion.

| Resource | Default path in yaml |
|----------|----------------------|
| Qwen3 base ckpt | `/data/megatron_checkpoints/Qwen3-235B-A22B` |
| Llama 2 70B base ckpt | `/data/megatron_checkpoints/Llama-2-70b-hf` |
| Qwen2.5 72B base ckpt | `/data/megatron_checkpoints/Qwen2.5-72B` |
| GovReport JSONL | `/data/scrolls_govreport` (`train.jsonl`, `validation.jsonl`) |
| Alpaca / LongAlign | Hugging Face hub (cache under `${DATA_PATH}/huggingface`) |

For GovReport packing on shared NFS caches, put lock files on node-local disk:

```bash
export PRIMUS_PACK_LOCK_DIR=/tmp/primus_pack_locks
mkdir -p "$PRIMUS_PACK_LOCK_DIR"
```

---

## Container setup

```bash
docker pull rocm/primus:v26.5
```

```bash
docker run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --ulimit nofile=65535:65535 \
  -v /path/to/Primus:/workspace/Primus \
  -v /path/to/host/data:/data \
  rocm/primus:v26.5
```

Alternative launch:

```bash
./runner/primus-cli container --image rocm/primus:v26.4 -- \
  train posttrain \
  --config examples/megatron/configs/MI355X/llama2_70B-BF16-lora-128k-longalign.yaml
```

---

## Llama 2 70B LoRA @ 128k (LongAlign)

Long-context SFT needs datasets with real long samples; packing **LongAlign-10k** into 128k bins avoids extreme padding from short instruction sets at 131072 tokens.

**Defaults:** `train_iters=5` (smoke), GBS=8, MBS=1, full layer recompute (80 layers).

```bash
export LONGALIGN_DATASET=THUDM/LongAlign-10k

./runner/primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI355X/llama2_70B-BF16-lora-128k-longalign.yaml
```

**Qwen2.5-72B** (LongAlign; default **32k** packed seq, **TP=1 CP=8**, full recompute). **Required on ROCm** before launch — reduces HBM fragmentation (OOM logs may mention `PYTORCH_CUDA_ALLOC_CONF`; on MI355X use **`PYTORCH_HIP_ALLOC_CONF`**):

```bash
cd /workspace/Primus   # or your Primus checkout

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export DATA_PATH=/data
export HF_HOME="${DATA_PATH}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_TOKEN=<your_hf_token>   # if needed for LongAlign / gated assets

# Megatron base (skip convert hook if already present):
export PRETRAINED_CHECKPOINT=/data/megatron_checkpoints/Qwen2.5-72B

export LONGALIGN_DATASET=THUDM/LongAlign-10k
export GPUS_PER_NODE=8
export PRIMUS_EXP_NAME=qwen2.5_72B_lora_$(date +%Y%m%d_%H%M%S)

./runner/primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI355X/qwen2.5_72B-BF16-lora-128k-longalign.yaml
```

Shorter smoke (**8192**, `CP=1`, less recompute):

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PRIMUS_SEQ_LENGTH=8192
export PRIMUS_MAX_POSITION_EMBEDDINGS=8192
export PRIMUS_CP=1
export PRIMUS_RECOMPUTE_LAYERS=8
export PRIMUS_DISABLE_PACK_CACHE=1

./runner/primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI355X/qwen2.5_72B-BF16-lora-128k-longalign.yaml
```

**128k LongAlign** (rebuilds pack cache):

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PRIMUS_SEQ_LENGTH=131072
export PRIMUS_MAX_POSITION_EMBEDDINGS=131072
export PRIMUS_CP=8
export PRIMUS_RECOMPUTE_LAYERS=80
export PRIMUS_DISABLE_PACK_CACHE=1

./runner/primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI355X/qwen2.5_72B-BF16-lora-128k-longalign.yaml
```

**First run:** global rank 0 tokenizes and packs LongAlign; watch for `[Pack] Tokenized N/M samples...` in logs. Cached packs live under `$HF_DATASETS_CACHE/primus_packed/` when the cache key is unchanged.

**Production runs:**

```bash
export PRIMUS_TRAIN_ITERS=500
export PRIMUS_LR_DECAY_ITERS=500
```

**If memory is tight:**

```bash
export PRIMUS_TP=1
export PRIMUS_CP=8
```

---

## Qwen3-235B-A22B LoRA (Alpaca)

```bash
./runner/primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI355X/qwen3_235B_A22B-BF16-lora-alpaca.yaml
```

**If iter-1 OOMs:**

```bash
export PRIMUS_MBS=1
export PRIMUS_GBS=8
export PRIMUS_RECOMPUTE_LAYERS=94
```

**After stable training (~85–90% VRAM):**

```bash
export PRIMUS_MBS=2
export PRIMUS_GBS=16
```

Alpaca on Hugging Face is typically **train-only**; validation runs only if you provide a validation split or set `PRIMUS_EVAL_ITERS=0`.

---

## Qwen3-235B-A22B LoRA (SCROLLS GovReport @ 8192)

### Data prep

Download and unpack GovReport to JSONL (train + validation only):

```bash
bash examples/megatron/scripts/prepare_scrolls_govreport.sh

./runner/primus-cli direct -- train posttrain \
  --config examples/megatron/configs/MI355X/qwen3_235B_A22B-BF16-lora-govreport.yaml
```

(`prepare_scrolls_govreport.sh` writes to `/data/scrolls_govreport` by default; the yaml uses the same path.)

**Tokenization on first launch (global rank 0 only; other ranks load the cache):**

1. Load JSONL via `sft_dataset_name=${SCROLLS_GOVREPORT_DIR}`.
2. Build user/assistant messages per sample (`primus/backends/megatron/sft/chat_template.py`).
3. Call Qwen3 `apply_chat_template` with assistant-only loss mask.
4. FFD-pack tokenized samples into 8192-token bins (`enable_packed_sequences: true`, `use_packed_attention: false`).

Cached packs are stored under `$HF_DATASETS_CACHE/primus_packed/`. Set `PRIMUS_DISABLE_PACK_CACHE=1` to force a rebuild.

There is **no `test.jsonl`** in the default GovReport export; Megatron skips the test dataloader when that file is missing.

**Performance tuning** (if VRAM/GPU util look low after iter-1 is stable):

```bash
export PRIMUS_RECOMPUTE_LAYERS=8   # raise toward 94 if OOM
export PRIMUS_MBS=2
export PRIMUS_GBS=16
```

**HF hub alternative** (no local JSONL): set `sft_dataset_name: tau/scrolls` and `sft_dataset_config: gov_report` in the yaml.

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| OOM at 128k Llama | `PRIMUS_TP=1`, `PRIMUS_CP=8`; confirm `recompute_num_layers: 80` |
| OOM at 128k Qwen2.5-72B LoRA | Default yaml uses TP=1 CP=8; set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`; if still OOM try `PRIMUS_SEQ_LENGTH=65536` (new pack cache) or LoRA on `linear_qkv`/`linear_proj` only |
| OOM on Qwen iter-1 | `PRIMUS_MBS=1`, raise `PRIMUS_RECOMPUTE_LAYERS`, disable overlap collectives |
| Hang at attention on ROCm | Keep `use_packed_attention: false` |
| Pack lock spam on NFS | `export PRIMUS_PACK_LOCK_DIR=/tmp/primus_pack_locks` |
| GovReport pack slow | Normal first run; ensure rank 0 has HF access and disk under `HF_DATASETS_CACHE` |
| Missing base checkpoint | See [Base Megatron checkpoints](#base-megatron-checkpoints): mount a valid `torch_dist` tree or allow `01_convert_checkpoints` by unsetting `pretrained_checkpoint` in the yaml |
| Disk quota / HF download fails under `Primus/data` | Point `DATA_PATH` and `HF_HOME` at a large filesystem; Qwen3-235B HF convert needs ~470 GB+ in cache; delete partial `models--Qwen--Qwen3-235B-A22B` blobs after a failed run |
| LoRA not applied | Logs show full fine-tuning — check `lora.enabled: true` in yaml |

---

## Related documentation

- [Post-training workflows](./posttraining.md) (Megatron Bridge)
- [Native SFT and LoRA](../04-technical-guides/native-sft-lora.md)
- [Performance tuning](../04-technical-guides/performance-tuning.md)

---

[← User guide](./README.md)

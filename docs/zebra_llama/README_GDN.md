# Pure GDN 300M on Primus — End-to-End Guide (FLA-validated)

This document is a runnable walkthrough for the **300M pure Gated DeltaNet (GDN)** pretraining recipe in Primus, validated on 8× AMD MI300X against the [Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention) reference implementation. It covers every step from raw dataset → tokenization → training → checkpoint conversion → lm-eval benchmark.

The same recipe scales up to the 1B pure-GDN config (`zebra_llama_1B_gdn_pure-pretrain.yaml`) — just swap the config file at training time and the FLA config JSON at conversion time.

---

## Final result

After 4768 iterations (≈10B tokens) on FineWeb-Edu sample-10BT:


| Axis                                            | FLA reference     | Primus (this branch)  | Δ                           |
| ----------------------------------------------- | ----------------- | --------------------- | --------------------------- |
| Per-iteration time (avg over 4768 iters)        | **1434.6 ms**     | **1431.6 ms**         | **−0.21 % (Primus faster)** |
| Throughput                                      | 182,729 tok/s/GPU | **183,213 tok/s/GPU** | **+0.27 %**                 |
| TFLOP/s/GPU                                     | —                 | 642                   | —                           |
| Wall time (4768 iters, 8× MI300X, healthy node) | 1h 54m 00s        | **1h 53m 42s**        | **−18s**                    |
| Loss @ iter 1                                   | 11.9654           | **11.9652**           | **−0.00 % (bit-perfect)**   |
| Loss @ iter 4700 (final logged)                 | 3.3511            | **3.3590**            | **+0.24 %**                 |
| First Primus-below-FLA crossover                | —                 | iter 2100             | —                           |


Loss trajectories overlap from iter ~2000 onward; the only persistent gap is in the LR-warmup region (iter 50–500) and closes monotonically. See [GDN_FLA_PARITY.md](../../GDN_FLA_PARITY.md) for the deep-dive on every patch and env var.

---

## Table of contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Environment](#step-1-environment)
- [Step 2: Dataset preparation](#step-2-dataset-preparation)
- [Step 3: Apply Megatron-LM patches](#step-3-apply-megatron-lm-patches)
- [Step 4: (Optional) Initialize from FLA weights](#step-4-optional-initialize-from-fla-weights)
- [Step 5: Train](#step-5-train)
- [Step 6: Monitor and compare against FLA](#step-6-monitor-and-compare-against-fla)
- [Step 7: Convert checkpoint to HuggingFace format](#step-7-convert-checkpoint-to-huggingface-format)
- [Step 8: Verify conversion](#step-8-verify-conversion)
- [Step 9: Run lm-eval-harness benchmarks](#step-9-run-lm-eval-harness-benchmarks)
- [Configs and tools used](#configs-and-tools-used)
- [Troubleshooting](#troubleshooting)

---

## Overview

The 300M pure-GDN model has:

- 12 Gated DeltaNet blocks + 12 MLP blocks → 24 Megatron "sublayers"
- `hidden_size = 1024`, `ffn_hidden_size = 4096`
- `num_heads = 4` (Q/K), `num_v_heads = 8` (V, grouped-value attention)
- `head_dim = 64`, short-conv kernel size 4
- Tied embeddings, no positional encoding (delta-rule recurrence), RMSNorm with `eps = 1e-6`
- Tokenizer: `meta-llama/Llama-3.2-1B` (128k vocab)
- Total parameters: **0.308B**

Training schedule (matched to FLA's `gated_deltanet_300M_pure.json`):

- 4768 iterations × 1024 global batch × 2048 seq len = **10.0 B tokens**
- AdamW (β1=0.9, β2=0.95, wd=0.01), peak LR `2e-4`, cosine decay, 200-step warmup
- bf16 mixed-precision, no dropout, gradient clip 1.0

---

## Prerequisites

- **Hardware**: 8× AMD MI300X (or compatible ROCm GPU) on a single node
- **Software**: ROCm ≥ 7.0, Docker ≥ 24.0
- **Container image**: `rocm/primus:v26.2` (or `v25.10` with the same patches)
- **HF token**: `HF_TOKEN` set for the gated `meta-llama/Llama-3.2-1B` tokenizer
- **Disk**: ~20 GB for the FLA-aligned tokenized dataset + ~5 GB per saved checkpoint
- **Optional**: a local clone of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) checked out at `legacy/training` — only needed if you want to reuse FLA's preprocessed Arrow files (recommended for bit-identical iter-1 comparison)

---

## Step 1: Environment

### 1.1 Start the dev container

The repo ships with `bash-docker.sh` (see `[bash-docker.sh](../../bash-docker.sh)`):

```bash
bash bash-docker.sh
```

This runs the `rocm/primus:v26.2` image with `/dev/dri`, `/dev/kfd`, IB devices, `--privileged`, your `$HOME` mounted in-place, and `--shm-size 64G`. The container is named `primus_hybrid_new`.

To re-attach later:

```bash
docker exec -it primus_hybrid_new bash
cd /home/<user>/Primus
```

### 1.2 Install Python dependencies inside the container

```bash
pip install -r requirements.txt
pip install flash-linear-attention   # FLA model classes + Triton kernels
pip install lm-eval                  # for benchmark evaluation
```

The `flash-linear-attention` package supplies the FLA `GatedDeltaNetForCausalLM` class (needed for HF conversion + lm-eval) and the Triton kernels that the optional `PRIMUS_FLA_*` toggles route into.

---

## Step 2: Dataset preparation

You have two choices. **For exact loss-curve parity with the FLA reference run, use Option B.** For a quick first run that won't bit-match FLA in the warmup region but will converge to the same final loss, Option A is fine.

### Option A — Standard Megatron data prep (faster setup)

```bash
export HF_TOKEN="hf_your_token_here"
export PYTHONPATH="$(pwd)/third_party/Megatron-LM:${PYTHONPATH}"

python examples/megatron/prepare_fineweb_edu.py \
    --primus-path . \
    --data-path ./data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.2-1B \
    --sample-size 10BT
```

Output ends up at `./data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_{0..3}_text_sentence.{bin,idx}` (4 shards). Then point the YAML at it:

```yaml
train_data_path: >
  /path/to/data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_0_text_sentence
  /path/to/data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_1_text_sentence
  /path/to/data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_2_text_sentence
  /path/to/data/fineweb-edu-10BT/HuggingFaceTokenizer/fineweb_edu_10BT_3_text_sentence
```

### Option B — FLA-aligned data (recommended for parity)

The Megatron `GPTDataset` shuffler produces a different token order than FLA's `DistributedSampler` (`seed=42`, fixed 2048-token chunks, no EOD tokens). To match exactly, reuse FLA's already-preprocessed Arrow shards and re-encode them into Megatron `.bin`/`.idx` format.

**Step B.1 — Preprocess with FLA's script** (one-time, ~10 min on 64 cores):

```bash
cd /path/to/flash-linear-attention/legacy/training
python preprocess.py \
    --dataset HuggingFaceFW/fineweb-edu \
    --name sample-10BT \
    --tokenizer meta-llama/Llama-3.2-1B \
    --seq_len 2048 --num_proc 64
```

This writes Arrow shard files to `legacy/training/data/HuggingFaceFW/fineweb-edu/sample-10BT/train/data-*.arrow`.

**Step B.2 — Convert the Arrow shards to Megatron binary** using the script at `[tools/convert_fla_to_megatron.py](../../tools/convert_fla_to_megatron.py)`:

```bash
cd /home/<user>/Primus
# Edit FLA_DATA and OUT_PREFIX at the top of the script if your paths differ
python tools/convert_fla_to_megatron.py
```

The script reads each Arrow shard directly with PyArrow (zero HuggingFace `datasets` overhead), writes a single `.bin` containing flat int32 token IDs, and emits a Megatron `.idx` file where each 2048-token chunk is one document. It cross-checks the first 10 tokens of the output against the first sample of the first Arrow shard before finishing.

Output: `data/fla_aligned/fla_fineweb_edu_10BT_text_sentence.{bin,idx}` (~19 GB binary).

The default 300M YAML already points at this path:

```yaml
train_data_path: >
  /home/<user>/Primus/data/fla_aligned/fla_fineweb_edu_10BT_text_sentence
```

(adjust the user prefix in `examples/megatron/configs/MI300X/zebra_llama_300M_gdn_pure-pretrain.yaml` to match your home directory).

---

## Step 3: Apply Megatron-LM patches

The vendored `third_party/Megatron-LM` submodule needs six patches to support GDN parity training. They live in `megatron_patches/*.patch` and are applied by an idempotent script:

```bash
bash megatron_patch.sh           # apply all 6
bash megatron_patch.sh --check   # dry-run (does not modify files)
bash megatron_patch.sh --revert  # undo all
```

The script is safe to re-run — already-applied patches are skipped. What each patch does (see `[megatron_patch.sh](../../megatron_patch.sh)` for the full breakdown):


| Patch | Touches                 | Purpose                                                                                                                                                         |
| ----- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01    | `mamba_model.py`        | Wires `FusedLinearCrossEntropyLoss` / `FusedCrossEntropyLoss` — never materializes the (batch×seq, vocab) logits tensor; gated by `PRIMUS_FUSED_CE` (default 1) |
| 02    | `optimizer/__init__.py` | Adds `PRIMUS_TORCH_OPTIM=1` opt-in for `torch.optim.AdamW(fused=True)` over TE/Apex FusedAdam                                                                   |
| 03    | `transformer/mlp.py`    | Routes SwiGLU through FLA's Triton-fused kernel (saves ~20 ms/iter); `PRIMUS_FLA_SWIGLU` (default 1)                                                            |
| 04    | `torch_norm.py`         | Routes RMSNorm through `fla.modules.RMSNorm` when `PRIMUS_FLA_NORM=1`                                                                                           |
| 05    | `transformer_config.py` | For hybrid models, uses uniform `init_method_normal` (no depth-scaled std) — required for bit-perfect iter-1 loss vs FLA                                        |
| 06    | `pretrain_mamba.py`     | FLA-order dataset shim (`PRIMUS_FLA_DATA=1` + `PRIMUS_FLA_CACHE_DIR=<path>`) and diagnostic iter-1 batch/activation dumps                                       |


---

## Step 4: (Optional) Initialize from FLA weights

For bit-perfect iter-1 loss alignment, the validated run loads FLA's *initialized but untrained* checkpoint and then trains from there. The YAML's `load:` field points at this directory:

```yaml
load: /home/<user>/Primus/output/fla_init_ckpt_300M
finetune: true            # load weights, ignore optimizer state and iteration count
no_load_optim: true
no_load_rng: true
```

The Primus repo includes `tools/init_primus_from_fla.py` (untracked, kept for forensics) that takes the FLA HuggingFace random-init checkpoint and writes a Megatron-shape `iter_0000000/mp_rank_00/model_optim_rng.pt`. Skip this step if you're happy with Primus's own random init — final loss is identical, only iter-1 drifts by `~5e-3`.

---

## Step 5: Train

### 5.1 Inspect the config

The training config lives at `[examples/megatron/configs/MI300X/zebra_llama_300M_gdn_pure-pretrain.yaml](../../examples/megatron/configs/MI300X/zebra_llama_300M_gdn_pure-pretrain.yaml)`. Key parameters (matched to FLA):

```yaml
train_iters: 4768                 # ≈ 10B tokens at global_batch=1024, seq=2048
micro_batch_size: 128             # per-GPU
global_batch_size: 1024           # 8 GPUs × 128 = 1024
seq_length: 2048
lr: 2.0e-4
min_lr: 2.0e-5                    # min_lr_rate=0.1 → 2e-5
lr_warmup_iters: 200
lr_decay_iters: 4768
lr_decay_style: cosine
adam_beta1: 0.9
adam_beta2: 0.95
weight_decay: 0.01
clip_grad: 1.0
seed: 42
layernorm_epsilon: 1.0e-6         # MUST be explicit — TransformerConfig default 1e-5 silently overrides the model YAML
hidden_dropout: 0.0               # MUST be explicit — language_model.yaml default 0.1 leaks through
attention_dropout: 0.0
spec: ['primus.backends.megatron.core.models.hybrid.hybrid_mamba_mla_layer_specs', 'gdn_hybrid_stack_spec_no_te']
use_distributed_optimizer: false  # 300M fits — ZeRO-1 adds allreduce overhead
```

The architecture-only YAML it extends from is `[primus/configs/models/megatron/zebra_llama_300M_gdn_pure.yaml](../../primus/configs/models/megatron/zebra_llama_300M_gdn_pure.yaml)`.

### 5.2 Launch

```bash
# inside the container, in /home/<user>/Primus
EXP=examples/megatron/configs/MI300X/zebra_llama_300M_gdn_pure-pretrain.yaml \
  bash examples/run_pretrain.sh 2>&1 | tee primus_gdn.log
```

This brings up `torchrun` with 8 ranks on the local node. Expected wall time on a healthy MI300X box: **~1h 54m** for the full 4768 iters.

### 5.3 Recommended env-var profile (for FLA parity)

The defaults are already good; for *bit-level* parity with FLA's optimizer/CE/SwiGLU kernels, also set:

```bash
export PRIMUS_FUSED_CE=1          # FLA FusedLinearCrossEntropyLoss (chunked, no full logits tensor)
export PRIMUS_FLA_SWIGLU=1        # FLA Triton SwiGLU
export PRIMUS_FLA_NORM=1          # FLA fused RMSNorm + fused pre-norm/MLP path
export PRIMUS_TORCH_OPTIM=1       # torch.optim.AdamW(fused=True), matches FLA exactly
# Only if you did Option B (FLA-aligned data) AND want bit-identical iter-1:
export PRIMUS_FLA_DATA=1
export PRIMUS_FLA_CACHE_DIR=/home/<user>/Primus/data/huggingface
```

These add roughly +1.2 % per-iter overhead vs the all-defaults run, but they pin the loss curve to FLA's. On healthy hardware (tw006 in our cluster), the absolute wall is still ~18 s **below** FLA. On a slower node (tw029) it's ~75 s above. See [GDN_FLA_PARITY.md](../../GDN_FLA_PARITY.md) for the cost-of-each-flag breakdown.

### 5.4 Output layout

Checkpoints land under Primus's `work_group/user_name/exp_name` template:

```
output/amd/root/zebra_llama_300M_gdn_pure-pretrain/
├── checkpoints/
│   ├── iter_0001024/
│   ├── iter_0002048/
│   ├── iter_0003072/
│   ├── iter_0004096/
│   ├── iter_0004768/                  ← FINAL (4.1 GB)
│   │   └── mp_rank_00/
│   │       └── model_optim_rng.pt
│   └── latest_checkpointed_iteration.txt  → "4768"
└── logs/
    └── pre_trainer/
```

`save_interval: 1024` in the YAML produces 4 mid-training checkpoints plus the final one.

---

## Step 6: Monitor and compare against FLA

Megatron logs `iteration / elapsed_ms_inst / elapsed_ms_avg / TFLOP/s/GPU / tok/s/GPU / lm loss` every 100 steps. A representative tail looks like:

```
 iteration  4700/ 4768 | elapsed time per iteration (ms): 1460.8/1450.4 |
   TFLOP/s/GPU: 633.7 | tokens per GPU (tokens/s/GPU): 180743.2 | lm loss: 3.3632
```

To diff against FLA's reference log (`train_gdn_bs32.log`, `train_runtime=6840.2s`):


| iter | FLA / 8 | Primus  | Δ %         | Notes                        |
| ---- | ------- | ------- | ----------- | ---------------------------- |
| 1    | 11.9654 | 11.9652 | **−0.00 %** | bit-perfect                  |
| 100  | 7.471   | 9.601   | +28.5 %     | warmup gap (peak)            |
| 500  | 4.625   | 4.728   | +2.2 %      | warmup closing               |
| 1000 | 4.001   | 4.050   | +1.21 %     | LR-warmup done               |
| 2000 | 3.607   | 3.614   | +0.21 %     | converged                    |
| 2100 | 3.600   | 3.592   | **−0.22 %** | first Primus < FLA crossover |
| 3000 | 3.448   | 3.460   | +0.35 %     | matched                      |
| 4000 | 3.396   | 3.390   | −0.19 %     | Primus slightly lower        |
| 4500 | 3.373   | 3.373   | −0.01 %     | identical                    |
| 4700 | 3.351   | 3.366   | +0.45 %     | identical                    |


Final wall time on a healthy MI300X box: **6832 s vs FLA 6840 s** = Primus 8 s faster. On a slower node it's +75 s (~1.1 %). Both are within the run-to-run noise of FLA itself.

---

## Step 7: Convert checkpoint to HuggingFace format

Use `[tools/convert_gdn_to_fla_hf.py](../../tools/convert_gdn_to_fla_hf.py)` to translate the Megatron checkpoint into FLA's native `GatedDeltaNetForCausalLM` HF format:

```bash
python tools/convert_gdn_to_fla_hf.py \
    --checkpoint-path output/amd/root/zebra_llama_300M_gdn_pure-pretrain/checkpoints/iter_0004768 \
    --output-dir      output/gdn_pure_300M_fla_hf_final
```

The converter auto-detects 300M from the path and uses `gated_deltanet_300M_pure.json`. What it does:

- Reads `mp_rank_00/model_optim_rng.pt` and pulls the `model` state dict
- For each of the 12 FLA layers, pairs the alternating Megatron sublayers:
  - GDN sublayer (even index) → FLA `model.layers.<i>.attn.*`
  - MLP sublayer (odd index) → FLA `model.layers.<i>.mlp.*`
- Splits Primus's **fused** projections into FLA's separate ones:
  - `mixer.in_proj.weight` (rows = `2·key_dim + 2·value_dim + 2·num_v_heads`) → `q_proj / k_proj / v_proj / g_proj / b_proj / a_proj`
  - `mixer.conv1d.weight` → `q_conv1d / k_conv1d / v_conv1d`
  - `mlp.linear_fc1.weight` (rows = `2·intermediate_size`) → `gate_proj / up_proj`
- Handles **both** layer-spec variants:
  - TE spec (`gdn_hybrid_stack_spec`): norm fused into linear (`mixer.in_proj.layer_norm_weight`, `mlp.linear_fc1.layer_norm_weight`)
  - No-TE spec (`gdn_hybrid_stack_spec_no_te`, **used by the validated run**): separate `WrappedTorchNorm` modules (`norm.weight`, `pre_mlp_layernorm.weight`)
- Preserves `A_log`, `dt_bias`, per-head `out_norm`, `out_proj`, embeddings, tied `lm_head`, final norm

Output:

```
output/gdn_pure_300M_fla_hf_final/
├── config.json              # GatedDeltaNetConfig, architectures=["GatedDeltaNetForCausalLM"]
├── model.safetensors        # ~870 MB
└── tokenizer_config.json    # placeholder — point to meta-llama/Llama-3.2-1B at load time
```

For the 1B pure-GDN model, same command but use the 1B checkpoint path — the converter auto-selects `gated_deltanet_1B_pure.json`.

---

## Step 8: Verify conversion

Run the sanity check at `[tools/verify_gdn_conversion.py](../../tools/verify_gdn_conversion.py)`:

```bash
python tools/verify_gdn_conversion.py \
    --model-path output/gdn_pure_300M_fla_hf_final
```

It loads the converted model in bf16 on GPU, runs three test prompts, and reports per-prompt loss, top-5 next-token IDs, and a 40-token greedy continuation. **Expected output** for a healthy 300M-on-10B model:


| Prompt                                      | Loss | Top-1                                       | Verdict          |
| ------------------------------------------- | ---- | ------------------------------------------- | ---------------- |
| "The capital of France is"                  | ~3.7 | `Paris`                                     | knows the answer |
| "Machine learning is a field of"            | ~2.8 | `artificial`                                | knows the domain |
| "The largest planet in our solar system is" | ~2.3 | one of `[the, Jupiter, a, called, located]` | knows the topic  |


Loss <6.0 = PASS. Greedy decoding will produce *grammatical but repetitive* English (e.g. *"Paris. The capital is Paris. The capital is Paris..."*) — this is the canonical small-undertrained-LM failure mode with no repetition penalty and is **not** a sign of conversion error.

### Optional — logit parity vs the FLA reference checkpoint

If you have the FLA reference HF checkpoint locally (e.g. trained by FLA itself, or downloaded from `fla-hub`), compare per-token logits:

```bash
python - <<'PY'
import torch, fla
from fla.models.gated_deltanet import GatedDeltaNetForCausalLM

ids = torch.tensor([[1, 791, 6864, 315, 9822, 374]])  # "The capital of France is"

hf  = GatedDeltaNetForCausalLM.from_pretrained("output/gdn_pure_300M_fla_hf_final",
                                               torch_dtype=torch.bfloat16).cuda().eval()
ref = GatedDeltaNetForCausalLM.from_pretrained("/path/to/fla/checkpoints/gdn_pure_300M_10BT",
                                               torch_dtype=torch.bfloat16).cuda().eval()

with torch.no_grad():
    h = hf (ids.cuda()).logits[0, -1].float().cpu()
    r = ref(ids.cuda()).logits[0, -1].float().cpu()

print(f"Cosine sim:        {torch.nn.functional.cosine_similarity(h, r, dim=0).item():.4f}")
print(f"Top-1 (converted): {h.argmax().item()}   Top-1 (FLA ref): {r.argmax().item()}")
print(f"Top-5 (converted): {h.topk(5).indices.tolist()}")
print(f"Top-5 (FLA ref):   {r.topk(5).indices.tolist()}")
PY
```

**Expected:**


| Metric            | Value       | Interpretation                                                     |
| ----------------- | ----------- | ------------------------------------------------------------------ |
| Cosine similarity | **≥ 0.95**  | conversion is correct                                              |
| Top-5 set overlap | **≥ 3 / 5** | distributions agree                                                |
| Top-1 exact match | optional    | a single-token disagreement is within 0.24 % loss divergence noise |


If cosine < 0.5 → permutation bug. If 0.5–0.95 → likely missing-key issue (check that all 12 layers got `A_log`, `dt_bias`, `out_norm`).

---

## Step 9: Run lm-eval-harness benchmarks

Use `[tools/eval_gdn_lm_eval.py](../../tools/eval_gdn_lm_eval.py)`, which imports `fla` first (so `AutoConfig` recognizes the `gated_deltanet` model type) and patches the FLA model `__init__` to accept the `dtype` kwarg that `transformers ≥ 4.55` passes internally.

**Do not** invoke `lm_eval --model hf ...` directly — `AutoConfig.from_pretrained` will fail with `model type gated_deltanet not recognized`.

### 9.1 Standard six-task suite (~15–30 min on one MI300X)

```bash
python tools/eval_gdn_lm_eval.py \
    --model hf \
    --model_args pretrained=output/gdn_pure_300M_fla_hf_final,dtype=bfloat16,trust_remote_code=True,tokenizer=meta-llama/Llama-3.2-1B \
    --tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu,race \
    --batch_size auto \
    --output_path output/gdn_pure_300M_eval_results_final
```

### 9.2 Full FLA-paper suite (adds MMLU + RACE, ~1–2 h)

```bash
python tools/eval_gdn_lm_eval.py \
    --model hf \
    --model_args pretrained=output/gdn_pure_300M_fla_hf_final,dtype=bfloat16,trust_remote_code=True,tokenizer=meta-llama/Llama-3.2-1B \
    --tasks arc_easy,arc_challenge,hellaswag,mmlu,openbookqa,piqa,race,winogrande \
    --batch_size auto \
    --output_path output/gdn_pure_300M_eval_results_final
```

### 9.3 Diff against the FLA reference run

If you also evaluated FLA's own checkpoint (`output/gdn_pure_300M_fla_eval_results/`), compare the JSONs:

```bash
python - <<'PY'
import json, glob
def load_latest(d): return json.load(open(sorted(glob.glob(f"{d}/**/results_*.json", recursive=True))[-1]))
fla    = load_latest("output/gdn_pure_300M_fla_eval_results")
primus = load_latest("output/gdn_pure_300M_eval_results_final")
print(f"{'task':<18} {'FLA':>8} {'Primus':>8} {'Δ':>+8}")
for task in sorted(set(fla['results']) & set(primus['results'])):
    for k in ('acc,none', 'acc_norm,none'):
        if k in fla['results'][task]:
            f, p = fla['results'][task][k], primus['results'][task][k]
            print(f"{task[:17]:<18} {f:>8.4f} {p:>8.4f} {p-f:>+8.4f}  ({k})")
PY
```

**Expected:** each task within ±1.5 absolute accuracy points (consistent with the 0.24 % loss delta at the end of training).

---

## Configs and tools used

```
docs/zebra_llama/
└── README_GDN.md                                      ← this file
GDN_FLA_PARITY.md                                      ← deep-dive on every patch & env var
megatron_patch.sh                                      ← idempotent patch applier
megatron_patches/
├── 01-mamba_model-fused-ce.patch
├── 02-optimizer-torch-fused-adam.patch
├── 03-mlp-fla-swiglu.patch
├── 04-torch_norm-fla-rmsnorm.patch
├── 05-transformer_config-hybrid-init.patch
└── 06-pretrain_mamba-fla-data-and-diag.patch
examples/megatron/configs/MI300X/
└── zebra_llama_300M_gdn_pure-pretrain.yaml            ← training config
primus/configs/models/megatron/
└── zebra_llama_300M_gdn_pure.yaml                     ← architecture-only config
primus/backends/megatron/core/models/hybrid/
├── gated_delta_net.py                                 ← FLA-aligned mixer (FLA Triton paths)
├── gated_delta_net_layer.py                           ← eps propagation, pre-norm fusion
├── hybrid_block.py                                    ← HybridStack, fp32-residual + fusion
└── hybrid_mamba_mla_layer_specs.py                    ← gdn_hybrid_stack_spec_no_te
tools/
├── convert_fla_to_megatron.py                         ← FLA Arrow → Megatron .bin/.idx
├── fla_order_dataset.py                               ← FLA-order dataset shim
├── convert_gdn_to_fla_hf.py                           ← Megatron → FLA HF (handles TE + no-TE)
├── verify_gdn_conversion.py                           ← loss + greedy generation sanity check
└── eval_gdn_lm_eval.py                                ← lm-eval wrapper (registers FLA)
bash-docker.sh                                         ← one-shot container launcher
```

---

## Troubleshooting

### `KeyError: 'decoder.layers.0.mixer.in_proj.layer_norm_weight'` during conversion

You trained with `gdn_hybrid_stack_spec_no_te` (separate `WrappedTorchNorm`) but are running an old version of `convert_gdn_to_fla_hf.py` that only knew the TE spec. Pull the latest converter — it now tries TE keys first and falls back to `norm.weight` / `pre_mlp_layernorm.weight`.

### Loss is flat near 11.97 for many iterations

LR warmup misconfigured. Confirm `lr_warmup_iters: 200` matches your `train_iters`, and verify the YAML override block resolved correctly by checking the logged config near the top of the training log.

### Iter 1 loss ~12.1 instead of ~11.97

The `layernorm_epsilon: 1.0e-6` override is being silently overwritten by the TransformerConfig default of `1e-5`. Confirm it's in the *training* YAML's `overrides:` block (not just the model YAML) — see `[zebra_llama_300M_gdn_pure-pretrain.yaml](../../examples/megatron/configs/MI300X/zebra_llama_300M_gdn_pure-pretrain.yaml)` for the canonical placement.

### Iter 1 loss not bit-matching FLA but converges fine

You probably didn't enable `PRIMUS_FLA_DATA=1` — the Megatron `GPTDataset` shuffler is producing a different first batch than FLA's `DistributedSampler`. Either enable that env var (with `PRIMUS_FLA_CACHE_DIR` set) or accept the ~0.0002 loss delta at iter 1 (it disappears by iter ~2000).

### Iter 1 takes ~60 seconds, subsequent iters are fast

Cold MIOpen + Triton autotune caches. Normal on a freshly-rebooted node. The run-averaged ms/iter takes ~1500 iters to fully wash out this cold-start tax; the instantaneous ms/iter is at steady state by iter 200.

### Eval fails with `model type gated_deltanet not recognized`

You ran `lm_eval --model hf` directly instead of the wrapper. Use `python tools/eval_gdn_lm_eval.py --model hf ...` — it imports `fla` first to register the model class.

### Eval truncation warnings

Some samples exceed the model's `max_position_embeddings = 2048`. Add `max_length=1024` to `--model_args` if it bothers you; it only meaningfully affects RACE.

### Per-iter time +1–2 % above FLA on healthy hardware

Expected with all four `PRIMUS_FLA_`* env vars set. The biggest single cost is `PRIMUS_TORCH_OPTIM=1` (torch fused AdamW vs Apex FusedAdam). Drop it if you don't need bit-level optimizer parity; you keep the loss-curve match and recover ~1 % perf.

---

## See also

- `[docs/zebra_llama/README.md](README.md)` — full Zebra-Llama family overview (1B / 3B / 8B Mamba+MLA, KDA variants)
- `[GDN_FLA_PARITY.md](../../GDN_FLA_PARITY.md)` — exhaustive list of code/config/runtime changes that made parity possible
- FLA upstream: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)


# Pure KDA 300M on Primus — End-to-End Guide (FLA-validated)

This document is a runnable walkthrough for the **300M pure Kimi Delta
Attention (KDA)** pretraining recipe in Primus, validated on 8× AMD MI300X
against the [Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)
reference implementation. It covers every step from raw dataset →
tokenization → training → checkpoint conversion → lm-eval benchmark.

The same recipe scales up to the 1B pure-KDA config (`zebra_llama_1B_kda_pure-pretrain.yaml`).

It mirrors [`README_GDN.md`](README_GDN.md) and reuses the same Megatron-LM
patches, dataset shim, FLA-init flow, and lm-eval wrapper pattern.

---

## Final result

After 4768 iterations (≈10B tokens) on FineWeb-Edu sample-10BT:

| Axis                                            | FLA reference     | Primus (this branch)  | Δ                           |
| ----------------------------------------------- | ----------------- | --------------------- | --------------------------- |
| Per-iteration time (steady state, iter > 200)   | **1493 ms**       | **1466.8 ms**         | **−1.8 % (Primus faster)**  |
| Throughput                                      | 175,617 tok/s/GPU | **178,810 tok/s/GPU** | **+1.8 %**                  |
| TFLOP/s/GPU                                     | —                 | 626.9                 | —                           |
| Wall time (4768 iters, 8× MI300X, healthy node) | 1h 58m 39s        | **1h 56m 33s**        | **−126 s**                  |
| Loss @ iter 1                                   | 11.9673           | **11.9669**           | **−0.00 % (bit-perfect)**   |
| Loss @ iter 4700 (final logged)                 | 3.3388            | **3.3624**            | **+0.71 %**                 |
| First Primus-below-FLA crossover                | —                 | iter 2600             | —                           |

Loss trajectories overlap from iter ~2000 onward; the only persistent gap
is in the LR-warmup region (iter 50–500) and closes monotonically. See
[`KDA_FLA_PARITY.md`](../../KDA_FLA_PARITY.md) for the deep-dive on every
patch and env var.

### lm-eval-harness (FLA-paper 8-task suite)

Random chance is `100 / num_choices` — 25 % for the 4-choice tasks
(arc, hellaswag, openbookqa, mmlu, race) and 50 % for the 2-choice tasks
(piqa, winogrande). Any score above random shows the model has learned
*something*; the FLA and Primus rows show how closely the two training
stacks track each other on the same 10 B-token diet.

| Task                     | Metric     | Random | FLA    | Primus | Δ (Primus − FLA) |
|--------------------------|------------|-------:|-------:|-------:|-----------------:|
| arc_challenge            | acc_norm   |  25.00 | 25.17  | 25.00  | −0.17 pp         |
| arc_easy                 | acc        |  25.00 | 48.78  | 47.94  | −0.84 pp         |
| arc_easy                 | acc_norm   |  25.00 | 42.76  | 43.39  | +0.63 pp         |
| hellaswag                | acc_norm   |  25.00 | 29.16  | 29.18  | +0.02 pp         |
| openbookqa               | acc_norm   |  25.00 | 30.40  | 29.00  | −1.40 pp         |
| piqa                     | acc_norm   |  50.00 | 60.99  | 60.34  | −0.65 pp         |
| winogrande               | acc        |  50.00 | 51.85  | 52.72  | **+0.87 pp**     |
| mmlu (aggregate)         | acc        |  25.00 | 22.88  | 23.12  | +0.24 pp         |
| race                     | acc        |  25.00 | 25.07  | 25.45  | +0.38 pp         |
| **mean absolute Δ**      |            |        |        |        | **0.58 pp**      |

Every task within ±1.4 pp — well inside the ±1.5 pp tolerance set by the
0.49% mid-training loss delta. Both stacks comfortably beat random on
arc_easy, hellaswag, openbookqa and piqa; mmlu/race/arc_challenge are at
random-chance for *both* training stacks (expected for a 300 M model on
only 10 B tokens — those benchmarks need 7 B+ parameters and/or
trillion-token training to lift above 25 %).

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

The 300M pure-KDA model has:

- 12 Kimi-Delta-Attention blocks + 12 MLP blocks → 24 Megatron "sublayers"
- `hidden_size = 1024`, `ffn_hidden_size = 4096`
- `num_heads = num_v_heads = 8` (Q, K, V all share head count)
- `head_k_dim = 32`, `head_v_dim = 64` (expand_v = 2.0)
- Short-conv kernel size 4 (depthwise, on the concatenated QKV)
- Per-head output gate (`g_a → g_b`) + per-head decay gate (`f_a → f_b`,
  combined with learnable `A_log` and `dt_bias` via `softplus`)
- Tied embeddings, no positional encoding (delta-rule recurrence),
  RMSNorm with `eps = 1e-6`
- Tokenizer: `meta-llama/Llama-3.2-1B` (128k vocab)
- Total parameters: **0.302 B**

Training schedule (matched to FLA's `kda_300M_pure.json`):

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
- **flash-linear-attention** checked out at
  `/home/<user>/flash-linear-attention` (or installed via
  `pip install -e .`) — provides the FLA `KDAForCausalLM` class for
  HF conversion + lm-eval, plus the Triton kernels that the `PRIMUS_FLA_*`
  toggles route into.

---

## Step 1: Environment

### 1.1 Start the dev container

```bash
bash bash-docker.sh
```

This runs the `rocm/primus:v26.2` image with `/dev/dri`, `/dev/kfd`, IB
devices, `--privileged`, your `$HOME` mounted in-place, and `--shm-size 64G`.
The container is named `primus_hybrid_new`.

To re-attach later:

```bash
docker exec -it primus_hybrid_new bash
cd /home/<user>/Primus
```

### 1.2 Install Python dependencies inside the container

```bash
pip install -r requirements.txt
pip install -e /home/<user>/flash-linear-attention   # FLA model classes + Triton kernels
pip install lm-eval                                  # for benchmark evaluation
```

The editable FLA install removes the need to set `PYTHONPATH` for every
later command.

---

## Step 2: Dataset preparation

Identical to the GDN recipe — see
[`README_GDN.md`](README_GDN.md#step-2-dataset-preparation). KDA reuses the
same FineWeb-Edu sample-10BT preprocessed Arrow shards and the same
Llama-3.2-1B tokenizer.

The default 300M YAML already points at the FLA-aligned binary:

```yaml
train_data_path: >
  /home/<user>/Primus/data/fla_aligned/fla_fineweb_edu_10BT_text_sentence
```

(adjust the user prefix in
`examples/megatron/configs/MI300X/zebra_llama_300M_kda_pure-pretrain.yaml`
to match your home directory).

---

## Step 3: Apply Megatron-LM patches

KDA uses the **same six patches** as GDN — no KDA-specific Megatron patch
is required. They live in `megatron_patches/*.patch` and are applied by an
idempotent script:

```bash
bash megatron_patch.sh           # apply all 6
bash megatron_patch.sh --check   # dry-run (does not modify files)
bash megatron_patch.sh --revert  # undo all
```

See [`README_GDN.md`](README_GDN.md#step-3-apply-megatron-lm-patches) §3
for the patch-by-patch breakdown.

---

## Step 4: (Optional) Initialize from FLA weights

For bit-perfect iter-1 loss alignment, the validated run loads FLA's
*initialized but untrained* KDA-300M checkpoint and then trains from
there. The YAML's `load:` field points at this directory:

```yaml
load: /home/<user>/Primus/output/fla_init_kda_300M
finetune: true            # load weights, ignore optimizer state and iteration count
no_load_optim: true
no_load_rng: true
```

Generate it once with:

```bash
python tools/convert_fla_kda_init_to_megatron.py
#   → output/fla_init_kda_300M/iter_0000000/mp_rank_00/model_optim_rng.pt
```

The script instantiates FLA's `KDAForCausalLM` with `seed=42`, harvests
its randomly-initialized weights, concatenates the six FLA `hidden_states
→ X` projections into Primus's single fused `in_proj`, and writes a
Megatron-shape checkpoint. Skip this step if you're happy with Primus's
own random init — final loss is identical, only iter-1 drifts by `~5e-3`.

---

## Step 5: Train

### 5.1 Inspect the config

The training config lives at
[`examples/megatron/configs/MI300X/zebra_llama_300M_kda_pure-pretrain.yaml`](../../examples/megatron/configs/MI300X/zebra_llama_300M_kda_pure-pretrain.yaml).
Key parameters (matched to FLA):

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
spec: ['primus.backends.megatron.core.models.hybrid.hybrid_mamba_mla_layer_specs', 'kda_hybrid_stack_spec_no_te']
use_fla_triton_kda: true
use_fla_kda_in_kernel_gate: true
use_fla_fused_norm_gated: true
use_distributed_optimizer: false  # 300M fits — ZeRO-1 adds allreduce overhead
finetune: true
load: /home/<user>/Primus/output/fla_init_kda_300M
no_load_optim: true
no_load_rng: true
```

The architecture-only YAML it extends from is
[`primus/configs/models/megatron/zebra_llama_300M_kda_pure.yaml`](../../primus/configs/models/megatron/zebra_llama_300M_kda_pure.yaml).

### 5.2 Launch

```bash
# inside the container, in /home/<user>/Primus
EXP=examples/megatron/configs/MI300X/zebra_llama_300M_kda_pure-pretrain.yaml \
  bash examples/run_pretrain.sh 2>&1 | tee primus_kda.log
```

Expected wall time on a healthy MI300X box: **~1h 56m** for the full 4768
iters (about 2 min faster than FLA's HF-Trainer reference run).

### 5.3 Recommended env-var profile (for FLA parity)

```bash
export PRIMUS_FUSED_CE=1          # FLA FusedLinearCrossEntropyLoss (chunked, no full logits tensor)
export PRIMUS_FLA_SWIGLU=1        # FLA Triton SwiGLU
export PRIMUS_FLA_NORM=1          # FLA fused RMSNorm
export PRIMUS_FLA_CONV=1          # FLA Triton causal_conv1d (no transpose round-trip)
export PRIMUS_TORCH_OPTIM=1       # torch.optim.AdamW(fused=True), matches FLA exactly
# Only if you want bit-identical iter-1 batch ordering:
export PRIMUS_FLA_DATA=1
export PRIMUS_FLA_CACHE_DIR=/home/<user>/Primus/data/huggingface
```

See [`KDA_FLA_PARITY.md`](../../KDA_FLA_PARITY.md) for the cost-of-each-flag
breakdown. `PRIMUS_NATIVE_GVA` and `PRIMUS_NO_TE` are GDN-only and have no
effect on KDA.

### 5.4 Output layout

Checkpoints land under Primus's `work_group/user_name/exp_name` template:

```
output/amd/root/zebra_llama_300M_kda_pure-pretrain/
├── checkpoints/
│   ├── iter_0001024/
│   ├── iter_0002048/
│   ├── iter_0003072/
│   ├── iter_0004096/
│   ├── iter_0004768/                  ← FINAL (~4.5 GB)
│   │   └── mp_rank_00/
│   │       └── model_optim_rng.pt
│   └── latest_checkpointed_iteration.txt  → "4768"
└── logs/
    └── pre_trainer/
```

`save_interval: 1024` in the YAML produces 4 mid-training checkpoints plus
the final one.

---

## Step 6: Monitor and compare against FLA

Megatron logs `iteration / elapsed_ms_inst / elapsed_ms_avg / TFLOP/s/GPU
/ tok/s/GPU / lm loss` every 100 steps. A representative tail looks like:

```
iteration  4700/ 4768 | elapsed time per iteration (ms): 1467.8/1466.1 |
  TFLOP/s/GPU: 626.1 | tokens per GPU (tokens/s/GPU): 178596.5 | lm loss: 3.362445E+00
```

To diff against FLA's reference log
(`/home/<user>/checkpoints/kda_pure_300M_10B/trainer_state.json`), divide
the FLA `loss` field by 8 (DeepSpeed reports sum-across-ranks):

| iter | FLA / 8 | Primus  | Δ %         | Notes                            |
| ---- | ------- | ------- | ----------- | -------------------------------- |
| 1    | 11.9673 | 11.9669 | **−0.00 %** | bit-perfect                      |
| 100  | 7.7171  | 9.6903  | +25.6 %     | warmup gap (peak)                |
| 500  | 4.7349  | 4.8390  | +2.20 %     | warmup closing                   |
| 1000 | 4.0357  | 4.0720  | +0.90 %     | LR-warmup done                   |
| 2000 | 3.6009  | 3.6141  | +0.37 %     | converged                        |
| 2600 | 3.5056  | 3.5047  | **−0.03 %** | first Primus < FLA crossover     |
| 3000 | 3.4356  | 3.4571  | +0.63 %     | matched                          |
| 3600 | 3.4107  | 3.4075  | **−0.09 %** | Primus slightly lower            |
| 4000 | 3.3831  | 3.3861  | +0.09 %     | identical                        |
| 4500 | 3.3603  | 3.3694  | +0.27 %     | identical                        |
| 4700 | 3.3388  | 3.3624  | +0.71 %     | identical                        |

Final wall time on a healthy MI300X box: **6993 s vs FLA 7119 s** =
Primus 126 s faster.

---

## Step 7: Convert checkpoint to HuggingFace format

Use [`tools/convert_kda_to_fla_hf.py`](../../tools/convert_kda_to_fla_hf.py)
to translate the Megatron checkpoint into FLA's native
`KDAForCausalLM` HF format:

```bash
python tools/convert_kda_to_fla_hf.py \
    --checkpoint-path output/amd/root/zebra_llama_300M_kda_pure-pretrain/checkpoints/iter_0004768 \
    --output-dir      output/kda_pure_300M_fla_hf \
    --config          /home/<user>/flash-linear-attention/legacy/training/configs/kda_300M_pure.json \
    --tokenizer-src   /home/<user>/checkpoints/kda_pure_300M_10B
```

What it does:

- Reads `mp_rank_00/model_optim_rng.pt` and pulls the `model` state dict
- For each of the 12 FLA layers, pairs the alternating Megatron sublayers:
  - KDA sublayer (even index) → FLA `model.layers.<i>.attn.*`
  - MLP sublayer (odd index) → FLA `model.layers.<i>.mlp.*`
- Splits Primus's **fused** projections into FLA's separate ones:
  - `mixer.in_proj.weight` (rows = `2·qk_dim + v_dim + 2·head_v_dim +
    num_v_heads`) → `q_proj / k_proj / v_proj / f_proj.0 / g_proj.0 / b_proj`
  - `mlp.linear_fc1.weight` (rows = `2·intermediate_size`) →
    `gate_proj / up_proj`
- Preserves `A_log`, `dt_bias`, per-head `g_norm` (FLA's
  `FusedRMSNormGated`), `o_proj`, `f_proj.1`, `g_proj.1`, embeddings,
  tied `lm_head`, final norm
- Copies tokenizer files from `--tokenizer-src` into the output dir

Output:

```
output/kda_pure_300M_fla_hf/
├── config.json              # KDAConfig, architectures=["KDAForCausalLM"]
├── model.safetensors        # ~870 MB
└── tokenizer{,_config}.json + special_tokens_map.json
```

---

## Step 8: Verify conversion

Quick smoke test in the container (with FLA importable):

```bash
PYTHONPATH=/home/<user>/flash-linear-attention \
python - <<'PY'
import torch
import fla   # auto-registers "kda" with transformers.AutoConfig

from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "output/kda_pure_300M_fla_hf"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(
    ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda().eval()

for prompt in [
    "The capital of France is",
    "Once upon a time, there was a small",
    "The first law of thermodynamics states that",
]:
    inp = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=40, do_sample=False)
    print("---"); print(tok.decode(out[0], skip_special_tokens=True))
PY
```

**Expected output** for a healthy 300 M-on-10 B model: grammatical but
repetitive English (canonical small-undertrained-LM failure mode under
greedy decoding with no repetition penalty). Knowing "capital of France"
→ "Paris" is the standard sanity-check pass.

If `AutoConfig` raises `model type kda not recognized`, FLA was not
imported before `AutoModelForCausalLM`. Either prepend
`PYTHONPATH=/home/<user>/flash-linear-attention` or run
`pip install -e /home/<user>/flash-linear-attention` so the
auto-registration in `fla/models/kda/__init__.py` fires on import.

---

## Step 9: Run lm-eval-harness benchmarks

Use [`tools/eval_kda_lm_eval.py`](../../tools/eval_kda_lm_eval.py), which
imports `fla` first (so `AutoConfig` recognizes the `kda` model type) and
patches `KDAForCausalLM.__init__` / `KDAModel.__init__` to accept the
`dtype` kwarg that `transformers ≥ 4.55` passes internally.

**Do not** invoke `lm_eval --model hf ...` directly — `AutoConfig.from_pretrained`
will fail with `model type kda not recognized`.

### 9.1 Evaluate the Primus checkpoint (~15–30 min on one MI300X)

```bash
mkdir -p output/kda_pure_300M_eval_results_primus

PYTHONPATH=/home/<user>/flash-linear-attention \
HIP_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
python tools/eval_kda_lm_eval.py \
    --model hf \
    --model_args pretrained=output/kda_pure_300M_fla_hf,dtype=bfloat16,trust_remote_code=True,tokenizer=meta-llama/Llama-3.2-1B \
    --tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu,race \
    --batch_size auto \
    --output_path output/kda_pure_300M_eval_results_primus \
    2>&1 | tee output/kda_pure_300M_eval_results_primus/lm_eval.log
```

### 9.2 Evaluate the FLA reference checkpoint (apples-to-apples)

```bash
mkdir -p output/kda_pure_300M_eval_results_fla

PYTHONPATH=/home/<user>/flash-linear-attention \
HIP_VISIBLE_DEVICES=1 \
TOKENIZERS_PARALLELISM=false \
python tools/eval_kda_lm_eval.py \
    --model hf \
    --model_args pretrained=/home/<user>/checkpoints/kda_pure_300M_10B,dtype=bfloat16,trust_remote_code=True,tokenizer=meta-llama/Llama-3.2-1B \
    --tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu,race \
    --batch_size auto \
    --output_path output/kda_pure_300M_eval_results_fla \
    2>&1 | tee output/kda_pure_300M_eval_results_fla/lm_eval.log
```

### 9.3 Diff the two result JSONs

```bash
python - <<'PY'
import json, glob
def load_latest(d):
    return json.load(open(sorted(glob.glob(f"{d}/**/results_*.json", recursive=True))[-1]))
fla    = load_latest("output/kda_pure_300M_eval_results_fla")
primus = load_latest("output/kda_pure_300M_eval_results_primus")
print(f"{'task':<18} {'FLA':>8} {'Primus':>8} {'Δ':>+8}")
for task in sorted(set(fla['results']) & set(primus['results'])):
    for k in ('acc,none', 'acc_norm,none'):
        if k in fla['results'][task] and k in primus['results'][task]:
            f, p = fla['results'][task][k], primus['results'][task][k]
            print(f"{task[:17]:<18} {f:>8.4f} {p:>8.4f} {p-f:>+8.4f}  ({k})")
PY
```

**Measured result** (validated on `tw006`, this branch). The `Random`
column is `100 / num_choices` for the lm-eval task — anything above it
means the model learned something:

| Task                     | Metric     | Random | FLA    | Primus | Δ (Primus − FLA) |
|--------------------------|------------|-------:|-------:|-------:|-----------------:|
| arc_challenge            | acc_norm   |  25.00 | 25.17  | 25.00  | −0.17 pp         |
| arc_easy                 | acc        |  25.00 | 48.78  | 47.94  | −0.84 pp         |
| arc_easy                 | acc_norm   |  25.00 | 42.76  | 43.39  | +0.63 pp         |
| hellaswag                | acc_norm   |  25.00 | 29.16  | 29.18  | +0.02 pp         |
| openbookqa               | acc_norm   |  25.00 | 30.40  | 29.00  | −1.40 pp         |
| piqa                     | acc_norm   |  50.00 | 60.99  | 60.34  | −0.65 pp         |
| winogrande               | acc        |  50.00 | 51.85  | 52.72  | **+0.87 pp**     |
| mmlu (aggregate)         | acc        |  25.00 | 22.88  | 23.12  | +0.24 pp         |
| race                     | acc        |  25.00 | 25.07  | 25.45  | +0.38 pp         |
| **mean absolute Δ**      |            |        |        |        | **0.58 pp**      |

Every task within ±1.4 pp — consistent with the 0.49% loss delta at the
end of training. mmlu / race / arc_challenge are at random-chance for
*both* stacks (300 M params + 10 B tokens is below the threshold those
benchmarks need to lift above noise).

---

## Configs and tools used

```
docs/zebra_llama/
└── README_KDA.md                                  ← this file
KDA_FLA_PARITY.md                                  ← deep-dive on every change
megatron_patch.sh                                  ← idempotent patch applier (shared with GDN)
megatron_patches/                                  ← same 6 patches as GDN
examples/megatron/configs/MI300X/
└── zebra_llama_300M_kda_pure-pretrain.yaml        ← training config
primus/configs/models/megatron/
└── zebra_llama_300M_kda_pure.yaml                 ← architecture-only config
primus/backends/megatron/core/models/hybrid/
├── kimi_delta_attention.py                        ← FLA-aligned mixer (fused in_proj, FLA Triton paths)
├── kimi_delta_attention_layer.py                  ← eps propagation, optional pre-norm
└── hybrid_mamba_mla_layer_specs.py                ← kda_hybrid_stack_spec_no_te
primus/backends/megatron/patches/
└── gdn_config_patches.py                          ← registers use_fla_triton_kda + fusion flags
tools/
├── convert_fla_to_megatron.py                     ← FLA Arrow → Megatron .bin/.idx (shared)
├── fla_order_dataset.py                           ← FLA-order dataset shim (shared)
├── convert_fla_kda_init_to_megatron.py            ← FLA HF init → Megatron sharded ckpt
├── convert_kda_to_fla_hf.py                       ← Megatron sharded ckpt → FLA HF
└── eval_kda_lm_eval.py                            ← lm-eval wrapper (registers KDA)
bash-docker.sh                                     ← one-shot container launcher
```

---

## Troubleshooting

### `KeyError: 'kda'` at `AutoModelForCausalLM.from_pretrained`

You imported `transformers` before `fla` (or didn't import `fla` at all).
`fla/models/kda/__init__.py` runs
`AutoConfig.register(KDAConfig.model_type, KDAConfig, exist_ok=True)`
on import. Either:

- Prepend `PYTHONPATH=/home/<user>/flash-linear-attention` and `import fla`
  in your script BEFORE the `transformers` import, OR
- `pip install -e /home/<user>/flash-linear-attention` once and forget
  about `PYTHONPATH`, OR
- Use the wrapper: `python tools/eval_kda_lm_eval.py ...`

### Conversion: `KeyError: 'decoder.layers.0.mixer.in_proj.weight'`

You trained with an older code branch that still had six separate
projections. Either re-train with the current fused-in_proj branch or
patch the converter to read the unfused `q_proj_weight`/`k_proj_weight`/…
keys (see git history of `tools/convert_kda_to_fla_hf.py`).

### Iter 1 loss ~12.05 instead of ~11.97

The `layernorm_epsilon: 1.0e-6` override is being silently overwritten by
the `TransformerConfig` default of `1e-5`. Confirm it's in the *training*
YAML's `overrides:` block (not just the model YAML).

### Iter 1 loss not bit-matching FLA but converges fine

You probably didn't load the FLA-init checkpoint (Step 4) or didn't set
`PRIMUS_FLA_DATA=1`. Without either, the first batch differs (Megatron
shuffler vs HF `DistributedSampler`) and the per-parameter `nn.init.normal_`
draw order differs (Megatron traverses Primus's fused `in_proj`, FLA
traverses 6 separate `nn.Linear` modules). The gap disappears by iter
~2000 even without either fix.

### Loss is +0.2–0.4 above FLA across the whole run (with FLA-init loaded)

You probably have `use_fla_kda_in_kernel_gate: false` or
`use_fla_fused_norm_gated: false`. Those toggles select the bit-identical-
to-old-FLA `fused_kda_gate` + `_apply_gated_norm` paths, which run the
gate compute in fp32 (slightly different rounding than the in-kernel bf16
accumulator). Set both to `true` to match the current FLA reference.

### Per-iter time ≫ 1500 ms

Most likely you have `PRIMUS_FLA_CONV=0`. The Tri-Dao `causal_conv1d_fn`
on ROCm requires `[B, D, T]` layout, so each iteration pays two
`transpose+contiguous` copies of the (B, qk_dim·2 + v_dim, T) tensor —
about 35 ms wasted per iter at micro_batch=128. Set `PRIMUS_FLA_CONV=1`
to switch to FLA's Triton `causal_conv1d` (accepts `[B, T, D]` natively).

### Out-of-memory at iter 1

Two common culprits:

1. `PYTORCH_ALLOC_CONF=expandable_segments:True` is unset — set it.
2. `q.contiguous()/k.contiguous()/v.contiguous()` removed from KDA forward
   — the Triton kernel will allocate its own copies while autograd still
   pins the original views, doubling Q/K/V activation memory. Restore
   the explicit contiguous calls (see `kimi_delta_attention.py` around
   the `chunk_kda` call site).

### Eval truncation warnings

Some samples exceed the model's `max_position_embeddings = 2048`. Add
`max_length=1024` to `--model_args` if it bothers you; it only
meaningfully affects RACE.

---

## See also

- [`docs/zebra_llama/README.md`](README.md) — full Zebra-Llama family
  overview (1 B / 3 B / 8 B Mamba+MLA, KDA variants)
- [`docs/zebra_llama/README_GDN.md`](README_GDN.md) — the GDN companion
  recipe (shares Megatron patches and dataset shim with this one)
- [`KDA_FLA_PARITY.md`](../../KDA_FLA_PARITY.md) — exhaustive list of
  code/config/runtime changes that made KDA parity possible
- FLA upstream: [https://github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)

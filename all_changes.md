# All Changes for FLA-Parity GDN + KDA Training in Primus

This document is the master changelog for every code, config, and tooling
change required to make **Gated DeltaNet (GDN)** and **Kimi Delta
Attention (KDA)** pretraining in Primus match the
[Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention)
reference implementation on **loss trajectory, step throughput, memory
footprint, and downstream lm-eval accuracy** on 8× AMD MI300X.

It is the union of the work documented in
[`GDN_FLA_PARITY.md`](GDN_FLA_PARITY.md) and
[`KDA_FLA_PARITY.md`](KDA_FLA_PARITY.md); read those for the per-architecture
deep dives. This file gives the per-file, per-line picture across the
whole codebase so a reviewer can audit the scope of the parity work in
one place.

---

## 1. Headline parity numbers

Both 300 M / 10 B-token runs on 8× MI300X (`tw006`), FLA-init checkpoint
loaded, full kernel fusions enabled:

| Recipe | ms/iter (Primus) | ms/iter (FLA) | Δ | tok/s/GPU (Primus) | lm-eval mean abs Δ (8 tasks) |
|--|--:|--:|--:|--:|--:|
| **GDN 300M** | 1431.6 | 1434.6 | **−0.21 %** | 183,213 | — (see GDN_FLA_PARITY) |
| **KDA 300M** | 1466.1 | 1493.1 | **−1.77 %** | 178,596 | **0.58 pp** |

Iter-1 loss is bit-perfect to FLA for both architectures when the FLA-init
checkpoint is loaded.

---

## 2. Repo-wide file inventory

The parity work spans **38 files** across the Primus working tree and **6
files** in the vendored Megatron-LM submodule. Grouped by kind:

### 2.1 New files (35)

| Category | Path | Lines | Purpose |
|--|--|--:|--|
| Patch infra | `megatron_patch.sh` | 195 | idempotent applier for the six Megatron-LM patches (apply/check/revert) |
| Patch infra | `bash-docker.sh` | 7 | one-shot `rocm/primus:v26.2` container launcher with the right `/dev/dri`, `/dev/kfd`, IB, `--shm-size` flags |
| Megatron patch | `megatron_patches/01-mamba_model-fused-ce.patch` | 79 | FLA `FusedLinearCrossEntropyLoss` / `FusedCrossEntropyLoss` integration |
| Megatron patch | `megatron_patches/02-optimizer-torch-fused-adam.patch` | 67 | opt-in `torch.optim.AdamW(fused=True)` over TE/Apex `FusedAdam` |
| Megatron patch | `megatron_patches/03-mlp-fla-swiglu.patch` | 60 | FLA Triton SwiGLU |
| Megatron patch | `megatron_patches/04-torch_norm-fla-rmsnorm.patch` | 21 | FLA Triton RMSNorm via `WrappedTorchNorm` |
| Megatron patch | `megatron_patches/05-transformer_config-hybrid-init.patch` | 30 | uniform (non-depth-scaled) init for hybrid models |
| Megatron patch | `megatron_patches/06-pretrain_mamba-fla-data-and-diag.patch` | 215 | FLA-order dataset shim + iter-1 batch / activation dumpers |
| Docs | `GDN_FLA_PARITY.md` | 218 | every change required for GDN parity (deep-dive) |
| Docs | `KDA_FLA_PARITY.md` | 274 | every change required for KDA parity (deep-dive) |
| Docs | `docs/zebra_llama/README.md` | 598 | family overview (Mamba+MLA 1B/3B/8B, GDN, KDA variants) |
| Docs | `docs/zebra_llama/README_GDN.md` | 552 | step-by-step GDN recipe |
| Docs | `docs/zebra_llama/README_KDA.md` | 623 | step-by-step KDA recipe |
| Docs | `all_changes.md` (this file) | — | master changelog |
| Config (training) | `examples/megatron/configs/MI300X/zebra_llama_300M_gdn_pure-pretrain.yaml` | 145 | GDN 300M training config |
| Config (training) | `examples/megatron/configs/MI300X/zebra_llama_300M_kda_pure-pretrain.yaml` | 179 | KDA 300M training config |
| Config (arch) | `primus/configs/models/megatron/zebra_llama_300M_gdn_pure.yaml` | 59 | GDN 300M architecture |
| Config (arch) | `primus/configs/models/megatron/zebra_llama_300M_kda_pure.yaml` | 58 | KDA 300M architecture |
| Tools (data) | `tools/convert_fla_to_megatron.py` | 104 | FLA Arrow shards → Megatron `.bin/.idx` |
| Tools (data) | `tools/fla_order_dataset.py` | — | `FLAOrderGPTDataset` shim (FLA `DistributedSampler` token order) |
| Tools (GDN convert) | `tools/convert_gdn_to_fla_hf.py` | — | Megatron sharded ckpt → FLA HF `GatedDeltaNetForCausalLM` |
| Tools (GDN verify) | `tools/verify_gdn_conversion.py` | — | loads + greedy-generates against 3 prompts to sanity-check conversion |
| Tools (GDN eval) | `tools/eval_gdn_lm_eval.py` | 43 | `lm-eval` wrapper that imports `fla` first (registers GDN) |
| Tools (KDA convert) | `tools/convert_fla_kda_init_to_megatron.py` | 350 | FLA HF init → Megatron sharded ckpt (for iter-1 parity) |
| Tools (KDA convert) | `tools/convert_kda_to_fla_hf.py` | 332 | Megatron sharded ckpt → FLA HF `KDAForCausalLM` |
| Tools (KDA eval) | `tools/eval_kda_lm_eval.py` | 45 | `lm-eval` wrapper that imports `fla` first (registers KDA) |

### 2.2 Modified files in Primus (10)

| Path | Why modified |
|--|--|
| `primus/backends/megatron/core/models/hybrid/gated_delta_net.py` | FLA Triton kernel signatures, GVA gating, optional FLA `causal_conv1d`, optional `FusedRMSNormGated` (95 lines) |
| `primus/backends/megatron/core/models/hybrid/gated_delta_net_layer.py` | propagate `eps=layernorm_epsilon`, defer fp32 residual cast for pre-norm fusion (24 lines) |
| `primus/backends/megatron/core/models/hybrid/hybrid_block.py` | fp32 residual handling, optional pre-norm/MLP fusion, optional TE→torch fallback for `final_norm` (46 lines) |
| `primus/backends/megatron/core/models/hybrid/hybrid_mamba_mla_layer_specs.py` | new `gdn_hybrid_stack_spec_no_te` + `kda_hybrid_stack_spec_no_te` (180 lines) |
| `primus/backends/megatron/core/models/hybrid/kimi_delta_attention.py` | fused `in_proj`, FLA Triton paths, `FusedRMSNormGated`, in-kernel gate fusion, `g_b_proj.bias=True`, FLA-matched `dt_bias` init, fp32 sigmoid for beta (365-line rewrite) |
| `primus/backends/megatron/core/models/hybrid/kimi_delta_attention_layer.py` | optional pre-norm field (`submodules.norm`) propagated with explicit `eps` (24 lines) |
| `primus/backends/megatron/patches/gdn_config_patches.py` | register linear-attention `TransformerConfig` fields + KDA fusion flags (`use_fla_triton_kda`, `use_fla_kda_in_kernel_gate`, `use_fla_fused_norm_gated`) so YAML can toggle without code changes |
| `primus/modules/trainer/megatron/pre_trainer.py` | `PRIMUS_DIAG`, `PRIMUS_DIAG_INTERVAL`, `PRIMUS_DUMP_ITER1_BATCH` instrumentation (~90 lines) |
| `primus/modules/trainer/megatron/trainer.py` | branch `train_valid_test_datasets_provider` to `FLAOrderGPTDataset` when `PRIMUS_FLA_DATA=1` (21 lines) |
| `primus/configs/models/megatron/mamba_base.yaml` + `language_model.yaml` + `zebra_llama_1B_gdn{,_pure}.yaml` | `bases:` → `extends:`, explicit `hidden_dropout: 0.0`, `use_short_conv: true` |

### 2.3 Vendored Megatron-LM submodule changes (6 files)

```
megatron/core/models/mamba/mamba_model.py        +49 lines
megatron/core/optimizer/__init__.py              ±42 lines
megatron/core/transformer/mlp.py                 ±34 lines
megatron/core/transformer/torch_norm.py          +5  lines
megatron/core/transformer/transformer_config.py  ±9  lines
pretrain_mamba.py                               +183 lines
─────────────────────────────────────────────────────────
6 files, 294 insertions(+), 28 deletions(−)
```

All applied via `bash megatron_patch.sh` (idempotent).

---

## 3. Megatron-LM submodule patches (in detail)

These six patches are the only changes inside the third-party Megatron-LM
checkout. KDA reuses every one of them — there is no KDA-specific
Megatron-LM patch. The script is in `megatron_patch.sh`; the diffs are in
`megatron_patches/*.patch`.

### 3.1 `01-mamba_model-fused-ce.patch`

**File:** `megatron/core/models/mamba/mamba_model.py`
**Why:** Megatron always materializes a `(batch · seq, vocab)` fp32 logits
tensor before computing cross-entropy. At 1024 batch × 2048 seq × 32 k
vocab that is 256 GB at fp32 — completely impossible. FLA never
materializes it; it chunks the matmul + CE per row.

**What changed:** Added a `_use_fused_cross_entropy` selector driven by
`PRIMUS_FUSED_CE`:

| `PRIMUS_FUSED_CE` | Path | Behaviour |
|--|--|--|
| `0` | native Megatron CE | materializes the full logits tensor (impossible at scale) |
| `1` (default) | `fla.modules.FusedLinearCrossEntropyLoss` | chunked LM-head + CE, no full logits tensor |
| `2` | `fla.modules.FusedCrossEntropyLoss` | materializes bf16 logits, matches FLA's reference loss bit-for-bit |

**Impact:** turns OOM into a 4 GB allocation; ~25 % loss-step speedup at
this batch size.

### 3.2 `02-optimizer-torch-fused-adam.patch`

**File:** `megatron/core/optimizer/__init__.py`
**Why:** TE/Apex `FusedAdam` and `torch.optim.AdamW(fused=True)` apply
the update in slightly different orders internally; the result is
mathematically equivalent in fp64 but **not** bit-identical in bf16.

**What changed:** opt-in selector `PRIMUS_TORCH_OPTIM=1` chooses
`torch.optim.AdamW(fused=True)`. Default off; on costs ~1 % iter time
but lets us prove iter-1 numerics match FLA exactly.

### 3.3 `03-mlp-fla-swiglu.patch`

**File:** `megatron/core/transformer/mlp.py`
**Why:** Megatron's SwiGLU is `silu(x_glu) * x_linear` — two separate
kernel launches + a `(B·S, 4·H)` intermediate tensor. FLA fuses it into
one Triton kernel for both forward and backward.

**What changed:** `PRIMUS_FLA_SWIGLU=1` (default) routes to
`fla.ops.swiglu.swiglu`. Saves ~20 ms/iter at our batch size; profiler
shows ~3.8× fewer GPU cycles spent on activation.

### 3.4 `04-torch_norm-fla-rmsnorm.patch`

**File:** `megatron/core/transformer/torch_norm.py`
**Why:** `WrappedTorchNorm` falls back to `torch.nn.RMSNorm`, which on
ROCm does **not** use the fused Triton path FLA uses. The numerics are
the same in fp32 but differ in bf16 reduction order.

**What changed:** when `PRIMUS_FLA_NORM=1`, `WrappedTorchNorm` returns
`fla.modules.RMSNorm` instead — matches FLA's normalization kernels
bit-for-bit and is faster on MI300X.

### 3.5 `05-transformer_config-hybrid-init.patch`

**File:** `megatron/core/transformer/transformer_config.py`
**Why:** Megatron's default `scaled_init_method_normal` divides the
output-layer std by `sqrt(2 · num_layers)` — appropriate for pure
transformers but **wrong** for hybrid GDN/KDA models, where FLA uses a
uniform `initializer_range` for every layer.

**What changed:** for `is_hybrid_model=True`, set
`output_layer_init_method = init_method_normal(self.init_method_std)`
(uniform). Without this fix the GDN output layer started ~24× smaller
than FLA's, producing iter-1 loss of 11.971 instead of 11.965.

### 3.6 `06-pretrain_mamba-fla-data-and-diag.patch`

**File:** `pretrain_mamba.py`
**Why:** two unrelated needs that share the same entry point.
1. The `train_valid_test_datasets_provider` registered for Mamba/GDN
   pretraining lives in `pretrain_mamba.py`, not `trainer.py`. The same
   FLA-order dataset shim that `trainer.py` got needs to be replicated
   here so GDN/KDA pretraining can also opt into bit-identical token
   ordering.
2. The parity hunt needed iter-1 batch dumps (token IDs) and per-layer
   activation dumps to diff against FLA. Both are forward-hook based.

**What changed:** (a) `PRIMUS_FLA_DATA=1` + `PRIMUS_FLA_CACHE_DIR=<path>`
swaps the dataset for `tools.fla_order_dataset.FLAOrderGPTDataset`.
(b) `PRIMUS_DUMP_ITER1_BATCH=<path>` / `PRIMUS_DUMP_ITER1_ACTS=<path>`
register tap-hooks that dump on the very first iteration only. All paths
are inert (cost ~ µs/iter) when the env vars are unset.

---

## 4. Primus model code changes

### 4.1 GDN — `primus/backends/megatron/core/models/hybrid/`

| File | Change |
|--|--|
| `gated_delta_net.py` | Pass `g=alpha`, `use_gate_in_kernel=True`, `A_log=...`, `dt_bias=...` directly to `chunk_gated_delta_rule`. Gate the `repeat_interleave` GVA pre-expansion behind `PRIMUS_NATIVE_GVA` (FLA's kernel handles GVA natively; the `repeat_interleave` backward is autograd-summed, **not** what FLA produces). Add optional FLA Triton `causal_conv1d` path under `PRIMUS_FLA_CONV`. Add optional FLA `FusedRMSNormGated` path under `PRIMUS_FLA_NORM`. Remove `@jit_fuser` on `_apply_gated_norm` so the gated path can branch. |
| `gated_delta_net_layer.py` | Forward `eps=self.config.layernorm_epsilon` to the pre-norm `build_module(...)`. Defer the `residual.to(fp32)` cast until after the optional pre-norm fusion path. Expose `_fuse_prenorm_with_next` flag. |
| `hybrid_block.py` | Force `residual_in_fp32=True` when `config.fp32_residual_connection`. Under `PRIMUS_FLA_NORM`, mark every GDN layer `_fuse_prenorm_with_next=True` and rewrite the forward loop to fuse a GDN block's mixer-out with the next MLP block's pre-MLP layernorm in one op. Under `PRIMUS_NO_TE`, use `WrappedTorchNorm` for `final_norm` instead of `TENorm`. |
| `hybrid_mamba_mla_layer_specs.py` | New `gdn_hybrid_stack_spec_no_te` ModuleSpec — plain `WrappedTorchNorm`, plain `Column/RowParallelLinear`, mirrors the TE variant submodule wiring. |

### 4.2 KDA — `primus/backends/megatron/core/models/hybrid/`

| File | Change |
|--|--|
| `kimi_delta_attention.py` | **Major refactor.** (a) Fuse six separate `hidden_states → X` projections (q, k, v, beta, f_a, g_a) into a single `in_proj: ColumnParallelLinear` of width `2·qk_dim + v_dim + 2·head_v_dim + num_v_heads` (matches GDN's fusion recipe). (b) Add optional FLA Triton `causal_conv1d` path under `PRIMUS_FLA_CONV` (accepts `[B, T, D]` directly, saves two transpose+contiguous copies per iter). (c) Add optional `FusedRMSNormGated` for the per-head output gate (one Triton kernel: RMSNorm + sigmoid-gate + multiply). (d) Add optional in-kernel gate fusion path (`chunk_kda(..., use_gate_in_kernel=True)`); the `−exp(A_log)·softplus(g+dt_bias)+cumsum` is recomputed in backward, the fp32 `[B,T,H,K]` gate tensor is never materialized. (e) `g_b_proj.bias=True` (matches `fla/layers/kda.py:189`). (f) Initialise `dt_bias` by FLA's log-uniform + inverse-softplus recipe (the previous `nn.init.ones_` gave `dt ≈ 1.31`, ~20× too large). (g) `beta = b_proj(h).float().sigmoid()` (fp32 sigmoid eliminates bf16 drift across 12 layers). (h) Materialize `q.contiguous() / k.contiguous() / v.contiguous()` after the split — without this `chunk_kda` allocates its own internal copies while autograd still pins the original views (2× activation memory wasted). (i) Removed `@torch.compiler.disable` decorator — Megatron does not wrap mixer forwards in `torch.compile` anyway, so the decorator was only adding ~25 ms/iter of dispatch overhead. |
| `kimi_delta_attention_layer.py` | Add `KimiDeltaAttentionLayerSubmodules.norm` field (default `IdentityOp`). When set to `WrappedTorchNorm`, the layer applies an explicit pre-norm matching `fla/models/kda/modeling_kda.py:113`. `eps` is forwarded explicitly because `WrappedTorchNorm` defaults to `1e-5` while KDA configs use `1e-6`. |
| `hybrid_mamba_mla_layer_specs.py` | New `kda_hybrid_stack_spec_no_te` ModuleSpec — plain `WrappedTorchNorm`, plain `ColumnParallelLinear`, plain `RowParallelLinear`, mixer `gate_norm=IdentityOp` (FLA has no re-norm for the gate path; pre-norm-once-and-reuse saves one norm launch per layer). |

### 4.3 Shared infrastructure

| File | Change |
|--|--|
| `primus/backends/megatron/patches/gdn_config_patches.py` | Register linear-attention fields (`linear_conv_kernel_dim`, `use_short_conv`, `linear_{key,value}_head_dim`, `linear_num_{key,value}_heads`) and KDA fusion flags (`use_fla_triton_kda`, `use_fla_triton_kda_hybrid`, `use_fla_kda_in_kernel_gate`, `use_fla_fused_norm_gated`) on `TransformerConfig` so YAML overrides propagate to runtime without touching third-party code. |
| `primus/modules/trainer/megatron/trainer.py` | `train_valid_test_datasets_provider` branches to `FLAOrderGPTDataset` when `PRIMUS_FLA_DATA=1` + `PRIMUS_FLA_CACHE_DIR` is set. |
| `primus/modules/trainer/megatron/pre_trainer.py` | `PRIMUS_DIAG`/`PRIMUS_DIAG_INTERVAL`/`PRIMUS_DIAG_BATCH` per-iter timing instrumentation; `PRIMUS_DUMP_ITER1_BATCH=<path>` iter-1 batch dumper. All early-exit on a single env-var lookup when unset. |

---

## 5. YAML configuration changes

### 5.1 Inheritance fix

Renamed `bases:` → `extends:` in four files (`mamba_base.yaml` and the
three `zebra_llama_*_gdn*.yaml` files). The Primus YAML resolver was
silently dropping `bases:` inheritance, which meant model configs were
missing the `hidden_dropout=0.0` default from `language_model.yaml` → it
was leaking through as `0.1` even though `mamba_base.yaml` set `0.0`.

### 5.2 New architecture configs (matched to FLA's JSONs exactly)

| File | Source of truth |
|--|--|
| `primus/configs/models/megatron/zebra_llama_300M_gdn_pure.yaml` | FLA `gated_deltanet_300M_pure.json` |
| `primus/configs/models/megatron/zebra_llama_300M_kda_pure.yaml` | FLA `kda_300M_pure.json` |

Both set: `num_layers: 24`, `hidden_size: 1024`, `ffn_hidden_size: 4096`,
`is_hybrid_model: true`, `hybrid_attention_ratio: 0.0` (pure linear
attention, no full-attention layers), tied embeddings, `add_bias_linear:
false`, `normalization: RMSNorm`, `norm_epsilon: 1.0e-6`,
`position_embedding_type: none`.

### 5.3 New training configs

`examples/megatron/configs/MI300X/zebra_llama_300M_{gdn,kda}_pure-pretrain.yaml`
encode the FLA-matched training schedule:

```yaml
train_iters: 4768                       # ≈10B tokens at 1024×2048
micro_batch_size: 128
global_batch_size: 1024
lr: 2.0e-4
min_lr: 2.0e-5                          # min_lr_rate=0.1
lr_warmup_iters: 200
lr_decay_iters: 4768
lr_decay_style: cosine
adam_beta1: 0.9
adam_beta2: 0.95
weight_decay: 0.01
clip_grad: 1.0
seed: 42

# Critical overrides:
layernorm_epsilon: 1.0e-6               # else TransformerConfig default 1e-5 overrides model YAML
hidden_dropout: 0.0                     # else language_model.yaml default 0.1 leaks through
attention_dropout: 0.0
barrier_with_L1_time: false             # else Megatron inserts 5-10 dist.barrier()/iter

# No-TE specs match FLA layer layout exactly:
spec: ['primus.backends.megatron.core.models.hybrid.hybrid_mamba_mla_layer_specs',
       '{gdn,kda}_hybrid_stack_spec_no_te']

# KDA-only fusion toggles (in the KDA YAML):
use_fla_triton_kda: true
use_fla_kda_in_kernel_gate: true
use_fla_fused_norm_gated: true

# Plain DDP — matches FLA; ZeRO-1 costs allreduce bandwidth and saves only ~3.6 GiB/rank at 300M
use_distributed_optimizer: false
overlap_grad_reduce: true
ddp_average_in_collective: true

# FLA-init checkpoint for bit-perfect iter-1 forward:
finetune: true
no_load_optim: true
no_load_rng: true
load: /home/<user>/Primus/output/fla_init_{gdn,kda}_300M
```

---

## 6. Tools added

All under `tools/`. Each script is self-documenting (top-level docstring +
runnable `--help`).

| Script | Direction | What it does |
|--|--|--|
| `convert_fla_to_megatron.py` | data | Reads FLA's preprocessed FineWeb-Edu Arrow shards directly via PyArrow (zero HF `datasets` overhead). Writes a single `.bin` of flat int32 token IDs + a Megatron `.idx` where each 2048-token chunk is one document. Cross-checks the first 10 tokens of the output against the first Arrow sample before exit. |
| `fla_order_dataset.py` | data | `FLAOrderGPTDataset` shim — quacks like Megatron's `GPTDataset` but reads tokens in the exact order FLA's HuggingFace `DistributedSampler(seed=42)` produces (fixed 2048-token chunks, no EOD insertion). Used when `PRIMUS_FLA_DATA=1`. |
| `convert_gdn_to_fla_hf.py` | GDN ckpt out | Reads Primus's sharded Megatron checkpoint, splits the fused `in_proj` back into separate q/k/v/g/b/a projections, splits `linear_fc1` into gate/up, handles both the TE and no-TE layer specs, emits FLA-loadable HF `GatedDeltaNetForCausalLM` directory. |
| `verify_gdn_conversion.py` | GDN sanity | Loads the converted HF model in bf16, runs 3 prompts, reports per-prompt loss + top-5 next-token IDs + 40-token greedy continuation. Loss < 6 = PASS. |
| `eval_gdn_lm_eval.py` | GDN eval | Thin `lm-eval` wrapper that `import fla` first (so `AutoConfig` recognises `gated_deltanet`) and monkey-patches `GatedDeltaNet{ForCausalLM,Model}.__init__` to absorb the `dtype` kwarg `transformers ≥ 4.55` passes internally. |
| `convert_fla_kda_init_to_megatron.py` | KDA ckpt in | Instantiates FLA's `KDAForCausalLM` with `seed=42`, harvests its randomly-initialised weights, concatenates the six FLA `hidden_states → X` projections into Primus's single fused `in_proj`, writes a Megatron-shape `iter_0000000/mp_rank_00/model_optim_rng.pt`. Used for bit-perfect iter-1 forward parity. |
| `convert_kda_to_fla_hf.py` | KDA ckpt out | Reverse of the above for a trained checkpoint — splits the fused `in_proj` back into q/k/v/f.0/g.0/b, remaps `g_norm`/`o_proj`/`f.1`/`g.1`/`A_log`/`dt_bias`/embeddings, copies tokenizer files, emits FLA-loadable HF `KDAForCausalLM` directory. |
| `eval_kda_lm_eval.py` | KDA eval | KDA equivalent of `eval_gdn_lm_eval.py` — imports `fla` first, monkey-patches `KDA{ForCausalLM,Model}.__init__` to absorb `dtype`. |

---

## 7. Runtime toggles (no re-patching needed)

Every env var below is read at module-import or first-call time and is
**inert when unset** — the cost of the check is one `os.environ.get()`
lookup per iteration.

| Env var | Default | Architectures | Effect |
|--|--|--|--|
| `PRIMUS_FUSED_CE` | `1` | both | `1` = FLA `FusedLinearCrossEntropyLoss` (chunked); `2` = FLA `FusedCrossEntropyLoss` (matches FLA bit-for-bit); `0` = native Megatron CE |
| `PRIMUS_FLA_SWIGLU` | `1` | both | replace Megatron's SwiGLU with FLA's Triton-fused kernel |
| `PRIMUS_FLA_NORM` | `0` (GDN) / `1` (KDA YAML) | both | FLA Triton `RMSNorm` via `WrappedTorchNorm`; for GDN also enables the pre-norm/MLP fusion in `HybridStack` |
| `PRIMUS_FLA_CONV` | `0` (GDN) / `1` (KDA YAML) | both | route depthwise short conv1d through FLA's Triton `causal_conv1d` (accepts `[B, T, D]` directly) |
| `PRIMUS_NATIVE_GVA` | `0` | GDN only | skip `repeat_interleave` pre-expand; let `chunk_gated_delta_rule` handle GVA inside the kernel (matches FLA's gradient layout) |
| `PRIMUS_NO_TE` | `0` | GDN only | use `WrappedTorchNorm` for `final_norm` in `HybridStack` instead of `TENorm` |
| `PRIMUS_TORCH_OPTIM` | `0` | both | `torch.optim.AdamW(fused=True)` instead of TE/Apex `FusedAdam` |
| `PRIMUS_FLA_DATA` | `0` | both | replace Megatron `GPTDataset` with `FLAOrderGPTDataset` (requires `PRIMUS_FLA_CACHE_DIR`) |
| `PRIMUS_FLA_CACHE_DIR` | unset | both | path to FLA's preprocessed HF dataset cache (used by `FLAOrderGPTDataset`) |
| `PRIMUS_DIAG` | `0` | both | enable per-iter diagnostic timing (also `PRIMUS_DIAG_INTERVAL=N`, `PRIMUS_DIAG_BATCH`) |
| `PRIMUS_DUMP_ITER1_BATCH` | unset | both | path to dump iter-1 token IDs for cross-framework comparison |
| `PRIMUS_DUMP_ITER1_ACTS` | unset | both | path to dump per-layer iter-1 activations (registers forward hooks) |
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | unset | KDA recommended | reduces allocator fragmentation; KDA's activation pattern needs it on |

KDA-specific YAML toggles (set in
`zebra_llama_300M_kda_pure-pretrain.yaml`, not env vars):

- `use_fla_triton_kda: true` — required to use FLA's Triton `chunk_kda`
- `use_fla_kda_in_kernel_gate: true` — fuse `−exp(A_log)·softplus(g+dt_bias)+cumsum` inside the kernel
- `use_fla_fused_norm_gated: true` — use `FusedRMSNormGated` for the gated output norm

---

## 8. Documentation map

```
all_changes.md             ← this file (master changelog)
GDN_FLA_PARITY.md          ← GDN deep-dive: every change + why
KDA_FLA_PARITY.md          ← KDA deep-dive: every change + why
docs/zebra_llama/
├── README.md              ← family overview (Mamba+MLA 1B/3B/8B, GDN, KDA)
├── README_GDN.md          ← step-by-step recipe for GDN 300M
└── README_KDA.md          ← step-by-step recipe for KDA 300M
```

Start with the README for your recipe (GDN or KDA), then drop into the
PARITY doc when you need to know *why* a particular flag/file/init exists.

---

## 9. How to reproduce the parity numbers

```bash
# 1. (one time) start the dev container
bash bash-docker.sh
docker exec -it primus_hybrid_new bash
cd /home/<user>/Primus

# 2. (one time) install dependencies
pip install -r requirements.txt
pip install -e /home/<user>/flash-linear-attention
pip install lm-eval

# 3. (one time) apply the six Megatron-LM patches
bash megatron_patch.sh

# 4. (one time) prepare the FLA-aligned dataset (~20 GB)
python tools/convert_fla_to_megatron.py

# 5. (one time, optional but recommended) generate the FLA-init checkpoint
python tools/convert_fla_kda_init_to_megatron.py        # for KDA
# (analogous tools/init_primus_from_fla.py for GDN; untracked)

# 6. Train (8 GPUs by default)
EXP=examples/megatron/configs/MI300X/zebra_llama_300M_kda_pure-pretrain.yaml \
  bash examples/run_pretrain.sh 2>&1 | tee primus_kda.log
#   → ≈1h 56m on a healthy MI300X box

# 7. Convert to HF + evaluate
python tools/convert_kda_to_fla_hf.py \
  --checkpoint-path output/amd/root/zebra_llama_300M_kda_pure-pretrain/checkpoints/iter_0004768 \
  --output-dir      output/kda_pure_300M_fla_hf \
  --config          /home/<user>/flash-linear-attention/legacy/training/configs/kda_300M_pure.json \
  --tokenizer-src   /home/<user>/checkpoints/kda_pure_300M_10B

python tools/eval_kda_lm_eval.py \
  --model hf \
  --model_args pretrained=output/kda_pure_300M_fla_hf,dtype=bfloat16,trust_remote_code=True,tokenizer=meta-llama/Llama-3.2-1B \
  --tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu,race \
  --batch_size auto \
  --output_path output/kda_pure_300M_eval_results_primus
```

Same procedure for GDN with `gdn` substituted everywhere.

---

## 10. Commits that made up this work

```
1a43061d  training matches fla for gdn                                      (Apr 24)
550ef6b7  Add GDN 300M training config and tools                            (May  6)
898a14c1  adding patch (megatron_patch.sh, hybrid_block, layer specs)       (May  7)
21c094dd  Add bash-docker script and GDN training documentation             (May 19)
5638c013  Add KDA training documentation and configurations for FLA parity  (May 20)
```

Branch: `vanbhati/kda-optimized-training-patch`.

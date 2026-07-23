# Merge Summary: `clairlee/kda-optimized-training-patch` ← `origin/main`

**Date:** 2026-07-15
**Merge commit:** `7ba8dc59`
**Pre-merge HEAD:** `8424d25f` (experimental gdn changes to avoid bf16 downcast)
**Main branch:** 1036 files changed, 122,835 insertions, 10,415 deletions

---

## 1. Merge Conflict Resolution

The merge introduced conflicts in 6 areas, all resolved manually:

| File | Conflict Type | Resolution |
|------|--------------|------------|
| `.gitignore` | Both sides added entries | Combined both sets of entries |
| `torch_fully_sharded_data_parallel.py` | Content conflict | Accepted main's refactored version (new import paths) |
| `hybrid_block.py` | Content conflict | Merged main's `**kwargs` addition and fallback `hybrid_attention_ratio`/`hybrid_mlp_ratio` logic with our branch's changes |
| `hybrid_mamba_mla_layer_specs.py` | Content conflict | Kept our GDN/KDA spec imports while accepting main's `MambaStack` → `HybridStack` rename |
| `pre_trainer.py` | Modify/delete (deleted in main) | Accepted deletion — file was removed in main's refactor |
| `trainer.py` | Modify/delete (deleted in main) | Accepted deletion — file was removed in main's refactor |

## 2. Import Path Fixes (Post-Merge)

Main refactored `primus.modules.module_utils` → `primus.core.utils.module_utils`. The following files on our branch still referenced the old path and were updated:

- `primus/backends/megatron/patches/empty_cache_interval_patches.py`
- `primus/backends/megatron/patches/fla_runtime_patches.py`
- `primus/backends/megatron/patches/gdn_config_patches.py`
- `tools/megatron_forward_zebra_llama.py`

## 3. GDN Numerical Stability Fixes

### 3a. `A_init_range` and dtype fix (commit `8424d25f`, pre-merge)

**Problem:** `A_init_range` defaulted to `(0, 16)`, so `A_log = log(uniform(0, 16))` could compute `log(0) = -inf` in bf16, producing NaN in training (detected at iteration 2).

**Fix in `gated_delta_net.py`:**
- Changed `A_init_range` default from `(0, 16)` to `(1, 16)`
- Changed `A_log` and `dt_bias` parameter dtypes from `config.params_dtype` (bf16) to `torch.float32`

### 3b. FLA Triton in-kernel gate NaN fix (post-merge, uncommitted)

**Problem:** FLA's `chunk_gated_delta_rule` Triton kernel produces sporadic NaN/Inf when `use_gate_in_kernel=True` on ROCm (AMD MI300X). The in-kernel gate computation (`g = -exp(A_log) * softplus(alpha + dt_bias)`) triggers a Triton codegen bug on ROCm that corrupts the recurrent state. This caused both `zebra_llama_1B_gdn-pretrain` and `zebra_llama_1B_gdn_pure-pretrain` to fail with NaN at iteration 1.

**Diagnosis:**
- Isolated the NaN to `core_attn_out` (output of `chunk_gated_delta_rule`) — all inputs (Q, K, V, alpha, beta, A_log, dt_bias) were clean
- Confirmed the kernel produces NaN with `use_gate_in_kernel=True` and clean inputs with `use_gate_in_kernel=False` across 5 repeated runs
- The bug is input-magnitude-dependent: inputs at scale ≤ 0.5 are usually fine, but real model magnitudes (alpha mean ~0.7, max ~4.0) reliably trigger it

**Fix in `gated_delta_net.py`:**
- Pre-compute the gate in fp32 before calling the kernel: `g = -exp(A_log) * softplus(alpha + dt_bias)`
- Pass the pre-computed gate with `use_gate_in_kernel=False` instead of relying on the buggy in-kernel path
- This is mathematically identical — only the computation location changes

## 4. Tokenizer Alignment (pre-merge)

All GDN model configs were updated to use the same tokenizer as KDA for consistency:

| Config | Old Tokenizer | New Tokenizer |
|--------|--------------|---------------|
| `zebra_llama_1B_gdn.yaml` | `fla-hub/gla-1.3B-100B` | `meta-llama/Llama-3.2-1B` |
| `zebra_llama_1B_gdn-pretrain.yaml` | `fla-hub/gla-1.3B-100B` | `meta-llama/Llama-3.2-1B` |

The `zebra_llama_1B_gdn_pure.yaml` already used `meta-llama/Llama-3.2-1B`.

## 5. Validation Results (Post-Fix)

Both GDN configs train successfully with no NaN:

| Config | Iterations | Loss (start → end) | Throughput |
|--------|-----------|-------------------|------------|
| `zebra_llama_1B_gdn-pretrain.yaml` | 10 | 12.23 → 4.41 | ~798 TFLOP/s/GPU |
| `zebra_llama_1B_gdn_pure-pretrain.yaml` | 10 | 12.19 → 4.63 | ~706 TFLOP/s/GPU |

## 6. Post-Merge Commit

All fixes were committed and pushed as `cb6fa5fe`:

> fix: pre-compute GDN gate in fp32 to avoid FLA Triton NaN on ROCm

- `primus/backends/megatron/core/models/hybrid/gated_delta_net.py` — FLA in-kernel gate fix + debug cleanup
- `primus/backends/megatron/patches/empty_cache_interval_patches.py` — import path fix
- `primus/backends/megatron/patches/fla_runtime_patches.py` — import path fix
- `primus/backends/megatron/patches/gdn_config_patches.py` — import path fix
- `tools/megatron_forward_zebra_llama.py` — import path fix

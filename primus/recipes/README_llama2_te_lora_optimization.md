# Llama2 LoRA: Transformer Engine op fuser and stable loss

This document explains the **TE op fuser** and **LoRA** optimizations added for Primus Llama2 70B LoRA post-training, why LM/validation loss can change when they are combined incorrectly, and how to configure a **fast backbone** with a **numerically stable LoRA path**.

## Concepts (two different “fusions”)

| Layer | What it controls | Typical effect |
|--------|------------------|----------------|
| **Model: `use_transformer_engine_op_fuser`** | Megatron layer spec uses TE’s operation-fuser path on the **backbone** (e.g. fused MLP / TE ops API). | Throughput; small numeric differences vs unfused TE are possible. |
| **LoRA: `TEFusedLoRALinear` vs `LoRALinear`** | How the **adapter** is applied around parallel TE linears when TP world size is 1. | Can **materially change** forward/backward and training loss if the fused path does not match the same tensor boundaries as `AdapterWrapper.base_linear_forward`. |

Important: enabling **`use_transformer_engine_op_fuser` on the model** does **not** require fused LoRA. You can keep **unfused LoRA** (`LoRALinear`) for stable curves while still using the fused backbone.

## What went wrong originally

With **`tensor_model_parallel_size: 1`**, Megatron-Bridge previously chose **`TEFusedLoRALinear`** whenever the model had **`use_transformer_engine_op_fuser=True`**.

`TEFusedLoRALinear` rebuilds the block as `te.ops.Sequential` (e.g. norm → quantize/fork → main linear) and runs the LoRA branch off that fork. Standard **`LoRALinear`** uses the wrapped module’s real forward and **`base_linear_forward`**, which can feed the adapter a **different activation** (dtype / quantization boundary). That is not a tiny FP noise issue: it can produce **wrong adapter gradients**, **stuck or rising LM loss**, and **bad validation loss**, even though the **loss function** (cross-entropy) is unchanged.

## What we changed

### 1. `third_party/Megatron-Bridge/src/megatron/bridge/peft/lora.py`

- **`use_te_fused_lora`** (default **`False`**): master switch. When `False`, adapters always use **`LoRALinear`** (stable default).
- **`te_fused_lora_include_modules`**: optional allowlist (wildcard rules same as `target_modules`).  
  - `None`: any layer may use fused LoRA if other conditions hold.  
  - `[]`: **never** use fused LoRA even if `use_te_fused_lora=True`.
- **`te_fused_lora_exclude_modules`**: denylist; matching modules always use **`LoRALinear`**.
- Helper **`_te_fused_lora_allowed_for_module`**: applies include/exclude before wrapping with **`TEFusedLoRALinear`**.
- **`TEFusedLoRALinear`** remains **unsupported for TP > 1** (unchanged); fused LoRA is only considered when `get_tensor_model_parallel_world_size() == 1`.

### 2. `primus/recipes/llama2_custom.py` (`_llama2_lora`)

- **`model_cfg.use_transformer_engine_op_fuser`**: recipe flag **`use_transformer_engine_op_fuser`** (default **`True`**).
- **`stable_lora_with_te_op_fuser`** (default **`True`**): single Primus knob — when **true**, sets **`use_te_fused_lora=False`** on **`LoRA`** regardless of backbone; when **false**, sets **`use_te_fused_lora = use_transformer_engine_op_fuser`** (legacy coupling).
- **`LoRA(...)`** also receives **`te_fused_lora_include_modules`** / **`te_fused_lora_exclude_modules`** when you experiment with fused LoRA (legacy mode).
- See [README_llama2_custom_enabled_knobs.md](README_llama2_custom_enabled_knobs.md) for the full truth table.

### 3. `examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml`

- Documents **`use_transformer_engine_op_fuser`** and **`stable_lora_with_te_op_fuser`**.
- Comments show **`te_fused_lora_include_modules`** / **`te_fused_lora_exclude_modules`** when **`stable_lora_with_te_op_fuser: false`**.

### 4. Other recipe toggles (context)

The recipe may also expose fusion / memory options (e.g. **`apply_rope_fusion`**, **`fused_single_qkv_rope`**, optional **activation recomputation**). Those are independent from the LoRA op-fuser issue above. Note: Megatron-LM can forbid **`recompute_granularity="full"`** together with **CUDA graphs**; if you enable both, validate against your stack.

## Recommended configurations

### A. Default (stable loss, fused backbone on)

```yaml
use_transformer_engine_op_fuser: true
stable_lora_with_te_op_fuser: true
```

Backbone TE op fuser on; LoRA always **`LoRALinear`** (same intent as former **`use_te_fused_lora: false`**).

### B. If LM/val loss still drifts vs an old baseline

Disable backbone op fuser for the next run:

```yaml
use_transformer_engine_op_fuser: false
```

You must **restart training**; the model graph is fixed at init.

### C. If you explicitly want fused LoRA (experimental)

1. Set **`stable_lora_with_te_op_fuser: false`** and keep **`use_transformer_engine_op_fuser: true`** (legacy: **`use_te_fused_lora`** tracks backbone when TP=1).
2. Prefer **narrowing** fusion to lower-risk modules, for example **MLP only**:

   ```yaml
   te_fused_lora_include_modules: ["*.linear_fc1", "*.linear_fc2"]
   ```

   or exclude attention:

   ```yaml
   te_fused_lora_exclude_modules: ["linear_qkv", "linear_proj", "*self_attention*"]
   ```

3. Watch **grad norm** and **validation loss** early; revert to **`stable_lora_with_te_op_fuser: true`** if unstable.

## Programmatic usage (non-YAML)

```python
from megatron.bridge.peft.lora import LoRA

LoRA(
    target_modules=["linear_qkv", "linear_proj"],
    use_te_fused_lora=False,  # default; stable
    # use_te_fused_lora=True,
    # te_fused_lora_include_modules=["*.linear_fc1"],
    # te_fused_lora_exclude_modules=["linear_qkv", "linear_proj"],
)
```

## Files to read in the repo

| File | Role |
|------|------|
| `third_party/Megatron-Bridge/src/megatron/bridge/peft/lora.py` | LoRA transform, fused vs unfused decision, include/exclude. |
| `third_party/Megatron-Bridge/src/megatron/bridge/peft/lora_layers.py` | `LoRALinear`, `TEFusedLoRALinear` implementations. |
| `third_party/Megatron-Bridge/src/megatron/bridge/models/gpt_provider.py` | Passes `use_te_op_fuser` from `use_transformer_engine_op_fuser` into layer spec. |
| `primus/recipes/llama2_custom.py` | Primus Llama2 LoRA recipe wiring. |
| `examples/megatron_bridge/configs/MI355X/llama2_70b_lora_posttrain.yaml` | Example overrides. |

## Summary

- **Loss changed** because **`TEFusedLoRALinear`** is not equivalent to **`LoRALinear`** for adapter inputs/gradients under FP8/TE boundaries—not because the CE loss API changed.
- **Fix (recipe):** **`stable_lora_with_te_op_fuser: true`** (default) keeps **unfused LoRA** while **`use_transformer_engine_op_fuser: true`** can stay on for the backbone.
- **Fine control:** `te_fused_lora_include_modules` / `te_fused_lora_exclude_modules` when experimenting with fused LoRA.

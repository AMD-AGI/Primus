# FLA `chunk_gated_delta_rule` In-Kernel Gate NaN Bug on ROCm

## Background: How GDN Computes Gates

The Gated Delta Rule recurrence uses a per-head, per-timestep decay gate `g` that controls how fast the recurrent state forgets:

```
g = -exp(A_log) * softplus(alpha + dt_bias)
```

- **`A_log`** — learnable per-head parameter (`[H]`), initialized as `log(uniform(1, 16))`, stored in fp32
- **`dt_bias`** — learnable per-head parameter (`[H]`), stored in fp32
- **`alpha`** — per-token, per-head projection from input hidden states (`[B, T, H]`), in bf16

The gate is always negative (exponential decay). Typical values at init: `g ∈ [-12, -0.001]`.

## Two Ways to Compute the Gate

FLA's kernel offers two paths:

| Mode | How it works | Advantage |
|------|-------------|-----------|
| `use_gate_in_kernel=True` | Raw `alpha`, `A_log`, `dt_bias` passed to Triton kernel; gate computed inside the fused op | Fewer kernel launches |
| `use_gate_in_kernel=False` | Gate pre-computed in PyTorch fp32; final `g` values passed to kernel | Runs through PyTorch's well-tested fp32 path |

## The Bug

When `use_gate_in_kernel=True`, the Triton compiler generates ROCm (HIP) GPU code that computes the gate inside a fused kernel alongside the chunked recurrence. On AMD MI300X, this codegen path produces **numerically corrupt results** — NaN and Inf in the output tensor.

### Diagnostic Evidence

**1. All kernel inputs are clean — only the output is corrupt:**

```
[GDN layer 1] query:   max=3.27, mean=0.20   ✓
[GDN layer 1] key:     max=3.06, mean=0.20   ✓
[GDN layer 1] value:   max=3.41, mean=0.20   ✓
[GDN layer 1] alpha:   max=4.06, mean=0.71   ✓
[GDN layer 1] beta:    max=0.98, mean=0.50   ✓
[GDN layer 1] A_log:   max=2.77, mean=1.90   ✓
[GDN layer 1] dt_bias: max=6.53, mean=4.50   ✓
[GDN layer 1] core_attn_out: NaN/Inf         ✗
```

**2. Switching only the gate path fixes it (5/5 runs, deterministic):**

```
use_gate_in_kernel=True  → NaN=True   (all 5 runs)
use_gate_in_kernel=False → NaN=False  (all 5 runs)
```

Same inputs, same kernel, same hardware — the only difference is where `g` is computed.

**3. Input magnitude matters — the bug is scale-dependent:**

| Input scale | NaN? | Notes |
|------------|------|-------|
| 0.01–0.50 | No | Tiny inputs → gate magnitudes stay small |
| 1.00 | Yes | Realistic model magnitudes trigger overflow |

Real model magnitudes: alpha mean ~0.7, max ~4.0. This consistently triggers the bug.

**4. NaN cascades through all layers.** Layer 1 output → layer 3 input → every subsequent layer → loss = NaN → training crash at iteration 1.

## Likely Root Cause

The Triton kernel fuses the gate computation with the chunked delta rule recurrence:

```
state[t] = exp(g[t]) * state[t-1] + beta[t] * (v[t] ⊗ k[t])
output[t] = q[t] @ state[t]
```

When the gate is computed in-kernel, the Triton compiler on ROCm likely uses different precision for intermediate `exp()` / `softplus()` operations or reorders floating-point ops in a way that causes overflow in the cumulative gate sum. The recurrence amplifies even small numerical errors across 2048 timesteps, eventually producing Inf → NaN.

## The Fix

Pre-compute the gate in fp32 using PyTorch before calling the Triton kernel:

```python
# Before (broken on ROCm):
core_attn_out, _ = chunk_gated_delta_rule(
    q, k, v, g=alpha, beta=beta,
    use_gate_in_kernel=True,
    A_log=self.A_log, dt_bias=self.dt_bias,
)

# After (stable):
g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)
core_attn_out, _ = chunk_gated_delta_rule(
    q, k, v, g=g.to(query.dtype), beta=beta,
    use_gate_in_kernel=False,
)
```

Mathematically identical — same formula, just computed outside the Triton kernel. Minimal performance cost (one small elementwise op). Numerically stable.

## Affected Configs

Both GDN training configs were affected:
- `zebra_llama_1B_gdn-pretrain.yaml`
- `zebra_llama_1B_gdn_pure-pretrain.yaml`

## Validation

| Config | Iterations | Loss (start → end) | Throughput |
|--------|-----------|-------------------|------------|
| `zebra_llama_1B_gdn-pretrain` | 10 | 12.23 → 4.41 | ~798 TFLOP/s/GPU |
| `zebra_llama_1B_gdn_pure-pretrain` | 10 | 12.19 → 4.63 | ~706 TFLOP/s/GPU |

Zero NaN iterations across all runs after the fix.

# FP8 Training for Flux Models

Complete guide for training Flux diffusion models with FP8 (8-bit floating point) precision on AMD MI300X GPUs using Transformer Engine's delayed scaling recipe.

## Overview

FP8 training provides significant memory and speed improvements while maintaining numerical stability through delayed scaling:

- **~2x memory reduction** (activations and weights)
- **1.5-2x training speedup** on AMD MI300X
- **Maintains numerical stability** via delayed scaling
- **Enables larger batch sizes** or higher resolutions

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)
- [AMD MI300X Specific](#amd-mi300x-specific)

---

## Prerequisites

### Hardware Requirements

- **AMD MI300X GPUs** with ROCm 6.0+ support
- **Minimum GPUs:**
  - Flux 535M: 1x MI300X (testing)
  - Flux 12B: 2x MI300X with TP=2 (can train with FP8)

### Software Requirements

1. **ROCm 6.0+** with FP8 tensor core support
2. **Transformer Engine 2.1.0+** with ROCm backend
3. **PyTorch** with ROCm support
4. **Megatron-LM** (included in Primus)

### Verification

Verify your environment has:
- Transformer Engine 2.1.0+ with ROCm backend
- FP8 support (run `python3 -c "import transformer_engine.pytorch as te; print(te.fp8.is_fp8_available())"`)

---

## Quick Start

### Test FP8 with Flux 535M (Recommended First Step)

```bash
# 1. Prepare test dataset (or use existing)
# See primus/configs/data/megatron/diffusion/README.md

# 2. Train Flux 535M with FP8
EXP=examples/megatron/configs/MI300X/diffusion/flux_535m_pretrain_fp8.yaml \
GPUS_PER_NODE=1 \
bash examples/run_pretrain.sh
```

### Production Training with Flux 12B

```bash
# After validating with 535M, scale to 12B (TransformerEngine FP8)
EXP=examples/megatron/configs/MI300X/diffusion/flux_12b_ddp_energon_schnell_resample_te_spec_fp8.yaml \
GPUS_PER_NODE=8 \
NNODES=4 \
bash examples/run_slurm_pretrain.sh

# Or local-spec FP8 (no TransformerEngine dependency)
EXP=examples/megatron/configs/MI300X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_fp8.yaml \
GPUS_PER_NODE=8 \
NNODES=4 \
bash examples/run_slurm_pretrain.sh
```

---

## Configuration

### FP8 Model Configuration

FP8 is configured at the model level. Two pre-configured files are available:

- `primus/configs/models/megatron/diffusion/flux_535m_fp8.yaml`
- `primus/configs/models/megatron/diffusion/flux_12b_fp8.yaml`

**Key FP8 Parameters:**

```yaml
# Enable FP8
fp8: "e4m3"  # E4M3 format (recommended)
             # Alternative: "hybrid" (E4M3 activations + E5M2 gradients)

# FP8 Recipe
fp8_recipe: "delayed"  # Delayed scaling (most stable)
                       # Alternatives: "tensorwise", "blockwise", "mxfp8"

# Scaling Configuration
fp8_margin: 0  # Margin for scaling factor (0 = no margin)
fp8_amax_history_len: 1024  # History window for delayed scaling
                            # Larger = more stable, smaller = adapts faster
fp8_amax_compute_algo: "most_recent"  # or "max"

# Gradient Precision
fp8_wgrad: true  # Enable FP8 for weight gradients (recommended)

# Attention Precision
fp8_dot_product_attention: false  # Keep attention in higher precision
fp8_multi_head_attention: false   # Keep MHA in higher precision
```

### Training Configuration Adjustments

**Batch Sizes with FP8:**

```yaml
# Flux 12B - can increase batch size with FP8 memory savings
micro_batch_size: 2  # vs 1 for BF16
global_batch_size: 256  # same as BF16

# Flux 535M - can increase significantly
micro_batch_size: 4  # vs 2 for BF16
global_batch_size: 32
```

**Optimizer Settings (Same as BF16):**

```yaml
optimizer: adamw
lr: 1.0e-4
min_lr: 1.0e-5
weight_decay: 0.01
clip_grad: 1.0  # Gradient clipping still important!
```

**Parallelism with FP8:**

```yaml
# Flux 12B - can potentially reduce TP with FP8
tensor_model_parallel_size: 2  # or reduce to 1 with FP8
pipeline_model_parallel_size: 1
context_parallel_size: 1
```

---

## Performance Benchmarks

### Memory Usage

| Model | Precision | Memory/GPU | Batch Size | Notes |
|-------|-----------|------------|------------|-------|
| Flux 535M | BF16 | ~7-10GB | 2 | Baseline |
| Flux 535M | FP8 | ~3-5GB | 4 | ~50% reduction |
| Flux 12B | BF16 | ~40-50GB | 1 | TP=2 required |
| Flux 12B | FP8 | ~20-25GB | 2 | TP=2, ~50% reduction |

### Training Speed

| Model | Precision | Steps/sec | Speedup | Hardware |
|-------|-----------|-----------|---------|----------|
| Flux 535M | BF16 | ~20-30 | 1.0x | 1x MI300X |
| Flux 535M | FP8 | ~30-50 | 1.5-2x | 1x MI300X |
| Flux 12B | BF16 | ~0.5-1.0 | 1.0x | 32x MI300X |
| Flux 12B | FP8 | ~0.8-1.5 | 1.5-2x | 32x MI300X |

### Expected Results

- **Memory:** ~50% reduction vs BF16
- **Speed:** 1.5-2x faster training
- **Quality:** Loss curves within 5% of BF16
- **Convergence:** Similar or faster than BF16

---

## Troubleshooting

### NaN or Inf in Losses

**Problem:** Training becomes unstable with NaN/Inf values

**Solutions:**

1. **Increase scaling history:**
   ```yaml
   fp8_amax_history_len: 2048  # or 4096
   ```

2. **Disable FP8 for weight gradients:**
   ```yaml
   fp8_wgrad: false
   ```

3. **Use more conservative scaling:**
   ```yaml
   fp8_amax_compute_algo: "max"  # instead of "most_recent"
   ```

4. **Add scaling margin:**
   ```yaml
   fp8_margin: 1  # or 2
   ```

5. **Keep first/last layers in BF16:**
   ```yaml
   first_last_layers_bf16: true
   num_layers_at_start_in_bf16: 2
   num_layers_at_end_in_bf16: 2
   ```

### Out of Memory Even with FP8

**Problem:** Still hitting OOM errors with FP8 enabled

**Solutions:**

1. **Reduce micro batch size:**
   ```yaml
   micro_batch_size: 1  # back to minimum
   ```

2. **Enable gradient checkpointing:**
   ```yaml
   recompute_granularity: "selective"  # or "full"
   recompute_method: "block"
   ```

3. **Increase tensor parallelism:**
   ```yaml
   tensor_model_parallel_size: 4  # distribute more
   ```

4. **Reduce sequence length:**
   ```yaml
   seq_length: 2048  # if applicable
   ```

### FP8 Not Available

**Problem:** Setup script shows "FP8 not available"

**Checks:**

1. **Verify GPU model:**
   ```bash
   rocm-smi --showproductname
   # Should show MI300X
   ```

2. **Check ROCm version:**
   ```bash
   rocm-smi --showversion
   # Should be 6.0+
   ```

3. **Verify Transformer Engine:**
   ```bash
   python3 -c "import transformer_engine; print(transformer_engine.__version__)"
   # Should be 2.1.0+
   ```

4. **Test FP8 directly:**
   ```python
   import transformer_engine.pytorch as te
   print(te.fp8.is_fp8_available())  # Should be True
   ```

### Slower Than Expected

**Problem:** FP8 training is not faster than BF16

**Checks:**

1. **Verify FP8 is actually enabled:**
   - Check logs for FP8 context messages
   - Run with `NCCL_DEBUG=INFO` to see precision info

2. **Check batch size:**
   - Ensure you increased micro_batch_size with FP8
   - Small batches may not show speedup

3. **Verify tensor cores:**
   - FP8 requires tensor core support
   - Check ROCm driver configuration

4. **Profile training:**
   ```yaml
   log_timers_to_tensorboard: true
   ```
   - Compare FP8 vs BF16 step times

---

## Best Practices

### Recommended Workflow

1. **Start with 535M:**
   - Validate FP8 works correctly
   - Test for 100-1000 steps
   - Verify no NaN/Inf

2. **Validate on small 12B run:**
   - Train for 1000-5000 steps
   - Compare loss with BF16 baseline
   - Check memory and speed improvements

3. **Production training:**
   - Monitor closely for first 10K steps
   - Watch for numerical issues
   - Compare checkpoints with BF16

### Training Configuration

**Conservative (stable):**
```yaml
fp8_recipe: "delayed"
fp8_amax_history_len: 2048
fp8_amax_compute_algo: "max"
fp8_wgrad: false
```

**Balanced (recommended):**
```yaml
fp8_recipe: "delayed"
fp8_amax_history_len: 1024
fp8_amax_compute_algo: "most_recent"
fp8_wgrad: true
```

**Aggressive (maximum performance):**
```yaml
fp8_recipe: "tensorwise"  # Requires TE 2.2.0+
fp8_amax_history_len: 512
fp8_amax_compute_algo: "most_recent"
fp8_wgrad: true
```

### Monitoring

**Key metrics to watch:**

1. **Loss curves:**
   - Should be smooth (no spikes)
   - Should decrease normally
   - Compare with BF16 baseline

2. **Gradient norms:**
   - Should be stable
   - No sudden jumps to infinity

3. **Memory usage:**
   - Should be ~50% of BF16
   - Check with `rocm-smi`

4. **Training speed:**
   - Should be 1.5-2x faster
   - Measure steps/second

### Checkpointing

- **FP8 checkpoints are compatible with BF16**
- Can switch between FP8/BF16 training
- Optimizer state includes FP8 scaling factors
- Checkpoints are same size as BF16

---

## Autotune (Local Spec FP8)

> **Scope:** This section covers the **local-spec** FP8 path (`PrimusTurboFloat8LocalSpecProvider`, no TransformerEngine), e.g. `flux_12b_ddp_energon_schnell_resample_local_spec_fp8.yaml`. The TransformerEngine prerequisites and checks elsewhere in this guide (`te.fp8.is_fp8_available()`, "FP8 Not Available") do **not** apply here -- this path quantizes via Primus Turbo directly.

### Enable autotune (and do not pin the FP8 GEMM backend)

The local-spec FP8 kernels benefit from the Primus-Turbo autotuner, which picks the best backend per GEMM shape. Enable it with `PRIMUS_TURBO_AUTO_TUNE=1`.

`PRIMUS_TURBO_AUTO_TUNE=1` is necessary but **not sufficient**: an explicit `PRIMUS_TURBO_GEMM_BACKEND` short-circuits autotune (the FP8 kernel dispatcher returns the user-specified backend before the autotune step), so it must be unset (or scoped so it does not cover FP8) for autotune to engage.

**Note:** some base images bake `PRIMUS_TURBO_GEMM_BACKEND` as an *empty string* rather than leaving it unset. An empty value is not treated as "unset" and can raise `KeyError ''` on the first FP8 GEMM. If you hit this, `unset PRIMUS_TURBO_GEMM_BACKEND` before launching.

```bash
unset PRIMUS_TURBO_GEMM_BACKEND          # or scope it so it does not cover FP8
export PRIMUS_TURBO_AUTO_TUNE=1

EXP=examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_fp8.yaml \
  bash examples/run_pretrain.sh
```

### Contrast with MXFP4

For MXFP4/FP4 + AITER with a tuned CSV, do the **opposite**: leave `PRIMUS_TURBO_AUTO_TUNE` unset, because autotune disables the AITER preshuffle fast path. See the [MXFP4 Training Guide](mxfp4_training.md) ("Preshuffle fast path"). Do not copy the MXFP4 env recipe for FP8.

---

## AMD MI300X Specific

### Environment Variables

```bash
# Optional: set for better performance
export HSA_FORCE_FINE_GRAIN_PCIE=1  # Better PCIe performance
export NCCL_DEBUG=INFO  # For debugging
export HSA_ENABLE_SDMA=0  # Disable SDMA for stability
```

### ROCm Optimization

1. **HipBLASLt tuning:**
   ```bash
   # Generate optimal GEMM kernels for your hardware
   # See ROCm documentation for details
   ```

2. **NCCL configuration:**
   ```bash
   export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
   export NCCL_NET_GDR_LEVEL=3  # GPU Direct RDMA
   ```

3. **Memory management:**
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=9.4.2  # For MI300X
   ```

### Known Issues

1. **Transformer Engine ROCm support:**
   - Verify TE version supports ROCm FP8
   - Some recipes may require specific TE versions

2. **Numerical stability:**
   - MI300X may require longer history (fp8_amax_history_len)
   - Start conservative and tune

3. **Multi-node training:**
   - Ensure RCCL/NCCL properly configured
   - Test single-node first

---

## Testing

### Integration Test

```bash
# Quick 100-step validation run
EXP=examples/megatron/configs/MI300X/diffusion/flux_535m_pretrain_fp8.yaml \
GPUS_PER_NODE=1 \
bash examples/run_pretrain.sh
```

### Convergence Test

1. Train both BF16 and FP8 for 5000 steps
2. Compare loss curves (should be within 5%)
3. Generate images from checkpoints
4. Compare quality visually

---

## Numerical verification status

To set expectations for external users, the precision/convergence claims in this
codebase fall into two tiers:

- **Backed by in-repo tests** (CI-runnable on supported hardware): structural and
  convention checks — attention TE-vs-local-spec equivalence, RNG/seed
  determinism, chimera init, VAE resample reproducibility, fused delayed-scale
  update, and MLPerf warmup FP8 state.
- **Asserted, not yet backed by an in-repo test:** end-to-end *tensor parity*
  against HuggingFace/Diffusers FLUX from a real checkpoint, and the exact MLPerf
  v5.1 eval sample-count / validation-timestep semantics. These are validated by
  internal reference runs but no committed test reproduces them.

Tracked follow-ups (file as public-repo issues):

1. A real-checkpoint forward-parity test (535M minimum) comparing Primus Flux
   against HF/Diffusers within a documented tolerance.
2. A robustness test for the MLPerf validation-timestep fallback path.

Treat any "bit-exact / matches NeMo / matches MLPerf / within X%" statement in
source comments as *asserted, unverified* until the parity test above lands.

## References

- [Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Megatron-LM FP8 Guide](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/llm/fp8.md)
- [FP8 Formats Explained (E4M3 vs E5M2)](https://arxiv.org/abs/2209.05433)

---

## Support

For issues or questions:

1. Check this guide first
2. Verify Transformer Engine and FP8 support (see Prerequisites)
3. Review logs for error messages
4. File issue with:
   - Hardware specs (GPU model, ROCm version)
   - Software versions (TE, PyTorch, Megatron-LM)
   - Config files used
   - Error logs

---

**Happy FP8 Training! 🚀**

*Last updated: 2026-01-10*

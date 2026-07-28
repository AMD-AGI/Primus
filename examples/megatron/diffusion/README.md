# Flux Diffusion Model Training Examples

Training examples for Flux diffusion models with Primus-Megatron on AMD GPUs.

## Related Documentation

- **Architecture & Developer Guide:** [docs/04-technical-guides/diffusion-models/README.md](../../../docs/04-technical-guides/diffusion-models/README.md)
- **API Reference:** [docs/04-technical-guides/diffusion-models/api_reference.md](../../../docs/04-technical-guides/diffusion-models/api_reference.md)
- **FP8 Training Guide:** [docs/04-technical-guides/diffusion-models/fp8_training.md](../../../docs/04-technical-guides/diffusion-models/fp8_training.md)
- **MXFP4 Training Guide:** [docs/04-technical-guides/diffusion-models/mxfp4_training.md](../../../docs/04-technical-guides/diffusion-models/mxfp4_training.md)
- **Dataset Preparation:** [primus/configs/data/megatron/diffusion/README.md](../../../primus/configs/data/megatron/diffusion/README.md)
- **Tests:** [tests/unit_tests/backends/megatron/diffusion/](../../../tests/unit_tests/backends/megatron/diffusion/)

---

## Quick Start

### Prerequisites

- AMD Instinct GPU(s) (MI300X, MI325X, MI355X)
- Docker or Podman with ROCm support
- Primus Docker image: `docker.io/rocm/primus:v26.1`
- Prepared dataset (see [Dataset Preparation](../../../primus/configs/data/megatron/diffusion/README.md))

### 5-Minute Test Run

1. **Prepare a small test dataset:**

```bash
mkdir -p /tmp/flux_test_data/raw && cd /tmp/flux_test_data/raw

for i in {000..099}; do
    convert -size 512x512 xc:blue sample_${i}.jpg
    echo "A blue square" > sample_${i}.txt
done

tar -cf train-000000.tar sample_*.jpg sample_*.txt

cat > dataset.yaml << 'EOF'
__module__: megatron.energon
__class__: CrudeWebdataset
subflavors:
  encoding: raw
EOF

energon prepare . --num-workers 4
```

2. **Launch training:**

```bash
EXP=examples/megatron/configs/MI300X/diffusion/flux_535m_pretrain.yaml \
DATA_PATH=/tmp/flux_test_data \
GPUS_PER_NODE=1 \
bash examples/run_pretrain.sh
```

---

## Model Variants

| Feature | Flux 535M | Flux 12B |
|---------|-----------|----------|
| Parameters | 535M | 12B |
| Joint Layers | 1 | 19 |
| Single Layers | 1 | 38 |
| Min GPUs | 1 | 8 (FSDP2 / DDP) |
| Recommended GPUs | 1-8 | 8-64 |
| Sharding | DP | FSDP2 (ZeRO-2/3) or DDP + distributed optimizer |
| Best For | Testing | Production |

---

## Available Configurations

The same configs are provided for both MI300X (`examples/megatron/configs/MI300X/diffusion/`)
and MI355X (`examples/megatron/configs/MI355X/diffusion/`). The only differences
are hardware-tuned batch sizes (MI300X has 192GB HBM3, MI355X has 256GB), so MI300X
uses smaller default micro/global batch sizes on the 12B DDP configs.

### Shared (MI300X and MI355X)

| Config | Description |
|--------|-------------|
| `flux_535m_pretrain.yaml` | Flux 535M, BF16, single/multi-GPU |
| `flux_535m_pretrain_fp8.yaml` | Flux 535M with FP8 precision |
| `flux_535m_with_guidance_embed.yaml` | Flux 535M with guidance embedding |
| `flux_12b_fsdp2_energon_schnell_resample_local_spec.yaml` | Flux 12B, FSDP2, BF16, local spec |
| `flux_12b_fsdp2_energon_schnell_resample_local_spec_fp8.yaml` | Flux 12B, FSDP2, FP8, local spec |
| `flux_12b_ddp_energon_schnell_resample_local_spec_fp8.yaml` | Flux 12B, DDP, FP8, local spec (delayed scaling) |
| `flux_12b_ddp_energon_schnell_resample_te_spec.yaml` | Flux 12B, DDP, BF16, TransformerEngine spec |
| `flux_12b_ddp_energon_schnell_resample_te_spec_fp8.yaml` | Flux 12B, DDP, FP8, TransformerEngine spec |

### MI355X only

| Config | Description |
|--------|-------------|
| `flux_12b_ddp_energon_schnell_resample_local_spec_mxfp4.yaml` | Flux 12B, DDP, MXFP4, local spec |
| `flux_12b_ddp_energon_schnell_resample_local_spec_fp8_mlperf.yaml` | MLPerf benchmark reproduction (local spec FP8) |
| `flux_12b_ddp_energon_schnell_resample_te_spec_fp8_mlperf.yaml` | MLPerf benchmark reproduction (TE spec FP8) |

> The `*_mlperf.yaml` configs reproduce the MLPerf Training Flux.1 benchmark
> (MLPerf logging + convergence target). Use the non-MLPerf configs above for
> general training.

---

## Training Modes

### Single-Node Training

```bash
# Flux 535M (1-8 GPUs)
EXP=examples/megatron/configs/MI300X/diffusion/flux_535m_pretrain.yaml \
GPUS_PER_NODE=8 \
bash examples/run_pretrain.sh

# Flux 12B (FSDP2, BF16, 8 GPUs)
EXP=examples/megatron/configs/MI300X/diffusion/flux_12b_fsdp2_energon_schnell_resample_local_spec.yaml \
GPUS_PER_NODE=8 \
bash examples/run_pretrain.sh
```

### Multi-Node Training (SLURM)

```bash
export DOCKER_IMAGE="docker.io/rocm/primus:v26.1"
export NNODES=8
export GPUS_PER_NODE=8

EXP=examples/megatron/configs/MI300X/diffusion/flux_12b_fsdp2_energon_schnell_resample_local_spec.yaml \
bash examples/run_slurm_pretrain.sh
```

### MLPerf Benchmark Reproduction (MI355X)

The `*_mlperf.yaml` configs reproduce the MLPerf Training Flux.1 benchmark and are
intended for benchmark reproduction rather than general training.

```bash
# Step 1: Ingest MLPerf data
primus-cli direct -- data diffusion-ingest \
  --config primus/configs/data/megatron/diffusion/preprocessing/mlperf_flux1.yaml

# Step 2: Train (configs already set vae_latent_mode: resample, vae_scale: 0.3611, vae_shift: 0.1159)
EXP=examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_fp8_mlperf.yaml \
GPUS_PER_NODE=8 \
bash examples/run_pretrain.sh
```

---

## FP8 Training

FP8 provides ~2x memory reduction and 1.5-2x training speedup on AMD MI300X/MI355X GPUs.

```bash
# Quick validation with 535M
EXP=examples/megatron/configs/MI300X/diffusion/flux_535m_pretrain_fp8.yaml \
GPUS_PER_NODE=1 \
bash examples/run_pretrain.sh

# Production with 12B (TransformerEngine FP8)
EXP=examples/megatron/configs/MI300X/diffusion/flux_12b_ddp_energon_schnell_resample_te_spec_fp8.yaml \
GPUS_PER_NODE=8 NNODES=4 \
bash examples/run_slurm_pretrain.sh

# Production with 12B (local-spec FP8, no TransformerEngine dependency)
EXP=examples/megatron/configs/MI300X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_fp8.yaml \
GPUS_PER_NODE=8 NNODES=4 \
bash examples/run_slurm_pretrain.sh
```

| Model | Precision | Memory/GPU | Batch Size | Speed |
|-------|-----------|------------|------------|-------|
| Flux 535M | BF16 | ~7-10GB | 2 | 1.0x |
| Flux 535M | FP8 | ~3-5GB | 4 | 1.5-2x |
| Flux 12B | BF16 | ~40-50GB | 1 | 1.0x |
| Flux 12B | FP8 | ~20-25GB | 2 | 1.5-2x |

For configuration details, tuning recipes, benchmarks, and troubleshooting, see the [FP8 Training Guide](../../../docs/04-technical-guides/diffusion-models/fp8_training.md).

---

## MXFP4 Training

MXFP4 (E2M1 + E8M0 block-of-32 scales) Flux 12B training on MI355X is supported via the local-spec provider (`PrimusTurboMXFP4LocalSpecProvider`, no TransformerEngine dependency). Forward and weight GEMMs run in FP4 through Primus-Turbo + AITER; attention, the optimizer state, and inter-rank communication stay in BF16.

```bash
# Path to a checkout of the `tuned_gemm_configs` directory.
# Set TUNED_GEMM_DIR to wherever you have the tuned configs available.
export TUNED_GEMM_DIR=${TUNED_GEMM_DIR:-/path/to/tuned_gemm_configs}

EXP=examples/megatron/configs/MI355X/diffusion/flux_12b_ddp_energon_schnell_resample_local_spec_mxfp4.yaml \
PRIMUS_TURBO_GEMM_BACKEND=FP4:AITER \
AITER_CONFIG_GEMM_A4W4=$TUNED_GEMM_DIR/mi355x/flux_12b.csv \
AITER_LOG_TUNED_CONFIG=1 \
bash examples/run_pretrain.sh
```

For configuration knobs, backend-selector semantics, tuned-GEMM verification, and troubleshooting, see the [MXFP4 Training Guide](../../../docs/04-technical-guides/diffusion-models/mxfp4_training.md).

---

## Converting HuggingFace Checkpoints

Convert pre-trained HuggingFace Flux checkpoints to Primus/Megatron-Core format:

```bash
python tools/checkpoint_conversion/convert_flux_hf_to_primus.py \
    --input black-forest-labs/FLUX.1-dev \
    --output checkpoints/primus_flux_12b.safetensors \
    --variant flux_12b
```

Supported variants: `flux_535m`, `flux_12b`, `custom` (with `--num-joint-layers` / `--num-single-layers`).

For gated models (FLUX.1-dev), set `export HF_TOKEN="your_token"` or run `huggingface-cli login`.

Primus also auto-detects tokens from `.hf_token` at the project root or `~/.cache/huggingface/token`.

---

## Configuration Reference

### Key Training Parameters

```yaml
modules:
  pre_trainer:
    framework: megatron
    config: pre_trainer.yaml
    model: diffusion/flux_535m.yaml
    trainer_class: FluxPretrainTrainer

    overrides:
      train_iters: 100000
      micro_batch_size: 2
      global_batch_size: 16
      lr: 1.0e-4
      min_lr: 1.0e-5
      weight_decay: 0.01
      clip_grad: 1.0
      lr_decay_style: cosine
      lr_warmup_iters: 1000
```

### Parallelism

The Flux 12B configs scale with FSDP2 (ZeRO-2/3) or DDP + distributed optimizer
rather than tensor/pipeline parallelism.

| Setting | Flux 535M | Flux 12B |
|---------|-----------|----------|
| `tensor_model_parallel_size` | 1 | 1 |
| `pipeline_model_parallel_size` | 1 | 1 |
| `context_parallel_size` | 1 | 1 |
| Sharding | DP | FSDP2 (`use_torch_fsdp2: true`) or DDP (`use_distributed_optimizer: true`) |

### Memory Optimization

```yaml
modules:
  pre_trainer:
    overrides:
      recompute_granularity: selective  # or 'full'
      recompute_method: block
      sequence_parallel: false
```

---

## Troubleshooting

### Dataset Not Found

Verify `data_path` in your config, ensure `energon prepare` was run, and check that `dataset.yaml` exists.

### Out of Memory (OOM)

Reduce `micro_batch_size`, increase `tensor_model_parallel_size`, enable `recompute_granularity: full`, or switch to pre-encoded data mode.

### NaN Loss

Reduce learning rate to `1.0e-5`, ensure `clip_grad: 1.0`, increase `lr_warmup_iters`, check dataset for corruption.

### NaN Loss with MLPerf Data

Ensure config includes the required normalization constants:

```yaml
vae_latent_mode: resample
vae_scale: 0.3611
vae_shift: 0.1159
```

### Encoder Download Fails

Set `export HF_TOKEN=your_token` or use a local model path via `encoder_model_path` in config overrides.

### Slow Training

Use pre-encoded data (2-3x faster), increase `num_workers`, enable `use_flash_attn: true`.

---

## Source Code Pointers

- **Model Architecture:** `primus/backends/megatron/core/models/diffusion/flux/`
- **Trainer:** `primus/modules/trainer/megatron/diffusion/flux_pretrain_trainer.py`
- **Data Pipeline:** `primus/backends/megatron/data/diffusion/`
- **Model Configs:** `primus/configs/models/megatron/diffusion/`

## Getting Help

- [GitHub Issues](https://github.com/AMD-AGI/Primus/issues)
- [GitHub Discussions](https://github.com/AMD-AGI/Primus/discussions)

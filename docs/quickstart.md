# Primus Quick Start Guide

Get up and running with Primus in 5 minutes!

## 📋 Prerequisites

- AMD ROCm drivers (≥ 7.0)
- Docker (≥ 24.0) with ROCm support
- AMD Instinct GPUs (MI300X, MI325X, etc.)

```bash
# Quick verification
rocm-smi && docker --version
```

---

## 🚀 Three Steps to Your First Training

### 1. Get Primus

```bash
# Pull Docker image
docker pull docker.io/rocm/primus:v25.9_gfx942

# Clone repository
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
```

### 2. Verify Installation

```bash
# Run a quick benchmark in container
./runner/primus-cli container --image rocm/primus:v25.9_gfx942 \
  -- benchmark gemm -M 4096 -N 4096 -K 4096
```

**Expected output:**
```
[BENCH] Markdown saved: ./gemm_report.md (overwrite)
[✔] GEMM benchmark finished. Results saved to ./gemm_report.md
```

✅ See this output? You're ready to train!

### 3. Run Training

Use the Docker image you just pulled:

```bash
# Run training in container (recommended for getting started)
./runner/primus-cli container --image rocm/primus:v25.9_gfx942 \
  -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
```

**Other modes:**

```bash
# Direct mode (if running on bare metal with ROCm installed)
./runner/primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml

# Slurm mode (for multi-node cluster)
./runner/primus-cli slurm srun -N 8 -p gpu -- container --image rocm/primus:v25.9_gfx942 \
  -- train pretrain --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
```

---

## 🎓 Command Structure

```bash
primus-cli [options] <mode> [mode-args] -- [command]
           ↓         ↓      ↓             ↓
           Global    Mode   Mode-specific Training
           options          parameters    command
```

**Common options:**
- `--debug` - Verbose logging
- `--dry-run` - Preview without executing

---

## 📚 Next Steps

**Learn More:**
- [CLI User Guide](./cli/PRIMUS-CLI-GUIDE.md) - Complete reference
- [CLI Architecture](./cli/CLI-ARCHITECTURE.md) - Design deep dive
- [Configuration Guide](./configuration.md) - YAML configuration
- [Examples](../examples/README.md) - Real-world templates

**Need Help?**
- [FAQ](./faq.md) - Common questions
- [GitHub Issues](https://github.com/AMD-AIG-AIMA/Primus/issues) - Report bugs

---

**That's it! Start training with `primus-cli` 🚀**

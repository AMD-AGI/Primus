# Primus Quick Start Guide

Get up and running with Primus in 5 minutes! This guide will walk you through your first training run on AMD GPUs.

## 📋 Prerequisites

Before you begin, ensure you have:

- **AMD ROCm drivers** (version ≥ 6.0 recommended)
- **Docker** (version ≥ 24.0) with ROCm support
- **ROCm-compatible AMD GPUs** (e.g., Instinct MI300X, MI250X)
- **Proper permissions** for Docker and GPU device access

### Verify Your Setup

```bash
# Check ROCm installation
rocm-smi

# Check Docker installation
docker --version

# Verify GPU access
ls -la /dev/kfd /dev/dri
```

---

## 🚀 Step 1: Get Primus

### Pull the Docker Image

```bash
docker pull docker.io/rocm/primus:v25.9_gfx942
```

### Clone the Repository

```bash
git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git
cd Primus
```

---

## 🎯 Step 2: Run Your First Command

Let's start with a simple GEMM benchmark to verify everything works:

```bash
# Run GEMM benchmark directly on your host
primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096
```

**Expected Output:**
```
[benchmark][INFO]: Starting GEMM benchmark...
[benchmark][INFO]: Matrix dimensions: M=4096, N=4096, K=4096
[benchmark][INFO]: Performance: XX.XX TFLOPS
```

✅ If you see performance metrics, congratulations! Primus is working correctly.

---

## 🏋️ Step 3: Run Your First Training

Now let's run a real training job. Primus CLI supports three execution modes:

### Option A: Direct Mode (Local Development)

Run directly on your current host:

```bash
primus-cli direct -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
```

**When to use:**
- ✅ Local development and debugging
- ✅ Single-node training
- ✅ Quick experiments

### Option B: Container Mode (Isolated Environment)

Run in a Docker container with isolated dependencies:

```bash
primus-cli container \
  --image rocm/primus:v25.9_gfx942 \
  --volume /data:/data \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
```

**When to use:**
- ✅ Ensuring environment consistency
- ✅ Testing different versions
- ✅ CI/CD pipelines

### Option C: Slurm Mode (Multi-Node Production)

Run distributed training on a Slurm cluster:

```bash
primus-cli slurm srun -N 8 -p gpu \
  -- train pretrain \
  --config examples/megatron/configs/MI300X/llama2_7B-pretrain.yaml
```

**When to use:**
- ✅ Multi-node distributed training
- ✅ Large-scale model training
- ✅ Production environments

---

## 🎓 Step 4: Understanding the Command

Let's break down the command structure:

```bash
primus-cli [global-options] <mode> [mode-args] -- [primus-commands]
│          │                 │      │             │  │
│          │                 │      │             │  └─ Your training command
│          │                 │      │             └─ Separator (required)
│          │                 │      └─ Mode-specific arguments
│          │                 └─ Execution mode (direct/container/slurm)
│          └─ Global options (--debug, --config, --dry-run)
└─ Primus CLI entry point
```

### Common Global Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config FILE` | Specify config file | `--config prod.yaml` |
| `--debug` | Enable debug logging | `--debug` |
| `--dry-run` | Show command without executing | `--dry-run` |

---

## 🔍 Step 5: Verify Your Training

### Check Training Logs

```bash
# Logs are typically in the output directory
tail -f output/exp_*/logs/train.log
```

### Monitor GPU Usage

```bash
# In another terminal
watch -n 1 rocm-smi
```

### Check Training Metrics

Look for output like:
```
[train][INFO]: Epoch 1/10, Step 100/1000
[train][INFO]: Loss: 3.245, Throughput: 1234 tokens/sec
[train][INFO]: GPU Memory: 45.2 GB / 128 GB
```

---

## 🎯 Common Use Cases

### Quick Debugging

```bash
# Use debug mode to see detailed execution flow
primus-cli --debug direct -- train pretrain --config config.yaml
```

### Dry Run (See Command Without Executing)

```bash
# Useful for verifying your configuration
primus-cli --dry-run slurm srun -N 4 -- train pretrain --config config.yaml
```

### Custom Configuration

```bash
# Use your own config file
primus-cli --config my-config.yaml container -- train pretrain
```

---

## 🐛 Troubleshooting

### Issue: "No container runtime found"

**Solution:**
```bash
# Install Docker or Podman
# For Docker:
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

### Issue: "Permission denied accessing GPU devices"

**Solution:**
```bash
# Add your user to the video and render groups
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Re-login for changes to take effect
```

### Issue: "Config file not found"

**Solution:**
```bash
# Use absolute path or relative to current directory
primus-cli --config ./configs/my-config.yaml direct -- train pretrain

# Or use the default config
primus-cli direct -- train pretrain
```

### Issue: Training is slow

**Solution:**
```bash
# Check GPU utilization
rocm-smi

# Enable debug mode to see bottlenecks
primus-cli --debug direct -- train pretrain --config config.yaml

# Verify you're using the right GPU count
primus-cli direct -- benchmark gemm  # Should show high TFLOPS
```

---

## 📚 Next Steps

Congratulations! You've completed the quick start guide. Here's what to explore next:

### Learn More About Primus CLI

- **[CLI User Guide](./cli/PRIMUS-CLI-GUIDE.md)** - Complete command reference and examples
- **[CLI Architecture](./cli/CLI-ARCHITECTURE.md)** - Understand the design and internals

### Advanced Topics

- **[Configuration Guide](./configuration.md)** - YAML configuration in depth
- **[Slurm & Container Usage](./slurm-container.md)** - Distributed training workflows
- **[Benchmark Suite](./benchmark.md)** - Performance testing and optimization

### Get Help

- **[FAQ](./faq.md)** - Common questions and answers
- **[Examples](../examples/README.md)** - Real-world training examples
- **[GitHub Issues](https://github.com/AMD-AIG-AIMA/Primus/issues)** - Report bugs or ask questions

---

## 💡 Pro Tips

### Tip 1: Use Configuration Files

Instead of long command lines, use YAML config files:

```yaml
# my-training.yaml
main:
  debug: false

container:
  options:
    image: "rocm/primus:v25.9_gfx942"
    memory: "256G"
    cpus: "32"

slurm:
  nodes: 8
  time: "12:00:00"
  partition: "gpu"
```

Then run:
```bash
primus-cli --config my-training.yaml slurm srun -- train pretrain
```

### Tip 2: Start Small, Scale Up

1. **Debug locally**: `primus-cli direct --`
2. **Test in container**: `primus-cli container --`
3. **Scale to cluster**: `primus-cli slurm srun -N 8 --`

### Tip 3: Use Dry Run for Verification

Always verify your command before running expensive jobs:

```bash
primus-cli --dry-run slurm sbatch -N 64 -t 72:00:00 -- train pretrain
```

---

**Ready to train? Happy training with Primus! 🚀**

# 🚀 From chaos to order: Building a unified entry point for AMD GPU LLM training

> ⚠️ **NOTE**: This is a draft version and not the final release.
>
> **Author**: AMD AI Brain - Training at Scale (TAS) Team
> **Published**: 2025-11-10
> **Tags**: `#ROCm` `#LLM-Training` `#Primus` `#DevTools` `#AMD-GPU`

---

## 📖 The beginning: Pain points in training workflows

Imagine this scenario:

You've just taken over a large model training project, and the project directory is littered with dozens of Bash scripts: `setup_env_mi300x.sh`, `run_megatron_8node.sh`, `check_network.sh`, `preprocess_data_v2_final_really.sh`... Each script has its own logic, and they're tightly coupled with each other.

When you want to reproduce a training run on a new GPU cluster, you need to:
1. Manually detect the GPU model and find the corresponding environment configuration script
2. Modify distributed parameters based on the number of nodes
3. Ensure data preprocessing scripts execute before training
4. Manually set dozens of environment variables
5. Pray that everything runs smoothly...

**This is the real problem we encountered when building the Primus platform.**

In the AMD GPU large model training ecosystem, we face complexity at multiple levels:
- 🔧 **Environment Setup**: Different GPU models (MI300X, MI250X) require different ROCm configurations
- 🔗 **Network Topology**: RCCL/NCCL environments and InfiniBand configurations vary widely
- 🎯 **Framework Orchestration**: Megatron, TorchTitan, JAX and other frameworks each have their own characteristics
- 🖥️ **Execution Environment**: Three scenarios - local development, containerization, and Slurm clusters
- 📊 **Performance Validation**: GEMM benchmarks, communication performance testing
- 🛠️ **Pre/Post Processing**: Data preprocessing, environment checks, hotfix patches

The traditional approach is to use a large number of Bash scripts to handle these aspects, but this brings:
- ❌ Repeated logic, difficult to maintain
- ❌ Lack of unified error handling
- ❌ Experiments difficult to reproduce
- ❌ High barrier to entry for newcomers

**Primus CLI was born to solve these pain points.**

---

## 💡 Design philosophy: One command, done

Our core philosophy is simple: **One command, from environment configuration to training launch, fully automated.**

```bash
# Just this simple!
primus-cli direct -- train pretrain --config deepseek_v2.yaml
```

### 🏗️ Three-layer architecture design

Primus CLI adopts a clear **three-layer structure + plugin system**:

```
┌─────────────────────────────────────────────────────┐
│                   Runtime Layer                     │
│            direct | container | slurm               │
│  Auto-detect env, configure GPU, manage distributed │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                Hook/Patch System                    │
│    Data preprocessing | Env checks | Hotfixes       │
│           Pluggable pre/post task logic             │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│               Task Execution Layer                  │
│       train | benchmark | preflight | analyze       │
│        Plugin-based business logic and tasks        │
└─────────────────────────────────────────────────────┘
```

### 🎯 Four design goals

| Goal | Implementation | User Benefits |
|------|---------------|---------------|
| **🔄 Consistency** | Unified CLI entry, supports Megatron/TorchTitan/JAX | No need to learn different commands for different frameworks |
| **🔌 Extensibility** | Auto-discovery plugins, dynamic registration of new features | Add new features without modifying core code |
| **🐛 Debuggability** | Rank-aware logging, detailed execution tracing | Quickly locate multi-node training issues |
| **📦 Reproducibility** | Auto-export runtime environment and config | One-click reproduce any historical experiment |

---

## 🔍 Deep dive: Architecture dissection

### ⚙️ Layer 1: Intelligent runtime abstraction

Different scenarios require different runtime environments, but users shouldn't have to worry about these details. Primus CLI provides three seamlessly switchable runtime modes:

| 🎭 Mode | 📝 Use Case | 🎯 Typical Command |
|---------|------------|-------------------|
| **🖥️ Direct** | Local dev, quick validation | `primus-cli direct -- train pretrain` |
| **🐳 Container** | Environment isolation, dependency management | `primus-cli container --image rocm/megatron:v25.8 -- train pretrain` |
| **🖧 Slurm** | Multi-node production training | `primus-cli slurm srun -N 8 -- train pretrain` |

**Key Highlight**: These three modes share the same command syntax, differing only in runtime environment preparation:

```bash
# Local testing
primus-cli direct -- benchmark gemm -M 4096

# Container testing (ensure environment consistency)
primus-cli container -- benchmark gemm -M 4096

# Production environment (8-node cluster)
primus-cli slurm srun -N 8 -- benchmark gemm -M 4096
```

**From development to production, just change the runtime mode - commands and parameters stay the same!**

---

### 🔁 Layer 2: Hook and patch system

Training is more than just running a Python script. You might need to:
- 🗂️ Preprocess datasets before training
- 🔍 Check GPU and network environment
- 🩹 Apply temporary hotfixes
- 📊 Collect and report metrics

Primus CLI's Hook system automates all of this:

```bash
# Directory structure
runner/helpers/hooks/
└── train/
    └── pretrain/
        ├── 01_check_environment.sh
        ├── 02_prepare_dataset.sh
        └── 03_setup_monitoring.sh
```

When you run `primus-cli train pretrain`, these Hooks execute automatically in order. No manual invocation, no training code modification needed.

The **Patch system** is for more flexible scenarios:

```bash
primus-cli direct --patch fixes/workaround_rccl.sh \
  -- train pretrain --config config.yaml
```

This is especially useful when you need to quickly apply temporary fixes or make environment-specific adjustments.

---

### 🧩 Layer 3: Task execution layer

This layer is responsible for executing specific business logic—training, testing, environment checks, and other actual tasks. Remember we said "zero-intrusion extension"? How is this achieved?

```bash
# Training task
primus-cli direct -- train pretrain --config deepseek_v2.yaml

# Performance testing task
primus-cli direct -- benchmark gemm --dtype bf16 -M 8192

# Environment check task
primus-cli direct -- preflight check --gpu --network
```

Each task is an independent Python plugin module, auto-registered via decorators:

```python
from primus.cli.registry import register_subcommand

@register_subcommand("train")
def run_train(args, unknown_args):
    # Your training business logic
    ...
```

**Want to add a new task?** Just create a new plugin file in `primus/cli/subcommands/` without modifying any core code. For example, if you want to add a topology analysis task:

```bash
# Add file: primus/cli/subcommands/analyze.py
# And you can use it directly
primus-cli direct -- analyze topology --visualize
```

This plugin-based design allows Primus CLI to quickly respond to new requirements while keeping the core stable as functionality expands.

---

## 🌐 The magic behind: Intelligent environment detection

This is probably the most "black tech" part of Primus CLI.

### Problem: Different GPUs need different configurations

AMD's GPU family is rich: MI300X, MI250X, MI210... Each GPU has its optimal ROCm configuration and environment variable settings. The traditional approach is to let users manually select configurations, but this is both error-prone and insufficiently automated.

### Solution: Three-step auto-configuration

**Step 1: Load Common Environment**

```bash
# base_env.sh provides unified logging and utility functions
source "${SCRIPT_DIR}/base_env.sh"
```

**Step 2: Auto-Detect GPU**

```bash
# Intelligently detect the current node's GPU model
GPU_MODEL=$(bash "${SCRIPT_DIR}/detect_gpu.sh")
# Output: MI300X
```

**Step 3: Load GPU-Specific Configuration**

```bash
# Based on detection result, auto-load optimal configuration
GPU_CONFIG_FILE="${SCRIPT_DIR}/${GPU_MODEL}.sh"
source "$GPU_CONFIG_FILE"  # Load MI300X.sh
```

Now, `MI300X.sh` can contain all best practices for this GPU model:
- ROCm environment variables (`HSA_*`, `HIP_*`)
- RCCL communication optimization parameters
- Memory management strategies
- Performance tuning options

**Users don't need to worry about these details at all - everything is automatic.**

### Real-world example

```bash
# On MI300X cluster
primus-cli direct -- train pretrain --config config.yaml
# Auto-loads: MI300X.sh → Optimized RCCL configuration

# Switch to MI250X cluster, same command
primus-cli direct -- train pretrain --config config.yaml
# Auto-loads: MI250X.sh → Adapted different configuration
```

**Cross-GPU migration with zero configuration changes!**

---

## 🧪 Foundation of scientific experiments: Reproducibility

In machine learning research, reproducibility is crucial. But reality is harsh:

> *"I ran this experiment three months ago, now I want to reproduce it... Wait, where's the config file? How did I set those environment variables again?"*

Does this sound familiar? Primus CLI completely solves this problem with an **automated snapshot mechanism**.

### Auto-record everything

Every time training starts, Primus CLI automatically saves the complete runtime context:

```
output/exp_2025_11_10_134522/
├── env/
│   ├── primus_env_dump.txt      # All environment variables
│   ├── gpu_model.txt            # GPU model info
│   ├── rocm_version.txt         # ROCm version
│   └── system_info.json         # System configuration
├── config/
│   ├── primus_config.yaml       # Primus config
│   ├── model_config.yaml        # Model config
│   └── data_config.yaml         # Data config
├── logs/
│   └── launch.log               # Launch logs
└── metadata.json                # Runtime metadata
```

### One-click reproduction

Three months later, when you want to reproduce this experiment:

```bash
# Just this simple!
primus-cli direct -- train pretrain --replay output/exp_2025_11_10_134522/
```

Primus CLI will automatically:
1. Restore all environment variables
2. Load original configuration files
3. Verify GPU and system environment
4. Start training (if environment is compatible)

### Real-world value

| Scenario | Traditional Approach | Using Primus CLI |
|----------|---------------------|------------------|
| **🔁 Performance Comparison** | Manually record config, easy to miss details | Auto-snapshot, precise reproduction |
| **🐛 Bug Debugging** | Hard to reproduce issues in new environment | `--replay` one-click reproduction |
| **📊 A/B Testing** | Manually ensure two runs are consistent | Auto-guarantee config consistency |
| **🏆 Paper Experiments** | Manually write experiment setup docs | One-click export complete environment |
| **🔄 Version Regression** | Rely on manual records | Automated CI integration |

---

## 📊 Real-world case: From development to production

Let's see how Primus CLI simplifies the entire workflow through a real scenario.

### Scenario: Training DeepSeek-V2 model

**Step 1: Local Development & Validation** 🖥️

```bash
# Quickly verify configuration on dev machine
primus-cli direct --debug -- train pretrain \
  --config configs/deepseek_v2_debug.yaml
```

Discover config issues, data format errors, etc. within minutes.

**Step 2: Container Environment Testing** 🐳

```bash
# Ensure it runs normally in standardized environment
primus-cli container \
  --image rocm/megatron-lm:v25.8_py310 \
  --mount /data:/data \
  -- train pretrain --config configs/deepseek_v2_debug.yaml
```

Containers ensure environment consistency, avoiding "works on my machine" problems.

**Step 3: Small-Scale Cluster Validation** 🖧

```bash
# 2-node test, verify distributed communication
primus-cli slurm srun -N 2 -p gpu-test \
  -- container --image rocm/megatron-lm:v25.8_py310 \
  -- train pretrain --config configs/deepseek_v2_small.yaml
```

**Step 4: Production Large-Scale Training** 🚀

```bash
# 64-node, 512-GPU production training
primus-cli slurm sbatch \
  -N 64 -p gpu-prod -t 72:00:00 \
  --job-name=deepseek-v2-prod \
  -o logs/train_%j.log \
  -- container --image rocm/megatron-lm:v25.8_py310 \
  -- train pretrain --config configs/deepseek_v2_prod.yaml
```

### Key insight

Notice? **From development to production, the core command structure remains unchanged**:
```
primus-cli <runtime> -- <subcommand> <args>
```

Only the runtime environment (`direct` → `container` → `slurm`) changes - everything else is unified.

---

## 🎯 Core advantages summary

After the detailed introduction above, let's summarize the core value Primus CLI brings:

| 🎁 Feature | 💪 Capability | 🚀 Value |
|-----------|--------------|----------|
| **Unified Entry** | One command, fits all scenarios | Lower learning curve, higher development efficiency |
| **Plugin-Based** | Zero-intrusion extension of new features | Quickly respond to new requirements, keep system stable |
| **Intelligent Environment** | Auto-detect GPU and optimize config | Zero cost for cross-platform migration |
| **Hook System** | Automate pre/post task processing | Decouple complex workflows, improve code reuse |
| **Reproducibility** | One-click snapshot and restore | Solid foundation for scientific experiments |
| **Three-Layer Runtime** | Direct/Container/Slurm seamless switching | Smooth path from development to production |
| **Unified Logging** | Rank-aware structured logging | Quickly locate multi-node issues |

---

## 🛣️ Future roadmap

Primus CLI continues to evolve, and our near-term plans include:

### Short-term goals (2025)
- 🎯 **Python Hook API**: Support writing Hooks in Python scripts for more flexible extension capabilities
- 🎯 **Intelligent Preflight**: Auto-check GPU health, network topology, InfiniBand connectivity before launch
- 🎯 **Configuration Template System**: Built-in best practice config templates for common models
- 🎯 **Performance Analysis Reports**: Auto-generate training performance analysis reports including GEMM, communication, I/O bottleneck analysis
- 🎯 **Enhanced Error Diagnostics**: Intelligent error analysis and fix suggestions (e.g., RCCL hang, OOM, and other common issues)
- 🎯 **Extended Framework Support**: Improve support for more training frameworks like TorchTitan, JAX/Flax
- 🎯 **CI/CD Integration**: Provide standardized testing and validation workflows, support automated regression testing

### Long-term vision
- 🌟 Become the **standard training entry point for the ROCm ecosystem**

---

## 🎓 Summary: The power of one command

Back to the question at the beginning: How do we make large model training go from complex to simple?

Primus CLI's answer: **Through carefully designed abstraction layers, hide complexity under a unified interface.**

```bash
# From this simple command...
primus-cli direct -- train pretrain --config deepseek_v2.yaml

# ...to everything behind it:
# ✅ Auto GPU detection and configuration
# ✅ Intelligent environment variable setup
# ✅ Data preprocessing Hooks
# ✅ Distributed communication optimization
# ✅ Log collection and analysis
# ✅ Experiment snapshots and reproduction
# ✅ Error handling and recovery
```

**This is Primus CLI: Making complex things simple, and simple things automatic.**

---

## 📚 Learn more

- 📖 **CLI Reference (user guide)**: [cli-reference.md](../02-user-guide/cli-reference.md)
- 🏛 **System Architecture**: [architecture.md](./architecture.md)
- 🔧 **Quick Start**: `primus-cli --help`
- 💬 **Issue Reporting**: GitHub Issues
- 🌐 **ROCm Ecosystem**: [rocm.github.io](https://rocm.github.io)

---

> *"The best interface is no interface."*
>
> But where interfaces cannot be eliminated, the best interface is **unified, simple, and automated**. That's Primus CLI.

**Happy Training with AMD ROCm! 🎉**

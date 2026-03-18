# Primus CLI Documentation

The Primus CLI provides a unified command-line interface for training, benchmarking, and managing large-scale foundation model workflows on AMD GPUs.

## 📚 Documentation Index

### User Guides

- **[User Guide](./PRIMUS-CLI-GUIDE.md)**
  - Complete user manual with examples
  - Quick start guide
  - Execution modes (Direct / Container / Slurm)
  - Configuration files and options
  - Best practices and troubleshooting

- **[Architecture Deep Dive](./CLI-ARCHITECTURE.md)**
  - Design philosophy and principles
  - Three-layer architecture explained
  - Plugin system and extensibility
  - Intelligent environment detection
  - From development to production workflow

## 🚀 Quick Start

### Basic Syntax

```bash
primus-cli [global-options] <mode> [mode-args] -- [Primus commands]
```

### Running from a source checkout

If you're running from the Primus repo root (after `git clone ... && cd Primus`), you can invoke the launcher as:

```bash
./primus-cli [global-options] <mode> [mode-args] -- [Primus commands]
```

### Your First Command

```bash
# Run GEMM benchmark directly on current host
primus-cli direct -- benchmark gemm -M 4096 -N 4096 -K 4096
```

## 🎯 Three Execution Modes

| Mode | Use Case | Command Example |
|------|----------|-----------------|
| **Direct** | Local development, quick validation | `primus-cli direct -- train pretrain` |
| **Container** | Environment isolation, dependency management | `primus-cli container --image rocm/primus:v26.1 -- train pretrain` |
| **Slurm** | Multi-node distributed training | `primus-cli slurm srun -N 8 -- train pretrain` |

## 📖 Learn More

- For detailed usage instructions, see the [User Guide](./PRIMUS-CLI-GUIDE.md)
- For architecture and design details, see [Architecture Deep Dive](./CLI-ARCHITECTURE.md)
- For the main Primus documentation, see [Primus README](../../README.md)

## 🔗 Related Documentation

- [Quickstart Guide](../quickstart.md) - Get started with Primus in 5 minutes
- [Configuration Guide](../configuration.md) - YAML configuration explained
- [Slurm & Container Usage](../slurm-container.md) - Distributed workflows
- [Benchmark Suite](../benchmark.md) - Performance evaluation

---

**Happy Training with Primus CLI! 🚀**

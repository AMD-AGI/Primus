## 🎯 Summary

Add CLI-based example scripts for Primus training workflows. These new examples use the `primus-cli` command-line interface, making it easier to launch training jobs across different environments.

## 📚 New Example Scripts

### 1. **Local Training with Container** - `examples/run_local_pretrain_cli.sh`
```bash
# Run training in Docker/Podman container
bash examples/run_local_pretrain_cli.sh
```

**Features**:
- ✅ Automatic container setup with ROCm image
- ✅ Volume mounting for data access
- ✅ Environment variable configuration
- ✅ Support for both PyTorch and JAX/MaxText backends

**Usage**:
```bash
# Set backend (optional, default is PyTorch)
export BACKEND=MaxText  # or leave unset for PyTorch

# Set experiment config
export EXP=examples/megatron/exp_pretrain.yaml

# Run
bash examples/run_local_pretrain_cli.sh
```

### 2. **Direct Mode Training** - `examples/run_pretrain_cli.sh`
```bash
# Run training directly without container
bash examples/run_pretrain_cli.sh
```

**Features**:
- ✅ Simple, minimal example
- ✅ Direct execution on host
- ✅ No container overhead

**Usage**:
```bash
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_pretrain_cli.sh
```

### 3. **Slurm Cluster Training** - `examples/run_slurm_pretrain_cli.sh`
```bash
# Submit training job to Slurm cluster
bash examples/run_slurm_pretrain_cli.sh
```

**Features**:
- ✅ Slurm job submission
- ✅ Multi-node training support
- ✅ Automatic resource allocation

**Usage**:
```bash
export NNODES=4
export NODES_LIST="node[01-04]"
export EXP=examples/megatron/exp_pretrain.yaml
bash examples/run_slurm_pretrain_cli.sh
```

## 🎯 Benefits

1. **Easier to Use**: Simple bash scripts instead of complex command lines
2. **Consistent Interface**: All examples use `primus-cli` for unified experience
3. **Environment Aware**: Automatic backend detection (PyTorch/MaxText)
4. **Production Ready**: Support for container, direct, and cluster modes
5. **Well Documented**: Clear examples with comments and usage instructions

## ✅ Testing

- [x] All scripts tested with dry-run mode
- [x] Shellcheck compliant
- [x] Pre-commit hooks passing
- [x] No breaking changes to existing examples

## 📝 Migration Notes

**Existing scripts (`examples/run_pretrain.sh`) remain unchanged and fully functional.**

These new CLI-based scripts are alternatives that provide:
- Simpler syntax
- Better integration with `primus-cli`
- More consistent experience across environments

Users can choose to use either the original scripts or these new CLI-based versions.

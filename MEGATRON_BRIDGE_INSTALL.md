# Megatron-Bridge Installation Guide

This guide covers the installation of Megatron-Bridge dependencies for the Primus project.

## Quick Install

### 1. Install Core Dependencies

```bash
pip install -r runner/helpers/hooks/train/posttrain/requirements-megatron-bridge.txt
```

This installs:
- `qwen-vl-utils` - Qwen VL utilities
- `timm` - PyTorch Image Models
- `open-clip-torch>=3.2.0` - OpenCLIP
- `flash-linear-attention` - Flash Linear Attention
- `megatron-energon` - Multi-modal data loading framework
- `bitstring` - Audio/Video processing support

### 2. Install Git Dependencies (Optional but Recommended)

Some packages need to be installed from specific Git commits:

```bash
# Install most packages from Git
pip install -r runner/helpers/hooks/train/posttrain/requirements-megatron-bridge-git.txt

# Note: causal-conv1d requires special handling
pip install --no-build-isolation git+https://github.com/Dao-AILab/causal-conv1d.git@9d700d167c4ad299b0a5265ed1bdb4ee4a0ca111
```

This installs:
- `transformer-engine@release_v2.9` - NVIDIA TransformerEngine
- `mamba-ssm` - Mamba State Space Models
- `nvidia-resiliency-ext` - NVIDIA Resiliency Extension
- `causal-conv1d` - Causal Convolution 1D

### 3. Install Megatron-Core (Local)

```bash
cd third_party/Megatron-Bridge/3rdparty/Megatron-LM
pip install -e .
```

## Package Details

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| qwen-vl-utils | latest | Qwen Vision-Language utilities |
| timm | latest | Image models and transformations |
| open-clip-torch | >=3.2.0 | CLIP model implementation |
| flash-linear-attention | latest | Efficient attention mechanisms |
| megatron-energon | latest | Multi-modal dataset loading |
| bitstring | latest | Binary data processing for AV |

### Git Dependencies

| Package | Commit/Branch | Purpose |
|---------|---------------|---------|
| transformer-engine | release_v2.9 | NVIDIA optimized transformer layers |
| mamba-ssm | 6b32be06d026e170b3fdaf3ae6282c5a6ff57b06 | State space models |
| nvidia-resiliency-ext | 54f85fe422d296cf04ea524130014bd3a2c3add1 | Training fault tolerance |
| causal-conv1d | 9d700d167c4ad299b0a5265ed1bdb4ee4a0ca111 | Causal convolutions |

### Local Dependencies

| Package | Location | Purpose |
|---------|----------|---------|
| megatron-core | third_party/Megatron-Bridge/3rdparty/Megatron-LM/ | Core Megatron functionality |

## Known Issues

### nvidia-modelopt Incompatibility

**Issue**: `nvidia-modelopt` has compatibility issues with newer PyTorch versions:
```
ImportError: cannot import name '_type_utils' from 'torch.onnx'
```

**Solution**: Skip installation unless you specifically need quantization features.

### fsspec Version Conflict

**Issue**: `megatron-energon` requires `fsspec>=2026.1.0`, but `datasets` requires `fsspec<=2025.9.0`.

**Solution**: This typically doesn't cause runtime issues. If problems occur:
```bash
pip install 'fsspec[http]<=2025.9.0'
```

## Optional Dependencies

### NeMo Run (for recipe management)

```bash
pip install "nemo-run>=0.5.0a0,<0.6.0"
```

### Tensor Inspection

```bash
pip install nvdlfw-inspect==0.2.1
```

## Installation Script

For automated installation, use the provided script:

```bash
bash runner/helpers/hooks/train/posttrain/setup_env_megatron_bridge.sh
```

This script handles:
- Transformers patching
- Megatron-Bridge configuration
- All dependency installation
- Environment variable setup

## Verification

After installation, verify the setup:

```python
# Test basic imports
import megatron.bridge
import megatron.energon
import transformer_engine
from causal_conv1d import causal_conv1d_fn

print("✅ All dependencies installed successfully!")
```

## Build Requirements

If building from source (e.g., for TransformerEngine):

```bash
pip install setuptools torch pybind11 Cython>=3.0.0 numpy<2.0.0 ninja nvidia-mathdx
```

## System Requirements

- Python >= 3.10
- CUDA-capable GPU (for full functionality)
- ROCm support (AMD GPUs)
- Sufficient disk space for models and datasets

## Support

For issues or questions:
- Check the [Megatron-Bridge documentation](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- Review the [Primus issues](https://github.com/AMD-AGI/Primus/issues)

#!/usr/bin/env bash
###############################################################################
# Megatron-Bridge Dependencies Installation Script for AMD GPUs
#
# This script installs Megatron-Bridge dependencies adapted for AMD ROCm.
# Based on pyproject.toml with AMD-specific modifications.
###############################################################################

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Determine PRIMUS_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

log_info "PRIMUS_ROOT: $PRIMUS_ROOT"
log_info "================================================"
log_info "Installing Megatron-Bridge Dependencies for AMD"
log_info "================================================"

# Check ROCm installation
if python3 -c "import torch; print(torch.version.hip)" &>/dev/null; then
    ROCM_VERSION=$(python3 -c "import torch; print(torch.version.hip)" 2>/dev/null || echo "unknown")
    log_success "ROCm detected: $ROCM_VERSION"
else
    log_warn "ROCm not detected in PyTorch. Some features may not work optimally."
fi

###############################################################################
# 1. Core Dependencies
###############################################################################
log_info "[1/6] Installing core dependencies..."

pip install -U pip setuptools wheel

pip install \
    "transformers<5.0.0" \
    datasets \
    accelerate \
    "omegaconf>=2.3.0" \
    "tensorboard>=2.19.0" \
    typing-extensions \
    rich \
    "wandb>=0.19.10" \
    "six>=1.17.0" \
    "regex>=2024.11.6" \
    "pyyaml>=6.0.2" \
    "tqdm>=4.67.1" \
    "hydra-core>1.3,<=1.3.2"

log_success "Core dependencies installed"

###############################################################################
# 2. Model-specific Dependencies
###############################################################################
log_info "[2/6] Installing model-specific dependencies..."

pip install \
    qwen-vl-utils \
    timm \
    "open-clip-torch>=3.2.0"

log_success "Model-specific dependencies installed"

###############################################################################
# 3. Transformer-Engine (AMD ROCm Version)
###############################################################################
log_info "[3/6] Installing Transformer-Engine for ROCm..."

# AMD uses transformer-engine-torch instead of transformer-engine
if pip install transformer-engine-torch; then
    log_success "Transformer-Engine (ROCm) installed"
else
    log_error "Failed to install transformer-engine-torch"
    log_info "Trying alternative installation..."
    pip install transformer-engine-torch --no-cache-dir || {
        log_warn "transformer-engine-torch installation failed. Training may be slower."
    }
fi

###############################################################################
# 4. Megatron-Core (AMD Adapted Version)
###############################################################################
log_info "[4/6] Installing Megatron-Core (AMD version)..."

MEGATRON_LM_DIR="${PRIMUS_ROOT}/third_party/Megatron-LM"

if [ ! -d "$MEGATRON_LM_DIR" ]; then
    log_error "Megatron-LM not found at: $MEGATRON_LM_DIR"
    log_info "Please clone Megatron-LM to third_party/"
    exit 1
fi

cd "$MEGATRON_LM_DIR"

if pip install -e ".[dev,mlm]"; then
    log_success "Megatron-Core (AMD) installed from $MEGATRON_LM_DIR"
else
    log_error "Failed to install Megatron-Core"
    exit 1
fi

###############################################################################
# 5. Optional Dependencies
###############################################################################
log_info "[5/6] Installing optional dependencies..."

# Flash Linear Attention (optional, may not support ROCm)
log_info "  - Trying flash-linear-attention..."
if pip install flash-linear-attention; then
    log_success "    flash-linear-attention installed"
else
    log_warn "    flash-linear-attention not available, using fallback implementation"
fi

# Mamba (optional, only needed for Mamba-based models)
log_info "  - Skipping mamba-ssm (optional, install manually if needed)"
log_info "  - Skipping causal-conv1d (optional, install manually if needed)"

# NVIDIA-specific packages (skip on AMD)
log_info "  - Skipping nvidia-resiliency-ext (NVIDIA-specific)"
log_info "  - Skipping nvidia-modelopt (NVIDIA-specific, has compatibility issues)"
log_info "  - Skipping nv-grouped-gemm (NVIDIA-specific)"

log_success "Optional dependencies processed"

###############################################################################
# 6. Megatron-Bridge (from source)
###############################################################################
log_info "[6/6] Installing Megatron-Bridge..."

MEGATRON_BRIDGE_DIR="${PRIMUS_ROOT}/third_party/Megatron-Bridge"

if [ ! -d "$MEGATRON_BRIDGE_DIR" ]; then
    log_error "Megatron-Bridge not found at: $MEGATRON_BRIDGE_DIR"
    log_info "Please clone Megatron-Bridge to third_party/"
    exit 1
fi

cd "$MEGATRON_BRIDGE_DIR"

if pip install -e .; then
    log_success "Megatron-Bridge installed from $MEGATRON_BRIDGE_DIR"
else
    log_error "Failed to install Megatron-Bridge"
    exit 1
fi

###############################################################################
# Summary
###############################################################################
echo
log_success "================================================"
log_success "✅ Installation Complete!"
log_success "================================================"
echo

log_info "Installed packages:"
log_info "  ✅ Core dependencies (transformers, datasets, omegaconf, etc.)"
log_info "  ✅ Model dependencies (qwen-vl-utils, timm, open-clip-torch)"
log_info "  ✅ Transformer-Engine (ROCm version)"
log_info "  ✅ Megatron-Core (AMD adapted)"
log_info "  ✅ Megatron-Bridge"

echo
log_info "Skipped packages (AMD-incompatible or optional):"
log_info "  ❌ nvidia-resiliency-ext (NVIDIA-specific)"
log_info "  ❌ nvidia-modelopt (compatibility issues)"
log_info "  ⚠️  mamba-ssm (optional, install if needed)"
log_info "  ⚠️  flash-linear-attention (optional, may have failed)"

echo
log_info "Next steps:"
log_info "  1. Verify ROCm: python3 -c 'import torch; print(torch.version.hip)'"
log_info "  2. Test imports: python3 -c 'import megatron.bridge'"
log_info "  3. Run training: cd examples && python train.py"

echo
log_success "You're ready to use Megatron-Bridge on AMD GPUs! 🚀"

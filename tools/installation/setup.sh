#!/usr/bin/env bash
# setup.sh — Reproduce the Primus training environment in a Python venv
# (no sudo, no docker). Mirrors the Dockerfile pins, adapted for:
#   * Python 3.10 (cp310 wheels)
#   * ROCm provided by the pip `rocm-sdk-devel` wheel (no system ROCm needed)
#   * Builds on the big /tmp disk (home quota is tiny)
#   * GPU arch auto-detected (gfx942 and/or gfx950); apt/sudo steps skipped
#
# Usage:
#   bash setup.sh                # run all default stages in order
#   bash setup.sh <stage>...     # run only specific stage(s), e.g.
#   bash setup.sh venv torch flash_attn
#
# Stages are re-runnable. List them with:  bash setup.sh --list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

log()  { echo -e "\n\033[1;36m[setup] $*\033[0m"; }
die()  { echo -e "\033[1;31m[setup][ERROR] $*\033[0m" >&2; exit 1; }
reload_env() { source "$SCRIPT_DIR/env.sh"; }

# ---- pinned versions / commits (from the Dockerfile) ----
PYTORCH_VERSION="2.12.0+rocm7.14.0a20260608"
TORCH_INDEX="https://rocm.nightlies.amd.com/whl-multi-arch"

FA_REPO="https://github.com/ROCm/flash-attention.git"
FA_BRANCH="6387433156558135a998d5568a9d74c1778666d8"
TE_REPO="https://github.com/ROCm/TransformerEngine.git"
TE_BRANCH="e6ede467a49cfda1859b145e045109e2f330bccc"
TORCHTUNE_REPO="https://github.com/pytorch/torchtune.git"
TORCHTUNE_BRANCH="b4c98ac2a37f0397d64c22579aed415ce7264db6"
TORCHAO_REPO="https://github.com/pytorch/ao.git"
TORCHAO_BRANCH="e9c7bead90b840b280f97374308255957108ce47"
GROUPED_GEMM_REPO="https://github.com/caaatch22/grouped_gemm.git"
GROUPED_GEMM_BRANCH="rocm"
CAUSAL_CONV1D_REPO="https://github.com/Dao-AILab/causal-conv1d"
CAUSAL_CONV1D_BRANCH="e940ead2fd962c56854455017541384909ca669f"
MAMBA_REPO="https://github.com/AndreasKaratzas/mamba.git"
MAMBA_BRANCH="enable-primus-hybrid-models"
# tilelang 0.1.8 (mamba_ssm dep) breaks with apache-tvm-ffi 0.1.12; pin lower.
TVM_FFI_VERSION="0.1.6"
PRIMUS_REPO="https://github.com/AMD-AGI/Primus.git"
PRIMUS_BRANCH="9a1547cd5885c4de6ad0935a5d59c08303dd0674"
AITER_REPO="https://github.com/ROCm/aiter.git"
AITER_COMMIT="b5e03ed191fca11ee423226537ef8d9435e432a6"
TURBO_REPO="https://github.com/AMD-AGI/Primus-Turbo.git"
TURBO_COMMIT="3974fc246be594d989156dd83e91da618274b0c8"

PIP="python -m pip"

# Fresh clone helper: clone into transient SRC_DIR, build, then optionally clean.
fresh_clone() {  # fresh_clone <url> <dir> [extra git clone args...]
    local url="$1"; local dir="$2"; shift 2
    rm -rf "${SRC_DIR:?}/$dir"
    git clone "$@" "$url" "$SRC_DIR/$dir"
}

# ============================ STAGES ============================

stage_venv() {
    log "Creating venv at $VENV_DIR (Python: $(python3 --version))"
    mkdir -p "$PRIMUS_BASE" "$SRC_DIR" "$WORKSPACE_DIR"
    [ -f "$VENV_DIR/bin/activate" ] || python3 -m venv "$VENV_DIR"
    reload_env
    $PIP install --upgrade pip
    # Build front-end tooling. patchelf via pip (system one is missing/no sudo).
    $PIP install \
        pybind11 \
        typeguard \
        wheel==0.45.1 \
        cmake==3.31.6 \
        ninja==1.11.1.3 \
        packaging==25.0 \
        setuptools==75.1.0 \
        patchelf
    rm -rf /root/.cache 2>/dev/null || true
}

stage_torch() {
    reload_env
    log "Installing PyTorch ${PYTORCH_VERSION} + ROCm SDK (arch: $PYTORCH_ROCM_ARCH) from nightly index"
    # Early pip deps (Dockerfile block before torch)
    $PIP install \
        cxxfilt==0.3.0 \
        tqdm==4.67.3 \
        pyyaml==6.0.3 \
        pytest==9.0.3 \
        matplotlib==3.10.9 \
        pandas==2.3.3 \
        py-cpuinfo==9.0.0 \
        build==1.5.0

    $PIP uninstall -y torch || true

    # Per-arch device wheels, derived from PYTORCH_ROCM_ARCH (auto-detected in
    # env.sh, ";"-separated). Installs the matching gfx942 and/or gfx950 sets.
    local _arch arch_args=()
    local _arches; IFS=';' read -ra _arches <<< "$PYTORCH_ROCM_ARCH"
    for _arch in "${_arches[@]}"; do
        _arch="${_arch// /}"; [ -z "$_arch" ] && continue
        arch_args+=( "amd-torch-device-${_arch}==${PYTORCH_VERSION}" \
                     "rocm-sdk-device-${_arch}" \
                     "amd-torchvision-device-${_arch}" )
    done
    [ ${#arch_args[@]} -gt 0 ] || die "no GPU arch resolved; export PYTORCH_ROCM_ARCH (e.g. gfx942;gfx950)"
    log "Installing device wheels for: $PYTORCH_ROCM_ARCH"

    $PIP install \
        --index-url "$TORCH_INDEX" \
        --pre \
        "torch==${PYTORCH_VERSION}" \
        rocm-sdk-devel \
        torchaudio \
        torchvision \
        apex \
        "${arch_args[@]}"

    log "Running rocm-sdk init"
    rocm-sdk init
    reload_env
    [ -n "${ROCM_PATH:-}" ] || die "ROCM_PATH not resolved after rocm-sdk init"
    log "ROCM_PATH=$ROCM_PATH"
    python -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available())"
}

stage_flash_attn() {
    reload_env
    log "Building flash-attention @ $FA_BRANCH"
    fresh_clone "$FA_REPO" flash-attention --recursive
    ( cd "$SRC_DIR/flash-attention" \
        && git checkout "$FA_BRANCH" \
        && git submodule update --init --recursive \
        && python setup.py install ) || die "flash-attention build failed"
    rm -rf "$SRC_DIR/flash-attention"
}

stage_te() {
    reload_env
    log "Building TransformerEngine @ $TE_BRANCH"
    fresh_clone "$TE_REPO" TransformerEngine --recursive
    ( cd "$SRC_DIR/TransformerEngine" \
        && git checkout "$TE_BRANCH" \
        && git submodule update --init --recursive \
        && $PIP install psutil \
        && MAX_JOBS="$MAX_JOBS" $PIP install --no-build-isolation . ) || die "TransformerEngine build failed"
    rm -rf "$SRC_DIR/TransformerEngine"
}

stage_torchtune() {
    reload_env
    log "Installing torchtune @ $TORCHTUNE_BRANCH (with use_grouped_mm patch)"
    fresh_clone "$TORCHTUNE_REPO" torchtune
    ( cd "$SRC_DIR/torchtune" \
        && git checkout "$TORCHTUNE_BRANCH" \
        && sed -i 's/use_grouped_mm = True/use_grouped_mm = False/g' torchtune/modules/moe/utils.py \
        && $PIP install . ) || die "torchtune install failed"
    rm -rf "$SRC_DIR/torchtune"
}

stage_torchao() {
    reload_env
    log "Building torchao @ $TORCHAO_BRANCH (with pad_inner_dim + swizzle patches)"
    fresh_clone "$TORCHAO_REPO" ao
    ( cd "$SRC_DIR/ao" \
        && git checkout "$TORCHAO_BRANCH" \
        && sed -i 's/pad_inner_dim: bool = False/pad_inner_dim: bool = True/g' torchao/float8/config.py \
        && sed -i 's/if defined(HIPBLASLT_VEC_EXT)/if false/g' torchao/csrc/rocm/swizzle/swizzle.cpp \
        && $PIP install --no-build-isolation . ) || die "torchao build failed"
    rm -rf "$SRC_DIR/ao"
}

stage_pydeps() {
    reload_env
    log "Installing main pip dependency set"
    $PIP install \
        datasets==3.6.0 \
        av==16.0.1 \
        transformers==4.55.0 \
        optree==0.18.0 \
        sympy \
        accelerate==1.9.0 \
        trl==0.21.0 \
        tensorboard==2.20.0 \
        peft \
        scipy \
        einops \
        flask-restful \
        nltk \
        pytest \
        pytest-cov \
        pytest_mock \
        pytest-csv \
        pytest-random-order \
        sentencepiece \
        wrapt \
        zarr==2.18.7 \
        numcodecs==0.12.1 \
        xarray \
        wandb \
        tensorstore==0.1.45 \
        pybind11 \
        tiktoken \
        pynvml \
        z3-solver \
        "huggingface_hub[cli]"
    python -m nltk.downloader punkt_tab || true
}

stage_grouped_gemm() {
    reload_env
    log "Building grouped_gemm @ $GROUPED_GEMM_BRANCH"
    fresh_clone "$GROUPED_GEMM_REPO" grouped_gemm
    ( cd "$SRC_DIR/grouped_gemm" \
        && git checkout "$GROUPED_GEMM_BRANCH" \
        && git submodule update --init --recursive \
        && $PIP install --no-build-isolation . ) || die "grouped_gemm build failed"
    rm -rf "$SRC_DIR/grouped_gemm"
}

stage_causal_conv1d() {
    reload_env
    log "Building causal-conv1d @ $CAUSAL_CONV1D_BRANCH"
    fresh_clone "$CAUSAL_CONV1D_REPO" causal-conv1d
    ( cd "$SRC_DIR/causal-conv1d" \
        && git checkout "$CAUSAL_CONV1D_BRANCH" \
        && $PIP install --no-build-isolation . ) || die "causal-conv1d build failed"
    rm -rf "$SRC_DIR/causal-conv1d"
}

stage_mamba() {
    reload_env
    log "Building mamba @ $MAMBA_BRANCH"
    fresh_clone "$MAMBA_REPO" mamba --branch "$MAMBA_BRANCH"
    # IMPORTANT: use pip, NOT `python setup.py install`. The legacy easy_install
    # path does not recognize pip-installed packages and re-fetches the LATEST
    # of every unpinned dep as .egg files, clobbering our pins (it pulled
    # transformers 5.x, removed accelerate/trl, and dragged in NVIDIA CUDA
    # packages). pip respects already-installed versions.
    # tilelang (a hard mamba_ssm dep) needs an apache-tvm-ffi that predates the
    # 0.1.12 registry breakage; pin it here.
    ( cd "$SRC_DIR/mamba" \
        && $PIP install "apache-tvm-ffi==${TVM_FFI_VERSION}" \
        && $PIP install --no-build-isolation . ) || die "mamba build failed"
    rm -rf "$SRC_DIR/mamba"
}

stage_primus() {
    reload_env
    log "Installing Primus @ $PRIMUS_BRANCH (kept in $WORKSPACE_DIR/Primus)"
    rm -rf "$WORKSPACE_DIR/Primus"
    git clone --recurse-submodules "$PRIMUS_REPO" "$WORKSPACE_DIR/Primus" || die "Primus clone failed"
    ( cd "$WORKSPACE_DIR/Primus" \
        && git checkout "$PRIMUS_BRANCH" \
        && git submodule update --init --recursive \
        && $PIP install -r requirements.txt ) || die "Primus install failed"
    # Megatron's dataset indexing needs the pybind11 helpers_cpp extension
    # compiled, else training fails with:
    #   ModuleNotFoundError: No module named 'megatron.core.datasets.helpers_cpp'
    local mlm
    for mlm in "$WORKSPACE_DIR/Primus/third_party/Megatron-LM" "$HOME/.cache/Primus/third_party/Megatron-LM"; do
        if [ -f "$mlm/megatron/core/datasets/Makefile" ]; then
            log "Compiling Megatron helpers_cpp in $mlm"
            make -C "$mlm/megatron/core/datasets" || die "helpers_cpp build failed in $mlm"
        fi
    done
}

stage_aiter() {
    reload_env
    log "Building aiter @ $AITER_COMMIT (kept in $WORKSPACE_DIR/aiter)"
    $PIP uninstall aiter amd-aiter -y || true
    rm -rf "$WORKSPACE_DIR/aiter"
    git clone --recursive "$AITER_REPO" "$WORKSPACE_DIR/aiter" || die "aiter clone failed"
    ( cd "$WORKSPACE_DIR/aiter" \
        && git checkout "$AITER_COMMIT" \
        && git submodule update --init --recursive \
        && PREBUILD_KERNELS=3 $PIP install --no-cache-dir --use-pep517 . ) || die "aiter build failed"
}

stage_turbo() {
    reload_env
    log "Building Primus-Turbo @ $TURBO_COMMIT"
    fresh_clone "$TURBO_REPO" Primus-Turbo --recursive
    ( cd "$SRC_DIR/Primus-Turbo" \
        && git checkout "$TURBO_COMMIT" \
        && git submodule update --init --recursive \
        && $PIP install -r requirements.txt \
        && $PIP install --no-build-isolation . -v ) || die "Primus-Turbo build failed"
    rm -rf "$SRC_DIR/Primus-Turbo"
}

stage_boto() {
    reload_env
    log "Installing boto3/botocore"
    $PIP install boto3==1.35.42 botocore==1.35.99
}

stage_manifest() {
    reload_env
    log "Writing manifest to $WORKSPACE_DIR/.manifest"
    mkdir -p "$WORKSPACE_DIR/.manifest"
    env > "$WORKSPACE_DIR/.manifest/env.txt"
    $PIP list > "$WORKSPACE_DIR/.manifest/requirements.txt"
    cp "$SCRIPT_DIR/env.sh" "$WORKSPACE_DIR/.manifest/env.sh"
}

# ---- Optional stages (DLRM / recommendation stack); not in default run ----
stage_torchrec() {
    reload_env
    log "Installing torchrec stack (optional)"
    $PIP install --no-deps torchrec
    $PIP install tensordict iopath torchmetrics==1.0.3 \
        "git+https://github.com/mlperf/logging.git" \
        --extra-index-url "$TORCH_INDEX"
}

DEFAULT_STAGES=(venv torch flash_attn te torchtune torchao pydeps \
                grouped_gemm causal_conv1d mamba primus aiter turbo boto manifest)

ALL_STAGES=("${DEFAULT_STAGES[@]}" torchrec)

run_stage() { local s="$1"; local fn="stage_$s"; declare -F "$fn" >/dev/null || die "unknown stage: $s"; "$fn"; }

main() {
    if [ "${1:-}" = "--list" ]; then
        echo "default: ${DEFAULT_STAGES[*]}"; echo "optional: torchrec"; exit 0
    fi
    local stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=("${DEFAULT_STAGES[@]}")
    log "Base dir: $PRIMUS_BASE | arch: $PYTORCH_ROCM_ARCH | stages: ${stages[*]}"
    df -h "$PRIMUS_BASE" 2>/dev/null | tail -1
    for s in "${stages[@]}"; do run_stage "$s"; done
    log "DONE. Activate later with:  source $SCRIPT_DIR/env.sh"
}

main "$@"

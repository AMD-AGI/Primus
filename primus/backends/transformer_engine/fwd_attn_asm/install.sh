#!/usr/bin/env bash
# =============================================================================
# install.sh -- build hand-tuned gfx950 attention kernels and wire them in.
#
# Steps performed (in order):
#   1. Clone https://github.com/mawad-amd/fwd-attn-asm  at $FWD_ATTN_ASM_REF
#      and assemble kernels/fwd_d64_opt128.s into <build>/fwd_d64_opt128.co.
#   2. Clone https://github.com/mawad-amd/bwd-attn-asm  at $BWD_ATTN_ASM_REF
#      and assemble kernels/bwd_d64_v3_causal_opt_16x32.s into
#      <build>/bwd_d64_v3_causal_opt_16x32.co.
#   3. (Optional, BWD_ATTN_ASM_ENABLE=1) overwrite the aiter v3 bwd .co
#      shipped inside TransformerEngine's QoLA submodule
#      (fmha_v3_bwd/bwd_hd64_bf16_causal_a16_rtz.co) with the hand-tuned
#      binary. The original aiter file is preserved as ``*.aiter_orig`` so
#      a later "cp .aiter_orig <name>" can revert without a rebuild.
#   4. Install the .pth shim + override Python module into ``site-packages``
#      so the runtime FMHA-fwd hook auto-activates at Python startup. The
#      hook itself stays gated by ``MLPERF_ENABLE_FWD_ATTN_ASM=1`` at
#      run time.
#
# This script is the migration replacement for the (formerly) inline
# Dockerfile blocks in the mlperf-training repo. It must run *before*
# ``pip install -e .`` is run on TransformerEngine, since the bwd swap
# changes a binary that QoLA bakes into ``te_libmha_bwd.so`` at TE build
# time.
#
# Inputs (env vars, all optional):
#   DEPS_DIR              workspace directory for clones and builds
#                         (default /workspace/deps)
#   TE_DIR                TransformerEngine checkout root
#                         (default ${DEPS_DIR}/TransformerEngine)
#   FWD_ATTN_ASM_REPO     fwd-attn-asm git URL
#                         (default https://github.com/mawad-amd/fwd-attn-asm.git)
#   FWD_ATTN_ASM_REF      fwd-attn-asm pinned SHA
#   BWD_ATTN_ASM_REPO     bwd-attn-asm git URL
#                         (default https://github.com/mawad-amd/bwd-attn-asm.git)
#   BWD_ATTN_ASM_REF      bwd-attn-asm pinned SHA
#   BWD_ATTN_ASM_ENABLE   "1" (default) embed hand-tuned bwd; "0" leaves
#                         the aiter binary untouched
#   SITE_PACKAGES         destination for the .pth shim + override module
#                         (default: auto-detect via python sysconfig)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPS_DIR="${DEPS_DIR:-/workspace/deps}"
TE_DIR="${TE_DIR:-${DEPS_DIR}/TransformerEngine}"
FWD_ATTN_ASM_REPO="${FWD_ATTN_ASM_REPO:-https://github.com/mawad-amd/fwd-attn-asm.git}"
FWD_ATTN_ASM_REF="${FWD_ATTN_ASM_REF:-53d3dadc3f3b0ac35ae536f2d1d7864a3e07ba22}"
BWD_ATTN_ASM_REPO="${BWD_ATTN_ASM_REPO:-https://github.com/mawad-amd/bwd-attn-asm.git}"
BWD_ATTN_ASM_REF="${BWD_ATTN_ASM_REF:-9b9fb6444f3fee388617f62432c3faea74079377}"
BWD_ATTN_ASM_ENABLE="${BWD_ATTN_ASM_ENABLE:-1}"

if [ -z "${SITE_PACKAGES:-}" ]; then
    SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
fi

mkdir -p "${DEPS_DIR}"

log() { echo "[fwd-attn-asm/install] $*"; }

# -----------------------------------------------------------------------------
# 1. fwd-attn-asm: clone + amdclang assemble
# -----------------------------------------------------------------------------
FWD_DIR="${DEPS_DIR}/fwd-attn-asm"
log "Cloning fwd-attn-asm @ ${FWD_ATTN_ASM_REF} -> ${FWD_DIR}"
if [ ! -d "${FWD_DIR}/.git" ]; then
    git clone "${FWD_ATTN_ASM_REPO}" "${FWD_DIR}"
fi
cd "${FWD_DIR}"
git fetch --depth=1 origin "${FWD_ATTN_ASM_REF}"
git checkout "${FWD_ATTN_ASM_REF}"
mkdir -p build
amdclang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
    -o build/fwd_d64_opt128.co kernels/fwd_d64_opt128.s
log "Built ${FWD_DIR}/build/fwd_d64_opt128.co"

# -----------------------------------------------------------------------------
# 2. bwd-attn-asm: clone + amdclang assemble
# -----------------------------------------------------------------------------
BWD_DIR="${DEPS_DIR}/bwd-attn-asm"
log "Cloning bwd-attn-asm @ ${BWD_ATTN_ASM_REF} -> ${BWD_DIR}"
if [ ! -d "${BWD_DIR}/.git" ]; then
    git clone "${BWD_ATTN_ASM_REPO}" "${BWD_DIR}"
fi
cd "${BWD_DIR}"
git fetch --depth=1 origin "${BWD_ATTN_ASM_REF}"
git checkout "${BWD_ATTN_ASM_REF}"
mkdir -p build
amdclang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
    -o build/bwd_d64_v3_causal_opt_16x32.co kernels/bwd_d64_v3_causal_opt_16x32.s
log "Built ${BWD_DIR}/build/bwd_d64_v3_causal_opt_16x32.co"

# -----------------------------------------------------------------------------
# 3. Optional: overwrite aiter's bwd .co inside TE/QoLA before TE pip install
# -----------------------------------------------------------------------------
AITER_CO="${TE_DIR}/3rdparty/QoLA/3rdparty/aiter/hsa/gfx950/fmha_v3_bwd/bwd_hd64_bf16_causal_a16_rtz.co"
HAND_TUNED_BWD_CO="${BWD_DIR}/build/bwd_d64_v3_causal_opt_16x32.co"

if [ -f "${AITER_CO}" ]; then
    if [ ! -f "${AITER_CO}.aiter_orig" ]; then
        cp -n "${AITER_CO}" "${AITER_CO}.aiter_orig"
    fi
    if [ "${BWD_ATTN_ASM_ENABLE}" = "1" ]; then
        log "Embedding ${HAND_TUNED_BWD_CO} into TE/QoLA aiter (${AITER_CO})"
        cp "${HAND_TUNED_BWD_CO}" "${AITER_CO}"
    else
        log "BWD_ATTN_ASM_ENABLE=0; restoring stock aiter bwd .co"
        cp "${AITER_CO}.aiter_orig" "${AITER_CO}"
    fi
else
    log "WARNING: TE/QoLA aiter bwd .co not found at ${AITER_CO}; skipping bwd swap."
    log "         (TE may not be cloned yet; rerun this script after TE is in place"
    log "          and BEFORE 'pip install -e .' on TransformerEngine.)"
fi

# -----------------------------------------------------------------------------
# 4. Install .pth shim + override module into site-packages
# -----------------------------------------------------------------------------
log "Installing override module + .pth shim into ${SITE_PACKAGES}"
install -m 0644 "${SCRIPT_DIR}/scripts/aiter_hd64_asm_override.py" \
    "${SITE_PACKAGES}/aiter_hd64_asm_override.py"
install -m 0644 "${SCRIPT_DIR}/scripts/aiter_hd64_asm_override.pth" \
    "${SITE_PACKAGES}/aiter_hd64_asm_override.pth"

# Persist the kernel path so the runtime hook can find the .co without
# the consumer having to set FMHA_HD64_ASM_CO. Consumers may override.
DEFAULT_CO="${FWD_DIR}/build/fwd_d64_opt128.co"
log "Default FMHA_HD64_ASM_CO will be ${DEFAULT_CO} (override with env var)."

log "done."

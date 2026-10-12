#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Install the MaxDiffusion (JAX) Python dependencies for primus-cli launches.
#
# Same shape as the maxtext / megatron / torchtitan siblings: this hook only
# installs packages. The other two concerns are split the same way they are for
# every other backend:
#   - backend path + launcher env (RUN_MODE, JAX coordinator, NVTE_FRAMEWORK)
#     are emitted by this directory's prepare.py
#   - runtime behavior (Shardy partitioner, TensorFlow preload) lives in
#     primus/backends/maxdiffusion/patches/
# so nothing here has to mutate the vendored third_party/maxdiffusion checkout.
#
# requirements-maxdiffusion.txt is deliberately separate from requirements-jax.txt
# (which MaxText installs): it pins transformers 4.x for MaxDiffusion's Flax code,
# and that pin must not be forced onto MaxText runs.
#
# torch is installed here from a ROCm wheel source, which needs
# --index-url/--find-links and therefore cannot live in the PyPI resolve of
# requirements-maxdiffusion.txt. JAX-only images ship no torch, but MaxDiffusion
# imports it at module load time -- max_utils -> attention_flax ->
# modeling_flax_utils -> modeling_flax_pytorch_utils -> `import torch` -- for the
# PyTorch->Flax weight conversion helpers, so the run dies during config parsing
# without it.
#
# The wheels bundle ROCm runtime libraries, so the source must match the ROCm in
# the image. Two are supported and they are NOT interchangeable:
#   ROCm >= 10 (TheRock, e.g. the jax_the_rock CI images): stable.repo.amd.com,
#     "+rocm10.0.0" wheels with a device-<gfx> extra that reuses the image's
#     already-installed rocm-sdk-device-* packages.
#   ROCm 7.x (classic, e.g. rocm/jax-training:maxtext-*): repo.radeon.com
#     find-links, the same install examples/maxdiffusion/setup_maxdiffusion_env.sh
#     performs.
# Overrides (all optional):
#   MAXDIFFUSION_ROCM_VERSION   wheel local-version, e.g. 10.0.0
#   MAXDIFFUSION_TORCH_GFX      gfx942 / gfx950 / all / gfx942;gfx950
#   MAXDIFFUSION_TORCH_VERSION / MAXDIFFUSION_TORCHVISION_VERSION
#   MAXDIFFUSION_TORCH_INDEX    pip --index-url for ROCm >= 10
#   MAXDIFFUSION_TORCH_LINKS    pip --find-links for classic ROCm 7.x
#
# PRIMUS_SKIP_PIP=1 skips this step, for images that already ship the stack.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PRIMUS_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
PRIMUS_ROOT="${PRIMUS_PATH:-${DEFAULT_PRIMUS_ROOT}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --primus_path)
      PRIMUS_ROOT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Load shared logging so output honors PRIMUS_LOG_LEVEL (DEBUG/INFO/WARN/ERROR).
# When invoked through primus-cli the LOG_* functions are already exported, but
# sourcing here keeps the hook usable standalone and under `set -u`.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../../hook_common.sh"

if [[ "${PRIMUS_SKIP_PIP:-0}" == "1" ]]; then
  LOG_INFO "PRIMUS_SKIP_PIP=1: skipping MaxDiffusion dependency install (deps from image)"
  exit 0
fi

DATA_PATH="${DATA_PATH:-${PRIMUS_ROOT}/data}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${DATA_PATH}/pip_cache}"

# Match pip verbosity to the active log level so WARN/ERROR runs stay quiet
# (suppresses the "Requirement already satisfied" wall) while DEBUG/INFO keep it.
PIP_FLAGS=()
case "${PRIMUS_LOG_LEVEL:-INFO}" in
  WARN|ERROR) PIP_FLAGS+=(-q -q) ;;
esac

LOG_INFO "Using pip cache: ${PIP_CACHE_DIR}"
mkdir -p "${PIP_CACHE_DIR}"

REQ_FILE="${PRIMUS_ROOT}/requirements-maxdiffusion.txt"
if [[ ! -f "${REQ_FILE}" ]]; then
  LOG_ERROR "Missing required MaxDiffusion requirements file: ${REQ_FILE}"
  exit 1
fi

LOG_INFO "Installing MaxDiffusion dependencies..."
pip install "${PIP_FLAGS[@]}" --cache-dir="${PIP_CACHE_DIR}" -r "${REQ_FILE}"
LOG_SUCCESS "MaxDiffusion dependencies installed"

if python3 -c "import torch" 2>/dev/null; then
  LOG_INFO "torch present: $(python3 -c 'import torch; print(torch.__version__)'), skipping torch install"
else
  # Keep X.Y.Z so the wheel local-version (+rocm10.0.0) matches the AMD index.
  normalize_rocm_version() {
    local v="$1"
    v="${v%%[-+]*}"
    v="${v%%.dev*}"
    v="${v%%.post*}"
    if [[ "${v}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "${v}"
    elif [[ "${v}" =~ ^[0-9]+\.[0-9]+$ ]]; then
      echo "${v}.0"
    elif [[ "${v}" =~ ^[0-9]+$ ]]; then
      echo "${v}.0.0"
    fi
  }

  detect_rocm_version() {
    local version info
    if [[ -n "${THE_ROCK_VERSION:-}" ]]; then
      normalize_rocm_version "${THE_ROCK_VERSION}"
      return
    fi
    if version=$(python3 -c 'import rocm_sdk; print(rocm_sdk.__version__)' 2>/dev/null) \
       && [[ -n "${version}" ]]; then
      normalize_rocm_version "${version}"
      return
    fi
    for info in "${ROCM_PATH:-/opt/rocm}/.info/version" /opt/rocm/.info/version; do
      if [[ -f "${info}" ]]; then
        normalize_rocm_version "$(cut -d- -f1 <"${info}")"
        return
      fi
    done
  }

  # Same KFD decoder as tools/installation/env.sh (90402 -> gfx942).
  gfxver_to_arch() {
    local v="$1"
    [[ -n "${v}" && "${v}" != "0" ]] || return 0
    printf 'gfx%d%x%x\n' "$(( v / 10000 ))" "$(( (v / 100) % 100 ))" "$(( v % 100 ))"
  }

  # Unique ";"-separated gfx names. Matches only rocminfo "Name: gfxNNN" (not
  # generic ISA triples like amdgcn-amd-amdhsa--gfx9-4-generic). Does not use
  # PYTORCH_ROCM_ARCH: CI images set that to every arch they *build* for
  # (gfx942;gfx950), which is the wrong extra on a single-arch node.
  detect_gpu_arches() {
    local arches="" line f v a
    if command -v rocminfo >/dev/null 2>&1; then
      while IFS= read -r line; do
        [[ "${line}" =~ ^gfx[0-9a-f]+$ ]] || continue
        case ";${arches};" in *";${line};"*) ;; *) arches="${arches:+${arches};}${line}" ;; esac
      done < <(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx/{print $2}')
    fi
    if [[ -z "${arches}" && -d /sys/class/kfd/kfd/topology/nodes ]]; then
      for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [[ -f "${f}" ]] || continue
        v="$(awk '/^gfx_target_version/{print $2}' "${f}" 2>/dev/null || true)"
        a="$(gfxver_to_arch "${v}")"
        [[ -n "${a}" ]] || continue
        case ";${arches};" in *";${a};"*) ;; *) arches="${arches:+${arches};}${a}" ;; esac
      done
    fi
    echo "${arches}"
  }

  # Pip extra(s): device-gfx942 or device-gfx942,device-gfx950.
  # `all` is opt-in only: device-all pulls every gfx wheel plus extra
  # rocm-sdk-device-* packages and can fight a TheRock image SDK.
  torch_device_extra() {
    local spec="${1:-}" part extras=()
    [[ -n "${spec}" ]] || return 0
    if [[ "${spec}" == "all" ]]; then
      echo "device-all"
      return
    fi
    spec="${spec//;/,}"
    IFS=',' read -ra parts <<< "${spec}"
    for part in "${parts[@]}"; do
      part="${part// /}"
      [[ "${part}" == device-* ]] && part="${part#device-}"
      if [[ "${part}" == "all" ]]; then
        echo "device-all"
        return
      fi
      [[ "${part}" =~ ^gfx[0-9a-f]+$ ]] || continue
      extras+=("device-${part}")
    done
    if (( ${#extras[@]} == 0 )); then
      return 0
    fi
    local IFS=,
    echo "${extras[*]}"
  }

  is_therock_rocm() {
    [[ -n "${THE_ROCK_VERSION:-}" ]] && return 0
    python3 -c "import rocm_sdk" 2>/dev/null
  }

  ROCM_VERSION="${MAXDIFFUSION_ROCM_VERSION:-$(detect_rocm_version)}"
  ROCM_VERSION="$(normalize_rocm_version "${ROCM_VERSION}")"
  ROCM_MAJOR="${ROCM_VERSION%%.*}"

  if is_therock_rocm || { [[ "${ROCM_MAJOR}" =~ ^[0-9]+$ ]] && (( ROCM_MAJOR >= 10 )); }; then
    # TheRock / ROCm >= 10: AMD index + per-arch extra. Pin +rocm<image> so pip
    # reuses the image's rocm-sdk-device-* packages instead of pulling ROCm 7.
    TORCH_VERSION="${MAXDIFFUSION_TORCH_VERSION:-2.13.0}"
    TORCHVISION_VERSION="${MAXDIFFUSION_TORCHVISION_VERSION:-0.28.0}"
    TORCH_INDEX="${MAXDIFFUSION_TORCH_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"
    ROCM_VERSION="${ROCM_VERSION:-10.0.0}"
    # Live GPU first. Do not take the first entry of PYTORCH_ROCM_ARCH (CI
    # images set gfx942;gfx950 as the *build* list). If nothing is visible,
    # install extras for every arch in that list, else Instinct gfx942;gfx950.
    ARCH_SPEC="${MAXDIFFUSION_TORCH_GFX:-$(detect_gpu_arches)}"
    DEVICE_EXTRA="$(torch_device_extra "${ARCH_SPEC}")"
    if [[ -z "${DEVICE_EXTRA}" ]]; then
      ARCH_SPEC="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}"
      DEVICE_EXTRA="$(torch_device_extra "${ARCH_SPEC}")"
      LOG_INFO "GPU arch not detected; installing torch extras for ${ARCH_SPEC}"
    fi
    if [[ -z "${DEVICE_EXTRA}" ]]; then
      LOG_ERROR "Could not resolve a torch device extra (tried '${ARCH_SPEC}'). Set MAXDIFFUSION_TORCH_GFX."
      exit 1
    fi

    LOG_INFO "ROCm ${ROCM_VERSION} (${DEVICE_EXTRA}): installing torch==${TORCH_VERSION}+rocm${ROCM_VERSION} from ${TORCH_INDEX}"
    pip install "${PIP_FLAGS[@]}" --cache-dir="${PIP_CACHE_DIR}" \
      --index-url "${TORCH_INDEX}" \
      "torch[${DEVICE_EXTRA}]==${TORCH_VERSION}+rocm${ROCM_VERSION}" \
      "torchvision[${DEVICE_EXTRA}]==${TORCHVISION_VERSION}+rocm${ROCM_VERSION}"
  else
    # Classic ROCm wheels; keep in sync with examples/maxdiffusion/setup_maxdiffusion_env.sh.
    TORCH_VERSION="${MAXDIFFUSION_TORCH_VERSION:-2.8.0}"
    TORCHVISION_VERSION="${MAXDIFFUSION_TORCHVISION_VERSION:-0.23.0}"
    TORCH_LINKS="${MAXDIFFUSION_TORCH_LINKS:-${MAXDIFFUSION_TORCH_INDEX:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/}}"

    LOG_INFO "ROCm ${ROCM_VERSION:-unknown}: installing torch==${TORCH_VERSION} from ${TORCH_LINKS}"
    pip install "${PIP_FLAGS[@]}" --cache-dir="${PIP_CACHE_DIR}" \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --find-links "${TORCH_LINKS}" --no-index
  fi

  LOG_SUCCESS "torch/torchvision installed"
fi

# Apply the two site-package compatibility fixes required by the JAX 0.11
# MaxDiffusion stack. Some AMD MaxText images already contain these changes,
# while generic/TheRock JAX images do not. Keep this idempotent so the same
# hook works in either environment.
LOG_INFO "Checking MaxDiffusion site-package compatibility..."
python3 - <<'PY'
from __future__ import annotations

import sysconfig
from pathlib import Path


site_packages = Path(sysconfig.get_paths()["purelib"])


def patch_flax_t5() -> str:
    """Use the JAX 0.11 names for jnp.clip keyword arguments."""
    path = site_packages / "transformers/models/t5/modeling_flax_t5.py"
    if not path.is_file():
        raise RuntimeError(f"required Flax T5 module not found: {path}")

    source = path.read_text()
    patched = source.replace("a_min=", "min=").replace("a_max=", "max=")
    if patched == source:
        if "a_min=" in source or "a_max=" in source:
            raise RuntimeError(f"failed to patch deprecated jnp.clip keywords in {path}")
        return f"already compatible: {path}"

    path.write_text(patched)
    verify = path.read_text()
    if "a_min=" in verify or "a_max=" in verify:
        raise RuntimeError(f"deprecated jnp.clip keywords remain in {path}")
    return f"patched JAX clip keywords: {path}"


def patch_te_empty_axis() -> str:
    """Treat both None and an empty TE mesh axis as an unsharded axis."""
    path = site_packages / "transformer_engine/jax/sharding.py"
    if not path.is_file():
        raise RuntimeError(f"required Transformer Engine sharding module not found: {path}")

    source = path.read_text()
    function_start = source.find("def get_mesh_axis_size(")
    if function_start < 0:
        raise RuntimeError(f"get_mesh_axis_size() not found in {path}")
    function_end = source.find("\ndef ", function_start + 1)
    if function_end < 0:
        function_end = len(source)
    function = source[function_start:function_end]

    # Images such as rocm/jax-training:maxtext-v26.6 already have this guard.
    if "if not axis:" in function or "if axis is None or axis == \"\":" in function:
        return f"already compatible: {path}"

    old = "    if axis is None:\n        return 1"
    new = "    if not axis:\n        return 1"
    if old not in function:
        raise RuntimeError(
            f"unsupported get_mesh_axis_size() layout in {path}; "
            "refusing to patch an unknown Transformer Engine version"
        )

    patched_function = function.replace(old, new, 1)
    patched = source[:function_start] + patched_function + source[function_end:]
    path.write_text(patched)

    verify = path.read_text()
    verify_start = verify.find("def get_mesh_axis_size(")
    verify_end = verify.find("\ndef ", verify_start + 1)
    if verify_end < 0:
        verify_end = len(verify)
    if "if not axis:" not in verify[verify_start:verify_end]:
        raise RuntimeError(f"empty-axis guard was not applied to {path}")
    return f"patched empty mesh axis handling: {path}"


for result in (patch_flax_t5(), patch_te_empty_axis()):
    print(f"[MaxDiffusion compatibility] {result}")
PY
LOG_SUCCESS "MaxDiffusion site-package compatibility checks passed"

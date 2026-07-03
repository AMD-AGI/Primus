#!/usr/bin/env bash
# =============================================================================
# build_rocshmem_backend.sh
#
# Build the ODC rocSHMEM P2P backend **entirely from source** — no committed
# binaries. Run this once inside the base image (tasimage/primus-odc:v26.2 or
# pr-722, ROCm 7.2.0 / gfx942) before using `ODC_P2P_BACKEND=rocshmem`.
#
# It performs three idempotent stages:
#   1. Clone rocSHMEM @ pinned commit and cmake-build the static `librocshmem.a`
#      for each requested variant (single / ro / gda).
#   2. hipcc-compile the three ctypes host bindings (librs_host5.so,
#      librs_host_ro.so, librs_host_gda.so) against those static libs.
#   3. `pip install --no-build-isolation -e .` to (re)build tensor_ipc.so.
#
# Everything is CPU-side cross-compilation for gfx942 — NO GPU is required.
#
# Usage:
#   bash odc_rocm_dev/build_rocshmem_backend.sh [variants...] [--force]
#     variants : any of {single ro gda tensor_ipc all}  (default: all)
#     --force  : rebuild even if outputs already exist
#
# Examples:
#   bash odc_rocm_dev/build_rocshmem_backend.sh              # build everything
#   bash odc_rocm_dev/build_rocshmem_backend.sh single       # single-node .so only
#   bash odc_rocm_dev/build_rocshmem_backend.sh single ro --force
#
# All output is tee'd to rocshmem_runtime/build.log so a detached run can be
# picked up later:
#   nohup bash odc_rocm_dev/build_rocshmem_backend.sh > /dev/null 2>&1 &
# =============================================================================
set -euo pipefail

# --- locate ourselves --------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../odc_rocm_dev
RT="${SCRIPT_DIR}/rocshmem_runtime"
SRC="${RT}/rocshmem_src"
LOG="${RT}/build.log"

# --- config ------------------------------------------------------------------
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_TARGETS="${GPU_TARGETS:-gfx942}"
OFFLOAD_ARCH="${OFFLOAD_ARCH:-${GPU_TARGETS}:xnack-}"
ROCSHMEM_URL="${ROCSHMEM_URL:-https://github.com/ROCm/rocSHMEM}"
ROCSHMEM_COMMIT="${ROCSHMEM_COMMIT:-17ff985c026f9f97f85068647e863ab541dd5645}"
JOBS="${JOBS:-$(nproc)}"
OMPI_INC="${OMPI_INC:-/usr/lib/x86_64-linux-gnu/openmpi/include}"
OMPI_LIB="${OMPI_LIB:-/usr/lib/x86_64-linux-gnu/openmpi/lib}"

HIPCC="${ROCM_PATH}/bin/hipcc"

# --- logging -----------------------------------------------------------------
mkdir -p "${RT}"
# tee everything (stdout+stderr) to build.log while still showing on console.
exec > >(tee -a "${LOG}") 2>&1
echo "============================================================"
echo "build_rocshmem_backend.sh @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  ROCM_PATH=${ROCM_PATH}  GPU_TARGETS=${GPU_TARGETS}  JOBS=${JOBS}"
echo "  rocSHMEM=${ROCSHMEM_URL}@${ROCSHMEM_COMMIT}"
echo "============================================================"

# --- parse args --------------------------------------------------------------
FORCE=0
VARIANTS=()
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    all|single|ro|gda|tensor_ipc) VARIANTS+=("$a") ;;
    *) echo "!! unknown arg: $a"; exit 2 ;;
  esac
done
if [ ${#VARIANTS[@]} -eq 0 ]; then VARIANTS=(all); fi
want() {
  local t="$1"
  for v in "${VARIANTS[@]}"; do
    [ "$v" = "all" ] && return 0
    [ "$v" = "$t" ] && return 0
  done
  return 1
}

# --- preflight ---------------------------------------------------------------
[ -x "${HIPCC}" ] || { echo "!! hipcc not found at ${HIPCC}"; exit 1; }
echo "-- hipcc: $(${HIPCC} --version | head -1)"
command -v cmake >/dev/null || { echo "!! cmake not found"; exit 1; }
echo "-- cmake: $(cmake --version | head -1)"

# =============================================================================
# Stage 1: clone rocSHMEM + build static librocshmem.a variants
# =============================================================================
clone_rocshmem() {
  if [ -d "${SRC}/.git" ]; then
    echo "== rocSHMEM source present at ${SRC}"
  else
    echo "== cloning rocSHMEM into ${SRC}"
    git clone "${ROCSHMEM_URL}" "${SRC}"
  fi
  ( cd "${SRC}" || exit 1
    git fetch --all --quiet 2>/dev/null || true
    git checkout --quiet "${ROCSHMEM_COMMIT}" )
  echo "== rocSHMEM at commit $(cd "${SRC}" && git rev-parse HEAD)"
}

# build_librocshmem <variant> <extra cmake flags...>
build_librocshmem() {
  local variant="$1"; shift
  local out="${RT}/rocshmem_${variant}"
  local bld="${SRC}/build_${variant}"
  if [ -f "${out}/lib/librocshmem.a" ] && [ "${FORCE}" -eq 0 ]; then
    echo "== librocshmem.a (${variant}) already built -> ${out}/lib (skip; --force to rebuild)"
    return 0
  fi
  echo "== configuring librocshmem.a (${variant}): $*"
  rm -rf "${bld}"
  cmake -S "${SRC}" -B "${bld}" \
    -DGPU_TARGETS="${GPU_TARGETS}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_FUNCTIONAL_TESTS=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_TOOLS=OFF \
    "$@"
  echo "== building librocshmem.a (${variant}) -j${JOBS}"
  cmake --build "${bld}" -j"${JOBS}"
  echo "== installing librocshmem.a (${variant}) -> ${out}"
  rm -rf "${out}"
  cmake --install "${bld}" --prefix "${out}"
  [ -f "${out}/lib/librocshmem.a" ] || { echo "!! ${variant}: librocshmem.a missing after install"; exit 1; }
  echo "== OK: ${out}/lib/librocshmem.a ($(du -h "${out}/lib/librocshmem.a" | cut -f1))"
}

NEED_SRC=0
want single && NEED_SRC=1
want ro     && NEED_SRC=1
want gda    && NEED_SRC=1
if [ "${NEED_SRC}" -eq 1 ]; then
  clone_rocshmem
fi

# single-node IPC library. NOTE: rocSHMEM's CMake defaults USE_RO=ON, which would
# pull the MPI Reverse-Offload transport (MPI_Win_create at init -> fatal under a
# torchrun launch with no mpirun). Must force USE_RO=OFF for the single-node lib.
want single && build_librocshmem single -DUSE_IPC=ON -DUSE_SINGLE_NODE=ON -DUSE_RO=OFF
# multi-node Reverse-Offload library (RO is the point here)
want ro     && build_librocshmem ro     -DUSE_IPC=ON -DUSE_RO=ON
# GPU-direct (GDA) library. Same USE_RO=OFF caveat as single (GDA is its own conduit).
# GDA_MLX5=ON is required: rocSHMEM defaults GDA_MLX5/BNXT/IONIC to OFF, which compiles
# out the MLX5 DirectVerbs provider and makes multi-node GDA fail at runtime with
# "open_dv_libs: no DV library could dlopen for ... MLX5 GDA support".
want gda    && build_librocshmem gda    -DUSE_GDA=ON -DUSE_RO=OFF -DGDA_MLX5=ON

# =============================================================================
# Stage 2: hipcc-compile the ctypes host bindings (.so)
#
# librocshmem.a is built with fgpu-rdc device code, so the final .so link MUST
# be a HIP device link (-fgpu-rdc --hip-link) or the __hip_gpubin_handle_*
# symbols stay undefined. We therefore compile each binding in two phases:
#   (1) hipcc -fgpu-rdc -c  <cpp> -o <obj>
#   (2) hipcc -fgpu-rdc --hip-link -shared <obj> <librocshmem.a> <libs...>
# =============================================================================

# compile_binding <variant> <cpp> <out.so> <extra link args...>
compile_binding() {
  local variant="$1" cpp="$2" out="$3"; shift 3
  local inc="${RT}/rocshmem_${variant}/include"
  local lib="${RT}/rocshmem_${variant}/lib/librocshmem.a"
  local obj="${cpp%.cpp}.o"
  echo "== compiling ${out} (fgpu-rdc, 2 phases)"
  "${HIPCC}" -fgpu-rdc -x hip --offload-arch="${OFFLOAD_ARCH}" -fPIC -O2 -std=c++17 \
    -c "${cpp}" -o "${obj}" \
    -I "${inc}" -I "${OMPI_INC}"
  "${HIPCC}" -fgpu-rdc --hip-link -shared "${obj}" \
    -o "${out}" --offload-arch="${OFFLOAD_ARCH}" \
    -x none "${lib}" \
    -L"${ROCM_PATH}/lib" -lamdhip64 -lhsa-runtime64 \
    "$@"
  rm -f "${obj}"
  [ -f "${out}" ] || { echo "!! ${out} not produced"; exit 1; }
  echo "== OK: $(ls -la "${out}")"
}

# single: librs_host5.so (IPC/XGMI single-node). rocshmem.hpp pulls <mpi.h> and
# librocshmem.a references MPI symbols unconditionally, so -lmpi is required
# even though the single-node runtime never drives the MPI transport.
if want single; then
  compile_binding single \
    "${RT}/host_bindings/rs_host.cpp" \
    "${RT}/host_bindings/librs_host5.so" \
    -L"${OMPI_LIB}" -lmpi
fi

# ro: librs_host_ro.so (Reverse-Offload; host-driven MPI/UCX transport)
if want ro; then
  compile_binding ro \
    "${RT}/ro_backend/rs_host_ro.cpp" \
    "${RT}/ro_backend/librs_host_ro.so" \
    -L"${OMPI_LIB}" -lmpi
fi

# gda: librs_host_gda.so (GPU-direct; host+device kernels, RDMA verbs)
if want gda; then
  compile_binding gda \
    "${RT}/gda_backend/rs_host_gda.cpp" \
    "${RT}/gda_backend/librs_host_gda.so" \
    -L"${OMPI_LIB}" -lmpi -libverbs -lmlx5 -lnuma
fi

# =============================================================================
# Stage 3: build tensor_ipc.so (odc C++/HIP extension)
# =============================================================================
if want tensor_ipc; then
  echo "== pip install -e odc (builds tensor_ipc.so)"
  ( cd "${SCRIPT_DIR}" && pip install --no-build-isolation -e . )
  python - <<'PY'
import odc  # noqa
print("== OK: import odc succeeded (tensor_ipc loaded)")
PY
fi

echo "============================================================"
echo "DONE @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Built artifacts:"
ls -la "${RT}"/host_bindings/*.so "${RT}"/ro_backend/*.so "${RT}"/gda_backend/*.so 2>/dev/null || true
echo "============================================================"

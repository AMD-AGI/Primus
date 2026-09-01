#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="${PRIMUS_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

BASE_IMAGE="${BASE_IMAGE:-unifiedtrainingdockers.azurecr.io/utd/nightly:primus_the_rock_rocm7.15_20260728}"
MORI_REPO="${MORI_REPO:-https://github.com/ROCm/mori.git}"
MORI_REF="${MORI_REF:-12d1bc32d0c93dcd5062e74f4e0f772e36e1aac4}"
MAX_JOBS="${MAX_JOBS:-32}"
SMOKE_NUMEL="${SMOKE_NUMEL:-67108864}"
KEEP_CONTAINER="${KEEP_CONTAINER:-0}"
LOG_DIR="${LOG_DIR:-/tmp/primus-mori-preflight-$(hostname -s)-$(date +%Y%m%d-%H%M%S)}"
CONTAINER_NAME="${CONTAINER_NAME:-primus_mori_preflight_${USER}_$(hostname -s)}"

mkdir -p "${LOG_DIR}"

CURRENT_PHASE="initialization"
PHASE_SUMMARY=()

section() {
    echo
    echo "================================================================================"
    echo "$*"
    echo "================================================================================"
}

format_seconds() {
    local seconds="$1"
    printf "%dm%02ds" "$((seconds / 60))" "$((seconds % 60))"
}

run_phase() {
    local name="$1"
    shift
    local start end elapsed rc
    CURRENT_PHASE="${name}"
    start="$(date +%s)"
    section "Phase: ${name}"

    set +e
    "$@" 2>&1 | tee "${LOG_DIR}/${name// /_}.log"
    rc="${PIPESTATUS[0]}"
    set -e

    end="$(date +%s)"
    elapsed="$((end - start))"
    PHASE_SUMMARY+=("${name}|${elapsed}|${rc}")
    echo "[preflight] ${name}: $(format_seconds "${elapsed}") (rc=${rc})"
    if [[ "${rc}" -ne 0 ]]; then
        return "${rc}"
    fi
}

cleanup() {
    local rc=$?
    if [[ "${rc}" -ne 0 ]]; then
        echo
        echo "[preflight] FAILED during phase: ${CURRENT_PHASE}" >&2
        echo "[preflight] Logs: ${LOG_DIR}" >&2
        docker inspect "${CONTAINER_NAME}" >"${LOG_DIR}/container-inspect.json" 2>/dev/null || true
    fi

    if [[ "${KEEP_CONTAINER}" != "1" ]]; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    else
        echo "[preflight] Keeping container: ${CONTAINER_NAME}"
    fi
}
trap cleanup EXIT INT TERM

command -v docker >/dev/null || {
    echo "[preflight] docker is required" >&2
    exit 1
}

section "Host identity"
echo "hostname        : $(hostname -f)"
echo "kernel          : $(uname -r)"
echo "user            : $(id)"
echo "base image      : ${BASE_IMAGE}"
echo "MORI revision   : ${MORI_REF}"
echo "Primus root     : ${PRIMUS_ROOT}"
echo "log directory   : ${LOG_DIR}"

section "GPU information"
if command -v rocm-smi >/dev/null; then
    rocm-smi --showproductname --showuse --showmemuse --csv 2>&1 || true
else
    echo "rocm-smi not found"
fi
if command -v rocminfo >/dev/null; then
    rocminfo 2>/dev/null |
        awk '/Name: +gfx/{print "GPU architecture : " $2}' |
        sort -u || true
fi

section "IP interfaces"
ip -o -4 addr show scope global 2>&1 || true

section "RDMA devices and links"
if command -v ibv_devices >/dev/null; then
    ibv_devices 2>&1 || true
fi
if command -v ibdev2netdev >/dev/null; then
    ibdev2netdev 2>&1 || true
fi
if command -v rdma >/dev/null; then
    rdma link show 2>&1 || true
fi

section "Valid GIDs"
for dev_path in /sys/class/infiniband/*; do
    [[ -d "${dev_path}" ]] || continue
    dev="${dev_path##*/}"
    for gid_path in "${dev_path}"/ports/1/gids/*; do
        [[ -f "${gid_path}" ]] || continue
        idx="${gid_path##*/}"
        gid="$(<"${gid_path}")"
        [[ "${gid}" != "0000:0000:0000:0000:0000:0000:0000:0000" ]] || continue
        type="unknown"
        ndev="unknown"
        if [[ -r "${dev_path}/ports/1/gid_attrs/types/${idx}" ]]; then
            type="$(<"${dev_path}/ports/1/gid_attrs/types/${idx}")"
        fi
        if [[ -r "${dev_path}/ports/1/gid_attrs/ndevs/${idx}" ]]; then
            ndev="$(<"${dev_path}/ports/1/gid_attrs/ndevs/${idx}")"
        fi
        printf "%-12s index=%-3s type=%-8s netdev=%-12s gid=%s\n" \
            "${dev}" "${idx}" "${type}" "${ndev}" "${gid}"
    done
done

section "NIC driver and firmware"
mapfile -t NETDEVS < <(
    for dev_path in /sys/class/infiniband/*; do
        [[ -d "${dev_path}" ]] || continue
        for ndev_path in "${dev_path}"/device/net/*; do
            [[ -e "${ndev_path}" ]] && basename "${ndev_path}"
        done
    done | sort -u
)
STACK_DATA=""
for netdev in "${NETDEVS[@]:-}"; do
    driver_info="$(ethtool -i "${netdev}" 2>&1 || true)"
    echo "--- ${netdev} ---"
    echo "${driver_info}"
    driver_version="$(awk -F': ' '$1=="version"{print $2}' <<<"${driver_info}")"
    firmware_version="$(awk -F': ' '$1=="firmware-version"{print $2}' <<<"${driver_info}")"
    STACK_DATA+="${netdev}|${driver_version}|${firmware_version};"
done
STACK_SHA="$(printf "%s" "${STACK_DATA}" | sha256sum | awk '{print $1}')"

detect_nic() {
    if compgen -G "/sys/class/infiniband/ionic*" >/dev/null; then
        echo ionic
    elif compgen -G "/sys/class/infiniband/bnxt_re*" >/dev/null; then
        echo bnxt
    elif compgen -G "/sys/class/infiniband/mlx5*" >/dev/null; then
        echo mlx5
    else
        echo unknown
    fi
}

DETECTED_NIC="$(detect_nic)"

section "Vendor library checks (detected NIC: ${DETECTED_NIC})"

find_library() {
    local name="$1"
    local path
    path="$(ldconfig -p 2>/dev/null | awk -v n="${name}" '$1==n{print $NF; exit}')"
    if [[ -z "${path}" ]]; then
        for candidate in \
            "/usr/local/lib/${name}" \
            "/usr/lib/x86_64-linux-gnu/${name}" \
            "/lib/x86_64-linux-gnu/${name}"; do
            if [[ -e "${candidate}" ]]; then
                path="${candidate}"
                break
            fi
        done
    fi
    echo "${path}"
}

has_symbol() {
    local path="$1"
    local symbol="$2"
    nm -D "${path}" 2>/dev/null |
        awk -v s="${symbol}" '$3 == s || index($3, s "@") == 1 {found=1} END {exit !found}'
}

check_symbols() {
    local path="$1"
    shift
    local symbol
    for symbol in "$@"; do
        if has_symbol "${path}" "${symbol}"; then
            echo "  ${symbol}: present"
        else
            echo "  ${symbol}: MISSING"
        fi
    done
}

ionic_ccqe_fw_supported() {
    local value="$1"
    if [[ ! "${value}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)-[[:alpha:]]-?([0-9]+)$ ]]; then
        return 1
    fi
    local major="${BASH_REMATCH[1]}"
    local minor="${BASH_REMATCH[2]}"
    local patch="${BASH_REMATCH[3]}"
    local build="${BASH_REMATCH[4]}"
    ((major > 1 ||
        (major == 1 && minor > 117) ||
        (major == 1 && minor == 117 && patch > 5) ||
        (major == 1 && minor == 117 && patch == 5 && build >= 58)))
}

VENDOR_LIB=""
VENDOR_NAMES=()
VENDOR_SHA="missing"
CCQE_CANDIDATE="n/a"
case "${DETECTED_NIC}" in
    ionic)
        VENDOR_LIB="$(find_library libionic.so)"
        VENDOR_NAMES=(libionic.so)
        ;;
    bnxt)
        VENDOR_LIB="$(find_library libbnxt_re.so)"
        VENDOR_NAMES=(libbnxt_re.so libbnxt_re-rdmav34.so)
        ;;
    mlx5)
        VENDOR_LIB="$(find_library libmlx5.so)"
        VENDOR_NAMES=(libmlx5.so)
        ;;
esac

if [[ -n "${VENDOR_LIB}" ]]; then
    VENDOR_LIB="$(readlink -f "${VENDOR_LIB}")"
    VENDOR_SHA="$(sha256sum "${VENDOR_LIB}" | awk '{print $1}')"
    echo "library         : ${VENDOR_LIB}"
    echo "sha256          : ${VENDOR_SHA}"
    case "${DETECTED_NIC}" in
        ionic)
            check_symbols "${VENDOR_LIB}" \
                ionic_dv_get_ctx \
                ionic_dv_get_cq \
                ionic_dv_get_qp \
                ionic_dv_create_cq_ex
            ;;
        bnxt)
            check_symbols "${VENDOR_LIB}" \
                bnxt_re_dv_umem_reg \
                bnxt_re_dv_umem_dereg \
                bnxt_re_dv_create_cq \
                bnxt_re_dv_destroy_cq \
                bnxt_re_dv_init_obj \
                bnxt_re_dv_create_qp \
                bnxt_re_dv_destroy_qp \
                bnxt_re_dv_modify_qp
            ;;
    esac
else
    echo "vendor library : not found"
fi

if [[ "${DETECTED_NIC}" == "ionic" ]]; then
    section "Ionic CCQE prerequisites"
    ccqe_total=0
    ccqe_fw_eligible=0
    for device_path in /sys/class/infiniband/ionic*; do
        [[ -d "${device_path}" ]] || continue
        device="${device_path##*/}"
        firmware="unknown"
        [[ -r "${device_path}/fw_ver" ]] && firmware="$(<"${device_path}/fw_ver")"
        ((ccqe_total += 1))
        if ionic_ccqe_fw_supported "${firmware}"; then
            ((ccqe_fw_eligible += 1))
            echo "  ${device}: firmware=${firmware} eligible=true"
        else
            echo "  ${device}: firmware=${firmware} eligible=false"
        fi
    done

    ccqe_symbol="false"
    if [[ -n "${VENDOR_LIB}" ]] && has_symbol "${VENDOR_LIB}" ionic_dv_create_cq_ex; then
        ccqe_symbol="true"
    fi
    if [[ "${ccqe_symbol}" == "true" && "${ccqe_total}" -gt 0 && "${ccqe_fw_eligible}" -eq "${ccqe_total}" ]]; then
        CCQE_CANDIDATE="true"
    elif [[ "${ccqe_symbol}" == "true" && "${ccqe_fw_eligible}" -gt 0 ]]; then
        CCQE_CANDIDATE="mixed"
    else
        CCQE_CANDIDATE="false"
    fi
    echo "  firmware eligible : ${ccqe_fw_eligible}/${ccqe_total}"
    echo "  create_cq_ex symbol: ${ccqe_symbol}"
    echo "  host candidate     : ${CCQE_CANDIDATE}"
    echo "  effective runtime  : not tested (requires RDMA CQ creation)"
fi

run_phase "pull base image" docker pull "${BASE_IMAGE}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_ARGS=(
    docker run -d
    --name "${CONTAINER_NAME}"
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    --privileged
    --ipc=host
    --network=host
    -e PYTHONDONTWRITEBYTECODE=1
    -v "${PRIMUS_ROOT}:/src/primus:ro"
)

if [[ -n "${VENDOR_LIB}" ]]; then
    for name in "${VENDOR_NAMES[@]}"; do
        DOCKER_ARGS+=(-v "${VENDOR_LIB}:/opt/mori-host-libs/${name}:ro")
    done
fi

DOCKER_ARGS+=("${BASE_IMAGE}" sleep infinity)
run_phase "start container" "${DOCKER_ARGS[@]}"

# shellcheck disable=SC2016
run_phase "container network check" \
    docker exec "${CONTAINER_NAME}" bash -lc '
        set -e
        echo "torch=$(python3 -c "import torch; print(torch.__version__)")"
        echo "ROCM_PATH=${ROCM_PATH}"
        ip -o -4 addr show scope global || true
        ibv_devices || true
        export LD_LIBRARY_PATH=/opt/mori-host-libs:${LD_LIBRARY_PATH}
        python3 - <<'"'"'PY'"'"'
import ctypes
for name in ("libionic.so", "libbnxt_re.so", "libmlx5.so"):
    try:
        ctypes.CDLL(name)
        print(f"{name}: loadable")
    except OSError as exc:
        print(f"{name}: unavailable ({exc})")
PY
    '

run_phase "install MORI" \
    docker exec \
        -e MORI_REPO="${MORI_REPO}" \
        -e MORI_REF="${MORI_REF}" \
        -e MAX_JOBS="${MAX_JOBS}" \
        "${CONTAINER_NAME}" \
        bash /src/primus/runner/helpers/mori/install_mori.sh

# Use MORI's own runtime detector so fingerprints catch cases where identical
# host firmware produces different CCQE decisions inside the training image.
# shellcheck disable=SC2016
run_phase "MORI runtime capability check" \
    docker exec "${CONTAINER_NAME}" bash -lc '
        export LD_LIBRARY_PATH="/opt/mori-host-libs:${LD_LIBRARY_PATH}"
        python3 - <<'"'"'PY'"'"'
from mori.jit.core import detect_nic_type, is_ccqe_enabled

nic = detect_nic_type()
ccqe = str(is_ccqe_enabled()).lower() if nic == "ionic" else "n/a"
print(f"[preflight] MORI_CAPABILITY nic={nic} ccqe_runtime={ccqe}")
PY
    '

CCQE_RUNTIME="n/a"
if [[ "${DETECTED_NIC}" == "ionic" ]]; then
    CCQE_RUNTIME="$(
        awk -F'ccqe_runtime=' '/MORI_CAPABILITY/ {print $2; exit}' \
            "${LOG_DIR}/MORI_runtime_capability_check.log"
    )"
    CCQE_RUNTIME="${CCQE_RUNTIME:-unknown}"
fi

# shellcheck disable=SC2016
run_phase "8-GPU MORI all-gather smoke" \
    docker exec \
        -e PYTHONPATH=/src/primus \
        -e NCCL_SOCKET_IFNAME=lo \
        -e SMOKE_NUMEL="${SMOKE_NUMEL}" \
        "${CONTAINER_NAME}" bash -lc '
            set -e
            export LD_LIBRARY_PATH="/opt/mori-host-libs:${LD_LIBRARY_PATH}"
            torchrun --standalone --nproc_per_node=8 \
                /src/primus/runner/helpers/mori/multinode_allgather_smoke.py \
                --numel "${SMOKE_NUMEL}"
        '

echo "[preflight] NODE_FINGERPRINT nic=${DETECTED_NIC} stack_sha=${STACK_SHA} vendor_sha=${VENDOR_SHA} ccqe_candidate=${CCQE_CANDIDATE} ccqe_runtime=${CCQE_RUNTIME}"

section "Timing summary"
total=0
for entry in "${PHASE_SUMMARY[@]}"; do
    IFS="|" read -r name elapsed rc <<<"${entry}"
    printf "%-32s %8s  rc=%s\n" "${name}" "$(format_seconds "${elapsed}")" "${rc}"
    total="$((total + elapsed))"
done
printf "%-32s %8s\n" "TOTAL" "$(format_seconds "${total}")"

echo
echo "[preflight] PASS"
echo "[preflight] Logs: ${LOG_DIR}"

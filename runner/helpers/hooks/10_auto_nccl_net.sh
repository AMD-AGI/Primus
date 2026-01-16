#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: best-effort auto-select NCCL/RCCL networking settings for distributed runs.
#
# Goals:
# - Prefer IB/RDMA data-plane when available by auto-populating NCCL_IB_HCA (if unset).
# - Avoid bootstrap hangs / wrong NIC selection by auto-populating NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME (if unset).
#
# This hook emits env.* lines which will be exported by execute_hooks.sh.
# It is intentionally quiet unless it sets something.
#

set -euo pipefail

# _is_distributed() {
#     local ws="${WORLD_SIZE:-1}"
#     if [[ "$ws" =~ ^[0-9]+$ ]] && [[ "$ws" -gt 1 ]]; then
#         return 0
#     fi
#     local nn="${NNODES:-1}"
#     if [[ "$nn" =~ ^[0-9]+$ ]] && [[ "$nn" -gt 1 ]]; then
#         return 0
#     fi
#     local st="${SLURM_NTASKS:-}"
#     if [[ "$st" =~ ^[0-9]+$ ]] && [[ "$st" -gt 1 ]]; then
#         return 0
#     fi
#     return 1
# }

# # Only apply heuristics for distributed intent.
# if ! _is_distributed; then
#     exit 0
# fi

# ---------------------------------------------------------------------------
# Data-plane (IB/RDMA) selection:
# If IB devices exist and user didn't explicitly configure NCCL_IB_HCA, emit it.
# This is what actually drives "use IB" vs "socket" for NCCL/RCCL.
# ---------------------------------------------------------------------------
if [[ -z "${NCCL_IB_HCA:-}" && -d /sys/class/infiniband ]]; then
    mapfile -t ib_devs < <(
        find /sys/class/infiniband -mindepth 1 -maxdepth 1 -printf '%f\n' 2>/dev/null | sort || true
    )
    if [[ ${#ib_devs[@]} -gt 0 ]]; then
        # Prefer ACTIVE ports first; otherwise include all.
        active=()
        for d in "${ib_devs[@]}"; do
            [[ -n "$d" ]] || continue
            state_file="/sys/class/infiniband/${d}/ports/1/state"
            if [[ -f "$state_file" ]] && grep -qi "ACTIVE" "$state_file"; then
                active+=("$d")
            fi
        done
        if [[ ${#active[@]} -gt 0 ]]; then
            IFS=, echo "env.NCCL_IB_HCA=${active[*]}"
        else
            IFS=, echo "env.NCCL_IB_HCA=${ib_devs[*]}"
        fi

        # If user didn't set NCCL_IB_DISABLE, default to 0 (enabled).
        if [[ -z "${NCCL_IB_DISABLE:-}" ]]; then
            echo "env.NCCL_IB_DISABLE=0"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Bootstrap/control-plane IFNAME selection (socket):
# If user didn't set IFNAMEs and MASTER_ADDR is present, pick interface by routing.
# If MASTER_ADDR resolves to multiple IPv4s, prefer a route dev=ib* if present.
# ---------------------------------------------------------------------------
if [[ -n "${NCCL_SOCKET_IFNAME:-}" || -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
    exit 0
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
    exit 0
fi

mapfile -t master_ips < <(getent ahostsv4 "${MASTER_ADDR}" 2>/dev/null | awk '{print $1}' | uniq || true)
if [[ ${#master_ips[@]} -eq 0 ]]; then
    exit 0
fi

pick_dev=""

for ip_ in "${master_ips[@]}"; do
    [[ -n "$ip_" ]] || continue
    route_line="$(ip -o route get "$ip_" 2>/dev/null | head -1 || true)"
    [[ -n "$route_line" ]] || continue
    dev="$(
        awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}' <<<"$route_line" || true
    )"
    [[ -n "$dev" ]] || continue

    # Skip obviously bad choices.
    if [[ "$dev" == "lo" || "$dev" == "docker0" || "$dev" == veth* ]]; then
        continue
    fi
    if [[ ! -d "/sys/class/net/${dev}" ]]; then
        continue
    fi

    # Prefer IB interface if available.
    if [[ "$dev" == ib* ]]; then
        pick_dev="$dev"
        break
    fi
    if [[ -z "$pick_dev" ]]; then
        pick_dev="$dev"
    fi
done

if [[ -n "$pick_dev" ]]; then
    echo "env.NCCL_SOCKET_IFNAME=${pick_dev}"
    echo "env.GLOO_SOCKET_IFNAME=${pick_dev}"
fi

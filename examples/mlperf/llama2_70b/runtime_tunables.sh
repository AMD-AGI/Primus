#!/bin/bash
# MLPerf host runtime tunables (MI355X submission reference).
# Run on the **host** before each timed trial (see run_with_docker.sh).
# Steps are best-effort; missing tools or permissions are ignored.

set +e

echo "[runtime_tunables] Applying host performance settings..."

if [[ -w /proc/sys/vm/drop_caches ]] 2>/dev/null; then
    sync
    echo 3 >/proc/sys/vm/drop_caches 2>/dev/null
else
    sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
fi

sudo modprobe amdgpu 2>/dev/null
if command -v rocm_agent_enumerator &>/dev/null; then
    rocm_agent_enumerator >/dev/null 2>&1
fi

if command -v cpupower &>/dev/null; then
    sudo cpupower idle-set -d 2 >/dev/null 2>&1
    sudo cpupower frequency-set -g performance >/dev/null 2>&1
fi

for _sysctl in \
    /proc/sys/kernel/nmi_watchdog \
    /proc/sys/kernel/numa_balancing \
    /proc/sys/kernel/randomize_va_space
do
    if [[ -w "${_sysctl}" ]] 2>/dev/null; then
        echo 0 >"${_sysctl}" 2>/dev/null
    else
        sudo sh -c "echo 0 > ${_sysctl}" 2>/dev/null
    fi
done

for _thp in \
    /sys/kernel/mm/transparent_hugepage/enabled \
    /sys/kernel/mm/transparent_hugepage/defrag
do
    if [[ -w "${_thp}" ]] 2>/dev/null; then
        echo always >"${_thp}" 2>/dev/null
    else
        sudo sh -c "echo always > ${_thp}" 2>/dev/null
    fi
done

echo "[runtime_tunables] Done."

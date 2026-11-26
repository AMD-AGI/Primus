#!/bin/bash
# MLPerf host runtime tunables (MI355X), same steps as llama2_70b/runtime_tunables.sh.
# Run on the host before a timed trial. Missing tools or permissions are ignored.

set +e

echo "[runtime_tunables] Applying host performance settings..."

if [[ -w /proc/sys/vm/drop_caches ]] 2>/dev/null; then
    sync
    echo 3 >/proc/sys/vm/drop_caches 2>/dev/null
else
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
fi

sudo -n modprobe amdgpu 2>/dev/null
if command -v rocm_agent_enumerator &>/dev/null; then
    rocm_agent_enumerator >/dev/null 2>&1
fi

if command -v cpupower &>/dev/null; then
    sudo -n cpupower idle-set -d 2 >/dev/null 2>&1
    sudo -n cpupower frequency-set -g performance >/dev/null 2>&1
fi

for _sysctl in \
    /proc/sys/kernel/nmi_watchdog \
    /proc/sys/kernel/numa_balancing \
    /proc/sys/kernel/randomize_va_space
do
    if [[ -w "${_sysctl}" ]] 2>/dev/null; then
        echo 0 >"${_sysctl}" 2>/dev/null
    else
        sudo -n sh -c "echo 0 > ${_sysctl}" 2>/dev/null
    fi
done

for _thp in \
    /sys/kernel/mm/transparent_hugepage/enabled \
    /sys/kernel/mm/transparent_hugepage/defrag
do
    if [[ -w "${_thp}" ]] 2>/dev/null; then
        echo always >"${_thp}" 2>/dev/null
    else
        sudo -n sh -c "echo always > ${_thp}" 2>/dev/null
    fi
done

echo "[runtime_tunables] Done."
